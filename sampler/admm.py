import time
import math
import torch
import wandb as wnb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from utils.diffusion import Scheduler, DiffusionSampler
from utils.logging import Trajectory
from .registry import register_sampler


@register_sampler(name='admm')
def get_sampler(**kwargs):
    latent = kwargs.get('latent', False)
    if 'latent' in kwargs:
        kwargs.pop('latent')
    if latent:
        raise NotImplementedError("Latent-space ADMM not implemented.")
    return ADMM(**kwargs)


# ============================================================
# Small OT helpers (optional comparison printing)
# ============================================================
@torch.no_grad()
def coupled_w2_upper_sq(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Coupled upper bound on W2^2 using index-wise coupling: mean_i ||A_i - B_i||^2
    A,B: [N,C,H,W] or [C,H,W]
    """
    if A.dim() == 3:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(0)
    diff = (A - B).flatten(1)
    return float(diff.pow(2).sum(dim=1).mean().detach().cpu())


@torch.no_grad()
def sliced_w2_sq(A: torch.Tensor,
                 B: torch.Tensor,
                 num_proj: int = 64,
                 downsample_to: int | None = None,
                 seed: int = 0,
                 eps: float = 1e-12) -> float:
    """
    Empirical Sliced Wasserstein-2^2 (SW2^2) between two batches of images.
    """
    if A.dim() == 3:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(0)

    if downsample_to is not None:
        A = F.interpolate(A, size=(downsample_to, downsample_to), mode="area")
        B = F.interpolate(B, size=(downsample_to, downsample_to), mode="area")

    A = A.flatten(1)
    B = B.flatten(1)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A {tuple(A.shape)} vs B {tuple(B.shape)}")

    N, d = A.shape
    g = torch.Generator(device=A.device)
    g.manual_seed(seed)

    theta = torch.randn(num_proj, d, generator=g, device=A.device)
    theta = theta / (theta.norm(dim=1, keepdim=True) + eps)

    proj_A = A @ theta.T
    proj_B = B @ theta.T

    proj_A, _ = proj_A.sort(dim=0)
    proj_B, _ = proj_B.sort(dim=0)

    return float((proj_A - proj_B).pow(2).mean().detach().cpu())


def _summarize_list(x: list[float]) -> dict:
    if len(x) == 0:
        return {"len": 0}
    arr = np.asarray(x, dtype=float)
    return {
        "len": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "final": float(arr[-1]),
    }


def _print_curve(name: str, curve: list[float], head: int = 5, tail: int = 5):
    if curve is None or len(curve) == 0:
        print(f"  {name}: <empty>")
        return
    s = _summarize_list(curve)
    print(f"  {name}: len={s['len']} mean={s['mean']:.3e} med={s['median']:.3e} "
          f"min={s['min']:.3e} max={s['max']:.3e} final={s['final']:.3e}")
    h = curve[:min(head, len(curve))]
    t = curve[max(0, len(curve)-tail):]
    print(f"    head: {['%.3e' % v for v in h]}")
    if len(curve) > head:
        print(f"    tail: {['%.3e' % v for v in t]}")


class ADMM(nn.Module):
    """
    ADMM-based PnP sampler with diffusion/score denoiser.

    Key tracked quantities:
      - x_opt_resid: ||∇ℓ(x) + ρ(x - z_old + u_old)|| / sqrt(d)
      - g_data_norm: ||∇ℓ(x)|| / sqrt(d)
      - g_pen_norm : ||ρ(x - z_old + u_old)|| / sqrt(d)
      - cos_gp     : cos(∇ℓ(x), ρ(x - z_old + u_old))

    Computational-error residual (ADMM x-subproblem in primal units):
      r_x^{k+1} = (1/ρ)∇ℓ(x^{k+1}) + x^{k+1} - z^k + u^k
      rx_norm   = ||r_x||/sqrt(d) = x_opt_resid / ρ
      rx_over_sigma = rx_norm / σ_k
      Rx_over_sigma2_per_dim = (||r_x||^2/d)/σ_k^2

    Dual mismatch (σ-normalized):
      u_in_norm      = mean_i ||u_i|| / sqrt(d_per_sample)    (before dual update)
      dual_over_sigma = u_in_norm / σ_k

    Total “AC center mismatch” proxy (requires GT):
      ze_misalign = mean_i ||(x+u)_i - gt_i|| / (σ_k * sqrt(d_per_sample))
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config,
                 lgvd_config, admm_config, device='cuda', **kwargs):
        super().__init__()
        self.annealing_scheduler_config, self.diffusion_scheduler_config = \
            self._check(annealing_scheduler_config, diffusion_scheduler_config)

        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.admm_config = admm_config
        self.device = device

        # diffusion params (only used if final_step == 'ode')
        self.betas = np.linspace(admm_config.denoise.diffusion.beta_start,
                                 admm_config.denoise.diffusion.beta_end,
                                 admm_config.denoise.diffusion.T,
                                 dtype=np.float64)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        print("No regularizers found!!!")
        self.regularizers = None

        self.trace = None
        self.trajectory = None
        self.metric_history = None

    # -------------------------
    # Trace helpers
    # -------------------------
    def _init_trace(self):
        self.trace = {}

    def _init_metric_history(self):
        self.metric_history = {}

    @staticmethod
    def _img_norm_mean(x: torch.Tensor) -> float:
        """
        mean_i ||x_i|| / sqrt(d_per_sample)
        x: [B,C,H,W]
        """
        with torch.no_grad():
            xf = x.detach().flatten(1)
            d = xf.shape[1]
            return float(xf.norm(dim=1).mean().detach() / math.sqrt(d))
    
    def _trace_add_value(self, key: str, value):
        if self.trace is None:
            return
        self.trace.setdefault(key, []).append(float(value))

    def _trace_add_tensor(self, key: str, tensor: torch.Tensor, downsample_to: int | None):
        if self.trace is None:
            return
        t = tensor.detach()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if downsample_to is not None:
            t = F.interpolate(t, size=(downsample_to, downsample_to), mode='area')
        self.trace.setdefault(key, []).append(t.cpu())

    def get_trace(self):
        return self.trace

    def _metric_history_add(self, key: str, value):
        if self.metric_history is None:
            return
        self.metric_history.setdefault(key, []).append(float(value))

    def get_metric_history(self):
        return self.metric_history

    # -------------------------
    # ML / data subproblem
    # -------------------------
    def optimize_ml_with_generic_gd(self, x_k, z_k, u_k, operator, measurement, wandb=False):
        ml_config = self.admm_config.ml
        progress_bar = tqdm.trange(ml_config.max_iter) if ml_config.verbose else range(ml_config.max_iter)
        last_loss = np.inf

        x_k.requires_grad = True
        lr = ml_config.lr
        if ml_config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam([x_k], lr=lr)
        elif ml_config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD([x_k], lr=lr)
        else:
            raise ValueError(f"Optimizer {ml_config.optimizer} not supported")

        conv_count = 0
        for iteration in progress_bar:
            optimizer.zero_grad()
            lk_loss = operator.loss(x_k, measurement)
            reg_loss = self.admm_config.rho / 2 * ((x_k - z_k + u_k) ** 2).sum()
            loss_val = reg_loss + lk_loss

            if (((('reg_use_freq' in ml_config) and (iteration % ml_config.reg_use_freq == 0))
                 or 'reg_use_freq' not in ml_config) and self.regularizers):
                extra_reg_loss = sum([reg(x_k) for reg in self.regularizers])
                loss_val += extra_reg_loss
            else:
                extra_reg_loss = torch.tensor(0).to(self.device)

            loss_val.backward()
            optimizer.step()

            # identity for your phase retrieval operator, but keep for compatibility
            x_k = operator.post_ml_op(x_k, measurement)

            if ml_config.clip:
                with torch.no_grad():
                    x_k.clamp_(-1.0, 1.0)

            delta_loss = abs(last_loss - loss_val.item())
            if ml_config.verbose:
                progress_bar.set_description(
                    f"Lr: {lr:.6f} Rho: {self.admm_config.rho: .6f} ML Loss: {loss_val.item():.2f} "
                    f"Lk Loss: {lk_loss.item():.2f} Reg Loss: {reg_loss.item():.2f} "
                    f"Extra: {extra_reg_loss.item():.2f} dL: {delta_loss:.2f}"
                )

            if last_loss < loss_val.item():
                lr /= ml_config.lr_decay
                if lr < ml_config.lr_min:
                    break
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif delta_loss < ml_config.tol:
                conv_count += 1
                if conv_count > ml_config.patience:
                    break
            else:
                conv_count = 0

            last_loss = loss_val.item()

        return x_k.detach()

    def optimize_ml(self, x_k, z_k, u_k, operator, measurement, wandb=False):
        if ("use_task_specific_solver" in self.admm_config.ml
                and self.admm_config.ml.use_task_specific_solver.activate):
            print("Using task specific solver")
            return operator.ml_solver(
                x_k=x_k, z_k=z_k, u_k=u_k, rho=self.admm_config.rho,
                measurement=measurement,
                solver_config=self.admm_config.ml.use_task_specific_solver,
                wandb=wandb)
        return self.optimize_ml_with_generic_gd(
            x_k=x_k, z_k=z_k, u_k=u_k, operator=operator, measurement=measurement, wandb=wandb
        )

    # -------------------------
    # Denoiser
    # -------------------------
    def optimize_denoising(self, x_k, u_k, model, sigma,
                           prior_use_type="denoise",
                           wandb=False,
                           return_internals: bool = False):
        denoise_config = self.admm_config.denoise
        with torch.no_grad():
            z_e = (x_k + u_k).clone()

            if prior_use_type not in ["denoise"]:
                raise Exception(f"Prior type {prior_use_type} not supported!!!")

            z_ac = z_e + torch.randn_like(z_e) * sigma
            z_dc = z_ac.clone()

            lr = denoise_config.lgvd.lr * sigma
            for _ in range(denoise_config.lgvd.num_steps):
                score_val = model.score(z_dc, sigma)
                diff_val = (z_ac - z_dc)
                z_dc += lr * score_val + \
                        lr * min(sigma * denoise_config.lgvd.reg_factor, 10) * diff_val + \
                        (2 * lr) ** 0.5 * torch.randn_like(z_e)

            if denoise_config.final_step == 'tweedie':
                z = model.tweedie(z_dc, sigma)
            elif denoise_config.final_step == 'ode':
                diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionSampler(diffusion_scheduler)
                z = sampler.sample(model, z_dc, SDE=False, verbose=False)
            else:
                raise Exception(f"Final step {denoise_config.final_step} not supported!!!")

        if return_internals:
            return z, {"z_e": z_e, "z_ac": z_ac, "z_dc": z_dc}
        return z

    # -------------------------
    # x-subproblem diagnostic (WITH COMPONENTS)
    # -------------------------
    def _x_opt_stationarity_residual(self, x_new, z_old, u_old, operator, measurement, rho: float):
        """
        Returns:
          r_total    = ||∇ℓ(x) + ρ(x - z_old + u_old)|| / sqrt(d)
          lk_loss    = ℓ(x)
          reg_loss   = 0.5*ρ||x - z_old + u_old||^2
          g_data_n   = ||∇ℓ(x)|| / sqrt(d)
          g_pen_n    = ||ρ(x - z_old + u_old)|| / sqrt(d)
          cos_gp     = cosine(∇ℓ(x), ρ(x - z_old + u_old))
        """
        x = x_new.detach().requires_grad_(True)
        with torch.enable_grad():
            lk_loss = operator.loss(x, measurement)
            g_data = torch.autograd.grad(lk_loss, x, retain_graph=True, create_graph=False)[0]

            diff = (x - z_old + u_old)
            reg_loss = 0.5 * rho * (diff ** 2).sum()
            g_pen = rho * diff

            g_total = g_data + g_pen

            denom = math.sqrt(x.numel())
            r_total = (g_total.norm() / denom).detach()
            g_data_n = (g_data.norm() / denom).detach()
            g_pen_n = (g_pen.norm() / denom).detach()

            num = (g_data.flatten() * g_pen.flatten()).sum()
            den = (g_data.norm() * g_pen.norm()).clamp_min(1e-12)
            cos_gp = (num / den).detach()

        return (
            float(r_total),
            float(lk_loss.detach()),
            float(reg_loss.detach()),
            float(g_data_n),
            float(g_pen_n),
            float(cos_gp),
        )

    # -------------------------
    # Printing / comparison
    # -------------------------
    def _print_summary(self, start_time: float, recorded_iters: int):
        print("\n================ ADMM TRACE SUMMARY ================")
        print(f"Recorded iterations: {recorded_iters}")
        print(f"Wall time: {time.time() - start_time:.2f}s")
        if self.trace is None:
            print("No trace stored (record=False).")
            print("====================================================\n")
            return

        print("Scalar curves:")
        keys = [
            "x_opt_resid", "lk_loss", "reg_loss",
            "g_data_norm", "g_pen_norm", "cos_gp",
            "rx_norm", "Rx_sq", "Rx_sq_per_dim", "rx_over_sigma", "Rx_over_sigma2_per_dim",
            "u_in_norm", "dual_over_sigma", "ze_misalign",
            "delta_t", "sigma"
        ]
        for k in keys:
            if k in self.trace:
                _print_curve(k, self.trace[k], head=5, tail=5)

        print("\nTensor traces (stored as CPU tensors per iter):")
        for k in ["x_k", "z_e", "z_k", "u_k", "z_ac", "z_dc"]:
            if k in self.trace and len(self.trace[k]) > 0:
                t0 = self.trace[k][0]
                print(f"  {k}: {len(self.trace[k])} tensors, example shape={tuple(t0.shape)}")

        print("====================================================\n")

    def _print_comparison(self,
                          compare_trace: dict,
                          compare_keys: tuple[str, ...],
                          compare_method: str,
                          compare_num_proj: int,
                          compare_downsample_to: int | None,
                          compare_map: dict | None,
                          curve_head: int,
                          curve_tail: int,
                          device: str):
        if self.trace is None or compare_trace is None:
            return

        print("\n============ ADMM vs COMPARE TRACE DISTANCES ==========")
        print(f"Method: {compare_method}  (swd=num_proj={compare_num_proj})  device={device}")

        for key in compare_keys:
            key_B = compare_map.get(key, key) if compare_map is not None else key

            if key not in self.trace:
                print(f"  skip {key}: not found in ADMM trace")
                continue
            if key_B not in compare_trace:
                print(f"  skip {key}: not found in compare trace as '{key_B}'")
                continue

            seqA = self.trace[key]
            seqB = compare_trace[key_B]
            T = min(len(seqA), len(seqB))
            curve = []
            for i in range(T):
                A = seqA[i].to(device)
                B = seqB[i].to(device)
                if compare_method.lower() in ("coupled", "upper", "w2_upper"):
                    curve.append(coupled_w2_upper_sq(A, B))
                elif compare_method.lower() in ("swd", "sw2", "sliced"):
                    curve.append(sliced_w2_sq(A, B, num_proj=compare_num_proj, downsample_to=compare_downsample_to))
                else:
                    raise ValueError(f"Unknown compare_method: {compare_method}")

            _print_curve(f"dist[{key}]", curve, head=curve_head, tail=curve_tail)

        print("=======================================================\n")

    # -------------------------
    # Main sampling loop
    # -------------------------
    def sample(self, model, ref_img, operator, measurement,
               evaluator=None,
               record: bool = False,
               verbose: bool = False,
               wandb: bool = False,

               record_every: int = 1,
               trace_downsample_to: int | None = 64,
               trace_internals: bool = False,

               print_summary: bool = True,
               print_curve_head: int = 5,
               print_curve_tail: int = 5,

               save_trace_path: str | None = None,

               compare_trace: dict | None = None,
               compare_keys: tuple[str, ...] = ("z_e", "z_k"),
               compare_method: str = "swd",
               compare_num_proj: int = 64,
               compare_downsample_to: int | None = None,
               compare_map: dict | None = None,

               compare_device: str = "cpu",

               **kwargs):

        start_time = time.time()
        eps = 1e-12
        self._init_metric_history()

        if record:
            self.trajectory = Trajectory()
            self._init_trace()
        else:
            self.trace = None
            self.trajectory = None

        pbar = tqdm.trange(self.admm_config.max_iter) if verbose else range(self.admm_config.max_iter)

        x_k, z_k, u_k = self.get_start(ref_img)

        x_k_old, z_k_old, u_k_old = None, None, None

        eta, gamma = self.admm_config.eta, self.admm_config.gamma
        delta_t_old = torch.inf
        delta_patience = 0

        recorded_iters = 0

        for step in pbar:
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma = float(self.annealing_scheduler.sigma_steps[t_sigma])

            # (1) ML/data subproblem (approx)
            z_old = z_k
            x_k = self.optimize_ml(x_k=x_k, z_k=z_old, u_k=u_k,
                                   operator=operator, measurement=measurement, wandb=wandb)

            rho_now = float(self.admm_config.rho)

            r_xopt, lk_loss, reg_loss, g_data_n, g_pen_n, cos_gp = self._x_opt_stationarity_residual(
                x_new=x_k, z_old=z_old, u_old=u_k,
                operator=operator, measurement=measurement, rho=rho_now
            )

            # r_x tracking
            rx_norm = r_xopt / max(rho_now, eps)                  # ||r_x||/sqrt(d)
            Rx_sq_per_dim = rx_norm ** 2                          # ||r_x||^2/d
            d_total = float(x_k.numel())
            Rx_sq = Rx_sq_per_dim * d_total
            rx_over_sigma = rx_norm / max(sigma, eps)
            Rx_over_sigma2_per_dim = Rx_sq_per_dim / max(sigma * sigma, eps)

            # denoiser input (BEFORE dual update)
            z_e = x_k + u_k

            # dual mismatch (σ-normalized)
            with torch.no_grad():
                u_flat = u_k.detach().flatten(1)
                d_per = u_flat.shape[1]
                u_in_norm = float(u_flat.norm(dim=1).mean().detach() / math.sqrt(d_per))
            dual_over_sigma = u_in_norm / max(sigma, eps)

            # total misalignment proxy using GT (σ-normalized), if available
            ze_misalign = None
            if 'gt' in kwargs and kwargs['gt'] is not None:
                with torch.no_grad():
                    gt = kwargs['gt']
                    diff = (z_e - gt).detach().flatten(1)
                    d_per2 = diff.shape[1]
                    ze_misalign = float(diff.norm(dim=1).mean().detach() / (max(sigma, eps) * math.sqrt(d_per2)))

            # (2) denoiser / prior
            if trace_internals and record:
                z_k, internals = self.optimize_denoising(
                    x_k=x_k, u_k=u_k, model=model, sigma=sigma,
                    prior_use_type=self.admm_config.denoise.type,
                    wandb=wandb, return_internals=True
                )
            else:
                z_k = self.optimize_denoising(
                    x_k=x_k, u_k=u_k, model=model, sigma=sigma,
                    prior_use_type=self.admm_config.denoise.type,
                    wandb=wandb, return_internals=False
                )
                internals = None

            # (3) dual update
            u_k = u_k + x_k - z_k
            print(self._img_norm_mean(u_k))

            # delta check
            delta_t = None
            if step != 0:
                denom = float(x_k.numel())
                delta_1 = (x_k - x_k_old).norm() ** 2 / denom
                delta_2 = (z_k - z_k_old).norm() ** 2 / denom
                delta_3 = (u_k - u_k_old).norm() ** 2 / denom
                delta_t = float((delta_1 + delta_2 + delta_3).detach())

                if delta_t < float(self.admm_config.delta_tol):
                    delta_patience += 1
                    if delta_patience > int(self.admm_config.delta_patience):
                        if verbose:
                            print(f"Converged with low delta at step {step}")
                        break
                else:
                    delta_patience = 0

                if (delta_t > eta * float(delta_t_old)) and (step > 0.8 * self.annealing_scheduler.num_steps):
                    self.admm_config.rho *= gamma
                    self.admm_config.rho = min(self.admm_config.rho, 500)
                    u_k /= gamma

                delta_t_old = delta_t

                if wandb:
                    logd = {
                        "ADMM Iteration": step + 1,
                        "delta_t": delta_t,
                        "rho": float(self.admm_config.rho),
                        "sigma": float(sigma),
                        "x_opt_resid": float(r_xopt),
                        "g_data_norm": float(g_data_n),
                        "g_pen_norm": float(g_pen_n),
                        "cos_gp": float(cos_gp),
                        "rx_norm": float(rx_norm),
                        "Rx_sq_per_dim": float(Rx_sq_per_dim),
                        "rx_over_sigma": float(rx_over_sigma),
                        "Rx_over_sigma2_per_dim": float(Rx_over_sigma2_per_dim),
                        "u_in_norm": float(u_in_norm),
                        "dual_over_sigma": float(dual_over_sigma),
                        "lk_loss": float(lk_loss),
                        "reg_loss": float(reg_loss),
                        "wall_time": time.time() - start_time,
                    }
                    if ze_misalign is not None:
                        logd["ze_misalign"] = float(ze_misalign)
                    wnb.log(logd)

            x_k_old, z_k_old, u_k_old = x_k.clone(), z_k.clone(), u_k.clone()

            # evaluation (optional)
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x_k_results = evaluator(gt, measurement, x_k)
                    z_k_results = evaluator(gt, measurement, z_k)

                self._metric_history_add("step", step + 1)
                self._metric_history_add("sigma", sigma)
                self._metric_history_add("rho", float(self.admm_config.rho))
                for metric_name, metric_value in x_k_results.items():
                    self._metric_history_add(f"x_k_{metric_name}", metric_value.item())
                for metric_name, metric_value in z_k_results.items():
                    self._metric_history_add(f"z_k_{metric_name}", metric_value.item())

                if verbose:
                    main = evaluator.main_eval_fn_name
                    postfix = {
                        f'x_{main}': f"{x_k_results[main].item():.2f}",
                        f'z_{main}': f"{z_k_results[main].item():.2f}",
                        'xopt': f"{r_xopt:.2e}",
                        'rx/s': f"{rx_over_sigma:.2e}",
                        'du/s': f"{dual_over_sigma:.2e}",
                    }
                    if ze_misalign is not None:
                        postfix['ze/s'] = f"{ze_misalign:.2e}"
                    pbar.set_postfix(postfix)

            # recording
            if record and (step % max(1, int(record_every)) == 0):
                recorded_iters += 1

                # Trajectory logger
                self.trajectory.add_tensor('x_k', x_k)
                self.trajectory.add_tensor('z_k', z_k)
                self.trajectory.add_tensor('u_k', u_k)
                self.trajectory.add_tensor('z_e', z_e)

                self.trajectory.add_value('sigma', sigma)
                self.trajectory.add_value('x_opt_resid', r_xopt)
                self.trajectory.add_value('lk_loss', lk_loss)
                self.trajectory.add_value('reg_loss', reg_loss)
                self.trajectory.add_value('g_data_norm', g_data_n)
                self.trajectory.add_value('g_pen_norm', g_pen_n)
                self.trajectory.add_value('cos_gp', cos_gp)

                self.trajectory.add_value('rx_norm', rx_norm)
                self.trajectory.add_value('Rx_sq', Rx_sq)
                self.trajectory.add_value('Rx_sq_per_dim', Rx_sq_per_dim)
                self.trajectory.add_value('rx_over_sigma', rx_over_sigma)
                self.trajectory.add_value('Rx_over_sigma2_per_dim', Rx_over_sigma2_per_dim)

                self.trajectory.add_value('u_in_norm', u_in_norm)
                self.trajectory.add_value('dual_over_sigma', dual_over_sigma)
                if ze_misalign is not None:
                    self.trajectory.add_value('ze_misalign', ze_misalign)

                if delta_t is not None:
                    self.trajectory.add_value('delta_t', delta_t)

                if internals is not None:
                    self.trajectory.add_tensor('z_ac', internals["z_ac"])
                    self.trajectory.add_tensor('z_dc', internals["z_dc"])

                # Plain trace dict
                self._trace_add_value("sigma", sigma)
                self._trace_add_value("x_opt_resid", r_xopt)
                self._trace_add_value("lk_loss", lk_loss)
                self._trace_add_value("reg_loss", reg_loss)
                self._trace_add_value("g_data_norm", g_data_n)
                self._trace_add_value("g_pen_norm", g_pen_n)
                self._trace_add_value("cos_gp", cos_gp)

                self._trace_add_value("rx_norm", rx_norm)
                self._trace_add_value("Rx_sq", Rx_sq)
                self._trace_add_value("Rx_sq_per_dim", Rx_sq_per_dim)
                self._trace_add_value("rx_over_sigma", rx_over_sigma)
                self._trace_add_value("Rx_over_sigma2_per_dim", Rx_over_sigma2_per_dim)

                self._trace_add_value("u_in_norm", u_in_norm)
                self._trace_add_value("dual_over_sigma", dual_over_sigma)
                if ze_misalign is not None:
                    self._trace_add_value("ze_misalign", ze_misalign)

                if delta_t is not None:
                    self._trace_add_value("delta_t", delta_t)

                self._trace_add_tensor("x_k", x_k, downsample_to=trace_downsample_to)
                self._trace_add_tensor("z_e", z_e, downsample_to=trace_downsample_to)
                self._trace_add_tensor("z_k", z_k, downsample_to=trace_downsample_to)
                self._trace_add_tensor("u_k", u_k, downsample_to=trace_downsample_to)
                if internals is not None:
                    self._trace_add_tensor("z_ac", internals["z_ac"], downsample_to=trace_downsample_to)
                    self._trace_add_tensor("z_dc", internals["z_dc"], downsample_to=trace_downsample_to)

        # auto-save
        if record and save_trace_path is not None:
            try:
                torch.save(self.trace, save_trace_path)
                print(f"[ADMM] Saved trace to: {save_trace_path}")
            except Exception as e:
                print(f"[ADMM] Failed to save trace: {e}")

        # print summary + optional comparisons
        if record and print_summary:
            self._print_summary(start_time=start_time, recorded_iters=recorded_iters)
            if compare_trace is not None:
                self._print_comparison(
                    compare_trace=compare_trace,
                    compare_keys=compare_keys,
                    compare_method=compare_method,
                    compare_num_proj=compare_num_proj,
                    compare_downsample_to=compare_downsample_to,
                    compare_map=compare_map,
                    curve_head=print_curve_head,
                    curve_tail=print_curve_tail,
                    device=compare_device
                )

        return z_k

    # -------------------------
    # init helpers
    # -------------------------
    def get_start(self, ref):
        init_values = []
        init_factor = getattr(self.admm_config, "init_factor", None)

        if init_factor is None:
            x0 = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
            z0 = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
            u0 = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
            return [x0.to(self.device), z0.to(self.device), u0.to(self.device)]

        for factor_key in init_factor:
            if init_factor[factor_key] is None:
                start_val = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
            else:
                try:
                    start_val = torch.randn_like(ref) * float(init_factor[factor_key])
                except Exception:
                    start_val = torch.randn_like(ref) * float(getattr(init_factor, "x", self.annealing_scheduler.sigma_max))
            init_values.append(start_val.to(self.device))

        return init_values

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')
        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config
