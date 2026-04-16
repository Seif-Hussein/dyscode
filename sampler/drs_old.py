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




@register_sampler(name='drs')
def get_sampler_my(**kwargs):
    """Moreau--Yosida (MY) PnP sampler (Algorithm 2-style).

    - expects kwargs contains 'latent'
    - raises if latent True
    """
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError("Latent-space MY splitting not implemented.")
    return MY(**kwargs)


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


class DRS(nn.Module):
    """
    PDHG (Chambolle–Pock) PnP sampler.

    This version adds diagnostics analogous to ADMM:
      - sigma-normalized misalignment of denoiser input (z_k) and output (x_k) to GT
      - dual influence on denoiser input: ||tau * A^T p|| / (sigma * sqrt(d))
      - dual variable norm and step norm
      - amplitude residuals (phase retrieval)
      - prox "sanity" for exact dual prox (phase retrieval): prox_move, prox_radial_resid

    Access recorded traces after running with record=True:
      - sampler.trace via sampler.get_trace()
    """

    SUPPORTED_LINEAR_NAMES = {
        "inpainting",
        "down_sampling",
        "gaussian_blur",
        "motion_blur",
    }

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config,
                 lgvd_config, admm_config, device='cuda', **kwargs):
        super().__init__()

        self.annealing_scheduler_config, self.diffusion_scheduler_config = \
            self._check(annealing_scheduler_config, diffusion_scheduler_config)

        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.admm_config = admm_config
        self.device = device

        # ---- Diffusion parameters (only used if denoise.final_step == 'ode')
        self.betas = np.linspace(admm_config.denoise.diffusion.beta_start,
                                 admm_config.denoise.diffusion.beta_end,
                                 admm_config.denoise.diffusion.T,
                                 dtype=np.float64)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        print("No regularizers found!!!")
        self.regularizers = None

        # PDHG hyperparameters
        self.tau = self._get_tau_default()
        self.sigma_dual = self._get_sigma_dual_default()
        self.theta_schedule = self._build_theta_schedule()

        pdhg_cfg = getattr(self.admm_config, "drs", None)
        self.print_operator_norm = bool(getattr(pdhg_cfg, "print_operator_norm", False)) if pdhg_cfg is not None else False
        self.norm_power_iters = int(getattr(pdhg_cfg, "norm_power_iters", 20)) if pdhg_cfg is not None else 20

        self.proj_min = float(getattr(getattr(self.admm_config, "proj", None), "min", -1.0)
                              if getattr(self.admm_config, "proj", None) is not None else -1.0)
        self.proj_max = float(getattr(getattr(self.admm_config, "proj", None), "max",  1.0)
                              if getattr(self.admm_config, "proj", None) is not None else 1.0)

        self.use_projection = bool(getattr(getattr(self.admm_config, "proj", None), "activate", True)
                                   if getattr(self.admm_config, "proj", None) is not None else True)

        self.force_theta_zero = bool(getattr(pdhg_cfg, "force_theta_zero", False)) if pdhg_cfg is not None else False

        # Trace storage
        self.trace = None
        self.trajectory = None

    # -------------------------
    # Trace helpers
    # -------------------------
    def _init_trace(self):
        self.trace = {}

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

    # -------------------------
    # Config helpers
    # -------------------------
    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')
        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    def _get_tau_default(self) -> float:
        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        if pdhg_cfg is not None and hasattr(pdhg_cfg, "tau"):
            return float(pdhg_cfg.tau)
        return 0.1

    def _get_sigma_dual_default(self) -> float:
        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        if pdhg_cfg is not None and hasattr(pdhg_cfg, "sigma_dual"):
            return float(pdhg_cfg.sigma_dual)
        return 0.1

    def _build_theta_schedule(self):
        K = int(self.admm_config.max_iter)
        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        if pdhg_cfg is not None:
            ts = getattr(pdhg_cfg, "theta_schedule", None)
            if ts is not None and bool(getattr(ts, "activate", False)):
                th0 = float(getattr(ts, "start", 0.0))
                th1 = float(getattr(ts, "end", 1.0))
                warm = int(getattr(ts, "warmup", 50))
                out = []
                for k in range(K):
                    t = min(1.0, k / max(1, warm))
                    out.append((1 - t) * th0 + t * th1)
                return out
            if hasattr(pdhg_cfg, "theta"):
                th = float(getattr(pdhg_cfg, "theta"))
                return [th for _ in range(K)]
        return [0.0 for _ in range(K)]

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_projection:
            return x
        return torch.clamp(x, min=self.proj_min, max=self.proj_max)

    # -------------------------
    # Operator mode selection
    # -------------------------
    def _mode(self, operator, measurement) -> str:
        name = getattr(operator, "name", None)

        if name == "phase_retrieval":
            if not (hasattr(operator, "forward_complex") and hasattr(operator, "adjoint_complex")):
                raise RuntimeError("phase_retrieval operator must implement forward_complex and adjoint_complex.")
            if not torch.is_tensor(measurement):
                raise RuntimeError("phase_retrieval measurement must be a tensor amplitude.")
            return "phase_retrieval"

        if name in self.SUPPORTED_LINEAR_NAMES:
            if not torch.is_tensor(measurement):
                raise RuntimeError(f"{name} measurement must be a tensor for MSE loss.")
            return "linear_mse"

        raise NotImplementedError(
            f"PDHG supports phase_retrieval and linear-MSE operators {sorted(self.SUPPORTED_LINEAR_NAMES)}.\n"
            f"Got operator.name={name}."
        )

    # -------------------------
    # Domain transforms for phase retrieval
    # -------------------------
    @staticmethod
    def _to_01(x_m11: torch.Tensor) -> torch.Tensor:
        return x_m11 * 0.5 + 0.5

    # -------------------------
    # Autograd adjoint for linear real operators: A^T p
    # -------------------------
    def _AT_autograd(self, operator, x_like: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x_ = x_like.detach().requires_grad_(True)
        with torch.enable_grad():
            Ax = operator(x_)
            if Ax.shape != p.shape:
                raise RuntimeError(f"A(x) shape {tuple(Ax.shape)} does not match p shape {tuple(p.shape)}")
            grad = torch.autograd.grad(
                outputs=Ax,
                inputs=x_,
                grad_outputs=p,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
        return grad.detach()

    # -------------------------
    # Dual prox updates
    # -------------------------
    def _dual_update_linear_mse(self, p: torch.Tensor, Ax_bar: torch.Tensor, y: torch.Tensor,
                               sigma_dual: float, sigma_n: float):
        q = p + sigma_dual * Ax_bar
        denom = 1.0 + sigma_dual * (sigma_n ** 2)
        return (q - sigma_dual * y) / denom

    def _dual_update_phase_retrieval(self, p: torch.Tensor, u_bar: torch.Tensor, y_amp: torch.Tensor,
                                     sigma_dual: float, sigma_n: float, eps: float = 1e-12):
        q = p + sigma_dual * u_bar
        w = q / sigma_dual

        r0 = w.abs()
        a = sigma_dual * (sigma_n ** 2)
        r_star = (a * r0 + y_amp) / (a + 1.0)

        w_prox = w * (r_star / (r0 + eps))
        return q - sigma_dual * w_prox

    def _dual_update_phase_retrieval_diag(self, p: torch.Tensor, u_bar: torch.Tensor, y_amp: torch.Tensor,
                                          sigma_dual: float, sigma_n: float, eps: float = 1e-12):
        """
        Same update as _dual_update_phase_retrieval, but returns diagnostics:
          - prox_move: ||w_prox - w|| / sqrt(n)
          - prox_radial_resid: mean(| |w_prox| - r_star |)  (should be ~0)
        """
        q = p + sigma_dual * u_bar
        w = q / sigma_dual

        r0 = w.abs()
        a = sigma_dual * (sigma_n ** 2)
        r_star = (a * r0 + y_amp) / (a + 1.0)

        w_prox = w * (r_star / (r0 + eps))
        p_new = q - sigma_dual * w_prox

        # diagnostics
        w_flat = w.reshape(w.shape[0], -1) if w.dim() >= 3 else w.flatten().unsqueeze(0)
        wp_flat = w_prox.reshape(w_prox.shape[0], -1) if w_prox.dim() >= 3 else w_prox.flatten().unsqueeze(0)
        n_per = wp_flat.shape[1]

        prox_move = float((wp_flat - w_flat).abs().pow(2).sum(dim=1).sqrt().mean().detach() / math.sqrt(n_per))
        prox_radial_resid = float((w_prox.abs() - r_star).abs().mean().detach())

        return p_new, prox_move, prox_radial_resid
    
    def phase_only_ac(self,operator, z_m11, sigma_phase):
        # z_m11: [B,C,H,W] in [-1,1]
        x01 = z_m11 * 0.5 + 0.5
        x01 = x01.clamp(0.0, 1.0)

        # Forward complex FFT (matches operator.__call__ convention)
        U = operator.forward_complex(x01)  # complex tensor, padded spectrum :contentReference[oaicite:4]{index=4}

        # Phase noise: exp(i * sigma_phase * N(0,1))
        phi = torch.randn_like(U.real) * sigma_phase
        U_ac = U * torch.exp(1j * phi)

        # Back to image domain via adjoint (ifft + crop)
        H, W = z_m11.shape[-2], z_m11.shape[-1]
        x01_ac = operator.adjoint_complex(U_ac, out_hw=(H, W))  # real [B,C,H,W] :contentReference[oaicite:5]{index=5}
        x01_ac = x01_ac.clamp(0.0, 1.0)

        z_ac = x01_ac * 2.0 - 1.0
        return z_ac.clamp(-1.0, 1.0)

    # -------------------------
    # Denoiser
    # -------------------------
    def optimize_denoising(self, z_in, model, d_k, sigma, prior_use_type="denoise", wandb=False):
        denoise_config = self.admm_config.denoise
        with torch.no_grad():
            noisy_im = z_in.clone()

            if prior_use_type not in ["denoise"]:
                raise Exception(f"Prior type {prior_use_type} not supported!!!")

            ac_noise = bool(getattr(denoise_config, "ac_noise", True))
            if ac_noise and sigma > 0:
                #print("hej")
                forward_z = noisy_im + torch.randn_like(noisy_im) * sigma
            else:
                #print("hej")
                forward_z = noisy_im

            lgvd_z = forward_z.clone()
            lr = denoise_config.lgvd.lr * sigma

            num_steps = int(getattr(denoise_config.lgvd, "num_steps", 0))
            reg_factor = float(getattr(denoise_config.lgvd, "reg_factor", 0.0))
            drift_clip = float(getattr(denoise_config.lgvd, "drift_clip", 10.0))
            noise_scale = float(getattr(denoise_config.lgvd, "noise_scale", 1.0))

            for _ in range(num_steps):
                score_val = model.score(lgvd_z, sigma)
                diff_val = (forward_z - lgvd_z + d_k)
                drift = lr * min(sigma * reg_factor, drift_clip) * diff_val
                lgvd_z += lr * score_val + drift + noise_scale * (2 * lr) ** 0.5 * torch.randn_like(noisy_im)

            if denoise_config.final_step == 'tweedie':
                z = model.tweedie(lgvd_z, sigma)
            elif denoise_config.final_step == 'ode':
                diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionSampler(diffusion_scheduler)
                z = sampler.sample(model, lgvd_z, SDE=False, verbose=False)
            else:
                raise Exception(f"Final step {denoise_config.final_step} not supported!!!")

            return z

    # -------------------------
    # Norm utilities
    # -------------------------
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

    @staticmethod
    def _complex_norm_mean(p: torch.Tensor) -> float:
        """
        mean_i ||p_i|| / sqrt(n_per_sample) for complex tensor p (phase retrieval dual).
        p: complex tensor [B, ...]
        """
        with torch.no_grad():
            pf = p.reshape(p.shape[0], -1)
            n = pf.shape[1]
            # Fro norm on complex => sqrt(sum |p|^2)
            return float((pf.abs().pow(2).sum(dim=1).sqrt().mean().detach()) / math.sqrt(n))

    @staticmethod
    def _complex_step_norm_mean(p_new: torch.Tensor, p_old: torch.Tensor) -> float:
        with torch.no_grad():
            df = (p_new - p_old).reshape(p_new.shape[0], -1)
            n = df.shape[1]
            return float((df.abs().pow(2).sum(dim=1).sqrt().mean().detach()) / math.sqrt(n))

    @staticmethod
    def _amp_resid(operator, x_m11: torch.Tensor, y_amp: torch.Tensor) -> float:
        """
        || |K(x01)| - y || / sqrt(n)
        """
        with torch.no_grad():
            U = operator.forward_complex(DRS._to_01(x_m11))
            r = (U.abs() - y_amp).reshape(U.shape[0], -1)
            n = r.shape[1]
            return float(r.norm(dim=1).mean().detach() / math.sqrt(n))

    @staticmethod
    def _misalign_to_gt(x: torch.Tensor, gt: torch.Tensor, sigma: float, eps: float = 1e-12) -> float:
        """
        mean_i ||x_i - gt_i|| / (sigma * sqrt(d))
        """
        with torch.no_grad():
            df = (x - gt).detach().flatten(1)
            d = df.shape[1]
            return float(df.norm(dim=1).mean().detach() / (max(sigma, eps) * math.sqrt(d)))

    # -------------------------
    # Operator norm estimation (kept)
    # -------------------------
    def _estimate_norm_sq(self, operator, ref_img, mode: str, iters: int = 20) -> float:
        H, W = ref_img.shape[-2], ref_img.shape[-1]
        if mode == "phase_retrieval":
            x = torch.randn_like(ref_img)
            x = x / (x.norm() + 1e-12)
            for _ in range(iters):
                u = operator.forward_complex(x)
                x = operator.adjoint_complex(u, out_hw=(H, W))
                x = x / (x.norm() + 1e-12)
            u = operator.forward_complex(x)
            num = (u.abs() ** 2).sum()
            den = (x ** 2).sum()
            return (num / (den + 1e-12)).item()

        x = torch.randn_like(ref_img)
        x = x / (x.norm() + 1e-12)
        for _ in range(iters):
            Ax = operator(x)
            x = self._AT_autograd(operator, x, Ax)
            x = x / (x.norm() + 1e-12)
        Ax = operator(x)
        num = (Ax ** 2).sum()
        den = (x ** 2).sum()
        return (num / (den + 1e-12)).item()

    # -------------------------
    # Summary printing
    # -------------------------
    def _print_summary(self, start_time: float, recorded_iters: int):
        print("\n================ PDHG TRACE SUMMARY ================")
        print(f"Recorded iterations: {recorded_iters}")
        print(f"Wall time: {time.time() - start_time:.2f}s")
        if self.trace is None:
            print("No trace stored (record=False).")
            print("====================================================\n")
            return

        print("Scalar curves:")
        keys = [
            "sigma",
            "delta",
            "tau", "sigma_dual", "theta",
            "p_norm", "p_step_norm",
            "dual_inject_norm", "dual_inject_over_sigma",
            "amp_resid_z", "amp_resid_x",
            "z_misalign", "x_misalign",
            "prox_move", "prox_radial_resid",
        ]
        for k in keys:
            if k in self.trace:
                _print_curve(k, self.trace[k], head=5, tail=5)

        print("\nTensor traces (stored as CPU tensors per iter):")
        for k in ["x_k", "z_k", "y_k"]:
            if k in self.trace and len(self.trace[k]) > 0:
                t0 = self.trace[k][0]
                print(f"  {k}: {len(self.trace[k])} tensors, example shape={tuple(t0.shape)}")

        print("====================================================\n")

    # -------------------------
    # Main sampler
    # -------------------------
    def sample(self, model, ref_img, operator,
               measurement, evaluator=None,
               record=False, verbose=False, wandb=False,
               record_every: int = 1,
               trace_downsample_to: int | None = 64,
               print_summary: bool = True,
               **kwargs):

        start_time = time.time()
        eps = 1e-12

        if record:
            self.trajectory = Trajectory()
            self._init_trace()
        else:
            self.trace = None
            self.trajectory = None

        mode = self._mode(operator, measurement)

        K = int(self.admm_config.max_iter)
        pbar = tqdm.trange(K) if verbose else range(K)

        # init (same pattern as DYS)
        x_k, z_k, y_k = self.get_start(ref_img)
        x_k = y_k
        x_bar = x_k.clone()

        sigma_n = float(getattr(operator, "sigma", 0.05))

        # dual init
        if mode == "phase_retrieval":
            with torch.no_grad():
                u0 = operator.forward_complex(self._to_01(x_bar))
                p_k = torch.zeros_like(u0)  # complex
        else:
            with torch.no_grad():
                Ax0 = operator(x_bar)
                if Ax0.shape != measurement.shape:
                    raise RuntimeError(f"measurement shape {tuple(measurement.shape)} != operator(x) shape {tuple(Ax0.shape)}")
                p_k = torch.zeros_like(Ax0)

        if self.print_operator_norm:
            try:
                norm_sq = self._estimate_norm_sq(operator, ref_img, mode=mode, iters=self.norm_power_iters)
                if mode == "phase_retrieval":
                    tqdm.tqdm.write(f"[PDHG] estimated ||K||^2 ≈ {norm_sq:.6f}  (note: x->u uses 0.5*K due to x01=(x+1)/2)")
                else:
                    tqdm.tqdm.write(f"[PDHG] estimated ||A||^2 ≈ {norm_sq:.6f}")
            except Exception as e:
                tqdm.tqdm.write(f"[PDHG] ||A||^2 estimation failed: {e}")

        x_old = None
        delta_patience = 0
        delta_tol = float(getattr(self.admm_config, "delta_tol", -1.0))
        delta_pat = int(getattr(self.admm_config, "delta_patience", 0))

        recorded_iters = 0
        # ---- step-size control params (tune these) ----
        tau0 = float(self.tau)                 # baseline tau from config
        sigma_dual0 = float(self.sigma_dual)   # baseline dual step from config
        alpha_max = 5                        # cap for du/sigma (try 0.3, 0.5, 1.0)

        # Optional PDHG stability coupling: tau*sigma_dual*||A||^2 <= theta0
        use_stability_coupling = True
        theta0 = 0.9                           # target product, < 1
        normA_sq = None

        # Estimate operator norm once if you want coupling.
        # Your estimator returns ||K||^2 for phase retrieval (note: A = 0.5*K), and ||A||^2 for linear ops. 
        if use_stability_coupling:
            try:
                normK_sq = self._estimate_norm_sq(operator, ref_img, mode=mode, iters=self.norm_power_iters)
                normA_sq = 0.25 * normK_sq if mode == "phase_retrieval" else normK_sq  # A = 0.5*K in phase retrieval 
            except Exception as e:
                normA_sq = None
                print(f"[PDHG] norm estimate failed, disabling coupling: {e}")
                use_stability_coupling = False

        # Baseline sigma0 from the schedule (used by coupled schedules)
        sigma0 = float(self.annealing_scheduler.sigma_steps[0])

        for step in pbar:
            # denoiser sigma schedule
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma_d = float(self.annealing_scheduler.sigma_steps[t_sigma])
           # self.tau
            # --- continuation-style schedule (example) ---
            """eta0 = 10
            gamma_eta = 1.01  # >1
            eta_k = eta0 * (gamma_eta ** step)

            sigma_d = sigma0 / math.sqrt(eta_k)   # denoiser sigma decays ~ 1/sqrt(eta_k)
            tau_k   = tau0 / eta_k                # coupling decays faster ~ 1/eta_k
            sigma_dual_k = sigma_dual0 * eta_k    # keeps tau_k*sigma_dual_k approx constant (PDHG-style)
            self.tau = float(tau_k)
            self.sigma_dual = float(sigma_dual_k)
            print(self.tau, self.sigma_dual,'\n')"""


            theta = 0.0 if self.force_theta_zero else float(self.theta_schedule[min(step, len(self.theta_schedule) - 1)])

            # =========================
            # (1) Dual update (exact prox)
            # =========================
            p_old = p_k
            prox_move = None
            prox_radial_resid = None

            if mode == "phase_retrieval":
                u_bar = operator.forward_complex(self._to_01(x_bar))
                p_new, prox_move, prox_radial_resid = self._dual_update_phase_retrieval_diag(
                    p=p_k,
                    u_bar=u_bar,
                    y_amp=measurement,
                    sigma_dual=self.sigma_dual,
                    sigma_n=sigma_n
                )
            else:
                Ax_bar = operator(x_bar)
                p_new = self._dual_update_linear_mse(
                    p=p_k,
                    Ax_bar=Ax_bar,
                    y=measurement,
                    sigma_dual=self.sigma_dual,
                    sigma_n=sigma_n
                )

            p_k = p_new

            # dual stats
            if mode == "phase_retrieval":
                p_norm = self._complex_norm_mean(p_k)
                p_step_norm = self._complex_step_norm_mean(p_k, p_old)
            else:
                # real tensor dual
                with torch.no_grad():
                    pf = p_k.detach().flatten(1)
                    n = pf.shape[1]
                    p_norm = float(pf.norm(dim=1).mean().detach() / math.sqrt(n))
                    df = (p_k - p_old).detach().flatten(1)
                    p_step_norm = float(df.norm(dim=1).mean().detach() / math.sqrt(n))

            # =========================
            # (2) Primal step (pre-denoise) -> z_k
            # =========================
            if mode == "phase_retrieval":
                # K* p in x01 coords, then chain rule dx01/dx = 1/2
                kstar_p_x01 = operator.adjoint_complex(p_k, out_hw=(x_k.shape[-2], x_k.shape[-1]))
                ATp = 0.5 * kstar_p_x01
            else:
                ATp = self._AT_autograd(operator, x_k, p_k)

            # denoiser input
            z_k = x_k - self.tau * ATp

            # =========================
            # Diagnostics: dual injection into denoiser input
            # =========================
            # This is the PDHG analogue of ADMM's "dual_over_sigma":
            # how large is the (tau*A^T p) shift, in units of sigma_d.
            dual_inject = self.tau * ATp
            # --- adaptive tau cap to enforce du/sigma <= alpha_max ---
            ATp_norm = self._img_norm_mean(ATp)                 # ||ATp||/sqrt(d)
            tau_cap = alpha_max * sigma_d / (ATp_norm + eps)    # ensures ||tau*ATp||/sigma <= alpha_max
            tau_k = min(tau0, tau_cap)                          # never exceed baseline tau0
            print('\n--------'+str(tau_k))

            # use tau_k for primal step + diagnostics
            z_k = x_k - tau_k * ATp
            dual_inject = tau_k * ATp
            dual_inject_norm = self._img_norm_mean(dual_inject)
            dual_inject_over_sigma = dual_inject_norm / max(sigma_d, eps)

            # (optional) store current tau in self so your existing trace/logging prints it without edits
            #self.tau = float(tau_k)
            #self.tau = sigma_d

            dual_inject_norm = self._img_norm_mean(dual_inject)
            dual_inject_over_sigma = dual_inject_norm / max(sigma_d, eps)
            print('\n'+str(sigma_d))
            print('\n'+str(dual_inject_norm))

            # data-consistency residuals (phase retrieval only)
            amp_resid_z = None
            amp_resid_x = None
            if mode == "phase_retrieval":
                amp_resid_z = self._amp_resid(operator, z_k, measurement)

            # =========================
            # (3) Denoise -> x_new
            # =========================
            z_k = x_k - self.tau * ATp
            #z_k = self.phase_only_ac(operator, z_k, sigma_phase=sigma_d)

            x_new = self.optimize_denoising(
                z_in=z_k,
                model=model,
                d_k=torch.zeros_like(z_k),
                sigma=sigma_d,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb
            )
            x_new = self._proj(x_new)

            if mode == "phase_retrieval":
                amp_resid_x = self._amp_resid(operator, x_new, measurement)

            # =========================
            # (4) Extrapolation
            # =========================
            x_bar = x_new + theta * (x_new - x_k)
            x_k = x_new
            y_k = x_k

            # =========================
            # Convergence check (optional)
            # =========================
            delta = None
            if step != 0 and x_old is not None:
                denom = float(np.prod(x_k.shape))
                delta = float(((x_k - x_old).norm() ** 2 / denom).detach())
                if delta_tol > 0 and float(delta) < delta_tol:
                    delta_patience += 1
                    if delta_patience > delta_pat:
                        print(f"Converged with low delta at step {step}")
                        break
                else:
                    delta_patience = 0
            x_old = x_k.clone()

            # =========================
            # Misalignment to GT (σ-normalized) if gt provided
            # =========================
            z_misalign = None
            x_misalign = None
            if 'gt' in kwargs and kwargs['gt'] is not None:
                gt = kwargs['gt']
                z_misalign = self._misalign_to_gt(z_k, gt, sigma=sigma_d)
                x_misalign = self._misalign_to_gt(x_k, gt, sigma=sigma_d)

            # =========================
            # Recording
            # =========================
            if record and (step % max(1, int(record_every)) == 0):
                recorded_iters += 1
                # trajectory logger
                self.trajectory.add_tensor('x_k', x_k)
                self.trajectory.add_tensor('z_k', z_k)
                self.trajectory.add_tensor('y_k', y_k)
                self.trajectory.add_value('sigma', sigma_d)
                self.trajectory.add_value('tau', float(self.tau))
                self.trajectory.add_value('sigma_dual', float(self.sigma_dual))
                self.trajectory.add_value('theta', float(theta))
                if delta is not None:
                    self.trajectory.add_value('delta', delta)

                self.trajectory.add_value('p_norm', p_norm)
                self.trajectory.add_value('p_step_norm', p_step_norm)

                self.trajectory.add_value('dual_inject_norm', dual_inject_norm)
                self.trajectory.add_value('dual_inject_over_sigma', dual_inject_over_sigma)

                if amp_resid_z is not None:
                    self.trajectory.add_value('amp_resid_z', amp_resid_z)
                if amp_resid_x is not None:
                    self.trajectory.add_value('amp_resid_x', amp_resid_x)

                if z_misalign is not None:
                    self.trajectory.add_value('z_misalign', z_misalign)
                if x_misalign is not None:
                    self.trajectory.add_value('x_misalign', x_misalign)

                if prox_move is not None:
                    self.trajectory.add_value('prox_move', prox_move)
                if prox_radial_resid is not None:
                    self.trajectory.add_value('prox_radial_resid', prox_radial_resid)

                # trace dict
                self._trace_add_value("sigma", sigma_d)
                self._trace_add_value("tau", float(self.tau))
                self._trace_add_value("sigma_dual", float(self.sigma_dual))
                self._trace_add_value("theta", float(theta))
                if delta is not None:
                    self._trace_add_value("delta", delta)

                self._trace_add_value("p_norm", p_norm)
                self._trace_add_value("p_step_norm", p_step_norm)

                self._trace_add_value("dual_inject_norm", dual_inject_norm)
                self._trace_add_value("dual_inject_over_sigma", dual_inject_over_sigma)

                if amp_resid_z is not None:
                    self._trace_add_value("amp_resid_z", amp_resid_z)
                if amp_resid_x is not None:
                    self._trace_add_value("amp_resid_x", amp_resid_x)

                if z_misalign is not None:
                    self._trace_add_value("z_misalign", z_misalign)
                if x_misalign is not None:
                    self._trace_add_value("x_misalign", x_misalign)

                if prox_move is not None:
                    self._trace_add_value("prox_move", prox_move)
                if prox_radial_resid is not None:
                    self._trace_add_value("prox_radial_resid", prox_radial_resid)

                self._trace_add_tensor("x_k", x_k, downsample_to=trace_downsample_to)
                self._trace_add_tensor("z_k", z_k, downsample_to=trace_downsample_to)
                self._trace_add_tensor("y_k", y_k, downsample_to=trace_downsample_to)

            # =========================
            # Eval prints / wandb
            # =========================
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    z_k_results = evaluator(gt, measurement, z_k)
                    x_k_results = evaluator(gt, measurement, x_k)

                if verbose:
                    main = evaluator.main_eval_fn_name
                    postfix = {
                        f'z_k_{main}': f"{z_k_results[main].item():.2f}",
                        f'x_k_{main}': f"{x_k_results[main].item():.2f}",
                        "du/s": f"{dual_inject_over_sigma:.2e}",
                    }
                    if z_misalign is not None:
                        postfix["z/s"] = f"{z_misalign:.2e}"
                    if x_misalign is not None:
                        postfix["x/s"] = f"{x_misalign:.2e}"
                    pbar.set_postfix(postfix)

                if wandb:
                    logd = {
                        "PDHG Iteration": step + 1,
                        "sigma": float(sigma_d),
                        "tau": float(self.tau),
                        "sigma_dual": float(self.sigma_dual),
                        "theta": float(theta),
                        "p_norm": float(p_norm),
                        "p_step_norm": float(p_step_norm),
                        "dual_inject_over_sigma": float(dual_inject_over_sigma),
                        "wall_time": time.time() - start_time,
                    }
                    if delta is not None:
                        logd["delta"] = float(delta)
                    if amp_resid_z is not None:
                        logd["amp_resid_z"] = float(amp_resid_z)
                    if amp_resid_x is not None:
                        logd["amp_resid_x"] = float(amp_resid_x)
                    if z_misalign is not None:
                        logd["z_misalign"] = float(z_misalign)
                    if x_misalign is not None:
                        logd["x_misalign"] = float(x_misalign)
                    if prox_move is not None:
                        logd["prox_move"] = float(prox_move)
                    if prox_radial_resid is not None:
                        logd["prox_radial_resid"] = float(prox_radial_resid)
                    wnb.log(logd)

        if record and print_summary:
            self._print_summary(start_time=start_time, recorded_iters=recorded_iters)

        return x_k

    # -------------------------
    # Recording / init
    # -------------------------
    def _record(self, y_k, x_k, z_k, sigma):
        # kept for backward compat; not used in the new recording path above
        self.trajectory.add_tensor('x_k', x_k)
        self.trajectory.add_tensor('z_k', z_k)
        self.trajectory.add_tensor('y_k', y_k)
        self.trajectory.add_value('sigma', sigma)

    def get_start(self, ref):
        sigma0 = float(getattr(self.annealing_scheduler, "sigma_max", 1.0))
        init_factor = getattr(self.admm_config, "init_factor", None)

        def make_noise(scale: float):
            return (torch.randn_like(ref) * scale).to(self.device)

        vals = {}
        if init_factor is not None:
            for k in init_factor:
                if init_factor[k] is None:
                    vals[k] = make_noise(sigma0)
                else:
                    try:
                        scale = float(init_factor[k])
                    except Exception:
                        scale = float(getattr(init_factor, "x", sigma0))
                    vals[k] = make_noise(scale)
        else:
            vals["x"] = make_noise(sigma0)
            vals["z"] = make_noise(sigma0)
            vals["y"] = make_noise(sigma0)

        x0 = vals.get("x", make_noise(sigma0))
        z0 = vals.get("z", x0.clone())
        y0 = vals.get("y", vals.get("u", vals.get("z", vals.get("x", make_noise(sigma0)))))

        return x0, z0, y0


class MY(DRS):
    """Moreau--Yosida (MY) splitting sampler (Algorithm 2-style).

    This implements the lifted (x,z,y) iteration with constant parameters
    (tau, rho, lam) and uses the diffusion/denoiser as an inexactness in the
    x-step (same modelling as in the paper's Algorithm 2).

    Compared to PDHG:
      - removes the PDHG dual prox on f^*
      - introduces measurement-space prox on f (z-update)
      - maintains a lifted variable y (measurement-space)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        my_cfg = getattr(self.admm_config, "my", None)
        # Parameters: tau (primal step), rho (penalty), lam (Moreau parameter lambda)
        if my_cfg is not None and hasattr(my_cfg, "tau"):
            self.tau = float(my_cfg.tau)
        self.rho = float(getattr(my_cfg, "rho", 1.0)) if my_cfg is not None else 1.0

        # "lambda" is a reserved keyword; config may store it as lam or lambda_
        if my_cfg is not None:
            if hasattr(my_cfg, "lam"):
                self.lam = float(my_cfg.lam)
            elif hasattr(my_cfg, "lambda_"):
                self.lam = float(my_cfg.lambda_)
            elif hasattr(my_cfg, "lambda"):
                self.lam = float(getattr(my_cfg, "lambda"))
            else:
                self.lam = 1
        else:
            self.lam = 1

        self.print_operator_norm = bool(getattr(my_cfg, "print_operator_norm", False)) if my_cfg is not None else False
        self.norm_power_iters = int(getattr(my_cfg, "norm_power_iters", 20)) if my_cfg is not None else self.norm_power_iters
        self.use_stability_coupling = bool(getattr(my_cfg, "use_stability_coupling", True)) if my_cfg is not None else True
        self.theta0 = float(getattr(my_cfg, "theta0", 0.9)) if my_cfg is not None else 0.9

    # -------------------------
    # Forward / adjoint wrappers
    # -------------------------
    def _K_forward(self, operator, x_m11: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "phase_retrieval":
            return operator.forward_complex(self._to_01(x_m11))
        return operator(x_m11)

    def _K_adj(self, operator, x_like: torch.Tensor, q: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "phase_retrieval":
            # chain rule for x01 = (x+1)/2
            return 0.5 * operator.adjoint_complex(q, out_hw=(x_like.shape[-2], x_like.shape[-1]))
        return self._AT_autograd(operator, x_like, q)

    # -------------------------
    # Prox of f (measurement space)
    # -------------------------
    @staticmethod
    def _prox_f_linear_mse(v: torch.Tensor, y: torch.Tensor, rho: float, sigma_n: float) -> torch.Tensor:
        # f(u) = (1/(2*sigma_n^2))||u - y||^2
        a = rho * (sigma_n ** 2)
        return (a * v + y) / (a + 1.0)

    @staticmethod
    def _prox_f_phase_retrieval(w: torch.Tensor, y_amp: torch.Tensor, rho: float, sigma_n: float,
                               eps: float = 1e-12) -> torch.Tensor:
        # f(u) = (1/(2*sigma_n^2))|| |u| - y ||^2  (separable, radial prox)
        r0 = w.abs()
        a = rho * (sigma_n ** 2)
        r = (a * r0 + y_amp) / (a + 1.0)
        return w * (r / (r0 + eps))

    # -------------------------
    # Summary printing (override)
    # -------------------------
    def _print_summary(self, start_time: float, recorded_iters: int):
        print("\n================ MY (Moreau--Yosida) TRACE SUMMARY ================")
        print(f"Recorded iterations: {recorded_iters}")
        print(f"Wall time: {time.time() - start_time:.2f}s")
        if self.trace is None:
            print("No trace stored (record=False).")
            print("===============================================================\n")
            return

        print("Scalar curves:")
        keys = [
            "sigma",
            "tau", "rho", "lam",
            "y_norm", "y_step_norm",
            "dual_inject_norm", "dual_inject_over_sigma",
            "amp_resid_x", "amp_resid_xhat",
            "z_misalign", "x_misalign",
            "delta",
        ]
        for k in keys:
            if k in self.trace:
                _print_curve(k, self.trace[k], head=5, tail=5)

        print("\nTensor traces (stored as CPU tensors per iter):")
        for k in ["x_k", "x_hat", "y_k"]:
            if k in self.trace and len(self.trace[k]) > 0:
                t0 = self.trace[k][0]
                print(f"  {k}: {len(self.trace[k])} tensors, example shape={tuple(t0.shape)}")
        print("===============================================================\n")

    # -------------------------
    # Main sampler (Algorithm 2-style)
    # -------------------------
    def sample(self, model, ref_img, operator,
               measurement, evaluator=None,
               record=False, verbose=False, wandb=False,
               record_every: int = 1,
               trace_downsample_to: int | None = 64,
               print_summary: bool = True,
               **kwargs):

        start_time = time.time()
        eps = 1e-12

        if record:
            self.trajectory = Trajectory()
            self._init_trace()
        else:
            self.trace = None
            self.trajectory = None

        mode = self._mode(operator, measurement)
        K = int(self.admm_config.max_iter)
        pbar = tqdm.trange(K) if verbose else range(K)

        sigma_n = float(getattr(operator, "sigma", 0.05))

        # --- init x in image space ---
        x0, _, _ = self.get_start(ref_img)
        x_k = x0

        # --- init lifted variables in measurement space ---
        with torch.no_grad():
            v0 = self._K_forward(operator, x_k, mode=mode)
            z_k = v0.clone()
            y_k = torch.zeros_like(v0)

        # optional stability coupling: tau * rho * ||K||^2 <= theta0
        tau = float(self.tau)
        rho = float(self.rho)
        lam = float(self.lam)
        rho = 100
        tau = 0.05
        lam = 1/rho
        print(tau,rho,lam)
        normA_sq = None
        if self.use_stability_coupling:
            try:
                normK_sq = self._estimate_norm_sq(operator, ref_img, mode=mode, iters=self.norm_power_iters)
                normA_sq = 0.25 * normK_sq if mode == "phase_retrieval" else normK_sq
                if rho * tau * normA_sq > self.theta0:
                    tau = self.theta0 / (rho * (normA_sq + eps))
            except Exception as e:
                normA_sq = None
                if verbose:
                    print(f"[MY] norm estimate failed, disabling coupling: {e}")

        if self.print_operator_norm and normA_sq is not None:
            if mode == "phase_retrieval":
                tqdm.tqdm.write(f"[MY] estimated ||A||^2 ≈ {normA_sq:.6f} (A = 0.5*K due to x01=(x+1)/2)")
            else:
                tqdm.tqdm.write(f"[MY] estimated ||A||^2 ≈ {normA_sq:.6f}")

        recorded_iters = 0
        x_old = None
        delta_patience = 0
        delta_tol = float(getattr(self.admm_config, "delta_tol", -1.0))
        delta_pat = int(getattr(self.admm_config, "delta_patience", 0))

        for step in pbar:
            # denoiser sigma schedule (kept)
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma_d = float(self.annealing_scheduler.sigma_steps[t_sigma])

            # =========================
            # (1) x-step: gradient on Q_rho + denoiser perturbation
            # =========================
            v_k = self._K_forward(operator, x_k, mode=mode)
            r_k = v_k - z_k - lam * y_k
            y_plus = y_k + rho * r_k

            KT_yplus = self._K_adj(operator, x_k, y_plus, mode=mode)
            dual_inject = tau * KT_yplus
            x_hat = x_k - dual_inject

            # Denoise (models inexactness): x_{k+1} = D(x_hat)
            x_new = self.optimize_denoising(
                z_in=x_hat,
                model=model,
                d_k=torch.zeros_like(x_hat),
                sigma=sigma_d,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb
            )
            x_new = self._proj(x_new)

            # =========================
            # (2) z-step: prox of f in measurement space
            # =========================
            v_new = self._K_forward(operator, x_new, mode=mode)
            w = v_new + (1.0 / rho - lam) * y_k

            if mode == "phase_retrieval":
                z_new = self._prox_f_phase_retrieval(w, measurement, rho=rho, sigma_n=sigma_n, eps=eps)
            else:
                z_new = self._prox_f_linear_mse(w, measurement, rho=rho, sigma_n=sigma_n)

            # =========================
            # (3) y-step: lifted residual update
            # =========================
            y_new = (y_k + rho * (v_new - z_new)) / (1.0 + rho * lam)

            # =========================
            # Diagnostics
            # =========================
            dual_inject_norm = self._img_norm_mean(dual_inject)
            dual_inject_over_sigma = dual_inject_norm / max(sigma_d, eps)

            # y norm / step norm in measurement space
            if mode == "phase_retrieval":
                y_norm = self._complex_norm_mean(y_new)
                y_step_norm = self._complex_step_norm_mean(y_new, y_k)
            else:
                with torch.no_grad():
                    yf = y_new.detach().flatten(1)
                    n = yf.shape[1]
                    y_norm = float(yf.norm(dim=1).mean().detach() / math.sqrt(n))
                    df = (y_new - y_k).detach().flatten(1)
                    y_step_norm = float(df.norm(dim=1).mean().detach() / math.sqrt(n))

            amp_resid_x = None
            amp_resid_xhat = None
            if mode == "phase_retrieval":
                amp_resid_x = self._amp_resid(operator, x_new, measurement)
                amp_resid_xhat = self._amp_resid(operator, x_hat, measurement)

            # =========================
            # Convergence check (optional)
            # =========================
            delta = None
            if step != 0 and x_old is not None:
                denom = float(np.prod(x_new.shape))
                delta = float(((x_new - x_old).norm() ** 2 / denom).detach())
                if delta_tol > 0 and float(delta) < delta_tol:
                    delta_patience += 1
                    if delta_patience > delta_pat:
                        print(f"Converged with low delta at step {step}")
                        x_k = x_new
                        z_k = z_new
                        y_k = y_new
                        break
                else:
                    delta_patience = 0
            x_old = x_new.clone()

            # =========================
            # Misalignment to GT (σ-normalized) if gt provided
            # =========================
            z_misalign = None
            x_misalign = None
            if 'gt' in kwargs and kwargs['gt'] is not None:
                gt = kwargs['gt']
                z_misalign = self._misalign_to_gt(x_hat, gt, sigma=sigma_d)
                x_misalign = self._misalign_to_gt(x_new, gt, sigma=sigma_d)

            # =========================
            # Recording
            # =========================
            if record and (step % max(1, int(record_every)) == 0):
                recorded_iters += 1

                self.trajectory.add_tensor('x_k', x_new)
                self.trajectory.add_tensor('x_hat', x_hat)
                self.trajectory.add_tensor('y_k', y_new.real if (mode == 'phase_retrieval') else y_new)
                self.trajectory.add_value('sigma', sigma_d)
                self.trajectory.add_value('tau', float(tau))
                self.trajectory.add_value('rho', float(rho))
                self.trajectory.add_value('lam', float(lam))
                if delta is not None:
                    self.trajectory.add_value('delta', delta)

                self.trajectory.add_value('y_norm', y_norm)
                self.trajectory.add_value('y_step_norm', y_step_norm)
                self.trajectory.add_value('dual_inject_norm', dual_inject_norm)
                self.trajectory.add_value('dual_inject_over_sigma', dual_inject_over_sigma)

                if amp_resid_x is not None:
                    self.trajectory.add_value('amp_resid_x', amp_resid_x)
                if amp_resid_xhat is not None:
                    self.trajectory.add_value('amp_resid_xhat', amp_resid_xhat)

                if z_misalign is not None:
                    self.trajectory.add_value('z_misalign', z_misalign)
                if x_misalign is not None:
                    self.trajectory.add_value('x_misalign', x_misalign)

                self._trace_add_value('sigma', sigma_d)
                self._trace_add_value('tau', float(tau))
                self._trace_add_value('rho', float(rho))
                self._trace_add_value('lam', float(lam))
                if delta is not None:
                    self._trace_add_value('delta', delta)
                self._trace_add_value('y_norm', y_norm)
                self._trace_add_value('y_step_norm', y_step_norm)
                self._trace_add_value('dual_inject_norm', dual_inject_norm)
                self._trace_add_value('dual_inject_over_sigma', dual_inject_over_sigma)
                if amp_resid_x is not None:
                    self._trace_add_value('amp_resid_x', amp_resid_x)
                if amp_resid_xhat is not None:
                    self._trace_add_value('amp_resid_xhat', amp_resid_xhat)
                if z_misalign is not None:
                    self._trace_add_value('z_misalign', z_misalign)
                if x_misalign is not None:
                    self._trace_add_value('x_misalign', x_misalign)

                self._trace_add_tensor('x_k', x_new, downsample_to=trace_downsample_to)
                self._trace_add_tensor('x_hat', x_hat, downsample_to=trace_downsample_to)
                # store y_k only as real tensor for convenience (complex not supported by interpolate)
                if mode == 'phase_retrieval':
                    self._trace_add_tensor('y_k', y_new.real, downsample_to=None)
                else:
                    self._trace_add_tensor('y_k', y_new, downsample_to=None)

            # =========================
            # Eval prints / wandb
            # =========================
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x_k_results = evaluator(gt, measurement, x_new)
                if verbose:
                    main = evaluator.main_eval_fn_name
                    postfix = {
                        f'x_{main}': f"{x_k_results[main].item():.2f}",
                        "du/s": f"{dual_inject_over_sigma:.2e}",
                    }
                    pbar.set_postfix(postfix)

                if wandb:
                    logd = {
                        "MY Iteration": step + 1,
                        "sigma": float(sigma_d),
                        "tau": float(tau),
                        "rho": float(rho),
                        "lam": float(lam),
                        "y_norm": float(y_norm),
                        "y_step_norm": float(y_step_norm),
                        "dual_inject_over_sigma": float(dual_inject_over_sigma),
                        "wall_time": time.time() - start_time,
                    }
                    if delta is not None:
                        logd["delta"] = float(delta)
                    if amp_resid_x is not None:
                        logd["amp_resid_x"] = float(amp_resid_x)
                    if amp_resid_xhat is not None:
                        logd["amp_resid_xhat"] = float(amp_resid_xhat)
                    if z_misalign is not None:
                        logd["z_misalign"] = float(z_misalign)
                    if x_misalign is not None:
                        logd["x_misalign"] = float(x_misalign)
                    wnb.log(logd)

            # roll iterate
            x_k = x_new
            z_k = z_new
            y_k = y_new

        if record and print_summary:
            self._print_summary(start_time=start_time, recorded_iters=recorded_iters)

        return x_k
