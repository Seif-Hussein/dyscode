import time
from xml.parsers.expat import model
import torch
import wandb as wnb
import numpy as np
import torch.nn as nn
import tqdm
import math
import sampler
from utils.diffusion import Scheduler, DiffusionSampler
from utils.logging import Trajectory
from .registry import register_sampler


@register_sampler(name='pdhg')
def get_sampler(**kwargs):
    """
    PDHG sampler.
    - expects kwargs contains 'latent'
    - raises if latent True
    """
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError("Latent-space PDHG not implemented.")
    return PDHG(**kwargs)


class PDHG(nn.Module):
    """
    PDHG (Chambolle–Pock) PnP sampler.

    Supports:
      - phase_retrieval: special complex PDHG with magnitude data term using
        operator.forward_complex / operator.adjoint_complex.

      - linear MSE operators (Gaussian noise) where operator.loss corresponds to
        1/(2*sigma_n^2) ||A(x) - y||^2, and A is linear:
          - inpainting
          - down_sampling
          - gaussian_blur
          - motion_blur

    For linear operators, A^T is computed via autograd VJP:
        A^T p = d/dx <A(x), p>
    which is exact for linear differentiable A.
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

        # Optional operator norm printing
        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        self.print_operator_norm = bool(getattr(pdhg_cfg, "print_operator_norm", True)) if pdhg_cfg is not None else False
        self.norm_power_iters = int(getattr(pdhg_cfg, "norm_power_iters", 20)) if pdhg_cfg is not None else 20

        # Projection settings (match ADMM/DYS clamp behavior)
        self.proj_min = float(getattr(getattr(self.admm_config, "proj", None), "min", -1.0)
                              if getattr(self.admm_config, "proj", None) is not None else -1.0)
        self.proj_max = float(getattr(getattr(self.admm_config, "proj", None), "max",  1.0)
                              if getattr(self.admm_config, "proj", None) is not None else 1.0)

        self.use_projection = bool(getattr(getattr(self.admm_config, "proj", None), "activate", True)
                                   if getattr(self.admm_config, "proj", None) is not None else True)

        # Optional: force theta = 0 for stability debugging
        self.force_theta_zero = bool(getattr(pdhg_cfg, "force_theta_zero", False)) if pdhg_cfg is not None else False

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
        # safe default
        return 0.1

    def _get_sigma_dual_default(self) -> float:
        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        if pdhg_cfg is not None and hasattr(pdhg_cfg, "sigma_dual"):
            return float(pdhg_cfg.sigma_dual)
        # safe default
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
        return [0.0 for _ in range(K)]  # safest

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

        # linear MSE operators
        if name in self.SUPPORTED_LINEAR_NAMES:
            if not torch.is_tensor(measurement):
                raise RuntimeError(f"{name} measurement must be a tensor for MSE loss.")
            return "linear_mse"

        # explicitly reject the others in your repo
        raise NotImplementedError(
            f"PDHG sampler supports phase_retrieval and linear-MSE operators {sorted(self.SUPPORTED_LINEAR_NAMES)}.\n"
            f"Got operator.name={name}. Nonlinear/quantized models are intentionally not supported here."
        )

    # -------------------------
    # Domain transforms for phase retrieval
    # -------------------------
    @staticmethod
    def _to_01(x_m11: torch.Tensor) -> torch.Tensor:
        # Must match PhaseRetrieval.__call__: x01 = 0.5*x + 0.5 (no clamp here)
        return x_m11 * 0.5 + 0.5

    # -------------------------
    # Autograd adjoint for linear real operators: A^T p
    # -------------------------
    def _AT_autograd(self, operator, x_like: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute A^T p via VJP: grad_x <A(x), p>.
        This is exact if A is linear and differentiable (and still a valid VJP in general).
        """
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
    def _dual_update_linear_mse(self, p: torch.Tensor, Ax_bar: torch.Tensor, y: torch.Tensor, sigma_dual: float, sigma_n: float):
        """
        f(v) = 1/(2*sigma_n^2) ||v - y||^2
        prox_{sigma f*}(q) = (q - sigma*y)/(1 + sigma*sigma_n^2)
        with q = p + sigma*Ax_bar
        """
        q = p + sigma_dual * Ax_bar
        denom = 1.0 + sigma_dual * (sigma_n ** 2)
        return (q - sigma_dual * y) / denom

    def _dual_update_phase_retrieval(self, p: torch.Tensor, u_bar: torch.Tensor, y_amp: torch.Tensor,
                                    sigma_dual: float, sigma_n: float, eps: float = 1e-12):
        """
        Phase retrieval lift:
          u = K(x01) (complex), y = |u| + noise

        f(u) = 1/(2*sigma_n^2) || |u| - y ||^2
        Dual update: p^{+} = prox_{sigma f*}( p + sigma u_bar )
        Implemented via Moreau:
          prox_{sigma f*}(q) = q - sigma * prox_{(1/sigma) f}(q/sigma)
        and prox_{(1/sigma) f} is radial shrink in magnitude.
        """
        q = p + sigma_dual * u_bar
        w = q / sigma_dual  # w = u_bar + p/sigma

        r0 = w.abs()
        a = sigma_dual * (sigma_n ** 2)  # a = sigma*sigma_n^2
        r_star = (a * r0 + y_amp) / (a + 1.0)

        w_prox = w * (r_star / (r0 + eps))
        return q - sigma_dual * w_prox
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

    # -------------------------
    # Denoiser (copied/adapted from your DYS/ADMM)
    # -------------------------
    def optimize_denoising(self, z_in,
                           model, d_k, sigma,
                           prior_use_type="denoise",
                           wandb=False):
        denoise_config = self.admm_config.denoise
        with torch.no_grad():
            noisy_im = z_in.clone()

            if prior_use_type in ["denoise"]:
                ac_noise = bool(getattr(denoise_config, "ac_noise", True))
                if ac_noise and sigma > 0:
                    forward_z = noisy_im + torch.randn_like(noisy_im) * sigma
                else:
                    forward_z = noisy_im

                lgvd_z = forward_z.clone()
                lr = denoise_config.lgvd.lr * sigma

                num_steps = int(getattr(denoise_config.lgvd, "num_steps", 0))
                reg_factor = float(getattr(denoise_config.lgvd, "reg_factor", 0.0))
                drift_clip = float(getattr(denoise_config.lgvd, "drift_clip", 10.0))
                noise_scale = float(getattr(denoise_config.lgvd, "noise_scale", 1.0))

                for _ in range(num_steps):
                    score_val = model.score(lgvd_z, sigma)
                    diff_val = (forward_z - lgvd_z)
                    #reg_factor = 1
                    #lr = 1e-3
                    drift = lr * min(sigma * reg_factor, drift_clip) * diff_val
                    lgvd_z += lr * score_val + drift + noise_scale * (2 * lr) ** 0.5 * torch.randn_like(noisy_im)

                if denoise_config.final_step == 'tweedie':
                    z = model.tweedie(lgvd_z, sigma)
                    #z = sigma**2/(self.tau+sigma**2)*noisy_im + self.tau/(self.tau+sigma**2)*z 
                    #z = lgvd_z
                elif denoise_config.final_step == 'ode':
                    diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                    sampler = DiffusionSampler(diffusion_scheduler)
                    z = sampler.sample(model, lgvd_z, SDE=False, verbose=False)
                else:
                    raise Exception(f"Final step {denoise_config.final_step} not supported!!!")

                return z
            else:
                raise Exception(f"Prior type {prior_use_type} not supported!!!")

    # -------------------------
    # Operator norm estimation
    # -------------------------
    def _estimate_norm_sq(self, operator, ref_img, mode: str, iters: int = 20) -> float:
        """
        Estimate ||A||^2 (linear MSE operators) or ||K||^2 (phase retrieval) by power iteration.
        """
        H, W = ref_img.shape[-2], ref_img.shape[-1]
        if mode == "phase_retrieval":
            # K: x01 -> forward_complex(x01), K*: adjoint_complex
            x = torch.randn_like(ref_img)
            x = x / (x.norm() + 1e-12)

            for _ in range(iters):
                u = operator.forward_complex(x)  # treat x as x01 here
                x = operator.adjoint_complex(u, out_hw=(H, W))
                x = x / (x.norm() + 1e-12)

            u = operator.forward_complex(x)
            num = (u.abs() ** 2).sum()
            den = (x ** 2).sum()
            return (num / (den + 1e-12)).item()

        # linear operators: A: x -> operator(x), A^T via autograd
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
    def _grad_log_likelihood_linear(self, operator, x, y, sigma_n):
        Ax = operator(x)
        # log p(y|x) ∝ -1/(2 sigma_n^2) ||Ax - y||^2
        # ∇_x log p(y|x) = -(1/sigma_n^2) A^T (Ax - y)
        return -(1.0 / (sigma_n ** 2)) * self._AT_autograd(operator, x, Ax - y)
    # -------------------------
    # Main sampler
    # -------------------------
    def sample(self, model, ref_img, operator,
               measurement, evaluator=None,
               record=False, verbose=False, wandb=False, **kwargs):

        if record:
            self.trajectory = Trajectory()

        mode = self._mode(operator, measurement)

        K = int(self.admm_config.max_iter)
        pbar = tqdm.trange(K) if verbose else range(K)

        # init (same pattern as DYS)
        x_k, z_k, y_k = self.get_start(ref_img)
        x_k = y_k
        x_bar = x_k.clone()

        # init dual
        sigma_n = float(getattr(operator, "sigma", 0.05))

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

        # optional ||A||^2 print
        if self.print_operator_norm:
            try:
                norm_sq = self._estimate_norm_sq(operator, ref_img, mode=mode, iters=self.norm_power_iters)
                if mode == "phase_retrieval":
                    tqdm.tqdm.write(f"[PDHG] estimated ||K||^2 ≈ {norm_sq:.6f}  (note: x->u uses 0.5*K due to x01=(x+1)/2)")
                else:
                    tqdm.tqdm.write(f"[PDHG] estimated ||A||^2 ≈ {norm_sq:.6f}")
            except Exception as e:
                tqdm.tqdm.write(f"[PDHG] ||A||^2 estimation failed: {e}")

        # convergence check settings (optional)
        x_old = None
        delta_patience = 0
        delta_tol = float(getattr(self.admm_config, "delta_tol", -1.0))
        delta_pat = int(getattr(self.admm_config, "delta_patience", 0))

        t0 = time.time()
        #sigstep = 30
       # sigma_d = 30
        d_rms_mean = 1
        dual_inject_norm = 1
        for step in pbar:
            # denoiser sigma schedule
            #mode = "phase_retrieval"
            #ashu
            #norm_sq = sampler._estimate_norm_sq(operator, ref_img, mode=mode, iters=50)
            #print(norm_sq)         # this is ||K||^2
            #print(norm_sq / 4.0)   # this is ||0.5 K||^2 (the effective one for your x-step)
            #quit()
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma_d = float(self.annealing_scheduler.sigma_steps[t_sigma])
            #self.tau = min(0.01,sigma_d**2)
            #sigma_d = 4/(step+1)
            #self.tau = 1*sigma_d**2
            #sigma_d = 10/(step+1)
            #self.tau = sigma_d
            #self.sigma_dual = 100
            #sigma_d = 0.000000001
            #sigma_d = 0.5/(step+1)**(0.5)
            #self.tau = sigma_d
            #sigma_d = max(sigma_d, dual_inject_norm/np.sqrt(2*0.05))
            #sigma_d = sigma_d*0.98
            #sigstep = sigstep + 1
            #print(sigma_d)
            #if sigma_d < 1:
            #    self.tau = 0.01*sigma_d**2

            theta = 0.0 if self.force_theta_zero else float(self.theta_schedule[min(step, len(self.theta_schedule) - 1)])

            # =========================
            # (1) Dual update
            # =========================
            if mode == "phase_retrieval":
                u_bar = operator.forward_complex(self._to_01(x_bar))
                p_new = self._dual_update_phase_retrieval(
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

            # =========================
            # (2) Primal step (pre-denoise) -> z_k
            # =========================
            if mode == "phase_retrieval":
                # K* p in x01 coords, then chain rule dx01/dx = 1/2
                kstar_p_x01 = operator.adjoint_complex(p_k, out_hw=(x_k.shape[-2], x_k.shape[-1]))
                ATp = 0.5 * kstar_p_x01
            else:
                # A^T p via autograd VJP
                ATp = self._AT_autograd(operator, x_k, p_k)

            z_k = x_k - self.tau * ATp

            # ATp shape: (B,C,H,W) or (B,H,W)
            Delta = self.tau * ATp
            d_rms = Delta.flatten(1).pow(2).mean(dim=1).sqrt()   # shape (B,)
            d_rms_mean = d_rms.mean().item()
            dual_inject = self.tau * ATp   
            dual_inject_norm = self._img_norm_mean(dual_inject)
            dual_inject_over_sigma = dual_inject_norm / sigma_d
            print(dual_inject_over_sigma, sigma_d)
            #quit()

            # =========================
            # (3) Denoise -> x_new
            # =========================
            x_new = self.optimize_denoising(
                z_in=z_k,
                model=model,
                d_k=torch.zeros_like(z_k),
                sigma=sigma_d,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb
            )
            x_new = self._proj(x_new)
            """for lgv_step in range(10):
                lr = self.admm_config.denoise.lgvd.lr * sigma_d  # or your preferred schedule

                score = model.score(x_k, sigma_d)
                glike = self._grad_log_likelihood_linear(operator, x_k, measurement, sigma_n)

                x_k = x_k + lr * (score + glike) + (2 * lr) ** 0.5 * torch.randn_like(x_k)
                x_new = self._proj(x_k)"""
            
    
            # =========================
            # (4) Extrapolation
            # =========================
            x_bar = x_new + theta * (x_new - x_k)
            x_k = x_new
            y_k = x_k  # keep naming compatibility

            # =========================
            # Convergence check (optional)
            # =========================
            if step != 0 and x_old is not None:
                denom = float(np.prod(x_k.shape))
                delta = (x_k - x_old).norm() ** 2 / denom
                if delta_tol > 0 and float(delta) < delta_tol:
                    delta_patience += 1
                    if delta_patience > delta_pat:
                        print(f"Converged with low delta at step {step}")
                        break
                else:
                    delta_patience = 0
                if wandb:
                    wnb.log({
                        "PDHG Iteration": step + 1,
                        "delta": float(delta),
                        "sigma_denoise": float(sigma_d),
                        "theta": float(theta),
                        "tau": float(self.tau),
                        "sigma_dual": float(self.sigma_dual),
                        "wall_time": time.time() - t0,
                    })
            x_old = x_k.clone()

            # =========================
            # Evaluation: print PSNR for BOTH z_k and x_k (as requested)
            # =========================
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    z_k_results = evaluator(gt, measurement, z_k)    # pre-denoise
                    x_k_results = evaluator(gt, measurement, x_k)    # post-denoise

                if verbose:
                    main = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        f'z_k_{main}': f"{z_k_results[main].item():.2f}",
                        f'x_k_{main}': f"{x_k_results[main].item():.2f}",
                    })

                if wandb:
                    for fn_name in x_k_results.keys():
                        wnb.log({
                            f'z_k_{fn_name}': z_k_results[fn_name].item(),
                            f'x_k_{fn_name}': x_k_results[fn_name].item(),
                        })

            if record:
                self._record(y_k=y_k, x_k=x_k, z_k=z_k, sigma=sigma_d)

        return x_k

    # -------------------------
    # Recording / init
    # -------------------------
    def _record(self, y_k, x_k, z_k, sigma):
        self.trajectory.add_tensor('x_k', x_k)   # post-denoise
        self.trajectory.add_tensor('z_k', z_k)   # pre-denoise
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