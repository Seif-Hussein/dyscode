import time
import torch
import wandb as wnb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from utils.diffusion import Scheduler, DiffusionSampler
from utils.logging import Trajectory
from .registry import register_sampler


@register_sampler(name='dys')
def get_sampler(**kwargs):
    """
    Kept under name='dys' for compatibility with existing configs.
    Now implements PDHG (Chambolle–Pock) specialized for phase retrieval.
    """
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError("Latent-space PDHG not implemented.")
    return PHDG(**kwargs)


class PHDG(nn.Module):
    """
    Plug-and-Play PDHG / Chambolle–Pock sampler specialized for phase retrieval:

        min_x  g(x) + f(K x)
    where:
        - x in [-1,1]^d (denoiser domain)
        - x01 = (x+1)/2 in [0,1]^d
        - K(x) = FFT(pad(x01))  (complex)
        - f(u) = (1/(2*sigma_n^2)) * || |u| - y ||^2  (y = measurement amplitude)
        - prox_{tau g} is replaced by diffusion denoiser (optimize_denoising)

    This removes the inner Adam solve used in ADMM for nonlinear operators and replaces it
    with cheap FFT/iFFT + a closed-form prox in Fourier space.
    """

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
        self.tau = self._get_tau_default()                    # primal step
        self.sigma_dual = self._get_sigma_dual_default()      # dual step
        self.theta_schedule = self._build_theta_schedule()    # extrapolation

        # Projection settings (kept for compatibility; applied in primal domain)
        self.proj_min = float(getattr(getattr(self.admm_config, "proj", None), "min", -1.0)
                              if getattr(self.admm_config, "proj", None) is not None else -1.0)
        self.proj_max = float(getattr(getattr(self.admm_config, "proj", None), "max",  1.0)
                              if getattr(self.admm_config, "proj", None) is not None else 1.0)
        self.use_projection = bool(getattr(getattr(self.admm_config, "proj", None), "activate", True)
                                   if getattr(self.admm_config, "proj", None) is not None else True)

    # -------------------------
    # Core utilities
    # -------------------------
    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')
        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    def _get_tau_default(self) -> float:
        """
        Primal stepsize tau.
        Priority:
          1) admm_config.pdhg.tau
          2) admm_config.dys.gamma (old name)
          3) admm_config.gamma_step
          4) admm_config.step_size
          5) admm_config.ml.lr
          6) 0.8
        """
        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        if pdhg_cfg is not None and hasattr(pdhg_cfg, "tau"):
            return float(pdhg_cfg.tau)

        dys_cfg = getattr(self.admm_config, "dys", None)
        if dys_cfg is not None and hasattr(dys_cfg, "gamma"):
            return float(dys_cfg.gamma)

        if hasattr(self.admm_config, "gamma_step"):
            return float(self.admm_config.gamma_step)

        if hasattr(self.admm_config, "step_size"):
            return float(self.admm_config.step_size)

        ml_cfg = getattr(self.admm_config, "ml", None)
        if ml_cfg is not None and hasattr(ml_cfg, "lr"):
            return float(ml_cfg.lr)

        return 0.8

    def _get_sigma_dual_default(self) -> float:
        """
        Dual stepsize sigma.
        Priority:
          1) admm_config.pdhg.sigma_dual
          2) admm_config.pdhg.sigma
          3) default 0.8

        Stability (rough): tau * sigma_dual * ||K||^2 < 1.
        With unitary FFT norm='ortho' and padding/cropping, ||K|| ~ 1 is typical.
        """
        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        if pdhg_cfg is not None:
            if hasattr(pdhg_cfg, "sigma_dual"):
                return float(pdhg_cfg.sigma_dual)
            if hasattr(pdhg_cfg, "sigma"):
                return float(pdhg_cfg.sigma)
        return 0.8

    def _build_theta_schedule(self):
        """
        Extrapolation parameter theta_k in [0,1] typically.
        Priority:
          1) admm_config.pdhg.theta_schedule (same schema as your dys.lambda_schedule)
          2) constant admm_config.pdhg.theta
          3) reuse admm_config.dys.lambda_schedule / dys.lambda
          4) default constant 1.0
        """
        K = int(self.admm_config.max_iter)

        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        if pdhg_cfg is not None:
            ts = getattr(pdhg_cfg, "theta_schedule", None)
            if ts is not None and bool(getattr(ts, "activate", False)):
                th0 = float(getattr(ts, "start", 0.2))
                th1 = float(getattr(ts, "end", 1.0))
                warm = int(getattr(ts, "warmup", 50))
                out = []
                for k in range(K):
                    if warm <= 0:
                        out.append(th1)
                    else:
                        t = min(1.0, k / warm)
                        out.append((1 - t) * th0 + t * th1)
                return out

            if hasattr(pdhg_cfg, "theta"):
                th = float(getattr(pdhg_cfg, "theta"))
                return [th for _ in range(K)]

        # fallback to old dys schedule if present
        dys_cfg = getattr(self.admm_config, "dys", None)
        if dys_cfg is not None:
            ls = getattr(dys_cfg, "lambda_schedule", None)
            if ls is not None and bool(getattr(ls, "activate", False)):
                lam0 = float(getattr(ls, "start", 0.2))
                lam1 = float(getattr(ls, "end", 1.0))
                warm = int(getattr(ls, "warmup", 50))
                out = []
                for k in range(K):
                    if warm <= 0:
                        out.append(lam1)
                    else:
                        t = min(1.0, k / warm)
                        out.append((1 - t) * lam0 + t * lam1)
                return out

            if hasattr(dys_cfg, "lambda"):
                lam = float(getattr(dys_cfg, "lambda"))
                return [lam for _ in range(K)]

        if hasattr(self.admm_config, "lambda"):
            lam = float(getattr(self.admm_config, "lambda"))
            return [lam for _ in range(K)]

        return [1.0 for _ in range(K)]

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_projection:
            return x
        return torch.clamp(x, min=self.proj_min, max=self.proj_max)

    @staticmethod
    def _to_01(x_m11: torch.Tensor) -> torch.Tensor:
        return (x_m11 * 0.5 + 0.5)

    @staticmethod
    def _to_m11(x01: torch.Tensor) -> torch.Tensor:
        return (x01 * 2.0 - 1.0)

    def _get_sigma_n(self, operator) -> float:
        # measurement noise std used in f(u) = 1/(2*sigma_n^2)|| |u|-y ||^2
        # Prefer operator.sigma (your PhaseRetrieval passes sigma=0.05)
        return float(getattr(operator, "sigma", 0.05))

    # -------------------------
    # K / K* for phase retrieval
    # -------------------------
    def _K_forward(self, x_m11: torch.Tensor, operator) -> torch.Tensor:
        """
        Compute u = K(x) = FFT(pad(x01)), complex.
        Uses operator.forward_complex if available; else falls back to torch.fft.
        """
        x01 = self._to_01(x_m11)
        x01 = x01.clamp(0.0, 1.0)

        if hasattr(operator, "forward_complex") and callable(getattr(operator, "forward_complex")):
            return operator.forward_complex(x01)

        # Fallback: assume operator has .pad
        pad = int(getattr(operator, "pad", 0))
        x_pad = F.pad(x01, (pad, pad, pad, pad))
        if not torch.is_complex(x_pad):
            x_pad = x_pad.to(torch.complex64)
        u = torch.fft.fft2(x_pad, norm="ortho")
        return u

    def _K_adjoint(self, p: torch.Tensor, operator, out_hw: tuple[int, int]) -> torch.Tensor:
        """
        Compute x01_like = K*(p) = crop(iFFT(p)), real, in [0,1] coordinates.
        Uses operator.adjoint_complex if available; else falls back to torch.fft.
        """
        if hasattr(operator, "adjoint_complex") and callable(getattr(operator, "adjoint_complex")):
            return operator.adjoint_complex(p, out_hw=out_hw)

        pad = int(getattr(operator, "pad", 0))
        x_pad = torch.fft.ifft2(p, norm="ortho")  # complex
        H, W = out_hw
        if pad > 0:
            x = x_pad[..., pad:pad + H, pad:pad + W]
        else:
            x = x_pad[..., :H, :W]
        return x.real

    # -------------------------
    # Denoiser block (copied/adapted from ADMM)
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

                for _ in range(num_steps):
                    score_val = model.score(lgvd_z, sigma)
                    diff_val = (forward_z - lgvd_z + d_k)
                    drift = lr * min(sigma * reg_factor, 10.0) * diff_val
                    lgvd_z += lr * score_val + drift + (2 * lr) ** 0.5 * torch.randn_like(noisy_im)

                if denoise_config.final_step == 'tweedie':
                    z = model.tweedie(lgvd_z, sigma)
                elif denoise_config.final_step == 'ode':
                    diffusion_scheduler = Scheduler(
                        **self.diffusion_scheduler_config, sigma_max=sigma)
                    sampler = DiffusionSampler(diffusion_scheduler)
                    z = sampler.sample(model, lgvd_z, SDE=False, verbose=False)
                else:
                    raise Exception(f"Final step {denoise_config.final_step} not supported!!!")

                denoised_img = z
            else:
                raise Exception(f"Prior type {prior_use_type} not supported!!!")

        return denoised_img

    # -------------------------
    # Prox for magnitude MSE (used via Moreau for dual update)
    # -------------------------
    @staticmethod
    def prox_mag_mse(u0: torch.Tensor, y: torch.Tensor,
                    sigma_dual: float, sigma_n: float,
                    eps: float = 1e-12) -> torch.Tensor:
        """
        Computes prox_{(1/sigma_dual) f}(u0) where
            f(u) = (1/(2 sigma_n^2)) || |u| - y ||^2
        u0: complex tensor, y: real >=0 tensor same shape (broadcast ok)
        """
        r0 = torch.abs(u0)
        a = sigma_dual * (sigma_n ** 2)          # a = sigma_dual*sigma_n^2
        r_star = (r0 * a + y) / (a + 1.0)        # (r0*a + y)/(a+1)
        scale = r_star / (r0 + eps)
        return u0 * scale

    # -------------------------
    # Main sampler
    # -------------------------
    def sample(self, model, ref_img, operator,
               measurement, evaluator=None,
               record=False, verbose=False, wandb=False, **kwargs):
        """
        measurement: for phase retrieval should be amplitude y >= 0 in the padded Fourier grid,
                     matching operator(ref_img) output shape (B,C,Hpad,Wpad).
        """
        if record:
            self.trajectory = Trajectory()

        K_iters = int(self.admm_config.max_iter)
        pbar = tqdm.trange(K_iters) if verbose else range(K_iters)

        # Start from the same init logic
        x0, z0, y0 = self.get_start(ref_img)

        # PDHG state in image domain [-1,1]
        x_k = y0.clone()              # primal variable
        x_bar = x_k.clone()           # extrapolated primal

        # Dual variable in Fourier domain (complex), init zeros with correct shape
        with torch.no_grad():
            u_bar = self._K_forward(x_bar, operator)
            p_k = torch.zeros_like(u_bar)  # complex

        # For delta checks / logging
        x_old = None
        z_old = None
        y_old = None
        delta_patience = 0
        delta_tol = float(getattr(self.admm_config, "delta_tol", -1.0))
        delta_pat = int(getattr(self.admm_config, "delta_patience", 0))

        t0 = time.time()

        sigma_n = self._get_sigma_n(operator)
        out_hw = (x_k.shape[-2], x_k.shape[-1])

        for step in pbar:
            # denoiser sigma schedule (same pattern as ADMM)
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma_d = float(self.annealing_scheduler.sigma_steps[t_sigma])

            # extrapolation theta schedule
            theta = float(self.theta_schedule[min(step, len(self.theta_schedule) - 1)])

            # ----------------------------
            # (1) Dual update: p^{k+1} = prox_{sigma f*}( p^k + sigma K x_bar )
            # via Moreau: prox_{sigma f*}(q) = q - sigma * prox_{f/sigma}(q/sigma)
            # ----------------------------
            u_bar = self._K_forward(x_bar, operator)                 # complex
            q = p_k + self.sigma_dual * u_bar                        # complex
            w = q / self.sigma_dual                                  # complex

            # prox_{(1/sigma_dual) f}(w) for magnitude MSE
            u_prox = self.prox_mag_mse(
                u0=w,
                y=measurement,
                sigma_dual=self.sigma_dual,
                sigma_n=sigma_n
            )
            p_k = q - self.sigma_dual * u_prox

            # ----------------------------
            # (2) Primal "gradient-like" step: z = x - tau * K* p
            # then PnP prox: x^{k+1} ≈ denoise(z)
            # ----------------------------
            kstar_p = self._K_adjoint(p_k, operator, out_hw=out_hw)   # real, in [0,1] coords
            # map K* p back into [-1,1] scale:
            # x01 = (x+1)/2, so a step in x01 translates to 2x01-1; gradient step in x01 domain:
            # z01 = x01 - tau*kstar_p; then z_m11 = 2*z01 - 1
            x01 = self._to_01(x_k).clamp(0.0, 1.0)
            z01 = x01 - self.tau * kstar_p
            z_k = self._to_m11(z01)                                  # this is the pre-denoise point (data step)

            # denoise (PnP prox)
            d_k = torch.zeros_like(z_k)                              # no ADMM-style dual correction in PDHG
            x_new = self.optimize_denoising(
                z_in=z_k,
                model=model,
                d_k=d_k,
                sigma=sigma_d,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb
            )

            # project/clamp in denoiser domain if requested
            x_new = self._proj(x_new)

            # ----------------------------
            # (3) Extrapolation: x_bar = x_new + theta (x_new - x_old)
            # ----------------------------
            x_bar = x_new + theta * (x_new - x_k)

            # ----------------------------
            # Convergence / logging in the same "delta_t" style
            # We'll log:
            #   - x_k  := z_k (data/primal step before denoise)
            #   - z_k  := x_new (denoised)
            #   - y_k  := x_bar (extrapolated)
            # ----------------------------
            y_k = x_bar

            if step != 0 and x_old is not None:
                denom = float(np.prod(x_new.shape))
                delta_1 = (z_k - x_old).norm() ** 2 / denom      # pre-denoise changes
                delta_2 = (x_new - z_old).norm() ** 2 / denom    # denoised changes
                delta_3 = (y_k - y_old).norm() ** 2 / denom      # extrapolated changes
                delta_t = delta_1 + delta_2 + delta_3

                if delta_tol > 0 and float(delta_t) < delta_tol:
                    delta_patience += 1
                    if delta_patience > delta_pat:
                        print(f"Converged with low delta at step {step}")
                        x_k = x_new
                        break
                else:
                    delta_patience = 0

                if wandb:
                    wnb.log({
                        "PDHG Iteration": step + 1,
                        "delta_t": float(delta_t),
                        "sigma_denoise": float(sigma_d),
                        "theta": float(theta),
                        "tau": float(self.tau),
                        "sigma_dual": float(self.sigma_dual),
                        "sigma_n": float(sigma_n),
                        "wall_time": time.time() - t0,
                    })

            # Update old values and iterate
            x_old = z_k.detach().clone()    # previous pre-denoise
            z_old = x_new.detach().clone()  # previous denoised
            y_old = y_k.detach().clone()    # previous extrapolated

            x_k = x_new

            # Evaluation (keep the same evaluator interface as your ADMM/DYS)
            x_k_results = z_k_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    # evaluator expects an image-space tensor; we provide both
                    x_k_results = evaluator(gt, measurement, z_k)    # "data/primal" (pre-denoise)
                    z_k_results = evaluator(gt, measurement, x_new)  # "denoised"

                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x_k' + '_' + main_eval_fn_name: f"{x_k_results[main_eval_fn_name].item():.2f}",
                        'z_k' + '_' + main_eval_fn_name: f"{z_k_results[main_eval_fn_name].item():.2f}",
                    })

                if wandb:
                    for fn_name in x_k_results.keys():
                        wnb.log({
                            f'x_k_{fn_name}': x_k_results[fn_name].item(),
                            f'z_k_{fn_name}': z_k_results[fn_name].item(),
                        })

            if record:
                self._record(
                    y_k=y_k, x_k=z_k, z_k=x_new,
                    sigma=sigma_d,
                    x_k_results=x_k_results,
                    z_k_results=z_k_results
                )

        # Return the denoised primal
        return x_k

    # -------------------------
    # Recording / init (compatible style)
    # -------------------------
    def _record(self, y_k, x_k, z_k, sigma, x_k_results, z_k_results):
        self.trajectory.add_tensor('x_k', x_k)
        self.trajectory.add_tensor('z_k', z_k)
        self.trajectory.add_tensor('y_k', y_k)
        self.trajectory.add_value('sigma', sigma)
        for name in x_k_results.keys():
            self.trajectory.add_value(f'x_k_{name}', x_k_results[name])
        for name in z_k_results.keys():
            self.trajectory.add_value(f'z_k_{name}', z_k_results[name])

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
