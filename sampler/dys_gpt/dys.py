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


@register_sampler(name='dys')
def get_sampler(**kwargs):
    """
    Match ADMM's callable conditions:
      - expects kwargs contains 'latent'
      - raises if latent True
      - returns an nn.Module sampler instance
    """
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError("Latent-space DYS not implemented.")
    return DYS(**kwargs)


class DYS(nn.Module):
    r"""
    Plug-and-Play Davis–Yin Splitting (DYS) sampler.

    We implement the relaxed DYS iteration for a 3-term objective

        minimize_x   f(x) + g(x) + h(x),

    where
      - h is differentiable (we use operator.loss as h, via autograd),
      - g is proximable (default: box constraint / clamping),
      - f is proximable (replaced by a *tamed* score-based prox via AC-DC, i.e., Algorithm 1
        in the Taming paper).

    Iteration (relaxed DYS):
        x_k = prox_{γ g}(y_k)
        r_k = 2 x_k - y_k - γ ∇h(x_k)
        z_k = prox_{γ f}(r_k)     (PnP: z_k = D_{σ_k}(r_k), with AC-DC)
        y_{k+1} = y_k + λ_k (z_k - x_k)

    Notes for consistency and debugging:
      * In your previous file version, z_k was being overwritten by a projection step (or clamp),
        effectively disabling the denoiser. That has been fixed: z_k now comes from the AC-DC
        denoiser by default, and optional physics projections are explicitly controlled via config.
      * The DC drift term is now implemented exactly as in Algorithm 1 in the Taming paper:
            (z_ac - w) / σ_s(k)^2
        with default schedule σ_s(k) = σ_s0 / sqrt(σ_k). See Appendix H.1 of the paper.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config,
                 lgvd_config, admm_config, device='cuda', **kwargs):
        super().__init__()

        self.annealing_scheduler_config, self.diffusion_scheduler_config = \
            self._check(annealing_scheduler_config, diffusion_scheduler_config)

        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.admm_config = admm_config  # keep name for compatibility with config tree
        self.device = device

        # ---- Diffusion parameters (only used if denoise.final_step == 'ode')
        self.betas = np.linspace(admm_config.denoise.diffusion.beta_start,
                                 admm_config.denoise.diffusion.beta_end,
                                 admm_config.denoise.diffusion.T,
                                 dtype=np.float64)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Optional regularizers hook (kept for compatibility with your broader framework)
        self.regularizers = getattr(self.admm_config, "regularizers", None)

        # DYS hyperparameters
        self.gamma_step = self._get_gamma_step_default()
        self.lambda_schedule = self._build_lambda_schedule()
        self.data_scale = self._get_data_scale_default()

        # Projection settings (defaults match your ADMM clamping behavior)
        proj_cfg = getattr(self.admm_config, "proj", None)
        self.proj_min = float(getattr(proj_cfg, "min", -1.0)) if proj_cfg is not None else -1.0
        self.proj_max = float(getattr(proj_cfg, "max",  1.0)) if proj_cfg is not None else  1.0
        self.use_projection = bool(getattr(proj_cfg, "activate", True)) if proj_cfg is not None else True

        # prox_g selection
        dys_cfg = getattr(self.admm_config, "dys", None)
        self.prox_g_mode = str(getattr(dys_cfg, "prox_g", "box")).lower() if dys_cfg is not None else "box"
        # Options: "box" (default), "identity", "amplitude" (Fourier magnitude prox for phase retrieval)

        # Fourier-amplitude prox settings (only used if prox_g_mode == "amplitude")
        self.fourier_tau = float(getattr(dys_cfg, "fourier_tau", float("inf"))) if dys_cfg is not None else float("inf")
        self.fourier_eps = float(getattr(dys_cfg, "fourier_eps", 1e-12)) if dys_cfg is not None else 1e-12
        self.fourier_clamp01 = bool(getattr(dys_cfg, "fourier_clamp01", True)) if dys_cfg is not None else True
        self.fourier_clamp_box = bool(getattr(dys_cfg, "fourier_clamp_box", True)) if dys_cfg is not None else True

        # Optional post-denoise projection (usually leave False to keep the splitting interpretation clean;
        # the next iteration's prox_g will enforce constraints anyway).
        self.post_denoise_proj = bool(getattr(dys_cfg, "post_denoise_proj", False)) if dys_cfg is not None else False

        # Optional additional smooth regularization h_reg(x) (keeps 3-operator structure)
        # Supported: "none" | "l2" (μ/2 ||x||^2) | "grad_l2" (μ/2 ||∇x||^2)
        self.smooth_reg_type = str(getattr(dys_cfg, "smooth_reg_type", "none")).lower() if dys_cfg is not None else "none"
        self.smooth_reg_weight = float(getattr(dys_cfg, "smooth_reg_weight", 0.0)) if dys_cfg is not None else 0.0

    # -------------------------
    # Config utilities
    # -------------------------
    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
        Mirrors ADMM._check(): remove sigma_max from diffusion scheduler config.
        Do NOT forcibly override sigma_final; allow the caller to set it (to match Taming, use sigma_final=0.1).
        """
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        # Backward-compatible default:
        if 'sigma_final' not in annealing_scheduler_config:
            annealing_scheduler_config['sigma_final'] = 0.0

        return annealing_scheduler_config, diffusion_scheduler_config

    def _get_gamma_step_default(self) -> float:
        """
        DYS stepsize γ for the gradient term.

        Priority order (so you can control it in configs without changing call sites):
          1) admm_config.dys.gamma
          2) admm_config.gamma_step
          3) admm_config.step_size
          4) admm_config.ml.lr   (often exists; reasonable fallback)
          5) 1e-2
        """
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

        return 1e-2

    def _get_data_scale_default(self) -> float:
        """
        Optional scaling applied to ∇ operator.loss.

        Motivation: in the Taming paper's ADMM x-subproblem, the data term is scaled by 1/ρ (Eq. 7a).
        A natural DYS analogue is to scale ∇ℓ by 1/ρ as well.

        Priority:
          1) admm_config.dys.data_scale
          2) 1 / admm_config.rho   (if present)
          3) 1.0
        """
        dys_cfg = getattr(self.admm_config, "dys", None)
        if dys_cfg is not None and hasattr(dys_cfg, "data_scale"):
            return float(dys_cfg.data_scale)

        if hasattr(self.admm_config, "rho"):
            rho = float(self.admm_config.rho)
            if rho > 0:
                return 1.0 / rho

        return 1.0

    def _build_lambda_schedule(self):
        """
        Build λ_k schedule.

        Priority:
          1) admm_config.dys.lambda_schedule (if present)
          2) constant admm_config.dys.lambda
          3) constant admm_config.lambda
          4) default constant 1.0

        Supported schedule format (recommended):
          admm_config.dys.lambda_schedule.activate = True
          admm_config.dys.lambda_schedule.start = 0.2
          admm_config.dys.lambda_schedule.end = 1.0
          admm_config.dys.lambda_schedule.warmup = 50
        """
        K = int(getattr(self.admm_config, "max_iter", 100))
        dys_cfg = getattr(self.admm_config, "dys", None)

        # schedule
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

            # constant lambda
            if hasattr(dys_cfg, "lambda"):
                lam = float(getattr(dys_cfg, "lambda"))
                return [lam for _ in range(K)]

        # fallback constant lambda in root config
        if hasattr(self.admm_config, "lambda"):
            lam = float(getattr(self.admm_config, "lambda"))
            return [lam for _ in range(K)]

        # default
        return [1.0 for _ in range(K)]

    # -------------------------
    # Prox operators
    # -------------------------
    def _proj_box(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_projection:
            return x
        return torch.clamp(x, min=self.proj_min, max=self.proj_max)

    @staticmethod
    def _fft2c(x: torch.Tensor) -> torch.Tensor:
        # centered, orthonormal FFT2 (fastMRI-style)
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        X = torch.fft.fft2(x, norm="ortho")
        X = torch.fft.fftshift(X, dim=(-2, -1))
        return X

    @staticmethod
    def _ifft2c(X: torch.Tensor) -> torch.Tensor:
        X = torch.fft.ifftshift(X, dim=(-2, -1))
        x = torch.fft.ifft2(X, norm="ortho")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    def _prox_fourier_amplitude(self,
                               x_in: torch.Tensor,
                               amplitude_meas: torch.Tensor,
                               pad: int,
                               tau: float = float("inf"),
                               eps: float = 1e-12,
                               clamp01: bool = True,
                               clamp_box: bool = True) -> torch.Tensor:
        """
        Fourier-magnitude proximal / projection for phase retrieval.

        This implements the (nonconvex) proximal map in the *Fourier domain*:
            prox_{τ * 0.5|| |u| - y ||^2}(v)
        which reduces to the well-known amplitude projection when τ = +inf:
            u <- y * exp(i angle(v))

        In complex scalar form (per Fourier coefficient), with v given:
            u* = a* exp(i angle(v)),
            a* = (tau*y + |v|) / (tau + 1)   (soft update)
            and for tau=inf: a*=y.

        We apply this to v = FFT(x_in) (after the same [−1,1]→[0,1] mapping and padding used
        in your PhaseRetrieval forward operator).

        Parameters
        ----------
        tau : float
            tau = +inf  => hard projection to measured magnitude
            finite tau  => soft magnitude update (useful for noisy amplitude)
        """
        # map [-1,1] -> [0,1]
        x01 = 0.5 * (x_in + 1.0)
        if clamp01:
            x01 = torch.clamp(x01, 0.0, 1.0)

        # pad
        if pad > 0:
            x01 = F.pad(x01, (pad, pad, pad, pad))

        # complex FFT
        if not torch.is_complex(x01):
            x01_c = x01.to(torch.complex64)
        else:
            x01_c = x01

        V = self._fft2c(x01_c)
        mag = torch.abs(V)
        phase = V / (mag + eps)

        # amplitude update
        if math.isinf(tau):
            new_mag = amplitude_meas
        else:
            tau_f = float(tau)
            new_mag = (tau_f * amplitude_meas + mag) / (tau_f + 1.0)

        U = phase * new_mag
        x01_new = self._ifft2c(U).real

        # crop
        if pad > 0:
            x01_new = x01_new[..., pad:-pad, pad:-pad]

        if clamp01:
            x01_new = torch.clamp(x01_new, 0.0, 1.0)

        # map back [0,1] -> [-1,1]
        x_new = 2.0 * x01_new - 1.0
        if clamp_box and self.use_projection:
            x_new = torch.clamp(x_new, min=self.proj_min, max=self.proj_max)

        return x_new

    def prox_g(self, y: torch.Tensor, operator, measurement) -> torch.Tensor:
        """
        prox_g (first prox in DYS).

        Modes:
          - "box": box projection (clamp) (default)
          - "identity": no prox (g = 0)
          - "amplitude": Fourier magnitude prox for phase retrieval (requires operator.pad and amplitude measurement)
        """
        mode = self.prox_g_mode

        if mode in ("box", "clamp"):
            return self._proj_box(y)

        if mode in ("identity", "none"):
            return y

        if mode in ("amplitude", "fourier", "fourier_amplitude"):
            pad = int(getattr(operator, "pad", 0))
            if pad is None:
                pad = 0
            return self._prox_fourier_amplitude(
                x_in=y,
                amplitude_meas=measurement,
                pad=pad,
                tau=self.fourier_tau,
                eps=self.fourier_eps,
                clamp01=self.fourier_clamp01,
                clamp_box=self.fourier_clamp_box,
            )

        # Fallback
        return self._proj_box(y)

    # -------------------------
    # Gradients
    # -------------------------
    def _grad_h_data(self, x: torch.Tensor, operator, measurement) -> torch.Tensor:
        """
        Compute ∇_x operator.loss(x, measurement) via autograd.

        NOTE: This requires operator.loss to be constructed from torch ops.
        """
        x_ = x.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = operator.loss(x_, measurement)  # scalar
            grad = torch.autograd.grad(loss, x_, create_graph=False, retain_graph=False)[0]
        return grad.detach()

    def _grad_h_smooth_reg(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional explicit smooth regularizer gradient (keeps DYS as a 3-operator method).

        - l2:       (μ/2)||x||^2        => μ x
        - grad_l2:  (μ/2)||∇x||^2       => μ * Laplacian(x) (with Neumann-ish padding)
        """
        mu = float(self.smooth_reg_weight)
        if mu <= 0 or self.smooth_reg_type in ("none", "off", ""):
            return torch.zeros_like(x)

        if self.smooth_reg_type == "l2":
            return mu * x

        if self.smooth_reg_type == "grad_l2":
            # Discrete Laplacian per channel (depthwise conv)
            # kernel = [[0,1,0],[1,-4,1],[0,1,0]]
            k = torch.tensor([[0.0, 1.0, 0.0],
                              [1.0, -4.0, 1.0],
                              [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype)
            k = k.view(1, 1, 3, 3)
            C = x.shape[1]
            weight = k.repeat(C, 1, 1, 1)  # (C,1,3,3) depthwise
            lap = F.conv2d(x, weight, bias=None, stride=1, padding=1, groups=C)
            return mu * lap

        # Unknown reg => no-op
        return torch.zeros_like(x)

    def _grad_h(self, x: torch.Tensor, operator, measurement) -> torch.Tensor:
        """
        Total gradient ∇h = data_scale * ∇ℓ + ∇h_reg.
        """
        grad_data = self._grad_h_data(x, operator=operator, measurement=measurement)
        grad_reg = self._grad_h_smooth_reg(x)
        return self.data_scale * grad_data + grad_reg

    # -------------------------
    # Tamed score denoiser (AC-DC, Algorithm 1 in Taming)
    # -------------------------
    def optimize_denoising(self,
                           z_in: torch.Tensor,
                           model,
                           d_k,
                           sigma: float,
                           prior_use_type: str = "denoise",
                           wandb: bool = False,
                           ) -> torch.Tensor:
        """
        AC-DC denoising (Algorithm 1 in Taming), used here as prox_f.

        This is *not* ADMM-specific: the paper explicitly notes the denoiser can be plugged into other
        proximal-operator based schemes (proximal gradient, variable splitting, etc.). The only change
        is the choice of the input iterate z_e; in DYS we set z_e := r_k.

        Config mapping to Taming (Appendix H.1):
          - J (DC steps)                 : denoise.lgvd.num_steps       (paper: 10)
          - η(k) = eta0 * σ(k)           : denoise.lgvd.lr              (paper eta0=5e-4)
          - σ_s(k) = σ_s0 / sqrt(σ(k))   : denoise.lgvd.sigma_s0        (paper σ_s0=0.1)
          - AC noise on/off              : denoise.ac_noise             (default True)
          - Final step                   : denoise.final_step ∈ {'tweedie','ode'}
        """
        denoise_config = self.admm_config.denoise

        if prior_use_type not in ["denoise"]:
            raise Exception(f"Prior type {prior_use_type} not supported!!!")

        with torch.no_grad():
            z_e = z_in

            # ---- AC step
            ac_noise = bool(getattr(denoise_config, "ac_noise", True))
            if ac_noise and sigma > 0:
                z_ac = z_e + torch.randn_like(z_e) * float(sigma)
            else:
                z_ac = z_e
            #z_ac = z_e
            # ---- DC step (conditional Langevin dynamics)
            J = int(getattr(denoise_config.lgvd, "num_steps", 0))
            w = z_ac.clone()

            if J > 0 and sigma > 0:
                # η(k) = eta0 * σ(k)
                eta0 = float(getattr(denoise_config.lgvd, "lr", 5e-4))
                eta = eta0 * float(sigma)

                # σ_s(k) = σ_s0 / sqrt(σ(k))
                # Backward compatibility: if sigma_s0 not provided but reg_factor exists, interpret
                # reg_factor ≈ 1 / sigma_s0^2.
                if hasattr(denoise_config.lgvd, "sigma_s0"):
                    sigma_s0 = float(getattr(denoise_config.lgvd, "sigma_s0"))
                else:   
                    reg_factor = float(getattr(denoise_config.lgvd, "reg_factor", 100.0))
                    sigma_s0 = 1.0 / math.sqrt(max(reg_factor, 1e-12))

                
                

                sigma_s = sigma_s0 / math.sqrt(max(float(sigma), 1e-12))
                inv_sigma_s2 = 1.0 / (sigma_s * sigma_s)

                # Optional clip on inv_sigma_s2 to avoid overly stiff dynamics (off by default)
                inv_clip = getattr(denoise_config.lgvd, "inv_sigma_s2_clip", None)
                if inv_clip is not None:
                    inv_sigma_s2 = min(inv_sigma_s2, float(inv_clip))

                sqrt_2eta = math.sqrt(max(2.0 * eta, 0.0))

                for _ in range(J):
                    score_val = model.score(w, sigma)
                    drift = min(float(sigma)*reg_factor,10) * (z_ac - w + d_k) + score_val
                    #print(eta*drift)
                    #quit()
                    w = w + eta * drift + sqrt_2eta * torch.randn_like(w)

            z_dc = w

            # ---- final step
            if denoise_config.final_step == 'tweedie':
                # assumes model.tweedie implements z + σ^2 score(z,σ)
                z_out = model.tweedie(z_dc, sigma)
            elif denoise_config.final_step == 'ode':
                diffusion_scheduler = Scheduler(
                    **self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionSampler(diffusion_scheduler)
                z_out = sampler.sample(model, z_dc, SDE=False, verbose=False)
            else:
                raise Exception(f"Final step {denoise_config.final_step} not supported!!!")

            if self.post_denoise_proj:
                z_out = self._proj_box(z_out)

        return z_out

    # -------------------------
    # Main sampler
    # -------------------------
    def sample(self, model, ref_img, operator,
               measurement, evaluator=None,
               record=False, verbose=False, wandb=False, **kwargs):
        """
        Callable under the same conditions as ADMM.sample(...).

        Returns:
          - z_k (denoised / prior output) by default,
          - or x_k if admm_config.dys.return_xA=True.
        """
        if record:
            self.trajectory = Trajectory()

        K = int(getattr(self.admm_config, "max_iter", 100))
        pbar = tqdm.trange(K) if verbose else range(K)

        # Initialize DYS state.
        x_k, z_k, y_k = self.get_start(ref_img)

        # Old values for convergence checks
        x_old, z_old, y_old = None, None, None
        delta_patience = 0

        delta_tol = float(getattr(self.admm_config, "delta_tol", -1.0))
        delta_pat = int(getattr(self.admm_config, "delta_patience", 0))

        # Optional: return x_k instead of z_k
        return_xA = bool(getattr(getattr(self.admm_config, "dys", None), "return_xA", False))

        t0 = time.time()

        for step in pbar:
            # sigma schedule (annealing)
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma = float(self.annealing_scheduler.sigma_steps[t_sigma])

            # lambda schedule
            lam = float(self.lambda_schedule[min(step, len(self.lambda_schedule) - 1)])

            # (1) prox_g
            x_k = self.prox_g(y_k, operator=operator, measurement=measurement)

            # (2) gradient step for h
            grad = self._grad_h(x_k, operator=operator, measurement=measurement)

            # reflected, gradient-corrected point
            r_k = 2.0 * x_k - y_k - float(self.gamma_step) * grad

            # (3) prox_f via tamed score denoiser (AC-DC)
            z_k = self.optimize_denoising(
                z_in=r_k,
                model=model,
                d_k = y_k - x_k,
                sigma=sigma,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb
            )

            # (4) relaxed update
            y_k = y_k + lam * (z_k - x_k)
            #print(y_k)
            #print(lam)

            # Convergence check (delta_t style)
            if step != 0:
                denom = float(np.prod(x_k.shape))  # normalize per element
                delta_1 = (x_k - x_old).norm() ** 2 / denom
                delta_2 = (z_k - z_old).norm() ** 2 / denom
                delta_3 = (y_k - y_old).norm() ** 2 / denom
                delta_t = delta_1 + delta_2 + delta_3

                if delta_tol > 0 and float(delta_t) < delta_tol:
                    delta_patience += 1
                    if delta_patience > delta_pat:
                        print(f"Converged with low delta at step {step}")
                        break
                else:
                    delta_patience = 0

                if wandb:
                    wnb.log({
                        "DYS Iteration": step + 1,
                        "delta_t": float(delta_t),
                        "sigma": float(sigma),
                        "lambda": lam,
                        "gamma_step": float(self.gamma_step),
                        "data_scale": float(self.data_scale),
                        "wall_time": time.time() - t0,
                    })

            # Update old values
            x_old, z_old, y_old = x_k.clone(), z_k.clone(), y_k.clone()

            # Evaluation
            x_k_results = z_k_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x_k_results = evaluator(gt, measurement, x_k)  # "prox_g" variable
                    z_k_results = evaluator(gt, measurement, z_k)  # "prior" variable

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
                    y_k=y_k, x_k=x_k, z_k=z_k,
                    sigma=sigma,
                    x_k_results=x_k_results,
                    z_k_results=z_k_results
                )

        return x_k if return_xA else z_k

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
        """
        Create initial x_k, z_k, y_k.

        To be callable under same config conditions as ADMM, we interpret
        admm_config.init_factor if present. If absent, we default to noise init.

        Convention:
          - If init_factor has keys x/z/u/y, use them.
          - y_k is the DYS shadow iterate; prefer key 'y', else 'u', else 'z', else 'x'.
        """
        sigma0 = float(getattr(self.annealing_scheduler, "sigma_max", 1.0))
        init_factor = getattr(self.admm_config, "init_factor", None)

        # Default: pure noise init
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
