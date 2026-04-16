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
def get_sampler(**kwargs):
    """PDHG sampler (registered as 'drs' for historical reasons).

    Expected kwargs contain 'latent'. This implementation supports only image-space.
    """
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError("Latent-space PDHG not implemented.")
    return DRS(**kwargs)


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
    t = curve[max(0, len(curve) - tail):]
    print(f"    head: {['%.3e' % v for v in h]}")
    if len(curve) > head:
        print(f"    tail: {['%.3e' % v for v in t]}")


class DRS(nn.Module):
    """PDHG (Chambolle–Pock) PnP sampler.

    Core iteration:
        p^{k+1} = prox_{σ f*}(p^k + σ K \bar x^k)          (dual)
        z^k     = x^k - τ K^* p^{k+1}                      (primal pre-step)
        x^{k+1} = D_{σ_d}(z^k)                             (diffusion denoiser)
        \bar x^{k+1} = x^{k+1} + θ (x^{k+1} - x^k)          (extrapolation)

    This file additionally supports an optional **PDHG-native dual corrector**:
    replace the exact proximal dual update by an inexact Langevin (ULA) chain in
    measurement space that targets the conditional density
        π_w(u) ∝ exp( - f_y(u) - (σ/2) ||u - w||^2 ),
    with w = K \bar x + p/σ.

    See the accompanying theorem in the chat for the contraction bound.
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

        # PDHG hyperparameters
        self.tau = self._get_tau_default()
        self.sigma_dual = self._get_sigma_dual_default()
        self.theta_schedule = self._build_theta_schedule()

        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        self.print_operator_norm = bool(getattr(pdhg_cfg, "print_operator_norm", False)) if pdhg_cfg is not None else False
        self.norm_power_iters = int(getattr(pdhg_cfg, "norm_power_iters", 20)) if pdhg_cfg is not None else 20
        self.force_theta_zero = bool(getattr(pdhg_cfg, "force_theta_zero", False)) if pdhg_cfg is not None else False

        # Projection box on x (kept for compatibility)
        proj_cfg = getattr(self.admm_config, "proj", None)
        self.proj_min = float(getattr(proj_cfg, "min", -1.0)) if proj_cfg is not None else -1.0
        self.proj_max = float(getattr(proj_cfg, "max", 1.0)) if proj_cfg is not None else 1.0
        self.use_projection = bool(getattr(proj_cfg, "activate", True)) if proj_cfg is not None else True

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
        # Keep original behaviour: the diffusion scheduler is re-instantiated per-sigma when needed.
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
    # Dual prox updates (exact)
    # -------------------------
    def _dual_update_linear_mse(self, p: torch.Tensor, Ax_bar: torch.Tensor, y: torch.Tensor,
                               sigma_dual: float, sigma_n: float):
        """Exact prox_{σ f*} for quadratic likelihood f_y(u)=1/(2σ_n^2)||u-y||^2."""
        q = p + sigma_dual * Ax_bar
        denom = 1.0 + sigma_dual * (sigma_n ** 2)
        return (q - sigma_dual * y) / denom

    def _dual_update_phase_retrieval_diag(self, p: torch.Tensor, u_bar: torch.Tensor, y_amp: torch.Tensor,
                                          sigma_dual: float, sigma_n: float, eps: float = 1e-12):
        """Exact prox_{σ f*} for amplitude likelihood, plus diagnostics."""
        q = p + sigma_dual * u_bar
        w = q / sigma_dual

        r0 = w.abs()
        a = sigma_dual * (sigma_n ** 2)
        r_star = (a * r0 + y_amp) / (a + 1.0)

        w_prox = w * (r_star / (r0 + eps))
        p_new = q - sigma_dual * w_prox

        w_flat = w.reshape(w.shape[0], -1)
        wp_flat = w_prox.reshape(w_prox.shape[0], -1)
        n_per = wp_flat.shape[1]

        prox_move = float((wp_flat - w_flat).abs().pow(2).sum(dim=1).sqrt().mean().detach() / math.sqrt(n_per))
        prox_radial_resid = float((w_prox.abs() - r_star).abs().mean().detach())

        return p_new, prox_move, prox_radial_resid

    # -------------------------
    # Prox of (1/σ) f_y in measurement space (used for warm-start)
    # -------------------------
    @staticmethod
    def _prox_f_over_sigma_linear_mse(w: torch.Tensor, y: torch.Tensor, sigma_dual: float, sigma_n: float):
        """prox_{(1/σ) f_y}(w) for f_y(u)=1/(2σ_n^2)||u-y||^2, where σ = sigma_dual."""
        # Solve: min_u (1/(2σ_n^2))||u-y||^2 + (σ/2)||u-w||^2.
        # Closed form: u = (σ σ_n^2 w + y) / (σ σ_n^2 + 1)
        a = sigma_dual * (sigma_n ** 2)
        return (a * w + y) / (a + 1.0)

    @staticmethod
    def _prox_f_over_sigma_phase_retrieval(w: torch.Tensor, y_amp: torch.Tensor, sigma_dual: float,
                                          sigma_n: float, eps: float = 1e-12):
        """prox_{(1/σ) f_y}(w) for amplitude likelihood, σ = sigma_dual."""
        r0 = w.abs()
        a = sigma_dual * (sigma_n ** 2)
        r_star = (a * r0 + y_amp) / (a + 1.0)
        return w * (r_star / (r0 + eps))

    # -------------------------
    # Gradients of f_y in measurement space (for ULA-u)
    # -------------------------
    @staticmethod
    def _grad_f_linear_mse(u: torch.Tensor, y: torch.Tensor, sigma_n: float) -> torch.Tensor:
        return (u - y) / (sigma_n ** 2)

    @staticmethod
    def _grad_f_phase_retrieval(u: torch.Tensor, y_amp: torch.Tensor, sigma_n: float, eps: float = 1e-12) -> torch.Tensor:
        # f(u) = (1/(2σ_n^2)) || |u| - y ||^2, u complex.
        r = u.abs()
        # gradient in R^{2m} corresponds to ((r - y)/σ_n^2) * u/r
        scale = (r - y_amp) / (sigma_n ** 2)
        return scale * (u / (r + eps))

    # -------------------------
    # PDHG-native dual corrector: ULA in measurement space
    # -------------------------
    def _dual_update_ula_u(self, p: torch.Tensor, v_bar: torch.Tensor, measurement: torch.Tensor,
                           mode: str, sigma_dual: float, sigma_n: float,
                           num_steps: int, step_size: float,
                           init: str = "prox", noise_scale: float = 1.0,
                           eps: float = 1e-12):
        """Replace the exact prox_{σ f*} dual step by a ULA chain in measurement space.

        We define w := v_bar + p/σ, and target
            π_w(u) ∝ exp( - f_y(u) - (σ/2)||u-w||^2 ).

        ULA step:
            u <- u - h (∇ f_y(u) + σ (u-w)) + sqrt(2h) ξ.

        Then set p_new = σ (w - u).

        Returns
        -------
        p_new : tensor (real or complex)
        ula_u_move : float, mean ||u_J - u_0|| / sqrt(n)
        """
        if num_steps <= 0:
            raise ValueError("num_steps must be positive for dual ULA corrector")
        if step_size <= 0:
            raise ValueError("step_size must be positive for dual ULA corrector")

        w = v_bar + p / sigma_dual

        # warm start
        if init == "prox":
            if mode == "phase_retrieval":
                u = self._prox_f_over_sigma_phase_retrieval(w=w, y_amp=measurement, sigma_dual=sigma_dual,
                                                           sigma_n=sigma_n, eps=eps)
            else:
                u = self._prox_f_over_sigma_linear_mse(w=w, y=measurement, sigma_dual=sigma_dual, sigma_n=sigma_n)
        elif init == "w":
            u = w.clone()
        else:
            raise ValueError(f"Unknown init='{init}' for dual ULA corrector (expected 'prox' or 'w').")

        u0 = u.clone()
        h = float(step_size)
        sqrt_2h = math.sqrt(2.0 * h) * float(noise_scale)

        for _ in range(int(num_steps)):
            if mode == "phase_retrieval":
                grad_f = self._grad_f_phase_retrieval(u=u, y_amp=measurement, sigma_n=sigma_n, eps=eps)
                drift = grad_f + sigma_dual * (u - w)
                noise = (torch.randn_like(u.real) + 1j * torch.randn_like(u.real)) / math.sqrt(2.0)
                u = u - h * drift + sqrt_2h * noise
            else:
                grad_f = self._grad_f_linear_mse(u=u, y=measurement, sigma_n=sigma_n)
                drift = grad_f + sigma_dual * (u - w)
                u = u - h * drift + sqrt_2h * torch.randn_like(u)

        # diagnostics
        with torch.no_grad():
            uf = (u - u0).reshape(u.shape[0], -1)
            n = uf.shape[1]
            ula_u_move = float(uf.norm(dim=1).mean().detach() / math.sqrt(n))

        p_new = sigma_dual * (w - u)
        return p_new, ula_u_move

    # -------------------------
    # Moreau-envelope likelihood gradient helpers (for x-space corrector, optional)
    # -------------------------
    @staticmethod
    def _moreau_grad_u_linear_mse(u: torch.Tensor, y: torch.Tensor, gamma: float, sigma_n: float) -> torch.Tensor:
        """∇ e_γ f_y(u) for f_y(u)=1/(2σ_n^2)||u-y||^2."""
        return (u - y) / (sigma_n ** 2 + gamma)

    @staticmethod
    def _prox_f_phase_retrieval(u: torch.Tensor, y_amp: torch.Tensor, gamma: float, sigma_n: float,
                               eps: float = 1e-12) -> torch.Tensor:
        """prox_{γ f_y}(u) for f_y(u)=1/(2σ_n^2)|| |u| - y ||^2 (amplitude)."""
        r0 = u.abs()
        r_star = (gamma * y_amp + (sigma_n ** 2) * r0) / (gamma + sigma_n ** 2)
        return u * (r_star / (r0 + eps))

    @staticmethod
    def _moreau_grad_u_phase_retrieval(u: torch.Tensor, y_amp: torch.Tensor, gamma: float, sigma_n: float,
                                      eps: float = 1e-12) -> torch.Tensor:
        """∇ e_γ f_y(u) via (1/γ)(u - prox_{γ f_y}(u)) for amplitude likelihood."""
        prox = DRS._prox_f_phase_retrieval(u=u, y_amp=y_amp, gamma=gamma, sigma_n=sigma_n, eps=eps)
        return (u - prox) / gamma

    def _moreau_grad_x(self, operator, x_m11: torch.Tensor, measurement: torch.Tensor,
                       mode: str, gamma: float, sigma_n: float, eps: float = 1e-12) -> torch.Tensor:
        """Return K^T ∇ e_γ f_y(Kx) in x-space (x in [-1,1])."""
        if mode == "phase_retrieval":
            H, W = x_m11.shape[-2], x_m11.shape[-1]
            u = operator.forward_complex(self._to_01(x_m11).clamp(0.0, 1.0))
            grad_u = self._moreau_grad_u_phase_retrieval(u=u, y_amp=measurement, gamma=gamma, sigma_n=sigma_n, eps=eps)
            kstar_grad_u_x01 = operator.adjoint_complex(grad_u, out_hw=(H, W))
            return 0.5 * kstar_grad_u_x01

        u = operator(x_m11)
        grad_u = self._moreau_grad_u_linear_mse(u=u, y=measurement, gamma=gamma, sigma_n=sigma_n)
        return self._AT_autograd(operator, x_m11, grad_u)

    # -------------------------
    # Denoiser (diffusion prior)
    # -------------------------
    def optimize_denoising(self, z_in, model, d_k, sigma,
                           operator=None, measurement=None, mode: str | None = None, sigma_n: float | None = None,
                           prior_use_type="denoise", wandb=False):
        denoise_config = self.admm_config.denoise
        with torch.no_grad():
            noisy_im = z_in.clone()

            if prior_use_type not in ["denoise"]:
                raise Exception(f"Prior type {prior_use_type} not supported!!!")

            ac_noise = bool(getattr(denoise_config, "ac_noise", True))
            if ac_noise and sigma > 0:
                forward_z = noisy_im + torch.randn_like(noisy_im) * sigma
            else:
                forward_z = noisy_im

            lgvd_z = forward_z.clone()

            # Langevin-in-denoiser parameters
            lr = float(getattr(denoise_config.lgvd, "lr", 0.0)) * float(sigma)
            lr = 1e-4
            num_steps = int(getattr(denoise_config.lgvd, "num_steps", 0))
            reg_factor = float(getattr(denoise_config.lgvd, "reg_factor", 0.0))
            drift_clip = float(getattr(denoise_config.lgvd, "drift_clip", 10.0))
            noise_scale = float(getattr(denoise_config.lgvd, "noise_scale", 1.0))

            corrector_type = str(getattr(denoise_config.lgvd, "corrector_type", "prox_likelihood"))
            clamp_each = bool(getattr(denoise_config.lgvd, "clamp_each_step", False))
            like_weight = float(getattr(denoise_config.lgvd, "likelihood_weight", 1.0))
            gamma0 = float(getattr(denoise_config.lgvd, "gamma", 1.0))
            gamma_scale_with_sigma = bool(getattr(denoise_config.lgvd, "gamma_scale_with_sigma", False))
            gamma0 = 1/self.sigma_dual
            print(lr)

            for _ in range(num_steps):
                score_val = model.score(lgvd_z, sigma)

                if corrector_type == "prox_likelihood":
                    if operator is None or measurement is None or mode is None or sigma_n is None:
                        raise RuntimeError(
                            "prox_likelihood corrector requires operator, measurement, mode, and sigma_n. "
                            "Pass these from PDHG.sample()."
                        )
                    gamma = gamma0 * (sigma ** 2) if gamma_scale_with_sigma else gamma0
                    gamma = float(max(gamma, 1e-12))
                    grad_like = self._moreau_grad_x(
                        operator=operator,
                        x_m11=lgvd_z,
                        measurement=measurement,
                        mode=mode,
                        gamma=gamma,
                        sigma_n=float(sigma_n),
                        eps=1e-12,
                    )
                    s_hat = score_val - like_weight * grad_like
                    lgvd_z = lgvd_z + lr * s_hat
                    lgvd_z = lgvd_z + noise_scale * math.sqrt(2.0 * lr) * torch.randn_like(noisy_im)
                else:
                    # Legacy heuristic drift toward forward_z (+ optional d_k)
                    diff_val = (forward_z - lgvd_z + d_k)
                    drift = lr * min(float(sigma) * reg_factor, drift_clip) * diff_val
                    lgvd_z = lgvd_z + lr * score_val + drift + noise_scale * math.sqrt(2.0 * lr) * torch.randn_like(noisy_im)

                if clamp_each:
                    lgvd_z = self._proj(lgvd_z)

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
        """mean_i ||x_i|| / sqrt(d_per_sample) for x: [B,C,H,W]"""
        with torch.no_grad():
            xf = x.detach().flatten(1)
            d = xf.shape[1]
            return float(xf.norm(dim=1).mean().detach() / math.sqrt(d))

    @staticmethod
    def _complex_norm_mean(p: torch.Tensor) -> float:
        """mean_i ||p_i|| / sqrt(n_per_sample) for complex tensor p."""
        with torch.no_grad():
            pf = p.reshape(p.shape[0], -1)
            n = pf.shape[1]
            return float((pf.abs().pow(2).sum(dim=1).sqrt().mean().detach()) / math.sqrt(n))

    @staticmethod
    def _complex_step_norm_mean(p_new: torch.Tensor, p_old: torch.Tensor) -> float:
        with torch.no_grad():
            df = (p_new - p_old).reshape(p_new.shape[0], -1)
            n = df.shape[1]
            return float((df.abs().pow(2).sum(dim=1).sqrt().mean().detach()) / math.sqrt(n))

    @staticmethod
    def _amp_resid(operator, x_m11: torch.Tensor, y_amp: torch.Tensor) -> float:
        """|| |K(x01)| - y || / sqrt(n)"""
        with torch.no_grad():
            U = operator.forward_complex(DRS._to_01(x_m11))
            r = (U.abs() - y_amp).reshape(U.shape[0], -1)
            n = r.shape[1]
            return float(r.norm(dim=1).mean().detach() / math.sqrt(n))

    @staticmethod
    def _misalign_to_gt(x: torch.Tensor, gt: torch.Tensor, sigma: float, eps: float = 1e-12) -> float:
        """mean_i ||x_i - gt_i|| / (sigma * sqrt(d))"""
        with torch.no_grad():
            df = (x - gt).detach().flatten(1)
            d = df.shape[1]
            return float(df.norm(dim=1).mean().detach() / (max(float(sigma), eps) * math.sqrt(d)))

    # -------------------------
    # Operator norm estimation
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
            "ula_u_move",
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

        # init (same pattern as other samplers in this repo)
        x_k, z_k, y_k = self.get_start(ref_img)
        x_k = y_k
        x_bar = x_k.clone()

        sigma_n = float(getattr(operator, "sigma", 0.05))

        # dual init
        if mode == "phase_retrieval":
            with torch.no_grad():
                u0 = operator.forward_complex(self._to_01(x_bar).clamp(0.0, 1.0))
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

        # dual corrector configuration
        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
        dual_corr_cfg = getattr(pdhg_cfg, "dual_corrector", None) if pdhg_cfg is not None else None
        dual_corr_type = str(getattr(dual_corr_cfg, "type", "exact_prox")) if dual_corr_cfg is not None else "exact_prox"
        dual_ula_steps = int(getattr(dual_corr_cfg, "num_steps", 0)) if dual_corr_cfg is not None else 0
        dual_ula_h = float(getattr(dual_corr_cfg, "step_size", 0.0)) if dual_corr_cfg is not None else 0.0
        dual_ula_init = str(getattr(dual_corr_cfg, "init", "prox")) if dual_corr_cfg is not None else "prox"
        dual_ula_noise_scale = float(getattr(dual_corr_cfg, "noise_scale", 1.0)) if dual_corr_cfg is not None else 1.0

        # If requested but no step size given, use a conservative default for linear MSE.
        # (The theorem in chat explains how to pick h using κ and L.)
        if dual_corr_type.lower() in {"ula_u", "ula", "pdhg_ula_u"} and dual_ula_steps > 0 and dual_ula_h <= 0:
            # For linear MSE: L_f = 1/σ_n^2, β=0 => κ=σ, L=σ+1/σ_n^2.
            # Use h = κ/(24 L^2).
            Lf = 1.0 / max(sigma_n ** 2, 1e-12)
            L = float(self.sigma_dual + Lf)
            kappa = float(max(self.sigma_dual, 1e-12))
            dual_ula_h = kappa / (24.0 * (L ** 2) + 1e-12)

        x_old = None
        delta_patience = 0
        delta_tol = float(getattr(self.admm_config, "delta_tol", -1.0))
        delta_pat = int(getattr(self.admm_config, "delta_patience", 0))

        recorded_iters = 0

        for step in pbar:
            # denoiser sigma schedule
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma_d = float(self.annealing_scheduler.sigma_steps[t_sigma])

            theta = 0.0 if self.force_theta_zero else float(self.theta_schedule[min(step, len(self.theta_schedule) - 1)])

            # =========================
            # (1) Dual update
            # =========================
            p_old = p_k
            prox_move = None
            prox_radial_resid = None
            ula_u_move = None

            if mode == "phase_retrieval":
                v_bar = operator.forward_complex(self._to_01(x_bar).clamp(0.0, 1.0))
            else:
                v_bar = operator(x_bar)

            if dual_corr_type.lower() in {"ula_u", "ula", "pdhg_ula_u"} and dual_ula_steps > 0:
                p_k, ula_u_move = self._dual_update_ula_u(
                    p=p_k,
                    v_bar=v_bar,
                    measurement=measurement,
                    mode=mode,
                    sigma_dual=float(self.sigma_dual),
                    sigma_n=float(sigma_n),
                    num_steps=int(dual_ula_steps),
                    step_size=float(dual_ula_h),
                    init=dual_ula_init,
                    noise_scale=float(dual_ula_noise_scale),
                    eps=eps,
                )
            else:
                # Exact prox_{σ f*}
                if mode == "phase_retrieval":
                    p_k, prox_move, prox_radial_resid = self._dual_update_phase_retrieval_diag(
                        p=p_k,
                        u_bar=v_bar,
                        y_amp=measurement,
                        sigma_dual=float(self.sigma_dual),
                        sigma_n=float(sigma_n),
                        eps=eps,
                    )
                else:
                    p_k = self._dual_update_linear_mse(
                        p=p_k,
                        Ax_bar=v_bar,
                        y=measurement,
                        sigma_dual=float(self.sigma_dual),
                        sigma_n=float(sigma_n),
                    )

            # dual stats
            if mode == "phase_retrieval":
                p_norm = self._complex_norm_mean(p_k)
                p_step_norm = self._complex_step_norm_mean(p_k, p_old)
            else:
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

            z_k = x_k - float(self.tau) * ATp

            # PDHG analogue of ADMM dual-injection diagnostic
            dual_inject = float(self.tau) * ATp
            dual_inject_norm = self._img_norm_mean(dual_inject)
            dual_inject_over_sigma = dual_inject_norm / max(float(sigma_d), eps)

            # data-consistency residuals (phase retrieval only)
            amp_resid_z = None
            amp_resid_x = None
            if mode == "phase_retrieval":
                amp_resid_z = self._amp_resid(operator, z_k, measurement)

            # =========================
            # (3) Denoise -> x_new
            # =========================
            x_new = self.optimize_denoising(
                z_in=z_k,
                model=model,
                d_k=torch.zeros_like(z_k),
                sigma=sigma_d,
                operator=operator,
                measurement=measurement,
                mode=mode,
                sigma_n=sigma_n,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb,
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

                if ula_u_move is not None:
                    self.trajectory.add_value('ula_u_move', ula_u_move)

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

                if ula_u_move is not None:
                    self._trace_add_value("ula_u_move", ula_u_move)

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
                    if ula_u_move is not None:
                        logd["ula_u_move"] = float(ula_u_move)
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
    # Init
    # -------------------------
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
