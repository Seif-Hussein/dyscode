import time
import math
import torch
import wandb as wnb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy.special import lambertw

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


class PDHG(nn.Module):
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

        pdhg_cfg = getattr(self.admm_config, "pdhg", None)
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

    @staticmethod
    def _param_vector(value, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        if torch.is_tensor(value):
            vec = value.to(device=device, dtype=dtype).reshape(-1)
        else:
            vec = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)

        if vec.numel() == 1:
            vec = vec.expand(batch_size)
        elif vec.numel() != batch_size:
            raise ValueError(f"Expected 1 or {batch_size} values, got {vec.numel()}")
        return vec

    @classmethod
    def _param_view(cls, value, target: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        vec = cls._param_vector(value, target.shape[0], target.device, dtype=dtype)
        return vec.reshape((-1,) + (1,) * (target.dim() - 1))

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

        if name == "transmission_ct":
            if not torch.is_tensor(measurement):
                raise RuntimeError("transmission_ct measurement must be a tensor of Poisson counts.")
            return "transmission_ct"

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
        sigma_dual_view = self._param_view(sigma_dual, Ax_bar, dtype=Ax_bar.dtype)
        sigma_n_view = self._param_view(sigma_n, Ax_bar, dtype=Ax_bar.dtype)
        q = p + sigma_dual_view * Ax_bar
        denom = 1.0 + sigma_dual_view * (sigma_n_view ** 2)
        return (q - sigma_dual * y) / denom

    def _dual_update_phase_retrieval(self, p: torch.Tensor, u_bar: torch.Tensor, y_amp: torch.Tensor,
                                     sigma_dual: float, sigma_n: float, eps: float = 1e-12):
        sigma_dual_view = self._param_view(sigma_dual, u_bar, dtype=u_bar.real.dtype)
        sigma_n_view = self._param_view(sigma_n, u_bar, dtype=u_bar.real.dtype)
        q = p + sigma_dual_view * u_bar
        w = q / sigma_dual_view

        r0 = w.abs()
        a = sigma_dual_view * (sigma_n_view ** 2)
        r_star = (a * r0 + y_amp) / (a + 1.0)

        w_prox = w * (r_star / (r0 + eps))
        return q - sigma_dual_view * w_prox

    def _dual_update_phase_retrieval_diag(self, p: torch.Tensor, u_bar: torch.Tensor, y_amp: torch.Tensor,
                                          sigma_dual: float, sigma_n: float, eps: float = 1e-12):
        """
        Same update as _dual_update_phase_retrieval, but returns diagnostics:
          - prox_move: ||w_prox - w|| / sqrt(n)
          - prox_radial_resid: mean(| |w_prox| - r_star |)  (should be ~0)
        """
        sigma_dual_view = self._param_view(sigma_dual, u_bar, dtype=u_bar.real.dtype)
        sigma_n_view = self._param_view(sigma_n, u_bar, dtype=u_bar.real.dtype)
        q = p + sigma_dual_view * u_bar
        w = q / sigma_dual_view

        r0 = w.abs()
        a = sigma_dual_view * (sigma_n_view ** 2)
        r_star = (a * r0 + y_amp) / (a + 1.0)

        w_prox = w * (r_star / (r0 + eps))
        p_new = q - sigma_dual_view * w_prox

        # diagnostics
        w_flat = w.reshape(w.shape[0], -1) if w.dim() >= 3 else w.flatten().unsqueeze(0)
        wp_flat = w_prox.reshape(w_prox.shape[0], -1) if w_prox.dim() >= 3 else w_prox.flatten().unsqueeze(0)
        n_per = wp_flat.shape[1]

        prox_move = float((wp_flat - w_flat).abs().pow(2).sum(dim=1).sqrt().mean().detach() / math.sqrt(n_per))
        prox_radial_resid = float((w_prox.abs() - r_star).abs().mean().detach())

        return p_new, prox_move, prox_radial_resid

    @staticmethod
    def _lambertw_principal(x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.detach().to(device='cpu', dtype=torch.float64).numpy()
        w_cpu = lambertw(x_cpu, k=0).real
        return torch.from_numpy(w_cpu).to(device=x.device, dtype=x.dtype)

    def _prox_transmission_ct(self, w: torch.Tensor, y_counts: torch.Tensor,
                              sigma_dual: float, operator, eps: float = 1e-12) -> torch.Tensor:
        sigma_view = self._param_view(sigma_dual, w, dtype=w.dtype)
        eta = float(getattr(operator, "eta", 1.0))
        eta_tensor = torch.as_tensor(eta, device=w.device, dtype=w.dtype)
        i0 = operator.incident_counts(w).to(device=w.device, dtype=w.dtype)
        arg = (eta_tensor * i0 / sigma_view.clamp_min(eps)) * torch.exp(
            (-w + (eta_tensor * y_counts) / sigma_view).clamp(min=-80.0, max=80.0)
        )
        arg = arg.clamp_min(0.0)
        return w - (eta_tensor * y_counts) / sigma_view + self._lambertw_principal(arg)

    def _dual_update_transmission_ct_diag(self, p: torch.Tensor, z_bar: torch.Tensor, y_counts: torch.Tensor,
                                          sigma_dual: float, operator, eps: float = 1e-12):
        sigma_view = self._param_view(sigma_dual, z_bar, dtype=z_bar.dtype)
        q = p + sigma_view * z_bar
        w = q / sigma_view
        z_prox = self._prox_transmission_ct(w=w, y_counts=y_counts, sigma_dual=sigma_dual, operator=operator, eps=eps)
        p_new = q - sigma_view * z_prox

        wf = w.reshape(w.shape[0], -1)
        zf = z_prox.reshape(z_prox.shape[0], -1)
        n_per = zf.shape[1]
        prox_move = float((zf - wf).norm(dim=1).mean().detach() / math.sqrt(n_per))

        eta = float(getattr(operator, "eta", 1.0))
        i0 = operator.incident_counts(z_prox).to(device=z_prox.device, dtype=z_prox.dtype)
        residual = sigma_view * (z_prox - w) + eta * (y_counts - i0 * torch.exp(-z_prox))
        prox_opt_resid = float(residual.reshape(residual.shape[0], -1).norm(dim=1).mean().detach() / math.sqrt(n_per))
        return p_new, prox_move, prox_opt_resid
    
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
            sigma_batch = self._param_vector(sigma, noisy_im.shape[0], noisy_im.device, dtype=noisy_im.dtype)
            sigma_view = sigma_batch.reshape((-1,) + (1,) * (noisy_im.dim() - 1))

            if prior_use_type not in ["denoise"]:
                raise Exception(f"Prior type {prior_use_type} not supported!!!")

            ac_noise = bool(getattr(denoise_config, "ac_noise", True))
            if ac_noise:
                forward_z = noisy_im + torch.randn_like(noisy_im) * sigma_view
            else:
                forward_z = noisy_im

            lgvd_z = forward_z.clone()
            lr = denoise_config.lgvd.lr * sigma_view

            num_steps = int(getattr(denoise_config.lgvd, "num_steps", 0))
            reg_factor = float(getattr(denoise_config.lgvd, "reg_factor", 0.0))
            drift_clip = float(getattr(denoise_config.lgvd, "drift_clip", 10.0))
            noise_scale = float(getattr(denoise_config.lgvd, "noise_scale", 1.0))
            drift_scale = torch.clamp(sigma_batch * reg_factor, max=drift_clip).reshape(
                (-1,) + (1,) * (noisy_im.dim() - 1)
            )

            for _ in range(num_steps):
                score_val = model.score(lgvd_z, sigma_batch)
                diff_val = (forward_z - lgvd_z + d_k)
                drift = lr * drift_scale * diff_val
                lgvd_z += lr * score_val + drift + noise_scale * (2 * lr) ** 0.5 * torch.randn_like(noisy_im)

            if denoise_config.final_step == 'tweedie':
                z = model.tweedie(lgvd_z, sigma_batch)
            elif denoise_config.final_step == 'ode':
                if sigma_batch.numel() != 1 and not torch.allclose(sigma_batch, sigma_batch[0].expand_as(sigma_batch)):
                    raise NotImplementedError("Per-sample sigma schedules are only supported for final_step='tweedie'.")
                diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionSampler(diffusion_scheduler)
                z = sampler.sample(model, lgvd_z, SDE=False, verbose=False)
            elif denoise_config.final_step == 'map_gd':
                gd_cfg = getattr(denoise_config, "map_gd", None)
                T = int(getattr(gd_cfg, "num_iters", 5)) if gd_cfg is not None else 5
                clamp_each = bool(getattr(gd_cfg, "clamp_each", True)) if gd_cfg is not None else True

                # step size: good default is proportional to sigma^2
                alpha = float(getattr(gd_cfg, "alpha", 0.25)) if gd_cfg is not None else 0.25
                eta = alpha * (sigma ** 2)

                # quadratic-tether center (same v as your MAP interpretation)
                v = forward_z

                # warm start
                x = lgvd_z.clone()

                for _ in range(T):
                    s = model.score(x, sigma)                         # ≈ ∇ log p_sigma(x)
                    grad = s - (x - v) / (sigma ** 2 + 1e-12)        # gradient of log-posterior
                    x = x + eta * grad

                    if clamp_each:
                        x = x.clamp(self.proj_min, self.proj_max)

                z = x
            else:
                raise Exception(f"Final step {denoise_config.final_step} not supported!!!")

            return z

    def sample_hparam_candidates(self, model, ref_img, operator, measurement,
                                 sigma_schedule_by_candidate: torch.Tensor,
                                 tau_by_candidate: torch.Tensor,
                                 sigma_dual_by_candidate: torch.Tensor,
                                 start_triplet: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                                 progress_callback=None,
                                 progress_every: int | None = None):
        """
        Evaluate multiple PDHG hyperparameter candidates in one batched run.

        This path is intentionally lightweight for tuning:
        - no trajectory recording
        - no wandb logging
        - intended for final_step='tweedie' where per-sample sigma is supported

        Args:
            sigma_schedule_by_candidate: [C, K] tensor of denoiser sigmas.
            tau_by_candidate: [C] tensor.
            sigma_dual_by_candidate: [C] tensor.
            start_triplet: Optional `(x0, z0, y0)` tuple generated from `get_start(ref_img)`.
                When provided, the same initialization is reused across candidate chunks.
            progress_callback: Optional callable invoked during the PDHG loop with keyword args
                `step`, `num_steps`, and `sigma_mean`.
            progress_every: How often to invoke `progress_callback` in iterations. Defaults to 0/disabled.

        Returns:
            Tensor of shape [C, B, C_img, H, W].
        """
        mode = self._mode(operator, measurement)
        if mode not in {"phase_retrieval", "transmission_ct"}:
            raise NotImplementedError(
                "Batched candidate evaluation is currently implemented for "
                "phase_retrieval and transmission_ct only."
            )

        if self.admm_config.denoise.final_step != "tweedie":
            raise NotImplementedError("Batched candidate evaluation currently requires final_step='tweedie'.")

        num_candidates, max_iter = sigma_schedule_by_candidate.shape
        if max_iter < int(self.admm_config.max_iter):
            raise ValueError("sigma_schedule_by_candidate must have at least max_iter columns.")

        batch_size = ref_img.shape[0]
        total_batch = num_candidates * batch_size
        device = ref_img.device

        tau_batch = self._param_vector(tau_by_candidate, num_candidates, device=device, dtype=ref_img.dtype)
        sigma_dual_batch = self._param_vector(
            sigma_dual_by_candidate, num_candidates, device=device, dtype=ref_img.dtype
        )
        sigma_n_batch = torch.full((num_candidates,), float(getattr(operator, "sigma", 0.05)),
                                   device=device, dtype=ref_img.dtype)

        ref_rep = ref_img.repeat((num_candidates, 1, 1, 1))
        if not torch.is_tensor(measurement):
            raise NotImplementedError("Batched candidate evaluation expects tensor measurements.")
        measurement_rep = measurement.repeat((num_candidates, 1, 1, 1))

        if mode == "transmission_ct":
            # Mirror the main PDHG path: zero attenuation corresponds to x = -1
            # under the operator's internal mu(x) mapping.
            x_k = -torch.ones_like(ref_rep)
            z_k = -torch.ones_like(ref_rep)
            y_k = -torch.ones_like(ref_rep)
            x_bar = x_k.clone()
            with torch.no_grad():
                p_k = torch.zeros_like(operator(x_bar))
        else:
            if start_triplet is None:
                base_x0, base_z0, base_y0 = self.get_start(ref_img)
            else:
                if len(start_triplet) != 3:
                    raise ValueError("start_triplet must contain exactly three tensors: (x0, z0, y0).")
                base_x0, base_z0, base_y0 = [
                    tensor.to(device=device, dtype=ref_img.dtype) for tensor in start_triplet
                ]

            x_k = base_y0.repeat((num_candidates, 1, 1, 1))
            z_k = base_z0.repeat((num_candidates, 1, 1, 1))
            y_k = base_y0.repeat((num_candidates, 1, 1, 1))
            x_bar = x_k.clone()

            with torch.no_grad():
                p_k = torch.zeros_like(operator.forward_complex(self._to_01(x_bar)))

        tau_batch_expanded = tau_batch.repeat_interleave(batch_size)
        sigma_dual_batch_expanded = sigma_dual_batch.repeat_interleave(batch_size)
        sigma_n_batch_expanded = sigma_n_batch.repeat_interleave(batch_size)
        theta_schedule = self.theta_schedule
        progress_every = int(progress_every or 0)
        total_steps = int(self.admm_config.max_iter)

        for step in range(total_steps):
            sigma_step = sigma_schedule_by_candidate[:, step].to(device=device, dtype=ref_img.dtype)
            sigma_step_expanded = sigma_step.repeat_interleave(batch_size)
            theta = 0.0 if self.force_theta_zero else float(theta_schedule[min(step, len(theta_schedule) - 1)])

            if mode == "phase_retrieval":
                u_bar = operator.forward_complex(self._to_01(x_bar))
                p_k = self._dual_update_phase_retrieval(
                    p=p_k,
                    u_bar=u_bar,
                    y_amp=measurement_rep,
                    sigma_dual=sigma_dual_batch_expanded,
                    sigma_n=sigma_n_batch_expanded,
                )

                kstar_p_x01 = operator.adjoint_complex(p_k, out_hw=(x_k.shape[-2], x_k.shape[-1]))
                at_p = 0.5 * kstar_p_x01
            else:
                z_bar = operator(x_bar)
                sigma_dual_view = self._param_view(
                    sigma_dual_batch_expanded, z_bar, dtype=z_bar.dtype
                )
                q = p_k + sigma_dual_view * z_bar
                w = q / sigma_dual_view
                z_prox = self._prox_transmission_ct(
                    w=w,
                    y_counts=measurement_rep,
                    sigma_dual=sigma_dual_batch_expanded,
                    operator=operator,
                )
                p_k = q - sigma_dual_view * z_prox
                at_p = self._AT_autograd(operator, x_k, p_k)

            tau_view = self._param_view(tau_batch_expanded, x_k, dtype=x_k.dtype)
            z_k = x_k - tau_view * at_p

            x_new = self.optimize_denoising(
                z_in=z_k,
                model=model,
                d_k=torch.zeros_like(z_k),
                sigma=sigma_step_expanded,
                prior_use_type=self.admm_config.denoise.type,
                wandb=False,
            )
            x_new = self._proj(x_new)

            x_bar = x_new + theta * (x_new - x_k)
            x_k = x_new
            y_k = x_k

            should_report = (
                progress_callback is not None and (
                    step == 0
                    or step == total_steps - 1
                    or (progress_every > 0 and (step + 1) % progress_every == 0)
                )
            )
            if should_report:
                progress_callback(
                    step=step + 1,
                    num_steps=total_steps,
                    sigma_mean=float(sigma_step.mean().detach().cpu()),
                )

        return x_k.view(num_candidates, batch_size, *x_k.shape[1:])

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
            U = operator.forward_complex(PDHG._to_01(x_m11))
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
        
    @staticmethod
    def _f_value(mode: str, u, y_meas, sigma_n: float, operator=None) -> float:
        # returns mean over batch of the *true* paper fidelity value
        #   linear:  (1/(2 sigma_n^2)) ||u - y||^2
        #   phase:   (1/(2 sigma_n^2)) || |u| - y ||^2
        with torch.no_grad():
            if mode == "phase_retrieval":
                r = (u.abs() - y_meas).reshape(u.shape[0], -1)
                return float(0.5 / (sigma_n**2) * r.pow(2).sum(dim=1).mean().detach())
            if mode == "transmission_ct":
                eta = float(getattr(operator, "eta", 1.0))
                i0 = operator.incident_counts(u).to(device=u.device, dtype=u.dtype)
                val = eta * (i0 * torch.exp(-u) + y_meas * u)
                return float(val.reshape(u.shape[0], -1).sum(dim=1).mean().detach())
            else:
                r = (u - y_meas).reshape(u.shape[0], -1)
                return float(0.5 / (sigma_n**2) * r.pow(2).sum(dim=1).mean().detach())

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

        sigma_n = float(getattr(operator, "sigma", 0.05))
        # init
        if mode == "transmission_ct":
            x_k = -torch.ones_like(ref_img)
            z_k = -torch.ones_like(ref_img)
            y_k = -torch.ones_like(ref_img)
        else:
            x_k, z_k, y_k = self.get_start(ref_img)
            x_k = y_k
        x_bar = x_k.clone()

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
        use_stability_coupling = False
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
            """if sigma_d > 5:
                self.tau = sigma_d**2*0.001
            else:
                #self.tau = sigma_d**2*0.1
                self.tau = 0.001*(float(self.annealing_scheduler.sigma_steps[t_sigma-1])**2)/(float(self.annealing_scheduler.sigma_steps[t_sigma])**2)
            tau_k = self.tau"""
            #self.tau = min(0.01,sigma_d**2)
            #self.sigma_dual = 1.01**step
            #self.tau = 1/self.sigma_dual
            #sigma_d = 1/math.sqrt(self.sigma_dual)
            tau_k = self.tau
            """rho_k = 0.001/sigma_d**2
            self.tau = rho_k*0.01
            self.sigma_dual = rho_k*100000
            tau_k = self.tau
            print('\n'+str(self.sigma_dual))"""
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
            elif mode == "transmission_ct":
                z_bar = operator(x_bar)
                p_new, prox_move, prox_radial_resid = self._dual_update_transmission_ct_diag(
                    p=p_k,
                    z_bar=z_bar,
                    y_counts=measurement,
                    sigma_dual=self.sigma_dual,
                    operator=operator,
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
            # v_k in the paper corresponds to w here:
            if mode == "phase_retrieval":
                w = (p_old + self.sigma_dual * u_bar) / self.sigma_dual
            elif mode == "transmission_ct":
                w = (p_old + self.sigma_dual * z_bar) / self.sigma_dual
            else:
                w = (p_old + self.sigma_dual * Ax_bar) / self.sigma_dual

            f_vk = self._f_value(mode, w, measurement, sigma_n, operator=operator)

            # also get the proximal point u_{k+1} (optional, but useful)
            u_k1 = w - p_k / self.sigma_dual   # p_k is already p_new at this point
            f_uk1 = self._f_value(mode, u_k1, measurement, sigma_n, operator=operator)

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
            #z_k = x_k - self.tau * ATp

            # =========================
            # Diagnostics: dual injection into denoiser input
            # =========================
            # This is the PDHG analogue of ADMM's "dual_over_sigma":
            # how large is the (tau*A^T p) shift, in units of sigma_d.
            dual_inject = tau_k * ATp
            # --- adaptive tau cap to enforce du/sigma <= alpha_max ---
            ATp_norm = self._img_norm_mean(ATp)                 # ||ATp||/sqrt(d)
            """ tau_cap = alpha_max * sigma_d / (ATp_norm + eps)    # ensures ||tau*ATp||/sigma <= alpha_max
            tau_k = min(tau0, tau_cap)                          # never exceed baseline tau0
            print('\n--------'+str(tau_k))"""

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
            #print('\n'+str(sigma_d))
            #print('\n'+str(dual_inject_norm))

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
            if mode == "phase_retrieval":
                Kxk = operator.forward_complex(self._to_01(x_k))
            else:
                Kxk = operator(x_k)

            f_Kxk = self._f_value(mode, Kxk, measurement, sigma_n, operator=operator)
            dual_inject_norm = tau_k * math.sqrt(f_vk) / sigma_d
            dual_inject_over_sigma = tau_k * math.sqrt(f_Kxk) / sigma_d
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
                self.trajectory.add_value('tau', float(tau_k))
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
                self._trace_add_value("tau", float(tau_k))
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
                        "tau": float(tau_k),
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
