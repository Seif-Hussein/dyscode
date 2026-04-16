import time
import math
import torch
import numpy as np
import torch.nn as nn
import tqdm
import wandb as wnb

from utils.diffusion import Scheduler
from utils.logging import Trajectory
from .registry import register_sampler


@register_sampler(name='dps')
def get_sampler(**kwargs):
    """Factory wrapper to match the ADMM/DYS callable conditions."""
    latent = kwargs.get('latent', False)
    kwargs.pop('latent', None)
    if latent:
        raise NotImplementedError("Latent-space DPS not implemented.")
    return DPS(**kwargs)


class DPS(nn.Module):
    """Diffusion Posterior Sampling (DPS) baseline.

    Paper:
      Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems"
      (ICLR 2023).

    This sampler is implemented to run inside the same Hydra pipeline as your
    existing samplers (ADMM/DYS/RED-diff):

        sampler.sample(model, ref_img, operator, measurement,
                       evaluator=None, record=False, verbose=False, wandb=False, **kwargs)

    Key implementation choices (aligned with the DPS paper):
      - Predictor step: one reverse diffusion Euler(-Maruyama) step using the *prior* score.
      - Likelihood step: a gradient step of ||y - A(x_hat0(x_i))||^2 w.r.t. x_i.
      - Step-size schedule: default is the paper's "normalized residual" schedule
          zeta_i = zeta_base / ||y - A(x_hat0)||,
        with zeta_base defaulting to 1.0 for FFHQ Gaussian linear inverse problems
        (Appendix D of the DPS paper).

    Assumptions (consistent with your codebase):
      - model.score(x, sigma) exists and is differentiable.
      - operator(x) is differentiable in torch.
      - measurement can be a torch.Tensor or a tuple/list of tensors.

    Notes:
      - DPS *does* require backpropagation through the score model via
        the gradient of the likelihood term with respect to x_i.
      - We use your existing sigma-scheduler (utils/diffusion.Scheduler) to
        define the noise levels and Euler step sizes.
    """

    def __init__(
        self,
        annealing_scheduler_config,
        diffusion_scheduler_config,  # accepted for compatibility; unused
        lgvd_config,                 # accepted for compatibility; unused
        admm_config,
        device='cuda',
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.admm_config = admm_config

        # Noise / time schedule for diffusion sampling
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)

        # DPS-specific config (optional): inverse_task.admm_config.dps
        self.dps_cfg = getattr(self.admm_config, 'dps', None)

        # Whether to run the stochastic reverse SDE step (Euler-Maruyama) or
        # deterministic probability-flow ODE (Euler).
        self.use_sde = bool(getattr(self.dps_cfg, 'SDE', True)) if self.dps_cfg is not None else True

        # Step-size schedule for the likelihood gradient step.
        # Default (DPS paper, Appendix D for FFHQ Gaussian linear tasks): zeta_base=1.0
        self.zeta_base = float(getattr(self.dps_cfg, 'zeta_base', 1.0)) if self.dps_cfg is not None else 1.0
        self.zeta_mode = str(getattr(self.dps_cfg, 'zeta_mode', 'residual_norm')) if self.dps_cfg is not None else 'residual_norm'
        self.zeta_mode = self.zeta_mode.lower()
        if self.zeta_mode not in ['residual_norm', 'constant']:
            raise ValueError("dps.zeta_mode must be one of: ['residual_norm', 'constant']")

        self.zeta_eps = float(getattr(self.dps_cfg, 'zeta_eps', 1e-8)) if self.dps_cfg is not None else 1e-8
        self.zeta_min = float(getattr(self.dps_cfg, 'zeta_min', 0.0)) if self.dps_cfg is not None else 0.0
        self.zeta_max = float(getattr(self.dps_cfg, 'zeta_max', 1e6)) if self.dps_cfg is not None else 1e6
        self.zeta_min = 0.6
        self.zeta_max = 0.6

        # How to compute x_hat0(x_i).
        # - 'tweedie_from_score': x0_hat = x + sigma^2 * score(x, sigma)
        # - 'model': x0_hat = model.tweedie(x, sigma)  (requires differentiable tweedie)
        self.x0hat_mode = str(getattr(self.dps_cfg, 'x0hat_mode', 'tweedie_from_score')) if self.dps_cfg is not None else 'tweedie_from_score'
        self.x0hat_mode = self.x0hat_mode.lower()
        if self.x0hat_mode not in ['tweedie_from_score', 'model']:
            raise ValueError("dps.x0hat_mode must be one of: ['tweedie_from_score', 'model']")

        # Optional clamping (OFF by default to stay closer to the original DPS algorithm)
        self.clamp_x = bool(getattr(self.dps_cfg, 'clamp_x', False)) if self.dps_cfg is not None else False
        self.clamp_x0 = bool(getattr(self.dps_cfg, 'clamp_x0', False)) if self.dps_cfg is not None else False
        self.clamp_min = float(getattr(self.dps_cfg, 'clamp_min', -1.0)) if self.dps_cfg is not None else -1.0
        self.clamp_max = float(getattr(self.dps_cfg, 'clamp_max',  1.0)) if self.dps_cfg is not None else 1.0

        # Early stopping (reuse existing inverse_task.admm_config fields if present)
        self.delta_tol = float(getattr(self.admm_config, 'delta_tol', -1.0))
        self.delta_patience = int(getattr(self.admm_config, 'delta_patience', 0))

        # Internal counters
        self.nfe = 0  # number of model.score calls

    # -------------------------
    # Counting wrapper
    # -------------------------
    def _score(self, model, x, sigma):
        self.nfe += 1
        return model.score(x, sigma)

    # -------------------------
    # Core DPS helpers
    # -------------------------
    def _x0_hat(self, model, x, sigma, score=None):
        """Compute x_hat0(x, sigma) used by DPS."""
        if self.x0hat_mode == 'model':
            if not hasattr(model, 'tweedie'):
                raise AttributeError("Model has no attribute 'tweedie', but dps.x0hat_mode='model'.")
            return model.tweedie(x, sigma)

        # Default: Tweedie-from-score (works for additive Gaussian noise parameterization)
        if score is None:
            score = self._score(model, x, sigma)
        return x + (float(sigma) ** 2) * score

    def _residual_sq_and_norm(self, operator, pred, measurement):
        """Return per-sample squared residual sums and L2 norms.

        Supports both tensor and tuple/list measurements.
        """
        if isinstance(measurement, (tuple, list)):
            if not isinstance(pred, (tuple, list)):
                raise TypeError("Operator returned a tensor but measurement is tuple/list.")
            if len(pred) != len(measurement):
                raise ValueError("Operator output tuple/list length != measurement length.")

            per_sample_sq = None
            for p, y in zip(pred, measurement):
                diff = p - y
                cur = diff.flatten(1).pow(2).sum(dim=1)
                per_sample_sq = cur if per_sample_sq is None else (per_sample_sq + cur)
            per_sample_norm = (per_sample_sq + 1e-12).sqrt()
            return per_sample_sq, per_sample_norm

        # Tensor measurement
        diff = pred - measurement
        per_sample_sq = diff.flatten(1).pow(2).sum(dim=1)
        per_sample_norm = (per_sample_sq + 1e-12).sqrt()
        return per_sample_sq, per_sample_norm

    def _compute_zeta(self, res_norm: torch.Tensor) -> torch.Tensor:
        """Compute per-sample zeta_i."""
        if self.zeta_mode == 'constant':
            z = torch.full_like(res_norm, fill_value=float(self.zeta_base))
        else:
            z = float(self.zeta_base) / (res_norm + float(self.zeta_eps))
        if self.zeta_min > 0:
            z = torch.clamp(z, min=float(self.zeta_min))
        if self.zeta_max < float('inf'):
            z = torch.clamp(z, max=float(self.zeta_max))
        return z

    def _broadcast_like_x(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape per-sample vector [B] to broadcast over x [B,C,H,W]."""
        while v.ndim < x.ndim:
            v = v.view(*v.shape, 1)
        return v
    def _enable_model_grads_temporarily(self, model: torch.nn.Module):
        # Save original requires_grad flags
        self._dps_prev_reqgrad = []
        for p in model.parameters():
            self._dps_prev_reqgrad.append(p.requires_grad)
            if not p.requires_grad:
                p.requires_grad_(True)

    def _restore_model_grads(self, model: torch.nn.Module):
        if not hasattr(self, "_dps_prev_reqgrad"):
            return
        for p, rg in zip(model.parameters(), self._dps_prev_reqgrad):
            p.requires_grad_(rg)
        delattr(self, "_dps_prev_reqgrad")


    # -------------------------
    # Main sampling API
    # -------------------------
    def sample(
        self,
        model,
        ref_img,
        operator,
        measurement,
        evaluator=None,
        record=False,
        verbose=False,
        wandb=False,
        **kwargs
    ):
        #self._enable_model_grads_temporarily(model)
        if record:
            self.trajectory = Trajectory()

        self.nfe = 0
        t0 = time.time()

        # Initial state (DPS paper: x_N ~ N(0, I); here scale by sigma_max)
        sigma0 = float(self.annealing_scheduler.sigma_steps[0])
        x = (torch.randn_like(ref_img) * sigma0).to(self.device)

        # Early stopping bookkeeping
        x_old = None
        low_delta_pat = 0

        # DPS returns the last x_hat0 (paper returns x_hat0 at i=0)
        last_x0_hat = None

        K = int(self.annealing_scheduler.num_steps)
        pbar = tqdm.trange(K) if verbose else range(K)

        for step in pbar:
            sigma = float(self.annealing_scheduler.sigma_steps[step])
            factor = float(self.annealing_scheduler.factor_steps[step])
            sqrt_factor = math.sqrt(factor)

            # ----- Likelihood gradient computed at x_i
            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                score = self._score(model, x_in, sigma)
                x0_hat = self._x0_hat(model, x_in, sigma, score=score)

                # Forward prediction and squared residual (no 1/(2*sigma_y^2) scaling;
                # matches Algorithm 1 / Eq. (21) in DPS paper).
                pred = operator(x0_hat)
                per_sample_sq, per_sample_norm = self._residual_sq_and_norm(operator, pred, measurement)

                loss = per_sample_sq.sum()  # scalar
                grad = torch.autograd.grad(loss, x_in, create_graph=False, retain_graph=False)[0]

            # Step size zeta_i (default: zeta_base / ||res||)
            with torch.no_grad():
                zeta = self._compute_zeta(per_sample_norm.detach())
                #zeta = 0.4
                zeta_b = self._broadcast_like_x(zeta, x)

                # ----- Predictor step (reverse diffusion)
                if self.use_sde:
                    noise = torch.randn_like(x)
                    x_prime = x + factor * score.detach() + sqrt_factor * noise
                else:
                    # probability-flow ODE (Euler)
                    x_prime = x + factor * score.detach() * 0.5

                # ----- Likelihood gradient step (DPS)
                x_next = x_prime - zeta_b * grad.detach()

                # Optional operator-specific post-processing hook
                x_next = operator.post_ml_op(x_next, measurement)

                if self.clamp_x:
                    x_next = torch.clamp(x_next, min=self.clamp_min, max=self.clamp_max)

                x = x_next

            # Store current x_hat0 for outputs/logging
            with torch.no_grad():
                last_x0_hat = x0_hat.detach()
                if self.clamp_x0:
                    last_x0_hat = torch.clamp(last_x0_hat, min=self.clamp_min, max=self.clamp_max)

            # ----- Early stopping (delta on x)
            if step > 0 and x_old is not None and self.delta_tol > 0:
                denom = float(x.numel())
                delta = (x.detach() - x_old).norm() ** 2 / denom
                if float(delta) < self.delta_tol:
                    low_delta_pat += 1
                    if low_delta_pat > self.delta_patience:
                        if verbose:
                            print(f"DPS converged (delta<{self.delta_tol}) at step {step}.")
                        break
                else:
                    low_delta_pat = 0
            x_old = x.detach().clone()

            # ----- Evaluation
            x0_results = {}
            if evaluator is not None and 'gt' in kwargs and last_x0_hat is not None:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0_results = evaluator(gt, measurement, last_x0_hat)

                if verbose:
                    main_eval = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        f'x0_{main_eval}': f"{x0_results[main_eval].item():.2f}",
                        'sigma': sigma,
                        'zeta_mean': float(zeta.mean().cpu()),
                        'nfe': int(self.nfe),
                    })

                if wandb:
                    for fn_name, val in x0_results.items():
                        wnb.log({f"x0_{fn_name}": val.item(), "DPS/iter": step + 1})

            # ----- Logging
            if wandb:
                wnb.log({
                    "DPS/iter": step + 1,
                    "DPS/sigma": float(sigma),
                    "DPS/factor": float(factor),
                    "DPS/zeta_base": float(self.zeta_base),
                    "DPS/zeta_mean": float(zeta.mean().cpu()),
                    "DPS/res_norm_mean": float(per_sample_norm.detach().mean().cpu()),
                    "DPS/res_sq_mean": float(per_sample_sq.detach().mean().cpu()),
                    "DPS/nfe": int(self.nfe),
                    "DPS/wall_time": float(time.time() - t0),
                })

            if record and last_x0_hat is not None:
                self._record(
                    x=x,
                    x0_hat=last_x0_hat,
                    sigma=sigma,
                    zeta=zeta,
                    res_norm=per_sample_norm.detach(),
                    res_sq=per_sample_sq.detach(),
                    x0_results=x0_results,
                )

        # Return last x_hat0 (matches Algorithm 1 in DPS paper)
        if last_x0_hat is None:
            last_x0_hat = x.detach()
        return last_x0_hat

    def _record(self, x, x0_hat, sigma, zeta, res_norm, res_sq, x0_results):
        self.trajectory.add_tensor('x', x)
        self.trajectory.add_tensor('x0_hat', x0_hat)
        self.trajectory.add_value('sigma', float(sigma))
        self.trajectory.add_value('zeta_mean', float(zeta.mean().cpu()))
        self.trajectory.add_value('res_norm_mean', float(res_norm.mean().cpu()))
        self.trajectory.add_value('res_sq_mean', float(res_sq.mean().cpu()))
        self.trajectory.add_value('nfe', int(self.nfe))
        for name, val in x0_results.items():
            self.trajectory.add_value(f'x0_{name}', float(val.detach().cpu()))
