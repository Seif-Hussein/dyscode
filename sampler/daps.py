import time
import math
import torch
import wandb as wnb
import numpy as np
import torch.nn as nn
import tqdm

from utils.diffusion import Scheduler
from utils.logging import Trajectory
from .registry import register_sampler


@register_sampler(name='daps')
def get_sampler(**kwargs):
    """
    Factory to match ADMM/DYS callable conditions:
      - expects kwargs contains 'latent'
      - raises if latent=True
      - returns an nn.Module sampler instance
    """
    latent = kwargs.get('latent', False)
    kwargs.pop('latent', None)
    if latent:
        raise NotImplementedError("Latent-space DAPS not implemented in this sampler.")
    return DAPS(**kwargs)


class DAPS(nn.Module):
    """
    DAPS (Decoupled Annealing Posterior Sampling), pixel-space version.

    Paper algorithm (Algorithm 1) structure:
      - Maintain noisy state x_t along an annealing schedule {sigma_i}.
      - At each sigma_i:
          1) compute prior mean xhat0(x_t) via probability-flow ODE (few Euler steps)
          2) sample x0 via Langevin dynamics using:
                log p(x0 | x_t)  approx  -||x0 - xhat0||^2 / (2 r_t^2)
                log p(y  | x0)  (Gaussian) -||A(x0) - y||^2 / (2 beta_y^2)
          3) resample x_t_next = x0 + sigma_next * N(0,I)
      - Return final x0
    """

    def __init__(
        self,
        annealing_scheduler_config,
        diffusion_scheduler_config,
        lgvd_config,          # unused, kept for compatibility with pipeline signature
        admm_config,
        device='cuda',
        **kwargs
    ):
        super().__init__()

        self.annealing_scheduler_config, self.diffusion_scheduler_config = \
            self._check(annealing_scheduler_config, diffusion_scheduler_config)

        self.annealing_scheduler = Scheduler(**self.annealing_scheduler_config)
        self.admm_config = admm_config
        self.device = device

        # ---- DAPS-specific config lives under inverse_task.admm_config.daps
        daps_cfg = getattr(self.admm_config, "daps", None)

        # Likelihood temperature / assumed measurement noise (paper uses beta_y=0.01 as hyperparameter)
        self.beta_y = float(getattr(daps_cfg, "beta_y", 0.01)) if daps_cfg is not None else 0.01

        # Langevin step base and decay
        self.eta0 = float(getattr(daps_cfg, "eta0", 2e-5)) if daps_cfg is not None else 2e-5
        self.delta = float(getattr(daps_cfg, "delta", 1e-2)) if daps_cfg is not None else 1e-2

        # Number of Langevin steps per annealing level (paper typically uses 100)
        self.inner_steps = int(getattr(daps_cfg, "inner_steps", 100)) if daps_cfg is not None else 100

        # Prior mean computation
        #   - 'ode' is the intended pixel-DAPS mechanism (few-step prob-flow ODE)
        #   - 'tweedie' is an optional cheap alternative (not paper-default for DAPS-4K)
        self.prior_mean = str(getattr(daps_cfg, "prior_mean", "ode")) if daps_cfg is not None else "ode"

        # Data term construction
        #   - 'mse' uses ||A(x)-y||^2/(2 beta_y^2)
        #   - 'operator_loss' uses operator.loss(x, y) (useful if measurement isn't a plain tensor)
        self.data_term = str(getattr(daps_cfg, "data_term", "mse")) if daps_cfg is not None else "mse"

        # r_t heuristic: r_t = max(r_ratio * sigma, r_min)
        # (paper says "specified using heuristics", so expose knobs)
        self.r_ratio = float(getattr(daps_cfg, "r_ratio", 1.0)) if daps_cfg is not None else 1.0
        self.r_min = float(getattr(daps_cfg, "r_min", 1e-3)) if daps_cfg is not None else 1e-3

        # Optional clipping (explicit domain projection). Default True for stability in this codebase.
        self.clip = bool(getattr(daps_cfg, "clip", True)) if daps_cfg is not None else True
        self.clip_min = float(getattr(daps_cfg, "clip_min", -1.0)) if daps_cfg is not None else -1.0
        self.clip_max = float(getattr(daps_cfg, "clip_max",  1.0)) if daps_cfg is not None else  1.0
        self.clip_every_ode = bool(getattr(daps_cfg, "clip_every_ode", False)) if daps_cfg is not None else False

        # NFE counting (score calls)
        self.nfe = 0

    # -------------------------
    # Config compatibility
    # -------------------------
    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
        Keep the same convention as the other samplers:
          - diffusion_scheduler_config should NOT carry sigma_max (passed per-call)
          - annealing ends at sigma_final=0 for the outer schedule unless overridden elsewhere
        """
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        # match existing pipeline behavior
        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    # -------------------------
    # Helpers
    # -------------------------
    def _score(self, model, x, sigma: float) -> torch.Tensor:
        """Wrapper to count NFEs."""
        self.nfe += 1
        return model.score(x, sigma)

    def _eta(self, sigma: float) -> float:
        """
        Linear decay from eta0 at sigma_max down to eta0*delta at sigma=0:
            eta(sigma) = eta0 * (delta + (sigma/sigma_max)*(1-delta))
        """
        sigma_max = float(getattr(self.annealing_scheduler, "sigma_max", max(self.annealing_scheduler.sigma_steps)))
        sigma_max = max(sigma_max, 1e-12)
        return float(self.eta0 * (self.delta + (sigma / sigma_max) * (1.0 - self.delta)))

    def _rt(self, sigma: float) -> float:
        """Heuristic r_t (floor to avoid divide-by-zero)."""
        return float(max(self.r_ratio * sigma, self.r_min))

    def _ode_denoise_to_x0(self, model, x_t: torch.Tensor, sigma_max: float) -> torch.Tensor:
        """
        Few-step probability-flow ODE solve (Euler), matching utils/diffusion.DiffusionSampler(SDE=False),
        but implemented inline so we can count NFEs.
        """
        sched = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma_max)
        x = x_t
        with torch.no_grad():
            for i in range(sched.num_steps):
                sigma_i = float(sched.sigma_steps[i])
                factor = float(sched.factor_steps[i])
                s = self._score(model, x, sigma_i)
                # Probability-flow ODE Euler step (same as DiffusionSampler._euler when SDE=False)
                x = x + factor * s * 0.5
            if self.clip and self.clip_every_ode:
                x = x.clamp(self.clip_min, self.clip_max)
        return x

    def _prior_mean(self, model, x_t: torch.Tensor, sigma: float) -> torch.Tensor:
        if self.prior_mean.lower() == "ode":
            return self._ode_denoise_to_x0(model, x_t, sigma_max=sigma)
        elif self.prior_mean.lower() == "tweedie":
            # count as one score-eval equivalent
            self.nfe += 1
            if hasattr(model, "tweedie"):
                with torch.no_grad():
                    return model.tweedie(x_t, sigma)
            # fallback Tweedie: x0 ≈ x + sigma^2 * score(x,sigma)
            with torch.no_grad():
                return x_t + (sigma ** 2) * model.score(x_t, sigma)
        else:
            raise ValueError(f"Unsupported prior_mean={self.prior_mean}. Use 'ode' or 'tweedie'.")

    def _data_term_and_grad(self, x: torch.Tensor, operator, measurement):
        """
        Return (energy, grad) for the data likelihood term.
        For Gaussian likelihood:  ||A(x)-y||^2 / (2 beta_y^2)
        """
        x_in = x.detach().requires_grad_(True)

        with torch.enable_grad():
            if self.data_term == "operator_loss" or (not torch.is_tensor(measurement)):
                # Generic fallback
                energy = operator.loss(x_in, measurement)
            else:
                y_hat = operator(x_in)
                energy = ((y_hat - measurement) ** 2).sum() / (2.0 * (self.beta_y ** 2))

            grad = torch.autograd.grad(energy, x_in, create_graph=False, retain_graph=False)[0]

        return energy.detach(), grad.detach()

    # -------------------------
    # Main sampler
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
        if record:
            self.trajectory = Trajectory()

        # reset NFE counter per sample call
        self.nfe = 0
        t0 = time.time()

        # Outer iterations: use annealing schedule length (or max_iter if smaller)
        K_sched = int(self.annealing_scheduler.num_steps)
        K_user = int(getattr(self.admm_config, "max_iter", K_sched))
        K = min(K_user, K_sched)

        # Annealing sigmas: sigma_steps has length K_sched+1 (includes sigma_final at end)
        sigmas = self.annealing_scheduler.sigma_steps

        # Initialize x_T ~ N(0, sigma_max^2 I) as in Algorithm 1
        sigma_start = float(sigmas[0])
        x_t = torch.randn_like(ref_img, device=self.device) * sigma_start

        pbar = tqdm.trange(K) if verbose else range(K)

        for step in pbar:
            sigma = float(sigmas[step])
            sigma_next = float(sigmas[step + 1])

            # (1) Prior mean via ODE (or Tweedie)
            xhat0 = self._prior_mean(model, x_t, sigma)

            # (2) Langevin dynamics on x0, initialized at xhat0 (Algorithm 1)
            x0 = xhat0.detach()
            rt = self._rt(sigma)
            rt2 = rt * rt
            eta = self._eta(sigma)
            sqrt_2eta = math.sqrt(max(2.0 * eta, 0.0))

            for _ in range(self.inner_steps):
                # data term grad
                _, data_grad = self._data_term_and_grad(x0, operator, measurement)

                # prior grad from Gaussian approx: d/dx [ ||x-xhat0||^2/(2 rt^2) ] = (x-xhat0)/rt^2
                prior_grad = (x0 - xhat0) / rt2

                # ULA update: x <- x - eta*(prior_grad + data_grad) + sqrt(2 eta) * N(0,I)
                noise = torch.randn_like(x0)
                x0 = x0 - eta * (prior_grad + data_grad) + sqrt_2eta * noise

                if self.clip:
                    x0 = x0.clamp(self.clip_min, self.clip_max)

                x0 = operator.post_ml_op(x0, measurement)

            # (3) Decoupled annealing: sample next noisy state
            with torch.no_grad():
                if sigma_next > 0:
                    x_t = x0 + sigma_next * torch.randn_like(x0)
                else:
                    x_t = x0

            # --- evaluation / logging
            results = {}
            if evaluator and ('gt' in kwargs):
                with torch.no_grad():
                    gt = kwargs['gt']
                    results = evaluator(gt, measurement, x0)

            if verbose and evaluator and results:
                main = evaluator.main_eval_fn_name
                pbar.set_postfix({
                    f'{main}': f"{results[main].item():.2f}",
                    'sigma': f"{sigma:.3g}",
                    'nfe': f"{self.nfe:d}",
                })

            if wandb:
                log_dict = {
                    "DAPS Iteration": step + 1,
                    "sigma": sigma,
                    "sigma_next": sigma_next,
                    "eta": eta,
                    "beta_y": self.beta_y,
                    "rt": rt,
                    "NFE": int(self.nfe),
                    "wall_time": time.time() - t0,
                }
                for k, v in results.items():
                    log_dict[f"x0_{k}"] = float(v.item()) if torch.is_tensor(v) else v
                wnb.log(log_dict)

            if record:
                self._record(
                    x_t=x_t,
                    xhat0=xhat0,
                    x0=x0,
                    sigma=sigma,
                    results=results
                )

        return x0

    # -------------------------
    # Recording
    # -------------------------
    def _record(self, x_t, xhat0, x0, sigma, results):
        self.trajectory.add_tensor('x_t', x_t)
        self.trajectory.add_tensor('xhat0', xhat0)
        self.trajectory.add_tensor('x0', x0)
        self.trajectory.add_value('sigma', sigma)
        self.trajectory.add_value('NFE', self.nfe)
        for name, val in results.items():
            self.trajectory.add_value(f'x0_{name}', val)
