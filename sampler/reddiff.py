import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb as wnb

from utils.diffusion import Scheduler
from utils.logging import Trajectory
from .registry import register_sampler

@register_sampler(name='reddiff')
def get_sampler_alias(**kwargs):
    latent = kwargs.get('latent', False)
    kwargs.pop('latent', None)
    if latent:
        raise NotImplementedError("Latent-space RED-diff not implemented.")
    return REDDIFF(**kwargs)



@register_sampler(name='red_diff')
def get_sampler(**kwargs):
    """Factory to match the ADMM callable conditions.

    In your codebase, samplers are registered by decorating a *callable*.
    ADMM registers a get_sampler(...) function; we mirror that pattern.
    """
    latent = kwargs.get('latent', False)
    if 'latent' in kwargs:
        kwargs.pop('latent')
    if latent:
        raise NotImplementedError("Latent-space RED-diff not implemented.")
    return REDDIFF(**kwargs)


def _maybe_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely fetch attribute or dict key from (possibly) OmegaConf objects."""
    if cfg is None:
        return default
    # OmegaConf DictConfig supports both attribute and dict-style.
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]
    except Exception:
        return default


def _resolve_subcfg(root: Any, names: Tuple[str, ...]) -> Any:
    """Return the first existing sub-config among candidate names."""
    for n in names:
        sub = _maybe_get(root, n, None)
        if sub is not None:
            return sub
    return None


@dataclass
class _SigmaPicker:
    """Maps optimization iterations -> diffusion noise levels (sigma)."""

    sigmas: torch.Tensor  # shape [S]
    max_iter: int
    mode: str = "descending"  # {descending, random}
    spacing: str = "linear"   # {linear, log, exp} for descending

    def __post_init__(self):
        if self.sigmas.ndim != 1:
            raise ValueError("sigmas must be a 1D tensor")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self.mode = str(self.mode).lower()
        self.spacing = str(self.spacing).lower()

        # Filter out nonpositive sigmas to avoid division-by-zero.
        self.sigmas = self.sigmas[self.sigmas > 0]
        if len(self.sigmas) == 0:
            raise ValueError("No positive sigma values available.")

        # Precompute a sigma index schedule for descending mode.
        if self.mode == "descending":
            self._idx_schedule = self._build_descending_idx_schedule(
                S=len(self.sigmas),
                L=self.max_iter,
                spacing=self.spacing,
            )
        elif self.mode == "random":
            self._idx_schedule = None
        else:
            raise ValueError(f"Unknown sigma picker mode: {self.mode}")

    @staticmethod
    def _build_descending_idx_schedule(S: int, L: int, spacing: str) -> np.ndarray:
        """Choose L indices in [0, S-1] spanning the whole sigma range.

        This is the key fix vs the naive idx=min(step,S-1):
        - If S != L (e.g., S=1000 diffusion timesteps, L=100 optimizer steps),
          we still traverse the *full* sigma range using a spacing rule.

        RED-diff discusses that the time-stepping schedule matters and
        considers uniform/log/exponential spacing over diffusion timesteps.
        """
        if L == 1:
            return np.array([S - 1], dtype=np.int64)

        spacing = str(spacing).lower()
        if spacing == "linear":
            idx = np.linspace(0, S - 1, L)
        elif spacing == "log":
            # More density near small sigma (late timesteps).
            t = np.geomspace(1.0, float(S), L) - 1.0
            idx = (t / (S - 1)) * (S - 1)
        elif spacing == "exp":
            # More density near large sigma (early timesteps).
            t = np.geomspace(1.0, float(S), L) - 1.0
            t = (S - 1) - t  # flip
            idx = (t / (S - 1)) * (S - 1)
        else:
            raise ValueError(f"Unknown spacing: {spacing}")

        idx = np.round(idx).astype(np.int64)
        idx = np.clip(idx, 0, S - 1)
        return idx

    def __call__(self, step: int, device: torch.device) -> torch.Tensor:
        if self.mode == "random":
            j = int(torch.randint(low=0, high=len(self.sigmas), size=(1,)).item())
        else:
            j = int(self._idx_schedule[min(step, self.max_iter - 1)])
        return self.sigmas[j].to(device)


class REDDIFF(nn.Module):
    """RED-diff baseline (Appendix H.4 in the Taming paper).

    What we implement (faithful to RED-diff Algorithm 1):
      - Maintain a mean variable μ (mu) to be optimized.
      - At each iteration, sample a diffusion noise level σ ("timestep") and ε~N(0,I).
      - Form x_t = μ + σ ε (VE-style; this matches your codebase's continuous σ schedule).
      - Use the pretrained score model to predict ε_θ(x_t, σ) via ε_θ = -σ * score(x_t, σ).
      - Use the RED-diff linear surrogate loss:
            L = 0.5 * || y - A(μ) ||^2  +  λ(σ) * < stopgrad(ε_θ - ε), μ >
        so that ∇_μ L contains the regularization direction (ε_θ - ε) without
        differentiating through the pretrained model.

    Notes:
      - RED-diff heavily depends on (i) time-stepping schedule and
        (ii) denoiser-weighting vs σ (often expressed via inverse SNR).
      - The Taming paper only specifies λ=0.25 and lr=0.5 for RED-diff.
        The RED-diff paper discusses additional design choices.

    This sampler is designed to run inside your existing Hydra pipeline:
      sampler.sample(model, ref_img, operator, measurement, evaluator, record, verbose, wandb, **kwargs)
    """

    def __init__(
        self,
        annealing_scheduler_config: Dict[str, Any],
        diffusion_scheduler_config: Dict[str, Any],
        lgvd_config: Any,
        admm_config: Any,
        device: str = 'cuda',
        **kwargs,
    ):
        super().__init__()

        # The codebase passes these four configs to every sampler.
        # RED-diff only needs the annealing schedule (noise levels σ_k).
        self.annealing_scheduler_config = dict(annealing_scheduler_config)
        self.diffusion_scheduler_config = dict(diffusion_scheduler_config)
        self.admm_config = admm_config
        self.device = torch.device(device)

        # Scheduler: provide a list of sigmas to pick from.
        # We do NOT force sigma_final=0 here; but if configs set it, we safely ignore σ<=0.
        self.annealing_scheduler = Scheduler(**self.annealing_scheduler_config)

        # RED-diff hyperparameters live under admm_config.red_diff (recommended).
        # We also accept admm_config.reddiff for robustness.
        self.cfg = _resolve_subcfg(self.admm_config, ("red_diff", "reddiff"))

        # Defaults are aligned with Appendix H.4 in the Taming paper for RED-diff.
        self.lr = float(_maybe_get(self.cfg, "lr", 0.5))
        self.lam = float(_maybe_get(self.cfg, "lambda", 0.25))

        # Number of optimization iterations.
        self.max_iter = int(_maybe_get(self.cfg, "max_iter", _maybe_get(self.admm_config, "max_iter", 100)))

        # Sigma selection schedule.
        self.time_sampling = str(_maybe_get(self.cfg, "time_sampling", "descending"))
        self.time_spacing = str(_maybe_get(self.cfg, "time_spacing", "linear"))

        # Denoiser-weighting / regularization scaling.
        # Choices: {constant, sigma, sigma2, inv_snr, sqrt_inv_snr}
        self.weight_type = str(_maybe_get(self.cfg, "weight_type", "inv_snr"))

        # Optional q(x0) dispersion (RED-diff allows it, but can be set 0).
        self.sigma_x0 = float(_maybe_get(self.cfg, "sigma_x0", 0.0))

        # Optional projection of μ after each step (helps stability in practice).
        self.clip = bool(_maybe_get(self.cfg, "clip", True))
        self.clip_min = float(_maybe_get(self.cfg, "clip_min", -1.0))
        self.clip_max = float(_maybe_get(self.cfg, "clip_max", 1.0))

        # Initialization.
        # For HDR, a useful heuristic is pseudo-inverse of y: μ0 ≈ y / scale.
        self.init_mode = str(_maybe_get(self.cfg, "init", "pinv"))
        self.init_damp = float(_maybe_get(self.cfg, "init_damp", 1.0))  # optional shrink to avoid clamp dead-zone

        # Loss choice for the data term.
        # - "mse": 0.5 * mean(||A(mu) - y||^2)  (matches many reference implementations)
        # - "nll": operator.loss(mu, y) / numel (your Operator's scaled likelihood)
        self.data_term = str(_maybe_get(self.cfg, "data_term", "mse"))

        # Build sigma picker.
        sigmas = torch.as_tensor(self.annealing_scheduler.sigma_steps, dtype=torch.float32)
        self.sigma_picker = _SigmaPicker(sigmas=sigmas, max_iter=self.max_iter,
                                         mode=self.time_sampling, spacing=self.time_spacing)

        # Bookkeeping
        self._nfe = 0

    # -------------------------
    # Core math blocks
    # -------------------------
    def _init_mu(self, ref_img: torch.Tensor, operator, measurement: torch.Tensor) -> torch.Tensor:
        """Initialize μ.

        RED-diff reference implementations often use a task-dependent pseudo-inverse
        initialization when available.

        For your current focus (HDR operator): A(x)=clip(scale*x, -1,1),
        a reasonable pseudo-inverse is y/scale, optionally shrunk.
        """
        if self.init_mode.lower() in ["noise", "random"]:
            mu0 = torch.randn_like(ref_img) * float(getattr(self.annealing_scheduler, "sigma_max", 1.0))
            return mu0.to(self.device)

        if self.init_mode.lower() in ["measurement", "y"]:
            mu0 = measurement
            if mu0.shape != ref_img.shape:
                # fallback: random if not shape-compatible
                mu0 = torch.randn_like(ref_img)
            return mu0.to(self.device)

        # "pinv" / heuristic pseudo-inverse
        if hasattr(operator, "pinv") and callable(getattr(operator, "pinv")):
            try:
                mu0 = operator.pinv(measurement)
                return mu0.to(self.device)
            except Exception:
                pass

        # Common heuristic for HDR in your operator implementation.
        scale = getattr(operator, "scale", None)
        if scale is not None:
            try:
                mu0 = measurement / float(scale)
            except Exception:
                mu0 = measurement
        else:
            mu0 = measurement

        mu0 = mu0.to(self.device)

        # Optional damping (useful when A involves clamp, to keep gradients alive).
        if self.init_damp != 1.0:
            mu0 = mu0 * self.init_damp

        if self.clip:
            mu0 = mu0.clamp(self.clip_min, self.clip_max)
        return mu0

    def _lambda_of_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute per-iteration regularization weight λ(σ)."""
        wt = self.weight_type.lower()
        if wt in ["const", "constant", "none"]:
            w = torch.ones_like(sigma)
        elif wt in ["sigma", "sqrt_inv_snr", "snr_sqrt_inv"]:
            w = sigma
        elif wt in ["sigma2", "inv_snr", "snr_inv"]:
            w = sigma ** 2
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")
        return self.lam * w

    def _predict_eps_from_score(self, score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Convert a score s(x,σ) into an epsilon predictor.

        For VE-style perturbations x = x0 + σ ε, the conditional score of p(x|x0)
        is -(x - x0)/σ^2 = -ε/σ. This motivates ε ≈ -σ * score.

        This is the same conversion implicitly used by Tweedie's formula:
          x0_hat = x + σ^2 score,  ε_hat = (x - x0_hat)/σ = -σ score.
        """
        return -sigma * score

    def _data_loss(self, mu: torch.Tensor, operator, measurement: torch.Tensor) -> torch.Tensor:
        if self.data_term.lower() == "nll":
            # operator.loss already includes 1/(2*sigma_y^2) scaling and sums;
            # normalize by number of elements to make LR less resolution-dependent.
            return operator.loss(mu, measurement) / float(mu.numel())

        # default: MSE in measurement domain
        yhat = operator(mu)
        return 0.5 * torch.mean((yhat - measurement) ** 2)

    # -------------------------
    # Sampler API
    # -------------------------
    def sample(
        self,
        model,
        ref_img: torch.Tensor,
        operator,
        measurement,
        evaluator=None,
        record: bool = False,
        verbose: bool = False,
        wandb: bool = False,
        **kwargs,
    ):
        if record:
            self.trajectory = Trajectory()

        # Initialize μ.
        mu = self._init_mu(ref_img, operator, measurement)
        mu = nn.Parameter(mu)

        # Optimizer.
        opt_name = str(_maybe_get(self.cfg, "optimizer", "adam")).lower()
        if opt_name == "adam":
            betas = _maybe_get(self.cfg, "betas", (0.9, 0.999))
            optimizer = torch.optim.Adam([mu], lr=self.lr, betas=tuple(betas))
        elif opt_name == "sgd":
            momentum = float(_maybe_get(self.cfg, "momentum", 0.0))
            optimizer = torch.optim.SGD([mu], lr=self.lr, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        pbar = tqdm.trange(self.max_iter) if verbose else range(self.max_iter)
        t0 = time.time()
        self._nfe = 0

        for step in pbar:
            sigma = self.sigma_picker(step, device=self.device)

            # Optional dispersion: x0 ~ N(mu, sigma_x0^2 I)
            if self.sigma_x0 > 0:
                x0 = mu + self.sigma_x0 * torch.randn_like(mu)
            else:
                x0 = mu

            # Sample x_t = x0 + sigma * eps
            eps = torch.randn_like(x0)
            x_t = x0 + sigma * eps

            # Score evaluation (stop-grad through the model by construction)
            with torch.no_grad():
                score = model.score(x_t, sigma)
                eps_hat = self._predict_eps_from_score(score, sigma)
                reg_dir = eps_hat - eps

            # Build linear surrogate loss giving ∇_mu reg = reg_dir
            # (matches the stopped-gradient objective in RED-diff).
            lam_t = self._lambda_of_sigma(sigma)
            reg_loss = torch.mean(reg_dir.detach() * x0)  # x0 ~ mu if sigma_x0 small

            data_loss = self._data_loss(mu=x0, operator=operator, measurement=measurement)
            loss = data_loss + lam_t * reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Optional projection to keep μ in the valid image box.
            if self.clip:
                with torch.no_grad():
                    mu.clamp_(self.clip_min, self.clip_max)

            # Bookkeeping / diagnostics
            self._nfe += 1

            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    mu_results = evaluator(gt, measurement, mu.detach())
            else:
                mu_results = None

            if verbose and evaluator and mu_results is not None:
                main_name = evaluator.main_eval_fn_name
                pbar.set_postfix({
                    f"{main_name}": f"{mu_results[main_name].item():.2f}",
                    "sigma": f"{float(sigma):.3g}",
                    "lam_t": f"{float(lam_t):.3g}",
                    "loss": f"{float(loss.item()):.3g}",
                })

            if wandb:
                log_dict = {
                    "RED-diff/iter": step + 1,
                    "RED-diff/sigma": float(sigma),
                    "RED-diff/lambda_t": float(lam_t),
                    "RED-diff/data_loss": float(data_loss.item()),
                    "RED-diff/reg_loss": float(reg_loss.item()),
                    "RED-diff/loss": float(loss.item()),
                    "RED-diff/NFE": self._nfe,
                    "RED-diff/wall_time": time.time() - t0,
                }
                if mu_results is not None:
                    for k, v in mu_results.items():
                        log_dict[f"mu_{k}"] = float(v.item())
                wnb.log(log_dict)

            if record:
                self.trajectory.add_tensor('mu', mu.detach())
                self.trajectory.add_value('sigma', float(sigma))
                self.trajectory.add_value('lambda_t', float(lam_t))
                self.trajectory.add_value('data_loss', float(data_loss.item()))
                self.trajectory.add_value('reg_loss', float(reg_loss.item()))

        return mu.detach()
