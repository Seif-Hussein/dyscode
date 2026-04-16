
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, List

import torch
import wandb as wnb
import numpy as np
import torch.nn as nn
import tqdm

from utils.diffusion import Scheduler, DiffusionSampler
from utils.logging import Trajectory
from .registry import register_sampler


@register_sampler(name='dys_diag')
def get_sampler(**kwargs):
    """
    Diagnostic-friendly DYS sampler.

    - Same call signature as existing samplers.
    - Adds:
        * configurable DYS splitting order (denoise-first vs denoise-last)
        * systematic AC-only vs AC-DC comparisons on-the-fly (no change to iterates)
        * logging of key scalars: data residual, score norm, DC stiffness, etc.

    Usage:
      set sampler: 'dys_diag' in your config/CLI.
    """
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError("Latent-space DYS not implemented.")
    return DYS_Diag(**kwargs)


class DYS_Diag(nn.Module):
    """
    Davis–Yin Splitting (DYS) with systematic diagnostics for AC–DC ("Taming") integration.

    This class is intentionally a *superset* of your current DYS implementation:
      - by default it reproduces the behavior of your current dys.py (denoise-first, clamp r_k)
      - additional features are opt-in via admm_config.dys.* and admm_config.denoise.lgvd.* keys.
    """

    def __init__(
        self,
        annealing_scheduler_config,
        diffusion_scheduler_config,
        lgvd_config,
        admm_config,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        self.annealing_scheduler_config, self.diffusion_scheduler_config = \
            self._check(annealing_scheduler_config, diffusion_scheduler_config)

        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.admm_config = admm_config
        self.device = device

        # Diffusion parameters (only used if denoise.final_step == 'ode')
        self.betas = np.linspace(
            admm_config.denoise.diffusion.beta_start,
            admm_config.denoise.diffusion.beta_end,
            admm_config.denoise.diffusion.T,
            dtype=np.float64,
        )
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Regularizers hook (mirrors ADMM file; left as-is)
        print("No regularizers found!!!")
        self.regularizers = None

        # DYS hyperparameters
        self.gamma_step = self._get_gamma_step_default()
        self.lambda_schedule = self._build_lambda_schedule()

        # Projection settings (defaults match your ADMM clamping behavior)
        self.proj_min = float(
            getattr(getattr(self.admm_config, "proj", None), "min", -1.0)
            if getattr(self.admm_config, "proj", None) is not None
            else -1.0
        )
        self.proj_max = float(
            getattr(getattr(self.admm_config, "proj", None), "max", 1.0)
            if getattr(self.admm_config, "proj", None) is not None
            else 1.0
        )
        self.use_projection = bool(
            getattr(getattr(self.admm_config, "proj", None), "activate", True)
            if getattr(self.admm_config, "proj", None) is not None
            else True
        )

        # DYS options
        dys_cfg = getattr(self.admm_config, "dys", None)
        self.order = str(getattr(dys_cfg, "order", "denoise_first")).lower()
        self.post_denoise_proj = bool(getattr(dys_cfg, "post_denoise_proj", False))
        # Optional: use a physics/data-consistency prox if the operator provides one (e.g. phase retrieval amplitude projection).
        # This is OFF by default; enable via admm_config.dys.use_physics_prox=True and set admm_config.dys.tau as needed.
        self.use_physics_prox = bool(getattr(dys_cfg, "use_physics_prox", False)) if dys_cfg is not None else False
        self.tau = float(getattr(dys_cfg, "tau", float("inf"))) if dys_cfg is not None else float("inf")


        # Diagnostics options
        self.diag_cfg = getattr(dys_cfg, "diagnostics", None)

    # -------------------------
    # Config / schedules
    # -------------------------
    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
        Mirrors ADMM._check() and your current DYS._check().

        NOTE: Your current DYS forces sigma_final=0. We keep that behavior here
        to stay apples-to-apples unless you change this in your own codebase.
        """
        if "sigma_max" in diffusion_scheduler_config:
            diffusion_scheduler_config.pop("sigma_max")

        # Preserve existing behavior: annealing ends at sigma_final=0
        annealing_scheduler_config["sigma_final"] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    def _get_gamma_step_default(self) -> float:
        """
        Priority order (so you can control it from configs):
          1) admm_config.dys.gamma
          2) admm_config.gamma_step
          3) admm_config.step_size
          4) admm_config.ml.lr
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

    def _build_lambda_schedule(self) -> List[float]:
        """
        Priority:
          1) admm_config.dys.lambda_schedule (if present and activate=True)
          2) constant admm_config.dys.lambda
          3) constant admm_config.lambda
          4) default 1.0
        """
        K = int(getattr(self.admm_config, "max_iter", 100))
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

    # -------------------------
    # Primitive maps
    # -------------------------
    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_projection:
            return x
        return torch.clamp(x, min=self.proj_min, max=self.proj_max)

    def _constraint_prox(self, x: torch.Tensor, operator, measurement) -> torch.Tensor:
        """
        Data-consistency / constraint proximal.

        - Default: box projection via clamp (self._proj).
        - If enabled and available: calls operator.proj_amplitude(x, measurement, tau=..., ...).
          This is useful for phase retrieval if you implemented a Fourier-magnitude projection in the operator.
        """
        if self.use_physics_prox and hasattr(operator, "proj_amplitude"):
            # Be robust to different proj_amplitude signatures across your experiments.
            try:
                return operator.proj_amplitude(
                    x, measurement,
                    tau=self.tau,
                    clamp_x=True,
                    clamp01=True,
                )
            except TypeError:
                # Fallback: try minimal signature
                return operator.proj_amplitude(x, measurement)
        return self._proj(x)

    def _grad_h(self, x: torch.Tensor, operator, measurement) -> torch.Tensor:
        """
        Compute ∇ operator.loss(x, measurement) via autograd.
        """
        x_ = x.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = operator.loss(x_, measurement)  # scalar
            grad = torch.autograd.grad(loss, x_, create_graph=False, retain_graph=False)[0]
        return grad.detach()

    # -------------------------
    # AC / DC denoiser (Taming-style) + diagnostics hooks
    # -------------------------
    def optimize_denoising(
        self,
        z_in: torch.Tensor,
        model,
        sigma: float,
        prior_use_type: str = "denoise",
        wandb: bool = False,
        *,
        # overrides (for systematic A/B testing)
        dc_num_steps: Optional[int] = None,
        dc_reg_factor: Optional[float] = None,
        dc_drift_clip: Optional[float] = None,
        dc_noise_scale: Optional[float] = None,
        ac_noise: Optional[bool] = None,
        # if provided, reuse the exact same AC-noised point across variants
        forward_z: Optional[torch.Tensor] = None,
        # return internal scalars for logging
        return_debug: bool = False,
    ):
        """
        Implements the same LGVD update as the released ADMM.py:
          lgvd_z += lr * score
                 + lr * min(sigma*reg_factor, drift_clip) * (forward_z - lgvd_z)
                 + sqrt(2*lr) * noise_scale * N(0,I)

        Key additions for systematic debugging:
          - dc_num_steps override (compare J=0 vs J>0 on same input)
          - dc_noise_scale (set 0.0 to test "deterministic DC")
          - dc_drift_clip configurable (default matches ADMM's 10)
          - return_debug to inspect lr, tether strength, etc.
        """
        denoise_config = self.admm_config.denoise

        with torch.no_grad():
            noisy_im = z_in.clone()

            if prior_use_type not in ["denoise"]:
                raise Exception(f"Prior type {prior_use_type} not supported!!!")

            # AC step
            if ac_noise is None:
                ac_noise_flag = bool(getattr(denoise_config, "ac_noise", True))
            else:
                ac_noise_flag = bool(ac_noise)

            if forward_z is None:
                if ac_noise_flag and sigma > 0:
                    forward_z = noisy_im + torch.randn_like(noisy_im) * sigma
                else:
                    forward_z = noisy_im
            else:
                # caller provides the AC-noised anchor explicitly
                forward_z = forward_z

            # DC (LGVD)
            lgvd_z = forward_z.clone()

            lr0 = float(getattr(denoise_config.lgvd, "lr", 0.0))
            lr = lr0 * float(sigma)

            num_steps = int(dc_num_steps) if dc_num_steps is not None else int(getattr(denoise_config.lgvd, "num_steps", 0))
            reg_factor = float(dc_reg_factor) if dc_reg_factor is not None else float(getattr(denoise_config.lgvd, "reg_factor", 0.0))

            drift_clip = float(dc_drift_clip) if dc_drift_clip is not None else float(getattr(denoise_config.lgvd, "drift_clip", 10.0))
            noise_scale = float(dc_noise_scale) if dc_noise_scale is not None else float(getattr(denoise_config.lgvd, "noise_scale", 1.0))

            # tether strength as in released ADMM code:
            # alpha = min(sigma * reg_factor, drift_clip)
            alpha_raw = float(sigma) * reg_factor
            alpha = min(alpha_raw, drift_clip)

            # Optional: cap the *product* lr*alpha based on explicit-Euler stability (0 < lr*alpha < 2)
            # This is OFF by default to remain faithful to the released implementation.
            max_tether_step = getattr(denoise_config.lgvd, "max_tether_step", None)
            if max_tether_step is not None:
                max_tether_step = float(max_tether_step)
                if lr > 0:
                    alpha = min(alpha, max_tether_step / (lr + 1e-12))

            score_calls = 0
            for _ in range(num_steps):
                score_val = model.score(lgvd_z, sigma)
                score_calls += 1
                diff_val = forward_z - lgvd_z

                lgvd_z += lr * score_val + lr * alpha * diff_val + noise_scale * (2 * lr) ** 0.5 * torch.randn_like(noisy_im)

            # final step
            if denoise_config.final_step == "tweedie":
                z = model.tweedie(lgvd_z, sigma)
            elif denoise_config.final_step == "ode":
                diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionSampler(diffusion_scheduler)
                z = sampler.sample(model, lgvd_z, SDE=False, verbose=False)
            else:
                raise Exception(f"Final step {denoise_config.final_step} not supported!!!")

            denoised_img = z

            if return_debug:
                debug = {
                    "sigma": float(sigma),
                    "lr0": lr0,
                    "lr": float(lr),
                    "num_steps": int(num_steps),
                    "reg_factor": float(reg_factor),
                    "alpha_raw": float(alpha_raw),
                    "alpha": float(alpha),
                    "tether_step": float(lr * alpha),
                    "drift_clip": float(drift_clip),
                    "noise_scale": float(noise_scale),
                    "score_calls": int(score_calls),
                }
                return denoised_img, debug

        return denoised_img

    def _compare_acdc_variants(
        self,
        *,
        z_in: torch.Tensor,
        model,
        sigma: float,
        operator=None,
        measurement=None,
        evaluator=None,
        gt: Optional[torch.Tensor] = None,
        # variant specs
        variants: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Systematic comparison: run multiple denoiser variants on the *same* input and the *same* AC noise.

        Returns a dict:
          results[name] = {
              'out': Tensor,
              'debug': {...},
              'data_error': float (optional),
              'eval': {...} (optional)
          }
        """
        with torch.no_grad():
            # reuse the same AC-noised anchor across all variants to make comparisons meaningful
            denoise_config = self.admm_config.denoise
            ac_noise_flag = bool(getattr(denoise_config, "ac_noise", True))
            if ac_noise_flag and sigma > 0:
                forward_z = z_in + torch.randn_like(z_in) * sigma
            else:
                forward_z = z_in

        out: Dict[str, Dict[str, Any]] = {}
        for v in variants:
            name = str(v.get("name", "variant"))
            num_steps = int(v.get("num_steps", 0))
            noise_scale = float(v.get("noise_scale", 1.0))
            # allow per-variant overrides; default to config values if not set
            reg_factor = v.get("reg_factor", None)
            drift_clip = v.get("drift_clip", None)

            den, dbg = self.optimize_denoising(
                z_in=z_in,
                model=model,
                sigma=sigma,
                prior_use_type=self.admm_config.denoise.type,
                dc_num_steps=num_steps,
                dc_noise_scale=noise_scale,
                dc_reg_factor=reg_factor,
                dc_drift_clip=drift_clip,
                ac_noise=False,          # we already formed forward_z explicitly
                forward_z=forward_z,     # same anchor across variants
                return_debug=True,
            )

            row: Dict[str, Any] = {"out": den, "debug": dbg}

            if operator is not None and measurement is not None:
                row["data_error"] = float(operator.error(den, measurement).mean().item())

            if evaluator is not None and gt is not None:
                ev = evaluator(gt, measurement, den)
                row["eval"] = {k: float(vv.item()) for k, vv in ev.items()}

            out[name] = row

        return out

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
        record: bool = False,
        verbose: bool = False,
        wandb: bool = False,
        **kwargs,
    ):
        if record:
            self.trajectory = Trajectory()

        K = int(getattr(self.admm_config, "max_iter", 100))
        pbar = tqdm.trange(K) if verbose else range(K)

        # init
        x_k, z_k, y_k = self.get_start(ref_img)

        # convergence bookkeeping
        x_old = z_old = y_old = None
        delta_patience = 0
        delta_tol = float(getattr(self.admm_config, "delta_tol", -1.0))
        delta_pat = int(getattr(self.admm_config, "delta_patience", 0))

        # output selection
        dys_cfg = getattr(self.admm_config, "dys", None)
        return_xA = bool(getattr(dys_cfg, "return_xA", False))

        # diagnostics flags
        diag_on = bool(getattr(self.diag_cfg, "activate", False)) if self.diag_cfg is not None else False
        diag_every = int(getattr(self.diag_cfg, "every", 1)) if self.diag_cfg is not None else 1
        diag_compare = bool(getattr(self.diag_cfg, "compare_acdc", False)) if self.diag_cfg is not None else False
        diag_compare_det = bool(getattr(self.diag_cfg, "compare_det_dc", True)) if self.diag_cfg is not None else True
        diag_log_score_norm = bool(getattr(self.diag_cfg, "score_norm", False)) if self.diag_cfg is not None else False
        diag_log_mismatch = bool(getattr(self.diag_cfg, "mismatch_gt", False)) if self.diag_cfg is not None else False

        t0 = time.time()

        for step in pbar:
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma = float(self.annealing_scheduler.sigma_steps[t_sigma])

            lam = float(self.lambda_schedule[min(step, len(self.lambda_schedule) - 1)])

            # -----------------
            # DYS update (order-dependent)
            # -----------------
            # The variable fed into the denoiser depends on order.
            denoise_input: Optional[torch.Tensor] = None
            r_k = None

            if self.order in ("denoise_first", "a", "prior_first", "denoise_y"):
                # (A) denoise-first (matches your current dys.py behavior):
                #   x_k = denoise(y_k)
                #   r_k = 2*x_k - y_k - gamma*grad
                #   z_k = proj(r_k)
                denoise_input = y_k

                x_k = self.optimize_denoising(
                    z_in=y_k,
                    model=model,
                    sigma=sigma,
                    prior_use_type=self.admm_config.denoise.type,
                    wandb=wandb,
                )
                x_k = operator.post_ml_op(x_k, measurement)

                grad = self._grad_h(x_k, operator=operator, measurement=measurement)
                r_k = 2.0 * x_k - y_k - self.gamma_step * grad

                z_k = self._constraint_prox(r_k, operator=operator, measurement=measurement)

            elif self.order in ("denoise_last", "b", "prior_last", "denoise_r"):
                # (B) denoise-last:
                #   x_k = proj(y_k)
                #   r_k = 2*x_k - y_k - gamma*grad
                #   z_k = denoise(r_k)
                x_k = self._constraint_prox(y_k, operator=operator, measurement=measurement)
                x_k = operator.post_ml_op(x_k, measurement)

                grad = self._grad_h(x_k, operator=operator, measurement=measurement)
                r_k = 2.0 * x_k - y_k - self.gamma_step * grad

                denoise_input = r_k
                z_k = self.optimize_denoising(
                    z_in=r_k,
                    model=model,
                    sigma=sigma,
                    prior_use_type=self.admm_config.denoise.type,
                    wandb=wandb,
                )

                if self.post_denoise_proj:
                    z_k = self._proj(z_k)

            else:
                raise ValueError(f"Unknown DYS order='{self.order}'. Use 'denoise_first' or 'denoise_last'.")

            # (4) relaxed update
            y_k = y_k + lam * (z_k - x_k)

            # -----------------
            # Convergence check
            # -----------------
            if step != 0:
                denom = float(np.prod(x_k.shape))
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
                    wnb.log(
                        {
                            "DYS Iteration": step + 1,
                            "delta_t": float(delta_t),
                            "sigma": float(sigma),
                            "lambda": float(lam),
                            "gamma_step": float(self.gamma_step),
                            "wall_time": time.time() - t0,
                        }
                    )

            x_old, z_old, y_old = x_k.clone(), z_k.clone(), y_k.clone()

            # -----------------
            # Standard evaluation (x_k and z_k)
            # -----------------
            x_k_results = z_k_results = {}
            gt = kwargs.get("gt", None)

            if evaluator is not None and gt is not None:
                with torch.no_grad():
                    x_k_results = evaluator(gt, measurement, x_k)
                    z_k_results = evaluator(gt, measurement, z_k)

                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix(
                        {
                            "x_k_" + main_eval_fn_name: f"{x_k_results[main_eval_fn_name].item():.2f}",
                            "z_k_" + main_eval_fn_name: f"{z_k_results[main_eval_fn_name].item():.2f}",
                        }
                    )

                if wandb:
                    for fn_name in x_k_results.keys():
                        wnb.log(
                            {
                                f"x_k_{fn_name}": x_k_results[fn_name].item(),
                                f"z_k_{fn_name}": z_k_results[fn_name].item(),
                            }
                        )

            # -----------------
            # Diagnostics: AC-only vs AC-DC on the same denoiser input
            # -----------------
            if diag_on and (step % max(diag_every, 1) == 0) and (denoise_input is not None):
                # log a few core scalars (no extra score calls unless requested)
                diag_log: Dict[str, Any] = {
                    "diag/sigma": float(sigma),
                    "diag/lambda": float(lam),
                    "diag/gamma_step": float(self.gamma_step),
                    "diag/order": 0 if self.order.startswith("denoise_first") else 1,
                    "diag/use_physics_prox": 1 if self.use_physics_prox else 0,
                }

                # data residuals (cheap; uses forward operator once)
                try:
                    diag_log["diag/data_err_xk"] = float(operator.error(x_k, measurement).mean().item())
                    diag_log["diag/data_err_zk"] = float(operator.error(z_k, measurement).mean().item())
                except Exception:
                    pass

                # mismatch ratio m_k = ||z_in - gt||/(sqrt(d)*sigma) (only if gt and requested)
                if diag_log_mismatch and gt is not None:
                    d = float(denoise_input[0].numel())
                    denom = (d ** 0.5) * max(float(sigma), 1e-12)
                    mk = float((denoise_input - gt).norm().item() / denom)
                    diag_log["diag/mismatch_ratio"] = mk

                # score norm (extra score call; optional)
                if diag_log_score_norm:
                    with torch.no_grad():
                        s = model.score(denoise_input, sigma)
                        diag_log["diag/score_norm_in"] = float(s.norm().item() / (denoise_input[0].numel() ** 0.5))

                # denoiser comparisons (potentially expensive; optional)
                if diag_compare:
                    J = int(getattr(self.admm_config.denoise.lgvd, "num_steps", 0))
                    variants = [{"name": "AC_only", "num_steps": 0, "noise_scale": 1.0}]
                    if diag_compare_det and J > 0:
                        variants.append({"name": "ACDC_det", "num_steps": J, "noise_scale": 0.0})
                    if J > 0:
                        variants.append({"name": "ACDC_stoch", "num_steps": J, "noise_scale": 1.0})

                    comp = self._compare_acdc_variants(
                        z_in=denoise_input,
                        model=model,
                        sigma=sigma,
                        operator=operator,
                        measurement=measurement,
                        evaluator=evaluator if (evaluator is not None and gt is not None) else None,
                        gt=gt if gt is not None else None,
                        variants=variants,
                    )

                    # flatten into scalar logs
                    for name, row in comp.items():
                        dbg = row.get("debug", {})
                        diag_log[f"diag/{name}/tether_step"] = float(dbg.get("tether_step", 0.0))
                        diag_log[f"diag/{name}/alpha"] = float(dbg.get("alpha", 0.0))
                        diag_log[f"diag/{name}/lr"] = float(dbg.get("lr", 0.0))
                        diag_log[f"diag/{name}/score_calls"] = float(dbg.get("score_calls", 0.0))
                        if "data_error" in row:
                            diag_log[f"diag/{name}/data_err"] = float(row["data_error"])
                        if "eval" in row:
                            # log the main metric and all others
                            for kname, kval in row["eval"].items():
                                diag_log[f"diag/{name}/{kname}"] = float(kval)

                if wandb:
                    wnb.log(diag_log)

            # record trajectory
            if record:
                self._record(
                    y_k=y_k, x_k=x_k, z_k=z_k,
                    sigma=sigma,
                    x_k_results=x_k_results,
                    z_k_results=z_k_results,
                    r_k=r_k,
                )

        return x_k if return_xA else z_k

    # -------------------------
    # Recording / init
    # -------------------------
    def _record(self, y_k, x_k, z_k, sigma, x_k_results, z_k_results, r_k=None):
        self.trajectory.add_tensor("x_k", x_k)
        self.trajectory.add_tensor("z_k", z_k)
        self.trajectory.add_tensor("y_k", y_k)
        if r_k is not None:
            self.trajectory.add_tensor("r_k", r_k)
        self.trajectory.add_value("sigma", sigma)
        for name in x_k_results.keys():
            self.trajectory.add_value(f"x_k_{name}", x_k_results[name])
        for name in z_k_results.keys():
            self.trajectory.add_value(f"z_k_{name}", z_k_results[name])

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