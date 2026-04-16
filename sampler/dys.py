import time
import torch
import wandb as wnb
import numpy as np
import torch.nn as nn
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
    """
    Plug-and-Play DYS / Three-Operator Splitting sampler.

    Uses:
      - operator.loss(x, measurement) as smooth data term h(x)
      - prox_{gamma f} as a projection (default: clamp to [-1, 1])
      - diffusion/score prior via the SAME optimize_denoising() routine as ADMM,
        but applied to the DYS state (no dual variable).

    Call signature matches ADMM.sample(...).
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

        # Optional regularizers hook (same as ADMM file)
        print("No regularizers found!!!")
        self.regularizers = None

        # DYS hyperparameters (best-effort defaults if not present in config)
        self.gamma_step = self._get_gamma_step_default()
        self.lambda_schedule = self._build_lambda_schedule()

        # Projection settings (defaults match your ADMM clamping behavior)
        self.proj_min = float(getattr(getattr(self.admm_config, "proj", None), "min", -1.0)
                              if getattr(self.admm_config, "proj", None) is not None else -1.0)
        self.proj_max = float(getattr(getattr(self.admm_config, "proj", None), "max",  1.0)
                              if getattr(self.admm_config, "proj", None) is not None else 1.0)

        # If config has an explicit boolean to disable projection, respect it.
        self.use_projection = bool(getattr(getattr(self.admm_config, "proj", None), "activate", True)
                                   if getattr(self.admm_config, "proj", None) is not None else True)

    # -------------------------
    # Core utilities
    # -------------------------
    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
        Mirrors ADMM._check(): remove sigma_max from diffusion scheduler config and set sigma_final=0.
        """
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        # Preserve existing behavior: annealing ends at sigma_final=0 unless your Scheduler overrides.
        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    def _get_gamma_step_default(self) -> float:
        """
        DYS needs a stepsize gamma for the gradient term.

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

    def _build_lambda_schedule(self):
        """
        Build lambda_k schedule.

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
        #print(getattr(self.admm_config, "max_iter", 100))
        #K = int(getattr(self.admm_config, "max_iter", 100))
        K = int(self.admm_config.max_iter)
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

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_projection:
            return x
        return torch.clamp(x, min=self.proj_min, max=self.proj_max)

    def _grad_h(self, x: torch.Tensor, operator, measurement) -> torch.Tensor:
        """
        Compute ∇_x operator.loss(x, measurement) via autograd.

        NOTE: This requires operator.__call__ to be differentiable and operator.loss
        to be constructed from torch ops (which is consistent with your Operator base class).
        """
        x_ = x.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = operator.loss(x_, measurement)  # scalar
            grad = torch.autograd.grad(loss, x_, create_graph=False, retain_graph=False)[0]
        return grad.detach()

    # -------------------------
    # Denoiser block (copied/adapted from ADMM)
    # -------------------------
    def optimize_denoising(self, z_in,
                           model, d_k, sigma,
                           prior_use_type="denoise",
                           wandb=False):
        """
        Apply the same "Taming" AC-DC denoiser, but with input = DYS state z_in.

        For AC-only behavior:
          set denoise_config.lgvd.num_steps = 0
        """
        denoise_config = self.admm_config.denoise
        with torch.no_grad():
            noisy_im = z_in.clone()

            if prior_use_type in ["denoise"]:
                # ---- AC step (optional noise injection)
                # If you want to disable AC noise injection explicitly, you can add
                # denoise_config.ac_noise = False in config.
                ac_noise = bool(getattr(denoise_config, "ac_noise", True))
                #ac_noise = 1
                if ac_noise and sigma > 0:
                    forward_z = noisy_im + torch.randn_like(noisy_im) * sigma
                else:
                    forward_z = noisy_im

                # ---- DC step (lgvd)
                lgvd_z = forward_z.clone()
                lr = denoise_config.lgvd.lr * sigma

                num_steps = int(getattr(denoise_config.lgvd, "num_steps", 0))
                reg_factor = float(getattr(denoise_config.lgvd, "reg_factor", 0.0))

                for _ in range(num_steps):
                    #print("hej")
                    score_val = model.score(lgvd_z, sigma)
                    #d_k = 0
                    diff_val = (forward_z - lgvd_z + d_k)

                    # drift toward forward_z (stabilizer); keep same cap heuristic as ADMM
                    drift = lr * min(sigma * reg_factor, 10.0) * diff_val
                    #print(drift)
                    #quit()

                    lgvd_z += lr * score_val + drift + (2 * lr) ** 0.5 * torch.randn_like(noisy_im)

                # ---- final step
                if denoise_config.final_step == 'tweedie':
                    z = model.tweedie(lgvd_z, sigma)
                    #print(z)
                    #print(lgvd_z)
                    #print("hej")
                elif denoise_config.final_step == 'ode':
                    diffusion_scheduler = Scheduler(
                        **self.diffusion_scheduler_config, sigma_max=sigma)
                    sampler = DiffusionSampler(diffusion_scheduler)
                    z = sampler.sample(model, lgvd_z, SDE=False, verbose=False)
                else:
                    raise Exception(f"Final step {denoise_config.final_step} not supported!!!")

                #denoised_img = torch.clamp(z, min=-1.0, max=1.0)
                denoised_img = z
            else:
                raise Exception(f"Prior type {prior_use_type} not supported!!!")

        return denoised_img
    def prox_mag_mse(u0: torch.Tensor, y: torch.Tensor, sigma_dual: float, sigma_n: float, eps: float = 1e-12):
        """
        Computes prox_{(1/sigma_dual) f}(u0) where
        f(u) = (1/(2 sigma_n^2)) || |u| - y ||^2
        u0: complex tensor, y: real >=0 tensor same shape (broadcast ok)
        """
        # alpha = (lambda / sigma_n^2) with lambda = 1/sigma_dual  => alpha = 1/(sigma_dual*sigma_n^2)
        # radius update: r* = (r0 + alpha*y)/(1+alpha) = (r0*(sigma_dual*sigma_n^2) + y) / (sigma_dual*sigma_n^2 + 1)
        r0 = torch.abs(u0)
        a = sigma_dual * (sigma_n ** 2)
        r_star = (r0 * a + y) / (a + 1.0)

        # keep phase of u0
        scale = r_star / (r0 + eps)
        return u0 * scale


    # -------------------------
    # Main sampler
    # -------------------------
    def sample(self, model, ref_img, operator,
               measurement, evaluator=None,
               record=False, verbose=False, wandb=False, **kwargs):
        """
        Callable under the same conditions as ADMM.sample(...).

        We keep the same output convention as ADMM: return the denoised variable
        (here: xB), because that is the "prior output". If you prefer returning xA
        (post-projection / more data-corrected), add a config flag return_xA.
        """
        if record:
            self.trajectory = Trajectory()

        #K = int(getattr(self.admm_config, "max_iter", 100))
        K = self.admm_config.max_iter
        pbar = tqdm.trange(K) if verbose else range(K)

        # Initialize DYS state. We reuse the ADMM-style init_factor config if present.
        # We'll build x_k (xA), z_k (xB), and y_k (state) to keep delta_t style logging similar.
        x_k, z_k, y_k = self.get_start(ref_img)

        # Old values for convergence checks
        x_old, z_old, y_old = None, None, None
        delta_t_old = torch.inf
        delta_patience = 0

        delta_tol = float(getattr(self.admm_config, "delta_tol", -1.0))
        delta_pat = int(getattr(self.admm_config, "delta_patience", 0))

        # Optional: return xA instead of xB
        return_xA = bool(getattr(getattr(self.admm_config, "dys", None), "return_xA", False))

        t0 = time.time()

        for step in pbar:
            # sigma schedule (same pattern as ADMM)

            # sigma schedule
            t_sigma = min(step, self.annealing_scheduler.num_steps - 1)
            sigma = self.annealing_scheduler.sigma_steps[t_sigma]


            # lambda schedule
            lam = float(self.lambda_schedule[min(step, len(self.lambda_schedule) - 1)])

            # (1) Prox/projection of the point (y_k)
            #x_k = self._proj(y_k)
            #x_k = torch.clamp(y_k,min=-1,max=1)
            """x_k = self.optimize_denoising(
                z_in=y_k,
                model=model,
                d_k = y_k - x_k,
                sigma=sigma,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb
            )"""

            #x_k = y_k


            # (2) Gradient at prox point
            grad = self._grad_h(x_k, operator=operator, measurement=measurement)

            # Reflected, gradient-corrected point
            #r_k = 2.0 * x_k - y_k - self.gamma_step * grad
            r_k = x_k - self.gamma_step * grad

            #r_k = torch.clamp(r_k,min=-1,max=1)

            # (3) Denoise the reflected point
            z_k = self.optimize_denoising(
                z_in=r_k,
                model=model,
                d_k = y_k - x_k,
                sigma=sigma,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb
            )
            x_k = z_k

            #z_k = torch.clamp(r_k,min=-1,max=1)

            # (4) Relaxed DYS update: NOTE SIGN (z_k - x_k)
            y_k = y_k + lam * (z_k - x_k)


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
                        "wall_time": time.time() - t0,
                    })

                delta_t_old = delta_t

            # Update old values
            x_old, z_old, y_old = x_k.clone(), z_k.clone(), y_k.clone()

            # Evaluation (keep same naming as ADMM for compatibility)
            x_k_results = z_k_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x_k_results = evaluator(gt, measurement, x_k)  # "data/prox" variable
                    z_k_results = evaluator(gt, measurement, z_k)  # "denoised" variable

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

        #return x_k if return_xA else z_k
        return x_k

    # -------------------------
    # Recording / init (compatible style)
    # -------------------------
    def _record(self, y_k, x_k, z_k, sigma, x_k_results, z_k_results):
        """
        Records intermediate states during sampling.
        Names follow ADMM conventions plus y_k for the DYS state.
        """
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
          - y_k is the DYS state; prefer key 'y', else 'u', else 'z', else 'x'.
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
                    # If init_factor is a DictConfig, init_factor[k] may be a number
                    try:
                        scale = float(init_factor[k])
                    except Exception:
                        # fallback (mirrors ADMM behavior)
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
