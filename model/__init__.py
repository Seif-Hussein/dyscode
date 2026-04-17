# Code taken from: https://github.com/zhangbingliang2019/DAPS/blob/main/model/__init__.py
# Original author: bingliang
from .edm import dnnlib
import pickle
import torch
import torch.nn as nn
from .ddpm.unet import create_model
from omegaconf import OmegaConf
import importlib
from abc import abstractmethod
from .precond import VPPrecond, LDM
import sys
import warnings

__MODEL__ = {}


def register_model(name: str):
    def wrapper(cls):
        if __MODEL__.get(name, None):
            if __MODEL__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __MODEL__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_model(name: str, **kwargs):
    if __MODEL__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __MODEL__[name](**kwargs)


class DiffusionModel(nn.Module):
    """
    A class representing a diffusion model.
    Methods:
        score(x, sigma): Calculates the score function of time-varying noisy distribution:

                \nabla_{x_t}\log p(x_t;\sigma_t)

        tweedie(x, sigma): Calculates the expectation of clean data (x0) given noisy data (xt):

             \mathbb{E}_{x_0 \sim p(x_0 \mid x_t)}[x_0 \mid x_t]
    """

    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Check if either `score` or `tweedie` is overridden
        if (self.score.__func__ is DiffusionModel.score and
                self.tweedie.__func__ is DiffusionModel.tweedie):
            raise NotImplementedError(
                "Either `score` or `tweedie` method must be implemented."
            )

    def score(self, x, sigma):
        """
            x       : noisy state at time t, torch.Tensor([B, *data.shape])
            sigma   : noise level at time t, scaler
        """
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def tweedie(self, x, sigma):
        """
            x       : noisy state at time t, torch.Tensor([B, *data.shape])
            sigma   : noise level at time t, scaler
        """
        return x + self.score(x, sigma) * sigma ** 2


class LatentDiffusionModel(nn.Module):
    """
    A class representing a latent diffusion model.
    Methods:
        encode(x0): Encodes the input `x0` into latent space.
        decode(z0): Decodes the latent variable `z0` into the output space.
        score(z, sigma): Calculates the score of the latent diffusion model given the latent variable `z` and standard deviation `sigma`.
        tweedie(z, sigma): Calculates the Tweedie distribution given the latent variable `z` and standard deviation `sigma`.
        Must overload either `score` or `tweedie` method.
    """

    def __init__(self):
        super(LatentDiffusionModel, self).__init__()
        # Check if either `score` or `tweedie` is overridden
        if (self.score.__func__ is LatentDiffusionModel.score and
                self.tweedie.__func__ is LatentDiffusionModel.tweedie):
            raise NotImplementedError(
                "Either `score` or `tweedie` method must be implemented."
            )

    @abstractmethod
    def encode(self, x0):
        pass

    @abstractmethod
    def decode(self, z0):
        pass

    def score(self, z, sigma):
        d = self.tweedie(z, sigma)
        return (d - z) / sigma ** 2

    def tweedie(self, z, sigma):
        return z + self.score(z, sigma) * sigma ** 2


@register_model(name='ddpm')
class DDPM(DiffusionModel):
    """
    DDPM (Diffusion Denoising Probabilistic Model)
    Attributes:
        model (VPPrecond): The neural network used for denoising.

    Methods:
        __init__(self, model_config, device='cuda'): Initializes the DDPM object.
        tweedie(self, x, sigma=2e-3): Applies the DDPM model to denoise the input, using VP preconditioning from EDM.
    """

    def __init__(self, model_config, device='cuda', requires_grad=False):
        super().__init__()
        self.model = VPPrecond(model=create_model(**model_config), learn_sigma=model_config['learn_sigma'],
                               conditional=model_config['class_cond']).to(device)
        self.model.eval()
        self.model.requires_grad_(requires_grad)

    def tweedie(self, x, sigma=2e-3):
        return self.model(x, torch.as_tensor(sigma).to(x.device))


@register_model(name='edm')
class EDM(DiffusionModel):
    """
    Diffusion models from EDM (Elucidating the Design Space of Diffusion-Based Generative Models).
    """

    def __init__(self, model_config, device='cuda', requires_grad=False):
        super().__init__()
        self.model = self.load_pretrained_model(model_config['model_path'], device=device)

        self.model.eval()
        self.model.requires_grad_(requires_grad)

    def load_pretrained_model(self, url, device='cuda'):
        with dnnlib.util.open_url(url) as f:
            sys.path.append('model/edm')
            model = pickle.load(f)['ema'].to(device)
        return model

    def tweedie(self, x, sigma=2e-3):
        return self.model(x, torch.as_tensor(sigma).to(x.device))


@register_model(name='ldm_ddpm')
class LatentDDPM(LatentDiffusionModel):
    """
    Latent Diffusion Models (High-Resolution Image Synthesis with Latent Diffusion Models).
    """

    def __init__(self, ldm_config, diffusion_path, device='cuda', requires_grad=False):
        super().__init__()
        config = OmegaConf.load(ldm_config)
        net = LDM(load_model_from_config(config, diffusion_path)).to(device)
        self.model = VPPrecond(model=net).to(device)

        self.model.eval()
        self.model.requires_grad_(requires_grad)

    def encode(self, x0):
        return self.model.model.encode(x0)

    def decode(self, z0):
        return self.model.model.decode(z0)

    def tweedie(self, x, sigma=2e-3):
        return self.model(x, torch.as_tensor(sigma).to(x.device))


@register_model(name='dm4ct_pixel_diffusers')
class DM4CTPixelDiffusers(DiffusionModel):
    """
    Thin adapter around a Diffusers DDPM checkpoint.

    DM4CT releases pixel-space CT priors as Diffusers pipelines with a discrete
    DDPM scheduler. This wrapper exposes the repo's expected continuous
    `tweedie(x, sigma)` / `score(x, sigma)` API by:
      1) interpreting the incoming state as x = x0 + sigma * eps,
      2) mapping sigma to the nearest DDPM VP sigma,
      3) rescaling into the DDPM's x_t parameterization before calling the UNet.
    """

    def __init__(self, model_config, device='cuda', requires_grad=False):
        super().__init__()

        try:
            from diffusers import DDPMPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers is required for model.name='dm4ct_pixel_diffusers'. "
                "Install it with `pip install diffusers`."
            ) from exc

        model_id = model_config["model_id"]
        local_files_only = bool(model_config.get("local_files_only", False))
        torch_dtype = self._resolve_torch_dtype(model_config.get("torch_dtype", None))
        self.clamp_output = bool(model_config.get("clamp_output", True))
        self.sigma_eps = float(model_config.get("sigma_eps", 1e-6))
        self.external_range = str(model_config.get("external_range", "minus_one_one"))

        load_kwargs = {"local_files_only": local_files_only}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        pipeline = DDPMPipeline.from_pretrained(model_id, **load_kwargs)
        self.unet = pipeline.unet.to(device)
        self.scheduler = pipeline.scheduler
        self.unet.eval()
        self.unet.requires_grad_(requires_grad)

        alpha_bar = torch.as_tensor(self.scheduler.alphas_cumprod, dtype=torch.float32)
        if alpha_bar.ndim != 1:
            alpha_bar = alpha_bar.flatten()

        vp_sigmas = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12) / alpha_bar.clamp_min(1e-12))
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("vp_sigmas", vp_sigmas)

    @staticmethod
    def _resolve_torch_dtype(name):
        if name is None:
            return None
        if not hasattr(torch, name):
            raise ValueError(f"Unknown torch dtype '{name}'")
        return getattr(torch, name)

    def _sigma_vector(self, sigma, batch_size: int, device, dtype) -> torch.Tensor:
        if torch.is_tensor(sigma):
            sigma_vec = sigma.to(device=device, dtype=dtype).reshape(-1)
        else:
            sigma_vec = torch.as_tensor(sigma, device=device, dtype=dtype).reshape(-1)

        if sigma_vec.numel() == 1:
            sigma_vec = sigma_vec.expand(batch_size)
        elif sigma_vec.numel() != batch_size:
            raise ValueError(f"Expected 1 or {batch_size} sigma values, got {sigma_vec.numel()}")
        return sigma_vec

    def _nearest_timestep(self, sigma_vec: torch.Tensor) -> torch.Tensor:
        sigma_safe = sigma_vec.clamp_min(self.sigma_eps)
        ref = self.vp_sigmas.to(device=sigma_safe.device, dtype=sigma_safe.dtype).clamp_min(self.sigma_eps)
        distance = (torch.log(ref).unsqueeze(0) - torch.log(sigma_safe).unsqueeze(1)).abs()
        return distance.argmin(dim=1).long()

    def _to_model_space(self, x: torch.Tensor, sigma_vec: torch.Tensor):
        if self.external_range == "minus_one_one":
            return x, sigma_vec
        if self.external_range == "zero_one":
            return x * 2.0 - 1.0, sigma_vec * 2.0
        raise ValueError(f"Unsupported external_range '{self.external_range}'")

    def _from_model_space(self, x: torch.Tensor) -> torch.Tensor:
        if self.external_range == "minus_one_one":
            return x
        if self.external_range == "zero_one":
            x = (x + 1.0) * 0.5
            if self.clamp_output:
                x = x.clamp(0.0, 1.0)
            return x
        raise ValueError(f"Unsupported external_range '{self.external_range}'")

    def _predict_x0(self, x: torch.Tensor, sigma) -> torch.Tensor:
        sigma_vec = self._sigma_vector(sigma, x.shape[0], x.device, x.dtype)
        x_model, sigma_model_vec = self._to_model_space(x, sigma_vec)
        sigma_view = sigma_model_vec.reshape((-1,) + (1,) * (x_model.dim() - 1))

        # Convert x = x0 + sigma * eps into the VP/DDPM state x_t.
        xt = x_model / torch.sqrt(1.0 + sigma_view ** 2)
        timesteps = self._nearest_timestep(sigma_model_vec)

        model_out = self.unet(xt, timesteps).sample
        alpha_bar = self.alpha_bar.to(device=x_model.device, dtype=x_model.dtype)[timesteps]
        alpha_view = alpha_bar.reshape((-1,) + (1,) * (x_model.dim() - 1))
        sqrt_alpha = torch.sqrt(alpha_view.clamp_min(1e-12))
        sqrt_one_minus = torch.sqrt((1.0 - alpha_view).clamp_min(1e-12))

        prediction_type = getattr(getattr(self.scheduler, "config", None), "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            x0 = (xt - sqrt_one_minus * model_out) / sqrt_alpha
        elif prediction_type == "sample":
            x0 = model_out
        elif prediction_type == "v_prediction":
            x0 = sqrt_alpha * xt - sqrt_one_minus * model_out
        else:
            raise NotImplementedError(f"Unsupported Diffusers prediction_type '{prediction_type}'")

        if self.clamp_output:
            x0 = x0.clamp(-1.0, 1.0)
        return self._from_model_space(x0)

    def tweedie(self, x, sigma=2e-3):
        return self._predict_x0(x, sigma)

    def score(self, x, sigma):
        sigma_vec = self._sigma_vector(sigma, x.shape[0], x.device, x.dtype)
        sigma_view = sigma_vec.reshape((-1,) + (1,) * (x.dim() - 1)).clamp_min(self.sigma_eps)
        x0 = self._predict_x0(x, sigma_vec)
        score = (x0 - x) / (sigma_view ** 2)
        if self.external_range == "zero_one":
            return score
        return score


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, ckpt, train=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    _, _ = model.load_state_dict(sd, strict=False)

    model.cuda()

    if train:
        model.train()
    else:
        model.eval()

    return model
