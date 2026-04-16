from .registry import register_operator
from .base import Operator
import torch
import torch.nn.functional as F
#from .fastmri_utils import fft2c_new
#import cv2
import numpy as np
from copy import deepcopy

import math
import torch

try:
    # If your fastmri_utils already has the inverse, this is the best option.
    from .fastmri_utils import fft2c_new, ifft2c_new
except ImportError:
    # Fallback: implement a fastMRI-style centered IFFT.
    # IMPORTANT: this assumes fft2c_new uses the usual centered + norm="ortho" convention.
    from .fastmri_utils import fft2c_new

    def ifft2c_new(data: torch.Tensor) -> torch.Tensor:
        """
        Centered 2D IFFT (fastMRI-style), intended to be the inverse of fft2c_new.

        Args:
            data: real-view complex tensor (..., 2) with last dim (real, imag)
        Returns:
            real-view complex tensor (..., 2)
        """
        if data.shape[-1] != 2:
            raise ValueError(f"ifft2c_new expects last dim=2, got shape {tuple(data.shape)}")

        x = torch.view_as_complex(data)
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return torch.view_as_real(x)

# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='phase_retrieval')
class PhaseRetrieval(Operator):
    def __init__(self, oversample=0.0, resolution=256, sigma=0.05):
        super().__init__(sigma)
        self.pad = int((oversample / 8.0) * resolution)

    def __call__(self, x):
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))
        amplitude = fft2_m.abs()
        return amplitude
    
    def forward_complex(self, x01: torch.Tensor) -> torch.Tensor:
        # MUST match __call__: pad -> fft2c_new
        x = F.pad(x01, (self.pad, self.pad, self.pad, self.pad))
        x_c = x.to(torch.complex64) if not torch.is_complex(x) else x
        U = torch.view_as_complex(fft2c_new(torch.view_as_real(x_c)))
        return U

    def adjoint_complex(self, p: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        # MUST match __call__: ifft2c_new -> crop
        p_c = p.to(torch.complex64) if not torch.is_complex(p) else p
        x_pad_c = torch.view_as_complex(ifft2c_new(torch.view_as_real(p_c)))  # complex padded image
        H, W = out_hw
        x = x_pad_c[..., self.pad:self.pad + H, self.pad:self.pad + W]
        return x.real


    @torch.no_grad()
    def proj_amplitude(
        self,
        x: torch.Tensor,
        y_amp: torch.Tensor,
        *,
        tau: float = float("inf"),
        eps: float = 1e-8,
        enforce_real: bool = True,
        clamp_x: bool = True,
        clamp01: bool = True,
    ) -> torch.Tensor:
        """
        Fourier-magnitude prox/projection consistent with PhaseRetrieval.__call__.

        Your forward model is:
            y_amp ≈ | FFT2c( pad( (x + 1)/2 ) ) |
        where x is in [-1,1] and pad is self.pad. :contentReference[oaicite:2]{index=2}

        This routine returns x_proj in [-1,1] whose oversampled FFT magnitude is
        (hard) set to y_amp, or (soft) pulled toward y_amp.

        Args:
            x:        real tensor in [-1,1], shape (..., H, W)
            y_amp:    measured amplitude, shape broadcastable to padded FFT output
            tau:
              - tau = inf: hard projection |U| := y_amp
              - tau < inf: prox of 0.5 * || |U| - y_amp ||_2^2 with step tau:
                           |U| := (|U| + tau*y_amp)/(1+tau)
            eps:      numerical stabilizer for phase normalization
            enforce_real: take real(...) after iFFT (recommended for real images)
            clamp_x:  clamp final output to [-1,1] (recommended for diffusion priors)
            clamp01:  clamp intermediate [0,1] image before/after iFFT (recommended)
        Returns:
            x_proj: real tensor in [-1,1], same shape as x
        """

        # 1) Map [-1,1] -> [0,1] exactly like __call__
        x01 = x * 0.5 + 0.5
        if clamp01:
            x01 = x01.clamp(0.0, 1.0)

        # 2) Pad exactly like __call__
        if self.pad > 0:
            x01 = F.pad(x01, (self.pad, self.pad, self.pad, self.pad))

        # 3) FFT (same convention as __call__)
        x01_c = x01.to(torch.complex64) if not torch.is_complex(x01) else x01
        U = torch.view_as_complex(fft2c_new(torch.view_as_real(x01_c)))  # complex spectrum

        # 4) Magnitude update (hard or soft prox)
        mag = U.abs()
        y = y_amp.to(device=mag.device, dtype=mag.dtype)

        if math.isinf(tau):
            mag_new = y
        else:
            mag_new = (mag + tau * y) / (1.0 + tau)

        # keep phase, replace magnitude
        U_new = U * (mag_new / (mag + eps))

        # 5) Inverse FFT
        x01_new_c = torch.view_as_complex(ifft2c_new(torch.view_as_real(U_new)))
        x01_new = x01_new_c.real if enforce_real else x01_new_c

        # 6) Crop back to original resolution
        if self.pad > 0:
            x01_new = x01_new[..., self.pad:-self.pad, self.pad:-self.pad]

        # 7) Optional [0,1] clamp in image domain
        if clamp01 and (not torch.is_complex(x01_new)):
            x01_new = x01_new.clamp(0.0, 1.0)

        # 8) Map back [0,1] -> [-1,1]
        x_new = x01_new * 2.0 - 1.0
        if clamp_x and (not torch.is_complex(x_new)):
            x_new = x_new.clamp(-1.0, 1.0)

        return x_new


    def get_more_aligned_option(self, template, options):
        template = deepcopy(template)
        options = deepcopy(options)

        template = ((template.detach().cpu().numpy()+1) /
                    2 * 255).astype(np.uint8)

        for idx in range(len(options)):
            options[idx] = ((options[idx].detach().cpu().numpy()+1)/2
                            * 255).astype(np.uint8)
        
        ''' Use pixel similarity'''
        result_lst = []
        for option in options:
            diff = ((template - option)**2).sum()
            result_lst.append(diff)
            
        min_idx = result_lst.index(min(result_lst))
        return min_idx
    
    