import math

import torch
import torch.nn.functional as F

from .base import Operator
from .registry import register_operator


@register_operator(name='transmission_ct')
class TransmissionCT(Operator):
    """
    Differentiable parallel-beam transmission CT operator.

    The optimization variable lives in the normalized image space [-1, 1].
    We map it to a nonnegative attenuation image mu(x) before applying the
    projector:

        mu(x) = mu_min + ((x + 1) / 2) * (mu_max - mu_min)

    The forward operator then returns line integrals z = A mu(x).
    Measurements are Poisson photon counts:

        y ~ Poisson(I0 * exp(-z))
    """

    def __init__(self,
                 resolution=512,
                 num_angles=60,
                 num_detectors=None,
                 angle_offset_deg=0.0,
                 detector_scale=1.0,
                 attenuation_min=0.0,
                 attenuation_max=1.0,
                 eta=1.0,
                 I0=1.0e4,
                 measurement_mode='poisson',
                 channels=1,
                 clamp_input=True,
                 sigma=1.0,
                 device='cuda'):
        super().__init__(sigma)
        self.resolution = int(resolution)
        self.num_angles = int(num_angles)
        self.num_detectors = int(num_detectors) if num_detectors is not None else int(resolution)
        self.angle_offset_deg = float(angle_offset_deg)
        self.detector_scale = float(detector_scale)
        self.attenuation_min = float(attenuation_min)
        self.attenuation_max = float(attenuation_max)
        self.eta = float(eta)
        self.I0 = float(I0)
        self.measurement_mode = str(measurement_mode).lower()
        self.channels = int(channels)
        self.clamp_input = bool(clamp_input)
        self.device = device

        angles = torch.arange(self.num_angles, dtype=torch.float32)
        angles = angles * (math.pi / max(1, self.num_angles))
        angles = angles + math.radians(self.angle_offset_deg)
        self.angles = angles

    def _angles_on(self, device, dtype):
        return self.angles.to(device=device, dtype=dtype)

    def _attenuation_image(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x + 1.0) * 0.5
        if self.clamp_input:
            x01 = x01.clamp(0.0, 1.0)
        return self.attenuation_min + x01 * (self.attenuation_max - self.attenuation_min)

    @staticmethod
    def _resize_detector_axis(sino: torch.Tensor, num_detectors: int) -> torch.Tensor:
        if sino.shape[-1] == num_detectors:
            return sino
        b, c, a, d = sino.shape
        flat = sino.reshape(b * c * a, 1, d)
        flat = F.interpolate(flat, size=num_detectors, mode="linear", align_corners=False)
        return flat.reshape(b, c, a, num_detectors)

    def _forward_mu(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W] tensor, got {tuple(mu.shape)}")

        b, c, h, w = mu.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")

        angles = self._angles_on(mu.device, mu.dtype)
        projections = []
        dx = 2.0 / max(h, 1)

        for angle in angles:
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            theta = torch.tensor(
                [[cos_a, -sin_a, 0.0],
                 [sin_a,  cos_a, 0.0]],
                device=mu.device,
                dtype=mu.dtype,
            ).unsqueeze(0).expand(b, -1, -1)
            grid = F.affine_grid(theta, mu.size(), align_corners=False)
            rotated = F.grid_sample(mu, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            proj = rotated.sum(dim=-2) * dx * self.detector_scale
            projections.append(proj)

        sino = torch.stack(projections, dim=-2)
        sino = self._resize_detector_axis(sino, self.num_detectors)
        return sino


    def __call__(self, x):
        mu = self._attenuation_image(x)
        return self._forward_mu(mu)

    def incident_counts(self, reference: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(self.I0):
            i0 = self.I0.to(device=reference.device, dtype=reference.dtype)
        else:
            i0 = torch.as_tensor(self.I0, device=reference.device, dtype=reference.dtype)
        while i0.dim() < reference.dim():
            i0 = i0.unsqueeze(0)
        return torch.broadcast_to(i0, reference.shape)

    def measure(self, x):
        z = self(x)
        rate = self.incident_counts(z) * torch.exp(-z).clamp_min(0.0)
        rate = rate.clamp_min(0.0).clamp_max(1.0e12)
        if self.measurement_mode == 'expected':
            return rate
        if self.measurement_mode != 'poisson':
            raise ValueError(
                f"Unsupported transmission CT measurement_mode={self.measurement_mode!r}. "
                "Expected one of: 'poisson', 'expected'."
            )
        return torch.poisson(rate)

    def loss(self, x, y):
        z = self(x)
        i0 = self.incident_counts(z)
        return self.eta * (i0 * torch.exp(-z) + y * z).sum()
