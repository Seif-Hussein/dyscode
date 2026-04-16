"""Trajectory distance utilities for your samplers.

This module implements two practical, simulation-friendly proxies for comparing
*distributions of iterates* between algorithms:

1) Coupled W2 upper bound (pathwise):
      U_k(A,B) = mean_i || s_{A,i}^{(k)} - s_{B,i}^{(k)} ||^2
   where i indexes the same image across both runs.
   This is an *upper bound* on W2^2 for the coupling that pairs index i with i.

2) Sliced-W2^2 (distributional):
      SW2_k(A,B) = average over random 1D projections of the 1D W2^2.
   This behaves like a Wasserstein distance but is computable in high dimension.

You feed it the `sampler.get_trace()` dicts produced by the instrumented ADMM/DYS.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F


def _as_tensor(x: torch.Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.to(dtype)
    return x


@torch.no_grad()
def coupled_w2_upper_sq(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean per-sample squared Euclidean distance between paired samples.

    Interpretable as a coupling-based *upper bound* on W2^2.
    A,B: [N, ...] with matching shapes.
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {tuple(A.shape)} vs {tuple(B.shape)}")
    diff = (A - B).flatten(1)
    return float(diff.pow(2).sum(dim=1).mean().detach().cpu())


@torch.no_grad()
def sliced_w2_sq(
    A: torch.Tensor,
    B: torch.Tensor,
    num_proj: int = 128,
    seed: int = 0,
) -> float:
    """Sliced Wasserstein-2 squared between two empirical distributions.

    A,B: [N, d] or [N, C, H, W]. We flatten over non-batch dims.

    Implementation: random directions theta_l, project, sort, average squared diffs.
    """
    if A.shape[0] != B.shape[0]:
        # This is OK in principle, but we keep it strict for reproducibility.
        raise ValueError(f"Batch mismatch: {A.shape[0]} vs {B.shape[0]}")

    A = A.flatten(1)
    B = B.flatten(1)

    N, d = A.shape
    device = A.device

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    theta = torch.randn(num_proj, d, generator=g, device=device)
    theta = theta / (theta.norm(dim=1, keepdim=True) + 1e-12)

    proj_A = A @ theta.T  # [N, L]
    proj_B = B @ theta.T

    proj_A, _ = proj_A.sort(dim=0)
    proj_B, _ = proj_B.sort(dim=0)

    return float((proj_A - proj_B).pow(2).mean().detach().cpu())


@torch.no_grad()
def compare_traces(
    trace_a: Dict[str, List[torch.Tensor]],
    trace_b: Dict[str, List[torch.Tensor]],
    keys: Tuple[str, ...] = ("x_k", "z_k", "z_e"),
    method: str = "swd",
    downsample_to: Optional[int] = None,
    num_proj: int = 128,
    seed: int = 0,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """Compute per-iteration distances between two trace dicts.

    Parameters
    - trace_a, trace_b: from sampler.get_trace()
    - keys: which states to compare. Common:
        * 'x_k'  : ADMM x after ML step / DYS prox point
        * 'z_k'  : denoised output
        * 'z_e'  : ADMM denoiser input (x+u); DYS uses 'r_k' instead
    - method: 'swd' or 'coupled'
    - downsample_to: if not None, downsample tensors to [downsample_to,downsample_to]
    """
    if method not in {"swd", "coupled"}:
        raise ValueError("method must be 'swd' or 'coupled'")

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    out: Dict[str, List[float]] = {}

    for k in keys:
        if k not in trace_a:
            raise KeyError(f"'{k}' missing from trace_a")
        if k not in trace_b:
            raise KeyError(f"'{k}' missing from trace_b")

        seq_a = trace_a[k]
        seq_b = trace_b[k]
        T = min(len(seq_a), len(seq_b))

        vals: List[float] = []
        for t in range(T):
            A = _as_tensor(seq_a[t], device=dev, dtype=torch.float32)
            B = _as_tensor(seq_b[t], device=dev, dtype=torch.float32)

            if downsample_to is not None and A.ndim == 4:
                A = F.interpolate(A, size=(downsample_to, downsample_to), mode="area")
                B = F.interpolate(B, size=(downsample_to, downsample_to), mode="area")

            if method == "coupled":
                vals.append(coupled_w2_upper_sq(A, B))
            else:
                vals.append(sliced_w2_sq(A, B, num_proj=num_proj, seed=seed))

        out[k] = vals

    return out


def summarize_curve(curve: List[float]) -> Dict[str, float]:
    """Small helper: summarize a per-iteration distance curve."""
    if len(curve) == 0:
        return {"mean": float("nan"), "max": float("nan"), "final": float("nan")}
    t = torch.tensor(curve, dtype=torch.float64)
    return {
        "mean": float(t.mean().item()),
        "max": float(t.max().item()),
        "final": float(t[-1].item()),
    }
