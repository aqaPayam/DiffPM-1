import torch
import math
from typing import Tuple

def cosine_beta_schedule(
    num_timesteps: int,
    s: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cosine schedule for betas, from Nichol & Dhariwal (2021).

    Args:
        num_timesteps (int): Number of diffusion timesteps T.
        s (float): Small offset to prevent singularities.

    Returns:
        betas (Tensor[T]), alphas (Tensor[T]), alpha_bars (Tensor[T]).
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps)
    f = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.clamp(betas, max=0.999)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
