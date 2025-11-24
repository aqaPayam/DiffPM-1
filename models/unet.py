import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embeddings import sinusoidal_embedding as get_sinusoidal_embedding


class ResBlock1D(nn.Module):
    def __init__(self, c: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=1)
        self.gn1   = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv1d(c, c, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, c)
        # FiLM-style additive bias from conditioning embedding
        self.film  = nn.Linear(emb_dim, c)  # produce an additive bias

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, W)
            emb: (B, emb_dim)
        Returns:
            (B, C, W)
        """
        h = self.conv1(x)
        h = self.gn1(h)

        # Project embedding to per-channel bias and broadcast over time
        bias = self.film(emb).unsqueeze(-1)  # (B, C, 1)
        h = h + bias
        h = F.silu(h)

        h = self.conv2(h)
        h = self.gn2(h)
        return x + h  # residual connection


class ImprovedDiffusionUNet1D(nn.Module):
    """
    1D diffusion backbone for windowed time series.

    Conditioning:
      - diffusion timestep t
      - window start_idx
      - full series_len
      - (optionally) an additional context vector, e.g. from an LSTM over windows

    If use_context=False, behavior is identical to the original version.
    """

    def __init__(
        self,
        in_channels: int,
        window_size: int,
        time_emb_dim: int,
        base_channels: int,
        n_res_blocks: int,
        use_context: bool = False,
        context_dim: Optional[int] = None,
    ):
        """
        Args:
            in_channels:   D, number of feature channels
            window_size:   W, length of each 1D window
            time_emb_dim:  E, embedding dim for time / conditioning
            base_channels: C, internal feature channels
            n_res_blocks:  number of residual blocks
            use_context:   if True, expect an extra context vector per sample
            context_dim:   dimension of the external context vector.
                           If None and use_context=True, defaults to time_emb_dim.
        """
        super().__init__()
        self.in_channels = in_channels
        self.W = window_size
        self.base_c = base_channels
        self.time_emb_dim = time_emb_dim

        self.use_context = use_context
        if self.use_context and context_dim is None:
            context_dim = time_emb_dim
        self.context_dim = context_dim

        # input conv: multi-channel in → C channels
        self.init_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        # MLP to process sinusoidal time embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # MLP to fuse start_idx & series_len embeddings
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Optional context projection MLP (e.g. from LSTM hidden state)
        if self.use_context:
            self.context_mlp = nn.Sequential(
                nn.Linear(self.context_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.context_mlp = None

        # stack of residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock1D(base_channels, time_emb_dim)
            for _ in range(n_res_blocks)
        ])

        # output conv: C channels → multi-channel out
        self.out_conv = nn.Conv1d(base_channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        start_idx: torch.Tensor,
        series_len: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, W, D) — normalized, noised window (multi-channel)
            t:          (B,)     — diffusion timestep indices
            start_idx:  (B,)     — window start indices in the full series
            series_len: (B,)     — full series lengths
            context:    (B, C_ctx), optional external context (e.g. LSTM hidden)

        Returns:
            eps_pred:   (B, W, D) — predicted noise (or v, depending on outer model)
        """
        # Expect input shape
        B, W, D = x.shape
        assert W == self.W, f"expected width {self.W}, got {W}"
        assert D == self.in_channels, f"expected {self.in_channels} channels, got {D}"

        if self.use_context:
            assert context is not None, "use_context=True but no context tensor was provided"
            assert context.dim() == 2 and context.size(0) == B, \
                f"context must be (B, C_ctx), got {tuple(context.shape)}"

        # 1) Input projection: (B, W, D) → (B, D, W) → init_conv → (B, C, W)
        h = x.permute(0, 2, 1)              # (B, D, W)
        h = self.init_conv(h)               # (B, C, W)

        # 2) Time embedding
        t_emb = get_sinusoidal_embedding(t.unsqueeze(-1), self.time_emb_dim)  # (B, E)
        t_emb = self.time_mlp(t_emb)                                          # (B, E)

        # 3) Position embedding (start_idx & series_len)
        se = get_sinusoidal_embedding(start_idx.unsqueeze(-1), self.time_emb_dim)  # (B, E)
        le = get_sinusoidal_embedding(series_len.unsqueeze(-1), self.time_emb_dim) # (B, E)
        cond = torch.cat([se, le], dim=-1)    # (B, 2E)
        cond_emb = self.cond_mlp(cond)        # (B, E)

        # 4) Optional context embedding
        if self.use_context:
            ctx_emb = self.context_mlp(context)    # (B, E)
            emb = t_emb + cond_emb + ctx_emb       # (B, E)
        else:
            emb = t_emb + cond_emb                 # (B, E)

        # 5) Residual blocks
        for block in self.res_blocks:
            h = block(h, emb)                 # (B, C, W)

        # 6) Output projection: (B, C, W) → (B, in_channels, W) → (B, W, D)
        out = self.out_conv(F.silu(h))        # (B, D, W)
        eps_pred = out.permute(0, 2, 1)       # (B, W, D)

        return eps_pred
