import math
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
        self.film  = nn.Linear(emb_dim, c)  # produce an additive bias

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, C, W)
        emb: (B, emb_dim)
        """
        h = self.conv1(x)
        h = self.gn1(h)
        bias = self.film(emb).unsqueeze(-1)  # (B, C, 1)
        h = h + bias
        h = F.silu(h)

        h = self.conv2(h)
        h = self.gn2(h)
        return x + h  # residual


class ImprovedDiffusionUNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_size: int,
        time_emb_dim: int,
        base_channels: int,
        n_res_blocks: int,
        # NEW: optional context from an external model (e.g. LSTM)
        context_dim: int = 0,
    ):
        """
        1D UNet-style backbone for diffusion on windows.

        Args (original):
            in_channels:   number of feature channels (D)
            window_size:   window length (W)
            time_emb_dim:  embedding dim for time / conditioning
            base_channels: base channel count (C)
            n_res_blocks:  number of ResBlock1D stacks

        New:
            context_dim:   dimensionality of an optional context vector
                           (e.g. from an LSTM over previous windows).
                           If 0, no context is used and behavior matches
                           the original implementation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.W = window_size
        self.base_c = base_channels
        self.time_emb_dim = time_emb_dim
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

        # NEW: MLP to project external context into the same emb space
        if context_dim > 0:
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, time_emb_dim),
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
        # NEW: optional context vector, e.g. from an LSTM
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x:          (B, W, D) — normalized, noised window (multi-channel)
        t:          (B,)     — diffusion timestep indices
        start_idx:  (B,)     — window start indices (in full series)
        series_len: (B,)     — full series lengths
        context:    (B, context_dim) or None
                    Extra conditioning signal, e.g. from an LSTM summarizing
                    previous windows. If None and context_dim == 0,
                    behavior matches the original implementation.

        returns:
        eps_pred:   (B, W, D) — predicted noise (multi-channel)
        """
        # Expect input shape
        B, W, D = x.shape
        assert W == self.W, f"expected width {self.W}, got {W}"
        assert D == self.in_channels, f"expected {self.in_channels} channels, got {D}"

        # 1) Input projection: (B, W, D) → (B, D, W) → init_conv → (B, C, W)
        h = x.permute(0, 2, 1)              # (B, D, W)
        h = self.init_conv(h)               # (B, C, W)

        # 2) Time embedding
        t_emb = get_sinusoidal_embedding(t.unsqueeze(-1), self.time_emb_dim)  # (B, E)
        t_emb = self.time_mlp(t_emb)                                          # (B, E)

        # 3) Position embedding
        se = get_sinusoidal_embedding(start_idx.unsqueeze(-1), self.time_emb_dim)
        le = get_sinusoidal_embedding(series_len.unsqueeze(-1), self.time_emb_dim)
        cond = torch.cat([se, le], dim=-1)    # (B, 2E)
        cond_emb = self.cond_mlp(cond)        # (B, E)

        # 4) Optional context embedding
        if self.context_mlp is not None:
            if context is None:
                raise ValueError(
                    "UNet was initialized with context_dim > 0, "
                    "but no `context` tensor was provided to forward()."
                )
            ctx_emb = self.context_mlp(context)   # (B, E)
        else:
            # No context used (backward-compatible mode)
            ctx_emb = 0.0

        # 5) Combine time + pos + (optional) context
        emb = t_emb + cond_emb + ctx_emb         # (B, E)

        # 6) Residual blocks
        for block in self.res_blocks:
            h = block(h, emb)                    # (B, C, W)

        # 7) Output projection: (B, C, W) → (B, in_channels, W)
        out = self.out_conv(F.silu(h))           # (B, D, W)
        eps_pred = out.permute(0, 2, 1)          # (B, W, D)

        return eps_pred
