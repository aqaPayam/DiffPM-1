import torch
import torch.nn as nn
from typing import Optional, Tuple


class WindowContextLSTM(nn.Module):
    """
    LSTM that processes sequences of windows and produces a context vector
    per window, plus utilities for step-by-step use during sampling.

    Two main usage patterns:

    1) TRAINING / OFFLINE:
       - You have a sequence of windows per series: x_windows (B, N, W, D)
       - You call forward(...) to get a context vector for each window:
           ctx_seq: (B, N, H_ctx)

    2) SAMPLING / ONLINE:
       - You generate windows one-by-one.
       - You call step(...) with each new window to update the hidden state.
       - At each step, you get a context vector to feed into the diffusion UNet.
    """

    def __init__(
        self,
        in_channels: int,
        window_size: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: number of features per time step (D)
            window_size: number of time steps in each window (W)
            hidden_dim: LSTM hidden size per direction
            num_layers: number of LSTM layers
            bidirectional: if True, use a bidirectional LSTM
            dropout: dropout between LSTM layers (ignored if num_layers=1)
        """
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.input_dim = in_channels * window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,   # inputs/outputs: (B, N, F)
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def context_dim(self) -> int:
        """Dimensionality of the context vector per window."""
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def forward(self, x_windows: torch.Tensor) -> torch.Tensor:
        """
        Process a full sequence of windows for each series.

        Args:
            x_windows: Tensor of shape (B, N, W, D)
                B = batch size
                N = number of windows per series
                W = window_size
                D = in_channels

        Returns:
            ctx_seq: Tensor of shape (B, N, H_ctx)
                H_ctx = context_dim = hidden_dim * (2 if bidirectional else 1)
                A context vector for each window position in the sequence.
        """
        B, N, W, D = x_windows.shape
        assert W == self.window_size, f"expected window_size={self.window_size}, got {W}"
        assert D == self.in_channels, f"expected in_channels={self.in_channels}, got {D}"

        # Flatten each window: (B, N, W, D) -> (B, N, W*D)
        x_flat = x_windows.reshape(B, N, W * D)  # features per step = W*D

        # LSTM output: (B, N, num_directions * hidden_dim)
        out, _ = self.lstm(x_flat)

        # This is our per-window context sequence
        ctx_seq = out  # (B, N, H_ctx)
        return ctx_seq

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state for step-by-step usage.

        Args:
            batch_size: B
            device: optional device

        Returns:
            (h0, c0): each of shape (num_layers * num_directions, B, hidden_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        return h0, c0

    def step(
        self,
        x_window: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process a single window (per batch element) and update hidden state.

        This is meant for **sequential generation**:
        - At each step, you feed the newly generated window,
        - You get a context vector for the current window,
        - You get an updated hidden state for the next step.

        Args:
            x_window: Tensor of shape (B, W, D)
            hidden: optional (h, c) from previous step

        Returns:
            ctx: Tensor of shape (B, H_ctx)  # context for this window
            new_hidden: (h_new, c_new)       # updated hidden state
        """
        B, W, D = x_window.shape
        assert W == self.window_size, f"expected window_size={self.window_size}, got {W}"
        assert D == self.in_channels, f"expected in_channels={self.in_channels}, got {D}"

        # Flatten each window: (B, W, D) -> (B, 1, W*D)
        x_flat = x_window.reshape(B, 1, W * D)

        # If no hidden is provided, initialize
        if hidden is None:
            hidden = self.init_hidden(batch_size=B, device=x_window.device)

        out, new_hidden = self.lstm(x_flat, hidden)
        # out: (B, 1, H_ctx)
        ctx = out[:, 0, :]  # (B, H_ctx)

        return ctx, new_hidden
