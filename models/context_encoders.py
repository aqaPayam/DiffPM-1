# models/context_encoders.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNContextEncoder(nn.Module):
    """
    Wrapper around RNN / GRU / LSTM that takes input of shape (B, S, D_in)
    and returns (B, S, H) context per time step.
    """
    def __init__(
        self,
        cell_type: str,          # "rnn" | "gru" | "lstm"
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        cell_type = cell_type.lower()
        if cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                nonlinearity="tanh",
                batch_first=False,  # we will feed (S, B, D)
            )
        elif cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=False,
            )
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=False,
            )
        else:
            raise ValueError(f"Unsupported cell_type '{cell_type}'")

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D_in)
        Returns:
            context: (B, S, H)
        """
        B, S, D_in = x.shape
        # RNN expects (S, B, D_in)
        x_rnn = x.permute(1, 0, 2)  # (S, B, D_in)
        out, _ = self.rnn(x_rnn)    # (S, B, H)
        out = out.permute(1, 0, 2)  # (B, S, H)
        return out


class TCNBlock(nn.Module):
    """
    A simple 1D TCN-style block over sequences.
    Input:  (B, S, C)
    Output: (B, S, C)
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal-ish
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, C)
        Returns:
            (B, S, C)
        """
        # Conv1d expects (B, C, S)
        x_in = x.transpose(1, 2)        # (B, C, S)
        y = self.conv(x_in)             # (B, C, S + pad_effect)
        # Chop off extra timesteps to keep length S (causal style)
        y = y[:, :, : x_in.size(2)]     # (B, C, S)
        y = y.transpose(1, 2)           # back to (B, S, C)
        y = F.relu(y)
        y = self.dropout(y)
        # Residual + norm
        return self.norm(x + y)


class TCNContextEncoder(nn.Module):
    """
    TCN over the sequence of window embeddings.
    Input:  (B, S, D_in)
    Output: (B, S, H)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TCNBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D_in)
        Returns:
            (B, S, H)
        """
        x = self.input_proj(x)  # (B, S, H)
        x = self.tcn(x)         # (B, S, H)
        return x


class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding for Transformer.
    Expects (S, B, D_model) / returns same shape.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, D)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (S, B, D)
        Returns:
            (S, B, D)
        """
        S, B, D = x.shape
        return x + self.pe[:S]


class TransformerContextEncoder(nn.Module):
    """
    Transformer encoder over the sequence of window embeddings.
    Input:  (B, S, D_in)
    Output: (B, S, H)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=False,  # we use (S, B, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D_in)
        Returns:
            (B, S, H)
        """
        B, S, D_in = x.shape
        x = self.input_proj(x)         # (B, S, H)
        x = x.permute(1, 0, 2)         # (S, B, H)
        x = self.pos_encoding(x)       # (S, B, H)
        x = self.encoder(x)            # (S, B, H)
        x = x.permute(1, 0, 2)         # (B, S, H)
        return x


def build_context_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int = 1,
    tcn_kernel_size: int = 3,
    transformer_nhead: int = 4,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Factory for context encoders over sequences of window embeddings.

    All encoders take input of shape (B, S, input_dim) and return (B, S, hidden_dim).

    encoder_type: "rnn" | "gru" | "lstm" | "tcn" | "transformer"
    """
    encoder_type = encoder_type.lower()
    if encoder_type in {"rnn", "gru", "lstm"}:
        return RNNContextEncoder(
            cell_type=encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif encoder_type == "tcn":
        return TCNContextEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
    elif encoder_type == "transformer":
        return TransformerContextEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=transformer_nhead,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown context encoder_type '{encoder_type}'")
