# training/train.py

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.diffusion_model import DiffusionModel
from models.context_encoders import build_context_encoder


def train_model(
    dataset,
    window_size: int,
    time_emb_dim: int,
    base_channels: int,
    n_res_blocks: int,
    timesteps: int,
    D: int,
    s: float,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    show_detail: bool = False,
    sample_interval: int = 5,
    # sequence-context options
    use_seq_context: bool = False,
    context_encoder_type: str = "lstm",   # "rnn" | "gru" | "lstm" | "tcn" | "transformer"
    context_hidden_dim: int = 128,
    context_num_layers: int = 1,
    # optional extra hyperparams for some encoders
    tcn_kernel_size: int = 3,
    transformer_nhead: int = 4,
    context_dropout: float = 0.0,
):
    """
    Train a 1D DiffusionModel on the provided dataset.

    Returns:
        model:          trained DiffusionModel
        loss_history:   list of epoch losses
        context_encoder (or None):
            - None if use_seq_context=False
            - nn.Module if use_seq_context=True
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Diffusion model (UNet) is unchanged; just tell it if we use context
    model = DiffusionModel(
        window_size=window_size,
        in_channels=D,
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        n_res_blocks=n_res_blocks,
        timesteps=timesteps,
        s=s,
        use_context=use_seq_context,
        context_dim=context_hidden_dim if use_seq_context else None,
    ).to(device)

    # Optional context encoder over sequences of windows
    if use_seq_context:
        context_encoder = build_context_encoder(
            encoder_type=context_encoder_type,
            input_dim=D,                      # we use mean over time -> D dims
            hidden_dim=context_hidden_dim,
            num_layers=context_num_layers,
            tcn_kernel_size=tcn_kernel_size,
            transformer_nhead=transformer_nhead,
            dropout=context_dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(context_encoder.parameters()),
            lr=lr
        )
    else:
        context_encoder = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    data_length = len(dataset)

    for epoch in range(1, epochs + 1):
        model.train()
        if context_encoder is not None:
            context_encoder.train()
        running_loss = 0.0

        for batch in dataloader:
            if not use_seq_context:
                # ===== STANDARD MODE: independent windows (original behavior) =====
                x0 = batch['window'].to(device).float()       # (B, W, D)
                start_idx = batch['start_idx'].to(device)     # (B,)
                series_len = batch['series_len'].to(device)   # (B,)

                optimizer.zero_grad()
                loss = model(x0, start_idx, series_len)       # context=None
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                # ===== SEQUENCE CONTEXT MODE =====
                # Expect:
                #   window:     (B, S, W, D)
                #   start_idx:  (B, S) or (B,)
                #   series_len: (B, S) or (B,)
                x0_seq = batch['window'].to(device).float()   # (B, S, W, D)
                assert x0_seq.dim() == 4, \
                    f"With use_seq_context=True, expected 'window' (B,S,W,D), got {tuple(x0_seq.shape)}"

                B, S, W, D_ = x0_seq.shape
                assert W == window_size, f"Expected window_size={window_size}, got {W}"
                assert D_ == D, f"Expected D={D}, got {D_}"

                start_idx = batch['start_idx'].to(device)
                series_len = batch['series_len'].to(device)

                # Normalize shapes of start_idx and series_len to (B, S)
                if start_idx.dim() == 1:
                    start_idx = start_idx.unsqueeze(1).expand(B, S)
                elif start_idx.dim() == 2:
                    assert start_idx.shape == (B, S)
                else:
                    raise ValueError(f"start_idx must be (B,) or (B,S), got {tuple(start_idx.shape)}")

                if series_len.dim() == 1:
                    series_len = series_len.unsqueeze(1).expand(B, S)
                elif series_len.dim() == 2:
                    assert series_len.shape == (B, S)
                else:
                    raise ValueError(f"series_len must be (B,) or (B,S), got {tuple(series_len.shape)}")

                # Window embedding: simple mean over time dimension -> (B, S, D)
                window_repr = x0_seq.mean(dim=2)              # (B, S, D)

                # context_seq: (B, S, H)
                context_seq = context_encoder(window_repr)    # generic encoder

                # Flatten for DiffusionModel
                x0_flat = x0_seq.reshape(B * S, W, D)         # (B*S, W, D)
                start_flat = start_idx.reshape(B * S)         # (B*S,)
                series_flat = series_len.reshape(B * S)       # (B*S,)
                context_flat = context_seq.reshape(B * S, context_hidden_dim)  # (B*S, H)

                optimizer.zero_grad()
                loss = model(x0_flat, start_flat, series_flat, context=context_flat)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

        # === Sampling & plotting at intervals (only in standard mode for now) ===
        if show_detail and (epoch % sample_interval == 0) and (not use_seq_context):
            # pick random index in [0, data_length)
            idx = torch.randint(0, data_length, (1,)).item()
            print("idx is : ", idx)
            example = dataset[idx]
            # example['window']: (W, D)
            real_window = example['window'].unsqueeze(0).to(device).float()
            # real_window shape: (1, W, D)
            start_ex = example['start_idx'].unsqueeze(0).to(device)
            len_ex = example['series_len'].unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                sample = model.sample(start_ex, len_ex, device=device)
                # sample shape: (1, W, D)

            real_np = real_window.squeeze(-1).squeeze(0).cpu().numpy()   # (W,)
            sample_np = sample.squeeze(-1).squeeze(0).cpu().numpy()      # (W,)

            plt.figure(figsize=(8, 4))
            plt.plot(real_np, label='Real Data')
            plt.plot(sample_np, label='Sampled Data', alpha=0.7)
            plt.title(f'Epoch {epoch} - Real vs. Sampled (idx={idx})')
            plt.legend()
            plt.tight_layout()
            plt.show()

    # === After training: loss plot & full-series sampling (standard mode only) ===
    if show_detail:
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, marker='o')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.tight_layout()
        plt.show()

        if not use_seq_context:
            model.eval()
            with torch.no_grad():
                start_full = torch.tensor([1],   dtype=torch.long, device=device)
                end_full   = torch.tensor([175], dtype=torch.long, device=device)
                shift = min(window_size - 1, 9)

                full_sample = model.sample_total(
                    start_full,
                    end_full,
                    shift,
                    min_value=-10,
                    max_value=10,
                    device=device
                )

            full_np = full_sample.squeeze(0).cpu().numpy()  # (L, D) or (L,)

            plt.figure(figsize=(10, 4))
            if full_np.ndim == 1 or full_np.shape[1] == 1:
                series = full_np[:, 0] if full_np.ndim > 1 else full_np
                plt.plot(series, label='Sampled')
            else:
                for d in range(full_np.shape[1]):
                    plt.plot(full_np[:, d], alpha=0.7, label=f'Channel {d}')
            plt.title("Full-Series Sampled Output (0 → 175, shift=1)")
            plt.xlabel("Time index")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return model, loss_history, context_encoder
