import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.diffusion_model import DiffusionModel


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
    # NEW: LSTM-related options
    use_lstm: bool = False,
    lstm_hidden_dim: int = 128,
    lstm_num_layers: int = 1,
):
    """
    Train a 1D DiffusionModel on the provided dataset.

    Two modes:

    1) Standard (use_lstm=False, default):
       - dataset is expected to yield dicts with:
           'window':     (B, W, D) per batch
           'start_idx':  (B,)
           'series_len': (B,)
       - This is identical to the original behavior.

    2) LSTM mode (use_lstm=True):
       - dataset is expected to yield dicts where 'window' is a *sequence of
         windows* for each sample in the batch:
           'window':     (B, S, W, D)
           'start_idx':  (B, S) or (B,)   (if (B,), same value is broadcast)
           'series_len': (B, S) or (B,)   (if (B,), same value is broadcast)
       - An LSTM runs over the S windows for each series and produces a context
         vector per window. This context is passed to the diffusion model as an
         additional conditioning signal.

    Returns:
        model: trained DiffusionModel
        loss_history: list of average training losses per epoch
    """
    # Prepare DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Instantiate the diffusion model
    # If use_lstm=True, the model is told to expect an external context vector.
    model = DiffusionModel(
        window_size=window_size,
        in_channels=D,
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        n_res_blocks=n_res_blocks,
        timesteps=timesteps,
        s=s,
        use_context=use_lstm,
        context_dim=lstm_hidden_dim if use_lstm else None,
    ).to(device)

    # Optional LSTM for window-sequence conditioning
    if use_lstm:
        # We embed each window by averaging over time, giving a D-dimensional
        # vector per window, and feed that into the LSTM.
        lstm = nn.LSTM(
            input_size=D,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=False,  # (S, B, input_size)
        ).to(device)

        # Joint optimizer over diffusion model + LSTM parameters
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(lstm.parameters()),
            lr=lr
        )
    else:
        lstm = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    data_length = len(dataset)

    for epoch in range(1, epochs + 1):
        model.train()
        if lstm is not None:
            lstm.train()
        running_loss = 0.0

        for batch in dataloader:
            if not use_lstm:
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
                # ===== LSTM MODE: sequences of windows =====
                # Expect:
                #   window:     (B, S, W, D)
                #   start_idx:  (B, S) or (B,)
                #   series_len: (B, S) or (B,)
                x0_seq = batch['window'].to(device).float()   # (B, S, W, D)
                assert x0_seq.dim() == 4, \
                    f"With use_lstm=True, expected 'window' with 4 dims (B,S,W,D), got {tuple(x0_seq.shape)}"

                B, S, W, D_ = x0_seq.shape
                assert W == window_size, f"Expected window_size={window_size}, got {W}"
                assert D_ == D, f"Expected D={D}, got {D_}"

                start_idx = batch['start_idx'].to(device)
                series_len = batch['series_len'].to(device)

                # Normalize shapes of start_idx and series_len to (B, S)
                if start_idx.dim() == 1:
                    # Broadcast the same series-level start index to all windows in the sequence
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

                # --- LSTM context over the sequence of windows ---

                # Simple window embedding: mean over time dimension → (B, S, D)
                # This is one possible choice; you could replace it with a learned
                # CNN-based encoder if desired.
                window_repr = x0_seq.mean(dim=2)              # (B, S, D)

                # LSTM expects input as (S, B, input_size)
                lstm_in = window_repr.permute(1, 0, 2)        # (S, B, D)
                lstm_out, _ = lstm(lstm_in)                   # (S, B, H)
                # Back to (B, S, H)
                context_seq = lstm_out.permute(1, 0, 2)       # (B, S, H)

                # Flatten sequences so we can call DiffusionModel in a single batch
                x0_flat = x0_seq.reshape(B * S, W, D)         # (B*S, W, D)
                start_flat = start_idx.reshape(B * S)         # (B*S,)
                series_flat = series_len.reshape(B * S)       # (B*S,)
                context_flat = context_seq.reshape(B * S, lstm_hidden_dim)  # (B*S, H)

                optimizer.zero_grad()
                loss = model(x0_flat, start_flat, series_flat, context=context_flat)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

        # === Sampling & plotting at intervals (only in standard mode for now) ===
        if show_detail and (epoch % sample_interval == 0) and (not use_lstm):
            # pick random index in [0, data_length)
            idx = torch.randint(0, data_length, (1,)).item()
            print("idx is : ", idx)
            example = dataset[idx]
            # example['window']: (W, D)
            real_window = example['window'].unsqueeze(0).to(device).float()
            # real_window shape: (1, W, D)
            start_ex = example['start_idx'].unsqueeze(0).to(device)
            # start_ex shape: (1,)
            len_ex = example['series_len'].unsqueeze(0).to(device)
            # len_ex shape: (1,)

            model.eval()
            with torch.no_grad():
                sample = model.sample(start_ex, len_ex, device=device)
                # sample shape: (1, W, D)

            # Convert to numpy 1D arrays for plotting (assumes D=1)
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

        # Full-series sampling only makes sense in the original
        # context-free-window setting. For LSTM mode you'd want a
        # different, sequential sampling routine.
        if not use_lstm:
            model.eval()
            with torch.no_grad():
                # Define absolute range: start=0, end=175
                start_full = torch.tensor([1],   dtype=torch.long, device=device)  # (1,)
                end_full   = torch.tensor([175], dtype=torch.long, device=device)  # (1,)
                shift = min(window_size - 1, 9)

                # Sample the entire series: returns (1, L, D) with L = end - start + 1
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
            # Single-channel case
            if full_np.ndim == 1 or full_np.shape[1] == 1:
                series = full_np[:, 0] if full_np.ndim > 1 else full_np
                plt.plot(series, label='Sampled')
            else:
                # Multi-channel case
                for d in range(full_np.shape[1]):
                    plt.plot(full_np[:, d], alpha=0.7, label=f'Channel {d}')
            plt.title("Full-Series Sampled Output (0 → 175, shift=1)")
            plt.xlabel("Time index")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return model, loss_history
