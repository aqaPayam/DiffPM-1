import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.diffusion_model import DiffusionModel
from models.lstm_context import WindowContextLSTM


# ============================================================
# 1) ORIGINAL TRAINING (no LSTM, single windows) – baseline
# ============================================================

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
):
    """
    Train a 1D DiffusionModel on a dataset of independent windows.

    Expected dataset __getitem__ output:
        {
            "window":    (W, D),
            "start_idx": scalar,
            "series_len": scalar
        }

    Dataloader batch:
        window:    (B, W, D)
        start_idx: (B,)
        series_len:(B,)
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Diffusion model WITHOUT context (context_dim=0)
    model = DiffusionModel(
        window_size=window_size,
        in_channels=D,
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        n_res_blocks=n_res_blocks,
        timesteps=timesteps,
        s=s,
        context_dim=0,   # <-- important: no LSTM context here
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    data_length = len(dataset)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            x0 = batch['window'].to(device).float()       # (B, W, D)
            start_idx = batch['start_idx'].to(device)     # (B,)
            series_len = batch['series_len'].to(device)   # (B,)

            optimizer.zero_grad()
            loss = model(x0, start_idx, series_len)       # no context
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[BASE] Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

        # Sampling & plotting at intervals
        if show_detail and (epoch % sample_interval == 0):
            idx = torch.randint(0, data_length, (1,)).item()
            print("idx is : ", idx)
            example = dataset[idx]

            real_window = example['window'].unsqueeze(0).to(device).float()  # (1, W, D)
            start_ex = example['start_idx'].unsqueeze(0).to(device)          # (1,)
            len_ex = example['series_len'].unsqueeze(0).to(device)           # (1,)

            model.eval()
            with torch.no_grad():
                sample = model.sample(start_ex, len_ex, device=device)       # (1, W, D)

            # Assuming D == 1 for plotting
            real_np = real_window.squeeze(-1).squeeze(0).cpu().numpy()       # (W,)
            sample_np = sample.squeeze(-1).squeeze(0).cpu().numpy()          # (W,)

            plt.figure(figsize=(8, 4))
            plt.plot(real_np, label='Real Data')
            plt.plot(sample_np, label='Sampled Data', alpha=0.7)
            plt.title(f'[BASE] Epoch {epoch} - Real vs. Sampled (idx={idx})')
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Loss curve
    if show_detail:
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, marker='o')
        plt.title('[BASE] Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.tight_layout()
        plt.show()

        # Full-series sampling example
        model.eval()
        with torch.no_grad():
            start_full = torch.tensor([1],   dtype=torch.long, device=device)
            end_full   = torch.tensor([175], dtype=torch.long, device=device)
            shift = min(window_size - 1, 9)

            full_sample = model.sample_total(
                start_full, end_full, shift,
                min_value=-10, max_value=10,
                device=device
            )  # (1, L, D)

        full_np = full_sample.squeeze(0).cpu().numpy()  # (L, D) or (L,)

        plt.figure(figsize=(10, 4))
        if full_np.ndim == 1 or full_np.shape[1] == 1:
            series = full_np[:, 0] if full_np.ndim > 1 else full_np
            plt.plot(series, label='Sampled')
        else:
            for d in range(full_np.shape[1]):
                plt.plot(full_np[:, d], alpha=0.7, label=f'Channel {d}')
        plt.title("[BASE] Full-Series Sampled Output (1 → 175)")
        plt.xlabel("Time index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model, loss_history


# ============================================================
# 2) NEW TRAINING WITH LSTM CONTEXT (sequences of windows)
# ============================================================

def train_model_with_lstm(
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
    lstm_hidden_dim: int,
    lstm_num_layers: int = 1,
    lstm_bidirectional: bool = False,
    show_detail: bool = False,
    sample_interval: int = 5,
):
    """
    Train a DiffusionModel that is conditioned on an LSTM-based context
    over sequences of windows.

    Expected dataset __getitem__ output:
        {
            "windows":   (N, W, D),   # sequence of N clean windows for one series
            "start_idx": (N,),        # start index of each window
            "series_len": scalar or (N,)  # full series length(s)
        }

    Dataloader batch:
        windows:   (B, N, W, D)
        start_idx: (B, N)
        series_len:(B,) or (B, N)
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # 1) LSTM context model (window-level)
    context_model = WindowContextLSTM(
        in_channels=D,
        window_size=window_size,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        bidirectional=lstm_bidirectional,
    ).to(device)

    context_dim = context_model.context_dim  # dimension of per-window context

    # 2) Diffusion model with context_dim > 0
    diffusion_model = DiffusionModel(
        window_size=window_size,
        in_channels=D,
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        n_res_blocks=n_res_blocks,
        timesteps=timesteps,
        s=s,
        context_dim=context_dim,  # <-- important: enables context input
    ).to(device)

    # Joint optimizer
    optimizer = torch.optim.Adam(
        list(diffusion_model.parameters()) + list(context_model.parameters()),
        lr=lr,
    )

    loss_history = []

    for epoch in range(1, epochs + 1):
        diffusion_model.train()
        context_model.train()
        running_loss = 0.0

        for batch in dataloader:
            # Batch shapes:
            # windows:   (B, N, W, D)
            # start_idx: (B, N)
            # series_len:(B,) or (B, N)
            windows = batch["windows"].to(device).float()
            start_idx = batch["start_idx"].to(device).long()
            series_len = batch["series_len"].to(device).long()

            B, N, W, D_ = windows.shape
            assert W == window_size, f"Expected W={window_size}, got {W}"
            assert D_ == D, f"Expected D={D}, got {D_}"

            # --- 1) LSTM context per window ---
            # ctx_seq: (B, N, context_dim)
            ctx_seq = context_model(windows)

            # --- 2) Choose which window to train on for each series ---
            window_indices = torch.randint(0, N, (B,), device=device)  # (B,)

            # x0: (B, W, D)
            x0 = windows[torch.arange(B, device=device), window_indices]     # (B, W, D)
            ctx = ctx_seq[torch.arange(B, device=device), window_indices]    # (B, context_dim)

            # start indices for chosen windows
            start_idx_n = start_idx[torch.arange(B, device=device), window_indices]  # (B,)

            # series_len handling
            if series_len.dim() == 1:
                series_len_n = series_len  # (B,)
            else:
                series_len_n = series_len[torch.arange(B, device=device), window_indices]

            optimizer.zero_grad()

            # --- 3) Diffusion loss with LSTM context ---
            loss = diffusion_model(
                x0=x0,
                start_idx=start_idx_n,
                series_len=series_len_n,
                context=ctx,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[LSTM] Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

        # Optional: visualize one example window
        if show_detail and (epoch % sample_interval == 0):
            diffusion_model.eval()
            context_model.eval()

            with torch.no_grad():
                example = dataset[0]   # take first series
                ex_windows = example["windows"].unsqueeze(0).to(device).float()  # (1, N, W, D)
                ex_start = example["start_idx"].unsqueeze(0).to(device).long()   # (1, N)

                ex_len = example["series_len"]
                if torch.is_tensor(ex_len):
                    ex_len = ex_len.to(device).long()
                    if ex_len.dim() == 0:
                        ex_len = ex_len.view(1)          # (1,)
                    elif ex_len.dim() == 1:
                        ex_len = ex_len[:1]              # (1,)
                else:
                    ex_len = torch.tensor([ex_len], dtype=torch.long, device=device)

                _, N_ex, W_ex, D_ex = ex_windows.shape

                # Context for full sequence
                ctx_seq_ex = context_model(ex_windows)  # (1, N_ex, context_dim)

                # Choose last window for visualization
                n_idx = N_ex - 1
                x0_ex = ex_windows[0, n_idx]           # (W, D)
                ctx_ex = ctx_seq_ex[0, n_idx].unsqueeze(0)  # (1, context_dim)
                start_ex = ex_start[0, n_idx].unsqueeze(0)  # (1,)
                len_ex = ex_len.view(1)                     # (1,)

                sample = diffusion_model.sample(
                    start_idx=start_ex,
                    series_len=len_ex,
                    device=device,
                    context=ctx_ex,
                )  # (1, W, D)

            # Assume D == 1 for plotting
            real_np = x0_ex.squeeze(-1).cpu().numpy()                     # (W,)
            sample_np = sample.squeeze(0).squeeze(-1).cpu().numpy()      # (W,)

            plt.figure(figsize=(8, 4))
            plt.plot(real_np, label="Real Data")
            plt.plot(sample_np, label="Sampled Data", alpha=0.7)
            plt.title(f"[LSTM] Epoch {epoch} - Real vs Sampled (last window of example[0])")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return diffusion_model, context_model, loss_history
