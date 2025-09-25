import torch
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
    D:int,
    s: float,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    show_detail: bool = False,
    sample_interval: int = 5,
):
    """
    Train a 1D DiffusionModel on the provided windowed dataset.

    Returns:
        model: trained DiffusionModel
        loss_history: list of average training losses per epoch
    """
    # Prepare DataLoader
    # dataset length: data_length = len(dataset)
    # DataLoader yields batches of size B
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )  # yields batch dict with:
       # 'window': (B, W, D)
       # 'start_idx': (B,)
       # 'series_len': (B,)

    # Instantiate the diffusion model
    # model input shapes: x0 (B, W, D), start_idx (B,), series_len (B,)
    model = DiffusionModel(
        window_size=window_size,
        in_channels=D,
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        n_res_blocks=n_res_blocks,
        timesteps=timesteps,
        s=s,
    ).to(device)  # model parameters on device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer for model.parameters()
    loss_history = []  # list of floats, len = epochs

    data_length = len(dataset)  # total windows available

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            # Batch contents:
            x0 = batch['window'].to(device).float()       # shape: (B, W, D)
            start_idx = batch['start_idx'].to(device)     # shape: (B,)
            series_len = batch['series_len'].to(device)   # shape: (B,)

            optimizer.zero_grad()
            loss = model(x0, start_idx, series_len)
            # loss: scalar tensor
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # add scalar

        avg_loss = running_loss / len(dataloader)  # scalar float
        loss_history.append(avg_loss)  # append float

        # Logging
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

        # Sampling & plotting at intervals
        if show_detail and (epoch % sample_interval == 0):
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

            # Convert to numpy 1D arrays for plotting
            real_np = real_window.squeeze(-1).squeeze(0).cpu().numpy()
            # real_np shape: (W,)
            sample_np = sample.squeeze(-1).squeeze(0).cpu().numpy()
            # sample_np shape: (W,)

            plt.figure(figsize=(8, 4))
            plt.plot(real_np, label='Real Data')
            plt.plot(sample_np, label='Sampled Data', alpha=0.7)
            plt.title(f'Epoch {epoch} - Real vs. Sampled (idx={idx})')
            plt.legend()
            plt.tight_layout()
            plt.show()




    # Plot the loss history if detailed output is enabled
    if show_detail:
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, marker='o')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.tight_layout()
        plt.show()





        # === after training completes: full‐series sampling & plotting ===
        model.eval()
        with torch.no_grad():
            # Define absolute range: start=0, end=175
            start_full = torch.tensor([1],   dtype=torch.long, device=device)  # (1,)
            end_full   = torch.tensor([175], dtype=torch.long, device=device)  # (1,)
            shift = min(window_size - 1, 9)

            # Sample the entire series: returns (1, L, D) with L = end - start + 1 = 176
            full_sample = model.sample_total(start_full, end_full, shift,min_value = -10 ,max_value =10, device=device)

        # Move to numpy: (L, D)
        full_np = full_sample.squeeze(0).cpu().numpy()

        # Plot
        plt.figure(figsize=(10, 4))
        # Single‐channel case
        if full_np.ndim == 1 or full_np.shape[1] == 1:
            series = full_np[:, 0] if full_np.ndim > 1 else full_np
            plt.plot(series, label='Sampled')
        else:
            # Multi‐channel case
            for d in range(full_np.shape[1]):
                plt.plot(full_np[:, d], alpha=0.7, label=f'Channel {d}')
        plt.title("Full‑Series Sampled Output (0 → 175, shift=1)")
        plt.xlabel("Time index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()




    return model, loss_history
