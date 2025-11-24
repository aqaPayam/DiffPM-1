import numpy as np
import torch
import matplotlib.pyplot as plt

from data_handling.decomposition import split_trend_residual, downsample_trend
from data_handling.window_dataset import WindowDataset, SequenceWindowDataset
from training.train import train_model


def run_training(
    data_npy: str,
    *,
    ma_window_size: int,
    window_size: int,
    time_emb_dim: int,
    base_channels: int,
    n_res_blocks: int,
    timesteps: int,
    s: float,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    show_detail: bool = False,
    sample_interval: int = 5,
    # NEW: LSTM / sequence options
    use_lstm: bool = False,
    lstm_hidden_dim: int = 128,
    lstm_num_layers: int = 1,
    seq_len: int = 4,
    seq_stride: int = 1,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    End-to-end training for trend and residual diffusion models.

    If use_lstm=False (default):
        - Uses WindowDataset (independent windows).
        - Behavior is identical to the original version (no LSTM context).

    If use_lstm=True:
        - Uses SequenceWindowDataset, which returns sequences of S windows
          per item (S = seq_len).
        - An LSTM over windows is used in train_model to provide context
          to the diffusion model.
    """
    # 1) Load raw data: shape (N, T, D)
    arr = np.load(data_npy)

    # 2) Decompose into trend & residual
    trends, resids = split_trend_residual(arr, window_size=ma_window_size)

    # 3) Downsample trend
    trends_ds = downsample_trend(trends, window_size=ma_window_size)

    # 4) trends_ds & resids are lists of length N, each an array (T_down, D) or (T, D)
    trend_arrs = [np.array(x) for x in trends_ds]
    resid_arrs = [np.array(x) for x in resids]

    # 5) Compute global mean & std per dimension across all trend and resid series
    big_trend  = np.vstack(trend_arrs)   # shape (sum T_down, D)
    trend_mu, trend_sigma = big_trend.mean(0, keepdims=True), big_trend.std(0, keepdims=True)

    big_resid  = np.vstack(resid_arrs)   # shape (sum T, D)
    resid_mu, resid_sigma = big_resid.mean(0, keepdims=True), big_resid.std(0, keepdims=True)

    # Avoid zero std
    trend_sigma[trend_sigma == 0] = 1.0
    resid_sigma[resid_sigma == 0] = 1.0

    # 6) Normalize each series: zero mean, unit std
    norm_trend_arrs = [(a - trend_mu) / trend_sigma for a in trend_arrs]
    norm_resid_arrs = [(a - resid_mu) / resid_sigma for a in resid_arrs]

    # 7) Plot normalized series (first sample only)
    def _plot_sample(arrs, title):
        if not arrs:
            return
        a0 = arrs[0]  # (T, D)
        T0, D0 = a0.shape
        plt.figure()
        for d in range(D0):
            plt.plot(range(T0), a0[:, d], label=f"dim {d}")
        if D0 > 1:
            plt.legend()
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Normalized value")
        plt.tight_layout()
        plt.show()

    _plot_sample(norm_trend_arrs, "Normalized Downsampled Trend (sample 0)")
    _plot_sample(norm_resid_arrs, "Normalized Residuals (sample 0)")

    _, D = norm_trend_arrs[0].shape  # feature dimension

    # 8) Build datasets (window-based or sequence-based depending on use_lstm)
    if not use_lstm:
        # Original behavior: independent windows
        trend_dataset = WindowDataset(norm_trend_arrs, window_size=window_size, normalize=False)
        resid_dataset = WindowDataset(norm_resid_arrs, window_size=window_size, normalize=False)
    else:
        # LSTM / sequence mode: each item is a sequence of windows
        trend_dataset = SequenceWindowDataset(
            norm_trend_arrs,
            window_size=window_size,
            seq_len=seq_len,
            normalize=False,
            seq_stride=seq_stride,
        )
        resid_dataset = SequenceWindowDataset(
            norm_resid_arrs,
            window_size=window_size,
            seq_len=seq_len,
            normalize=False,
            seq_stride=seq_stride,
        )

    # 9) Train trend model
    print("Training TREND model...")
    trend_model, trend_loss = train_model(
        dataset=trend_dataset,
        window_size=window_size,
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        n_res_blocks=n_res_blocks,
        timesteps=timesteps,
        D=D,
        s=s,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
        show_detail=show_detail,
        sample_interval=sample_interval,
        use_lstm=use_lstm,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
    )

    # 10) Train residual model
    print("Training RESIDUAL model...")
    resid_model, resid_loss = train_model(
        dataset=resid_dataset,
        window_size=window_size,
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        n_res_blocks=n_res_blocks,
        timesteps=timesteps,
        D=D,
        s=s,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
        show_detail=show_detail,
        sample_interval=sample_interval,
        use_lstm=use_lstm,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
    )

    # 11) Return both trained models
    return trend_model, resid_model
