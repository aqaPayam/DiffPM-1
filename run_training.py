import numpy as np
import torch
import matplotlib.pyplot as plt

from data_handling.decomposition import split_trend_residual, downsample_trend
from data_handling.window_dataset import WindowDataset
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
) -> tuple[torch.nn.Module, torch.nn.Module]:
    # 1) Load raw data: shape (N, T, D)
    arr = np.load(data_npy)

    # 2) Decompose into trend & residual
    trends, resids = split_trend_residual(arr, window_size=ma_window_size)

    # 3) Downsample trend
    trends_ds = downsample_trend(trends, window_size=ma_window_size)

    # 4) trends_ds & resids are lists of length N, each an array (T, D)
    trend_arrs = [np.array(x) for x in trends_ds]
    resid_arrs = [np.array(x) for x in resids]

    # 5) Compute global mean & std per dimension across all trend and resid series
    big_trend  = np.vstack(trend_arrs)   # shape (N*T, D)
    trend_mu, trend_sigma   = big_trend.mean(0, keepdims=True), big_trend.std(0, keepdims=True)

    big_resid  = np.vstack(resid_arrs)
    resid_mu, resid_sigma   = big_resid.mean(0, keepdims=True), big_resid.std(0, keepdims=True)

    # 6) Normalize each series: zero mean, unit std
    norm_trend_arrs = [(a - trend_mu) / trend_sigma for a in trend_arrs]
    norm_resid_arrs = [(a - resid_mu) / resid_sigma for a in resid_arrs]

    # 7) Plot normalized series (first sample only)
    def _plot_sample(arrs, title):
        a0 = arrs[0]  # (T, D)
        T, D = a0.shape
        plt.figure()
        for d in range(D):
            plt.plot(range(T), a0[:, d], label=f"dim {d}")
        if D > 1:
            plt.legend()
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Normalized value")
        plt.show()

    _plot_sample(norm_trend_arrs, "Normalized Downsampled Trend (sample 0)")
    _plot_sample(norm_resid_arrs,  "Normalized Residuals (sample 0)")
    __ , D = norm_trend_arrs[0].shape

    # 8) Build windowed datasets
    trend_window_ds  = WindowDataset(norm_trend_arrs, window_size=window_size)
    resid_window_ds = WindowDataset(norm_resid_arrs, window_size=window_size)

    # 9) Train trend model
    print(r"hello world, version 2")
    trend_model, trend_loss = train_model(
        dataset=trend_window_ds,
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
    )

    # 10) Train residual model
    resid_model, resid_loss = train_model(
        dataset=resid_window_ds,
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
    )

    # 11) Return both trained models
    return trend_model, resid_model


