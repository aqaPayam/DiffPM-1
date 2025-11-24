# run_training.py

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
    # sequence-context options
    use_seq_context: bool = False,
    context_encoder_type: str = "lstm",   # "rnn" | "gru" | "lstm" | "tcn" | "transformer"
    context_hidden_dim: int = 128,
    context_num_layers: int = 1,
    seq_len: int = 4,
    seq_stride: int = 1,
    tcn_kernel_size: int = 3,
    transformer_nhead: int = 4,
    context_dropout: float = 0.0,
):
    """
    End-to-end training for trend and residual diffusion models.

    Returns:
        trend_model
        trend_context_encoder (or None)
        residual_model
        residual_context_encoder (or None)
    """
    # 1) Load raw data: shape (N, T, D)
    arr = np.load(data_npy)

    # 2) Decompose into trend & residual
    trends, resids = split_trend_residual(arr, window_size=ma_window_size)

    # 3) Downsample trend
    trends_ds = downsample_trend(trends, window_size=ma_window_size)

    # 4) Convert to numpy arrays
    trend_arrs = [np.array(x) for x in trends_ds]
    resid_arrs = [np.array(x) for x in resids]

    # 5) Global mean & std per dimension
    big_trend = np.vstack(trend_arrs)
    trend_mu, trend_sigma = big_trend.mean(0, keepdims=True), big_trend.std(0, keepdims=True)

    big_resid = np.vstack(resid_arrs)
    resid_mu, resid_sigma = big_resid.mean(0, keepdims=True), big_resid.std(0, keepdims=True)

    trend_sigma[trend_sigma == 0] = 1.0
    resid_sigma[resid_sigma == 0] = 1.0

    # 6) Normalize series
    norm_trend_arrs = [(a - trend_mu) / trend_sigma for a in trend_arrs]
    norm_resid_arrs = [(a - resid_mu) / resid_sigma for a in resid_arrs]

    # 7) Optional plotting
    def _plot_sample(arrs, title):
        if not arrs:
            return
        a0 = arrs[0]
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

    if show_detail:
        _plot_sample(norm_trend_arrs, "Normalized Downsampled Trend (sample 0)")
        _plot_sample(norm_resid_arrs, "Normalized Residuals (sample 0)")

    _, D = norm_trend_arrs[0].shape

    # 8) Build windowed or sequence datasets
    if not use_seq_context:
        trend_dataset = WindowDataset(norm_trend_arrs, window_size=window_size, normalize=False)
        resid_dataset = WindowDataset(norm_resid_arrs, window_size=window_size, normalize=False)
    else:
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
    trend_model, trend_loss, trend_context_encoder = train_model(
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
        use_seq_context=use_seq_context,
        context_encoder_type=context_encoder_type,
        context_hidden_dim=context_hidden_dim,
        context_num_layers=context_num_layers,
        tcn_kernel_size=tcn_kernel_size,
        transformer_nhead=transformer_nhead,
        context_dropout=context_dropout,
    )

    # 10) Train residual model
    print("Training RESIDUAL model...")
    residual_model, residual_loss, residual_context_encoder = train_model(
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
        use_seq_context=use_seq_context,
        context_encoder_type=context_encoder_type,
        context_hidden_dim=context_hidden_dim,
        context_num_layers=context_num_layers,
        tcn_kernel_size=tcn_kernel_size,
        transformer_nhead=transformer_nhead,
        context_dropout=context_dropout,
    )

    return trend_model, trend_context_encoder, residual_model, residual_context_encoder
