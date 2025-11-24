import numpy as np
import torch

from run_training import run_training
from helper import (
    save_diffpm,
    generate_full_series,
    ref_feature_stats,
    denorm_with_stats,
    smooth_and_rescale,
)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
save_npy_path = ""  # training data .npy (shape (N, T, D))
ckpt_path = "checkpoints/diffpm_pipeline.pt"
gen_save_path = "generated/full_series_norm.npy"
denorm_save_path = "generated/full_series_denorm.npy"
final_processed_save_path = "generated/full_series_denorm_smoothed_rescaled.npy"

# === Shared hyperparameters ===
ma_window_size = 3
window_size = 10
time_emb_dim = 128
base_channels = 64
n_res_blocks = 8
timesteps = 2000
s = 0.007
batch_size = 32
epochs = 1500
lr = 1e-3

# === Sequence context configuration ===
use_seq_context = True
context_encoder_type = "lstm"     # "rnn" | "gru" | "lstm" | "tcn" | "transformer"
context_hidden_dim = 128
context_num_layers = 1
seq_len = 4
seq_stride = 1
tcn_kernel_size = 3
transformer_nhead = 4
context_dropout = 0.0

# === Train ===
trend_model, trend_context_encoder, residual_model, residual_context_encoder = run_training(
    data_npy=save_npy_path,
    ma_window_size=ma_window_size,
    window_size=window_size,
    time_emb_dim=time_emb_dim,
    base_channels=base_channels,
    n_res_blocks=n_res_blocks,
    timesteps=timesteps,
    s=s,
    batch_size=batch_size,
    epochs=epochs,
    lr=lr,
    device=device,
    show_detail=False,
    sample_interval=5,
    use_seq_context=use_seq_context,
    context_encoder_type=context_encoder_type,
    context_hidden_dim=context_hidden_dim,
    context_num_layers=context_num_layers,
    seq_len=seq_len,
    seq_stride=seq_stride,
    tcn_kernel_size=tcn_kernel_size,
    transformer_nhead=transformer_nhead,
    context_dropout=context_dropout,
)

# === Meta and checkpoint ===
arr = np.load(save_npy_path)
if arr.ndim != 3:
    raise ValueError(f"Expected data shape (N, T, D). Got {arr.shape}")
N, T, D = arr.shape

meta = {
    "in_channels": int(D),
    "window_size": window_size,
    "time_emb_dim": time_emb_dim,
    "base_channels": base_channels,
    "n_res_blocks": n_res_blocks,
    "timesteps": timesteps,
    "s": s,
    # context-related meta so build_model_from_meta & generate_full_series know
    "use_context": use_seq_context,
    "context_dim": context_hidden_dim,
    "context_encoder_type": context_encoder_type,
    "context_num_layers": context_num_layers,
    "tcn_kernel_size": tcn_kernel_size,
    "transformer_nhead": transformer_nhead,
    "context_dropout": context_dropout,
}

# Build context_meta for helper.save_diffpm
context_meta = {
    "encoder_type": context_encoder_type,
    "hidden_dim": context_hidden_dim,
    "num_layers": context_num_layers,
    "tcn_kernel_size": tcn_kernel_size,
    "transformer_nhead": transformer_nhead,
    "dropout": context_dropout,
}

residual_ctx_state = (
    residual_context_encoder.state_dict() if residual_context_encoder is not None else None
)
trend_ctx_state = (
    trend_context_encoder.state_dict() if trend_context_encoder is not None else None
)

save_diffpm(
    residual_model=residual_model,
    trend_model=trend_model,
    meta=meta,
    path=ckpt_path,
    context_meta=context_meta,
    residual_context_state=residual_ctx_state,
    trend_context_state=trend_ctx_state,
)

# === Generate full series ===
generate_full_series(
    checkpoint_path=ckpt_path,
    data_npy=save_npy_path,
    ma_window_size=ma_window_size,
    shift=1,
    clip_min=-100,
    clip_max=100,
    seed=42,
    device=device,
    plot_result=True,
    save_path=gen_save_path,
)

# === Denormalize ===
ref = np.load(save_npy_path)
mean, std = ref_feature_stats(ref)
gen = np.load(gen_save_path)
denorm = denorm_with_stats(gen, mean, std)
np.save(denorm_save_path, denorm)

# === Smoothing + Rescaling ===
real_for_stats = ref[0][:denorm.shape[0]] if ref.ndim == 3 else ref[:denorm.shape[0]]
processed = smooth_and_rescale(denorm, real_for_stats, smooth_window=5)
np.save(final_processed_save_path, processed)
