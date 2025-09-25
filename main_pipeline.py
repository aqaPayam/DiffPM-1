import numpy as np
import torch

from run_training import run_training
from helper import save_diffpm, generate_full_series, ref_feature_stats, denorm_with_stats, smooth_and_rescale


# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Paths ===
save_npy_path = ""  # training data .npy (shape (N, T, D))
ckpt_path = "checkpoints/diffpm_pipeline.pt"
gen_save_path = "generated/full_series_norm.npy"
denorm_save_path = "generated/full_series_denorm.npy"
final_processed_save_path = "generated/full_series_denorm_smoothed_rescaled.npy"


# === Train ===
trend_model, residual_model = run_training(
    data_npy=save_npy_path,
    ma_window_size=3,
    window_size=10,
    time_emb_dim=128,
    base_channels=64,
    n_res_blocks=8,
    timesteps=2000,
    s=0.007,
    batch_size=32,
    epochs=1500,
    lr=1e-3,
    device=device,
    show_detail=False,
    sample_interval=5,
)


# === Meta and checkpoint ===
arr = np.load(save_npy_path)
if arr.ndim != 3:
    raise ValueError(f"Expected data shape (N, T, D). Got {arr.shape}")
N, T, D = arr.shape

meta = {
    "in_channels": int(D),
    "window_size": 10,
    "time_emb_dim": 128,
    "base_channels": 64,
    "n_res_blocks": 8,
    "timesteps": 2000,
    "s": 0.007,
}

save_diffpm(residual_model, trend_model, meta, ckpt_path)


# === Generate full series ===
generate_full_series(
    checkpoint_path=ckpt_path,
    data_npy=save_npy_path,
    ma_window_size=3,
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


