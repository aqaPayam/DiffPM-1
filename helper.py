import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from models.diffusion_model import DiffusionModel
from data_handling.decomposition import upsample_trend as proj_upsample


# save_load_utils.py
import os
import torch
from models.diffusion_model import DiffusionModel


def _upgrade_state_dict(sd: dict) -> dict:
    """
    Ensure new buffers exist (e.g., 'tilde_betas') so loading works with strict=True.
    Computes them from 'betas' and 'alpha_bars' if missing.
    """
    if "tilde_betas" not in sd and ("betas" in sd and "alpha_bars" in sd):
        betas = sd["betas"].clone()
        alpha_bars = sd["alpha_bars"]
        tilde = betas.clone()
        tilde[0] = betas[0]
        tilde[1:] = ((1.0 - alpha_bars[:-1]) / (1.0 - alpha_bars[1:])) * betas[1:]
        sd["tilde_betas"] = tilde
    return sd


def save_diffpm(
    residual_model: DiffusionModel,
    trend_model: DiffusionModel,
    meta: dict,
    path: str = "checkpoints/diffpm_penguins.pt",
):
    """
    Save both models + meta in a format that loads cleanly with strict=True.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "format_version": 2,
        "meta": dict(meta),  # shallow copy
        "trend_kind": "torch_module",
        "trend_class_name": trend_model.__class__.__name__,
        # upgrade state dicts to include derived buffers if older code omitted them
        "residual_state": _upgrade_state_dict(residual_model.state_dict()),
        "trend_state": _upgrade_state_dict(trend_model.state_dict()),
    }

    torch.save(payload, path)
    print(f"[save_diffpm] Saved checkpoint → {path}")


def build_model_from_meta(meta: dict, device: torch.device) -> DiffusionModel:
    """
    Rebuild DiffusionModel exactly as trained (required keys) with sensible
    defaults for any newer optional args.
    """
    required = ["in_channels", "window_size", "time_emb_dim",
                "base_channels", "n_res_blocks", "timesteps"]
    for k in required:
        if k not in meta:
            raise KeyError(f"meta is missing required key: {k}")

    model = DiffusionModel(
        in_channels=meta["in_channels"],
        window_size=meta["window_size"],
        time_emb_dim=meta["time_emb_dim"],
        base_channels=meta["base_channels"],
        n_res_blocks=meta["n_res_blocks"],
        timesteps=meta["timesteps"],
        s=meta.get("s", 0.007),
        # newer options (use defaults if not present in old metas)
        use_v_prediction=meta.get("use_v_prediction", False),
        sampler=meta.get("sampler", "ddpm"),
        ddim_steps=meta.get("ddim_steps", None),
        ddim_eta=float(meta.get("ddim_eta", 0.0)),
        normalize_per_feature=meta.get("normalize_per_feature", False),
    ).to(device)
    return model



def generate_full_series(
    checkpoint_path: str,
    data_npy: str,
    ma_window_size: int = 2,
    shift: int = 9,
    clip_min: float | None = -7,
    clip_max: float | None = 6,
    seed: int = 42,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    plot_result: bool = True,
    save_path: str = "Generated/ETTh1_multi_full_90_1.npy",
):
    """
    WITHOUT de-normalization:
      - loads checkpoint, rebuilds residual & trend models
      - samples residual over [1, T] and trend over downsampled index
      - upsamples trend (linear), recombines (normalized space), prints stats
      - optionally plots, and saves `full_np` (normalized space) to `save_path`
    """
    if ma_window_size <= 0:
        raise ValueError(f"ma_window_size must be >= 1 (got {ma_window_size})")

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load checkpoint and rebuild models
    print("Loading checkpoint…")
    payload = torch.load(checkpoint_path, map_location=device)
    meta = payload["meta"]
    print("Checkpoint loaded")

    print("Rebuilding residual model…")
    residual_model = build_model_from_meta(meta, device)
    residual_model.load_state_dict(payload["residual_state"])
    residual_model.eval()
    print("Residual model ready")

    print("Rebuilding trend model…")
    if payload.get("trend_kind", "torch_module") != "torch_module":
        raise ValueError("Trend model was not saved as a torch module in this checkpoint")
    trend_model = build_model_from_meta(meta, device)
    trend_model.load_state_dict(payload["trend_state"])
    trend_model.eval()
    print("Trend model ready")

    # load data to get N,T,D 
    print("Loading training data for global stats…")
    arr = np.load(data_npy)            # shape (N, T, D)
    if arr.ndim != 3:
        raise ValueError(f"Expected data shape (N, T, D). Got {arr.shape}")
    N, T, D = arr.shape
    global_mean = arr.mean(axis=(0, 1)).astype(np.float32)     # (D,)
    global_std  = arr.std(axis=(0, 1)).astype(np.float32)      # (D,)
    global_std  = np.where(global_std < 1e-12, 1.0, global_std)
    print(f"Data loaded  N={N}  T={T}  D={D}")
    print(f"Global mean per channel: {global_mean}")
    print(f"Global std per channel:  {global_std}")
    print(f"Sampling device: {device}")

    # total sampling with progress
    with torch.no_grad():
        # residual over the full length [1, T]
        start_full = torch.tensor([1], dtype=torch.long, device=device)
        end_full   = torch.tensor([T], dtype=torch.long, device=device)

        clip_kwargs = {}
        if clip_min is not None:
            clip_kwargs["min_value"] = clip_min
        if clip_max is not None:
            clip_kwargs["max_value"] = clip_max

        desc_res = f"Sampling residual windows over [1, {T}] with shift {shift}"
        pbar_res = tqdm(total=1, desc=desc_res)
        t_res0 = time.time()
        full_residual = residual_model.sample_total(
            start_full, end_full, shift, device=device, **clip_kwargs
        )  # (1, T, D)
        pbar_res.update(1)
        pbar_res.close()
        print(f"Residual sampling done in {time.time() - t_res0:.2f} s")

        # trend in downsampled index space [1, ceil(T / k)]
        k = int(ma_window_size)
        T_trend = int(math.ceil(T / k))
        start_tr = torch.tensor([1], dtype=torch.long, device=device)
        end_tr   = torch.tensor([T_trend], dtype=torch.long, device=device)
        shift_tr = max(1, shift // max(1, k))
        print(f"Trend index space length: {T_trend}  with shift {shift_tr}")

        desc_tr = f"Sampling trend windows over [1, {T_trend}] with shift {shift_tr}"
        pbar_tr = tqdm(total=1, desc=desc_tr)
        t_tr0 = time.time()
        trend_down = trend_model.sample_total(
            start_tr, end_tr, shift_tr, device=device, **clip_kwargs
        )  # (1, T_trend, D)
        pbar_tr.update(1)
        pbar_tr.close()
        print(f"Trend sampling done in {time.time() - t_tr0:.2f} s")

    # move to numpy
    residual_np   = full_residual.squeeze(0).cpu().numpy()   # (T, D)
    trend_down_np = trend_down.squeeze(0).cpu().numpy()      # (T_trend, D)
    print(f"Residual sample shape:   {residual_np.shape}")
    print(f"Trend downsample shape:  {trend_down_np.shape}")

    # upsample trend back to T
    print("Upsampling trend back to original length…")
    t_up0 = time.time()
    trend_up_list = proj_upsample(
        [trend_down_np],
        original_length=T,
        window_size=k,
        method="linear"
    )
    trend_up_np = trend_up_list[0]
    print(f"Upsampling done in {time.time() - t_up0:.2f} s  trend upsample shape: {trend_up_np.shape}")

    # recombine (NO de-normalization anymore)
    print("Recombining trend and residual in normalized space…")
    full_np = trend_up_np + residual_np   # (T, D)
    print("Recombine complete")

    # quick stats (normalized space)
    print(f"Generated normalized mean per channel: {full_np.mean(axis=0)}")
    print(f"Generated normalized std  per channel: {full_np.std(axis=0)}")
    print(f"Total wall time: {time.time() - t0:.2f} s")

    # plot normalized output (or multi-channel)
    plt.figure(figsize=(10, 4))
    if full_np.ndim == 1 or full_np.shape[1] == 1:
        series = full_np[:, 0] if full_np.ndim > 1 else full_np
        plt.plot(series, label="Sampled (normalized)")
    else:
        for d in range(full_np.shape[1]):
            plt.plot(full_np[:, d], alpha=0.7, label=f"Channel {d} (norm)")
    plt.title(f"Full series (normalized) 1 to {T}  shift {shift}")
    plt.xlabel("Time index")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.tight_layout()
    if plot_result:
        plt.show()
    else:
        plt.close()

    # save normalized output
    np.save(save_path, full_np)
    print(f"Saved generated (normalized) series to: {save_path}")


# === Denormalization utilities ===
def ref_feature_stats(ref_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature (D,) mean/std from the full reference array.
    Works for (N,T,D) or (T,D) or any array where the last axis is D.
    """
    if ref_arr.ndim < 2:
        raise ValueError(f"Reference array must be at least 2D (…, D); got shape {ref_arr.shape}")
    # average over all axes except the last (feature axis)
    axes = tuple(range(ref_arr.ndim - 1))
    mean = ref_arr.mean(axis=axes).astype(np.float32)  # (D,)
    std  = ref_arr.std(axis=axes).astype(np.float32)   # (D,)
    std[std < 1e-12] = 1.0
    return mean, std


def denorm_with_stats(gen_arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Feature-wise denorm: x_denorm[..., d] = x_norm[..., d]*std[d] + mean[d]
    Preserves the original shape of gen_arr (no reshaping).
    """
    if gen_arr.ndim < 2:
        raise ValueError(f"Generated array must be at least 2D (…, D); got shape {gen_arr.shape}")
    Dg = gen_arr.shape[-1]
    if Dg != mean.shape[0] or Dg != std.shape[0]:
        raise ValueError(f"Feature dimension mismatch: gen D={Dg}, stats D={mean.shape[0]}")
    return (gen_arr.astype(np.float32) * std + mean).astype(np.float32)


def smooth_and_rescale(syn_data: np.ndarray, real_data: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """
    Apply per-feature centered rolling mean smoothing (min_periods=1),
    then match feature-wise mean/std of syn to real.
    Expects 2D arrays of shape (T, D).
    """
    processed_syn = syn_data.copy()
    n_features = processed_syn.shape[1]

    # 1) Smoothing
    for i in range(n_features):
        processed_syn[:, i] = pd.Series(processed_syn[:, i]).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()

    # 2) Rescale variance & mean to match real data
    for i in range(n_features):
        syn_mean = processed_syn[:, i].mean()
        syn_std = processed_syn[:, i].std()
        real_mean = real_data[:, i].mean()
        real_std = real_data[:, i].std()

        if syn_std > 0:
            processed_syn[:, i] = real_mean + (real_std / syn_std) * (processed_syn[:, i] - syn_mean)

    return processed_syn.astype(np.float32)