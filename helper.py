import time
import math
import os
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from models.diffusion_model import DiffusionModel
from data_handling.decomposition import upsample_trend as proj_upsample
from models.context_encoders import build_context_encoder


# ---------------------------------------------------------------------
# State dict upgrade helper
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Saving / loading DiffPM (with optional context encoders)
# ---------------------------------------------------------------------

def save_diffpm(
    residual_model: DiffusionModel,
    trend_model: DiffusionModel,
    meta: dict,
    path: str = "checkpoints/diffpm_penguins.pt",
    # NEW: optional context encoder metadata + state dicts
    context_meta: Optional[dict] = None,
    residual_context_state: Optional[dict] = None,
    trend_context_state: Optional[dict] = None,
):
    """
    Save both models + meta in a format that loads cleanly with strict=True.

    New (optional) fields:
      - context_meta: dict with configuration for the sequence context encoder, e.g.
            {
                "encoder_type": "lstm",
                "hidden_dim": 128,
                "num_layers": 1,
                "tcn_kernel_size": 3,
                "transformer_nhead": 4,
                "dropout": 0.0,
            }
      - residual_context_state: state_dict of the residual context encoder
      - trend_context_state:    state_dict of the trend context encoder

    If you are not using sequence context, you can ignore these new arguments.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "format_version": 3,
        "meta": dict(meta),  # shallow copy
        "trend_kind": "torch_module",
        "trend_class_name": trend_model.__class__.__name__,
        # diffusion model weights
        "residual_state": _upgrade_state_dict(residual_model.state_dict()),
        "trend_state": _upgrade_state_dict(trend_model.state_dict()),
    }

    # Optional context encoder info
    if context_meta is not None:
        payload["context_meta"] = dict(context_meta)
    if residual_context_state is not None:
        payload["residual_context_state"] = residual_context_state
    if trend_context_state is not None:
        payload["trend_context_state"] = trend_context_state

    torch.save(payload, path)
    print(f"[save_diffpm] Saved checkpoint → {path}")


def build_model_from_meta(meta: dict, device: torch.device) -> DiffusionModel:
    """
    Rebuild DiffusionModel exactly as trained (required keys) with sensible
    defaults for any newer optional args.

    New meta keys (optional, for context-enabled models):
      - use_context: bool
      - context_dim: int  (must match context_hidden_dim used at training)
    """
    required = ["in_channels", "window_size", "time_emb_dim",
                "base_channels", "n_res_blocks", "timesteps"]
    for k in required:
        if k not in meta:
            raise KeyError(f"meta is missing required key: {k}")

    use_context = meta.get("use_context", False)
    context_dim = meta.get("context_dim", None)

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
        # context-related
        use_context=use_context,
        context_dim=context_dim,
    ).to(device)
    return model


# ---------------------------------------------------------------------
# Sequence-aware full-series sampling with context
# ---------------------------------------------------------------------

def _sample_full_with_context(
    model: DiffusionModel,
    context_encoder: torch.nn.Module,
    start_idx: int,
    end_idx: int,
    shift: int,
    device: torch.device,
    clip_min: Optional[float],
    clip_max: Optional[float],
    context_hidden_dim: int,
) -> torch.Tensor:
    """
    Sequential, context-aware sampling over [start_idx, end_idx] for a single series
    (batch size 1). Uses an autoregressive scheme over windows:

      - Generate window 0 with zero context.
      - For each subsequent window j:
          - Build a sequence of embeddings from all previously generated windows.
          - Run the context_encoder over that sequence to get context for window j.
          - Sample window j from the diffusion model with that context.
      - Overlap-add the windows with stride `shift`, like sample_total.

    Args:
        model:            DiffusionModel with use_context=True
        context_encoder:  context encoder over window embeddings (B, S, D) -> (B, S, H)
        start_idx:        first index (e.g. 1)
        end_idx:          last index  (inclusive)
        shift:            stride between window starts (1 <= shift < window_size)
        device:           torch.device
        clip_min:         if not None, drop values < clip_min when averaging
        clip_max:         if not None, drop values > clip_max when averaging
        context_hidden_dim: H, dimension of context vectors

    Returns:
        Tensor of shape (1, L, D) where L = end_idx - start_idx + 1.
    """
    model.eval()
    context_encoder.eval()

    W = model.window_size       # window length
    D = model.in_channels       # number of channels
    s0 = start_idx
    e0 = end_idx
    L = e0 - s0 + 1             # total length

    if not (1 <= shift < W):
        raise ValueError(f"shift must be between 1 and {W-1}")

    # window start positions
    positions = list(range(s0, e0 - W + 2, shift))
    num_windows = len(positions)
    if num_windows == 0:
        raise ValueError("No valid windows for given range and shift.")

    # accumulators
    sum_series = torch.zeros(1, L, D, device=device)
    count      = torch.zeros(1, L, D, device=device)

    # list of embeddings for previously generated windows
    # each entry is a tensor of shape (D,)
    generated_reprs = []

    for j, ws in enumerate(positions):
        # Build context for this window
        if len(generated_reprs) == 0:
            # no past: zero context
            context_vec = torch.zeros(1, context_hidden_dim, device=device)
        else:
            # sequence of past window embeddings: (1, S, D)
            prev_repr = torch.stack(generated_reprs, dim=0).unsqueeze(0)  # (1, S, D)
            context_seq = context_encoder(prev_repr)                      # (1, S, H)
            context_vec = context_seq[:, -1, :]                           # (1, H) last step

        # Prepare conditioning indices
        start_tensor = torch.tensor([ws], dtype=torch.long, device=device)   # (1,)
        series_len_tensor = torch.tensor([L], dtype=torch.long, device=device)  # (1,)

        # Sample ONE window (B=1) with given context
        with torch.no_grad():
            x_win = model.sample(
                start_idx=start_tensor,
                series_len=series_len_tensor,
                device=device,
                context=context_vec,
            )  # (1, W, D)

        # Compute embedding of this generated window for future context
        window_repr = x_win.mean(dim=1).squeeze(0)  # (D,)
        generated_reprs.append(window_repr)

        # Clip mask (if requested)
        if clip_min is not None or clip_max is not None:
            if clip_min is not None and clip_max is not None:
                mask = (x_win >= clip_min) & (x_win <= clip_max)
            elif clip_min is not None:
                mask = (x_win >= clip_min)
            else:  # clip_max is not None
                mask = (x_win <= clip_max)
            mask_f = mask.float()
        else:
            mask_f = torch.ones_like(x_win)

        # Overlap-add into the full series
        offset = ws - s0   # into [0, L-W]
        sum_series[:, offset:offset+W, :] += x_win * mask_f
        count[:,      offset:offset+W, :] += mask_f

    # safe divide
    avg = torch.where(count > 0, sum_series / count, torch.zeros_like(sum_series))
    return avg  # (1, L, D)


# ---------------------------------------------------------------------
# Main full-series generation helper
# ---------------------------------------------------------------------

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
    Full-series generation with optional sequence context.

    Behavior:

    - Loads checkpoint, rebuilds residual & trend DiffusionModels (with or without context).
    - If the checkpoint/meta indicates context usage and context encoder weights are present:
        - Builds context encoders and performs sequential, context-aware sampling over windows.
    - Otherwise:
        - Falls back to the original context-free sample_total() behavior.

    Steps (normalized space):
      - sample residual over [1, T]
      - sample trend over downsampled index [1, ceil(T / ma_window_size)]
      - upsample trend back to length T
      - recombine trend + residual (normalized)
      - optionally plot and save full_np (normalized)
    """
    if ma_window_size <= 0:
        raise ValueError(f"ma_window_size must be >= 1 (got {ma_window_size})")

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -----------------------------------------------------------
    # Load checkpoint and rebuild models
    # -----------------------------------------------------------
    print("Loading checkpoint…")
    payload = torch.load(checkpoint_path, map_location=device)
    meta = payload["meta"]
    print("Checkpoint loaded")

    # context-related meta (optional)
    use_context = bool(meta.get("use_context", False))
    context_dim = meta.get("context_dim", None)

    context_meta = payload.get("context_meta", {})
    encoder_type       = context_meta.get("encoder_type", meta.get("context_encoder_type", "lstm"))
    context_hidden_dim = context_meta.get("hidden_dim", context_dim if context_dim is not None else 128)
    context_num_layers = context_meta.get("num_layers", meta.get("context_num_layers", 1))
    tcn_kernel_size    = context_meta.get("tcn_kernel_size", meta.get("tcn_kernel_size", 3))
    transformer_nhead  = context_meta.get("transformer_nhead", meta.get("transformer_nhead", 4))
    context_dropout    = context_meta.get("dropout", meta.get("context_dropout", 0.0))

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

    # -----------------------------------------------------------
    # Load training data to get N,T,D and global stats (for info)
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # Build context encoders if context is enabled & states exist
    # -----------------------------------------------------------
    residual_context_encoder = None
    trend_context_encoder = None

    if use_context and context_dim is not None:
        print("Context-enabled model detected. Building context encoders...")

        # Residual context encoder
        if "residual_context_state" in payload:
            residual_context_encoder = build_context_encoder(
                encoder_type=encoder_type,
                input_dim=D,
                hidden_dim=context_hidden_dim,
                num_layers=context_num_layers,
                tcn_kernel_size=tcn_kernel_size,
                transformer_nhead=transformer_nhead,
                dropout=context_dropout,
            ).to(device)
            residual_context_encoder.load_state_dict(payload["residual_context_state"])
            print("Residual context encoder loaded.")

        # Trend context encoder
        if "trend_context_state" in payload:
            trend_context_encoder = build_context_encoder(
                encoder_type=encoder_type,
                input_dim=D,
                hidden_dim=context_hidden_dim,
                num_layers=context_num_layers,
                tcn_kernel_size=tcn_kernel_size,
                transformer_nhead=transformer_nhead,
                dropout=context_dropout,
            ).to(device)
            trend_context_encoder.load_state_dict(payload["trend_context_state"])
            print("Trend context encoder loaded.")

        # If context is enabled but no context state was saved, we can still proceed,
        # but the encoders will be None and sampling will fall back to context-free.
        if residual_context_encoder is None or trend_context_encoder is None:
            print("Warning: use_context=True but missing context encoder states in checkpoint.")
            print("         Falling back to context-free sampling for missing parts.")
    else:
        print("Context-free model or no context metadata. Using sample_total as before.")

    # -----------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------
    with torch.no_grad():
        # --- Residual over the full length [1, T] ---
        start_full = 1
        end_full   = T

        desc_res = f"Sampling residual over [1, {T}] with shift {shift}"
        pbar_res = tqdm(total=1, desc=desc_res)
        t_res0 = time.time()

        if use_context and residual_context_encoder is not None:
            full_residual = _sample_full_with_context(
                model=residual_model,
                context_encoder=residual_context_encoder,
                start_idx=start_full,
                end_idx=end_full,
                shift=shift,
                device=device,
                clip_min=clip_min,
                clip_max=clip_max,
                context_hidden_dim=context_hidden_dim,
            )  # (1, T, D)
        else:
            # original context-free behavior
            start_tensor = torch.tensor([start_full], dtype=torch.long, device=device)
            end_tensor   = torch.tensor([end_full],   dtype=torch.long, device=device)
            clip_kwargs = {}
            if clip_min is not None:
                clip_kwargs["min_value"] = clip_min
            if clip_max is not None:
                clip_kwargs["max_value"] = clip_max

            full_residual = residual_model.sample_total(
                start_tensor, end_tensor, shift, device=device, **clip_kwargs
            )  # (1, T, D)

        pbar_res.update(1)
        pbar_res.close()
        print(f"Residual sampling done in {time.time() - t_res0:.2f} s")

        # --- Trend in downsampled index space [1, ceil(T / k)] ---
        k = int(ma_window_size)
        T_trend = int(math.ceil(T / k))
        start_tr = 1
        end_tr   = T_trend
        shift_tr = max(1, shift // max(1, k))
        print(f"Trend index space length: {T_trend}  with shift {shift_tr}")

        desc_tr = f"Sampling trend over [1, {T_trend}] with shift {shift_tr}"
        pbar_tr = tqdm(total=1, desc=desc_tr)
        t_tr0 = time.time()

        if use_context and trend_context_encoder is not None:
            trend_down = _sample_full_with_context(
                model=trend_model,
                context_encoder=trend_context_encoder,
                start_idx=start_tr,
                end_idx=end_tr,
                shift=shift_tr,
                device=device,
                clip_min=clip_min,
                clip_max=clip_max,
                context_hidden_dim=context_hidden_dim,
            )  # (1, T_trend, D)
        else:
            start_tr_tensor = torch.tensor([start_tr], dtype=torch.long, device=device)
            end_tr_tensor   = torch.tensor([end_tr],   dtype=torch.long, device=device)
            clip_kwargs = {}
            if clip_min is not None:
                clip_kwargs["min_value"] = clip_min
            if clip_max is not None:
                clip_kwargs["max_value"] = clip_max

            trend_down = trend_model.sample_total(
                start_tr_tensor, end_tr_tensor, shift_tr, device=device, **clip_kwargs
            )  # (1, T_trend, D)

        pbar_tr.update(1)
        pbar_tr.close()
        print(f"Trend sampling done in {time.time() - t_tr0:.2f} s")

    # -----------------------------------------------------------
    # Post-processing: upsample trend, add residual, plot, save
    # -----------------------------------------------------------
    residual_np   = full_residual.squeeze(0).cpu().numpy()   # (T, D)
    trend_down_np = trend_down.squeeze(0).cpu().numpy()      # (T_trend, D)
    print(f"Residual sample shape:   {residual_np.shape}")
    print(f"Trend downsample shape:  {trend_down_np.shape}")

    # upsample trend back to original length T
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

    # recombine (still in normalized space)
    print("Recombining trend and residual in normalized space…")
    full_np = trend_up_np + residual_np   # (T, D)
    print("Recombine complete")

    # quick stats (normalized space)
    print(f"Generated normalized mean per channel: {full_np.mean(axis=0)}")
    print(f"Generated normalized std  per channel: {full_np.std(axis=0)}")
    print(f"Total wall time: {time.time() - t0:.2f} s")

    # plot normalized output
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


# ---------------------------------------------------------------------
# Denormalization utilities (unchanged)
# ---------------------------------------------------------------------

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
