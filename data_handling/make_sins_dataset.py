import os
import math
import numpy as np
from typing import Tuple

# -----------------------------
# Config
# -----------------------------
SEQ_LEN = 24          # window length
DIM     = 5           # number of channels
N_TRAIN = 10_000
N_TEST  = 1_000
SEED    = 123

# -----------------------------
# Sine generation hyperparams
# -----------------------------
AMP_RANGE: Tuple[float, float]   = (0.5, 1.5)   # amplitude per channel
CYCLES_PER_WIN_RANGE            = (0.25, 2.0)  # cycles over a 24-step window
PHASE_RANGE: Tuple[float, float] = (0.0, 2*math.pi)  # radians
ADD_NOISE_STD = 0.0             # set >0.0 iff you want tiny Gaussian noise

def _sample_sine_window(seq_len: int,
                        dim: int,
                        rng: np.random.Generator,
                        amp_range=AMP_RANGE,
                        cycles_range=CYCLES_PER_WIN_RANGE,
                        phase_range=PHASE_RANGE,
                        noise_std: float = ADD_NOISE_STD) -> np.ndarray:
    """
    Returns a (seq_len, dim) array: each channel is A * sin(2π * c * t/seq_len + φ).
    """
    t = np.arange(seq_len, dtype=np.float32)
    X = np.zeros((seq_len, dim), dtype=np.float32)

    for d in range(dim):
        A = rng.uniform(*amp_range)
        cycles = rng.uniform(*cycles_range)       # cycles per window
        w = 2 * math.pi * cycles / seq_len        # rad per step
        phi = rng.uniform(*phase_range)
        x = A * np.sin(w * t + phi)
        if noise_std > 0:
            x = x + rng.normal(0.0, noise_std, size=seq_len).astype(np.float32)
        X[:, d] = x.astype(np.float32)
    return X

def _generate_split(n_windows: int,
                    seq_len: int,
                    dim: int,
                    seed: int) -> np.ndarray:
    """
    Generate (n_windows, seq_len, dim) windows.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n_windows, seq_len, dim), dtype=np.float32)
    for i in range(n_windows):
        out[i] = _sample_sine_window(seq_len, dim, rng)
    return out

def _fit_minmax(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature (across all time and windows) min & max on TRAIN set.
    Input shape: (N, L, D).
    Returns (mins[D], maxs[D]).
    """
    # Flatten N and L for per-feature stats
    flat = train.reshape(-1, train.shape[-1])
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    # Avoid zero range
    eps = 1e-8
    maxs = np.maximum(maxs, mins + eps)
    return mins.astype(np.float32), maxs.astype(np.float32)

def _apply_minmax_to_neg_one_one(x: np.ndarray,
                                 mins: np.ndarray,
                                 maxs: np.ndarray) -> np.ndarray:
    """
    Map per-feature to [-1, 1] using train mins/maxs.
    """
    # x: (N, L, D)
    x_norm = (x - mins[None, None, :]) / (maxs[None, None, :] - mins[None, None, :])
    x_norm = x_norm * 2.0 - 1.0
    return x_norm.astype(np.float32)

def make_sines(out_dir: str,
               seq_len: int = SEQ_LEN,
               dim: int = DIM,
               n_train: int = N_TRAIN,
               n_test: int = N_TEST,
               seed: int = SEED) -> None:
    """
    Generate Sines dataset and save as .npy files with Diffusion-TS-like names.
    """
    assert out_dir and isinstance(out_dir, str), "Please provide a valid output directory path."
    os.makedirs(out_dir, exist_ok=True)

    # 1) Generate train/test ground-truth (unnormalized) sets
    train_gt = _generate_split(n_train, seq_len, dim, seed=seed)
    test_gt  = _generate_split(n_test,  seq_len, dim, seed=seed + 1)

    # 2) Fit train min/max (per feature), then scale both sets to [-1, 1]
    mins, maxs = _fit_minmax(train_gt)
    train_norm = _apply_minmax_to_neg_one_one(train_gt, mins, maxs)
    test_norm  = _apply_minmax_to_neg_one_one(test_gt,  mins, maxs)

    # 3) Save to disk (file names mirror Diffusion-TS processed naming)
    np.save(os.path.join(out_dir, f"sines_ground_truth_{seq_len}_train.npy"), train_gt)
    np.save(os.path.join(out_dir, f"sines_ground_truth_{seq_len}_test.npy"),  test_gt)
    np.save(os.path.join(out_dir, f"sines_norm_truth_{seq_len}_train.npy"),   train_norm)
    np.save(os.path.join(out_dir, f"sines_norm_truth_{seq_len}_test.npy"),    test_norm)

    # 4) Save scaling params so you can invert-normalize later if needed
    np.savez(os.path.join(out_dir, f"sines_scale_{seq_len}.npz"),
             mins=mins, maxs=maxs)

    print(f"[OK] Saved Sines dataset to: {out_dir}")
    print(f"     train: {train_gt.shape}  test: {test_gt.shape}")
    print(f"     scale mins: {mins}  maxs: {maxs}")
