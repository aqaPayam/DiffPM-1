import numpy as np
from typing import List, Optional, Tuple


def split_trend_residual(
    data: np.ndarray,
    window_size: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Decompose each (T, D) series into a low-frequency trend and residual.

    Args
    ----
    data : np.ndarray, shape (N, T, D)
        Batch of N multivariate series.
    window_size : int, optional
        Width of the moving-average window. If None, defaults per-series to max(3, T//20).

    Returns
    -------
    trend_list : List[np.ndarray]  # each (T, D)
    resid_list : List[np.ndarray]  # each (T, D)
    """
    trends, resids = [], []

    for series in data:
        T, D = series.shape
        win = window_size if window_size is not None else max(3, T // 20)

        # moving average (trend) + residual
        mov = np.array([
            np.convolve(series[:, d], np.ones(win) / win, mode="same")
            for d in range(D)
        ]).T  # shape (T, D)
        trends.append(mov)
        resids.append(series - mov)

    return trends, resids


def downsample_trend(
    trend_list: List[np.ndarray],
    window_size: int
) -> List[np.ndarray]:
    """
    Decimate each trend by its MA window_size.

    Args
    ----
    trend_list : List[np.ndarray], each (T, D)
    window_size : int
        Must match the window_size used to compute the trend.

    Returns
    -------
    downsampled : List[np.ndarray], each (⌈T/window_size⌉, D)
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    return [trend[::window_size] for trend in trend_list]


def upsample_trend(
    downsampled: List[np.ndarray],
    original_length: int,
    window_size: int,
    method: str = "linear"
) -> List[np.ndarray]:
    """
    Restore each decimated trend back to length T via interpolation.

    Args
    ----
    downsampled : List[np.ndarray], each (T_down, D)
    original_length : int
        The T used in split_trend_residual.
    window_size : int
        The same window_size used for downsampling.
    method : {'linear', 'sinc'}
        'linear'   → fast linear interp
        'sinc'     → ideal band-limited (virtually zero error)

    Returns
    -------
    upsampled : List[np.ndarray], each (original_length, D)
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    x_new = np.arange(original_length)

    result = []
    for tr_ds in downsampled:
        T_ds, D = tr_ds.shape
        x_ds = np.arange(T_ds) * window_size
        tr_us = np.zeros((original_length, D))

        if method == "linear":
            for d in range(D):
                tr_us[:, d] = np.interp(x_new, x_ds, tr_ds[:, d])

        elif method == "sinc":
            # build once per-dimension for band-limited interp
            for d in range(D):
                sinc_mat = np.sinc((x_new[:, None] - x_ds[None, :]) / window_size)
                tr_us[:, d] = sinc_mat.dot(tr_ds[:, d])

        else:
            raise ValueError(f"Unknown method '{method}'")

        result.append(tr_us)

    return result
