import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Sequence, Union


class WindowDataset(Dataset):
    """
    Deterministic “window-of-series” dataset with optional per-feature z-score normalization.
    Builds an index map of all contiguous windows (length `window_size`) across every series.

    Each item is a dict containing:
      - 'window':     FloatTensor of shape (window_size, D_features), optionally normalized
      - 'start_idx':  LongTensor scalar indicating start index in that series
      - 'series_len': LongTensor scalar indicating length of the full series
    """
    def __init__(
        self,
        series_list: Union[np.ndarray, Sequence[np.ndarray]],
        window_size: int,
        normalize: bool = False
    ):
        super().__init__()
        # Normalize input into a list of (T, D) numpy arrays
        if isinstance(series_list, np.ndarray):
            # series_list is assumed to be (N, T, D)
            self.series_list = [s for s in series_list]
        else:
            self.series_list = list(series_list)

        self.window_size = window_size
        self.normalize = normalize

        # Build index map: each entry is (series_idx, start_idx)
        self.index_map = []
        for i, series in enumerate(self.series_list):
            T, D = series.shape
            if T < window_size:
                continue
            for start in range(T - window_size + 1):
                self.index_map.append((i, start))
        if not self.index_map:
            raise ValueError(
                f"No series of length >= {window_size}; cannot build any windows."
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        series_idx, start = self.index_map[idx]
        series = self.series_list[series_idx]   # (T, D)
        T, D = series.shape

        window = series[start : start + self.window_size].astype(np.float32)  # (W, D)

        if self.normalize:
            # --- normalize per feature over the window ---
            mean = window.mean(axis=0, keepdims=True)   # shape (1, D)
            std  = window.std(axis=0, keepdims=True)    # shape (1, D)
            std[std == 0] = 1.0                         # avoid divide-by-zero
            window = (window - mean) / std

        return {
            'window':     torch.from_numpy(window),           # FloatTensor (W, D)
            'start_idx':  torch.tensor(start, dtype=torch.long),
            'series_len': torch.tensor(T, dtype=torch.long),
        }


class SequenceWindowDataset(Dataset):
    """
    Dataset that returns *sequences* of contiguous windows from each series,
    suitable for LSTM-based conditioning over windows.

    For each series of length T and chosen window_size W and seq_len S, we form
    sequences of S windows:

        [window starting at t,
         window starting at t+1,
         ...,
         window starting at t+S-1]

    where each window has length W in time. This requires:

        t + (S - 1) + W <= T  =>  t <= T - (W + S - 1)

    Each item is a dict containing:
      - 'window':     FloatTensor (S, W, D_features)
      - 'start_idx':  LongTensor (S,) with the start index of each window
      - 'series_len': LongTensor (S,) with the length T of the full series
    """
    def __init__(
        self,
        series_list: Union[np.ndarray, Sequence[np.ndarray]],
        window_size: int,
        seq_len: int,
        normalize: bool = False,
        seq_stride: int = 1,
    ):
        """
        Args:
            series_list: np.ndarray of shape (N, T, D) or sequence of (T, D) arrays.
            window_size: W, length of each window.
            seq_len:     S, number of windows per sequence.
            normalize:   if True, normalize each window separately (per-feature z-score).
            seq_stride:  stride in *time indices* between consecutive sequences
                         within the same series (default: 1).
        """
        super().__init__()
        if isinstance(series_list, np.ndarray):
            # series_list is assumed to be (N, T, D)
            self.series_list = [s for s in series_list]
        else:
            self.series_list = list(series_list)

        self.window_size = window_size
        self.seq_len = seq_len
        self.normalize = normalize
        self.seq_stride = seq_stride

        self.index_map = []  # entries are (series_idx, start_time_index_for_first_window)

        for i, series in enumerate(self.series_list):
            T, D = series.shape
            # Need enough length for S windows of length W, shifted by 1 each
            min_total = self.window_size + self.seq_len - 1
            if T < min_total:
                continue

            # maximal start index for the first window in the sequence
            max_start = T - min_total
            for start_t in range(0, max_start + 1, self.seq_stride):
                self.index_map.append((i, start_t))

        if not self.index_map:
            raise ValueError(
                f"No valid sequences with window_size={window_size} and seq_len={seq_len}; "
                f"check that your series are long enough."
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        series_idx, start_t = self.index_map[idx]
        series = self.series_list[series_idx]   # (T, D)
        T, D = series.shape

        S = self.seq_len
        W = self.window_size

        # Allocate array for the sequence of windows: (S, W, D)
        windows = np.zeros((S, W, D), dtype=np.float32)
        start_indices = np.zeros(S, dtype=np.int64)

        for s in range(S):
            win_start = start_t + s
            win_end = win_start + W
            window = series[win_start:win_end].astype(np.float32)  # (W, D)

            if self.normalize:
                # per-window normalization (like WindowDataset)
                mean = window.mean(axis=0, keepdims=True)   # (1, D)
                std  = window.std(axis=0, keepdims=True)    # (1, D)
                std[std == 0] = 1.0
                window = (window - mean) / std

            windows[s] = window
            start_indices[s] = win_start

        # series_len repeated for each window in the sequence
        series_len = np.full(S, T, dtype=np.int64)

        return {
            'window':     torch.from_numpy(windows),                     # (S, W, D)
            'start_idx':  torch.from_numpy(start_indices),               # (S,)
            'series_len': torch.from_numpy(series_len),                  # (S,)
        }
