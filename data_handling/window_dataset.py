import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Sequence, Union


class WindowDataset(Dataset):
    """
    Deterministic “window‐of‐series” dataset with optional per‐feature z‐score normalization.
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
        normalize: bool = False  # <-- NEW ARGUMENT
    ):
        super().__init__()
        # Normalize input into a list of (T, D) numpy arrays
        if isinstance(series_list, np.ndarray):
            self.series_list = [s for s in series_list]
        else:
            self.series_list = list(series_list)

        self.window_size = window_size
        self.normalize = normalize  # <-- SAVE IT

        # Build index map: each entry is (series_idx, start_idx)
        self.index_map = []
        print(np.shape(self.series_list))
        for i, series in enumerate(self.series_list):
            T, D = series.shape
            print(T, D, i)
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

        window = series[start : start + self.window_size].astype(np.float32)  # (window_size, D)

        if self.normalize:
            # --- normalize per feature over the window ---
            mean = window.mean(axis=0, keepdims=True)   # shape (1, D)
            std  = window.std(axis=0, keepdims=True)    # shape (1, D)
            std[std == 0] = 1.0                          # avoid divide-by-zero
            window = (window - mean) / std

        return {
            'window':     torch.from_numpy(window),           # FloatTensor (window_size, D)
            'start_idx':  torch.tensor(start, dtype=torch.long),
            'series_len': torch.tensor(T, dtype=torch.long),
        }
