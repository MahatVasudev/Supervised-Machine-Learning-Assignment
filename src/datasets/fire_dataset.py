import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from ..config import CACHE_DIR


class FireDataset(Dataset):
    def __init__(self, csv_files: list[str], seq_len=7, lat_col='lat_bin', lon_col='long_bin', date_col='date', target_col='fire_count', downsample=1, dates: list | None = None, cache_dir=CACHE_DIR):
        self.seq_len = seq_len
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.date_col = date_col
        self.target_col = target_col
        self.csv_files = csv_files
        self.downsample = downsample
        self.cache_dir = cache_dir
        self.global_max = 1036  # default
        self.global_min = 0  # default
        self.global_mean = 0.33488836147712114  # default
        self.global_std = 2.925631213095511  # default
        self.dates = dates

        os.makedirs(cache_dir, exist_ok=True)

    def set_global_mean(self, global_mean):
        self.global_mean = global_mean

    def set_global_min(self, global_min):
        self.global_min = global_min

    def set_global_max(self, global_max):
        self.global_max = global_max

    def set_global_std(self, global_std):
        self.global_std = global_std

    def _cache_path(self, date):
        """Get on-disk cache path for this date."""
        return os.path.join(self.cache_dir, f"{str(date.date())}.npy")


class FireSpreadDatasetLazy(FireDataset, Dataset):
    def __init__(self, csv_files: list[str], seq_len=7,
                 lat_col='lat_bin', lon_col='long_bin', date_col='date',
                 target_col='fire_count', downsample=1,
                 dates: list = None, cache_dir=CACHE_DIR):
        """
        Optimized Lazy-loading dataset for large fire data with:
        - One-time CSV read at init
        - In-memory & optional on-disk cache
        - Faster grid lookup by date
        """

        super().__init__(seq_len=seq_len, lat_col=lat_col, lon_col=lon_col, date_col=date_col,
                         target_col=target_col, csv_files=csv_files, dates=dates, downsample=downsample, cache_dir=cache_dir)

        dfs_meta = []

        for f in self.csv_files:
            df = pd.read_csv(
                f, usecols=[lat_col, lon_col, date_col, target_col])
            df[date_col] = pd.to_datetime(df[date_col])
            dfs_meta.append(df)
        self.data_meta = pd.concat(dfs_meta, ignore_index=True)

        self.lat_bins = np.sort(self.data_meta[lat_col].unique())[::downsample]
        self.lon_bins = np.sort(self.data_meta[lon_col].unique())[::downsample]
        self.num_lat = len(self.lat_bins)
        self.num_lon = len(self.lon_bins)

        all_dates_sorted = sorted(self.data_meta[date_col].unique())
        self.dates = sorted(dates) if dates is not None else all_dates_sorted
        self.num_sequences = len(self.dates) - self.seq_len

        self.data_by_date = {
            date: df for date, df in self.data_meta.groupby(self.data_meta[date_col])
        }

        self._grid_cache = {}

    def __len__(self):
        return self.num_sequences

    def _load_grid_for_day(self, date):
        """Load grid for a given date (RAM or disk cache if possible)."""
        if date in self._grid_cache:
            return self._grid_cache[date]

        # Check disk cache
        cache_path = self._cache_path(date)
        if os.path.exists(cache_path):
            grid = np.load(cache_path)
            self._grid_cache[date] = grid
            return grid

        # Otherwise build from grouped data
        if date not in self.data_by_date:
            grid = np.zeros((self.num_lat, self.num_lon), dtype=np.float32)
            self._grid_cache[date] = grid
            return grid

        df_day = self.data_by_date[date]
        temp = df_day.pivot(
            index=self.lat_col, columns=self.lon_col, values=self.target_col).fillna(0)
        temp = temp.reindex(index=self.lat_bins,
                            columns=self.lon_bins, fill_value=0)
        grid = temp.values.astype(np.float32)

        # Save to disk for next time
        np.save(cache_path, grid)
        self._grid_cache[date] = grid
        return grid

    def __getitem__(self, idx):
        # Sequence of input days
        seq_dates = self.dates[idx:idx + self.seq_len]
        X_seq = np.stack([self._load_grid_for_day(d)
                         for d in seq_dates], axis=0)
        X_seq = X_seq[:, np.newaxis, :, :]  # (seq_len, 1, H, W)
        X_seq = (X_seq - self.global_mean) / (self.global_std)
        # Target day
        Y_date = self.dates[idx + self.seq_len]
        Y_grid = self._load_grid_for_day(Y_date)[np.newaxis, :, :]  # (1, H, W)

        Y_grid = (Y_grid - self.global_mean) / self.global_std
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(Y_grid, dtype=torch.float32)


class FireGridAutoEncoderDataset(FireDataset, Dataset):
    def __init__(self, csv_files: list[str],
                 lat_col='lat_bin', lon_col='long_bin', date_col='date',
                 target_col='fire_count', downsample=1,
                 dates: list = None, cache_dir=CACHE_DIR, seq_len=None):
        """
        Single-day dataset for AutoEncoder training.
        - Fully supports disk caching (.npy)
        - Returns (X, X) for reconstruction
        - Loads each day lazily + caches in RAM
        """

        super().__init__(seq_len=0, lat_col=lat_col, lon_col=lon_col,
                         date_col=date_col, target_col=target_col, csv_files=csv_files,
                         dates=dates, downsample=downsample, cache_dir=cache_dir)

        # ---- Load metadata once ----
        dfs_meta = []
        for f in self.csv_files:
            df = pd.read_csv(
                f, usecols=[lat_col, lon_col, date_col, target_col])
            df[date_col] = pd.to_datetime(df[date_col])
            dfs_meta.append(df)

        self.data_meta = pd.concat(dfs_meta, ignore_index=True)

        # Spatial bins
        self.lat_bins = np.sort(self.data_meta[lat_col].unique())[::downsample]
        self.lon_bins = np.sort(self.data_meta[lon_col].unique())[::downsample]
        self.num_lat = len(self.lat_bins)
        self.num_lon = len(self.lon_bins)

        # Dates to use
        all_dates_sorted = sorted(self.data_meta[date_col].unique())
        self.dates = sorted(dates) if dates is not None else all_dates_sorted

        self.num_sequences = len(self.dates)  # each day is a sample

        # Group once by date
        self.data_by_date = {
            date: df for date, df in self.data_meta.groupby(self.data_meta[date_col])
        }

        # RAM cache for grids
        self._grid_cache = {}

    def __len__(self):
        return self.num_sequences

    def _load_grid_for_day(self, date):
        """Load a single day's grid using RAM cache or disk cache."""
        # 1) RAM cache
        if date in self._grid_cache:
            return self._grid_cache[date]

        # 2) Disk cache
        cache_path = self._cache_path(date)
        if os.path.exists(cache_path):
            grid = np.load(cache_path)
            self._grid_cache[date] = grid
            return grid

        # 3) Build from CSV metadata
        if date not in self.data_by_date:
            grid = np.zeros((self.num_lat, self.num_lon), dtype=np.float32)
            self._grid_cache[date] = grid
            return grid

        df_day = self.data_by_date[date]
        temp = df_day.pivot(index=self.lat_col,
                            columns=self.lon_col,
                            values=self.target_col).fillna(0)

        temp = temp.reindex(index=self.lat_bins,
                            columns=self.lon_bins,
                            fill_value=0)

        grid = temp.values.astype(np.float32)

        # Save to disk for future fast loading
        np.save(cache_path, grid)
        self._grid_cache[date] = grid

        return grid

    def __getitem__(self, idx):
        """Return (X, X) for AutoEncoder training."""
        date = self.dates[idx]

        grid = self._load_grid_for_day(date)
        grid = grid[np.newaxis, :, :]  # (1, H, W)

        # Normalize
        grid = (grid - self.global_mean) / self.global_std

        X = torch.tensor(grid, dtype=torch.float32)
        return X, X  # target is same as input
