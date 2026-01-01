import datetime as dt
import os
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CACHE_DIR, DATETIME_FORMAT
from src.utils.logger import Logger


class FireDataset(Dataset):
    def __init__(
        self,
        csv_files: list[str],
        seq_len=7,
        lat_col="lat_bin",
        lon_col="long_bin",
        date_col="date",
        target_col="fire_count",
        downsample=1,
        dates: list | None = None,
        # Another negative of this approach i have to keep track of which date has been cached for atleast 1 epoch, also i have determine whether the whole year has been cached so that i dont load the dataset anymore
        cache_dir=CACHE_DIR,
        standardization: (
            Literal["z-score"] | Literal["min-max"] | Literal["none"]
        ) = "z-score",
    ):

        # TODO: Allow fixed data split for training, testing and validation,
        # this should be done by some 'seed' value but should be able to make same training, validation and testing values even after reboot

        self.seq_len = seq_len
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.date_col = date_col
        self.target_col = target_col
        self.csv_files = csv_files
        self.downsample = downsample
        self.cache_dir = cache_dir
        self.standard_choice = standardization
        # HACK: Total Bullshit values, turns out these are not correct values, better use the metadata values...
        self.cached_year = "0000"
        self.global_max = 1036  # default
        self.global_min = 0  # default
        self.global_mean = 0.33488836147712114  # default
        self.global_std = 2.925631213095511  # default
        self.dates = dates
        self.cache_data: Optional[pd.DataFrame] = None
        self.__make_map_of_files()

        os.makedirs(cache_dir, exist_ok=True)

    def set_global_mean(self, global_mean):
        self.global_mean = global_mean

    def set_global_min(self, global_min):
        self.global_min = global_min

    def set_global_max(self, global_max):
        self.global_max = global_max

    def set_global_std(self, global_std):
        self.global_std = global_std

    def set_latlongbins(self, latbins: List, longbins: List):
        self.lat_bins = latbins
        self.long_bins = longbins

        self.num_lat = len(self.lat_bins)
        self.num_lon = len(self.long_bins)

    def __make_map_of_files(self):
        # DONOT CALL THIS EVER
        map = {}
        for file in self.csv_files:
            ext_year = extract_year_of_final_data(file)
            if ext_year in map:
                Logger.warning_line(
                    f"Given {ext_year} Year data multiple times, skipping.."
                )
                continue
            map[ext_year] = file

        self.file_map = map

    def _standardization(
        self, x: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        match self.standard_choice:
            case "z-score":
                return (x - self.global_mean) / self.global_std

            case "min-max":
                return (x - self.global_max) / (self.global_max - self.global_min)

            case "none":
                return x

            case _:
                Logger.fatal(
                    "standardization technique not found, please enter a valid technique"
                )

    def _cache_exists(self, date):
        return os.path.exists(self._cache_path(date))

    def _load_grid_for_day(self, date: str | pd.Timestamp | dt.datetime) -> np.ndarray:
        if isinstance(date, str):
            new_date = dt.datetime.strptime(date, DATETIME_FORMAT)
        elif isinstance(date, pd.Timestamp):
            new_date = date.to_pydatetime()
        else:
            new_date = date
        year = str(new_date.year)

        if self._cache_exists(date):
            return np.load(self._cache_path(date))

        if self.cached_year != year:
            self.cache_data = pd.read_csv(self.file_map[year])
            self.cached_year = year

        if self.cache_data is None:
            Logger.fatal(f"Cache Data of {year} not loaded")

        if date not in self.cache_data[self.date_col].unique():
            grid = np.zeros((self.num_lat, self.num_lon))
            np.save(self._cache_path(date), grid)
            return grid

        self.grouped = self.cache_data.groupby(self.cache_data[self.date_col])

        df_day = self.grouped.get_group(date)
        temp = df_day.pivot(
            index=self.lat_col, columns=self.lon_col, values=self.target_col
        ).fillna(0)
        temp = temp.reindex(index=self.lat_bins, columns=self.long_bins, fill_value=0)
        grid = temp.values.astype(np.float32)

        # Save to disk for next time
        np.save(self._cache_path(date), grid)
        return grid

    def _cache_path(self, date):
        """Get on-disk cache path for this date."""
        return os.path.join(self.cache_dir, f"{str(date)}.npy")


# Lets make a dictionary which will keep the years as keys and the data path as the value
# What we want to do is, since we will get the date as idx we can get the date from the list,
# and extract year, then we can load the years data and take that specific day or hours or whatever, and then remove it from memory
# The postitives of this approach is that we donot put the whole data into memory, but only the necessary file
# The negative of this approach is that will get peaks of resource usage, like a continous triangle, buut only for the first epoch
# Another negative of this approach i have to keep track of which date has been cached for atleast 1 epoch,
# also i have determine whether the whole year has been cached so that i dont load the dataset anymore


def extract_year_of_final_data(path: str) -> str:
    paths_splited = os.path.split(path)
    filename = paths_splited[-1]

    # HACK: VERY BAD CODE: should work for now tho
    return filename.split("_")[1]


class FireSpreadDatasetLazy(FireDataset, Dataset):
    def __init__(
        self,
        csv_files: list[str],
        seq_len=7,
        lat_col="lat_bin",
        lon_col="long_bin",
        date_col="date",
        target_col="fire_count",
        downsample=1,
        dates: list = [],
        cache_dir=CACHE_DIR,
        standardization: (
            Literal["z-score"] | Literal["min-max"] | Literal["none"]
        ) = "z-score",
    ):
        """
        Optimized Lazy-loading dataset for large fire data with:
        - One-time CSV read at init
        - In-memory & optional on-disk cache
        - Faster grid lookup by date
        """

        super().__init__(
            seq_len=seq_len,
            lat_col=lat_col,
            lon_col=lon_col,
            date_col=date_col,
            target_col=target_col,
            csv_files=csv_files,
            dates=dates,
            downsample=downsample,
            cache_dir=cache_dir,
            standardization=standardization,
        )

        self.dates = dates
        self.num_sequences = len(self.dates) - self.seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Sequence of input days
        seq_dates = self.dates[idx : idx + self.seq_len]

        X_seq = np.stack([self._load_grid_for_day(d) for d in seq_dates], axis=0)
        X_seq = X_seq[:, np.newaxis, :, :]  # (seq_len, 1, H, W)
        X_seq = self._standardization(X_seq)
        # Target day
        Y_date = self.dates[idx + self.seq_len]
        Y_grid = self._load_grid_for_day(Y_date)[np.newaxis, :, :]  # (1, H, W)

        Y_grid = self._standardization(Y_grid)
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(
            Y_grid, dtype=torch.float32
        )


class FireGridAutoEncoderDataset(FireDataset, Dataset):
    def __init__(
        self,
        csv_files: list[str],
        lat_col="lat_bin",
        lon_col="long_bin",
        date_col="date",
        target_col="fire_count",
        downsample=1,
        dates: list = [],
        cache_dir=CACHE_DIR,
        standardization: (
            Literal["z-score"] | Literal["min-max"] | Literal["none"]
        ) = "z-score",
    ):
        """
        Single-day dataset for AutoEncoder training.
        - Fully supports disk caching (.npy)
        - Returns (X, X) for reconstruction
        - Loads each day lazily + caches in RAM
        """

        super().__init__(
            seq_len=0,
            lat_col=lat_col,
            lon_col=lon_col,
            date_col=date_col,
            target_col=target_col,
            csv_files=csv_files,
            dates=dates,
            downsample=downsample,
            cache_dir=cache_dir,
            standardization=standardization,
        )

        # ---- Load metadata once ----

        # Dates to use

        self.dates = dates
        self.num_sequences = len(dates)  # each day is a sample

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """Return (X, X) for AutoEncoder training."""
        date = self.dates[idx]

        grid = self._load_grid_for_day(date)
        grid = grid[np.newaxis, :, :]  # (1, H, W)

        # Normalize
        grid = self._standardization(grid)

        X = torch.tensor(grid, dtype=torch.float32)
        return X, X  # target is same as input
