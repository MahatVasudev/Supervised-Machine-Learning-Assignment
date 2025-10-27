import torch
import numpy as np
from torch.utils.data import Dataset,  DataLoader
import pandas as pd
import os


class FireDataset(Dataset):
    def __init__(self, dfs: list[str], seq_len=7,
                 spatial_features=['lat_bin', 'long_bin'],
                 features=['fire_count'], date_feature='date', target='fire_count'):
        self.seq_len = seq_len
        self.target = target
        self.features = features
        self.str_dfs = dfs
        self.data = []
        self.spatial_features = spatial_features
        self.date_feature = date_feature
        self.file_index_map = self.__build_index_map()

    def __len__(self):
        return sum((end - start) for _, start, end in self.file_index_map)

    def __getitem__(self, index: int):
        for path, start, end in self.file_index_map:
            if start <= index < end:
                local_idx = index - start
                df = self.__load_dataset(path)
                break

        sequences = self.make_sequence_data(data=df, update_self=False)
        X, y = sequences[local_idx % len(sequences)]

        return torch.tensor(X), torch.tensor(y)

    def __build_index_map(self) -> list[list[str, int, int]]:
        index_map = []
        total = 0
        for path in self.str_dfs:
            count = sum(1 for _ in open(path)) - 1
            index_map.append((path, total, total + count))
            total += count

        return index_map

    def __load_dataset(self, path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df[self.date_feature] = pd.to_datetime(
            df[self.date_feature], format="%Y-%m-%d")

        return df

    def __make_dataset(self):
        dfs = []
        for data in self.str_dfs:
            dfs.append(pd.read_csv(data))

        data_full = pd.concat(dfs, ignore_index=True)

        data_full[self.date_feature] = pd.to_datetime(
            data_full[self.date_feature], format='%Y-%m-%d')

        self.make_sequence_data(data_full, update_self=True)

    def make_sequence_data(self, data: pd.DataFrame | None = None, update_self: bool = False):

        cdata = data if data is not None else self.data

        seq_data = []
        for _, group in cdata.groupby(self.spatial_features):
            group = group.sort_values(self.date_feature)
            values = group[self.spatial_features +
                           self.features + [self.target]].values
            for i in range(len(values) - self.seq_len):
                X = values[i:i+self.seq_len, :-1]
                Y = values[i+self.seq_len, -1]
                seq_data.append((X, Y))

        if update_self:
            self.data = seq_data
            del seq_data
        else:
            return seq_data


class FireSpreadDatasetLazy(Dataset):
    def __init__(self, csv_files: list[str], seq_len=7,
                 lat_col='lat_bin', lon_col='long_bin', date_col='date',
                 target_col='fire_count', downsample=1,
                 dates: list = None, cache_dir="./cache"):
        """
        Optimized Lazy-loading dataset for large fire data with:
        - One-time CSV read at init
        - In-memory & optional on-disk cache
        - Faster grid lookup by date
        """
        self.seq_len = seq_len
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.date_col = date_col
        self.target_col = target_col
        self.csv_files = csv_files
        self.downsample = downsample
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 1️⃣ Load metadata once
        dfs_meta = []
        for f in csv_files:
            df = pd.read_csv(
                f, usecols=[lat_col, lon_col, date_col, target_col])
            df[date_col] = pd.to_datetime(df[date_col])
            dfs_meta.append(df)
        self.data_meta = pd.concat(dfs_meta, ignore_index=True)

        # 2️⃣ Create bins
        self.lat_bins = np.sort(self.data_meta[lat_col].unique())[::downsample]
        self.lon_bins = np.sort(self.data_meta[lon_col].unique())[::downsample]
        self.num_lat = len(self.lat_bins)
        self.num_lon = len(self.lon_bins)

        # 3️⃣ Extract all unique dates
        all_dates_sorted = sorted(self.data_meta[date_col].unique())
        self.dates = sorted(dates) if dates is not None else all_dates_sorted
        self.num_sequences = len(self.dates) - self.seq_len

        # 4️⃣ Group by date once
        self.data_by_date = {
            date: df for date, df in self.data_meta.groupby(self.data_meta[date_col])
        }

        # 5️⃣ RAM cache
        self._grid_cache = {}

    def __len__(self):
        return self.num_sequences

    def _cache_path(self, date):
        """Get on-disk cache path for this date."""
        return os.path.join(self.cache_dir, f"{str(date.date())}.npy")

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

        # Target day
        Y_date = self.dates[idx + self.seq_len]
        Y_grid = self._load_grid_for_day(Y_date)[np.newaxis, :, :]  # (1, H, W)

        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(Y_grid, dtype=torch.float32)


csv_files = ["../datasets/global_modis/final_2020_by_day.csv",
             "../datasets/global_modis/final_2021_by_day.csv",
             "../datasets/global_modis/final_2024_by_day.csv"]

dfs = [pd.read_csv(f, usecols=['date']) for f in csv_files]

all_dates_df = pd.concat(dfs, ignore_index=True)

all_dates = sorted(all_dates_df['date'].astype('datetime64[ns]').unique())

n_total = len(all_dates)

n_train = int(0.7 * n_total)

n_val = int(0.15 * n_total)

train_dates = all_dates[:n_train]
test_dates = all_dates[n_train: n_train+n_val]
valid_dates = all_dates[n_train+n_val:]

seq_len = 7
train_dataset = FireSpreadDatasetLazy(
    csv_files, seq_len=seq_len, dates=train_dates)
val_dataset = FireSpreadDatasetLazy(
    csv_files, seq_len=seq_len, dates=valid_dates)

batch_size = 2
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader = DataLoader(
    val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
# test_loader = DataLoader(
#   test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)


if __name__ == '__main__':
    print(os.curdir)
