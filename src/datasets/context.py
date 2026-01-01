import gc
import math
import os
from typing import List, Literal, Optional, Tuple, Type

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.utils.data_tools import (final_global_filename,
                                  find_final_data_bytime,
                                  find_final_metadata_bytime)
from src.utils.logger import Logger
from src.utils.utils import function_profiler

from ..config import GLOBAL_MODIS_DIR, SRC_DATASET_DIR
from .config import DatasetContextConfig
from .fire_dataset import (FireDataset, FireGridAutoEncoderDataset,
                           FireSpreadDatasetLazy)

# TODO: Add Regional Only Data Support


@function_profiler(
    save_result=True, show_result=False, result_name="FireDatasetContext Lifetime"
)
class FireDatasetContext:
    def __init__(
        self,
        dataset: Type[FireDataset],
        config: DatasetContextConfig,
        workers: float = 0.2,
    ):
        assert issubclass(
            dataset, FireDataset
        ), "The `dataset` parameter must belong to FireDataset class"

        assert isinstance(
            config, DatasetContextConfig
        ), "The `config` parameter must belong to a DatasetContextConfig Class"

        assert (
            0 < workers <= 1
        ), f"The workers cannot be less than 0 and greater 1; Given value workers:{workers}"

        self.cfg = config

        # takes workersx10% of the data and process individually on that amount, trying to reduce memory overhead
        self.workers = workers
        self.n_csv = len(self.cfg.csv_files)
        self.dataset = dataset

        if self.n_csv == 1:
            Logger.warning_line("A single worker will be used")
            self.workers = 1

        self.process_at_time = max(math.ceil(self.workers * self.n_csv), 1)
        if not self.cfg.test_needed:
            self.cfg.val_size = 1 - self.cfg.train_size
            self.cfg.test_size = 0

        self.train_dates, self.valid_dates, self.test_dates = self.generate_dates()
        self.lat_bins, self.long_bins = self.generate_lat_long()
        self.find_global_stats()

    @function_profiler(
        save_result=True, show_result=False, result_name="Generate Dates Lifetime"
    )
    def generate_dates(self) -> Tuple[List, List, Optional[List]]:

        dfs = []
        train_dates = []
        val_dates = []
        test_dates = []
        for fidx in range(self.n_csv):
            dfs.append(
                pd.read_csv(self.cfg.csv_files[fidx], usecols=[self.cfg.date_col])
            )

            if len(dfs) == self.process_at_time or fidx == self.n_csv - 1:
                all_dates_df = pd.concat(dfs)
                all_dates = sorted(
                    all_dates_df["date"].astype("datetime64[ns]").unique()
                )
                n_total = len(all_dates)
                n_train = int(self.cfg.train_size * n_total)
                n_val = int(self.cfg.val_size * n_total)
                train_dates.extend(all_dates[:n_train])
                val_dates.extend(all_dates[n_train : n_train + n_val])
                if self.cfg.test_needed:
                    test_dates.extend(all_dates[n_train + n_val :])
                dfs = []
        n_len = len(train_dates) + len(val_dates) + len(test_dates)
        print(len(train_dates) / n_len, len(val_dates) / n_len, len(test_dates) / n_len)

        del dfs
        return train_dates, val_dates, test_dates

    @function_profiler(
        save_result=True, show_result=False, result_name="Generate Lat Long Lifetime"
    )
    def generate_lat_long(self) -> Tuple[List, List]:

        all_lats: set = set()
        all_longs: set = set()
        lat_df = []
        long_df = []
        for fidx in range(self.n_csv):
            lat_df.append(
                pd.read_csv(self.cfg.csv_files[fidx], usecols=[self.cfg.lat_col])
            )

            long_df.append(
                pd.read_csv(self.cfg.csv_files[fidx], usecols=[self.cfg.long_col])
            )
            if len(lat_df) == self.process_at_time or fidx == self.n_csv - 1:
                new_lats = pd.concat(lat_df)

                new_longs = pd.concat(long_df)
                new_lats = np.array(new_lats[self.cfg.lat_col].unique())[
                    :: self.cfg.downsample
                ]

                new_longs = np.array(new_longs[self.cfg.long_col].unique())[
                    :: self.cfg.downsample
                ]
                all_lats = all_lats.union(set(new_lats))
                all_longs = all_longs.union(set(new_longs))

                lat_df = []
                long_df = []
        del lat_df, long_df

        lats: list = sorted(all_lats)
        longs: list = sorted(all_longs)
        return lats, longs

    def load_dataset(self) -> Tuple[FireDataset, FireDataset, Optional[FireDataset]]:
        train_dataset = self.dataset(
            csv_files=self.cfg.csv_files,
            seq_len=self.cfg.seq_len,
            dates=self.train_dates,
            lat_col=self.cfg.lat_col,
            lon_col=self.cfg.long_col,
            target_col=self.cfg.target_col,
            cache_dir=self.cfg.cache_dir,
            standardization=self.cfg.standardization_method,
            downsample=self.cfg.downsample,
        )

        valid_dataset = self.dataset(
            csv_files=self.cfg.csv_files,
            seq_len=self.cfg.seq_len,
            dates=self.valid_dates,
            lat_col=self.cfg.lat_col,
            lon_col=self.cfg.long_col,
            target_col=self.cfg.target_col,
            cache_dir=self.cfg.cache_dir,
            standardization=self.cfg.standardization_method,
            downsample=self.cfg.downsample,
        )

        test_dataset = None
        self.check_change_global_variable(train_dataset)
        self.check_change_global_variable(valid_dataset)
        train_dataset.set_latlongbins(latbins=self.lat_bins, longbins=self.long_bins)
        valid_dataset.set_latlongbins(latbins=self.lat_bins, longbins=self.long_bins)
        if self.cfg.test_needed:
            test_dataset = self.dataset(
                csv_files=self.cfg.csv_files,
                seq_len=self.cfg.seq_len,
                dates=self.test_dates,
                lat_col=self.cfg.lat_col,
                lon_col=self.cfg.long_col,
                target_col=self.cfg.target_col,
                cache_dir=self.cfg.cache_dir,
                standardization=self.cfg.standardization_method,
                downsample=self.cfg.downsample,
            )
            self.check_change_global_variable(test_dataset)

            test_dataset.set_latlongbins(latbins=self.lat_bins, longbins=self.long_bins)
        return train_dataset, valid_dataset, test_dataset

    def find_global_stats(self):

        if self.cfg.metadata_csv_files == []:

            return

        global_max, global_min, global_mean, global_std = None, None, None, None

        for f in self.cfg.metadata_csv_files:
            if not os.path.exists(f):
                Logger.warning_line(
                    f"{f} not found; skipping; but may not give accurate results"
                )
                continue

            df = pd.read_csv(f, index_col=0)

            if "fire_count" not in df.columns:
                Logger.warning_line(f"fire_count not found; skipping file")
                continue

            max_val = df.loc["max", "fire_count"]
            min_val = df.loc["min", "fire_count"]
            mean_val = df.loc["mean", "fire_count"]
            std_val = df.loc["std", "fire_count"]

            # Update global metrics across all files
            global_max = max(max_val, global_max) if global_max is not None else max_val
            global_min = min(min_val, global_min) if global_min is not None else min_val
            global_mean = (
                mean_val if global_mean is None else (global_mean + mean_val) / 2
            )
            global_std = std_val if global_std is None else (global_std + std_val) / 2

        self.global_max = global_max
        self.global_min = global_min
        self.global_mean = global_mean
        self.global_std = global_std

        gc.collect()

    def check_change_global_variable(self, dataset: FireDataset):
        dataset.set_global_max(self.global_max)
        dataset.set_global_min(self.global_min)
        dataset.set_global_mean(self.global_mean)
        dataset.set_global_std(self.global_std)

    @function_profiler(
        interval=1e-5,
        save_result=True,
        show_result=True,
        result_name="dataloader lifetime",
    )
    def load_dataloader(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        train, valid, test = self.load_dataset()

        train_loader = DataLoader(
            train,
            batch_size=self.cfg.batch_size["train"],
            shuffle=self.cfg.shuffle["train"],
            num_workers=self.cfg.num_workers["train"],
        )
        valid_loader = DataLoader(
            valid,
            batch_size=self.cfg.batch_size["val"],
            shuffle=self.cfg.shuffle["val"],
            num_workers=self.cfg.num_workers["val"],
        )

        test_loader = None

        if self.cfg.test_needed and test is not None:
            test_loader = DataLoader(
                test,
                batch_size=self.cfg.batch_size["test"],
                shuffle=self.cfg.shuffle["test"],
                num_workers=self.cfg.num_workers["test"],
            )

        return train_loader, valid_loader, test_loader

    def summary(self) -> None:
        """
        Print a concise summary of dataset configuration and splits.
        Does NOT preload full datasets.
        """

        total_dates = (
            len(self.train_dates)
            + len(self.valid_dates)
            + (len(self.test_dates) if self.test_dates else 0)
        )

        Logger.info(
            "FireDatasetContext Summary",
            f"Dataset class      : {self.dataset.__name__}",
            f"CSV files          : {len(self.cfg.csv_files)} files",
            *[f"  - {f}" for f in self.cfg.csv_files],
            f"Metadata files     : {len(
                self.cfg.metadata_csv_files) if self.cfg.metadata_csv_files != [] else 'Not Given'}",
            *[f"  - {f}" for f in self.cfg.metadata_csv_files],
            f"Cache directory    : {self.cfg.cache_dir}",
        )

        Logger.info(
            "Split configuration",
            f"Total days         : {total_dates}",
            f"Train days         : {len(self.train_dates)} ({
                self.cfg.train_size:.0%})",
            f"Validation days    : {len(self.valid_dates)} ({
                self.cfg.val_size:.0%})",
            f"Test days          : {
                len(self.test_dates) if self.test_dates else 0}",
            f"Test enabled       : {self.cfg.test_needed}",
        )

        Logger.info(
            "Sample structure",
            f"Sequence length    : {self.cfg.seq_len}",
            f"Downsample factor  : {self.cfg.downsample}",
            f"Latitude column   : {self.cfg.lat_col}",
            f"Longitude column  : {self.cfg.long_col}",
            f"Target column     : {self.cfg.target_col}",
        )

        Logger.info(
            "Dataloader config",
            f"Batch size (train) : {self.cfg.batch_size['train']}",
            f"Batch size (val)   : {self.cfg.batch_size['val']}",
            f"Batch size (test)  : {self.cfg.batch_size['test']}",
            f"Shuffle (train)   : {self.cfg.shuffle['train']}",
            f"Shuffle (val)     : {self.cfg.shuffle['val']}",
            f"Shuffle (test)    : {self.cfg.shuffle['test']}",
            f"Workers (train)   : {self.cfg.num_workers['train']}",
            f"Workers (val)     : {self.cfg.num_workers['val']}",
            f"Workers (test)    : {self.cfg.num_workers['test']}",
        )

        Logger.info(
            "Normalization",
            f"Mode: {self.cfg.standardization_method}",
            f"Global max         : {self.global_max}",
            f"Global min         : {self.global_min}",
            f"Global mean        : {self.global_mean}",
            f"Global std         : {self.global_std}",
        )


def get_only_filenames(
    years: List[str | int],
    n_time: str,
    mode: Literal["final"] | Literal["metadata"] = "final",
) -> List[str]:

    if mode == "final":
        data = find_final_data_bytime(years, n_time)
    elif mode == "metadata":
        data = find_final_metadata_bytime(years, n_time)
    else:
        data = []
        Logger.fatal("get_only_filenames: Mode not found")

    # HACK: Need to find a better way to code this, very inefficient
    paths = []

    if data == []:
        Logger.fatal("get_only_filenames: No Datasets found, please generate it...")

    for _, path in data:
        paths.append(path)

    return paths


def __test_get_only_final_filenames():
    print("Testing Get only final filenames function")

    years_args = [
        (2020, "2021", 2022),
        (2022, 2023, 2024),
        ("2018", "2019", "2020"),
        ("2015", "2016"),
        ("2020", "2016", "2015", "2018"),
    ]
    n_time = "1d"

    expected_results = [
        [
            os.path.join(GLOBAL_MODIS_DIR, final_global_filename(x, n_time))
            for x in ("2020", "2021", "2022")
        ],
        [
            os.path.join(GLOBAL_MODIS_DIR, final_global_filename(x, n_time))
            for x in ("2022", "2023", "2024")
        ],
        [
            os.path.join(GLOBAL_MODIS_DIR, final_global_filename(x, n_time))
            for x in ("2018", "2019", "2020")
        ],
        [],
        [
            os.path.join(GLOBAL_MODIS_DIR, final_global_filename(x, n_time))
            for x in ("2020", "2018")
        ],
    ]

    for test_arg in range(len(years_args)):
        result = get_only_filenames(years_args[test_arg], n_time=n_time)

        if result == expected_results[test_arg]:
            Logger.testing_success(
                f"Got Correct Result, expected: {
                    expected_results[test_arg]}, got: {result}"
            )

        else:
            Logger.testing_failure(
                f"Got Incorrect Result, expected: {
                    expected_results[test_arg]}, got: {result}"
            )


csv_files = get_only_filenames(
    ["2018", "2019", "2020", "2021", "2022", "2023", "2024"], "1d"
)

meta_files = get_only_filenames(
    ["2018", "2019", "2020", "2021", "2022", "2023", "2024"], "1d", mode="metadata"
)
DEFAULT_AUTOENCODER_DATA_CONFIG = DatasetContextConfig(
    csv_files=csv_files,
    metadata_csv_files=meta_files,
    standardization_method="z-score",
    seq_len=0,
    cache_dir=os.path.join(SRC_DATASET_DIR, "autoencoder_cache", "cache"),
    train_size=0.7,
    test_needed=False,
    shuffle=dict(train=True, val=False, test=False),
)

DEFAULT_CONVLSTM_DATA_CONFIG = DatasetContextConfig(
    csv_files=csv_files,
    metadata_csv_files=meta_files,
    standardization_method="min-max",
    seq_len=7,
    cache_dir=os.path.join(SRC_DATASET_DIR, "conv_cache", "cache"),
    train_size=0.7,
    test_needed=False,
    shuffle=dict(train=True, val=False, test=False),
)

if __name__ == "__main__":
    cnfg = DEFAULT_CONVLSTM_DATA_CONFIG
    context = FireDatasetContext(
        dataset=FireSpreadDatasetLazy, workers=0.3, config=cnfg
    )

    context.summary()

    train_data, val_data, test_data = context.load_dataloader()

    for X, y in train_data:
        print(X)
        print(y)
