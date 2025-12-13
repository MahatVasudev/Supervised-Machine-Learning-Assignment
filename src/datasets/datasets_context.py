import pandas as pd
import os
from ..config import SRC_DATASET_DIR, DATASETS_DIR
from .fire_dataset import FireDataset, FireGridAutoEncoderDataset
from torch.utils.data import DataLoader
from typing import Type, Optional, List, Tuple
from .config import DatasetContextConfig
from src.utils.logger import Logger


class FireDatasetContext:
    def __init__(self, dataset: Type[FireDataset], config: DatasetContextConfig):
        assert issubclass(
            dataset, FireDataset), "The `dataset` parameter must belong to FireDataset class"

        assert isinstance(
            config, DatasetContextConfig), "The `config` parameter must belong to a DatasetContextConfig Class"

        self.cfg = config
        self.dataset = dataset
        if not self.cfg.test_needed:
            self.cfg.val_size = 1 - self.cfg.train_size
            self.cfg.test_size = 0

        self.train_dates, self.valid_dates, self.test_dates = self.generate_dates()

    def generate_dates(self) -> Tuple[List, List, Optional[List]]:
        dfs = [pd.read_csv(f, usecols=['date']) for f in self.cfg.csv_files]
        all_dates_df = pd.concat(dfs, ignore_index=True)
        all_dates = sorted(all_dates_df['date'].astype(
            'datetime64[ns]').unique())
        n_total = len(all_dates)
        n_train = int(self.cfg.train_size * n_total)
        n_val = int(self.cfg.val_size * n_total)
        train_dates = all_dates[:n_train]
        valid_dates = all_dates[n_train: n_train+n_val]

        if not self.cfg.test_needed:
            return train_dates, valid_dates, None

        test_dates = all_dates[n_train + n_val:]
        return train_dates, valid_dates, test_dates

    def load_dataset(self) -> Tuple[FireDataset, FireDataset, Optional[FireDataset]]:
        train_dataset = self.dataset(
            csv_files=self.cfg.csv_files,
            seq_len=self.cfg.seq_len,
            dates=self.train_dates,
            lat_col=self.cfg.lat_col,
            lon_col=self.cfg.long_col,
            target_col=self.cfg.target_col,
            cache_dir=self.cfg.cache_dir,
            downsample=self.cfg.downsample
        )

        valid_dataset = self.dataset(
            csv_files=self.cfg.csv_files,
            seq_len=self.cfg.seq_len,
            dates=self.valid_dates,
            lat_col=self.cfg.lat_col,
            lon_col=self.cfg.long_col,
            target_col=self.cfg.target_col,
            cache_dir=self.cfg.cache_dir,
            downsample=self.cfg.downsample
        )

        test_dataset = None
        self.check_change_global_variable(train_dataset)
        self.check_change_global_variable(valid_dataset)
        if self.cfg.test_needed:
            test_dataset = self.dataset(
                csv_files=self.cfg.csv_files,
                seq_len=self.cfg.seq_len,
                dates=self.test_dates,
                lat_col=self.cfg.lat_col,
                lon_col=self.cfg.long_col,
                target_col=self.cfg.target_col,
                cache_dir=self.cfg.cache_dir,
                downsample=self.cfg.downsample
            )
            self.check_change_global_variable(test_dataset)

        return train_dataset, valid_dataset, test_dataset

    def check_change_global_variable(self, dataset: FireDataset):
        if self.cfg.global_max is not None:
            assert self.cfg.global_max >= 0, "Count cannot be negative"
            dataset.set_global_max(self.cfg.global_max)

        if self.cfg.global_min is not None:
            assert self.cfg.global_min >= 0, "Count cannot be negative"
            dataset.set_global_min(self.cfg.global_min)
        if self.cfg.global_mean is not None:
            assert self.cfg.global_mean >= 0, "Count cannot be negative"
            dataset.set_global_mean(self.cfg.global_mean)
        if self.cfg.global_std is not None:
            dataset.set_global_std(self.cfg.global_std)

    def load_dataloader(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        train, valid, test = self.load_dataset()

        train_loader = DataLoader(
            train, batch_size=self.cfg.batch_size['train'],
            shuffle=self.cfg.shuffle['train'],
            num_workers=self.cfg.num_workers['train']
        )
        valid_loader = DataLoader(
            valid, batch_size=self.cfg.batch_size['val'],
            shuffle=self.cfg.shuffle['val'],
            num_workers=self.cfg.num_workers['val']
        )

        test_loader = None

        if self.cfg.test_needed:
            test_loader = DataLoader(
                test, batch_size=self.cfg.batch_size['test'],
                shuffle=self.cfg.shuffle['test'],
                num_workers=self.cfg.num_workers['test']
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
            f"Global max         : {
                self.cfg.global_max if self.cfg.global_max is not None else 'dataset default'}",
            f"Global min         : {
                self.cfg.global_min if self.cfg.global_min is not None else 'dataset default'}",
            f"Global mean        : {
                self.cfg.global_mean if self.cfg.global_mean is not None else 'dataset default'}",
            f"Global std         : {
                self.cfg.global_std if self.cfg.global_std is not None else 'dataset default'}",
        )


csv_files = [os.path.join(DATASETS_DIR, "global_modis", "final_2020_by_day.csv"),
             os.path.join(DATASETS_DIR, "global_modis",
                          "final_2021_by_day.csv"),
             os.path.join(DATASETS_DIR, "global_modis",
                          "final_2024_by_day.csv")
             ]

DEFAULT_AUTOENCODER_DATA_CONFIG = DatasetContextConfig(
    csv_files=csv_files,
    seq_len=0,
    cache_dir=os.path.join(SRC_DATASET_DIR, 'autoencoder_cache', 'cache'),
    train_size=0.7,
    test_needed=False,
    shuffle=dict(train=True, val=False, test=False)
)

DEFAULT_CONVLSTM_DATA_CONFIG = DatasetContextConfig(
    csv_files=csv_files,
    seq_len=7,
    cache_dir=os.path.join(SRC_DATASET_DIR, 'conv_cache', 'cache'),
    train_size=0.7,
    test_needed=False,
    shuffle=dict(train=True, val=False, test=False)
)

if __name__ == "__main__":
    cnfg = DEFAULT_AUTOENCODER_DATA_CONFIG
    context = FireDatasetContext(
        dataset=FireGridAutoEncoderDataset, config=cnfg)

    context.summary()

    train_data, val_data, test_data = context.load_dataloader()
