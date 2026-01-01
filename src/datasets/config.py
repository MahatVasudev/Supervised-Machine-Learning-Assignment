from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict

from ..config import CACHE_DIR


class SplitIntConfig(TypedDict):
    train: int
    val: int
    test: int


class SplitBoolConfig(TypedDict):
    train: bool
    val: bool
    test: bool


@dataclass
class FireDatasetConfig: ...


@dataclass
class DatasetContextConfig:

    # Needed Parameters
    csv_files: list[str]  # list of csv files locations

    metadata_csv_files: list[str] = field(default_factory=lambda: [])
    ###
    batch_size: SplitIntConfig = field(
        default_factory=lambda: {"train": 4, "val": 2, "test": 2}
    )
    shuffle: SplitBoolConfig = field(
        default_factory=lambda: {"train": True, "val": False, "test": False}
    )
    num_workers: SplitIntConfig = field(
        default_factory=lambda: {"train": 4, "val": 0, "test": 0}
    )

    standardization_method: (
        Literal["z-score"] | Literal["min-max"] | Literal["none"]
    ) = "z-score"

    seq_len: int = 7
    train_size: float = 0.7
    test_size: float = 0.15
    val_size: float = 0.15
    test_needed: bool = True
    # for FireDataset class
    downsample: int = 1
    lat_col: str = "lat_bin"
    long_col: str = "long_bin"
    date_col: str = "date"
    target_col: str = "fire_count"
    cache_dir: str = CACHE_DIR
    global_max: Optional[int] = None
    global_min: Optional[int] = None
    global_mean: Optional[float] = None
    global_std: Optional[float] = None
