import os

global __dir__

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

DATASETS_DIR = os.path.join(PARENT_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "plots")
PERFORMANCE_LOGS_DIR = os.path.join(LOGS_DIR, "performance")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
SRC_DATASET_DIR = os.path.join(BASE_DIR, "datasets")
GLOBAL_MODIS_DIR = os.path.join(DATASETS_DIR, "global_modis")
MODIS_DIR = os.path.join(DATASETS_DIR, "modis")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SRC_DATASET_DIR, exist_ok=True)
os.makedirs(GLOBAL_MODIS_DIR, exist_ok=True)
os.makedirs(MODIS_DIR, exist_ok=True)


DATETIME_FORMAT = "%Y-%m-%d %H%M"
DATE_FORMAT = "%Y-%m-%d"
