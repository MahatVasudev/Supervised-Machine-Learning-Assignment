from shutil import rmtree
from src.config import CACHE_DIR, SRC_DATASET_DIR
import os

if __name__ == "__main__":

    # CLEANING CACHE SCRIPT

    # All cache folder
    datascript_cache_folder_name = ["conv_cache", "autoencoder_cache"]

    cache_path = [
        CACHE_DIR, *[os.path.join(SRC_DATASET_DIR, kc) for kc in datascript_cache_folder_name]]
    for path in cache_path:
        rmtree(path)
