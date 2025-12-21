# NOTE: This is a script file to generate the appropriate dataset for model training, there are some base assumption made here which are:
#   - you have extracted the .zip file of all countries of particular year in PARENT_DIR/datasets/modis/<year>
#   - all data inside the <year> folder will be saved in different countries name in .csv format
# data will be saved inside PARENT_DIR/datasets/global_modis by final_<year>_by_day.csv
# data will be saved in day as its time unit

import pandas as pd
import math
import os
from typing import Tuple
from src.utils.data_tools import find_global_year_data, save_file
from src.utils.data_tools import check_year_log,  check_selected_exists
from src.utils.data_tools import extract_country_name, time_decoder, gather_country_files
from src.utils.logger import Logger
from src.config import GLOBAL_MODIS_DIR, MODIS_DIR
from src.utils.argparser import download_script_parser


parser = download_script_parser.parse_args()
YEARS = parser.years
MODE = parser.mode
BATCH_SIZE = parser.batch_size
BIN_SIZE = parser.bin_size
TIME = parser.time

DATETIME_FORMAT = "%Y-%m-%d %H%M"
DATE_FORMAT = "%Y-%m-%d"

os.makedirs(GLOBAL_MODIS_DIR, exist_ok=True)
os.makedirs(MODIS_DIR, exist_ok=True)


def main_pipeline():
    Logger.warning_line(
        "This may take a long time, please wait patiently for the process to finish")
    if YEARS == [] or type(YEARS) is not list:
        Logger.fatal(
            f"Years not Provided or Years is not a list\n got input {YEARS}")

    # All according to Mode selected by the user

    match MODE:
        case 0:
            decoded_time = time_decoder(TIME)
            check_year_log('global', YEARS)

            workflow_yearly_data(years_selected=YEARS, batch=BATCH_SIZE)

            years, global_filnames = find_global_year_data(YEARS)
            for files, year in zip(global_filnames, years):
                Logger.info_line(f"Creating Final Data of year {year}")
                final_data = creating_final_dataset(files, decoded_time)
                save_file(file=final_data, file_year=year,
                          n_time=TIME, mode='final')
        case 1:
            check_year_log('global', YEARS)

            workflow_yearly_data(years_selected=YEARS, batch=BATCH_SIZE)
        case 2:
            decoded_time = time_decoder(TIME)
            check_year_log('final', YEARS)
            years, global_filnames = find_global_year_data(YEARS)
            for files, year in zip(global_filnames, years):
                Logger.info_line(f"Working on year {year}...")
                final_data, final_metadata = creating_final_dataset(
                    files, decoded_time)
                Logger.info_line(f"final data of {year} made")
                save_file(file=final_data, file_year=year,
                          n_time=TIME, mode='final')
                save_file(file=final_metadata, file_year=year,
                          n_time=TIME, mode='final_metadata')
                Logger.executed(f"final data of {year} is saved with metadata")
        case _:
            Logger.fatal("Selected Mode Not Found")


def creating_final_dataset(original_data_path: str, n_time: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(original_data_path)
    df = add_acq_datetime(df)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'],
                                        format=DATETIME_FORMAT)
    times_n = pd.date_range(df['acq_datetime'].min(), df['acq_datetime'].max(),
                            freq=pd.DateOffset(months=n_time.get("M", 0),
                                               days=n_time.get("d", 0),
                                               hours=n_time.get("h", 0),
                                               minutes=n_time.get("m", 0)))

    df = lat_long_binned(df, n_bin=BIN_SIZE)
    df = df.groupby(['lat_bin',
                     'long_bin',
                     'acq_datetime']).size().reset_index(name='fire_count')
    grid_keys = df[['lat_bin', 'long_bin']].drop_duplicates()
    filled_data = []
    for _, row in grid_keys.iterrows():
        cell_data = df[(df['lat_bin'] == row.lat_bin) & (
            df['long_bin'] == row.long_bin)].set_index('acq_datetime').reindex(times_n, fill_value=0)

        cell_data['lat_bin'] = row.lat_bin
        cell_data['long_bin'] = row.long_bin
        filled_data.append(cell_data)

    final_data = pd.concat(filled_data).reset_index().rename(
        columns={'index': 'date'})

    final_meta_data = final_data.describe()
    return final_data, final_meta_data


def workflow_yearly_data(years_selected: list[str | int], batch: int) -> None:
    year_skips = []
    for curr_year in years_selected:
        Logger.info_line(f"Making Global Dataset of year {curr_year}")
        if not check_selected_exists(curr_year):
            year_skips.append(1)
            Logger.error(f"{curr_year} year data not found; Skipping...")
            continue
        year_skips.append(0)
        country_data = gather_country_files(
            os.path.join(MODIS_DIR, str(curr_year)))

        cache_df: pd.DataFrame | None = None

        n = len(country_data)
        files_appended = []
        current_batch = 0
        for cidx in range(n):
            current_country_file = country_data[cidx]
            current_country = extract_country_name(
                country_file=current_country_file, curr_year=curr_year)
            df = pd.read_csv(current_country_file)
            df['country'] = current_country
            files_appended.append(df)

            if (cidx + 1) % batch == 0 or cidx == n-1:
                cache_batch_df = pd.concat(files_appended, ignore_index=True)
                current_batch += 1
                if cache_df is None:
                    cache_df = cache_batch_df
                else:
                    cache_df = pd.concat(
                        [cache_df, cache_batch_df], ignore_index=True)
                files_appended.clear()
                Logger.info_line(
                    f"Progress [{current_batch}/{math.ceil(n/batch)}]")

                del cache_batch_df

        if cache_df is not None:
            save_file(cache_df, curr_year, mode='global')
            Logger.info_line(f"Global MODIS data year {
                             curr_year} has been made")
            del cache_df

    short_summary = [f"Year {y}: {
        "completed" if s == 0 else "skipped"}" for y, s in zip(years_selected, year_skips)]
    Logger.executed("Process Finished", " | ".join(short_summary))


def add_acq_datetime(data: pd.DataFrame) -> pd.DataFrame:
    data["acq_date"] = pd.to_datetime(data["acq_date"], format=DATE_FORMAT)
    data["acq_datetime"] = pd.to_datetime(
        data["acq_date"].astype(str) + " " + data["acq_time"].astype(str).str.zfill(4), format=DATETIME_FORMAT)
    return data


def lat_long_binned(data: pd.DataFrame, n_bin: float = 0.25) -> pd.DataFrame:
    data['lat_bin'] = (data['latitude'] / n_bin) * n_bin
    data['long_bin'] = (data['longitude'] / n_bin) * n_bin

    return data


def docstring():
    # runs when --help is called

    # YEARS need to be written in a list example: [2021,2022,2023]

    # MODE defines what process you want to run;
    #   0 -> full process (global_file + final_file), 1 -> only making the global_file, 2 -> only making the final_file
    #   no other files will be accepted

    # BATCH_SIZE is used in global data making; it defines how much countries it will concat at once
    ...


if __name__ == "__main__":
    main_pipeline()
