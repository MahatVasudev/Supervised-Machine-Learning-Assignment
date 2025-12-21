from numpy.typing import NDArray
from .logger import Logger
import os
from src.config import DATASETS_DIR, GLOBAL_MODIS_DIR, MODIS_DIR
import string
from typing import Dict, Literal, List, Tuple
import glob


def find_final_data_bytime(years_selected: list[str | int], by_time: str) -> List[Tuple[str, str]]:
    files = []
    for year in years_selected:
        if not check_final_data_exists(year, by_time):
            Logger.warning(
                "Final Data of year {0} and by time {1} not found, Skipping....".format(year, by_time))
            continue

        files.append((year, os.path.join(GLOBAL_MODIS_DIR,
                     final_global_filename(year, by_time))))

    return files


def find_final_metadata_bytime(years_selected: List[str | int], by_time: str) -> List[Tuple[str, str]]:
    files = []
    for year in years_selected:
        if not check_metadata_exists(year, n_time=by_time):
            Logger.warning(
                "Metadata of year {0} and by time {1} not found, Skipping....".format(year, by_time))
            continue

        files.append((year, os.path.join(GLOBAL_MODIS_DIR,
                     final_metadata_filename(year, by_time))))

    return files


def find_global_year_data(years_selected: list[str | int]) -> Tuple[List[str | int], List[str]]:
    found_data: list[str] = []
    found_year: list[int | str] = []
    for year in years_selected:
        if not check_global_year_data(year):
            Logger.warning(f"modis global year {year} not found")
        else:
            found_data.append(os.path.join(
                GLOBAL_MODIS_DIR, modis_global_filename(year)))
            found_year.append(year)

    return found_year, found_data


def check_year_log(mode: Literal['global'] | Literal['final'], years: List):

    func = check_selected_exists if mode == 'global' else check_global_year_data
    suggested_dir = MODIS_DIR if mode == 'global' else GLOBAL_MODIS_DIR

    def suggested_filename(y): return modis_global_filename(
        y) if mode == 'final' else y
    for y in years:
        if not func(y):
            Logger.warning(f"{y} Year Data Not Found; will be skipped;\nIf you do have the data please keep it in {
                           suggested_dir} with folder name {suggested_filename(y)}")


def extract_country_name(country_file: str, curr_year: str | int) -> str:
    text = country_file.split("/")[-1]
    text = text.split(f'_{curr_year}_')[-1]
    text = text.removesuffix(".csv")
    text = text.split('_')
    return ' '.join(text)


def check_final_data_exists(years_selected: int | str, bytime: str) -> bool:
    return os.path.isfile(os.path.join(GLOBAL_MODIS_DIR, final_global_filename(years_selected, bytime)))


def check_selected_exists(year_selected: int | str) -> bool:
    # HACK: Bad code, very breakable, make it better later
    years = glob.glob(os.path.join(MODIS_DIR, "**"))
    year_select = os.path.join(MODIS_DIR, str(year_selected))
    return year_select in years


def check_metadata_exists(year_selected: int | str, n_time: str) -> bool:
    return os.path.isfile(os.path.join(GLOBAL_MODIS_DIR,
                                       final_metadata_filename(file_year=str(year_selected), n_time=n_time)))


def gather_country_files(year_selected: str) -> list[str]:
    return glob.glob(os.path.join(year_selected, "**.csv"))


def save_file(file, file_year, mode: Literal['global'] | Literal['final'] | Literal['final_metadata'], n_time=None):
    if mode == 'global':
        file.to_csv(os.path.join(GLOBAL_MODIS_DIR,
                    modis_global_filename(file_year)), index=False)
    elif mode == 'final':
        assert n_time is not None, "Please provide n_time when saving final data"
        file.to_csv(os.path.join(GLOBAL_MODIS_DIR, final_global_filename(
            file_year=file_year, n_time=n_time)), index=False)

    elif mode == 'final_metadata':
        file.to_csv(os.path.join(GLOBAL_MODIS_DIR, final_metadata_filename(
            file_year=file_year, n_time=n_time)), index=True)


def modis_global_filename(file_year):
    return f"modis_global_year_{file_year}.csv"


def final_global_filename(file_year, n_time):
    return f"final_{file_year}_by_{n_time}.csv"


def final_metadata_filename(file_year, n_time):
    return f"final_{file_year}_by_{n_time}_metadata.csv"


def check_global_year_data(year_selected: str | int) -> bool:
    return os.path.isfile(os.path.join(GLOBAL_MODIS_DIR, modis_global_filename(str(year_selected))))


def __test_country_name_conversion():
    # TEST: Testing the name conversion of country
    print("----------------------------\nTesting Country Name Conversion \n-------------------------")
    selected_year = 2024
    Logger.info_line(f"selected year: {selected_year}")
    loc_year = os.path.join(MODIS_DIR, str(selected_year))
    data = gather_country_files(loc_year)

    n = len(data)
    try:
        for cidx in range(n):
            extracted_name = extract_country_name(data[cidx], selected_year)
            Logger.info_line(
                f"[{cidx}/{n}]: File Location {data[cidx]} | Extracted Name: {extracted_name}")

        Logger.testing_success("Name Conversion ran Successfully")
    except Exception as e:
        Logger.testing_failure(f"Name Conversion Failed\n{e}")


def __test_years_detection():
    print("-----------------------------\nTesting Years Detection \n-----------------------------")
    # TEST: Testing whether we detect names exist or not
    files_list = [(2023, False), (2021, True), (2024, True), (2025, False)]

    for f, y in files_list:
        y_output = check_selected_exists(f)
        status = "found" if y_output == True else "not found"
        if y == y_output:
            Logger.testing_success(
                f"Year {f} {status}")
        else:
            Logger.testing_failure(
                f"Year {f} {status}")


def __testing_time_decoder():
    # TEST: Testing Time Decoder

    print("-------------------- Test Time Decoder ----------------------")

    list_queries = ["1d23h15m", "2M4h6m",
                    "2d4M6h", "2F3h", "2d3G", "02d5h", "2h5d", "22", "22d", "d22", "10d22"]
    list_result = [{"d": 1, "h": 23, "m": 15}, {"M": 2, "h": 4,
                                                "m": 6}, None, None, None, {"d": 2, "h": 5}, None, None, {"d": 22}, None, None]

    for q, r in zip(list_queries, list_result):
        r_hat = time_decoder(q)

        if type(r_hat) is not type(r):
            Logger.testing_failure(
                f"Not Executed Correctly; expected result: {r}, got: {r_hat}")

        else:
            if r_hat is None:
                Logger.testing_success(f"Successfully executed!! expected result: {
                                       r} and got {r_hat}")
            elif r_hat == r:
                Logger.testing_success(f"Successfully executed!! expected result: {
                                       r} and got {r_hat}")

            else:
                Logger.testing_failure(
                    f"Not Executed Correctly; expected result: {r}, got: {r_hat}")


def time_decoder(n_time: str) -> dict:
    # example; input "1d23h12m" -> day: 1, hour: 23, minute: 12
    # give syntax error if order is not correct 23h1d is wrong
    # give error if same metric used multiple times 1d1d is wrong
    # no other alphabetic character should be present

    correct_syntax = 'example; input "1d23h12m" -> day: 1, hour: 23, minute: 12\ngives syntax error if order is not correct 23h1d is wrong\ngive error if same metric used multiple times 1d1d is wrong\nno other alphabetic character should be present'
    datetime_keywords = ["M", "d", "h", "m"]
    def index_finder(x): return datetime_keywords.index(x)
    visited_keywords = set()
    visited_keywords_seq = []
    database = dict()
    c_gathered = ""
    for c in n_time:
        if c in datetime_keywords:
            if c in visited_keywords:
                Logger.fatal(f"Wrong Syntax... {n_time}\n{correct_syntax}")
                # return None
            if c_gathered == "":
                Logger.fatal(f"Wrong Syntax... {n_time}\n{correct_syntax}")
                # return None

            if len(visited_keywords_seq) > 0 and index_finder(c) <= visited_keywords_seq[-1]:
                Logger.fatal(f"Wrong Syntax... {n_time}\n{correct_syntax}")
                # return None
            database[c] = int(c_gathered)
            c_gathered = ""
            visited_keywords.add(c)
            visited_keywords_seq.append(index_finder(c))

        elif c in string.ascii_lowercase or c in string.ascii_uppercase or c in string.punctuation:
            Logger.fatal(f"Wrong Syntax... {n_time}\n{correct_syntax}")
            # return None

        else:
            c_gathered = c_gathered + c

    if c_gathered != "":
        Logger.fatal(f"Wrong Syntax... {n_time}\n{correct_syntax}")
        # return None

    return database
