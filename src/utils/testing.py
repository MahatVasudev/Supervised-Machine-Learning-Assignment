import time
from .logger import Logger


def timer(func):

    def exec_func(*args, **kwargs):

        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time() - start_time

        Logger.info_line(f"Program Took {end_time} seconds to execute")

        return result

    return exec_func
