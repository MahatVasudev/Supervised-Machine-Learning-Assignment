import datetime
import time


def timer(func):

    def exec_func(*args, **kwargs):

        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time() - start_time

        print(f"Program Took {end_time} seconds to execute")

        return result

    return exec_func
