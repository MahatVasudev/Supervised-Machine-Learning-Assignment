import os
import threading
import time
from functools import wraps
from typing import Any

import psutil

from src.utils.plotting import plot_lifetime


def function_profiler(
    interval: float = 0.1,
    save_result: bool = False,
    result_name: str = "",
    show_result: bool = True,
) -> Any:
    """
    function profiler is a function that will plot the resources (currently only cpu and memory, will implement gpu in future)
    --parameters--

    interval: float = 0.1 default,
        represents in how many seconds should the function the check the usage of resources

    save_result: bool = False default,
        whether to save the plot created will be saved under $PERFORMANCE_LOGS_DIR$/*

    result_name: str = "" default, whether you want to give it a custom name,
        by default it will look like "{time in nanoseconds}.jpg"
        if something is written in `result_name` it will be {time in nanoseconds}_{result_name}.jpg,
        only valid when save_result is True

    show_result: bool = True default,
        this will show the result on screen when the function finishes executing

    ---

    ##### edge_case: save_result OR show_result has to be True, which means they both can be true but both CANNOT be False

    """

    # NOTE: This decorator should be mostly used for testing

    assert (
        save_result or show_result
    ), "You have to have atleaset save_result or show_result enabled"

    curr_time = time.time_ns()
    filename = f"{str(curr_time)}"
    if len(result_name) > 0:
        filename = filename + f"_{result_name}"

    def decorater(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())

            timestamps = []
            cpu_usage = []
            memory_usage = []

            start_time = time.perf_counter()
            stop_event = threading.Event()

            def sampler():
                while not stop_event.is_set():
                    now = time.perf_counter() - start_time
                    timestamps.append(now)

                    cpu_usage.append(process.cpu_percent(interval=None))
                    memory_usage.append(process.memory_info().rss / (1024 * 1024))

                    time.sleep(interval)

            thread = threading.Thread(target=sampler, daemon=True)
            thread.start()

            try:
                result = func(*args, **kwargs)
            finally:
                stop_event.set()
                thread.join()

            plot_lifetime(
                timestamps=timestamps,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                save_path=filename if save_result else None,
                show=show_result,
            )

            return result

        return wrapper

    return decorater
