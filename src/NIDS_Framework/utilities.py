from typing import Callable
import time
import multiprocessing
import statistics
import logging
import os

import psutil


def collect_stats(
    pid: int,
    stop_event: multiprocessing.Event,
    interval: float,
    stats_queue: multiprocessing.Queue,
) -> None:
    process = psutil.Process(pid)
    start_time = time.time()
    exec_times = []
    cpu_usages = []
    memory_usages = []

    while not stop_event.is_set():
        current_time = time.time() - start_time
        cpu_usage = process.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss / 1024 / 1024

        exec_times.append(current_time)
        cpu_usages.append(cpu_usage)
        memory_usages.append(memory_usage)

        time.sleep(interval)
    stats_queue.put((exec_times, cpu_usages, memory_usages))


def trace_stats(
    interval: float = 0.1, file_path: str = "logs/log.csv", reset_logs: bool = False
) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            stop_event = multiprocessing.Event()
            stats_queue = multiprocessing.Queue()

            try:
                curr_pid = os.getpid()
                process = multiprocessing.Process(
                    target=collect_stats,
                    args=(curr_pid, stop_event, interval, stats_queue),
                )
                logging.info("Function '%s' started.", func.__name__)
                process.start()
                result = func(*args, **kwargs)
                return result

            finally:
                stop_event.set()
                process.join()

                exec_times, cpu_usages, memory_usages = stats_queue.get()
                exec_time = exec_times[-1]
                med_cpu_usage = statistics.median(cpu_usages)
                med_memory_usage = statistics.median(memory_usages)

                if not os.path.exists(file_path) or reset_logs:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w") as f:
                        f.write(f"FUNCTION,EXEC_TIME,CPU_USAGE,MEMORY_USAGE\n")

                with open(file_path, "a") as f:
                    f.write(
                        f"{func.__name__},{exec_time:.2f},{med_cpu_usage:.2f},{med_memory_usage:.2f}\n"
                    )
                logging.info("Function '%s' finished.", func.__name__)

        return wrapper

    return decorator
