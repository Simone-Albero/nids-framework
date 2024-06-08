from typing import Callable
import time
import multiprocessing
import statistics
import logging
import os
import functools

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
    global_cpu_usages = []
    proc_cpu_usages = []
    rss_memory_usages = []
    vms_memory_usages = []

    # starting_cpu_usage = psutil.cpu_percent(interval=0.1)

    while not stop_event.is_set():
        current_time = time.time() - start_time
        global_cpu_usage = psutil.cpu_percent(interval=0.1) # - starting_cpu_usage
        proc_cpu_usage = process.cpu_percent(interval=0.1)
        memory_usage = process.memory_info()
        rss_memory_usage = memory_usage.rss / 1024 / 1024
        vms_memory_usage = memory_usage.rss / 1024 / 1024

        exec_times.append(current_time)
        global_cpu_usages.append(global_cpu_usage)
        proc_cpu_usages.append(proc_cpu_usage)
        rss_memory_usages.append(rss_memory_usage)
        vms_memory_usages.append(vms_memory_usage)

        time.sleep(interval)
    stats_queue.put((exec_times, global_cpu_usages, proc_cpu_usages, rss_memory_usages, vms_memory_usages))


def trace_stats(
    interval: float = 0.1, file_path: str = "logs/log.csv", reset_logs: bool = False
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
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

                exec_times, global_cpu_usage, proc_cpu_usages, rss_memory_usages, vms_memory_usages = stats_queue.get()
                exec_time = exec_times[-1]
                med_gloabl_cpu_usage = statistics.median(global_cpu_usage)
                med_proc_cpu_usage = statistics.median(proc_cpu_usages)
                med_rss_memory_usage = statistics.median(rss_memory_usages)
                med_vms_memory_usage = statistics.median(vms_memory_usages)

                if not os.path.exists(file_path) or reset_logs:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w") as f:
                        f.write(f"FUNCTION,EXEC_TIME,GLOBAL_CPU_USAGE,PROC_CPU_USAGE,PROC_RSS_MEMORY_USAGE,PROC_VMS_MEMORY_USAGE\n")

                with open(file_path, "a") as f:
                    f.write(
                        f"{func.__name__},{exec_time:.2f},{med_gloabl_cpu_usage:.2f},{med_proc_cpu_usage:.2f},{med_rss_memory_usage:.2f},{med_vms_memory_usage:.2f}\n"
                    )
                logging.info("Function '%s' finished.", func.__name__)

        return wrapper

    return decorator
