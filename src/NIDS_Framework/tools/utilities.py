from typing import Callable, Optional
import time
import multiprocessing
import statistics
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
    global_cpu_usages = []
    proc_cpu_usages = []
    rss_memory_usages = []
    vms_memory_usages = []

    while not stop_event.is_set():
        global_cpu_usage = psutil.cpu_percent(interval=0.1)
        proc_cpu_usage = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        rss_memory_usage = memory_info.rss / 1024 / 1024
        vms_memory_usage = memory_info.rss / 1024 / 1024

        global_cpu_usages.append(global_cpu_usage)
        proc_cpu_usages.append(proc_cpu_usage)
        rss_memory_usages.append(rss_memory_usage)
        vms_memory_usages.append(vms_memory_usage)
        time.sleep(interval)

    stats_queue.put(
        (
            global_cpu_usages,
            proc_cpu_usages,
            rss_memory_usages,
            vms_memory_usages,
        )
    )


def trace_stats(
    interval: Optional[float] = 0.1, file_path: Optional[str] = "logs/log.csv", reset_logs: Optional[bool] = False
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stop_event = multiprocessing.Event()
            stats_queue = multiprocessing.Queue()
            start_time = time.time()

            try:
                curr_pid = os.getpid()
                process = multiprocessing.Process(
                    target=collect_stats,
                    args=(curr_pid, stop_event, interval, stats_queue),
                )
                process.start()
                return func(*args, **kwargs)

            finally:
                exec_time = time.time() - start_time
                stop_event.set()
                process.join()

                (
                    global_cpu_usage,
                    proc_cpu_usages,
                    rss_memory_usages,
                    vms_memory_usages,
                ) = stats_queue.get()
                print(exec_time, global_cpu_usage, proc_cpu_usages, rss_memory_usages, vms_memory_usages)
                med_gloabl_cpu_usage = statistics.median(global_cpu_usage)
                med_proc_cpu_usage = statistics.median(proc_cpu_usages)
                med_rss_memory_usage = statistics.median(rss_memory_usages)
                med_vms_memory_usage = statistics.median(vms_memory_usages)

                if not os.path.exists(file_path) or reset_logs:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w") as f:
                        f.write(
                            f"FUNCTION,EXEC_TIME,GLOBAL_CPU_USAGE,PROC_CPU_USAGE,PROC_RSS_MEMORY_USAGE,PROC_VMS_MEMORY_USAGE\n"
                        )

                with open(file_path, "a") as f:
                    f.write(
                        f"{func.__name__},{exec_time:.2f},{med_gloabl_cpu_usage:.2f},{med_proc_cpu_usage:.2f},{med_rss_memory_usage:.2f},{med_vms_memory_usage:.2f}\n"
                    )

        return wrapper
    return decorator
