from typing import Callable
import time
import multiprocessing
import statistics
import logging
import os

import psutil

def collect_stats(stop_event, interval, stats_queue):
    start_time = time.time()
    exec_times = []
    cpu_usages = []
    memory_usages = []
    while not stop_event.is_set():
        current_time = time.time() - start_time
        cpu_usage = psutil.cpu_percent(interval=interval)
        memory_usage = psutil.virtual_memory().used / 1024 / 1024  # in MB
        exec_times.append(current_time)
        cpu_usages.append(cpu_usage)
        memory_usages.append(memory_usage)
        time.sleep(interval)

    stats_queue.put((exec_times, cpu_usages, memory_usages))

def trace_stats(interval: float = 0.1, file_path: str = "logs/log.txt") -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            stop_event = multiprocessing.Event()
            stats_queue = multiprocessing.Queue()

            try:
                process = multiprocessing.Process(target=collect_stats, args=(stop_event, interval, stats_queue))
                logging.info("Function '%s' started.", func.__name__)
                process.start()
                result = func(*args, **kwargs)
                return result
            
            finally:
                stop_event.set()
                process.join()

                # Get the collected statistics from the queue
                exec_times, cpu_usages, memory_usages = stats_queue.get()

                exec_time = exec_times[-1]
                med_cpu_usage = statistics.median(cpu_usages)
                med_memory_usage = statistics.median(memory_usages)

                if not os.path.exists(file_path): os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                with open(file_path, 'a') as f:
                    f.write(f"Function '{func.__name__}' statistics:\n")
                    f.write(f"\tExec time: {exec_time:.2f}\n")
                    f.write(f"\tMedian CPU usage: {med_cpu_usage:.2f}\n")
                    f.write(f"\tMedian Memory usage in MB: {med_memory_usage:.2f}\n")

                logging.info("Function '%s' finished.", func.__name__)

        return wrapper
    return decorator