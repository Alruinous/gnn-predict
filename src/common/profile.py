from __future__ import annotations

import gc
import os
import time
from functools import wraps

import psutil
import torch


def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        gc.collect()
        assert torch.cuda.is_available(), "CUDA is not available"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cpu_start = process.cpu_times()
        memory_start = process.memory_info().rss
        start_time = time.perf_counter()

        func_res = func(*args, **kwargs)

        end_time = time.perf_counter()
        cpu_end = process.cpu_times()
        memory_end = process.memory_info().rss
        duration = end_time - start_time
        cpu_time = cpu_end.user - cpu_start.user + cpu_end.system - cpu_start.system
        profile_res = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "cpu_time": cpu_time,
            "cpu_cores": float(cpu_time) / duration if duration > 0 else 0,
            "memory_delta": memory_end - memory_start,
        }
        return func_res, profile_res

    return wrapper
