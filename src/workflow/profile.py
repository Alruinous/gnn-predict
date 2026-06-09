import gc
import os
import time
import tracemalloc
from functools import wraps

import psutil
import torch

from workflow.types import WorkflowExperimentResult


def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        gc.collect()
        assert torch.cuda.is_available(), "CUDA is not available"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cpu_start = process.cpu_times()
        memory_base = process.memory_info().rss
        start_time = time.perf_counter()

        # TODO 启动一个进程或线程运行方法，并再使用一个后台线程监控内存使用情况
        func_res = func(*args, **kwargs)

        end_time = time.perf_counter()
        cpu_end = process.cpu_times()
        duration = end_time - start_time
        cpu_time = cpu_end.user - cpu_start.user + cpu_end.system - cpu_start.system
        cpu_cores = float(cpu_time) / duration if duration > 0 else 0

        profile_res = WorkflowExperimentResult()
        # TODO 汇总和返回结果
        # 暂时考虑汇总 start_time, end_time, duration, cpu_time, cpu_cores
        # 内存还没确定如何实现
        return func_res, profile_res
