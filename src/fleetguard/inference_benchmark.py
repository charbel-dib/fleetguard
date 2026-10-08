"""Isolated CPU microbenchmark: loaded-model latency and whole-process peak RSS."""

import argparse
import ctypes
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from fleetguard.artifacts import load_model, positive_scores, prepare_input
from fleetguard.io import write_json


def peak_rss_bytes():
    if sys.platform.startswith("linux"):
        # Scope the current exec image, excluding the parent's fork/exec peak.
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("Linux process peak RSS unavailable.")
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def benchmark(model_dir, input_path, *, batch_sizes, repeats):
    frame = pd.read_csv(input_path)
    tracemalloc.start()
    started = time.perf_counter()
    model, metadata = load_model(model_dir)
    load_seconds = time.perf_counter() - started
    features = prepare_input(frame, metadata["feature_names"])
    if len(features) < max(batch_sizes):
        raise ValueError("Benchmark fixture is smaller than a requested batch.")
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows = []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for size in batch_sizes:
            batch = features.iloc[:size]
            # Exclude warm-up from percentiles; measure input-to-score inference only.
            tracemalloc.start()
            started = time.perf_counter()
            positive_scores(model, batch)
            first_seconds = time.perf_counter() - started
            _, warmup_peak = tracemalloc.get_traced_memory()
            python_peak = max(python_peak, warmup_peak)
            tracemalloc.stop()
            durations = []
            for _ in range(repeats):
                started = time.perf_counter()
                scores = positive_scores(model, batch)
                durations.append(time.perf_counter() - started)
                if not np.isfinite(scores).all():
                    raise RuntimeError("Nonfinite benchmark scores.")
            rows.append(
                {
                    "batch_rows": size,
                    "repeats": repeats,
                    "warmup_seconds": first_seconds,
                    "median_ms": float(np.median(durations) * 1000),
                    "p95_ms": float(np.quantile(durations, 0.95) * 1000),
                    "rows_per_second_at_median": float(size / np.median(durations)),
                }
            )
    return {
        "load_seconds": load_seconds,
        "pipeline_bytes": (model_dir / "pipeline.joblib").stat().st_size,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "python_traced_peak_bytes": python_peak,
        "memory_scope": "whole benchmark process; RSS includes imports and native allocations",
        "rss_measurement": (
            "proc VmHWM"
            if sys.platform.startswith("linux")
            else "PeakWorkingSetSize"
            if sys.platform == "win32"
            else "getrusage ru_maxrss"
        ),
        "latency_scope": "sensor DataFrame to score; excludes CSV parsing, HTTP and network",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "available_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_limit": 1,
        "threadpools": pools,
        "batches": rows,
        "python_tracing_enabled_during_latency": False,
        "python_memory_scope": "maximum traced peak of model loading or one batch warm-up",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("Budgets must be positive.")
    write_json(
        args.output,
        benchmark(
            args.model,
            args.input,
            batch_sizes=args.batch_sizes,
            repeats=args.repeats,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
