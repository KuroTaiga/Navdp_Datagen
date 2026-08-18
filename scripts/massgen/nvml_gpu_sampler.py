#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class NvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample one GPU via NVML and write JSONL.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval-sec", type=float, default=0.1)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    return parser.parse_args()


def _check(code: int, fn_name: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{fn_name} failed with NVML code {code}")


def _load_nvml() -> ctypes.CDLL:
    lib_path = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
    nvml = ctypes.CDLL(lib_path)
    nvml.nvmlInit_v2.restype = ctypes.c_int
    nvml.nvmlShutdown.restype = ctypes.c_int
    nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    nvml.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(NvmlUtilization)]
    nvml.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
    nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(NvmlMemory)]
    nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
    nvml.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    nvml.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
    nvml.nvmlDeviceGetTemperature.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    nvml.nvmlDeviceGetTemperature.restype = ctypes.c_int
    return nvml


def main() -> int:
    args = _parse_args()
    interval_sec = max(0.01, float(args.interval_sec))
    stop = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    nvml = _load_nvml()
    _check(nvml.nvmlInit_v2(), "nvmlInit_v2")
    handle = ctypes.c_void_p()
    try:
        _check(
            nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(int(args.gpu_index)), ctypes.byref(handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        next_sample = started
        sample_index = 0
        with args.output.open("w", encoding="utf-8") as out:
            while not stop:
                now = time.perf_counter()
                if float(args.duration_sec) > 0.0 and now - started >= float(args.duration_sec):
                    break
                if now < next_sample:
                    time.sleep(min(next_sample - now, interval_sec))
                    continue

                util = NvmlUtilization()
                memory = NvmlMemory()
                power_mw = ctypes.c_uint()
                temp_c = ctypes.c_uint()
                _check(nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)), "nvmlDeviceGetUtilizationRates")
                _check(nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)), "nvmlDeviceGetMemoryInfo")
                power_code = nvml.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power_mw))
                temp_code = nvml.nvmlDeviceGetTemperature(handle, ctypes.c_uint(0), ctypes.byref(temp_c))

                payload = {
                    "sample_index": sample_index,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": time.perf_counter() - started,
                    "gpu_index": int(args.gpu_index),
                    "gpu_util_pct": int(util.gpu),
                    "mem_util_pct": int(util.memory),
                    "memory_used_mib": int(memory.used) / 1024.0 / 1024.0,
                    "memory_free_mib": int(memory.free) / 1024.0 / 1024.0,
                    "memory_total_mib": int(memory.total) / 1024.0 / 1024.0,
                    "power_w": (int(power_mw.value) / 1000.0) if int(power_code) == 0 else None,
                    "temperature_c": int(temp_c.value) if int(temp_code) == 0 else None,
                }
                out.write(json.dumps(payload, sort_keys=True) + "\n")
                out.flush()
                sample_index += 1
                next_sample += interval_sec
                if next_sample < time.perf_counter() - interval_sec:
                    next_sample = time.perf_counter()
    finally:
        nvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
