#!/usr/bin/env python3
"""torch.profiler (Kineto) over CUPTI, for a small CUDA workload.

Runs the same program under whatever libcupti the process loads and prints the
sets of names the profiler collected -- GPU kernels, CUDA runtime calls, copies
and fills -- so the output of a run on a real GPU (NVIDIA's CUPTI) and a run on
a simulated one (libvgpucupti) can be compared line by line. Times are not
printed: on the simulator they are time spent simulating.

usage: profiler_check.py [--json out.json]
"""
import ctypes
import json
import os
import sys
import tempfile

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def workload():
    torch.manual_seed(0)
    x = torch.randn(256, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda")
    z = None
    for _ in range(2):
        with record_function("step"):
            y = (x @ w).relu()
            z = y.sum()
            torch.cuda.synchronize()
    host = torch.ones(1024, pin_memory=True)
    dev = host.to("cuda", non_blocking=True)
    back = dev.cpu()
    torch.cuda.synchronize()
    return float(z), float(back.sum())


def names_by_category(path):
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    out = {}
    for e in events:
        cat = e.get("cat")
        if cat in ("kernel", "cuda_runtime", "cuda_driver", "gpu_memcpy", "gpu_memset", "gpu_user_annotation",
                   "cuda_sync", "user_annotation", "cuda_profiler_range"):
            out.setdefault(cat, set()).add(e.get("name", ""))
    return out


def main():
    # PyTorch reaches the driver API through the runtime's initialisation; the
    # simulator's driver shim wants cuInit first, as a driver does of a program
    # that never called the runtime (idempotent, so harmless against NVIDIA's).
    ctypes.CDLL("libcuda.so.1").cuInit(0)
    keep = None
    if len(sys.argv) > 2 and sys.argv[1] == "--json":
        keep = sys.argv[2]
    workload()  # warm-up: libraries load and kernels are first used outside the profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = workload()
    path = keep or os.path.join(tempfile.mkdtemp(), "trace.json")
    prof.export_chrome_trace(path)
    names = names_by_category(path)
    print("result:", result)
    for cat in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver", "cuda_sync", "user_annotation"):
        print(f"[{cat}]")
        for n in sorted(names.get(cat, ())):
            print("  " + n)


if __name__ == "__main__":
    main()
