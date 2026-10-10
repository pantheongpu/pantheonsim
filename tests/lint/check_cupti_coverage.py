#!/usr/bin/env python3
"""Every exported runtime and driver function is a call a profiler can see.

CUPTI's callback API and its RUNTIME / DRIVER activity records are made from
the calls the shims report (vgpu/profiling.hpp): the runtime's `traced_call` and
the driver's `traced` report one, with pointers to its arguments, and a function
that is exported without either is a call NVIDIA's CUPTI would have shown a
profiler and this one silently does not. That happened to the pool-sharing calls,
external memory and semaphores, the CUDA 12 _v2 spellings of the graph calls and
all but 55 of the driver's functions.

So this reads every `VGPU_EXPORT cuda*` / `VGPU_EXPORT cu*` function in
nvidia/src/runtime_api.cpp and driver_api.cpp and requires that its body reports
the call or that it is listed below with the reason it does not.

usage: check_cupti_coverage.py
"""
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parents[2] / "nvidia" / "src"

# Exported but not a call a profiler is told of, each with why.
RUNTIME_EXEMPT = {
    # They report themselves with an ApiCall of their own, because they return a
    # structure or a string rather than a status code.
    "cudaCreateChannelDesc": "returns a structure; reported by hand with its return value",
    "cudaGetErrorString": "returns a string; reported by hand with its return value",
    "cudaGetErrorName": "returns a string; reported by hand with its return value",
}
DRIVER_EXEMPT = {
    "vgpu_driver_bind_primary_v1": "the runtime shim's private hook, not a CUDA function",
    # event_record() reports the call itself (traced) on the path where the driver records the event; on a
    # capturing stream the record is the runtime's, which reports cudaEventRecordWithFlags.
    "cuEventRecord": "reported by event_record()",
    "cuEventRecordWithFlags": "reported by event_record()",
}


def body_at(text, start):
    depth, i = 1, start
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start:i]


def exports(text, ret, prefix):
    for m in re.finditer(r"^VGPU_EXPORT\s+" + ret + r"\s+(" + prefix + r"\w+)\s*\(", text, re.M):
        i, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        j = i
        while text[j] in " \n\t":
            j += 1
        if text[j] != "{":
            continue   # a declaration
        yield m.group(1), body_at(text, j + 1)


def check(path, ret, prefix, report, exempt):
    text = (SRC / path).read_text()
    # The functions the header sweeps added live in .inc files the file includes (runtime_sweep.inc, driver_sweep.inc,
    # and runtime_132.inc / driver_132.inc for the ones only the CUDA 13.2 headers declare).
    for suffix in ("_sweep.inc", "_132.inc"):
        inc = SRC / path.replace("_api.cpp", suffix)
        if inc.exists():
            text += "\n" + inc.read_text()
    bad, n = [], 0
    for name, body in exports(text, ret, prefix):
        n += 1
        if name in exempt or report.search(body):
            continue
        bad.append(name)
    return n, bad


def main():
    failed = False
    for path, ret, prefix, report, exempt, label in (
        ("runtime_api.cpp", r"(?:cudaError_t|cudaChannelFormatDesc|const char\s*\*)", "cuda", re.compile(r"\btraced_call\s*\("),
         RUNTIME_EXEMPT, "runtime"),
        ("driver_api.cpp", "CUresult", "cu", re.compile(r"\btraced\s*\("), DRIVER_EXEMPT, "driver"),
    ):
        n, bad = check(path, ret, prefix, report, exempt)
        for name in bad:
            print(f"FAIL: {name} is exported by the {label} shim without being reported to a profiler "
                  f"(wrap it in {'traced_call' if label == 'runtime' else 'traced'}, or list it in "
                  f"tests/lint/check_cupti_coverage.py with the reason)")
            failed = True
        print(f"cupti coverage ({label}): {n} exports, {n - len(bad)} reported or exempt")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
