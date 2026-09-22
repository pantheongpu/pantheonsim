# CUPTI

`libvgpucupti` presents itself as `libcupti.so.<major>`, the interface every
profiler reads a CUDA program through: Nsight Systems, `nvprof`, PyTorch's
profiler, DCGM. Without it, the exact counters and the launch timeline this
engine already keeps are reachable only through `VGPU_COUNTERS`, which no
existing tool knows how to read.

## What the timestamps mean

**Wall-clock nanoseconds spent simulating.** Not a prediction of how long a
device would take, and not convertible into one.

There is no timing model here and there is not going to be one by accident.
A timeline drawn from these records is truthful about *what ran, in what order,
with what launch geometry, moving how many bytes*. It says nothing about how
fast hardware would be. Enabling activity recording prints that in the program's
own output, once, so nobody reads a flame graph from a simulator and believes it
is a measurement of a GPU.

This is the same line the telemetry surface draws: report what is known, and
refuse to invent the rest. See `ARCHITECTURE.md`.

## Loading the profiler

A profiler does not ask the driver to profile. It sets `CUDA_INJECTION64_PATH`
and relies on CUDA initialization to open that library and call its
`InitializeInjection` entry point, which is where the tool installs its hooks.

Until that was implemented, nvprof loaded, ran the program correctly, and
reported "No profile data collected" -- which reads as a broken profiler rather
than a driver that never invited it in. It is loaded once, at initialization,
from both the driver and runtime entry points: a program that only uses the
runtime API never reaches `cuInit`, and a profiler attached to one would
otherwise never start.

## nvprof

Works, on a matching toolkit major:

```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.34%  3.1957ms         1  3.1957ms  3.1957ms  3.1957ms  vecAdd(float const *, float const *, float*, int)
                    1.44%  46.911us         2  23.455us  22.869us  24.042us  [CUDA memcpy HtoD]
                    0.22%  7.0950us         1  7.0950us  7.0950us  7.0950us  [CUDA memcpy DtoH]
      API calls:   98.00%  3.1975ms         1  3.1975ms  3.1975ms  3.1975ms  cudaLaunchKernel
                    1.70%  55.423us         3  18.474us  7.6570us  24.692us  cudaMemcpy
                    0.16%  5.0910us         3  1.6970us     492ns  2.9510us  cudaFree
                    0.14%  4.5850us         3  1.5280us     300ns  3.9690us  cudaMalloc
```

`--print-gpu-trace`, `--print-api-trace` and `--export-profile` work too.

Two constraints are nvprof's own rather than this engine's. It refuses compute
capability 8.0 and above, so profiling uses a Turing profile; and it links a
CUPTI of its own toolkit's major, so a CUDA 12 nvprof needs the CUDA 12 shim.
`nvidia/tests/e2e/run_nvprof.sh` checks both and skips when they cannot be met;
CI's build job installs Ubuntu's `nvidia-profiler` (nvprof 12.0) beside its CUDA
12.0 toolkit, so there it runs. `--metrics` and `--events` print nvprof's own
warning that it does not profile compute capability 7.5 and above, as on a real
T4.

## PyTorch's profiler

libtorch links CUPTI by soname, so with the shims preloaded it loads this one
and finds the simulated GPU. Its kernels do not run: prebuilt PyTorch wheels
carry SASS only, and the simulator executes PTX. Profiling PyTorch here needs a
PyTorch built with PTX for the profile's architecture.

## Nsight Systems

Does not collect CUDA data, and cannot yet.

`nsys` runs the program correctly, and its injection library loads and
initializes through the same path nvprof uses -- but it does not use this
library at all. It ships its own CUPTI and loads it by absolute path from its
own install directory, and that copy reaches the driver through
`cuGetExportTable`, NVIDIA's undocumented internal interface. Its own error
strings say so: "Failed to get cuGetExportTable".

So an `nsys` report from a program running here contains OS runtime traces --
`pthread_create`, file I/O -- and no CUDA trace data. The blocker is not
anything in the public API; it is a set of version-specific tables of function
pointers with no specification, which `TODO.md` has long listed as deferred for
being brittle. Nothing about the CUPTI here changes that, and implementing the
Activity API cannot route around it.

The practical answer today is nvprof, which collects through the public API
this does implement.

## What is implemented

The Activity API, which is what produces a timeline:

- `cuptiActivityRegisterCallbacks`, `cuptiActivityEnable` / `Disable` (and the
  per-context spellings), `cuptiActivityFlushAll` / `Flush` / `FlushPeriod`,
  `cuptiActivityGetNextRecord`, `cuptiActivityGetNumDroppedRecords`
- `cuptiGetVersion`, `cuptiGetResultString`, `cuptiGetLastError`,
  `cuptiGetTimestamp`, `cuptiFinalize`
- `cuptiActivityPushExternalCorrelationId` / `Pop`
- `cuptiGetCallbackName`, for the runtime functions whose calls are recorded

Records produced:

| kind | carries |
| --- | --- |
| `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` | name, grid and block, dynamic shared bytes, device, stream, correlation id, start and end |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | direction, bytes, device, stream, correlation id, start and end |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | the runtime call (`cbid`), process and thread, return value, correlation id, start and end |

A kernel or copy has the correlation id of the runtime call that issued it,
which is how a profiler connects the GPU timeline to the host calls.

Enabling a kind that is not produced succeeds and yields nothing. Refusing
would stop a profiler that asks for everything and uses what arrives, which is
most of them; returning nothing for a kind with no data behind it is the honest
answer.

## What is not implemented, and why

**The Callback API delivers no callbacks.** `cuptiSubscribe`,
`cuptiEnableCallback` and `cuptiEnableDomain` accept a subscriber so a consumer
can attach, but nothing is dispatched. The points a real CUPTI intercepts are
inside the driver, and synthesising them here would mean reporting API entries
and exits that did not happen the way the consumer is told they did. The
`RUNTIME` activity records carry the runtime calls themselves.

**No metrics or events.** The Profiling and Event APIs report hardware
performance counters. The exact counters this engine keeps -- instruction mix,
sectors and coalescing, shared-memory bank conflicts, tensor-core issues -- are
a different set from the ones a device exposes, and mapping them onto
NVIDIA's metric names would claim an equivalence that does not hold. They are
reported through `VGPU_COUNTERS`; see the README.

**Nothing derived from time.** No cycles, no stall reasons, no achieved
occupancy, no cache hit rates. There is no timing model and no cache model to
derive them from.
