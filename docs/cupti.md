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
 GPU activities:   92.15%  521.27us  1  vecAdd(float const *, float const *, float*, int)
                    7.09%  40.108us  2  [CUDA memcpy HtoD]
                    0.75%  4.2680us  1  [CUDA memcpy DtoH]
```

Two constraints are nvprof's own rather than this engine's. It refuses compute
capability 8.0 and above, so profiling uses a Turing profile; and it links a
CUPTI of its own toolkit's major, so a CUDA 12 nvprof needs the CUDA 12 shim.
`tests/e2e/run_nvprof.sh` checks both and skips when they cannot be met.

"No API activities were profiled" is expected and correct: that line refers to
the Callback API, which is deliberately not dispatched -- see below.

## What is implemented

The Activity API, which is what produces a timeline:

- `cuptiActivityRegisterCallbacks`, `cuptiActivityEnable` / `Disable` (and the
  per-context spellings), `cuptiActivityFlushAll` / `Flush` / `FlushPeriod`,
  `cuptiActivityGetNextRecord`, `cuptiActivityGetNumDroppedRecords`
- `cuptiGetVersion`, `cuptiGetResultString`, `cuptiGetLastError`,
  `cuptiGetTimestamp`, `cuptiFinalize`
- `cuptiActivityPushExternalCorrelationId` / `Pop`

Records produced:

| kind | carries |
| --- | --- |
| `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` | name, grid and block, dynamic shared bytes, device, stream, correlation id, start and end |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | direction, bytes, device, stream, correlation id, start and end |

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
activity records above carry the same information for the kinds that exist.

**No metrics or events.** The Profiling and Event APIs report hardware
performance counters. The exact counters this engine keeps -- instruction mix,
sectors and coalescing, shared-memory bank conflicts, tensor-core issues -- are
a different set from the ones a device exposes, and mapping them onto
NVIDIA's metric names would claim an equivalence that does not hold. They are
reported through `VGPU_COUNTERS`; see the README.

**Nothing derived from time.** No cycles, no stall reasons, no achieved
occupancy, no cache hit rates. There is no timing model and no cache model to
derive them from.
