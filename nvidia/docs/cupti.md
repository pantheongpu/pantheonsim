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
and finds the simulated GPU. PyTorch's CUDA 13 wheels carry PTX for
compute_120 only, so its kernels run on `nvidia/rtx5090`
([pytorch.md](pytorch.md)); on older profiles they do not. The activity
controls Kineto calls per API function, its timestamp callback and device-side
event timestamps are not supported, and say so.

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
- `cuptiGetCallbackName`, for the runtime and driver functions whose calls are
  recorded (`cudaMalloc_v3020`, `cuMemAlloc_v2`)

Records produced:

| kind | carries |
| --- | --- |
| `CONCURRENT_KERNEL` / `KERNEL` | name (the mangled entry name, as NVIDIA's reports it), grid and block, dynamic and static shared bytes, registers and local bytes per thread, device, context, stream, correlation id, start and end |
| `MEMCPY` | direction, source and destination memory kind (pageable, pinned, device, managed), bytes, the async flag, device, stream, correlation id, start and end |
| `MEMSET` | value, bytes, memory kind, the async flag, stream, correlation id |
| `RUNTIME` | the runtime call (`cbid`, for every function the toolkit's `cupti_runtime_cbid.h` names), process and thread, return value, correlation id |
| `DRIVER` | the driver call a program made itself (`cbid` from the toolkit's `cupti_driver_cbid.h`), process and thread, return value, correlation id; see below for which calls |
| `SYNCHRONIZATION` | event, stream-wait-event, stream and context waits, with stream and event ids |
| `DEVICE` | every device once, when the kind is first flushed: name, compute capability, multiprocessors, memory and limits |
| `CONTEXT`, `STREAM` | a context or stream coming into being, while the kind is enabled |
| `MARKER`, `MARKER_DATA`, `NAME` | NVTX instants and ranges (start and end paired by id), their colour, category and payload, and named threads: see NVTX below |

A kernel or copy has the correlation id of the runtime or driver call that
issued it, which is how a profiler connects the GPU timeline to the host call
that launched it. A call made through another public call (`cudaMemcpyAsync`
goes through `cudaMemcpy` here, `cuStreamSynchronize` through
`cuCtxSynchronize`) is one call, as it is on NVIDIA's runtime and driver, which
do not route their API through itself.

The record structures are the newest the toolkit being built against defines
(`CUpti_ActivityKernel11` under CUDA 13.2, `Kernel10` under 13.0, `Kernel9`
before; `Memcpy6` from API version 26), because a consumer built with that
toolkit reads those offsets.

Enabling a kind that is not produced succeeds and yields nothing. Refusing
would stop a profiler that asks for everything and uses what arrives, which is
most of them; returning nothing for a kind with no data behind it is the honest
answer.

## The Callback API

`cuptiSubscribe` takes one subscriber at a time (a second is refused, as on
NVIDIA's). Enabled domains and callbacks are delivered as the call happens:

- **Runtime API**: ENTER and EXIT around each call, with `functionName`, the
  correlation id, `symbolName` on a kernel launch, the return value on exit
  and the toolkit's own parameter structure in `functionParams`. Delivered for
  the ~50 calls whose structures are filled completely: allocation and free,
  the copy and fill families, launches, streams, events, device queries,
  synchronization, graph launch and stream capture. A call whose parameters
  this cannot describe is not delivered, rather than delivered with a structure
  of zeros that a consumer would read as what the program passed. The
  activity `RUNTIME` records cover every call.
- **NVTX**: one callback per call, before it, with the toolkit's parameter structures.
- **Driver API**: the same, for the calls a program makes itself. Delivered
  for the ~55 whose structures are filled completely, under NVIDIA's own
  spelling of each (`functionName` is `cuMemAlloc_v2`, as `cuptiGetCallbackName`
  gives it):
  `cuInit`, `cuDeviceGet`, `cuDeviceGetCount`, `cuDeviceGetName`,
  `cuDeviceGetAttribute`, `cuDeviceTotalMem_v2`, `cuDeviceComputeCapability`;
  `cuCtxCreate_v2` / `_v3` / `_v4`, `cuCtxDestroy_v2`, `cuCtxSetCurrent`,
  `cuCtxGetCurrent`, `cuCtxSynchronize`; `cuMemAlloc_v2`, `cuMemFree_v2`,
  `cuMemAllocHost_v2`, `cuMemFreeHost`, `cuMemHostAlloc`; the host-to-device,
  device-to-host and device-to-device copies in `_v2` and `Async_v2` spellings;
  `cuMemsetD8` / `D16` / `D32` (`_v2`) and their `Async` forms;
  `cuModuleLoad`, `LoadData`, `LoadDataEx`, `LoadFatBinary`, `GetFunction`,
  `Unload`; `cuLaunchKernel` (with `symbolName`); `cuStreamCreate`,
  `CreateWithPriority`, `Destroy_v2`, `Synchronize`, `WaitEvent`, `Query`;
  `cuEventCreate`, `Record`, `Synchronize`, `Query`, `Destroy_v2`,
  `ElapsedTime` (and `_v2` where the toolkit has it). The callback-id table is
  read from the toolkit's `cupti_driver_cbid.h` at configure time, as the
  runtime's is. The pre-CUDA 3.2 spellings (`cuMemAlloc`, whose parameters are
  32-bit) are not reported. Other driver calls are in neither the callbacks nor
  the activity records; the work they issue (a kernel launched with
  `cuLaunchCooperativeKernel`, a copy made with `cuMemcpy`) still is, with a
  correlation id of its own.
- **Resource**: context created and destroyed (by `cuCtxCreate*` and
  `cuCtxDestroy`, and by a runtime program's first call on a device), stream
  created and destroyed, driver initialisation finished.
- **Synchronize**: stream and context synchronized (a stream query that finds
  the stream idle counts, as on the card).

The first `cuInit` of a process is not reported -- no callback and no record
-- because the profiler is attached by that very call; a later one is. A
subscriber is told driver initialisation finished before the next call.

`nvidia/tests/e2e/run_cupti_case.sh trace` runs one program that traces itself
through both APIs and compares what it prints, with timestamps and id values
removed, with the trace NVIDIA's CUPTI printed for the same program on an RTX
3060 (`nvidia/tests/data/cupti_trace.expected`). `--card` runs the program
against NVIDIA's libraries on a GPU and compares with the same file, which is
how the expected output is known to be what hardware prints.
`run_cupti_trace_driver.sh` does the same for a program that uses only the
driver API (`cupti_trace_driver.cu`: a PTX module in a string, device and host
memory, copies, fills, two kernels, streams, events and every wait), against
`cupti_trace_driver.expected`.

Known differences, all of the lazy-loading kind: a real driver raises
module-loaded callbacks when it first loads a kernel's code; this loads whole
modules at once and does not. Register counts come from this project's
analysis, not from the compiler, and are not compared.

## NVTX

NVTX is header-only and calls nothing until a tool is injected: it opens the
library named by `NVTX_INJECTION64_PATH` and calls its `InitializeInjectionNvtx2`
with a table to fill in. `libcupti` exports that entry point, so a profiler that
sets the variable to this library (as it does to NVIDIA's) receives markers,
ranges (push/pop and start/end), domains, registered strings and thread names as
`MARKER`, `MARKER_DATA` and `NAME` records and as `CUPTI_CB_DOMAIN_NVTX`
callbacks. `nvidia/tests/e2e/run_cupti_case.sh nvtx` compares them with what NVIDIA's
CUPTI printed for the same calls. The wide-character spellings (`nvtxMarkW`,
`nvtxRangePushW`, ...) are not delivered, because NVIDIA's CUPTI does not
deliver them either (measured).

## External correlation

`cuptiActivityPushExternalCorrelationId` / `Pop` keep one stack per kind per
thread. While any is non-empty, each runtime call is reported with an
`EXTERNAL_CORRELATION` record per kind (innermost tag first by kind, ahead of
the call's own record) carrying the tag and the call's correlation id: how
PyTorch's profiler ties a kernel to the operator that launched it. Popping an
empty stack is `CUPTI_ERROR_QUEUE_EMPTY`, as on NVIDIA's (measured).
`run_cupti_case.sh extcorr` compares all of it with the card.

Two more, found by tracing a driver program on the card. The real driver makes
a context with seven streams of its own, and reports a `STREAM` record for each
under the `cuCtxCreate` call that made them; this makes none. And the real
runtime is built on the real driver, so a runtime program's trace on NVIDIA's
CUPTI also lists the driver calls its runtime made (`cuMemAlloc_v2` under
`cudaMalloc`); here a call made through another public call is not a call of its
own, so a runtime program has runtime records and a driver program driver
records, and neither has the other's.

## What is not implemented, and why

**The Callback API covers the runtime, driver, resource, synchronize and NVTX
domains.** Module and graph resources deliver nothing.
Runtime and driver calls outside the sets above are not delivered as callbacks;
the runtime's are all in the activity records, the driver's are in them only
for the set above. Retaining a primary context (`cuDevicePrimaryCtxRetain`)
raises no context-created callback and makes no context record; a context made
with `cuCtxCreate` does.

**No metrics or events.** The Profiling and Event APIs report hardware
performance counters. The exact counters this engine keeps -- instruction mix,
sectors and coalescing, shared-memory bank conflicts, tensor-core issues -- are
a different set from the ones a device exposes, and mapping them onto
NVIDIA's metric names would claim an equivalence that does not hold. They are
reported through `VGPU_COUNTERS`; see the README.

**Nothing derived from time.** No cycles, no stall reasons, no achieved
occupancy, no cache hit rates. There is no timing model and no cache model to
derive them from.
