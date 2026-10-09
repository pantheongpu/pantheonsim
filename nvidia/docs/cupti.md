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
([pytorch.md](pytorch.md)); on older profiles they do not.

`torch.profiler` (Kineto) works. `nvidia/tests/pytorch/profiler_check.py` runs
a small workload (random matrices, a product, a reduction, pinned and pageable
copies) under the profiler and prints the names of the kernels, copies, fills
and runtime calls it collected, so a run against NVIDIA's libcupti and a run
against this one can be compared line by line. It was run both ways on an RTX
3060 (torch 2.10 with CUDA 13): the kernel, copy and runtime-call name sets are
the same except for what is listed below, and the result of the workload
differs in its last float digit (NVIDIA's product sums in another order).
It needs PyTorch, so it is not part of CI.

What Kineto needed, found by tracing the calls it makes:

* It does not call `cuptiActivityFlushAll` unless CUPTI has handed it a
  buffer, which it counts in its buffer-request callback. NVIDIA's asks for the
  buffer as the first record of a batch is made (a kernel launch returning, the
  first call returning while the runtime kind is on), not when the batch is
  delivered, so this does the same: a profiler that counts buffers and flushes
  only if there are any got "GPU trace is empty!" until it did. Measured by
  `run_cupti_case.sh buffers`.
* It enables the runtime and driver kinds and then switches single functions
  off with `cuptiActivityEnableRuntimeApi` / `EnableDriverApi`, and registers a
  timestamp callback and device event timestamps (CUDA 13): all supported
  now; see "Switches".

Differences from the card on this workload:

* **No cuBLAS kernel.** cuBLAS here is a host routine (cublas.md), so a matrix
  product has no kernel record and no `ampere_sgemm_...` name. The copies the
  routine makes to reach its operands are not reported either (they would be
  copies of the library, not of the program). The same holds for every library
  shim: a library call is a host routine, and its kernels are not in the
  trace.
* Times are time spent simulating, as everywhere in this file.

`VGPU_TRACE=1` also logs, to stderr, each activity enable and disable, the
per-function switches, each flush and what it delivered: how to see what a
profiler asked of this library.

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
| `CONTEXT`, `STREAM` | a context or stream coming into being, while the kind is enabled; a context brings the eight streams the driver makes in it (ids, flags and priorities are an RTX 3060's) |
| `MARKER`, `MARKER_DATA`, `NAME` | NVTX instants and ranges (start and end paired by id), their colour, category and payload, and named threads: see NVTX below |
| `MEMORY2` (and `MEMORY`) | an allocation or a release: address, bytes, memory kind, the allocating context and device, the pool when there is one; `MEMORY` (the older kind) lists each allocation once when it is released or at the flush |
| `MEMORY_POOL` | a pool created, resized (the release threshold moved, memory trimmed) or destroyed; pools grow in 32 MiB steps |
| `GRAPH_TRACE` | one record per launch of a graph, with its graph id, while the kind is on (the nodes' own records are then not made, as on the card) |
| `CUDA_EVENT` | an event recorded: event id, stream, context, a device timestamp where asked for (`cuptiActivityEnableCudaEventDeviceTimestamps`, CUDA 13) |
| `FUNCTION` | a kernel's code loaded: function id, module id, name |
| `OVERHEAD` | the first launch of a kernel (function loading), and the request for a buffer; see below |
| `MEMCPY2` | a copy between two devices that can reach each other: see below |

A kernel or copy has the correlation id of the runtime or driver call that
issued it, which is how a profiler connects the GPU timeline to the host call
that launched it. A call made through another public call (`cudaMemcpyAsync`
goes through `cudaMemcpy` here, `cuStreamSynchronize` through
`cuCtxSynchronize`) is one call, as it is on NVIDIA's runtime and driver, which
do not route their API through itself.

The record structures are the newest the toolkit being built against defines
(`CUpti_ActivityKernel11` under CUDA 13.2, `Kernel10` under 13.0, `Kernel9`
before; `Memcpy6` from API version 26; likewise `Memory4`, `MemoryPool3`,
`Overhead3`, `GraphTrace2`, `CudaEvent2` where the toolkit has them), because
a consumer built with that toolkit reads those offsets.

Enabling a kind that is not produced succeeds and yields nothing, unless the
card says otherwise: an RTX 3060 refuses the counter and metric kinds of the
legacy profiler (`CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED`), a few others as
`NOT_COMPATIBLE` or `INVALID_KIND`, and the Unified Memory counter kind as
`NOT_READY`; this answers each as the card does (`enable_result`, measured).
Refusing everything would stop a profiler that asks for all and uses what
arrives, which is most of them.

## The Callback API

`cuptiSubscribe` takes one subscriber at a time (a second is refused, as on
NVIDIA's). Enabled domains and callbacks are delivered as the call happens:

- **Runtime API**: ENTER and EXIT around each call, with `functionName`, the
  correlation id, `symbolName` on a kernel launch, the return value on exit
  and the toolkit's own parameter structure in `functionParams`. Delivered
  for every exported runtime function that has a callback id and a parameter
  structure this can fill (about 370 with arguments and 150 without, in the
  CUDA 13.0 header). The parameter structures are made from the toolkit's own `generated_cuda_runtime_api_meta.h` at configure time
  (`scripts/gen_cupti_runtime_conv.py`): for each callback id with a
  `<function>_v<N>_params` structure whose fields are plain values, the call's
  i-th argument is copied into the i-th field, by the field's size; a few
  functions that need more (the launches, the copies with a kind, the graph
  and capture calls) are written by hand. So the structures follow the toolkit
  the shim is built against, and the generator was run against the 12.0, 12.8
  and 13.0 headers. A function whose structure has an array, a bit-field or a
  function pointer for a field is not delivered as a callback, rather than
  delivered with a structure of zeros that a consumer would read as what the
  program passed; the activity `RUNTIME` records cover every call. A function
  that takes no arguments is delivered with no structure, as on the card.
  `run_cupti_case.sh params` compares the structures of about
  100 calls, field by field, with the card's.
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
- **Resource**, every callback id the toolkit names for these:
  * context created, and destroyed (`CONTEXT_DESTROY_STARTING`, for a driver
    context and for a runtime program at `cudaDeviceReset`, which first tells
    the modules and then the streams the context holds, in that order);
  * stream created, destroyed, and `STREAM_ATTRIBUTE_CHANGED` (from
    `cudaStreamSetAttribute` / `CopyAttributes`; the attribute data is the
    toolkit's `CUpti_StreamAttrData`, from API version 22);
  * driver initialisation finished (with null data, as the card passes it);
  * `MODULE_LOADED`, `MODULE_UNLOAD_STARTING` and `MODULE_PROFILED`, for
    modules loaded through the driver API and for the runtime's own (the
    fat binaries of the program, announced when first used), with the module
    id, the cubin pointer and size. The code is given when it is a cubin this
    engine holds (a module loaded as a cubin, or a fatbin with SASS for the
    device); otherwise, a module of PTX the engine compiles for itself, the
    pointer is null and the size 0 -- never a made-up buffer. Module ids are
    numbered per device, in load order, as the card's are
    (`run_cupti_case.sh resource`);
  * graphs: `GRAPH_CREATED`, `DESTROY_STARTING`, `CLONED`, `GRAPHNODE_*` and
    `GRAPHEXEC_*`, with the ids `cuptiGetGraphId`, `cuptiGetGraphExecId` and
    `cuptiGetGraphNodeId` return, numbered as the card numbers them, and
    including the card's lowering of an instantiated graph (empty nodes are
    dropped and the executable's nodes are new nodes). `run_cupti_case.sh graph`.
- **Synchronize**: stream and context synchronized (a stream query that finds
  the stream idle counts, as on the card).

The first `cuInit` of a process is not reported -- no callback and no record
-- because the profiler is attached by that very call; a later one is. A
subscriber is told driver initialisation finished before the next call.

### Switches

All measured on an RTX 3060 and compared by cases of `run_cupti_case.sh`:

* **Callback ids** (`cuptiEnableCallback`, `cuptiEnableDomain`,
  `cuptiGetCallbackState`): a callback is enabled by its id or by its whole
  domain, each id has its own state, an id the domain does not have is
  `CUPTI_ERROR_INVALID_PARAMETER`, and a second subscriber is refused (`misc`).
* **`cuptiActivityEnableRuntimeApi` / `EnableDriverApi`** (CUDA 12.6 and
  later toolkits): the id 0 is accepted and does nothing; an unnamed id is a
  bad parameter. With the `RUNTIME` (`DRIVER`) kind on, every function is
  recorded except those switched off; with the kind off, only the functions
  switched on are, but only while the kind has not been turned off since API
  calls began to be traced (the card's rule is odd: after the kind has been
  turned off, switching a function on records nothing until the kind is turned
  on again; turning it on forgets every choice). `filter` and `filter_driver`.
* **Disabling a kind** delivers what was recorded before: a record is judged
  by the settings in force when it was made, not when the buffer is flushed.
* **`cuptiActivityRegisterTimestampCallback`** sets the program's clock
  (one at a time; null takes it away): the records' times are put on it,
  device-side work by one linear interpolation between two readings taken
  when the clock was registered and when the batch is flushed (the card
  interpolates too). `cuptiGetTimestamp` then reads it.
* **`cuptiActivityEnableCudaEventDeviceTimestamps`** (CUDA 13 only): the
  `CUDA_EVENT` record's device timestamp is filled while on, zero otherwise.
* **Buffers**: a buffer is requested when the first record of a batch is made,
  see "PyTorch's profiler"; `FlushAll` fills and completes it, and a flush
  with nothing to deliver makes no buffer-completed call.
* `cuptiGetResultString` has the card's string for every result code.

### Activity records the card makes that need a trigger

* **`OVERHEAD`**: `LAZY_FUNCTION_LOADING` for the first launch of a kernel
  (with the launch's correlation id), and `ACTIVITY_BUFFER_REQUEST` for a
  request made by the profiler. The card's records for its own setup
  (instrumentation, resource) and the cost of a flush that also delivers other
  kinds are not made: NVIDIA's are unreliable there (a flush can complete
  before a kernel's record is written, and the kernel is then lost), so the
  case enables the kind alone (`overhead`).
* **`MEMORY2` / `MEMORY_POOL`**: from `cudaMalloc` and `cudaFree`, the async
  pool calls, `cudaMemPool*`, managed and host allocations (`memory`).
* **`GRAPH_TRACE`**: from `cudaGraphLaunch` (`graph`).
* **`CUDA_EVENT`**: from `cudaEventRecord` and `cuEventRecord` (`misc`).
* **`STREAM`** records for the driver's own eight streams: made with every
  context, as the card does (`resource`).
* **`MEMCPY2`** (the peer-to-peer kind): see below.

### Not checked against a card

* **`MEMCPY2`.** A copy between two devices that can reach each other (peer
  access enabled with `cudaDeviceEnablePeerAccess`) is one `MEMCPY2` record
  with the source and destination device and context, produced by the
  runtime's peer copies (`cudaMemcpyPeer`, `cudaMemcpyPeerAsync`,
  `cudaMemcpy3DPeer`, and copies between devices' pointers). That is **derived
  from documentation (the `CUpti_ActivityMemcpyPtoP4` header), not checked
  against a card**: the two RTX 3060s this was developed on have no path
  between them (`cudaDeviceCanAccessPeer` is 0 there), so NVIDIA's CUPTI never
  made one. What was measured is the case without a path, which the card does
  as two copies through the host (a device-to-host and a host-to-device,
  `copyKind` 2 and 1, under one correlation id): `misc`. The case
  `nvidia/tests/e2e/cupti_peer.cu` exercises the peer copies and prints both;
  it has no expected file because none could be made, and
  `run_cupti_case.sh peer --card --update` makes it on a machine with two GPUs
  that can reach each other (it skips where there is no path). It is not part
  of CTest. Note that the shim's two simulated RTX 3060s *do* report a peer
  path (profiles do not model the platform), where a real pair on this machine
  does not.

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

The cases (each prints deterministic text; ids become ranks, "c0" being the
first correlation id seen, so only which records share an id is compared):

| case | what it compares with the card |
| --- | --- |
| `trace`, `nvtx`, `extcorr` | callbacks and records of both APIs; NVTX; external correlation |
| `params` | the parameter structure of about 100 runtime calls, field by field (197 lines) |
| `memory` | allocation, release and pool records, and the older memory kind |
| `graph` | graph ids, graph-trace records, and the graph resource callbacks |
| `resource` | module, stream-attribute and context callbacks; context, stream and function records |
| `misc` | which kinds can be enabled and what they answer, per-function records, callback switches, CUDA event records, copies between devices, a program clock (needs two GPUs on the card) |
| `overhead` | function loading and buffer-request overhead |
| `filter`, `filter_driver` | `EnableRuntimeApi` / `EnableDriverApi` rules |
| `buffers` | when the program is asked for a buffer |
| `um` | the Unified Memory counter kind, below |

Known differences: Register counts come from this project's analysis, not
from the compiler, and are not compared. A real driver raises module-loaded
callbacks when it first loads a kernel's code lazily; a runtime program's
modules are announced here when first used and not per kernel.

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

A context made through the driver API is reported with the driver's eight
streams of its own as `STREAM` records, under the `cuCtxCreate` call that made
them, as the card does. And the real runtime is built on the real driver, so a
runtime program's trace on NVIDIA's CUPTI also lists the driver calls its
runtime made (`cuMemAlloc_v2` under `cudaMalloc`); here a call made through
another public call is not a call of its own, so a runtime program has runtime
records and a driver program driver records, and neither has the other's.

## What is not implemented, and why

**Unified Memory counters.** `cuptiActivityConfigureUnifiedMemoryCounter`
accepts any list and `cuptiActivityEnable` / `Disable` of
`UNIFIED_MEMORY_COUNTER` answer `CUPTI_ERROR_NOT_READY`, which is what NVIDIA's
answers on the machine this was measured on (an RTX 3060 under WSL2, where
managed memory is not paged on demand: `cudaDevAttrConcurrentManagedAccess` is
0). On a Linux GPU the card produces page-fault and migration counters; this
engine has no model of demand paging to count from, so the kind is refused
rather than enabled and silent (`run_cupti_case.sh um`). The case continues on
a machine where the kind can be enabled, to record what a managed allocation
does there (host and device touches, prefetches in both directions, a copy,
two devices in turn); that part was written but never run, and is not checked
against anything.

**Not made, and not checked on a card:**

* `OVERHEAD` kinds other than function loading and buffer request
  (instrumentation, resource, driver, compiler, ...).
* Driver-API pools and graph calls (`cuMemPool*`, `cuGraph*`) raise no resource
  callbacks and make no `MEMORY_POOL` / graph records; the runtime's do.
* A runtime program's trace does not list the driver calls underneath it
  (above).
* Kind numbers above the toolkit's count: this refuses them as not
  compatible; the card answers some and crashes on others, so there is no
  answer to copy.
* The answers to enabling a kind are those of a compute capability 8.6 device;
  on one older than 7.5 NVIDIA's CUPTI accepts the legacy profiler's kinds,
  which this does not model.
* A lowered graph whose nodes are not kernel, memset, memcpy or empty nodes.
* Kernel records for the library shims (cuBLAS, cuDNN, ...): those routines
  run on the host (see PyTorch's profiler above).

**The Callback API covers the runtime, driver, resource, synchronize and NVTX
domains.** Driver calls outside the list above are not delivered as callbacks,
and are in the activity records only for that list. Retaining a primary context
(`cuDevicePrimaryCtxRetain`) raises no context-created callback and makes no
context record; a context made with `cuCtxCreate` does.

**No metrics or events.** The Profiling and Event APIs report hardware
performance counters. The exact counters this engine keeps -- instruction mix,
sectors and coalescing, shared-memory bank conflicts, tensor-core issues -- are
a different set from the ones a device exposes, and mapping them onto
NVIDIA's metric names would claim an equivalence that does not hold. They are
reported through `VGPU_COUNTERS`; see the README.

**Nothing derived from time.** No cycles, no stall reasons, no achieved
occupancy, no cache hit rates. There is no timing model and no cache model to
derive them from.
