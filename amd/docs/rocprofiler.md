# rocprofiler-sdk and rocprofv3

`librocprofiler-sdk` (`amd/src/rocprofiler_sdk.cpp`, built into `build/shim` under its real soname
`librocprofiler-sdk.so.1`) is the interface AMD's profiler, `rocprofv3`, reads a HIP program through.
Inside `vgpu shell`, with ROCm's `rocprofv3` installed, the profiler runs unmodified on a simulated
GPU: [amd/README.md](../README.md#profiling) says what it traces and counts. This page is about how
that is checked against a real card, and what is known to differ.

## What the timestamps mean

**Wall-clock nanoseconds spent simulating.** There is no timing model, so a start, an end or a
duration in a trace says what the simulation took, not what a card would. A trace is truthful about
what ran, in what order, with what geometry, moving how many bytes. Nothing in the comparison below
is a time.

## The card oracle

`amd/tests/rocprof/<case>.cpp` are small hipcc programs (milliseconds on a card). For each,
`rocprofv3` was run on a real **Radeon RX 6800** (gfx1030, HIP device 0) and a real
**Radeon RX 6700 XT** (gfx1031, run as gfx1030 under `HSA_OVERRIDE_GFX_VERSION=10.3.0`), with ROCm
7.2.0 and rocprofv3 1.1.0:

```
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --memory-allocation-trace --hsa-trace \
          --marker-trace --scratch-memory-trace --stats --output-format json -d <dir> -- ./case
```

`amd/tests/rocprof_normalize.py` (Python standard library only) reduces the JSON to text that does not
depend on time, addresses or the machine, and the result is `amd/tests/data/rocprof_<case>.expected`
(`rocprof_<case>.rx6700xt.expected` for the second card). The simulator runs the same program under the
same command, inside `vgpu shell --gpu amd/rx6800` (or `amd/rx6700xt`), and its trace is normalised the
same way and compared. Both cards give the same trace for every case but the agent line (name, ISA,
compute units); all eight cases were recorded on both and the simulator matches each.

| case | what the program does |
| --- | --- |
| `vecadd` | `hipMalloc` x3, `hipMemcpy` in, a launch, `hipDeviceSynchronize`, `hipMemcpy` out, `hipFree` x3 |
| `memset_d2d` | `hipMemset`, `hipMemcpy` and `hipMemcpyAsync` device to device, `hipMemsetAsync`, `hipMemsetD32Async` |
| `copies` | 1 MiB copies to and from pinned memory (`hipHostMalloc`), on a stream and blocking |
| `streams_events` | two streams with a kernel each, ordered by `hipEventRecord` and `hipStreamWaitEvent` |
| `graph` | a stream captured into a graph (copy, kernel, copy), instantiated and launched twice |
| `async_alloc` | `hipMallocAsync`, copies and a kernel on the block, `hipFreeAsync` |
| `markers` | ROCTx marks, a nested pair of ranges and a range started and stopped by id |
| `geometry` | a 3D grid, static plus dynamic LDS (2048 + 512 bytes), and a private segment (1040 bytes) |
| `runtime_kernels` | copies within a device and small host transfers, fills, by size and alignment: the runtime's own kernels |
| `scratch` | kernels with private segments of 272 to 8208 bytes on three queues: scratch-memory records |
| `first_calls` | which calls start the runtime and make its staging buffers; allocation variants; streams made with flags |
| `graph_sizes` | what instantiating a graph allocates, by its kernel nodes, and what destroying it frees |
| `async_sizes` | the memory pool's mapped ranges for blocks of 1 KiB to 100 MiB, and when a synchronization gives them back |

What a normalised trace holds, in the order the profiler numbered the calls:

- `[agent]`: the GPU the HIP runtime sees: names, compute units, SIMDs, wave size, LDS, ISA id.
- `[kernels]`: the program's own kernels as the code object declares them: LDS, scratch, kernarg size
  and the register counts rocprofiler-sdk reports. `[runtime kernels]`: the same for the 16 kernels of
  the runtime's own code object.
- `[hip api, thread N]`: every HIP call, runtime and compiler (`__hipRegisterFatBinary`,
  `__hipPushCallConfiguration`, ...), with its arguments; a pointer or handle prints as `<ptr>`.
- `[dispatches]`: each dispatch, the program's and the runtime's own (`__amd_rocclr_copyBuffer`,
  `__amd_rocclr_fillBufferAligned`): demangled name, grid and workgroup size, LDS and scratch bytes.
- `[copies]`: memory-copy records: direction, bytes, the kind of agent at each end.
- `[scratch]`: scratch-memory records by queue (numbered as they first appear): the allocations in
  order with their flags and sizes, then the frees.
- `[allocations]`: the allocations the program asked for (the record of the requested size under
  `hipMalloc`, `hipHostMalloc`, ...) and the frees under `hipFree` and its relatives, with the agent
  (a free names none).
- `[runtime allocations]`: every other allocation record, under the HIP call that made it.
- `[markers]`: ROCTx marks and ranges with their messages.

### Running it

```
# the simulator (needs ROCm with rocprofv3 and hipcc to build and drive the case; SKIPs without):
VGPU_BUILD_DIR=build amd/tests/e2e/run_rocprof_case.sh vecadd
VGPU_BUILD_DIR=build amd/tests/e2e/run_rocprof_case.sh vecadd --gpu rx6700xt
# the real card, against ROCm's own libraries, compared with the same files:
amd/tests/e2e/run_rocprof_case.sh vecadd --card
```

The tests are `amd_rocprof_<case>` and `amd_rocprof_<case>_rx6700xt` in ctest; they print
`SKIP: ...` and are skipped where there is no rocprofv3 (GitHub's hosted runners, and the
pantheonsim.com runner's image), or where ROCm's rocprofv3 is not the release the files were recorded with
(`ROCPROF_ANY_VERSION=1` compares anyway). `ROCPROF_KEEP=<dir>` keeps the raw JSON of a run.
`test_amd_rocprofiler_rx6800` and `test_amd_rocprofiler_rx6700xt` (`amd/tests/unit/test_rocprofiler_rdna.cpp`)
are the part of it that needs no ROCm: a tool of their own checks the agent, the kernels' register
counts, which copies are copy records, and the allocation records' ids, and they run everywhere.

### Regenerating the expected files from the card

On a machine with the card, ROCm 7.2 (`hipcc`, `rocprofv3`) and nothing else on the card:

```
cases="vecadd memset_d2d copies streams_events graph async_alloc markers geometry
       runtime_kernels scratch first_calls graph_sizes async_sizes"
for c in $cases; do
  amd/tests/e2e/run_rocprof_case.sh $c --card --update          # the RX 6800 (HIP device 0)
done
for c in $cases; do
  amd/tests/e2e/run_rocprof_case.sh $c --card --update --gpu rx6700xt   # the RX 6700 XT (HIP device 1)
done
```

`ROCPROF_CARD_DEVICE=<n>` names another HIP device. The script sets `HIP_VISIBLE_DEVICES` itself: a
machine with an unsupported iGPU (gfx1036) must not see it. Each file starts with a header that says
which card and ROCm it came from. Run each twice and `diff` the files before committing: the traces
were identical from run to run on both cards. (One update of `graph` came back with its last records
missing, and a rerun was complete: check a fresh `--card` run, without `--update`, agrees.) The comparison with `--card` needs no update, so it also
shows whether a new ROCm prints something different.

## Where the simulator and the card differed, and what was changed

All of these were found by the comparison; each is covered by an `.expected` file and by
`test_amd_rocprofiler_rdna`.

- **The code object's URI** was a pointer into a stack buffer that rocprofv3 keeps after the callback
  returns, so a trace could contain garbage bytes and `results.json` was not valid UTF-8. Strings the
  tool may keep (the URI, kernel names) are now held for the life of the process.
- **Kernel names** are the descriptor's symbol, `<kernel>.kd`, as the HSA runtime names them. rocprofv3
  demangles that to `vadd(float const*, float const*, float*, int) [clone .kd]` in the JSON.
- **Register counts**: from gfx10 on, rocprofiler-sdk reports `sgpr_count` 128 whatever the kernel uses and
  `arch_vgpr_count` rounded up to the descriptor's granule (8 in wave32, 4 in wave64): a kernel using 6
  registers reports 8, one using 19 reports 24. Measured on gfx1030 and gfx1031; the same rule is applied to
  gfx11 and gfx12, which nothing was measured on. The Instinct parts keep the code object's counts.
- **The agent** of an RDNA card is what KFD says: the RX 6800 is 60 compute units of 2 SIMDs, 32 waves each
  (HIP's `multiProcessorCount` is 30, counting workgroup processors), the RX 6700 XT 40. With
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` the runtime says gfx1030 for the RX 6700 XT but its agent is still
  `gfx1031` (target 100301).
- **Memory-copy records** are made only for what the runtime gives a DMA engine: a transfer between the host
  and a device of more than 16 KiB (16384 bytes is a kernel, 16388 a copy record, pageable or pinned,
  blocking or asynchronous). Smaller ones and every copy within a device are done by a kernel the runtime
  runs itself (`__amd_rocclr_copyBuffer`), which is a dispatch, not a copy. The Instinct parts were not
  measured and keep a record for every copy.
- **Allocation records** have an id of their own and name the HIP call that made them as their ancestor
  (`correlation_id.ancestor`), as the card's do; before they shared the call's id and had no ancestor.
  `hipHostMalloc` and `hipHostFree` now produce records, on the CPU agent.
- **The HIP compiler API** (`__hipRegisterFatBinary`, `__hipRegisterFunction`, `__hipRegisterVar`,
  `__hipRegisterManagedVar`, `__hipPushCallConfiguration`, `__hipPopCallConfiguration`,
  `__hipUnregisterFatBinary`) is recorded, with its operation names, as `--hip-trace` records it.

## Known differences

The comparison leaves these out, on both sides, with `rocprof_normalize.py --strip-known-gaps`
(`KNOWN_GAPS` in that script); a gap that is closed is removed from the list and the tests cover it from
then on. `--card` compares everything.

| what | the card | the simulator | where |
| --- | --- | --- | --- |
| **Arguments of HIP calls** | every record has its arguments: `hipMalloc(ptr, size=4096)`, `hipMemcpy(dst, src, sizeBytes, kind=HostToDevice)`, `hipLaunchKernel(function_address, numBlocks={z=1, y=1, x=4}, ...)` | `args` is empty. `rocprofiler_iterate_buffer_tracing_record_args` and `rocprofiler_iterate_callback_tracing_kind_operation_args` return NOT_AVAILABLE / NOT_IMPLEMENTED, and the hook libamdhip64 calls (`api_enter(name)`) carries the name only | `rocprofiler_sdk.cpp` near the end; `ApiCall` in `hip_api.cpp`. Closing it needs every HIP entry point to hand over its arguments and a table of their types and names |
| **ROCTx markers** | `roctxMarkA`, `roctxRangePushA`/`Pop` (`roctxThreadRangeA` records) and `roctxRangeStartA`/`Stop` (`roctxProcessRangeA`), with their messages | none: `rocprofiler_set_api_table` accepts ROCTx's table and wraps nothing | `rocprofiler_set_api_table` in `rocprofiler_sdk.cpp` |
| **HSA API trace** | 340 to 1100 HSA calls for these programs (queue, signal and memory-pool setup of ROCm's runtime) | none: the HIP runtime reaches the simulated devices directly | by design (the header of `rocprofiler_sdk.cpp`) |
| **Agent fields the profile does not have** | shader engines and arrays, SDMA engines, queues, clocks, memory banks, caches, io links, firmware | zero | `build_agents` in `rocprofiler_sdk.cpp`; not compared |
| **Correlation ids** | the runtime's internal calls take ids too (a `hipMalloc` is 3, then 95, 97, 99) | consecutive | not compared |
| **Copy records on Instinct cards** | not measured | one per copy | `on_copied` in `rocprofiler_sdk.cpp` |

The programs print their own results (`wrong 0 of 1024`), so a case that runs under the profiler also
computed the right answer.

## The runtime's own work

A real ROCm runtime does work of its own for a program, and `rocprofv3` shows it. The simulator's HIP
runtime does not do it, so `rocprofiler_sdk.cpp` ("What ROCm's runtime does for the program") adds the
records. Every rule below was measured on an RX 6800 and an RX 6700 XT, ROCm 7.2.0, and is checked
against both cards by the cases named; it applies to the gfx1030 family (RDNA 2) and nothing else
(`measured_gpu`). Nothing here has a time: the records' timestamps are the simulation's.

**Kernels** (`runtime_kernels`, and every case that copies or fills). The runtime's code object (always
loaded, the same 16 kernels on both cards: `[runtime kernels]`) runs:

- `__amd_rocclr_copyBuffer` for a copy within a device and for a host transfer of up to 16384 bytes.
  Both ends 16-byte aligned: workgroups of 512, 16 bytes to a thread, `grid = min(roundup(ceil(bytes/16),
  512), cu/2 * 512)` (at least one workgroup). Otherwise (any end not 16-byte aligned): workgroups of
  1024, 4 bytes to a thread, `min(roundup(ceil(bytes/4), 1024), cu/2 * 1024)`. `cu` is the agent's
  compute-unit count (60 and 40), so the large copies' grids differ between the cards (15360 and
  10240; 30720 and 20480 unaligned).
- `__amd_rocclr_fillBufferAligned`, in workgroups of 256, for a fill of device memory: the bytes up to
  the next 16-byte boundary (the head), the 16-byte units (the body, `min(roundup(units, 256), cu/2 * 256)`
  threads), and the remainder (the tail), each its own dispatch, in that order, when not empty. The
  head and tail are one workgroup.

**Allocations** (`first_calls`, `graph_sizes`, `async_sizes`, and the cases that stream, copy or
instantiate):

- The first HIP runtime call of the program, whichever it is (`hipGetDeviceCount` as well as `hipMalloc`),
  makes a 4 MiB block and a 56-byte block on the host.
- A 4 MiB and a 16 MiB host block (staging buffers) for each stream a program creates
  (`hipStreamCreate`, `...WithFlags`, `...WithPriority`), made by that call; and one pair for the default
  stream, made by the first work queued on it (a launch, a copy or a fill with no stream), by
  `hipMallocAsync`/`hipMallocFromPoolAsync`,, or by the first `hipGraphInstantiate`.
- `hipStreamDestroy` frees three blocks: the first with the address 0, then two. Every FREE record
  names no agent (the simulator named the GPU).
- `hipGraphInstantiate` allocates on the device `131072 + 384 * kernel nodes` bytes (a graph of copies only
  allocates nothing); `hipGraphExecDestroy` frees it.
- The memory pool (`hipMallocAsync`) maps 32 MiB ranges as the blocks it holds need them
  (`VMEM_ALLOCATE`, `ceil(live / 32 MiB)` of them in all), and the first `hipStreamSynchronize` or
  `hipDeviceSynchronize` with no block held gives all of them back (`VMEM_FREE`). `hipFreeAsync` itself
  records nothing.

**Scratch** (`scratch`, `geometry`). A queue (a stream, or the default stream) has a scratch pool of
`cu * 1024 * roundup(private, 32)` bytes (1024 lanes to a compute unit: 32 waves of 32), made with the
first kernel on it that has a private segment: one `SCRATCH_MEMORY_ALLOC` record with flags 0. A kernel
that needs more makes a bigger pool (an ALLOC) and frees the old (a FREE, flags 0, size 0). When the pool
would be larger than ROCr's `HSA_SCRATCH_SINGLE_LIMIT` (140 MiB) the launch gets one block of `128 *
array_count * roundup(private, 32)` bytes with flags 1 (`array_count` is 8 on the RX 6800, 4 on the
RX 6700 XT) and a FREE with flags 1 when it ends; the pool is freed and the next kernel makes a new
one. The comparison orders a queue's allocations and its frees apart, as the two records of one moment
cannot be told apart in a trace.

### Not modelled, and why

These are what the cards show or what the model cannot say without measuring; none is compared.

- **Instinct parts and other RDNA**: not measured, so they get none of the above (their small copies
  are still not copy records, an older rule).
- **Which call a block belongs to**, for programs whose first calls differ from the cases': a block is
  attributed to the call that makes it in the cases; the default stream's pair for `hipEventRecord` or
  `hipStreamWaitEvent` as the first use is not known.
- **Peer copies, 2D and 3D copies and fills, `hipMemcpyToSymbol`, `hipMemset2D`**: the card uses other
  kernels (`copyBufferRect`, `fillBufferAligned2D`) and splits; none was measured, so there is no
  dispatch for them (a 2D copy is still no copy record).
- **Pageable host memory** that is not 16-byte aligned: the copy goes through the runtime's staging
  buffer on the card, and whether the kernel sees the program's alignment was not measured; the
  simulator uses the program's.
- **Counters of the runtime's kernels**: the simulator's dispatch counting does not see them, and what a
  card's counter collection shows for them was not measured.
- **Graph allocation nodes, memory pools made by the program, `hipMallocManaged`, `hipMallocPitch`,
  `hipHostRegister`**: the runtime's own blocks, if any, were not measured beyond the cases (a
  `hipMallocManaged` makes no record on the card; `hipMallocPitch` makes one of `pitch * height`).
- **Exact values that are not a function of anything the program does**: the staging blocks' and
  graph blocks' addresses, the runtime's code object (its URI, size, load base), queue ids, and the
  timestamps. A queue is numbered in the order it first appears in a scratch record.
- **FREE records after a graph**: the card's trace is sometimes one `hipGraphExecDestroy` FREE short
  (`graph_sizes`, one run in four to five on either card), and the `graph` case now and then comes back
  without the three FREE records of its stream's destruction (two `--update` runs did). The first are left
  out of the comparison (`UNSTABLE` in `rocprof_normalize.py`); the simulator makes both. The expected
  files hold the complete trace, confirmed by six card runs each for `graph`; an occasional `--card`
  failure of `graph` is the card dropping them, not a change.
- **The single-use scratch block's size** (`128 * array_count * roundup(private, 32)`) is fitted to the
  two cards (8 and 4 arrays); a third card with another array count would test it.
