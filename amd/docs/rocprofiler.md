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
| `markers_control` | `roctxProfilerPause` and `roctxProfilerResume` around HIP calls, a named stream, nested ranges, marks and a range from a second thread |
| `geometry` | a 3D grid, static plus dynamic LDS (2048 + 512 bytes), and a private segment (1040 bytes) |

What a normalised trace holds, in the order the profiler numbered the calls:

- `[agent]`: the GPU the HIP runtime sees: names, compute units, SIMDs, wave size, LDS, ISA id.
- `[kernels]`: the program's own kernels as the code object declares them: LDS, scratch, kernarg size
  and the register counts rocprofiler-sdk reports.
- `[hip api, thread N]`: every HIP call, runtime and compiler (`__hipRegisterFatBinary`,
  `__hipPushCallConfiguration`, ...), with its arguments; a pointer or handle prints as `<ptr>`.
- `[dispatches]`: each dispatch of the program's kernels: demangled name, grid and workgroup size, LDS
  and scratch bytes.
- `[copies]`: memory-copy records: direction, bytes, the kind of agent at each end.
- `[allocations]`: the allocations the program asked for (the record of the requested size under
  `hipMalloc`, `hipHostMalloc`, ...) and the frees under `hipFree` and its relatives.
- `[markers]`: ROCTx marks and ranges with their messages, each with the range it was made in.
- `[marker scopes]` (only when there is one): the HIP calls made inside a ROCTx range, with the range.

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
for c in vecadd memset_d2d copies streams_events graph async_alloc markers markers_control geometry; do
  amd/tests/e2e/run_rocprof_case.sh $c --card --update          # the RX 6800 (HIP device 0)
done
for c in vecadd memset_d2d copies streams_events graph async_alloc markers markers_control geometry; do
  amd/tests/e2e/run_rocprof_case.sh $c --card --update --gpu rx6700xt   # the RX 6700 XT (HIP device 1)
done
```

`ROCPROF_CARD_DEVICE=<n>` names another HIP device. The script sets `HIP_VISIBLE_DEVICES` itself: a
machine with an unsupported iGPU (gfx1036) must not see it. Each file starts with a header that says
which card and ROCm it came from. Run each twice and `diff` the files before committing: the traces
were identical from run to run on both cards. The comparison with `--card` needs no update, so it also
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

- **ROCTx** (`roctxMarkA`, `roctxRangePushA`/`Pop`, `roctxRangeStartA`/`Stop`, the profiler control and the
  names): `rocprofiler_set_api_table` wraps the entries of ROCTx's core, control and name tables, as
  rocprofiler-sdk does. A call a context subscribed to is delivered as `MARKER_CORE_API` /
  `MARKER_CONTROL_API` / `MARKER_NAME_API` callbacks (enter, then exit, arguments and return value in the
  payload) and buffer records. What rocprofv3 asks for is `MARKER_CORE_RANGE_API` callbacks: a mark is an
  enter and an exit at once, a range an enter where it begins and an exit where it ends, with the message in
  both and the same user data (rocprofv3 reads the messages from them, and writes the `marker_api` records
  itself); the buffer kind has records too, written when a range ends. The correlation id of a range is the id
  of the call that began it, and **a call made while a range is open has that range as its ancestor**
  (`correlation_id.ancestor`), on the card as here: a HIP call, a mark, a range begun inside it, the innermost
  first, a range begun with `roctxRangeStartA` until it is stopped. HIP records carry that ancestor now too,
  and the normalised trace shows it: ` in 'compute'` after a mark or range, and a `[marker scopes]` section
  for the HIP calls.

## Known differences

The comparison leaves these out, on both sides, with `rocprof_normalize.py --strip-known-gaps`
(`KNOWN_GAPS` in that script); a gap that is closed is removed from the list and the tests cover it from
then on. `--card` compares everything.

| what | the card | the simulator | where |
| --- | --- | --- | --- |
| **Arguments of HIP calls** | every record has its arguments: `hipMalloc(ptr, size=4096)`, `hipMemcpy(dst, src, sizeBytes, kind=HostToDevice)`, `hipLaunchKernel(function_address, numBlocks={z=1, y=1, x=4}, ...)` | `args` is empty. `rocprofiler_iterate_buffer_tracing_record_args` and `rocprofiler_iterate_callback_tracing_kind_operation_args` return NOT_AVAILABLE / NOT_IMPLEMENTED, and the hook libamdhip64 calls (`api_enter(name)`) carries the name only | `rocprofiler_sdk.cpp` near the end; `ApiCall` in `hip_api.cpp`. Closing it needs every HIP entry point to hand over its arguments and a table of their types and names |
| **HSA API trace** | 340 to 1100 HSA calls for these programs (queue, signal and memory-pool setup of ROCm's runtime) | none: the HIP runtime reaches the simulated devices directly | by design (the header of `rocprofiler_sdk.cpp`) |
| **The runtime's own kernels** | `__amd_rocclr_copyBuffer` for small and device-to-device copies, `__amd_rocclr_fillBufferAligned` for fills, as dispatches | none | `rocprof_normalize.py --runtime-kernels` shows them in a card's trace |
| **Scratch records** | a `SCRATCH_MEMORY_ALLOC` for a kernel with a private segment, sized as the runtime's pool (64880640 bytes in `geometry`) | none | not compared |
| **The runtime's own allocations** | 4 MiB and 16 MiB staging blocks and a 56-byte block on the host, a 131456-byte allocation per graph instantiation, a `VMEM_ALLOCATE` pool range for `hipMallocAsync`; FREE records carry no agent | only the program's allocations; FREE records name the GPU | the normaliser keeps the program's (`ALLOCATORS` in `rocprof_normalize.py`) |
| **Agent fields the profile does not have** | shader engines and arrays, SDMA engines, queues, clocks, memory banks, caches, io links, firmware | zero | `build_agents` in `rocprofiler_sdk.cpp`; not compared |
| **Correlation ids** | the runtime's internal calls take ids too (a `hipMalloc` is 3, then 95, 97, 99) | consecutive | not compared |
| **Copy records on Instinct cards** | not measured | one per copy | `on_copied` in `rocprofiler_sdk.cpp` |

The programs print their own results (`wrong 0 of 1024`), so a case that runs under the profiler also
computed the right answer.
