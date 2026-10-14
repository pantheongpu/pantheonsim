# Concurrent kernels from different streams: design note (r7-sched), NOT IMPLEMENTED

Written 2026-10-12 against `443e79c0`. The brief asked for an implementation
only if every observable effect could be shown unchanged. It cannot be, so this
is the analysis and a sketch of what a real implementation would have to do.

## What the code does today

There is nothing to overlap. Every runtime and driver call runs to completion on
the calling thread under one recursive mutex:

* `guard_impl` takes `State::mu` for the whole call (`nvidia/src/runtime_api.cpp`
  ~885, 274); that mutex is the process-wide `shared_api_mutex()` shared with
  the driver shim (`src/runtime/shared_runtime.cpp:82-128`, `driver_api.cpp:188`).
* `launch_kernel_impl` (`runtime_api.cpp` ~1444) builds the arguments and calls
  `Device::launch` -> `run_kernel` -> `sass::launch` / `exec::launch`
  (`src/runtime/runtime.cpp` ~1040-1090) inside that lock, and returns when the
  last block has finished.
* The runtime's own comments say so: "Every stream runs on the same synchronous
  engine" (~3232), "Work is synchronous here, so the stream orders nothing"
  (~894), events "complete the moment [they are] recorded" and time the host
  clock between them (~6065), `cudaStreamSynchronize` has "nothing to wait for"
  (~1813).

So "launch a kernel on stream B while stream A's kernel is running" is not a
question of dispatching in parallel: stream A's kernel has finished before the
call that would start B's returns. Two host threads launching on two streams are
serialised by the API lock, and `tests/unit/test_sched.cpp` /
`nvidia/tests/e2e/sched_pool.cu` check that this keeps working with the pool.

## What it would take

A kernel running concurrently with a later API call needs, at least:

1. **Asynchronous launch.** `cudaLaunchKernel` would enqueue and return; a
   per-stream queue and a thread to drain it. The API lock could no longer be
   held for the duration of a kernel, so every call that reads simulator state
   (allocation table, module/symbol tables, `State::kernels`, texture tables)
   would have to be audited for use while a kernel runs. The allocation table's
   `BigReaderLock` (`include/vgpu/brlock.hpp`) exists for exactly one allocator
   writer against many readers; the rest of `State` has no such protection.
2. **Ordering and events.** `cudaEventRecord`, `cudaStreamWaitEvent`,
   `cudaStreamSynchronize`, `cudaDeviceSynchronize`, default-stream (legacy and
   per-thread) blocking rules, `cudaLaunchHostFunc`, stream callbacks, and
   `cudaMemcpyAsync` / `cudaMemsetAsync` / `cudaMallocAsync` ordering would
   become real dependencies. Today an event's timestamp is the host clock at
   record time (~6065) and the elapsed time between two is a sum of host time;
   a program's timing output would change meaning.
3. **Graph capture and graphs.** `vgpu_record_launch_if_capturing` (~1510) and
   the capture-coverage lint (`tests/lint/check_capture_coverage.py`) assume a
   stream is a record the call consults synchronously; graph launch runs nodes
   inline in a topological order (~7933: "any such order is a correct execution
   of the graph").
4. **Unified memory.** Managed-memory migration and the fault model assume no
   kernel overlaps a host access.
5. **Profiling records.** CUPTI activity records (kernel start/end, correlation
   ids, API callbacks, `t_api_correlation`, the order records are emitted in,
   `src/core/profiling.cpp`, `nvidia/src/cupti_api.cpp`) are committed with exact
   expected output files (`run_cupti_case.sh --card`); a kernel's start/end times
   and the interleaving of its record with the API calls issued while it ran are
   part of them.
6. **Determinism.** Two kernels overlapping on the same buffers is a data race
   the programmer is allowed to have on hardware (kernels on different streams
   are unordered), and today it never happens. Real overlap would make the
   outcome run-dependent, which is incompatible with `VGPU_THREADS=1` giving one
   reproducible answer unless a deterministic overlap schedule is also defined
   (a fixed interleaving of two grids' blocks), which is new semantics to
   specify and test, not an optimisation.
7. **Resources.** A pool shared by two grids needs a budget (today a grid may
   take every host CPU); two grids on the 4-block pantheon launch shape would
   fill the 20 CPUs, but the same code on 64-block grids would only compete.

None of these is a change that can be shown to leave an observable effect
unchanged; they define the behaviour. Items 2 and 5 alone change what a program
and a profiler are told.

## Where the potential is, and what is cheaper

The pantheon stress launches measured in `host-parallelism.md` are 4 blocks per
launch on a 20-CPU host. Concurrent streams would help them only if the
workload issues several launches on different streams without waiting between
them. Which stream the traced launches used was not recorded (the trace line
does not show it), and the rest of the suite was not traced: NOT MEASURED.

The cheaper lever for that shape is inside a launch: more host threads per block
(warps of a block), analysed in `host-parallelism.md` H4 and rejected for the
same determinism reasons.

## Recommendation

Leave streams synchronous. If it is ever wanted, start by writing the schedule
(what order two grids' blocks interleave in, what an event timestamp means)
and the tests that pin it, before touching the executors; the worker pool in
`include/vgpu/exec/worker_pool.hpp` already supports several callers at once
(`several_host_threads_can_call_at_once`, `callers_waiting_on_each_other_get_threads_of_their_own`),
so it is not the obstacle.
