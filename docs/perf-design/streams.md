# Concurrent kernels from different streams: design note (r7-sched), NOT IMPLEMENTED

(The section "Asynchronous launch, one kernel at a time" at the end describes what was built later: a launch that returns
before its kernel has run, with at most one kernel in flight. Concurrent kernels are still not implemented.)

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

## Asynchronous launch, one kernel at a time (pantheonsim #523)

What is built: `cudaLaunchKernel` on a stream made with `cudaStreamNonBlocking` (`VGPU_ASYNC_LAUNCH=all`: any `cudaStreamCreate`
stream) validates and prepares everything in the call, under the API lock, as before (arguments, module load, shared-memory limit,
the device's configuration checks through `Device::check_launch`, injected failures, the cooperative and graph-launch rules).
It then queues the kernel as an item on the stream's lane (`runtime_api.cpp`, "Stream lanes") and the lane's worker thread runs
`Device::launch` while holding the API lock. The launching thread returns once the worker holds the lock.

Why that is safe, and why it is not more:

* Everything that reads or writes the machine takes the API lock (the runtime and the driver share one,
  `shared_runtime.cpp`), so it waits for the running kernel exactly as it used to wait inside the launch. The thing that makes this
  sound is that the worker has the lock before the launch returns: a call made after the launch on the same thread cannot get in
  front of the kernel. Calls that do not take the lock are the ones that should see the kernel pending: `cudaEventQuery`,
  `cudaEventElapsedTime`, `cudaStreamQuery`, event records (queued behind the kernel on the lane, stamped when it ends, so an event
  pair times the kernel).
* At most one kernel is in flight, and a launch (or copy) on a stream first waits for that stream's lane to drain, as it already did
  for host functions. A second launch also waits for the first kernel by way of the API lock. There is no overlap of kernels, so
  the determinism argument of item 6 above is untouched; what is lost relative to a card is only kernel-kernel and
  kernel-copy overlap, which nothing could observe here before either.
* Errors: a refused configuration is thrown by `check_launch` in the call. A sticky fault is the context's (`set_error` sets it from
  the worker, and every call that needs the context already reports it, as the launch of a faulting kernel always returned
  success). The one thing that moves is a failure that is not sticky (a kernel that hit the step limit): it is kept for the
  launching thread (`g_deferred_errors`) and reaches its `g_last_error` and `g_async_error` at its next runtime call.

What is not covered, each a reason a launch stays synchronous (`async_launch_eligible`):

* **The CUDA library shims** (cuBLAS, cuDNN, cuFFT, cuSOLVER, cuSPARSE, cuRAND, NPP, nvJPEG, ...): their math runs on the calling
  thread, reads and writes device memory through the allocation table with no API lock, and there is no common entry to hook (a
  dozen files, hundreds of exports). A pending kernel would race with the next library call. So the launch is synchronous while a
  library shim is loaded (looked for with `dl_iterate_phdr` at each eligible launch). What would have to change: a drain at the entry
  of every library call, or in `MemoryManager`'s host-side accessors (which are hot paths for the engine and would need to tell the
  kernel's own threads from the caller's).
* **Managed memory** (the fault and migration model assumes no kernel overlaps a host access), a profiler or `VGPU_TRACE` (record
  order and kernel start/end are committed in expected files), `VGPU_RACE`, `VGPU_MAX_STEPS`, `VGPU_STRICT` (their failures are
  not sticky and tests read them from the launch).
* **Cooperative, cluster, graph and `cudaLaunchKernelEx` launches, graph capture, the default streams** (the legacy stream's
  ordering against blocking streams is the lane barrier's, which stays as for host functions), and a launch from inside another
  runtime call (the thread holds the API lock the worker needs).
* **A kernel that waits for the host** (spins on a flag the host sets after the launch): the worker holds the API lock for the
  kernel, so a host thread that sets the flag through a runtime call (a `cudaMemcpy`) waits for the kernel, and the program hangs
  where a card would not; setting it with a plain store to pinned memory works. This hung before too (the launch itself did not
  return), so it is not a regression, but it is the next limit to remove: it needs the kernel to run without the API lock, which is the
  audit of item 1 above.

### The driver API's host functions (pantheonsim #487)

`cuLaunchHostFunc` uses the same lanes: libcuda reaches them by name (`vgpu_async_*_v1` in `runtime_api.cpp`, found with `dlsym` as the
capture hooks are; see "Host functions" in `driver_capture.inc`), so the two libraries share one set of lanes and one set of stream
handles, and a driver stream made with `CU_STREAM_NON_BLOCKING` is told to the runtime (`vgpu_async_stream_flags_v1`) because the
runtime does not know its flags. Every hook returns at once until a host function has been queued by either API, so programs that
never queue one are untouched. The driver has no common stream choke point, so the order is enforced where the work is issued:
`StreamOrder` (an RAII wait, in `driver_api.cpp` before `StreamContext`) in front of the kernel launches, `StreamContext` for the copies
and fills, and explicit `StreamOrder`s in the 2D, array, peer and stream-memory-operation paths and `cuStreamAddCallback`. It waits before
the call takes the API lock, once per call stack (a thread-local flag, so an async copy that ends in the synchronous one waits for its own
stream and not for the legacy stream's dependencies), and not for the replay of a captured launch, which runs under the lock. A driver event
is also recorded on its runtime shadow event once a host function exists; `cuEventQuery`, `cuEventSynchronize`, `cuEventElapsedTime`
and `cuStreamWaitEvent` read the shadow's pending state and time (`vgpu_async_event_*_v1`).

### Work queued behind a host function (pantheonsim #634)

A copy, fill or launch on a stream whose lane still holds a host function used to wait for the lane in the call (`async_order`), which
keeps the order but hangs a program whose host function waits for something the calling thread does after the call: Celeritas'
`AsyncSnapshot` test queues a host function that spins on a flag, then a device-to-host `cudaMemcpyAsync` and an event, checks that the
event is not ready, and only then sets the flag. On a card the copy call returns at once. Now such work is queued on the lane too, behind
the host function, and the call returns:

* `cudaMemcpyAsync`, `cudaMemsetAsync`, `cudaMemcpy2DAsync`, `cudaMemset2DAsync` (and `cudaMemset3DAsync`, slice by slice),
  `cudaMemcpy3DAsync` between pitched pointers, `cudaMemcpyToSymbolAsync`, `cudaMemcpyFromSymbolAsync`, `cudaMemcpyPeerAsync` and
  `cudaFreeAsync` become an `Op` item (`async_defer`): the plain call, run by the lane's worker under the API lock, in the lane's order,
  so it sees what the host function wrote and the host function does not see what it writes. Only when every byte it touches is device
  memory or page-locked host memory (`cudaMallocHost`, `cudaHostAlloc`, `cudaHostRegister`), and only when the call would not refuse it
  (its kind and ranges are checked first), so the refusals stay the call's.
* A kernel launch becomes a `Kernel` item that the launch does not wait to start, on any stream the runtime made, when it could be
  launched asynchronously at all (`async_launch_eligible`); otherwise it waits as before.
* Events, `cudaStreamWaitEvent`, `cudaStreamQuery` and the synchronizing calls need nothing new: an event recorded after the queued work
  is queued behind it, and `cudaMemcpy`, `cudaStreamSynchronize`, `cudaDeviceSynchronize` and `cudaEventSynchronize` wait for lanes to
  drain, the legacy default stream's barriers included. `cudaFree`, `cudaFreeHost`, `cudaHostUnregister` and `cudaDeviceReset` wait for
  queued work first (`async_wait_deferred`), as the card synchronises in them. A queued copy that fails reports at the issuing thread's
  next call, as an asynchronous launch's failure does.
* The 2D, 3D and symbol copies now also wait for their own stream's lane rather than the legacy default stream's when they do run in the
  call (they used to reach `cudaMemcpy` without the stream, so on a non-blocking stream they did not follow a host function on it), and
  `cudaMemcpyPeer`, `cudaMemcpyToSymbol` and `cudaMemcpyFromSymbol` follow host functions as `cudaMemcpy` does.

What still waits in the call: work on the per-thread default stream, copies to or from pageable memory (a card copies device to pageable
memory inside the call too), managed memory, arrays, while a profiler listens or a CUDA library shim is loaded (their work does not take
the API lock), calls made from inside another runtime call, and the driver API's copies and launches (its `StreamOrder`). Programs that
never queue a host function are not changed: every hook returns after one relaxed load.
