# Design: asynchronous kernel launch (NOT implemented)

Status: a plan, written 2026-11-18 from a read of the tree at `88152597`. Nothing here was built or measured. It covers
pantheonsim issues #487 (host functions), #523 (kernels run inside the launch call) and #634 (work queued behind a host function),
and the two programs they hang: the CUDA sample `UnifiedMemoryPerf` and any program whose kernel spins on a flag the host sets
after the launch.

## 1. What happens today

The docs say "kernels run to completion inside the launch call" ([README](../README.md), [status.md](status.md#what-does-not-work),
[gap-register.md](gap-register.md), `ci/external/cuda-samples-known.txt`). That is the default, but the tree has already moved part of
the way, so the claim needs the qualifications below. The design note [perf-design/streams.md](perf-design/streams.md) holds the
history.

**NVIDIA, the default path.**

* Every runtime and driver call takes one recursive mutex, `shared_api_mutex()` (`src/runtime/shared_runtime.cpp`; `guard_impl` in
  `nvidia/src/runtime_api.cpp`; the driver side in `nvidia/src/driver_api.cpp`). `launch_kernel_impl` (`runtime_api.cpp`, ~l.1925)
  prepares the launch and calls `Device::launch` -> `run_kernel` -> `sass::launch` / `exec::launch` (`src/runtime/runtime.cpp`)
  inside that lock. It returns when the last block has finished. Inside the kernel, blocks run on a host thread pool;
  `VGPU_THREADS=1` runs them one after another in a fixed order (ARCHITECTURE.md, scheduler notes).
* The driver launch, `launch_kernel_common` (`driver_api.cpp` ~l.2883), is the same: synchronous, under the lock.
* Events are stamped by the host clock when recorded; `cudaStreamSynchronize` has nothing to wait for.

**NVIDIA, the partial asynchronous path (exists, narrow).**

* `async_launch_eligible` (`runtime_api.cpp` ~l.1898) lets `cudaLaunchKernel` on a stream made with `cudaStreamNonBlocking`
  (`VGPU_ASYNC_LAUNCH=all`: any `cudaStreamCreate` stream; `=0` off) queue the kernel on the stream's *lane* ("Stream lanes", ~l.7056,
  one worker thread per non-empty lane). The launch returns once the worker holds the API lock (`async_enqueue_kernel`).
  Host functions, event records and event waits are lane items too (#487), and copies or kernels queued behind a host function are
  too (#634).
* It stays synchronous for: legacy and per-thread default streams, capture, cooperative, cluster and graph launches,
  `cudaLaunchKernelEx`, any loaded CUDA library shim (cuBLAS, cuDNN, ...: their math runs with no lock), managed memory (which is
  why `UnifiedMemoryPerf`, a managed-memory sample, cannot benefit), a profiler or `VGPU_TRACE`, `VGPU_RACE`, `VGPU_MAX_STEPS`,
  `VGPU_STRICT`, and a launch from inside another runtime call.
* **The worker holds the API lock for the whole kernel.** So even where the launch returns early, a kernel that waits for the host
  hangs when the host's release goes through any runtime call (a `cudaMemcpy` takes the lock). A plain store to pinned memory does
  work. The reproducer in gap-register.md (`spin<<<1,1,0,s>>>(f); *f = 1; cudaStreamSynchronize(s)`) uses `cudaStreamCreate`, so it
  needs `VGPU_ASYNC_LAUNCH=all` to take the async path at all, and then only the pinned-store form passes. I did not run it; this is
  from reading `streams.md` and the code comments. **To confirm before any work:** run the reproducer with and without
  `VGPU_ASYNC_LAUNCH=all` on the CI host.

**AMD.** `hipLaunchKernel` (`amd/src/hip_api.cpp` ~l.2883) runs the grid inside the call under the HIP API lock; I found no lane
worker equivalent for kernels. The same options apply, with `gcn_exec`'s wave scheduler in place of the SASS one. Not read in depth.

## 2. What a card does that this must reproduce

1. A launch returns before the kernel finishes; `cudaStreamQuery` and `cudaEventQuery` can say not-ready.
2. A kernel may wait on host-visible memory (pinned, managed, mapped) that the host writes later, and on a flag another stream's
   kernel writes.
3. Two streams' kernels may overlap; programs may rely on it (a producer on stream A, a consumer spinning on stream B) and on
   the opposite, `cudaStreamWaitEvent` ordering.
4. `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaFree`, `cudaMemcpy` (pageable) wait as documented.

## 3. Options

### A. Worker thread per stream, kernel runs without the API lock (the direction `streams.md` already points at)

Make the lane worker run `Device::launch` *without* the global mutex; every host-side call that touches machine state either takes
a narrower lock or waits for the streams it conflicts with.

* Needed: an audit of everything a running kernel and a host call may both touch: allocation table (`BigReaderLock` in
  `include/vgpu/brlock.hpp` covers one writer vs many readers only), module and symbol tables, `State::kernels`, texture and surface
  tables, managed-memory state, the memo cache, profiling and CUPTI correlation, the library shims' direct memory access.
* Library shims (cuBLAS etc.) read and write device memory with no lock and have no common entry: either keep forcing synchronous
  launches while one is loaded (today's rule; keeps `UnifiedMemoryPerf`-like and library programs out), or add a drain at every
  library entry (a dozen files, hundreds of exports).
* Pros: reuses the lane, event and host-function code that exists and is tested (`AsyncSnapshot`, #634 tests). Smallest step from
  here. Cons: the audit is the work; a missed shared structure is a rare data race, the worst kind of bug in a project whose selling
  point is exact results.

### B. Cooperative scheduling on the calling thread

The launch returns after running the grid *until it would block* on something only the host can change, then parks the remaining
blocks and resumes them at the next API call, event query or synchronise.

* Needed: a way to know a warp is spinning on host-written memory: the engine has a min-PC scheduler and step budgets
  (`a warp ran 1073741824 instructions`) but no notion of "waiting for the host". A heuristic (the same load of the same address
  with no progress for N rounds) would make results depend on N; a precise one needs the memory model to flag host-visible
  addresses (pinned, managed, mapped) and treat a repeated load of one as a yield point.
* Pros: no new threads, no lock changes, deterministic by construction (resume points are API calls). Cons: a host thread that
  spins on a flag the kernel sets (the other direction) with no API call in the loop never lets the grid resume: it hangs where a card
  would not. Covers a subset of cases (it was designed for host-writes-flag-after-launch), and the heuristic is brittle.

### C. Fibres / coroutines for blocks

Run blocks as user-space fibres so a blocked block can be suspended mid-kernel and the host thread resumed.

* Needed: the SASS and PTX engines keep per-thread state in C++ frames and the JIT'd regions (`docs/perf-design/round11-jit.md`)
  are native code; suspending mid-region means stackful fibres (`ucontext` or hand-written switch), with sanitizer support
  (ASan/TSan need fibre annotations) and a per-block stack. That is a rewrite of the scheduler's control flow for no gain over A
  that I can see, except removing the second thread. Not recommended.

**Recommendation:** A, in the stages of section 6, keeping today's synchronous path as the default and as the fallback for every
case the audit does not clear. B is worth a spike only if A's audit shows the library shims cannot be cleared.

## 4. What must stay deterministic and bit-exact

* The result of one kernel is a function of its inputs, not of the host thread's timing: block execution must not change. Only the
  *relative* timing of host code and kernel changes, and only for programs that race, which have no fixed answer on a card either.
* `VGPU_THREADS=1` must still give one reproducible answer for a single stream. With two overlapping kernels it cannot be fixed
  by accident: define a deterministic overlap schedule (for example: at most one kernel runs at a time, chosen in launch order,
  and a kernel that blocks on host memory yields to the next lane) and state it. Real concurrent execution of two grids is out of
  scope: it makes results run-dependent for programs that race on buffers, which today never happens.
* Event timestamps and `%globaltimer` keep their instruction-count meaning; events recorded behind a kernel are stamped when it
  ends (already so on the lane path).
* Sticky versus non-sticky errors: today a step-limit failure is kept for the launching thread (`g_deferred_errors`); the rule
  must be kept per launch, not per worker.
* The committed expected outputs that are sensitive to order: CUPTI records (`run_cupti_case.sh --card` expected files), the
  trace tests, `VGPU_RACE` and `VGPU_STRICT` output, the memo cache keys. The async path must stay off for these (as it is) until
  each has a test proving the same bytes.
* The speed work's "same bits" rule applies: a stage ships only with the A/B bit comparison of the existing workload suite.

## 5. Test plan

No new test may rely on timing sleeps; use flags and events.

1. Reproducers as tests (CI-wired, no card): (a) the `spin` program with a plain pinned store and with a `cudaMemcpy` release
   (the second fails today even on the async path); (b) kernel waits for a flag set by a host function; (c) two streams, producer
   and consumer kernels; (d) `UnifiedMemoryPerf` moved from `cuda-samples-known.txt` to passing.
2. Differential: every existing e2e test and the external suites (`ci/external/*.txt`) run with `VGPU_ASYNC_LAUNCH=all` and the new
   mode, outputs compared byte for byte with the synchronous run. `VGPU_THREADS=1` and default thread count both.
3. Sanitizers: TSan on the whole set is the main check for a missed lock (`BURST_RUNS_ON` jobs, ~24 min today per `docs/ci.md`);
   ASan for lifetime of the pending-kernel structures. A stress test: N host threads launching and querying on N streams.
4. Error paths: eager and sticky errors, `cudaDeviceReset` and `cudaFree` with a kernel pending, process exit with a kernel pending,
   capture and graph launches (`tests/lint/check_capture_coverage.py` stays green).
5. Card comparison: the real-GPU check (`docs/ci-card-check.md`) is the way to confirm that a program that spins on the host behaves
   as on hardware; until it exists, only a run on a rented card can.
6. Profiling: committed CUPTI and trace expected files unchanged.

## 6. Stages and effort

Estimates are for one engineer who knows the code; they are guesses, not measurements.

| Stage | Content | Estimate |
| --- | --- | --- |
| 0 | Confirm the claim above with the reproducer; correct the docs where they say "always synchronous" | 0.5 day |
| 1 | Run the kernel on the lane worker without the API lock for one narrow case (non-blocking stream, no library, no managed memory) behind `VGPU_ASYNC_LAUNCH=nolock`; audit list of shared state, one lock per structure | 2 to 3 weeks |
| 2 | Tests of section 5 (1, 2, 3, 4) and the TSan stress | 1 to 2 weeks, partly overlapping 1 |
| 3 | Make the library shims drain (or take the narrow locks): lifts the "library loaded" exclusion | 2 to 4 weeks |
| 4 | Managed memory with a running kernel (fault and migration model) so `UnifiedMemoryPerf` itself passes | 2 to 4 weeks, the least certain |
| 5 | Profilers, trace, race detector, capture, default streams | 2 to 3 weeks |
| 6 | AMD equivalent (HIP lanes, `gcn_exec`) | 3 to 5 weeks |

Total for NVIDIA through stage 5: about 3 to 4 months; stage 1 and 2 alone (the spin-on-pinned-flag case) are the cheap, valuable
part. The risk is in stages 3 and 4. Decision needed from the maintainer: whether the `UnifiedMemoryPerf` sample (stage 4) justifies
its share, or whether to stop at stage 2 and document the managed-memory limit.
