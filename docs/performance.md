# Simulator performance

This page is the shared measurement base for making the simulator faster
without changing a result bit, a reported counter, or the order of an observable
effect (ARCHITECTURE.md: correctness, determinism and diagnosability outrank
speed). It holds the harness (how to measure), the baseline numbers, the
instruction mixes, and the profiles of both engines. Every number here was
measured by the commands shown, on the machine named below, at the commit named
below. Nothing is inferred from the code alone; where a sentence says what the
code does, it cites the line.

**Measured at** `origin/main` 443e79c0 ("gfx1250: operand forms ... (#356)"), a
Release build (`cmake -DCMAKE_BUILD_TYPE=Release`: `-O3 -DNDEBUG`, link-time
optimisation `-flto=auto -fno-fat-lto-objects`, GCC 13.3.0, no `-march`, so the
baseline x86-64 instruction set; the few `__attribute__((target("fma")))` /
`target("avx2")` functions in `src/exec/interpreter.cpp` are the only
CPU-specific code) with the CUDA 13.0 toolkit headers, program compiled with
nvcc 13.0.88 for sm_86 and run on the `nvidia/rtx3060` profile, one host thread
(`VGPU_THREADS=1`).

**Machine**: 12th Gen Intel(R) Core(TM) i9-12900K, 20 logical CPUs, a WSL2 VM (Linux 6.18.40.1-microsoft-standard-WSL2) that **other jobs share**:
the load average was between 5 and 15 during these measurements. The numbers are
for comparing two builds on one machine, not absolute speeds; see "Noise" for
what that costs.

## The harness

```
tools/bench/bench.sh --build build                       # time every workload, SASS and PTX
tools/bench/bench.sh --ab buildA buildB --runs 7         # alternating A B A B ... on one program
tools/bench/bench.sh --check-counters --build build      # counts (not times) against tools/bench/expected
tools/bench/bench.sh --update-expected --build build     # rewrite tools/bench/expected/cuda-<release>.json
tools/bench/bench.sh mix --build build                   # instruction-class mix per workload
tools/bench/bench.sh mix --build build --cmd ./prog args # ... of any program's launches
tools/bench/bench.sh profile --build build-g             # perf: top functions per engine and group
tools/bench/bench.sh --help
```

`tools/bench.sh` (the old vectoradd timer) now forwards here. The harness is
`tools/bench/bench.py`, Python 3 standard library only, run as `python3 -I`.

What a run is:

* The workloads are `tools/bench/kernels/bench.cu`, one self-checking program
  (`bench_kernels <workload> <quick|full>`), compiled once with the nvcc whose
  major version matches the shim's `libcudart.so.N` (`-arch=sm_86`: SASS and PTX
  in the one binary; cached under `~/.cache/vgpu-bench`). The binary is the same
  for both engines and for both builds of an A/B run: SASS is the default,
  `VGPU_SASS=0` runs the PTX interpreter.
* Each workload checks its own results against a host reference and prints
  `PASS` only when they match bit for bit (floating point is compared with
  `memcmp`; integer-valued data is used wherever a summation order could change
  bits). A run without `PASS` is a failure of the harness, so a speed-up that
  breaks a result fails instead of looking fast.
* One counting run per workload, engine and build (`VGPU_COUNTERS=1`,
  `VGPU_SASS_LOG=1`) gives the exact instruction counters (deterministic, so
  taken once) and confirms which engine ran; a SASS row that fell back to PTX is
  flagged. Then N timed runs without counters: **warp-instructions per CPU
  second** (instructions from the counting run over the workload's device-phase
  CPU time, which excludes process start-up and the host-side check), wall and CPU
  seconds of the whole process, and peak RSS (sampled from `/proc/<pid>/status`
  `VmHWM`: `wait4`'s `ru_maxrss` is useless from a Python launcher, it reports
  the parent's size). Launch and API workloads also report units per second.
* Every statistic is reported as median, **best** (the fastest run: contention
  only slows a run down, so it is the steadiest) and spread ((max-min)/median).
* `--ab A B` alternates A B A B ... per workload and engine and reports the ratio
  of medians, the ratio of bests, a Mann-Whitney p-value on the per-run values,
  and whether the two builds' counters are identical (they must be). Its verdict
  is `faster`/`slower` only when p <= 0.05 and both ratios moved more than 2% the
  same way, otherwise `same`. The compared metric is Minst/s, except calls per
  second for `launch` and `api` and the whole-process wall time for `startup`
  (whose device phase is a few milliseconds).
* `--cpus 2-5` pins every run to a CPU list; `--threads N` sets `VGPU_THREADS`
  (default 1: the strictly ordered block sequence); `--env K=V` adds environment
  (e.g. `VGPU_FASTPATH=0`); `--json FILE` writes everything, including per-run
  values and the host/toolchain/load meta block.

### Workloads

| workload | group | what it exercises |
| --- | --- | --- |
| int_alu | alu | 32-bit integer ALU chains: imad, lop3, shifts, popc, remainder |
| f32_fma | alu | f32 `fmaf` chains on normal numbers plus unfused add/mul |
| f64_fma | alu | f64 `fma` chains on normal numbers |
| f64_subnormal | alu | pantheon `fp64_virus`'s four coupled f64 chains (kernels/fp64_virus/fp64_virus.cpp), which decay into subnormals after ~43 outer iterations (measured on the host), so almost every FMA has a subnormal operand: the assist cost TODO.md "Performance" describes |
| wmma | tensor | `nvcuda::wmma` 16x16x16 f16 -> f32 |
| mma_ptx | tensor | `mma.sync.m16n8k16` f16 -> f32, fragments loaded by the PTX ISA layout |
| mem | memory | global loads/stores: scalar, 16-byte vector, uncoalesced gather |
| shared | memory | shared memory with barriers: tree reduction, 32x32 tile transpose with and without padding |
| atomics | memory | atomicAdd (one address, 256 bins, f32, 64-bit), atomicMax, an atomicCAS loop |
| texture | memory | `tex2D` point and linear (at texel centres), `tex1Dfetch` |
| local | memory | a per-thread array indexed dynamically: `ld.local`/`st.local` (LDL/STL) |
| divergent | control | data-dependent loops (collatz) and a `switch`: divergent control flow |
| warp | control | `__shfl_*`, `__ballot`, `__any`, `__all` |
| launch | api | 40000 one-warp launches on two streams: the launch path |
| api | api | cudaMalloc/Memcpy/Memset/Free, events, streams in a loop (60000 rounds of 15 calls) |
| startup | api | context + module registration + one launch (read the whole-process wall time) |

Existing programs were looked at first. `tests/kernels` holds only
`vector_add.ptx` (what the old timer ran). The `nvidia/tests/e2e/*.cu` programs
are 40-150 line functional checks that finish in milliseconds. The pantheon
workloads (tests/workloads/run_pantheon_workloads.sh, docs/pantheon-workloads.md)
are duration-driven soak programs that need a separate checkout, a Makefile build
and `vgpu shell`; their instruction count depends on how long they ran, so they
cannot back a deterministic counter check. `f64_subnormal` reproduces the one
pantheon kernel TODO.md singles out. They are available as an optional mode,
below; running the whole pantheon suite stays the job of the `workloads` CI job.

### Pantheon workloads (optional mode)

```
tools/bench/bench.sh pantheon --build build --pantheon-bin ~/pantheon/build-vgpu-86 \
    --workloads int_virus,fp64_virus --runs 5 --duration 3
```

Runs binaries from a separate pantheon checkout, built for sm_86 with a shared
cudart (`make PLATFORM=CUDA BUILD_DIR=build-vgpu-86 build-vgpu-86/int_virus`, as
the runner script does inside `vgpu shell`), with the runner's knobs
(`0 <duration> 2 --kernel_loops 2 --grid_size 4`, 1 GB VRAM). They do as much work
as fits in the duration, so counters are on in every run and each run's rate is
its own instructions over its own CPU time; there is no counter check for them,
and A/B compares rates only. Measured here (the binaries found in
`build-vgpu-86`, built 2026-09-14/25 from an older pantheon, 3 runs of 3 s,
`nvidia/rtx3060`, load 12-13):

| workload | engine | warp-instr per run (median) | Minst/s median | Minst/s best | spread | cpu s per run |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| int_virus | sass | 42508800 | 14.17 | 15.42 | 11% | 3.00 |
| int_virus | ptx | 96102912 | 31.98 | 34.92 | 18% | 3.01 |
| compute_virus | sass | 17779392 | 5.97 | 8.90 | 59% | 2.98 |
| compute_virus | ptx | 33490080 | 11.14 | 17.18 | 62% | 3.01 |
| fp64_virus | sass | 7385568 | 2.44 | 2.50 | 8% | 3.03 |
| fp64_virus | ptx | 79481088 | 26.46 | 36.03 | 64% | 3.00 |
| mma_virus | sass | 3554304 | 1.17 | 1.19 | 34% | 3.05 |
| mma_virus | ptx | 9435872 | 3.13 | 3.14 | 46% | 3.02 |
| pulse_virus | sass | 470080 | 11.40 | 11.52 | 2% | 0.04 |
| pulse_virus | ptx | 492960 | 15.87 | 15.93 | 1% | 0.03 |

`pulse_virus` sleeps most of each period (about 0.04 s of CPU in 3 s), so its
rate says little. At pantheon's default `--kernel_loops 2` the fp64_virus chain
has not reached the subnormal range; `f64_subnormal` above is the case that has.

### Counter check (the PR gate)

`--check-counters` runs the workloads at the `quick` size on one host thread
(counts of the `atomics` retry loop depend on thread scheduling beyond that),
parses `VGPU_COUNTERS`, and compares **counts, never times**, with
`tools/bench/expected/cuda-<release>.json`: per kernel, instructions,
thread-instructions, divergent branches, barriers, atomics and their bytes, global/shared/local
operations, bytes, sectors, requests, bank conflicts, and the class mix, for both engines. Any
difference fails and lists the kernel and the field. It is the ctest test
`bench_counters` (`tools/bench/run_bench_check.sh`), about one second of
simulation plus a two-second nvcc compile.

Counts depend on the program the toolkit's compiler built, so there is one
expected file per CUDA release: `cuda-12.0.json` (nvcc 12.0.140, the Ubuntu
`nvidia-cuda-toolkit` package that ci.yml's build job installs; the one on this machine reports the same version) and `cuda-13.0.json` (nvcc 13.0.88). The two differ (for
example `int_alu` on SASS is 1120000 warp-instructions with 12.0 and 1120256
with 13.0). The expected file also records the SHA-256 of the compiled program's
`.nv_fatbin` section (the device code; bit-identical across fresh compiles of one
source, checked twice with nvcc 12.0.140). If a toolchain builds different device
code from the same source (another patch level, anything that changes ptxas'
output), the fingerprint differs and the test **skips** with that message instead
of comparing counts of different code; with a matching fingerprint a count that
moved is a simulator change. The compile is pinned: nothing from the caller's
`NVCC_APPEND_FLAGS`/`NVCC_PREPEND_FLAGS` reaches it except a `-ccbin`.
On any other release the test **skips** with a message naming the
releases it has; it does not guess. A change to the simulator that is meant to
be bit- and counter-identical must leave this test green; a deliberate counting
change regenerates the files (`--update-expected`) and says why in its commit.
Under an ASan/TSan shim the test skips by design (the Release job runs it); that path was not exercised here.

I checked that it can fail: raising one stored instruction count by one makes it
exit 1 with `int_alu/sass: _Z9k_int_aluPji: instructions expected 1120257, got 1120256`.

### CI

`.github/workflows/bench.yml` runs on `workflow_dispatch` and nightly, never on
pull requests, and gates nothing: it builds Release on a hosted runner, runs the
harness, uploads `bench.json` as an artifact, prints the table to the job
summary, and runs `--check-counters` as a cross-check. Timing on a hosted runner
is a shared VM's: read it as a trend. **Host instruction counts are not
available here and are not promised**: `perf stat -e instructions:u` on this
machine answers `<not supported>` (WSL2 exposes no PMU; measured), and I could
not measure what a hosted runner offers (no CI dispatch this round). The
workflow records `perf_event_paranoid` and whether a `perf` binary exists in the
job summary, and the harness records the same in the JSON `meta.host_counters`,
so the first nightly run answers it. Everything above is counter-based on
purpose: simulated instruction counts are exact on any host.

## Baseline

`--runs 9 --warmup 1`, one thread, `nvidia/rtx3060`, build at 443e79c0, load
average 10-13 on the 20 logical CPUs while it ran. "Minst/s" is warp
instructions per second of the workload's device phase; wall is the whole
process.

| workload | engine | warp-instr | Minst/s median | Minst/s best | spread | wall s median | wall s best | RSS MB | other |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| int_alu | sass | 11142656 | 19.01 | 19.29 | 4% | 0.599 | 0.590 | 8 |  |
| int_alu | ptx | 11430656 | 40.11 | 40.80 | 5% | 0.298 | 0.294 | 13 |  |
| f32_fma | sass | 5386752 | 17.96 | 18.16 | 49% | 0.317 | 0.313 | 8 |  |
| f32_fma | ptx | 5426432 | 40.53 | 44.11 | 24% | 0.151 | 0.140 | 13 |  |
| f64_fma | sass | 2303744 | 2.23 | 2.32 | 14% | 1.040 | 1.001 | 8 |  |
| f64_fma | ptx | 2270976 | 39.52 | 40.81 | 40% | 0.067 | 0.065 | 13 |  |
| f64_subnormal | sass | 852544 | 1.09 | 1.11 | 13% | 0.784 | 0.770 | 8 |  |
| f64_subnormal | ptx | 865120 | 6.97 | 7.28 | 19% | 0.128 | 0.122 | 13 |  |
| wmma | sass | 308096 | 0.45 | 0.46 | 11% | 0.689 | 0.667 | 8 |  |
| wmma | ptx | 169888 | 4.22 | 4.41 | 25% | 0.044 | 0.042 | 13 |  |
| mma_ptx | sass | 301728 | 0.88 | 0.89 | 9% | 0.346 | 0.341 | 8 |  |
| mma_ptx | ptx | 169920 | 0.56 | 0.57 | 12% | 0.306 | 0.301 | 13 |  |
| mem | sass | 4896768 | 3.29 | 3.35 | 4% | 1.498 | 1.472 | 40 |  |
| mem | ptx | 3258368 | 3.90 | 4.07 | 9% | 0.846 | 0.811 | 44 |  |
| shared | sass | 1395008 | 3.55 | 3.62 | 5% | 0.396 | 0.388 | 10 |  |
| shared | ptx | 1040704 | 1.93 | 1.98 | 4% | 0.544 | 0.530 | 14 |  |
| atomics | sass | 313472 | 2.22 | 2.25 | 22% | 0.144 | 0.143 | 8 |  |
| atomics | ptx | 224000 | 1.74 | 1.77 | 4% | 0.132 | 0.131 | 13 |  |
| texture | sass | 482304 | 2.97 | 3.03 | 18% | 0.166 | 0.162 | 8 |  |
| texture | ptx | 250112 | 0.87 | 1.05 | 25% | 0.304 | 0.244 | 13 |  |
| local | sass | 6958848 | 7.64 | 7.79 | 22% | 0.915 | 0.896 | 8 |  |
| local | ptx | 8880896 | 8.64 | 13.83 | 75% | 1.060 | 0.646 | 13 |  |
| divergent | sass | 9969033 | 6.52 | 7.45 | 37% | 1.537 | 1.345 | 8 |  |
| divergent | ptx | 9976708 | 12.84 | 13.77 | 17% | 0.786 | 0.734 | 13 |  |
| warp | sass | 5126144 | 7.96 | 9.46 | 25% | 0.652 | 0.549 | 8 |  |
| warp | ptx | 5895424 | 24.19 | 27.95 | 35% | 0.250 | 0.220 | 13 |  |
| launch | sass | 360000 | 1.35 | 1.38 | 21% | 0.271 | 0.265 | 8 | 149 k launches/s (best 153 k) |
| launch | ptx | 360000 | 1.47 | 1.56 | 23% | 0.250 | 0.235 | 13 | 163 k launches/s (best 173 k) |
| api | sass | 0 | - | - | 18% | 0.211 | 0.199 | 8 | 4333 k api_calls/s (best 4619 k) |
| api | ptx | 0 | - | - | 18% | 0.239 | 0.222 | 8 | 3844 k api_calls/s (best 4137 k) |
| startup | sass | 9 | 0.01 | 0.01 | 23% | 0.005 | 0.004 | 7 |  |
| startup | ptx | 9 | 0.00 | 0.00 | 32% | 0.015 | 0.013 | 13 |  |

What stands out (each is a ratio of two rows above, same program, same counts):

* The engines are not uniformly ordered. PTX is about twice as fast per
  warp-instruction as SASS on integer and f32 work (int_alu 40 vs 19, f32_fma 41
  vs 18, warp 24 vs 8 Minst/s), 18x faster on f64 (f64_fma 39.5 vs 2.2) and 16x
  faster in wall time on `wmma` (0.044 s vs 0.689 s). SASS is ahead on `shared`
  (3.6 vs 1.9), `atomics` (2.2 vs 1.7), `texture` (3.0 vs 0.9) and `mma_ptx`
  (0.88 vs 0.56 Minst/s); `mem`, `local` and `divergent` favour PTX by 1.2-2x at the
  medians. Both engines run the same kernel to the same results; SASS executes a
  different instruction stream, often a longer one (wmma: 308k vs 170k warp-instructions).
* `f64_subnormal` and `f64_fma` have the same instruction mix, but
  `f64_subnormal` runs at half the SASS rate (1.09 vs 2.23 Minst/s) and a sixth
  of the PTX rate (6.97 vs 39.5). That is consistent with the Intel microcode
  assist for subnormal operands that TODO.md "Performance" describes; this page did not
  measure the assist separately.
* `launch` is about 150k launches/s and `api` about 4M calls/s (about 0.25 us per
  call, process-local).

## Instruction mix

From `VGPU_COUNTERS` (`inst_by_class`, counted per active lane; the classes
partition the thread-instructions and the table checks that they sum to 100%).
`lanes` is thread-instructions per warp-instruction. The engines count different
instruction streams (SASS counts lanes after the guard predicate, PTX before it; see
`src/sass/exec_ops.inc:133-142` and `src/exec/interpreter.cpp:2805-2815`), so
mixes are comparable within an engine, not across.

| workload | engine | warp-instr | lanes | fp16 | fp32 | fp64 | int | cvt | ctrl | mem | tensor | misc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| int_alu | sass | 11142656 | 30.6 | 0.0% | 0.0% | 0.0% | 98.2% | 0.0% | 1.8% | 0.0% | 0.0% | 0.0% |
| int_alu | ptx | 11430656 | 32.0 | 0.0% | 0.0% | 0.0% | 92.4% | 0.8% | 3.4% | 0.0% | 0.0% | 3.4% |
| f32_fma | sass | 5386752 | 32.0 | 0.0% | 97.0% | 0.0% | 1.5% | 0.0% | 0.7% | 0.0% | 0.0% | 0.8% |
| f32_fma | ptx | 5426432 | 32.0 | 0.0% | 96.3% | 0.0% | 0.8% | 0.0% | 0.7% | 0.0% | 0.0% | 2.2% |
| f64_fma | sass | 2303744 | 32.0 | 0.0% | 0.0% | 94.6% | 3.1% | 0.0% | 0.7% | 0.0% | 0.0% | 1.5% |
| f64_fma | ptx | 2270976 | 32.0 | 0.0% | 0.0% | 95.9% | 0.9% | 0.0% | 0.8% | 0.1% | 0.0% | 2.4% |
| f64_subnormal | sass | 852544 | 32.0 | 0.0% | 0.0% | 96.1% | 2.3% | 0.0% | 1.5% | 0.0% | 0.0% | 0.0% |
| f64_subnormal | ptx | 865120 | 32.0 | 0.0% | 0.0% | 94.7% | 1.5% | 0.0% | 1.5% | 0.8% | 0.0% | 1.6% |
| wmma | sass | 308096 | 22.0 | 0.0% | 0.0% | 0.0% | 6.1% | 0.0% | 2.9% | 0.2% | 90.6% | 0.2% |
| wmma | ptx | 169888 | 32.0 | 0.0% | 0.0% | 0.0% | 14.4% | 0.1% | 14.2% | 0.1% | 56.5% | 14.7% |
| mma_ptx | sass | 301728 | 12.3 | 0.0% | 0.0% | 0.0% | 11.2% | 0.0% | 5.2% | 0.3% | 83.0% | 0.3% |
| mma_ptx | ptx | 169920 | 32.0 | 0.0% | 0.0% | 0.0% | 14.6% | 0.1% | 14.2% | 0.3% | 56.5% | 14.4% |
| mem | sass | 4896768 | 31.9 | 0.0% | 0.0% | 0.0% | 81.0% | 0.0% | 6.0% | 12.1% | 0.0% | 0.8% |
| mem | ptx | 3258368 | 32.0 | 0.0% | 0.0% | 0.0% | 61.0% | 0.9% | 9.4% | 18.7% | 0.0% | 10.0% |
| shared | sass | 1395008 | 18.7 | 0.0% | 0.0% | 0.0% | 28.3% | 0.0% | 29.8% | 39.3% | 0.0% | 2.7% |
| shared | ptx | 1040704 | 30.7 | 0.0% | 0.0% | 0.0% | 14.3% | 0.3% | 41.7% | 32.5% | 0.0% | 11.2% |
| atomics | sass | 313472 | 27.8 | 0.0% | 0.0% | 0.0% | 63.3% | 4.7% | 7.2% | 17.5% | 0.0% | 7.2% |
| atomics | ptx | 224000 | 30.6 | 0.0% | 0.0% | 0.0% | 49.7% | 12.3% | 2.7% | 31.4% | 0.0% | 3.9% |
| texture | sass | 482304 | 27.9 | 0.0% | 18.3% | 0.0% | 52.2% | 7.4% | 0.9% | 11.0% | 0.0% | 10.1% |
| texture | ptx | 250112 | 32.0 | 0.0% | 30.7% | 0.0% | 34.2% | 12.3% | 1.7% | 0.4% | 0.0% | 20.6% |
| local | sass | 6958848 | 32.0 | 0.0% | 0.0% | 0.0% | 81.9% | 0.0% | 1.4% | 16.7% | 0.0% | 0.0% |
| local | ptx | 8880896 | 32.0 | 0.0% | 0.0% | 0.0% | 71.7% | 13.0% | 1.1% | 13.1% | 0.0% | 1.1% |
| divergent | sass | 9969033 | 14.9 | 0.0% | 0.0% | 0.0% | 64.0% | 0.0% | 28.7% | 0.0% | 0.0% | 7.3% |
| divergent | ptx | 9976708 | 15.9 | 0.0% | 0.0% | 0.0% | 38.8% | 7.7% | 27.0% | 0.0% | 0.0% | 26.5% |
| warp | sass | 5126144 | 30.7 | 0.0% | 0.0% | 0.0% | 74.0% | 0.0% | 1.0% | 0.0% | 0.0% | 25.0% |
| warp | ptx | 5895424 | 32.0 | 0.0% | 0.0% | 0.0% | 53.9% | 0.0% | 0.9% | 0.0% | 0.0% | 45.2% |
| launch | sass | 360000 | 32.0 | 0.0% | 0.0% | 0.0% | 22.2% | 0.0% | 11.1% | 22.2% | 0.0% | 44.4% |
| launch | ptx | 360000 | 32.0 | 0.0% | 0.0% | 0.0% | 33.3% | 11.1% | 11.1% | 33.3% | 0.0% | 11.1% |
| api | sass | 0 | - | - | - | - | - | - | - | - | - | - |
| api | ptx | 0 | - | - | - | - | - | - | - | - | - | - |
| startup | sass | 9 | 32.0 | 0.0% | 0.0% | 0.0% | 22.2% | 0.0% | 11.1% | 22.2% | 0.0% | 44.4% |
| startup | ptx | 9 | 32.0 | 0.0% | 0.0% | 0.0% | 33.3% | 11.1% | 11.1% | 33.3% | 0.0% | 11.1% |

The SASS engine keeps no per-opcode histogram (`inst_by_opcode` is filled only by
the PTX interpreter, `src/exec/interpreter.cpp:2816-2823`); `bench.sh mix` prints
the ten most-used PTX opcodes per launch beside the class table. The SASS
`mix` for `startup`/`api` is empty or tiny because they launch one or no kernel.

## Noise

Same build as A and as B, 5 runs each, interleaved A B A B, 32 workload/engine pairs:

| statistic | median ratio B/A | best-of-5 ratio B/A |
| --- | ---: | ---: |
| median of \|ratio-1\| | 3.9% | 2.2% |
| 90th percentile of \|ratio-1\| | 9.7% | 11.1% |
| worst \|ratio-1\| | 20.5% (divergent/ptx) | 30.6% (mem/sass) |
| run-to-run spread (max-min)/median, median over workloads | 24% | |

Reading it: with 5 runs a side, a difference under about 5% between two builds
is inside what the same binary shows against itself on this machine, and a 10%
difference can still be an accident for the short workloads. For a claim, use
`--runs 9` or more, prefer the best-of ratio and the p-value, repeat when the
load average changes, and when a difference matters, repeat the A/B with the
order swapped. Two things that helped when measured: the minimum is much steadier
than the median, and a lot of the run-to-run difference is where the scheduler
puts the process rather than the code. Six consecutive runs of `f32_fma` (SASS,
full) took 0.303 to 0.595 s of device-phase CPU time unpinned; pinned to logical
CPU 4 they took 0.304-0.316 s and to CPU 17 0.305-0.314 s, while pinned to CPU 0
(which another job was using) 0.306-0.558 s. `--cpus` exists for that: pin to a
CPU that is quiet, and check the load before you trust a run.

## Profiles

**Tool.** `perf` is not installed. It was extracted without installing from
Ubuntu's linux-tools package (TODO.md "Performance"):
`apt-get download linux-tools-6.17.0-42-generic linux-hwe-6.17-tools-6.17.0-42`,
then `dpkg-deb -x` of both into a directory; the binary is
`usr/lib/linux-hwe-6.17-tools-6.17.0-42/perf` (version 6.17.13 here), run with
`PERF=<path>`. Because WSL2 exposes no PMU, sampling is the software
`cpu-clock:u` event (user space only) at 2999 Hz, which `bench.sh profile` drives:

```
PERF=$HOME/.cache/perf-tools/x/usr/lib/linux-hwe-6.17-tools-6.17.0-42/perf \
  tools/bench/bench.sh profile --build build-g --top 25            # functions
  tools/bench/bench.sh profile --build build-g --top 25 --by-line  # source lines
```

**Build.** The same flags as the baseline (Release `-O3 -DNDEBUG`, LTO, GCC 13.3,
baseline x86-64) plus `-DCMAKE_CXX_FLAGS=-g1` (line tables, no effect on code
generation), configured in a separate directory. Workloads at the `full` size,
one run each; samples are summed per group and per engine. Counts of samples are
in each heading.

**Caveats.** (1) A flat profile of a link-time-optimised build: small functions
are inlined into their callers, so a row such as `fast_alu` is "everything
inlined into fast_alu"; the line tables below say where inside it. Clone suffixes
(`.lto_priv`, `.constprop`) and argument lists are dropped, which folds a
function's lambdas into rows of the same name. (2) Software-clock sampling skids:
a sample lands on the instruction after the one that was running (a libm row
below is a `ret`, right after the `vfmadd213sd` that costs the time). (3) User
space only: kernel time is not in the tables and was not quantified. (4) Libraries without symbols (libm, libc) show as addresses;
the ones that matter were identified with `objdump` and are named in the notes.
(5) Line numbers are those of 443e79c0 and move as the other speed branches land.
(6) One run per workload per engine, on a loaded machine: shares are stable to a
point or two, not to the decimal shown.

### Findings from the profiles (leads, each checked against the code)

Not changes: the next agents decide what is safe. Every optimisation must keep
all counters, results and error messages identical.

1. **SASS f64 goes the slow general route and copies strings per lane.**
   `DADD/DMUL/DFMA` are handled in `Runner::exec_float`, per lane through
   `each_lane`, `f64src`, `std::fma` and `write64`
   (`src/sass/exec_ops.inc:1369-1383`; dispatched at `:197-199`); the fast path `fast_alu`
   (`:987`) handles `FFMA/FADD/FMUL` (`:1240-1262`) and has no f64 case. `Runner::write64` makes a copy of its operand,
   `Operand hi = o;` (`src/sass/exec.cpp:593`), and `Operand` holds two
   `std::string` members (`include/vgpu/sass/sass.hpp:74-75`): with
   `perf --call-graph dwarf` on f64_fma, `Operand::Operand(const Operand&)` and
   `~basic_string` appear under `write64`, and `std::string::_M_construct` is 6.4% of
   the SASS alu group. The alu group also spends 15.4% on one libm address,
   `0x327f9` in libm.so.6, which `objdump -d` shows is the `ret` after
   `vfmadd213sd` in libm's FMA3 `fma` (the hardware FMA reached through `fma@plt`
   per lane; the PTX path calls an inlined `target("fma")` lane loop,
   `src/exec/interpreter.cpp:932`). Net: SASS f64_fma 2.2 vs PTX 39.5 Minst/s.
2. **SASS tensor ops are a scalar triple loop with software half conversion.**
   `exec_mma` is 76% of the SASS tensor group, and libgcc's `__extendhfdf2`
   (half -> double, a software routine on a build without F16C) another 16-19%.
   The line view puts 49% of the group on `exec_ops.inc:3009-3014`, the
   `a = a + prod` product loop. The PTX interpreter's wmma uses `wmma_tile_avx2`
   (`interpreter.cpp:1027`). Net: wmma 0.45 (SASS) vs 4.2 (PTX) Minst/s, and
   0.7 s vs 0.044 s wall.
3. **SASS memory operations pay for counters with a heap allocation and a sort
   each.** `count_access` (`src/sass/exec_ops.inc:1782-1829`) builds a
   `std::vector<uint64_t>` of sectors per memory instruction and `std::sort`s it
   (`:1786-1790`); its shared-memory branch builds 32 vectors and sorts each
   (`:1809-1820`).
   `count_access` is 12.0% of the SASS memory group, `std::__introsort_loop`
   4.6%, `MemoryManager::resolve` 8.5%, `BigReaderLock::unlock_shared` 6.5%
   (`include/vgpu/brlock.hpp:27-39`: `lock_shared` is a seq_cst `fetch_add`
   plus a load, `unlock_shared` a `fetch_sub`, on a slot chosen through a
   `thread_local`; in the line view the two atomic operations, `atomic_base.h:505` and
   `:645`, are 5.2% and 4.5% of the SASS memory group. `__tls_get_addr` is another 1-2%
   in the memory and api groups, which I did not trace to that `thread_local`). These counts are what `VGPU_COUNTERS` reports exactly, so any
   change here is held by the counter check.
4. **`std::popcount` is a libgcc call.** The build has no `-mpopcnt`, so
   `std::popcount` calls `__popcountdi2`: call graph on int_alu (SASS) shows it
   from the per-lane `POPC` lambda in `exec_int` (6.2%) and from
   `Runner::execute` (1.2%, the per-instruction `std::popcount(ex)`,
   `exec_ops.inc:139`); it is 2-8% of the integer-heavy groups on both engines.
5. **PTX dispatch.** In the PTX alu and control groups `run_warp_until_yield`,
   `dispatch` and the `fast_*` helpers account for most samples (22-30% each for the
   first two in the control group); in the PTX memory group `exec_ld`/`exec_st`
   are 24%/17% with `resolve` and the lock another 15%. `string_fortified.h:29`
   (an inlined `memcpy`) is 15% of the PTX alu group's line view; its callers
   were not determined.
6. **API calls.** In the api group `memset` (libc `rep stos` at `0x18984a`) is 9%,
   `clock_gettime` through the vDSO 4-7% and `profiling::ApiCall::ApiCall` 4-5%
   (so each runtime call reads the clock; not investigated further).

#### SASS engine, alu group: 7886 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 16.0% | `libcudart vgpu::sass::Runner::fast_alu` |
| 2 | 15.4% | `libm 0x00000000000327f9` |
| 3 | 14.1% | `libcudart vgpu::sass::Runner::read64` |
| 4 | 12.2% | `libcudart vgpu::sass::Runner::write64` |
| 5 | 9.0% | `libcudart vgpu::sass::Runner::exec_float::{lambda#1}::operator` |
| 6 | 7.1% | `libcudart vgpu::sass::Runner::exec_float` |
| 7 | 6.4% | `libcudart void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>` |
| 8 | 2.4% | `libcudart vgpu::sass::Runner::execute` |
| 9 | 2.0% | `libgcc_s __popcountdi2` |
| 10 | 1.7% | `libcudart fmaf@plt` |
| 11 | 1.5% | `libcudart vgpu::sass::straight` |
| 12 | 1.4% | `libcudart vgpu::sass::Runner::read32` |
| 13 | 1.1% | `libcudart vgpu::sass::Runner::fast_alu::{lambda#1}::operator<vgpu::sass::Runner::fast_alu::IntMod>` |
| 14 | 1.1% | `libcudart vgpu::sass::Runner::step_warp` |
| 15 | 1.0% | `libcudart vgpu::sass::Runner::fast_alu::{lambda#1}::operator<vgpu::sass::Runner::fast_alu::Same>` |
| 16 | 1.0% | `libcudart vgpu::sass::Runner::fast_alu::{lambda#4}::operator` |
| 17 | 1.0% | `libcudart vgpu::sass::Runner::fast_alu::{lambda#1}::operator<vgpu::sass::Runner::fast_alu::FMod>` |
| 18 | 0.8% | `libcudart fma@plt` |
| 19 | 0.7% | `libcudart vgpu::sass::Code::instr` |
| 20 | 0.6% | `libcudart vgpu::sass::Runner::exec_int` |
| 21 | 0.6% | `libcudart vgpu::sass::Runner::read32_slow` |
| 22 | 0.4% | `libcudart vgpu::sass::Runner::write32` |
| 23 | 0.4% | `libcudart __popcountdi2@plt` |
| 24 | 0.3% | `libcudart vgpu::sass::Runner::exec_int::{lambda#5}::operator` |
| 25 | 0.2% | `libm 0x0000000000042969` |

Per workload, top 5: **int_alu** (1717 samples): 45% `vgpu::sass::Runner::fast_alu`, 8% `libgcc_s __popcountdi2`, 6% `vgpu::sass::Runner::execute`, 5% `vgpu::sass::Runner::read32`, 5% `vgpu::sass::Runner::fast_alu::{lambda#1}::oper`; **f32_fma** (922 samples): 49% `vgpu::sass::Runner::fast_alu`, 15% `fmaf@plt`, 8% `vgpu::sass::Runner::fast_alu::{lambda#1}::oper`, 5% `vgpu::sass::Runner::execute`, 5% `vgpu::sass::Runner::fast_alu::{lambda#4}::oper`; **f64_fma** (2933 samples): 25% `vgpu::sass::Runner::read64`, 23% `vgpu::sass::Runner::write64`, 17% `vgpu::sass::Runner::exec_float::{lambda#1}::op`, 15% `vgpu::sass::Runner::exec_float`, 11% `void std::__cxx11::basic_string<char, std::cha`; **f64_subnormal** (2314 samples): 47% `libm 0x00000000000327f9`, 16% `vgpu::sass::Runner::read64`, 12% `vgpu::sass::Runner::write64`, 9% `vgpu::sass::Runner::exec_float::{lambda#1}::op`, 7% `void std::__cxx11::basic_string<char, std::cha`

#### SASS engine, tensor group: 3002 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 76.1% | `libcudart vgpu::sass::Runner::exec_mma` |
| 2 | 16.2% | `libgcc_s __extendhfdf2` |
| 3 | 2.8% | `libcudart __extendhfdf2@plt` |
| 4 | 1.6% | `libcudart vgpu::sass::Runner::step_warp` |
| 5 | 0.7% | `libcudart vgpu::sass::Runner::execute` |
| 6 | 0.5% | `libcudart vgpu::sass::Module::code_at` |
| 7 | 0.4% | `libcudart vgpu::sass::Runner::fast_alu` |
| 8 | 0.3% | `libc cfree` |
| 9 | 0.2% | `libcudart vgpu::sass::straight` |
| 10 | 0.2% | `libcudart vgpu::sass::Code::instr` |
| 11 | 0.1% | `libc 0x0000000000189804` |
| 12 | 0.1% | `libc malloc` |
| 13 | 0.1% | `libgcc_s __popcountdi2` |
| 14 | 0.1% | `libc 0x00000000000ab7d0` |
| 15 | 0.1% | `libcudart vgpu::sass::Runner::run::{lambda#1}::operator` |
| 16 | 0.1% | `libstdc++ operator new` |
| 17 | 0.0% | `ld-linux-x86-64 0x000000000000b133` |
| 18 | 0.0% | `ld-linux-x86-64 0x0000000000016848` |
| 19 | 0.0% | `libc 0x000000000006a5b8` |
| 20 | 0.0% | `libc 0x00000000000aac09` |
| 21 | 0.0% | `libc 0x00000000000ab28e` |
| 22 | 0.0% | `libc 0x00000000000ab2dc` |
| 23 | 0.0% | `libc 0x00000000000ab394` |
| 24 | 0.0% | `libc 0x00000000000ab68d` |
| 25 | 0.0% | `libc 0x00000000000ac013` |

Per workload, top 5: **wmma** (1996 samples): 78% `vgpu::sass::Runner::exec_mma`, 15% `libgcc_s __extendhfdf2`, 3% `__extendhfdf2@plt`, 2% `vgpu::sass::Runner::step_warp`, 1% `vgpu::sass::Runner::execute`; **mma_ptx** (1006 samples): 73% `vgpu::sass::Runner::exec_mma`, 18% `libgcc_s __extendhfdf2`, 2% `__extendhfdf2@plt`, 2% `vgpu::sass::Runner::step_warp`, 1% `vgpu::sass::Runner::fast_alu`

#### SASS engine, memory group: 9179 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 12.0% | `libcudart vgpu::sass::count_access` |
| 2 | 9.1% | `libcudart vgpu::sass::Runner::fast_alu` |
| 3 | 8.5% | `libcudart vgpu::MemoryManager::resolve` |
| 4 | 6.5% | `libcudart vgpu::BigReaderLock::unlock_shared` |
| 5 | 5.6% | `libcudart vgpu::sass::Runner::exec_int::{lambda#7}::operator` |
| 6 | 5.5% | `libcudart vgpu::sass::Runner::exec_mem` |
| 7 | 4.6% | `libcudart void std::__introsort_loop<__gnu_cxx::__normal_iterator<unsigned long*, std::vector<unsigned long, std::allocator<unsign` |
| 8 | 3.4% | `libcudart vgpu::sass::Runner::exec_mem::{lambda#1}::operator` |
| 9 | 2.7% | `libcudart vgpu::sass::Runner::mem_write` |
| 10 | 2.4% | `libcudart vgpu::MemoryManager::load_scalar` |
| 11 | 2.1% | `libcudart vgpu::sass::Runner::mem_read` |
| 12 | 2.1% | `libcudart vgpu::sass::Runner::step_warp` |
| 13 | 1.9% | `libcudart vgpu::sass::Runner::execute` |
| 14 | 1.9% | `libcudart vgpu::load_at` |
| 15 | 1.8% | `libcudart vgpu::sass::Runner::read32_slow` |
| 16 | 1.6% | `libcudart vgpu::sass::Runner::read32` |
| 17 | 1.6% | `libcudart vgpu::MemoryManager::scalar_location` |
| 18 | 1.4% | `libc malloc` |
| 19 | 1.3% | `libcudart vgpu::sass::Runner::exec_int::{lambda#11}::operator` |
| 20 | 1.1% | `ld-linux-x86-64 __tls_get_addr` |
| 21 | 1.1% | `libcudart vgpu::sass::Runner::write32` |
| 22 | 1.0% | `libcudart vgpu::sass::Runner::fast_alu::{lambda#1}::operator<vgpu::sass::Runner::fast_alu::IntMod>` |
| 23 | 1.0% | `libcudart vgpu::sass::Runner::cbank32` |
| 24 | 0.9% | `libcudart __tls_get_addr@plt` |
| 25 | 0.9% | `libcudart vgpu::exec::fetch_texel` |

Per workload, top 5: **mem** (4265 samples): 14% `vgpu::MemoryManager::resolve`, 12% `vgpu::sass::Runner::exec_int::{lambda#7}::oper`, 11% `vgpu::BigReaderLock::unlock_shared`, 6% `vgpu::sass::Runner::fast_alu`, 5% `vgpu::sass::count_access`; **shared** (1158 samples): 19% `vgpu::sass::count_access`, 7% `vgpu::MemoryManager::resolve`, 6% `vgpu::sass::Runner::exec_mem`, 5% `vgpu::BigReaderLock::unlock_shared`, 4% `vgpu::sass::Runner::exec_mem::{lambda#1}::oper`; **atomics** (415 samples): 16% `vgpu::MemoryManager::resolve`, 14% `vgpu::BigReaderLock::unlock_shared`, 8% `vgpu::sass::Runner::exec_mem`, 6% `vgpu::sass::Runner::mem_write`, 4% `vgpu::MemoryManager::load_scalar`; **texture** (528 samples): 15% `vgpu::exec::fetch_texel`, 11% `vgpu::exec::tex_finish`, 9% `vgpu::sass::Runner::exec_tex`, 7% `vgpu::MemoryManager::resolve`, 7% `vgpu::exec::tex_footprint_linear`; **local** (2813 samples): 24% `vgpu::sass::count_access`, 18% `vgpu::sass::Runner::fast_alu`, 12% `void std::__introsort_loop<__gnu_cxx::__normal`, 10% `vgpu::sass::Runner::exec_mem`, 6% `vgpu::sass::Runner::exec_mem::{lambda#1}::oper`

#### SASS engine, control group: 5077 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 24.3% | `libcudart vgpu::sass::Runner::fast_alu` |
| 2 | 21.4% | `libcudart vgpu::sass::Runner::step_warp` |
| 3 | 12.3% | `libcudart vgpu::sass::Runner::execute` |
| 4 | 11.3% | `libcudart vgpu::sass::Runner::fast_alu::{lambda#4}::operator` |
| 5 | 4.6% | `libcudart vgpu::sass::Runner::exec_control` |
| 6 | 3.8% | `libcudart vgpu::sass::Runner::read32` |
| 7 | 3.1% | `libcudart vgpu::sass::Runner::read32_slow` |
| 8 | 3.0% | `libcudart vgpu::sass::straight` |
| 9 | 2.4% | `libcudart vgpu::sass::Runner::exec_int` |
| 10 | 2.3% | `libcudart vgpu::sass::Runner::exec_control::{lambda#3}::operator` |
| 11 | 2.2% | `libgcc_s __popcountdi2` |
| 12 | 1.5% | `libcudart vgpu::sass::Runner::fast_alu::{lambda#1}::operator<vgpu::sass::Runner::fast_alu::Same>` |
| 13 | 1.3% | `libcudart vgpu::sass::Runner::write32` |
| 14 | 1.2% | `libcudart vgpu::sass::Runner::fast_alu::{lambda#1}::operator<vgpu::sass::Runner::fast_alu::IntMod>` |
| 15 | 1.0% | `libcudart vgpu::sass::Code::instr` |
| 16 | 1.0% | `libcudart vgpu::sass::Runner::exec_int::{lambda#31}::operator const::{lambda#1}::operator` |
| 17 | 0.7% | `libcudart vgpu::sass::Module::code_at` |
| 18 | 0.6% | `libcudart vgpu::sass::Runner::fetch` |
| 19 | 0.6% | `libcudart __popcountdi2@plt` |
| 20 | 0.4% | `libcudart vgpu::sass::Runner::exec_control::{lambda#2}::operator` |
| 21 | 0.2% | `bench_kernels.13.0.88.fbfe9ef9fc28316f w_divergent` |
| 22 | 0.2% | `libcudart vgpu::sass::Code::instr const@plt` |
| 23 | 0.1% | `libcudart vgpu::sass::Runner::run::{lambda#1}::operator` |
| 24 | 0.1% | `bench_kernels.13.0.88.fbfe9ef9fc28316f w_warp` |
| 25 | 0.1% | `libcudart vgpu::sass::Runner::init_block` |

Per workload, top 5: **divergent** (3736 samples): 25% `vgpu::sass::Runner::fast_alu`, 24% `vgpu::sass::Runner::step_warp`, 15% `vgpu::sass::Runner::fast_alu::{lambda#4}::oper`, 6% `vgpu::sass::Runner::execute`, 6% `vgpu::sass::Runner::exec_control`; **warp** (1341 samples): 29% `vgpu::sass::Runner::execute`, 22% `vgpu::sass::Runner::fast_alu`, 15% `vgpu::sass::Runner::step_warp`, 6% `vgpu::sass::Runner::read32_slow`, 5% `vgpu::sass::Runner::write32`

#### SASS engine, api group: 1187 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 9.2% | `libc 0x000000000018984a` |
| 2 | 6.9% | `libcudart vgpu::MemoryManager::resolve` |
| 3 | 5.0% | `libcudart vgpu::BigReaderLock::unlock_shared` |
| 4 | 4.4% | `[vdso] __vdso_clock_gettime` |
| 5 | 4.4% | `libcudart vgpu::profiling::ApiCall::ApiCall` |
| 6 | 2.3% | `ld-linux-x86-64 __tls_get_addr` |
| 7 | 1.9% | `libcudart vgpu::sass::Runner::step_warp` |
| 8 | 1.9% | `libcudart vgpu::sass::count_access` |
| 9 | 1.9% | `libc malloc` |
| 10 | 1.9% | `libcudart vgpu::MemoryManager::materialize` |
| 11 | 1.9% | `libcudart vgpu::MemoryManager::free` |
| 12 | 1.7% | `libcudart vgpu::sass::Runner::exec_control::{lambda#3}::operator` |
| 13 | 1.7% | `libcudart vgpu::BigReaderLock::lock` |
| 14 | 1.5% | `libcudart vgpu::sass::Runner::exec_mem::{lambda#1}::operator` |
| 15 | 1.5% | `libcudart vgpu::sass::Runner::mem_read` |
| 16 | 1.3% | `libcudart vgpu::sass::Runner::exec_int::{lambda#5}::operator` |
| 17 | 1.3% | `libcudart vgpu::sass::Runner::write64` |
| 18 | 1.3% | `libstdc++ std::_Rb_tree_insert_and_rebalance` |
| 19 | 1.3% | `libc 0x0000000000189087` |
| 20 | 1.2% | `libcudart vgpu::sass::Runner::init_block` |
| 21 | 1.2% | `libcudart vgpu::sass::Runner::read32` |
| 22 | 1.0% | `libcudart vgpu::MemoryManager::scalar_location` |
| 23 | 1.0% | `libc cfree` |
| 24 | 0.9% | `libcudart vgpu::MemoryManager::load_scalar` |
| 25 | 0.9% | `libcudart vgpu::runtime::Device::launch` |

Per workload, top 5: **launch** (613 samples): 10% `vgpu::MemoryManager::resolve`, 9% `vgpu::BigReaderLock::unlock_shared`, 4% `vgpu::sass::Runner::step_warp`, 4% `vgpu::sass::count_access`, 3% `vgpu::sass::Runner::exec_control::{lambda#3}::`; **api** (564 samples): 19% `libc 0x000000000018984a`, 9% `[vdso] __vdso_clock_gettime`, 9% `vgpu::profiling::ApiCall::ApiCall`, 4% `vgpu::MemoryManager::materialize`, 4% `vgpu::MemoryManager::free`; **startup** (10 samples): 10% `ld-linux-x86-64 0x000000000000f9e2`, 10% `ld-linux-x86-64 0x00000000000274fe`, 10% `libc malloc`, 10% `libc 0x0000000000069c97`, 10% `libc 0x00000000000ab2b9`

#### PTX engine, alu group: 3434 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 22.1% | `libcudart vgpu::exec::Interpreter::run_warp_until_yield` |
| 2 | 17.3% | `libcudart vgpu::exec::fma_lanes_f64_hw` |
| 3 | 10.6% | `libcudart vgpu::exec::Interpreter::fast_fma` |
| 4 | 10.4% | `libcudart vgpu::exec::Interpreter::dispatch` |
| 5 | 10.1% | `libcudart vgpu::exec::Interpreter::fast_int_bin` |
| 6 | 3.8% | `libgcc_s __popcountdi2` |
| 7 | 3.7% | `libcudart vgpu::exec::Interpreter::read_narrow` |
| 8 | 3.2% | `libcudart vgpu::exec::Interpreter::read_operand` |
| 9 | 2.2% | `libcudart vgpu::exec::fma_lanes_f32_hw` |
| 10 | 1.4% | `libcudart vgpu::exec::Interpreter::write_reg` |
| 11 | 1.1% | `libm 0x00000000000327f9` |
| 12 | 1.0% | `libcudart auto vgpu::exec::Interpreter::fast_setp::{lambda#2}::operator<vgpu::exec::Interpreter::fast_setp::{lambda#1}>` |
| 13 | 0.9% | `bench_kernels.13.0.88.fbfe9ef9fc28316f w_f32_fma` |
| 14 | 0.7% | `bench_kernels.13.0.88.fbfe9ef9fc28316f w_int_alu` |
| 15 | 0.7% | `libcudart vgpu::exec::Interpreter::float_bin` |
| 16 | 0.6% | `libcudart __popcountdi2@plt` |
| 17 | 0.6% | `libcudart vgpu::exec::Interpreter::fast_float_bin` |
| 18 | 0.4% | `bench_kernels.13.0.88.fbfe9ef9fc28316f fmaf@plt` |
| 19 | 0.4% | `bench_kernels.13.0.88.fbfe9ef9fc28316f w_f64_fma` |
| 20 | 0.3% | `libcudart vgpu::ptx::Parser::intern` |
| 21 | 0.3% | `libm 0x0000000000042969` |
| 22 | 0.3% | `libc malloc` |
| 23 | 0.3% | `libcudart vgpu::exec::Interpreter::fast_path` |
| 24 | 0.3% | `libm 0x00000000000327f0` |
| 25 | 0.2% | `libcudart vgpu::exec::Interpreter::fast_cvt_int` |

Per workload, top 5: **int_alu** (1703 samples): 30% `vgpu::exec::Interpreter::run_warp_until_yield`, 20% `vgpu::exec::Interpreter::fast_int_bin`, 19% `vgpu::exec::Interpreter::dispatch`, 8% `libgcc_s __popcountdi2`, 6% `vgpu::exec::Interpreter::read_operand`; **f32_fma** (689 samples): 43% `vgpu::exec::Interpreter::fast_fma`, 20% `vgpu::exec::Interpreter::run_warp_until_yield`, 11% `vgpu::exec::fma_lanes_f32_hw`, 6% `vgpu::exec::Interpreter::read_narrow`, 5% `bench_kernels.13.0.88.fbfe9ef9fc28316f w_f32_f`; **f64_fma** (384 samples): 20% `vgpu::exec::Interpreter::run_warp_until_yield`, 14% `vgpu::exec::Interpreter::fast_fma`, 12% `vgpu::exec::fma_lanes_f64_hw`, 10% `vgpu::exec::Interpreter::dispatch`, 9% `libm 0x00000000000327f9`; **f64_subnormal** (658 samples): 83% `vgpu::exec::fma_lanes_f64_hw`, 5% `vgpu::exec::Interpreter::run_warp_until_yield`, 3% `vgpu::exec::Interpreter::fast_fma`, 0% `vgpu::ptx::Parser::intern`, 0% `libc cfree`

#### PTX engine, tensor group: 1232 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 70.5% | `libcudart vgpu::exec::Interpreter::exec_mma` |
| 2 | 5.9% | `libcudart vgpu::exec::Interpreter::fast_wmma_mma::{lambda#1}::operator` |
| 3 | 3.6% | `libcudart vgpu::exec::f16_to_double` |
| 4 | 3.3% | `libcudart vgpu::exec::wmma_tile_avx2` |
| 5 | 2.4% | `libcudart vgpu::exec::Interpreter::run_warp_until_yield` |
| 6 | 2.0% | `libcudart vgpu::exec::Interpreter::read_operand` |
| 7 | 1.5% | `libcudart vgpu::exec::Interpreter::fast_wmma_mma` |
| 8 | 1.1% | `libcudart vgpu::exec::gather8x8` |
| 9 | 0.6% | `libcudart vgpu::exec::scatter8x8` |
| 10 | 0.5% | `libcudart vgpu::exec::Interpreter::dispatch` |
| 11 | 0.4% | `libc 0x0000000000188d80` |
| 12 | 0.4% | `libcudart std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string` |
| 13 | 0.2% | `libcudart memcmp@plt` |
| 14 | 0.2% | `libcudart vgpu::ptx::lex` |
| 15 | 0.2% | `libcudart vgpu::ptx::Parser::parse_instruction` |
| 16 | 0.2% | `libc 0x0000000000189813` |
| 17 | 0.2% | `libcudart decltype std::__do_visit<void, std::__detail::__variant::_Variant_storage<false, vgpu::ptx::RegOperand, vgpu::ptx::ImmIn` |
| 18 | 0.2% | `libcudart vgpu::exec::Interpreter::write_reg` |
| 19 | 0.2% | `libc 0x000000000018891d` |
| 20 | 0.2% | `libcudart auto vgpu::exec::Interpreter::fast_setp::{lambda#2}::operator<vgpu::exec::Interpreter::fast_setp::{lambda#1}>` |
| 21 | 0.2% | `libcudart std::_Rb_tree<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_strin` |
| 22 | 0.2% | `libstdc++ std::_Hash_bytes` |
| 23 | 0.2% | `libc 0x00000000000ab297` |
| 24 | 0.2% | `libc 0x0000000000188708` |
| 25 | 0.2% | `libc 0x0000000000188928` |

Per workload, top 5: **wmma** (218 samples): 33% `vgpu::exec::Interpreter::fast_wmma_mma::{lambd`, 19% `vgpu::exec::wmma_tile_avx2`, 8% `vgpu::exec::Interpreter::fast_wmma_mma`, 6% `vgpu::exec::gather8x8`, 6% `vgpu::exec::Interpreter::run_warp_until_yield`; **mma_ptx** (1014 samples): 86% `vgpu::exec::Interpreter::exec_mma`, 4% `vgpu::exec::f16_to_double`, 2% `vgpu::exec::Interpreter::read_operand`, 2% `vgpu::exec::Interpreter::run_warp_until_yield`, 0% `std::__cxx11::basic_string<char, std::char_tra`

#### PTX engine, memory group: 8804 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 24.3% | `libcudart vgpu::exec::Interpreter::exec_ld` |
| 2 | 16.5% | `libcudart vgpu::exec::Interpreter::exec_st` |
| 3 | 8.9% | `libcudart vgpu::MemoryManager::resolve` |
| 4 | 6.5% | `libcudart vgpu::BigReaderLock::unlock_shared` |
| 5 | 5.7% | `libcudart vgpu::exec::Interpreter::run_warp_until_yield` |
| 6 | 2.8% | `libcudart vgpu::exec::Interpreter::fast_int_bin` |
| 7 | 2.5% | `libcudart vgpu::exec::Interpreter::check_local` |
| 8 | 2.2% | `libcudart vgpu::exec::Interpreter::read_operand` |
| 9 | 2.1% | `libcudart vgpu::exec::Interpreter::fast_int_bin64` |
| 10 | 2.0% | `libcudart vgpu::MemoryManager::scalar_location` |
| 11 | 1.9% | `libcudart vgpu::exec::Interpreter::lane_local` |
| 12 | 1.8% | `libcudart vgpu::exec::Interpreter::dispatch` |
| 13 | 1.5% | `libcudart vgpu::MemoryManager::load_scalar` |
| 14 | 1.4% | `ld-linux-x86-64 __tls_get_addr` |
| 15 | 1.3% | `libcudart vgpu::load_at` |
| 16 | 1.0% | `libcudart vgpu::MemoryManager::store_scalar` |
| 17 | 1.0% | `libcudart __tls_get_addr@plt` |
| 18 | 1.0% | `libcudart vgpu::exec::fetch_texel` |
| 19 | 0.9% | `libcudart vgpu::exec::tex_finish` |
| 20 | 0.9% | `libc 0x0000000000188dd9` |
| 21 | 0.7% | `libcudart vgpu::exec::Interpreter::fast_cvt_int` |
| 22 | 0.7% | `libcudart vgpu::exec::Interpreter::shared_at` |
| 23 | 0.6% | `libcudart vgpu::store_at` |
| 24 | 0.5% | `libcudart vgpu::exec::Interpreter::exec_tex` |
| 25 | 0.5% | `libcudart vgpu::exec::tex_footprint_linear` |

Per workload, top 5: **mem** (2641 samples): 22% `vgpu::MemoryManager::resolve`, 15% `vgpu::BigReaderLock::unlock_shared`, 11% `vgpu::exec::Interpreter::exec_ld`, 11% `vgpu::exec::Interpreter::exec_st`, 5% `vgpu::MemoryManager::scalar_location`; **shared** (1729 samples): 37% `vgpu::exec::Interpreter::exec_ld`, 33% `vgpu::exec::Interpreter::exec_st`, 5% `vgpu::MemoryManager::resolve`, 4% `vgpu::BigReaderLock::unlock_shared`, 3% `vgpu::exec::Interpreter::shared_at`; **atomics** (417 samples): 24% `vgpu::MemoryManager::resolve`, 19% `vgpu::BigReaderLock::unlock_shared`, 7% `vgpu::exec::Interpreter::exec_atom`, 5% `vgpu::MemoryManager::load_scalar`, 5% `vgpu::MemoryManager::scalar_location`; **texture** (518 samples): 17% `vgpu::exec::fetch_texel`, 16% `vgpu::exec::tex_finish`, 9% `vgpu::exec::Interpreter::exec_tex`, 9% `vgpu::exec::tex_footprint_linear`, 5% `vgpu::BigReaderLock::unlock_shared`; **local** (3499 samples): 34% `vgpu::exec::Interpreter::exec_ld`, 17% `vgpu::exec::Interpreter::exec_st`, 10% `vgpu::exec::Interpreter::run_warp_until_yield`, 7% `vgpu::exec::Interpreter::fast_int_bin`, 6% `vgpu::exec::Interpreter::check_local`

#### PTX engine, control group: 3040 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 30.2% | `libcudart vgpu::exec::Interpreter::dispatch` |
| 2 | 27.1% | `libcudart vgpu::exec::Interpreter::run_warp_until_yield` |
| 3 | 8.8% | `libcudart vgpu::exec::Interpreter::compare` |
| 4 | 6.1% | `libcudart vgpu::exec::Interpreter::read_operand` |
| 5 | 6.0% | `libcudart vgpu::exec::Interpreter::fast_int_bin` |
| 6 | 4.9% | `libcudart vgpu::exec::Interpreter::int_bin` |
| 7 | 3.7% | `libcudart vgpu::exec::Interpreter::exec_shfl` |
| 8 | 2.1% | `libcudart vgpu::exec::Interpreter::cmp_ftz` |
| 9 | 1.9% | `libcudart vgpu::exec::Interpreter::write_reg` |
| 10 | 1.4% | `libcudart vgpu::exec::Interpreter::read_narrow` |
| 11 | 1.3% | `libcudart auto vgpu::exec::Interpreter::fast_setp::{lambda#2}::operator<vgpu::exec::Interpreter::fast_setp::{lambda#1}>` |
| 12 | 1.1% | `libcudart vgpu::exec::Interpreter::read_pred` |
| 13 | 0.9% | `libcudart auto vgpu::exec::Interpreter::fast_setp::{lambda#2}::operator<vgpu::exec::Interpreter::fast_setp::{lambda#2}>` |
| 14 | 0.5% | `libgcc_s __popcountdi2` |
| 15 | 0.5% | `bench_kernels.13.0.88.fbfe9ef9fc28316f w_divergent` |
| 16 | 0.4% | `libcudart std::vector<vgpu::exec::Path, std::allocator<vgpu::exec::Path> >::push_back` |
| 17 | 0.4% | `libcudart vgpu::exec::Interpreter::fast_cvt_int` |
| 18 | 0.3% | `libcudart std::vector<unsigned long, std::allocator<unsigned long> >::~vector` |
| 19 | 0.3% | `bench_kernels.13.0.88.fbfe9ef9fc28316f w_warp` |
| 20 | 0.2% | `libcudart vgpu::exec::Interpreter::require_warp32` |
| 21 | 0.2% | `libcudart vgpu::ptx::Parser::intern` |
| 22 | 0.2% | `libcudart __popcountdi2@plt` |
| 23 | 0.2% | `libcudart vgpu::exec::Interpreter::sync_mask_groups` |
| 24 | 0.1% | `libc 0x000000000018891d` |
| 25 | 0.1% | `libcudart vgpu::exec::Interpreter::fast_path` |

Per workload, top 5: **divergent** (2460 samples): 35% `vgpu::exec::Interpreter::dispatch`, 28% `vgpu::exec::Interpreter::run_warp_until_yield`, 11% `vgpu::exec::Interpreter::compare`, 6% `vgpu::exec::Interpreter::int_bin`, 6% `vgpu::exec::Interpreter::read_operand`; **warp** (580 samples): 25% `vgpu::exec::Interpreter::run_warp_until_yield`, 19% `vgpu::exec::Interpreter::exec_shfl`, 12% `vgpu::exec::Interpreter::fast_int_bin`, 10% `vgpu::exec::Interpreter::dispatch`, 8% `vgpu::exec::Interpreter::read_operand`

#### PTX engine, api group: 1081 samples

| # | share | function |
| ---: | ---: | --- |
| 1 | 9.0% | `libc 0x000000000018984a` |
| 2 | 6.6% | `[vdso] __vdso_clock_gettime` |
| 3 | 6.2% | `libcudart vgpu::MemoryManager::resolve` |
| 4 | 5.0% | `libcudart vgpu::profiling::ApiCall::ApiCall` |
| 5 | 4.0% | `libcudart vgpu::exec::Interpreter::exec_st` |
| 6 | 4.0% | `libcudart vgpu::BigReaderLock::unlock_shared` |
| 7 | 3.4% | `libcudart vgpu::exec::Interpreter::setup_block` |
| 8 | 2.6% | `libcudart vgpu::exec::Interpreter::exec_ld` |
| 9 | 2.5% | `ld-linux-x86-64 __tls_get_addr` |
| 10 | 2.2% | `libcudart vgpu::MemoryManager::materialize` |
| 11 | 2.1% | `libc getenv` |
| 12 | 1.9% | `libcudart vgpu::exec::Interpreter::run_warp_until_yield` |
| 13 | 1.7% | `libcudart vgpu::BigReaderLock::lock` |
| 14 | 1.6% | `libc 0x0000000000189087` |
| 15 | 1.3% | `libc cfree` |
| 16 | 1.3% | `libcudart vgpu::MemoryManager::free` |
| 17 | 1.2% | `libcudart vgpu::MemoryManager::scalar_location` |
| 18 | 1.2% | `libstdc++ std::_Rb_tree_insert_and_rebalance` |
| 19 | 1.2% | `libc malloc` |
| 20 | 1.2% | `libcudart __tls_get_addr@plt` |
| 21 | 1.1% | `libcudart vgpu::runtime::Device::launch` |
| 22 | 1.1% | `libc pthread_mutex_lock` |
| 23 | 1.0% | `libcudart vgpu::telemetry::Publisher::refresh` |
| 24 | 0.9% | `libcudart vgpu::MemoryManager::load_scalar` |
| 25 | 0.7% | `libcudart vgpu::exec::Interpreter::read_operand` |

Per workload, top 5: **launch** (554 samples): 9% `vgpu::MemoryManager::resolve`, 8% `vgpu::exec::Interpreter::exec_st`, 7% `vgpu::BigReaderLock::unlock_shared`, 7% `vgpu::exec::Interpreter::setup_block`, 5% `vgpu::exec::Interpreter::exec_ld`; **api** (501 samples): 19% `libc 0x000000000018984a`, 11% `[vdso] __vdso_clock_gettime`, 10% `vgpu::profiling::ApiCall::ApiCall`, 5% `vgpu::MemoryManager::materialize`, 4% `ld-linux-x86-64 __tls_get_addr`; **startup** (26 samples): 8% `memcmp@plt`, 4% `libc __isoc23_strtol`, 4% `libc malloc`, 4% `libc 0x00000000000a9bc9`, 4% `libc 0x00000000000a9dc2`


### By source line

Top 15 lines per engine and group (`--by-line`). Paths are the file's basename;
`src/sass/exec_ops.inc` is `#include`d into `src/sass/exec.cpp`, so its lines
carry the inlined helpers' names for the lines they came from (`stl_vector.h`,
`atomic_base.h`, ...).

#### SASS engine, alu group: 8047 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 14.8% | `libm[327f9]` |
| 2 | 4.1% | `exec.cpp:107` |
| 3 | 3.5% | `exec_ops.inc:1275` |
| 4 | 3.1% | `basic_string.tcc:252` |
| 5 | 3.0% | `exec_ops.inc:1280` |
| 6 | 3.0% | `exec.cpp:582` |
| 7 | 2.8% | `exec.cpp:566` |
| 8 | 2.7% | `exec.cpp:591` |
| 9 | 2.4% | `exec_ops.inc:1276` |
| 10 | 2.3% | `exec_ops.inc:1163` |
| 11 | 2.1% | `exec_ops.inc:1248` |
| 12 | 2.1% | `exec.cpp:562` |
| 13 | 2.0% | `exec_ops.inc:1252` |
| 14 | 1.9% | `exec.cpp:563` |
| 15 | 1.9% | `exec.cpp:108` |

Per workload, top 5: **int_alu** (1787 samples): 10% `exec_ops.inc:1163`, 6% `exec_ops.inc:1143`, 3% `exec_ops.inc:1218`, 3% `exec_ops.inc:1070`, 3% `exec_ops.inc:1222`; **f32_fma** (985 samples): 17% `exec_ops.inc:1248`, 16% `exec_ops.inc:1252`, 13% `fmaf@plt+0`, 4% `exec_ops.inc:1059`, 4% `exec.cpp:107`; **f64_fma** (2991 samples): 7% `exec_ops.inc:1275`, 6% `exec.cpp:107`, 6% `basic_string.tcc:252`, 6% `exec.cpp:566`, 5% `exec_ops.inc:1280`; **f64_subnormal** (2284 samples): 48% `libm[327f9]`, 4% `exec.cpp:582`, 4% `exec.cpp:107`, 4% `exec_ops.inc:1280`, 3% `basic_string.tcc:252`

#### SASS engine, tensor group: 3590 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 23.9% | `exec_ops.inc:3009` |
| 2 | 11.3% | `exec_ops.inc:3012` |
| 3 | 6.0% | `exec_ops.inc:3011` |
| 4 | 4.8% | `exec_ops.inc:3010` |
| 5 | 3.2% | `exec_ops.inc:3014` |
| 6 | 2.9% | `exec_ops.inc:2943` |
| 7 | 2.7% | `__extendhfdf2@plt+0` |
| 8 | 2.1% | `__extendhfdf2+28` |
| 9 | 2.1% | `__extendhfdf2+13` |
| 10 | 2.0% | `__extendhfdf2+17` |
| 11 | 2.0% | `exec_ops.inc:2950` |
| 12 | 1.8% | `__extendhfdf2+39` |
| 13 | 1.7% | `string_fortified.h:29` |
| 14 | 1.6% | `exec_ops.inc:2361` |
| 15 | 1.4% | `exec_ops.inc:3000` |

Per workload, top 5: **wmma** (1942 samples): 26% `exec_ops.inc:3009`, 12% `exec_ops.inc:3012`, 6% `exec_ops.inc:3011`, 4% `exec_ops.inc:3014`, 4% `exec_ops.inc:3010`; **mma_ptx** (1648 samples): 21% `exec_ops.inc:3009`, 11% `exec_ops.inc:3012`, 6% `exec_ops.inc:3010`, 5% `exec_ops.inc:3011`, 4% `exec_ops.inc:2943`

#### SASS engine, memory group: 9477 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 5.2% | `atomic_base.h:505` |
| 2 | 4.5% | `atomic_base.h:645` |
| 3 | 2.3% | `exec_ops.inc:1788` |
| 4 | 1.9% | `stl_iterator.h:1125` |
| 5 | 1.7% | `exec.cpp:336` |
| 6 | 1.3% | `memory.cpp:42` |
| 7 | 1.3% | `exec_ops.inc:1846` |
| 8 | 1.3% | `stl_algo.h:1801` |
| 9 | 1.1% | `exec.cpp:766` |
| 10 | 1.1% | `exec_ops.inc:1143` |
| 11 | 1.1% | `stl_vector.h:1148` |
| 12 | 1.0% | `stl_vector.h:993` |
| 13 | 1.0% | `memory.cpp:1170` |
| 14 | 1.0% | `exec_ops.inc:1163` |
| 15 | 1.0% | `exec.cpp:547` |

Per workload, top 5: **mem** (4711 samples): 8% `atomic_base.h:505`, 7% `atomic_base.h:645`, 3% `exec.cpp:336`, 3% `memory.cpp:42`, 2% `stl_vector.h:993`; **shared** (1248 samples): 4% `atomic_base.h:505`, 3% `exec_ops.inc:1788`, 3% `atomic_base.h:645`, 2% `exec_ops.inc:1846`, 2% `stl_iterator.h:1125`; **atomics** (431 samples): 10% `atomic_base.h:645`, 7% `atomic_base.h:505`, 2% `__tls_get_addr@plt+0`, 2% `__tls_get_addr+28`, 2% `exec_ops.inc:2193`; **texture** (488 samples): 5% `atomic_base.h:505`, 3% `interpreter.cpp:1867`, 3% `interpreter.cpp:1581`, 3% `interpreter.cpp:1580`, 3% `exec_ops.inc:2526`; **local** (2599 samples): 5% `stl_iterator.h:1125`, 4% `stl_algo.h:1801`, 4% `exec_ops.inc:1788`, 3% `exec_ops.inc:1163`, 2% `exec_ops.inc:1143`

#### SASS engine, control group: 5273 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 8.4% | `exec_ops.inc:1097` |
| 2 | 6.4% | `exec.cpp:1234` |
| 3 | 3.7% | `exec_ops.inc:1163` |
| 4 | 3.6% | `exec.cpp:1230` |
| 5 | 3.1% | `exec.cpp:1227` |
| 6 | 2.8% | `exec_ops.inc:10` |
| 7 | 2.7% | `exec.cpp:1307` |
| 8 | 2.4% | `exec_ops.inc:1169` |
| 9 | 2.3% | `exec.cpp:547` |
| 10 | 2.2% | `exec.cpp:335` |
| 11 | 2.2% | `stl_vector.h:1148` |
| 12 | 2.1% | `exec_ops.inc:1096` |
| 13 | 2.1% | `bit:261` |
| 14 | 1.7% | `exec.cpp:336` |
| 15 | 1.6% | `exec.cpp:1315` |

Per workload, top 5: **divergent** (3836 samples): 11% `exec_ops.inc:1097`, 8% `exec.cpp:1234`, 4% `exec.cpp:1230`, 4% `exec.cpp:1227`, 3% `exec_ops.inc:1163`; **warp** (1437 samples): 5% `exec_ops.inc:1163`, 4% `exec_ops.inc:10`, 4% `exec.cpp:1307`, 4% `exec.cpp:612`, 3% `exec.cpp:1234`

#### SASS engine, api group: 1231 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 9.2% | `libc[18984a]` |
| 2 | 4.0% | `__vdso_clock_gettime+85` |
| 3 | 3.3% | `atomic_base.h:505` |
| 4 | 3.2% | `atomic_base.h:645` |
| 5 | 2.5% | `profiling.cpp:264` |
| 6 | 2.4% | `atomic_base.h:895` |
| 7 | 1.9% | `stl_tree.h:790` |
| 8 | 1.4% | `__tls_get_addr@plt+0` |
| 9 | 1.2% | `__tls_get_addr+25` |
| 10 | 1.0% | `exec_ops.inc:3056` |
| 11 | 0.9% | `exec_ops.inc:3057` |
| 12 | 0.9% | `libc[189087]` |
| 13 | 0.8% | `exec.cpp:766` |
| 14 | 0.8% | `brlock.hpp:51` |
| 15 | 0.7% | `exec_ops.inc:10` |

Per workload, top 5: **launch** (642 samples): 5% `atomic_base.h:645`, 5% `atomic_base.h:505`, 2% `exec_ops.inc:3056`, 2% `exec_ops.inc:3057`, 2% `exec.cpp:766`; **api** (584 samples): 19% `libc[18984a]`, 8% `__vdso_clock_gettime+85`, 5% `atomic_base.h:895`, 5% `profiling.cpp:264`, 4% `stl_tree.h:790`; **startup** (5 samples): 20% `:0`, 20% `decode.cpp:362`, 20% `ld-linux-x86-64[15361]`, 20% `operator new+0`, 20% `std::basic_ostream<wchar_t, std::char_traits<w`

#### PTX engine, alu group: 2314 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 15.4% | `string_fortified.h:29` |
| 2 | 5.7% | `interpreter.cpp:904` |
| 3 | 4.7% | `interpreter.cpp:4503` |
| 4 | 2.7% | `stl_vector.h:993` |
| 5 | 2.1% | `interpreter.cpp:6209` |
| 6 | 1.9% | `memory.hpp:227` |
| 7 | 1.8% | `stl_vector.h:1129` |
| 8 | 1.7% | `interpreter.cpp:6283` |
| 9 | 1.5% | `interpreter.cpp:2874` |
| 10 | 1.5% | `interpreter.cpp:3350` |
| 11 | 1.5% | `interpreter.cpp:4366` |
| 12 | 1.4% | `interpreter.cpp:3265` |
| 13 | 1.4% | `interpreter.cpp:4392` |
| 14 | 1.3% | `interpreter.cpp:2827` |
| 15 | 1.2% | `interpreter.cpp:3241` |

Per workload, top 5: **int_alu** (883 samples): 5% `interpreter.cpp:6209`, 5% `interpreter.cpp:904`, 4% `interpreter.cpp:6283`, 4% `interpreter.cpp:4392`, 3% `stl_vector.h:993`; **f32_fma** (682 samples): 16% `interpreter.cpp:4503`, 11% `interpreter.cpp:904`, 4% `string_fortified.h:29`, 4% `interpreter.cpp:3265`, 4% `memory.hpp:227`; **f64_fma** (384 samples): 9% `string_fortified.h:29`, 5% `interpreter.cpp:3256`, 4% `interpreter.cpp:939`, 4% `memory.hpp:227`, 4% `stl_vector.h:993`; **f64_subnormal** (365 samples): 81% `string_fortified.h:29`, 2% `stl_vector.h:993`, 1% `interpreter.cpp:939`, 1% `interpreter.cpp:3350`, 1% `interpreter.cpp:904`

#### PTX engine, tensor group: 1016 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 45.2% | `interpreter.cpp:7634` |
| 2 | 3.1% | `interpreter.cpp:7382` |
| 3 | 3.0% | `interpreter.cpp:628` |
| 4 | 2.8% | `interpreter.cpp:7488` |
| 5 | 2.8% | `interpreter.cpp:7635` |
| 6 | 2.6% | `interpreter.cpp:7381` |
| 7 | 2.5% | `interpreter.cpp:1051` |
| 8 | 2.4% | `interpreter.cpp:1030` |
| 9 | 1.9% | `interpreter.cpp:7489` |
| 10 | 1.9% | `interpreter.cpp:7510` |
| 11 | 1.8% | `interpreter.cpp:7633` |
| 12 | 1.5% | `interpreter.cpp:7498` |
| 13 | 1.4% | `interpreter.cpp:7658` |
| 14 | 1.0% | `interpreter.cpp:7484` |
| 15 | 1.0% | `interpreter.cpp:9317` |

Per workload, top 5: **wmma** (121 samples): 21% `interpreter.cpp:1051`, 20% `interpreter.cpp:1030`, 7% `interpreter.cpp:1050`, 6% `interpreter.cpp:8536`, 4% `stl_vector.h:1129`; **mma_ptx** (895 samples): 51% `interpreter.cpp:7634`, 4% `interpreter.cpp:7382`, 3% `interpreter.cpp:628`, 3% `interpreter.cpp:7488`, 3% `interpreter.cpp:7635`

#### PTX engine, memory group: 7637 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 6.8% | `atomic_base.h:505` |
| 2 | 5.7% | `atomic_base.h:645` |
| 3 | 3.7% | `interpreter.cpp:3417` |
| 4 | 3.5% | `interpreter.cpp:3376` |
| 5 | 2.5% | `interpreter.cpp:3380` |
| 6 | 2.4% | `interpreter.cpp:3416` |
| 7 | 2.3% | `interpreter.cpp:3382` |
| 8 | 1.7% | `memory.cpp:42` |
| 9 | 1.7% | `interpreter.cpp:3408` |
| 10 | 1.4% | `interpreter.cpp:975` |
| 11 | 1.2% | `__tls_get_addr@plt+0` |
| 12 | 1.2% | `interpreter.cpp:9796` |
| 13 | 1.1% | `interpreter.cpp:3381` |
| 14 | 1.1% | `brlock.hpp:69` |
| 15 | 1.0% | `interpreter.cpp:9726` |

Per workload, top 5: **mem** (3256 samples): 11% `atomic_base.h:505`, 10% `atomic_base.h:645`, 4% `memory.cpp:42`, 2% `interpreter.cpp:3417`, 2% `brlock.hpp:69`; **shared** (1582 samples): 17% `interpreter.cpp:3376`, 12% `interpreter.cpp:3380`, 11% `interpreter.cpp:3382`, 5% `interpreter.cpp:3381`, 4% `interpreter.cpp:3378`; **atomics** (397 samples): 15% `atomic_base.h:505`, 13% `atomic_base.h:645`, 4% `__tls_get_addr@plt+0`, 3% `brlock.hpp:69`, 3% `brlock.hpp:70`; **texture** (455 samples): 6% `atomic_base.h:505`, 4% `atomic_base.h:645`, 4% `interpreter.cpp:1581`, 3% `interpreter.cpp:7140`, 3% `interpreter.cpp:1872`; **local** (1947 samples): 10% `interpreter.cpp:3417`, 8% `interpreter.cpp:3416`, 5% `interpreter.cpp:975`, 3% `interpreter.cpp:3408`, 3% `interpreter.cpp:9796`

#### PTX engine, control group: 2960 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 7.1% | `interpreter.cpp:4784` |
| 2 | 5.3% | `interpreter.cpp:4782` |
| 3 | 3.0% | `interpreter.cpp:904` |
| 4 | 2.9% | `interpreter.cpp:5022` |
| 5 | 2.8% | `interpreter.cpp:2737` |
| 6 | 2.6% | `interpreter.cpp:3241` |
| 7 | 2.6% | `stl_vector.h:993` |
| 8 | 2.3% | `interpreter.cpp:4785` |
| 9 | 2.3% | `stl_vector.h:1129` |
| 10 | 1.9% | `interpreter.cpp:4718` |
| 11 | 1.8% | `variant:1627` |
| 12 | 1.7% | `interpreter.cpp:4783` |
| 13 | 1.7% | `interpreter.cpp:9048` |
| 14 | 1.7% | `interpreter.cpp:9246` |
| 15 | 1.6% | `interpreter.cpp:6283` |

Per workload, top 5: **divergent** (2300 samples): 9% `interpreter.cpp:4784`, 7% `interpreter.cpp:4782`, 4% `interpreter.cpp:2737`, 3% `interpreter.cpp:5022`, 3% `interpreter.cpp:4785`; **warp** (660 samples): 5% `interpreter.cpp:904`, 5% `<artificial>:6738`, 4% `interpreter.cpp:3241`, 3% `interpreter.cpp:6209`, 2% `<artificial>:6739`

#### PTX engine, api group: 1128 samples

| # | share | source line |
| ---: | ---: | --- |
| 1 | 8.1% | `libc[18984a]` |
| 2 | 5.0% | `__vdso_clock_gettime+85` |
| 3 | 4.4% | `atomic_base.h:505` |
| 4 | 2.7% | `atomic_base.h:645` |
| 5 | 2.7% | `atomic_base.h:895` |
| 6 | 2.6% | `profiling.cpp:264` |
| 7 | 2.3% | `interpreter.cpp:2623` |
| 8 | 1.9% | `getenv+72` |
| 9 | 1.6% | `__tls_get_addr+25` |
| 10 | 1.5% | `libc[189087]` |
| 11 | 1.3% | `stl_tree.h:790` |
| 12 | 1.2% | `interpreter.cpp:9726` |
| 13 | 0.8% | `telemetry.cpp:103` |
| 14 | 0.8% | `telemetry.cpp:109` |
| 15 | 0.7% | `interpreter.cpp:3416` |

Per workload, top 5: **launch** (606 samples): 6% `atomic_base.h:505`, 4% `interpreter.cpp:2623`, 4% `atomic_base.h:645`, 3% `__vdso_clock_gettime+85`, 3% `getenv+72`; **api** (500 samples): 18% `libc[18984a]`, 7% `__vdso_clock_gettime+85`, 6% `atomic_base.h:895`, 5% `profiling.cpp:264`, 3% `libc[189087]`; **startup** (22 samples): 5% `ast.hpp:114`, 5% `ast.hpp:26`, 5% `basic_string.h:1072`, 5% `char_traits.h:399`, 5% `hashtable.h:1964`


## Round 8: the SASS engine's operations

A = a fresh Release build (CUDA 13.0 toolkit, GCC 12, LTO) of origin/main at 61d827c5 (source-identical to 5ea3dc04: the commits
between touch only a workflow, a lint script and CMakeLists), B = this branch built the same way. Both pinned to CPUs 6-7,
`tools/bench/bench.sh --ab A B --engines sass,ptx --runs 9`, one host thread, alternating A B A B, `bench_counters` identical in every
pair. The machine was shared and loaded (load average 12-14 while it ran): a ratio within about 5% of 1 is noise (see "Noise"); the
SASS rows that moved by 1.1x or more are well outside it. The PTX rows are a control: this branch does not touch the PTX interpreter, only
`include/vgpu/simd` (a `lop3` kernel the PTX engine does not call, and appended table entries), and they read 0.88x-1.08x, all "same".

| workload | engine | metric | main (median) | this branch (median) | median ratio | best ratio | p | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| int_alu | sass | winst_per_cpu_s | 22.642 | 42.526 | 1.878x | 1.777x | 0.000 | faster |
| int_alu | ptx | winst_per_cpu_s | 43.331 | 43.276 | 0.999x | 1.045x | 0.863 | same |
| f32_fma | sass | winst_per_cpu_s | 43.815 | 42.819 | 0.977x | 1.059x | 0.796 | same |
| f32_fma | ptx | winst_per_cpu_s | 62.617 | 55.341 | 0.884x | 1.024x | 0.258 | same |
| f64_fma | sass | winst_per_cpu_s | 19.487 | 34.422 | 1.766x | 1.801x | 0.004 | faster |
| f64_fma | ptx | winst_per_cpu_s | 32.689 | 31.387 | 0.960x | 0.959x | 0.489 | same |
| f64_subnormal | sass | winst_per_cpu_s | 5.756 | 6.564 | 1.140x | 1.122x | 0.003 | faster |
| f64_subnormal | ptx | winst_per_cpu_s | 6.455 | 6.682 | 1.035x | 1.024x | 0.730 | same |
| wmma | sass | winst_per_cpu_s | 4.790 | 4.922 | 1.027x | 1.012x | 0.666 | same |
| wmma | ptx | winst_per_cpu_s | 3.753 | 4.051 | 1.079x | 0.941x | 0.387 | same |
| mma_ptx | sass | winst_per_cpu_s | 8.949 | 9.484 | 1.060x | 1.094x | 0.019 | faster |
| mma_ptx | ptx | winst_per_cpu_s | 0.490 | 0.476 | 0.972x | 1.032x | 1.000 | same |
| mem | sass | winst_per_cpu_s | 2.488 | 7.482 | 3.007x | 2.457x | 0.000 | faster |
| mem | ptx | winst_per_cpu_s | 8.017 | 8.517 | 1.062x | 0.989x | 0.546 | same |
| shared | sass | winst_per_cpu_s | 8.974 | 11.686 | 1.302x | 1.313x | 0.000 | faster |
| shared | ptx | winst_per_cpu_s | 8.184 | 8.203 | 1.002x | 0.977x | 0.931 | same |
| atomics | sass | winst_per_cpu_s | 2.792 | 7.009 | 2.510x | 2.499x | 0.000 | faster |
| atomics | ptx | winst_per_cpu_s | 2.303 | 2.311 | 1.004x | 0.981x | 0.387 | same |
| texture | sass | winst_per_cpu_s | 3.045 | 3.376 | 1.109x | 1.144x | 0.014 | faster |
| texture | ptx | winst_per_cpu_s | 1.643 | 1.780 | 1.084x | 1.021x | 0.190 | same |
| local | sass | winst_per_cpu_s | 10.985 | 14.285 | 1.300x | 1.263x | 0.000 | faster |
| local | ptx | winst_per_cpu_s | 10.875 | 10.977 | 1.009x | 1.030x | 0.436 | same |
| divergent | sass | winst_per_cpu_s | 9.857 | 17.568 | 1.782x | 1.845x | 0.000 | faster |
| divergent | ptx | winst_per_cpu_s | 14.135 | 13.673 | 0.967x | 0.964x | 0.297 | same |
| warp | sass | winst_per_cpu_s | 10.852 | 19.842 | 1.828x | 1.592x | 0.000 | faster |
| warp | ptx | winst_per_cpu_s | 24.791 | 26.714 | 1.078x | 1.021x | 0.297 | same |
| launch | sass | units_per_s | 285.356 | 401.163 | 1.406x | 1.373x | 0.000 | faster |
| launch | ptx | units_per_s | 316.935 | 304.565 | 0.961x | 0.985x | 0.489 | same |
| api | sass | units_per_s | 4735.795 | 4560.842 | 0.963x | 0.983x | 0.258 | same |
| api | ptx | units_per_s | 4375.708 | 4535.902 | 1.037x | 1.020x | 0.931 | same |
| startup | sass | wall_s | 0.004 | 0.006 | 0.694x | 0.815x | 0.040 | slower |
| startup | ptx | wall_s | 0.016 | 0.016 | 0.998x | 0.977x | 0.931 | same |

`startup` read 0.694x once (4 ms against 6 ms of wall time) and 1.035x (p = 0.44) when repeated with 21 runs; `f32_fma` read 0.977x
once and 1.051x (p = 0.014) with 21. Both are noise at those sizes.

**What the profiles said** (`bench.sh profile --engines sass`, and `perf record` of about 70 HeCBench programs run on the simulator):
after round 7 the general path was still doing, one lane at a time through `read32`/`write32`/`pred`, every form `fast_alu` declined: a
64-bit address calculation (IADD3 with a carry out, IADD3.X, IMAD.WIDE, IMAD.X, ISETP.EX, LEA.HI.X: one lambda was 19% of `mem`), every
float form with .FTZ/.SAT or a rounding mode (`exec_float` was 67% of aobench), half-precision forms with a swizzle or an immediate
(115,664 of 120,000 sampled instructions of `ert` are `HFMA2 R8, R8, R7, 2, 2`), MUFU, the conversions, FMNMX, DSETP, PRMT, SHFL/VOTE
(44% of `warp`), and the memory forms that are not a coalesced 1-8 byte access (`resolve` + `unlock_shared` were 20% of `clenergy`, whose
only memory instruction is a broadcast `LDG.E.128`).

1. **The merged tree's three regressions.** DFMA 0.945x of r7-sass alone: `fast_alu` cleared 768 bytes of scratch before every
   DFMA/DADD/DMUL (16% of f64_fma's samples on that line) and `dop_m` unpacked each lane with five conversions. Fixed: scratch not cleared,
   and an AVX2 kernel that unpacks eight lanes with two `unpack` instructions and keeps the rule for a NaN result (look at the result;
   if it is a NaN, apply `dop_nan` to the lane's operands). f64_fma 1.77x of main. Subnormal 0.946x: 74% of the time is the host's
   microcode assist inside the vector FMA, so the loss was the gather around it; now 1.14x of main (the assist itself stays). Atomics
   0.92-0.965x of r7-memory alone: not isolated to one line; the per-lane path does two table lookups and a stripe lock a lane, each through a
   thread-local; replaced by taking a warp's atomics together (item 4), 2.5x of main.
2. **Wider forms over 32 lanes**, as kernels in `include/vgpu/simd` (each with an independent scalar reference in
   `tests/unit/test_simd.cpp` at every level the host has) and as `Runner::fast_wide` / `fast_float` in `src/sass/exec_ops.inc` (kept
   out of `fast_alu` on purpose: adding them to it made the plain forms 3% slower, measured on `local`). IADD3 with carries in and
   out (`iadd3x`), IMAD.WIDE / .HI / .X (`imad_wide`), ISETP.EX, PLOP3, SEL, SHF (`shf`), LEA and LEA.HI.X (from `shf` and `iadd3x`),
   POPC/BREV/FLO (`lane_unary`), PRMT (`prmt0`, pshufb; the other modes in a flat loop), LOP3 by expression for the 19 tables compilers
   use, masked stores as blends (`blend_m`), FFMA/FADD/FMUL with .FTZ, .SAT and a rounding mode (`ffma_x`...; the mode is set once for the
   warp with `fesetround` around the call), FSETP and DSETP (`fcmp_mask`, `dcmp`), FMNMX (`fmnmx`), MUFU.RCP/RSQ/SQRT (`mufu`, in double as
   the executor does; the other functions in a flat libm loop), I2F/I2FP from 8-32 bits (`i2f`), F2I to 32 bits (`f2i`), F2FP.PACK_AB
   (f16, bf16, .RELU), HADD2/HMUL2/HFMA2 with swizzles, |.|, - and split immediates, HADD2.F32, and DADD/DMUL/DFMA under a directed
   rounding mode.
3. **Warp-wide**: SHFL and VOTE read their sources once per instruction and write masks.
4. **Memory**: scattered, gathered and broadcast 1-16 byte global loads and stores through `MemoryManager::load_batch` / `store_batch` (a
   16-byte access as two 8-byte pieces, each one relaxed atomic access as `load_scalar` makes it; every case the batch declines -- a fault
   armed, a host mapping, a misaligned address, a lane outside the allocation -- takes the old loop); coalesced 16-byte runs through
   `load_run`/`store_run` as 8-byte elements and 4-byte runs copied as rows; a warp's atomics on global memory as one step (lane addresses
   grouped, the stripe locks taken in ascending order, each distinct address loaded and stored once, the lanes applied in lane order; shared,
   vector, host-mapped and misaligned atomics take the old loop); local memory moved between the register file and the thread's window;
   the address fields decoded once per instruction instead of per lane (that lambda was 9-13% of `shared`); byte and half extension by
   shifts instead of a `switch` per lane (6.7% of the memory-heavy programs); a uniform `BRA` moves the 32 counters in one loop.

**Same bits.** `VGPU_FASTPATH=0` sends every one of these to the old code (the uniform BRA included: it is gated on the same switch).
`nvidia/tests/e2e/sass_fastpath.cu` runs about 35 kernels over special and random values with lanes on different paths of a guard and
requires the two runs' output *and counters* to be byte for byte the same (354 launches); the same program prints identical output with
`VGPU_SIMD=base` and `=off`, and identical output on this branch and on main (0 lines differ in its 28 output lines). Faults: five processes each make
one lane of a warp reach outside its local window / the allocation (gather, scatter, atomic) / shared memory and require the same error text
with and without the fast paths. Host references in the program cover 64-bit sums, compares, shifts and multiplies, funnel shifts, popc, brev,
clz, mulhi, division by constants, atomics' order-independent final values, LEA forms, broadcast loads. The unit test adds 6 tests with
independent references (`wide_integer_kernels_at_every_level`, `float_forms_at_every_level`, `lane_unary_at_every_level`,
`prmt0_at_every_level`, `dcmp_at_every_level`, `double_lanes_in_place_at_every_level`, plus the existing LOP3 test now covering the expression
tables); it passes under `-fsanitize=address,undefined` (kernels compiled sanitized, AVX2 included).

**Mutation checks** (each run against the unit test or the e2e, then reverted): 25 one-line breakages of the kernels (the signed multiply,
the second carry, a shift's sign fill, a byte permute's sign replication and operand pick, the NaN rule of `dop_m`, each compare's
predicate, the unsigned conversion's top bit, a flush mask, ...) were all caught by `test_simd`; 30 of the executor's wiring (the ISETP.EX
formula, PLOP3's outputs, IMAD.HI's register, SEL's order, an atomics' order, a shuffle's range predicate, a vote's negation, LEA's carry,
a rounding scope, the flags of FFMA.FTZ/SAT, MUFU's source modifiers, ...) by the e2e's hash lines or host checks. Found weak and fixed:
the local-window bounds check (now the fault tests), MUFU source modifiers and I2F's `.H1`/`.B1` selects (forms added to the e2e), HADD2.F32's
multiply and flush branches (not reachable from code ptxas emits for sm_86; removed rather than left untested). One mutation of the 16-bit
store mask in the gather path is equivalent (the memory call truncates).

**Not done, and why.** F2F, FRND, F2I from f64/f16, I2F from 64 bits, HSETP2/HSET2/HMNMX2, bf16 arithmetic, the f32-result forms of
HMUL2/HFMA2 (ptxas of CUDA 13.0.88 emits only HADD2.F32 for sm_86), MUFU.RCP64H, vector and shared atomics: still the general path. The
sector and bank counting (`memcount::distinct_sectors`, `sass_bank_passes`) is 10-17% of the memory-bound programs and is r7-memory's
file. The per-instruction overhead left (`step_warp`, `execute`, operand decoding, `exec_control`'s per-lane BSYNC bookkeeping: 30-45% of an
ALU-bound program) is a matter for the tier. Subnormal operands stay at the host's assist rate. Having the kernels write the destination
register in place (no scratch copy) gave no measurable gain and was dropped. AVX-512 was not available, so there is none.
## The pre-decoded tier (round 8, `VGPU_TIER=1`)

`src/sass/tier.hpp`, `tier.cpp` and `tier_ops.inc` are route (a) of [docs/perf-design/01-whole-kernel-translation.md](perf-design/01-whole-kernel-translation.md):
each instruction of a code section is analysed once, the first time the section runs (`Code::tier_program`, exec.cpp),
into a micro-op: a handler for its form and its operands resolved to register-file rows, constant rows, or the few things
that can only be read when it runs (a bank-0 word, a uniform register, a register with a modifier). `Runner::tier_run`
(tier_ops.inc) takes the place of `Runner::execute` for one instruction at a time inside `step_warp`'s straight-line
loop. It is **not** a second scheduler: the group search, the 256-instruction chunk, the step count, `max_steps`, the
`slice`/`give_way`/`ws_held` rules and the spin counter are the interpreter's, and every instruction without a handler,
or whose preconditions the handler cannot prove (a register past the warp's file, a lane mask forced by BMOV.PQUAD MACTIVE,
a global access that is not the ordinary case), goes to `Runner::execute` / `exec_mem`, which is the existing code.
Divergent warps work: a handler takes the executing lane mask exactly as the interpreter's code does.

Off unless `VGPU_TIER=1`; `VGPU_FASTPATH=0`, `VGPU_SIMD=off` and `VGPU_SASS_TRACE` switch it off even then.
With it unset the engine takes the same path as before (the A/B below).

What has a handler (`tier::Form`): MOV; IADD3 (plain, and with carries in and out); IMAD (plain, with carries, `.WIDE`,
`.HI`); LOP3 (with its predicate); LEA; SHF; ISETP (and `.EX`); FSETP; FSEL; FFMA/FADD/FMUL; DADD/DMUL/DFMA; POPC; I2FP.F32;
SEL; PRMT; IMNMX; PLOP3; S2R; global loads and stores of 1, 2, 4 and 8 bytes (Ampere and later), shared and local loads
and stores of 1 to 16 bytes; and, as the instruction that ends a run, BRA (fixed target, no `.DIV`/`.U.ANY`), BSSY, NOP.
Where `Runner::fast_alu` has a form, the handler has the same conditions and the same SIMD lane kernel
(include/vgpu/simd); the other forms are ports of the general code in `exec_int`/`exec_float` row by row. Not covered, and
interpreted as before: uniform-datapath ops (UIADD3, ...), BSYNC, EXIT, BAR, WARPSYNC, SHFL/VOTE, atomics, texture, MMA,
MUFU, F2I/I2F other than I2FP.F32, half-precision, 16- and 32-byte global accesses, Turing's global address form,
generic LD/ST, LDC, CALL/RET/BRX.

**Measured** with `tools/bench/bench.sh --ab <r7-final build> <this build> --runs 11 --engines sass --env VGPU_TIER=1`,
both builds Release CUDA 13.0 `5fea0b07` + this branch, one thread, `nvidia/rtx3060`, quick size, the machine shared (load 8
falling to 4 over the run). Counters are identical in every row. "Spread" is (max - min) / median of the 11 runs.

| workload | A: r7-final Minst/s (spread) | B: tier Minst/s (spread) | B/A median | B/A best | p | warp instructions in the tier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| int_alu | 27.3 (42%) | 82.9 (49%) | 3.04x | 3.09x | <0.001 | 100.0% |
| f32_fma | 49.5 (5%) | 136.9 (29%) | 2.76x | 2.84x | <0.001 | 99.9% |
| f64_fma | 20.8 (9%) | 44.5 (4%) | 2.14x | 2.14x | <0.001 | 99.6% |
| f64_subnormal | 6.4 (4%) | 7.5 (3%) | 1.18x | 1.18x | <0.001 | 99.9% |
| wmma | 5.6 (4%) | 5.9 (5%) | 1.05x | 1.04x | <0.001 | 49.8% |
| mma_ptx | 10.3 (8%) | 10.9 (4%) | 1.06x | 1.05x | <0.001 | 72.8% |
| mem | 5.7 (46%) | 10.5 (8%) | 1.86x | 1.88x | <0.001 | 91.9% |
| shared | 10.1 (4%) | 14.8 (4%) | 1.47x | 1.48x | <0.001 | 86.5% |
| atomics | 3.2 (2%) | 4.1 (4%) | 1.28x | 1.28x | <0.001 | 64.6% |
| texture | 3.8 (3%) | 4.5 (3%) | 1.19x | 1.18x | <0.001 | 87.6% |
| local | 13.2 (4%) | 22.1 (4%) | 1.68x | 1.69x | <0.001 | 100.0% |
| divergent | 11.6 (2%) | 16.6 (3%) | 1.43x | 1.43x | <0.001 | 90.5% |
| warp | 15.6 (3%) | 23.7 (3%) | 1.52x | 1.52x | <0.001 | 76.2% |
| launch (k launches/s) | 371 (4%) | 508 (4%) | 1.37x | 1.37x | <0.001 | 77.8% |
| api (k calls/s) | 5621 (5%) | 5681 (4%) | 1.01x | 1.00x | 0.40 | no kernel |
| startup (wall) | 0.003 s (10%) | 0.003 s | 1.01x | 1.00x | 0.52 | 77.8% |

The last column is `VGPU_TIER_STATS=1` ("tier: N of M warp instructions ran in the tier"): warp instructions a handler ran
(including BRA, BSSY, NOP, PLOP3, S2R, SEL at the end of a run) over all warp instructions of the quick-size run.
Most of what is left is instructions that end a run on purpose and are rich in state (BSYNC, EXIT, BAR, SHFL/VOTE, HMMA,
atomics, texture). The two workloads that gain least are bound by something the tier does not touch: `f64_subnormal` by
the CPU's subnormal assist, `wmma`/`mma_ptx` by the scalar tensor-core loop (half of their instructions are HMMA).
`VGPU_TIER` unset against r7-final (6 workloads, 9 runs, load 3 to 11): all within the noise, B/A best 0.98 to 1.03.

**How it is held to the interpreter** (all wired in CMake):
* `e2e_sass_tier` (nvidia/tests/e2e/run_sass_tier.sh): `sass_tier.cu`, a program of 185 expressions (launched with block sizes that are not whole warps) over random inputs mixed with
  zeros, extremes, NaN payloads, infinities, subnormals and shift counts past the width (integer, 128-bit and pointer
  arithmetic, f32, f64, branches, global/shared/local memory of every width, signed sub-word loads, generic accesses,
  atomics, shuffles), built for sm_75 to sm_120 and run on each generation's profile; plus the other SASS test
  programs of the architecture sweep (41 listed; CUDA 12.0 skips two). Each runs with `VGPU_TIER=0` and `1` with `VGPU_COUNTERS=1`: every result and every
  counter of every launch must print identically; `sass_tier` also under four host threads, and the tier must have run
  at least 30% of its warp instructions (it cannot pass by doing nothing). `VGPU_TIER_OTHER_ARCHS=1` runs the others on
  every architecture (a soak; 144 runs passed).
* `bench_counters_tier`: `run_bench_check.sh` with `VGPU_TIER=1` (the 16 workloads, counts against tools/bench/expected).
* `test_tier` (tests/unit): the analysis over every instruction of the disassembler corpora (sm_80 to sm_120: 780 to 4,700
  instructions with a handler per corpus): operand rows inside the program, register bounds, the guard flags, Turing left to the interpreter.
* Mutation checks (a handler broken on purpose, `e2e_sass_tier` run on sm_86): FFMA rounded twice, the ISETP comparison table with
  LE and GT swapped, one counter left out of the run loop, the BRA target off by 16, PLOP3's first input inverted,
  IMAD.WIDE's signedness swapped, IADD3's carry-in dropped, the signed byte load not sign-extended: all fail the test.
  **Not caught**: the carry-out of LEA (it needs a pointer whose low word overflows; no kernel here makes one) and the
  second carry-in of IADD3 (ptxas always passes `!PT` there).

**What it would take to make it the default**: (1) a soak over what the tests do not reach: the PyTorch sweep
(`e2e_pytorch_sass_*`), the pantheon workloads and the CUTLASS/CuTe programs with `VGPU_TIER=1`, comparing results the way
`run_sass_tier.sh` does; (2) the two CI-only legs this round could not run: the sanitizer jobs (ASan/UBSan leak checks and
TSan read the tier's shared `Program` built under `std::call_once`; no external-linkage global was added) and the hosted CUDA
12.0 job (the code builds and `e2e_sass_tier` passes with CUDA 12.0.140's nvcc locally); (3) a decision on memory: a MOp is
about 100 bytes per instruction, built for a whole code section when it first runs, on top of the 176-byte `Instr`; (4) the
rest of the run-ending ops (BSYNC, EXIT, BAR, SHFL/VOTE) if `divergent`, `shared` and `warp` are to pass 2x.

## The tier, round 9 (`VGPU_TIER=1`)

Round 8 left the tier opt-in with about half of the instructions of the control-heavy workloads still in the interpreter. This round
widened what has a handler, made the tier's run loop cheaper, tested every behaviour of `Runner::step_warp`'s straight-line loop
that the tier copies, and put the tier in the CI jobs that can soak it. It is still **not** the default.

**What has a handler now** (`src/sass/tier.cpp`, `tier_ops.inc`; each is the SIMD kernel or the scalar rule the interpreter's fast
path or general code uses, with the operands resolved once):
FFMA/FADD/FMUL with a rounding mode, `.FTZ` or `.SAT`; FMNMX; MUFU (RCP, RSQ, SQRT as kernels; SIN, COS, EX2, LG2, TANH through libm as
`fast_float` does them); F2I (f32 to 32 bits); I2F from 8, 16 and 32 bits, also under a directed rounding mode; F2FP.PACK_AB; DSETP;
DADD/DMUL/DFMA also under a directed rounding mode; packed half HADD2/HMUL2/HFMA2 and HADD2.F32; BREV, FLO, IABS; LDC; 16-byte
global loads and stores; constant-bank operands of banks other than 0 (read when the instruction runs: the module's `__constant__`
data lives in memory); a lane-constant SHFL as a table worked out once; and, as instructions that end a run, VOTE, SHFL, HMMA (the SIMD
kernel), CS2R, BSYNC, BAR (SYNC, ARV and the reductions), YIELD, EXIT (when no lane waits at a BSYNC or a WARPSYNC and no block barrier has an
arrival; otherwise the interpreter's), the conditional `BRA Pn` of loops, and the barriers and fences that do nothing here (DEPBAR, MEMBAR,
ERRBAR, NANOSLEEP). The uniform datapath is computed once for the warp from the first lane that runs it, as `exec_int` does: UMOV, ULDC,
UIADD3, UIMAD, UISETP, ULEA, ULOP3, USHF, USEL, UFLO, UPOPC (the forms with a per-lane operand or destination, UIADD3.64 and the 64-bit
and `.EX` compares stay the interpreter's).
Still the interpreter's: global atomics and RED, shared atomics, REDUX, MATCH, VOTEU, TEX/TLD, CALL/RET, BRX, BRA.DIV/.CONV, WARPSYNC, 32-byte
accesses, generic LD/ST, F2F, FRND, FCHK, the double conversions, MUFU.RCP64H/RSQ64H, and Turing's global address form.

**The chain.** In round 8 every run of straight instructions ended in a return to the scheduler. Between two calls of `step_warp` the work
is: release the lanes a WARPSYNC set keeps waiting (`ws_held`), pick the group at the lowest address, honour a slice a YIELD gave and a
give-way, find the code section. When there is no `ws_held`, slice, give-way or forced mask and every runnable lane stands at one
address inside the section just run, that work comes to "group = the runnable lanes, pick = their address", and the next call would
start exactly where the tier's loop already is. The loop goes on there instead of returning (`tier_run` in `tier_ops.inc`; the
conditions are checked by the same state the scheduler reads, not derived), and tells its caller how many calls it took over
(`Warp::extra`), never more than the turn the caller has left (`Warp::budget`): a warp's turn is 4096 calls of `step_warp`
(`run_block`, `run_cluster`) or 256 (a cooperative launch), so the instruction count of a warp that waits for another one's store, which
depends on that, is unchanged. `VGPU_TIER_CHAIN=0` turns the chain off. On the benchmark workloads the tier makes 5,000 to 66,000 runs of which
{{CHAINED}} are chained.

**What `step_warp`'s straight-line loop does besides executing** (the list is at the top of the tier's run loop in `tier_ops.inc`, and each item has a test):
the store gate (`sass_tier_steps.cu`: gate_plain, gate_long, gate_generic, gate_diverged), the step count and its limit
(`steps`: the least `VGPU_MAX_STEPS` a kernel finishes under, which must be the same, and the error text one below it), the 256-instruction
chunk and the end of a section (gate_long, turn_long), the turn of 4096 calls (turn_plain, turn_shfl, turn_atomic, turn_vote, turn_branch,
turn_long: the first iteration at which the loop of one warp reads the store another warp makes at its start, a measure of calls per
iteration), YIELD loops and spin locks (spin_*, spinlock), reconvergence and barrier reductions on blocks that are not whole warps, special
registers, the lanes a guard leaves, and faults named by instruction, lane and address after a long run of straight instructions
(`fault gather|scatter|local|shared|atomic|trap|forever`). `VGPU_MAX_STEPS` now reaches the SASS engine (before, only the PTX engine read it).

**Small costs the profiles showed** (`perf record` of the benchmark workloads, symbolised with `nm`; `perf report` could not): a 128-byte row copy
was a call into libc's memmove (15% of the integer workload), the predicate of LOP3 a scalar loop (8% of `divergent`), the 1/2/4/8-byte
`memcpy` of the shared and local handlers a libc call a lane, the lowest address of a diverged warp a chain of dependent compares
(a tenth of `divergent`) and then a branch per lane, BSYNC's release a scan of the lanes' wait fields, and the shuffles' arithmetic per
lane at every execution. All now vector or table work; the benchmarks below are from after them.

{{TABLE}}

**How it is held to the interpreter.** `e2e_sass_tier` (see round 8) now also runs `sass_tier_steps` (above), the tier without its chain
(`VGPU_TIER_CHAIN=0`) on `sass_tier` and `sass_tier_steps`, and the faults; its statistics go to a file (`VGPU_TIER_STATS_FILE`), because
printed at exit among a program's buffered output they tore a line. **`e2e_sass_archs` runs every one of its programs on every
architecture a third time with `VGPU_TIER=1` and requires the output of the interpreter's run** (`VGPU_SASS_TIER=0` skips it): that is the
tier's soak in the CI job that builds Hopper's and Blackwell's programs (the hosted build-and-test job runs `e2e_sass_tier`, and
`bench_counters_tier`, as before). `test_tier` checks the new forms' analysis over the disassembler corpora.

{{SOAK}}

{{MEMORY}}

{{MUTATIONS}}

## Using this to check a change

1. Build baseline and candidate in separate build directories (same flags).
2. `tools/bench/bench.sh --check-counters --build <candidate>` (or ctest
   `bench_counters`): counts must not move.
3. `tools/bench/bench.sh --ab <baseline> <candidate> --runs 9 --workloads <group>`
   with nothing else of yours running; note `uptime`. Quote median, best and p,
   not one number.
4. Add a same-bits test for what you touched (tests/unit/test_fastpath.cpp is the
   pattern) and show it fails when the change is wrong.

## Not measured

* Host instruction counts and IPC (no PMU on this VM; unknown on hosted runners).
* `VGPU_THREADS` > 1 scaling (harness supports `--threads`; not part of the baseline).
* Other GPU profiles / architectures (the program is compiled for sm_86; a Hopper
  profile runs its PTX, not its SASS).
* The pantheon soak workloads and the PyTorch/cuBLAS paths.
