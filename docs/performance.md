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

**Which NaN the PTX engine returns (r9-correct3).** With a NaN in both operands of an f64 add, sub, mul, min, max or fma, an RTX 3060
returns one of them by the position ptxas gives the operands in the SASS instruction, which it chooses itself: the operand defined later
in the PTX wins, whatever its name, whatever its position in the instruction (probed with PTX loaded through the driver, so that the
text is exactly what ptxas saw; the SASS executor, which runs ptxas's choice, was already right). In SASS terms the NaN of Rb is
returned, else that of Rc, else Ra's (quieted, sign and payload kept), with the commutative operands put in definition order, so for
`fma(a, b, c)` the later multiplicand, then c, then the earlier multiplicand. `mark_nan_order` (src/ptx/contract.cpp, run when a function
is parsed) records the choice in `OpFloatBin::nan_a_first` and `OpFma::nan_swap_ab`, and the PTX engine's general path, its fast paths and
the pre-decoded micro-ops read it. A register's definition order is that of its latest definition before the instruction; a constant
operand counts as latest; a `mov` of a 64-bit register is a copy that has the rank of what it copies; an expression ptxas has already
computed from the same values (when the first result is read later: dead code goes first), and `min` or `max` of a value with itself,
have the earlier result's rank; and `sub.f64 a, <constant>` is `add.f64 a, <the constant negated>`, so a NaN constant that is the answer
comes back with its sign flipped. All of it was checked on the card with 49 hand-written PTX forms (now `f64_nan_pairs_follow_the_operand_order_ptxas_gives`
in test_fastpath, fast and general path, unguarded and under both guards) and about 3,000 random straight-line kernels, in which the only
unmatched kernels were the identities listed above before they were handled. Not reproduced: a loop-carried or branch-dependent
definition order (ranks follow the text), predicated definitions (counted as definitions), and any program whose registers ptxas renames
differently from the text's order of definition; none of the programs tried had one. `nvidia/tests/e2e/round9_probe.cu` (from the r9-verify
agent) prints a hash and the first values of 37 sections, among them every NaN pair of dadd, dmul, dfma, dmin and dmax, and
`run_round9_probe.sh` requires both engines at every SIMD level to equal `nvidia/tests/data/round9_probe.rtx3060.txt`, apart from the
hashes of `f32_lg2` and `f32_sin` (the SFU approximations), and requires `VGPU_TC_EXACT=0` to change only the three wmma lines.

**Not done, and why.** F2F, FRND, F2I from f64/f16, I2F from 64 bits, HSETP2/HSET2/HMNMX2, bf16 arithmetic, the f32-result forms of
HMUL2/HFMA2 (ptxas of CUDA 13.0.88 emits only HADD2.F32 for sm_86), MUFU.RCP64H, vector and shared atomics: still the general path. The
sector and bank counting (`memcount::distinct_sectors`, `sass_bank_passes`) is 10-17% of the memory-bound programs and is r7-memory's
file. The per-instruction overhead left (`step_warp`, `execute`, operand decoding, `exec_control`'s per-lane BSYNC bookkeeping: 30-45% of an
ALU-bound program) is a matter for the tier. Subnormal operands stay at the host's assist rate. Having the kernels write the destination
register in place (no scratch copy) gave no measurable gain and was dropped. AVX-512 was not available, so there is none.
## Round 9: the SASS operations round 8 left (r9-sassops)

**What the profile said first.** Round 8's "not done" list was a list of forms, not of hot spots, so the first step was to count. A
throwaway build (not committed) recorded, for every instruction that reached the general path, how often it did. A static scan of the
504 HeCBench programs built for sm_86 (`cuobjdump -sass`) picked the 240 whose code holds any of the forms; each ran for up to 25 s on one
thread (`VGPU_THREADS=1`, `VGPU_COUNTERS=1` for the denominator), and the 130 that completed a launch in that time are the sample. The result is that none of these forms is a large share of any program: the largest are P2R 2.1% of the warp instructions of
`sort`, I2F.U32.RP 1.0% of `topk`, F2F.F64.F32 0.90% of `myocyte`, IMNMX 0.78% of `segsort`, MUFU.RCP64H 0.47% of `myocyte`, HSET2 0.55% of
`crossEntropy`, ATOMS.CAST.SPIN.64 0.36% of `aop`; the mean over the 130 programs is below 0.04% for every form. The uniform-datapath ops
(ULDC.64 is in 124 of the 130, 0.03% on average, at most 0.34%) are one-lane operations: nothing to widen, and left alone. So the gain
shows in a handful of programs, as below, and nowhere in the bench workloads (which do not use these forms).

**What was done** (`Runner::fast_float` / `fast_wide` and the atomics case of `exec_mem` in `src/sass/exec_ops.inc`; one new kernel in `src/simd`):

| form | how |
| --- | --- |
| I2F, I2FP from 8, 16, 32 and 64 bits, to f16, bf16, f32, f64, in each rounding mode | the existing `i2f` kernel inside one `RoundingScope` for 32-bit sources to f32; a loop over the rows for the rest (the scope around an out-of-line loop, so nothing is hoisted past the mode change) |
| F2I from f16, f32 and f64 to 8, 16, 32 and 64 bits | the `f2i` kernel for f32 to 32 bits as before; a loop over the rows otherwise |
| F2F among f16, bf16, f32, f64 | loops over the rows |
| FRND on f16, f32, f64 | loops over the rows |
| HSET2, HSETP2 (f16 and bf16 pairs, swizzles, \|x\|, -x, immediate pairs) | a loop over the rows with the general rule's element functions |
| HFMA2 on bf16 pairs | new kernel `hfma2_bf_m` (the exact product and sum in binary64, then to binary32, then to bfloat16, as the general rule rounds; the product of two bf16 values is exact in binary64, so no FMA unit is needed: it is the same at both SIMD levels) |
| MUFU.RCP64H, MUFU.RSQ64H | a loop over the rows |
| IMNMX (32 bits), IABS, P2R, FCHK | `ibin_m` (min and max under the two halves of the predicate), loops over the rows, the predicate rows |
| ATOMS (shared atomics: 32 and 64 bits, every operation, CAS.SPIN, POPC.INC) | the lanes' addresses decoded once, then each lane's read-modify-write in lane order straight on the block's shared bytes |

A form whose result the general rule does not promise as a 32-lane loop goes back to it: a 64-bit conversion with an out-of-range lane
(the general rule's clamp is `2^63 - 1` in a double, which is `2^63`, and the cast of that is the host's, `0x8000000000000000`), an address
outside the block's own shared memory or in another block of a cluster, vector atomics, a rounding mode on an f16 or bf16 result of F2F or I2F
(the general rule ignores it in F2F and rounds twice in I2F, which an RTX 3060 does not: see below), and `.FTZ` on a packed compare. The
tier (`VGPU_TIER=1`) hands every one of these to `Runner::execute`, so it gets them too.

**A kernel bug the rounding modes found.** `i2f` for an unsigned 32-bit source computed the value as a double by flipping the sign bit and
adding 2^31 back; under round-down the sum 2^31 + (-2^31) is -0, so `I2F.U32.RM` of 0 gave -0.0. The kernel now adds the low 31 bits and
2^31 times the top bit (both non-negative). Found by the e2e comparison of the first draft; `i2f_under_directed_rounding_at_every_level`
(tests/unit/test_simd.cpp) checks every host rounding mode, signed and unsigned, at every level.

**Measured** with the bench harness's approach: A is a fresh Release build (CUDA 13.0, GCC 12, LTO) of `origin/r8-a` (3ef7a6bd), B this
branch built the same way, `taskset` to CPUs 6-7, one host thread, A B A B ... alternating, 9 runs (user CPU seconds), the machine loaded
(load average 8 to 17: ratios within about 5% of 1 are noise). Counters and results identical in every pair.

| program | A median (s) | B median (s) | A/B median | A/B best | what it does |
| --- | ---: | ---: | ---: | ---: | --- |
| micro `bf16` | 0.93 | 0.14 | 6.6x | 7.4x | HFMA2.BF16_V2 chain |
| micro `i2frp` | 1.03 | 0.20 | 5.2x | 6.3x | I2F with .RP, .RM, .RZ, from 32 bits |
| micro `f2f` | 0.11 | 0.04 | 2.8x | 2.7x | F2F.F64.F32 / F2F.F32.F64 chain |
| micro `f2i64` | 0.55 | 0.26 | 2.1x | 2.0x | F2I.S64.F64 and F2I.U32.F64 |
| micro `i2f64` | 0.28 | 0.14 | 2.0x | 2.1x | I2F from 64 bits |
| micro `shatom` | 0.83 | 0.43 | 1.9x | 2.0x | ATOMS.ADD / ATOMS.MAX, 32 bits |
| micro `frnd` | 0.24 | 0.13 | 1.9x | 1.7x | FRND floor, ceil, trunc, rint |
| micro `hset2` | 0.50 | 0.29 | 1.7x | 2.1x | HSET2 plus HADD2 |
| micro `shatom64` | 0.86 | 0.63 | 1.4x | 1.4x | ATOMS.ADD.64 / ATOMS.CAS.64 |
| micro `imnmx` | 0.27 | 0.24 | 1.13x | 1.20x | IABS, IMNMX |
| micro `rcp64h` | 0.66 | 0.59 | 1.12x | 1.15x | double division and rsqrt |
| HeCBench `aop -runs 1` | 6.84 | 4.96 | 1.38x | 1.26x | ATOMS.CAST.SPIN.64 loops (0.36% of the instructions) |
| HeCBench `heat2d 512 512 20` | 1.60 | 1.35 | 1.19x | 1.23x | IABS (0.29%) |
| HeCBench `is 256 256 256` | 0.60 | 0.50 | 1.20x | 1.22x | I2F.F64, F2I.F64 (0.35%) |
| HeCBench `depixel 512 512 1` | 0.43 | 0.39 | 1.10x | 1.06x | IMNMX.U32 (0.43%) |
| HeCBench `myocyte 100` | 2.34 | 2.35 | 1.00x | 0.97x | F2F.F64.F32 0.9%, RCP64H 0.5%: spread 133%, not resolved |

The micro programs are kernels that loop over the form (written for this round and not committed; the numbers say how much faster one
form gets, not how much of a real program it is). The bench harness (`tools/bench/bench.sh --ab`, 9 runs, SASS engine, all 16 workloads)
reads 0.87x to 1.05x with `same` for every row and identical counters; that spread is the loaded machine, as a repeat of the six
plain-ALU/memory workloads with 21 runs shows (int_alu 1.05x, f32_fma 1.05x, divergent 1.06x, mem 1.04x, local 0.99x, warp 1.07x, p = 0.35 to
0.78). `bench.sh --check-counters` passes.

**Same bits.** `nvidia/tests/e2e/sass_fastpath.cu` gains `conv_forms` (209 forms in four groups of conversions, packed compares, bf16, MUFU,
IABS/IMNMX, FCHK; three input regimes -- special values, in-range values with a NaN, and the same with no NaN or infinity -- under three
lane guards; one hash per launch, one per form with `sass_fastpath forms`, the lanes of one with `sass_fastpath dump <form>`),
`shared_atomic_forms` (the global atomic program's operations on shared memory, with a host reference for the order-independent results),
`pred_press` (22 predicates live across a loop, which ptxas packs with P2R; the host computes the same function) and a `sharedatomic` fault
run (the same error text with and without the fast paths). `run_sass_fastpath.sh` requires fast and general to print identically,
counters included. It also prints identically with `VGPU_SIMD=base` and `=off`, and the same on three repeats. Two outputs of the program
vary from run to run and are not claims about the code: with four host threads the old values returned by the global `atomic_forms`
depend on the order the blocks run in (pre-existing), and the counters are written by the runtime on its own schedule, which is why the
conv hashes are one line per launch.

**Mutation checks** (each made in the executor or the kernel, `sass_fastpath` or `test_simd` run, then reverted): 20 one-line breakages of
the new code in the executor -- an unsigned 64-bit source taken as signed, the rounding mode of I2F and of F2F dropped, the nearest-even
rounding of F2I changed to a floor, the signed clamp off by one, the 64-bit decline removed, F2F to bf16 through the f16 narrowing, FRND
floor and ceil exchanged, the f16 result of FRND widened, HSET2's bf16 1.0 replaced by the full mask, MUFU.RSQ64H computed as a
reciprocal, IMNMX's signedness exchanged, IABS not negating, FCHK's predicate cleared, P2R's bit order exchanged, the shared atomics' counter
left out and their store left out, the f16 source of F2I taken from the high half, |x| and -x dropped from a 64-bit source, HSETP2's two
predicates exchanged -- were all caught by the e2e (the hashes or the counters). Two kernel breakages (the bf16 rounding constant;
the old unsigned `i2f`) were caught by `test_simd`. Not caught, and why: `.FTZ` on HSET2 (an equivalent mutant: the form is now left to the
general rule), HADD2/HMUL2 on bf16 (ptxas lowers a bf16 sum and product to `HFMA2.BF16_V2` for sm_80 and sm_86, so the fast path only
takes HFMA2).

**Where the general rule differed from an RTX 3060, and what was done.** The first run of `sass_fastpath` on a card with CUDA 13.0 found
seven differences in the general rule, which the fast paths copied. All seven are fixed in the general rule, the fast paths and the SIMD
kernel together, and `conv_forms` is now pinned to the card: `nvidia/tests/data/conv_forms.rtx3060.txt` holds the card's hash for each of the
209 forms, in each of 3 input regimes and 3 shapes, and `run_sass_fastpath.sh` requires the fast paths, the general path and every SIMD
level to print it (`run_sass_fastpath.sh --card [--update]` runs the program on a real GPU and compares or rewrites it).
(1) `F2I` of a NaN is 0, except `1 << (bits-1)` when the source is f64 or the destination 64 bits wide (and `0x8000` for f64 to a 16-bit
integer); the |x| and -x modifiers do not apply to a NaN. (2) A NaN through `F2F` and `FRND` is the canonical `0x7fffffff` (f32) or `0x7fff`
(f16, bf16), f64 keeps sign and payload with the quiet bit set, and a NaN narrowed from f64 to f16 keeps its sign and top payload
(`sign | 0x7e00 | payload >> 42`); `0/0` is `0x7fffffff` (the MUFU/FFMA canonical NaN of this round). (3) `F2F` to f16 or bf16 honours
`.RZ`, `.RM`, `.RP`. (4) `I2F` to f16 or bf16 rounds once, from the integer (`narrow_from_int`); the values an f16 can hold are exact in an f32, so
this is not visible on f16, and `I2F` to bf16 does not exist on sm_86 (ptxas converts through f32), but the helper is pinned by
`test_simd` against a table of every f16 and bf16 value in every mode. (5) The decoder now reads the half select (bits 60-61) of an f16 source
for `FRND`, `F2I` and `F2F` (the corpus gained the two `FRND ... R109.H1` lines, which also showed that nvdisasm names a same-size
`FRND.F16` once). (6) `.FTZ` on `HSET2`/`HSETP2` flushes the subnormal halves (f16: exponent field 0; bf16 likewise) before the
comparison. (7) A signed 64-bit conversion at or past 2^63 saturates to `0x7fff...` (and `0x8000...` below -2^63), against the powers of two
themselves (2^63 - 1 is not a double). Found on the way, with the card as the oracle: a bf16 `HFMA2` rounded twice (the fma to f32, then to
bf16) and, with exact ties in the f64 intermediate, once more wrongly; it is now the exact result rounded once (`simd::fma_round_odd`, a
TwoSum-based round-to-odd fma, then `narrow_from_double`), in the kernel and in the general rule.
The PTX interpreter had the same family: `cvt.rXi.ftz.f32.f32` flushed after the rounding (the card flushes the source first), the
round-to-integral f64 forms left a signalling NaN signalling (`0x7ff0000000000001` becomes `0x7ff8000000000001`; the f32 forms were already
canonical, and `.ftz`/`.sat` agree with the card), a 64-bit integer to f32 rounded through a double, and the fused f64 micro-op did not follow the
NaN rule (b's, else c's, else a's, quieted) in its contracted forms, so `VGPU_FASTPATH=1` and `=0` could return different NaNs for an fma with two
signalling NaNs. `float_half_forms` now pins the round-to-integral forms of NaNs (`nan_cvt_rint*`, f32 and f64) to the card.
Remaining PTX-engine differences from the card, none of them a rule the PTX text states: `rsqrt(double)` (the card runs a software sequence from
`MUFU.RSQ64H`; `rsqrt.approx.f64` here is correctly rounded, 1 ulp from the card in about a third of the lanes, and the SASS executor, which
runs the card's sequence, matches); `static_cast<double>(-fabsf(b))` of a NaN (ptxas folds `-|x|` into the F2F; the PTX engine runs `abs` and
`neg` as instructions). (The third gap listed earlier, an fma whose multiplicands ptxas swapped, is closed: see "Which NaN the PTX engine returns".)
**Atomics order.** The old values a warp's lanes get back from an atomic on one address are in lane order on the card
and in both engines (probed with 32 threads, `atomicAdd`, `atomicExch`, two interleaved addresses; identical). Between warps of a block and between blocks the card's order
is its scheduler's: in five of five repeats a global `atomicAdd` / `atomicExch` by 4 blocks of 64 threads gave different old values from the first run in at least one of them, and for shared memory after a
barrier the card's warp order was the same on every repeat but not a function of the warp number (64 threads: warp 1 first; 128: warps
2, 3, 0, 1; 256: 4 to 7, then 0 to 3), which neither engine reproduces. Not deterministic on the card across configurations, so not fixed: a program
that depends on it is racing.

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
(include/vgpu/simd); the other forms are ports of the general code in `exec_int`/`exec_float` row by row. (This is round 8's list; round 9, below, widens it.) Not covered, and
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

**What it would take to make it the default** (round 8's list; the state after round 9 is at the end of the next section): (1) a soak over what the
tests do not reach: the PyTorch sweep (`e2e_pytorch_sass_*`) and the CUTLASS/CuTe programs with `VGPU_TIER=1`, comparing results the way
`run_sass_tier.sh` does; (2) the CI-only legs: the sanitizer jobs (ASan/UBSan leak checks and TSan read the tier's shared `Program` built under
`std::call_once`; no external-linkage global was added); (3) a decision on memory: a MOp is 96 bytes per instruction, built for a whole code
section when it first runs, on top of the 184-byte `Instr`.

## The tier, round 9 (`VGPU_TIER=1`)

Round 8 left the tier opt-in with about half of the instructions of the control-heavy workloads still in the interpreter. This round
widened what has a handler, made the tier's run loop cheaper, tested every behaviour of `Runner::step_warp`'s straight-line loop
that the tier copies, and put the tier in the CI jobs that can soak it. It is still **not** the default.

**What has a handler now** (`src/sass/tier.cpp`, `tier_ops.inc`; each is the SIMD kernel or the scalar rule the interpreter's fast
path or general code uses, with the operands resolved once):
FFMA/FADD/FMUL with a rounding mode, `.FTZ` or `.SAT`; FMNMX; MUFU (RCP, RSQ, SQRT as kernels; SIN, COS, EX2, LG2, TANH by the interpreter's own code); F2I (f32 to 32 bits); I2F from 8, 16 and 32 bits, also under a directed rounding mode; F2FP.PACK_AB (the interpreter's
code); DSETP;
DADD/DMUL/DFMA also under a directed rounding mode; packed half HADD2/HMUL2/HFMA2 (HADD2.F32: the interpreter's code); BREV, FLO, IABS; LDC; 16-byte
global loads and stores; constant-bank operands of banks other than 0 (read when the instruction runs: the module's `__constant__`
data lives in memory); a lane-constant SHFL as a table worked out once; and, as instructions that end a run, VOTE, SHFL, HMMA (the SIMD
kernel, with the kind exec_mma chooses), CS2R, BSYNC, BAR (SYNC, ARV and the reductions), YIELD, EXIT (when no lane waits at a BSYNC or a WARPSYNC and no block barrier has an
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
depends on that, is unchanged. `VGPU_TIER_CHAIN=0` turns the chain off. `VGPU_TIER_STATS=1` prints the count. On the benchmark workloads (full size) the share of runs the chain started is 99% or more for `int_alu`,
`f32_fma`, `f64_fma`, `wmma`, `mma_ptx`, `texture`, `warp` and `local` (`warp`: 1,281,280 of 1,282,048 runs), 96% for `mem`, 85% for
`atomics`, 67% for `launch`, 35% for `divergent` (3.6 million runs: a diverged warp usually ends a run at a branch with more than one address
left) and 25% for `shared` (barriers end its runs).

**What `step_warp`'s straight-line loop does besides executing** (the list is at the top of the tier's run loop in `tier_ops.inc`, and each item has a test):
the store gate (`sass_tier_steps.cu`: gate_plain, gate_long, gate_generic, gate_diverged), the step count and its limit
(`steps`: the least `VGPU_MAX_STEPS` a kernel finishes under, which must be the same, and the error text one below it), the 256-instruction
chunk and the end of a section (gate_long, turn_long), the turn of 4096 calls (turn_plain, turn_shfl, turn_atomic, turn_vote, turn_branch,
turn_long: the first iteration at which the loop of one warp reads the store another warp makes at its start, a measure of calls per
iteration), YIELD loops and spin locks (spin_*, spinlock), reconvergence and barrier reductions on blocks that are not whole warps, special
registers, the lanes a guard leaves, and faults named by instruction, lane and address after a long run of straight instructions
(`fault gather|scatter|local|shared|atomic|trap|forever`). `VGPU_MAX_STEPS` now reaches the SASS engine (before, only the PTX engine read it).

**The tier follows the interpreter's rules (second pass, on the merged tree).** The round-9 correctness work (r9-correct2, card-checked) changed
the interpreter's result rules, and `e2e_sass_tier` then failed on every architecture: the tier's handlers had copied the old rules.
What differed, form by form: DADD/DMUL/DFMA take the NaN of the registers as they are (`simd::dop_nan`) and leave `|x|` and `-x` to the kernel
for every kind of source, constants included (the tier folded the modifiers into a constant's bits first); MUFU's libm functions answer a NaN
with 0x7fffffff and MUFU.TANH does not flush; F2FP.PACK_AB packs every NaN as 0x7fff and `.RELU` turns a `-0` into `+0`; HADD2.F32 of a NaN half is
0x7fffffff; and an sm_86 cubin's HMMA sums as the RTX 3060's tensor core does (`VGPU_TC_EXACT`, vgpu/exec/tc_ampere.hpp). (The kernels the tier shares with the interpreter -- FFMA/FADD/FMUL's `-0` under `.SAT`, FMNMX's two NaNs, MUFU RCP/RSQ/SQRT, the unsigned
I2F -- had been fixed in the kernels and needed nothing; PRMT is a shared function too, and `prmt2` agreed on the merged tree.) The new
`sass_tier` below, run with the tier on and off on the merged tree before this change, disagreed in its `float`, `forms2`, `half2`, `double2` and `mma` programs. The rule since: **a handler computes nothing a number format has rules for; it calls what the interpreter calls.**
FMNMX, FFMA/FADD/FMUL, FSEL, DADD/DMUL/DFMA, DSETP, the packed-half HFMA2 family, MUFU RCP/RSQ/SQRT, F2I, I2F, I2FP, PRMT, IMNMX, BREV/FLO, POPC
and HMMA call the same SIMD kernel or shared function as `fast_alu`/`fast_float`/`fast_wide`/`exec_mma`, with the same arguments (HMMA's choice of
the sum is one function, `Runner::hmma_kernel_kind`, which `exec_mma` calls too). MUFU's libm functions, F2FP.PACK_AB, HADD2.F32 and FSETP with `|x|`
or `-x` run the interpreter's own code (`Runner::tier_float_op`: `fast_alu`, then `exec_float`). What the tier still computes by itself is
integer, predicate and control-flow work with no number-format rules (MOV, LEA, SHF, IMAD.WIDE/.HI, IABS, SHFL, VOTE, the uniform datapath, the addresses of the memory forms, ...).

**The guard.** `sass::tier::form_impl` says for every form whether its result is a kernel call (K), the interpreter's code (I) or the tier's own (O),
with a reason for each O. `test_tier` holds that table to the list checked in at `tests/unit/test_tier.cpp` (adding an O form, or letting a K form stop
sharing, is a change of that file too), and reads `tier_ops.inc`: a K handler must call a SIMD kernel, an I handler the interpreter, the HMMA handler
`hmma_kernel_kind`, and no handler may contain scalar float arithmetic (`std::fabs`, `ftz(`, `static_cast<float>`, `f32_bits(`, `std::sin`,
`float_cmp(`, `dop_nan(`, ...). That is a lint, not a proof (a rule can still be written in integers), but the drift that happened is exactly what it
catches, and it fails at the unit-test level, with no program to run. Checked by breaking each rule it has: a `fabs` in `h_ffma`, `h_fsetp` with no kernel
call, `h_hmma` with no `hmma_kernel_kind`, a `static_cast<float>` in `h_mufu` all fail it.

**The programs.** `sass_tier.cu` now also runs the forms the card fixed, on inputs made of NaNs of both signs with distinct payloads in every
position, zeros of both signs, subnormals and the extremes mixed with random bits: `.SAT`/`.FTZ` of `-0` and directed rounding on FFMA/FADD/FMUL;
FMNMX with two NaNs and `.NAN`; MUFU's NaNs; F2I of every size and rounding (from f32, f64 and the low and high half of an f16);
F2F and FRND among f16, bf16, f32 and f64 in each rounding; I2F to every format from every size; the packed halves (HFMA2, HADD2, HMUL2, `.SAT`,
`.FTZ`, bf16), HSET2/HSETP2 with and without `.FTZ`; F2FP with `.RELU`; DADD/DMUL/DFMA with a NaN in each operand and `|x|`, `-x` on each,
including operands read from constant banks; DSETP with both predicates; VABSDIFF; PRMT with selector bits past the low two; and mma.sync of f16, bf16 and tf32
products. `run_sass_tier.sh` also runs `sass_tier`, `wmma_gemm`, `wmma_types`, `mma_fragment_layout` and `modern_dtypes` on sm_86 with `VGPU_TC_EXACT=0`, so
both sums are held to the interpreter's. Mutation checks (scratch copy, `TMUT=n`): 23 breakages that put the old rules back into the tier (the modifiers applied to the bits,
the direct HMMA kernel call without the tensor-core sum, TANH flushed, a NaN half as 0x7e00, HADD2.F32's NaN, the multiplier and addend of HFMA2
exchanged, FSETP ignoring `-x`, IMNMX, I2F without its rounding mode, FMNMX's flags, F2I's rounding, FFMA's `.SAT`, FADD's and DADD's rounding mode, PRMT's mode, a
bf16 HFMA2 on the f16 kernel, a constant's modifiers folded twice, ...) are all caught by `sass_tier` except one: DSETP's second predicate for a
comparison (the complement), which ptxas never uses (only MIN/MAX's second predicate, which is covered). HMNMX2 (`min.f16x2`, `max.bf16x2`) is not in the
SASS decoder (opcode 0x240: "unknown opcode", the program falls back to PTX), so `sass_tier` leaves it out; that is not the tier's.

**Measured on the merged tree** (a quiet machine, the same method as above; counters identical in every comparison). The tier on against off in this build:
`int_alu` 2.74x, `f32_fma` 2.72x, `f64_fma` 1.58x, `f64_subnormal` 1.08x, `mem` 1.47x, `shared` 1.49x, `atomics` 1.34x, `texture` 1.09x, `local` 1.87x,
`divergent` 1.82x, `warp` 3.60x, `launch` 1.23x, `api` and `startup` the same; `wmma` 1.12x and `mma_ptx` 1.14x with the sm_86 tensor-core sum (which costs
the interpreter about 3.6x on these workloads, so the tier's saving is a smaller share), `wmma` 1.59x and `mma_ptx` 1.62x with `VGPU_TC_EXACT=0` (the
sequential sum). Against the tier as merged before this change (`r9-final` with `VGPU_TIER=1`), 11 alternating runs each: every workload is within 3% (`warp` 1.03x, `mem` 1.03x,
`texture` 0.98x), which is the cost of handing MUFU's libm functions, F2FP.PACK_AB and HADD2.F32 to the interpreter's code; `wmma` and `mma_ptx` read 0.20x
because the merged tier summed sequentially where the card (and now the tier) does not.

**Small costs the profiles showed** (`perf record` of the benchmark workloads, symbolised with `nm`; `perf report` could not): a 128-byte row copy
was a call into libc's memmove (15% of the integer workload), the predicate of LOP3 a scalar loop (8% of `divergent`), the 1/2/4/8-byte
`memcpy` of the shared and local handlers a libc call a lane, the lowest address of a diverged warp a chain of dependent compares
(a tenth of `divergent`) and then a branch per lane, BSYNC's release a scan of the lanes' wait fields, and the shuffles' arithmetic per
lane at every execution. All now vector or table work; the benchmarks below are from after them.

**Measured** (`tools/bench/bench.sh`, one host thread, the RTX 3060 profile, CPU 17 of an i9-12900K shared with other jobs, so
the load average was 6 to 12; the counters of every launch were identical in every comparison). First, the tier on in both builds:
round 8's (origin/r8-a, 3ef7a6bd) against this round's, 11 alternating runs each:

| workload | r8 tier | r9 tier | r9 / r8 |
|---|---:|---:|---:|
| int_alu | 113.7 | 120.8 | 1.06x |
| f32_fma | 126.6 | 136.1 | 1.08x |
| f64_fma, f64_subnormal | 60.1, 7.50 | 57.9, 7.48 | same |
| wmma | 6.43 | 8.80 | 1.37x |
| mma_ptx | 11.7 | 16.7 | 1.42x |
| mem | 17.3 | 19.4 | 1.12x |
| shared | 13.8 | 17.6 | 1.27x |
| atomics, texture | 9.26, 4.31 | 9.72, 4.51 | 1.05x |
| local | 21.4 | 25.1 | 1.17x |
| divergent | 23.7 | 36.4 | 1.54x |
| warp | 37.2 | 77.6 | 2.09x |
| launch (k launches/s) | 461 | 519 | 1.13x |
| api, startup | | | same |

(warp instructions per CPU second, in millions.) Second, one build (this round's) with the tier off and on, three alternating rounds of five
runs, the median of the rounds: `int_alu` 2.7x, `f32_fma` 2.7x, `f64_fma` 1.6x, `wmma` 1.6x, `mma_ptx` 1.6x, `mem` 1.5x, `shared` 1.4x,
`atomics` 1.3x, `local` 1.6x, `divergent` 1.8x, `warp` 3.6x, `launch` 1.26x, `texture` 1.06x, `f64_subnormal` 1.08x (the subnormal
emulation dominates both), `api` and `startup` the same. The tier does not reach 2x on `shared` (barriers), `atomics`
(a global atomic is still the interpreter's: 71% of its warp instructions run in the tier), `texture` or `f64_subnormal`.

**The tier off costs about 2 to 4% on the two shortest workloads.** With `VGPU_TIER=0` against the r8 build, `launch` (0.1 s) measured 0.95x to 0.97x
and `warp` (0.26 s) 0.97x to 0.98x in repeated alternating runs (p below 0.001, identical counters); every other workload was the same. The
interpreter's code did not change except that its loop is now `turn<false>` and `step_warp` reads its step limit from a member. I could not
find a cause by profile (the shares per function look alike), and the figure moves with the layout of the compiled code: marking one tier-only helper
`noinline` made `launch` the same as r8 and `warp` 0.95x, another variant the reverse. It is a few percent on a workload of a tenth of a second and
it is reported rather than chased further.

**How it is held to the interpreter.** `e2e_sass_tier` (see round 8) now also runs `sass_tier_steps` (above), the tier without its chain
(`VGPU_TIER_CHAIN=0`) on `sass_tier` and `sass_tier_steps`, and the faults; its statistics go to a file (`VGPU_TIER_STATS_FILE`), because
printed at exit among a program's buffered output they tore a line. **`e2e_sass_archs` runs every one of its programs on every
architecture a third time with `VGPU_TIER=1` and requires the output of the interpreter's run** (`VGPU_SASS_TIER=0` skips it): that is the
tier's soak in the CI job that builds Hopper's and Blackwell's programs (the hosted build-and-test job runs `e2e_sass_tier`, and
`bench_counters_tier`, as before). `test_tier` checks the new forms' analysis over the disassembler corpora.

**Soak** (the programs run twice on the SASS engine of the RTX 3060 profile, `VGPU_TIER=0` then `1`, one host thread, with the stdout, stderr,
every file the program writes and every launch's counters compared; the scripts are scratch tooling and not in the tree): the HeCBench
programs (the reduced lists), the CUDA Samples v13.0 programs, the Rodinia apps and PolyBench/GPU, 271 programs in the last pass, of which
the tier ran 98.6% of the warp instructions. 38 are byte-identical and 140 identical but for timing lines; 82 run longer than their limit
(30 to 120 s) in both modes and are not compared. 11 were flagged, and every one is a difference between two runs of the same mode: a timing
line the filter missed (`ccsd-trpdrv`, `ising`, `lebesgue`, `lulesh`, `mallocFree`, `fpdc`), the name of a shared-memory segment or a thread
launch order (`simpleIPC`, `streamOrderedAllocationIPC`, `threadMigration`), or a program that reads uninitialised or timer-seeded data
(`fp16ScalarProduct`, `btree`, `particlefilter`, `backprop`; the interpreter differs from itself in `backprop`, as run twice, and in the others as the earlier passes found). No difference in a result
or a counter was caused by the tier. The nine pantheon workloads (`--verify`, one second each) gave the same exit status in both modes; their outputs and counters differ because they run for a time, and the tier gets further in it, so they are not a bit-for-bit check. The soak runs of this round
found one bug in the tier before the programs reached it, the stale register-file pointer of a chained pass after `USETMAXREG` (found
by `e2e_sass_archs`' tier leg on Hopper, fixed in ab7363b5: the register file's address and the registers check are now worked out for every pass).

**Memory.** A `MOp` is 96 bytes, and a code section's program is built whole, under `std::call_once`, the first time a warp runs in it:
the pool rows (128 bytes each; constants, shuffle tables, resolved operands) come to about 25% on top. For the disassembler corpus of
sm_86 (7,564 instructions) that is 726 KB of MOps and 1,588 rows (203 KB), 0.93 MB or 123 bytes an instruction, against the 184-byte `Instr` the section
already holds; building it takes 150 to 400 ns an instruction, 2.2 ms for that corpus. In the soak the peak resident size with the tier on was
within a median of 132 KB of the interpreter's over 257 programs. Building the sections lazily was considered and not done: a section is the unit
of the `call_once`, a kernel's code is a few hundred instructions at most in everything run here, and a per-instruction build would need a
published flag on each `MOp` and a second place where the pool grows while other threads read it, to save under a megabyte on a program that
runs at all. A very large library section (a framework's merged cubin) is where it would pay; none has shown it.

**Mutation checks.** Fifty one-line breakages of the tier, each switched on by an environment variable in a scratch copy (a different form
each: the sign of a conversion, the rounding mode of an FFMA, the BSYNC member filter, the chain's checks of the budget, the give-way and
whether the lanes are together, a shuffle table off by one, the count of barriers, the 16-byte store's last lane, ...). 42 are caught by
`sass_tier`, `sass_tier_steps`, `sass_fastpath`, `wmma_gemm` or `bsync_recursive_abi`, three (13, 32 and 41) only after tests were added for them.
Eight survive, and each is a case the compiler's output does not reach or one that cannot change the result: (9) LDC with a negative register
offset (ptxas folds the constant part into the register, so the register is never negative); (15) a second predicate on a conditional BRA; (35, 37)
the second carry out of UIADD3 and the carry of ULEA with a negated addend (ptxas passes `!PT` there); (36) UISETP's complement predicate;
(51) a bank other than 0 as an operand of the uniform datapath (ptxas keeps `LOP3 R, R, c[0x3][...]` in the vector datapath); (26) a BSYNC
that some lanes of the group skip by a guard (no program here has one); and (19) the chain continuing past `give_way`, which is an equivalent mutant: the
YIELD that sets `give_way` also sets `yield`, which the chain tests first, and the next call of `step_warp` consumes `give_way` before the
tier runs. The form analysis of these is still checked over the disassembler corpora by `test_tier`.

## Round 9: counting a memory access (`include/vgpu/exec/mem_counters.hpp`), `Module::code_at`, and the shared/local fast paths

A = a fresh Release build (CUDA 13.0 toolkit, GCC 12, LTO, `-g1`) of origin/r8-a at 3ef7a6bd, B = this branch built the same way.
Both pinned to CPU 6, `tools/bench/bench.sh --ab A B --engines sass,ptx --runs 21`, one host thread, alternating A B A B,
`bench_counters` identical in every pair. The machine was quiet (load average 1.7-2.8) for the table below; earlier runs of the same
comparison at load 12-17 had spreads of 40-70% and are not quoted. HeCBench programs run with the arguments of `~/.cache/HeCBench/reduced*.txt`.

**What the profiles said** (`bench.sh profile`, and `perf record` of 51 HeCBench programs on the SASS engine, run for at most 50 s each):
across the programs the time in `distinct_sectors` was 4.1% on average (14 programs at 5% or more, `base64e` 18.8%), in the bank-pass
count 2.3% on average (8 programs at 5% or more, `shmembench` 49.7%), in `Module::code_at` 0.6% (`bh` 4.8%), in `count_access` itself
0.5%; the texel reads (`load_scalar`, `scalar_location`, `resolve_scalar`) are 15% of the `texture` bench workload and 0.0% of every HeCBench
program (none samples texture code). `perf`'s per-line view of `shmembench` after the counting had been made cheap showed the other cost of a shared-memory access: 20% of its
samples were the instruction after a 16-byte load in libc's `memcpy`, which waited for the four 4-byte stores that had just built its source
(store forwarding does not join them).

**What changed** (everything gives the same numbers as before, lane for lane; `VGPU_FASTCOUNT=0` runs the earlier code):

* *Sectors* (`distinct_sectors`, both engines): lanes on a fixed stride are counted by arithmetic (one lane or a broadcast; a stride up to
  the access width plus 31 bytes, which covers unit stride, strides 2 to 8, 16-byte vectors and the byte strides of a packed record; a
  column, a stride that is a multiple of 32 bytes). Lanes whose sectors only go one way are counted by a comparison each; a warp that stays
  within the 64 sectors around its first lane (768 bytes below, 1.2 KB above), in any order, by the set bits of one word (a thread-private array); the rest, up to 32 lanes, by a
  128-slot hash set that needs no clearing. A warp none of those fit goes to the earlier loop. Every closed form checks every active
  lane and gives up the moment one does not fit.
* *Bank passes* (`sass_bank_passes`, `ptx_bank_passes`): lanes on a fixed stride whose words are one to a lane, end to end, overlapping,
  or all the same (unit stride of 1 to 16 bytes, a broadcast, a column of a tile, strides of 1 to 3 bytes) by arithmetic (the number of
  distinct words per bank of a run of consecutive words, and `gcd(stride, 32)` for a stride of whole words); anything else by the earlier
  table.
* *A coalesced access* the SASS executors move as a run (`load_run`/`store_run`) is counted from its first and last byte
  (`count_run`, `memcount::run_sectors`) without building the address array.
* `Module::code_at` (`src/sass/module.cpp`): a table with an entry for each 256 bytes of the code window, built when the module loads (a
  binary search if the sections are not on that grid, the original scan if they are not in order); `step_warp` asks it only when a warp
  moves to another section (`Warp::code_hint`, `src/sass/exec.cpp`), where the tier alone did before.
* The shared and local fast paths of `exec_mem` and the tier (`h_shared`, `h_local`) copy each register word with a copy whose size is
  known at compile time (`store_lane_bytes`, `load_lane_word`), test "every lane is inside" without a branch per lane
  (`include/vgpu/exec/window_check.hpp`), and a full warp's addresses are computed in a counted loop.


**Same bits.** `bench_counters` (16 workloads, both engines, and with `VGPU_TIER=1`, and with `VGPU_FASTCOUNT=0` which runs the earlier
code) matches `tools/bench/expected` unchanged; `VGPU_COUNTERS` of 51 of 58 HeCBench programs on the SASS engine and 23 of 24 on the PTX
interpreter is byte for byte the same as the r8-a build's (the others: three ran past the 120 s limit and so printed a truncated, timing-dependent
list, one is `btree`, whose counters differ between two runs of the *same* build -- a race in the program --, one finished in time on this
branch and not on r8-a, two print no counters at all). `tests/unit/test_memcount_forms.cpp` compares `distinct_sectors`,
`sass_bank_passes`, `ptx_bank_passes` and `run_sectors` with a definition written from scratch (sorted vectors of the sectors and the words)
over every stride from -70 to 130 bytes and a spread of larger ones, every width, base alignments, masks of the usual shapes, warps of
32 and 64 lanes, 120,000 random warps of nine shape families (some with one lane moved by a byte, a word or a sector), 150,000 warps of a few
lanes in no order, wide accesses of 129 to 5000 bytes, and addresses at the top of the address space; `test_memfast`'s 300,000 random warps
(against the code of before `mem_counters.hpp`) and `test_sass_access_count` pass unchanged; `test_sass_code_at.cpp` compares `code_at` with
the original scan for modules `load()` places (page table), ones off its grid (binary search) and ones in no order (scan); the window checks
are compared with the per-lane tests over 400,000 random accesses. A mutant of each closed form's constants and conditions (about 60 in
all: an off-by-one in each bound, a dropped lane check, the sign of the stride, the byte in the word of the SASS rule, the gcd, the
endpoint test, the direction test, the hash probe, ...) is caught by those tests except the ones that only turn a closed form off or move
the border between two (which the general code then counts correctly). `nvidia/tests/e2e/sass_vector_memory.cu` moves 1, 2, 4, 8 and 16
bytes through shared and local memory by lane-dependent indices and checks every word on the SASS interpreter, the tier and the PTX interpreter
(also on the RTX 3060 itself: 0 of 4,608 words differ); a build with one wrong word in each of the four shared/local paths fails it in the
engine the path belongs to, and a build whose `count_run` counts one sector too many fails `bench_counters` (`global_sectors expected 256, got 320`, ...) on the
SASS interpreter and on the tier. Builds: CUDA 13.0 and CUDA 12.0 (`/usr/bin/nvcc`, g++-12), unit tests and the e2e pass on both.

### Results

Per-call cost of the counting alone, ns for one warp of 32 lanes (`g++-12 -O3`, minimum of three alternating runs, pinned to one CPU;
the shapes are in the first column):

| shape (32 lanes) | sectors, SASS | sectors, PTX | bank passes, SASS | bank passes, PTX |
| --- | ---: | ---: | ---: | ---: |
| unit4 | 39.9 -> 8.7 | 40.5 -> 8.6 | 45.8 -> 8.4 | 42.9 -> 6.8 |
| unit8 | 45.7 -> 8.6 | 45.3 -> 8.6 | 84.9 -> 8.2 | 83.5 -> 6.7 |
| unit16 | 45.5 -> 8.6 | 45.2 -> 8.6 | 205.5 -> 8.0 | 201.8 -> 6.7 |
| unit1 | 42.8 -> 8.6 | 42.9 -> 8.6 | 49.6 -> 7.7 | 47.4 -> 6.5 |
| unit4_half | 30.6 -> 6.6 | 31.2 -> 6.7 | 28.9 -> 6.6 | 33.9 -> 5.3 |
| stride2x4 | 44.4 -> 8.6 | 45.2 -> 8.6 | 51.1 -> 8.8 | 47.3 -> 7.8 |
| stride4x4 | 45.4 -> 8.6 | 45.8 -> 8.6 | 65.5 -> 8.8 | 65.9 -> 8.0 |
| bcast4 | 42.8 -> 8.6 | 43.3 -> 8.6 | 48.4 -> 7.5 | 46.7 -> 6.0 |
| rev4 | 58.4 -> 8.5 | 56.9 -> 8.7 | 45.1 -> 8.2 | 39.6 -> 6.8 |
| col32 | 44.8 -> 8.8 | 45.7 -> 8.7 | 239.8 -> 8.8 | 240.3 -> 7.8 |
| pad33 | 44.5 -> 30.3 | 46.0 -> 30.8 | 46.0 -> 8.9 | 42.9 -> 8.0 |
| rand_wide | 150.8 -> 47.5 | 153.0 -> 47.4 | 47.8 -> 53.5 | 49.7 -> 59.2 |
| rand_small | 78.2 -> 35.6 | 78.9 -> 34.3 | 43.6 -> 48.2 | 43.8 -> 50.7 |
| unit4_unal | 45.4 -> 8.7 | 44.9 -> 8.7 | 71.5 -> 8.2 | 42.9 -> 6.7 |
| sh_unit4 | 41.0 -> 8.6 | 40.6 -> 8.6 | 45.0 -> 8.1 | 42.8 -> 6.7 |
| sh_str2 | 44.4 -> 8.7 | 44.0 -> 8.6 | 53.4 -> 8.9 | 47.9 -> 7.9 |
| sh_bcast | 42.7 -> 8.6 | 43.9 -> 8.7 | 47.1 -> 7.4 | 46.9 -> 6.0 |
| sh_2way_pairs | 71.3 -> 35.1 | 68.8 -> 35.1 | 51.6 -> 52.5 | 49.0 -> 54.7 |
| sh_unit8 | 44.6 -> 8.7 | 45.1 -> 8.7 | 79.5 -> 8.3 | 82.7 -> 6.8 |
| sh_unit16 | 44.8 -> 8.7 | 46.1 -> 8.7 | 206.1 -> 8.3 | 203.9 -> 6.8 |
| sh_red | 15.9 -> 4.5 | 16.0 -> 4.5 | 19.9 -> 20.5 | 20.0 -> 20.1 |
| sh_tile_col | 45.8 -> 8.6 | 45.7 -> 8.6 | 241.1 -> 8.8 | 237.5 -> 7.8 |

The shapes that did not move are the ones with no regular structure and a conflict (a bank-conflicted warp that is not on a stride,
`sh_red`'s two lanes); a warp that is none of the shapes pays 3 to 8 ns more than before for the closed forms that gave up
(random 4-byte words over 400 KB: bank passes 48 -> 54 ns, sectors improved by the hash).

Time in the functions, `perf record` (cpu-clock) of the same programs on r8-a (A) and this branch (B), the SASS engine, shares of the program's
samples (programs of the 51 with the highest shares; the programs that run past 50 s were cut there, so the totals are not comparable, the
shares are):

| program | `distinct_sectors` (A -> B) | bank-pass count (A -> B) | `Module::code_at` (A -> B) |
| --- | ---: | ---: | ---: |
| atomicReduction | 11.0% -> 0.9% | 0.0% -> 0.0% | 0.5% -> 0.0% |
| base64e | 18.8% -> 6.6% | 0.0% -> 0.0% | 0.0% -> 0.0% |
| bh | 4.4% -> 0.4% | 0.0% -> 0.0% | 4.8% -> 2.9% |
| fpdc | 4.0% -> 2.6% | 10.2% -> 3.4% | 0.2% -> 0.0% |
| radixsort2 | 1.1% -> 0.4% | 5.1% -> 2.8% | 3.8% -> 0.1% |
| s3d | 9.9% -> 1.2% | 0.0% -> 0.0% | 1.3% -> 0.0% |
| shmembench | 0.0% -> 0.0% | 49.7% -> 8.5% | 0.0% -> 0.0% |
| sortKV | 2.0% -> 0.8% | 8.0% -> 4.2% | 1.4% -> 0.0% |
| blockScan | 3.2% -> 0.3% | 13.1% -> 4.9% | 2.4% -> 0.3% |
| bn | 11.8% -> 5.5% | 0.0% -> 0.0% | 0.7% -> 0.0% |
| bsearch | 13.7% -> 6.9% | 0.0% -> 0.0% | 0.0% -> 0.0% |
| chemv | 10.1% -> 3.1% | 0.0% -> 0.0% | 0.0% -> 0.0% |
| contract | 12.6% -> 1.7% | 0.0% -> 0.0% | 0.1% -> 0.0% |
| **mean of 23 programs** | 7.1% -> 2.5% | 4.5% -> 1.2% | 0.8% -> 0.2% |

`tools/bench/bench.sh --ab`, 21 runs, the 16 bench workloads on both engines, one host thread pinned to CPU 6, load average 2.5:

| workload | engine | metric | r8-a (median) | this branch (median) | median ratio | best ratio | p | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| int_alu | sass | winst_per_cpu_s | 47.543 | 47.852 | 1.007x | 1.010x | 0.070 | same |
| int_alu | ptx | winst_per_cpu_s | 66.287 | 64.462 | 0.972x | 0.967x | 0.000 | slower |
| f32_fma | sass | winst_per_cpu_s | 54.027 | 54.025 | 1.000x | 0.995x | 0.379 | same |
| f32_fma | ptx | winst_per_cpu_s | 79.216 | 79.146 | 0.999x | 0.996x | 0.900 | same |
| f64_fma | sass | winst_per_cpu_s | 39.255 | 38.771 | 0.988x | 0.989x | 0.005 | same |
| f64_fma | ptx | winst_per_cpu_s | 39.709 | 39.569 | 0.996x | 1.003x | 0.880 | same |
| f64_subnormal | sass | winst_per_cpu_s | 7.509 | 7.419 | 0.988x | 0.985x | 0.000 | same |
| f64_subnormal | ptx | winst_per_cpu_s | 7.623 | 7.652 | 1.004x | 1.004x | 0.218 | same |
| wmma | sass | winst_per_cpu_s | 5.947 | 6.044 | 1.016x | 1.015x | 0.000 | same |
| wmma | ptx | winst_per_cpu_s | 4.560 | 4.583 | 1.005x | 0.986x | 0.651 | same |
| mma_ptx | sass | winst_per_cpu_s | 10.863 | 10.829 | 0.997x | 1.002x | 0.458 | same |
| mma_ptx | ptx | winst_per_cpu_s | 0.575 | 0.573 | 0.998x | 0.997x | 0.651 | same |
| mem | sass | winst_per_cpu_s | 13.754 | 15.528 | 1.129x | 1.139x | 0.000 | faster |
| mem | ptx | winst_per_cpu_s | 13.500 | 14.509 | 1.075x | 1.070x | 0.000 | faster |
| shared | sass | winst_per_cpu_s | 13.352 | 16.786 | 1.257x | 1.254x | 0.000 | faster |
| shared | ptx | winst_per_cpu_s | 10.433 | 12.354 | 1.184x | 1.167x | 0.000 | faster |
| atomics | sass | winst_per_cpu_s | 8.017 | 8.049 | 1.004x | 1.003x | 0.513 | same |
| atomics | ptx | winst_per_cpu_s | 2.930 | 2.817 | 0.961x | 0.969x | 0.000 | slower |
| texture | sass | winst_per_cpu_s | 4.319 | 4.406 | 1.020x | 1.025x | 0.000 | faster |
| texture | ptx | winst_per_cpu_s | 2.364 | 2.433 | 1.029x | 1.022x | 0.000 | faster |
| local | sass | winst_per_cpu_s | 16.363 | 25.126 | 1.536x | 1.537x | 0.000 | faster |
| local | ptx | winst_per_cpu_s | 19.277 | 23.764 | 1.233x | 1.239x | 0.000 | faster |
| divergent | sass | winst_per_cpu_s | 21.140 | 20.962 | 0.992x | 0.992x | 0.001 | same |
| divergent | ptx | winst_per_cpu_s | 37.271 | 36.547 | 0.981x | 0.973x | 0.000 | same |
| warp | sass | winst_per_cpu_s | 23.428 | 22.673 | 0.968x | 0.973x | 0.000 | slower |
| warp | ptx | winst_per_cpu_s | 35.927 | 36.069 | 1.004x | 1.004x | 0.615 | same |
| launch | sass | units_per_s | 472.328 | 499.158 | 1.057x | 1.064x | 0.000 | faster |
| launch | ptx | units_per_s | 380.717 | 391.351 | 1.028x | 1.036x | 0.000 | faster |
| api | sass | units_per_s | 5574.896 | 5529.988 | 0.992x | 1.007x | 0.056 | same |
| api | ptx | units_per_s | 5584.200 | 5545.423 | 0.993x | 0.996x | 0.035 | same |
| startup | sass | wall_s | 0.003 | 0.003 | 1.020x | 0.996x | 0.039 | same |
| startup | ptx | wall_s | 0.011 | 0.010 | 1.016x | 1.001x | 0.302 | same |

The memory workloads are the changes: `mem` 1.13x / 1.08x (SASS / PTX), `shared` 1.26x / 1.18x, `local` 1.54x / 1.23x, `launch` 1.06x / 1.03x and `texture` 1.02x / 1.03x (not investigated). The rows marked "slower" are PTX `int_alu` (0.972x), PTX `atomics` (0.961x) and SASS `warp`
(0.968x); none of them runs a function this branch changed (`perf record` of `atomics` on the PTX engine lists the same functions in the
same order on both builds), and **two builds of the same r8-a source differ by as much**: this A (with `-g1`) against another agent's build
of the same source without it, `bench.sh --ab`, 21 runs, load 2.7: `int_alu` ptx 1.040x, sass 1.060x; `atomics` ptx 1.037x, sass 1.043x;
`mem` sass 1.042x, all "faster", p = 0.000. A 3-6% difference between two builds of one source is code placement (LTO partitions, alignment),
not the source; the same comparison between this A and a copy of itself (PTX `atomics`) reads 1.009x. They are reported, not hidden, and not attributed to a
change.

HeCBench programs (`perf`-sampled in the profile above; timed here by a small script, not in the repository because it needs this machine's
HeCBench checkout, that alternates A and B runs pinned to CPU 6 and sums the child's user and system CPU seconds: 9 runs of the shorter programs,
3 of `shmembench`, `blockScan`, `fpdc`, `aop` and `s3d` on the final build; load 2; the rows other than those five were measured on the build
before `window_check.hpp` was made a header, which moves no code; `VGPU_COUNTERS` output compared on the first pair of runs):

| program | r8-a CPU s, median [range] | this branch | B / A | counters |
| --- | ---: | ---: | ---: | --- |
| shmembench | 67.77 [67.51-67.85] | 21.43 [21.35-21.45] | 0.316 | identical |
| blockScan | 10.43 [10.42-10.45] | 8.40 [8.39-8.50] | 0.806 | identical |
| s3d | 0.93 [0.92-0.93] | 0.76 [0.76-0.76] | 0.821 | identical |
| fpdc | 21.11 [20.96-21.35] | 18.91 [18.87-18.91] | 0.896 | identical |
| aop | 5.48 [5.47-5.50] | 4.99 [4.99-4.99] | 0.911 | identical |
| bsearch | 0.37 [0.36-0.38] | 0.33 [0.33-0.34] | 0.901 | identical |
| sortKV | 0.64 [0.64-0.65] | 0.58 [0.58-0.59] | 0.909 | identical |
| perplexity | 0.73 [0.73-0.74] | 0.67 [0.66-0.67] | 0.908 | identical |
| axpby | 1.06 [1.06-1.08] | 0.97 [0.97-0.98] | 0.912 | identical |
| chacha20 | 0.38 [0.37-0.38] | 0.35 [0.35-0.35] | 0.937 | identical |
| radixsort2 | 8.59 [8.54-8.67] | 7.86 [7.82-7.89] | 0.915 | identical |
| bwt | 5.57 [5.56-5.59] | 5.20 [5.16-5.25] | 0.933 | identical |
| boxfilter | 1.43 [1.42-1.44] | 1.34 [1.33-1.35] | 0.937 | identical |
| clock | 0.53 [0.52-0.53] | 0.50 [0.50-0.51] | 0.954 | identical |
| asmooth | 0.59 [0.59-0.60] | 0.57 [0.57-0.57] | 0.968 | identical |

`shmembench` does nothing but shared-memory loads and stores of several widths: 3.2x faster, about two thirds of the saving the count
(49.7% of the samples before, 8.5% after) and one third the copy that waited on its own stores.

### Not done, and negative results

* *Texel reads.* `texel_channel_bits` makes one `load_scalar` per channel, which on the `texture` workload is 15% of the samples
  (`resolve_scalar` 3.8, `scalar_location` 3.4, `texel_raw` 4.0, `load_scalar` 2.0, `load_at` 1.9). The obvious change -- one `load_run`
  for all the channels of a texel -- would not be cheaper: `MemoryManager::run_allocation` takes the allocation table's reader lock on every
  call and has none of the per-thread lookup memory `load_scalar` uses (`find_ordinary`), so one locked lookup replaces four cached ones.
  A cached multi-element read in the memory manager would be the right place; not done (the memory layer is not this branch's, and none of
  the 51 HeCBench programs samples texture code).
* *PTX atomics.* The interpreter's `exec_atom` does `load_scalar` then `store_scalar` for each lane: two lookups (40% of the PTX `atomics` workload is
  in `resolve_scalar`, `scalar_location`, `load_scalar`, `store_scalar`). A read-modify-write that looks the address up once would halve it;
  it belongs to the memory layer and the interpreter, and was not touched.
* *Local memory layout.* `base64e` spends about a third of its time in the local-memory fast path, where lane `l`'s byte `o` is at
  `local[l * local_size + o]`, so a warp-wide access touches 32 cache lines; interleaving by words, as the hardware does, would touch one. Every
  user of `Warp::local` would change; not done.
* *Bank passes of warps that are not on a stride* (a tile read two rows at a time, a reduction's every-eighth lane) and sector counts
  for 33 to 64 active lanes that are neither on a stride nor monotone still take the earlier code. `sass_bank_passes` is still 3 to 5% of
  `blockScan`, `sortKV`, `fpdc`; `shmembench` 8.5%.
* *Two-section ping-pong.* `bh` still spends about 5% in `fetch` and `Module::code_at`: a divergent warp alternates between a kernel's section
  and a device function's, so the one-entry `Warp::code_hint` misses; a two-entry hint would catch it.


## The AMD engine

The sections above measure the NVIDIA engines (SASS and PTX). The AMD interpreter
(`amd/src/gcn_exec.cpp`, `gcn_decode.cpp`, `rdna_decode.cpp`) had no baseline. This
section is that baseline and its first profile. **No simulator source was changed to produce
it**: the harness, workloads and profiler are new files under `tools/bench-amd/`, and one
`add_test` in `CMakeLists.txt` wires the counter check. Every number was measured by the
commands shown.

**Measured at** `speedup` 8f9eb524 plus the files of this section; Release build (`-O3
-DNDEBUG`, LTO, GCC 13.3.0, baseline x86-64: no POPCNT, F16C, FMA or AVX2 outside the few
`target(...)` functions); programs built by `hipcc` of ROCm 7.2 (HIP 7.2.26015) for `gfx942`
and `gfx1030`, run on the `amd/mi300x` and `amd/rx6800` profiles.

**Machine**: server2, AMD Ryzen 9 9950X (16 cores, 32 logical CPUs) shared with a Postgres,
a CI runner and other people's builds. Every run was pinned (`taskset`) to CPUs 24-31 (the
SMT siblings of cores 8-15). The 1-minute load average before each workload was 9 to 42
during run A below and 10 to 52 (median 35) during run B; the A/B experiments at the end ran at
load 10 to 22, 10 to 13 and 27 to 46. The numbers compare two builds on one machine; the
spreads say how much the machine took away. On this host read the **best** column.

### The harness

```
tools/bench-amd/bench.sh --build build --cpus 24-31          # time every workload, MI300X and RX 6800
tools/bench-amd/bench.sh --build buildA --build buildB       # alternating A B A B ... with ratio and p
tools/bench-amd/bench.sh --check-counters --build build      # counts (not times) against tools/bench-amd/expected
tools/bench-amd/bench.sh --update-expected --build build     # rewrite expected/amd-hip-<release>.json
tools/bench-amd/profile/profile.sh build-g valu_int mi300x   # sampled profile of one workload
```

`tools/bench-amd/bench.py` is the AMD twin of `tools/bench/bench.py` (it imports that file's
statistics, Mann-Whitney p-value and process runner; the verdict rule is the same).

* The workloads are `kernels/bench_amd.hip`, one self-checking HIP program
  (`bench_amd <workload> <quick|full>`) compiled per target with `hipcc -O2 -ffp-contract=off`
  and run through VirtualGPU's `libamdhip64` like any user program. Each workload compares its
  results with a host reference bit for bit and prints `PASS` only on a match. A timed run that
  fails is recorded, repeated (up to three times) and listed at the end, never dropped. (One
  first-attempt failure was seen once, during the first attempt of run B before this was added: valu_int
  printed MISMATCH with a device phase of 0.4 ms, i.e. the kernel apparently did not run; 160 quick runs
  under heavy CPU load and 25 harness repeats could not reproduce it, and run B as reported had no failures.
  Unexplained; launch errors are checked now.)
* A counting run per workload, target and build gives exact instruction counts:
  `rocprofv3 --pmc SQ_INSTS_* ...` inside `vgpu shell -y --gpu amd/<profile> --count 1
  --no-isolate -c`, one host thread, summed over the dispatches. Timed runs have no profiler.
* **Wave-instructions per CPU second** (Minst/s): counted instructions over the CPU time (all
  threads) of the device phase, which the program times around its launches and
  `hipDeviceSynchronize`. Launch and API workloads report calls per second; `startup` is the
  whole-process wall time. 9 runs after a warm-up, alternating over the configurations: one
  host thread (`VGPU_THREADS=1`) and the default (all CPUs the process may use: 8 here).
  With several threads CPU time is a sum, so the last column of the baseline gives the rate by
  wall clock, which is what a user sees.

*What counts.* The interpreter's total (`DispatchStats::instructions`) is not a profiler counter,
so the harness sums `SQ_INSTS_{SALU,SMEM,VALU,VMEM,LDS,BRANCH,SENDMSG}`, which omits SOPP
instructions that are none of those (`s_waitcnt`, `s_nop`, `s_endpgm`, `s_barrier`). Against
`amd_exec_bench`'s exact count (same quick-size launches) the sum is low by 0.09% on `valu_int`
(823040 vs 823808), 0.13% on `valu_f32` (568832 vs 569600), 0.02% on `divergent` (952528 vs
952720) and 11% on `wave` (311168 vs 350080: each `ds_bpermute` is followed by an `s_waitcnt`).

### Workloads

| workload | group | what it exercises |
| --- | --- | --- |
| valu_int | alu | 32-bit integer chain: multiply-add, xor, shifts, rotate, popcount |
| valu_f32 | alu | f32 fma chains plus unfused mul/add |
| valu_f64 | alu | f64 fma chains |
| mfma | matrix | `v_mfma_f32_16x16x16_f16` in a loop (MI300X only; RDNA2 has no matrix unit) |
| mem | memory | global memory: 16-byte vector copy, scalar scale, indexed gather |
| lds | memory | LDS with barriers: tree reduction, 32x32 tile transpose |
| atomics | memory | global atomic add (one address, 256 bins), max, a CAS loop |
| divergent | control | data-dependent `while` (collatz) and a `switch` |
| wave | control | `__shfl*` (`ds_bpermute`), `__ballot`, `__any`, `__all` |
| launch | api | 100000 one-wave launches on two streams |
| api | api | `hipMalloc`/`Memcpy`/`Memset`/`Free`, events, a stream: 13 calls a round |
| startup | api | one launch of a trivial kernel; read whole-process wall time |

The RX 6800 binary is wave32 (`gfx1030`), so its counts are about twice the MI300X's for the
same source. WMMA (RDNA3+) is not covered. Mix of what they run (full-size counts; MFMA is part
of VALU; SMEM is under 0.1% everywhere):

| workload | target | wave-instr | waves | VALU | SALU | VMEM | LDS | BRANCH | MFMA (in VALU) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valu_int | mi300x | 8195840 | 256 | 81% | 13% | 0% | 0% | 6% | 0% |
| valu_int | rx6800 | 14343680 | 512 | 79% | 14% | 0% | 0% | 7% | 0% |
| valu_f32 | mi300x | 5637632 | 256 | 73% | 18% | 0% | 0% | 9% | 0% |
| valu_f32 | rx6800 | 12298752 | 512 | 75% | 17% | 0% | 0% | 8% | 0% |
| valu_f64 | mi300x | 8199424 | 256 | 62% | 25% | 0% | 0% | 12% | 0% |
| valu_f64 | rx6800 | 14350336 | 512 | 57% | 29% | 0% | 0% | 14% | 0% |
| mfma | mi300x | 2050368 | 64 | 25% | 50% | 0% | 0% | 25% | 25% |
| mem | mi300x | 4967168 | 768 | 54% | 18% | 18% | 0% | 9% | 0% |
| mem | rx6800 | 12042240 | 1536 | 62% | 15% | 15% | 0% | 8% | 0% |
| lds | mi300x | 3087680 | 512 | 14% | 33% | 15% | 21% | 17% | 0% |
| lds | rx6800 | 7265600 | 1024 | 11% | 30% | 30% | 16% | 14% | 0% |
| atomics | mi300x | 11955856 | 64 | 29% | 50% | 7% | 0% | 14% | 0% |
| atomics | rx6800 | 12391696 | 128 | 29% | 49% | 8% | 0% | 14% | 0% |
| divergent | mi300x | 17902446 | 1024 | 50% | 35% | 0% | 0% | 15% | 0% |
| divergent | rx6800 | 33425580 | 2048 | 46% | 35% | 0% | 0% | 19% | 0% |
| wave | mi300x | 6147968 | 128 | 50% | 33% | 0% | 12% | 4% | 0% |
| wave | rx6800 | 12295168 | 256 | 50% | 33% | 0% | 12% | 4% | 0% |

### Baseline

`tools/bench-amd/bench.sh --build build --cpus 24-31 --runs 9`, run twice in a row (A, B). A cell is
`median [best] spread` of Minst/s (1 thread); 8-thread columns are by wall clock.

| workload | target | wave-instr | 1 thread, run A: median [best] spread | run B | 8 threads by wall clock, A / B (median) | best 8-thread wall rate / best 1-thread rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| valu_int | mi300x | 8195840 | 3.10 [3.15] 3% | 2.02 [3.18] 71% | 23.7 / 14.2 | 7.6x |
| valu_int | rx6800 | 14343680 | 3.25 [3.31] 3% | 3.32 [3.33] 21% | 24.8 / 17.1 | 7.6x |
| valu_f32 | mi300x | 5637632 | 2.32 [2.35] 5% | 1.33 [2.11] 63% | 14.3 / 6.1 | 7.3x |
| valu_f32 | rx6800 | 12298752 | 2.30 [2.97] 46% | 2.92 [3.01] 44% | 15.6 / 15.1 | 7.5x |
| valu_f64 | mi300x | 8199424 | 2.82 [2.89] 26% | 2.80 [2.91] 52% | 21.5 / 12.9 | 7.5x |
| valu_f64 | rx6800 | 14350336 | 3.04 [3.08] 4% | 3.11 [3.16] 49% | 23.0 / 12.0 | 7.6x |
| mfma | mi300x | 2050368 | 0.91 [0.91] 2% | 0.83 [0.86] 25% | 7.0 / 3.3 | 7.7x |
| mem | mi300x | 4967168 | 2.49 [2.50] 8% | 2.36 [2.39] 9% | 19.1 / 9.2 | 7.7x |
| mem | rx6800 | 12042240 | 3.32 [3.37] 35% | 3.07 [3.17] 37% | 25.1 / 13.7 | 7.6x |
| lds | mi300x | 3087680 | 1.37 [1.77] 48% | 1.11 [1.65] 67% | 9.1 / 7.2 | 7.5x |
| lds | rx6800 | 7265600 | 2.97 [3.14] 38% | 3.08 [3.15] 7% | 19.0 / 14.6 | 7.5x |
| atomics | mi300x | 11955856 | 6.29 [6.36] 49% | 6.12 [6.25] 48% | 1.1 / 1.1 | 0.3x |
| atomics | rx6800 | 12391696 | 6.67 [6.71] 3% | 4.26 [6.49] 70% | 2.2 / 2.4 | 0.6x |
| divergent | mi300x | 17902446 | 5.15 [5.31] 9% | 3.50 [5.46] 66% | 28.5 / 15.5 | 7.3x |
| divergent | rx6800 | 33425580 | 3.75 [5.62] 62% | 5.56 [5.75] 45% | 14.9 / 23.4 | 7.4x |
| wave | mi300x | 6147968 | 1.88 [2.26] 27% | 3.03 [3.08] 38% | 8.4 / 16.1 | 7.5x |
| wave | rx6800 | 12295168 | 2.32 [2.84] 23% | 3.99 [4.03] 2% | 9.8 / 26.0 | 7.5x |
| launch | mi300x | | 304 [334] 28% klaunch/s | 458 [467] 5% klaunch/s | 304 / 463 klaunch/s (all threads) | |
| launch | rx6800 | | 239 [248] 26% klaunch/s | 370 [377] 4% klaunch/s | 237 / 373 klaunch/s (all threads) | |
| api | mi300x | | 432 [1004] 196% kcall/s | 1265 [1426] 20% kcall/s | 580 / 1245 kcall/s (all threads) | |
| api | rx6800 | | 1113 [1280] 23% kcall/s | 1228 [1423] 105% kcall/s | 1114 / 1020 kcall/s (all threads) | |
| startup | mi300x | | 0.002 s [0.002] | 0.002 s [0.002] | | |
| startup | rx6800 | | 0.005 s [0.005] | 0.005 s [0.005] | | |

Reading it. The two runs' bests agree within 3% on 9 of 17 single-thread rows and within 7% on
15; the exceptions are `wave` (run A never got a clean run: 1.9 vs 3.1 on the MI300X) and
`valu_f32` on the MI300X (11%). Medians moved by up to 1.7x between runs at load 30 to 50, so use
the larger best. In round numbers, the interpreter runs **about 3 Minst/s per host thread on
VALU code** (valu_int 3.3, valu_f64 2.9 to 3.2, valu_f32 2.1 to 3.0), i.e. about 300 ns per
64-lane wave-instruction; **`mfma` 0.9** (a quarter of its instructions, nearly all its time),
**`lds` 1.7** on the MI300X (3.1 on the RX 6800), `mem` 2.4 to 3.4, `wave` 3.1 to 4.0, and the
scalar- and branch-heavy `atomics` (6.3 to 6.7) and `divergent` (5.2 to 5.7) are fastest.
An RDNA wave32 instruction costs as much as a wave64 one (the lane loops run over 64 lanes
with the upper 32 masked off), so per lane it is twice as expensive. With 8 CPUs every workload
but `atomics` is 7.3x to 7.7x faster by wall clock (best over best) at 3 to 16% more CPU time.
`atomics` is the exception, slower with threads (below). `launch` is 300 to 460 klaunch/s,
`api` 1.0 to 1.4 Mcall/s, `startup` 2 ms (MI300X) and 5 ms (RX 6800), which is the whole process.

### The counter check

`--check-counters` runs the workloads at the `quick` size on one host thread under `rocprofv3` and
compares the `SQ_INSTS_*`, `SQ_WAVES`, `SQ_INSTS_VMEM_RD/WR` and `TA_FLAT_ATOMIC_WAVEFRONTS` totals,
per workload and target, with `tools/bench-amd/expected/amd-hip-7.2.26015.json` (counts depend on
the compiler, so one file per hipcc release). It is the ctest test `bench_amd_counters` (about 6 s),
and skips (`SKIP:`) without a ROCm that has `hipcc` and `rocprofv3`, without an expected file for
the release, or with a sanitizer-built shim (both skip paths exercised). The counts at one
thread are deterministic, including the `atomics` CAS loop: `--update-expected` followed by two
`--check-counters` matched all 21 workload/target pairs. Mutation check: raising one stored count
by one makes it exit 1 with `mi300x/valu_int: SQ_INSTS_VALU expected 666625, got 666624`.

### Profiles

`perf` is unusable on server2 (`perf_event_paranoid` 4, no sudo: "Failure to open any events").
The profiles come from `tools/bench-amd/profile/sigprof.c`, an `LD_PRELOAD` sampler (an `ITIMER_PROF`
timer, 2000 Hz here, recording the interrupted pc and with `SIGPROF_STACK=1` ten caller frames) and
`report.py` (`addr2line`; `--within`, `--inclusive`, `--lines`, `--callers-of`). The simulator was
built with `-g` and the same `-O3`/LTO. All are `VGPU_THREADS=1`, full size, 1700 to 5500 samples
inside `Machine::step` each (that filter leaves out the program's own host reference). A sample is
charged to the function containing the instruction, so inlined callees count toward their caller.
libc has no symbols here: the entry labelled `__nss_database_lookup` is `memcmp` (its callers by
the recorded chains are `OpName::rfind`, `std::string::rfind`/`compare`/`find`).

Share of the samples inside `Machine::step`, MI300X:

| workload | samples | name machinery | operand reads | libc + libgcc | `Inst` ctor/dtor |
| --- | ---: | ---: | ---: | ---: | ---: |
| valu_int | 2543 | 26.7% | 28.2% | 7.8% | 1.9% |
| valu_f32 | 2460 | 22.3% | 29.4% | 4.1% | 1.3% |
| valu_f64 | 2750 | 32.9% | 3.3% | 3.9% | 1.2% |
| mfma | 2209 | 4.2% | 0.0% | 40.3% (`__extendhfdf2`) | 0.4% |
| mem | 1945 | 16.1% | 21.6% | 4.0% | 1.1% |
| lds | 1753 | 47.6% | 3.8% | 7.5% | 0.7% |
| atomics | 1850 | 33.0% | 9.9% | 5.0% | 2.0% |
| divergent | 3114 | 41.0% | 11.2% | 5.0% | 2.2% |
| wave | 1872 | 26.9% | 23.9% | 7.8% | 1.2% |

*Name machinery* is the `OpName` class, `gcn_exec.cpp` lines 90 to 125: `name_hash` (line 100, an FNV
hash of the instruction's name taken on every `OpName` construction) and `operator==` (line 111, once per
rung of every `if (op == "..."_op) ... else if` ladder). *Operand reads* are `scalar` (559 to 603) and
`lane_src` (638 to 645): a source that is not a VGPR goes through the `scalar` switch once per lane
though it is the same in every lane. (Line ranges are read from the profile's line view.) On the RX 6800 the name machinery is
34% (`valu_int`), 29% (`valu_f32`), 40% (`valu_f64`), 47% (`lds`, `divergent`), 43% (`atomics`).

Top functions per workload (self time inside `step`; "lane loops" are the `each` lambdas of `vector_alu`):

| workload | top functions |
| --- | --- |
| valu_int | lane loops 23.2%, `OpName::operator==` 22.3%, `vector_alu` 11.8%, `scalar` 11.3%, memcmp 6.3%, `step` 5.3%, `more_alu` 5.0% |
| valu_f32 | `OpName::operator==` 19.0%, `vector_alu` 15.3%, `lane_src` 13.1%, `scalar` 13.0%, lane loops 10.6%, `step` 4.9% |
| valu_f64 | lane loops 33.9%, `OpName::operator==` 27.0%, `step` 9.1%, `vector_alu` 6.2%, `scalar` 4.5%, `sgpr` 4.4% |
| mfma | `matrix_multiply` 52.0%, `__extendhfdf2` 37.8%, `step` 3.0% |
| mem | lane loops 20.3%, `scalar` 13.1%, `OpName::operator==` 11.6%, `MemoryManager::scalar_location` 7.5%, `step` 5.9%, `sgpr` 5.9%, `global_access` 4.3% |
| lds | `OpName::operator==` 41.8%, `lds_access` 14.6%, memcmp 7.5%, `step` 6.9%, `scalar_location` 5.8% |
| atomics | `step` 18.1%, `OpName::operator==` 17.2%, `scalar_location` 7.1%, `scalar_alu` 5.6%, `lane_src` 5.0% |
| divergent | `OpName::operator==` 31.0%, `step` 11.5%, lane loops 11.4%, `vector_alu` 8.7%, `scalar` 5.4% |
| wave | `OpName::operator==` 23.5%, lane loops 12.9%, `lds_access` 10.6%, `scalar` 9.1%, `lane_src` 5.6%, `compare` 4.8% |

**`launch` and `api`** are not interpreter-bound: `Machine::step` is 39% inclusive of the `launch`
samples; the rest is the work-queue hand-off (`WorkQueue::submit`, condition variables), `set_up_group`
(12.6%, the register-file set-up of each one-wave launch) and `MemoryManager::free`. `api` is about 90% hand-off and
allocator.

**`atomics` with threads.** `memory_atomic_lock` (`gcn_exec.cpp` line 8244, 251 striped mutexes by address) is
taken for every lane of every atomic; a workload that hits one address from every work-group queues the
host threads on one mutex. At 8 threads 87% of the samples (quick size, 315 samples) are in `pthread_mutex_lock`,
`__lll_lock_wait_private` and `__lll_lock_wake_private`, the run takes 10.9 to 11.1 s of wall clock against
1.9 s on one thread (5.6x to 5.9x slower), at 65 to 75 s of CPU. Counts also stop being deterministic there, which is
why the counter check is at one thread.

### What is re-derived on every execution and could be decoded once

`DecodeCache` decodes each instruction once per module and `fetch` is cheap (1 to 3%). What the decoded
`gcn::Inst` lacks is the operation's identity in a form the executor can use:

1. **The operation as a name.** Every helper builds an `OpName` (FNV over the characters) and compares against
   precomputed literal hashes rung by rung. An instruction `vector_alu`'s own ladder handles has first failed
   the 118 rungs of `more_alu` (called at line 3498), and `rdna_alu`/`cdna5_alu` before that on those parts;
   `vector_alu` alone has 217 rungs, the file 1050. `rdna_alu` and `cdna5_alu` also copy the name into a
   `std::string`, strip the suffix and hash it again, per instruction (lines 1935 to 1938, 2384 to 2387).
2. **Classifications from the name's text:** `rfind("v_mfma", 0)`, `rfind("v_cmp", 0)`, `find("f64")`, `find("_u16")`,
   `is_plain_store`'s `find("_store")`, `as_short` assembled as a string in `vector_alu`. Each is a `memcmp` or scan per execution.
3. **Operand classes.** Whether a source is a VGPR or lane-invariant is known at decode, but `lane_src`
   re-tests it and calls `scalar` per lane.
4. **Per-instruction frame:** a default-constructed `Inst remapped`, `Narrowed`, `FloatMode`, `debug::active()`, `tracing()`.
5. **Matrix operands:** `matrix_multiply` allocates three `std::vector<double>` per MFMA, decodes every element
   of A, B and C through a chain of `type ==` tests, and does 4096 double multiply-adds for a 16x16x16.

### Tried on scratch copies (not committed)

Each is a patched copy of the tree built in its own directory; `bench.sh --build base --build patched --runs 9 --threads 1
--cpus 24-31`, MI300X, counters identical in every case (the harness prints `COUNTER DRIFT` otherwise), results
`PASS`. Ratio = patched / base Minst/s (>1 faster); p is Mann-Whitney.

| experiment | workload | median ratio | best ratio | p | load | verdict |
| --- | --- | ---: | ---: | ---: | --- | --- |
| X1+X4: half to double by bit manipulation in `matrix_multiply`'s `value` (exhaustive check: all 65536 patterns equal the compiler's conversion, NaNs and subnormals included), and `Inst remapped` as `std::optional<Inst>` | mfma | 1.024 | 1.093 | 0.136 | 10-22 | same |
| | valu_int | 1.009 | 0.969 | 0.796 | | same |
| | valu_f32 | 0.944 | 0.923 | 0.006 | | slower |
| | valu_f64 | 0.949 | 0.943 | 0.546 | | same |
| | mem / lds | 1.006 / 0.989 | 1.106 / 1.000 | 0.80 / 0.34 | | same |
| | atomics | 0.950 | 0.948 | 0.000 | | slower |
| | divergent | 0.943 | 0.952 | 0.000 | | slower |
| | wave | 0.965 | 0.973 | 0.001 | | slower |
| X5: the name hash cached in the `Inst` (every `OpName(in.name)` site and `vector_alu`'s; a throwaway version keyed by the string's address, single-thread only) | valu_int | 1.071 | 1.012 | 0.546 | 10-13 | same |
| | valu_f32 / valu_f64 | 1.008 / 0.956 | 1.012 / 0.957 | 0.04 / 0.000 | | same / slower |
| | mem, lds, atomics, divergent, wave | 0.99 to 1.01 | 0.999 to 1.013 | 0.02 to 0.19 | | same |
| X6: `lane_src` switches on the operand kind and reads VGPRs, inline constants and literals directly (no `scalar` call) | valu_int | 1.083 | 1.108 | 0.161 | 27-46 | same |
| | valu_f32 | 0.873 | 0.960 | 0.436 | | same |
| | mem | 1.062 | 1.075 | 0.024 | | faster |
| | wave | 1.049 | 1.088 | 0.387 | | same |
| | divergent | 1.047 | 1.046 | 0.000 | | faster |

What this says, and where the profile misled:

* **X1**: `__extendhfdf2` vanished from the profile (it was 37.8% of `mfma`), yet the speed-up is not significant
  (median 2%, best 9%). A profile of the patched build puts 555 of 2074 samples in the `value` lambda itself: the
  libgcc call was a small part of the per-element decode, so the cost is `value()`'s type chain and the vectors, not the conversion. My
  first estimate (1.5x to 1.7x) was wrong. The libgcc routine's hot instruction is the one after `stmxcsr`, so
  skid made it look like the whole cost.
* **X4** (lazy `Inst remapped`) is slower, 5 to 6% on four workloads at p <= 0.006 and faster on none: dropped. The profile's
  1 to 3% for `Inst` construction was not a saving.
* **X5** is no change (within 1.3%, one 4% slowdown on `valu_f64`) even though `name_hash` was 5 to 16% of the samples on the
  workloads it should help. The compiler may already be hoisting or the sampled lines were skid from the ladder. Dropped; an earlier
  version of X5 that missed `vector_alu`'s construction site was also within noise.
* **X6** is a small gain on `mem` and `divergent` (about 5%, p <= 0.024) at load 27 to 46, noise elsewhere: promising but
  needs a quiet-machine confirmation before it counts.

The lesson for whoever does item 1 to 3 below: shares of samples overstate what a change removes, and the
name ladder (line 111) is the only item whose share is large and has not been tried.

### Ranked cheap wins

Estimates are what a sample share could at most give, discounted by the lesson above; none of these was applied
except where the table above says so. Every change needs the same-bits test and the A/B rule.

| # | change | evidence | est. gain | risk |
| --- | --- | --- | --- | --- |
| 1 | Replace the `op == "..."_op` ladders by a `switch` on the hash (`consteval` case labels, a compile-time check that no two rungs collide) or an opcode id assigned at decode; or at minimum order `more_alu`'s 118 rungs behind a cheap class test | line 111 is 12 to 48% of samples (median 26%; `lds` 42%, `divergent` 31%, RX 6800 `valu_int` 28%); not tried, since it touches about a thousand rungs | 10 to 30% on ALU-bound kernels, more on `lds` (untested; X5 shows the hash itself is not the cost, the compares are) | medium: mechanical but large; `||` rungs and fall-through to later helpers; counter check plus a test executing every opcode both ways |
| 2 | Decode-time flags (`is_mfma`, `is_cmp`, `is_f64`, `is_store`, short-form name) instead of the `rfind`/`find` text tests; no name copy in `rdna_alu`/`cdna5_alu` | `memcmp` 4 to 10% of samples; RDNA copies the name per instruction | 3 to 8%, more on RDNA | low |
| 3 | Host atomics (`__atomic_*`) for aligned 32/64-bit add/min/max/and/or/xor/exchange/cas instead of the striped mutex | 87% of samples at 8 threads in the mutex; 5.6x to 5.9x slower in wall clock than one thread | `atomics` at the default thread count: 5x or more; none at one thread | medium: ordering, float and packed atomics; counts only deterministic at one thread |
| 4 | Lane-invariant operands read once per instruction (SGPR, constant, literal): finish X6 into the `each` helpers | 22 to 29% operand-read share on `valu_int`, `valu_f32`, `mem`, `wave`; X6 alone measured about 5% on two workloads | 5 to 15% on VALU-bound kernels | medium: DPP, SDWA `sel`, `abs`/`neg`, `narrow_dst` all pass through `lane_src` |
| 5 | `matrix_multiply`: no per-MFMA heap vectors, element decode hoisted by operand type, bit-conversion of halves (X1) | `mfma` is 25% of its instructions and ~90% of its time; X1 alone gave 2 to 9% | up to 1.5x on `mfma` (untested beyond X1) | low to medium: exhaustive conversion test; results compared bit for bit on rocWMMA/rocBLAS paths |
| 6 | Contiguous-lane fast path for global memory (one allocation lookup per wave) | `scalar_location` + `load_scalar` 10% of `mem`, 6 to 8% of `lds`/`atomics` | 5 to 12% on memory workloads | medium: the memory layer is shared with the CUDA engines |
| 7 | Cheaper one-wave launch (`set_up_group`, `VgprFile` zeroing) | 12.6% of `launch` samples | 5 to 10% of launch rate | low |
| 8 | `popcnt` target attribute for `__popcountdi2` (1.4% of `valu_int`) | small | 1 to 2% | very low |

### Not done, and not measured here

* The `std::optional<Inst>` change (X4) and the cached hash (X5) are negative results and stay out; X1 and X6 are inconclusive or small.
* Loops over 64 lanes with a mask test (wave32 pays for 32 masked lanes), `ctz` lane iteration: not measured.
* RDNA3/4 `v_wmma_*`, other MFMA shapes, gfx1250: not timed; one RDNA part only (RDNA2).
* The effect of any ranked change on pantheon, PyTorch or rocBLAS workloads; those spend more time in
  `matrix_multiply`, the memory manager and `lds_access` than these microbenchmarks.
* Host hardware counters (no PMU access here).


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
  profile runs its PTX, not its SASS). The AMD engine has its own section, "The AMD engine".
* The pantheon soak workloads and the PyTorch/cuBLAS paths.
