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

Opt-in when this round was written (`VGPU_TIER=1`); it has been on by default since the section "The tier on by default" below. `VGPU_TIER=0`, `VGPU_FASTPATH=0`, `VGPU_SIMD=off` and `VGPU_SASS_TRACE` switch it off.
The next paragraphs are as written then ("with `VGPU_TIER` unset" meant the interpreter).

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
that the tier copies, and put the tier in the CI jobs that can soak it. It was still not the default then (it is now, see below).

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

## The tier on by default (`VGPU_TIER=0` turns it off)

The SASS engine now runs its straight-line instructions through the pre-decoded tier unless `VGPU_TIER=0`. The code change is one line
(`tier::requested()`); `VGPU_FASTPATH=0`, `VGPU_SIMD=off` and `VGPU_SASS_TRACE` still switch it off. Evidence for the decision, all on
the 32-thread AMD box ("server2", shared; Release, CUDA 13.0, GCC 13), A = origin/main 8f9eb524 (tier opt-in, unset = off), B = this change:

* **Tier-related tests with the tier forced on, every architecture.** `e2e_sass_tier` with `VGPU_TIER_OTHER_ARCHS=1` (sm_75, 80, 86, 89, 90,
  100, 120; `sass_tier`, `sass_tier_steps`, the 39 other programs of its list on every architecture each can be built for, the faults, four
  host threads, the tier without its chain, `VGPU_TC_EXACT=0`): 264 runs compared byte for byte and counter for counter with the
  interpreter, no difference (9 min 40 s). `sass_tier_steps` fails in both modes, identically, on sm_100 and sm_120 (exit 1; it was so
  before). One build failure, not a difference: `sass_fastpath` uses bf16 `__hfma2`, which nvcc will not build for sm_75; the soak mode started
  it at sm_75 (the list now says sm_80). `test_tier` and `test_fastpath` pass. The 16 bench workloads x SASS and PTX: the counters of
  VGPU_TIER=0, of `VGPU_TIER=1` and of the default are the same in all 2,418 fields (checked with `bench.py --check-counters` against a
  file written by the interpreter on this machine; the committed file is skipped by this box's nvcc 13.0.88, whose device code hashes
  differently, and its counts equal the fresh ones in every field).
* **The whole non-AMD ctest suite with the tier on (the default), 404 tests, `-j4`, 83 minutes under a loaded machine.** 364 pass, 24 skip (tool or library absent), 16 failed. The
  failures were: 12 `e2e_*_paths`/`nvcomp`/`cudss`/`cufile`/`nvshmem_host`/`blas_host_sums` tests that ran while other jobs took the host's
  memory ("device out of memory ... below the reserve"), all of which pass when run again; `lint_curl_retries` (needs a git checkout; the
  build tree here has none); `e2e_lockstep_blocks`, which fails with the tier on, with `VGPU_TIER=0` and on the unchanged A build
  (timed-out waits on a busy host); `e2e_cutlass_hopper` (first run failed under the same memory pressure, then passed with the tier on in
  9.0 min and with `VGPU_TIER=0` in 6.3 min on a machine whose load changed between the two, and not a timing measurement: it includes
  compiling CUTLASS); and `e2e_cutlass_gemm`, whose `sm90 group_gemm` case ran for over 30 minutes with the tier on and was killed by the
  1 h ctest limit; the same binary with `VGPU_TIER=0` ran 50 min and reported the same `Failure` (CUTLASS's own check at
  group_gemm.cu:117), so the failure is not the tier's but the case is too slow on this box to say more. The PyTorch tests skipped (no torch
  here). Of the 16, 14 pass when run alone (or fail identically with the tier off), so the tier ran under every `e2e_*` that launches a kernel.
* **Corpora** (`tools/verify/run_corpora.sh`, HeCBench reduced list 34, Rodinia 12, CUDA Samples 98; one host thread, counters on;
  configurations: A, A again, B default, B with `VGPU_TIER=1 VGPU_TIER_STATS=1`, B with `VGPU_TIER=0`): 649 runs. B against A, and B with the
  tier named against A: 0 programs differ in stdout, stderr, output files or counters. Programs that differ between two runs of A itself
  (`atomicCost`, `backprop`, `cudaGraphsPerfScaling`, `memMapIPCDrv`, `radixSortThrust`, ...) are the same ones that differ between A and B.
  The tier ran 97.0% (HeCBench), 99.1% (Rodinia) and 96.5% (Samples) of the warp instructions. 5 HeCBench and 13 Samples programs time out
  (120 s) in A and were not compared. PolyBench was not run here.
* **Speed and cost**, `tools/bench/bench.sh --ab` (A, B), `--engines sass`, `--cpus 20-23` of a quiet range (box load 2 to 9), no environment set
  for either; counters identical in every row:

| workload | A median | B median | B/A median | B/A best | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| launch (k launches/s, 11 runs) | 617 | 721 | 1.17x | 1.17x | <0.001 |
| startup (wall s, 11 runs) | 0.002 | 0.002 | 1.01x | 1.00x | 0.90 (same) |
| api (k calls/s, 11 runs) | 8887 | 8796 | 0.99x | 0.96x | 0.013 (below 2%: same) |
| int_alu / warp / mem | 57 / 31 / 20 | 143 / 99 / 28 Minst/s | 2.49x / 3.17x / 1.40x | 2.50x / 3.16x / 1.41x | <0.001 |
| f32_fma / f64_fma / f64_subnormal | 64 / 43 / 44 | 145 / 64 / 62 | 2.26x / 1.49x / 1.42x | 2.27x / 1.50x / 1.43x | <0.001 |
| shared / atomics / local / divergent (9 runs) | 24 / 9.9 / 31 / 24 | 31 / 12.5 / 57 / 39 | 1.28x / 1.26x / 1.82x / 1.60x | 1.29x / 1.29x / 1.82x / 1.83x | <0.001 |
| wmma / mma_ptx / texture | 2.0 / 3.8 / 5.2 | 2.2 / 4.2 / 5.5 | 1.10x / 1.11x / 1.06x | 1.09x / 1.10x / 1.06x | <0.001 |

  Startup and the API-call workload do not regress (a section is analysed when its first warp runs, at 150 to 400 ns an instruction; the
  startup program's one small kernel costs nothing measurable), and peak RSS is the same to the first decimal (launch 8.8 vs 8.9 MB, mem 40.9 vs
  41.0 MB). Lazy or threshold-based analysis was therefore not pursued: there is no measured cost for it to remove. The cost that remains is
  the opposite one: with `VGPU_TIER=0` the interpreter is what the round 9 notes say (2 to 4% slower on `launch` and `warp` than the
  r8 build).

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

* *Soft path for subnormal f64/f32 operands.* Not done: it cannot be measured on server2, whose CPU (AMD Ryzen 9 9950X, Zen 5) takes no
  assist on FMA or ADD. Micro-benchmark (4096 values, -O2 -mfma, pinned to CPUs 28-31, ns per op, normal / all-subnormal operands): f64 fma
  0.42 / 0.38, add 0.38 / 0.40, mul 0.38 / 0.71, div 0.75 / 2.13, sqrt 2.06 / 4.25; f32 fma 0.42 / 0.53, add 0.38 / 0.38, mul 0.38 / 1.39,
  div 0.47 / 1.81, sqrt 2.12 / 5.43. The harness agrees: `f64_subnormal` runs at the rate of `f64_fma` (SASS 38.8 vs 40.3 Mi/s, PTX 21.6 vs
  27.0, 3 runs, load 13). The 6-10x gap in the f64_subnormal / f64_fma rows was seen on Intel (i9-12900K), so any soft path has to be
  measured there; on Zen it could only lose.
* *Texel reads and PTX atomics* are done (see "Memory lookups" below).
* *SASS atomics on the one-lookup read-modify-write.* Not done, and nothing to gain: the SASS executor's global atomics already take a warp
  at once (`load_batch` / `store_batch`: each distinct address looked up and stored once), so `MemoryManager::rmw_scalar` would only replace
  the loop that serves the general path (`VGPU_FASTPATH=0`), vector forms, shared/local and the faulting cases. The SASS `atomics` workload
  is not in that loop (B/A 0.997x with the PTX change in the build, p 0.67).
* *Local memory layout.* Done for the PTX interpreter, not for the SASS executor; see "Local memory layout and the PTX `local`
  regression" below (the SASS executor's interleaved layout was neutral on `local` and `base64e` and its whole-warp row copies cost 6% on `local`).
* *Bank passes of warps that are not on a stride* (a tile read two rows at a time, a reduction's every-eighth lane) and sector counts
  for 33 to 64 active lanes that are neither on a stride nor monotone still take the earlier code. Measured again on server2 (counting the
  calls of `sass_bank_passes` that fall through `bank_passes_progression` to the table; `VGPU_THREADS=1`): `blockScan 64 1` 22.1 M calls,
  3.2 M (15%) in the table, all 4-byte, 32 lanes, 78% of them one pass; `fpdc 255 8 1 2` 41.9 M calls, 2.1 M (5%) in the table, all one pass
  of 16 lanes; `sortKV 65536 1` 0.63 M calls, 0.14 M (22%), 8-byte mostly unaligned, up to 17 passes; `shmembench 1` 167.9 M calls, none
  in the table. So the table is not where the 3 to 9% is: for `shmembench` and most of `blockScan` and `fpdc` it is the stride test
  (`lane_progression`, `on_stride`) that answers, once per access, which the program cannot avoid asking.
  A tried alternative for the table's share: a 64-bit mask per bank over the 64 rows around the first lane's (bit = row, passes = the largest
  popcount; a pre-test of "every lane in a bank of its own" for 4-byte accesses). Exact (same passes as the table over the cases tried) and,
  alone, 1.5 to 2.5x faster on random, gathered and 8-byte unaligned warps, 2x on a permuted row of one pass, but 0.6 to 0.7x on a two-way
  conflict of 4-byte lanes (the table is already cheap when banks hold one or two words), and a window of 64 rows sends wider warps back to
  the table. Whole programs, A/B alternating 9 runs, one host thread, load average 10 to 24: `blockScan 64 1` 6.40 s vs 6.37 s median
  (p=1.0), `sortKV 65536 1` 0.401 vs 0.404 s (B slower, p=0.002). Dropped: the calls that could gain are 0.6 to 15% of the calls, so even
  a free table would be under 0.5% of a program. Not shipped.
* *Two-section ping-pong.* This entry was stale: the one-entry `Warp::code_hint` already catches what `bh` did. A throwaway build with a
  second entry and counters (first entry hit / second entry hit / `fetch`), over `bh 1500 2`, `bh 2000 2` and 28 other HeCBench programs of
  `reduced.txt` (25 s each, one thread): the second entry was never hit (0 of 8.3e8 lookups in `bh`, 4.4k to 4.6k misses, which are the
  first lookup of each warp); the other programs' misses are one per warp. A synthetic kernel whose lanes split between two `__noinline__`
  device functions and the kernel's own code did not ping-pong either (64 misses of 15 M). A two-entry hint has nothing to catch, so
  it was not shipped. (`bh` itself is not a usable benchmark here: with `VGPU_THREADS=1` it spins for minutes at `bh 2000 2` and above,
  and with the default threads its instruction count differs from run to run, 8e8 to 1e9.)

## Memory lookups: PTX atomics and texels

**Atomics.** `exec_atom` did `load_routed` then `store_routed` for each lane: two address lookups (40% of the PTX `atomics` workload was in
`resolve_scalar`, `scalar_location`, `load_scalar`, `store_scalar`). `MemoryManager::rmw_scalar` resolves the address once (the per-thread
lookup memory of `find_ordinary`), reads the old value, calls the operation and stores. It declines (returns false having done nothing, and
the interpreter then makes the two calls as before, with every diagnostic) whenever the two calls could differ: a stuck cell or a fault armed
for loads or stores, a host mapping possibly in range, a width other than 1/2/4/8, a misaligned address, an access not wholly inside one
ordinary allocation (freed, out of range, a mapped range, `VGPU_LOOKUP_CACHE=0`). The stripe lock, the host-atomic route and the order of
effects are the interpreter's, unchanged; a store of a uniform chunk's own byte still does not make the chunk real.

**Texels.** `texel_channel_bits` did one `load_scalar` per channel. `MemoryManager::load_fields` reads the end-to-end fields of a texel with
one lookup (and a packed 10:10:10:2 texel as one word, not four), and declines under the same conditions, plus a field that is not naturally
aligned or has no load (3 bytes). `tex_finish` and the point fetch use it (`texel_channels`, `VGPU_FASTPATH=0` turns it off); a gather reads
one channel and is unchanged. It reads only the channels the caller would have read.

**Measured**, `tools/bench/bench.sh --ab base new --engines ptx|sass --runs 25 --cpus 24-27`, Release, GCC 13.3, Ryzen 9 9950X (server2, shared:
load 10-50), base = `speedup` at 8f9eb524, counters identical in every row:

| workload | engine | B/A median | B/A best | p |
| --- | --- | ---: | ---: | ---: |
| atomics | ptx | 1.38x | 1.38x | <0.001 |
| texture | ptx | 1.05x | 1.05x | <0.001 |
| texture | sass | 1.06x | 1.06x | <0.001 |
| atomics | sass | 1.01x (noise) | 1.01x | |
| mem, shared, local | sass | 0.99x, 1.01x, 1.06x (all `same`) | | 0.8, 0.8, 0.7 |
| mem | ptx | 0.99x (`same`) | 0.99x | 0.01 |
| shared, local | ptx | 0.96-0.99x, 0.97-0.98x | 0.96-0.99x | <0.001 |

The last row is code layout, not these changes: a build whose only difference from the base is an unused 7-line function in
`interpreter.cpp` (and the new `memory.cpp`), measured the same way, gave shared 0.970x and 0.979x, local 0.983x and 0.971x (two sizes of
dummy), with the harness's own verdict `slower` on three of four. The interpreter is one translation unit whose inlining and function
placement move with any edit; the memory-layer changes alone (base interpreter) gave 0.990x and 0.993x, p 0.15. `texel_channels` is
`noinline` for the same reason (it is not hot enough to need inlining and the large inlined body moved the other workloads). The texture
gain is smaller than the 15% of samples the old code spent, because the sampling arithmetic around the reads dominates.

**Tests.** `tests/unit/test_memrmw.cpp`: `rmw_scalar` and `load_fields` against the `load_scalar` / `store_scalar` calls over random
addresses (inside, at the ends, past them, freed, outside every window, host pointers, misaligned; untouched, uniform and real chunks), with
a stuck cell and with faults on loads and on stores, comparing values, errors and their text, final contents and which chunks became real;
four threads adding under a stripe lock; PTX kernels for every atomic op, width and type (add/and/or/xor/exch/min/max/inc/dec/cas, u/s32/64,
f32/f64, f16, bf16 and their x2 forms) on untouched, uniform and real chunks with several lanes on one word, run with and without the
lookup memory, equal in the values found, memory, chunks made real and every counter, one block and eight blocks on several host threads, and
the out-of-range and misaligned errors; texture fetches in nine formats (including 10:10:10:2 and a 3-byte texel) over real, uniform and
untouched memory, point and linear, with and without the lookup memory. `e2e_sass_fastpath` also runs with `VGPU_LOOKUP_CACHE=0` and requires
the same answers, counters and fault messages.

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
  Launch errors are checked now.

  *Explained later (`speedup4-amdcorrect`).* About 28,000 further `valu_int` quick runs on the MI300X, in loops of 4
  to 16 at once on 4 to 8 CPUs of a box at load 35 to 95, at `VGPU_THREADS` 1, 3, 16 and the default, found no
  race: none gave a wrong answer, and the queue, the worker pool, the decode cache and the launch/synchronize
  hand-off, read for a missed wake-up, an early completion or a stale flag, have none. What did fail, in 3 bursts of
  50 to 400 consecutive runs on server2 while other sessions' builds held 28 of its 30 GB and its swap was full, was the
  launch itself: the simulator's host-memory guard (`VGPU_MEMORY_HOST_RESERVE_MB`, default 6% of RAM clamped to 1 to
  4 GiB) refuses device memory while `MemAvailable` is below the reserve (the message, with `VGPU_QUIET` unset:
  "device out of memory: the host has 618 MB of memory left ... below the reserve of 1840 MB"), the first launch of a
  program writes its linked code object into device memory (`place`), and that refusal came back as
  `hipErrorInvalidImage` ("device kernel image is invalid"), blaming the program. A program that does not read
  `hipGetLastError` after a launch then synchronizes (nothing queued), reads the untouched buffer back, and prints
  MISMATCH with the time of a launch that did nothing: the 0.4 ms. Reproduced on demand with the guard's reserve set
above the host's memory and the launch check taken out of the program: `MISMATCH in valu_int`,
`device_wall_s=0.000427`, `FAIL` (with the check: exit 3, `hipGetLastError() -> out of memory`). The first attempt's
stderr was not kept, so that this was that failure is the explanation that fits every fact, not a proof. Fixed in two ways: a module the host has no room
  for is `hipErrorOutOfMemory` (`load_failure`; test `test_amd_hip_module_oom`, run with the reserve set to more than
  any host has), and the benchmark program checks launch errors (it does since the first attempt above). A
  benchmark on a machine near its memory limit therefore fails loudly now, and the harness repeats the run.
* A counting run per workload, target and build gives exact instruction counts, one host thread, summed over the
  dispatches, from the shim's own probe: `VGPU_DISPATCH_STATS=<file>` (amd/src/hip_api.cpp) appends one line per
  finished launch with the `DispatchStats` numbers the profiler's SQ_* counters are made of, and needs no profiler
  (`--counts-via probe`, the default). `--counts-via rocprof` takes them from `rocprofv3 --pmc SQ_INSTS_* ...` inside
  `vgpu shell -y --gpu amd/<profile> --count 1 --no-isolate -c` instead, and `both` uses the probe's and requires
  rocprofv3's to equal them wherever rocprofv3 ran. Timed runs have no profiler.
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

### The host refusing a thread

`execute()` spreads a grid over host threads and a stream's work runs on a host thread of its own. Where the host refuses to
start one (EAGAIN at a process or user limit, or no address space left; reproduced with `ulimit -v` and `RLIMIT_NPROC`), the
pool's vector was destroyed with a joinable `std::thread` in it, which is `std::terminate` (the program printed "terminate
called without an active exception" mid-launch), and a stream's first `submit` threw `std::system_error` out of
`hipLaunchKernel`, an extern "C" entry point. Now a range of work-groups whose thread is refused runs on the launching thread
(the ranges are independent, so the answers and counts are the same; `start_or_run_here`), the lockstep pool, whose workers
wait on one another, fails the launch with an `Err::Internal` that names the cause, and a refused stream thread is
`hipErrorOutOfMemory` with the stream left usable. Tests: `test_amd_gcn_thread_refused` (an 8-group dispatch at
`VGPU_THREADS=4` under `RLIMIT_NPROC=1` still gives the right answers and counts all 8 waves) and
`test_amd_hip_thread_refused`; both say SKIP and pass where the process is exempt from the limit (root). Mutation checks: with
the fallback removed the first dies with the `std::system_error`, with the catch removed the second aborts, with the DS guard
removed `test_amd_gcn_same_bits` segfaults.

### The counter check

`--check-counters` runs the workloads at the `quick` size on one host thread, counted by the shim's
`VGPU_DISPATCH_STATS` probe (and, where rocprofv3 is there and the shim is not sanitizer-built, by rocprofv3 too, which
must agree: `--counts-via both`, what the ctest does), and compares the `SQ_INSTS_*`, `SQ_WAVES`, `SQ_INSTS_VMEM_RD/WR` and `TA_FLAT_ATOMIC_WAVEFRONTS` totals,
per workload and target, with `tools/bench-amd/expected/amd-hip-7.2.26015.json` (counts depend on
the compiler, so one file per hipcc release). It is the ctest test `bench_amd_counters` (about 6 s),
and skips (`SKIP:`) without a ROCm that has `hipcc` or without an expected file for the release. A sanitizer-built
shim, or no rocprofv3, no longer skips: the probe counts alone are compared (a NOTE says the rocprofv3 cross-check
was not made). The counts at one
thread are deterministic, including the `atomics` CAS loop: `--update-expected` followed by two
`--check-counters` matched all 21 workload/target pairs. Mutation check: raising one stored count
by one makes it exit 1 with `mi300x/valu_int: SQ_INSTS_VALU expected 666625, got 666624`.

*Why rocprofv3 "wrote no counter CSV" on server2.* `vgpu shell` puts its own `rocprofv3` first on PATH (a wrapper that
runs ROCm's tool with `--rocm-root` pointing at VirtualGPU's `librocprofiler-sdk.so.1`), and writes that wrapper only
when `shim/librocprofiler-sdk.so.1` exists, i.e. when the `vgpurocprof` target was built. A build of only `vgpu` and
`vgpuhip` (what a quick benchmark build is) has no wrapper, so the bare `rocprofv3` of the counting run is ROCm's own: it
loads ROCm's profiler SDK over the simulated agents, finds no counters, writes nothing at all, and the program still
passes (rc 0, `PASS`), which is why the harness could say only "no counter CSV". Reproduced on server2 by removing
the library from a copy of the shim (no file written), and by running `/opt/rocm/bin/rocprofv3` by path inside the
shell (the same). With `vgpurocprof` built, rocprofv3 wrote its CSV in all of about 2,500 counting runs under load
(4 at a time, 1 and default threads), and the harness counted all 21 workload/target pairs identically to
`expected/` by both routes (`counters match ... counted by both, rocprofv3 agreed on 21`). The harness now says which
library is missing, and its default counts need no profiler, so "counters identical" is a real check on any host.

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

### Ranked wins 1, 2, 4 and 8 applied (`speedup2-amdops`)

Branch `speedup2-amdops` from `speedup` a12f8d84. Same machine, harness and pinning as above (server2, CPUs
26-31, `VGPU_THREADS=1`, 11 alternating runs, Mann-Whitney p; load average 4.7 to 26 per workload, median 8);
base is a12f8d84 built Release in its own directory, B the branch. Counters identical on every row
(`--check-counters`: 21 of 21 pairs match).

| workload | target | median ratio | best ratio | p |
| --- | --- | ---: | ---: | ---: |
| valu_int | mi300x / rx6800 | 2.11 / 2.59 | 2.16 / 2.58 | 0.000 |
| valu_f32 | mi300x / rx6800 | 2.11 / 2.82 | 2.04 / 2.97 | 0.000 |
| valu_f64 | mi300x / rx6800 | 1.89 / 2.66 | 1.93 / 2.67 | 0.000 |
| mem | mi300x / rx6800 | 1.51 / 1.90 | 1.51 / 1.92 | 0.000 |
| lds | mi300x / rx6800 | 2.24 / 2.33 | 2.28 / 2.33 | 0.000 |
| atomics | mi300x / rx6800 | 1.86 / 2.34 | 1.84 / 2.35 | 0.000 |
| divergent | mi300x / rx6800 | 2.51 / 3.03 | 2.50 / 3.12 | 0.000 |
| wave | mi300x / rx6800 | 1.68 / 1.81 | 1.68 / 1.94 | 0.000 |

What changed:

* **Win 1, the ladders.** `amd/tools/ladder-to-switch.py` rewrote every `if (op == "a"_op || ...) ... else if` chain of
  `gcn_exec.cpp` whose rungs test only names (783 case labels, 502 rungs, 30 chains; a chain with an impure rung in the
  middle keeps that rung as an `if` and converts the pure runs either side) into `switch (op.hash())`. The case keeps the
  rung's original condition, tested verbatim, so a hash collision still falls to the else-branch as the text comparison did;
  `if` statements in a row count as one chain only where each body returns. The SOPP tail of `Machine::step` (about 30
  rungs, each constructing and hashing an `OpName`) is a hand-written switch in the ladder's order. A first A/B of this alone
  (`valu_int` 1.55x, `lds` 1.93x, `divergent` 2.17x, `mem` 1.20x, p <= 0.003) is what the sample shares of the profile above
  said could be had; the rest of the table adds wins 2 and 4.
* **Win 2, decode-time facts.** `gcn::finalize` (called at the end of `gcn::decode`, halves of a VOPD included) fills in
  `Inst::name_hash`, the short form's name and hash that `vector_alu` built as a `std::string` on every instruction, and
  `InstFlag` bits for the `rfind`/`find` tests of `step` (`v_mfma`, `v_cmp`, `f64`, `_store`, `_atomic`, `ds_gws_`, ...). An
  `Inst` not decoded this way (`flags == 0`) works them out as before. `rdna_alu` and `cdna5_alu` take a view of the name
  instead of copying it.
* **Win 4, lane-invariant sources.** `InvariantSources` reads an SGPR, inline constant or literal source once per
  instruction (in `vector_alu` and `compare`) and `lane_src` returns it from a four-entry table; the global-access loop reads
  its scalar base once. Alone (A = the branch with the table switched off, 11 runs, MI300X): `valu_int` 1.37 (p 0.000),
  `valu_f32` 1.12 (p 0.001), `mem` 1.21 (p 0.010), `wave` 1.09 and `divergent` 0.96 (not significant).
* **Win 8, POPCNT: dropped.** A `target("popcnt")` copy of the `v_bcnt_u32_b32` lane loop: median ratio 1.013 on `valu_int`
  (p 0.61), 1.004 to 1.018 on the others (p 0.056 to 0.70). Under the 2% bar and not significant; removed.

The same-bits test is `test_amd_gcn_same_bits` (`amd/tests/unit/test_gcn_same_bits.cpp`, ctest-wired as a unit test): every
instruction shape of the seven ISA corpora and of a SOPP corpus (15,566 shapes, `amd::digest_instruction` in
`gcn_exec.cpp`) runs once on a wave of pseudo-random registers, LDS, scratch and memory in four ways (any bits, small
integers, floats and doubles with the special values among them, half the lanes off; branches are sent to a distant target)
and a digest of all registers, masks, LDS, memory, counts and error text is compared with
`amd/tests/data/same_bits_digests.txt`, which was recorded from the unmodified a12f8d84 plus only the probe. Mutation
checks (each makes it fail): two case labels swapped (123 shapes differ), the invariant table off by one (3,088), a wrong
decode flag (1,527), `s_cbranch_scc0` inverted (6). Instructions that read the clock are recorded as unstable and skipped;
`ds_nop`, `ds_consume` and `ds_append` are skipped because the interpreter crashed on them with the fuzz corpus's operands:
the decoder gives these three no address operand (`Inst::src` is empty), and `lds_access` read `in.src[0]` off the end of an
empty vector (a SEGV in `lane_src`, found with a signal handler on the base commit f81cc0c1: all three shapes of the
gfx1250 corpus die there). They stay out of the digest, because the base crashes and so no base digest of them can exist;
`ds_nop` now does nothing (it counts as an LDS instruction) and `ds_append`/`ds_consume` throw `Err::Unsupported` ("has
no address operand and is not modelled"; the wave-wide counter at M0 is not documented well enough to model), checked by
`ds_instructions_without_an_address_fail_gracefully` in the same test file on RDNA2, RDNA3, RDNA4 and gfx1250.

Two things the digest caught that a pass over the ladders would not have: GCC expands `floor`, `ceil` and `trunc` in line
where it thinks the code hot (a signaling NaN then comes back unchanged; libm returns a quiet one), and which places those
are moved with the switch. The vector instructions (`v_floor/ceil/trunc_f32/f64`) were in-line before and are now
`floor_inline` and friends with that behaviour spelled out; the scalar ones were libm and `-fno-fp-int-builtin-inexact`
keeps them so. Likewise which of two NaN operands an `fma` or a double add propagates follows the order the compiler gives
its operands: `v_fma_mix*_bf16` and `v_dot2c_f32_f16` needed their operands written in an order (and, for the dot product,
held there with an empty `asm`) to give the bits the old build gave. Other sites with two NaN operands of different
payloads (`fmin`/`fmax`, the other `fma`s) may be compiled differently now; the corpus run with these inputs found none
beyond those two, but it cannot prove there are no more.

Tests run on server2 against the branch: `test_amd_*` (all 38 unit binaries that run there: pass), ctest `amd|gcn|rocprof|
hipcc|hip_` (all pass or skip but `amd_rocprofiler_abi` (rocprofiler-sdk header version), `amd_rocprofv3` and `amd_tools`'
user-namespace steps, which fail for the host, not the change), `bench_amd_counters` (21 of 21), PyTorch tests skip there.

### Not done, and not measured here

* The `std::optional<Inst>` change (X4) and the cached hash (X5) are negative results and stay out; X1 and X6 are inconclusive or small.
* Loops over 64 lanes with a mask test (wave32 pays for 32 masked lanes), `ctz` lane iteration: not measured.
* RDNA3/4 `v_wmma_*`, other MFMA shapes, gfx1250: not timed here; see "The AMD matrix instructions, every form" (branch `speedup4-amdmatrix`).
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

This list is the baseline's; later rounds measured some of it, as noted.

* Host instruction counts and IPC (no PMU on this VM; unknown on hosted runners).
* `VGPU_THREADS` > 1 scaling in the baseline (harness supports `--threads`); later host-thread measurements
  are in [perf-design/host-parallelism.md](perf-design/host-parallelism.md).
* Other GPU profiles / architectures in the baseline (the program is compiled for sm_86); the SASS engine
  now runs every generation from sm_75 to sm_120a, and [perf-design/round10-real-workloads.md](perf-design/round10-real-workloads.md)
  runs programs on their own generation's profile.
  The AMD engine has its own section, "The AMD engine".
* The pantheon soak workloads outside the optional mode above, and the PyTorch/cuBLAS paths in the baseline.

## Tier 1 together (branch `speedup`: tier on by default, PTX atomics and texel reads, AMD baseline)

Measured on server2 (Ryzen 9 9950X, 32 logical CPUs, load 1.4 to 2.8 during the runs, CPUs 24-27, one host thread, `nvidia/rtx3060`,
CUDA 13.0.88, Release), `bench.sh --ab` base (`8f9eb524`) against the merged tree, 9 runs (SASS) and 11 runs (PTX), alternating. Counters
are identical in every row.

SASS engine (the default), B/A of the median, all p < 0.001 unless noted:
int_alu 2.75x, f32_fma 2.30x, f64_fma 1.51x, f64_subnormal 1.45x, wmma 1.09x, mma_ptx 1.10x, mem 1.41x, shared 1.39x, atomics 1.28x,
texture 1.12x, local 1.92x, divergent 1.94x, warp 3.38x, launch 1.21x; api 0.99x (p 0.014, under 2%: same) and startup 1.01x (same).

PTX engine (VGPU_SASS=0 and PTX-only modules): atomics 1.38x, texture 1.06x, mem 1.01x (same), shared 0.99x (same), **local 0.93x
(slower, p < 0.001)**. The PTX `local` regression comes with the memory-lookup change (`rmw_scalar`, `load_fields`); the branch that made
it measured 3% on a loaded machine and attributed it to code layout, but on a quiet one it is 7%. It is kept because the SASS engine is the
default on every NVIDIA GPU and PTX atomics gain 38%; the PTX local-memory path is the next section's subject (the cause, and a rewrite of the PTX local array that more than makes it up).

Dropped after measurement: the two-entry code hint and the bank-pass rewrite (see "Not done, and negative results"). On the AMD side the
harness (tools/bench-amd) counts a Radeon card's instructions with SQ_WAVE32_INSTS + SQ_WAVE64_INSTS, the counters a simulated RX 6800
offers.

## Tier 2: what the tier still hands back to the interpreter (branch `speedup2-tiercov`)

**Measured first.** `VGPU_TIER_STATS=2` (it already prints, per mnemonic, the straight-line instructions the interpreter ran and the ones that
ended a run and were the interpreter's) over the bench workloads (`quick` size) and over the corpora: HeCBench's `reduced.txt` (34 programs), the
Rodinia apps of `run.sh`, and the CUDA Samples of `exes.txt` (85 programs gave statistics; the others fail without a card or are not CUDA programs),
every one with the base build `a12f8d84`, `VGPU_THREADS=1`, the RTX 3060 profile. The tier ran 97.3% of the corpora's 5.98 G warp instructions
(HeCBench 97.2%, Rodinia 99.1%, Samples 97.4%); 0.34% were straight-line instructions with no handler and 2.31% ended a run in the interpreter.
Bench workloads: 100% for `int_alu`, `f32_fma`, `f64_subnormal`, `local`, `divergent`, `warp`, `launch`, `wmma`, `mma_ptx`; 99.8% `f64_fma`
(`I2F.F64.U32`), 99.5% `mem` (`UIMAD.WIDE.U32`), 98.9% `shared` (`P2R`, a waiting `EXIT`), 91.0% `texture` (`TEX`, `TLD`), 81.3% `atomics`
(`ATOMG.CAS`, `RED`, `REDUX`, `VOTEU`, `S2UR`). So the chain-ending instructions named in the notes above (divergent branches 35% of `divergent`'s
runs, barriers 25% of `shared`'s) are already the tier's: they end a *run* and return to the scheduler, they are not handed to the interpreter.

What is left, by dynamic count over all three corpora (share of the 5.98 G; the tier cannot do much about a count this small):

| instruction (mnemonic.modifiers) | warp instructions | share | where |
|---|---|---|---|
| IMMA.16816.U8.U8 | 33.6 M | 0.56% | one CUDA Sample (immaTensorCoreGemm) |
| LDSM.16.M88.2 / .4 | 12.6 M each | 0.21% each | the same sample |
| BRA.CONV | 11.5 M | 0.19% | atomicAggregate (3.1 M), CUB-style code |
| FCHK | 10.6 M | 0.18% | black-scholes 8.2 M, bilateral 1.2 M, silu |
| CALL.REL.NOINC / RET.REL.NODEC | 10.4 M / 8.0 M | 0.17% / 0.13% | the IEEE-division slow path of the same three programs |
| F2F.F64.F32 | 7.3 M | 0.12% | |
| ATOMS.CAST.SPIN.64 | 6.0 M | 0.10% | aop 4.4 M, atomicPerf 1.6 M |
| MATCH.ANY / .ANY.U64 | 4.1 M / 3.1 M | 0.07% / 0.05% | atomicAggregate |
| REDUX.OR, P2R, VOTEU.ANY, S2UR, RED.E.ADD | 3.4 to 4.1 M each | 0.06 to 0.07% each | atomicAggregate, atomicIntrinsics, radixsort2 |
| everything else (ATOMS.POPC.INC, ATOMG, R2UR, REDUX.MAX, BRA.DIV, CALL.ABS, ATOMS.ADD, R2P, MUFU.RCP64H, WARPSYNC, ...) | | under 0.04% each | |

No texture, BRX, generic LD/ST, 32-byte access, Turing global form, FRND or MUFU.RSQ64H stands above 0.01% in these corpora (TEX/TLD are 9% of the
`texture` bench workload only).

**What was kept: handlers for MATCH, IMMA and LDSM.** Each was chosen from the table above and kept because its A/B, on a program that runs it,
shows more than 2% with p <= 0.05. Each is a `tier::Form` with a handler in tier_ops.inc (form table entry O: integers and data movement, no number
format), the same results as the interpreter's code, and a fallback to `exec_mem` / the interpreter for anything unusual.

* `Form::Match` (`MATCH.ANY` / `.ALL`, 32 and 64 bits). The interpreter's code compares every lane's value with every other's through nested
  lambdas (about a thousand comparisons for a warp); the handler reads the lanes' values once and works out the lane mask of one equality
  class at a time, giving it to every lane of the class (`.ANY`), or compares each value with the first lane's (`.ALL`).
* `Form::Imma` (dense `IMMA`, 8- and 4-bit elements of both operands, s32 accumulator, plain or `.SAT`). `exec_mma` builds `std::vector<double>`
  matrices on the heap for every call and sums in doubles and int64; the handler gathers the fragments into two int16 matrices (B kept
  column by column), sums the products in 32-bit wrapping integers (the same bits as the interpreter's int64 sum cut to 32 bits; under `.SAT`
  in 64 bits, clamped to the s32 range after every 128 bits of K, as the interpreter does) and writes the executing lanes. Sparse IMMA (`.SP`)
  and the non-integer MMAs stay the interpreter's.
* `Form::Ldsm` (`LDSM` of 16-bit elements: `.M88`, `.MT88`, x1, x2, x4). A whole warp whose rows are all inside the block's own shared memory
  is moved straight into the registers (the window test of `h_shared`: another block's window, a row that could end past the array, and a partial
  warp, for which the interpreter reports a missing row address, all go to `exec_mem`). The 8-bit, 4-bit and 6-bit packed forms stay the
  interpreter's. The counters (`shared_loads`, `shared_bytes_read`) are the interpreter's.

| workload (what runs the instruction) | A | B | B/A median | B/A best | p | load |
|---|---|---|---|---|---|---|
| HeCBench atomicAggregate `main 1`: same build, `VGPU_TIER_OFF=match` (A) / not (B), 15 runs | 7.30 s | 5.19 s | 1.41x | 1.39x | <0.001 | 14 to 27 |
| the same, base build `a12f8d84` (A) against this build (B), 15 runs | 9.66 s | 5.26 s | 1.84x | 1.46x | <0.001 | 27 |
| CUDA Sample immaTensorCoreGemm: same build, `VGPU_TIER_OFF=imma` (A) / not (B), 5 runs | 57.2 s | 20.6 s | 2.78x | 2.83x | 0.009 | 7 |
| immaTensorCoreGemm: `VGPU_TIER_OFF=ldsm` (A) / not (B), IMMA on in both, 5 runs | 20.44 s | 17.52 s | 1.17x | 1.17x | 0.009 | 10 |
| CUDA Sample cudaTensorCoreGemm (HMMA): `VGPU_TIER_OFF=ldsm` (A) / not (B), 5 runs | 39.28 s | 35.35 s | 1.11x | 1.12x | 0.009 | 14 |

The three forms are 66.0 M of the corpora's 5.98 G warp instructions (1.1 points): the tier's share goes from 97.3% to about 98.4%.

(User plus system CPU seconds of the process, alternating runs, CPUs 16-19, `VGPU_THREADS=1`; 5 runs is the least that can reach p 0.009.) The
same-build rows remove any layout effect of the changed binary. MATCH is 3.8% of atomicAggregate's warp instructions and 29% of its time;
IMMA is 62% of immaTensorCoreGemm's warp instructions and about two thirds of its time.

**Tried and dropped, by the rule (an A/B that does not show more than 2% with p <= 0.05 on a workload that runs the instruction):**

* *A "direct" dispatch for the instructions that end a run and have no handler.* `tier_run` called `exec_mem`, `exec_warp`, `exec_control`,
  `exec_tex` or `exec_int` itself for ATOMG/ATOM/ATOMS/RED, REDUX, VOTEU, TEX/TLD, CALL/RET/BRX/WARPSYNC, BRA.DIV/.CONV and R2P/S2UR/R2UR, with the
  counters and the advance of `execute()` and without its guard, `fast_alu` probe and tracing check. It passed `e2e_sass_tier` and `test_tier`.
  The same build with `VGPU_TIER_OFF=direct` against without, 11 alternating runs (CPUs 28-31, load 11 to 28): atomicAggregate 1.089x median
  (best 1.027x, p 0.18), aop 1.031x (best 1.036x, p 0.053), atomicPerf 1.019x (best 1.032x, p 0.58); `bench.sh --ab` base against the build with it,
  11 runs: `atomics` 1.008x (p 0.047), `texture` 1.005x (p 0.047), both "same". These instructions spend their time in the memory manager's locks,
  `texture_fetch` and the lane loops, not in `execute()`'s prologue. Removed; nothing to keep.
* *A handler for FCHK* (the IEEE division's check; the interpreter answers "slow path" for every lane, so it is one predicate write). Same
  build, `VGPU_TIER_OFF=fchk` (A) against not (B), 15 alternating runs, CPUs 28-31, load 8 to 29: black-scholes 22.88 s vs 22.60 s, 1.012x (p 0.017);
  bilateral 1.736 s vs 1.705 s, 1.018x (p 0.011). Under the 2% line. Removed. (`sass_tier.cu` now runs an IEEE division so the
  FCHK, CALL and RET paths are held to the interpreter whatever is done to them later.)
* *CALL/RET, BRA.CONV/.DIV, REDUX, VOTEU, shared and global atomics, RED, TEX/TLD as handlers of their own.* The "direct" experiment above is
  the upper bound of what a handler for them can save when it only resolves operands once (the interpreter reads a register operand inline,
  so there is little to resolve); no handler was written for them. MATCH was the one whose interpreter code does more work than its
  result needs.
* *Whole-suite A/B of the kept changes.* `bench.sh --ab` base `a12f8d84` against this build, all SASS workloads, `--size full`, 9 runs,
  CPUs 16-19, load 3: counters identical in all, every verdict "same" (medians 0.98x to 1.01x, `wmma` and `mma_ptx` 0.998x and 0.997x;
  `launch` 0.970x median with p 0.008 but 0.988x best, `startup` is 2 ms of wall: code-layout noise of the kind this file documents
  elsewhere, not claimed either way). An earlier run of the MATCH-only build at load 20 had `mem` 1.035x and `launch` 1.028x "faster", which
  no MATCH explains.

**Same results with the tier off and on.** `tools/verify/run_corpora.sh` with `A` = the base build, `B` = this build, `T0` = this build with
`VGPU_TIER=0`: 28 of HeCBench's 34 programs (the other six, atomicReduction, babelstream, base64e, bh, binomial-small and atomicCost, time out
at 120 s in the base build, so `--skip-timeouts A` left them out) and Rodinia's 12, each run three times, and the CUDA Samples that run the new forms or the instructions around them (immaTensorCoreGemm,
cudaTensorCoreGemm, bf16TensorCoreGemm, warpAggregatedAtomicsCG, reductionMultiBlockCG, mergeSort, cdpAdvancedQuicksort, shfl_scan,
binaryPartitionCG, simpleAtomicIntrinsics, with a 500 s timeout). Exit status, stdout, stderr, output files and per-launch counters are
identical between B and T0 and between B and A except: `rodinia/backprop`'s `output.dat` (three different hashes in A, B and T0: the program is
not deterministic), `samples/UnifiedMemoryStreams` (host threads print in their own order), `atomicCost-cuda` (a stack address in an error
message), `samples/immaTensorCoreGemm` (a `TOPS:` rate line, 0.01 against 0.00 because B is 3x faster), and a `black-scholes-cuda` run of B
that the shared machine's memory ended with out-of-memory, which ran identically (B = T0 = A) when repeated.

**How the kept handlers are held.** `sass_tier.cu` gained `k_direct` (MATCH.ANY/.ALL/.ANY.U64 in whole and in diverged warps, blocks that are not
whole warps (128, 48 and 33 threads), and, beside it, the instructions the interpreter keeps: global and shared atomics of every kind and width,
RED, REDUX of every operation, the warp-aggregated increment's VOTEU/BRA.CONV, indirect and direct calls, `__syncwarp`, an IEEE division; each
block uses bins of its own so that four host threads give the one-thread answer), `k_imma` (mma.sync s8/u8/s4/u4 in m8n8k16, m16n8k16, m16n8k32,
m16n8k64, plain and `.satfinite`, on registers of random bits) and `k_ldsm` (ldmatrix x1, x2, x4, plain and `.trans`, rows shared between lanes).
`e2e_sass_tier` (every program run with `VGPU_TIER=0` and `1`) holds every result and counter to the interpreter's, also under four threads and
without the chain. `test_tier` requires MATCH to be taken in every corpus (sm_80, 86, 89, 90, 120), checks the operand rows and register windows
of IMMA and LDSM over every instruction of the disassembler corpora, and lists `match:O ldsm:O imma:O` in the form table. Mutation checks (a
scratch copy of tier_ops.inc, `e2e_sass_tier` on sm_86, each one caught by `sass_tier`): MATCH.ALL's predicate inverted, MATCH's 64-bit high word
ignored, MATCH.ALL's result written to all lanes; IMMA's A and B signedness exchanged, IMMA.SAT's clamp removed, the clamp interval doubled,
the accumulator dropped under `.SAT`; LDSM's transposed column index changed, `shared_loads` counted 31 for 32 lanes, the two halves of a
transposed word exchanged.
## Local memory layout and the PTX `local` regression (branch `speedup2-localmem`)

Measured on server2 (Ryzen 9 9950X, shared with other jobs: load 6 to 35 during the runs, so every row is an alternating A/B of 31 runs
(`bench.sh --ab`, 15 for the HeCBench program) on CPUs 22-23 whose SMT siblings were idle (checked with `mpstat` before each batch), one host
thread, `nvidia/rtx3060`, CUDA 13.0.88, Release with LTO, GCC 13.3). Builds: `base` = `8f9eb524`, `head` = `speedup` at `a12f8d84`,
`new` = this branch. Counters are identical in every row (the `counters` column of `bench.sh --ab`, and `VGPU_COUNTERS=1` output of the
HeCBench program diffed on both engines).

### Where the PTX `local` regression came from: code placement

The 7% (3% on a loaded machine) that arrived with `26e5d708` (`rmw_scalar`, `load_fields`) is not in any instruction the `local` workload runs:

* `libcudart.so` built at `c8ace690` (before) and at `8f9eb524` are byte-identical (`cmp`); an A/B of the two nevertheless read 0.967x median
  (best 1.005x, p < 0.001) once and 0.999x the next time. The harness's p-value does not see drift that is shared by the runs of one
  batch: a single batch on this machine is good for about +-3%, and the 7% of the Tier 1 text was one such batch. Alternating A/B of `base`
  and `post` (`26e5d708`), 41 runs each: post/base 0.895x (a noisy batch), 0.966x, 0.988x, and with A and B swapped base/post 1.030x, 1.030x, 1.028x: `post` is about 3% slower.
* The hot functions of the PTX `local` run, `exec_ld`, `exec_st`, `run_warp_until_yield`, `fast_int_bin` and `check_local`, are
  instruction for instruction the same in the two builds: `objdump -d` with addresses and call targets masked, 1885, 1557, 2249, 114 and
  127 lines, 0 lines differ. Their sizes are equal. They sit 0x660 bytes later (32 bytes modulo the 64-byte cache line:
  `exec_ld` 0x27c1c0 -> 0x27c820), because the texel and atomic code before them in the one LTO image grew.
* Putting 64-byte alignment on those three functions (`__attribute__((aligned(64)))`) took `post` from 0.97x to 0.987x, 0.986x, 0.990x of
  the base on `local`. On the tree of this branch the same pins gave `local` 1.014x, 1.007x but `shared` 0.957x and `mem` 0.981x: a
  different placement is faster for one workload and slower for another, so the pins are not kept. `-falign-functions=64` on all of
  `interpreter.cpp` (and, in a separate build, on all of `sass/exec.cpp`: SASS `shared` 0.93x, `local` 0.965x) was no better.
* The same holds the other way round: a build that changes only `interpreter.cpp` (this branch without its SASS edits; the SASS source is
  `head`'s) moves the SASS engine's `shared` to 0.93x and 0.935x of `head`, and the final tree's SASS `local` / `shared` read 0.964x /
  0.967x of `head` with no SASS source change at all (below). Any edit to the interpreter moves the placement of everything after it.

So there is no code to fix in `rmw_scalar` or `load_fields`, and nothing that a source change can promise to keep: the PTX `local` loss is a
property of where the linker happened to put code. What this branch does instead is make the PTX local-memory path itself cheaper, which
moves the workload far past the old 3%.

### The PTX local array, interleaved by words (`include/vgpu/exec/local_layout.hpp`)

`Warp::local` of the PTX interpreter was `std::vector<std::vector<uint8_t>>`: a vector per lane, each grown to the current frame on
its own, and every access went `lane_local(w, lane)` (an emptiness test, a `local_top()`, a size test) then a pointer chase to the lane's
heap block. It is now one array, `local_array(w)` (one size test per warp access, not per lane), laid out as the hardware does: byte `o`
of lane `l` at `(o / 4) * 128 + l * 4 + o % 4`. A warp-wide 4-byte access at one offset is 128 contiguous bytes. Growing the array for a
callee's frame is appending zero words (rows already there stay where they are). Every user of `Warp::local` in the PTX interpreter
goes through `exec::local_load` / `local_store` (fast paths: `exec_ld`, `exec_st`) or `load_routed` / `store_routed` (the general path);
the checks (`check_local`, the frame bounds, the alloca stack, which has its own per-lane buffers) and every message are unchanged, so
faults, out-of-range texts with offsets and counters are the same. Nothing else reads the array: no debugger, checkpoint or inspection
code touches `Warp::local`.

PTX engine, B/A of the medians (> 1: new is faster), p from the harness's Mann-Whitney test:

| workload | new / base (8f9eb524) | p | new / head (a12f8d84) | p |
| --- | ---: | ---: | ---: | ---: |
| local | **1.094x** | <0.001 | **1.141x** (best 1.098x) | <0.001 |
| shared | 1.004x (same) | 0.12 | 1.049x (best 1.005x, same) | 0.004 |
| mem | 1.007x (same) | 0.19 | 0.983x (same) | 0.89 |
| atomics | 1.375x | <0.001 | 0.983x (same) | 0.12 |
| texture | 1.119x | <0.001 | 1.041x (same) | 0.16 |
| divergent | 0.992x (same) | 0.57 | 1.011x (same) | 0.25 |
| HeCBench `base64e` (reduced: 1,572,864 elements, 4 cycles) | | | **1.198x** (best 1.204x) | <0.001 |

PTX `local` is back above the base (1.094x, with the atomics gain of 1.375x intact), and `base64e` is 1.20x. Other batches of the same comparison read `local` 1.09x to 1.11x of the base (three batches) and 1.11x to 1.18x of head (six), and
`base64e` 1.198x to 1.217x (five).

Why it wins: not cache lines. The warp's 8 KB of touched local memory is L1-resident in either layout; what was removed is the per-lane
indirection and the per-lane `lane_local` bookkeeping.

### Not done: the same layout in the SASS executor

The SASS executor (`Warp::local` in `src/sass/exec.cpp`, the loop in `exec_mem`, the tier's `h_local`, `mem_read` / `mem_write`) was
converted in a scratch build too, with the same helper, and with a second step on top: when all 32 lanes access one offset (a spilled
register, a constant index) the access is a 128-byte `memcpy` between the array and the register file instead of 32 lane moves. It does
not win, so it is not in this branch:

| SASS, `head` vs scratch build | local | base64e |
| --- | ---: | ---: |
| interleaved array, lane loops only | 0.997x, 0.999x (same) | 0.991x (p 0.007, same) |
| plus whole-warp row copies | 0.942x (p < 0.001, slower) | 1.001x, 1.028x, 1.045x (p 0.26, 0.03, 0.05) |

`local` indexes by lane-dependent values, so no warp-wide row exists there and the branch and the larger handler (`h_local` 1.4 KB ->
2.5 to 2.9 KB once each width has a row-copy and a lane-loop body) cost 6%; `base64e` stores its 65-byte table with 65 one-byte `STL`s at constant
offsets, the case the row copy is for, and gains at most 4% (not significant on a quiet run). The doc's expectation that touching 32 cache lines
was the cost did not hold: the working set fits in L1. The SASS executor keeps one frame per lane.

### Tests

* `tests/unit/test_locallayout.cpp` (`LocalLayout32Lanes`, `LocalLayoutOtherWarpWidths`): 60,000 random stores and loads per seed, of 1 to 16
  bytes at every offset (also across words), in random lanes, against a plain array per lane (the old layout), the array grown four times
  mid-run, then every byte read back and "a warp-wide word is contiguous" checked; 32, 64 and 8 lanes. Nine mutants of the header (a
  wrong row stride, the lane term, the in-word offset of a store and a load of 1 byte, the high word of 8 bytes in store and load, a 2-byte
  access at offset 3, the chunk length of the byte-wise copy, a 4-byte access at offset 1) are all caught (one by an abort).
* `nvidia/tests/e2e/sass_vector_memory.cu` gains `k_alias`: local memory written at one width and read at others (bytes, halves, words
  and pairs at lane-dependent places over a byte-filled array, and only some lanes), a callee handed a pointer into its caller's frame, and a
  recursive function whose per-call arrays (indexed by the seed, so they stay in local memory) must survive the callee's frames; the host
  runs the same body as the reference, so the layout is not in the oracle. `run_sass_vector_memory.sh` now also runs the SASS executor and
  the PTX interpreter with `VGPU_FASTPATH=0`, which are the only runs that reach `Runner::mem_read` / `mem_write` and the PTX
  `load_routed` / `store_routed` for local memory. The test passes on `head` too (it does not know the layout). A recursion that
  grows the array under live data was missing from the tests before: the PTX interpreter logged 96 -> 144 -> 192 -> 240 -> 288 growths on it.
* Mutation check of the PTX paths (each a one-line change, rebuilt and run through `run_sass_vector_memory.sh`): the fast load (a wrong
  element offset), the fast store (the element offset dropped), the routed load (+1), the routed store (low byte cleared) and the growth
  (the array replaced by a zero one) are each caught, by the `ptx` run (fast paths) or the `ptx-general` run (routed paths).
  The mutants of the discarded SASS half (fourteen of them, rows and lane loops of `exec_mem` and the tier, `mem_read`, `mem_write`) were all
  caught by their engines too.
* Run: the new unit test, all 208 `test_*` tests, `e2e_sass_vector_memory`, `e2e_sass_fastpath` and the 61 tests whose names contain
  sass, tier, ptx, fastpath, exec, fault, lockstep, local, vector, debug, diag, race, trap, assert, stack, call, device_functions,
  memcount, memfast or window: all pass but `e2e_lockstep_blocks`, which times out on a loaded server2 on `head` and `post` as well
  (passed and failed in alternate runs of the same build). `bench.sh --check-counters` is skipped on server2 (its nvcc builds other
  device code than the recorded counters), so the counters were compared between builds instead: `bench.sh --ab` over every workload
  and both engines (`counters identical`), and the HeCBench program's `VGPU_COUNTERS=1` output.

### Reading these numbers

The SASS rows of the same final tree against `head` (SASS source unchanged; interpreter edits only): local 0.964x, shared 0.967x, mem
1.027x, atomics 0.992x, texture 0.988x, divergent 0.994x, int_alu 0.996x, `base64e` 1.019x. The first two are the placement effect
described above (the same sources measured 0.998x and 0.93x in two builds that differed only in the PTX file). A merge that adds code
elsewhere changes them again; judge SASS changes on the merged tree, with an A/A of two copies of the same build next to the A/B.
## Tensor cores: the sm_86 chunked sum, faster (branch `speedup2-tensor`)

The card-measured sum of an RTX 3060's tensor core (`include/vgpu/exec/tc_ampere.hpp`, `VGPU_TC_EXACT`, on by default for sm_86) cost
3.5-4.5x over the plain sequential f32 model. Measured on server2 (`bench.py --ab`, base `speedup` a12f8d84 against this branch, CPUs
pinned, one host thread, `nvidia/rtx3060`), `VGPU_TC_EXACT=1` against `=0` on the base build: SASS wmma 0.199 s against 0.058 s of CPU, mma_ptx
0.102 s against 0.032 s. A sampling profile of the SASS `wmma` row (SIGPROF, `-g` build) put 66% of its samples in the vector kernel
(`mma_avx2_chunked<8>`) and 17% in the `hmma_k` wrapper around it: the model's own kernel, not the dispatch around it, is where the time goes.
The kernel alone, m16n8k16 f16, on one pinned core of a loaded box (a micro-benchmark that calls it 100,000 times; loads vary run to run,
so only same-process comparisons are quoted): 668-690 ns before.

What changed, all with the same result bits (the first is the test's oracle, a frozen copy of the earlier header):

* **Chunk length as a template constant** (8 or 4 products), so the product loops unroll.
* **Frame terms with a sentinel**: an operand that is zero gets a term of `-2^28` at decode time, so the largest frame position is a
  plain `max` of `ta + tb` per product instead of `mullo`, compare and blend; an all-zero chunk needs no special case.
* **One variable shift per product**: A's significands are kept times 16, so a product is already its magnitude `<< 4`, and the aligned term is
  that shifted right by `top - T` (truncation toward zero), computed in registers that do not spill (the products are formed again in the second
  pass, which measured faster than keeping sixteen vectors live: 566 -> 524 ns).
* **The sum's bit length from one float conversion** of its magnitude with the low 7 bits cleared (24 significant bits at most, so exact),
  replacing two double conversions and three permutes; the new accumulator stays in its register, and the lanes the vector code does not finish
  (an infinity or a NaN in the chunk, a result outside the f32 normal range) are the only ones that go through memory.
* **Four output rows at a time, stage by stage**: the 16 rows of an m16n8 tile are independent chains of about 60 cycles each, and one at a
  time the out-of-order window held about one. Rows per pass, same kernel, ns per call: 1: 527-549, 2: 480-522, 3: 433, **4: 421-425**, 6: 450,
  8: 413 (4 kept: it is the smallest that reaches the plateau, and 16 is a multiple of it; `VGPU_TC86_ROWS`).
* **An f16 accumulator** (mma/wmma `.f16` C and D) had no vector form and ran the scalar one, 8,776 ns per m16n8k16; the sum rounded to nearest even to
  f16 is now vector too (`sum_to_f16_avx2`), 552 ns.
* **The PTX engine's `mma.sync`** for the dense f16, bf16 and tf32 forms (f32 or f16 accumulator) goes through the SIMD kernel the SASS HMMA already
  uses (`Interpreter::exec_mma_simd`): the fragments are read register by register as before, gathered, and D is written the same way, instead of
  building `std::vector<double>` tiles with a switch per element.

### Results

`bench.sh --ab`, 11 to 15 alternating runs, counters identical in every row (the harness compares them), B/A of the median of warp-instructions per
CPU second, p from the harness's Mann-Whitney test. Loads are server2's, shared with other jobs (a quiet pair of CPUs was used where there was one):

| workload | engine | A (base) | B (this branch) | B/A | p | load |
|---|---|---|---|---|---|---|
| wmma | sass | 2.012 | 2.852 | 1.42x | < 0.001 | 7.8-10 |
| wmma | ptx | 1.179 | 1.672 | 1.42x | < 0.001 | 7.8-10 |
| mma_ptx | sass | 3.833 | 5.348 | 1.40x | 0.004 | 7.8-10 |
| mma_ptx | ptx | 0.847 | 0.945 | 1.12x | < 0.001 | 7.8-10 |
| (the same, with the PTX `mma.sync` route) mma_ptx | ptx | 0.885 | 2.165 | 2.45x | < 0.001 | 9 |

The final tree (all four commits), SASS engine, 21 alternating runs on CPUs 12-15, load 26-28: wmma 1.932 -> 2.802 (1.45x, best 1.50x, p < 0.001), mma_ptx 3.788 -> 5.651
(1.49x, best 1.48x, p < 0.001), counters identical.

A second set at load 25-28 with every CPU busy (so every number is lower and the ratios are less sharp): wmma sass 1.15x (best of runs 1.50x,
p 0.008), wmma ptx 1.46x, mma_ptx sass 1.48x, mma_ptx ptx 2.35x, all p < 0.001, counters identical.

An f16-accumulator program (`wmma` 16x16x16 and `mma.sync.m16n8k16` with half C and D, 6,000 iterations, 4 warps; CPU time of the process, 11
alternating runs, load 25-28, output hash identical): SASS 0.344 s -> 0.051 s (6.8x, p < 0.001), PTX 0.462 s -> 0.117 s (3.9x, p < 0.001).

### Same-bits tests (wired into CMake)

* `tests/unit/test_tc_ampere_fast.cpp`: the current `tc86::mma`, `mma_avx2` and the f16-accumulator form against `tests/unit/tc86_v0.hpp`, a
  frozen copy of the earlier header, over 26 million elements (13 shapes including M not a multiple of the row group, K not a multiple of the chunk, the
  largest 32 x 32 x 16; f16, bf16 and tf32; f32 and f16 accumulators; eight operand modes: plain, zeros, subnormals, infinities and NaNs of both kinds,
  largest finite values and powers of two, raw 32-bit patterns, cancelling pairs across chunk borders, one huge term among tiny ones; accumulators at,
  above and below the products' size). Also `sum_to_f32_avx2` and `sum_to_f16_avx2` lane by lane against the scalar conversions on every
  power-of-two neighbourhood, f16 grid ties and random sums for every frame.
* `tests/unit/test_mma_fast.cpp`: PTX `mma.sync` of every dense form with `VGPU_FASTPATH=1` and `=0` on sm_86, sm_80 and sm_90 profiles, half the
  lanes of a warp predicated off, 2.9 million words compared.
* Mutation checks (the test must fail), run by editing a copy of the header or `interpreter.cpp`: 33 mutants of the kernel (sentinel too small, the
  accumulator's frame position off by one, the `>> 4` bias, a dropped sign, `g` off by one, the bit-length mask, the remainder rows, the NaN/infinity masks
  of rows, columns and accumulator, the out-of-range redo, truncation replaced by rounding, the tie rules and overflow bound of the f16 rounding, the
  sign of a zero) and 7 of the PTX route (no sm_86 sum, accumulator type, C and D register order, f16/bf16 kind, a wrong K, a bf16/tf32 kind swap). 36 of 40
  are killed. The four survivors are all equivalent mutants: a low-bit mask of 63 instead of 127 in the sum's bit length (|S| < 2^30.2, so the
  top 24 bits start at or above bit 6), a wider "in range" test in the f16 sum (the shift it guards is never past -10 for a finite sum), and the
  PTX route with K cut to 8 or the bf16 and tf32 kinds swapped, which only make it decline (the fragment arity no longer matches) so the
  general path answers.

### Tests run

`ctest -j4` on the whole tree (629 tests): 97% passed. The 17 failures are not this change: 11 `amd_*` (no ROCm on server2), two lints that call
`git ls-files` in the rsync'd copy, `e2e_lockstep_blocks` (timed-out waits under the box's load; the base build fails the same way, run side by
side), and three nvcc CUTLASS compiles (`e2e_cutlass_hopper/gemm/sm100`) that were `Killed` mid-compile while Postgres held 21 GB (re-run alone, hopper was killed again and the other two were not completed). They need a
quiet machine or CI. Passed, among others: `test_tc_ampere`, `test_tc_ampere_fast`, `test_mma_fast`, `test_simd`, `test_fastpath`, `test_exec3`,
`e2e_mma_forms`, `e2e_mma_layout`, `e2e_wmma_types`, `e2e_wmma_gemm`, `e2e_float_half_forms`, `e2e_simd_lane_forms`, `e2e_sass_tier`.

### Not done

* **IMMA (int8/int4) and FP8 forms** still take the general element-by-element path (one `std::vector<double>` tile pair per instruction, a switch per
  element): about 2.3 us per m16n8k32 s8 `mma.sync`, 4 to 5 times the f16 route. An integer SIMD kernel is straightforward (exact arithmetic) but the
  `.satfinite` clamp after every 128 bits of K and the s4/u4/b1 layouts make it a piece of its own; none of the benchmark rows uses it.
* **Fusing the f16->f32 conversion and layout transposes of `hmma_k` into the decode of the exact kernel** (the wrapper is about a fifth of an
  HMMA now): not tried.
* **Sparse `mma.sp`, m8n8k4, block-scaled and narrow-float forms** are unchanged; they do not use the chunked sum (sm_86 only) except through
  the general path.
## The AMD engine: atomics across threads, contiguous lanes, and the matrix multiply (branch `speedup2-amdmem`)

Four of the ranked wins of "The AMD engine" (3, 5, 6 and 7). Three kept, one dropped. Measured on server2 (Ryzen 9 9950X, 32 logical
CPUs shared with other people's builds: 1-minute load 11 to 18 during the final A/B, pinned to the idle CPUs 8-15), Release, GCC 13.3,
ROCm 7.2 `hipcc` programs, simulated MI300X and RX 6800 only (no AMD card touched). `tools/bench-amd` A/B, base `speedup` a12f8d84
against this branch, 11 runs, alternating; ratios are B/A of the median (the best-of ratio agrees within 3% in every row but the 8-thread atomics, whose base runs 11 to 13 s and is the noisiest), all
workloads self-check bit for bit against their host reference.

| workload | target | threads | B/A median | B/A best | p | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| atomics | mi300x | 8 | **8.66x** | 8.00x | < 0.001 | faster (wall 12.7 s to 1.5 s) |
| atomics | rx6800 | 8 | **4.27x** | 4.13x | < 0.001 | faster |
| atomics | mi300x | 1 | 1.008 | 1.015 | 0.33 | same |
| atomics | rx6800 | 1 | 1.032 | 1.041 | 0.10 | same |
| mfma | mi300x | 1 | **2.31x** | 2.30x | < 0.001 | faster |
| mfma | mi300x | 8 | **2.31x** | 2.33x | < 0.001 | faster |
| mem | mi300x | 1 | **1.18x** | 1.16x | < 0.001 | faster |
| mem | mi300x | 8 | 1.17x | 1.14x | < 0.001 | faster |
| mem | rx6800 | 1 | 1.12x | 1.11x | < 0.001 | faster |
| mem | rx6800 | 8 | 1.11x | 1.10x | < 0.001 | faster |
| valu_int (control) | mi300x / rx6800 | 1, 8 | 1.005 to 1.017 | | 0.06 to 0.56 | same |
| launch | mi300x / rx6800 | 1, 8 | 0.985 to 1.012 | | 0.24 to 0.75 | same |

The `valu_int` and `launch` rows are controls, code this branch does not change; they bound the noise of the run (about 2%).
`rocprofv3` wrote no counter CSV on this host during the session (the `--check-counters` path), so the harness was run with the
documented full-size instruction counts and its "counters identical" column is only the same number twice; the evidence that counts
and bits do not move is the unit tests below (they compare `DispatchStats` and memory images between the new and the old path) and
`ctest` over the AMD and memory tests (119 of 119, including the four RDNA/CDNA 5 targets of the tests that have them).

### What changed

* **Atomics across threads** (`MemoryManager::atomic_rmw`, `src/core/memory.cpp`). The striped `memory_atomic_lock` was taken for every
  lane of every atomic; at 8 threads `atomics` was slower than at one. With work-groups on several threads an atomic on an ordinary word
  of device memory is now a compare-and-swap on the word itself (4 and 8 bytes; the operation is applied to the value the swap
  replaces, and the result is stored only where the locked version stored it: when it differs, or, for buffer atomics, always, since
  that made a never-written chunk real). A chunk that is not real yet is made so first and the swap runs on it. Everything else (a
  fault armed, a host mapping, a word on the end of an allocation, an address that is no allocation's, a misaligned or 1/2/16-byte
  access) returns false having done nothing, and the caller takes the lock as before; one address always takes one road, so the two
  never meet on a word. One thread keeps its plain load and store with no lock. The scalar `s_atomic_*`, the vector and flat atomics,
  the buffer atomics and the host side of the hostcall ready stack (`Hostcall::service`) all use it, so a host thread and a kernel
  still exclude one another.
* **Matrix multiply** (`matrix_multiply`, `amd/src/mfma_math.hpp`). The binary16 widening was a call into libgcc (`__extendhfdf2`: 35%
  of the `mfma` samples; the baseline x86-64 has no F16C); it is now integer work, exact for all 65536 values including NaN payloads.
  The per-instruction heap vectors are per-thread buffers; B is kept by column so each dot product reads two contiguous runs; a shape's
  lane and output offsets (a handful of divisions by the shape's sizes each) are worked out once per thread and shape; binary16
  sources and a float accumulator in registers are read without the generic operand lambda; and float results are summed four at a time
  (four independent chains, each in the order it always was, so the bits do not change). A thread-local looked up per instruction
  cost 6% (`__tls_get_addr`); the machine holds the pointer.
* **Contiguous lanes** (`Machine::memory_run`, `MemoryManager::load_quad_run / store_quad_run`). A global load or store whose active
  lanes take consecutive addresses, one, two or four words a lane, goes to the memory manager as one `load_run` / `store_run` call (or
  the quad calls where each lane moves sixteen aligned bytes, which stay one 16-byte access each). Refused whenever lane by lane could
  differ (a fault armed, a host mapping, a run not inside one allocation, misaligned, peers attached), and then the lanes go one at a
  time as before and meet any error where they would. A load whose first address the lane before it just read does not start a run
  (the one-read-per-address rule of `load_once`), and the last lane's words become that rule's memory.
* **Dropped: cheaper one-wave launch set-up (7).** A launch's set-up is a work-group's register file, `Group`/`Wave` vectors and the
  hand-off (`launch` is 350 to 450 klaunch/s). The one removable piece, the wave's 64 KB zero-initialised register file (256 rows), costs
  0.7 us a launch measured in isolation on this host (`std::vector<std::array<uint32_t, 64>>(256)` built and freed, under load), against
  2.2 to 2.9 us a launch. Sizing it by the kernel's declared VGPR count is not bit-safe (any instruction may index a register past
  the declared count, as `s_set_gpr_idx` and a kernel's undefined reads do, and the file reads zero today), tracking what a wave wrote
  would add a store to every register access, and clearing a recycled file costs as much as making a new one. Nothing was shipped and
  `launch` is unchanged in the A/B above.

### Lessons

* The first version of the run path sat inside `global_access`'s lane loop, and `atomics` at **one** thread (which runs none of the new
  code) got 2.2 to 2.4% slower (31 runs, p < 0.003, twice); removing the block gave 1.000. The cause was the size of that loop, not the
  work: the same logic as a separate `[[gnu::noinline]]` member brought `atomics` back to 1.008 / 1.032 (same) and made `mem` faster than
  inline (1.18x, from 1.13x). Keep a new fast path out of a hot loop's body.
* The first matrix-multiply change (widening and the buffers only) measured 0.97x, slower: the generic operand lambda, not the
  conversion, was the cost. The profile (`tools/bench-amd/profile`) is what pointed at the per-instruction thread-local and the
  divisions.
* A mutation that is an equivalent program proves nothing: transposing the matrix plan's C offsets changes no result (it is only a
  storage map, read back through the same table), and the first version of the order test could not see a reversed dot product
  because a float result rounded from a double hides the order unless products cancel exactly (the test now uses powers of two whose
  products of 2^-28, 2^1 and 2^30 cancel or are swallowed according to the order).

### Tests

All wired into CMake. Each was shown to fail when the change is wrong (mutation checks, run on server2 against scratch copies):

| test | what it holds | mutations that fail it |
| --- | --- | --- |
| `test_memrmw_lockfree` (`tests/unit`) | `atomic_rmw` against load_scalar / operation / store_scalar over 120 000 random accesses (aligned or not, over the edges, freed, host-mapped, faults armed or cells stuck, uniform, real and never-written chunks, `always_store` both ways): values, memory left, chunks made real, error text; declines without calling the operation; six threads, no lock, adding, xor-ing and taking maxima on words in an unwritten, a uniform and a real chunk, final values equal the serial result; `load_quad_run / store_quad_run` against the quads one at a time | CAS replaced by a plain store; `always_store` ignored; a quad run reading a uniform chunk as zero |
| `test_amd_gcn_atomics` (extended) | 20 000 work-groups of `returning` (add, sub, xor, exchange) and 512 of `wide` (float and 64-bit add) on 8, 3 and 1 threads, lock-free and locked (`amd::set_fast_paths`): the words add up | CAS replaced by a plain store |
| `test_amd_gcn_memruns` (new, kernels `amd/tests/data/memruns.c`, built for gfx942, 1030, 1100, 1201, 1250; run on all five) | seven copy kernels (1, 2 and 4 words, 32- and 64-bit offsets) over 18 address patterns (consecutive, reversed, strided, repeated, overlapping stores, misaligned, over chunk and allocation edges, off the end, wild, random) and 7 lane masks, 3 draws each, new path on and off against twin memories: memory images, resident bytes, error text, wave / instruction / VMEM / flat counts | stores in reverse lane order; the exec mask ignored when extending a run; a load run landing in the wrong lane |
| `test_amd_mfma_math` (new) | all 65 536 binary16 values widened to the compiler's double, bit for bit (NaN payload, zero sign); the contiguous dot products against the strided ones they replaced; four at once against one at a time; order of summation | subnormal bias off by one; four-wide sums out of order |
| `test_amd_gcn_asm` (extended, kernels `asm_mfma_order.s`) | `v_mfma_f32_4x4x4_16b_f16`, `16x16x16_f16` and `32x32x8_f16` with a non-zero accumulator on random halves and on powers of two whose sums depend on the order, against the arithmetic written out from the layouts | B halves swapped; four-wide sums out of order |

## Tier 2 together (branch `speedup`, all pieces merged)

Merged tree against the Tier 1 tree (both Release, server2, CPUs 24-27 for NVIDIA and 24-31 for AMD, one host thread unless the AMD row
says `all`, 9 alternating runs, load 5-12). Counters are identical in every NVIDIA row; the AMD rows rest on the same-bits digest test
because rocprofv3 counting was unavailable on that host.

NVIDIA, B/A of the median: `wmma` SASS 1.48x, PTX 1.51x; `mma_ptx` SASS 1.47x, PTX 2.45x; PTX `local` 1.16x; `mem` SASS 1.03x;
`shared` SASS and PTX 1.02x. Everything else is within 2% (the Tier 1 work already took the rest). The tier's new MATCH, IMMA and LDSM
handlers show up only on programs that use them: atomicAggregate 1.4x, immaTensorCoreGemm 2.8x (IMMA) and 1.17x (LDSM).

AMD, B/A of the median against Tier 1's tree, p < 0.001 (simulated MI300X / RX 6800): valu_int 2.1 / 2.5x, valu_f32 2.1 / 2.7x,
valu_f64 1.9 / 2.5x, mfma 2.4x, mem 1.8 / 2.2x, lds 2.9 / 2.8x, atomics 1.8 / 2.3x at one thread and 12x / 8x with all threads, divergent
2.4 / 3.0x, wave 1.7 / 2.0x, launch 1.2x. api and startup are unchanged.

`test_amd_gcn_same_bits` (15,566 instruction shapes against digests recorded from the unmodified base) is the guard on all of the AMD
changes: the first merge failed it on 65 MFMA shapes (a different NaN payload when both operands are NaNs) and the fix keeps the fast loops
and recomputes only NaN results in the base's operand order.

## The AMD matrix instructions, every form (branch `speedup4-amdmatrix`)

The first rewrite of `matrix_multiply` (`speedup2-amdmem`) covered binary16 inputs into float accumulators on CDNA. This one measures what
each other matrix form costs per instruction, and speeds the forms that cost most: integer and 8-bit-float MFMA, bfloat16, float, xf32,
double and sparse (`v_smfmac_*`) MFMA, the gfx950 f8f6f4 and block-scaled forms, RDNA3/4 `v_wmma_*`, and gfx1250's `v_wmma_*` /
`v_swmmac_*`. Results are bit for bit what they were: the guard is `test_amd_gcn_same_bits` (digests recorded from f81cc0c1: 16,304 shapes,
the base's 15,566 and 738 new ones that cover every matrix form), plus the new old-vs-new test below.

Measured on server2 (Ryzen 9 9950X, shared; load average 2 to 14 during the runs), Release, GCC 13.3, simulated MI300X, RX 7900 XTX and
RX 9070 XT, one host thread, CPUs 24-31 for the harness (under `flock /tmp/speedup-bench.lock`), 20-21 for the per-instruction probe.

### What each form cost (before)

`amd_matrix_bench` (new, `amd/tools/matrix-bench.cpp`) runs one instruction of the ISA corpora over and over on a wave whose registers hold
small finite numbers (`time_instruction`, flavour 4 of the same-bits probe), and prints nanoseconds. Median over the operand variants of
a mnemonic (`amd/tests/data/isa_corpus_matrix_*.txt` holds several per mnemonic: registers and accumulation registers, constants for C,
`cbsz`/`abid`, `neg_lo`, `clamp`), 3 alternating runs of base and new, 150 iterations each:

| instruction | before ns | after ns | |
| --- | ---: | ---: | ---: |
| `v_mfma_f32_16x16x16_f16` (gfx942) | 1776 | 582 | 3.0x |
| `v_mfma_f32_32x32x8_f16` | 4209 | 1178 | 3.6x |
| `v_mfma_f32_16x16x16_bf16` | 2148 | 489 | 4.4x |
| `v_mfma_f32_32x32x8_bf16` | 4747 | 1092 | 4.3x |
| `v_mfma_f32_16x16x4_f32` | 1136 | 348 | 3.3x |
| `v_mfma_f32_32x32x2_f32` | 3346 | 875 | 3.8x |
| `v_mfma_f32_16x16x8_xf32` | 1547 | 401 | 3.9x |
| `v_mfma_f64_16x16x4_f64` | 2490 | 654 | 3.8x |
| `v_mfma_i32_16x16x32_i8` | 5223 | 893 | 5.8x |
| `v_mfma_i32_32x32x16_i8` | 10660 | 1343 | 7.9x |
| `v_mfma_f32_16x16x32_fp8_fp8` | 7609 | 1133 | 6.7x |
| `v_mfma_f32_32x32x16_fp8_fp8` | 10530 | 1855 | 5.7x |
| `v_smfmac_f32_16x16x32_f16` | 3223 | 1057 | 3.0x |
| `v_smfmac_i32_16x16x64_i8` | 8454 | 1524 | 5.5x |
| `v_smfmac_f32_16x16x64_fp8_fp8` | 11218 | 1696 | 6.6x |
| `v_mfma_f32_16x16x128_f8f6f4` (gfx950) | 25121 | 3387 | 7.4x |
| `v_mfma_f32_32x32x64_f8f6f4` | 29822 | 4785 | 6.2x |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | 31101 | 6116 | 5.1x |
| `v_wmma_f32_16x16x16_f16` (gfx1100) | 33172 | 1242 | 26.7x |
| `v_wmma_f16_16x16x16_f16` | 34596 | 2988 | 11.6x |
| `v_wmma_f32_16x16x16_bf16` | 10496 | 1434 | 7.3x |
| `v_wmma_i32_16x16x16_iu8` | 15105 | 1385 | 10.9x |
| `v_wmma_i32_16x16x16_iu4` | 16440 | 1556 | 10.6x |
| `v_wmma_f32_16x16x16_f16` (gfx1201) | 34113 | 1262 | 27.0x |
| `v_wmma_f32_16x16x16_fp8_fp8` | 38463 | 2192 | 17.6x |
| `v_wmma_i32_16x16x16_iu8` | 16198 | 2185 | 7.4x |
| `v_wmma_i32_16x16x32_iu4` | 31470 | 3422 | 9.2x |
| `v_wmma_f32_32x16x128_f4` (gfx1250, 20 iterations) | 783952 | 56200 | 13.9x |
| `v_wmma_f32_16x16x128_f8f6f4` (gfx1250) | 277056 | 25570 | 10.8x |
| `v_wmma_f32_16x16x32_f16` (gfx1250) | 81358 | 7956 | 10.2x |
| `v_wmma_i32_16x16x64_iu8` (gfx1250) | 63663 | 7272 | 8.8x |
| `v_swmmac_f32_16x16x64_f16` (gfx1250) | 89051 | 11309 | 7.9x |

The ranking before: RDNA `v_wmma_*` at 10 to 40 us an instruction was ten times anything else (it compared the type's name, as a
`std::string`, for each of the 2 x 16 x 16 x 16 element reads); the gfx1250 forms at 60 us to 0.8 ms were beyond that; then the 8-bit
float and integer MFMA and the sparse ones at 5 to 30 us (an element decoded by a chain of `type ==` tests, an `ldexp` for each 8-bit
float, a double for each integer); then the binary16, bfloat16 and float forms at 1 to 5 us. (The probe's numbers wobble by a few
percent with the load and the operand variants: variants whose destination overlaps a source turn the data into NaNs and infinities
after a few hundred repetitions, and the NaN redo costs the same before and after.)

### What changed

* **A tile at a time** (`matrix_multiply_fast`, `wmma_fast`; `matrix_multiply` and `wmma` keep the old element-at-a-time code as
  `matrix_multiply_reference` / `wmma_reference`, used for anything the fast path does not take: operands that are not registers,
  registers beyond the file, a sparse form whose A is not in registers). Each operand is decoded once, with its type fixed at compile time
  (`matrix_elem<T>`, `matrix_decode_a/b`): A into rows of K, B into rows of K by column, C into the tile; the 8-, 6- and 4-bit floats by
  a 256-entry table built from `f8_to_float` itself; then the whole tile is summed (`mfma_tile_f`, `mfma_tile_fma`, `mfma_tile_int` in
  `amd/src/mfma_math.hpp`) and written back through the shape's place table.
* **The sums**: every output is still a double sum of its products taken in order of k from the accumulator, rounded once to float, so the
  bits are those of `mfma_dot`; what changed is that 4 rows x 4 columns (SSE2, 2-wide) or 4 x 8 (AVX2, 4-wide, chosen at run time by
  `__builtin_cpu_supports`; no fused multiply-add, so each product and each sum is rounded as before) are worked out together,
  and A's scalars are broadcast against a row of B. A sum that ends as a NaN is redone in the fixed operand order of last round
  (`mfma_tile_ordered`), so NaN payloads are what they were. The double-precision forms use one hardware `fma` a term (`target("fma")`,
  chosen at run time) with the same redo. Integer forms are exact: int16 tiles and `pmaddwd` (SSE2) or `vpmaddwd` (AVX2), four rows by
  4 or 8 columns; the accumulator is added afterwards (wrap for MFMA, saturate for WMMA with CLAMP).
* **Widening**: binary16 operands go to doubles four lanes at a time (`half4_to_double`, in float: a rebias and a shift, the subnormals as
  mantissa x 2^-24, the NaN payload kept; exhaustively equal to the compiler's conversion for all 65,536 values), bfloat16, float and xf32
  by a shift or mask and `cvtps2pd`; integer tiles are filled from the register bytes without going through doubles. The scalar
  `half_bits_to_double` cost 1.8 ns a value, 512 of them 0.9 us of a 16x16x16 MFMA's 1.8 us.
* **RDNA3/4 `v_wmma_*`** (`wmma_fast`): the old code looked up the element type by comparing `std::string`s for every one of the
  2 x 16 x 16 x K element reads; now A (16 x K) and B (K x 16) are decoded once and summed as a 16 x 16 x K tile.
* **gfx1250 `v_wmma_*` / `v_swmmac_*`** (`wmma1250`): the matrices are decoded once (`adec`, `bdec`), each block of K (the whole of it, or
  one scale's 16 or 32) is summed as a tile, and the scale and the accumulator are applied to the block sums exactly as before.
* **Operand order of the NaN rule, made explicit.** Where two operands of an x86 add or multiply are NaNs the first one's payload wins and the
  compiler chooses which is first; three places depended on that choice and now spell it out, each matched to the recorded digests:
  the MFMA dot (B first of the multiply, the sum first of the add, last round); the WMMA dot (`WMMA_ORDER` 2: B first of the multiply,
  the product first of the add: found by trying all four against the digests: no digest wrong for 2, 19 to 79 wrong for each of the others); and
  the double-precision fma (`mfma_fma`: the first NaN of B, A, the sum, quieted). The last was found by a unit test: two copies of the
  same loop in one translation unit (`mfma_dot_fma` and the new tile code) gave different payloads for `fma(NaN, NaN, NaN)`, because
  GCC swaps the first two operands of `fma` freely. The reference code uses the same functions, so it cannot drift either.

### Results

`tools/bench-amd/bench.sh --build <base> --build <new> --threads 1 --size full --runs 9 --cpus 24-31 --no-counts`, base = f81cc0c1 (plus the
probe), new = this branch; ratio = B/A of the median (the best-of ratio agrees within 2% in every row); without the counting runs (rocprofv3
wrote no counter CSV here) the harness takes the count as 1 and the ratios are the ratios of the speeds; the work is identical (the
tests below are the evidence that bits and counts do not move). `mfma` is the existing workload; the other seven are new (`tools/bench-amd/kernels/bench_amd.hip`, `run_mfma_i8` ... `run_wmma_iu8`: each
self-checks `D = n x (A x B)` for small integers against the host, bit for bit, and fails the run otherwise); the RDNA targets are the new
`rx7900xtx` and `rx9070xt`, which the default `--target` (mi300x and rx6800) does not include.

| workload | target | what it runs | A cpu s | B cpu s | B/A median | B/A best | p | spread A / B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mfma | mi300x | `v_mfma_f32_16x16x16_f16` (existing) | 0.910 | 0.358 | **2.54x** | 2.55x | < 0.001 | 1 / 3% |
| mfma_i8 | mi300x | `v_mfma_i32_16x16x32_i8` | 1.213 | 0.246 | **4.93x** | 4.99x | < 0.001 | 2 / 4% |
| mfma_fp8 | mi300x | `v_mfma_f32_16x16x32_fp8_fp8` | 2.050 | 0.314 | **6.53x** | 6.59x | < 0.001 | 46 / 66% |
| mfma_bf16 | mi300x | `v_mfma_f32_16x16x16_bf16` | 0.978 | 0.316 | **3.09x** | 3.09x | < 0.001 | 1 / 4% |
| mfma_f64 | mi300x | `v_mfma_f64_16x16x4_f64` | 1.312 | 0.395 | **3.32x** | 2.91x | < 0.001 | 43 / 115% |
| smfmac | mi300x | `v_smfmac_f32_16x16x32_f16` | 0.864 | 0.298 | **2.90x** | 2.93x | < 0.001 | 19 / 79% |
| wmma | rx7900xtx | `v_wmma_f32_16x16x16_f16` | 8.689 | 0.360 | **24.2x** | 24.3x | < 0.001 | 7 / 89% |
| wmma | rx9070xt | `v_wmma_f32_16x16x16_f16` | 9.581 | 0.345 | **27.7x** | 26.2x | < 0.001 | 10 / 80% |
| wmma_iu8 | rx7900xtx | `v_wmma_i32_16x16x16_iu8` | 6.124 | 0.711 | **8.61x** | 9.42x | < 0.001 | 32 / 36% |
| wmma_iu8 | rx9070xt | `v_wmma_i32_16x16x16_iu8` | 4.482 | 0.571 | **7.84x** | 6.92x | < 0.001 | 51 / 54% |

Final tree, 9 alternating runs, size full (64 waves, 4,000 to 8,000 instructions each), 1-minute load average before each workload 1.6 to
31.9 (median 4.0: other people's builds arrived during the f64 to iu8 rows, which is where the spreads are large; the ratios are
across alternating runs, so they carry the load of both builds alike). An earlier run of the same A/B on the tree before the
vector widening (lower load, 1.9 to 6.0, spreads 1 to 3% except bf16 and the RX 9070 XT's `wmma`) gave `mfma` 1.50x, `mfma_i8` 4.83x, `mfma_fp8` 6.56x,
`mfma_bf16` 2.22x, `mfma_f64` 3.17x, `smfmac` 2.03x, `wmma` 19.5x / 14.4x, `wmma_iu8` 8.94x / 6.78x, all p < 0.001, which shows what the widening
(`half4_to_double`, `cvtps2pd`) added: 1.5x to 2.5x on `mfma`, 2.2x to 3.1x on `mfma_bf16`, 2.0x to 2.9x on `smfmac`. The counters are identical in every row
(the harness compares the two builds' counts only with the counting runs, which are unavailable here; `test_amd_matrix_fast` compares
`DispatchStats` and every counter through the digest instead).

### Tests

All wired into CMake; each was shown to fail on a scratch copy when the code is wrong (mutation checks, server2):

| test | what it holds | mutations that fail it |
| --- | --- | --- |
| `test_amd_gcn_same_bits` (extended: four new MFMA/WMMA corpora and gfx1250's, `VGPU_SAME_BITS_REFERENCE=1` runs the reference path) | 738 more shapes (`amd/tests/data/isa_corpus_matrix_*.txt`, written by `amd/tools/matrix-asm.py`: every MFMA, SMFMAC and WMMA form of gfx942, gfx950, gfx1100, gfx1201 and gfx1250 in several operand variants), digests recorded from f81cc0c1 with the probe only; the 15,574 earlier digests are unchanged (the file is the base's plus the appended sections); passes on the tile path and on the reference path | k terms in reverse order (AVX2 kernel); NaN redo removed; wrong NaN operand order; fma NaN priority A first; integer pairs swapped; AVX2 kernel drops a row; fp8 table of the wrong format; sparse index set ignored; gfx12 WMMA half of K swapped; constant C ignored; A and B scales swapped; gfx1250 block offset ignored; gfx1250 sparse A not expanded; binary16 infinity exponent; subnormal scale; B step; xf32 mask; bfloat16 half; int8 sign; WMMA binary16 tile transposed |
| `test_amd_matrix_fast` (new) | every matrix instruction of eleven corpora (all the shapes above and the CDNA3, CDNA4, RDNA3 and RDNA4 ones from ROCm's libraries) on 8 seeds x 5 register flavours (any bits, small integers, floats with specials, half the lanes off, small finite numbers), tile path against the reference path through `set_matrix_fast_path`: registers, LDS, memory, counters, errors; and again with the registers of each instruction moved to random places that overlap one another (the destination over a source, the accumulator half over the destination, either bank) | the same list except the fma priority (both paths share `mfma_fma`; the digests and `test_amd_mfma_math` catch that one); and, in the second test only, the accumulator of an integer form read after the destination is written |
| `test_amd_mfma_math` (extended) | a tile against `mfma_dot` per output (random doubles of every magnitude, zeros, infinities, NaNs with random payloads) for 13 shapes; terms summed in order (1, 2^-60, -1); NaN payload for each of the four operand orders; fused tile against `mfma_dot_fma`; the priority of `mfma_fma`; integer tile against int64; the 4-wide kernels against the 2-wide ones (`mfma_avx2_switch`); all 65,536 binary16 values four at a time against `half_bits_to_double` | order, NaN redo, wrong order, integer pairs, AVX2 row, infinity exponent, subnormal scale |
| `test_amd_gcn_asm`, `test_amd_hipcc_mfma`, `amd_rocblas` (existing) | the layouts, order and rocBLAS GEMMs of the first rewrite | (unchanged) |

Run on server2 against this branch: `ctest -R ^test_amd` (143 of 143), `amd_rocblas`, `test_amd_hipcc_mfma_decode` (pass; `amd_hipsparselt` and
`amd_pytorch_xf32` skip there, for lack of the libraries).

### Found on the way

* **WMMA with CLAMP and a signed integer input is wrong in the reference code, and so on both paths** (left as it is, as the digests
  record it): `return is_signed ? static_cast<int8_t>(byte) : byte;` has the type `unsigned`, so a negative 8- or 4-bit element
  becomes 4294967295 - |v| + 1 as a double. Without CLAMP only the low 32 bits are kept and those are right; with CLAMP the int64 sum
  saturates the wrong way. `wmma_fast` hands exactly that case (CLAMP with a signed input) to the reference so the two keep agreeing.
  The fix is one cast; it changes results, so it is not in a speed-up branch.
* `mfma_dot_fma` gave a different NaN than an identical loop elsewhere (above); `part += a * bv` in `wmma_reference` and `wmma1250` had no
  fixed order; both are explicit now.

* Under `-fsanitize=address,undefined` (-O1, server2): `test_amd_mfma_math`, `test_amd_matrix_fast` and `test_amd_gcn_asm` have no
  AddressSanitizer report; UBSan reports one thing in the matrix code, the signed overflow of `isum += a * b` in `wmma_reference` (the
  signed-input bug above: 4294967234 * 4294967287); `test_amd_gcn_same_bits` at -O1 differs from the digests on 85 shapes, none of them
  matrix instructions (`v_pk_add_f32`, `v_fma_mixlo_f16`, `v_div_fixup_f16`, ...: float codegen that depends on the optimisation level, as the
  test's header says).

### Not done

* `v_wmma_f16_16x16x16_f16` / `v_wmma_bf16_*` (16-bit results) stay at 3 us: 256 `static_cast<_Float16>(double)` calls into libgcc,
  about 10 ns each, are most of it. A direct double to binary16 rounding would have to equal libgcc's on every double; not attempted.
* gfx12 `v_swmmac_*` (RDNA4's sparse WMMA) is not implemented by the interpreter at all (an error), so there is nothing to speed up.
* The per-instruction shape lookup (`matrix_shape(OpName(...))`, a hash and a string compare, about 50 ns) and the C gather / D scatter
  loops (about 2 ns an output) are now the larger part of the small forms (4x4x4, 16x16x4): not worked on.
* The fast path is not taken for a source that is not a register, for registers past the end of the file, or (WMMA) with CLAMP on a
  signed integer input; those run the reference code, unchanged.
* The harness has no gfx1250 target (hipcc has no gfx1250 programs here); those rows come from `amd_matrix_bench` only.
## AMD libraries and real workloads: packed halves, dot products, half conversions, and the code object reader (branch `speedup4-amdlibs`)

The AMD benchmarks above are synthetic. This piece profiled the interpreter on what actually runs on it -- rocBLAS (ROCm 7.2's, unmodified),
hipBLASLt, MIOpen, the pantheon GPU workloads built for gfx942, the HIP and library samples of `amd/tests` -- all on the simulated MI300X, and
kept the changes the profile and an A/B supported.

### What was measured, and how

* **Cost per instruction.** A scratch patch (not committed) wraps `Machine::step` in `rdtsc` and keeps, per mnemonic, the count, the CPU time
  inside `step` and the mean active lanes, written at exit (`VGPU_THREADS=1`). It adds about 15 to 20 ns per instruction (`s_waitcnt` reads
  19 to 22 ns), so the cheap forms are overstated and the expensive ones are not. Pinned to CPUs 24-31 (0-15 for the "after" columns; see the
  caveat in the last paragraph of this section).
* **Host functions.** `tools/bench-amd/profile` (SIGPROF), on a `-g` Release build.
* **Programs** (1.70 billion wave-instructions in all): `amd/tests/rocblas/blas` (saxpy, sdot, sgemm, dgemm, `gemm_ex` on halves, bfloat16s and
  bytes, complex); a scratch driver for larger rocBLAS calls (sgemm, dgemm, hgemm, `gemm_ex` f16, bf16 and i8, gemv, level 1, strided batched,
  trsm, syrk, cgemm, 512 on a side); a hipBLASLt matmul (f16, bf16, f32 at 1024); `MIOpenDriver` (forward, backward-data and backward-weights
  convolutions in fp32, fp16 and bf16, with the library's own search over its solvers; batch norm, pooling, softmax, activation, GEMM); the 46
  pantheon workloads (built with the flags of `amd/tools/run-pantheon-workloads.sh`, 5 s each); `amd/tests/hipcc` and `amd/tests/libraries`
  samples; hipSPARSELt's `spmm`. No PyTorch here, and no rocprofv3 counting on this host.

### Ranked: which instruction forms cost the time

Share of the wave-instructions ("count") and of the time inside `step` ("time"), and the cost of one execution (64 lanes). Forms that
another piece of this round owns are marked: (tier) the vector ALU and compare forms of `amdtier`, (mem) memory and LDS of `amdmemtext`,
(mfma) `amdmatrix`.

| suite | form | count | time | ns per execution |
| --- | --- | ---: | ---: | ---: |
| MIOpen fp16 convolution (its search tries the slow solvers) | `v_dot2_f32_f16` | 24% | 42% | 1390 |
| | `v_pk_fma_f16` | 3.0% | 13% | 3440 |
| | `buffer_load_short_d16` (mem) | 3.3% | 5.6% | 1346 |
| | `v_pk_mul_f16` / `v_pk_add_f16` | 1.1% / 1.0% | 4.2% / 3.3% | 2911 / 2518 |
| | `ds_read_b128` (mem) | 2.3% | 4.0% | 1388 |
| MIOpen fp32 convolution | `v_fmac_f32` (tier) | 40% | 42% | 299 |
| | `ds_read_b128` (mem) | 3.8% | 16% | 1227 |
| | the DPP forms (`v_fmac/add/mov_b32_dpp`) | 8.4% | 11% | 252 to 364 |
| | `ds_write_b32`, `buffer_load_dword` (mem) | 2.9%, 2.0% | 6.1%, 5.4% | 606, 763 |
| rocBLAS (blas + 14 larger calls) | `v_pk_fma_f16` (hgemm) | 3.9% | 28% | 3142 |
| | `v_mfma_f32_16x16x4_f32`, `v_mfma_f64_16x16x4_f64` (mfma) | 4.1%, 0.9% | 9.9%, 7.0% | 1070, 3625 |
| | `ds_read_b32/b64`, `buffer_load_*` (mem) | 7.2% | 10% | 552 to 1306 |
| hipBLASLt (f16, bf16, f32) | `v_mfma_*` (mfma) | 27% | 61% | 999 to 3536 |
| | `ds_read_*`, `ds_write_*`, `buffer_load_*` (mem) | 24% | 33% | 929 to 1230 |
| pantheon, 46 workloads | `v_lshl_add_u64` (tier) | 5.3% | 11.6% | 497 |
| | `v_pk_fma_f16` | 0.3% | 5.5% | 3824 |
| | `v_xor_b32`, `v_add_u32`, `v_mov_b32`, `v_fma_f32` (tier) | 6.4%, 2.6%, 3.9%, 2.3% | 5.2%, 4.3%, 4.0%, 4.6% | 186, 385, 239, 452 |
| | `global_load_dwordx4` / `dword` (mem) | 1.1% / 0.8% | 4.3% / 3.1% | 883 / 872 |
| | `v_mfma_f32_16x16x16_f16` (mfma) | 0.5% | 4.2% | 2114 |

By workload, the instruction that is most of the time: `compute_virus_agg` `v_pk_fma_f16` 69% (8144 ns) and `v_cmp_ngt_f16` 15% (7099 ns, tier's
compare path); `omni_virus` `v_pk_fma_f16` 77%; `tensor_virus` `v_pk_fma_f16` 70%; `transformer_train_step` `v_pk_fma_f32` 52% (tier);
`rope_stress` `v_sin_f32` 35% (1504 ns); `memory_retention_bake`, `pulse_virus`, `memory_thermal_asym` `v_fma_f32` 72 to 95% (tier); `fp64_virus`
`v_fma_f64` 88% (tier); `mma_virus`, `transformer_virus` `v_mfma_f32_16x16x16_f16` 93%, 84% (mfma); `atomic_virus` `global_atomic_add` 54% (mem).

The expensive non-memory forms, every one with at least 0.05% of all the time (cost of one execution, 64 lanes): `v_dot2_f32_f16` 1390 ns,
`v_pk_fma_f16` 3514, `v_pk_mul_f16` 2911, `v_pk_add_f16` 2518, `v_cvt_flr_i32_f32_e64` 2207, `v_fmaak_f32` 929 (tier), `v_add_f16_e64` 1182,
`v_subb_co_u32` 1419, `v_pk_fma_f32` 840 (tier), `v_cvt_f16_f32_sdwa` 1600, `v_sin_f32` 1504, `v_pk_mad_u16` 5110, `v_pk_max_f16` 2501,
`v_cmp_ngt_f16_e64` 7099 (tier), `v_fma_mix_f32` 797; against 190 to 300 for an ordinary ALU form. The packed-half and dot-product forms are
a few times the cost of anything else because every lane crosses libgcc's soft `_Float16` conversion (`__extendhfsf2`, `__truncsfhf2`: each a
call, three or four per lane per half) and the operation is chosen by string comparison in the lane loop.

**Host functions** (`blas`, 5 s, one thread, 2454 samples at 1 kHz): `Machine::step` 52% (`matrix_multiply` 12%, `vector_alu` 14%,
`lds_access` 7.5%, `buffer_access` 7%); **`hipModuleLoadData` 31%** -- 25 Tensile code objects of 0.5 to 20 MB inflated, of which
`load_code_object` 19.4% and `read_bundle` 8.9% (zstd 5%, the rest copies). A single 19.8 MB Tensile library
(`TensileLibrary_Type_DD_*_gfx942.co`, 689 kernels) took 5.5 ms to inflate and 25.5 ms to parse.

### What changed

All of it bit for bit; the new files are `amd/src/packed_math.hpp` and the tests below.

* **Packed 16-bit float arithmetic, a wave at a time** (`v_pk_fma_f16`, `v_pk_add_f16`, `v_pk_mul_f16`; `Machine::packed_f16_wave`).
  Halves are widened and narrowed with integer code that equals the compiler's conversion for all 65536 halves and all 2^32 floats (the
  libgcc call is gone); the multiply-add is the host's FMA unit's, and on a host with AVX2, F16C and FMA (checked at run time, and with
  MXCSR in its default state) eight lanes at a time. A lane with a NaN in it follows the old code's rule exactly: the first operand's payload
  for an add or a multiply (the order the recorded results show, held by inline assembly), libm's `fmaf` with the multiplicands in the
  recorded order for a fused multiply-add. Not taken for a MODE other than the default, gfx1250, a source that is not a vector register, an SGPR or a
  constant, a part-register destination: the old per-lane code runs those.
* **`v_dot2_f32_f16` and `v_dot2c_f32_f16`** the same way (products exact in double, summed in the order the old code summed them; AVX2 four
  doubles wide). A lane whose sum is a NaN is handed back to the old code.
* **`v_cvt_f16_f32`, `v_cvt_f32_f16`, and half comparisons** use the same integer conversions (`lane_half_f`); `write_half` tests the cheap
  `fp16_ovfl` before widening its result; the packed 16-bit integer instructions (`v_pk_mad_u16` and the other fourteen) resolve which
  arithmetic they are once per instruction instead of comparing the name in each half of each lane.
* **The code object reader** (`codeobject.cpp`): fields read with one bounds check, names as views of the file (no `std::string` per symbol),
  hash maps for the lookups (names are only looked up, never iterated), `std::vector` room for the kernels reserved up front (a `Kernel` is
  copied, not moved, when the vector grows), the MessagePack tree's arrays and maps reserved, `.text` copied with one `memcpy`.

### Results

Alternating A B, 9 runs after a warm-up, CPU seconds of the process (user + system), Mann-Whitney p, one host thread, CPUs 24-31 under
`flock /tmp/speedup-bench.lock`. A is `speedup` f81cc0c1 as built (Release, `-g`), B this branch. Outputs were compared (a hash of the
results printed by each program) and are equal in every row. The load average is that of the machine during the runs (others' builds).

| program | A median [best] | B median [best] | A/B median | best | p | load |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `v_pk_fma_f16`, finite halves (microbenchmark) | 1.133 s [1.077] | 0.070 s [0.060] | 16.2x | 18.0x | 0.0000 | 15 to 16 |
| `v_pk_fma_f16`, random bits (NaNs and infinities of every payload) | 1.250 s [1.171] | 0.094 s [0.077] | 13.3x | 15.3x | 0.0000 | 16 to 17 |
| `v_pk_mul_f16` (the loop is a third other instructions) | 1.363 s [1.331] | 0.967 s [0.881] | 1.41x | 1.51x | 0.0000 | 17 to 18 |
| `v_dot2c_f32_f16` | 1.287 s [1.244] | 0.300 s [0.249] | 4.29x | 5.00x | 0.0000 | 18 to 19 |
| rocBLAS `hgemm` 512 (whole process) | 4.433 s [3.894] | 0.848 s [0.809] | 5.22x | 4.82x | 0.0000 | 19 to 26 |
| `v_cvt_f16_f32` + `v_cvt_f32_f16` loop | 0.472 s [0.456] | 0.297 s [0.295] | 1.59x | 1.55x | 0.0000 | 9.5 to 10 |
| half comparisons (`v_cmp_{lt,ge,neq}_f16`) | 1.154 s [1.143] | 0.464 s [0.455] | 2.49x | 2.51x | 0.0000 | 7.6 to 8.9 |
| rocBLAS `blas` program (whole process, 25 Tensile objects) | 2.422 s [2.409] | 2.101 s [2.073] | 1.15x | 1.16x | 0.0000 | 12 to 19 |
| rocBLAS sgemm 256 (one library load) | 0.082 s [0.080] | 0.073 s [0.071] | 1.12x | 1.13x | 0.0000 | 11 |
| `load_code_object`, a 19.8 MB Tensile object (inflate + parse, 20 times) | 0.632 s | 0.310 s | 2.04x | 2.04x | 0.0000 | 7 |
| `load_code_object`, rocBLAS `Kernels.so` 0.83 MB (40 times) | 0.090 s | 0.053 s | 1.71x | 1.73x | 0.0000 | 7 |
| `v_sin_f32` + `v_cos_f32` (no change made: control) | 0.453 s | 0.450 s | 1.01x | 1.02x | 0.0040 | 10 |

Cost of one execution (64 lanes, the step-timing patch, the same program before and after): `v_pk_fma_f16` 3103 to 261 ns (rocBLAS hgemm:
3106 to 142), `v_pk_add_f16` 2464 to 248, `v_pk_mul_f16` 2304 to 148, `v_dot2c_f32_f16` 1130 to 149, `v_cvt_f16_f32` 789 to 293,
`v_cvt_f32_f16` 375 to 243, `v_cmp_*_f16_sdwa` 905, 894, 1277 to 512, 518, 428. The pantheon workloads run for a fixed 5 s, so what changes is how much
they do in it: wave-instructions executed in the 5 s, three alternating runs each, old / new: `omni_virus` 7.4 to 7.9 M / 24 to 34 M,
`tensor_virus` 11.1 to 12.0 M / 29 to 33 M, `compute_virus_agg` 5.0 to 5.3 M / 24 to 28 M; `llm_prefill` and `rope_stress` (no
change made to their forms) are the same within the run-to-run spread (15 to 22 M and 14 to 18 M both ways).

### Tests

* `test_amd_gcn_same_bits` (the 15,566 shapes of the ISA corpora against digests recorded from the unmodified base): passes unchanged.
* `test_amd_gcn_packed_same_bits` (new): 2,143 instructions of the families this changed -- `v_pk_{fma,add,mul}_f16`, `v_dot2*_f32_f16`,
  `v_cvt_f16_f32`/`v_cvt_f32_f16` with every output modifier, `v_cmp*_f16` with every condition and operand modifier, `v_sin/cos_f32`, the
  fourteen packed 16-bit integer instructions -- in every modifier combination (`amd/tools/packed-corpus.py`, assembled by llvm-mc, with raw
  VOP3P encodings for the op_sel/neg combinations the assembler refuses), run in the four flavours of the first test (random bits among them,
  so NaNs of every payload), against `amd/tests/data/packed_same_bits_digests.txt`, recorded from the unmodified base.
* `test_amd_packed_math` (new, 7 tests): the integer half conversions against the compiler's (all 65536 halves; all NaN floats; the
  rounding boundaries of every exponent; millions at random; `VGPU_PACKED_EXHAUSTIVE=1` runs all 2^32 floats -- it passed); the packed
  arithmetic against a copy of the old per-lane code on random waves with masks, constants, scalar sources, negation and clamp, with the host's
  FMA and libm's, and the AVX2 code against the scalar code lane for lane; the F16C conversion against the integer one for every NaN; the dot
  products (scalar and AVX2) against the old code.
* `test_amd_codeobject_same_bits` (new): the 241 code objects in the repository (the objects, hsaco files, and the bundles inside the hipcc
  executables) against digests of everything the reader returns, or of its error text, recorded from the unmodified base. 801 more -- every
  gfx942 object of rocBLAS, hipBLASLt and hipSPARSELt -- were compared old against new once: identical digests (`VGPU_CODEOBJECT_EXTRA=<file>`).
* Mutation checks, each of which makes the named test fail: the fused multiply-add's multiplicands in source order (both digest tests); the add's or the multiply's NaN order flipped (10 and 5 shapes of the first test); the AVX2 conversion rounding toward zero (both
  digest tests, two unit tests); the dot2c sum order (AVX2) or the dot2 sum order (scalar), neither of which the digests see, both the unit test;
  a rounding-boundary term dropped from the half narrowing (36 shapes and the unit test); global offsets 4 off in the reader (the code object
  test).
* ctest `amd|gcn|rocprof|hipcc|hip_` on server2 (a `-DVGPU_WERROR=ON` build of everything but a NVIDIA test that does not compile under
  -Werror on this base, `test_mma_fast`): all pass or skip but `amd_rocprofv3`, `amd_rocprofiler_abi` and `amd_tools` (host: no counter CSV,
  header version, user namespaces), as on the base; `amd_hipcc` and `amd_rocprof_first_calls_rx6700xt` failed once under `-j4` and passed alone.

### Not done, and negative results

* **Sine and cosine** (`v_sin_f32`, `v_cos_f32`, 35% of `rope_stress`): a double-precision sine with a three-part pi/2 reduction, a 25th-order
  Taylor series and a rounding-boundary check, which gave libm's rounded-to-float answer on every one of the 4.7 billion arguments
  checked (all floats under 2^14 turns, 91% answered; the rest fall back) -- and was **slower than glibc's `sin`** (18.7 ns against 11.9 per
  lane). Dropped, together with its header and test. glibc's table-driven sine cannot be beaten at libm's own answer.
* **A sweep of `static_cast<float>(lane_half(...))` over every site** was tried and reverted: it changed which NaN payload `v_fma_f16` and
  `v_div_fixup_f16` produce (the operands of the libm call reached it in another order), which the first digest test caught on 20 shapes. The
  conversion is used only where there is no arithmetic on the result (`v_cvt_f32_f16`, the comparisons, `write_half`'s test).
* **`v_pk_min_f16`, `v_pk_max_f16`** (2501 ns): their NaN and signed-zero results are `std::fmin`/`fmax`, whose operand order is the compiler's
  choice; left to the old code. **Dot product lanes with a NaN sum** go back to the old code too (the recorded digests hold only four dot shapes).
* **Integer dot products** (`v_dot4_i32_i8`, `v_dot4c`, `v_dot2_i32_i16`): measured at 195 to 310 ns, already an ALU form's cost.
* **DPP** (`*_dpp`, 252 to 364 ns, 4.5% of all the time): the shuffle adds 50 to 100 ns to an ordinary form; not worth a rewrite alone.
* **`v_div_scale/fmas/fixup_f32`** (800 to 1200 ns): 0.24% of all the time.
* **`carry_alu`** runs `op.find("_co_u32")` on every vector ALU form that reaches the generic path before its own ladder; a decode-time flag
  would remove it (touches `amd_gcn.hpp`/`gcn_decode.cpp`, which `amdtier` edits). Not measured.
* **`read_bundle`** still copies a bundle's code twice (the 4 MB window into the string, then `load_code_object` into the vector); 9% of the
  `blas` run. A zstd window written straight into place, or a `CodeObject` that keeps the bundle's bytes, would take most of it.
* Unexplained: `v_cvt_flr_i32_f32_e64` (2207 ns), `v_subb_co_u32` (1419) and `v_mul_f32` (3947 ns in the fp16 convolution) cost 5 to 15 times
  their siblings in MIOpen's OpenCL-compiled direct convolutions; the three have the same execution count, as though in one loop. Denormal
  operands would explain a float multiply but not an integer subtract.

Caveat on the "after" columns: they were measured on CPUs 0-15 (the benchmark cores were taken by other pieces' runs), so compare them with
the same-session A/B table above, not with the "before" columns' absolute values.

### Shared lines this touches (for the merge)

`amd/src/gcn_exec.cpp`: one `#include` after `mfma_math.hpp`; `lane_half_f` (new) after `lane_half`; one line of `write_half`; a block of new members
(`packed_halves`, `packed_plain`, `packed_f16_wave`, `dot2_f16_wave`) before `lane_float`; the packed integer block of `vector_alu`'s default
branch (rewritten in place); a prefix in each of `v_dot2_f32_f16`, `v_dot2c_f32_f16` and `v_pk_fma_f16`'s case; three lines each of
`v_cvt_f16_f32`, `v_cvt_f32_f16`; **one line of `Machine::compare`** (the half comparison, `amdtier`'s area). `CMakeLists.txt`: three test registrations after
`test_amd_mfma_math`.

## The AMD engine: memory instruction names read once, and the LDS accesses (branch `speedup4-amdmemtext`)

What was left of the text work on the memory instructions, and what the profile of the memory workloads showed after it.
Measured on server2 (Ryzen 9 9950X, 32 logical CPUs shared with other people's builds and benchmarks), Release, GCC 13.3, ROCm 7.2
`hipcc` programs, simulated MI300X and RX 6800 only (no AMD card touched), `tools/bench-amd` with a cache of its own, one host
thread unless the row says 8, pinned to CPUs 24-31, A and B alternating, 11 runs (9 for 8 threads), the benchmark lock held for each
run. Base: `speedup` f81cc0c1 built in its own directory. Every workload checks its result bit for bit against a host reference, and
the harness's "counters identical" column is `yes` in every row (the instruction counts of the two builds agree, quick size, through
rocprofv3 inside `vgpu shell`).

### What changed

* **Names parsed at decode, not at every execution** (`gcn::MemFacts`, `Inst::mem`, filled in by `gcn::finalize`). `narrow`,
  `half_access` and `parse_atomic` ran on the instruction's name text for every global, flat, scratch, buffer and LDS instruction
  executed, and `buffer_access` and the LDS ladder's default case ran them again for **every lane** (a `find("_d16")` per lane for
  every `ds_read_b64`, `ds_read2_b32`, `buffer_load_dword`...; a compare chain of up to sixteen strings per lane for a buffer atomic).
  `scalar_load` did a `find("prefetch")`, three prefix tests and, for `s_atomic_*`, a `std::string` and a chain of string compares.
  `mem_facts_of` works out, once, what the name says: the width and sign of a narrow access, the `_d16` forms, the atomic operation
  (global/flat, buffer and scalar atomics each have their own list of names), the prefix tests, the transposing loads, how
  `global_access` takes the instruction, how scalar memory does, and which LDS access it is. The parsers moved from `gcn_exec.cpp`
  to `gcn_decode.cpp` (one copy, `gcn::parse_narrow`, `parse_half`, `parse_atomic_body`, `transpose_bits_of`; the executor's
  `narrow`, `half_access` and `parse_atomic` call them) and an instruction nobody finalized (a test's hand-built one) gets the same
  facts from `gcn::mem_facts(in, scratch)`. The struct is 14 bytes (an `Inst` is 352): it is copied for VGPR indexing and gfx1250's VGPR MSBs.
* **The LDS accesses of real kernels without the ladder** (`Machine::lds_fast`): `ds_read_b32`, `ds_write_b32`, `ds_read_b64` /
  `_b96` / `_b128`, `ds_write_b64` / `_b128`, `ds_add_u32`, `ds_read2_b32`, `ds_write2_b32`, `ds_read_u8` / `i8` / `u16`,
  `ds_write_b8` / `b16`: one lane loop over the active lanes, the vector registers read and written directly, the same 32-bit
  address arithmetic and the same rules for an address past the work-group's LDS (a read gives zero, a write goes nowhere, a
  work-group given no LDS is caught, same text). Before it each lane went through a `switch` on the name's hash and then a text
  compare of the whole name (`OpName::operator==` was 8% of `ldsx`, a function call per lane), three more name tests, `lane_src`,
  `set_word` and `write_lane` calls, and, for the wide forms, the `half_access` scan. A fast path declines (the ladder runs) when
  an operand is not a plain vector register or a cross-lane instruction is running.
* **Global accesses read the address register directly** where it is a plain vector register (`lane_src` and `lane_src64` were
  separate calls per lane: 8-11% of the memory workloads), and so do the byte and short loads and stores; `memory_run`'s run
  detection reads the scalar base once instead of once per lane; a flat access finds its aperture (`flat_access`) with the scalar base,
  the immediate offset and the target worked out once. Same lanes, same order, same errors.
* **`Machine::step` no longer builds and tears down an `Inst` for every instruction.** `Inst remapped;` (gfx1250's VGPR MSBs) and
  `Inst indexed;` (the vector case, for VGPR indexing) were default-constructed on every step and every vector instruction: 2.3% +
  2.6% of the `smem` profile (`gcn::Inst::Inst`, `~Inst`), 4.8% of `divergent`. They are now a pointer made by a `noinline` copy where
  one is needed (`InstCopy`). The first version used `std::optional<Inst>`: with it `step` went from 6% to 21% of the `divergent` samples and `scalar_alu` and
  `more_alu` turned up as functions of their own (the inlined copy constructor and destructor made `step` larger, which is the likely
  reason the compiler stopped inlining them), and the whole engine measured 4 to 12% **slower** than the base (`valu_int`,
  `divergent`, `wave`, even `launch`; the controls caught it); out of line `step` is back to 6% and it measures faster. This part is `step`'s and the vector case's, not the memory instructions': the `amdtier` piece edits the same lines.

### Results (B/A of the median, base f81cc0c1 against this branch)

One thread, 11 runs, p < 0.001 unless noted. The rows from `ldsx` to `mem` and the `valu_int` control are one A/B (load average 3.5 to 6.6, median 5.2);
`atomics`, `valu_f32`, `divergent`, `wave` and `launch` are from another, base against the tree before the `flat_access` edit (load 1.9 to
16.6, median 4.3):

| workload | what it runs | mi300x | rx6800 |
| --- | --- | ---: | ---: |
| `ldsx` (new) | `ds_write_b128`, `ds_read_b96`, `ds_read2_b32`, `ds_add_u32` with barriers | **2.28x** | **1.99x** |
| `lds` | tree reduction, tile transpose | **2.14x** | **1.90x** |
| `buf` (new) | buffer loads and stores with offsets, some past the end | **1.33x** | **1.32x** |
| `flat` (new) | generic-pointer loads and stores that reach global memory | 1.15x | 1.14x |
| `smem` (new) | `s_load_dword`, `s_load_dwordx4` of uniform table entries | 1.11x | 1.19x |
| `narrow` (new) | byte and short global loads and stores | 1.10x | 1.09x |
| `mem` | 16-byte copy, scalar scale, gather | 1.10x | 1.12x |
| `atomics` | global atomic add, max, CAS loop | 1.13x | 1.18x |
| `valu_int`, `valu_f32` (controls: no memory instructions) | | 1.04x, 1.04x | 1.05x, 1.09x |
| `divergent` | | 1.09x (p 0.04) | 1.15x (p 0.02) |
| `wave` | | 1.01x (p 0.15, same) | 1.05x |
| `launch` | | 1.00x (same) | same (spread 60%) |

Eight threads (9 runs, load 2.1 to 6.0): `ldsx` 2.30x / 2.01x, `lds` 2.13x / 1.88x, `flat` 1.13x / 1.13x, `mem` 1.10x / 1.12x,
`atomics` 1.14x / 1.13x, `valu_int` 1.02x. The rows without a name for a change above are the `Inst` construction (all workloads) and
whatever the compiler made of the new layout; this table does not split them. What was split: the `scalar_load` facts alone, against
the same tree with the base's `scalar_load` text tests put back (15 runs, load 2 to 4): `smem` 1.034x on the MI300X and 1.050x on the
RX 6800 (p < 0.001; the `valu_int` control 0.99x and 1.00x, p 0.2 to 0.4).

### What the profile says is left

`MemoryManager::scalar_location` (15 to 19% of `buf`, `narrow`, `mem`: the chunk lookup of every lane's access), `load_scalar` and
`store_scalar`, then `lane_src` and `write_lane` called from `vector_alu`. The buffer path has no run detection (the global one does):
a lane-by-lane `load_words` for every lane of every `buffer_load_dword`. None of this is name text.

### Tests

Wired into CMake. Each was shown to fail when the change is wrong (mutation checks on scratch copies of the source, run on server2,
the build rebuilt each time; "caught" is a failing, crashing or silent test run).

* `test_amd_gcn_same_bits` (the existing guard, 15,566 shapes, digests from the unmodified base): passes unchanged. It runs memory
  instructions on random registers, so most of its addresses fault; it could not see most of the changes above.
* `test_amd_gcn_memory_same_bits` (new; `amd/tests/data/mem_same_bits_digests.txt`): every DS, flat, global, scratch, MUBUF, MTBUF and
  scalar memory shape of the ISA corpora (2,927 of them) run in twelve ways on a wave whose address registers are **aimed** at three
  allocations (`amd::digest_memory_instruction`, in `gcn_exec.cpp` next to `digest_instruction`): aligned, misaligned, on the end of an
  allocation, past it, wild, in the shared and the private aperture (gfx1250's too), runs of consecutive lanes, the same address in every
  lane, mixed per lane, half the lanes off; buffer resources valid, invalid, swizzled, scratch-resource, too small; LDS accesses past the
  work-group's LDS. Each of the first few shapes of a family (the atomic operations, the byte and short loads and stores, the LDS
  widths) is also run **under every name of its family**, so every atomic operation and width is reached. The digest holds registers,
  LDS, scratch, the three allocations, the instruction counts and the error text. The golden was recorded from a build of the **base**
  commit plus the probe function and nothing else (a scratch tree of f81cc0c1 with the function spliced in, run twice for
  determinism), never from the tree under test. About half of the (instruction, mode) pairs complete; the rest end in the error the
  base gives (invalid pointer, out of bounds, misaligned, past the LDS...) and so compare error texts. `s_memtime` and
  `s_memrealtime` (a clock) are left out.
* `test_amd_gcn_mem_facts` (new): `mem_facts_of` against the old per-execution tests, copied into the test as they were, for every
  instruction name of the corpora and for about 4,000 made-up names (every segment, form and suffix in combination); and the
  executor's fallback for an unfinalized instruction against the finalized one.
* `bench_amd_counters`: the new workloads have entries in `tools/bench-amd/expected/amd-hip-7.2.26015.json` (counts recorded with the
  base build; the existing entries are unchanged).

| mutation | caught by |
| --- | --- |
| `sbyte` loses its sign in `parse_narrow` | `test_amd_gcn_memory_same_bits`, `test_amd_gcn_mem_facts` |
| a `_x2` atomic's size not set to 8 in `parse_atomic_body` | both |
| narrow stores not told apart from loads in `global_access` | `memory_same_bits` |
| buffer `atomic_sub` adds | `memory_same_bits` |
| buffer `atomic_add_f32` with two NaNs takes the other operand's payload (see below) | `memory_same_bits` |
| scalar `s_atomic_umin` takes the maximum | `memory_same_bits` |
| scalar `s_load_i8` / `i16` with the sign inverted | `memory_same_bits` |
| `ds_read_i8` without its sign extension | `memory_same_bits` |
| `ds_write_b16` writes one byte | `memory_same_bits` |
| `ds_read2_b32` reads both words from the second offset | `memory_same_bits` |
| the `_d16` test of the LDS default case inverted | `memory_same_bits` (a crash) |
| atomics always four bytes | `memory_same_bits` |
| `scratch_store` sub-dword stores told apart from loads wrongly | `memory_same_bits` |
| narrow load written to the next register | `memory_same_bits` |
| run detection that over-reaches (`>=` for `==`) | `memory_same_bits` |
| flat access ignoring gfx1250's private aperture / the shared aperture | `memory_same_bits` |
| run detection and flat aperture detection that drop SCALE_OFFSET | **not caught**: the first only makes runs not be found (which is invisible, runs being exact), the second changes which aperture a lane is in only for addresses within a factor of 16 of an aperture's edge, which the probe's 32-bit offsets do not reach |

### Lessons

* **The two-NaN trap again.** The first run of the memory digests failed on one instruction, `buffer_atomic_add_f32` (gfx1250), where
  both the memory word and the register data are NaNs: x86 returns the first operand's payload, and which operand GCC makes the first
  is its choice. The base's build had the register's data first; the rewrite of the chain of `if`s into a `switch` put the other
  first. The rewrite now says it (`nan(b) && nan(v) ? v | quiet : b + v`), which is the base's bits.
* **A control that is not a control.** The first full A/B of the branch showed the memory workloads 10 to 120% faster and `valu_int`
  and `divergent` 6 to 12% slower; the controls were right. `std::optional<Inst>` in `step` was the cause (see above); out of line
  the same code is faster than the base for every workload. Always run the ALU workloads and `launch` next to the memory ones.
* **Tests that cannot reach the code.** The mutation of the `_d16` test in the LDS default case was first reported as not caught;
  the test had crashed, printing nothing, and the script looked for the word FAIL. And a mutation run left its object files newer
  than a restored source (rsync keeps the old modification time), so the next build used the mutant: touch after restoring.
* **A latent bug the probe found, left alone.** `scratch_at` checks `offset + bytes > scratch_per_lane`, which wraps for a negative
  `offset:` immediate with a small register offset (`scratch_store_b128 v9, ..., offset:-16` with v9 = 0): the check passes and
  the store lands before the lane's scratch, on the heap. The base does it, so does this branch (fixing it changes
  `test_amd_gcn_same_bits`'s recorded digests of those shapes, which may not be re-recorded); the probe skips scratch accesses with a
  negative offset. The fix is one line: `offset > per_lane || bytes > per_lane - offset`.

### Not done

* Scalar-load facts, `lds_fast`, the direct register reads and `memory_run`'s hoisted base were each kept only on the evidence of
  the table above; none was dropped. Not tried: a fast path for the rest of the ladder's LDS cases (`ds_read2st64`, the atomics with
  return, `ds_bpermute`), run detection for buffer loads, and moving `Inst plain;` out of `cross_lane_alu` (a default `Inst` per DPP
  instruction, the `wave` workload's remaining construction cost; it is the vector paths').


## The AMD pre-decoded tier (branch `speedup4-amdtier`, `VGPU_AMD_TIER`)

The AMD interpreter had been given opcode `switch`es and decode-time flags (the "AMD engine" sections above). What those left was the
frame every instruction still paid (`Inst remapped`, `FloatMode`, `debug::active()`, the mix counters, an `OpName` and a ladder per
helper) and, in every lane loop, the operand reads (`lane_src`, the `InvariantSources` table) and the EXEC test. This tier is the NVIDIA
tier's twin (`src/sass/tier*.cpp`): `gcn::finalize` calls `tier_classify` (`amd/src/gcn_tier.cpp`) when an instruction is decoded, which
resolves the hottest forms into a handler pointer and operand classes (`Inst::fast`, `amd/include/vgpu/amd_gcn_tier.hpp`): a VGPR row,
an SGPR, a constant, VCC, EXEC or the null register, each classified once; `Machine::step` calls the handler before it does anything else.
A handler works on whole register rows with kernels written once and compiled twice (`amd/src/gcn_tier_kernels.inc`: the baseline, and
under `#pragma GCC target("avx2,fma,popcnt")`, picked from `simd::level()` when the instruction is decoded). A scalar source becomes a
row filled once, EXEC is applied by one merge (a full EXEC writes straight into the destination), and a 64-bit operand is gathered from
its register pair. The part that needs the interpreter's `Machine` (global loads and stores) is `amd/src/gcn_tier_machine.inc`, included
into it: `global_access` with the work its name asked for done at decode, and the lane loop, `memory_run`, the load memo and every memory
call exactly as they were, in the same order.

**The contract is the tier's usual one**: a handler does the whole instruction, or nothing and says so, and the interpreter then runs it
as before. What it does not model it never claims (`tier_classify` leaves `fast.run` null): SDWA, DPP, VOPD, `op_sel` on a VOP3, 16-bit
halves, gfx1250, VGPR indexing and the MSBs (the wave's state is tested at run time), a `MODE` other than round-to-nearest with denormals
kept, tracing and the debugger. **NaNs.** Which of two NaN payloads wins depends on the order the compiler gave the operands, the trap the
MFMA work met, so a float handler that finds a NaN in an active lane of its result gives the instruction back before it has written
anything and the interpreter decides the bits; `v_min_f32` and `v_max_f32` also give back a pair of opposite zeros (which one wins is
libm's). A NaN *input* to a comparison, a convert or a min/max needs no such care, as none can pass a payload on. `neg`, `abs`, `omod` and
`clamp` are handled. Counters are those of the interpreter: `DispatchStats` is updated by the same instruction, before it can fail.

What has a handler (about 5,100 of the 15,500 decodable shapes of the ISA corpora):

* 32-bit vector integer and bit operations, VOP1/2/3 and their promoted forms with any scalar, constant, VCC or EXEC source: `v_mov_b32`,
  `v_not_b32`, `v_add_u32`, `v_sub_u32`, `v_subrev_u32` (not with `clamp`), `v_mul_lo_u32`, `v_mul_hi_u32/i32`, `v_mul_u32_u24`,
  `v_mul_i32_i24`, `v_mad_u32_u24`, `v_mad_i32_i24`, `v_and/or/xor_b32`, the three shifts, `v_min/max_u32/i32`, `v_bfi_b32`, `v_bfe_u32`,
  `v_alignbit/alignbyte_b32`, `v_add3_u32`, `v_or3_b32`, `v_xor3_b32`, `v_and_or_b32`, `v_xad_u32`, `v_lshl_or_b32`, `v_lshl_add_u32`,
  `v_add_lshl_u32`, `v_bcnt_u32_b32`, `v_mbcnt_lo/hi_u32_b32`, `v_med3_u32/i32`, `v_min3_u32`, `v_max3_u32`, `v_cvt_i32_f32`,
  `v_cvt_u32_f32`, `v_cvt_f32_u32/i32/ubyte0`, `v_cndmask_b32`, `v_readfirstlane_b32`, `v_readlane_b32`, `v_writelane_b32`; the carry
  family (`v_add_co_u32`, `v_addc_co_u32`, `v_sub_co_u32`, `v_subb_co_u32`, `v_subrev_co_u32`, `v_subbrev_co_u32`) and `v_mad_u64_u32`.
* binary32: `v_add/sub/subrev/mul_f32`, `v_fma_f32`, `v_fmac_f32`, `v_fmaak_f32`, `v_fmamk_f32`, `v_min/max_f32`, `v_rcp_f32`, and the packed
  `v_pk_add/mul/fma_f32`. binary64 and 64-bit integers: `v_add/mul/fma/fmac_f64`, `v_mov_b64`, `v_lshlrev/lshrrev_b64`, `v_ashrrev_i64`,
  `v_lshl_add_u64`, `v_cvt_f64_u32/i32`.
* `v_cmp` and `v_cmpx` of i32, u32, f32, i64, u64 and f64 (all sixteen float tests), into a scalar pair, VCC, a single SGPR (wave32) or EXEC.
* scalar ALU (`s_mov`, add and subtract with their carries, `s_mul_i32`, the logic family at 32 and 64 bits, shifts, `s_cselect`,
  `s_min/max`, `s_bcnt1`, `s_ff1`, `s_abs`, `s_bfe_u32`, `s_lshl1..4_add_u32`, `s_movk_i32`), the scalar compares, the `saveexec` family
  at 64 and 32 bits, the branches (`s_branch`, `s_cbranch_scc0/1`, `execz/nz`, `vccz/nz`), `s_waitcnt` (and its yield), `s_barrier` and the
  hints that do nothing.
* LDS (`ds_read/write_b32/b64/b128`, `ds_read2/write2[st64]_b32`, `ds_bpermute_b32`: a wave whose accesses do not all land in the LDS hands
  the instruction back) and global memory (`global_load/store_dword[x2,x3,x4]` and the byte and short loads and stores).

`VGPU_AMD_TIER_STATS=1` prints, at exit, the instructions the interpreter ran anyway. Over the bench workloads that is the per-wave
`s_load_*` and `s_endpgm` (about 1,000 of 570,000 on `valu_f32`), `global_atomic_*` (the `atomics` workload, which this piece leaves alone: the
earlier atomics round took its share) and `v_lshlrev_b32_sdwa`.

### Tests (wired into CMake)

* `test_amd_gcn_tier_same_bits` (`amd/tests/unit/test_gcn_tier_same_bits.cpp`): every instruction shape of the eight ISA corpora that has a
  handler (5,129 of the 15,500 that decode) is run through `amd::digest_instruction` with the tier off and on, in eleven flavours of
  pseudo-random state (numbered 0 to 4 and 10 to 15 after the merge with amdmatrix) and three seeds each (125,628 runs each way, about a minute): any bits; small integers; floats with the special values
  and NaNs; half the lanes off; two flavours of floats without a NaN (`10`, `11`; the tier's flavours were numbered 4 to 9 on their branch and moved to 10 to 15 on merging, because amdmatrix's probe flavour 4 took that number), so that the float handlers run instead of handing back;
  two (`12`, `13`) that put a second wave in the work-group and loads in flight, for the store gates and `s_waitcnt`; and two (`14`, `15`, for
  flat and global accesses only) that point the address registers into memory that exists, in consecutive lanes and scattered. The digest
  is everything the instruction changed or threw: all registers, masks, SCC, the program counter, LDS, memory, the wave's yield state,
  the counters and the text of an error. `VGPU_TIER_ONLY=<text>` restricts it to some instructions, `VGPU_TIER_SEEDS=n` runs more seeds.
* `test_amd_gcn_same_bits` (the guard; unchanged, digests recorded from the unmodified base) passes with the tier on, which is now the
  default, and with `VGPU_AMD_TIER=0`.
* Every `test_amd_gcn_*` unit test (25 binaries: real kernels, LDS, atomics, counters, memory runs, cross-lane, packed, doubles, spills)
  passes with the tier on and off alike (`VGPU_AMD_TIER=1` and `0`), and so do `test_amd_mfma_math` and `test_amd_hipcc_ops` (kernels built by `hipcc`).
* `amd_exec_bench` with `VGPU_BENCH_DUMP=1` prints all of a dispatch's counters (every `DispatchStats` field, the instruction mix) and a hash
  of the buffers it was given, filled with pseudo-random bytes. The nine bench kernels on the simulated MI300X (gfx942) and RX 6800
  (gfx1030): the counters and the hashes are identical with the tier on and off, and the two targets' buffers agree with each other.


### Why not `include/vgpu/simd`

The shared kernels (`simd.hpp`) were written for the NVIDIA executors and carry their semantics: a shift of 32 or more shifts every bit out
(AMD takes the count modulo 32), a NaN result is the canonical `0x7fffffff`, and `ffma` is fused only where PTX says so. The AMD interpreter's
own expressions are the specification here, so the tier has its own kernels (`gcn_tier_kernels.inc`, the same two-level scheme and the same
`simd::level()` choice, `fp-contract=off` for the file) and the tier-vs-interpreter test is what holds them to it. The first thing that test caught is an instance: the sign of the zero in `fmaxf(-0, +0)` (and `fminf`) differed between the interpreter's
compiled `v_max_f32` and the tier's, so the handlers give any pair of opposite zeros back to the interpreter.

### Mutation checks

Each of these was applied to a copy of the tree, built, and run against the new test (it must fail); the tree was then restored. All 19
fail (`FAIL`, with `VGPU_TIER_ONLY` set to the instruction concerned to keep each run short):

| # | mutation |
| --- | --- |
| M1 | `v_subrev_u32` with its operands swapped |
| M2 | a float handler keeps a NaN result |
| M3 | the integer handler writes lanes EXEC switched off |
| M4 | `s_cbranch_scc0` inverted |
| M5 | a comparison keeps the bits of inactive lanes |
| M6 | the LDS store gate dropped |
| M7 | a branch counted as a scalar instruction |
| M8 | `s_add_u32` loses its carry |
| M9 | `ds_bpermute_b32` reads a lane that is off |
| M10 | `v_cmpx` leaves EXEC alone |
| M11 | the global store gate dropped |
| M12 | `s_waitcnt` never yields |
| M13 | the reversed subtract in `v_subrev_co_u32` not reversed |
| M14 | the LDS range check loosened |
| M15 | `global_load_sbyte` loses its sign |
| M16 | the load memo dropped after a run |
| M17 | the last word of `global_load_dwordx2` lost |
| M18 | `global_store_short` stores a byte |
| M19 | a byte access sized as a dword when the runs are found |

M15 (run first, before flavours 14 and 15 existed) passed the real-kernel unit tests (`test_amd_gcn_memory`, `memruns`, `ops`, `lds`) with the
tier on, and the digest saw no global access with a valid address either (random registers fault): that is why flavours 14 and 15 (8 and 9 on the branch) exist, and
M15 to M19 were run with them.

### Results

Measured on server2 (Ryzen 9 9950X, load average before each workload 2.0 to 24.7, median 5.8, other people's builds and benchmarks
around, serialised with `flock /tmp/speedup-bench.lock`), Release, GCC 13.3, `hipcc` 7.2 programs, simulated MI300X (gfx942) and RX 6800
(gfx1030) only. `tools/bench-amd/bench.py` A/B, 9 runs alternating on CPUs 24-31, one host thread; base = `speedup` f81cc0c1 built
Release, B = this branch, `VGPU_AMD_TIER` unset (on). The host has no usable `rocprofv3` counter output, so the harness ran with
`--no-counts` and its rate column is 1e6 over the device phase's CPU seconds (the same count on both sides; the ratio is the speed-up);
what stands in for the counters is the bit-for-bit self-check of every workload and the `VGPU_BENCH_DUMP` comparison above. Ratios
are B/A of the median (the best-of ratio agrees within 5%), p is the Mann-Whitney p of the harness:

| workload | mi300x | rx6800 | p |
| --- | ---: | ---: | ---: |
| valu_int | **8.70x** | **8.09x** | < 0.001 |
| valu_f32 | **10.0x** | **7.93x** | < 0.001 |
| valu_f64 | **4.56x** | **5.27x** | < 0.001 |
| mem | **2.95x** | **3.22x** | < 0.001 |
| lds | **3.65x** | **3.17x** | < 0.001 |
| divergent | **4.39x** | **4.35x** | < 0.001 |
| wave | **11.2x** | **9.12x** | < 0.001 |
| atomics | **1.81x** | **1.96x** | < 0.001 (the global atomics are still the interpreter's) |
| launch | 1.12x | 1.10x | < 0.001 (kernels of a few instructions) |
| api | 1.01 | 1.02 | 1.0 / 0.26 (same) |
| startup | 1.00 | 0.99 | 0.93 / 0.22 (same) |

`api` and `startup` (not interpreter-bound) are the controls that bound the noise. With the tier **off** (`VGPU_AMD_TIER=0`, the same
build against the base): valu_int 0.966 / 0.965, mem 0.998 / 0.978, divergent 0.967 / 0.980, launch 0.986 / 1.001, startup 0.980 /
0.992: the opt-out is not free, about 2 to 3.5% on the instruction-bound workloads (the extra test in `step` and the larger `Inst`;
significant for valu_int on the RX 6800 and divergent, p <= 0.001), which does not matter as it exists to rule the tier out.

With all the host threads of the pinned CPUs (`VGPU_THREADS` unset, 7 runs, CPU time as above, p at the harness's floor of 0.001): valu_int
8.61 / 7.01x, valu_f32 9.69 / 6.70x, mem 3.00 / 3.25x, lds 3.65 / 3.14x, divergent 4.42 / 4.24x (MI300X / RX 6800); the work-group
threads share nothing the tier writes (the table of kernels is read-only, a handler touches its own wave, the bail counter is off unless a
test turns it on).

A first run of the same A/B, before the tier's per-instruction checks were hoisted into the `Machine` (the tier flag, tracing and the debugger
are now looked at once per dispatch instead of once per instruction; the other end of the same session's load differed), gave valu_int
8.28 / 6.82x, valu_f32 9.25 / 6.69x, valu_f64 4.51 / 5.10x, mem 2.96 / 3.12x, lds 4.08 / 3.03x, divergent 4.25 / 4.03x, wave 10.1 / 8.65x,
atomics 1.78 / 1.89x: within 5% of the table above except the RX 6800's valu rows (6.8 and 6.7 against 8.1 and 7.9). The second run is the
one reported; no claim is made about what the hoisting alone is worth.

### Not done

* `s_load_*` and `s_endpgm` (a few per wave: about 1,000 of 570,000 instructions of `valu_f32`), `global_atomic_*` (the `atomics`
  workload is 1.8 / 2.0x faster, from the instructions around the atomics; the atomics themselves are still the interpreter's), flat, scratch and buffer accesses, the narrow and 16-bit forms (SDWA, DPP, `op_sel`, halves, `v_cvt_f16_*`, `v_pk_*_f16`),
  the transcendentals (`v_exp_f32`, `v_log_f32`, `v_sqrt_f32`, `v_floor_f32` and the like), the division
  helpers (`v_div_scale`, `v_div_fmas`), MFMA (its own piece), gfx1250 (its own interceptions in `cdna5_alu`).
* A NaN-aware float path: a NaN result hands the instruction back, which costs the interpreter's time on a program that computes NaNs
  all the time (`VGPU_AMD_TIER_STATS=1` shows the count as "handed back").
* Profiling the tier itself (`perf` is unusable here): what is left of the per-instruction cost is the fetch from the decode cache, the
  `Regs` frame and the indirect call.


## The AMD runtime: calls, queues and work-group threads (branch `speedup4-amdmt`)

What a HIP call, a stream's hand-off and a launch's threads cost, measured and cut where it could be cut without moving a result.
Base is `speedup` f81cc0c1 (Tier 1 and Tier 2). Measured on server2 (Ryzen 9 9950X, 32 logical CPUs shared with other people's builds
and benchmarks), Release, GCC 13.3, `tools/bench-amd` pinned to CPUs 24-31, simulated MI300X and RX 6800 only (no card touched), A and
B alternating, 7 to 21 runs. The final A/B below ran at a 1-minute load of 5 to 9 (median 6); the leave-one-out runs at 7 to 15. Every
ratio is the median wall time of A over B (above 1: B is faster), p the harness's Mann-Whitney p. Counts and bits cannot move in these
changes, and `test_amd_gcn_same_bits` (the guard, 15,566 shapes against digests recorded from the unmodified base) passes unchanged.

### New rows in the harness (`tools/bench-amd`)

All self-checking, none counted by the profiler (`--no-counts`; rocprofv3 writes no counter CSV on this host).

| workload | what it does | units |
| --- | --- | --- |
| `apicheap` | `hipGetDevice`, `hipGetLastError`, `hipPeekAtLastError`, `hipDeviceGetAttribute`: the fixed cost of a HIP call | calls |
| `launch_mid` / `launch_wide` | launches of 128 / 1024 one-wave work-groups of a tiny kernel on one stream: the hand-off to the work-group threads | launches |
| `launch_sync` | four chained one-wave launches, then `hipStreamSynchronize`: the hand-off between a stream's thread and its caller | calls |
| `memcpy` | 16 KiB and 64 KiB `hipMemcpy` (H2D, D2D, D2H), `hipMemcpyAsync`, `hipMemsetAsync`, `hipMemset`, each round ending in a stream synchronize | calls |
| `stream` | `hipStreamCreateWithFlags`, `hipEventRecord`, `hipStreamWaitEvent`, two launches, queries, synchronize, `hipStreamDestroy` | calls |
| `graph` | a graph of four chained kernel nodes captured from a stream, launched in a loop | launches |

### Where a launch's time goes

`launch` (100,000 one-wave launches on two streams) at one work-group thread, `tools/bench-amd/profile` on a `-g` build of this
branch (4000 Hz, call chains, so inclusive): the two stream threads are 94% of the samples and the calling thread 12%, so the rate is
the stream thread's per-launch work and not the hand-off. Inside it: `execute` 49% (the work-group's set-up, `set_up_group`, 14%, the
one wave's instructions 28%), `MemoryManager::free` 20% (the kernarg segment's, with its quarantine record), `find_ordinary_miss` 13%
(every allocation and free changes the table lock's epoch, so each kernel's first accesses look the table up again); half of the
samples have no source line, which is libc (malloc and free, the zeroing of the 64 KiB register file and of the kernarg chunk, futex). `strace -c` on 2,000 launches:
2.7 futex calls a launch on the base, and 1.0 `gettid` and `getpid` each, which are the main-thread check of `launch_in_order`. An
experiment that is not shippable (a scratch kernarg segment allocated once per thread and never freed, which moves every later device
address and the quarantine) makes `launch` 1.65x (MI300X) and 1.42x (RX 6800) faster, p < 0.001: the allocate-write-free of the
kernarg segment is about a third of a launch. Switching telemetry off (`VGPU_TELEMETRY=0`, 15 alternating pairs) is worth 2.8%.
`alloc`+`free` of 128 bytes alone is 147 ns on one thread; with the 192-byte write into the fresh allocation, which makes a 64 KiB
chunk real and frees it again, 455 ns.

### What changed

* **A HIP call with nothing watching is one load** (`ApiCall`, `hip_api.cpp`). The previous round's argument capture was already built
  only after the profiler test, but each call still read a function-static, a thread-local depth and the profiler pointer, and the
  profiler side was inline at ~560 call sites. One relaxed load of `g_api_watch` (0: nothing; 1: a profiler attached or `VGPU_TRACE_API`;
  2: not settled yet) now decides, and entering and leaving are out of line, cold. A profiler attaching sets 1, so calls made after that
  count their depth and report as before; one attaching while another thread is inside a call sees that call's nested calls reported as
  outermost, which is the one difference.
* **Hand-offs spin before they sleep** (`hip_queue.hpp`). A thread that waits for the other side of a stream's queue (the caller for a
  result, the stream's thread for work) polls for 400 pauses (a few microseconds) before it takes the futex, with `VGPU_QUEUE_SPIN=0` for none
  (it is also none when the process may use one CPU). The queue has a condition variable for each side, a notify that is a no-op with no
  sleeper, and a thread that is spinning stops at once when its queue is destroyed (without that, `hipStreamDestroy` waited out the spin,
  and `stream` was 14% slower).
* **A waited-for call runs on the calling thread when its stream is idle** (`WorkQueue::run_inline`, used by `in_order`). `hipMemcpy`
  and friends wake the stream's thread, sleep, and are woken in turn; with every item done, nothing running and every marker finished
  the work is numbered, finished and counted as a queued item is, on the caller. Anything queued meanwhile waits for it, a failure is
  the queue's as before, a profiler being attached (its copy records carry the thread) keeps the stream's thread. A stream used only
  this way never starts its thread. Not done for asynchronous items: they must return before the work does.
* **A launch's work-group threads are kept** (`gcn_pool.hpp`, used by the three places `execute` started and joined `std::thread`s).
  The same static ranges, a thread for each task and all of them alive at once (work-groups may wait on one another), the first task
  on the calling thread, the floating-point environment and signal mask of the launching thread, made afresh in a forked child. The
  CPUs a program may use stay the limit (`worker_count`).
* **Smaller things.** A wave's work-item ids are counted up, not divided out per lane (three 64-bit divisions a lane before); the
  main-thread check in `launch_in_order` is asked once per thread and fork, not with two system calls a launch; a device-to-device
  copy's buffer is no longer zero-filled first.

### Results

Base against this branch, 9 runs, one host thread and all of them (CPUs 24-31), load 5 to 9. p < 0.001 unless given.

| workload | MI300X, 1 thread | MI300X, 8 | RX 6800, 1 | RX 6800, 8 |
| --- | ---: | ---: | ---: | ---: |
| `api` (13 calls, 5 of them hand-offs) | **3.29x** | **3.32x** | **3.27x** | **3.17x** |
| `memcpy` | **2.22x** | **2.20x** | **2.21x** | **2.13x** |
| `launch_sync` | **1.23x** | **1.24x** | 1.18x (p 0.004) | 1.16x (p 0.003) |
| `stream` | 1.16x | 1.14x | 1.19x | 1.19x |
| `apicheap` (about 30 ns a call) | 1.08x | 1.10x | 1.09x | 1.10x |
| `graph` | 1.04x | 1.04x | 1.04x | 1.04x |
| `launch_mid` (128 work-groups) | 1.04x (p 0.03) | **2.51x** | 1.02x (p 0.01) | **1.88x** |
| `launch_wide` (1024 work-groups) | 1.07x | 1.11x (p 0.49) | 1.03x (p 0.003) | 1.19x (p 0.008) |
| `launch` (one wave, two streams) | 1.01x (p 0.55) | 1.02x (p 0.09) | 1.04x (p 0.30) | 1.05x (p 0.34) |

`launch` is the rate of the stream thread's own work (above), and none of this touches it. CPU time goes down where the hand-offs were
the cost (`api` 1.23 to 0.96 s, `launch_mid` with all threads 2.40 to 1.91 s) and up where a thread now polls before it sleeps:
`stream` +30%, `memcpy` +14%, `launch_sync` +10%, in processes that last 0.1 to 0.4 s; `VGPU_QUEUE_SPIN=0` gives that back and loses
the wall-clock gain below.

The same tree against itself with one change removed (11 to 21 runs, load 7 to 15; `ApiCall`'s single load touches only `apicheap`, whose
base-against-branch row above is its effect, 1.08x to 1.10x):

| change | workload | MI300X | RX 6800 |
| --- | --- | ---: | ---: |
| spinning (`VGPU_QUEUE_SPIN` 400 against 0) | `api` | 2.49x | 2.51x |
| | `memcpy` | 1.82x | 1.77x |
| | `launch_sync` | 1.18x | 1.13x |
| the queue's changes together (spinning, waited-for calls on the caller, the separate counters) | `api` | 3.35x | 3.23x |
| | `memcpy` | 2.24x | 2.24x |
| | `stream` | 1.14x | 1.14x |
| | `launch_sync` | 1.10x (p 0.24) | 1.15x |
| the pool (against a thread for each task, one launch) | `launch_mid`, 8 threads | 2.40x | 1.81x |
| | `launch_wide`, 8 threads | 1.05x (p 0.01) | 1.06x (p 0.02) |
| ids counted up + the main-thread check cached | `launch_wide`, 1 thread | 1.10x (p 0.003) | 1.04x |
| | `launch` | 0.99x | 1.07x (p 0.04) |
| the device-to-device buffer not zeroed | `memcpy` | 1.04x | 1.05x |

The workloads this does not touch (`valu_int`, `valu_f32`, `mem`, `lds`, `atomics`, `divergent`, `wave`, `mfma`, 7 runs, one thread
and all, both cards) moved between 0.998x and 1.08x, none slower with p < 0.05 (the one ratio under 0.95, `mem` on the MI300X at one
thread, 0.86x with p 0.21, has a best-of ratio of 1.19: noise). Small gains of 2 to 5% with p of 0.001 to 0.04 on several (`mfma`
1.03x to 1.04x, `wave` and `atomics` on the RX 6800) are a launch's set-up, where the ids loop is the only change on their path.

Scaling with the final pool, one to eight threads (MI300X, 5 runs, wall time of the device phase, speed-up over one thread):
`mem` 1.96x, 3.81x, 7.43x at 2, 4 and 8; `lds` 1.96x, 3.88x, 7.41x; `divergent` 1.93x, 3.60x, 6.63x; `launch_mid` 1.92x, 3.61x, 6.42x;
`launch_wide` 1.99x, 3.83x, 6.51x; `valu_int` 1.79x, 2.94x, 4.29x (0.336 s at eight threads, the base's time to the millisecond, so it is not this change; probably
the host's other jobs on the sibling threads of CPUs 24-31). No false sharing or shared counter was found: a dispatch's
`DispatchStats` are one for each thread, merged at the end; the allocation table's reader lock spreads threads over 64 slots by a hash
of the thread id (26 distinct slots for 32 threads); the per-thread lookup memory is keyed by the lock's epoch. Past eight threads the host
could not say: 16 CPUs (16-31) gave the same time as 8 for `valu_int` and `mem`, with other jobs on the sibling threads of every core.

### Tests

Both wired into CMake. Mutations were run on scratch copies and each fails the test named.

| test | what it holds | mutations that fail it |
| --- | --- | --- |
| `test_amd_gcn_threads` (`amd/tests/unit`, also run for gfx1100 and gfx1030; the kernels are `vector_add.c` and its LDS reduction) | the pool: n tasks (1 to 40) that wait for one another all run once and at once; the first on the caller; six launching threads do not mix their tasks; a task starts with the launcher's rounding mode and signal mask; a forked child makes its own threads. A dispatch's memory image and counts (waves, instructions, barriers, the class counts and the mix) on 2, 3, 5, 8, 17 and 32 threads equal one thread's, for 1 to 3,000 work-groups of both kernels; 60 dispatches in a row and from four threads at once equal the serial run; a dispatch that fails (a store past its output) throws on 1, 2 and 8 threads and the next one runs | task index off by one; the first task on a new thread; the float environment not applied; the fork child keeping its parent's pool (killed by its alarm); the last work-group of the last range dropped; the first thread's counts not merged; the first error swallowed |
| `test_amd_hip_queue` | order and thread of queued items; `run_inline` on an idle queue (caller's thread, numbered, counted, its failure kept until asked for); an exception is the queue's failure; refused when an item runs or waits or a marker is pending; an item queued while inline work runs waits for it; many threads queueing and waiting, each queued item before the waited-for one that follows it; a wait longer than the spin sleeps and wakes | the sleeper not told to wait for inline work; the idle check dropped; the marker check dropped; the exception not caught; the inline item not numbered; the inline item not marked done. Also clean under ASan, UBSan and TSan |

Also run on the final tree (a full build on server2): every `test_amd_*` unit test (including `test_amd_gcn_same_bits`, which passes
unchanged, `test_amd_gcn_atomics`, `test_amd_gcn_exec`, `test_amd_hip_abi`, `test_amd_rocprofiler`, `test_amd_rocprofiler_args` and the
tests that run for gfx1100, gfx1201, gfx1030 and gfx1250), `test_worker_pool`, `test_memory`, and the AMD end-to-end set: `amd_hip`,
`amd_hipcc`, `amd_graph_host_source`, `amd_hip_abi`, `amd_workgroup`, `amd_hsa`, `amd_hip_on_hsa`, `amd_lost_gpu`, `amd_memtest_patterns`,
`amd_rocprof_*`, `amd_rocblas`, `amd_libraries`, `amd_session`, `amd_debugger`. 202 of 202 pass (`amd_hipsparselt` twice skipped: no
hipSPARSELt). Not run: the PyTorch, vLLM and Ollama sets. Three tests that are not in that list fail on this host with causes that do not involve this change (not re-run on the base: `amd_rocprofiler_abi`: a rocprofiler-sdk header version assertion; `amd_rocprofv3`: `unshare` refused, no user
namespaces; `amd_tools`: the same, and `rocminfo` against ROCm 7.2's).

### Not done, and negative results

* **The NVIDIA engines' worker pool (`vgpu::exec::run_parallel`) for the AMD dispatch, dropped.** The first version of this change
  used it. It is the better pool for a launch with nothing else running, and it is slower than a thread for each task when the CPUs are
  full: `launch_wide` with 8 threads on 8 CPUs 0.64x (MI300X) and 0.70x (RX 6800), p < 0.001 (the launching thread, which it leaves
  waiting and polling, is the ninth thread and takes a CPU from a work-group thread for a time slice, about 200 us a launch);
  `launch_mid` 1.28x (p < 0.001) and 0.94x (p 0.14). `gcn_pool.hpp` runs the first task on the launching thread instead, which
  gave the results above. The shared pool's wait is the same for the NVIDIA engines: worth a look there with the CPUs full.
* **Kernarg alloc and free per launch** (about a third of a `launch`, above): not changed here. A scratch segment moves every later device
  address (the allocator is a bump) and the quarantine of freed ranges, which diagnostics report; making `alloc` and `free` cheaper
  while keeping both means the allocation table's nodes (`live_`, `freed_`, `freed_order_`, the last a map that is only ever popped at
  the front), the lock's epoch and a 64 KiB zeroed chunk for 192 bytes. That is `src/core/memory.cpp` and `brlock.hpp`, not this piece;
  it was done afterwards, keeping every address and diagnostic: "The allocation table and a launch's bookkeeping" at the end.
* **The wave's 64 KiB register file** (14% of `launch` in `set_up_group`, with the ids loop that is gone): the previous round's
  conclusion stands, no bit-safe way to size or recycle it was found.
* **Dynamic scheduling of work-groups across threads**: not done. The ranges are static and contiguous so that groups that wait on later
  ones on their own thread (Stream-K GEMMs) behave as they did; handing out groups one by one changes which groups share a thread.
* **Running small asynchronous copies, fills and event records on the caller when the stream is idle**: not done. It would be faster
  (the hand-off is the cost) but it changes when `hipEventQuery` and `hipStreamQuery` first say yes, and a program may see that.
* **Reusing a stream's thread for the next stream** (`stream` spends 30% in `pthread_create` and thread exit): not done, not
  measured. The queue's thread runs host functions, which may use thread-local state of the runtime.
* **Graph launches**: 12 us for four kernel nodes, 3 us a node, the cost of a plain launch's stream-thread work; the plan is rebuilt
  each launch but is small beside it. Nothing cheap.
* **Event records**: `hipEventRecord` takes `s.mutex` and the event table's mutex and queues an item; no measurement pointed at it.


## The AMD pieces together (branch `speedup5-amdintegrate`)

`speedup` b25ab636 (Tier 1 and 2, amdcorrect, amdmatrix, amdlibs) with `speedup4-amdmemtext`, `speedup4-amdtier` and `speedup4-amdmt` merged in that
order. Every golden file is as recorded (none re-recorded; the base's build still passes them, so they still describe it).

**Where the three clashed, and what was kept.**

* `digest_instruction` / `time_instruction` / `digest_memory_instruction` (`gcn_exec.cpp`, `amd_exec.hpp`): all three probes, the `error` out-parameter
  of the first included. Flavour 4 of `draw` was taken twice: amdmatrix's (every byte 0x3C to 0x41, used by `test_amd_matrix_fast` and
  `amd_matrix_bench`) and the tier's NaN-free floats. The tier's six flavours (4 to 9 on their branch) are now 10 to 15, in `draw`,
  `digest_instruction` and `test_amd_gcn_tier_same_bits`; flavours 0 to 3, which the golden digests use, are untouched.
* `Machine::step`: the tier hook first (`tier_step`), then the lazy `InstCopy` of amdmemtext for VGPR MSBs and VGPR indexing; `Inst::fast` and
  `Inst::mem` both live in `Inst`, and `finalize` fills in both.
* `lds_access`: amdcorrect's `ds_nop` / append / consume guard first, then amdmemtext's `MemFacts` and `lds_fast`.
* Dispatch threads: `pool::run` (amdmt) is the launcher, and it degrades as `start_or_run_here` did: a task the host gives no thread runs on the
  launching thread after the task of its own, in turn, with the bits it would have had (`Pool::Refused::Degrade`). The lockstep launch (tasks that
  wait on one another) uses `pool::run_together` (`Refused::Strict`): every thread is found or made before any task starts, and one the host
  refuses throws `std::system_error` with nothing started, which the dispatch reports as amdcorrect's "the host would not start a thread for the
  dispatch's work-groups: ..." error. `start_or_run_here` is gone. `test_amd_gcn_thread_refused` has a second test for both modes.
* `hip_api.cpp` stream submit: amdmt's `run_inline` and spinning queue, with `no_thread` (hipErrorOutOfMemory) around both `submit` calls.
* The NaN trap again: the merge moved GCC's choice of operand order in `wmma1250`'s scaled products and sums, and 73 gfx1250 WMMA shapes of
  `test_amd_gcn_same_bits` changed their digests. The four operations are pinned (`mfma_mul_o<0>` twice, `mfma_add_o<2>`, `mfma_add_o<0>`), the
  order found by trying all sixteen choices against the recorded digests (one passed).
* `bench-amd`: every workload of every piece is kept; `--counts-via probe` works for all of them; `expected/` gained the five matrix workloads
  that had no entry (recorded with the probe from b25ab636 and equal to the integrated tree's), nothing else changed.

**A/B** (`tools/bench-amd/bench.sh`, 9 runs alternating, one thread and all threads, CPUs 24-31 under `flock /tmp/speedup-bench.lock`, size
full, without counting runs (f81cc0c1 has no probe; the work is identical, `counters identical` is `yes` in every row), server2 load average
1 to 4 before each workload, median ratio B/A, p < 0.001 in every row except `startup`). Columns: the integrated tree against `speedup` f81cc0c1
and against b25ab636, and each of the three pieces' own branch against f81cc0c1 (the figures the pieces reported).

| workload | target | thr | integrated / f81cc0c1 | integrated / b25ab636 | tier alone / f81 | memtext alone / f81 | mt alone / f81 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valu_int | mi300x | 1 | 8.51 | 8.72 | 8.83 |  |  |
| valu_int | mi300x | all | 8.49 | 8.59 |  |  |  |
| valu_int | rx6800 | 1 | 7.01 | 7.15 | 7.45 |  |  |
| valu_int | rx6800 | all | 7.01 | 7.09 |  |  |  |
| valu_f32 | mi300x | 1 | 9.36 | 9.26 | 9.71 |  |  |
| valu_f32 | mi300x | all | 9.61 | 9.44 |  |  |  |
| valu_f32 | rx6800 | 1 | 6.49 | 6.36 | 6.88 |  |  |
| valu_f32 | rx6800 | all | 6.72 | 6.54 |  |  |  |
| valu_f64 | mi300x | 1 | 4.58 | 4.86 | 4.60 |  |  |
| valu_f64 | mi300x | all | 4.59 | 4.82 |  |  |  |
| valu_f64 | rx6800 | 1 | 5.64 | 5.73 | 5.38 |  |  |
| valu_f64 | rx6800 | all | 5.37 | 5.57 |  |  |  |
| mfma | mi300x | 1 | 2.70 | 1.08 | 1.05 |  |  |
| mfma | mi300x | all | 2.72 | 1.08 |  |  |  |
| mfma_i8 | mi300x | 1 | 4.95 | 0.97 |  |  |  |
| mfma_i8 | mi300x | all | 4.96 | 0.98 |  |  |  |
| mfma_fp8 | mi300x | 1 | 7.45 | 1.04 |  |  |  |
| mfma_fp8 | mi300x | all | 7.45 | 1.04 |  |  |  |
| mfma_bf16 | mi300x | 1 | 3.37 | 1.06 |  |  |  |
| mfma_bf16 | mi300x | all | 3.36 | 1.06 |  |  |  |
| mfma_f64 | mi300x | 1 | 3.73 | 1.12 |  |  |  |
| mfma_f64 | mi300x | all | 3.25 | 1.11 |  |  |  |
| smfmac | mi300x | 1 | 3.00 | 1.03 |  |  |  |
| smfmac | mi300x | all | 3.01 | 1.02 |  |  |  |
| mem | mi300x | 1 | 3.08 | 3.10 | 2.99 | 1.10 |  |
| mem | mi300x | all | 3.09 | 3.12 |  |  |  |
| mem | rx6800 | 1 | 3.28 | 3.14 | 3.28 | 1.13 |  |
| mem | rx6800 | all | 3.27 | 3.17 |  |  |  |
| lds | mi300x | 1 | 3.79 | 3.71 | 3.66 | 2.13 |  |
| lds | mi300x | all | 3.87 | 3.86 |  |  |  |
| lds | rx6800 | 1 | 3.30 | 3.31 | 3.12 | 1.90 |  |
| lds | rx6800 | all | 3.30 | 3.33 |  |  |  |
| narrow | mi300x | 1 | 1.94 | 1.98 | 1.91 | 1.11 |  |
| narrow | mi300x | all | 1.95 | 1.99 |  |  |  |
| narrow | rx6800 | 1 | 2.10 | 2.05 | 2.11 | 1.10 |  |
| narrow | rx6800 | all | 2.08 | 2.06 |  |  |  |
| smem | mi300x | 1 | 3.60 | 3.81 | 3.30 | 1.11 |  |
| smem | mi300x | all | 3.64 | 3.85 |  |  |  |
| smem | rx6800 | 1 | 3.12 | 3.17 | 2.81 | 1.18 |  |
| smem | rx6800 | all | 3.15 | 3.23 |  |  |  |
| buf | mi300x | 1 | 1.97 | 1.92 | 1.39 | 1.32 |  |
| buf | mi300x | all | 1.98 | 1.93 |  |  |  |
| buf | rx6800 | 1 | 2.06 | 2.02 | 1.45 | 1.32 |  |
| buf | rx6800 | all | 2.07 | 2.03 |  |  |  |
| ldsx | mi300x | 1 | 6.96 | 7.00 | 2.70 | 2.28 |  |
| ldsx | mi300x | all | 7.39 | 7.37 |  |  |  |
| ldsx | rx6800 | 1 | 6.32 | 6.17 | 2.83 | 2.02 |  |
| ldsx | rx6800 | all | 6.46 | 6.40 |  |  |  |
| flat | mi300x | 1 | 3.35 | 3.29 | 2.64 | 1.13 |  |
| flat | mi300x | all | 3.27 | 3.31 |  |  |  |
| flat | rx6800 | 1 | 3.50 | 3.55 | 2.78 | 1.13 |  |
| flat | rx6800 | all | 3.53 | 3.54 |  |  |  |
| atomics | mi300x | 1 | 2.05 | 2.08 | 1.85 | 1.15 |  |
| atomics | mi300x | all | 1.96 | 1.91 |  |  |  |
| atomics | rx6800 | 1 | 2.23 | 2.25 | 1.98 | 1.19 |  |
| atomics | rx6800 | all | 2.06 | 1.97 |  |  |  |
| divergent | mi300x | 1 | 4.40 | 4.47 | 4.53 |  |  |
| divergent | mi300x | all | 4.41 | 4.48 |  |  |  |
| divergent | rx6800 | 1 | 4.30 | 4.21 | 4.32 |  |  |
| divergent | rx6800 | all | 4.24 | 4.27 |  |  |  |
| wave | mi300x | 1 | 11.12 | 11.03 | 11.41 |  |  |
| wave | mi300x | all | 10.92 | 10.75 |  |  |  |
| wave | rx6800 | 1 | 8.84 | 8.79 | 9.12 |  |  |
| wave | rx6800 | all | 8.74 | 8.66 |  |  |  |
| launch | mi300x | 1 | 1.08 | 1.14 |  |  | 1.05 |
| launch | mi300x | all | 1.08 | 1.14 |  |  | 0.99 |
| launch | rx6800 | 1 | 1.14 | 1.15 |  |  | 1.04 |
| launch | rx6800 | all | 1.14 | 1.14 |  |  | 1.04 |
| api | mi300x | 1 | 3.27 | 3.23 |  |  | 3.50 |
| api | mi300x | all | 3.34 | 3.44 |  |  | 3.49 |
| api | rx6800 | 1 | 3.45 | 3.30 |  |  | 3.44 |
| api | rx6800 | all | 3.30 | 3.27 |  |  | 3.31 |
| apicheap | mi300x | 1 | 1.09 | 1.10 |  |  | 1.10 |
| apicheap | mi300x | all | 1.10 | 1.10 |  |  | 1.11 |
| apicheap | rx6800 | 1 | 1.11 | 1.09 |  |  | 1.11 |
| apicheap | rx6800 | all | 1.11 | 1.09 |  |  | 1.12 |
| launch_wide | mi300x | 1 | 2.31 | 2.33 |  |  | 1.04 |
| launch_wide | mi300x | all | 2.66 | 2.70 |  |  | 1.14 |
| launch_wide | rx6800 | 1 | 2.09 | 2.06 |  |  | 1.01 |
| launch_wide | rx6800 | all | 1.42 | 1.62 |  |  | 1.01 |
| launch_mid | mi300x | 1 | 2.31 | 2.33 |  |  | 1.05 |
| launch_mid | mi300x | all | 4.86 | 4.85 |  |  | 2.46 |
| launch_mid | rx6800 | 1 | 2.08 | 2.06 |  |  | 1.02 |
| launch_mid | rx6800 | all | 3.57 | 3.50 |  |  | 1.93 |
| launch_sync | mi300x | 1 | 1.61 | 1.59 |  |  | 1.22 |
| launch_sync | mi300x | all | 1.62 | 1.62 |  |  | 1.23 |
| launch_sync | rx6800 | 1 | 1.55 | 1.54 |  |  | 1.16 |
| launch_sync | rx6800 | all | 1.56 | 1.55 |  |  | 1.16 |
| memcpy | mi300x | 1 | 2.11 | 2.16 |  |  | 2.17 |
| memcpy | mi300x | all | 2.13 | 2.18 |  |  | 2.17 |
| memcpy | rx6800 | 1 | 2.12 | 2.15 |  |  | 2.11 |
| memcpy | rx6800 | all | 2.13 | 2.12 |  |  | 2.15 |
| stream | mi300x | 1 | 1.21 | 1.22 |  |  | 1.16 |
| stream | mi300x | all | 1.22 | 1.21 |  |  | 1.18 |
| stream | rx6800 | 1 | 1.29 | 1.47 |  |  | 1.20 |
| stream | rx6800 | all | 1.26 | 1.45 |  |  | 1.20 |
| graph | mi300x | 1 | 1.36 | 1.36 |  |  | 1.04 |
| graph | mi300x | all | 1.38 | 1.36 |  |  | 1.04 |
| graph | rx6800 | 1 | 1.33 | 1.34 |  |  | 1.03 |
| graph | rx6800 | all | 1.33 | 1.33 |  |  | 1.04 |
| startup | mi300x | 1 | 1.04 | 0.98 |  |  |  |
| startup | mi300x | all | 1.02 | 1.00 |  |  |  |
| startup | rx6800 | 1 | 1.01 | 1.00 |  |  |  |
| startup | rx6800 | all | 1.01 | 1.01 |  |  |  |
| wmma | rx7900xtx | 1 | 25.12 | | | | |
| wmma | rx7900xtx | all | 25.38 | | | | |
| wmma | rx9070xt | 1 | 25.30 | | | | |
| wmma | rx9070xt | all | 25.63 | | | | |
| wmma_iu8 | rx7900xtx | 1 | 9.24 | | | | |
| wmma_iu8 | rx7900xtx | all | 9.50 | | | | |
| wmma_iu8 | rx9070xt | 1 | 7.16 | | | | |
| wmma_iu8 | rx9070xt | all | 7.34 | | | | |

(`startup` is unchanged, 0.98 to 1.05. `wmma` and `wmma_iu8` on the simulated RX 7900 XTX and RX 9070 XT: 25.1 to 25.6 x and 7.2 to 9.5 x f81cc0c1.
`v_pk_fma_f16` costs 200 to 260 ns an execution in b25ab636 and in the integrated tree alike (`amd_matrix_bench`; f81cc0c1's was 3103 ns), a
`v_mfma_f32_16x16x16_f16` 1.6 us in both: the tier takes no packed 16-bit or matrix instruction, so those gains are untouched.)

**What the claims became.** Tier: 4.6 to 11.1 x on `valu_*` / `wave` / `divergent` (alone 4.6 to 11.4 x; at most 6 % less, rx6800 `valu_f32`
and `valu_int`). Mt: `api` 3.3 to 3.4 x (alone 3.3 to 3.5 x), `memcpy` 2.1 x (2.1 to 2.2 x), `launch_mid` with all threads 4.9 / 3.6 x
(alone 2.5 / 1.9 x), no row below 0.93 of the piece alone. Matrix: the `mfma*` rows are 2.7 to 7.4 x and `wmma` 25 x, the same as b25ab636's
(`mfma` is 1.08 x faster than b25ab636 for the tier's scalar and branch instructions). Libs: unchanged, as above.
**One piece's gain shrank: amdmemtext's, on the instructions the tier now runs itself.** Its contribution on top of the tier (integrated / tier
alone) against what it gave alone: `mem` 1.00 to 1.03 x (alone 1.10 to 1.13), `lds` 1.04 to 1.06 x (alone 1.9 to 2.1), `narrow` 1.00 to 1.02 x
(alone 1.10), `atomics` 1.11 to 1.13 x (alone 1.15 to 1.19); not shrunk where the tier does not reach: `ldsx` 2.2 to 2.6 x (alone 2.0 to 2.3),
`buf` 1.42 x (1.32), `flat` 1.26 x (1.13), `smem` 1.09 to 1.11 x (1.11 to 1.18). The memtext code is still what runs the instructions the tier hands
back (a wave with VGPR MSBs or indexing, MODE not at its default, the debugger or tracing, an access that is not plain, wide LDS forms) and
the interpreter's own steps of the other workloads.

**Tests** (server2, Release; each with `VGPU_AMD_TIER=1` and `=0`): `ctest -R "amd|gcn|rocprof|hipcc|hip_|memrmw|memruns|mem_facts|mfma|packed|matrix|tier|bench_amd"`,
251 tests, 3 failed in both modes (the rest passed or were skipped), the same three as in b25ab636 (`amd_rocprofiler_abi`: the host's rocprofiler-sdk is another
version; `amd_rocprofv3` and `amd_tools`: unprivileged user namespaces and `rocminfo` do not work on server2); the rest skipped for want of
ROCm libraries or cards. Among them `test_amd_gcn_same_bits`, `_packed_same_bits`, `_memory_same_bits`, `_tier_same_bits`,
`test_amd_codeobject_same_bits`, `test_amd_matrix_fast`, `test_amd_gcn_threads*`, `test_amd_hip_queue`, `test_amd_gcn_thread_refused`,
`test_amd_hip_thread_refused`, `bench_amd_counters`. The NVIDIA tests that share code (`test_memory`, `test_memrmw*`, `test_worker_pool`,
`test_fastpath`, `test_exec*`): 8 of 8.


## The speedup branch merged with origin/main (#382, #400) (branch `speedup6-main`)

`speedup` bb46caaf with origin/main ab904cbc: "AMD: gfx942/gfx950 guide sweep, gfx90a images, hostcall, PC sampling, RVS, vLLM graphs"
(#382, which changed the behaviour of many instructions and edited `gcn_exec.cpp` inside the old if-ladders) and #400 (cuTensorNet).

**How the 32 conflicts of `gcn_exec.cpp` were merged.** `amd/tools/ladder-to-switch.py` is a deterministic rewrite, so the file was merged three
ways *after* it: the merge base's ladders converted, origin/main's converted, and `speedup`'s file (already converted), `git merge-file`
over the three. That took the 32 hunks down to 21, all real: the decoded memory facts of amdmemtext against #382's text-based memory code, the
tile MFMA code of amdmatrix against #382's MFMA changes, the SOPP switch against new SOPP rungs, `Machine::step`, and the packed and conversion
paths. Each was written by hand keeping both sides (and the build script's two `for src in` lists were joined).

**The golden digests were re-recorded, from origin/main plus the probe only.** The arbiter is origin/main ab904cbc with the probe functions
(`digest_instruction`, `digest_memory_instruction`, `time_instruction`, and stubs of `set_matrix_fast_path`) added and nothing else
(the same four test programs, built there). Against the committed digests, run on the *old* corpora (so that every seed and position
is the same), the arbiter differs from them in 1,044 of 16,304 `same_bits` shapes, 336 of 2,143 `packed_same_bits` shapes and 66 of
15,566 `memory_same_bits` shapes; every one is a thing #382 changed on purpose:

| family (shapes changed) | what #382 did |
| --- | --- |
| MFMA and SMFMAC, 639 | CBSZ/ABID checks and BLGP lane permutations, the f8f6f4 K layout, denormal flush of float inputs, sparse products skipping pruned places, the lane layout of several-block forms |
| `v_div_scale`, `v_div_fmas`, `v_div_fixup` f16/f32/f64, 193 | the division sequence of the guides' pseudocode (`gcn_divide.hpp`) |
| SOPP corpus, 60 | `s_setkill`, `s_sethalt`, `s_set_gpr_idx_mode`, `s_cbranch_cdbg*`, `s_incperflevel`, VSKIP, ... |
| memory, 43 + 66 | 64-bit and `cond_sub`/`sub_clamp` atomics, ADDTID, buffer loads into LDS, scalar buffer stores and atomics |
| `v_dot2*` 10 + 237, `v_cvt_f32_f16` 40 + 39, `v_sin`/`v_cos` 4 + 60, min/max, `v_fma_f16`, omod | the guides' denormal flush, `abs` of a NaN, `v_sin`/`v_cos` in turns, omod ignored by MODE |
| scalar, 27 | `s_movrel*`, `s_round_mode`, `s_max` SCC on a tie, ... |

#382 also grew the corpora (`isa_corpus.txt`, `_gfx90a`, `_gfx950`, `_images_gfx90a`: 16,806 shapes now, 502 more), and the seed of a
shape is its position, so `same_bits_digests.txt` and `mem_same_bits_digests.txt` are re-recorded whole from the arbiter, `packed_same_bits_digests.txt`
whole as well (its corpus did not change; 336 digests did), and `codeobject_digests.txt` from the arbiter with the six entries of `speedup`'s own
code objects kept (one hipcc object's digest differs: #382 rebuilt it). The merged tree passes all of them as they are committed; the arbiter
build, the old-corpora run and the recorded files are the evidence, and no digest was taken from this tree.

**What the merge itself broke, found by those digests and the tests**, each fixed without touching a golden file:
* GCC's operand order for two NaNs or two zeros moved (the trap of every round): `wmma1250`'s scaled products and sums, the packed min and max
  on RDNA and gfx90a, `v_dot2_f32_f16` and `v_dot2c_f32_f16` (one order for the guides' flushing path, another for the rest) are pinned (`first_mul`,
  `first_add`, `pk_fminf`, `pk_fmaxf`); `v_maxmin_f32` and its kin call libm's `fminf`/`fmaxf` explicitly, as the recorded build did.
* `v_cvt_f32_f16` with `|x|`: `lane_half_f` cleared the sign of a NaN by comparison; #382 clears the bit.
* `gfx9_added_alu` renames an instruction (`v_mad_legacy_u16` to `v_mad_u16`): the copy is now `finalize`d (the hash and the flags are of the name).
* The matrix tile path hands to the reference whatever #382 changed in the reference: the small-float and scaled forms, CBSZ past the blocks, BLGP,
  several blocks with several groups of K a lane; a sparse tile whose B has a NaN or infinity, a float tile with a denormal input (flush); EXEC is
  ignored, as it is on the card. (`v_mfma_*_f8f6f4` and the scaled forms run at the reference's speed until the tile code learns the new K layout.)
* The tier: no handler where gfx942/gfx950 apply or ignore an output modifier or a clamp, `s_max_*` SCC on a tie has its own handlers there, `v_lshl_add_u64`
  has a guide-architecture one (count over 4 is zero), and no tier while PC sampling is attached or VSKIP is set.
* `dot2_f16_wave` (amdlibs' whole-wave dot product) is not used on gfx942/gfx950, whose dot products flush denormals.
* A tbuffer instruction through the private segment buffer stays refused, as in origin/main and the base (amdmemtext's decoded facts would have run it;
  the corpora now have the shape).
* `ds_append`/`ds_consume` run (#382 models them); amdcorrect's "not modelled" guard lets them through, and its test says so. The decoded memory
  facts know the new names (`cond_sub`, `sub_clamp`, ADDTID, `s_buffer_*`, the volatile cache forms).

**Tests** (server2, Release, each with `VGPU_AMD_TIER=1` and `=0`): `ctest -R "amd|gcn|rocprof|hipcc|hip_|memrmw|memruns|mem_facts|mfma|packed|matrix|tier|bench_amd|hostcall|pcsampl|rvs|vllm|graph|pytorch"`,
310 tests; the same three fail as on `speedup` and on origin/main (`amd_rocprofiler_abi`, `amd_rocprofv3`, `amd_tools`: this host's ROCm and user
namespaces); the vLLM, PyTorch and `amd_rvs` ones are skipped for want of the packages. #382's own (`test_amd_gcn_cdna_gaps`, `test_amd_gcn_divide`,
`test_amd_hostcall`, `test_amd_rocprofiler_pcsampling*`, and the new programs of `amd_hipcc`) pass. #400's and the shared NVIDIA ones
(`e2e_cutensornet_*`, `e2e_ipc`, `test_memory`, `test_memrmw*`, `test_worker_pool`, `test_fastpath`, `test_exec*`, `e2e_sass_tier`): 24 of 24.

**A/B** (`tools/bench-amd`, 9 runs alternating, 1 thread and all threads, CPUs 24-31 under `flock`, full size, `--no-counts` because origin/main has no probe;
`counters identical: yes` in every row; p < 0.001 except `startup`): the merged tree against a build of origin/main ab904cbc, and (one thread) against
`speedup` bb46caaf, to see what the merge cost. Ratio of the medians, > 1 is faster.

| workload | target | thr | merged / origin/main ab904cbc | merged / speedup bb46caaf (1 thread) |
| --- | --- | ---: | ---: | ---: |
| valu_int | mi300x | 1 | 20.22 | 0.99 |
| valu_int | mi300x | all | 20.38 |  |
| valu_int | rx6800 | 1 | 18.04 | 1.00 |
| valu_int | rx6800 | all | 17.87 |  |
| valu_f32 | mi300x | 1 | 21.32 | 1.04 |
| valu_f32 | mi300x | all | 21.08 |  |
| valu_f32 | rx6800 | 1 | 16.65 | 0.99 |
| valu_f32 | rx6800 | all | 17.52 |  |
| valu_f64 | mi300x | 1 | 9.30 | 0.91 |
| valu_f64 | mi300x | all | 9.37 |  |
| valu_f64 | rx6800 | 1 | 13.89 | 0.95 |
| valu_f64 | rx6800 | all | 13.89 |  |
| mfma | mi300x | 1 | 7.53 | 0.95 |
| mfma | mi300x | all | 7.58 |  |
| mfma_i8 | mi300x | 1 | 7.39 | 1.00 |
| mfma_i8 | mi300x | all | 7.53 |  |
| mfma_fp8 | mi300x | 1 | 9.12 | 0.94 |
| mfma_fp8 | mi300x | all | 9.09 |  |
| mfma_bf16 | mi300x | 1 | 6.22 | 0.94 |
| mfma_bf16 | mi300x | all | 6.29 |  |
| mfma_f64 | mi300x | 1 | 5.04 | 0.99 |
| mfma_f64 | mi300x | all | 4.59 |  |
| smfmac | mi300x | 1 | 7.11 | 0.89 |
| smfmac | mi300x | all | 7.26 |  |
| mem | mi300x | 1 | 5.85 | 0.98 |
| mem | mi300x | all | 5.88 |  |
| mem | rx6800 | 1 | 7.14 | 0.99 |
| mem | rx6800 | all | 7.21 |  |
| lds | mi300x | 1 | 11.31 | 0.98 |
| lds | mi300x | all | 11.73 |  |
| lds | rx6800 | 1 | 9.45 | 0.94 |
| lds | rx6800 | all | 9.60 |  |
| narrow | mi300x | 1 | 2.92 | 0.96 |
| narrow | mi300x | all | 3.04 |  |
| narrow | rx6800 | 1 | 3.80 | 0.99 |
| narrow | rx6800 | all | 3.85 |  |
| smem | mi300x | 1 | 8.69 | 0.89 |
| smem | mi300x | all | 8.88 |  |
| smem | rx6800 | 1 | 8.69 | 0.81 |
| smem | rx6800 | all | 8.73 |  |
| buf | mi300x | 1 | 2.69 | 1.01 |
| buf | mi300x | all | 2.76 |  |
| buf | rx6800 | 1 | 3.27 | 1.00 |
| buf | rx6800 | all | 3.30 |  |
| ldsx | mi300x | 1 | 27.56 | 0.95 |
| ldsx | mi300x | all | 28.99 |  |
| ldsx | rx6800 | 1 | 22.95 | 0.95 |
| ldsx | rx6800 | all | 23.79 |  |
| flat | mi300x | 1 | 6.67 | 0.95 |
| flat | mi300x | all | 6.70 |  |
| flat | rx6800 | 1 | 7.85 | 0.92 |
| flat | rx6800 | all | 7.85 |  |
| atomics | mi300x | 1 | 4.28 | 0.98 |
| atomics | mi300x | all | 24.35 |  |
| atomics | rx6800 | 1 | 5.86 | 1.01 |
| atomics | rx6800 | all | 16.89 |  |
| divergent | mi300x | 1 | 12.94 | 1.00 |
| divergent | mi300x | all | 12.87 |  |
| divergent | rx6800 | 1 | 13.89 | 1.00 |
| divergent | rx6800 | all | 13.88 |  |
| wave | mi300x | 1 | 18.76 | 0.97 |
| wave | mi300x | all | 18.60 |  |
| wave | rx6800 | 1 | 16.99 | 1.00 |
| wave | rx6800 | all | 16.85 |  |
| launch | mi300x | 1 | 1.44 | 1.00 |
| launch | mi300x | all | 1.44 |  |
| launch | rx6800 | 1 | 1.48 | 1.00 |
| launch | rx6800 | all | 1.47 |  |
| api | mi300x | 1 | 3.21 | 1.00 |
| api | mi300x | all | 3.35 |  |
| api | rx6800 | 1 | 3.33 | 1.01 |
| api | rx6800 | all | 3.34 |  |
| apicheap | mi300x | 1 | 1.11 | 0.98 |
| apicheap | mi300x | all | 1.09 |  |
| apicheap | rx6800 | 1 | 1.11 | 0.99 |
| apicheap | rx6800 | all | 1.10 |  |
| launch_wide | mi300x | 1 | 4.27 | 1.01 |
| launch_wide | mi300x | all | 3.25 |  |
| launch_wide | rx6800 | 1 | 4.08 | 1.02 |
| launch_wide | rx6800 | all | 3.17 |  |
| launch_mid | mi300x | 1 | 4.26 | 1.02 |
| launch_mid | mi300x | all | 6.31 |  |
| launch_mid | rx6800 | 1 | 4.09 | 1.01 |
| launch_mid | rx6800 | all | 5.46 |  |
| launch_sync | mi300x | 1 | 2.15 | 1.00 |
| launch_sync | mi300x | all | 2.20 |  |
| launch_sync | rx6800 | 1 | 2.10 | 1.00 |
| launch_sync | rx6800 | all | 2.10 |  |
| memcpy | mi300x | 1 | 2.17 | 1.01 |
| memcpy | mi300x | all | 2.18 |  |
| memcpy | rx6800 | 1 | 2.18 | 0.99 |
| memcpy | rx6800 | all | 2.19 |  |
| stream | mi300x | 1 | 1.47 | 0.99 |
| stream | mi300x | all | 1.44 |  |
| stream | rx6800 | 1 | 1.63 | 0.98 |
| stream | rx6800 | all | 1.55 |  |
| graph | mi300x | 1 | 1.91 | 1.01 |
| graph | mi300x | all | 1.91 |  |
| graph | rx6800 | 1 | 1.86 | 1.01 |
| graph | rx6800 | all | 1.87 |  |
| startup | mi300x | 1 | 1.06 | 1.09 |
| startup | mi300x | all | 1.07 |  |
| startup | rx6800 | 1 | 1.02 | 0.99 |
| startup | rx6800 | all | 1.02 |  |
| wmma | rx7900xtx | 1 | 26.52 | |
| wmma | rx7900xtx | all | 26.82 | |
| wmma | rx9070xt | 1 | 27.66 | |
| wmma | rx9070xt | all | 28.24 | |
| wmma_iu8 | rx7900xtx | 1 | 10.34 | |
| wmma_iu8 | rx7900xtx | all | 10.52 | |
| wmma_iu8 | rx9070xt | 1 | 7.46 | |
| wmma_iu8 | rx9070xt | all | 7.77 | |

Against origin/main every workload is faster (startup within noise); #382's guide sweep made main slower than the `speedup` pieces' baseline f81cc0c1
on the gfx942 workloads (the ratios are larger than the 2-11 x of the pieces). Against `speedup` before the merge, the gfx942 and RX 6800 rows
lost 3 to 19 % (`smem` 11 % and 19 %, `smfmac` 11 %, `valu_f64` on the MI300X 9 %, `flat` on the RX 6800 8 %, the MFMA rows 5 to 6 %,
`lds` and `ldsx` 5 to 6 %): the checks #382 added to the interpreter's `step` (PC sampling, VSKIP), the denormal flushes and the omod/clamp rules of the guides, and
the tile path's declines. Not profiled further. The first run of this A/B had `mem` and `flat` on the MI300X at 0.46 x and 0.48 x and
`launch_wide`/`launch_mid` at 0.68 x: the tier had been told not to run `v_lshl_add_u64` on gfx942, which the address arithmetic of those kernels uses;
it has a handler now.

## The AMD merge's cost, found and won back (branch `sn-amdmerge`)

The merge of #382 into `speedup` (above) left the gfx942 and RX 6800 rows 3 to 19 % slower than `speedup` before it (`bb46caaf`), "not profiled
further". Profiled now, with the SIGPROF sampler (`tools/bench-amd/profile/`, a `-g` build of each tree, `SIGPROF_STACK=1`), instruction counts and the
set of instructions that run in the interpreter (`VGPU_AMD_TIER_STATS=1`) identical in both trees on every bench workload (so the tier's declines, the
omod/clamp rules and the denormal flushes are not the cause: nothing the benchmarks run was handed back), the causes were three:

* **`scalar_alu` was an if-ladder of 81 rungs again.** The interpreter's `op == "name"_op` ladders are switches on the name's hash
  (`amd/tools/ladder-to-switch.py`); #382 edited inside the old ladders, and the merge kept its ladder form for `scalar_alu`. One rung of it
  (`s_flbit_i32_i64`, #382's) had a `break` inside a `for`, which is outside a loop of the converter's own: the converter leaves a chain alone that has one (a switch would
  capture the `break`), and said nothing (`chains: 0`). Every scalar instruction the tier does not do (`s_mul_hi_u32`, `s_cselect`, `s_lshr_b32` ...) walked down up
  to 81 hash comparisons: 9 % of the RX 6800 `smem` kernel's samples (`OpName::operator==`). The rung is written without the `break`, the converter run, and
  `lint_amd_ladders` (`tests/lint/check_ladders.py`) fails for any chain of 12 or more instruction names left as an if-ladder in `gcn_exec.cpp`
  (on the file before this change it fails with `gcn_exec.cpp:1484: an if-ladder of 81 rungs naming 111 instructions`).
* **`scalar_load` grew a frame.** #382 added the buffer stores and buffer atomics to it, and the plain dword load (`s_load_dword*`, the case that matters) paid
  for the function's size. The plain load is the function now (`scalar_load`, address in `scalar_load_base`) and the rest is `scalar_load_general`, out of line.
* **`Machine::step` was one big function.** Its frame (0x6f8 bytes, a stack protector, six callee-saved registers pushed) was paid for by every
  instruction, the tier's included, and #382's additions (PC sampling, VSKIP) made it bigger. `step` is now the debugger check (once per machine, `debug_on`,
  instead of a call to `debug::active()` per instruction; the session's check is `debug_check`, out of line), the fetch and the tier; the interpreter's own work is
  `step_interp`, out of line. This is where the rows beyond the pre-merge numbers come from.

Alternating A/B, `tools/bench-amd/bench.sh --build pre --build work`, 11 runs, size full, 1 thread, pinned to CPUs 20-27 of server2 (load 14 to 26: other
agents' builds), median ratio of the rate (> 1: faster); the A/A noise on the same rows (two copies of one build) is 0.2 to 1 %. `pre` is `bb46caaf`;
"merged" is the ratio the merge had (the table above); counters identical in every row.

| workload | target | merged / pre | now / pre |
| --- | --- | ---: | ---: |
| valu_int | mi300x | | 1.22 |
| valu_int | rx6800 | | 1.21 |
| valu_f32 | mi300x | | 1.13 |
| valu_f32 | rx6800 | | 1.05 |
| valu_f64 | mi300x | 0.91 | 1.08 |
| valu_f64 | rx6800 | 0.95 | 1.10 |
| mfma | mi300x | 0.95 | 1.05 |
| mfma_i8 | mi300x | 1.00 | 1.07 |
| mfma_fp8 | mi300x | 0.94 | 0.89 |
| mfma_bf16 | mi300x | 0.94 | 1.12 |
| mfma_f64 | mi300x | 0.99 | 1.00 |
| smfmac | mi300x | 0.89 | 0.93 |
| mem | mi300x | 0.98 | 1.05 |
| mem | rx6800 | 0.99 | 1.08 |
| lds | mi300x | 0.98 | 1.09 |
| lds | rx6800 | 0.94 | 1.13 |
| narrow | mi300x | 0.96 | 0.98 |
| narrow | rx6800 | 0.99 | 1.04 |
| smem | mi300x | 0.89 | 1.22 |
| smem | rx6800 | 0.81 | 1.27 |
| buf | mi300x | 1.01 | 1.03 |
| buf | rx6800 | 1.00 | 1.04 |
| ldsx | mi300x | 0.95 | 1.12 |
| ldsx | rx6800 | 0.95 | 1.19 |
| flat | mi300x | 0.95 | 1.04 |
| flat | rx6800 | 0.92 | 1.05 |
| atomics | mi300x | 0.98 | 1.05 |
| atomics | rx6800 | 1.01 | 1.06 |
| divergent | mi300x | 1.00 | 1.21 |
| divergent | rx6800 | 1.00 | 1.27 |
| wave | mi300x | 0.97 | 1.20 |
| wave | rx6800 | 1.00 | 1.26 |

The `scalar_alu` switch alone (before the other two) took `smem` from 0.88 and 0.84 of `pre` to 0.96 and 0.98, `lds` on the RX 6800 5 % up; the other two
pieces gave the rest. Left: `mfma_fp8` (0.89) and `smfmac` (0.93). Profiled, the matrix tile path's code is the same in both trees and the
samples are on the same lines; the extra time is in the small-float table lookup of the decode (`matrix_elem<'e'>`: 61 samples of `pre`, 63 of the merged tree, 100 of the tree with
the split `step`), and by direct runs the merged tree was 0.97 of `pre` on `mfma_fp8` and 0.91 on `smfmac`, the split one 0.89 and 0.93. Code that is the same
running at different speeds with the layout of the code and the heap points at an effect of addresses (a store-to-load false dependency between the tile buffers and the
table is a guess, not measured); it is left for the matrix tile path's rewrite, not a fix here.
The same-bits guards pass unchanged in Release (`test_amd_gcn_same_bits`, `_packed_`, `_memory_`, `_tier_`, `test_amd_codeobject_same_bits`, `test_amd_matrix_fast`,
`test_amd_mfma_math`, `test_amd_nan_order`, each also at `VGPU_SIMD=base`).

## A scratch access that wrapped its range check (branch `speedup7-scratch`)

`scratch_at` tested `offset + bytes > scratch_per_lane` with an unsigned offset. An instruction offset that is negative (`scratch_store_b128 v9,
v[1:4], off offset:-16`) with a register offset of zero is a number just under 2^64; adding the size wrapped the sum to zero and passed
the test, and the access landed before the work-item's scratch, on the heap (amdmemtext found it as a crash of the memory digest probe, which
skipped such shapes). The test is now `offset > per_lane || bytes > per_lane - offset`, and the access is refused with the error every other
access past the work-item's scratch gets. The other range checks were read and left: flat accesses to the private aperture take their offset
from an address that is at least the aperture's base (or 52 bits), the LDS ones from the shared aperture or a 32-bit field, so none of them wraps;
`scratch_buffer_access`, `scratch_load_lds` and gfx1250's signed sum go through `scratch_at`.

The golden digests changed only for the shapes this changes, recorded from origin/main plus the probe plus only this fix (the same arbiter as the merge's):
3 shapes of `same_bits_digests.txt` (gfx1100 `scratch_load_b128 v[5:8], v9, off offset:-16`, `scratch_store_b128 v9, v[1:4], off offset:-16`,
`scratch_store_b64 v1, v[2:3], off offset:-8`) and 28 of `mem_same_bits_digests.txt`: the 28 negative-offset scratch shapes of gfx1100 (7), gfx1201 (4) and
gfx1250 (17), which the probe used to skip (the skip is gone; their digest was 0). `test_amd_gcn_scratch_negative` runs 12 stores and 12 loads of each of
gfx942, gfx1030 (global accesses with the segment bits set to scratch), gfx1100 and gfx1250 at minus their size from zero registers (digest flavour 16, all zero)
and wants the refusal; with the check as it was, it fails (`gave []`).

## One PTX atomic read-modify-write (branch `speedup8-ci`)

`speedup` and origin/main (#402, round 10) each had made a PTX device atomic one table lookup instead of two: `MemoryManager::rmw_scalar` with
the interpreter's `rmw_global`, and `rmw_begin` / `rmw_commit` with the interpreter's inline fast path. Main's is the one kept (it covers the same
cases, declines in the same ones, and has its own mutation-checked test, `test_atom_rmw`); `rmw_scalar_fn`, `rmw_scalar` and `rmw_global` are gone, and
`test_memrmw` makes its one-lookup read-modify-writes with `rmw_begin` / `rmw_commit`. `atomic_rmw` (the lock-free compare-and-swap form the SASS and AMD
engines use) and `load_fields` stay.

## What PR #403's CI found (branch `speedup8-ci`)

* **`test_amd_gcn_same_bits` on the CUDA 12.4 job, 11 of 16,806 shapes** (`v_add_f16` / `v_mul_f16` and their sdwa, e64 and dpp forms): the job builds with GCC 12
  (this machine's is 13). Reproduced here with GCC 12.4's packages unpacked in a directory (`apt-get download` and `dpkg -x`, no root). A half add
  or multiply is done in float, and with a NaN in both operands the result keeps one operand's payload; GCC 12 made the first operand that one, GCC 13
  the second (the recorded digests are the second's). `half_add`, `half_sub` and `half_mul` spell the order out (`first_addf` / `first_mulf` in
  `gcn_exec.cpp`). All the digest tests pass with both compilers, and with `VGPU_SIMD=base` (the AMD engine's AVX2/FMA/F16C paths now obey it too, and the
  digest tests are registered a second time with it: `*_simd_base`).
* **The sanitizer jobs (Debug, no optimization)**: signed overflow in the integer reference code (`v_mad_i32_i24`, the WMMA/MFMA integer sums, which the
  hardware wraps) is now done unsigned (`wrap_madd`), with the same results. `test_amd_mfma_math` compared a NaN-carrying sum of the fixed operand
  order with the compiler's own order in the test; it compares a NaN as a NaN now. In an unoptimized build 25 to 82 of the 16,000 shapes of the digest tests have other NaN payloads
  (or, for `v_cos_f32` and a few atomics, other results; those were not traced further: the same shapes pass in every optimized build and with the tier off or on),
  so the three digest tests allow 2 % of the shapes to differ in a build without optimization, with a note, and none in an optimized one.
* **ThreadSanitizer**: no data race was reported anywhere. `test_amd_gcn_threads` died in `a_forked_child_launches_on_threads_of_its_own`: TSan does not
  support threads started in a forked child of a threaded process. That case is skipped under TSan.
* **Lint**: the `interp` job of `sass-ptx.yml` has the pr-guard step; `test_mma_fast.cpp` had a narrowing conversion; the whole tree builds with `-DVGPU_WERROR=ON`
  (a missing `default:` in `global_access`'s switch and a narrowing in the probe were ours).

## Sanitizer fixes (round 11) (branch `speedup-merge`, pushed to `speedup`)

PR #403's sanitizer jobs failed on its own AMD code. Reproduced on the rig (c7i.4xlarge, the CI image `cuda12.0-9f7f0f8400f7`, CUDA 12.0 and gcc 12) with the
CI recipe: `-DCMAKE_BUILD_TYPE=Debug -fsanitize=address,undefined` and `-fsanitize=thread -O1`. Both jobs' failures were in the AMD interpreter
and its tests. Three kinds of cause, none a race and none a timeout.

**1. Signed integer overflow (undefined behaviour; the three UBSan lines of the ASan job).** The hardware wraps; x86 does too, so the results were right, but
the compiler may assume they do not happen. Fixed with unsigned arithmetic, the same bits: `v_mad_i32_i24` (the product of two 24-bit signed factors and
the sum, 2 places), the integer multiply-accumulate of the matrix instructions (`mfma_dot_int`, the tile path's `int64` sum in `matrix_multiply` and the
sparse one, through `wrap_madd` in `amd/src/mfma_math.hpp`), and the 32-bit sum of `mfma_tile_int_scalar`. Test: `test_amd_mfma_math`
`integer_dot_products_wrap_without_overflowing_a_signed_type` (fails under UBSan with the old arithmetic). After the fix the ASan+UBSan run of the whole AMD set
(`ctest -R '^test_amd'`, 162 tests, 2 skipped by design) is green: 1762 s (`test_amd_gcn_tier_same_bits` 1762 s, `test_amd_matrix_fast` 1155 s,
`test_amd_gcn_same_bits` 999 s, `test_amd_gcn_atomics_gfx1250` 677 s, `test_amd_gcn_memory_same_bits` 666 s, the rest under 100 s) at `-j8` on 8 hyper-threads while
another session used the other 8.

**2. The NaN a float instruction leaves was the compiler's choice.** `x + y`, `x * y`, `std::fma(x, y, z)` and `std::fmin/fmax(x, y)` with two NaN operands give the
payload and sign of one of them (the hardware and SSE take the first operand's), and C++ does not say which operand is first. GCC swaps them
as it likes: differently at -O0, -O1 and -O3, differently in two copies of one loop, and `fminf`/`fmaxf` it passes to libm as `(y, x)` even at -O0 (checked in the
disassembly). The recorded digests of `test_amd_gcn_same_bits`, `_memory_same_bits` and `_packed_same_bits` held the choices of the Release build, so those tests were
right in the Release build and wrong in every other one: the same tests fail at plain `-O0` and plain `-O1` without any sanitizer (measured at the head `224874ea`: 50 and 22 instructions of
16806), under ASan 50 (after the overflow fixes), under TSan 82 in CI's run (that one was before `224874ea` fixed the f16 add). It is a real defect of the
interpreter (the result of an instruction depended on the optimization level) and not of the tests. Fixed at the cause: `amd/src/nan_order.hpp` has `nan_add`, `nan_sub`, `nan_mul`, `nan_fma`,
`nan_fmin`, `nan_fmax` with the operand order as a template argument; the result is the plain operation unless it is a NaN with a NaN operand, and
then the first NaN operand of that order, quieted. The orders are the ones that reproduce the recorded digests (found by trying them against the digests in the Release
build with the order of each site chosen at run time, then frozen as template arguments), so no golden digest moved; they are not the same at every site:
`v_add_f32`, `v_mul_f32`, `v_add_f64`, `v_mul_f64`, `v_pk_add_f32`, `v_pk_mul_f32` take the second operand first (GCC commuted them), `v_sub_f32`, `v_subrev_f32`, `v_mul_legacy_f32`
the first; `fma` takes (b, a, c) in `v_fma_f32`, `v_fmac_f32`, `v_fmaak/fmamk_f32`, `v_fma_mix*`, `v_fma_f64`, `v_fmac_f16`, and (a, b, c) in
`v_fma_f16`, `v_pk_fma_f32`, `v_pk_fmac_f16`; `v_max3_f32` and the max in `v_med3_f32` take the library's answer for `(y, x)`; the packed adds of the atomics are `y`-first in both halves for float16 and `x`-first in the low half and `y`-first in the high half for bfloat16, `ds_pk_add_f16` is `x`-first in the low half and
`y`-first in the high half (all found by trying). `nan_fmin`/`nan_fmax` are glibc 2.39's fmin and fmax written out (a tie of the two zeros gives the
second argument, two NaNs the first, quieted, a signaling NaN beside a number that NaN quieted, a quiet one beside a number the number), checked against the library on 9 x 9 values of
both widths. Two more that are not an order: `v_sin_f32`/`v_cos_f32` on gfx942, gfx950 and gfx1250 and `v_fract_f16` pass a NaN through as it came (the float to double to float round trip that would quiet a
signaling NaN is folded away by the optimizer at -O1 and above and not at -O0, and the recorded bits are the folded one's). Test: `test_amd_nan_order` (4 tests: the
helpers are the plain operation for numbers, the order named for NaNs, the library's fmin/fmax where its answer does not depend on the argument order), and the same-bits tests themselves,
which now pass at -O0 (also with ASan+UBSan), -O1, -O3 and with TSan. Not covered by any digest, so the order chosen is a guess that matches the natural one (first operand first): the bfloat16
`ds_pk_add`, `v_fmac_f64`, the scalar float instructions, `v_fmaak/fmamk_f64`, and `v_pk_fma_bf16`; they pass today and are not changed.

**3. Two test defects.** `test_amd_mfma_math` compared `mfma_dot` with a plain `old_sum += a * b` loop; with two NaNs the plain loop's payload is the compiler's, and
`mfma_dot` has the fixed order (that was the TSan failure at -O1 and the ASan one at -O0): the NaN case is now compared with the same sum in the fixed order,
spelled out on the strided layout. `test_amd_gcn_threads` `a_forked_child_launches_on_threads_of_its_own` (and the two `_gfx1100`/`_gfx1030` runs) died under TSan with
"starting new threads after multi-threaded fork is not supported": that is what the test checks on purpose (the pool is rebuilt in the child), so the test binary sets
`die_after_fork=0` through `__tsan_default_options` when built with TSan. Not a finding.

**Merge with `origin/speedup` (e50e35c5).** The other session had meanwhile made the same overflow fixes (`wrap_madd`, kept; `mfma_mac_wrap` is gone), made `test_amd_mfma_math`
compare a NaN as a NaN (mine compares the fixed order exactly, kept), skipped the forked-child case of `test_amd_gcn_threads` under TSan (it runs again, with `die_after_fork=0`), and let the
three digest tests differ in up to 2 % of the shapes in a build without optimization or under a sanitizer. With the order of the NaNs spelled out the digests hold in every build, so that
tolerance is removed again (the three tests are as they were at `224874ea`): an instruction that differs in any build is a failure. The tier test's one seed under TSan is kept.

**Slow, not failing.** Under a sanitizer these tests take long: `test_amd_gcn_tier_same_bits` 1762 s (ASan), 2373 s (TSan), `test_amd_matrix_fast` 1155 s and 1527 s,
`test_amd_gcn_same_bits` 999 s and 1398 s, `test_amd_gcn_memory_same_bits` 666 s and 897 s, `test_amd_gcn_atomics_gfx1250` 677 s and 463 s (loaded rig, `-j8`; CI's own times
were 79 s to fail and 900 s for the tier test under TSan). CMake sets no `TIMEOUT` for them and ctest's default here is none, so none timed out; the CI job's own time limit
is what they eat: the proposal (not made) is a `TIMEOUT` property of 3600 on these five, and moving `test_amd_gcn_tier_same_bits` and `test_amd_matrix_fast` out of the TSan job's list
(they run no threads of their own; `-E` in `ci.yml` line 739) if the job's limit is tight.

## The allocation table and a launch's bookkeeping (branch `sn-memalloc`, from `speedup-next`)

`src/core/memory.cpp`, `include/vgpu/memory.hpp`, `include/vgpu/brlock.hpp`, `src/core/memory_backing.cpp` and the kernarg write of
`run_launch` (`amd/src/hip_api.cpp`). Both vendors' memory accesses and the AMD launch go through them. Every address, quarantine
entry, diagnostic, count and result is as it was: `test_memalloc` holds the answers to those of the code before this change (digest
recorded from it), and the guards of the speed work pass in every build.

### What was measured first

The leads of the previous round held. `tools/bench-amd/profile` (SIGPROF) on a `-g` build of the base: `buf` 27% in
`MemoryManager::scalar_location` (4% in `__tls_get_addr`: the lookup memory is a `thread_local` of a shared library, so every access called
for its address), `MemoryManager::free` 15% and `alloc` 6% of `launch`, `materialize` 14% (a 64 KiB chunk made and zeroed for a 192-byte
kernarg segment), `find_ordinary_miss` and `BigReaderLock::lock` visible. The NVIDIA `launch` workload makes no allocation and spends none
of its time here (the 13% of the earlier note was the AMD launch); NVIDIA's memory workloads reach the table through `find_ordinary`.
A loop of `alloc(192)`, `write`, four loads, a load and a store of a long-lived allocation and `free` (one thread, pinned, minimum of 9)
took 560 ns on the base.

### What changed

* **An allocation does not void what threads remember, and a free voids only what it frees.** Each thread keeps the last eight
  allocations it found, with the table lock's epoch; every `alloc` and `free` took the lock with two epoch changes, so every kernel's
  first accesses after a launch's kernarg `free` went back to the table. Now the lock has a quiet hold (`lock_quiet`, `ExclusiveGuard(l,
  true)`) that leaves the epoch alone, used by `alloc` (a new entry changes no other) and by `free`, which calls `note_free(base)` first.
  The epoch's low 16 bits count frees since its random part was drawn, and a 64-entry log of the bases freed holds the last ones:
  `still_valid(seen, base, &now)` answers whether an entry remembered at `seen` survived the frees since (none of them its own, the
  log not overrun, the random part unchanged), and `find_ordinary_miss` renews such an entry in place of walking the table. Every other
  mutator still takes the full lock. A nested full hold inside a quiet one voids at once and again at its end (`bumped_`).
* **The writer looks at the slots in use.** The lock keeps a mask of the slots some thread has counted in (set the first time, before
  the count, so the Dekker argument of the flag and the counts holds), and a writer waits for those and not for all 64 cache lines.
* **The lock's thread identity and the lookup memory are found from the thread pointer, not through a thread-local.** (The first
  version kept them in initial-exec thread-locals; the core is linked into every shim library, so each library took static TLS, and
  glibc's roughly 1.6 KB surplus ran out in a process that loads several of them: `dlopen` of libamdhip64 or the profiler failed with
  'cannot allocate memory in static TLS block'. `lint_static_tls` now holds every shim to no static TLS.) The lock reads
  `__builtin_thread_pointer()` for identity and hashes it into a slot; the memory of lookups sits in a 1024-slot table keyed by the thread
  pointer (8 probes, claimed by compare-and-swap on a thread's first lookup, given back by a destructor when it ends); a thread that finds
  no free slot, or is ending, reads an empty memory that is never written. The price against the initial-exec version is one dependent
  load per access: about 3-6% on `buf`, `narrow` and `atomics`; the gains of the allocation work on `launch` are untouched.
* **`load_scalar` / `store_scalar` try one inline lookup first** (`hot_location`): the thread's last allocation, no host mapping in
  reach, a chunk that is real. Anything else (another allocation, an untouched or uniform chunk, a fault armed, a miss) takes the
  general path, which is unchanged. `run_allocation` (`load_run`, `store_run`, which the AMD engine uses for contiguous lanes) uses the
  lookup memory too, where it took the lock and walked the map on every call.
* **Nodes and chunks come back.** `free` keeps the node of `live_` (with the allocation's array of chunk slots) for the next `alloc`,
  `freed_` keeps the node of the entry the quarantine evicted for the next free, the eviction order is a deque of iterators (it was a
  map popped only at its front), and inserts take the end as their hint (addresses are a bump). A chunk of an allocation that was
  never written past its end goes back to `backing` with `release(chunk, dirty_bytes)`: below 4 KiB dirty it is cleared there and kept
  (at most 64) and the next `acquire()` hands it out, counted as in use again, without a 64 KiB zero. Every write is bounds-checked
  against the allocation's size, and a chunk made whole from a uniform chunk is a whole chunk of the allocation, so for the last
  chunk the bytes past `size` are still zero.
* **The kernarg segment is written, not copied through a padded vector.** A new allocation reads zero, so `run_launch` writes the
  arguments only.

### Results

Alternating runs, 11 per cell, one host thread pinned to CPUs 12-13 (`--threads all`: CPUs 8-15), `flock /tmp/speedup-bench.lock`,
load 3 to 5 on server2 (32 threads, shared). Ratio is B/A of the medians (base `90bf40da`, this branch), p is Mann-Whitney, counters
identical in every pair (the NVIDIA `atomics` counters at 8 threads differ between two runs of the base too: a spin loop).

| workload | MI300X, 1 thread | RX 6800, 1 thread | MI300X, all | RX 6800, all |
| --- | ---: | ---: | ---: | ---: |
| `launch` | 1.319x (p 0.000) | 1.116x (0.001) | 1.303x (0.000) | 1.212x (0.000) |
| `launch_sync` | 1.474x | 1.151x | | |
| `graph` | 1.228x | 1.167x | | |
| `narrow` | 1.246x | 1.190x | | |
| `atomics` | 1.167x | 1.145x | | |
| `buf` | 1.139x | 1.134x | 1.148x | 1.120x |
| `mem` | 1.056x | 1.036x | 1.059x | 1.045x |
| `stream` | 1.028x | 1.033x | 1.055x | 1.030x |
| `api` | 1.025x | 1.029x (same) | | |
| `flat` | 1.019x (same) | 1.019x (same) | 1.026x | 1.025x |
| `smem`, `ldsx`, `launch_mid`, `launch_wide`, `apicheap`, `memcpy` | same (0.98x to 1.01x) | same | same | same |

NVIDIA (RTX 3060 profile, SASS and PTX, 11 runs): `atomics` 1.053x SASS and 1.063x PTX (p 0.000 and 0.002), `mem` 1.018x / 1.019x at
one thread (p 0.000, under the 2% bar: "same") and 1.028x at 8 threads (faster), `texture`, `launch`, `shared` same. `local` SASS reads
0.975x (p 0.000) at one thread and 0.968x at 8: it does not use the memory manager (`h_local` works in the warp's own vector), and
the same loss appears in builds of the base to which only dead functions were added in front of `load_scalar` (90 and 160 lines of
volatile arithmetic in `memory.cpp`: 0.975x each), and not with 12 or 40 lines, with 24, 200 or 1000 bytes allocated at start-up, or in
the commit before `hot_location` (all 1.00x). That is a code-placement effect (not located further), so it is not dropped for a cause
in this change. `local` on PTX: same.

`alloc`+`write`+loads+`free` of the loop above: 560 ns to 179 ns (3.1x, min of 9, one thread); a load and a store in one allocation
8.1 ns to 5.5 ns.

### Tests (wired into CMake)

`test_memalloc` (13 tests; also run under ASan+UBSan and TSan): a model of the bump allocator with a bounded quarantine
(addresses, `OutOfMemory`, `DoubleFree`, `InvalidFree`, `InvalidPointer`, `UseAfterFree` and `OutOfBounds` for 33,000 random operations
on four configurations, `used()` and `live_allocations()` after every one); the quarantine's exact bound (4096 entries: the 4097th
oldest is "retired", the rest are double frees); the digest of 12,000 to 40,000 random operations (values, addresses, error codes and
texts, `used()`, `resident_bytes()`; four configurations including OOM and the quarantine's bound) equal to the one the code before this
change gives (recorded from `90bf40da` with `-DVGPU_MEMALLOC_DIGESTS_ONLY`); a freed allocation read through the thread's memory after
0 to 140,000 other frees (the count wraps at 65,536) is a use-after-free and its neighbours stay right; six remembered allocations
freed in random orders; four readers beside two threads that allocate and free; a new allocation reads zero, in its slack too,
after the last one held scalar writes, a copy, a uniform fill and a write over a uniform chunk (3 page sizes, sizes from 1 byte to 200,000);
the backing's counts and zeroing for chunks given back nearly clean; `resident_bytes()` through 300 allocate-write-free cycles; the lock:
a quiet hold leaves the epoch, a full one changes it (nested both ways), `note_free`/`still_valid` (other bases survive, the freed one and
more than 64 frees and a full hold and the count's wrap do not), and exclusion of 80 reader threads by quiet and full writers.

Mutations (scratch copies, each run on a clean rebuild), what fails: no `note_free` in `free`: the digest, the stale-read test, the
frees test; `still_valid` ignoring the log: the same and the epoch test; recycled node keeping its chunk pointers: aborts (double
free); pool clearing one byte short or not at all, the last chunk's dirty bound one byte short: the digest, the zero test, the pool test;
quarantine bound off by one: the quarantine test; the writer ignoring the slots in use or readers not announcing their slot:
segmentation faults in the first test with readers beside a writer (the exclusion tests would fail); a full hold nested in a quiet one
not voiding: the epoch test. Not killed, by construction: the wrap handling of the count (an unhandled carry only changes the random part,
which `still_valid` treats as a full void) and not reusing the quarantine's node (speed only); the `seq` comparison on eviction guards a
pointer freed twice, which the bump allocator never produces.

Also passed on this tree: `test_memory`, `test_memfast`, `test_memrmw`, `test_memrmw_lockfree`, `test_worker_pool`, the guards
(`test_amd_gcn_same_bits`, `_memory_`, `_tier_`, `_packed_`, `test_amd_codeobject_same_bits`, `test_amd_matrix_fast`, `test_amd_mfma_math`,
`test_amd_nan_order`, `test_amd_packed_math`, `test_amd_gcn_threads`, `test_amd_hip_queue`) and `e2e_sass_tier` in Release, ASan+UBSan (-O0) and
TSan (-O1) (the sanitizer runs build without `VGPU_WERROR`: GCC 13 rejects `atomic_thread_fence` under `-fsanitize=thread -Werror`), every
`test_*` and the AMD end-to-end set and `e2e_sass_*` in Release (267 of 267), `bench-amd --check-counters` (36 pairs match). The NVIDIA
`bench --check-counters` skips on this host (its nvcc builds different device code than the recorded counters); `--ab` reports identical counters.

### Not done, and negative results

* **`local` on SASS reads 2.5% slower** in these builds with nothing of it in this change (above): a code-placement effect.
* **`load_scalar` of an untouched or uniform chunk, a miss and the fault-armed case** take the general path; the hot path is the real
  chunk in the thread's last allocation, which is what the lanes of one instruction do.
* **A flat sorted array for the quarantine** (`freed_`): inserts at the end and evictions at the front would be free, but the access
  pattern is the program's and a free in the middle moves up to 64 KiB; the map with recycled nodes keeps the worst case.
* **Sizing the chunk to the allocation** (a 192-byte allocation holding 192 bytes): `kChunkSize` is assumed by every read and by
  `resident_bytes()`, `ram_bytes()` and the spill file; recycling the 64 KiB chunk keeps them.
* **A dirty-extent flag for chunks made from uniform ones** (`wide_dirty`) was written and dropped: a uniform chunk exists only as a
  whole chunk of an allocation (`fill`), and the last chunk of an allocation that is a whole chunk has its whole size as dirty bound, so
  no input reaches it (the mutation that removes it cannot fail a test).
* **`SIGPROF_STACK=1` on the AMD `launch` workload** hung twice in five runs of one `-g` build (all three threads in a futex wait) and
  never in 350 runs without the sampler (150 `launch`, 100 `launch_mid` and 100 `launch_wide` at 8 threads) or in 16 more with it:
  the sampler's unwinder takes the loader's lock inside a signal handler. Not a finding of the memory manager; noted so that the
  profile is read with that in mind.
## AMD speed gaps closed (branch `sn-amdgaps`)

**Matrix tile path.** The f8f6f4 forms (`v_mfma_f32_16x16x128_f8f6f4`, `v_mfma_f32_32x32x64_f8f6f4`) and their block-scaled twins, BLGP's lane permutation
of B, the doubles' BLGP negations and the four-block `v_mfma_f64_4x4x4_4b_f64` now take the tile path (`matrix_multiply_fast`, `matrix_small_k`, scale
tables by block of 32 K and row). `mfma_scale_mul` (mfma_math.hpp) fixes the order of the two operands of the scale multiply, used by both paths. A bug of
the reference path found on the way: its binary16 shortcut ignored BLGP's permutation of B. New bench-amd workloads `mfma_f8f6f4`, `mfma_scale` (target
`mi350x`, gfx950) and `mfma_f64_4b`. A/B (9 runs, one thread, CPUs 24-31, server2 load 3 to 25): mfma_f8f6f4 7.7x, mfma_scale 8.0x, mfma_f64_4b 2.7x, p < 0.001,
counters identical. 8-bit float operands decode each register word once (mfma_fp8 1.15x of the base).

**Tier coverage** (`VGPU_AMD_TIER_STATS=1`). Over the 46 pantheon workloads (run-pantheon-workloads.sh) the interpreter ran 7.1% of the instructions (574 M of
8.07 G: `v_xor_b32_sdwa`, `s_load_*`, `v_cndmask_b32_e64`, `s_mul_hi_u32`, `s_endpgm`, `v_pk_fma_f16`, 29 M NaN handbacks of `v_fma_f32`); rocBLAS's test
program 12.7%. Added: SDWA integer forms, VOP3 forms of existing handlers, `v_cndmask_b32_e64`, scalar ALU leftovers (`s_mul_hi`, `s_brev`, `s_cmpk_*`,
`s_bitcmp*`, `s_getpc`), `s_load_*`/`s_endpgm` through the Machine without step()'s frame, one-input float functions (single and double), 16-bit integer and
half forms on the CDNA parts, class and 16-bit compares, `v_pk_fma/add/mul/min/max_f16` (routed to packed_math.hpp), `v_pk_mov_b32`, `v_div_scale/fmas/fixup`,
`v_fma_mix*`, `v_accvgpr_read/write`. After: 0.5% of the pantheon instructions (MFMA, atomics, a few 16-bit forms), 7.1% of rocBLAS's. A/B on bench-amd:
smem 1.14x, valu_int 1.03x, atomics 1.04x, others unchanged.

**NaN-aware float path.** A NaN result of `v_add/sub/mul/fma_f32`, the double forms and `v_pk_*_f32` is worked out again in the tier by the interpreter's rule
(`nan_add<1>`, `nan_sub<0>`, `nan_mul<1>`, `nan_fma<2>`, `nan_fma<0>` for the packed fma); one-input functions of the host's libm pass their NaN through. The 29 M
handbacks came from the pantheon virus workloads (random data). Min/max, nearbyint, angles in turns and `v_div_fmas` still hand a NaN back. The tier test gained
flavour 17 (halves with specials and signaling NaNs) and generated corpora `isa_corpus_tier_<gfx942|gfx90a|gfx950>.txt` (amd/tools/tier-corpus.py; tier on against
tier off, no recorded digests); they caught bugs of this branch's own (`v_fract_f32` NaN, clamp on cvt_f32_* off the guide parts, a negated signaling half NaN in class,
`v_div_fmas_f64` under -O1).

**Packed min/max.** `v_pk_min/max_f16` run whole-wave (`packed_minmax_lanes`; `pk_fminf`/`pk_fmaxf` and the guide parts' bit rule now live in packed_math.hpp;
unit test against the per-lane code). The dot2 NaN lanes are unchanged.

### Not done, and negative results (this branch)

* Trampolines for the vector memory instructions the tier has no handler for (buffer, flat, scratch, LDS, atomics) through the Machine's functions without
  step()'s frame: no measurable gain on mem, flat, buf, ldsx (p > 0.05); dropped.
* NaN lanes of `v_min/max_f32` (MODE.IEEE decides) and the dot2 NaN lanes stay with the interpreter.

## The API calls' own cost (branch `sp-apipath`, from `d8ec6128`)

What a program pays for a runtime call that is not simulation: `cudaMalloc`, `cudaMemcpy`, `cudaEventRecord`, `hipMemsetAsync`, `hipFree`.
The harness rows are `api` (NVIDIA: 15 calls a round; AMD: 13 calls a round, 5 of them hand-offs to a stream's thread), `apicheap`,
`memcpy`, `stream`, `launch_sync`, `graph`. The numbers below come from the pinned benchmark CPU (`--cpus 6`), so on the AMD side the
stream threads share that one CPU and every hand-off is two context switches (`VGPU_QUEUE_SPIN` is 0 there); wall-clock gains on the AMD rows
are therefore much larger than the cycle counts, which are given beside them.

### Where a call's time went (perf, one host thread, 4 KiB buffers)

* NVIDIA `api`: 19% was the zeroing of a fresh 64 KiB chunk (`backing::acquire`, libc `rep stos`) for each `cudaMalloc` of 4 to 6 KiB: the
  pool of chunks given back nearly clean took back only 4 KiB, and these buffers dirty more. Then `ApiCall::ApiCall` 5%, `__tls_get_addr` 3 to 5%
  (nine thread-locals of the profiling core, each its own lookup), the clock 4%, `pthread_mutex_*` 5%, the telemetry writers 4%, and a per-call copy
  of the arguments into a `KeptArgs` nobody reads. A call is `traced_call`'s `ApiCall` and `guard`'s, nested.
* AMD `api`: 70 mutex acquisitions a round (13 calls), ~15 futex calls, 6.4 context switches. `hipFree` alone took 14 locks (every queue
  asked `tail()`, `wait()` and `take_error()` under its mutex, though idle).

### What changed

* **Profiling core** (`include/vgpu/profiling.hpp`, `src/core/profiling.cpp`). The thread-locals are one struct (`ThreadApi`), constant-initialized
  (otherwise every access asks whether the thread's `__tls_init` has run) and found once per call (`ApiCall` keeps the pointer: its destructor
  and `silenced()` do not look it up again). `ApiCall`'s work for a call somebody listens to, or that has arguments noted, is out of line
  (`begin`, `end`); for the others the constructor is a dozen instructions. `enabled()` and `hooked()` are inline loads of exported flags
  (`detail::g_on`, `g_hooked`; the CUPTI shim still shares the runtime's, as it shared its functions). `note_context_made()` reads before it exchanges.
* **Runtime and driver** (`nvidia/src/runtime_api.cpp`, `driver_api.cpp`). The arguments of a call are copied into `KeptArgs` only when a recorder
  or subscriber is on (an `optional` in the frame). `capture_unsafe_gate()` (every allocation, copy and launch) reads an atomic that mirrors
  `g_captures` before it takes `g_graph_mu`; the five places that change `g_captures` update it.
* **Telemetry** (`src/core/telemetry.cpp`). The `note_*` calls refresh under the lock they hold (`refresh_locked`), so the writers' mutex is a plain
  one; the segment's sequence number is stepped by a load and a store under that mutex (fence and release store for the ordering) instead of two locked increments.
* **Memory backing** (`src/core/memory_backing.cpp`). The pool of chunks given back keeps those with up to 32 KiB written (`kPoolMaxDirty`, was 4 KiB):
  cleared under the lock to the byte promised, as before. `pooled_chunks()` for the test.
* **HIP queue** (`amd/src/hip_queue.hpp`). `idle()`, `tail()` and `take_error()` answer from atomics without the queue's mutex (`idle` reads the
  completed count before the submitted one; a failure is noted before its item counts as done); `wait()` of a finished item returns without the
  lock. `run_inline` takes any callable and can tell the item's number.
* **HIP runtime** (`amd/src/hip_api.cpp`). `hipEventRecord` on a stream with nothing in it (and every marker it follows done) reads the clock on
  the caller (a profiler attached keeps the stream's thread); the result of a waited-for item stays in `in_order`'s frame (a `shared_ptr` and a second
  function object less); `hipFree` takes the lock once to find the owner and the queues; the three synchronizations take the lock for
  `release_to_thresholds` only once a pool exists (`g_pools_made`). **A copy or fill of at most 256 KiB (`kInlineMemoryBytes`) queued on a stream with
  nothing in it, nothing it follows unfinished and no profiler attached, is done in the call** (`in_order`): `hipMemsetAsync`, `hipMemcpyAsync`
  and the device-to-device copy and fill that HIP queues and returns from at once. A failure is the stream's, kept for the next synchronization, as
  for a queued item. Kernels, host functions, events behind work and larger copies are queued exactly as before. This moves *when* such work happens
  (at the call, never later) and starts no stream thread for a stream used only this way; a program cannot tell except by timing or by a race.

### Results

15 alternating runs, `--cpus 6`, load 12 to 55 on the server (the A/A column is the base against a copy of itself: ratios of medians, all p > 0.1).

| workload | MI300X 1 thread | MI300X all | RX 6800 1 | RX 6800 all | A/A |
| --- | ---: | ---: | ---: | ---: | --- |
| `api` | **14.5x** | **14.5x** | **14.7x** | **14.7x** | 0.96 to 1.06 |
| `memcpy` | **4.96x** | **5.10x** | **5.22x** | **5.13x** | 0.97 to 1.05 |
| `apicheap`, `stream`, `launch_sync`, `graph` | 0.99 to 1.00 | 0.99 to 1.01 | 0.99 to 1.01 | 1.00 to 1.01 | 0.97 to 1.03 |

| NVIDIA (RTX 3060 profile) | SASS | PTX | A/A |
| --- | ---: | ---: | --- |
| `api` | **1.48x** (p < 0.001) | **1.48x** (p < 0.001) | 1.02 / 1.00 |
| `launch` | 1.03x (p 0.14) | 1.03x (p 0.004) | 1.00 |
| `startup` | 1.00x | 1.00x | 1.01 |

Counters are identical in every pair. The same builds under `perf stat` (user instructions, cycles, context switches of the process; 3 runs, spread under 1%):

| | instructions | cycles | context switches |
| --- | ---: | ---: | ---: |
| NVIDIA `api`, base | 1187 M | 615 M | 5 |
| NVIDIA `api`, now | 959 M | 414 M | 5 |
| AMD RX 6800 `api`, base | 2.43 G | 1.6 to 2.0 G | 640 k |
| AMD RX 6800 `api`, now | 1.52 G | 0.55 G | 6 |
| AMD RX 6800 `memcpy`, base | 0.70 G | 1.29 G | 197 k |
| AMD RX 6800 `memcpy`, now | 0.47 G | 0.75 G | 3 |

Where it comes from, as the changes were added (NVIDIA `api`, wall clock, 15 runs against the base): the call-path, capture-gate, telemetry-mutex and chunk
pool changes together 1.34x; the thread-local struct, `ApiCall`'s fast path and the sequence number then took the cycles from 456 M to 414 M (1.48x). The
profile before any of it put 19% of the cycles in the chunk zeroing. AMD `api`: the lock-free queue methods and the inline event record alone left the process
switching 600 k times (wall clock 1.1x); the inline small copies and fills took the switches to 6.

### Tests (wired into CMake)

`test_profiling_api` (the recorder and subscriber contracts of `ApiCall`: outermost call, nesting, correlation, whose arguments are whose, silence, threads,
the context hook once; the flags), `test_runtime_capture_gate` (a global capture refuses an allocation on any thread and the gate opens after it; relaxed
does not refuse), `test_telemetry` (two steps of the sequence number for each update, even between), `test_memalloc` (chunks with up to 32 KiB dirty are
kept, cleared and counted; the limit is pinned by a `static_assert`), `test_amd_hip_queue` (`idle`/`tail`/`take_error`, a failure seen the moment its item is
seen done, `run_inline` with a lambda and the item's number), `test_amd_hip_event_inline` (a record on an idle stream is complete at the call, also with the
host refusing threads; behind a host function, and on a blocking stream behind the null stream's work, it is not), `test_amd_hip_async_inline` (a small fill
and copy on an idle stream need no thread and are done; behind a running host function they are not; a megabyte is queued), `test_amd_hip_pool_sync` (each
of the three synchronizations gives a pool's excess back, none made before), `test_amd_hip_thread_refused` (now with a host function as the first item).

Mutations (a scratch copy, each rebuilt and run; what fails): every call outermost; noted argument count not cleared; the fast path ignoring noted
arguments; `set_enabled` writing the other flag: `test_profiling_api`. Capture flag never set: `test_runtime_capture_gate`. `begin_update` storing nothing:
`test_telemetry`; `note_memory` calling the locking `refresh` under the now plain mutex: `test_telemetry` (deadlock, killed by timeout). Pool limit 4 KiB:
`test_memalloc` (the `static_assert`); the pool clearing one byte short: `test_memalloc`. `take_error` always ok, `idle()` always true (timeout),
`run_inline` not telling the number, a failure noted after its item counts as done: `test_amd_hip_queue`. Event record never inline; ignoring what the stream
follows: `test_amd_hip_event_inline`. Async copies never inline; no size limit; everything inline, host functions too: `test_amd_hip_async_inline`
(and `test_amd_hip_thread_refused` for the last). Synchronizations never giving a pool back; the flag never set: `test_amd_hip_pool_sync`.

### Not done, and negative results

* **TLS descriptors** (`-mtls-dialect=gnu2`, which replaces the `__tls_get_addr` call with a descriptor call): `api` 1.034x SASS and 1.055x PTX
  (p 0.003 and < 0.001) on the tree before the thread-local work above, `launch` same. Not kept: it needs the loader of the machine that runs the shims to support
  TLS descriptors (glibc's x86-64 support is not something to assume of every distribution the shims are installed on), for a few percent.
* **One `ApiCall` for a call that `traced_call` and `guard` both open**: the inner one is a nested no-op, but dropping it needs every `guard` caller
  audited for a public entry that opens none of its own.
* **A waiter running the queued work itself** (instead of sleeping while the stream's thread does): the wake-up at submission (a futex call and, on one CPU,
  a switch to the woken thread) stays, so the gain would be the second switch only, and the work would run on another thread than the stream's.
* **Stream threads kept for reuse**: `stream` creates and destroys a stream a round, so starts a thread a round (8000 `clone3` calls a run); a parked
  thread would carry thread-locals from one stream to the next (the current device, the last error).
* **`hipEventSynchronize` and `hipEventQuery`** each take three event/runtime locks to look at one event (`in_capture`, `remote_event`, `event_marker`); one
  look under both would save two lock pairs of about 40 a round: not done.
* The AMD `launch_sync`, `stream` and `graph` rows are the kernel launch (`set_up_group` is 55% of `launch_sync`), not the call.

## The AMD engine: whole-wave memory accesses (branch `sp-amdmem`, the `speedup` round)

The memory instructions of the AMD engine went to the memory manager a lane at a time: every lane of a gather, a buffer access or an atomic looked up its allocation
(`lookups()` and the table behind it), asked `HostMaps::may_contain`, and made its own `load_scalar` / `store_scalar` (an atomic made two, one for the load and one
for the store). The profile of `buf` (RX 6800) was `load_scalar` 25%, `store_scalar` 18%, `buffer_access` 20%, `load_words` 7%, `may_contain` 4%; of `atomics`,
`global_access` 20%, `load_scalar` 18%, `atomic_rmw` 6%, `store_scalar` 4%, and with a single lookup for the load and the store `Machine::atomic_rmw` alone was 47% of the
MI300X run, nearly all of it the chain of dependent loads that finds the allocation (`lookups()`, the thread's last way, the chunk table). Now a wave's accesses go to the memory manager together wherever the manager can take them as the calls
they stand for, and decline (having done nothing) wherever it cannot, so the lanes then go one at a time and meet any error where they would.

Measured on server3 (Ryzen 7 PRO 8840U, one SMT sibling of one core, the machine at a 1-minute load of 30 to 130 from other builds), Release, GCC 13.3, simulated
MI300X and RX 6800, one host thread, `tools/bench-amd` A/B against `d8ec6128`, 11 alternating runs; ratios are B/A of the median (the best-of ratio agrees within
3% in every row). Every row's counters are identical, and `bench-amd --check-counters` matches (37 pairs). The A/A control of the same harness (base against the
same base, 9 runs) gave medians between 0.964 and 1.044 (`memcpy` mi300x 0.964, `ldsx` rx6800 1.044, the rest within 2%).

| workload | what it runs | mi300x | rx6800 |
| --- | --- | ---: | ---: |
| `buf` | buffer dword loads and stores, some past the end | **2.74x** | **2.37x** |
| `narrow` | byte and short global loads and stores | **1.78x** | **1.70x** |
| `mem` | 16-byte copy, scalar scale, gather | **1.58x** | **1.36x** |
| `lds` | tree reduction and tile transpose (global rows, LDS words) | **1.45x** | **1.27x** |
| `flat` | generic pointers that reach global memory | **1.33x** | **1.16x** |
| `atomics` | global atomic add, max, CAS loop | **1.23x** | **1.24x** |
| `ldsx` | wide LDS accesses and LDS atomics | 1.13x | 1.05x (p 0.065, same) |
| `smem`, `memcpy`, `stream` | not changed | 1.00, 1.00, 1.00 | 0.99, 0.99, 1.00 |
| `valu_int`, `launch` (controls) | not changed | 0.99, 1.01 | 0.99, 0.99 |

p < 0.001 for every "faster" row. Eight host threads were not measured (the benchmark is pinned to one CPU on this host); the threaded atomics path is held by the
tests below, in Release and under ThreadSanitizer.

NVIDIA: the SASS and PTX engines share `batch_ok` / `load_batch` / `store_batch`, which got the one-pass test below. `tools/bench` A/B, 9 runs, counters
identical in every row: `mem` ptx 1.04x (p 0.004), the rest of mem, shared, atomics, texture, local, int_alu, divergent and warp within 2.5% of the base,
`atomics` ptx 0.97x on a 85 ms workload whose spread is 11 to 17% (p 0.006, best-of 0.985; not beyond the harness's 2% on both).

### What changed

* **`MemoryManager::batch_ok`** (`src/core/memory.cpp`) makes one pass over the addresses for the three tests every access must pass (alignment by the OR of the
  addresses, inside the first access's allocation by the largest offset, and no host mapping in reach by `HostMaps::may_overlap(lowest, highest)`, which rules a
  range wholly inside the device windows out as one address, and otherwise asks `may_contain` of each as before). `load_batch` takes the width as a constant.
* **A wave's gather or scatter** (`Machine::wave_scalars`, called from `tier_global` in `gcn_tier_machine.inc` and from `global_access`): the active lanes that
  make no run (or the lanes after the run that ended) are one `load_batch` / `store_batch` of bytes, shorts, words or pairs; lanes of a load after a lane with the
  same address take its value (the load memo's rule).
* **Buffer dword loads and stores** (`Machine::buffer_words`): the wave's addresses and whether each is inside the resource, then the lanes inside as one batch; the
  lanes outside read zero or write nothing, as before. `fits` moved out of the lane loop (it reads `index`, now declared before it).
* **Contiguous runs** (`Machine::memory_run`): the run is found from the address registers' rows (the lane limit from EXEC at once, a 64-bit compare per lane),
  and a run of one word a lane in a vector register is moved straight to or from the register's row (`load_run` into the row, `store_run` from it); two and four
  words a lane are transposed row by row.
* **Atomics** (`MemoryManager::rmw_batch`, `Machine::wave_atomics`): a wave's atomics are one call, lane by lane in order, with one lookup for the lot: the plain
  load and store-if-changed on one thread, the compare-and-swap of `atomic_rmw` (one `atomic_rmw_slot` per lane, the loop that was `atomic_rmw_fn`'s) across
  threads. The old values go to the destination register after the call, or, for the lanes made before a store that could not get host memory, before the error
  leaves. A single atomic (`Machine::atomic_rmw`, the fallback and the scalar and buffer forms) uses `MemoryManager::rmw_if_changed` on one thread: the load, the
  operation and the store of a changed result at the word `hot_location` finds, without a second lookup. The data and the old value are read and written in the
  register rows.
* **`h_ds`** (the tier's LDS handler): the bounds test of the lanes is branch-free and `ds_read_b32` / `ds_write_b32` have loops of their own (the other widths
  keep the general one).

### Tests (all wired in CMake; each shown to fail under the mutation next to it, built and run on server3)

| test | what it holds | mutations that fail it |
| --- | --- | --- |
| `test_amd_gcn_memory_same_bits`, new `the_wave_wide_memory_paths_give_the_bits_lane_by_lane_gives` | every memory shape of the eight ISA corpora (DS, global, flat, buffer, scalar) in the twelve address modes, with half the lanes off in some: digest with the whole-wave paths on against off (`amd::set_fast_paths`), with no recording to trust | batch stores in reverse lane order (181 of 16068 instructions differ); narrow batch loads not sign-extended (114); buffer lanes outside the resource keep their register (28); atomics do not give the old value back (77); each also fails the recorded-digests test |
| `test_amd_gcn_memruns` (existing) | runs of lanes against lane by lane, 18 address patterns and 7 masks, 7 kernels, 5 targets | the run reaches across lanes that are off (`copy32s`, mask 1); a load run one lane late (every pattern) |
| `test_memrmw`, new `a_wave_of_atomics_as_a_batch_matches_the_same_accesses_one_at_a_time`, `..._declines_with_faults_armed_or_stuck_cells`, `a_single_thread_atomic_stores_only_a_changed_word_and_only_in_a_real_chunk`, and `changed_new`/`changed_old` in the random run | `rmw_batch` (plain and lock-free) and `rmw_if_changed` against the calls they stand for: 24 000 batches of 1 to 64 accesses of 1, 2, 4 and 8 bytes (repeats, chunk and allocation edges, misaligned, other allocations, host memory, uniform, real and untouched chunks, an operation that throws), values, errors and the accesses made before them, memory, chunks made real after every round; declines with faults armed | the plain batch stores an unchanged result (chunks made real differ, round 7); the lock-free batch stores always (round 0); `rmw_if_changed` ignores an armed fault |
| `test_memfast`, new `a_batch_declines_where_a_host_mapping_covers_one_of_its_addresses`, mode 10 of the batch fuzz (a misaligned lane inside the allocation) | `batch_ok`'s three tests | host-map test removed; alignment test removed (`MisalignedAccess` out of the reference call); allocation-bound test removed (segmentation fault) |
| `test_amd_gcn_tier_same_bits` (existing, `VGPU_TIER_ONLY=ds_write`) | the tier's handlers against the interpreter | `ds_write_b32` with the second offset (21 of 1140 runs differ) |

Also run on this tree, Release: `test_amd_gcn_same_bits` (and the `simd_base` twin), `_packed_`, `_tier_`, `_memory_`, `test_amd_codeobject_same_bits`,
`test_amd_matrix_fast`, `test_amd_mfma_math`, `test_amd_nan_order`, `test_amd_gcn_memruns` (five targets), `test_amd_gcn_atomics` (four), `test_memfast`, `test_memory`,
`test_memrmw`, `test_memrmw_lockfree`, `test_atom_rmw`, `test_memalloc`, `test_memcount_forms`, `test_sass_access_count`, and the AMD execution tests (89 of 89:
`test_amd_gcn_{exec,memory,bytes,int64,mixed,half,calls,builtins,packed,spill,crosslane,doubles,divide,narrow,lds,idioms,asm,cdna_gaps,abandon,threads,counters,mem_facts,globals,grid,ops}`
on every target they have, `test_amd_hostcall`, `test_amd_hip_queue`). Under ASan+UBSan (-O0, `halt_on_error`, leak detection on): `test_memrmw`, `test_memfast`,
`test_amd_gcn_memruns`, `test_amd_gcn_atomics`, `test_amd_gcn_memory_same_bits` (both tests), and `test_amd_gcn_tier_same_bits` restricted to `ds_`, `global_`,
`buffer_` and `flat_` (the full tier test at -O0 was not run). Under TSan (-O1, `setarch -R`): `test_memrmw`, `test_memrmw_lockfree`, `test_amd_gcn_memruns`,
`test_amd_gcn_atomics` (the threaded atomics). `e2e_sass_tier` and `e2e_sass_archs` skip on server3 (no nvcc): not run.

### Not done, and negative results

* **Wide scalar loads as one `load_run`** (`s_load_dwordx2` .. `x16`): `smem` 1.015x (p 0.30) on the MI300X and 1.038x (p 0.11) on the RX 6800; that workload is the
  scalar ALU's, `load_scalar` is 1.4% of it. Dropped.
* **`OpName::operator==` with the hash test inline and the text test out of line** (so the compare is small enough to be inlined at its uses): `lds` rx6800 1.038x and `flat`
  mi300x 1.041x (p < 0.001), the other fourteen rows of `lds`, `ldsx`, `mem`, `flat`, `valu_int`, `divergent`, `wave`, `buf` the same. Dropped: a gain under 4% in two rows,
  and a change of no behaviour, which no test can fail.
* **A single atomic through `rmw_scalar`'s `rmw_locate`** (the first version of the one-lookup atomic): `atomics` mi300x 0.948x (p 0.05), rx6800 1.004x, slower than the two
  calls: `rmw_locate` is out of line and checks the faults, the maps and the allocation on its own. `rmw_if_changed` on `hot_location` measured 1.058x / 1.046x before the
  batch (and the batch is what is kept for a wave).
* **`HostMaps::may_overlap`** is conservative where the maps' range and the addresses' range overlap (each address is then asked, as before): a program with a pinned
  buffer whose range spans the device windows (`in_device_window` set) gets no gain from it.
* **Not batched**: buffer atomics, byte and short buffer accesses, formatted (MTBUF) accesses, dwordx3 and dwordx4 gathers (a dwordx4 is one 16-byte access each, which
  `load_batch` is not), anything with peers attached, a load after a run that read the lane's address (the load memo), and every access that reaches LDS or private memory
  through a flat address.
* **Bench noise**: the pinned CPU's sibling was idle, but the host's load was 30 to 130 throughout; a run of 9 to 11 alternating pairs held the medians within 2% (A/A above).

## The SASS engine outside the JIT: diverged groups, barriers, atomics and memory (branch `sp-sassctl`, base d8ec6128)

Profiled first (one host thread, `cycles:u`, the RTX 3060 profile, `perf annotate` on a `-g` build): `divergent` spent 14% in `tier_lanes_at` and
only 35% of its runs were chained; `shared` 12% in `tier_release_bar` (a lane-by-lane loop over the block's warps) and a `switch` on the load's
sub-word extension in every lane of `h_shared`/`h_global`; `atomics` left the tier for `execute()` and spent 26% of `exec_mem` on a 64-bit
**division** (`addr % n`, n = 4 or 8) per lane, an O(n^2) search for the distinct addresses, a `host_address` call per distinct address, and up to 26
mutex pairs per instruction (the stripe was one per 8 bytes); `mem` called `MemoryManager::resolve` once per lane of every 16-byte access;
`warp` did 32 scalar lookups per shuffle. The ALU handlers and the SIMD kernels were not touched.

### What changed

* **The chain goes on into a diverged warp** (`tier_run_t`, tail): when the runnable lanes stand at several addresses and nothing asks the scheduler
  for more (no slice, give-way, WARPSYNC set or forced mask), the loop picks the group at the lowest address, as `step_warp` does, and goes on into it
  if it is in this section. `divergent`: 99.98% of the runs are chained (was 35%). `tier_lanes_at` (now `exec::sasslanes::lanes_at`,
  `include/vgpu/exec/sass_lanes.hpp`) compares four lanes per movemask.
* **Atomics** (`exec_mem`, `include/vgpu/exec/sass_atomics.hpp`): the lanes' distinct addresses come from a 64-entry hash instead of a scan; the
  alignment test is a mask; the host-mapping probe is gone (`load_batch` already declines every address `host_address` would answer); the stripe locks are
  held by a `StripeLocks` (no `unique_lock` array) and are per cache line (`atomic_stripe`: a warp's 32 consecutive words take two locks, not sixteen);
  operands are read as rows; the lane loop of each common operation (add, f32 add, min and max of unsigned, and, or, xor, exchange, compare-and-swap)
  is its own loop over a named functor (`sassatom::*Step`, chosen by `step_for`), the rest runs `sassatom::apply`, the former lambda. ATOMG, ATOM, ATOMS and RED
  are a tier form (`Form::Atom`) whose handler is `exec_mem`: the run loop counts and advances them, `execute()`'s probe of `fast_alu` and its dispatch are gone.
* **Shuffles** (`h_shfl`, `tier::shfl_shape`, `tier::shfl_move`): a full warp whose lane and clamp are the same in all lanes, with a segment mask of 0 and a
  lane under 32 (every `__shfl_xor/down/up/_sync` of a loop's offset), moves its words as a butterfly (`pshufd` and vector swaps), a shift or a splat; the
  in-range predicate is worked out for the warp. The analysis does it for constant lanes, the handler for a lane held in a register (`row_uniform`).
* **Barriers** (`sasslanes::wait/waiting_on/wake`): arrival and release of BAR and BSYNC move the 32 kinds, arguments and addresses a vector at a time;
  a block keeps its live thread count (`Block::live`, set by `init_block`, lowered by both EXITs) instead of counting the warps' lanes at every arrival.
* **Memory handlers**: `h_shared` moves a full warp's 32 consecutive words (or one word every lane names) as a block and counts them as one pass (`count_shared`,
  also the tail of `count_access`); `h_global` moves a coalesced run of words straight between memory and the register row (a load that declines has
  written nothing); a run of consecutive 16-byte pieces is one `load_run`/`store_run`; the addresses of a full warp in `h_global`, `h_shared` and
  `h_local` are counted loops; a word load no longer goes through the sub-word extension `switch`.

### Results

`bench.py --ab` base (build of d8ec6128) against this branch, `--size full`, 15 alternating runs, one host thread, CPU 4 of server3 (load 4 to 6: the other sessions
were idle), counters identical in every row. The A/A control of the same build against itself (15 runs, the first six rows) reads 0.995x to 1.001x. An earlier
11-run pass of all 16 workloads at load 15 to 33 gave the same picture (shared 1.99x, atomics 1.75x, mem 1.38x, local 1.20x, divergent 1.18x, warp 1.16x, launch
1.10x; texture, f32_fma, f64_fma, f64_subnormal, wmma, mma_ptx, api and startup 0.984x to 1.018x, "same"; its A/A: 0.985x to 1.018x, startup 1.026x with p 0.019 on 7 ms).

| workload | base Mi/s | this branch | B/A median | best | p | verdict |
|---|---|---|---|---|---|---|
| shared | 13.89 | 27.85 | **2.01x** | 2.01x | <0.001 | faster |
| atomics | 8.14 | 14.27 | **1.75x** | 1.75x | <0.001 | faster |
| mem | 18.80 | 26.90 | **1.43x** | 1.49x | <0.001 | faster |
| local | 35.37 | 43.02 | **1.22x** | 1.21x | <0.001 | faster |
| warp | 40.18 | 47.23 | **1.18x** | 1.20x | <0.001 | faster |
| divergent | 23.57 | 27.67 | **1.17x** | 1.17x | <0.001 | faster |
| launch (k/s) | 503 | 557 | 1.11x | 1.11x | <0.001 | faster (its kernel's coalesced load and store) |
| int_alu | 86.30 | 88.41 | 1.024x | 1.024x | <0.001 | faster by the table's rule; not a function this branch changed (the first pass read 1.018x, p 0.010), take it as code placement |
| texture | 4.16 | 4.17 | 1.002x | 0.999x | 0.51 | same |

One change at a time (each against the build before it, 9 to 15 runs, the load of the day varying, so the product of these is not the table above):
chain into a diverged group `divergent` 1.096x; `lanes_at` four per movemask `divergent` +6%, `warp` 1.08x; atomics batch 1.49x; `Form::Atom` 1.074x; the alignment
mask 1.034x (p 0.008); cache-line stripes 1.057x at one thread (and 2.0x at 8 host threads, load 94, counters differ there as the CAS loop's do); shuffle
shapes `warp` 1.128x; barrier state `shared` 1.23x and `divergent` 1.028x; `Block::live` `shared` 1.047x (p 0.008); shared word runs `shared` 1.095x; direct
row runs `mem` 1.133x and `shared` 1.152x; 16-byte runs `mem` 1.183x; counted address loops `mem` 1.067x, `shared` 1.086x and 1.088x; local and shared lane
copies without the extension `switch` `local` 1.188x.

`VGPU_TIER_STATS=1` after: `divergent` 3,592,605 of 3,594,537 runs chained (was 1.25 M of 3.6 M), `atomics` 94.9% of the warp instructions in the tier (REDUX 12,800,
VOTEU 3,200 and S2UR 128 are the interpreter's, see below), the others 99.2% to 100%.

### Tests (wired into CMake)

* `test_sass_lanes` (new): `lanes_at` and `lowest_pc` against the lane loop (addresses that differ only in the low or the high half), `wait`, `waiting_on`
  and `wake` against the lane loop (whole warps and masks, other kinds and arguments among the lanes), `row_uniform`, and `shfl_move` for every shape the analysis gives
  against the per-lane rule.
* `test_sass_atomics` (new): `sassatom::apply` and every `*Step` against a frozen copy of the lambda `exec_mem` had (all operations and types, compare-and-swap, the
  edges of each type, f32 sums that are subnormal).
* `test_tier`: the form table lists `atom`; `shfl_shape` against the per-lane rule for every mode, lane 0 to 40 and every clamp and segment mask.
* The programs that run the whole engine are `bench_counters*` (results checked against the host, counters against the recorded ones), `e2e_sass_tier` (tier against
  the interpreter, `k_direct` has the atomics of every operation and type with their old values), `e2e_sass_fastpath` (the fast paths against `VGPU_FASTPATH=0`: coalesced
  and scattered loads and stores of 1 to 16 bytes, shared accesses with conflicts and broadcasts, `atomic_forms` over one address, a few, distinct and scattered ones,
  `shared_atomic_forms`) and `e2e_sass_archs`. None of the e2e programs could be built on the experiment server (no nvcc); the harness of `tools/bench` ran with the
  cached program, which checks every result and every counter against the base build.

### Mutation checks (a one-line change on the experiment server, the test named failed; reverted)

| mutation | test that failed |
|---|---|
| `lanes_at` without the high halves | `test_sass_lanes` |
| `wake` of a whole warp adds 32, not 16, to the addresses | `test_sass_lanes` |
| f32 atomic add without the flush of a subnormal result | `test_sass_atomics` (it did not fail until the operands included sums that are subnormal: the pool now has them) |
| `step_for` gives the and-step to OR | `test_sass_atomics` |
| shuffle shape Down's in-range count off by one | `test_tier`, `test_sass_lanes` |
| butterfly `pshufd` order of the xor-1 swap | `test_sass_lanes` |
| `Form::Atom` without `dst = kNoReg` | `test_tier` (3 of 8 failed before it was set) |
| chain group = all runnable lanes | bench `divergent`: `collatz thread 0` |
| atomic address hash treats a collision as the same address | bench `atomics`: `hist[0]` |
| `h_atom` drops the lowest lane | bench `atomics`: launch timeout (a CAS loop never ends) |
| atomics' stripe locks not taken (8 host threads) | bench `atomics`: `single counter` |
| `Block::live` three warps short of the block | bench `shared`: `block 0 reduction` |
| BAR arrival leaves out the lowest lane | bench `shared`: hangs (killed by the harness) |
| shared word run copies from `at + 4` | bench `shared`: `block 0 reduction` |
| shared word run counted as two passes | bench `shared`: counters differ (`shared_bank_conflicts`) |
| direct row run starts one lane late | bench `shared`: `block 0 reduction` |
| 16-byte run transposes the words | bench `mem`: `stream4[0]` |
| local lane loop stops at lane 30 | bench `local`: `thread 351` |

### Not done, and negative results

* *Pick the first lane's group first, look only at the other lanes for a lower address* (`tier_pick_group`), and an early return of `tier_release_bsync` when a member is not
  waiting at all: `divergent` 1.000x (p 0.49), `warp` 0.987x; dropped.
* *A tier form for REDUX and VOTEU* (exec_warp does not use the group, so the handler is one line): `atomics` 1.010x (p 0.51); dropped.
* *The shared word run for any contiguous prefix of lanes*: with a `memcpy` of `count` words and the lanes found by bit scan `shared` 0.891x; with counted loops
  and four-word vector copies 1.008x (p 0.17); only the full warp has the fast path.
* Earlier entries of this file say SASS atomics had nothing to gain ("the one-lookup read-modify-write", and the "direct" dispatch of the instructions that end a
  run, `atomics` 1.008x). The first was about `MemoryManager::rmw_scalar`, which the batch already avoided; the second left `exec_mem` as it was. The gain here is in
  `exec_mem`'s own lane code (the division, the scan, the probes, the locks) and, once that was cheap, in the tier form (1.074x).
* *Atomics' value rules through a template taking the step as a generic lambda* (`sassatom::dispatch`): `atomics` 0.96x (p < 0.001) with one more percent of
  instructions and three times the L1i misses (`exec_mem` split in half, the lane loops in an out-of-line `dispatch`); replaced by named functors and a `switch`, 1.004x of the
  version it was cut from. A lesson for the next one: after moving code into a header, A/B again, whatever the instruction count says.
* *Not taking the stripe locks at all* is 1.14x of `atomics` at one host thread (measured to see what is left); not a change one can make.
  A spin lock for the stripes, and one `MemoryManager::rmw_batch` instead of `load_batch` and `store_batch` (each runs `batch_ok` over the addresses: 6 to 9% of
  `atomics`) were not tried.
* `memcount::detail::sectors_fast` is 25% of `local` (a thread's private array indexed by a value of its own: the window form walks all 32 lanes); the counts have closed
  forms for strides and windows, not for this; left.
* `step_warp` itself is 45% of `divergent` and flat (the three counters, `w.steps`, the guard, the call per op): the divergent path's cost is the per-lane work of
  `tier_set_pcs` and the branch select, 64-bit addresses a lane. Keeping the counters in registers across a run would touch the loop that tier 2 hooks into.
* Sanitizers, on the experiment server: `test_tier`, `test_sass_lanes` and `test_sass_atomics` built `-fsanitize=address,undefined -O0` (Debug) and run with
  `detect_leaks=1`, `halt_on_error=1`: clean. The cached bench program run on a shim built the same way (`LD_PRELOAD` of libasan, quick size) for `divergent`, `shared`,
  `warp`, `atomics`, `mem` and `local`: all PASS, no report. The shim built `-fsanitize=thread -g -O1` (`setarch -R`, `VGPU_THREADS=4`) for `atomics`, `shared`,
  `divergent` and `mem`: PASS, no warning. Not run there: the e2e programs and their sanitizer legs (no nvcc), `test_jit` (fails on the base too: no `c++` on its PATH).
