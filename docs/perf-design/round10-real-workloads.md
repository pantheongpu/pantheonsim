# Round 10, step 0: where the time goes on real workloads

Measured 2026-10-18 on a dedicated AWS c7i.4xlarge (Xeon Platinum 8488C, 16 vCPU, AVX-512, 30 GB, Ubuntu 24.04), nothing else running,
`origin/main` 89d6bcd5 built Release in the CI toolchain image (CUDA 13.0), PyTorch 2.14.1+cu130. Run-to-run spread on the synthetic
workloads was 0.2 to 1.4% (the shared WSL machine of the earlier rounds had 3 to 60%). Raw files are in `data/round10/`.

## 1. The synthetic workloads, tier off and on (`bench.sh --runs 5 --cpus 2-5`, SASS engine, `VGPU_THREADS=1`)

Warp instructions per device-CPU second, in millions; "x" is tier on over tier off.

| workload | tier off | tier on | x |
| --- | ---: | ---: | ---: |
| int_alu | 37.65 | 99.90 | 2.65 |
| f32_fma | 41.74 | 109.41 | 2.62 |
| f64_fma | 29.63 | 46.41 | 1.57 |
| f64_subnormal | 5.58 | 5.96 | 1.07 |
| wmma | 1.26 | 1.40 | 1.11 |
| mma_ptx | 2.44 | 2.72 | 1.12 |
| mem | 11.86 | 16.76 | 1.41 |
| shared | 13.35 | 19.71 | 1.48 |
| atomics | 6.01 | 7.87 | 1.31 |
| texture | 3.19 | 3.41 | 1.07 |
| local | 19.00 | 37.26 | 1.96 |
| divergent | 16.28 | 30.86 | 1.89 |
| warp | 19.35 | 64.59 | 3.34 |

These agree with the numbers in docs/performance.md for the tier, with one difference: `wmma` and `mma_ptx` gain 1.1x here, not 1.4 to 1.6x.

## 2. The PyTorch sweep (quick tier, 169 checks, 13 processes, `nvidia/rtx5090`)

| | wall |
| --- | ---: |
| `VGPU_TIER=0` | 100.6 s |
| `VGPU_TIER=1` | 101.6 s |

**The tier changes nothing here.** Every group has the same ok / known-failure counts in both runs (the `compile` group fails in both: this rig has
no Inductor toolchain set up, which is not a simulator result). With `VGPU_TIER_STATS=1` the whole sweep executes 51.1 M warp
instructions, 88.9% of them in the tier. At the 20 to 100 M/s of section 1 that is under 5 seconds of the 100.

## 3. Where a PyTorch process spends its time

A process that does `import torch`, creates a tensor and runs four small elementwise kernels takes 2.2 s: `import torch` 0.94 s, the first
CUDA call 0.06 s, a 256x256 matmul 0.02 s (host-computed cuBLAS), and the four kernels 0.59 s.

The first launch of each distinct kernel in a process costs 0.05 to 0.2 s, and later launches cost about nothing (`data/round10/mini2.py`):

| step | rtx5090 | rtx3060 |
| --- | ---: | ---: |
| add (first elementwise kernel) | 0.145 s | 0.128 s |
| add again (same kernel) | 0.000 s | 0.001 s |
| mul (new kernel, same file) | 0.091 s | 0.081 s |
| relu (new kernel) | 0.124 s | 0.103 s |
| sum reduction (new kernel) | 0.217 s | 0.172 s |
| softmax (new kernel) | 0.161 s | 0.140 s |
| layer_norm (new kernel) | 0.055 s | 0.046 s |
| layer_norm again | 0.003 s | 0.003 s |

`perf` on the `nn` sweep group (9 s wall, BLAS threads off so idle spinning does not hide it): the simulator library is 29% of the samples,
libc 40% (mostly `malloc`/`free`), zstd 5%. Under the library the inclusive time is in SASS decoding: `vgpu::sass::decode`,
`std::vector<Operand>::emplace_back` (6.9% inclusive), the decoder lookup `std::_Hashtable<unsigned, void(*)(Instr&, const Word&)>` (3.3%), and
text formatting at decode time (`float_text` 2.3% inclusive, `signed_hex`). zstd is the fatbin decompression. Together with the allocator work
they drive, loading and decoding kernels is roughly a third of that group.

Not measured here: CUTLASS GEMM/FMHA run time, the pantheon stress workloads at full size, other architectures' profiles.


## Round 10, step 0 (continued): the heavy real workloads, the tier, and the soak

Measured 2026-10-18/19 on the same dedicated c7i.4xlarge (about 3.5 hours up by the end), `origin/main` 89d6bcd5 as built in step 0,
CI toolchain image (CUDA 13.0), PyTorch 2.14.1+cu130. The measure workstream had CPUs 0,1,2,3,8,9,10,11; **timing runs were pinned to
CPU 1 and CPU 2 with their hyper-thread siblings (9, 10) idle**, tier off and tier on **in pairs at the same time, one on each CPU, with the
CPUs swapped every repetition** (so neither mode always had the same core), and a repetition is one pair. All runs use `VGPU_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`. The other two workstreams ran on their own cores at the same time (load average 11 to 64 in the raw lines,
`data/round10/measure/rig/*summary*.txt`); wall times below are of the pinned process, which a neighbour's load does not change except
through memory bandwidth and L3 (spread is given). Raw numbers, the scripts that produced them and the analysis scripts are in
`docs/perf-design/data/round10/measure/` (`scripts/`). Nothing in `src/` was changed. "Tier coverage" is `VGPU_TIER_STATS=1`:
the share of warp instructions that ran in the tier (the rest ran in the interpreter).

### A. Where the time goes, on every heavy workload

| workload | tier on/off, wall | tier coverage | where the CPU time is (perf, tier on) |
| --- | --- | --- | --- |
| PyTorch sweep, 12 groups (full tier) | 0.93 to 1.00 per group, 0.975 summed | 83 to 93% of 70.5 M warp instr | in the three groups profiled: 29 to 65% SASS decode/parse, 5 to 10% zstd, 3 to 13% SASS execution (section D) |
| CUTLASS GEMM, Hopper/Blackwell, SASS | 0.96 to 1.00 | 71 to 89% | 24 to 90% tensor-core math, 5 to 45% the test's host reference (section C) |
| CUTLASS GEMM, Ampere sparse | 1.00 | 78% | 45% tensor-core math, 45% host reference |
| CuTe TMA load (Hopper) | **0.77** (1.30x) | 95% | memory/TMA 27%, tier handlers 24% |
| CUTLASS FMHA (Blackwell) | 1.00 | **0%: it runs the PTX interpreter** (FFMA2 is not decoded) | 88% PTX interpreter |
| pantheon stress workloads | 1.14x to 1.92x faster | 92 to 100% | not profiled |
| HeCBench / polybench / CUDA Samples (sum of per-program wall, timeouts and the two flaky runs left out) | 0.64 / 0.76 / 0.79 | 98% of 64.5 G over all runs | not profiled |

**Summary.** The tier is worth 1.5x to 1.9x on pantheon's integer, FP and scheduler stress kernels and 1.3x on a CuTe copy test, and
0 to 7% on everything that looks like machine-learning work. On the PyTorch sweep and on CUTLASS GEMMs the time is not instruction
dispatch: it is (1) decoding every instruction of every kernel of a fatbin when the fatbin is registered, (2) zstd decompression of the
fatbin, (3) the tensor-core arithmetic (`wgmma`, `tcgen05`, `mma` narrow-type sums), (4) the host reference the CUTLASS tests run on the
CPU, none of which the tier touches.

### B. PyTorch sweep, tier off against on

`run_pytorch_sweep.sh` groups, `nvidia/rtx5090`, one process per group, quick tier 3 pairs and full tier 5 pairs per group (the `compile`
group is left out: this rig has no Inductor toolchain). Only runs that ended with exit status 0 are used for wall time and RSS (see "the exit
abort" below); "n/a" means fewer than 2 such runs on one side.

**Quick tier**

| group | wall off, s (min-max) | wall on, s (min-max) | on/off | RSS off / on, MB | warp instr | in tier % | clean runs off/on of all |
|---|---:|---:|---:|---:|---:|---:|---:|
| amp | 6.16 (6.13-6.35) | 6.25 (6.16-6.35) | 1.015 | 1420 / 1429 | 1,129,111 | 89.7 | 3/2 of 3 |
| attention | 7.51 (7.37-7.52) | 7.18 (7.07-7.20) | 0.956 | 1355 / 1362 | 9,685,910 | 92.7 | 3/3 of 3 |
| fft | n/a | n/a | n/a | n/a / n/a | 0 | 0.0 | 0/0 of 3 |
| graphs | 5.94 (5.83-5.96) | 5.94 (5.83-5.96) | 1.000 | 1235 / 1236 | 641,454 | 84.4 | 3/3 of 3 |
| linalg | 7.45 (7.44-7.48) | 7.46 (7.46-7.49) | 1.001 | 1376 / 1387 | 310,438 | 86.7 | 3/3 of 3 |
| losses | 6.61 (6.61-6.62) | 6.61 (6.61-6.63) | 1.000 | 1326 / 1330 | 241,565 | 90.5 | 3/3 of 3 |
| nn | 8.96 (8.95-8.98) | 8.93 (8.90-8.96) | 0.996 | 1717 / 1735 | 3,397,697 | 84.5 | 2/2 of 3 |
| ops | 14.71 (14.70-14.74) | 14.31 (14.29-14.36) | 0.973 | 2250 / 2283 | 18,660,684 | 89.5 | 3/3 of 3 |
| optim | 7.02 (6.93-7.08) | 6.75 (6.71-6.83) | 0.962 | 1360 / 1363 | 7,691,793 | 87.3 | 3/3 of 3 |
| rnn | 4.88 (4.83-4.92) | 4.88 (4.81-4.92) | 1.000 | 1132 / 1134 | 470,005 | 87.9 | 3/3 of 3 |
| sdpa | 3.60 (3.58-3.62) | n/a | n/a | 809 / 817 | 7,431,505 | 86.8 | 3/1 of 3 |
| sparse | 4.89 (4.86-4.91) | 4.88 (4.85-4.91) | 0.998 | 1023 / 1030 | 653,443 | 83.2 | 3/3 of 3 |
| sum over the groups with clean runs on both sides | 74.1 | 73.2 | 0.987 | | 50,313,605 | 88.9 | |

**Full tier**

| group | wall off, s (min-max) | wall on, s (min-max) | on/off | RSS off / on, MB | warp instr | in tier % | clean runs off/on of all |
|---|---:|---:|---:|---:|---:|---:|---:|
| amp | 6.50 (6.19-6.53) | 6.48 (6.16-6.53) | 0.997 | 1447 / 1455 | 1,173,534 | 89.8 | 4/5 of 5 |
| attention | 8.04 (7.77-8.16) | 7.61 (7.33-7.75) | 0.947 | 1355 / 1362 | 13,658,132 | 92.6 | 5/5 of 5 |
| fft | n/a | 3.38 (3.34-3.41) | n/a | 821 / 824 | 114,101 | 83.5 | 1/2 of 5 |
| graphs | 6.72 (6.60-6.76) | 6.71 (6.58-6.75) | 0.999 | 1317 / 1325 | 1,190,074 | 87.6 | 5/5 of 5 |
| linalg | 7.51 (7.48-7.58) | 7.52 (7.49-7.61) | 1.001 | 1377 / 1387 | 315,449 | 86.7 | 5/5 of 5 |
| losses | n/a | n/a | n/a | 1391 / 1395 | 290,457 | 90.0 | 1/1 of 5 |
| nn | 9.75 (9.65-9.76) | n/a | n/a | 1806 / n/a | 0 | 0.0 | 4/0 of 5 |
| ops | 14.74 (14.74-14.84) | 14.34 (14.33-14.41) | 0.973 | 2250 / 2283 | 18,660,684 | 89.5 | 5/5 of 5 |
| optim | 9.24 (8.98-9.46) | 8.85 (8.61-9.72) | 0.958 | 1559 / 1562 | 12,418,652 | 86.7 | 5/5 of 5 |
| rnn | 6.14 (5.87-6.25) | 6.13 (5.85-6.25) | 0.998 | 1258 / 1260 | 759,485 | 88.9 | 5/5 of 5 |
| sdpa | 7.23 (6.93-7.33) | 6.70 (6.42-6.81) | 0.927 | 1136 / 1155 | 21,294,184 | 87.2 | 5/5 of 5 |
| sparse | 6.10 (5.83-6.21) | 6.10 (5.83-6.21) | 1.000 | 1151 / 1158 | 655,059 | 83.2 | 5/5 of 5 |
| sum over the groups with clean runs on both sides | 72.2 | 70.4 | 0.975 | | 70,529,811 | 88.8 | |

A row with 0 warp instructions is a group whose tier-on runs all ended in the exit abort below before the statistics were written
(full `nn`, quick `fft`); the quick tier's `nn` statistics (3.4 M warp instructions, 84.5% in the tier) stand for full `nn`. Warp
instructions are 0.2 to 21 M per group: at 20 to 100 M/s that is about a second at most of a 4 to 15 s process (section D). The serial repeat of the four groups that abort
most (12 alternating runs each, one CPU, `data/round10/measure/sweep-abort-serial.txt`; single CPU, so these walls are not comparable with the pair walls above):

| group | aborts at exit off / on (of runs) | clean-run wall off, s | clean-run wall on, s | on/off |
|---|---:|---:|---:|---:|
| nn | 7/12 , 5/12 | 9.69 (9.64-9.73, n=5) | 9.66 (9.58-9.72, n=7) | 0.997 |
| sdpa | 0/12 , 0/12 | 7.20 (7.07-7.39, n=12) | 6.44 (6.28-11.57, n=12) | 0.894 |
| losses | 4/12 , 11/12 | 7.33 (7.29-15.13, n=8) | n/a (n=1) | n/a |
| amp | 1/12 , 1/12 | 6.31 (6.14-6.52, n=11) | 6.23 (6.09-6.50, n=11) | 0.987 |

**The exit abort.** `fft`, `losses`, `nn` and sometimes `amp` and `sdpa` end with `terminate called without an active exception` and
exit status 134, **in both modes**, after every check has printed `ok`. Across the two sweeps 14 of 96 tier-off runs and 19 of 96 tier-on runs
did it; the rate by group differs with the mode in both directions (`nn` 7/12 off, 5/12 on; `losses` 4/12 off, 11/12 on) and is a timing
race. A core file (read with gdb) shows the cause: PyTorch's autograd worker thread (`pt_autograd_0`) destroys a `GraphTask` after the
main thread has begun `Py_FinalizeEx`; `~ThreadLocalState` takes the GIL (`pybind11::gil_scoped_acquire`), CPython 3.12 calls
`PyThread_exit_thread` -> `pthread_exit`, and the forced unwind reaches `std::terminate`. That is PyTorch against CPython at interpreter
shutdown, not a simulator frame. It matters to us because `run_pytorch_sweep.sh` fails a group whose exit status is not 0: a flaky red
that the tier did not cause and does not cure. (A fix would be `os._exit(0)` after the last `ok` line in `sweep.py`; not tried.)

### C. CUTLASS GEMM, FMHA and CuTe

Built from CUTLASS v4.8.0 as `run_cutlass_*.sh` do (their gtest filters, their FMHA cases), but with `-code=sm_XXa,compute_XXa` so the
SASS exists (`VGPU_CUTLASS_SASS=1`'s flags); each runs on its generation's profile with one host thread, 3 pairs each:

| case | sim GPU | wall off (s) | wall on (s) | on/off | RSS off / on MB | warp instr | in tier % | result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tma_load | h100 | 5.0 (5.0-5.2) | 3.9 (3.9-3.9) | 0.767 | 293 / 318 | 48,235,588 | 95.3 | 8 passed; n=3 |
| sm80_sparse | a100 | 64.9 (63.5-67.0) | 64.8 (62.8-65.2) | 0.998 | 70 / 73 | 28,081,680 | 78.5 | 19 passed; n=3 |
| sm90_group_gemm | h100 | 341.5 (340.8-345.5) | 337.0 (336.6-337.5) | 0.987 | 230 / 231 | 262,235,995 | 86.0 | 2 passed; n=3 |
| sm90_pingpong | h100 | 283.8 (283.7-286.1) | 279.6 (279.4-281.1) | 0.985 | 36 / 36 | 235,811,188 | 84.1 | 2 passed; n=3 |
| sm90_sparse_f16 | h100 | 171.0 (170.9-171.3) | 164.9 (164.8-165.1) | 0.964 | 35 / 36 | 202,083,525 | 88.8 | 5 passed; n=3 |
| sm100_f16 | b200 | 262.6 (262.5-263.1) | 259.0 (258.9-259.7) | 0.986 | 41 / 43 | 257,971,645 | 81.4 | 7 passed; n=3 |
| sm100_f8 | b200 | 116.4 (115.8-118.5) | 115.7 (115.0-119.0) | 0.994 | 30 / 30 | 26,514,808 | 86.1 | 4 passed; n=3 |
| sm100_blockscaled | b200 | 44.8 (44.8-45.3) | 44.8 (44.8-44.9) | 0.999 | 26 / 26 | 13,355,344 | 81.8 | 1 passed; n=3 |
| sm120_f4_f6 | rtx5090 | 26.1 (26.0-26.4) | 26.1 (26.0-26.1) | 1.000 | 24 / 24 | 2,822,337 | 70.7 | 2 passed; n=3 |
| sm120_bs_mxf4 | rtx5090 | 33.1 (33.0-33.5) | 33.0 (33.0-33.2) | 0.998 | 31 / 31 | 4,800,160 | 80.4 | 5 passed; n=3 |
| fmha1 | b200 | 94.5 (93.8-95.7) | 94.0 (93.8-94.4) | 0.995 | 101 / 101 | 0 | 0.0 | 1 [OK] passed; n=3 |
| fmha2 | b200 | 94.0 (93.9-94.5) | 94.0 (93.8-94.3) | 1.000 | 120 / 120 | 0 | 0.0 | 1 [OK] passed; n=3 |
| fmha3 | b200 | 96.9 (96.8-97.5) | 96.9 (96.8-97.3) | 1.000 | 106 / 106 | 0 | 0.0 | 1 [OK] passed; n=3 |
| fmha4 | b200 | 155.8 (155.4-156.1) | 155.7 (155.6-156.0) | 1.000 | 120 / 120 | 0 | 0.0 | 2 [OK] passed; n=3 |

All of them print the gtest/`[OK]` lines of the unmodified tests and pass; the 42 pairs are byte for byte the same apart from gtest's
millisecond timings (section F). RSS is the process's peak.

Compile time, reported apart from the runs (nvcc 13.0.88 `-O1`, SASS and PTX, two compiles at a time on four cores; one repeat only):
sm80 sparse f16 35 s, sm90 group GEMM 41 s, sm120 f4_f6 45 s, sm120 mxf4 108 s, sm100 f16 96 s, sm100 f8 99 s, sm100 blockscaled 138 s,
sm90 ping-pong 110 s, sm90 sparse f16 110 s, FMHA 109 s, CuTe `tma_load` 304 s. The first nvcc invocation for sm80 failed (`sm_80a` is not a
target); that is not counted.

Where the time goes (perf `-F 300`, tier on, one CPU, flat self time by symbol class; the classes are in `docs/perf-design/data/round10/measure/scripts/pcat.py`):

| case | tensor-core math | test program (host reference, fills) | memory system / TMA | libc / libm | tier handlers / SIMD | other |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sm120 f4_f6 | 88% | 6% | 2% | 2% | 0% | |
| sm120 mxf4 | 89% | 5% | 2% | 3% | 0% | |
| sm100 blockscaled | 71% | 7% | 4% | 12% | 0% | |
| sm100 f16 | 28% | 17% | 22% | 22% | 0% | other simulator 6%, scheduler 3% |
| sm100 f8 | 24% | 28% | 21% | 22% | 0% | decode 2.5% |
| sm90 group GEMM | 56% | 19% | 9% | 11% | 0% | scheduler 2% |
| sm90 ping-pong | 58% | 11% | 12% | 14% | 0% | scheduler 2% |
| sm80 sparse f16 | 45% | 45% | 4% | 3% | 1% | scheduler 3% |
| CuTe tma_load | 0% | 10% | 27% | 15% | 24% | scheduler (non-tier) 13% |
| FMHA (PTX interpreter) | 0% | 0% | 8% | 0% | 1% | **PTX interpreter 88%** |

The functions are `vgpu::exec::mma_narrow_sum` (77% of sm120 f4_f6 by itself), `Runner::exec_gmma` (52% of the sm90 group GEMM),
`tcgen05_mma` + `SassTcHost::smem_load` (17% of sm100 f16), `exec_mma` + `mma_sparse_column` (40% of sm80 sparse), and
`cutlass::reference::host::gett_mainloop` / `compute_gemm` for the host reference. None of these is dispatch work, so a faster dispatch does not touch them.

**FMHA does not run SASS.** `77_blackwell_fmha` built for `sm_100a` stops at `SASS: unknown opcode 0xc49 (sm_100)` and falls back
to its PTX ("running PTX instead of SASS"), so the tier never runs and the four cases take 94 to 156 s in the PTX interpreter in both modes. `cuobjdump -sass` of the
binary shows 0xc49 is **`FFMA2`**; the same cubin also contains `FMUL2` (6144 instructions), `FADD2` (3752), `FMNMX3` (3640) and `FFMA2`
(3584); whether the decoder has the other three was not checked (the first unknown opcode ends the check). The message is also cut short:
`unsupported()` formats its prefix into `char at[96]`, so for a CUTLASS kernel name the offset and function are lost.

### D. The PyTorch process: where the time goes

`perf record --call-graph dwarf`, tier on, one CPU, full-tier groups (`docs/perf-design/data/round10/measure/scripts/profgrp.sh`, `prof2.py` beside it); every sample is put in the first
class whose function is anywhere in its stack, innermost class first (execution, then zstd, then decode, ...):

| class (share of samples) | `ops` (14.3 s, longest) | `nn` (9.3 s) | `sdpa` (6.7 s) |
| --- | ---: | ---: | ---: |
| SASS decode/parse: `vgpu::sass::unsupported` -> `decode`, with the malloc/free/snprintf under it | **64.7%** | 45.4% | 29.2% |
| fatbin decompress (zstd, `extract_elf`) | 10.2% | 6.3% | 4.9% |
| SASS execution (`sass::launch`, `run_kernel`, worker threads) | 7.6% | 3.1% | 12.7% |
| python import | 6.8% | 9.2% | 6.1% |
| python/torch op code, and samples whose stack could not be unwound | 5.2% | 30.2% | 33.5% |
| other: the rest of the simulator, `pick_cubin` remainder, loader, kernel | 5.1% | 5.0% | 12.9% |
| host-side library (cuBLAS host GEMM and so on) | 0.4% | 0.8% | 0.7% |

Inside the `ops` decode share: **`vgpu::cuda::pick_cubin` is 75% of the process inclusive, `vgpu::sass::unsupported` 63%, `decode` 35%,
`extract_images`/`extract_elf` (zstd) 10%, `parse_cubin` 3%.** 38% of all `ops` samples are in the decode class with a leaf frame in the
allocator or libc (malloc, free, new, delete, memcpy, snprintf internals) or a page fault. `__snprintf_chk` is **26% of the whole process**, and
22.1% of it is called directly from `unsupported()`.

The cause is in `src/sass/module.cpp`, `unsupported()`: when a fatbin is registered, `pick_cubin` (nvidia/src/fatbin_sass.cpp) calls it
to ask "can the executor run every instruction of this cubin?", and it decodes **every instruction of every `.text` section** (torch's
cubins have thousands of kernels) and, for each one, first formats a prefix with `snprintf(at, sizeof at, "sm_%d %s+0x%zx: ", ...)` that is
used only if the instruction turns out unsupported. That prefix, the `Instr` (with its `std::vector<Operand>`) and its text are built and thrown
away for every instruction. This is the "a PyTorch process pays 0.05 to 0.2 s per kernel" of step 0 seen from the other side: the
first launch of a new fatbin pays for the whole fatbin. Candidate changes, **none tried (src is not mine to change; effect NOT
MEASURED)**: build the prefix only on failure (about 22% of the `ops` process by the numbers above), decode into a reused `Instr`, or
check support per kernel when it is first launched rather than per fatbin at registration; zstd (10%) can be cached on disk or done
once per fatbin on demand.

### E. The pantheon stress workloads (tier off against on)

Built from `/home/saqib/tools/pantheon/pantheon` (ae46a79) for sm_86 in the rig container with `vgpu shell` (compile time not recorded), run as `tools/bench/bench.sh pantheon` does (`0 20 2 --kernel_loops 2 --grid_size 4`,
1 GB VRAM, `VGPU_COUNTERS=1`, `nvidia/rtx3060`, `--gemm_size 256` for the two that take it), 20 s each, 3 pairs. They are duration-driven, so
the table is warp instructions per CPU second (counters' `inst_executed` summed for tier off; the tier's own count for tier on; counters
are on in both, which lowers both rates) with the instruction count of the 20 s:

| workload | Minst/s off | Minst/s on | x | warp instr / 20 s (off, on) | in tier % | top fallbacks (share of all warp instr) |
|---|---:|---:|---:|---|---:|---|
| int_virus | 31.04 (30.89-31.21) | 54.90 (52.50-55.85) | 1.77 | 620.5 M, 1097.4 M | 100.0 |  |
| fp64_virus | 25.65 (25.59-25.79) | 39.27 (37.47-39.89) | 1.53 | 512.2 M, 784.7 M | 100.0 |  |
| compute_virus | 17.94 (17.90-17.95) | 34.08 (33.43-34.12) | 1.90 | 358.7 M, 681.6 M | 100.0 |  |
| mma_virus | 3.01 (3.01-3.02) | 3.45 (3.44-3.46) | 1.15 | 60.2 M, 68.9 M | 99.6 | F2F.F16.F32 0.4% |
| omni_virus | 12.96 (12.93-13.08) | 20.69 (20.67-20.77) | 1.60 | 259.3 M, 413.8 M | 99.7 | CALL.REL.NOINC 0.2%, HFMA2 0.1%, FFMA 0.1% |
| tensor_virus | 12.40 (12.23-12.43) | 20.59 (20.54-20.61) | 1.66 | 247.8 M, 411.7 M | 98.5 | F2F.F16.F32 1.5% |
| sfu_stress | 21.22 (21.20-21.23) | 36.24 (36.21-36.32) | 1.71 | 424.0 M, 723.8 M | 97.7 | CALL.REL.NOINC 1.2%, FCHK 0.6%, RET.REL.NODEC 0.6% |
| scheduler_virus | 27.57 (26.75-27.68) | 52.97 (52.08-53.12) | 1.92 | 551.1 M, 1057.7 M | 100.0 |  |
| transformer_virus | 2.39 (2.39-2.40) | 2.72 (2.72-2.74) | 1.14 | 47.9 M, 54.5 M | 100.0 |  |
| atomic_virus | 7.75 (7.74-7.76) | 9.01 (8.99-9.02) | 1.16 | 177.2 M, 201.4 M | 91.7 | RED.E.ADD.STRONG.GPU 8.3% |

The three that gain least (`mma_virus`, `transformer_virus`, `atomic_virus`: 1.14 to 1.16x) were not profiled (`atomic_virus` has 8.3%
of its instructions in the interpreter as `RED.E.ADD.STRONG.GPU`); the other seven are the 1.5x to 1.9x the synthetic benchmarks of section 1
promised.

### F. The soak: tier off against tier on, byte for byte

What was compared, and every difference found:

| set | pairs | identical | differences |
| --- | ---: | ---: | --- |
| PyTorch sweep, quick, 12 groups x 3 | 36 | 36 (33 plus 3 that differ by the exit-abort line only) | none |
| PyTorch sweep, full, 12 groups x 5 | 60 | 60 (52 plus 8, the same) | none |
| CUTLASS/CuTe/FMHA, 14 cases x 3, one host thread | 42 | 42 | none (gtest `(N ms)` normalised) |
| the same 6 CUTLASS cases and the 12 sweep groups with `VGPU_COUNTERS=1` (every counter of every launch, 170,000 lines) | 18 | 17 | `ops`: the out-of-memory message contains the MemAvailable figure (1822 MB against 1827 MB) |
| the same with `VGPU_THREADS=4` (shared Program) | 18 | 18 | none |
| polybench/GPU (CPU-checked) | 15 | 14 | `SYRK`: tier off failed once (below) |
| Rodinia | 4 | 4 | none |
| HeCBench | 105 | 104 | `car-cuda`: tier on FAILed once (below) |
| CUDA Samples v13.0 | 129 | 124 | 2 finished only with the tier inside 300 s; 3 print timings or thread order |

Total 427 pairs; 19 pairs differ (11 of them by the exit-abort line only) and every one is explained below; **none is a tier bug**:

* *Timing lines* (gtest `(N ms)`, `Gsamples/s`, `Kernel timing`, `updates per ns`, `done in`, dates): normalised away.
* *Thread order and timing in the program*: CUDA Samples `UnifiedMemoryStreams` (which task ran on the host or device depends on timing),
  `threadMigration` (the order of the thread messages and which device a thread printed), `SobolQRNG` (a rate): races in the
  programs.
* *The step budget*: CUDA Samples `MonteCarloMultiGPU` (279 s with the tier, over 300 s without) and `simplePitchLinearTexture` (184 s, over
  300 s) completed only with the tier within the harness's 300 s; HeCBench `car-cuda` took 446 s without and 311 s with the tier when
  rerun alone (both PASS, RMSE 0.006348, byte for byte the same output); 8 more CUDA Samples and 1 HeCBench program hit 300 s in both modes (not compared past that).
* *The exit abort* of section B, 3 + 8 sweep pairs.
* *Two single failures, neither tied to the tier*: polybench `SYRK` in tier-off mode (the kernel never ran: GPU time 0.0005 s, 1,046,529
  wrong outputs; about a dozen reruns, tier off and on, alone and in pairs, all PASS) and HeCBench `car-cuda` in tier-on mode (it stopped after 6,728
  warp instructions with `RMSE: 100005.96 FAIL`, 18 s; the two reruns above PASS). `car-cuda` ran next to its twin in the other mode (11.9 GB per process on a 30 GB host); `SYRK` ran while the
  container was compiling CUDA Samples (three nvcc at once); `VGPU_QUIET=1` hid the simulator's messages in both. The simulator refuses a
  device allocation when the host's MemAvailable falls under its 1892 MB reserve, and that message was seen (counters pair, `ops`,
  06:15) while two `car-cuda` reruns of mine were alive. The cause of the two failures is therefore **very probably the host-memory guard
  tripped by my own parallel runs, but NOT proven for those two runs**.
* *The suite harness itself*: the first `rodinia` run raced between the two modes (they share `run-<app>` directories) and was rerun serially (4/4
  identical); the `*_nvrtc` samples exit 1 in both modes on this rig (no NVRTC headers on the host side), and `cudaGraphsPerfScaling`
  did not build (no `nvtx3`), `simpleCUFFT_callback` was left out (no static cuFFT in the image).

Not covered by a byte-for-byte comparison: the pantheon workloads (duration-driven), FMHA (never reaches the tier: PTX), the PyTorch
`multi` / `dist` / `compile` groups (not in this sweep or no toolchain), any architecture but sm_120 for PyTorch and sm_86 for the external suites'
binaries, and **the sanitizer legs** (ASan/UBSan leak checks and TSan on the tier's shared `Program` built under `std::call_once`; those
are CI-only and were not run here: NOT MEASURED).

**Memory (peak RSS, tier on against off, 280 matched pairs from all of the above):** median difference 0.0 MB, mean +1.3 MB; largest
absolute +57 MB (full `nn`, 1805 to 1862 MB, +3%), then +33 MB (`ops`, +1%), +32 MB (HeCBench `fwt-cuda`, 137 to 168 MB, +23%), +27 MB
(`UnifiedMemoryStreams`, +22%), +24 MB (CuTe `tma_load`, +8%); the largest relative increase is +37% on a 21 MB program (`SYRK`, 21 to 28
MB). The 96 bytes per instruction of a MOp for a whole code section is therefore small next to the process in every case measured; the
heaviest process in the soak (`car-cuda`) is 12.8 GB either way.

### G. Which instructions fall back to the interpreter (order for plan item #4)

`VGPU_TIER_STATS` of 238 runs (12 torch groups, 10 CUTLASS SASS cases, 10 pantheon workloads, 206 external-suite programs that ran a SASS kernel): 64.5 G warp
instructions, **97.8% in the tier overall** (pantheon is 13 G of that). The statistics list the 24 commonest of each kind per process;
the lists cover 99.8% of the interpreter's instructions. "ends a run" means the interpreter's instruction also ends the tier's chain of
straight instructions (so it costs a return to the scheduler), "straight" means it runs inside a run.

PyTorch only (12 groups, 73.9 M warp instructions, **88.6% in the tier**; the interpreter's 11.4% below). Base opcode first:

| rank | mnemonic (the listed forms) | warp instr | share of the sweep | main groups |
| ---: | --- | ---: | ---: | --- |
| 1 | `LDCU` (.64 1.62 M, plain 1.38 M, .128 0.11 M) | 3.17 M | 4.3% | sdpa, ops, optim, attention |
| 2 | `IADD.64` | 1.48 M | 2.0% | sdpa, optim, ops |
| 3 | `ISETP.{GE,NE,GT,LT,EQ}.{S64,U64}.*` | 0.70 M | 0.95% | optim, ops, attention |
| 4 | `S2UR` | 0.49 M | 0.66% | ops, attention, sdpa |
| 5 | `LDSM.16.{M88.4,MT88.4,M88.2}` | 0.32 M | 0.43% | sdpa, attention |
| 6 | `ATOMS.POPC.INC.32` | 0.17 M | 0.23% | ops (radix sort) |
| 7 | `MOV.64` | 0.19 M | 0.26% | optim, ops |
| 8 | `UIADD3.64` | 0.14 M | 0.18% | ops |
| 9 | `LDGSTS.E.BYPASS.LTC128B.128` | 0.13 M | 0.17% | attention, sdpa |
| 10 | `WARPSYNC.ALL` | 0.10 M | 0.13% | ops, optim |

CUTLASS (the 10 SASS cases, 1.08 G warp instructions, **85.1% in the tier**, 14.9% in the interpreter): `VIADD` 28.0 M (2.6% of the CUTLASS
instructions; 26% each from sm90 sparse, ping-pong and group GEMM), `R2UR` 14.8 M (sm100 f16 55%), `SYNCS.PHASECHK.TRANS64.TRYWAIT` 10.1 M,
`ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU` 9.8 M (sm100 f16 only), `WARPGROUP.DEPBAR.LE` 9.0 M, `WARPGROUP.ARRIVE` 9.0 M,
`HGMMA.64x128x16.F32` 8.3 M, `LDCU` 7.1 M (sm100 f16 91%), `SYNCS.ARRIVE.TRANS64.RED.A1T0` 4.8 M, `UIMAD.WIDE.U32` 3.9 M, `LD.E`/`ST.E`
3.3 M each (ping-pong), `S2UR` 3.0 M, `WARPSYNC.ALL` 2.7 M. `VIADD` is a plain integer add: the cheapest single item on the list, and 2.6% of the
CUTLASS warp instructions.

Everything (238 runs, summed; dominated by single programs): `F2F.F64.F32` 140 M (CUDA Samples `reductionMultiBlockCG` 75%),
`RET.REL.NODEC` 129 M and `CALL.ABS.NOINC` 103 M and `FCHK` 104 M (`simplePitchLinearTexture`, `HSOpticalFlow`,
`NV12toBGRandResize`: device-function calls and division checks), `TEX.SCR.LL` 99 M (texture samples; `convolutionTexture`), `LD.E.64`
92 M (HeCBench `fresnel-cuda`), `ATOMS.POPC.INC.32` 67 M (`histogram`, `radixSortThrust`), `P2R` 65 M, `CALL.REL.NOINC` 56 M, `BRA.DIV` 46 M,
`BRA.CONV` 44 M, `IMMA.16816.U8.U8` 34 M. The macro average (each run counts the same; its first place, `IMMA.16816.U8.U8`, is one program) puts `S2UR` (37 runs above 0.1%), `LDCU`
and `LDCU.64`, `RET.REL.NODEC` (40 runs), `CALL.REL.NOINC` (39), `FCHK` (33), `BRA.CONV` (21), `CALL.ABS.NOINC` (21) and `IADD.64` first. Full tables with
sources: `data/round10/measure/fallback-rank-*.txt`, per-run statistics in `stats-all.txt`.

So the order by how many workloads each helps: (1) the constant-bank / uniform-register moves `LDCU*`, `S2UR`, `R2UR` and `IADD.64`, `VIADD` (in
the PyTorch sweep and every CUTLASS run); (2) CALL/RET + `FCHK` (device functions: CUDA Samples and sfu_stress); (3) the 64-bit
`ISETP.*.S64/U64` and `MOV.64` forms (optim/ops); (4) `LDSM`, `LDGSTS`, `ATOMS.POPC.INC`, `WARPGROUP.*`, `SYNCS.*` (shared-memory
and Hopper/Blackwell async); (5) `TEX`, `F2F.F64.F32`, `BRA.DIV/.CONV`. **But the whole interpreter fraction is 11% of PyTorch's warp
instructions, and warp instructions are under 10% of a PyTorch process's time (section D), so none of this moves the sweep's wall time
measurably.** It would matter for pantheon-style stress kernels and HeCBench-style compute programs.

### H. Answers

* **Is the tier safe to make the default?** On everything measured, yes as far as results go: 427 pairs of tier-off and tier-on runs
  (the full and quick PyTorch sweeps, 14 CUTLASS/CuTe/FMHA cases, 253 external programs; counters on, 1 and 4 host threads) gave the same stdout,
  the same exit status and the same counters, and every difference traced to a timing line, a race in the program, the 300 s
  budget, PyTorch's exit abort or my own parallel runs filling the host's memory. No tier bug was found. Cost: RSS +1.3 MB on average
  and +57 MB at worst; the process does not get slower. **Not done: the sanitizer legs (CI-only).** Make it the default only
  after the nightly ASan/UBSan and TSan jobs have passed with `VGPU_TIER=1` once.
* **Is it worth it?** For the sweep, no (0 to 7% per group, 2.5% over the whole sweep, 0 to 4% on CUTLASS GEMMs). For pantheon stress
  workloads (1.5x to 1.9x), HeCBench (0.64 of the wall) and CuTe (1.3x), yes. It costs nothing to leave on.
* **What remains for ML work:** the tier is not where the time is. In order: (1) `unsupported()` at fatbin registration (65 to 75% of
  a heavy PyTorch group, 22 points of it one `snprintf`), (2) zstd of the fatbins (5 to 10%), (3) tensor-core arithmetic in CUTLASS
  (`mma_narrow_sum`, `exec_gmma`, `tcgen05_mma`: 24 to 90% of a GEMM run), (4) `FFMA2`/`FMUL2`/`FADD2`/`FMNMX3` so that Blackwell FMHA
  runs SASS at all, (5) the fallback list above.
