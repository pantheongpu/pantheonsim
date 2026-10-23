# Study 7: can the host's real GPU make the simulation faster? (round 12, a measurement)

Status: measured 2026-10-23 on `host-gpu-proto` (= origin/main `480d412f`). **No product code was written**: the scripts, probes and raw data are in
[data/round12](data/round12). Nothing here is built into the simulator. The question (the user's): the simulator could use the real GPU of the machine it
runs on, on by default with an opt-out; "lets prototype and see if running on the real gpu can make the simulation faster; if there is any real benefit we
can go that route". The parent's prior, which this round tried to disprove with data: default-on changes results from machine to machine and drops the
simulator's diagnostics, and a simulated H100 cannot be computed by an A10G, so at most an opt-in mode with an equivalence guarantee.

## 0. The result in ten lines

1. **The ceiling is enormous.** Same binary, same architecture (sm_86): the A10G runs the stress workloads' kernels at 17.8 TOPS (`int_virus`),
   140 TFLOPS (`mma_virus`) and 166 GB/s (`memory_bank_thrash`); the simulator, 16 threads, sustains 0.004 TOPS, 0.02 TFLOPS and 0.8 GB/s on the same
   program at the knobs it can finish: **122x to 26,000x** on the compute and memory workloads (section 2.2). On the 16 harness workloads' device phases the
   card is **8x to 105x** faster (section 2.1). A matrix multiply (cuBLAS `sgemm` 2048) takes 0.84 ms on the card and 210 ms in the simulator (250x), a
   dependent-FMA loop 1,000x (section 6).
2. **A whole process is usually NOT faster on the card.** Process wall time: the card's run is 1.3x to 50x *slower* than the simulator's on all 16 harness
   workloads at the harness's full size (CUDA context start-up alone is 0.15 s; the simulator's is 1.5 ms); on the device phase alone `launch` is 1.1x faster
   on the card, `api` 8x slower and `startup` 100x slower. The card wins only when there is enough kernel work to amortise its fixed costs (section 2.1, 6).
3. **Native and simulated outputs agree for most programs of the corpora, once timing lines are removed:** of 164 programs (Rodinia 12, PolyBench 20, HeCBench
   34, CUDA Samples 98) **131 are identical, 6 more differ only because the simulator timed out, and 27 differ or fail for reasons a user would need to
   know** (section 3). Of the 33 that are not identical, **11 are not reproducible on the card itself** (card vs card: 6 nondeterministic programs, 5 timing quantities the
   filter missed), 3 are SFU/fast-math rounding, 1 is a cuRAND stream, 1 an out-of-bounds read in the benchmark, 3 device-printf order, 1 more timing,
   3 reported identity (driver version, capability), 1 a capability gap in the simulator, 6 simulator timeouts, 3 failures common to both (section 3.2).
4. **A different simulated card is mostly NOT computable by the A10G.** A program built for sm_89, sm_90 or sm_120 runs on the A10G in **1 of 16** harness
   workloads (the one with no kernel); built for sm_80 (compute_80 PTX) it runs 16 of 16 and agrees with the simulated A100. Where a program (built for
   sm_86, or a multi-architecture fatbin) does run natively and on the simulated L4, H100 or RTX 5090, the outputs agree on 84% to 85% of the programs that ran
   on both (87% for the A10G's own profile), and the difference is device identity: an H100's 264-block, 1024-thread grid and 232 KB of shared
   memory would be refused by the real card (section 4, 5.2). That is agreement with the simulator's engine, not with a real H100 (no H100 here).
5. **What an integration would have to cover is large and measured:** the 164 programs call 112 distinct CUDA runtime functions and 47 driver functions
   (section 5.1); the simulator's device pointers, streams (synchronous), events (host clock), graphs, IPC (files) and managed memory are its own (section 5.3).
   One thing is better than feared: the real device accepted a reservation at the simulator's own addresses (0x2000'0000'0000...) in 84 to 100% of tries, so
   pointer values *can* be preserved with the VMM API (section 6.3).
6. **What is lost on a native launch** is everything the simulator is for: counters, fault injection, OOB / use-after-free / race reports, deterministic
   scheduling, CUPTI records, the NVML/telemetry model, the profile's identity (section 5.4).
7. **Library offload does not preserve bits:** cuBLAS on the card and the simulator's cuBLAS gave 8 different hashes of 8 products, though the
   inputs were equal (section 6.2).
8. **The verdicts (section 7):** (1) card in the machine, faster: measured 100x to 26,000x on kernel-bound work, but then the simulator is redundant for the
   answer and only worth it as an explicit per-launch opt-in; (2) a different card: **no**; (3) hybrid rack: no, research only; (4) validation against
   hardware (oracle mode): **yes, and it is cheap**, as a separate comparison tool, which is what this round's scripts already are; (5) library offload: no.
9. **Default-on is contradicted by the data** in three ways: results differ per machine (section 3), the default profile is an H100 (`gpu` default in
   `docs/config.md`) which no machine without an H100 can compute, and the diagnostics vanish.
10. Not measured: any AMD GPU (none available; the only ones are on another machine without ROCm), multi-GPU, anything on the WSL machine, A10G power or
    thermals, a built passthrough (section 8 lists the rest).

## 1. Rig and method

* g5.4xlarge, 16 vCPU (AMD EPYC 7R32), 62 GB, one NVIDIA A10G (sm_86, 80 SMs, 22.1 GiB, driver 595.91), host CUDA 13.2, Ubuntu 24.04. Nothing else ran
  on it except my own jobs (load is stated where it matters). The simulator is `~/hg/build` (Release, CI image CUDA 13.0 / gcc 12) run on the host.
* **Same program binary on both sides.** Programs are built with a shared cudart (`-cudart shared`, sm_86 SASS plus compute_86 PTX). Simulator side:
  `LD_LIBRARY_PATH=build/shim VGPU_QUIET=1 VGPU_GPU=nvidia/a10g`. Native side: `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64` (the host's real libcudart,
  libcuda, cuBLAS). PTX-JIT side: native plus `CUDA_FORCE_PTX_JIT=1 CUDA_CACHE_DISABLE=1`. [common.py](data/round12/common.py) is the whole launcher.
* Timing runs are serial and nothing else ran on the rig during them (E1, E5). The E2/E3 corpus passes ran in parallel (functional comparison only, load 15 to
  26), with `VGPU_THREADS=3` or `5` and timeouts of 300 to 1500 s, which is why some heavy programs show as simulator timeouts.
* Comparison rules ([compare.py](data/round12/compare.py)): stdout+stderr and the new or changed files of the working directory (sha256) are compared.
  Lines carrying a time or throughput word, a number followed by `s`, a pointer or a pid, and a bare number that follows such a line (PolyBench prints
  `GPU Time in seconds:` and the number on the next line) are dropped. For E3 (`IDENT=1`) lines that only state which device it is (name, capability, SM count,
  driver version) are dropped too, and counted in neither direction. **Anything printed on a dropped line is NOT compared.** Classes: identical (bytes), timing-only
  (identical after the filter), numeric-close (same text, every number within 1e-4 relative), differs, one side fails.
* Run-to-run noise is separated from divergence by running the card twice (`native1`, `native2`) and the simulator twice (`sim1`, then `sim2` for the 38
  programs that differed or timed out, with a longer timeout, and `sim3` for the other 126).

## 2. E1: the ceiling, same architecture (sm_86 program, `nvidia/a10g` profile vs the real A10G)

### 2.1 The 16 harness workloads (tools/bench, `full` size, SASS and PTX engines)

7 native runs and 5 simulator runs each, median [min..max]; the simulator at its defaults (default tier, `VGPU_THREADS` = 16); `device phase` is the
program's own `BENCH device_wall_s`, `process` the whole `time`. Every run of every workload passed its bit-exact host reference on both sides
(**PASS 7/7 native, 5/5 simulated, for all 16**). Raw: [raw/e1_harness.jsonl](data/round12/raw/e1_harness.jsonl).

| workload | native device phase (SASS) | sim device phase (SASS) | native faster by | native process | sim process | PASS native / sim | native PTX-JIT device phase (cold) | sim PTX device phase |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| int_alu | 0.551 ms [0.544..0.571] | 39.7 ms [28.4..43.8] | 72x | 269 ms | 63 ms | 7/7 / 5/5 | 152 ms | 72 ms |
| f32_fma | 0.488 ms [0.468..0.491] | 18.9 ms [14.7..23.6] | 38.6x | 280 ms | 52 ms | 7/7 / 5/5 | 152 ms | 42.6 ms |
| f64_fma | 0.837 ms [0.821..0.841] | 16.5 ms [13.4..19] | 19.7x | 269 ms | 33 ms | 7/7 / 5/5 | 152 ms | 38.3 ms |
| f64_subnormal | 0.93 ms [0.908..0.936] | 10.1 ms [9.31..11.6] | 10.8x | 254 ms | 16 ms | 7/7 / 5/5 | 152 ms | 32.5 ms |
| wmma | 0.563 ms [0.559..0.574] | 58.2 ms [58..58.4] | 103x | 259 ms | 66 ms | 7/7 / 5/5 | 152 ms | 71 ms |
| mma_ptx | 0.493 ms [0.481..0.497] | 30.8 ms [30.5..31] | 62.4x | 254 ms | 36 ms | 7/7 / 5/5 | 152 ms | 59.5 ms |
| mem | 8.55 ms [8.37..8.82] | 71.9 ms [67.9..72] | 8.4x | 274 ms | 89 ms | 7/7 / 5/5 | 158 ms | 93.4 ms |
| shared | 0.82 ms [0.797..0.858] | 19.7 ms [18.4..23.5] | 24x | 254 ms | 28 ms | 7/7 / 5/5 | 152 ms | 49.3 ms |
| atomics | 4.27 ms [4.25..4.28] | 179 ms [176..183] | 42x | 254 ms | 189 ms | 7/7 / 5/5 | 155 ms | 176 ms |
| texture | 0.441 ms [0.436..0.46] | 38.7 ms [34.3..45.8] | 87.7x | 254 ms | 47 ms | 7/7 / 5/5 | 151 ms | 56 ms |
| local | 0.966 ms [0.943..1.01] | 42.4 ms [40.8..53.3] | 43.9x | 249 ms | 54 ms | 7/7 / 5/5 | 152 ms | 122 ms |
| divergent | 0.65 ms [0.633..0.665] | 68.3 ms [67.1..75.6] | 105x | 254 ms | 83 ms | 7/7 / 5/5 | 152 ms | 115 ms |
| warp | 0.546 ms [0.539..0.573] | 22.3 ms [20..27.8] | 40.9x | 259 ms | 36 ms | 7/7 / 5/5 | 152 ms | 53.5 ms |
| launch | 125 ms [125..126] | 141 ms [141..141] | 1.13x | 376 ms | 148 ms | 7/7 / 5/5 | 276 ms | 215 ms |
| api | 2.46e+03 ms [2.46e+03..2.51e+03] | 301 ms [298..302] | 0.122x | 2716 ms | 305 ms | 7/7 / 5/5 | 2469 ms | 300 ms |
| startup | 150 ms [149..155] | 1.55 ms [1.53..1.62] | 0.0103x | 254 ms | 5 ms | 7/7 / 5/5 | 300 ms | 21 ms |

Reading it:

* **Device phase, SASS engine: native is faster by 8.4x (`mem`) to 105x (`divergent`) on the 13 kernel workloads.** These `full` sizes are small (10 to 180 ms
  simulated, 0.4 to 8.6 ms native); a bigger launch widens the gap (2.2 and section 6).
* **Whole process: the card is slower than the simulator on every one of the 16.** The card's process costs 250 ms even for a 0.5 ms kernel: CUDA context
  creation (the `startup` workload's device phase is 150 ms native, 1.5 ms simulated) plus the program's own host-side reference check, which both sides pay.
  The simulator's process is 5 to 190 ms.
* **Where native is not faster:** `launch` (40,000 one-warp launches, each synchronised: 1.13x on the device phase), `api` (cudaMalloc/Memcpy/Free in a loop:
  the card is **8x slower**, 2.46 s vs 0.30 s), `startup` (0.01x).
* **Native PTX goes through the driver's JIT**, which compiles the whole module at first use: 150 ms cold (`CUDA_CACHE_DISABLE=1`), and it was not faster
  with the JIT cache enabled and warm (220 ms median under load from the concurrent simulator runs, [raw/e1_ptxwarm.jsonl](data/round12/raw/e1_ptxwarm.jsonl);
  the cache did not help `CUDA_FORCE_PTX_JIT`). So against the simulator's PTX engine the card is **slower** on 12 of the 13 kernel workloads (0.21x to 0.81x of the simulator's speed; `atomics` is 1.13x faster).
* Simulator `VGPU_JIT=1` (the tier-2 JIT) on the four ALU workloads (SASS engine, device phase): 0.84x to 1.08x of the tier alone (at these sizes the compile does not pay).

### 2.2 The pantheon stress workloads (duration driven: compare WORK per second)

`int_virus`, `compute_virus`, `fp64_virus`, `mma_virus`, `pulse_virus`, `omni_virus`, `memory_bank_thrash` (prebuilt sm_86 binaries, `-cudart shared`, from
`pantheon/build-vgpu-86`, built 2026-09-14/25 from an older pantheon). 15 s per run, 3 runs per cell, `--verify` on (the golden-pass check runs outside the
timed phase), the program's own `Throughput:` line. Two launch geometries: *harness knobs* (`--kernel_loops 2 --grid_size 4`, 1 GB VRAM cap: what the
simulator can finish, the repo's own runner) on both sides, and the *program's defaults* (auto grid, default loops) natively only. Median [min..max].
Raw: [raw/e1_pantheon.jsonl](data/round12/raw/e1_pantheon.jsonl).

| workload | unit | native, harness knobs | native, program defaults | sim sass | sim ptx | sim sass+JIT | verify (all runs) |
|---|---|---|---|---|---|---|---|
| int_virus | TOPS | 0.005792 [0.00533..0.00585] | 17.77 [17.7..17.8] | 0.004195 [0.00418..0.0042] | 0.00259 [0.00255..0.00259] | 0.008923 [0.00757..0.00906] | PASS x15 |
| compute_virus | TFLOPS | 0.0001641 [0.000152..0.000172] | 9.221 [9.21..9.26] | 0.0003483 [0.000329..0.000349] | 0.0002095 [0.000206..0.000211] | 0.0004006 [0.000381..0.000419] | PASS x15 |
| fp64_virus | TFLOPS | 0.00778 [0.00756..0.0082] | 0.5406 [0.541..0.541] | 0.004432 [0.00437..0.00447] | 0.003253 [0.00324..0.00336] | 0.004458 [0.00443..0.00447] | PASS x15 |
| mma_virus | TFLOPS | 0.3336 [0.287..0.374] | 139.6 [140..140] | 0.02097 [0.0206..0.0213] | 0.01231 [0.0122..0.0124] | - | PASS x12 |
| pulse_virus | TFLOPS | 7.841e-06 [7.84e-06..7.84e-06] | 6.435 [6.43..6.44] | 7.83e-06 [7.83e-06..7.83e-06] | 7.824e-06 [7.82e-06..7.83e-06] | - | PASS x12 |
| omni_virus | TFLOPS | 0.02103 [0.0196..0.0242] | 24.7 [24.7..24.7] | 0.001058 [0.00105..0.00106] | 0.0002024 [0.000202..0.000204] | - | PASS x12 |
| memory_bank_thrash | GB/s | 0.7439 [0.572..0.808] | 166.5 [166..167] | 0.836 [0.828..0.844] | 0.5459 [0.537..0.565] | - | PASS x12 |

| workload | native(knobs)/sim sass | native(knobs)/sim ptx | native(defaults)/sim sass(knobs) | sim JIT/sim sass |
|---|---:|---:|---:|---:|
| int_virus | 1.38x | 2.24x | 4.24e+03x | 2.13x |
| compute_virus | 0.471x | 0.783x | 2.65e+04x | 1.15x |
| fp64_virus | 1.76x | 2.39x | 122x | 1.01x |
| mma_virus | 15.9x | 27.1x | 6.66e+03x | - |
| pulse_virus | 1x | 1x | 8.22e+05x | - |
| omni_virus | 19.9x | 104x | 2.34e+04x | - |
| memory_bank_thrash | 0.89x | 1.36x | 199x | - |

* **The verification passed in all 93 runs (native and simulated).**
* At the *harness knobs* the card is **not** much faster (0.47x to 1.76x on the ALU and bandwidth workloads; 16x `mma`, 20x `omni`): four blocks and two
  loops per launch make the card launch-bound, which is the same finding as `launch`/`api` above. `pulse_virus` sleeps most of each period: its rate is the
  same (1.0x) and says nothing.
* At the *program's defaults* the card sustains 17.8 TOPS / 9.2 TFLOPS (fp32 ALU) / 0.54 TFLOPS (fp64, 1:64 on the A10G) / 140 TFLOPS (fp16 tensor) / 24.7
  TFLOPS (omni) / 166 GB/s. **Native defaults / simulator knobs = 122x (`fp64_virus`) to 26,000x (`compute_virus`)**, 4,200x `int_virus`, 6,700x `mma_virus`,
  23,000x `omni_virus`, 200x `memory_bank_thrash`. This is a ceiling, not a like-for-like A/B: the two sides did different amounts of work per launch, and the
  simulator at the program's defaults would take hours per launch. The rate comparison is valid because both are sustained rates of the same kernels.
* Simulator JIT (`VGPU_JIT=1`, 15 s runs, the JIT cache warms during the three runs): 2.1x on `int_virus`, 1.15x `compute_virus`, 1.0x `fp64_virus`.

## 3. E2: can a native run stand in for the simulation? (same architecture, programs' own output)

164 programs from the four corpora, run natively twice and on the simulator (`nvidia/a10g`) once, then once more for every program that differed. Programs
and arguments are exactly those of the existing card/sim runners (Rodinia `run.sh`/`run2.sh`, PolyBench with no arguments, HeCBench `reduced.txt`, CUDA
Samples `exes.txt`), run from [corpus.py](data/round12/corpus.py). Full per-program table: [raw/e2_table_a10g.md](data/round12/raw/e2_table_a10g.md).

### 3.1 Totals

| corpus | programs | identical native vs simulated | differ or fail |
| --- | ---: | ---: | --- |
| Rodinia | 12 | 9 | 2 card-nondeterministic, 1 out-of-bounds read |
| PolyBench/GPU | 20 | 20 | none |
| HeCBench (reduced list) | 34 | 27 | 3 simulator timeouts, 2 SFU rounding, 2 card timing noise |
| CUDA Samples 13.0 | 98 | 75 | 23, see below |
| **total** | **164** | **131 (80%)** | **33** |

Of the 131 identical, 28 are byte-identical in stdout and files and 103 identical after removing timing lines. Many Samples print only `Test passed`
and some numbers: *identical* means identical in everything the program prints outside the dropped lines.

Native vs native (two runs on the same card): 150 of 164 identical (42 byte-identical), 11 differ, 3 fail; **so 11 programs cannot be compared reliably at
all** (the card disagrees with itself). Simulator vs simulator: the 126 programs that matched the card in run 1 were run again (`sim3`) and are identical again in all 126 (31 byte-identical, 95
after the timing filter); the 38 that did not match were run again with a 1500 s timeout (`sim2`): 12 identical, 7 differ (backprop, particlefilter,
fp16ScalarProduct, UnifiedMemoryStreams, cudaGraphsPerfScaling, simplePrintf, simpleStreams: the programs that are unstable on the card or print in an order that
depends on the host-thread schedule), 18 hit a timeout or exit code in one or both runs. So **the simulator is run-to-run stable on every program that is stable on the
card**, except device-printf order under several host threads (`simplePrintf`).

### 3.2 The 33 that are not identical, classified

| class | n | programs | evidence |
| --- | ---: | --- | --- |
| card nondeterministic (time-seeded, uninitialised, random choice) | 6 | backprop, particlefilter_float, fp16ScalarProduct, UnifiedMemoryStreams, cdpSimplePrint, ptxjit | card vs card differs |
| timing / measured quantities the filter missed | 6 | babelstream, base64e, cudaGraphsPerfScaling, graphMemoryFootprint, UnifiedMemoryPerf, simpleStreams | card vs card (5) or sim vs sim (1) differs |
| genuine arithmetic: SFU / fast-math (`ex2.approx`, `-use_fast_math`) | 3 | aobench (ao.ppm bytes), s3d (1e-8 relative), MC_SingleAsianOptionP (6th digit) | deterministic on both sides, differ between them; aobench and s3d match the documented SFU divergence (memory note hecbench-on-pantheonsim); the cause for MC_SingleAsianOptionP was not isolated |
| genuine: library stream | 1 | MC_EstimatePiP | host-API cuRAND (likely the generator, not isolated): 3.13632 (card) vs 3.13900 (sim), each stable run to run |
| program bug the card hides | 1 | srad_v2 | reads the row above its image; the simulator reports it (memory note: rodinia-on-pantheonsim) |
| device printf / assert record order | 3 | simpleAssert, simpleAssert_nvrtc, simplePrintf | same messages, different thread/block order |
| reported identity | 3 | graphConditionalNodes, graphMemoryNodes (driver 13.2 vs 13.0), systemWideAtomics (pageableMemoryAccess 1 vs 0) | text of the program's own output |
| capability gap in the simulator | 1 | streamOrderedAllocationIPC | simulator says "does not support Memory pool based IPC" and skips |
| simulator timeout (300 to 1500 s) | 6 | aligned-types, bh, binomial, dmmaTensorCoreGemm, globalToShmemAsyncCopy, simpleAttributes | the card finishes in seconds |
| fail on both | 3 | simpleP2P, streamOrderedAllocationP2P (need 2 GPUs), simpleMPI (needs mpirun) | rc 2, 2, 1 on both |

The table sums to the 33. **No program showed a difference that is an unexplained simulator arithmetic bug**: the arithmetic cases are the known SFU/library
divergences. The precision of that statement is limited by what the programs print (section 8).

### 3.3 What E2 means for "native stands in for the simulation"

* For about 4 in 5 programs and 100% of PolyBench, a native run produces the same program output. For the rest the differences are exactly the things a
  user of the simulator wants to see reported (OOB reads, SFU rounding, order-dependent printf) or the card's own noise.
* The simulator's value is therefore not in the output of well-behaved programs but in what it says about the others; a native stand-in gives the
  *card's* answer, which is a valid result but not the simulator's.

## 4. E3: can the A10G compute for a DIFFERENT simulated card through PTX?

### 4.1 What can run on the A10G at all (the 16 harness workloads, rebuilt for five architectures)

[bench.cu](../../tools/bench/kernels/bench.cu) compiled for `sm_80, 86, 89, 90, 120` (each with the SASS and the PTX of that architecture; the `sm_75` build
does not compile, `mma.m16n8k16` needs sm_80). Each program has a bit-exact host reference (PASS). Run natively on the A10G (embedded SASS, and
`CUDA_FORCE_PTX_JIT=1`) and on the simulator with the matching profile and with the sm_86 build on other profiles. `VGPU_SASS_LOG` says which engine ran.
Raw: [raw/e3_harness.jsonl](data/round12/raw/e3_harness.jsonl).

| build (SASS+PTX of) | native A10G, embedded SASS | native A10G, CUDA_FORCE_PTX_JIT=1 | first native failure |
|---|---|---|---|
| sm_80 | 16/16 PASS | 16/16 PASS |  |
| sm_86 | 16/16 PASS | 16/16 PASS |  |
| sm_89 | 1/16 PASS | 1/16 PASS | FAIL int_alu |
| sm_90 | 1/16 PASS | 1/16 PASS | FAIL int_alu |
| sm_120 | 1/16 PASS | 1/16 PASS | FAIL int_alu |

| program build | simulated profile | sim engine | sim PASS | native A10G (JIT of PTX / SASS) can run it? | workloads where sim fails |
|---|---|---|---|---|---|
| sm_80 | nvidia/a100 | PTX x1, SASS x15 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/a10g | PTX x1, SASS x15 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_89 | nvidia/l4 | PTX x1, SASS x15 | 16/16 | SASS 1/16, PTX-JIT 1/16 | - |
| sm_89 | nvidia/l40s | PTX x1, SASS x15 | 16/16 | SASS 1/16, PTX-JIT 1/16 | - |
| sm_90 | nvidia/h100 | PTX x1, SASS x15 | 16/16 | SASS 1/16, PTX-JIT 1/16 | - |
| sm_120 | nvidia/rtx5090 | PTX x1, SASS x15 | 16/16 | SASS 1/16, PTX-JIT 1/16 | - |
| sm_86 | nvidia/l4 | PTX x1, SASS x15 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/l40s | PTX x1, SASS x15 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/h100 | PTX x16 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/rtx5090 | PTX x16 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/a100 | PTX x16 | 1/16 | SASS 16/16, PTX-JIT 16/16 | int_alu:FAIL int_alu; f32_fma:FAIL f32_fma; f64_fma:FAIL f64_fma; f64_subnormal:FAIL f64_subnormal; wmma:FAIL wmma; mma_ptx:FAIL mma_ptx; mem:FAIL mem; shared:FAIL shared; atomics:FAIL atomics; textur |
| sm_86 | nvidia/t4 | PTX x16 | 1/16 | SASS 16/16, PTX-JIT 16/16 | int_alu:FAIL int_alu; f32_fma:FAIL f32_fma; f64_fma:FAIL f64_fma; f64_subnormal:FAIL f64_subnormal; wmma:FAIL wmma; mma_ptx:FAIL mma_ptx; mem:FAIL mem; shared:FAIL shared; atomics:FAIL atomics; textur |

* The A10G runs **16 of 16** for builds up to sm_86 (sm_80 PTX or SASS, sm_86) and **1 of 16** for sm_89, sm_90 and sm_120 builds (the one passing workload,
  `api`, launches no kernel): `no kernel image is available`, because PTX of a later architecture cannot be JIT-compiled for an earlier device and the SASS
  is for another chip. This is the common case of a user simulating a newer card: **programs built for the card being simulated cannot run on the host card.**
* The sm_86 build (PTX plus sm_86 SASS) runs everywhere it should: the simulated L4 and L40S run the sm_86 SASS (cubin compatibility within sm_8x, as
  hardware), the H100 and RTX 5090 run its PTX, all 16/16 PASS, and the A10G passes the same 16. The A100 and T4 profiles refuse it (1 of 16 passes, the workload with no kernel; `cudaErrorInvalidPtx`,
  as real hardware would for compute_86 PTX on sm_80 and sm_75): the simulator is faithful there, and the A10G cannot stand in for them with this build.
* The sm_80 build is the legitimate way to simulate an A100 through the A10G: native A10G JIT of compute_80 PTX and the simulator's A100 SASS both pass 16/16.
  A passing PASS here means both sides equalled the host reference bit for bit, so they equal each other on these 16 programs.

### 4.2 The corpora, native PTX-JIT on the A10G vs the simulator on other profiles

Rodinia, PolyBench and HeCBench are sm_86 builds (sm_86 SASS plus compute_86 PTX). The CUDA Samples are multi-architecture fatbins (SASS for sm_75, 80,
86, 87, 89, 90, 100, 110, 120; checked with `strings`, and `VGPU_SASS_LOG=1` shows "running SASS: an sm_75 / sm_80 / sm_89 / sm_90 cubin" for `vectorAdd` on the
T4, A100, L4 and H100 profiles). So the two groups differ in what the simulator runs on another profile: the sm_86-only builds run through the sm_86 SASS (L4,
L40S: cubin compatibility within sm_8x) or through the PTX engine (H100, RTX 5090), or are refused (A100, T4); the Samples run the **profile's own SASS**.
The native side is `CUDA_FORCE_PTX_JIT=1` on the A10G in all cases (`ptxjit1`, 164 programs). `IDENT=1` removes the lines that state the device (name,
capability, SM count, driver version); anything else counts. 13 programs cannot run in PTX-JIT mode on the card at all (below), so the denominator of the last
column is the programs that ran on both sides. The first row is the reference: the A10G profile itself, which shows the noise floor of this comparison (card
nondeterminism, SFU rounding, cuRAND, the OOB read, printf order).

| simulated profile | group | engine the simulator ran | programs | identical | numeric-close | differs | native cannot run | simulator timeout | simulator fails | fail on both | identical / (ran on both) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a10g | sm_86-only builds | SASS sm_86 | 66 | 56 | 1 | 6 | 0 | 3 | 0 | 0 | 56 / 63 (89%) |
| a10g | CUDA Samples | SASS of the profile's own architecture | 98 | 66 | 1 | 10 | 13 | 4 | 1 | 3 | 66 / 77 (86%) |
| l4 | sm_86-only builds | SASS sm_86 (cubin-compatible with sm_89) | 66 | 50 | 1 | 5 | 0 | 10 | 0 | 0 | 50 / 56 (89%) |
| l4 | CUDA Samples | SASS sm_89 | 98 | 60 | 1 | 14 | 13 | 6 | 1 | 3 | 60 / 75 (80%) |
| h100 | sm_86-only builds | PTX compute_86 (JIT in the simulator) | 66 | 49 | 1 | 5 | 0 | 11 | 0 | 0 | 49 / 55 (89%) |
| h100 | CUDA Samples | SASS sm_90 | 98 | 58 | 1 | 13 | 12 | 9 | 1 | 4 | 58 / 72 (81%) |
| rtx5090 | sm_86-only builds | PTX compute_86 | 66 | 50 | 1 | 5 | 0 | 10 | 0 | 0 | 50 / 56 (89%) |
| rtx5090 | CUDA Samples | SASS sm_120 | 98 | 60 | 1 | 13 | 12 | 7 | 1 | 4 | 60 / 74 (81%) |
| a100 | sm_86-only builds | none: sm_86 SASS and compute_86 PTX are refused | 66 | n/a | n/a | n/a | 0 | n/a | n/a | n/a | **no kernel ran**: every launch fails with 'module targets sm_86 but this device is compute capability 8.0; PTX runs on newer architectures, not older ones'; the programs do not check the error, so 24 of 66 print output that matches or looks plausible (not a comparison) |
| a100 | CUDA Samples | SASS sm_80 | 98 | 61 | 1 | 13 | 13 | 6 | 1 | 3 | 61 / 75 (81%) |
| t4 | sm_86-only builds | none: refused | 66 | n/a | n/a | n/a | 0 | n/a | n/a | n/a | **no kernel ran**: every launch fails with 'module targets sm_86 but this device is compute capability 7.5; PTX runs on newer architectures, not older ones'; the programs do not check the error, so 24 of 66 print output that matches or looks plausible (not a comparison) |
| t4 | CUDA Samples | SASS sm_75 | 98 | 55 | 1 | 12 | 13 | 5 | 9 | 3 | 55 / 68 (81%) |

Programs that run on both and differ on the H100 profile (first differing line; the lists for the other profiles are in [raw/e3_corpora_summary.md](data/round12/raw/e3_corpora_summary.md)):

* **h100** (20)
  * `hecbench/aobench-cuda` differs: output files: ao.ppm
  * `hecbench/s3d-cuda` numeric-close: 
  * `rodinia/backprop` differs: output files: output.dat
  * `rodinia/gaussian` differs: -Total Global Memory			 - 23131776 KB | +Total Global Memory			 - 83036032 KB
  * `rodinia/particlefilter_float` differs: -XE: 26.303270 | -YE: 35.531096
  * `rodinia/srad_v2` differs: -1.69188 2.06711 1.87182 1.34890 1.12669 1.69617 2.51154 1.19081 1.58601 1.33976 1.57482 1.32192 2.00494 2.137 | -1.42858 1.79608 2.11331 1.75094 1.44
  * `samples/MC_SingleAsianOptionP` numeric-close: 
  * `samples/binaryPartitionCG` differs: -Launching 160 blocks with 768 threads... | +Launching 264 blocks with 1024 threads...
  * `samples/cudaCompressibleMemory` differs: -allocating non-compressible Z buffer
  * `samples/cudaGraphsPerfScaling` differs: -20, 1, 0, 22.421, 37.742, 10.861, 30.842, 3.620, 21.041, 17.408, 0.000, 16.384, 0.000, 9.770, 2.048, 1.000, | +20, 1, 0, 27.022, 26.061, 98.706, 99.1
  * `samples/fp16ScalarProduct` differs: -Result native operators	: 646406.000000 | -Result intrinsics	: 646406.000000
  * `samples/graphMemoryNodes` differs: -Check verified that d_negSquare and d_input share a virtual address. | -Check verified that d_negSquare and d_input share a virtual address.
  * `samples/ptxjit` differs: -ptxas info    : 0 bytes gmem | -ptxas info    : Function properties for myKernel
  * `samples/reductionMultiBlockCG` differs: -numThreads: 768 | -numBlocks: 80
  * `samples/simpleAWBarrier` differs: -Launching normVecByDotProductAWBarrier kernel with numBlocks = 160 blockSize = 768 | +Launching normVecByDotProductAWBarrier kernel with numBlocks = 
  * `samples/simpleAssert` differs: -/home/saqib/.cache/cuda-samples/Samples/0_Introduction/simpleAssert/simpleAssert.cu:64: void testKernel(int): | -/home/saqib/.cache/cuda-samples/Samp
  * `samples/simpleAssert_nvrtc` differs: -./simpleAssert_kernel.cu:38: void testKernel(int): block: [1,0,0], thread: [28,0,0] Assertion 'gtid < N' fail | -./simpleAssert_kernel.cu:38: void te
  * `samples/simpleOccupancy` differs: -Potential occupancy: 33.3333% | +Potential occupancy: 50%
  * `samples/simplePrintf` differs: +[2, 0]:		Value is:10 | -[1, 6]:		Value is:10
  * `samples/systemWideAtomics` differs: -CAN access pageable memory | +CANNOT access pageable memory

Reading it:

* **The profile adds little beyond the noise floor where the program runs:** the sm_86-only builds agree on 89% of the programs that ran on both for the A10G
  profile itself, the L4, the H100 and the RTX 5090 (the PTX engine gives the same agreement as the SASS engine); the Samples agree on 86% for the A10G
  profile and 80% to 81% for L4, H100, RTX 5090, A100 (their own SASS). The extra ~5 points are identity: launch geometry from the SM count
  (`binaryPartitionCG`: 160 blocks of 768 threads on the A10G, 116 on the L4, 264 of 1024 on the H100, 340 on the RTX 5090; `reductionMultiBlockCG`: 80, 58, 170
  blocks; `simpleAWBarrier` the same way), `simpleOccupancy` (33% vs 50%), `gaussian` printing the memory size, `cudaCompressibleMemory` (the capability differs).
* **That is agreement between the A10G and the simulator's engine for the other profile, not agreement with a real H100 or L4: NOT MEASURED** (no such card).
  It only says the simulator and the A10G do not disagree more on those profiles than on the A10G's own.
* **A100 and T4: the sm_86-only builds do not run at all** on the simulated card (`ptx-parse: module targets sm_86 but this device is compute capability 8.0; PTX
  runs on newer architectures, not older ones`), exactly as hardware refuses them. The programs do not check the launch error and print results that are
  wrong (PolyBench `GEMM`: 261,121 non-matching outputs; HeCBench `adam`: `FAIL`, checksum 0) or by accident plausible; the table does not count them. The
  Samples run their own sm_80 / sm_75 SASS and agree 81% for the programs that ran on both.

Why 13 programs do not run natively in PTX-JIT mode (`A-fails` against every profile): `MC_EstimatePiP/Q` (cuRAND's device code is SASS only: "Could not
generate pseudo-random numbers"), `UnifiedMemoryStreams` (cublasCreate fails: cuBLAS has no PTX for sm_86 JIT), the CUDA Dynamic Parallelism samples
`cdp*` (relocatable device code: `cudaErrorInvalidSource`), and the driver-API samples that load a prebuilt `.fatbin` without PTX (`matrixMulDrv`,
`vectorAddMMAP`, `threadMigration`, `simpleTextureDrv`, `vectorAddDrv`, `simpleDrvRuntime`: error 209). I.e. forcing PTX JIT breaks anything that depends on
NVIDIA's own libraries or on prebuilt images, which is a property of `CUDA_FORCE_PTX_JIT`, not of the A10G.

### 4.3 Programs a user simulating an H100 will mostly run: what the A10G does not have

A static census of this repository's own e2e programs (files in `nvidia/tests/e2e`, 250 of them `.cu`; a proxy for what the project thinks matters, not for what users
run), counting files that mention: `wgmma` in 10, TMA / `cp.async.bulk` / tensor maps in 7, thread-block clusters in 9, `tcgen05` in 6, `mbarrier` in 6, `setmaxnreg` in 2, `ldmatrix`/
`stmatrix` in 8, fp8 in 14, fp4/mx in 7, bf16 in 34. None of wgmma, TMA, clusters, tcgen05, setmaxnreg or fp8 MMA exists on sm_86 (clusters and TMA are
sm_90; tcgen05 is sm_100; the A10G reports `clusterLaunch = 0`, section 5.2), so **a program that uses them cannot run on the A10G in any mode**, and a
program that is only built for sm_90 is the 1-of-16 case above. The host-architecture-specific kernel selection that cuBLAS/cuDNN/CUTLASS do at run time
is also the host's: the real cuBLAS on this A10G would pick sm_86 kernels, not the H100 kernels whose arithmetic a user wants to see.

## 5. E4: what an integration would have to handle (a census, no code)

### 5.1 (a) The API surface actually used

`ltrace -f -c` on a native run of all 164 programs counts calls that cross the program's own PLT (`corpus.py ... census`; ltrace slows launch-heavy programs
but all 164 finished; one program, with a static runtime, shows nothing). Full list with call counts and number of programs using each:
[raw/e4_census.md](data/round12/raw/e4_census.md).

| family | distinct functions | calls | programs using it |
| --- | ---: | ---: | ---: |
| CUDA runtime API (cuda*) | 112 | 150,865 | 150 |
| compiler-generated glue (`__cudaLaunchKernel`, `__cudaRegister*`, push/pop call configuration) | 9 | 203,960 | 149 |
| CUDA driver API (cu*) | 47 | 1,110 | 16 |
| cuBLAS / cuBLASLt | 4 + 5 | 78 + 112 | 1 |
| NVRTC | 6 | 36 | 6 |
| cuRAND | 7 | 10 | 3 |

That is the size of a passthrough for these programs: ~170 entry points (runtime + driver + glue), each needing the simulator's argument translation, plus
the registration glue (`__cudaRegisterFatBinary` etc. is where the simulator finds and decodes kernels, so a passthrough replaces module loading too).
The most-called: `cudaDeviceSynchronize` 37,911 (92 programs), `cudaEventRecord` 20,380, `cudaGetLastError` 18,417, `cudaEventElapsedTime` 10,171,
`cudaMemcpy` 10,082 (115 programs), `cudaMalloc/Free` 3,293/3,481 (133/136 programs), `cudaGraphLaunch` 3,019 (6 programs). Not covered by this census:
the library calls made *inside* NVIDIA's libraries, and the PyTorch-style workloads (cuDNN, cuBLASLt, NCCL, cuSPARSE ... thousands of functions).

### 5.2 (b) Identity conflicts: what a program reads vs what the card would accept

What the simulator reports for each profile, run with [probe_props.cu](data/round12/probe_props.cu) (raw: [raw/e4_props.txt](data/round12/raw/e4_props.txt)),
and what happens when the program sizes a launch from it:

| profile | SMs | max threads/SM | shared mem opt-in per block | cluster launch | persistent grid (1024-thread blocks x occupancy x SMs) |
| --- | ---: | ---: | ---: | :---: | ---: |
| **real A10G** (and sim `a10g`) | 80 | 1536 | 101,376 | no | 80 |
| T4 (sm_75) | 40 | 1024 | 65,536 | no | 0 (sim: `InvalidPtx` for an sm_86 build) |
| A100 (sm_80) | 108 | 2048 | 166,912 | no | 0 (same) |
| L4 / L40S (sm_89) | 58 / 142 | 1536 | 101,376 | no | 58 / 142 |
| H100 (sm_90) | 132 | 2048 | 232,448 | **yes** | 264 |
| RTX 5090 (sm_120) | 170 | 1536 | 101,376 | yes | 170 |

* **Shared memory:** the probe asks for the opt-in maximum the profile reports. On the simulated H100 it is granted (232,448 B). Sent to the real A10G
  the same call returns `cudaErrorInvalidValue` (measured, the last block of raw/e4_props.txt). A program that sizes shared memory from the device (CUTLASS, FlashAttention
  style kernels) would be refused on the host card.
* **Grid sized from SM count:** `binaryPartitionCG` launches 160 blocks x 768 threads on the A10G and 264 x 1024 under the H100 profile;
  `reductionMultiBlockCG` and `simpleAWBarrier` size the same way; `simpleOccupancy` prints 33% occupancy on the A10G and 50% under the H100 profile
  (E3 table above). Run natively *as* the H100 profile such a program would launch 264 blocks of 1024 threads, more than the A10G keeps resident (80 SMs x 1 block of 1024
  threads, from the occupancy call in the table): a cooperative launch of that grid would be refused by the documented rules (NOT MEASURED; the shared-memory case above was).
* **Clusters:** the H100 and 5090 profiles report cluster launch, the A10G does not: `cudaLaunchKernelEx` with a cluster attribute is refused natively.
* **Capability and driver strings:** compute capability, `Ampere` vs `Hopper` banners, driver version (13.2 vs 13.0), `pageableMemoryAccess`, memory size
  (22.1 vs 79.2 GiB; `gaussian` prints "Total Global Memory"), L2 size, memory bus width: all differ, and programs print or branch on them.

### 5.3 (c) Memory, pointers and handles in a passthrough (from the code)

| thing | the simulator today | in a native passthrough | evidence |
| --- | --- | --- | --- |
| `cudaMalloc` pointers | fake virtual addresses `0x2000'0000'0000 + device x 1 TiB`, per-allocation bounds, 256 B aligned | must be real device memory. The pointer value can be kept: the real driver honoured a reservation at the simulator's own addresses (section 6.3), but not always (84 of 100) | `include/vgpu/memory.hpp:51-79,84-100`; probe_va |
| size of VRAM | sparse: 64 KiB chunks materialise on first write, so a 141 GB device runs on a small host; optional spill to disk | physical memory is real: allocation beyond 22 GiB fails; a program that allocates 100 GB and touches 1 GB works only simulated | `include/vgpu/memory.hpp:6,79`; `docs/config.md` (`ram_mb`, `memory_dir`) |
| pointers stored inside device memory (arrays of pointers, linked lists) | simulated addresses, valid because every address is the simulator's | valid only if the real address equals the simulated one (VMM at fixed address) or every such pointer is rewritten, which cannot be done in general | measured: probe_va |
| `cudaMallocManaged`, `__managed__` | host memory at its real host address, accepted by `owns()` | real UVM; coherent with the host, advice and prefetch are the driver's | `memory.hpp:95-100`; `runtime_api.cpp:312-316,754-779` |
| pinned / registered host memory | host ranges kernels may address | direct, but the register/unregister bookkeeping moves to the driver | `runtime_api.cpp:363-368` |
| IPC handles | a file in the share directory plus a payload (pid, size, device) | real `cudaIpc*` works only between two native processes; a native process and a simulated one cannot exchange handles | `runtime_api.cpp:4659-4717`; E2: the simulator skips `streamOrderedAllocationIPC` |
| peer access, several devices | up to 16 virtual devices; peer path from the profile | one real GPU; every peer copy involving a virtual device bounces through the host | `runtime_api.cpp:2499-2577` |
| streams | **one synchronous engine**: a launch returns when the kernel has finished, one recursive mutex | truly asynchronous; a passthrough either keeps the synchronous semantics (a sync after every launch, giving up the overlap) or changes which racy programs fail | `docs/perf-design/streams.md`; `runtime_api.cpp` ~885, ~3323 |
| events | complete when recorded, time the host clock between them | hardware timestamps; `cudaEventElapsedTime` becomes real GPU time | `docs/perf-design/streams.md` |
| graphs | the simulator's graph engine (capture, instantiate, launch) | real graphs; handles, memory-alloc nodes (E2: `graphMemoryNodes` checks that two buffers share a virtual address) need translation | `nvidia/src/driver_graph.cpp:1-24` |
| VMM (`cuMemCreate`/`Map`) | 2 MiB granularity, access states modelled; `cuMemAddressReserve` with a fixed hint returns `CUDA_ERROR_NOT_SUPPORTED` (measured) | real, same granularity (2 MiB, measured) | `memory.hpp:336-360`; e5_driver |
| kernels, function pointers | kernel and function addresses in their own windows `0x6ffb'...` | the driver chooses; the same window address was not available (measured) | `memory.hpp:56-67`; probe_va |
| textures, surfaces, arrays, memory pools, device `malloc`/`printf` | simulator objects | driver objects | `runtime_api.cpp:342`, `memory.hpp:129-142` |
| module loading | the simulator reads the fatbin, picks SASS if the architecture matches else PTX | `cuModuleLoadData` (70 us measured) and the driver's own rule, which JIT-compiles PTX | `docs/performance.md`; e5_driver |

### 5.4 (d) What a user loses when a launch runs natively

| feature | where | native launch |
| --- | --- | --- |
| exact counters (`VGPU_COUNTERS`, `vgpu ncu` metrics) | `docs/diagnostics.md`, `docs/config.md` (`counters`), `docs/cli.md` (vgpu ncu) | gone; the card has hardware counters (CUPTI/ncu) but a different set with no exact-instruction-count guarantee |
| fault injection (ECC, bit flips on load/store/shared/alu/copy, stuck bits, hang, link, throttle) | `docs/cli.md` (vgpu fault), `src/core/faults.cpp` | impossible: the card cannot be told to flip a bit |
| out-of-bounds / use-after-free / double-free diagnostics | per-allocation bounds, quarantine of 4,096 freed allocations (`include/vgpu/memory.hpp:82`), `docs/diagnostics.md` | an illegal address kills the context (Xid 13/31) or silently reads a neighbour: E2 `srad_v2` reads out of bounds without a fault on the card |
| race detection | `VGPU_RACE` (`docs/config.md`) | gone (compute-sanitizer is a separate tool) |
| deterministic scheduling, random/adversarial schedules | `VGPU_SCHEDULER`, `VGPU_SCHEDULER_SEED` (`docs/config.md:127-128`) | the hardware's order, nondeterministic (E2: 11 programs differ card vs card) |
| strict mode (division by zero, unwritten register), step budget against hangs | `VGPU_STRICT`, `VGPU_MAX_STEPS` | undefined behaviour proceeds; a hang is the watchdog's |
| CUPTI callbacks and activity records | `nvidia/src/cupti_api.cpp` | the card's own CUPTI, with timing, not the simulator's exact records |
| NVML / nvidia-smi / telemetry model, injected health states | `docs/telemetry.md` | simulated state does not change when work runs natively |
| debugger and instruction trace | `vgpu debug` is AMD only (`docs/cli.md`); `VGPU_SASS_TRACE` | no equivalent |
| machine independence | the same bits on any host | results depend on the card, driver and library versions (E2: driver 13.2 vs 13.0; section 6.2: cuBLAS bits) |
| a card that does not exist, a rack of 16, VRAM override | `gpu`, `count`, `vram_mb` (`docs/config.md`) | one A10G, 22 GiB |

## 6. E5: measured costs of a passthrough's pieces (throwaway probes outside the product tree)

Probes: [e5_launch.cu](data/round12/e5_launch.cu), [e5_driver.cu](data/round12/e5_driver.cu), [e5_gemm.cu](data/round12/e5_gemm.cu),
[probe_va.cu](data/round12/probe_va.cu). Each is one binary run on both sides. A passthrough layered on the simulator would pay at least the native column
plus its own translation (NOT MEASURED: no passthrough was built).

### 6.1 Per-operation cost, small launches and copies (A10G native vs simulator `a10g`)

| operation | native | simulator | note |
| --- | ---: | ---: | --- |
| empty one-warp kernel, launch + sync (runtime API) | 7.7 us | 2.5 us | sim 3x faster |
| same, launch only (pipelined) | 2.9 us | 2.4 us | |
| empty kernel via `cuLaunchKernel` + `cuCtxSynchronize` | 8.0 to 11.4 us | 3.5 us | two native runs |
| H2D + tiny kernel + D2H, 256 floats (1 KiB) | 17.0 us | 15.4 us | equal |
| same, 4,096 floats | 20 us | 131 to 148 us | native 7x faster |
| same, 65,536 floats (256 KiB) | 83 us | 480 to 500 us | 6x |
| same, 1M floats (4 MiB) | 890 us | 6.6 ms | 7.4x |
| same, 16M floats (64 MiB) | 12.1 ms | 99 ms | 8.2x |
| `cuMemAlloc` 1 MiB / `cuMemFree` | 55 to 82 / 51 to 67 us | 0.9 / 1.1 us | the card is 60x to 90x slower |
| reserve + create + map + set-access at a chosen address (2 MiB) | 92 to 130 us | not supported with a hint | |
| `cuModuleLoadData` + get function | 71 to 76 us | 290 us to 41 ms | the 41 ms run included the first-use set-up |
| 4 KiB `cuMemcpyHtoD` / `DtoH` | 7 to 8 / 10 to 13 us | 0.3 / 0.3 us | |
| 64 MiB `cuMemcpyHtoD` / `DtoH` | 5.3 / 7.1 ms | 6.5 / 6.5 ms | PCIe-bound vs memcpy-bound: equal |

**For small launches the real card is not faster and its per-call costs (launch latency 3 to 11 us, allocation 55 to 80 us) are higher than the
simulator's**: a workload dominated by tiny launches (the `launch` and `api` harness workloads) would slow down if its launches were
passed through. The card wins when a launch carries real work or large copies (4 KiB and up).

### 6.2 Compute scaling and library offload (the same binary, equal inputs)

Dependent-FMA loop, 1M threads (hash = FNV of the 1M results): native and simulated hashes are **identical at every size**.

| FMAs per thread | native (fma/s) | simulator (fma/s) | native faster |
| ---: | ---: | ---: | ---: |
| 1 | 8.4e10 (12 us per launch) | 1.7e8 (6.0 ms) | 480x |
| 10 | 8.4e11 | 1.4e9 (7.6 ms) | 610x |
| 100 | 5.6e12 | 7.3e9 (14.4 ms) | 770x |
| 1,000 | 1.25e13 | 1.26e10 (83 ms) | 990x |
| 10,000 | 1.42e13 (0.70 ms) | 1.37e10 (768 ms) | **1,040x** |

(Native times below 0.1 ms are derived from the rates the program prints.) `cublasSgemm` and `cublasGemmEx`
(fp16 in, fp32 accumulate), random inputs in [-1, 1):

| n | native sgemm | sim sgemm | native GemmEx f16 | sim GemmEx f16 | hashes equal? |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 0.030 ms | 1.8 ms | 0.031 ms | 2.0 ms | no (8 of 8 pairs differ) |
| 512 | 0.040 ms | 5.5 ms | 0.028 ms | 5.3 ms | no |
| 1024 | 0.168 ms | 35.7 ms | 0.066 ms | 34.0 ms | no |
| 2048 | 0.840 ms (20.4 TFLOPS) | 210 ms (81.6 GFLOPS) | 0.328 ms (52.4 TFLOPS) | 214 ms | no |

The simulator's cuBLAS is **60x (n=256) to 650x (n=2048, f16) slower** than NVIDIA's on the card, and its results are not bit-equal to NVIDIA's for random inputs (accumulation
order and, for f16, tensor-core accumulation): **offloading cuBLAS to the card would speed a 2048 GEMM up by 250x to 650x and change the bits of every GEMM.**

### 6.3 Can the real device hold the simulator's pointer values?

[probe_va.cu](data/round12/probe_va.cu), native only: `cuMemAddressReserve` with the simulator's addresses as hints, then map real memory, run a kernel through
the same pointer value and read it back (raw output: [raw/probe_va.txt](data/round12/raw/probe_va.txt)).

| hint (what the simulator would use) | result |
| --- | --- |
| device-0 window base `0x2000'0000'0000`, 64 MiB | exact; mapped; kernel wrote through it; host read back OK |
| device-1 window base `0x2100'0000'0000`, 64 MiB | exact; OK |
| device-0 + 3 GiB, 1 GiB | exact; OK |
| device-0 + 500 GiB, 8 GiB (an address deep in a big card) | exact (reserved, not mapped) |
| `0x7ffe'0000'0000` | exact; OK |
| `0x6ffb'0000'0000` (the simulator's function-pointer window) | **not honoured** (driver chose `0x7785...`) |
| 100 consecutive 2 MiB reservations from the window base (`e5_driver`) | 84 of 100 honoured |

So pointer values can usually be kept, which removes the worst translation problem (pointers inside device memory), but it is not guaranteed (the driver may
refuse a hint), costs 92 to 130 us per allocation, and the simulator's own `cuMemAddressReserve` does not yet accept a hint.

## 7. Verdict, per use case

Estimates of integration cost are mine, from reading the code and the census in section 5, not measured; "engineer-weeks" means one engineer who knows
the code base. All speed-ups are the measured ones above; nothing was built. I report what the data says and do not argue for a result.

### (1) Simulate the card that is in the machine, faster (A10G profile on an A10G host)

* **Speed-up:** device phase 8x to 105x on the harness (short runs); 122x to 26,000x on the stress workloads at the card's full launch size (a ceiling, section
  2.2); cuBLAS 60x to 650x; a dependent-FMA loop 1,040x. **No speed-up, or a slowdown, for whole short processes** (1.3x to 50x slower: context start-up 0.15 s),
  small launches (launch + sync 7.7 us vs 2.5 us), allocation (60x to 90x slower) and API-bound programs (`api` 8x slower): a workload with under about a
  second of simulated work gets slower.
* **Bit-identical:** 131 of 164 programs (80%; PolyBench 20/20, Rodinia 9/12, HeCBench 27/34, Samples 75/98). The 33 others are noise on the card itself (11) plus one more timing line (simpleStreams), SFU
  rounding (3), cuRAND (1), an out-of-bounds read the card hides (1), printf order (3), reported identity (3), simulator timeouts (6), simulator capability gap (1) and
  failures on both (3) (section 3.2: 11 + 1 + 3 + 1 + 1 + 3 + 3 + 6 + 1 + 3 = 33). The identical set is "what the program printed", not "every byte on the device".
* **Lost:** every diagnostic of section 5.4. The only thing a native run gives is the card's answer, which is the answer the user would get without the simulator.
* **Integration cost (estimate):** a launch-only prototype (VMM at the simulator's addresses, `cuModuleLoadData`, `cuLaunchKernel`, copies, events), enough to run
  the 16 harness workloads, 4 to 6 engineer-weeks; covering the 112 + 47 + 9 entry points seen in the corpora with the semantics of section 5.3 (synchronous
  streams, graphs, managed memory, error codes, IPC), 3 to 6 months; plus a GPU runner in CI (none exists; `docs/ci-card-check.md` is a proposal that was never
  enabled) and a permanent second code path whose equivalence must be tested.
* **Worth building: only as an explicit opt-in, never default.** Not a default (it changes results from machine to machine, section 3; it disables the
  simulator's reason to exist). If built: restricted to a profile that equals the host card (name, SM count, memory, capability), per-launch or per-process,
  refused with `counters`, `race`, `strict`, `scheduler`, `fault`, CUPTI, trace, or `count` > 1; kernel-heavy programs only (a launch must carry >~ 100 us of
  work to beat the simulator's 2.5 us launch). Honestly, a user who wants the card's answer fast can already run the program without the simulator; the
  option adds value only for one run that mixes native-speed phases with checked phases, and the data does not show that demand.

### (2) Simulate a different card, faster (e.g. the A10G computes for a simulated H100)

* **Speed-up:** the same large factors where it runs, but **it mostly does not run**: sm_89 / sm_90 / sm_120 builds pass on the A10G in 1 of 16 harness workloads
  (the one without a kernel); the sm_86 PTX build runs and passes on the simulated L4, L40S, H100 and RTX 5090 and on the A10G, but the A100 and T4 profiles
  refuse it (`InvalidPtx`) like hardware (section 4.1). On the corpora where it does run, the A10G's PTX-JIT output agrees with the simulator on 89% of the sm_86-only programs on the L4, H100 and RTX 5090 profiles (the same as for the A10G profile itself) and 80% to 81% of the Samples (86% for the A10G's own), the extra 5 points being launch geometry and reported identity; this is agreement with the simulator's engine for those profiles, not with a real H100 or L4 (NOT MEASURED).
* **Wrong card:** identity-dependent programs differ by construction (binaryPartitionCG, simpleOccupancy, reductionMultiBlockCG: launch geometry from the profile;
  232 KB of shared memory refused with `cudaErrorInvalidValue`; no clusters, section 5.2). Programs for the simulated card's own features (wgmma, TMA, clusters,
  tcgen05, fp8: section 4.3) cannot execute on sm_86 at all. Arithmetic specific to the simulated architecture (SFU rounding, tensor-core accumulation) would be
  the A10G's, so the output is not evidence about the simulated card.
* **Bit-identical:** see the E3 table (the fraction of programs that run on both and agree); the cuBLAS/cuRAND results it cannot reproduce are those of section 6.2.
* **Lost:** everything in 5.4, and the premise (a card that is not in the machine).
* **Integration cost (estimate):** the cost of (1) plus identity translation and a per-launch feasibility check; unbounded for programs that need features the
  card lacks.
* **Worth building: no.**

### (3) A hybrid rack (the real card as one device of several)

* **Speed-up:** the card's device phase for the launches that go to it (section 2); a rack of `count: 8` with one real device would run that one at card speed
  and the other seven at simulator speed, so a multi-device program is bounded by the slowest participant. NOT MEASURED: no multi-device program was run.
* **Bit-identical / lost:** per device as in (1); in addition peer copies and collectives between the real and a virtual device must bounce through the host
  (the host has one real GPU; peer access and NVLink topologies are the simulator's model, section 5.3), and the simulator's synchronous-stream ordering
  between devices no longer holds.
* **Integration cost (estimate):** (1) plus a cross-device memory path, 2 to 3 months more.
* **Worth building: no**, until someone asks for it: the one measured enabler (pointer-preserving VMM on the card, 84 to 100% of reservations honoured) is
  necessary but not enough, and no workload in the corpora needs it.

### (4) Validation against hardware (oracle mode): run the program on both and compare

* **Speed-up:** not applicable; it costs one native run (0.15 s of CUDA context start-up plus the program) per comparison.
* **What it found in this round, with zero simulator changes:** the card's own run-to-run noise (11 programs, which a comparison must first exclude), the
  deterministic simulator/card differences (SFU rounding in aobench, s3d, MC_SingleAsianOptionP; the cuRAND stream of MC_EstimatePiP), and a capability gap
  (`streamOrderedAllocationIPC`: the simulator reports no memory-pool IPC). It also separates a card-hidden bug (`srad_v2`) from a simulator bug.
  This is the existing "card check" idea (`docs/ci-card-check.md`) run by hand, and the PolyBench/HeCBench/Rodinia/Samples runners already exist for the RTX 3060.
* **Bit-identical:** it is the measurement of the bit-identity: 131 of 164.
* **Lost:** nothing; it does not replace the simulation, it audits it. It needs a real card of the same architecture (profile = card) to be meaningful; across
  architectures it compares different machines (section 4).
* **Integration cost (estimate):** 1 to 2 engineer-weeks for a `vgpu check ./program` or a script (run both ways, timing/identity filter, card-vs-card noise
  pass, classify), reusing [corpus.py](data/round12/corpus.py), [compare.py](data/round12/compare.py) and [classify_e2.py](data/round12/classify_e2.py); the cost of a GPU
  runner is the same one `ci-card-check.md` already priced.
* **Worth building: yes, as a separate tool / CI job outside the runtime** (no change to the simulator's launch path), same-architecture only.

### (5) Library offload (cuBLAS, cuDNN, cuFFT ... run by NVIDIA's libraries on the card)

* **Speed-up:** cuBLAS `sgemm` 60x (n=256) to 250x (n=2048), `GemmEx` f16 66x to 650x (section 6.2). The simulator's cuBLAS reaches 80 GFLOPS where the card does 20,000 to 52,000.
* **Bit-identical:** **no: 8 of 8 GEMM results differ** in bits from the simulator's for equal random inputs (accumulation order; for f16, tensor-core
  accumulation). The speed would buy different numbers, so a result from an offloaded GEMM is not the simulator's result and a program that compares or
  hashes outputs would change behaviour. The cuRAND stream also differs (MC_EstimatePiP).
* **Lost:** the counters and diagnostics of the kernels inside the library (the simulator counts them; NVIDIA's are opaque).
* **Integration cost (estimate):** high: the library's pointers must be real device pointers, so either all device memory is real (VMM at the simulator's
  addresses, (1)) or every allocation is mirrored with copies before and after each call. The copies are cheap next to the simulated GEMM (a 64 MiB copy is
  5 to 7 ms; the three 16 MiB matrices of a 2048 GEMM cost about 5 ms against 210 ms simulated), so the cost is complexity and bits, not time. Plus handle, workspace,
  descriptor and stream translation for every library function a workload calls (the repo implements cuBLAS, cuBLASLt, cuDNN, cuFFT, cuSPARSE, cuSOLVER and more as
  clean-room libraries in `nvidia/src/*_api.cpp`).
* **Worth building: no.** (If a user wants fast GEMMs and does not care about bits, they do not need the simulator.) The cheaper lever, which this round did
  not examine, is a faster CPU GEMM in the simulator.

### Default-on with an opt-out, as asked

On the data it fails on its own terms: (a) results change from machine to machine (driver 13.2 vs 13.0 and `pageableMemoryAccess` in program output; cuBLAS and cuRAND
bits; the card's own nondeterminism in 11 of 164 programs), (b) the default profile is an H100, which an A10G host cannot compute (sections 4, 5.2) and most hosts
have no GPU at all, and (c) it silently drops what the product is for (section 5.4). An opt-in that is explicit, per profile-equals-host-card and incompatible
with diagnostics is the only form with an equivalence argument, and (4) is the form with the clearest value for the cost.

## 8. What was NOT measured, and the limits of what was

* **AMD:** no AMD GPU was available (the only ones are on another machine without ROCm); nothing here says anything about HIP or the AMD half of the simulator.
* **No passthrough was built.** Section 6 gives the native costs of its pieces, not the cost of the integrated thing; translation overhead, locking against the
  simulator's single mutex, and the correctness of mixing simulated and real kernels over one address space are NOT MEASURED.
* One card (A10G), one driver (595.91), one host toolkit (13.2) vs the simulator's 13.0 runtime; other cards (a 3060, an H100) would shift the ratios, not the
  structure. The simulator ran with `VGPU_THREADS` = 16 for E1 and 3 to 5 for the corpora (load 15 to 26), so corpus timeouts are not simulator speed.
* E1's harness `full` size is small (simulated 10 to 180 ms); the stress workloads' native rate at the program's defaults is not a like-for-like A/B (different
  launch sizes), only a ceiling. The simulator was not run at the program's defaults.
* **E2's "identical" is "identical in what the programs print".** Many Samples print `Test passed` and a few numbers; lines with a time word, a number followed by `s`,
  pointers and pids were not compared; a difference confined to such lines would be missed. The filter was tuned on the card-vs-card pair (37 programs
  differed with the first filter, 14 after adding the number-next-to-a-time-label and number-followed-by-`s` rules, 11 after the pid rule), which is a form of fitting the
  filter to noise; the per-program table shows each decision.
* The corpora are sm_86 builds; builds for the simulated card's own architecture (the realistic case for "simulate an H100") were measured only for the 16 harness
  workloads (section 4.1), not for the corpora, because building the corpora was out of scope (no builds on the WSL machine, and the rig builds were limited to
  the harness).
* `ltrace` counts calls at the program's PLT, not the calls libraries make to each other; PyTorch-class workloads were not traced.
* The `CUDA_FORCE_PTX_JIT=1` mode breaks programs that use NVIDIA's own libraries or prebuilt images (13 programs), so "native PTX-JIT" is not a faithful stand-in for
  "an H100 build running on the A10G" for those programs.

## 9. Reproduce

All of it is in [data/round12](data/round12) (scripts) and [data/round12/raw](data/round12/raw) (outputs). On the rig host (`~/hg` = this branch, simulator built in
`~/hg/build`; programs from the corpora under `~/.cache/...`, bench binary built with `tools/bench/bench.sh --cache`):

```
build_probes.sh                                   # e5_launch, e5_driver, e5_gemm, probe_props, probe_va
e1_harness.py <bench_kernels> out.jsonl 7 5       # E1 harness; e1_table.py out.jsonl
e1_pantheon.py ~/corpora/pantheon-bin out.jsonl 15 3
corpus.py <dir> native|ptxjit|sim|census [--profile P --jobs J --threads T --timeout S --tag TAG --only a,b]
compare.py <dir> tagA tagB [IDENT=1]              # classes
classify_e2.py <dir> e2_notes.json > table.md
e3_harness.py <variant dir with bench_sm80..sm120> out.jsonl
census.py <dir>/census
```
