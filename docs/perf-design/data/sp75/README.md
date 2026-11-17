# Round sp75: a block that outlasts the wait (data)

The design is section 12 of [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md); the round's text is *Round sp75* in
[../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md). Tracking: pantheonsim #656, #657 (and the diagnoses of #659, #662).

Everything here was built and run on server3 (Ryzen 7 PRO 8840U, GCC 13.3, `Release`, `-Werror`, no LTO, `VGPU_THREADS=3`, profile
`nvidia/a10g`), against the stand-in driver (`nvidia/tests/e2e/host_gpu_stub_driver.c`). **server3 has no card: nothing in this directory is a
card result but [card.md](card.md), the card run of 2026-11-15 (one A10G).** Builds and runs held the machine's build lock, `nice -n 19`, `-j3`; another session used the machine at the same time, so wall
times here are to within a few percent at best. The corpus is the sm_86 binaries of `tools/bench/real/pack_corpora.sh` (nvcc 13.0). For the
stand-in test the compiler part of the dev machine's CUDA 13.0 toolkit was copied into the round's scratch directory, named with `VGPU_NVCC`,
and deleted with it (as in Round sp69).

| file | what |
|---|---|
| `census.py` | runs programs once per form in turbo mode with `VGPU_HOST_GPU_LOG=1` against the stand-in driver and keeps what the mode said: the summary, the `host_gpu_kept` reasons, the samples, the `host_gpu_nosample` lines; `--cut N` ends a run once N decisions are logged |
| (`census.py --weighted`, `--report`) | Round r9, #656: per kernel the plain static count beside the loop-weighted one (a loop's body weighed by its turns where the bound is an immediate), and the launches the floor keeps whose weighted estimate would reach it; changes no decision. Result: [../sp76/census-weighted.md](../sp76/census-weighted.md) |
| `table.py` | the tables of a census: programs whose decisions differ between two forms, samples whose copy did not answer, the small grids, wall times, reasons |
| `census.md` | `table.py` over the census of the corpus (124 programs): `old` (commit 6cf503d3) against `new` (this round), and the follow-up of 31 programs under one build with the switch off and on |
| `rerun.py`, `rerun.md` | wall times of the programs with a sampled launch, a changed decision or a small grid, the switch off against on, 9 runs a form interleaved (the stand-in driver: every launch simulated in both) |
| `launches.md` | the round's programs launch by launch: the reason each launch was kept under the old rule, and what the new rule decides |
| `mutants.py`, `mutants.txt` | mutants of the rule in the two headers, against `tests/unit/test_host_gpu_logic.cpp` |
| `e2e-mutants.sh`, `mutants.txt` | mutants of the engine's part loop, the copy's answer, the mode's plan and switch, against `nvidia/tests/e2e/run_host_gpu_stub.sh` |
| `same-output.py`, `same-output.md` | stdout of programs with the mode off, in exact mode and in turbo mode (stand-in driver), old build against new |
| `card-run.sh` | the card run: stages `build`, `decisions`, `changed`, `keep`, `small`, `rest` |
| `card.md` | **the card run, made on 2026-11-15**: what each stage measured on one A10G |

## Tests on server3

```
./build/test_host_gpu_logic                                   # 115/115 tests passed (109 before the round)
VGPU_NVCC=$PWD/cuda/bin/nvcc VGPU_BUILD_DIR=$PWD/build VGPU_HOST_GPU_TEST_PROFILE=nvidia/a10g bash src/nvidia/tests/e2e/run_host_gpu_stub.sh   # PASS, steps 1 to 40 (also with the default profile, nvidia/rtx3060)
python3 src/tools/bench/hostgpu/run.py --dry-run              # PASS: 0 checks failed (the forms exactb0 and turbob0)
python3 docs/perf-design/data/sp75/mutants.py --src src --work mut          # 42 of 42 killed (mutants.txt)
bash docs/perf-design/data/sp75/e2e-mutants.sh src mut-e2e cuda/bin/nvcc    # 6 of 6 killed (mutants.txt)
python3 docs/perf-design/data/sp75/same-output.py --corp corp --stub libstubcuda.so --old build-old --new build --programs ...   # 0 different (same-output.md)
```

## The card run (`card-run.sh`; one A10G; made on 2026-11-15: the results are in [card.md](card.md))

From the root of the tree on the rig, corpora in `~/corp`, build as in `../summary-2026-11-13/README.md` (for the stages after
`build`: `SP75_BUILD`). The table is what the round expected before the run:

```
docs/perf-design/data/sp75/card-run.sh all ~/sp75-out        # or stage by stage: build decisions changed keep small rest
```

| stage | programs | forms | what to read |
|---|---|---|---|
| `build` | card CI | | unit tests, `e2e_host_gpu`, `e2e_host_gpu_stub` steps 1 to 40 pass |
| `decisions` | cudaTensorCoreGemm, tf32TensorCoreGemm, bf16TensorCoreGemm, aidw-cuda, silu-cuda, s3d-cuda | one logged turbo run each | each GEMM: `sampled: ... (the last block stopped at the limit: at least) ... sent`, then `starts the driver: sampled` and `1 launches ran on the card`. aidw: `AIDW_Kernel` sent on its sample; `AIDW_Kernel_Tiled`'s first launch `host_gpu_nosample: ... reason=the driver has been started` (simulated: the run is cut at 120 s). silu-cuda and s3d-cuda (#662): silu's `starts the driver: size rule (... 0.25 s of eligible launches simulated so far ...)` and 20 launches on the card; `host_gpu_overhead:` of both (what deciding and scanning cost: s3d's is the open question, 27 ms on the rig against 2 ms here) |
| `changed` | the four GEMMs (immaTensorCoreGemm was not in the summary run); aidw | `sim,exact,exactb0,turbo,turbob0`; aidw `sim,turbo,turbob0` | turbo: tf32 208 s to a few seconds, bf16 159 s likewise, cudaTensorCoreGemm 16 s to about 1 s, immaTensorCoreGemm (about 3 s) to about 1 s (what is left is the programs' host code and the driver start), `1 / 1` launches on the card, output `identical` or `numeric`; `turbob0` at the times of the summary run (208, 159, 16 s). Exact mode: the GEMM kernels are not in the exact set (tensor and approximate instructions), so `exact` = `exactb0` = the simulator's time. aidw turbo: 365 s to about 180 s, `3 / 4` launches on the card; `turbob0` 365 s, `2 / 4` |
| `keep` | the keep list of the integration of sp64-sp68 and the other programs with a sampled launch | `sim,exact,exactb0,turbo,turbob0`, 3 runs | no program more than 3 % slower than its `b0` form; the same launches on the card; outputs as they were |
| `small` | lavaMD, s3d-cuda, lud, radixsort2-cuda, mergeSort, segmentationTreeThrust | `sim,turbo,turbob0`, 10 runs | the programs with a small grid and no followed loop that are NOT sampled (under 30 ms counted): turbo = turbob0 |
| `rest` | the rest of the corpus | `sim,turbo,turbob0`, 1 run | no program more than 3 % slower than `turbob0`; no zombie, no process left |

If a GEMM is not sent on the rig: its `host_gpu_sample:` line (stage `decisions`, `log-*.txt`) has `block_limit_s` (0: with the rig's host
threads a block would have to be watched for over 60 ms, `kSampleBlockCapSeconds`), `unfinished`, `estimate_s`, and the `sampled:` line the card
path the estimate was held against.

## What the mode costs silu and s3d (#662; server3)

| program | off s | exact s | turbo s | turbo, no sampling s | runs | `host_gpu_overhead` with the log on |
|---|---:|---:|---:|---:|---:|---|
| hecbench/silu-cuda | 0.2503 | 0.2551 | 0.2556 | 0.2589 | 15 | exact: `decide_s=0.0017 scan_s=0.0013` (5 kernels, 50 launches); turbo: `decide_s=0.0020 scan_s=0.0012` |
| hecbench/s3d-cuda | 0.2316 | 0.2336 | 0.2343 | 0.2308 | 15 | exact: `decide_s=0.0516 scan_s=0.0513` (54 kernels, 54 launches); turbo: `decide_s=0.0702 scan_s=0.0696` (the log's printing is in it) |

Medians of wall time, interleaved. silu on one host thread (`VGPU_THREADS=1`, the rig's launch times): `silu_forward_kernel starts the driver:
size rule (sim 14.740 ms, card path with the start 250.457 ms, 0.251 s of eligible launches simulated so far, 5 of this kernel)`; 20 launches
sent after it, 0.204 s of simulator time between them (the rig's record: 20 of 50 on the card, 0.48 s to 0.55 s).

Seen on the way: hecbench/bh-cuda did not end within 30 s in 4 of 6 runs under the old build, and in 1 of 6 under the new with the switch
on and with it off (turbo mode, the stand-in driver). It was there before the round and was not looked into.
