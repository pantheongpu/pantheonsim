# Program history, driver compile cache, fixed cost of a turbo process: measurements (2026-11-17)

Design: [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) section 17. One AWS g5.xlarge (A10G, sm_86, 80 SMs, driver 595.112, CUDA 13.0 toolkit, native Linux), a10g profile unless said, `VGPU_THREADS=4`, `tools/bench/hostgpu/history_bench.py`
(forms interleaved per repetition; a fresh history directory per repetition for run 1/2/3). Wall seconds, median. "before" = `VGPU_HOST_GPU=turbo` with the driver's own work-queue count (`CUDA_DEVICE_MAX_CONNECTIONS` set to an empty value) and no history, "turbo" = turbo with no history (one work queue),
"run 1/2/3" = turbo with `VGPU_HOST_GPU_HISTORY` on a directory made fresh for the repetition. Raw numbers: [history/history_bench.json](history/history_bench.json). "same" = the output text equalled the simulator's (timing lines dropped) in every run of every turbo form.

## 1. First, second and third run (final build)

| program | runs | native | sim | before | turbo | run 1 | run 2 | run 3 | same | history line of run 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rodinia/bfs | 5 | 0.238 | 0.014 | 0.240 | 0.175 | 0.176 | 0.014 | 0.014 | True | candidate after=tiny held=1 released=0 |
| hecbench/backprop-cuda | 5 | 0.240 | 0.013 | 0.239 | 0.177 | 0.176 | 0.013 | 0.013 | True | candidate after=tiny held=1 released=0 |
| hecbench/attention-cuda | 5 | 0.256 | 0.035 | 0.246 | 0.183 | 0.183 | 0.036 | 0.036 | True | candidate after=tiny held=1 released=0 |
| rodinia/srad_v2 | 5 | 0.303 | 0.063 | 0.443 | 0.374 | 0.373 | 0.064 | 0.064 | True | unknown after=candidate held=0 released=0 |
| hecbench/atan2-cuda | 5 | 0.243 | 0.029 | 0.245 | 0.186 | 0.182 | 0.181 | 0.182 | True | unknown after=unknown held=0 released=0 |
| polybench/2DCONV | 5 | 0.805 | 0.939 | 0.746 | 0.727 | 0.724 | 0.725 | 0.724 | True | unknown after=unknown held=0 released=0 |
| polybench/ATAX | 5 | 0.327 | 1.091 | 0.321 | 0.256 | 0.259 | 0.260 | 0.259 | True | unknown after=unknown held=0 released=0 |
| polybench/GEMM | 5 | 0.673 | 0.740 | 0.559 | 0.455 | 0.466 | 0.468 | 0.462 | True | unknown after=unknown held=0 released=0 |
| polybench/FDTD-2D | 3 | 4.467 | - | 4.257 | 4.205 | 4.167 | 4.181 | 4.185 | - | unknown after=unknown held=0 released=0 |
| polybench/LU | 3 | 0.990 | - | 1.052 | 0.957 | 0.937 | 1.005 | 0.971 | - | unknown after=unknown held=0 released=0 |
| hecbench/aobench-cuda | 3 | 0.240 | 0.712 | 0.244 | 0.176 | 0.177 | 0.177 | 0.179 | True | unknown after=unknown held=0 released=0 |
| hecbench/bilateral-cuda | 3 | 0.339 | 1.472 | 0.333 | 0.277 | 0.279 | 0.274 | 0.271 | True | unknown after=unknown held=0 released=0 |
| rodinia/lavaMD | 3 | 0.278 | 1.528 | 0.275 | 0.214 | 0.217 | 0.218 | 0.218 | True | unknown after=unknown held=0 released=0 |
| hecbench/perplexity-cuda | 3 | 0.272 | 0.396 | 0.269 | 0.212 | 0.208 | 0.212 | 0.208 | True | unknown after=unknown held=0 released=0 |
| hecbench/bh-cuda | 3 | 0.291 | 0.929 | 0.245 | 0.184 | 0.184 | 0.189 | 0.186 | True | unknown after=unknown held=0 released=0 |

Notes: FDTD-2D and LU have no simulator run (92 s and 28 s each). The four programs after LU are the ones an earlier, looser candidate rule (card path under 3 ms, at most 131,072 threads) sent to a trial run that lost time (aobench +0.47 s, bilateral +0.52 s,
GEMM +0.21 s against turbo, one run each, wall); under the rule in the design none of them is a candidate and run 2 equals run 1.

## 2. What fitted the candidate rule: 45 programs, simulator launch seconds against the first card run

`candidate_calibration.json`: per program `sim_launch_s` (seconds of the simulator's launches, from `VGPU_HOST_GPU_FIT=0` with the start cost made too large to pay, so every launch is the simulator's), `sim_launches`, `card_s` (the card path of a turbo run), `threads` (all launches), `work` and `loop_work`
(threads x static instructions, all kernels and kernels the first version of the counter called loops; `back_edge` is true of every SASS kernel, which ends in a branch to itself, so the final rule uses the flow cycle test instead). A "null" sim is a simulator run over its 12 s limit.

## 3. Driver compile cache (PTX-only builds for compute_90, h100 profile on the A10G; scripts `history/compile_cache_experiment.sh`, `compile_cache_cold_vs_warm.sh`)

| case | wall s | card path s (LOG=1) |
|---|---:|---:|
| GEMM, default cache `~/.nv/ComputeCache` removed first, run 1 (first run on a new rig) | 3.865 | |
| GEMM, default cache, runs 2 and 3 | 0.572, 0.574 | |
| GEMM, fresh `CUDA_CACHE_PATH`, run 1, run 2 (twice) | 0.600, 0.525; 0.551, 0.540 | |
| GEMM, `CUDA_CACHE_DISABLE=1`, runs 1 and 2 | 0.572, 0.558 | |
| GEMM, `CUDA_CACHE_MAXSIZE=4096`, runs 1 and 2 | 0.554, 0.554 | |
| GEMM, after `drop_caches`, cache disabled | 0.819 | 0.171 |
| GEMM, warm, cache disabled | 0.584 | 0.023 |
| GEMM, after `drop_caches`, fresh cache dir | 1.091 | 0.171 |
| GEMM, warm, cache hit | 0.539 | 0.0018 |
| GEMM, after `drop_caches`, cache hit | 0.702 | 0.0065 |
| ATAX, default cache runs 1 to 3 | 0.456, 0.322, 0.322 | |
| s3d (54 kernels, 2.9 MB of cache), cache disabled, runs 1 and 2 | 4.171, 3.818 | 3.284, 3.019 |
| s3d, fresh cache dir, run 1 and 2 | 3.821, 0.845 | 3.014, 0.025 |
| s3d with `VGPU_HOST_GPU_HISTORY` on a fresh directory (the cache beside it), runs 1 to 3 | 3.963, 0.762, 0.772 | |

## 4. Fixed cost of a turbo process (`history/driver_start_probe.c`, `work_queue_sweep.sh`, `process_timeline.sh`)

* Standalone program making the calls of `begin_start`, ms (three runs, plain and RTLD_DEEPBIND): dlopen 1.5, cuInit 47 to 48, cuDevicePrimaryCtxRetain 92 to 98, everything else 0.0 (the 1 TiB reservation included), total 141 to 149; wall of the program 234 to 242 ms. After `drop_caches`: 12 + 79 + 141 = 235 ms of calls.
* `CUDA_DEVICE_MAX_CONNECTIONS` (bfs under turbo, six runs each, wall s): unset 0.232 to 0.245; 1 0.148 to 0.162; 2 0.170 to 0.185; 4 0.204 to 0.216; 8 0.274 to 0.281; 32 0.751 to 0.773. In the probe, cuDevicePrimaryCtxRetain with 1: 46.7 to 47.0 ms. FDTD-2D under turbo 4.276 and 4.244 unset, 4.141 and 4.144 with 1.
* No effect on the probe's calls: `CUDA_MODULE_LOADING` LAZY or EAGER, `CUDA_CACHE_DISABLE=1`, `CUDA_VISIBLE_DEVICES=0`, persistence mode on (142 to 147 ms) or off (141 to 151 ms).
* strace -f -ttt of bfs: the last thread of a process that used the card (turbo or native) exits 85 to 90 ms after `exit_group`; the mode off 1 ms.

## 5. Mutations (each alone, rebuilt, unit test and stub steps 58 to 63 run on the rig)

See `history/M*.summary` (first failing line of each; "stub rc=1" or "unit rc=1" means it failed there; both must not pass for the mutation to be caught by at least one).
