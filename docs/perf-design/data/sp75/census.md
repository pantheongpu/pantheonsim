# Round sp75: the census (server3, the stand-in driver, turbo mode)

Every program once in each form, `VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1`, profile `nvidia/a10g`, `VGPU_THREADS=3`, against the stand-in driver,
which refuses every module: a launch the mode sends "fails on the card" and the simulator runs it. `old`: commit 6cf503d3 (the rules before the
round). `new`: this round's. A run is cut at the time limit; a cut run has no summary, and its decisions are the lines it logged before the cut.

**How to read this file.** The first pass (below) ran while the rule was being finished: its `new` runs of PolyBench used a limit of 18 ms (15 ms after),
and those of the first 58 programs (PolyBench, Rodinia, HeCBench up to blockAccess) counted a small grid's sample at 3 ms for a process's
age, which is how HeCBench bilateral came to be sampled; the rest counted the two runs at the cap. Its `old` build does not print a sampled
launch's decision, so a program cut at the time limit shows as "different" whenever the new build sampled a launch. Its wall times are single
runs on a shared machine (see the last table: differences of up to 2.5 times with no decision changed) and are not the timing evidence;
`rerun.md` is. **The follow-up** at the end is the pass to read for decisions: the 31 programs that differed, were cut or hold a sampled
launch, under one build (62c35bed) with `VGPU_HOST_GPU_SAMPLE_BLOCK=0` (`knob`, the rule before) and without (`new`). The slack of a small
grid's limit was raised from 1.25 to 1.4 after both passes (ae13d5e6); the four GEMMs and aidw were run again with it (`launches.md`).

Programs: 124 (13 cut at the time limit in a form).

## Programs whose decisions differ between old and new

| program | old: kept / too small / sent / sampled / sent on a sample / copies that did not answer | new | new: the sampled launches |
|---|---|---|---|
| hecbench/aidw-cuda | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | answered=1 ran=1 (the last block stopped at the limit: at least) estimate 1973.320 ms against a card path of 497.198 ms with the start: sent |
| hecbench/asyncAllocation-cuda | 800 / 763 / 35 / 0 / 0 / 0 | 800 / 726 / 72 / 0 / 0 / 0 |  |
| hecbench/atomicPerf-cuda | 18 / 0 / 10 / 2 / 0 / 0 | 18 / 0 / 11 / 1 / 0 / 0 | answered=1 ran=32 estimate 23.896 ms against a card path of 252.995 ms with the start: kept |
| hecbench/bh-cuda | 17 / 0 / 0 / 1 / 0 / 0 | 17 / 0 / 0 / 0 / 0 / 0 |  |
| hecbench/bilateral-cuda | 3 / 0 / 0 / 0 / 0 / 0 | 3 / 0 / 0 / 1 / 0 / 0 | answered=1 ran=1 estimate 226.919 ms against a card path of 278.420 ms with the start: kept |
| hecbench/binomial-cuda | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | answered=1 ran=1 estimate 953.779 ms against a card path of 369.234 ms with the start: sent |
| hecbench/radixsort2-cuda | 600 / 383 / 211 / 0 / 0 / 0 | 600 / 400 / 194 / 0 / 0 / 0 |  |
| polybench/2MM | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | answered=1 ran=2 estimate 9520.824 ms against a card path of 1455.489 ms with the start: sent |
| polybench/LU | 4096 / 3141 / 949 / 0 / 0 / 0 | 4096 / 3143 / 947 / 0 / 0 / 0 |  |
| rodinia/gaussian | 1022 / 546 / 472 / 0 / 0 / 0 | 1022 / 545 / 473 / 0 / 0 / 0 |  |
| rodinia/lud | 190 / 145 / 39 / 0 / 0 / 0 | 190 / 136 / 48 / 0 / 0 / 0 |  |
| rodinia/particlefilter_float | 36 / 7 / 21 / 1 / 0 / 0 | 36 / 10 / 18 / 1 / 0 / 0 | inside its blocks: answered=1 ran=20 estimate 0.000 ms against a card path of 250.783 ms with the start: kept |
| samples/bf16TensorCoreGemm | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | answered=1 ran=1 (the last block stopped at the limit: at least) estimate 1150.763 ms against a card path of 541.492 ms with the start: sent |
| samples/cudaTensorCoreGemm | 1 / 0 / 0 / 0 / 0 / 0 | 1 / 0 / 1 / 1 / 0 / 0 | answered=1 ran=1 (the last block stopped at the limit: at least) estimate 878.321 ms against a card path of 396.707 ms with the start: sent |
| samples/immaTensorCoreGemm | 1 / 0 / 0 / 0 / 0 / 0 | 1 / 0 / 1 / 1 / 0 / 0 | answered=1 ran=1 estimate 768.667 ms against a card path of 376.849 ms with the start: sent |
| samples/interval | 100 / 3 / 95 / 0 / 0 / 0 | 100 / 2 / 96 / 0 / 0 / 0 |  |
| samples/radixSortThrust | 600 / 104 / 490 / 0 / 0 / 0 | 600 / 104 / 490 / 1 / 0 / 0 | answered=1 ran=0 estimate 0.000 ms against a card path of 253.215 ms with the start: kept |
| samples/tf32TensorCoreGemm | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | answered=1 ran=1 (the last block stopped at the limit: at least) estimate 1204.998 ms against a card path of 548.271 ms with the start: sent |

## Samples whose copy did not answer under the old rule (it was ended at the wait's end, or the launch threw in it)

| program | old: copies that did not answer | new | new: blocks stopped at the limit |
|---|---:|---:|---|
| rodinia/srad_v2 | 1 | 1 |  |

## Launches on a grid of 2 to 255 blocks of a kernel with a loop and no followed loop (new: the candidates of the bounded sample of a small grid)

| program | kernel | blocks | instructions | static estimate ms | new |
|---|---|---:|---:|---:|---|
| samples/radixSortThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1` | 161 | 4272 | 82.53 | not sampled: its earlier whole launch was short |
| hecbench/aop-cuda | `_Z18prepare_svd_kernelILi256E9PayoffPutEviiT0_PK` | 99 | 2344 | 18.56 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z11ratx_kernelIfEvPKT_S2_PS0_S3_S2_S0_` | 216 | 2240 | 9.68 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z12ratxb_kernelIfEvPKT_S2_PS0_S3_S2_S0_` | 216 | 1976 | 8.54 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z11ratt_kernelIfEvPKT_PS0_S0_` | 108 | 1952 | 8.43 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z11ratt_kernelIdEvPKT_PS0_S0_` | 32 | 4864 | 6.23 | not sampled: static estimate under the floor and the process too young |
| hecbench/bilateral-cuda | `_Z15bilateralFilterILi3EEvPKfPfiifff` | 128 | 600 | 6.14 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z11qssa_kernelIfEvPT_S1_S1_` | 108 | 1376 | 5.94 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z11ratx_kernelIdEvPKT_S2_PS0_S3_S2_S0_` | 64 | 4472 | 5.72 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z12rdsmh_kernelIfEvPKT_PS0_S0_` | 108 | 1304 | 5.63 | not sampled: static estimate under the floor and the process too young |
| samples/binaryPartitionCG | `_Z20oddEvenCountAndSumCGPiS_S_j` | 160 | 136 | 5.22 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z12ratxb_kernelIdEvPKT_S2_PS0_S3_S2_S0_` | 64 | 3808 | 4.87 | not sampled: static estimate under the floor and the process too young |
| rodinia/particlefilter_float | `_Z17likelihood_kernelPdS_S_S_S_PiS0_S_PhS_S_iiii` | 20 | 1424 | 4.56 | not sampled: static estimate under the floor and the process too young |
| hecbench/bilateral-cuda | `_Z15bilateralFilterILi6EEvPKfPfiifff` | 128 | 416 | 4.26 | not sampled: static estimate under the floor and the process too young |
| samples/scalarProd | `_Z13scalarProdGPUPfS_S_ii` | 128 | 416 | 4.26 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z12qssab_kernelIfEvPT_S1_S1_` | 216 | 952 | 4.11 | not sampled: static estimate under the floor and the process too young |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29D` | 22 | 1544 | 4.08 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z12ratt9_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 | 904 | 3.91 | not sampled: static estimate under the floor and the process too young |
| hecbench/s3d-cuda | `_Z12rdsmh_kernelIdEvPKT_PS0_S0_` | 32 | 3008 | 3.85 | not sampled: static estimate under the floor and the process too young |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29D` | 11 | 2656 | 3.51 | not sampled: static estimate under the floor and the process too young |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29D` | 11 | 2592 | 3.42 | not sampled: static estimate under the floor and the process too young |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29D` | 11 | 2592 | 3.42 | not sampled: static estimate under the floor and the process too young |
| rodinia/lavaMD | `_Z15kernel_gpu_cuda7par_str7dim_strP7box_strP11F` | 216 | 392 | 3.39 | not sampled: static estimate under the floor and the process too young |
| samples/jacobiCudaGraphs | `__nv_static_30__43dbd644_9_jacobi_cu_3dc991e0__Z` | 66 | 624 | 3.30 | not sampled: static estimate under the floor and the process too young |

(88 kernels in all; the 24 with the largest static estimate are listed.)

(A small grid that IS sampled has no `host_gpu_nosample` line: it is in the first table.)

## Wall time, new against old, over 3 % slower in this one run (the stand-in driver: every launch is simulated in both; reruns in README.md)

| program | old s | new s | |
|---|---:|---:|---:|
| rodinia/sc_gpu | 0.662 | 2.359 | +256.1 % |
| polybench/GEMVER | 0.451 | 0.943 | +108.9 % |
| samples/dct8x8 | 1.344 | 2.189 | +62.9 % |
| polybench/MVT | 0.790 | 1.104 | +39.7 % |
| polybench/JACOBI1D | 0.532 | 0.709 | +33.3 % |
| rodinia/hotspot | 0.388 | 0.499 | +28.5 % |
| polybench/JACOBI2D | 0.226 | 0.271 | +20.0 % |
| rodinia/nn | 0.016 | 0.019 | +19.4 % |
| polybench/SYRK | 2.925 | 3.223 | +10.2 % |
| polybench/3MM | 1.238 | 1.360 | +9.8 % |
| samples/interval | 7.174 | 7.811 | +8.9 % |
| polybench/GESUMMV | 0.240 | 0.259 | +8.0 % |
| rodinia/pathfinder | 0.427 | 0.453 | +6.1 % |
| polybench/3DCONV | 0.371 | 0.393 | +5.8 % |
| rodinia/lud | 2.208 | 2.315 | +4.9 % |

Sum over the programs not cut: old 294.2 s, new 272.9 s.

## What keeps a launch in the simulator (new; the reasons are this round's `host_gpu_kept` texts; programs not cut)

| reason | launches | sim s | programs |
|---|---:|---:|---:|
| failed on the card | 6029 | 93.48 | 22 |
| failed on the card (sent by its sample) | 8 | 46.31 | 8 |
| size rule: no estimate, not sampled: static estimate under the floor and the process too young | 164 | 9.79 | 61 |
| size rule: the estimate does not beat the card path | 31788 | 8.94 | 27 |
| size rule: the estimate does not repay the driver start | 2835 | 7.17 | 25 |
| safety: argument points into mapped host memory | 38 | 3.48 | 4 |
| size rule: no estimate, not sampled: the driver has been started | 68 | 2.00 | 8 |
| size rule: no estimate, not sampled: the kernel has no loop | 126 | 1.72 | 37 |
| size rule: no estimate, not sampled: its earlier whole launch was short | 71 | 1.11 | 31 |
| size rule: the sampled estimate does not repay the card path and the driver start | 8 | 0.99 | 8 |
| size rule: no estimate, not sampled: a grid under 256 blocks and no counted loop to follow | 17 | 0.71 | 8 |
| safety: cooperative or cluster launch | 1 | 0.46 | 1 |
| class never | 23 | 0.35 | 5 |
| size rule: no estimate, the sample gave none | 3 | 0.14 | 3 |
| safety: the kernel calls device-side launch | 4 | 0.01 | 4 |
| safety: the device heap (kernel malloc) holds device memory the card copy would lack | 4 | 0.00 | 2 |
| safety: not a SASS module | 1 | 0.00 | 1 |
| safety: the kernel calls malloc/free | 2 | 0.00 | 1 |

# The follow-up: the switch off (knob) against on (new), one build

31 programs, 45 s a run. Columns as above, with `knob` in the place of `old`.

## Programs whose decisions differ between old and new

| program | old: kept / too small / sent / sampled / sent on a sample / copies that did not answer | new | new: the sampled launches |
|---|---|---|---|
| hecbench/aidw-cuda | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | answered=1 ran=1 (the last block stopped at the limit: at least) estimate 1793.147 ms against a card path of 474.676 ms with the start: sent |
| hecbench/asyncAllocation-cuda | 800 / 763 / 35 / 0 / 0 / 0 | 800 / 788 / 10 / 0 / 0 / 0 |  |
| polybench/LU | 4096 / 3141 / 949 / 0 / 0 / 0 | 4096 / 3147 / 943 / 0 / 0 / 0 |  |
| rodinia/gaussian | 1022 / 546 / 472 / 0 / 0 / 0 | 1022 / 547 / 471 / 0 / 0 / 0 |  |
| rodinia/lud | 190 / 133 / 51 / 0 / 0 / 0 | 190 / 132 / 52 / 0 / 0 / 0 |  |
| samples/bf16TensorCoreGemm | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | answered=1 ran=1 (the last block stopped at the limit: at least) estimate 1263.713 ms against a card path of 555.611 ms with the start: sent |
| samples/cudaTensorCoreGemm | 1 / 0 / 0 / 0 / 0 / 0 | 1 / 0 / 1 / 1 / 0 / 0 | answered=1 ran=1 (the last block stopped at the limit: at least) estimate 882.041 ms against a card path of 397.172 ms with the start: sent |
| samples/immaTensorCoreGemm | 1 / 0 / 0 / 0 / 0 / 0 | 1 / 0 / 1 / 1 / 0 / 0 | answered=1 ran=1 estimate 765.708 ms against a card path of 376.479 ms with the start: sent |
| samples/radixSortThrust | 600 / 104 / 490 / 0 / 0 / 0 | 600 / 104 / 490 / 1 / 0 / 0 | answered=1 ran=0 estimate 0.000 ms against a card path of 253.215 ms with the start: kept |
| samples/tf32TensorCoreGemm | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | answered=1 ran=1 (the last block stopped at the limit: at least) estimate 1145.477 ms against a card path of 540.831 ms with the start: sent |

## Samples whose copy did not answer under the old rule (it was ended at the wait's end, or the launch threw in it)

| program | old: copies that did not answer | new | new: blocks stopped at the limit |
|---|---:|---:|---|
| rodinia/srad_v2 | 1 | 1 |  |

Sampled launches, per program, with the switch off and on (from the same records): aidw `kept` (no answer) against `sent` on a stopped block;
bf16TensorCoreGemm, cudaTensorCoreGemm, tf32TensorCoreGemm not sampled against `sent` on a stopped block; immaTensorCoreGemm not sampled against
`sent` (its block ended); radixSortThrust not sampled against one sample that ran no block. The same on both sides: binomial, black-scholes,
shmembench, 2MM, SYRK, SYR2K `sent`; COVAR and CORR `sent` inside their blocks; atomicPerf (two) and CORR's first `kept`; particlefilter_float
`kept` inside its blocks with no estimate; srad_v2 `kept`, its launch throwing in the copy. No block of these was stopped at the limit.
