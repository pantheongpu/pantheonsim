# Round "turbo-next" card measurements, 2026-11-17 (A10G, sm_86, 80 SMs; one run each unless said; VGPU_THREADS=4; before-build = turbo-fit fae3230b)

Every number is measured on that A10G. Differences are from VERIFY=all (each launch the card ran is compared with the simulator's result). "x sim" = simulator s / turbo s.

## 1a. math-heavy, atomic-heavy and tensor-core programs, profile a10g, turbo FIT=1 (fae3230b)

| program | sim s | turbo s | launches on card / all | verify words different / outside tolerance (largest abs, rel) | turbo output | native s | x sim |
|---|---:|---:|---:|---|---|---:|---:|
| hecbench/adam-cuda | 0.04 | 1.42 | 1 / 1 | 0 / 0 (abs 0, rel 0) | identical | 0.33 | 0.03x |
| hecbench/aidw-cuda | TIMEOUT | 44.31 | 4 / 4 | TIMEOUT | - | 44.41 | - |
| hecbench/aobench-cuda | 0.72 | 0.24 | 1 / 1 | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 3.03x |
| hecbench/aop-cuda | 1.74 | 0.43 | 301 / 301 | 32595 / 0 (abs 2.27e+33, rel 1.15e-05) | identical | 0.43 | 4.10x |
| hecbench/asmooth-cuda | 0.24 | 0.27 | 3 / 3 | 0 / 0 (abs 0, rel 0) | identical | 0.27 | 0.89x |
| hecbench/atomicAggregate-cuda | 4.48 | 0.25 | 6 / 6 | 0 / 0 (abs 0, rel 0) | identical | 0.33 | 17.90x |
| hecbench/atomicCost-cuda | 22.06 rc[-9] | 72.14 | 4 / 6 | - | different (38 words against 0) | 21.87 | 0.31x |
| hecbench/atomicIntrinsics-cuda | 0.51 | 0.36 | 4 / 4 | 0 / 0 (abs 0, rel 0) | identical | 0.26 | 1.41x |
| hecbench/atomicPerf-cuda | 1.12 | 0.26 | 18 / 18 | 0 / 0 (abs 0, rel 0) | identical | 0.25 | 4.36x |
| hecbench/atomicReduction-cuda | TIMEOUT | 2.13 | 2100 / 2100 | TIMEOUT | - | 1.94 | - |
| hecbench/attention-cuda | 0.12 | 0.31 | 3 / 3 | 1 / 0 (abs 0.0022, rel 5.36e-07) | identical | 0.26 | 0.39x |
| hecbench/bgmv-cuda | 0.14 | 0.25 | 12 / 12 | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 0.55x |
| hecbench/bh-cuda | 1.04 | 0.25 | 17 / 17 | 23488 / 12583 (abs 3.42e-41, rel 0) (1 kernels) | identical | 0.24 | 4.13x |
| hecbench/binomial-cuda | TIMEOUT | 1.34 | 1000 / 1000 | TIMEOUT | - | 1.32 | - |
| hecbench/perplexity-cuda | 0.55 | 0.27 | 1 / 1 | 0 / 0 (abs 0, rel 0) | identical | 0.27 | 2.00x |
| hecbench/radixsort2-cuda | 3.20 | 0.30 | 600 / 600 | 0 / 0 (abs 0, rel 0) | identical | 0.25 | 10.65x |
| hecbench/s3d-cuda | 0.70 | 0.41 | 54 / 54 | 0 / 0 (abs 0, rel 0) | identical | 0.25 | 1.71x |
| hecbench/sortKV-cuda | 0.44 | 0.35 | 48 / 48 | 0 / 0 (abs 0, rel 0) | identical | 0.27 | 1.26x |
| samples/MC_EstimatePiP | 0.04 rc[1] | 0.02 rc[1] | - | - | identical | 3.10 | 2.23x |
| samples/MC_SingleAsianOptionP | 3.38 | 0.33 | 3 / 3 | 0 / 0 (abs 0, rel 0) | identical | 0.37 | 10.37x |
| samples/bf16TensorCoreGemm | 182.85 | 3.79 | 1 / 1 | 0 / 0 (abs 0, rel 0) | different (abs 2.52, rel 252) | 4.51 | 48.26x |
| samples/binaryPartitionCG | 0.02 | 0.29 | 1 / 1 | 0 / 0 (abs 0, rel 0) | identical | 0.50 | 0.05x |
| samples/cudaTensorCoreGemm | 16.02 | 1.24 | 1 / 1 | 0 / 0 (abs 0, rel 0) | different (abs 0.49, rel 49) | 1.11 | 12.95x |
| samples/histogram | 11.58 | 1.32 | 68 / 68 | 0 / 0 (abs 0, rel 0) | identical | 1.46 | 8.76x |
| samples/immaTensorCoreGemm | 3.53 | 1.11 | 1 / 1 | 0 / 0 (abs 0, rel 0) | different (abs 0.52, rel 10.4) | 1.06 | 3.18x |
| samples/mergeSort | 6.99 | 0.49 | 49 / 49 | 0 / 0 (abs 0, rel 0) | identical | 0.74 | 14.36x |
| samples/reduction | 5.78 | 0.71 | 201 / 201 | 0 / 0 (abs 0, rel 0) | identical | 0.56 | 8.10x |
| samples/scalarProd | 0.06 | 0.28 | 1 / 1 | 0 / 0 (abs 0, rel 0) | identical | 0.38 | 0.20x |
| samples/shfl_scan | 0.09 | 0.29 | 5 / 5 | 0 / 0 (abs 0, rel 0) | identical | 0.51 | 0.32x |
| samples/simpleAtomicIntrinsics | 0.01 | 0.24 | 1 / 1 | 2 / 0 (abs 4.3e-42, rel 0) | identical | 0.40 | 0.03x |
| samples/tf32TensorCoreGemm | 237.91 | 2.93 | 1 / 1 | 0 / 0 (abs 0, rel 0) | different (abs 1.03, rel 0) | 2.63 | 81.10x |
| samples/threadFenceReduction | 0.35 | 0.38 | 101 / 101 | 0 / 0 (abs 0, rel 0) | identical | 1.42 | 0.91x |
| samples/warpAggregatedAtomicsCG | 2.05 | 0.74 | 3 / 3 | 16000741 / 0 (abs 1.39e-38, rel 0) | identical | 0.81 | 2.78x |

Notes (measured): TIMEOUT = the simulator did not finish in 240 s (aidw, atomicReduction, binomial: native 44.4, 1.94 and 1.32 s). atomicCost: the simulator run was killed (rc -9) and the turbo run took 72 s with a different output text (38 words against 0): not understood, unknown. The tensor-core samples print their own throughput numbers, so "different" in the turbo-output column there is text that differs by run; their verify output is identical except immaTensorCoreGemm.

## 1b. noise: three runs of the 14 earlier programs, profile a10g, turbo FIT=1 (fae3230b)

| program | sim s (3 runs) | turbo s (3 runs) | x sim (medians) | native s (3 runs) |
|---|---|---|---:|---|
| polybench/2DCONV | 0.98/0.94/0.94 | 0.89/0.89/0.89 | 1.06x | 0.81/0.80/0.80 |
| polybench/ATAX | 1.14/1.09/1.09 | 0.43/0.42/0.37 | 2.60x | 0.33/0.33/0.32 |
| polybench/FDTD-2D | 95.86/97.78/95.81 | 4.24/4.22/4.26 | 22.62x | 4.14/4.12/4.13 |
| polybench/GEMM | 0.72/0.71/0.71 | 0.53/0.54/0.58 | 1.31x | 0.73/0.66/0.67 |
| polybench/LU | 28.33/28.35/28.11 | 1.10/1.21/1.04 | 25.74x | 1.07/0.99/0.97 |
| rodinia/backprop | 0.68/0.68/0.67 | 0.46/0.45/0.46 | 1.48x | 0.45/0.40/0.45 |
| rodinia/gaussian | 3.82/3.82/3.81 | 0.32/0.27/0.32 | 11.91x | 0.31/0.25/0.25 |
| rodinia/srad_v2 | 0.07/0.06/0.06 | 0.53/0.44/0.43 | 0.14x | 0.29/0.30/0.30 |
| rodinia/hotspot | 1.34/1.34/1.34 | 0.27/0.26/0.27 | 5.03x | 0.24/0.24/0.24 |
| rodinia/bfs | 0.10/0.01/0.01 | 0.24/0.24/0.25 | 0.06x | 0.24/0.24/0.24 |
| hecbench/atan2-cuda | 0.03/0.03/0.03 | 0.25/0.25/0.25 | 0.12x | 0.24/0.25/0.24 |
| hecbench/bilateral-cuda | 1.50/1.49/1.50 | 0.33/0.34/0.33 | 4.47x | 0.33/0.33/0.33 |
| hecbench/black-scholes-cuda | 28.57/28.43/28.45 | 6.01/5.92/5.93 | 4.79x | 5.31/5.32/5.31 |
| hecbench/silu-cuda | 0.65/0.65/0.65 | 0.28/0.29/0.28 | 2.29x | 0.33/0.32/0.28 |

The spread between repeats is under 10 % everywhere but the sub-second programs (a few hundredths of a second).

## 1c. builds for sm_90 only, profile h100 (sm_90 simulated), A10G card, turbo before retargeting (fae3230b) and after (this branch)

"ptx90" = -gencode arch=compute_90,code=compute_90 (PTX only); "cubin90" = code=sm_90 (SASS only); "sm90" = -arch=sm_90 (SASS and PTX). Native runs are impossible (sm_90 on an sm_86 card). Output equality is not judged here (the crude text compare of the script flags timing lines); the verify run is.

| program | build | simulator s | turbo before: s, launches on card/all | turbo after (retarget): s, launches on card/all | after, VERIFY=all: words different |
|---|---|---:|---|---|---|
| GEMM | ptx90 | 1.5083 | 1.6292 s, 0/1 | 2.9988 s, 1/1 | 0 (card launches 1) |
| GEMM | cubin90 | .75080 | .96606 s, 0/1 | .97611 s, 0/1 | 0 (card launches 0) |
| GEMM | sm90 | .75120 | .97244 s, 0/1 | .52414 s, 1/1 | 0 (card launches 1) |
| ATAX | ptx90 | 2.3198 | 2.6181 s, 0/2 | .45778 s, 2/2 | 0 (card launches 2) |
| ATAX | cubin90 | 1.1240 | 1.3516 s, 0/2 | 1.3975 s, 0/2 | 0 (card launches 0) |
| ATAX | sm90 | 1.1285 | 1.3719 s, 0/2 | .32436 s, 2/2 | 0 (card launches 2) |
| 2DCONV | ptx90 | 1.3324 | 1.5750 s, 0/1 | .75551 s, 1/1 | 0 (card launches 1) |
| 2DCONV | cubin90 | .98561 | 1.2214 s, 0/1 | 1.0958 s, 0/1 | 0 (card launches 0) |
| 2DCONV | sm90 | .99052 | 1.2116 s, 0/1 | .76602 s, 1/1 | 0 (card launches 1) |
| FDTD-2D | ptx90 | 194.52 | 194.29 s, 0/1500 | 4.2043 s, 1500/1500 | 0 (card launches 1500) |
| FDTD-2D | cubin90 | 121.10 | 118.01 s, 0/1500 | 119.03 s, 0/1500 | 0 (card launches 0) |
| FDTD-2D | sm90 | 117.46 | 117.87 s, 0/1500 | 4.2482 s, 1500/1500 | 0 (card launches 1500) |
| atomicAggregate-cuda | ptx90 | 7.0940 | 7.3486 s, 0/6 | .26926 s, 6/6 | 0 (card launches 6) |
| atomicAggregate-cuda | cubin90 | 14.039 | 14.254 s, 0/6 | 14.314 s, 0/6 | 0 (card launches 0) |
| atomicAggregate-cuda | sm90 | 14.026 | 14.295 s, 0/6 | .24737 s, 6/6 | 0 (card launches 6) |
| silu-cuda | ptx90 | .60849 | .83923 s, 0/50 | .32484 s, 50/50 | 0 (card launches 50) |
| silu-cuda | cubin90 | .81053 | 1.0463 s, 0/50 | 1.0236 s, 0/50 | 0 (card launches 0) |
| silu-cuda | sm90 | .81248 | 1.0442 s, 0/50 | .26061 s, 50/50 | 0 (card launches 50) |

## 2. driver start while the program sets up (VGPU_HOST_GPU_PRESTART, default on with FIT=1), profile a10g, three runs each

| program | turbo before (3 runs) | turbo with prestart (3 runs) | median before | median after |
|---|---|---|---:|---:|
| polybench/2DCONV | 0.89/0.89/0.89 | 0.75/0.74/0.75 | 0.89 | 0.75 |
| polybench/ATAX | 0.43/0.42/0.37 | 0.32/0.32/0.32 | 0.42 | 0.32 |
| polybench/FDTD-2D | 4.24/4.22/4.26 | 4.22/4.10/4.19 | 4.24 | 4.19 |
| polybench/GEMM | 0.53/0.54/0.58 | 0.53/0.53/0.52 | 0.54 | 0.53 |
| polybench/LU | 1.10/1.21/1.04 | 1.00/1.00/1.00 | 1.10 | 1.00 |
| rodinia/backprop | 0.46/0.45/0.46 | 0.34/0.34/0.34 | 0.46 | 0.34 |
| rodinia/gaussian | 0.32/0.27/0.32 | 0.27/0.27/0.33 | 0.32 | 0.27 |
| rodinia/srad_v2 | 0.53/0.44/0.43 | 0.44/0.47/0.43 | 0.44 | 0.44 |
| rodinia/hotspot | 0.27/0.26/0.27 | 0.27/0.26/0.27 | 0.27 | 0.27 |
| rodinia/bfs | 0.24/0.24/0.25 | 0.24/0.24/0.24 | 0.24 | 0.24 |
| hecbench/atan2-cuda | 0.25/0.25/0.25 | 0.29/0.30/0.29 | 0.25 | 0.29 |
| hecbench/bilateral-cuda | 0.33/0.34/0.33 | 0.34/0.33/0.33 | 0.33 | 0.33 |
| hecbench/black-scholes-cuda | 6.01/5.92/5.93 | 5.80/5.78/5.85 | 5.93 | 5.80 |
| hecbench/silu-cuda | 0.28/0.29/0.28 | 0.26/0.26/0.27 | 0.28 | 0.26 |

Verify pass of the same build (s2_verify): every program 0 words different, outputs identical.

## 4. profile limits against the card (raw lines)

```
shared=80KB build=base plain: set_attribute=1 launch=1 mode=shared sum=0  [vgpu] cudaLaunchKernel: VirtualGPU error [launch-config]: kernel '_Z11shared_userPi' asks for 81920 bytes of dynamic shared memory; it may use 49152 (cudaFuncAttributeMaxDynamicSharedMemorySize raise || turbo: set_attribute=1 launch=1 mode=shared sum=0  [vgpu] cudaLaunchKernel: VirtualGPU error [launch-config]: kernel '_Z11shared_userPi' asks for 81920 bytes of dynamic shared memory; it may use 49152 (cudaFuncAttributeMaxDynamicSharedMemorySize raise 
shared=80KB build=new plain: set_attribute=1 launch=1 mode=shared sum=0  [vgpu] cudaLaunchKernel: VirtualGPU error [launch-config]: kernel '_Z11shared_userPi' asks for 81920 bytes of dynamic shared memory; it may use 49152 (cudaFuncAttributeMaxDynamicSharedMemorySize raise || turbo:  [vgpu] cudaLaunchKernel: VirtualGPU error [launch-config]: kernel '_Z11shared_userPi' asks for 81920 bytes of dynamic shared memory; it may use 49152 (cudaFuncAttributeMaxDynamicSharedMemorySize raise 
shared=40KB build=base plain: set_attribute=0 launch=0 mode=shared sum=228480   || turbo: set_attribute=0 launch=0 mode=shared sum=228480   host_gpu: _Z11shared_userPi on the card (resident): kernel 0.089 ms, total 20.147 ms (sim estimate 0
shared=40KB build=new plain: set_attribute=0 launch=0 mode=shared sum=228480   || turbo: set_attribute=0 launch=0 mode=shared sum=228480   host_gpu: _Z11shared_userPi on the card (resident): kernel 0.063 ms, total 0.554 ms (sim estimate 0.
H100 profile shared=120KB build=base plain: set_attribute=0 launch=0 mode=shared sum=228480  || turbo: set_attribute=0 launch=0 mode=shared sum=228480  host_gpu: _Z11shared_userPi stays in the simulator: does not fit the card: more shared memory than the card allows a blo
H100 profile shared=120KB build=new plain: set_attribute=0 launch=0 mode=shared sum=228480  || turbo: set_attribute=0 launch=0 mode=shared sum=228480  host_gpu: _Z11shared_userPi stays in the simulator: does not fit the card: more shared memory than the card allows a blo
```

## 3b. a module the driver refuses after retargeting (measured, A10G, profile h100)

A PTX-only sm_90 program with a form the text scan does not know (`atom.add.noftz.bf16x2`, sm_90): the scan admits the module, the driver's compile for sm_86 fails, and the module goes back to the simulator. Plain simulator and turbo print the same result (`bf sum=100729878 err=0`). The failure is loud (stderr), not silent:

```
host_gpu: cuModuleLoadData failed (218) for the card's compute_90 PTX retargeted to sm_86
host_gpu: _Z5bfaddP14__nv_bfloat162i stays in the simulator: the driver did not load the card's image
host_gpu: 0 launches ran on the card, 2 in the simulator (... 1 failed on the card)
host_gpu_fit: policy=fit card_sm=86 card_sms=80 refused_image=1 refused_fit=0 retargeted=0 retarget_refused=0
```

The whole module is demoted (its other kernel `plain` ran in the simulator too); turbo is not ended. Error 218 is CUDA_ERROR_INVALID_PTX.
