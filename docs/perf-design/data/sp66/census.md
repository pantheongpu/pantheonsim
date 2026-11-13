# Round sp66: the census (server3, the stand-in driver, turbo mode)

Every program once, `VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1`, profile `nvidia/a10g`, `VGPU_THREADS=3`, against the stand-in driver, which refuses
every module: a launch the mode sends "fails on the card" and the simulator runs it, so every launch is timed in the simulator. Reasons are the
mode's own (`host_gpu_kept:` lines). `sent`: the mode sent the launch (on a card it runs there). `size rule`: kept by the size rule, the
first-launch rule or a sample (Round sp65's subject, not this round's).

* `old`: the rules before the round (commit aa0b45d6, which only names the reasons): PolyBench/GPU 9 of 20 (its other programs, and all of
  PolyBench in Round sp59, have no launch refused for anything but size), Rodinia 12, HeCBench 28 of 34 (aidw, aligned-types, babelstream,
  base64e, binomial and atomicCost run past the census's time limit in the simulator; none had a refusal in sp59), the CUDA Samples 13.0 that run
  in the simulator, 58: 107 programs.
* `new`: this round's rules (the branch head), on the 46 programs the old census found a refusal in; the other programs' decisions cannot change (so the `new` columns
  are over those 46, the `old` columns over all 107).
  HeCBench atomicReduction (all its launches `sent` in both) is left out of `new`: it runs past the time limit.

Refusals that the old census found and that remain (`new`): mapped host memory (HeCBench zerocopy, Samples simpleIPC and simpleZeroCopy: 3.4 s),
a cooperative launch (simpleAWBarrier, 0.5 s), texture and surface instructions (dct8x8, simpleCubemapTexture, simpleLayeredTexture,
simpleSurfaceWrite: 0.3 s), a call through a register (newdelete's virtual call), and the device runtime, malloc and the heap (the cdp* samples,
newdelete: under 0.01 s). What moved to `size rule` (161 s, nearly all of it bf16TensorCoreGemm and tf32TensorCoreGemm, one long launch each that
no rule yet sends on its first launch) is now admitted and is Round sp65's to decide. Every module of the corpus with variables got its card image
(no refusal "module variables: ..." in `new`).

Instructions that refused a kernel (`host_gpu_refused:`), old census: `CALL.ABS.NOINC` to another code section's first instruction in 11 programs
(all of the 156.9 s of `class: never`: bf16TensorCoreGemm, tf32TensorCoreGemm, cudaTensorCoreGemm, histogram, interval, mergeSort, sortingNetworks,
scalarProd, shfl_scan, simpleHyperQ, jacobiCudaGraphs); `ATOMS.CAST.SPIN`, `ATOMG.E.CAS`, `ATOMG.E.EXCH` (aop, atomicPerf, bh,
simpleAtomicIntrinsics). Once module variables were lifted the census showed `B2R.RESULT` and `NANOSLEEP` (cub's onesweep radix sort and its
decoupled look-back: radixSortThrust, segmentationTreeThrust, radixsort2, sortKV, 13.9 s), which joined the extra class.

Programs: old 107 (0 timed out: none), new 46 (0 timed out: none)

Simulator seconds of all launches: old 497.7 s, new 226.0 s

| reason | old launches | old sim s | old programs | new launches | new sim s | new programs |
|---|---:|---:|---:|---:|---:|---:|
| sent | 4383 | 228.42 | 11 | 4384 | 60.68 | 16 |
| size rule | 6072 | 61.78 | 63 | 2582 | 161.00 | 34 |
| class: never | 161 | 156.91 | 11 | 23 | 0.35 | 5 |
| safety: the module holds __device__, string literals (the kernel calls none) | 1114 | 20.79 | 2 | 0 | 0 | 0 |
| safety: inside a graph with conditionals | 3178 | 8.52 | 3 | 0 | 0 | 0 |
| safety: the module holds __device__ (the kernel calls none) | 1805 | 8.19 | 10 | 0 | 0 | 0 |
| safety: the module holds __constant__ (the kernel calls none) | 2142 | 4.88 | 3 | 0 | 0 | 0 |
| safety: argument points into mapped host memory | 38 | 3.39 | 4 | 30 | 3.45 | 3 |
| safety: the module holds __constant__, __device__ (the kernel calls none) | 5 | 3.03 | 2 | 0 | 0 | 0 |
| class: atomics | 210 | 0.97 | 4 | 0 | 0 | 0 |
| safety: cooperative or cluster launch | 1 | 0.45 | 1 | 1 | 0.46 | 1 |
| safety: texture objects | 23 | 0.36 | 4 | 0 | 0 | 0 |
| safety: the kernel calls device-side launch | 0 | 0 | 0 | 4 | 0.01 | 4 |
| safety: the kernel calls an unknown external function, device-side launch | 2 | 0.00 | 2 | 0 | 0 | 0 |
| safety: the kernel calls an unknown external function, device-side launch, malloc/free | 1 | 0.00 | 1 | 0 | 0 | 0 |
| safety: not a SASS module | 1 | 0.00 | 1 | 1 | 0.00 | 1 |
| safety: the device heap (kernel malloc) holds device memory the card copy would lack | 4 | 0.00 | 2 | 4 | 0.00 | 2 |
| safety: the kernel calls malloc/free | 2 | 0.00 | 1 | 2 | 0.00 | 1 |
| safety: the kernel calls vprintf | 2 | 0.00 | 2 | 0 | 0 | 0 |
| safety: the module holds __device__, malloc/free (the kernel calls none) | 1 | 0.00 | 1 | 0 | 0 | 0 |
| safety: the kernel calls an unknown external function, device-side launch, vprintf | 1 | 0.00 | 1 | 0 | 0 | 0 |
| safety: the module holds __device__, an unknown external function, device-side launch, string literals (the kernel calls none) | 5 | 0.00 | 1 | 0 | 0 | 0 |

Per program, the simulator seconds kept for a reason other than the size rule (`sent`: would run on a card):

| program | run | all sim s | sent s | size rule s | refused s | refused for |
|---|---|---:|---:|---:|---:|---|
| hecbench/aop-cuda | old | 0.74 | 0.00 | 0.09 | 0.65 | class: atomics 0.65 |
| hecbench/aop-cuda | new | 0.75 | 0.00 | 0.75 | 0.00 |  |
| hecbench/asyncAllocation-cuda | old | 1.24 | 0.00 | 0.00 | 1.24 | safety: the module holds __device__ (the kernel calls none) 1.24 |
| hecbench/asyncAllocation-cuda | new | 1.22 | 0.03 | 1.19 | 0.00 |  |
| hecbench/atomicPerf-cuda | old | 0.49 | 0.00 | 0.18 | 0.32 | class: atomics 0.32 |
| hecbench/atomicPerf-cuda | new | 0.49 | 0.23 | 0.27 | 0.00 |  |
| hecbench/radixsort2-cuda | old | 1.18 | 0.00 | 0.00 | 1.18 | safety: the module holds __device__ (the kernel calls none) 1.18 |
| hecbench/radixsort2-cuda | new | 1.21 | 0.94 | 0.26 | 0.00 |  |
| hecbench/sortKV-cuda | old | 0.09 | 0.00 | 0.00 | 0.09 | safety: the module holds __device__ (the kernel calls none) 0.09 |
| hecbench/sortKV-cuda | new | 0.09 | 0.00 | 0.09 | 0.00 |  |
| hecbench/zerocopy-cuda | old | 3.27 | 0.00 | 0.00 | 3.27 | safety: argument points into mapped host memory 3.27 |
| hecbench/zerocopy-cuda | new | 3.33 | 0.00 | 0.00 | 3.33 | safety: argument points into mapped host memory 3.33 |
| rodinia/particlefilter_float | old | 1.28 | 0.00 | 0.00 | 1.28 | safety: the module holds __device__ (the kernel calls none) 1.28 |
| rodinia/particlefilter_float | new | 1.16 | 0.84 | 0.32 | 0.00 |  |
| samples/LargeKernelParameter | old | 0.21 | 0.00 | 0.00 | 0.21 | safety: the module holds __constant__ (the kernel calls none) 0.21 |
| samples/MC_EstimatePiInlineP | old | 1.39 | 0.00 | 0.00 | 1.39 | safety: the module holds __constant__, __device__ (the kernel calls none) 1.39 |
| samples/MC_EstimatePiInlineP | new | 1.49 | 0.00 | 1.48 | 0.00 |  |
| samples/MC_SingleAsianOptionP | old | 1.64 | 0.00 | 0.00 | 1.64 | safety: the module holds __constant__, __device__ (the kernel calls none) 1.64 |
| samples/MC_SingleAsianOptionP | new | 1.66 | 0.00 | 1.66 | 0.00 |  |
| samples/bf16TensorCoreGemm | old | 56.53 | 0.00 | 0.00 | 56.53 | class: never 56.53 |
| samples/bf16TensorCoreGemm | new | 56.59 | 0.00 | 56.59 | 0.00 |  |
| samples/convolutionSeparable | old | 4.51 | 0.00 | 0.00 | 4.51 | safety: the module holds __constant__ (the kernel calls none) 4.51 |
| samples/convolutionSeparable | new | 4.52 | 3.99 | 0.54 | 0.00 |  |
| samples/cudaTensorCoreGemm | old | 5.21 | 0.00 | 0.00 | 5.21 | class: never 5.21 |
| samples/cudaTensorCoreGemm | new | 5.54 | 0.00 | 5.54 | 0.00 |  |
| samples/dct8x8 | old | 0.40 | 0.00 | 0.00 | 0.40 | safety: texture objects 0.24; safety: the module holds __constant__ (the kernel calls none) 0.15 |
| samples/dct8x8 | new | 0.39 | 0.00 | 0.16 | 0.23 | class: never 0.23 |
| samples/histogram | old | 4.80 | 0.00 | 0.04 | 4.76 | class: never 4.76 |
| samples/histogram | new | 4.87 | 4.31 | 0.56 | 0.00 |  |
| samples/interval | old | 5.76 | 0.00 | 0.00 | 5.76 | class: never 5.76 |
| samples/interval | new | 5.78 | 5.49 | 0.28 | 0.00 |  |
| samples/jacobiCudaGraphs | old | 7.95 | 0.00 | 0.00 | 7.95 | safety: inside a graph with conditionals 7.95; class: never 0.00 |
| samples/jacobiCudaGraphs | new | 7.66 | 7.41 | 0.25 | 0.00 |  |
| samples/mergeSort | old | 3.28 | 0.00 | 0.07 | 3.22 | class: never 3.22 |
| samples/mergeSort | new | 3.17 | 2.29 | 0.88 | 0.00 |  |
| samples/radixSortThrust | old | 14.00 | 0.00 | 0.00 | 14.00 | safety: the module holds __device__, string literals (the kernel calls none) 14.00 |
| samples/radixSortThrust | new | 14.16 | 13.89 | 0.27 | 0.00 |  |
| samples/reduction | old | 3.44 | 0.00 | 0.00 | 3.44 | safety: the module holds __device__ (the kernel calls none) 3.44 |
| samples/reduction | new | 3.47 | 3.19 | 0.28 | 0.00 |  |
| samples/segmentationTreeThrust | old | 6.79 | 0.00 | 0.00 | 6.79 | safety: the module holds __device__, string literals (the kernel calls none) 6.79 |
| samples/segmentationTreeThrust | new | 6.75 | 2.45 | 4.30 | 0.00 |  |
| samples/shfl_scan | old | 0.03 | 0.00 | 0.02 | 0.01 | class: never 0.01 |
| samples/shfl_scan | new | 0.03 | 0.00 | 0.03 | 0.00 |  |
| samples/simpleAWBarrier | old | 0.45 | 0.00 | 0.00 | 0.45 | safety: cooperative or cluster launch 0.45 |
| samples/simpleAWBarrier | new | 0.46 | 0.00 | 0.00 | 0.46 | safety: cooperative or cluster launch 0.46 |
| samples/simpleCudaGraphs | old | 0.57 | 0.00 | 0.00 | 0.57 | safety: inside a graph with conditionals 0.57 |
| samples/simpleCudaGraphs | new | 0.57 | 0.28 | 0.28 | 0.00 |  |
| samples/simpleIPC | old | 0.11 | 0.00 | 0.00 | 0.11 | safety: argument points into mapped host memory 0.11 |
| samples/simpleIPC | new | 0.12 | 0.00 | 0.00 | 0.12 | safety: argument points into mapped host memory 0.12 |
| samples/simpleLayeredTexture | old | 0.08 | 0.00 | 0.00 | 0.08 | safety: texture objects 0.08 |
| samples/simpleLayeredTexture | new | 0.08 | 0.00 | 0.00 | 0.08 | class: never 0.08 |
| samples/simpleSurfaceWrite | old | 0.03 | 0.00 | 0.00 | 0.03 | safety: texture objects 0.03 |
| samples/simpleSurfaceWrite | new | 0.04 | 0.00 | 0.00 | 0.04 | class: never 0.04 |
| samples/sortingNetworks | old | 5.04 | 4.00 | 0.37 | 0.67 | class: never 0.67 |
| samples/sortingNetworks | new | 4.96 | 4.47 | 0.49 | 0.00 |  |
| samples/tf32TensorCoreGemm | old | 80.74 | 0.00 | 0.00 | 80.74 | class: never 80.74 |
| samples/tf32TensorCoreGemm | new | 83.31 | 0.00 | 83.31 | 0.00 |  |
| samples/threadFenceReduction | old | 0.14 | 0.00 | 0.00 | 0.14 | safety: the module holds __device__ (the kernel calls none) 0.14 |
| samples/threadFenceReduction | new | 0.14 | 0.00 | 0.14 | 0.00 |  |
| samples/warpAggregatedAtomicsCG | old | 0.80 | 0.00 | 0.00 | 0.80 | safety: the module holds __device__ (the kernel calls none) 0.80 |

