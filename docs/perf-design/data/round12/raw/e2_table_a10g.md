| case | native vs native | native vs sim (run 1) | native vs sim (run 2) | sim vs sim | verdict | note |
|---|---|---|---|---|---|---|
| hecbench/adam-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/aidw-cuda | timing-only | B-fails | timing-only | A-fails | bit-identical |  |
| hecbench/aligned-types-cuda | timing-only | B-fails | B-fails | both-fail-same | sim: timeout | timeout |
| hecbench/aobench-cuda | timing-only | differs | differs | timing-only | differs: SFU / fast-math arithmetic | ao.ppm bytes differ; built with -use_fast_math (ex2/sqrt .approx), the documented SFU divergence (memory note hecbench-on-pantheonsim) |
| hecbench/aop-cuda | identical | timing-only | - | - | bit-identical |  |
| hecbench/asmooth-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/asyncAllocation-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/atan2-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/atomicAggregate-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/atomicCost-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/atomicIntrinsics-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/atomicPerf-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/atomicReduction-cuda | timing-only | B-fails | timing-only | A-fails | bit-identical |  |
| hecbench/attention-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/axpby-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/babelstream-cuda | differs | B-fails | differs | A-fails | native run-to-run: timing the filter missed | bandwidth columns |
| hecbench/background-subtract-cuda | identical | identical | - | - | bit-identical |  |
| hecbench/backprop-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/base64e-cuda | differs | B-fails | differs | A-fails | native run-to-run: timing the filter missed | measured overlap percentage |
| hecbench/bgmv-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/bh-cuda | timing-only | B-fails | B-fails | both-fail-same | sim: timeout | timeout |
| hecbench/bilateral-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/binomial-cuda | timing-only | B-fails | B-fails | both-fail-same | sim: timeout | timeout |
| hecbench/bitonic-sort-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/black-scholes-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/blockAccess-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/fpdc-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/perplexity-cuda | identical | timing-only | - | - | bit-identical |  |
| hecbench/radixsort2-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/s3d-cuda | timing-only | numeric-close | numeric-close | timing-only | differs: SFU / fast-math arithmetic | numbers agree to ~1e-8 relative; expf compiles to ex2.approx |
| hecbench/shmembench-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/silu-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/sortKV-cuda | timing-only | timing-only | - | - | bit-identical |  |
| hecbench/zerocopy-cuda | timing-only | timing-only | - | - | bit-identical |  |
| polybench/2DCONV | timing-only | timing-only | - | - | bit-identical |  |
| polybench/2MM | timing-only | B-fails | timing-only | A-fails | bit-identical |  |
| polybench/3DCONV | timing-only | timing-only | - | - | bit-identical |  |
| polybench/3MM | timing-only | timing-only | - | - | bit-identical |  |
| polybench/ADI | timing-only | timing-only | - | - | bit-identical |  |
| polybench/ATAX | timing-only | timing-only | - | - | bit-identical |  |
| polybench/BICG | timing-only | timing-only | - | - | bit-identical |  |
| polybench/CORR | timing-only | timing-only | - | - | bit-identical |  |
| polybench/COVAR | timing-only | timing-only | - | - | bit-identical |  |
| polybench/FDTD-2D | timing-only | timing-only | - | - | bit-identical |  |
| polybench/GEMM | timing-only | timing-only | - | - | bit-identical |  |
| polybench/GEMVER | timing-only | timing-only | - | - | bit-identical |  |
| polybench/GESUMMV | timing-only | timing-only | - | - | bit-identical |  |
| polybench/GRAMSCHM | timing-only | timing-only | - | - | bit-identical |  |
| polybench/JACOBI1D | timing-only | timing-only | - | - | bit-identical |  |
| polybench/JACOBI2D | timing-only | timing-only | - | - | bit-identical |  |
| polybench/LU | timing-only | timing-only | - | - | bit-identical |  |
| polybench/MVT | timing-only | timing-only | - | - | bit-identical |  |
| polybench/SYR2K | timing-only | timing-only | - | - | bit-identical |  |
| polybench/SYRK | timing-only | timing-only | - | - | bit-identical |  |
| rodinia/backprop | differs | differs | differs | differs | native nondeterministic (time-seeded / uninitialised) | output.dat differs between two card runs |
| rodinia/bfs | identical | identical | - | - | bit-identical |  |
| rodinia/gaussian | timing-only | timing-only | - | - | bit-identical |  |
| rodinia/hotspot | identical | identical | - | - | bit-identical |  |
| rodinia/lavaMD | timing-only | timing-only | - | - | bit-identical |  |
| rodinia/lud | timing-only | timing-only | - | - | bit-identical |  |
| rodinia/needle | identical | identical | - | - | bit-identical |  |
| rodinia/nn | identical | identical | - | - | bit-identical |  |
| rodinia/particlefilter_float | differs | differs | differs | differs | native nondeterministic (time-seeded / uninitialised) | estimate differs between two card runs (memory note: varies even on the card) |
| rodinia/pathfinder | identical | identical | - | - | bit-identical |  |
| rodinia/sc_gpu | timing-only | timing-only | - | - | bit-identical |  |
| rodinia/srad_v2 | identical | differs | differs | identical | differs: program reads out of bounds | srad_v2 reads the row above its image; the card returns what happens to be there, the simulator something else (memory note rodinia-on-pantheonsim); card run-to-run identical |
| samples/LargeKernelParameter | timing-only | timing-only | - | - | bit-identical |  |
| samples/MC_EstimatePiInlineP | timing-only | timing-only | - | - | bit-identical |  |
| samples/MC_EstimatePiInlineQ | timing-only | timing-only | - | - | bit-identical |  |
| samples/MC_EstimatePiP | timing-only | differs | differs | timing-only | differs: cuRAND stream is not bit-equal | GPU result 3.13632 (card) vs 3.13900 (sim): the simulator's cuRAND batch generator draws different numbers; sim vs sim identical |
| samples/MC_EstimatePiQ | timing-only | timing-only | - | - | bit-identical |  |
| samples/MC_SingleAsianOptionP | timing-only | numeric-close | numeric-close | timing-only | differs: SFU / fast-math arithmetic | 6th significant digit of a Monte Carlo price (cuRAND + expf); sim vs sim identical |
| samples/StreamPriorities | timing-only | timing-only | - | - | bit-identical |  |
| samples/UnifiedMemoryPerf | differs | B-fails | B-fails | both-fail-same | native run-to-run: timing the filter missed | a table of times; sim timed out |
| samples/UnifiedMemoryStreams | differs | differs | differs | differs | native nondeterministic (time-seeded / uninitialised) | random choice of host/device per task |
| samples/alignedTypes | timing-only | timing-only | - | - | bit-identical |  |
| samples/asyncAPI | timing-only | timing-only | - | - | bit-identical |  |
| samples/bf16TensorCoreGemm | timing-only | B-fails | timing-only | A-fails | bit-identical |  |
| samples/binaryPartitionCG | identical | identical | - | - | bit-identical |  |
| samples/cdpAdvancedQuicksort | timing-only | timing-only | - | - | bit-identical |  |
| samples/cdpBezierTessellation | identical | identical | - | - | bit-identical |  |
| samples/cdpQuadtree | identical | identical | - | - | bit-identical |  |
| samples/cdpSimplePrint | differs | differs | differs | identical | native nondeterministic (time-seeded / uninitialised) | order of device printf from child grids differs between two card runs; sim vs sim identical |
| samples/cdpSimpleQuicksort | identical | identical | - | - | bit-identical |  |
| samples/clock | timing-only | timing-only | - | - | bit-identical |  |
| samples/clock_nvrtc | timing-only | timing-only | - | - | bit-identical |  |
| samples/convolutionSeparable | timing-only | timing-only | - | - | bit-identical |  |
| samples/convolutionTexture | timing-only | timing-only | - | - | bit-identical |  |
| samples/cudaCompressibleMemory | timing-only | timing-only | - | - | bit-identical |  |
| samples/cudaGraphsPerfScaling | differs | differs | differs | differs | native run-to-run: timing the filter missed | a CSV of per-launch microseconds |
| samples/cudaOpenMP | identical | identical | - | - | bit-identical |  |
| samples/cudaTensorCoreGemm | timing-only | timing-only | - | - | bit-identical |  |
| samples/dct8x8 | timing-only | timing-only | - | - | bit-identical |  |
| samples/dmmaTensorCoreGemm | timing-only | B-fails | B-fails | both-fail-same | sim: timeout | timeout |
| samples/eigenvalues | timing-only | timing-only | - | - | bit-identical |  |
| samples/fp16ScalarProduct | differs | differs | differs | differs | native nondeterministic (time-seeded / uninitialised) | the input is random: three card runs give three results |
| samples/globalToShmemAsyncCopy | timing-only | B-fails | B-fails | both-fail-same | sim: timeout | timeout |
| samples/graphConditionalNodes | identical | differs | differs | identical | differs: reported identity (driver version) | host driver reports 13.2, simulator 13.0 (VGPU_DRIVER_VERSION / VGPU_CUDA_VERSION would change it) |
| samples/graphMemoryFootprint | differs | B-fails | differs | A-fails | native run-to-run: timing the filter missed | memory-footprint bytes the driver reports vary run to run on the card (402653184 vs 469762048); sim timed out |
| samples/graphMemoryNodes | identical | differs | differs | identical | differs: reported identity (driver version) | host driver reports 13.2, simulator 13.0 |
| samples/histogram | timing-only | timing-only | - | - | bit-identical |  |
| samples/immaTensorCoreGemm | timing-only | timing-only | - | - | bit-identical |  |
| samples/inlinePTX | identical | identical | - | - | bit-identical |  |
| samples/inlinePTX_nvrtc | identical | identical | - | - | bit-identical |  |
| samples/interval | timing-only | timing-only | - | - | bit-identical |  |
| samples/jacobiCudaGraphs | timing-only | timing-only | - | - | bit-identical |  |
| samples/matrixMul | timing-only | timing-only | - | - | bit-identical |  |
| samples/matrixMulDrv | timing-only | timing-only | - | - | bit-identical |  |
| samples/matrixMulDynlinkJIT | identical | identical | - | - | bit-identical |  |
| samples/matrixMul_nvrtc | identical | identical | - | - | bit-identical |  |
| samples/memMapIPCDrv | timing-only | timing-only | - | - | bit-identical |  |
| samples/mergeSort | timing-only | timing-only | - | - | bit-identical |  |
| samples/newdelete | identical | identical | - | - | bit-identical |  |
| samples/ptxjit | differs | differs | differs | identical | native nondeterministic (time-seeded / uninitialised) | prints ptxas info of a JIT whose driver cache is warm on the second run |
| samples/radixSortThrust | timing-only | timing-only | - | - | bit-identical |  |
| samples/reduction | timing-only | timing-only | - | - | bit-identical |  |
| samples/reductionMultiBlockCG | timing-only | timing-only | - | - | bit-identical |  |
| samples/scalarProd | timing-only | timing-only | - | - | bit-identical |  |
| samples/scan | timing-only | B-fails | timing-only | A-fails | bit-identical |  |
| samples/segmentationTreeThrust | timing-only | timing-only | - | - | bit-identical |  |
| samples/shfl_scan | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleAWBarrier | identical | identical | - | - | bit-identical |  |
| samples/simpleAssert | identical | differs | differs | identical | differs: device printf / assert record order | same 4 assert messages, different thread order; sim vs sim identical |
| samples/simpleAssert_nvrtc | identical | differs | differs | identical | differs: device printf / assert record order | same assert messages, different thread order; sim vs sim identical |
| samples/simpleAtomicIntrinsics | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleAtomicIntrinsics_nvrtc | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleAttributes | timing-only | B-fails | B-fails | both-fail-same | sim: timeout | timeout |
| samples/simpleCallback | identical | identical | - | - | bit-identical |  |
| samples/simpleCooperativeGroups | identical | identical | - | - | bit-identical |  |
| samples/simpleCubemapTexture | identical | timing-only | - | - | bit-identical |  |
| samples/simpleCudaGraphs | identical | identical | - | - | bit-identical |  |
| samples/simpleDrvRuntime | identical | identical | - | - | bit-identical |  |
| samples/simpleHyperQ | identical | timing-only | - | - | bit-identical |  |
| samples/simpleIPC | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleLayeredTexture | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleMPI | both-fail-differ | both-fail-differ | both-fail-differ | both-fail-differ | fails on both (differently) | rc=1/1 |
| samples/simpleMultiCopy | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleMultiGPU | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleOccupancy | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleP2P | both-fail-same | both-fail-same | both-fail-same | both-fail-same | fails the same way on both | rc=2/2 |
| samples/simplePitchLinearTexture | timing-only | timing-only | - | - | bit-identical |  |
| samples/simplePrintf | identical | differs | differs | differs | differs: device printf / assert record order | block order of device printf lines differs; with 5 host threads the simulator differs from itself too |
| samples/simpleStreams | timing-only | differs | differs | differs | differs: timing the filter missed | kernel / non-streamed milliseconds printed without a time word; sim vs sim differs too |
| samples/simpleSurfaceWrite | identical | timing-only | - | - | bit-identical |  |
| samples/simpleTemplates | timing-only | timing-only | - | - | bit-identical |  |
| samples/simpleTexture | identical | timing-only | - | - | bit-identical |  |
| samples/simpleTextureDrv | identical | timing-only | - | - | bit-identical |  |
| samples/simpleVoteIntrinsics | identical | identical | - | - | bit-identical |  |
| samples/simpleZeroCopy | identical | identical | - | - | bit-identical |  |
| samples/sortingNetworks | timing-only | timing-only | - | - | bit-identical |  |
| samples/streamOrderedAllocation | timing-only | timing-only | - | - | bit-identical |  |
| samples/streamOrderedAllocationIPC | timing-only | B-fails | B-fails | both-fail-same | sim: capability gap | simulator reports no memory-pool IPC support and skips (exit 2); the card runs it |
| samples/streamOrderedAllocationP2P | both-fail-same | both-fail-same | both-fail-same | both-fail-same | fails the same way on both | rc=2/2 |
| samples/systemWideAtomics | identical | differs | differs | identical | differs: reported identity (capability) | pageableMemoryAccess: 1 on this host (HMM), fixed 0 in nvidia/src/device_attributes.cpp:185 |
| samples/template | timing-only | timing-only | - | - | bit-identical |  |
| samples/tf32TensorCoreGemm | timing-only | timing-only | - | - | bit-identical |  |
| samples/threadFenceReduction | timing-only | timing-only | - | - | bit-identical |  |
| samples/threadMigration | timing-only | timing-only | - | - | bit-identical |  |
| samples/transpose | timing-only | timing-only | - | - | bit-identical |  |
| samples/vectorAdd | identical | identical | - | - | bit-identical |  |
| samples/vectorAddDrv | identical | identical | - | - | bit-identical |  |
| samples/vectorAddMMAP | identical | identical | - | - | bit-identical |  |
| samples/vectorAdd_nvrtc | identical | identical | - | - | bit-identical |  |
| samples/warpAggregatedAtomicsCG | identical | identical | - | - | bit-identical |  |

* bit-identical: 131
* sim: timeout: 6
* native nondeterministic (time-seeded / uninitialised): 6
* native run-to-run: timing the filter missed: 5
* differs: SFU / fast-math arithmetic: 3
* differs: device printf / assert record order: 3
* differs: reported identity (driver version): 2
* fails the same way on both: 2
* differs: program reads out of bounds: 1
* differs: cuRAND stream is not bit-equal: 1
* fails on both (differently): 1
* differs: timing the filter missed: 1
* sim: capability gap: 1
* differs: reported identity (capability): 1
* total: 164
