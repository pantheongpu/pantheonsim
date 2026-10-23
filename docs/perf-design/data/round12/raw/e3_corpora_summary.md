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

Programs that run on both and differ, by profile (first differing line):

* **l4** (21)
  * `hecbench/aobench-cuda` differs: output files: ao.ppm
  * `hecbench/s3d-cuda` numeric-close: 
  * `rodinia/backprop` differs: output files: output.dat
  * `rodinia/gaussian` differs: -Total Global Memory			 - 23131776 KB | +Total Global Memory			 - 23106688 KB
  * `rodinia/particlefilter_float` differs: -XE: 26.303270 | -YE: 35.531096
  * `rodinia/srad_v2` differs: -1.69188 2.06711 1.87182 1.34890 1.12669 1.69617 2.51154 1.19081 1.58601 1.33976 1.57482 1.32192 2.00494 2.137 | -1.42858 1.79608 2.11331 1.75094 1.44
  * `samples/MC_SingleAsianOptionP` numeric-close: 
  * `samples/binaryPartitionCG` differs: -Launching 160 blocks with 768 threads... | +Launching 116 blocks with 768 threads...
  * `samples/cudaCompressibleMemory` differs: -allocating non-compressible Z buffer
  * `samples/cudaGraphsPerfScaling` differs: -20, 1, 0, 22.421, 37.742, 10.861, 30.842, 3.620, 21.041, 17.408, 0.000, 16.384, 0.000, 9.770, 2.048, 1.000, | +20, 1, 0, 31.211, 20.592, 83.184, 83.6
  * `samples/fp16ScalarProduct` differs: -Result native operators	: 646406.000000 | -Result intrinsics	: 646406.000000
  * `samples/graphMemoryNodes` differs: -Check verified that d_negSquare and d_input share a virtual address. | -Check verified that d_negSquare and d_input share a virtual address.
  * `samples/ptxjit` differs: -ptxas info    : 0 bytes gmem | -ptxas info    : Function properties for myKernel
  * `samples/reductionMultiBlockCG` differs: -numBlocks: 80 | +numBlocks: 58
  * `samples/simpleAWBarrier` differs: -Launching normVecByDotProductAWBarrier kernel with numBlocks = 160 blockSize = 768 | +Launching normVecByDotProductAWBarrier kernel with numBlocks = 
  * `samples/simpleAssert` differs: -/home/saqib/.cache/cuda-samples/Samples/0_Introduction/simpleAssert/simpleAssert.cu:64: void testKernel(int): | -/home/saqib/.cache/cuda-samples/Samp
  * `samples/simpleAssert_nvrtc` differs: -./simpleAssert_kernel.cu:38: void testKernel(int): block: [1,0,0], thread: [28,0,0] Assertion 'gtid < N' fail | -./simpleAssert_kernel.cu:38: void te
  * `samples/simpleOccupancy` differs: -Potential occupancy: 33.3333% | +Potential occupancy: 50%
  * `samples/simplePrintf` differs: +[0, 0]:		Value is:10 | +[0, 1]:		Value is:10
  * `samples/simpleStreams` differs: -kernel:		21.47 | -non-streamed:	5.36
  * `samples/systemWideAtomics` differs: -CAN access pageable memory | +CANNOT access pageable memory
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
* **rtx5090** (20)
  * `hecbench/aobench-cuda` differs: output files: ao.ppm
  * `hecbench/s3d-cuda` numeric-close: 
  * `rodinia/backprop` differs: output files: output.dat
  * `rodinia/gaussian` differs: -Total Global Memory			 - 23131776 KB | +Total Global Memory			 - 33389568 KB
  * `rodinia/particlefilter_float` differs: -XE: 26.303270 | -YE: 35.531096
  * `rodinia/srad_v2` differs: -1.69188 2.06711 1.87182 1.34890 1.12669 1.69617 2.51154 1.19081 1.58601 1.33976 1.57482 1.32192 2.00494 2.137 | -1.42858 1.79608 2.11331 1.75094 1.44
  * `samples/MC_SingleAsianOptionP` numeric-close: 
  * `samples/binaryPartitionCG` differs: -Launching 160 blocks with 768 threads... | +Launching 340 blocks with 768 threads...
  * `samples/cudaCompressibleMemory` differs: -allocating non-compressible Z buffer
  * `samples/fp16ScalarProduct` differs: -Result native operators	: 646406.000000 | -Result intrinsics	: 646406.000000
  * `samples/graphMemoryNodes` differs: -Check verified that d_negSquare and d_input share a virtual address. | -Check verified that d_negSquare and d_input share a virtual address.
  * `samples/ptxjit` differs: -ptxas info    : 0 bytes gmem | -ptxas info    : Function properties for myKernel
  * `samples/reductionMultiBlockCG` differs: -numBlocks: 80 | +numBlocks: 170
  * `samples/simpleAWBarrier` differs: -Launching normVecByDotProductAWBarrier kernel with numBlocks = 160 blockSize = 768 | +Launching normVecByDotProductAWBarrier kernel with numBlocks = 
  * `samples/simpleAssert` differs: -/home/saqib/.cache/cuda-samples/Samples/0_Introduction/simpleAssert/simpleAssert.cu:64: void testKernel(int): | -/home/saqib/.cache/cuda-samples/Samp
  * `samples/simpleAssert_nvrtc` differs: -./simpleAssert_kernel.cu:38: void testKernel(int): block: [1,0,0], thread: [28,0,0] Assertion 'gtid < N' fail | -./simpleAssert_kernel.cu:38: void te
  * `samples/simpleOccupancy` differs: -Potential occupancy: 33.3333% | +Potential occupancy: 50%
  * `samples/simplePrintf` differs: -[1, 0]:		Value is:10 | -[1, 1]:		Value is:10
  * `samples/simpleStreams` differs: -kernel:		21.47 | -non-streamed:	5.36
  * `samples/systemWideAtomics` differs: -CAN access pageable memory | +CANNOT access pageable memory
* **a100** (14)
  * `samples/MC_SingleAsianOptionP` numeric-close: 
  * `samples/binaryPartitionCG` differs: -Launching 160 blocks with 768 threads... | +Launching 216 blocks with 1024 threads...
  * `samples/cudaGraphsPerfScaling` differs: -20, 1, 0, 22.421, 37.742, 10.861, 30.842, 3.620, 21.041, 17.408, 0.000, 16.384, 0.000, 9.770, 2.048, 1.000, | +20, 1, 0, 18.011, 18.681, 80.344, 80.7
  * `samples/fp16ScalarProduct` differs: -Result native operators	: 646406.000000 | -Result intrinsics	: 646406.000000
  * `samples/graphMemoryNodes` differs: -Check verified that d_negSquare and d_input share a virtual address. | -Check verified that d_negSquare and d_input share a virtual address.
  * `samples/ptxjit` differs: -ptxas info    : 0 bytes gmem | -ptxas info    : Function properties for myKernel
  * `samples/reductionMultiBlockCG` differs: -numThreads: 768 | -numBlocks: 80
  * `samples/simpleAWBarrier` differs: -Launching normVecByDotProductAWBarrier kernel with numBlocks = 160 blockSize = 768 | +Launching normVecByDotProductAWBarrier kernel with numBlocks = 
  * `samples/simpleAssert` differs: -/home/saqib/.cache/cuda-samples/Samples/0_Introduction/simpleAssert/simpleAssert.cu:64: void testKernel(int): | -/home/saqib/.cache/cuda-samples/Samp
  * `samples/simpleAssert_nvrtc` differs: -./simpleAssert_kernel.cu:38: void testKernel(int): block: [1,0,0], thread: [28,0,0] Assertion 'gtid < N' fail | -./simpleAssert_kernel.cu:38: void te
  * `samples/simpleOccupancy` differs: -Potential occupancy: 33.3333% | +Potential occupancy: 50%
  * `samples/simplePrintf` differs: -[1, 0]:		Value is:10 | -[1, 1]:		Value is:10
  * `samples/simpleStreams` differs: -kernel:		21.47 | -non-streamed:	5.36
  * `samples/systemWideAtomics` differs: -CAN access pageable memory | +CANNOT access pageable memory
* **t4** (13)
  * `samples/MC_SingleAsianOptionP` numeric-close: 
  * `samples/binaryPartitionCG` differs: -Launching 160 blocks with 768 threads... | +Launching 40 blocks with 1024 threads...
  * `samples/cudaGraphsPerfScaling` differs: -20, 1, 0, 22.421, 37.742, 10.861, 30.842, 3.620, 21.041, 17.408, 0.000, 16.384, 0.000, 9.770, 2.048, 1.000, | +20, 1, 0, 26.562, 25.761, 98.556, 99.1
  * `samples/fp16ScalarProduct` differs: -Result native operators	: 646406.000000 | -Result intrinsics	: 646406.000000
  * `samples/graphMemoryNodes` differs: -Check verified that d_negSquare and d_input share a virtual address. | -Check verified that d_negSquare and d_input share a virtual address.
  * `samples/ptxjit` differs: -ptxas info    : 0 bytes gmem | -ptxas info    : Function properties for myKernel
  * `samples/reductionMultiBlockCG` differs: -numThreads: 768 | -numBlocks: 80
  * `samples/simpleAWBarrier` differs: -Launching normVecByDotProductAWBarrier kernel with numBlocks = 160 blockSize = 768 | +Launching normVecByDotProductAWBarrier kernel with numBlocks = 
  * `samples/simpleAssert` differs: -/home/saqib/.cache/cuda-samples/Samples/0_Introduction/simpleAssert/simpleAssert.cu:64: void testKernel(int): | -/home/saqib/.cache/cuda-samples/Samp
  * `samples/simpleAssert_nvrtc` differs: -./simpleAssert_kernel.cu:38: void testKernel(int): block: [1,0,0], thread: [28,0,0] Assertion 'gtid < N' fail | -./simpleAssert_kernel.cu:38: void te
  * `samples/simpleOccupancy` differs: -Potential occupancy: 33.3333% | +Potential occupancy: 50%
  * `samples/simplePrintf` differs: +[0, 0]:		Value is:10 | +[0, 1]:		Value is:10
  * `samples/systemWideAtomics` differs: -CAN access pageable memory | +CANNOT access pageable memory
