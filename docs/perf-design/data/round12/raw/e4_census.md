Programs traced: 164; programs with at least one traced call: 163; no ltrace output (timeout, static CUDA runtime, or failed): 1

| API family | distinct functions | total calls | programs using it |
|---|---:|---:|---:|
| runtime (compiler-generated registration/launch glue) | 9 | 203960 | 149 |
| CUDA runtime API | 112 | 150865 | 150 |
| CUDA driver API | 47 | 1110 | 16 |
| cuBLASLt | 5 | 112 | 1 |
| cuBLAS | 4 | 78 | 1 |
| NVRTC | 6 | 36 | 6 |
| cuRAND | 7 | 10 | 3 |

### CUDA runtime API: every function seen (calls over all programs / programs using it)

`cudaDeviceSynchronize` 37911/92, `cudaEventRecord` 20380/27, `cudaGetLastError` 18417/47, `cudaEventSynchronize` 10181/21, `cudaEventElapsedTime` 10171/24, `cudaMemcpy` 10082/115, `cudaStreamSynchronize` 5696/24, `cudaMemset` 5075/23, `cudaMemcpyAsync` 4439/27, `cudaFree` 3481/136, `cudaMalloc` 3293/133, `cudaGraphLaunch` 3019/6, `cudaGraphExecKernelNodeSetParams` 2954/1, `cudaEventCreate` 2927/24, `cudaPeekAtLastError` 2055/6, `cudaMemPrefetchAsync` 1722/1, `cudaMemsetAsync` 1281/9, `cudaGetDevice` 1186/17, `cudaMemcpyToSymbol` 1103/4, `cudaFreeAsync` 872/4, `cudaMallocAsync` 869/3, `cudaDeviceGetAttribute` 822/77, `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` 316/11, `cudaGetDeviceProperties` 236/68, `cudaFreeHost` 229/24, `cudaLaunchKernelExC` 219/1, `cudaHostAlloc` 206/17, `cudaMallocManaged` 184/3, `cudaSetDevice` 164/93, `cudaStreamDestroy` 148/16, `cudaStreamCreate` 129/12, `cudaStreamAttachMemAsync` 120/1, `cudaHostGetDevicePointer` 108/3, `cudaLaunchCooperativeKernel` 101/2, `cudaGetDeviceCount` 92/81, `cudaHostRegister` 43/2, `cudaHostUnregister` 43/2, `cudaGraphExecDestroy` 42/5, `cudaGraphDestroy` 40/5, `cudaGraphInstantiate` 39/5, `cudaGraphCreate` 34/5, `cudaDeviceGetGraphMemAttribute` 34/1, `cudaEventDestroy` 32/12, `cudaGraphAddMemAllocNode` 32/2, `cudaStreamCreateWithFlags` 31/13, `cudaGraphAddMemFreeNode` 24/2, `cudaMallocHost` 23/7, `cudaGraphAddKernelNode` 23/4, `cudaFuncGetAttributes` 20/15, `cudaGraphAddNode` 15/1, `cudaStreamWaitEvent` 14/4, `cudaEventQuery` 13/1, `cudaCreateTextureObject` 8/7, `cudaMemcpy2D` 8/2, `cudaStreamAddCallback` 8/1, `cudaFreeArray` 7/7, `cudaCreateChannelDesc` 7/7, `cudaStreamEndCapture` 7/4, `cudaDestroyTextureObject` 7/6, `cudaGraphInstantiateWithFlags` 6/1, `cudaEventCreateWithFlags` 6/3, `cudaMallocPitch` 6/2, `cudaDeviceGraphMemTrim` 6/1, `cudaGraphAddMemcpyNode1D` 6/1, `cudaFuncSetAttribute` 5/5, `cudaMallocArray` 5/5, `cudaStreamBeginCapture` 5/4, `cudaGraphConditionalHandleCreate` 5/1, `cudaDeviceSetLimit` 4/4, `cudaMemcpyToArray` 4/3, `cudaDriverGetVersion` 4/4, `cudaGraphAddMemcpyNode` 3/2, `cudaGraphAddMemsetNode` 3/2, `cudaDeviceGetDefaultMemPool` 2/2, `cudaMemPoolSetAttribute` 2/2, `cudaFuncSetCacheConfig` 2/2, `cudaStreamCreateWithPriority` 2/1, `cudaProfilerStop` 2/2, `cudaProfilerStart` 2/2, `cudaGraphUpload` 2/1, `cudaMemcpy2DToArray` 2/1, `cudaStreamBeginCaptureToGraph` 2/1, `cudaStreamGetCaptureInfo` 2/1, `cudaMalloc3DArray` 2/2, `cudaMemcpy3D` 2/2, `cudaGraphClone` 2/1, `cudaGraphGetNodes` 2/1, `cudaMemset2D` 2/1, `cudaSetDeviceFlags` 2/2, `cudaMemGetInfo` 1/1, `cudaDeviceGetStreamPriorityRange` 1/1, `cudaStreamUpdateCaptureDependencies` 1/1, `cudaGetErrorString` 1/1, `cudaStreamSetAttribute` 1/1, `cudaCtxResetPersistingL2Cache` 1/1, `cudaGraphAddHostNode` 1/1, `cudaLaunchHostFunc` 1/1, `cudaIpcOpenEventHandle` 1/1, `cudaIpcCloseMemHandle` 1/1, `cudaIpcOpenMemHandle` 1/1, `cudaIpcGetMemHandle` 1/1, `cudaIpcGetEventHandle` 1/1, `cudaCreateSurfaceObject` 1/1, `cudaDestroySurfaceObject` 1/1, `cudaMallocFromPoolAsync` 1/1, `cudaMemPoolDestroy` 1/1, `cudaMemPoolExportPointer` 1/1, `cudaMemPoolExportToShareableHandle` 1/1, `cudaMemPoolImportPointer` 1/1, `cudaMemPoolImportFromShareableHandle` 1/1, `cudaMemPoolCreate` 1/1, `cudaMemPoolGetAccess` 1/1

### CUDA driver API: every function seen (calls over all programs / programs using it)

`cuLaunchKernel` 315/14, `cuCtxSynchronize` 307/7, `cuDeviceGetAttribute` 128/14, `cuDeviceGet` 39/14, `cuInit` 29/14, `cuDeviceGetName` 26/12, `cuMemAlloc_v2` 20/9, `cuMemFree_v2` 20/9, `cuDeviceGetCount` 20/13, `cuMemGetAllocationGranularity` 19/3, `cuModuleGetFunction` 15/14, `cuCtxCreate_v4` 14/13, `cuModuleLoadData` 13/13, `cuCtxGetDevice` 13/1, `cuMemcpyHtoD_v2` 12/7, `cuMemcpyDtoH_v2` 11/10, `cuMemRelease` 11/3, `cuMemUnmap` 10/3, `cuMemSetAccess` 10/3, `cuMemCreate` 10/3, `cuMemAddressFree` 10/3, `cuMemAddressReserve` 10/3, `cuMemMap` 10/3, `cuCtxDestroy_v2` 8/7, `cuCtxPopCurrent_v2` 3/1, `cuMemGetAllocationPropertiesFromHandle` 2/1, `cuOccupancyMaxPotentialBlockSize` 2/1, `cuStreamSynchronize` 2/1, `cuModuleUnload` 2/2, `cuCtxPushCurrent_v2` 2/1, `cuDeviceTotalMem_v2` 1/1, `cuStreamCreate` 1/1, `cuMemExportToShareableHandle` 1/1, `cuModuleLoadDataEx` 1/1, `cuMemcpyDtoHAsync_v2` 1/1, `cuMemImportFromShareableHandle` 1/1, `cuStreamDestroy_v2` 1/1, `cuOccupancyMaxActiveBlocksPerMultiprocessor` 1/1, `cuLinkAddData_v2` 1/1, `cuLinkCreate_v2` 1/1, `cuLinkComplete` 1/1, `cuLinkDestroy` 1/1, `cuArrayCreate_v2` 1/1, `cuArrayDestroy` 1/1, `cuMemcpy2D_v2` 1/1, `cuTexObjectDestroy` 1/1, `cuTexObjectCreate` 1/1

### runtime (compiler-generated registration/launch glue): every function seen (calls over all programs / programs using it)

`__cudaLaunchKernel` 67195/141, `__cudaPopCallConfiguration` 67195/141, `__cudaPushCallConfiguration` 67195/141, `__cudaRegisterFunction` 982/148, `__cudaRegisterVar` 475/23, `__cudaGetKernel` 433/141, `__cudaRegisterFatBinary` 162/149, `__cudaRegisterFatBinaryEnd` 162/149, `__cudaUnregisterFatBinary` 161/148

### NVRTC

`nvrtcCompileProgram` 6/6, `nvrtcGetProgramLogSize` 6/6, `nvrtcGetProgramLog` 6/6, `nvrtcCreateProgram` 6/6, `nvrtcGetCUBINSize` 6/6, `nvrtcGetCUBIN` 6/6

### cuBLAS

`cublasDgemv_v2` 34/1, `cublasSetStream_v2` 34/1, `cublasDestroy_v2` 5/1, `cublasCreate_v2` 5/1

### cuBLASLt

`cublasLtDDDMatmulAlgoGetHeuristic` 34/1, `cublasLtDDDMatmul` 34/1, `cublasLtGetEnvironmentMode` 34/1, `cublasLtCtxInit` 5/1, `cublasLtShutdownCtx` 5/1

### cuRAND

`curandGenerateUniform` 2/2, `curandCreateGenerator` 2/2, `curandDestroyGenerator` 2/2, `curandGetDirectionVectors32` 1/1, `curandSetPseudoRandomGeneratorSeed` 1/1, `curandSetQuasiRandomGeneratorDimensions` 1/1, `curandSetGeneratorOrdering` 1/1

### other symbols matched by the pattern (libc / thrust etc.)


