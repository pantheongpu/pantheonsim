// The few runtime calls the numerics programs make, under one name for CUDA (nvcc) and HIP (hipcc).
#pragma once
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuToDevice hipMemcpyHostToDevice
#define gpuToHost hipMemcpyDeviceToHost
#define gpuSync hipDeviceSynchronize
#define gpuOk hipSuccess
#define GPU_API "hip"
#define GPU_IS_HIP 1
#else
#include <cuda_runtime.h>
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuToDevice cudaMemcpyHostToDevice
#define gpuToHost cudaMemcpyDeviceToHost
#define gpuSync cudaDeviceSynchronize
#define gpuOk cudaSuccess
#define GPU_API "cuda"
#define GPU_IS_HIP 0
#endif
