// Pre-CUDA-12 spellings the 2010s suites still use (Rodinia, PolyBench/GPU),
// force-included (-include) when they are built against CUDA 13.
#pragma once
#include <cuda_runtime.h>
#define cudaThreadSynchronize cudaDeviceSynchronize
