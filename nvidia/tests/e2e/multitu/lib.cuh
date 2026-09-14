// A second translation unit, which is the whole point of -rdc=true: device
// code that is compiled separately and linked. Nothing here is reachable
// without cross-unit resolution -- the kernel in main.cu calls functions
// defined only in lib.cu and reads a __constant__ declared only there.
#pragma once
__device__ int scale(int x, int k);
__device__ float fscale(float x, float k);
__device__ int biased(int i);
extern __constant__ int lib_bias[4];
