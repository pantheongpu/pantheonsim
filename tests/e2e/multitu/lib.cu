#include "lib.cuh"
__device__ int scale(int x, int k) { return x * k + 1; }
__device__ float fscale(float x, float k) { return x * k + 0.5f; }
__constant__ int lib_bias[4];
__device__ int biased(int i) { return scale(i, 2) + lib_bias[i % 4]; }
