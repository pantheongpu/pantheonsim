// E4(b): what a program reads of the device, and what happens when it sizes a launch from it.
// Build: nvcc -arch=sm_86 -cudart shared probe_props.cu -o probe_props ; run natively (A10G) and under VGPU_GPU=<profile>.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
extern __shared__ char dyn[];
__global__ void k(int* out) { if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = 1; dyn[0] = 1; } }
int main(int argc, char** argv) {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int cl = -1, optin = -1, ccmaj = 0, ccmin = 0, maxblk = 0;
  cudaDeviceGetAttribute(&cl, cudaDevAttrClusterLaunch, 0);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&maxblk, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
  printf("name=%s cc=%d.%d sms=%d smem_block=%zu smem_optin=%d smem_sm=%zu max_thr_sm=%d max_blk_sm=%d regs_sm=%d l2=%d mem_gib=%.1f bus=%d cluster=%d coop=%d async=%d\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.sharedMemPerBlock, optin, p.sharedMemPerMultiprocessor, p.maxThreadsPerMultiProcessor,
         maxblk, p.regsPerMultiprocessor, p.l2CacheSize, p.totalGlobalMem / 1073741824.0, p.memoryBusWidth, cl, p.cooperativeLaunch, p.asyncEngineCount);
  if (argc > 1) { optin = atoi(argv[1]); printf("(override: pretend the simulated profile reported opt-in smem=%d)\n", optin); }
  int* d; cudaMalloc(&d, 4);
  // (a) a launch sized from the reported opt-in shared memory
  cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
  printf("set_attr(%d B) -> %s\n", optin, cudaGetErrorName(e));
  k<<<1, 128, optin>>>(d); e = cudaGetLastError(); if (e == cudaSuccess) e = cudaDeviceSynchronize();
  printf("launch dyn_smem=%d -> %s\n", optin, cudaGetErrorName(e));
  // (b) 1024 threads x max resident blocks per SM x SM count (a persistent-kernel style grid)
  int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, 1024, 0);
  printf("occupancy blocks/SM @1024 thr = %d -> persistent grid %d blocks\n", nb, nb * p.multiProcessorCount);
  // (c) cluster launch support
  printf("clusterLaunch attr %d\n", cl);
  int dev = 0; cudaGetDevice(&dev);
  return 0;
}
