// The shared memory the driver reserves in every block, as each API reports
// it: cuDeviceGetAttribute(111), cudaDeviceGetAttribute, cudaDeviceProp, and
// the SM's whole shared memory beside it. An RTX 3060 says 1024 and 102400; a
// T4 (compute capability 7.5) says 0 and 65536. Prints one line.
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0)) { puts("error: no device"); return 1; }
  int rt = -1, sm = -1, drv = -1;
  cudaDeviceGetAttribute(&rt, cudaDevAttrReservedSharedMemoryPerBlock, 0);
  cudaDeviceGetAttribute(&sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  CUdevice d;
  cuInit(0);
  cuDeviceGet(&d, 0);
  cuDeviceGetAttribute(&drv, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, d);
  printf("reserved %d %d %zu sm %d %zu\n", drv, rt, p.reservedSharedMemPerBlock, sm, p.sharedMemPerMultiprocessor);
  return 0;
}
