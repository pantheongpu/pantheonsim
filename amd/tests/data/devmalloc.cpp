// Device-side malloc and free (hipcc --genco --offload-arch=gfx942 devmalloc.cpp): the device
// library's allocator asks the host for its slabs and large blocks through the hostcall's
// device-memory service (amd/src/hostcall.cpp).
#include <hip/hip_runtime.h>

// Each thread takes a small block, fills it, sums it and gives it back.
extern "C" __global__ void small_blocks(int* out) {
  int* p = (int*)malloc(64);
  if (!p) {
    out[blockIdx.x * blockDim.x + threadIdx.x] = -1;
    return;
  }
  for (int i = 0; i < 16; ++i) p[i] = threadIdx.x + i;
  int s = 0;
  for (int i = 0; i < 16; ++i) s += p[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  free(p);
}

// One thread of each block takes a block of 3 MiB, which the allocator does not serve from a
// slab, writes its end and reads it back; and the same pointer is wanted again after free.
extern "C" __global__ void large_block(long long* out) {
  if (threadIdx.x != 0) return;
  long long total = 0;
  for (int round = 0; round < 3; ++round) {
    char* p = (char*)malloc(3u << 20);
    if (!p) {
      out[blockIdx.x] = -1;
      return;
    }
    p[0] = (char)(round + 1);
    p[(3u << 20) - 1] = (char)(blockIdx.x + 10);
    total += p[0] + p[(3u << 20) - 1];
    free(p);
  }
  out[blockIdx.x] = total;
}
