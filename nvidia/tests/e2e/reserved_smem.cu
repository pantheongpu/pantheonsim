// The block's reserved shared memory, and cooperative groups' large tiles.
//
// On sm_80 and later each block has 1 KiB of shared memory reserved past its
// own, and cooperative groups syncs and reduces a tile wider than a warp in it,
// at %reserved_smem_offset_1. The simulator did not know those registers, so
// every kernel with such a tile failed to launch (gunrock/loops' group-mapped
// SpMV). The expected offsets are what an RTX 3060 reports for the same
// kernels: past the static and dynamic shared memory, 128-byte aligned.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
namespace cg = cooperative_groups;

template <int N>
__global__ void offsets(unsigned* out, int ndyn) {
  __shared__ char user[N];
  extern __shared__ char dyn[];
  unsigned v[5];
  asm volatile("mov.u32 %0, %%reserved_smem_offset_begin;" : "=r"(v[0]));
  asm volatile("mov.u32 %0, %%reserved_smem_offset_end;" : "=r"(v[1]));
  asm volatile("mov.u32 %0, %%reserved_smem_offset_cap;" : "=r"(v[2]));
  asm volatile("mov.u32 %0, %%reserved_smem_offset_0;" : "=r"(v[3]));
  asm volatile("mov.u32 %0, %%reserved_smem_offset_1;" : "=r"(v[4]));
  // Volatile, so the compiler keeps the static array: without it the kernel
  // has no static shared memory at all.
  ((volatile char*)user)[threadIdx.x % N] = 1;
  if (ndyn) ((volatile char*)dyn)[0] = 1;   // with no dynamic bytes, dyn[0] is past the end
  if (threadIdx.x == 0)
    for (int i = 0; i < 5; ++i) out[i] = v[i];
}

// Two 128-thread tiles, each synced and reduced through the reserved memory,
// next to the kernel's own shared memory, which must come through untouched.
__global__ void tiles(int* out) {
  __shared__ int user[8];
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<128>(block);
  if (threadIdx.x < 8) user[threadIdx.x] = threadIdx.x;
  tile.sync();
  const int s = cg::reduce(tile, (int)threadIdx.x + user[threadIdx.x & 7], cg::plus<int>());
  if (tile.thread_rank() == 0) out[threadIdx.x / 128] = s;
}

template <int N>
int check(unsigned* d, int dyn, unsigned begin) {
  offsets<N><<<1, 32, dyn>>>(d, dyn);
  unsigned h[5] = {0};
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  const unsigned want[5] = {begin, begin + 288, begin + 1024, begin, begin};
  int bad = 0;
  for (int i = 0; i < 5; ++i)
    if (h[i] != want[i]) {
      std::printf("FAIL static %d dynamic %d: register %d is %u, want %u\n", N, dyn, i, h[i], want[i]);
      ++bad;
    }
  return bad;
}

int main() {
  unsigned* d;
  cudaMalloc(&d, 64);
  int bad = check<16>(d, 0, 128) + check<16>(d, 256, 384) + check<1000>(d, 0, 1024) +
            check<1000>(d, 4096, 5120) + check<8192>(d, 100, 8320);
  int* t;
  cudaMalloc(&t, 8);
  tiles<<<1, 256>>>(t);
  int h[2] = {0, 0};
  cudaMemcpy(h, t, sizeof h, cudaMemcpyDeviceToHost);
  int w0 = 0, w1 = 0;
  for (int i = 0; i < 128; ++i) {
    w0 += i + (i & 7);
    w1 += i + 128 + (i & 7);
  }
  if (h[0] != w0 || h[1] != w1) {
    std::printf("FAIL 128-thread tile reduce: %d %d, want %d %d\n", h[0], h[1], w0, w1);
    ++bad;
  }
  if (cudaGetLastError() != cudaSuccess) {
    std::printf("FAIL %s\n", cudaGetErrorString(cudaGetLastError()));
    ++bad;
  }
  std::printf(bad ? "FAILED\n" : "PASS\n");
  return bad ? 1 : 0;
}
