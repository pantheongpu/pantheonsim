// cooperative_groups tiles of more than one warp (tiled_partition<64> and
// up). From sm_80 they keep their barriers and exchange slots in the driver's
// reserved shared memory, which the kernel finds through
// %reserved_smem_offset_1 (cooperative_groups/details/memory.h), so the
// registers have to name a region that is really there, past the kernel's own
// shared memory, and the kernel's own data must survive the tiles' use of it.
//
// Built at -O3 by run_cg_multi_warp_tiles.sh (-G also needs barriers inside
// device calls, which come separately); also reads the
// reserved-region registers directly and checks the layout an RTX 3060
// reports: begin, offset 0 and offset 1 at %total_smem_size (the kernel's
// shared memory in 128-byte units), end 288 bytes on, cap 1 KiB on.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

constexpr int kThreads = 256;
constexpr int kOut = 7;

__global__ void tiles(int* out, unsigned* layout) {
  extern __shared__ int dyn[];
  __shared__ int guard[100];   // 400 bytes: total_smem_size rounds up
  const int t = threadIdx.x;
  if (t < 100) guard[t] = 0x5a5a0000 + t;
  dyn[t] = t * 7;
  cg::thread_block block = cg::this_thread_block();
  block.sync();

  auto t64 = cg::tiled_partition<64>(block);
  auto t128 = cg::tiled_partition<128>(block);
  int* o = out + t * kOut;
  o[0] = cg::reduce(t64, t, cg::plus<int>());
  o[1] = cg::reduce(t128, t & 15, cg::greater<int>()) + 100 * cg::reduce(t128, 1, cg::plus<int>());
  o[2] = t64.shfl(t * 3, 63);          // the tile's last thread's value
  t64.sync();
  o[3] = static_cast<int>(t64.meta_group_rank() * 1000 + t64.thread_rank());
  o[4] = t128.any(t == 200) + 2 * t128.all(t < 128);
  block.sync();
  // The kernel's own shared memory is untouched by the tiles' scratch.
  o[5] = t < 100 ? guard[t] - 0x5a5a0000 : t;
  o[6] = dyn[t] / 7;

  if (t == 0) {
    unsigned v[6];
    asm volatile("mov.u32 %0, %%reserved_smem_offset_begin;" : "=r"(v[0]));
    asm volatile("mov.u32 %0, %%reserved_smem_offset_end;" : "=r"(v[1]));
    asm volatile("mov.u32 %0, %%reserved_smem_offset_cap;" : "=r"(v[2]));
    asm volatile("mov.u32 %0, %%reserved_smem_offset_0;" : "=r"(v[3]));
    asm volatile("mov.u32 %0, %%reserved_smem_offset_1;" : "=r"(v[4]));
    asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(v[5]));
    for (int i = 0; i < 6; ++i) layout[i] = v[i];
  }
}

int main() {
  int* d;
  unsigned* dl;
  cudaMalloc(&d, kThreads * kOut * sizeof(int));
  cudaMalloc(&dl, 6 * sizeof(unsigned));
  const int dyn_bytes = kThreads * sizeof(int);
  tiles<<<2, kThreads, dyn_bytes>>>(d, dl);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("kernel failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  int h[kThreads * kOut];
  unsigned l[6];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  cudaMemcpy(l, dl, sizeof l, cudaMemcpyDeviceToHost);

  int bad = 0;
  auto check = [&](int t, int slot, int want, const char* what) {
    const int got = h[t * kOut + slot];
    if (got != want && bad++ < 20) std::printf("thread %d, %s: got %d, want %d\n", t, what, got, want);
  };
  for (int t = 0; t < kThreads; ++t) {
    const int b64 = t / 64 * 64, b128 = t / 128 * 128;
    check(t, 0, 64 * b64 + 64 * 63 / 2, "reduce over a 64-thread tile");
    check(t, 1, 15 + 100 * 128, "reduce over a 128-thread tile");
    check(t, 2, (b64 + 63) * 3, "shfl across a 64-thread tile");
    check(t, 3, (t / 64) * 1000 + t % 64, "meta_group_rank and thread_rank");
    check(t, 4, (b128 == 128 ? 1 : 0) + 2 * (b128 == 0 ? 1 : 0), "any and all over 128 threads");
    check(t, 5, t, "the kernel's static shared memory");
    check(t, 6, t, "the kernel's dynamic shared memory");
  }
  // 400 static bytes, dynamic from offset 400 (16-byte aligned), 1024 bytes:
  // 1424, in 128-byte units 1536.
  const unsigned total = 1536;
  const unsigned want[6] = {total, total + 0x120, total + 0x400, total, total, total};
  const char* names[6] = {"begin", "end", "cap", "offset_0", "offset_1", "total_smem_size"};
  for (int i = 0; i < 6; ++i)
    if (l[i] != want[i] && bad++ < 30)
      std::printf("%%reserved_smem/%s: got %#x, want %#x\n", names[i], l[i], want[i]);
  if (bad) {
    std::printf("%d mismatches\nFAIL\n", bad);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
