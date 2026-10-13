// A CUDA program that needs more host memory than the host can spare gets
// cudaErrorMemoryAllocation, as a program on a full card does, instead of
// growing the host's RAM until the kernel's OOM killer takes something else
// down. ctest runs this with VGPU_MEMORY_HOST_RESERVE_MB set to more memory
// than any host has, so the host-memory guard (include/vgpu/memory_backing.hpp)
// refuses every chunk of real memory; chunks that cost nothing are not refused.
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

#include "vtest.hpp"

namespace {
constexpr size_t kChunk = 64 * 1024;
}

VTEST(a_copy_into_device_memory_the_host_cannot_back_is_out_of_memory) {
  void* d = nullptr;
  VCHECK_EQ(static_cast<int>(cudaMalloc(&d, 4 << 20)), static_cast<int>(cudaSuccess));
  std::vector<uint8_t> host(1 << 20, 0x5a);
  VCHECK_EQ(static_cast<int>(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice)),
            static_cast<int>(cudaErrorMemoryAllocation));
  VCHECK_EQ(static_cast<int>(cudaGetLastError()), static_cast<int>(cudaErrorMemoryAllocation));
  // Not fatal, and not sticky: the program can free what it has and go on.
  VCHECK_EQ(static_cast<int>(cudaFree(d)), static_cast<int>(cudaSuccess));
  VCHECK_EQ(static_cast<int>(cudaMalloc(&d, 1 << 20)), static_cast<int>(cudaSuccess));
  VCHECK_EQ(static_cast<int>(cudaFree(d)), static_cast<int>(cudaSuccess));
}

VTEST(a_fill_of_whole_chunks_with_one_byte_costs_nothing_and_succeeds) {
  void* d = nullptr;
  VCHECK_EQ(static_cast<int>(cudaMalloc(&d, 4 << 20)), static_cast<int>(cudaSuccess));
  // From the start of the allocation: 32 whole chunks, held as tags.
  VCHECK_EQ(static_cast<int>(cudaMemset(d, 0xAB, 32 * kChunk)), static_cast<int>(cudaSuccess));
  uint8_t back[4] = {0, 0, 0, 0};
  VCHECK_EQ(static_cast<int>(cudaMemcpy(back, d, sizeof back, cudaMemcpyDeviceToHost)), static_cast<int>(cudaSuccess));
  VCHECK_EQ(int(back[0]), 0xAB);
  VCHECK_EQ(static_cast<int>(cudaFree(d)), static_cast<int>(cudaSuccess));
}

VTEST_MAIN
