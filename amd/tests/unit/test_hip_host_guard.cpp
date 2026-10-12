// A HIP program that needs more host memory than the host can spare gets
// hipErrorOutOfMemory, as a program on a full card does, instead of growing the
// host's RAM until the kernel's OOM killer takes something else down. ctest runs
// this with VGPU_MEMORY_HOST_RESERVE_MB set to more memory than any host has, so
// the host-memory guard (include/vgpu/memory_backing.hpp) refuses every chunk
// of real memory; chunks that cost nothing are not refused.
#include <cstdint>
#include <vector>

#include "vgpu_hip.h"
#include "vtest.hpp"

namespace {
constexpr size_t kChunk = 64 * 1024;
}

VTEST(a_copy_into_device_memory_the_host_cannot_back_is_out_of_memory) {
  void* d = nullptr;
  VCHECK_EQ(static_cast<int>(hipMalloc(&d, 4 << 20)), static_cast<int>(hipSuccess));
  std::vector<uint8_t> host(1 << 20, 0x5a);
  VCHECK_EQ(static_cast<int>(hipMemcpy(d, host.data(), host.size(), hipMemcpyHostToDevice)),
            static_cast<int>(hipErrorOutOfMemory));
  VCHECK_EQ(static_cast<int>(hipGetLastError()), static_cast<int>(hipErrorOutOfMemory));
  // Not fatal: the program can free what it has and go on.
  VCHECK_EQ(static_cast<int>(hipFree(d)), static_cast<int>(hipSuccess));
  VCHECK_EQ(static_cast<int>(hipMalloc(&d, 1 << 20)), static_cast<int>(hipSuccess));
  VCHECK_EQ(static_cast<int>(hipFree(d)), static_cast<int>(hipSuccess));
}

VTEST(a_fill_of_whole_chunks_with_one_byte_costs_nothing_and_succeeds) {
  void* d = nullptr;
  VCHECK_EQ(static_cast<int>(hipMalloc(&d, 4 << 20)), static_cast<int>(hipSuccess));
  // From the start of the allocation: 32 whole chunks, held as tags.
  VCHECK_EQ(static_cast<int>(hipMemset(d, 0xAB, 32 * kChunk)), static_cast<int>(hipSuccess));
  uint8_t back[4] = {0, 0, 0, 0};
  // Reading real bytes back needs no new chunk either.
  VCHECK_EQ(static_cast<int>(hipMemcpy(back, d, sizeof back, hipMemcpyDeviceToHost)), static_cast<int>(hipSuccess));
  VCHECK_EQ(int(back[0]), 0xAB);
  VCHECK_EQ(static_cast<int>(hipFree(d)), static_cast<int>(hipSuccess));
}

VTEST_MAIN
