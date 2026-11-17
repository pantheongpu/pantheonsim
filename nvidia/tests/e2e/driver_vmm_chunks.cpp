// A range reserved with cuMemAddressReserve and mapped back to back from several physical chunks (cuMemCreate + cuMemMap), the layout PyTorch's
// expandable segments and vLLM's sleep mode use. On a card the whole range is one virtual allocation, so one copy, memset or device-to-device
// copy over it works. The simulator checked a bulk access against the first mapping only, and a copy over two or more chunks failed with
// "device memory read of N bytes ... runs past the end of the 2097152-byte mapping" (pantheonsim issue #686; all four copies of the issue's
// reproducer return 0 on an RTX 3080 Ti, driver 595.99.02). A gap in the range, or a chunk nothing mapped, must still be refused.
#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) || cuDeviceGet(&dev, 0) || cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx)) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  size_t gran = 0;
  IS(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_SUCCESS);
  const int chunks = 4;
  CUdeviceptr base = 0;
  IS(cuMemAddressReserve(&base, (chunks + 1) * gran, 0, 0, 0), CUDA_SUCCESS);   // (one more than is mapped: a chunk nothing maps)
  CUmemAccessDesc access = {};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUmemGenericAllocationHandle handles[chunks];
  for (int c = 0; c < chunks; ++c) {
    IS(cuMemCreate(&handles[c], gran, &prop, 0), CUDA_SUCCESS);
    IS(cuMemMap(base + c * gran, gran, 0, handles[c], 0), CUDA_SUCCESS);
    IS(cuMemSetAccess(base + c * gran, gran, &access, 1), CUDA_SUCCESS);
  }
  const size_t total = chunks * gran;

  // A host-to-device copy over every chunk, and back, byte for byte.
  std::vector<unsigned char> host(total), back(total, 0);
  for (size_t i = 0; i < total; ++i) host[i] = static_cast<unsigned char>(i * 2654435761u >> 24);
  IS(cuMemcpyHtoD(base, host.data(), total), CUDA_SUCCESS);
  for (int n = 1; n <= chunks; ++n) {
    std::fill(back.begin(), back.end(), 0);
    char what[64];
    std::snprintf(what, sizeof what, "a copy of %d chunk(s) from the start", n);
    check(cuMemcpyDtoH(back.data(), base, n * gran) == CUDA_SUCCESS && std::memcmp(back.data(), host.data(), n * gran) == 0, what);
  }
  // Starting in the middle of one chunk and ending in the middle of another.
  std::fill(back.begin(), back.end(), 0);
  const size_t from = gran / 2, len = 2 * gran;
  check(cuMemcpyDtoH(back.data(), base + from, len) == CUDA_SUCCESS && std::memcmp(back.data(), host.data() + from, len) == 0,
        "a copy that starts mid-chunk and ends mid-chunk");

  // A memset over every chunk (the pattern's phase is the byte's own), and one that starts mid-chunk.
  IS(cuMemsetD8(base, 0x5a, total), CUDA_SUCCESS);
  IS(cuMemcpyDtoH(back.data(), base, total), CUDA_SUCCESS);
  bool all = true;
  for (size_t i = 0; i < total; ++i) all &= back[i] == 0x5a;
  check(all, "a memset over every chunk");
  IS(cuMemsetD32(base + gran - 8, 0x01020304u, 4), CUDA_SUCCESS);   // (16 bytes: eight in this chunk, eight in the next)
  IS(cuMemcpyDtoH(back.data(), base + gran - 8, 16), CUDA_SUCCESS);
  bool word_pattern = true;
  for (int i = 0; i < 16; ++i) word_pattern &= back[i] == static_cast<unsigned char>(0x01020304u >> (8 * (i % 4)));
  check(word_pattern, "a 32-bit memset across a chunk boundary keeps its phase");

  // A device-to-device copy between two halves of the range, each over two chunks.
  IS(cuMemcpyHtoD(base, host.data(), total), CUDA_SUCCESS);
  IS(cuMemcpyDtoD(base + 2 * gran, base, 2 * gran), CUDA_SUCCESS);
  IS(cuMemcpyDtoH(back.data(), base, total), CUDA_SUCCESS);
  check(std::memcmp(back.data() + 2 * gran, host.data(), 2 * gran) == 0, "a device-to-device copy over two chunks");

  // Still refused: a copy that runs on into the chunk nothing maps.
  check(cuMemcpyDtoH(back.data(), base + (chunks - 1) * gran, 2 * gran) != CUDA_SUCCESS,
        "a copy that runs past the last mapped chunk into reserved space is refused");

  for (int c = 0; c < chunks; ++c) {
    cuMemUnmap(base + c * gran, gran);
    cuMemRelease(handles[c]);
  }
  cuMemAddressFree(base, (chunks + 1) * gran);
  IS(cuDevicePrimaryCtxRelease(dev), CUDA_SUCCESS);
  std::printf(failures ? "FAIL: %d checks\n" : "PASS: every chunked-range check\n", failures);
  return failures ? 1 : 0;
}
