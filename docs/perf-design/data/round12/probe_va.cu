// E4(c)/E5: can a passthrough keep the simulator's pointer values on the real device? The simulator's device pointers live at
// 0x2000'0000'0000 + device*1TiB (include/vgpu/memory.hpp). Reserve exactly that address on the A10G with the VMM API, back it with
// real memory, run a kernel through the same pointer value and read it back. Throwaway probe (native only; links the real libcuda).
// Build: nvcc -arch=sm_86 probe_va.cu -o probe_va -lcuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#define DK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s = ""; cuGetErrorName(r_, &s); printf("  %s -> %s\n", #x, s); } } while (0)
__global__ void fill(unsigned* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = 0xC0DE0000u + i; }
int main() {
  cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0); CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, dev); cuCtxSetCurrent(ctx);
  size_t gran = 0; CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
  cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  printf("granularity %zu\n", gran);
  struct T { unsigned long long hint; size_t size; const char* what; } tests[] = {
    {0x200000000000ull, 64ull << 20, "device 0 window base (0x2000'0000'0000), 64 MiB"},
    {0x210000000000ull, 64ull << 20, "device 1 window base (0x2100'0000'0000), 64 MiB"},
    {0x200000000000ull + (3ull << 30), 1ull << 30, "device 0 window + 3 GiB, 1 GiB"},
    {0x200000000000ull + (500ull << 30), 8ull << 30, "device 0 window + 500 GiB, 8 GiB (a 141 GB-card style address)"},
    {0x6ffb00000000ull, 2ull << 20, "simulator's function window 0x6ffb'0000'0000"},
    {0x7ffe00000000ull, 2ull << 20, "high user-space style address 0x7ffe'0000'0000"},
  };
  for (auto& t : tests) {
    CUdeviceptr p = 0; CUresult r = cuMemAddressReserve(&p, t.size, 0, t.hint, 0);
    bool exact = r == CUDA_SUCCESS && p == t.hint;
    printf("reserve %s -> %s, got 0x%llx %s\n", t.what, r == CUDA_SUCCESS ? "ok" : "ERROR", (unsigned long long)p, exact ? "(EXACT)" : "(different address)");
    if (r != CUDA_SUCCESS) continue;
    if (exact && t.size <= (1ull << 30)) {
      CUmemGenericAllocationHandle h; CUresult c = cuMemCreate(&h, t.size, &prop, 0);
      if (c == CUDA_SUCCESS) {
        DK(cuMemMap(p, t.size, 0, h, 0));
        CUmemAccessDesc ad = {}; ad.location = prop.location; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        DK(cuMemSetAccess(p, t.size, &ad, 1));
        int n = 1 << 16; fill<<<n / 256, 256>>>((unsigned*)p, n); cudaError_t e = cudaDeviceSynchronize();
        std::vector<unsigned> o(n); DK(cuMemcpyDtoH(o.data(), p, n * 4));
        bool ok = e == cudaSuccess; for (int i = 0; i < n; ++i) if (o[i] != 0xC0DE0000u + i) { ok = false; break; }
        printf("  mapped, kernel wrote through the same pointer value, host read back: %s\n", ok ? "OK" : "WRONG");
        cuMemUnmap(p, t.size); cuMemRelease(h);
      } else printf("  cuMemCreate failed %d\n", c);
    }
    cuMemAddressFree(p, t.size);
  }
  return 0;
}
