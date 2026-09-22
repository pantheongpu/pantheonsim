// The virtual memory management API, the way a framework's allocator uses it:
// reserve address space, create physical memory, map it, grant access, run a
// kernel over it, then grow the buffer in place by mapping a second handle
// right after the first. That last part is the whole point of the API and what
// PyTorch's expandable segments and NCCL's windows do -- a buffer that grows
// without moving, so no pointer a kernel already holds becomes stale.
//
// Compiled against NVIDIA's cuda.h and linked against libvgpucuda, like
// driver_abi.cu: the structs and enumerators here are the vendor header's, so a
// layout this shim reads differently shows up as a wrong answer.
#include <cuda.h>
#include <cstdio>
#include <vector>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
  const char* n = nullptr; cuGetErrorName(r_, &n); \
  printf("FAIL %s -> %s\n", #x, n ? n : "?"); return 1; } } while (0)
#define WANT(x, want) do { CUresult r_ = (x); if (r_ != (want)) { \
  const char* n = nullptr; cuGetErrorName(r_, &n); \
  printf("FAIL %s -> %s, expected %s\n", #x, n ? n : "?", #want); return 1; } } while (0)

// Adds one to n unsigned ints, so the test can tell which bytes a kernel saw.
static const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry bump(.param .u64 p, .param .u32 n)
{
  .reg .pred %p<2>;
  .reg .b32 %r<6>;
  .reg .b64 %rd<5>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r2, [n];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %tid.x;
  setp.ge.s32 %p1, %r1, %r2;
  @%p1 bra DONE;
  mul.wide.s32 %rd3, %r1, 4;
  add.s64 %rd4, %rd2, %rd3;
  ld.global.u32 %r3, [%rd4];
  add.s32 %r4, %r3, 1;
  st.global.u32 [%rd4], %r4;
DONE:
  ret;
}
)";

int main() {
  CUdevice dev;
  CUcontext ctx;
  CK(cuInit(0));
  CK(cuDeviceGet(&dev, 0));
  // cuCtxCreate resolves to a different versioned symbol per toolkit: CUDA 13
  // renames it to cuCtxCreate_v4, which takes a parameter block.
#if CUDA_VERSION >= 13000
  CK(cuCtxCreate(&ctx, nullptr, 0, dev));
#else
  CK(cuCtxCreate(&ctx, 0, dev));
#endif

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  size_t granularity = 0;
  CK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (granularity == 0 || (granularity & (granularity - 1))) {
    printf("FAIL granularity %zu is not a power of two\n", granularity);
    return 1;
  }

  // Room for two chunks, so the buffer can grow into the second.
  CUdeviceptr base = 0;
  CK(cuMemAddressReserve(&base, 2 * granularity, 0, 0, 0));

  // Reserved, nothing mapped: a copy into it has to fail rather than land
  // somewhere. This is the check that a silent success would have hidden.
  std::vector<unsigned> host(granularity / sizeof(unsigned), 0);
  if (cuMemcpyHtoD(base, host.data(), sizeof(unsigned)) == CUDA_SUCCESS) {
    printf("FAIL a copy into unmapped reserved space succeeded\n");
    return 1;
  }

  CUmemGenericAllocationHandle first = 0;
  CK(cuMemCreate(&first, granularity, &prop, 0));
  CK(cuMemMap(base, granularity, 0, first, 0));

  // Mapped, but no access granted yet: still not usable.
  if (cuMemcpyHtoD(base, host.data(), sizeof(unsigned)) == CUDA_SUCCESS) {
    printf("FAIL a copy succeeded before cuMemSetAccess\n");
    return 1;
  }

  CUmemAccessDesc rw{};
  rw.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  rw.location.id = 0;
  rw.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CK(cuMemSetAccess(base, granularity, &rw, 1));

  unsigned long long flags = 0;
  CK(cuMemGetAccess(&flags, &rw.location, base));
  if (flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
    printf("FAIL cuMemGetAccess reports %llu\n", flags);
    return 1;
  }

  // A kernel over the mapped memory.
  const int n = 64;
  for (int i = 0; i < n; ++i) host[i] = static_cast<unsigned>(i);
  CK(cuMemcpyHtoD(base, host.data(), n * sizeof(unsigned)));

  CUmodule mod;
  CUfunction fn;
  CK(cuModuleLoadData(&mod, kPtx));
  CK(cuModuleGetFunction(&fn, mod, "bump"));
  int count = n;
  void* args[] = {&base, &count};
  CK(cuLaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, nullptr, args, nullptr));
  CK(cuCtxSynchronize());
  std::vector<unsigned> back(n, 0);
  CK(cuMemcpyDtoH(back.data(), base, n * sizeof(unsigned)));
  for (int i = 0; i < n; ++i) {
    if (back[i] != static_cast<unsigned>(i) + 1) {
      printf("FAIL mapped memory: back[%d] = %u, expected %u\n", i, back[i], i + 1);
      return 1;
    }
  }

  // Grow in place: a second handle mapped at the end of the first, which is
  // what makes the API worth having. The pointer the kernel already used stays
  // valid and its contents stay put.
  CUmemGenericAllocationHandle second = 0;
  CK(cuMemCreate(&second, granularity, &prop, 0));
  CK(cuMemMap(base + granularity, granularity, 0, second, 0));
  CK(cuMemSetAccess(base + granularity, granularity, &rw, 1));
  const unsigned marker = 0xA5A5A5A5u;
  CK(cuMemcpyHtoD(base + granularity, &marker, sizeof marker));
  unsigned got_first = 0, got_second = 0;
  CK(cuMemcpyDtoH(&got_first, base, sizeof got_first));
  CK(cuMemcpyDtoH(&got_second, base + granularity, sizeof got_second));
  if (got_first != 1 || got_second != marker) {
    printf("FAIL after growing: first %u (want 1), second %#x (want %#x)\n", got_first, got_second,
           marker);
    return 1;
  }

  // The handle behind an address, and what it says about itself.
  CUmemGenericAllocationHandle retained = 0;
  CK(cuMemRetainAllocationHandle(&retained, reinterpret_cast<void*>(base)));
  CUmemAllocationProp got{};
  CK(cuMemGetAllocationPropertiesFromHandle(&got, retained));
  if (got.type != CU_MEM_ALLOCATION_TYPE_PINNED || got.location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
      got.location.id != 0) {
    printf("FAIL properties from handle: type %d location %d/%d\n", static_cast<int>(got.type),
           static_cast<int>(got.location.type), got.location.id);
    return 1;
  }
  CK(cuMemRelease(retained));

  // What a device refuses. An unaligned size, a mapping outside the
  // reservation, and freeing address space that is still mapped.
  CUdeviceptr scratch = 0;
  WANT(cuMemAddressReserve(&scratch, granularity + 1, 0, 0, 0), CUDA_ERROR_INVALID_VALUE);
  CUmemGenericAllocationHandle odd = 0;
  WANT(cuMemCreate(&odd, granularity - 1, &prop, 0), CUDA_ERROR_INVALID_VALUE);
  WANT(cuMemAddressFree(base, 2 * granularity), CUDA_ERROR_INVALID_VALUE);

  // Unmap the first chunk: the address stops working, the second stays.
  CK(cuMemUnmap(base, granularity));
  if (cuMemcpyDtoH(&got_first, base, sizeof got_first) == CUDA_SUCCESS) {
    printf("FAIL reading an unmapped address succeeded\n");
    return 1;
  }
  CK(cuMemcpyDtoH(&got_second, base + granularity, sizeof got_second));
  if (got_second != marker) {
    printf("FAIL the still-mapped half changed: %#x\n", got_second);
    return 1;
  }

  CK(cuMemUnmap(base + granularity, granularity));
  CK(cuMemRelease(first));
  CK(cuMemRelease(second));
  CK(cuMemAddressFree(base, 2 * granularity));
  CK(cuModuleUnload(mod));
  CK(cuCtxDestroy(ctx));
  printf("PASS\n");
  return 0;
}
