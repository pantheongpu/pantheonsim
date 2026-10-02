// Managed memory through the driver API, as bindings that load libcuda alone
// use it (cudarc's alloc_unified): cuMemAllocManaged, stream attachment,
// prefetches and advice, the pointer attributes, and a kernel and the host
// reading and writing the same bytes. The expected answers are what an RTX
// 3080 Ti's driver gives. Also cuLaunchHostFunc, which runs the function
// once the work queued before it is done, and cuEventElapsedTime's refusals
// (as an RTX 3060 gives them).
#include <cuda.h>

#include <cstdio>
#include <cstring>
#include <initializer_list>

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

// out[i] = in[i] * 2 + 1 for i < n.
static const char* kPtx = R"(
.version 7.0
.target sm_80
.address_size 64
.visible .entry twice_plus_one(.param .u64 in, .param .u64 out, .param .u32 n)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [n];
    setp.ge.u32 %p, %r1, %r2;
    @%p ret;
    ld.param.u64 %rd1, [in];
    ld.param.u64 %rd2, [out];
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.u32 %r3, [%rd4];
    shl.b32 %r4, %r3, 1;
    add.u32 %r5, %r4, 1;
    st.global.u32 [%rd5], %r5;
    ret;
}
)";

// A module with __managed__ globals, as nvcc emits them for
// "__managed__ int counter = 5; __managed__ float scaled[4] = {1, 2, 3, 4};",
// and a kernel that updates both.
static const char* kManagedPtx = R"(
.version 7.0
.target sm_80
.address_size 64
.global .attribute(.managed) .align 4 .u32 counter = 5;
.global .attribute(.managed) .align 4 .b8 scaled[16] = {0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64};
.global .align 4 .u32 plain = 9;
.visible .entry bump()
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .f32 %f<3>;
    .reg .b64 %rd<4>;
    mov.u32 %r1, %tid.x;
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra scale;
    ld.global.u32 %r2, [counter];
    add.u32 %r3, %r2, 10;
    st.global.u32 [counter], %r3;
scale:
    mov.u64 %rd1, scaled;
    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.f32 %f2, %f1, %f1;
    st.global.f32 [%rd3], %f2;
    ret;
}
)";

static int host_calls = 0;
static void on_host(void* p) { host_calls += *static_cast<int*>(p); }

int main() {
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) || cuDeviceGet(&dev, 0) || cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx)) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  int v = -1;
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev);
  check(v == 1, "managed memory supported");
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, dev);
  check(v == 1, "concurrent managed access");

  CUdeviceptr z = 0, a = 0, b = 0, d = 0;
  IS(cuMemAllocManaged(&z, 0, CU_MEM_ATTACH_GLOBAL), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemAllocManaged(&z, 64, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemAllocManaged(&z, 64, CU_MEM_ATTACH_SINGLE), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemAllocManaged(&a, 400, CU_MEM_ATTACH_GLOBAL), CUDA_SUCCESS);
  IS(cuMemAllocManaged(&b, 400, CU_MEM_ATTACH_HOST), CUDA_SUCCESS);
  IS(cuMemAlloc(&d, 256), CUDA_SUCCESS);
  check(a % 256 == 0, "managed allocation aligned");

  unsigned mt = 0;
  int managed = -1;
  CUdeviceptr dp = 0;
  void* hp = nullptr;
  IS(cuPointerGetAttribute(&mt, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, a), CUDA_SUCCESS);
  check(mt == CU_MEMORYTYPE_DEVICE, "managed memory reports device memory");
  IS(cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, a), CUDA_SUCCESS);
  check(managed == 1, "managed pointer is managed");
  IS(cuPointerGetAttribute(&dp, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, a + 8), CUDA_SUCCESS);
  check(dp == a + 8, "device pointer is the address");
  IS(cuPointerGetAttribute(&hp, CU_POINTER_ATTRIBUTE_HOST_POINTER, a), CUDA_SUCCESS);
  check(reinterpret_cast<CUdeviceptr>(hp) == a, "host pointer is the address");
  managed = -1;
  IS(cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, d), CUDA_SUCCESS);
  check(managed == 0, "device memory is not managed");

  CUstream s;
  cuStreamCreate(&s, 0);
  IS(cuStreamAttachMemAsync(s, a, 0, CU_MEM_ATTACH_SINGLE), CUDA_SUCCESS);
  IS(cuStreamAttachMemAsync(s, a, 400, CU_MEM_ATTACH_GLOBAL), CUDA_SUCCESS);
  IS(cuStreamAttachMemAsync(s, a, 200, CU_MEM_ATTACH_GLOBAL), CUDA_ERROR_INVALID_VALUE);
  IS(cuStreamAttachMemAsync(s, a + 64, 0, CU_MEM_ATTACH_GLOBAL), CUDA_ERROR_INVALID_VALUE);
  IS(cuStreamAttachMemAsync(s, d, 0, CU_MEM_ATTACH_GLOBAL), CUDA_ERROR_INVALID_VALUE);
  IS(cuStreamAttachMemAsync(s, a, 0, 7), CUDA_ERROR_INVALID_VALUE);

  // The location forms arrived in CUDA 12.2; hosted CI builds against 12.0.
#if CUDA_VERSION >= 12020
  CUmemLocation loc{};
  loc.type = CU_MEM_LOCATION_TYPE_DEVICE;
  loc.id = 0;
  IS(cuMemPrefetchAsync_v2(a, 400, loc, 0, s), CUDA_SUCCESS);
  IS(cuMemPrefetchAsync_v2(a + 100, 200, loc, 0, s), CUDA_SUCCESS);
  IS(cuMemPrefetchAsync_v2(a, 5000, loc, 0, s), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemPrefetchAsync_v2(d, 256, loc, 0, s), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemAdvise_v2(a, 400, CU_MEM_ADVISE_SET_READ_MOSTLY, loc), CUDA_SUCCESS);
  IS(cuMemAdvise_v2(d, 256, CU_MEM_ADVISE_SET_READ_MOSTLY, loc), CUDA_ERROR_INVALID_VALUE);
  loc.id = 64;
  IS(cuMemPrefetchAsync_v2(a, 400, loc, 0, s), CUDA_ERROR_INVALID_DEVICE);
  loc.type = CU_MEM_LOCATION_TYPE_HOST;
  loc.id = 0;
  IS(cuMemPrefetchAsync_v2(a, 400, loc, 0, s), CUDA_SUCCESS);
#endif

  // The host writes, a kernel reads and writes, the host reads.
  unsigned* ha = reinterpret_cast<unsigned*>(a);
  for (unsigned i = 0; i < 100; ++i) ha[i] = i;
  CUmodule mod;
  CUfunction fn;
  IS(cuModuleLoadData(&mod, kPtx), CUDA_SUCCESS);
  IS(cuModuleGetFunction(&fn, mod, "twice_plus_one"), CUDA_SUCCESS);
  unsigned n = 100;
  void* args[] = {&a, &b, &n};
  IS(cuLaunchKernel(fn, 1, 1, 1, 128, 1, 1, 0, s, args, nullptr), CUDA_SUCCESS);
  IS(cuStreamSynchronize(s), CUDA_SUCCESS);
  bool right = true;
  const unsigned* hb = reinterpret_cast<const unsigned*>(b);
  for (unsigned i = 0; i < 100; ++i) right = right && hb[i] == 2 * i + 1;
  check(right, "a kernel's writes to managed memory reach the host");

  IS(cuMemsetD8(a, 7, 400), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(reinterpret_cast<unsigned char*>(a)[399] == 7, "cuMemsetD8 on managed memory, seen by the host");
  unsigned char buf[4] = {};
  IS(cuMemcpyDtoH(buf, a, 4), CUDA_SUCCESS);
  check(buf[0] == 7 && buf[3] == 7, "cuMemcpyDtoH from managed memory");

  int by = 3;
  IS(cuLaunchHostFunc(s, on_host, &by), CUDA_SUCCESS);
  IS(cuStreamSynchronize(s), CUDA_SUCCESS);
  check(host_calls == 3, "cuLaunchHostFunc ran the function once");

  // Elapsed time needs two recorded events that keep timing.
  CUevent nt, e1, e2, never;
  float ms = 0;
  IS(cuEventCreate(&nt, CU_EVENT_DISABLE_TIMING), CUDA_SUCCESS);
  IS(cuEventCreate(&e1, 0), CUDA_SUCCESS);
  IS(cuEventCreate(&e2, 0), CUDA_SUCCESS);
  IS(cuEventCreate(&never, 0), CUDA_SUCCESS);
  IS(cuEventRecord(nt, s), CUDA_SUCCESS);
  IS(cuEventRecord(e1, s), CUDA_SUCCESS);
  IS(cuEventRecord(e2, s), CUDA_SUCCESS);
  IS(cuEventSynchronize(e2), CUDA_SUCCESS);
  IS(cuEventElapsedTime(&ms, e1, e2), CUDA_SUCCESS);
  IS(cuEventElapsedTime(&ms, nt, e2), CUDA_ERROR_INVALID_HANDLE);
  IS(cuEventElapsedTime(&ms, e1, nt), CUDA_ERROR_INVALID_HANDLE);
  IS(cuEventElapsedTime(&ms, never, e2), CUDA_ERROR_INVALID_HANDLE);
  IS(cuEventElapsedTime(nullptr, e1, e2), CUDA_ERROR_INVALID_VALUE);
  for (CUevent e : {nt, e1, e2, never}) cuEventDestroy(e);

  IS(cuMemFree(b), CUDA_SUCCESS);
  IS(cuMemFree(a), CUDA_SUCCESS);
  cuMemFree(d);
  cuModuleUnload(mod);

  // A module's __managed__ globals, loaded through the driver: managed memory
  // at the address cuModuleGetGlobal returns, holding the initial values, that
  // the host reads and writes in place and a kernel's writes reach -- as an
  // RTX 3060 has it. They used to be ordinary device memory.
  {
    CUmodule mm = nullptr;
    CUfunction bump = nullptr;
    CUdeviceptr counter = 0, scaled = 0, plain = 0, base = 0;
    size_t bytes = 0, range = 0;
    IS(cuModuleLoadData(&mm, kManagedPtx), CUDA_SUCCESS);
    IS(cuModuleGetFunction(&bump, mm, "bump"), CUDA_SUCCESS);
    IS(cuModuleGetGlobal(&counter, &bytes, mm, "counter"), CUDA_SUCCESS);
    check(bytes == 4, "the managed global's size");
    IS(cuModuleGetGlobal(&scaled, &bytes, mm, "scaled"), CUDA_SUCCESS);
    check(bytes == 16, "the managed array's size");
    IS(cuModuleGetGlobal(&plain, nullptr, mm, "plain"), CUDA_SUCCESS);
    int is_managed = -1;
    IS(cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, counter), CUDA_SUCCESS);
    check(is_managed == 1, "a managed global is managed memory");
    is_managed = -1;
    IS(cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, plain), CUDA_SUCCESS);
    check(is_managed == 0, "a plain global is not");
    IS(cuMemGetAddressRange(&base, &range, counter), CUDA_SUCCESS);
    check(base == counter && range == 4, "its allocation is the variable");
    auto* hc = reinterpret_cast<unsigned*>(counter);
    auto* hs = reinterpret_cast<float*>(scaled);
    check(*hc == 5 && hs[1] == 2.0f, "the host reads the initial values in place");
    *hc = 100;
    IS(cuLaunchKernel(bump, 1, 1, 1, 4, 1, 1, 0, nullptr, nullptr, nullptr), CUDA_SUCCESS);
    IS(cuCtxSynchronize(), CUDA_SUCCESS);
    check(*hc == 110, "the kernel saw the host's write, and the host sees the kernel's");
    check(hs[0] == 2.0f && hs[1] == 4.0f && hs[2] == 6.0f && hs[3] == 8.0f, "the managed array, scaled in place");
    unsigned via_copy = 0;
    IS(cuMemcpyDtoH(&via_copy, counter, sizeof via_copy), CUDA_SUCCESS);
    check(via_copy == 110, "a copy reads it too");
    IS(cuModuleUnload(mm), CUDA_SUCCESS);
  }
  cuStreamDestroy(s);
  std::printf(failures ? "FAIL: %d managed-memory checks\n" : "PASS: every managed-memory check\n", failures);
  return failures ? 1 : 0;
}
