// The driver shim is written against a clean-room header (include/vgpu_cuda.h),
// not NVIDIA's cuda.h, so nothing in the build checks the two agree. This does:
// it is compiled against the vendor header and linked against libvgpucuda, so a
// signature that has drifted shows up as a link error or a wrong answer rather
// than as a segfault in somebody's program.
//
// It also pins the versioned spellings. CUDA 13 maps cuCtxCreate onto
// cuCtxCreate_v4 and cuEventElapsedTime onto cuEventElapsedTime_v2, and a shim
// missing those exports fails to load at all.
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
  const char* n = nullptr; cuGetErrorName(r_, &n); \
  printf("FAIL %s -> %s\n", #x, n ? n : "?"); return 1; } } while (0)

static const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry add_one(.param .u64 p, .param .u32 n)
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
  CK(cuInit(0));
  int count = 0;
  CK(cuDeviceGetCount(&count));
  printf("devices: %s\n", count > 0 ? "at least one" : "none");
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  char name[128] = {0};
  CK(cuDeviceGetName(name, sizeof name, dev));
  printf("name non-empty: %s\n", name[0] ? "yes" : "no");
  int major = 0, minor = 0;
  CK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  printf("compute capability sane: %s\n", (major >= 3 && minor >= 0) ? "yes" : "no");
  size_t total = 0;
  CK(cuDeviceTotalMem(&total, dev));
  printf("total memory positive: %s\n", total > 0 ? "yes" : "no");

  // cuCtxCreate resolves to whichever versioned symbol this toolkit renames it
  // to; the point of calling it through the vendor header is that we do not
  // choose which.
  CUcontext ctx = nullptr;
#if CUDA_VERSION >= 13000
  CK(cuCtxCreate(&ctx, nullptr, 0, dev));
#else
  CK(cuCtxCreate(&ctx, 0, dev));
#endif
  printf("context created: %s\n", ctx ? "yes" : "no");

  CUdeviceptr buf = 0;
  const int n = 64;
  CK(cuMemAlloc(&buf, n * sizeof(int)));
  std::vector<int> host(n, 41);
  CK(cuMemcpyHtoD(buf, host.data(), n * sizeof(int)));

  CUmodule mod;
  CK(cuModuleLoadData(&mod, kPtx));
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "add_one"));
  int nn = n;
  void* args[] = {&buf, &nn};
  CK(cuLaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, nullptr, args, nullptr));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(host.data(), buf, n * sizeof(int)));
  int wrong = 0;
  for (int i = 0; i < n; ++i) if (host[i] != 42) ++wrong;
  printf("kernel result wrong: %d\n", wrong);

  // Events, including the elapsed-time call CUDA 13 renames.
  CUevent a, b;
  CK(cuEventCreate(&a, CU_EVENT_DEFAULT));
  CK(cuEventCreate(&b, CU_EVENT_DEFAULT));
  CK(cuEventRecord(a, nullptr));
  CK(cuEventRecord(b, nullptr));
  CK(cuEventSynchronize(b));
  float ms = -1;
  CK(cuEventElapsedTime(&ms, a, b));
  printf("elapsed time non-negative: %s\n", ms >= 0 ? "yes" : "no");
  CK(cuEventDestroy(a));
  CK(cuEventDestroy(b));

  size_t freeb = 0, totalb = 0;
  CK(cuMemGetInfo(&freeb, &totalb));
  printf("free <= total: %s\n", freeb <= totalb ? "yes" : "no");

  CUdeviceptr base = 0;
  size_t size = 0;
  CK(cuMemGetAddressRange(&base, &size, buf + 16));
  printf("address range covers the pointer: %s\n",
         (base == buf && size == n * sizeof(int)) ? "yes" : "no");

  CK(cuMemFree(buf));
  CK(cuModuleUnload(mod));
  CK(cuCtxDestroy(ctx));
  printf("RESULT: driver ABI agrees with the vendor header\n");
  return 0;
}
