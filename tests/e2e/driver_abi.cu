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

static const char* kPtxSecond = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry set_seven(.param .u64 p)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<5>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.s32 %rd3, %r1, 4;
  add.s64 %rd4, %rd2, %rd3;
  mov.u32 %r2, 7;
  st.global.u32 [%rd4], %r2;
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

  // Runtime JIT linking. Numba resolves and calls this family for every kernel
  // it compiles, so a missing or mis-typed entry point stops it before its
  // first launch. Two inputs, so the merge is exercised rather than a
  // single-module pass-through.
  CUlinkState link;
  CK(cuLinkCreate(0, nullptr, nullptr, &link));
  CK(cuLinkAddData(link, CU_JIT_INPUT_PTX, (void*)kPtx, strlen(kPtx) + 1, "add_one", 0, nullptr,
                   nullptr));
  CK(cuLinkAddData(link, CU_JIT_INPUT_PTX, (void*)kPtxSecond, strlen(kPtxSecond) + 1, "set_seven",
                   0, nullptr, nullptr));
  void* image = nullptr;
  size_t image_size = 0;
  CK(cuLinkComplete(link, &image, &image_size));
  printf("link produced an image: %s\n", (image && image_size > 0) ? "yes" : "no");

  CUmodule linked;
  CK(cuModuleLoadDataEx(&linked, image, 0, nullptr, nullptr));
  CUfunction seven, plus;
  CK(cuModuleGetFunction(&seven, linked, "set_seven"));
  CK(cuModuleGetFunction(&plus, linked, "add_one"));
  printf("both linked kernels resolve: yes\n");

  // Run one from each input, in order, so the result proves both bodies
  // survived the merge rather than only their names.
  void* seven_args[] = {&buf};
  CK(cuLaunchKernel(seven, 1, 1, 1, n, 1, 1, 0, nullptr, seven_args, nullptr));
  CK(cuLaunchKernel(plus, 1, 1, 1, n, 1, 1, 0, nullptr, args, nullptr));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(host.data(), buf, n * sizeof(int)));
  int linked_wrong = 0;
  for (int i = 0; i < n; ++i) if (host[i] != 8) ++linked_wrong;
  printf("linked kernel result wrong: %d\n", linked_wrong);
  CK(cuModuleUnload(linked));
  CK(cuLinkDestroy(link));

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
