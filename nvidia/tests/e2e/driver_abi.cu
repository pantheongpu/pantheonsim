// The driver shim is written against a clean-room header (nvidia/include/vgpu_cuda.h),
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
#include <thread>
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

  // Device attributes, checked through the vendor header's own enum names.
  //
  // This is the check that was missing: the shim's attribute table is written
  // with bare integers, and ten of them were filed under the wrong number --
  // including MAX_BLOCKS_PER_MULTIPROCESSOR, added after a CUB scan launched no
  // blocks, which went to 134 (HOST_NUMA_ID) and left the real 106 answering
  // zero. Compiling against cuda.h is what makes a wrong number visible.
  struct { CUdevice_attribute attr; const char* name; int min; } kPositive[] = {
      {CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, "MAX_BLOCKS_PER_MULTIPROCESSOR", 1},
      {CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, "MAX_THREADS_PER_MULTIPROCESSOR", 1},
      {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "MAX_SHARED_MEMORY_PER_SM", 1},
      {CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "MAX_REGISTERS_PER_SM", 1},
      {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, "SHARED_MEMORY_PER_BLOCK_OPTIN", 1},
      {CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, "MAXIMUM_TEXTURE1D_WIDTH", 1},
      {CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, "MAXIMUM_TEXTURE2D_HEIGHT", 1},
      {CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, "MAXIMUM_SURFACE3D_DEPTH", 1},
      {CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, "TEXTURE_PITCH_ALIGNMENT", 1},
  };
  int attr_bad = 0;
  for (auto& a : kPositive) {
    int v = 0;
    CK(cuDeviceGetAttribute(&v, a.attr, dev));
    if (v < a.min) { printf("  %s answered %d\n", a.name, v); ++attr_bad; }
  }
  printf("attributes that must be positive are: %s\n", attr_bad ? "NOT all positive" : "all positive");

  // And the converse: a capability VirtualGPU does not implement must answer
  // no. A block count leaking into a NUMA query is what the wrong numbering
  // looked like from the outside.
  //
  // HOST_NUMA_ID is guarded because the enum is newer than some toolkits this
  // builds against -- CUDA 12.0 has no such name. Checked against 13.0, where
  // it exists, rather than guessing at the release that introduced it.
#if CUDA_VERSION >= 13000
  int numa_id = 0;
  CK(cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev));
  printf("HOST_NUMA_ID is not a block count: %s\n", numa_id <= 0 ? "yes" : "no");
#endif
  int managed = -1;
  CK(cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev));
  printf("MANAGED_MEMORY answers no: %s\n", managed == 0 ? "yes" : "no");

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

  // The current context belongs to the calling thread: a new thread starts
  // with none, and what it pushes stays on its own stack. The stack used to be
  // shared by the whole process.
  CUcontext mine = nullptr, seen = reinterpret_cast<CUcontext>(1), still = nullptr;
  CK(cuCtxGetCurrent(&mine));
  CUresult pushed = CUDA_ERROR_UNKNOWN;
  std::thread([&] {
    cuCtxGetCurrent(&seen);
    pushed = cuCtxPushCurrent(mine);
  }).join();
  CK(cuCtxGetCurrent(&still));
  if (seen != nullptr || pushed != CUDA_SUCCESS || still != mine) {
    printf("FAIL context stack shared between threads: new thread saw %p\n", (void*)seen);
    return 1;
  }

  // Primary contexts are reference counted; releasing more than was retained
  // is an error. It used to succeed however often it was called.
  CUcontext primary = nullptr;
  CK(cuDevicePrimaryCtxRetain(&primary, 0));
  CUresult rel = CUDA_SUCCESS;
  int releases = 0;
  while ((rel = cuDevicePrimaryCtxRelease(0)) == CUDA_SUCCESS && releases < 100) ++releases;
  unsigned int pflags = 0;
  int active = -1;
  CK(cuDevicePrimaryCtxGetState(0, &pflags, &active));
  if (releases < 1 || rel != CUDA_ERROR_INVALID_CONTEXT || active != 0) {
    printf("FAIL primary context over-release: %d releases, then %d, active %d\n", releases,
           (int)rel, active);
    return 1;
  }

  // Every code cuda.h declares has a name and a description; an undeclared
  // one is CUDA_ERROR_INVALID_VALUE with NULL. Several real codes used to be
  // reported as undeclared.
  for (int code : {4, 5, 6, 100, 200, 300, 400, 600, 704}) {
    const char* nm = nullptr;
    const char* str = nullptr;
    if (cuGetErrorName(static_cast<CUresult>(code), &nm) != CUDA_SUCCESS || !nm ||
        cuGetErrorString(static_cast<CUresult>(code), &str) != CUDA_SUCCESS || !str) {
      printf("FAIL cuGetErrorName/cuGetErrorString(%d)\n", code);
      return 1;
    }
  }
  const char* none = "not reset";
  if (cuGetErrorName(static_cast<CUresult>(12345), &none) != CUDA_ERROR_INVALID_VALUE || none) {
    printf("FAIL an undeclared code has a name\n");
    return 1;
  }

  // Tensor maps: the requirements cuda.h lists for cuTensorMapEncodeTiled are
  // checked, each with the code the documentation gives, and the modes not
  // implemented say so rather than encoding something that would be misread.
  {
    alignas(64) CUtensorMap map;
    const cuuint64_t dim[2] = {64, 32}, stride[1] = {128};
    const cuuint32_t box[2] = {64, 8}, one[2] = {1, 1}, big[2] = {257, 8}, wide[2] = {64, 8};
    auto enc = [&](void* addr, const cuuint32_t* b, CUtensorMapSwizzle swz,
                   CUtensorMapDataType type = CU_TENSOR_MAP_DATA_TYPE_UINT16) {
      return cuTensorMapEncodeTiled(&map, type, 2, addr, dim, stride, b, one,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    void* base = reinterpret_cast<void*>(static_cast<uintptr_t>(buf));
    void* odd = reinterpret_cast<void*>(static_cast<uintptr_t>(buf) + 8);
    CK(enc(base, box, CU_TENSOR_MAP_SWIZZLE_128B));
    CK(cuTensorMapReplaceAddress(&map, base));
    struct { const char* what; CUresult got, want; } cases[] = {
        {"a global address off 16 bytes", enc(odd, box, CU_TENSOR_MAP_SWIZZLE_NONE), CUDA_ERROR_INVALID_VALUE},
        {"a box of 257", enc(base, big, CU_TENSOR_MAP_SWIZZLE_NONE), CUDA_ERROR_INVALID_VALUE},
        {"a 128-byte box row under the 64-byte swizzle", enc(base, wide, CU_TENSOR_MAP_SWIZZLE_64B),
         CUDA_ERROR_INVALID_VALUE},
#if CUDA_VERSION >= 12080
        // Blackwell's swizzle atoms, which CUDA 12.0's header does not name.
        {"the 128B swizzle with 32B atoms", enc(base, box, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B),
         CUDA_ERROR_NOT_SUPPORTED},
#endif
        {"a replaced address off 16 bytes", cuTensorMapReplaceAddress(&map, odd), CUDA_ERROR_INVALID_VALUE},
    };
    for (const auto& c : cases)
      if (c.got != c.want) {
        printf("FAIL cuTensorMapEncodeTiled with %s -> %d, expected %d\n", c.what, (int)c.got, (int)c.want);
        return 1;
      }
  }

  CK(cuMemFree(buf));
  CK(cuModuleUnload(mod));
  CK(cuCtxDestroy(ctx));
  printf("RESULT: driver ABI agrees with the vendor header\n");
  return 0;
}
