// Driver API entry points CUDA's samples and older bindings reach for, checked
// against what an RTX 3060's driver answers: registered host memory and its
// flags, 2D memsets, occupancy-based block sizes, peer access, PCI bus ids,
// mipmapped arrays, the pre-CUDA 4 launch (cuParamSet*, cuLaunchGrid), texture
// references, texture and surface objects, and the deprecated odds and ends (cuDeviceGetProperties,
// cuCtxAttach). Every check passes on the card as well; where the simulated
// machine differs from that one by design -- the device pointer of registered
// memory is its host address, simulated devices are peers, read-only
// registration is not supported -- the check asks the driver first.
#include <cuda.h>
#include <cudaProfiler.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char* kPtx = R"(
.version 7.0
.target sm_80
.address_size 64
.global .attribute(.managed) .align 4 .u32 managed_global = 9;
.visible .entry fill(.param .u64 p, .param .u32 n, .param .f32 v)
{
  .reg .pred %q;
  .reg .b32 %r<6>;
  .reg .f32 %f<2>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [n];
  ld.param.f32 %f1, [v];
  mov.u32 %r2, %tid.x;
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %ntid.x;
  mad.lo.u32 %r2, %r3, %r4, %r2;
  setp.ge.u32 %q, %r2, %r1;
  @%q bra done;
  mul.wide.u32 %rd2, %r2, 4;
  add.u64 %rd3, %rd1, %rd2;
  st.global.f32 [%rd3], %f1;
done:
  ret;
}
.visible .entry mixed(.param .u32 a, .param .u64 p, .param .u8 c, .param .u16 h, .param .f64 d)
{
  .reg .b32 %r<4>;
  .reg .b16 %s<3>;
  .reg .f64 %fd<2>;
  .reg .b64 %rd<3>;
  ld.param.u32 %r1, [a];
  ld.param.u64 %rd1, [p];
  ld.param.u8 %s1, [c];
  ld.param.u16 %s2, [h];
  ld.param.f64 %fd1, [d];
  st.global.u32 [%rd1], %r1;
  cvt.u32.u16 %r2, %s1;
  st.global.u32 [%rd1+4], %r2;
  cvt.u32.u16 %r3, %s2;
  st.global.u32 [%rd1+8], %r3;
  st.global.f64 [%rd1+16], %fd1;
  ret;
}
.visible .entry texread(.param .u64 t, .param .u64 out, .param .f32 x, .param .f32 y)
{
  .reg .f32 %f<7>;
  .reg .b64 %rd<3>;
  ld.param.u64 %rd1, [t];
  ld.param.u64 %rd2, [out];
  ld.param.f32 %f5, [x];
  ld.param.f32 %f6, [y];
  tex.2d.v4.f32.f32 {%f1, %f2, %f3, %f4}, [%rd1, {%f5, %f6}];
  st.global.f32 [%rd2], %f1;
  ret;
}
.visible .entry surfread(.param .u64 s, .param .u64 out, .param .u32 x, .param .u32 y)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<3>;
  ld.param.u64 %rd1, [s];
  ld.param.u64 %rd2, [out];
  ld.param.u32 %r2, [x];
  ld.param.u32 %r3, [y];
  suld.b.2d.b32.trap {%r1}, [%rd1, {%r2, %r3}];
  st.global.u32 [%rd2], %r1;
  ret;
}
.visible .entry big(.param .u64 p)
{
  .reg .b64 %rd<2>;
  .reg .b32 %r<2>;
  .shared .align 4 .b8 sm[40000];
  ld.param.u64 %rd1, [p];
  mov.u32 %r1, 7;
  st.global.u32 [%rd1], %r1;
  ret;
}
)";

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static int attr(CUdevice_attribute a, CUdevice dev = 0) {
  int v = -1;
  cuDeviceGetAttribute(&v, a, dev);
  return v;
}

static std::vector<float> read_floats(CUdeviceptr d, size_t n) {
  std::vector<float> h(n, -1.0f);
  cuMemcpyDtoH(h.data(), d, n * sizeof(float));
  return h;
}

// ---- host memory: registered, pinned and managed ----
static void host_memory(CUfunction kfill) {
  const size_t N = 1 << 16;
  char* h = static_cast<char*>(std::aligned_alloc(4096, N));
  std::memset(h, 0x11, N);
  unsigned f = 99;
  CUdeviceptr d = 0, d100 = 0;
  IS(cuMemHostGetFlags(&f, h), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostGetDevicePointer(&d, h, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostRegister(nullptr, N, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostRegister(h, 0, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostRegister(h, N, 0x10), CUDA_ERROR_INVALID_VALUE);
  // Read-only registration where the device says it has it.
  const CUresult ro = cuMemHostRegister(h, N, CU_MEMHOSTREGISTER_READ_ONLY);
  if (attr(CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED)) {
    check(ro == CUDA_SUCCESS, "read-only registration where it is supported");
    IS(cuMemHostUnregister(h), CUDA_SUCCESS);
  } else {
    check(ro == CUDA_ERROR_NOT_SUPPORTED, "read-only registration refused where it is not supported");
  }

  IS(cuMemHostRegister(h, N, 0), CUDA_SUCCESS);
  IS(cuMemHostRegister(h, N, 0), CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
  IS(cuMemHostRegister(h + 4096, 4096, 0), CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
  IS(cuMemHostRegister(h + N - 4096, 8192, 0), CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
  f = 99;
  IS(cuMemHostGetFlags(&f, h), CUDA_SUCCESS);
  check(f == CU_MEMHOSTALLOC_DEVICEMAP, "registered memory's flags are DEVICEMAP");
  f = 99;
  IS(cuMemHostGetFlags(&f, h + 100), CUDA_SUCCESS);
  check(f == CU_MEMHOSTALLOC_DEVICEMAP, "and so are an interior pointer's");
  IS(cuMemHostGetFlags(nullptr, h), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostGetDevicePointer(&d, h, 0), CUDA_SUCCESS);
  IS(cuMemHostGetDevicePointer(&d100, h + 100, 0), CUDA_SUCCESS);
  check(d != 0 && d100 == d + 100, "an interior pointer's device address is as far into the range");
  CUdeviceptr scratch = 0;   // a refused call may still write here, as the card's does
  IS(cuMemHostGetDevicePointer(&scratch, h, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostGetDevicePointer(nullptr, h, 0), CUDA_ERROR_INVALID_VALUE);
  // A kernel writes through the device pointer, and the host sees it.
  {
    unsigned n = 4;
    float v = 2.5f;
    void* args[] = {&d, &n, &v};
    IS(cuLaunchKernel(kfill, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr), CUDA_SUCCESS);
    IS(cuCtxSynchronize(), CUDA_SUCCESS);
    float got = 0;
    std::memcpy(&got, h + 12, 4);
    check(got == 2.5f, "a kernel writes registered memory through its device pointer");
  }
  // The registered pointer is not filled, by any of the memsets.
  std::memset(h, 0x11, 64);
  IS(cuMemsetD8((CUdeviceptr)h, 0x33, 16), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD16((CUdeviceptr)h, 0x3344, 8), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD32((CUdeviceptr)h, 0x33445566, 4), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D8((CUdeviceptr)h, 64, 0x33, 16, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD8Async((CUdeviceptr)h, 0x33, 16, nullptr), CUDA_ERROR_INVALID_VALUE);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(h[0] == 0x11 && h[15] == 0x11 && h[64] == 0x11, "and the host memory is untouched");
  IS(cuMemsetD8((CUdeviceptr)h + 8, 0x33, 0), CUDA_SUCCESS);
  IS(cuMemHostUnregister(h + 100), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostUnregister(h), CUDA_SUCCESS);
  IS(cuMemHostUnregister(h), CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED);
  IS(cuMemHostUnregister(nullptr), CUDA_ERROR_INVALID_VALUE);

  IS(cuMemHostRegister(h, N, CU_MEMHOSTREGISTER_PORTABLE), CUDA_SUCCESS);
  f = 99;
  IS(cuMemHostGetFlags(&f, h), CUDA_SUCCESS);
  check(f == (CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP), "PORTABLE is reported back");
  IS(cuMemHostUnregister(h), CUDA_SUCCESS);
  // A registration's bytes exactly, not the pages it touches.
  IS(cuMemHostRegister(h + 1, 100, 0), CUDA_SUCCESS);
  IS(cuMemHostGetFlags(&f, h + 1), CUDA_SUCCESS);
  IS(cuMemHostGetFlags(&f, h), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostGetFlags(&f, h + 200), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostUnregister(h + 1), CUDA_SUCCESS);
  std::free(h);

  // Pinned memory: mapped, filled by memsets, and not registered over.
  void* pa = nullptr;
  IS(cuMemHostAlloc(&pa, 4096, 0), CUDA_SUCCESS);
  f = 99;
  IS(cuMemHostGetFlags(&f, pa), CUDA_SUCCESS);
  check(f == CU_MEMHOSTALLOC_DEVICEMAP, "pinned memory's flags are DEVICEMAP");
  d = 0;
  IS(cuMemHostGetDevicePointer(&d, static_cast<char*>(pa) + 8, 0), CUDA_SUCCESS);
  check(d == (CUdeviceptr)pa + 8, "pinned memory's device pointer is its host address");
  IS(cuMemsetD8((CUdeviceptr)pa, 0x44, 16), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(static_cast<unsigned char*>(pa)[15] == 0x44, "cuMemsetD8 fills pinned memory");
  IS(cuMemsetD32((CUdeviceptr)pa, 0x44454647, 4), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(static_cast<unsigned char*>(pa)[0] == 0x47 && static_cast<unsigned char*>(pa)[15] == 0x44,
        "cuMemsetD32 fills pinned memory");
  IS(cuMemHostRegister(pa, 4096, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostUnregister(pa), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemFreeHost(pa), CUDA_SUCCESS);
  IS(cuMemHostAlloc(&pa, 4096, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED),
     CUDA_SUCCESS);
  IS(cuMemHostGetFlags(&f, pa), CUDA_SUCCESS);
  check(f == 7, "a pinned allocation's flags come back as given");
  IS(cuMemFreeHost(pa), CUDA_SUCCESS);
  IS(cuMemAllocHost(&pa, 4096), CUDA_SUCCESS);
  IS(cuMemHostGetFlags(&f, pa), CUDA_SUCCESS);
  check(f == CU_MEMHOSTALLOC_DEVICEMAP, "cuMemAllocHost memory's flags are DEVICEMAP");
  IS(cuMemFreeHost(pa), CUDA_SUCCESS);

  CUdeviceptr man = 0;
  IS(cuMemAllocManaged(&man, 4096, CU_MEM_ATTACH_GLOBAL), CUDA_SUCCESS);
  IS(cuMemHostRegister((void*)man, 4096, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemHostGetFlags(&f, (void*)man), CUDA_SUCCESS);
  check(f == CU_MEMHOSTALLOC_DEVICEMAP, "managed memory's flags are DEVICEMAP");
  IS(cuMemFree(man), CUDA_SUCCESS);
}

// ---- cuMemsetD2D* ----
static void memsets_2d() {
  CUdeviceptr pd;
  size_t pitch = 0;
  IS(cuMemAllocPitch(&pd, &pitch, 40, 4, 4), CUDA_SUCCESS);
  std::vector<unsigned char> b(pitch * 4);
  IS(cuMemsetD8(pd, 0, pitch * 4), CUDA_SUCCESS);
  IS(cuMemsetD2D8(pd, pitch, 0xAB, 10, 3), CUDA_SUCCESS);
  cuMemcpyDtoH(b.data(), pd, b.size());
  check(b[0] == 0xAB && b[9] == 0xAB && b[10] == 0 && b[pitch + 9] == 0xAB && b[2 * pitch + 9] == 0xAB &&
            b[3 * pitch] == 0,
        "cuMemsetD2D8 fills 10 bytes of each of 3 rows");
  IS(cuMemsetD2D16(pd, pitch, 0xBEEF, 5, 2), CUDA_SUCCESS);
  cuMemcpyDtoH(b.data(), pd, b.size());
  check(b[0] == 0xEF && b[1] == 0xBE && b[9] == 0xBE && b[10] == 0 && b[pitch + 9] == 0xBE,
        "cuMemsetD2D16 fills 5 elements of each of 2 rows");
  IS(cuMemsetD2D32(pd, pitch, 0x01020304, 3, 2), CUDA_SUCCESS);
  cuMemcpyDtoH(b.data(), pd, b.size());
  check(b[0] == 4 && b[11] == 1 && b[12] == 0 && b[pitch + 11] == 1, "cuMemsetD2D32 fills 3 elements of 2 rows");
  // A pitch holds a row, in whole elements -- checked only with more than one row.
  IS(cuMemsetD2D8(pd, 8, 0xAB, 10, 3), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D8(pd, 11, 0xAB, 10, 3), CUDA_SUCCESS);
  IS(cuMemsetD2D8(pd, 0, 0xAB, 10, 1), CUDA_SUCCESS);
  IS(cuMemsetD2D8(pd, 0, 0xAB, 10, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D16(pd, 11, 0xAB, 5, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D16(pd, 11, 0xAB, 5, 1), CUDA_SUCCESS);
  IS(cuMemsetD2D16(pd, 12, 0xAB, 5, 2), CUDA_SUCCESS);
  IS(cuMemsetD2D32(pd, 14, 0xAB, 3, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D32(pd, 16, 0xAB, 3, 2), CUDA_SUCCESS);
  // The pointer at the element's alignment, in 1D and 2D alike.
  IS(cuMemsetD2D16(pd + 1, pitch, 0xAB, 5, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D32(pd + 2, pitch, 0xAB, 3, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD16(pd + 1, 0xAB, 3), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD32(pd + 2, 0xAB, 3), CUDA_ERROR_INVALID_VALUE);
  // Nothing to fill is a success, whatever the pointer; past the end is not.
  IS(cuMemsetD2D8(pd, pitch, 0xAB, 0, 3), CUDA_SUCCESS);
  IS(cuMemsetD2D8(pd, pitch, 0xAB, 10, 0), CUDA_SUCCESS);
  IS(cuMemsetD2D8(0, 0, 0xAB, 0, 0), CUDA_SUCCESS);
  IS(cuMemsetD8(0, 0xAB, 0), CUDA_SUCCESS);
  IS(cuMemsetD8(0, 0xAB, 4), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D8(0, pitch, 0xAB, 10, 3), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D8(pd, pitch, 0xAB, pitch, 5), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD8(pd + 4 * pitch - 4, 0xAB, 8), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD2D8(pd + 3 * pitch, pitch, 0xAB, 10, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemsetD8(pd, 0, pitch * 4), CUDA_SUCCESS);
  IS(cuMemsetD2D8Async(pd, pitch, 0x5A, 10, 3, nullptr), CUDA_SUCCESS);
  IS(cuMemsetD2D16Async(pd, pitch, 0x5A5A, 5, 3, nullptr), CUDA_SUCCESS);
  IS(cuMemsetD2D32Async(pd, pitch, 0x5A5A5A5A, 3, 3, nullptr), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  cuMemcpyDtoH(b.data(), pd, b.size());
  check(b[0] == 0x5A && b[2 * pitch + 11] == 0x5A && b[2 * pitch + 12] == 0 && b[3 * pitch] == 0, "the async fills");
  IS(cuMemFree(pd), CUDA_SUCCESS);
}

// ---- occupancy ----
static size_t b2d_calls = 0;
static size_t CUDA_CB b2d(int block) {
  ++b2d_calls;
  return block >= 512 ? 60000 : 1024;
}

// The search CUDA documents, run here on cuOccupancyMaxActiveBlocksPerMultiprocessor.
static void reference(CUfunction f, CUoccupancyB2DSize cb, size_t dyn, int limit, int* grid, int* block,
                      size_t* calls) {
  const int per_sm = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
  const int warp = attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
  const int dev_max = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  int func_max = 0;
  cuFuncGetAttribute(&func_max, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);
  if (limit == 0) limit = dev_max;
  if (dev_max < limit) limit = dev_max;
  if (func_max < limit) limit = func_max;
  const int aligned = (limit + warp - 1) / warp * warp;
  int best = 0, best_block = 0, best_blocks = 0;
  *calls = 0;
  for (int s = aligned; s > 0; s -= warp) {
    const int t = limit < s ? limit : s;
    ++*calls;
    int blocks = 0;
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, f, t, cb ? cb(t) : dyn);
    if (blocks * t > best) {
      best = blocks * t;
      best_block = t;
      best_blocks = blocks;
    }
    if (best == per_sm) break;
  }
  *grid = best_blocks * attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  *block = best_block;
}

static void occupancy(CUfunction kfill, CUfunction kbig) {
  struct Case {
    CUfunction f;
    CUoccupancyB2DSize cb;
    size_t dyn;
    int limit;
    const char* what;
  } cases[] = {
      {kfill, nullptr, 0, 0, "the best block size is the one the documented search finds"},
      {kfill, nullptr, 0, 100, "under a block size limit"},
      {kfill, nullptr, 20000, 0, "with constant dynamic shared memory"},
      {kfill, b2d, 0, 0, "with shared memory from a callback"},
      {kbig, nullptr, 0, 0, "for a kernel with 40000 bytes of static shared memory"},
      {kbig, nullptr, 20000, 0, "for one that cannot fit a block"},
      {kfill, nullptr, 0, 5000, "with a limit above the device's"},
  };
  for (const Case& c : cases) {
    int grid = -1, block = -1, want_grid = -2, want_block = -2;
    size_t calls = 0;
    reference(c.f, c.cb, c.dyn, c.limit, &want_grid, &want_block, &calls);
    b2d_calls = 0;
    const CUresult r = cuOccupancyMaxPotentialBlockSize(&grid, &block, c.f, c.cb, c.dyn, c.limit);
    const size_t made = b2d_calls;
    char what[160];
    std::snprintf(what, sizeof what, "%s (%d x %d)", c.what, want_grid, want_block);
    check(r == CUDA_SUCCESS && grid == want_grid && block == want_block && (!c.cb || made == calls), what);
  }
  int grid = -1, block = -1;
  IS(cuOccupancyMaxPotentialBlockSize(&grid, &block, kbig, nullptr, 20000, 0), CUDA_SUCCESS);
  check(grid == 0 && block == 0, "no block fits: 0 and 0");
  IS(cuOccupancyMaxPotentialBlockSize(&grid, &block, kfill, nullptr, 0, 100), CUDA_SUCCESS);
  check(block <= 100 && block % 32 == 0, "the limit caps the block size");
  IS(cuOccupancyMaxPotentialBlockSize(nullptr, &block, kfill, nullptr, 0, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuOccupancyMaxPotentialBlockSize(&grid, nullptr, kfill, nullptr, 0, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuOccupancyMaxPotentialBlockSize(&grid, &block, nullptr, nullptr, 0, 0), CUDA_ERROR_INVALID_HANDLE);
  IS(cuOccupancyMaxPotentialBlockSize(&grid, &block, kfill, nullptr, 0, -5), CUDA_ERROR_INVALID_VALUE);
  IS(cuOccupancyMaxPotentialBlockSizeWithFlags(&grid, &block, kfill, nullptr, 0, 0, CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE),
     CUDA_SUCCESS);
  IS(cuOccupancyMaxPotentialBlockSizeWithFlags(&grid, &block, kfill, nullptr, 0, 0, 2), CUDA_ERROR_INVALID_VALUE);
  int nb = -1;
  IS(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kfill, 2048, 0), CUDA_SUCCESS);
  check(nb == 0, "no block larger than a block may be is resident");
  IS(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kfill, 1024, 200000), CUDA_SUCCESS);
  check(nb == 0, "nor one asking for more shared memory than a block may have");
  IS(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kfill, 0, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kbig, 128, 0), CUDA_SUCCESS);
  check(nb >= 1 && nb * 40000 <= attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        "shared memory bounds the resident blocks");
}

// ---- peers ----
static void peers(CUcontext ctx) {
  int count = 0, can = -1;
  cuDeviceGetCount(&count);
  IS(cuDeviceCanAccessPeer(&can, 0, 0), CUDA_SUCCESS);
  check(can == 0, "a device is not its own peer");
  IS(cuDeviceCanAccessPeer(&can, 0, count), CUDA_ERROR_INVALID_DEVICE);
  IS(cuCtxEnablePeerAccess(ctx, 0), CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
  IS(cuCtxEnablePeerAccess(ctx, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuCtxEnablePeerAccess(nullptr, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuCtxDisablePeerAccess(ctx), CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
  IS(cuCtxDisablePeerAccess(nullptr), CUDA_ERROR_INVALID_VALUE);
  if (count < 2) return;
  CUdevice dev1;
  CUcontext ctx1;
  IS(cuDeviceGet(&dev1, 1), CUDA_SUCCESS);
  IS(cuDevicePrimaryCtxRetain(&ctx1, dev1), CUDA_SUCCESS);
  IS(cuDeviceCanAccessPeer(&can, 0, 1), CUDA_SUCCESS);
  if (can) {
    IS(cuCtxEnablePeerAccess(ctx1, 0), CUDA_SUCCESS);
    IS(cuCtxEnablePeerAccess(ctx1, 0), CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
    IS(cuCtxDisablePeerAccess(ctx1), CUDA_SUCCESS);
    IS(cuCtxDisablePeerAccess(ctx1), CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
  } else {
    IS(cuCtxEnablePeerAccess(ctx1, 0), CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
  }
  CUdeviceptr a0, a1;
  IS(cuMemAlloc(&a0, 64), CUDA_SUCCESS);
  IS(cuCtxPushCurrent(ctx1), CUDA_SUCCESS);
  IS(cuMemAlloc(&a1, 64), CUDA_SUCCESS);
  IS(cuMemsetD8(a1, 0x5A, 64), CUDA_SUCCESS);
  IS(cuCtxPopCurrent(&ctx1), CUDA_SUCCESS);
  IS(cuMemcpyPeer(a0, ctx, a1, ctx1, 64), CUDA_SUCCESS);
  IS(cuMemcpyPeerAsync(a0, ctx, a1, ctx1, 0, nullptr), CUDA_SUCCESS);
  unsigned char b[64] = {};
  IS(cuMemcpyDtoH(b, a0, 64), CUDA_SUCCESS);
  check(b[0] == 0x5A && b[63] == 0x5A, "cuMemcpyPeer copies between the devices");
  IS(cuMemFree(a0), CUDA_SUCCESS);
  IS(cuCtxPushCurrent(ctx1), CUDA_SUCCESS);
  IS(cuMemFree(a1), CUDA_SUCCESS);
  IS(cuCtxPopCurrent(&ctx1), CUDA_SUCCESS);
  IS(cuDevicePrimaryCtxRelease(dev1), CUDA_SUCCESS);
}

// ---- PCI bus ids ----
static void pci() {
  int count = 0;
  cuDeviceGetCount(&count);
  for (int i = 0; i < count; ++i) {
    char id[64] = {};
    IS(cuDeviceGetPCIBusId(id, sizeof id, i), CUDA_SUCCESS);
    const std::string full = id;
    const std::string no_domain = full.substr(full.find(':') + 1);
    const std::string no_fn = full.substr(0, full.rfind('.'));
    const std::string fn1 = no_fn + ".1";
    for (const std::string& s : std::vector<std::string>{full, no_domain, no_fn, "00000000:" + no_domain}) {
      CUdevice got = -1;
      const CUresult r = cuDeviceGetByPCIBusId(&got, s.c_str());
      check(r == CUDA_SUCCESS && got == i, ("'" + s + "' names its device").c_str());
    }
    CUdevice got = -1;
    IS(cuDeviceGetByPCIBusId(&got, fn1.c_str()), CUDA_ERROR_INVALID_DEVICE);
    IS(cuDeviceGetByPCIBusId(&got, (full + "junk").c_str()), CUDA_ERROR_INVALID_VALUE);
    IS(cuDeviceGetByPCIBusId(&got, (full + " ").c_str()), CUDA_ERROR_INVALID_VALUE);
    char shortbuf[8] = {};
    IS(cuDeviceGetPCIBusId(shortbuf, sizeof shortbuf, i), CUDA_ERROR_INVALID_VALUE);
  }
  CUdevice got = -1;
  IS(cuDeviceGetByPCIBusId(&got, "garbage"), CUDA_ERROR_INVALID_VALUE);
  IS(cuDeviceGetByPCIBusId(&got, ""), CUDA_ERROR_INVALID_VALUE);
  IS(cuDeviceGetByPCIBusId(&got, "0001:01:00.0"), CUDA_ERROR_INVALID_DEVICE);
  IS(cuDeviceGetByPCIBusId(&got, nullptr), CUDA_ERROR_INVALID_VALUE);
  IS(cuDeviceGetByPCIBusId(nullptr, "0000:01:00.0"), CUDA_ERROR_INVALID_VALUE);
}

// ---- mipmapped arrays ----
static bool level(CUmipmappedArray mm, unsigned l, size_t w, size_t h, size_t d, unsigned flags) {
  CUarray a = nullptr;
  CUDA_ARRAY3D_DESCRIPTOR got{};
  return cuMipmappedArrayGetLevel(&a, mm, l) == CUDA_SUCCESS && cuArray3DGetDescriptor(&got, a) == CUDA_SUCCESS &&
         got.Width == w && got.Height == h && got.Depth == d && got.Flags == flags &&
         got.Format == CU_AD_FORMAT_FLOAT && got.NumChannels == 1;
}

static void mipmaps() {
  CUDA_ARRAY3D_DESCRIPTOR md{};
  md.Width = 64;
  md.Height = 32;
  md.Format = CU_AD_FORMAT_FLOAT;
  md.NumChannels = 1;
  CUmipmappedArray mm = nullptr;
  IS(cuMipmappedArrayCreate(&mm, &md, 4), CUDA_SUCCESS);
  check(level(mm, 0, 64, 32, 0, 0) && level(mm, 1, 32, 16, 0, 0) && level(mm, 3, 8, 4, 0, 0),
        "each level halves the last");
  CUarray la = nullptr, lb = nullptr;
  IS(cuMipmappedArrayGetLevel(&la, mm, 4), CUDA_ERROR_INVALID_VALUE);
  IS(cuMipmappedArrayGetLevel(nullptr, mm, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMipmappedArrayGetLevel(&la, mm, 1), CUDA_SUCCESS);
  IS(cuMipmappedArrayGetLevel(&lb, mm, 1), CUDA_SUCCESS);
  check(la == lb, "a level is the same array each time");
  float src[32], back[32] = {};
  for (int i = 0; i < 32; ++i) src[i] = static_cast<float>(i);
  IS(cuMemcpyHtoA(la, 0, src, sizeof src), CUDA_SUCCESS);
  IS(cuMemcpyAtoH(back, la, 0, sizeof back), CUDA_SUCCESS);
  check(back[31] == 31.0f, "a level holds what is copied into it");
  CUDA_MEMCPY2D c{};
  c.srcMemoryType = CU_MEMORYTYPE_HOST;
  c.srcHost = src;
  c.srcPitch = 64;
  c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  c.dstArray = la;
  c.WidthInBytes = 64;
  c.Height = 2;
  IS(cuMemcpy2D(&c), CUDA_SUCCESS);
  c.Height = 17;   // level 1 has 16 rows
  IS(cuMemcpy2D(&c), CUDA_ERROR_INVALID_VALUE);
  // A level belongs to its mipmapped array: destroying it is a success that
  // leaves it in place.
  IS(cuArrayDestroy(la), CUDA_SUCCESS);
  check(level(mm, 1, 32, 16, 0, 0), "a level survives cuArrayDestroy");
  IS(cuMipmappedArrayDestroy(mm), CUDA_SUCCESS);
  IS(cuMipmappedArrayDestroy(mm), CUDA_ERROR_CONTEXT_IS_DESTROYED);
  IS(cuMipmappedArrayGetLevel(&la, mm, 0), CUDA_ERROR_CONTEXT_IS_DESTROYED);
  CUDA_ARRAY3D_DESCRIPTOR gone{};
  IS(cuArray3DGetDescriptor(&gone, lb), CUDA_ERROR_CONTEXT_IS_DESTROYED);

  // No levels makes one; too many is cut to the full chain.
  IS(cuMipmappedArrayCreate(&mm, &md, 0), CUDA_SUCCESS);
  check(level(mm, 0, 64, 32, 0, 0), "0 levels makes one");
  IS(cuMipmappedArrayGetLevel(&la, mm, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuMipmappedArrayDestroy(mm), CUDA_SUCCESS);
  IS(cuMipmappedArrayCreate(&mm, &md, 100), CUDA_SUCCESS);
  check(level(mm, 5, 2, 1, 0, 0) && level(mm, 6, 1, 1, 0, 0), "100 levels are the full chain of 7");
  IS(cuMipmappedArrayGetLevel(&la, mm, 7), CUDA_ERROR_INVALID_VALUE);
  IS(cuMipmappedArrayDestroy(mm), CUDA_SUCCESS);

  CUDA_ARRAY3D_DESCRIPTOR d3 = md;
  d3.Width = 16;
  d3.Height = 8;
  d3.Depth = 4;
  IS(cuMipmappedArrayCreate(&mm, &d3, 5), CUDA_SUCCESS);
  check(level(mm, 1, 8, 4, 2, 0) && level(mm, 2, 4, 2, 1, 0) && level(mm, 4, 1, 1, 1, 0),
        "a 3D array's levels halve its depth too");
  IS(cuMipmappedArrayDestroy(mm), CUDA_SUCCESS);
  CUDA_ARRAY3D_DESCRIPTOR dl = d3;
  dl.Depth = 3;
  dl.Flags = CUDA_ARRAY3D_LAYERED;
  IS(cuMipmappedArrayCreate(&mm, &dl, 4), CUDA_SUCCESS);
  check(level(mm, 3, 2, 1, 3, CUDA_ARRAY3D_LAYERED), "a layered array's levels keep their layers");
  IS(cuMipmappedArrayGetLevel(&la, mm, 4), CUDA_ERROR_INVALID_VALUE);
  IS(cuMipmappedArrayDestroy(mm), CUDA_SUCCESS);
  CUDA_ARRAY3D_DESCRIPTOR d1 = md;
  d1.Width = 100;
  d1.Height = 0;
  IS(cuMipmappedArrayCreate(&mm, &d1, 3), CUDA_SUCCESS);
  check(level(mm, 2, 25, 0, 0, 0), "a 1D array's levels");
  IS(cuMipmappedArrayDestroy(mm), CUDA_SUCCESS);
  CUDA_ARRAY3D_DESCRIPTOR bad = md;
  bad.Width = 0;
  IS(cuMipmappedArrayCreate(&mm, &bad, 1), CUDA_ERROR_INVALID_VALUE);
  bad = md;
  bad.NumChannels = 3;
  IS(cuMipmappedArrayCreate(&mm, &bad, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuMipmappedArrayCreate(nullptr, &md, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuMipmappedArrayCreate(&mm, nullptr, 2), CUDA_ERROR_INVALID_VALUE);
}

// ---- the pre-CUDA 4 launch ----
static void legacy_launch(CUfunction kfill, CUfunction kmixed) {
  CUdeviceptr out;
  IS(cuMemAlloc(&out, 256 * 4), CUDA_SUCCESS);
  IS(cuMemsetD32(out, 0, 256), CUDA_SUCCESS);
  IS(cuFuncSetBlockShape(kfill, 64, 1, 1), CUDA_SUCCESS);
  IS(cuFuncSetSharedSize(kfill, 0), CUDA_SUCCESS);
  IS(cuParamSetv(kfill, 0, &out, 8), CUDA_SUCCESS);
  IS(cuParamSeti(kfill, 8, 200), CUDA_SUCCESS);
  IS(cuParamSetf(kfill, 12, 1.5f), CUDA_SUCCESS);
  IS(cuParamSetSize(kfill, 16), CUDA_SUCCESS);
  IS(cuLaunchGrid(kfill, 3, 1), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  std::vector<float> o = read_floats(out, 256);
  check(o[0] == 1.5f && o[191] == 1.5f && o[192] == 0.0f, "cuLaunchGrid runs the launch set up on the function");
  IS(cuMemsetD32(out, 0, 256), CUDA_SUCCESS);
  IS(cuLaunch(kfill), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  o = read_floats(out, 256);
  check(o[63] == 1.5f && o[64] == 0.0f, "cuLaunch runs one block");
  IS(cuParamSetf(kfill, 12, 4.0f), CUDA_SUCCESS);
  IS(cuLaunchGridAsync(kfill, 1, 1, nullptr), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(read_floats(out, 1)[0] == 4.0f, "cuLaunchGridAsync, with a parameter changed");
  IS(cuLaunchGrid(kfill, 0, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuLaunchGrid(kfill, 1, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuLaunchGrid(kfill, 65536, 65536), CUDA_ERROR_INVALID_VALUE);
  // A parameter size past the kernel's parameters fails the launch; one short of
  // them launches.
  IS(cuParamSetSize(kfill, 24), CUDA_SUCCESS);
  IS(cuLaunchGrid(kfill, 1, 1), CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
  IS(cuParamSetSize(kfill, 17), CUDA_SUCCESS);
  IS(cuLaunchGrid(kfill, 1, 1), CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
  IS(cuParamSetSize(kfill, 12), CUDA_SUCCESS);
  IS(cuLaunchGrid(kfill, 1, 1), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  // What may be set.
  IS(cuParamSetSize(kfill, 32764), CUDA_SUCCESS);
  IS(cuParamSetSize(kfill, 32765), CUDA_ERROR_INVALID_VALUE);
  IS(cuParamSetv(kfill, 32760, &out, 4), CUDA_SUCCESS);
  IS(cuParamSetv(kfill, 32761, &out, 4), CUDA_ERROR_INVALID_VALUE);
  IS(cuParamSeti(kfill, 32764, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuParamSetv(kfill, 0, nullptr, 8), CUDA_ERROR_INVALID_VALUE);
  IS(cuParamSetv(kfill, 0, &out, 0), CUDA_SUCCESS);
  IS(cuParamSetSize(kfill, 16), CUDA_SUCCESS);
  IS(cuFuncSetBlockShape(kfill, 1024, 1, 1), CUDA_SUCCESS);
  IS(cuFuncSetBlockShape(kfill, 1025, 1, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuFuncSetBlockShape(kfill, 32, 33, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuFuncSetBlockShape(kfill, 1, 1, 65), CUDA_ERROR_INVALID_VALUE);
  IS(cuFuncSetBlockShape(kfill, 0, 1, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuFuncSetBlockShape(kfill, -1, 1, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuFuncSetSharedSize(kfill, 49152), CUDA_SUCCESS);
  IS(cuFuncSetSharedSize(kfill, 49153), CUDA_ERROR_INVALID_VALUE);
  IS(cuFuncSetSharedSize(kfill, 0), CUDA_SUCCESS);
  IS(cuFuncSetBlockShape(nullptr, 1, 1, 1), CUDA_ERROR_INVALID_HANDLE);
  IS(cuParamSetSize(nullptr, 8), CUDA_ERROR_INVALID_HANDLE);
  IS(cuLaunchGrid(nullptr, 1, 1), CUDA_ERROR_INVALID_HANDLE);
  IS(cuFuncSetSharedSize(nullptr, 0), CUDA_ERROR_INVALID_HANDLE);

  // Parameters of every width, each at its own alignment.
  CUdeviceptr mo;
  IS(cuMemAlloc(&mo, 32), CUDA_SUCCESS);
  IS(cuMemsetD8(mo, 0, 32), CUDA_SUCCESS);
  unsigned char c8 = 9;
  unsigned short h16 = 513;
  double f64 = 6.25;
  IS(cuFuncSetBlockShape(kmixed, 1, 1, 1), CUDA_SUCCESS);
  IS(cuParamSeti(kmixed, 0, 77), CUDA_SUCCESS);
  IS(cuParamSetv(kmixed, 8, &mo, 8), CUDA_SUCCESS);
  IS(cuParamSetv(kmixed, 16, &c8, 1), CUDA_SUCCESS);
  IS(cuParamSetv(kmixed, 18, &h16, 2), CUDA_SUCCESS);
  IS(cuParamSetv(kmixed, 24, &f64, 8), CUDA_SUCCESS);
  IS(cuParamSetSize(kmixed, 32), CUDA_SUCCESS);
  IS(cuLaunchGrid(kmixed, 1, 1), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  unsigned mv[8] = {};
  IS(cuMemcpyDtoH(mv, mo, 32), CUDA_SUCCESS);
  double got = 0;
  std::memcpy(&got, mv + 4, 8);
  check(mv[0] == 77 && mv[1] == 9 && mv[2] == 513 && got == 6.25, "u32, u64, u8, u16 and f64 parameters");

  // A function never shaped runs blocks of one thread.
  CUfunction fresh;
  CUmodule mod2;
  IS(cuModuleLoadData(&mod2, kPtx), CUDA_SUCCESS);
  IS(cuModuleGetFunction(&fresh, mod2, "fill"), CUDA_SUCCESS);
  IS(cuMemsetD32(out, 0, 256), CUDA_SUCCESS);
  unsigned n = 256;
  float three = 3.0f;
  IS(cuParamSetv(fresh, 0, &out, 8), CUDA_SUCCESS);
  IS(cuParamSetv(fresh, 8, &n, 4), CUDA_SUCCESS);
  IS(cuParamSetv(fresh, 12, &three, 4), CUDA_SUCCESS);
  IS(cuParamSetSize(fresh, 16), CUDA_SUCCESS);
  IS(cuLaunchGrid(fresh, 2, 1), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  o = read_floats(out, 256);
  int written = 0;
  for (float x : o) written += x == 3.0f;
  check(written == 2, "an unshaped function runs one thread a block");
  // The grid's height is its second dimension.
  IS(cuMemsetD32(out, 0, 256), CUDA_SUCCESS);
  IS(cuFuncSetBlockShape(fresh, 16, 1, 1), CUDA_SUCCESS);
  IS(cuLaunchGrid(fresh, 2, 2), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  o = read_floats(out, 256);
  written = 0;
  for (float x : o) written += x == 3.0f;
  check(written == 32, "a 2 x 2 grid of 16 threads covers 32 elements");
  // The legacy shape is not the one cuLaunchKernel uses.
  IS(cuFuncSetBlockShape(fresh, 7, 1, 1), CUDA_SUCCESS);
  IS(cuMemsetD32(out, 0, 256), CUDA_SUCCESS);
  void* args[] = {&out, &n, &three};
  IS(cuLaunchKernel(fresh, 1, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  o = read_floats(out, 256);
  check(o[63] == 3.0f && o[64] == 0.0f, "cuLaunchKernel's shape is its own");
  IS(cuModuleUnload(mod2), CUDA_SUCCESS);
  IS(cuMemFree(out), CUDA_SUCCESS);
  IS(cuMemFree(mo), CUDA_SUCCESS);
}

// ---- texture and surface references ----
static void texrefs(CUmodule mod, CUfunction kfill) {
  CUtexref t = nullptr;
  IS(cuModuleGetTexRef(&t, mod, "nope"), CUDA_ERROR_NOT_FOUND);
  CUsurfref sr = nullptr;
  IS(cuModuleGetSurfRef(&sr, mod, "nope"), CUDA_ERROR_NOT_FOUND);
  CUarray sa = nullptr;
  IS(cuSurfRefSetArray(nullptr, nullptr, 0), CUDA_ERROR_INVALID_HANDLE);
  IS(cuSurfRefGetArray(&sa, nullptr), CUDA_ERROR_INVALID_HANDLE);
  IS(cuTexRefDestroy(nullptr), CUDA_ERROR_INVALID_HANDLE);

  IS(cuTexRefCreate(&t), CUDA_SUCCESS);
  CUaddress_mode am = (CUaddress_mode)9;
  CUfilter_mode fm = (CUfilter_mode)9;
  unsigned fl = 99;
  CUarray_format ff = (CUarray_format)0;
  int nc = -1;
  IS(cuTexRefGetAddressMode(&am, t, 0), CUDA_SUCCESS);
  IS(cuTexRefGetFilterMode(&fm, t), CUDA_SUCCESS);
  IS(cuTexRefGetFlags(&fl, t), CUDA_SUCCESS);
  IS(cuTexRefGetFormat(&ff, &nc, t), CUDA_SUCCESS);
  check(am == CU_TR_ADDRESS_MODE_CLAMP && fm == CU_TR_FILTER_MODE_POINT && fl == 0 && ff == CU_AD_FORMAT_FLOAT &&
            nc == 1,
        "a new texture reference: clamped, point-filtered, one float channel");
  IS(cuTexRefSetAddressMode(t, 1, CU_TR_ADDRESS_MODE_MIRROR), CUDA_SUCCESS);
  IS(cuTexRefGetAddressMode(&am, t, 1), CUDA_SUCCESS);
  check(am == CU_TR_ADDRESS_MODE_MIRROR, "an address mode reads back");
  IS(cuTexRefSetAddressMode(t, 3, CU_TR_ADDRESS_MODE_MIRROR), CUDA_ERROR_INVALID_VALUE);
  IS(cuTexRefGetAddressMode(&am, t, 3), CUDA_ERROR_INVALID_VALUE);
  IS(cuTexRefSetFilterMode(t, CU_TR_FILTER_MODE_LINEAR), CUDA_SUCCESS);
  IS(cuTexRefGetFilterMode(&fm, t), CUDA_SUCCESS);
  check(fm == CU_TR_FILTER_MODE_LINEAR, "the filter mode reads back");
  IS(cuTexRefSetFlags(t, CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_READ_AS_INTEGER), CUDA_SUCCESS);
  IS(cuTexRefSetFlags(t, 0x1000), CUDA_ERROR_INVALID_VALUE);
  IS(cuTexRefGetFlags(&fl, t), CUDA_SUCCESS);
  check(fl == 3, "the flags read back, and an unknown one leaves them");
  IS(cuTexRefSetFormat(t, CU_AD_FORMAT_HALF, 4), CUDA_SUCCESS);
  IS(cuTexRefSetFormat(t, CU_AD_FORMAT_HALF, 3), CUDA_ERROR_INVALID_VALUE);
  IS(cuTexRefGetFormat(&ff, &nc, t), CUDA_SUCCESS);
  check(ff == CU_AD_FORMAT_HALF && nc == 4, "the format reads back");

  // Linear memory binds at the 512-byte texture alignment below the address,
  // and the distance is the offset returned.
  CUdeviceptr lin;
  IS(cuMemAlloc(&lin, 8192), CUDA_SUCCESS);
  const CUdeviceptr al = (lin + 511) / 512 * 512;
  size_t off = 99;
  CUdeviceptr ga = 0;
  IS(cuTexRefGetAddress(&ga, t), CUDA_ERROR_INVALID_VALUE);
  IS(cuTexRefSetAddress(&off, t, al + 4, 256), CUDA_SUCCESS);
  IS(cuTexRefGetAddress(&ga, t), CUDA_SUCCESS);
  check(off == 4 && ga == al, "an address 4 bytes past the alignment binds with offset 4");
  IS(cuTexRefSetAddress(&off, t, al + 516, 256), CUDA_SUCCESS);
  IS(cuTexRefGetAddress(&ga, t), CUDA_SUCCESS);
  check(off == 4 && ga == al + 512, "and one 516 past it at the next");
  IS(cuTexRefSetAddress(nullptr, t, al, 256), CUDA_SUCCESS);
  CUDA_ARRAY_DESCRIPTOR ad{};
  ad.Width = 16;
  ad.Height = 16;
  ad.Format = CU_AD_FORMAT_FLOAT;
  ad.NumChannels = 1;
  IS(cuTexRefSetAddress2D(t, &ad, al, 64), CUDA_SUCCESS);
  IS(cuTexRefSetAddress2D(t, &ad, al, 96), CUDA_SUCCESS);
  IS(cuTexRefSetAddress2D(t, &ad, al, 80), CUDA_ERROR_INVALID_VALUE);    // pitch not at 32 bytes
  IS(cuTexRefSetAddress2D(t, &ad, al, 32), CUDA_ERROR_INVALID_VALUE);    // pitch below a row
  IS(cuTexRefSetAddress2D(t, &ad, al + 256, 64), CUDA_ERROR_INVALID_VALUE);  // address not at 512
  IS(cuTexRefSetAddress2D(t, &ad, al + 512, 64), CUDA_SUCCESS);
  CUarray arr = nullptr, garr = nullptr;
  IS(cuArrayCreate(&arr, &ad), CUDA_SUCCESS);
  IS(cuTexRefGetArray(&garr, t), CUDA_ERROR_INVALID_VALUE);
  IS(cuTexRefSetArray(t, arr, 2), CUDA_ERROR_INVALID_VALUE);
  IS(cuTexRefSetArray(t, arr, CU_TRSA_OVERRIDE_FORMAT), CUDA_SUCCESS);
  IS(cuTexRefGetArray(&garr, t), CUDA_SUCCESS);
  IS(cuTexRefGetFormat(&ff, &nc, t), CUDA_SUCCESS);
  check(garr == arr && ff == CU_AD_FORMAT_FLOAT && nc == 1, "an array binds, and overrides the format");
  IS(cuTexRefGetAddress(&ga, t), CUDA_ERROR_INVALID_VALUE);
  IS(cuParamSetTexRef(kfill, CU_PARAM_TR_DEFAULT, t), CUDA_SUCCESS);
  IS(cuTexRefDestroy(t), CUDA_SUCCESS);
  IS(cuArrayDestroy(arr), CUDA_SUCCESS);
  IS(cuMemFree(lin), CUDA_SUCCESS);
}

// ---- texture and surface objects ----
static float fetch(CUfunction k, CUtexObject t, CUdeviceptr out, float x, float y) {
  void* args[] = {&t, &out, &x, &y};
  float v = -1;
  if (cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr) || cuCtxSynchronize() ||
      cuMemcpyDtoH(&v, out, sizeof v))
    return -1234;
  return v;
}

static void texture_objects(CUfunction ktex, CUfunction ksurf) {
  CUdeviceptr out;
  IS(cuMemAlloc(&out, 16), CUDA_SUCCESS);
  // A 4 x 4 float array holding 0..15.
  CUDA_ARRAY3D_DESCRIPTOR ad{};
  ad.Width = 4;
  ad.Height = 4;
  ad.Format = CU_AD_FORMAT_FLOAT;
  ad.NumChannels = 1;
  ad.Flags = CUDA_ARRAY3D_SURFACE_LDST;
  CUarray arr;
  IS(cuArray3DCreate(&arr, &ad), CUDA_SUCCESS);
  float texels[16];
  for (int i = 0; i < 16; ++i) texels[i] = static_cast<float>(i);
  CUDA_MEMCPY2D c{};
  c.srcMemoryType = CU_MEMORYTYPE_HOST;
  c.srcHost = texels;
  c.srcPitch = 16;
  c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  c.dstArray = arr;
  c.WidthInBytes = 16;
  c.Height = 4;
  IS(cuMemcpy2D(&c), CUDA_SUCCESS);
  CUDA_RESOURCE_DESC rd{};
  rd.resType = CU_RESOURCE_TYPE_ARRAY;
  rd.res.array.hArray = arr;
  CUDA_TEXTURE_DESC td{};
  td.addressMode[0] = td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  td.filterMode = CU_TR_FILTER_MODE_POINT;
  CUtexObject t = 0;
  IS(cuTexObjectCreate(&t, &rd, &td, nullptr), CUDA_SUCCESS);
  check(fetch(ktex, t, out, 2.5f, 1.5f) == 6.0f, "a point-filtered fetch from an array");
  check(fetch(ktex, t, out, 9.5f, 1.5f) == 7.0f, "clamped past the edge");
  IS(cuTexObjectDestroy(t), CUDA_SUCCESS);
  td.filterMode = CU_TR_FILTER_MODE_LINEAR;
  td.flags = CU_TRSF_NORMALIZED_COORDINATES;
  IS(cuTexObjectCreate(&t, &rd, &td, nullptr), CUDA_SUCCESS);
  check(fetch(ktex, t, out, 2.0f / 4, 1.5f / 4) == 5.5f, "a linear fetch at normalized coordinates");
  IS(cuTexObjectDestroy(t), CUDA_SUCCESS);
  td = {};
  td.addressMode[0] = td.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
  td.borderColor[0] = 7.0f;
  IS(cuTexObjectCreate(&t, &rd, &td, nullptr), CUDA_SUCCESS);
  check(fetch(ktex, t, out, -3.0f, 1.5f) == 7.0f, "the border colour outside");
  IS(cuTexObjectDestroy(t), CUDA_SUCCESS);
  // A surface over the same array: x counts bytes.
  CUsurfObject sobj = 0;
  IS(cuSurfObjectCreate(&sobj, &rd), CUDA_SUCCESS);
  {
    unsigned x = 8, y = 3;
    void* args[] = {&sobj, &out, &x, &y};
    IS(cuLaunchKernel(ksurf, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr), CUDA_SUCCESS);
    IS(cuCtxSynchronize(), CUDA_SUCCESS);
    float v = -1;
    IS(cuMemcpyDtoH(&v, out, sizeof v), CUDA_SUCCESS);
    check(v == 14.0f, "a surface load from the array");
  }
  IS(cuSurfObjectDestroy(sobj), CUDA_SUCCESS);

  // 8-bit texels come back as floats in [0, 1] unless read as integers.
  CUDA_ARRAY3D_DESCRIPTOR bd = ad;
  bd.Format = CU_AD_FORMAT_UNSIGNED_INT8;
  bd.Flags = 0;
  CUarray bytes;
  IS(cuArray3DCreate(&bytes, &bd), CUDA_SUCCESS);
  unsigned char b8[16];
  for (int i = 0; i < 16; ++i) b8[i] = static_cast<unsigned char>(i * 17);
  c.srcHost = b8;
  c.srcPitch = 4;
  c.dstArray = bytes;
  c.WidthInBytes = 4;
  IS(cuMemcpy2D(&c), CUDA_SUCCESS);
  rd.res.array.hArray = bytes;
  td = {};
  IS(cuTexObjectCreate(&t, &rd, &td, nullptr), CUDA_SUCCESS);
  check(fetch(ktex, t, out, 3.5f, 3.5f) == 1.0f && fetch(ktex, t, out, 0.5f, 1.5f) == 68.0f / 255.0f,
        "unsigned 8-bit texels promoted to [0, 1]");
  IS(cuTexObjectDestroy(t), CUDA_SUCCESS);

  // Pitched linear memory.
  CUdeviceptr pm;
  size_t pitch = 0;
  IS(cuMemAllocPitch(&pm, &pitch, 16, 4, 4), CUDA_SUCCESS);
  CUDA_MEMCPY2D up{};
  up.srcMemoryType = CU_MEMORYTYPE_HOST;
  up.srcHost = texels;
  up.srcPitch = 16;
  up.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  up.dstDevice = pm;
  up.dstPitch = pitch;
  up.WidthInBytes = 16;
  up.Height = 4;
  IS(cuMemcpy2D(&up), CUDA_SUCCESS);
  CUDA_RESOURCE_DESC pd{};
  pd.resType = CU_RESOURCE_TYPE_PITCH2D;
  pd.res.pitch2D.devPtr = pm;
  pd.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
  pd.res.pitch2D.numChannels = 1;
  pd.res.pitch2D.width = 4;
  pd.res.pitch2D.height = 4;
  pd.res.pitch2D.pitchInBytes = pitch;
  IS(cuTexObjectCreate(&t, &pd, &td, nullptr), CUDA_SUCCESS);
  check(fetch(ktex, t, out, 1.5f, 3.5f) == 13.0f, "a fetch from pitched memory");
  IS(cuTexObjectDestroy(t), CUDA_SUCCESS);
  IS(cuTexObjectCreate(nullptr, &pd, &td, nullptr), CUDA_ERROR_INVALID_VALUE);
  IS(cuTexObjectCreate(&t, nullptr, &td, nullptr), CUDA_ERROR_INVALID_VALUE);
  IS(cuArrayDestroy(arr), CUDA_SUCCESS);
  IS(cuArrayDestroy(bytes), CUDA_SUCCESS);
  IS(cuMemFree(pm), CUDA_SUCCESS);
  IS(cuMemFree(out), CUDA_SUCCESS);
}

// ---- the rest ----
static void misc(CUcontext ctx, CUdevice dev) {
  CUdevprop p{};
  IS(cuDeviceGetProperties(&p, dev), CUDA_SUCCESS);
  check(p.maxThreadsPerBlock == attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) &&
            p.maxThreadsDim[2] == attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z) &&
            p.maxGridSize[1] == attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y) &&
            p.sharedMemPerBlock == attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) &&
            p.SIMDWidth == attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE) &&
            p.regsPerBlock == attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK) &&
            p.memPitch == attr(CU_DEVICE_ATTRIBUTE_MAX_PITCH) &&
            p.textureAlign == attr(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT) && p.totalConstantMemory == 65536,
        "cuDeviceGetProperties agrees with the attributes");
  IS(cuDeviceGetProperties(nullptr, dev), CUDA_ERROR_INVALID_VALUE);
  int count = 0;
  cuDeviceGetCount(&count);
  IS(cuDeviceGetProperties(&p, count), CUDA_ERROR_INVALID_DEVICE);

  CUcontext at = nullptr;
  IS(cuCtxAttach(&at, 0), CUDA_SUCCESS);
  check(at == ctx, "cuCtxAttach returns the current context");
  IS(cuCtxAttach(&at, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuCtxDetach(ctx), CUDA_SUCCESS);
  CUdevice d = -1;
  IS(cuCtxGetDevice(&d), CUDA_SUCCESS);
  check(d == dev, "and a detach after an attach leaves it");

  IS(cuProfilerStart(), CUDA_SUCCESS);
  IS(cuProfilerStop(), CUDA_SUCCESS);
  CUdeviceptr gp;
  size_t gs;
  CUarray garr;
  IS(cuGraphicsUnregisterResource(nullptr), CUDA_ERROR_INVALID_HANDLE);
  IS(cuGraphicsMapResources(0, nullptr, nullptr), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphicsUnmapResources(0, nullptr, nullptr), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphicsResourceSetMapFlags(nullptr, 0), CUDA_ERROR_INVALID_HANDLE);
  IS(cuGraphicsResourceGetMappedPointer(&gp, &gs, nullptr), CUDA_ERROR_INVALID_HANDLE);
  IS(cuGraphicsSubResourceGetMappedArray(&garr, nullptr, 0, 0), CUDA_ERROR_INVALID_HANDLE);

  // Array to array, and the asynchronous array copies.
  CUDA_ARRAY_DESCRIPTOR ad{};
  ad.Width = 16;
  ad.Format = CU_AD_FORMAT_UNSIGNED_INT8;
  ad.NumChannels = 4;
  CUarray A, B;
  IS(cuArrayCreate(&A, &ad), CUDA_SUCCESS);
  IS(cuArrayCreate(&B, &ad), CUDA_SUCCESS);
  unsigned char src[64], back[64] = {};
  for (int i = 0; i < 64; ++i) src[i] = static_cast<unsigned char>(i);
  IS(cuMemcpyHtoAAsync(A, 0, src, 64, nullptr), CUDA_SUCCESS);
  CUdeviceptr zero;
  IS(cuMemAlloc(&zero, 64), CUDA_SUCCESS);
  IS(cuMemsetD8(zero, 0, 64), CUDA_SUCCESS);
  IS(cuMemcpyDtoA(B, 0, zero, 64), CUDA_SUCCESS);
  IS(cuMemcpyAtoA(B, 4, A, 8, 32), CUDA_SUCCESS);
  IS(cuMemcpyAtoHAsync(back, B, 0, 64, nullptr), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(back[3] == 0 && back[4] == 8 && back[35] == 39 && back[36] == 0, "cuMemcpyAtoA between offsets");
  IS(cuMemcpyAtoA(B, 40, A, 0, 32), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemcpyAtoA(B, 0, A, 40, 32), CUDA_ERROR_INVALID_VALUE);
  CUdeviceptr d2;
  IS(cuMemAlloc(&d2, 64), CUDA_SUCCESS);
  IS(cuMemcpyHtoD(zero, src, 64), CUDA_SUCCESS);
  IS(cuMemcpyDtoDAsync(d2, zero, 64, nullptr), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  IS(cuMemcpyDtoH(back, d2, 64), CUDA_SUCCESS);
  check(back[63] == 63, "cuMemcpyDtoDAsync");
  IS(cuArrayDestroy(A), CUDA_SUCCESS);
  IS(cuArrayDestroy(B), CUDA_SUCCESS);
  IS(cuMemFree(zero), CUDA_SUCCESS);
  IS(cuMemFree(d2), CUDA_SUCCESS);

  // A JIT info log is an empty string, not whatever the buffer held.
  char log[256];
  std::memset(log, 'X', sizeof log);
  CUjit_option opts[] = {CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER};
  void* vals[] = {(void*)(size_t)sizeof log, log};
  CUmodule m;
  IS(cuModuleLoadDataEx(&m, kPtx, 2, opts, vals), CUDA_SUCCESS);
  check(log[0] == '\0', "cuModuleLoadDataEx writes its (empty) info log");
  IS(cuModuleUnload(m), CUDA_SUCCESS);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);   // every line out, should a call crash
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) || cuDeviceGet(&dev, 0)) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  // CUDA 13's cuCtxCreate takes a parameter block; a zeroed one (what CUDA's
  // own samples pass) is an ordinary context.
#if CUDA_VERSION >= 13000
  CUctxCreateParams params{};
  IS(cuCtxCreate(&ctx, &params, 0, dev), CUDA_SUCCESS);
#else
  IS(cuCtxCreate(&ctx, 0, dev), CUDA_SUCCESS);
#endif
  check(attr(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED) == 1, "virtual memory management is supported");
  {
    // A module with a __managed__ variable (.attribute(.managed)) loads, and
    // the variable holds its initial value.
    CUmodule m;
    CUdeviceptr g = 0;
    size_t bytes = 0;
    unsigned v = 0;
    check(cuModuleLoadData(&m, kPtx) == CUDA_SUCCESS && cuModuleGetGlobal(&g, &bytes, m, "managed_global") == CUDA_SUCCESS &&
              bytes == 4 && cuMemcpyDtoH(&v, g, 4) == CUDA_SUCCESS && v == 9 && cuModuleUnload(m) == CUDA_SUCCESS,
          "a managed global in a module");
  }
  CUmodule mod;
  CUfunction kfill, kmixed, kbig, ktex, ksurf;
  if (cuModuleLoadData(&mod, kPtx) || cuModuleGetFunction(&kfill, mod, "fill") ||
      cuModuleGetFunction(&kmixed, mod, "mixed") || cuModuleGetFunction(&kbig, mod, "big") ||
      cuModuleGetFunction(&ktex, mod, "texread") || cuModuleGetFunction(&ksurf, mod, "surfread")) {
    std::printf("FAIL: the module did not load\n");
    return 1;
  }
  host_memory(kfill);
  memsets_2d();
  occupancy(kfill, kbig);
  peers(ctx);
  pci();
  mipmaps();
  legacy_launch(kfill, kmixed);
  texrefs(mod, kfill);
  texture_objects(ktex, ksurf);
  misc(ctx, dev);
  IS(cuModuleUnload(mod), CUDA_SUCCESS);
  IS(cuCtxDestroy(ctx), CUDA_SUCCESS);
  std::printf(failures ? "FAIL: %d driver checks\n" : "PASS: every driver check\n", failures);
  return failures ? 1 : 0;
}
