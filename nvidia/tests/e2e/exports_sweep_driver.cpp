// The driver functions the toolkit declares that this library once lacked, one after another with the argument cases
// they were measured on -- every answer an RTX 3060 gave under NVIDIA's CUDA 13.0 driver
// (nvidia/tests/data/exports_sweep_driver.rtx3060.expected). Not here: a call that crashes the card (a handle that is
// garbage), and what the simulator refuses on purpose where the card works (a LUID, a conditional-graph handle,
// green contexts): the sweep's documentation lists them. Lines of functions newer than the toolkit at hand are
// not compiled, and carry [12.8] or [13] so the runner can leave them out of the expected file.
//
// The program reads libk.cubin (exports_sweep_module.cu, built by the runner) from the directory it runs in.
#include <cuda.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

static const char* nm(CUresult r) {
  const char* s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}
static std::string g_notes;
static void note(const char* fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  g_notes += g_notes.empty() ? "" : "; ";
  g_notes += buf;
}
struct Ctx {
  CUdevice dev;
  CUcontext ctx;
};
static Ctx init() {
  Ctx c;
  cuInit(0);
  cuDeviceGet(&c.dev, 0);
  cuDevicePrimaryCtxRetain(&c.ctx, c.dev);
  cuCtxSetCurrent(c.ctx);
  return c;
}
static CUlibrary lib() {
  static std::vector<char> img;
  std::ifstream in("libk.cubin", std::ios::binary);
  img.assign(std::istreambuf_iterator<char>(in), {});
  CUlibrary l = nullptr;
  cuLibraryLoadData(&l, img.data(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  return l;
}
static CUmodule mod() {
  CUmodule m = nullptr;
  cuModuleLoad(&m, "libk.cubin");
  return m;
}
static CUfunction fun() {
  CUfunction f = nullptr;
  cuModuleGetFunction(&f, mod(), "k1");
  return f;
}
static CUkernel kern() {
  CUkernel k = nullptr;
  cuLibraryGetKernel(&k, lib(), "k1");
  return k;
}
static CUstream strm() {
  CUstream s = nullptr;
  cuStreamCreate(&s, 0);
  return s;
}
static CUevent evt() {
  CUevent e = nullptr;
  cuEventCreate(&e, 0);
  return e;
}
static CUarray arr() {
  CUDA_ARRAY_DESCRIPTOR d{};
  d.Width = 64;
  d.Height = 16;
  d.Format = CU_AD_FORMAT_FLOAT;
  d.NumChannels = 1;
  CUarray a = nullptr;
  cuArrayCreate(&a, &d);
  return a;
}
static CUmipmappedArray mip() {
  CUDA_ARRAY3D_DESCRIPTOR d{};
  d.Width = 64;
  d.Height = 64;
  d.Format = CU_AD_FORMAT_FLOAT;
  d.NumChannels = 1;
  CUmipmappedArray m = nullptr;
  cuMipmappedArrayCreate(&m, &d, 4);
  return m;
}

struct Case {
  const char* label;
  const char* tag;
  std::function<int()> run;
};
#define C(l, ...) cases.push_back({l, "", [&]() -> int { __VA_ARGS__ }})
#if CUDA_VERSION >= 12080
#define C128(l, ...) cases.push_back({l, " [12.8]", [&]() -> int { __VA_ARGS__ }})
#else
#define C128(l, ...) (void)0
#endif
#if CUDA_VERSION >= 13000
#define C13(l, ...) cases.push_back({l, " [13]", [&]() -> int { __VA_ARGS__ }})
#else
#define C13(l, ...) (void)0
#endif

int main() {
  std::vector<Case> cases;
  C("cuDeviceGetLuid null luid", { auto c = init(); unsigned mask = 77; return cuDeviceGetLuid(nullptr, &mask, c.dev); });
  C("cuDeviceGetLuid bad dev", { init(); char luid[8]; unsigned mask; return cuDeviceGetLuid(luid, &mask, 9); });
  C("cuDeviceGetTexture1DLinearMaxWidth float", { auto c = init(); size_t w = 77; CUresult r = cuDeviceGetTexture1DLinearMaxWidth(&w, CU_AD_FORMAT_FLOAT, 1, c.dev); note("w=%zu", w); return r; });
  C("cuDeviceGetTexture1DLinearMaxWidth uint8 x4", { auto c = init(); size_t w = 77; CUresult r = cuDeviceGetTexture1DLinearMaxWidth(&w, CU_AD_FORMAT_UNSIGNED_INT8, 4, c.dev); note("w=%zu", w); return r; });
  C("cuDeviceGetTexture1DLinearMaxWidth float x3", { auto c = init(); size_t w = 77; return cuDeviceGetTexture1DLinearMaxWidth(&w, CU_AD_FORMAT_FLOAT, 3, c.dev); });
  C("cuDeviceGetTexture1DLinearMaxWidth bad format", { auto c = init(); size_t w = 77; return cuDeviceGetTexture1DLinearMaxWidth(&w, (CUarray_format)0x99, 1, c.dev); });
  C("cuDeviceGetTexture1DLinearMaxWidth bad dev", { init(); size_t w = 77; return cuDeviceGetTexture1DLinearMaxWidth(&w, CU_AD_FORMAT_FLOAT, 1, 9); });
  C("cuDeviceGetTexture1DLinearMaxWidth null", { auto c = init(); return cuDeviceGetTexture1DLinearMaxWidth(nullptr, CU_AD_FORMAT_FLOAT, 1, c.dev); });
  C13("cuDeviceGetHostAtomicCapabilities", { auto c = init(); unsigned caps[3] = {9,9,9}; CUatomicOperation ops[3] = {CU_ATOMIC_OPERATION_INTEGER_ADD, CU_ATOMIC_OPERATION_FLOAT_ADD, CU_ATOMIC_OPERATION_CAS}; CUresult r = cuDeviceGetHostAtomicCapabilities(caps, ops, 3, c.dev); note("%u %u %u", caps[0], caps[1], caps[2]); return r; });
  C13("cuDeviceGetHostAtomicCapabilities null", { auto c = init(); CUatomicOperation ops[1] = {CU_ATOMIC_OPERATION_INTEGER_ADD}; return cuDeviceGetHostAtomicCapabilities(nullptr, ops, 1, c.dev); });
  C13("cuDeviceGetHostAtomicCapabilities count 0", { auto c = init(); unsigned caps[1]; CUatomicOperation ops[1] = {CU_ATOMIC_OPERATION_INTEGER_ADD}; return cuDeviceGetHostAtomicCapabilities(caps, ops, 0, c.dev); });
  C13("cuDeviceGetHostAtomicCapabilities bad op", { auto c = init(); unsigned caps[1]; CUatomicOperation ops[1] = {(CUatomicOperation)77}; return cuDeviceGetHostAtomicCapabilities(caps, ops, 1, c.dev); });
  C13("cuDeviceGetHostAtomicCapabilities bad dev", { init(); unsigned caps[1]; CUatomicOperation ops[1] = {CU_ATOMIC_OPERATION_INTEGER_ADD}; return cuDeviceGetHostAtomicCapabilities(caps, ops, 1, 9); });
  C13("cuDeviceGetP2PAtomicCapabilities 0->1", { init(); unsigned caps[1] = {9}; CUatomicOperation ops[1] = {CU_ATOMIC_OPERATION_INTEGER_ADD}; CUresult r = cuDeviceGetP2PAtomicCapabilities(caps, ops, 1, 0, 1); note("%u", caps[0]); return r; });
  C13("cuDeviceGetP2PAtomicCapabilities 0->0", { init(); unsigned caps[1]; CUatomicOperation ops[1] = {CU_ATOMIC_OPERATION_INTEGER_ADD}; return cuDeviceGetP2PAtomicCapabilities(caps, ops, 1, 0, 0); });
  C13("cuDeviceGetP2PAtomicCapabilities 0->9", { init(); unsigned caps[1]; CUatomicOperation ops[1] = {CU_ATOMIC_OPERATION_INTEGER_ADD}; return cuDeviceGetP2PAtomicCapabilities(caps, ops, 1, 0, 9); });
  C("cuDeviceGetNvSciSyncAttributes", { auto c = init(); char b[64]; return cuDeviceGetNvSciSyncAttributes(b, c.dev, 0); });
  C("cuDeviceGetNvSciSyncAttributes null", { auto c = init(); return cuDeviceGetNvSciSyncAttributes(nullptr, c.dev, 0); });
  C("cuDeviceGetNvSciSyncAttributes bad dev", { init(); char b[64]; return cuDeviceGetNvSciSyncAttributes(b, 9, 0); });
  C("cuFlushGPUDirectRDMAWrites", { init(); return cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX, CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER); });
  C("cuFlushGPUDirectRDMAWrites bad target", { init(); return cuFlushGPUDirectRDMAWrites((CUflushGPUDirectRDMAWritesTarget)5, CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER); });
  C("cuFlushGPUDirectRDMAWrites bad scope", { init(); return cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX, (CUflushGPUDirectRDMAWritesScope)7); });
  C13("cuCtxGetDevice_v2", { auto c = init(); CUdevice d = 77; CUresult r = cuCtxGetDevice_v2(&d, c.ctx); note("d=%d", d); return r; });
  C13("cuCtxGetDevice_v2 null ctx", { init(); CUdevice d = 77; CUresult r = cuCtxGetDevice_v2(&d, nullptr); note("d=%d", d); return r; });
  C13("cuCtxGetDevice_v2 null out", { auto c = init(); return cuCtxGetDevice_v2(nullptr, c.ctx); });
  C13("cuCtxSynchronize_v2", { auto c = init(); return cuCtxSynchronize_v2(c.ctx); });
  C13("cuCtxSynchronize_v2 null ctx", { init(); return cuCtxSynchronize_v2(nullptr); });
  C("cuCtxResetPersistingL2Cache", { init(); return cuCtxResetPersistingL2Cache(); });
  C128("cuCtxRecordEvent", { auto c = init(); return cuCtxRecordEvent(c.ctx, evt()); });
  C128("cuCtxRecordEvent null ctx", { init(); return cuCtxRecordEvent(nullptr, evt()); });
  C128("cuCtxRecordEvent null event", { auto c = init(); return cuCtxRecordEvent(c.ctx, nullptr); });
  C128("cuCtxWaitEvent", { auto c = init(); CUevent e = evt(); cuEventRecord(e, 0); return cuCtxWaitEvent(c.ctx, e); });
  C128("cuCtxWaitEvent unrecorded", { auto c = init(); return cuCtxWaitEvent(c.ctx, evt()); });
  C128("cuCtxWaitEvent null event", { auto c = init(); return cuCtxWaitEvent(c.ctx, nullptr); });
  C128("cuModuleGetFunctionCount", { init(); unsigned n = 77; CUresult r = cuModuleGetFunctionCount(&n, mod()); note("n=%u", n); return r; });
  C128("cuModuleGetFunctionCount null", { init(); return cuModuleGetFunctionCount(nullptr, mod()); });
  C128("cuModuleGetFunctionCount null mod", { init(); unsigned n; return cuModuleGetFunctionCount(&n, nullptr); });
  C128("cuModuleEnumerateFunctions", { init(); CUfunction fs[8] = {}; CUresult r = cuModuleEnumerateFunctions(fs, 3, mod()); note("%d %d %d", fs[0] != 0, fs[1] != 0, fs[2] != 0); return r; });
  C128("cuModuleEnumerateFunctions more than exist", { init(); CUfunction fs[8] = {}; return cuModuleEnumerateFunctions(fs, 8, mod()); });
  C128("cuModuleEnumerateFunctions 0", { init(); CUfunction fs[8] = {}; return cuModuleEnumerateFunctions(fs, 0, mod()); });
  C128("cuModuleEnumerateFunctions null", { init(); return cuModuleEnumerateFunctions(nullptr, 3, mod()); });
  C128("cuModuleEnumerateFunctions null mod", { init(); CUfunction fs[8]; return cuModuleEnumerateFunctions(fs, 3, nullptr); });
  C("cuLibraryLoadFromFile", { init(); CUlibrary l = nullptr; CUresult r = cuLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); note("l=%d", l != 0); return r; });
  C("cuLibraryLoadFromFile missing", { init(); CUlibrary l = nullptr; return cuLibraryLoadFromFile(&l, "/nonexistent", nullptr, nullptr, 0, nullptr, nullptr, 0); });
  C("cuLibraryLoadFromFile null path", { init(); CUlibrary l = nullptr; return cuLibraryLoadFromFile(&l, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0); });
  C("cuLibraryLoadFromFile null out", { init(); return cuLibraryLoadFromFile(nullptr, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); });
  C("cuLibraryLoadFromFile not an image", { init(); CUlibrary l = nullptr; return cuLibraryLoadFromFile(&l, "/etc/hostname", nullptr, nullptr, 0, nullptr, nullptr, 0); });
  C128("cuLibraryGetKernelCount", { init(); unsigned n = 77; CUresult r = cuLibraryGetKernelCount(&n, lib()); note("n=%u", n); return r; });
  C128("cuLibraryGetKernelCount null", { init(); return cuLibraryGetKernelCount(nullptr, lib()); });
  C128("cuLibraryGetKernelCount null lib", { init(); unsigned n; return cuLibraryGetKernelCount(&n, nullptr); });
  C128("cuLibraryEnumerateKernels", { init(); CUkernel ks[8] = {}; CUresult r = cuLibraryEnumerateKernels(ks, 3, lib()); note("%d %d %d", ks[0] != 0, ks[1] != 0, ks[2] != 0); return r; });
  C128("cuLibraryEnumerateKernels 8", { init(); CUkernel ks[8] = {}; return cuLibraryEnumerateKernels(ks, 8, lib()); });
  C128("cuLibraryEnumerateKernels 0", { init(); CUkernel ks[8] = {}; return cuLibraryEnumerateKernels(ks, 0, lib()); });
  C128("cuLibraryEnumerateKernels null", { init(); return cuLibraryEnumerateKernels(nullptr, 3, lib()); });
  C128("cuLibraryEnumerateKernels null lib", { init(); CUkernel ks[8]; return cuLibraryEnumerateKernels(ks, 3, nullptr); });
  C128("cuKernelGetLibrary", { init(); CUkernel k = kern(); CUlibrary l = nullptr; CUresult r = cuKernelGetLibrary(&l, k); note("l=%d", l != 0); return r; });
  C128("cuKernelGetLibrary null", { init(); return cuKernelGetLibrary(nullptr, kern()); });
  C128("cuKernelGetLibrary null kernel", { init(); CUlibrary l; return cuKernelGetLibrary(&l, nullptr); });
  C("cuLibraryGetGlobal", { init(); CUdeviceptr p = 0; size_t n = 77; CUresult r = cuLibraryGetGlobal(&p, &n, lib(), "gvar"); note("n=%zu p=%d", n, p != 0); return r; });
  C("cuLibraryGetGlobal missing", { init(); CUdeviceptr p; size_t n; return cuLibraryGetGlobal(&p, &n, lib(), "nosuch"); });
  C("cuLibraryGetGlobal null name", { init(); CUdeviceptr p; size_t n; return cuLibraryGetGlobal(&p, &n, lib(), nullptr); });
  C("cuLibraryGetGlobal null lib", { init(); CUdeviceptr p; size_t n; return cuLibraryGetGlobal(&p, &n, nullptr, "gvar"); });
  C("cuLibraryGetGlobal null both outs", { init(); return cuLibraryGetGlobal(nullptr, nullptr, lib(), "gvar"); });
  C("cuLibraryGetGlobal a kernel", { init(); CUdeviceptr p; size_t n; return cuLibraryGetGlobal(&p, &n, lib(), "k1"); });
  C("cuLibraryGetManaged", { init(); CUdeviceptr p = 0; size_t n = 77; CUresult r = cuLibraryGetManaged(&p, &n, lib(), "mvar"); note("n=%zu p=%d", n, p != 0); return r; });
  C("cuLibraryGetManaged not managed", { init(); CUdeviceptr p; size_t n; return cuLibraryGetManaged(&p, &n, lib(), "gvar"); });
  C("cuLibraryGetManaged missing", { init(); CUdeviceptr p; size_t n; return cuLibraryGetManaged(&p, &n, lib(), "nosuch"); });
  C("cuLibraryGetUnifiedFunction", { init(); void* f = nullptr; return cuLibraryGetUnifiedFunction(&f, lib(), "k1"); });
  C("cuLibraryGetUnifiedFunction null lib", { init(); void* f; return cuLibraryGetUnifiedFunction(&f, nullptr, "k1"); });
  C("cuKernelSetCacheConfig", { auto c = init(); return cuKernelSetCacheConfig(kern(), CU_FUNC_CACHE_PREFER_SHARED, c.dev); });
  C("cuKernelSetCacheConfig bad config", { auto c = init(); return cuKernelSetCacheConfig(kern(), (CUfunc_cache)9, c.dev); });
  C("cuKernelSetCacheConfig null kernel", { auto c = init(); return cuKernelSetCacheConfig(nullptr, CU_FUNC_CACHE_PREFER_SHARED, c.dev); });
  C("cuKernelSetCacheConfig bad dev", { init(); return cuKernelSetCacheConfig(kern(), CU_FUNC_CACHE_PREFER_SHARED, 9); });
  C("cuKernelSetCacheConfig current dev (-1?)", { init(); return cuKernelSetCacheConfig(kern(), CU_FUNC_CACHE_PREFER_SHARED, -1); });
  C128("cuKernelGetParamInfo", { init(); size_t off = 77, sz = 77; CUresult r = cuKernelGetParamInfo(kern(), 0, &off, &sz); note("off=%zu sz=%zu", off, sz); return r; });
  C128("cuKernelGetParamInfo past end", { init(); size_t off, sz; return cuKernelGetParamInfo(kern(), 1, &off, &sz); });
  C128("cuKernelGetParamInfo null off", { init(); size_t sz; return cuKernelGetParamInfo(kern(), 0, nullptr, &sz); });
  C128("cuKernelGetParamInfo null size", { init(); size_t off; return cuKernelGetParamInfo(kern(), 0, &off, nullptr); });
  C128("cuKernelGetParamInfo null kernel", { init(); size_t off, sz; return cuKernelGetParamInfo(nullptr, 0, &off, &sz); });
  C("cuFuncGetModule", { init(); CUmodule m = nullptr; CUresult r = cuFuncGetModule(&m, fun()); note("m=%d", m != 0); return r; });
  C("cuFuncGetModule null", { init(); return cuFuncGetModule(nullptr, fun()); });
  C("cuFuncGetModule null func", { init(); CUmodule m; return cuFuncGetModule(&m, nullptr); });
  C128("cuFuncGetName", { init(); const char* n = nullptr; CUresult r = cuFuncGetName(&n, fun()); note("%s", n ? n : "(null)"); return r; });
  C128("cuFuncGetName null", { init(); return cuFuncGetName(nullptr, fun()); });
  C128("cuFuncGetName null func", { init(); const char* n; return cuFuncGetName(&n, nullptr); });
  C128("cuFuncGetParamInfo", { init(); size_t off = 77, sz = 77; CUresult r = cuFuncGetParamInfo(fun(), 0, &off, &sz); note("off=%zu sz=%zu", off, sz); return r; });
  C128("cuFuncGetParamInfo past end", { init(); size_t off, sz; return cuFuncGetParamInfo(fun(), 5, &off, &sz); });
  C128("cuFuncGetParamInfo null func", { init(); size_t off, sz; return cuFuncGetParamInfo(nullptr, 0, &off, &sz); });
  C("cuOccupancyAvailableDynamicSMemPerBlock", { init(); size_t s = 77; CUresult r = cuOccupancyAvailableDynamicSMemPerBlock(&s, fun(), 4, 128); note("%zu", s); return r; });
  C("cuOccupancyAvailableDynamicSMemPerBlock too many blocks", { init(); size_t s = 77; return cuOccupancyAvailableDynamicSMemPerBlock(&s, fun(), 100, 128); });
  C("cuOccupancyAvailableDynamicSMemPerBlock 0 blocks", { init(); size_t s = 77; return cuOccupancyAvailableDynamicSMemPerBlock(&s, fun(), 0, 128); });
  C("cuOccupancyAvailableDynamicSMemPerBlock null", { init(); return cuOccupancyAvailableDynamicSMemPerBlock(nullptr, fun(), 1, 128); });
  C("cuOccupancyAvailableDynamicSMemPerBlock null func", { init(); size_t s; return cuOccupancyAvailableDynamicSMemPerBlock(&s, nullptr, 1, 128); });
  C128("cuDeviceRegisterAsyncNotification", { auto c = init(); CUasyncCallbackHandle h = nullptr; CUresult r = cuDeviceRegisterAsyncNotification(c.dev, [](CUasyncNotificationInfo*, void*, CUasyncCallbackHandle) {}, nullptr, &h); note("h=%d", h != 0); if (!r) note("unregister %s", nm(cuDeviceUnregisterAsyncNotification(c.dev, h))); return r; });
  C128("cuDeviceRegisterAsyncNotification null cb", { auto c = init(); CUasyncCallbackHandle h; return cuDeviceRegisterAsyncNotification(c.dev, nullptr, nullptr, &h); });
  C128("cuDeviceRegisterAsyncNotification bad dev", { init(); CUasyncCallbackHandle h; return cuDeviceRegisterAsyncNotification(9, [](CUasyncNotificationInfo*, void*, CUasyncCallbackHandle) {}, nullptr, &h); });
  C128("cuDeviceUnregisterAsyncNotification null", { auto c = init(); return cuDeviceUnregisterAsyncNotification(c.dev, nullptr); });
  C("cuMemcpy3DPeer", { init(); CUDA_MEMCPY3D_PEER p{}; return cuMemcpy3DPeer(&p); });
  C13("cuMemcpyBatchAsync_v2 count 0", { init(); CUdeviceptr d[1] = {1}; size_t sz[1] = {1}; CUmemcpyAttributes at{}; size_t idx[1] = {0}; return cuMemcpyBatchAsync_v2(d, d, sz, 0, &at, idx, 1, 0); });
  C13("cuMemcpyBatchAsync_v2 null", { init(); size_t sz[1] = {1}; CUmemcpyAttributes at{}; size_t idx[1] = {0}; return cuMemcpyBatchAsync_v2(nullptr, nullptr, sz, 1, &at, idx, 1, 0); });
  C13("cuMemcpyBatchAsync_v2 bad order", { init(); CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16]; CUdeviceptr dst[1] = {d1}; CUdeviceptr src[1] = {(CUdeviceptr)h}; size_t sz[1] = {16}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_INVALID; size_t idx[1] = {0}; return cuMemcpyBatchAsync_v2(dst, src, sz, 1, &at, idx, 1, 0); });
  C13("cuMemcpy3DBatchAsync_v2 count 0", { init(); CUDA_MEMCPY3D_BATCH_OP op{}; return cuMemcpy3DBatchAsync_v2(0, &op, 0, 0); });
  C13("cuMemcpy3DBatchAsync_v2 null", { init(); return cuMemcpy3DBatchAsync_v2(1, nullptr, 0, 0); });
  C13("cuMemcpy3DBatchAsync_v2 flags", { init(); CUDA_MEMCPY3D_BATCH_OP op{}; return cuMemcpy3DBatchAsync_v2(1, &op, 1, 0); });
  C13("cuMemcpy3DBatchAsync_v2 zero extent", { init(); CUDA_MEMCPY3D_BATCH_OP op{}; op.src.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER; return cuMemcpy3DBatchAsync_v2(1, &op, 0, 0); });
  C("cuArrayGetMemoryRequirements", { auto c = init(); CUDA_ARRAY_MEMORY_REQUIREMENTS r{}; return cuArrayGetMemoryRequirements(&r, arr(), c.dev); });
  C("cuArrayGetMemoryRequirements null", { auto c = init(); return cuArrayGetMemoryRequirements(nullptr, arr(), c.dev); });
  C("cuArrayGetMemoryRequirements null array", { auto c = init(); CUDA_ARRAY_MEMORY_REQUIREMENTS r{}; return cuArrayGetMemoryRequirements(&r, nullptr, c.dev); });
  C("cuMipmappedArrayGetMemoryRequirements", { auto c = init(); CUDA_ARRAY_MEMORY_REQUIREMENTS r{}; return cuMipmappedArrayGetMemoryRequirements(&r, mip(), c.dev); });
  C("cuMipmappedArrayGetMemoryRequirements null", { auto c = init(); CUDA_ARRAY_MEMORY_REQUIREMENTS r{}; return cuMipmappedArrayGetMemoryRequirements(&r, nullptr, c.dev); });
  C("cuArrayGetSparseProperties", { init(); CUDA_ARRAY_SPARSE_PROPERTIES p{}; return cuArrayGetSparseProperties(&p, arr()); });
  C("cuArrayGetSparseProperties null array", { init(); CUDA_ARRAY_SPARSE_PROPERTIES p{}; return cuArrayGetSparseProperties(&p, nullptr); });
  C("cuMipmappedArrayGetSparseProperties", { init(); CUDA_ARRAY_SPARSE_PROPERTIES p{}; return cuMipmappedArrayGetSparseProperties(&p, mip()); });
  C("cuMipmappedArrayGetSparseProperties null", { init(); CUDA_ARRAY_SPARSE_PROPERTIES p{}; return cuMipmappedArrayGetSparseProperties(&p, nullptr); });
  C("cuArrayGetPlane", { init(); CUarray pl = nullptr; return cuArrayGetPlane(&pl, arr(), 0); });
  C("cuArrayGetPlane null array", { init(); CUarray pl = nullptr; return cuArrayGetPlane(&pl, nullptr, 0); });
  C("cuArrayGetPlane null out", { init(); return cuArrayGetPlane(nullptr, arr(), 0); });
  C("cuMemGetHandleForAddressRange", { init(); CUdeviceptr d; cuMemAlloc(&d, 1 << 20); int fd = -1; return cuMemGetHandleForAddressRange(&fd, d, 1 << 20, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0); });
  C("cuMemGetHandleForAddressRange null", { init(); CUdeviceptr d; cuMemAlloc(&d, 1 << 20); return cuMemGetHandleForAddressRange(nullptr, d, 1 << 20, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0); });
  C("cuMemGetHandleForAddressRange bad type", { init(); CUdeviceptr d; cuMemAlloc(&d, 1 << 20); int fd; return cuMemGetHandleForAddressRange(&fd, d, 1 << 20, (CUmemRangeHandleType)5, 0); });
  C128("cuMemBatchDecompressAsync", { init(); CUmemDecompressParams p{}; size_t ei = 77; CUresult r = cuMemBatchDecompressAsync(&p, 1, 0, &ei, 0); note("ei=%zu", ei); return r; });
  C128("cuMemBatchDecompressAsync count 0", { init(); CUmemDecompressParams p{}; size_t ei; return cuMemBatchDecompressAsync(&p, 0, 0, &ei, 0); });
  C("cuMemMapArrayAsync", { init(); CUarrayMapInfo m{}; return cuMemMapArrayAsync(&m, 1, 0); });
  C("cuMemMapArrayAsync count 0", { init(); CUarrayMapInfo m{}; return cuMemMapArrayAsync(&m, 0, 0); });
  C("cuMemMapArrayAsync null", { init(); return cuMemMapArrayAsync(nullptr, 1, 0); });
  C13("cuMemGetDefaultMemPool", { init(); CUmemoryPool p = nullptr; CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; CUresult r = cuMemGetDefaultMemPool(&p, &l, CU_MEM_ALLOCATION_TYPE_PINNED); note("p=%d", p != 0); return r; });
  C13("cuMemGetDefaultMemPool invalid type", { init(); CUmemoryPool p; CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; return cuMemGetDefaultMemPool(&p, &l, CU_MEM_ALLOCATION_TYPE_INVALID); });
  C13("cuMemGetDefaultMemPool managed", { init(); CUmemoryPool p; CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; return cuMemGetDefaultMemPool(&p, &l, (CUmemAllocationType)2); });
  C13("cuMemGetDefaultMemPool bad dev", { init(); CUmemoryPool p; CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 9}; return cuMemGetDefaultMemPool(&p, &l, CU_MEM_ALLOCATION_TYPE_PINNED); });
  C13("cuMemGetDefaultMemPool null", { init(); CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; return cuMemGetDefaultMemPool(nullptr, &l, CU_MEM_ALLOCATION_TYPE_PINNED); });
  C13("cuMemGetDefaultMemPool null loc", { init(); CUmemoryPool p; return cuMemGetDefaultMemPool(&p, nullptr, CU_MEM_ALLOCATION_TYPE_PINNED); });
  C13("cuMemGetMemPool", { init(); CUmemoryPool p = nullptr, d = nullptr; CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; CUresult r = cuMemGetMemPool(&p, &l, CU_MEM_ALLOCATION_TYPE_PINNED); cuMemGetDefaultMemPool(&d, &l, CU_MEM_ALLOCATION_TYPE_PINNED); note("same as default %d", p == d); return r; });
  C13("cuMemSetMemPool", { init(); CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; CUmemPoolProps pp{}; pp.allocType = CU_MEM_ALLOCATION_TYPE_PINNED; pp.location = l; CUmemoryPool mine; cuMemPoolCreate(&mine, &pp); CUresult r = cuMemSetMemPool(&l, CU_MEM_ALLOCATION_TYPE_PINNED, mine); CUmemoryPool cur; cuMemGetMemPool(&cur, &l, CU_MEM_ALLOCATION_TYPE_PINNED); note("now mine %d", cur == mine); return r; });
  C13("cuMemSetMemPool null pool", { init(); CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; return cuMemSetMemPool(&l, CU_MEM_ALLOCATION_TYPE_PINNED, nullptr); });
  C13("cuMemPrefetchBatchAsync", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); CUdeviceptr dp[1] = {m}; size_t sz[1] = {1 << 20}; CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; size_t li[1] = {0}; return cuMemPrefetchBatchAsync(dp, sz, 1, &l, li, 1, 0, 0); });
  C13("cuMemDiscardBatchAsync", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); CUdeviceptr dp[1] = {m}; size_t sz[1] = {1 << 20}; return cuMemDiscardBatchAsync(dp, sz, 1, 0, 0); });
  C13("cuMemDiscardAndPrefetchBatchAsync", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); CUdeviceptr dp[1] = {m}; size_t sz[1] = {1 << 20}; CUmemLocation l{CU_MEM_LOCATION_TYPE_DEVICE, 0}; size_t li[1] = {0}; return cuMemDiscardAndPrefetchBatchAsync(dp, sz, 1, &l, li, 1, 0, 0); });
  C("cuMemRangeGetAttribute readmostly", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); int v = -9; CUresult r = cuMemRangeGetAttribute(&v, sizeof v, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, m, 1 << 20); note("v=%d", v); return r; });
  C("cuMemRangeGetAttribute preferred", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); int v = -9; CUresult r = cuMemRangeGetAttribute(&v, sizeof v, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, m, 1 << 20); note("v=%d", v); return r; });
  C("cuMemRangeGetAttribute last prefetch", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); int v = -9; CUresult r = cuMemRangeGetAttribute(&v, sizeof v, CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION, m, 1 << 20); note("v=%d", v); return r; });
  C("cuMemRangeGetAttribute accessedby", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); int v[2] = {-9, -9}; CUresult r = cuMemRangeGetAttribute(v, sizeof v, CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, m, 1 << 20); note("v=%d %d", v[0], v[1]); return r; });
  C("cuMemRangeGetAttribute bad size", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); int v; return cuMemRangeGetAttribute(&v, 1, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, m, 1 << 20); });
  C("cuMemRangeGetAttribute not managed", { init(); CUdeviceptr d; cuMemAlloc(&d, 1 << 20); int v; return cuMemRangeGetAttribute(&v, sizeof v, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, d, 1 << 20); });
  C("cuMemRangeGetAttribute bad attr", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); int v; return cuMemRangeGetAttribute(&v, sizeof v, (CUmem_range_attribute)99, m, 1 << 20); });
  C("cuMemRangeGetAttribute null", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); return cuMemRangeGetAttribute(nullptr, 4, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, m, 1 << 20); });
  C("cuMemRangeGetAttributes", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); int a = -9, b = -9; void* data[2] = {&a, &b}; size_t sizes[2] = {4, 4}; CUmem_range_attribute at[2] = {CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION}; CUresult r = cuMemRangeGetAttributes(data, sizes, at, 2, m, 1 << 20); note("%d %d", a, b); return r; });
  C("cuMemRangeGetAttributes count 0", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); return cuMemRangeGetAttributes(nullptr, nullptr, nullptr, 0, m, 1 << 20); });
  C("cuMemRangeGetAttributes null", { init(); CUdeviceptr m; cuMemAllocManaged(&m, 1 << 20, CU_MEM_ATTACH_GLOBAL); return cuMemRangeGetAttributes(nullptr, nullptr, nullptr, 1, m, 1 << 20); });
  C128("cuStreamGetDevice", { auto c = init(); CUdevice d = 77; CUresult r = cuStreamGetDevice(strm(), &d); note("d=%d (dev %d)", d, c.dev); return r; });
  C128("cuStreamGetDevice null stream", { init(); CUdevice d = 77; CUresult r = cuStreamGetDevice(nullptr, &d); note("d=%d", d); return r; });
  C128("cuStreamGetDevice null out", { init(); return cuStreamGetDevice(strm(), nullptr); });
  C("cuStreamGetId", { init(); unsigned long long id = 77; CUresult r = cuStreamGetId(strm(), &id); note("id>0 %d", id > 0); return r; });
  C("cuStreamGetId null stream", { init(); unsigned long long id = 77; CUresult r = cuStreamGetId(nullptr, &id); note("id set %d", id != 77); return r; });
  C("cuStreamGetId null out", { init(); return cuStreamGetId(strm(), nullptr); });
  C("cuStreamCopyAttributes", { init(); return cuStreamCopyAttributes(strm(), strm()); });
  C("cuStreamCopyAttributes null dst", { init(); return cuStreamCopyAttributes(nullptr, strm()); });
  C("cuStreamCopyAttributes null src", { init(); return cuStreamCopyAttributes(strm(), nullptr); });
  C("cuStreamGetAttribute sync policy", { init(); CUstreamAttrValue v; std::memset(&v, 0x55, sizeof v); CUresult r = cuStreamGetAttribute(strm(), CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, &v); note("v=%d", (int)v.syncPolicy); return r; });
  C("cuStreamGetAttribute priority", { init(); CUstreamAttrValue v; std::memset(&v, 0x55, sizeof v); CUresult r = cuStreamGetAttribute(strm(), CU_STREAM_ATTRIBUTE_PRIORITY, &v); note("v=%d", (int)v.priority); return r; });
  C("cuStreamGetAttribute access policy window", { init(); CUstreamAttrValue v; std::memset(&v, 0x55, sizeof v); CUresult r = cuStreamGetAttribute(strm(), CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &v); note("num_bytes=%zu hit=%.1f", v.accessPolicyWindow.num_bytes, v.accessPolicyWindow.hitRatio); return r; });
  C("cuStreamGetAttribute bad attr", { init(); CUstreamAttrValue v; return cuStreamGetAttribute(strm(), (CUstreamAttrID)99, &v); });
  C("cuStreamSetAttribute sync policy", { init(); CUstreamAttrValue v{}; v.syncPolicy = CU_SYNC_POLICY_SPIN; CUstream s = strm(); CUresult r = cuStreamSetAttribute(s, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, &v); CUstreamAttrValue g{}; cuStreamGetAttribute(s, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, &g); note("now %d", (int)g.syncPolicy); return r; });
  C("cuStreamSetAttribute priority", { init(); CUstreamAttrValue v{}; v.priority = -1; return cuStreamSetAttribute(strm(), CU_STREAM_ATTRIBUTE_PRIORITY, &v); });
  C("cuStreamSetAttribute access policy window", { init(); CUdeviceptr d; cuMemAlloc(&d, 1 << 20); CUstreamAttrValue v{}; v.accessPolicyWindow.base_ptr = (void*)d; v.accessPolicyWindow.num_bytes = 1 << 20; v.accessPolicyWindow.hitRatio = 0.5f; v.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING; v.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING; return cuStreamSetAttribute(strm(), CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &v); });
  C("cuStreamSetAttribute bad attr", { init(); CUstreamAttrValue v{}; return cuStreamSetAttribute(strm(), (CUstreamAttrID)99, &v); });
  C("cuImportExternalMemory fd", { init(); CUexternalMemory em = nullptr; CUDA_EXTERNAL_MEMORY_HANDLE_DESC d{}; d.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD; d.handle.fd = 5; d.size = 4096; return cuImportExternalMemory(&em, &d); });
  C("cuImportExternalMemory null desc", { init(); CUexternalMemory em; return cuImportExternalMemory(&em, nullptr); });
  C("cuImportExternalMemory null out", { init(); CUDA_EXTERNAL_MEMORY_HANDLE_DESC d{}; return cuImportExternalMemory(nullptr, &d); });
  C("cuImportExternalMemory bad type", { init(); CUexternalMemory em; CUDA_EXTERNAL_MEMORY_HANDLE_DESC d{}; d.type = (CUexternalMemoryHandleType)99; return cuImportExternalMemory(&em, &d); });
  C("cuImportExternalMemory win32 type", { init(); CUexternalMemory em; CUDA_EXTERNAL_MEMORY_HANDLE_DESC d{}; d.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32; d.size = 4096; return cuImportExternalMemory(&em, &d); });
  C("cuExternalMemoryGetMappedBuffer null", { init(); CUdeviceptr p; CUDA_EXTERNAL_MEMORY_BUFFER_DESC d{}; return cuExternalMemoryGetMappedBuffer(&p, nullptr, &d); });
  C("cuExternalMemoryGetMappedMipmappedArray null", { init(); CUmipmappedArray m; CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC d{}; return cuExternalMemoryGetMappedMipmappedArray(&m, nullptr, &d); });
  C("cuDestroyExternalMemory null", { init(); return cuDestroyExternalMemory(nullptr); });
  C("cuImportExternalSemaphore fd", { init(); CUexternalSemaphore es; CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC d{}; d.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD; d.handle.fd = 5; return cuImportExternalSemaphore(&es, &d); });
  C("cuImportExternalSemaphore null", { init(); CUexternalSemaphore es; return cuImportExternalSemaphore(&es, nullptr); });
  C("cuSignalExternalSemaphoresAsync null", { init(); CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS p{}; return cuSignalExternalSemaphoresAsync(nullptr, &p, 1, 0); });
  C("cuWaitExternalSemaphoresAsync null", { init(); CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS p{}; return cuWaitExternalSemaphoresAsync(nullptr, &p, 1, 0); });
  C("cuWaitExternalSemaphoresAsync count 0", { init(); CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS p{}; CUexternalSemaphore e[1] = {nullptr}; return cuWaitExternalSemaphoresAsync(e, &p, 0, 0); });
  C("cuDestroyExternalSemaphore null", { init(); return cuDestroyExternalSemaphore(nullptr); });
  C("cuGraphExternalSemaphoresSignalNodeGetParams null node", { init(); CUDA_EXT_SEM_SIGNAL_NODE_PARAMS p{}; return cuGraphExternalSemaphoresSignalNodeGetParams(nullptr, &p); });
  C("cuGraphExternalSemaphoresSignalNodeSetParams null node", { init(); CUDA_EXT_SEM_SIGNAL_NODE_PARAMS p{}; return cuGraphExternalSemaphoresSignalNodeSetParams(nullptr, &p); });
  C("cuGraphExternalSemaphoresWaitNodeGetParams null node", { init(); CUDA_EXT_SEM_WAIT_NODE_PARAMS p{}; return cuGraphExternalSemaphoresWaitNodeGetParams(nullptr, &p); });
  C("cuGraphExternalSemaphoresWaitNodeSetParams null node", { init(); CUDA_EXT_SEM_WAIT_NODE_PARAMS p{}; return cuGraphExternalSemaphoresWaitNodeSetParams(nullptr, &p); });
  C("cuGraphExecExternalSemaphoresSignalNodeSetParams null", { init(); CUDA_EXT_SEM_SIGNAL_NODE_PARAMS p{}; return cuGraphExecExternalSemaphoresSignalNodeSetParams(nullptr, nullptr, &p); });
  C("cuGraphExecExternalSemaphoresWaitNodeSetParams null", { init(); CUDA_EXT_SEM_WAIT_NODE_PARAMS p{}; return cuGraphExecExternalSemaphoresWaitNodeSetParams(nullptr, nullptr, &p); });
  C128("cuGraphConditionalHandleCreate bad flags", { auto c = init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphConditionalHandle h = 0; return cuGraphConditionalHandleCreate(&h, g, c.ctx, 1, 7); });
  C128("cuGraphConditionalHandleCreate null graph", { auto c = init(); CUgraphConditionalHandle h = 0; return cuGraphConditionalHandleCreate(&h, nullptr, c.ctx, 1, 0); });
  C128("cuGraphConditionalHandleCreate null out", { auto c = init(); CUgraph g; cuGraphCreate(&g, 0); return cuGraphConditionalHandleCreate(nullptr, g, c.ctx, 1, 0); });
  C128("cuGraphConditionalHandleCreate null ctx", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphConditionalHandle h; return cuGraphConditionalHandleCreate(&h, g, nullptr, 1, 0); });
  C("cuLaunchCooperativeKernelMultiDevice", { init(); CUDA_LAUNCH_PARAMS p{}; return cuLaunchCooperativeKernelMultiDevice(&p, 1, 0); });
  C("cuLaunchCooperativeKernelMultiDevice count 0", { init(); CUDA_LAUNCH_PARAMS p{}; return cuLaunchCooperativeKernelMultiDevice(&p, 0, 0); });
  C("cuLaunchCooperativeKernelMultiDevice null", { init(); return cuLaunchCooperativeKernelMultiDevice(nullptr, 1, 0); });
  C128("cuCoredumpGetAttribute enable", { init(); int v = -9; size_t sz = sizeof v; CUresult r = cuCoredumpGetAttribute(CU_COREDUMP_ENABLE_ON_EXCEPTION, &v, &sz); note("v=%d sz=%zu", v, sz); return r; });
  C128("cuCoredumpGetAttribute trigger host", { init(); int v = -9; size_t sz = sizeof v; CUresult r = cuCoredumpGetAttribute(CU_COREDUMP_TRIGGER_HOST, &v, &sz); note("v=%d sz=%zu", v, sz); return r; });
  C128("cuCoredumpGetAttribute lightweight", { init(); int v = -9; size_t sz = sizeof v; CUresult r = cuCoredumpGetAttribute(CU_COREDUMP_LIGHTWEIGHT, &v, &sz); note("v=%d sz=%zu", v, sz); return r; });
  C128("cuCoredumpGetAttribute pipe", { init(); char b[256] = {}; size_t sz = sizeof b; CUresult r = cuCoredumpGetAttribute(CU_COREDUMP_PIPE, b, &sz); note("'%s' sz=%zu", b, sz); return r; });
  C128("cuCoredumpGetAttribute file", { init(); char b[512] = {}; size_t sz = sizeof b; CUresult r = cuCoredumpGetAttribute(CU_COREDUMP_FILE, b, &sz); note("'%s' sz=%zu", b, sz); return r; });
  C128("cuCoredumpGetAttribute bad attr", { init(); int v; size_t sz = 4; return cuCoredumpGetAttribute((CUcoredumpSettings)99, &v, &sz); });
  C128("cuCoredumpGetAttribute null", { init(); size_t sz = 4; return cuCoredumpGetAttribute(CU_COREDUMP_ENABLE_ON_EXCEPTION, nullptr, &sz); });
  C128("cuCoredumpSetAttributeGlobal after init", { init(); int v = 1; size_t sz = sizeof v; return cuCoredumpSetAttributeGlobal(CU_COREDUMP_ENABLE_ON_EXCEPTION, &v, &sz); });
  C128("cuCoredumpSetAttribute file", { init(); const char* p = "/tmp/x.nvcudmp"; size_t sz = strlen(p) + 1; return cuCoredumpSetAttribute(CU_COREDUMP_FILE, (void*)p, &sz); });
  C128("cuCoredumpSetAttribute bad attr", { init(); int v = 1; size_t sz = 4; return cuCoredumpSetAttribute((CUcoredumpSettings)99, &v, &sz); });
  C128("cuDeviceGetDevResource bad type", { auto c = init(); CUdevResource r{}; return cuDeviceGetDevResource(c.dev, &r, (CUdevResourceType)99); });
  C128("cuDeviceGetDevResource bad dev", { init(); CUdevResource r{}; return cuDeviceGetDevResource(9, &r, CU_DEV_RESOURCE_TYPE_SM); });
  C128("cuGreenCtxDestroy null", { init(); return cuGreenCtxDestroy(nullptr); });
  C128("cuCtxFromGreenCtx null", { init(); CUcontext x; return cuCtxFromGreenCtx(&x, nullptr); });
  C128("cuGreenCtxGetDevResource null", { init(); CUdevResource r; return cuGreenCtxGetDevResource(nullptr, &r, CU_DEV_RESOURCE_TYPE_SM); });
  C128("cuGreenCtxRecordEvent null", { init(); return cuGreenCtxRecordEvent(nullptr, evt()); });
  C128("cuGreenCtxWaitEvent null", { init(); return cuGreenCtxWaitEvent(nullptr, evt()); });
  C128("cuGreenCtxStreamCreate null", { init(); CUstream s; return cuGreenCtxStreamCreate(&s, nullptr, 1, 0); });
  C13("cuGreenCtxGetId null", { init(); unsigned long long id; return cuGreenCtxGetId(nullptr, &id); });
  C13("cuLogsCurrent", { init(); CUlogIterator it = 777; CUresult r = cuLogsCurrent(&it, 0); note("it=%u", it); return r; });
  C13("cuLogsCurrent null", { init(); return cuLogsCurrent(nullptr, 0); });
  C13("cuLogsDumpToMemory", { init(); CUlogIterator it = 0; char b[64]; size_t sz = 64; CUresult r = cuLogsDumpToMemory(&it, b, &sz, 0); note("sz=%zu", sz); return r; });
  C13("cuLogsDumpToMemory null buf", { init(); CUlogIterator it = 0; size_t sz = 64; return cuLogsDumpToMemory(&it, nullptr, &sz, 0); });
  C13("cuLogsDumpToFile", { init(); CUlogIterator it = 0; return cuLogsDumpToFile(&it, "/tmp/cu_logs_probe2.txt", 0); });
  C13("cuLogsDumpToFile null path", { init(); CUlogIterator it = 0; return cuLogsDumpToFile(&it, nullptr, 0); });
  C13("cuLogsRegisterCallback", { init(); CUlogsCallbackHandle h = nullptr; CUresult r = cuLogsRegisterCallback([](void*, CUlogLevel, char*, size_t) {}, nullptr, &h); note("h!=0 %d", h != 0); if (!r) note("unregister %s", nm(cuLogsUnregisterCallback(h))); return r; });
  C13("cuLogsRegisterCallback null cb", { init(); CUlogsCallbackHandle h; return cuLogsRegisterCallback(nullptr, nullptr, &h); });
  C13("cuLogsUnregisterCallback null", { init(); return cuLogsUnregisterCallback(nullptr); });
  C128("cuCheckpointProcessGetState self", { init(); CUprocessState st = (CUprocessState)77; CUresult r = cuCheckpointProcessGetState(getpid(), &st); note("st=%d", (int)st); return r; });
  C128("cuCheckpointProcessGetState null", { init(); return cuCheckpointProcessGetState(getpid(), nullptr); });
  C128("cuCheckpointProcessGetState bad pid", { init(); CUprocessState st; return cuCheckpointProcessGetState(999999, &st); });
  C128("cuCheckpointProcessGetRestoreThreadId", { init(); int tid = -9; CUresult r = cuCheckpointProcessGetRestoreThreadId(getpid(), &tid); note("tid set %d", tid != -9); return r; });
  C128("cuCheckpointProcessLock", { init(); return cuCheckpointProcessLock(getpid(), nullptr); });
  C128("cuCheckpointProcessCheckpoint", { init(); return cuCheckpointProcessCheckpoint(getpid(), nullptr); });
  C128("cuCheckpointProcessRestore", { init(); return cuCheckpointProcessRestore(getpid(), nullptr); });
  C128("cuCheckpointProcessUnlock", { init(); return cuCheckpointProcessUnlock(getpid(), nullptr); });
  C128("cuTensorMapEncodeIm2colWide", { init(); CUtensorMap m; CUdeviceptr d; cuMemAlloc(&d, 1 << 20); cuuint64_t dim[4] = {64, 64, 4, 1}; cuuint64_t str[3] = {64 * 2, 64 * 64 * 2, 64 * 64 * 4 * 2}; cuuint32_t es[4] = {1, 1, 1, 1}; return cuTensorMapEncodeIm2colWide(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, (void*)d, dim, str, 0, 0, 16, 16, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_IM2COL_WIDE_MODE_W, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); });
  C128("cuTensorMapEncodeIm2colWide null", { init(); CUtensorMap m; cuuint32_t es[4] = {1,1,1,1}; return cuTensorMapEncodeIm2colWide(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, nullptr, nullptr, nullptr, 0, 0, 16, 16, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_IM2COL_WIDE_MODE_W, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); });
  C("cuTexObjectGetResourceDesc", { init(); CUdeviceptr d; cuMemAlloc(&d, 4096); CUDA_RESOURCE_DESC rd{}; rd.resType = CU_RESOURCE_TYPE_LINEAR; rd.res.linear.devPtr = d; rd.res.linear.format = CU_AD_FORMAT_FLOAT; rd.res.linear.numChannels = 1; rd.res.linear.sizeInBytes = 4096; CUDA_TEXTURE_DESC td{}; CUtexObject t; cuTexObjectCreate(&t, &rd, &td, nullptr); CUDA_RESOURCE_DESC g{}; CUresult r = cuTexObjectGetResourceDesc(&g, t); note("type=%d size=%zu", (int)g.resType, g.res.linear.sizeInBytes); return r; });
  C("cuTexObjectGetTextureDesc", { init(); CUdeviceptr d; cuMemAlloc(&d, 4096); CUDA_RESOURCE_DESC rd{}; rd.resType = CU_RESOURCE_TYPE_LINEAR; rd.res.linear.devPtr = d; rd.res.linear.format = CU_AD_FORMAT_FLOAT; rd.res.linear.numChannels = 1; rd.res.linear.sizeInBytes = 4096; CUDA_TEXTURE_DESC td{}; td.filterMode = CU_TR_FILTER_MODE_LINEAR; td.flags = CU_TRSF_READ_AS_INTEGER; CUtexObject t; cuTexObjectCreate(&t, &rd, &td, nullptr); CUDA_TEXTURE_DESC g{}; CUresult r = cuTexObjectGetTextureDesc(&g, t); note("filter=%d flags=%u", (int)g.filterMode, g.flags); return r; });
  C("cuTexObjectGetResourceViewDesc", { init(); CUdeviceptr d; cuMemAlloc(&d, 4096); CUDA_RESOURCE_DESC rd{}; rd.resType = CU_RESOURCE_TYPE_LINEAR; rd.res.linear.devPtr = d; rd.res.linear.format = CU_AD_FORMAT_FLOAT; rd.res.linear.numChannels = 1; rd.res.linear.sizeInBytes = 4096; CUDA_TEXTURE_DESC td{}; CUtexObject t; cuTexObjectCreate(&t, &rd, &td, nullptr); CUDA_RESOURCE_VIEW_DESC g{}; CUresult r = cuTexObjectGetResourceViewDesc(&g, t); note("fmt=%d", (int)g.format); return r; });
  C("cuTexObjectGetResourceDesc bad", { init(); CUDA_RESOURCE_DESC g{}; return cuTexObjectGetResourceDesc(&g, 0x1234); });
  C("cuTexObjectGetResourceDesc null", { init(); return cuTexObjectGetResourceDesc(nullptr, 0x1234); });
  C("cuSurfObjectGetResourceDesc", { init(); CUarray a = arr(); CUDA_RESOURCE_DESC rd{}; rd.resType = CU_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = a; CUsurfObject so; CUresult r0 = cuSurfObjectCreate(&so, &rd); note("create %s", nm(r0)); CUDA_RESOURCE_DESC g{}; CUresult r = cuSurfObjectGetResourceDesc(&g, so); note("type=%d", (int)g.resType); return r; });
  C("cuSurfObjectGetResourceDesc bad", { init(); CUDA_RESOURCE_DESC g{}; return cuSurfObjectGetResourceDesc(&g, 0x1234); });
  C("cuTexRefSetMipmappedArray", { init(); CUtexref t; cuTexRefCreate(&t); return cuTexRefSetMipmappedArray(t, mip(), CU_TRSF_READ_AS_INTEGER); });
  C("cuTexRefGetMipmappedArray unset", { init(); CUtexref t; cuTexRefCreate(&t); CUmipmappedArray m = (CUmipmappedArray)0x77; CUresult r = cuTexRefGetMipmappedArray(&m, t); note("m=%p", (void*)m); return r; });
  C("cuTexRefSetMipmapFilterMode", { init(); CUtexref t; cuTexRefCreate(&t); CUresult r = cuTexRefSetMipmapFilterMode(t, CU_TR_FILTER_MODE_LINEAR); CUfilter_mode m = CU_TR_FILTER_MODE_POINT; CUresult g = cuTexRefGetMipmapFilterMode(&m, t); note("get %s %d", nm(g), (int)m); return r; });
  C("cuTexRefSetMipmapFilterMode bad", { init(); CUtexref t; cuTexRefCreate(&t); return cuTexRefSetMipmapFilterMode(t, (CUfilter_mode)7); });
  C("cuTexRefSetMipmapLevelBias", { init(); CUtexref t; cuTexRefCreate(&t); CUresult r = cuTexRefSetMipmapLevelBias(t, 1.5f); float b = 0; CUresult g = cuTexRefGetMipmapLevelBias(&b, t); note("get %s %.2f", nm(g), b); return r; });
  C("cuTexRefSetMipmapLevelClamp", { init(); CUtexref t; cuTexRefCreate(&t); CUresult r = cuTexRefSetMipmapLevelClamp(t, 1.f, 3.f); float a = 0, b = 0; CUresult g = cuTexRefGetMipmapLevelClamp(&a, &b, t); note("get %s %.2f %.2f", nm(g), a, b); return r; });
  C("cuTexRefSetMaxAnisotropy", { init(); CUtexref t; cuTexRefCreate(&t); CUresult r = cuTexRefSetMaxAnisotropy(t, 8); int a = 0; CUresult g = cuTexRefGetMaxAnisotropy(&a, t); note("get %s %d", nm(g), a); return r; });
  C("cuTexRefGetMaxAnisotropy default", { init(); CUtexref t; cuTexRefCreate(&t); int a = -9; CUresult g = cuTexRefGetMaxAnisotropy(&a, t); note("%d", a); return g; });
  C("cuTexRefSetBorderColor", { init(); CUtexref t; cuTexRefCreate(&t); float c[4] = {1, 0.5f, 0.25f, 0}; CUresult r = cuTexRefSetBorderColor(t, c); float g[4] = {-1, -1, -1, -1}; CUresult x = cuTexRefGetBorderColor(g, t); note("get %s %.2f %.2f %.2f %.2f", nm(x), g[0], g[1], g[2], g[3]); return r; });
  C("cuTexRefGetBorderColor default", { init(); CUtexref t; cuTexRefCreate(&t); float g[4] = {-1, -1, -1, -1}; CUresult x = cuTexRefGetBorderColor(g, t); note("%.2f %.2f %.2f %.2f", g[0], g[1], g[2], g[3]); return x; });
  C("cuTexRefSetMaxAnisotropy null ref", { init(); return cuTexRefSetMaxAnisotropy(nullptr, 4); });

  // Each case in a process of its own: the card crashes on some arguments, and a case leaves the context and its
  // handles behind; the parent never starts CUDA.
  int bad = 0;
  for (const Case& c : cases) {
    std::fflush(stdout);
    const pid_t k = fork();
    if (k == 0) {
      g_notes.clear();
      const CUresult r = static_cast<CUresult>(c.run());
      std::printf("%s%s: %s%s%s\n", c.label, c.tag, nm(r), g_notes.empty() ? "" : " -- ", g_notes.c_str());
      std::fflush(stdout);
      _exit(0);
    }
    int status = 0;
    waitpid(k, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::printf("%s%s: CRASHED (signal %d)\n", c.label, c.tag, WIFSIGNALED(status) ? WTERMSIG(status) : 0);
      ++bad;
    }
  }
  std::printf("%s: %zu cases\n", bad ? "FAIL" : "PASS", cases.size());
  return bad != 0;
}
