// The driver functions the toolkit declares that this library once lacked, one after another with the argument cases
// they were measured on -- every answer an RTX 3060 gave under NVIDIA's CUDA 13.0 driver
// (nvidia/tests/data/exports_sweep_driver.rtx3060.expected). Not here: a call that crashes the card (a handle that is
// garbage), and what the simulator refuses on purpose where the card works (a LUID, a conditional-graph handle,
// green contexts): the sweep's documentation lists them. Lines of functions newer than the toolkit at hand are
// not compiled, and carry [12.8] or [13] so the runner can leave them out of the expected file.
//
// The program reads libk.cubin (exports_sweep_module.cu, built by the runner) from the directory it runs in.
#include <cuda.h>
#include <dlfcn.h>

#include <dlfcn.h>

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

static CUmodule modf() {
  CUmodule m = nullptr;
  cuModuleLoad(&m, "libf.cubin");
  return m;
}
static CUfunction funf(const char* name) {
  CUfunction f = nullptr;
  cuModuleGetFunction(&f, modf(), name);
  return f;
}
static CUkernel kernf(const char* name) {
  static std::vector<char> img;
  std::ifstream in("libf.cubin", std::ios::binary);
  img.assign(std::istreambuf_iterator<char>(in), {});
  CUlibrary l = nullptr;
  cuLibraryLoadData(&l, img.data(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  CUkernel k = nullptr;
  cuLibraryGetKernel(&k, l, name);
  return k;
}
// What cuFuncGetAttribute says of a function: attributes 0 to 15, but for the registers and local memory (the
// compiler's).
static void note_func(CUfunction f) {
  int v[16];
  for (int a = 0; a < 16; ++a)
    if (cuFuncGetAttribute(&v[a], static_cast<CUfunction_attribute>(a), f) != CUDA_SUCCESS) v[a] = -777;
  note("maxThreads %d, static %d, const %d, ptx %d, binary %d, cacheCA %d, maxDynamic %d, carveout %d, clusterMustBeSet %d, "
       "cluster %d %d %d, nonPortable %d, policy %d",
       v[0], v[1], v[2], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
}
static int dynamic_of(CUfunction f) {
  int v = -99;
  cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f);
  return v;
}
// A launch of one warp with `dyn` bytes of dynamic shared memory: the code it answers.
static int launch_code(CUfunction f, unsigned dyn) {
  CUdeviceptr d = 0;
  cuMemAlloc(&d, 4096);
  void* args[1] = {&d};
  const CUresult r = cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, dyn, nullptr, args, nullptr);
  cuCtxSynchronize();
  cuMemFree(d);
  return static_cast<int>(r);
}

// The functions only the CUDA 13.2 header declares are called by name through the loader, with the few structures they
// take declared here at the sizes and offsets cuda.h gives them, so the same cases build with every toolkit and run
// against a driver or a simulator that has them (an RTX 3060 under driver 596.36 answered them all). One that is
// missing answers CUDA_ERROR_NOT_FOUND.
template <class F>
static F drv(const char* name) {
  return reinterpret_cast<F>(dlsym(RTLD_DEFAULT, name));
}
#define D132(name, Sig, ...) (drv<Sig>(#name) ? drv<Sig>(#name)(__VA_ARGS__) : CUDA_ERROR_NOT_FOUND)
struct MemcpyAttr {                 // CUmemcpyAttributes
  int srcAccessOrder;
  int srcLocType, srcLocId, dstLocType, dstLocId;
  unsigned flags;
};
struct Operand3D {                  // CUmemcpy3DOperand
  int type;                         // 1 pointer, 2 array
  unsigned long long ptr;
  size_t rowLength, layerHeight;
  int locType, locId;
};
struct BatchOp3D {                  // CUDA_MEMCPY3D_BATCH_OP
  Operand3D src, dst;
  size_t width, height, depth;
  int srcAccessOrder;
  unsigned flags;
};
struct NodeParams {                 // CUgraphNodeParams: a type, a union of 232 bytes
  int type;
  int reserved0[3];
  unsigned char u[232];
  long long reserved2;
  template <class T> T get(size_t off) const { T v; std::memcpy(&v, u + off, sizeof v); return v; }
};
static_assert(sizeof(MemcpyAttr) == 24 && sizeof(BatchOp3D) == 112 && sizeof(NodeParams) == 256, "cuda.h layouts");
using GraphFn = CUresult (*)(CUgraph, unsigned*);

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

#if CUDA_VERSION >= 12080
// CUDA 12.8 and 12.9 spell the batched copies cuMemcpyBatchAsync and cuMemcpy3DBatchAsync and take a failure index;
// CUDA 13 renamed them _v2 and dropped it, and the driver (the card's too) exports both. Called by name, so that
// the same lines run with either toolkit's cuda.h.
using BatchV1 = CUresult (*)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t, CUmemcpyAttributes*, size_t*, size_t, size_t*, CUstream);
using Batch3DV1 = CUresult (*)(size_t, CUDA_MEMCPY3D_BATCH_OP*, size_t*, unsigned long long, CUstream);
template <class F>
static F by_name(const char* name) { return reinterpret_cast<F>(dlsym(RTLD_DEFAULT, name)); }
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
  // The batched copies as CUDA 12.8 and 12.9 spell them, with the failure index (-1 is SIZE_MAX, "no copy in particular").
  C128("cuMemcpyBatchAsync count 0", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d[1] = {1}; size_t sz[1] = {1}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t idx[1] = {0}, fi = 77; CUresult r = f(d, d, sz, 0, &at, idx, 1, &fi, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpyBatchAsync null dsts", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; size_t sz[1] = {1}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t idx[1] = {0}, fi = 77; CUresult r = f(nullptr, nullptr, sz, 1, &at, idx, 1, &fi, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpyBatchAsync null failIdx", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16] = {}; CUdeviceptr dst[1] = {d1}; CUdeviceptr src[1] = {(CUdeviceptr)h}; size_t sz[1] = {16}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t idx[1] = {0}; CUstream s = strm(); CUresult r = f(dst, src, sz, 1, &at, idx, 1, nullptr, s); cuStreamSynchronize(s); return r; });
  C128("cuMemcpyBatchAsync legacy stream", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16] = {}; CUdeviceptr dst[1] = {d1}; CUdeviceptr src[1] = {(CUdeviceptr)h}; size_t sz[1] = {16}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t idx[1] = {0}, fi = 77; CUresult r = f(dst, src, sz, 1, &at, idx, 1, &fi, 0); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpyBatchAsync bad order", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16] = {}; CUdeviceptr dst[1] = {d1}; CUdeviceptr src[1] = {(CUdeviceptr)h}; size_t sz[1] = {16}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_INVALID; size_t idx[1] = {0}, fi = 77; CUresult r = f(dst, src, sz, 1, &at, idx, 1, &fi, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpyBatchAsync size 0", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16] = {}; CUdeviceptr dst[1] = {d1}; CUdeviceptr src[1] = {(CUdeviceptr)h}; size_t sz[1] = {0}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t idx[1] = {0}, fi = 77; CUresult r = f(dst, src, sz, 1, &at, idx, 1, &fi, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpyBatchAsync two copies", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16]; for (int i = 0; i < 16; ++i) h[i] = (char)(i + 1); CUdeviceptr dst[2] = {d1, d1 + 64}; CUdeviceptr src[2] = {(CUdeviceptr)h, (CUdeviceptr)h}; size_t sz[2] = {16, 16}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t idx[1] = {0}, fi = 77; CUstream s = strm(); CUresult r = f(dst, src, sz, 2, &at, idx, 1, &fi, s); cuStreamSynchronize(s); char o[16] = {}; cuMemcpyDtoH(o, d1 + 64, 16); note("fail=%lld back=%d,%d", (long long)fi, o[0], o[15]); return r; });
  C128("cuMemcpy3DBatchAsync count 0", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUDA_MEMCPY3D_BATCH_OP op{}; size_t fi = 77; CUresult r = f(0, &op, &fi, 0, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync null list", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; size_t fi = 77; CUresult r = f(1, nullptr, &fi, 0, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync flags", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUDA_MEMCPY3D_BATCH_OP op{}; size_t fi = 77; CUresult r = f(1, &op, &fi, 1, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync zero extent", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUDA_MEMCPY3D_BATCH_OP op{}; op.src.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER; size_t fi = 77; CUresult r = f(1, &op, &fi, 0, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync legacy stream", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUDA_MEMCPY3D_BATCH_OP op{}; op.src.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.extent.width = 4; op.extent.height = 1; op.extent.depth = 1; op.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t fi = 77; CUresult r = f(1, &op, &fi, 0, 0); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpyBatchAsync second size 0", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16] = {}; CUdeviceptr dst[2] = {d1, d1 + 64}; CUdeviceptr src[2] = {(CUdeviceptr)h, (CUdeviceptr)h}; size_t sz[2] = {16, 0}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t idx[1] = {0}, fi = 77; CUresult r = f(dst, src, sz, 2, &at, idx, 1, &fi, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpyBatchAsync attribute flags", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16] = {}; CUdeviceptr dst[1] = {d1}; CUdeviceptr src[1] = {(CUdeviceptr)h}; size_t sz[1] = {16}; CUmemcpyAttributes at{}; at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; at.flags = 2; size_t idx[1] = {0}, fi = 77; CUresult r = f(dst, src, sz, 1, &at, idx, 1, &fi, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpyBatchAsync attribute index past count", { init(); auto f = by_name<BatchV1>("cuMemcpyBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); char h[16] = {}; CUdeviceptr dst[1] = {d1}; CUdeviceptr src[1] = {(CUdeviceptr)h}; size_t sz[1] = {16}; CUmemcpyAttributes at[2] = {}; at[0].srcAccessOrder = at[1].srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t idx[2] = {0, 5}, fi = 77; CUresult r = f(dst, src, sz, 1, at, idx, 2, &fi, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync bad order", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUDA_MEMCPY3D_BATCH_OP op{}; op.src.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.extent.width = 4; op.extent.height = 1; op.extent.depth = 1; op.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_INVALID; size_t fi = 77; CUresult r = f(1, &op, &fi, 0, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync bad operand type", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUDA_MEMCPY3D_BATCH_OP op{}; op.src.type = (CUmemcpy3DOperandType)9; op.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.extent.width = 4; op.extent.height = 1; op.extent.depth = 1; op.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t fi = 77; CUresult r = f(1, &op, &fi, 0, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync op flags", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUDA_MEMCPY3D_BATCH_OP op{}; op.src.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.extent.width = 4; op.extent.height = 1; op.extent.depth = 1; op.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; op.flags = 2; size_t fi = 77; CUresult r = f(1, &op, &fi, 0, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync second zero extent", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); CUDA_MEMCPY3D_BATCH_OP op[2] = {}; for (auto& o : op) { o.src.type = CU_MEMCPY_OPERAND_TYPE_POINTER; o.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER; o.src.op.ptr.ptr = d1; o.dst.op.ptr.ptr = d1 + 256; o.extent.width = 4; o.extent.height = 1; o.extent.depth = 1; o.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; } op[1].extent.depth = 0; size_t fi = 77; CUresult r = f(2, op, &fi, 0, strm()); note("fail=%lld", (long long)fi); return r; });
  C128("cuMemcpy3DBatchAsync copies", { init(); auto f = by_name<Batch3DV1>("cuMemcpy3DBatchAsync"); if (!f) return CUDA_ERROR_NOT_FOUND; CUdeviceptr d1; cuMemAlloc(&d1, 4096); unsigned char h[16]; for (int i = 0; i < 16; ++i) h[i] = (unsigned char)(i + 1); cuMemcpyHtoD(d1, h, 16); CUDA_MEMCPY3D_BATCH_OP op{}; op.src.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.dst.type = CU_MEMCPY_OPERAND_TYPE_POINTER; op.src.op.ptr.ptr = d1; op.dst.op.ptr.ptr = d1 + 256; op.extent.width = 16; op.extent.height = 1; op.extent.depth = 1; op.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM; size_t fi = 77; CUstream s = strm(); CUresult r = f(1, &op, &fi, 0, s); cuStreamSynchronize(s); unsigned char o[16] = {}; cuMemcpyDtoH(o, d1 + 256, 16); note("fail=%lld back=%d,%d", (long long)fi, o[0], o[15]); return r; });
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
  C("cuStreamGetAttribute sync policy", { init(); CUstreamAttrValue v; std::memset(&v, 0x55, sizeof v); CUresult r = cuStreamGetAttribute(strm(), CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, &v); int raw; std::memcpy(&raw, &v.syncPolicy, sizeof raw); /* the card writes -1, not one of the enum's values: read the bytes */ note("v=%d", raw); return r; });
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

  // ---- function and kernel attributes (nvidia/src/func_attrs.hpp has the rules; every line is an RTX 3060 under CUDA 13.0)
  C("cuFuncGetAttribute, no static shared memory", { init(); note_func(funf("fa_dyn")); return CUDA_SUCCESS; });
  C("cuFuncGetAttribute, 256 bytes static", { init(); note_func(funf("fa_s256")); return CUDA_SUCCESS; });
  C("cuFuncGetAttribute, 40000 bytes static", { init(); note_func(funf("fa_s40000")); return CUDA_SUCCESS; });
  C("cuFuncGetAttribute, 48 KiB static", { init(); note_func(funf("fa_s49152")); return CUDA_SUCCESS; });
  C("cuFuncGetAttribute, __launch_bounds__(128)", { init(); note_func(funf("fa_lb128")); return CUDA_SUCCESS; });
  C("cuFuncGetAttribute, __launch_bounds__(512, 2)", { init(); note_func(funf("fa_lb512")); return CUDA_SUCCESS; });
  C("cuFuncGetAttribute, registers limit the block", { init(); note_func(funf("fa_heavy400")); return CUDA_SUCCESS; });
  C("cuFuncGetAttribute, module constants", { init(); note_func(funf("fa_const")); return CUDA_SUCCESS; });
  C("cuFuncGetAttribute attribute 16", { init(); int v = -5; CUresult r = cuFuncGetAttribute(&v, (CUfunction_attribute)16, funf("fa_dyn")); note("v=%d", v); return r; });
  C("cuFuncGetAttribute attribute -1", { init(); int v = -5; CUresult r = cuFuncGetAttribute(&v, (CUfunction_attribute)-1, funf("fa_dyn")); note("v=%d", v); return r; });
  C("cuFuncGetAttribute attribute 99", { init(); int v = -5; CUresult r = cuFuncGetAttribute(&v, (CUfunction_attribute)99, funf("fa_dyn")); note("v=%d", v); return r; });
  C("cuFuncGetAttribute null function", { init(); int v; return cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, nullptr); });
  C("cuFuncGetAttribute null result", { init(); return cuFuncGetAttribute(nullptr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funf("fa_dyn")); });
  C("cuFuncSetAttribute null function", { init(); return cuFuncSetAttribute(nullptr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 3); });
  for (int a = 0; a <= 17; ++a) {
    static char labels[24][64];
    static int n = 0;
    std::snprintf(labels[n], sizeof labels[n], "cuFuncSetAttribute attribute %d to 1", a);
    const char* label = labels[n++];
    cases.push_back({label, "", [a]() -> int { init(); CUfunction f = funf("fa_lb128"); CUresult r = cuFuncSetAttribute(f, static_cast<CUfunction_attribute>(a), 1); note_func(f); return r; }});
  }
  static const int dyn_values[] = {-1, 0, 1, 1024, 49152, 49153, 65536, 101376, 101377, 102400};
  for (const int v : dyn_values) {
    static char labels[16][96];
    static int n = 0;
    std::snprintf(labels[n], sizeof labels[n], "cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES %d, no static", v);
    const char* label = labels[n++];
    cases.push_back({label, "", [v]() -> int { init(); CUfunction f = funf("fa_dyn"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, v); note("now %d", dynamic_of(f)); return r; }});
  }
  C("cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES 101120, 256 static", { init(); CUfunction f = funf("fa_s256"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 101120); note("now %d", dynamic_of(f)); return r; });
  C("cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES 101121, 256 static", { init(); CUfunction f = funf("fa_s256"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 101121); note("now %d", dynamic_of(f)); return r; });
  C("cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES 61376, 40000 static", { init(); CUfunction f = funf("fa_s40000"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 61376); note("now %d", dynamic_of(f)); return r; });
  C("cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES 61377, 40000 static", { init(); CUfunction f = funf("fa_s40000"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 61377); note("now %d", dynamic_of(f)); return r; });
  C("cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES 52225, 48 KiB static", { init(); CUfunction f = funf("fa_s49152"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 52225); note("now %d", dynamic_of(f)); return r; });
  static const int carve_values[] = {-2, -1, 0, 1, 100, 101};
  for (const int v : carve_values) {
    static char labels[16][96];
    static int n = 0;
    std::snprintf(labels[n], sizeof labels[n], "cuFuncSetAttribute PREFERRED_SHARED_MEMORY_CARVEOUT %d", v);
    const char* label = labels[n++];
    cases.push_back({label, "", [v]() -> int { init(); CUfunction f = funf("fa_lb128"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, v); int now = -99; cuFuncGetAttribute(&now, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, f); note("now %d", now); return r; }});
  }
  C("cuFuncSetAttribute required cluster dimensions", { init(); CUfunction f = funf("fa_lb128"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, 3); cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, 2); cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, 5); note_func(f); return r; });
  C("cuFuncSetAttribute required cluster width -1", { init(); return cuFuncSetAttribute(funf("fa_lb128"), CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, -1); });
  C("cuFuncSetAttribute CLUSTER_SCHEDULING_POLICY_PREFERENCE 2", { init(); CUfunction f = funf("fa_lb128"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, 2); note_func(f); return r; });
  C("cuFuncSetAttribute CLUSTER_SCHEDULING_POLICY_PREFERENCE 3", { init(); return cuFuncSetAttribute(funf("fa_lb128"), CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, 3); });
  C("cuFuncSetAttribute NON_PORTABLE_CLUSTER_SIZE_ALLOWED 7", { init(); CUfunction f = funf("fa_lb128"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 7); note_func(f); return r; });
  C("cuFuncSetAttribute NON_PORTABLE_CLUSTER_SIZE_ALLOWED -1", { init(); CUfunction f = funf("fa_lb128"); CUresult r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, -1); note_func(f); return r; });
  C("cuFuncSetAttribute on one function, the other function of the module", { init(); CUfunction f = funf("fa_dyn"); cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 70000); CUfunction g = funf("fa_dyn2"); CUfunction again = funf("fa_dyn"); note("other %d, same function again %d", dynamic_of(g), dynamic_of(again)); return CUDA_SUCCESS; });
  C("cuFuncSetAttribute in the context of device 0, the module of device 1", { cuInit(0); CUdevice d0, d1; cuDeviceGet(&d0, 0); cuDeviceGet(&d1, 1); CUcontext c0, c1; cuDevicePrimaryCtxRetain(&c0, d0); cuDevicePrimaryCtxRetain(&c1, d1);
    cuCtxSetCurrent(c0); CUfunction f0 = funf("fa_dyn"); cuCtxSetCurrent(c1); CUfunction f1 = funf("fa_dyn"); cuCtxSetCurrent(c0);
    cuFuncSetAttribute(f0, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 70000); note("device 0 %d, device 1 %d", dynamic_of(f0), dynamic_of(f1)); return CUDA_SUCCESS; });
  C("cuFuncSetAttribute, MAX_DYNAMIC_SHARED_SIZE_BYTES a CUkernel as the function", { init(); CUkernel k = kernf("fa_dyn"); CUfunction as = reinterpret_cast<CUfunction>(k); CUresult r = cuFuncSetAttribute(as, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65000); CUdevice d; cuCtxGetDevice(&d); int v = -99; cuKernelGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k, d); note("kernel now %d", v); return r; });
  C("cuFuncGetAttribute, a CUkernel as the function", { init(); CUkernel k = kernf("fa_s256"); int v = -5; CUresult r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, reinterpret_cast<CUfunction>(k)); note("v=%d", v); return r; });
  C("cuKernelGetAttribute, 256 bytes static", { init(); CUkernel k = kernf("fa_s256"); CUdevice d; cuCtxGetDevice(&d); int v[16]; for (int a = 0; a < 16; ++a) if (cuKernelGetAttribute(&v[a], static_cast<CUfunction_attribute>(a), k, d) != CUDA_SUCCESS) v[a] = -777; note("maxThreads %d, static %d, const %d, ptx %d, binary %d, maxDynamic %d, carveout %d", v[0], v[1], v[2], v[5], v[6], v[8], v[9]); return CUDA_SUCCESS; });
  C("cuKernelGetAttribute attribute 16", { init(); CUkernel k = kernf("fa_s256"); CUdevice d; cuCtxGetDevice(&d); int v = -5; return cuKernelGetAttribute(&v, (CUfunction_attribute)16, k, d); });
  C("cuKernelGetAttribute null result", { init(); CUkernel k = kernf("fa_s256"); CUdevice d; cuCtxGetDevice(&d); return cuKernelGetAttribute(nullptr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k, d); });
  C("cuKernelGetAttribute null kernel", { init(); CUdevice d; cuCtxGetDevice(&d); int v; return cuKernelGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, nullptr, d); });
  C("cuKernelGetAttribute device 5", { init(); CUkernel k = kernf("fa_s256"); int v; return cuKernelGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k, 5); });
  C("cuKernelSetAttribute device 5", { init(); CUkernel k = kernf("fa_s256"); return cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 100, k, 5); });
  C("cuKernelSetAttribute null kernel", { init(); CUdevice d; cuCtxGetDevice(&d); return cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 100, nullptr, d); });
  C("cuKernelSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES 70000, then the function", { init(); CUkernel k = kernf("fa_s256"); CUdevice d; cuCtxGetDevice(&d); CUresult r = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 70000, k, d); CUfunction f = nullptr; cuKernelGetFunction(&f, k); int v = -99; cuKernelGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k, d); note("kernel %d, function %d", v, dynamic_of(f)); return r; });
  C("cuKernelSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES 101121, 256 static", { init(); CUkernel k = kernf("fa_s256"); CUdevice d; cuCtxGetDevice(&d); CUresult r = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 101121, k, d); int v = -99; cuKernelGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k, d); note("now %d", v); return r; });
  C("cuKernelSetAttribute attribute 1", { init(); CUkernel k = kernf("fa_s256"); CUdevice d; cuCtxGetDevice(&d); return cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 1, k, d); });
  C("cuKernelSetAttribute carveout 30, then the function", { init(); CUkernel k = kernf("fa_s256"); CUdevice d; cuCtxGetDevice(&d); CUresult r = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 30, k, d); CUfunction f = nullptr; cuKernelGetFunction(&f, k); int v = -99; cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, f); note("function %d", v); return r; });
  C("cuKernelSetAttribute on device 0, read on device 1", { cuInit(0); CUdevice d0, d1; cuDeviceGet(&d0, 0); cuDeviceGet(&d1, 1); CUcontext c0, c1; cuDevicePrimaryCtxRetain(&c0, d0); cuDevicePrimaryCtxRetain(&c1, d1); cuCtxSetCurrent(c0); CUkernel k = kernf("fa_s256");
    CUresult r = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65000, k, d0); int v0 = -99, v1 = -99; cuKernelGetAttribute(&v0, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k, d0); cuKernelGetAttribute(&v1, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k, d1); note("device 0 %d, device 1 %d", v0, v1); return r; });
  // A launch may ask for what the function's limit allows, no more.
  C("cuLaunchKernel dynamic shared memory, not opted in", { init(); CUfunction f = funf("fa_dyn"); note("49152 -> %d, 49153 -> %d, 65536 -> %d", launch_code(f, 49152), launch_code(f, 49153), launch_code(f, 65536)); return CUDA_SUCCESS; });
  C("cuLaunchKernel dynamic shared memory, 40000 bytes static", { init(); CUfunction f = funf("fa_s40000"); note("9152 -> %d, 9153 -> %d", launch_code(f, 9152), launch_code(f, 9153)); return CUDA_SUCCESS; });
  C("cuLaunchKernel dynamic shared memory, 48 KiB static", { init(); CUfunction f = funf("fa_s49152"); note("0 -> %d, 1 -> %d", launch_code(f, 0), launch_code(f, 1)); return CUDA_SUCCESS; });
  C("cuLaunchKernel dynamic shared memory, opted in to 60000", { init(); CUfunction f = funf("fa_dyn"); cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 60000); note("49153 -> %d, 60000 -> %d, 60001 -> %d", launch_code(f, 49153), launch_code(f, 60000), launch_code(f, 60001)); return CUDA_SUCCESS; });
  C("cuLaunchKernel dynamic shared memory, opted in to 100", { init(); CUfunction f = funf("fa_dyn"); cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 100); note("100 -> %d, 101 -> %d, 49152 -> %d", launch_code(f, 100), launch_code(f, 101), launch_code(f, 49152)); return CUDA_SUCCESS; });
  C("cuLaunchKernel dynamic shared memory, a CUkernel opted in to 60000", { init(); CUkernel k = kernf("fa_dyn"); CUdevice d; cuCtxGetDevice(&d); cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 60000, k, d); CUfunction f = reinterpret_cast<CUfunction>(k); note("60000 -> %d, 60001 -> %d", launch_code(f, 60000), launch_code(f, 60001)); return CUDA_SUCCESS; });
  C("cuLaunchKernel with more threads than __launch_bounds__", { init(); CUfunction f = funf("fa_lb128"); CUdeviceptr d; cuMemAlloc(&d, 4096); void* args[1] = {&d}; int r1 = cuLaunchKernel(f, 1, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr), r2 = cuLaunchKernel(f, 1, 1, 1, 129, 1, 1, 0, nullptr, args, nullptr), r3 = cuLaunchKernel(f, 1, 1, 1, 64, 2, 1, 0, nullptr, args, nullptr), r4 = cuLaunchKernel(f, 1, 1, 1, 64, 3, 1, 0, nullptr, args, nullptr); note("128 -> %d, 129 -> %d, 64x2 -> %d, 64x3 -> %d", r1, r2, r3, r4); return CUDA_SUCCESS; });
  // The occupancy calls follow the limit.
  C("cuOccupancyMaxActiveBlocksPerMultiprocessor, dynamic shared memory past the limit", { init(); CUfunction f = funf("fa_dyn2"); int n[5] = {-9, -9, -9, -9, -9}; const size_t dyn[5] = {0, 49152, 49153, 65536, 101376}; for (int i = 0; i < 5; ++i) cuOccupancyMaxActiveBlocksPerMultiprocessor(&n[i], f, 128, dyn[i]); note("blocks at 0, 49152, 49153, 65536, 101376: %d %d %d %d %d", n[0], n[1], n[2], n[3], n[4]); return CUDA_SUCCESS; });
  C("cuOccupancyMaxActiveBlocksPerMultiprocessor, opted in to 60000", { init(); CUfunction f = funf("fa_dyn2"); cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 60000); int n[5] = {-9, -9, -9, -9, -9}; const size_t dyn[5] = {0, 49152, 49153, 65536, 101376}; for (int i = 0; i < 5; ++i) cuOccupancyMaxActiveBlocksPerMultiprocessor(&n[i], f, 128, dyn[i]); note("blocks at 0, 49152, 49153, 65536, 101376: %d %d %d %d %d", n[0], n[1], n[2], n[3], n[4]); return CUDA_SUCCESS; });

  // ---- CUDA 13.2

  // ---- the functions only CUDA 13.2 declares
  C("cuFuncGetParamCount", { init(); size_t n = 99; CUresult r = D132(cuFuncGetParamCount, CUresult (*)(CUfunction, size_t*), funf("fa_three"), &n); note("n=%zu", n); return r; });
  C("cuFuncGetParamCount, none", { init(); size_t n = 99; CUresult r = D132(cuFuncGetParamCount, CUresult (*)(CUfunction, size_t*), funf("fa_none"), &n); note("n=%zu", n); return r; });
  C("cuFuncGetParamCount, one", { init(); size_t n = 99; CUresult r = D132(cuFuncGetParamCount, CUresult (*)(CUfunction, size_t*), funf("fa_dyn"), &n); note("n=%zu", n); return r; });
  C("cuFuncGetParamCount null result", { init(); return D132(cuFuncGetParamCount, CUresult (*)(CUfunction, size_t*), funf("fa_three"), nullptr); });
  C("cuFuncGetParamCount null function", { init(); size_t n = 99; CUresult r = D132(cuFuncGetParamCount, CUresult (*)(CUfunction, size_t*), nullptr, &n); note("n=%zu", n); return r; });
  C("cuFuncGetParamCount a CUkernel", { init(); size_t n = 99; CUresult r = D132(cuFuncGetParamCount, CUresult (*)(CUfunction, size_t*), reinterpret_cast<CUfunction>(kernf("fa_three")), &n); note("n=%zu", n); return r; });
  C("cuKernelGetParamCount", { init(); size_t n = 99; CUresult r = D132(cuKernelGetParamCount, CUresult (*)(CUkernel, size_t*), kernf("fa_three"), &n); note("n=%zu", n); return r; });
  C("cuKernelGetParamCount, none", { init(); size_t n = 99; CUresult r = D132(cuKernelGetParamCount, CUresult (*)(CUkernel, size_t*), kernf("fa_none"), &n); note("n=%zu", n); return r; });
  C("cuKernelGetParamCount null result", { init(); return D132(cuKernelGetParamCount, CUresult (*)(CUkernel, size_t*), kernf("fa_three"), nullptr); });
  C("cuKernelGetParamCount null kernel", { init(); size_t n = 99; return D132(cuKernelGetParamCount, CUresult (*)(CUkernel, size_t*), nullptr, &n); });
  C("cuKernelGetParamCount a CUfunction", { init(); size_t n = 99; CUresult r = D132(cuKernelGetParamCount, CUresult (*)(CUkernel, size_t*), reinterpret_cast<CUkernel>(funf("fa_three")), &n); note("n=%zu", n); return r; });
  using HostV2 = CUresult (*)(CUstream, CUhostFn, void*, unsigned);
  C("cuLaunchHostFunc_v2 blocking", { init(); static int ran; ran = 0; CUstream st = strm(); CUresult r = D132(cuLaunchHostFunc_v2, HostV2, st, [](void* u) { ++*static_cast<int*>(u); }, &ran, 0); cuStreamSynchronize(st); note("ran %d", ran); return r; });
  C("cuLaunchHostFunc_v2 spin wait", { init(); static int ran; ran = 0; CUstream st = strm(); CUresult r = D132(cuLaunchHostFunc_v2, HostV2, st, [](void* u) { ++*static_cast<int*>(u); }, &ran, 1); cuStreamSynchronize(st); note("ran %d", ran); return r; });
  C("cuLaunchHostFunc_v2 mode 2", { init(); static int ran; ran = 0; CUstream st = strm(); CUresult r = D132(cuLaunchHostFunc_v2, HostV2, st, [](void* u) { ++*static_cast<int*>(u); }, &ran, 2); cuStreamSynchronize(st); note("ran %d", ran); return r; });
  C("cuLaunchHostFunc_v2 mode 77", { init(); static int ran; ran = 0; CUstream st = strm(); CUresult r = D132(cuLaunchHostFunc_v2, HostV2, st, [](void* u) { ++*static_cast<int*>(u); }, &ran, 77); cuStreamSynchronize(st); note("ran %d", ran); return r; });
  C("cuLaunchHostFunc_v2 null function", { init(); return D132(cuLaunchHostFunc_v2, HostV2, strm(), nullptr, nullptr, 0); });
  C("cuLaunchHostFunc_v2 legacy stream", { init(); static int ran; ran = 0; CUresult r = D132(cuLaunchHostFunc_v2, HostV2, nullptr, [](void* u) { ++*static_cast<int*>(u); }, &ran, 0); cuCtxSynchronize(); note("ran %d", ran); return r; });
  using CopyAttr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, MemcpyAttr*, CUstream);
  C("cuMemcpyWithAttributesAsync", { init(); CUstream st = strm(); CUdeviceptr a, b; cuMemAlloc(&a, 256); cuMemAlloc(&b, 256); char h[256], o[256]; for (int i = 0; i < 256; ++i) h[i] = static_cast<char>(i * 3); cuMemcpyHtoD(a, h, 256); MemcpyAttr at{}; at.srcAccessOrder = 1; CUresult r = D132(cuMemcpyWithAttributesAsync, CopyAttr, b, a, 256, &at, st); cuStreamSynchronize(st); cuMemcpyDtoH(o, b, 256); note("copied right %d", std::memcmp(h, o, 256) == 0); return r; });
  C("cuMemcpyWithAttributesAsync, pageable source", { init(); CUstream st = strm(); CUdeviceptr b; cuMemAlloc(&b, 256); static char h[256]; char o[256]; for (int i = 0; i < 256; ++i) h[i] = static_cast<char>(i * 5); MemcpyAttr at{}; at.srcAccessOrder = 3; CUresult r = D132(cuMemcpyWithAttributesAsync, CopyAttr, b, reinterpret_cast<CUdeviceptr>(h), 256, &at, st); cuStreamSynchronize(st); cuMemcpyDtoH(o, b, 256); note("copied right %d", std::memcmp(h, o, 256) == 0); return r; });
  C("cuMemcpyWithAttributesAsync null attributes", { init(); CUdeviceptr a; cuMemAlloc(&a, 256); return D132(cuMemcpyWithAttributesAsync, CopyAttr, a, a, 16, nullptr, strm()); });
  C("cuMemcpyWithAttributesAsync size 0", { init(); CUdeviceptr a; cuMemAlloc(&a, 256); MemcpyAttr at{}; at.srcAccessOrder = 1; return D132(cuMemcpyWithAttributesAsync, CopyAttr, a, a, 0, &at, strm()); });
  C("cuMemcpyWithAttributesAsync invalid access order", { init(); CUdeviceptr a; cuMemAlloc(&a, 256); MemcpyAttr at{}; return D132(cuMemcpyWithAttributesAsync, CopyAttr, a, a, 16, &at, strm()); });
  C("cuMemcpyWithAttributesAsync access order 9", { init(); CUdeviceptr a; cuMemAlloc(&a, 256); MemcpyAttr at{}; at.srcAccessOrder = 9; return D132(cuMemcpyWithAttributesAsync, CopyAttr, a, a, 16, &at, strm()); });
  C("cuMemcpyWithAttributesAsync flags 0x80", { init(); CUdeviceptr a; cuMemAlloc(&a, 256); MemcpyAttr at{}; at.srcAccessOrder = 1; at.flags = 0x80; return D132(cuMemcpyWithAttributesAsync, CopyAttr, a, a, 16, &at, strm()); });
  C("cuMemcpyWithAttributesAsync legacy stream", { init(); CUdeviceptr a; cuMemAlloc(&a, 256); MemcpyAttr at{}; at.srcAccessOrder = 1; return D132(cuMemcpyWithAttributesAsync, CopyAttr, a, a, 16, &at, nullptr); });
  using Copy3D = CUresult (*)(BatchOp3D*, unsigned long long, CUstream);
  C("cuMemcpy3DWithAttributesAsync", { init(); CUstream st = strm(); CUdeviceptr a, b; cuMemAlloc(&a, 4096); cuMemAlloc(&b, 4096); char h[4096], o[4096]; for (int i = 0; i < 4096; ++i) h[i] = static_cast<char>(i * 7); cuMemcpyHtoD(a, h, 4096); BatchOp3D op{}; op.src.type = 1; op.src.ptr = a; op.dst.type = 1; op.dst.ptr = b; op.width = 64; op.height = 4; op.depth = 2; op.srcAccessOrder = 1; CUresult r = D132(cuMemcpy3DWithAttributesAsync, Copy3D, &op, 0, st); cuStreamSynchronize(st); cuMemcpyDtoH(o, b, 4096); note("copied right %d", std::memcmp(h, o, 64 * 4 * 2) == 0); return r; });
  C("cuMemcpy3DWithAttributesAsync null", { init(); return D132(cuMemcpy3DWithAttributesAsync, Copy3D, nullptr, 0, strm()); });
  C("cuMemcpy3DWithAttributesAsync flags 1", { init(); BatchOp3D op{}; return D132(cuMemcpy3DWithAttributesAsync, Copy3D, &op, 1, strm()); });
  C("cuMemcpy3DWithAttributesAsync zero extent", { init(); BatchOp3D op{}; op.src.type = 1; op.dst.type = 1; return D132(cuMemcpy3DWithAttributesAsync, Copy3D, &op, 0, strm()); });
  C("cuMemcpy3DWithAttributesAsync legacy stream", { init(); CUdeviceptr a; cuMemAlloc(&a, 4096); BatchOp3D op{}; op.src.type = 1; op.src.ptr = a; op.dst.type = 1; op.dst.ptr = a + 2048; op.width = 64; op.height = 1; op.depth = 1; op.srcAccessOrder = 1; return D132(cuMemcpy3DWithAttributesAsync, Copy3D, &op, 0, nullptr); });
  C("cuGraphGetId", { init(); CUgraph g, h; cuGraphCreate(&g, 0); cuGraphCreate(&h, 0); unsigned a = 9999, b = 9999; CUresult r = D132(cuGraphGetId, GraphFn, g, &a); D132(cuGraphGetId, GraphFn, h, &b); unsigned c = 0; D132(cuGraphGetId, GraphFn, g, &c); note("second graph's id is the first's plus %d, same again %d", static_cast<int>(b) - static_cast<int>(a), c == a); return r; });
  C("cuGraphGetId null result", { init(); CUgraph g; cuGraphCreate(&g, 0); return D132(cuGraphGetId, GraphFn, g, nullptr); });
  C("cuGraphGetId null graph", { init(); unsigned a = 9999; return D132(cuGraphGetId, GraphFn, nullptr, &a); });
  using ExecIdFn = CUresult (*)(CUgraphExec, unsigned*);
  C("cuGraphExecGetId", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n; cuGraphAddEmptyNode(&n, g, nullptr, 0); CUgraphExec e1, e2; cuGraphInstantiateWithFlags(&e1, g, 0); cuGraphInstantiateWithFlags(&e2, g, 0); unsigned a = 9999, b = 9999, gi = 9999; CUresult r = D132(cuGraphExecGetId, ExecIdFn, e1, &a); D132(cuGraphExecGetId, ExecIdFn, e2, &b); D132(cuGraphGetId, GraphFn, g, &gi); note("two execs differ %d, exec id is graph id %d, exec ids run on by %d", a != b, a == gi, static_cast<int>(b) - static_cast<int>(a)); return r; });
  C("cuGraphExecGetId null result", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphExec e; cuGraphInstantiateWithFlags(&e, g, 0); return D132(cuGraphExecGetId, ExecIdFn, e, nullptr); });
  C("cuGraphExecGetId null exec", { init(); unsigned a = 9999; return D132(cuGraphExecGetId, ExecIdFn, nullptr, &a); });
  using NodeIdFn = CUresult (*)(CUgraphNode, unsigned*);
  C("cuGraphNodeGetLocalId", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n1, n2, n3; cuGraphAddEmptyNode(&n1, g, nullptr, 0); cuGraphAddEmptyNode(&n2, g, &n1, 1); cuGraphAddEmptyNode(&n3, g, &n2, 1); unsigned a = 9999, b = 9999, c = 9999; CUresult r = D132(cuGraphNodeGetLocalId, NodeIdFn, n1, &a); D132(cuGraphNodeGetLocalId, NodeIdFn, n2, &b); D132(cuGraphNodeGetLocalId, NodeIdFn, n3, &c); note("ids %u %u %u", a, b, c); return r; });
  C("cuGraphNodeGetLocalId, nodes of two graphs", { init(); CUgraph g, h; cuGraphCreate(&g, 0); cuGraphCreate(&h, 0); CUgraphNode a1, a2, b1; cuGraphAddEmptyNode(&a1, g, nullptr, 0); cuGraphAddEmptyNode(&a2, g, nullptr, 0); cuGraphAddEmptyNode(&b1, h, nullptr, 0); unsigned x = 9999, y = 9999, z = 9999; CUresult r = D132(cuGraphNodeGetLocalId, NodeIdFn, a1, &x); D132(cuGraphNodeGetLocalId, NodeIdFn, a2, &y); D132(cuGraphNodeGetLocalId, NodeIdFn, b1, &z); note("ids %u %u %u", x, y, z); return r; });
  C("cuGraphNodeGetLocalId null result", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n; cuGraphAddEmptyNode(&n, g, nullptr, 0); return D132(cuGraphNodeGetLocalId, NodeIdFn, n, nullptr); });
  C("cuGraphNodeGetLocalId null node", { init(); unsigned a = 9999; return D132(cuGraphNodeGetLocalId, NodeIdFn, nullptr, &a); });
  using ToolsIdFn = CUresult (*)(CUgraphNode, unsigned long long*);
  C("cuGraphNodeGetToolsId", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n1, n2; cuGraphAddEmptyNode(&n1, g, nullptr, 0); cuGraphAddEmptyNode(&n2, g, nullptr, 0); unsigned long long a = 9999, b = 9999; unsigned gid = 0; CUresult r = D132(cuGraphNodeGetToolsId, ToolsIdFn, n1, &a); D132(cuGraphNodeGetToolsId, ToolsIdFn, n2, &b); D132(cuGraphGetId, GraphFn, g, &gid); note("differ %d, graph id in the high half %d, node index in the low half %llu %llu", a != b, (a >> 32) == gid, a & 0xffffffffull, b & 0xffffffffull); return r; });
  C("cuGraphNodeGetToolsId null result", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n; cuGraphAddEmptyNode(&n, g, nullptr, 0); return D132(cuGraphNodeGetToolsId, ToolsIdFn, n, nullptr); });
  C("cuGraphNodeGetToolsId null node", { init(); unsigned long long a = 9999; return D132(cuGraphNodeGetToolsId, ToolsIdFn, nullptr, &a); });
  using ContainFn = CUresult (*)(CUgraphNode, CUgraph*);
  C("cuGraphNodeGetContainingGraph", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n; cuGraphAddEmptyNode(&n, g, nullptr, 0); CUgraph got = nullptr; CUresult r = D132(cuGraphNodeGetContainingGraph, ContainFn, n, &got); note("is the graph %d", got == g); return r; });
  C("cuGraphNodeGetContainingGraph, node of a child graph", { init(); CUgraph g, c; cuGraphCreate(&g, 0); cuGraphCreate(&c, 0); CUgraphNode in, ch; cuGraphAddEmptyNode(&in, c, nullptr, 0); cuGraphAddChildGraphNode(&ch, g, nullptr, 0, c); CUgraph got = nullptr, got2 = nullptr; CUresult r = D132(cuGraphNodeGetContainingGraph, ContainFn, in, &got); D132(cuGraphNodeGetContainingGraph, ContainFn, ch, &got2); note("child's node in the child %d, in the parent %d; the child graph node in the parent %d", got == c, got == g, got2 == g); return r; });
  C("cuGraphNodeGetContainingGraph null result", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n; cuGraphAddEmptyNode(&n, g, nullptr, 0); return D132(cuGraphNodeGetContainingGraph, ContainFn, n, nullptr); });
  C("cuGraphNodeGetContainingGraph null node", { init(); CUgraph got; return D132(cuGraphNodeGetContainingGraph, ContainFn, nullptr, &got); });
  using ParamsFn = CUresult (*)(CUgraphNode, NodeParams*);
  C("cuGraphNodeGetParams, a memset node", { init(); CUgraph g; cuGraphCreate(&g, 0); CUdeviceptr d; cuMemAlloc(&d, 1024); CUDA_MEMSET_NODE_PARAMS mp{}; mp.dst = d; mp.value = 7; mp.elementSize = 1; mp.width = 256; mp.height = 1; CUcontext c; cuCtxGetCurrent(&c); CUgraphNode n; cuGraphAddMemsetNode(&n, g, nullptr, 0, &mp, c); NodeParams p; std::memset(&p, 0, sizeof p); CUresult r = D132(cuGraphNodeGetParams, ParamsFn, n, &p); note("type %d, dst right %d, value %u, width %zu, element %u", p.type, p.get<unsigned long long>(0) == d, p.get<unsigned>(16), p.get<size_t>(24), p.get<unsigned>(20)); return r; });
  C("cuGraphNodeGetParams, a kernel node", { init(); CUgraph g; cuGraphCreate(&g, 0); CUfunction f = fun(); CUdeviceptr d; cuMemAlloc(&d, 1024); void* args[1] = {&d}; CUDA_KERNEL_NODE_PARAMS kp{}; kp.func = f; kp.gridDimX = 2; kp.gridDimY = 1; kp.gridDimZ = 1; kp.blockDimX = 32; kp.blockDimY = 1; kp.blockDimZ = 1; kp.kernelParams = args; CUgraphNode n; cuGraphAddKernelNode(&n, g, nullptr, 0, &kp); NodeParams p; std::memset(&p, 0, sizeof p); CUresult r = D132(cuGraphNodeGetParams, ParamsFn, n, &p); note("type %d, grid %u, block %u, shared %u, function stored %d", p.type, p.get<unsigned>(8), p.get<unsigned>(20), p.get<unsigned>(32), p.get<void*>(0) != nullptr || p.get<void*>(56) != nullptr); return r; });
  C("cuGraphNodeGetParams, an empty node", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n; cuGraphAddEmptyNode(&n, g, nullptr, 0); NodeParams p; std::memset(&p, 0, sizeof p); CUresult r = D132(cuGraphNodeGetParams, ParamsFn, n, &p); note("type %d", p.type); return r; });
  C("cuGraphNodeGetParams, a child graph node", { init(); CUgraph g, c; cuGraphCreate(&g, 0); cuGraphCreate(&c, 0); CUgraphNode in, ch; cuGraphAddEmptyNode(&in, c, nullptr, 0); cuGraphAddChildGraphNode(&ch, g, nullptr, 0, c); NodeParams p; std::memset(&p, 0, sizeof p); CUresult r = D132(cuGraphNodeGetParams, ParamsFn, ch, &p); CUgraph got = p.get<CUgraph>(0); note("type %d, graph is a clone %d", p.type, got != nullptr && got != c); return r; });
  C("cuGraphNodeGetParams, a host node", { init(); CUgraph g; cuGraphCreate(&g, 0); static int x; CUDA_HOST_NODE_PARAMS hp{}; hp.fn = [](void*) {}; hp.userData = &x; CUgraphNode n; cuGraphAddHostNode(&n, g, nullptr, 0, &hp); NodeParams p; std::memset(&p, 0, sizeof p); CUresult r = D132(cuGraphNodeGetParams, ParamsFn, n, &p); note("type %d, user data right %d", p.type, p.get<void*>(8) == &x); return r; });
  C("cuGraphNodeGetParams null node", { init(); NodeParams p; std::memset(&p, 0, sizeof p); return D132(cuGraphNodeGetParams, ParamsFn, nullptr, &p); });
  C("cuGraphNodeGetParams null result", { init(); CUgraph g; cuGraphCreate(&g, 0); CUgraphNode n; cuGraphAddEmptyNode(&n, g, nullptr, 0); return D132(cuGraphNodeGetParams, ParamsFn, n, nullptr); });
  using CoreReg = CUresult (*)(void (*)(void*, int, CUdevice), void*, void**);
  using CoreDereg = CUresult (*)(void*);
  C("cuCoredumpRegisterStartCallback", { init(); void* h = nullptr; CUresult r = D132(cuCoredumpRegisterStartCallback, CoreReg, [](void*, int, CUdevice) {}, nullptr, &h); note("handle set %d", h != nullptr); return r; });
  C("cuCoredumpRegisterCompleteCallback", { init(); void* h = nullptr; CUresult r = D132(cuCoredumpRegisterCompleteCallback, CoreReg, [](void*, int, CUdevice) {}, nullptr, &h); note("handle set %d", h != nullptr); return r; });
  C("cuCoredumpRegisterStartCallback null callback", { init(); void* h = nullptr; CUresult r = D132(cuCoredumpRegisterStartCallback, CoreReg, nullptr, nullptr, &h); note("handle set %d", h != nullptr); return r; });
  C("cuCoredumpRegisterStartCallback null handle", { init(); return D132(cuCoredumpRegisterStartCallback, CoreReg, [](void*, int, CUdevice) {}, nullptr, nullptr); });
  C("cuCoredumpRegisterCompleteCallback null callback", { init(); void* h = nullptr; return D132(cuCoredumpRegisterCompleteCallback, CoreReg, nullptr, nullptr, &h); });
  C("cuCoredumpRegisterCompleteCallback null handle", { init(); return D132(cuCoredumpRegisterCompleteCallback, CoreReg, [](void*, int, CUdevice) {}, nullptr, nullptr); });
  C("cuCoredumpDeregisterStartCallback", { init(); void* h = nullptr; D132(cuCoredumpRegisterStartCallback, CoreReg, [](void*, int, CUdevice) {}, nullptr, &h); return D132(cuCoredumpDeregisterStartCallback, CoreDereg, h); });
  C("cuCoredumpDeregisterCompleteCallback", { init(); void* h = nullptr; D132(cuCoredumpRegisterCompleteCallback, CoreReg, [](void*, int, CUdevice) {}, nullptr, &h); return D132(cuCoredumpDeregisterCompleteCallback, CoreDereg, h); });
  C("cuCoredumpDeregisterStartCallback null", { init(); return D132(cuCoredumpDeregisterStartCallback, CoreDereg, nullptr); });
  C("cuCoredumpDeregisterCompleteCallback null", { init(); return D132(cuCoredumpDeregisterCompleteCallback, CoreDereg, nullptr); });
  C("cuMulticastBindMem_v2", { init(); return D132(cuMulticastBindMem_v2, CUresult (*)(unsigned long long, CUdevice, size_t, unsigned long long, size_t, size_t, unsigned long long), 0, 0, 0, 0, 0, 1 << 21, 0); });
  C("cuMulticastBindAddr_v2", { init(); return D132(cuMulticastBindAddr_v2, CUresult (*)(unsigned long long, CUdevice, size_t, CUdeviceptr, size_t, unsigned long long), 0, 0, 0, 0, 1 << 21, 0); });
  C("cuStreamBeginCaptureToCig null parameters", { init(); return D132(cuStreamBeginCaptureToCig, CUresult (*)(CUstream, void*), strm(), nullptr); });
  C("cuStreamBeginCaptureToCig", { init(); void* cp[2] = {nullptr, nullptr}; void* params = cp; return D132(cuStreamBeginCaptureToCig, CUresult (*)(CUstream, void*), strm(), &params); });
  C("cuStreamEndCaptureToCig", { init(); return D132(cuStreamEndCaptureToCig, CUresult (*)(CUstream), strm()); });

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
