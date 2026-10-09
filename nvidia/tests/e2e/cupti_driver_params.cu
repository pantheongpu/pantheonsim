// The parameter structures the callback API hands a subscriber for driver
// calls: a program that uses only the driver API makes a few dozen calls of
// every family it has (devices, contexts, memory of every kind, pools, virtual
// memory, modules, launches, streams, events, arrays, linking) and a subscriber
// prints each call's entry with its parameters -- every field by name, a number
// as itself and a pointer as set or null -- and its exit with the value it
// returned. This is cupti_params.cu for the driver domain.
//
// Run against NVIDIA's libcupti on a card it prints what the toolkit's own
// structures hold for each call (nvidia/tests/data/cupti_driver_params.expected);
// run against this project's it must print the same, which checks that the
// argument of each position reaches the field of that position, for every
// function a table made from the toolkit's header converts (see
// scripts/gen_cupti_driver_conv.py) and that the shim answers each call as the
// driver does. The functions are those whose structure is the same in CUDA 12.0,
// 12.8 and 13.x (gen_cupti_driver_params.py makes cupti_driver_params.inc).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cudaProfiler.h>
#include <cupti.h>

// The toolkit's header spells some calls with the newest version of each, which
// differs between toolkits and would make the trace depend on which one compiled
// this. Each is called by the spelling every driver has exported for years.
extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext*, unsigned int, CUdevice);
#ifdef cuEventElapsedTime
#undef cuEventElapsedTime
#endif
extern "C" CUresult CUDAAPI cuEventElapsedTime(float*, CUevent, CUevent);

namespace {

std::vector<std::string> g_lines;

template <class T>
std::string field(const char* name, const T& v) {
  char buf[96];
  if constexpr (std::is_pointer_v<T>) {
    std::snprintf(buf, sizeof buf, " %s=%s", name, v ? "set" : "null");
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    // CUdeviceptr, allocation handles and the driver's other 64-bit numbers are
    // addresses and tokens that differ from one driver to the next; sizes and
    // flags of this type are small.
    if (std::strcmp(name, "handle") == 0) std::snprintf(buf, sizeof buf, " %s=%s", name, v ? "set" : "null");
    else if (v < (1ull << 31)) std::snprintf(buf, sizeof buf, " %s=%llu", name, v);
    else std::snprintf(buf, sizeof buf, " %s=big", name);
  } else if constexpr (std::is_enum_v<T> || std::is_integral_v<T>) {
    std::snprintf(buf, sizeof buf, " %s=%lld", name, static_cast<long long>(v));
  } else if constexpr (std::is_floating_point_v<T>) {
    std::snprintf(buf, sizeof buf, " %s=%g", name, static_cast<double>(v));
  } else {
    std::snprintf(buf, sizeof buf, " %s=<%zu bytes>", name, sizeof(T));
  }
  return buf;
}

std::string params_of(CUpti_CallbackId id, const void* d) {
  switch (id) {
#include "cupti_driver_params.inc"
    default: return "";
  }
}

bool g_on = false;

void CUPTIAPI on_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API || !g_on) return;
  const auto* cb = static_cast<const CUpti_CallbackData*>(data);
  const bool enter = cb->callbackSite == CUPTI_API_ENTER;
  std::string line = std::string("CB ") + cb->functionName + (enter ? " ENTER" : " EXIT");
  if (enter) {
    line += params_of(cbid, cb->functionParams);
  } else if (cb->functionReturnValue) {
    line += " ret=" + std::to_string(static_cast<int>(*static_cast<const CUresult*>(cb->functionReturnValue)));
  }
  if (std::getenv("PARAMS_LIVE")) std::fprintf(stderr, "%s\n", line.c_str());
  g_lines.push_back(line);
}

// A kernel and nothing else: it scales n floats in place.
const char kPtx[] = R"PTX(
.version 7.0
.target sm_75
.address_size 64

.visible .global .align 4 .u32 counter;

.visible .entry scale(.param .u64 p, .param .f32 a, .param .u32 n)
{
  .reg .pred %q;
  .reg .b32 %r<6>;
  .reg .b64 %rd<5>;
  .reg .f32 %f<4>;
  ld.param.u64 %rd1, [p];
  ld.param.f32 %f1, [a];
  ld.param.u32 %r1, [n];
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r5, %r2, %r3, %r4;
  setp.ge.s32 %q, %r5, %r1;
  @%q bra DONE;
  cvta.to.global.u64 %rd2, %rd1;
  mul.wide.s32 %rd3, %r5, 4;
  add.s64 %rd4, %rd2, %rd3;
  ld.global.f32 %f2, [%rd4];
  mul.f32 %f3, %f2, %f1;
  st.global.f32 [%rd4], %f3;
DONE:
  ret;
}
)PTX";

void CUDA_CB host_fn(void*) {}
void CUDA_CB stream_cb(CUstream, CUresult, void*) {}

}  // namespace

// A failed call is not an error here: the callbacks carry each result, and the
// calls that are refused are part of what is compared. Only the ones the rest
// of the program needs are checked.
#define CALL(x) ((void)(x))
#define NEED(x)                                                                        \
  do {                                                                                 \
    const CUresult r_ = (x);                                                           \
    if (r_ != CUDA_SUCCESS) {                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #x, (int)r_); \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

int main() {
  CUpti_SubscriberHandle sub;
  if (cuptiSubscribe(&sub, on_callback, nullptr) != CUPTI_SUCCESS) return 1;
  cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_DRIVER_API);

  // The first cuInit is how the profiler is attached: not reported. Everything
  // after is.
  NEED(cuInit(0));
  g_on = true;

  // ---- devices ----
  int count = 0, version = 0, value = 0;
  CUdevice dev = 0;
  NEED(cuInit(0));
  CALL(cuDriverGetVersion(&version));
  NEED(cuDeviceGetCount(&count));
  NEED(cuDeviceGet(&dev, 0));
  char name[64];
  CALL(cuDeviceGetName(name, sizeof name, dev));
  size_t total = 0;
  CALL(cuDeviceTotalMem(&total, dev));
  CALL(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev));
  CUuuid uuid;
  CALL(cuDeviceGetUuid_v2(&uuid, dev));
  char bus[32];
  CALL(cuDeviceGetPCIBusId(bus, sizeof bus, dev));
  CUdevice by_bus = -1;
  CALL(cuDeviceGetByPCIBusId(&by_bus, bus));
  CALL(cuDeviceCanAccessPeer(&value, dev, dev));
  CUmemoryPool default_pool = nullptr;
  CALL(cuDeviceGetDefaultMemPool(&default_pool, dev));
  CUmemoryPool current_pool = nullptr;
  CALL(cuDeviceGetMemPool(&current_pool, dev));
  CALL(cuDeviceSetMemPool(dev, default_pool));
  CALL(cuDeviceGetExecAffinitySupport(&value, CU_EXEC_AFFINITY_TYPE_SM_COUNT, dev));

  // ---- primary context ----
  CUcontext primary = nullptr;
  unsigned int pflags = 0;
  int active = 0;
  CALL(cuDevicePrimaryCtxGetState(dev, &pflags, &active));
  NEED(cuDevicePrimaryCtxRetain(&primary, dev));
  CALL(cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_AUTO));
  CALL(cuDevicePrimaryCtxGetState(dev, &pflags, &active));
  CALL(cuDevicePrimaryCtxRelease(dev));

  // ---- contexts ----
  CUcontext ctx = nullptr;
  NEED(cuCtxCreate_v2(&ctx, 0, dev));
  CUcontext popped = nullptr;
  CALL(cuCtxPushCurrent(ctx));
  CALL(cuCtxPopCurrent(&popped));
  NEED(cuCtxSetCurrent(ctx));
  CUcontext current = nullptr;
  CALL(cuCtxGetCurrent(&current));
  CUdevice cdev = -1;
  CALL(cuCtxGetDevice(&cdev));
  unsigned int cflags = 0;
  CALL(cuCtxGetFlags(&cflags));
  unsigned long long cid = 0;
  CALL(cuCtxGetId(ctx, &cid));
  unsigned int api = 0;
  CALL(cuCtxGetApiVersion(ctx, &api));
  size_t limit = 0;
  CALL(cuCtxGetLimit(&limit, CU_LIMIT_STACK_SIZE));
  CALL(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, limit));
  CUfunc_cache cache;
  CALL(cuCtxGetCacheConfig(&cache));
  CALL(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_NONE));
  int least = 0, greatest = 0;
  CALL(cuCtxGetStreamPriorityRange(&least, &greatest));
  CALL(cuCtxSynchronize());

  // ---- memory ----
  size_t free_b = 0, total_b = 0;
  CALL(cuMemGetInfo(&free_b, &total_b));
  CUdeviceptr d = 0, d2 = 0, pitched = 0, managed = 0;
  size_t pitch = 0;
  NEED(cuMemAlloc(&d, 4096));
  NEED(cuMemAlloc(&d2, 4096));
  CALL(cuMemAllocPitch(&pitched, &pitch, 100, 8, 4));
  CALL(cuMemAllocManaged(&managed, 4096, CU_MEM_ATTACH_GLOBAL));
  void* pinned = nullptr;
  void* host_alloc = nullptr;
  CALL(cuMemAllocHost(&pinned, 4096));
  CALL(cuMemHostAlloc(&host_alloc, 4096, CU_MEMHOSTALLOC_DEVICEMAP));
  CUdeviceptr dev_view = 0;
  CALL(cuMemHostGetDevicePointer(&dev_view, host_alloc, 0));
  unsigned int hflags = 0;
  CALL(cuMemHostGetFlags(&hflags, host_alloc));
  CUdeviceptr base = 0;
  size_t extent = 0;
  CALL(cuMemGetAddressRange(&base, &extent, d + 16));
  static float registered[1024];
  CALL(cuMemHostRegister(registered, sizeof registered, 0));
  CALL(cuMemHostUnregister(registered));
  float host[64] = {};
  NEED(cuMemsetD8(d, 1, 256));
  CALL(cuMemsetD16(d, 2, 128));
  CALL(cuMemsetD32(d, 3, 64));
  CALL(cuMemsetD2D8(pitched, pitch, 4, 100, 8));
  CALL(cuMemsetD2D16(pitched, pitch, 5, 50, 8));
  CALL(cuMemsetD2D32(pitched, pitch, 6, 25, 8));
  NEED(cuMemcpyHtoD(d, host, sizeof host));
  NEED(cuMemcpyDtoH(host, d, sizeof host));
  NEED(cuMemcpyDtoD(d2, d, sizeof host));
  CALL(cuMemcpy(d2, d, sizeof host));
  CUDA_MEMCPY2D c2;
  std::memset(&c2, 0, sizeof c2);
  c2.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  c2.srcDevice = d;
  c2.srcPitch = 256;
  c2.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  c2.dstDevice = d2;
  c2.dstPitch = 256;
  c2.WidthInBytes = 128;
  c2.Height = 4;
  CALL(cuMemcpy2D(&c2));
  CALL(cuMemcpy2DUnaligned(&c2));
  CALL(cuMemcpyPeer(d2, ctx, d, ctx, 256));

  // ---- pointers ----
  CUmemorytype mtype;
  CALL(cuPointerGetAttribute(&mtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, d));
  CUpointer_attribute attrs[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};
  int ordinal = -1;
  void* results[2] = {&mtype, &ordinal};
  CALL(cuPointerGetAttributes(2, attrs, results, d));
  int sync = 1;
  CALL(cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d));

  // ---- streams and events ----
  CUstream s1 = nullptr, s2 = nullptr;
  NEED(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING));
  NEED(cuStreamCreateWithPriority(&s2, CU_STREAM_DEFAULT, 0));
  unsigned int sflags = 0;
  int sprio = 0;
  CALL(cuStreamGetFlags(s1, &sflags));
  CALL(cuStreamGetPriority(s2, &sprio));
  CUcontext sctx = nullptr;
  CALL(cuStreamGetCtx(s1, &sctx));
  CALL(cuStreamQuery(s1));
  CUstreamCaptureStatus capturing;
  CALL(cuStreamIsCapturing(s1, &capturing));
  CUevent e1 = nullptr, e2 = nullptr;
  NEED(cuEventCreate(&e1, CU_EVENT_DEFAULT));
  NEED(cuEventCreate(&e2, CU_EVENT_DEFAULT));
  NEED(cuEventRecord(e1, s1));
  NEED(cuEventSynchronize(e1));
  CALL(cuStreamWaitEvent(s2, e1, 0));
  NEED(cuEventRecord(e2, s1));
  CALL(cuStreamSynchronize(s1));
  float ms = 0;
  CALL(cuEventElapsedTime(&ms, e1, e2));
  CALL(cuStreamAddCallback(s1, stream_cb, nullptr, 0));
  CALL(cuLaunchHostFunc(s1, host_fn, nullptr));
  CALL(cuStreamAttachMemAsync(s1, managed, 0, CU_MEM_ATTACH_GLOBAL));
  CALL(cuMemsetD8Async(d, 7, 256, s1));
  CALL(cuMemsetD16Async(d, 8, 128, s1));
  CALL(cuMemsetD32Async(d, 9, 64, s1));
  CALL(cuMemsetD2D8Async(pitched, pitch, 4, 100, 8, s1));
  CALL(cuMemsetD2D16Async(pitched, pitch, 5, 50, 8, s1));
  CALL(cuMemsetD2D32Async(pitched, pitch, 6, 25, 8, s1));
  CALL(cuMemcpyHtoDAsync(d, host, sizeof host, s1));
  CALL(cuMemcpyDtoHAsync(host, d, sizeof host, s1));
  CALL(cuMemcpyDtoDAsync(d2, d, sizeof host, s1));
  CALL(cuMemcpyAsync(d2, d, sizeof host, s1));
  CALL(cuMemcpy2DAsync(&c2, s1));
  CALL(cuMemcpyPeerAsync(d2, ctx, d, ctx, 256, s1));
  CALL(cuStreamSynchronize(s1));

  // ---- pools ----
  CUmemoryPool pool = nullptr;
  CUmemPoolProps props;
  std::memset(&props, 0, sizeof props);
  props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = 0;
  CALL(cuMemPoolCreate(&pool, &props));
  cuuint64_t threshold = 1 << 20;
  CALL(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold));
  CALL(cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold));
  CUmemAccessDesc access;
  std::memset(&access, 0, sizeof access);
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CALL(cuMemPoolSetAccess(pool, &access, 1));
  CUmemAccess_flags got_flags;
  CALL(cuMemPoolGetAccess(&got_flags, pool, &access.location));
  CUdeviceptr pooled = 0;
  CALL(cuMemAllocFromPoolAsync(&pooled, 1 << 16, pool, s1));
  {
    int fd = -1;
    CUmemoryPool imported = nullptr;
    CUmemPoolPtrExportData exported;
    std::memset(&exported, 0, sizeof exported);
    CUdeviceptr imported_ptr = 0;
    CALL(cuMemPoolExportToShareableHandle(&fd, pool, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    CALL(cuMemPoolImportFromShareableHandle(&imported, &fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    CALL(cuMemPoolExportPointer(&exported, pooled));
    CALL(cuMemPoolImportPointer(&imported_ptr, pool, &exported));
  }
  CALL(cuMemFreeAsync(pooled, s1));
  CALL(cuStreamSynchronize(s1));
  CALL(cuMemPoolTrimTo(pool, 0));
  CALL(cuMemPoolDestroy(pool));
  CUdeviceptr async_ptr = 0;
  CALL(cuMemAllocAsync(&async_ptr, 1 << 12, s1));
  CALL(cuMemFreeAsync(async_ptr, s1));
  CALL(cuStreamSynchronize(s1));

  // ---- virtual memory ----
  {
    CUmemAllocationProp ap;
    std::memset(&ap, 0, sizeof ap);
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = 0;
    size_t reported = 0;
    CALL(cuMemGetAllocationGranularity(&reported, &ap, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // The card's minimum is 2 MiB (this simulator reports a smaller unit); a
    // size that is a multiple of both is what the calls are made with.
    const size_t gran = 2u << 20;
    CUdeviceptr va = 0;
    CUmemGenericAllocationHandle h = 0;
    CALL(cuMemAddressReserve(&va, gran, 0, 0, 0));
    CALL(cuMemCreate(&h, gran, &ap, 0));
    CALL(cuMemMap(va, gran, 0, h, 0));
    CALL(cuMemSetAccess(va, gran, &access, 1));
    unsigned long long flags_out = 0;
    CALL(cuMemGetAccess(&flags_out, &access.location, va));
    CUmemAllocationProp back;
    CALL(cuMemGetAllocationPropertiesFromHandle(&back, h));
    CUmemGenericAllocationHandle retained = 0;
    CALL(cuMemRetainAllocationHandle(&retained, reinterpret_cast<void*>(va)));
    CALL(cuMemUnmap(va, gran));
    CALL(cuMemRelease(h));
    CALL(cuMemAddressFree(va, gran));
  }

  // ---- modules and launches ----
  CUmodule mod = nullptr;
  CUfunction fn = nullptr;
  NEED(cuModuleLoadData(&mod, kPtx));
  NEED(cuModuleGetFunction(&fn, mod, "scale"));
  CUdeviceptr gdev = 0;
  size_t gsize = 0;
  CALL(cuModuleGetGlobal(&gdev, &gsize, mod, "counter"));
  CUmoduleLoadingMode mode;
  CALL(cuModuleGetLoadingMode(&mode));
  int fattr = 0;
  CALL(cuFuncGetAttribute(&fattr, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
  CALL(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 1024));
  CALL(cuFuncSetCacheConfig(fn, CU_FUNC_CACHE_PREFER_NONE));
  int blocks = 0;
  CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, 128, 0));
  CALL(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&blocks, fn, 128, 0, CU_OCCUPANCY_DEFAULT));
  int min_grid = 0, block_size = 0;
  CALL(cuOccupancyMaxPotentialBlockSize(&min_grid, &block_size, fn, nullptr, 0, 0));
  CALL(cuOccupancyMaxPotentialBlockSizeWithFlags(&min_grid, &block_size, fn, nullptr, 0, 0, CU_OCCUPANCY_DEFAULT));
  float scale = 2.f;
  int n = 32;
  void* args[] = {&d, &scale, &n};
  CALL(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, s1, args, nullptr));
  CALL(cuLaunchCooperativeKernel(fn, 1, 1, 1, 32, 1, 1, 0, s1, args));
  // The pre-CUDA 4 launch: a block shape, a parameter buffer and a grid.
  CALL(cuFuncSetBlockShape(fn, 32, 1, 1));
  CALL(cuFuncSetSharedSize(fn, 0));
  CALL(cuParamSetSize(fn, 16));
  CALL(cuParamSetv(fn, 0, &d, sizeof d));
  CALL(cuParamSetf(fn, 8, scale));
  CALL(cuParamSeti(fn, 12, n));
  CALL(cuLaunchGridAsync(fn, 1, 1, s1));
  CALL(cuLaunchGrid(fn, 1, 1));
  CALL(cuLaunch(fn));
  CALL(cuStreamSynchronize(s1));
  CALL(cuCtxSynchronize());

  // ---- linking ----
  {
    CUlinkState link = nullptr;
    CALL(cuLinkCreate(0, nullptr, nullptr, &link));
    CALL(cuLinkAddData(link, CU_JIT_INPUT_PTX, const_cast<char*>(kPtx), sizeof kPtx, "scale.ptx", 0, nullptr,
                       nullptr));
    void* cubin = nullptr;
    size_t cubin_size = 0;
    CALL(cuLinkComplete(link, &cubin, &cubin_size));
    CALL(cuLinkDestroy(link));
  }

  // ---- arrays ----
  CUarray arr = nullptr;
  {
    CUDA_ARRAY_DESCRIPTOR ad;
    std::memset(&ad, 0, sizeof ad);
    ad.Width = 16;
    ad.Height = 16;
    ad.Format = CU_AD_FORMAT_FLOAT;
    ad.NumChannels = 1;
    CALL(cuArrayCreate(&arr, &ad));
    CUDA_ARRAY_DESCRIPTOR back;
    CALL(cuArrayGetDescriptor(&back, arr));
    CUDA_MEMCPY2D to_array;
    std::memset(&to_array, 0, sizeof to_array);
    to_array.srcMemoryType = CU_MEMORYTYPE_HOST;
    to_array.srcHost = host;
    to_array.srcPitch = 64;
    to_array.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    to_array.dstArray = arr;
    to_array.WidthInBytes = 64;
    to_array.Height = 1;
    CALL(cuMemcpy2D(&to_array));
    CALL(cuMemcpyHtoA(arr, 0, host, 64));
    CALL(cuMemcpyAtoH(host, arr, 0, 64));
    CALL(cuArrayDestroy(arr));
  }
  {
    CUDA_ARRAY3D_DESCRIPTOR ad;
    std::memset(&ad, 0, sizeof ad);
    ad.Width = 8;
    ad.Height = 8;
    ad.Depth = 8;
    ad.Format = CU_AD_FORMAT_FLOAT;
    ad.NumChannels = 1;
    CUarray arr3 = nullptr;
    CALL(cuArray3DCreate(&arr3, &ad));
    CUDA_ARRAY3D_DESCRIPTOR back;
    CALL(cuArray3DGetDescriptor(&back, arr3));
    CALL(cuArrayDestroy(arr3));
  }

  // ---- errors and entry points ----
  const char* text = nullptr;
  CALL(cuGetErrorString(CUDA_ERROR_INVALID_VALUE, &text));
  CALL(cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &text));
  void* proc = nullptr;
  CUdriverProcAddressQueryResult found;
  CALL(cuGetProcAddress("cuMemAlloc_v2", &proc, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &found));
  CALL(cuProfilerStart());
  CALL(cuProfilerStop());

  // ---- teardown ----
  CALL(cuModuleUnload(mod));
  CALL(cuEventDestroy(e1));
  CALL(cuEventDestroy(e2));
  CALL(cuStreamDestroy(s1));
  CALL(cuStreamDestroy(s2));
  CALL(cuMemFree(d));
  CALL(cuMemFree(d2));
  CALL(cuMemFree(pitched));
  CALL(cuMemFree(managed));
  CALL(cuMemFreeHost(pinned));
  CALL(cuMemFreeHost(host_alloc));
  CALL(cuCtxSetCurrent(nullptr));
  CALL(cuCtxDestroy(ctx));
  CALL(cuDevicePrimaryCtxReset(dev));

  g_on = false;
  cuptiUnsubscribe(sub);
  for (const std::string& l : g_lines) std::printf("%s\n", l.c_str());
  std::printf("# end\n");
  return 0;
}
