// The arguments the profiler prints for each kind of HIP parameter: null and
// non-null pointers, handles, strings, enums, integers of every width, structs
// passed by value, function pointers. Calls are made to be refused or harmless;
// what they return is not compared.
#include "common.h"

static void host_fn(void*) {}
static void stream_cb(hipStream_t, hipError_t, void*) {}

int main() {
  int wrong = 0;
  hipStream_t s = nullptr, s2 = nullptr;
  hipEvent_t e = nullptr, e2 = nullptr;
  char name[64] = "";
  void* p = nullptr;
  size_t pitch = 0, freeb = 0, total = 0;
  int iv = 0, lo = 0, hi = 0;
  unsigned int uf = 0;
  float ms = 0;

  // pointers: null and not, data and handles
  (void)hipMemcpy(nullptr, nullptr, 0, hipMemcpyHostToHost);
  (void)hipMemcpy(nullptr, nullptr, 0, hipMemcpyDefault);
  (void)hipMemcpy(nullptr, nullptr, 0, static_cast<hipMemcpyKind>(1024));
  (void)hipMemcpy(nullptr, nullptr, 0, static_cast<hipMemcpyKind>(77));
  (void)hipMalloc(&p, 256);
  (void)hipMemset(p, 7, 256);
  (void)hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(p), 9, 64);
  (void)hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(p), 9, 32);
  (void)hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(p), 9, 16);
  (void)hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(p), name, 4);
  (void)hipMemcpyDtoH(name, reinterpret_cast<hipDeviceptr_t>(p), 4);
  (void)hipFree(p);
  (void)hipFree(nullptr);
  (void)hipStreamSynchronize(nullptr);
  (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  (void)hipStreamCreateWithPriority(&s2, hipStreamDefault, 0);
  (void)hipDeviceGetStreamPriorityRange(&lo, &hi);
  (void)hipStreamGetFlags(s, &uf);
  (void)hipStreamQuery(s);
  (void)hipEventCreateWithFlags(&e, hipEventDisableTiming);
  (void)hipEventCreate(&e2);
  (void)hipEventRecord(e2, s);
  (void)hipEventQuery(e2);
  (void)hipEventElapsedTime(&ms, e2, e2);
  (void)hipEventDestroy(nullptr);
  (void)hipEventDestroy(e);
  (void)hipEventDestroy(e2);
  (void)hipLaunchHostFunc(s, host_fn, nullptr);
  (void)hipLaunchHostFunc(s, nullptr, nullptr);
  (void)hipStreamAddCallback(s, stream_cb, nullptr, 0);
  (void)hipStreamSynchronize(s);
  (void)hipStreamDestroy(s);
  (void)hipStreamDestroy(s2);

  // strings and character buffers
  (void)hipDeviceGetName(name, 64, 0);
  (void)hipDeviceGetPCIBusId(nullptr, 0, 0);
  (void)hipModuleGetFunction(nullptr, nullptr, nullptr);
  (void)hipModuleGetFunction(nullptr, nullptr, "kernel_name");
  (void)hipGetErrorName(hipErrorInvalidValue);
  (void)hipGetErrorString(hipErrorInvalidValue);

  // enums and integers
  (void)hipDeviceSetCacheConfig(hipFuncCachePreferNone);
  (void)hipDeviceSetSharedMemConfig(hipSharedMemBankSizeDefault);
  (void)hipDeviceGetAttribute(&iv, hipDeviceAttributeWarpSize, 0);
  (void)hipDeviceGetLimit(&total, hipLimitStackSize);
  (void)hipSetDevice(0);
  (void)hipGetDevice(&iv);
  (void)hipGetDeviceCount(&iv);
  (void)hipDriverGetVersion(&iv);
  (void)hipRuntimeGetVersion(&iv);
  (void)hipMemGetInfo(&freeb, &total);
  (void)hipMallocPitch(&p, &pitch, 100, 4);
  (void)hipMemset2D(p, pitch, 3, 100, 4);
  (void)hipFree(p);
  (void)hipMallocManaged(&p, 4096, hipMemAttachGlobal);
  (void)hipMemAdvise(p, 4096, hipMemAdviseSetReadMostly, 0);
  hipMemLocation loc{};
  loc.type = hipMemLocationTypeDevice;
  loc.id = 0;
  (void)hipMemAdvise_v2(p, 4096, hipMemAdviseSetReadMostly, loc);
  (void)hipMemPrefetchAsync_v2(p, 4096, loc, 0, nullptr);
  for (int type : {0, 2, 3, 4, 9}) {
    loc.type = static_cast<hipMemLocationType>(type);
    (void)hipMemAdvise_v2(p, 4096, hipMemAdviseSetReadMostly, loc);
  }
  (void)hipFree(p);
  (void)hipHostMalloc(&p, 4096, hipHostMallocDefault);
  (void)hipHostFree(p);

  // structs by value, struct pointers
  hipPitchedPtr pp{};
  (void)hipMalloc3D(&pp, make_hipExtent(64, 2, 1));
  (void)hipMemset3D(pp, 5, make_hipExtent(64, 2, 1));
  (void)hipFree(pp.ptr);
  hipDeviceProp_t prop;
  (void)hipGetDeviceProperties(&prop, 0);
  hipCtx_t ctx = nullptr;
  (void)hipCtxGetCurrent(&ctx);

  // graphs
  hipGraph_t g = nullptr;
  hipGraphNode_t n = nullptr;
  size_t count = 0;
  (void)hipGraphCreate(&g, 0);
  (void)hipGraphAddEmptyNode(&n, g, nullptr, 0);
  (void)hipGraphGetNodes(g, nullptr, &count);
  (void)hipGraphDestroy(g);
  (void)hipGraphDestroy(nullptr);

  std::printf("args wrong %d\n", wrong);
  return 0;
}
