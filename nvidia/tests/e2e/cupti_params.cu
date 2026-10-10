// The parameter structures the callback API hands a subscriber for runtime
// calls: a program makes a few dozen calls of every family the runtime has
// (streams, events, memory of every kind, pools, arrays, graphs, launches,
// device queries) and a subscriber prints each call's entry with its parameters
// -- every field by name, a number as itself and a pointer as set or null --
// and its exit with the value it returned.
//
// Run against NVIDIA's libcupti on a card it prints what the toolkit's own
// structures hold for each call (nvidia/tests/data/cupti_params.expected); run
// against this project's it must print the same, which checks that the
// argument of each position reaches the field of that position, for every
// function a table made from the toolkit's header converts (see
// scripts/gen_cupti_runtime_conv.py). The functions are those whose structure
// is the same in CUDA 12.0, 12.8 and 13.x (gen_cupti_params.py makes the table
// of fields, cupti_params.inc).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cupti.h>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) std::printf("[%s -> %d]\n", #x, static_cast<int>(e_));     \
  } while (0)

namespace {

std::vector<std::string> g_lines;

template <class T>
std::string field(const char* name, const T& v) {
  char buf[96];
  if constexpr (std::is_pointer_v<T>) {
    std::snprintf(buf, sizeof buf, " %s=%s", name, v ? "set" : "null");
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
#include "cupti_params.inc"
    default: return "";
  }
}

void CUPTIAPI on_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
  const auto* cb = static_cast<const CUpti_CallbackData*>(data);
  const bool enter = cb->callbackSite == CUPTI_API_ENTER;
  std::string line = std::string("CB ") + cb->functionName + (enter ? " ENTER" : " EXIT");
  if (enter) {
    line += params_of(cbid, cb->functionParams);
  } else if (std::strcmp(cb->functionName, "cudaGetErrorString") == 0 ||
             std::strcmp(cb->functionName, "cudaGetErrorName") == 0 ||
             std::strcmp(cb->functionName, "cudaCreateChannelDesc") == 0 || !cb->functionReturnValue) {
    // These return a string or a structure, not a status.
  } else {
    line += " ret=" + std::to_string(static_cast<int>(*static_cast<const cudaError_t*>(cb->functionReturnValue)));
  }
  if (std::getenv("PARAMS_LIVE")) std::fprintf(stderr, "%s\n", line.c_str());
  g_lines.push_back(line);
}

}  // namespace

__device__ int d_symbol[16];
__global__ void touch(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.f;
}

static void CUDART_CB host_cb(void*) {}
static void CUDART_CB stream_cb(cudaStream_t, cudaError_t, void*) {}

int main() {
  CUpti_SubscriberHandle sub;
  if (cuptiSubscribe(&sub, on_callback, nullptr) != CUPTI_SUCCESS) return 1;
  cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API);

  // Warm-up: make the context, so lazy creation is not in the output.
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  g_lines.clear();

  // ---- device and version queries ----
  int n = 0, v = 0, dev = 0;
  CK(cudaGetDeviceCount(&n));
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerBlock, 0));
  CK(cudaRuntimeGetVersion(&v));
  CK(cudaDriverGetVersion(&v));
  size_t limit = 0;
  CK(cudaDeviceGetLimit(&limit, cudaLimitStackSize));
  int least = 0, greatest = 0;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  cudaFuncCache cache;
  CK(cudaDeviceGetCacheConfig(&cache));
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  int by_bus = -1;
  CK(cudaDeviceGetByPCIBusId(&by_bus, bus));

  // ---- streams and events ----
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithPriority(&s2, cudaStreamDefault, 0));
  unsigned int flags = 0;
  int priority = 0;
  CK(cudaStreamGetFlags(s1, &flags));
  CK(cudaStreamGetPriority(s2, &priority));
  CK(cudaStreamQuery(s1));
  cudaStreamCaptureStatus capturing;
  CK(cudaStreamIsCapturing(s1, &capturing));
  cudaEvent_t e1;
  CK(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
  CK(cudaEventRecord(e1, s1));
  CK(cudaEventSynchronize(e1));
  CK(cudaStreamWaitEvent(s2, e1, 0));
  CK(cudaStreamAddCallback(s1, stream_cb, nullptr, 0));
  CK(cudaLaunchHostFunc(s1, host_cb, nullptr));
  CK(cudaStreamSynchronize(s1));
  CK(cudaEventDestroy(e1));

  // ---- memory ----
  float *d = nullptr, *d2 = nullptr, *pitched = nullptr, *managed = nullptr, *pinned = nullptr;
  size_t pitch = 0;
  CK(cudaMalloc(&d, 4096));
  CK(cudaMalloc(&d2, 4096));
  CK(cudaMallocPitch(reinterpret_cast<void**>(&pitched), &pitch, 100, 8));
  CK(cudaMallocManaged(reinterpret_cast<void**>(&managed), 4096));
  CK(cudaHostAlloc(reinterpret_cast<void**>(&pinned), 4096, cudaHostAllocMapped));
  void* dev_ptr = nullptr;
  CK(cudaHostGetDevicePointer(&dev_ptr, pinned, 0));
  unsigned int host_flags = 0;
  CK(cudaHostGetFlags(&host_flags, pinned));
  cudaPointerAttributes attr;
  CK(cudaPointerGetAttributes(&attr, d));
  float host[64] = {};
  CK(cudaMemset(d, 0, 4096));
  CK(cudaMemsetAsync(d2, 0, 4096, s1));
  CK(cudaMemset2D(pitched, pitch, 0, 100, 8));
  CK(cudaMemcpy(d, host, sizeof host, cudaMemcpyHostToDevice));
  CK(cudaMemcpyAsync(host, d, sizeof host, cudaMemcpyDeviceToHost, s1));
  CK(cudaMemcpy2D(d2, 256, d, 256, 128, 4, cudaMemcpyDeviceToDevice));
  CK(cudaMemcpy2DAsync(d2, 256, d, 256, 128, 4, cudaMemcpyDeviceToDevice, s1));
  CK(cudaMemcpyToSymbol(d_symbol, host, sizeof(int) * 4));
  CK(cudaMemcpyFromSymbol(host, d_symbol, sizeof(int) * 4));
  void* sym = nullptr;
  size_t sym_size = 0;
  CK(cudaGetSymbolAddress(&sym, d_symbol));
  CK(cudaGetSymbolSize(&sym_size, d_symbol));
  CK(cudaMemcpyPeer(d2, 0, d, 0, 256));
  CK(cudaStreamSynchronize(s1));
  CK(cudaFreeHost(pinned));
  CK(cudaFree(managed));
  CK(cudaFree(pitched));

  // ---- arrays ----
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaArray_t arr;
  CK(cudaMallocArray(&arr, &desc, 16, 16));
  CK(cudaMemcpy2DToArray(arr, 0, 0, host, 64, 64, 1, cudaMemcpyHostToDevice));
  CK(cudaMemcpy2DFromArray(host, 64, arr, 0, 0, 64, 1, cudaMemcpyDeviceToHost));
  CK(cudaFreeArray(arr));

  // ---- pools ----
  cudaMemPool_t pool, def;
  cudaMemPoolProps props;
  std::memset(&props, 0, sizeof props);
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = 0;
  CK(cudaDeviceGetDefaultMemPool(&def, 0));
  CK(cudaMemPoolCreate(&pool, &props));
  unsigned long long threshold = 1 << 20;
  CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  CK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  void* pooled = nullptr;
  CK(cudaMallocFromPoolAsync(&pooled, 1 << 16, pool, s1));
  // Sharing a pool, or a pointer from one, with another process: refused on
  // this machine, and the callbacks carry the refusals.
  {
    int fd = -1;
    cudaMemPool_t imported = nullptr;
    cudaMemPoolPtrExportData exported;
    std::memset(&exported, 0, sizeof exported);
    void* imported_ptr = nullptr;
    (void)cudaMemPoolExportToShareableHandle(&fd, pool, cudaMemHandleTypePosixFileDescriptor, 0);
    (void)cudaMemPoolImportFromShareableHandle(&imported, &fd, cudaMemHandleTypePosixFileDescriptor, 0);
    (void)cudaMemPoolExportPointer(&exported, pooled);
    (void)cudaMemPoolImportPointer(&imported_ptr, pool, &exported);
  }
  CK(cudaFreeAsync(pooled, s1));
  CK(cudaStreamSynchronize(s1));
  CK(cudaMemPoolTrimTo(pool, 0));
  CK(cudaMemPoolDestroy(pool));
  CK(cudaMallocAsync(&pooled, 1 << 12, s1));
  CK(cudaFreeAsync(pooled, s1));
  CK(cudaStreamSynchronize(s1));

  // ---- external memory and semaphores: nothing to import on this machine ----
  {
    cudaExternalMemoryHandleDesc md;
    std::memset(&md, 0, sizeof md);
    md.type = cudaExternalMemoryHandleTypeOpaqueFd;
    md.handle.fd = -1;
    md.size = 4096;
    cudaExternalMemory_t em = nullptr;
    (void)cudaImportExternalMemory(&em, &md);
    cudaExternalMemoryBufferDesc bd;
    std::memset(&bd, 0, sizeof bd);
    bd.size = 256;
    void* mapped = nullptr;
    (void)cudaExternalMemoryGetMappedBuffer(&mapped, em, &bd);
    (void)cudaDestroyExternalMemory(em);
    cudaExternalSemaphoreHandleDesc sd;
    std::memset(&sd, 0, sizeof sd);
    sd.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    sd.handle.fd = -1;
    cudaExternalSemaphore_t es = nullptr;
    (void)cudaImportExternalSemaphore(&es, &sd);
    cudaExternalSemaphoreSignalParams sp;
    std::memset(&sp, 0, sizeof sp);
    (void)cudaSignalExternalSemaphoresAsync(&es, &sp, 0, s1);   // a null handle crashes the card's
    cudaExternalSemaphoreWaitParams wp;
    std::memset(&wp, 0, sizeof wp);
    (void)cudaWaitExternalSemaphoresAsync(&es, &wp, 0, s1);
    (void)cudaDestroyExternalSemaphore(es);
    (void)cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                             cudaFlushGPUDirectRDMAWritesToOwner);
  }

  // ---- kernels ----
  CK(cudaFuncSetCacheConfig(reinterpret_cast<const void*>(touch), cudaFuncCachePreferNone));
  CK(cudaFuncSetAttribute(reinterpret_cast<const void*>(touch), cudaFuncAttributeMaxDynamicSharedMemorySize, 1024));
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(touch)));
  int blocks = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reinterpret_cast<const void*>(touch), 128, 0));
  touch<<<1, 32, 0, s1>>>(d, 32);
  void* args[] = {&d, &n};
  n = 32;
  CK(cudaLaunchKernel(reinterpret_cast<const void*>(touch), dim3(1), dim3(32), args, 0, s1));
  CK(cudaStreamSynchronize(s1));

  // ---- graphs ----
  cudaGraph_t g, clone, child;
  CK(cudaGraphCreate(&g, 0));
  CK(cudaGraphCreate(&child, 0));
  cudaGraphNode_t kn, en, mn, cn;
  cudaKernelNodeParams kp;
  std::memset(&kp, 0, sizeof kp);
  kp.func = reinterpret_cast<void*>(touch);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(32);
  kp.kernelParams = args;
  CK(cudaGraphAddKernelNode(&kn, g, nullptr, 0, &kp));
  CK(cudaGraphAddEmptyNode(&en, g, &kn, 1));
  cudaMemsetParams mp;
  std::memset(&mp, 0, sizeof mp);
  mp.dst = d2;
  mp.value = 0;
  mp.elementSize = 1;
  mp.width = 64;
  mp.height = 1;
  CK(cudaGraphAddMemsetNode(&mn, g, &en, 1, &mp));
  CK(cudaGraphAddMemcpyNode1D(&cn, g, &mn, 1, d2, d, 64, cudaMemcpyDeviceToDevice));
  CK(cudaGraphAddEmptyNode(&en, child, nullptr, 0));
  cudaGraphNode_t chn;
  CK(cudaGraphAddChildGraphNode(&chn, g, &cn, 1, child));
  size_t nodes = 0;
  CK(cudaGraphGetNodes(g, nullptr, &nodes));
  std::vector<cudaGraphNode_t> all(nodes);
  CK(cudaGraphGetNodes(g, all.data(), &nodes));
  cudaGraphNodeType type;
  CK(cudaGraphNodeGetType(kn, &type));
  {
    cudaKernelNodeAttrValue av;
    std::memset(&av, 0, sizeof av);
    (void)cudaGraphKernelNodeGetAttribute(kn, cudaKernelNodeAttributeAccessPolicyWindow, &av);
    (void)cudaGraphKernelNodeSetAttribute(kn, cudaKernelNodeAttributeAccessPolicyWindow, &av);
    (void)cudaGraphKernelNodeCopyAttributes(kn, kn);
  }
  CK(cudaGraphClone(&clone, g));
  cudaGraphNode_t found;
  CK(cudaGraphNodeFindInClone(&found, kn, clone));
  cudaGraphExec_t ge;
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphLaunch(ge, s1));
  CK(cudaStreamSynchronize(s1));
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroyNode(chn));
  CK(cudaGraphDestroy(clone));
  CK(cudaGraphDestroy(g));
  CK(cudaGraphDestroy(child));

  // A graph with semaphore nodes that name no semaphore; it is never launched.
  {
    cudaGraph_t sg;
    CK(cudaGraphCreate(&sg, 0));
    cudaExternalSemaphoreSignalNodeParams ssp;
    cudaExternalSemaphoreWaitNodeParams swp;
    std::memset(&ssp, 0, sizeof ssp);
    std::memset(&swp, 0, sizeof swp);
    cudaGraphNode_t sn, wn;
    (void)cudaGraphAddExternalSemaphoresSignalNode(&sn, sg, nullptr, 0, &ssp);
    (void)cudaGraphAddExternalSemaphoresWaitNode(&wn, sg, nullptr, 0, &swp);
    (void)cudaGraphExternalSemaphoresSignalNodeGetParams(sn, &ssp);
    (void)cudaGraphExternalSemaphoresSignalNodeSetParams(sn, &ssp);
    (void)cudaGraphExternalSemaphoresWaitNodeGetParams(wn, &swp);
    (void)cudaGraphExternalSemaphoresWaitNodeSetParams(wn, &swp);
    cudaGraphExec_t sge;
    CK(cudaGraphInstantiate(&sge, sg, 0));
    (void)cudaGraphExecExternalSemaphoresSignalNodeSetParams(sge, sn, &ssp);
    (void)cudaGraphExecExternalSemaphoresWaitNodeSetParams(sge, wn, &swp);
    CK(cudaGraphExecDestroy(sge));
    CK(cudaGraphDestroy(sg));
  }

  CK(cudaStreamDestroy(s1));
  CK(cudaStreamDestroy(s2));
  CK(cudaFree(d));
  CK(cudaFree(d2));

  cuptiUnsubscribe(sub);
  for (const std::string& l : g_lines) std::printf("%s\n", l.c_str());
  std::printf("# end\n");
  return 0;
}
