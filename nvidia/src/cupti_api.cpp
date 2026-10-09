// libvgpucupti — VirtualGPU's implementation of the CUPTI Activity API.
//
// WHAT THIS IS FOR
// ----------------
// Everything that profiles CUDA reads it through CUPTI: Nsight Systems, nvprof,
// PyTorch's profiler, DCGM. Without this library the counters and the timeline
// this engine already has are reachable only through VGPU_COUNTERS, which no
// existing tool knows how to read. With it, a program's own profiler sees the
// kernels it launched, in order, with the arguments it launched them with.
//
// WHAT THE TIMESTAMPS MEAN
// ------------------------
// They are wall-clock nanoseconds spent *simulating*, not a prediction of how
// long a device would take. There is no timing model here and there is not
// going to be one by accident. So a timeline drawn from these is truthful about
// what ran and in what order, and says nothing about how fast hardware would be.
// That is worth stating plainly rather than letting someone read a flame graph
// and believe it.
//
// The activity records themselves come from the toolkit's own headers. Their
// layout is version-specific, and hand-rolling a copy of a versioned struct is
// how cudaFuncAttributes once reported register counts in the tens of thousands.
#include <cupti.h>
#include <cuda_runtime_api.h>

// NVTX's types, for the tool side of its injection protocol. The header-only
// library is not needed: only its declarations, which NVTX_NO_IMPL asks for.
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>) && __has_include(<generated_nvtx_meta.h>)
#define VGPU_NVTX 1
#define NVTX_NO_IMPL 1
#include <nvtx3/nvToolsExt.h>
#if __has_include(<nvtx3/nvToolsExtSync.h>)
#include <nvtx3/nvToolsExtSync.h>
#endif
#include <generated_nvtx_meta.h>
#endif
#endif

#include <dlfcn.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vgpu/profiling.hpp"

#ifndef VGPU_EXPORT
#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// The newest layout of each record the toolkit being built against defines. A
// consumer built with that toolkit reads these fields at these offsets, so
// handing it an older layout reads other fields.
#if CUPTI_API_VERSION >= 130200
using KernelRecord = CUpti_ActivityKernel11;
using DeviceRecord = CUpti_ActivityDevice6;
using ContextRecord = CUpti_ActivityContext4;
using MemcpyRecord = CUpti_ActivityMemcpy6;
using SyncRecord = CUpti_ActivitySynchronization2;
#elif CUPTI_API_VERSION >= 130000
using KernelRecord = CUpti_ActivityKernel10;
using DeviceRecord = CUpti_ActivityDevice5;
using ContextRecord = CUpti_ActivityContext3;
using MemcpyRecord = CUpti_ActivityMemcpy6;
using SyncRecord = CUpti_ActivitySynchronization2;
#elif CUPTI_API_VERSION >= 26
using KernelRecord = CUpti_ActivityKernel9;
using DeviceRecord = CUpti_ActivityDevice5;
using ContextRecord = CUpti_ActivityContext3;
using MemcpyRecord = CUpti_ActivityMemcpy6;
using SyncRecord = CUpti_ActivitySynchronization2;
#else
using KernelRecord = CUpti_ActivityKernel9;
using DeviceRecord = CUpti_ActivityDevice4;
using ContextRecord = CUpti_ActivityContext;
using MemcpyRecord = CUpti_ActivityMemcpy5;
using SyncRecord = CUpti_ActivitySynchronization;
#endif
using MemsetRecord = CUpti_ActivityMemset4;

// The records of the kinds below, at the layout of the toolkit being built
// against (the same rule: a consumer built with that toolkit reads those offsets).
#if CUPTI_API_VERSION >= 24
using Memory2Record = CUpti_ActivityMemory4;
#else
using Memory2Record = CUpti_ActivityMemory3;
#endif
#if CUPTI_API_VERSION >= 130000
using MemoryPoolRecord = CUpti_ActivityMemoryPool3;
#else
using MemoryPoolRecord = CUpti_ActivityMemoryPool2;
#endif
#if CUPTI_API_VERSION >= 22
using OverheadRecord = CUpti_ActivityOverhead3;
using GraphTraceRecord = CUpti_ActivityGraphTrace2;
#else
using OverheadRecord = CUpti_ActivityOverhead;
using GraphTraceRecord = CUpti_ActivityGraphTrace;
#endif
#if CUPTI_API_VERSION >= 26
using CudaEventRecord = CUpti_ActivityCudaEvent2;
#else
using CudaEventRecord = CUpti_ActivityCudaEvent;
#endif
using PeerCopyRecord = CUpti_ActivityMemcpyPtoP4;
using MemoryV1Record = CUpti_ActivityMemory;
using FunctionRecord = CUpti_ActivityFunction;

// Kinds the older toolkits' headers do not name. The numbers are CUPTI's own
// and stable.
namespace akind {
constexpr int kEvent = 6, kMetric = 7, kSourceLocator = 14, kGlobalAccess = 15, kBranch = 16, kOverhead = 17,
              kCdpKernel = 18, kEnvironment = 20, kEventInstance = 21, kMemcpy2 = 22, kMetricInstance = 23,
              kInstructionExecution = 24, kUnifiedMemoryCounter = 25, kFunction = 26, kModule = 27,
              kDeviceAttribute = 28, kSharedAccess = 29, kPcSampling = 30, kPcSamplingRecordInfo = 31,
              kInstructionCorrelation = 32, kCudaEvent = 36, kInstantaneousEvent = 41,
              kInstantaneousEventInstance = 42, kInstantaneousMetric = 43, kInstantaneousMetricInstance = 44,
              kMemory = 45, kMemory2 = 49, kMemoryPool = 50, kGraphTrace = 51, kDeviceGraphTrace = 53,
              kCount = 56;
}  // namespace akind

namespace {

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

// VGPU_TRACE=1 (the engine's trace switch) logs the activity calls a tool makes
// and what they answered, which is how to see what a profiler asked for.
bool tracing() {
  static const bool on = [] {
    const char* t = std::getenv("VGPU_TRACE");
    return t && t[0] && t[0] != '0';
  }();
  return on;
}
#define CUPTI_TRACE(...)                                          \
  do {                                                            \
    if (tracing()) {                                              \
      std::fprintf(stderr, "[vgpu][cupti] " __VA_ARGS__);         \
      std::fprintf(stderr, "\n");                                 \
    }                                                             \
  } while (0)

std::mutex g_mu;
CUpti_BuffersCallbackRequestFunc g_request = nullptr;
CUpti_BuffersCallbackCompleteFunc g_complete = nullptr;
std::vector<bool> g_kinds(64, false);   // indexed by kind; CUPTI_ACTIVITY_KIND_COUNT differs between toolkits
bool g_announced = false;
bool g_devices_delivered = false;
std::atomic<bool> g_cuda_event_device_timestamps{false};

// Which API functions' records are wanted (cuptiActivityEnableRuntimeApi and
// cuptiActivityEnableDriverApi), as measured on an RTX 3060:
//  * with the kind on, every function is recorded but those switched off;
//  * with the kind off, only the functions switched on are -- but only while
//    the kind has never been turned off: once it has, switching a function on
//    records nothing until the kind is turned on again;
//  * turning the kind on forgets every per-function choice.
struct ApiFilter {
  bool kind = false;
  bool dead = false;
  std::unordered_map<uint32_t, bool> per_function;
  bool any_on() const {
    if (kind) return true;
    for (const auto& kv : per_function)
      if (kv.second) return true;
    return false;
  }
  bool wanted(uint32_t cbid, bool active) const {
    if (!active) return false;
    const auto it = per_function.find(cbid);
    if (kind) return it == per_function.end() || it->second;
    return !dead && it != per_function.end() && it->second;
  }
};
ApiFilter g_runtime_filter, g_driver_filter;
// Whether API calls are traced at all. Measured on an RTX 3060: the choice of
// which API functions are recorded takes effect only if something that records
// API calls was switched on when the first context was made (a kind, or a
// function), or when a kind of API records is switched on afterwards; a
// function switched on at any other time is accepted and records nothing.
bool g_api_active = false;

// Kinds this can actually produce. Enabling anything else succeeds -- refusing
// would stop a profiler that asks for everything and uses what arrives -- but
// nothing will come back for it, which is the honest outcome for a device
// with no timing model and no hardware counters behind these kinds. (Kinds an
// RTX 3060 refuses are refused too; see enable_result.)
bool produced(int k) {
  return k == CUPTI_ACTIVITY_KIND_KERNEL || k == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
         k == CUPTI_ACTIVITY_KIND_MEMCPY || k == CUPTI_ACTIVITY_KIND_MEMSET ||
         k == CUPTI_ACTIVITY_KIND_RUNTIME || k == CUPTI_ACTIVITY_KIND_DRIVER ||
         k == CUPTI_ACTIVITY_KIND_SYNCHRONIZATION ||
         k == CUPTI_ACTIVITY_KIND_DEVICE || k == CUPTI_ACTIVITY_KIND_CONTEXT ||
         k == CUPTI_ACTIVITY_KIND_STREAM || k == CUPTI_ACTIVITY_KIND_MARKER ||
         k == CUPTI_ACTIVITY_KIND_MARKER_DATA || k == CUPTI_ACTIVITY_KIND_NAME ||
         k == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION || k == akind::kMemcpy2 || k == akind::kCudaEvent ||
         k == akind::kMemory || k == akind::kMemory2 || k == akind::kMemoryPool || k == akind::kGraphTrace ||
         k == akind::kOverhead || k == akind::kFunction;
}

// Whether anything is wanted that the runtime has to record.
bool anything_recorded() {
  for (int i = 0; i < static_cast<int>(g_kinds.size()); ++i)
    if (g_kinds[static_cast<size_t>(i)] && produced(i)) return true;
  return g_runtime_filter.any_on() || g_driver_filter.any_on();
}

void announce_once() {
  if (g_announced || quiet()) return;
  g_announced = true;
  std::fprintf(stderr,
               "[vgpu] CUPTI activity recording enabled. Timestamps are wall-clock time spent "
               "simulating, not device time: VirtualGPU has no timing model, so a timeline from "
               "these is truthful about order and about how long simulation took, and says "
               "nothing about hardware speed.\n");
}

// Kernel names reach the consumer as pointers that must outlive the record, so
// they are interned rather than pointed at a temporary.
const char* intern(const std::string& s) {
  static std::mutex mu;
  static std::unordered_map<std::string, std::string> pool;
  std::lock_guard<std::mutex> lock(mu);
  return pool.emplace(s, s).first->second.c_str();
}

/* ---- identifiers ----
   Contexts and streams are the driver's pointers; a profiler wants small
   integers that mean the same thing in the records and in the calls that ask
   (cuptiGetStreamId). The default stream is 7, as CUDA 12 and 13 number it.
   Ids 6 to 13 are the driver's own streams of the first context (see
   internal_streams), so a program's streams start at 14, as they do on an RTX
   3060 once a context exists. */

std::mutex g_id_mu;
std::unordered_map<uint64_t, uint32_t> g_stream_ids;
std::unordered_map<uint64_t, uint32_t> g_context_ids;
std::unordered_map<uint64_t, uint32_t> g_event_ids;
std::unordered_map<uint32_t, uint32_t> g_device_context;   // device -> its context's id
std::unordered_map<uint32_t, uint32_t> g_device_first_stream;   // device -> the first of its context's own streams
uint32_t g_next_stream_id = 14;
uint32_t g_next_context_id = 1;
bool g_first_context_streams_done = false;

// `generation`: how many times the handle had been destroyed when the work was done
// (vgpu::profiling::stream_generation). A real driver numbers a stream made after another
// was destroyed afresh even when it gets the same pointer, and so does this.
uint32_t stream_id_of(uint64_t handle, uint32_t generation) {
  if (handle == 0 || handle == 1 || handle == 2) return 7;   // 0, cudaStreamLegacy, cudaStreamPerThread
  std::lock_guard<std::mutex> lock(g_id_mu);
  const uint64_t key = handle ^ (uint64_t{generation} << 48);   // user-space pointers have 47 bits
  const auto it = g_stream_ids.find(key);
  if (it != g_stream_ids.end()) return it->second;
  return g_stream_ids.emplace(key, g_next_stream_id++).first->second;
}

uint32_t event_id_of(uint64_t handle) {
  std::lock_guard<std::mutex> lock(g_id_mu);
  return g_event_ids.emplace(handle, 1u + static_cast<uint32_t>(g_event_ids.size())).first->second;
}

uint32_t context_id_of(uint64_t handle) {
  std::lock_guard<std::mutex> lock(g_id_mu);
  const auto it = g_context_ids.find(handle);
  if (it != g_context_ids.end()) return it->second;
  return g_context_ids.emplace(handle, g_next_context_id++).first->second;
}

// The driver's current context, which is the one a runtime program is running
// on. A program that never touched the driver API has one all the same.
CUcontext current_context() {
  using Get = CUresult (*)(CUcontext*);
  static const Get get = reinterpret_cast<Get>(dlsym(RTLD_DEFAULT, "cuCtxGetCurrent"));
  static int primary = 0;
  CUcontext c = nullptr;
  if (get) {
    vgpu::profiling::Silence silent;
    get(&c);
  }
  return c ? c : reinterpret_cast<CUcontext>(&primary);
}

uint32_t kStandInContext() { return context_id_of(reinterpret_cast<uint64_t>(current_context())); }

// The context a device's work ran in: the one its context record announced, or
// (for work whose context was made before recording began) the current one.
uint32_t ctx_of_device(uint32_t device) {
  {
    std::lock_guard<std::mutex> lock(g_id_mu);
    const auto it = g_device_context.find(device);
    if (it != g_device_context.end()) return it->second;
  }
  return kStandInContext();
}

// One of the driver's own streams in a device's context, by its place among the
// eight (see note_context); the first context's streams are 6 to 13.
uint32_t internal_stream_id(uint32_t device, int index) {
  std::lock_guard<std::mutex> lock(g_id_mu);
  const auto it = g_device_first_stream.find(device);
  return (it != g_device_first_stream.end() ? it->second : 6u) + static_cast<uint32_t>(index);
}

// A context coming into being: its id, and the eight streams the driver makes
// for itself inside it (they get ids of their own, ahead of any stream the
// program makes). The first of the eight is returned, or 0 if this context was
// announced already. `handle` is the driver's context, 0 for the runtime's
// primary context of the device (of which a device has a new one after each
// reset, `generation`).
uint32_t note_context(uint32_t device, uint64_t handle, uint64_t generation, uint32_t* ctx_id) {
  std::lock_guard<std::mutex> lock(g_id_mu);
  const uint64_t key = handle ? handle : (uint64_t{1} << 62) + device + (generation << 16);
  const auto it = g_context_ids.find(key);
  if (it != g_context_ids.end()) {
    *ctx_id = it->second;
    return 0;
  }
  *ctx_id = g_context_ids[key] = g_next_context_id++;
  if (!handle) g_device_context[device] = *ctx_id;
  uint32_t first = 0;
  if (!g_first_context_streams_done) {
    g_first_context_streams_done = true;
    first = 6;
  } else {
    first = g_next_stream_id;
    g_next_stream_id += 8;
  }
  if (!handle) g_device_first_stream[device] = first;
  return first;
}

/* ---- records ---- */

size_t fill_kernel(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* k = reinterpret_cast<KernelRecord*>(out);
  std::memset(k, 0, sizeof *k);
  k->kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
  k->start = e.start_ns;
  k->end = e.end_ns;
  k->deviceId = e.device;
  k->contextId = ctx_of_device(e.device);
  k->correlationId = e.correlation;
  k->streamId = stream_id_of(e.stream, e.stream_gen);
  k->gridX = static_cast<int32_t>(e.grid[0]);
  k->gridY = static_cast<int32_t>(e.grid[1]);
  k->gridZ = static_cast<int32_t>(e.grid[2]);
  k->blockX = static_cast<int32_t>(e.block[0]);
  k->blockY = static_cast<int32_t>(e.block[1]);
  k->blockZ = static_cast<int32_t>(e.block[2]);
  k->dynamicSharedMemory = e.shared_bytes;
  k->staticSharedMemory = e.static_shared_bytes;
  k->registersPerThread = static_cast<uint16_t>(e.registers_per_thread);
  k->localMemoryPerThread = e.local_bytes_per_thread;
  k->graphNodeId = e.graph_node_id;
  k->graphId = e.graph_id;
  k->name = intern(e.name);
  return sizeof *k;
}

CUpti_ActivityMemoryKind memory_kind(vgpu::profiling::MemKind k) {
  using vgpu::profiling::MemKind;
  switch (k) {
    case MemKind::Pageable: return CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE;
    case MemKind::Pinned: return CUPTI_ACTIVITY_MEMORY_KIND_PINNED;
    case MemKind::Device: return CUPTI_ACTIVITY_MEMORY_KIND_DEVICE;
    case MemKind::Array: return CUPTI_ACTIVITY_MEMORY_KIND_ARRAY;
    case MemKind::Managed: return CUPTI_ACTIVITY_MEMORY_KIND_MANAGED;
    default: return CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN;
  }
}

size_t fill_memcpy(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* m = reinterpret_cast<MemcpyRecord*>(out);
  std::memset(m, 0, sizeof *m);
  m->kind = CUPTI_ACTIVITY_KIND_MEMCPY;
  m->start = e.start_ns;
  m->end = e.end_ns;
  m->deviceId = e.device;
  m->contextId = ctx_of_device(e.device);
  m->correlationId = e.correlation;
  m->streamId = e.internal_stream >= 0 ? internal_stream_id(e.device, e.internal_stream) : stream_id_of(e.stream, e.stream_gen);
  m->bytes = e.bytes;
  m->srcKind = memory_kind(e.src_kind);
  m->dstKind = memory_kind(e.dst_kind);
  if (e.async) m->flags |= CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
  m->graphNodeId = e.graph_node_id;
  m->graphId = e.graph_id;
  switch (e.copy_kind) {
    case 1: m->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD; break;
    case 2: m->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOH; break;
    case 3: m->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOD; break;
    default: m->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOH; break;
  }
  return sizeof *m;
}

size_t fill_memset(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* m = reinterpret_cast<MemsetRecord*>(out);
  std::memset(m, 0, sizeof *m);
  m->kind = CUPTI_ACTIVITY_KIND_MEMSET;
  m->start = e.start_ns;
  m->end = e.end_ns;
  m->deviceId = e.device;
  m->contextId = ctx_of_device(e.device);
  m->correlationId = e.correlation;
  m->streamId = stream_id_of(e.stream, e.stream_gen);
  m->bytes = e.bytes;
  // A fill that is a node of a launched graph reports neither its value nor its
  // memory kind (an RTX 3060's records for one carry zero for both).
  m->value = e.graph_node_id ? 0 : e.value;
  m->memoryKind = e.graph_node_id ? 0 : static_cast<uint16_t>(memory_kind(e.dst_kind));
  m->graphNodeId = e.graph_node_id;
  m->graphId = e.graph_id;
  if (e.async) m->flags |= CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC;
  return sizeof *m;
}

size_t fill_sync(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<SyncRecord*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;
  r->type = static_cast<CUpti_ActivitySynchronizationType>(e.sync_kind);
  r->start = e.start_ns;
  r->end = e.end_ns;
  r->correlationId = e.correlation;
  r->contextId = ctx_of_device(e.device);
  r->streamId = (e.sync_kind == 1 || e.sync_kind == 4) ? CUPTI_SYNCHRONIZATION_INVALID_VALUE : stream_id_of(e.stream, e.stream_gen);
  r->cudaEventId = e.handle ? event_id_of(e.handle) : CUPTI_SYNCHRONIZATION_INVALID_VALUE;
  return sizeof *r;
}

size_t fill_stream(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<CUpti_ActivityStream*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = CUPTI_ACTIVITY_KIND_STREAM;
  if (e.op == 1) {
    // One of the driver's own streams (see expand): ids and flag are as measured.
    r->contextId = e.context_id;
    r->streamId = e.stream_id;
    r->priority = static_cast<uint32_t>(e.priority);
    r->flag = static_cast<CUpti_ActivityStreamFlag>(e.flags);
    r->correlationId = e.correlation;
    return sizeof *r;
  }
  r->contextId = ctx_of_device(e.device);
  r->streamId = stream_id_of(e.handle, e.stream_gen);
  r->priority = static_cast<uint32_t>(e.priority);
  r->flag = (e.flags & cudaStreamNonBlocking) ? CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING
                                              : CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT;
  r->correlationId = e.correlation;
  return sizeof *r;
}

size_t fill_context(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<ContextRecord*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = CUPTI_ACTIVITY_KIND_CONTEXT;
  r->contextId = e.context_id ? e.context_id : ctx_of_device(e.device);
  r->deviceId = e.device;
  r->computeApiKind = CUPTI_ACTIVITY_COMPUTE_API_CUDA;
  r->nullStreamId = static_cast<uint16_t>(e.stream_id ? e.stream_id : stream_id_of(0, 0));
  return sizeof *r;
}

size_t fill_device(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* d = reinterpret_cast<DeviceRecord*>(out);
  std::memset(d, 0, sizeof *d);
  d->kind = CUPTI_ACTIVITY_KIND_DEVICE;
  d->id = e.device;
  cudaDeviceProp prop;
  std::memset(&prop, 0, sizeof prop);
  {
    vgpu::profiling::Silence silent;
    if (cudaGetDeviceProperties(&prop, static_cast<int>(e.device)) != cudaSuccess) return sizeof *d;
  }
  d->name = intern(prop.name);
  d->globalMemorySize = prop.totalGlobalMem;
  d->constantMemorySize = static_cast<uint32_t>(prop.totalConstMem);
  d->l2CacheSize = static_cast<uint32_t>(prop.l2CacheSize);
  d->numThreadsPerWarp = static_cast<uint32_t>(prop.warpSize);
  d->numMultiprocessors = static_cast<uint32_t>(prop.multiProcessorCount);
  d->maxRegistersPerBlock = static_cast<uint32_t>(prop.regsPerBlock);
  d->maxSharedMemoryPerBlock = static_cast<uint32_t>(prop.sharedMemPerBlock);
  d->maxThreadsPerBlock = static_cast<uint32_t>(prop.maxThreadsPerBlock);
  d->maxBlockDimX = static_cast<uint32_t>(prop.maxThreadsDim[0]);
  d->maxBlockDimY = static_cast<uint32_t>(prop.maxThreadsDim[1]);
  d->maxBlockDimZ = static_cast<uint32_t>(prop.maxThreadsDim[2]);
  d->maxGridDimX = static_cast<uint32_t>(prop.maxGridSize[0]);
  d->maxGridDimY = static_cast<uint32_t>(prop.maxGridSize[1]);
  d->maxGridDimZ = static_cast<uint32_t>(prop.maxGridSize[2]);
  d->computeCapabilityMajor = static_cast<uint32_t>(prop.major);
  d->computeCapabilityMinor = static_cast<uint32_t>(prop.minor);
  d->eccEnabled = static_cast<uint32_t>(prop.ECCEnabled);
  d->isCudaVisible = 1;
  return sizeof *d;
}

size_t fill_marker(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<CUpti_ActivityMarker2*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = CUPTI_ACTIVITY_KIND_MARKER;
  r->flags = static_cast<CUpti_ActivityFlag>(e.flags);
  r->timestamp = e.start_ns;
  r->id = static_cast<uint32_t>(e.handle);
  r->objectKind = CUPTI_ACTIVITY_OBJECT_THREAD;
  r->objectId.pt.processId = e.process_id;
  r->objectId.pt.threadId = e.thread_id;
  r->name = e.name.empty() ? nullptr : intern(e.name);
  r->domain = e.domain.empty() ? nullptr : intern(e.domain);
  return sizeof *r;
}

size_t fill_marker_data(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<CUpti_ActivityMarkerData*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = CUPTI_ACTIVITY_KIND_MARKER_DATA;
  r->flags = static_cast<CUpti_ActivityFlag>(e.flags);
  r->id = static_cast<uint32_t>(e.handle);
  r->payloadKind = static_cast<CUpti_MetricValueKind>(e.payload_kind);
  std::memcpy(&r->payload, &e.payload, sizeof e.payload);
  r->color = e.color;
  r->category = e.category;
  return sizeof *r;
}

size_t fill_external(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<CUpti_ActivityExternalCorrelation*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
  r->externalKind = static_cast<CUpti_ExternalCorrelationKind>(e.flags);
  r->externalId = e.handle;
  r->correlationId = e.correlation;
  return sizeof *r;
}

size_t fill_name(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<CUpti_ActivityName*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = CUPTI_ACTIVITY_KIND_NAME;
  r->objectKind = CUPTI_ACTIVITY_OBJECT_THREAD;
  r->objectId.pt.processId = e.process_id;
  r->objectId.pt.threadId = static_cast<uint32_t>(e.handle);
  r->name = intern(e.name);
  return sizeof *r;
}


/* ---- records of the memory, graph, event and function kinds ---- */

constexpr uint32_t kInvalidStreamId = 0xffffffffu;   // CUPTI_INVALID_STREAM_ID

CUpti_ActivityMemoryPoolType pool_type_of(const vgpu::profiling::Event& e) {
  return e.pool_handle ? CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL : CUPTI_ACTIVITY_MEMORY_POOL_TYPE_INVALID;
}

size_t fill_memory2(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<Memory2Record*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = static_cast<CUpti_ActivityKind>(akind::kMemory2);
  r->memoryOperationType = e.op == 1 ? CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION
                                     : CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE;
  r->memoryKind = memory_kind(e.src_kind);
  r->correlationId = e.correlation;
  r->address = e.address;
  r->bytes = e.bytes;
  r->timestamp = e.start_ns;
  r->processId = e.process_id;
  r->deviceId = e.device;
  r->contextId = ctx_of_device(e.device);
  r->streamId = e.async ? stream_id_of(e.stream, e.stream_gen) : kInvalidStreamId;
  r->isAsync = e.async ? 1 : 0;
  r->memoryPoolConfig.memoryPoolType = pool_type_of(e);
  r->memoryPoolConfig.address = e.pool_handle;
  r->memoryPoolConfig.releaseThreshold = e.pool_threshold;
  r->memoryPoolConfig.pool.size = e.pool_size;
  r->memoryPoolConfig.utilizedSize = e.pool_utilized;
  return sizeof *r;
}

size_t fill_memory_v1(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<MemoryV1Record*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = static_cast<CUpti_ActivityKind>(akind::kMemory);
  r->memoryKind = memory_kind(e.src_kind);
  r->address = e.address;
  r->bytes = e.bytes;
  r->start = e.start_ns;
  r->end = e.end_ns;
  r->processId = e.process_id;
  r->deviceId = e.device;
  r->contextId = ctx_of_device(e.device);
  return sizeof *r;
}

size_t fill_memory_pool(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<MemoryPoolRecord*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = static_cast<CUpti_ActivityKind>(akind::kMemoryPool);
  r->memoryPoolOperationType = e.op == 1   ? CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED
                               : e.op == 2 ? CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED
                                           : CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED;
  r->memoryPoolType = CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL;
  r->correlationId = e.correlation;
  r->processId = e.process_id;
  r->deviceId = e.device;
  // As an RTX 3060 fills it: 1 when a trim gave memory back, 0 otherwise --
  // not the number of bytes the trim was asked to keep.
  r->minBytesToKeep = e.pool_released ? 1 : 0;
  r->address = e.pool_handle;
  r->size = e.pool_size;
  r->releaseThreshold = e.pool_threshold;
  r->timestamp = e.start_ns;
  r->utilizedSize = e.pool_utilized;
  return sizeof *r;
}

size_t fill_graph_trace(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<GraphTraceRecord*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = static_cast<CUpti_ActivityKind>(akind::kGraphTrace);
  r->correlationId = e.correlation;
  r->start = e.start_ns;
  r->end = e.end_ns;
  r->deviceId = e.device;
  r->graphId = e.graph_id;
  r->contextId = ctx_of_device(e.device);
  r->streamId = stream_id_of(e.stream, e.stream_gen);
#if CUPTI_API_VERSION >= 22
  r->endDeviceId = e.device;
  r->endContextId = ctx_of_device(e.device);
#endif
  return sizeof *r;
}

size_t fill_peer_copy(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<PeerCopyRecord*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = static_cast<CUpti_ActivityKind>(akind::kMemcpy2);
  r->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_PTOP;
  r->srcKind = static_cast<uint8_t>(memory_kind(e.src_kind));
  r->dstKind = static_cast<uint8_t>(memory_kind(e.dst_kind));
  if (e.async) r->flags |= CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
  r->bytes = e.bytes;
  r->start = e.start_ns;
  r->end = e.end_ns;
  r->deviceId = e.device;
  r->contextId = ctx_of_device(e.device);
  r->streamId = stream_id_of(e.stream, e.stream_gen);
  r->srcDeviceId = e.src_device;
  r->srcContextId = ctx_of_device(e.src_device);
  r->dstDeviceId = e.dst_device;
  r->dstContextId = ctx_of_device(e.dst_device);
  r->correlationId = e.correlation;
  r->graphNodeId = e.graph_node_id;
  r->graphId = e.graph_id;
  return sizeof *r;
}

std::atomic<uint64_t> g_cuda_event_sync_id{0};   // numbered as records are filled

size_t fill_cuda_event(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<CudaEventRecord*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = static_cast<CUpti_ActivityKind>(akind::kCudaEvent);
  r->correlationId = e.correlation;
  r->contextId = ctx_of_device(e.device);
  r->streamId = stream_id_of(e.stream, e.stream_gen);
  r->eventId = event_id_of(e.handle);
#if CUPTI_API_VERSION >= 26
  r->deviceId = e.device;
  // The simulated device keeps no clock of its own: its "device timestamp" is
  // the time the record was made on the engine's clock, and zero while the
  // collection of device timestamps is off, as on NVIDIA's.
  r->deviceTimestamp = g_cuda_event_device_timestamps.load() ? e.start_ns : 0;
  r->cudaEventSyncId = ++g_cuda_event_sync_id;
#endif
  return sizeof *r;
}

size_t fill_function(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<FunctionRecord*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = static_cast<CUpti_ActivityKind>(akind::kFunction);
  r->id = static_cast<uint32_t>(e.handle);
  r->contextId = ctx_of_device(e.device);
  r->moduleId = e.module_id;
  r->functionIndex = e.function_index;
  r->name = intern(e.name);
  return sizeof *r;
}

size_t fill_overhead(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<OverheadRecord*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = static_cast<CUpti_ActivityKind>(akind::kOverhead);
  r->overheadKind = static_cast<CUpti_ActivityOverheadKind>(e.flags);
  r->objectKind = CUPTI_ACTIVITY_OBJECT_THREAD;
  r->objectId.pt.processId = e.process_id;
  r->objectId.pt.threadId = e.thread_id;
  r->start = e.start_ns;
  r->end = e.end_ns;
#if CUPTI_API_VERSION >= 22
  r->correlationId = e.correlation;
#endif
  return sizeof *r;
}

/* ---- the runtime functions whose calls are recorded ----
   By the callback ID CUPTI names each with (cupti_runtime_cbid.h). The table
   is generated from that header at configure time, so it is the toolkit's own
   and not a hand-kept copy of it: {function, versioned name, ID}. A call not
   listed is not reported -- a record with no ID would name no function. */

struct CbidRow {
  const char* fn;
  const char* versioned;
  CUpti_CallbackId id;
};
const CbidRow kRuntimeCbids[] = {
#define VGPU_CBID_ROW(fn, versioned, id) {fn, versioned, id},
#include "cupti_runtime_cbids.inc"
#undef VGPU_CBID_ROW
};

// The newest ID of each function: the one a program built against this
// toolkit calls through.
const std::unordered_map<std::string, CUpti_CallbackId>& runtime_cbids() {
  static const std::unordered_map<std::string, CUpti_CallbackId> ids = [] {
    std::unordered_map<std::string, CUpti_CallbackId> m;
    for (const auto& r : kRuntimeCbids) {
      auto& slot = m[r.fn];
      if (r.id > slot) slot = r.id;
    }
    return m;
  }();
  return ids;
}

const char* runtime_cbid_name(CUpti_CallbackId id) {
  for (const auto& r : kRuntimeCbids)
    if (r.id == id) return r.versioned;
  return nullptr;
}

CUpti_CallbackId runtime_cbid(const std::string& name) {
  const auto& ids = runtime_cbids();
  // The runtime records cudaLaunchKernelEx under its own name; CUPTI reports
  // it as the C entry point it goes through.
  const auto it = ids.find(name == "cudaLaunchKernelEx" ? "cudaLaunchKernelExC" : name);
  return it == ids.end() ? static_cast<CUpti_CallbackId>(CUPTI_RUNTIME_TRACE_CBID_INVALID) : it->second;
}

/* ---- the driver functions whose calls are recorded ----
   The same, from cupti_driver_cbid.h. A driver function's ID is its own
   spelling's (cuMemAlloc_v2 and cuMemAlloc have one each), and that spelling is
   also the name a subscriber is told. */

#include "cupti_driver_have.h"

struct DriverCbidRow {
  const char* fn;
  CUpti_CallbackId id;
};
const DriverCbidRow kDriverCbids[] = {
#define VGPU_DRIVER_CBID_ROW(fn, id) {fn, id},
#include "cupti_driver_cbids.inc"
#undef VGPU_DRIVER_CBID_ROW
};

CUpti_CallbackId driver_cbid(const std::string& name) {
  static const std::unordered_map<std::string, CUpti_CallbackId> ids = [] {
    std::unordered_map<std::string, CUpti_CallbackId> m;
    for (const auto& r : kDriverCbids) m[r.fn] = r.id;
    return m;
  }();
  const auto it = ids.find(name);
  return it == ids.end() ? static_cast<CUpti_CallbackId>(CUPTI_DRIVER_TRACE_CBID_INVALID) : it->second;
}

const char* driver_cbid_name(CUpti_CallbackId id) {
  for (const auto& r : kDriverCbids)
    if (r.id == id) return r.fn;
  return nullptr;
}

size_t fill_api(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* a = reinterpret_cast<CUpti_ActivityAPI*>(out);
  std::memset(a, 0, sizeof *a);
  a->kind = e.domain_driver ? CUPTI_ACTIVITY_KIND_DRIVER : CUPTI_ACTIVITY_KIND_RUNTIME;
  a->cbid = e.domain_driver ? driver_cbid(e.name) : runtime_cbid(e.name);
  a->start = e.start_ns;
  a->end = e.end_ns;
  a->processId = e.process_id;
  a->threadId = e.thread_id;
  a->correlationId = e.correlation;
  a->returnValue = static_cast<uint32_t>(e.result);
  return sizeof *a;
}

size_t kind_size(CUpti_ActivityKind k) {
  switch (static_cast<int>(k)) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: return sizeof(MemcpyRecord);
    case CUPTI_ACTIVITY_KIND_MEMSET: return sizeof(MemsetRecord);
    case CUPTI_ACTIVITY_KIND_RUNTIME:
    case CUPTI_ACTIVITY_KIND_DRIVER: return sizeof(CUpti_ActivityAPI);
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: return sizeof(SyncRecord);
    case CUPTI_ACTIVITY_KIND_DEVICE: return sizeof(DeviceRecord);
    case CUPTI_ACTIVITY_KIND_CONTEXT: return sizeof(ContextRecord);
    case CUPTI_ACTIVITY_KIND_STREAM: return sizeof(CUpti_ActivityStream);
    case CUPTI_ACTIVITY_KIND_MARKER: return sizeof(CUpti_ActivityMarker2);
    case CUPTI_ACTIVITY_KIND_MARKER_DATA: return sizeof(CUpti_ActivityMarkerData);
    case CUPTI_ACTIVITY_KIND_NAME: return sizeof(CUpti_ActivityName);
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: return sizeof(CUpti_ActivityExternalCorrelation);
    case akind::kMemory: return sizeof(MemoryV1Record);
    case akind::kMemory2: return sizeof(Memory2Record);
    case akind::kMemoryPool: return sizeof(MemoryPoolRecord);
    case akind::kGraphTrace: return sizeof(GraphTraceRecord);
    case akind::kMemcpy2: return sizeof(PeerCopyRecord);
    case akind::kCudaEvent: return sizeof(CudaEventRecord);
    case akind::kFunction: return sizeof(FunctionRecord);
    case akind::kOverhead: return sizeof(OverheadRecord);
    default: return sizeof(KernelRecord);
  }
}

size_t record_size(const vgpu::profiling::Event& e) {
  using K = vgpu::profiling::EventKind;
  switch (e.kind) {
    case K::Kernel: return sizeof(KernelRecord);
    case K::Api: return sizeof(CUpti_ActivityAPI);
    case K::Memset: return sizeof(MemsetRecord);
    case K::Sync: return sizeof(SyncRecord);
    case K::Stream: return sizeof(CUpti_ActivityStream);
    case K::Context: return sizeof(ContextRecord);
    case K::Device: return sizeof(DeviceRecord);
    case K::Marker: return sizeof(CUpti_ActivityMarker2);
    case K::MarkerData: return sizeof(CUpti_ActivityMarkerData);
    case K::Name: return sizeof(CUpti_ActivityName);
    case K::ExternalCorrelation: return sizeof(CUpti_ActivityExternalCorrelation);
    case K::Memory: return sizeof(Memory2Record);
    case K::MemoryV1: return sizeof(MemoryV1Record);
    case K::MemoryPool: return sizeof(MemoryPoolRecord);
    case K::GraphTrace: return sizeof(GraphTraceRecord);
    case K::Memcpy2: return sizeof(PeerCopyRecord);
    case K::CudaEvent: return sizeof(CudaEventRecord);
    case K::Function: return sizeof(FunctionRecord);
    case K::Overhead: return sizeof(OverheadRecord);
    default: return sizeof(MemcpyRecord);
  }
}

size_t fill(uint8_t* out, const vgpu::profiling::Event& e) {
  using K = vgpu::profiling::EventKind;
  switch (e.kind) {
    case K::Kernel: return fill_kernel(out, e);
    case K::Api: return fill_api(out, e);
    case K::Memset: return fill_memset(out, e);
    case K::Sync: return fill_sync(out, e);
    case K::Stream: return fill_stream(out, e);
    case K::Context: return fill_context(out, e);
    case K::Device: return fill_device(out, e);
    case K::Marker: return fill_marker(out, e);
    case K::MarkerData: return fill_marker_data(out, e);
    case K::Name: return fill_name(out, e);
    case K::ExternalCorrelation: return fill_external(out, e);
    case K::Memory: return fill_memory2(out, e);
    case K::MemoryV1: return fill_memory_v1(out, e);
    case K::MemoryPool: return fill_memory_pool(out, e);
    case K::GraphTrace: return fill_graph_trace(out, e);
    case K::Memcpy2: return fill_peer_copy(out, e);
    case K::CudaEvent: return fill_cuda_event(out, e);
    case K::Function: return fill_function(out, e);
    case K::Overhead: return fill_overhead(out, e);
    default: return fill_memcpy(out, e);
  }
}

// Whether an event of the program is wanted by what is enabled. Memory events
// are, if either memory kind is (the older kind is made from them).
bool kind_wanted(const vgpu::profiling::Event& e) {
  using K = vgpu::profiling::EventKind;
  switch (e.kind) {
    case K::Kernel: return g_kinds[CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL] || g_kinds[CUPTI_ACTIVITY_KIND_KERNEL];
    case K::Api: {
      if (e.domain_driver) {
        const CUpti_CallbackId id = driver_cbid(e.name);
        return id != CUPTI_DRIVER_TRACE_CBID_INVALID && g_driver_filter.wanted(id, g_api_active);
      }
      const CUpti_CallbackId id = runtime_cbid(e.name);
      return id != CUPTI_RUNTIME_TRACE_CBID_INVALID && g_runtime_filter.wanted(id, g_api_active);
    }
    case K::Memset: return g_kinds[CUPTI_ACTIVITY_KIND_MEMSET];
    case K::Sync: return g_kinds[CUPTI_ACTIVITY_KIND_SYNCHRONIZATION];
    case K::Stream: return g_kinds[CUPTI_ACTIVITY_KIND_STREAM];
    case K::Context: return g_kinds[CUPTI_ACTIVITY_KIND_CONTEXT];
    case K::Device: return g_kinds[CUPTI_ACTIVITY_KIND_DEVICE];
    case K::Marker: return g_kinds[CUPTI_ACTIVITY_KIND_MARKER];
    case K::MarkerData: return g_kinds[CUPTI_ACTIVITY_KIND_MARKER_DATA];
    case K::Name: return g_kinds[CUPTI_ACTIVITY_KIND_NAME];
    case K::ExternalCorrelation: return g_kinds[CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION];
    case K::Memory: return g_kinds[akind::kMemory2] || g_kinds[akind::kMemory];
    case K::MemoryPool: return g_kinds[akind::kMemoryPool];
    case K::GraphTrace: return g_kinds[akind::kGraphTrace];
    case K::Memcpy2: return g_kinds[akind::kMemcpy2];
    case K::CudaEvent: return g_kinds[akind::kCudaEvent];
    case K::Function: return g_kinds[akind::kFunction];
    case K::Module: return false;
    case K::Overhead: return g_kinds[akind::kOverhead];   // the loading of a function, told by the runtime
    case K::MemoryV1: return false;                        // made here, never queued by the runtime
    default: return g_kinds[CUPTI_ACTIVITY_KIND_MEMCPY];
  }
}

// The buffer CUPTI asks the program for when the first record of a batch is
// made, not when the batch is delivered (measured on an RTX 3060: with a
// kernel kind on, the request comes as the launch returns, with the runtime
// kind on as the first call returns, and none at all while nothing is recorded;
// after a flush completes the buffer the next record asks again). Profilers
// that count the buffers handed out and flush only if there are any -- Kineto,
// under torch.profiler -- depend on it. The buffer is kept here until a flush
// fills and completes it.
struct HeldBuffer {
  uint8_t* data = nullptr;
  size_t size = 0;
  size_t max_records = 0;
  uint64_t request_start = 0, request_end = 0;
  bool held = false;       // a buffer is held, or being asked for
  bool ready = false;      // the request returned
};
std::mutex g_held_mu;      // before g_mu; held across the program's request callback
HeldBuffer g_held;

// Whether a record of the event would be delivered by the settings now. Under g_mu.
bool would_deliver_locked(const vgpu::profiling::Event& e) {
  using K = vgpu::profiling::EventKind;
  if (g_kinds[akind::kGraphTrace] && e.graph_id != 0 &&
      (e.kind == K::Kernel || e.kind == K::Memcpy || e.kind == K::Memset || e.kind == K::Memcpy2))
    return false;
  if (e.kind == K::Context && g_kinds[CUPTI_ACTIVITY_KIND_STREAM]) return true;
  return kind_wanted(e);
}

void on_event_recorded(const vgpu::profiling::Event& e) {
  // The program's request callback may call the runtime, which records.
  thread_local bool inside = false;
  if (inside) return;
  struct Guard {
    Guard() { inside = true; }
    ~Guard() { inside = false; }
  } guard;
  std::lock_guard<std::mutex> held_lock(g_held_mu);
  if (g_held.held) return;
  CUpti_BuffersCallbackRequestFunc request = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_request || !would_deliver_locked(e)) return;
    request = g_request;
  }
  g_held.held = true;
  g_held.request_start = vgpu::profiling::host_ns();
  uint8_t* data = nullptr;
  size_t size = 0, max_records = 0;
  request(&data, &size, &max_records);
  g_held.request_end = vgpu::profiling::host_ns();
  if (!data || size == 0) {   // the program gave none: ask again at the flush
    g_held = HeldBuffer{};
    return;
  }
  g_held.data = data;
  g_held.size = size;
  g_held.max_records = max_records;
  g_held.ready = true;
}

}  // namespace

/* ---- version and errors ---- */

VGPU_EXPORT CUptiResult cuptiGetVersion(uint32_t* version) {
  if (!version) return CUPTI_ERROR_INVALID_PARAMETER;
  *version = CUPTI_API_VERSION;
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiGetResultString(CUptiResult result, const char** str) {
  if (!str) return CUPTI_ERROR_INVALID_PARAMETER;
  // Every code CUPTI 13.0 names, in order (36 and 37 are gone from it, and a
  // number it does not know is answered as "<unknown>" and an invalid
  // parameter, as it does).
  static const char* const kNames[] = {
      "CUPTI_SUCCESS", "CUPTI_ERROR_INVALID_PARAMETER", "CUPTI_ERROR_INVALID_DEVICE",
      "CUPTI_ERROR_INVALID_CONTEXT", "CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID", "CUPTI_ERROR_INVALID_EVENT_ID",
      "CUPTI_ERROR_INVALID_EVENT_NAME", "CUPTI_ERROR_INVALID_OPERATION", "CUPTI_ERROR_OUT_OF_MEMORY",
      "CUPTI_ERROR_HARDWARE", "CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT", "CUPTI_ERROR_API_NOT_IMPLEMENTED",
      "CUPTI_ERROR_MAX_LIMIT_REACHED", "CUPTI_ERROR_NOT_READY", "CUPTI_ERROR_NOT_COMPATIBLE",
      "CUPTI_ERROR_NOT_INITIALIZED", "CUPTI_ERROR_INVALID_METRIC_ID", "CUPTI_ERROR_INVALID_METRIC_NAME",
      "CUPTI_ERROR_QUEUE_EMPTY", "CUPTI_ERROR_INVALID_HANDLE", "CUPTI_ERROR_INVALID_STREAM",
      "CUPTI_ERROR_INVALID_KIND", "CUPTI_ERROR_INVALID_EVENT_VALUE", "CUPTI_ERROR_DISABLED",
      "CUPTI_ERROR_INVALID_MODULE", "CUPTI_ERROR_INVALID_METRIC_VALUE", "CUPTI_ERROR_HARDWARE_BUSY",
      "CUPTI_ERROR_NOT_SUPPORTED", "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED",
      "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE",
      "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES",
      "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS", "CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED",
      "CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED", "CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE",
      "CUPTI_ERROR_INSUFFICIENT_PRIVILEGES", nullptr, nullptr, "CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED",
      "CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED", "CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES",
      "CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED", "CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED",
      "CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED", "CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED",
      "CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED", "CUPTI_ERROR_INVALID_CHIP_NAME"};
  const int code = static_cast<int>(result);
  if (code >= 0 && code < static_cast<int>(sizeof kNames / sizeof kNames[0]) && kNames[code]) {
    *str = kNames[code];
    return CUPTI_SUCCESS;
  }
  if (code == 999) {
    *str = "CUPTI_ERROR_UNKNOWN";
    return CUPTI_SUCCESS;
  }
  if (code == 1000) {
    *str = "CUPTI_ERROR_CANT_OPEN_FILE";
    return CUPTI_SUCCESS;
  }
  *str = "<unknown>";
  return CUPTI_ERROR_INVALID_PARAMETER;
}

VGPU_EXPORT CUptiResult cuptiGetLastError(void) { return CUPTI_SUCCESS; }

VGPU_EXPORT CUptiResult cuptiGetTimestamp(uint64_t* timestamp) {
  if (!timestamp) return CUPTI_ERROR_INVALID_PARAMETER;
  *timestamp = vgpu::profiling::host_ns();
  return CUPTI_SUCCESS;
}

/* ---- activity ---- */

VGPU_EXPORT CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
  if (!funcBufferRequested || !funcBufferCompleted) return CUPTI_ERROR_INVALID_PARAMETER;
  vgpu::profiling::set_record_hook(&on_event_recorded);
  std::lock_guard<std::mutex> lock(g_mu);
  g_request = funcBufferRequested;
  g_complete = funcBufferCompleted;
  return CUPTI_SUCCESS;
}

namespace {

// What enabling a kind answers, as measured on an RTX 3060 (compute capability
// 8.6, so none of the legacy profiler's kinds): the counter-based kinds are
// refused as the legacy profiler's, a few others as not compatible with this
// device or toolkit, and two are not enableable by themselves. The rest are
// accepted, whether or not anything is produced for them.
CUptiResult enable_result(int k) {
  if (k <= 0 || k >= akind::kCount) return CUPTI_ERROR_NOT_COMPATIBLE;
  switch (k) {
    case akind::kEvent: case akind::kMetric: case akind::kSourceLocator: case akind::kGlobalAccess:
    case akind::kBranch: case akind::kEventInstance: case akind::kMetricInstance:
    case akind::kInstructionExecution: case akind::kSharedAccess: case akind::kPcSampling:
    case akind::kPcSamplingRecordInfo: case akind::kInstructionCorrelation:
    case akind::kInstantaneousEvent: case akind::kInstantaneousEventInstance:
    case akind::kInstantaneousMetric: case akind::kInstantaneousMetricInstance:
      return CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED;
    case akind::kCdpKernel: case akind::kEnvironment: case akind::kDeviceAttribute:
      return CUPTI_ERROR_NOT_COMPATIBLE;
    case akind::kModule: case akind::kDeviceGraphTrace:
      return CUPTI_ERROR_INVALID_KIND;
    // Unified Memory counters: on this machine (a WSL2 driver, where managed
    // memory is not paged on demand) NVIDIA's answers "not ready" to enabling
    // and to disabling the kind, though configuring the counters is accepted.
    // This has no page-fault model to count from, so it says the same
    // everywhere rather than enable a kind that produces nothing.
    case akind::kUnifiedMemoryCounter:
      return CUPTI_ERROR_NOT_READY;
    default:
      return CUPTI_SUCCESS;
  }
}

void refresh_recording() {
  vgpu::profiling::set_enabled(anything_recorded());
}

}  // namespace

namespace {
// Events already judged by the settings that were in force when they happened.
std::vector<vgpu::profiling::Event> g_sealed;   // under g_mu
void seal_locked();

// What was switched on when the first context was made decides whether API
// calls are traced (see g_api_active).
void on_first_context() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_runtime_filter.any_on() || g_driver_filter.any_on()) g_api_active = true;
}
}  // namespace

VGPU_EXPORT CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
  vgpu::profiling::set_context_hook(&on_first_context);
  const int k = static_cast<int>(kind);
  CUPTI_TRACE("cuptiActivityEnable(%d)", k);
  std::lock_guard<std::mutex> lock(g_mu);
  if (const CUptiResult r = enable_result(k); r != CUPTI_SUCCESS) {
    CUPTI_TRACE("  refused: %d", static_cast<int>(r));
    return r;
  }
  // The two kernel kinds are one or the other.
  if ((k == CUPTI_ACTIVITY_KIND_KERNEL && g_kinds[CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL]) ||
      (k == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && g_kinds[CUPTI_ACTIVITY_KIND_KERNEL]))
    return CUPTI_ERROR_NOT_COMPATIBLE;
  seal_locked();
  g_kinds[static_cast<size_t>(k)] = true;
  if (k == CUPTI_ACTIVITY_KIND_RUNTIME) g_runtime_filter = ApiFilter{true, false, {}};
  if (k == CUPTI_ACTIVITY_KIND_DRIVER) g_driver_filter = ApiFilter{true, false, {}};
  if ((k == CUPTI_ACTIVITY_KIND_RUNTIME || k == CUPTI_ACTIVITY_KIND_DRIVER) && vgpu::profiling::context_made())
    g_api_active = true;
  if (k == CUPTI_ACTIVITY_KIND_DEVICE) g_devices_delivered = false;
  if (produced(k)) announce_once();
  refresh_recording();
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
  const int k = static_cast<int>(kind);
  CUPTI_TRACE("cuptiActivityDisable(%d)", k);
  std::lock_guard<std::mutex> lock(g_mu);
  // Disabling a kind that is not enabled, or does not exist, is not an error.
  if (k <= 0 || k >= static_cast<int>(g_kinds.size())) return CUPTI_SUCCESS;
  if (k == akind::kUnifiedMemoryCounter) return CUPTI_ERROR_NOT_READY;   // see enable_result
  seal_locked();
  g_kinds[static_cast<size_t>(k)] = false;
  // Switched off while API calls are traced, a kind leaves the function
  // switches inert until it is switched on again; before that, it just goes.
  if (k == CUPTI_ACTIVITY_KIND_RUNTIME) g_runtime_filter = ApiFilter{false, g_api_active, {}};
  if (k == CUPTI_ACTIVITY_KIND_DRIVER) g_driver_filter = ApiFilter{false, g_api_active, {}};
  refresh_recording();
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiActivityEnableContext(CUcontext, CUpti_ActivityKind kind) {
  return cuptiActivityEnable(kind);
}
VGPU_EXPORT CUptiResult cuptiActivityDisableContext(CUcontext, CUpti_ActivityKind kind) {
  return cuptiActivityDisable(kind);
}

// Records of single API functions, by the function's callback id. The id 0 is
// accepted and does nothing; an id the toolkit does not name is a bad
// parameter. See ApiFilter for what the two settings do together.
VGPU_EXPORT CUptiResult cuptiActivityEnableRuntimeApi(CUpti_CallbackId cbid, uint8_t enable) {
  vgpu::profiling::set_context_hook(&on_first_context);
  CUPTI_TRACE("cuptiActivityEnableRuntimeApi(%u, %d)", static_cast<unsigned>(cbid), static_cast<int>(enable));
  if (cbid == 0) return CUPTI_SUCCESS;
  if (!runtime_cbid_name(cbid)) return CUPTI_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mu);
  seal_locked();
  g_runtime_filter.per_function[cbid] = enable != 0;
  if (enable) announce_once();
  refresh_recording();
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiActivityEnableDriverApi(CUpti_CallbackId cbid, uint8_t enable) {
  vgpu::profiling::set_context_hook(&on_first_context);
  CUPTI_TRACE("cuptiActivityEnableDriverApi(%u, %d)", static_cast<unsigned>(cbid), static_cast<int>(enable));
  if (cbid == 0) return CUPTI_SUCCESS;
  if (!driver_cbid_name(cbid)) return CUPTI_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mu);
  seal_locked();
  g_driver_filter.per_function[cbid] = enable != 0;
  if (enable) announce_once();
  refresh_recording();
  return CUPTI_SUCCESS;
}

// A clock of the tool's own for the records' times (cuptiActivityRegisterTimestampCallback).
// API calls, waits and markers read it as they happen; work on the device is
// stamped by the engine's clock and put on this one when the records are made,
// by linear interpolation between a reading taken when the clock was registered
// and one taken at the flush -- NVIDIA's does the same with its device clock,
// reading the function while it processes the buffer, and reads it at the start
// and end of each API call.
std::atomic<CUpti_TimestampCallbackFunc> g_user_clock{nullptr};
// A reading of the program's clock and of the engine's taken together when the
// clock was registered: the start of the interval device work is put on the
// program's clock across (see expand).
uint64_t g_clock_anchor_user = 0, g_clock_anchor_engine = 0;
uint64_t user_clock_trampoline() {
  const CUpti_TimestampCallbackFunc f = g_user_clock.load();
  return f ? f() : vgpu::profiling::now_ns();
}

VGPU_EXPORT CUptiResult cuptiActivityRegisterTimestampCallback(CUpti_TimestampCallbackFunc funcTimestamp) {
  if (!funcTimestamp) return CUPTI_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mu);
  g_user_clock.store(funcTimestamp);
  vgpu::profiling::set_host_clock(&user_clock_trampoline);
  g_clock_anchor_engine = vgpu::profiling::now_ns();
  g_clock_anchor_user = funcTimestamp();
  return CUPTI_SUCCESS;
}

// A CUDA event's record carries a device timestamp when asked to.
VGPU_EXPORT CUptiResult cuptiActivityEnableCudaEventDeviceTimestamps(uint8_t enable) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_cuda_event_device_timestamps.store(enable != 0);
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiActivityGetNextRecord(uint8_t* buffer, size_t validBufferSizeBytes,
                                                   CUpti_Activity** record) {
  if (!buffer || !record) return CUPTI_ERROR_INVALID_PARAMETER;
  // The walk is by each record's own size, which is why the sizes written into
  // the buffer and the ones read back here have to agree exactly.
  size_t offset = 0;
  if (*record) {
    const auto* cur = *record;
    offset = reinterpret_cast<const uint8_t*>(cur) - buffer;
    offset += kind_size(cur->kind);
  }
  if (offset >= validBufferSizeBytes) return CUPTI_ERROR_MAX_LIMIT_REACHED;
  *record = reinterpret_cast<CUpti_Activity*>(buffer + offset);
  return CUPTI_SUCCESS;
}

namespace {

// Allocations seen and not yet reported by the older memory kind, which lists
// each one once: when it is freed, or at the first flush after it was made if
// it still lives then.
struct LiveAlloc {
  uint64_t start = 0, bytes = 0;
  vgpu::profiling::MemKind kind = vgpu::profiling::MemKind::Unknown;
  uint32_t device = 0, process_id = 0;
  bool reported = false;
};
std::map<uint64_t, LiveAlloc> g_live_allocs;   // under g_mu

vgpu::profiling::Event memory_v1_event(uint64_t address, const LiveAlloc& a, uint64_t end) {
  vgpu::profiling::Event v;
  v.kind = vgpu::profiling::EventKind::MemoryV1;
  v.address = address;
  v.bytes = a.bytes;
  v.src_kind = a.kind;
  v.device = a.device;
  v.process_id = a.process_id;
  v.start_ns = a.start;
  v.end_ns = end;
  return v;
}

bool device_side(vgpu::profiling::EventKind k) {
  using K = vgpu::profiling::EventKind;
  return k == K::Kernel || k == K::Memcpy || k == K::Memset || k == K::Memcpy2 || k == K::GraphTrace;
}

// The events a flush turns into records, in the order they happened: those the
// enabled kinds want, with the extras CUPTI makes from them -- the driver's
// own streams beside a context, the older memory kind beside the newer. Under
// g_mu.
void expand(std::vector<vgpu::profiling::Event>& events, std::vector<vgpu::profiling::Event>& out) {
  using K = vgpu::profiling::EventKind;
  // The driver makes eight streams of its own in each context, which CUPTI lists
  // beside it: ids and flags are an RTX 3060's (a default stream, the null
  // stream, five non-blocking ones and one of the highest priority).
  static const struct { uint32_t flag; int32_t priority; } kInternal[8] = {
      {CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT, 0},      {CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL, 0},
      {CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING, 0}, {CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING, 0},
      {CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING, 0}, {CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING, 0},
      {CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING, 0}, {CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT, -5}};
  const bool graph_trace = g_kinds[akind::kGraphTrace];
  for (auto& e : events) {
    if (e.kind == K::Context) {
      uint32_t ctx = 0;
      const uint32_t first = note_context(e.device, e.handle, e.sync_id, &ctx);
      if (first && g_kinds[CUPTI_ACTIVITY_KIND_STREAM]) {
        for (uint32_t i = 0; i < 8; ++i) {
          vgpu::profiling::Event st;
          st.kind = K::Stream;
          st.op = 1;
          st.device = e.device;
          st.correlation = e.correlation;
          st.start_ns = st.end_ns = e.start_ns;
          st.context_id = ctx;
          st.stream_id = first + i;
          st.flags = kInternal[i].flag;
          st.priority = kInternal[i].priority;
          out.push_back(std::move(st));
        }
      }
      e.context_id = ctx;
      e.stream_id = first ? first + 1 : 0;   // the null stream is the second of the eight
    }
    if (e.kind == K::Memory) {
      // Track the allocation whether or not the newer kind is wanted: the older one is made from it.
      if (e.op == 1) {
        LiveAlloc a;
        a.start = e.start_ns;
        a.bytes = e.bytes;
        a.kind = e.src_kind;
        a.device = e.device;
        a.process_id = e.process_id;
        g_live_allocs[e.address] = a;
      } else if (e.op == 2) {
        const auto it = g_live_allocs.find(e.address);
        if (it != g_live_allocs.end()) {
          if (!it->second.reported && g_kinds[akind::kMemory]) out.push_back(memory_v1_event(e.address, it->second, e.start_ns));
          g_live_allocs.erase(it);
        }
      }
      if (g_kinds[akind::kMemory2]) out.push_back(std::move(e));
      continue;
    }
    if (graph_trace && e.graph_id != 0 &&
        (e.kind == K::Kernel || e.kind == K::Memcpy || e.kind == K::Memset || e.kind == K::Memcpy2))
      continue;   // a launched graph is one record while its trace is wanted
    if (!kind_wanted(e)) continue;
    out.push_back(std::move(e));
  }
}

// The events recorded so far, judged by the settings as they are now. Called
// before a setting changes, so an event is judged by what was enabled when it
// happened. Under g_mu.
void seal_locked() {
  std::vector<vgpu::profiling::Event> events = vgpu::profiling::drain();
  if (!events.empty()) expand(events, g_sealed);
}

// Work on the device, stamped by the engine's clock, put on the program's
// by linear interpolation between the reading taken when its clock was
// registered and one taken now.
void on_program_clock(std::vector<vgpu::profiling::Event>& events, uint64_t engine_now, uint64_t user_now) {
  using K = vgpu::profiling::EventKind;
  const auto map = [&](uint64_t t) -> uint64_t {
    if (user_now <= g_clock_anchor_user || engine_now <= g_clock_anchor_engine || t <= g_clock_anchor_engine)
      return g_clock_anchor_user;
    const unsigned __int128 span_user = user_now - g_clock_anchor_user;
    const unsigned __int128 span_engine = engine_now - g_clock_anchor_engine;
    const unsigned __int128 at = std::min<uint64_t>(t, engine_now) - g_clock_anchor_engine;
    return g_clock_anchor_user + static_cast<uint64_t>(at * span_user / span_engine);
  };
  for (auto& e : events) {
    if (device_side(e.kind)) {
      e.start_ns = map(e.start_ns);
      e.end_ns = map(e.end_ns);
    }
    if (e.kind == K::CudaEvent) e.start_ns = map(e.start_ns);
  }
}

}  // namespace

VGPU_EXPORT CUptiResult cuptiActivityFlushAll(uint32_t flags) {
  CUPTI_TRACE("cuptiActivityFlushAll(%u)", static_cast<unsigned>(flags));
  CUpti_BuffersCallbackRequestFunc request = nullptr;
  CUpti_BuffersCallbackCompleteFunc complete = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    request = g_request;
    complete = g_complete;
  }
  if (!request || !complete) {
    CUPTI_TRACE("  no buffer callbacks registered");
    return CUPTI_ERROR_NOT_INITIALIZED;
  }

  const uint64_t flush_start = vgpu::profiling::host_ns();
  std::vector<vgpu::profiling::Event> events = vgpu::profiling::drain();
  std::vector<vgpu::profiling::Event> wanted;
  bool devices = false, overhead = false;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_kinds[CUPTI_ACTIVITY_KIND_DEVICE] && !g_devices_delivered) devices = g_devices_delivered = true;
    overhead = g_kinds[akind::kOverhead];
    // A tool's own clock: work on the device was stamped by the engine's, and
    // is put on the tool's by one pair of readings taken now.
    wanted = std::move(g_sealed);
    g_sealed.clear();
    expand(events, wanted);
    if (g_user_clock.load() != nullptr) on_program_clock(wanted, vgpu::profiling::now_ns(), flush_start);
    // Allocations still alive, listed once by the older kind.
    if (g_kinds[akind::kMemory])
      for (auto& kv : g_live_allocs)
        if (!kv.second.reported) {
          kv.second.reported = true;
          wanted.push_back(memory_v1_event(kv.first, kv.second, flush_start));
        }
  }
  // Every device, once, the first time records are asked for after the kind
  // was enabled: a device is a fact about the machine, not an event in it.
  if (devices) {
    int count = 0;
    {
      vgpu::profiling::Silence silent;
      if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
    }
    std::vector<vgpu::profiling::Event> listed;
    for (int d = 0; d < count; ++d) {
      vgpu::profiling::Event e;
      e.kind = vgpu::profiling::EventKind::Device;
      e.device = static_cast<uint32_t>(d);
      listed.push_back(std::move(e));
    }
    wanted.insert(wanted.begin(), std::make_move_iterator(listed.begin()),
                  std::make_move_iterator(listed.end()));
  }
  // A buffer asked for when the first record was made, if there is one.
  HeldBuffer held;
  {
    std::lock_guard<std::mutex> lock(g_held_mu);
    if (g_held.ready) {
      held = g_held;
      g_held = HeldBuffer{};
    }
  }
  // Nothing to deliver, no buffer: the cost of a flush is not a record of its own.
  CUPTI_TRACE("  %zu records to deliver", wanted.size());
  if (wanted.empty()) {
    if (held.held) complete(nullptr, 0, held.data, 0, 0);   // asked for, and nothing came of it
    return CUPTI_SUCCESS;
  }
  const bool delivering_work = std::any_of(wanted.begin(), wanted.end(), [](const vgpu::profiling::Event& e) {
    return e.kind != vgpu::profiling::EventKind::Overhead && e.kind != vgpu::profiling::EventKind::Device;
  });

  size_t i = 0;
  bool first_buffer = true;
  while (i < wanted.size()) {
    uint8_t* buffer = nullptr;
    size_t size = 0;
    size_t max_records = 0;
    uint64_t request_start = 0, request_end = 0;
    const bool from_record = first_buffer && held.held;
    if (from_record) {
      buffer = held.data;
      size = held.size;
      max_records = held.max_records;
      request_start = held.request_start;
      request_end = held.request_end;
    } else {
      request_start = vgpu::profiling::host_ns();
      request(&buffer, &size, &max_records);
      request_end = vgpu::profiling::host_ns();
    }
    if (!buffer || size == 0) return CUPTI_ERROR_MAX_LIMIT_REACHED;
    if (first_buffer) {
      first_buffer = false;
      // What the profiler itself cost, measured: the flush up to the request
      // (when it delivers records of other kinds), and the consumer's
      // buffer-request callback just made. Told after the records they are the
      // cost of.
      if (overhead) {
        vgpu::profiling::Event f, r;
        f.kind = r.kind = vgpu::profiling::EventKind::Overhead;
        f.process_id = r.process_id = static_cast<uint32_t>(::getpid());
        f.thread_id = r.thread_id = static_cast<uint32_t>(::syscall(SYS_gettid));
        f.flags = uint32_t{1} << 16;   // CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH
        f.start_ns = flush_start;
        f.end_ns = request_start;
        r.flags = uint32_t{7} << 16;   // CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST
        r.start_ns = request_start;
        r.end_ns = request_end;
        // The flush's own cost is the time until it asked for a buffer; a
        // buffer asked for earlier leaves none.
        if (delivering_work && !from_record) wanted.push_back(std::move(f));
        wanted.push_back(std::move(r));
      }
    }

    size_t used = 0;
    size_t written = 0;
    while (i < wanted.size()) {
      const size_t need = record_size(wanted[i]);
      if (used + need > size) break;
      if (max_records && written >= max_records) break;
      used += fill(buffer + used, wanted[i]);
      ++written;
      ++i;
    }
    // A buffer too small for even one record would loop for ever; hand it back
    // empty and stop rather than spin.
    complete(nullptr, 0, buffer, used, used);
    if (written == 0) return CUPTI_ERROR_MAX_LIMIT_REACHED;
  }
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiActivityFlush(CUcontext, uint32_t, uint32_t flag) {
  return cuptiActivityFlushAll(flag);
}

VGPU_EXPORT CUptiResult cuptiActivityFlushPeriod(uint32_t) { return CUPTI_SUCCESS; }

VGPU_EXPORT CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext, uint32_t, size_t* dropped) {
  if (!dropped) return CUPTI_ERROR_INVALID_PARAMETER;
  *dropped = 0;
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiFinalize(void) {
  std::lock_guard<std::mutex> held_lock(g_held_mu);
  g_held = HeldBuffer{};
  std::lock_guard<std::mutex> lock(g_mu);
  vgpu::profiling::set_enabled(false);
  vgpu::profiling::set_host_clock(nullptr);
  g_user_clock.store(nullptr);
  g_request = nullptr;
  g_complete = nullptr;
  std::fill(g_kinds.begin(), g_kinds.end(), false);
  g_runtime_filter = ApiFilter{};
  g_driver_filter = ApiFilter{};
  g_cuda_event_device_timestamps.store(false);
  g_live_allocs.clear();
  g_sealed.clear();
  vgpu::profiling::set_record_hook(nullptr);
  return CUPTI_SUCCESS;
}

/* ---- callback API ----
   A subscriber (there is one at a time, as in NVIDIA's) is called as each
   runtime-API call is entered and as it returns, with the arguments the
   program passed and the value the call returned. The runtime reports each
   call it serves (vgpu/profiling.hpp); this turns the arguments into the
   toolkit's own parameter structures, which is what a consumer casts
   functionParams to.

   Only the calls this can describe completely are delivered. A callback for a
   function whose parameter structure is not filled would hand the consumer a
   structure of zeros as though they were what the program passed, and a
   consumer that reads it would be wrong rather than missing something. The
   activity records carry every call; callbacks carry the ones below.

   The driver API is delivered the same way, for the calls a program makes
   itself and whose parameters are filled completely (below). Resource and
   synchronize domains: a stream made or destroyed, a context created or
   destroyed, a stream or context waited on. Module and graph resources are not
   delivered. */

namespace {

struct Conv {
  CUpti_CallbackId cbid;
  void (*fill)(void* storage, const void* const* a);   // null: the call has no parameters
  int nargs = -1;   // how many arguments the structure is made from, where that is checked
  // Whether the arguments the shim declares are the sizes of the fields the
  // toolkit names (null: not checked, for the rows made by hand against a card).
  bool (*sizes_ok)(const uint16_t* sizes) = nullptr;
};

template <class T>
T arg(const void* const* a, int i) {
  return *static_cast<const T*>(a[i]);
}
#define P(i, field) p->field = arg<std::remove_reference_t<decltype(p->field)>>(a, i)
#define CONV(fn, ver, body)                                                                      \
  {#fn, Conv{CUPTI_RUNTIME_TRACE_CBID_##fn##_##ver, [](void* storage, const void* const* a) {   \
               auto* p = new (storage) fn##_##ver##_params;                                      \
               body                                                                              \
             }}},
#define CONV0(fn, ver) {#fn, Conv{CUPTI_RUNTIME_TRACE_CBID_##fn##_##ver, nullptr}},

const std::unordered_map<std::string, Conv>& conversions() {
  static const std::unordered_map<std::string, Conv> table = {
    CONV(cudaMalloc, v3020, P(0, devPtr); P(1, size);)
    CONV(cudaMallocHost, v3020, P(0, ptr); P(1, size);)
    CONV(cudaHostAlloc, v3020, P(0, pHost); P(1, size); P(2, flags);)
    CONV(cudaMallocManaged, v6000, P(0, devPtr); P(1, size); P(2, flags);)
    CONV(cudaMallocPitch, v3020, P(0, devPtr); P(1, pitch); P(2, width); P(3, height);)
    CONV(cudaFree, v3020, P(0, devPtr);)
    CONV(cudaFreeHost, v3020, P(0, ptr);)
    CONV(cudaMemcpy, v3020, P(0, dst); P(1, src); P(2, count); P(3, kind);)
    CONV(cudaMemcpyAsync, v3020, P(0, dst); P(1, src); P(2, count); P(3, kind); P(4, stream);)
    CONV(cudaMemcpy2D, v3020, P(0, dst); P(1, dpitch); P(2, src); P(3, spitch); P(4, width); P(5, height); P(6, kind);)
    CONV(cudaMemcpy2DAsync, v3020, P(0, dst); P(1, dpitch); P(2, src); P(3, spitch); P(4, width); P(5, height); P(6, kind); P(7, stream);)
    CONV(cudaMemcpyToSymbol, v3020, P(0, symbol); P(1, src); P(2, count); P(3, offset); P(4, kind);)
    CONV(cudaMemcpyToSymbolAsync, v3020, P(0, symbol); P(1, src); P(2, count); P(3, offset); P(4, kind); P(5, stream);)
    CONV(cudaMemcpyFromSymbol, v3020, P(0, dst); P(1, symbol); P(2, count); P(3, offset); P(4, kind);)
    CONV(cudaMemcpyFromSymbolAsync, v3020, P(0, dst); P(1, symbol); P(2, count); P(3, offset); P(4, kind); P(5, stream);)
    CONV(cudaMemset, v3020, P(0, devPtr); P(1, value); P(2, count);)
    CONV(cudaMemsetAsync, v3020, P(0, devPtr); P(1, value); P(2, count); P(3, stream);)
    CONV(cudaMemset2D, v3020, P(0, devPtr); P(1, pitch); P(2, value); P(3, width); P(4, height);)
    CONV(cudaMemset2DAsync, v3020, P(0, devPtr); P(1, pitch); P(2, value); P(3, width); P(4, height); P(5, stream);)
    CONV(cudaMemGetInfo, v3020, P(0, free); P(1, total);)
    CONV(cudaHostRegister, v4000, P(0, ptr); P(1, size); P(2, flags);)
    CONV(cudaHostUnregister, v4000, P(0, ptr);)
    CONV(cudaLaunchKernel, v7000, P(0, func); P(1, gridDim); P(2, blockDim); P(3, args); P(4, sharedMem); P(5, stream);)
    CONV(cudaLaunchCooperativeKernel, v9000, P(0, func); P(1, gridDim); P(2, blockDim); P(3, args); P(4, sharedMem); P(5, stream);)
    CONV(cudaStreamCreate, v3020, P(0, pStream);)
    CONV(cudaStreamCreateWithFlags, v5000, P(0, pStream); P(1, flags);)
    CONV(cudaStreamCreateWithPriority, v5050, P(0, pStream); P(1, flags); P(2, priority);)
    CONV(cudaStreamDestroy, v5050, P(0, stream);)
    CONV(cudaStreamSynchronize, v3020, P(0, stream);)
    CONV(cudaStreamQuery, v3020, P(0, stream);)
    CONV(cudaStreamWaitEvent, v3020, P(0, stream); P(1, event); P(2, flags);)
    CONV(cudaEventCreate, v3020, P(0, event);)
    CONV(cudaEventCreateWithFlags, v3020, P(0, event); P(1, flags);)
    CONV(cudaEventRecord, v3020, P(0, event); P(1, stream);)
    CONV(cudaEventSynchronize, v3020, P(0, event);)
    CONV(cudaEventQuery, v3020, P(0, event);)
    CONV(cudaEventDestroy, v3020, P(0, event);)
    CONV0(cudaDeviceSynchronize, v3020)
    CONV(cudaSetDevice, v3020, P(0, device);)
    CONV(cudaGetDevice, v3020, P(0, device);)
    CONV(cudaGetDeviceCount, v3020, P(0, count);)
    CONV(cudaDeviceGetAttribute, v5000, P(0, value); P(1, attr); P(2, device);)
    CONV0(cudaDeviceReset, v3020)
    CONV(cudaFuncGetAttributes, v3020, P(0, attr); P(1, func);)
    CONV0(cudaGetLastError, v3020)
    CONV0(cudaPeekAtLastError, v3020)
    CONV(cudaDriverGetVersion, v3020, P(0, driverVersion);)
    CONV(cudaRuntimeGetVersion, v3020, P(0, runtimeVersion);)
    CONV(cudaGraphLaunch, v10000, P(0, graphExec); P(1, stream);)
    CONV(cudaStreamBeginCapture, v10000, P(0, stream); P(1, mode);)
    CONV(cudaStreamEndCapture, v10000, P(0, stream); P(1, pGraph);)
    CONV(cudaMemcpyPeer, v4000, P(0, dst); P(1, dstDevice); P(2, src); P(3, srcDevice); P(4, count);)
    CONV(cudaMemcpyPeerAsync, v4000, P(0, dst); P(1, dstDevice); P(2, src); P(3, srcDevice); P(4, count); P(5, stream);)
    CONV(cudaMallocAsync, v11020, P(0, devPtr); P(1, size); P(2, hStream);)
    CONV(cudaFreeAsync, v11020, P(0, devPtr); P(1, hStream);)
  };
  return table;
}
#undef P
#undef CONV
#undef CONV0

/* Every other runtime function whose parameter structure the toolkit defines is
   converted by rows made from the toolkit's own generated_cuda_runtime_api_meta.h
   at configure time (scripts/gen_cupti_runtime_conv.py): each field is the
   argument in the same position, copied by the field's size, and a function whose
   fields are not all plain values (an array, a function pointer) has no row. A
   row also says how many arguments the call has, and a call that arrives with a
   different number is not delivered rather than delivered half-filled. The rows
   made by hand above, which are checked against an RTX 3060, win. */
struct GeneratedConv {
  const char* fn;
  Conv conv;
};
const GeneratedConv kGeneratedConversions[] = {
#include "cupti_runtime_conv.inc"
};

const std::unordered_map<std::string, Conv>& all_conversions() {
  static const std::unordered_map<std::string, Conv> table = [] {
    std::unordered_map<std::string, Conv> t = conversions();
    // The newest version of each function, as with the ids.
    std::unordered_map<std::string, Conv> generated;
    for (const auto& g : kGeneratedConversions) {
      auto it = generated.find(g.fn);
      if (it == generated.end() || g.conv.cbid > it->second.cbid) generated[g.fn] = g.conv;
    }
    for (auto& kv : generated) t.emplace(kv.first, kv.second);
    return t;
  }();
  return table;
}

/* The driver's. Parameter structures are the toolkit's own, named as the
   callback ID is (cuMemAlloc_v2_params). Each argument is copied into its field
   byte for byte, by the field's own size, so an argument kept in a differently
   typed variable (an event as void*) is read the same, and an enum value a
   program passed that the toolkit does not name is never loaded as an enum. */
#define DP(i, field) std::memcpy(&p->field, a[i], sizeof p->field)
#define DCONV(fn, body)                                                                          \
  {#fn, Conv{CUPTI_DRIVER_TRACE_CBID_##fn, [](void* storage, const void* const* a) {            \
               auto* p = new (storage) fn##_params();                                            \
               body                                                                              \
             }}},
#define DCONV0(fn) {#fn, Conv{CUPTI_DRIVER_TRACE_CBID_##fn, nullptr}},

const std::unordered_map<std::string, Conv>& driver_conversions() {
  static const std::unordered_map<std::string, Conv> table = {
    DCONV(cuInit, DP(0, Flags);)
    DCONV(cuDeviceGet, DP(0, device); DP(1, ordinal);)
    DCONV(cuDeviceGetCount, DP(0, count);)
    DCONV(cuDeviceGetName, DP(0, name); DP(1, len); DP(2, dev);)
    DCONV(cuDeviceGetAttribute, DP(0, pi); DP(1, attrib); DP(2, dev);)
    DCONV(cuDeviceTotalMem_v2, DP(0, bytes); DP(1, dev);)
    DCONV(cuDeviceComputeCapability, DP(0, major); DP(1, minor); DP(2, dev);)
    DCONV(cuCtxCreate_v2, DP(0, pctx); DP(1, flags); DP(2, dev);)
    DCONV(cuCtxCreate_v3, DP(0, pctx); DP(1, paramsArray); DP(2, numParams); DP(3, flags); DP(4, dev);)
#ifdef VGPU_HAVE_DRIVER_CBID_cuCtxCreate_v4
    DCONV(cuCtxCreate_v4, DP(0, pctx); DP(1, ctxCreateParams); DP(2, flags); DP(3, dev);)
#endif
    DCONV(cuCtxDestroy_v2, DP(0, ctx);)
    DCONV(cuCtxSetCurrent, DP(0, ctx);)
    DCONV(cuCtxGetCurrent, DP(0, pctx);)
    DCONV0(cuCtxSynchronize)
    DCONV(cuMemAlloc_v2, DP(0, dptr); DP(1, bytesize);)
    DCONV(cuMemFree_v2, DP(0, dptr);)
    DCONV(cuMemAllocHost_v2, DP(0, pp); DP(1, bytesize);)
    DCONV(cuMemFreeHost, DP(0, p);)
    DCONV(cuMemHostAlloc, DP(0, pp); DP(1, bytesize); DP(2, Flags);)
    DCONV(cuMemcpyHtoD_v2, DP(0, dstDevice); DP(1, srcHost); DP(2, ByteCount);)
    DCONV(cuMemcpyDtoH_v2, DP(0, dstHost); DP(1, srcDevice); DP(2, ByteCount);)
    DCONV(cuMemcpyDtoD_v2, DP(0, dstDevice); DP(1, srcDevice); DP(2, ByteCount);)
    DCONV(cuMemcpyHtoDAsync_v2, DP(0, dstDevice); DP(1, srcHost); DP(2, ByteCount); DP(3, hStream);)
    DCONV(cuMemcpyDtoHAsync_v2, DP(0, dstHost); DP(1, srcDevice); DP(2, ByteCount); DP(3, hStream);)
    DCONV(cuMemcpyDtoDAsync_v2, DP(0, dstDevice); DP(1, srcDevice); DP(2, ByteCount); DP(3, hStream);)
    DCONV(cuMemsetD8_v2, DP(0, dstDevice); DP(1, uc); DP(2, N);)
    DCONV(cuMemsetD16_v2, DP(0, dstDevice); DP(1, us); DP(2, N);)
    DCONV(cuMemsetD32_v2, DP(0, dstDevice); DP(1, ui); DP(2, N);)
    DCONV(cuMemsetD8Async, DP(0, dstDevice); DP(1, uc); DP(2, N); DP(3, hStream);)
    DCONV(cuMemsetD16Async, DP(0, dstDevice); DP(1, us); DP(2, N); DP(3, hStream);)
    DCONV(cuMemsetD32Async, DP(0, dstDevice); DP(1, ui); DP(2, N); DP(3, hStream);)
    DCONV(cuModuleLoad, DP(0, module); DP(1, fname);)
    DCONV(cuModuleLoadData, DP(0, module); DP(1, image);)
    DCONV(cuModuleLoadDataEx, DP(0, module); DP(1, image); DP(2, numOptions); DP(3, options); DP(4, optionValues);)
    DCONV(cuModuleLoadFatBinary, DP(0, module); DP(1, fatCubin);)
    DCONV(cuModuleGetFunction, DP(0, hfunc); DP(1, hmod); DP(2, name);)
    DCONV(cuModuleUnload, DP(0, hmod);)
    DCONV(cuLaunchKernel, DP(0, f); DP(1, gridDimX); DP(2, gridDimY); DP(3, gridDimZ); DP(4, blockDimX);
          DP(5, blockDimY); DP(6, blockDimZ); DP(7, sharedMemBytes); DP(8, hStream); DP(9, kernelParams);
          DP(10, extra);)
    DCONV(cuStreamCreate, DP(0, phStream); DP(1, Flags);)
    DCONV(cuStreamCreateWithPriority, DP(0, phStream); DP(1, flags); DP(2, priority);)
    DCONV(cuStreamDestroy_v2, DP(0, hStream);)
    DCONV(cuStreamSynchronize, DP(0, hStream);)
    DCONV(cuStreamWaitEvent, DP(0, hStream); DP(1, hEvent); DP(2, Flags);)
    DCONV(cuStreamQuery, DP(0, hStream);)
    DCONV(cuEventCreate, DP(0, phEvent); DP(1, Flags);)
    DCONV(cuEventRecord, DP(0, hEvent); DP(1, hStream);)
    DCONV(cuEventSynchronize, DP(0, hEvent);)
    DCONV(cuEventQuery, DP(0, hEvent);)
    DCONV(cuEventDestroy_v2, DP(0, hEvent);)
    DCONV(cuEventElapsedTime, DP(0, pMilliseconds); DP(1, hStart); DP(2, hEnd);)
#ifdef VGPU_HAVE_DRIVER_CBID_cuEventElapsedTime_v2
    DCONV(cuEventElapsedTime_v2, DP(0, pMilliseconds); DP(1, hStart); DP(2, hEnd);)
#endif
  };
  return table;
}
#undef DP
#undef DCONV
#undef DCONV0

/* Every other driver function whose parameter structure the toolkit defines is
   converted by rows made from its own generated_cuda_meta.h at configure time
   (scripts/gen_cupti_driver_conv.py), as the runtime's are. The functions that
   take no arguments, which that header gives no structure, are listed here. A
   row checks that the shim declares the function with arguments of the sizes the
   toolkit's fields have. The rows above, which are checked against an RTX 3060,
   win. */
const GeneratedConv kGeneratedDriverConversions[] = {
#include "cupti_driver_conv.inc"
};

const std::unordered_map<std::string, Conv>& all_driver_conversions() {
  static const std::unordered_map<std::string, Conv> table = [] {
    std::unordered_map<std::string, Conv> t = driver_conversions();
    for (const auto& g : kGeneratedDriverConversions) t.emplace(g.fn, g.conv);
    for (const char* no_args : {"cuProfilerStart", "cuProfilerStop"}) {
      const CUpti_CallbackId id = driver_cbid(no_args);
      if (id != CUPTI_DRIVER_TRACE_CBID_INVALID) t.emplace(no_args, Conv{id, nullptr, 0});
    }
    return t;
  }();
  return table;
}

std::mutex g_cb_mu;
CUpti_CallbackFunc g_cb_fn = nullptr;
void* g_cb_user = nullptr;
bool g_subscribed = false;

// What a subscriber has switched on, per domain: a whole domain at once, then
// single callbacks within it -- each id has its own state, so switching a
// domain on and one id off leaves that one off, and switching a domain on
// again sets every id (measured on an RTX 3060).
struct DomainState {
  bool all = false;
  std::unordered_map<uint32_t, bool> per_id;
  bool on(uint32_t id) const {
    const auto it = per_id.find(id);
    return it != per_id.end() ? it->second : all;
  }
};
constexpr int kDomains = 8;
DomainState g_cb_state[kDomains];   // by CUpti_CallbackDomain

bool cb_enabled(CUpti_CallbackDomain d, uint32_t id, CUpti_CallbackFunc* fn, void** user) {
  std::lock_guard<std::mutex> lock(g_cb_mu);
  if (!g_subscribed || !g_cb_fn) return false;
  const int di = static_cast<int>(d);
  if (di <= 0 || di >= kDomains || !g_cb_state[di].on(id)) return false;
  *fn = g_cb_fn;
  *user = g_cb_user;
  return true;
}

thread_local uint64_t t_correlation_data = 0;   // the subscriber's, from entry to exit of one call

void on_api(const vgpu::profiling::ApiInfo& info) {
  const bool driver = info.domain == vgpu::profiling::Domain::Driver;
  const auto& table = driver ? all_driver_conversions() : all_conversions();
  // CUPTI calls cudaLaunchKernelEx by the C entry point it goes through.
  const std::string name = (!driver && std::strcmp(info.name, "cudaLaunchKernelEx") == 0) ? "cudaLaunchKernelExC" : info.name;
  const auto it = table.find(name);
  if (it == table.end()) return;
  const Conv& conv = it->second;
  const CUpti_CallbackDomain domain = driver ? CUPTI_CB_DOMAIN_DRIVER_API : CUPTI_CB_DOMAIN_RUNTIME_API;
  // The arguments are what the parameter structure is built from; without
  // them there is nothing true to put in it.
  if (conv.fill && !info.args) return;
  if (conv.nargs >= 0 && info.nargs != conv.nargs) {
    CUPTI_TRACE("%s not delivered to callbacks: %d arguments, the toolkit's structure has %d", info.name, info.nargs,
                conv.nargs);
    return;
  }
  // A field is filled from the argument's bytes, so a field the toolkit makes
  // of another size than the argument is not delivered rather than filled from
  // the wrong bytes.
  if (conv.sizes_ok && (!info.arg_sizes || !conv.sizes_ok(info.arg_sizes))) {
    CUPTI_TRACE("%s not delivered to callbacks: an argument is not the size of the toolkit's field", info.name);
    return;
  }
  CUpti_CallbackFunc fn = nullptr;
  void* user = nullptr;
  if (!cb_enabled(domain, conv.cbid, &fn, &user)) return;

  alignas(16) unsigned char storage[512];
  if (conv.fill) conv.fill(storage, info.args);
  // A runtime call returns a cudaError_t, a driver call a CUresult; both are the
  // call's int.
  cudaError_t returned = static_cast<cudaError_t>(info.result);
  CUresult driver_returned = static_cast<CUresult>(info.result);
  CUpti_CallbackData data;
  std::memset(&data, 0, sizeof data);
  data.callbackSite = info.enter ? CUPTI_API_ENTER : CUPTI_API_EXIT;
  data.functionName = info.name;
  data.functionParams = conv.fill ? storage : nullptr;
  data.functionReturnValue = info.enter ? nullptr
                             : info.return_value ? const_cast<void*>(info.return_value)
                             : driver ? static_cast<void*>(&driver_returned) : static_cast<void*>(&returned);
  data.symbolName = info.symbol;
  data.context = current_context();
  data.contextUid = context_id_of(reinterpret_cast<uint64_t>(data.context));
  data.correlationData = &t_correlation_data;
  data.correlationId = info.correlation;
  if (info.enter) t_correlation_data = 0;
  vgpu::profiling::Silence silent;   // the subscriber's own CUDA calls are its own
  fn(user, domain, conv.cbid, &data);
}

CUpti_CallbackId resource_cbid(vgpu::profiling::Resource what) {
  using R = vgpu::profiling::Resource;
  switch (what) {
    case R::CuInitFinished: return CUPTI_CBID_RESOURCE_CU_INIT_FINISHED;
    case R::ContextCreated: return CUPTI_CBID_RESOURCE_CONTEXT_CREATED;
    case R::ContextDestroyStarting: return CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING;
    case R::StreamCreated: return CUPTI_CBID_RESOURCE_STREAM_CREATED;
    case R::StreamDestroyStarting: return CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING;
    case R::ModuleLoaded: return CUPTI_CBID_RESOURCE_MODULE_LOADED;
    case R::ModuleUnloadStarting: return CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING;
    case R::ModuleProfiled: return CUPTI_CBID_RESOURCE_MODULE_PROFILED;
    case R::GraphCreated: return CUPTI_CBID_RESOURCE_GRAPH_CREATED;
    case R::GraphDestroyStarting: return CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING;
    case R::GraphCloned: return CUPTI_CBID_RESOURCE_GRAPH_CLONED;
    case R::GraphNodeCreateStarting: return CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING;
    case R::GraphNodeCreated: return CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED;
    case R::GraphNodeDestroyStarting: return CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING;
    case R::GraphNodeDependencyCreated: return CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED;
    case R::GraphNodeDependencyDestroyStarting: return CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING;
    case R::GraphExecCreateStarting: return CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING;
    case R::GraphExecCreated: return CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED;
    case R::GraphExecDestroyStarting: return CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING;
    case R::GraphNodeCloned: return CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED;
    // The newer ones by number, which older toolkits' headers do not name.
    case R::StreamAttributeChanged: return 21;
    case R::GraphNodeUpdated: return 22;
    case R::GraphNodeSetParams: return 23;
  }
  return 0;
}

void on_resource(const vgpu::profiling::ResourceInfo& info) {
  using R = vgpu::profiling::Resource;
  const CUpti_CallbackId id = resource_cbid(info.what);
  CUpti_CallbackFunc fn = nullptr;
  void* user = nullptr;
  if (!id || !cb_enabled(CUPTI_CB_DOMAIN_RESOURCE, id, &fn, &user)) return;
  // The driver finishing initialisation carries no data, as NVIDIA's.
  if (info.what == R::CuInitFinished) {
    vgpu::profiling::Silence silent;
    fn(user, CUPTI_CB_DOMAIN_RESOURCE, id, nullptr);
    return;
  }
  CUpti_ResourceData data;
  std::memset(&data, 0, sizeof data);
  data.context = info.context ? current_context() : nullptr;
  if (info.what == R::StreamCreated || info.what == R::StreamDestroyStarting ||
      info.what == R::StreamAttributeChanged)
    data.resourceHandle.stream = reinterpret_cast<CUstream>(info.handle);
  CUpti_ModuleResourceData module;
  CUpti_GraphData graph;
  std::memset(&module, 0, sizeof module);
  std::memset(&graph, 0, sizeof graph);
#if CUPTI_API_VERSION >= 22
  CUpti_StreamAttrData attribute;
  std::memset(&attribute, 0, sizeof attribute);
#endif
  switch (info.what) {
    case R::ModuleLoaded:
    case R::ModuleUnloadStarting:
    case R::ModuleProfiled:
      module.moduleId = info.module_id;
      module.cubinSize = info.cubin_size;
      module.pCubin = static_cast<const char*>(info.cubin);
      data.resourceDescriptor = &module;
      break;
    case R::GraphCreated: case R::GraphDestroyStarting: case R::GraphCloned:
    case R::GraphNodeCreateStarting: case R::GraphNodeCreated: case R::GraphNodeDestroyStarting:
    case R::GraphNodeDependencyCreated: case R::GraphNodeDependencyDestroyStarting:
    case R::GraphExecCreateStarting: case R::GraphExecCreated: case R::GraphExecDestroyStarting:
    case R::GraphNodeCloned: case R::GraphNodeUpdated: case R::GraphNodeSetParams:
      graph.graph = reinterpret_cast<CUgraph>(info.graph);
      graph.originalGraph = reinterpret_cast<CUgraph>(info.original_graph);
      graph.node = reinterpret_cast<CUgraphNode>(info.node);
      graph.originalNode = reinterpret_cast<CUgraphNode>(info.original_node);
      graph.nodeType = static_cast<CUgraphNodeType>(info.node_type);
      graph.dependency = reinterpret_cast<CUgraphNode>(info.dependency);
      graph.graphExec = reinterpret_cast<CUgraphExec>(info.graph_exec);
      data.resourceDescriptor = &graph;
      break;
#if CUPTI_API_VERSION >= 22
    case R::StreamAttributeChanged:
      attribute.stream = reinterpret_cast<CUstream>(info.handle);
      attribute.attr = static_cast<CUstreamAttrID>(info.attribute);
      attribute.value = static_cast<const CUstreamAttrValue*>(info.attribute_value);
      data.resourceDescriptor = &attribute;
      break;
#endif
    default: break;
  }
  vgpu::profiling::Silence silent;
  fn(user, CUPTI_CB_DOMAIN_RESOURCE, id, &data);
}

void on_sync(vgpu::profiling::SyncKind what, uint64_t stream) {
  const CUpti_CallbackId id = what == vgpu::profiling::SyncKind::Stream
                                  ? CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED
                                  : CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED;
  CUpti_CallbackFunc fn = nullptr;
  void* user = nullptr;
  if (!cb_enabled(CUPTI_CB_DOMAIN_SYNCHRONIZE, id, &fn, &user)) return;
  CUpti_SynchronizeData data;
  std::memset(&data, 0, sizeof data);
  data.context = current_context();
  data.stream = reinterpret_cast<CUstream>(stream);
  vgpu::profiling::Silence silent;
  fn(user, CUPTI_CB_DOMAIN_SYNCHRONIZE, id, &data);
}

void install_hooks() {
  vgpu::profiling::Hooks h;
  h.api = &on_api;
  h.resource = &on_resource;
  h.sync = &on_sync;
  vgpu::profiling::set_hooks(h);
}

bool known_domain(CUpti_CallbackDomain d) {
  // Every domain CUPTI has: the driver, runtime, resource, synchronize and NVTX
  // ones, and the state domain 13.0 and 12.4 added after NVTX.
  const int di = static_cast<int>(d);
  return di >= 1 && di <= 6;
}

// Which callback ids a domain has. 0 is accepted in each of the first four: it
// names nothing and changes nothing, as on NVIDIA's.
bool valid_callback_id(CUpti_CallbackDomain d, uint32_t id) {
  if (id == 0) return true;
  switch (static_cast<int>(d)) {
    case CUPTI_CB_DOMAIN_DRIVER_API: return driver_cbid_name(id) != nullptr;
    case CUPTI_CB_DOMAIN_RUNTIME_API: return runtime_cbid_name(id) != nullptr;
    case CUPTI_CB_DOMAIN_RESOURCE: return id <= 23;
    case CUPTI_CB_DOMAIN_SYNCHRONIZE: return id <= 2;
    case CUPTI_CB_DOMAIN_NVTX: return id < static_cast<uint32_t>(CUPTI_CBID_NVTX_SIZE);
    default: return id <= 3;   // the state domain
  }
}

bool own_handle(CUpti_SubscriberHandle sub) {
  return sub == reinterpret_cast<CUpti_SubscriberHandle>(&g_subscribed);
}

CUptiResult set_callback(uint32_t enable, CUpti_SubscriberHandle sub, CUpti_CallbackDomain d, uint32_t id,
                         bool whole_domain) {
  std::lock_guard<std::mutex> lock(g_cb_mu);
  if (!g_subscribed || !own_handle(sub)) return CUPTI_ERROR_INVALID_PARAMETER;
  if (!known_domain(d)) return CUPTI_ERROR_INVALID_PARAMETER;
  DomainState& st = g_cb_state[static_cast<int>(d)];
  if (whole_domain) {
    st.all = enable != 0;
    st.per_id.clear();
    return CUPTI_SUCCESS;
  }
  if (!valid_callback_id(d, id)) return CUPTI_ERROR_INVALID_PARAMETER;
  st.per_id[id] = enable != 0;
  return CUPTI_SUCCESS;
}

}  // namespace

VGPU_EXPORT CUptiResult cuptiSubscribe(CUpti_SubscriberHandle* subscriber, CUpti_CallbackFunc callback,
                                       void* userdata) {
  if (!subscriber) return CUPTI_ERROR_INVALID_PARAMETER;
  {
    std::lock_guard<std::mutex> lock(g_cb_mu);
    // One subscriber at a time, which is how NVIDIA's behaves. (It accepts a
    // null callback, which is then never called.)
    if (g_subscribed) return CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED;
    g_subscribed = true;
    g_cb_fn = callback;
    g_cb_user = userdata;
    for (auto& st : g_cb_state) st = DomainState{};
  }
  install_hooks();
  *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(&g_subscribed);
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
  {
    std::lock_guard<std::mutex> lock(g_cb_mu);
    // Unsubscribing a handle already unsubscribed succeeds, as on NVIDIA's.
    if (!own_handle(subscriber)) return CUPTI_ERROR_INVALID_PARAMETER;
    if (!g_subscribed) return CUPTI_SUCCESS;
    g_subscribed = false;
    g_cb_fn = nullptr;
    g_cb_user = nullptr;
    for (auto& st : g_cb_state) st = DomainState{};
  }
  vgpu::profiling::set_hooks(vgpu::profiling::Hooks{});
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiEnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                                            CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
  return set_callback(enable, subscriber, domain, cbid, false);
}
VGPU_EXPORT CUptiResult cuptiEnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                                          CUpti_CallbackDomain domain) {
  return set_callback(enable, subscriber, domain, 0, true);
}
VGPU_EXPORT CUptiResult cuptiEnableAllDomains(uint32_t enable, CUpti_SubscriberHandle subscriber) {
  for (int d = 1; d <= 6; ++d) {
    const CUptiResult r = set_callback(enable, subscriber, static_cast<CUpti_CallbackDomain>(d), 0, true);
    if (r != CUPTI_SUCCESS) return r;
  }
  return CUPTI_SUCCESS;
}
// Whether one callback is switched on.
VGPU_EXPORT CUptiResult cuptiGetCallbackState(uint32_t* enable, CUpti_SubscriberHandle subscriber,
                                              CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
  if (!enable) return CUPTI_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_cb_mu);
  if (!g_subscribed || !own_handle(subscriber) || !known_domain(domain) || !valid_callback_id(domain, cbid))
    return CUPTI_ERROR_INVALID_PARAMETER;
  *enable = g_cb_state[static_cast<int>(domain)].on(cbid) ? 1 : 0;
  return CUPTI_SUCCESS;
}

/* ---- NVTX ----
   A program's own markers and ranges. NVTX is header-only and calls nothing
   until a tool hands it function pointers: it looks for a library exporting
   InitializeInjectionNvtx2 (named by NVTX_INJECTION64_PATH, as a profiler
   sets it) and gives it a table of callbacks to fill in. This is that tool.

   What it produces matches what NVIDIA's CUPTI reports for the same calls
   (checked on an RTX 3060): a MARKER record per instant, per range start and
   per range end, with the id that pairs a start with its end; a MARKER_DATA
   record beside the ones made from attribute structures; a NAME record for a
   named thread. NVTX callbacks are delivered once per call, before it. */

#ifdef VGPU_NVTX
namespace {

struct NvtxDomain {
  std::string name;
};

std::atomic<uint32_t> g_marker_id{1};
thread_local std::vector<uint32_t> t_range_stack;

std::string utf8(const wchar_t* w) {
  std::string out;
  for (; w && *w; ++w) {
    uint32_t c = static_cast<uint32_t>(*w);
    if (c < 0x80) out += static_cast<char>(c);
    else if (c < 0x800) { out += static_cast<char>(0xC0 | (c >> 6)); out += static_cast<char>(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { out += static_cast<char>(0xE0 | (c >> 12)); out += static_cast<char>(0x80 | ((c >> 6) & 0x3F)); out += static_cast<char>(0x80 | (c & 0x3F)); }
    else { out += static_cast<char>(0xF0 | (c >> 18)); out += static_cast<char>(0x80 | ((c >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((c >> 6) & 0x3F)); out += static_cast<char>(0x80 | (c & 0x3F)); }
  }
  return out;
}

// A string registered with a domain is a handle to it; the handle is the
// string's own storage here.
struct RegisteredString {
  std::string text;
};

std::string message_of(const nvtxEventAttributes_t* a) {
  if (!a) return {};
  switch (a->messageType) {
    case NVTX_MESSAGE_TYPE_ASCII: return a->message.ascii ? a->message.ascii : "";
    case NVTX_MESSAGE_TYPE_UNICODE: return utf8(a->message.unicode);
    case NVTX_MESSAGE_TYPE_REGISTERED:
      return a->message.registered ? reinterpret_cast<const RegisteredString*>(a->message.registered)->text : "";
    default: return {};
  }
}

// The subscriber's NVTX callback, once per call and before it.
template <class Params>
void nvtx_callback(CUpti_CallbackId id, const char* name, Params* params) {
  CUpti_CallbackFunc fn = nullptr;
  void* user = nullptr;
  if (!cb_enabled(CUPTI_CB_DOMAIN_NVTX, id, &fn, &user)) return;
  CUpti_NvtxData data;
  std::memset(&data, 0, sizeof data);
  data.functionName = name;
  data.functionParams = params;
  vgpu::profiling::Silence silent;
  fn(user, CUPTI_CB_DOMAIN_NVTX, id, &data);
}

bool recording() { return vgpu::profiling::enabled(); }

void emit_marker(uint32_t flags, uint32_t id, const std::string& name, const NvtxDomain* domain,
                 const nvtxEventAttributes_t* attrib) {
  if (!recording()) return;
  vgpu::profiling::Event e;
  e.kind = vgpu::profiling::EventKind::Marker;
  e.start_ns = e.end_ns = vgpu::profiling::host_ns();
  e.flags = flags;
  e.handle = id;
  e.name = name;
  if (domain) e.domain = domain->name;
  e.process_id = static_cast<uint32_t>(::getpid());
  e.thread_id = static_cast<uint32_t>(::syscall(SYS_gettid));
  vgpu::profiling::record(std::move(e));
  // Only a start or an instant made from attributes carries them.
  if (attrib && (flags & (CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS | CUPTI_ACTIVITY_FLAG_MARKER_START))) {
    vgpu::profiling::Event d;
    d.kind = vgpu::profiling::EventKind::MarkerData;
    d.start_ns = d.end_ns = vgpu::profiling::host_ns();
    d.handle = id;
    const bool argb = attrib->colorType == NVTX_COLOR_ARGB;
    d.flags = argb ? CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB : 0;   // none: the flag stays clear, as NVIDIA's
    d.color = argb ? attrib->color : 0;
    d.category = attrib->category;
    // CUPTI_METRIC_VALUE_KIND_: DOUBLE 0, UINT64 1, INT64 4. A call with no
    // payload is reported as an unsigned zero.
    switch (attrib->payloadType) {
      case NVTX_PAYLOAD_TYPE_DOUBLE:
        d.payload_kind = 0;
        std::memcpy(&d.payload, &attrib->payload.dValue, sizeof d.payload);
        break;
      case NVTX_PAYLOAD_TYPE_FLOAT: {
        const double v = attrib->payload.fValue;
        d.payload_kind = 0;
        std::memcpy(&d.payload, &v, sizeof d.payload);
        break;
      }
      case NVTX_PAYLOAD_TYPE_INT64: d.payload_kind = 4; d.payload = static_cast<uint64_t>(attrib->payload.llValue); break;
      case NVTX_PAYLOAD_TYPE_INT32: d.payload_kind = 4; d.payload = static_cast<uint64_t>(static_cast<int64_t>(attrib->payload.iValue)); break;
      case NVTX_PAYLOAD_TYPE_UNSIGNED_INT64: d.payload_kind = 1; d.payload = attrib->payload.ullValue; break;
      case NVTX_PAYLOAD_TYPE_UNSIGNED_INT32: d.payload_kind = 1; d.payload = attrib->payload.uiValue; break;
      default: d.payload_kind = 1; break;
    }
    vgpu::profiling::record(std::move(d));
  }
}

uint32_t begin_range(const std::string& name, const NvtxDomain* d, const nvtxEventAttributes_t* a) {
  const uint32_t id = g_marker_id.fetch_add(1);
  emit_marker(CUPTI_ACTIVITY_FLAG_MARKER_START, id, name, d, a);
  return id;
}
void end_range(uint32_t id, const NvtxDomain* d) {
  emit_marker(CUPTI_ACTIVITY_FLAG_MARKER_END, id, {}, d, nullptr);
}

/* The entry points handed to NVTX. Each is the default domain's unless it
   names one. */

void NVTX_API nv_mark_ex(const nvtxEventAttributes_t* a) {
  nvtxMarkEx_params p{a};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxMarkEx, "nvtxMarkEx", &p);
  emit_marker(CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS, g_marker_id.fetch_add(1), message_of(a), nullptr, a);
}
void NVTX_API nv_mark_a(const char* m) {
  nvtxMarkA_params p{m};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxMarkA, "nvtxMarkA", &p);
  emit_marker(CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS, g_marker_id.fetch_add(1), m ? m : "", nullptr, nullptr);
}
nvtxRangeId_t NVTX_API nv_range_start_ex(const nvtxEventAttributes_t* a) {
  nvtxRangeStartEx_params p{a};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxRangeStartEx, "nvtxRangeStartEx", &p);
  return begin_range(message_of(a), nullptr, a);
}
nvtxRangeId_t NVTX_API nv_range_start_a(const char* m) {
  nvtxRangeStartA_params p{m};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxRangeStartA, "nvtxRangeStartA", &p);
  return begin_range(m ? m : "", nullptr, nullptr);
}
void NVTX_API nv_range_end(nvtxRangeId_t id) {
  nvtxRangeEnd_params p{id};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxRangeEnd, "nvtxRangeEnd", &p);
  end_range(static_cast<uint32_t>(id), nullptr);
}
int NVTX_API nv_range_push_ex(const nvtxEventAttributes_t* a) {
  nvtxRangePushEx_params p{a};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxRangePushEx, "nvtxRangePushEx", &p);
  t_range_stack.push_back(begin_range(message_of(a), nullptr, a));
  return static_cast<int>(t_range_stack.size()) - 1;
}
int NVTX_API nv_range_push_a(const char* m) {
  nvtxRangePushA_params p{m};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxRangePushA, "nvtxRangePushA", &p);
  t_range_stack.push_back(begin_range(m ? m : "", nullptr, nullptr));
  return static_cast<int>(t_range_stack.size()) - 1;
}
int NVTX_API nv_range_pop() {
  nvtxRangePop_params p{0};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxRangePop, "nvtxRangePop", &p);
  if (t_range_stack.empty()) return -1;   // NVTX_NO_PUSH_POP_TRACKING
  const uint32_t id = t_range_stack.back();
  t_range_stack.pop_back();
  end_range(id, nullptr);
  return static_cast<int>(t_range_stack.size());
}
void NVTX_API nv_name_os_thread_a(uint32_t tid, const char* name) {
  nvtxNameOsThreadA_params p{tid, name};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxNameOsThreadA, "nvtxNameOsThreadA", &p);
  if (!recording()) return;
  vgpu::profiling::Event e;
  e.kind = vgpu::profiling::EventKind::Name;
  e.start_ns = e.end_ns = vgpu::profiling::host_ns();
  e.handle = tid;
  e.name = name ? name : "";
  e.process_id = static_cast<uint32_t>(::getpid());
  vgpu::profiling::record(std::move(e));
}
nvtxDomainHandle_t NVTX_API nv_domain_create_a(const char* name) {
  nvtxDomainCreateA_params p{name};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxDomainCreateA, "nvtxDomainCreateA", &p);
  return reinterpret_cast<nvtxDomainHandle_t>(new NvtxDomain{name ? name : ""});
}
void NVTX_API nv_domain_destroy(nvtxDomainHandle_t d) {
  nvtxDomainDestroy_params p{d};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxDomainDestroy, "nvtxDomainDestroy", &p);
  delete reinterpret_cast<NvtxDomain*>(d);
}
void NVTX_API nv_domain_mark_ex(nvtxDomainHandle_t d, const nvtxEventAttributes_t* a) {
  nvtxDomainMarkEx_params p{d, {a}};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxDomainMarkEx, "nvtxDomainMarkEx", &p);
  emit_marker(CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS, g_marker_id.fetch_add(1), message_of(a),
              reinterpret_cast<NvtxDomain*>(d), a);
}
nvtxRangeId_t NVTX_API nv_domain_range_start_ex(nvtxDomainHandle_t d, const nvtxEventAttributes_t* a) {
  nvtxDomainRangeStartEx_params p{d, {a}};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxDomainRangeStartEx, "nvtxDomainRangeStartEx", &p);
  return begin_range(message_of(a), reinterpret_cast<NvtxDomain*>(d), a);
}
void NVTX_API nv_domain_range_end(nvtxDomainHandle_t d, nvtxRangeId_t id) {
  nvtxDomainRangeEnd_params p{d, {id}};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxDomainRangeEnd, "nvtxDomainRangeEnd", &p);
  end_range(static_cast<uint32_t>(id), reinterpret_cast<NvtxDomain*>(d));
}
int NVTX_API nv_domain_range_push_ex(nvtxDomainHandle_t d, const nvtxEventAttributes_t* a) {
  nvtxDomainRangePushEx_params p{d, {a}};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxDomainRangePushEx, "nvtxDomainRangePushEx", &p);
  t_range_stack.push_back(begin_range(message_of(a), reinterpret_cast<NvtxDomain*>(d), a));
  return static_cast<int>(t_range_stack.size()) - 1;
}
int NVTX_API nv_domain_range_pop(nvtxDomainHandle_t d) {
  nvtxDomainRangePop_params p{d};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxDomainRangePop, "nvtxDomainRangePop", &p);
  if (t_range_stack.empty()) return -1;
  const uint32_t id = t_range_stack.back();
  t_range_stack.pop_back();
  end_range(id, reinterpret_cast<NvtxDomain*>(d));
  return static_cast<int>(t_range_stack.size());
}
nvtxStringHandle_t NVTX_API nv_domain_register_string_a(nvtxDomainHandle_t d, const char* s) {
  nvtxDomainRegisterStringA_params p{d, s};
  nvtx_callback(CUPTI_CBID_NVTX_nvtxDomainRegisterStringA, "nvtxDomainRegisterStringA", &p);
  return reinterpret_cast<nvtxStringHandle_t>(new RegisteredString{s ? s : ""});
}

template <class Fn>
void install(NvtxFunctionTable table, unsigned size, unsigned id, Fn fn) {
  if (id <= size && table[id]) *table[id] = reinterpret_cast<NvtxFunctionPointer>(fn);
}

}  // namespace

// NVTX's entry point for a tool: fill in the callbacks the program's NVTX calls
// go through. Calls it does not fill are no-ops, as with no tool attached --
// NVIDIA's CUPTI fills none of the wide-character spellings (checked on an RTX
// 3060: no callback and no record for them), and neither does this.
extern "C" __attribute__((visibility("default"))) int NVTX_API InitializeInjectionNvtx2(
    NvtxGetExportTableFunc_t get_table) {
  const auto* callbacks = static_cast<const NvtxExportTableCallbacks*>(get_table(NVTX_ETID_CALLBACKS));
  if (!callbacks || !callbacks->GetModuleFunctionTable) return 0;
  NvtxFunctionTable core = nullptr, core2 = nullptr;
  unsigned core_size = 0, core2_size = 0;
  if (!callbacks->GetModuleFunctionTable(NVTX_CB_MODULE_CORE, &core, &core_size) || !core) return 0;
  install(core, core_size, NVTX_CBID_CORE_MarkEx, nv_mark_ex);
  install(core, core_size, NVTX_CBID_CORE_MarkA, nv_mark_a);
  install(core, core_size, NVTX_CBID_CORE_RangeStartEx, nv_range_start_ex);
  install(core, core_size, NVTX_CBID_CORE_RangeStartA, nv_range_start_a);
  install(core, core_size, NVTX_CBID_CORE_RangeEnd, nv_range_end);
  install(core, core_size, NVTX_CBID_CORE_RangePushEx, nv_range_push_ex);
  install(core, core_size, NVTX_CBID_CORE_RangePushA, nv_range_push_a);
  install(core, core_size, NVTX_CBID_CORE_RangePop, nv_range_pop);
  install(core, core_size, NVTX_CBID_CORE_NameOsThreadA, nv_name_os_thread_a);
  if (callbacks->GetModuleFunctionTable(NVTX_CB_MODULE_CORE2, &core2, &core2_size) && core2) {
    install(core2, core2_size, NVTX_CBID_CORE2_DomainMarkEx, nv_domain_mark_ex);
    install(core2, core2_size, NVTX_CBID_CORE2_DomainRangeStartEx, nv_domain_range_start_ex);
    install(core2, core2_size, NVTX_CBID_CORE2_DomainRangeEnd, nv_domain_range_end);
    install(core2, core2_size, NVTX_CBID_CORE2_DomainRangePushEx, nv_domain_range_push_ex);
    install(core2, core2_size, NVTX_CBID_CORE2_DomainRangePop, nv_domain_range_pop);
    install(core2, core2_size, NVTX_CBID_CORE2_DomainRegisterStringA, nv_domain_register_string_a);
    install(core2, core2_size, NVTX_CBID_CORE2_DomainCreateA, nv_domain_create_a);
    install(core2, core2_size, NVTX_CBID_CORE2_DomainDestroy, nv_domain_destroy);
  }
  return 1;
}
#endif  // VGPU_NVTX

/* ---- identifiers and attributes ----
   Small, real, and needed before a tool will get as far as asking for
   anything interesting. */

VGPU_EXPORT CUptiResult cuptiDeviceSupported(CUdevice, int* support) {
  if (!support) return CUPTI_ERROR_INVALID_PARAMETER;
  *support = 1;   // activity tracing works; events and metrics say so themselves
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiGetContextId(CUcontext context, uint32_t* id) {
  if (!id) return CUPTI_ERROR_INVALID_PARAMETER;
  *id = context_id_of(reinterpret_cast<uint64_t>(context));
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiGetDeviceId(CUcontext, uint32_t* id) {
  if (!id) return CUPTI_ERROR_INVALID_PARAMETER;
  int device = 0;
  {
    vgpu::profiling::Silence silent;
    if (cudaGetDevice(&device) != cudaSuccess) device = 0;
  }
  *id = static_cast<uint32_t>(device);
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiGetStreamId(CUcontext, CUstream stream, uint32_t* id) {
  if (!id) return CUPTI_ERROR_INVALID_PARAMETER;
  *id = stream_id_of(reinterpret_cast<uint64_t>(stream), vgpu::profiling::stream_generation(reinterpret_cast<uint64_t>(stream)));
  return CUPTI_SUCCESS;
}
// The numbers a profiler gives graphs, executable graphs and their nodes
// (vgpu/profiling.hpp): the same ones the launch records carry.
VGPU_EXPORT CUptiResult cuptiGetGraphId(CUgraph graph, uint32_t* id) {
  if (!graph || !id) return CUPTI_ERROR_INVALID_PARAMETER;
  return vgpu::profiling::graph_id_of(reinterpret_cast<uint64_t>(graph), id) ? CUPTI_SUCCESS
                                                                            : CUPTI_ERROR_INVALID_PARAMETER;
}
VGPU_EXPORT CUptiResult cuptiGetGraphExecId(CUgraphExec graph, uint32_t* id) {
  if (!graph || !id) return CUPTI_ERROR_INVALID_PARAMETER;
  return vgpu::profiling::graph_id_of(reinterpret_cast<uint64_t>(graph), id) ? CUPTI_SUCCESS
                                                                            : CUPTI_ERROR_INVALID_PARAMETER;
}
VGPU_EXPORT CUptiResult cuptiGetGraphNodeId(CUgraphNode node, uint64_t* id) {
  if (!node || !id) return CUPTI_ERROR_INVALID_PARAMETER;
  return vgpu::profiling::graph_node_id_of(reinterpret_cast<uint64_t>(node), id)
             ? CUPTI_SUCCESS
             : CUPTI_ERROR_INVALID_PARAMETER;
}
VGPU_EXPORT CUptiResult cuptiGetStreamIdEx(CUcontext c, CUstream stream, uint8_t, uint32_t* id) {
  return cuptiGetStreamId(c, stream, id);
}
// The names of the runtime and driver functions, for a tool labelling what it
// receives: the runtime's versioned spelling, as NVIDIA's reports it
// (cudaMalloc_v3020), and the driver function's own (cuMemAlloc_v2).
VGPU_EXPORT CUptiResult cuptiGetCallbackName(CUpti_CallbackDomain domain, uint32_t cbid, const char** name) {
  if (!name) return CUPTI_ERROR_INVALID_PARAMETER;
  *name = nullptr;
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
    if (const char* n = runtime_cbid_name(cbid)) {
      *name = n;
      return CUPTI_SUCCESS;
    }
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API)
    if (const char* n = driver_cbid_name(cbid)) {
      *name = n;
      return CUPTI_SUCCESS;
    }
  return CUPTI_ERROR_INVALID_PARAMETER;
}
VGPU_EXPORT CUptiResult cuptiActivitySetAttribute(CUpti_ActivityAttribute, size_t*, void*) {
  // Buffer sizes and watermarks: this hands whole buffers to the consumer's own
  // allocator on flush, so there is nothing to tune.
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiActivityEnableLatencyTimestamps(uint8_t) { return CUPTI_SUCCESS; }
VGPU_EXPORT CUptiResult cuptiEnableNonOverlappingMode(void) { return CUPTI_SUCCESS; }
VGPU_EXPORT CUptiResult cuptiDisableNonOverlappingMode(void) { return CUPTI_SUCCESS; }

/* ---- events and metrics ----
   Hardware performance counters, which this does not have. The exact counters
   this engine keeps -- instruction mix, sectors and coalescing, bank
   conflicts, tensor issues -- are a different set from a device's, and
   answering to NVIDIA's metric names would claim an equivalence that does not
   hold. So the device reports no event domains and no metrics, which is the
   truth, and a tool that asks is told plainly rather than handed a number.

   Reporting zero rather than failing outright matters: a profiler that cannot
   enumerate counters still goes on to trace activity, which does work. */

VGPU_EXPORT CUptiResult cuptiDeviceGetNumEventDomains(CUdevice, uint32_t* n) {
  if (!n) return CUPTI_ERROR_INVALID_PARAMETER;
  *n = 0;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiDeviceEnumEventDomains(CUdevice, size_t* size, CUpti_EventDomainID*) {
  if (size) *size = 0;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiDeviceGetNumMetrics(CUdevice, uint32_t* n) {
  if (!n) return CUPTI_ERROR_INVALID_PARAMETER;
  *n = 0;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiDeviceEnumMetrics(CUdevice, size_t* size, CUpti_MetricID*) {
  if (size) *size = 0;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiDeviceGetAttribute(CUdevice, CUpti_DeviceAttribute, size_t*, void*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiDeviceGetEventDomainAttribute(CUdevice, CUpti_EventDomainID,
                                                           CUpti_EventDomainAttribute, size_t*,
                                                           void*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiEventDomainEnumEvents(CUpti_EventDomainID, size_t* size,
                                                   CUpti_EventID*) {
  if (size) *size = 0;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiEventDomainGetNumEvents(CUpti_EventDomainID, uint32_t* n) {
  if (!n) return CUPTI_ERROR_INVALID_PARAMETER;
  *n = 0;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiEventGetAttribute(CUpti_EventID, CUpti_EventAttribute, size_t*, void*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiEventGetIdFromName(CUdevice, const char*, CUpti_EventID*) {
  return CUPTI_ERROR_INVALID_EVENT_NAME;
}
VGPU_EXPORT CUptiResult cuptiEventGroupGetAttribute(CUpti_EventGroup, CUpti_EventGroupAttribute,
                                                    size_t*, void*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiEventGroupSetAttribute(CUpti_EventGroup, CUpti_EventGroupAttribute,
                                                    size_t, void*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiEventGroupReadEvent(CUpti_EventGroup, CUpti_ReadEventFlags, CUpti_EventID,
                                                 size_t*, uint64_t*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiEventGroupReadAllEvents(CUpti_EventGroup, CUpti_ReadEventFlags,
                                                     size_t*, uint64_t*, size_t*, CUpti_EventID*,
                                                     size_t*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiEventGroupSetsCreate(CUcontext, size_t, CUpti_EventID*,
                                                  CUpti_EventGroupSets**) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiEventGroupSetsDestroy(CUpti_EventGroupSets*) { return CUPTI_SUCCESS; }
VGPU_EXPORT CUptiResult cuptiEventGroupSetEnable(CUpti_EventGroupSet*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiEventGroupSetDisable(CUpti_EventGroupSet*) { return CUPTI_SUCCESS; }
VGPU_EXPORT CUptiResult cuptiSetEventCollectionMode(CUcontext, CUpti_EventCollectionMode) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiMetricGetIdFromName(CUdevice, const char*, CUpti_MetricID*) {
  return CUPTI_ERROR_INVALID_METRIC_NAME;
}
VGPU_EXPORT CUptiResult cuptiMetricGetAttribute(CUpti_MetricID, CUpti_MetricAttribute, size_t*,
                                                void*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiMetricGetNumEvents(CUpti_MetricID, uint32_t* n) {
  if (!n) return CUPTI_ERROR_INVALID_PARAMETER;
  *n = 0;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiMetricEnumEvents(CUpti_MetricID, size_t* size, CUpti_EventID*) {
  if (size) *size = 0;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiMetricCreateEventGroupSets(CUcontext, size_t, CUpti_MetricID*,
                                                        CUpti_EventGroupSets**) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiMetricGetRequiredEventGroupSets(CUcontext, CUpti_MetricID,
                                                             CUpti_EventGroupSets**) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiMetricGetValue(CUdevice, CUpti_MetricID, size_t, CUpti_EventID*,
                                            size_t, uint64_t*, uint64_t, CUpti_MetricValue*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}

/* ---- replay, sampling and the other subsystems this has no device for ---- */

VGPU_EXPORT CUptiResult cuptiEnableKernelReplayMode(CUcontext) { return CUPTI_ERROR_NOT_SUPPORTED; }
VGPU_EXPORT CUptiResult cuptiDisableKernelReplayMode(CUcontext) { return CUPTI_SUCCESS; }
VGPU_EXPORT CUptiResult cuptiKernelReplaySubscribeUpdate(CUpti_KernelReplayUpdateFunc, void*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiActivityConfigurePCSampling(CUcontext, CUpti_ActivityPCSamplingConfig*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
// Measured on an RTX 3060 under WSL2: any list is accepted whatever its scope,
// kind, device or enable field says, and nothing or a null list is a bad
// parameter. The counters it configures are not produced (see enable_result).
VGPU_EXPORT CUptiResult cuptiActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) {
  if (!config || count == 0) return CUPTI_ERROR_INVALID_PARAMETER;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiGetAutoBoostState(CUcontext, CUpti_ActivityAutoBoostState*) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiNvtxInitialize(void*) { return CUPTI_SUCCESS; }
VGPU_EXPORT CUptiResult cuptiNvtxInitialize2(void*) { return CUPTI_SUCCESS; }
VGPU_EXPORT CUptiResult cuptiOpenACCInitialize(void*) { return CUPTI_ERROR_NOT_SUPPORTED; }

VGPU_EXPORT CUptiResult cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                                              uint64_t id) {
  // Kind 0 is invalid; the rest index the stacks. NVIDIA's reports a kind
  // outside its range as an invalid parameter.
  if (kind == CUPTI_EXTERNAL_CORRELATION_KIND_INVALID || !vgpu::profiling::push_external(static_cast<int>(kind), id))
    return CUPTI_ERROR_INVALID_PARAMETER;
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                                             uint64_t* last) {
  if (kind == CUPTI_EXTERNAL_CORRELATION_KIND_INVALID) return CUPTI_ERROR_INVALID_PARAMETER;
  uint64_t popped = 0;
  // An empty stack is CUPTI_ERROR_QUEUE_EMPTY (18), measured on an RTX 3060.
  if (!vgpu::profiling::pop_external(static_cast<int>(kind), &popped)) return CUPTI_ERROR_QUEUE_EMPTY;
  if (last) *last = popped;
  return CUPTI_SUCCESS;
}
