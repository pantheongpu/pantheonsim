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

namespace {

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

std::mutex g_mu;
CUpti_BuffersCallbackRequestFunc g_request = nullptr;
CUpti_BuffersCallbackCompleteFunc g_complete = nullptr;
std::vector<bool> g_kinds(CUPTI_ACTIVITY_KIND_COUNT, false);
bool g_announced = false;
bool g_devices_delivered = false;

// Kinds this can actually produce. Enabling anything else succeeds -- refusing
// would stop a profiler that asks for everything and uses what arrives -- but
// nothing will come back for it, which is the honest outcome for a device
// with no timing model and no hardware counters behind these kinds.
bool produced(CUpti_ActivityKind k) {
  return k == CUPTI_ACTIVITY_KIND_KERNEL || k == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
         k == CUPTI_ACTIVITY_KIND_MEMCPY || k == CUPTI_ACTIVITY_KIND_MEMSET ||
         k == CUPTI_ACTIVITY_KIND_RUNTIME || k == CUPTI_ACTIVITY_KIND_SYNCHRONIZATION ||
         k == CUPTI_ACTIVITY_KIND_DEVICE || k == CUPTI_ACTIVITY_KIND_CONTEXT ||
         k == CUPTI_ACTIVITY_KIND_STREAM || k == CUPTI_ACTIVITY_KIND_MARKER ||
         k == CUPTI_ACTIVITY_KIND_MARKER_DATA || k == CUPTI_ACTIVITY_KIND_NAME ||
         k == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
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
   (cuptiGetStreamId). The default stream is 7, as CUDA 12 and 13 number it. */

std::mutex g_id_mu;
std::unordered_map<uint64_t, uint32_t> g_stream_ids;
std::unordered_map<uint64_t, uint32_t> g_context_ids;
std::unordered_map<uint64_t, uint32_t> g_event_ids;

uint32_t stream_id_of(uint64_t handle) {
  if (handle == 0 || handle == 1 || handle == 2) return 7;   // 0, cudaStreamLegacy, cudaStreamPerThread
  std::lock_guard<std::mutex> lock(g_id_mu);
  return g_stream_ids.emplace(handle, 8u + static_cast<uint32_t>(g_stream_ids.size())).first->second;
}

uint32_t event_id_of(uint64_t handle) {
  std::lock_guard<std::mutex> lock(g_id_mu);
  return g_event_ids.emplace(handle, 1u + static_cast<uint32_t>(g_event_ids.size())).first->second;
}

uint32_t context_id_of(uint64_t handle) {
  std::lock_guard<std::mutex> lock(g_id_mu);
  return g_context_ids.emplace(handle, 1u + static_cast<uint32_t>(g_context_ids.size())).first->second;
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

/* ---- records ---- */

size_t fill_kernel(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* k = reinterpret_cast<KernelRecord*>(out);
  std::memset(k, 0, sizeof *k);
  k->kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
  k->start = e.start_ns;
  k->end = e.end_ns;
  k->deviceId = e.device;
  k->contextId = kStandInContext();
  k->correlationId = e.correlation;
  k->streamId = stream_id_of(e.stream);
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
  m->contextId = kStandInContext();
  m->correlationId = e.correlation;
  m->streamId = stream_id_of(e.stream);
  m->bytes = e.bytes;
  m->srcKind = memory_kind(e.src_kind);
  m->dstKind = memory_kind(e.dst_kind);
  if (e.async) m->flags |= CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
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
  m->contextId = kStandInContext();
  m->correlationId = e.correlation;
  m->streamId = stream_id_of(e.stream);
  m->bytes = e.bytes;
  m->value = e.value;
  m->memoryKind = static_cast<uint16_t>(memory_kind(e.dst_kind));
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
  r->contextId = kStandInContext();
  r->streamId = (e.sync_kind == 1 || e.sync_kind == 4) ? CUPTI_SYNCHRONIZATION_INVALID_VALUE : stream_id_of(e.stream);
  r->cudaEventId = e.handle ? event_id_of(e.handle) : CUPTI_SYNCHRONIZATION_INVALID_VALUE;
  return sizeof *r;
}

size_t fill_stream(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* r = reinterpret_cast<CUpti_ActivityStream*>(out);
  std::memset(r, 0, sizeof *r);
  r->kind = CUPTI_ACTIVITY_KIND_STREAM;
  r->contextId = kStandInContext();
  r->streamId = stream_id_of(e.handle);
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
  r->contextId = kStandInContext();
  r->deviceId = e.device;
  r->computeApiKind = CUPTI_ACTIVITY_COMPUTE_API_CUDA;
  r->nullStreamId = static_cast<uint16_t>(stream_id_of(0));
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
  return it == ids.end() ? CUPTI_RUNTIME_TRACE_CBID_INVALID : it->second;
}

size_t fill_api(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* a = reinterpret_cast<CUpti_ActivityAPI*>(out);
  std::memset(a, 0, sizeof *a);
  a->kind = e.domain_driver ? CUPTI_ACTIVITY_KIND_DRIVER : CUPTI_ACTIVITY_KIND_RUNTIME;
  a->cbid = runtime_cbid(e.name);
  a->start = e.start_ns;
  a->end = e.end_ns;
  a->processId = e.process_id;
  a->threadId = e.thread_id;
  a->correlationId = e.correlation;
  a->returnValue = static_cast<uint32_t>(e.result);
  return sizeof *a;
}

size_t kind_size(CUpti_ActivityKind k) {
  switch (k) {
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
    default: return fill_memcpy(out, e);
  }
}

bool kind_wanted(const vgpu::profiling::Event& e) {
  using K = vgpu::profiling::EventKind;
  switch (e.kind) {
    case K::Kernel: return g_kinds[CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL] || g_kinds[CUPTI_ACTIVITY_KIND_KERNEL];
    case K::Api:
      return !e.domain_driver && g_kinds[CUPTI_ACTIVITY_KIND_RUNTIME] &&
             runtime_cbid(e.name) != CUPTI_RUNTIME_TRACE_CBID_INVALID;
    case K::Memset: return g_kinds[CUPTI_ACTIVITY_KIND_MEMSET];
    case K::Sync: return g_kinds[CUPTI_ACTIVITY_KIND_SYNCHRONIZATION];
    case K::Stream: return g_kinds[CUPTI_ACTIVITY_KIND_STREAM];
    case K::Context: return g_kinds[CUPTI_ACTIVITY_KIND_CONTEXT];
    case K::Device: return g_kinds[CUPTI_ACTIVITY_KIND_DEVICE];
    case K::Marker: return g_kinds[CUPTI_ACTIVITY_KIND_MARKER];
    case K::MarkerData: return g_kinds[CUPTI_ACTIVITY_KIND_MARKER_DATA];
    case K::Name: return g_kinds[CUPTI_ACTIVITY_KIND_NAME];
    case K::ExternalCorrelation: return g_kinds[CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION];
    default: return g_kinds[CUPTI_ACTIVITY_KIND_MEMCPY];
  }
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
  switch (result) {
    case CUPTI_SUCCESS: *str = "CUPTI_SUCCESS"; break;
    case CUPTI_ERROR_INVALID_PARAMETER: *str = "CUPTI_ERROR_INVALID_PARAMETER"; break;
    case CUPTI_ERROR_NOT_INITIALIZED: *str = "CUPTI_ERROR_NOT_INITIALIZED"; break;
    case CUPTI_ERROR_NOT_SUPPORTED: *str = "CUPTI_ERROR_NOT_SUPPORTED"; break;
    case CUPTI_ERROR_MAX_LIMIT_REACHED: *str = "CUPTI_ERROR_MAX_LIMIT_REACHED"; break;
    default: *str = "CUPTI_ERROR_UNKNOWN"; break;
  }
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiGetLastError(void) { return CUPTI_SUCCESS; }

VGPU_EXPORT CUptiResult cuptiGetTimestamp(uint64_t* timestamp) {
  if (!timestamp) return CUPTI_ERROR_INVALID_PARAMETER;
  *timestamp = vgpu::profiling::now_ns();
  return CUPTI_SUCCESS;
}

/* ---- activity ---- */

VGPU_EXPORT CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
  if (!funcBufferRequested || !funcBufferCompleted) return CUPTI_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mu);
  g_request = funcBufferRequested;
  g_complete = funcBufferCompleted;
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
  if (kind >= CUPTI_ACTIVITY_KIND_COUNT) return CUPTI_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mu);
  g_kinds[kind] = true;
  if (kind == CUPTI_ACTIVITY_KIND_DEVICE) g_devices_delivered = false;
  if (produced(kind)) {
    announce_once();
    vgpu::profiling::set_enabled(true);
  }
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
  if (kind >= CUPTI_ACTIVITY_KIND_COUNT) return CUPTI_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mu);
  g_kinds[kind] = false;
  bool any = false;
  for (int i = 0; i < CUPTI_ACTIVITY_KIND_COUNT; ++i)
    if (g_kinds[i] && produced(static_cast<CUpti_ActivityKind>(i))) any = true;
  if (!any) vgpu::profiling::set_enabled(false);
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiActivityEnableContext(CUcontext, CUpti_ActivityKind kind) {
  return cuptiActivityEnable(kind);
}
VGPU_EXPORT CUptiResult cuptiActivityDisableContext(CUcontext, CUpti_ActivityKind kind) {
  return cuptiActivityDisable(kind);
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

VGPU_EXPORT CUptiResult cuptiActivityFlushAll(uint32_t) {
  CUpti_BuffersCallbackRequestFunc request = nullptr;
  CUpti_BuffersCallbackCompleteFunc complete = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    request = g_request;
    complete = g_complete;
  }
  if (!request || !complete) return CUPTI_ERROR_NOT_INITIALIZED;

  std::vector<vgpu::profiling::Event> events = vgpu::profiling::drain();
  std::vector<vgpu::profiling::Event> wanted;
  bool devices = false;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_kinds[CUPTI_ACTIVITY_KIND_DEVICE] && !g_devices_delivered) devices = g_devices_delivered = true;
    for (auto& e : events)
      if (kind_wanted(e)) wanted.push_back(std::move(e));
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
  if (wanted.empty()) return CUPTI_SUCCESS;

  size_t i = 0;
  while (i < wanted.size()) {
    uint8_t* buffer = nullptr;
    size_t size = 0;
    size_t max_records = 0;
    request(&buffer, &size, &max_records);
    if (!buffer || size == 0) return CUPTI_ERROR_MAX_LIMIT_REACHED;

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
  std::lock_guard<std::mutex> lock(g_mu);
  vgpu::profiling::set_enabled(false);
  g_request = nullptr;
  g_complete = nullptr;
  std::fill(g_kinds.begin(), g_kinds.end(), false);
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

   Resource and synchronize domains: a stream made or destroyed, a context
   created, a stream or device waited on. Module and graph resources, and the
   driver domain, are not delivered. */

namespace {

struct Conv {
  CUpti_CallbackId cbid;
  void (*fill)(void* storage, const void* const* a);   // null: the call has no parameters
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

std::mutex g_cb_mu;
CUpti_CallbackFunc g_cb_fn = nullptr;
void* g_cb_user = nullptr;
bool g_subscribed = false;
// {domain, id} pairs a subscriber enabled; kWholeDomain stands for all of one.
std::unordered_set<uint64_t> g_cb_on;
constexpr uint32_t kWholeDomain = 0xffffffffu;

uint64_t cb_key(CUpti_CallbackDomain d, uint32_t id) { return (uint64_t(d) << 32) | id; }

bool cb_enabled(CUpti_CallbackDomain d, uint32_t id, CUpti_CallbackFunc* fn, void** user) {
  std::lock_guard<std::mutex> lock(g_cb_mu);
  if (!g_subscribed || !g_cb_fn) return false;
  if (!g_cb_on.count(cb_key(d, kWholeDomain)) && !g_cb_on.count(cb_key(d, id))) return false;
  *fn = g_cb_fn;
  *user = g_cb_user;
  return true;
}

thread_local uint64_t t_correlation_data = 0;   // the subscriber's, from entry to exit of one call

void on_api(const vgpu::profiling::ApiInfo& info) {
  if (info.domain != vgpu::profiling::Domain::Runtime) return;
  const auto& table = conversions();
  const auto it = table.find(info.name);
  if (it == table.end()) return;
  const Conv& conv = it->second;
  // The arguments are what the parameter structure is built from; without
  // them there is nothing true to put in it.
  if (conv.fill && !info.args) return;
  CUpti_CallbackFunc fn = nullptr;
  void* user = nullptr;
  if (!cb_enabled(CUPTI_CB_DOMAIN_RUNTIME_API, conv.cbid, &fn, &user)) return;

  alignas(16) unsigned char storage[256];
  if (conv.fill) conv.fill(storage, info.args);
  cudaError_t returned = static_cast<cudaError_t>(info.result);
  CUpti_CallbackData data;
  std::memset(&data, 0, sizeof data);
  data.callbackSite = info.enter ? CUPTI_API_ENTER : CUPTI_API_EXIT;
  data.functionName = info.name;
  data.functionParams = conv.fill ? storage : nullptr;
  data.functionReturnValue = info.enter ? nullptr : &returned;
  data.symbolName = info.symbol;
  data.context = current_context();
  data.contextUid = context_id_of(reinterpret_cast<uint64_t>(data.context));
  data.correlationData = &t_correlation_data;
  data.correlationId = info.correlation;
  if (info.enter) t_correlation_data = 0;
  vgpu::profiling::Silence silent;   // the subscriber's own CUDA calls are its own
  fn(user, CUPTI_CB_DOMAIN_RUNTIME_API, conv.cbid, &data);
}

void on_resource(vgpu::profiling::Resource what, uint64_t handle, uint32_t) {
  using R = vgpu::profiling::Resource;
  CUpti_CallbackId id = 0;
  switch (what) {
    case R::CuInitFinished: id = CUPTI_CBID_RESOURCE_CU_INIT_FINISHED; break;
    case R::ContextCreated: id = CUPTI_CBID_RESOURCE_CONTEXT_CREATED; break;
    case R::ContextDestroyStarting: id = CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING; break;
    case R::StreamCreated: id = CUPTI_CBID_RESOURCE_STREAM_CREATED; break;
    case R::StreamDestroyStarting: id = CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING; break;
  }
  CUpti_CallbackFunc fn = nullptr;
  void* user = nullptr;
  if (!cb_enabled(CUPTI_CB_DOMAIN_RESOURCE, id, &fn, &user)) return;
  CUpti_ResourceData data;
  std::memset(&data, 0, sizeof data);
  data.context = current_context();
  if (what == R::StreamCreated || what == R::StreamDestroyStarting)
    data.resourceHandle.stream = reinterpret_cast<CUstream>(handle);
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
  return d == CUPTI_CB_DOMAIN_DRIVER_API || d == CUPTI_CB_DOMAIN_RUNTIME_API ||
         d == CUPTI_CB_DOMAIN_RESOURCE || d == CUPTI_CB_DOMAIN_SYNCHRONIZE ||
         d == CUPTI_CB_DOMAIN_NVTX;
}

CUptiResult set_callback(uint32_t enable, CUpti_SubscriberHandle sub, CUpti_CallbackDomain d, uint32_t id) {
  std::lock_guard<std::mutex> lock(g_cb_mu);
  if (!g_subscribed || sub != reinterpret_cast<CUpti_SubscriberHandle>(&g_subscribed))
    return CUPTI_ERROR_INVALID_PARAMETER;
  if (!known_domain(d)) return CUPTI_ERROR_INVALID_PARAMETER;
  if (enable) g_cb_on.insert(cb_key(d, id));
  else g_cb_on.erase(cb_key(d, id));
  return CUPTI_SUCCESS;
}

}  // namespace

VGPU_EXPORT CUptiResult cuptiSubscribe(CUpti_SubscriberHandle* subscriber, CUpti_CallbackFunc callback,
                                       void* userdata) {
  if (!subscriber || !callback) return CUPTI_ERROR_INVALID_PARAMETER;
  {
    std::lock_guard<std::mutex> lock(g_cb_mu);
    // One subscriber at a time, which is how NVIDIA's behaves.
    if (g_subscribed) return CUPTI_ERROR_MAX_LIMIT_REACHED;
    g_subscribed = true;
    g_cb_fn = callback;
    g_cb_user = userdata;
    g_cb_on.clear();
  }
  install_hooks();
  *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(&g_subscribed);
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
  {
    std::lock_guard<std::mutex> lock(g_cb_mu);
    if (!g_subscribed || subscriber != reinterpret_cast<CUpti_SubscriberHandle>(&g_subscribed))
      return CUPTI_ERROR_INVALID_PARAMETER;
    g_subscribed = false;
    g_cb_fn = nullptr;
    g_cb_user = nullptr;
    g_cb_on.clear();
  }
  vgpu::profiling::set_hooks(vgpu::profiling::Hooks{});
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiEnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                                            CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
  return set_callback(enable, subscriber, domain, cbid);
}
VGPU_EXPORT CUptiResult cuptiEnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                                          CUpti_CallbackDomain domain) {
  return set_callback(enable, subscriber, domain, kWholeDomain);
}
VGPU_EXPORT CUptiResult cuptiEnableAllDomains(uint32_t enable, CUpti_SubscriberHandle subscriber) {
  for (auto d : {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_CB_DOMAIN_RESOURCE,
                 CUPTI_CB_DOMAIN_SYNCHRONIZE, CUPTI_CB_DOMAIN_NVTX}) {
    const CUptiResult r = set_callback(enable, subscriber, d, kWholeDomain);
    if (r != CUPTI_SUCCESS) return r;
  }
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
  e.start_ns = e.end_ns = vgpu::profiling::now_ns();
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
    d.start_ns = d.end_ns = vgpu::profiling::now_ns();
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
  e.start_ns = e.end_ns = vgpu::profiling::now_ns();
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
  *id = stream_id_of(reinterpret_cast<uint64_t>(stream));
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiGetStreamIdEx(CUcontext c, CUstream stream, uint8_t, uint32_t* id) {
  return cuptiGetStreamId(c, stream, id);
}
// The names of the runtime functions, for a tool labelling what it receives:
// the versioned spelling, as NVIDIA's reports it (cudaMalloc_v3020).
VGPU_EXPORT CUptiResult cuptiGetCallbackName(CUpti_CallbackDomain domain, uint32_t cbid, const char** name) {
  if (!name) return CUPTI_ERROR_INVALID_PARAMETER;
  *name = nullptr;
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
    if (const char* n = runtime_cbid_name(cbid)) {
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
// Per-function filters on API activity records, a timestamp source of the
// caller's, and device-side timestamps for CUDA events: none is modelled --
// every API record is kept, times are the host's, and the simulated device
// keeps no clock -- so each says so. PyTorch's profiler (Kineto) links all
// four and carries on when told.
VGPU_EXPORT CUptiResult cuptiActivityEnableDriverApi(CUpti_CallbackId, uint8_t) { return CUPTI_ERROR_NOT_SUPPORTED; }
VGPU_EXPORT CUptiResult cuptiActivityEnableRuntimeApi(CUpti_CallbackId, uint8_t) { return CUPTI_ERROR_NOT_SUPPORTED; }
VGPU_EXPORT CUptiResult cuptiActivityRegisterTimestampCallback(CUpti_TimestampCallbackFunc) {
  return CUPTI_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUptiResult cuptiActivityEnableCudaEventDeviceTimestamps(uint8_t) { return CUPTI_ERROR_NOT_SUPPORTED; }
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
VGPU_EXPORT CUptiResult cuptiActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig*, uint32_t) {
  return CUPTI_ERROR_NOT_SUPPORTED;
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
