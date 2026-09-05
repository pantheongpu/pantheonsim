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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vgpu/profiling.hpp"

#ifndef VGPU_EXPORT
#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

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

// Kinds this can actually produce. Enabling anything else succeeds -- refusing
// would stop a profiler that asks for everything and uses what arrives -- but
// nothing will come back for it, which is the honest outcome for a device
// with no timing model and no hardware counters behind these kinds.
bool produced(CUpti_ActivityKind k) {
  return k == CUPTI_ACTIVITY_KIND_KERNEL || k == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
         k == CUPTI_ACTIVITY_KIND_MEMCPY || k == CUPTI_ACTIVITY_KIND_MEMSET;
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

size_t fill_kernel(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(out);
  std::memset(k, 0, sizeof *k);
  k->kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
  k->start = e.start_ns;
  k->end = e.end_ns;
  k->deviceId = e.device;
  k->correlationId = e.correlation;
  k->streamId = static_cast<uint32_t>(e.stream);
  k->gridX = static_cast<int32_t>(e.grid[0]);
  k->gridY = static_cast<int32_t>(e.grid[1]);
  k->gridZ = static_cast<int32_t>(e.grid[2]);
  k->blockX = static_cast<int32_t>(e.block[0]);
  k->blockY = static_cast<int32_t>(e.block[1]);
  k->blockZ = static_cast<int32_t>(e.block[2]);
  k->dynamicSharedMemory = e.shared_bytes;
  k->name = intern(e.name);
  return sizeof *k;
}

size_t fill_memcpy(uint8_t* out, const vgpu::profiling::Event& e) {
  auto* m = reinterpret_cast<CUpti_ActivityMemcpy5*>(out);
  std::memset(m, 0, sizeof *m);
  m->kind = CUPTI_ACTIVITY_KIND_MEMCPY;
  m->start = e.start_ns;
  m->end = e.end_ns;
  m->deviceId = e.device;
  m->correlationId = e.correlation;
  m->streamId = static_cast<uint32_t>(e.stream);
  m->bytes = e.bytes;
  switch (e.copy_kind) {
    case 1: m->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD; break;
    case 2: m->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOH; break;
    case 3: m->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOD; break;
    default: m->copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOH; break;
  }
  return sizeof *m;
}

size_t record_size(const vgpu::profiling::Event& e) {
  return e.kind == vgpu::profiling::EventKind::Kernel ? sizeof(CUpti_ActivityKernel9)
                                                      : sizeof(CUpti_ActivityMemcpy5);
}

bool kind_wanted(const vgpu::profiling::Event& e) {
  if (e.kind == vgpu::profiling::EventKind::Kernel)
    return g_kinds[CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL] || g_kinds[CUPTI_ACTIVITY_KIND_KERNEL];
  return g_kinds[CUPTI_ACTIVITY_KIND_MEMCPY];
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
    offset += cur->kind == CUPTI_ACTIVITY_KIND_MEMCPY ? sizeof(CUpti_ActivityMemcpy5)
                                                      : sizeof(CUpti_ActivityKernel9);
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
  {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto& e : events)
      if (kind_wanted(e)) wanted.push_back(std::move(e));
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
      used += wanted[i].kind == vgpu::profiling::EventKind::Kernel
                  ? fill_kernel(buffer + used, wanted[i])
                  : fill_memcpy(buffer + used, wanted[i]);
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
   Accepted so a subscriber can attach, but no callbacks are delivered: the
   interception points a real CUPTI hooks are inside the driver, and inventing
   them here would mean reporting API calls that did not happen the way the
   consumer is told they did. Activity records carry the same information for
   the kinds above. */

VGPU_EXPORT CUptiResult cuptiSubscribe(CUpti_SubscriberHandle* subscriber, CUpti_CallbackFunc,
                                       void*) {
  if (!subscriber) return CUPTI_ERROR_INVALID_PARAMETER;
  static int token = 0;
  *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(&token);
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle) { return CUPTI_SUCCESS; }
VGPU_EXPORT CUptiResult cuptiEnableCallback(uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain,
                                            CUpti_CallbackId) {
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiEnableDomain(uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain) {
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiEnableAllDomains(uint32_t, CUpti_SubscriberHandle) {
  return CUPTI_SUCCESS;
}

VGPU_EXPORT CUptiResult cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind,
                                                              uint64_t) {
  return CUPTI_SUCCESS;
}
VGPU_EXPORT CUptiResult cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind,
                                                             uint64_t* last) {
  if (last) *last = 0;
  return CUPTI_SUCCESS;
}
