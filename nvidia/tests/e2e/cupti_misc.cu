// The rest of what a profiler asks of CUPTI, one case: which activity kinds can
// be enabled and what is answered when they cannot, which API functions are
// recorded under the per-function switches, the callbacks a subscriber can
// switch on and off, a timestamp source of the program's own, CUDA event
// records, copies between devices and a clock of the program's own. Run against
// NVIDIA's libcupti on a card it gives nvidia/tests/data/cupti_misc.expected
// (two RTX 3060s, CUDA 13.0, which have no peer-to-peer path between them).
//
// What the card does, and the shim reproduces:
//  * the kinds of the legacy counter profiler are refused on a compute
//    capability 7.5 or newer device (CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED),
//    a few more as not compatible, two as invalid; 0 and the count are not
//    compatible; the two kernel kinds cannot both be on;
//  * a function can be recorded alone before its kind was ever enabled, and left
//    out of a kind that is on; enabling the kind forgets those choices; once the
//    kind has been disabled the choices record nothing until it is enabled again;
//  * a callback is enabled by its id or its whole domain, each id has its own
//    state, and a second subscriber is refused;
//  * a timestamp callback is read at the start and end of every API call, and
//    the records of work on the device are put on its clock;
//  * a copy between two devices without a peer path is two copies through the host;
// Not reproduced: kinds above the count (the card answers some and crashes on
// others), and the legacy answer for devices older than 7.5.
#include "cupti_test_util.h"

#include <cuda.h>

#include <initializer_list>

using cupti_test::fmt;
using cupti_test::out;

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_) {                                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#if CUPTI_API_VERSION >= 130000
using KernelRecord = CUpti_ActivityKernel10;
#else
using KernelRecord = CUpti_ActivityKernel9;
#endif
#if CUPTI_API_VERSION >= 26
using CudaEventRecord = CUpti_ActivityCudaEvent2;
#else
using CudaEventRecord = CUpti_ActivityCudaEvent;
#endif
#if CUPTI_API_VERSION >= 22
using OverheadRecord = CUpti_ActivityOverhead3;
#else
using OverheadRecord = CUpti_ActivityOverhead;
#endif

namespace {

const char* result_name(CUptiResult r) {
  const char* s = "?";
  cuptiGetResultString(r, &s);
  return s;
}

bool g_clock_phase = false;
uint64_t g_clock_base = 5'000'000'000'000ull;
uint64_t g_clock_calls = 0, g_clock_last = 0;
uint64_t CUPTIAPI program_clock() {
  g_clock_last = g_clock_base + 1000 * ++g_clock_calls;
  return g_clock_last;
}

int g_clock_api_in_range = 0, g_clock_api_total = 0, g_clock_work_in_range = 0, g_clock_work_total = 0;

void CUPTIAPI on_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    const auto* cb = static_cast<const CUpti_CallbackData*>(data);
    if (cb->callbackSite == CUPTI_API_ENTER) out(std::string("  callback: ") + cb->functionName);
  } else if (domain == CUPTI_CB_DOMAIN_RESOURCE && cbid == CUPTI_CBID_RESOURCE_STREAM_CREATED) {
    out("  callback: stream created");
  }
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    switch (static_cast<int>(r->kind)) {
      case CUPTI_ACTIVITY_KIND_RUNTIME:
      case CUPTI_ACTIVITY_KIND_DRIVER: {
        const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
        const bool runtime = r->kind == CUPTI_ACTIVITY_KIND_RUNTIME;
        const char* name = nullptr;
        cuptiGetCallbackName(runtime ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API, a->cbid, &name);
        if (g_clock_phase) {
          ++g_clock_api_total;
          if (a->start > g_clock_base && a->end > a->start && a->end <= g_clock_last) ++g_clock_api_in_range;
        } else if (runtime || (name && std::strstr(name, "cuMem") == name)) {
          // The driver functions the runtime itself calls underneath are the
          // driver's business; the ones a program makes are cuMem*.
          out(fmt("  %s %s", runtime ? "runtime" : "driver", name ? name : "?"));
        }
        break;
      }
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const auto* k = reinterpret_cast<const KernelRecord*>(r);
        if (g_clock_phase) {
          ++g_clock_work_total;
          if (k->start > g_clock_base && k->end >= k->start && k->end <= g_clock_last) ++g_clock_work_in_range;
        }
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const auto* m = reinterpret_cast<const MemcpyRecord*>(r);
        if (g_clock_phase) {
          ++g_clock_work_total;
          if (m->start > g_clock_base && m->end >= m->start && m->end <= g_clock_last) ++g_clock_work_in_range;
        } else {
          out(fmt("  memcpy kind=%d src=%s dst=%s bytes=%llu async=%d device=%u context=#x%u stream=#s%u corr=#c%u",
                  (int)m->copyKind, cupti_test::mem_kind(m->srcKind), cupti_test::mem_kind(m->dstKind),
                  (unsigned long long)m->bytes, (m->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) ? 1 : 0, m->deviceId,
                  m->contextId, m->streamId, m->correlationId));
        }
        break;
      }
      case 36: {   // CUPTI_ACTIVITY_KIND_CUDA_EVENT
        const auto* e = reinterpret_cast<const CudaEventRecord*>(r);
        out(fmt("  cuda-event id=#e%u stream=#s%u corr=#c%u", e->eventId, e->streamId, e->correlationId));
#if CUPTI_API_VERSION >= 26
        out(fmt("    device timestamp %s, sync id %llu", e->deviceTimestamp ? "set" : "none",
                (unsigned long long)e->cudaEventSyncId));
#endif
        break;
      }
      case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
        const auto* s = reinterpret_cast<const SyncRecord*>(r);
        out(fmt("  sync type=%d event=#e%u corr=#c%u", (int)s->type, s->cudaEventId, s->correlationId));
        break;
      }
      default: break;
    }
  }
}

void section(const char* name) {
  out(std::string("# ") + name);
  cuptiActivityFlushAll(0);
}

#define SHOW(call)                                                                       \
  do {                                                                                   \
    const CUptiResult r_ = (call);                                                       \
    out(fmt("%s -> %s", #call, result_name(r_)));                                        \
  } while (0)

}  // namespace

__global__ void touch(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.f;
}

int main() {
  float* d = nullptr;
  float host[64] = {};
  // ---- which kinds can be enabled ----
  out("# enabling kinds");
  // The unified-memory counter kind answers by whether the machine supports
  // unified memory page faults, which differs between the machines this is
  // run on, and is left out.
  for (int k = 0; k <= 56; ++k) {
    if (k == 25) continue;
    const CUptiResult r = cuptiActivityEnable(static_cast<CUpti_ActivityKind>(k));
    out(fmt("enable kind %d -> %s", k, result_name(r)));
    if (r == CUPTI_SUCCESS) cuptiActivityDisable(static_cast<CUpti_ActivityKind>(k));
  }
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
#if CUPTI_API_VERSION >= 130000
  SHOW(cuptiActivityEnableCudaEventDeviceTimestamps(1));
  SHOW(cuptiActivityEnableCudaEventDeviceTimestamps(0));
#endif

  // ---- result strings ----
  out("# result strings");
  for (int code : {-1, 0, 1, 12, 13, 14, 18, 21, 27, 30, 36, 37, 38, 39, 42, 43, 46, 47, 100, 999, 1000}) {
    const char* s = "?";
    const CUptiResult r = cuptiGetResultString(static_cast<CUptiResult>(code), &s);
    out(fmt("result %d: %s (%d)", code, s, (int)r));
  }

  // ---- callbacks that can be switched on and off ----
  out("# callbacks");
  CUpti_SubscriberHandle sub;
  SHOW(cuptiSubscribe(&sub, on_callback, nullptr));
  CUpti_SubscriberHandle second;
  SHOW(cuptiSubscribe(&second, on_callback, nullptr));
  SHOW(cuptiSubscribe(nullptr, on_callback, nullptr));
  const auto bad = reinterpret_cast<CUpti_SubscriberHandle>(static_cast<uintptr_t>(0x1234));
  SHOW(cuptiEnableCallback(1, bad, CUPTI_CB_DOMAIN_RUNTIME_API, 1));
  SHOW(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_INVALID, 1));
  SHOW(cuptiEnableCallback(1, sub, static_cast<CUpti_CallbackDomain>(99), 1));
  SHOW(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API, 0));
  SHOW(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API, 99999));
  SHOW(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RESOURCE, 0));
  SHOW(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RESOURCE, 99));
  SHOW(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_SYNCHRONIZE, 3));
  SHOW(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_NVTX, 99999));
  SHOW(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_INVALID));
  SHOW(cuptiEnableDomain(1, sub, static_cast<CUpti_CallbackDomain>(99)));
  SHOW(cuptiEnableDomain(1, bad, CUPTI_CB_DOMAIN_RUNTIME_API));
  SHOW(cuptiEnableAllDomains(1, bad));
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  out("- the domain on, then cudaMalloc off");
  SHOW(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API));
  SHOW(cuptiEnableCallback(0, sub, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020));
  CK(cudaMalloc(&d, 4096));
  CK(cudaFree(d));
  out("- the domain off, then cudaFree on");
  SHOW(cuptiEnableDomain(0, sub, CUPTI_CB_DOMAIN_RUNTIME_API));
  SHOW(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020));
  CK(cudaMalloc(&d, 4096));
  CK(cudaFree(d));
  out("- the domain on again");
  SHOW(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API));
  CK(cudaMalloc(&d, 4096));
  CK(cudaFree(d));
  out("- all off");
  SHOW(cuptiEnableAllDomains(0, sub));
  CK(cudaMalloc(&d, 4096));
  CK(cudaFree(d));
  SHOW(cuptiUnsubscribe(sub));
  SHOW(cuptiUnsubscribe(sub));
  SHOW(cuptiSubscribe(&second, on_callback, nullptr));
  SHOW(cuptiUnsubscribe(second));

#if CUPTI_API_VERSION >= 24
  // ---- recording single functions ----
  out("# per-function records");
  cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed);
  SHOW(cuptiActivityEnableRuntimeApi(0, 1));
  SHOW(cuptiActivityEnableRuntimeApi(99999, 1));
  SHOW(cuptiActivityEnableDriverApi(0, 1));
  SHOW(cuptiActivityEnableDriverApi(99999, 1));
  auto traffic = [&]() -> int {
    CK(cudaMalloc(&d, 4096));
    CK(cudaMemcpy(d, host, sizeof host, cudaMemcpyHostToDevice));
    CK(cudaFree(d));
    return 0;
  };
  out("- only cudaMalloc");
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020, 1));
  if (traffic()) return 1;
  section("only cudaMalloc");
  out("- and cudaFree");
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020, 1));
  if (traffic()) return 1;
  section("malloc and free");
  out("- the kind on, cudaMalloc off");
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020, 0));
  if (traffic()) return 1;
  section("kind on, malloc off");
  out("- the kind on again");
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  if (traffic()) return 1;
  section("kind on again");
  out("- cudaFree off");
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020, 0));
  if (traffic()) return 1;
  section("free off");
  out("- the kind off, then cudaMemcpy on");
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020, 1));
  if (traffic()) return 1;
  section("kind off");
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));

  out("- the driver's, with the kind never on");
  {
    CUdeviceptr p;
    SHOW(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2, 1));
    cuMemAlloc(&p, 4096);
    cuMemsetD8(p, 0, 4096);
    cuMemFree(p);
    section("driver: only cuMemAlloc");
    SHOW(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2, 1));
    cuMemAlloc(&p, 4096);
    cuMemsetD8(p, 0, 4096);
    cuMemFree(p);
    section("driver: alloc and free");
    SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    SHOW(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2, 0));
    cuMemAlloc(&p, 4096);
    cuMemsetD8(p, 0, 4096);
    cuMemFree(p);
    section("driver: kind on, memset off");
    SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  }

#endif

  // ---- CUDA events ----
  out("# cuda events");
  for (int kind : std::initializer_list<int>{36, CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_SYNCHRONIZATION})
    cuptiActivityEnable(static_cast<CUpti_ActivityKind>(kind));
  {
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e1, e2;
    CK(cudaEventCreate(&e1));
    CK(cudaEventCreate(&e2));
    CK(cudaEventRecord(e1, s));
    touch<<<1, 32, 0, s>>>(nullptr, 0);
    CK(cudaEventRecord(e2, s));
    CK(cudaEventSynchronize(e2));
    CK(cudaStreamWaitEvent(nullptr, e2, 0));
    CK(cudaEventRecord(e1, nullptr));
    CK(cudaDeviceSynchronize());
    section("without device timestamps");
#if CUPTI_API_VERSION >= 130000
    SHOW(cuptiActivityEnableCudaEventDeviceTimestamps(1));
    CK(cudaEventRecord(e1, s));
    CK(cudaDeviceSynchronize());
    section("with device timestamps");
    SHOW(cuptiActivityEnableCudaEventDeviceTimestamps(0));
#endif
    CK(cudaEventDestroy(e1));
    CK(cudaEventDestroy(e2));
    CK(cudaStreamDestroy(s));
  }
  for (int kind : std::initializer_list<int>{36, CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_SYNCHRONIZATION})
    cuptiActivityDisable(static_cast<CUpti_ActivityKind>(kind));

  // ---- a copy between two devices ----
  int devices = 0;
  CK(cudaGetDeviceCount(&devices));
  if (devices >= 2) {
    out("# copies between devices");
    for (int kind : {CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_RUNTIME})
      cuptiActivityEnable(static_cast<CUpti_ActivityKind>(kind));
    float *a = nullptr, *b = nullptr;
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&a, 4096));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&b, 4096));
    CK(cudaSetDevice(0));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    section("setup");
    CK(cudaMemcpyPeer(b, 1, a, 0, 4096));
    CK(cudaMemcpyPeerAsync(b, 1, a, 0, 4096, s));
    CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(b, a, 4096, cudaMemcpyDefault));
    CK(cudaMemcpyAsync(b, a, 4096, cudaMemcpyDeviceToDevice, s));
    CK(cudaStreamSynchronize(s));
    section("copies");
    CK(cudaStreamDestroy(s));
    CK(cudaFree(a));
    CK(cudaSetDevice(1));
    CK(cudaFree(b));
    CK(cudaSetDevice(0));
    for (int kind : {CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_RUNTIME})
      cuptiActivityDisable(static_cast<CUpti_ActivityKind>(kind));
  }

  // ---- a clock of the program's own: it cannot be taken back ----
  out("# the program's clock");
  cuptiActivityFlushAll(0);   // what is pending was stamped before the clock was the program's
  SHOW(cuptiActivityRegisterTimestampCallback(nullptr));
  SHOW(cuptiActivityRegisterTimestampCallback(program_clock));
  g_clock_phase = true;
  CK(cudaMalloc(&d, 4096));
  for (int kind : {CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL}) {
    cuptiActivityEnable(static_cast<CUpti_ActivityKind>(kind));
  }
  CK(cudaMemcpy(d, host, sizeof host, cudaMemcpyHostToDevice));
  touch<<<1, 32>>>(d, 32);
  CK(cudaDeviceSynchronize());
  cuptiActivityFlushAll(0);
  out(fmt("API records on the program's clock: %s", g_clock_api_total && g_clock_api_in_range == g_clock_api_total ? "all" : "not all"));
  out(fmt("work records on the program's clock: %s", g_clock_work_total && g_clock_work_in_range == g_clock_work_total ? "all" : "not all"));
  uint64_t t0 = 0, t1 = 0;
  cuptiGetTimestamp(&t0);
  cuptiGetTimestamp(&t1);
  out(fmt("cuptiGetTimestamp reads it: %s, one call apart by %llu", t0 > g_clock_base ? "yes" : "no",
          (unsigned long long)(t1 - t0)));

  CK(cudaFree(d));
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
