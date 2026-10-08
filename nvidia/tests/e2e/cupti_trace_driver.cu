// A program that uses only the CUDA driver API, traces itself through CUPTI
// and prints the trace in a form that does not depend on time, addresses or
// the numbers a CUPTI picks for its own ids, so the same program can be run
// against NVIDIA's libcupti on a card and against this project's, and the two
// outputs compared line by line. It is cupti_trace.cu for the driver domain.
//
// What is printed, and what is not:
//   * Callback API: every driver-API call the program makes, as ENTER and EXIT,
//     with the function name NVIDIA's CUPTI reports (cuMemAlloc_v2, not
//     cuMemAlloc), the return value and, for the calls it is checked on, the
//     parameters; the resource and synchronize callbacks between them.
//   * Activity API: DRIVER records (named through cuptiGetCallbackName), and the
//     kernels, copies, fills, waits and streams they issued, with launch
//     geometry, byte counts, copy direction and memory kinds. Timestamps are
//     checked for order only and never printed. Correlation and stream ids are
//     printed as their rank among those the run saw ("c3" is the fourth call).
//   * Not printed: register counts and the device's memory size, and the
//     module-load callbacks, which a real driver raises when it lazily loads a
//     kernel's code.
// The kernels are PTX in a string, loaded with cuModuleLoadData, so nothing
// here needs a device compiler and nothing but the driver API is called.
#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>

#include <cuda.h>
#include <cupti.h>

// The toolkit's header spells the context-creation and elapsed-time calls with
// the newest version of each, which differs between toolkits and would make
// the trace depend on which one compiled this. Both are called by the spelling
// every driver has exported since before the versions began.
extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext*, unsigned int, CUdevice);
#ifdef cuEventElapsedTime
#undef cuEventElapsedTime
#endif
extern "C" CUresult CUDAAPI cuEventElapsedTime(float*, CUevent, CUevent);

#define CK(x)                                                                          \
  do {                                                                                 \
    CUresult e_ = (x);                                                                 \
    if (e_) {                                                                          \
      const char* s_ = "?";                                                            \
      cuGetErrorName(e_, &s_);                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, s_);                \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)
#define CP(x)                                                                          \
  do {                                                                                 \
    CUptiResult r_ = (x);                                                              \
    if (r_ != CUPTI_SUCCESS) {                                                         \
      const char* s_ = "?";                                                            \
      cuptiGetResultString(r_, &s_);                                                   \
      std::fprintf(stderr, "FAIL %s:%d: CUPTI %s\n", __FILE__, __LINE__, s_);          \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

// The newest layout of each record this toolkit defines; the library under
// test hands out the same ones, which is the point of reading them here.
#if CUPTI_API_VERSION >= 130200
using KernelRecord = CUpti_ActivityKernel11;
using DeviceRecord = CUpti_ActivityDevice6;
#elif CUPTI_API_VERSION >= 130000
using KernelRecord = CUpti_ActivityKernel10;
using DeviceRecord = CUpti_ActivityDevice5;
#elif CUPTI_API_VERSION >= 26
using KernelRecord = CUpti_ActivityKernel9;
using DeviceRecord = CUpti_ActivityDevice5;
#else
using KernelRecord = CUpti_ActivityKernel9;
using DeviceRecord = CUpti_ActivityDevice4;
#endif
#if CUPTI_API_VERSION >= 26
using MemcpyRecord = CUpti_ActivityMemcpy6;
using SyncRecord = CUpti_ActivitySynchronization2;
#else
using MemcpyRecord = CUpti_ActivityMemcpy5;
using SyncRecord = CUpti_ActivitySynchronization;
#endif

namespace {

// Lines carry their raw correlation and stream ids as "#c<raw>" and "#s<raw>"
// until the end of the run, when every id is known and each becomes its rank.
struct Line {
  std::string text;
  uint64_t corr = 0;   // activity records sort by the call that issued them
  int rank = 0;        // ... the work ahead of the call itself
};
std::vector<Line> g_setup;       // callbacks of the warm-up
std::vector<Line> g_lines;       // callbacks of the workload, in the order they happened
std::vector<Line> g_resources;   // device and context records
std::vector<Line> g_records;     // activity records
std::map<int, int> g_other;      // kinds seen but not printed
int g_bad_time = 0;

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

void rewrite_ids(std::vector<Line>* all[], int n) {
  std::map<uint64_t, int> rank[2];   // correlation, stream
  const char tags[2] = {'c', 's'};
  for (int i = 0; i < n; ++i)
    for (const auto& l : *all[i])
      for (int t = 0; t < 2; ++t)
        for (size_t at = 0; (at = l.text.find(std::string("#") + tags[t], at)) != std::string::npos;) {
          at += 2;
          rank[t][std::strtoull(l.text.c_str() + at, nullptr, 10)] = 0;
        }
  for (auto& m : rank) {
    int k = 0;
    for (auto& entry : m) entry.second = k++;
  }
  for (int i = 0; i < n; ++i)
    for (auto& l : *all[i])
      for (int t = 0; t < 2; ++t) {
        const std::string key = std::string("#") + tags[t];
        for (size_t at = 0; (at = l.text.find(key, at)) != std::string::npos;) {
          size_t end = at + 2;
          while (end < l.text.size() && std::isdigit(static_cast<unsigned char>(l.text[end]))) ++end;
          const uint64_t raw = std::strtoull(l.text.c_str() + at + 2, nullptr, 10);
          const std::string rep = std::string(1, tags[t]) + std::to_string(rank[t].at(raw));
          l.text.replace(at, end - at, rep);
          at += rep.size();
        }
      }
}

// ---- callbacks ----
void CUPTIAPI on_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    // Module loading is raised when a real driver lazily loads a kernel's code;
    // the simulator loads whole modules at once and does not raise it.
    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED || cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED ||
        cbid == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING)
      return;
    g_lines.push_back({fmt("CB resource %d", (int)cbid)});
    return;
  }
  if (domain == CUPTI_CB_DOMAIN_SYNCHRONIZE) {
    g_lines.push_back({fmt("CB synchronize %d", (int)cbid)});
    return;
  }
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
  const auto* cb = static_cast<const CUpti_CallbackData*>(data);
  const bool enter = cb->callbackSite == CUPTI_API_ENTER;
  std::string line = fmt("CB driver %s %s", cb->functionName ? cb->functionName : "?", enter ? "ENTER" : "EXIT");
  if (!enter) line += fmt(" ret=%d", static_cast<int>(*static_cast<const CUresult*>(cb->functionReturnValue)));
  const void* fp = cb->functionParams;
  switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuInit: line += fmt(" flags=%u", static_cast<const cuInit_params*>(fp)->Flags); break;
    case CUPTI_DRIVER_TRACE_CBID_cuDeviceGet:
      line += fmt(" ordinal=%d", static_cast<const cuDeviceGet_params*>(fp)->ordinal);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuDeviceGetName:
      line += fmt(" len=%d dev=%d", static_cast<const cuDeviceGetName_params*>(fp)->len,
                  (int)static_cast<const cuDeviceGetName_params*>(fp)->dev);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuDeviceGetAttribute: {
      const auto* p = static_cast<const cuDeviceGetAttribute_params*>(fp);
      line += fmt(" attrib=%d dev=%d", (int)p->attrib, (int)p->dev);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx:
      line += fmt(" options=%u", static_cast<const cuModuleLoadDataEx_params*>(fp)->numOptions);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority: {
      const auto* p = static_cast<const cuStreamCreateWithPriority_params*>(fp);
      line += fmt(" flags=%u priority=%d", p->flags, p->priority);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2: {
      const auto* p = static_cast<const cuCtxCreate_v2_params*>(fp);
      line += fmt(" flags=%u dev=%d", p->flags, (int)p->dev);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
      line += fmt(" size=%zu", static_cast<const cuMemAlloc_v2_params*>(fp)->bytesize);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2:
      line += fmt(" size=%zu", static_cast<const cuMemAllocHost_v2_params*>(fp)->bytesize);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc: {
      const auto* p = static_cast<const cuMemHostAlloc_params*>(fp);
      line += fmt(" size=%zu flags=%u", p->bytesize, p->Flags);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
      line += fmt(" count=%zu", static_cast<const cuMemcpyHtoD_v2_params*>(fp)->ByteCount);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
      line += fmt(" count=%zu", static_cast<const cuMemcpyDtoH_v2_params*>(fp)->ByteCount);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
      line += fmt(" count=%zu", static_cast<const cuMemcpyDtoD_v2_params*>(fp)->ByteCount);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
      line += fmt(" count=%zu", static_cast<const cuMemcpyHtoDAsync_v2_params*>(fp)->ByteCount);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
      line += fmt(" count=%zu", static_cast<const cuMemcpyDtoHAsync_v2_params*>(fp)->ByteCount);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
      line += fmt(" count=%zu", static_cast<const cuMemcpyDtoDAsync_v2_params*>(fp)->ByteCount);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2: {
      const auto* p = static_cast<const cuMemsetD8_v2_params*>(fp);
      line += fmt(" value=%u n=%zu", (unsigned)p->uc, p->N);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2: {
      const auto* p = static_cast<const cuMemsetD16_v2_params*>(fp);
      line += fmt(" value=%u n=%zu", (unsigned)p->us, p->N);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2: {
      const auto* p = static_cast<const cuMemsetD32_v2_params*>(fp);
      line += fmt(" value=%u n=%zu", p->ui, p->N);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async: {
      const auto* p = static_cast<const cuMemsetD8Async_params*>(fp);
      line += fmt(" value=%u n=%zu", (unsigned)p->uc, p->N);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async: {
      const auto* p = static_cast<const cuMemsetD16Async_params*>(fp);
      line += fmt(" value=%u n=%zu", (unsigned)p->us, p->N);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async: {
      const auto* p = static_cast<const cuMemsetD32Async_params*>(fp);
      line += fmt(" value=%u n=%zu", p->ui, p->N);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
      line += fmt(" name=%s", static_cast<const cuModuleGetFunction_params*>(fp)->name);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: {
      const auto* p = static_cast<const cuLaunchKernel_params*>(fp);
      line += fmt(" grid=%ux%ux%u block=%ux%ux%u shared=%u", p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX,
                  p->blockDimY, p->blockDimZ, p->sharedMemBytes);
      if (cb->symbolName) line += fmt(" symbol=%s", cb->symbolName);
      break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamCreate:
      line += fmt(" flags=%u", static_cast<const cuStreamCreate_params*>(fp)->Flags);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuEventCreate:
      line += fmt(" flags=%u", static_cast<const cuEventCreate_params*>(fp)->Flags);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent:
      line += fmt(" flags=%u", static_cast<const cuStreamWaitEvent_params*>(fp)->Flags);
      break;
    default: break;
  }
  line += fmt(" corr=#c%u", cb->correlationId);
  g_lines.push_back({line});
}

// ---- activity ----
const char* copy_kind(uint32_t k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "PtoP";
    default: return "?";
  }
}
const char* mem_kind(uint32_t k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE: return "pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED: return "pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE: return "device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY: return "array";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED: return "managed";
    default: return "unknown";
  }
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
  static uint8_t storage[1 << 20] __attribute__((aligned(8)));
  *buffer = storage;
  *size = sizeof storage;
  *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    switch (r->kind) {
      case CUPTI_ACTIVITY_KIND_KERNEL:
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const auto* k = reinterpret_cast<const KernelRecord*>(r);
        if (k->end < k->start) ++g_bad_time;
        g_records.push_back({fmt("REC kernel name=%s grid=%dx%dx%d block=%dx%dx%d dynShared=%d statShared=%d stream=#s%u corr=#c%u",
                                 k->name ? k->name : "?", k->gridX, k->gridY, k->gridZ, k->blockX, k->blockY,
                                 k->blockZ, k->dynamicSharedMemory, k->staticSharedMemory, k->streamId,
                                 k->correlationId), k->correlationId, 0});
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const auto* m = reinterpret_cast<const MemcpyRecord*>(r);
        if (m->end < m->start) ++g_bad_time;
        g_records.push_back({fmt("REC memcpy kind=%s src=%s dst=%s bytes=%llu async=%d stream=#s%u corr=#c%u",
                                 copy_kind(m->copyKind), mem_kind(m->srcKind), mem_kind(m->dstKind),
                                 (unsigned long long)m->bytes, (m->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) ? 1 : 0,
                                 m->streamId, m->correlationId), m->correlationId, 0});
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMSET: {
        const auto* m = reinterpret_cast<const CUpti_ActivityMemset4*>(r);
        if (m->end < m->start) ++g_bad_time;
        g_records.push_back({fmt("REC memset value=%u bytes=%llu kind=%s async=%d stream=#s%u corr=#c%u", m->value,
                                 (unsigned long long)m->bytes, mem_kind(m->memoryKind),
                                 (m->flags & CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC) ? 1 : 0, m->streamId,
                                 m->correlationId), m->correlationId, 0});
        break;
      }
      case CUPTI_ACTIVITY_KIND_DRIVER: {
        const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
        if (a->end < a->start) ++g_bad_time;
        const char* name = nullptr;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, a->cbid, &name);
        g_records.push_back({fmt("REC driver %s ret=%u corr=#c%u", name ? name : "?", a->returnValue, a->correlationId),
                             a->correlationId, 1});
        break;
      }
      case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
        const auto* s = reinterpret_cast<const SyncRecord*>(r);
        g_records.push_back({fmt("REC sync type=%d stream=%s corr=#c%u", (int)s->type,
                                 s->streamId == CUPTI_SYNCHRONIZATION_INVALID_VALUE ? "none" : "set", s->correlationId),
                             s->correlationId, 0});
        break;
      }
      case CUPTI_ACTIVITY_KIND_DEVICE: {
        const auto* d = reinterpret_cast<const DeviceRecord*>(r);
        // Device 0 only: the card has a second one and the simulator, by
        // default, one.
        if (d->id == 0)
          g_resources.push_back({fmt("REC device id=%u name=%s cc=%u.%u sms=%u", d->id, d->name ? d->name : "?",
                                     d->computeCapabilityMajor, d->computeCapabilityMinor, d->numMultiprocessors)});
        break;
      }
      case CUPTI_ACTIVITY_KIND_CONTEXT: {
        const auto* c = reinterpret_cast<const CUpti_ActivityContext*>(r);
        g_resources.push_back({fmt("REC context device=%u", c->deviceId)});
        break;
      }
      case CUPTI_ACTIVITY_KIND_STREAM: {
        const auto* t = reinterpret_cast<const CUpti_ActivityStream*>(r);
        g_records.push_back({fmt("REC stream flag=%d priority=%u corr=#c%u", (int)t->flag, t->priority,
                                 t->correlationId), t->correlationId, 0});
        break;
      }
      default: ++g_other[r->kind]; break;
    }
  }
}

// Two kernels, as PTX text. scale multiplies n floats in place; reverse_block
// reverses each block's slice through dynamic shared memory and declares a
// static shared array as well, which the activity record reports.
const char kPtx[] = R"PTX(
.version 7.0
.target sm_75
.address_size 64

.extern .shared .align 4 .b8 tile[];

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

.visible .entry reverse_block(.param .u64 p, .param .u32 n)
{
  .reg .pred %q<4>;
  .reg .b32 %r<14>;
  .reg .b64 %rd<8>;
  .reg .f32 %f<4>;
  .shared .align 4 .b8 stat[32];
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [n];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r5, %r2, %r3, %r4;
  mov.f32 %f1, 0f00000000;
  setp.ge.u32 %q1, %r4, 8;
  @%q1 bra NOSTAT;
  mov.u64 %rd3, stat;
  mul.wide.u32 %rd4, %r4, 4;
  add.s64 %rd3, %rd3, %rd4;
  st.shared.f32 [%rd3], %f1;
NOSTAT:
  mov.f32 %f2, 0f00000000;
  setp.ge.s32 %q2, %r5, %r1;
  @%q2 bra NOLOAD;
  mul.wide.s32 %rd5, %r5, 4;
  add.s64 %rd5, %rd2, %rd5;
  ld.global.f32 %f2, [%rd5];
NOLOAD:
  mov.u64 %rd6, tile;
  mul.wide.u32 %rd7, %r4, 4;
  add.s64 %rd6, %rd6, %rd7;
  st.shared.f32 [%rd6], %f2;
  bar.sync 0;
  @%q2 bra DONE2;
  sub.s32 %r6, %r3, 1;
  sub.s32 %r7, %r6, %r4;
  mov.u64 %rd6, tile;
  mul.wide.u32 %rd7, %r7, 4;
  add.s64 %rd6, %rd6, %rd7;
  ld.shared.f32 %f3, [%rd6];
  mul.wide.s32 %rd5, %r5, 4;
  add.s64 %rd5, %rd2, %rd5;
  st.global.f32 [%rd5], %f3;
DONE2:
  ret;
}
)PTX";

}  // namespace

int main() {
  CUpti_SubscriberHandle sub;
  CP(cuptiSubscribe(&sub, on_callback, nullptr));
  CP(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_DRIVER_API));
  CP(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RESOURCE));
  CP(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_SYNCHRONIZE));
  CP(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
  for (auto kind : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
                    CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, CUPTI_ACTIVITY_KIND_DEVICE,
                    CUPTI_ACTIVITY_KIND_CONTEXT, CUPTI_ACTIVITY_KIND_STREAM})
    CP(cuptiActivityEnable(kind));

  // Everything below the warm-up is the workload; the context is made first so
  // that initialisation is not part of the trace.
  CUdevice dev;
  CUcontext ctx;
  CK(cuInit(0));
  CK(cuDeviceGet(&dev, 0));
  CK(cuCtxCreate_v2(&ctx, 0, dev));
  g_setup = g_lines;
  g_lines.clear();
  CP(cuptiActivityFlushAll(0));
  g_records.clear();

  const int n = 512;
  const size_t bytes = n * sizeof(float);
  CUmodule mod;
  CUfunction scale, reverse_block;
  CUstream s1;
  CUevent ev, t0, t1;
  CUdeviceptr a = 0, b = 0;
  void *pinned = nullptr, *pinned2 = nullptr;
  float host[n], check[n], out[n];

  CK(cuInit(0));   // a second one: the first, which brought the driver up, is not reported
  int devices = 0, warp = 0;
  char name[64];
  size_t total = 0;
  CK(cuDeviceGetCount(&devices));
  CK(cuDeviceGetName(name, sizeof name, dev));
  CK(cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev));
  CK(cuDeviceTotalMem(&total, dev));
  if (devices < 1 || warp != 32 || total == 0 || !name[0]) {
    std::fprintf(stderr, "FAIL: device queries (%d devices, warp %d)\n", devices, warp);
    return 1;
  }

  CK(cuModuleLoadData(&mod, kPtx));
  CK(cuModuleGetFunction(&scale, mod, "scale"));
  CK(cuModuleGetFunction(&reverse_block, mod, "reverse_block"));
  {
    // The same module a second and a third way: from a file, and with options.
    char path[] = "/tmp/vgpu_cupti_trace_driver_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return 1;
    const ssize_t wrote = write(fd, kPtx, sizeof kPtx);
    close(fd);
    CUmodule from_file, with_options;
    const CUresult loaded = wrote == static_cast<ssize_t>(sizeof kPtx) ? cuModuleLoad(&from_file, path) : CUDA_ERROR_UNKNOWN;
    unlink(path);
    CK(loaded);
    CK(cuModuleUnload(from_file));
    CK(cuModuleLoadDataEx(&with_options, kPtx, 0, nullptr, nullptr));
    CK(cuModuleUnload(with_options));
  }
  CUstream s2;
  CK(cuStreamCreate(&s1, 0));
  CK(cuStreamCreateWithPriority(&s2, CU_STREAM_NON_BLOCKING, -1));
  CK(cuEventCreate(&ev, 0));
  CK(cuEventCreate(&t0, 0));
  CK(cuEventCreate(&t1, 0));
  CK(cuMemAlloc(&a, bytes));
  CK(cuMemAlloc(&b, bytes));
  CK(cuMemAllocHost(&pinned, bytes));
  CK(cuMemHostAlloc(&pinned2, bytes, 0));
  for (int i = 0; i < n; ++i) {
    host[i] = float(i);
    static_cast<float*>(pinned)[i] = 1.f;
  }

  CK(cuMemcpyHtoD(a, host, bytes));
  CK(cuMemcpyHtoDAsync(b, pinned, bytes, s1));
  CK(cuMemsetD8(b, 0, 64));                        // b[0..15] = 0
  CK(cuMemsetD16(b + 500 * 4, 0, 8));              // b[500..503] = 0
  CK(cuMemsetD32(b + 510 * 4, 0, 1));              // b[510] = 0
  CK(cuMemsetD8Async(b + 508 * 4, 0, 8, s1));      // b[508..509] = 0
  CK(cuMemsetD16Async(b + 504 * 4, 0, 4, s1));     // b[504..505] = 0
  CK(cuMemsetD32Async(a, 0x3f800000u, 32, s1));    // a[0..31] = 1.0f

  float factor = 2.f;
  int count = n;
  void* scale_args[] = {&a, &factor, &count};
  void* rev_args[] = {&b, &count};
  CK(cuEventRecord(t0, s1));
  CK(cuLaunchKernel(scale, 2, 1, 1, 256, 1, 1, 0, 0, scale_args, nullptr));
  CK(cuLaunchKernel(reverse_block, 4, 1, 1, 128, 1, 1, 128 * sizeof(float), s1, rev_args, nullptr));
  CK(cuEventRecord(t1, s1));
  CK(cuEventRecord(ev, s1));
  CK(cuStreamWaitEvent(0, ev, 0));
  CK(cuEventSynchronize(ev));
  CK(cuStreamSynchronize(s1));
  float ms = -1.f;
  CK(cuEventElapsedTime(&ms, t0, t1));
  CK(cuStreamQuery(s1));
  CK(cuEventQuery(ev));

  CK(cuMemcpyDtoH(out, a, bytes));                 // a after the fill and the scale
  CK(cuMemcpyDtoHAsync(pinned2, b, bytes, s1));    // b after the fills and the reverse
  CK(cuStreamSynchronize(s1));
  CK(cuMemcpyDtoD(b, a, bytes));
  CK(cuMemcpyDtoDAsync(a, b, bytes, s1));
  CK(cuCtxSynchronize());

  // What the program computed, checked against the same arithmetic on the host.
  int bad = 0;
  for (int i = 0; i < n; ++i) check[i] = i < 32 ? 2.f : 2.f * float(i);
  for (int i = 0; i < n; ++i) bad += out[i] != check[i];
  float expect_b[n];
  for (int i = 0; i < n; ++i) expect_b[i] = 1.f;
  for (int i = 0; i < 16; ++i) expect_b[i] = 0.f;
  for (int i = 500; i < 506; ++i) expect_b[i] = 0.f;
  for (int i = 508; i < 511; ++i) expect_b[i] = 0.f;
  for (int blk = 0; blk < 4; ++blk) std::reverse(expect_b + blk * 128, expect_b + blk * 128 + 128);
  for (int i = 0; i < n; ++i) bad += static_cast<float*>(pinned2)[i] != expect_b[i];
  if (bad) {
    std::fprintf(stderr, "FAIL: %d wrong results\n", bad);
    return 1;
  }

  CK(cuMemFreeHost(pinned));
  CK(cuMemFreeHost(pinned2));
  CK(cuMemFree(a));
  CK(cuMemFree(b));
  CK(cuEventDestroy(ev));
  CK(cuEventDestroy(t0));
  CK(cuEventDestroy(t1));
  CK(cuStreamDestroy(s1));
  CK(cuStreamDestroy(s2));
  CK(cuModuleUnload(mod));
  CUcontext current = nullptr;
  CK(cuCtxGetCurrent(&current));
  CK(cuCtxSetCurrent(nullptr));
  CK(cuCtxSetCurrent(ctx));
  CK(cuCtxDestroy(ctx));

  CP(cuptiActivityFlushAll(0));
  CP(cuptiUnsubscribe(sub));

  // Records arrive in the order they completed, which for work on a second
  // stream is not fixed; they are listed in the order of the calls that issued
  // them, the work ahead of its call.
  std::stable_sort(g_records.begin(), g_records.end(), [](const Line& x, const Line& y) {
    return x.corr != y.corr ? x.corr < y.corr : x.rank < y.rank;
  });
  std::vector<Line>* all[] = {&g_setup, &g_lines, &g_resources, &g_records};
  rewrite_ids(all, 4);

  std::printf("# setup\n");
  for (const auto& l : g_setup) std::printf("%s\n", l.text.c_str());
  std::printf("# callbacks\n");
  for (const auto& l : g_lines) std::printf("%s\n", l.text.c_str());
  std::printf("# resources\n");
  for (const auto& l : g_resources) std::printf("%s\n", l.text.c_str());
  std::printf("# activity\n");
  for (const auto& l : g_records) std::printf("%s\n", l.text.c_str());
  for (const auto& kc : g_other) std::fprintf(stderr, "unprinted kind %d x%d\n", kc.first, kc.second);
  std::printf("# end (%d records ending before they start)\n", g_bad_time);
  return g_bad_time ? 1 : 0;
}
