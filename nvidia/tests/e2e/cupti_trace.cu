// A program that traces itself through CUPTI and prints the trace in a form
// that does not depend on time, addresses or the numbers a CUPTI picks for its
// own ids, so the same program can be run against NVIDIA's libcupti on a card
// and against this project's, and the two outputs compared line by line.
//
// What is printed, and what is not:
//   * Callback API: every runtime-API call the program makes, as ENTER and EXIT,
//     with the function name, the return value and (for the calls it is checked
//     on) the parameters; the resource and synchronize callbacks between them.
//     All of it is things the program did, so a faithful CUPTI reports it
//     identically on any device.
//   * Activity API: kernels, copies, fills, waits, streams and runtime calls,
//     with launch geometry, byte counts, copy direction and memory kinds,
//     stream and correlation. Timestamps are checked for order only
//     (start <= end) and never printed. Correlation and stream ids are printed
//     as their rank among those the run saw ("c3" is the fourth call), because
//     their values differ between CUPTI implementations and mean nothing across
//     them -- what means something is which records share one.
//   * Not printed: register counts and the device's memory size (they come
//     from the compiler and the board), and the module-load callbacks, which a
//     real driver raises when it lazily loads a kernel's code.
// Whatever a real CUPTI reports that this one does not is the work remaining;
// the expected output committed beside the test is what the card printed.
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

#include <cuda_runtime.h>
#include <cupti.h>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_) {                                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
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
    // Module loading is raised when a real driver lazily loads a kernel's code
    // (6 and 8); the simulator loads whole modules at once and does not raise it.
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
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
  const auto* cb = static_cast<const CUpti_CallbackData*>(data);
  const bool enter = cb->callbackSite == CUPTI_API_ENTER;
  std::string line = fmt("CB runtime %s %s", cb->functionName ? cb->functionName : "?", enter ? "ENTER" : "EXIT");
  if (!enter) line += fmt(" ret=%d", static_cast<int>(*static_cast<const cudaError_t*>(cb->functionReturnValue)));
  switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020: {
      const auto* p = static_cast<const cudaMemcpy_v3020_params*>(cb->functionParams);
      line += fmt(" count=%zu kind=%d", p->count, static_cast<int>(p->kind));
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: {
      const auto* p = static_cast<const cudaMemcpyAsync_v3020_params*>(cb->functionParams);
      line += fmt(" count=%zu kind=%d", p->count, static_cast<int>(p->kind));
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020: {
      const auto* p = static_cast<const cudaMemset_v3020_params*>(cb->functionParams);
      line += fmt(" value=%d count=%zu", p->value, p->count);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020: {
      const auto* p = static_cast<const cudaMemsetAsync_v3020_params*>(cb->functionParams);
      line += fmt(" value=%d count=%zu", p->value, p->count);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020: {
      const auto* p = static_cast<const cudaMalloc_v3020_params*>(cb->functionParams);
      line += fmt(" size=%zu", p->size);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020: {
      const auto* p = static_cast<const cudaHostAlloc_v3020_params*>(cb->functionParams);
      line += fmt(" size=%zu flags=%u", p->size, p->flags);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: {
      const auto* p = static_cast<const cudaLaunchKernel_v7000_params*>(cb->functionParams);
      line += fmt(" grid=%ux%ux%u block=%ux%ux%u shared=%zu", p->gridDim.x, p->gridDim.y, p->gridDim.z,
                  p->blockDim.x, p->blockDim.y, p->blockDim.z, p->sharedMem);
      if (cb->symbolName) line += fmt(" symbol=%s", cb->symbolName);
      break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020: {
      const auto* p = static_cast<const cudaSetDevice_v3020_params*>(cb->functionParams);
      line += fmt(" device=%d", p->device);
      break;
    }
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
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
        if (a->end < a->start) ++g_bad_time;
        const char* name = nullptr;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name);
        g_records.push_back({fmt("REC runtime %s corr=#c%u", name ? name : "?", a->correlationId), a->correlationId, 1});
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

}  // namespace

// External linkage: an anonymous namespace puts a hash of the file into the
// kernels' mangled names, and the trace prints them.
__global__ void scale(float* p, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= a;
}
__global__ void reverse_block(float* p, int n) {
  extern __shared__ float tile[];
  __shared__ float stat[8];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < 8) stat[threadIdx.x] = 0.f;
  tile[threadIdx.x] = i < n ? p[i] : 0.f;
  __syncthreads();
  if (i < n) p[i] = tile[blockDim.x - 1 - threadIdx.x] + stat[0];
}

int main() {
  CUpti_SubscriberHandle sub;
  CP(cuptiSubscribe(&sub, on_callback, nullptr));
  CP(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API));
  CP(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RESOURCE));
  CP(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_SYNCHRONIZE));
  CP(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
  for (auto kind : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
                    CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, CUPTI_ACTIVITY_KIND_DEVICE,
                    CUPTI_ACTIVITY_KIND_CONTEXT, CUPTI_ACTIVITY_KIND_STREAM})
    CP(cuptiActivityEnable(kind));

  // Everything below the warm-up is the workload; the device is made current
  // first so lazy context creation is not part of the trace.
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  g_setup = g_lines;
  g_lines.clear();
  CP(cuptiActivityFlushAll(0));
  g_records.clear();

  const int n = 512;
  float *a = nullptr, *b = nullptr, *pinned = nullptr;
  float host[n];
  for (int i = 0; i < n; ++i) host[i] = float(i);
  cudaStream_t s1;
  cudaEvent_t ev;
  CK(cudaStreamCreate(&s1));
  CK(cudaEventCreate(&ev));
  CK(cudaMalloc(&a, n * sizeof(float)));
  CK(cudaMalloc(&b, n * sizeof(float)));
  CK(cudaMallocHost(&pinned, n * sizeof(float)));
  for (int i = 0; i < n; ++i) pinned[i] = 1.f;

  CK(cudaMemcpy(a, host, n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpyAsync(b, pinned, n * sizeof(float), cudaMemcpyHostToDevice, s1));
  CK(cudaMemset(b, 0, 64));
  CK(cudaMemsetAsync(a, 0x11, 128, s1));
  scale<<<2, 256>>>(a, 2.f, n);
  reverse_block<<<4, 128, 128 * sizeof(float), s1>>>(b, n);
  CK(cudaEventRecord(ev, s1));
  CK(cudaStreamWaitEvent(0, ev, 0));
  CK(cudaEventSynchronize(ev));
  CK(cudaStreamSynchronize(s1));
  CK(cudaMemcpy(b, a, n * sizeof(float), cudaMemcpyDeviceToDevice));
  CK(cudaMemcpy(host, b, n * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaDeviceSynchronize());
  CK(cudaEventDestroy(ev));
  CK(cudaFreeHost(pinned));
  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaStreamDestroy(s1));

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
