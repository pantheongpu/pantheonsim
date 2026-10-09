// Unified Memory counters (CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER): which
// counters can be configured, and what a managed allocation produces as the
// host and the device touch it, prefetch it and advise on it. Needs a device
// and a driver that do demand paging of managed memory (Linux, compute
// capability 6.0 or newer); where the driver does not, the case prints what
// configuring and enabling answered, which is what the WSL machine this was
// developed on gives. Run against NVIDIA's libcupti on such a machine it gives
// nvidia/tests/data/cupti_um.expected.
#include <algorithm>

#include "cupti_test_util.h"

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
using UmRecord = CUpti_ActivityUnifiedMemoryCounter3;
#else
using UmRecord = CUpti_ActivityUnifiedMemoryCounter2;
#endif

namespace {

uint64_t g_base = 0;
size_t g_bytes = 0;
std::vector<std::string> g_batch;

const char* result_name(CUptiResult r) {
  const char* s = "?";
  cuptiGetResultString(r, &s);
  return s;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    if (static_cast<int>(r->kind) != CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
      g_batch.push_back(fmt("  other record kind %d", static_cast<int>(r->kind)));
      continue;
    }
    const auto* u = reinterpret_cast<const UmRecord*>(r);
    const bool inside = u->address >= g_base && u->address < g_base + g_bytes;
    g_batch.push_back(fmt("  counter %d value=%llu src=%u dst=%u flags=%u offset=%lld%s order=%s pid=%s",
                          static_cast<int>(u->counterKind), (unsigned long long)u->value, u->srcId, u->dstId, u->flags,
                          inside ? (long long)(u->address - g_base) : -1ll, inside ? "" : " (outside)",
                          u->end >= u->start ? "ok" : "BAD", u->processId ? "yes" : "no"));
  }
}

// What the phase produced, in a fixed order: records come back in the order
// they were made, which is not a thing to compare.
void phase(const char* title) {
  cuptiActivityFlushAll(0);
  out(std::string("# ") + title);
  std::sort(g_batch.begin(), g_batch.end());
  for (const auto& l : g_batch) out(l);
  out(fmt("  records: %zu", g_batch.size()));
  g_batch.clear();
}

}  // namespace

__global__ void touch(int* p, size_t n, int v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] += v;
}

int main() {
  cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed);

  out("# configuring counters");
  for (int k = 1; k <= 9; ++k) {
    CUpti_ActivityUnifiedMemoryCounterConfig c = {};
    c.scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    c.kind = static_cast<CUpti_ActivityUnifiedMemoryCounterKind>(k);
    c.deviceId = 0;
    c.enable = 1;
    out(fmt("  kind %d, one device: %s", k, result_name(cuptiActivityConfigureUnifiedMemoryCounter(&c, 1))));
  }
  {
    CUpti_ActivityUnifiedMemoryCounterConfig c[2] = {};
    c[0].scope = c[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES;
    c[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    c[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    c[0].enable = c[1].enable = 1;
    out(fmt("  two kinds, all devices: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(c, 2))));
    CUpti_ActivityUnifiedMemoryCounterConfig odd = c[0];
    out(fmt("  none of a non-null list: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(&odd, 0))));
    odd.kind = static_cast<CUpti_ActivityUnifiedMemoryCounterKind>(0);
    out(fmt("  kind 0: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(&odd, 1))));
    odd.kind = static_cast<CUpti_ActivityUnifiedMemoryCounterKind>(100);
    out(fmt("  kind 100: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(&odd, 1))));
    odd = c[0];
    odd.scope = static_cast<CUpti_ActivityUnifiedMemoryCounterScope>(0);
    out(fmt("  scope 0: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(&odd, 1))));
    odd.scope = static_cast<CUpti_ActivityUnifiedMemoryCounterScope>(9);
    out(fmt("  scope 9: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(&odd, 1))));
    odd = c[0];
    odd.enable = 0;
    out(fmt("  disabling: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(&odd, 1))));
    odd = c[0];
    odd.deviceId = 77;
    out(fmt("  device 77: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(&odd, 1))));
    out(fmt("  nothing: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(nullptr, 0))));
    out(fmt("  a null list: %s", result_name(cuptiActivityConfigureUnifiedMemoryCounter(nullptr, 1))));
  }
  out(fmt("enable the kind: %s", result_name(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER))));
  out(fmt("enable it again: %s", result_name(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER))));
  out(fmt("disable it: %s", result_name(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER))));
  // A driver that does not page managed memory on demand refuses the kind; the
  // rest of the case is for one that does.
  if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) != CUPTI_SUCCESS) {
    cupti_test::print_all();
    std::printf("# end\n");
    return 0;
  }
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));

  const size_t n = size_t{1} << 20;   // ints: 4 MiB
  g_bytes = n * sizeof(int);
  int* p = nullptr;
  CK(cudaMallocManaged(&p, g_bytes));
  g_base = reinterpret_cast<uint64_t>(p);
  phase("allocated");
  for (size_t i = 0; i < n; ++i) p[i] = 1;
  phase("written by the host");
  touch<<<64, 256>>>(p, n, 1);
  CK(cudaDeviceSynchronize());
  phase("written by the device");
  long sum = 0;
  for (size_t i = 0; i < n; ++i) sum += p[i];
  out(fmt("  sum %ld", sum));
  phase("read by the host");
  CK(cudaMemPrefetchAsync(p, g_bytes,
#if CUDART_VERSION >= 13000
                          cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0,
#else
                          0,
#endif
                          0));
  CK(cudaDeviceSynchronize());
  phase("prefetched to the device");
  CK(cudaMemPrefetchAsync(p, g_bytes,
#if CUDART_VERSION >= 13000
                          cudaMemLocation{cudaMemLocationTypeHost, 0}, 0,
#else
                          cudaCpuDeviceId,
#endif
                          0));
  CK(cudaDeviceSynchronize());
  phase("prefetched to the host");
  int* d = nullptr;
  CK(cudaMalloc(&d, g_bytes));
  CK(cudaMemcpy(d, p, g_bytes, cudaMemcpyDefault));
  CK(cudaDeviceSynchronize());
  phase("copied from managed memory to a device allocation");
  CK(cudaFree(d));

  int devices = 0;
  CK(cudaGetDeviceCount(&devices));
  if (devices >= 2) {
    for (int round = 0; round < 3; ++round) {
      CK(cudaSetDevice(1));
      touch<<<64, 256>>>(p, n, 1);
      CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(0));
      touch<<<64, 256>>>(p, n, 1);
      CK(cudaDeviceSynchronize());
    }
    phase("touched by two devices in turn");
  }
  CK(cudaSetDevice(0));
  CK(cudaFree(p));
  phase("freed");
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
