// External correlation ids: a framework tags the work it is about to issue with
// a push, and CUPTI reports every runtime call made under a tag with it. This
// is how PyTorch's profiler ties a kernel to the operator that launched it.
// Printed for comparison with what NVIDIA's CUPTI gave the same program on an
// RTX 3060; correlation ids are printed as their rank among the ones seen.
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cupti.h>

namespace {
struct Rec {
  std::string text;
  uint32_t corr;
  int rank;
};
std::vector<Rec> g_recs;

void CUPTIAPI request(uint8_t** b, size_t* s, size_t* m) {
  static uint8_t storage[1 << 20] __attribute__((aligned(8)));
  *b = storage;
  *s = sizeof storage;
  *m = 0;
}
void CUPTIAPI complete(CUcontext, uint32_t, uint8_t* b, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(b, valid, &r) == CUPTI_SUCCESS) {
    char line[160];
    if (r->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
      const auto* e = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(r);
      std::snprintf(line, sizeof line, "EXT kind=%d id=%llu", (int)e->externalKind, (unsigned long long)e->externalId);
      g_recs.push_back({line, e->correlationId, 0});
    } else if (r->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
      const auto* e = reinterpret_cast<const CUpti_ActivityAPI*>(r);
      const char* name = "?";
      cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, e->cbid, &name);
      std::snprintf(line, sizeof line, "RUNTIME %s", name);
      g_recs.push_back({line, e->correlationId, 1});
    }
  }
}
}  // namespace

__global__ void touch(int* p) { p[0] = 1; }

int main() {
  cuptiActivityRegisterCallbacks(request, complete);
  cudaFree(nullptr);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);

  int* d = nullptr;
  cudaMalloc(&d, 4);
  cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 100);
  touch<<<1, 1>>>(d);
  cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 200);
  cudaMemset(d, 0, 4);
  cudaDeviceSynchronize();
  uint64_t last = 0;
  CUptiResult r1 = cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, &last);
  std::printf("pop custom1 -> %d, popped %llu\n", (int)r1, (unsigned long long)last);
  cudaMemset(d, 0, 4);
  // A second push of the same kind nests: the innermost is the one reported.
  cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 101);
  cudaMemset(d, 0, 4);
  cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &last);
  std::printf("popped %llu\n", (unsigned long long)last);
  cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &last);
  std::printf("popped %llu\n", (unsigned long long)last);
  cudaFree(d);
  cuptiActivityFlushAll(0);
  CUptiResult empty = cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &last);
  std::printf("pop from an empty stack -> %d\n", (int)empty);
  CUptiResult bad = cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_INVALID, 1);
  std::printf("push of an invalid kind -> %d\n", (int)bad);

  std::stable_sort(g_recs.begin(), g_recs.end(), [](const Rec& a, const Rec& b) {
    return a.corr != b.corr ? a.corr < b.corr : a.rank < b.rank;
  });
  std::map<uint32_t, int> rank;
  for (const auto& r : g_recs) rank.emplace(r.corr, 0);
  int n = 0;
  for (auto& kv : rank) kv.second = n++;
  for (const auto& r : g_recs) std::printf("%s corr=c%d\n", r.text.c_str(), rank[r.corr]);
  return 0;
}
