// The profiler's own cost as a record: what CUPTI reports as overhead while
// nothing else is being recorded. The first launch of a kernel loads it, which
// is a record of its own (CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING) with
// the correlation of the launch that caused it, and a flush that delivers
// records reports the request for the buffer they came in
// (CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST). Launching the same kernel
// again loads nothing. Run against NVIDIA's libcupti on a card it gives
// nvidia/tests/data/cupti_overhead.expected.
//
// Not reproduced: the records NVIDIA's makes while it sets up (instrumentation
// and resource overhead, when the context is created), which are its own
// bookkeeping and have no counterpart here; and the overhead of a flush that
// also delivers other kinds, which NVIDIA's delivers unreliably (a flush can
// complete before a kernel's record is written, and the kernel's record is then
// lost), so this case enables the one kind alone.
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

#if CUPTI_API_VERSION >= 22
using OverheadRecord = CUpti_ActivityOverhead3;
#else
using OverheadRecord = CUpti_ActivityOverhead;
#endif

namespace {

int g_lazy = 0, g_request = 0, g_flush = 0, g_buffers = 0, g_late = 0;
#if CUPTI_API_VERSION >= 22
uint32_t g_lazy_corr = 0;
#endif

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  ++g_buffers;
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    if (static_cast<int>(r->kind) != 17) continue;   // CUPTI_ACTIVITY_KIND_OVERHEAD
    const auto* o = reinterpret_cast<const OverheadRecord*>(r);
    const int kind = static_cast<int>(o->overheadKind);
    if (o->end < o->start) ++g_late;
    if (kind == (5 << 16)) {
      ++g_lazy;
#if CUPTI_API_VERSION >= 22
      g_lazy_corr = o->correlationId;
#endif
    }
    if (kind == (7 << 16)) ++g_request;
    if (kind == (1 << 16)) ++g_flush;
  }
}

void flush_and_tell(const char* what) {
  g_lazy = g_request = g_flush = g_buffers = 0;
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  out(fmt("%s: function loading %s, buffer request %s, flush cost %s, times in order %s", what,
          g_lazy ? "yes" : "no", g_request ? "yes" : "no", g_flush ? "yes" : "no", g_late ? "NO" : "yes"));
}

}  // namespace

__global__ void first_kernel(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.f;
}
__global__ void second_kernel(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] -= 1.f;
}

int main() {
  cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed);
  if (cuptiActivityEnable(static_cast<CUpti_ActivityKind>(17)) != CUPTI_SUCCESS) {
    std::fprintf(stderr, "FAIL: cannot enable the overhead kind\n");
    return 1;
  }
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);   // what the context cost is NVIDIA's own
  float* d = nullptr;
  CK(cudaMalloc(&d, 4096));
  first_kernel<<<1, 32>>>(d, 32);
  CK(cudaDeviceSynchronize());
  flush_and_tell("first launch of a kernel");
  first_kernel<<<1, 32>>>(d, 32);
  CK(cudaDeviceSynchronize());
  flush_and_tell("second launch of it");
  second_kernel<<<1, 32>>>(d, 32);
  CK(cudaDeviceSynchronize());
  flush_and_tell("first launch of another");
  flush_and_tell("nothing since");
  CK(cudaFree(d));
  cuptiActivityDisable(static_cast<CUpti_ActivityKind>(17));
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
