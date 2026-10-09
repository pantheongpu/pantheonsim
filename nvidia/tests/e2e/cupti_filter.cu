// Recording single runtime functions (cuptiActivityEnableRuntimeApi), the way a
// profiler that wants a few calls and not all of them does it: before the first
// CUDA call. Run against NVIDIA's libcupti on a card it gives
// nvidia/tests/data/cupti_filter.expected (an RTX 3060, CUDA 13.0).
//
// What the card does, and the shim reproduces:
//  * a function switched on before the context exists is recorded and nothing
//    else is, though the kind is not on; switching more on afterwards works;
//  * once the kind of runtime records is on, every function is recorded but the
//    ones switched off, and switching the kind on again forgets those choices;
//  * once the kind has been switched off, the function switches record nothing
//    until it is switched on again.
// (A function switched on after the context exists, with nothing of the kind
// on when it was made, records nothing: that is cupti_misc.)
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

namespace {

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    if (r->kind != CUPTI_ACTIVITY_KIND_RUNTIME) continue;
    const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
    const char* name = nullptr;
    cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name);
    out(fmt("  %s", name ? name : "?"));
  }
}

const char* result_name(CUptiResult r) {
  const char* s = "?";
  cuptiGetResultString(r, &s);
  return s;
}

}  // namespace

#define SHOW(call)                                                  \
  do {                                                              \
    const CUptiResult r_ = (call);                                  \
    out(fmt("%s -> %s", #call, result_name(r_)));                   \
  } while (0)

int main() {
#if CUPTI_API_VERSION >= 24
  cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed);
  SHOW(cuptiActivityEnableRuntimeApi(0, 1));
  SHOW(cuptiActivityEnableRuntimeApi(99999, 1));
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020, 1));
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  cuptiActivityFlushAll(0);
  float* d = nullptr;
  float host[64] = {};
  auto traffic = [&]() -> int {
    CK(cudaMalloc(&d, 4096));
    CK(cudaMemcpy(d, host, sizeof host, cudaMemcpyHostToDevice));
    CK(cudaFree(d));
    return 0;
  };
  auto section = [&](const char* name) {
    out(std::string("# ") + name);
    cuptiActivityFlushAll(0);
  };
  if (traffic()) return 1;
  section("only cudaMalloc");
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020, 1));
  if (traffic()) return 1;
  section("and cudaFree");
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020, 2));
  if (traffic()) return 1;
  section("and cudaMemcpy (switched on with 2)");
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020, 0));
  if (traffic()) return 1;
  section("the kind on, cudaMalloc off");
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  if (traffic()) return 1;
  section("the kind on again");
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020, 0));
  if (traffic()) return 1;
  section("cudaFree off");
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  if (traffic()) return 1;
  section("the kind off");
  SHOW(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020, 1));
  if (traffic()) return 1;
  section("the kind off, cudaMemcpy on");
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  if (traffic()) return 1;
  section("the kind on");
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
#else
  out("(this toolkit's CUPTI has no per-function switches)");
#endif
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
