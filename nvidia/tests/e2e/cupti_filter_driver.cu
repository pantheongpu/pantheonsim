// Recording single driver functions (cuptiActivityEnableDriverApi) in a program
// that uses only the driver API, before the context exists, as a profiler that
// wants a few calls does it. Run against NVIDIA's libcupti on a card it gives
// nvidia/tests/data/cupti_filter_driver.expected (an RTX 3060, CUDA 13.0).
//
// The same rules as the runtime's (cupti_filter.cu): the functions switched on
// before the first context are recorded, others switched on after are too; a
// kind switched on makes everything recorded but what is switched off.
// Not reproduced: the driver calls a runtime makes underneath its own (a
// cudaMalloc is a cuMemAlloc to NVIDIA's CUPTI); this program has no runtime
// calls.
#include "cupti_test_util.h"

#include <cuda.h>

using cupti_test::fmt;
using cupti_test::out;

#define CU(x)                                                                          \
  do {                                                                                 \
    CUresult r_ = (x);                                                                 \
    if (r_) {                                                                          \
      std::fprintf(stderr, "FAIL %s:%d: driver error %d\n", __FILE__, __LINE__, (int)r_); \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

namespace {

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    if (r->kind != CUPTI_ACTIVITY_KIND_DRIVER) continue;
    const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
    const char* name = nullptr;
    cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, a->cbid, &name);
    // The calls that make the context are not what this is about.
    if (name && std::strstr(name, "cuMem") == name) out(fmt("  %s", name));
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
  SHOW(cuptiActivityEnableDriverApi(0, 1));
  SHOW(cuptiActivityEnableDriverApi(99999, 1));
  SHOW(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2, 1));
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
#if CUDA_VERSION >= 13000
  CU(cuCtxCreate(&ctx, nullptr, 0, dev));
#else
  CU(cuCtxCreate(&ctx, 0, dev));
#endif
  cuptiActivityFlushAll(0);
  CUdeviceptr p;
  auto traffic = [&]() -> int {
    CU(cuMemAlloc(&p, 4096));
    CU(cuMemsetD8(p, 0, 4096));
    CU(cuMemFree(p));
    return 0;
  };
  auto section = [&](const char* name) {
    out(std::string("# ") + name);
    cuptiActivityFlushAll(0);
  };
  if (traffic()) return 1;
  section("only cuMemAlloc");
  SHOW(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2, 1));
  if (traffic()) return 1;
  section("and cuMemFree");
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  SHOW(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2, 0));
  if (traffic()) return 1;
  section("the kind on, cuMemsetD8 off");
  SHOW(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  if (traffic()) return 1;
  section("the kind on again");
  SHOW(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CU(cuCtxDestroy(ctx));
#else
  out("(this toolkit's CUPTI has no per-function switches)");
#endif
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
