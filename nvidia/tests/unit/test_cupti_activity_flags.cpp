// Activity records of the kinds that need no callback to be asked for: switching on only an activity kind (no
// subscriber, no API-call kind) must make the runtime's, the driver's and the CUPTI shim's own producers write their
// records. The flag that says "somebody records" is set by the CUPTI shim and read by the runtime, the driver and the
// shim's NVTX entry; all three libraries are loaded here, in the order a program that links -lcupti -lcuda -lcudart
// loads them, which is where a flag each library defined for itself went unseen (markers, streams, contexts and CUDA
// events were not recorded, the API-call records were).
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <dlfcn.h>

#include <cstdlib>
#include <map>
#include <string>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define HAVE_NVTX 1
#endif

#include "vtest.hpp"

extern "C" const char* __asan_default_options() { return "detect_leaks=1:detect_odr_violation=0"; }

namespace {
std::map<int, int> g_seen;   // activity kind -> records
uint8_t g_buffer[1 << 20] __attribute__((aligned(8)));

void CUPTIAPI request(uint8_t** b, size_t* s, size_t* m) {
  *b = g_buffer;
  *s = sizeof g_buffer;
  *m = 0;
}
void CUPTIAPI complete(CUcontext, uint32_t, uint8_t* b, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(b, valid, &r) == CUPTI_SUCCESS) ++g_seen[static_cast<int>(r->kind)];
}
int seen(CUpti_ActivityKind k) { return g_seen[static_cast<int>(k)]; }
}  // namespace

VTEST(only_activity_kinds_on_the_producers_of_every_library_record) {
#ifdef HAVE_NVTX
  // NVTX reaches a tool through the library named here, as a profiler sets it; before the first NVTX call.
  Dl_info info{};
  VCHECK(dladdr(reinterpret_cast<void*>(&cuptiActivityEnable), &info) != 0);
  setenv("NVTX_INJECTION64_PATH", info.dli_fname, 1);
#endif
  VCHECK_EQ(cuptiActivityRegisterCallbacks(request, complete), CUPTI_SUCCESS);
  for (CUpti_ActivityKind k : {CUPTI_ACTIVITY_KIND_STREAM, CUPTI_ACTIVITY_KIND_CONTEXT, CUPTI_ACTIVITY_KIND_CUDA_EVENT,
                               CUPTI_ACTIVITY_KIND_MARKER, CUPTI_ACTIVITY_KIND_MARKER_DATA})
    VCHECK_EQ(cuptiActivityEnable(k), CUPTI_SUCCESS);

  VCHECK_EQ(cudaSetDevice(0), cudaSuccess);
  cudaStream_t s = nullptr;
  VCHECK_EQ(cudaStreamCreate(&s), cudaSuccess);                     // the runtime's stream record
  cudaEvent_t e = nullptr;
  VCHECK_EQ(cudaEventCreate(&e), cudaSuccess);
  VCHECK_EQ(cudaEventRecord(e, s), cudaSuccess);                    // the runtime's CUDA event record
  CUstream ds = nullptr;
  VCHECK_EQ(cuStreamCreate(&ds, 0), CUDA_SUCCESS);                  // the driver's stream record
  CUevent de = nullptr;
  VCHECK_EQ(cuEventCreate(&de, 0), CUDA_SUCCESS);
  VCHECK_EQ(cuEventRecord(de, ds), CUDA_SUCCESS);                   // the driver's CUDA event record
#ifdef HAVE_NVTX
  nvtxMarkA("a mark");                                              // the CUPTI shim's own record
#endif
  VCHECK_EQ(cudaStreamSynchronize(s), cudaSuccess);
  VCHECK_EQ(cuptiActivityFlushAll(1), CUPTI_SUCCESS);

  VCHECK(seen(CUPTI_ACTIVITY_KIND_CONTEXT) >= 1);
  VCHECK(seen(CUPTI_ACTIVITY_KIND_STREAM) >= 2);        // one by the runtime, one by the driver (and the context's own)
  VCHECK(seen(CUPTI_ACTIVITY_KIND_CUDA_EVENT) >= 2);
#ifdef HAVE_NVTX
  VCHECK(seen(CUPTI_ACTIVITY_KIND_MARKER) >= 1);
#endif
  VCHECK_EQ(cudaStreamDestroy(s), cudaSuccess);
  VCHECK_EQ(cuptiFinalize(), CUPTI_SUCCESS);
}

VTEST_MAIN
