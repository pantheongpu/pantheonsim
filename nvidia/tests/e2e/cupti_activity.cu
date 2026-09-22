// A CUDA program that profiles itself through CUPTI, the way a real profiler
// does: register buffer callbacks, enable the activity kinds, run work, flush,
// and walk the records that come back.
//
// The point is that an unmodified consumer of the CUPTI Activity API gets the
// kernels and copies it actually issued, in order, with the launch geometry it
// used -- and the runtime calls that issued them, each carrying the same
// correlation ID as its work, which is how a profiler ties a kernel on the GPU
// timeline to the host call that launched it.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cupti.h>
#include <cuda_runtime.h>

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t e_ = (x);                                                            \
    if (e_) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } \
  } while (0)
#define CP(x)                                                                        \
  do {                                                                               \
    CUptiResult r_ = (x);                                                            \
    if (r_ != CUPTI_SUCCESS) { std::printf("FAIL %s:%d: CUPTI status %d\n", __FILE__, __LINE__, (int)r_); return 1; } \
  } while (0)

static int g_kernels = 0, g_memcpies = 0;
static int g_grid_x = 0, g_block_x = 0;
static char g_name[256] = {0};
static uint64_t g_bytes = 0;
static int g_bad_time = 0;
static int g_api = 0, g_api_malloc = 0;
static uint32_t g_kernel_corr = 0, g_launch_corr = 0, g_launch_cbid = 0;
static uint32_t g_copy_corr[2] = {0, 0}, g_memcpy_api_corr[2] = {0, 0};
static int g_copies_seen = 0, g_memcpy_api_seen = 0;

static void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
  static uint8_t storage[64 * 1024];
  *buffer = storage;
  *size = sizeof storage;
  *max_records = 0;
}

static void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* record = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
        record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(record);
      ++g_kernels;
      g_grid_x = k->gridX;
      g_block_x = k->blockX;
      if (k->name) { std::strncpy(g_name, k->name, sizeof g_name - 1); }
      g_kernel_corr = k->correlationId;
      if (k->end < k->start) ++g_bad_time;
    } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto* m = reinterpret_cast<CUpti_ActivityMemcpy5*>(record);
      ++g_memcpies;
      g_bytes += m->bytes;
      if (g_copies_seen < 2) g_copy_corr[g_copies_seen++] = m->correlationId;
      if (m->end < m->start) ++g_bad_time;
    } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
      auto* a = reinterpret_cast<CUpti_ActivityAPI*>(record);
      ++g_api;
      if (a->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020) ++g_api_malloc;
      if (a->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
        g_launch_corr = a->correlationId;
        g_launch_cbid = a->cbid;
      }
      if (a->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 && g_memcpy_api_seen < 2)
        g_memcpy_api_corr[g_memcpy_api_seen++] = a->correlationId;
      if (a->end < a->start) ++g_bad_time;
    }
  }
}

__global__ void addOne(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.0f;
}

int main() {
  uint32_t version = 0;
  CP(cuptiGetVersion(&version));
  if (version == 0) { std::printf("FAIL: CUPTI reported version 0\n"); return 1; }

  CP(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
  CP(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CP(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CP(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));

  const int n = 1024;
  float* d = nullptr;
  float h[n];
  for (int i = 0; i < n; ++i) h[i] = (float)i;
  CK(cudaMalloc(&d, n * sizeof(float)));
  CK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
  addOne<<<4, 256>>>(d, n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaFree(d));

  CP(cuptiActivityFlushAll(0));

  if (h[7] != 8.0f) { std::printf("FAIL: kernel did not run (h[7]=%g)\n", h[7]); return 1; }
  if (g_kernels != 1) { std::printf("FAIL: %d kernel records, expected 1\n", g_kernels); return 1; }
  if (g_memcpies != 2) { std::printf("FAIL: %d memcpy records, expected 2\n", g_memcpies); return 1; }
  if (g_grid_x != 4 || g_block_x != 256) {
    std::printf("FAIL: launch geometry %dx%d, expected 4x256\n", g_grid_x, g_block_x);
    return 1;
  }
  if (std::strstr(g_name, "addOne") == nullptr) {
    std::printf("FAIL: kernel name '%s' does not mention addOne\n", g_name);
    return 1;
  }
  if (g_bytes != 2ull * n * sizeof(float)) {
    std::printf("FAIL: %llu bytes copied, expected %llu\n", (unsigned long long)g_bytes,
                (unsigned long long)(2ull * n * sizeof(float)));
    return 1;
  }
  if (g_api_malloc != 1) { std::printf("FAIL: %d cudaMalloc runtime records, expected 1\n", g_api_malloc); return 1; }
  if (!g_launch_corr || g_launch_corr != g_kernel_corr) {
    std::printf("FAIL: the launch's runtime record (correlation %u) does not match its kernel's (%u)\n",
                g_launch_corr, g_kernel_corr);
    return 1;
  }
  if (g_memcpy_api_seen != 2 || g_copy_corr[0] != g_memcpy_api_corr[0] || g_copy_corr[1] != g_memcpy_api_corr[1]) {
    std::printf("FAIL: the copies' correlations (%u, %u) do not match their cudaMemcpy calls' (%u, %u)\n",
                g_copy_corr[0], g_copy_corr[1], g_memcpy_api_corr[0], g_memcpy_api_corr[1]);
    return 1;
  }
  const char* launch_name = nullptr;
  CP(cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, g_launch_cbid, &launch_name));
  if (!launch_name || std::strcmp(launch_name, "cudaLaunchKernel") != 0) {
    std::printf("FAIL: callback %u is named '%s', expected cudaLaunchKernel\n", g_launch_cbid,
                launch_name ? launch_name : "(null)");
    return 1;
  }
  if (g_bad_time) { std::printf("FAIL: %d records end before they start\n", g_bad_time); return 1; }

  std::printf("PASS\n");
  return 0;
}
