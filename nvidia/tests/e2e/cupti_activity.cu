// A CUDA program that profiles itself through CUPTI, the way a real profiler
// does: register buffer callbacks, enable the activity kinds, run work, flush,
// and walk the records that come back.
//
// The point is that an unmodified consumer of the CUPTI Activity API gets the
// kernels and copies it actually issued, in order, with the launch geometry it
// used.
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
      if (k->end < k->start) ++g_bad_time;
    } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto* m = reinterpret_cast<CUpti_ActivityMemcpy5*>(record);
      ++g_memcpies;
      g_bytes += m->bytes;
      if (m->end < m->start) ++g_bad_time;
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
  if (g_bad_time) { std::printf("FAIL: %d records end before they start\n", g_bad_time); return 1; }

  std::printf("PASS\n");
  return 0;
}
