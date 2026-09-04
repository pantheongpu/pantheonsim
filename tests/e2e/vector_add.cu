// A completely ordinary CUDA program. Compiled with nvcc, unaware of VirtualGPU.
// The e2e test runs it against libvgpucudart and checks the result.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// Frees everything before returning, on every path. Ordinary CUDA programs are
// often careless about this; the e2e run happens under LeakSanitizer, and a
// leak here would be reported against the shim it is exercising.
static int finish(const char* msg, float* ha, float* hb, float* hc,
                  float* da, float* db, float* dc, int rc) {
  cudaFree(da); cudaFree(db); cudaFree(dc);
  free(ha); free(hb); free(hc);
  printf("%s\n", msg);
  return rc;
}

int main() {
  const int n = 4096;
  size_t bytes = n * sizeof(float);
  float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) { ha[i] = i * 0.5f; hb[i] = i * 0.25f; }
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  if (cudaMalloc(&da, bytes) || cudaMalloc(&db, bytes) || cudaMalloc(&dc, bytes))
    return finish("FAIL malloc", ha, hb, hc, da, db, dc, 1);
  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
  vecAdd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    char msg[256];
    snprintf(msg, sizeof msg, "FAIL launch: %s", cudaGetErrorString(e));
    return finish(msg, ha, hb, hc, da, db, dc, 1);
  }
  cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) {
    float want = ha[i] + hb[i];
    if (hc[i] != want) {
      char msg[256];
      snprintf(msg, sizeof msg, "FAIL c[%d]=%g want %g", i, hc[i], want);
      return finish(msg, ha, hb, hc, da, db, dc, 1);
    }
  }
  return finish("PASS", ha, hb, hc, da, db, dc, 0);
}
