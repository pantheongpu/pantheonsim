// E5: per-launch cost and the compute scaling curve, the SAME binary on the simulator and on the real A10G. Throwaway tool, not product code.
// Build: nvcc -arch=sm_86 -O2 -cudart shared e5_launch.cu -o e5_launch      (the shim substitutes libcudart for the simulator)
// usage: e5_launch <mode> [reps]    mode: launch | copy | scale
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <time.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorName(e_)); exit(1); } } while (0)
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
__global__ void empty_k() {}
__global__ void axpb(const float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = x[i] * 2.0f + 1.0f; }
__global__ void burn(float* y, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float a = i * 0.001f, b = 1.0001f;
  for (int k = 0; k < iters; ++k) a = fmaf(a, b, 0.5f);
  y[i] = a;
}
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "launch"; int reps = argc > 2 ? atoi(argv[2]) : 200;
  CK(cudaFree(0));
  if (!strcmp(mode, "launch")) {            // one-warp empty launches, synchronised each time and back to back
    for (int i = 0; i < 20; ++i) empty_k<<<1, 32>>>(); CK(cudaDeviceSynchronize());
    double t0 = now(); for (int i = 0; i < reps; ++i) { empty_k<<<1, 32>>>(); CK(cudaDeviceSynchronize()); } double t1 = now();
    double t2 = now(); for (int i = 0; i < reps; ++i) empty_k<<<1, 32>>>(); CK(cudaDeviceSynchronize()); double t3 = now();
    printf("launch+sync_us %.2f   launch_pipelined_us %.2f\n", (t1 - t0) / reps * 1e6, (t3 - t2) / reps * 1e6);
  } else if (!strcmp(mode, "copy")) {       // H2D n floats, launch axpb, D2H n floats, per iteration
    for (int n : {256, 4096, 65536, 1 << 20, 1 << 24}) {
      std::vector<float> h(n, 1.0f), o(n);
      float *dx, *dy; CK(cudaMalloc(&dx, n * 4)); CK(cudaMalloc(&dy, n * 4));
      int it = n >= (1 << 24) ? 5 : reps;
      for (int w = 0; w < 2; ++w) { CK(cudaMemcpy(dx, h.data(), n * 4, cudaMemcpyHostToDevice)); axpb<<<(n + 255) / 256, 256>>>(dx, dy, n); CK(cudaMemcpy(o.data(), dy, n * 4, cudaMemcpyDeviceToHost)); }
      double t0 = now();
      for (int i = 0; i < it; ++i) { CK(cudaMemcpy(dx, h.data(), n * 4, cudaMemcpyHostToDevice)); axpb<<<(n + 255) / 256, 256>>>(dx, dy, n); CK(cudaMemcpy(o.data(), dy, n * 4, cudaMemcpyDeviceToHost)); }
      double t1 = now();
      bool ok = true; for (int i = 0; i < n; ++i) if (o[i] != 3.0f) { ok = false; break; }
      printf("n=%d iter_us %.1f ok=%d\n", n, (t1 - t0) / it * 1e6, ok);
      cudaFree(dx); cudaFree(dy);
    }
  } else {                                  // 1M threads, K dependent fmas each: compute scaling at a fixed launch count
    int n = 1 << 20; float* dy; CK(cudaMalloc(&dy, n * 4));
    for (int iters : {1, 10, 100, 1000, 10000}) {
      burn<<<n / 256, 256>>>(dy, n, 1); CK(cudaDeviceSynchronize());
      double t0 = now(); burn<<<n / 256, 256>>>(dy, n, iters); CK(cudaDeviceSynchronize()); double t1 = now();
      std::vector<float> o(n); CK(cudaMemcpy(o.data(), dy, n * 4, cudaMemcpyDeviceToHost));
      unsigned h = 2166136261u; for (int i = 0; i < n; ++i) { unsigned u; memcpy(&u, &o[i], 4); h = (h ^ u) * 16777619u; }
      printf("iters=%d  seconds %.4f  fma_per_s %.3g  result_hash %08x\n", iters, t1 - t0, double(n) * iters / (t1 - t0), h);
      fflush(stdout);
    }
  }
  return 0;
}
