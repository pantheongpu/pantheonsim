// What host-GPU mode costs a program whose launches stay in the simulator (pantheonsim #662, #660): `kernels` launches 50 distinct large
// kernels once each on one warp, `repeat` one kernel 5000 times. Neither has a launch worth a card.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define STEP(N) x = x * 1.0001f + (float)(N + 1); y = y + x * 0.5f; x = x - y * 0.25f;
#define KERNEL(N) \
  __global__ void k##N(float* a) { \
    int i = threadIdx.x; \
    float x = a[i], y = 1.0f; \
    _Pragma("unroll") for (int r = 0; r < 40; ++r) { STEP(N) } \
    a[i] = x + y; \
  }
#define K10(D) KERNEL(D##0) KERNEL(D##1) KERNEL(D##2) KERNEL(D##3) KERNEL(D##4) KERNEL(D##5) KERNEL(D##6) KERNEL(D##7) KERNEL(D##8) KERNEL(D##9)
K10(1) K10(2) K10(3) K10(4) K10(5)

__global__ void rep(float* a) {
  int i = threadIdx.x;
  a[i] = a[i] * 1.0001f + 1.0f;
}

int main(int argc, char** argv) {
  float* d;
  cudaMalloc(&d, 4096);
  cudaMemset(d, 0, 4096);
  if (argc > 1 && !std::strcmp(argv[1], "repeat")) {
    for (int i = 0; i < 5000; ++i) rep<<<1, 32>>>(d);
  } else if (argc > 1 && !std::strcmp(argv[1], "one")) {
    k10<<<1, 32>>>(d);
  } else {
#define L(N) k##N<<<1, 32>>>(d);
#define L10(D) L(D##0) L(D##1) L(D##2) L(D##3) L(D##4) L(D##5) L(D##6) L(D##7) L(D##8) L(D##9)
    L10(1) L10(2) L10(3) L10(4) L10(5)
  }
  float h[32];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  double s = 0;
  for (float v : h) s += v;
  std::printf("sum %.6f\n", s);
  return 0;
}
