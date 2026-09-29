// Which code runs: SASS when the binary carries some for the GPU, the PTX
// otherwise (run_sass_path.sh). A few instructions of each kind, checked on
// the host, so a wrong answer from either path fails.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void mix(const float* x, const int* k, float* y, int* z, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  __shared__ float s[128];
  s[threadIdx.x] = x[i] * 2.0f + 1.0f;
  __syncthreads();
  const float v = s[(threadIdx.x + 1) % blockDim.x];
  y[i] = v > 10.0f ? sqrtf(v) : v * v;
  z[i] = (k[i] << 3) ^ (k[i] >> 1) + __popc(k[i]) + __shfl_xor_sync(~0u, k[i], 1);
}

int main() {
  const int n = 256;
  float hx[n], hy[n];
  int hk[n], hz[n];
  for (int i = 0; i < n; ++i) {
    hx[i] = 0.25f * static_cast<float>(i % 37);
    hk[i] = i * 2654435761u;
  }
  float *x, *y;
  int *k, *z;
  cudaMalloc(&x, sizeof hx);
  cudaMalloc(&y, sizeof hy);
  cudaMalloc(&k, sizeof hk);
  cudaMalloc(&z, sizeof hz);
  cudaMemcpy(x, hx, sizeof hx, cudaMemcpyHostToDevice);
  cudaMemcpy(k, hk, sizeof hk, cudaMemcpyHostToDevice);
  mix<<<n / 128, 128>>>(x, k, y, z, n);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("launch: %s\n", cudaGetErrorName(e));
    return 1;
  }
  cudaMemcpy(hy, y, sizeof hy, cudaMemcpyDeviceToHost);
  cudaMemcpy(hz, z, sizeof hz, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const int b = i / 128 * 128, t = i % 128;
    const float v = hx[b + (t + 1) % 128] * 2.0f + 1.0f;
    const float want_y = v > 10.0f ? std::sqrt(v) : v * v;
    // In unsigned arithmetic, as the device wraps (a negative int's left
    // shift and overflowing sum are undefined in C++); >> stays arithmetic.
    const unsigned u = static_cast<unsigned>(hk[i]);
    const int want_z = static_cast<int>(
        (u << 3) ^ (static_cast<unsigned>(hk[i] >> 1) + __builtin_popcount(u) + static_cast<unsigned>(hk[i ^ 1])));
    if (hy[i] != want_y || hz[i] != want_z) ++bad;
  }
  std::printf("wrong: %d\n", bad);
  return bad != 0;
}
