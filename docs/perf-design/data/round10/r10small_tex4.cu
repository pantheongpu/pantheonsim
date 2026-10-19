// Round 10 "small": a 4-channel texture workload (the bench `texture` workload reads one-channel float textures, where a
// texel is one load). float4 point, float4 linear, uchar4 normalized-float linear. Self-checking (PASS).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("FAIL %s\n", cudaGetErrorString(e)); return 1; } } while (0)
__global__ void k_tex4(cudaTextureObject_t pf, cudaTextureObject_t lf, cudaTextureObject_t lu, float* out, int w, int h, int iters) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;
  for (int k = 0; k < iters; ++k) {
    int x = (i * 13u + uint32_t(k) * 7u) % uint32_t(w), y = (i * 5u + uint32_t(k) * 3u) % uint32_t(h);
    float4 a = tex2D<float4>(pf, float(x) + 0.5f, float(y) + 0.5f);
    float4 b = tex2D<float4>(lf, float(x) + 0.5f, float(y) + 0.5f);
    float4 c = tex2D<float4>(lu, float(x) + 0.5f, float(y) + 0.5f);
    acc += a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w + (c.x + c.y + c.z + c.w) * 255.0f;
  }
  out[i] = acc;
}
int main() {
  const int W = 128, H = 128, iters = 1500;
  std::vector<float4> imf(W * H);
  std::vector<uchar4> imu(W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      imf[y * W + x] = make_float4(float(x), float(y), float((x * 7 + y * 13) % 251), 1.0f);
      imu[y * W + x] = make_uchar4(x & 255, y & 255, (x + y) & 255, 255);
    }
  cudaChannelFormatDesc ff = cudaCreateChannelDesc<float4>(), fu = cudaCreateChannelDesc<uchar4>();
  cudaArray_t af, au;
  CK(cudaMallocArray(&af, &ff, W, H));
  CK(cudaMallocArray(&au, &fu, W, H));
  CK(cudaMemcpy2DToArray(af, 0, 0, imf.data(), W * 16, W * 16, H, cudaMemcpyHostToDevice));
  CK(cudaMemcpy2DToArray(au, 0, 0, imu.data(), W * 4, W * 4, H, cudaMemcpyHostToDevice));
  cudaResourceDesc rf{}, ru{};
  rf.resType = ru.resType = cudaResourceTypeArray;
  rf.res.array.array = af;
  ru.res.array.array = au;
  cudaTextureDesc td{};
  td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
  td.readMode = cudaReadModeElementType;
  td.filterMode = cudaFilterModePoint;
  cudaTextureObject_t pf = 0, lf = 0, lu = 0;
  CK(cudaCreateTextureObject(&pf, &rf, &td, nullptr));
  td.filterMode = cudaFilterModeLinear;
  CK(cudaCreateTextureObject(&lf, &rf, &td, nullptr));
  td.readMode = cudaReadModeNormalizedFloat;
  CK(cudaCreateTextureObject(&lu, &ru, &td, nullptr));
  const int blocks = 16, threads = 128, n = blocks * threads;
  float* d;
  CK(cudaMalloc(&d, n * 4));
  k_tex4<<<blocks, threads>>>(pf, lf, lu, d, W, H, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> o(n);
  CK(cudaMemcpy(o.data(), d, n * 4, cudaMemcpyDeviceToHost));
  double sum = 0;
  for (float v : o) { if (!(v == v)) { std::printf("FAIL nan\n"); return 1; } sum += v; }
  std::printf("PASS checksum=%.6f\n", sum);
  return 0;
}
