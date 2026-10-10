// The packed FP8 conversions sm_89 and later have (PTX cvt to and from e4m3x2 and e5m2x2), over
// values that include NaN, infinities, zeros, subnormals, ties and values past each format's range.
// Every result is printed, so the transcript of a real GPU and of VirtualGPU can be compared
// (nvidia/tests/e2e/run_lowprec.sh cvt). Built for the architecture it runs on; compiled for less
// than sm_89 it prints SKIP.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#include "lowprec_common.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define HAVE_FP8_CVT 1
#endif

constexpr int kForms = 9;

__global__ void convert(const float* in, unsigned* out, int n) {
#ifdef HAVE_FP8_CVT
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = in[i], b = in[(i * 7 + 3) % n];
  unsigned r[kForms];
  unsigned h2;   // the pair as f16x2
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h2) : "f"(b), "f"(a));
  unsigned short s[kForms];
  asm volatile(
      "{ .reg .b16 c0, c1, c2, c3, c4, c5;\n"
      "cvt.rn.satfinite.e4m3x2.f32 c0, %6, %7;\n"
      "cvt.rn.satfinite.e5m2x2.f32 c1, %6, %7;\n"
      "cvt.rn.relu.satfinite.e4m3x2.f32 c2, %6, %7;\n"
      "cvt.rn.relu.satfinite.e5m2x2.f32 c3, %6, %7;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 c4, %8;\n"
      "cvt.rn.satfinite.e5m2x2.f16x2 c5, %8;\n"
      "mov.b16 %0, c0; mov.b16 %1, c1; mov.b16 %2, c2; mov.b16 %3, c3; mov.b16 %4, c4; mov.b16 %5, c5; }\n"
      : "=h"(s[0]), "=h"(s[1]), "=h"(s[2]), "=h"(s[3]), "=h"(s[4]), "=h"(s[5])
      : "f"(b), "f"(a), "r"(h2));
  r[0] = s[0]; r[1] = s[1]; r[2] = s[2]; r[3] = s[3]; r[4] = s[4]; r[5] = s[5];
  // and back to f16x2
  unsigned back4, back5;
  asm volatile("{ .reg .b16 c; mov.b16 c, %2; cvt.rn.f16x2.e4m3x2 %0, c; mov.b16 c, %3; cvt.rn.f16x2.e5m2x2 %1, c; }"
               : "=r"(back4), "=r"(back5) : "h"(s[0]), "h"(s[1]));
  r[6] = back4;
  r[7] = back5;
  r[8] = h2;
  unsigned* o = out + (size_t)i * kForms;
  for (int f = 0; f < kForms; ++f) o[f] = r[f];
#endif
}

int main() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { std::printf("FAIL: no device\n"); return 1; }
  std::printf("# device %s sm_%d%d\n", prop.name, prop.major, prop.minor);
#ifndef HAVE_FP8_CVT
  // The host pass of every build lands here; the device code decides below.
#endif
  constexpr int n = 512;
  std::vector<float> in(n);
  const float fixed[] = {0.f, -0.f, 1.f, -1.f, 0.5f, 1.5f, 2.5f, 3.5f, 448.f, 449.f, 464.f, 465.f, 480.f, 500.f, -500.f, 57344.f,
                         57345.f, 61440.f, 65504.f, 65536.f, 1e5f, 1e-3f, 0.001953125f, 0.0009765625f, 0.00146484375f, 1.52587890625e-5f,
                         7.62939453125e-6f, 1.1444091796875e-5f, 1.0f / 0.0f, -1.0f / 0.0f, 0.0f / 0.0f, 0.1f, 0.3f, 100.1f, 0.0625f,
                         0.015625f, 0.01171875f, 0.017578125f, 240.f, 256.f, 272.f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f,
                         -0.001953125f, -448.f, -449.f, -57344.f, -61440.f};
  int k = 0;
  for (float f : fixed) in[k++] = f;
  for (; k < n; ++k) {
    const uint32_t r = lp::mix((uint32_t)k * 2654435761u + 17);
    const float mant = (float)(r & 0xffffff) / 16777216.f;
    const int e = (int)((r >> 24) % 28) - 14;
    in[k] = std::ldexp(mant * 2.f - 1.f, e);
  }
  float* din = nullptr;
  unsigned* dout = nullptr;
  cudaMalloc(&din, n * sizeof(float));
  cudaMalloc(&dout, (size_t)n * kForms * sizeof(unsigned));
  cudaMemcpy(din, in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dout, 0xEE, (size_t)n * kForms * sizeof(unsigned));
  convert<<<(n + 127) / 128, 128>>>(din, dout, n);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  std::vector<unsigned> out((size_t)n * kForms);
  cudaMemcpy(out.data(), dout, out.size() * sizeof(unsigned), cudaMemcpyDeviceToHost);
  if (out[0] == 0xEEEEEEEE) {   // the kernel body was compiled out: this architecture has no FP8 conversions
    std::printf("SKIP: built for an architecture before sm_89\nPASS\n");
    return 0;
  }
  static const char* names[kForms] = {"e4m3x2.f32", "e5m2x2.f32", "relu.e4m3x2.f32", "relu.e5m2x2.f32",
                                     "e4m3x2.f16x2", "e5m2x2.f16x2", "f16x2.e4m3x2", "f16x2.e5m2x2", "f16x2.f32"};
  for (int f = 0; f < kForms; ++f) {
    for (int i = 0; i < n; ++i) std::printf("%s %d %08x\n", names[f], i, out[(size_t)i * kForms + f]);
  }
  cudaFree(din);
  cudaFree(dout);
  std::printf("PASS\n");
  return 0;
}
