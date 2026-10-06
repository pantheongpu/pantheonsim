// The packed narrow conversions sm_100a and sm_120a have, as their SASS F2FP
// runs them: f32 pairs to e2m1 (fp4), e3m2 and e2m3 (fp6), e4m3 and e5m2
// (fp8) and ue8m0 scales, with .satfinite and .relu, and each back to f16x2
// (bf16x2 for ue8m0) -- the conversions a block-scaled GEMM's epilogue makes
// when it writes nvfp4. Over values that include NaN, infinities,
// subnormals and values past each format's range; every result is printed,
// so the SASS run and the PTX run must print the same.
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM110_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define HAVE_FORMS 1
#endif

constexpr int kOut = 12;

__global__ void convert(const float* in, unsigned* out, int n) {
#ifdef HAVE_FORMS
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = in[i], b = in[(i * 7 + 3) % n];
  const float pa = fabsf(a), pb = fabsf(b);   // ue8m0 takes no negatives
  unsigned* o = out + i * kOut;
  unsigned r[kOut];
  asm volatile("{ .reg .b8 c0, c1; .reg .b16 h1, h2, h3, h4, h5;\n"
               "cvt.rn.satfinite.e2m1x2.f32 c0, %12, %13;\n"
               "cvt.rn.relu.satfinite.e2m1x2.f32 c1, %12, %13;\n"
               "cvt.rn.satfinite.e3m2x2.f32 h1, %12, %13;\n"
               "cvt.rn.satfinite.e2m3x2.f32 h2, %12, %13;\n"
               "cvt.rn.satfinite.e4m3x2.f32 h3, %12, %13;\n"
               "cvt.rn.satfinite.e5m2x2.f32 h4, %12, %13;\n"
               "cvt.rz.satfinite.ue8m0x2.f32 h5, %14, %15;\n"
               "mov.b32 %0, {c0, c1, c0, c1};\n"
               "mov.b32 %1, {h1, h2};\n"
               "mov.b32 %2, {h3, h4};\n"
               "cvt.u32.u16 %3, h5;\n"
               "cvt.rn.f16x2.e2m1x2 %4, c0;\n"
               "cvt.rn.f16x2.e2m1x2 %5, c1;\n"
               "cvt.rn.f16x2.e3m2x2 %6, h1;\n"
               "cvt.rn.f16x2.e2m3x2 %7, h2;\n"
               "cvt.rn.f16x2.e4m3x2 %8, h3;\n"
               "cvt.rn.f16x2.e5m2x2 %9, h4;\n"
               "cvt.rn.bf16x2.ue8m0x2 %10, h5;\n"
               "cvt.rp.satfinite.ue8m0x2.f32 h5, %14, %15;\n"
               "cvt.u32.u16 %11, h5; }"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11])
               : "f"(a), "f"(b), "f"(pa), "f"(pb));
  for (int k = 0; k < kOut; ++k) o[k] = r[k];
#else
  (void)in; (void)out; (void)n;
#endif
}

int main() {
  const int n = 4096;
  static float h[n];
  for (int i = 0; i < n; ++i) {
    const unsigned u = static_cast<unsigned>(i) * 2654435761u;
    float x = static_cast<float>(static_cast<int>(u % 40001) - 20000) / 1999.0f;
    if (i % 97 == 0) x = std::nanf("");
    if (i % 89 == 0) x = (i % 2 ? 1 : -1) * INFINITY;
    if (i % 83 == 0) x = (i % 2 ? 1e-40f : -3e-39f);
    if (i % 79 == 0) x = (i % 2 ? 1e30f : -7e5f);
    if (i % 73 == 0) x = 0.0f;
    if (i % 71 == 0) x = -0.0f;
    h[i] = x;
  }
  float* d = nullptr;
  unsigned* o = nullptr;
  cudaMalloc(&d, sizeof h);
  cudaMalloc(&o, n * kOut * sizeof(unsigned));
  cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
  convert<<<n / 128, 128>>>(d, o, n);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  static unsigned r[n * kOut];
  cudaMemcpy(r, o, sizeof r, cudaMemcpyDeviceToHost);
  // Each conversion's results hashed, and a sample of raw values.
  static const char* const names[kOut] = {"e2m1x2 (and .relu)", "e3m2x2, e2m3x2", "e4m3x2, e5m2x2", "ue8m0x2 .rz",
                                          "f16x2 from e2m1x2", "f16x2 from e2m1x2 .relu", "f16x2 from e3m2x2",
                                          "f16x2 from e2m3x2", "f16x2 from e4m3x2", "f16x2 from e5m2x2",
                                          "bf16x2 from ue8m0x2", "ue8m0x2 .rp"};
  for (int k = 0; k < kOut; ++k) {
    unsigned hsh = 2166136261u;
    for (int i = 0; i < n; ++i) hsh = (hsh ^ r[i * kOut + k]) * 16777619u;
    std::printf("%-24s %08x  [97] %08x [89] %08x [83] %08x [5] %08x\n", names[k], hsh, r[97 * kOut + k],
                r[89 * kOut + k], r[83 * kOut + k], r[5 * kOut + k]);
  }
  cudaFree(d);
  cudaFree(o);
  return 0;
}
