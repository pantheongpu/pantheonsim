// The worked examples of Whitehead and Fit-Florea, "Precision & Performance: Floating Point and IEEE 754 Compliance for
// NVIDIA GPUs" (https://docs.nvidia.com/cuda/floating-point/), run in a kernel on operands read from memory (nothing is
// folded by the compiler). One line an example: its name, the result's bits, and the answer the paper documents ("-" where
// it documents none, or where its printed inputs do not give its printed answer: docs/numerics/math-functions.md).
// The program's exit status is the number of results that differ from a documented answer.
//
//   assoc_*      section 2.2: (A + B) + C against A + (B + C), A = 2^1 x (1 + 2^-23), B = 2^0 x (1 + 2^-23), C = 2^3 x (1 + 2^-23)
//   fma_*        section 2.3, Figure 1: a = 0x3F800001, b = 0xBF800002, r = a * a + b: 2^-46 fused, 0 as a multiply then an add
//   dot_*        section 3: the serial, FMA and parallel dot products of the two printed vectors
//   cos_5992555  section 5.1: cos(5992555.0), in double and in single precision
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "gpu_api.h"

// A product and a sum the compiler may not fuse.
#if GPU_IS_HIP
#pragma clang fp contract(off)
__device__ float mul_sep(float a, float b) { return a * b; }
__device__ float add_sep(float a, float b) { return a + b; }
#pragma clang fp contract(fast)
#else
__device__ float mul_sep(float a, float b) { return __fmul_rn(a, b); }
__device__ float add_sep(float a, float b) { return __fadd_rn(a, b); }
#endif

__global__ void examples(const float* in, float* out, double* outd) {
  const float A = in[0], B = in[1], C = in[2], a = in[3], b = in[4];
  const float* va = in + 5; const float* vb = in + 9;
  out[0] = add_sep(add_sep(A, B), C);
  out[1] = add_sep(A, add_sep(B, C));
  out[2] = a * a + b;                      // as the paper's code is written: the compiler chooses
  out[3] = fmaf(a, a, b);
  out[4] = add_sep(mul_sep(a, a), b);
  float serial = 0, fused = 0;
  for (int i = 0; i < 4; ++i) { serial = add_sep(serial, mul_sep(va[i], vb[i])); fused = fmaf(va[i], vb[i], fused); }
  out[5] = serial;
  out[6] = fused;
  out[7] = add_sep(add_sep(mul_sep(va[0], vb[0]), mul_sep(va[1], vb[1])), add_sep(mul_sep(va[2], vb[2]), mul_sep(va[3], vb[3])));
  out[8] = cosf(in[13]);
  outd[0] = cos(static_cast<double>(in[13]));
}

int main() {
  const uint32_t words[5] = {0x40000001u, 0x3F800001u, 0x41000001u, 0x3F800001u, 0xBF800002u};
  float in[14];
  memcpy(in, words, sizeof words);
  const float va[4] = {1.907607f, -.7862027f, 1.147311f, .9604002f}, vb[4] = {-.9355000f, -.6915108f, 1.724470f, -.7097529f};
  memcpy(in + 5, va, sizeof va); memcpy(in + 9, vb, sizeof vb);
  in[13] = 5992555.0f;
  float *din, *dout; double* doutd;
  gpuMalloc(&din, sizeof in); gpuMalloc(&dout, 9 * sizeof(float)); gpuMalloc(&doutd, sizeof(double));
  gpuMemcpy(din, in, sizeof in, gpuToDevice);
  examples<<<1, 1>>>(din, dout, doutd);
  if (gpuSync() != gpuOk) { printf("FAILED (the kernel did not run)\n"); return 100; }
  float out[9]; double outd[1];
  gpuMemcpy(out, dout, sizeof out, gpuToHost); gpuMemcpy(outd, doutd, sizeof outd, gpuToHost);
  // The documented answers. dot_*: the paper prints 0x3D653510, 0x3D653501 and 0x3D653500 for vectors it prints to seven digits; those
  // digits give other sums (exactly 0.05595783, not the paper's 0.05595875), so no answer is held here. fma_as_written: Figure 1's
  // GPU output (the default, fused). cos: the paper gives two CPU libraries' answers, not a GPU's.
  static const struct { const char* name; const char* documented; } kLines[9] = {
      {"assoc_ab_c", "41300002"}, {"assoc_a_bc", "41300001"},
#if GPU_IS_HIP    // the paper is about NVIDIA's compiler; whether hipcc fuses a * a + b is its own choice per target
      {"fma_as_written", "-"},
#elif defined(NO_FMAD)   // built with fusing off (section 4.3): the multiply and the add are separate, the x86 answer of Figure 1
      {"fma_as_written", "00000000"},
#else
      {"fma_as_written", "28800000"},
#endif
      {"fma_fused", "28800000"},
      {"fma_separate", "00000000"}, {"dot_serial", "-"}, {"dot_fma", "-"}, {"dot_parallel", "-"}, {"cosf_5992555", "-"}};
  int differ = 0;
  for (int i = 0; i < 9; ++i) {
    uint32_t u; memcpy(&u, &out[i], 4);
    char got[16]; snprintf(got, sizeof got, "%08x", u);
    const bool bad = strcmp(kLines[i].documented, "-") != 0 && strcmp(kLines[i].documented, got) != 0;
    differ += bad;
    printf("%s %s %s%s\n", kLines[i].name, got, kLines[i].documented, bad ? " DIFFERS" : "");
  }
  uint64_t u; memcpy(&u, &outd[0], 8);
  printf("cos_5992555 %016llx -\n", static_cast<unsigned long long>(u));
  return differ;
}
