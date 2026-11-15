// The two case studies of Zahid, Laguna, Le, "Testing GPU Numerics: Finding Numerical Differences Between NVIDIA and AMD
// GPUs" (arXiv 2410.09172), as fixed tests. Each is evaluated twice: with the paper's literals in the kernel (the compiler
// may fold them: build at -O0 and at -O3) and with the same values read from memory (the device computes them).
//
//   Figure 4 (-O0):  fmod(1.5917195493481116e+289, 1.5793E-307) is 1.4424471839615771e-307 with nvcc on an NVIDIA card and
//                    7.1923082856620736e-309 with hipcc on an AMD card, and the whole program prints 8.6551990944767196e-306
//                    and 9.3404611450291972e-306. The exact remainder is 7.1923082856620736e-309 (a subnormal): AMD's.
//   Figure 5 (-O0):  ceil(1.5955E-125) is 0 with nvcc and 1 with hipcc, and the program prints Inf and 1.34887e-306.
//                    The ceiling of a positive number is at least 1.
//
// One line a value: name, bits, %.17g. Nothing is asserted here: the lines are compared against a card's and against the
// paper's figures (tools/numerics/published.tsv; docs/numerics/math-functions.md).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "gpu_api.h"

// Figure 4's compute(), its literals as printed. var_5 is the paper's array argument.
__device__ double fig4(double comp, int var_1, double var_2, double var_3, double var_4, double* var_5, double var_6, double var_7,
                       double var_8, double var_9, double var_10) {
  if (comp >= (var_2 * (var_3 + var_4))) {
    for (int i = 0; i < var_1; ++i) {
      var_5[i] = -0.0 / -1.5942E305 - fmod((+1.7085E-315 + var_6), -1.9289E305 / (-1.2924E-311 - +0.0 + var_7 + +1.3278E-316));
      comp += var_5[i] - fmod((-1.7538E305 * (var_8 / (+0.0 / var_9 - +1.3065E-306))), +1.5793E-307);
      comp += +1.8753E-306 + var_10;
    }
  }
  return comp;
}
// Figure 5's compute().
__device__ double fig5(double comp) {
  double tmp_1 = +1.1147E-307;
  comp += tmp_1 / ceil(+1.5955E-125);
  return comp;
}

__global__ void cases(const double* in, double* scratch, double* out) {
  out[0] = fig4(in[0], static_cast<int>(in[1]), in[2], in[3], in[4], scratch, in[6], in[7], in[8], in[9], in[10]);
  out[1] = -1.7538E305 * (in[8] / (+0.0 / in[9] - +1.3065E-306));          // the dividend of the second fmod
  out[2] = fmod(1.5917195493481116e+289, 1.5793E-307);                     // literals
  out[3] = fmod(in[11], in[12]);                                           // the same values from memory
  out[4] = fig5(in[13]);
  out[5] = ceil(+1.5955E-125);                                             // literal
  out[6] = ceil(in[14]);                                                   // from memory
  out[7] = in[13] + in[15] / ceil(in[14]);                                 // Figure 5 with every operand from memory
}

int main() {
  // Figure 4's input line: comp, var_1, var_2, var_3, var_4, var_5 (the value its array is filled with), var_6 ... var_10; then the
  // operands of the two expressions the paper singles out, and Figure 5's input and literals.
  const double in[16] = {+0.0, 5, +1.7612E-322, +1.1649E-307, -0.0, +0.0, +1.5461E-311, -1.3680E306, +1.1757E-322, +1.7130E-319,
                         +1.6782E-321, 1.5917195493481116e+289, 1.5793E-307, +1.2374E-306, +1.5955E-125, +1.1147E-307};
  double *din, *dscratch, *dout;
  gpuMalloc(&din, sizeof in); gpuMalloc(&dscratch, 16 * sizeof(double)); gpuMalloc(&dout, 8 * sizeof(double));
  double scratch[16];
  for (double& v : scratch) v = in[5];
  gpuMemcpy(din, in, sizeof in, gpuToDevice);
  gpuMemcpy(dscratch, scratch, sizeof scratch, gpuToDevice);
  cases<<<1, 1>>>(din, dscratch, dout);
  if (gpuSync() != gpuOk) { printf("FAILED (the kernel did not run)\n"); return 100; }
  double out[8];
  gpuMemcpy(out, dout, sizeof out, gpuToHost);
  static const char* kNames[8] = {"fig4_program", "fig4_dividend", "fmod_literal", "fmod_memory", "fig5_program", "ceil_literal",
                                  "ceil_memory", "fig5_memory"};
  for (int i = 0; i < 8; ++i) {
    uint64_t u; memcpy(&u, &out[i], 8);
    printf("%s %016llx %.17g\n", kNames[i], static_cast<unsigned long long>(u), out[i]);
  }
  return 0;
}
