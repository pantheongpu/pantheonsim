// Multiply-add contraction: which mul/add pairs run fused, told apart by the
// results (a fused d is exactly fma(a, b, c); an unfused one is the product
// rounded, then the sum). The expected answer for each case is what an RTX
// 3060 does with the same PTX (its ptxas contracts); the simulator reproduces
// it in src/ptx/contract.cpp. Inputs where fused and unfused agree say
// nothing and are skipped.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

constexpr int kCases = 11;
constexpr int kStride = kCases + 2;   // the results, the f64 one, the copied product

__global__ void k(const double* in, unsigned long long* o) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const double da = in[3 * i], db = in[3 * i + 1], dc = in[3 * i + 2];
  const float a = (float)da, b = (float)db, c = (float)dc;
  float r[kCases];
  double d0;
  float t2;
  asm("{ .reg .f32 t; mul.f32 t, %1, %2; add.f32 %0, t, %3; }" : "=f"(r[0]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t; mul.f32 t, %1, %2; sub.f32 %0, %3, t; }" : "=f"(r[1]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t; mul.f32 t, %2, %3; add.f32 %0, t, %4; mov.f32 %1, t; }" : "=f"(r[2]), "=f"(t2) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t; mul.rn.f32 t, %1, %2; add.f32 %0, t, %3; }" : "=f"(r[3]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t; mul.f32 t, %1, %2; add.rn.f32 %0, t, %3; }" : "=f"(r[4]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t, u; mul.f32 t, %1, %2; add.f32 u, %3, %3; mul.f32 u, u, u; add.f32 %0, t, %3; }" : "=f"(r[5]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t; mul.ftz.f32 t, %1, %2; add.ftz.f32 %0, t, %3; }" : "=f"(r[6]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t; mul.f32 t, %1, %2; add.f32 %0, %3, t; }" : "=f"(r[7]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t; mul.f32 t, %1, %2; sub.f32 %0, t, %3; }" : "=f"(r[8]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f32 t; mul.f32 t, %2, %3; add.f32 %0, t, %4; add.f32 %1, t, %2; }" : "=f"(r[9]), "=f"(r[10]) : "f"(a), "f"(b), "f"(c));
  asm("{ .reg .f64 t; mul.f64 t, %1, %2; add.f64 %0, t, %3; }" : "=d"(d0) : "d"(da), "d"(db), "d"(dc));
  for (int j = 0; j < kCases; ++j) o[kStride * i + j] = __float_as_uint(r[j]);
  o[kStride * i + kCases] = __double_as_longlong(d0);
  // The copy is stored: a product used only by a mov nothing reads is dead
  // code to ptxas, and then the add is its only use and does fuse.
  o[kStride * i + kCases + 1] = __float_as_uint(t2);
}

static unsigned fb(float x) { unsigned u; std::memcpy(&u, &x, 4); return u; }
static unsigned long long dbits(double x) { unsigned long long u; std::memcpy(&u, &x, 8); return u; }

int main() {
  const int n = 4096;
  std::vector<double> h(3 * n);
  std::srand(5);
  for (int i = 0; i < 3 * n; ++i)
    h[i] = ((std::rand() / (double)RAND_MAX) - 0.5) * std::ldexp(1.0, (std::rand() % 20) - 10) *
           (1.0 + std::rand() / (double)RAND_MAX / 1e3);
  double* d;
  unsigned long long* o;
  cudaMalloc(&d, 3 * n * sizeof(double));
  cudaMalloc(&o, kStride * n * sizeof(unsigned long long));
  cudaMemcpy(d, h.data(), 3 * n * sizeof(double), cudaMemcpyHostToDevice);
  k<<<n / 128, 128>>>(d, o);
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("kernel failed\nFAIL\n"); return 1; }
  std::vector<unsigned long long> ho(kStride * n);
  cudaMemcpy(ho.data(), o, kStride * n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  const char* names[kCases + 1] = {"t + c", "c - t", "t used by an add and a mov", "mul.rn", "add.rn",
                                   "instructions between", ".ftz", "c + t", "t - c", "two adds: first",
                                   "two adds: second", "f64 t + c"};
  // What an RTX 3060 does: 1 fused, 0 not.
  const int card_fuses[kCases + 1] = {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1};
  int bad = 0;
  for (int j = 0; j <= kCases; ++j) {
    int fused = 0, unfused = 0;
    for (int i = 0; i < n; ++i) {
      const double da = h[3 * i], dbb = h[3 * i + 1], dc = h[3 * i + 2];
      const float a = (float)da, b = (float)dbb, c = (float)dc;
      unsigned long long got = ho[kStride * i + j], f, u;
      if (j == kCases) {
        volatile double p = da * dbb;
        f = dbits(std::fma(da, dbb, dc));
        u = dbits(p + dc);
      } else {
        volatile float p = a * b;
        float fv, uv;
        switch (j) {
          case 1: fv = std::fmaf(-a, b, c); uv = c - p; break;
          case 8: fv = std::fmaf(a, b, -c); uv = p - c; break;
          case 10: fv = std::fmaf(a, b, a); uv = p + a; break;
          case 7: fv = std::fmaf(a, b, c); uv = c + p; break;
          default: fv = std::fmaf(a, b, c); uv = p + c; break;
        }
        f = fb(fv);
        u = fb(uv);
        got &= 0xffffffffull;
      }
      if (f == u) continue;
      if (got == f) ++fused; else if (got == u) ++unfused;
    }
    const bool ok = card_fuses[j] ? (fused > 0 && unfused == 0) : (unfused > 0 && fused == 0);
    std::printf("%-28s fused %4d unfused %4d%s\n", names[j], fused, unfused, ok ? "" : "  <-- the card does otherwise");
    bad += !ok;
  }
  cudaFree(d);
  cudaFree(o);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
