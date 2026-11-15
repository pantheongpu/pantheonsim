// What the bench's three ALU kernels cost when a wave's 64 lanes run the kernel's own arithmetic as host code (round sp72): the floor under any
// compiled tier, since nothing of the simulator is left -- no dispatch, no operand decode, no register file, no counters, no EXEC.
//   g++ -O3 -mavx2 -mfma -ffp-contract=off native_ceiling.cpp -o native_ceiling && ./native_ceiling
// Prints nanoseconds per wave and loop iteration; divide by the wave-instructions the simulator runs per iteration (the round's table) to compare.
// The functions are tools/bench-amd/kernels/bench_amd.hip's, with the lane loop written out.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

// (Eight lanes a vector, written with the compiler's vector type: as scalar code g++ turns popc32 into a popcount instruction and leaves the loop scalar.)
typedef uint32_t V8u __attribute__((vector_size(32), aligned(4)));
static inline V8u int_step(V8u x, uint32_t i) {
  x = x * 1664525u + 1013904223u;
  x ^= x >> 13;
  x = ((x << 7) | (x >> 25)) + (x & 0xFF00FFu);
  x = (x << 3) ^ (x >> 5) ^ i;
  V8u p = x - ((x >> 1) & 0x55555555u);
  p = (p & 0x33333333u) + ((p >> 2) & 0x33333333u);
  p = (((p + (p >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
  return x + p * 2654435761u;
}
[[gnu::noinline]] void valu_int(uint32_t* x, uint32_t iters) {
  for (uint32_t i = 0; i < iters; ++i)
    for (int l = 0; l < 64; l += 8) {
      V8u v;
      __builtin_memcpy(&v, x + l, 32);
      v = int_step(v, i);
      __builtin_memcpy(x + l, &v, 32);
    }
}
[[gnu::noinline]] void valu_f32(float* x, float* y, uint32_t iters) {
  for (uint32_t i = 0; i < iters; ++i)
    for (int l = 0; l < 64; ++l) {
      float a = x[l], b = y[l];
      a = std::fma(a, 0.999f, b);
      b = std::fma(b, 0.5f, a * 0.001f);
      a = std::fma(a, b, 0.25f);
      a = a * 0.5f + 0.125f;
      x[l] = a + b * 0.0625f;
      y[l] = std::fma(y[l], 0.75f, 0.1f);
    }
}
[[gnu::noinline]] void valu_f64(double* x, double* y, uint32_t iters) {
  for (uint32_t i = 0; i < iters; ++i)
    for (int l = 0; l < 64; ++l) {
      double a = x[l], b = y[l];
      a = std::fma(a, 0.999, b);
      b = std::fma(b, 0.5, a * 0.001);
      x[l] = std::fma(a, 0.5, 0.125);
      y[l] = b;
    }
}
template <typename F>
double ns_per_iter(F&& f, uint32_t iters) {
  double best = 1e30;
  for (int rep = 0; rep < 9; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / iters;
    if (ns < best) best = ns;
  }
  return best;
}
int main() {
  const uint32_t iters = 2000000;
  alignas(32) uint32_t xi[64];
  alignas(32) float xf[64], yf[64];
  alignas(32) double xd[64], yd[64];
  for (int l = 0; l < 64; ++l) xi[l] = l * 2654435761u + 1, xf[l] = 1.0f + (l & 15) * 0.01f, yf[l] = 0.5f + (l & 7) * 0.02f, xd[l] = xf[l], yd[l] = yf[l];
  std::printf("valu_int %.2f ns a wave-iteration\n", ns_per_iter([&] { valu_int(xi, iters); }, iters));
  std::printf("valu_f32 %.2f ns a wave-iteration\n", ns_per_iter([&] { valu_f32(xf, yf, iters); }, iters));
  std::printf("valu_f64 %.2f ns a wave-iteration\n", ns_per_iter([&] { valu_f64(xd, yd, iters); }, iters));
  std::printf("check %u %a %a\n", xi[7], static_cast<double>(xf[7]), xd[7]);
  return 0;
}
