// sp68: the four refinement steps of nvcc's sqrt.rn (fast path and the scaled slow path) with a reciprocal square root k ulp off the simulator's,
// for every positive float; counts the inputs whose result is not the IEEE root, per k.
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include "vgpu/host_gpu_ref.hpp"
using namespace vgpu::hostgpu;
static float f(uint32_t b) { return std::bit_cast<float>(b); }
static uint32_t u(float v) { return std::bit_cast<uint32_t>(v); }
static float ftz(float v) { return ref::flush(v); }
static uint32_t idiom(uint32_t x, int k) {
  const float xf = f(x);
  if (x - 0x0d000000u <= 0x727fffffu) {   // fast path
    const float y = f(ref::mufu(ref::Fn::Rsq, x) + k);
    const float s = ftz(ftz(y) * xf), h = ftz(ftz(y) * 0.5f);
    const float d = std::fmaf(-s, s, xf);
    return u(std::fmaf(d, h, s));
  }
  // slow path, positive finite nonzero only here (the caller gives those)
  const float sc = std::fmaf(xf, 18446744073709551616.0f, 0.0f);
  const float y = f(ref::mufu(ref::Fn::Rsq, u(sc)) + k);
  const float t = ftz(sc * y), h = ftz(y * 0.5f);
  const float n = ftz(-t + -0.0f);
  const float e = std::fmaf(t, n, sc);
  const float z = std::fmaf(e, h, t);
  return u(ftz(z * 2.3283064365386962891e-10f));
}
int main(int argc, char** argv) {
  const int K = argc > 1 ? atoi(argv[1]) : 4;
  const unsigned T = argc > 2 ? atoi(argv[2]) : 4;
  const uint32_t stride = argc > 3 ? atoi(argv[3]) : 1;
  std::vector<std::atomic<uint64_t>> bad(2 * K + 1), badfast(2 * K + 1);
  std::vector<std::atomic<uint32_t>> lastbad(2 * K + 1), firstfast(2 * K + 1);
  for (auto& b : firstfast) b = 0xffffffffu;
  std::vector<std::atomic<uint32_t>> first(2 * K + 1);
  for (auto& b : first) b = 0;
  std::vector<std::thread> th;
  for (unsigned t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      std::vector<uint64_t> loc(2 * K + 1), locf(2 * K + 1); std::vector<uint32_t> last(2 * K + 1);
      for (uint64_t x = 1 + t * (uint64_t)stride; x < 0x7f800000u; x += (uint64_t)T * stride) {
        const uint32_t want = ref::fsqrt_rn((uint32_t)x);
        for (int k = -K; k <= K; ++k)
          if (idiom((uint32_t)x, k) != want) { last[k + K] = std::max(last[k + K], (uint32_t)x); if (x - 0x0d000000u <= 0x727fffffu) { ++locf[k + K]; uint32_t cur = firstfast[k + K]; while ((uint32_t)x < cur && !firstfast[k + K].compare_exchange_weak(cur, (uint32_t)x)) {} } if (!loc[k + K]++) { uint32_t z = 0; first[k + K].compare_exchange_strong(z, (uint32_t)x); } }
      }
      for (int k = 0; k <= 2 * K; ++k) { bad[k] += loc[k]; badfast[k] += locf[k]; uint32_t cur = lastbad[k]; while (last[k] > cur && !lastbad[k].compare_exchange_weak(cur, last[k])) {} }
    });
  for (auto& x : th) x.join();
  for (int k = -K; k <= K; ++k) std::printf("k=%+d first_fast=0x%08x wrong=%llu fast_path_wrong=%llu first=0x%08x last=0x%08x\n", k, firstfast[k + K].load(), (unsigned long long)bad[k + K].load(), (unsigned long long)badfast[k + K].load(), first[k + K].load(), lastbad[k + K].load());
}
