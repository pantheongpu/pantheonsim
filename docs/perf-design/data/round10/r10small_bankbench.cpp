// Per-call cost of sass_bank_passes / ptx_bank_passes for warps that are not on a stride, ns per call (minimum of 7 runs).
// build: g++-12 -O3 -std=c++20 -I<include dir holding vgpu/exec/mem_counters.hpp> r10small_bankbench.cpp
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include "vgpu/exec/mem_counters.hpp"
namespace mc = vgpu::exec::memcount;
struct Rng { uint64_t s = 88172645463325252ull; uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; } };
int main() {
  Rng r;
  const int kSets = 256;
  struct Shape { const char* name; std::vector<std::vector<uint64_t>> sets; };
  std::vector<Shape> shapes;
  auto mk = [&](const char* n, auto gen) {
    Shape sh{n, {}};
    for (int i = 0; i < kSets; ++i) { std::vector<uint64_t> a(32); gen(a); sh.sets.push_back(a); }
    shapes.push_back(sh);
  };
  mk("perm_one_row", [&](std::vector<uint64_t>& a) { unsigned p[32]; for (unsigned i = 0; i < 32; ++i) p[i] = i; for (unsigned i = 31; i > 0; --i) std::swap(p[i], p[r.next() % (i + 1)]); for (unsigned l = 0; l < 32; ++l) a[l] = (1024 + p[l]) * 4; });
  mk("perm_random_rows", [&](std::vector<uint64_t>& a) { unsigned p[32]; for (unsigned i = 0; i < 32; ++i) p[i] = i; for (unsigned i = 31; i > 0; --i) std::swap(p[i], p[r.next() % (i + 1)]); for (unsigned l = 0; l < 32; ++l) a[l] = (1024 + 32 * (r.next() % 16) + p[l]) * 4; });
  mk("two_rows_tile", [&](std::vector<uint64_t>& a) { for (unsigned l = 0; l < 32; ++l) a[l] = (1024 + (l / 16) * 33 * 16 + (l % 16) * 33) * 4 + 0; });
  mk("rand_small_conflict", [&](std::vector<uint64_t>& a) { for (unsigned l = 0; l < 32; ++l) a[l] = (1024 + r.next() % 48) * 4; });
  mk("every_eighth", [&](std::vector<uint64_t>& a) { for (unsigned l = 0; l < 32; ++l) a[l] = (1024 + (l * 8 + (l / 4)) % 256) * 4; });
  volatile uint64_t sink = 0;
  volatile uint32_t v_lanes = 32, v_width = 4;   // not compile-time constants, as in the engines
  volatile uint64_t v_mask = 0xFFFFFFFFu;
  for (auto& sh : shapes) {
    for (int engine = 0; engine < 2; ++engine) {
      double best = 1e9;
      for (int rep = 0; rep < 7; ++rep) {
        const int n = 400000;
        auto t0 = std::chrono::steady_clock::now();
        uint64_t acc = 0;
        for (int i = 0; i < n; ++i) {
          const auto& a = sh.sets[i % kSets];
          acc += engine ? mc::sass_bank_passes(a.data(), v_mask, v_lanes, v_width) : mc::ptx_bank_passes(a.data(), v_mask, v_lanes, v_width);
        }
        sink += acc;
        best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / n);
      }
      std::printf("%-22s %-5s %6.1f ns\n", sh.name, engine ? "sass" : "ptx", best);
    }
  }
  return 0;
}
