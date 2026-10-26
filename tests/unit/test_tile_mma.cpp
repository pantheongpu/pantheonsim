// The tile-product SIMD kernels (simd::Kernels::tile_f32, tile_i32, tile_f64) and the operand code around them
// (include/vgpu/exec/tile_mma.hpp) against the scalar loops they replace in the engines' warpgroup and mma.sync forms, bit for bit, at every
// level this CPU can run: zeros of both signs, subnormals, infinities, NaNs, the extremes, accumulators that saturate, and random values.
//
// The references are the engines' original element-at-a-time forms, written out here: float products rounded and then added in k order,
// integer sums in 64 bits clamped after each block, fused multiply-adds in the current rounding mode.
#pragma GCC optimize("fp-contract=off")

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vgpu/exec/numerics.hpp"
#include "vgpu/exec/tile_mma.hpp"
#include "vgpu/exec/wgmma.hpp"
#include "vgpu/simd/fp16.hpp"
#include "vgpu/simd/simd.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

uint32_t fb(float f) { return std::bit_cast<uint32_t>(f); }
float bf(uint32_t u) { return std::bit_cast<float>(u); }

std::vector<simd::Level> levels() {
  std::vector<simd::Level> v;
  for (int i = 0; i < simd::kLevels; ++i) {
    const auto l = static_cast<simd::Level>(i);
    if (simd::supported(l)) v.push_back(l);
    else std::printf("  [ SKIP ] simd level %s: this CPU cannot run it\n", simd::level_name(l));
  }
  return v;
}

// A float from a mix of everything that breaks arithmetic.
float special_f32(std::mt19937& g) {
  const uint32_t h = g();
  switch (h % 16) {
    case 0: return 0.0f;
    case 1: return -0.0f;
    case 2: return bf(h & 0x807fffffu);                        // a subnormal
    case 3: return bf(0x7f800000u | (h & 0x80000000u));        // an infinity
    case 4: return bf(0x7fc00000u | (h & 0x803fffffu));        // a NaN
    case 5: return bf((h & 0x80000000u) | 0x7f7fffffu);        // the largest
    case 6: return bf((h & 0x80000000u) | 0x00800000u);        // the smallest normal
    case 7: return bf((h & 0x80000000u) | 0x3f800000u);        // one
    default: {
      const uint32_t e = 1 + (h >> 8) % 254;
      return bf((h & 0x80000000u) | (e << 23) | (g() & 0x7fffffu));
    }
  }
}
float tame_f32(std::mt19937& g, int span) {
  const uint32_t h = g();
  return bf((h & 0x80000000u) | ((127 - span + (h >> 8) % (2 * span + 1)) << 23) | (g() & 0x7fffffu));
}

bool same_f32(float a, float b) { return (a != a && b != b) || fb(a) == fb(b); }

struct Shape {
  unsigned M, N, K;
};
const Shape kShapes[] = {{16, 8, 16}, {16, 8, 8}, {16, 16, 8}, {16, 64, 16}, {16, 256, 32}, {16, 24, 7}, {8, 8, 4}, {4, 40, 5},
                         {1, 8, 3},   {3, 16, 9},  {16, 8, 64}, {8, 8, 1},    {5, 32, 12}};

void ref_f32(const Shape& s, const float* A, const float* B, const float* C, float* D) {
  for (unsigned i = 0; i < s.M; ++i)
    for (unsigned j = 0; j < s.N; ++j) {
      float acc = C ? C[i * s.N + j] : 0.0f;
      for (unsigned k = 0; k < s.K; ++k) {
        const float p = A[i * s.K + k] * B[k * s.N + j];
        acc = acc + p;
      }
      D[i * s.N + j] = acc;
    }
}

}  // namespace

VTEST(tile_f32_matches_the_scalar_loop) {
  std::mt19937 g(0x51a9);
  long compared = 0;
  for (simd::Level l : levels()) {
    const simd::Kernels& k = simd::kernels_for(l);
    for (const Shape& s : kShapes)
      for (int mode = 0; mode < 6; ++mode)
        for (int trial = 0; trial < 12; ++trial) {
          std::vector<float> A(s.M * s.K), B(s.K * s.N), C(s.M * s.N), want(s.M * s.N), got(s.M * s.N);
          const auto fill = [&](std::vector<float>& v) {
            for (float& x : v) x = mode == 0 ? tame_f32(g, 6) : mode == 1 ? special_f32(g) : mode == 2 ? tame_f32(g, 100) : mode == 3 ? bf(g()) : mode == 4 ? (g() % 3 ? 0.0f : special_f32(g)) : tame_f32(g, 30);
          };
          fill(A);
          fill(B);
          fill(C);
          const bool null_c = trial % 3 == 0;
          ref_f32(s, A.data(), B.data(), null_c ? nullptr : C.data(), want.data());
          k.tile_f32(s.M, s.N, s.K, A.data(), B.data(), null_c ? nullptr : C.data(), s.N, got.data(), s.N);
          for (size_t i = 0; i < got.size(); ++i) {
            ++compared;
            if (!same_f32(got[i], want[i]))
              throw vtest::Failure(std::string(simd::level_name(l)) + " tile_f32 " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" +
                                   std::to_string(s.K) + " mode " + std::to_string(mode) + " element " + std::to_string(i) + ": got 0x" +
                                   std::to_string(fb(got[i])) + " want 0x" + std::to_string(fb(want[i])));
          }
          if (!null_c) {   // D may be C
            std::vector<float> inplace = C;
            k.tile_f32(s.M, s.N, s.K, A.data(), B.data(), inplace.data(), s.N, inplace.data(), s.N);
            for (size_t i = 0; i < got.size(); ++i)
              if (!same_f32(inplace[i], want[i])) throw vtest::Failure("tile_f32 with D = C differs at element " + std::to_string(i));
          }
        }
  }
  VCHECK(compared > 400000);
}

// tile_f32c is tile_f32 with every NaN of D written as the canonical 0x7fffffff, in the same pass: the other bits are tile_f32's.
VTEST(tile_f32c_is_the_f32_tile_with_canonical_nans) {
  std::mt19937 g(0x6c01);
  long compared = 0, nans = 0;
  for (simd::Level l : levels()) {
    const simd::Kernels& k = simd::kernels_for(l);
    for (const Shape& s : kShapes)
      for (int mode = 0; mode < 6; ++mode)
        for (int trial = 0; trial < 12; ++trial) {
          std::vector<float> A(s.M * s.K), B(s.K * s.N), C(s.M * s.N), want(s.M * s.N), got(s.M * s.N);
          const auto fill = [&](std::vector<float>& v) {
            for (float& x : v) x = mode == 0 ? tame_f32(g, 6) : mode == 1 ? special_f32(g) : mode == 2 ? tame_f32(g, 100) : mode == 3 ? bf(g()) : mode == 4 ? (g() % 3 ? 0.0f : special_f32(g)) : tame_f32(g, 30);
          };
          fill(A);
          fill(B);
          fill(C);
          const bool null_c = trial % 3 == 0;
          k.tile_f32(s.M, s.N, s.K, A.data(), B.data(), null_c ? nullptr : C.data(), s.N, want.data(), s.N);
          k.tile_f32c(s.M, s.N, s.K, A.data(), B.data(), null_c ? nullptr : C.data(), s.N, got.data(), s.N);
          for (size_t i = 0; i < got.size(); ++i) {
            ++compared;
            const bool is_nan = (fb(want[i]) & 0x7fffffffu) > 0x7f800000u;
            nans += is_nan;
            if (is_nan ? fb(got[i]) != 0x7fffffffu : fb(got[i]) != fb(want[i]))
              throw vtest::Failure(std::string(simd::level_name(l)) + " tile_f32c " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K) + " mode " + std::to_string(mode) +
                                   " element " + std::to_string(i) + ": 0x" + std::to_string(fb(got[i])) + " for 0x" + std::to_string(fb(want[i])));
          }
        }
  }
  VCHECK(compared > 400000);
  VCHECK(nans > 5000);   // (the special operands do make NaNs, of both signs and payloads)
}

namespace {
void ref_i32(const Shape& s, const int32_t* A, const int32_t* B, const int32_t* C, unsigned block, bool sat, int32_t* D) {
  for (unsigned i = 0; i < s.M; ++i)
    for (unsigned j = 0; j < s.N; ++j) {
      int64_t acc = C ? C[i * s.N + j] : 0;
      const unsigned blk = sat ? block : s.K;
      for (unsigned k0 = 0; k0 < s.K; k0 += blk) {
        for (unsigned k = k0; k < k0 + blk && k < s.K; ++k) acc += static_cast<int64_t>(A[i * s.K + k]) * static_cast<int64_t>(B[k * s.N + j]);
        if (sat) acc = std::clamp<int64_t>(acc, INT32_MIN, INT32_MAX);
      }
      D[i * s.N + j] = static_cast<int32_t>(acc);
    }
}
}  // namespace

VTEST(tile_i32_matches_the_scalar_loop) {
  std::mt19937 g(0x7e11);
  long compared = 0;
  for (simd::Level l : levels()) {
    const simd::Kernels& k = simd::kernels_for(l);
    for (const Shape& s : kShapes)
      for (int mode = 0; mode < 6; ++mode)
        for (int trial = 0; trial < 12; ++trial) {
          std::vector<int32_t> A(s.M * s.K), B(s.K * s.N), C(s.M * s.N), want(s.M * s.N), got(s.M * s.N);
          for (int32_t& x : A) x = mode == 0 ? int32_t(g() % 256) - 128 : mode == 1 ? int32_t(g() % 256) : mode == 2 ? (g() & 1 ? 127 : -128) : mode == 3 ? int32_t(g() % 16) - 8 : mode == 4 ? (g() & 1 ? 255 : 0) : int32_t(g() % 31) - 15;
          for (int32_t& x : B) x = mode == 0 ? int32_t(g() % 256) - 128 : mode == 1 ? int32_t(g() % 256) : mode == 2 ? (g() & 1 ? 127 : -128) : mode == 3 ? int32_t(g() % 16) - 8 : mode == 4 ? (g() & 1 ? -128 : 255) : int32_t(g() % 31) - 15;
          for (int32_t& x : C) {
            const uint32_t h = g();
            x = h % 5 == 0 ? INT32_MAX - int32_t(h >> 8) % 200000 : h % 5 == 1 ? INT32_MIN + int32_t(h >> 8) % 200000 : h % 5 == 2 ? 0 : static_cast<int32_t>(g());
          }
          const bool null_c = trial % 4 == 0, sat = trial % 2 == 1;
          const unsigned block = trial % 3 == 0 ? s.K : trial % 3 == 1 ? 16 : 3;
          ref_i32(s, A.data(), B.data(), null_c ? nullptr : C.data(), block, sat, want.data());
          k.tile_i32(s.M, s.N, s.K, A.data(), B.data(), null_c ? nullptr : C.data(), s.N, block, sat, got.data(), s.N);
          for (size_t i = 0; i < got.size(); ++i) {
            ++compared;
            if (got[i] != want[i])
              throw vtest::Failure(std::string(simd::level_name(l)) + " tile_i32 " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" +
                                   std::to_string(s.K) + " mode " + std::to_string(mode) + (sat ? " sat" : "") + " block " + std::to_string(block) +
                                   " element " + std::to_string(i) + ": got " + std::to_string(got[i]) + " want " + std::to_string(want[i]));
          }
        }
  }
  VCHECK(compared > 400000);
}

// The integer kernel on operands of every size, not only the 8-bit ones its contract names: a 16-bit dot-product form (VNNI at the Avx512 level) takes
// numbers that fit 16 bits and must leave the rest to the multiplier form, with the same wrapping sum either way; odd K, a K of 1 and the limits of
// 16 bits (where a pair of products reaches 2^31) included. No .sat: that clamp is defined on sums that fit 32 bits. Separately, .sat with
// a block that is odd, as the pair form cannot take, and one that is not a multiple of the pair.
VTEST(tile_i32_wraps_exactly_on_operands_of_any_size) {
  std::mt19937 g(0x16b1);
  long compared = 0;
  const Shape shapes[] = {{16, 8, 16}, {16, 8, 7}, {4, 24, 1}, {16, 16, 33}, {5, 32, 12}, {8, 8, 64}, {1, 8, 2}, {3, 40, 9}, {16, 64, 33}, {5, 72, 12}, {8, 128, 7}, {4, 64, 1}};
  for (simd::Level l : levels()) {
    const simd::Kernels& k = simd::kernels_for(l);
    for (const Shape& s : shapes)
      for (int mode = 0; mode < 7; ++mode)
        for (int trial = 0; trial < 10; ++trial) {
          std::vector<int32_t> A(s.M * s.K), B(s.K * s.N), C(s.M * s.N), want(s.M * s.N), got(s.M * s.N);
          const auto draw = [&]() -> int32_t {
            switch (mode) {
              case 0: return static_cast<int32_t>(g());                                      // any 32-bit number
              case 1: return int32_t(g() % 65536) - 32768;                                   // any 16-bit number
              case 2: return g() & 1 ? 32767 : -32768;                                       // its limits
              case 3: return g() % 50 == 0 ? static_cast<int32_t>(g()) : int32_t(g() % 256) - 128;   // 8-bit with the odd wide one
              case 4: return g() & 1 ? 32768 : -32769;                                       // just outside
              case 5: return int32_t(g() % 3) - 1;
              default: return int32_t(g() % 4096) - 2048;
            }
          };
          for (int32_t& x : A) x = draw();
          for (int32_t& x : B) x = draw();
          for (int32_t& x : C) x = static_cast<int32_t>(g());
          const bool null_c = trial % 3 == 0;
          // wrapping sums: the 64-bit sum cut to 32 bits
          for (unsigned i = 0; i < s.M; ++i)
            for (unsigned j = 0; j < s.N; ++j) {
              uint32_t acc = null_c ? 0u : static_cast<uint32_t>(C[i * s.N + j]);
              for (unsigned kk = 0; kk < s.K; ++kk) acc += static_cast<uint32_t>(A[i * s.K + kk]) * static_cast<uint32_t>(B[kk * s.N + j]);
              want[i * s.N + j] = static_cast<int32_t>(acc);
            }
          k.tile_i32(s.M, s.N, s.K, A.data(), B.data(), null_c ? nullptr : C.data(), s.N, s.K, false, got.data(), s.N);
          for (size_t i = 0; i < got.size(); ++i) {
            ++compared;
            if (got[i] != want[i])
              throw vtest::Failure(std::string(simd::level_name(l)) + " tile_i32 wrapping " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K) + " mode " +
                                   std::to_string(mode) + " element " + std::to_string(i));
          }
        }
    // .sat with every block length, on 8-bit operands: odd blocks, blocks that do not divide K, a block longer than K
    for (const Shape& s : shapes)
      for (unsigned block : {1u, 2u, 3u, 5u, 8u, 16u, 100u})
        for (int trial = 0; trial < 8; ++trial) {
          std::vector<int32_t> A(s.M * s.K), B(s.K * s.N), C(s.M * s.N), want(s.M * s.N), got(s.M * s.N);
          for (int32_t& x : A) x = int32_t(g() % 256) - 128;
          for (int32_t& x : B) x = trial % 2 ? int32_t(g() % 256) : int32_t(g() % 256) - 128;
          for (int32_t& x : C) {
            const uint32_t h = g();
            x = h % 3 == 0 ? INT32_MAX - int32_t(h >> 8) % 3000 : h % 3 == 1 ? INT32_MIN + int32_t(h >> 8) % 3000 : static_cast<int32_t>(g());
          }
          const bool null_c = trial % 4 == 0;
          ref_i32(s, A.data(), B.data(), null_c ? nullptr : C.data(), block, true, want.data());
          k.tile_i32(s.M, s.N, s.K, A.data(), B.data(), null_c ? nullptr : C.data(), s.N, block, true, got.data(), s.N);
          for (size_t i = 0; i < got.size(); ++i) {
            ++compared;
            if (got[i] != want[i])
              throw vtest::Failure(std::string(simd::level_name(l)) + " tile_i32 sat " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K) + " block " +
                                   std::to_string(block) + " element " + std::to_string(i));
          }
        }
  }
  VCHECK(compared > 20000);
}

// C and D with row strides of their own (tcgen05.mma works on Tensor Memory rows 512 words apart, a tile inside a wider one).
VTEST(tile_kernels_take_strided_c_and_d) {
  std::mt19937 g(0x3d3);
  for (simd::Level l : levels()) {
    const simd::Kernels& k = simd::kernels_for(l);
    for (const Shape& s : kShapes)
      for (int trial = 0; trial < 6; ++trial) {
        const size_t ldc = s.N + 8 * (trial % 3) + (trial & 1 ? 24 : 0), ldd = s.N + 16 * (trial % 2) + 8;
        std::vector<float> A(s.M * s.K), B(s.K * s.N), C(s.M * ldc), D(s.M * ldd, bf(0x7fc00001u)), want(s.M * s.N);
        for (float& x : A) x = tame_f32(g, 8);
        for (float& x : B) x = tame_f32(g, 8);
        for (float& x : C) x = tame_f32(g, 12);
        std::vector<float> Cdense(s.M * s.N);
        for (unsigned i = 0; i < s.M; ++i)
          for (unsigned j = 0; j < s.N; ++j) Cdense[i * s.N + j] = C[i * ldc + j];
        ref_f32(s, A.data(), B.data(), Cdense.data(), want.data());
        k.tile_f32(s.M, s.N, s.K, A.data(), B.data(), C.data(), ldc, D.data(), ldd);
        for (unsigned i = 0; i < s.M; ++i) {
          for (unsigned j = 0; j < s.N; ++j)
            if (!same_f32(D[i * ldd + j], want[i * s.N + j])) throw vtest::Failure("strided tile_f32, element " + std::to_string(i * s.N + j));
          for (size_t j = s.N; j < ldd; ++j)
            if (fb(D[i * ldd + j]) != 0x7fc00001u) throw vtest::Failure("strided tile_f32 wrote past a row of D");
        }
        std::vector<int32_t> Ai(s.M * s.K), Bi(s.K * s.N), Ci(s.M * ldc), Di(s.M * ldd, 0x55555555), wi(s.M * s.N), Cid(s.M * s.N);
        for (int32_t& x : Ai) x = int32_t(g() % 256) - 128;
        for (int32_t& x : Bi) x = int32_t(g() % 256) - 128;
        for (int32_t& x : Ci) x = static_cast<int32_t>(g());
        for (unsigned i = 0; i < s.M; ++i)
          for (unsigned j = 0; j < s.N; ++j) Cid[i * s.N + j] = Ci[i * ldc + j];
        const bool sat = trial & 1;
        ref_i32(s, Ai.data(), Bi.data(), Cid.data(), s.K, sat, wi.data());
        k.tile_i32(s.M, s.N, s.K, Ai.data(), Bi.data(), Ci.data(), ldc, s.K, sat, Di.data(), ldd);
        for (unsigned i = 0; i < s.M; ++i) {
          for (unsigned j = 0; j < s.N; ++j)
            if (Di[i * ldd + j] != wi[i * s.N + j]) throw vtest::Failure("strided tile_i32, element " + std::to_string(i * s.N + j));
          for (size_t j = s.N; j < ldd; ++j)
            if (Di[i * ldd + j] != 0x55555555) throw vtest::Failure("strided tile_i32 wrote past a row of D");
        }
      }
  }
}

VTEST(canon_nan_rows_canonicalises_exactly_the_nans) {
  std::mt19937 g(0x6a6e);
  for (int trial = 0; trial < 400; ++trial) {
    const uint32_t rows = 1 + g() % 9, cols = 1 + g() % 41, ld = cols + g() % 7;
    std::vector<uint32_t> w(size_t{rows} * ld), want(w.size());
    for (size_t i = 0; i < w.size(); ++i) {
      w[i] = g() % 3 == 0 ? fb(special_f32(g)) : g();
      want[i] = (w[i] & 0x7fffffffu) > 0x7f800000u && (i % ld) < cols ? 0x7fffffffu : w[i];
    }
    exec::canon_nan_rows(w.data(), rows, cols, ld);
    for (size_t i = 0; i < w.size(); ++i)
      if (w[i] != want[i]) throw vtest::Failure("canon_nan_rows, word " + std::to_string(i));
  }
}

VTEST(tile_f64_matches_the_fma_chain_in_every_rounding_mode) {
  std::mt19937_64 g(0x64f);
  long compared = 0;
  const int modes[] = {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD};
  const int saved = std::fegetround();
  for (simd::Level l : levels()) {
    const simd::Kernels& k = simd::kernels_for(l);
    for (int rm : modes)
      for (const Shape& s : {Shape{8, 8, 4}, Shape{8, 16, 4}, Shape{4, 12, 3}, Shape{2, 5, 7}})
        for (int trial = 0; trial < 60; ++trial) {
          std::vector<double> A(s.M * s.K), B(s.K * s.N), C(s.M * s.N), want(s.M * s.N), got(s.M * s.N);
          const auto val = [&]() -> double {
            const uint64_t h = g();
            switch (h % 12) {
              case 0: return 0.0;
              case 1: return -0.0;
              case 2: return std::bit_cast<double>(0x7ff0000000000000ull | (h & 0x8000000000000000ull));
              case 3: return std::bit_cast<double>(0x7ff8000000000000ull | (h >> 12));
              case 4: return std::bit_cast<double>(h & 0x800fffffffffffffull);   // a subnormal
              case 5: return std::bit_cast<double>((h & 0x8000000000000000ull) | 0x7fefffffffffffffull);
              default: return std::ldexp(static_cast<double>(static_cast<int64_t>(g() >> 11)) / 9007199254740992.0, static_cast<int>(h >> 40) % 40 - 20);
            }
          };
          for (double& x : A) x = val();
          for (double& x : B) x = val();
          for (double& x : C) x = val();
          const bool null_c = trial % 5 == 0;
          std::fesetround(rm);
          for (unsigned i = 0; i < s.M; ++i)
            for (unsigned j = 0; j < s.N; ++j) {
              double acc = null_c ? 0.0 : C[i * s.N + j];
              for (unsigned kk = 0; kk < s.K; ++kk) acc = std::fma(A[i * s.K + kk], B[kk * s.N + j], acc);
              want[i * s.N + j] = acc;
            }
          k.tile_f64(s.M, s.N, s.K, A.data(), B.data(), null_c ? nullptr : C.data(), got.data());
          std::fesetround(saved);
          for (size_t i = 0; i < got.size(); ++i) {
            ++compared;
            const bool nan_both = got[i] != got[i] && want[i] != want[i];
            if (!nan_both && std::bit_cast<uint64_t>(got[i]) != std::bit_cast<uint64_t>(want[i]))
              throw vtest::Failure(std::string(simd::level_name(l)) + " tile_f64 rounding " + std::to_string(rm) + " element " + std::to_string(i));
          }
        }
  }
  std::fesetround(saved);
  VCHECK(compared > 50000);
}

namespace {

// The matrix element by element, as exec_wgmma reads shared memory.
struct RefRead {
  bool ok = true;
  std::vector<uint32_t> raw;   // [mn][k]
};
RefRead ref_read(const std::vector<uint8_t>& smem, const exec::WgmmaDesc& d, bool k_major, uint32_t eb, uint32_t rows, uint32_t K, uint64_t mask) {
  RefRead r;
  r.raw.assign(size_t{rows} * K, 0);
  for (uint32_t mn = 0; mn < rows; ++mn)
    for (uint32_t k = 0; k < K; ++k) {
      const uint64_t at = exec::wgmma_smem_offset(d, k_major, eb, mn, k) & mask;
      if (at + eb > smem.size()) {
        r.ok = false;
        return r;
      }
      uint32_t v = 0;
      std::memcpy(&v, &smem[at], eb);
      r.raw[size_t{mn} * K + k] = v;
    }
  return r;
}

}  // namespace

VTEST(wgmma_operand_reads_match_the_element_by_element_form) {
  std::mt19937 g(0x2b6d);
  const exec::TileElem types[] = {exec::TileElem::F16, exec::TileElem::BF16, exec::TileElem::TF32, exec::TileElem::E4M3,
                                  exec::TileElem::E5M2, exec::TileElem::S8, exec::TileElem::U8};
  long compared = 0, refused = 0;
  for (int trial = 0; trial < 1500; ++trial) {
    const exec::TileElem t = types[trial % 7];
    const uint32_t eb = exec::tile_elem_bytes(t);
    const bool k_major = trial % 5 != 0 && (eb != 4 && eb != 1 ? true : true);
    const bool mn_major = !k_major && eb == 2;   // only the 16-bit forms are MN-major
    const bool km = mn_major ? false : true;
    const uint32_t rowsets[] = {8, 16, 24, 64, 128, 256};
    const uint32_t rows = rowsets[g() % 6];
    const uint32_t K = eb == 2 ? 16 : eb == 4 ? 8 : 32;
    std::vector<uint8_t> smem(size_t(g() % 3 == 0 ? 6000 : 65536));
    for (uint8_t& b : smem) b = static_cast<uint8_t>(g());
    exec::WgmmaDesc d;
    d.swizzle = (uint32_t[]){0, 32, 64, 128}[g() % 4];
    d.start = 16 * (g() % 64);
    d.lbo = 16 * (g() % 200);
    d.sbo = 16 * (g() % 200);
    const uint64_t mask = trial % 4 == 0 ? 0xffffff : ~uint64_t{0};
    const bool negate = trial % 3 == 0, transpose = trial % 2 == 0;
    const RefRead ref = ref_read(smem, d, km, eb, rows, K, mask);
    const bool is_int = exec::tile_elem_is_int(t);
    std::vector<float> gf(size_t{rows} * K);
    std::vector<int32_t> gi(size_t{rows} * K);
    const bool ok = is_int ? exec::wgmma_read_i32(smem.data(), smem.size(), d, km, t, rows, K, negate, transpose, mask, gi.data())
                           : exec::wgmma_read_f32(smem.data(), smem.size(), d, km, t, rows, K, negate, transpose, mask, gf.data());
    if (ok != ref.ok) throw vtest::Failure("trial " + std::to_string(trial) + ": the read " + (ok ? "succeeded" : "was refused") + " where the element form " + (ref.ok ? "reads it" : "is out of range"));
    if (!ok) {
      ++refused;
      continue;
    }
    for (uint32_t mn = 0; mn < rows; ++mn)
      for (uint32_t k = 0; k < K; ++k) {
        const size_t at = transpose ? size_t{k} * rows + mn : size_t{mn} * K + k;
        const uint32_t raw = ref.raw[size_t{mn} * K + k];
        ++compared;
        if (is_int) {
          const int32_t want = negate ? -exec::tile_decode_i32(t, raw) : exec::tile_decode_i32(t, raw);
          if (gi[at] != want) throw vtest::Failure("trial " + std::to_string(trial) + ": int element (" + std::to_string(mn) + "," + std::to_string(k) + ")");
        } else {
          const float dec = exec::tile_decode_f32(t, raw);
          const float want = negate ? bf(fb(dec) ^ 0x80000000u) : dec;
          if (!same_f32(gf[at], want)) throw vtest::Failure("trial " + std::to_string(trial) + ": float element (" + std::to_string(mn) + "," + std::to_string(k) + ")");
        }
      }
  }
  VCHECK(compared > 500000);
  VCHECK(refused > 20);
}

VTEST(tile_decode_f32_matches_the_scalar_decoders) {
  // f16 against the compiler's conversion, all 65536 codes; bf16 and tf32 by their definitions; fp8 against fp8_value.
  for (uint32_t h = 0; h < 65536; ++h) {
    const float want = static_cast<float>(std::bit_cast<_Float16>(static_cast<uint16_t>(h)));
    if (!same_f32(exec::tile_decode_f32(exec::TileElem::F16, h), want)) throw vtest::Failure("f16 code " + std::to_string(h));
    if (fb(exec::tile_decode_f32(exec::TileElem::BF16, h)) != (h << 16)) throw vtest::Failure("bf16 code " + std::to_string(h));
  }
  for (uint32_t b = 0; b < 256; ++b) {
    const float e4 = static_cast<float>(exec::fp8_value(b, false)), e5 = static_cast<float>(exec::fp8_value(b, true));
    if (!same_f32(exec::tile_decode_f32(exec::TileElem::E4M3, b), e4)) throw vtest::Failure("e4m3 code " + std::to_string(b));
    if (!same_f32(exec::tile_decode_f32(exec::TileElem::E5M2, b), e5)) throw vtest::Failure("e5m2 code " + std::to_string(b));
    if (exec::tile_decode_i32(exec::TileElem::S8, b) != static_cast<int8_t>(b) || exec::tile_decode_i32(exec::TileElem::U8, b) != static_cast<int32_t>(b))
      throw vtest::Failure("int8 code " + std::to_string(b));
  }
  std::mt19937 g(5);
  for (int i = 0; i < 100000; ++i) {
    const uint32_t w = g();
    if (fb(exec::tile_decode_f32(exec::TileElem::TF32, w)) != (w & 0xffffe000u)) throw vtest::Failure("tf32 word");
  }
}

VTEST(accumulator_registers_and_tiles_convert_back_and_forth) {
  std::mt19937 g(0x99);
  for (int trial = 0; trial < 300; ++trial) {
    const uint32_t N = 8 * (1 + g() % 32);
    for (int kind = 0; kind < 3; ++kind) {   // f32, f16, s32
      const uint32_t nregs = kind == 1 ? N / 4 : N / 2;
      std::vector<uint32_t> regs(size_t{nregs} * 32);
      for (uint32_t& w : regs) w = kind == 1 ? (static_cast<uint32_t>(fb(tame_f32(g, 5)) >> 16) & 0xbfffu) | 0x3c000000u : kind == 0 ? fb(special_f32(g)) : g();
      if (kind == 1)
        for (uint32_t& w : regs) {
          const auto half = [&] { return static_cast<uint32_t>(g() & 0x83ffu) | (static_cast<uint32_t>(8 + g() % 16) << 10); };
          w = half() | half() << 16;
        }
      std::vector<float> tf(size_t{16} * N);
      std::vector<int32_t> ti(size_t{16} * N);
      std::vector<uint32_t> back(regs.size());
      if (kind == 2) {
        exec::wgmma_acc_to_tile_i32(regs.data(), N, ti.data());
        // element by element: register 4j + 2h + u, lane 4g + t is row g + 8h, column 8j + 2t + u
        for (uint32_t j = 0; j < N / 8; ++j)
          for (uint32_t h = 0; h < 2; ++h)
            for (uint32_t u = 0; u < 2; ++u)
              for (uint32_t lane = 0; lane < 32; ++lane)
                if (static_cast<uint32_t>(ti[size_t{lane / 4 + 8 * h} * N + 8 * j + 2 * (lane % 4) + u]) != regs[size_t{4 * j + 2 * h + u} * 32 + lane])
                  throw vtest::Failure("s32 tile layout, N " + std::to_string(N));
        exec::wgmma_tile_to_acc_i32(ti.data(), N, back.data());
      } else {
        exec::wgmma_acc_to_tile(regs.data(), N, kind == 1, tf.data());
        for (uint32_t j = 0; j < N / 8; ++j)
          for (uint32_t h = 0; h < 2; ++h)
            for (uint32_t u = 0; u < 2; ++u)
              for (uint32_t lane = 0; lane < 32; ++lane) {
                const float got = tf[size_t{lane / 4 + 8 * h} * N + 8 * j + 2 * (lane % 4) + u];
                float want;
                if (kind == 1) want = simd::f16_to_f32(static_cast<uint16_t>(regs[size_t{2 * j + h} * 32 + lane] >> (16 * u)));
                else want = bf(regs[size_t{4 * j + 2 * h + u} * 32 + lane]);
                if (!same_f32(got, want)) throw vtest::Failure(std::string(kind == 1 ? "f16" : "f32") + " tile layout, N " + std::to_string(N));
              }
        exec::wgmma_tile_to_acc(tf.data(), N, kind == 1, back.data());
      }
      for (size_t i = 0; i < regs.size(); ++i) {
        uint32_t want = regs[i];
        if (kind == 0 && (want & 0x7fffffffu) > 0x7f800000u) want = 0x7fffffffu;   // a NaN comes back canonical
        if (kind == 1 && (((want & 0x7c00u) == 0x7c00u && (want & 0x3ffu)) || (((want >> 16) & 0x7c00u) == 0x7c00u && ((want >> 16) & 0x3ffu)))) continue;
        if (back[i] != want) throw vtest::Failure("round trip of kind " + std::to_string(kind) + ", N " + std::to_string(N) + ", word " + std::to_string(i));
      }
    }
  }
  // A tile whose NaNs are already canonical (tile_f32c's) goes the same way without the pass for them.
  {
    std::vector<float> c(16 * 8, 1.5f);
    c[3] = bf(0x7fffffffu);
    std::vector<uint32_t> a(4 * 32), b(4 * 32);
    exec::wgmma_tile_to_acc(c.data(), 8, false, a.data(), false);
    exec::wgmma_tile_to_acc(c.data(), 8, false, b.data(), true);
    VCHECK(a == b);
  }
  // NaNs written back: canonical in both accumulator formats.
  std::vector<float> t(16 * 8, bf(0x7fc12345u));
  std::vector<uint32_t> regs(4 * 32), regh(2 * 32);
  exec::wgmma_tile_to_acc(t.data(), 8, false, regs.data());
  for (uint32_t w : regs) VCHECK_EQ(w, 0x7fffffffu);
  t.assign(16 * 8, bf(0xffc00001u));
  exec::wgmma_tile_to_acc(t.data(), 8, true, regh.data());
  for (uint32_t w : regh) VCHECK_EQ(w, 0x7fff7fffu);
}

VTEST_MAIN
