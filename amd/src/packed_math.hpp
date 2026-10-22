// The arithmetic of the packed 16-bit float instructions (v_pk_fma_f16, v_pk_add_f16, v_pk_mul_f16) in the form the
// interpreter runs it: a whole wave at a time, integer conversions in place of the compiler's calls into libgcc, and
// the fused multiply-add on the host's FMA unit where it has one. Kept in a header so a unit test can hold it against
// the straightforward version it replaced, bit for bit (amd/tests/unit/test_packed_math.cpp).
//
// What the instructions do, per half: both sources widened to float, the operation done in float (a product of two
// halves is exact in a float; the sum is rounded once, to float), and the float rounded once more, to a half, to
// nearest even. Both roundings are the host's own, so the answer is bit for bit what the old code gave. The one thing
// the old code left to the compiler is which of two NaN operands an x86 add or multiply passes on: that order is fixed
// here (kAddFirstX, kMulFirstX) to the one the recorded results of test_amd_gcn_same_bits show, and a fused
// multiply-add that has a NaN in it is done by libm's fmaf, as before.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "amd_simd_cap.hpp"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace vgpu::amd::packed {

// The float a binary16 value (its 16 bits) converts to: exact, as `static_cast<float>(_Float16)` is, a NaN's payload
// shifted up and made quiet. Integer work only: the baseline x86-64 has no F16C, so the compiler's conversion is a
// call into libgcc.
inline float half_to_float(uint16_t h) {
  const uint32_t sign = uint32_t{h & 0x8000u} << 16;
  const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
  uint32_t bits;
  if (e == 0x1F) {   // infinity or NaN
    bits = sign | 0x7F800000u | (m << 13) | (m ? 0x400000u : 0u);
  } else if (e) {    // normal: rebias 15 -> 127
    bits = sign | ((e + 112) << 23) | (m << 13);
  } else if (!m) {   // zero
    bits = sign;
  } else {           // subnormal: m * 2^-24, normalised
    const uint32_t top = 31 - static_cast<uint32_t>(__builtin_clz(m));   // 0..9
    bits = sign | ((top + 103) << 23) | (((m << (10 - top)) & 0x3FF) << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// The binary16 a float rounds to, to nearest even, as `static_cast<_Float16>(float)` gives it: infinity from 65520,
// subnormal halves rounded at their own spacing, a NaN made quiet with the top ten bits of its payload.
inline uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u, a = x & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | ((a & 0x7FFFFFu) >> 13) | 0x200u);   // a NaN: quiet, top bits of its payload
  if (a >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);   // 65520 and up, and infinity
  if (a >= 0x38800000u) {                                                 // a normal half
    const uint32_t h = (a - 0x38000000u + 0xFFFu + ((a >> 13) & 1)) >> 13;
    return static_cast<uint16_t>(sign | h);
  }
  // Below 2^-14: a subnormal half, or zero. The float's significand (with its hidden bit) shifted down to units of
  // 2^-24 and rounded to nearest even.
  const uint32_t e = a >> 23;
  if (e < 102) return static_cast<uint16_t>(sign);   // under 2^-25 (half the smallest half), or zero
  const uint32_t m = (a & 0x7FFFFFu) | 0x800000u, shift = 126 - e;   // 14 to 24
  uint32_t r = m >> shift;
  const uint32_t rest = m & ((1u << shift) - 1), half = 1u << (shift - 1);
  if (rest > half || (rest == half && (r & 1))) ++r;
  return static_cast<uint16_t>(sign | r);
}

enum class Op { Fma, Add, Mul };

// One half of one source: where its 16 bits come from, and the sign flip the instruction asks for. A register's lanes
// are read from `row` (the 32-bit words, `shift` 0 or 16 picking the half); with no row the half is `imm`.
struct HalfSource {
  const uint32_t* row = nullptr;
  uint16_t imm = 0;
  uint8_t shift = 0;
  uint16_t flip = 0;   // 0x8000 where the source is negated
  uint16_t at(uint32_t lane) const {
    const uint16_t bits = row ? static_cast<uint16_t>(row[lane] >> shift) : imm;
    return bits ^ flip;
  }
};

struct WaveArgs {
  Op op = Op::Fma;
  bool clamp = false;
  uint64_t exec = 0;
  HalfSource src[3][2];   // source k, low result then high
  uint32_t* dst = nullptr;
};

// Which operand of an add or a multiply is the first, for the x86 rule that two NaN operands give the first's payload
// (found by trying both against the recorded results of test_amd_gcn_same_bits, which have NaNs of different payloads
// among their operands).
#ifndef PACKED_ADD_FIRST_X
#define PACKED_ADD_FIRST_X 1
#endif
#ifndef PACKED_MUL_FIRST_X
#define PACKED_MUL_FIRST_X 1
#endif
inline float add_ordered(float first, float second) {
#if defined(__x86_64__)
  __asm__("addss %1, %0" : "+x"(first) : "x"(second));
  return first;
#else
  return first + second;
#endif
}
inline float mul_ordered(float first, float second) {
#if defined(__x86_64__)
  __asm__("mulss %1, %0" : "+x"(first) : "x"(second));
  return first;
#else
  return first * second;
#endif
}

// libm's fmaf, called as such wherever this is compiled, and with its first two arguments in a fixed order (through a pointer
// the compiler cannot see through). A fused multiply-add with two or three NaN operands passes on one of them by the
// position it has in the call, and the interpreter's recorded results (test_amd_gcn_same_bits) are those of a build that
// put the multiplicands in the call the other way round from the source (the compiler's choice, as with an add or a
// multiply); PACKED_FMA_SWAPPED says so, and the test holds it to the recorded bits.
#ifndef PACKED_FMA_SWAPPED
#define PACKED_FMA_SWAPPED 1
#endif
inline float fmaf_libm(float x, float y, float z) {
  static float (*volatile const fma_fn)(float, float, float) = &::fmaf;
  return PACKED_FMA_SWAPPED ? fma_fn(y, x, z) : fma_fn(x, y, z);
}

template <bool Hw>
#if defined(__GNUC__)
__attribute__((always_inline))
#endif
inline void packed_f16_lanes(const WaveArgs& a) {
  for (uint64_t todo = a.exec; todo; todo &= todo - 1) {
    const uint32_t lane = static_cast<uint32_t>(__builtin_ctzll(todo));
    uint32_t r = 0;
    for (uint32_t half = 0; half < 2; ++half) {
      const float x = half_to_float(a.src[0][half].at(lane)), y = half_to_float(a.src[1][half].at(lane));
      float v;
      if (a.op == Op::Add) {
        v = x + y;
        if (v != v) v = PACKED_ADD_FIRST_X ? add_ordered(x, y) : add_ordered(y, x);
      } else if (a.op == Op::Mul) {
        v = x * y;
        if (v != v) v = PACKED_MUL_FIRST_X ? mul_ordered(x, y) : mul_ordered(y, x);
      } else {
        const float z = half_to_float(a.src[2][half].at(lane));
        v = Hw ? __builtin_fmaf(x, y, z) : fmaf_libm(x, y, z);
        if (v != v) v = fmaf_libm(x, y, z);   // (libm's, whichever way the unit above ordered its NaNs)
      }
      if (a.clamp) v = v != v || v <= 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
      r |= static_cast<uint32_t>(float_to_half(v)) << (16 * half);
    }
    a.dst[lane] = r;
  }
}

#if defined(__x86_64__)
__attribute__((target("fma"))) inline void packed_f16_lanes_fma(const WaveArgs& a) { packed_f16_lanes<true>(a); }
inline bool host_has_fma() {
  static const bool has = !simd_capped_to_base() && __builtin_cpu_supports("fma");
  return has;
}

// The same eight lanes at a time on AVX2: F16C converts a vector of halves to floats and back (to nearest even, the
// rounding of the scalar code above; a NaN comes out quiet with the top ten bits of its payload, as float_to_half has it),
// and the multiply-add is the FMA unit's. A lane whose result is a NaN is redone by the scalar code's rule (the order of
// two NaN operands, and libm's fmaf), as in packed_f16_lanes. Not used where MXCSR flushes denormals, which F16C's
// conversions would honour and the integer ones do not.
__attribute__((target("avx2,fma,f16c"))) inline __m256 half_lanes_avx2(const HalfSource& h, uint32_t base) {
  if (!h.row) return _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(h.imm ^ h.flip)));
  __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(h.row + base));
  v = h.shift ? _mm256_srli_epi32(v, 16) : _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF));
  v = _mm256_xor_si256(v, _mm256_set1_epi32(h.flip));
  const __m256i packed = _mm256_packus_epi32(v, v);   // per 128-bit half: lanes 0-3 twice, 4-7 twice
  const __m128i halves = _mm_unpacklo_epi64(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
  return _mm256_cvtph_ps(halves);
}

__attribute__((target("avx2,fma,f16c"))) inline void packed_f16_wave_avx2(const WaveArgs& a) {
  alignas(32) float x[8], y[8], z[8], r[8];
  for (uint32_t base = 0; base < 64; base += 8) {
    const uint32_t active = static_cast<uint32_t>(a.exec >> base) & 0xFF;
    if (!active) continue;
    __m128i out_half[2];
    for (uint32_t half = 0; half < 2; ++half) {
      const __m256 vx = half_lanes_avx2(a.src[0][half], base), vy = half_lanes_avx2(a.src[1][half], base);
      __m256 vz = _mm256_setzero_ps(), v;
      if (a.op == Op::Add) v = _mm256_add_ps(vx, vy);
      else if (a.op == Op::Mul) v = _mm256_mul_ps(vx, vy);
      else {
        vz = half_lanes_avx2(a.src[2][half], base);
        v = _mm256_fmadd_ps(vx, vy, vz);
      }
      if (_mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q))) {
        // Some lane is a NaN (a NaN operand, or infinity times zero): its value by the scalar code's rule.
        _mm256_store_ps(x, vx), _mm256_store_ps(y, vy), _mm256_store_ps(z, vz), _mm256_store_ps(r, v);
        for (uint32_t i = 0; i < 8; ++i) {
          if (r[i] == r[i]) continue;
          if (a.op == Op::Add) r[i] = PACKED_ADD_FIRST_X ? add_ordered(x[i], y[i]) : add_ordered(y[i], x[i]);
          else if (a.op == Op::Mul) r[i] = PACKED_MUL_FIRST_X ? mul_ordered(x[i], y[i]) : mul_ordered(y[i], x[i]);
          else r[i] = fmaf_libm(x[i], y[i], z[i]);
        }
        v = _mm256_load_ps(r);
      }
      out_half[half] = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    const __m256i result = _mm256_or_si256(_mm256_cvtepu16_epi32(out_half[0]), _mm256_slli_epi32(_mm256_cvtepu16_epi32(out_half[1]), 16));
    if (active == 0xFF) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a.dst + base), result);
    } else {
      const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
      const __m256i mask = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(static_cast<int>(active)), bit), bit);
      _mm256_maskstore_epi32(reinterpret_cast<int*>(a.dst + base), mask, result);
    }
  }
}

// Whether the AVX2 code may run: the host has the units, and its MXCSR neither flushes denormals nor reads them as zero.
inline bool host_has_avx2_f16c() {
  static const bool has = !simd_capped_to_base() && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
  return has;
}
#endif

// Every lane of EXEC gets its result.
inline void packed_f16_wave(const WaveArgs& a) {
#if defined(__x86_64__)
  if (!a.clamp && host_has_avx2_f16c() && (_mm_getcsr() & ((1u << 15) | (1u << 6))) == 0 && (_mm_getcsr() & 0x6000) == 0)
    return packed_f16_wave_avx2(a);
  if (a.op == Op::Fma && host_has_fma()) return packed_f16_lanes_fma(a);
#endif
  packed_f16_lanes<false>(a);
}

// v_dot2_f32_f16 and v_dot2c_f32_f16: two products of halves, each exact in a double, added to a float in double and
// rounded once to a float. The addends go in the order the interpreter always summed them -- the VOP3P form adds the
// float first and then the low and the high product, the VOP2 form (dot2c) adds the high product to the low and
// then the float -- since a double add rounds at every step. As above, a lane whose sum is a NaN is handed back.
struct DotArgs {
  bool c_form = false;           // v_dot2c_f32_f16: (p1 + p0) + c, c being the destination's old value
  uint64_t exec = 0;
  HalfSource a[2], b[2];         // the two sources' halves: [0] the low product's, [1] the high one's
  const uint32_t* crow = nullptr;   // the float addend's lanes, or
  uint32_t cimm = 0;                // its bits, the same in every lane
  uint32_t cand = 0xFFFFFFFFu;      // absolute value: 0x7FFFFFFF
  uint32_t cflip = 0;               // negation: 0x80000000
  uint32_t* dst = nullptr;
};

inline uint64_t dot2_f16_lanes(const DotArgs& d) {
  uint64_t redo = 0;
  for (uint64_t todo = d.exec; todo; todo &= todo - 1) {
    const uint32_t lane = static_cast<uint32_t>(__builtin_ctzll(todo));
    const double a0 = half_to_float(d.a[0].at(lane)), b0 = half_to_float(d.b[0].at(lane));
    const double a1 = half_to_float(d.a[1].at(lane)), b1 = half_to_float(d.b[1].at(lane));
    uint32_t cb = d.crow ? d.crow[lane] : d.cimm;
    cb = (cb & d.cand) ^ d.cflip;
    float cf;
    std::memcpy(&cf, &cb, 4);
    const double c = cf;
    double sum;
    if (d.c_form) {
      const double p0 = a0 * b0, p1 = a1 * b1;
      sum = (p1 + p0) + c;
    } else {
      sum = c;
      sum += a0 * b0;
      sum += a1 * b1;
    }
    if (sum != sum) {
      redo |= uint64_t{1} << lane;
      continue;
    }
    const float r = static_cast<float>(sum);
    uint32_t rb;
    std::memcpy(&rb, &r, 4);
    d.dst[lane] = rb;
  }
  return redo;
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma,f16c"))) inline __m256d widen_part(__m256 v, int part) {
  return _mm256_cvtps_pd(part ? _mm256_extractf128_ps(v, 1) : _mm256_castps256_ps128(v));
}

// The same eight lanes at a time on AVX2: the halves widened by F16C, the products and sums in four-wide doubles in the
// order above (a product of two halves is exact in a double, so a multiply-add the compiler fuses rounds the same), the
// float result by the rounding of MXCSR (nearest, checked by the caller). A lane whose sum is a NaN is left alone and
// handed back, as in the scalar code.
__attribute__((target("avx2,fma,f16c"))) inline uint64_t dot2_f16_wave_avx2(const DotArgs& d) {
  uint64_t redo = 0;
  for (uint32_t base = 0; base < 64; base += 8) {
    const uint32_t active = static_cast<uint32_t>(d.exec >> base) & 0xFF;
    if (!active) continue;
    const __m256 a0 = half_lanes_avx2(d.a[0], base), b0 = half_lanes_avx2(d.b[0], base);
    const __m256 a1 = half_lanes_avx2(d.a[1], base), b1 = half_lanes_avx2(d.b[1], base);
    __m256i cb = d.crow ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d.crow + base)) : _mm256_set1_epi32(static_cast<int>(d.cimm));
    cb = _mm256_xor_si256(_mm256_and_si256(cb, _mm256_set1_epi32(static_cast<int>(d.cand))), _mm256_set1_epi32(static_cast<int>(d.cflip)));
    const __m256 c = _mm256_castsi256_ps(cb);
    __m128 res[2];
    int nan_bits = 0;
    for (int part = 0; part < 2; ++part) {
      const __m256d p0 = _mm256_mul_pd(widen_part(a0, part), widen_part(b0, part)), p1 = _mm256_mul_pd(widen_part(a1, part), widen_part(b1, part));
      const __m256d cd = widen_part(c, part);
      const __m256d sum = d.c_form ? _mm256_add_pd(_mm256_add_pd(p1, p0), cd) : _mm256_add_pd(_mm256_add_pd(cd, p0), p1);
      nan_bits |= _mm256_movemask_pd(_mm256_cmp_pd(sum, sum, _CMP_UNORD_Q)) << (4 * part);
      res[part] = _mm256_cvtpd_ps(sum);
    }
    const __m256 result = _mm256_insertf128_ps(_mm256_castps128_ps256(res[0]), res[1], 1);
    const uint32_t write = active & ~static_cast<uint32_t>(nan_bits);
    redo |= static_cast<uint64_t>(active & static_cast<uint32_t>(nan_bits)) << base;
    if (write == 0xFF) {
      _mm256_storeu_ps(reinterpret_cast<float*>(d.dst + base), result);
    } else if (write) {
      const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
      const __m256i mask = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(static_cast<int>(write)), bit), bit);
      _mm256_maskstore_ps(reinterpret_cast<float*>(d.dst + base), mask, result);
    }
  }
  return redo;
}
#endif

// Every lane of EXEC gets its result but those whose sum is a NaN, which are returned (a bit each) for the old code.
inline uint64_t dot2_f16_wave(const DotArgs& d) {
#if defined(__x86_64__)
  if (host_has_avx2_f16c() && (_mm_getcsr() & ((1u << 15) | (1u << 6) | 0x6000u)) == 0) return dot2_f16_wave_avx2(d);
#endif
  return dot2_f16_lanes(d);
}

}  // namespace vgpu::amd::packed
