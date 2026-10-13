// The numeric lane kernels of both engines, with one implementation per host
// instruction-set level and a runtime choice between them.
//
// The build is baseline x86-64 (no -m flags anywhere), so a kernel that wants
// AVX2, FMA or F16C is compiled twice: once for the baseline and once under a
// target pragma, and the choice is made once, when the first kernel is asked
// for, from __builtin_cpu_supports. (Not target_clones/IFUNC: the resolver runs
// before main, and under ThreadSanitizer it runs before the sanitizer's runtime
// is up -- see src/exec/interpreter.cpp, wmma_tile.)
//
// The one rule for a kernel: every level gives the same bits as the scalar
// code it replaces, for every input. Floating-point products are rounded and
// then summed (never fused) where the card does that, and fused only where the
// operation is a fused multiply-add. tests/unit/test_simd.cpp runs each kernel
// at every level the CPU has against a plain scalar reference.
//
// VGPU_SIMD=off turns the kernels off in the engines (they keep their scalar
// code); VGPU_SIMD=base|avx2 caps the level (default: the best the CPU has); the
// baseline kernels are always compiled in, so base is the reference the other
// levels are checked against and the escape hatch if a host misbehaves.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace vgpu::simd {

enum class Level : int { Base = 0, Avx2 = 1 };   // Avx2 = AVX2 + FMA + F16C
inline constexpr int kLevels = 2;


// The F16C check (the conversions against the scalar ones, every half and a spread of floats; a host that fails it
// gets the baseline for the four kernels that use F16C and AVX2 for the rest) is made on the first use of one of those
// kernels, once, whatever the number of threads asking -- not when the library loads, where it cost 0.4 ms a process.
// kernels_for(Level::Avx2) makes it too.
enum class F16cState { Unchecked, Agrees, Disagrees };
F16cState f16c_state();
namespace testing {
// Back to Unchecked, for a test of the first use; not for use while another thread runs a kernel.
void reset_f16c_check();
// How many times the check has run since the process started or reset_f16c_check.
int f16c_checks_run();
}  // namespace testing

const char* level_name(Level l);
// Whether this host can run `l` (Base always).
bool supported(Level l);
// The level the kernels in use run at: the best supported, capped by VGPU_SIMD.
Level level();

enum class IntCmp : unsigned { Eq, Ne, Lt, Le, Gt, Ge };
enum class IntBin : unsigned { Add, Sub, Mul, And, Or, Xor, Min, Max, Shl, Shr };

struct Kernels {
  Level level;
  // binary16 -> binary32 (exact) and binary32 -> binary16 (to nearest even),
  // n elements, bit for bit what vgpu::simd::f16_to_f32_bits / f32_to_f16_bits
  // (simd/fp16.hpp) give.
  void (*f16_to_f32)(const uint16_t* in, float* out, size_t n);
  void (*f32_to_f16)(const float* in, uint16_t* out, size_t n);

  // The 32 lanes of one warp, r[l] from a[l], b[l], c[l]. r may not alias the sources (they may alias each other).
  // iadd3: a + b + c + add (wrapping). imad: a * b + c (low 32 bits). lop3: the three-input logic function whose
  // truth table is `lut` (bit 4a + 2b + c). ffma/fadd/fmul: the f32 operation on the bits, a fused multiply-add
  // for ffma (one rounding), the sum or product rounded to nearest even, a NaN result the canonical 0x7fffffff.
  void (*iadd3)(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t add);
  void (*imad)(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c);
  void (*lop3)(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t lut);
  void (*ffma)(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c);
  void (*fadd)(uint32_t* r, const uint32_t* a, const uint32_t* b);
  void (*fmul)(uint32_t* r, const uint32_t* a, const uint32_t* b);

  // Masked forms over n lanes (a multiple of 8, at most 64), writing only the lanes whose bit of `mask` is set; the
  // others of d keep their value, and d may be one of the sources. ffma_m: as ffma. hfma2_m: each 32-bit lane is
  // two binary16 halves (`halves` = 2) or one in the low half (1; the high half of the result is then zero); each
  // half is the double-precision fused multiply-add of its three inputs rounded once to binary16 (a NaN the canonical
  // 0x7fff) -- the PTX interpreter's fma.rn.f16x2, whose rounding is that of the double, not of an exact fused op.
  void (*ffma_m)(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n);
  void (*hfma2_m)(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n,
                  unsigned halves);
  // 32-bit integer lanes over n lanes (a multiple of 32, at most 64). icmp_mask: bit l of the result is `a[l] cmp b[l]`
  // for cmp in {Eq, Ne, Lt, Le, Gt, Ge} (IntCmp), signed or unsigned. ibin_m: d[l] = a[l] op b[l] for the lanes of
  // `mask` (the others keep their value; d may be a source): wrapping add/sub/mul, and/or/xor, min/max, and
  // shl/shr where a count of 32 or more shifts every bit out (a signed shr then fills with the sign), as PTX defines
  // them. imad_m: a * b + c, wrapping.
  uint64_t (*icmp_mask)(unsigned cmp, bool sgn, const uint32_t* a, const uint32_t* b, unsigned n);
  void (*ibin_m)(unsigned op, bool sgn, uint32_t* d, const uint32_t* a, const uint32_t* b, uint64_t mask, unsigned n);
  void (*imad_m)(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n);

  // binary64 lanes held as the SASS executor holds them, a register pair: the low words of 32 lanes and the high
  // words. kind 0: a + b, 1: a * b, 2: fma(a, b, c), all to nearest even (c is unused for 0 and 1). mods: bit 2i is
  // |source i|, bit 2i + 1 is -source i (the absolute value first). Writes only the lanes of `mask` (the others keep
  // their value; d may be a source); a NaN result is whatever the host gives.
  void (*dop_m)(unsigned kind, unsigned mods, uint32_t* dlo, uint32_t* dhi, const uint32_t* alo, const uint32_t* ahi,
                const uint32_t* blo, const uint32_t* bhi, const uint32_t* clo, const uint32_t* chi, uint32_t mask);

  // binary64 -> binary16 to nearest even in one rounding, a NaN the canonical 0x7fff.
  void (*f64_to_f16_canon)(const double* in, uint16_t* out, size_t n);

  // One warp's mma.m16n8kK of f16, bf16 or tf32 operands with a float or half
  // accumulator, on the registers as the warp holds them: register r of a
  // fragment is the 32 words at x + 32 * r. kind 0 = f16, 1 = bf16, 2 = tf32;
  // K a multiple of 8 (4 for tf32). a has K / (2 * per_reg) registers, b
  // K / (4 * per_reg) (per_reg = 2, tf32 1), c and out 4 registers (acc_f16:
  // 2); c may be null for a zero accumulator. D = C + A x B, each element
  // summed in k order from C, each product rounded to f32 and each sum
  // rounded to f32 (never fused), a NaN result the canonical 0x7fffffff (f16
  // accumulator: 0x7fff), exactly as the SASS executor's general form
  // computes it. `out` may not alias the inputs.
  void (*hmma)(unsigned kind, unsigned K, bool acc_f16, const uint32_t* a, const uint32_t* b,
               const uint32_t* c, uint32_t* out);

  // The wider integer forms of the SASS executor's plain-form ALU path, 32 lanes each (r, rlo and rhi may not alias the
  // sources).
  //  blend_m: d[l] = r[l] for the lanes whose bit of `mask` is set; the others of d keep their value.
  //  iadd3x: r[l] = a + b + c + add + bit l of cin_a + bit l of cin_b, wrapping; bit l of *ge1 is whether that sum
  //    (taken in 64 bits) reached 2^32, bit l of *ge2 whether it reached 2 * 2^32 (IADD3's two carry-out predicates).
  //  imad_wide: the 64-bit register pair (rlo, rhi) = a * b + (clo, chi) + bit l of cin, the product of the 32-bit
  //    operands signed (`sgn`) or unsigned, the sum modulo 2^64; bit l of *carry_out is the carry out of that 64-bit sum.
  //  shf: SASS's funnel shift of (top:lo) by n: type 0 S64, 1 U64, 2 S32, 3 U32; the count n is taken modulo the width
  //    (`wrap`, 64 for the 64-bit types, else 32) or limited to it; a left shift (`right` false) shifts in zeros, a right
  //    shift of a signed type the sign (the count limited to 63); r is the high word of the result if `hi`, else the low.
  void (*blend_m)(uint32_t* d, const uint32_t* r, uint32_t mask);
  void (*iadd3x)(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t add, uint32_t cin_a,
                 uint32_t cin_b, uint32_t* ge1, uint32_t* ge2);
  void (*imad_wide)(uint32_t* rlo, uint32_t* rhi, const uint32_t* a, const uint32_t* b, const uint32_t* clo,
                    const uint32_t* chi, bool sgn, uint32_t cin, uint32_t* carry_out);
  void (*shf)(uint32_t* r, const uint32_t* lo, const uint32_t* n, const uint32_t* top, unsigned type, bool wrap,
              bool right, bool hi);
  //  lane_unary: r[l] from a[l], by `kind`: 0 the number of set bits, 1 the bits in reverse order, 2 + (sgn ? 1 : 0) +
  //    (sh ? 2 : 0) SASS's FLO: the index of the highest set bit (of the complement for a negative value if `sgn`; all
  //    ones if there is none), or if `sh` that index counted from bit 31 (31 - index; all ones stays).
  void (*lane_unary)(unsigned kind, uint32_t* r, const uint32_t* a);
};

// The kernels at the level in use, and at a given one (for tests; a level the
// host cannot run is not available: check supported() first).
namespace detail {
// Set when the library is loaded (src/simd/simd.cpp): the baseline kernels and "on" until then, so a call from another
// translation unit's static initialisation still works.
//
// Hidden: libvgpu.a is linked into every shim (libcuda, libcudart, ...), each of which has its own copy of these and
// is meant to -- each is set when its library loads, from the same CPU and the same environment. A default-visibility
// definition in two libraries one process loads is an ODR violation to AddressSanitizer, which aborts the process
// when the second loads.
extern std::atomic<const Kernels*> g_kernels __attribute__((visibility("hidden")));
extern bool g_enabled __attribute__((visibility("hidden")));
}  // namespace detail

// The kernels at the level in use. Inline: it is called once or twice per simulated instruction.
inline const Kernels& kernels() { return *detail::g_kernels.load(std::memory_order_relaxed); }

// Whether the engines use these kernels at all: VGPU_SIMD=off sends every
// instruction down the code the kernels replaced, for A/B runs and to rule
// the kernels out when chasing a difference.
inline bool enabled() { return detail::g_enabled; }

// Population count of a 32-bit word without a libgcc call (a baseline x86-64 build has no POPCNT instruction, so
// std::popcount is a call into libgcc: 2% of a warp-instruction's time, at one per instruction).
inline unsigned popcount32(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0f0f0f0fu;
  return (x * 0x01010101u) >> 24;
}
const Kernels& kernels_for(Level l);

}  // namespace vgpu::simd
