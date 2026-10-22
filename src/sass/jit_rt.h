// The part of the JIT's generated code that does not change from one region to the next. It is embedded in the simulator (CMake
// writes it, and src/simd/kernels.inc, into jit_embed.cpp), written into the cache directory at first use, verified by its hash, and
// its hash is part of every key: the generated code needs nothing from the source tree at run time.
//
// The lane kernels are src/simd/kernels.inc itself, the text the simulator's own kernels (src/simd/simd.cpp) are compiled from,
// included here with VGPU_LANES_ONLY so that it leaves out what the SASS ALU forms do not use. One source for the arithmetic: a form's
// generated code calls the kernel the tier's handler calls (the generator's include of this file defines VGPU_SIMD_AVX2 first: 1 for the
// kernels of the AVX2 level, 0 for the baseline, as the simulator is running).
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__x86_64__)
#include <emmintrin.h>
#include <immintrin.h>
#endif

#define VGPU_LANES_ONLY 1
// The generated code inlines every kernel it calls: the arithmetic of a region is then one function the compiler can schedule as a whole.
#define VGPU_KN [[gnu::always_inline]] inline
#define VGPU_KA [[gnu::always_inline]]

namespace {

namespace jk {
#define VGPU_SIMD_NS jk_kernels
#include "vgpu_jit_kernels.inc"
#undef VGPU_SIMD_NS
}  // namespace jk

using u32 = uint32_t;
using u64 = uint64_t;
using namespace jk::jk_kernels;

// Modifiers on a source that has to be worked out when the instruction runs (tier_mod): for floats |x| then -x on the bits; for
// integers -x as 0 - x, then ~x.
[[gnu::always_inline]] inline u32 tmod(bool flt, unsigned mod, u32 v) {
  if (flt) {
    if (mod & 4u) v &= 0x7fffffffu;
    if (mod & 1u) v ^= 0x80000000u;
  } else {
    if (mod & 1u) v = 0u - v;
    if (mod & 2u) v = ~v;
  }
  return v;
}

[[gnu::always_inline]] inline void splat(u32* t, u32 v) {
  for (unsigned l = 0; l < 32; ++l) t[l] = v;
}

// A register with a modifier: its 32 lanes worked through tmod.
[[gnu::always_inline]] inline const u32* dynreg(u32* t, const u32* r, bool flt, unsigned mod) {
  for (unsigned l = 0; l < 32; ++l) t[l] = tmod(flt, mod, r[l]);
  return t;
}

// A result row to the destination register for the executing lanes (tier_store): the whole row, or a blend of the lanes.
[[gnu::always_inline]] inline void store_row(u32* d, u32 ex, const u32* r) {
  if (ex == 0xffffffffu) {
    for (unsigned l = 0; l < 32; ++l) d[l] = r[l];
  } else {
    blend_m_k(d, r, ex);
  }
}

// The register file's row and a local copy of it (register residence).
[[gnu::always_inline]] inline void ldrow(u32* x, const u32* r) {
  for (unsigned l = 0; l < 32; ++l) x[l] = r[l];
}
[[gnu::always_inline]] inline void strow(u32* r, const u32* x) {
  for (unsigned l = 0; l < 32; ++l) r[l] = x[l];
}

inline u32 combine(unsigned bop, u32 r, u32 acc) { return bop == 0 ? (r & acc) : bop == 1 ? (r | acc) : (r ^ acc); }

alignas(32) const u32 kZeros32[32] = {};
alignas(32) const u32 kOnes32[32] = {~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u,
                                      ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};

}  // namespace
