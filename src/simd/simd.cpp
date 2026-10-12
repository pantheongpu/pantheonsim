// Dispatch for the SIMD kernels (include/vgpu/simd/simd.hpp). The kernels are
// in kernels.inc, compiled once for the baseline and once under a target
// pragma.
#include "vgpu/simd/simd.hpp"

#include <bit>
#include <cstdlib>
#include <cstring>

#include "vgpu/simd/fp16.hpp"

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#include <immintrin.h>
#define VGPU_SIMD_X86 1
#endif

// A product the card rounds before the sum must not become an FMA in a build
// that has one (the AVX2 level has FMA): no contraction anywhere in this file.
#pragma GCC optimize("fp-contract=off")

namespace vgpu::simd {

#define VGPU_SIMD_NS base_impl
#include "kernels.inc"
#undef VGPU_SIMD_NS

#ifdef VGPU_SIMD_X86
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#define VGPU_SIMD_NS avx2_impl
#include "kernels.inc"
#undef VGPU_SIMD_NS
#pragma GCC pop_options
#endif

namespace {

const Kernels kBase = {Level::Base, base_impl::f16_to_f32_k, base_impl::f32_to_f16_k, base_impl::hmma_k};
#ifdef VGPU_SIMD_X86
const Kernels kAvx2 = {Level::Avx2, avx2_impl::f16_to_f32_k, avx2_impl::f32_to_f16_k, avx2_impl::hmma_k};

// The F16C conversions against the scalar ones: every half, and a spread of
// floats. A host whose conversion differs (a stray MXCSR, an emulator) is
// treated as not having the level -- the same check the PTX engine makes of
// its F16C path.
bool avx2_conversions_agree() {
  for (uint32_t h = 0; h < 65536; h += 8) {
    uint16_t in[8];
    float got[8];
    for (uint32_t i = 0; i < 8; ++i) in[i] = static_cast<uint16_t>(h + i);
    kAvx2.f16_to_f32(in, got, 8);
    for (uint32_t i = 0; i < 8; ++i)
      if (std::bit_cast<uint32_t>(got[i]) != f16_to_f32_bits(in[i])) return false;
  }
  uint32_t x = 0x9E3779B9u;
  for (int blk = 0; blk < 4096; ++blk) {
    float in[8];
    uint16_t got[8];
    for (int i = 0; i < 8; ++i) {
      x = x * 1664525u + 1013904223u;
      in[i] = std::bit_cast<float>(blk < 2048 ? x : (x & 0x80000000u) | (0x30000000u + (x & 0x0FFFFFFFu)));
    }
    kAvx2.f32_to_f16(in, got, 8);
    for (int i = 0; i < 8; ++i)
      if (in[i] == in[i] && got[i] != f32_to_f16_bits(in[i])) return false;
  }
  return true;
}
#endif

bool cpu_has_avx2() {
#ifdef VGPU_SIMD_X86
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
  return false;
#endif
}

bool avx2_usable() {
#ifdef VGPU_SIMD_X86
  static const bool ok = cpu_has_avx2() && avx2_conversions_agree();
  return ok;
#else
  return false;
#endif
}

Level choose() {
  Level best = avx2_usable() ? Level::Avx2 : Level::Base;
  if (const char* v = std::getenv("VGPU_SIMD")) {
    if (std::strcmp(v, "base") == 0) best = Level::Base;
    else if (std::strcmp(v, "avx2") == 0 && best != Level::Avx2) best = Level::Base;   // asked for more than the host has
  }
  return best;
}

}  // namespace

bool enabled() {
  static const bool on = [] {
    const char* v = std::getenv("VGPU_SIMD");
    return !(v && std::strcmp(v, "off") == 0);
  }();
  return on;
}

const char* level_name(Level l) { return l == Level::Avx2 ? "avx2" : "base"; }

bool supported(Level l) { return l == Level::Base || (l == Level::Avx2 && avx2_usable()); }

Level level() {
  static const Level l = choose();
  return l;
}

const Kernels& kernels_for(Level l) {
#ifdef VGPU_SIMD_X86
  if (l == Level::Avx2 && avx2_usable()) return kAvx2;
#endif
  return kBase;
}

const Kernels& kernels() {
  static const Kernels* k = &kernels_for(level());
  return *k;
}

}  // namespace vgpu::simd
