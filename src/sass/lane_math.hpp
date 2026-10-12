// Lane loops of the SASS executor's fast paths that need a CPU feature the
// baseline x86-64 build does not assume.
//
// FFMA over a warp's 32 lanes. std::fma is a call into libm per lane in this
// build (no -mfma), which glibc then routes through an ifunc to the very
// vfmadd instruction; asking the CPU once and running the loop on the FMA unit
// gives the same bits -- both are the correctly rounded fused multiply-add --
// without the call. The NaN rule is the executor's: any NaN result is the
// card's canonical 0x7fffffff, whatever its payload was on the host.
//
// Contraction: nothing else in these functions multiplies and adds, so
// compiling them with the FMA target cannot fuse anything that was not fused
// on purpose.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace vgpu::sass {

namespace lane_math_detail {
inline uint32_t canon(float v) {
  if (v != v) return 0x7fffffffu;
  uint32_t b;
  std::memcpy(&b, &v, 4);
  return b;
}
inline float f32_of(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
}  // namespace lane_math_detail

// r[l] = fma(a[l], b[l], c[l]) for the 32 lanes, with NaNs made canonical:
// the libm version, which is the reference.
inline void ffma_lanes_libm(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
  using namespace lane_math_detail;
  for (unsigned l = 0; l < 32; ++l) r[l] = canon(std::fma(f32_of(a[l]), f32_of(b[l]), f32_of(c[l])));
}

#if defined(__x86_64__) && defined(__GNUC__)
// The same on the CPU's FMA unit. Only call it when ffma_lanes_hw_available().
__attribute__((target("fma"))) inline void ffma_lanes_hw(uint32_t* r, const uint32_t* a, const uint32_t* b,
                                                         const uint32_t* c) {
  using namespace lane_math_detail;
  for (unsigned l = 0; l < 32; ++l) r[l] = canon(__builtin_fmaf(f32_of(a[l]), f32_of(b[l]), f32_of(c[l])));
}
inline bool ffma_lanes_hw_available() {
  static const bool ok = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("fma") != 0;
  }();
  return ok;
}
#else
inline bool ffma_lanes_hw_available() { return false; }
inline void ffma_lanes_hw(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
  ffma_lanes_libm(r, a, b, c);
}
#endif

inline void ffma_lanes(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
  if (ffma_lanes_hw_available()) ffma_lanes_hw(r, a, b, c);
  else ffma_lanes_libm(r, a, b, c);
}

}  // namespace vgpu::sass
