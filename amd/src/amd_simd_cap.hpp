// The AMD engine's host SIMD code (AVX2 / FMA / F16C paths of mfma_math.hpp and packed_math.hpp) obeys the same cap as the other engines':
// VGPU_SIMD=base or off keeps it to the baseline x86-64 code, so the digest tests can be run on the baseline path on a machine that has
// more (the CI's machines are not this one's).
#pragma once

#include <cstdlib>
#include <cstring>

namespace vgpu::amd {

inline bool simd_capped_to_base() {
  static const bool capped = [] {
    const char* v = std::getenv("VGPU_SIMD");
    return v && (std::strcmp(v, "base") == 0 || std::strcmp(v, "off") == 0);
  }();
  return capped;
}

}  // namespace vgpu::amd
