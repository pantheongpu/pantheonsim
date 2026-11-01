// The AMD matrix tile kernels and wide decoders live in a translation unit of their own (amd/src/gcn_matrix_kernels.cpp), built without
// link-time optimization and with their functions on 64-byte boundaries, so that their code layout does not depend on what is linked ahead
// of them (their few-instruction loops were 4 to 11 % slower in some positions of a 64-byte line on a Zen 5 host; docs/performance.md,
// Round sp13). The arithmetic is held against the reference by test_amd_mfma_math and test_amd_matrix_fast; this holds the property the
// speed rests on: each of them is a function of its own (not inlined into a caller, whose position would then decide) and starts a line.
#include <cstdint>

#include "../../src/gcn_matrix_kernels.hpp"
#include "../../src/mfma_math.hpp"
#include "vtest.hpp"

namespace {

#if defined(__x86_64__) && defined(__GNUC__)
template <class F>
uintptr_t where(F f) {
  return reinterpret_cast<uintptr_t>(f);
}

VTEST(the_tile_kernels_start_on_a_cache_line) {
  VCHECK_EQ(where(&vgpu::amd::mfma_tile_f_avx2) % 64, 0u);
  VCHECK_EQ(where(&vgpu::amd::mfma_tile_fma_hw) % 64, 0u);
  VCHECK_EQ(where(&vgpu::amd::mfma_tile_fma_avx2) % 64, 0u);
  VCHECK_EQ(where(&vgpu::amd::mfma_tile_int_avx2) % 64, 0u);
}

VTEST(the_wide_decoders_start_on_a_cache_line) {
  using namespace vgpu::amd;
  VCHECK_EQ(where(&matrix_decode_wide_avx2<'h', false>) % 64, 0u);
  VCHECK_EQ(where(&matrix_decode_wide_avx2<'h', true>) % 64, 0u);
  VCHECK_EQ(where(&matrix_decode_wide_avx2<'b', false>) % 64, 0u);
  VCHECK_EQ(where(&matrix_decode_wide_avx2<'b', true>) % 64, 0u);
  VCHECK_EQ(where(&matrix_decode_wide_avx2<'f', false>) % 64, 0u);
  VCHECK_EQ(where(&matrix_decode_wide_avx2<'f', true>) % 64, 0u);
  VCHECK_EQ(where(&matrix_decode_wide_avx2<'F', false>) % 64, 0u);
  VCHECK_EQ(where(&matrix_decode_wide_avx2<'F', true>) % 64, 0u);
}
#else
VTEST(nothing_to_check_off_x86_64) {}
#endif

}  // namespace

VTEST_MAIN
