// The AVX2 operand decoders of the AMD matrix path, defined in gcn_matrix_kernels.cpp (see there for why they are a unit of their own).
#pragma once

#include <cstdint>

namespace vgpu::amd {

#if defined(__x86_64__) && defined(__GNUC__)
// Element e of lanes l to l + 7 of the register rows `rows` (MatrixSrc::rows) as doubles, to dst[dest[lane] + e * step]: T is 'h' (binary16), 'b'
// (bfloat16), 'f' (float) or 'F' (xf32); Run: four lanes' runs start at consecutive doubles (B's columns). Instantiated for the eight
// combinations in gcn_matrix_kernels.cpp. The caller has checked for AVX2 and F16C.
template <char T, bool Run>
void matrix_decode_wide_avx2(const uint32_t* const* rows, const uint32_t* dest, uint32_t step, uint32_t per_lane, double* dst, uint32_t lanes);
#endif

}  // namespace vgpu::amd
