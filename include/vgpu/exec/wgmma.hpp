// Hopper's warpgroup MMA operands in shared memory: the matrix descriptor
// and where an element sits, shared by the PTX interpreter (wgmma, tcgen05)
// and the SASS executor (HGMMA and its kin).
#pragma once

#include <bit>
#include <cstdint>

#include "vgpu/exec/tensormap.hpp"

namespace vgpu::exec {

// A shared-memory matrix descriptor (PTX ISA 9.7.17.5.1.2.2).
struct WgmmaDesc {
  uint64_t start = 0, lbo = 0, sbo = 0;
  uint32_t swizzle = 0;   // bytes in a swizzled row: 0 (none), 32, 64 or 128
  uint32_t atom = 16;     // bytes the swizzle moves as one (tcgen05's mode 1: 32)
  // tcgen05, sm_103a: `lbo` is the absolute address where a K-major row
  // continues past the end of its swizzle row (bit 52).
  bool lbo_abs = false;
};

// Decodes a wgmma descriptor. Throws vgpu::Error for a nonzero base offset
// (bits 49-51): only swizzle patterns that start on their repeat boundary
// are implemented.
WgmmaDesc decode_wgmma_desc(uint64_t d);

// Shared-window offset of element (mn, k) of a matrix a descriptor
// describes: A is M x K and B is N x K, so `mn` is the row of A or the
// column of B; `eb` the element's bytes. The strides are the canonical
// layouts of 9.7.17.5.1.2.1.3: a core matrix is 8 rows of 16 bytes, LBO and
// SBO step between core matrices, and a swizzled layout XORs address bits
// 4-6 with bits 7-9 (Swizzle<3,4,3> for 128B; 64B and 32B keep fewer), the
// same function of the address a TMA copy applies when it writes the tile.
// (Inline: the tile code calls it once per 16 bytes of an operand. A swizzled row is a power of two bytes, so its divisions are shifts.)
inline uint64_t wgmma_smem_offset(const WgmmaDesc& d, bool k_major, uint32_t eb, uint32_t mn, uint32_t k) {
  const uint64_t W = d.swizzle;
  uint64_t off;
  if (k_major) {
    const uint64_t kb = uint64_t{k} * eb;
    // Absolute mode (sm_103a): a row's bytes run to the end of its swizzle
    // row, then on at the same row of the atom at `lbo` -- how CUTLASS's
    // SM103 kernels have a K = 96 fp4 block straddle two pipeline buffers.
    if (d.lbo_abs && W && (d.start & (W - 1)) + kb >= W)
      return swizzle_address(d.lbo + (mn % 8) * W + (mn / 8) * d.sbo + ((d.start & (W - 1)) + kb - W),
                             static_cast<uint32_t>(W), d.atom);
    off = W ? (mn % 8) * W + (mn / 8) * d.sbo + kb
            : (mn % 8) * 16 + (mn / 8) * d.sbo + kb % 16 + (kb / 16) * d.lbo;
  } else {
    const uint64_t mb = uint64_t{mn} * eb;
    off = W ? (mb & (W - 1)) + (mb >> std::countr_zero(W)) * d.lbo + (k % 8) * W + (k / 8) * d.sbo
            : mb % 16 + (mb / 16) * d.sbo + (k % 8) * 16 + (k / 8) * d.lbo;
  }
  return swizzle_address(d.start + off, static_cast<uint32_t>(W), d.atom);
}


}  // namespace vgpu::exec
