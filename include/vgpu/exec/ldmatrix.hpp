// ldmatrix's 8-bit forms (m8n16, m16n16), the same for both engines: the PTX
// interpreter's ldmatrix and the SASS executor's LDSM (.U4x16P64TO8,
// .U6x16P32TO8, .8 with .M816 and .MT1616) read a matrix's rows, unpack each
// to sixteen bytes, and hand every lane its words here, so the fp4 and fp6
// operands an mma .kind::f8f6f4 takes arrive the same whichever code runs.
#pragma once

#include <cstdint>

namespace vgpu::exec {

// What a row holds: sixteen values of `bits` each from its first 16 * bits / 8
// bytes (8: .b8; 6: .b6x16_p32's packed front; 4: .b4x16_p64's, or .s4).
struct LdmRow {
  uint32_t bits = 8;
  bool sign4 = false;   // .s4: a 4-bit value sign-extended to its byte
  uint32_t bytes() const { return 16 * bits / 8; }
};

// The row's sixteen values, each in the low bits of its byte -- CUTLASS shifts
// e2m1 up by 2 itself before an mma.
inline void ldm_unpack_row(const uint8_t* raw, LdmRow f, uint8_t out[16]) {
  for (uint32_t c = 0; c < 16; ++c) {
    uint32_t v = 0;
    for (uint32_t k = 0; k < f.bits; ++k) v |= ((raw[(c * f.bits + k) / 8] >> ((c * f.bits + k) % 8)) & 1u) << k;
    if (f.sign4 && (v & 8)) v |= 0xF0;
    out[c] = static_cast<uint8_t>(v);
  }
}

// Lane `lane`'s register `rr` (0, or 1 for a 16-row matrix's second) of a
// matrix whose unpacked rows are `tile`: four consecutive columns of row
// lane / 4 (+ 8 for the second register), which for the transposed 16x16 are
// four stored rows at element lane / 4 (PTX ISA figures 108-109).
inline uint32_t ldm_word(const uint8_t (&tile)[16][16], uint32_t lane, uint32_t rr, bool trans) {
  const uint32_t row = lane / 4 + 8 * rr, col0 = 4 * (lane % 4);
  uint32_t word = 0;
  for (uint32_t j = 0; j < 4; ++j) word |= uint32_t{trans ? tile[col0 + j][row] : tile[row][col0 + j]} << (8 * j);
  return word;
}

}  // namespace vgpu::exec
