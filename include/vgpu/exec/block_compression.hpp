// Block-compressed texture formats (BC1 to BC5; BC6H and BC7 are not implemented): a block of 4 x 4 texels in
// 8 or 16 bytes, decoded to what the texture unit hands the shader.
//
// The block layouts are from the public format descriptions (the Khronos Data Format Specification's
// block-compressed chapter, and the formats' D3D documentation). The arithmetic is NOT the description's
// integer formulas: it is what an RTX 3060's texture unit returns, measured by point sampling blocks with
// every endpoint pair (see the comments at each decoder), and the decoders reproduce those measurements
// value for value.
#pragma once

#include <algorithm>
#include <cstdint>

namespace vgpu::exec {

// The formats a texture can be stored in. BC1, BC2 and BC3 decode to four unsigned normalized channels
// (8-bit; the sRGB variants are the same blocks, decoded from sRGB at the fetch), BC4 to one and BC5 to two
// 16-bit channels, unsigned or signed normalized. BC6H and BC7 are
// named here so that a texture of them can be created, but they are not decoded (decode_block says false).
enum class BlockFormat : uint8_t { None, BC1, BC2, BC3, BC4U, BC4S, BC5U, BC5S, BC6HU, BC6HS, BC7 };

inline uint32_t block_bytes(BlockFormat f) {
  return f == BlockFormat::BC1 || f == BlockFormat::BC4U || f == BlockFormat::BC4S ? 8u : 16u;
}

// A decoded 4 x 4 block: for texel i (row-major: x = i % 4, y = i / 4), channel c. The raw channel bits as the
// texture path reads them: 8-bit codes for BC1, BC2 and BC3, 16-bit unsigned codes (BC4U, BC5U) or 16-bit two's
// complement (BC4S, BC5S).
struct DecodedBlock {
  uint16_t v[16][4];
};

namespace bc_detail {

// ---- BC1, BC2, BC3: the colour half ----
//
// Measured on an RTX 3060 (driver 13.2) with every endpoint pair of every channel, in both the four-colour
// and the three-colour mode, point sampled (the texture unit's values; the Khronos description's
// integer-interpolation formulas are NOT what the card returns, so these are the card's own):
//  - red and blue, four colours: the 5-bit endpoints widen to 16 bits (round(q * 65535 / 31)), the two blended
//    colours are (2 * x0 + x1) / 3 and (x0 + 2 * x1) / 3 (integer division) and the result is the high byte;
//  - green, four colours: the 6-bit endpoints widen to 8 bits by bit replication and are blended with
//    weights of 321/1024 and 703/1024 on the second endpoint, rounded to nearest;
//  - three colours (c0 <= c1): the middle colour of red and blue is (33 * (q0 + q1)) >> 3, of green the
//    weight 513/1024 on the second endpoint (514 fits equally), and the fourth colour is transparent black.
// BC2 and BC3 colour blocks are always four-colour.
inline int widen5_16(int q) { return (q * 65535 + 15) / 31; }
inline int widen6_8(int q) { return (q << 2) | (q >> 4); }

inline void colour_block(const uint8_t* b, bool four_color_only, DecodedBlock* out) {
  const uint16_t c0 = static_cast<uint16_t>(b[0] | (b[1] << 8)), c1 = static_cast<uint16_t>(b[2] | (b[3] << 8));
  const int r0 = (c0 >> 11) & 31, g0 = (c0 >> 5) & 63, b0 = c0 & 31;
  const int r1 = (c1 >> 11) & 31, g1 = (c1 >> 5) & 63, b1 = c1 & 31;
  const bool four = c0 > c1 || four_color_only;
  int p[4][4];
  const int rx0 = widen5_16(r0), rx1 = widen5_16(r1), bx0 = widen5_16(b0), bx1 = widen5_16(b1);
  const int ge0 = widen6_8(g0), ge1 = widen6_8(g1);
  p[0][0] = rx0 >> 8; p[0][1] = ge0; p[0][2] = bx0 >> 8;
  p[1][0] = rx1 >> 8; p[1][1] = ge1; p[1][2] = bx1 >> 8;
  if (four) {
    p[2][0] = ((2 * rx0 + rx1) / 3) >> 8;
    p[3][0] = ((rx0 + 2 * rx1) / 3) >> 8;
    p[2][2] = ((2 * bx0 + bx1) / 3) >> 8;
    p[3][2] = ((bx0 + 2 * bx1) / 3) >> 8;
    p[2][1] = (ge0 * (1024 - 321) + ge1 * 321 + 512) >> 10;
    p[3][1] = (ge0 * (1024 - 703) + ge1 * 703 + 512) >> 10;
  } else {
    p[2][0] = (33 * (r0 + r1)) >> 3;
    p[2][2] = (33 * (b0 + b1)) >> 3;
    p[2][1] = (ge0 * (1024 - 513) + ge1 * 513 + 512) >> 10;
    p[3][0] = p[3][1] = p[3][2] = 0;
  }
  for (int k = 0; k < 4; ++k) p[k][3] = 255;
  if (!four) p[3][3] = 0;   // transparent black
  uint32_t idx = static_cast<uint32_t>(b[4]) | (static_cast<uint32_t>(b[5]) << 8) |
                 (static_cast<uint32_t>(b[6]) << 16) | (static_cast<uint32_t>(b[7]) << 24);
  for (int i = 0; i < 16; ++i, idx >>= 2)
    for (int k = 0; k < 4; ++k) out->v[i][k] = static_cast<uint16_t>(p[idx & 3][k]);
}

// ---- BC4 and BC5, and the alpha of BC3: one channel in 8 bytes ----
//
// Two 8-bit endpoints and 3-bit indices into a palette of eight values (a0 > a1) or of six values plus the
// two ends (a0 <= a1). The card delivers 16-bit values (the texture unit's unsigned normalized channel), and
// it blends the endpoints in 16 bits with weights that are not n/7 and n/5: measured for all 65536 endpoint
// pairs and all 8 indices, the palette is a0 * (257 - W) + a1 * W, W being (in 1/257ths)
//   eight values: 36, 72, 113, 144, 185, 221 for indices 2..7;  six values: 48, 96, 161, 209 for indices 2..5,
// index 0 and 1 being the endpoints (a * 257), and in the six-value mode index 6 is 0 and index 7 is 65535.
// No rounding enters (the arithmetic is exact).
inline void single_channel_u(const uint8_t* b, uint16_t out[16]) {
  static constexpr int kW8[8] = {0, 0, 36, 72, 113, 144, 185, 221};
  static constexpr int kW6[8] = {0, 0, 48, 96, 161, 209, 0, 0};
  const int a0 = b[0], a1 = b[1];
  int pal[8];
  pal[0] = a0 * 257;
  pal[1] = a1 * 257;
  const bool eight = a0 > a1;
  for (int k = 2; k < 8; ++k) {
    if (eight) {
      pal[k] = a0 * (257 - kW8[k]) + a1 * kW8[k];
    } else if (k < 6) {
      pal[k] = a0 * (257 - kW6[k]) + a1 * kW6[k];
    } else {
      pal[k] = k == 6 ? 0 : 65535;
    }
  }
  uint64_t bits = 0;
  for (int i = 0; i < 6; ++i) bits |= static_cast<uint64_t>(b[2 + i]) << (8 * i);
  for (int i = 0; i < 16; ++i, bits >>= 3) out[i] = static_cast<uint16_t>(pal[bits & 7]);
}

// BC3's alpha block is the same layout but the card delivers it at 8 bits, computed on its own (it is not the
// 16-bit palette narrowed: measured with all 65536 endpoint pairs and all indices): the palette is
// floor((a0 * 2048 + (a1 - a0) * N + 1024) / 2048), with N = 289, 578, 892, 1156, 1470, 1759 for indices 2..7
// of the eight-value form and 385, 770, 1278, 1663 for indices 2..5 of the six-value one (index 6 is 0 and
// index 7 is 255 there).
inline void alpha_bc3(const uint8_t* b, uint16_t out[16]) {
  static constexpr int kN8[8] = {0, 0, 289, 578, 892, 1156, 1470, 1759};
  static constexpr int kN6[8] = {0, 0, 385, 770, 1278, 1663, 0, 0};
  const int a0 = b[0], a1 = b[1];
  int pal[8];
  pal[0] = a0;
  pal[1] = a1;
  const bool eight = a0 > a1;
  for (int k = 2; k < 8; ++k) {
    const int n = eight ? kN8[k] : kN6[k];
    if (eight || k < 6) {
      const int num = a0 * 2048 + (a1 - a0) * n + 1024;
      pal[k] = num >= 0 ? num >> 11 : -((-num + 2047) >> 11);   // floor
    } else {
      pal[k] = k == 6 ? 0 : 255;
    }
  }
  uint64_t bits = 0;
  for (int i = 0; i < 6; ++i) bits |= static_cast<uint64_t>(b[2 + i]) << (8 * i);
  for (int i = 0; i < 16; ++i, bits >>= 3) out[i] = static_cast<uint16_t>(pal[bits & 7]);
}

// The signed form (BC4S, BC5S): endpoints are int8 (-128 reads as -127; the palette mode compares the
// clamped values), delivered as signed normalized 16-bit (a * 32767 / 127, rounded to nearest) and blended
// as the unsigned one is, with the same step sizes in the 32767/127 scale: palette = round((a0 * 32767 +
// (a1 - a0) * M) / 127), M being W * 127 for the small steps and 32767 - W * 127 for the large ones --
// 4572, 9144, 14479, 18288, 23623, 28195 (eight values) and 6096, 12192, 20575, 26671 (six values) --
// and in the six-value mode index 6 is -32767 and index 7 is +32767. Measured for all 65536 pairs and indices.
inline void single_channel_s(const uint8_t* b, uint16_t out[16]) {
  static constexpr int kM8[8] = {0, 0, 4572, 9144, 14479, 18288, 23623, 28195};
  static constexpr int kM6[8] = {0, 0, 6096, 12192, 20575, 26671, 0, 0};
  const int a0 = std::max(static_cast<int>(static_cast<int8_t>(b[0])), -127);
  const int a1 = std::max(static_cast<int>(static_cast<int8_t>(b[1])), -127);
  // floor((num + 63.5) / 127), in integers that stay exact for negative numerators too
  auto rounded = [](int num) {
    const int n2 = 2 * num + 127;
    return n2 >= 0 ? n2 / 254 : -((-n2 + 253) / 254);
  };
  int pal[8];
  pal[0] = rounded(a0 * 32767);
  pal[1] = rounded(a1 * 32767);
  const bool eight = a0 > a1;
  for (int k = 2; k < 8; ++k) {
    if (eight) {
      pal[k] = rounded(a0 * 32767 + (a1 - a0) * kM8[k]);
    } else if (k < 6) {
      pal[k] = rounded(a0 * 32767 + (a1 - a0) * kM6[k]);
    } else {
      pal[k] = k == 6 ? -32767 : 32767;
    }
  }
  uint64_t bits = 0;
  for (int i = 0; i < 6; ++i) bits |= static_cast<uint64_t>(b[2 + i]) << (8 * i);
  for (int i = 0; i < 16; ++i, bits >>= 3) out[i] = static_cast<uint16_t>(static_cast<int16_t>(pal[bits & 7]));
}

}  // namespace bc_detail

// Decodes one block of BC1 to BC5. (BC6H and BC7 are not decoded: see decode_block.)
inline void decode_block_ldr(BlockFormat f, const uint8_t* b, DecodedBlock* out) {
  using namespace bc_detail;
  switch (f) {
    case BlockFormat::BC1:
      colour_block(b, false, out);
      break;
    case BlockFormat::BC2: {
      colour_block(b + 8, true, out);
      for (int i = 0; i < 16; ++i) {
        const int a4 = (b[i / 2] >> (4 * (i & 1))) & 15;
        out->v[i][3] = static_cast<uint16_t>(a4 * 17);
      }
      break;
    }
    case BlockFormat::BC3: {
      colour_block(b + 8, true, out);
      uint16_t a[16];
      alpha_bc3(b, a);
      for (int i = 0; i < 16; ++i) out->v[i][3] = a[i];
      break;
    }
    case BlockFormat::BC4U:
    case BlockFormat::BC4S: {
      uint16_t r[16];
      if (f == BlockFormat::BC4U) single_channel_u(b, r); else single_channel_s(b, r);
      for (int i = 0; i < 16; ++i) out->v[i][0] = r[i], out->v[i][1] = out->v[i][2] = out->v[i][3] = 0;
      break;
    }
    case BlockFormat::BC5U:
    case BlockFormat::BC5S: {
      uint16_t r[16], g[16];
      if (f == BlockFormat::BC5U) { single_channel_u(b, r); single_channel_u(b + 8, g); }
      else { single_channel_s(b, r); single_channel_s(b + 8, g); }
      for (int i = 0; i < 16; ++i) out->v[i][0] = r[i], out->v[i][1] = g[i], out->v[i][2] = out->v[i][3] = 0;
      break;
    }
    default:
      break;
  }
}

// Decodes one block of any format this file knows. False for the formats that are not implemented (BC6H and
// BC7: their partition and mode tables cannot be taken from the public format description by measurement
// alone with the effort spent so far); the caller reports them by name.
inline bool decode_block(BlockFormat f, const uint8_t* b, DecodedBlock* out) {
  switch (f) {
    case BlockFormat::BC1: case BlockFormat::BC2: case BlockFormat::BC3:
    case BlockFormat::BC4U: case BlockFormat::BC4S: case BlockFormat::BC5U: case BlockFormat::BC5S:
      decode_block_ldr(f, b, out);
      return true;
    default:
      return false;
  }
}

}  // namespace vgpu::exec
