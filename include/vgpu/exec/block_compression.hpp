// Block-compressed texture formats (BC1 to BC7): a block of 4 x 4 texels in 8 or 16 bytes, decoded to what the
// texture unit hands the shader.
//
// The block layouts are from the public format descriptions (the Khronos Data Format Specification's
// block-compressed chapter, and the formats' D3D documentation). The arithmetic is NOT the description's
// integer formulas: it is what an RTX 3060's texture unit returns, measured by point sampling blocks with
// every endpoint pair (see the comments at each decoder), and the decoders reproduce those measurements
// value for value.
#pragma once

#include <algorithm>
#include <cstdint>

#include "vgpu/exec/bc67_tables.hpp"

namespace vgpu::exec {

// The formats a texture can be stored in. BC1, BC2 and BC3 decode to four unsigned normalized channels
// (8-bit; the sRGB variants are the same blocks, decoded from sRGB at the fetch), BC4 to one and BC5 to two
// 16-bit channels, unsigned or signed normalized, BC7 to four 8-bit ones (the sRGB variant is the same blocks) and
// BC6H to three half floats (the raw 16 bits) with alpha 1.0.
enum class BlockFormat : uint8_t { None, BC1, BC2, BC3, BC4U, BC4S, BC5U, BC5S, BC6HU, BC6HS, BC7 };

inline uint32_t block_bytes(BlockFormat f) {
  return f == BlockFormat::BC1 || f == BlockFormat::BC4U || f == BlockFormat::BC4S ? 8u : 16u;
}

// A decoded 4 x 4 block: for texel i (row-major: x = i % 4, y = i / 4), channel c. The raw channel bits as the
// texture path reads them: 8-bit codes for BC1, BC2, BC3 and BC7, 16-bit unsigned codes (BC4U, BC5U), 16-bit two's
// complement (BC4S, BC5S) or half-float bits (BC6H).
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

// ---- BC7 and BC6H (BPTC): the layouts are the public ones (Khronos Data Format Specification, "BPTC
// Compressed Texture Image Formats"; partition, anchor and bit-position tables in bc67_tables.hpp). ----

// Bits [pos, pos + n) of the 128-bit block, little endian, n <= 32.
inline uint32_t block_bits(const uint8_t* b, uint32_t pos, uint32_t n) {
  uint32_t v = 0;
  for (uint32_t i = 0; i < n; ++i) v |= static_cast<uint32_t>((b[(pos + i) >> 3] >> ((pos + i) & 7)) & 1) << i;
  return v;
}

inline constexpr int kWeights2[4] = {0, 21, 43, 64};
inline constexpr int kWeights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
inline constexpr int kWeights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
inline int bptc_weight(int index_bits, int index) {
  return index_bits == 2 ? kWeights2[index] : index_bits == 3 ? kWeights3[index] : kWeights4[index];
}
inline int bptc_lerp(int e0, int e1, int w) { return ((64 - w) * e0 + w * e1 + 32) >> 6; }

// The index stream of a BPTC block: `count` indices of `bits` bits from `pos` on, the anchor texels (the first
// of each subset in `anchors`) one bit short. Returns the position after the last.
inline uint32_t read_indices(const uint8_t* b, uint32_t pos, int bits, const int anchors[3], int nanchors, uint8_t out[16]) {
  for (int i = 0; i < 16; ++i) {
    bool anchor = false;
    for (int k = 0; k < nanchors; ++k) anchor = anchor || anchors[k] == i;
    const int n = anchor ? bits - 1 : bits;
    out[i] = static_cast<uint8_t>(block_bits(b, pos, static_cast<uint32_t>(n)));
    pos += static_cast<uint32_t>(n);
  }
  return pos;
}

struct Bc7Mode {
  uint8_t ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2;
};
inline constexpr Bc7Mode kBc7Modes[8] = {
    {3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0}, {3, 6, 0, 0, 5, 0, 0, 0, 2, 0},
    {2, 6, 0, 0, 7, 0, 1, 0, 2, 0}, {1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2},
    {1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0}};

inline void decode_bc7(const uint8_t* b, DecodedBlock* out) {
  int mode = 0;
  while (mode < 8 && !((b[0] >> mode) & 1)) ++mode;
  if (mode == 8) {   // a low byte of zero is reserved: all channels of all texels are 0
    for (int i = 0; i < 16; ++i) out->v[i][0] = out->v[i][1] = out->v[i][2] = out->v[i][3] = 0;
    return;
  }
  const Bc7Mode m = kBc7Modes[mode];
  uint32_t pos = static_cast<uint32_t>(mode) + 1;
  const uint32_t part = block_bits(b, pos, m.pb);
  pos += m.pb;
  const uint32_t rot = block_bits(b, pos, m.rb);
  pos += m.rb;
  const uint32_t isb = block_bits(b, pos, m.isb);
  pos += m.isb;
  int e[6][4];   // endpoint, channel
  for (int ch = 0; ch < 3; ++ch)
    for (int k = 0; k < 2 * m.ns; ++k) {
      e[k][ch] = static_cast<int>(block_bits(b, pos, m.cb));
      pos += m.cb;
    }
  for (int k = 0; k < 2 * m.ns; ++k) e[k][3] = 255;
  if (m.ab)
    for (int k = 0; k < 2 * m.ns; ++k) {
      e[k][3] = static_cast<int>(block_bits(b, pos, m.ab));
      pos += m.ab;
    }
  int pbit[6] = {0, 0, 0, 0, 0, 0};
  if (m.epb)
    for (int k = 0; k < 2 * m.ns; ++k) pbit[k] = static_cast<int>(block_bits(b, pos++, 1));
  if (m.spb)
    for (int s = 0; s < m.ns; ++s) pbit[2 * s] = pbit[2 * s + 1] = static_cast<int>(block_bits(b, pos++, 1));
  const bool pb = m.epb || m.spb;
  for (int k = 0; k < 2 * m.ns; ++k)
    for (int ch = 0; ch < 4; ++ch) {
      if (ch == 3 && !m.ab) continue;
      int bits = ch == 3 ? m.ab : m.cb;
      int v = e[k][ch];
      if (pb && (ch < 3 || m.epb)) {   // the P-bit goes below the data (shared P-bits: colour only)
        v = (v << 1) | pbit[k];
        ++bits;
      }
      e[k][ch] = (v << (8 - bits)) | (v >> (2 * bits - 8));
    }
  int anchors[3] = {0, 0, 0};
  uint8_t subset[16] = {};
  if (m.ns == 2) {
    anchors[1] = bc_tables::kAnchor2[part];
    for (int i = 0; i < 16; ++i) subset[i] = bc_tables::kPartition2[part][i];
  } else if (m.ns == 3) {
    anchors[1] = bc_tables::kAnchor3a[part];
    anchors[2] = bc_tables::kAnchor3b[part];
    for (int i = 0; i < 16; ++i) subset[i] = bc_tables::kPartition3[part][i];
  }
  uint8_t i1[16], i2[16] = {};
  pos = read_indices(b, pos, m.ib, anchors, m.ns, i1);
  if (m.ib2) read_indices(b, pos, m.ib2, anchors, m.ns, i2);
  for (int i = 0; i < 16; ++i) {
    const int s = subset[i];
    const int* e0 = e[2 * s];
    const int* e1 = e[2 * s + 1];
    int cib = m.ib, aib = m.ib, ci = i1[i], ai = i1[i];
    if (m.ib2) {
      if (isb) { cib = m.ib2; ci = i2[i]; } else { aib = m.ib2; ai = i2[i]; }
    }
    int c[4];
    for (int ch = 0; ch < 3; ++ch) c[ch] = bptc_lerp(e0[ch], e1[ch], bptc_weight(cib, ci));
    c[3] = bptc_lerp(e0[3], e1[3], bptc_weight(aib, ai));
    if (rot) std::swap(c[3], c[rot - 1]);
    for (int ch = 0; ch < 4; ++ch) out->v[i][ch] = static_cast<uint16_t>(c[ch]);
  }
}

struct Bc6hMode {
  uint8_t epb, delta[3];   // delta[0] == 0: the endpoints are not transformed
  bool two;                // two subsets (partition bits)
};
// In the order of bc_tables::kBc6hModeId (modes 0 1 2 6 10 14 18 22 26 30 3 7 11 15).
inline constexpr Bc6hMode kBc6hModes[14] = {
    {10, {5, 5, 5}, true}, {7, {6, 6, 6}, true},  {11, {5, 4, 4}, true}, {11, {4, 5, 4}, true},
    {11, {4, 4, 5}, true}, {9, {5, 5, 5}, true},  {8, {6, 5, 5}, true},  {8, {5, 6, 5}, true},
    {8, {5, 5, 6}, true},  {6, {0, 0, 0}, true},  {10, {0, 0, 0}, false}, {11, {9, 9, 9}, false},
    {12, {8, 8, 8}, false}, {16, {4, 4, 4}, false}};

inline int sign_extend(int x, int bits) {
  const int m = 1 << (bits - 1);
  return (x ^ m) - m;
}
inline int bc6h_unquantize(int x, int epb, bool is_signed) {
  if (!is_signed) {
    if (epb >= 15) return x;
    if (x == 0) return 0;
    if (x == (1 << epb) - 1) return 0xFFFF;
    return ((x << 15) + 0x4000) >> (epb - 1);
  }
  if (epb >= 16) return x;
  const bool neg = x < 0;
  if (neg) x = -x;
  int unq;
  if (x == 0) unq = 0;
  else if (x >= (1 << (epb - 1)) - 1) unq = 0x7FFF;
  else unq = ((x << 15) + 0x4000) >> (epb - 1);
  return neg ? -unq : unq;
}

inline void decode_bc6h(const uint8_t* b, bool is_signed, DecodedBlock* out) {
  auto reserved = [&] {
    for (int i = 0; i < 16; ++i) out->v[i][0] = out->v[i][1] = out->v[i][2] = 0, out->v[i][3] = 0x3C00;
  };
  int mode_value = static_cast<int>(block_bits(b, 0, 2));
  if (mode_value >= 2) mode_value = static_cast<int>(block_bits(b, 0, 5));
  int mi = -1;
  for (int k = 0; k < 14; ++k)
    if (bc_tables::kBc6hModeId[k] == mode_value) mi = k;
  if (mi < 0) return reserved();
  const Bc6hMode m = kBc6hModes[mi];
  int raw[3][4] = {};   // channel, endpoint
  uint32_t part = 0;
  for (uint32_t bit = 0; bit < 82; ++bit) {
    const uint16_t code = bc_tables::kBc6hBits[mi][bit];
    const int kind = code >> 12;
    const int val = (b[bit >> 3] >> (bit & 7)) & 1;
    if (kind == 1) part |= static_cast<uint32_t>(val) << (code & 31);
    else if (kind == 3) raw[(code >> 10) & 3][(code >> 8) & 3] |= val << (code & 31);
  }
  const int nep = m.two ? 4 : 2;
  int ep[4][3];
  for (int ch = 0; ch < 3; ++ch) {
    const int mask = (1 << m.epb) - 1;
    ep[0][ch] = is_signed ? sign_extend(raw[ch][0], m.epb) : raw[ch][0];
    for (int k = 1; k < nep; ++k) {
      if (m.delta[0]) {
        const int d = sign_extend(raw[ch][k], m.delta[ch]);
        int v = (raw[ch][0] + d) & mask;   // wrapped at the endpoint's width
        ep[k][ch] = is_signed ? sign_extend(v, m.epb) : v;
      } else {
        ep[k][ch] = is_signed ? sign_extend(raw[ch][k], m.epb) : raw[ch][k];
      }
    }
  }
  int q[4][3];
  for (int k = 0; k < nep; ++k)
    for (int ch = 0; ch < 3; ++ch) q[k][ch] = bc6h_unquantize(ep[k][ch], m.epb, is_signed);
  int anchors[3] = {0, 0, 0};
  uint8_t subset[16] = {};
  int ib = 4;
  if (m.two) {
    ib = 3;
    anchors[1] = bc_tables::kAnchor2[part];
    for (int i = 0; i < 16; ++i) subset[i] = bc_tables::kPartition2[part][i];
  }
  uint8_t idx[16];
  read_indices(b, m.two ? 82 : 65, ib, anchors, m.two ? 2 : 1, idx);
  for (int i = 0; i < 16; ++i) {
    const int w = bptc_weight(ib, idx[i]);
    for (int ch = 0; ch < 3; ++ch) {
      const int i0 = q[2 * subset[i]][ch], i1 = q[2 * subset[i] + 1][ch];
      const int v = bptc_lerp(i0, i1, w);
      int h;
      // The magnitude is scaled first and the sign restored only when it is not zero: the card returns +0 where
      // the specification's "| 0x8000" would make -0 (measured, a small negative interpolated value).
      if (is_signed) {
        const int mag = ((v < 0 ? -v : v) * 31) >> 5;
        h = v < 0 && mag != 0 ? mag | 0x8000 : mag;
      } else {
        h = (v * 31) >> 6;
      }
      out->v[i][ch] = static_cast<uint16_t>(h);
    }
    out->v[i][3] = 0x3C00;   // 1.0
  }
}

}  // namespace bc_detail

// Decodes one block of BC1 to BC5.
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

// Decodes one block of any of the formats. False for BlockFormat::None.
inline bool decode_block(BlockFormat f, const uint8_t* b, DecodedBlock* out) {
  switch (f) {
    case BlockFormat::BC1: case BlockFormat::BC2: case BlockFormat::BC3:
    case BlockFormat::BC4U: case BlockFormat::BC4S: case BlockFormat::BC5U: case BlockFormat::BC5S:
      decode_block_ldr(f, b, out);
      return true;
    case BlockFormat::BC6HU:
    case BlockFormat::BC6HS:
      bc_detail::decode_bc6h(b, f == BlockFormat::BC6HS, out);
      return true;
    case BlockFormat::BC7:
      bc_detail::decode_bc7(b, out);
      return true;
    default:
      return false;
  }
}

}  // namespace vgpu::exec
