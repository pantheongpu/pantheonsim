// RDNA's image, sampler and formatted buffer resources, and the texels they
// describe (vgpu/amd_image.hpp).
#include "vgpu/amd_image.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>

namespace vgpu::amd::image {

namespace {

uint32_t bits(uint32_t v, uint32_t hi, uint32_t lo) { return (v >> lo) & ((hi - lo == 31) ? ~0u : ((1u << (hi - lo + 1)) - 1)); }
uint32_t as_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}
float as_float(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

// Each data format's channels: how many, and each one's width in bits (the
// channels in memory order, x first).
struct Layout {
  uint32_t count;
  uint32_t width[4];
};
Layout layout(Data d) {
  switch (d) {
    case Data::D8: return {1, {8}};
    case Data::D16: return {1, {16}};
    case Data::D8_8: return {2, {8, 8}};
    case Data::D32: return {1, {32}};
    case Data::D16_16: return {2, {16, 16}};
    // The names give the channels from the top bit down: 2_10_10_10 is
    // three 10-bit channels with a 2-bit alpha above them.
    case Data::D10_10_10_2: return {4, {2, 10, 10, 10}};
    case Data::D2_10_10_10: return {4, {10, 10, 10, 2}};
    case Data::D8_8_8_8: return {4, {8, 8, 8, 8}};
    case Data::D32_32: return {2, {32, 32}};
    case Data::D16_16_16_16: return {4, {16, 16, 16, 16}};
    case Data::D32_32_32: return {3, {32, 32, 32}};
    case Data::D32_32_32_32: return {4, {32, 32, 32, 32}};
    default: return {0, {}};
  }
}

// One channel's bits, as the number format reads them.
uint32_t decode_channel(Num n, uint32_t raw, uint32_t width) {
  const uint64_t max = (uint64_t{1} << width) - 1;
  const auto sign_extend = [&](uint32_t v) {
    return width == 32 ? static_cast<int32_t>(v) : static_cast<int32_t>(v << (32 - width)) >> (32 - width);
  };
  switch (n) {
    case Num::Srgb:
    case Num::Unorm: return as_bits(static_cast<float>(raw) / static_cast<float>(max));
    case Num::Snorm: {
      const float v = static_cast<float>(sign_extend(raw)) / static_cast<float>((uint64_t{1} << (width - 1)) - 1);
      return as_bits(std::max(-1.0f, v));
    }
    case Num::Uscaled: return as_bits(static_cast<float>(raw));
    case Num::Sscaled: return as_bits(static_cast<float>(sign_extend(raw)));
    case Num::Uint: return raw;
    case Num::Sint: return static_cast<uint32_t>(sign_extend(raw));
    case Num::Float:
      if (width == 32) return raw;
      if (width == 16) {
        _Float16 h;
        const uint16_t b = static_cast<uint16_t>(raw);
        std::memcpy(&h, &b, 2);
        return as_bits(static_cast<float>(h));
      }
      return 0;
  }
  return 0;
}

// And a channel's bits from a value given the same way: a float saturated
// and rounded to the nearest step for the normalized formats, an integer's
// low bits for the integer ones.
uint32_t encode_channel(Num n, uint32_t v, uint32_t width) {
  const uint64_t max = (uint64_t{1} << width) - 1;
  const uint32_t mask = static_cast<uint32_t>(max);
  switch (n) {
    case Num::Srgb:
    case Num::Unorm: {
      const float f = std::clamp(as_float(v), 0.0f, 1.0f);
      return static_cast<uint32_t>(std::lround(f * static_cast<float>(max))) & mask;
    }
    case Num::Snorm: {
      const float f = std::clamp(as_float(v), -1.0f, 1.0f);
      const float m = static_cast<float>((uint64_t{1} << (width - 1)) - 1);
      return static_cast<uint32_t>(static_cast<int32_t>(std::lround(f * m))) & mask;
    }
    case Num::Uscaled: return static_cast<uint32_t>(std::max(0.0f, as_float(v))) & mask;
    case Num::Sscaled: return static_cast<uint32_t>(static_cast<int32_t>(as_float(v))) & mask;
    case Num::Uint:
    case Num::Sint: return v & mask;
    case Num::Float:
      if (width == 32) return v;
      if (width == 16) {
        const _Float16 h = static_cast<_Float16>(as_float(v));
        uint16_t b;
        std::memcpy(&b, &h, 2);
        return b;
      }
      return 0;
  }
  return 0;
}

// RDNA gives a format as one code (BUF_FMT_* and IMG_FMT_*, the same
// numbers for the formats both have) in place of GCN's two fields; gfx11
// renumbered them, dropping the 10- and 11-bit channels' non-float forms.
// These are the codes as the LLVM assembler's format:[...] names them.
const Format kGfx10[] = {
    {},   // 0: invalid
    {Data::D8, Num::Unorm}, {Data::D8, Num::Snorm}, {Data::D8, Num::Uscaled}, {Data::D8, Num::Sscaled},
    {Data::D8, Num::Uint}, {Data::D8, Num::Sint}, {Data::D16, Num::Unorm}, {Data::D16, Num::Snorm},
    {Data::D16, Num::Uscaled}, {Data::D16, Num::Sscaled}, {Data::D16, Num::Uint}, {Data::D16, Num::Sint},
    {Data::D16, Num::Float}, {Data::D8_8, Num::Unorm}, {Data::D8_8, Num::Snorm}, {Data::D8_8, Num::Uscaled},
    {Data::D8_8, Num::Sscaled}, {Data::D8_8, Num::Uint}, {Data::D8_8, Num::Sint}, {Data::D32, Num::Uint},
    {Data::D32, Num::Sint}, {Data::D32, Num::Float}, {Data::D16_16, Num::Unorm}, {Data::D16_16, Num::Snorm},
    {Data::D16_16, Num::Uscaled}, {Data::D16_16, Num::Sscaled}, {Data::D16_16, Num::Uint},
    {Data::D16_16, Num::Sint}, {Data::D16_16, Num::Float}, {Data::D10_11_11, Num::Unorm},
    {Data::D10_11_11, Num::Snorm}, {Data::D10_11_11, Num::Uscaled}, {Data::D10_11_11, Num::Sscaled},
    {Data::D10_11_11, Num::Uint}, {Data::D10_11_11, Num::Sint}, {Data::D10_11_11, Num::Float},
    {Data::D11_11_10, Num::Unorm}, {Data::D11_11_10, Num::Snorm}, {Data::D11_11_10, Num::Uscaled},
    {Data::D11_11_10, Num::Sscaled}, {Data::D11_11_10, Num::Uint}, {Data::D11_11_10, Num::Sint},
    {Data::D11_11_10, Num::Float}, {Data::D10_10_10_2, Num::Unorm}, {Data::D10_10_10_2, Num::Snorm},
    {Data::D10_10_10_2, Num::Uscaled}, {Data::D10_10_10_2, Num::Sscaled}, {Data::D10_10_10_2, Num::Uint},
    {Data::D10_10_10_2, Num::Sint}, {Data::D2_10_10_10, Num::Unorm}, {Data::D2_10_10_10, Num::Snorm},
    {Data::D2_10_10_10, Num::Uscaled}, {Data::D2_10_10_10, Num::Sscaled}, {Data::D2_10_10_10, Num::Uint},
    {Data::D2_10_10_10, Num::Sint}, {Data::D8_8_8_8, Num::Unorm}, {Data::D8_8_8_8, Num::Snorm},
    {Data::D8_8_8_8, Num::Uscaled}, {Data::D8_8_8_8, Num::Sscaled}, {Data::D8_8_8_8, Num::Uint},
    {Data::D8_8_8_8, Num::Sint}, {Data::D32_32, Num::Uint}, {Data::D32_32, Num::Sint},
    {Data::D32_32, Num::Float}, {Data::D16_16_16_16, Num::Unorm}, {Data::D16_16_16_16, Num::Snorm},
    {Data::D16_16_16_16, Num::Uscaled}, {Data::D16_16_16_16, Num::Sscaled}, {Data::D16_16_16_16, Num::Uint},
    {Data::D16_16_16_16, Num::Sint}, {Data::D16_16_16_16, Num::Float}, {Data::D32_32_32, Num::Uint},
    {Data::D32_32_32, Num::Sint}, {Data::D32_32_32, Num::Float}, {Data::D32_32_32_32, Num::Uint},
    {Data::D32_32_32_32, Num::Sint}, {Data::D32_32_32_32, Num::Float},
};

const Format kGfx11[] = {
    {},   // 0: invalid
    {Data::D8, Num::Unorm}, {Data::D8, Num::Snorm}, {Data::D8, Num::Uscaled}, {Data::D8, Num::Sscaled},
    {Data::D8, Num::Uint}, {Data::D8, Num::Sint}, {Data::D16, Num::Unorm}, {Data::D16, Num::Snorm},
    {Data::D16, Num::Uscaled}, {Data::D16, Num::Sscaled}, {Data::D16, Num::Uint}, {Data::D16, Num::Sint},
    {Data::D16, Num::Float}, {Data::D8_8, Num::Unorm}, {Data::D8_8, Num::Snorm}, {Data::D8_8, Num::Uscaled},
    {Data::D8_8, Num::Sscaled}, {Data::D8_8, Num::Uint}, {Data::D8_8, Num::Sint}, {Data::D32, Num::Uint},
    {Data::D32, Num::Sint}, {Data::D32, Num::Float}, {Data::D16_16, Num::Unorm}, {Data::D16_16, Num::Snorm},
    {Data::D16_16, Num::Uscaled}, {Data::D16_16, Num::Sscaled}, {Data::D16_16, Num::Uint},
    {Data::D16_16, Num::Sint}, {Data::D16_16, Num::Float}, {Data::D10_11_11, Num::Float},
    {Data::D11_11_10, Num::Float}, {Data::D10_10_10_2, Num::Unorm}, {Data::D10_10_10_2, Num::Snorm},
    {Data::D10_10_10_2, Num::Uint}, {Data::D10_10_10_2, Num::Sint}, {Data::D2_10_10_10, Num::Unorm},
    {Data::D2_10_10_10, Num::Snorm}, {Data::D2_10_10_10, Num::Uscaled}, {Data::D2_10_10_10, Num::Sscaled},
    {Data::D2_10_10_10, Num::Uint}, {Data::D2_10_10_10, Num::Sint}, {Data::D8_8_8_8, Num::Unorm},
    {Data::D8_8_8_8, Num::Snorm}, {Data::D8_8_8_8, Num::Uscaled}, {Data::D8_8_8_8, Num::Sscaled},
    {Data::D8_8_8_8, Num::Uint}, {Data::D8_8_8_8, Num::Sint}, {Data::D32_32, Num::Uint},
    {Data::D32_32, Num::Sint}, {Data::D32_32, Num::Float}, {Data::D16_16_16_16, Num::Unorm},
    {Data::D16_16_16_16, Num::Snorm}, {Data::D16_16_16_16, Num::Uscaled}, {Data::D16_16_16_16, Num::Sscaled},
    {Data::D16_16_16_16, Num::Uint}, {Data::D16_16_16_16, Num::Sint}, {Data::D16_16_16_16, Num::Float},
    {Data::D32_32_32, Num::Uint}, {Data::D32_32_32, Num::Sint}, {Data::D32_32_32, Num::Float},
    {Data::D32_32_32_32, Num::Uint}, {Data::D32_32_32_32, Num::Sint}, {Data::D32_32_32_32, Num::Float},
};

// sRGB's transfer function, each way.
float srgb_to_linear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }
float linear_to_srgb(float c) {
  c = std::clamp(c, 0.0f, 1.0f);
  return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// The image formats with sRGB color: 8, 8_8 and 8_8_8_8, at these codes
// (ROCR-Runtime's resource_nv.h, resource_gfx11.h, resource_gfx12.h).
uint32_t srgb_base(Gen g) { return g == Gen::Gfx10 ? 128 : 64; }

}  // namespace

Format from_code(uint32_t code, Gen g) {
  if (code >= srgb_base(g) && code < srgb_base(g) + 3) {
    static const Data kData[] = {Data::D8, Data::D8_8, Data::D8_8_8_8};
    return {kData[code - srgb_base(g)], Num::Srgb};
  }
  const bool gfx11 = g != Gen::Gfx10;
  const Format* t = gfx11 ? kGfx11 : kGfx10;
  const uint32_t n = gfx11 ? std::size(kGfx11) : std::size(kGfx10);
  return code < n ? t[code] : Format{};
}
uint32_t code(Format f, Gen g) {
  if (f.num == Num::Srgb)
    return f.data == Data::D8 ? srgb_base(g) : f.data == Data::D8_8 ? srgb_base(g) + 1
           : f.data == Data::D8_8_8_8 ? srgb_base(g) + 2 : 0;
  const bool gfx11 = g != Gen::Gfx10;
  const Format* t = gfx11 ? kGfx11 : kGfx10;
  const uint32_t n = gfx11 ? std::size(kGfx11) : std::size(kGfx10);
  for (uint32_t c = 1; c < n; ++c)
    if (t[c].data == f.data && t[c].num == f.num) return c;
  return 0;
}

uint32_t channels(Format f) { return layout(f.data).count; }
uint32_t texel_bytes(Format f) {
  const Layout l = layout(f.data);
  uint32_t b = 0;
  for (uint32_t k = 0; k < l.count; ++k) b += l.width[k];
  return b / 8;
}
bool integer(Format f) { return f.num == Num::Uint || f.num == Num::Sint; }

void read_texel(Format f, const uint8_t* bytes, uint32_t out[4]) {
  const Layout l = layout(f.data);
  out[0] = out[1] = out[2] = 0;
  out[3] = integer(f) ? 1 : as_bits(1.0f);
  uint64_t lo = 0, hi = 0;   // the texel's bits, up to 128 of them
  const uint32_t n = texel_bytes(f);
  std::memcpy(&lo, bytes, std::min<uint32_t>(n, 8));
  if (n > 8) std::memcpy(&hi, bytes + 8, n - 8);
  uint32_t at = 0;
  for (uint32_t k = 0; k < l.count; ++k) {
    const uint32_t w = l.width[k];
    uint64_t raw = at < 64 ? lo >> at : hi >> (at - 64);
    if (at < 64 && at + w > 64) raw |= hi << (64 - at);
    out[k] = decode_channel(f.num, static_cast<uint32_t>(raw & ((uint64_t{1} << w) - 1)), w);
    if (f.num == Num::Srgb && k < 3) out[k] = as_bits(srgb_to_linear(as_float(out[k])));
    at += w;
  }
}

void write_texel(Format f, const uint32_t in[4], uint8_t* bytes) {
  const Layout l = layout(f.data);
  uint64_t lo = 0, hi = 0;
  uint32_t at = 0;
  for (uint32_t k = 0; k < l.count; ++k) {
    const uint32_t w = l.width[k];
    const uint32_t value = f.num == Num::Srgb && k < 3 ? as_bits(linear_to_srgb(as_float(in[k]))) : in[k];
    const uint64_t v = encode_channel(f.num, value, w);
    if (at < 64) lo |= v << at;
    if (at + w > 64) hi |= at >= 64 ? v << (at - 64) : v >> (64 - at);
    at += w;
  }
  const uint32_t n = texel_bytes(f);
  std::memcpy(bytes, &lo, std::min<uint32_t>(n, 8));
  if (n > 8) std::memcpy(bytes + 8, &hi, n - 8);
}

// ---- The image resource (T#) ----------------------------------------------------
//
// As ROCm's image runtime writes it (ROCR-Runtime's resource_nv.h,
// resource_gfx11.h and resource_gfx12.h). word 0: the base address's bits
// 39:8. word 1: bits 47:40 [7:0]; the format's code, [28:20] on gfx10,
// [27:20] on gfx11, [24:17] on gfx12 (with BASE_LEVEL [29:25]); the width
// less one's low two bits [31:30]. word 2: the rest of the width less one
// [11:0] ([13:0] on gfx12); the height less one [27:14] ([29:14] on gfx12),
// where HIP's texture functions read it. word 3: the four DST_SELs [11:0];
// BASE_LEVEL [15:12] and LAST_LEVEL [19:16] (gfx12: LAST_LEVEL [19:15]);
// the tiling [24:20] (0, linear); TYPE [31:28]. word 4 [13:0]: a 3D image's
// depth less one, an array's last layer, or else the row pitch less one, in
// texels.

namespace {
bool array_type(Type t) { return t == Type::Img1DArray || t == Type::Img2DArray || t == Type::Cube; }
}  // namespace

void encode(const Image& i, Gen g, uint32_t w[8]) {
  std::memset(w, 0, 8 * sizeof(uint32_t));
  const uint32_t wm1 = std::max(1u, i.width) - 1, hm1 = std::max(1u, i.height) - 1;
  const uint32_t pitch = i.pitch ? i.pitch : std::max(1u, i.width);
  const uint32_t fmt = code(i.format, g);
  w[0] = static_cast<uint32_t>(i.base >> 8);
  w[1] = static_cast<uint32_t>(i.base >> 40) & 0xFF;
  w[1] |= (wm1 & 3) << 30;
  for (int k = 0; k < 4; ++k) w[3] |= (static_cast<uint32_t>(i.sel[k]) & 7) << (3 * k);
  w[3] |= (static_cast<uint32_t>(i.type) & 0xF) << 28;
  if (g == Gen::Gfx12) {
    w[1] |= (fmt & 0xFF) << 17 | (i.base_level & 0x1F) << 25;
    w[2] = ((wm1 >> 2) & 0x3FFF) | (hm1 & 0xFFFF) << 14;
    w[3] |= (i.last_level & 0x1F) << 15;
  } else {
    w[1] |= (fmt & (g == Gen::Gfx10 ? 0x1FF : 0xFF)) << 20;
    w[2] = ((wm1 >> 2) & 0xFFF) | (hm1 & 0x3FFF) << 14;
    w[3] |= (i.base_level & 0xF) << 12 | (i.last_level & 0xF) << 16;
  }
  if (i.type == Type::Img3D || array_type(i.type)) w[4] = (std::max(1u, i.depth) - 1) & 0x1FFF;
  else w[4] = (pitch - 1) & 0x3FFF;
}

Image decode_image(const uint32_t w[8], Gen g) {
  Image i;
  i.base = uint64_t{w[0]} << 8 | uint64_t{w[1] & 0xFF} << 40;
  for (int k = 0; k < 4; ++k) i.sel[k] = static_cast<Sel>(bits(w[3], 3 * k + 2, 3 * k));
  i.type = static_cast<Type>(bits(w[3], 31, 28));
  if (g == Gen::Gfx12) {
    i.format = from_code(bits(w[1], 24, 17), g);
    i.base_level = bits(w[1], 29, 25);
    i.width = (bits(w[1], 31, 30) | bits(w[2], 13, 0) << 2) + 1;
    i.height = bits(w[2], 29, 14) + 1;
    i.last_level = bits(w[3], 19, 15);
  } else {
    i.format = from_code(g == Gen::Gfx10 ? bits(w[1], 28, 20) : bits(w[1], 27, 20), g);
    i.width = (bits(w[1], 31, 30) | bits(w[2], 11, 0) << 2) + 1;
    i.height = bits(w[2], 27, 14) + 1;
    i.base_level = bits(w[3], 15, 12);
    i.last_level = bits(w[3], 19, 16);
  }
  if (i.type == Type::Img3D || array_type(i.type)) {
    i.depth = bits(w[4], 12, 0) + 1;
    i.pitch = i.width;
  } else {
    i.pitch = bits(w[4], 13, 0) + 1;
  }
  return i;
}

uint32_t level_width(const Image& i, uint32_t level) { return std::max(1u, i.width >> level); }
uint32_t level_height(const Image& i, uint32_t level) {
  return i.type == Type::Img1D || i.type == Type::Img1DArray ? 1 : std::max(1u, i.height >> level);
}
uint32_t level_depth(const Image& i, uint32_t level) {
  if (i.type == Type::Img3D) return std::max(1u, i.depth >> level);
  if (i.type == Type::Img1DArray || i.type == Type::Img2DArray || i.type == Type::Cube) return i.depth;
  return 1;
}
namespace {
uint32_t level_pitch(const Image& i, uint32_t level) {
  return level == 0 ? (i.pitch ? i.pitch : i.width) : level_width(i, level);
}
uint64_t level_bytes(const Image& i, uint32_t level) {
  return uint64_t{level_pitch(i, level)} * level_height(i, level) * level_depth(i, level) * texel_bytes(i.format);
}
}  // namespace

uint64_t texel_offset(const Image& i, uint32_t level, uint32_t x, uint32_t y, uint32_t z) {
  uint64_t at = 0;
  for (uint32_t l = 0; l < level; ++l) at += level_bytes(i, l);
  const uint64_t row = uint64_t{level_pitch(i, level)} * texel_bytes(i.format);
  return at + (uint64_t{z} * level_height(i, level) + y) * row + uint64_t{x} * texel_bytes(i.format);
}

uint64_t image_bytes(const Image& i) {
  uint64_t n = 0;
  for (uint32_t l = 0; l <= i.last_level; ++l) n += level_bytes(i, l);
  return n;
}

// ---- The sampler resource (S#) -------------------------------------------------
//
// word 0: CLAMP_X [2:0], CLAMP_Y [5:3], CLAMP_Z [8:6], FORCE_UNNORMALIZED
// [15], where HIP's texture functions read it. word 1: MIN_LOD [11:0],
// MAX_LOD [23:12] (unsigned 4.8; on gfx12 [12:0] and [25:13]). word 2:
// LOD_BIAS [13:0] (signed 6.8), XY_MAG_FILTER [21:20] (read by HIP's
// texture functions too), XY_MIN_FILTER [23:22], MIP_FILTER [27:26]. word 3:
// BORDER_COLOR_TYPE [31:30].

void encode(const Sampler& s, Gen g, uint32_t w[4]) {
  w[0] = w[1] = w[2] = w[3] = 0;
  for (int k = 0; k < 3; ++k) w[0] |= (static_cast<uint32_t>(s.clamp[k]) & 7) << (3 * k);
  if (s.unnormalized) w[0] |= 1u << 15;
  const float top = g == Gen::Gfx12 ? 31.99f : 15.99f;
  const uint32_t lod_mask = g == Gen::Gfx12 ? 0x1FFF : 0xFFF, max_at = g == Gen::Gfx12 ? 13 : 12;
  const auto fixed = [&](float v) { return static_cast<uint32_t>(std::clamp(v, 0.0f, top) * 256.0f) & lod_mask; };
  w[1] = fixed(s.min_lod) | fixed(s.max_lod) << max_at;
  w[2] = static_cast<uint32_t>(static_cast<int32_t>(std::clamp(s.lod_bias, -32.0f, 31.99f) * 256.0f)) & 0x3FFF;
  w[2] |= (s.mag_linear ? 1u : 0u) << 20 | (s.min_linear ? 1u : 0u) << 22 | (s.mip_filter & 3u) << 26;
  w[3] = (s.border & 3u) << 30;
}

Sampler decode_sampler(const uint32_t w[4], Gen g) {
  Sampler s;
  for (int k = 0; k < 3; ++k) s.clamp[k] = static_cast<Clamp>(bits(w[0], 3 * k + 2, 3 * k));
  s.unnormalized = bits(w[0], 15, 15);
  if (g == Gen::Gfx12) {
    s.min_lod = static_cast<float>(bits(w[1], 12, 0)) / 256.0f;
    s.max_lod = static_cast<float>(bits(w[1], 25, 13)) / 256.0f;
  } else {
    s.min_lod = static_cast<float>(bits(w[1], 11, 0)) / 256.0f;
    s.max_lod = static_cast<float>(bits(w[1], 23, 12)) / 256.0f;
  }
  s.lod_bias = static_cast<float>(static_cast<int32_t>(bits(w[2], 13, 0) << 18) >> 18) / 256.0f;
  s.mag_linear = bits(w[2], 21, 20) != 0;
  s.min_linear = bits(w[2], 23, 22) != 0;
  s.mip_filter = static_cast<uint8_t>(bits(w[2], 27, 26));
  s.border = static_cast<uint8_t>(bits(w[3], 31, 30));
  return s;
}

// ---- A formatted buffer's resource (V#) ----------------------------------------
//
// word 0: the base address's low bits; word 1: its high 16 [15:0], the
// stride [29:16]; word 2: the number of records; word 3: the four DST_SELs
// [11:0], the format's code [18:12] ([17:12] from gfx11), OOB_SELECT
// [29:28] (0: an index is in range below the record count).

void encode_buffer(uint64_t base, uint32_t stride, uint32_t records, Format format, Gen g, uint32_t w[4]) {
  w[0] = static_cast<uint32_t>(base);
  w[1] = static_cast<uint32_t>(base >> 32) & 0xFFFF;
  w[1] |= (stride & 0x3FFF) << 16;
  w[2] = records;
  w[3] = 4 | 5 << 3 | 6 << 6 | 7 << 9;   // x, y, z, w
  w[3] |= (code(format, g) & (g == Gen::Gfx10 ? 0x7F : 0x3F)) << 12;
}

Format buffer_format(const uint32_t w[4], Gen g) {
  return from_code(g == Gen::Gfx10 ? bits(w[3], 18, 12) : bits(w[3], 17, 12), g);
}
Sel buffer_sel(const uint32_t w[4], int channel) { return static_cast<Sel>(bits(w[3], 3 * channel + 2, 3 * channel)); }

}  // namespace vgpu::amd::image
