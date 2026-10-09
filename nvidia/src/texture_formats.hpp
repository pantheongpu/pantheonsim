// What a CUDA array's channel format means to the texture unit, in one place for the runtime
// (cudaChannelFormatDesc) and the driver (CUarray_format): how a texel is stored, how it is
// delivered, and which read modes and filters the format allows. Measured on an RTX 3060 (driver 13.2):
// cudaMallocArray's validation of a descriptor, and cudaCreateTextureObject's of a read mode, a filter
// and the sRGB flag against it.
#pragma once

#include <cstdint>

#include "vgpu/exec/block_compression.hpp"
#include "vgpu/exec/texture.hpp"

namespace vgpu::cuda {

struct TexFormat {
  vgpu::exec::ChannelKind kind = vgpu::exec::ChannelKind::Float;   // how the sampled channels are interpreted
  uint32_t channels = 0;
  uint32_t bits[4] = {0, 0, 0, 0};    // the sampled channels' widths (a block-compressed format's decoded ones)
  uint32_t texel_bytes = 0;           // stored: a texel, or a whole block
  vgpu::exec::BlockFormat block = vgpu::exec::BlockFormat::None;
  bool packed_1010102 = false;
  // The ways it may be read.
  bool norm_only = false;     // cudaReadModeNormalizedFloat only: normalized integers, block-compressed (but BC6H)
  bool elem_only = false;     // cudaReadModeElementType only: floats, BC6H, 10:10:10:2
  bool srgb_only = false;     // the sRGB block-compressed kinds: the sRGB flag is required
  bool integer = false;       // plain integer channels: linear filtering needs the normalized read, 8 or 16 bits
  bool is_int32 = false;      // plain 32-bit integers: no normalized read
};

// The descriptor cudaMallocArray and its relatives accept, from cudaChannelFormatDesc's fields: the kind
// (cudaChannelFormatKind) and the four widths. False for one the card refuses
// (cudaErrorInvalidChannelDescriptor): none, NV12 (which only external memory makes), widths that are not
// the format's, a channel count of 3, channels with gaps.
inline bool texture_format_from_runtime(int kind, int x, int y, int z, int w, TexFormat* out) {
  using vgpu::exec::BlockFormat;
  using vgpu::exec::ChannelKind;
  TexFormat f;
  const int b[4] = {x, y, z, w};
  int n = 0;
  while (n < 4 && b[n] != 0) ++n;
  for (int i = n; i < 4; ++i)
    if (b[i] != 0) return false;   // a gap
  auto same = [&](int bits) {
    for (int i = 0; i < n; ++i)
      if (b[i] != bits) return false;
    return true;
  };
  auto layout = [&](int bits, int count, ChannelKind k) {
    if (n != count || !same(bits)) return false;
    f.kind = k;
    f.channels = static_cast<uint32_t>(n);
    for (int i = 0; i < n; ++i) f.bits[i] = static_cast<uint32_t>(bits);
    f.texel_bytes = static_cast<uint32_t>(n * bits / 8);
    return true;
  };
  switch (kind) {
    case 0:   // signed
    case 1: {   // unsigned
      if (n != 1 && n != 2 && n != 4) return false;
      if (b[0] != 8 && b[0] != 16 && b[0] != 32) return false;
      if (!layout(b[0], n, kind == 0 ? ChannelKind::Signed : ChannelKind::Unsigned)) return false;
      f.integer = true;
      f.is_int32 = b[0] == 32;
      break;
    }
    case 2: {   // float
      if (n != 1 && n != 2 && n != 4) return false;
      if (b[0] != 16 && b[0] != 32) return false;
      if (!layout(b[0], n, ChannelKind::Float)) return false;
      f.elem_only = true;
      break;
    }
    case 5: case 6: case 7:     // unsigned normalized, 8-bit, 1, 2 and 4 channels
    case 8: case 9: case 10:    // 16-bit
    case 11: case 12: case 13:  // signed normalized, 8-bit
    case 14: case 15: case 16: {   // 16-bit
      const int idx = kind - 5;
      const bool is_signed = kind >= 11;
      const int bits = ((idx % 6) < 3) ? 8 : 16;
      const int count = (idx % 3) == 0 ? 1 : (idx % 3) == 1 ? 2 : 4;
      if (!layout(bits, count, is_signed ? ChannelKind::Signed : ChannelKind::Unsigned)) return false;
      f.norm_only = true;
      break;
    }
    case 17: case 18: case 19: case 20: case 21: case 22:
    case 23: case 24: case 25: case 26: case 27: case 28: case 29: case 30: {
      // Block-compressed: BC1, BC2, BC3 (each with an sRGB kind after it), BC4, BC5 (unsigned then signed),
      // BC6H (unsigned then signed), BC7 (with sRGB).
      BlockFormat bf = BlockFormat::None;
      int channels = 4, bits = 8;
      bool srgb = false, sgn = false;
      switch (kind) {
        case 17: case 18: bf = BlockFormat::BC1; srgb = kind == 18; break;
        case 19: case 20: bf = BlockFormat::BC2; srgb = kind == 20; break;
        case 21: case 22: bf = BlockFormat::BC3; srgb = kind == 22; break;
        case 23: bf = BlockFormat::BC4U; channels = 1; break;
        case 24: bf = BlockFormat::BC4S; channels = 1; sgn = true; break;
        case 25: bf = BlockFormat::BC5U; channels = 2; break;
        case 26: bf = BlockFormat::BC5S; channels = 2; sgn = true; break;
        case 27: bf = BlockFormat::BC6HU; channels = 3; bits = 16; break;
        case 28: bf = BlockFormat::BC6HS; channels = 3; bits = 16; sgn = true; break;
        default: bf = BlockFormat::BC7; srgb = kind == 30; break;
      }
      // The descriptor's widths are fixed by the format.
      if (n != (bf == BlockFormat::BC4U || bf == BlockFormat::BC4S ? 1 : bf == BlockFormat::BC5U || bf == BlockFormat::BC5S ? 2
                                                                         : bf == BlockFormat::BC6HU || bf == BlockFormat::BC6HS ? 3 : 4))
        return false;
      const int want = bf == BlockFormat::BC6HU || bf == BlockFormat::BC6HS ? 16 : 8;
      if (!same(want)) return false;
      f.block = bf;
      f.channels = static_cast<uint32_t>(channels);
      for (int i = 0; i < channels; ++i) f.bits[i] = static_cast<uint32_t>(bits);
      // BC1, BC2, BC3 and BC7 are delivered at 8 bits; the channels of BC4 and BC5 at 16 bits (measured: their
      // values are not multiples of 1/255).
      if (bf == BlockFormat::BC4U || bf == BlockFormat::BC4S || bf == BlockFormat::BC5U || bf == BlockFormat::BC5S)
        for (int i = 0; i < channels; ++i) f.bits[i] = 16;
      f.kind = (bf == BlockFormat::BC6HU || bf == BlockFormat::BC6HS) ? ChannelKind::Float
               : sgn ? ChannelKind::Signed : ChannelKind::Unsigned;
      uint32_t virt = 0;
      for (uint32_t i = 0; i < f.channels; ++i) virt += f.bits[i] / 8;
      f.texel_bytes = vgpu::exec::block_bytes(bf);   // stored per block; sampled texels are virtual
      f.norm_only = !(bf == BlockFormat::BC6HU || bf == BlockFormat::BC6HS);
      f.elem_only = !f.norm_only;
      f.srgb_only = srgb;
      (void)virt;
      break;
    }
    case 31: {   // 10:10:10:2 unsigned normalized
      if (x != 10 || y != 10 || z != 10 || w != 2) return false;
      f.kind = ChannelKind::Unsigned;
      f.channels = 4;
      f.bits[0] = f.bits[1] = f.bits[2] = 10;
      f.bits[3] = 2;
      f.texel_bytes = 4;
      f.packed_1010102 = true;
      f.elem_only = true;
      break;
    }
    default:
      return false;
  }
  *out = f;
  return true;
}

// What cudaCreateTextureObject does with a format, a read mode and a filter, and the sRGB flag (the card's
// answers: 0 for success, else the error). 27 is cudaErrorInvalidNormSetting, 26 cudaErrorInvalidFilterSetting.
// Checked in this order: a filter an integer format cannot do, the read mode, the sRGB flag.
inline int texture_read_check(const TexFormat& f, bool normalized_read, bool linear, bool srgb) {
  if (f.integer) {
    if (linear && (!normalized_read || f.is_int32)) return 26;
    if (normalized_read && f.is_int32) return 27;
    return 0;
  }
  if (f.elem_only && normalized_read) return 27;
  if (f.norm_only && !normalized_read) return 27;
  if (f.srgb_only && !srgb) return 1;
  return 0;
}

// What a sampled texel of this format takes in a texture descriptor's (virtual) layout: a block-compressed
// array's texels are decoded, so their size is what they decode to.
inline uint32_t sampled_texel_bytes(const TexFormat& f) {
  if (f.block == vgpu::exec::BlockFormat::None) return f.texel_bytes;
  uint32_t n = 0;
  for (uint32_t i = 0; i < f.channels; ++i) n += f.bits[i] / 8;
  return n;
}

// The driver's CUarray_format (the raw value) with a channel count: the same formats under other names.
// False for a format that names none, one of the video formats, a channel count the format does not have
// (a format fixes it: BC1, BC2, BC3, BC7 and 10:10:10:2 have 4, BC4 1, BC5 2, BC6H 3, the normalized
// 8- and 16-bit ones 1, 2 or 4 by name; the plain integer and float formats 1, 2 or 4) -- the card answers
// CUDA_ERROR_INVALID_VALUE for each (measured on an RTX 3060, driver 13.2).
inline bool texture_format_from_driver(unsigned code, unsigned channels, TexFormat* out) {
  const int n = static_cast<int>(channels);
  auto widths = [&](int bits, int& x, int& y, int& z, int& w) {
    x = bits;
    y = n >= 2 ? bits : 0;
    z = n >= 3 ? bits : 0;
    w = n >= 4 ? bits : 0;
  };
  int x = 0, y = 0, z = 0, w = 0, kind = -1;
  switch (code) {
    case 0x01: kind = 1; widths(8, x, y, z, w); break;      // unsigned int 8, 16, 32
    case 0x02: kind = 1; widths(16, x, y, z, w); break;
    case 0x03: kind = 1; widths(32, x, y, z, w); break;
    case 0x08: kind = 0; widths(8, x, y, z, w); break;      // signed int 8, 16, 32
    case 0x09: kind = 0; widths(16, x, y, z, w); break;
    case 0x0a: kind = 0; widths(32, x, y, z, w); break;
    case 0x10: kind = 2; widths(16, x, y, z, w); break;     // half, float
    case 0x20: kind = 2; widths(32, x, y, z, w); break;
    case 0xc0: case 0xc1: case 0xc2:                         // unsigned normalized 8-bit, 1, 2, 4 channels
      kind = 5 + static_cast<int>(code - 0xc0); widths(8, x, y, z, w); if (n != (code == 0xc0 ? 1 : code == 0xc1 ? 2 : 4)) return false; break;
    case 0xc3: case 0xc4: case 0xc5:                         // 16-bit
      kind = 8 + static_cast<int>(code - 0xc3); widths(16, x, y, z, w); if (n != (code == 0xc3 ? 1 : code == 0xc4 ? 2 : 4)) return false; break;
    case 0xc6: case 0xc7: case 0xc8:                         // signed normalized 8-bit
      kind = 11 + static_cast<int>(code - 0xc6); widths(8, x, y, z, w); if (n != (code == 0xc6 ? 1 : code == 0xc7 ? 2 : 4)) return false; break;
    case 0xc9: case 0xca: case 0xcb:                         // 16-bit
      kind = 14 + static_cast<int>(code - 0xc9); widths(16, x, y, z, w); if (n != (code == 0xc9 ? 1 : code == 0xca ? 2 : 4)) return false; break;
    case 0x50:                                               // 10:10:10:2
      if (n != 4) return false;
      kind = 31; x = y = z = 10; w = 2; break;
    default:
      if (code >= 0x91 && code <= 0x9e) {                    // BC1 .. BC7 in the runtime's order
        kind = 17 + static_cast<int>(code - 0x91);
        const int want = (kind == 23 || kind == 24) ? 1 : (kind == 25 || kind == 26) ? 2 : (kind == 27 || kind == 28) ? 3 : 4;
        if (n != want) return false;
        widths(want == 3 ? 16 : 8, x, y, z, w);
        break;
      }
      return false;
  }
  if (n < 1 || n > 4) return false;
  return texture_format_from_runtime(kind, x, y, z, w, out);
}

// What a view may call a texel: the resource view formats (cudaResViewFormat* and CU_RES_VIEW_FORMAT_*, the same
// numbers) as storage. False for a value that is none.
inline bool view_format(int v, TexFormat* f) {
  using vgpu::exec::BlockFormat;
  using vgpu::exec::ChannelKind;
  *f = TexFormat{};
  auto plain = [&](ChannelKind k, uint32_t bits, uint32_t channels) {
    f->kind = k;
    f->channels = channels;
    for (uint32_t i = 0; i < channels; ++i) f->bits[i] = bits;
    f->texel_bytes = bits * channels / 8;
    f->integer = k != ChannelKind::Float;
    f->is_int32 = bits == 32 && k != ChannelKind::Float;
    f->elem_only = k == ChannelKind::Float;
    return true;
  };
  static const uint32_t kCh[3] = {1, 2, 4};
  if (v >= 0x01 && v <= 0x03) return plain(ChannelKind::Unsigned, 8, kCh[v - 1]);
  if (v >= 0x04 && v <= 0x06) return plain(ChannelKind::Signed, 8, kCh[v - 4]);
  if (v >= 0x07 && v <= 0x09) return plain(ChannelKind::Unsigned, 16, kCh[v - 7]);
  if (v >= 0x0a && v <= 0x0c) return plain(ChannelKind::Signed, 16, kCh[v - 0x0a]);
  if (v >= 0x0d && v <= 0x0f) return plain(ChannelKind::Unsigned, 32, kCh[v - 0x0d]);
  if (v >= 0x10 && v <= 0x12) return plain(ChannelKind::Signed, 32, kCh[v - 0x10]);
  if (v >= 0x13 && v <= 0x15) return plain(ChannelKind::Float, 16, kCh[v - 0x13]);
  if (v >= 0x16 && v <= 0x18) return plain(ChannelKind::Float, 32, kCh[v - 0x16]);
  // The block-compressed views have the descriptors of the block-compressed channel kinds.
  static const struct { int view, kind, x, y, z, w; } kBlock[] = {
      {0x19, 17, 8, 8, 8, 8}, {0x1a, 19, 8, 8, 8, 8}, {0x1b, 21, 8, 8, 8, 8}, {0x1c, 23, 8, 0, 0, 0},
      {0x1d, 24, 8, 0, 0, 0}, {0x1e, 25, 8, 8, 0, 0}, {0x1f, 26, 8, 8, 0, 0}, {0x20, 27, 16, 16, 16, 0},
      {0x21, 28, 16, 16, 16, 0}, {0x22, 29, 8, 8, 8, 8}};
  for (const auto& e : kBlock)
    if (e.view == v) return texture_format_from_runtime(e.kind, e.x, e.y, e.z, e.w, f);
  return false;
}

// A resource view reinterprets the texels of what it views: the same storage, named as another format of the
// same size. Measured on an RTX 3060: the format is a real view format (a zero descriptor, format none, is
// invalid), its texel is as many bytes as the resource's, and the extent given is the resource's own -- for a
// block-compressed view of a 64- or 128-bit-texel array, four times it in each of width and height, the array
// holding blocks. A first mipmap level, or a layer range, is not modelled. Applies the view to the texture
// descriptor `d` (made from the resource, whose format is `format`) and returns 0, or the CUDA error number
// (1 invalid value, 801 not supported; the runtime's and the driver's numbers agree).
inline int apply_view_format(int view, uint64_t vwidth, uint64_t vheight, uint64_t vdepth, unsigned first_mip,
                             unsigned last_mip, unsigned first_layer, unsigned last_layer, bool array_resource,
                             vgpu::exec::TextureDesc* d, TexFormat* format) {
  TexFormat vf;
  if (!view_format(view, &vf)) return 1;
  const uint64_t storage_texel = format->texel_bytes;
  const uint64_t w = d->width, h = d->height;
  if (array_resource && (d->mip_levels || d->layers || d->cubemap)) return 801;   // a view's level and layer range
  if (vf.texel_bytes != storage_texel) return 1;
  if (first_mip != 0 || last_mip != 0 || first_layer != 0 || last_layer != 0) return 1;
  uint64_t want_w = w, want_h = h;
  const bool from_block = format->block != vgpu::exec::BlockFormat::None;
  if (vf.block != vgpu::exec::BlockFormat::None && !from_block) {
    want_w = w * 4;
    want_h = h ? h * 4 : 0;
  } else if (from_block && vf.block == vgpu::exec::BlockFormat::None) {
    want_w = (w + 3) / 4;
    want_h = h ? (h + 3) / 4 : 0;
  }
  if (vwidth != want_w || vheight != want_h || vdepth != d->depth) return 1;
  // The view's format over the same bytes.
  const uint64_t row_bytes = d->pitch_bytes ? d->pitch_bytes : w * storage_texel;
  d->block = vf.block;
  d->packed_1010102 = vf.packed_1010102;
  d->kind = vf.kind;
  d->channels = vf.channels;
  for (int i = 0; i < 4; ++i) d->channel_bits[i] = vf.bits[i];
  d->texel_bytes = sampled_texel_bytes(vf);
  d->width = static_cast<uint32_t>(vwidth);
  d->height = static_cast<uint32_t>(vheight);
  d->pitch_bytes = static_cast<uint32_t>(row_bytes);
  *format = vf;
  return 0;
}

}  // namespace vgpu::cuda
