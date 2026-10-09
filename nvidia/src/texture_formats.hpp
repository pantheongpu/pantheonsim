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

}  // namespace vgpu::cuda
