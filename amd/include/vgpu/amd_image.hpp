// RDNA's image and sampler resources (T# and S#) and a formatted buffer's
// resource (V#): what the runtime writes for a texture, a surface or a
// sampler, and what the executor's image and formatted buffer instructions
// read. The fields sit where AMD's RDNA ISA manuals put them wherever a
// kernel reads them itself (HIP's texture functions read the height from the
// image resource's word 2 and two of the sampler's bits). A format is named
// here as GCN names it, a data format (how the bits are laid out) and a
// number format (what they mean); the resources hold RDNA's single code.
#pragma once

#include <cstdint>

namespace vgpu::amd::image {

// Where the resources' fields sit differs by generation: RDNA2 (gfx10.3),
// RDNA3 (gfx11) and RDNA4 (gfx12).
enum class Gen : uint8_t { Gfx10, Gfx11, Gfx12 };

// GCN's data formats (BUF_DATA_FORMAT_*): the channels and their widths.
enum class Data : uint8_t {
  Invalid = 0, D8 = 1, D16 = 2, D8_8 = 3, D32 = 4, D16_16 = 5, D10_11_11 = 6, D11_11_10 = 7, D10_10_10_2 = 8,
  D2_10_10_10 = 9, D8_8_8_8 = 10, D32_32 = 11, D16_16_16_16 = 12, D32_32_32 = 13, D32_32_32_32 = 14,
};
// And the number formats (BUF_NUM_FORMAT_*): what a channel's bits mean.
// Srgb: 8-bit unorm whose color channels (not alpha) are sRGB-encoded, read
// back as linear values.
enum class Num : uint8_t { Unorm = 0, Snorm = 1, Uscaled = 2, Sscaled = 3, Uint = 4, Sint = 5, Float = 7, Srgb = 9 };

struct Format {
  Data data = Data::Invalid;
  Num num = Num::Unorm;
};
// A texel's size in bytes, and how many channels it has (0 for a format this
// does not read or write).
uint32_t texel_bytes(Format f);
uint32_t channels(Format f);
// RDNA's single code for a format (BUF_FMT_*, IMG_FMT_*), which gfx11
// renumbered (gfx12 keeps gfx11's); 0 and Format{} where there is none.
uint32_t code(Format f, Gen g);
Format from_code(uint32_t code, Gen g);
// Whether a channel reads back as an integer (Uint, Sint) rather than a float.
bool integer(Format f);

// A texel's four channels from its bytes: floats' bits, or the integers, for
// the channels it has; the rest 0, alpha 1 (1.0f, or 1 for an integer format).
void read_texel(Format f, const uint8_t* bytes, uint32_t out[4]);
// The bytes of a texel from four channels given the same way.
void write_texel(Format f, const uint32_t in[4], uint8_t* bytes);

// Where each of the four results comes from (DST_SEL_*): 0 or 1, or a channel.
enum class Sel : uint8_t { Zero = 0, One = 1, X = 4, Y = 5, Z = 6, W = 7 };
// The resource's shape (SQ_RSRC_IMG_*), as its TYPE field holds it.
enum class Type : uint8_t { Img1D = 8, Img2D = 9, Img3D = 10, Cube = 11, Img1DArray = 12, Img2DArray = 13 };

// An image resource (T#), eight words. The texels are laid out linearly, row
// by row: level 0 at the base, each later level after the one before it,
// rows `pitch` texels apart at level 0 and as wide as the level after it.
struct Image {
  uint64_t base = 0;
  uint32_t width = 1, height = 1, depth = 1;   // depth: a 3D image's depth, an array's layers
  uint32_t pitch = 0;                          // texels from one row to the next at level 0 (0: the width);
                                               // a 1D or 2D image's only
  Format format;
  Sel sel[4] = {Sel::X, Sel::Y, Sel::Z, Sel::W};
  Type type = Type::Img2D;
  uint32_t base_level = 0, last_level = 0;
};
void encode(const Image& i, Gen g, uint32_t words[8]);
Image decode_image(const uint32_t words[8], Gen g);

// Where level `level`'s texel (x, y, z) is, from the image's base; and each
// level's size and pitch.
uint32_t level_width(const Image& i, uint32_t level);
uint32_t level_height(const Image& i, uint32_t level);
uint32_t level_depth(const Image& i, uint32_t level);   // an array's layers at every level
uint64_t texel_offset(const Image& i, uint32_t level, uint32_t x, uint32_t y, uint32_t z);
// How many bytes the whole image takes, every level.
uint64_t image_bytes(const Image& i);

// What a coordinate past an edge does (SQ_TEX_CLAMP_*).
enum class Clamp : uint8_t {
  Wrap = 0, Mirror = 1, ClampLastTexel = 2, MirrorOnceLastTexel = 3, ClampHalfBorder = 4,
  MirrorOnceHalfBorder = 5, ClampBorder = 6, MirrorOnceBorder = 7,
};
// A sampler resource (S#), four words.
struct Sampler {
  Clamp clamp[3] = {Clamp::ClampLastTexel, Clamp::ClampLastTexel, Clamp::ClampLastTexel};
  bool unnormalized = false;    // coordinates in texels rather than 0..1 (word 0 bit 15)
  bool mag_linear = false;      // XY_MAG_FILTER bilinear (word 2 bit 20)
  bool min_linear = false;      // XY_MIN_FILTER bilinear
  uint8_t mip_filter = 0;       // 0 none (the base level), 1 point, 2 linear
  uint8_t border = 0;           // 0 transparent black, 1 opaque black, 2 opaque white
  float min_lod = 0, max_lod = 15.99f, lod_bias = 0;
};
void encode(const Sampler& s, Gen g, uint32_t words[4]);
Sampler decode_sampler(const uint32_t words[4], Gen g);

// A formatted buffer resource (V#), four words: `records` elements `stride`
// bytes apart, each of `format`.
void encode_buffer(uint64_t base, uint32_t stride, uint32_t records, Format format, Gen g, uint32_t words[4]);
Format buffer_format(const uint32_t words[4], Gen g);
Sel buffer_sel(const uint32_t words[4], int channel);

}  // namespace vgpu::amd::image
