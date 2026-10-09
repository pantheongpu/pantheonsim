// Texture and surface objects.
//
// A texture object is a 64-bit handle the host creates and the kernel receives
// as an ordinary parameter. The PTX that reads it -- tex, suld, sust -- carries
// only the handle and the coordinates, so everything else about the fetch (what
// memory backs it, how wide a texel is, what happens off the edge) has to come
// from a table the launch can consult. This is that table.
//
// Nothing here models a texture *cache*: VirtualGPU has no memory-hierarchy
// model, and a texture fetch reads the same bytes an ordinary load would. What
// it does model is the addressing and the format conversion, which are the
// parts that change results rather than timing.
#pragma once

#include <cstdint>
#include <map>
#include <optional>

#include "vgpu/exec/block_compression.hpp"

namespace vgpu {
class MemoryManager;
}

namespace vgpu::exec {

// How a channel's bits are read. Matches cudaChannelFormatKind for the three
// kinds that have a defined conversion to what a kernel asks for.
enum class ChannelKind : uint8_t { Unsigned, Signed, Float };

// Point takes the nearest texel; Linear blends between them. Only Point is
// implemented -- see the note in interpreter.cpp on why Linear is refused
// rather than approximated.
enum class TexFilter : uint8_t { Point, Linear };

// What happens to a coordinate outside [0, size).
enum class TexAddress : uint8_t { Wrap, Clamp, Mirror, Border };

// Which instruction family a handle is valid for. A surface object and a
// texture object are different things with the same shape of handle, and using
// one where the other belongs is a mistake worth naming.
enum class TexKind : uint8_t { Texture, Surface };

struct TextureDesc {
  uint64_t base = 0;        // device address of the backing memory
  uint32_t width = 0;       // in texels
  uint32_t height = 0;      // 0 for a 1D object
  uint32_t depth = 0;       // 0 for 1D and 2D, and for layered and cubemap textures
  // A layered texture's layer count (0 when not layered), and whether it is a
  // cubemap. Layers, and a cubemap's six faces per layer, are consecutive
  // slices of width x height texels.
  uint32_t layers = 0;
  bool cubemap = false;
  // A mipmapped texture: level l is max(1, size >> l) in each dimension and
  // starts at level_base[l]. The bias and the level clamps are in 1/256ths of
  // a level, truncated toward zero, as the hardware holds them (measured).
  uint32_t mip_levels = 0;   // 0 when not mipmapped
  uint64_t level_base[17] = {};
  // Within one fetch of a mipmapped layered or cubemap texture: the slice
  // (layer x faces + face) to read in whichever level it lands on. Every
  // level of such a texture has all the slices, each of that level's size.
  uint32_t mip_slice = 0;
  TexFilter mip_filter = TexFilter::Point;
  int32_t mip_bias = 0, mip_min = 0, mip_max = 0;
  // cudaTextureDesc::maxAnisotropy as given (0 reads as 1). Only tex.grad looks at it: a fetch with gradients
  // is the one place anisotropic filtering acts, and it is not modelled (see texture_fetch).
  uint32_t max_aniso = 1;
  uint32_t pitch_bytes = 0; // distance between rows; width*texel_bytes if dense
  uint32_t channels = 1;    // 1..4
  uint32_t channel_bits[4] = {32, 0, 0, 0};
  uint32_t texel_bytes = 4;
  // Block-compressed storage: `base` is where the 4 x 4 blocks are (row after row, ceil(width / 4) blocks a
  // row), width and height are in texels, and a texel is decoded as the format describes (channel_bits and
  // kind say what comes out). Not set for the plain formats.
  BlockFormat block = BlockFormat::None;
  // 10:10:10:2 unsigned normalized, packed in one 32-bit word (channel_bits say 10, 10, 10, 2).
  bool packed_1010102 = false;
  ChannelKind kind = ChannelKind::Float;
  TexKind object = TexKind::Texture;
  // Coordinates in [0,1) rather than [0,size). Independent of the filter.
  bool normalized_coords = false;
  // Integer channels delivered as floats scaled into [0,1] or [-1,1], which is
  // what cudaReadModeNormalizedFloat asks for.
  bool read_as_normalized_float = false;
  // cudaTextureDesc::sRGB: an 8-bit unsigned normalized texture's colour
  // channels are decoded from sRGB to linear (see tex_srgb in interpreter.cpp).
  bool srgb = false;
  TexFilter filter = TexFilter::Point;
  // cudaTextureDesc::borderColor, as the float bits the program set: what
  // border addressing returns outside the texture, converted to the
  // texture's format by the rules measured on an RTX 3060 (see
  // tex_border_raw in interpreter.cpp).
  uint32_t border_bits[4] = {0, 0, 0, 0};
  TexAddress address[3] = {TexAddress::Clamp, TexAddress::Clamp, TexAddress::Clamp};
  // True when the backing store came from cudaMallocArray rather than being a
  // view over linear device memory. The distinction matters for diagnostics
  // only: both are ordinary memory here.
  bool from_array = false;
  // Within one fetch only: the texel offsets (tex's operand e, -8..7 per
  // axis, applied to the texel indices before the address mode), as
  // fetch_texel copies them into its working descriptor.
  int32_t fetch_offset[3] = {0, 0, 0};
};

// Handle -> descriptor, owned by the device and consulted during a launch.
using TextureTable = std::map<uint64_t, TextureDesc>;

// ---- sampling, shared by the PTX interpreter and the SASS executor ----
//
// These throw vgpu::Error saying what is wrong with the access; the caller
// adds which instruction and lane.

// The descriptor for a handle, checked to be the kind of object wanted.
const TextureDesc& texture_lookup(const TextureTable* table, uint64_t handle, TexKind want);

// One texture fetch (tex, tld4) by one thread.
struct TexFetch {
  uint32_t dims = 1;           // spatial coordinates: 1, 2 or 3 (3 for a cube's direction)
  bool layered = false;        // a layer index comes with the coordinates
  bool cube = false;
  uint32_t layer = 0;          // unsigned: past the end reads the last layer
  uint32_t coord[3] = {};      // register bits: f32 when float_coords, s32 otherwise
  bool float_coords = true;
  bool float_result = true;    // f32 destination (else s32/u32)
  bool explicit_lod = false;
  double lod = 0;
  // tex.grad: dPdx and dPdy in the texture's coordinate units, one float per spatial coordinate (register bits).
  bool grad = false;
  uint32_t ddx[3] = {}, ddy[3] = {};
  int gather = -1;             // tld4: the component (0..3) gathered, or -1
  int32_t offset[3] = {0, 0, 0};   // the offset operand's texels, already in range
};
// out: the four components (tld4: the four texels' component).
void texture_fetch(const MemoryManager& mem, const TextureDesc& d, const TexFetch& f, uint32_t out[4]);

// One surface access (suld, sust) by one thread; the policy values are
// ptx::SurfaceOob's.
enum : uint8_t { kSurfaceTrap = 0, kSurfaceClamp = 1, kSurfaceZero = 2 };
struct SurfaceAccess {
  uint32_t dims = 1;
  bool layered = false;
  uint32_t layer = 0;
  int64_t x = 0;               // in bytes
  int64_t y = 0, z = 0;        // in rows and slices
  uint32_t bytes = 4;          // the whole access
  uint8_t oob = kSurfaceTrap;
};
// The address the access reads or writes, or none for a .zero access out of
// range.
std::optional<uint64_t> surface_address(const TextureDesc& d, const SurfaceAccess& a);

// sust.p: the texel a formatted store writes, from the values of its R, G, B and A operands (as many
// as the instruction gave: `n`). Each channel of the surface's format takes one: an unsigned channel
// the value clamped to its range, a signed one the value as an s32 clamped to its range, a 32-bit
// float channel the bits as they are and a 16-bit one the f32 rounded toward zero; a channel the
// operands do not reach is written 0 (measured on an RTX 3060, over every CUDA surface format).
// Returns the texel's bytes in out.
uint32_t surface_pack_texel(const TextureDesc& d, const uint32_t* values, uint32_t n, uint8_t out[16]);

}  // namespace vgpu::exec
