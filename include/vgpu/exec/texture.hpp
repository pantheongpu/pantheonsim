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
  TexFilter mip_filter = TexFilter::Point;
  int32_t mip_bias = 0, mip_min = 0, mip_max = 0;
  uint32_t pitch_bytes = 0; // distance between rows; width*texel_bytes if dense
  uint32_t channels = 1;    // 1..4
  uint32_t channel_bits[4] = {32, 0, 0, 0};
  uint32_t texel_bytes = 4;
  ChannelKind kind = ChannelKind::Float;
  TexKind object = TexKind::Texture;
  // Coordinates in [0,1) rather than [0,size). Independent of the filter.
  bool normalized_coords = false;
  // Integer channels delivered as floats scaled into [0,1] or [-1,1], which is
  // what cudaReadModeNormalizedFloat asks for.
  bool read_as_normalized_float = false;
  TexFilter filter = TexFilter::Point;
  TexAddress address[3] = {TexAddress::Clamp, TexAddress::Clamp, TexAddress::Clamp};
  // True when the backing store came from cudaMallocArray rather than being a
  // view over linear device memory. The distinction matters for diagnostics
  // only: both are ordinary memory here.
  bool from_array = false;
};

// Handle -> descriptor, owned by the device and consulted during a launch.
using TextureTable = std::map<uint64_t, TextureDesc>;

}  // namespace vgpu::exec
