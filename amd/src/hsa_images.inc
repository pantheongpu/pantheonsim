// HSA's images extension: images and samplers on the Radeon agents, whose
// kernels read them with the image instructions. Included by hsa_api.cpp.
//
// An image's handle is the address of its resource -- the image resource
// (T#), or a buffer resource (V#) for a 1D buffer image, then the channel
// type, order and width in words 8-10 -- in host memory every device
// reaches, as ROCm's image runtime keeps it (a kernel is given the handle;
// ROCm's HIP copies the words into its texture objects). A sampler's is its
// four-word resource's. The texels are the caller's memory, laid out
// linearly: rows one after another, as wide as the image or as the pitch a
// linear layout names, slices one after another. Fields, swizzles and
// limits are ROCR-Runtime's (image_manager_*.cpp, image_lut_kv.cpp,
// resource_*.h); vgpu/amd_image.hpp holds the layouts.

namespace {
namespace im = vgpu::amd::image;

im::Gen agent_gen(int gpu) {
  const std::string& a = shared::profile(gpu).gcn_arch;
  return a.rfind("gfx10", 0) == 0 ? im::Gen::Gfx10 : a.rfind("gfx11", 0) == 0 ? im::Gen::Gfx11 : im::Gen::Gfx12;
}

enum Geometry : uint32_t { k1D = 0, k2D = 1, k3D = 2, k1DA = 3, k2DA = 4, k1DB = 5, k2DDepth = 6, k2DADepth = 7 };

// The texel format a channel type and order make, with where each of the
// four results comes from; false for a combination this does not model
// (the packed 5-, 6- and 10-bit types, 24-bit depth, sRGB).
bool texel_format(const hsa_ext_image_format_t& f, im::Format* out, im::Sel sel[4]) {
  using S = im::Sel;
  static const S kSwizzle[20][4] = {
      {S::Zero, S::Zero, S::Zero, S::X}, {S::X, S::Zero, S::Zero, S::One}, {S::X, S::Zero, S::Zero, S::One},
      {S::X, S::Y, S::Zero, S::One},     {S::X, S::Y, S::Zero, S::One},    {S::X, S::Zero, S::Zero, S::Y},
      {S::Z, S::Y, S::X, S::One},        {S::Z, S::Y, S::X, S::One},       {S::X, S::Y, S::Z, S::W},
      {S::Z, S::Y, S::X, S::W},          {S::Y, S::Z, S::W, S::X},         {S::Y, S::X, S::W, S::Z},
      {S::X, S::Y, S::Z, S::One},        {S::X, S::Y, S::Z, S::One},       {S::X, S::Y, S::Z, S::W},
      {S::Z, S::Y, S::X, S::W},          {S::X, S::X, S::X, S::X},         {S::X, S::X, S::X, S::One},
      {S::X, S::Zero, S::Zero, S::Zero}, {S::Y, S::Zero, S::Zero, S::Zero},
  };
  // Channels by order: A R RX RG RGX RA RGB RGBX RGBA BGRA ARGB ABGR, then
  // the sRGB orders (of which SRGBA, in 8-bit unorm, is modelled),
  // INTENSITY, LUMINANCE, DEPTH.
  static const int kChannels[20] = {1, 1, 1, 2, 2, 2, 0, 0, 4, 4, 4, 4, 0, 0, 4, 0, 1, 1, 1, 0};
  if (f.channel_order >= 20 || !kChannels[f.channel_order]) return false;
  const int n = kChannels[f.channel_order];
  uint32_t bits = 0;
  im::Num num;
  switch (f.channel_type) {
    case 0: bits = 8, num = im::Num::Snorm; break;
    case 1: bits = 16, num = im::Num::Snorm; break;
    case 2: bits = 8, num = im::Num::Unorm; break;
    case 3: bits = 16, num = im::Num::Unorm; break;
    case 8: bits = 8, num = im::Num::Sint; break;
    case 9: bits = 16, num = im::Num::Sint; break;
    case 10: bits = 32, num = im::Num::Sint; break;
    case 11: bits = 8, num = im::Num::Uint; break;
    case 12: bits = 16, num = im::Num::Uint; break;
    case 13: bits = 32, num = im::Num::Uint; break;
    case 14: bits = 16, num = im::Num::Float; break;
    case 15: bits = 32, num = im::Num::Float; break;
    default: return false;
  }
  if (f.channel_order == 18 && f.channel_type != 3 && f.channel_type != 15) return false;   // depth: 16-bit unorm or float
  if (f.channel_order == 14) {   // SRGBA
    if (f.channel_type != 2) return false;
    num = im::Num::Srgb;
  }
  static const im::Data k8[] = {im::Data::D8, im::Data::D8_8, im::Data::Invalid, im::Data::D8_8_8_8};
  static const im::Data k16[] = {im::Data::D16, im::Data::D16_16, im::Data::Invalid, im::Data::D16_16_16_16};
  static const im::Data k32[] = {im::Data::D32, im::Data::D32_32, im::Data::Invalid, im::Data::D32_32_32_32};
  out->data = bits == 8 ? k8[n - 1] : bits == 16 ? k16[n - 1] : k32[n - 1];
  out->num = num;
  for (int k = 0; k < 4; ++k) sel[k] = kSwizzle[f.channel_order][k];
  return true;
}

struct HsaImage {
  int gpu = 0;
  uint32_t* words = nullptr;   // the resource and words 8-10, in host memory the devices reach
  uint64_t data = 0;           // the texels
  hsa_ext_image_descriptor_t desc{};
  uint32_t element = 0;
  size_t row_pitch = 0, slice_pitch = 0;
};
std::map<uint64_t, HsaImage> g_images;       // under g_mutex, by handle
std::map<uint64_t, uint32_t*> g_samplers;    // under g_mutex, by handle
constexpr size_t kImageWords = 12;
constexpr size_t kImageAlignment = 256;

uint32_t* host_words(size_t n) {
  void* p = std::aligned_alloc(64, 64 * ((n * 4 + 63) / 64));
  if (!p) return nullptr;
  std::memset(p, 0, 64 * ((n * 4 + 63) / 64));
  shared::map_host(p, 64 * ((n * 4 + 63) / 64));
  return static_cast<uint32_t*>(p);
}
void free_words(uint32_t* p) {
  shared::unmap_host(p);
  std::free(p);
}

// The rows and slices of an image's texels, in a linear layout: `pitch`
// bytes to a row (0: the width), and slices exactly as many rows apart as
// the image has.
hsa_status_t image_layout(const hsa_ext_image_descriptor_t& d, uint32_t element, size_t pitch, size_t slice_pitch,
                    size_t* row, size_t* slice, size_t* bytes) {
  const size_t rows = d.geometry == k1DA ? std::max<size_t>(d.array_size, 1)
                      : d.geometry == k1D || d.geometry == k1DB ? 1
                                                                 : std::max<size_t>(d.height, 1);
  const size_t slices = d.geometry == k3D ? std::max<size_t>(d.depth, 1)
                        : d.geometry == k2DA || d.geometry == k2DADepth ? std::max<size_t>(d.array_size, 1)
                                                                        : 1;
  *row = pitch ? pitch : d.width * element;
  if (*row < d.width * element || *row % element) return static_cast<hsa_status_t>(HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED);
  *slice = *row * rows;
  if (slice_pitch && slice_pitch != *slice && slices > 1)
    return static_cast<hsa_status_t>(HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED);
  *bytes = *slice * slices;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t check_descriptor(int gpu, const hsa_ext_image_descriptor_t* d, im::Format* f, im::Sel sel[4]) {
  if (gpu < 0) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!d || d->geometry > k2DADepth) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (!agent_has_images(gpu)) return static_cast<hsa_status_t>(HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED);
  if (!texel_format(d->format, f, sel)) return static_cast<hsa_status_t>(HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED);
  const uint32_t* m = kImageMaxDims[d->geometry];
  if (!d->width || d->width > m[0] || d->height > m[1] || d->depth > m[2] || d->array_size > m[3])
    return static_cast<hsa_status_t>(HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t create_image(hsa_agent_t agent, const hsa_ext_image_descriptor_t* d, const void* data, size_t pitch,
                          size_t slice_pitch, hsa_ext_image_t* image) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const int gpu = gpu_of(agent);
  im::Format f;
  im::Sel sel[4];
  if (const hsa_status_t s = check_descriptor(gpu, d, &f, sel); s != HSA_STATUS_SUCCESS) return s;
  if (!data || !image) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  HsaImage img;
  img.gpu = gpu;
  img.data = reinterpret_cast<uint64_t>(data);
  img.desc = *d;
  img.element = im::texel_bytes(f);
  size_t bytes = 0;
  if (const hsa_status_t s = image_layout(*d, img.element, pitch, slice_pitch, &img.row_pitch, &img.slice_pitch, &bytes);
      s != HSA_STATUS_SUCCESS)
    return s;
  img.words = host_words(kImageWords);
  if (!img.words) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  const im::Gen g = agent_gen(gpu);
  if (d->geometry == k1DB) {
    im::encode_buffer(img.data, img.element, static_cast<uint32_t>(d->width), f, g, img.words);
    img.words[3] &= ~0xFFFu;
    for (int k = 0; k < 4; ++k) img.words[3] |= (static_cast<uint32_t>(sel[k]) & 7) << (3 * k);
  } else {
    im::Image i;
    i.base = img.data;
    i.width = static_cast<uint32_t>(d->width);
    i.height = static_cast<uint32_t>(d->geometry == k1D || d->geometry == k1DA ? 1 : std::max<size_t>(d->height, 1));
    i.depth = static_cast<uint32_t>(d->geometry == k3D ? std::max<size_t>(d->depth, 1) : std::max<size_t>(d->array_size, 1));
    i.pitch = static_cast<uint32_t>(img.row_pitch / img.element);
    i.format = f;
    for (int k = 0; k < 4; ++k) i.sel[k] = sel[k];
    static const im::Type kTypes[8] = {im::Type::Img1D,      im::Type::Img2D,      im::Type::Img3D,
                                       im::Type::Img1DArray, im::Type::Img2DArray, im::Type::Img1D,
                                       im::Type::Img2D,      im::Type::Img2DArray};
    i.type = kTypes[d->geometry];
    im::encode(i, g, img.words);
  }
  img.words[8] = d->format.channel_type;
  img.words[9] = d->format.channel_order;
  img.words[10] = static_cast<uint32_t>(d->width);
  image->handle = reinterpret_cast<uint64_t>(img.words);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_images[image->handle] = img;
  return HSA_STATUS_SUCCESS;
}

bool find_image(hsa_ext_image_t image, HsaImage* out) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_images.find(image.handle);
  if (it == g_images.end()) return false;
  *out = it->second;
  return true;
}

// Copies a region between an image and memory laid out with the given
// pitches, a row at a time.
hsa_status_t move_region(const HsaImage& img, void* memory, size_t row_pitch, size_t slice_pitch,
                         const hsa_ext_image_region_t* r, bool into_image) {
  if (!memory || !r) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const size_t width = size_t{r->range.x} * img.element;
  const size_t rows = std::max<uint32_t>(r->range.y, 1), slices = std::max<uint32_t>(r->range.z, 1);
  if (!row_pitch) row_pitch = width;
  if (!slice_pitch) slice_pitch = row_pitch * rows;
  auto* mem = static_cast<uint8_t*>(memory);
  for (size_t z = 0; z < slices; ++z)
    for (size_t y = 0; y < rows; ++y) {
      const uint64_t at = img.data + (r->offset.z + z) * img.slice_pitch + (r->offset.y + y) * img.row_pitch +
                          size_t{r->offset.x} * img.element;
      uint8_t* host = mem + z * slice_pitch + y * row_pitch;
      std::string why;
      const bool ok = into_image ? shared::copy(reinterpret_cast<void*>(at), host, width, &why)
                                 : shared::copy(host, reinterpret_cast<const void*>(at), width, &why);
      if (!ok) return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, why);
    }
  return HSA_STATUS_SUCCESS;
}
}  // namespace

hsa_status_t hsa_ext_image_get_capability(hsa_agent_t agent, uint32_t geometry, const hsa_ext_image_format_t* format,
                                          uint32_t* capability) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const int gpu = gpu_of(agent);
  if (gpu < 0) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!format || !capability || geometry > k2DADepth) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  im::Format f;
  im::Sel sel[4];
  // Read, write and read-write, and the layout the same whichever.
  *capability = agent_has_images(gpu) && texel_format(*format, &f, sel) ? 0x1 | 0x2 | 0x4 | 0x10 : 0;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_ext_image_data_get_info(hsa_agent_t agent, const hsa_ext_image_descriptor_t* d,
                                         hsa_access_permission_t, hsa_ext_image_data_info_t* info) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  im::Format f;
  im::Sel sel[4];
  if (const hsa_status_t s = check_descriptor(gpu_of(agent), d, &f, sel); s != HSA_STATUS_SUCCESS) return s;
  if (!info) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  size_t row = 0, slice = 0, bytes = 0;
  if (const hsa_status_t s = image_layout(*d, im::texel_bytes(f), 0, 0, &row, &slice, &bytes); s != HSA_STATUS_SUCCESS)
    return s;
  info->size = bytes;
  info->alignment = kImageAlignment;
  return HSA_STATUS_SUCCESS;
}
// With a layout: a linear one's row pitch is the caller's (a multiple of the
// texel), an opaque one's the width.
hsa_status_t hsa_ext_image_data_get_info_with_layout(hsa_agent_t agent, const hsa_ext_image_descriptor_t* d,
                                                     hsa_access_permission_t, hsa_ext_image_data_layout_t layout_kind, size_t row_pitch, size_t slice_pitch,
                                                     hsa_ext_image_data_info_t* info) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  im::Format f;
  im::Sel sel[4];
  if (const hsa_status_t s = check_descriptor(gpu_of(agent), d, &f, sel); s != HSA_STATUS_SUCCESS) return s;
  if (!info) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  size_t row = 0, slice = 0, bytes = 0;
  if (const hsa_status_t s = image_layout(*d, im::texel_bytes(f), layout_kind == 1 ? row_pitch : 0,
                                          layout_kind == 1 ? slice_pitch : 0, &row, &slice, &bytes);
      s != HSA_STATUS_SUCCESS)
    return s;
  info->size = bytes;
  info->alignment = kImageAlignment;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_ext_image_get_capability_with_layout(hsa_agent_t agent, uint32_t geometry,
                                                      const hsa_ext_image_format_t* format,
                                                      hsa_ext_image_data_layout_t, uint32_t* capability) {
  return hsa_ext_image_get_capability(agent, geometry, format, capability);
}
hsa_status_t hsa_ext_image_create(hsa_agent_t agent, const hsa_ext_image_descriptor_t* d, const void* data,
                                  hsa_access_permission_t, hsa_ext_image_t* image) {
  return create_image(agent, d, data, 0, 0, image);
}
// A linear layout's pitches are the caller's; an opaque one is the layout
// hsa_ext_image_create gives.
hsa_status_t hsa_ext_image_create_with_layout(hsa_agent_t agent, const hsa_ext_image_descriptor_t* d,
                                              const void* data, hsa_access_permission_t,
                                              hsa_ext_image_data_layout_t layout_kind, size_t row_pitch,
                                              size_t slice_pitch, hsa_ext_image_t* image) {
  return layout_kind == 1 ? create_image(agent, d, data, row_pitch, slice_pitch, image)
                          : create_image(agent, d, data, 0, 0, image);
}
hsa_status_t hsa_ext_image_destroy(hsa_agent_t agent, hsa_ext_image_t image) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  uint32_t* words = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_images.find(image.handle);
    if (it == g_images.end()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    words = it->second.words;
    g_images.erase(it);
  }
  free_words(words);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_ext_image_import(hsa_agent_t agent, const void* src, size_t row_pitch, size_t slice_pitch,
                                  hsa_ext_image_t image, const hsa_ext_image_region_t* region) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  HsaImage img;
  if (!find_image(image, &img)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return move_region(img, const_cast<void*>(src), row_pitch, slice_pitch, region, true);
}
hsa_status_t hsa_ext_image_export(hsa_agent_t agent, hsa_ext_image_t image, void* dst, size_t row_pitch,
                                  size_t slice_pitch, const hsa_ext_image_region_t* region) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  HsaImage img;
  if (!find_image(image, &img)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return move_region(img, dst, row_pitch, slice_pitch, region, false);
}
// One image's region into another's, both in the same format.
hsa_status_t hsa_ext_image_copy(hsa_agent_t agent, hsa_ext_image_t src, const hsa_dim3_t* src_offset,
                                hsa_ext_image_t dst, const hsa_dim3_t* dst_offset, const hsa_dim3_t* range) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  HsaImage s, d;
  if (!find_image(src, &s) || !find_image(dst, &d) || !src_offset || !dst_offset || !range)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (s.element != d.element) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::vector<uint8_t> row(size_t{range->x} * s.element);
  for (uint32_t z = 0; z < std::max<uint32_t>(range->z, 1); ++z)
    for (uint32_t y = 0; y < std::max<uint32_t>(range->y, 1); ++y) {
      const hsa_ext_image_region_t from{{src_offset->x, src_offset->y + y, src_offset->z + z}, {range->x, 1, 1}};
      const hsa_ext_image_region_t to{{dst_offset->x, dst_offset->y + y, dst_offset->z + z}, {range->x, 1, 1}};
      if (const hsa_status_t st = move_region(s, row.data(), 0, 0, &from, false); st != HSA_STATUS_SUCCESS) return st;
      if (const hsa_status_t st = move_region(d, row.data(), 0, 0, &to, true); st != HSA_STATUS_SUCCESS) return st;
    }
  return HSA_STATUS_SUCCESS;
}
// Every texel of a region set to one value, given as four channels (floats
// or integers, as the format reads).
hsa_status_t hsa_ext_image_clear(hsa_agent_t agent, hsa_ext_image_t image, const void* data,
                                 const hsa_ext_image_region_t* region) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  HsaImage img;
  if (!find_image(image, &img) || !data || !region) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  im::Format f;
  im::Sel sel[4];
  texel_format(img.desc.format, &f, sel);
  uint8_t texel[16] = {};
  im::write_texel(f, static_cast<const uint32_t*>(data), texel);
  std::vector<uint8_t> row(size_t{region->range.x} * img.element);
  for (size_t x = 0; x < region->range.x; ++x) std::memcpy(&row[x * img.element], texel, img.element);
  for (uint32_t z = 0; z < std::max<uint32_t>(region->range.z, 1); ++z)
    for (uint32_t y = 0; y < std::max<uint32_t>(region->range.y, 1); ++y) {
      const hsa_ext_image_region_t r{{region->offset.x, region->offset.y + y, region->offset.z + z},
                                     {region->range.x, 1, 1}};
      if (const hsa_status_t st = move_region(img, row.data(), 0, 0, &r, true); st != HSA_STATUS_SUCCESS) return st;
    }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_ext_sampler_create_v2(hsa_agent_t agent, const hsa_ext_sampler_descriptor_v2_t* d,
                                       hsa_ext_sampler_t* sampler) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const int gpu = gpu_of(agent);
  if (gpu < 0) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!d || !sampler) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (!agent_has_images(gpu)) return static_cast<hsa_status_t>(HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED);
  if (d->filter_mode > 1 || d->coordinate_mode > 1) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  im::Sampler s;
  for (int k = 0; k < 3; ++k) {
    switch (d->address_modes[k]) {
      case 0:
      case 1: s.clamp[k] = im::Clamp::ClampLastTexel; break;
      case 2: s.clamp[k] = im::Clamp::ClampBorder; break;
      case 3: s.clamp[k] = im::Clamp::Wrap; break;
      case 4: s.clamp[k] = im::Clamp::Mirror; break;
      default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }
  s.unnormalized = d->coordinate_mode == 0;
  s.mag_linear = s.min_linear = d->filter_mode == 1;
  uint32_t* words = host_words(4);
  if (!words) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  im::encode(s, agent_gen(gpu), words);
  sampler->handle = reinterpret_cast<uint64_t>(words);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_samplers[sampler->handle] = words;
  return HSA_STATUS_SUCCESS;
}
// The first version's descriptor is the second's with one address mode
// for every axis.
hsa_status_t hsa_ext_sampler_create(hsa_agent_t agent, const uint32_t* d, hsa_ext_sampler_t* sampler) {
  if (!d) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const hsa_ext_sampler_descriptor_v2_t v2{d[0], d[1], {d[2], d[2], d[2]}};
  return hsa_ext_sampler_create_v2(agent, &v2, sampler);
}
hsa_status_t hsa_ext_sampler_destroy(hsa_agent_t agent, hsa_ext_sampler_t sampler) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  uint32_t* words = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_samplers.find(sampler.handle);
    if (it == g_samplers.end()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    words = it->second;
    g_samplers.erase(it);
  }
  free_words(words);
  return HSA_STATUS_SUCCESS;
}
