// HIP's arrays, texture objects and surface objects. Included by
// hip_api.cpp, whose allocations and copies of every shape they build on.
//
// The MI300 family (gfx942, gfx950) has no texture units: hipcc refuses the
// texture API in its device code (__HIP_NO_IMAGE_SUPPORT), and ROCm's HIP on
// it says image support is 0 and answers every call that would make an
// array, a texture or a surface with hipErrorNotSupported. Those answers,
// found by asking it (ROCm's libamdhip64 on this HSA runtime,
// amd/tests/hipcc/textures.cpp), are kept here for such a device, so a
// program or library that calls them is told what it would be told on the
// card rather than failing to load for want of the symbol.
//
// The Radeon GPUs (RDNA2, RDNA3, RDNA4) have them, and there each call
// answers as ROCm's HIP does (ROCm's clr, hip_memory.cpp, hip_texture.cpp
// and hip_surface.cpp, is the reference for every check and error). An
// array is device memory holding its texels row after row, slice after
// slice (a 1D layered array's layers are its rows), which the copies reach
// as pitched memory. A texture or surface object is a small allocation in
// device memory holding what the kernel reads: an image resource (T#) or,
// for linear memory, a buffer resource (V#), the channel type, order and
// width after it (words 8-10), and a texture's sampler (S#) at word 12 --
// as ROCm's image runtime writes them (vgpu/amd_image.hpp). HIP's texture
// functions read the height and the sampler's filter and normalization bits
// themselves; the texel fetches, filtering and addressing are the
// executor's image instructions.

namespace {
namespace im = vgpu::amd::image;

hipError_t no_images(const char* name) {
  const ApiCall api(name);
  return record(state(), hipErrorNotSupported);
}
hipError_t no_such_array(const char* name) {   // a handle to what cannot exist
  const ApiCall api(name);
  return record(state(), hipErrorInvalidHandle);
}
hipError_t freeing_nothing(const char* name) {
  const ApiCall api(name);
  return record(state(), hipErrorInvalidValue);
}
hipError_t destroying(const char* name, uint64_t object) {   // none is ever made, so only 0 is fine
  const ApiCall api(name);
  return record(state(), object ? hipErrorInvalidValue : hipSuccess);
}

// has_images (hip_api.cpp): whether a device has texture units.
im::Gen image_gen(const vgpu::DeviceProfile& p) {
  return p.gcn_arch.rfind("gfx10", 0) == 0 ? im::Gen::Gfx10
         : p.gcn_arch.rfind("gfx11", 0) == 0 ? im::Gen::Gfx11
                                              : im::Gen::Gfx12;
}
// The current device's, under s.mutex.
bool images_on(State& s) {
  const vgpu::runtime::Device* d = device(s);
  return d && has_images(d->profile());
}
bool images_here() {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return images_on(s);
}

// The limits ROCm's image runtime gives every Radeon target (ROCR-Runtime's
// image_lut_kv.cpp), which HIP's properties report.
constexpr uint32_t kMax1D = 16384, kMax2D = 16384, kMax3D = 16384, kMax3DDepth = 8192, kMaxLayers = 8192;
constexpr uint32_t kImageAlignment = 256;   // an image's base and a pitched row, in bytes

// hipArray_Format.
enum ArrayFormat : int { kU8 = 0x01, kU16 = 0x02, kU32 = 0x03, kS8 = 0x08, kS16 = 0x09, kS32 = 0x0a, kHalf = 0x10, kFloat = 0x20 };
constexpr unsigned kArrayLayered = 0x01, kArraySurfaceLoadStore = 0x02, kArrayCubemap = 0x04, kArrayTextureGather = 0x08;

uint32_t channel_bytes(int format) {
  switch (format) {
    case kU8: case kS8: return 1;
    case kU16: case kS16: case kHalf: return 2;
    case kU32: case kS32: case kFloat: return 4;
    default: return 0;
  }
}

}  // namespace

// hipChannelFormatDesc, which hipCreateChannelDesc returns by value.
struct ChannelFormatDesc {
  int x, y, z, w;
  int f;   // hipChannelFormatKind: 0 signed, 1 unsigned, 2 float
};
ChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, int f) { return {x, y, z, w, f}; }

namespace {

// HIP_ARRAY_DESCRIPTOR and HIP_ARRAY3D_DESCRIPTOR.
struct ArrayDescriptor {
  size_t Width, Height;
  int Format;
  unsigned NumChannels;
};
struct Array3DDescriptor {
  size_t Width, Height, Depth;
  int Format;
  unsigned NumChannels, Flags;
};

// What hipChannelFormatDesc a format with so many channels is, and back.
ChannelFormatDesc channel_desc(unsigned channels, int format) {
  const int bits = static_cast<int>(8 * channel_bytes(format));
  const int kind = format == kHalf || format == kFloat ? 2 : format == kU8 || format == kU16 || format == kU32 ? 1 : 0;
  if (!bits || (channels != 1 && channels != 2 && channels != 4)) return {0, 0, 0, 0, 0};
  return {bits, channels > 1 ? bits : 0, channels > 2 ? bits : 0, channels > 2 ? bits : 0, kind};
}
int array_format(const ChannelFormatDesc& d) {
  switch (d.f) {
    case 1: return d.x == 8 ? kU8 : d.x == 16 ? kU16 : d.x == 32 ? kU32 : 0;
    case 0: return d.x == 8 ? kS8 : d.x == 16 ? kS16 : d.x == 32 ? kS32 : 0;
    case 2: return d.x == 16 ? kHalf : d.x == 32 ? kFloat : 0;
    default: return 0;
  }
}
unsigned channel_count(const ChannelFormatDesc& d) { return (d.x != 0) + (d.y != 0) + (d.z != 0) + (d.w != 0); }
// Every channel as wide as the first, and none after one that is missing.
bool well_formed(const ChannelFormatDesc& d) {
  if (d.x == 0) return false;
  if ((d.y && d.y != d.x) || (d.z && d.z != d.x) || (d.w && d.w != d.x)) return false;
  if (d.y == 0) return !(d.z > 0 || d.w > 0);
  if (d.z == 0) return !(d.w > 0);
  return true;
}

// A format as the image hardware reads it: an integer format read as a
// normalized float where asked (never a 32-bit one).
im::Format image_format(int format, unsigned channels, bool normalized) {
  const uint32_t bits = 8 * channel_bytes(format);
  static const im::Data k8[] = {im::Data::D8, im::Data::D8_8, im::Data::Invalid, im::Data::D8_8_8_8};
  static const im::Data k16[] = {im::Data::D16, im::Data::D16_16, im::Data::Invalid, im::Data::D16_16_16_16};
  static const im::Data k32[] = {im::Data::D32, im::Data::D32_32, im::Data::D32_32_32, im::Data::D32_32_32_32};
  im::Format f;
  if (!channels || channels > 4) return f;
  f.data = bits == 8 ? k8[channels - 1] : bits == 16 ? k16[channels - 1] : bits == 32 ? k32[channels - 1] : im::Data::Invalid;
  switch (format) {
    case kU8: case kU16: f.num = normalized ? im::Num::Unorm : im::Num::Uint; break;
    case kS8: case kS16: f.num = normalized ? im::Num::Snorm : im::Num::Sint; break;
    case kU32: f.num = im::Num::Uint; break;
    case kS32: f.num = im::Num::Sint; break;
    default: f.num = im::Num::Float; break;
  }
  return f;
}
// The channel type and order (HSA_EXT_IMAGE_CHANNEL_TYPE_*, _ORDER_*) the
// runtime puts after the resource, which HIP's surface functions read.
uint32_t hsa_channel_type(int format, bool normalized) {
  switch (format) {
    case kU8: return normalized ? 2 : 11;
    case kS8: return normalized ? 0 : 8;
    case kU16: return normalized ? 3 : 12;
    case kS16: return normalized ? 1 : 9;
    case kU32: return 13;
    case kS32: return 10;
    case kHalf: return 14;
    default: return 15;
  }
}
uint32_t hsa_channel_order(unsigned channels) { return channels == 1 ? 1 : channels == 2 ? 3 : 8; }
void channel_sels(unsigned channels, im::Sel sel[4]) {
  sel[0] = im::Sel::X;
  sel[1] = channels > 1 ? im::Sel::Y : im::Sel::Zero;
  sel[2] = channels > 2 ? im::Sel::Z : im::Sel::Zero;
  sel[3] = channels > 2 ? im::Sel::W : im::Sel::One;
}

// An array: device memory, and what it holds.
struct HipArray {
  int device = 0;
  uint8_t* data = nullptr;
  size_t width = 0, height = 0, depth = 0;
  int format = 0;
  unsigned channels = 0, flags = 0;
  uint32_t element() const { return channel_bytes(format) * channels; }
  bool layered_1d() const { return (flags & kArrayLayered) && height == 0; }
  size_t pitch() const { return width * element(); }
  size_t rows() const { return layered_1d() ? depth : std::max<size_t>(height, 1); }
  size_t slices() const { return layered_1d() ? 1 : std::max<size_t>(depth, 1); }
  im::Type type() const {
    if (flags & kArrayLayered) return height ? im::Type::Img2DArray : im::Type::Img1DArray;
    return depth ? im::Type::Img3D : height ? im::Type::Img2D : im::Type::Img1D;
  }
};
std::set<const HipArray*> g_arrays;   // under s.mutex

HipArray* find_array_locked(const void* handle) {
  auto it = g_arrays.find(static_cast<const HipArray*>(handle));
  return it == g_arrays.end() ? nullptr : const_cast<HipArray*>(*it);
}
HipArray* find_array(const void* handle) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return find_array_locked(handle);
}
extern "C++" bool is_array_handle(const void* p) { return find_array(p) != nullptr; }

// Makes an array as ROCm's HIP does (ihipArrayCreate): 1, 2 or 4 channels,
// no cubemap, a gather array 2D, and the dimensions within the device's.
hipError_t make_array(void** out, const Array3DDescriptor& d) {
  if (!out) return hipErrorInvalidValue;
  if (d.NumChannels != 1 && d.NumChannels != 2 && d.NumChannels != 4) return hipErrorInvalidValue;
  if (d.Flags & ~(kArrayLayered | kArraySurfaceLoadStore | kArrayTextureGather)) return hipErrorInvalidValue;
  if ((d.Flags & kArrayTextureGather) && (!d.Width || !d.Height || d.Depth)) return hipErrorInvalidValue;
  if (!channel_bytes(d.Format)) return hipErrorInvalidValue;
  const bool layered = d.Flags & kArrayLayered;
  const uint64_t w = d.Width, h = d.Height, z = d.Depth;
  bool shape = false, fits = false;
  if (layered) {
    shape = w && z;
    fits = w <= kMax2D && h <= kMax2D && z <= kMaxLayers;
  } else {
    shape = w && (h || !z);
    fits = z ? w <= kMax3D && h <= kMax3D && z <= kMax3DDepth : h ? w <= kMax2D && h <= kMax2D : w <= kMax1D;
  }
  if (!shape || !fits) return hipErrorInvalidValue;
  auto a = std::make_unique<HipArray>();
  a->width = d.Width, a->height = d.Height, a->depth = d.Depth;
  a->format = d.Format, a->channels = d.NumChannels, a->flags = d.Flags;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!images_on(s)) return hipErrorNotSupported;
    a->device = s.current;
  }
  void* p = nullptr;
  if (hipMalloc(&p, a->pitch() * a->rows() * a->slices()) != hipSuccess) return hipErrorOutOfMemory;
  a->data = static_cast<uint8_t*>(p);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  *out = a.get();
  g_arrays.insert(a.release());
  return hipSuccess;
}

// Frees one (ihipArrayDestroy): an array that is not one, or is one no
// more, is a destroyed context's, as ROCm's HIP says.
hipError_t free_array(void* handle) {
  if (!handle) return hipErrorInvalidValue;
  HipArray* a = nullptr;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    a = find_array_locked(handle);
    if (!a) return hipErrorContextIsDestroyed;
    g_arrays.erase(a);
  }
  std::unique_ptr<HipArray> owned(a);
  const hipError_t e = hipFree(a->data);   // after the device's work that might use it
  return e == hipSuccess ? hipSuccess : hipErrorInvalidValue;
}

}  // namespace

hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t* width, const ChannelFormatDesc* desc, int device) {
  const ApiCall api("hipDeviceGetTexture1DLinearMaxWidth");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!width || !desc) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (device < 0 || device >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  // The widest linear texture, in elements of the format given (none is
  // the wrong value): the properties' widest, in bytes, over the element's.
  const size_t element = static_cast<size_t>(desc->x + desc->y + desc->z + desc->w) / 8;
  if (!element) return record(s, hipErrorInvalidValue);
  *width = has_images(s.rt->device(device).profile()) ? static_cast<size_t>(INT32_MAX) / element : 0;
  return record(s, hipSuccess);
}

// What a hipcc-built program registers before main for a texture or surface
// reference it declares: nothing to keep, since none can be bound.
void __hipRegisterTexture(void** modules, void* host_var, char*, const char* device_name, int, int, int) {
  // A texture reference is a variable of the program's like any other: the
  // device's copy is what tex1Dfetch and the rest read the bound object from.
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!modules || !host_var || !device_name) return;
  s.host_vars[host_var] = HostVar{reinterpret_cast<FatBinary*>(modules), device_name, 88 /* textureReference */};
}
void __hipRegisterSurface(void*, void*, char*, const char*, int, int) {}

// A mipmapped array: its arguments checked, then refused -- ROCm's HIP on
// Linux has no mipmaps (its texture objects refuse one on every device).
hipError_t hipMallocMipmappedArray(void** array, const void* desc, vgpu::amd::abi::Extent, unsigned int, unsigned int) {
  if (!array || !desc) {
    const ApiCall api("hipMallocMipmappedArray");
    return record(state(), hipErrorInvalidValue);
  }
  return no_images("hipMallocMipmappedArray");
}

// ---- Making, asking about and freeing arrays ------------------------------------

hipError_t hipMallocArray(void** array, const ChannelFormatDesc* desc, size_t width, size_t height,
                          unsigned int flags) {
  if (!images_here()) return no_images("hipMallocArray");
  const ApiCall api("hipMallocArray");
  if (!array || !desc || !well_formed(*desc)) return record(state(), hipErrorInvalidValue);
  const Array3DDescriptor d{width, height, 0, array_format(*desc), channel_count(*desc), flags};
  return record(state(), make_array(array, d));
}
hipError_t hipMalloc3DArray(void** array, const ChannelFormatDesc* desc, vgpu::amd::abi::Extent extent,
                            unsigned int flags) {
  if (!images_here()) return no_images("hipMalloc3DArray");
  const ApiCall api("hipMalloc3DArray");
  if (!array || !desc || !well_formed(*desc)) return record(state(), hipErrorInvalidValue);
  const Array3DDescriptor d{extent.width, extent.height, extent.depth, array_format(*desc), channel_count(*desc),
                            flags};
  return record(state(), make_array(array, d));
}
hipError_t hipArrayCreate(void** array, const ArrayDescriptor* desc) {
  if (!images_here()) return no_images("hipArrayCreate");
  const ApiCall api("hipArrayCreate");
  if (!desc) return record(state(), hipErrorInvalidValue);
  return record(state(), make_array(array, Array3DDescriptor{desc->Width, desc->Height, 0, desc->Format,
                                                              desc->NumChannels, 0}));
}
hipError_t hipArray3DCreate(void** array, const Array3DDescriptor* desc) {
  if (!images_here()) return no_images("hipArray3DCreate");
  const ApiCall api("hipArray3DCreate");
  if (!desc) return record(state(), hipErrorInvalidValue);
  return record(state(), make_array(array, *desc));
}
hipError_t hipFreeArray(void* array) {
  if (!images_here()) return freeing_nothing("hipFreeArray");
  const ApiCall api("hipFreeArray");
  return record(state(), free_array(array));
}
hipError_t hipArrayDestroy(void* array) {
  if (!images_here()) return freeing_nothing("hipArrayDestroy");
  const ApiCall api("hipArrayDestroy");
  return record(state(), free_array(array));
}

hipError_t hipGetChannelDesc(ChannelFormatDesc* desc, const void* array) {
  if (!images_here()) return no_such_array("hipGetChannelDesc");
  const ApiCall api("hipGetChannelDesc");
  if (!desc) return record(state(), hipErrorInvalidValue);
  const HipArray* a = find_array(array);
  if (!a) return record(state(), hipErrorInvalidHandle);
  *desc = channel_desc(a->channels, a->format);
  return record(state(), hipSuccess);
}
hipError_t hipArrayGetInfo(ChannelFormatDesc* desc, vgpu::amd::abi::Extent* extent, unsigned int* flags,
                           void* array) {
  if (!images_here()) return no_such_array("hipArrayGetInfo");
  const ApiCall api("hipArrayGetInfo");
  if (!array) return record(state(), hipErrorInvalidHandle);
  if (!desc && !extent && !flags) return record(state(), hipSuccess);
  const HipArray* a = find_array(array);
  if (!a) return record(state(), hipErrorInvalidHandle);
  if (desc) *desc = channel_desc(a->channels, a->format);
  if (extent) *extent = {a->width, a->height, a->depth};
  if (flags) *flags = a->flags;
  return record(state(), hipSuccess);
}
hipError_t hipArrayGetDescriptor(ArrayDescriptor* desc, void* array) {
  if (!images_here()) return no_such_array("hipArrayGetDescriptor");
  const ApiCall api("hipArrayGetDescriptor");
  if (!array) return record(state(), hipErrorInvalidHandle);
  if (!desc) return record(state(), hipErrorInvalidValue);
  const HipArray* a = find_array(array);
  if (!a) return record(state(), hipErrorInvalidHandle);
  *desc = {a->width, a->height, a->format, a->channels};
  return record(state(), hipSuccess);
}
hipError_t hipArray3DGetDescriptor(Array3DDescriptor* desc, void* array) {
  if (!images_here()) return no_such_array("hipArray3DGetDescriptor");
  const ApiCall api("hipArray3DGetDescriptor");
  if (!array) return record(state(), hipErrorInvalidHandle);
  if (!desc) return record(state(), hipErrorInvalidValue);
  const HipArray* a = find_array(array);
  if (!a) return record(state(), hipErrorInvalidHandle);
  *desc = {a->width, a->height, a->depth, a->format, a->channels, a->flags};
  return record(state(), hipSuccess);
}

// ---- Copies to and from arrays ---------------------------------------------------
//
// Each is a 2D copy (hipMemcpyParam2D's) with an array on one side or both:
// an offset along a row and a width in bytes, rows below that.

namespace {
hipError_t param_2d(const vgpu::amd::abi::Memcpy2D* p, hipStream_t stream, bool async);

// The side a kind names (host or device memory), as hip::getMemoryType
// does for a 2D copy to or from an array.
int memory_type(hipMemcpyKind kind, bool source) {
  using namespace vgpu::amd::abi;
  switch (static_cast<int>(kind)) {
    case hipMemcpyHostToHost: return kMemTypeHost;
    case hipMemcpyHostToDevice: return source ? kMemTypeHost : kMemTypeDevice;
    case hipMemcpyDeviceToHost: return source ? kMemTypeDevice : kMemTypeHost;
    case hipMemcpyDeviceToDevice:
    case 1024: return kMemTypeDevice;   // hipMemcpyDeviceToDeviceNoCU
    default: return kMemTypeUnified;
  }
}
// The directions, and hipMemcpyDeviceToDeviceNoCU (1024), a device-to-device
// copy ROCm's HIP accepts wherever it accepts the others.
bool valid_kind(int kind) { return (kind >= hipMemcpyHostToHost && kind <= hipMemcpyDefault) || kind == 1024; }

// hipMemcpy2DValidateArray: the rows and bytes inside the array.
hipError_t check_array_rows(const void* handle, size_t x, size_t y, size_t width, size_t height) {
  const HipArray* a = find_array(handle);
  if (!a) return hipErrorInvalidHandle;
  if (x + width > a->pitch()) return hipErrorInvalidValue;
  if (y + height > (a->height ? a->height : 1)) return hipErrorInvalidValue;
  return hipSuccess;
}

hipError_t to_array(void* dst, size_t x, size_t y, const void* src, size_t spitch, size_t width, size_t height,
                    hipMemcpyKind kind, hipStream_t stream, bool async) {
  vgpu::amd::abi::Memcpy2D d{};
  d.srcMemoryType = memory_type(kind, true);
  d.srcHost = src;
  d.srcDevice = const_cast<void*>(src);
  d.srcPitch = spitch;
  d.dstXInBytes = x, d.dstY = y;
  d.dstMemoryType = vgpu::amd::abi::kMemTypeArray;
  d.dstArray = dst;
  d.WidthInBytes = width, d.Height = height;
  return param_2d(&d, stream, async);
}
hipError_t from_array(void* dst, size_t dpitch, const void* src, size_t x, size_t y, size_t width, size_t height,
                      hipMemcpyKind kind, hipStream_t stream, bool async) {
  vgpu::amd::abi::Memcpy2D d{};
  d.srcXInBytes = x, d.srcY = y;
  d.srcMemoryType = vgpu::amd::abi::kMemTypeArray;
  d.srcArray = const_cast<void*>(src);
  d.dstMemoryType = memory_type(kind, false);
  d.dstHost = dst;
  d.dstDevice = dst;
  d.dstPitch = dpitch;
  d.WidthInBytes = width, d.Height = height;
  return param_2d(&d, stream, async);
}

// hipMemcpy2DToArray and its forms: the kind, the source and the array's
// rows checked first.
hipError_t copy_2d_to_array(void* dst, size_t x, size_t y, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream, bool async) {
  if (!valid_kind(kind)) return hipErrorInvalidMemcpyDirection;
  if (!src) return hipErrorInvalidValue;
  if (spitch < width) return hipErrorInvalidPitchValue;
  if (const hipError_t e = check_array_rows(dst, x, y, width, height); e != hipSuccess) return e;
  return to_array(dst, x, y, src, spitch, width, height, kind, stream, async);
}
hipError_t copy_2d_from_array(void* dst, size_t dpitch, const void* src, size_t x, size_t y, size_t width,
                              size_t height, hipMemcpyKind kind, hipStream_t stream, bool async) {
  if (!valid_kind(kind)) return hipErrorInvalidMemcpyDirection;
  if (!dst) return hipErrorInvalidValue;
  if (dpitch < width) return hipErrorInvalidPitchValue;
  if (const hipError_t e = check_array_rows(src, x, y, width, height); e != hipSuccess) return e;
  return from_array(dst, dpitch, src, x, y, width, height, kind, stream, async);
}
// hipMemcpyToArray and hipMemcpyFromArray: `count` bytes as the array's
// rows, as ROCm's HIP divides them.
hipError_t copy_count_to_array(void* dst, size_t x, size_t y, const void* src, size_t count, hipMemcpyKind kind,
                               hipStream_t stream, bool async) {
  const HipArray* a = find_array(dst);
  if (!a) return hipErrorInvalidValue;
  const size_t width = count / (a->height ? a->height : 1), height = count / a->width / a->element();
  return to_array(dst, x, y, src, 0, width, height, kind, stream, async);
}
hipError_t copy_count_from_array(void* dst, const void* src, size_t x, size_t y, size_t count, hipMemcpyKind kind,
                                 hipStream_t stream, bool async) {
  const HipArray* a = find_array(src);
  if (!a) return hipErrorInvalidValue;
  const size_t width = count / (a->height ? a->height : 1), height = count / a->width / a->element();
  return from_array(dst, 0, src, x, y, width, height, kind, stream, async);
}
}  // namespace

hipError_t hipMemcpy2DToArray(void* dst, size_t x, size_t y, const void* src, size_t spitch, size_t width,
                              size_t height, hipMemcpyKind kind) {
  if (!images_here()) return no_such_array("hipMemcpy2DToArray");
  const ApiCall api("hipMemcpy2DToArray");
  return record(state(), copy_2d_to_array(dst, x, y, src, spitch, width, height, kind, nullptr, false));
}
hipError_t hipMemcpy2DToArray_spt(void* dst, size_t x, size_t y, const void* src, size_t spitch, size_t width,
                                  size_t height, hipMemcpyKind kind) {
  if (!images_here()) return no_such_array("hipMemcpy2DToArray_spt");
  const ApiCall api("hipMemcpy2DToArray_spt");
  return record(state(), copy_2d_to_array(dst, x, y, src, spitch, width, height, kind, spt(nullptr), false));
}
hipError_t hipMemcpy2DToArrayAsync(void* dst, size_t x, size_t y, const void* src, size_t spitch, size_t width,
                                   size_t height, hipMemcpyKind kind, hipStream_t stream) {
  if (!images_here()) return no_such_array("hipMemcpy2DToArrayAsync");
  const ApiCall api("hipMemcpy2DToArrayAsync");
  return record(state(), copy_2d_to_array(dst, x, y, src, spitch, width, height, kind, stream, true));
}
hipError_t hipMemcpy2DToArrayAsync_spt(void* dst, size_t x, size_t y, const void* src, size_t spitch,
                                       size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
  if (!images_here()) return no_such_array("hipMemcpy2DToArrayAsync_spt");
  const ApiCall api("hipMemcpy2DToArrayAsync_spt");
  return record(state(), copy_2d_to_array(dst, x, y, src, spitch, width, height, kind,
                                          spt(stream), true));
}
hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, const void* src, size_t x, size_t y, size_t width,
                                size_t height, hipMemcpyKind kind) {
  if (!images_here()) return no_such_array("hipMemcpy2DFromArray");
  const ApiCall api("hipMemcpy2DFromArray");
  return record(state(), copy_2d_from_array(dst, dpitch, src, x, y, width, height, kind, nullptr, false));
}
hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch, const void* src, size_t x, size_t y, size_t width,
                                    size_t height, hipMemcpyKind kind) {
  if (!images_here()) return no_such_array("hipMemcpy2DFromArray_spt");
  const ApiCall api("hipMemcpy2DFromArray_spt");
  return record(state(),
                copy_2d_from_array(dst, dpitch, src, x, y, width, height, kind, spt(nullptr), false));
}
hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, const void* src, size_t x, size_t y, size_t width,
                                     size_t height, hipMemcpyKind kind, hipStream_t stream) {
  if (!images_here()) return no_such_array("hipMemcpy2DFromArrayAsync");
  const ApiCall api("hipMemcpy2DFromArrayAsync");
  return record(state(), copy_2d_from_array(dst, dpitch, src, x, y, width, height, kind, stream, true));
}
hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch, const void* src, size_t x, size_t y,
                                         size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
  if (!images_here()) return no_such_array("hipMemcpy2DFromArrayAsync_spt");
  const ApiCall api("hipMemcpy2DFromArrayAsync_spt");
  return record(state(), copy_2d_from_array(dst, dpitch, src, x, y, width, height, kind,
                                            spt(stream), true));
}
hipError_t hipMemcpyToArray(void* dst, size_t x, size_t y, const void* src, size_t count, hipMemcpyKind kind) {
  if (!images_here()) return no_such_array("hipMemcpyToArray");
  const ApiCall api("hipMemcpyToArray");
  return record(state(), copy_count_to_array(dst, x, y, src, count, kind, nullptr, false));
}
hipError_t hipMemcpyFromArray(void* dst, const void* src, size_t x, size_t y, size_t count, hipMemcpyKind kind) {
  if (!images_here()) return no_such_array("hipMemcpyFromArray");
  const ApiCall api("hipMemcpyFromArray");
  return record(state(), copy_count_from_array(dst, src, x, y, count, kind, nullptr, false));
}
hipError_t hipMemcpyFromArray_spt(void* dst, const void* src, size_t x, size_t y, size_t count,
                                  hipMemcpyKind kind) {
  if (!images_here()) return no_such_array("hipMemcpyFromArray_spt");
  const ApiCall api("hipMemcpyFromArray_spt");
  return record(state(), copy_count_from_array(dst, src, x, y, count, kind, spt(nullptr), false));
}
hipError_t hipMemcpy2DArrayToArray(void* dst, size_t dx, size_t dy, const void* src, size_t sx, size_t sy,
                                   size_t width, size_t height, hipMemcpyKind kind) {
  if (!images_here()) return no_such_array("hipMemcpy2DArrayToArray");
  const ApiCall api("hipMemcpy2DArrayToArray");
  if (!valid_kind(kind)) return record(state(), hipErrorInvalidMemcpyDirection);
  if (const hipError_t e = check_array_rows(src, sx, sy, width, height); e != hipSuccess) return record(state(), e);
  if (const hipError_t e = check_array_rows(dst, dx, dy, width, height); e != hipSuccess) return record(state(), e);
  vgpu::amd::abi::Memcpy2D d{};
  d.srcXInBytes = sx, d.srcY = sy, d.srcMemoryType = vgpu::amd::abi::kMemTypeArray, d.srcArray = const_cast<void*>(src);
  d.dstXInBytes = dx, d.dstY = dy, d.dstMemoryType = vgpu::amd::abi::kMemTypeArray, d.dstArray = dst;
  d.WidthInBytes = width, d.Height = height;
  return record(state(), param_2d(&d, nullptr, false));
}

// ---- Texture and surface objects ---------------------------------------------------

namespace {

// hipResourceDesc, hipTextureDesc and hipResourceViewDesc.
struct ResourceDesc {
  int resType;   // 0 array, 1 mipmapped array, 2 linear, 3 pitch 2D
  union {
    struct {
      void* array;
    } array;
    struct {
      void* mipmap;
    } mipmap;
    struct {
      void* devPtr;
      ChannelFormatDesc desc;
      size_t sizeInBytes;
    } linear;
    struct {
      void* devPtr;
      ChannelFormatDesc desc;
      size_t width, height, pitchInBytes;
    } pitch2D;
  } res;
};
struct TextureDesc {
  int addressMode[3];   // 0 wrap, 1 clamp, 2 mirror, 3 border
  int filterMode;       // 0 point, 1 linear
  int readMode;         // 0 the element's type, 1 normalized float
  int sRGB;
  float borderColor[4];
  int normalizedCoords;
  unsigned maxAnisotropy;
  int mipmapFilterMode;
  float mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp;
};
struct ResourceViewDesc {   // no reserved words, unlike the driver API's
  int format;
  size_t width, height, depth;
  unsigned firstMipmapLevel, lastMipmapLevel, firstLayer, lastLayer;
};

// A resource view's format (hipResourceViewFormat): a hipArray_Format and
// its channels, for the formats with 1, 2 or 4 channels of 8, 16 or 32
// bits; 0 for the rest.
bool view_format(int view, int* format, unsigned* channels) {
  static const int kFormats[] = {kU8, kS8, kU16, kS16, kU32, kS32, kHalf, kFloat};
  // 0x01-0x18: {unsigned char, signed char, unsigned short, signed short,
  // unsigned int, signed int, half, float} x {1, 2, 4} channels.
  if (view < 0x01 || view > 0x18) return false;
  *format = kFormats[(view - 1) / 3];
  *channels = (view - 1) % 3 == 0 ? 1 : (view - 1) % 3 == 1 ? 2 : 4;
  return true;
}

// An object's memory as the kernel reads it: the image resource and words
// 8-10, and a texture's sampler at word 12.
constexpr size_t kObjectWords = 20;
struct Object {
  bool texture = true;
  int device = 0;
  ResourceDesc res{};
  TextureDesc tex{};
  ResourceViewDesc view{};
};
std::map<uint64_t, Object> g_objects;   // under s.mutex, by the object's address

// Writes an object's words into a fresh allocation on the current device;
// the allocation's address is the handle.
hipError_t place_object(const uint32_t words[kObjectWords], Object o, void** handle) {
  void* p = nullptr;
  if (hipMalloc(&p, kObjectWords * 4) != hipSuccess || !p) return hipErrorOutOfMemory;
  if (hipMemcpy(p, words, kObjectWords * 4, hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipFree(p);
    return hipErrorOutOfMemory;
  }
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  o.device = s.current;
  g_objects[reinterpret_cast<uint64_t>(p)] = o;
  *handle = p;
  return hipSuccess;
}

// The image resource of an array, as a texture reads it (in `format`) or a
// surface writes it (in the array's own); words 8-10 after it.
void array_resource(const HipArray& a, int format, unsigned channels, bool normalized, bool srgb, im::Gen g,
                    uint32_t words[kObjectWords]) {
  im::Image i;
  i.base = reinterpret_cast<uint64_t>(a.data);
  i.width = static_cast<uint32_t>(a.width);
  i.height = static_cast<uint32_t>(a.layered_1d() ? 1 : std::max<size_t>(a.height, 1));
  i.depth = static_cast<uint32_t>(std::max<size_t>(a.depth, 1));
  i.format = image_format(format, channels, normalized);
  if (srgb) i.format.num = im::Num::Srgb;
  channel_sels(channels, i.sel);
  i.type = a.type();
  im::encode(i, g, words);
  words[8] = hsa_channel_type(format, normalized);
  words[9] = srgb ? 14 /* SRGBA */ : hsa_channel_order(channels);
  words[10] = static_cast<uint32_t>(a.width);
}

// The sampler: each axis's address mode (wrap and mirror need normalized
// coordinates; without them they clamp), point or bilinear filtering, and
// no mipmaps -- ROCm's image runtime's sampler, whose border is always
// transparent black.
void sampler_resource(const TextureDesc& t, bool linear_memory, im::Gen g, uint32_t words[4]) {
  im::Sampler smp;
  for (int k = 0; k < 3; ++k) {
    int mode = t.addressMode[k];
    if (!t.normalizedCoords && (mode == 0 || mode == 2)) mode = 1;
    smp.clamp[k] = linear_memory ? im::Clamp::Wrap
                   : mode == 0   ? im::Clamp::Wrap
                   : mode == 2   ? im::Clamp::Mirror
                   : mode == 3   ? im::Clamp::ClampBorder
                                 : im::Clamp::ClampLastTexel;
  }
  smp.unnormalized = !t.normalizedCoords;
  smp.mag_linear = smp.min_linear = !linear_memory && t.filterMode == 1;
  im::encode(smp, g, words);
}

// hipCreateTextureObject, checked as ROCm's HIP checks it.
hipError_t create_texture(void** handle, const ResourceDesc* res, const TextureDesc* tex,
                          const ResourceViewDesc* view) {
  im::Gen g;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!images_on(s)) return hipErrorNotSupported;
    g = image_gen(device(s)->profile());
  }
  if (!handle || !res || !tex) return hipErrorInvalidChannelDescriptor;
  if (view && res->resType != 0 && res->resType != 1) return hipErrorUnknown;
  if (res->resType == 1) return hipErrorNotSupported;   // no mipmaps on ROCm's Linux runtime
  if (res->resType < 0 || res->resType > 3) return hipErrorInvalidValue;
  if (res->resType == 0 && !res->res.array.array) return hipErrorInvalidValue;
  const auto aligned = [](const void* p, uint64_t n) { return reinterpret_cast<uint64_t>(p) % n == 0; };
  if (res->resType == 2) {
    const auto& l = res->res.linear;
    const uint64_t element = static_cast<uint64_t>(l.desc.x + l.desc.y + l.desc.z + l.desc.w) / 8;
    if (!l.devPtr || !aligned(l.devPtr, kImageAlignment) || l.sizeInBytes >= 0xFFFFFFFFull * element)
      return hipErrorInvalidChannelDescriptor;
  }
  if (res->resType == 3) {
    const auto& p = res->res.pitch2D;
    if (!p.devPtr || !aligned(p.devPtr, kImageAlignment) || p.width > kMax2D || p.height > kMax2D ||
        p.pitchInBytes % kImageAlignment)
      return hipErrorInvalidValue;
  }
  if (tex->maxAnisotropy != 0 || tex->mipmapLevelBias != 0) return hipErrorNotSupported;
  // sRGB: a four-channel 8-bit texture read as normalized floats is sRGBA
  // (any other four-channel one is no format there is); with fewer channels
  // it is ignored, as ROCm's HIP does.
  const auto srgb_of = [&](int format, unsigned channels, bool normalized, bool* srgb) {
    *srgb = tex->sRGB == 1 && channels == 4;
    return !*srgb || (format == kU8 && normalized);
  };

  uint32_t words[kObjectWords] = {};
  const bool normalized = tex->readMode == 1;
  if (res->resType == 0) {
    const HipArray* a = find_array(res->res.array.array);
    if (!a) return hipErrorInvalidValue;
    int format = a->format;
    unsigned channels = a->channels;
    // A view names the format the texels are read in; one naming none
    // (or one this does not know) is the wrong value, as ROCm's HIP has it.
    if (view && !view_format(view->format, &format, &channels)) return hipErrorInvalidValue;
    bool srgb = false;
    if (!srgb_of(format, channels, normalized, &srgb)) return hipErrorInvalidValue;
    // A 32-bit integer format is never read normalized.
    array_resource(*a, format, channels, normalized && format != kU32 && format != kS32, srgb, g, words);
  } else {
    const ChannelFormatDesc& d = res->resType == 2 ? res->res.linear.desc : res->res.pitch2D.desc;
    const int format = array_format(d);
    const unsigned channels = channel_count(d);
    if (!format || !well_formed(d) || channels == 3) return hipErrorInvalidValue;
    bool srgb = false;
    if (!srgb_of(format, channels, normalized, &srgb)) return hipErrorInvalidValue;
    // A buffer resource's format field has no room for sRGB's codes on gfx11
    // and gfx12, so linear memory read as sRGB is not modelled.
    if (srgb && res->resType == 2) return fail(hipErrorNotSupported, "sRGB over linear memory is not modelled");
    im::Format f = image_format(format, channels, normalized);
    if (srgb) f.num = im::Num::Srgb;
    const uint32_t element = channel_bytes(format) * channels;
    if (res->resType == 2) {
      // Linear memory: a buffer resource, one element to a record, which
      // tex1Dfetch reads with a formatted buffer load.
      const auto& l = res->res.linear;
      const uint32_t width = static_cast<uint32_t>(l.sizeInBytes / element);
      im::encode_buffer(reinterpret_cast<uint64_t>(l.devPtr), element, width, f, g, words);
      im::Sel sel[4];
      channel_sels(channels, sel);
      words[3] &= ~0xFFFu;
      for (int k = 0; k < 4; ++k) words[3] |= (static_cast<uint32_t>(sel[k]) & 7) << (3 * k);
      words[10] = width;
    } else {
      const auto& p = res->res.pitch2D;
      if (p.pitchInBytes % element) return hipErrorInvalidValue;
      im::Image i;
      i.base = reinterpret_cast<uint64_t>(p.devPtr);
      i.width = static_cast<uint32_t>(p.width);
      i.height = static_cast<uint32_t>(std::max<size_t>(p.height, 1));
      i.pitch = static_cast<uint32_t>(p.pitchInBytes / element);
      i.format = f;
      channel_sels(channels, i.sel);
      i.type = im::Type::Img2D;
      im::encode(i, g, words);
      words[10] = static_cast<uint32_t>(p.width);
    }
    words[8] = hsa_channel_type(format, normalized && format != kU32 && format != kS32);
    words[9] = srgb ? 14 /* SRGBA */ : hsa_channel_order(channels);
  }
  sampler_resource(*tex, res->resType == 2, g, words + 12);
  Object o;
  o.res = *res;
  o.tex = *tex;
  if (view) o.view = *view;
  return place_object(words, o, handle);
}

hipError_t destroy_object(void* handle, bool texture) {
  if (!handle) return hipSuccess;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = g_objects.find(reinterpret_cast<uint64_t>(handle));
    if (it == g_objects.end() || it->second.texture != texture) return hipErrorInvalidValue;
    g_objects.erase(it);
  }
  return hipFree(handle) == hipSuccess ? hipSuccess : hipErrorInvalidValue;
}

// What an object was made from, for the getters.
hipError_t object_of(const void* handle, bool texture, Object* out) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!images_on(s)) return hipErrorNotSupported;
  auto it = g_objects.find(reinterpret_cast<uint64_t>(handle));
  if (it == g_objects.end() || it->second.texture != texture) return hipErrorInvalidValue;
  *out = it->second;
  return hipSuccess;
}

// The driver API's forms: HIP_RESOURCE_DESC, HIP_TEXTURE_DESC and
// HIP_RESOURCE_VIEW_DESC, whose linear resources name a format and a
// channel count, and whose texture flags carry the read mode, normalization
// and sRGB.
struct DrvResourceDesc {
  int resType;
  union {
    struct {
      void* hArray;
    } array;
    struct {
      void* hMipmappedArray;
    } mipmap;
    struct {
      void* devPtr;
      int format;
      unsigned numChannels;
      size_t sizeInBytes;
    } linear;
    struct {
      void* devPtr;
      int format;
      unsigned numChannels;
      size_t width, height, pitchInBytes;
    } pitch2D;
    struct {
      int reserved[32];
    } reserved;
  } res;
  unsigned flags;
};
struct DrvTextureDesc {
  int addressMode[3];
  int filterMode;
  unsigned flags;   // 1 read as integer, 2 normalized coordinates, 0x10 sRGB
  unsigned maxAnisotropy;
  int mipmapFilterMode;
  float mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp;
  float borderColor[4];
  int reserved[12];
};
// HIP_RESOURCE_VIEW_DESC: hipResourceViewDesc's fields, and 16 reserved
// words after them.
struct DrvResourceViewDesc {
  ResourceViewDesc view;
  unsigned reserved[16];
};

ResourceDesc resource_from_driver(const DrvResourceDesc& d) {
  ResourceDesc r{};
  r.resType = d.resType;
  switch (d.resType) {
    case 0: r.res.array.array = d.res.array.hArray; break;
    case 1: r.res.mipmap.mipmap = d.res.mipmap.hMipmappedArray; break;
    case 2:
      r.res.linear.devPtr = d.res.linear.devPtr;
      r.res.linear.desc = channel_desc(d.res.linear.numChannels, d.res.linear.format);
      r.res.linear.sizeInBytes = d.res.linear.sizeInBytes;
      break;
    case 3:
      r.res.pitch2D.devPtr = d.res.pitch2D.devPtr;
      r.res.pitch2D.desc = channel_desc(d.res.pitch2D.numChannels, d.res.pitch2D.format);
      r.res.pitch2D.width = d.res.pitch2D.width;
      r.res.pitch2D.height = d.res.pitch2D.height;
      r.res.pitch2D.pitchInBytes = d.res.pitch2D.pitchInBytes;
      break;
    default: break;
  }
  return r;
}
DrvResourceDesc resource_to_driver(const ResourceDesc& r) {
  DrvResourceDesc d{};
  d.resType = r.resType;
  switch (r.resType) {
    case 0: d.res.array.hArray = r.res.array.array; break;
    case 1: d.res.mipmap.hMipmappedArray = r.res.mipmap.mipmap; break;
    case 2:
      d.res.linear.devPtr = r.res.linear.devPtr;
      d.res.linear.format = array_format(r.res.linear.desc);
      d.res.linear.numChannels = channel_count(r.res.linear.desc);
      d.res.linear.sizeInBytes = r.res.linear.sizeInBytes;
      break;
    case 3:
      d.res.pitch2D.devPtr = r.res.pitch2D.devPtr;
      d.res.pitch2D.format = array_format(r.res.pitch2D.desc);
      d.res.pitch2D.numChannels = channel_count(r.res.pitch2D.desc);
      d.res.pitch2D.width = r.res.pitch2D.width;
      d.res.pitch2D.height = r.res.pitch2D.height;
      d.res.pitch2D.pitchInBytes = r.res.pitch2D.pitchInBytes;
      break;
    default: break;
  }
  return d;
}
TextureDesc texture_from_driver(const DrvTextureDesc& d) {
  TextureDesc t{};
  for (int k = 0; k < 3; ++k) t.addressMode[k] = d.addressMode[k];
  t.filterMode = d.filterMode;
  t.readMode = (d.flags & 1) ? 0 : 1;
  t.normalizedCoords = (d.flags & 2) ? 1 : 0;
  t.sRGB = (d.flags & 0x10) ? 1 : 0;
  for (int k = 0; k < 4; ++k) t.borderColor[k] = d.borderColor[k];
  t.maxAnisotropy = d.maxAnisotropy;
  t.mipmapFilterMode = d.mipmapFilterMode;
  t.mipmapLevelBias = d.mipmapLevelBias;
  t.minMipmapLevelClamp = d.minMipmapLevelClamp;
  t.maxMipmapLevelClamp = d.maxMipmapLevelClamp;
  return t;
}
DrvTextureDesc texture_to_driver(const TextureDesc& t) {
  DrvTextureDesc d{};
  for (int k = 0; k < 3; ++k) d.addressMode[k] = t.addressMode[k];
  d.filterMode = t.filterMode;
  d.flags = (t.readMode == 0 ? 1u : 0u) | (t.normalizedCoords ? 2u : 0u) | (t.sRGB ? 0x10u : 0u);
  d.maxAnisotropy = t.maxAnisotropy;
  d.mipmapFilterMode = t.mipmapFilterMode;
  d.mipmapLevelBias = t.mipmapLevelBias;
  d.minMipmapLevelClamp = t.minMipmapLevelClamp;
  d.maxMipmapLevelClamp = t.maxMipmapLevelClamp;
  for (int k = 0; k < 4; ++k) d.borderColor[k] = t.borderColor[k];
  return d;
}
}  // namespace

hipError_t hipCreateTextureObject(void** object, const ResourceDesc* res, const TextureDesc* tex,
                                  const ResourceViewDesc* view) {
  if (!images_here()) return no_images("hipCreateTextureObject");
  const ApiCall api("hipCreateTextureObject");
  return record(state(), create_texture(object, res, tex, view));
}
hipError_t hipTexObjectCreate(void** object, const DrvResourceDesc* res, const DrvTextureDesc* tex,
                              const DrvResourceViewDesc* view) {
  if (!images_here()) return no_images("hipTexObjectCreate");
  const ApiCall api("hipTexObjectCreate");
  if (!object || !res || !tex) return record(state(), hipErrorInvalidValue);
  const ResourceDesc r = resource_from_driver(*res);
  const TextureDesc t = texture_from_driver(*tex);
  return record(state(), create_texture(object, &r, &t, view ? &view->view : nullptr));
}
hipError_t hipDestroyTextureObject(void* object) {
  if (!images_here()) return destroying("hipDestroyTextureObject", reinterpret_cast<uint64_t>(object));
  const ApiCall api("hipDestroyTextureObject");
  return record(state(), destroy_object(object, true));
}
hipError_t hipTexObjectDestroy(void* object) {
  if (!images_here()) return destroying("hipTexObjectDestroy", reinterpret_cast<uint64_t>(object));
  const ApiCall api("hipTexObjectDestroy");
  return record(state(), destroy_object(object, true));
}

hipError_t hipGetTextureObjectResourceDesc(ResourceDesc* out, void* object) {
  if (!images_here()) return no_such_array("hipGetTextureObjectResourceDesc");
  const ApiCall api("hipGetTextureObjectResourceDesc");
  if (!out || !object) return record(state(), hipErrorInvalidValue);
  Object o;
  if (const hipError_t e = object_of(object, true, &o); e != hipSuccess) return record(state(), e);
  *out = o.res;
  return record(state(), hipSuccess);
}
hipError_t hipGetTextureObjectTextureDesc(TextureDesc* out, void* object) {
  if (!images_here()) return no_such_array("hipGetTextureObjectTextureDesc");
  const ApiCall api("hipGetTextureObjectTextureDesc");
  if (!out || !object) return record(state(), hipErrorInvalidValue);
  Object o;
  if (const hipError_t e = object_of(object, true, &o); e != hipSuccess) return record(state(), e);
  *out = o.tex;
  return record(state(), hipSuccess);
}
hipError_t hipGetTextureObjectResourceViewDesc(ResourceViewDesc* out, void* object) {
  if (!images_here()) return no_such_array("hipGetTextureObjectResourceViewDesc");
  const ApiCall api("hipGetTextureObjectResourceViewDesc");
  if (!out || !object) return record(state(), hipErrorInvalidValue);
  Object o;
  if (const hipError_t e = object_of(object, true, &o); e != hipSuccess) return record(state(), e);
  *out = o.view;
  return record(state(), hipSuccess);
}
hipError_t hipTexObjectGetResourceDesc(DrvResourceDesc* out, void* object) {
  if (!images_here()) return no_such_array("hipTexObjectGetResourceDesc");
  const ApiCall api("hipTexObjectGetResourceDesc");
  if (!out || !object) return record(state(), hipErrorInvalidValue);
  Object o;
  if (const hipError_t e = object_of(object, true, &o); e != hipSuccess) return record(state(), e);
  *out = resource_to_driver(o.res);
  return record(state(), hipSuccess);
}
hipError_t hipTexObjectGetTextureDesc(DrvTextureDesc* out, void* object) {
  if (!images_here()) return no_such_array("hipTexObjectGetTextureDesc");
  const ApiCall api("hipTexObjectGetTextureDesc");
  if (!out || !object) return record(state(), hipErrorInvalidValue);
  Object o;
  if (const hipError_t e = object_of(object, true, &o); e != hipSuccess) return record(state(), e);
  *out = texture_to_driver(o.tex);
  return record(state(), hipSuccess);
}
hipError_t hipTexObjectGetResourceViewDesc(DrvResourceViewDesc* out, void* object) {
  if (!images_here()) return no_such_array("hipTexObjectGetResourceViewDesc");
  const ApiCall api("hipTexObjectGetResourceViewDesc");
  if (!out || !object) return record(state(), hipErrorInvalidValue);
  Object o;
  if (const hipError_t e = object_of(object, true, &o); e != hipSuccess) return record(state(), e);
  *out = DrvResourceViewDesc{o.view, {}};
  return record(state(), hipSuccess);
}

// A surface: an array made for loads and stores (or with no flags at all),
// read and written in its own format.
hipError_t hipCreateSurfaceObject(void** object, const ResourceDesc* res) {
  if (!images_here()) return no_images("hipCreateSurfaceObject");
  const ApiCall api("hipCreateSurfaceObject");
  if (!object || !res || res->resType != 0 || !res->res.array.array) return record(state(), hipErrorInvalidValue);
  const HipArray* a = find_array(res->res.array.array);
  if (!a) return record(state(), hipErrorInvalidValue);
  if (a->flags != 0 && !(a->flags & kArraySurfaceLoadStore)) return record(state(), hipErrorInvalidValue);
  im::Gen g;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    g = image_gen(device(s)->profile());
  }
  uint32_t words[kObjectWords] = {};
  array_resource(*a, a->format, a->channels, false, false, g, words);
  Object o;
  o.texture = false;
  o.res = *res;
  return record(state(), place_object(words, o, object));
}
hipError_t hipDestroySurfaceObject(void* object) {
  if (!images_here()) return destroying("hipDestroySurfaceObject", reinterpret_cast<uint64_t>(object));
  const ApiCall api("hipDestroySurfaceObject");
  return record(state(), destroy_object(object, false));
}


// ---- Texture references ----------------------------------------------------------
//
// A texture<T, dim, mode> variable is a textureReference: how to sample, the
// format, and the texture object it is bound to. The program's host copy is
// what these calls change; binding makes the object and copies the whole
// reference to the device's copy of the variable, where the kernel reads the
// object from. A reference from hipModuleGetTexRef is the runtime's own,
// standing for a module's variable. Each check and answer is ROCm 7.1's
// HIP's -- including its getters that write their value and still answer
// hipErrorInvalidValue.

namespace {
struct TextureReference {
  int normalized;
  int readMode;
  int filterMode;
  int addressMode[3];
  ChannelFormatDesc channelDesc;
  int sRGB;
  unsigned maxAnisotropy;
  int mipmapFilterMode;
  float mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp;
  void* textureObject;
  int numChannels;
  int format;   // hipArray_Format
};
static_assert(sizeof(TextureReference) == 88, "textureReference's layout");

// hipModuleGetTexRef's references: the module and variable each stands for.
std::map<const TextureReference*, std::pair<const void*, std::string>> g_module_refs;   // under s.mutex
std::vector<std::unique_ptr<TextureReference>> g_module_ref_storage;                      // under s.mutex

// Where a reference's device copy is: a variable the program registered
// (`module_ref` false), or one of a loaded module's.
hipError_t reference_on_device(const TextureReference* t, bool module_ref, uint64_t* address) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  size_t size = 0;
  if (!module_ref) return symbol_on(s, t, address, &size);
  auto it = g_module_refs.find(t);
  if (it == g_module_refs.end()) return hipErrorNotFound;
  for (const auto& m : s.modules)
    if (m.get() == it->second.first) {
      const vgpu::amd::GlobalVar* g = vgpu::amd::find_global(m->object, it->second.second);
      if (!g) return hipErrorNotFound;
      *address = m->globals + g->offset;
      return hipSuccess;
    }
  return hipErrorNotFound;
}
hipError_t publish_reference(const TextureReference* t, uint64_t address) {
  return hipMemcpy(reinterpret_cast<void*>(address), t, sizeof *t, hipMemcpyHostToDevice);
}

TextureDesc reference_desc(const TextureReference* t) {
  TextureDesc d{};
  for (int k = 0; k < 3; ++k) d.addressMode[k] = t->addressMode[k];
  d.filterMode = t->filterMode;
  d.readMode = t->readMode;
  d.sRGB = t->sRGB;
  d.normalizedCoords = t->normalized;
  d.maxAnisotropy = t->maxAnisotropy;
  d.mipmapFilterMode = t->mipmapFilterMode;
  d.mipmapLevelBias = t->mipmapLevelBias;
  d.minMipmapLevelClamp = t->minMipmapLevelClamp;
  d.maxMipmapLevelClamp = t->maxMipmapLevelClamp;
  return d;
}
// The view an array is bound through: its size, in the format the binding
// names (hipResourceViewFormat for 1, 2 or 4 channels of 8, 16 or 32 bits).
ResourceViewDesc array_view(const HipArray& a, const ChannelFormatDesc& desc) {
  ResourceViewDesc v{};
  const int format = array_format(desc);
  const unsigned channels = channel_count(desc);
  static const int kFormats[] = {kU8, kS8, kU16, kS16, kU32, kS32, kHalf, kFloat};
  for (int i = 0; i < 8; ++i)
    if (kFormats[i] == format && (channels == 1 || channels == 2 || channels == 4))
      v.format = 1 + 3 * i + (channels == 1 ? 0 : channels == 2 ? 1 : 2);
  v.width = a.width, v.height = a.height, v.depth = a.depth;
  return v;
}
// Replaces the object a reference is bound to.
hipError_t rebind(TextureReference* t, const ResourceDesc& res, const ResourceViewDesc* view) {
  if (const hipError_t e = destroy_object(t->textureObject, true); e != hipSuccess) return e;
  t->textureObject = nullptr;
  const TextureDesc d = reference_desc(t);
  return create_texture(&t->textureObject, &res, &d, view);
}
// The offset from an address up to the next image alignment, which a
// binding takes the base down by; with none to say it in, it has to be 0.
hipError_t alignment_offset(size_t* offset, const void* p) {
  const uint64_t a = reinterpret_cast<uint64_t>(p);
  const size_t off = (kImageAlignment - a % kImageAlignment) % kImageAlignment;
  if (off && !offset) return hipErrorInvalidValue;
  if (offset) *offset = off;
  return hipSuccess;
}
}  // namespace

hipError_t hipGetTextureReference(const TextureReference** texref, const void* symbol) {
  if (!images_here()) return no_images("hipGetTextureReference");
  const ApiCall api("hipGetTextureReference");
  if (!texref) return record(state(), hipErrorInvalidValue);
  *texref = static_cast<const TextureReference*>(symbol);
  return record(state(), hipSuccess);
}

hipError_t hipModuleGetTexRef(TextureReference** texref, hipModule_t module, const char* name) {
  if (!images_here()) return no_images("hipModuleGetTexRef");
  const ApiCall api("hipModuleGetTexRef");
  if (!texref || !name || !*name) return record(state(), hipErrorInvalidValue);
  if (!module) return record(state(), hipErrorInvalidResourceHandle);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const Module* m = nullptr;
  for (const auto& owned : s.modules)
    if (owned.get() == reinterpret_cast<const Module*>(module)) m = owned.get();
  const vgpu::amd::GlobalVar* g = m ? vgpu::amd::find_global(m->object, name) : nullptr;
  if (!g || g->size != sizeof(TextureReference))
    return record(s, fail(hipErrorNotFound, std::string("the module has no texture named ") + name));
  // As texture<char>() makes one, then read as normalized floats, one
  // float channel.
  auto t = std::make_unique<TextureReference>();
  std::memset(t.get(), 0, sizeof *t);
  t->addressMode[0] = t->addressMode[1] = t->addressMode[2] = 1;   // clamp
  t->channelDesc = {8, 0, 0, 0, 0};
  t->readMode = 1;
  t->format = kFloat;
  t->numChannels = 1;
  *texref = t.get();
  g_module_refs[t.get()] = {m, name};
  g_module_ref_storage.push_back(std::move(t));
  return record(s, hipSuccess);
}

hipError_t hipBindTexture(size_t* offset, TextureReference* texref, const void* ptr, const ChannelFormatDesc* desc,
                          size_t size) {
  if (!images_here()) return no_images("hipBindTexture");
  const ApiCall api("hipBindTexture");
  uint64_t at = 0;
  if (const hipError_t e = reference_on_device(texref, false, &at); e != hipSuccess) return record(state(), e);
  if (!texref) return record(state(), hipErrorUnknown);
  if (!ptr) return record(state(), hipErrorNotFound);
  if (!desc) return record(state(), hipErrorInvalidValue);
  if (const hipError_t e = destroy_object(texref->textureObject, true); e != hipSuccess) return record(state(), e);
  texref->textureObject = nullptr;
  if (const hipError_t e = alignment_offset(offset, ptr); e != hipSuccess) return record(state(), e);
  ResourceDesc res{};
  res.resType = 2;
  res.res.linear.devPtr = static_cast<char*>(const_cast<void*>(ptr)) - (offset ? *offset : 0);
  res.res.linear.desc = *desc;
  res.res.linear.sizeInBytes = size;
  if (const hipError_t e = rebind(texref, res, nullptr); e != hipSuccess) return record(state(), e);
  return record(state(), publish_reference(texref, at));
}

hipError_t hipBindTexture2D(size_t* offset, TextureReference* texref, const void* ptr, const ChannelFormatDesc* desc,
                            size_t width, size_t height, size_t pitch) {
  if (!images_here()) return no_images("hipBindTexture2D");
  const ApiCall api("hipBindTexture2D");
  if (!texref) return record(state(), hipErrorUnknown);
  if (!ptr) return record(state(), hipErrorNotFound);
  uint64_t at = 0;
  if (const hipError_t e = reference_on_device(texref, false, &at); e != hipSuccess) return record(state(), e);
  if (!desc || !pitch) return record(state(), hipErrorInvalidValue);
  if (const hipError_t e = destroy_object(texref->textureObject, true); e != hipSuccess) return record(state(), e);
  texref->textureObject = nullptr;
  if (const hipError_t e = alignment_offset(offset, ptr); e != hipSuccess) return record(state(), e);
  ResourceDesc res{};
  res.resType = 3;
  res.res.pitch2D.devPtr = static_cast<char*>(const_cast<void*>(ptr)) - (offset ? *offset : 0);
  res.res.pitch2D.desc = *desc;
  res.res.pitch2D.width = width;
  res.res.pitch2D.height = height;
  res.res.pitch2D.pitchInBytes = pitch;
  if (const hipError_t e = rebind(texref, res, nullptr); e != hipSuccess) return record(state(), e);
  return record(state(), publish_reference(texref, at));
}

hipError_t hipBindTextureToArray(TextureReference* texref, const void* array, const ChannelFormatDesc* desc) {
  if (!images_here()) return no_images("hipBindTextureToArray");
  const ApiCall api("hipBindTextureToArray");
  if (!texref || !array || !desc) return record(state(), hipErrorInvalidValue);
  uint64_t at = 0;
  if (const hipError_t e = reference_on_device(texref, false, &at); e != hipSuccess) return record(state(), e);
  const HipArray* a = find_array(array);
  if (!a) return record(state(), hipErrorInvalidValue);
  ResourceDesc res{};
  res.resType = 0;
  res.res.array.array = const_cast<void*>(array);
  const ResourceViewDesc view = array_view(*a, *desc);
  if (const hipError_t e = rebind(texref, res, &view); e != hipSuccess) return record(state(), e);
  return record(state(), publish_reference(texref, at));
}

// A mipmapped array: there are none to bind (see hipMallocMipmappedArray).
hipError_t hipBindTextureToMipmappedArray(TextureReference* texref, const void* array, const ChannelFormatDesc* desc) {
  if (!images_here()) return no_images("hipBindTextureToMipmappedArray");
  const ApiCall api("hipBindTextureToMipmappedArray");
  uint64_t at = 0;
  if (const hipError_t e = reference_on_device(texref, false, &at); e != hipSuccess) return record(state(), e);
  if (!texref || !array || !desc) return record(state(), hipErrorInvalidValue);
  return record(state(), hipErrorNotSupported);
}

// No texture is no value to unbind; one is unbound by destroying its
// object (the device's copy keeps the stale handle, as ROCm's HIP leaves it).
hipError_t hipUnbindTexture(TextureReference* texref) {
  if (!texref) return record(state(), hipErrorInvalidValue);
  if (!images_here()) return no_images("hipUnbindTexture");
  const ApiCall api("hipUnbindTexture");
  if (const hipError_t e = destroy_object(texref->textureObject, true); e != hipSuccess) return record(state(), e);
  texref->textureObject = nullptr;
  return record(state(), hipSuccess);
}

hipError_t hipGetTextureAlignmentOffset(size_t* offset, const TextureReference* texref) {
  if (!images_here()) return no_images("hipGetTextureAlignmentOffset");
  const ApiCall api("hipGetTextureAlignmentOffset");
  if (!texref) return record(state(), hipErrorInvalidTexture);
  if (!offset) return record(state(), hipErrorInvalidValue);
  *offset = 0;
  return record(state(), hipSuccess);
}

// The driver API's form: a module's reference, bound to memory or an array.
hipError_t hipTexRefSetAddress(size_t* offset, TextureReference* texref, void* ptr, size_t bytes) {
  if (!images_here()) return no_images("hipTexRefSetAddress");
  const ApiCall api("hipTexRefSetAddress");
  if (!texref) return record(state(), hipErrorInvalidValue);
  uint64_t at = 0;
  if (const hipError_t e = reference_on_device(texref, true, &at); e != hipSuccess) return record(state(), e);
  if (const hipError_t e = destroy_object(texref->textureObject, true); e != hipSuccess) return record(state(), e);
  texref->textureObject = nullptr;
  if (const hipError_t e = alignment_offset(offset, ptr); e != hipSuccess) return record(state(), e);
  ResourceDesc res{};
  res.resType = 2;
  res.res.linear.devPtr = static_cast<char*>(ptr) - (offset ? *offset : 0);
  res.res.linear.desc = channel_desc(texref->numChannels, texref->format);
  res.res.linear.sizeInBytes = bytes;
  if (const hipError_t e = rebind(texref, res, nullptr); e != hipSuccess) return record(state(), e);
  return record(state(), publish_reference(texref, at));
}
hipError_t hipTexRefSetAddress2D(TextureReference* texref, const ArrayDescriptor* desc, void* ptr, size_t pitch) {
  if (!images_here()) return no_images("hipTexRefSetAddress2D");
  const ApiCall api("hipTexRefSetAddress2D");
  if (!texref || !desc) return record(state(), hipErrorInvalidValue);
  uint64_t at = 0;
  if (const hipError_t e = reference_on_device(texref, true, &at); e != hipSuccess) return record(state(), e);
  ResourceDesc res{};
  res.resType = 3;
  res.res.pitch2D.devPtr = ptr;
  res.res.pitch2D.desc = channel_desc(desc->NumChannels, desc->Format);
  res.res.pitch2D.width = desc->Width;
  res.res.pitch2D.height = desc->Height;
  res.res.pitch2D.pitchInBytes = pitch;
  if (const hipError_t e = rebind(texref, res, nullptr); e != hipSuccess) return record(state(), e);
  return record(state(), publish_reference(texref, at));
}
hipError_t hipTexRefSetArray(TextureReference* texref, const void* array, unsigned flags) {
  if (!images_here()) return no_images("hipTexRefSetArray");
  const ApiCall api("hipTexRefSetArray");
  if (!texref || !array) return record(state(), hipErrorInvalidValue);
  if (flags != 1 /* HIP_TRSA_OVERRIDE_FORMAT */) return record(state(), hipErrorInvalidValue);
  uint64_t at = 0;
  if (const hipError_t e = reference_on_device(texref, true, &at); e != hipSuccess) return record(state(), e);
  const HipArray* a = find_array(array);
  if (!a) return record(state(), hipErrorInvalidValue);
  texref->format = a->format;
  texref->numChannels = static_cast<int>(a->channels);
  ResourceDesc res{};
  res.resType = 0;
  res.res.array.array = const_cast<void*>(array);
  const ResourceViewDesc view = array_view(*a, channel_desc(a->channels, a->format));
  if (const hipError_t e = rebind(texref, res, &view); e != hipSuccess) return record(state(), e);
  return record(state(), publish_reference(texref, at));
}
hipError_t hipTexRefSetMipmappedArray(TextureReference* texref, void* array, unsigned flags) {
  if (!images_here()) return no_images("hipTexRefSetMipmappedArray");
  const ApiCall api("hipTexRefSetMipmappedArray");
  if (!texref || !array || flags != 1) return record(state(), hipErrorInvalidValue);
  uint64_t at = 0;
  if (const hipError_t e = reference_on_device(texref, true, &at); e != hipSuccess) return record(state(), e);
  return record(state(), hipErrorNotSupported);
}

// What a reference is bound to.
namespace {
hipError_t bound_resource(const TextureReference* t, ResourceDesc* res) {
  if (!t->textureObject) return hipErrorInvalidValue;
  Object o;
  if (const hipError_t e = object_of(t->textureObject, true, &o); e != hipSuccess) return e;
  *res = o.res;
  return hipSuccess;
}
}  // namespace
hipError_t hipTexRefGetAddress(void** ptr, const TextureReference* texref) {
  if (!images_here()) return no_images("hipTexRefGetAddress");
  const ApiCall api("hipTexRefGetAddress");
  if (!ptr || !texref) return record(state(), hipErrorInvalidValue);
  ResourceDesc res{};
  if (const hipError_t e = bound_resource(texref, &res); e != hipSuccess) return record(state(), e);
  if (res.resType == 2) *ptr = res.res.linear.devPtr;
  else if (res.resType == 3) *ptr = res.res.pitch2D.devPtr;
  else return record(state(), hipErrorInvalidValue);
  return record(state(), hipSuccess);
}
hipError_t hipTexRefGetArray(void** array, const TextureReference* texref) {
  if (!images_here()) return no_images("hipTexRefGetArray");
  const ApiCall api("hipTexRefGetArray");
  if (!array || !texref) return record(state(), hipErrorInvalidValue);
  ResourceDesc res{};
  if (const hipError_t e = bound_resource(texref, &res); e != hipSuccess) return record(state(), e);
  if (res.resType != 0) return record(state(), hipErrorInvalidValue);
  *array = res.res.array.array;
  return record(state(), hipSuccess);
}
hipError_t hipTexRefGetMipMappedArray(void** array, const TextureReference* texref) {
  if (!images_here()) return no_images("hipTexRefGetMipMappedArray");
  const ApiCall api("hipTexRefGetMipMappedArray");
  if (!array || !texref) return record(state(), hipErrorInvalidValue);
  ResourceDesc res{};
  if (const hipError_t e = bound_resource(texref, &res); e != hipSuccess) return record(state(), e);
  return record(state(), hipErrorInvalidValue);   // nothing is ever bound to a mipmapped array
}

// The reference's own fields.
#define VGPU_TEXREF_CALL(name, args, cond, body, answer)                  \
  hipError_t name args {                                                    \
    if (!images_here()) return no_images(#name);                            \
    const ApiCall api(#name);                                               \
    if (cond) return record(state(), hipErrorInvalidValue);                 \
    body;                                                                   \
    return record(state(), answer);                                         \
  }
VGPU_TEXREF_CALL(hipTexRefSetFormat, (TextureReference* t, int format, int channels), !t,
                 (t->format = format, t->numChannels = channels), hipSuccess)
VGPU_TEXREF_CALL(hipTexRefGetFormat, (int* format, int* channels, const TextureReference* t),
                 !format || !channels || !t, (*format = t->format, *channels = t->numChannels), hipSuccess)
VGPU_TEXREF_CALL(hipTexRefSetFilterMode, (TextureReference* t, int mode), !t, t->filterMode = mode, hipSuccess)
VGPU_TEXREF_CALL(hipTexRefGetFilterMode, (int* mode, const TextureReference* t), !mode || !t,
                 *mode = t->filterMode, hipSuccess)
VGPU_TEXREF_CALL(hipTexRefSetMaxAnisotropy, (TextureReference* t, unsigned ratio), !t, t->maxAnisotropy = ratio,
                 hipSuccess)
VGPU_TEXREF_CALL(hipTexRefGetMaxAnisotropy, (int* ratio, const TextureReference* t), !ratio || !t,
                 *ratio = static_cast<int>(t->maxAnisotropy), hipSuccess)
VGPU_TEXREF_CALL(hipTexRefSetMipmapFilterMode, (TextureReference* t, int mode), !t, t->mipmapFilterMode = mode,
                 hipSuccess)
VGPU_TEXREF_CALL(hipTexRefGetMipmapFilterMode, (int* mode, const TextureReference* t), !mode || !t,
                 *mode = t->mipmapFilterMode, hipErrorInvalidValue)
VGPU_TEXREF_CALL(hipTexRefSetMipmapLevelBias, (TextureReference* t, float bias), !t, t->mipmapLevelBias = bias,
                 hipSuccess)
VGPU_TEXREF_CALL(hipTexRefGetMipmapLevelBias, (float* bias, const TextureReference* t), !bias || !t,
                 *bias = t->mipmapLevelBias, hipErrorInvalidValue)
VGPU_TEXREF_CALL(hipTexRefSetMipmapLevelClamp, (TextureReference* t, float lo, float hi), !t,
                 (t->minMipmapLevelClamp = lo, t->maxMipmapLevelClamp = hi), hipSuccess)
VGPU_TEXREF_CALL(hipTexRefGetMipmapLevelClamp, (float* lo, float* hi, const TextureReference* t), !lo || !hi || !t,
                 (*lo = t->minMipmapLevelClamp, *hi = t->maxMipmapLevelClamp), hipErrorInvalidValue)
// The reference has no border color to keep: ROCm's HIP answers success
// and touches nothing.
VGPU_TEXREF_CALL(hipTexRefSetBorderColor, (TextureReference* t, float* color), !t || !color, (void)0, hipSuccess)
VGPU_TEXREF_CALL(hipTexRefGetBorderColor, (float* color, const TextureReference* t), !color || !t, (void)0,
                 hipSuccess)
// Flags: read as integers (1), normalized coordinates (2), sRGB (0x10).
VGPU_TEXREF_CALL(hipTexRefSetFlags, (TextureReference* t, unsigned flags), !t,
                 (t->readMode = (flags & 1) ? 0 : 1, t->normalized = (flags & 2) ? 1 : 0,
                  t->sRGB = (flags & 0x10) ? 1 : 0),
                 hipSuccess)
VGPU_TEXREF_CALL(hipTexRefGetFlags, (unsigned* flags, const TextureReference* t), !flags || !t,
                 *flags = (t->readMode == 0 ? 1u : 0u) | (t->normalized == 1 ? 2u : 0u) | (t->sRGB == 1 ? 0x10u : 0u),
                 hipSuccess)
#undef VGPU_TEXREF_CALL
// An axis's address mode: three to set, two to ask about (as ROCm's HIP has it).
hipError_t hipTexRefSetAddressMode(TextureReference* t, int dim, int mode) {
  if (!images_here()) return no_images("hipTexRefSetAddressMode");
  const ApiCall api("hipTexRefSetAddressMode");
  if (!t || dim < 0 || dim > 2) return record(state(), hipErrorInvalidValue);
  t->addressMode[dim] = mode;
  return record(state(), hipSuccess);
}
hipError_t hipTexRefGetAddressMode(int* mode, const TextureReference* t, int dim) {
  if (!images_here()) return no_images("hipTexRefGetAddressMode");
  const ApiCall api("hipTexRefGetAddressMode");
  if (!mode || !t || (dim != 0 && dim != 1)) return record(state(), hipErrorInvalidValue);
  *mode = t->addressMode[dim];
  return record(state(), hipSuccess);
}

// ---- What stays refused ------------------------------------------------------------
//
// Mipmaps (which ROCm's HIP on Linux does not have) and graphics interop.
#define VGPU_NO_IMAGES(name) \
  hipError_t name() { return no_images(#name); }
VGPU_NO_IMAGES(hipMipmappedArrayCreate)
VGPU_NO_IMAGES(hipGetMipmappedArrayLevel)
VGPU_NO_IMAGES(hipMipmappedArrayGetLevel)
VGPU_NO_IMAGES(hipMemMapArrayAsync)
VGPU_NO_IMAGES(hipGraphicsSubResourceGetMappedArray)
#undef VGPU_NO_IMAGES
#define VGPU_FREEING_NOTHING(name) \
  hipError_t name() { return freeing_nothing(#name); }
VGPU_FREEING_NOTHING(hipFreeMipmappedArray)
VGPU_FREEING_NOTHING(hipMipmappedArrayDestroy)
#undef VGPU_FREEING_NOTHING
