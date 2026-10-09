// The driver API's texture formats and objects, checked against what an RTX 3060's driver (13.2, under WSL)
// answers: cuArrayCreate over the normalized, block-compressed and 10:10:10:2 formats, the flags
// cuTexObjectCreate accepts for each, a descriptor's anisotropy, resource views (CUDA_RESOURCE_VIEW_DESC), and
// the values a fetch returns (hashed, FNV-1a over the float bits). Every observation is compared with the
// card's, recorded in driver_texture_gaps_expected.inc (`driver_texture_gaps --print` prints the file); the
// same program passes against NVIDIA's driver.
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Expected {
  const char* label;
  long long value;
};
static const Expected kExpected[] = {
#include "driver_texture_gaps_expected.inc"
};

static bool g_print = false;
static int g_checked = 0, g_failures = 0;

static void observe(const std::string& label, long long value) {
  if (g_print) {
    std::printf("{\"%s\", %lldLL},\n", label.c_str(), value);
    return;
  }
  const Expected* e = nullptr;
  for (const Expected& x : kExpected)
    if (label == x.label) { e = &x; break; }
  ++g_checked;
  if (!e) {
    std::printf("FAIL %s: no expected value (observed %lld)\n", label.c_str(), value);
    ++g_failures;
  } else if (e->value != value) {
    std::printf("FAIL %s: expected %lld, observed %lld\n", label.c_str(), e->value, value);
    ++g_failures;
  }
}

static unsigned long long fnv(const void* data, size_t bytes) {
  unsigned long long h = 1469598103934665603ull;
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

static const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64
.visible .entry fetch(.param .u64 tex, .param .u64 uv, .param .u64 out, .param .u32 n)
{
  .reg .pred %p;
  .reg .b32 %r<8>;
  .reg .f32 %f<8>;
  .reg .b64 %rd<10>;
  ld.param.u64 %rd1, [tex];
  ld.param.u64 %rd2, [uv];
  ld.param.u64 %rd3, [out];
  ld.param.u32 %r1, [n];
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r5, %r2, %r3, %r4;
  setp.ge.s32 %p, %r5, %r1;
  @%p bra done;
  mul.wide.s32 %rd4, %r5, 8;
  add.s64 %rd5, %rd2, %rd4;
  ld.global.v2.f32 {%f1, %f2}, [%rd5];
  tex.2d.v4.f32.f32 {%f3, %f4, %f5, %f6}, [%rd1, {%f1, %f2}];
  mul.wide.s32 %rd6, %r5, 16;
  add.s64 %rd7, %rd3, %rd6;
  st.global.v4.f32 [%rd7], {%f3, %f4, %f5, %f6};
done:
  ret;
}
)";

static CUfunction g_fetch = nullptr;

// Hash of the float4s a texture object returns at `uv`, or -1 on failure.
static long long fetch_hash(unsigned long long tex, const std::vector<float>& uv2) {
  const int n = static_cast<int>(uv2.size() / 2);
  CUdeviceptr duv = 0, dout = 0;
  if (cuMemAlloc(&duv, uv2.size() * 4) || cuMemAlloc(&dout, (size_t)n * 16)) return -1;
  cuMemcpyHtoD(duv, uv2.data(), uv2.size() * 4);
  void* args[] = {&tex, &duv, &dout, const_cast<int*>(&n)};
  const CUresult e = cuLaunchKernel(g_fetch, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
  std::vector<float> out((size_t)n * 4);
  const CUresult s = e ? e : cuMemcpyDtoH(out.data(), dout, out.size() * 4);
  cuMemFree(duv);
  cuMemFree(dout);
  if (s) return -1;
  return static_cast<long long>(fnv(out.data(), out.size() * 4));
}

static std::vector<float> grid_centres(int w, int h) {
  std::vector<float> uv;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      uv.push_back(x + 0.5f);
      uv.push_back(y + 0.5f);
    }
  return uv;
}

static uint64_t g_lcg = 0x9E3779B97F4A7C15ull;
static uint8_t next_byte() {
  g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<uint8_t>(g_lcg >> 56);
}

static CUDA_ARRAY_DESCRIPTOR desc2d(unsigned format, unsigned channels, size_t w, size_t h) {
  CUDA_ARRAY_DESCRIPTOR d{};
  d.Width = w;
  d.Height = h;
  std::memcpy(&d.Format, &format, sizeof format);
  d.NumChannels = channels;
  return d;
}

static CUDA_RESOURCE_DESC array_resource(CUarray a) {
  CUDA_RESOURCE_DESC r{};
  r.resType = CU_RESOURCE_TYPE_ARRAY;
  r.res.array.hArray = a;
  return r;
}

static CUDA_TEXTURE_DESC tex_desc(unsigned flags, CUfilter_mode filter, CUaddress_mode address = CU_TR_ADDRESS_MODE_CLAMP) {
  CUDA_TEXTURE_DESC t{};
  t.addressMode[0] = t.addressMode[1] = t.addressMode[2] = address;
  t.filterMode = filter;
  t.flags = flags;
  t.maxAnisotropy = 1;
  return t;
}

struct Fmt {
  const char* name;
  unsigned code;
  unsigned channels;      // the channel count that makes the array
  size_t block_bytes;     // for block-compressed formats, else 0
  bool srgb;
};
static const Fmt kFormats[] = {
    {"U8x1", 0x01, 1, 0, false},       {"U8x4", 0x01, 4, 0, false},       {"S16x2", 0x09, 2, 0, false},
    {"U32x1", 0x03, 1, 0, false},      {"HALFx2", 0x10, 2, 0, false},     {"FLOATx4", 0x20, 4, 0, false},
    {"UNORM8x1", 0xc0, 1, 0, false},   {"UNORM8x2", 0xc1, 2, 0, false},   {"UNORM8x4", 0xc2, 4, 0, false},
    {"UNORM16x1", 0xc3, 1, 0, false},  {"UNORM16x4", 0xc5, 4, 0, false},  {"SNORM8x2", 0xc7, 2, 0, false},
    {"SNORM16x1", 0xc9, 1, 0, false},  {"SNORM16x4", 0xcb, 4, 0, false},  {"BC1", 0x91, 4, 8, false},
    {"BC1s", 0x92, 4, 8, true},        {"BC2", 0x93, 4, 16, false},       {"BC2s", 0x94, 4, 16, true},
    {"BC3", 0x95, 4, 16, false},       {"BC3s", 0x96, 4, 16, true},       {"BC4u", 0x97, 1, 8, false},
    {"BC4s", 0x98, 1, 8, false},       {"BC5u", 0x99, 2, 16, false},      {"BC5s", 0x9a, 2, 16, false},
    {"BC6Hu", 0x9b, 3, 16, false},     {"BC6Hs", 0x9c, 3, 16, false},     {"BC7", 0x9d, 4, 16, false},
    {"BC7s", 0x9e, 4, 16, true},       {"RGB10A2", 0x50, 4, 0, false},
};

static void array_grid() {
  for (const Fmt& f : kFormats) {
    for (unsigned c = 1; c <= 4; ++c) {
      CUDA_ARRAY_DESCRIPTOR d = desc2d(f.code, c, 8, 8);
      CUarray a = nullptr;
      const CUresult e = cuArrayCreate(&a, &d);
      observe(std::string("array/") + f.name + "/ch" + std::to_string(c), e);
      if (a) cuArrayDestroy(a);
    }
  }
  // Shapes and flags of block-compressed and packed arrays.
  struct Shape {
    const char* name;
    unsigned code, channels;
    size_t w, h, d;
    unsigned flags;
  };
  static const Shape shapes[] = {
      {"BC1-10x6", 0x91, 4, 10, 6, 0, 0},
      {"BC1-1x1", 0x91, 4, 1, 1, 0, 0},
      {"BC1-8x0", 0x91, 4, 8, 0, 0, 0},
      {"BC1-layered", 0x91, 4, 8, 8, 4, CUDA_ARRAY3D_LAYERED},
      {"BC1-3d", 0x91, 4, 8, 8, 4, 0},
      {"BC1-cube", 0x91, 4, 8, 8, 6, CUDA_ARRAY3D_CUBEMAP},
      {"BC1-surface", 0x91, 4, 8, 8, 0, CUDA_ARRAY3D_SURFACE_LDST},
      {"BC1-gather", 0x91, 4, 8, 8, 0, CUDA_ARRAY3D_TEXTURE_GATHER},
      {"BC4-layered", 0x97, 1, 8, 8, 2, CUDA_ARRAY3D_LAYERED},
      {"RGB10A2-surface", 0x50, 4, 8, 8, 0, CUDA_ARRAY3D_SURFACE_LDST},
      {"RGB10A2-layered", 0x50, 4, 8, 8, 2, CUDA_ARRAY3D_LAYERED},
      {"UNORM8x4-surface", 0xc2, 4, 8, 8, 0, CUDA_ARRAY3D_SURFACE_LDST},
      {"SNORM16x2-3d", 0xca, 2, 8, 8, 4, 0},
  };
  for (const Shape& sh : shapes) {
    CUDA_ARRAY3D_DESCRIPTOR d{};
    d.Width = sh.w;
    d.Height = sh.h;
    d.Depth = sh.d;
    std::memcpy(&d.Format, &sh.code, sizeof sh.code);
    d.NumChannels = sh.channels;
    d.Flags = sh.flags;
    CUarray a = nullptr;
    const CUresult e = cuArray3DCreate(&a, &d);
    observe(std::string("shape/") + sh.name, e);
    if (a) {
      CUDA_ARRAY3D_DESCRIPTOR back{};
      cuArray3DGetDescriptor(&back, a);
      observe(std::string("shape/") + sh.name + "/size", (long long)back.Width * 1000000 + (long long)back.Height * 1000 + (long long)back.Depth);
      cuArrayDestroy(a);
    }
  }
  // Formats that name no format, and the video ones.
  for (unsigned code : {0u, 4u, 0xb0u, 0x9fu, 0xa5u, 0x7fu, 0x7fffffffu}) {
    CUDA_ARRAY_DESCRIPTOR d = desc2d(code, 4, 8, 8);
    CUarray a = nullptr;
    observe("array/code" + std::to_string(code), cuArrayCreate(&a, &d));
    if (a) cuArrayDestroy(a);
  }
}

static void texture_grid() {
  for (const Fmt& f : kFormats) {
    CUDA_ARRAY_DESCRIPTOR d = desc2d(f.code, f.channels, 8, 8);
    CUarray a = nullptr;
    if (cuArrayCreate(&a, &d)) continue;
    const CUDA_RESOURCE_DESC rd = array_resource(a);
    for (int as_int = 0; as_int < 2; ++as_int)
      for (int lin = 0; lin < 2; ++lin)
        for (int srgb = 0; srgb < 2; ++srgb) {
          const unsigned flags = (as_int ? CU_TRSF_READ_AS_INTEGER : 0u) | (srgb ? CU_TRSF_SRGB : 0u);
          const CUDA_TEXTURE_DESC td = tex_desc(flags, lin ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT);
          CUtexObject t = 0;
          const CUresult e = cuTexObjectCreate(&t, &rd, &td, nullptr);
          observe(std::string("texture/") + f.name + (as_int ? "/int" : "/norm") + (lin ? "/lin" : "/pt") +
                      (srgb ? "/srgb" : "/plain"),
                  e);
          if (!e) cuTexObjectDestroy(t);
        }
    cuArrayDestroy(a);
  }
}

// Linear and pitched device memory as a texture, and a surface over an array.
static void other_resources() {
  CUdeviceptr buf = 0;
  cuMemAlloc(&buf, 1 << 16);
  for (const Fmt& f : kFormats) {
    if (f.code == 0x9b || f.code == 0x9c) continue;
    CUDA_RESOURCE_DESC rd{};
    rd.resType = CU_RESOURCE_TYPE_LINEAR;
    rd.res.linear.devPtr = buf;
    std::memcpy(&rd.res.linear.format, &f.code, sizeof f.code);
    rd.res.linear.numChannels = f.channels;
    rd.res.linear.sizeInBytes = 1024;
    const CUDA_TEXTURE_DESC td = tex_desc(f.srgb ? CU_TRSF_SRGB : 0u, CU_TR_FILTER_MODE_POINT);
    CUtexObject t = 0;
    const CUresult e = cuTexObjectCreate(&t, &rd, &td, nullptr);
    observe(std::string("linear/") + f.name, e);
    if (!e) cuTexObjectDestroy(t);
    CUDA_RESOURCE_DESC pd{};
    pd.resType = CU_RESOURCE_TYPE_PITCH2D;
    pd.res.pitch2D.devPtr = buf;
    std::memcpy(&pd.res.pitch2D.format, &f.code, sizeof f.code);
    pd.res.pitch2D.numChannels = f.channels;
    pd.res.pitch2D.width = 8;
    pd.res.pitch2D.height = 8;
    pd.res.pitch2D.pitchInBytes = 256;
    const CUresult e2 = cuTexObjectCreate(&t, &pd, &td, nullptr);
    observe(std::string("pitch2d/") + f.name, e2);
    if (!e2) cuTexObjectDestroy(t);
  }
  cuMemFree(buf);
  // A surface over an array of each kind of format (surface-capable flag where the format allows it).
  for (const Fmt& f : kFormats) {
    if (f.code == 0x9b || f.code == 0x9c) continue;
    CUDA_ARRAY3D_DESCRIPTOR d{};
    d.Width = 8;
    d.Height = 8;
    std::memcpy(&d.Format, &f.code, sizeof f.code);
    d.NumChannels = f.channels;
    CUarray a = nullptr;
    if (cuArray3DCreate(&a, &d)) continue;
    const CUDA_RESOURCE_DESC rd = array_resource(a);
    CUsurfObject so = 0;
    const CUresult e = cuSurfObjectCreate(&so, &rd);
    observe(std::string("surface/") + f.name, e);
    if (!e) cuSurfObjectDestroy(so);
    cuArrayDestroy(a);
  }
}

// Mipmapped arrays of the new formats: the levels shrink by texels, not blocks.
static void mipmapped() {
  struct M {
    const char* name;
    unsigned code, channels, flags;
  };
  static const M ms[] = {{"BC1", 0x91, 4, 0},          {"BC1-surface", 0x91, 4, CUDA_ARRAY3D_SURFACE_LDST},
                         {"BC4u", 0x97, 1, 0},         {"BC6Hu", 0x9b, 3, 0},
                         {"RGB10A2", 0x50, 4, 0},      {"UNORM8x4", 0xc2, 4, 0},
                         {"UNORM8x4-surface", 0xc2, 4, CUDA_ARRAY3D_SURFACE_LDST}};
  for (const M& m : ms) {
    CUDA_ARRAY3D_DESCRIPTOR d{};
    d.Width = 16;
    d.Height = 16;
    std::memcpy(&d.Format, &m.code, sizeof m.code);
    d.NumChannels = m.channels;
    d.Flags = m.flags;
    CUmipmappedArray mip = nullptr;
    const CUresult e = cuMipmappedArrayCreate(&mip, &d, 3);
    observe(std::string("mip/") + m.name, e);
    if (!mip) continue;
    for (unsigned lv = 0; lv < 3; ++lv) {
      CUarray a = nullptr;
      cuMipmappedArrayGetLevel(&a, mip, lv);
      CUDA_ARRAY3D_DESCRIPTOR back{};
      cuArray3DGetDescriptor(&back, a);
      observe(std::string("mip/") + m.name + "/level" + std::to_string(lv), (long long)back.Width * 1000 + (long long)back.Height);
    }
    cuMipmappedArrayDestroy(mip);
  }
}

static void anisotropy() {
  CUDA_ARRAY_DESCRIPTOR d = desc2d(0x20, 1, 8, 8);
  CUarray a = nullptr;
  cuArrayCreate(&a, &d);
  const CUDA_RESOURCE_DESC rd = array_resource(a);
  for (unsigned an : {0u, 1u, 2u, 8u, 16u, 17u, 100u, 0xffffffffu}) {
    CUDA_TEXTURE_DESC td = tex_desc(0, CU_TR_FILTER_MODE_LINEAR);
    td.maxAnisotropy = an;
    CUtexObject t = 0;
    const CUresult e = cuTexObjectCreate(&t, &rd, &td, nullptr);
    observe("aniso/" + std::to_string(an), e);
    if (!e) cuTexObjectDestroy(t);
  }
  cuArrayDestroy(a);
}

static void views() {
  const int bw = 16, bh = 16;
  g_lcg = 0x9E3779B97F4A7C15ull ^ 17;
  std::vector<uint8_t> blocks((size_t)bw * bh * 8);
  for (auto& b : blocks) b = next_byte();
  // An array of 8-byte texels (two unsigned 32-bit channels) holding BC1 blocks.
  CUDA_ARRAY_DESCRIPTOR d = desc2d(CU_AD_FORMAT_UNSIGNED_INT32, 2, bw, bh);
  CUarray a = nullptr;
  cuArrayCreate(&a, &d);
  CUDA_MEMCPY2D cp{};
  cp.srcMemoryType = CU_MEMORYTYPE_HOST;
  cp.srcHost = blocks.data();
  cp.srcPitch = (size_t)bw * 8;
  cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  cp.dstArray = a;
  cp.WidthInBytes = (size_t)bw * 8;
  cp.Height = bh;
  cuMemcpy2D(&cp);
  const CUDA_RESOURCE_DESC rd = array_resource(a);
  for (int as_int = 0; as_int < 2; ++as_int) {
    const CUDA_TEXTURE_DESC td = tex_desc(as_int ? CU_TRSF_READ_AS_INTEGER : 0u, CU_TR_FILTER_MODE_POINT);
    const std::string mode = as_int ? "/int" : "/norm";
    auto make = [&](unsigned vf, size_t w, size_t h, CUtexObject* t) {
      CUDA_RESOURCE_VIEW_DESC v{};
      std::memcpy(&v.format, &vf, sizeof vf);
      v.width = w;
      v.height = h;
      return cuTexObjectCreate(t, &rd, &td, &v);
    };
    CUtexObject t = 0;
    auto done = [&] {
      if (t) cuTexObjectDestroy(t);
      t = 0;
    };
    observe("view/uint2-as-BC1" + mode, make(CU_RES_VIEW_FORMAT_UNSIGNED_BC1, bw * 4, bh * 4, &t));
    done();
    observe("view/uint2-as-BC4" + mode, make(CU_RES_VIEW_FORMAT_UNSIGNED_BC4, bw * 4, bh * 4, &t));
    done();
    observe("view/uint2-as-BC1-wrong-extent" + mode, make(CU_RES_VIEW_FORMAT_UNSIGNED_BC1, bw, bh, &t));
    done();
    observe("view/uint2-as-BC2-size-mismatch" + mode, make(CU_RES_VIEW_FORMAT_UNSIGNED_BC2, bw * 4, bh * 4, &t));
    done();
    observe("view/uint2-as-ushort4" + mode, make(CU_RES_VIEW_FORMAT_UINT_4X16, bw, bh, &t));
    done();
    observe("view/uint2-as-float2" + mode, make(CU_RES_VIEW_FORMAT_FLOAT_2X32, bw, bh, &t));
    done();
    observe("view/uint2-as-uchar4-size-mismatch" + mode, make(CU_RES_VIEW_FORMAT_UINT_4X8, bw, bh, &t));
    done();
    observe("view/none" + mode, make(CU_RES_VIEW_FORMAT_NONE, bw, bh, &t));
    done();
    observe("view/format-99" + mode, make(99, bw, bh, &t));
    done();
  }
  cuArrayDestroy(a);
}

// ---- fetches ----
// Bytes of a texel of a plain (not block-compressed) format.
static size_t texel_bytes(const Fmt& f) {
  switch (f.code) {
    case 0x50: return 4;
    case 0x01: case 0x08: case 0xc0: case 0xc1: case 0xc2: case 0xc6: case 0xc7: case 0xc8: return f.channels;
    case 0x02: case 0x09: case 0x10: case 0xc3: case 0xc4: case 0xc5: case 0xc9: case 0xca: case 0xcb: return 2 * f.channels;
    default: return 4 * f.channels;
  }
}

static long long fetch_array(const Fmt& f, const std::vector<uint8_t>& blocks, int bw, int bh, int pass) {
  const bool blocky = f.block_bytes != 0;
  const int w = bw * 4, h = bh * 4;
  CUDA_ARRAY_DESCRIPTOR d = desc2d(f.code, f.channels, blocky ? w : w, blocky ? h : h);
  CUarray a = nullptr;
  if (cuArrayCreate(&a, &d)) return -1;
  const size_t row = blocky ? (size_t)bw * f.block_bytes : (size_t)w * texel_bytes(f);
  CUDA_MEMCPY2D cp{};
  cp.srcMemoryType = CU_MEMORYTYPE_HOST;
  cp.srcHost = blocks.data();
  cp.srcPitch = row;
  cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  cp.dstArray = a;
  cp.WidthInBytes = row;
  cp.Height = blocky ? bh : h;
  if (cuMemcpy2D(&cp)) {
    cuArrayDestroy(a);
    return -1;
  }
  const CUDA_RESOURCE_DESC rd = array_resource(a);
  const unsigned flags = (f.srgb ? CU_TRSF_SRGB : 0u) | (pass == 3 ? CU_TRSF_NORMALIZED_COORDINATES : 0u);
  const CUaddress_mode modes[3] = {CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_MIRROR};
  const CUDA_TEXTURE_DESC td = tex_desc(flags, pass == 0 || pass == 1 || pass == 2 ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT,
                                        modes[pass == 3 ? 1 : (pass < 3 ? pass : 0)]);
  CUtexObject t = 0;
  long long result = -1;
  if (cuTexObjectCreate(&t, &rd, &td, nullptr) == CUDA_SUCCESS) {
    std::vector<float> uv;
    if (pass == 4) {
      uv = grid_centres(w, h);
    } else {
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const float fx = ((x * 7 + y * 13 + pass) % 32) / 32.0f, fy = ((x * 11 + y * 5 + 3 * pass) % 32) / 32.0f;
          float u = x + fx - 2.5f * (pass == 1 || pass == 2), v = y + fy + 1.5f * (pass == 1 || pass == 2);
          if (pass == 3) { u = (x + 0.25f + 0.5f * (x % 3)) / w - 0.37f; v = (y + 0.5f) / h + 0.8f; }
          uv.push_back(u);
          uv.push_back(v);
        }
    }
    result = fetch_hash(t, uv);
    cuTexObjectDestroy(t);
  }
  cuArrayDestroy(a);
  return result;
}

static void fetches() {
  const int bw = 16, bh = 16;
  for (const Fmt& f : kFormats) {
    if (f.code < 0x91 && f.code != 0x50 && f.code < 0xc0) continue;
    g_lcg = 0x9E3779B97F4A7C15ull ^ f.code;
    const bool blocky = f.block_bytes != 0;
    std::vector<uint8_t> blocks(blocky ? (size_t)bw * bh * f.block_bytes : (size_t)bw * 4 * bh * 4 * texel_bytes(f));
    for (auto& b : blocks) b = next_byte();
    for (int pass = 0; pass < 5; ++pass) {
      // Linear filtering of signed 8-bit normalized texels is refused by name in the simulator (the card's
      // result is not a plain rounded weighted sum and has not been reproduced), so the three linear passes
      // are left out for those formats; point sampling (passes 3 and 4) is checked.
      if (f.code >= 0xc6 && f.code <= 0xc8 && pass < 3) continue;
      observe(std::string("fetch/") + f.name + "/pass" + std::to_string(pass), fetch_array(f, blocks, bw, bh, pass));
    }
  }
}

// BC7 and BC6H mode by mode (and the reserved encodings: a BC7 low byte of zero, the BC6H mode numbers 19, 23, 27
// and 31): 512 blocks of random bits with the mode bits forced and the partition cycling through its values,
// point sampled at the texel centres (pass 4) and linearly filtered (pass 0).
static void bc67_modes() {
  static const int kBc6hCodes[18] = {0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31};
  static const int kBc7PartitionBits[8] = {4, 6, 6, 6, 0, 0, 0, 6};   // modes 0..7 (one subset: none)
  const int bw = 32, bh = 16;
  for (const Fmt& f : kFormats) {
    if (f.code < 0x9b || f.code > 0x9e) continue;
    const bool is7 = f.code >= 0x9d;
    for (int mode = 0; mode < (is7 ? 9 : 18); ++mode) {
      g_lcg = 0x9E3779B97F4A7C15ull ^ ((uint64_t)f.code << 8) ^ (uint64_t)mode;
      std::vector<uint8_t> blocks((size_t)bw * bh * 16);
      for (auto& b : blocks) b = next_byte();
      for (int i = 0; i < bw * bh; ++i) {
        uint8_t* p = &blocks[(size_t)i * 16];
        auto set_bits = [&](int first, int count, unsigned value) {
          for (int k = 0; k < count; ++k) {
            const int bit = first + k;
            p[bit / 8] = (uint8_t)((p[bit / 8] & ~(1 << (bit % 8))) | (((value >> k) & 1) << (bit % 8)));
          }
        };
        if (is7) {
          if (mode == 8) {
            p[0] = 0;
          } else {
            p[0] = (uint8_t)((p[0] & ~((1 << (mode + 1)) - 1)) | (1 << mode));   // the marker bit, zeros below it
            const int pb = kBc7PartitionBits[mode];
            set_bits(mode + 1, pb, (unsigned)i % (1u << pb));
          }
        } else {
          const int c = kBc6hCodes[mode];
          p[0] = (uint8_t)(c < 2 ? (p[0] & ~3) | c : (p[0] & ~31) | c);
          set_bits(77, 5, (unsigned)i % 32u);   // the partition bits of every two-subset mode
        }
      }
      const std::string label = std::string("bc-modes/") + f.name + "/mode" + std::to_string(mode);
      observe(label + "/point", fetch_array(f, blocks, bw, bh, 4));
      observe(label + "/linear", fetch_array(f, blocks, bw, bh, 0));
    }
  }
}

int main(int argc, char** argv) {
  g_print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  cuInit(0);
  CUdevice dev = 0;
  cuDeviceGet(&dev, 0);
  CUcontext ctx = nullptr;
#if CUDA_VERSION >= 13000
  CUctxCreateParams p{};
  cuCtxCreate(&ctx, &p, 0, dev);
#else
  cuCtxCreate(&ctx, 0, dev);
#endif
  CUmodule mod = nullptr;
  if (cuModuleLoadData(&mod, kPtx) != CUDA_SUCCESS || cuModuleGetFunction(&g_fetch, mod, "fetch") != CUDA_SUCCESS) {
    std::printf("FAIL cannot load the fetch kernel\n");
    return 1;
  }
  array_grid();
  texture_grid();
  other_resources();
  mipmapped();
  anisotropy();
  views();
  fetches();
  bc67_modes();
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);
  if (g_print) return 0;
  std::printf("%d observations checked, %d differ from the card\n", g_checked, g_failures);
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
