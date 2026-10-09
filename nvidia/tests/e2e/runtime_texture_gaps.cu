// Texture formats and objects the runtime answers as an RTX 3060's runtime does: which channel descriptors
// make an array, which read modes, filters and sRGB flags each format takes, block-compressed textures
// (BC1 to BC5 sampled; BC6H and BC7 only made), 10:10:10:2, resource views, and a descriptor's anisotropy.
//
// Every observation (an error code, or the FNV-1a hash of the values a set of fetches returned, as float
// bits) is compared with what the card gave, recorded in runtime_texture_gaps_expected.inc
// (`runtime_texture_gaps --print` prints the file). The same program passes against NVIDIA's runtime on a
// card, which is how the file was made. A texture of BC6H or BC7 blocks is the one place the simulator
// answers differently: it says cudaErrorNotSupported by name (its decoders are not written), and the check
// reports that rather than failing.
#include <cuda_runtime.h>

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
#include "runtime_texture_gaps_expected.inc"
};

static bool g_print = false;
static int g_checked = 0, g_failures = 0, g_unsupported_bc67 = 0;

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
    const bool bc67 = (label.find("BC6H") != std::string::npos || label.find("BC7") != std::string::npos) &&
                      value == (long long)cudaErrorNotSupported;
    if (bc67) {
      ++g_unsupported_bc67;
    } else {
      std::printf("FAIL %s: expected %lld, observed %lld\n", label.c_str(), e->value, value);
      ++g_failures;
    }
  }
}

static unsigned long long fnv(const void* data, size_t bytes, unsigned long long h = 1469598103934665603ull) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

__global__ void fetch_uv(cudaTextureObject_t t, const float2* uv, float4* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2D<float4>(t, uv[i].x, uv[i].y);
}
__global__ void gather_uv(cudaTextureObject_t t, const float2* uv, float4* out, int n, int comp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2Dgather<float4>(t, uv[i].x, uv[i].y, comp);
}

static uint64_t g_lcg = 0x9E3779B97F4A7C15ull;
static uint8_t next_byte() {
  g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<uint8_t>(g_lcg >> 56);
}

// Fetches `uv` from the texture and returns the hash of the float4s (or -1 if something failed).
static long long fetch_hash(cudaTextureObject_t t, const std::vector<float2>& uv, int gather = -1) {
  const int n = static_cast<int>(uv.size());
  float2* duv = nullptr;
  float4* dout = nullptr;
  cudaMalloc(&duv, n * sizeof(float2));
  cudaMalloc(&dout, n * sizeof(float4));
  cudaMemcpy(duv, uv.data(), n * sizeof(float2), cudaMemcpyHostToDevice);
  if (gather < 0) fetch_uv<<<(n + 255) / 256, 256>>>(t, duv, dout, n);
  else gather_uv<<<(n + 255) / 256, 256>>>(t, duv, dout, n, gather);
  std::vector<float4> out(n);
  const cudaError_t e = cudaMemcpy(out.data(), dout, n * sizeof(float4), cudaMemcpyDeviceToHost);
  cudaFree(duv);
  cudaFree(dout);
  if (e != cudaSuccess) return -1;
  return static_cast<long long>(fnv(out.data(), out.size() * sizeof(float4)));
}

static std::vector<float2> grid_centres(int w, int h) {
  std::vector<float2> uv;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) uv.push_back(make_float2(x + 0.5f, y + 0.5f));
  return uv;
}

// ---- channel descriptors, and the read mode / filter / sRGB grid ----
struct Kind {
  const char* name;
  int kind, x, y, z, w;
};
static const Kind kKinds[] = {
    {"S8", 0, 8, 0, 0, 0},          {"U8", 1, 8, 0, 0, 0},          {"S16x2", 0, 16, 16, 0, 0},
    {"U16", 1, 16, 0, 0, 0},        {"S32", 0, 32, 0, 0, 0},        {"U32x2", 1, 32, 32, 0, 0},
    {"F16", 2, 16, 0, 0, 0},        {"F32", 2, 32, 0, 0, 0},        {"UN8", 5, 8, 0, 0, 0},
    {"UN8x2", 6, 8, 8, 0, 0},       {"UN8x4", 7, 8, 8, 8, 8},       {"UN16", 8, 16, 0, 0, 0},
    {"UN16x4", 10, 16, 16, 16, 16}, {"SN8", 11, 8, 0, 0, 0},       {"SN8x4", 13, 8, 8, 8, 8},
    {"SN16", 14, 16, 0, 0, 0},      {"SN16x2", 15, 16, 16, 0, 0},   {"BC1", 17, 8, 8, 8, 8},
    {"BC1s", 18, 8, 8, 8, 8},       {"BC2", 19, 8, 8, 8, 8},        {"BC2s", 20, 8, 8, 8, 8},
    {"BC3", 21, 8, 8, 8, 8},        {"BC3s", 22, 8, 8, 8, 8},       {"BC4u", 23, 8, 0, 0, 0},
    {"BC4s", 24, 8, 0, 0, 0},       {"BC5u", 25, 8, 8, 0, 0},       {"BC5s", 26, 8, 8, 0, 0},
    {"BC6Hu", 27, 16, 16, 16, 0},   {"BC6Hs", 28, 16, 16, 16, 0},   {"BC7", 29, 8, 8, 8, 8},
    {"BC7s", 30, 8, 8, 8, 8},       {"RGB10A2", 31, 10, 10, 10, 2},
};

static void format_grid() {
  for (const Kind& k : kKinds) {
    cudaChannelFormatDesc d = {k.x, k.y, k.z, k.w, static_cast<cudaChannelFormatKind>(k.kind)};
    cudaArray_t a = nullptr;
    const cudaError_t e = cudaMallocArray(&a, &d, 8, 8);
    observe(std::string("array/") + k.name, e);
    cudaGetLastError();
    if (e) continue;
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = a;
    for (int rm = 0; rm < 2; ++rm)
      for (int lin = 0; lin < 2; ++lin)
        for (int srgb = 0; srgb < 2; ++srgb) {
          cudaTextureDesc td = {};
          td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
          td.filterMode = lin ? cudaFilterModeLinear : cudaFilterModePoint;
          td.readMode = rm ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
          td.sRGB = srgb;
          cudaTextureObject_t t = 0;
          const cudaError_t te = cudaCreateTextureObject(&t, &rd, &td, nullptr);
          observe(std::string("texture/") + k.name + (rm ? "/norm" : "/elem") + (lin ? "/lin" : "/pt") +
                      (srgb ? "/srgb" : "/plain"),
                  te);
          if (!te) cudaDestroyTextureObject(t);
          cudaGetLastError();
        }
    cudaFreeArray(a);
  }
  // Descriptors the card refuses: widths that are not the format's, three channels, gaps, no format.
  struct Bad {
    const char* name;
    int kind, x, y, z, w;
  };
  static const Bad bad[] = {
      {"BC1-7bit", 17, 7, 8, 8, 8},     {"BC1-3ch", 17, 8, 8, 8, 0},     {"BC4-2ch", 23, 8, 8, 0, 0},
      {"BC5-1ch", 25, 8, 0, 0, 0},      {"BC6H-8bit", 27, 8, 8, 8, 0},   {"RGB10A2-8888", 31, 8, 8, 8, 8},
      {"RGB10A2-2101010", 31, 2, 10, 10, 10}, {"UN8-3ch", 7, 8, 8, 8, 0}, {"gap", 1, 8, 0, 8, 0},
      {"kind-none", 4, 8, 0, 0, 0},     {"kind-NV12", 32, 8, 8, 8, 8},   {"kind-99", 99, 8, 0, 0, 0},
      {"U8-12bit", 1, 12, 0, 0, 0},
  };
  for (const Bad& b : bad) {
    cudaChannelFormatDesc d = {b.x, b.y, b.z, b.w, static_cast<cudaChannelFormatKind>(b.kind)};
    cudaArray_t a = nullptr;
    observe(std::string("bad-array/") + b.name, cudaMallocArray(&a, &d, 8, 8));
    if (a) cudaFreeArray(a);
    cudaGetLastError();
  }
}

// ---- a descriptor's other fields ----
static void descriptor_fields() {
  cudaChannelFormatDesc f = cudaCreateChannelDesc<float>();
  cudaArray_t a = nullptr;
  cudaMallocArray(&a, &f, 8, 8);
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypeArray;
  rd.res.array.array = a;
  cudaTextureDesc base = {};
  base.addressMode[0] = base.addressMode[1] = base.addressMode[2] = cudaAddressModeClamp;
  auto try_td = [&](const std::string& name, const cudaTextureDesc& td) {
    cudaTextureObject_t t = 0;
    const cudaError_t e = cudaCreateTextureObject(&t, &rd, &td, nullptr);
    observe("desc/" + name, e);
    if (!e) cudaDestroyTextureObject(t);
    cudaGetLastError();
  };
  try_td("default", base);
  // maxAnisotropy: any value is accepted (and changes nothing for a fetch with no derivatives).
  for (unsigned an : {0u, 1u, 2u, 8u, 16u, 17u, 100u, 0xffffffffu}) {
    cudaTextureDesc td = base;
    td.maxAnisotropy = an;
    try_td("maxAnisotropy-" + std::to_string(an), td);
  }
  {
    cudaTextureDesc td = base;
    td.addressMode[0] = static_cast<cudaTextureAddressMode>(7);
    try_td("addressMode-bad", td);
  }
  {
    cudaTextureDesc td = base;
    td.filterMode = static_cast<cudaTextureFilterMode>(5);
    try_td("filterMode-bad", td);
  }
  {
    cudaTextureDesc td = base;
    td.mipmapFilterMode = static_cast<cudaTextureFilterMode>(5);
    try_td("mipmapFilterMode-bad", td);
  }
  {
    cudaTextureDesc td = base;
    td.seamlessCubemap = 1;
    try_td("seamlessCubemap-on-2D", td);
  }
  {
    cudaTextureDesc td = base;
    td.disableTrilinearOptimization = 1;
    try_td("disableTrilinearOptimization", td);
  }
  // The fetch with an anisotropy set is the fetch without.
  {
    cudaTextureDesc td = base;
    td.filterMode = cudaFilterModeLinear;
    td.normalizedCoords = 0;
    cudaTextureObject_t t1 = 0, t2 = 0;
    float host[64];
    for (int i = 0; i < 64; ++i) host[i] = static_cast<float>((i * 37) % 11) * 0.25f;
    cudaMemcpy2DToArray(a, 0, 0, host, 8 * sizeof(float), 8 * sizeof(float), 8, cudaMemcpyHostToDevice);
    td.maxAnisotropy = 1;
    cudaCreateTextureObject(&t1, &rd, &td, nullptr);
    td.maxAnisotropy = 16;
    cudaCreateTextureObject(&t2, &rd, &td, nullptr);
    std::vector<float2> uv;
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) uv.push_back(make_float2(x * 0.53f - 0.7f, y * 0.47f + 0.2f));
    observe("aniso/same-fetch", fetch_hash(t1, uv) == fetch_hash(t2, uv) ? 1 : 0);
    cudaDestroyTextureObject(t1);
    cudaDestroyTextureObject(t2);
  }
  cudaFreeArray(a);
}

// ---- block-compressed arrays: sizes, copies, and what a fetch returns ----
struct Fmt {
  const char* name;
  int kind, x, y, z, w;
  int block_bytes;
  bool srgb;
};
static const Fmt kBc[] = {
    {"BC1", 17, 8, 8, 8, 8, 8, false},   {"BC1s", 18, 8, 8, 8, 8, 8, true},  {"BC2", 19, 8, 8, 8, 8, 16, false},
    {"BC3", 21, 8, 8, 8, 8, 16, false},  {"BC3s", 22, 8, 8, 8, 8, 16, true}, {"BC4u", 23, 8, 0, 0, 0, 8, false},
    {"BC4s", 24, 8, 0, 0, 0, 8, false},  {"BC5u", 25, 8, 8, 0, 0, 16, false}, {"BC5s", 26, 8, 8, 0, 0, 16, false},
};

// An array of blocks_w x blocks_h blocks holding `blocks`, and a texture of it; false if it could not be made.
struct BcTex {
  cudaArray_t arr = nullptr;
  cudaTextureObject_t tex = 0;
  ~BcTex() {
    if (tex) cudaDestroyTextureObject(tex);
    if (arr) cudaFreeArray(arr);
  }
};

static bool make_bc(const Fmt& f, int bw, int bh, const std::vector<uint8_t>& blocks, cudaTextureFilterMode filter,
                    cudaTextureAddressMode address, bool normalized, BcTex* out) {
  cudaChannelFormatDesc d = {f.x, f.y, f.z, f.w, static_cast<cudaChannelFormatKind>(f.kind)};
  if (cudaMallocArray(&out->arr, &d, bw * 4, bh * 4) != cudaSuccess) return false;
  if (cudaMemcpy2DToArray(out->arr, 0, 0, blocks.data(), (size_t)bw * f.block_bytes, (size_t)bw * f.block_bytes, bh,
                          cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypeArray;
  rd.res.array.array = out->arr;
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = address;
  td.filterMode = filter;
  td.readMode = cudaReadModeNormalizedFloat;
  td.sRGB = f.srgb;
  td.normalizedCoords = normalized;
  return cudaCreateTextureObject(&out->tex, &rd, &td, nullptr) == cudaSuccess;
}

static std::vector<uint8_t> random_blocks(size_t n) {
  std::vector<uint8_t> b(n);
  for (auto& x : b) x = next_byte();
  return b;
}

static void bc_fetches() {
  const int bw = 16, bh = 16, w = bw * 4, h = bh * 4;
  for (const Fmt& f : kBc) {
    g_lcg = 0x9E3779B97F4A7C15ull ^ (uint64_t)f.kind;
    const std::vector<uint8_t> blocks = random_blocks((size_t)bw * bh * f.block_bytes);
    // Point sampled at every texel centre.
    {
      BcTex t;
      if (!make_bc(f, bw, bh, blocks, cudaFilterModePoint, cudaAddressModeClamp, false, &t)) {
        observe(std::string("bc/") + f.name + "/point", -1);
        continue;
      }
      observe(std::string("bc/") + f.name + "/point", fetch_hash(t.tex, grid_centres(w, h)));
      // The array reports its size in texels.
      cudaExtent ext = {};
      cudaChannelFormatDesc cd = {};
      unsigned int fl = 0;
      cudaArrayGetInfo(&cd, &ext, &fl, t.arr);
      observe(std::string("bc/") + f.name + "/extent", (long long)ext.width * 1000 + (long long)ext.height);
      // And a copy out gives the blocks back.
      std::vector<uint8_t> back(blocks.size());
      cudaMemcpy2DFromArray(back.data(), (size_t)bw * f.block_bytes, t.arr, 0, 0, (size_t)bw * f.block_bytes, bh,
                            cudaMemcpyDeviceToHost);
      observe(std::string("bc/") + f.name + "/roundtrip", back == blocks ? 1 : 0);
      // tld4: the four texels around a point, one channel.
      std::vector<float2> uv;
      for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) uv.push_back(make_float2(x * 4.0f + 1.0f, y * 4.0f + 3.0f));
      BcTex tp;
      if (make_bc(f, bw, bh, blocks, cudaFilterModePoint, cudaAddressModeClamp, false, &tp))
        observe(std::string("bc/") + f.name + "/gather0", fetch_hash(tp.tex, uv, 0));
    }
    // Linear, under each address mode, and point with normalized coordinates out of range.
    const cudaTextureAddressMode modes[3] = {cudaAddressModeClamp, cudaAddressModeWrap, cudaAddressModeMirror};
    for (int pass = 0; pass < 4; ++pass) {
      BcTex t;
      const bool lin = pass < 3;
      if (!make_bc(f, bw, bh, blocks, lin ? cudaFilterModeLinear : cudaFilterModePoint,
                   modes[pass == 3 ? 1 : pass], pass == 3, &t) &&
          lin) {
        observe(std::string("bc/") + f.name + "/pass" + std::to_string(pass), -1);
        continue;
      }
      std::vector<float2> uv;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const float fx = ((x * 7 + y * 13 + pass) % 32) / 32.0f, fy = ((x * 11 + y * 5 + 3 * pass) % 32) / 32.0f;
          float u = x + fx - 2.5f * (pass == 1 || pass == 2), v = y + fy + 1.5f * (pass == 1 || pass == 2);
          if (pass == 3) { u = (x + 0.25f + 0.5f * (x % 3)) / w - 0.37f; v = (y + 0.5f) / h + 0.8f; }   // normalized
          uv.push_back(make_float2(u, v));
        }
      observe(std::string("bc/") + f.name + "/pass" + std::to_string(pass), fetch_hash(t.tex, uv));
    }
  }
}

// Every endpoint pair (a sample of them: a0 = 4i + 1, a1 = 4j + 3), all eight indices; and BC1's three-colour
// mode, endpoint pairs of each channel; the unsigned and signed one-channel formats and BC3's alpha.
static void bc_exhaustive() {
  // BC4 / BC5, unsigned and signed.
  for (const Fmt& f : kBc) {
    if (f.kind < 21 || f.kind == 22) continue;   // BC3 (its alpha) and BC4, BC5
    const bool one = f.kind == 23 || f.kind == 24;
    const bool alpha = f.kind == 21;   // BC3: the alpha half
    const int bw = 64, bh = 64;
    std::vector<uint8_t> blocks((size_t)bw * bh * f.block_bytes, 0);
    for (int i = 0; i < bw * bh; ++i) {
      uint8_t* p = &blocks[(size_t)i * f.block_bytes];
      const int a0 = (i / bw) * 4 + 1, a1 = (i % bw) * 4 + 3;
      uint64_t bits = 0;
      for (int j = 0; j < 16; ++j) bits |= (uint64_t)((j + i) & 7) << (3 * j);
      p[0] = (uint8_t)a0;
      p[1] = (uint8_t)a1;
      std::memcpy(p + 2, &bits, 6);
      if (!one && !alpha) {   // BC5: the second channel is the first with the endpoints swapped
        p[8] = (uint8_t)a1;
        p[9] = (uint8_t)a0;
        std::memcpy(p + 10, &bits, 6);
      }
    }
    BcTex t;
    if (!make_bc(f, bw, bh, blocks, cudaFilterModePoint, cudaAddressModeClamp, false, &t)) {
      observe(std::string("bc-pairs/") + f.name, -1);
      continue;
    }
    observe(std::string("bc-pairs/") + f.name, fetch_hash(t.tex, grid_centres(bw * 4, bh * 4)));
  }
  // BC1 three-colour mode: for each channel, endpoint pairs with c0 <= c1, every texel index 2 (and a quarter
  // of them 0, 1, 3).
  {
    const Fmt& f = kBc[0];
    const int bw = 64, bh = 32;
    std::vector<uint8_t> blocks((size_t)bw * bh * 8, 0);
    int n = 0;
    auto add = [&](int r0, int g0, int b0, int r1, int g1, int b1) {
      if (n >= bw * bh) return;
      const uint16_t c0 = (uint16_t)((r0 << 11) | (g0 << 5) | b0), c1 = (uint16_t)((r1 << 11) | (g1 << 5) | b1);
      uint8_t* p = &blocks[(size_t)n++ * 8];
      p[0] = c0 & 255; p[1] = c0 >> 8; p[2] = c1 & 255; p[3] = c1 >> 8;
      uint32_t idx = 0;
      for (int j = 0; j < 16; ++j) idx |= (uint32_t)((j % 4 == 0) ? (j / 4) % 4 : 2) << (2 * j);
      std::memcpy(p + 4, &idx, 4);
    };
    for (int a = 0; a < 32; a += 1)
      for (int b = a; b < 32; b += 3) add(a, 0, 0, b, 0, 0);
    for (int a = 0; a < 64; a += 3)
      for (int b = a; b < 64; b += 5) add(0, a, 0, 0, b, 0);
    for (int a = 0; a < 32; a += 1)
      for (int b = a; b < 32; b += 3) add(0, 0, a, 0, 0, b);
    // Blocks where the green of c0 is the larger one but red decides: three-colour with g0 > g1.
    for (int a = 0; a < 64; a += 5) add(1, a, 3, 20, 63 - a, 4);
    BcTex t;
    if (make_bc(f, bw, bh, blocks, cudaFilterModePoint, cudaAddressModeClamp, false, &t))
      observe("bc-pairs/BC1-three-colour", fetch_hash(t.tex, grid_centres(bw * 4, bh * 4)));
    else
      observe("bc-pairs/BC1-three-colour", -1);
  }
  // BC1 four-colour mode: endpoint pairs of each channel.
  {
    const Fmt& f = kBc[0];
    const int bw = 64, bh = 64;
    std::vector<uint8_t> blocks((size_t)bw * bh * 8, 0);
    for (int i = 0; i < bw * bh; ++i) {
      const int r0 = i % 32, r1 = (i / 64) % 32, g0 = i / 64, g1 = i % 64, b0 = (i / 32) % 32, b1 = (i / 1024 + i) % 32;
      const uint16_t c0 = (uint16_t)((r0 << 11) | (g0 << 5) | b0), c1 = (uint16_t)((r1 << 11) | (g1 << 5) | b1);
      uint8_t* p = &blocks[(size_t)i * 8];
      // c0 must exceed c1 for the four-colour mode: order them.
      const uint16_t hi = c0 > c1 ? c0 : c1, lo = c0 > c1 ? c1 : c0;
      p[0] = hi & 255; p[1] = hi >> 8; p[2] = lo & 255; p[3] = lo >> 8;
      uint32_t idx = 0;
      for (int j = 0; j < 16; ++j) idx |= (uint32_t)(j & 3) << (2 * j);
      std::memcpy(p + 4, &idx, 4);
    }
    BcTex t;
    if (make_bc(f, bw, bh, blocks, cudaFilterModePoint, cudaAddressModeClamp, false, &t))
      observe("bc-pairs/BC1-four-colour", fetch_hash(t.tex, grid_centres(bw * 4, bh * 4)));
    else
      observe("bc-pairs/BC1-four-colour", -1);
  }
}

// ---- 10:10:10:2 ----
static void packed_1010102() {
  const int W = 1024;
  std::vector<uint32_t> w(W);
  for (int i = 0; i < W; ++i)
    w[i] = (uint32_t)i | ((uint32_t)(1023 - i) << 10) | ((uint32_t)((i * 5) & 1023) << 20) | ((uint32_t)(i & 3) << 30);
  cudaChannelFormatDesc d = {10, 10, 10, 2, static_cast<cudaChannelFormatKind>(31)};   // UnsignedNormalized1010102 (CUDA 12.1+)
  cudaArray_t a = nullptr;
  if (cudaMallocArray(&a, &d, W, 1) != cudaSuccess) {
    observe("rgb10a2/array", -1);
    return;
  }
  cudaMemcpy2DToArray(a, 0, 0, w.data(), W * 4, W * 4, 1, cudaMemcpyHostToDevice);
  for (int lin = 0; lin < 2; ++lin) {
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = a;
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = lin ? cudaFilterModeLinear : cudaFilterModePoint;
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t t = 0;
    if (cudaCreateTextureObject(&t, &rd, &td, nullptr) != cudaSuccess) {
      observe(std::string("rgb10a2/") + (lin ? "linear" : "point"), -1);
      continue;
    }
    // At texel centres (every code), and between them.
    std::vector<float2> uv;
    for (int i = 0; i < W; ++i) uv.push_back(make_float2(i + 0.5f, 0.5f));
    observe(std::string("rgb10a2/") + (lin ? "linear-centres" : "point-centres"), fetch_hash(t, uv));
    uv.clear();
    for (int i = 0; i < W; ++i) uv.push_back(make_float2(i + 0.5f + (i % 7) / 8.0f, 0.5f));
    observe(std::string("rgb10a2/") + (lin ? "linear-between" : "point-between"), fetch_hash(t, uv));
    cudaDestroyTextureObject(t);
  }
  cudaFreeArray(a);
}

// ---- resource views ----
static void views() {
  // A view reinterprets the same bytes as another format of the same texel size. Blocks live in a uint2
  // array (8-byte texels) and are viewed as BC1 and the like. The card checks the read mode against the
  // array's own format (a 32-bit integer one takes the element-type read only), whatever the view says.
  const int bw = 16, bh = 16;
  g_lcg = 0x9E3779B97F4A7C15ull ^ 17;
  const std::vector<uint8_t> blocks = random_blocks((size_t)bw * bh * 8);
  cudaChannelFormatDesc u2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
  cudaArray_t a = nullptr;
  cudaMallocArray(&a, &u2, bw, bh);
  cudaMemcpy2DToArray(a, 0, 0, blocks.data(), (size_t)bw * 8, (size_t)bw * 8, bh, cudaMemcpyHostToDevice);
  cudaResourceDesc rd = {};
  rd.resType = cudaResourceTypeArray;
  rd.res.array.array = a;
  for (int norm = 0; norm < 2; ++norm) {
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;
    td.readMode = norm ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    const std::string mode = norm ? "/norm" : "/elem";
    auto make = [&](cudaResourceViewFormat vf, size_t w, size_t h, cudaTextureObject_t* t) {
      cudaResourceViewDesc v = {};
      v.format = vf;
      v.width = w;
      v.height = h;
      const cudaError_t e = cudaCreateTextureObject(t, &rd, &td, &v);
      cudaGetLastError();
      return e;
    };
    cudaTextureObject_t t = 0;
    auto done = [&] {
      if (t) cudaDestroyTextureObject(t);
      t = 0;
    };
    observe("view/uint2-as-BC1" + mode, make(cudaResViewFormatUnsignedBlockCompressed1, bw * 4, bh * 4, &t));
    if (t) {
      BcTex ref;
      make_bc(kBc[0], bw, bh, blocks, cudaFilterModePoint, cudaAddressModeClamp, false, &ref);
      observe("view/uint2-as-BC1-same-as-array" + mode,
              fetch_hash(t, grid_centres(bw * 4, bh * 4)) == fetch_hash(ref.tex, grid_centres(bw * 4, bh * 4)) ? 1 : 0);
    }
    done();
    observe("view/uint2-as-BC4" + mode, make(cudaResViewFormatUnsignedBlockCompressed4, bw * 4, bh * 4, &t));
    done();
    observe("view/uint2-as-BC1-wrong-extent" + mode, make(cudaResViewFormatUnsignedBlockCompressed1, bw, bh, &t));
    done();
    observe("view/uint2-as-BC2-size-mismatch" + mode, make(cudaResViewFormatUnsignedBlockCompressed2, bw * 4, bh * 4, &t));
    done();
    observe("view/uint2-as-BC7-size-mismatch" + mode, make(cudaResViewFormatUnsignedBlockCompressed7, bw * 4, bh * 4, &t));
    done();
    observe("view/uint2-as-ushort4" + mode, make(cudaResViewFormatUnsignedShort4, bw, bh, &t));
    done();
    observe("view/uint2-as-float2" + mode, make(cudaResViewFormatFloat2, bw, bh, &t));
    done();
    observe("view/uint2-as-uchar4-size-mismatch" + mode, make(cudaResViewFormatUnsignedChar4, bw, bh, &t));
    done();
    observe("view/none" + mode, make(cudaResViewFormatNone, bw, bh, &t));
    done();
    observe("view/format-99" + mode, make(static_cast<cudaResourceViewFormat>(99), bw, bh, &t));
    done();
  }
  cudaFreeArray(a);
}

int main(int argc, char** argv) {
  g_print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  cudaFree(0);
  format_grid();
  descriptor_fields();
  bc_fetches();
  bc_exhaustive();
  packed_1010102();
  views();
  if (g_print) return 0;
  if (g_unsupported_bc67)
    std::printf("note: %d observation(s) were cudaErrorNotSupported for BC6H / BC7 (not implemented here)\n",
                g_unsupported_bc67);
  std::printf("%d observations checked, %d differ from the card\n", g_checked, g_failures);
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
