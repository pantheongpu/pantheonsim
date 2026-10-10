// sRGB textures (cudaTextureDesc::sRGB): 8-bit unsigned normalized texels
// decoded to linear through the texture unit's table -- x, y and z of a
// four-channel texture, x of a one- or two-channel one -- and other formats
// left alone, point sampled, linearly filtered, gathered, and through 1D, 3D,
// layered, cubemap and mipmapped textures, with border colours. Each case's
// results are hashed and compared with an RTX 3060's (sm_86), recorded in
// texture_srgb_expected.inc (`texture_srgb --print` prints them; `--dump N`
// prints case N's values). Prints PASS on the last line, and
// runs the same on a GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_fp16.h>

struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "texture_srgb_expected.inc"
};

constexpr int kSide = 12;              // coordinates per axis
constexpr int kN = kSide * kSide;      // fetches per case
__constant__ float kCoord[kSide] = {-1.25f, -0.6f, -0.1f, 0.2f, 0.5f, 0.75f, 1.9f, 3.3f, 3.6f, 3.95f, 4.4f, 5.5f};

enum Geom { G2D, G1D, G3D, GLayered, GCube, GMip, GGather0, GGather3, GNorm };

template <class T>
__global__ void fetch(cudaTextureObject_t t, int geom, T* out) {
  const int i = threadIdx.x;
  const float x = kCoord[i % kSide], y = kCoord[i / kSide];
  T r{};
  switch (geom) {
    case G2D: r = tex2D<T>(t, x, y); break;
    case G1D: r = tex1D<T>(t, x); break;
    case G3D: r = tex3D<T>(t, x, y, x * 0.7f - y * 0.3f); break;
    case GLayered: r = tex2DLayered<T>(t, x, y, 1); break;
    case GCube: r = texCubemap<T>(t, 1.0f, (x - 2.0f) * 0.4f, (y - 2.0f) * 0.4f); break;
    case GMip: r = tex2DLod<T>(t, x * 0.5f, y * 0.5f, 0.5f); break;
    case GGather0: r = tex2Dgather<T>(t, x, y, 0); break;
    case GGather3: r = tex2Dgather<T>(t, x, y, 3); break;
    case GNorm: r = tex2D<T>(t, x * 0.25f, y * 0.25f); break;
  }
  out[i] = r;
}

struct Format {
  const char* name;
  int bits[4];
  cudaChannelFormatKind kind;
  bool normalized;   // read as normalized float
  int out;           // 0 float4, 1 uint4, 2 int4
};

static const Format kFormats[] = {
    {"u8x4n", {8, 8, 8, 8}, cudaChannelFormatKindUnsigned, true, 0},
    {"u8x2n", {8, 8, 0, 0}, cudaChannelFormatKindUnsigned, true, 0},
    {"u8x1n", {8, 0, 0, 0}, cudaChannelFormatKindUnsigned, true, 0},
    {"u16x2n", {16, 16, 0, 0}, cudaChannelFormatKindUnsigned, true, 0},
    {"s16x2n", {16, 16, 0, 0}, cudaChannelFormatKindSigned, true, 0},
    {"f32x4", {32, 32, 32, 32}, cudaChannelFormatKindFloat, false, 0},
    {"f16x4", {16, 16, 16, 16}, cudaChannelFormatKindFloat, false, 0},
    {"u8x4", {8, 8, 8, 8}, cudaChannelFormatKindUnsigned, false, 1},
};

static const unsigned kBorders[][4] = {
    {0, 0, 0, 0},
    {0x3e800000u, 0x3f000000u, 0x3dcccccdu, 0x3f333333u},   // 0.25, 0.5, 0.1, 0.7
    {0x3f19999au, 0x3eaaa64cu, 0x3f7d70a4u, 0x3c23d70au},   // 0.6, 0.3333, 0.99, 0.01
};

struct Case {
  std::string tag;
  int fmt, border, geom;
  bool linear;
  cudaTextureAddressMode ay;   // the y (and z) mode; x is always border
};

static int texel_bytes(const Format& f) { return (f.bits[0] + f.bits[1] + f.bits[2] + f.bits[3]) / 8; }

// Deterministic texel data, finite for the float formats.
static void fill(const Format& f, std::vector<unsigned char>& buf, size_t texels, unsigned salt) {
  const int tb = texel_bytes(f);
  buf.assign(texels * tb, 0);
  for (size_t t = 0; t < texels; ++t) {
    size_t off = t * tb;
    for (int c = 0; c < 4 && f.bits[c]; ++c) {
      const unsigned seed = static_cast<unsigned>(t * 4 + c) * 2654435761u + salt;
      if (f.kind == cudaChannelFormatKindFloat) {
        const float v = static_cast<float>(static_cast<int>(seed >> 20) % 2001 - 1000) / 256.0f;
        if (f.bits[c] == 32) std::memcpy(&buf[off], &v, 4);
        else { const __half h = __float2half(v); std::memcpy(&buf[off], &h, 2); }
      } else {
        for (int b = 0; b < f.bits[c] / 8; ++b) buf[off + b] = static_cast<unsigned char>(seed >> (8 * b + 7));
      }
      off += f.bits[c] / 8;
    }
  }
}

static cudaExtent extent_of(int geom) {
  switch (geom) {
    case G1D: return make_cudaExtent(4, 0, 0);
    case G3D: return make_cudaExtent(4, 4, 4);
    case GLayered: return make_cudaExtent(4, 4, 2);
    case GCube: return make_cudaExtent(4, 4, 6);
    default: return make_cudaExtent(4, 4, 0);
  }
}

// Runs one case; returns false if the runtime refused something.
static bool run(const Case& c, void* dout, std::vector<unsigned>& res, std::string* why) {
  const Format& f = kFormats[c.fmt];
  const cudaChannelFormatDesc cd = cudaCreateChannelDesc(f.bits[0], f.bits[1], f.bits[2], f.bits[3], f.kind);
  cudaResourceDesc rd{};
  cudaArray_t arr = nullptr;
  cudaMipmappedArray_t mip = nullptr;
  std::vector<unsigned char> buf;
  const cudaExtent e = extent_of(c.geom);
  const int tb = texel_bytes(f);
  if (c.geom == GMip) {
    if (cudaMallocMipmappedArray(&mip, &cd, e, 2) != cudaSuccess) { *why = "cudaMallocMipmappedArray"; return false; }
    for (unsigned l = 0; l < 2; ++l) {
      cudaArray_t lv;
      cudaGetMipmappedArrayLevel(&lv, mip, l);
      const size_t w = 4 >> l;
      fill(f, buf, w * w, 7 + l);
      cudaMemcpy2DToArray(lv, 0, 0, buf.data(), w * tb, w * tb, w, cudaMemcpyHostToDevice);
    }
    rd.resType = cudaResourceTypeMipmappedArray;
    rd.res.mipmap.mipmap = mip;
  } else {
    unsigned flags = c.geom == GLayered ? cudaArrayLayered : c.geom == GCube ? cudaArrayCubemap : 0;
    if (cudaMalloc3DArray(&arr, &cd, e, flags) != cudaSuccess) { *why = "cudaMalloc3DArray"; return false; }
    const size_t texels = e.width * (e.height ? e.height : 1) * (e.depth ? e.depth : 1);
    fill(f, buf, texels, 7);
    cudaMemcpy3DParms p{};
    p.srcPtr = make_cudaPitchedPtr(buf.data(), e.width * tb, e.width, e.height ? e.height : 1);
    p.dstArray = arr;
    p.extent = make_cudaExtent(e.width, e.height ? e.height : 1, e.depth ? e.depth : 1);
    p.kind = cudaMemcpyHostToDevice;
    if (cudaMemcpy3D(&p) != cudaSuccess) { *why = "cudaMemcpy3D"; return false; }
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
  }
  cudaTextureDesc td{};
  td.addressMode[0] = cudaAddressModeBorder;
  td.addressMode[1] = td.addressMode[2] = c.ay;
  td.filterMode = c.linear ? cudaFilterModeLinear : cudaFilterModePoint;
  td.readMode = f.normalized ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
  td.normalizedCoords = c.geom == GCube || c.geom == GNorm;
  std::memcpy(td.borderColor, kBorders[c.border], 16);
  td.sRGB = 1;
  if (c.geom == GMip) {
    td.mipmapFilterMode = cudaFilterModeLinear;
    td.maxMipmapLevelClamp = 1;
  }
  cudaTextureObject_t t;
  cudaError_t err = cudaCreateTextureObject(&t, &rd, &td, nullptr);
  if (err != cudaSuccess) { *why = std::string("cudaCreateTextureObject: ") + cudaGetErrorString(err); return false; }
  if (f.out == 0) fetch<float4><<<1, kN>>>(t, c.geom, static_cast<float4*>(dout));
  else if (f.out == 1) fetch<uint4><<<1, kN>>>(t, c.geom, static_cast<uint4*>(dout));
  else fetch<int4><<<1, kN>>>(t, c.geom, static_cast<int4*>(dout));
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { *why = std::string("the fetch: ") + cudaGetErrorString(err); return false; }
  res.resize(4 * kN);
  cudaMemcpy(res.data(), dout, 16 * kN, cudaMemcpyDeviceToHost);
  cudaDestroyTextureObject(t);
  if (arr) cudaFreeArray(arr);
  if (mip) cudaFreeMipmappedArray(mip);
  return true;
}

int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  const int dump = argc > 2 && std::strcmp(argv[1], "--dump") == 0 ? atoi(argv[2]) : -1;
  std::vector<Case> cases;
  const int nf = sizeof kFormats / sizeof kFormats[0], nb = sizeof kBorders / sizeof kBorders[0];
  for (int fi = 0; fi < nf; ++fi) {
    const Format& f = kFormats[fi];
    // Linear filtering needs a float or normalized texture. Signed 8-bit
    // normalized is left out of this grid (texture_forms.cu checks its filter
    // against the card).
    const bool filterable = (f.kind == cudaChannelFormatKindFloat || f.normalized) &&
                            !(f.kind == cudaChannelFormatKindSigned && f.bits[0] == 8);
    for (int b = 0; b < nb; ++b) {
      auto add = [&](const char* g, int geom, bool lin, cudaTextureAddressMode ay) {
        cases.push_back({std::string(f.name) + " b" + std::to_string(b) + " " + g, fi, b, geom, lin, ay});
      };
      add("2d point", G2D, false, cudaAddressModeBorder);
      add("2d point clamp-y", G2D, false, cudaAddressModeClamp);
      add("1d point", G1D, false, cudaAddressModeBorder);
      add("gather0", GGather0, false, cudaAddressModeBorder);
      add("gather3", GGather3, false, cudaAddressModeBorder);
      if (!filterable) continue;
      add("2d linear", G2D, true, cudaAddressModeBorder);
      add("2d linear wrap-y", G2D, true, cudaAddressModeWrap);
      add("1d linear", G1D, true, cudaAddressModeBorder);
      add("norm linear", GNorm, true, cudaAddressModeBorder);
      if (fi == 0 || fi == 1) {
        add("3d linear", G3D, true, cudaAddressModeBorder);
        add("layered linear", GLayered, true, cudaAddressModeBorder);
        add("cube linear", GCube, true, cudaAddressModeBorder);
        add("mip linear", GMip, true, cudaAddressModeBorder);
      }
    }
  }
  void* dout;
  cudaMalloc(&dout, 16 * kN);
  int bad = 0;
  const int nexp = sizeof kExpected / sizeof kExpected[0];
  for (int i = 0; i < static_cast<int>(cases.size()); ++i) {
    std::vector<unsigned> r;
    std::string why;
    if (!run(cases[i], dout, r, &why)) {
      printf("%s: %s\n", cases[i].tag.c_str(), why.c_str());
      return 1;
    }
    if (i == dump) {
      for (int k = 0; k < kN; ++k)
        printf("%3d %08x %08x %08x %08x\n", k, r[4 * k], r[4 * k + 1], r[4 * k + 2], r[4 * k + 3]);
      return 0;
    }
    unsigned long long h = 1469598103934665603ull;
    for (unsigned x : r) { h ^= x; h *= 1099511628211ull; }
    if (print) { printf("    {\"%s\", 0x%016llxull},\n", cases[i].tag.c_str(), h); continue; }
    if (dump >= 0) continue;
    if (i >= nexp || std::strcmp(kExpected[i].tag, cases[i].tag.c_str()) || h != kExpected[i].hash) {
      if (bad++ < 30)
        printf("%d %s: 0x%016llx, the card gave 0x%016llx\n", i, cases[i].tag.c_str(), h, i < nexp ? kExpected[i].hash : 0ull);
    }
  }
  if (print) return 0;
  printf("%zu sRGB cases, %d differ\n%s\n", cases.size(), bad, bad || nexp != static_cast<int>(cases.size()) ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
