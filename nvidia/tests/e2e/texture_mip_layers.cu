// Mipmapped layered and cubemap textures: 1D and 2D layered, cubemap and
// layered cubemap mipmaps, fetched at explicit levels of detail with point and
// linear filtering, between levels by point and linear mip filtering, under
// clamp, wrap and border addressing. Each case's results are hashed and
// compared with an RTX 3060's (sm_86), recorded in
// texture_mip_layers_expected.inc (`texture_mip_layers --print` prints them;
// `--dump N` prints case N's values). Prints PASS on the last line, and runs
// the same on a GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "texture_mip_layers_expected.inc"
};

constexpr int kSide = 12;
constexpr int kLods = 8;
constexpr int kN = kSide * kSide * kLods;   // fetches per case
__constant__ float kCoord[kSide] = {-0.3f, -0.02f, 0.05f, 0.13f, 0.29f, 0.41f, 0.5f, 0.66f, 0.8f, 0.97f, 1.04f, 1.6f};
__constant__ float kLod[kLods] = {-1.0f, 0.0f, 0.3f, 0.5f, 1.0f, 1.62f, 2.5f, 7.0f};

enum Geom { GLayered2D, GLayered1D, GCube, GCubeLayered };

template <class T>
__global__ void fetch(cudaTextureObject_t t, int geom, int layer, T* out) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const float x = kCoord[i % kSide], y = kCoord[(i / kSide) % kSide], lod = kLod[i / (kSide * kSide)];
  T r{};
  // Cube directions: a face from the coordinate pair, the major axis cycling.
  // u and v are scaled differently so that no two axes tie: a tie puts the
  // face coordinate exactly on the face's edge, where the card's projection
  // (which multiplies by its approximate reciprocal, 1 ulp low for 2.64) and
  // an exact division disagree about which side of the edge it lands on.
  const float u = (x - 0.5f) * 2.4f, v = (y - 0.5f) * 2.3f;
  const int f = i % 3;
  const float dx = f == 0 ? 1.0f : u, dy = f == 1 ? -1.0f : v, dz = f == 2 ? 1.0f : (f == 0 ? v : u);
  switch (geom) {
    case GLayered2D: r = tex2DLayeredLod<T>(t, x, y, layer, lod); break;
    case GLayered1D: r = tex1DLayeredLod<T>(t, x, layer, lod); break;
    case GCube: r = texCubemapLod<T>(t, dx, dy, dz, lod); break;
    case GCubeLayered: r = texCubemapLayeredLod<T>(t, dx, dy, dz, layer, lod); break;
  }
  out[i] = r;
}

struct Case {
  std::string tag;
  int geom;
  bool unorm8;                     // u8x4 read as normalized float, else f32x1
  bool linear, mip_linear;
  cudaTextureAddressMode mode;
  int layer;
};

int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  const int dump = argc > 2 && std::strcmp(argv[1], "--dump") == 0 ? atoi(argv[2]) : -1;
  std::vector<Case> cases;
  const char* gname[] = {"2d-layered", "1d-layered", "cube", "cube-layered"};
  const char* mname[] = {"wrap", "clamp", "mirror", "border"};
  for (int g = 0; g < 4; ++g)
    for (int u8 = 0; u8 < 2; ++u8)
      for (int lin = 0; lin < 2; ++lin)
        for (int ml = 0; ml < 2; ++ml)
          for (cudaTextureAddressMode mode : {cudaAddressModeClamp, cudaAddressModeWrap, cudaAddressModeBorder})
            for (int layer : {0, 1, 7}) {
              if ((g == GCube) && layer) continue;
              cases.push_back({std::string(gname[g]) + (u8 ? " u8x4n" : " f32") + (lin ? " linear" : " point") +
                                   (ml ? " mip-linear " : " mip-point ") + mname[mode] + " layer" + std::to_string(layer),
                               g, u8 != 0, lin != 0, ml != 0, mode, layer});
            }
  float4* dout;
  cudaMalloc(&dout, sizeof(float4) * kN);
  int bad = 0;
  const int nexp = sizeof kExpected / sizeof kExpected[0];
  for (int ci = 0; ci < static_cast<int>(cases.size()); ++ci) {
    const Case& c = cases[ci];
    const cudaChannelFormatDesc cd = c.unorm8 ? cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned)
                                              : cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    const int tb = c.unorm8 ? 4 : 4;
    const bool cube = c.geom == GCube || c.geom == GCubeLayered;
    const int layers = c.geom == GCube ? 1 : 3;
    const size_t w = c.geom == GLayered1D ? 16 : 8, h = c.geom == GLayered1D ? 0 : (cube ? 8 : 6);
    const size_t slices = cube ? 6 * layers : layers;
    const unsigned levels = c.geom == GLayered1D ? 5 : 4;
    unsigned flags = cudaArrayLayered;
    if (cube) flags = c.geom == GCube ? cudaArrayCubemap : (cudaArrayCubemap | cudaArrayLayered);
    cudaMipmappedArray_t mip;
    cudaError_t err = cudaMallocMipmappedArray(&mip, &cd, make_cudaExtent(w, h, slices), levels, flags);
    if (err != cudaSuccess) { printf("%s: cudaMallocMipmappedArray: %s\n", c.tag.c_str(), cudaGetErrorString(err)); return 1; }
    for (unsigned l = 0; l < levels; ++l) {
      cudaArray_t lv;
      cudaGetMipmappedArrayLevel(&lv, mip, l);
      const size_t lw = std::max<size_t>(1, w >> l), lh = h ? std::max<size_t>(1, h >> l) : 1;
      std::vector<unsigned char> buf(lw * lh * slices * tb);
      for (size_t i = 0; i < buf.size() / tb; ++i) {
        const unsigned seed = static_cast<unsigned>(i * 2654435761u + l * 97 + 13);
        if (c.unorm8) {
          for (int b = 0; b < 4; ++b) buf[i * 4 + b] = static_cast<unsigned char>(seed >> (5 + 6 * b));
        } else {
          const float v = static_cast<float>(l * 1000 + (seed >> 20) % 997) / 64.0f;
          std::memcpy(&buf[i * 4], &v, 4);
        }
      }
      cudaMemcpy3DParms p{};
      p.srcPtr = make_cudaPitchedPtr(buf.data(), lw * tb, lw, lh);
      p.dstArray = lv;
      p.extent = make_cudaExtent(lw, h ? lh : 1, slices);
      p.kind = cudaMemcpyHostToDevice;
      if ((err = cudaMemcpy3D(&p)) != cudaSuccess) { printf("%s: cudaMemcpy3D: %s\n", c.tag.c_str(), cudaGetErrorString(err)); return 1; }
    }
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeMipmappedArray;
    rd.res.mipmap.mipmap = mip;
    cudaTextureDesc td{};
    for (auto& m : td.addressMode) m = c.mode;
    td.filterMode = c.linear ? cudaFilterModeLinear : cudaFilterModePoint;
    td.mipmapFilterMode = c.mip_linear ? cudaFilterModeLinear : cudaFilterModePoint;
    td.readMode = c.unorm8 ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    td.normalizedCoords = 1;
    td.maxMipmapLevelClamp = static_cast<float>(levels - 1);
    cudaTextureObject_t t;
    if ((err = cudaCreateTextureObject(&t, &rd, &td, nullptr)) != cudaSuccess) {
      printf("%s: cudaCreateTextureObject: %s\n", c.tag.c_str(), cudaGetErrorString(err));
      return 1;
    }
    fetch<float4><<<kN / 128, 128>>>(t, c.geom, c.layer, dout);
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) { printf("%s: the fetch: %s\n", c.tag.c_str(), cudaGetErrorString(err)); return 1; }
    std::vector<unsigned> r(4 * kN);
    cudaMemcpy(r.data(), dout, sizeof(float4) * kN, cudaMemcpyDeviceToHost);
    cudaDestroyTextureObject(t);
    cudaFreeMipmappedArray(mip);
    if (ci == dump) {
      for (int k = 0; k < kN; ++k)
        printf("%4d %08x %08x %08x %08x\n", k, r[4 * k], r[4 * k + 1], r[4 * k + 2], r[4 * k + 3]);
      return 0;
    }
    unsigned long long hsh = 1469598103934665603ull;
    for (unsigned x : r) { hsh ^= x; hsh *= 1099511628211ull; }
    if (print) { printf("    {\"%s\", 0x%016llxull},\n", c.tag.c_str(), hsh); continue; }
    if (dump >= 0) continue;
    if (ci >= nexp || std::strcmp(kExpected[ci].tag, c.tag.c_str()) || hsh != kExpected[ci].hash)
      if (bad++ < 30)
        printf("%d %s: 0x%016llx, the card gave 0x%016llx\n", ci, c.tag.c_str(), hsh, ci < nexp ? kExpected[ci].hash : 0ull);
  }
  if (print) return 0;
  printf("%zu mipmapped layered and cubemap cases, %d differ\n%s\n", cases.size(), bad,
         bad || nexp != static_cast<int>(cases.size()) ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
