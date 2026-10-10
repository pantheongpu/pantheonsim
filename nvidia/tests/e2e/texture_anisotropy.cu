// maxAnisotropy and explicit-level fetches, against an RTX 3060 (sm_86).
//
// A texture whose descriptor asks for anisotropic filtering (maxAnisotropy of 2 or
// more) and whose mip filter is linear blends the two levels an explicit level of
// detail falls between with a sharper weight than the fraction of the level: the
// blend stays at the lower level for the first part of the fraction and at the
// upper for the last, in between going up 1.5, 1.75 or 2 times as fast for a
// maxAnisotropy of 2-3, 4-7 or 8 and over. The weight is measured as 256ths over
// every fraction, every geometry (1D, 2D, 3D, cubemap, layered), every mip filter,
// bias and level clamp in nvidia/docs/textures.md. A plain fetch, and a fetch
// whose mip filter is point, are not affected.
//
// Each case's results are hashed and compared with the card's, recorded in
// texture_anisotropy_expected.inc (`texture_anisotropy --print` prints them;
// `--dump N` prints case N's values). Prints PASS on the last line, and runs the
// same on a GPU.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "texture_anisotropy_expected.inc"
};

constexpr int kLevels = 5;
constexpr int kFine = 4 * 256;   // explicit levels of detail in 1/256ths: every fraction of four levels
constexpr int kCoords = 6;
constexpr int kN = kFine * kCoords;   // fetches per case

__constant__ float kCoord[kCoords] = {0.07f, 0.31f, 0.5f, 0.63f, 0.9f, 1.3f};

enum Geom { G1D, G2D, G3D, GCube, G1DLayered, G2DLayered, GCubeLayered, GCount };
static const char* kGeom[] = {"1d", "2d", "3d", "cube", "1d-layered", "2d-layered", "cube-layered"};

template <class T>
__global__ void fetch(cudaTextureObject_t t, int geom, bool plain, float bias_lod_shift, T* out) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int q = i % kFine;
  const float lod = static_cast<float>(q) / 256.0f + bias_lod_shift;
  const float x = kCoord[i / kFine], y = kCoord[(i / kFine + 2) % kCoords], z = kCoord[(i / kFine + 4) % kCoords];
  T r{};
  switch (geom) {
    case G1D: r = plain ? tex1D<T>(t, x) : tex1DLod<T>(t, x, lod); break;
    case G2D: r = plain ? tex2D<T>(t, x, y) : tex2DLod<T>(t, x, y, lod); break;
    case G3D: r = plain ? tex3D<T>(t, x, y, z) : tex3DLod<T>(t, x, y, z, lod); break;
    case GCube: r = plain ? texCubemap<T>(t, x - 0.5f, y - 0.5f, z - 0.4f) : texCubemapLod<T>(t, x - 0.5f, y - 0.5f, z - 0.4f, lod); break;
    case G1DLayered: r = plain ? tex1DLayered<T>(t, x, 1) : tex1DLayeredLod<T>(t, x, 1, lod); break;
    case G2DLayered: r = plain ? tex2DLayered<T>(t, x, y, 1) : tex2DLayeredLod<T>(t, x, y, 1, lod); break;
    case GCubeLayered:
      r = plain ? texCubemapLayered<T>(t, x - 0.5f, y - 0.5f, z - 0.4f, 1)
                : texCubemapLayeredLod<T>(t, x - 0.5f, y - 0.5f, z - 0.4f, 1, lod);
      break;
  }
  out[i] = r;
}

struct Case {
  std::string tag;
  int geom;
  bool unorm8;
  bool linear, mip_linear;
  unsigned aniso;
  float bias;
  float min_clamp, max_clamp;
  bool plain = false;
  float lod_shift = 0;   // added to the explicit level of detail
};

int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  const int dump = argc > 2 && std::strcmp(argv[1], "--dump") == 0 ? atoi(argv[2]) : -1;
  std::vector<Case> cases;
  auto add = [&](std::string tag, int geom, bool u8, bool lin, bool mlin, unsigned an, float bias, float mn, float mx,
                 bool plain = false) {
    cases.push_back({std::move(tag), geom, u8, lin, mlin, an, bias, mn, mx, plain});
  };
  const float kNoMax = kLevels - 1;
  // every geometry, both texel types, point and linear filtering, one class of each size
  for (int g = 0; g < GCount; ++g)
    for (int u8 = 0; u8 < 2; ++u8)
      for (int lin = 0; lin < 2; ++lin)
        for (unsigned an : {2u, 4u, 8u})
          add(std::string(kGeom[g]) + (u8 ? " u8x4n" : " f32") + (lin ? " linear" : " point") + " mip-linear aniso" +
                  std::to_string(an),
              g, u8, lin, true, an, 0.f, 0.f, kNoMax);
  // every maxAnisotropy value that sorts differently, and ones that do not
  for (unsigned an : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 12u, 16u, 17u, 100u, 255u, 65536u, 0xFFFFFFFFu})
    add("2d f32 linear mip-linear aniso" + std::to_string(an), G2D, false, true, true, an, 0.f, 0.f, kNoMax);
  // the bias, in 1/256ths of a level (truncated), against the thresholds the ramp's start moves at
  for (unsigned an : {2u, 3u, 4u, 7u, 8u, 16u})
    for (int bq : {-300, -40, -1, 0, 1, 42, 43, 44, 54, 55, 56, 63, 64, 65, 120, 255, 256, 300}) {
      char name[96];
      std::snprintf(name, sizeof name, "2d f32 linear mip-linear aniso%u bias%d/256", an, bq);
      add(name, G2D, false, true, true, an, bq / 256.0f, 0.f, kNoMax);
    }
  // a bias that is not a multiple of 1/256, in 256ths: where the ramp starts moves with its fraction
  // against 128/3, 3 * 128/7 and 64 (the thresholds of the three rates)
  for (float bq : {0.3f, 0.9f, 42.6f, 42.7f, 42.9f, 43.2f, 54.8f, 54.9f, 55.3f, 63.9f, 64.1f, -0.5f, -1.5f, 130.3f})
    for (unsigned an : {2u, 4u, 8u}) {
      char name[96];
      std::snprintf(name, sizeof name, "2d f32 linear mip-linear aniso%u bias%.1f/256", an, bq);
      add(name, G2D, false, true, true, an, bq / 256.0f, 0.f, kNoMax);
    }
  // level clamps apply to the sharpened level of detail
  // (an array: nvcc 12.0 to 12.8 cannot deduce the element type of a braced list written in the loop header)
  static const std::pair<float, float> kClamps[] = {{1.4f, 4.f}, {0.f, 2.3f}, {1.4f, 2.6f}, {2.0f, 2.0f}, {0.6f, 3.9f}, {3.0f, 1.0f}};
  for (unsigned an : {2u, 4u, 8u})
    for (auto [mn, mx] : kClamps)
      for (float bias : {0.f, 0.4f}) {
        char name[96];
        std::snprintf(name, sizeof name, "2d f32 linear mip-linear aniso%u min%.1f max%.1f bias%.1f", an, mn, mx, bias);
        add(name, G2D, false, true, true, an, bias, mn, mx);
      }
  // not affected: a point mip filter, and a fetch with no explicit level
  for (unsigned an : {2u, 16u}) {
    add("2d f32 linear mip-point aniso" + std::to_string(an), G2D, false, true, false, an, 0.f, 0.f, kNoMax);
    add("2d f32 point mip-point aniso" + std::to_string(an), G2D, false, false, false, an, 0.f, 0.f, kNoMax);
    add("2d f32 linear mip-linear plain aniso" + std::to_string(an), G2D, false, true, true, an, 0.f, 0.f, kNoMax, true);
    add("cube u8x4n linear mip-linear plain aniso" + std::to_string(an), GCube, true, true, true, an, 0.f, 0.f, kNoMax, true);
  }

  float4* dout;
  cudaMalloc(&dout, sizeof(float4) * kN);
  int bad = 0;
  const int nexp = sizeof kExpected / sizeof kExpected[0];
  for (int ci = 0; ci < static_cast<int>(cases.size()); ++ci) {
    const Case& c = cases[ci];
    const cudaChannelFormatDesc cd = c.unorm8 ? cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned)
                                              : cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    const bool cube = c.geom == GCube || c.geom == GCubeLayered;
    const bool layered = c.geom == G1DLayered || c.geom == G2DLayered || c.geom == GCubeLayered;
    const size_t w = 16;
    const size_t h = (c.geom == G1D || c.geom == G1DLayered) ? 0 : 16;
    const size_t d = c.geom == G3D ? 16 : 0;
    const size_t layers = layered ? 3 : 1;
    const size_t slices = cube ? 6 * layers : layers;
    unsigned flags = 0;
    if (layered) flags |= cudaArrayLayered;
    if (cube) flags |= cudaArrayCubemap;
    cudaExtent ext = make_cudaExtent(w, h, c.geom == G3D ? d : (layered || cube) ? slices : 0);
    cudaMipmappedArray_t mip;
    cudaError_t err = cudaMallocMipmappedArray(&mip, &cd, ext, kLevels, flags);
    if (err != cudaSuccess) { printf("%s: cudaMallocMipmappedArray: %s\n", c.tag.c_str(), cudaGetErrorString(err)); return 1; }
    for (unsigned l = 0; l < kLevels; ++l) {
      cudaArray_t lv;
      cudaGetMipmappedArrayLevel(&lv, mip, l);
      const size_t lw = std::max<size_t>(1, w >> l), lh = h ? std::max<size_t>(1, h >> l) : 1;
      const size_t ld = c.geom == G3D ? std::max<size_t>(1, d >> l) : slices;
      std::vector<unsigned char> buf(lw * lh * ld * 4);
      for (size_t i = 0; i < buf.size() / 4; ++i) {
        const unsigned seed = static_cast<unsigned>(i * 2654435761u + l * 97 + 13);
        if (c.unorm8) {
          for (int b = 0; b < 4; ++b) buf[i * 4 + b] = static_cast<unsigned char>(seed >> (5 + 6 * b));
        } else {
          const float v = static_cast<float>(l * 1000 + (seed >> 20) % 997) / 64.0f;
          std::memcpy(&buf[i * 4], &v, 4);
        }
      }
      cudaMemcpy3DParms p{};
      p.srcPtr = make_cudaPitchedPtr(buf.data(), lw * 4, lw, lh);
      p.dstArray = lv;
      p.extent = make_cudaExtent(lw, h ? lh : 1, ld);
      p.kind = cudaMemcpyHostToDevice;
      if ((err = cudaMemcpy3D(&p)) != cudaSuccess) { printf("%s: cudaMemcpy3D: %s\n", c.tag.c_str(), cudaGetErrorString(err)); return 1; }
    }
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeMipmappedArray;
    rd.res.mipmap.mipmap = mip;
    cudaTextureDesc td{};
    for (auto& m : td.addressMode) m = cudaAddressModeClamp;
    td.filterMode = c.linear ? cudaFilterModeLinear : cudaFilterModePoint;
    td.mipmapFilterMode = c.mip_linear ? cudaFilterModeLinear : cudaFilterModePoint;
    td.readMode = c.unorm8 ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    td.normalizedCoords = 1;
    td.maxAnisotropy = c.aniso;
    td.mipmapLevelBias = c.bias;
    td.minMipmapLevelClamp = c.min_clamp;
    td.maxMipmapLevelClamp = c.max_clamp;
    cudaTextureObject_t t;
    if ((err = cudaCreateTextureObject(&t, &rd, &td, nullptr)) != cudaSuccess) {
      printf("%s: cudaCreateTextureObject: %s\n", c.tag.c_str(), cudaGetErrorString(err));
      return 1;
    }
    fetch<float4><<<kN / 128, 128>>>(t, c.geom, c.plain, c.lod_shift, dout);
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) { printf("%s: the fetch: %s\n", c.tag.c_str(), cudaGetErrorString(err)); return 1; }
    std::vector<unsigned> r(4 * kN);
    cudaMemcpy(r.data(), dout, sizeof(float4) * kN, cudaMemcpyDeviceToHost);
    cudaDestroyTextureObject(t);
    cudaFreeMipmappedArray(mip);
    if (ci == dump) {
      for (int k = 0; k < kN; ++k)
        printf("%5d %08x %08x %08x %08x\n", k, r[4 * k], r[4 * k + 1], r[4 * k + 2], r[4 * k + 3]);
      cudaFree(dout);
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
  cudaFree(dout);
  if (print) return 0;
  printf("%zu anisotropic explicit-level cases, %d differ\n%s\n", cases.size(), bad,
         bad || nexp != static_cast<int>(cases.size()) ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
