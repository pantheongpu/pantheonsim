// tex.grad -- the level of detail an RTX 3060 derives from explicit gradients -- checked against the card.
//
// A mipmapped texture, each level filled with its own noise, is fetched with tex1DGrad, tex2DGrad and their
// layered forms (tex.grad.1d, .2d, .a1d and .a2d: SASS TXD) for thousands of random gradient pairs: every size of
// gradient, every relative size and sign of the components, trilinear and nearest-level blends, level bias and
// clamps, clamp and wrap addressing; and the special values (zero, infinite, NaN, huge and tiny gradients). The
// results are hashed and compared with the card's, recorded in texture_grad_expected.inc (`texture_grad --print`
// prints them). The program passes on a GPU as well (build it with the same nvcc and run it).
//
// Sizes that are not a power of two are covered too (the "npot" cases: the card multiplies the gradient by the
// size in a way of its own, see exec/texture_grad.hpp).
//
// Not covered, and refused by the simulator by name: 3D and cube textures (ptxas builds those from quads of
// plain fetches) and maxAnisotropy above 1.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Expected {
  const char* tag;
  unsigned long long hash;
};
static const Expected kExpected[] = {
#include "texture_grad_expected.inc"
};

static bool g_print = false;
static int g_fails = 0;
static uint32_t g_rng = 20261010;
static uint32_t rnd() {
  g_rng = g_rng * 1664525u + 1013904223u;
  return g_rng >> 8;
}
static float frand(float lo, float hi) { return lo + (hi - lo) * (rnd() & 0xFFFFFF) / float(1 << 24); }

static unsigned long long fnv(unsigned long long h, const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}
static void report(const std::string& tag, unsigned long long h) {
  if (g_print) {
    std::printf("    {\"%s\", 0x%016llxull},\n", tag.c_str(), h);
    return;
  }
  const Expected* e = nullptr;
  for (const Expected& x : kExpected)
    if (tag == x.tag) e = &x;
  const bool ok = e && e->hash == h;
  std::printf("%-40s %016llx %s\n", tag.c_str(), h, ok ? "ok" : e ? "MISMATCH" : "MISSING");
  g_fails += !ok;
}

// One fetch per thread: in = {layer, x, y, dpdx.x, dpdx.y, dpdy.x, dpdy.y} as floats (unused ones ignored).
__global__ void k1d(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex1DGrad<float>(t, in[7 * i + 1], in[7 * i + 3], in[7 * i + 5]);
}
__global__ void k2d(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = tex2DGrad<float>(t, in[7 * i + 1], in[7 * i + 2], make_float2(in[7 * i + 3], in[7 * i + 4]),
                              make_float2(in[7 * i + 5], in[7 * i + 6]));
}
__global__ void k1dl(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex1DLayeredGrad<float>(t, in[7 * i + 1], int(in[7 * i]), in[7 * i + 3], in[7 * i + 5]);
}
__global__ void k2dl(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = tex2DLayeredGrad<float>(t, in[7 * i + 1], in[7 * i + 2], int(in[7 * i]),
                                     make_float2(in[7 * i + 3], in[7 * i + 4]),
                                     make_float2(in[7 * i + 5], in[7 * i + 6]));
}

enum Kind { K1D, K2D, K1DL, K2DL };

struct Tex {
  cudaMipmappedArray_t mip = nullptr;
  cudaArray_t arr = nullptr;
  cudaTextureObject_t t = 0;
  ~Tex() {
    if (t) cudaDestroyTextureObject(t);
    if (mip) cudaFreeMipmappedArray(mip);
    if (arr) cudaFreeArray(arr);
  }
};

// A float texture of `w` x `h` (h = 0 for 1D) texels and `layers` layers (0: not layered), `levels` levels (0:
// not mipmapped), every texel a different number.
static bool make_texture(Tex* tx, int w, int h, int layers, int levels, int filter_linear, int mip_linear,
                         int address, float bias, float min_clamp, float max_clamp, unsigned aniso = 1) {
  const cudaChannelFormatDesc d = cudaCreateChannelDesc<float>();
  const unsigned flags = layers ? cudaArrayLayered : 0;
  cudaExtent ext = make_cudaExtent(w, h, layers);
  cudaResourceDesc rd = {};
  if (levels) {
    if (cudaMallocMipmappedArray(&tx->mip, &d, ext, levels, flags) != cudaSuccess) return false;
    for (int l = 0; l < levels; ++l) {
      cudaArray_t a = nullptr;
      cudaGetMipmappedArrayLevel(&a, tx->mip, l);
      const int lw = w >> l ? w >> l : 1, lh = h ? (h >> l ? h >> l : 1) : 1, ln = layers ? layers : 1;
      std::vector<float> host(size_t(lw) * lh * ln);
      for (size_t i = 0; i < host.size(); ++i) host[i] = float(rnd() & 0xFFFF) / 64.0f + float(l) * 1000.0f;
      cudaMemcpy3DParms p = {};
      p.srcPtr = make_cudaPitchedPtr(host.data(), lw * 4, lw, lh);
      p.dstArray = a;
      p.extent = make_cudaExtent(lw, h ? lh : 1, layers ? layers : 1);
      p.kind = cudaMemcpyHostToDevice;
      if (cudaMemcpy3D(&p) != cudaSuccess) return false;
    }
    rd.resType = cudaResourceTypeMipmappedArray;
    rd.res.mipmap.mipmap = tx->mip;
  } else {
    if (cudaMallocArray(&tx->arr, &d, w, h, flags) != cudaSuccess && !layers) return false;
    if (layers && cudaMalloc3DArray(&tx->arr, &d, ext, flags) != cudaSuccess) return false;
    const int ln = layers ? layers : 1, lh = h ? h : 1;
    std::vector<float> host(size_t(w) * lh * ln);
    for (auto& v : host) v = float(rnd() & 0xFFFF) / 64.0f;
    cudaMemcpy3DParms p = {};
    p.srcPtr = make_cudaPitchedPtr(host.data(), w * 4, w, lh);
    p.dstArray = tx->arr;
    p.extent = make_cudaExtent(w, h ? h : 1, ln);
    p.kind = cudaMemcpyHostToDevice;
    if (cudaMemcpy3D(&p) != cudaSuccess) return false;
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = tx->arr;
  }
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = static_cast<cudaTextureAddressMode>(address);
  td.filterMode = filter_linear ? cudaFilterModeLinear : cudaFilterModePoint;
  td.mipmapFilterMode = mip_linear ? cudaFilterModeLinear : cudaFilterModePoint;
  td.readMode = cudaReadModeElementType;
  td.normalizedCoords = 1;
  td.maxAnisotropy = aniso;
  td.mipmapLevelBias = bias;
  td.minMipmapLevelClamp = min_clamp;
  td.maxMipmapLevelClamp = max_clamp;
  return cudaCreateTextureObject(&tx->t, &rd, &td, nullptr) == cudaSuccess;
}

// A random gradient: a size 2^-e (e from `emin` to `emax`) times a mantissa, each component scaled and signed at
// random; sometimes one component zero or the two vectors parallel.
static void random_gradients(float* g, int emin, int emax, int mode) {
  const float mag = std::ldexp(frand(1.0f, 2.0f), -(emin + int(rnd() % unsigned(emax - emin + 1))));
  for (int i = 0; i < 4; ++i) {
    float s = frand(0.125f, 1.0f);
    if (rnd() & 1) s = -s;
    g[i] = mag * s;
  }
  if (mode == 1) {          // dPdx and dPdy parallel
    const float s = frand(-1.5f, 1.5f);
    g[2] = g[0] * s;
    g[3] = g[1] * s;
  } else if (mode == 2) {   // components zero
    g[rnd() & 3] = 0.0f;
  }
}

static unsigned long long run(Kind kind, const Tex& tx, int layers, const std::vector<float>& in) {
  const int n = int(in.size() / 7);
  float *din = nullptr, *dout = nullptr;
  cudaMalloc(&din, in.size() * 4);
  cudaMalloc(&dout, size_t(n) * 4);
  cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
  (void)layers;
  const int blocks = (n + 127) / 128;
  switch (kind) {
    case K1D: k1d<<<blocks, 128>>>(tx.t, din, dout, n); break;
    case K2D: k2d<<<blocks, 128>>>(tx.t, din, dout, n); break;
    case K1DL: k1dl<<<blocks, 128>>>(tx.t, din, dout, n); break;
    case K2DL: k2dl<<<blocks, 128>>>(tx.t, din, dout, n); break;
  }
  std::vector<float> out(n);
  const cudaError_t e = cudaMemcpy(out.data(), dout, size_t(n) * 4, cudaMemcpyDeviceToHost);
  cudaFree(din);
  cudaFree(dout);
  if (e != cudaSuccess) {
    std::printf("kernel failed: %s\n", cudaGetErrorString(e));
    cudaGetLastError();
    return 0;
  }
  return fnv(1469598103934665603ull, out.data(), out.size() * 4);
}

struct Case {
  const char* name;
  Kind kind;
  int w, h, layers, levels, filter_linear, mip_linear, address;
  float bias, min_clamp, max_clamp;
  int emin, emax;
};

int main(int argc, char** argv) {
  g_print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  cudaFree(0);
  static const Case cases[] = {
      // name                 kind  w     h    L  lv lin mip addr bias  min  max  emin emax
      {"2d-trilinear-1024x512", K2D, 1024, 512, 0, 11, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 13},
      {"2d-trilinear-wrap", K2D, 256, 256, 0, 9, 1, 1, 0, 0.0f, 0.0f, 20.0f, 1, 11},
      {"2d-nearest-level", K2D, 512, 512, 0, 10, 0, 0, 1, 0.0f, 0.0f, 20.0f, 1, 12},
      {"2d-point-trilinear", K2D, 512, 128, 0, 10, 0, 1, 1, 0.0f, 0.0f, 20.0f, 1, 12},
      {"2d-bias-clamps", K2D, 1024, 1024, 0, 11, 1, 1, 1, 0.75f, 1.5f, 6.25f, 1, 13},
      {"2d-negative-bias", K2D, 512, 512, 0, 10, 1, 1, 1, -1.5f, 0.0f, 20.0f, 1, 12},
      {"2d-4096x1024", K2D, 4096, 1024, 0, 13, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 16},
      {"2d-tiny-8x8", K2D, 8, 8, 0, 4, 1, 1, 1, 0.0f, 0.0f, 20.0f, 0, 6},
      {"2d-not-mipmapped", K2D, 256, 128, 0, 0, 1, 0, 1, 0.0f, 0.0f, 20.0f, 0, 12},
      {"1d-trilinear-2048", K1D, 2048, 0, 0, 12, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 14},
      {"1d-nearest-level", K1D, 1024, 0, 0, 11, 0, 0, 1, 0.0f, 0.0f, 20.0f, 1, 13},
      {"1d-not-mipmapped", K1D, 512, 0, 0, 0, 1, 0, 1, 0.0f, 0.0f, 20.0f, 0, 12},
      {"1d-layered", K1DL, 1024, 0, 5, 11, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 13},
      {"1d-layered-bias", K1DL, 512, 0, 3, 10, 1, 1, 0, 0.5f, 0.0f, 7.5f, 1, 12},
      {"2d-layered", K2DL, 512, 256, 3, 10, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 12},
      {"2d-layered-nearest", K2DL, 256, 256, 4, 9, 0, 0, 1, 0.0f, 0.0f, 20.0f, 1, 11},
      // Sizes that are not a power of two.
      {"2d-npot-100x60", K2D, 100, 60, 0, 7, 1, 1, 1, 0.0f, 0.0f, 20.0f, 0, 9},
      {"2d-npot-1023x1025", K2D, 1023, 1025, 0, 11, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 13},
      {"2d-npot-wrap-300x200", K2D, 300, 200, 0, 9, 1, 1, 0, 0.0f, 0.0f, 20.0f, 1, 11},
      {"2d-npot-nearest-level-129x257", K2D, 129, 257, 0, 9, 0, 0, 1, 0.0f, 0.0f, 20.0f, 1, 11},
      {"2d-npot-bias-clamps-513x259", K2D, 513, 259, 0, 10, 1, 1, 1, 0.5f, 1.25f, 6.5f, 1, 12},
      {"2d-npot-3x5", K2D, 3, 5, 0, 3, 1, 1, 1, 0.0f, 0.0f, 20.0f, 0, 6},
      {"2d-npot-4097x100", K2D, 4097, 100, 0, 13, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 15},
      {"2d-npot-pow2-by-npot-512x100", K2D, 512, 100, 0, 10, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 12},
      {"1d-npot-1000", K1D, 1000, 0, 0, 10, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 13},
      {"1d-npot-7", K1D, 7, 0, 0, 3, 1, 1, 1, 0.0f, 0.0f, 20.0f, 0, 6},
      {"1d-layered-npot-333", K1DL, 333, 0, 5, 9, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 12},
      {"2d-layered-npot-130x70", K2DL, 130, 70, 3, 8, 1, 1, 1, 0.0f, 0.0f, 20.0f, 1, 11},
  };
  for (const Case& c : cases) {
    g_rng = 20261010u ^ (uint32_t)(c.w * 31 + c.h * 7 + c.layers);
    Tex tx;
    if (!make_texture(&tx, c.w, c.h, c.layers, c.levels, c.filter_linear, c.mip_linear, c.address, c.bias,
                      c.min_clamp, c.max_clamp)) {
      std::printf("FAIL %s: cannot make the texture\n", c.name);
      ++g_fails;
      continue;
    }
    const int n = 6000;
    for (int mode = 0; mode < 3; ++mode) {
      std::vector<float> in;
      for (int i = 0; i < n; ++i) {
        float g[4];
        random_gradients(g, c.emin, c.emax, mode);
        in.push_back(float(c.layers ? int(rnd() % unsigned(c.layers + 1)) : 0));   // one past the end is a case too
        in.push_back(frand(-0.25f, 1.25f));
        in.push_back(frand(-0.25f, 1.25f));
        in.push_back(g[0]);
        in.push_back(g[1]);
        in.push_back(g[2]);
        in.push_back(g[3]);
      }
      report(std::string(c.name) + "/mode" + std::to_string(mode), run(c.kind, tx, c.layers, in));
    }
  }
  // Special values: zero, infinite, NaN, huge and tiny gradients, in 1D and 2D, with and without a level bias.
  for (int variant = 0; variant < 2; ++variant) {
    Tex t2, t1;
    g_rng = 77u + variant;
    if (!make_texture(&t2, 512, 512, 0, 10, 1, 1, 1, variant ? 3.0f : 0.0f, 0.0f, 20.0f) ||
        !make_texture(&t1, 1024, 0, 0, 11, 1, 1, 1, variant ? -2.0f : 0.0f, 0.0f, 20.0f)) {
      std::printf("FAIL special: cannot make the textures\n");
      ++g_fails;
      continue;
    }
    const float inf = 1.0f / 0.0f, nan = 0.0f / 0.0f;
    const float vals[] = {0.0f, -0.0f, 1e-30f, 1e-38f, 1e-40f, 0.001f, 0.0625f, 0.5f, 1.0f, 1e30f, 3e38f, inf, -inf, nan};
    std::vector<float> in;
    for (float a : vals)
      for (float b : vals) {
        float x = frand(0.0f, 1.0f);
        const float row[7] = {0, x, frand(0.0f, 1.0f), a, b, b, a};
        in.insert(in.end(), row, row + 7);
      }
    report(std::string("special-2d/") + std::to_string(variant), run(K2D, t2, 0, in));
    report(std::string("special-1d/") + std::to_string(variant), run(K1D, t1, 0, in));
    Tex t2n, t1n;
    if (!make_texture(&t2n, 300, 190, 0, 9, 1, 1, 1, variant ? 3.0f : 0.0f, 0.0f, 20.0f) ||
        !make_texture(&t1n, 1000, 0, 0, 10, 1, 1, 1, variant ? -2.0f : 0.0f, 0.0f, 20.0f)) {
      std::printf("FAIL special npot: cannot make the textures\n");
      ++g_fails;
      continue;
    }
    report(std::string("special-2d-npot/") + std::to_string(variant), run(K2D, t2n, 0, in));
    report(std::string("special-1d-npot/") + std::to_string(variant), run(K1D, t1n, 0, in));
  }
  if (g_print) return 0;
  std::printf("%d mismatches\n%s\n", g_fails, g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
