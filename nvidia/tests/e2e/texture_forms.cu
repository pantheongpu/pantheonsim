// The texture instruction's operands beyond a plain fetch, checked against an
// RTX 3060 (sm_86), written as PTX: the offset vector (immediate and register,
// tex and tld4, every addressing mode, point and linear, 1D to 3D, layered
// and mipmapped), the depth-compare reference (which CUDA's textures ignore),
// the destination predicate, half-precision results (.f16 and .f16x2), tld4
// on layered textures, cubemaps and cubemap arrays, and the linear filtering
// of signed 8-bit normalized texels. Each case's results are hashed and
// compared with the card's, recorded in texture_forms_expected.inc
// (`texture_forms --print` prints them). Prints PASS on the last line, and runs
// the same on a GPU.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "texture_forms_expected.inc"
};

static bool g_print = false;
static int g_fails = 0, g_skips = 0;
static uint32_t g_rng = 20261009;
static float frand(float lo, float hi) { g_rng = g_rng * 1664525u + 1013904223u; return lo + (hi - lo) * ((g_rng >> 8) & 0xFFFFFF) / float(1 << 24); }

static unsigned long long fnv(unsigned long long h, const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
static const unsigned long long kBasis = 1469598103934665603ull;
static void report(const std::string& tag, unsigned long long h) {
  if (g_print) { std::printf("    {\"%s\", 0x%016llxull},\n", tag.c_str(), h); return; }
  const Expected* e = nullptr;
  for (const Expected& x : kExpected) if (tag == x.tag) e = &x;
  const bool ok = e && e->hash == h;
  std::printf("%-34s %016llx %s\n", tag.c_str(), h, ok ? "ok" : e ? "MISMATCH" : "MISSING");
  g_fails += !ok;
}

// ---- kernels ------------------------------------------------------------------
// Offsets are PTX immediates, so each combination is its own kernel.
template <int OX, int OY> __global__ void k2d(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1], r0, r1, r2, r3;
  asm volatile("tex.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], {%7,%8};" : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
               : "l"(t), "f"(x), "f"(y), "n"(OX), "n"(OY));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
template <int OX> __global__ void k1d(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], r0, r1, r2, r3;
  asm volatile("tex.1d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5}], {%6};" : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
               : "l"(t), "f"(x), "n"(OX));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
template <int OX, int OY, int OZ> __global__ void k3d(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[3 * i], y = in[3 * i + 1], z = in[3 * i + 2], r0, r1, r2, r3;
  asm volatile("tex.3d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}], {%8,%9,%10,%10};"
               : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y), "f"(z), "n"(OX), "n"(OY), "n"(OZ));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
template <int OX, int OY> __global__ void ka2d(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  unsigned l = (unsigned)in[3 * i]; float x = in[3 * i + 1], y = in[3 * i + 2], r0, r1, r2, r3;
  asm volatile("tex.a2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}], {%8,%9};"
               : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "r"(l), "f"(x), "f"(y), "n"(OX), "n"(OY));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
template <int OX, int OY> __global__ void klevel(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[3 * i], y = in[3 * i + 1], lod = in[3 * i + 2], r0, r1, r2, r3;
  asm volatile("tex.level.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], %7, {%8,%9};"
               : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y), "f"(lod), "n"(OX), "n"(OY));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
template <int OX, int OY> __global__ void kgather(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1], r0, r1, r2, r3;
  asm volatile("tld4.r.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], {%7,%8};"
               : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y), "n"(OX), "n"(OY));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
// Register offsets: ptxas packs them four bits an axis for tex, six for tld4.
__global__ void kreg(cudaTextureObject_t t, const float* in, const int* offs, float* out, int n, int gather) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1], r0, r1, r2, r3; int ox = offs[2 * i], oy = offs[2 * i + 1];
  if (gather)
    asm volatile("tld4.r.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], {%7,%8};"
                 : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y), "r"(ox), "r"(oy));
  else
    asm volatile("tex.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], {%7,%8};"
                 : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y), "r"(ox), "r"(oy));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
// The destination predicate and the depth reference.
__global__ void kpred(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1], r0, r1, r2, r3; unsigned p;
  asm volatile("{ .reg .pred q; tex.2d.v4.f32.f32 {%0,%1,%2,%3}|q, [%5, {%6,%7}]; selp.u32 %4, 1, 0, q; }"
               : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3), "=r"(p) : "l"(t), "f"(x), "f"(y));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = (float)p;
}
__global__ void kplain(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1], r0, r1, r2, r3;
  asm volatile("tex.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}];" : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
__global__ void kdepth(cudaTextureObject_t t, const float* in, float* out, int n, int gather) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1], ref = x * 3.0f - 5.0f, r0, r1, r2, r3;
  if (gather)
    asm volatile("tld4.r.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], %7;"
                 : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y), "f"(ref));
  else
    asm volatile("tex.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], %7;"
                 : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y), "f"(ref));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
__global__ void kplaing(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1], r0, r1, r2, r3;
  asm volatile("tld4.r.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}];" : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
// Half-precision results.
__global__ void kh4(cudaTextureObject_t t, const float* in, unsigned short* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1]; unsigned short a, b, c, d;
  asm volatile("tex.2d.v4.f16.f32 {%0,%1,%2,%3}, [%4, {%5,%6}];" : "=h"(a), "=h"(b), "=h"(c), "=h"(d) : "l"(t), "f"(x), "f"(y));
  out[4 * i] = a; out[4 * i + 1] = b; out[4 * i + 2] = c; out[4 * i + 3] = d;
}
__global__ void kh2(cudaTextureObject_t t, const float* in, unsigned* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[2 * i], y = in[2 * i + 1]; unsigned a, b;
  asm volatile("tex.2d.v2.f16x2.f32 {%0,%1}, [%2, {%3,%4}];" : "=r"(a), "=r"(b) : "l"(t), "f"(x), "f"(y));
  out[2 * i] = a; out[2 * i + 1] = b;
}
// tld4 on layered textures, cubemaps and cubemap arrays.
__global__ void kga2d(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  unsigned l = (unsigned)in[3 * i]; float x = in[3 * i + 1], y = in[3 * i + 2], r0, r1, r2, r3;
  asm volatile("tld4.r.a2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}];"
               : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "r"(l), "f"(x), "f"(y));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
__global__ void kgcube(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = in[3 * i], y = in[3 * i + 1], z = in[3 * i + 2], r0, r1, r2, r3;
  asm volatile("tld4.r.cube.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}];"
               : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "f"(x), "f"(y), "f"(z));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
__global__ void kgacube(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  unsigned l = (unsigned)in[4 * i]; float x = in[4 * i + 1], y = in[4 * i + 2], z = in[4 * i + 3], r0, r1, r2, r3;
  asm volatile("tld4.r.acube.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%8}];"
               : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "l"(t), "r"(l), "f"(x), "f"(y), "f"(z));
  out[4 * i] = r0; out[4 * i + 1] = r1; out[4 * i + 2] = r2; out[4 * i + 3] = r3;
}
__global__ void kfetch1(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = tex1D<float>(t, in[i]);
}
__global__ void kfetch2(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = tex2D<float>(t, in[2 * i], in[2 * i + 1]);
}
__global__ void kfetch3(cudaTextureObject_t t, const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = tex3D<float>(t, in[3 * i], in[3 * i + 1], in[3 * i + 2]);
}

// ---- host helpers -----------------------------------------------------------------
static const char* const kAddr[] = {"wrap", "clamp", "mirror", "border"};
struct Dev {
  float *in = nullptr, *out = nullptr; int* offs = nullptr; int n;
  explicit Dev(int count) : n(count) {
    cudaMalloc(&in, size_t(n) * 4 * 4); cudaMalloc(&out, size_t(n) * 16); cudaMalloc(&offs, size_t(n) * 8);
  }
  ~Dev() { cudaFree(in); cudaFree(out); cudaFree(offs); }
};
static void upload(Dev& d, const std::vector<float>& v) { cudaMemcpy(d.in, v.data(), v.size() * 4, cudaMemcpyHostToDevice); }
static unsigned long long hash_out(Dev& d, unsigned long long h, int floats_per = 4) {
  std::vector<float> o(size_t(d.n) * floats_per);
  cudaMemcpy(o.data(), d.out, o.size() * 4, cudaMemcpyDeviceToHost);
  return fnv(h, o.data(), o.size() * 4);
}
static cudaTextureObject_t make_tex(const cudaResourceDesc& r, int filt, int addr, int norm, int mip_linear = 0) {
  cudaTextureDesc d{};
  for (int i = 0; i < 3; ++i) d.addressMode[i] = static_cast<cudaTextureAddressMode>(addr);
  d.filterMode = filt ? cudaFilterModeLinear : cudaFilterModePoint;
  d.normalizedCoords = norm;
  d.mipmapFilterMode = mip_linear ? cudaFilterModeLinear : cudaFilterModePoint;
  d.maxMipmapLevelClamp = 20;
  d.readMode = cudaReadModeElementType;
  for (int i = 0; i < 4; ++i) d.borderColor[i] = 9.5f;
  cudaTextureObject_t t = 0;
  if (cudaCreateTextureObject(&t, &r, &d, nullptr) != cudaSuccess) { std::printf("cannot create a texture\n"); std::exit(2); }
  return t;
}
static cudaResourceDesc array_res(cudaArray_t a) {
  cudaResourceDesc r{}; r.resType = cudaResourceTypeArray; r.res.array.array = a; return r;
}
static std::vector<float> randoms(size_t n, float lo, float hi) {
  std::vector<float> v(n); for (auto& x : v) x = frand(lo, hi); return v;
}

template <class K, class... A> void go(int n, K k, A... args) { k<<<(n + 127) / 128, 128>>>(args...); }
#define LAUNCH(k, ...) go(d.n, k, __VA_ARGS__)

int main(int argc, char** argv) {
  g_print = argc > 1 && !std::strcmp(argv[1], "--print");
  const int n = 1024;
  Dev d(n);
  auto cd = cudaCreateChannelDesc<float>();

  // ---- offsets on a 2D texture -----------------------------------------------------
  {
    cudaArray_t a; cudaMallocArray(&a, &cd, 16, 8);
    auto tx = randoms(128, -100, 100); cudaMemcpy2DToArray(a, 0, 0, tx.data(), 64, 64, 8, cudaMemcpyHostToDevice);
    const cudaResourceDesc r = array_res(a);
    for (int norm = 0; norm < 2; ++norm)
      for (int filt = 0; filt < 2; ++filt)
        for (int addr = 0; addr < 4; ++addr) {
          if (!norm && (addr == 0 || addr == 2)) continue;   // wrap and mirror need normalized coordinates
          std::vector<float> in(2 * n);
          for (int i = 0; i < n; ++i) { in[2*i] = norm ? frand(-1.5f, 2.5f) : frand(-20, 36); in[2*i+1] = norm ? frand(-1.5f, 2.5f) : frand(-12, 20); }
          upload(d, in);
          cudaTextureObject_t t = make_tex(r, filt, addr, norm);
          unsigned long long h = kBasis;
          LAUNCH(k2d<1, 0>, t, d.in, d.out, n); h = hash_out(d, h);
          LAUNCH(k2d<0, 1>, t, d.in, d.out, n); h = hash_out(d, h);
          LAUNCH(k2d<-1, -1>, t, d.in, d.out, n); h = hash_out(d, h);
          LAUNCH(k2d<7, 7>, t, d.in, d.out, n); h = hash_out(d, h);
          LAUNCH(k2d<-8, -8>, t, d.in, d.out, n); h = hash_out(d, h);
          LAUNCH(k2d<3, -5>, t, d.in, d.out, n); h = hash_out(d, h);
          LAUNCH(k2d<-8, 7>, t, d.in, d.out, n); h = hash_out(d, h);
          report(std::string("tex.2d offsets n") + char('0' + norm) + " f" + char('0' + filt) + " " + kAddr[addr], h);
          cudaDestroyTextureObject(t);
        }
    // The destination predicate (always set) and the depth reference (no effect on a CUDA texture).
    std::vector<float> in(2 * n); for (int i = 0; i < n; ++i) { in[2*i] = frand(-20, 36); in[2*i+1] = frand(-12, 20); }
    upload(d, in);
    for (int filt = 0; filt < 2; ++filt) {
      cudaTextureObject_t t = make_tex(r, filt, 1, 0);
      LAUNCH(kpred, t, d.in, d.out, n);
      std::vector<float> o(4 * n); cudaMemcpy(o.data(), d.out, o.size() * 4, cudaMemcpyDeviceToHost);
      int set = 0; for (int i = 0; i < n; ++i) set += o[4 * i + 3] == 1.0f;
      const unsigned long long h = fnv(kBasis, o.data(), o.size() * 4);
      report(std::string("tex.2d dest predicate f") + char('0' + filt), h);
      if (!g_print && set != n) { std::printf("the predicate was set for %d of %d fetches\n", set, n); ++g_fails; }
      // With a reference the fetch is the plain one.
      LAUNCH(kplain, t, d.in, d.out, n); const unsigned long long plain = hash_out(d, kBasis);
      LAUNCH(kdepth, t, d.in, d.out, n, 0); const unsigned long long ref = hash_out(d, kBasis);
      if (!g_print && plain != ref) { std::printf("tex.2d with a depth reference differs from the plain fetch\n"); ++g_fails; }
      report(std::string("tex.2d depth reference f") + char('0' + filt), ref);
      cudaDestroyTextureObject(t);
    }
    // Register offsets: tex wraps them to four bits, tld4 to six.
    std::vector<int> offs(2 * n); const int cand[] = {-8, -9, 7, 8, 15, 16, 17, 31, 32, 33, -16, -17, -32, -33, 63, 64, 100, -100, 0, 3, -3};
    for (int i = 0; i < n; ++i) { offs[2*i] = cand[i % 21]; offs[2*i+1] = cand[(i / 21) % 21]; }
    cudaMemcpy(d.offs, offs.data(), n * 8, cudaMemcpyHostToDevice);
    for (int filt = 0; filt < 2; ++filt) {
      cudaTextureObject_t t = make_tex(r, filt, 1, 0);
      LAUNCH(kreg, t, d.in, d.offs, d.out, n, 0);
      report(std::string("tex.2d register offsets f") + char('0' + filt), hash_out(d, kBasis));
      cudaDestroyTextureObject(t);
    }
    cudaFreeArray(a);
  }
  // ---- 1D, 3D ---------------------------------------------------------------------------
  {
    cudaArray_t a; cudaMallocArray(&a, &cd, 32, 0);
    auto tx = randoms(32, -50, 50); cudaMemcpy2DToArray(a, 0, 0, tx.data(), 128, 128, 1, cudaMemcpyHostToDevice);
    const cudaResourceDesc r = array_res(a);
    for (int filt = 0; filt < 2; ++filt)
      for (int addr = 0; addr < 4; addr += 1) {
        std::vector<float> in(2 * n); for (int i = 0; i < n; ++i) { in[2*i] = frand(-1.5f, 2.5f); in[2*i+1] = 0; }
        upload(d, in);
        cudaTextureObject_t t = make_tex(r, filt, addr, 1);
        unsigned long long h = kBasis;
        LAUNCH(k1d<2>, t, d.in, d.out, n); h = hash_out(d, h);
        LAUNCH(k1d<-7>, t, d.in, d.out, n); h = hash_out(d, h);
        report(std::string("tex.1d offsets f") + char('0' + filt) + " " + kAddr[addr], h);
        cudaDestroyTextureObject(t);
      }
    cudaFreeArray(a);
    cudaArray_t a3; cudaMalloc3DArray(&a3, &cd, make_cudaExtent(8, 8, 8));
    auto t3 = randoms(512, -9, 9);
    cudaMemcpy3DParms q{}; q.srcPtr = make_cudaPitchedPtr(t3.data(), 32, 8, 8); q.dstArray = a3; q.extent = make_cudaExtent(8, 8, 8); q.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&q);
    const cudaResourceDesc r3 = array_res(a3);
    for (int norm = 0; norm < 2; ++norm)
      for (int filt = 0; filt < 2; ++filt)
        for (int addr = 0; addr < 4; ++addr) {
          if (!norm && (addr == 0 || addr == 2)) continue;
          std::vector<float> in(3 * n); for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) in[3*i+k] = norm ? frand(-1.2f, 2.2f) : frand(-10, 18);
          cudaMemcpy(d.in, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
          cudaTextureObject_t t = make_tex(r3, filt, addr, norm);
          unsigned long long h = kBasis;
          LAUNCH(k3d<2, -3, 5>, t, d.in, d.out, n); h = hash_out(d, h);
          LAUNCH(k3d<-8, 7, -1>, t, d.in, d.out, n); h = hash_out(d, h);
          report(std::string("tex.3d offsets n") + char('0' + norm) + " f" + char('0' + filt) + " " + kAddr[addr], h);
          cudaDestroyTextureObject(t);
        }
    cudaFreeArray(a3);
  }
  // ---- layered 2D: offsets, and tld4 --------------------------------------------------------
  {
    cudaArray_t al; cudaMalloc3DArray(&al, &cd, make_cudaExtent(8, 8, 4), cudaArrayLayered);
    auto t = randoms(256, -9, 9);
    cudaMemcpy3DParms q{}; q.srcPtr = make_cudaPitchedPtr(t.data(), 32, 8, 8); q.dstArray = al; q.extent = make_cudaExtent(8, 8, 4); q.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&q);
    const cudaResourceDesc r = array_res(al);
    for (int filt = 0; filt < 2; ++filt)
      for (int addr = 1; addr < 4; addr += 2) {
        std::vector<float> in(3 * n);
        for (int i = 0; i < n; ++i) { in[3*i] = float(int(frand(0, 5))); in[3*i+1] = frand(-3, 11); in[3*i+2] = frand(-3, 11); }
        cudaMemcpy(d.in, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
        cudaTextureObject_t tx = make_tex(r, filt, addr, 0);
        LAUNCH(ka2d<3, -2>, tx, d.in, d.out, n);
        report(std::string("tex.a2d offsets f") + char('0' + filt) + " " + kAddr[addr], hash_out(d, kBasis));
        LAUNCH(kga2d, tx, d.in, d.out, n);
        report(std::string("tld4.a2d f") + char('0' + filt) + " " + kAddr[addr], hash_out(d, kBasis));
        cudaDestroyTextureObject(tx);
      }
    cudaFreeArray(al);
  }
  // ---- tld4 with offsets, register offsets and a depth reference ----------------------------
  {
    cudaArray_t ag; cudaMallocArray(&ag, &cd, 16, 8, cudaArrayTextureGather);
    auto t = randoms(128, -9, 9); cudaMemcpy2DToArray(ag, 0, 0, t.data(), 64, 64, 8, cudaMemcpyHostToDevice);
    const cudaResourceDesc r = array_res(ag);
    for (int norm = 0; norm < 2; ++norm)
      for (int addr = 0; addr < 4; ++addr) {
        if (!norm && (addr == 0 || addr == 2)) continue;
        std::vector<float> in(2 * n);
        for (int i = 0; i < n; ++i) { in[2*i] = norm ? frand(-1.5f, 2.5f) : frand(-5, 22); in[2*i+1] = norm ? frand(-1.5f, 2.5f) : frand(-5, 14); }
        upload(d, in);
        cudaTextureObject_t tx = make_tex(r, 1, addr, norm);
        unsigned long long h = kBasis;
        LAUNCH(kgather<1, -2>, tx, d.in, d.out, n); h = hash_out(d, h);
        LAUNCH(kgather<-8, 7>, tx, d.in, d.out, n); h = hash_out(d, h);
        report(std::string("tld4 offsets n") + char('0' + norm) + " " + kAddr[addr], h);
        cudaDestroyTextureObject(tx);
      }
    std::vector<float> in(2 * n); for (int i = 0; i < n; ++i) { in[2*i] = frand(-2, 18); in[2*i+1] = frand(-2, 10); }
    upload(d, in);
    std::vector<int> offs(2 * n); const int cand[] = {-8, -9, 7, 8, 15, 16, 17, 31, 32, 33, -16, -17, -32, -33, 63, 64, 100, -100, 0, 3, -3};
    for (int i = 0; i < n; ++i) { offs[2*i] = cand[i % 21]; offs[2*i+1] = cand[(i / 21) % 21]; }
    cudaMemcpy(d.offs, offs.data(), n * 8, cudaMemcpyHostToDevice);
    cudaTextureObject_t tx = make_tex(r, 1, 1, 0);
    LAUNCH(kreg, tx, d.in, d.offs, d.out, n, 1);
    report("tld4 register offsets", hash_out(d, kBasis));
    LAUNCH(kplaing, tx, d.in, d.out, n); const unsigned long long plain = hash_out(d, kBasis);
    LAUNCH(kdepth, tx, d.in, d.out, n, 1); const unsigned long long ref = hash_out(d, kBasis);
    if (!g_print && plain != ref) { std::printf("tld4 with a depth reference differs from the plain gather\n"); ++g_fails; }
    report("tld4 depth reference", ref);
    cudaDestroyTextureObject(tx);
    cudaFreeArray(ag);
  }
  // ---- tld4 on cubemaps and cubemap arrays --------------------------------------------------
  {
    cudaArray_t ac; cudaMalloc3DArray(&ac, &cd, make_cudaExtent(8, 8, 6), cudaArrayCubemap);
    auto t = randoms(8 * 8 * 6, -9, 9);
    cudaMemcpy3DParms q{}; q.srcPtr = make_cudaPitchedPtr(t.data(), 32, 8, 8); q.dstArray = ac; q.extent = make_cudaExtent(8, 8, 6); q.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&q);
    std::vector<float> in(3 * n); for (auto& v : in) v = frand(-1, 1);
    cudaMemcpy(d.in, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
    const cudaResourceDesc r = array_res(ac);
    for (int filt = 0; filt < 2; ++filt)
      for (int addr = 0; addr < 4; ++addr) {
        cudaTextureObject_t tx = make_tex(r, filt, addr, 1);
        LAUNCH(kgcube, tx, d.in, d.out, n);
        report(std::string("tld4.cube f") + char('0' + filt) + " " + kAddr[addr], hash_out(d, kBasis));
        cudaDestroyTextureObject(tx);
      }
    cudaFreeArray(ac);
    cudaArray_t aa; cudaMalloc3DArray(&aa, &cd, make_cudaExtent(8, 8, 18), cudaArrayCubemap | cudaArrayLayered);
    auto t2 = randoms(8 * 8 * 18, -9, 9);
    cudaMemcpy3DParms q2{}; q2.srcPtr = make_cudaPitchedPtr(t2.data(), 32, 8, 8); q2.dstArray = aa; q2.extent = make_cudaExtent(8, 8, 18); q2.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&q2);
    std::vector<float> in4(4 * n); for (int i = 0; i < n; ++i) { in4[4*i] = float(int(frand(0, 4))); for (int k = 1; k < 4; ++k) in4[4*i+k] = frand(-1, 1); }
    cudaMemcpy(d.in, in4.data(), in4.size() * 4, cudaMemcpyHostToDevice);
    const cudaResourceDesc r2 = array_res(aa);
    for (int filt = 0; filt < 2; ++filt)
      for (int addr = 1; addr < 4; addr += 2) {
        cudaTextureObject_t tx = make_tex(r2, filt, addr, 1);
        LAUNCH(kgacube, tx, d.in, d.out, n);
        report(std::string("tld4.acube f") + char('0' + filt) + " " + kAddr[addr], hash_out(d, kBasis));
        cudaDestroyTextureObject(tx);
      }
    cudaFreeArray(aa);
  }
  // ---- offsets with an explicit level of detail -----------------------------------------------
  {
    cudaMipmappedArray_t mm; cudaMallocMipmappedArray(&mm, &cd, make_cudaExtent(16, 8, 0), 5);
    for (int l = 0; l < 5; ++l) {
      int w = 16 >> l, h = 8 >> l; if (h < 1) h = 1;
      auto t = randoms(size_t(w) * h, -9, 9); cudaArray_t lv; cudaGetMipmappedArrayLevel(&lv, mm, l);
      cudaMemcpy2DToArray(lv, 0, 0, t.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice);
    }
    cudaResourceDesc r{}; r.resType = cudaResourceTypeMipmappedArray; r.res.mipmap.mipmap = mm;
    std::vector<float> in(3 * n); for (int i = 0; i < n; ++i) { in[3*i] = frand(-0.3f, 1.3f); in[3*i+1] = frand(-0.3f, 1.3f); in[3*i+2] = frand(0, 4); }
    cudaMemcpy(d.in, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
    for (int filt = 0; filt < 2; ++filt)
      for (int mip = 0; mip < 2; ++mip)
        for (int addr = 0; addr < 4; ++addr) {
          cudaTextureObject_t tx = make_tex(r, filt, addr, 1, mip);
          LAUNCH(klevel<2, -3>, tx, d.in, d.out, n);
          report(std::string("tex.level offsets f") + char('0' + filt) + " m" + char('0' + mip) + " " + kAddr[addr], hash_out(d, kBasis));
          cudaDestroyTextureObject(tx);
        }
    cudaFreeMipmappedArray(mm);
  }
  // ---- half-precision results -----------------------------------------------------------------
  {
    std::vector<float> in(2 * n); for (int i = 0; i < n; ++i) { in[2*i] = frand(0, 8); in[2*i+1] = frand(0, 8); }
    upload(d, in);
    unsigned short* h4; unsigned* h2; cudaMalloc(&h4, n * 8); cudaMalloc(&h2, n * 8);
    const char* const kind[] = {"f32x4", "f16x4", "u8x4n"};
    for (int k = 0; k < 3; ++k)
      for (int filt = 0; filt < 2; ++filt) {
        cudaChannelFormatDesc c = k == 0 ? cudaCreateChannelDesc<float4>() : k == 1 ? cudaCreateChannelDescHalf4() : cudaCreateChannelDesc<uchar4>();
        cudaArray_t a; cudaMallocArray(&a, &c, 8, 8);
        if (k == 0) { auto t = randoms(8 * 8 * 4, -3, 3); cudaMemcpy2DToArray(a, 0, 0, t.data(), 8 * 16, 8 * 16, 8, cudaMemcpyHostToDevice); }
        else if (k == 1) {
          std::vector<unsigned short> t(8 * 8 * 4);
          for (auto& v : t) { float f = frand(-3, 3); if (f > -0.1f && f < 0.1f) f += 0.5f; uint32_t b; std::memcpy(&b, &f, 4); const uint32_t s = b >> 31, e = (b >> 23) & 0xff, m = b & 0x7fffff;
            v = static_cast<unsigned short>((s << 15) | ((e - 112) << 10) | (m >> 13)); }   // normal values only: |f| in [2^-14, 2^15)
          cudaMemcpy2DToArray(a, 0, 0, t.data(), 8 * 8, 8 * 8, 8, cudaMemcpyHostToDevice);
        } else {
          std::vector<uint8_t> t(8 * 8 * 4); for (auto& v : t) v = static_cast<uint8_t>(frand(0, 256));
          cudaMemcpy2DToArray(a, 0, 0, t.data(), 8 * 4, 8 * 4, 8, cudaMemcpyHostToDevice);
        }
        cudaResourceDesc r = array_res(a);
        cudaTextureDesc td{}; for (int i = 0; i < 3; ++i) td.addressMode[i] = cudaAddressModeClamp;
        td.filterMode = filt ? cudaFilterModeLinear : cudaFilterModePoint;
        td.readMode = k == 2 ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
        cudaTextureObject_t t; cudaCreateTextureObject(&t, &r, &td, nullptr);
        std::vector<unsigned short> o4(4 * n); std::vector<unsigned> o2(2 * n);
        kh4<<<n / 128, 128>>>(t, d.in, h4, n); cudaMemcpy(o4.data(), h4, n * 8, cudaMemcpyDeviceToHost);
        report(std::string("tex.v4.f16 ") + kind[k] + " f" + char('0' + filt), fnv(kBasis, o4.data(), o4.size() * 2));
        kh2<<<n / 128, 128>>>(t, d.in, h2, n); cudaMemcpy(o2.data(), h2, n * 8, cudaMemcpyDeviceToHost);
        report(std::string("tex.v2.f16x2 ") + kind[k] + " f" + char('0' + filt), fnv(kBasis, o2.data(), o2.size() * 4));
        cudaDestroyTextureObject(t);
        cudaFreeArray(a);
      }
    cudaFree(h4); cudaFree(h2);
  }
  // ---- linear filtering of signed 8-bit normalized texels -----------------------------------------
  {
    cudaChannelFormatDesc c = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSigned);
    // 1D: every pair of neighbours at every weight (the sum of each pair is what the result depends on).
    const int W = 256;
    cudaArray_t a; cudaMallocArray(&a, &c, W, 0);
    std::vector<int8_t> t(W); for (int rep = 0; rep < 2; ++rep) {
      for (int i = 0; i < W; ++i) t[i] = rep == 0 ? static_cast<int8_t>(i - 128) : static_cast<int8_t>(frand(0, 256));
      cudaMemcpy2DToArray(a, 0, 0, t.data(), W, W, 1, cudaMemcpyHostToDevice);
      cudaResourceDesc r = array_res(a);
      cudaTextureDesc td{}; td.addressMode[0] = cudaAddressModeClamp; td.filterMode = cudaFilterModeLinear; td.readMode = cudaReadModeNormalizedFloat;
      cudaTextureObject_t tx; cudaCreateTextureObject(&tx, &r, &td, nullptr);
      const int m = (W - 1) * 256;
      std::vector<float> xs(m); for (int i = 0; i < W - 1; ++i) for (int f = 0; f < 256; ++f) xs[i * 256 + f] = i + 0.5f + f / 256.0f;
      float *dx, *dout; cudaMalloc(&dx, m * 4); cudaMalloc(&dout, m * 4); cudaMemcpy(dx, xs.data(), m * 4, cudaMemcpyHostToDevice);
      kfetch1<<<(m + 127) / 128, 128>>>(tx, dx, dout, m);
      std::vector<float> o(m); cudaMemcpy(o.data(), dout, m * 4, cudaMemcpyDeviceToHost);
      report(rep == 0 ? "snorm8 1d sweep" : "snorm8 1d random", fnv(kBasis, o.data(), m * 4));
      cudaFree(dx); cudaFree(dout); cudaDestroyTextureObject(tx);
    }
    cudaFreeArray(a);
    // 2D and 3D, random texels and coordinates.
    cudaArray_t a2; cudaMallocArray(&a2, &c, 16, 16);
    std::vector<int8_t> t2(256); for (auto& v : t2) v = static_cast<int8_t>(frand(0, 256));
    cudaMemcpy2DToArray(a2, 0, 0, t2.data(), 16, 16, 16, cudaMemcpyHostToDevice);
    cudaArray_t a3; cudaMalloc3DArray(&a3, &c, make_cudaExtent(8, 8, 8));
    std::vector<int8_t> t3(512); for (auto& v : t3) v = static_cast<int8_t>(frand(0, 256));
    cudaMemcpy3DParms q{}; q.srcPtr = make_cudaPitchedPtr(t3.data(), 8, 8, 8); q.dstArray = a3; q.extent = make_cudaExtent(8, 8, 8); q.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&q);
    for (int dim = 2; dim <= 3; ++dim) {
      cudaResourceDesc r = array_res(dim == 2 ? a2 : a3);
      cudaTextureDesc td{}; for (int i = 0; i < 3; ++i) td.addressMode[i] = cudaAddressModeClamp; td.filterMode = cudaFilterModeLinear; td.readMode = cudaReadModeNormalizedFloat;
      cudaTextureObject_t tx; cudaCreateTextureObject(&tx, &r, &td, nullptr);
      std::vector<float> in(dim * n); for (auto& v : in) v = frand(-1, (dim == 2 ? 17 : 9));
      cudaMemcpy(d.in, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
      if (dim == 2) kfetch2<<<n / 128, 128>>>(tx, d.in, d.out, n); else kfetch3<<<n / 128, 128>>>(tx, d.in, d.out, n);
      std::vector<float> o(n); cudaMemcpy(o.data(), d.out, n * 4, cudaMemcpyDeviceToHost);
      report(dim == 2 ? "snorm8 2d random" : "snorm8 3d random", fnv(kBasis, o.data(), n * 4));
      cudaDestroyTextureObject(tx);
    }
    cudaFreeArray(a2); cudaFreeArray(a3);
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(e)); ++g_fails; }
  if (g_print) return 0;
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
