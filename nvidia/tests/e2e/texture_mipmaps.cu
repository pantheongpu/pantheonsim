// Mipmapped textures, checked bit for bit against an RTX 3060 (sm_86):
// explicit level of detail (tex2DLod, tex1DLod, tex3DLod) and plain fetches
// of mipmapped textures, point and linear filtering within a level, point
// and linear filtering between levels, with level biases and clamps. Each
// case's results are hashed and compared with the hashes the hardware
// produced; prints PASS on the last line, and runs the same on a GPU.
// (tex2DGrad is left out: the simulator refuses it, see the parser.)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Expected { const char* tag; uint64_t hash; };
static const Expected kExpected[] = {
    {"2LPP", 0x03dc0dae6fd8f537ull},
    {"2BPP", 0x1c28abaae27f50eeull},
    {"2LPL", 0xaf242d4f60218947ull},
    {"2BPL", 0x1c28abaae27f50eeull},
    {"2LLP", 0xd7176b02b7292b1aull},
    {"2BLP", 0x4a0b79ff4166cd32ull},
    {"2LLL", 0x2c2f1edb48cf061aull},
    {"2BLL", 0x4a0b79ff4166cd32ull},
    {"2LLLb", 0xcec671de4178a43full},
    {"2BLLb", 0x4a0b79ff4166cd32ull},
    {"2LLLb2", 0xb268c469a43013a2ull},
    {"2BLLb2", 0x4a0b79ff4166cd32ull},
    {"2LLLc", 0xa87e44d55146bf4cull},
    {"2BLLc", 0xc4c7b96ba331d0d7ull},
    {"2LPLc", 0x2542ee6892b8fc15ull},
    {"2BPLc", 0xddb9a6864d1471a1ull},
    {"1D00", 0x5db782a7a0bf400full},
    {"3D00", 0x49442659a535488cull},
    {"1D01", 0xffeddc6939bda35cull},
    {"3D01", 0x86b912c5973e6983ull},
    {"1D10", 0xe217883c7b63a563ull},
    {"3D10", 0x3cccb24e3ab7002eull},
    {"1D11", 0xeae6d5798ee7607cull},
    {"3D11", 0xd62efc3ec1b87ec5ull},
};
static int fails = 0;
static void check(const char* tag, const std::vector<float>& o) {
  uint64_t h = 1469598103934665603ull;   // FNV-1a over every result's bits
  for (float f : o) { uint32_t b; memcpy(&b, &f, 4); for (int k = 0; k < 4; ++k) { h ^= (b >> (8 * k)) & 0xff; h *= 1099511628211ull; } }
  uint64_t want = 0;
  for (const auto& e : kExpected) if (!strcmp(e.tag, tag)) want = e.hash;
  printf("%-8s %016llx %s\n", tag, (unsigned long long)h, h == want ? "ok" : "MISMATCH");
  fails += h != want;
}
static uint32_t rng = 4242;
static uint32_t next() { rng = rng * 1664525u + 1013904223u; return rng; }
static float frand(float lo, float hi) { return lo + (hi - lo) * (next() >> 8) / float(1 << 24); }
__global__ void k2lod(cudaTextureObject_t t, const float* p, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = tex2DLod<float>(t, p[3*i], p[3*i+1], p[3*i+2]); }
__global__ void k2(cudaTextureObject_t t, const float* p, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = tex2D<float>(t, p[3*i], p[3*i+1]); }
__global__ void k1lod(cudaTextureObject_t t, const float* p, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = tex1DLod<float>(t, p[4*i], p[4*i+3]); }
__global__ void k3lod(cudaTextureObject_t t, const float* p, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = tex3DLod<float>(t, p[4*i], p[4*i+1], p[4*i+2], p[4*i+3]); }
static cudaTextureObject_t make(cudaMipmappedArray_t mm, int minf, int mipf, float bias, float mn, float mx) {
  cudaResourceDesc r{}; r.resType = cudaResourceTypeMipmappedArray; r.res.mipmap.mipmap = mm;
  cudaTextureDesc d{}; for (int i = 0; i < 3; ++i) d.addressMode[i] = cudaAddressModeClamp; d.normalizedCoords = 1;
  d.filterMode = minf ? cudaFilterModeLinear : cudaFilterModePoint; d.mipmapFilterMode = mipf ? cudaFilterModeLinear : cudaFilterModePoint;
  d.mipmapLevelBias = bias; d.minMipmapLevelClamp = mn; d.maxMipmapLevelClamp = mx; d.maxAnisotropy = 1;
  cudaTextureObject_t t; cudaCreateTextureObject(&t, &r, &d, nullptr); return t;
}
int main() {
  auto c = cudaCreateChannelDesc<float>();
  // 2D: 16x8, five levels
  { const int W = 16, H = 8, L = 5;
    cudaMipmappedArray_t mm; cudaMallocMipmappedArray(&mm, &c, make_cudaExtent(W, H, 0), L);
    for (int l = 0; l < L; ++l) { int w = W >> l, h = H >> l; if (w < 1) w = 1; if (h < 1) h = 1;
      std::vector<float> t(w * h); for (auto& v : t) v = frand(-100, 100);
      cudaArray_t lv; cudaGetMipmappedArrayLevel(&lv, mm, l); cudaMemcpy2DToArray(lv, 0, 0, t.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice); }
    const int n = 2048; std::vector<float> p(3 * n), o(n);
    for (int i = 0; i < n; ++i) { p[3*i] = frand(-0.2f, 1.2f); p[3*i+1] = frand(-0.2f, 1.2f); p[3*i+2] = frand(-1.5f, 6.5f); }
    float *dp, *dout; cudaMalloc(&dp, p.size() * 4); cudaMalloc(&dout, n * 4); cudaMemcpy(dp, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
    struct Cfg { const char* tag; int minf, mipf; float bias, mn, mx; };
    Cfg cfgs[] = {{"PP", 0, 0, 0, 0, 100}, {"PL", 0, 1, 0, 0, 100}, {"LP", 1, 0, 0, 0, 100}, {"LL", 1, 1, 0, 0, 100},
                  {"LLb", 1, 1, -0.3f, 0, 100}, {"LLb2", 1, 1, 0.61f, 0, 100}, {"LLc", 1, 1, 0.17f, 0.7f, 2.3f}, {"PLc", 0, 1, -1.1f, 0.33f, 3.1f}};
    for (auto& cf : cfgs) {
      auto tx = make(mm, cf.minf, cf.mipf, cf.bias, cf.mn, cf.mx);
      k2lod<<<n / 128, 128>>>(tx, dp, dout, n); cudaMemcpy(o.data(), dout, n * 4, cudaMemcpyDeviceToHost);
      char tag[16]; snprintf(tag, 16, "2L%s", cf.tag); check(tag, o);
      k2<<<n / 128, 128>>>(tx, dp, dout, n); cudaMemcpy(o.data(), dout, n * 4, cudaMemcpyDeviceToHost);
      snprintf(tag, 16, "2B%s", cf.tag); check(tag, o);
    } }
  // 1D (16, five levels) and 3D (8x4x8, four levels)
  { cudaMipmappedArray_t m1; cudaMallocMipmappedArray(&m1, &c, make_cudaExtent(16, 0, 0), 5);
    for (int l = 0; l < 5; ++l) { int w = 16 >> l; std::vector<float> t(w); for (auto& v : t) v = frand(-50, 50);
      cudaArray_t a; cudaGetMipmappedArrayLevel(&a, m1, l); cudaMemcpy2DToArray(a, 0, 0, t.data(), w * 4, w * 4, 1, cudaMemcpyHostToDevice); }
    cudaMipmappedArray_t m3; cudaMallocMipmappedArray(&m3, &c, make_cudaExtent(8, 4, 8), 4);
    for (int l = 0; l < 4; ++l) { int w = 8 >> l, h = 4 >> l, dd = 8 >> l; if (h < 1) h = 1; std::vector<float> t(w * h * dd); for (auto& v : t) v = frand(-50, 50);
      cudaArray_t a; cudaGetMipmappedArrayLevel(&a, m3, l);
      cudaMemcpy3DParms q{}; q.srcPtr = make_cudaPitchedPtr(t.data(), w * 4, w, h); q.dstArray = a; q.extent = make_cudaExtent(w, h, dd); q.kind = cudaMemcpyHostToDevice; cudaMemcpy3D(&q); }
    const int n = 1024; std::vector<float> p(4 * n), o(n);
    for (int i = 0; i < n; ++i) { for (int k = 0; k < 3; ++k) p[4*i+k] = frand(-0.2f, 1.2f); p[4*i+3] = frand(-1, 5); }
    float *dp, *dout; cudaMalloc(&dp, p.size() * 4); cudaMalloc(&dout, n * 4); cudaMemcpy(dp, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
    for (int minf = 0; minf < 2; ++minf) for (int mipf = 0; mipf < 2; ++mipf) for (int dim : {1, 3}) {
      auto tx = make(dim == 1 ? m1 : m3, minf, mipf, dim == 1 ? 0.3f : -0.45f, 0, 100);
      if (dim == 1) k1lod<<<n/128,128>>>(tx, dp, dout, n); else k3lod<<<n/128,128>>>(tx, dp, dout, n);
      cudaMemcpy(o.data(), dout, n * 4, cudaMemcpyDeviceToHost);
      char tag[16]; snprintf(tag, 16, "%dD%d%d", dim, minf, mipf); check(tag, o);
    } }
  const cudaError_t e = cudaGetLastError();
  printf("%s\n", e == cudaSuccess && fails == 0 ? "PASS" : "FAIL");
  return e == cudaSuccess && fails == 0 ? 0 : 1;
}
