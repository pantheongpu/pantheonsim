#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Texture gather (tex2Dgather, PTX tld4), checked bit for bit against an RTX
// 3060 (sm_86): the four texels of the bilinear footprint, counter-clockwise
// from the lower left, for every component of a float4 texture under clamp,
// border and wrap addressing, and for 8-bit normalized and 32-bit integer
// textures. Each case's results are hashed and compared with the hashes the
// hardware produced; also checks that gather is refused on a layered array,
// as the CUDA runtime refuses it. Prints PASS on the last line.
struct Expected { const char* tag; uint64_t hash; };
static const Expected kExpected[] = {
    {"F00", 0xf60821ec485f2bfaull},
    {"F01", 0x124a1cf37a6424bcull},
    {"F02", 0x2e6335dface06df1ull},
    {"F03", 0x1542614e22cb18b0ull},
    {"F10", 0x6b5fcfe80c4f685eull},
    {"F11", 0xce02553133b05a98ull},
    {"F12", 0xd06564b173a0577aull},
    {"F13", 0x24a880afe4091731ull},
    {"F20", 0x07e423a8e6eaf42eull},
    {"F21", 0xf1e8daaddef1d9daull},
    {"F22", 0x576216e9d406d5a4ull},
    {"F23", 0x938d224f70d420efull},
    {"U", 0xca22463b00a12feeull},
    {"I", 0xc3cbb4ca27cad16aull},
};
static int fails = 0;
static uint32_t rng = 555;
static uint32_t next() { rng = rng * 1664525u + 1013904223u; return rng; }
static float frand(float lo, float hi) { return lo + (hi - lo) * (next() >> 8) / float(1 << 24); }
template <class T, class R> __global__ void kg(cudaTextureObject_t t, const float* p, R* o, int n, int comp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  switch (comp) { case 0: o[i] = tex2Dgather<R>(t, p[2*i], p[2*i+1], 0); break; case 1: o[i] = tex2Dgather<R>(t, p[2*i], p[2*i+1], 1); break;
                  case 2: o[i] = tex2Dgather<R>(t, p[2*i], p[2*i+1], 2); break; default: o[i] = tex2Dgather<R>(t, p[2*i], p[2*i+1], 3); } }
cudaTextureObject_t mk(cudaArray_t a, cudaTextureAddressMode m, bool norm, bool nf) {
  cudaResourceDesc r{}; r.resType = cudaResourceTypeArray; r.res.array.array = a;
  cudaTextureDesc d{}; for (int i = 0; i < 3; ++i) d.addressMode[i] = m; d.normalizedCoords = norm;
  d.readMode = nf ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
  cudaTextureObject_t t; cudaError_t e = cudaCreateTextureObject(&t, &r, &d, nullptr); if (e) printf("create %s\n", cudaGetErrorString(e)); return t;
}
int main() {
  const int n = 512; std::vector<float> p(2 * n); float* dp; cudaMalloc(&dp, p.size() * 4); float4* dout; cudaMalloc(&dout, n * 16);
  std::vector<float4> o(n);
  auto dump = [&](const char* tag) { cudaMemcpy(o.data(), dout, n * 16, cudaMemcpyDeviceToHost);
    uint64_t h = 1469598103934665603ull;   // FNV-1a over every result's bits
    for (auto& v : o) { uint32_t b[4]; memcpy(b, &v, 16); for (uint32_t x : b) for (int k = 0; k < 4; ++k) { h ^= (x >> (8 * k)) & 0xff; h *= 1099511628211ull; } }
    uint64_t want = 0;
    for (const auto& e : kExpected) if (!strcmp(e.tag, tag)) want = e.hash;
    printf("%-4s %016llx %s\n", tag, (unsigned long long)h, h == want ? "ok" : "MISMATCH");
    fails += h != want; };
  // float4 5x4
  { std::vector<float> t(5 * 4 * 4); for (auto& v : t) v = frand(-9, 9);
    auto c = cudaCreateChannelDesc<float4>(); cudaArray_t a; cudaMallocArray(&a, &c, 5, 4, cudaArrayTextureGather);
    cudaMemcpy2DToArray(a, 0, 0, t.data(), 80, 80, 4, cudaMemcpyHostToDevice);
    const cudaTextureAddressMode modes[3] = {cudaAddressModeClamp, cudaAddressModeBorder, cudaAddressModeWrap};
    for (int m = 0; m < 3; ++m) {
      auto tx = mk(a, modes[m], m == 2, false);
      for (int i = 0; i < n; ++i) { if (m == 2) { p[2*i] = frand(-1.3f, 2.3f); p[2*i+1] = frand(-1.3f, 2.3f); } else { p[2*i] = frand(-1, 6); p[2*i+1] = frand(-1, 5); } }
      cudaMemcpy(dp, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
      for (int comp = 0; comp < 4; ++comp) { kg<float4, float4><<<n/128,128>>>(tx, dp, dout, n, comp); char tag[8]; snprintf(tag, 8, "F%d%d", m, comp); dump(tag); } } }
  // uchar4 normalized, 4x4
  { std::vector<uint8_t> t(64); for (auto& v : t) v = next() >> 24;
    auto c = cudaCreateChannelDesc<uchar4>(); cudaArray_t a; cudaMallocArray(&a, &c, 4, 4, cudaArrayTextureGather);
    cudaMemcpy2DToArray(a, 0, 0, t.data(), 16, 16, 4, cudaMemcpyHostToDevice);
    auto tx = mk(a, cudaAddressModeClamp, false, true);
    for (int i = 0; i < n; ++i) { p[2*i] = frand(-1, 5); p[2*i+1] = frand(-1, 5); }
    cudaMemcpy(dp, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
    kg<uchar4, float4><<<n/128,128>>>(tx, dp, dout, n, 2); dump("U"); }
  // int2 element type, 4x4
  { std::vector<int> t(32); for (auto& v : t) v = int(next());
    auto c = cudaCreateChannelDesc<int2>(); cudaArray_t a; cudaMallocArray(&a, &c, 4, 4, cudaArrayTextureGather);
    cudaMemcpy2DToArray(a, 0, 0, t.data(), 32, 32, 4, cudaMemcpyHostToDevice);
    auto tx = mk(a, cudaAddressModeClamp, false, false);
    for (int i = 0; i < n; ++i) { p[2*i] = frand(-1, 5); p[2*i+1] = frand(-1, 5); }
    cudaMemcpy(dp, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
    kg<int2, int4><<<n/128,128>>>(tx, dp, (int4*)dout, n, 1); dump("I"); }
  // Gather is for 2D arrays: a layered one with the gather flag is refused.
  { auto c = cudaCreateChannelDesc<float>(); cudaArray_t a;
    const cudaError_t e = cudaMalloc3DArray(&a, &c, make_cudaExtent(4, 4, 2), cudaArrayLayered | cudaArrayTextureGather);
    printf("layered gather array refused: %s\n", e == cudaErrorInvalidValue ? "yes" : "no");
    fails += e != cudaErrorInvalidValue;
    cudaGetLastError(); }   // the refusal is the expected error, not a failure
  const cudaError_t e = cudaGetLastError();
  printf("%s\n", e == cudaSuccess && fails == 0 ? "PASS" : "FAIL");
  return e == cudaSuccess && fails == 0 ? 0 : 1;
}
