// Layered and cubemap textures, and layered surfaces, checked bit for bit
// against an RTX 3060 (sm_86): 1D and 2D layered textures (point and linear,
// clamp and border, layer indices past both ends), cubemaps with directions
// chosen to tie between axes, layered cubemaps, and a layered surface written
// and read back. Each section's results are hashed and compared with the
// hashes the hardware produced; prints PASS on the last line, and runs the
// same on a GPU.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Expected { const char* tag; uint64_t hash; };
static const Expected kExpected[] = {
    {"L100", 0x9d25a685d5ba6e26ull},
    {"L101", 0x97ead67532398996ull},
    {"L110", 0x321ec3a1d078f9adull},
    {"L111", 0x1a22df3e46a7ee7bull},
    {"L20", 0x478b8109ff920d5eull},
    {"L21", 0x3bbb4391c8add2d5ull},
    {"C0", 0xcd8b0716e8481624ull},
    {"C1", 0x9a18a2d3166c8129ull},
    {"CL0", 0xcde213f6cfc9f1feull},
    {"CL1", 0x423d76c0f35bfcedull},
    {"CM00", 0xfab37f572a231217ull},
    {"CM01", 0xfab37f572a231217ull},
    {"CM02", 0xfab37f572a231217ull},
    {"CM10", 0x01b9ddd0a57c6d05ull},
    {"CM11", 0xaeef224a376ff710ull},
    {"CM12", 0x8f69e20e868de5ddull},
    {"S", 0x9b367a07c2d92ecaull},
};
static int fails = 0;
static uint32_t rng = 777;
static uint32_t next() { rng = rng * 1664525u + 1013904223u; return rng; }
static float frand(float lo, float hi) { return lo + (hi - lo) * (next() >> 8) / float(1 << 24); }
__global__ void k1l(cudaTextureObject_t t, const float* xs, const int* ls, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = tex1DLayered<float>(t, xs[i], ls[i]); }
__global__ void k2l(cudaTextureObject_t t, const float* xs, const int* ls, float4* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = tex2DLayered<float4>(t, xs[2*i], xs[2*i+1], ls[i]); }
__global__ void kc(cudaTextureObject_t t, const float* xs, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = texCubemap<float>(t, xs[3*i], xs[3*i+1], xs[3*i+2]); }
__global__ void kcl(cudaTextureObject_t t, const float* xs, const int* ls, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = texCubemapLayered<float>(t, xs[3*i], xs[3*i+1], xs[3*i+2], ls[i]); }
__global__ void ks(cudaSurfaceObject_t s, int w, int h, int layers, float* o) {
  // write then read back through the layered surface
  for (int l = 0; l < layers; ++l) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
    surf2DLayeredwrite(float(1000 * l + 10 * y + x), s, x * 4, y, l);
  __syncthreads();
  int k = 0;
  for (int l = 0; l < layers; ++l) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
    o[k++] = surf2DLayeredread<float>(s, x * 4, y, l);
}
cudaTextureObject_t mk(cudaArray_t a, cudaTextureAddressMode m, bool lin, bool norm) {
  cudaResourceDesc r{}; r.resType = cudaResourceTypeArray; r.res.array.array = a;
  cudaTextureDesc d{}; for (int i = 0; i < 3; ++i) d.addressMode[i] = m;
  d.filterMode = lin ? cudaFilterModeLinear : cudaFilterModePoint; d.normalizedCoords = norm;
  cudaTextureObject_t t; cudaError_t e = cudaCreateTextureObject(&t, &r, &d, nullptr);
  if (e) printf("create: %s\n", cudaGetErrorString(e)); return t;
}
float *dx, *dout; int* dl; std::vector<float> hx(3 * 4096), ho(4 * 4096); std::vector<int> hl(4096);
void dump(const char* tag, int n, int per) {
  cudaMemcpy(ho.data(), dout, n * per * 4, cudaMemcpyDeviceToHost);
  uint64_t h = 1469598103934665603ull;   // FNV-1a over every result's bits
  for (int i = 0; i < n * per; ++i) {
    uint32_t b; memcpy(&b, &ho[i], 4);
    for (int k = 0; k < 4; ++k) { h ^= (b >> (8 * k)) & 0xff; h *= 1099511628211ull; }
  }
  uint64_t want = 0;
  for (const auto& e : kExpected) if (!strcmp(e.tag, tag)) want = e.hash;
  printf("%-4s %016llx %s\n", tag, (unsigned long long)h, h == want ? "ok" : "MISMATCH");
  fails += h != want;
}
void up(int nx, int nl) { cudaMemcpy(dx, hx.data(), nx * 4, cudaMemcpyHostToDevice); cudaMemcpy(dl, hl.data(), nl * 4, cudaMemcpyHostToDevice); }
int main() {
  cudaMalloc(&dx, 3 * 4096 * 4); cudaMalloc(&dout, 4 * 4096 * 4); cudaMalloc(&dl, 4096 * 4);
  const int n = 1024;
  // 1D layered float, width 5, 3 layers
  { std::vector<float> t(5 * 3); for (auto& v : t) v = frand(-50, 50);
    auto c = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMalloc3DArray(&a, &c, make_cudaExtent(5, 0, 3), cudaArrayLayered);
    cudaMemcpy3DParms p{}; p.srcPtr = make_cudaPitchedPtr(t.data(), 20, 5, 1); p.dstArray = a; p.extent = make_cudaExtent(5, 1, 3); p.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p);
    for (int lin = 0; lin < 2; ++lin) for (int m = 0; m < 2; ++m) {
      auto tx = mk(a, m ? cudaAddressModeBorder : cudaAddressModeClamp, lin, false);
      for (int i = 0; i < n; ++i) { hx[i] = frand(-1, 6); hl[i] = int(next() % 7) - 2; }
      up(n, n); k1l<<<n/128,128>>>(tx, dx, dl, dout, n); cudaDeviceSynchronize();
      char tag[8]; snprintf(tag, 8, "L1%d%d", lin, m); dump(tag, n, 1);
    } }
  // 2D layered float4 4x3, 3 layers
  { std::vector<float> t(4 * 3 * 3 * 4); for (auto& v : t) v = frand(-9, 9);
    auto c = cudaCreateChannelDesc<float4>(); cudaArray_t a; cudaMalloc3DArray(&a, &c, make_cudaExtent(4, 3, 3), cudaArrayLayered);
    cudaMemcpy3DParms p{}; p.srcPtr = make_cudaPitchedPtr(t.data(), 64, 4, 3); p.dstArray = a; p.extent = make_cudaExtent(4, 3, 3); p.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p);
    for (int lin = 0; lin < 2; ++lin) {
      auto tx = mk(a, cudaAddressModeClamp, lin, false);
      for (int i = 0; i < n; ++i) { hx[2*i] = frand(-1, 5); hx[2*i+1] = frand(-1, 4); hl[i] = int(next() % 7) - 2; }
      up(2*n, n); k2l<<<n/128,128>>>(tx, dx, dl, (float4*)dout, n); cudaDeviceSynchronize();
      char tag[8]; snprintf(tag, 8, "L2%d", lin); dump(tag, n, 4);
    } }
  // cubemap float, 4x4 faces
  { const int W = 4; std::vector<float> t(W * W * 6); for (int f = 0; f < 6; ++f) for (int i = 0; i < W * W; ++i) t[f * W * W + i] = 100 * f + i + frand(0, 0.5f);
    auto c = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMalloc3DArray(&a, &c, make_cudaExtent(W, W, 6), cudaArrayCubemap);
    cudaMemcpy3DParms p{}; p.srcPtr = make_cudaPitchedPtr(t.data(), W * 4, W, W); p.dstArray = a; p.extent = make_cudaExtent(W, W, 6); p.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p);
    for (int lin = 0; lin < 2; ++lin) {
      auto tx = mk(a, cudaAddressModeClamp, lin, true);
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) hx[3*i+k] = frand(-2, 2);
        if (i % 8 == 1) hx[3*i+1] = hx[3*i];             // |x| == |y| ties
        if (i % 8 == 2) hx[3*i+2] = -hx[3*i];            // |x| == |z|
        if (i % 8 == 3) { hx[3*i+1] = hx[3*i]; hx[3*i+2] = hx[3*i]; }
        if (i % 8 == 4) hx[3*i] = 0.f;
      }
      up(3*n, 1); kc<<<n/128,128>>>(tx, dx, dout, n); cudaDeviceSynchronize();
      char tag[8]; snprintf(tag, 8, "C%d", lin); dump(tag, n, 1);
    }
    // layered cubemap: 2 layers
    std::vector<float> t2(W * W * 12); for (size_t i = 0; i < t2.size(); ++i) t2[i] = float(i) + frand(0, 0.5f);
    cudaArray_t a2; cudaMalloc3DArray(&a2, &c, make_cudaExtent(W, W, 12), cudaArrayCubemap | cudaArrayLayered);
    p.dstArray = a2; p.srcPtr = make_cudaPitchedPtr(t2.data(), W * 4, W, W); p.extent = make_cudaExtent(W, W, 12);
    cudaMemcpy3D(&p);
    for (int lin = 0; lin < 2; ++lin) {
      auto tx = mk(a2, cudaAddressModeClamp, lin, true);
      for (int i = 0; i < n; ++i) { for (int k = 0; k < 3; ++k) hx[3*i+k] = frand(-2, 2); hl[i] = int(next() % 4) - 1; }
      up(3*n, n); kcl<<<n/128,128>>>(tx, dx, dl, dout, n); cudaDeviceSynchronize();
      char tag[8]; snprintf(tag, 8, "CL%d", lin); dump(tag, n, 1);
    } }
  // cubemap under every address mode: point clamps to the face whatever the
  // mode; linear applies it inside the face (wrap, border, mirror)
  { const int W = 4; std::vector<float> t(W * W * 6); for (int i = 0; i < W * W * 6; ++i) t[i] = float(i) * 1.25f - 20;
    auto c = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMalloc3DArray(&a, &c, make_cudaExtent(W, W, 6), cudaArrayCubemap);
    cudaMemcpy3DParms p{}; p.srcPtr = make_cudaPitchedPtr(t.data(), W * 4, W, W); p.dstArray = a; p.extent = make_cudaExtent(W, W, 6); p.kind = cudaMemcpyHostToDevice; cudaMemcpy3D(&p);
    const cudaTextureAddressMode modes[3] = {cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeMirror};
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) hx[3*i+k] = frand(-2, 2);
    up(3*n, 1);
    for (int lin = 0; lin < 2; ++lin) for (int m = 0; m < 3; ++m) {
      auto tx = mk(a, modes[m], lin, true);
      kc<<<n/128,128>>>(tx, dx, dout, n); cudaDeviceSynchronize();
      char tag[8]; snprintf(tag, 8, "CM%d%d", lin, m); dump(tag, n, 1);
    } }
  // layered surface 3x2, 2 layers
  { auto c = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMalloc3DArray(&a, &c, make_cudaExtent(3, 2, 2), cudaArrayLayered | cudaArraySurfaceLoadStore);
    cudaResourceDesc r{}; r.resType = cudaResourceTypeArray; r.res.array.array = a;
    cudaSurfaceObject_t s; cudaCreateSurfaceObject(&s, &r);
    ks<<<1,1>>>(s, 3, 2, 2, dout); cudaDeviceSynchronize(); dump("S", 12, 1); }
  const cudaError_t e = cudaGetLastError();
  printf("%s\n", e == cudaSuccess && fails == 0 ? "PASS" : "FAIL");
  return e == cudaSuccess && fails == 0 ? 0 : 1;
}
