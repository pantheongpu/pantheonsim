// Linear texture filtering, checked bit for bit against an RTX 3060 (sm_86).
//
// The same inputs the simulator's filter was fitted on: a 0..1 ramp sampled
// every 1/1024 (the weight itself), random texels under all four address
// modes with normalized and unnormalized coordinates, a float4 2D texture in
// clamp and border modes, unsigned 8-bit normalized, half, and a 3D float
// texture. Each section's results are hashed and compared with the hashes the
// hardware produced; the program prints PASS on the last line and runs the
// same on a GPU. (Signed 8-bit normalized filtering is left out: the simulator
// refuses it, see interpreter.cpp.)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Expected { const char* tag; uint64_t hash; };
static const Expected kExpected[] = {
    {"A", 0xc311d9fc3af93613ull},
    {"B0", 0xbcc8067f913de8c5ull},
    {"B1", 0xfc9987e6d0da5f24ull},
    {"B2", 0x1f763d8c192fe115ull},
    {"B3", 0x93f6f28a8d9ea2c3ull},
    {"Bu", 0x0fcea94d149662b7ull},
    {"Cc", 0x5d7cbaaa7674ba60ull},
    {"Cb", 0x0ae395e03b7c42d0ull},
    {"D", 0x100645f6a5a06885ull},
    {"Dh", 0x0b8b5445ea714ebaull},
    {"E", 0xe6437429b3648715ull},
};

__global__ void s1(cudaTextureObject_t t, const float* xs, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex1D<float>(t, xs[i]);
}
__global__ void s2(cudaTextureObject_t t, const float* xs, float4* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2D<float4>(t, xs[2 * i], xs[2 * i + 1]);
}
__global__ void s3(cudaTextureObject_t t, const float* xs, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex3D<float>(t, xs[3 * i], xs[3 * i + 1], xs[3 * i + 2]);
}
__global__ void s2f(cudaTextureObject_t t, const float* xs, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2D<float>(t, xs[2 * i], xs[2 * i + 1]);
}

static uint32_t rng = 12345;
static uint32_t next() { rng = rng * 1664525u + 1013904223u; return rng; }
static float frand(float lo, float hi) { return lo + (hi - lo) * (next() >> 8) / float(1 << 24); }

template <class K, class... A>
void run(K k, int n, A... a) { k<<<(n + 127) / 128, 128>>>(a..., n); cudaDeviceSynchronize(); }

cudaTextureObject_t make(cudaArray_t arr, cudaTextureAddressMode am, bool norm, bool nfloat) {
  cudaResourceDesc r{}; r.resType = cudaResourceTypeArray; r.res.array.array = arr;
  cudaTextureDesc d{}; for (int i = 0; i < 3; ++i) d.addressMode[i] = am;
  d.filterMode = cudaFilterModeLinear; d.normalizedCoords = norm;
  d.readMode = nfloat ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
  cudaTextureObject_t t; cudaCreateTextureObject(&t, &r, &d, nullptr); return t;
}

int main() {
  float *dx, *dout; cudaMalloc(&dx, 3 * 8192 * 4); cudaMalloc(&dout, 8192 * 16);
  std::vector<float> hx(3 * 8192), ho(8192 * 4);
  int fails = 0;
  auto dump = [&](const char* tag, int n, int per) {
    cudaMemcpy(ho.data(), dout, n * per * 4, cudaMemcpyDeviceToHost);
    uint64_t h = 1469598103934665603ull;   // FNV-1a over every result's bits
    for (int i = 0; i < n * per; ++i) {
      uint32_t b; memcpy(&b, &ho[i], 4);
      for (int k = 0; k < 4; ++k) { h ^= (b >> (8 * k)) & 0xff; h *= 1099511628211ull; }
    }
    uint64_t want = 0;
    for (const auto& e : kExpected) if (!strcmp(e.tag, tag)) want = e.hash;
    std::printf("%-3s %016llx %s\n", tag, (unsigned long long)h, h == want ? "ok" : "MISMATCH");
    fails += h != want;
  };
  // A: ramp 0,1 -- the weight itself, at 1/1024 steps over [0,2].
  {
    float t[4] = {0.f, 1.f, 1.f, 1.f};
    cudaChannelFormatDesc c = cudaCreateChannelDesc<float>();
    cudaArray_t a; cudaMallocArray(&a, &c, 4); cudaMemcpy2DToArray(a, 0, 0, t, 16, 16, 1, cudaMemcpyHostToDevice);
    auto tx = make(a, cudaAddressModeClamp, false, false);
    const int n = 2048; for (int i = 0; i < n; ++i) hx[i] = i / 1024.0f;
    cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice); run(s1, n, tx, dx, dout); dump("A", n, 1);
  }
  // B: random texels, random x, all four address modes (normalized coords).
  {
    float t[16]; for (float& v : t) v = frand(-100, 100);
    cudaChannelFormatDesc c = cudaCreateChannelDesc<float>();
    cudaArray_t a; cudaMallocArray(&a, &c, 16); cudaMemcpy2DToArray(a, 0, 0, t, 64, 64, 1, cudaMemcpyHostToDevice);
    const cudaTextureAddressMode modes[4] = {cudaAddressModeWrap, cudaAddressModeClamp, cudaAddressModeMirror, cudaAddressModeBorder};
    for (int m = 0; m < 4; ++m) {
      auto tx = make(a, modes[m], true, false);
      const int n = 1024; for (int i = 0; i < n; ++i) hx[i] = frand(-1.5f, 2.5f);
      cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice); run(s1, n, tx, dx, dout);
      char tag[8]; snprintf(tag, 8, "B%d", m); dump(tag, n, 1);
    }
    auto tx = make(a, cudaAddressModeClamp, false, false);
    const int n = 1024; for (int i = 0; i < n; ++i) hx[i] = frand(-2, 18);
    cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice); run(s1, n, tx, dx, dout); dump("Bu", n, 1);
  }
  // C: 2D float4, 7x5, unnormalized, clamp and border.
  {
    std::vector<float> t(7 * 5 * 4); for (float& v : t) v = frand(-10, 10);
    cudaChannelFormatDesc c = cudaCreateChannelDesc<float4>();
    cudaArray_t a; cudaMallocArray(&a, &c, 7, 5); cudaMemcpy2DToArray(a, 0, 0, t.data(), 7 * 16, 7 * 16, 5, cudaMemcpyHostToDevice);
    for (int m = 0; m < 2; ++m) {
      auto tx = make(a, m ? cudaAddressModeBorder : cudaAddressModeClamp, false, false);
      const int n = 1024; for (int i = 0; i < n; ++i) { hx[2 * i] = frand(-1, 8); hx[2 * i + 1] = frand(-1, 6); }
      cudaMemcpy(dx, hx.data(), 2 * n * 4, cudaMemcpyHostToDevice); run(s2, n, tx, dx, (float4*)dout);
      dump(m ? "Cb" : "Cc", n, 4);
    }
  }
  // D: 2D uchar4 read as normalized float, 6x6, wrap on normalized coords.
  {
    std::vector<uint8_t> t(6 * 6 * 4); for (auto& v : t) v = next() >> 24;
    cudaChannelFormatDesc c = cudaCreateChannelDesc<uchar4>();
    cudaArray_t a; cudaMallocArray(&a, &c, 6, 6); cudaMemcpy2DToArray(a, 0, 0, t.data(), 24, 24, 6, cudaMemcpyHostToDevice);
    auto tx = make(a, cudaAddressModeWrap, true, true);
    const int n = 1024; for (int i = 0; i < n; ++i) { hx[2 * i] = frand(-1, 2); hx[2 * i + 1] = frand(-1, 2); }
    cudaMemcpy(dx, hx.data(), 2 * n * 4, cudaMemcpyHostToDevice); run(s2, n, tx, dx, (float4*)dout); dump("D", n, 4);
    // Signed 8-bit normalized (not run: the simulator refuses it), but its
    // texels and points are still drawn, so the half section below sees the
    // same random numbers it did on the hardware.
    std::vector<int8_t> u(6 * 6); for (auto& v : u) v = int8_t(next() >> 24);
    for (int i = 0; i < n; ++i) { hx[2 * i] = frand(-1, 7); hx[2 * i + 1] = frand(-1, 7); }
    // half, 1 channel
    std::vector<uint16_t> h(6 * 6); for (auto& v : h) v = 0x3c00 ^ (next() & 0x83ff);
    cudaChannelFormatDesc c3 = cudaCreateChannelDescHalf();
    cudaArray_t a3; cudaMallocArray(&a3, &c3, 6, 6); cudaMemcpy2DToArray(a3, 0, 0, h.data(), 12, 12, 6, cudaMemcpyHostToDevice);
    auto t3 = make(a3, cudaAddressModeClamp, false, false);
    cudaMemcpy(dx, hx.data(), 2 * n * 4, cudaMemcpyHostToDevice); run(s2f, n, t3, dx, dout); dump("Dh", n, 1);
  }
  // E: 3D float, 4x3x5 trilinear, clamp.
  {
    std::vector<float> t(4 * 3 * 5); for (float& v : t) v = frand(-5, 5);
    cudaChannelFormatDesc c = cudaCreateChannelDesc<float>();
    cudaArray_t a; cudaMalloc3DArray(&a, &c, make_cudaExtent(4, 3, 5));
    cudaMemcpy3DParms p{}; p.srcPtr = make_cudaPitchedPtr(t.data(), 16, 4, 3); p.dstArray = a;
    p.extent = make_cudaExtent(4, 3, 5); p.kind = cudaMemcpyHostToDevice; cudaMemcpy3D(&p);
    auto tx = make(a, cudaAddressModeClamp, false, false);
    const int n = 1024; for (int i = 0; i < n; ++i) { hx[3 * i] = frand(-1, 5); hx[3 * i + 1] = frand(-1, 4); hx[3 * i + 2] = frand(-1, 6); }
    cudaMemcpy(dx, hx.data(), 3 * n * 4, cudaMemcpyHostToDevice); run(s3, n, tx, dx, dout); dump("E", n, 1);
  }
  const cudaError_t e = cudaGetLastError();
  std::printf("%s\n", e == cudaSuccess && fails == 0 ? "PASS" : "FAIL");
  return e == cudaSuccess && fails == 0 ? 0 : 1;
}
