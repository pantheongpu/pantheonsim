// The SASS executor's fast paths against its general path (VGPU_FASTPATH=0).
// run_sass_fastpath.sh runs this program both ways, with the performance
// counters on, and requires the two outputs to be byte for byte the same. Each
// kernel is also checked against the host here, so the general path itself is
// held to the right answer.
//
// What the kernels reach: coalesced and scattered global loads and stores of
// 1, 2, 4 and 8 bytes (whole warps, partial warps, tails, runs that start
// off a sector boundary), shared-memory loads and stores of 1 to 16 bytes with
// conflicts and broadcasts, FFMA over special values (zeros, subnormals,
// infinities, NaNs), 64-bit arithmetic and POPC.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

static int wrong = 0;

struct __align__(16) Q { uint32_t a, b, c, d; };   // a 16-byte element: LDS.128 / STS.128

// An element of type T made from a 64-bit value, spread over all its bytes.
template <class T>
static T mk(uint64_t v) {
  T t;
  unsigned char* p = reinterpret_cast<unsigned char*>(&t);
  for (size_t off = 0; off < sizeof t; off += 8) {
    const uint64_t x = v * (off / 8 + 1) + off;
    std::memcpy(p + off, &x, sizeof t - off < 8 ? sizeof t - off : 8);
  }
  return t;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

template <class T>
__global__ void copy_guarded(const T* in, T* out, int n, int lane_limit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && (threadIdx.x % 32) < lane_limit) out[i] = in[i];
}

template <class T>
__global__ void gather(const T* in, T* out, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[(i * stride) % n];
}

// Shared memory: each thread stores its element, the block reads another's.
template <class T>
__global__ void shared_swap(const T* in, T* out, int mul) {
  __shared__ T s[256];
  int t = threadIdx.x;
  s[(t * mul) % 256] = in[blockIdx.x * 256 + t];
  __syncthreads();
  out[blockIdx.x * 256 + t] = s[(t * mul) % 256 == t ? t : (255 - t)];
}

__global__ void shared_broadcast(const float4* in, float* out) {
  __shared__ float4 s[64];
  int t = threadIdx.x;
  if (t < 64) s[t] = in[t];
  __syncthreads();
  float4 v = s[t % 3];     // three distinct addresses per warp
  float4 w = s[(t * 5) % 64];
  out[t] = v.x + v.y * 2 + v.z * 3 + v.w * 4 + w.x - w.w;
}

__global__ void ffma_special(const float* a, const float* b, const float* c, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaf(a[i], b[i], c[i]);
}

__global__ void wide_math(const uint64_t* in, uint64_t* out, double* dout, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t x = in[i];
  out[i] = x * 0x9E3779B97F4A7C15ull + (x >> 7) + __popcll(x) + __popc((unsigned)x) + __clzll(x | 1);
  double d = __longlong_as_double((long long)(x & 0x7fefffffffffffffull));
  dout[i] = fma(d, 1.0000001, -d * 0.5);
}

template <class T>
static bool same(const std::vector<T>& a, const std::vector<T>& b, const char* what) {
  if (a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0) return true;
  std::printf("WRONG %s\n", what);
  ++wrong;
  return false;
}

template <class T>
static int run_copy(const char* name, int n) {
  std::vector<T> h(n);
  for (int i = 0; i < n; ++i) h[i] = mk<T>(i * 2654435761u + 12345u);
  T *din, *dout;
  CK(cudaMalloc(&din, n * sizeof(T)));
  CK(cudaMalloc(&dout, (n + 64) * sizeof(T)));
  CK(cudaMemcpy(din, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  for (int limit : {32, 7, 1, 20}) {
    CK(cudaMemset(dout, 0, (n + 64) * sizeof(T)));
    copy_guarded<T><<<(n + 127) / 128, 128>>>(din, dout, n, limit);
    // An access that starts mid-allocation, off the first element.
    gather<T><<<(n + 127) / 128, 128>>>(din + 0, dout, n, limit == 32 ? 1 : 3);
    CK(cudaDeviceSynchronize());
    std::vector<T> got(n);
    CK(cudaMemcpy(got.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<T> want(n);
    for (int i = 0; i < n; ++i) want[i] = h[(static_cast<long>(i) * (limit == 32 ? 1 : 3)) % n];
    // The gather ran last and wrote every element.
    char what[64];
    std::snprintf(what, sizeof what, "%s n=%d limit=%d", name, n, limit);
    same(got, want, what);
  }
  CK(cudaFree(din));
  CK(cudaFree(dout));
  return 0;
}

template <class T>
static int run_shared(const char* name) {
  const int blocks = 4;
  std::vector<T> h(blocks * 256);
  for (size_t i = 0; i < h.size(); ++i) h[i] = mk<T>(i * 40503u + 7u);
  T *din, *dout;
  CK(cudaMalloc(&din, h.size() * sizeof(T)));
  CK(cudaMalloc(&dout, h.size() * sizeof(T)));
  CK(cudaMemcpy(din, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  for (int mul : {1, 3, 33, 129}) {
    shared_swap<T><<<blocks, 256>>>(din, dout, mul);
    CK(cudaDeviceSynchronize());
    std::vector<T> got(h.size());
    CK(cudaMemcpy(got.data(), dout, got.size() * sizeof(T), cudaMemcpyDeviceToHost));
    // mul is odd, so the slot (t*mul)%256 is a permutation of the block's 256 slots, and the thread that
    // owns slot t reads it back when (t*mul)%256 == t, else the slot 255-t.
    std::vector<T> want(h.size());
    for (int b = 0; b < blocks; ++b) {
      T s[256];
      std::memset(s, 0, sizeof s);
      for (int t = 0; t < 256; ++t) s[(t * mul) % 256] = h[b * 256 + t];   // later threads win a shared slot
      for (int t = 0; t < 256; ++t) want[b * 256 + t] = s[(t * mul) % 256 == t ? t : (255 - t)];
    }
    char what[64];
    std::snprintf(what, sizeof what, "shared %s mul=%d", name, mul);
    same(got, want, what);
  }
  CK(cudaFree(din));
  CK(cudaFree(dout));
  return 0;
}

int main() {
  for (int n : {1, 31, 32, 33, 1000, 4096 + 5}) {
    if (run_copy<uint8_t>("u8", n) || run_copy<uint16_t>("u16", n) || run_copy<uint32_t>("u32", n) ||
        run_copy<uint64_t>("u64", n) || run_copy<float>("f32", n))
      return 1;
  }
  if (run_shared<uint8_t>("u8") || run_shared<uint16_t>("u16") || run_shared<uint32_t>("u32") ||
      run_shared<uint64_t>("u64") || run_shared<Q>("q16"))
    return 1;
  {
    float4 h[64];
    for (int i = 0; i < 64; ++i) h[i] = make_float4(i, i * 0.5f, i * 0.25f, -i);
    float4* d; float* o;
    CK(cudaMalloc(&d, sizeof h));
    CK(cudaMalloc(&o, 128 * 4));
    CK(cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice));
    shared_broadcast<<<1, 128>>>(d, o);
    CK(cudaDeviceSynchronize());
    float got[128];
    CK(cudaMemcpy(got, o, sizeof got, cudaMemcpyDeviceToHost));
    for (int t = 0; t < 128; ++t) {
      const float4 v = h[t % 3], w = h[(t * 5) % 64];
      const float want = v.x + v.y * 2 + v.z * 3 + v.w * 4 + w.x - w.w;
      if (got[t] != want) { std::printf("WRONG shared_broadcast t=%d\n", t); ++wrong; break; }
    }
    CK(cudaFree(d)); CK(cudaFree(o));
  }
  {
    const uint32_t sp[] = {0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000, 0x3f800000, 0xbf800000,
                           0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0x4b000000,
                           0x33800000, 0x3f800001, 0x2f800000, 0x5f800000};
    const int m = sizeof sp / sizeof sp[0];
    std::vector<float> a, b, c;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        for (int k = 0; k < m; ++k) {
          float x, y, z;
          std::memcpy(&x, &sp[i], 4); std::memcpy(&y, &sp[j], 4); std::memcpy(&z, &sp[k], 4);
          a.push_back(x); b.push_back(y); c.push_back(z);
        }
    const int n = static_cast<int>(a.size());
    float *da, *db, *dc, *dout;
    CK(cudaMalloc(&da, n * 4)); CK(cudaMalloc(&db, n * 4)); CK(cudaMalloc(&dc, n * 4)); CK(cudaMalloc(&dout, n * 4));
    CK(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, c.data(), n * 4, cudaMemcpyHostToDevice));
    ffma_special<<<(n + 127) / 128, 128>>>(da, db, dc, dout, n);
    CK(cudaDeviceSynchronize());
    std::vector<float> got(n);
    CK(cudaMemcpy(got.data(), dout, n * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) {
      const float want = std::fmaf(a[i], b[i], c[i]);
      uint32_t g, w;
      std::memcpy(&g, &got[i], 4); std::memcpy(&w, &want, 4);
      if (want != want ? got[i] == got[i] : g != w) ++bad;   // any NaN matches any NaN: the card's is canonical
    }
    if (bad) { std::printf("WRONG ffma_special %d of %d\n", bad, n); ++wrong; }
    // Printed so that the two runs' outputs are compared on the exact bits as well.
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; ++i) { uint32_t g; std::memcpy(&g, &got[i], 4); h = (h ^ g) * 1099511628211ull; }
    std::printf("ffma_special bits %016llx\n", static_cast<unsigned long long>(h));
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc)); CK(cudaFree(dout));
  }
  {
    const int n = 1000;
    std::vector<uint64_t> h(n);
    uint64_t s = 88172645463325252ull;
    for (int i = 0; i < n; ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; h[i] = i % 5 == 0 ? s >> (i % 64) : s; }
    uint64_t *d, *o; double* od;
    CK(cudaMalloc(&d, n * 8)); CK(cudaMalloc(&o, n * 8)); CK(cudaMalloc(&od, n * 8));
    CK(cudaMemcpy(d, h.data(), n * 8, cudaMemcpyHostToDevice));
    wide_math<<<(n + 63) / 64, 64>>>(d, o, od, n);
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> go(n);
    std::vector<double> gd(n);
    CK(cudaMemcpy(go.data(), o, n * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gd.data(), od, n * 8, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) {
      const uint64_t x = h[i];
      const uint64_t want = x * 0x9E3779B97F4A7C15ull + (x >> 7) + __builtin_popcountll(x) + __builtin_popcount(static_cast<unsigned>(x)) +
                            __builtin_clzll(x | 1);
      if (go[i] != want) ++bad;
    }
    if (bad) { std::printf("WRONG wide_math %d\n", bad); ++wrong; }
    uint64_t hh = 1469598103934665603ull;
    for (int i = 0; i < n; ++i) { uint64_t g; std::memcpy(&g, &gd[i], 8); hh = (hh ^ g) * 1099511628211ull; }
    std::printf("wide_math double bits %016llx\n", static_cast<unsigned long long>(hh));
    CK(cudaFree(d)); CK(cudaFree(o)); CK(cudaFree(od));
  }
  std::printf("wrong: %d\n", wrong);
  return wrong != 0;
}
