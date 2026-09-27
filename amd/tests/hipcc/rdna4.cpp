// RDNA4 (gfx1201) instructions the executor runs its own way, each against the
// host (rdna4.gfx1201, wave32):
//
//   - WMMA in gfx12's layout: lane l holds row (column) l % 16 and half of K,
//     the half l / 16; element i of lane l is D[i + 8 * (l / 16)][l % 16],
//     16-bit results two to a register. f16, bf16, iu8 and OCP fp8 inputs.
//   - The scalar float unit: arithmetic on values every lane shares compiles
//     to s_add_f32, s_mul_f32, s_fmac_f32, conversions, and the vector unit's
//     transcendentals into scalar registers (v_s_rcp_f32, v_s_sqrt_f32, ...).
//   - The split barrier (s_barrier_signal / s_barrier_wait): waves hand values
//     to one another through LDS.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef short v8s __attribute__((ext_vector_type(8)));
typedef __bf16 v8y __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

// Loads a lane's half of row (column) r of a 16x16 matrix stored k-major for
// B, row-major for A: K values 8 * (l / 16) through 8 * (l / 16) + 7.
__global__ void wmma_f16(const _Float16* a, const _Float16* b, const float* c, float* d) {
  const int l = threadIdx.x, r = l % 16, k0 = 8 * (l / 16);
  v8h va, vb;
  v8f vc;
  for (int e = 0; e < 8; ++e) {
    va[e] = a[r * 16 + k0 + e];
    vb[e] = b[(k0 + e) * 16 + r];
    vc[e] = c[(e + 8 * (l / 16)) * 16 + r];
  }
  const v8f vd = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, vb, vc);
  for (int e = 0; e < 8; ++e) d[(e + 8 * (l / 16)) * 16 + r] = vd[e];
}
__global__ void wmma_bf16(const short* a, const short* b, const float* c, float* d) {
  const int l = threadIdx.x, r = l % 16, k0 = 8 * (l / 16);
  v8s sa, sb;
  v8f vc;
  for (int e = 0; e < 8; ++e) {
    sa[e] = a[r * 16 + k0 + e];
    sb[e] = b[(k0 + e) * 16 + r];
    vc[e] = c[(e + 8 * (l / 16)) * 16 + r];
  }
  v8y va, vb;
  __builtin_memcpy(&va, &sa, 16);
  __builtin_memcpy(&vb, &sb, 16);
  const v8f vd = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va, vb, vc);
  for (int e = 0; e < 8; ++e) d[(e + 8 * (l / 16)) * 16 + r] = vd[e];
}
__global__ void wmma_f16_out(const _Float16* a, const _Float16* b, const _Float16* c, _Float16* d) {
  const int l = threadIdx.x, r = l % 16, k0 = 8 * (l / 16);
  v8h va, vb, vc;
  for (int e = 0; e < 8; ++e) {
    va[e] = a[r * 16 + k0 + e];
    vb[e] = b[(k0 + e) * 16 + r];
    vc[e] = c[(e + 8 * (l / 16)) * 16 + r];
  }
  const v8h vd = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(va, vb, vc);
  for (int e = 0; e < 8; ++e) d[(e + 8 * (l / 16)) * 16 + r] = vd[e];
}
template <bool Fp8>
__global__ void wmma_8bit(const unsigned char* a, const unsigned char* b, const void* c, void* d) {
  const int l = threadIdx.x, r = l % 16, k0 = 8 * (l / 16);
  unsigned char ab[8], bb[8];
  for (int e = 0; e < 8; ++e) {
    ab[e] = a[r * 16 + k0 + e];
    bb[e] = b[(k0 + e) * 16 + r];
  }
  v2i va, vb;
  __builtin_memcpy(&va, ab, 8);
  __builtin_memcpy(&vb, bb, 8);
  if constexpr (Fp8) {
    v8f vc;
    for (int e = 0; e < 8; ++e) vc[e] = static_cast<const float*>(c)[(e + 8 * (l / 16)) * 16 + r];
    const v8f vd = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(va, vb, vc);
    for (int e = 0; e < 8; ++e) static_cast<float*>(d)[(e + 8 * (l / 16)) * 16 + r] = vd[e];
  } else {
    v8i vc;
    for (int e = 0; e < 8; ++e) vc[e] = static_cast<const int*>(c)[(e + 8 * (l / 16)) * 16 + r];
    const v8i vd = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, va, false, vb, vc, false);
    for (int e = 0; e < 8; ++e) static_cast<int*>(d)[(e + 8 * (l / 16)) * 16 + r] = vd[e];
  }
}

// Uniform float arithmetic: every lane computes the same, so the compiler
// uses the scalar unit.
__global__ void scalar_float(float a, float b, int i, float* out) {
  if (threadIdx.x != 0) return;
  out[0] = a + b;
  out[1] = a * b;
  out[2] = a - b;
  out[3] = __builtin_fmaf(a, b, 1.5f);
  out[4] = static_cast<float>(i);
  out[5] = static_cast<float>(static_cast<int>(a * 3.0f));
  out[6] = __builtin_amdgcn_rcpf(b);
  out[7] = __builtin_sqrtf(b);
  out[8] = __builtin_amdgcn_exp2f(a);
  out[9] = fmaxf(a, b);
}

// Each wave writes its lanes to LDS, waits at the barrier, and reads another
// wave's.
__global__ void barrier_exchange(int* out) {
  __shared__ int lds[256];
  const int t = threadIdx.x;
  lds[t] = t * 3 + 1;
  __syncthreads();
  out[t] = lds[(t + 96) % 256];
}

int wrong = 0;
void report(const char* what, int bad, int of) {
  std::printf("%s: %d of %d wrong\n", what, bad, of);
  wrong += bad;
}
template <typename T>
T* upload(const T* h, size_t n) {
  T* d = nullptr;
  (void)hipMalloc(&d, n * sizeof(T));
  (void)hipMemcpy(d, h, n * sizeof(T), hipMemcpyHostToDevice);
  return d;
}
template <typename T>
void download(T* h, const T* d, size_t n) {
  (void)hipMemcpy(h, d, n * sizeof(T), hipMemcpyDeviceToHost);
}

// OCP E4M3: bias 7, S.1111.111 NaN.
double e4m3(unsigned v) {
  const unsigned mag = v & 0x7F;
  const int e = static_cast<int>(mag >> 3), m = static_cast<int>(mag & 7);
  const double x = e == 0 ? std::ldexp(m, -9) : std::ldexp(8 | m, e - 10);
  return v & 0x80 ? -x : x;
}

int main() {
  uint32_t seed = 11;
  const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
  _Float16 ah[256], bh[256], ch[256], dh[256];
  short ab[256], bb[256];
  float c[256], d[256], a8f[256], b8f[256];
  unsigned char a8[256], b8[256], a8u[256], b8u[256];
  int ci[256], di[256];
  for (int i = 0; i < 256; ++i) {
    const float x = static_cast<float>(static_cast<int>(next() % 9) - 4) / 2, y = static_cast<float>(static_cast<int>(next() % 9) - 4) / 2;
    ah[i] = static_cast<_Float16>(x);
    bh[i] = static_cast<_Float16>(y);
    uint32_t fx, fy;
    std::memcpy(&fx, &x, 4);
    std::memcpy(&fy, &y, 4);
    ab[i] = static_cast<short>(fx >> 16);
    bb[i] = static_cast<short>(fy >> 16);
    c[i] = static_cast<float>(static_cast<int>(next() % 17) - 8);
    ch[i] = static_cast<_Float16>(c[i]);
    // fp8: small values (exponents near the bias), never NaN.
    a8[i] = static_cast<unsigned char>((next() % 2) << 7 | (5 + next() % 5) << 3 | next() % 8);
    b8[i] = static_cast<unsigned char>((next() % 2) << 7 | (5 + next() % 5) << 3 | next() % 8);
    a8f[i] = static_cast<float>(e4m3(a8[i]));
    b8f[i] = static_cast<float>(e4m3(b8[i]));
    a8u[i] = static_cast<unsigned char>(next());
    b8u[i] = static_cast<unsigned char>(next());
    ci[i] = static_cast<int>(next() % 1001) - 500;
  }
  const auto ref = [](int row, int col, auto av, auto bv, double acc) {
    for (int k = 0; k < 16; ++k) acc += static_cast<double>(av(row * 16 + k)) * static_cast<double>(bv(k * 16 + col));
    return acc;
  };
  const auto fl = [](const _Float16* p) { return [p](int i) { return static_cast<float>(p[i]); }; };
  int bad = 0;
  float* dc = upload(c, 256);
  float* dd = upload(c, 256);
  wmma_f16<<<1, 32>>>(upload(ah, 256), upload(bh, 256), dc, dd);
  CHECK(hipDeviceSynchronize());
  download(d, dd, 256);
  for (int i = 0; i < 256; ++i) bad += d[i] != static_cast<float>(ref(i / 16, i % 16, fl(ah), fl(bh), c[i]));
  report("v_wmma_f32_16x16x16_f16", bad, 256);

  bad = 0;
  wmma_bf16<<<1, 32>>>(upload(ab, 256), upload(bb, 256), dc, dd);
  CHECK(hipDeviceSynchronize());
  download(d, dd, 256);
  for (int i = 0; i < 256; ++i) bad += d[i] != static_cast<float>(ref(i / 16, i % 16, fl(ah), fl(bh), c[i]));
  report("v_wmma_f32_16x16x16_bf16", bad, 256);

  bad = 0;
  _Float16* ddh = upload(ch, 256);
  wmma_f16_out<<<1, 32>>>(upload(ah, 256), upload(bh, 256), upload(ch, 256), ddh);
  CHECK(hipDeviceSynchronize());
  download(dh, ddh, 256);
  for (int i = 0; i < 256; ++i)
    bad += dh[i] != static_cast<_Float16>(ref(i / 16, i % 16, fl(ah), fl(bh), static_cast<double>(ch[i])));
  report("v_wmma_f16_16x16x16_f16", bad, 256);

  bad = 0;
  wmma_8bit<true><<<1, 32>>>(upload(a8, 256), upload(b8, 256), dc, dd);
  CHECK(hipDeviceSynchronize());
  download(d, dd, 256);
  for (int i = 0; i < 256; ++i) {
    const double want = ref(i / 16, i % 16, [&](int j) { return a8f[j]; }, [&](int j) { return b8f[j]; }, c[i]);
    bad += std::fabs(d[i] - want) > 1e-6 * std::fmax(1.0, std::fabs(want));
  }
  report("v_wmma_f32_16x16x16_fp8_fp8 (OCP)", bad, 256);

  bad = 0;
  int* ddi = upload(ci, 256);
  wmma_8bit<false><<<1, 32>>>(upload(a8u, 256), upload(b8u, 256), upload(ci, 256), ddi);
  CHECK(hipDeviceSynchronize());
  download(di, ddi, 256);
  for (int i = 0; i < 256; ++i)
    bad += di[i] != static_cast<int>(ref(i / 16, i % 16, [&](int j) { return static_cast<int>(static_cast<signed char>(a8u[j])); },
                                         [&](int j) { return static_cast<int>(b8u[j]); }, ci[i]));
  report("v_wmma_i32_16x16x16_iu8 (signed A, unsigned B)", bad, 256);

  {
    const float a = 2.75f, b = 1.25f;
    float out[10];
    float* dout = upload(out, 10);
    scalar_float<<<1, 32>>>(a, b, -7, dout);
    CHECK(hipDeviceSynchronize());
    download(out, dout, 10);
    const float want[10] = {a + b, a * b, a - b, std::fma(a, b, 1.5f), -7.0f, static_cast<float>(static_cast<int>(a * 3.0f)),
                            1.0f / b, std::sqrt(b), std::exp2(a), std::fmax(a, b)};
    int bad2 = 0;
    for (int k = 0; k < 10; ++k) bad2 += std::fabs(out[k] - want[k]) > 1e-6f * std::fabs(want[k]);
    report("scalar float arithmetic and transcendentals", bad2, 10);
  }
  {
    int out[256];
    int* dout = upload(out, 256);
    barrier_exchange<<<1, 256>>>(dout);
    CHECK(hipDeviceSynchronize());
    download(out, dout, 256);
    int bad2 = 0;
    for (int t = 0; t < 256; ++t) bad2 += out[t] != ((t + 96) % 256) * 3 + 1;
    report("the split barrier orders LDS between waves", bad2, 256);
  }
  std::printf("%s\n", wrong ? "FAIL" : "all RDNA4 checks right");
  return wrong != 0;
}
