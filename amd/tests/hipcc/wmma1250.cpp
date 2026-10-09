// gfx1250 (CDNA 5, MI455X) matrix instructions, each against the host (wmma1250.gfx1250, wave32). The oracle is AMD's
// CDNA5 ISA (section 7.12): D = A * B + C with C and D in the layout it gives (a lane per column, the rows in
// registers). A and B are loaded here in an order of this program's own -- each half of the wave takes the next half of
// K -- which a WMMA is indifferent to as long as A and B use the same one; the answers are exact (small multiples of
// 1/8), so any difference is a wrong lane or register.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef __bf16 v16y __attribute__((ext_vector_type(16)));
typedef __bf16 v8y __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

__global__ void k_f16(const _Float16* a, const _Float16* b, const float* c, float* d) {
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  v16h va, vb;
  v8f vc;
  for (int s = 0; s < 16; ++s) {
    va[s] = a[r * 32 + h * 16 + s];
    vb[s] = b[(h * 16 + s) * 16 + r];
  }
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const v8f vd = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, va, false, vb, (short)0, vc, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}
__global__ void k_f16_out(const _Float16* a, const _Float16* b, const _Float16* c, _Float16* d) {
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  v16h va, vb;
  v8h vc;
  for (int s = 0; s < 16; ++s) {
    va[s] = a[r * 32 + h * 16 + s];
    vb[s] = b[(h * 16 + s) * 16 + r];
  }
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const v8h vd = __builtin_amdgcn_wmma_f16_16x16x32_f16(false, va, false, vb, (short)0, vc, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}
__global__ void k_bf16(const uint16_t* a, const uint16_t* b, const float* c, float* d) {
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  uint16_t sa[16], sb[16];
  v8f vc;
  for (int s = 0; s < 16; ++s) {
    sa[s] = a[r * 32 + h * 16 + s];
    sb[s] = b[(h * 16 + s) * 16 + r];
  }
  v16y va, vb;
  __builtin_memcpy(&va, sa, 32);
  __builtin_memcpy(&vb, sb, 32);
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const v8f vd = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, va, false, vb, (short)0, vc, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}
__global__ void k_bf16_to_bf16(const uint16_t* a, const uint16_t* b, const float* c, uint16_t* d) {
  // C is f32, D is bf16: bf16f32's D takes the 16-bit layout.
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  uint16_t sa[16], sb[16];
  v8f vc;
  for (int s = 0; s < 16; ++s) {
    sa[s] = a[r * 32 + h * 16 + s];
    sb[s] = b[(h * 16 + s) * 16 + r];
  }
  v16y va, vb;
  __builtin_memcpy(&va, sa, 32);
  __builtin_memcpy(&vb, sb, 32);
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const v8y vd = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(false, va, false, vb, (short)0, vc, false, false);
  uint16_t sd[8];
  __builtin_memcpy(sd, &vd, 16);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = sd[e];
}
__global__ void k_f32(const float* a, const float* b, const float* c, float* d) {
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  v2f va, vb;
  v8f vc;
  for (int s = 0; s < 2; ++s) {
    va[s] = a[r * 4 + h * 2 + s];
    vb[s] = b[(h * 2 + s) * 16 + r];
  }
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const v8f vd = __builtin_amdgcn_wmma_f32_16x16x4_f32(false, va, false, vb, (short)0, vc, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}
template <int K, bool A8, bool B8>
__global__ void k_8bit(const uint8_t* a, const uint8_t* b, const float* c, float* d, int which) {
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  constexpr int S = K / 2;   // bytes of A (and B) a lane holds
  uint8_t sa[S], sb[S];
  for (int s = 0; s < S; ++s) {
    sa[s] = a[r * K + h * S + s];
    sb[s] = b[(h * S + s) * 16 + r];
  }
  v8i va = {}, vb = {};
  __builtin_memcpy(&va, sa, S);
  __builtin_memcpy(&vb, sb, S);
  v8f vc;
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  v8f vd = vc;
  if (which == 0) vd = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(va, vb, (short)0, vc, false, false);
  if (which == 1) vd = __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(va, vb, (short)0, vc, false, false);
  if (which == 2) vd = __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(va, vb, (short)0, vc, false, false);
  if (which == 3) vd = __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(va, vb, (short)0, vc, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}
__global__ void k_iu8(const int8_t* a, const int8_t* b, const int* c, int* d, int a_signed, int b_signed) {
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  int8_t sa[32], sb[32];
  for (int s = 0; s < 32; ++s) {
    sa[s] = a[r * 64 + h * 32 + s];
    sb[s] = b[(h * 32 + s) * 16 + r];
  }
  v8i va, vb, vc;
  __builtin_memcpy(&va, sa, 32);
  __builtin_memcpy(&vb, sb, 32);
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  v8i vd;
  if (a_signed && b_signed) vd = __builtin_amdgcn_wmma_i32_16x16x64_iu8(true, va, true, vb, vc, false, false);
  else if (a_signed) vd = __builtin_amdgcn_wmma_i32_16x16x64_iu8(true, va, false, vb, vc, false, false);
  else vd = __builtin_amdgcn_wmma_i32_16x16x64_iu8(false, va, false, vb, vc, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}


// The layouts below are the ISA's tables, read register-first: for each register of a lane, the K it holds.
typedef _Float16 v32h __attribute__((ext_vector_type(32)));
typedef int v16i __attribute__((ext_vector_type(16)));

// Sparse f16: A packed 16x32 (lane m + 16 * (k' / 16), 16 values a lane, two to a register), B 64x16 (lane n + 16 * (k % 32 / 16),
// registers 0-7 for k < 32, 8-15 for the rest), the index in one register a lane.
__global__ void k_swmmac_f16(const _Float16* ap, const _Float16* b, const float* c, const unsigned* idx, float* d) {
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  v16h va;
  v32h vb;
  v8f vc;
  for (int s = 0; s < 16; ++s) va[s] = ap[r * 32 + 16 * h + s];
  for (int s = 0; s < 32; ++s) vb[s] = b[(32 * (s / 16) + 16 * h + s % 16) * 16 + r];
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const v8f vd = __builtin_amdgcn_swmmac_f32_16x16x64_f16(false, va, false, vb, vc, (int)idx[l], false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}
// 16x16x128 f8f6f4 (unscaled): A in the format a_fmt, B in b_fmt; the data arrive as 8-bit, 6-bit or 4-bit elements packed per the table.
__device__ inline void put_elem(unsigned* regs, int nregs, int bitpos, int width, unsigned value) {
  for (int i = 0; i < width; ++i) {
    const int bit = bitpos + i;
    if ((value >> i) & 1) regs[bit / 32] |= 1u << (bit % 32);
  }
}
// Fills the registers of lane (row/col rc, half h) for format fmt (0 fp8, 1 bf8, 2 fp6, 3 bf6, 4 fp4) from M[rc][k] (k-major bytes).
__device__ inline void pack_f8f6f4(v16i* v, const uint8_t* m, int rc, int h, int fmt) {
  unsigned regs[16] = {};
  if (fmt <= 1) {   // 8 bits: register v holds 4 K at 32 * (v / 4) + 16 * h + 4 * (v % 4)
    for (int vg = 0; vg < 16; ++vg)
      for (int by = 0; by < 4; ++by) put_elem(regs, 16, 32 * vg + 8 * by, 8, m[rc * 128 + 32 * (vg / 4) + 16 * h + 4 * (vg % 4) + by]);
  } else if (fmt <= 3) {   // 6 bits: the lane's K = 64 * set + 32 * h + i is bits 6 * i of the set's 6 registers
    for (int set = 0; set < 2; ++set)
      for (int i = 0; i < 32; ++i) put_elem(regs, 16, 32 * 6 * set + 6 * i, 6, m[rc * 128 + 64 * set + 32 * h + i]);
  } else {   // 4 bits: register v holds 8 K at 64 * (v / 4) + 32 * h + 8 * (v % 4)
    for (int vg = 0; vg < 8; ++vg)
      for (int nb = 0; nb < 8; ++nb) put_elem(regs, 16, 32 * vg + 4 * nb, 4, m[rc * 128 + 64 * (vg / 4) + 32 * h + 8 * (vg % 4) + nb]);
  }
  for (int i = 0; i < 16; ++i) (*v)[i] = static_cast<int>(regs[i]);
}
__global__ void k_f8f6f4(const uint8_t* a, const uint8_t* b, const float* c, float* d, int fa, int fb) {
  // a: 16x128, row-major; b: 16x128 with row n holding column n of B.
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  v16i va, vb;
  pack_f8f6f4(&va, a, r, h, fa);
  pack_f8f6f4(&vb, b, r, h, fb);
  v8f vc;
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  v8f vd = vc;
  if (fa == 0 && fb == 0) vd = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(0, va, 0, vb, (short)0, vc);
  else if (fa == 2 && fb == 4) vd = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(2, va, 4, vb, (short)0, vc);
  else if (fa == 4 && fb == 4) vd = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(4, va, 4, vb, (short)0, vc);
  else if (fa == 3 && fb == 1) vd = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(3, va, 1, vb, (short)0, vc);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}
// The block-scaled form: 32 K to a scale, E8M0 scales, A's from lanes 0-15 of a register (bytes = K blocks 0-3) and B's likewise.
__global__ void k_scaled(const uint8_t* a, const uint8_t* b, const float* c, float* d, const uint8_t* sa, const uint8_t* sb, int fa, int fb) {
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  v16i va, vb;
  pack_f8f6f4(&va, a, r, h, fa);
  pack_f8f6f4(&vb, b, r, h, fb);
  v8f vc;
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const int scale_a = l < 16 ? (sa[r * 4] | sa[r * 4 + 1] << 8 | sa[r * 4 + 2] << 16 | sa[r * 4 + 3] << 24) : 0;
  const int scale_b = l < 16 ? (sb[r * 4] | sb[r * 4 + 1] << 8 | sb[r * 4 + 2] << 16 | sb[r * 4 + 3] << 24) : 0;
  v8f vd = vc;
  if (fa == 0 && fb == 0)
    vd = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0, va, 0, vb, (short)0, vc, 0, 0, scale_a, 0, 0, scale_b, false, false);
  else if (fa == 4 && fb == 4)
    vd = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(4, va, 4, vb, (short)0, vc, 0, 0, scale_a, 0, 0, scale_b, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}

static int g_checks = 0, g_failed = 0;
static void report(const char* name, int wrong, int n) {
  ++g_checks;
  if (wrong) ++g_failed;
  std::printf("%s: %d of %d wrong\n", name, wrong, n);
}
static uint32_t g_seed = 99;
static int rnd(int n) {
  g_seed = g_seed * 1103515245u + 12345u;
  return static_cast<int>((g_seed >> 8) % n);
}
static uint16_t half_bits(float f) {
  _Float16 h = static_cast<_Float16>(f);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
static float half_val(uint16_t b) {
  _Float16 h;
  std::memcpy(&h, &b, 2);
  return static_cast<float>(h);
}
static uint16_t bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>(u >> 16);
}
static float bf16_val(uint16_t b) {
  uint32_t u = uint32_t{b} << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// OCP E4M3 and E5M2 (bias 7 and 15): the value of a byte, and a byte for the few values the test uses.
static float e4m3(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  return (b & 0x80 ? -1.0f : 1.0f) * (e == 0 ? m / 8.0f * std::ldexp(1.0f, -6) : (1.0f + m / 8.0f) * std::ldexp(1.0f, e - 7));
}
static float e5m2(uint8_t b) {
  const int e = (b >> 2) & 31, m = b & 3;
  return (b & 0x80 ? -1.0f : 1.0f) * (e == 0 ? m / 4.0f * std::ldexp(1.0f, -14) : (1.0f + m / 4.0f) * std::ldexp(1.0f, e - 15));
}

// Small floats, as the ISA lists them: E2M1 (fp4), E2M3 (fp6), E3M2 (bf6).
static float small_float(uint32_t b, int mant, int bias, int bits) {
  const uint32_t sign = (b >> (bits - 1)) & 1, mag = b & ((1u << (bits - 1)) - 1);
  const int e = static_cast<int>(mag >> mant), m = static_cast<int>(mag & ((1u << mant) - 1));
  const float v = e == 0 ? std::ldexp(static_cast<float>(m), 1 - bias - mant) : std::ldexp(static_cast<float>((1 << mant) | m), e - bias - mant);
  return sign ? -v : v;
}
static float decode_fmt(int fmt, uint8_t b) {
  switch (fmt) {
    case 0: return e4m3(b);
    case 1: return e5m2(b);
    case 2: return small_float(b, 3, 1, 6);
    case 3: return small_float(b, 2, 3, 6);
    default: return small_float(b, 1, 1, 4);
  }
}
static uint8_t pick_code(int fmt) {
  // Codes with small magnitudes: a few of each format's lowest exponents.
  static const uint8_t k8[8] = {0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0, 0x34};
  static const uint8_t k5[8] = {0x00, 0x38, 0x3C, 0x40, 0xB8, 0xBC, 0xC0, 0x3A};
  switch (fmt) {
    case 0: return k8[rnd(8)];
    case 1: return k5[rnd(8)];
    case 2: return static_cast<uint8_t>(rnd(24) | (rnd(2) << 5));   // exponent bits 0-2 only
    case 3: return static_cast<uint8_t>(rnd(16) | (rnd(2) << 5));
    default: return static_cast<uint8_t>(rnd(6) | (rnd(2) << 3));
  }
}

template <class T>
static T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  if (hipMalloc(&d, v.size() * sizeof(T)) != hipSuccess) return nullptr;
  hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}
template <class T>
static std::vector<T> download(T* d, size_t n) {
  std::vector<T> v(n);
  hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
  return v;
}

int main() {
  // 16x32 half and bfloat16 inputs: small multiples of 1/8.
  const int K = 32;
  std::vector<float> fa(16 * K), fb(K * 16), fc(256);
  for (auto& x : fa) x = (rnd(33) - 16) / 8.0f;
  for (auto& x : fb) x = (rnd(33) - 16) / 8.0f;
  for (auto& x : fc) x = (rnd(129) - 64) / 8.0f;
  const auto reference = [&](int Kd, const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C) {
    std::vector<double> D(256);
    for (int m = 0; m < 16; ++m)
      for (int n = 0; n < 16; ++n) {
        double s = C[m * 16 + n];
        for (int k = 0; k < Kd; ++k) s += double{A[m * Kd + k]} * B[k * 16 + n];
        D[m * 16 + n] = s;
      }
    return D;
  };
  {
    std::vector<_Float16> ha(fa.size()), hb(fb.size());
    for (size_t i = 0; i < fa.size(); ++i) ha[i] = static_cast<_Float16>(fa[i]);
    for (size_t i = 0; i < fb.size(); ++i) hb[i] = static_cast<_Float16>(fb[i]);
    float* dd = nullptr;
    CHECK(hipMalloc(&dd, 256 * sizeof(float)));
    k_f16<<<1, 32>>>(upload(ha), upload(hb), upload(fc), dd);
    CHECK(hipDeviceSynchronize());
    const auto out = download(dd, 256);
    const auto ref = reference(K, fa, fb, fc);
    int wrong = 0;
    for (int i = 0; i < 256; ++i) wrong += out[i] != static_cast<float>(ref[i]);
    report("f16 16x16x32 into f32", wrong, 256);
    // The same into a half result.
    std::vector<_Float16> hc(256);
    std::vector<float> fc16(256);
    for (int i = 0; i < 256; ++i) hc[i] = static_cast<_Float16>(fc[i]), fc16[i] = static_cast<float>(hc[i]);
    _Float16* dh = nullptr;
    CHECK(hipMalloc(&dh, 256 * sizeof(_Float16)));
    k_f16_out<<<1, 32>>>(upload(ha), upload(hb), upload(hc), dh);
    CHECK(hipDeviceSynchronize());
    const auto outh = download(dh, 256);
    const auto refh = reference(K, fa, fb, fc16);
    wrong = 0;
    for (int i = 0; i < 256; ++i) wrong += static_cast<float>(outh[i]) != static_cast<float>(static_cast<_Float16>(refh[i]));
    report("f16 16x16x32 into f16", wrong, 256);
  }
  {
    std::vector<uint16_t> ba(fa.size()), bb(fb.size());
    for (size_t i = 0; i < fa.size(); ++i) ba[i] = bf16_bits(fa[i]);
    for (size_t i = 0; i < fb.size(); ++i) bb[i] = bf16_bits(fb[i]);
    float* dd = nullptr;
    CHECK(hipMalloc(&dd, 256 * sizeof(float)));
    k_bf16<<<1, 32>>>(upload(ba), upload(bb), upload(fc), dd);
    CHECK(hipDeviceSynchronize());
    const auto out = download(dd, 256);
    const auto ref = reference(K, fa, fb, fc);
    int wrong = 0;
    for (int i = 0; i < 256; ++i) wrong += out[i] != static_cast<float>(ref[i]);
    report("bf16 16x16x32 into f32", wrong, 256);
    uint16_t* db = nullptr;
    CHECK(hipMalloc(&db, 256 * sizeof(uint16_t)));
    k_bf16_to_bf16<<<1, 32>>>(upload(ba), upload(bb), upload(fc), db);
    CHECK(hipDeviceSynchronize());
    const auto outb = download(db, 256);
    wrong = 0;
    for (int i = 0; i < 256; ++i) wrong += outb[i] != bf16_bits(static_cast<float>(ref[i]));
    report("bf16 16x16x32 into bf16 with an f32 C", wrong, 256);
  }
  {
    std::vector<float> a4(16 * 4), b4(4 * 16);
    for (auto& x : a4) x = (rnd(33) - 16) / 8.0f;
    for (auto& x : b4) x = (rnd(33) - 16) / 8.0f;
    float* dd = nullptr;
    CHECK(hipMalloc(&dd, 256 * sizeof(float)));
    k_f32<<<1, 32>>>(upload(a4), upload(b4), upload(fc), dd);
    CHECK(hipDeviceSynchronize());
    const auto out = download(dd, 256);
    const auto ref = reference(4, a4, b4, fc);
    int wrong = 0;
    for (int i = 0; i < 256; ++i) wrong += out[i] != static_cast<float>(ref[i]);
    report("f32 16x16x4", wrong, 256);
  }
  {
    // 8-bit floats: bytes picked from the small values, in each of the four format pairs.
    const int K8 = 64;
    std::vector<uint8_t> ba(16 * K8), bb(K8 * 16);
    static const uint8_t kPick[8] = {0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0, 0x34};   // E4M3: 0, .5, 1, 2, -.5, -1, -2, .625
    static const uint8_t kPick5[8] = {0x00, 0x38, 0x3C, 0x40, 0xB8, 0xBC, 0xC0, 0x3A};   // E5M2: 0, .5, 1, 2, -.5, -1, -2, .75
    int wrong_all = 0;
    for (int which = 0; which < 4; ++which) {
      const bool a_fp8 = which < 2, b_fp8 = which % 2 == 0;
      std::vector<float> fa8(16 * K8), fb8(K8 * 16);
      for (int i = 0; i < 16 * K8; ++i) {
        ba[i] = (a_fp8 ? kPick : kPick5)[rnd(8)];
        fa8[i] = a_fp8 ? e4m3(ba[i]) : e5m2(ba[i]);
      }
      for (int i = 0; i < K8 * 16; ++i) {
        bb[i] = (b_fp8 ? kPick : kPick5)[rnd(8)];
        fb8[i] = b_fp8 ? e4m3(bb[i]) : e5m2(bb[i]);
      }
      float* dd = nullptr;
      CHECK(hipMalloc(&dd, 256 * sizeof(float)));
      k_8bit<64, true, true><<<1, 32>>>(upload(ba), upload(bb), upload(fc), dd, which);
      CHECK(hipDeviceSynchronize());
      const auto out = download(dd, 256);
    const auto ref = reference(K8, fa8, fb8, fc);
      for (int i = 0; i < 256; ++i) wrong_all += out[i] != static_cast<float>(ref[i]);
    }
    report("fp8 and bf8 16x16x64, each pairing", wrong_all, 4 * 256);
  }
  {
    // iu8: A signed or not, B signed or not.
    const int K8 = 64;
    std::vector<int8_t> a8(16 * K8), b8(K8 * 16);
    std::vector<int> c32(256);
    for (auto& x : a8) x = static_cast<int8_t>(rnd(256) - 128);
    for (auto& x : b8) x = static_cast<int8_t>(rnd(256) - 128);
    for (auto& x : c32) x = rnd(2001) - 1000;
    int wrong = 0;
    for (int mode = 0; mode < 3; ++mode) {
      const int as = mode >= 1, bs = mode == 2;
      int* dd = nullptr;
      CHECK(hipMalloc(&dd, 256 * sizeof(int)));
      k_iu8<<<1, 32>>>(upload(a8), upload(b8), upload(c32), dd, as, bs);
      CHECK(hipDeviceSynchronize());
      const auto out = download(dd, 256);
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
          int64_t s = c32[m * 16 + n];
          for (int k = 0; k < K8; ++k) {
            const int64_t x = as ? a8[m * K8 + k] : static_cast<uint8_t>(a8[m * K8 + k]);
            const int64_t y = bs ? b8[k * 16 + n] : static_cast<uint8_t>(b8[k * 16 + n]);
            s += x * y;
          }
          wrong += out[m * 16 + n] != static_cast<int>(s);
        }
    }
    report("iu8 16x16x64, signed and unsigned", wrong, 3 * 256);
  }

  {
    // Sparse f16 (swmmac 16x16x64): A is 4:2 sparse with two indices a group of four columns.
    std::vector<float> expanded(16 * 64, 0.0f), packed(16 * 32);
    std::vector<unsigned> idx(32, 0);
    for (int m = 0; m < 16; ++m)
      for (int g = 0; g < 16; ++g) {
        int i0 = rnd(3), i1 = i0 + 1 + rnd(3 - i0);   // i0 < i1, both in 0..3
        const float v0 = (rnd(33) - 16) / 8.0f, v1 = (rnd(33) - 16) / 8.0f;
        expanded[m * 64 + 4 * g + i0] = v0;
        expanded[m * 64 + 4 * g + i1] = v1;
        packed[m * 32 + 2 * g] = v0;
        packed[m * 32 + 2 * g + 1] = v1;
        // The pair's two 2-bit indices go to the lane of the group's half (lane m + 16 * (g / 8)), at bit 4 * (g % 8).
        idx[m + 16 * (g / 8)] |= static_cast<unsigned>(i0 | i1 << 2) << (4 * (g % 8));
      }
    std::vector<float> bk(64 * 16);
    for (auto& x : bk) x = (rnd(33) - 16) / 8.0f;
    std::vector<_Float16> hp(packed.size()), hb(bk.size());
    for (size_t i = 0; i < packed.size(); ++i) hp[i] = static_cast<_Float16>(packed[i]);
    for (size_t i = 0; i < bk.size(); ++i) hb[i] = static_cast<_Float16>(bk[i]);
    float* dd = nullptr;
    CHECK(hipMalloc(&dd, 256 * sizeof(float)));
    k_swmmac_f16<<<1, 32>>>(upload(hp), upload(hb), upload(fc), upload(idx), dd);
    CHECK(hipDeviceSynchronize());
    const auto out = download(dd, 256);
    int wrong = 0;
    for (int m = 0; m < 16; ++m)
      for (int n = 0; n < 16; ++n) {
        double sum = fc[m * 16 + n];
        for (int k = 0; k < 64; ++k) sum += double{expanded[m * 64 + k]} * bk[k * 16 + n];
        wrong += out[m * 16 + n] != static_cast<float>(sum);
      }
    report("sparse f16 16x16x64 (swmmac)", wrong, 256);
  }
  {
    // The 16x16x128 small-float forms, unscaled and scaled, in the register layouts the ISA draws for each width.
    const int formats[4][2] = {{0, 0}, {2, 4}, {4, 4}, {3, 1}};
    int wrong = 0;
    for (const auto& f : formats) {
      std::vector<uint8_t> a(16 * 128), b(16 * 128);   // b: row n holds column n of B
      std::vector<float> va(16 * 128), vb(16 * 128);
      for (int i = 0; i < 16 * 128; ++i) a[i] = pick_code(f[0]), va[i] = decode_fmt(f[0], a[i]);
      for (int i = 0; i < 16 * 128; ++i) b[i] = pick_code(f[1]), vb[i] = decode_fmt(f[1], b[i]);
      float* dd = nullptr;
      CHECK(hipMalloc(&dd, 256 * sizeof(float)));
      k_f8f6f4<<<1, 32>>>(upload(a), upload(b), upload(fc), dd, f[0], f[1]);
      CHECK(hipDeviceSynchronize());
      const auto out = download(dd, 256);
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
          double sum = fc[m * 16 + n];
          for (int k = 0; k < 128; ++k) sum += double{va[m * 128 + k]} * vb[n * 128 + k];
          wrong += out[m * 16 + n] != static_cast<float>(sum);
        }
    }
    report("f8f6f4 16x16x128 (fp8, fp6 x fp4, fp4, bf6 x bf8)", wrong, 4 * 256);
    wrong = 0;
    const int scaled_formats[2][2] = {{0, 0}, {4, 4}};
    for (const auto& f : scaled_formats) {
      std::vector<uint8_t> a(16 * 128), b(16 * 128), sa(16 * 4), sb(16 * 4);
      std::vector<float> va(16 * 128), vb(16 * 128);
      for (int i = 0; i < 16 * 128; ++i) a[i] = pick_code(f[0]), va[i] = decode_fmt(f[0], a[i]);
      for (int i = 0; i < 16 * 128; ++i) b[i] = pick_code(f[1]), vb[i] = decode_fmt(f[1], b[i]);
      for (auto& x : sa) x = static_cast<uint8_t>(125 + rnd(5));   // 2^-2 .. 2^2
      for (auto& x : sb) x = static_cast<uint8_t>(125 + rnd(5));
      float* dd = nullptr;
      CHECK(hipMalloc(&dd, 256 * sizeof(float)));
      k_scaled<<<1, 32>>>(upload(a), upload(b), upload(fc), dd, upload(sa), upload(sb), f[0], f[1]);
      CHECK(hipDeviceSynchronize());
      const auto out = download(dd, 256);
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
          double sum = fc[m * 16 + n];
          for (int blk = 0; blk < 4; ++blk) {
            double part = 0;
            for (int k = 32 * blk; k < 32 * blk + 32; ++k) part += double{va[m * 128 + k]} * vb[n * 128 + k];
            sum += part * std::ldexp(1.0, sa[m * 4 + blk] - 127) * std::ldexp(1.0, sb[n * 4 + blk] - 127);
          }
          wrong += out[m * 16 + n] != static_cast<float>(sum);
        }
    }
    report("block-scaled 16x16x128 (fp8, fp4; E8M0 scales, 32 to a scale)", wrong, 2 * 256);
  }
  std::printf("wmma1250: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
