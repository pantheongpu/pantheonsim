// RDNA3 (gfx1100) instructions the executor runs its own way, each against the
// host. Built twice: rdna3.gfx1100 as HIP builds for RDNA (wave32) and
// rdna3.w64.gfx1100 with -mwavefrontsize64 (-DVGPU_W64), which MIOpen's and
// rocFFT's wave64 kernels use.
//
//   - VOPD: both halves read before either writes (a Y that reads X's
//     destination sees it as it was).
//   - A 64-bit float operand's literal is the double's high half (sin() in
//     double reduces its argument against constants given that way).
//   - WMMA (wave32): D = A x B + C for a 16x16x16 tile, f16/bf16/iu8 inputs,
//     f32/f16/i32 results, as AMD lays the tile out across the wave.
//   - DPP with bank masks, row mirrors and VOP3 output modifiers, in wave64.
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

#ifndef VGPU_W64
constexpr int kWave = 32;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

// VOPD: X writes v2 (a + b); Y shifts v2 as it was (the lane) left by 2.
__global__ void vopd(const float* a, const float* b, float* sum, int* shifted) {
  const int l = threadIdx.x;
  float s;
  int sh;
  asm volatile(
      "v_mov_b32 v2, %2\n\t"
      "v_mov_b32 v12, %3\n\t"
      "v_mov_b32 v13, %4\n\t"
      "v_dual_add_f32 v2, v12, v13 :: v_dual_lshlrev_b32 v3, 2, v2\n\t"
      "v_mov_b32 %0, v2\n\t"
      "v_mov_b32 %1, v3"
      : "=v"(s), "=v"(sh)
      : "v"(l), "v"(a[l]), "v"(b[l])
      : "v2", "v3", "v12", "v13");
  sum[l] = s;
  shifted[l] = sh;
}

// A tile as AMD lays it out for gfx11 WMMA: lane l holds row (A) or column
// (B) l % 16, all sixteen of K; register i of lane l is D[2i + l / 16][l % 16].
__global__ void wmma_f16(const _Float16* a, const _Float16* b, const float* c, float* d) {
  const int l = threadIdx.x, r = l % 16;
  v16h va, vb;
  v8f vc;
  for (int k = 0; k < 16; ++k) {
    va[k] = a[r * 16 + k];
    vb[k] = b[k * 16 + r];
  }
  for (int i = 0; i < 8; ++i) vc[i] = c[(2 * i + l / 16) * 16 + r];
  const v8f vd = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, vb, vc);
  for (int i = 0; i < 8; ++i) d[(2 * i + l / 16) * 16 + r] = vd[i];
}
__global__ void wmma_bf16(const short* a, const short* b, const float* c, float* d) {
  const int l = threadIdx.x, r = l % 16;
  v16s va, vb;
  v8f vc;
  for (int k = 0; k < 16; ++k) {
    va[k] = a[r * 16 + k];
    vb[k] = b[k * 16 + r];
  }
  for (int i = 0; i < 8; ++i) vc[i] = c[(2 * i + l / 16) * 16 + r];
  const v8f vd = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, vc);
  for (int i = 0; i < 8; ++i) d[(2 * i + l / 16) * 16 + r] = vd[i];
}
// f16 results into the high half of each register (OP_SEL), the low half kept.
__global__ void wmma_f16_out(const _Float16* a, const _Float16* b, const _Float16* c, _Float16* d) {
  const int l = threadIdx.x, r = l % 16;
  v16h va, vb, vc = {};
  for (int k = 0; k < 16; ++k) {
    va[k] = a[r * 16 + k];
    vb[k] = b[k * 16 + r];
  }
  for (int i = 0; i < 8; ++i) vc[2 * i + 1] = c[(2 * i + l / 16) * 16 + r];
  const v16h vd = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(va, vb, vc, true);
  for (int i = 0; i < 8; ++i) d[(2 * i + l / 16) * 16 + r] = vd[2 * i + 1];
}
// Signed A, unsigned B.
__global__ void wmma_iu8(const signed char* a, const unsigned char* b, const int* c, int* d) {
  const int l = threadIdx.x, r = l % 16;
  v4i va, vb;
  v8i vc;
  unsigned char ab[16], bb[16];
  for (int k = 0; k < 16; ++k) {
    ab[k] = static_cast<unsigned char>(a[r * 16 + k]);
    bb[k] = b[k * 16 + r];
  }
  __builtin_memcpy(&va, ab, 16);
  __builtin_memcpy(&vb, bb, 16);
  for (int i = 0; i < 8; ++i) vc[i] = c[(2 * i + l / 16) * 16 + r];
  const v8i vd = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, va, false, vb, vc, false);
  for (int i = 0; i < 8; ++i) d[(2 * i + l / 16) * 16 + r] = vd[i];
}
#else
constexpr int kWave = 64;
// wave64: DPP with a bank mask into an fmac, a quad permutation into an add,
// output modifiers, and the row mirrors.
__global__ void dpp64(const float* in, float* out) {
  const int l = threadIdx.x;
  const float a = in[l], b = in[64 + l], d = in[128 + l];
  float r = d;
  asm volatile("v_fmac_f32_dpp %0, %1, %2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc" : "+v"(r) : "v"(a), "v"(b));
  out[l] = r;
  asm volatile("v_add_f32_dpp %0, %1, %2 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf" : "=v"(r) : "v"(a), "v"(b));
  out[64 + l] = r;
  asm volatile("v_add_f32_e64 %0, %1, %2 div:2" : "=v"(r) : "v"(a), "v"(b));
  out[128 + l] = r;
  asm volatile("v_subrev_f32_e64 %0, %1, %2 div:2" : "=v"(r) : "v"(a), "v"(b));
  out[192 + l] = r;
  asm volatile("v_mov_b32_dpp %0, %1 row_mirror row_mask:0xf bank_mask:0xf" : "=v"(r) : "v"(a));
  out[256 + l] = r;
  asm volatile("v_mov_b32_dpp %0, %1 row_half_mirror row_mask:0xf bank_mask:0xf" : "=v"(r) : "v"(a));
  out[320 + l] = r;
}
#endif

// sin() in double, whose range reduction compares against 64-bit literals.
__global__ void dsin(const double* x, double* y) { y[threadIdx.x] = sin(x[threadIdx.x]); }

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

int main() {
  uint32_t seed = 7;
  const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };

  {
    double x[kWave], y[kWave];
    for (int i = 0; i < kWave; ++i) x[i] = -6.0 + 0.37 * i;
    double* dx = upload(x, kWave);
    double* dy = upload(x, kWave);
    dsin<<<1, kWave>>>(dx, dy);
    CHECK(hipDeviceSynchronize());
    download(y, dy, kWave);
    int bad = 0;
    for (int i = 0; i < kWave; ++i) bad += std::fabs(y[i] - std::sin(x[i])) > 1e-12;
    report("sin in double", bad, kWave);
  }

#ifndef VGPU_W64
  {
    float a[32], b[32], s[32];
    int sh[32];
    for (int i = 0; i < 32; ++i) a[i] = 1.5f * i, b[i] = 100.0f;
    float* da = upload(a, 32);
    float* db = upload(b, 32);
    float* ds = upload(a, 32);
    int* dsh = upload(sh, 32);
    vopd<<<1, 32>>>(da, db, ds, dsh);
    CHECK(hipDeviceSynchronize());
    download(s, ds, 32);
    download(sh, dsh, 32);
    int bad = 0;
    for (int i = 0; i < 32; ++i) bad += s[i] != a[i] + b[i] || sh[i] != i << 2;
    report("VOPD reads both halves' sources first", bad, 32);
  }
  {
    // Small integers and halves, so every sum is exact.
    _Float16 ah[256], bh[256], ch[256], dh[256];
    short ab[256], bb[256];
    float c[256], d[256];
    signed char ai[256];
    unsigned char bi[256];
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
      ai[i] = static_cast<signed char>(static_cast<int>(next() % 256) - 128);
      bi[i] = static_cast<unsigned char>(next() % 256);
      ci[i] = static_cast<int>(next() % 1001) - 500;
    }
    const auto ref = [&](int row, int col, auto av, auto bv, double acc) {
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
    report("v_wmma_f16_16x16x16_f16 (high halves)", bad, 256);

    bad = 0;
    int* ddi = upload(ci, 256);
    wmma_iu8<<<1, 32>>>(upload(ai, 256), upload(bi, 256), upload(ci, 256), ddi);
    CHECK(hipDeviceSynchronize());
    download(di, ddi, 256);
    for (int i = 0; i < 256; ++i)
      bad += di[i] != static_cast<int>(ref(i / 16, i % 16, [&](int j) { return static_cast<int>(ai[j]); },
                                           [&](int j) { return static_cast<int>(bi[j]); }, ci[i]));
    report("v_wmma_i32_16x16x16_iu8 (signed A, unsigned B)", bad, 256);
  }
#else
  {
    float h[192], o[384];
    for (int i = 0; i < 192; ++i) h[i] = (i % 64) + 0.25f * (i / 64) + 1;
    float* di = upload(h, 192);
    float* dout = upload(o, 384);
    dpp64<<<1, 64>>>(di, dout);
    CHECK(hipDeviceSynchronize());
    download(o, dout, 384);
    const float *a = h, *b = h + 64, *d = h + 128;
    static const int qa[4] = {2, 3, 0, 1}, qb[4] = {0, 0, 0, 2};
    int bad = 0;
    for (int l = 0; l < 64; ++l) {
      const int bank = (l >> 2) & 3;
      bad += o[l] != ((0xc >> bank) & 1 ? d[l] + a[(l & ~3) + qa[l & 3]] * b[l] : d[l]);
      bad += o[64 + l] != a[(l & ~3) + qb[l & 3]] + b[l];
      bad += o[128 + l] != (a[l] + b[l]) / 2;
      bad += o[192 + l] != (b[l] - a[l]) / 2;
      bad += o[256 + l] != a[(l & ~15) + 15 - (l & 15)];
      bad += o[320 + l] != a[(l & ~7) + 7 - (l & 7)];
    }
    report("wave64 DPP, bank masks, row mirrors and output modifiers", bad, 384);
  }
#endif
  std::printf("%s\n", wrong ? "FAIL" : "all RDNA3 checks right");
  return wrong != 0;
}
