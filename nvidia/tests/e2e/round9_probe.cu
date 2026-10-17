// round9_probe.cu: a small program that exercises what round 9 changed on purpose, and prints raw bits, so that the simulator (both
// engines) and the real card can be compared line by line: nvidia/tests/e2e/run_round9_probe.sh, against the card's lines in
// nvidia/tests/data/round9_probe.rtx3060.txt. (Written by the r9-verify agent; kept here unchanged but for this comment.)
//
//   nvcc -arch=sm_86 -cudart shared round9_probe.cu -o round9_probe        (embeds sm_86 SASS and compute_86 PTX)
//
// Sections (each prints "<name> <hash> <first values>"): tensor-core sums (f16, bf16, tf32 wmma), float NaN selection and
// canonicalisation, NaN to integer conversion, f32 <-> f16 NaN, double NaN, negative zero under .sat/.ftz, bf16 HFMA2,
// 64-bit conversions with saturation, __sad / __usad (VABSDIFF), and a launch with a bad configuration (error codes).
// Inputs come from a fixed LCG, so every run of the same build prints the same text.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace nvcuda;

static uint32_t g_seed = 12345u;
static uint32_t lcg() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

static uint64_t fnv(const void* p, size_t n) {
  const unsigned char* b = static_cast<const unsigned char*>(p);
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
static void report(const char* name, const void* p, size_t bytes, size_t show_words) {
  std::printf("%-22s %016llx", name, (unsigned long long)fnv(p, bytes));
  const uint32_t* w = static_cast<const uint32_t*>(p);
  for (size_t i = 0; i < show_words && (i + 1) * 4 <= bytes; ++i) std::printf(" %08x", w[i]);
  std::printf("\n");
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA error %d at line %d: %s\n", (int)e, __LINE__, #x); } } while (0)

// ---- tensor cores: D = sum over KT tiles of A_k * B_k, all from random data with a wide exponent range
template <typename T> __device__ T from_float(float f);
template <> __device__ half from_float<half>(float f) { return __float2half(f); }
template <> __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float f) { return __float2bfloat16(f); }

template <typename T>
__global__ void wmma16(const float* a, const float* b, float* d, int kt) {
  __shared__ T sa[16 * 16], sb[16 * 16];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int k = 0; k < kt; ++k) {
    for (int i = threadIdx.x; i < 256; i += 32) { sa[i] = from_float<T>(a[k * 256 + i]); sb[i] = from_float<T>(b[k * 256 + i]); }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
    wmma::load_matrix_sync(fa, sa, 16);
    wmma::load_matrix_sync(fb, sb, 16);
    wmma::mma_sync(acc, fa, fb, acc);
    __syncwarp();
  }
  wmma::store_matrix_sync(d, acc, 16, wmma::mem_row_major);
}

__global__ void wmma_tf32(const float* a, const float* b, float* d, int kt) {
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int k = 0; k < kt; ++k) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fb;
    wmma::load_matrix_sync(fa, a + k * 128, 8);
    wmma::load_matrix_sync(fb, b + k * 128, 8);
    for (int i = 0; i < fa.num_elements; ++i) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
    for (int i = 0; i < fb.num_elements; ++i) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
    wmma::mma_sync(acc, fa, fb, acc);
  }
  wmma::store_matrix_sync(d, acc, 16, wmma::mem_row_major);
}

// ---- float and double operations on NaNs and other edge values: one thread per input pair
__global__ void fops(const float* x, const float* y, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = x[i], b = y[i], r[16];
  r[0] = a + b;
  r[1] = a * b;
  r[2] = fmaf(a, b, a);
  r[3] = fminf(a, b);
  r[4] = fmaxf(a, b);
  r[5] = __int_as_float(__float2int_rz(a));
  r[6] = __half2float(__float2half(a));
  r[7] = __bfloat162float(__float2bfloat16(a));
  r[8] = __frcp_rn(a);
  r[9] = sqrtf(a);
  r[10] = __expf(a);
  r[11] = __logf(a);
  r[12] = __sinf(a);
  r[13] = rintf(a);
  r[14] = truncf(a);
  r[15] = __uint_as_float(__float2uint_rn(a));
  for (int k = 0; k < 16; ++k) out[i * 16 + k] = r[k];
}

__global__ void dops(const double* x, const double* y, double* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double a = x[i], b = y[i];
  out[i * 6 + 0] = a + b;
  out[i * 6 + 1] = a * b;
  out[i * 6 + 2] = fma(a, b, a);
  out[i * 6 + 3] = fmin(a, b);
  out[i * 6 + 4] = fmax(a, b);
  out[i * 6 + 5] = (double)(float)a;
}

// negative zero under .sat and .ftz
__global__ void satzero(float* out) {
  float nz = -0.0f, sub = __int_as_float(0x80000001), one = 1.0f, z;
  float r[8];
  asm("add.sat.f32 %0, %1, %2;" : "=f"(r[0]) : "f"(nz), "f"(nz));
  asm("mul.sat.f32 %0, %1, %2;" : "=f"(r[1]) : "f"(nz), "f"(one));
  asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(r[2]) : "f"(nz), "f"(one), "f"(nz));
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(r[3]) : "f"(sub), "f"(nz));
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r[4]) : "f"(sub), "f"(one));
  asm("cvt.sat.f32.f32 %0, %1;" : "=f"(r[5]) : "f"(nz));
  asm("cvt.ftz.f32.f32 %0, %1;" : "=f"(r[6]) : "f"(sub));
  z = __saturatef(nz); r[7] = z;
  for (int k = 0; k < 8; ++k) out[k] = r[k];
}

// bf16 HFMA2 and half HFMA2 on edge values
__global__ void hfma(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  __nv_bfloat162 xa, xb, xc, r;
  memcpy(&xa, &a[i], 4); memcpy(&xb, &b[i], 4); memcpy(&xc, &c[i], 4);
  r = __hfma2(xa, xb, xc);
  memcpy(&out[2 * i], &r, 4);
  __half2 ha, hb, hc, hr;
  memcpy(&ha, &a[i], 4); memcpy(&hb, &b[i], 4); memcpy(&hc, &c[i], 4);
  hr = __hfma2(ha, hb, hc);
  memcpy(&out[2 * i + 1], &hr, 4);
}

// 64-bit conversions, saturation, and VABSDIFF (__sad / __usad)
__global__ void ints(const double* d, const float* f, const int* s, long long* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i * 8 + 0] = (long long)d[i];
  out[i * 8 + 1] = (long long)(unsigned long long)d[i];
  out[i * 8 + 2] = (long long)f[i];
  out[i * 8 + 3] = (long long)__double2ll_rn(d[i]);
  out[i * 8 + 4] = (long long)__sad(s[i], s[(i + 1) % n], (unsigned)s[(i + 2) % n]);
  out[i * 8 + 5] = (long long)__usad((unsigned)s[i], (unsigned)s[(i + 1) % n], (unsigned)s[(i + 2) % n]);
  int sat;
  asm("add.sat.s32 %0, %1, %2;" : "=r"(sat) : "r"(s[i]), "r"(s[(i + 3) % n]));
  out[i * 8 + 6] = sat;
  long long sl;
  asm("cvt.rzi.sat.s64.f64 %0, %1;" : "=l"(sl) : "d"(d[i]));
  out[i * 8 + 7] = sl;
}

__global__ void nothing() {}

static float f_from_bits(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static double d_from_bits(uint64_t b) { double f; std::memcpy(&f, &b, 8); return f; }

int main(int argc, char** argv) {
  const bool full = argc > 1;   // "full": also print every double result with its inputs (to see which pair differs)
  // tensor cores
  const int KT = 32;
  std::vector<float> ha(KT * 256), hb(KT * 256);
  for (int i = 0; i < KT * 256; ++i) {
    // values with a spread of exponents and both signs: the order and rounding of the sum matter
    int e = (int)(lcg() % 12) - 6;
    float m = 1.0f + (lcg() % 1024) / 1024.0f;
    ha[i] = ((lcg() & 1) ? -m : m) * std::ldexp(1.0f, e);
    e = (int)(lcg() % 12) - 6; m = 1.0f + (lcg() % 1024) / 1024.0f;
    hb[i] = ((lcg() & 1) ? -m : m) * std::ldexp(1.0f, e);
  }
  float *da, *db, *dd;
  CK(cudaMalloc(&da, KT * 256 * 4)); CK(cudaMalloc(&db, KT * 256 * 4)); CK(cudaMalloc(&dd, 256 * 4));
  CK(cudaMemcpy(da, ha.data(), KT * 256 * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, hb.data(), KT * 256 * 4, cudaMemcpyHostToDevice));
  std::vector<float> hd(256);
  CK(cudaMemset(dd, 0, 256 * 4)); wmma16<half><<<1, 32>>>(da, db, dd, KT); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hd.data(), dd, 1024, cudaMemcpyDeviceToHost)); report("wmma_f16", hd.data(), 1024, 4);
  CK(cudaMemset(dd, 0, 256 * 4)); wmma16<__nv_bfloat16><<<1, 32>>>(da, db, dd, KT); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hd.data(), dd, 1024, cudaMemcpyDeviceToHost)); report("wmma_bf16", hd.data(), 1024, 4);
  CK(cudaMemset(dd, 0, 256 * 4)); wmma_tf32<<<1, 32>>>(da, db, dd, KT * 2); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hd.data(), dd, 1024, cudaMemcpyDeviceToHost)); report("wmma_tf32", hd.data(), 1024, 4);

  // float edge values, including NaNs with sign and payload
  const uint32_t fb[] = {0x7fc00000u, 0xffc00000u, 0x7fc00001u, 0xffa55555u, 0x7f800001u, 0x7f800000u, 0xff800000u,
                         0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x3f800000u, 0xbf800000u, 0x7f7fffffu,
                         0x4f000000u, 0xcf000000u, 0x4f800000u, 0x477fe000u, 0x33000000u, 0x3effffffu};
  const int NF = (int)(sizeof(fb) / sizeof(fb[0]));
  const int NP = NF * NF;
  std::vector<float> fx(NP), fy(NP);
  for (int i = 0; i < NF; ++i) for (int j = 0; j < NF; ++j) { fx[i * NF + j] = f_from_bits(fb[i]); fy[i * NF + j] = f_from_bits(fb[j]); }
  float *dx, *dy, *dout;
  CK(cudaMalloc(&dx, NP * 4)); CK(cudaMalloc(&dy, NP * 4)); CK(cudaMalloc(&dout, NP * 16 * 4));
  CK(cudaMemcpy(dx, fx.data(), NP * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dy, fy.data(), NP * 4, cudaMemcpyHostToDevice));
  fops<<<(NP + 127) / 128, 128>>>(dx, dy, dout, NP); CK(cudaDeviceSynchronize());
  std::vector<float> fo(NP * 16);
  CK(cudaMemcpy(fo.data(), dout, NP * 16 * 4, cudaMemcpyDeviceToHost));
  const char* fn[16] = {"fadd", "fmul", "ffma", "fmin", "fmax", "f2i", "f2h2f", "f2bf2f", "rcp", "sqrt", "ex2", "lg2", "sin", "rint", "trunc", "f2u"};
  for (int k = 0; k < 16; ++k) {
    std::vector<uint32_t> col(NP);
    for (int p = 0; p < NP; ++p) std::memcpy(&col[p], &fo[p * 16 + k], 4);
    char nm[32]; std::snprintf(nm, sizeof nm, "f32_%s", fn[k]);
    report(nm, col.data(), NP * 4, 6);
  }

  // doubles
  const uint64_t db64[] = {0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff8000000000001ull, 0xfff4000000000000ull,
                           0x7ff0000000000001ull, 0x7ff0000000000000ull, 0xfff0000000000000ull, 0x0ull, 0x8000000000000000ull,
                           0x1ull, 0x3ff0000000000000ull, 0xbff0000000000000ull, 0x7fefffffffffffffull, 0x43e0000000000000ull,
                           0xc3e0000000000000ull, 0x43f0000000000000ull, 0x47efffffe0000000ull, 0x3df0000000000000ull};
  const int ND = (int)(sizeof(db64) / sizeof(db64[0])), NPD = ND * ND;
  std::vector<double> dxv(NPD), dyv(NPD);
  for (int i = 0; i < ND; ++i) for (int j = 0; j < ND; ++j) { dxv[i * ND + j] = d_from_bits(db64[i]); dyv[i * ND + j] = d_from_bits(db64[j]); }
  double *ddx, *ddy, *ddo;
  CK(cudaMalloc(&ddx, NPD * 8)); CK(cudaMalloc(&ddy, NPD * 8)); CK(cudaMalloc(&ddo, NPD * 6 * 8));
  CK(cudaMemcpy(ddx, dxv.data(), NPD * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ddy, dyv.data(), NPD * 8, cudaMemcpyHostToDevice));
  dops<<<(NPD + 63) / 64, 64>>>(ddx, ddy, ddo, NPD); CK(cudaDeviceSynchronize());
  std::vector<double> dov(NPD * 6);
  CK(cudaMemcpy(dov.data(), ddo, NPD * 6 * 8, cudaMemcpyDeviceToHost));
  const char* dn[6] = {"dadd", "dmul", "dfma", "dmin", "dmax", "d2f2d"};
  for (int k = 0; k < 6; ++k) {
    std::vector<uint64_t> col(NPD);
    for (int p = 0; p < NPD; ++p) std::memcpy(&col[p], &dov[p * 6 + k], 8);
    char nm[32]; std::snprintf(nm, sizeof nm, "f64_%s", dn[k]);
    report(nm, col.data(), NPD * 8, 6);
  }

  if (full) {
    for (int k = 0; k < 6; ++k)
      for (int p = 0; p < NPD; ++p) {
        uint64_t xa, xb, r;
        std::memcpy(&xa, &dxv[p], 8); std::memcpy(&xb, &dyv[p], 8); std::memcpy(&r, &dov[p * 6 + k], 8);
        std::printf("D %s %016llx %016llx -> %016llx\n", dn[k], (unsigned long long)xa, (unsigned long long)xb, (unsigned long long)r);
      }
  }

  // negative zero under .sat / .ftz
  float* dz; CK(cudaMalloc(&dz, 32)); satzero<<<1, 1>>>(dz); CK(cudaDeviceSynchronize());
  float hz[8]; CK(cudaMemcpy(hz, dz, 32, cudaMemcpyDeviceToHost)); report("negzero_sat_ftz", hz, 32, 8);

  // bf16 / half HFMA2 on random bit patterns (NaNs, infinities, subnormals included)
  const int NH = 4096;
  std::vector<uint32_t> ha32(NH), hb32(NH), hc32(NH);
  for (int i = 0; i < NH; ++i) { ha32[i] = lcg() * 977u + lcg(); hb32[i] = lcg() * 131u + lcg(); hc32[i] = lcg() * 7919u + lcg(); }
  uint32_t *pa, *pb, *pc, *po; CK(cudaMalloc(&pa, NH * 4)); CK(cudaMalloc(&pb, NH * 4)); CK(cudaMalloc(&pc, NH * 4)); CK(cudaMalloc(&po, NH * 8));
  CK(cudaMemcpy(pa, ha32.data(), NH * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(pb, hb32.data(), NH * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(pc, hc32.data(), NH * 4, cudaMemcpyHostToDevice));
  hfma<<<NH / 128, 128>>>(pa, pb, pc, po, NH); CK(cudaDeviceSynchronize());
  std::vector<uint32_t> hh(NH * 2); CK(cudaMemcpy(hh.data(), po, NH * 8, cudaMemcpyDeviceToHost));
  std::vector<uint32_t> hbf(NH), hhf(NH);
  for (int i = 0; i < NH; ++i) { hbf[i] = hh[2 * i]; hhf[i] = hh[2 * i + 1]; }
  report("bf16_hfma2", hbf.data(), NH * 4, 6); report("f16_hfma2", hhf.data(), NH * 4, 6);

  // 64-bit conversions, integer saturation, __sad
  const int NI = 512;
  std::vector<double> id(NI); std::vector<float> iff(NI); std::vector<int> is(NI);
  for (int i = 0; i < NI; ++i) {
    int e = (int)(lcg() % 140) - 20;
    id[i] = ((lcg() & 1) ? -1.0 : 1.0) * std::ldexp(1.0 + (lcg() % 4096) / 4096.0, e);
    iff[i] = (float)id[i];
    is[i] = (int)(lcg() * 2654435761u);
  }
  id[0] = d_from_bits(0x7ff8000000000000ull); id[1] = d_from_bits(0x7ff0000000000000ull); id[2] = d_from_bits(0xfff0000000000000ull);
  iff[0] = f_from_bits(0x7fc00000u); iff[1] = f_from_bits(0x7f800000u); iff[2] = f_from_bits(0xff800000u);
  double* pd; float* pf; int* ps; long long* pl;
  CK(cudaMalloc(&pd, NI * 8)); CK(cudaMalloc(&pf, NI * 4)); CK(cudaMalloc(&ps, NI * 4)); CK(cudaMalloc(&pl, NI * 8 * 8));
  CK(cudaMemcpy(pd, id.data(), NI * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(pf, iff.data(), NI * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ps, is.data(), NI * 4, cudaMemcpyHostToDevice));
  ints<<<NI / 128, 128>>>(pd, pf, ps, pl, NI); CK(cudaDeviceSynchronize());
  std::vector<long long> il(NI * 8); CK(cudaMemcpy(il.data(), pl, NI * 64, cudaMemcpyDeviceToHost));
  const char* in[8] = {"d2ll", "d2ull", "f2ll", "d2ll_rn", "sad", "usad", "adds32", "cvt_sat_s64"};
  for (int k = 0; k < 8; ++k) {
    std::vector<long long> col(NI);
    for (int p = 0; p < NI; ++p) col[p] = il[p * 8 + k];
    char nm[32]; std::snprintf(nm, sizeof nm, "int_%s", in[k]);
    report(nm, col.data(), NI * 8, 4);
  }

  // launch errors: a zero-size grid, a block that is too large, and what the next synchronize reports
  nothing<<<0, 32>>>();
  cudaError_t e1 = cudaGetLastError();
  nothing<<<1, 4096>>>();
  cudaError_t e2 = cudaGetLastError();
  cudaError_t e3 = cudaDeviceSynchronize();
  nothing<<<1, 32>>>();
  cudaError_t e4 = cudaGetLastError();
  cudaError_t e5 = cudaDeviceSynchronize();
  std::printf("launch_errors %d %d %d %d %d\n", (int)e1, (int)e2, (int)e3, (int)e4, (int)e5);
  return 0;
}
