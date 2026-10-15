// The f32, f16 and f64 forms ptxas lowers to single SASS instructions with a flush, a saturation, a rounding mode, a NaN rule or a
// byte permute (FFMA.FTZ.SAT, FMNMX, MUFU, PRMT in every mode, F2FP.PACK_AB with .RELU, HADD2.F32, DADD.RP, ...), over values that
// break arithmetic -- zeros of both signs, subnormals, infinities, NaNs of both signs and payloads, the extremes -- hashed per
// form. The expected file (nvidia/tests/data/float_half_forms.rtx3060.txt) is an RTX 3060's, and BOTH of the simulator's engines
// (SASS, the default, and PTX, VGPU_SASS=0) must print exactly its lines: where the two disagreed with each other and the card,
// this program found the bug. run_float_half_forms.sh runs it on the simulator at every SIMD level, or on a real GPU with --card.
//
// What is not here: the SFU's approximations of rcp, sqrt, rsqrt, sin, cos, ex2, lg2 and tanh are not correctly rounded and differ
// from the host's libm (by one ulp for rcp, sqrt, rsqrt and ex2, by more for the others), in a way this simulator does not
// reproduce (nvidia/docs/libraries.md, "What is not implemented"). For those forms only the inputs whose answers are not
// approximations are hashed: zeros, subnormals, infinities and NaNs, and the negative numbers whose root or logarithm is a NaN.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

constexpr int kFloatForms = 41;
__device__ __forceinline__ uint32_t fbits_(float f) { return __float_as_uint(f); }
__global__ void float_forms(const float* in, uint32_t* out, int n, int shape) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = in[i], b = in[(i + 1) % n], c = in[(i + 5) % n];
  uint32_t o[kFloatForms] = {};
  const bool take = shape == 0 ? true : shape == 1 ? (__float_as_uint(b) & 3) != 0 : (i & 4) == 0;
  if (take) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); o[0] = fbits_(r);
    asm("fma.rn.ftz.sat.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); o[1] = fbits_(r);
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[2] = fbits_(r);
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[3] = fbits_(r);
    asm("mul.rn.ftz.sat.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[4] = fbits_(r);
    asm("add.rn.sat.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[5] = fbits_(r);
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(-a), "f"(fabsf(b)), "f"(-fabsf(c))); o[6] = fbits_(r);
    asm("min.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[7] = fbits_(r);
    asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[8] = fbits_(r);
    asm("min.ftz.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[9] = fbits_(r);
    asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(-b)); o[10] = fbits_(r);
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); o[11] = fbits_(r);
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); o[12] = fbits_(r);
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); o[13] = fbits_(r);
    asm("sin.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); o[14] = fbits_(r);
    asm("cos.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(fabsf(a))); o[15] = fbits_(r);
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); o[16] = fbits_(r);
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); o[17] = fbits_(r);
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); o[18] = fbits_(r);
    int si; unsigned ui;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(si) : "f"(a)); o[19] = static_cast<uint32_t>(si);
    asm("cvt.rzi.u32.f32 %0, %1;" : "=r"(ui) : "f"(a)); o[20] = ui;
    asm("cvt.rmi.ftz.s32.f32 %0, %1;" : "=r"(si) : "f"(a)); o[21] = static_cast<uint32_t>(si);
    asm("cvt.rpi.u32.f32 %0, %1;" : "=r"(ui) : "f"(-b)); o[22] = ui;
    const int ia = static_cast<int>(__float_as_uint(a)), ib = static_cast<int>(__float_as_uint(b));
    asm("cvt.rn.f32.s32 %0, %1;" : "=f"(r) : "r"(ia)); o[23] = fbits_(r);
    asm("cvt.rn.f32.u32 %0, %1;" : "=f"(r) : "r"(ib)); o[24] = fbits_(r);
    // The compares, each a predicate turned into a bit.
    uint32_t m = 0, q;
#define CMP(OP, BIT, FTZ) \
    asm("{ .reg .pred p; setp." OP FTZ ".f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(q) : "f"(a), "f"(b)); m |= q << BIT;
    CMP("lt", 0, "") CMP("le", 1, ".ftz") CMP("eq", 2, "") CMP("ne", 3, ".ftz") CMP("gt", 4, "") CMP("ge", 5, ".ftz")
    CMP("num", 6, "") CMP("nan", 7, ".ftz") CMP("ltu", 8, "") CMP("leu", 9, ".ftz") CMP("equ", 10, "") CMP("neu", 11, ".ftz")
    CMP("gtu", 12, "") CMP("geu", 13, ".ftz")
#undef CMP
    o[25] = m;
    o[26] = static_cast<uint32_t>(a > b ? 1 : 0) | (static_cast<uint32_t>(fabsf(a) < fabsf(b) ? 2 : 0));
    o[27] = fbits_(fminf(a, b)) ^ fbits_(fmaxf(b, c));
    // Directed rounding, with a flush and a saturation, and the narrow integers to f32 (I2F.U8, .S8, .U16, .S16 with a byte selected).
    asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); o[28] = fbits_(r);
    asm("mul.rm.ftz.sat.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[29] = fbits_(r);
    asm("add.rp.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); o[30] = fbits_(r);
    asm("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(-a), "f"(b), "f"(c)); o[31] = fbits_(r);
    o[32] = fbits_(static_cast<float>(static_cast<unsigned char>(ia >> 8)));
    o[33] = fbits_(static_cast<float>(static_cast<signed char>(ia >> 16)));
    o[34] = fbits_(static_cast<float>(static_cast<unsigned short>(ib >> 16)));
    o[35] = fbits_(static_cast<float>(static_cast<short>(ib)));
    // The half of a register as the source of a byte or half conversion (I2F's .H1 and .B1 selects), and MUFU with a source modifier.
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f32.u8 %0, hi; }" : "=f"(r) : "r"(ia)); o[36] = fbits_(r);
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f32.s16 %0, hi; }" : "=f"(r) : "r"(ib)); o[37] = fbits_(r);
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(-a)); o[38] = fbits_(r);
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(fabsf(-b))); o[39] = fbits_(r);
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(-fabsf(c))); o[40] = fbits_(r);
  }
  for (int k = 0; k < kFloatForms; ++k) out[static_cast<size_t>(i) * kFloatForms + k] = o[k];
}


// Packed halves (HADD2, HMUL2, HFMA2 with swizzled, negated and immediate operands), the byte permutes in every mode (PRMT), and
// double arithmetic under every rounding mode (DADD, DMUL, DFMA with .RM, .RP, .RZ). Only the two runs' agreement is checked
// here (the kernels have host references in tests/unit/test_simd.cpp).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
constexpr int kHalfForms = 34;
__global__ void half_forms(const uint32_t* in, uint32_t* out, const uint64_t* din, uint64_t* dout, int n, int shape) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t x = in[i], y = in[(i + 3) % n], z = in[(i + 11) % n];
  uint32_t o[kHalfForms] = {};
  const bool take = shape == 0 ? true : shape == 1 ? (y & 3) != 0 : (i & 2) == 0;
  uint64_t d[8] = {};
  if (take) {
    const __half2 a = *reinterpret_cast<const __half2*>(&x), b = *reinterpret_cast<const __half2*>(&y), c = *reinterpret_cast<const __half2*>(&z);
    const auto bits = [](__half2 h) { return *reinterpret_cast<const uint32_t*>(&h); };
    o[0] = bits(__hfma2(a, b, c));
    o[1] = bits(__hadd2(a, b));
    o[2] = bits(__hmul2(a, b));
    o[3] = bits(__hfma2(__hneg2(a), b, __habs2(c)));
    o[4] = bits(__hfma2(__low2half2(a), b, c));
    o[5] = bits(__hfma2(a, __high2half2(b), __low2half2(c)));
    o[6] = bits(__hfma2(a, __float2half2_rn(2.0f), c));
    o[7] = bits(__hfma2(a, b, __halves2half2(__float2half(1.5f), __float2half(-0.25f))));
    o[8] = bits(__hmul2(a, __halves2half2(__float2half(3.0f), __float2half(0.125f))));
    o[9] = bits(__hadd2(__high2half2(a), __habs2(b)));
    o[10] = bits(__hfma2(__halves2half2(__high2half(a), __low2half(a)), b, c));
    // PRMT in each mode.
    o[11] = __byte_perm(x, y, z);
    asm("prmt.b32.f4e %0, %1, %2, %3;" : "=r"(o[12]) : "r"(x), "r"(y), "r"(z));
    asm("prmt.b32.b4e %0, %1, %2, %3;" : "=r"(o[13]) : "r"(x), "r"(y), "r"(z));
    asm("prmt.b32.rc8 %0, %1, %2, %3;" : "=r"(o[14]) : "r"(x), "r"(y), "r"(z));
    asm("prmt.b32.ecl %0, %1, %2, %3;" : "=r"(o[15]) : "r"(x), "r"(y), "r"(z));
    asm("prmt.b32.ecr %0, %1, %2, %3;" : "=r"(o[16]) : "r"(x), "r"(y), "r"(z));
    asm("prmt.b32.rc16 %0, %1, %2, %3;" : "=r"(o[17]) : "r"(x), "r"(y), "r"(z));
    o[18] = __byte_perm(x, 0x01020304u, 0x7531u);
    o[19] = __byte_perm(x, y, 0x8888u ^ (z & 0x7777u));
    // Halves to f32 (HADD2.F32 with a swizzle), and f32 pairs packed to f16x2 and bf16x2 (F2FP.PACK_AB), with a negation and .RELU.
    const float f0 = __half2float(__low2half(a)), f1 = __half2float(__high2half(a));
    o[24] = __float_as_uint(f0) ^ __float_as_uint(f1 * 2.0f);
    o[25] = __float_as_uint(__half2float(__high2half(b)) + __half2float(__low2half(c)));
    o[26] = bits(__float22half2_rn(make_float2(f0 * 1.5f, -f1)));
    o[27] = bits(__float22half2_rn(make_float2(__uint_as_float(x), __uint_as_float(y))));
    o[28] = *reinterpret_cast<const uint32_t*>(&(const __nv_bfloat162&)__float22bfloat162_rn(make_float2(__uint_as_float(y), f0)));
    o[29] = bits(__float22half2_rn(make_float2(fmaxf(__uint_as_float(z), 0.0f), fabsf(__uint_as_float(x)))));
    o[30] = __float_as_uint(__half2float(__low2half(__hmul2(a, b))) * __half2float(__high2half(c)));
    o[31] = bits(__hfma2(a, b, c)) ^ bits(__hadd2(__hneg2(a), c));
    asm("cvt.rn.relu.f16x2.f32 %0, %1, %2;" : "=r"(o[32]) : "f"(__uint_as_float(x)), "f"(-f1));
    asm("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(o[33]) : "f"(f0), "f"(__uint_as_float(y)));
    // The 32-bit multiply-add with carries.
    const unsigned long long wide = (static_cast<unsigned long long>(x) << 32 | y) * 0x9E3779B1ull + z;
    o[20] = static_cast<uint32_t>(wide >> 32) ^ static_cast<uint32_t>(wide);
    o[21] = x * y + z + (x < y);
    const double p = __longlong_as_double(static_cast<long long>(din[i])), q = __longlong_as_double(static_cast<long long>(din[(i + 1) % n]));
    const double r = __longlong_as_double(static_cast<long long>(din[(i + 5) % n]));
    d[0] = static_cast<uint64_t>(__double_as_longlong(__dadd_rd(p, q)));
    d[1] = static_cast<uint64_t>(__double_as_longlong(__dadd_ru(p, -q)));
    d[2] = static_cast<uint64_t>(__double_as_longlong(__dmul_rz(p, q)));
    d[3] = static_cast<uint64_t>(__double_as_longlong(__dmul_rd(-p, fabs(q))));
    d[4] = static_cast<uint64_t>(__double_as_longlong(__fma_rz(p, q, r)));
    d[5] = static_cast<uint64_t>(__double_as_longlong(__fma_ru(p, q, -r)));
    d[6] = static_cast<uint64_t>(__double_as_longlong(__fma_rd(p, q, r)));
    d[7] = static_cast<uint64_t>(__double_as_longlong(__fma_rn(p, q, r)));
    // DSETP: every comparison, and fmin / fmax (its MIN and MAX forms), with |x| and -x.
    o[22] = static_cast<uint32_t>(p < q) | static_cast<uint32_t>(p <= q) << 1 | static_cast<uint32_t>(p == q) << 2 | static_cast<uint32_t>(p != q) << 3 |
            static_cast<uint32_t>(p > q) << 4 | static_cast<uint32_t>(p >= q) << 5 | static_cast<uint32_t>(!(p < q)) << 6 |
            static_cast<uint32_t>(!(p <= q)) << 7 | static_cast<uint32_t>(!(p == q)) << 8 | static_cast<uint32_t>(!(p > q)) << 9 |
            static_cast<uint32_t>(!(p >= q)) << 10 | static_cast<uint32_t>(isnan(p)) << 11 | static_cast<uint32_t>(!isnan(q)) << 12 |
            static_cast<uint32_t>(fabs(p) < -q) << 13 | static_cast<uint32_t>(-p >= fabs(q)) << 14 | static_cast<uint32_t>(isnan(p) && isnan(q)) << 15 |
            static_cast<uint32_t>(isnan(p + q)) << 16 | static_cast<uint32_t>((p < q) != (r > q)) << 17;
    {
      const long long mn = __double_as_longlong(fmin(p, q)), mx = __double_as_longlong(fmax(-p, fabs(q)));
      o[23] = static_cast<uint32_t>(mn) ^ static_cast<uint32_t>(mn >> 32) ^ (static_cast<uint32_t>(mx) * 3u) ^ static_cast<uint32_t>(mx >> 32);
    }
  }
  for (int k = 0; k < kHalfForms; ++k) out[static_cast<size_t>(i) * kHalfForms + k] = o[k];
  for (int k = 0; k < 8; ++k) dout[static_cast<size_t>(i) * 8 + k] = d[k];
}


// tf32 tensor core operands that carry bits below the format's 10-bit mantissa (no __float_to_tf32 first): the tensor core reads
// only the top 19 bits of each, so the products and sums are those of the truncated numbers. The values are multiples of 1/4 with
// garbage in the low 13 bits, so every sum of truncated products is exact and its order cannot matter; the PTX engine used to
// round the operands when it loaded them (and an RTX 3060, like the SASS executor, truncates).
__global__ void k_wmma_tf32(const float* a, const float* b, const float* c, float* d) {
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> fc;
  const int t = blockIdx.x;
  wmma::load_matrix_sync(fa, a + t * 128, 8);
  wmma::load_matrix_sync(fb, b + t * 128, 8);
  wmma::load_matrix_sync(fc, c + t * 256, 16, wmma::mem_row_major);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(d + t * 256, fc, 16, wmma::mem_row_major);
}
static uint32_t mix32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }

static uint64_t fnv_step(uint64_t h, uint64_t v, int bytes) {
  for (int i = 0; i < bytes; ++i) h = (h ^ ((v >> (8 * i)) & 0xff)) * 1099511628211ull;
  return h;
}

// Whether the f32 form `k`'s answer for a lane with this source is exact enough to compare: every form but the SFU's, and for those
// the inputs that are not an ordinary positive number.
static bool exact_lane(int k, uint32_t src) {
  const bool sfu = (k >= 11 && k <= 18) || k == 38 || k == 39 || k == 40;
  if (!sfu) return true;
  const uint32_t e = (src >> 23) & 0xff;
  if (e == 0 || e == 0xff) return true;   // zero, subnormal, infinity, NaN
  if (k == 40) return true;               // the root of -|c|: a NaN for every nonzero c
  if (src >> 31) return k == 12 || k == 13 || k == 17 || k == 39;   // negative: sqrt, rsqrt and lg2 give a NaN
  return false;
}
static uint32_t src_of(int k, uint32_t a, uint32_t b, uint32_t c) { return k == 39 ? b : k == 40 ? c : a; }

int main() {
  const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
                         0x3f800000u, 0xbf800000u, 0x3f000000u, 0x40000000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
                         0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0x7f800001u, 0xffbfffffu, 0x3effffffu, 0x4b000000u, 0x4b800001u,
                         0x33800000u, 0x3f800001u, 0x4f000000u, 0xcf000000u, 0x4f800000u, 0x4effffffu, 0x3f7fffffu, 0x40490fdbu};
  const int ns = sizeof sp / sizeof sp[0];
  {
    const int n = 32 * 11 + 7;
    std::vector<float> h(n);
    uint64_t s = 0x4F1BBCDCBFA53E0Aull;
    for (int i = 0; i < n; ++i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      const uint32_t bits = i % 3 == 0 ? static_cast<uint32_t>(s) : sp[s % ns];
      std::memcpy(&h[i], &bits, 4);
    }
    float* din; uint32_t* dout;
    CK(cudaMalloc(&din, n * 4));
    CK(cudaMalloc(&dout, static_cast<size_t>(n) * kFloatForms * 4));
    CK(cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0, static_cast<size_t>(n) * kFloatForms * 4));
    float_forms<<<(n + 63) / 64, 64>>>(din, dout, n, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(static_cast<size_t>(n) * kFloatForms);
    CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<uint32_t> in(n);
    std::memcpy(in.data(), h.data(), n * 4);
    for (int k = 0; k < kFloatForms; ++k) {
      uint64_t hh = 1469598103934665603ull;
      int used = 0;
      for (int i = 0; i < n; ++i) {
        if (!exact_lane(k, src_of(k, in[i], in[(i + 1) % n], in[(i + 5) % n]))) continue;
        hh = fnv_step(hh, got[static_cast<size_t>(i) * kFloatForms + k], 4);
        ++used;
      }
      std::printf("float_form_%02d %3d lanes %016llx\n", k, used, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(din)); CK(cudaFree(dout));
  }
  {
    const int n = 32 * 9 + 5;
    std::vector<uint32_t> h(n);
    std::vector<uint64_t> hd(n);
    uint64_t s = 0x1234567887654321ull;
    const uint16_t halves[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7c01, 0x4000, 0x3555, 0x5640};
    const uint64_t dbl[] = {0x0000000000000000ull, 0x8000000000000000ull, 0x0000000000000001ull, 0x000fffffffffffffull, 0x0010000000000000ull,
                            0x3ff0000000000000ull, 0xbff0000000000000ull, 0x3ff0000000000001ull, 0x7fefffffffffffffull, 0x7ff0000000000000ull,
                            0xfff0000000000000ull, 0x7ff8000000000000ull, 0x4000000000000000ull, 0x3fb999999999999aull, 0x3ca0000000000000ull};
    for (int i = 0; i < n; ++i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      h[i] = i % 3 == 0 ? static_cast<uint32_t>(s) : static_cast<uint32_t>(halves[s % 18]) | static_cast<uint32_t>(halves[(s >> 8) % 18]) << 16;
      hd[i] = i % 3 == 0 ? s : dbl[s % 15];
    }
    uint32_t *din, *dout; uint64_t *ddin, *ddout;
    CK(cudaMalloc(&din, n * 4)); CK(cudaMalloc(&dout, static_cast<size_t>(n) * kHalfForms * 4));
    CK(cudaMalloc(&ddin, n * 8)); CK(cudaMalloc(&ddout, static_cast<size_t>(n) * 8 * 8));
    CK(cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddin, hd.data(), n * 8, cudaMemcpyHostToDevice));
    half_forms<<<(n + 63) / 64, 64>>>(din, dout, ddin, ddout, n, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(static_cast<size_t>(n) * kHalfForms);
    std::vector<uint64_t> gotd(static_cast<size_t>(n) * 8);
    CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gotd.data(), ddout, gotd.size() * 8, cudaMemcpyDeviceToHost));
    for (int k = 0; k < kHalfForms; ++k) {
      uint64_t hh = 1469598103934665603ull;
      for (int i = 0; i < n; ++i) hh = fnv_step(hh, got[static_cast<size_t>(i) * kHalfForms + k], 4);
      std::printf("half_form_%02d %016llx\n", k, static_cast<unsigned long long>(hh));
    }
    for (int k = 0; k < 8; ++k) {
      uint64_t hh = 1469598103934665603ull;
      for (int i = 0; i < n; ++i) hh = fnv_step(hh, gotd[static_cast<size_t>(i) * 8 + k], 8);
      std::printf("double_form_%d %016llx\n", k, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(din)); CK(cudaFree(dout)); CK(cudaFree(ddin)); CK(cudaFree(ddout));
  }
  {
    const int tiles = 16;
    std::vector<float> a(tiles * 128), b(tiles * 128), c(tiles * 256), d(tiles * 256);
    const auto mk = [](uint32_t seed, int i) {
      const uint32_t h = mix32(seed * 0x9E3779B9u + i);
      float v = static_cast<float>(static_cast<int>(h % 17) - 8) * 0.25f;
      uint32_t bits;
      std::memcpy(&bits, &v, 4);
      if (v != 0.0f) bits |= mix32(h) & 0x1FFFu;
      std::memcpy(&v, &bits, 4);
      return v;
    };
    for (size_t i = 0; i < a.size(); ++i) { a[i] = mk(1, static_cast<int>(i)); b[i] = mk(2, static_cast<int>(i)); }
    for (size_t i = 0; i < c.size(); ++i) c[i] = static_cast<float>(static_cast<int>(mix32(static_cast<uint32_t>(i) + 77) % 9) - 4) * 0.5f;
    float *da, *db, *dc, *dd;
    CK(cudaMalloc(&da, a.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dc, c.size() * 4)); CK(cudaMalloc(&dd, d.size() * 4));
    CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, c.data(), c.size() * 4, cudaMemcpyHostToDevice));
    k_wmma_tf32<<<tiles, 32>>>(da, db, dc, dd);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(d.data(), dd, d.size() * 4, cudaMemcpyDeviceToHost));
    uint64_t hh = 1469598103934665603ull;
    for (float f : d) { uint32_t u; std::memcpy(&u, &f, 4); hh = fnv_step(hh, u, 4); }
    std::printf("wmma_tf32_truncated %016llx\n", static_cast<unsigned long long>(hh));
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc)); CK(cudaFree(dd));
  }
  return 0;
}
