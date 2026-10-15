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
static __host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }

// Double arithmetic on NaNs with distinct signs and payloads (the answer is a NaN that is one of the operands', quieted: b's if b is
// a NaN, else a's, and for an fma b's, else c's, else a's -- an RTX 3060; the instruction's |x| and -x change the arithmetic, not a
// NaN that is the answer), and the conversions of NaNs. Hashed per operation over every (a, b, c) of a table. The lines named
// sassonly_ use an operand modifier ptxas folds into the instruction (-a, |a|, -c): the PTX engine runs its negation and absolute
// value as separate instructions, which flip or clear a NaN's sign, so only the SASS engine can match the card there.
constexpr int kNV = 14;
__constant__ unsigned long long kDV[kNV];
__global__ void k_nan_dbl(unsigned long long* out, int op) {
  const int t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= kNV * kNV) return;
  const double a = __longlong_as_double(static_cast<long long>(kDV[t / kNV])), b = __longlong_as_double(static_cast<long long>(kDV[t % kNV])),
               c = __longlong_as_double(static_cast<long long>(kDV[(t * 5 + 3) % kNV]));
  double r = 0;
  switch (op) {
    case 0: r = __dadd_rn(a, b); break;
    case 1: r = __dmul_rn(a, b); break;
    case 2: r = __fma_rn(a, b, c); break;
    case 3: r = __dadd_ru(a, b); break;
    case 4: r = __dmul_rd(a, b); break;
    case 5: r = __fma_rz(a, b, c); break;
    case 6: r = fmin(a, b); break;
    case 7: r = fmax(a, b); break;
    case 8: r = __dadd_rn(-a, b); break;
    case 9: r = __dmul_rn(a, -b); break;
    case 10: r = __fma_rn(a, b, -c); break;
    case 11: r = fabs(a) + b; break;
    case 12: r = -fabs(a); break;
    case 13: r = __dadd_rn(fabs(a), -fabs(b)); break;
  }
  out[t] = static_cast<unsigned long long>(__double_as_longlong(r));
}
constexpr int kNF = 12;
__constant__ uint32_t kFV[kNF];
__global__ void k_nan_cvt(uint32_t* out, int op) {
  const int t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= kNF) return;
  const float a = __uint_as_float(kFV[t]);
  uint32_t r = 0;
  switch (op) {
    case 0: r = static_cast<uint32_t>(__double_as_longlong(static_cast<double>(a)) >> 32); break;
    case 1: r = __float_as_uint(static_cast<float>(__longlong_as_double((static_cast<long long>(kFV[t]) << 32) | kFV[(t + 3) % kNF]))); break;
    case 2: { __half h = __float2half_rn(a); r = *reinterpret_cast<unsigned short*>(&h); break; }
    case 3: { __nv_bfloat16 h = __float2bfloat16_rn(a); r = *reinterpret_cast<unsigned short*>(&h); break; }
    case 4: r = __float_as_uint(__half2float(__ushort_as_half(static_cast<unsigned short>(kFV[t])))); break;
    case 5: r = __float_as_uint(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(kFV[t])))); break;
    case 6: r = __float_as_uint(sqrtf(a)); break;
  }
  out[t] = r;
}

// __sad and __usad: |a - b| + c as signed and as unsigned 32-bit numbers (one SASS VABSDIFF each, with a register, an immediate b or
// an immediate c), over the extremes and random words. The difference is taken at full width and the sum wraps.
__global__ void k_sad(const uint32_t* in, uint32_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
  out[6 * i + 0] = static_cast<uint32_t>(__sad(static_cast<int>(a), static_cast<int>(b), c));
  out[6 * i + 1] = __usad(a, b, c);
  out[6 * i + 2] = __usad(a, 12345u, 7u);
  out[6 * i + 3] = static_cast<uint32_t>(__sad(static_cast<int>(a), -77, b));
  out[6 * i + 4] = static_cast<uint32_t>(__sad(static_cast<int>(a), static_cast<int>(b), 0x80000000u));
  out[6 * i + 5] = __usad(a, 0xffffffffu, c);
}

// mma.sync of f16, bf16 and tf32 operands with random registers, summed by the tensor core (include/vgpu/exec/tc_ampere.hpp in the
// simulator): style 0 finite values over wide exponent ranges (a sequential f32 sum differs from the card's chunked, truncating one in
// the low bits of most results), style 1 uniformly random bits (infinities, NaNs, subnormals, products past the f32 range). Each
// form's hash is over every lane's result registers.
__device__ __forceinline__ uint32_t rw(int style, char kind, uint32_t seed, uint32_t lane, uint32_t idx) {
  uint32_t h = mix32(seed * 0x9E3779B9u ^ mix32(lane * 131u + idx * 7919u + 17u));
  if (style == 1) return h;
  if (kind == 'h') {
    uint32_t g = mix32(h + 1), w = 0;
    for (int e = 0; e < 2; ++e) {
      const uint32_t x = e ? g : h;
      w |= (((x & 0x8000u) | (((x >> 16) % 30 + 1) << 10) | ((x >> 1) & 0x3ffu))) << (16 * e);
    }
    return w;
  }
  if (kind == 'b') {
    uint32_t g = mix32(h + 1), w = 0;
    for (int e = 0; e < 2; ++e) {
      const uint32_t x = e ? g : h;
      w |= (((x & 0x8000u) | (((x >> 16) % 250 + 3) << 7) | ((x >> 1) & 0x7fu))) << (16 * e);
    }
    return w;
  }
  return (h & 0x807fffffu) | (((h >> 8) % 240 + 8) << 23);   // 't': any float, 8-bit exponent, low mantissa bits as they are
}
__device__ __forceinline__ uint32_t rc(int style, uint32_t seed, uint32_t lane, uint32_t idx) {
  uint32_t h = mix32(seed * 0x85EBCA6Bu ^ mix32(lane * 977u + idx * 31u + 5u));
  if (style == 1) return h;
  return (h & 0x807fffffu) | (((h >> 8) % 120 + 60) << 23);
}
#define MMA_FORM(NAME, SHAPE, ATYPE, KIND, NA, NB, NC, ACC16)                                                                              \
  __global__ void NAME(int style, uint32_t seed, uint32_t* out) {                                                                          \
    const uint32_t lane = threadIdx.x;                                                                                                     \
    uint32_t a[NA], b[NB], c[NC], d[NC];                                                                                                   \
    for (int i = 0; i < NA; ++i) a[i] = rw(style, KIND, seed, lane, i);                                                                    \
    for (int i = 0; i < NB; ++i) b[i] = rw(style, KIND, seed + 7, lane, i);                                                                \
    for (int i = 0; i < NC; ++i) c[i] = ACC16 ? rw(style, 'h', seed + 13, lane, i) : rc(style, seed + 13, lane, i);                        \
    MMA_ASM_##NAME;                                                                                                                        \
    for (int i = 0; i < NC; ++i) out[lane * NC + i] = d[i];                                                                                \
  }
#define MMA_ASM_k_mma_f16_k16 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]))
#define MMA_ASM_k_mma_f16_k8 asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]))
#define MMA_ASM_k_mma_f16acc_k16 asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};" : "=r"(d[0]), "=r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]))
#define MMA_ASM_k_mma_bf16_k16 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]))
#define MMA_ASM_k_mma_bf16_k8 asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]))
#define MMA_ASM_k_mma_tf32_k8 asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]))
#define MMA_ASM_k_mma_tf32_k4 asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]))
MMA_FORM(k_mma_f16_k16, 0, 0, 'h', 4, 2, 4, false)
MMA_FORM(k_mma_f16_k8, 0, 0, 'h', 2, 1, 4, false)
MMA_FORM(k_mma_f16acc_k16, 0, 0, 'h', 4, 2, 2, true)
MMA_FORM(k_mma_bf16_k16, 0, 0, 'b', 4, 2, 4, false)
MMA_FORM(k_mma_bf16_k8, 0, 0, 'b', 2, 1, 4, false)
MMA_FORM(k_mma_tf32_k8, 0, 0, 't', 4, 2, 4, false)
MMA_FORM(k_mma_tf32_k4, 0, 0, 't', 2, 1, 4, false)

// Hand-picked cases of the tensor core's rule (signs of zero, cancellation, subnormal and overflowing terms, infinities, NaNs, terms
// below the alignment grid), one m16n8k8 f16 mma each: only element (0,0) is read. A row 0 = a[0..7], B column 0 = b[0..7], C(0,0) = c.
__global__ void k_case(const uint16_t* a, const uint16_t* b, const float* c, float* d, int n) {
  const int lane = threadIdx.x, g = lane >> 2, t4 = lane & 3;
  const int cs = blockIdx.x;
  uint32_t ra0 = 0, ra1 = 0, rb0 = 0;
  const uint16_t* ca = a + cs * 8; const uint16_t* cb = b + cs * 8;
  if (g == 0) { ra0 = ca[2 * t4] | (uint32_t)ca[2 * t4 + 1] << 16; }
  if (g == 0) { rb0 = cb[2 * t4] | (uint32_t)cb[2 * t4 + 1] << 16; }   // B col 0 -> lane group g = 0 holds column 0
  float c0 = (g == 0 && t4 == 0) ? c[cs] : 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  float d0, d1, d2, d3;
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(ra0), "r"(ra1), "r"(rb0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  if (g == 0 && t4 == 0) d[cs] = d0;
}
struct EdgeCase { const char* name; uint16_t a[8], b[8]; uint32_t c; };

static uint64_t fnv_step(uint64_t h, uint64_t v, int bytes) {
  for (int i = 0; i < bytes; ++i) h = (h ^ ((v >> (8 * i)) & 0xff)) * 1099511628211ull;
  return h;
}

// Whether the f32 form `k`'s answer for a lane is exact enough to compare: every form but the SFU's, and for those the inputs that are
// not an ordinary number the function approximates -- the source as the instruction sees it (-a for form 38, |b| for 39, -|c| for 40):
// a zero, a subnormal, an infinity or a NaN, and a negative number whose root or logarithm is a NaN.
static bool exact_lane(int k, uint32_t a, uint32_t b, uint32_t c) {
  const bool sfu = (k >= 11 && k <= 18) || k == 38 || k == 39 || k == 40;
  if (!sfu) return true;
  const uint32_t src = k == 38 ? (a ^ 0x80000000u) : k == 39 ? (b & 0x7fffffffu) : k == 40 ? (c | 0x80000000u) : a;
  const uint32_t e = (src >> 23) & 0xff;
  if (e == 0 || e == 0xff) return true;   // zero, subnormal, infinity, NaN
  if (src >> 31) return k == 12 || k == 13 || k == 17 || k == 40;   // negative: sqrt, rsqrt and lg2 give a NaN
  return false;
}

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
        if (!exact_lane(k, in[i], in[(i + 1) % n], in[(i + 5) % n])) continue;
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
  {
    const unsigned long long dv[kNV] = {0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff8000000000123ull, 0xfff8000000000456ull,
                                        0x7ff0000000000001ull, 0xfff0000000000002ull, 0x3ff0000000000000ull, 0xc000000000000000ull,
                                        0x0ull, 0x8000000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
                                        0x7ff80000deadbeefull, 0x3ca0000000000000ull};
    const uint32_t fv[kNF] = {0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0xffc54321u, 0x7f800001u, 0xff800002u,
                              0x3f800000u, 0xc0000000u, 0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u};
    CK(cudaMemcpyToSymbol(kDV, dv, sizeof dv));
    CK(cudaMemcpyToSymbol(kFV, fv, sizeof fv));
    unsigned long long* d; uint32_t* df;
    CK(cudaMalloc(&d, kNV * kNV * 8)); CK(cudaMalloc(&df, kNF * 4));
    static const char* dn[] = {"dadd", "dmul", "dfma", "dadd_ru", "dmul_rd", "dfma_rz", "fmin", "fmax", "sassonly_dadd_neg_a", "sassonly_dmul_neg_b",
                               "sassonly_dfma_neg_c", "sassonly_fabs_a_plus_b", "sassonly_neg_fabs_a", "sassonly_abs_a_minus_abs_b"};
    for (int op = 0; op < 14; ++op) {
      k_nan_dbl<<<2, 128>>>(d, op);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      unsigned long long h[kNV * kNV];
      CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
      uint64_t hh = 1469598103934665603ull;
      for (int t = 0; t < kNV * kNV; ++t) hh = fnv_step(hh, h[t], 8);
      std::printf("nan_%s %016llx\n", dn[op], static_cast<unsigned long long>(hh));
    }
    static const char* cn[] = {"f32_to_f64_hi", "f64_to_f32", "f32_to_f16", "f32_to_bf16", "f16_to_f32", "bf16_to_f32", "sqrtf"};
    for (int op = 0; op < 7; ++op) {
      k_nan_cvt<<<1, 32>>>(df, op);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      uint32_t h[kNF];
      CK(cudaMemcpy(h, df, sizeof h, cudaMemcpyDeviceToHost));
      uint64_t hh = 1469598103934665603ull;
      for (int t = 0; t < kNF; ++t) hh = fnv_step(hh, h[t], 4);
      std::printf("nan_cvt_%s %016llx\n", cn[op], static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(d)); CK(cudaFree(df));
  }
  {
    const int n = 4096;
    std::vector<uint32_t> in(3 * n), out(6 * n);
    const uint32_t special[] = {0, 1, 0x7fffffffu, 0x80000000u, 0xffffffffu, 0x7ffffffbu, 0x80000005u, 100, 0xffffff00u, 0x0000ffffu, 0x00010000u};
    uint32_t seed = 12345;
    for (int i = 0; i < 3 * n; ++i) { seed = seed * 1664525u + 1013904223u; in[i] = (i % 2) ? special[(seed >> 8) % 11] : seed; }
    uint32_t *din, *dout;
    CK(cudaMalloc(&din, in.size() * 4)); CK(cudaMalloc(&dout, out.size() * 4));
    CK(cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    k_sad<<<n / 64, 64>>>(din, dout, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
    static const char* sn[] = {"sad_s32", "usad", "usad_imm_b_c", "sad_imm_b", "sad_imm_c", "usad_imm_b"};
    for (int f = 0; f < 6; ++f) {
      uint64_t hh = 1469598103934665603ull;
      for (int i = 0; i < n; ++i) hh = fnv_step(hh, out[6 * i + f], 4);
      std::printf("%s %016llx\n", sn[f], static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(din)); CK(cudaFree(dout));
  }
  {
    uint32_t* d;
    CK(cudaMalloc(&d, 32 * 4 * 4));
    uint32_t h[128];
    struct Form { const char* name; void (*fn)(int, uint32_t, uint32_t*); int words; };
    const Form forms[] = {{"mma_f16_k16", k_mma_f16_k16, 128}, {"mma_f16_k8", k_mma_f16_k8, 128}, {"mma_f16acc_k16", k_mma_f16acc_k16, 64},
                          {"mma_bf16_k16", k_mma_bf16_k16, 128}, {"mma_bf16_k8", k_mma_bf16_k8, 128}, {"mma_tf32_k8", k_mma_tf32_k8, 128},
                          {"mma_tf32_k4", k_mma_tf32_k4, 128}};
    for (const Form& f : forms)
      for (int style = 0; style < 2; ++style) {
        uint64_t hh = 1469598103934665603ull;
        for (uint32_t seed = 1; seed <= 24; ++seed) {
          f.fn<<<1, 32>>>(style, seed, d);
          CK(cudaGetLastError());
          CK(cudaDeviceSynchronize());
          CK(cudaMemcpy(h, d, f.words * 4, cudaMemcpyDeviceToHost));
          for (int i = 0; i < f.words; ++i) hh = fnv_step(hh, h[i], 4);
        }
        std::printf("%s style%d %016llx\n", f.name, style, static_cast<unsigned long long>(hh));
      }
    CK(cudaFree(d));
  }
  {
    std::vector<EdgeCase> cs;
    auto add = [&](const char* nm, std::initializer_list<uint16_t> a, std::initializer_list<uint16_t> b, uint32_t c) {
      EdgeCase k{}; k.name = nm; int i = 0; for (auto x : a) k.a[i++] = x; i = 0; for (auto x : b) k.b[i++] = x; k.c = c; cs.push_back(k); };
    const uint16_t one = 0x3c00, mone = 0xbc00, z = 0, mz = 0x8000, big = 0x7bff, sub = 0x0001, inf = 0x7c00, minf = 0xfc00, nan = 0x7e00;
    const uint32_t F0 = 0, FM0 = 0x80000000u, F1 = 0x3f800000u, FM1 = 0xbf800000u, FMAX = 0x7f7fffffu, FNMAX = 0xff7fffffu, FSUB = 1u, FINF = 0x7f800000u, FNINF = 0xff800000u;
    add("zeros +0", {z,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, F0);
    add("zeros c=-0", {z,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, FM0);
    add("neg zero products c=-0", {mz,mz,mz,mz,mz,mz,mz,mz}, {one,one,one,one,one,one,one,one}, FM0);
    add("neg zero products c=+0", {mz,mz,mz,mz,mz,mz,mz,mz}, {one,one,one,one,one,one,one,one}, F0);
    add("neg zero products one pos c=-0", {mz,mz,mz,mz,mz,mz,mz,z}, {one,one,one,one,one,one,one,one}, FM0);
    add("cancel +1-1 c=0", {one,mone,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, F0);
    add("cancel +1-1 c=-0", {one,mone,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, FM0);
    add("cancel c=1, -1", {mone,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, F1);
    add("cancel c=-1, +1", {one,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, FM1);
    add("overflow c=max + big*big", {big,z,z,z,z,z,z,z}, {big,one,one,one,one,one,one,one}, FMAX);
    add("overflow c=max + 1*1", {one,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, FMAX);
    add("overflow c=-max - big*big", {mone,z,z,z,z,z,z,z}, {big,one,one,one,one,one,one,one}, FNMAX);
    add("overflow c=max + 2e30-ish", {big,big,big,big,big,big,big,big}, {big,big,big,big,big,big,big,big}, FMAX);
    add("tiny c=2^-149 + 0", {z,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, FSUB);
    add("tiny c=2^-149 + sub*sub", {sub,z,z,z,z,z,z,z}, {sub,one,one,one,one,one,one,one}, FSUB);
    add("sub*sub only", {sub,z,z,z,z,z,z,z}, {sub,one,one,one,one,one,one,one}, F0);
    add("sub*one + c=sub f32", {sub,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, 0x00000400u);
    add("c subnormal f32 neg + products", {sub,sub,sub,sub,sub,sub,sub,sub}, {sub,sub,sub,sub,sub,sub,sub,sub}, 0x80000003u);
    add("f32 min normal - tiny", {mone,z,z,z,z,z,z,z}, {sub,one,one,one,one,one,one,one}, 0x00800000u);
    add("inf*1", {inf,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, F0);
    add("inf*0", {inf,z,z,z,z,z,z,z}, {z,one,one,one,one,one,one,one}, F0);
    add("inf + -inf", {inf,minf,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, F0);
    add("inf + c=-inf", {inf,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, FNINF);
    add("c=inf + finite", {one,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, FINF);
    add("nan in a", {nan,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, F0);
    add("nan c payload", {one,z,z,z,z,z,z,z}, {one,one,one,one,one,one,one,one}, 0xffc12345u);
    add("sum 1+2^-24", {one,sub,z,z,z,z,z,z}, {one,0x0400,one,one,one,one,one,one}, F1);
    // random-ish exact edges: alignment depth checks
    add("1 + 2^-23*0.5 (below grid)", {one,0x0001,z,z,z,z,z,z}, {one,0x0001,one,one,one,one,one,one}, F1);
    add("c=1; product 2^-25", {0x1400,z,z,z,z,z,z,z}, {0x1400,one,one,one,one,one,one,one}, F1);
    add("c=1; product 2^-26", {0x1000,z,z,z,z,z,z,z}, {0x1800,one,one,one,one,one,one,one}, F1);

    const int n = static_cast<int>(cs.size());
    std::vector<uint16_t> a(8 * n), b(8 * n); std::vector<float> c(n), d(n);
    for (int i = 0; i < n; ++i) { std::memcpy(&a[8 * i], cs[i].a, 16); std::memcpy(&b[8 * i], cs[i].b, 16); std::memcpy(&c[i], &cs[i].c, 4); }
    uint16_t *da, *db; float *dc, *dd;
    CK(cudaMalloc(&da, a.size() * 2)); CK(cudaMalloc(&db, b.size() * 2)); CK(cudaMalloc(&dc, n * 4)); CK(cudaMalloc(&dd, n * 4));
    CK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, c.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dd, 0xAB, n * 4));
    k_case<<<n, 32>>>(da, db, dc, dd, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(d.data(), dd, n * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      uint32_t u; std::memcpy(&u, &d[i], 4);
      std::printf("mma_edge_%02d %08x\n", i, u);   // (one line per case: the names are in float_half_forms.cu)
    }
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc)); CK(cudaFree(dd));
  }
  return 0;
}
