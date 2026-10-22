// The SASS executor's pre-decoded tier (src/sass/tier.hpp) against the interpreter (VGPU_TIER=0).
// run_sass_tier.sh runs this program both ways with the performance counters on and requires byte-for-byte the same
// output: every result's hash, and every counter of every launch.
//
// Each kernel computes a spread of expressions from three input arrays whose values break arithmetic -- zeros of both
// signs, the extremes, NaNs quiet and signalling with payloads, infinities, subnormals, shift counts past the width,
// carries and borrows at every word boundary -- mixed with random bits, so that ptxas emits the forms the tier has a
// handler for (IADD3 with carries, IMAD and its WIDE/HI/X forms, LOP3, SHF, LEA, SEL, ISETP and its .EX form, POPC,
// I2FP, FFMA/FADD/FMUL with |x| and -x, FSETP, FSEL, DFMA/DADD/DMUL, global/shared/local loads and stores of every size)
// and the forms it leaves to the interpreter (division, conversions, transcendentals, atomics, shuffles, warp votes).
// Launches use blocks that are not whole warps and guards that leave lanes idle, so the handlers see partial masks.
//
// The program prints no verdict of its own on the arithmetic: the interpreter is the reference here, and it is held
// to the card by the other SASS tests.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;
typedef int64_t i64;

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

__device__ __forceinline__ float F(u32 b) { return __uint_as_float(b); }
__device__ __forceinline__ u32 B(float f) { return __float_as_uint(f); }
__device__ __forceinline__ double D(u64 b) { return __longlong_as_double((long long)b); }
__device__ __forceinline__ u64 Bd(double d) { return (u64)__double_as_longlong(d); }

__constant__ u32 kc[64];

// ---- integer ------------------------------------------------------------------------------------
constexpr int kInt = 64;
__global__ void k_int(const u32* A, const u32* Bv, const u32* C, u32* out, int n, u32 p0, u32 p1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 a = A[i], b = Bv[i], c = C[i];
  i32 sa = (i32)a, sb = (i32)b, sc = (i32)c;
  u64 x = ((u64)a << 32) | b, y = ((u64)c << 32) | a;
  u32 r[kInt];
  int k = 0;
  r[k++] = a + b + c;            r[k++] = a - b - c;           r[k++] = a + b - c;          r[k++] = -a + b - c;
  r[k++] = a * b + c;            r[k++] = a * b - c;           r[k++] = c - a * b;          r[k++] = a * 37u + 1000003u;
  r[k++] = (a << 3) + b;         r[k++] = (a << 7) - b;        r[k++] = a * p0 + p1;        r[k++] = (a + p0) * (b ^ p1);
  r[k++] = (a & b) ^ c;          r[k++] = (a | b) & ~c;        r[k++] = ~(a ^ b) | c;       r[k++] = a & ~b;
  r[k++] = ~a;                   r[k++] = a ^ b ^ c;           r[k++] = (a & b) | (~a & c); r[k++] = (a | b) ^ (b & c);
  r[k++] = __funnelshift_l(a, b, c);  r[k++] = __funnelshift_r(a, b, c);
  r[k++] = __funnelshift_lc(a, b, c); r[k++] = __funnelshift_rc(a, b, c);
  r[k++] = (u32)(x >> (c & 63));  r[k++] = (u32)((i64)x >> (c & 63)); r[k++] = (u32)(x << (c & 63));
  r[k++] = (u32)(((u64)a << (b & 31)) >> 32); r[k++] = a >> (b & 31); r[k++] = (u32)(sa >> (b & 31));
  r[k++] = (a < b) ? c : a;      r[k++] = (sa < sb) ? sb : sc; r[k++] = (a == b) ? c : b;   r[k++] = (a != c) ? a : b;
  r[k++] = (sa >= sc) ? a : b;   r[k++] = (a <= 0x80000000u) ? b : c;
  r[k++] = (u32)(x + y);         r[k++] = (u32)((x + y) >> 32); r[k++] = (u32)(x - y);        r[k++] = (u32)((x - y) >> 32);
  r[k++] = (u32)(x * y);         r[k++] = (u32)((x * y) >> 32); r[k++] = (u32)((u64)a * b);   r[k++] = (u32)(((u64)a * b) >> 32);
  r[k++] = (u32)((i64)sa * sb);  r[k++] = (u32)(((i64)sa * sb) >> 32); r[k++] = __umulhi(a, b); r[k++] = (u32)__mulhi(sa, sb);
  r[k++] = (u32)__umul64hi(x, y); r[k++] = (u32)(__mul64hi((i64)x, (i64)y));
  r[k++] = (u32)(x < y);         r[k++] = (u32)((i64)x < (i64)y); r[k++] = (u32)(x == y);     r[k++] = (u32)(x >= y);
  r[k++] = (u32)(x + c);         r[k++] = (u32)((x + (u64)a * 8) >> 32);
  r[k++] = __popc(a);            r[k++] = __clz(a);            r[k++] = __brev(a);          r[k++] = __ffs(a);
  r[k++] = __byte_perm(a, b, c);  r[k++] = __byte_perm(a, b, 0x5410);
  r[k++] = (a >> 5) & 0x1ff;     r[k++] = (a & 0xfff0f00fu) | (b & 0x000f0ff0u);
  r[k++] = a / 7u;               r[k++] = a % 13u;             r[k++] = (u32)(sa / -5);     r[k++] = a / (b | 1u);
  // r[] holds kInt values at most.
  for (int j = 0; j < kInt && j < k; ++j) out[(size_t)i * kInt + j] = r[j];
  for (int j = k; j < kInt; ++j) out[(size_t)i * kInt + j] = 0;
}

// 128-bit and 64-bit-index arithmetic: carries and borrows that cross words (IADD3.X, IMAD.X, LEA.HI.X, ISETP.EX).
__global__ void k_wide(const u32* A, const u32* Bv, const u32* C, const u64* P, u32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned __int128 x = ((unsigned __int128)(((u64)A[i] << 32) | Bv[i]) << 64) | (((u64)C[i] << 32) | A[(i + 1) % n]);
  unsigned __int128 y = ((unsigned __int128)(((u64)Bv[(i + 2) % n] << 32) | C[(i + 3) % n]) << 64) | ((u64)A[(i + 5) % n] << 17);
  unsigned __int128 s = x + y, d = x - y, m = x * y, nx = -x;
  u64 idx = P[i];
  const u32* base = A + (idx & 0xffff);
  const char* cp = (const char*)base + (i64)(idx >> 3) * 8 - (i64)(idx >> 41);
  const u32* rp = (const u32*)P[(i + 1) % n];   // a pointer value that is anything: only the arithmetic on it is looked at
  const u32* rq = (const u32*)P[(i + 2) % n];
  u32 r[32];
  int k = 0;
  r[k++] = (u32)(u64)(rp + (i32)A[i]);        r[k++] = (u32)((u64)(rp + (i32)A[i]) >> 32);
  r[k++] = (u32)(u64)(rq + (A[i] >> 3));      r[k++] = (u32)((u64)(rq + (A[i] >> 3)) >> 32);
  r[k++] = (u32)(u64)((const double*)rq + (i32)Bv[i]); r[k++] = (u32)((u64)((const double*)rq + (i32)Bv[i]) >> 32);
  r[k++] = (u32)(u64)((const char*)rp - (i32)C[i]);    r[k++] = (u32)((u64)((const char*)rp - (i32)C[i]) >> 32);
  r[k++] = (u32)s; r[k++] = (u32)(s >> 32); r[k++] = (u32)(s >> 64); r[k++] = (u32)(s >> 96);
  r[k++] = (u32)d; r[k++] = (u32)(d >> 32); r[k++] = (u32)(d >> 64); r[k++] = (u32)(d >> 96);
  r[k++] = (u32)m; r[k++] = (u32)(m >> 32); r[k++] = (u32)(m >> 64); r[k++] = (u32)(m >> 96);
  r[k++] = (u32)nx; r[k++] = (u32)(nx >> 64); r[k++] = (u32)(x < y); r[k++] = (u32)((__int128)x < (__int128)y);
  r[k++] = (u32)(x == y); r[k++] = (u32)(x >= y);
  r[k++] = (u32)(u64)cp; r[k++] = (u32)((u64)cp >> 32); r[k++] = (u32)(u64)(cp + (idx >> 20)); r[k++] = (u32)((u64)(cp - (idx << 3)) >> 32);
  r[k++] = (u32)(idx * (u64)A[i]); r[k++] = (u32)((idx * (u64)A[i]) >> 32);
  for (int j = 0; j < 32; ++j) out[(size_t)i * 32 + j] = j < k ? r[j] : 0;
}

// Branches on data, so that lanes diverge and the handlers see partial masks of every shape.
__global__ void k_branch(const u32* A, const u32* Bv, const u32* C, u32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 a = A[i], b = Bv[i], c = C[i];
  u32 acc = a;
  for (int it = 0; it < 12 + (int)(c & 7); ++it) {
    if (acc & 1) acc = acc * 3u + b;
    else if (acc & 2) acc = (acc >> 1) ^ c;
    else acc = acc + (b << (it & 7)) - 5u;
    if ((acc & 0xff) == (b & 0xff)) acc += it;
    switch (acc & 3) {
      case 0: acc ^= 0x9e3779b9u; break;
      case 1: acc = acc * 5u + 1u; break;
      case 2: acc -= c; break;
      default: acc = (acc << 5) | (acc >> 27);
    }
  }
  out[i] = acc;
}

// ---- float and double ---------------------------------------------------------------------------
constexpr int kFlt = 64;
__global__ void k_float(const u32* A, const u32* Bv, const u32* C, u32* out, int n, u32 p0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = F(A[i]), y = F(Bv[i]), z = F(C[i]);
  float q = F(p0);
  u32 r[kFlt];
  int k = 0;
  r[k++] = B(fmaf(x, y, z));     r[k++] = B(fmaf(-x, y, z));   r[k++] = B(fmaf(x, y, -z));  r[k++] = B(fmaf(-x, y, -z));
  r[k++] = B(fmaf(fabsf(x), y, z)); r[k++] = B(fmaf(x, fabsf(y), -fabsf(z))); r[k++] = B(fmaf(x, 2.5f, y)); r[k++] = B(fmaf(x, y, 0.5f));
  r[k++] = B(fmaf(x, q, z));     r[k++] = B(fmaf(q, y, z));    r[k++] = B(fmaf(x, y, q));   r[k++] = B(fmaf(-fabsf(x), q, y));
  r[k++] = B(x + y);             r[k++] = B(x - y);            r[k++] = B(-x - y);          r[k++] = B(x + 1.0f);
  r[k++] = B(fabsf(x) + fabsf(y)); r[k++] = B(x + q);          r[k++] = B(x - q);           r[k++] = B(__fadd_rn(x, -y));
  r[k++] = B(x * y);             r[k++] = B(-x * y);           r[k++] = B(x * 3.0f);        r[k++] = B(x * q);
  r[k++] = B(fabsf(x) * -fabsf(y)); r[k++] = B(__fmul_rn(x, y)); r[k++] = B(x * 0.5f);      r[k++] = B(x * -2.0f);
  r[k++] = (x < y) ? B(x) : B(y);   r[k++] = (x >= z) ? B(z) : B(x); r[k++] = (x == y) ? B(z) : B(x); r[k++] = (x != y) ? B(y) : B(z);
  r[k++] = (x > y && y > z) ? 1u : 0u; r[k++] = (x < y || z != z) ? 1u : 0u; r[k++] = (fabsf(x) < 1.0f) ? B(-x) : B(fabsf(y));
  r[k++] = (x > q) ? B(x) : B(q);    r[k++] = (x != x) ? 7u : 9u;
  r[k++] = B(fminf(x, y));       r[k++] = B(fmaxf(x, y));      r[k++] = B(copysignf(x, y)); r[k++] = B(fabsf(x));
  r[k++] = B(truncf(x));         r[k++] = B(floorf(x));        r[k++] = B(ceilf(x));        r[k++] = B(rintf(x));
  r[k++] = (u32)(i32)x;          r[k++] = (u32)(unsigned)fabsf(x); r[k++] = B((float)(i32)A[i]); r[k++] = B((float)A[i]);
  r[k++] = B((float)(i32)Bv[i]); r[k++] = B((float)(u32)(Bv[i] >> 3)); r[k++] = B(sqrtf(fabsf(x)));  r[k++] = B(rsqrtf(fabsf(y)));
  r[k++] = B(1.0f / z);          r[k++] = B(x / y);            r[k++] = B(__fdividef(x, y)); r[k++] = B(__expf(x * 0.01f));
  r[k++] = B(__logf(fabsf(x))); r[k++] = B(__sinf(x * 0.1f)); r[k++] = B(__cosf(y * 0.1f)); r[k++] = B(__saturatef(x));
  for (int j = 0; j < kFlt; ++j) out[(size_t)i * kFlt + j] = j < k ? r[j] : 0;
}

constexpr int kDbl = 24;
__global__ void k_double(const u64* A, const u64* Bv, const u64* C, u64* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double x = D(A[i]), y = D(Bv[i]), z = D(C[i]);
  u64 r[kDbl];
  int k = 0;
  r[k++] = Bd(fma(x, y, z));     r[k++] = Bd(fma(-x, y, z));   r[k++] = Bd(fma(x, y, -z));  r[k++] = Bd(fma(fabs(x), y, z));
  r[k++] = Bd(fma(x, 1.0000001, y)); r[k++] = Bd(fma(x, y, 0.25)); r[k++] = Bd(x + y);       r[k++] = Bd(x - y);
  r[k++] = Bd(x * y);            r[k++] = Bd(-x * y);          r[k++] = Bd(x * 3.5);        r[k++] = Bd(x + 1.5);
  r[k++] = Bd(fabs(x) + fabs(y)); r[k++] = (x < y) ? 1u : 0u;  r[k++] = (x >= z) ? 3u : 4u; r[k++] = (x != x) ? 5u : 6u;
  r[k++] = Bd(fmin(x, y));       r[k++] = Bd(fmax(x, y));      r[k++] = Bd(sqrt(fabs(x)));  r[k++] = Bd(x / y);
  r[k++] = Bd((double)(float)x); r[k++] = (u64)(i64)x;         r[k++] = Bd((double)(i64)A[i]); r[k++] = Bd(__dmul_rn(x, y));
  for (int j = 0; j < kDbl; ++j) out[(size_t)i * kDbl + j] = j < k ? r[j] : 0;
}


// ---- the forms whose result rules the card fixed (round 9) --------------------------------------------------------------------------
// The tier must give what the interpreter gives for these, and the interpreter's rules for them are the card's: a -0 under .SAT, two NaNs in FMNMX,
// the NaN a MUFU makes, the NaN of a packed half or a conversion, F2I/F2F/FRND with a NaN and with the .H1 select, the NaN rule of DADD/DMUL/DFMA
// (the first NaN of (b, a) or (b, c, a), taken from the register and not from |x| or -x), the bfloat16 HFMA2, VABSDIFF, PRMT's selector bits, and the
// tensor-core sum (HMMA; an sm_86 cubin's, VGPU_TC_EXACT). The tier's handlers call the interpreter's kernels and code for them; these kernels are
// what holds that: they run the forms on inputs made of NaNs of both signs with distinct payloads in every position, zeros of both signs,
// subnormals and the extremes mixed with random bits.
#define OF(S, ...) do { float r_; asm(S : "=f"(r_) : __VA_ARGS__); o[k++] = B(r_); } while (0)
#define OU(S, ...) do { u32 r_; asm(S : "=r"(r_) : __VA_ARGS__); o[k++] = r_; } while (0)
#define OH(S, ...) do { unsigned short h_; asm(S : "=h"(h_) : __VA_ARGS__); o[k++] = h_; } while (0)
#define OL(S, ...) do { u64 q_; asm(S : "=l"(q_) : __VA_ARGS__); o[k++] = (u32)q_; o[k++] = (u32)(q_ >> 32); } while (0)
#define OD(S, ...) do { double d_; asm(S : "=d"(d_) : __VA_ARGS__); const u64 q_ = Bd(d_); o[k++] = (u32)q_; o[k++] = (u32)(q_ >> 32); } while (0)

constexpr int kF2 = 160;
__global__ void k_forms2(const u32* A, const u32* Bv, const u32* C, u32* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const u32 ua = A[i], ub = Bv[i], uc = C[i];
  const float a = F(ua), b = F(ub), c = F(uc);
  const double da = D(((u64)ua << 32) | ub), db = D(((u64)uc << 32) | ua);
  u32 o[kF2] = {};
  int k = 0;
  // .SAT and .FTZ of a negative zero, and of values that flush to one; directed rounding.
  OF("fma.rn.sat.f32 %0, %1, %2, %3;", "f"(a), "f"(b), "f"(c));
  OF("fma.rn.ftz.sat.f32 %0, %1, %2, %3;", "f"(-a), "f"(b), "f"(c));
  OF("mul.rn.ftz.sat.f32 %0, %1, %2;", "f"(a), "f"(-b));
  OF("mul.rn.sat.f32 %0, %1, %2;", "f"(a), "f"(b));
  OF("add.rn.ftz.sat.f32 %0, %1, %2;", "f"(a), "f"(b));
  OF("add.rn.sat.f32 %0, %1, %2;", "f"(-a), "f"(-b));
  OF("fma.rm.sat.f32 %0, %1, %2, %3;", "f"(a), "f"(-b), "f"(c));
  OF("mul.rp.ftz.f32 %0, %1, %2;", "f"(a), "f"(b));
  OF("add.rz.ftz.f32 %0, %1, %2;", "f"(-a), "f"(b));
  // FMNMX with two NaNs, with .NAN, with a flush.
  OF("min.f32 %0, %1, %2;", "f"(a), "f"(b));
  OF("max.f32 %0, %1, %2;", "f"(a), "f"(b));
#if __CUDA_ARCH__ >= 800
  OF("min.NaN.f32 %0, %1, %2;", "f"(a), "f"(b));
  OF("max.ftz.NaN.f32 %0, %1, %2;", "f"(a), "f"(b));
#endif
  OF("min.ftz.f32 %0, %1, %2;", "f"(-a), "f"(b));
  OF("max.f32 %0, %1, %2;", "f"(fabsf(a)), "f"(-b));
  // MUFU: the NaN a negative operand or an infinity makes, with and without a flush, with a source modifier.
  OF("sin.approx.f32 %0, %1;", "f"(a));
  OF("cos.approx.ftz.f32 %0, %1;", "f"(a));
  OF("sin.approx.ftz.f32 %0, %1;", "f"(-a));
  OF("ex2.approx.f32 %0, %1;", "f"(a));
  OF("ex2.approx.ftz.f32 %0, %1;", "f"(-fabsf(a)));
  OF("lg2.approx.f32 %0, %1;", "f"(a));
  OF("lg2.approx.ftz.f32 %0, %1;", "f"(-a));
  OF("rsqrt.approx.f32 %0, %1;", "f"(a));
  OF("rsqrt.approx.ftz.f32 %0, %1;", "f"(-a));
  OF("sqrt.approx.f32 %0, %1;", "f"(a));
  OF("sqrt.approx.ftz.f32 %0, %1;", "f"(-fabsf(b)));
  OF("rcp.approx.ftz.f32 %0, %1;", "f"(a));
  OF("rcp.approx.ftz.f32 %0, %1;", "f"(-b));
  OF("tanh.approx.f32 %0, %1;", "f"(a));
  OF("tanh.approx.f32 %0, %1;", "f"(-b));
  // F2I of every size and rounding; a NaN, an infinity, the edges; from f16 (the low half and .H1) and from f64.
  OU("cvt.rzi.s32.f32 %0, %1;", "f"(a));
  OU("cvt.rni.u32.f32 %0, %1;", "f"(a));
  OU("cvt.rpi.s32.f32 %0, %1;", "f"(-a));
  OU("cvt.rmi.ftz.u32.f32 %0, %1;", "f"(b));
  OH("cvt.rzi.s16.f32 %0, %1;", "f"(a));
  OH("cvt.rni.u16.f32 %0, %1;", "f"(-b));
  OL("cvt.rzi.s64.f32 %0, %1;", "f"(a));
  OL("cvt.rni.u64.f32 %0, %1;", "f"(b));
  OU("cvt.rzi.s32.f64 %0, %1;", "d"(da));
  OU("cvt.rpi.u32.f64 %0, %1;", "d"(-db));
  OL("cvt.rni.s64.f64 %0, %1;", "d"(da));
  OL("cvt.rzi.u64.f64 %0, %1;", "d"(db));
  OU("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rzi.s32.f16 %0, lo; }", "r"(ua));
  OU("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rni.u32.f16 %0, hi; }", "r"(ub));
  OU("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rpi.s32.f16 %0, hi; }", "r"(uc));
  // F2F and FRND: f16, bf16, f32 and f64 into each other, in each rounding mode; from the high half.
  OH("cvt.rn.f16.f32 %0, %1;", "f"(a));
  OH("cvt.rz.f16.f32 %0, %1;", "f"(b));
  OH("cvt.rm.f16.f32 %0, %1;", "f"(c));
  OH("cvt.rp.f16.f32 %0, %1;", "f"(-a));
  OH("cvt.rn.f16.f64 %0, %1;", "d"(da));
  OH("cvt.rz.f16.f64 %0, %1;", "d"(db));
  OF("cvt.f32.f16 %0, %1;", "h"((unsigned short)ua));
  OF("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.f32.f16 %0, hi; }", "r"(ub));
  OD("cvt.f64.f32 %0, %1;", "f"(a));
  OD("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.f64.f16 %0, hi; }", "r"(uc));
  OF("cvt.rn.f32.f64 %0, %1;", "d"(da));
  OF("cvt.rz.f32.f64 %0, %1;", "d"(db));
  OF("cvt.rp.f32.f64 %0, %1;", "d"(-da));
  OF("cvt.rni.f32.f32 %0, %1;", "f"(a));
  OF("cvt.rzi.f32.f32 %0, %1;", "f"(b));
  OF("cvt.rmi.ftz.f32.f32 %0, %1;", "f"(c));
  OF("cvt.rpi.f32.f32 %0, %1;", "f"(-a));
  OD("cvt.rni.f64.f64 %0, %1;", "d"(da));
  OD("cvt.rmi.f64.f64 %0, %1;", "d"(-db));
  OH("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rni.f16.f16 %0, hi; }", "r"(ua));
  OH("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rzi.f16.f16 %0, lo; }", "r"(ub));
  // I2F to every format and rounding, from every integer size (and the high half or byte).
  OF("cvt.rn.f32.s32 %0, %1;", "r"(ua));
  OF("cvt.rz.f32.u32 %0, %1;", "r"(ub));
  OF("cvt.rm.f32.s32 %0, %1;", "r"(uc));
  OF("cvt.rp.f32.u32 %0, %1;", "r"(ua));
  OF("cvt.rm.f32.s64 %0, %1;", "l"(((u64)ua << 32) | ub));
  OF("cvt.rp.f32.u64 %0, %1;", "l"(((u64)uc << 32) | ub));
  OD("cvt.rn.f64.s64 %0, %1;", "l"(((u64)ub << 32) | uc));
  OD("cvt.rn.f64.u32 %0, %1;", "r"(ua));
  OH("cvt.rn.f16.s32 %0, %1;", "r"(ua));
  OH("cvt.rz.f16.u32 %0, %1;", "r"(ub));
  OH("cvt.rm.f16.s64 %0, %1;", "l"(((u64)ub << 32) | uc));
  OH("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rp.f16.u16 %0, hi; }", "r"(uc));
  OF("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rp.f32.s16 %0, hi; }", "r"(ua));
  OF("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rz.f32.u16 %0, lo; }", "r"(ub));
  // The sum of absolute differences.
  o[k++] = __usad(ua, ub, uc);
  o[k++] = (u32)__sad((int)ua, (int)ub, (int)uc);
  o[k++] = __usad(ub, uc, ua) ^ (u32)__sad((int)uc, (int)ua, (int)ub);
#if __CUDA_ARCH__ >= 800
  OH("cvt.rn.bf16.f32 %0, %1;", "f"(a));
  OH("cvt.rz.bf16.f32 %0, %1;", "f"(b));
  OH("cvt.rm.bf16.f32 %0, %1;", "f"(-c));
  OU("cvt.rn.relu.bf16x2.f32 %0, %1, %2;", "f"(a), "f"(b));
  OU("cvt.rn.relu.f16x2.f32 %0, %1, %2;", "f"(-a), "f"(b));
  OU("cvt.rn.bf16x2.f32 %0, %1, %2;", "f"(b), "f"(c));
  OU("cvt.rz.f16x2.f32 %0, %1, %2;", "f"(c), "f"(a));
#endif
  for (int j = 0; j < kF2; ++j) out[(size_t)i * kF2 + j] = o[j];
}

constexpr int kH2 = 64;
__global__ void k_half2(const u32* A, const u32* Bv, const u32* C, u32* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const u32 x = A[i], y = Bv[i], z = C[i];
  u32 o[kH2] = {};
  int k = 0;
  // packed f16: HFMA2, HADD2, HMUL2 with NaNs of both signs in each half, saturated, flushed; the comparisons (HSET2, HSETP2) with and without .FTZ.
  OU("fma.rn.f16x2 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("fma.rn.ftz.f16x2 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("fma.rn.sat.f16x2 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("fma.rn.f16x2 %0, %1, %2, %3;", "r"(x ^ 0x80008000u), "r"(y & 0x7fff7fffu), "r"(z));
  OU("add.rn.f16x2 %0, %1, %2;", "r"(x), "r"(y));
  OU("mul.rn.f16x2 %0, %1, %2;", "r"(x), "r"(y));
  OU("set.eq.f16x2.f16x2 %0, %1, %2;", "r"(x), "r"(y));
  OU("set.lt.ftz.f16x2.f16x2 %0, %1, %2;", "r"(x), "r"(y));
  OU("set.ge.ftz.f16x2.f16x2 %0, %1, %2;", "r"(y), "r"(z));
  OU("set.nan.f16x2.f16x2 %0, %1, %2;", "r"(x), "r"(z));
  OU("set.neu.f16x2.f16x2 %0, %1, %2;", "r"(x), "r"(z));
  OU("{ .reg .pred p, q; setp.gt.ftz.f16x2 p|q, %1, %2; selp.u32 %0, 3, 0, p; }", "r"(x), "r"(y));
  OU("{ .reg .pred p, q; setp.le.f16x2 p|q, %1, %2; selp.u32 %0, 5, 6, q; }", "r"(y), "r"(z));
  // The half to float conversion (HADD2.F32) of each half, the two floats packed to f16x2 (F2FP.PACK_AB), with .RELU, from NaNs and from -0.
  OF("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.f32.f16 %0, lo; }", "r"(x));
  OF("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.f32.f16 %0, hi; }", "r"(y));
#if __CUDA_ARCH__ >= 800
  OU("cvt.rn.f16x2.f32 %0, %1, %2;", "f"(F(x)), "f"(F(y)));
  OU("cvt.rn.relu.f16x2.f32 %0, %1, %2;", "f"(F(y)), "f"(-F(z)));
  OU("cvt.rn.f16x2.f32 %0, %1, %2;", "f"(-F(z)), "f"(fabsf(F(x))));
  // bfloat16 pairs: HFMA2.BF16_V2 is one correctly rounded result; the packs.
  OU("fma.rn.bf16x2 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("fma.rn.bf16x2 %0, %1, %2, %3;", "r"(x ^ 0x80008000u), "r"(y), "r"(z & 0x7fff7fffu));
  OU("fma.rn.relu.bf16x2 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("fma.rn.bf16x2 %0, %1, %2, %3;", "r"(x), "r"(0x3f803f80u), "r"(z));   // a sum
  OU("fma.rn.bf16x2 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(0x80008000u));   // a product
  OU("cvt.rn.bf16x2.f32 %0, %1, %2;", "f"(F(x)), "f"(F(y)));
  OU("cvt.rn.relu.bf16x2.f32 %0, %1, %2;", "f"(-F(y)), "f"(F(z)));
#endif
  for (int j = 0; j < kH2; ++j) out[(size_t)i * kH2 + j] = o[j];
}

constexpr int kD2 = 56;
__constant__ double kdc[4];
__global__ void k_double2(const u64* A, const u64* Bv, const u64* C, u64* out, int n, double kp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double x = D(A[i]), y = D(Bv[i]), z = D(C[i]);
  u64 r[kD2] = {};
  int k = 0;
  // DADD, DMUL and DFMA with a NaN in any operand, with |x| and -x on each (the NaN taken is the register's, not the modified one), in each rounding.
#define ODD(S, ...) do { double d_; asm(S : "=d"(d_) : __VA_ARGS__); r[k++] = Bd(d_); } while (0)
  ODD("fma.rn.f64 %0, %1, %2, %3;", "d"(x), "d"(y), "d"(z));
  ODD("fma.rn.f64 %0, %1, %2, %3;", "d"(-x), "d"(y), "d"(z));
  ODD("fma.rn.f64 %0, %1, %2, %3;", "d"(fabs(x)), "d"(-y), "d"(-fabs(z)));
  ODD("fma.rn.f64 %0, %1, %2, %3;", "d"(x), "d"(fabs(y)), "d"(-z));
  ODD("fma.rz.f64 %0, %1, %2, %3;", "d"(-fabs(x)), "d"(y), "d"(fabs(z)));
  ODD("fma.rm.f64 %0, %1, %2, %3;", "d"(x), "d"(y), "d"(z));
  ODD("fma.rp.f64 %0, %1, %2, %3;", "d"(-x), "d"(-y), "d"(z));
  ODD("add.rn.f64 %0, %1, %2;", "d"(x), "d"(y));
  ODD("add.rn.f64 %0, %1, %2;", "d"(-x), "d"(y));
  ODD("add.rn.f64 %0, %1, %2;", "d"(fabs(x)), "d"(fabs(y)));
  ODD("add.rz.f64 %0, %1, %2;", "d"(x), "d"(-fabs(y)));
  ODD("add.rm.f64 %0, %1, %2;", "d"(-fabs(x)), "d"(y));
  ODD("mul.rn.f64 %0, %1, %2;", "d"(x), "d"(y));
  ODD("mul.rn.f64 %0, %1, %2;", "d"(-x), "d"(fabs(y)));
  ODD("mul.rp.f64 %0, %1, %2;", "d"(fabs(x)), "d"(y));
  ODD("mul.rz.f64 %0, %1, %2;", "d"(x), "d"(-y));
  r[k++] = Bd(fmin(x, y));
  r[k++] = Bd(fmax(-x, fabs(y)));
  r[k++] = (x < y) | ((x == z) << 1) | ((y >= z) << 2) | ((x != x) << 3) | ((fabs(y) > z) << 4);
  // An operand that is the same for every lane -- a kernel argument or a __constant__ value, which the instruction reads from a constant bank -- with
  // |x| and -x on it (the kernel applies them to it as to a register).
  ODD("fma.rn.f64 %0, %1, %2, %3;", "d"(x), "d"(-kp), "d"(y));
  ODD("fma.rn.f64 %0, %1, %2, %3;", "d"(x), "d"(fabs(kp)), "d"(-z));
  ODD("fma.rz.f64 %0, %1, %2, %3;", "d"(-kdc[1]), "d"(y), "d"(z));
  ODD("add.rn.f64 %0, %1, %2;", "d"(x), "d"(-kp));
  ODD("add.rn.f64 %0, %1, %2;", "d"(-fabs(kdc[2])), "d"(y));
  ODD("mul.rn.f64 %0, %1, %2;", "d"(-kdc[3]), "d"(z));
  ODD("mul.rp.f64 %0, %1, %2;", "d"(x), "d"(-kp));
  // DSETP with both of its predicates used (the second is the complement, or for .NUM/.NAN whether both operands are NaN).
#define ODS(S, ...) do { u32 q_; asm(S : "=r"(q_) : __VA_ARGS__); r[k++] = q_; } while (0)
  ODS("{ .reg .pred p, q; setp.lt.f64 p|q, %1, %2; selp.u32 %0, 3, 4, q; }", "d"(x), "d"(y));
  ODS("{ .reg .pred p, q; setp.lt.f64 p|q, %1, %2; selp.u32 %0, 5, 6, p; }", "d"(x), "d"(-y));
  ODS("{ .reg .pred p, q, c; setp.ne.u32 c, %3, 0; setp.ge.and.f64 p|q, %1, %2, c; selp.u32 %0, 7, 8, q; }", "d"(y), "d"(z), "r"((u32)(x == x)));
  ODS("{ .reg .pred p, q; setp.nan.f64 p|q, %1, %2; selp.u32 %0, 9, 10, q; }", "d"(x), "d"(z));
  ODS("{ .reg .pred p, q; setp.num.f64 p|q, %1, %2; selp.u32 %0, 11, 12, q; }", "d"(y), "d"(z));
  ODS("{ .reg .pred p, q; setp.lt.f64 p|q, %1, %2; selp.u32 %0, 13, 14, q; }", "d"(x), "d"(kp));
  for (int j = 0; j < kD2; ++j) out[(size_t)i * kD2 + j] = r[j];
}

// PRMT with selectors of any bits, in every mode.
constexpr int kM2 = 16;
__global__ void k_prmt2(const u32* A, const u32* Bv, const u32* C, u32* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const u32 x = A[i], y = Bv[i], z = C[i];
  u32 o[kM2] = {};
  int k = 0;
  OU("prmt.b32 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("prmt.b32.f4e %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("prmt.b32.f4e %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z >> 7));
  OU("prmt.b32.b4e %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("prmt.b32.b4e %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z >> 5));
  OU("prmt.b32.rc8 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("prmt.b32.ecl %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("prmt.b32.ecl %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z >> 9));
  OU("prmt.b32.ecr %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  OU("prmt.b32.ecr %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z >> 3));
  OU("prmt.b32.rc16 %0, %1, %2, %3;", "r"(x), "r"(y), "r"(z));
  for (int j = 0; j < kM2; ++j) out[(size_t)i * kM2 + j] = o[j];
}

// The tensor cores: mma.sync of f16, bf16 and tf32 products into f32 and f16 accumulators (HMMA). An sm_86 cubin sums as an RTX 3060's tensor core does
// (chunks, truncated) unless VGPU_TC_EXACT=0; every other one sums sequentially. All 32 lanes of a warp take part.
constexpr int kMma = 24;
__global__ void k_mma(const u32* A, const u32* Bv, const u32* C, u32* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  u32 a[4], b[2], c[4], o[kMma] = {};
  for (int j = 0; j < 4; ++j) { a[j] = A[((size_t)i * 4 + j) % n]; c[j] = C[((size_t)i * 4 + j) % n]; }
  for (int j = 0; j < 2; ++j) b[j] = Bv[((size_t)i * 2 + j) % n];
  int k = 0;
  {   // m16n8k8, f16 (sm_75 has it)
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "f"(F(c[0])), "f"(F(c[1])), "f"(F(c[2])), "f"(F(c[3])));
    o[k++] = B(d0); o[k++] = B(d1); o[k++] = B(d2); o[k++] = B(d3);
  }
#if __CUDA_ARCH__ >= 800
  {   // m16n8k16, f16, f32 accumulator
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(F(c[0])), "f"(F(c[1])), "f"(F(c[2])), "f"(F(c[3])));
    o[k++] = B(d0); o[k++] = B(d1); o[k++] = B(d2); o[k++] = B(d3);
  }
  {   // m16n8k16, f16, f16 accumulator
    u32 d0, d1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
    o[k++] = d0; o[k++] = d1;
  }
  {   // m16n8k16, bf16, f32 accumulator
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(F(c[0])), "f"(F(c[1])), "f"(F(c[2])), "f"(F(c[3])));
    o[k++] = B(d0); o[k++] = B(d1); o[k++] = B(d2); o[k++] = B(d3);
  }
  {   // m16n8k8, tf32
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(F(c[0])), "f"(F(c[1])), "f"(F(c[2])), "f"(F(c[3])));
    o[k++] = B(d0); o[k++] = B(d1); o[k++] = B(d2); o[k++] = B(d3);
  }
  {   // m16n8k8, bf16
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "f"(F(c[0])), "f"(F(c[1])), "f"(F(c[2])), "f"(F(c[3])));
    o[k++] = B(d0); o[k++] = B(d1); o[k++] = B(d2); o[k++] = B(d3);
  }
#endif
  if (i < n)
    for (int j = 0; j < kMma; ++j) out[(size_t)i * kMma + j] = o[j];
}

// ---- memory -------------------------------------------------------------------------------------
template <class T>
__global__ void k_copy(const T* in, T* out, int n, int lane_limit, int stride, int shift) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && (threadIdx.x % 32) < lane_limit) {
    out[i] = in[((long)i * stride + shift) % n];
    out[n + i] = in[(i ^ shift) % n];   // a second store stream: scattered and coalesced
  }
}

template <class T>
__global__ void k_shared(const T* in, T* out, int mul, int add) {
  __shared__ T s[512];
  int t = threadIdx.x;
  s[(t * mul + add) % 512] = in[blockIdx.x * blockDim.x + t];
  __syncthreads();
  out[blockIdx.x * blockDim.x + t] = s[(t * mul) % 512] ^ s[(t * 5 + 1) % 512] ;
}

struct __align__(16) Q { u32 a, b, c, d; };
__device__ __forceinline__ Q operator^(Q x, Q y) { return Q{x.a ^ y.a, x.b ^ y.b, x.c ^ y.c, x.d ^ y.d}; }

// Local memory: a per-thread array indexed by data (LDL/STL, byte, word and 16-byte forms).
__global__ void k_local(const u32* A, const u32* Bv, u32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 arr[24];
  unsigned char bytes[40];
  Q q[6];
  for (int j = 0; j < 24; ++j) arr[j] = A[(i + j) % n] * (u32)(j + 1);
  for (int j = 0; j < 40; ++j) bytes[j] = (unsigned char)(Bv[(i + j) % n] >> (j & 24));
  for (int j = 0; j < 6; ++j) q[j] = Q{A[(i + j) % n], Bv[(i + j) % n], (u32)j, (u32)i};
  u32 b = Bv[i], acc = 0;
  for (int it = 0; it < 24; ++it) {
    arr[(b + it) % 24] += arr[(b >> 3) % 24] ^ it;
    bytes[(b >> 1) % 40] += (unsigned char)arr[it];
    q[b % 6].c += q[(b >> 4) % 6].a;
    acc += arr[(it * 5) % 24] + bytes[(it * 7) % 40] + q[it % 6].b + q[(it + 1) % 6].c;
  }
  out[i] = acc;
}

// Constant memory (read through the constant bank or the global window) and kernel arguments as operands.
__global__ void k_const(const u32* A, u32* out, int n, u32 p0, u32 p1, u32 p2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 a = A[i];
  out[i] = kc[a & 63] + kc[(a >> 6) & 63] * p0 + (a ^ p1) + p2 + blockDim.x * 3u + gridDim.x + blockIdx.x;
}

// Generic loads and stores, and atomics, left to the interpreter.
__global__ void k_generic(u32* g, u32* cnt, const u32* A, int n) {
  __shared__ u32 s[256];
  int t = threadIdx.x;
  s[t] = A[blockIdx.x * blockDim.x + t];
  __syncthreads();
  u32* p = (t & 1) ? &s[(t * 7) & 255] : &g[blockIdx.x * blockDim.x + t];
  u32 v = *p;
  *p = v * 3u + t;
  atomicAdd(&cnt[(t + blockIdx.x) & 63], v | 1u);   // commutative: the same sum in any order
  g[blockIdx.x * blockDim.x + t] ^= __shfl_xor_sync(0xffffffffu, v, 3) + __popc(__ballot_sync(0xffffffffu, v & 4));
}

// The warp-wide, atomic and call instructions that end a run: MATCH (the tier's own handler: every mode, 32 and 64 bits, whole and diverged
// warps) beside the ones the interpreter keeps (global atomics and RED, shared atomics, REDUX, VOTEU, CALL/RET, WARPSYNC, BRA.DIV/.CONV),
// so that the tier is held to the interpreter's answers on the neighbours of each. Each block works on bins of its own, so that the order in which the host
// threads run the blocks changes nothing; the order of the lanes and of the block's warps is the executor's, the same both ways.
__device__ __noinline__ u32 d_leaf(u32 x, u32 y) { return (x * 2654435761u) ^ (y >> 3); }
__device__ __noinline__ u32 d_f0(u32 x) { return x > 100u ? d_leaf(x, x + 1u) : x * 7u; }
__device__ __noinline__ u32 d_f1(u32 x) { return x ^ (x << 5); }
__device__ __noinline__ u32 d_f2(u32 x) { return d_f0(x) + d_f1(x); }
__device__ __noinline__ u32 d_f3(u32 x) { return ~x; }
typedef u32 (*d_fn)(u32);
__device__ d_fn d_ftab[4] = {d_f0, d_f1, d_f2, d_f3};
constexpr int kDir = 40;
__global__ void k_direct(const u32* A, const u32* Bv, u32* out, u32* bins, unsigned long long* bins64, float* binsf, double* binsd, int n) {
  const int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
  const bool in = i < n;
  const u32 a = in ? A[i] : 0u, b = in ? Bv[i] : 0u;
  u32* bn = bins + blockIdx.x * 64;
  unsigned long long* b64 = bins64 + blockIdx.x * 8;
  float* bf = binsf + blockIdx.x * 8;
  double* bd = binsd + blockIdx.x * 8;
  __shared__ u32 sh[64];
  __shared__ unsigned long long sh64[8];
  if (t < 64) sh[t] = t * 3u;
  if (t < 8) sh64[t] = t;
  __syncthreads();
  u32 r[kDir];
  int k = 0;
  const u32 lane = threadIdx.x & 31;
  const u32 full = __activemask();
  r[k++] = __match_any_sync(full, a & 7u);
  int pr = 0;
  r[k++] = __match_all_sync(full, (u32)(a & 0xf0u) | ((lane >> 5)), &pr);
  r[k++] = (u32)pr;
  r[k++] = __match_any_sync(full, ((u64)a << 20) | (b & 3u));
#if __CUDA_ARCH__ >= 800
  r[k++] = __reduce_add_sync(full, a);
  r[k++] = __reduce_min_sync(full, a);
  r[k++] = __reduce_max_sync(full, b);
  r[k++] = __reduce_or_sync(full, a);
  r[k++] = __reduce_and_sync(full, a | b);
  r[k++] = __reduce_xor_sync(full, a ^ b);
  r[k++] = (u32)__reduce_min_sync(full, (i32)a);
  r[k++] = (u32)__reduce_max_sync(full, (i32)b);
#else
  for (int j = 0; j < 8; ++j) r[k++] = a + j;
#endif
  // a warp-aggregated increment: the leader adds for everyone (VOTEU/REDUX/BRA.CONV in ptxas's code)
  {
    const u32 m = __activemask();
    const u32 leader = __ffs(m) - 1;
    u32 old = 0;
    if (lane == leader) old = atomicAdd(&bn[0], (u32)__popc(m));
    old = __shfl_sync(m, old, leader);
    r[k++] = old + __popc(m & ((1u << lane) - 1u));
  }
  // atomics of every kind, with their old values, and the same without (RED)
  r[k++] = atomicAdd(&bn[1], a | 1u);
  r[k++] = atomicMax(&bn[2], a);
  r[k++] = atomicMin(&bn[3], b);
  r[k++] = atomicAnd(&bn[4], a | 0x0f0f0f0fu);
  r[k++] = atomicOr(&bn[5], b & 0xff00u);
  r[k++] = atomicXor(&bn[6], a & 0xfffu);
  r[k++] = atomicExch(&bn[7 + (lane & 3)], a);
  r[k++] = atomicCAS(&bn[12], b & 7u, a);
  r[k++] = atomicInc(&bn[13], b & 63u);
  r[k++] = atomicDec(&bn[14], b & 63u);
  if (in) {
    atomicAdd(&bn[16 + (a & 7)], 1u);
    atomicMax(&bn[24], a);
    atomicMin((i32*)&bn[25], (i32)b);
    atomicOr(&bn[26], 1u << (a & 31));
    atomicXor(&bn[27], b);
  }
  r[k++] = (u32)atomicAdd(&b64[0], (unsigned long long)a * 3u + b);
  r[k++] = (u32)(atomicAdd(&b64[1], (unsigned long long)a << 32) >> 32);
  r[k++] = (u32)atomicCAS(&b64[2], (unsigned long long)(b & 3u), (unsigned long long)a);
  r[k++] = (u32)atomicExch(&b64[3], (unsigned long long)a);
  r[k++] = __float_as_uint(atomicAdd(&bf[0], (float)(a & 0xff) * 0.25f));
  atomicAdd(&bf[1], (float)(b & 0x3f));
#if __CUDA_ARCH__ >= 600
  r[k++] = (u32)__double_as_longlong(atomicAdd(&bd[0], (double)(a & 0xff) * 0.5));
#else
  r[k++] = a;
#endif
  // shared atomics
  r[k++] = atomicAdd(&sh[a & 15], b | 1u);
  r[k++] = atomicMax(&sh[16 + (b & 7)], a);
  r[k++] = atomicCAS(&sh[24 + (a & 3)], a & 7u, b);
  r[k++] = (u32)atomicAdd(&sh64[a & 3], (unsigned long long)b);
  __syncthreads();
  // calls and returns, through a table (an indirect call) and directly, in a diverged warp
  r[k++] = d_ftab[a & 3](b);
  r[k++] = d_f2(a);
  // collectives inside divergent code: BRA.DIV/.CONV and WARPSYNC
  if (a & 1) {
    const u32 m = __activemask();
    r[k++] = __match_any_sync(m, b & 3u);
    __syncwarp(m);
  } else {
    const u32 m = __activemask();
    r[k++] = __ballot_sync(m, b & 8u) + __popc(m);
    __syncwarp(m);
  }
  // an IEEE division: FCHK, then the slow path through a CALL when the check says so (the interpreter says so always); FCHK itself is the
  // interpreter's (a handler for it measured +1.2% and +1.8%, docs/performance.md)
  r[k++] = B(F(a) / F(b));
  r[k++] = B(F(b ^ 0x3f800000u) / F(a | 0x00800000u));
  while (k < kDir) r[k++] = lane;
  if (in)
    for (int j = 0; j < kDir; ++j) out[(size_t)i * kDir + j] = r[j];
  __syncthreads();
  if (t < 64) out[(size_t)n * kDir + blockIdx.x * 64 + t] = sh[t];
}

// The integer tensor cores (IMMA): mma.sync of s8, u8 and s4 products into s32, plain and .satfinite, in the shapes m8n8k16, m16n8k16, m16n8k32
// (s8, u8) and m8n8k32, m16n8k32, m16n8k64 (s4), on registers of random bits (sums that wrap, and that clamp under .satfinite).
constexpr int kImma = 40;
#define IMMA4(shape, mods, T1, T2)                                                                                                         \
  {                                                                                                                                         \
    u32 d0, d1, d2, d3;                                                                                                                     \
    asm volatile("mma.sync.aligned." shape ".row.col." mods "s32." T1 "." T2 ".s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" \
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                                                                   \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));          \
    o[k++] = d0; o[k++] = d1; o[k++] = d2; o[k++] = d3;                                                                                     \
  }
#define IMMA2(shape, mods, T1, T2)                                                                                                          \
  {                                                                                                                                         \
    u32 d0, d1, d2, d3;                                                                                                                     \
    asm volatile("mma.sync.aligned." shape ".row.col." mods "s32." T1 "." T2 ".s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"          \
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                                                                   \
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));                                            \
    o[k++] = d0; o[k++] = d1; o[k++] = d2; o[k++] = d3;                                                                                     \
  }
#define IMMA1(shape, mods, T1, T2)                                                                                                          \
  {                                                                                                                                         \
    u32 d0, d1;                                                                                                                             \
    asm volatile("mma.sync.aligned." shape ".row.col." mods "s32." T1 "." T2 ".s32 {%0,%1}, {%2}, {%3}, {%4,%5};"                           \
                 : "=r"(d0), "=r"(d1) : "r"(a[0]), "r"(b[0]), "r"(c[0]), "r"(c[1]));                                                        \
    o[k++] = d0; o[k++] = d1;                                                                                                               \
  }
__global__ void k_imma(const u32* A, const u32* Bv, const u32* C, u32* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  u32 a[4], b[2], c[4], o[kImma] = {};
  for (int j = 0; j < 4; ++j) { a[j] = A[((size_t)i * 4 + j) % n]; c[j] = C[((size_t)i * 4 + j) % n]; }
  for (int j = 0; j < 2; ++j) b[j] = Bv[((size_t)i * 2 + j) % n];
  int k = 0;
#if __CUDA_ARCH__ >= 800
  IMMA4("m16n8k32", "", "s8", "s8")
  IMMA4("m16n8k32", "", "u8", "u8")
  IMMA4("m16n8k32", "", "s8", "u8")
  IMMA4("m16n8k32", "satfinite.", "s8", "s8")
  IMMA4("m16n8k32", "satfinite.", "u8", "s8")
  IMMA2("m16n8k16", "", "s8", "s8")
  IMMA2("m16n8k16", "", "u8", "s8")
  IMMA2("m16n8k16", "satfinite.", "u8", "u8")
  IMMA4("m16n8k64", "", "s4", "s4")
  IMMA4("m16n8k64", "", "u4", "u4")
  IMMA4("m16n8k64", "satfinite.", "s4", "u4")
  IMMA2("m16n8k32", "", "s4", "s4")
  IMMA2("m16n8k32", "satfinite.", "u4", "u4")
#endif
  IMMA1("m8n8k16", "", "s8", "u8")
  IMMA1("m8n8k16", "satfinite.", "s8", "s8")
  IMMA1("m8n8k32", "", "s4", "s4")
  IMMA1("m8n8k32", "satfinite.", "u4", "s4")
  if (i < n)
    for (int j = 0; j < kImma; ++j) out[(size_t)i * kImma + j] = o[j];
}

// ldmatrix (LDSM): one, two and four 8x8 matrices of 16-bit elements, plain and transposed, from rows at addresses of the lanes' own (16-byte
// aligned, some rows shared between lanes, some past the first kilobyte). Whole warps only: ldmatrix takes the rows from all the lanes.
constexpr int kLdsm = 16;
__global__ void k_ldsm(const u32* A, u32* out, int n) {
  __shared__ __align__(16) u32 sm[1024];
  const int t = threadIdx.x;
  for (int j = t; j < 1024; j += blockDim.x) sm[j] = A[((size_t)blockIdx.x * 1024 + j) % n];
  __syncthreads();
  const u32 lane = t & 31;
  const u32 base = (u32)__cvta_generic_to_shared(sm);
  const u32 row = base + ((lane * 5 + blockIdx.x * 3 + (t >> 5)) % 64) * 16;
  u32 r[kLdsm] = {};
  int k = 0;
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r[0]) : "r"(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(r[1]) : "r"(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[2]), "=r"(r[3]) : "r"(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r[4]), "=r"(r[5]) : "r"(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]) : "r"(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]) : "r"(row));
  (void)k;
  for (int j = 0; j < kLdsm; ++j) out[((size_t)blockIdx.x * blockDim.x + t) * kLdsm + j] = r[j];
}

// The store gate (Runner::step_warp): a warp's first store to global memory ends its turn, once, so every warp of the block
// loads the word before any warp stores. Four warps each doing `*w += 1` end with 1, not 4, as on a card. The tier must
// stop at the store as the interpreter does.
__global__ void k_gate(u32* word, u32* seen) {
  if ((threadIdx.x & 31) == 0) {
    u32 v = *word;
    seen[threadIdx.x >> 5] = v;
    *word = v + 1;
  }
}

// ---- driver -------------------------------------------------------------------------------------
static u64 rng = 88172645463325252ull;
static u64 next_rand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static const u32 kSpecial32[] = {0u, 1u, 2u, 3u, 31u, 32u, 33u, 63u, 64u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xffffffffu, 0xfffffffeu,
                                 0xffffu, 0x10000u, 0xdeadbeefu, 0x55555555u, 0xaaaaaaaau, 0x00ff00ffu,
                                 // floats
                                 0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffc00001u, 0x7fffffffu,
                                 0x00000001u, 0x007fffffu, 0x80000001u, 0x807fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu, 0x4b000000u,
                                 0x33800000u, 0x3f800001u, 0x2f800000u, 0x5f800000u, 0x42280000u, 0xc2c80000u, 0x3dcccccdu, 0x40490fdbu,
                                 // NaNs of both signs with distinct payloads, the extremes of the narrow formats, and pairs of halves (f16 and bf16) that are NaN, infinite,
                                 // subnormal, zero of either sign, or the largest
                                 0x7fa00000u, 0xffa00001u, 0x7fc12345u, 0xffc54321u, 0x80000000u, 0x00400000u, 0x80400000u, 0x7e00fe01u,
                                 0x7c017e01u, 0xfe01fc00u, 0x7fff7fffu, 0x00010001u, 0x80018001u, 0x03ff03ffu, 0x80008000u, 0x3c003c00u,
                                 0x7f807fc0u, 0x7fc17f81u, 0xff80ffc0u, 0x00800080u, 0x7bff7bffu, 0xfbfffbffu, 0x7e007e00u, 0xfc00fc00u,
                                 0x7f7f7f7fu, 0x007f007fu, 0x807f807fu, 0x7fc0ff80u, 0x3f80bf80u, 0x4000c000u};
static const u64 kSpecial64[] = {0ull, 1ull, 0x8000000000000000ull, 0x7fffffffffffffffull, 0xffffffffffffffffull, 0x7ff0000000000000ull,
                                 0xfff0000000000000ull, 0x7ff8000000000000ull, 0x7ff0000000000001ull, 0x0000000000000001ull,
                                 0x000fffffffffffffull, 0x0010000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull,
                                 0x7fefffffffffffffull, 0x4330000000000000ull, 0x3cb0000000000000ull, 0x3ff0000000000001ull,
                                 0x00000000ffffffffull, 0xffffffff00000000ull, 0x4059000000000000ull, 0x3fb999999999999aull,
                                 // NaNs of both signs, quiet and signalling, with distinct payloads; subnormals; the edges of the integer conversions
                                 0xfff8000000000001ull, 0x7ff4000000000000ull, 0xfff0000000000007ull, 0x7ff8dead0000beefull, 0xfff4000000001234ull,
                                 0x43e0000000000000ull, 0xc3e0000000000000ull, 0x41e0000000000000ull, 0xc1e0000000000000ull, 0x8000000000000001ull};

static void fill32(std::vector<u32>& v, int salt) {
  const int m = sizeof kSpecial32 / sizeof kSpecial32[0];
  for (size_t i = 0; i < v.size(); ++i) {
    const u64 r = next_rand();
    switch ((i + salt) % 4) {
      case 0: v[i] = kSpecial32[(i * 7 + salt * 13 + (r >> 60)) % m]; break;
      case 1: v[i] = (u32)r; break;
      case 2: v[i] = kSpecial32[r % m] ^ ((r >> 32) & 1 ? 0u : 0x80000000u); break;
      default: v[i] = (u32)(r >> 32) & ((r & 8) ? 0xffffffffu : 0xffu); break;
    }
  }
}
static void fill64(std::vector<u64>& v, int salt) {
  const int m = sizeof kSpecial64 / sizeof kSpecial64[0];
  for (size_t i = 0; i < v.size(); ++i) {
    const u64 r = next_rand();
    switch ((i + salt) % 3) {
      case 0: v[i] = kSpecial64[(i * 5 + salt) % m]; break;
      case 1: v[i] = r; break;
      default: v[i] = (r & 1) ? kSpecial64[r % m] : (0x3ff0000000000000ull | (r >> 12)); break;   // normal doubles near 1
    }
  }
}

// Values for the tensor cores: pairs of halves that are finite and of moderate size as f16 pairs, as bf16 pairs and as floats alike, with an
// occasional special, so that the sums are mostly numbers.
static void fill_mma(std::vector<u32>& v, int salt) {
  const int m = sizeof kSpecial32 / sizeof kSpecial32[0];
  for (size_t i = 0; i < v.size(); ++i) {
    const u64 r = next_rand();
    const auto half = [](u64 q) -> u32 { return (u32)(((q >> 8) & 1) << 15 | (0x38 + (q % 16)) << 10 | ((q >> 16) & 0x3ff)); };
    u32 w = half(r) << 16 | half(r >> 32);
    if (((r >> 60) & 15) == (u64)(salt & 15)) w = kSpecial32[(r >> 20) % m];
    v[i] = w;
  }
}

static u64 hash_bytes(const void* p, size_t n) {
  u64 h = 1469598103934665603ull;
  const unsigned char* c = (const unsigned char*)p;
  for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
  return h;
}

template <class T>
static T* dev_copy(const std::vector<T>& h) {
  T* d;
  if (cudaMalloc(&d, h.size() * sizeof(T) + 64) || cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)) return nullptr;
  return d;
}

template <class T>
static int report(const char* name, T* d, size_t count) {
  std::vector<T> h(count);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost));
  std::printf("%-28s %016llx\n", name, (unsigned long long)hash_bytes(h.data(), count * sizeof(T)));
  return 0;
}

template <class T>
static int run_copy(const char* name, int n) {
  std::vector<T> h(n);
  for (int i = 0; i < n; ++i) {
    u64 x = next_rand();
    std::memcpy(&h[i], &x, sizeof(T) < 8 ? sizeof(T) : 8);
  }
  T* din = dev_copy(h);
  T* dout;
  CK(cudaMalloc(&dout, 2 * (size_t)n * sizeof(T) + 64));
  for (int limit : {32, 7, 1, 20})
    for (int stride : {1, 3}) {
      CK(cudaMemset(dout, 0, 2 * (size_t)n * sizeof(T)));
      k_copy<T><<<(n + 127) / 128, 128>>>(din, dout, n, limit, stride, stride == 1 ? 0 : 2);
      char label[64];
      std::snprintf(label, sizeof label, "copy %s n=%d lim=%d st=%d", name, n, limit, stride);
      if (report(label, dout, 2 * (size_t)n)) return 1;
    }
  CK(cudaFree(din));
  CK(cudaFree(dout));
  return 0;
}

// Signed sub-word elements widen on load (LDG.S8, LDS.S16, ...): the sign extension is the handlers' to get right.
template <class T>
__global__ void k_widen(const T* in, i32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T s[256];
  if (i < n) {
    s[threadIdx.x & 255] = in[i];
    out[i] = (i32)in[(i * 3) % n] + (i32)s[threadIdx.x & 255] * 2;
  }
  __syncthreads();
  if (i < n) out[n + i] = (i32)s[(threadIdx.x * 5) & 255];
}

static int run_signed(const char* name, int n) {
  std::vector<signed char> h8(n);
  std::vector<short> h16(n);
  for (int i = 0; i < n; ++i) {
    const u64 x = next_rand();
    h8[i] = (signed char)x;
    h16[i] = (short)(x >> 20);
  }
  signed char* d8 = dev_copy(h8);
  short* d16 = dev_copy(h16);
  i32* out;
  CK(cudaMalloc(&out, 2 * (size_t)n * 4 + 64));
  k_widen<signed char><<<(n + 127) / 128, 128>>>(d8, out, n);
  char label[64];
  std::snprintf(label, sizeof label, "widen %s s8 n=%d", name, n);
  if (report(label, out, 2 * (size_t)n)) return 1;
  k_widen<short><<<(n + 127) / 128, 128>>>(d16, out, n);
  std::snprintf(label, sizeof label, "widen %s s16 n=%d", name, n);
  if (report(label, out, 2 * (size_t)n)) return 1;
  CK(cudaFree(d8)); CK(cudaFree(d16)); CK(cudaFree(out));
  return 0;
}

template <class T>
static int run_shared(const char* name) {
  const int blocks = 3, threads = 192;
  std::vector<T> h(blocks * threads);
  for (size_t i = 0; i < h.size(); ++i) {
    u64 x = next_rand();
    std::memcpy(&h[i], &x, sizeof(T) < 8 ? sizeof(T) : 8);
  }
  T* din = dev_copy(h);
  T* dout;
  CK(cudaMalloc(&dout, h.size() * sizeof(T) + 64));
  for (int mul : {1, 3, 33, 129, 2}) {
    k_shared<T><<<blocks, threads>>>(din, dout, mul, mul & 3);
    char label[64];
    std::snprintf(label, sizeof label, "shared %s mul=%d", name, mul);
    if (report(label, dout, h.size())) return 1;
  }
  CK(cudaFree(din));
  CK(cudaFree(dout));
  return 0;
}

int main() {
  const int n = 4096 + 77;
  std::vector<u32> a(n), b(n), c(n);
  fill32(a, 0);
  fill32(b, 1);
  fill32(c, 2);
  u32 *da = dev_copy(a), *db = dev_copy(b), *dc = dev_copy(c);
  if (!da || !db || !dc) return 1;
  {
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * kInt * 4));
    for (int threads : {128, 48, 33}) {
      k_int<<<(n + threads - 1) / threads, threads>>>(da, db, dc, out, n, 0x9e3779b9u, 0x80000000u);
      char label[64];
      std::snprintf(label, sizeof label, "int threads=%d", threads);
      if (report(label, out, (size_t)n * kInt)) return 1;
    }
    CK(cudaFree(out));
  }
  {
    std::vector<u64> pv(n);
    fill64(pv, 3);
    u64* dp = dev_copy(pv);
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * 32 * 4));
    for (int threads : {128, 36}) {
      k_wide<<<(n + threads - 1) / threads, threads>>>(da, db, dc, dp, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "wide threads=%d", threads);
      if (report(label, out, (size_t)n * 32)) return 1;
    }
    CK(cudaFree(out)); CK(cudaFree(dp));
  }
  {
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * 4));
    for (int threads : {64, 40}) {
      k_branch<<<(n + threads - 1) / threads, threads>>>(da, db, dc, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "branch threads=%d", threads);
      if (report(label, out, n)) return 1;
    }
    CK(cudaFree(out));
  }
  {
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * kFlt * 4));
    for (int threads : {128, 50}) {
      k_float<<<(n + threads - 1) / threads, threads>>>(da, db, dc, out, n, 0x40490fdbu);
      char label[64];
      std::snprintf(label, sizeof label, "float threads=%d", threads);
      if (report(label, out, (size_t)n * kFlt)) return 1;
    }
    CK(cudaFree(out));
  }
  {   // the forms whose rules the card fixed (see k_forms2)
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * kF2 * 4));
    for (int threads : {128, 50}) {
      k_forms2<<<(n + threads - 1) / threads, threads>>>(da, db, dc, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "forms2 threads=%d", threads);
      if (report(label, out, (size_t)n * kF2)) return 1;
    }
    CK(cudaFree(out));
    CK(cudaMalloc(&out, (size_t)n * kH2 * 4));
    for (int threads : {128, 50}) {
      k_half2<<<(n + threads - 1) / threads, threads>>>(da, db, dc, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "half2 threads=%d", threads);
      if (report(label, out, (size_t)n * kH2)) return 1;
    }
    CK(cudaFree(out));
    CK(cudaMalloc(&out, (size_t)n * kM2 * 4));
    k_prmt2<<<(n + 127) / 128, 128>>>(da, db, dc, out, n);
    if (report("prmt2", out, (size_t)n * kM2)) return 1;
    CK(cudaFree(out));
  }
  {   // the tensor cores, on values that sum to numbers (HMMA)
    const int nm = 2048;
    std::vector<u32> ma(nm), mb(nm), mc(nm);
    fill_mma(ma, 1);
    fill_mma(mb, 2);
    fill_mma(mc, 3);
    u32 *dma = dev_copy(ma), *dmb = dev_copy(mb), *dmc = dev_copy(mc), *out;
    CK(cudaMalloc(&out, (size_t)nm * kMma * 4));
    for (int threads : {64, 128}) {
      k_mma<<<nm / threads, threads>>>(dma, dmb, dmc, out, nm);
      char label[64];
      std::snprintf(label, sizeof label, "mma threads=%d", threads);
      if (report(label, out, (size_t)nm * kMma)) return 1;
    }
    // (and on the general inputs, whose sums are NaNs and infinities)
    CK(cudaFree(out));
    CK(cudaMalloc(&out, (size_t)4096 * kMma * 4));
    k_mma<<<32, 128>>>(da, db, dc, out, 4096);
    if (report("mma special", out, (size_t)4096 * kMma)) return 1;
    CK(cudaFree(out)); CK(cudaFree(dma)); CK(cudaFree(dmb)); CK(cudaFree(dmc));
  }
  {   // ldmatrix
    u32* out;
    CK(cudaMalloc(&out, (size_t)16 * 128 * kLdsm * 4));
    for (int threads : {64, 128}) {
      k_ldsm<<<16, threads>>>(da, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "ldsm threads=%d", threads);
      if (report(label, out, (size_t)16 * threads * kLdsm)) return 1;
    }
    CK(cudaFree(out));
  }
  {   // the integer tensor cores
    const int nm = 2048;
    u32* out;
    CK(cudaMalloc(&out, (size_t)nm * kImma * 4));
    for (int threads : {64, 128}) {
      k_imma<<<nm / threads, threads>>>(da, db, dc, out, nm);
      char label[64];
      std::snprintf(label, sizeof label, "imma threads=%d", threads);
      if (report(label, out, (size_t)nm * kImma)) return 1;
    }
    CK(cudaFree(out));
  }
  {
    std::vector<u64> xa(n), xb(n), xc(n);
    fill64(xa, 0);
    fill64(xb, 1);
    fill64(xc, 2);
    u64 *dxa = dev_copy(xa), *dxb = dev_copy(xb), *dxc = dev_copy(xc), *out;
    CK(cudaMalloc(&out, (size_t)n * kDbl * 8));
    for (int threads : {64, 37}) {
      k_double<<<(n + threads - 1) / threads, threads>>>(dxa, dxb, dxc, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "double threads=%d", threads);
      if (report(label, out, (size_t)n * kDbl)) return 1;
    }
    CK(cudaFree(out));
    CK(cudaMalloc(&out, (size_t)n * kD2 * 8));
    {
      const double kd[4] = {1.5, -0.0, std::nan("0x123"), 3.25};
      CK(cudaMemcpyToSymbol(kdc, kd, sizeof kd));
    }
    for (int threads : {64, 37}) {
      k_double2<<<(n + threads - 1) / threads, threads>>>(dxa, dxb, dxc, out, n, -2.5);
      char label[64];
      std::snprintf(label, sizeof label, "double2 threads=%d", threads);
      if (report(label, out, (size_t)n * kD2)) return 1;
    }
    CK(cudaFree(dxa)); CK(cudaFree(dxb)); CK(cudaFree(dxc)); CK(cudaFree(out));
  }
  for (int m : {1, 31, 32, 33, 1000, 4096 + 5}) {
    if (run_copy<unsigned char>("u8", m) || run_copy<unsigned short>("u16", m) || run_copy<u32>("u32", m) ||
        run_copy<u64>("u64", m) || run_copy<Q>("q16", m) || run_signed("s8/s16", m))
      return 1;
  }
  if (run_shared<unsigned char>("u8") || run_shared<unsigned short>("u16") || run_shared<u32>("u32") ||
      run_shared<u64>("u64") || run_shared<Q>("q16"))
    return 1;
  {
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * 4));
    for (int threads : {64, 48}) {
      k_local<<<(n + threads - 1) / threads, threads>>>(da, db, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "local threads=%d", threads);
      if (report(label, out, n)) return 1;
    }
    u32 table[64];
    for (int i = 0; i < 64; ++i) table[i] = (u32)next_rand();
    CK(cudaMemcpyToSymbol(kc, table, sizeof table));
    k_const<<<(n + 127) / 128, 128>>>(da, out, n, 17u, 0xabcdu, 99u);
    if (report("const", out, n)) return 1;
    CK(cudaFree(out));
  }
  {
    u32 *g, *cnt;
    CK(cudaMalloc(&g, 1024 * 4));
    CK(cudaMalloc(&cnt, 64 * 4));
    CK(cudaMemset(g, 1, 1024 * 4));
    CK(cudaMemset(cnt, 0, 64 * 4));
    k_generic<<<4, 256>>>(g, cnt, da, n);
    if (report("generic", g, 1024) || report("generic atomics", cnt, 64)) return 1;
    CK(cudaFree(g)); CK(cudaFree(cnt));
  }
  {   // MATCH, atomics, REDUX, VOTEU, calls, WARPSYNC, BRA.DIV/.CONV (see k_direct)
    for (int threads : {128, 48, 33}) {
      const int blocks = (n + threads - 1) / threads;
      u32 *out, *bins;
      unsigned long long* bins64;
      float* binsf;
      double* binsd;
      CK(cudaMalloc(&out, ((size_t)n * kDir + (size_t)blocks * 64) * 4));
      CK(cudaMalloc(&bins, (size_t)blocks * 64 * 4));
      CK(cudaMalloc(&bins64, (size_t)blocks * 8 * 8));
      CK(cudaMalloc(&binsf, (size_t)blocks * 8 * 4));
      CK(cudaMalloc(&binsd, (size_t)blocks * 8 * 8));
      CK(cudaMemset(out, 0, ((size_t)n * kDir + (size_t)blocks * 64) * 4));
      CK(cudaMemset(bins, 0x11, (size_t)blocks * 64 * 4));
      CK(cudaMemset(bins64, 0x22, (size_t)blocks * 8 * 8));
      CK(cudaMemset(binsf, 0, (size_t)blocks * 8 * 4));
      CK(cudaMemset(binsd, 0, (size_t)blocks * 8 * 8));
      k_direct<<<blocks, threads>>>(da, db, out, bins, bins64, binsf, binsd, n);
      char label[64];
      std::snprintf(label, sizeof label, "direct threads=%d", threads);
      if (report(label, out, (size_t)n * kDir + (size_t)blocks * 64)) return 1;
      if (report("direct bins", bins, (size_t)blocks * 64) || report("direct bins64", bins64, (size_t)blocks * 8) ||
          report("direct binsf", binsf, (size_t)blocks * 8) || report("direct binsd", binsd, (size_t)blocks * 8))
        return 1;
      CK(cudaFree(out)); CK(cudaFree(bins)); CK(cudaFree(bins64)); CK(cudaFree(binsf)); CK(cudaFree(binsd));
    }
  }
  {
    u32 *word, *seen;
    CK(cudaMalloc(&word, 4));
    CK(cudaMalloc(&seen, 16 * 4));
    CK(cudaMemset(word, 0, 4));
    CK(cudaMemset(seen, 0xff, 16 * 4));
    k_gate<<<1, 128>>>(word, seen);
    if (report("store gate word", word, 1) || report("store gate seen", seen, 4)) return 1;
    u32 w = 0, s[4];
    CK(cudaMemcpy(&w, word, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s, seen, 16, cudaMemcpyDeviceToHost));
    std::printf("store gate: word=%u seen=%u,%u,%u,%u %s\n", w, s[0], s[1], s[2], s[3],
                w == 1 && !s[0] && !s[1] && !s[2] && !s[3] ? "ok" : "WRONG");
    CK(cudaFree(word)); CK(cudaFree(seen));
  }
  CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc));
  std::printf("done\n");
  return 0;
}
