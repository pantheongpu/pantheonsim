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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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
  // DFMA with a constant and a negated operand, DMUL by a constant, DADD with |x| and a constant.
  dout[i] = fma(d, 1.0000001, -d * 0.5) + fabs(d) * 0.25 - (d + 1.5) + __longlong_as_double((long long)(x >> 3));
}


// The wider forms of the 32-lane integer path: 64-bit adds with carries (IADD3.X), 64-bit compares (ISETP.EX), widening
// multiplies (IMAD.WIDE), funnel and 64-bit shifts (SHF), and boolean expressions of predicates (PLOP3). Lanes take
// different paths on `sel`, so the forms run with part of the warp active as well as with all of it.
struct WideIn { uint64_t a, b, c; uint32_t x, y, n; int32_t sx, sy; };
struct WideOut {
  uint64_t add3, sub, mulw, mulws, madw, mulc, fl, fr, flc, frc, shl, shr, sar;
  uint32_t flags;
};
__global__ void wide_forms(const WideIn* in, WideOut* out, int count, const uint64_t* base, uint32_t mod) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const WideIn v = in[i];
  WideOut o = {};
  o.add3 = v.a + v.b + v.c;
  o.sub = v.a - v.b;
  // IMAD.WIDE with an addend in a register pair, with a constant one, and the product alone.
  o.mulw = static_cast<uint64_t>(v.x) * v.y + v.c;
  o.mulws = static_cast<uint64_t>(static_cast<int64_t>(v.sx) * v.sy) + v.a;
  o.madw = static_cast<uint64_t>(v.x) * 0x9E3779B1u + 0x123456789abcdefull;
  o.mulc = base[i % 4] + static_cast<uint64_t>(v.y % mod) * 8;
  o.fl = __funnelshift_l(v.x, v.y, v.n);
  o.fr = __funnelshift_r(v.x, v.y, v.n);
  o.flc = __funnelshift_lc(v.x, v.y, v.n);
  o.frc = __funnelshift_rc(v.x, v.y, v.n);
  // PTX's own shifts: a count of 64 or more gives zero (shl, shr.u64) or the sign (shr.s64).
  asm("shl.b64 %0, %1, %2;" : "=l"(o.shl) : "l"(v.a), "r"(v.n));
  asm("shr.u64 %0, %1, %2;" : "=l"(o.shr) : "l"(v.a), "r"(v.n));
  asm("shr.s64 %0, %1, %2;" : "=l"(o.sar) : "l"(v.a), "r"(v.n));
  uint32_t f = 0;
  f |= (v.a < v.b) << 0;
  f |= (v.a <= v.b) << 1;
  f |= (v.a == v.b) << 2;
  f |= (v.a != v.b) << 3;
  f |= (static_cast<int64_t>(v.a) < static_cast<int64_t>(v.b)) << 4;
  f |= (static_cast<int64_t>(v.a) >= static_cast<int64_t>(v.b)) << 5;
  f |= (v.a > v.b) << 6;
  f |= (static_cast<int64_t>(v.a) > static_cast<int64_t>(v.c)) << 7;
  // Predicates combined: ptxas turns these into PLOP3 with different tables.
  const bool p = v.x < v.y, q = v.sx > v.sy, r = (v.n & 1) != 0, t = v.c > v.a;
  f |= ((p && q) || (r && !t)) << 8;
  f |= ((p ^ q) && (r || t)) << 9;
  f |= (!(p || q) && (r ^ t)) << 10;
  f |= ((p || q || r) && !(t && p)) << 11;
  o.flags = f;
  out[i] = o;
}


// POPC, BREV, FLO, SEL, IMAD.HI, and the warp-wide SHFL and VOTE, with lanes on different paths of a guard (the shuffles and
// votes name the lanes that are active). The first nine results are checked against the host; the others are compared
// between the two paths (and with the general path's counters) by the script.
constexpr int kLaneOps = 24;
__global__ void lane_ops(const uint32_t* in, uint32_t* out, int count, int shape) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t x = in[i], y = in[(i + 1) % count];
  const int32_t sx = static_cast<int32_t>(x), sy = static_cast<int32_t>(y);
  uint32_t o[kLaneOps] = {};
  o[0] = __popc(x);
  o[1] = __popc(~x);
  o[2] = __brev(x);
  o[3] = __clz(x);
  o[4] = __clz(sx);                       // FLO.SH on a signed value
  o[5] = (x & 8) ? y : x;                 // SEL
  o[6] = __umulhi(x, y);                  // IMAD.HI.U32
  o[7] = static_cast<uint32_t>(__mulhi(sx, sy));
  o[8] = x / 1000003u + static_cast<uint32_t>(sy / 7);
  // The lanes that get here are some of the warp's: a guard on the data.
  const bool take = shape == 0 ? true : shape == 1 ? (x & 3) != 0 : shape == 2 ? (i & 1) == 0 : (x >> 7 & 7) != 5;
  if (take) {
    const unsigned m = __activemask();
    o[9] = __shfl_sync(m, x, (y >> 3) & 31);
    o[10] = __shfl_down_sync(m, x, y & 7);
    o[11] = __shfl_up_sync(m, x, (y >> 5) & 7);
    o[12] = __shfl_xor_sync(m, x, (y >> 9) & 31);
    o[13] = __shfl_sync(m, x, (y >> 3) & 31, 8);
    o[14] = __shfl_down_sync(m, x, y & 3, 16);
    o[15] = __shfl_xor_sync(m, x, 5, 4);
    o[16] = __ballot_sync(m, x & 1);
    o[17] = __any_sync(m, (x & 6) == 6);
    o[18] = __all_sync(m, (x & 1) != 0 || (y & 1) != 0);
    o[19] = m;
    o[20] = __match_any_sync(m, x & 7);
    o[21] = __popc(__ballot_sync(m, sx < 0));
    o[22] = __shfl_sync(m, y, 3);              // a constant lane
    o[23] = __shfl_sync(m, x, i & 31, 32);
  }
  for (int k = 0; k < kLaneOps; ++k) out[static_cast<size_t>(i) * kLaneOps + k] = o[k];
}


// LEA in its forms: a shift and add (with a negated or zero second operand, whose carry out the 64-bit index arithmetic
// uses), the .HI forms that add the high word of a 64-bit index (.SX32 from a 32-bit signed one), and 64-bit negation.
__global__ void lea_forms(const uint32_t* in, uint64_t* out, uint64_t base, int pad, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t x = in[i], y = in[(i + 3) % count];
  const int32_t sx = static_cast<int32_t>(x);
  uint64_t o[10];
  o[0] = base + (static_cast<uint64_t>(x) << 3);
  o[1] = base + static_cast<uint64_t>(static_cast<int64_t>(sx) * 16);
  o[2] = (x << 5) + y;
  o[3] = static_cast<uint32_t>((x << 2) - y);
  o[4] = static_cast<uint64_t>(static_cast<int64_t>(sx) * 4 - pad);   // pad: zero or not, from the host
  o[5] = static_cast<uint64_t>(-static_cast<int64_t>(static_cast<uint64_t>(x) << 32 | y));
  o[6] = base - (static_cast<uint64_t>(y) << 4) + static_cast<uint64_t>(pad);
  o[7] = static_cast<uint64_t>(static_cast<int64_t>(sx) * 3 + static_cast<int64_t>(base));
  o[8] = (static_cast<uint64_t>(y) << 32 | x) * 8 + (base >> 3);
  o[9] = static_cast<uint64_t>(sx >> 3) + (static_cast<uint64_t>(x) << 33);
  const bool take = (i % 3) != 0;
  for (int k = 0; k < 10; ++k) if (take || k < 5) out[static_cast<size_t>(i) * 10 + k] = o[k];
}


// Atomics on global memory: one address for every lane, a few addresses, distinct ones, scattered ones; with the old value
// returned (ATOMG) and without (RED); add, min, max, exchange, compare-and-swap, increment, decrement, and, or, xor, in 32 and 64
// bits and in float and double. The sums, extremes and bit operations also have a host reference (any order gives the same).
constexpr int kAtomSlots = 64;
struct AtomMem {
  uint32_t add[kAtomSlots], umax[kAtomSlots], exch[kAtomSlots], cas[kAtomSlots], inc[kAtomSlots], dec[kAtomSlots];
  uint32_t band[kAtomSlots], bor[kAtomSlots], bxor[kAtomSlots], red[kAtomSlots];
  int32_t smin[kAtomSlots];
  unsigned long long add64[kAtomSlots], max64[kAtomSlots];
  float fadd[kAtomSlots];
  double dadd[kAtomSlots];
};
__global__ void atomic_forms(AtomMem* m, uint32_t* out, int count, int pattern) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const unsigned idx = pattern == 0 ? 0u : pattern == 1 ? i % 4u : pattern == 2 ? static_cast<unsigned>(i) % kAtomSlots : (i * 7u) % 33u;
  const bool take = pattern == 1 ? (i % 5) != 0 : pattern == 3 ? (i & 8) == 0 : true;
  uint32_t o[12] = {};
  if (take) {
    o[0] = atomicAdd(&m->add[idx], i + 1u);
    o[1] = atomicMax(&m->umax[idx], i * 2654435761u);
    o[2] = static_cast<uint32_t>(atomicMin(&m->smin[idx], static_cast<int>(i * 31) - 500));
    o[3] = atomicExch(&m->exch[idx], i);
    o[4] = atomicCAS(&m->cas[idx], static_cast<uint32_t>(i & 3), static_cast<uint32_t>(i + 100));
    o[5] = atomicInc(&m->inc[idx], 7u);
    o[6] = atomicDec(&m->dec[idx], 5u);
    o[7] = atomicAnd(&m->band[idx], ~(1u << (i & 31)));
    o[8] = atomicOr(&m->bor[idx], 1u << (i & 31));
    o[9] = atomicXor(&m->bxor[idx], i * 40503u);
    o[10] = static_cast<uint32_t>(atomicAdd(&m->add64[idx], static_cast<unsigned long long>(i) << 33));
    atomicMax(&m->max64[idx], static_cast<unsigned long long>(i) * 0x9E3779B97F4A7C15ull);
    atomicAdd(&m->fadd[idx], 0.5f * i);
    atomicAdd(&m->dadd[idx], 0.25 * i);
    atomicAdd(&m->red[idx], 3u);                 // a reduction, no old value
    o[11] = static_cast<uint32_t>(__float_as_uint(atomicAdd(&m->fadd[(idx + 1) % kAtomSlots], 1.0f)));
  }
  for (int k = 0; k < 12; ++k) out[static_cast<size_t>(i) * 12 + k] = o[k];
}


// Accesses that fault, one lane of the warp outside what it may reach. A fast path takes an access only when every active lane is
// inside; with one outside it must hand the instruction to the general path, which reports the fault as it always did. Each
// runs in a process of its own (a fault ends the context): `sass_fastpath <name>`.
__global__ void oob_local(uint32_t* out, int far) {
  uint32_t arr[256];
  const int t = threadIdx.x;
  for (int i = 0; i < 256; ++i) arr[i] = i * t;
  volatile int idx = (t == 7) ? far : t;
  out[t] = arr[idx];
}
__global__ void oob_gather(const uint32_t* in, uint32_t* out, long far) {
  const int t = threadIdx.x;
  out[t] = in[(t == 5) ? far : t];
}
__global__ void oob_scatter(uint32_t* out, long far) {
  const int t = threadIdx.x;
  out[(t == 9) ? far : t] = t;
}
__global__ void oob_atomic(uint32_t* out, long far) {
  const int t = threadIdx.x;
  atomicAdd(&out[(t == 3) ? far : t % 4], 1u);
}
__global__ void oob_shared(uint32_t* out, int far) {
  __shared__ uint32_t s[64];
  const int t = threadIdx.x;
  s[t] = t;
  __syncthreads();
  volatile int idx = (t == 11) ? far : t;
  out[t] = s[idx];
}
static int run_fault(const char* name) {
  uint32_t *a, *b;
  CK(cudaMalloc(&a, 4096));
  CK(cudaMalloc(&b, 4096));
  CK(cudaMemset(a, 0, 4096));
  CK(cudaMemset(b, 0, 4096));
  const std::string n = name;
  if (n == "local") oob_local<<<1, 32>>>(a, 1 << 22);
  else if (n == "gather") oob_gather<<<1, 32>>>(a, b, 1l << 34);
  else if (n == "scatter") oob_scatter<<<1, 32>>>(a, 1l << 34);
  else if (n == "atomic") oob_atomic<<<1, 32>>>(a, 1l << 34);
  else if (n == "shared") oob_shared<<<1, 32>>>(a, 1 << 20);
  else return 2;
  const cudaError_t e = cudaDeviceSynchronize();
  std::printf("fault %s: %s\n", name, cudaGetErrorString(e));
  return e == cudaSuccess ? 1 : 0;   // it must fault
}


// The f32 forms with a flush or a saturation, compares, MUFU, conversions and min/max, written as PTX so that ptxas gives each
// its own SASS (FFMA.FTZ, FADD.SAT, FMNMX.FTZ, MUFU.RCP, F2I.TRUNC, I2FP, FSETP...). The special values come from the host: zeros,
// subnormals, infinities, NaNs of both signs, the extremes. Only the two runs' agreement is checked for these (the kernels have
// their own host references in tests/unit/test_simd.cpp).
constexpr int kFloatForms = 36;
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
  }
  for (int k = 0; k < kFloatForms; ++k) out[static_cast<size_t>(i) * kFloatForms + k] = o[k];
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


// Loads where every active lane reads the same address (a broadcast), whole warps and partial ones, and loads of two addresses.
template <class T>
__global__ void bcast_load(const T* in, T* out, int n, int mode) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (mode == 1 && (threadIdx.x & 4)) return;
  const int j = mode == 2 ? (threadIdx.x & 8 ? 3 : 7) : blockIdx.x;
  out[i] = in[j];
}
template <class T>
static int run_bcast(const char* name) {
  const int n = 777;
  std::vector<T> h(64);
  for (size_t i = 0; i < h.size(); ++i) h[i] = mk<T>(i * 7919u + 3u);
  T *din, *dout;
  CK(cudaMalloc(&din, h.size() * sizeof(T)));
  CK(cudaMalloc(&dout, n * sizeof(T)));
  CK(cudaMemcpy(din, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  for (int mode = 0; mode < 3; ++mode) {
    CK(cudaMemset(dout, 0, n * sizeof(T)));
    bcast_load<T><<<(n + 63) / 64, 64>>>(din, dout, n, mode);
    CK(cudaDeviceSynchronize());
    std::vector<T> got(n), want(n);
    std::memset(want.data(), 0, n * sizeof(T));
    for (int i = 0; i < n; ++i) {
      const int t = i % 64, b = i / 64;
      if (mode == 1 && (t & 4)) continue;
      want[i] = h[mode == 2 ? ((t & 8) ? 3 : 7) : b];
    }
    CK(cudaMemcpy(got.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
    char what[64];
    std::snprintf(what, sizeof what, "bcast %s mode=%d", name, mode);
    same(got, want, what);
  }
  CK(cudaFree(din)); CK(cudaFree(dout));
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1) return run_fault(argv[1]);
  for (int n : {1, 31, 32, 33, 1000, 4096 + 5}) {
    if (run_copy<uint8_t>("u8", n) || run_copy<uint16_t>("u16", n) || run_copy<uint32_t>("u32", n) ||
        run_copy<uint64_t>("u64", n) || run_copy<float>("f32", n))
      return 1;
  }
  if (run_bcast<uint8_t>("u8") || run_bcast<uint16_t>("u16") || run_bcast<uint32_t>("u32") || run_bcast<uint64_t>("u64") || run_bcast<float>("f32"))
    return 1;
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

  {
    // wide_forms over edge and random values, with lanes taking different paths of a guard (the count is not a
    // multiple of 32, and the grid has partial warps).
    const uint64_t edge[] = {0ull, 1ull, 0x7fffffffull, 0x80000000ull, 0xffffffffull, 0x100000000ull, 0x7fffffffffffffffull,
                             0x8000000000000000ull, 0xffffffffffffffffull, 0xfffffffffffffffeull, 0x00000000ffff0000ull,
                             0xffff000000000000ull, 0x0000ffffffff0000ull, 0x5555555555555555ull, 0xaaaaaaaaaaaaaaaaull};
    const int ne = sizeof edge / sizeof edge[0];
    const int n = 1000 + 7;
    std::vector<WideIn> h(n);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    const auto next = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int i = 0; i < n; ++i) {
      WideIn w;
      const int style = i % 3;
      w.a = style == 0 ? next() : edge[next() % ne];
      w.b = style == 1 ? next() : edge[next() % ne];
      w.c = i % 4 == 0 ? w.a : style == 2 ? next() : edge[next() % ne];
      if (i % 5 == 0) w.b = w.a;               // equal operands: the low words decide
      if (i % 7 == 0) w.b = w.a ^ 0xffffffffull;   // equal high words
      if (i % 11 == 0) w.b = w.a + (1ull << 32);
      w.x = static_cast<uint32_t>(style == 0 ? next() : edge[next() % ne]);
      w.y = static_cast<uint32_t>(style == 1 ? next() : edge[next() % ne]);
      w.n = static_cast<uint32_t>(i % 6 == 0 ? next() : next() % 140);
      w.sx = static_cast<int32_t>(w.x);
      w.sy = static_cast<int32_t>(w.y);
      h[i] = w;
    }
    WideIn* din; WideOut* dout; uint64_t* dbase;
    CK(cudaMalloc(&din, n * sizeof(WideIn)));
    CK(cudaMalloc(&dout, n * sizeof(WideOut)));
    CK(cudaMalloc(&dbase, 4 * 8));
    const uint64_t base[4] = {0x1000ull, 0xffffffffffff0000ull, 0x7fffffff00000000ull, 0xfffffffffffffff8ull};
    CK(cudaMemcpy(din, h.data(), n * sizeof(WideIn), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbase, base, sizeof base, cudaMemcpyHostToDevice));
    for (int mod : {1000, 3}) {
      CK(cudaMemset(dout, 0, n * sizeof(WideOut)));
      wide_forms<<<(n + 63) / 64, 64>>>(din, dout, n, dbase, mod);
      CK(cudaDeviceSynchronize());
      std::vector<WideOut> got(n);
      CK(cudaMemcpy(got.data(), dout, n * sizeof(WideOut), cudaMemcpyDeviceToHost));
      int bad = 0;
      uint64_t hh = 1469598103934665603ull;
      for (int i = 0; i < n; ++i) {
        const WideIn& v = h[i];
        WideOut w = {};
        w.add3 = v.a + v.b + v.c;
        w.sub = v.a - v.b;
        w.mulw = static_cast<uint64_t>(v.x) * v.y + v.c;
        w.mulws = static_cast<uint64_t>(static_cast<int64_t>(v.sx) * v.sy) + v.a;
        w.madw = static_cast<uint64_t>(v.x) * 0x9E3779B1u + 0x123456789abcdefull;
        w.mulc = base[i % 4] + static_cast<uint64_t>(v.y % mod) * 8;
        const uint64_t cat = (static_cast<uint64_t>(v.y) << 32) | v.x;   // __funnelshift_*(lo = x, hi = y, n) work on hi:lo
        const uint32_t nl = v.n & 31, nlc = std::min<uint32_t>(v.n, 32);
        w.fl = static_cast<uint32_t>((cat << nl) >> 32);
        w.fr = static_cast<uint32_t>(cat >> nl);
        w.flc = static_cast<uint32_t>((cat << nlc) >> 32);
        w.frc = static_cast<uint32_t>(cat >> nlc);
        w.shl = v.n >= 64 ? 0 : v.a << v.n;
        w.shr = v.n >= 64 ? 0 : v.a >> v.n;
        w.sar = static_cast<uint64_t>(static_cast<int64_t>(v.a) >> std::min<uint32_t>(v.n, 63));
        uint32_t f = 0;
        const int64_t sa = static_cast<int64_t>(v.a), sb = static_cast<int64_t>(v.b), sc = static_cast<int64_t>(v.c);
        f |= (v.a < v.b) << 0; f |= (v.a <= v.b) << 1; f |= (v.a == v.b) << 2; f |= (v.a != v.b) << 3;
        f |= (sa < sb) << 4; f |= (sa >= sb) << 5; f |= (v.a > v.b) << 6; f |= (sa > sc) << 7;
        const bool p = v.x < v.y, q = v.sx > v.sy, r = (v.n & 1) != 0, t = v.c > v.a;
        f |= ((p && q) || (r && !t)) << 8; f |= ((p ^ q) && (r || t)) << 9;
        f |= (!(p || q) && (r ^ t)) << 10; f |= ((p || q || r) && !(t && p)) << 11;
        w.flags = f;
        if (std::memcmp(&w, &got[i], sizeof w) != 0) ++bad;
        const uint64_t* words = reinterpret_cast<const uint64_t*>(&got[i]);
        for (size_t k = 0; k < 13; ++k) hh = (hh ^ words[k]) * 1099511628211ull;
        hh = (hh ^ got[i].flags) * 1099511628211ull;
      }
      if (bad) { std::printf("WRONG wide_forms mod=%d %d of %d\n", mod, bad, n); ++wrong; }
      std::printf("wide_forms mod=%d bits %016llx\n", mod, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(din)); CK(cudaFree(dout)); CK(cudaFree(dbase));
  }

  {
    const int n = 32 * 9 + 5;
    std::vector<uint32_t> h(n);
    uint64_t s = 0x2545F4914F6CDD1Dull;
    const uint32_t edge[] = {0u, 1u, 0x7fffffffu, 0x80000000u, 0xffffffffu, 0x80000001u, 0x40000000u, 0xfffffffeu, 0x00010000u, 0xaaaaaaaau};
    for (int i = 0; i < n; ++i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      h[i] = i % 3 == 0 ? edge[s % 10] : i % 3 == 1 ? static_cast<uint32_t>(s) : static_cast<uint32_t>(s) >> (s >> 40 & 31);
    }
    uint32_t *din, *dout;
    CK(cudaMalloc(&din, n * 4));
    CK(cudaMalloc(&dout, static_cast<size_t>(n) * kLaneOps * 4));
    CK(cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice));
    for (int shape = 0; shape < 4; ++shape) {
      CK(cudaMemset(dout, 0, static_cast<size_t>(n) * kLaneOps * 4));
      lane_ops<<<(n + 63) / 64, 64>>>(din, dout, n, shape);
      CK(cudaDeviceSynchronize());
      std::vector<uint32_t> got(static_cast<size_t>(n) * kLaneOps);
      CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
      int bad = 0;
      uint64_t hh = 1469598103934665603ull;
      for (int i = 0; i < n; ++i) {
        const uint32_t x = h[i], y = h[(i + 1) % n];
        const int32_t sx = static_cast<int32_t>(x), sy = static_cast<int32_t>(y);
        uint32_t want[9];
        want[0] = __builtin_popcount(x);
        want[1] = __builtin_popcount(~x);
        want[2] = 0;
        for (int b = 0; b < 32; ++b) want[2] |= ((x >> b) & 1u) << (31 - b);
        want[3] = x ? __builtin_clz(x) : 32;
        want[4] = x ? __builtin_clz(x) : 32;
        want[5] = (x & 8) ? y : x;
        want[6] = static_cast<uint32_t>((static_cast<uint64_t>(x) * y) >> 32);
        want[7] = static_cast<uint32_t>((static_cast<int64_t>(sx) * sy) >> 32);
        want[8] = x / 1000003u + static_cast<uint32_t>(sy / 7);
        for (int k = 0; k < 9; ++k)
          if (got[static_cast<size_t>(i) * kLaneOps + k] != want[k]) ++bad;
        for (int k = 0; k < kLaneOps; ++k) hh = (hh ^ got[static_cast<size_t>(i) * kLaneOps + k]) * 1099511628211ull;
      }
      if (bad) { std::printf("WRONG lane_ops shape=%d %d\n", shape, bad); ++wrong; }
      std::printf("lane_ops shape=%d bits %016llx\n", shape, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(din)); CK(cudaFree(dout));
  }

  {
    const int n = 32 * 7 + 9;
    std::vector<uint32_t> h(n);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    const uint32_t edge[] = {0u, 1u, 0x7fffffffu, 0x80000000u, 0xffffffffu, 0x80000001u, 0x40000000u, 0xfffffffeu, 0x00010000u, 3u};
    for (int i = 0; i < n; ++i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      h[i] = i % 2 == 0 ? edge[s % 10] : static_cast<uint32_t>(s);
    }
    uint32_t* din; uint64_t* dout;
    CK(cudaMalloc(&din, n * 4));
    CK(cudaMalloc(&dout, static_cast<size_t>(n) * 10 * 8));
    CK(cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice));
    for (uint64_t base : {0x7f0000001000ull, 0xffffffff00000008ull, 0ull})
      for (int pad : {0, 5, -7}) {
        CK(cudaMemset(dout, 0, static_cast<size_t>(n) * 10 * 8));
        lea_forms<<<(n + 63) / 64, 64>>>(din, dout, base, pad, n);
        CK(cudaDeviceSynchronize());
        std::vector<uint64_t> got(static_cast<size_t>(n) * 10);
        CK(cudaMemcpy(got.data(), dout, got.size() * 8, cudaMemcpyDeviceToHost));
        int bad = 0;
        uint64_t hh = 1469598103934665603ull;
        for (int i = 0; i < n; ++i) {
          const uint32_t x = h[i], y = h[(i + 3) % n];
          const int32_t sx = static_cast<int32_t>(x);
          uint64_t w[10] = {};
          w[0] = base + (static_cast<uint64_t>(x) << 3);
          w[1] = base + static_cast<uint64_t>(static_cast<int64_t>(sx) * 16);
          w[2] = static_cast<uint32_t>((x << 5) + y);
          w[3] = static_cast<uint32_t>((x << 2) - y);
          w[4] = static_cast<uint64_t>(static_cast<int64_t>(sx) * 4 - pad);
          w[5] = static_cast<uint64_t>(-static_cast<int64_t>(static_cast<uint64_t>(x) << 32 | y));
          w[6] = base - (static_cast<uint64_t>(y) << 4) + static_cast<uint64_t>(pad);
          w[7] = static_cast<uint64_t>(static_cast<int64_t>(sx) * 3 + static_cast<int64_t>(base));
          w[8] = (static_cast<uint64_t>(y) << 32 | x) * 8 + (base >> 3);
          w[9] = static_cast<uint64_t>(sx >> 3) + (static_cast<uint64_t>(x) << 33);
          const bool take = (i % 3) != 0;
          for (int k = 0; k < 10; ++k) {
            const uint64_t want = (take || k < 5) ? w[k] : 0;
            if (got[static_cast<size_t>(i) * 10 + k] != want) ++bad;
            hh = (hh ^ got[static_cast<size_t>(i) * 10 + k]) * 1099511628211ull;
          }
        }
        if (bad) { std::printf("WRONG lea_forms base=%llx pad=%d: %d\n", static_cast<unsigned long long>(base), pad, bad); ++wrong; }
        std::printf("lea_forms base=%llx pad=%d bits %016llx\n", static_cast<unsigned long long>(base), pad, static_cast<unsigned long long>(hh));
      }
    CK(cudaFree(din)); CK(cudaFree(dout));
  }

  {
    const int n = 32 * 5 + 11;
    AtomMem* dm; uint32_t* dout;
    CK(cudaMalloc(&dm, sizeof(AtomMem)));
    CK(cudaMalloc(&dout, static_cast<size_t>(n) * 12 * 4));
    for (int pattern = 0; pattern < 4; ++pattern) {
      AtomMem init = {};
      for (int k = 0; k < kAtomSlots; ++k) {
        init.smin[k] = 1000;
        init.band[k] = 0xffffffffu;
        init.dec[k] = 3;
        init.inc[k] = 2;
        init.cas[k] = k & 1;
      }
      CK(cudaMemcpy(dm, &init, sizeof init, cudaMemcpyHostToDevice));
      CK(cudaMemset(dout, 0, static_cast<size_t>(n) * 12 * 4));
      atomic_forms<<<(n + 63) / 64, 64>>>(dm, dout, n, pattern);
      CK(cudaDeviceSynchronize());
      AtomMem got;
      std::vector<uint32_t> gout(static_cast<size_t>(n) * 12);
      CK(cudaMemcpy(&got, dm, sizeof got, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(gout.data(), dout, gout.size() * 4, cudaMemcpyDeviceToHost));
      AtomMem want = init;   // the order-independent ones
      for (int i = 0; i < n; ++i) {
        const unsigned idx = pattern == 0 ? 0u : pattern == 1 ? i % 4u : pattern == 2 ? static_cast<unsigned>(i) % kAtomSlots : (i * 7u) % 33u;
        const bool take = pattern == 1 ? (i % 5) != 0 : pattern == 3 ? (i & 8) == 0 : true;
        if (!take) continue;
        want.add[idx] += i + 1u;
        want.umax[idx] = std::max(want.umax[idx], static_cast<uint32_t>(i * 2654435761u));
        want.smin[idx] = std::min(want.smin[idx], static_cast<int>(i * 31) - 500);
        want.band[idx] &= ~(1u << (i & 31));
        want.bor[idx] |= 1u << (i & 31);
        want.bxor[idx] ^= i * 40503u;
        want.add64[idx] += static_cast<unsigned long long>(i) << 33;
        want.max64[idx] = std::max(want.max64[idx], static_cast<unsigned long long>(i) * 0x9E3779B97F4A7C15ull);
        want.fadd[idx] += 0.5f * i;
        want.dadd[idx] += 0.25 * i;
        want.red[idx] += 3u;
        want.fadd[(idx + 1) % kAtomSlots] += 1.0f;
      }
      int bad = 0;
      for (int k = 0; k < kAtomSlots; ++k) {
        bad += got.add[k] != want.add[k] || got.umax[k] != want.umax[k] || got.smin[k] != want.smin[k] || got.band[k] != want.band[k] ||
               got.bor[k] != want.bor[k] || got.bxor[k] != want.bxor[k] || got.add64[k] != want.add64[k] ||
               got.max64[k] != want.max64[k] || got.fadd[k] != want.fadd[k] || got.dadd[k] != want.dadd[k] || got.red[k] != want.red[k];
      }
      if (bad) { std::printf("WRONG atomic_forms pattern=%d: %d slots\n", pattern, bad); ++wrong; }
      // Printed so that the two runs are compared on every returned value and on the order-dependent memory too.
      uint64_t hh = 1469598103934665603ull;
      for (uint32_t v : gout) hh = (hh ^ v) * 1099511628211ull;
      const uint32_t* mw = reinterpret_cast<const uint32_t*>(&got);
      for (size_t k = 0; k < sizeof got / 4; ++k) hh = (hh ^ mw[k]) * 1099511628211ull;
      std::printf("atomic_forms pattern=%d bits %016llx\n", pattern, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(dm)); CK(cudaFree(dout));
  }

  {
    // float_forms over the special values and random bits (the NaNs include both signs and payloads).
    const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
                           0x3f800000u, 0xbf800000u, 0x3f000000u, 0x40000000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
                           0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0x7f800001u, 0xffbfffffu, 0x3effffffu, 0x4b000000u, 0x4b800001u,
                           0x33800000u, 0x3f800001u, 0x4f000000u, 0xcf000000u, 0x4f800000u, 0x4effffffu, 0x3f7fffffu, 0x40490fdbu};
    const int ns = sizeof sp / sizeof sp[0];
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
    for (int shape = 0; shape < 3; ++shape) {
      CK(cudaMemset(dout, 0, static_cast<size_t>(n) * kFloatForms * 4));
      float_forms<<<(n + 63) / 64, 64>>>(din, dout, n, shape);
      CK(cudaDeviceSynchronize());
      std::vector<uint32_t> got(static_cast<size_t>(n) * kFloatForms);
      CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
      uint64_t hh = 1469598103934665603ull;
      for (uint32_t v : got) hh = (hh ^ v) * 1099511628211ull;
      std::printf("float_forms shape=%d bits %016llx\n", shape, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(din)); CK(cudaFree(dout));
  }
  std::printf("wrong: %d\n", wrong);
  return wrong != 0;
}
