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
// Round 9 added: the conversions (F2F, FRND, F2I and I2F in every size and rounding mode), packed f16 compares (HSET2, HSETP2), bf16
// HFMA2, MUFU.RCP64H/RSQ64H, IMNMX, IABS, FCHK, P2R and the atomics on shared memory. `sass_fastpath forms` prints one hash per
// conv_forms form, `sass_fastpath dump <form>` the lanes of one, for comparing with a card.
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


// The same atomics on the block's shared memory (ATOMS): the lanes of a warp on one address, a few, distinct ones, scattered ones;
// with the old value returned. The memory starts as the host's `init`, and each block's final memory is copied out for the check.
__global__ void shared_atomic_forms(const AtomMem* init, AtomMem* fin, uint32_t* out, int count, int pattern) {
  __shared__ AtomMem sm;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(init);
  uint32_t* dst = reinterpret_cast<uint32_t*>(&sm);
  for (unsigned k = threadIdx.x; k < sizeof(AtomMem) / 4; k += blockDim.x) dst[k] = src[k];
  __syncthreads();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t o[12] = {};
  if (i < count) {
    const unsigned idx = pattern == 0 ? 0u : pattern == 1 ? i % 4u : pattern == 2 ? static_cast<unsigned>(i) % kAtomSlots : (i * 7u) % 33u;
    const bool take = pattern == 1 ? (i % 5) != 0 : pattern == 3 ? (i & 8) == 0 : true;
    if (take) {
      o[0] = atomicAdd(&sm.add[idx], i + 1u);
      o[1] = atomicMax(&sm.umax[idx], i * 2654435761u);
      o[2] = static_cast<uint32_t>(atomicMin(&sm.smin[idx], static_cast<int>(i * 31) - 500));
      o[3] = atomicExch(&sm.exch[idx], i);
      o[4] = atomicCAS(&sm.cas[idx], static_cast<uint32_t>(i & 3), static_cast<uint32_t>(i + 100));
      o[5] = atomicInc(&sm.inc[idx], 7u);
      o[6] = atomicDec(&sm.dec[idx], 5u);
      o[7] = atomicAnd(&sm.band[idx], ~(1u << (i & 31)));
      o[8] = atomicOr(&sm.bor[idx], 1u << (i & 31));
      o[9] = atomicXor(&sm.bxor[idx], i * 40503u);
      o[10] = static_cast<uint32_t>(atomicAdd(&sm.add64[idx], static_cast<unsigned long long>(i) << 33));
      atomicMax(&sm.max64[idx], static_cast<unsigned long long>(i) * 0x9E3779B97F4A7C15ull);
      atomicAdd(&sm.fadd[idx], 0.5f * i);
      atomicAdd(&sm.dadd[idx], 0.25 * i);
      atomicAdd(&sm.red[idx], 3u);
      o[11] = static_cast<uint32_t>(__float_as_uint(atomicAdd(&sm.fadd[(idx + 1) % kAtomSlots], 1.0f)));
    }
    for (int k = 0; k < 12; ++k) out[static_cast<size_t>(i) * 12 + k] = o[k];
  }
  __syncthreads();
  const uint32_t* from = reinterpret_cast<const uint32_t*>(&sm);
  uint32_t* to = reinterpret_cast<uint32_t*>(&fin[blockIdx.x]);
  for (unsigned k = threadIdx.x; k < sizeof(AtomMem) / 4; k += blockDim.x) to[k] = from[k];
}


// More atomics: signed 64-bit min and max, a constant-one add with its old value (ATOMS.POPC.INC on shared memory), the same without
// (RED), on global and on shared memory, over the address patterns above. The results are order-independent, so the host checks them.
struct AtomMore {
  long long smin[kAtomSlots], smax[kAtomSlots];
  uint32_t one[kAtomSlots], two[kAtomSlots];
  int32_t smax32[kAtomSlots];
  double dmin_bits[kAtomSlots];   // (a double target for the F64 add of a negative and a positive: the NaN-free rule)
};
__device__ __forceinline__ void atomic_more_body(AtomMore& m, uint32_t* o, int i, unsigned idx) {
  o[0] = static_cast<uint32_t>(atomicMin(&m.smin[idx], static_cast<long long>(i) * 31 - 500));
  o[1] = static_cast<uint32_t>(atomicMax(&m.smax[idx], 1000 - static_cast<long long>(i) * 40503) >> 16);
  o[2] = atomicAdd(&m.one[idx], 1u);
  atomicAdd(&m.two[idx], 1u);
  o[3] = static_cast<uint32_t>(atomicMax(&m.smax32[idx], 77 - i * 5));
  atomicAdd(&m.dmin_bits[idx], (i & 1) ? -0.5 * i : 0.125 * i);
}
__global__ void atomic_more(AtomMore* m, uint32_t* out, int count, int pattern) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const unsigned idx = pattern == 0 ? 0u : pattern == 1 ? i % 4u : pattern == 2 ? static_cast<unsigned>(i) % kAtomSlots : (i * 7u) % 33u;
  const bool take = pattern == 1 ? (i % 5) != 0 : pattern == 3 ? (i & 8) == 0 : true;
  uint32_t o[4] = {};
  if (take) atomic_more_body(*m, o, i, idx);
  for (int k = 0; k < 4; ++k) out[static_cast<size_t>(i) * 4 + k] = o[k];
}
__global__ void shared_atomic_more(const AtomMore* init, AtomMore* fin, uint32_t* out, int count, int pattern) {
  __shared__ AtomMore sm;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(init);
  uint32_t* dst = reinterpret_cast<uint32_t*>(&sm);
  for (unsigned k = threadIdx.x; k < sizeof(AtomMore) / 4; k += blockDim.x) dst[k] = src[k];
  __syncthreads();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t o[4] = {};
  if (i < count) {
    const unsigned idx = pattern == 0 ? 0u : pattern == 1 ? i % 4u : pattern == 2 ? static_cast<unsigned>(i) % kAtomSlots : (i * 7u) % 33u;
    const bool take = pattern == 1 ? (i % 5) != 0 : pattern == 3 ? (i & 8) == 0 : true;
    if (take) atomic_more_body(sm, o, i, idx);
    for (int k = 0; k < 4; ++k) out[static_cast<size_t>(i) * 4 + k] = o[k];
  }
  __syncthreads();
  const uint32_t* from = reinterpret_cast<const uint32_t*>(&sm);
  uint32_t* to = reinterpret_cast<uint32_t*>(&fin[blockIdx.x]);
  for (unsigned k = threadIdx.x; k < sizeof(AtomMore) / 4; k += blockDim.x) to[k] = from[k];
}

// Loads and stores of 1, 2, 4 and 8 bytes on host-mapped memory (cudaHostAlloc, mapped: zero-copy), whole and partial warps, consecutive,
// strided and reversed lanes, and a broadcast. The host checks what the kernel wrote into its own bytes.
__global__ void hostmap_forms(const uint8_t* a8, uint8_t* b8, const uint16_t* a16, uint16_t* b16, const uint32_t* a32, uint32_t* b32,
                              const uint64_t* a64, uint64_t* b64, int n, int pattern) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (pattern == 3 && (i % 5) == 0) return;
  const int j = pattern == 1 ? (i * 7) % n : pattern == 2 ? n - 1 - i : i;
  b8[j] = static_cast<uint8_t>(a8[j] + 1);
  b16[j] = static_cast<uint16_t>(a16[j] ^ 0x5a5a);
  b32[j] = a32[j] * 3u + 1u + a32[0];
  b64[j] = a64[j] + a32[i] + (static_cast<uint64_t>(a32[0]) << 32);
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
__global__ void oob_shared_atomic(uint32_t* out, int far) {
  __shared__ uint32_t s[64];
  const int t = threadIdx.x;
  s[t] = t;
  __syncthreads();
  volatile int idx = (t == 11) ? far : t % 8;
  out[t] = atomicAdd(&s[idx], 1u);
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
  else if (n == "sharedatomic") oob_shared_atomic<<<1, 32>>>(a, 1 << 20);
  else return 2;
  const cudaError_t e = cudaDeviceSynchronize();
  std::printf("fault %s: %s\n", name, cudaGetErrorString(e));
  return e == cudaSuccess ? 1 : 0;   // it must fault
}


// The f32 forms with a flush or a saturation, compares, MUFU, conversions and min/max, written as PTX so that ptxas gives each
// its own SASS (FFMA.FTZ, FADD.SAT, FMNMX.FTZ, MUFU.RCP, F2I.TRUNC, I2FP, FSETP...). The special values come from the host: zeros,
// subnormals, infinities, NaNs of both signs, the extremes. Only the two runs' agreement is checked for these (the kernels have
// their own host references in tests/unit/test_simd.cpp).
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

// Conversions and packed compares the 32-lane path takes over from the per-lane one (F2F, FRND, F2I from f16 and f64 and to 8, 16
// and 64 bits, I2F from 64 bits and with a rounding mode, to f16, bf16 and f64, HSET2, bf16 HADD2/HMUL2/HFMA2, MUFU.RCP64H and
// RSQ64H), each written so that ptxas gives it its own SASS. The inputs are special values from the host (zeros, subnormals, the
// extremes, infinities, NaNs of both signs, the edges of the 32-, 64-bit and 16-bit integer ranges) mixed with random bits; the
// lanes take part under three different guards. The results of each form are hashed separately so that a difference between the
// two runs names the form (and so that a card run names the form it disagrees on).
constexpr int kConvForms = 209;
__device__ __forceinline__ uint32_t lo32_(uint64_t v) { return static_cast<uint32_t>(v); }
__device__ __forceinline__ uint32_t hi32_(uint64_t v) { return static_cast<uint32_t>(v >> 32); }
__global__ void conv_forms(const float* fin, const double* din, const uint32_t* win, const uint64_t* lin, uint32_t* out, int n, int shape) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = fin[i], b = fin[(i + 1) % n];
  const double p = din[i], q = din[(i + 7) % n];
  const uint32_t x = win[i], y = win[(i + 3) % n];
  const uint64_t u = lin[i];
  const long long sl = static_cast<long long>(u);
  const unsigned long long ul = u;
  uint32_t o[kConvForms] = {};
  int k = 0;
  const bool take = shape == 0 ? true : shape == 1 ? (__float_as_uint(b) & 3) != 0 : (i & 4) == 0;
  if (take) {
#define O32(e) o[k++] = static_cast<uint32_t>(e)
#define O64(e) { const uint64_t v_ = static_cast<uint64_t>(e); o[k++] = lo32_(v_); o[k++] = hi32_(v_); }
#define OF(e) o[k++] = __float_as_uint(e)
#define OD(e) O64(__double_as_longlong(e))
#define OH(e) o[k++] = __half_as_ushort(e)
#define OB(e) o[k++] = __bfloat16_as_ushort(e)
    const __half ha = __ushort_as_half(static_cast<unsigned short>(x)), hb = __ushort_as_half(static_cast<unsigned short>(x >> 16));
    const __nv_bfloat16 ba = __ushort_as_bfloat16(static_cast<unsigned short>(y)), bb = __ushort_as_bfloat16(static_cast<unsigned short>(y >> 16));
    // F2F
    OH(__float2half_rn(a)); OH(__float2half_rz(a)); OH(__float2half_rd(b)); OH(__float2half_ru(b));
    OF(__half2float(ha)); OF(__half2float(hb));
    OB(__float2bfloat16_rn(a)); OB(__float2bfloat16_rz(a)); OB(__float2bfloat16_rd(b)); OB(__float2bfloat16_ru(b));
    OF(__double2float_rn(p)); OF(__double2float_rz(p)); OF(__double2float_rd(q)); OF(__double2float_ru(q));
    OD(static_cast<double>(a)); OD(static_cast<double>(__half2float(ha)));
    OH(__double2half(p)); OB(__double2bfloat16(q));
    OF(__double2float_rz(-fabs(p))); OD(static_cast<double>(-fabsf(b)));
    // FRND (f32, f16, f64; the four roundings; a flush)
    OF(floorf(a)); OF(ceilf(a)); OF(truncf(a)); OF(rintf(b)); OF(nearbyintf(-b));
    { float r; asm("cvt.rzi.ftz.f32.f32 %0, %1;" : "=f"(r) : "f"(a)); OF(r); asm("cvt.rmi.ftz.f32.f32 %0, %1;" : "=f"(r) : "f"(-b)); OF(r); }
    OD(floor(p)); OD(ceil(p)); OD(trunc(q)); OD(rint(p)); OD(floor(-fabs(q)));
    OH(hfloor(ha)); OH(hceil(hb)); OH(htrunc(ha)); OH(hrint(hb));
    // F2I: to 8, 16, 32 and 64 bits, from f32, f16 and f64, signed and unsigned, in each rounding mode
    O32(__double2int_rn(p)); O32(__double2int_rz(p)); O32(__double2int_rd(q)); O32(__double2int_ru(q));
    O32(__double2uint_rn(p)); O32(__double2uint_rz(q)); O32(__double2uint_rd(p)); O32(__double2uint_ru(q));
    O64(__double2ll_rn(p)); O64(__double2ll_rz(p)); O64(__double2ll_rd(q)); O64(__double2ll_ru(q));
    O64(__double2ull_rn(p)); O64(__double2ull_rz(q)); O64(__double2ull_rd(p)); O64(__double2ull_ru(q));
    O64(__float2ll_rn(a)); O64(__float2ll_rz(b)); O64(__float2ll_rd(a)); O64(__float2ll_ru(b));
    O64(__float2ull_rn(a)); O64(__float2ull_rz(b)); O64(__float2ull_rd(b)); O64(__float2ull_ru(a));
    O32(__half2int_rn(ha)); O32(__half2int_rz(hb)); O32(__half2uint_rd(ha)); O32(__half2uint_ru(hb));
    O32(__half2short_rn(ha)); O32(__half2ushort_rz(hb)); O32(__half2short_rd(hb));
    O64(__half2ll_rn(ha)); O64(__half2ull_rz(hb));
    { unsigned short r;
      asm("cvt.rzi.s16.f32 %0, %1;" : "=h"(r) : "f"(a)); O32(r);
      asm("cvt.rni.u16.f32 %0, %1;" : "=h"(r) : "f"(b)); O32(r);
      asm("cvt.rmi.s16.f64 %0, %1;" : "=h"(r) : "d"(p)); O32(r);
      asm("cvt.rpi.u16.f64 %0, %1;" : "=h"(r) : "d"(q)); O32(r); }
    O32(__float2int_rd(a)); O32(__float2uint_ru(b)); O32(__float2int_rn(-b));
    // I2F: from 64 bits, with each rounding mode, to f32, f64, f16 and bf16; from 8, 16 and 32 bits to f16, bf16 and f64
    OF(__ll2float_rn(sl)); OF(__ll2float_rz(sl)); OF(__ll2float_rd(sl)); OF(__ll2float_ru(sl));
    OF(__ull2float_rn(ul)); OF(__ull2float_rz(ul)); OF(__ull2float_rd(ul)); OF(__ull2float_ru(ul));
    OD(__ll2double_rn(sl)); OD(__ll2double_rz(sl)); OD(__ll2double_rd(sl)); OD(__ll2double_ru(sl));
    OD(__ull2double_rn(ul)); OD(__ull2double_rz(ul)); OD(__ull2double_rd(ul)); OD(__ull2double_ru(ul));
    OF(__int2float_rz(static_cast<int>(x))); OF(__int2float_rd(static_cast<int>(y))); OF(__int2float_ru(static_cast<int>(x)));
    OF(__uint2float_rz(y)); OF(__uint2float_rd(x)); OF(__uint2float_ru(y));
    OD(__int2double_rn(static_cast<int>(x))); OD(__uint2double_rn(y));
    OH(__int2half_rn(static_cast<int>(x))); OH(__int2half_rz(static_cast<int>(y))); OH(__uint2half_rd(x)); OH(__uint2half_ru(y));
    OH(__ll2half_rn(sl)); OH(__ull2half_rz(ul)); OH(__short2half_rn(static_cast<short>(x))); OH(__ushort2half_rd(static_cast<unsigned short>(y)));
    OB(__int2bfloat16_rn(static_cast<int>(x))); OB(__uint2bfloat16_rz(y)); OB(__ll2bfloat16_rd(sl)); OB(__ull2bfloat16_ru(ul));
    { float r;
      asm("cvt.rz.f32.s16 %0, %1;" : "=f"(r) : "h"(static_cast<short>(x))); OF(r);
      asm("cvt.rp.f32.u8 %0, %1;" : "=f"(r) : "h"(static_cast<unsigned short>(y & 0xff))); OF(r); }
    // HSET2 / HSETP2: the packed compares for halves and bfloat16 pairs, as 1.0 / 0.0 masks and as one boolean
    const __half2 h2a = __halves2half2(ha, hb), h2b = __halves2half2(hb, __ushort_as_half(static_cast<unsigned short>(y)));
    const __nv_bfloat162 b2a = __halves2bfloat162(ba, bb), b2b = __halves2bfloat162(bb, __ushort_as_bfloat16(static_cast<unsigned short>(x)));
    const auto hb2 = [](__half2 h) { return *reinterpret_cast<const uint32_t*>(&h); };
    const auto bb2 = [](__nv_bfloat162 h) { return *reinterpret_cast<const uint32_t*>(&h); };
    O32(hb2(__heq2(h2a, h2b))); O32(hb2(__hne2(h2a, h2b))); O32(hb2(__hlt2(h2a, h2b))); O32(hb2(__hle2(h2a, h2b)));
    O32(hb2(__hgt2(h2a, h2b))); O32(hb2(__hge2(h2a, h2b))); O32(hb2(__hequ2(h2a, h2b))); O32(hb2(__hneu2(h2a, h2b)));
    O32(hb2(__hltu2(h2a, h2b))); O32(hb2(__hgeu2(h2a, h2b))); O32(hb2(__hisnan2(h2a)));
    O32(hb2(__hlt2(__low2half2(h2a), __hneg2(h2b)))); O32(hb2(__hgt2(__habs2(h2a), __high2half2(h2b))));
    O32(__hbeq2(h2a, h2b)); O32(__hbltu2(h2a, h2b)); O32(__hbge2(h2b, h2a)); O32(__hbne2(h2a, h2b));
    O32(bb2(__heq2(b2a, b2b))); O32(bb2(__hlt2(b2a, b2b))); O32(bb2(__hgeu2(b2a, b2b))); O32(bb2(__hne2(b2a, b2b)));
    O32(bb2(__hgt2(__low2bfloat162(b2a), __hneg2(b2b)))); O32(__hbeq2(b2a, b2b)); O32(__hblt2(b2b, b2a));
    // HSETP2: two predicates from a packed compare (PTX's setp on f16x2), with a boolean combine and a flush
    {
      const uint32_t xa = hb2(h2a), xb = hb2(h2b), xc = static_cast<uint32_t>(x ^ y);
      uint32_t t;
#define HSETP(OPS) asm("{ .reg .pred p, q, c; .reg .u32 t1, t2; setp.ne.u32 c, %3, 0; setp." OPS " p|q, %1, %2; " \
                       "selp.u32 t1, 1, 0, p; selp.u32 t2, 2, 0, q; or.b32 %0, t1, t2; }" : "=r"(t) : "r"(xa), "r"(xb), "r"(xc)); O32(t)
#define HSETPC(OPS) asm("{ .reg .pred p, q, c; .reg .u32 t1, t2; setp.ne.u32 c, %3, 0; setp." OPS " p|q, %1, %2, c; " \
                        "selp.u32 t1, 1, 0, p; selp.u32 t2, 2, 0, q; or.b32 %0, t1, t2; }" : "=r"(t) : "r"(xa), "r"(xb), "r"(xc)); O32(t)
      HSETP("lt.f16x2"); HSETP("ge.ftz.f16x2"); HSETP("equ.f16x2"); HSETPC("ne.or.f16x2"); HSETPC("gtu.xor.f16x2");
#undef HSETP
#undef HSETPC
    }
    // bf16 packed arithmetic
    O32(bb2(__hfma2(b2a, b2b, __halves2bfloat162(bb, ba)))); O32(bb2(__hadd2(b2a, b2b))); O32(bb2(__hmul2(b2a, b2b)));
    O32(bb2(__hfma2(__hneg2(b2a), b2b, __habs2(b2b)))); O32(bb2(__hfma2(__low2bfloat162(b2a), b2b, b2a)));
    O32(bb2(__hfma2(b2a, __high2bfloat162(b2b), __low2bfloat162(b2a)))); O32(bb2(__hfma2(b2a, __float2bfloat162_rn(2.0f), b2b)));
    O32(bb2(__hfma2(b2a, b2b, __halves2bfloat162(__float2bfloat16(1.5f), __float2bfloat16(-0.25f)))));
    O32(bb2(__hmul2(b2a, __halves2bfloat162(__float2bfloat16(3.0f), __float2bfloat16(0.125f)))));
    O32(bb2(__hadd2(__high2bfloat162(b2a), __habs2(b2b)))); O32(bb2(__hsub2(b2a, b2b)));
    // MUFU.RCP64H / RSQ64H: double division, reciprocal, square root and reciprocal square root
    OD(__drcp_rn(p)); OD(__ddiv_rn(p, q)); OD(rsqrt(fabs(q))); OD(sqrt(p)); OD(1.0 / q); OD(rsqrt(p));
    // IABS and IMNMX (signed and unsigned, a predicate choosing the minimum or the maximum per lane)
    { const int sx = static_cast<int>(x), sy = static_cast<int>(y);
      O32(abs(sx)); O32(abs(sy - 7)); O32((x & 1) ? min(sx, sy) : max(sx, sy)); O32((y & 2) ? min(x, y) : max(x, y)); O32(max(sx, sy) ^ min(x, y)); }
    // FCHK (the division's check for the slow path)
    OF(a / b); OF(b / a);
#undef O32
#undef O64
#undef OF
#undef OD
#undef OH
#undef OB
  }
  for (int j = 0; j < kConvForms; ++j) out[static_cast<size_t>(j) * n + i] = o[j];
}

// Many predicates live across a loop, each used to guard an update: ptxas runs out of predicate registers and packs them into a general
// register with P2R and unpacks them with R2P. The lanes diverge on the predicates, so both run under partial masks. The host computes
// the same function.
__global__ void pred_press(const uint32_t* in, uint32_t* out, int n, int rounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t v[12];
  for (int k = 0; k < 12; ++k) v[k] = in[(i + k) % n];
#define PP(k) const bool p##k = ((v[(k) % 12] ^ (v[(k + 5) % 12] >> ((k) % 7))) & (1u << ((k) % 5))) ? true : (v[(k + 3) % 12] % 7u == (uint32_t)((k) % 5));
  PP(0) PP(1) PP(2) PP(3) PP(4) PP(5) PP(6) PP(7) PP(8) PP(9) PP(10) PP(11) PP(12) PP(13) PP(14) PP(15) PP(16) PP(17) PP(18) PP(19) PP(20) PP(21)
  uint32_t acc = i;
  for (int r = 0; r < rounds; ++r) {
    for (int k = 0; k < 12; ++k) { v[k] = v[k] * 2654435761u + acc; acc ^= v[(k + 1) % 12] >> 3; }
#define PU(k) if (p##k) acc += v[(k) % 12] * (k + 3); else acc ^= v[(k + 7) % 12] + (k);
    PU(0) PU(1) PU(2) PU(3) PU(4) PU(5) PU(6) PU(7) PU(8) PU(9) PU(10) PU(11) PU(12) PU(13) PU(14) PU(15) PU(16) PU(17) PU(18) PU(19) PU(20) PU(21)
  }
  out[i] = acc;
#undef PP
#undef PU
}
static uint32_t pred_press_host(const std::vector<uint32_t>& in, int n, int i, int rounds) {
  uint32_t v[12];
  for (int k = 0; k < 12; ++k) v[k] = in[(i + k) % n];
  bool p[22];
  for (int k = 0; k < 22; ++k)
    p[k] = ((v[k % 12] ^ (v[(k + 5) % 12] >> (k % 7))) & (1u << (k % 5))) ? true : (v[(k + 3) % 12] % 7u == static_cast<uint32_t>(k % 5));
  uint32_t acc = i;
  for (int r = 0; r < rounds; ++r) {
    for (int k = 0; k < 12; ++k) { v[k] = v[k] * 2654435761u + acc; acc ^= v[(k + 1) % 12] >> 3; }
    for (int k = 0; k < 22; ++k) {
      if (p[k]) acc += v[k % 12] * (k + 3);
      else acc ^= v[(k + 7) % 12] + k;
    }
  }
  return acc;
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

static int g_dump_form = -1;       // `sass_fastpath dump <form>`: also print the inputs and results of one conv_forms form
static bool g_list_forms = false;  // `sass_fastpath forms`: one line per conv_forms form (to see which one differs on a card)
static bool g_card_forms = false;  // `sass_fastpath cardforms`: only the conv_forms lines, one per form, regime and shape (the card file's)
// conv_forms over special values and random bits; one hash line per form and shape (or, with `cardforms`, only those lines).
static int run_conv_forms() {
  {
    const uint32_t fsp[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
                            0x3f800000u, 0xbf800000u, 0x3f000000u, 0xbf000000u, 0x3fc00000u, 0x40000000u, 0x7f7fffffu, 0xff7fffffu,
                            0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0x7f800001u, 0x4b000000u, 0x4b800001u,
                            0x33800000u, 0x3f800001u, 0x4f000000u, 0xcf000000u, 0x4f800000u, 0x5f000000u, 0xdf000000u, 0x5f800000u,
                            0x477fe000u, 0x477ff000u, 0x38000000u, 0x33000000u, 0x33000001u, 0x3f7fffffu, 0x40490fdbu, 0x4effffffu,
                            0x47800000u, 0xc7800000u, 0x7bffu, 0x3b800000u};
    const uint64_t dsp[] = {0x0000000000000000ull, 0x8000000000000000ull, 0x0000000000000001ull, 0x000fffffffffffffull, 0x0010000000000000ull,
                            0x3ff0000000000000ull, 0xbff0000000000000ull, 0x3ff0000000000001ull, 0x7fefffffffffffffull, 0x7ff0000000000000ull,
                            0xfff0000000000000ull, 0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff0000000000001ull, 0x4000000000000000ull,
                            0x3fb999999999999aull, 0x3ca0000000000000ull, 0x43e0000000000000ull /* 2^63 */, 0xc3e0000000000000ull /* -2^63 */,
                            0x43dfffffffffffffull /* just below 2^63 */, 0x43f0000000000000ull /* 2^64 */, 0x43efffffffffffffull,
                            0x41dfffffffc00000ull /* 2^31 - 1 */, 0x41e0000000000000ull /* 2^31 */, 0xc1e0000000000000ull, 0xc1e0000000200000ull,
                            0x41efffffffe00000ull /* 2^32 - 1 */, 0x41f0000000000000ull, 0x3fe0000000000000ull, 0xbfe0000000000000ull,
                            0x3fe0000000000001ull, 0x3ff8000000000000ull, 0x4004000000000000ull, 0x40f0000000000000ull /* 65536 */,
                            0x380ffffffe000000ull /* near the f32 subnormals */, 0x47efffffe0000000ull /* f32 max */, 0x47effffff0000000ull,
                            0x3e70000000000000ull /* 2^-24 */, 0x3e60000000000000ull, 0x40effc0000000000ull /* 65504 */, 0x40effe0000000000ull};
    const uint16_t hsp[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe00,
                            0x7c01, 0x4000, 0x3555, 0x5640, 0x3800, 0x4200, 0xc200, 0x6400, 0x5bff, 0x3bff};
    const uint16_t bsp[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0x7f81,
                            0xffc1, 0x4049, 0x4f80, 0x3fc0, 0x3f81, 0x4000, 0xc000, 0x3f00, 0x4300, 0x7f00};
    const int n = 32 * 11 + 7;
    std::vector<float> hf(n);
    std::vector<double> hd(n);
    std::vector<uint32_t> hw(n);
    std::vector<uint64_t> hl(n);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < n; ++i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      const uint32_t fb = i % 3 == 0 ? static_cast<uint32_t>(s) : fsp[s % (sizeof fsp / sizeof fsp[0])];
      std::memcpy(&hf[i], &fb, 4);
      const uint64_t db = i % 4 == 0 ? s : dsp[(s >> 5) % (sizeof dsp / sizeof dsp[0])];
      std::memcpy(&hd[i], &db, 8);
      // Words made of two halves (the low half of the word a half; a pair of bfloat16 comes from the second draw), or random.
      hw[i] = i % 5 == 0 ? static_cast<uint32_t>(s >> 11)
                         : static_cast<uint32_t>(hsp[(s >> 9) % (sizeof hsp / sizeof hsp[0])]) | static_cast<uint32_t>(i % 2 ? bsp[(s >> 17) % (sizeof bsp / sizeof bsp[0])] : hsp[(s >> 17) % (sizeof hsp / sizeof hsp[0])]) << 16;
      // 64-bit integers: random, small, the edges of 2^24, 2^53, 2^63 and 2^64, and sign extensions of 32-bit values.
      static const uint64_t lsp[] = {0ull, 1ull, ~0ull, 0x8000000000000000ull, 0x7fffffffffffffffull, 0x0000000100000000ull, 0xffffffff00000000ull,
                                     0x0000000001000001ull, 0x0020000000000001ull, 0x0020000000000003ull, 0x00ffffffffffffffull,
                                     0x8000000000000400ull, 0x8000008000000001ull, 0xfffffffffffffc01ull, 0x00000000ffffffffull, 0x0000000080000000ull};
      hl[i] = i % 3 == 0 ? s : lsp[(s >> 7) % (sizeof lsp / sizeof lsp[0])];
    }
    float* dfin; double* ddin; uint32_t *dwin, *dout; uint64_t* dlin;
    CK(cudaMalloc(&dfin, n * 4)); CK(cudaMalloc(&ddin, n * 8)); CK(cudaMalloc(&dwin, n * 4)); CK(cudaMalloc(&dlin, n * 8));
    CK(cudaMalloc(&dout, static_cast<size_t>(n) * kConvForms * 4));
    // Two regimes of inputs. The first is the special values above, which put an out-of-range lane in nearly every warp (a 64-bit
    // conversion with such a lane is left to the general path, and says so by agreeing with it). The second keeps every value
    // inside the 64-bit ranges -- fractions, ties, zeros of both signs, NaNs -- so that those conversions take the 32-lane path.
    for (int regime = 0; regime < 3; ++regime) {
      if (regime == 2) {
        // The same without a NaN or an infinity anywhere: what an RTX 3060 and the simulator agree on (their NaNs differ in payload and
        // sign, and in what a conversion to an integer makes of one).
        for (int i = 0; i < n; ++i) {
          if (!(hd[i] == hd[i]) || std::fabs(hd[i]) > 1.0e300) hd[i] = 0.75;
          if (!(hf[i] == hf[i]) || std::fabs(hf[i]) > 1.0e30f) hf[i] = 0.75f;
          static const uint16_t fin_h[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x4000, 0x3555, 0x5640, 0x3800, 0x4200, 0xc200};
          static const uint16_t fin_b[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x4049, 0x4f80, 0x3fc0, 0x4000, 0xc000, 0x3f00};
          hw[i] = static_cast<uint32_t>(fin_h[(i * 7 + 3) % 16]) | static_cast<uint32_t>(i % 2 ? fin_b[(i * 5 + 1) % 16] : fin_h[(i * 5 + 1) % 16]) << 16;
        }
      }
      if (regime == 1) {
        for (int i = 0; i < n; ++i) {
          s ^= s << 13; s ^= s >> 7; s ^= s << 17;
          const double m = static_cast<double>(static_cast<int64_t>(s >> 11) % 1000000007ll) * (i % 2 ? 0.25 : 1.0 / 1024.0);
          const double v = i % 7 == 0 ? static_cast<double>(static_cast<int>(s % 9) - 4) + (i % 2 ? 0.5 : 0.0)
                         : i % 11 == 0 ? -0.0 : i % 13 == 0 ? std::nan("") : (i % 5 == 0 ? m * 1.0e6 : m);
          hd[i] = v;
          hf[i] = static_cast<float>(i % 17 == 0 ? 0.5 + (s % 5) : i % 19 == 0 ? -0.0 : v * 1.0e-3);
        }
      }
      CK(cudaMemcpy(dfin, hf.data(), n * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(ddin, hd.data(), n * 8, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dwin, hw.data(), n * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dlin, hl.data(), n * 8, cudaMemcpyHostToDevice));
      for (int shape = 0; shape < 3; ++shape) {
        CK(cudaMemset(dout, 0, static_cast<size_t>(n) * kConvForms * 4));
        conv_forms<<<(n + 63) / 64, 64>>>(dfin, ddin, dwin, dlin, dout, n, shape);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> got(static_cast<size_t>(n) * kConvForms);
        CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
        if (g_dump_form >= 0 && g_dump_form < kConvForms)
          for (int i = 0; i < n; ++i) {
            uint32_t fb_; uint64_t db_;
            std::memcpy(&fb_, &hf[i], 4); std::memcpy(&db_, &hd[i], 8);
            std::printf("dump regime=%d shape=%d form=%d lane=%d f=%08x d=%016llx w=%08x l=%016llx -> %08x\n", regime, shape, g_dump_form, i, fb_,
                        static_cast<unsigned long long>(db_), hw[i], static_cast<unsigned long long>(hl[i]), got[static_cast<size_t>(g_dump_form) * n + i]);
          }
        // One line for the launch (the lines are compared between runs together with the counters, which the runtime prints on its own
        // schedule: a short output keeps the two from meeting), or one per form with `forms`.
        uint64_t all = 1469598103934665603ull;
        for (int f = 0; f < kConvForms; ++f) {
          uint64_t hh = 1469598103934665603ull;
          for (int i = 0; i < n; ++i) hh = (hh ^ got[static_cast<size_t>(f) * n + i]) * 1099511628211ull;
          all = (all ^ hh) * 1099511628211ull;
          if (g_list_forms) std::printf("conv_forms regime=%d shape=%d form=%03d bits %016llx\n", regime, shape, f, static_cast<unsigned long long>(hh));
        }
        if (!g_list_forms) std::printf("conv_forms regime=%d shape=%d bits %016llx\n", regime, shape, static_cast<unsigned long long>(all));
      }
    }
    CK(cudaFree(dfin)); CK(cudaFree(ddin)); CK(cudaFree(dwin)); CK(cudaFree(dlin)); CK(cudaFree(dout));
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 2 && std::string(argv[1]) == "dump") g_dump_form = std::atoi(argv[2]);
  else if (argc > 1 && std::string(argv[1]) == "forms") g_list_forms = true;
  else if (argc > 1 && std::string(argv[1]) == "cardforms") g_card_forms = g_list_forms = true;
  else if (argc > 1) return run_fault(argv[1]);
  if (g_card_forms) return run_conv_forms();
  for (int n : {1, 31, 32, 33, 1000, 4096 + 5}) {
    if (run_copy<uint8_t>("u8", n) || run_copy<uint16_t>("u16", n) || run_copy<uint32_t>("u32", n) ||
        run_copy<uint64_t>("u64", n) || run_copy<float>("f32", n) || run_copy<Q>("q16", n))
      return 1;
  }
  if (run_bcast<uint8_t>("u8") || run_bcast<uint16_t>("u16") || run_bcast<uint32_t>("u32") || run_bcast<uint64_t>("u64") || run_bcast<float>("f32") || run_bcast<Q>("q16"))
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
    // pred_press against the host.
    const int n = 32 * 6 + 9;
    std::vector<uint32_t> h(n);
    uint64_t s = 0xD1B54A32D192ED03ull;
    for (int i = 0; i < n; ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; h[i] = static_cast<uint32_t>(s >> 9); }
    uint32_t *din, *dout;
    CK(cudaMalloc(&din, n * 4)); CK(cudaMalloc(&dout, n * 4));
    CK(cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice));
    for (int rounds : {1, 3}) {
      CK(cudaMemset(dout, 0, n * 4));
      pred_press<<<(n + 63) / 64, 64>>>(din, dout, n, rounds);
      CK(cudaDeviceSynchronize());
      std::vector<uint32_t> got(n);
      CK(cudaMemcpy(got.data(), dout, n * 4, cudaMemcpyDeviceToHost));
      int bad = 0;
      uint64_t hh = 1469598103934665603ull;
      for (int i = 0; i < n; ++i) {
        bad += got[i] != pred_press_host(h, n, i, rounds);
        hh = (hh ^ got[i]) * 1099511628211ull;
      }
      if (bad) { std::printf("WRONG pred_press rounds=%d: %d lanes\n", rounds, bad); ++wrong; }
      std::printf("pred_press rounds=%d bits %016llx\n", rounds, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(din)); CK(cudaFree(dout));
  }

  {
    // shared_atomic_forms: three blocks of 64 threads (the last partly used), each on a shared AtomMem copied from the host.
    const int n = 32 * 5 + 11, threads = 64, blocks = (n + threads - 1) / threads;
    AtomMem *dinit, *dfin; uint32_t* dout;
    CK(cudaMalloc(&dinit, sizeof(AtomMem)));
    CK(cudaMalloc(&dfin, blocks * sizeof(AtomMem)));
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
      CK(cudaMemcpy(dinit, &init, sizeof init, cudaMemcpyHostToDevice));
      CK(cudaMemset(dout, 0, static_cast<size_t>(n) * 12 * 4));
      shared_atomic_forms<<<blocks, threads>>>(dinit, dfin, dout, n, pattern);
      CK(cudaDeviceSynchronize());
      std::vector<AtomMem> got(blocks);
      std::vector<uint32_t> gout(static_cast<size_t>(n) * 12);
      CK(cudaMemcpy(got.data(), dfin, blocks * sizeof(AtomMem), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(gout.data(), dout, gout.size() * 4, cudaMemcpyDeviceToHost));
      int bad = 0;
      for (int b = 0; b < blocks; ++b) {
        AtomMem want = init;   // the order-independent ones, per block
        for (int t = 0; t < threads; ++t) {
          const int i = b * threads + t;
          if (i >= n) break;
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
        const AtomMem& g = got[b];
        for (int k = 0; k < kAtomSlots; ++k)
          bad += g.add[k] != want.add[k] || g.umax[k] != want.umax[k] || g.smin[k] != want.smin[k] || g.band[k] != want.band[k] ||
                 g.bor[k] != want.bor[k] || g.bxor[k] != want.bxor[k] || g.add64[k] != want.add64[k] || g.max64[k] != want.max64[k] ||
                 g.fadd[k] != want.fadd[k] || g.dadd[k] != want.dadd[k] || g.red[k] != want.red[k];
      }
      if (bad) { std::printf("WRONG shared_atomic_forms pattern=%d: %d slots\n", pattern, bad); ++wrong; }
      uint64_t hh = 1469598103934665603ull;
      for (uint32_t v : gout) hh = (hh ^ v) * 1099511628211ull;
      for (const AtomMem& g : got) {
        const uint32_t* mw = reinterpret_cast<const uint32_t*>(&g);
        for (size_t k = 0; k < sizeof g / 4; ++k) hh = (hh ^ mw[k]) * 1099511628211ull;
      }
      std::printf("shared_atomic_forms pattern=%d bits %016llx\n", pattern, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(dinit)); CK(cudaFree(dfin)); CK(cudaFree(dout));
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

  {
    // half_forms over random words, special halves and special doubles.
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
    for (int shape = 0; shape < 3; ++shape) {
      half_forms<<<(n + 63) / 64, 64>>>(din, dout, ddin, ddout, n, shape);
      CK(cudaDeviceSynchronize());
      std::vector<uint32_t> got(static_cast<size_t>(n) * kHalfForms);
      std::vector<uint64_t> gotd(static_cast<size_t>(n) * 8);
      CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(gotd.data(), ddout, gotd.size() * 8, cudaMemcpyDeviceToHost));
      uint64_t hh = 1469598103934665603ull;
      for (uint32_t v : got) hh = (hh ^ v) * 1099511628211ull;
      for (uint64_t v : gotd) hh = (hh ^ v) * 1099511628211ull;
      std::printf("half_forms shape=%d bits %016llx\n", shape, static_cast<unsigned long long>(hh));
    }
    CK(cudaFree(din)); CK(cudaFree(dout)); CK(cudaFree(ddin)); CK(cudaFree(ddout));
  }

  {
    // atomic_more on global and on shared memory: the final memory has a host reference (sums, extremes), and the old values are hashed.
    const int n = 32 * 5 + 11;
    const int blocks = 3, threads = 64;
    for (int space = 0; space < 2; ++space)
      for (int pattern = 0; pattern < 4; ++pattern) {
        AtomMore init = {};
        for (int k = 0; k < kAtomSlots; ++k) {
          init.smin[k] = 1000;
          init.smax[k] = -(1ll << 40);
          init.smax32[k] = -100;
          init.dmin_bits[k] = 1.0;
        }
        AtomMore *dinit, *dfin;
        uint32_t* dout;
        CK(cudaMalloc(&dinit, sizeof init)); CK(cudaMalloc(&dfin, sizeof init * blocks));
        CK(cudaMalloc(&dout, static_cast<size_t>(blocks) * threads * 4 * 4));
        CK(cudaMemcpy(dinit, &init, sizeof init, cudaMemcpyHostToDevice));
        CK(cudaMemset(dout, 0, static_cast<size_t>(blocks) * threads * 4 * 4));
        if (space == 0) atomic_more<<<blocks, threads>>>(dinit, dout, n, pattern);
        else shared_atomic_more<<<blocks, threads>>>(dinit, dfin, dout, n, pattern);
        CK(cudaDeviceSynchronize());
        std::vector<AtomMore> got(space == 0 ? 1 : blocks);
        std::vector<uint32_t> gout(static_cast<size_t>(blocks) * threads * 4);
        CK(cudaMemcpy(got.data(), space == 0 ? dinit : dfin, got.size() * sizeof(AtomMore), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gout.data(), dout, gout.size() * 4, cudaMemcpyDeviceToHost));
        // The reference: for each memory (the global one; each block's shared one) the threads that took part.
        int bad = 0;
        for (size_t m = 0; m < got.size(); ++m) {
          AtomMore want = init;
          const int lo = space == 0 ? 0 : static_cast<int>(m) * threads, hi = space == 0 ? n : std::min(n, lo + threads);
          for (int i = lo; i < hi; ++i) {
            const unsigned idx = pattern == 0 ? 0u : pattern == 1 ? i % 4u : pattern == 2 ? static_cast<unsigned>(i) % kAtomSlots : (i * 7u) % 33u;
            const bool take = pattern == 1 ? (i % 5) != 0 : pattern == 3 ? (i & 8) == 0 : true;
            if (!take) continue;
            want.smin[idx] = std::min<long long>(want.smin[idx], static_cast<long long>(i) * 31 - 500);
            want.smax[idx] = std::max<long long>(want.smax[idx], 1000 - static_cast<long long>(i) * 40503);
            want.one[idx] += 1; want.two[idx] += 1;
            want.smax32[idx] = std::max<int32_t>(want.smax32[idx], 77 - i * 5);
          }
          for (int k = 0; k < kAtomSlots; ++k)
            if (got[m].smin[k] != want.smin[k] || got[m].smax[k] != want.smax[k] || got[m].one[k] != want.one[k] ||
                got[m].two[k] != want.two[k] || got[m].smax32[k] != want.smax32[k])
              ++bad;
        }
        if (bad) { std::printf("WRONG atomic_more space=%d pattern=%d: %d slots\n", space, pattern, bad); ++wrong; }
        uint64_t hh = 1469598103934665603ull;
        for (uint32_t v : gout) hh = (hh ^ v) * 1099511628211ull;
        for (const AtomMore& g : got) {
          const uint32_t* mw = reinterpret_cast<const uint32_t*>(&g);
          for (size_t k = 0; k < sizeof g / 4; ++k) hh = (hh ^ mw[k]) * 1099511628211ull;
        }
        std::printf("atomic_more space=%d pattern=%d bits %016llx\n", space, pattern, static_cast<unsigned long long>(hh));
        CK(cudaFree(dinit)); CK(cudaFree(dfin)); CK(cudaFree(dout));
      }
  }

  {
    // hostmap_forms: zero-copy buffers read and written by a kernel, checked on the host.
    const int n = 32 * 5 + 13;
    uint8_t *h8, *hb8, *d8, *db8; uint16_t *h16, *hb16, *d16, *db16; uint32_t *h32, *hb32, *d32, *db32; uint64_t *h64, *hb64, *d64, *db64;
    CK(cudaHostAlloc(&h8, n, cudaHostAllocMapped)); CK(cudaHostAlloc(&hb8, n, cudaHostAllocMapped));
    CK(cudaHostAlloc(&h16, n * 2, cudaHostAllocMapped)); CK(cudaHostAlloc(&hb16, n * 2, cudaHostAllocMapped));
    CK(cudaHostAlloc(&h32, n * 4, cudaHostAllocMapped)); CK(cudaHostAlloc(&hb32, n * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc(&h64, n * 8, cudaHostAllocMapped)); CK(cudaHostAlloc(&hb64, n * 8, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer(&d8, h8, 0)); CK(cudaHostGetDevicePointer(&db8, hb8, 0));
    CK(cudaHostGetDevicePointer(&d16, h16, 0)); CK(cudaHostGetDevicePointer(&db16, hb16, 0));
    CK(cudaHostGetDevicePointer(&d32, h32, 0)); CK(cudaHostGetDevicePointer(&db32, hb32, 0));
    CK(cudaHostGetDevicePointer(&d64, h64, 0)); CK(cudaHostGetDevicePointer(&db64, hb64, 0));
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int pattern = 0; pattern < 4; ++pattern) {
      for (int i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        h8[i] = static_cast<uint8_t>(s); h16[i] = static_cast<uint16_t>(s >> 8); h32[i] = static_cast<uint32_t>(s >> 16); h64[i] = s * 0x2545F4914F6CDD1Dull;
        hb8[i] = 0xee; hb16[i] = 0xeeee; hb32[i] = 0xeeeeeeeeu; hb64[i] = 0xeeeeeeeeeeeeeeeeull;
      }
      hostmap_forms<<<(n + 63) / 64, 64>>>(d8, db8, d16, db16, d32, db32, d64, db64, n, pattern);
      CK(cudaDeviceSynchronize());
      int bad = 0;
      uint64_t hh = 1469598103934665603ull;
      std::vector<char> written(n, 0);
      for (int i = 0; i < n; ++i) {
        if (pattern == 3 && (i % 5) == 0) continue;
        written[pattern == 1 ? (i * 7) % n : pattern == 2 ? n - 1 - i : i] = 1;
      }
      for (int j = 0; j < n; ++j) {
        // (the lane that wrote slot j read a32[i] for its own i: the slot's writer is found again)
        int i = j;
        if (pattern == 1) { for (i = 0; i < n && (i * 7) % n != j; ++i) {} }
        else if (pattern == 2) i = n - 1 - j;
        uint8_t w8 = 0xee; uint16_t w16 = 0xeeee; uint32_t w32 = 0xeeeeeeeeu; uint64_t w64 = 0xeeeeeeeeeeeeeeeeull;
        if (written[j]) {
          w8 = static_cast<uint8_t>(h8[j] + 1); w16 = static_cast<uint16_t>(h16[j] ^ 0x5a5a); w32 = h32[j] * 3u + 1u + h32[0];
          w64 = h64[j] + h32[i] + (static_cast<uint64_t>(h32[0]) << 32);
        }
        if (hb8[j] != w8 || hb16[j] != w16 || hb32[j] != w32 || hb64[j] != w64) ++bad;
        hh = (hh ^ hb8[j]) * 1099511628211ull; hh = (hh ^ hb16[j]) * 1099511628211ull;
        hh = (hh ^ hb32[j]) * 1099511628211ull; hh = (hh ^ hb64[j]) * 1099511628211ull;
      }
      if (bad) { std::printf("WRONG hostmap_forms pattern=%d: %d slots\n", pattern, bad); ++wrong; }
      std::printf("hostmap_forms pattern=%d bits %016llx\n", pattern, static_cast<unsigned long long>(hh));
    }
    CK(cudaFreeHost(h8)); CK(cudaFreeHost(hb8)); CK(cudaFreeHost(h16)); CK(cudaFreeHost(hb16));
    CK(cudaFreeHost(h32)); CK(cudaFreeHost(hb32)); CK(cudaFreeHost(h64)); CK(cudaFreeHost(hb64));
  }
  if (run_conv_forms()) return 1;
  std::printf("wrong: %d\n", wrong);
  return wrong != 0;
}
