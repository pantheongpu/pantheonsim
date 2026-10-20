// Blackwell's packed fp32 arithmetic, add/mul/fma.{rn,rz,rm,rp}{,.ftz}.f32x2 (SASS FADD2, FMUL2, FFMA2 on sm_100), through nvcc and
// ptxas for sm_100: run on the SASS engine and on the PTX engine (VGPU_SASS=0) -- run_sass_archs.sh holds the two to the same
// output -- and checked against the arithmetic written out on the host. The slots cover every operand form ptxas emits (registers,
// one float for both lanes as an immediate, a kernel parameter in a uniform register, one 32-bit register for both lanes), the
// swizzles (halves swapped, either half broadcast), |x|, -x and the per-lane signs (SASS's .NP) of each operand, all four
// rounding modes with and without .ftz, over zeros of both signs, subnormals, infinities, NaNs and the largest and smallest
// finite values. A second kernel holds known answers (IEEE 754 by hand: ties, directed roundings, overflow, flushing).
//
// Not verified on hardware: no sm_100 card was available. The expected values here come from IEEE 754 and from the host's float
// arithmetic with the rounding mode set; what the card does for a NaN's payload and for the forms below that ptxas emits only
// in other code is unchecked (NaNs are compared as NaNs).
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(x)                                                                                          \
  do {                                                                                                    \
    cudaError_t e_ = (x);                                                                                 \
    if (e_ != cudaSuccess) {                                                                              \
      std::printf("FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                 \
      std::exit(1);                                                                                       \
    }                                                                                                     \
  } while (0)

enum { ADD = 0, MUL = 1, FMA = 2 };
// Operand sources: the thread's own value, the other thread values, kernel parameters, immediates.
enum { R = 0, VA = 1, VB = 2, VC = 3, K0 = 4, K1 = 5, KB = 6, I1 = 7, I2 = 8 };
// Operand modifiers, applied to the pair before the operation.
enum { ID = 0, SW, BL, BH, NG, NL, NH, AB, NA, AL, AH, SW_NL, SW_NH, BL_NG, BH_NH, SW_NA };

constexpr uint64_t kImm1 = 0x3f8000003f800000ull;   // 1.0 for both lanes: one float
constexpr uint64_t kImm2 = 0x40a00000bf800000ull;   // 5.0 and -1.0: two registers

// SLOT(op, rnd, ftz, source a, mod a, source b, mod b, source c, mod c); rnd 0 rn, 1 rz, 2 rm, 3 rp. Add and mul ignore c.
#define SLOTS_ROUNDING(X)                                                                                                 \
  X(FMA, 0, 0, R, ID, R, ID, R, ID) X(FMA, 1, 0, R, ID, R, ID, R, ID) X(FMA, 2, 0, R, ID, R, ID, R, ID)                 \
  X(FMA, 3, 0, R, ID, R, ID, R, ID) X(FMA, 0, 1, R, ID, R, ID, R, ID) X(FMA, 1, 1, R, ID, R, ID, R, ID)                 \
  X(FMA, 2, 1, R, ID, R, ID, R, ID) X(FMA, 3, 1, R, ID, R, ID, R, ID)                                                   \
  X(ADD, 0, 0, R, ID, R, ID, R, ID) X(ADD, 1, 0, R, ID, R, ID, R, ID) X(ADD, 2, 0, R, ID, R, ID, R, ID)                 \
  X(ADD, 3, 0, R, ID, R, ID, R, ID) X(ADD, 0, 1, R, ID, R, ID, R, ID) X(ADD, 1, 1, R, ID, R, ID, R, ID)                 \
  X(ADD, 2, 1, R, ID, R, ID, R, ID) X(ADD, 3, 1, R, ID, R, ID, R, ID)                                                   \
  X(MUL, 0, 0, R, ID, R, ID, R, ID) X(MUL, 1, 0, R, ID, R, ID, R, ID) X(MUL, 2, 0, R, ID, R, ID, R, ID)                 \
  X(MUL, 3, 0, R, ID, R, ID, R, ID) X(MUL, 0, 1, R, ID, R, ID, R, ID) X(MUL, 1, 1, R, ID, R, ID, R, ID)                 \
  X(MUL, 2, 1, R, ID, R, ID, R, ID) X(MUL, 3, 1, R, ID, R, ID, R, ID)

// One modifier at a time on each operand, round to nearest.
#define SLOTS_FMA_MOD(X, M) X(FMA, 0, 0, R, M, R, ID, R, ID) X(FMA, 0, 0, R, ID, R, M, R, ID) X(FMA, 0, 0, R, ID, R, ID, R, M)
#define SLOTS_BIN_MOD(X, M) X(ADD, 0, 0, R, M, R, ID, R, ID) X(ADD, 0, 0, R, ID, R, M, R, ID)                          \
                            X(MUL, 0, 0, R, M, R, ID, R, ID) X(MUL, 0, 0, R, ID, R, M, R, ID)
#define SLOTS_MODS(X)                                                                                                     \
  SLOTS_FMA_MOD(X, SW) SLOTS_FMA_MOD(X, BL) SLOTS_FMA_MOD(X, BH) SLOTS_FMA_MOD(X, NG) SLOTS_FMA_MOD(X, NL)              \
  SLOTS_FMA_MOD(X, NH) SLOTS_FMA_MOD(X, AB) SLOTS_FMA_MOD(X, NA) SLOTS_FMA_MOD(X, AL) SLOTS_FMA_MOD(X, AH)              \
  SLOTS_FMA_MOD(X, SW_NL) SLOTS_FMA_MOD(X, SW_NH) SLOTS_FMA_MOD(X, BL_NG) SLOTS_FMA_MOD(X, BH_NH) SLOTS_FMA_MOD(X, SW_NA) \
  SLOTS_BIN_MOD(X, SW) SLOTS_BIN_MOD(X, BL) SLOTS_BIN_MOD(X, BH) SLOTS_BIN_MOD(X, NG) SLOTS_BIN_MOD(X, NL)              \
  SLOTS_BIN_MOD(X, NH) SLOTS_BIN_MOD(X, AB) SLOTS_BIN_MOD(X, NA) SLOTS_BIN_MOD(X, AL) SLOTS_BIN_MOD(X, AH)              \
  SLOTS_BIN_MOD(X, SW_NL) SLOTS_BIN_MOD(X, SW_NH) SLOTS_BIN_MOD(X, BL_NG) SLOTS_BIN_MOD(X, BH_NH) SLOTS_BIN_MOD(X, SW_NA)

// Modifiers on several operands at once, other roundings and .ftz with them.
#define SLOTS_COMBOS(X)                                                                                                   \
  X(FMA, 0, 0, R, NG, R, NG, R, NG) X(FMA, 0, 0, R, SW, R, BL, R, NH) X(FMA, 0, 0, R, BH, R, SW, R, NL)                 \
  X(FMA, 0, 0, R, NL, R, NH, R, SW_NH) X(FMA, 1, 1, R, AB, R, NA, R, SW) X(FMA, 3, 0, R, NH, R, BL, R, AB)              \
  X(FMA, 2, 1, R, SW, R, NG, R, BH_NH) X(ADD, 0, 0, R, NG, R, NG, R, ID) X(ADD, 1, 1, R, SW, R, NH, R, ID)               \
  X(ADD, 3, 0, R, AB, R, SW_NL, R, ID) X(MUL, 2, 0, R, NH, R, SW, R, ID) X(MUL, 0, 1, R, BL, R, NA, R, ID)              \
  X(MUL, 3, 1, R, SW_NH, R, BH, R, ID)

// Other operand forms: a kernel parameter pair (a uniform register), a parameter's low word for both lanes, immediates (one float
// for both lanes, or two different ones), each in every position, and a thread's own value used twice.
#define SLOTS_FORM(X, S) X(FMA, 0, 0, S, ID, R, ID, R, ID) X(FMA, 0, 0, R, ID, S, ID, R, ID) X(FMA, 0, 0, R, ID, R, ID, S, ID)
#define SLOTS_FORM2(X, S) X(ADD, 0, 0, S, ID, R, ID, R, ID) X(ADD, 0, 0, R, ID, S, ID, R, ID)                          \
                          X(MUL, 0, 0, S, ID, R, ID, R, ID) X(MUL, 0, 0, R, ID, S, ID, R, ID)
#define SLOTS_FORMS(X)                                                                                                    \
  SLOTS_FORM(X, K0) SLOTS_FORM(X, K1) SLOTS_FORM(X, KB) SLOTS_FORM(X, I1) SLOTS_FORM(X, I2)                             \
  SLOTS_FORM2(X, K0) SLOTS_FORM2(X, K1) SLOTS_FORM2(X, KB) SLOTS_FORM2(X, I1) SLOTS_FORM2(X, I2)                         \
  X(FMA, 0, 0, R, ID, R, ID, K0, NG) X(FMA, 0, 0, R, ID, K1, SW, R, ID) X(FMA, 0, 0, K1, AB, R, ID, R, ID)              \
  X(FMA, 1, 1, R, ID, KB, NG, R, ID) X(FMA, 3, 0, I1, ID, R, ID, R, NG) X(ADD, 2, 0, R, ID, K0, NG, R, ID)               \
  X(MUL, 3, 0, K1, SW, R, ID, R, ID) X(ADD, 0, 1, R, ID, I2, ID, R, ID) X(MUL, 0, 1, I2, ID, R, ID, R, ID)              \
  X(FMA, 0, 0, R, ID, VA, ID, VA, ID) X(FMA, 0, 0, R, ID, VB, ID, VA, ID) X(FMA, 0, 0, R, ID, VA, ID, VB, ID)           \
  X(MUL, 0, 0, R, ID, VA, ID, R, ID) X(ADD, 0, 0, R, ID, VA, ID, R, ID) X(ADD, 0, 0, R, ID, VB, ID, R, ID)              \
  X(FMA, 0, 0, R, ID, VB, SW, VA, NG) X(ADD, 0, 0, R, SW, VA, ID, R, ID)

#define SLOTS(X) SLOTS_ROUNDING(X) SLOTS_MODS(X) SLOTS_COMBOS(X) SLOTS_FORMS(X)

// ---- the device side ----------------------------------------------------------------------------------------------------------

#define PK(name, body)                                                                                                    \
  __device__ __forceinline__ uint64_t name(uint64_t a) {                                                                  \
    uint64_t d;                                                                                                           \
    asm("{ .reg .b32 l,h,t; mov.b64 {l,h}, %1; " body " }" : "=l"(d) : "l"(a));                                           \
    return d;                                                                                                             \
  }
PK(m_sw, "mov.b64 %0, {h,l};")
PK(m_bl, "mov.b64 %0, {l,l};")
PK(m_bh, "mov.b64 %0, {h,h};")
PK(m_ng, "neg.f32 l,l; neg.f32 h,h; mov.b64 %0, {l,h};")
PK(m_nl, "neg.f32 l,l; mov.b64 %0, {l,h};")
PK(m_nh, "neg.f32 h,h; mov.b64 %0, {l,h};")
PK(m_ab, "abs.f32 l,l; abs.f32 h,h; mov.b64 %0, {l,h};")
PK(m_na, "abs.f32 l,l; abs.f32 h,h; neg.f32 l,l; neg.f32 h,h; mov.b64 %0, {l,h};")
PK(m_al, "abs.f32 l,l; mov.b64 %0, {l,h};")
PK(m_ah, "abs.f32 h,h; mov.b64 %0, {l,h};")
PK(m_sw_nl, "neg.f32 t,h; mov.b64 %0, {t,l};")
PK(m_sw_nh, "neg.f32 t,l; mov.b64 %0, {h,t};")
PK(m_bl_ng, "neg.f32 t,l; mov.b64 %0, {t,t};")
PK(m_bh_nh, "neg.f32 t,h; mov.b64 %0, {h,t};")
PK(m_sw_na, "abs.f32 l,l; abs.f32 h,h; neg.f32 t,h; neg.f32 l,l; mov.b64 %0, {t,l};")

template <int M> __device__ __forceinline__ uint64_t modify(uint64_t x) {
  if constexpr (M == ID) return x;
  else if constexpr (M == SW) return m_sw(x);
  else if constexpr (M == BL) return m_bl(x);
  else if constexpr (M == BH) return m_bh(x);
  else if constexpr (M == NG) return m_ng(x);
  else if constexpr (M == NL) return m_nl(x);
  else if constexpr (M == NH) return m_nh(x);
  else if constexpr (M == AB) return m_ab(x);
  else if constexpr (M == NA) return m_na(x);
  else if constexpr (M == AL) return m_al(x);
  else if constexpr (M == AH) return m_ah(x);
  else if constexpr (M == SW_NL) return m_sw_nl(x);
  else if constexpr (M == SW_NH) return m_sw_nh(x);
  else if constexpr (M == BL_NG) return m_bl_ng(x);
  else if constexpr (M == BH_NH) return m_bh_nh(x);
  else return m_sw_na(x);
}

struct Args {
  uint64_t a, b, c, k0, k1, kb;   // the thread's three pairs; the parameters (kb: one word for both lanes)
};

template <int S> __device__ __forceinline__ uint64_t pick(const Args& g, uint64_t own) {
  if constexpr (S == R) return own;
  else if constexpr (S == VA) return g.a;
  else if constexpr (S == VB) return g.b;
  else if constexpr (S == VC) return g.c;
  else if constexpr (S == K0) return g.k0;
  else if constexpr (S == K1) return g.k1;
  else if constexpr (S == KB) return g.kb;
  else if constexpr (S == I1) return kImm1;
  else return kImm2;
}

// "fma.rn.ftz.f32x2" and the others: the PTX spelling is a string literal, so each is written out.
template <int OP, int RND, int FTZ>
__device__ __forceinline__ uint64_t f2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t d;
#define GEN3(MOD)                                                                                                         \
  if constexpr (OP == FMA) asm("fma." MOD ".f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));                  \
  else if constexpr (OP == ADD) asm("add." MOD ".f32x2 %0, %1, %2; // %3" : "=l"(d) : "l"(a), "l"(b), "l"(c));            \
  else asm("mul." MOD ".f32x2 %0, %1, %2; // %3" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  if constexpr (RND == 0 && FTZ == 0) { GEN3("rn") }
  else if constexpr (RND == 1 && FTZ == 0) { GEN3("rz") }
  else if constexpr (RND == 2 && FTZ == 0) { GEN3("rm") }
  else if constexpr (RND == 3 && FTZ == 0) { GEN3("rp") }
  else if constexpr (RND == 0 && FTZ == 1) { GEN3("rn.ftz") }
  else if constexpr (RND == 1 && FTZ == 1) { GEN3("rz.ftz") }
  else if constexpr (RND == 2 && FTZ == 1) { GEN3("rm.ftz") }
  else { GEN3("rp.ftz") }
  return d;
}

constexpr int kSlots = 0
#define COUNT(op, r, f, sa, ma, sb, mb, sc, mc) +1
    SLOTS(COUNT)
#undef COUNT
    ;

extern "C" __global__ void packed(const uint64_t* in, uint64_t* out, uint64_t k0, uint64_t k1, uint32_t kw, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Args g{in[3 * i], in[3 * i + 1], in[3 * i + 2], k0, k1, (uint64_t(kw) << 32) | kw};
  uint64_t* o = out + size_t(i) * kSlots;
  int s = 0;
#define SLOT(op, r, f, sa, ma, sb, mb, sc, mc)                                                                            \
  o[s++] = f2<op, r, f>(modify<ma>(pick<sa>(g, g.a)), modify<mb>(pick<sb>(g, g.b)), modify<mc>(pick<sc>(g, g.c)));
  SLOTS(SLOT)
#undef SLOT
}

// Known answers: one per thread-0 slot, from inputs in memory so that ptxas cannot fold them.
#define KATS(X)                                                                                                           \
  /* 1 + 2^-24 is a tie between 1 and 1+2^-23: even (1) for rn, up only for rp */                                         \
  X(ADD, 0, 0, 0x3f8000003f800000ull, 0x3380000033800000ull, 0, 0x3f8000003f800000ull)                                    \
  X(ADD, 1, 0, 0x3f8000003f800000ull, 0x3380000033800000ull, 0, 0x3f8000003f800000ull)                                    \
  X(ADD, 2, 0, 0x3f8000003f800000ull, 0x3380000033800000ull, 0, 0x3f8000003f800000ull)                                    \
  X(ADD, 3, 0, 0x3f8000003f800000ull, 0x3380000033800000ull, 0, 0x3f8000013f800001ull)                                    \
  /* the lanes are independent: lane 0 (2.0 + 2^-24) goes up for rp, lane 1 (1.0 + 0) is exact */                         \
  X(ADD, 3, 0, 0x3f80000040000000ull, 0x0000000033800000ull, 0, 0x3f80000040000001ull)                                    \
  /* the largest float times two, lane 0 negative: inf for rn; the largest for rz; rm sends only the negative lane to */  \
  /* infinity, rp only the positive one */                                                                                 \
  X(MUL, 0, 0, 0x7f7fffffff7fffffull, 0x4000000040000000ull, 0, 0x7f800000ff800000ull)                                    \
  X(MUL, 1, 0, 0x7f7fffffff7fffffull, 0x4000000040000000ull, 0, 0x7f7fffffff7fffffull)                                    \
  X(MUL, 2, 0, 0x7f7fffffff7fffffull, 0x4000000040000000ull, 0, 0x7f7fffffff800000ull)                                    \
  X(MUL, 3, 0, 0x7f7fffffff7fffffull, 0x4000000040000000ull, 0, 0x7f800000ff7fffffull)                                    \
  /* the smallest subnormal halved (lane 0 positive, lane 1 negative): a tie to even, so 0 for rn; rp rounds the positive */ \
  /* up and the negative to -0, rm the reverse; .ftz flushes the input */                                                  \
  X(MUL, 0, 0, 0x8000000100000001ull, 0x3f0000003f000000ull, 0, 0x8000000000000000ull)                                    \
  X(MUL, 3, 0, 0x8000000100000001ull, 0x3f0000003f000000ull, 0, 0x8000000000000001ull)                                    \
  X(MUL, 2, 0, 0x8000000100000001ull, 0x3f0000003f000000ull, 0, 0x8000000100000000ull)                                    \
  X(MUL, 3, 1, 0x8000000100000001ull, 0x3f0000003f000000ull, 0, 0x8000000000000000ull)                                    \
  /* a subnormal result (the sum of two) stays without .ftz and is flushed with it, keeping the sign */                    \
  X(ADD, 0, 0, 0x0000000200000001ull, 0x0000000100000001ull, 0, 0x0000000300000002ull)                                    \
  X(ADD, 0, 1, 0x0000000200000001ull, 0x0000000100000001ull, 0, 0x0000000000000000ull)                                    \
  X(ADD, 0, 1, 0x8000000200000001ull, 0x8000000100000001ull, 0, 0x8000000000000000ull)                                    \
  /* zeros of opposite signs add to +0, or to -0 rounding down; two -0 add to -0 */                                        \
  X(ADD, 0, 0, 0x0000000080000000ull, 0x8000000000000000ull, 0, 0x0000000000000000ull)                                    \
  X(ADD, 2, 0, 0x0000000080000000ull, 0x8000000000000000ull, 0, 0x8000000080000000ull)                                    \
  X(ADD, 0, 0, 0x8000000080000000ull, 0x8000000080000000ull, 0, 0x8000000080000000ull)                                    \
  /* fma rounds once: (1+2^-12)^2 = 1 + 2^-11 + 2^-24 (not a float) less 1 + 2^-11 is exactly 2^-24, where a rounded */   \
  /* product would give 0; then the same square plus 2^-30 in the four modes */                                            \
  X(FMA, 0, 0, 0x3f8008003f800800ull, 0x3f8008003f800800ull, 0xbf801000bf801000ull, 0x3380000033800000ull)               \
  X(FMA, 0, 0, 0x3f8008003f800800ull, 0x3f8008003f800800ull, 0x3080000030800000ull, 0x3f8010013f801001ull)               \
  X(FMA, 1, 0, 0x3f8008003f800800ull, 0x3f8008003f800800ull, 0x3080000030800000ull, 0x3f8010003f801000ull)               \
  X(FMA, 2, 0, 0x3f8008003f800800ull, 0x3f8008003f800800ull, 0x3080000030800000ull, 0x3f8010003f801000ull)               \
  X(FMA, 3, 0, 0x3f8008003f800800ull, 0x3f8008003f800800ull, 0x3080000030800000ull, 0x3f8010013f801001ull)

extern "C" __global__ void kat(const uint64_t* in, uint64_t* out) {
  if (threadIdx.x != 0) return;
  int k = 0;
#define KAT(op, r, f, a, b, c, e) out[k] = f2<op, r, f>(in[3 * k], in[3 * k + 1], in[3 * k + 2]); ++k;
  KATS(KAT)
#undef KAT
}

// ---- the host side -----------------------------------------------------------------------------------------------------------

struct SlotInfo {
  int op, rnd, ftz, sa, ma, sb, mb, sc, mc;
};
static const SlotInfo kSlotTable[] = {
#define ENTRY(op, r, f, sa, ma, sb, mb, sc, mc) {op, r, f, sa, ma, sb, mb, sc, mc},
    SLOTS(ENTRY)
#undef ENTRY
};

static float tof(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static uint32_t tob(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

static uint64_t host_modify(int m, uint64_t x) {
  uint32_t l = uint32_t(x), h = uint32_t(x >> 32);
  const uint32_t S = 0x80000000u, A = 0x7fffffffu;
  switch (m) {
    case ID: break;
    case SW: std::swap(l, h); break;
    case BL: h = l; break;
    case BH: l = h; break;
    case NG: l ^= S; h ^= S; break;
    case NL: l ^= S; break;
    case NH: h ^= S; break;
    case AB: l &= A; h &= A; break;
    case NA: l = (l & A) ^ S; h = (h & A) ^ S; break;
    case AL: l &= A; break;
    case AH: h &= A; break;
    case SW_NL: std::swap(l, h); l ^= S; break;
    case SW_NH: std::swap(l, h); h ^= S; break;
    case BL_NG: l ^= S; h = l; break;
    case BH_NH: l = h; h ^= S; break;
    case SW_NA: {   // |x| on both, then the halves swapped and both negated
      const uint32_t nl = (h & A) ^ S, nh = (l & A) ^ S;
      l = nl;
      h = nh;
      break;
    }
  }
  return uint64_t(l) | uint64_t(h) << 32;
}

static uint32_t flush(uint32_t b, bool ftz) {
  return ftz && (b & 0x7f800000u) == 0 ? (b & 0x80000000u) : b;
}

static uint32_t host_lane(int op, int rnd, bool ftz, uint32_t a, uint32_t b, uint32_t c) {
  static const int modes[] = {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD};
  const int old = std::fegetround();
  std::fesetround(modes[rnd]);
  volatile float x = tof(flush(a, ftz)), y = tof(flush(b, ftz)), z = tof(flush(c, ftz)), r;
  if (op == FMA) r = std::fmaf(x, y, z);
  else if (op == ADD) r = x + y;
  else r = x * y;
  std::fesetround(old);
  return flush(tob(r), ftz);
}

static bool is_nan_bits(uint32_t b) { return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0; }

static uint64_t host_source(int s, uint64_t own, const uint64_t* v, uint64_t k0, uint64_t k1, uint32_t kw) {
  switch (s) {
    case R: return own;
    case VA: return v[0];
    case VB: return v[1];
    case VC: return v[2];
    case K0: return k0;
    case K1: return k1;
    case KB: return (uint64_t(kw) << 32) | kw;
    case I1: return kImm1;
    default: return kImm2;
  }
}

static uint64_t splitmix(uint64_t& s) {
  uint64_t z = (s += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

static const uint32_t kSpecial[] = {
    0x00000000, 0x80000000,                           // zeros
    0x00000001, 0x80000001, 0x007fffff, 0x807fffff,   // subnormals
    0x00400000, 0x00800000, 0x80800000,               // half the smallest normal, the smallest normal
    0x3f800000, 0xbf800000, 0x3fc00000, 0x40490fdb,   // 1, -1, 1.5, pi
    0x3f7fffff, 0x3f800001, 0x33800000, 0x34000000,   // around 1, 2^-24, 2^-23
    0x4b000000, 0x7f000000, 0x7f7fffff, 0xff7fffff,   // 2^23, 2^127, the largest finite
    0x7f800000, 0xff800000,                           // infinities
    0x7fc00000, 0xffc00000, 0x7fc12345, 0x7f800001, 0xffa00000,   // NaNs, with payloads, signalling
    0x3dcccccd, 0x40a00000, 0x2f800000, 0x5f800000    // 0.1, 5, 2^-34, 2^64
};
constexpr int kNSpecial = int(sizeof kSpecial / sizeof kSpecial[0]);

static uint32_t pick_value(uint64_t& st) {
  const uint64_t r = splitmix(st);
  if (r % 10 < 7) return kSpecial[(r >> 8) % kNSpecial];
  // A random float with an exponent in a range where products overflow and underflow now and then.
  const uint32_t e = 0x60 + uint32_t((r >> 40) % 0x40);
  return (uint32_t(r >> 63) << 31) | (e << 23) | (uint32_t(r >> 8) & 0x7fffffu);
}

int main() {
  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  if (count == 0) { std::printf("FAIL no device\n"); return 1; }
  constexpr int kThreads = 2048, kLaunches = 4;
  std::vector<uint64_t> in(3 * kThreads);
  uint64_t st = 12345;
  for (auto& v : in) v = uint64_t(pick_value(st)) | uint64_t(pick_value(st)) << 32;
  // Kernel parameter pairs: the same specials, and different halves.
  uint64_t K[kLaunches][2];
  uint32_t KW[kLaunches];
  for (int l = 0; l < kLaunches; ++l) {
    for (int j = 0; j < 2; ++j) K[l][j] = uint64_t(pick_value(st)) | uint64_t(pick_value(st)) << 32;
    KW[l] = pick_value(st);
  }
  K[0][0] = 0x4000000040000000ull; K[0][1] = 0x3f0000003f000000ull; KW[0] = 0x40400000;   // 2, 0.5, 3: plain numbers first

  uint64_t *din, *dout;
  CHECK(cudaMalloc(&din, in.size() * 8));
  CHECK(cudaMalloc(&dout, size_t(kThreads) * kSlots * 8));
  CHECK(cudaMemcpy(din, in.data(), in.size() * 8, cudaMemcpyHostToDevice));
  std::vector<uint64_t> out(size_t(kThreads) * kSlots);
  long ref_bad = 0, shown = 0;
  std::printf("slots %d threads %d launches %d\n", kSlots, kThreads, kLaunches);
  for (int l = 0; l < kLaunches; ++l) {
    packed<<<kThreads / 64, 64>>>(din, dout, K[l][0], K[l][1], KW[l], kThreads);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out.data(), dout, out.size() * 8, cudaMemcpyDeviceToHost));
    for (int s = 0; s < kSlots; ++s) {
      const SlotInfo& si = kSlotTable[s];
      uint64_t hash = 14695981039346656037ull;
      long bad = 0;
      for (int i = 0; i < kThreads; ++i) {
        const uint64_t got = out[size_t(i) * kSlots + s];
        hash = (hash ^ got) * 1099511628211ull;
        const uint64_t v[3] = {in[3 * i], in[3 * i + 1], in[3 * i + 2]};
        const uint64_t a = host_modify(si.ma, host_source(si.sa, v[0], v, K[l][0], K[l][1], KW[l]));
        const uint64_t b = host_modify(si.mb, host_source(si.sb, v[1], v, K[l][0], K[l][1], KW[l]));
        const uint64_t c = host_modify(si.mc, host_source(si.sc, v[2], v, K[l][0], K[l][1], KW[l]));
        uint64_t want = 0;
        for (int h = 0; h < 2; ++h)
          want |= uint64_t(host_lane(si.op, si.rnd, si.ftz, uint32_t(a >> (32 * h)), uint32_t(b >> (32 * h)), uint32_t(c >> (32 * h)))) << (32 * h);
        bool same = true;
        for (int h = 0; h < 2; ++h) {
          const uint32_t g = uint32_t(got >> (32 * h)), w = uint32_t(want >> (32 * h));
          if (!(g == w || (is_nan_bits(g) && is_nan_bits(w)))) same = false;
        }
        if (!same) {
          ++bad;
          if (shown++ < 12)
            std::printf("MISMATCH launch %d slot %d (op %d rnd %d ftz %d, mods %d/%d/%d) thread %d: got %016llx want %016llx; operands %016llx %016llx %016llx\n", l, s,
                        si.op, si.rnd, si.ftz, si.ma, si.mb, si.mc, i, (unsigned long long)got, (unsigned long long)want, (unsigned long long)a,
                        (unsigned long long)b, (unsigned long long)c);
        }
      }
      ref_bad += bad;
      // PACKED_DUMP=1 lists every result with its operands (to diff two engines' output when the hashes differ).
      if (std::getenv("PACKED_DUMP"))
        for (int i = 0; i < kThreads; ++i)
          std::printf("dump %d %3d %4d %016llx  in %016llx %016llx %016llx\n", l, s, i, (unsigned long long)out[size_t(i) * kSlots + s],
                      (unsigned long long)in[3 * i], (unsigned long long)in[3 * i + 1], (unsigned long long)in[3 * i + 2]);
      // What the engines print must agree; the hash is over the raw results, NaN payloads included.
      std::printf("launch %d slot %3d hash %016llx\n", l, s, (unsigned long long)hash);
    }
  }

  // Known answers.
  struct Kat { int op, rnd, ftz; uint64_t a, b, c, want; };
  const Kat kats[] = {
#define KATROW(op, r, f, a, b, c, e) {op, r, f, a, b, c, e},
      KATS(KATROW)
#undef KATROW
  };
  const int nk = int(sizeof kats / sizeof kats[0]);
  std::vector<uint64_t> kin(3 * nk), kout(nk);
  for (int k = 0; k < nk; ++k) { kin[3 * k] = kats[k].a; kin[3 * k + 1] = kats[k].b; kin[3 * k + 2] = kats[k].c; }
  uint64_t *dkin, *dkout;
  CHECK(cudaMalloc(&dkin, kin.size() * 8));
  CHECK(cudaMalloc(&dkout, kout.size() * 8));
  CHECK(cudaMemcpy(dkin, kin.data(), kin.size() * 8, cudaMemcpyHostToDevice));
  kat<<<1, 32>>>(dkin, dkout);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(kout.data(), dkout, kout.size() * 8, cudaMemcpyDeviceToHost));
  int kat_bad = 0;
  for (int k = 0; k < nk; ++k) {
    bool same = true;
    for (int h = 0; h < 2; ++h) {
      const uint32_t g = uint32_t(kout[k] >> (32 * h)), w = uint32_t(kats[k].want >> (32 * h));
      if (!(g == w || (is_nan_bits(g) && is_nan_bits(w)))) same = false;
    }
    std::printf("kat %2d op %d rnd %d ftz %d: %016llx%s\n", k, kats[k].op, kats[k].rnd, kats[k].ftz, (unsigned long long)kout[k],
                same ? "" : " WRONG");
    if (!same) {
      ++kat_bad;
      std::printf("KAT %d: got %016llx want %016llx\n", k, (unsigned long long)kout[k], (unsigned long long)kats[k].want);
    }
  }
  std::printf("host reference mismatches: %ld, known-answer mismatches: %d\n", ref_bad, kat_bad);
  if (ref_bad || kat_bad) return 1;
  std::printf("PASS\n");
  return 0;
}
