// The AMD interpreter's pre-decoded tier (vgpu/amd_gcn_tier.hpp). gcn::finalize calls tier_classify when an
// instruction is decoded; the interpreter's step() calls Inst::fast.run, if there is one, before it does anything else.
//
// Every handler is the interpreter's code for the same instruction, written over whole register rows instead of lane by
// lane: the same reads (a scalar source once, a register row as it is), the same writes (only the lanes EXEC names, a
// scalar destination exactly as write_scalar writes it), the same arithmetic. What the tier is not sure of it leaves
// to the interpreter: classify gives no handler to a form it does not model (modifiers, sub-dword and cross-lane
// forms, registers past the file), and a handler that meets a NaN result gives the instruction back before it has
// changed anything (which of two NaN payloads wins depends on the order the compiler gave the operands, so the tier
// does not take that decision).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>

#include "gcn_divide.hpp"
#include "gcn_tier_ds.hpp"
#include "nan_order.hpp"
#include "packed_math.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vgpu/simd/simd.hpp"

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#include <immintrin.h>
#define VGPU_TIER_X86 1
#endif

// A product the card rounds before the sum must not become an FMA in a build that has one.
#pragma GCC optimize("fp-contract=off")

namespace vgpu::amd::gcn {

namespace tier {
namespace {
// -1: not looked at yet.
std::atomic<int> g_state{-1};
// On unless VGPU_AMD_TIER=0.
constexpr bool kDefaultOn = true;
}  // namespace

bool enabled() {
  int s = g_state.load(std::memory_order_relaxed);
  if (s < 0) {
    const char* e = std::getenv("VGPU_AMD_TIER");
    s = e && *e ? (*e != '0') : (kDefaultOn ? 1 : 0);
    g_state.store(s, std::memory_order_relaxed);
  }
  return s != 0;
}
void set_enabled(bool on) { g_state.store(on ? 1 : 0, std::memory_order_relaxed); }
namespace {
std::atomic<uint64_t> g_bails{0};
}
namespace {
struct Stats {
  std::mutex m;
  std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> by_name;   // none, handed back
  ~Stats() {
    if (by_name.empty()) return;
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> v(by_name.begin(), by_name.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.first + a.second.second > b.second.first + b.second.second; });
    uint64_t total = 0;
    for (const auto& e : v) total += e.second.first + e.second.second;
    std::fprintf(stderr, "AMD tier: %llu instructions ran in the interpreter\n", static_cast<unsigned long long>(total));
    std::fprintf(stderr, "  %-40s %12s %12s\n", "instruction", "no handler", "handed back");
    for (size_t i = 0; i < v.size() && i < 60; ++i)
      std::fprintf(stderr, "  %-40s %12llu %12llu\n", v[i].first.c_str(), static_cast<unsigned long long>(v[i].second.first),
                   static_cast<unsigned long long>(v[i].second.second));
  }
};
Stats& stats_table() {
  static Stats s;
  return s;
}
}  // namespace
bool stats() {
  static const bool on = [] {
    const char* e = std::getenv("VGPU_AMD_TIER_STATS");
    return e && *e && *e != '0';
  }();
  return on;
}
void note_interp(const char* name, bool handed_back) {
  Stats& s = stats_table();
  std::lock_guard<std::mutex> lock(s.m);
  auto& e = s.by_name[name];
  (handed_back ? e.second : e.first) += 1;
}
static bool g_count_bails = false;   // (set before any dispatch runs; a counter shared by every host thread is not free)
void count_bails(bool on) { g_count_bails = on; }
void note_bail() {
  if (g_count_bails) g_bails.fetch_add(1, std::memory_order_relaxed);
}
uint64_t bails() { return g_bails.load(std::memory_order_relaxed); }
}  // namespace tier

#ifndef VGPU_TIER_X86

void tier_classify(Inst&) {}   // (the lane kernels are x86's: elsewhere the interpreter runs everything)

#else

namespace {

using namespace tier;
using namespace tier_ds;

constexpr uint32_t kSgprs = 106;   // gcn_exec.cpp's kSgprs: s0 through s105
constexpr uint32_t kVgprs = 256;

// ---- The kernel lists: one X-list per handler family, the same order in the enums and in each level's table.
#define TIER_INT_OPS(X)                                                                                              \
  X(MOV, k_mov) X(NOT, k_not) X(ADD, k_add) X(SUB, k_sub) X(SUBREV, k_subrev) X(MUL_LO, k_mul_lo)                   \
  X(MUL_HI_U, k_mul_hi_u) X(MUL_HI_I, k_mul_hi_i) X(MUL_U24, k_mul_u24) X(MUL_I24, k_mul_i24)                       \
  X(MAD_U24, k_mad_u24) X(MAD_I24, k_mad_i24) X(AND, k_and) X(OR, k_or) X(XOR, k_xor) X(LSHLREV, k_lshlrev)         \
  X(LSHRREV, k_lshrrev) X(ASHRREV, k_ashrrev) X(MIN_U, k_min_u) X(MAX_U, k_max_u) X(MIN_I, k_min_i)                \
  X(MAX_I, k_max_i) X(BFI, k_bfi) X(ALIGNBIT, k_alignbit) X(ALIGNBYTE, k_alignbyte) X(ADD3, k_add3) X(OR3, k_or3)  \
  X(AND_OR, k_and_or) X(XAD, k_xad) X(LSHL_OR, k_lshl_or) X(LSHL_ADD, k_lshl_add) X(ADD_LSHL, k_add_lshl)          \
  X(BCNT, k_bcnt) X(BFE_U, k_bfe_u) X(MED3_U, k_med3_u) X(MED3_I, k_med3_i) X(MIN3_U, k_min3_u)                     \
  X(MAX3_U, k_max3_u) X(XOR3, k_xor3) X(MBCNT_LO, k_mbcnt_lo) X(MBCNT_HI, k_mbcnt_hi) X(CVT_I32_F32, k_cvt_i32_f32) X(CVT_U32_F32, k_cvt_u32_f32) X(CVT_F32_U, k_cvt_f32_u) X(CVT_F32_I, k_cvt_f32_i) X(CVT_F32_UBYTE0, k_cvt_f32_ubyte0) X(ADD_U16, k_add_u16) X(SUB_U16, k_sub_u16) X(SUBREV_U16, k_subrev_u16) X(LSHLREV_B16, k_lshlrev_b16) X(LSHRREV_B16, k_lshrrev_b16) X(ASHRREV_I16, k_ashrrev_i16) X(MUL_LO_U16, k_mul_lo_u16) X(MIN_U16, k_min_u16) X(MAX_U16, k_max_u16) X(MIN_I16, k_min_i16) X(MAX_I16, k_max_i16) X(PACK_B32_F16, k_pack_b32_f16) X(BFE_I, k_bfe_i) X(PERM, k_perm) X(CVT_F16_F32, k_cvt_f16_f32) X(FFBH_U, k_ffbh_u) X(FFBL, k_ffbl) X(FFBH_I, k_ffbh_i) \
  X(PK_MULLO, k_pk_mullo) X(PK_ADDU, k_pk_addu) X(PK_ADDI, k_pk_addi) X(PK_SUBU, k_pk_subu) X(PK_SUBI, k_pk_subi) X(PK_SHL, k_pk_shl) X(PK_SHR, k_pk_shr) \
  X(PK_ASHR, k_pk_ashr) X(PK_MAXI, k_pk_maxi) X(PK_MINI, k_pk_mini) X(PK_MAXU, k_pk_maxu) X(PK_MINU, k_pk_minu) X(PK_MADU, k_pk_madu) X(PK_MADI, k_pk_madi) \
  X(PK_ADDU_C, k_pk_addu_c) X(PK_ADDI_C, k_pk_addi_c) X(PK_SUBU_C, k_pk_subu_c) X(PK_SUBI_C, k_pk_subi_c) X(PK_MADU_C, k_pk_madu_c) X(PK_MADI_C, k_pk_madi_c)
#define TIER_F32_OPS(X) X(FADD, k_fadd) X(FSUB, k_fsub) X(FSUBREV, k_fsubrev) X(FMUL, k_fmul) X(FMA, k_ffma) X(FMIN, k_fmin) X(FMAX, k_fmax) X(FRCP, k_frcp) X(FSQRT, k_fsqrt) X(FRSQ, k_frsq) X(FEXP, k_fexp) X(FLOG, k_flog) X(FFLOOR, k_ffloor) X(FCEIL, k_fceil) X(FTRUNC, k_ftrunc) X(FRNDNE, k_frndne) X(FFRACT, k_ffract) X(FSIN_R, k_fsin_r) X(FSIN_T, k_fsin_t) X(FCOS_R, k_fcos_r) X(FCOS_T, k_fcos_t) X(FLDEXP, k_fldexp) X(FCVT_F32_F16, k_fcvt_f32_f16)
#define TIER_F64_OPS(X) X(DADD, k_dadd) X(DMUL, k_dmul) X(DFMA, k_dfma) X(DRCP, k_drcp) X(DSQRT, k_dsqrt) X(DFLOOR, k_dfloor) X(DCEIL, k_dceil) X(DTRUNC, k_dtrunc) X(DRNDNE, k_drndne) X(DMIN, k_dmin) X(DMAX, k_dmax)
#define TIER_L64_OPS(X) X(MOV, l_mov) X(LSHLREV, l_lshlrev) X(LSHRREV, l_lshrrev) X(ASHRREV, l_ashrrev) X(LSHL_ADD, l_lshl_add) X(LSHL_ADD_G, l_lshl_add_g) X(CVT_F64_U, l_cvt_f64_u) X(CVT_F64_I, l_cvt_f64_i) X(LDEXP, l_ldexp) X(FREXP_MANT, l_frexp_mant) X(FREXP_EXP, l_frexp_exp) X(CVT_F64_F32, l_cvt_f64_f32) X(CVT_F32_F64, l_cvt_f32_f64) X(CVT_I32_F64, l_cvt_i32_f64) X(CVT_U32_F64, l_cvt_u32_f64)

#define TIER_ENUM(id, fn) id,
enum IntId { TIER_INT_OPS(TIER_ENUM) kIntCount };
enum F32Id { TIER_F32_OPS(TIER_ENUM) kF32Count };
enum F64Id { TIER_F64_OPS(TIER_ENUM) kF64Count };
#define TIER_ENUM_L(id, fn) L_##id,
enum L64Id { TIER_L64_OPS(TIER_ENUM_L) kL64Count };
#undef TIER_ENUM
#define TIER_PICK(id, fn) fn,

namespace base_impl {
#define TIER_NS base_impl_ns
#define TIER_AVX2 0
#include "gcn_tier_kernels.inc"
#undef TIER_NS
#undef TIER_AVX2
}  // namespace base_impl
namespace avx2_impl {
#pragma GCC push_options
#pragma GCC target("avx2,fma,popcnt")
#define TIER_NS avx2_impl_ns
#define TIER_AVX2 1
#include "gcn_tier_kernels.inc"
#undef TIER_NS
#undef TIER_AVX2
#pragma GCC pop_options
}  // namespace avx2_impl

using K3 = void (*)(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, unsigned);
using K64 = void (*)(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, unsigned);
using CmpFn = uint64_t (*)(const uint32_t*, const uint32_t*, unsigned);
using Cmp64Fn = uint64_t (*)(const uint64_t*, const uint64_t*, unsigned);
using Mad64Fn = uint64_t (*)(uint32_t*, uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, unsigned);
using CarryFn = uint64_t (*)(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, unsigned);
using F64RowsFn = uint64_t (*)(uint32_t*, uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint64_t*, unsigned);

struct Tables {
  const K3* ints;
  const K3* f32;
  const K64* f64;
  const K64* l64;
  const CmpFn* cmp_u32;
  const CmpFn* cmp_i32;
  const CmpFn* cmp_f32;
  const Cmp64Fn* cmp_u64;
  const Cmp64Fn* cmp_i64;
  const Cmp64Fn* cmp_f64;
  const CmpFn* cmp_i16;
  const CmpFn* cmp_u16;
  const CmpFn* cmp_f16;
  CmpFn cmp_class32;
  CmpFn cmp_class16;
  CmpFn cmp_class16_neg;
  Mad64Fn mad_u64;
  const F64RowsFn* f64_rows;   // add, multiply, fused multiply-add, each for the 8 sets of uniform operands
  const CarryFn* carry;   // v_add_co_u32 and its kin, by Op::sub
  bool avx2;   // the handlers built for AVX2 (TIER_DUAL) are the ones to use
};
const Tables& tables() {
  static const Tables base = {base_impl::base_impl_ns::kIntTab, base_impl::base_impl_ns::kF32Tab, base_impl::base_impl_ns::kF64Tab,
                              base_impl::base_impl_ns::kL64Tab, base_impl::base_impl_ns::kCmpU32, base_impl::base_impl_ns::kCmpI32,
                              base_impl::base_impl_ns::kCmpF32, base_impl::base_impl_ns::kCmpU64, base_impl::base_impl_ns::kCmpI64, base_impl::base_impl_ns::kCmpF64,
                              base_impl::base_impl_ns::kCmpI16, base_impl::base_impl_ns::kCmpU16, base_impl::base_impl_ns::kCmpF16, base_impl::base_impl_ns::cmp_class_f32, base_impl::base_impl_ns::cmp_class_f16<false>,
                              base_impl::base_impl_ns::cmp_class_f16<true>, base_impl::base_impl_ns::k_mad_u64, base_impl::base_impl_ns::kF64Rows, base_impl::base_impl_ns::kCarry, false};
  static const Tables avx2 = {avx2_impl::avx2_impl_ns::kIntTab, avx2_impl::avx2_impl_ns::kF32Tab, avx2_impl::avx2_impl_ns::kF64Tab,
                              avx2_impl::avx2_impl_ns::kL64Tab, avx2_impl::avx2_impl_ns::kCmpU32, avx2_impl::avx2_impl_ns::kCmpI32,
                              avx2_impl::avx2_impl_ns::kCmpF32, avx2_impl::avx2_impl_ns::kCmpU64, avx2_impl::avx2_impl_ns::kCmpI64, avx2_impl::avx2_impl_ns::kCmpF64,
                              avx2_impl::avx2_impl_ns::kCmpI16, avx2_impl::avx2_impl_ns::kCmpU16, avx2_impl::avx2_impl_ns::kCmpF16, avx2_impl::avx2_impl_ns::cmp_class_f32, avx2_impl::avx2_impl_ns::cmp_class_f16<false>,
                              avx2_impl::avx2_impl_ns::cmp_class_f16<true>, avx2_impl::avx2_impl_ns::k_mad_u64, avx2_impl::avx2_impl_ns::kF64Rows, avx2_impl::avx2_impl_ns::kCarry, true};
  return simd::enabled() && simd::level() == simd::Level::Avx2 ? avx2 : base;
}

// ---- Register rows

typedef uint32_t __attribute__((vector_size(32))) V8u;
typedef float __attribute__((may_alias)) F32;
typedef double __attribute__((may_alias)) F64;

struct MaskTable {
  alignas(32) uint32_t m[256][8];
  MaskTable() {
    for (unsigned b = 0; b < 256; ++b)
      for (unsigned l = 0; l < 8; ++l) m[b][l] = (b >> l & 1) ? ~0u : 0u;
  }
};
const MaskTable& mask_table() {
  static const MaskTable t;
  return t;
}

// d[l] = r[l] for the lanes of `ex` among the first n; the others of d keep their value.
[[gnu::always_inline]] inline void merge(uint32_t* d, const uint32_t* r, uint64_t ex, unsigned n) {
  const MaskTable& t = mask_table();
  for (unsigned i = 0; i < n; i += 8) {
    V8u m, x, y;
    std::memcpy(&m, t.m[(ex >> i) & 0xFF], 32);
    std::memcpy(&x, r + i, 32);
    std::memcpy(&y, d + i, 32);
    y = (x & m) | (y & ~m);
    std::memcpy(d + i, &y, 32);
  }
}

// A register row (32 or 64 words) copied, in vector moves of a size known here (a copy of n * 4 bytes is `rep movs` otherwise).
[[gnu::always_inline]] inline void copy_row(uint32_t* d, const uint32_t* s, unsigned n) {
  if (n == 64) std::memcpy(d, s, 256);
  else std::memcpy(d, s, 128);
}
// Bytes copied in 32-byte moves, then the rest.
[[gnu::always_inline]] inline void copy_bytes(void* dst, const void* src, size_t bytes) {
  uint8_t* d = static_cast<uint8_t*>(dst);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  for (; bytes >= 32; bytes -= 32, d += 32, s += 32) {
    V8u v;
    std::memcpy(&v, s, 32);
    std::memcpy(d, &v, 32);
  }
  if (bytes) std::memcpy(d, s, bytes);
}
[[gnu::always_inline]] inline bool full(uint64_t ex, unsigned n) { return n == 64 ? ex == ~uint64_t{0} : (ex & 0xFFFFFFFFull) == 0xFFFFFFFFull; }
// The 64-bit results r[0..n) into the register pair (lo, hi), for the lanes of `ex` (a few lanes: their own loop; more: both words of all of them
// and a merge, which the vector units do whole).
[[gnu::always_inline]] inline void store64(uint32_t* lo, uint32_t* hi, const uint64_t* r, uint64_t ex, unsigned n) {
  if (full(ex, n)) {
    for (unsigned i = 0; i < n; ++i) lo[i] = static_cast<uint32_t>(r[i]), hi[i] = static_cast<uint32_t>(r[i] >> 32);
  } else if (__builtin_popcountll(ex) > 8) {
    alignas(32) uint32_t rl[64], rh[64];
    for (unsigned i = 0; i < n; ++i) rl[i] = static_cast<uint32_t>(r[i]), rh[i] = static_cast<uint32_t>(r[i] >> 32);
    merge(lo, rl, ex, n);
    merge(hi, rh, ex, n);
  } else {
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      lo[l] = static_cast<uint32_t>(r[l]), hi[l] = static_cast<uint32_t>(r[l] >> 32);
    }
  }
}

[[gnu::always_inline]] inline uint64_t rd(const Regs& R, const Src& s);
[[gnu::always_inline]] inline void fill(uint32_t* t, uint32_t v, unsigned n) {
  for (unsigned i = 0; i < n; ++i) t[i] = v;
}
[[gnu::always_inline]] inline uint32_t mods32(const Src& s, uint32_t v) {
  if (s.abs) v &= s.h16 ? ~0x8000u : 0x7FFFFFFFu;
  if (s.neg) v ^= s.h16 ? 0x8000u : 0x80000000u;
  return v;
}
// A 32-bit lane source as a row: the register's own, or a copy in `t` where a modifier changes it or the value is the same in every lane.
[[gnu::always_inline]] inline const uint32_t* load32(const Src& s, Regs& R, uint32_t* t, unsigned n) {
  switch (s.kind) {
    case kVgpr: {
      const uint32_t* p = R.vgpr[s.index];
      if (!(s.neg | s.abs)) return p;
      const uint32_t keep = s.abs ? (s.h16 ? ~0x8000u : 0x7FFFFFFFu) : ~0u, flip = s.neg ? (s.h16 ? 0x8000u : 0x80000000u) : 0u;
      for (unsigned i = 0; i < n; ++i) t[i] = (p[i] & keep) ^ flip;
      return t;
    }
    case kConst: fill(t, mods32(s, s.value), n); return t;
    case kSgpr: fill(t, mods32(s, R.sgpr[s.index]), n); return t;
    default: fill(t, mods32(s, static_cast<uint32_t>(rd(R, s))), n); return t;   // (VCC and EXEC, read as scalar() does and cut to 32 bits)
  }
}
// A 64-bit lane source: a register pair, a constant or scalar register(s), with the modifiers on bit 63.
[[gnu::always_inline]] inline void load64(const Src& s, Regs& R, uint64_t* t, unsigned n) {
  const uint64_t keep = s.abs ? 0x7FFFFFFFFFFFFFFFull : ~uint64_t{0}, flip = s.neg ? 0x8000000000000000ull : 0;
  switch (s.kind) {
    case kVgpr: {
      const uint32_t* lo = R.vgpr[s.index];
      if (s.wide) {
        const uint32_t* hi = R.vgpr[s.index + 1];
        for (unsigned i = 0; i < n; ++i) t[i] = ((static_cast<uint64_t>(hi[i]) << 32 | lo[i]) & keep) ^ flip;
      } else {
        for (unsigned i = 0; i < n; ++i) t[i] = (static_cast<uint64_t>(lo[i]) & keep) ^ flip;
      }
      return;
    }
    case kConst: {
      const uint64_t v = (s.value64 & keep) ^ flip;
      for (unsigned i = 0; i < n; ++i) t[i] = v;
      return;
    }
    case kSgpr: {
      const uint64_t v = ((s.wide ? static_cast<uint64_t>(R.sgpr[s.index + 1]) << 32 | R.sgpr[s.index] : R.sgpr[s.index]) & keep) ^ flip;
      for (unsigned i = 0; i < n; ++i) t[i] = v;
      return;
    }
    default:
      for (unsigned i = 0; i < n; ++i) t[i] = 0;
  }
}
// A 32-bit consumer's read of a 64-bit-instruction source (a shift count): the low word, zero-extended.
inline void load32_to64(const Src& s, Regs& R, uint64_t* t, unsigned n) {
  alignas(32) uint32_t w[64];
  const uint32_t* p = load32(s, R, w, n);
  for (unsigned i = 0; i < n; ++i) t[i] = p[i];
}

// ---- Scalar operands (what the interpreter's scalar() and write_scalar do)

[[gnu::always_inline]] inline uint64_t rd(const Regs& R, const Src& s) {
  switch (s.kind) {
    case kConst: return s.value64;
    case kSgpr: return s.wide ? static_cast<uint64_t>(R.sgpr[s.index + 1]) << 32 | R.sgpr[s.index] : R.sgpr[s.index];
    case kVcc: return s.wide ? *R.vcc : static_cast<uint32_t>(*R.vcc);
    case kVccHi: return static_cast<uint32_t>(*R.vcc >> 32);
    case kExec: return *R.exec;
    case kExecLo: return static_cast<uint32_t>(*R.exec);
    case kExecHi: return static_cast<uint32_t>(*R.exec >> 32);
    default: return 0;
  }
}
[[gnu::always_inline]] inline void wr(Regs& R, const Src& s, uint64_t v) {
  switch (s.kind) {
    case kSgpr:
      R.sgpr[s.index] = static_cast<uint32_t>(v);
      if (s.wide) R.sgpr[s.index + 1] = static_cast<uint32_t>(v >> 32);
      return;
    case kVcc: *R.vcc = s.wide ? v : (*R.vcc & ~0xFFFFFFFFull) | static_cast<uint32_t>(v); return;
    case kVccHi: *R.vcc = (*R.vcc & 0xFFFFFFFFull) | (v << 32); return;
    case kExec: *R.exec = v; return;
    case kExecLo: *R.exec = (*R.exec & ~0xFFFFFFFFull) | static_cast<uint32_t>(v); return;
    case kExecHi: *R.exec = (*R.exec & 0xFFFFFFFFull) | (v << 32); return;
    default: return;
  }
}


// A handler whose body is compiled twice: for the baseline x86-64 and for AVX2 (so that the loops of its glue -- the operand rows, the EXEC merge, the
// scans for NaNs, the 64-bit pairs -- are vectorized as the lane kernels are). The tier picks one when the instruction is decoded (Tables::avx2).
#define TIER_DUAL(h)                                                                      \
  int h##_base(const Op& o, Regs& R) { return h##_body(o, R); }                           \
  [[gnu::target("avx2,fma,popcnt")]] int h##_avx2(const Op& o, Regs& R) { return h##_body(o, R); }
#define TIER_RUN(h) (T.avx2 ? h##_avx2 : h##_base)

// ---- Vector handlers

[[gnu::always_inline]] inline unsigned lanes_of(uint64_t ex) { return (ex >> 32) ? 64 : 32; }

// NaN anywhere among the active lanes of a binary32 / binary64 result.
[[gnu::always_inline]] inline bool nan32(const uint32_t* r, uint64_t ex, unsigned n) {
  uint32_t any = 0;
  if (full(ex, n)) {
    for (unsigned i = 0; i < n; ++i) any |= (r[i] & 0x7FFFFFFFu) > 0x7F800000u;
    return any != 0;
  }
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    any |= (r[l] & 0x7FFFFFFFu) > 0x7F800000u;
  }
  return any != 0;
}
[[gnu::always_inline]] inline bool nan64(const uint64_t* r, uint64_t ex, unsigned n) {
  uint32_t any = 0;
  if (full(ex, n)) {
    for (unsigned i = 0; i < n; ++i) any |= (r[i] & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull;
    return any != 0;
  }
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    any |= (r[l] & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull;
  }
  return any != 0;
}
// The NaN rules of the interpreter's instructions (the Order of each is the one its source line uses, which reproduces the recorded
// digests): codes for Op::nan above 1.
enum NanRule : uint8_t { kNanHandBack = 0, kNanPass = 1, kNanAdd, kNanSub, kNanSubRev, kNanMul, kNanFma, kNanPkFma };
template <typename T>
inline T nan_rule(uint8_t rule, T x, T y, T z) {
  switch (rule) {
    case kNanAdd: return vgpu::amd::nan_add<1>(x, y);
    case kNanSub: return vgpu::amd::nan_sub<0>(x, y);
    case kNanSubRev: return vgpu::amd::nan_sub<0>(y, x);
    case kNanMul: return vgpu::amd::nan_mul<1>(x, y);
    case kNanFma: return vgpu::amd::nan_fma<2>(x, y, z);
    default: return vgpu::amd::nan_fma<0>(x, y, z);   // kNanPkFma
  }
}
// The lanes of a binary32 result that are NaNs (among EXEC's), done again by the interpreter's rule from the operands the kernel had.
inline void fix_nan32(uint8_t rule, uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t ex) {
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    if ((r[l] & 0x7FFFFFFFu) <= 0x7F800000u) continue;
    float x, y, z, v;
    std::memcpy(&x, &a[l], 4), std::memcpy(&y, &b[l], 4), std::memcpy(&z, &c[l], 4);
    v = nan_rule<float>(rule, x, y, z);
    std::memcpy(&r[l], &v, 4);
  }
}
inline void fix_nan64(uint8_t rule, uint64_t* r, const uint64_t* a, const uint64_t* b, const uint64_t* c, uint64_t ex) {
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    if ((r[l] & 0x7FFFFFFFFFFFFFFFull) <= 0x7FF0000000000000ull) continue;
    double x, y, z, v;
    std::memcpy(&x, &a[l], 8), std::memcpy(&y, &b[l], 8), std::memcpy(&z, &c[l], 8);
    v = nan_rule<double>(rule, x, y, z);
    std::memcpy(&r[l], &v, 8);
  }
}
// A float result as write_float / write_double leave it: the output multiplier, then the clamp.
inline void post32(uint32_t* r, uint64_t ex, unsigned omod, bool clamp) {
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    float v;
    std::memcpy(&v, &r[l], 4);
    if (omod) v *= omod == 1 ? 2.0f : omod == 2 ? 4.0f : 0.5f;
    if (clamp) v = std::isnan(v) || v <= 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
    std::memcpy(&r[l], &v, 4);
  }
}
inline void post64(uint64_t* r, uint64_t ex, unsigned omod, bool clamp) {
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    double v;
    std::memcpy(&v, &r[l], 8);
    if (omod) v *= omod == 1 ? 2.0 : omod == 2 ? 4.0 : 0.5;
    if (clamp) v = std::isnan(v) || v <= 0.0 ? 0.0 : v > 1.0 ? 1.0 : v;
    std::memcpy(&r[l], &v, 8);
  }
}

// A 32-bit integer (or bit) operation over 1 to 3 sources.
[[gnu::always_inline]] inline int h_int_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t t0[64], t1[64], t2[64], r[64];
  const uint32_t* a = load32(o.src[0], R, t0, n);
  const uint32_t* b = o.src[1].kind ? load32(o.src[1], R, t1, n) : a;
  const uint32_t* c = o.src[2].kind ? load32(o.src[2], R, t2, n) : a;
  uint32_t* d = R.vgpr[o.dst.index];
  const K3 k = reinterpret_cast<K3>(o.kern);
  if (full(ex, n)) {
    k(d, a, b, c, n);
  } else {
    k(r, a, b, c, n);
    merge(d, r, ex, n);
  }
  return true;
}
TIER_DUAL(h_int)

// A 32-bit integer operation whose sources are all vector registers (no modifiers): the kernel reads the registers' rows and, with every lane on,
// writes the destination's row, with nothing built on the way. A partial EXEC works in a row of its own and merges (out of line, so that this
// frame has no arrays to guard).
[[gnu::always_inline]] inline void int_partial_body(uint32_t* d, K3 k, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t ex, unsigned n) {
  alignas(32) uint32_t r[64];
  k(r, a, b, c, n);
  merge(d, r, ex, n);
}
[[gnu::noinline]] void int_partial_base(uint32_t* d, K3 k, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t ex, unsigned n) {
  int_partial_body(d, k, a, b, c, ex, n);
}
[[gnu::noinline, gnu::target("avx2,fma,popcnt")]] void int_partial_avx2(uint32_t* d, K3 k, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t ex, unsigned n) {
  int_partial_body(d, k, a, b, c, ex, n);
}
template <bool Avx2>
[[gnu::always_inline]] inline int h_intv_t(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  const uint32_t* a = R.vgpr[o.src[0].index];
  const uint32_t* b = o.src[1].kind ? R.vgpr[o.src[1].index] : a;
  const uint32_t* c = o.src[2].kind ? R.vgpr[o.src[2].index] : a;
  uint32_t* d = R.vgpr[o.dst.index];
  const K3 k = reinterpret_cast<K3>(o.kern);
  if (__builtin_expect(full(ex, n), 1)) k(d, a, b, c, n);
  else if (Avx2) int_partial_avx2(d, k, a, b, c, ex, n);
  else int_partial_base(d, k, a, b, c, ex, n);
  return true;
}
int h_intv_base(const Op& o, Regs& R) { return h_intv_t<false>(o, R); }
[[gnu::target("avx2,fma,popcnt")]] int h_intv_avx2(const Op& o, Regs& R) { return h_intv_t<true>(o, R); }

// v_mov_b32: a row copied, or a scalar spread into the destination itself (no temporary row to build and copy).
[[gnu::always_inline]] inline int h_mov_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  uint32_t* d = R.vgpr[o.dst.index];
  if (full(ex, n)) {
    const uint32_t* a = load32(o.src[0], R, d, n);
    if (a != d) copy_row(d, a, n);
  } else {
    alignas(32) uint32_t t[64];
    merge(d, load32(o.src[0], R, t, n), ex, n);
  }
  return true;
}
TIER_DUAL(h_mov)

// ---- SDWA (an integer operation on parts of its registers)
// The part of a 32-bit value an SDWA source names, with or without its sign (the interpreter's selected()).
inline uint32_t sdwa_pick(uint32_t v, uint8_t sel, bool sext) {
  if (sel <= 3) {
    const uint8_t byte = static_cast<uint8_t>(v >> (8 * sel));
    return sext ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(byte))) : byte;
  }
  if (sel <= 5) {
    const uint16_t half = static_cast<uint16_t>(sel == 4 ? v : v >> 16);
    return sext ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(half))) : half;
  }
  return v;
}
// A source row as load32 gives it, cut to its part (in place in `t`, which a scalar source already is).
inline const uint32_t* load32_sel(const Src& s, Regs& R, uint32_t* t, unsigned n) {
  const uint32_t* p = load32(s, R, t, n);
  if (s.sel == 6) return p;
  for (unsigned i = 0; i < n; ++i) t[i] = sdwa_pick(p[i], s.sel, s.sext);
  return t;
}
// The result's part of the destination, and the rest of it as dst_unused says (the interpreter's write_lane with a narrowed destination):
// zero for 0, the part's sign for 1, what the register held for 2.
inline uint32_t sdwa_dst(uint32_t v, uint32_t was, uint8_t sel, uint8_t unused) {
  const uint32_t mask = sel <= 3 ? 0xFFu << (8 * sel) : sel == 4 ? 0xFFFFu : sel == 5 ? 0xFFFF0000u : ~0u;
  v = sel <= 3 ? (v & 0xFFu) << (8 * sel) : sel == 4 ? v & 0xFFFFu : sel == 5 ? (v & 0xFFFFu) << 16 : v;
  if (unused == 1 && sel < 6) {
    const uint32_t top = mask & ~(mask >> 1);
    if (v & top) v |= ~(mask | (top - 1));
  }
  if (unused == 2) v = (was & ~mask) | (v & mask);
  return v;
}
int h_int_sdwa(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t t0[64], t1[64], t2[64], r[64];
  const uint32_t* a = load32_sel(o.src[0], R, t0, n);
  const uint32_t* b = o.src[1].kind ? load32_sel(o.src[1], R, t1, n) : a;
  const uint32_t* c = o.src[2].kind ? load32_sel(o.src[2], R, t2, n) : a;
  reinterpret_cast<K3>(o.kern)(r, a, b, c, n);
  uint32_t* d = R.vgpr[o.dst.index];
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    d[l] = sdwa_dst(r[l], d[l], o.dsel, o.dunused);
  }
  return true;
}


// ---- DPP (an integer operation whose first source is another lane's)
// The lane a DPP control makes a lane read (Machine::dpp_source: the same table, kept in step by the tier test). 1: that lane; 0: none; -1: a control
// the interpreter does not model (it raises an error there, so such an instruction is not classified).
inline int dpp_from(uint32_t ctrl, uint32_t lane, uint32_t* from) {
  const uint32_t row = lane & ~15u, in_row = lane & 15u;
  if (ctrl <= 0xFF) {
    *from = (lane & ~3u) + ((ctrl >> (2 * (lane & 3))) & 3);
    return 1;
  }
  if (ctrl >= 0x101 && ctrl <= 0x10F) {
    const uint32_t n = ctrl - 0x100;
    if (in_row + n > 15) return 0;
    *from = row + in_row + n;
    return 1;
  }
  if (ctrl >= 0x111 && ctrl <= 0x11F) {
    const uint32_t n = ctrl - 0x110;
    if (n > in_row) return 0;
    *from = row + in_row - n;
    return 1;
  }
  if (ctrl >= 0x121 && ctrl <= 0x12F) {
    *from = row + ((in_row + 16 - (ctrl - 0x120)) & 15);
    return 1;
  }
  if (ctrl == 0x130) {
    if (lane == 63) return 0;
    *from = lane + 1;
    return 1;
  }
  if (ctrl == 0x134) {
    *from = (lane + 1) % 64;
    return 1;
  }
  if (ctrl == 0x138) {
    if (lane == 0) return 0;
    *from = lane - 1;
    return 1;
  }
  if (ctrl == 0x13C) {
    *from = (lane + 63) % 64;
    return 1;
  }
  if (ctrl == 0x140) {
    *from = row + (15 - in_row);
    return 1;
  }
  if (ctrl == 0x141) {
    *from = (lane & ~7u) + (7 - (lane & 7));
    return 1;
  }
  if (ctrl == 0x142) {
    if (lane < 16) return 0;
    *from = row - 1;
    return 1;
  }
  if (ctrl == 0x143) {
    if (lane < 32) return 0;
    *from = (lane & ~31u) - 1;
    return 1;
  }
  if (ctrl >= 0x150 && ctrl <= 0x15F) {
    *from = row + (ctrl - 0x150);
    return 1;
  }
  if (ctrl >= 0x160 && ctrl <= 0x16F) {
    *from = row + (in_row ^ (ctrl - 0x160));
    return 1;
  }
  return -1;
}
// Op::off0 of a DPP instruction: row mask (bits 0-3), bank mask (4-7), bound_ctrl (8), FI (9), DPP8 (10).
// The first source as each lane sees it, and the lanes the instruction writes (Machine::dpp_shuffle): `values` is zero where nothing is read.
inline uint64_t dpp_gather(const Op& o, const Regs& R, uint64_t ex, uint32_t* values) {
  const uint32_t flags = static_cast<uint32_t>(o.off0), ctrl = o.sub;
  const bool dpp8 = flags >> 10 & 1, fi = flags >> 9 & 1, bound = flags >> 8 & 1;
  const uint32_t row_mask = flags & 15, bank_mask = flags >> 4 & 15;
  const uint32_t* src = R.vgpr[o.src[0].index];
  uint64_t writes = 0;
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned lane = static_cast<unsigned>(__builtin_ctzll(m));
    if (dpp8) {
      const uint32_t from8 = (lane & ~7u) + ((ctrl >> (3 * (lane & 7))) & 7);
      values[lane] = fi || ((ex >> from8) & 1) ? src[from8] : 0u;
      writes |= uint64_t{1} << lane;
      continue;
    }
    if (!((row_mask >> (lane >> 4)) & 1) || !((bank_mask >> ((lane >> 2) & 3)) & 1)) continue;
    uint32_t from = 0;
    const bool there = dpp_from(ctrl, lane, &from) > 0 && (fi || ((ex >> from) & 1));
    if (!there && !bound) continue;
    values[lane] = there ? src[from] : 0u;
    writes |= uint64_t{1} << lane;
  }
  return writes;
}
int h_int_dpp(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t t0[64] = {}, t1[64], t2[64], r[64];
  const uint64_t writes = dpp_gather(o, R, ex, t0);
  if (!writes) return true;
  const uint32_t* b = o.src[1].kind ? load32(o.src[1], R, t1, n) : t0;
  const uint32_t* c = o.src[2].kind ? load32(o.src[2], R, t2, n) : t0;
  reinterpret_cast<K3>(o.kern)(r, t0, b, c, n);
  merge(R.vgpr[o.dst.index], r, writes, n);
  return true;
}

// A binary32 operation: the result is thrown away if any active lane is a NaN.
[[gnu::always_inline]] inline int h_f32_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t t0[64], t1[64], t2[64], r[64];
  const uint32_t* a = load32(o.src[0], R, t0, n);
  const uint32_t* b = o.src[1].kind ? load32(o.src[1], R, t1, n) : a;
  const uint32_t* c = o.src[2].kind ? load32(o.src[2], R, t2, n) : a;
  if (o.sub == 1) {   // min and max: which zero wins when one is -0 and the other +0 is the compiler's and libm's business
    uint32_t any = 0;
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      any |= ((a[l] | b[l]) << 1) == 0 && a[l] != b[l];
    }
    if (any) return false;
  }
  reinterpret_cast<K3>(o.kern)(r, a, b, c, n);
  if (nan32(r, ex, n)) {
    if (o.nan == kNanHandBack) return false;
    if (o.nan != kNanPass) fix_nan32(o.nan, r, a, b, c, ex);
  }
  if (o.omod | o.clamp) post32(r, ex, o.omod, o.clamp);
  uint32_t* d = R.vgpr[o.dst.index];
  if (full(ex, n)) copy_row(d, r, n);
  else merge(d, r, ex, n);
  return true;
}
TIER_DUAL(h_f32)

// binary64: operands and result are register pairs.
[[gnu::always_inline]] inline int h_f64_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint64_t a[64], b[64], c[64], r[64];
  load64(o.src[0], R, a, n);
  if (o.src[1].kind) load64(o.src[1], R, b, n);
  if (o.src[2].kind) load64(o.src[2], R, c, n);
  if (o.sub == 1) {   // minimum and maximum: which zero wins when one is -0 and the other +0 is the interpreter's
    uint32_t any = 0;
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      any |= ((a[l] | b[l]) << 1) == 0 && a[l] != b[l];
    }
    if (any) return false;
  }
  reinterpret_cast<K64>(o.kern)(r, a, b, c, n);
  if (nan64(r, ex, n)) {
    if (o.nan == kNanHandBack) return false;
    if (o.nan != kNanPass) fix_nan64(o.nan, r, a, b, c, ex);
  }
  if (o.omod | o.clamp) post64(r, ex, o.omod, o.clamp);
  store64(R.vgpr[o.dst.index], R.vgpr[o.dst.index + 1], r, ex, n);
  return true;
}
TIER_DUAL(h_f64)

// binary64 add, multiply and fused multiply-add on register pairs, without the 64-bit arrays: the lane kernel reads the pairs' rows and writes a pair of
// rows of its own, and says which lanes came out a NaN. None of EXEC's: the rows are copied to the destination (all of them, or the lanes of EXEC);
// one: nothing has changed, and the general handler does the instruction (which of two NaN payloads wins is its business). A source that is a scalar or
// a constant is a value the kernel spreads itself (the kernel is the one for that set of operands); a modifier on a vector pair is built into a row of
// the high words (bit 63 is all they touch). `kern` is a kernel of Tables::f64_rows; Op::sub is the operation.
// The general handler for the instruction (with the kernel on 64-bit lanes that Tables::f64 has for it).
[[gnu::noinline]] int f64_general(const Op& o, Regs& R, bool avx2);
template <bool Avx2>
[[gnu::always_inline]] inline int h_f64r_t(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t th[3][64], rl[64], rh[64];
  uint64_t u[3] = {0, 0, 0};
  const uint32_t *lo[3] = {th[0], th[1], th[2]}, *hi[3] = {th[0], th[1], th[2]};
  for (int k = 0; k < (o.sub == 2 ? 3 : 2); ++k) {
    const Src& s = o.src[k];
    if (s.kind == kVgpr) {
      if (!s.wide) return f64_general(o, R, Avx2);   // (one register, not a pair: the general handler reads it as load64 does)
      lo[k] = R.vgpr[s.index];
      hi[k] = R.vgpr[s.index + 1];
      if (s.neg | s.abs) {   // (the modifiers act on bit 63)
        const uint32_t keep = s.abs ? 0x7FFFFFFFu : ~0u, flip = s.neg ? 0x80000000u : 0u;
        for (unsigned i = 0; i < n; ++i) th[k][i] = (hi[k][i] & keep) ^ flip;
        hi[k] = th[k];
      }
    } else {
      const uint64_t keep = s.abs ? 0x7FFFFFFFFFFFFFFFull : ~uint64_t{0}, flip = s.neg ? 0x8000000000000000ull : 0;
      u[k] = (s.kind == kConst ? s.value64 : s.kind == kSgpr ? (s.wide ? static_cast<uint64_t>(R.sgpr[s.index + 1]) << 32 | R.sgpr[s.index] : R.sgpr[s.index]) : 0) & keep;
      u[k] ^= s.kind == kConst || s.kind == kSgpr ? flip : 0;   // (anything else reads as zero, whatever the modifiers: load64's)
    }
  }
  const uint64_t nan = reinterpret_cast<F64RowsFn>(o.kern)(rl, rh, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], u, n);
  if (nan & ex) return f64_general(o, R, Avx2);
  uint32_t* dlo = R.vgpr[o.dst.index];
  uint32_t* dhi = R.vgpr[o.dst.index + 1];
  if (full(ex, n)) {
    copy_row(dlo, rl, n);
    copy_row(dhi, rh, n);
  } else if (__builtin_popcountll(ex) > 8) {
    merge(dlo, rl, ex, n);
    merge(dhi, rh, ex, n);
  } else {
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      dlo[l] = rl[l], dhi[l] = rh[l];
    }
  }
  return true;
}
int f64_general(const Op& o, Regs& R, bool avx2) {
  Op general = o;
  general.kern = reinterpret_cast<AnyFn>(tables().f64[o.sub == 0 ? DADD : o.sub == 1 ? DMUL : DFMA]);
  return avx2 ? h_f64_avx2(general, R) : h_f64_base(general, R);
}
int h_f64r_base(const Op& o, Regs& R) { return h_f64r_t<false>(o, R); }
[[gnu::target("avx2,fma,popcnt")]] int h_f64r_avx2(const Op& o, Regs& R) { return h_f64r_t<true>(o, R); }

// A 64-bit integer operation: sources by `sub` bits (bit k set: source k is a 64-bit pair, else a 32-bit count).
[[gnu::always_inline]] inline int h_l64_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint64_t a[64], b[64], c[64], r[64];
  const Src* s[3] = {&o.src[0], &o.src[1], &o.src[2]};
  uint64_t* t[3] = {a, b, c};
  for (int k = 0; k < 3; ++k) {
    if (!s[k]->kind) continue;
    if (o.sub >> k & 1) load64(*s[k], R, t[k], n);
    else load32_to64(*s[k], R, t[k], n);
  }
  reinterpret_cast<K64>(o.kern)(r, a, b, c, n);
  store64(R.vgpr[o.dst.index], R.vgpr[o.dst.index + 1], r, ex, n);
  return true;
}
TIER_DUAL(h_l64)

// A 64-bit lane function whose result is one 32-bit word (the conversions of a double to a float or an integer, frexp's exponent): the low word of
// the kernel's result, written to the destination register for the lanes of EXEC.
[[gnu::always_inline]] inline int h_l64n_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint64_t a[64], b[64], c[64], r[64];
  alignas(32) uint32_t w[64];
  const Src* s[3] = {&o.src[0], &o.src[1], &o.src[2]};
  uint64_t* t[3] = {a, b, c};
  for (int k = 0; k < 3; ++k) {
    if (!s[k]->kind) continue;
    if (o.sub >> k & 1) load64(*s[k], R, t[k], n);
    else load32_to64(*s[k], R, t[k], n);
  }
  reinterpret_cast<K64>(o.kern)(r, a, b, c, n);
  for (unsigned i = 0; i < n; ++i) w[i] = static_cast<uint32_t>(r[i]);
  uint32_t* d = R.vgpr[o.dst.index];
  if (full(ex, n)) copy_row(d, w, n);
  else merge(d, w, ex, n);
  return true;
}
TIER_DUAL(h_l64n)

// v_mov_b64 of a register pair (as two row copies), a scalar pair or a constant (spread into the destination's rows), or the general path.
[[gnu::always_inline]] inline int h_mov64_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  uint32_t* lo = R.vgpr[o.dst.index];
  uint32_t* hi = R.vgpr[o.dst.index + 1];
  const Src& s = o.src[0];
  if (full(ex, n) && !(s.neg | s.abs)) {
    if (s.kind == kVgpr && s.wide && (s.index == o.dst.index || s.index + 1 < o.dst.index || o.dst.index + 1 < s.index)) {   // (a pair that overlaps the destination in part is read whole first, below)
      if (s.index != o.dst.index) {
        copy_row(lo, R.vgpr[s.index], n);
        copy_row(hi, R.vgpr[s.index + 1], n);
      }
      return true;
    }
    if (s.kind == kConst || s.kind == kSgpr) {
      const uint64_t v = rd(R, s);
      fill(lo, static_cast<uint32_t>(v), n);
      fill(hi, static_cast<uint32_t>(v >> 32), n);
      return true;
    }
  }
  alignas(32) uint64_t a[64];
  load64(s, R, a, n);
  store64(lo, hi, a, ex, n);
  return true;
}
TIER_DUAL(h_mov64)

// Packed binary32 (v_pk_add_f32, _mul_f32, _fma_f32): two floats in a register pair, each half its own arithmetic. `sub`: bit k is
// the half of source k the low result reads (op_sel), bit 3 + k the half the high result reads (op_sel_hi), bits 6 + k and 9 + k
// negate it for the low and the high result. Both halves are worked out before either is written.
int h_pk(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t t[2][3][64], r[2][64];
  const uint32_t* p[2][3];
  for (int h = 0; h < 2; ++h)
    for (int k = 0; k < 3; ++k) {
      p[h][k] = t[h][0];
      const Src& s = o.src[k];
      if (!s.kind) continue;
      const unsigned top = o.sub >> (h * 3 + k) & 1;
      const bool neg = o.sub >> (6 + h * 3 + k) & 1;
      uint32_t* dst = t[h][k];
      p[h][k] = dst;
      if (s.kind == kVgpr) {
        const uint32_t* row = R.vgpr[s.index + top];
        for (unsigned i = 0; i < n; ++i) dst[i] = neg ? row[i] ^ 0x80000000u : row[i];
      } else {
        uint32_t v = s.value;
        if (s.kind == kSgpr)
          v = s.wide ? R.sgpr[s.index + top] : (top ? 0u : R.sgpr[s.index]);
        if (neg) v ^= 0x80000000u;
        fill(dst, v, n);
      }
    }
  const K3 kern = reinterpret_cast<K3>(o.kern);
  kern(r[0], p[0][0], p[0][1], p[0][2], n);
  kern(r[1], p[1][0], p[1][1], p[1][2], n);
  if (nan32(r[0], ex, n) || nan32(r[1], ex, n)) {
    if (o.nan == kNanHandBack) return false;
    for (int h = 0; h < 2; ++h) fix_nan32(o.nan, r[h], p[h][0], p[h][1], p[h][2], ex);
  }
  if (o.clamp) post32(r[0], ex, 0, true), post32(r[1], ex, 0, true);
  uint32_t* d0 = R.vgpr[o.dst.index];
  uint32_t* d1 = R.vgpr[o.dst.index + 1];
  if (full(ex, n)) {
    copy_row(d0, r[0], n);
    copy_row(d1, r[1], n);
  } else {
    merge(d0, r[0], ex, n);
    merge(d1, r[1], ex, n);
  }
  return true;
}

// The packed 16-bit integer instructions (v_pk_add_u16 ...): each result half is the kernel over the halves the sources supply (Op::sub holds,
// for source k, the half result 0 reads at bit k and the half result 1 reads at bit 3 + k; a constant's two halves are in Src::value and value64).
// The kernels give the half in the low 16 bits; the two make a lane's word.
int h_pk16(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t t[2][3][64], r[2][64], d[64];
  const uint32_t* p[2][3];
  for (int h = 0; h < 2; ++h)
    for (int k = 0; k < 3; ++k) {
      p[h][k] = t[h][0];
      const Src& s = o.src[k];
      if (!s.kind) continue;
      const unsigned shift = 16 * (o.sub >> (h * 3 + k) & 1);
      uint32_t* dst = t[h][k];
      p[h][k] = dst;
      if (s.kind == kVgpr) {
        const uint32_t* row = R.vgpr[s.index];
        for (unsigned i = 0; i < n; ++i) dst[i] = (row[i] >> shift) & 0xFFFFu;
      } else if (s.kind == kSgpr) {
        fill(dst, (R.sgpr[s.index] >> shift) & 0xFFFFu, n);
      } else {
        fill(dst, static_cast<uint32_t>(h ? s.value64 : s.value) & 0xFFFFu, n);
      }
    }
  const K3 kern = reinterpret_cast<K3>(o.kern);
  kern(r[0], p[0][0], p[0][1], p[0][2], n);
  kern(r[1], p[1][0], p[1][1], p[1][2], n);
  for (unsigned i = 0; i < n; ++i) d[i] = (r[0][i] & 0xFFFFu) | r[1][i] << 16;
  uint32_t* dd = R.vgpr[o.dst.index];
  if (full(ex, n)) copy_row(dd, d, n);
  else merge(dd, d, ex, n);
  return true;
}

// v_pk_mov_b32: the low register of the destination pair from the first source, the high from the second, each the register of its pair
// op_sel names (`sub` bits 0 and 1). Both are read before either is written.
int h_pkmov(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t w[2][64];
  for (int k = 0; k < 2; ++k) {
    const Src& s = o.src[k];
    const unsigned top = o.sub >> k & 1;
    if (s.kind == kVgpr) {
      copy_row(w[k], R.vgpr[s.index + top], n);
    } else {
      const uint32_t v = s.kind == kSgpr ? (s.wide ? R.sgpr[s.index + top] : top ? 0u : R.sgpr[s.index]) : s.value;
      fill(w[k], v, n);
    }
  }
  uint32_t* d0 = R.vgpr[o.dst.index];
  uint32_t* d1 = R.vgpr[o.dst.index + 1];
  if (full(ex, n)) {
    copy_row(d0, w[0], n);
    copy_row(d1, w[1], n);
  } else {
    merge(d0, w[0], ex, n);
    merge(d1, w[1], ex, n);
  }
  return true;
}

// v_permlane16_b32 and v_permlanex16_b32 (RDNA): each lane of a row of 16 reads the lane of its row (of the other row of its pair, for the x form) that its
// four bits of the 64-bit selector name; a source lane that is off is read where FI (Op::sub bit 0) says so, else the lane gets zero where BOUND_CTRL
// (bit 1) says so and keeps its value where neither does. Op::off0 is 1 for the x form.
int h_permlane(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const uint64_t sel = (rd(R, o.src[1]) & 0xFFFFFFFFu) | (rd(R, o.src[2]) & 0xFFFFFFFFu) << 32;
  const bool fi = o.sub & 1, bound = o.sub >> 1 & 1;
  const unsigned cross = o.off0 ? 1 : 0;
  alignas(32) uint32_t from[64];
  std::memcpy(from, R.vgpr[o.src[0].index], sizeof from);
  uint32_t* const out = R.vgpr[o.dst.index];
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned lane = static_cast<unsigned>(__builtin_ctzll(m));
    const unsigned s = ((lane / 16) ^ cross) * 16 + static_cast<unsigned>((sel >> (4 * (lane % 16))) & 0xF);
    if (fi || (ex >> s & 1)) out[lane] = from[s];
    else if (bound) out[lane] = 0;
  }
  return true;
}

// v_div_scale_*, v_div_fmas_* and v_div_fixup_* (gcn_divide.hpp): per lane, the interpreter's divide_step. `sub`: 0 scale, 1 fmas, 2 fixup,
// plus 4 for the double forms. The scale writes a mask of the lanes whose condition holds to its scalar destination, whatever EXEC says;
// fmas reads VCC. The sources carry the float modifiers; the result is written whole (no output modifier or clamp).
int h_div(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  const unsigned what = o.sub & 3;
  const bool dbl = o.sub & 4;
  uint64_t cond = 0;
  if (ex) {
    const unsigned n = lanes_of(ex);
    if (dbl) {
      alignas(32) uint64_t a[64], b[64], c[64], r[64];
      load64(o.src[0], R, a, n), load64(o.src[1], R, b, n), load64(o.src[2], R, c, n);
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        double x, y, z, v;
        std::memcpy(&x, &a[l], 8), std::memcpy(&y, &b[l], 8), std::memcpy(&z, &c[l], 8);
        if (what == 0) {
          const auto res = gcn::divide::div_scale(x, y, z);
          v = res.value;
          if (res.vcc) cond |= uint64_t{1} << l;
        } else if (what == 1) {
          v = gcn::divide::div_fmas(x, y, z, (*R.vcc >> l) & 1);
          if (v != v) return false;   // (a fused multiply-add with NaNs: which payload comes through is the compiler's choice; the interpreter's)
        } else {
          v = gcn::divide::div_fixup(x, y, z);
        }
        std::memcpy(&r[l], &v, 8);
      }
      uint32_t* lo = R.vgpr[o.dst.index];
      uint32_t* hi = R.vgpr[o.dst.index + 1];
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        lo[l] = static_cast<uint32_t>(r[l]), hi[l] = static_cast<uint32_t>(r[l] >> 32);
      }
    } else {
      alignas(32) uint32_t t0[64], t1[64], t2[64];
      const uint32_t* a = load32(o.src[0], R, t0, n);
      const uint32_t* b = load32(o.src[1], R, t1, n);
      const uint32_t* c = load32(o.src[2], R, t2, n);
      uint32_t* d = R.vgpr[o.dst.index];
      alignas(32) uint32_t rr[64];
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        float x, y, z, v;
        std::memcpy(&x, &a[l], 4), std::memcpy(&y, &b[l], 4), std::memcpy(&z, &c[l], 4);
        if (what == 0) {
          const auto res = gcn::divide::div_scale(x, y, z);
          v = res.value;
          if (res.vcc) cond |= uint64_t{1} << l;
        } else if (what == 1) {
          v = gcn::divide::div_fmas(x, y, z, (*R.vcc >> l) & 1);
          if (v != v) return false;   // (see above)
        } else {
          v = gcn::divide::div_fixup(x, y, z);
        }
        std::memcpy(&rr[l], &v, 4);
      }
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        d[l] = rr[l];
      }
    }
  }
  if (what == 0 && o.dst2.kind) wr(R, o.dst2, cond);
  return true;
}

// v_fma_mix_f32 / v_fma_mixlo_f16 / v_fma_mixhi_f16: each source is a float, or -- where its op_sel_hi bit is set -- a half, the one op_sel names;
// a fused multiply-add of floats (nan_order.hpp's rule for NaNs), clamped if asked, then the whole of the destination (mix), or rounded to a
// half into its low (mixlo) or high (mixhi) half with the other half kept. `sub`: bit k a half source, bit 3 + k its high half, bit 6 + k negated,
// bit 9 + k absolute value, bits 12-13 the kind (0 mix, 1 mixlo, 2 mixhi).
int h_fma_mix(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint32_t t[3][64];
  const uint32_t* p[3];
  for (int k = 0; k < 3; ++k) p[k] = load32(o.src[k], R, t[k], n);
  const unsigned kind = o.sub >> 12 & 3;
  uint32_t* d = R.vgpr[o.dst.index];
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    float v[3];
    for (int k = 0; k < 3; ++k) {
      float f;
      if (o.sub >> k & 1) {
        f = vgpu::amd::packed::half_to_float(static_cast<uint16_t>(p[k][l] >> ((o.sub >> (3 + k) & 1) ? 16 : 0)));
      } else {
        std::memcpy(&f, &p[k][l], 4);
      }
      if (o.sub >> (9 + k) & 1) f = std::fabs(f);
      v[k] = (o.sub >> (6 + k) & 1) ? -f : f;
    }
    float r = vgpu::amd::nan_fma<2>(v[0], v[1], v[2]);
    if (o.clamp) r = std::isnan(r) || r <= 0.0f ? 0.0f : r > 1.0f ? 1.0f : r;
    if (kind == 0) {
      std::memcpy(&d[l], &r, 4);
    } else {
      const uint32_t bits = vgpu::amd::packed::float_to_half(r);
      d[l] = kind == 1 ? (d[l] & 0xFFFF0000u) | bits : (d[l] & 0x0000FFFFu) | bits << 16;
    }
  }
  return true;
}

// v_mad_u64_u32: a 32x32 product added to a 64-bit value; the carry out of 64 bits is the scalar destination's bit.
[[gnu::always_inline]] inline int h_mad_u64_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  uint64_t carry = 0;
  if (ex) {
    const unsigned n = lanes_of(ex);
    alignas(32) uint32_t t0[64], t1[64], tcl[64], tch[64], rl[64], rh[64];
    const uint32_t* a = load32(o.src[0], R, t0, n);
    const uint32_t* b = load32(o.src[1], R, t1, n);
    const Src& s2 = o.src[2];
    const uint32_t *cl, *ch;
    if (s2.kind == kVgpr && s2.wide && !(s2.neg | s2.abs)) {   // (the pair's two rows as they are)
      cl = R.vgpr[s2.index], ch = R.vgpr[s2.index + 1];
    } else {
      alignas(32) uint64_t c[64];
      load64(s2, R, c, n);
      for (unsigned i = 0; i < n; ++i) tcl[i] = static_cast<uint32_t>(c[i]), tch[i] = static_cast<uint32_t>(c[i] >> 32);
      cl = tcl, ch = tch;
    }
    uint32_t* lo = R.vgpr[o.dst.index];
    uint32_t* hi = R.vgpr[o.dst.index + 1];
    const Mad64Fn k = reinterpret_cast<Mad64Fn>(o.kern);
    if (full(ex, n)) {
      carry = k(lo, hi, a, b, cl, ch, n);
    } else if (__builtin_popcountll(ex) > 8) {
      carry = k(rl, rh, a, b, cl, ch, n) & ex;
      merge(lo, rl, ex, n);
      merge(hi, rh, ex, n);
    } else {   // (a few lanes: the lanes' own loop is cheaper than the row kernel and the two merges)
      uint64_t r[8];
      unsigned idx[8], cnt = 0;
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        const unsigned __int128 p = static_cast<unsigned __int128>(a[l]) * b[l] + (static_cast<uint64_t>(ch[l]) << 32 | cl[l]);
        r[cnt] = static_cast<uint64_t>(p), idx[cnt++] = l;
        if (static_cast<uint64_t>(p >> 64)) carry |= uint64_t{1} << l;
      }
      for (unsigned i = 0; i < cnt; ++i) lo[idx[i]] = static_cast<uint32_t>(r[i]), hi[idx[i]] = static_cast<uint32_t>(r[i] >> 32);
    }
  }
  if (o.dst2.kind) wr(R, o.dst2, carry);
  return true;
}
TIER_DUAL(h_mad_u64)

// v_add_co_u32, v_addc_co_u32, v_sub_co_u32, v_subb_co_u32 and the reversed ones (carry_alu): the carry or borrow of each lane into a
// scalar mask, written whatever EXEC is. `sub`: bit 0 add, bit 1 takes a carry in (source 2, a mask), bit 2 reversed.
[[gnu::always_inline]] inline int h_carry_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  const uint64_t carried_in = (o.sub & 2) ? rd(R, o.src[2]) : 0;
  uint64_t out = 0;
  if (ex) {
    const unsigned n = lanes_of(ex);
    alignas(32) uint32_t t0[64], t1[64], r[64];
    const uint32_t* a = load32(o.src[0], R, t0, n);
    const uint32_t* b = load32(o.src[1], R, t1, n);
    uint32_t* d = R.vgpr[o.dst.index];
    const CarryFn k = reinterpret_cast<CarryFn>(o.kern);
    if (full(ex, n)) {
      out = k(d, a, b, carried_in, n);
    } else {
      out = k(r, a, b, carried_in, n) & ex;
      merge(d, r, ex, n);
    }
  }
  wr(R, o.dst2, out);
  return 1;
}
TIER_DUAL(h_carry)

// v_cndmask_b32: lane l takes source 1 where bit l of the scalar mask is set, else source 0 (modifiers on either).
[[gnu::always_inline]] inline int h_cndmask_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  const uint64_t cond = rd(R, o.src[2]);
  alignas(32) uint32_t t0[64], t1[64], r[64];
  const uint32_t* a = load32(o.src[0], R, t0, n);
  const uint32_t* b = load32(o.src[1], R, t1, n);
  const MaskTable& tab = mask_table();
  for (unsigned i = 0; i < n; i += 8) {
    V8u m, x, y;
    std::memcpy(&m, tab.m[(cond >> i) & 0xFF], 32);
    std::memcpy(&x, a + i, 32);
    std::memcpy(&y, b + i, 32);
    x = (y & m) | (x & ~m);
    std::memcpy(r + i, &x, 32);
  }
  uint32_t* d = R.vgpr[o.dst.index];
  if (full(ex, n)) copy_row(d, r, n);
  else merge(d, r, ex, n);
  return true;
}
TIER_DUAL(h_cndmask)

// v_readfirstlane_b32: the first active lane's value (lane 0 with none) into a scalar destination.
int h_readfirstlane(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  const unsigned lane = ex ? static_cast<unsigned>(__builtin_ctzll(ex)) : 0;
  wr(R, o.dst, R.vgpr[o.src[0].index][lane]);
  return true;
}

// A comparison: one bit a lane, into a scalar destination and, for the X forms, EXEC. `kern` is the test for the first n lanes.
[[gnu::always_inline]] inline int h_cmp_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  uint64_t result = 0;
  if (ex) {
    const unsigned n = lanes_of(ex);
    alignas(32) uint32_t t0[64], t1[64];
    const uint32_t* a = load32(o.src[0], R, t0, n);
    const uint32_t* b = load32(o.src[1], R, t1, n);
    result = reinterpret_cast<CmpFn>(o.kern)(a, b, n) & ex;
  }
  if (o.dst.kind) wr(R, o.dst, result);
  if (o.sub) *R.exec = result;   // the X forms
  return true;
}
TIER_DUAL(h_cmp)


int h_never(const Op&, Regs&) { return 0; }

// ---- LDS
enum DsKind : uint32_t { kDsRead, kDsWrite, kDsRead2, kDsWrite2, kDsBperm, kDsAdd, kDsSwizzle, kDsPerm };
[[gnu::always_inline]] inline int h_ds_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (o.store && *R.store_gate && ex && R.nwaves > 1) {
    *R.store_gate = false;
    *R.soft_yield = true;
    return 3;
  }
  if (!ex) return 1;
  const uint32_t* addr = R.vgpr[o.src[0].index];
  if (o.sub == kDsBperm) {
    // A lane reads what another lane holds (a lane that is off gives zero); the data is read across the wave before any lane is written.
    alignas(32) uint32_t before[64];
    std::memcpy(before, R.vgpr[o.src[1].index], sizeof before);
    const uint32_t x = static_cast<uint32_t>(o.off0);
    uint32_t* const out = R.vgpr[o.dst.index];
    const unsigned lanes = R.lanes;
    if (full(ex, lanes)) {   // (every lane on: every lane reads, and the source lane is on)
      for (unsigned l = 0; l < lanes; ++l) out[l] = before[((addr[l] + x) >> 2) & (lanes - 1)];
      return 1;
    }
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      const uint32_t from = ((addr[l] + x) >> 2) & (lanes - 1);
      out[l] = ex >> from & 1 ? before[from] : 0;
    }
    return 1;
  }
  if (o.sub == kDsSwizzle) {
    // ds_swizzle_b32: a lane reads another lane of its group of 32 as the offset's pattern says (a lane that is off gives zero); the data is read
    // across the wave before any lane is written (the interpreter's swizzle_source).
    alignas(32) uint32_t before[64];
    std::memcpy(before, R.vgpr[o.src[0].index], sizeof before);
    const uint32_t pattern = static_cast<uint32_t>(o.off0);
    uint32_t* const out = R.vgpr[o.dst.index];
    for (uint64_t m = ex; m; m &= m - 1) {
      const uint32_t l = static_cast<uint32_t>(__builtin_ctzll(m));
      uint32_t from;
      if ((pattern & 0xFF00) == 0x8000) {
        from = (l & ~3u) + ((pattern >> (2 * (l & 3))) & 3);
      } else if ((pattern & 0xE000) == 0xC000) {
        const uint32_t by = (pattern >> 5) & 0x1F, right = (pattern >> 10) & 1;
        from = (l & ~31u) | ((right ? l - by : l + by) & 31u);
      } else {
        from = (l & ~31u) | ((((l & 31u) & (pattern & 0x1F)) | ((pattern >> 5) & 0x1F)) ^ ((pattern >> 10) & 0x1F));
      }
      out[l] = ex >> from & 1 ? before[from] : 0;
    }
    return 1;
  }
  if (o.sub == kDsPerm) {
    // ds_permute_b32: each lane that is on sends its value to the lane its address names (the higher-numbered sender's lands last), and a lane no
    // one sent to gets zero.
    uint32_t sent[64] = {};
    const uint32_t* const data = R.vgpr[o.src[1].index];
    const uint32_t x = static_cast<uint32_t>(o.off0);
    const unsigned lanes = R.lanes;
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      sent[((addr[l] + x) >> 2) & (lanes - 1)] = data[l];
    }
    uint32_t* const out = R.vgpr[o.dst.index];
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      out[l] = sent[l];
    }
    return 1;
  }
  const uint32_t x0 = static_cast<uint32_t>(o.off0), x1 = static_cast<uint32_t>(o.off1);
  const unsigned w = o.words;
  const uint64_t size = R.lds_size;
  const uint64_t span = o.sub == kDsRead || o.sub == kDsWrite ? 4ull * w : 4;
  uint8_t* lds = R.lds;
  const unsigned n = R.lanes;
  const bool all_on = full(ex, n);
  // The lanes that are on a single run of consecutive lanes (all of them, or a prefix as a reduction's last steps leave), their addresses a
  // run of consecutive words (or, for the wide forms, of consecutive groups of w words): the whole run is one copy to or from the LDS (a
  // wide access also moves the words between the lane-major layout of the LDS and the register rows). Only the forms whose copies cannot
  // overlap one another (the plain read and write, and the two-offset ones when the two runs are apart: otherwise the order the lanes store
  // in decides the bytes).
  const unsigned first = static_cast<unsigned>(__builtin_ctzll(ex));
  const uint64_t shifted = ex >> first;
  if ((shifted & (shifted + 1)) == 0 && (o.sub == kDsRead || o.sub == kDsWrite || o.sub == kDsRead2 || o.sub == kDsWrite2) && w <= 4) {
    const unsigned cnt = static_cast<unsigned>(__builtin_popcountll(ex));
    const uint32_t stride = o.sub == kDsRead || o.sub == kDsWrite ? 4u * w : 4u;
    const uint32_t base = addr[first] + x0;
    uint32_t diff = 0;
    for (unsigned l = 0; l < cnt; ++l) diff |= (addr[first + l] + x0) ^ (base + stride * l);
    const bool two = o.sub == kDsRead2 || o.sub == kDsWrite2;
    const uint32_t base1 = base + (x1 - x0);
    const uint64_t run = uint64_t{stride} * cnt;
    const bool apart = !two || (base1 > base ? base1 - base : base - base1) >= run;
    if (!diff && apart) {
      // (the run, and the second run, inside the LDS: the same test as every lane's, the addresses being consecutive and not wrapping)
      if (uint64_t{base} + run > size || (two && uint64_t{base1} + run > size)) return 0;
      switch (o.sub) {
        case kDsRead:
          if (w == 1) {
            copy_bytes(R.vgpr[o.dst.index] + first, lds + base, 4u * cnt);
          } else if (w == 2 || w == 4) {
            uint32_t* rows[4];
            for (unsigned k = 0; k < w; ++k) rows[k] = R.vgpr[o.dst.index + k] + first;
            ds_run_read(rows, lds + base, cnt, w);
          } else {
            alignas(32) uint32_t tmp[256];
            copy_bytes(tmp, lds + base, 4u * w * cnt);
            for (unsigned k = 0; k < w; ++k) {
              uint32_t* const out = R.vgpr[o.dst.index + k] + first;
              for (unsigned l = 0; l < cnt; ++l) out[l] = tmp[w * l + k];
            }
          }
          return 1;
        case kDsWrite:
          if (w == 1) {
            copy_bytes(lds + base, R.vgpr[o.src[1].index] + first, 4u * cnt);
          } else if (w == 2 || w == 4) {
            uint32_t* rows[4];
            for (unsigned k = 0; k < w; ++k) rows[k] = R.vgpr[o.src[1].index + k] + first;
            ds_run_write(rows, lds + base, cnt, w);
          } else {
            alignas(32) uint32_t tmp[256];
            for (unsigned k = 0; k < w; ++k) {
              const uint32_t* const in = R.vgpr[o.src[1].index + k] + first;
              for (unsigned l = 0; l < cnt; ++l) tmp[w * l + k] = in[l];
            }
            copy_bytes(lds + base, tmp, 4u * w * cnt);
          }
          return 1;
        case kDsRead2:
          copy_bytes(R.vgpr[o.dst.index] + first, lds + base, 4u * cnt);
          copy_bytes(R.vgpr[o.dst.index + 1] + first, lds + base1, 4u * cnt);
          return 1;
        default:
          copy_bytes(lds + base, R.vgpr[o.src[1].index] + first, 4u * cnt);
          copy_bytes(lds + base1, R.vgpr[o.src[2].index] + first, 4u * cnt);
          return 1;
      }
    }
  }
  // Every lane must land inside the LDS (an access past it reads zeros and writes nowhere on a card, or is an error: the
  // interpreter's business); nothing has changed yet when one does not.
  // (Branch-free: the lanes' tests are ORed and looked at once.)
  {
    unsigned past = 0;
    if (all_on) {
      for (unsigned l = 0; l < n; ++l) past |= static_cast<uint64_t>(static_cast<uint32_t>(addr[l] + x0)) + span > size;
      if (o.sub == kDsRead2 || o.sub == kDsWrite2)
        for (unsigned l = 0; l < n; ++l) past |= static_cast<uint64_t>(static_cast<uint32_t>(addr[l] + x1)) + span > size;
    } else if (o.sub == kDsRead2 || o.sub == kDsWrite2) {
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        past |= (static_cast<uint64_t>(static_cast<uint32_t>(addr[l] + x0)) + span > size) | (static_cast<uint64_t>(static_cast<uint32_t>(addr[l] + x1)) + span > size);
      }
    } else {
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        past |= static_cast<uint64_t>(static_cast<uint32_t>(addr[l] + x0)) + span > size;
      }
    }
    if (past) return 0;
  }
  if (o.sub == kDsAdd) {   // ds_add_u32: each lane in turn adds its register to the word (lanes that share a word add one after the other)
    const uint32_t* const in = R.vgpr[o.src[1].index];
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      uint8_t* const p = lds + static_cast<uint32_t>(addr[l] + x0);
      uint32_t v;
      std::memcpy(&v, p, 4);
      v += in[l];
      std::memcpy(p, &v, 4);
    }
    return 1;
  }
  if (all_on && w == 1 && (o.sub == kDsRead || o.sub == kDsWrite)) {   // (every lane on, any addresses: a counted loop, no mask to walk)
    if (o.sub == kDsRead) {
      uint32_t* const out = R.vgpr[o.dst.index];
      for (unsigned l = 0; l < n; ++l) std::memcpy(&out[l], lds + static_cast<uint32_t>(addr[l] + x0), 4);
    } else {
      const uint32_t* const in = R.vgpr[o.src[1].index];
      for (unsigned l = 0; l < n; ++l) std::memcpy(lds + static_cast<uint32_t>(addr[l] + x0), &in[l], 4);
    }
    return 1;
  }
  if (all_on && (w == 4 || w == 2) && (o.sub == kDsRead || o.sub == kDsWrite)) {   // (every lane on, any addresses: four lanes at a time, stored in lane order)
    uint32_t* rows[4];
    for (unsigned k = 0; k < w; ++k) rows[k] = R.vgpr[(o.sub == kDsRead ? o.dst.index : o.src[1].index) + k];
    if (o.sub == kDsRead) ds_all_read(rows, lds, addr, x0, w, n);
    else ds_all_write(rows, lds, addr, x0, w, n);
    return 1;
  }
  switch (o.sub) {
    case kDsRead:
      if (w == 1) {   // (the common one: a word a lane)
        uint32_t* const out = R.vgpr[o.dst.index];
        for (uint64_t m = ex; m; m &= m - 1) {
          const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
          std::memcpy(&out[l], lds + static_cast<uint32_t>(addr[l] + x0), 4);
        }
        return 1;
      }
      {
        uint32_t* rows[4];
        for (unsigned k = 0; k < w && k < 4; ++k) rows[k] = R.vgpr[o.dst.index + k];
        if (w == 2) ds_lanes_read(rows, lds, addr, x0, ex, 2);
        else if (w == 3) ds_lanes_read(rows, lds, addr, x0, ex, 3);
        else if (w == 4) ds_lanes_read(rows, lds, addr, x0, ex, 4);
        else
          for (uint64_t m = ex; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
            const uint8_t* p = lds + static_cast<uint32_t>(addr[l] + x0);
            for (unsigned k = 0; k < w; ++k) {
              uint32_t v;
              std::memcpy(&v, p + 4 * k, 4);
              R.vgpr[o.dst.index + k][l] = v;
            }
          }
      }
      return 1;
    case kDsWrite:
      if (w == 1) {
        const uint32_t* const in = R.vgpr[o.src[1].index];
        for (uint64_t m = ex; m; m &= m - 1) {
          const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
          std::memcpy(lds + static_cast<uint32_t>(addr[l] + x0), &in[l], 4);
        }
        return 1;
      }
      {
        uint32_t* rows[4];
        for (unsigned k = 0; k < w && k < 4; ++k) rows[k] = R.vgpr[o.src[1].index + k];
        if (w == 2) ds_lanes_write(rows, lds, addr, x0, ex, 2);
        else if (w == 3) ds_lanes_write(rows, lds, addr, x0, ex, 3);
        else if (w == 4) ds_lanes_write(rows, lds, addr, x0, ex, 4);
        else
          for (uint64_t m = ex; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
            uint8_t* p = lds + static_cast<uint32_t>(addr[l] + x0);
            for (unsigned k = 0; k < w; ++k) std::memcpy(p + 4 * k, &R.vgpr[o.src[1].index + k][l], 4);
          }
      }
      return 1;
    case kDsRead2:
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        uint32_t v0, v1;
        std::memcpy(&v0, lds + static_cast<uint32_t>(addr[l] + x0), 4);
        std::memcpy(&v1, lds + static_cast<uint32_t>(addr[l] + x1), 4);
        R.vgpr[o.dst.index][l] = v0;
        R.vgpr[o.dst.index + 1][l] = v1;
      }
      return 1;
    default:
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        std::memcpy(lds + static_cast<uint32_t>(addr[l] + x0), &R.vgpr[o.src[1].index][l], 4);
        std::memcpy(lds + static_cast<uint32_t>(addr[l] + x1), &R.vgpr[o.src[2].index][l], 4);
      }
      return 1;
  }
}
TIER_DUAL(h_ds)

// A 64-bit comparison (integers, or binary64 with the source modifiers): `kern` is the test for the first n lanes.
[[gnu::always_inline]] inline int h_cmp64_body(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  uint64_t result = 0;
  if (ex) {
    const unsigned n = lanes_of(ex);
    alignas(32) uint64_t a[64], b[64];
    load64(o.src[0], R, a, n);
    load64(o.src[1], R, b, n);
    result = reinterpret_cast<Cmp64Fn>(o.kern)(a, b, n) & ex;
  }
  if (o.dst.kind) wr(R, o.dst, result);
  if (o.sub) *R.exec = result;
  return 1;
}
TIER_DUAL(h_cmp64)

// v_readlane_b32: one lane (the selector's low bits) of a register into a scalar destination, whatever EXEC says.
int h_readlane(const Op& o, Regs& R) {
  const uint32_t which = static_cast<uint32_t>(rd(R, o.src[1])) & (R.lanes - 1);
  wr(R, o.dst, R.vgpr[o.src[0].index][which]);
  return 1;
}
// v_writelane_b32: a scalar value into one lane of a register, whatever EXEC says.
int h_writelane(const Op& o, Regs& R) {
  const uint32_t lane = static_cast<uint32_t>(rd(R, o.src[1])) & 63;
  R.vgpr[o.dst.index][lane] = static_cast<uint32_t>(rd(R, o.src[0]));
  return 1;
}

// ---- Scalar handlers

// Two sources, one destination, SCC from the result (where the instruction sets it). Each is also built for the shapes of operands the
// compilers' scalar code is nearly all made of -- 32-bit scalar registers and constants, no pair, no VCC or EXEC -- with the operand reads
// and the write fixed at compile time (`s2_variants`, picked when the instruction is decoded): the general handler's switches on the
// operand's kind cost about as much as the operation.
// (Shapes: 1 a scalar register, one word; 2 a constant.)
template <int K>
[[gnu::always_inline]] inline uint64_t rd_k(const Regs& R, const Src& s) {
  if constexpr (K == 1) return R.sgpr[s.index];
  else if constexpr (K == 2) return s.value64;
  else return rd(R, s);
}
template <int K>
[[gnu::always_inline]] inline void wr_k(Regs& R, const Src& s, uint64_t v) {
  if constexpr (K == 1) R.sgpr[s.index] = static_cast<uint32_t>(v);
  else wr(R, s, v);
}
struct S2Variants {
  Fn ss, sc, cc;   // (A a register, B a register), (A a register, B a constant), (both constants; a one-source handler's constant)
};
std::unordered_map<Fn, S2Variants>& s2_variants() {
  static std::unordered_map<Fn, S2Variants> m;
  return m;
}
#define TIER_S2_BODY(name, ka, kb, ...)                                                                                      \
  [[maybe_unused]] const uint64_t a = rd_k<ka>(R, o.src[0]), b = rd_k<kb>(R, o.src[1]);                                       \
  uint64_t v = 0;                                                                                                            \
  __VA_ARGS__;                                                                                                               \
  wr_k<(ka != 0 && kb != 0) ? 1 : 0>(R, o.dst, v);                                                                           \
  return true;
#define TIER_S2(name, ...)                                                                                                   \
  int name(const Op& o, Regs& R) { TIER_S2_BODY(name, 0, 0, __VA_ARGS__) }                                                   \
  template <int KA, int KB>                                                                                                  \
  int name##_k(const Op& o, Regs& R) { TIER_S2_BODY(name, KA, KB, __VA_ARGS__) }                                             \
  [[maybe_unused]] const bool name##_registered = (s2_variants()[name] = {name##_k<1, 1>, name##_k<1, 2>, name##_k<2, 2>}, true);
#define TIER_SCC(v) (*R.scc = (v))
TIER_S2(s_mov, v = a)
TIER_S2(s_add_u32, {
  const uint64_t sum = static_cast<uint32_t>(a) + static_cast<uint64_t>(static_cast<uint32_t>(b));
  v = static_cast<uint32_t>(sum);
  TIER_SCC(sum >> 32);
})
TIER_S2(s_addc_u32, {
  const uint64_t sum = static_cast<uint32_t>(a) + static_cast<uint64_t>(static_cast<uint32_t>(b)) + *R.scc;
  v = static_cast<uint32_t>(sum);
  TIER_SCC(sum >> 32);
})
TIER_S2(s_sub_u32, {
  const uint64_t take = static_cast<uint32_t>(b);
  v = static_cast<uint32_t>(static_cast<uint32_t>(a) - take);
  TIER_SCC(static_cast<uint32_t>(a) < take);
})
TIER_S2(s_subb_u32, {
  const uint64_t take = static_cast<uint64_t>(static_cast<uint32_t>(b)) + *R.scc;
  v = static_cast<uint32_t>(static_cast<uint32_t>(a) - take);
  TIER_SCC(static_cast<uint32_t>(a) < take);
})
TIER_S2(s_add_i32, {
  const int64_t sum = static_cast<int32_t>(a) + static_cast<int64_t>(static_cast<int32_t>(b));
  v = static_cast<uint32_t>(sum);
  TIER_SCC(sum != static_cast<int32_t>(sum));
})
TIER_S2(s_sub_i32, {
  const int64_t diff = static_cast<int32_t>(a) - static_cast<int64_t>(static_cast<int32_t>(b));
  v = static_cast<uint32_t>(diff);
  TIER_SCC(diff != static_cast<int32_t>(diff));
})
TIER_S2(s_mul_i32, v = static_cast<uint32_t>(a) * static_cast<uint32_t>(b))
TIER_S2(s_and_b32, { v = static_cast<uint32_t>(a) & static_cast<uint32_t>(b); TIER_SCC(v != 0); })
TIER_S2(s_or_b32, { v = static_cast<uint32_t>(a) | static_cast<uint32_t>(b); TIER_SCC(v != 0); })
TIER_S2(s_xor_b32, { v = static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b); TIER_SCC(v != 0); })
TIER_S2(s_andn2_b32, { v = static_cast<uint32_t>(a & ~b); TIER_SCC(v != 0); })
// (RDNA4's 64-bit scalar arithmetic: SCC untouched.)
TIER_S2(s_add_nc_u64, v = a + b)
TIER_S2(s_sub_nc_u64, v = a - b)
TIER_S2(s_mul_u64, v = a * b)
TIER_S2(s_and_b64, { v = a & b; TIER_SCC(v != 0); })
TIER_S2(s_or_b64, { v = a | b; TIER_SCC(v != 0); })
TIER_S2(s_xor_b64, { v = a ^ b; TIER_SCC(v != 0); })
TIER_S2(s_andn2_b64, { v = a & ~b; TIER_SCC(v != 0); })
TIER_S2(s_lshl_b32, { v = static_cast<uint32_t>(a) << (b & 31); TIER_SCC(static_cast<uint32_t>(v) != 0); })
TIER_S2(s_lshr_b32, {
  v = static_cast<uint32_t>(a) >> (b & 31);
  TIER_SCC(static_cast<uint32_t>(a) >> (b & 31));
})
TIER_S2(s_ashr_i32, { v = static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31)); TIER_SCC(v != 0); })
TIER_S2(s_lshl_b64, { v = a << (b & 63); TIER_SCC(v != 0); })
TIER_S2(s_lshr_b64, { v = a >> (b & 63); TIER_SCC(v != 0); })
TIER_S2(s_cselect, v = *R.scc ? a : b)
TIER_S2(s_not_b32, { v = static_cast<uint32_t>(~a); TIER_SCC(v != 0); })
TIER_S2(s_not_b64, { v = ~a; TIER_SCC(v != 0); })
TIER_S2(s_orn2_b32, { v = static_cast<uint32_t>(a) | ~static_cast<uint32_t>(b); TIER_SCC(v != 0); })
TIER_S2(s_orn2_b64, { v = a | ~b; TIER_SCC(v != 0); })
TIER_S2(s_nand_b32, { v = ~(static_cast<uint32_t>(a) & static_cast<uint32_t>(b)); TIER_SCC(v != 0); })
TIER_S2(s_nor_b32, { v = ~(static_cast<uint32_t>(a) | static_cast<uint32_t>(b)); TIER_SCC(v != 0); })
TIER_S2(s_xnor_b32, { v = ~(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b)); TIER_SCC(v != 0); })
TIER_S2(s_nand_b64, { v = ~(a & b); TIER_SCC(v != 0); })
TIER_S2(s_nor_b64, { v = ~(a | b); TIER_SCC(v != 0); })
TIER_S2(s_xnor_b64, { v = ~(a ^ b); TIER_SCC(v != 0); })
TIER_S2(s_min_u32, {
  const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
  *R.scc = x < y;
  v = *R.scc ? x : y;
})
TIER_S2(s_max_u32, {
  const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
  *R.scc = x > y;
  v = *R.scc ? x : y;
})
// (gfx942 and gfx950: SCC = S0 >= S1, so a tie sets it; the table has these under the names with _ge, chosen by the instruction's architecture.)
TIER_S2(s_max_u32_ge, {
  const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
  *R.scc = x >= y;
  v = *R.scc ? x : y;
})
TIER_S2(s_max_i32_ge, {
  const int32_t x = static_cast<int32_t>(a), y = static_cast<int32_t>(b);
  *R.scc = x >= y;
  v = static_cast<uint32_t>(*R.scc ? x : y);
})
TIER_S2(s_min_i32, {
  const int32_t x = static_cast<int32_t>(a), y = static_cast<int32_t>(b);
  *R.scc = x < y;
  v = static_cast<uint32_t>(*R.scc ? x : y);
})
TIER_S2(s_max_i32, {
  const int32_t x = static_cast<int32_t>(a), y = static_cast<int32_t>(b);
  *R.scc = x > y;
  v = static_cast<uint32_t>(*R.scc ? x : y);
})
TIER_S2(s_bcnt1_b32, { v = static_cast<uint32_t>(__builtin_popcount(static_cast<uint32_t>(a))); TIER_SCC(v != 0); })
TIER_S2(s_bcnt1_b64, { v = static_cast<uint32_t>(__builtin_popcountll(a)); TIER_SCC(v != 0); })
TIER_S2(s_ff1_b32, { const uint32_t x = static_cast<uint32_t>(a); v = x ? static_cast<uint32_t>(__builtin_ctz(x)) : 0xFFFFFFFFu; })
TIER_S2(s_ff1_b64, v = a ? static_cast<uint32_t>(__builtin_ctzll(a)) : 0xFFFFFFFFu)
TIER_S2(s_lshl1_add_u32, { const uint64_t t = (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 1) + static_cast<uint32_t>(b); v = static_cast<uint32_t>(t); TIER_SCC((t >> 32) != 0); })
TIER_S2(s_lshl2_add_u32, { const uint64_t t = (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 2) + static_cast<uint32_t>(b); v = static_cast<uint32_t>(t); TIER_SCC((t >> 32) != 0); })
TIER_S2(s_lshl3_add_u32, { const uint64_t t = (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 3) + static_cast<uint32_t>(b); v = static_cast<uint32_t>(t); TIER_SCC((t >> 32) != 0); })
TIER_S2(s_lshl4_add_u32, { const uint64_t t = (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 4) + static_cast<uint32_t>(b); v = static_cast<uint32_t>(t); TIER_SCC((t >> 32) != 0); })
TIER_S2(s_abs_i32, {
  const int32_t x = static_cast<int32_t>(a);
  v = x < 0 ? 0u - static_cast<uint32_t>(x) : static_cast<uint32_t>(x);
  TIER_SCC(v != 0);
})
TIER_S2(s_bfe_u32, {
  const uint32_t start = static_cast<uint32_t>(b) & 31, width = (static_cast<uint32_t>(b) >> 16) & 0x7F;
  uint64_t t = static_cast<uint32_t>(a) >> start;
  if (width == 0) t = 0;
  else if (width < 32) t &= (uint64_t{1} << width) - 1;
  v = static_cast<uint32_t>(t);
  TIER_SCC(v != 0);
})

TIER_S2(s_mul_hi_u32, v = static_cast<uint32_t>((static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32))
TIER_S2(s_mul_hi_i32, v = static_cast<uint32_t>(static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a)) * static_cast<int32_t>(b)) >> 32))
TIER_S2(s_brev_b32, {
  const uint32_t x = static_cast<uint32_t>(a);
  uint32_t r = 0;
  for (uint32_t k = 0; k < 32; ++k) r |= ((x >> k) & 1) << (31 - k);
  v = r;
})
TIER_S2(s_brev_b64, {
  uint64_t r = 0;
  for (uint32_t k = 0; k < 64; ++k) r |= ((a >> k) & 1) << (63 - k);
  v = r;
})
TIER_S2(s_bcnt0_b32, { v = 32 - __builtin_popcount(static_cast<uint32_t>(a)); TIER_SCC(v != 0); })
TIER_S2(s_bcnt0_b64, { v = 64 - __builtin_popcountll(a); TIER_SCC(v != 0); })
TIER_S2(s_ff0_b32, { const uint64_t clear = ~static_cast<uint32_t>(a) & 0xFFFFFFFFull; v = clear ? static_cast<uint32_t>(__builtin_ctzll(clear)) : 0xFFFFFFFFu; })
TIER_S2(s_ff0_b64, { const uint64_t clear = ~a; v = clear ? static_cast<uint32_t>(__builtin_ctzll(clear)) : 0xFFFFFFFFu; })
TIER_S2(s_sext_i8, v = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(a))))
TIER_S2(s_sext_i16, v = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(a))))
TIER_S2(s_pack_ll, v = (static_cast<uint32_t>(a) & 0xFFFFu) | (static_cast<uint32_t>(b) & 0xFFFFu) << 16)
TIER_S2(s_pack_lh, v = (static_cast<uint32_t>(a) & 0xFFFFu) | (static_cast<uint32_t>(b) & 0xFFFF0000u))
TIER_S2(s_pack_hh, v = static_cast<uint32_t>(a) >> 16 | (static_cast<uint32_t>(b) & 0xFFFF0000u))
TIER_S2(s_pack_hl, v = static_cast<uint32_t>(a) >> 16 | static_cast<uint32_t>(b) << 16)
TIER_S2(s_bfm_b32, v = ((1u << (a & 31)) - 1) << (b & 31))
// (conditional moves: the destination is written only where SCC is set)
int s_cmov(const Op& o, Regs& R) {
  const uint64_t a = rd(R, o.src[0]);
  if (*R.scc) wr(R, o.dst, a);
  return true;
}
int s_getpc(const Op& o, Regs& R) {
  wr(R, o.dst, *R.pc);   // (the address of the instruction after this one: the pc has moved on already)
  return true;
}
// SOPK: the sign-extended 16-bit constant is `off0`. sub 0: s_addk_i32, 1: s_mulk_i32, 2: s_cmovk_i32 (the destination is the source too).
int s_sopk(const Op& o, Regs& R) {
  const int32_t k = o.off0;
  if (o.sub == 0) {
    const int32_t x = static_cast<int32_t>(rd(R, o.src[0]));
    const int32_t sum = static_cast<int32_t>(static_cast<uint32_t>(x) + static_cast<uint32_t>(k));
    wr(R, o.dst, static_cast<uint32_t>(sum));
    *R.scc = ((x >= 0) == (k >= 0)) && ((sum >= 0) != (x >= 0));
  } else if (o.sub == 1) {
    wr(R, o.dst, static_cast<uint32_t>(rd(R, o.src[0]) * static_cast<uint32_t>(k)));
  } else if (*R.scc) {
    wr(R, o.dst, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(k))));
  }
  return true;
}
// s_cmpk_*: the register against the constant, signed (sub 0 to 5: eq, lg, gt, ge, lt, le) or, for sub 6 to 11, unsigned with the constant's low 16 bits.
int s_cmpk(const Op& o, Regs& R) {
  const uint32_t x = static_cast<uint32_t>(rd(R, o.src[0]));
  const bool is_unsigned = o.sub >= 6;
  const uint32_t k = is_unsigned ? static_cast<uint16_t>(o.off0) : static_cast<uint32_t>(o.off0);
  bool r;
  if (is_unsigned) {
    switch (o.sub - 6) {
      case 0: r = x == k; break;
      case 1: r = x != k; break;
      case 2: r = x > k; break;
      case 3: r = x >= k; break;
      case 4: r = x < k; break;
      default: r = x <= k; break;
    }
  } else {
    const int32_t sx = static_cast<int32_t>(x), sk = static_cast<int32_t>(k);
    switch (o.sub) {
      case 0: r = sx == sk; break;
      case 1: r = sx != sk; break;
      case 2: r = sx > sk; break;
      case 3: r = sx >= sk; break;
      case 4: r = sx < sk; break;
      default: r = sx <= sk; break;
    }
  }
  *R.scc = r;
  return true;
}
// s_bitcmp0/1_b32/b64 (SOPC): SCC is the bit, or its complement. sub bit 0: the 64-bit form, bit 1: tests for a set bit.
int s_bitcmp(const Op& o, Regs& R) {
  const uint64_t a = rd(R, o.src[0]), b = rd(R, o.src[1]);
  const uint64_t bit = (a >> (b & (o.sub & 1 ? 63 : 31))) & 1;
  *R.scc = (o.sub & 2) ? bit == 1 : bit == 0;
  return true;
}

// SOPC: only SCC changes. `sub` is the test and the type.
enum ScalarCmp : uint32_t {
  kEqU32, kLgU32, kLtU32, kLeU32, kGtU32, kGeU32, kEqI32, kLgI32, kLtI32, kLeI32, kGtI32, kGeI32, kEqU64, kLgU64
};
// (The comparison fixed at compile time, with the operand shapes of the S2 handlers: `s_cmp_pick`, when the instruction is decoded.)
template <uint32_t Cmp, int KA, int KB>
int s_cmp_k(const Op& o, Regs& R) {
  const uint64_t a = rd_k<KA>(R, o.src[0]), b = rd_k<KB>(R, o.src[1]);
  const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
  const int32_t sx = static_cast<int32_t>(a), sy = static_cast<int32_t>(b);
  bool r = false;
  switch (Cmp) {
    case kEqU32: r = x == y; break;
    case kLgU32: r = x != y; break;
    case kLtU32: r = x < y; break;
    case kLeU32: r = x <= y; break;
    case kGtU32: r = x > y; break;
    case kGeU32: r = x >= y; break;
    case kEqI32: r = sx == sy; break;
    case kLgI32: r = sx != sy; break;
    case kLtI32: r = sx < sy; break;
    case kLeI32: r = sx <= sy; break;
    case kGtI32: r = sx > sy; break;
    case kGeI32: r = sx >= sy; break;
    case kEqU64: r = a == b; break;
    default: r = a != b; break;
  }
  *R.scc = r;
  return true;
}
int s_cmp(const Op& o, Regs& R) {
  switch (o.sub) {
#define TIER_CMP_CASE(c) case c: return s_cmp_k<c, 0, 0>(o, R);
    TIER_CMP_CASE(kEqU32) TIER_CMP_CASE(kLgU32) TIER_CMP_CASE(kLtU32) TIER_CMP_CASE(kLeU32) TIER_CMP_CASE(kGtU32) TIER_CMP_CASE(kGeU32)
    TIER_CMP_CASE(kEqI32) TIER_CMP_CASE(kLgI32) TIER_CMP_CASE(kLtI32) TIER_CMP_CASE(kLeI32) TIER_CMP_CASE(kGtI32) TIER_CMP_CASE(kGeI32)
    TIER_CMP_CASE(kEqU64)
#undef TIER_CMP_CASE
    default: return s_cmp_k<kLgU64, 0, 0>(o, R);
  }
}
// The handler for comparison `sub` over one-word scalar registers or constants (null where the shapes are not those).
Fn s_cmp_pick(uint32_t sub, const Src& a, const Src& b) {
  const auto reg = [](const Src& x) { return x.kind == kSgpr && !x.wide; };
  const auto con = [](const Src& x) { return x.kind == kConst; };
  if (sub > kGeI32) return nullptr;   // (the 64-bit ones read pairs)
  static const Fn ss[12] = {s_cmp_k<kEqU32, 1, 1>, s_cmp_k<kLgU32, 1, 1>, s_cmp_k<kLtU32, 1, 1>, s_cmp_k<kLeU32, 1, 1>, s_cmp_k<kGtU32, 1, 1>, s_cmp_k<kGeU32, 1, 1>,
                            s_cmp_k<kEqI32, 1, 1>, s_cmp_k<kLgI32, 1, 1>, s_cmp_k<kLtI32, 1, 1>, s_cmp_k<kLeI32, 1, 1>, s_cmp_k<kGtI32, 1, 1>, s_cmp_k<kGeI32, 1, 1>};
  static const Fn sc[12] = {s_cmp_k<kEqU32, 1, 2>, s_cmp_k<kLgU32, 1, 2>, s_cmp_k<kLtU32, 1, 2>, s_cmp_k<kLeU32, 1, 2>, s_cmp_k<kGtU32, 1, 2>, s_cmp_k<kGeU32, 1, 2>,
                            s_cmp_k<kEqI32, 1, 2>, s_cmp_k<kLgI32, 1, 2>, s_cmp_k<kLtI32, 1, 2>, s_cmp_k<kLeI32, 1, 2>, s_cmp_k<kGtI32, 1, 2>, s_cmp_k<kGeI32, 1, 2>};
  if (reg(a) && reg(b)) return ss[sub];
  if (reg(a) && con(b)) return sc[sub];
  return nullptr;
}

// Divergence: keep EXEC, narrow it. sub 0: and, 1: or, 2: andn2.
int s_saveexec(const Op& o, Regs& R) {
  const uint64_t a = rd(R, o.src[0]);
  const uint64_t saved = *R.exec;
  *R.exec = o.sub == 0 ? a & saved : o.sub == 1 ? a | saved : a & ~saved;
  wr(R, o.dst, saved);
  *R.scc = *R.exec != 0;
  return true;
}

// RDNA's wave32 forms: EXEC is its low half (exec_lo). sub: 0 and, 1 or, 2 xor, 3 and_not1 (x & ~exec), 4 or_not1, 5 and_not0 (~x & exec),
// 6 or_not0.
int s_saveexec32(const Op& o, Regs& R) {
  const uint32_t saved = static_cast<uint32_t>(*R.exec), x = static_cast<uint32_t>(rd(R, o.src[0]));
  const uint32_t v = o.sub == 0   ? x & saved
                     : o.sub == 1 ? x | saved
                     : o.sub == 2 ? x ^ saved
                     : o.sub == 3 ? x & ~saved
                     : o.sub == 4 ? x | ~saved
                     : o.sub == 5 ? ~x & saved
                                  : ~x | saved;
  *R.exec = (*R.exec & ~0xFFFFFFFFull) | v;
  wr(R, o.dst, saved);
  *R.scc = v != 0;
  return 1;
}

// ---- Program flow (SOPP)
// s_waitcnt (and RDNA4's s_wait_loadcnt): the loads still outstanding must come down to `sub`; a wave with others to run gives them
// a turn first (waited_for_loads).
int b_waitcnt(const Op& o, Regs& R) {
  if (*R.loads <= o.sub) return 1;
  *R.loads = o.sub;
  if (R.nwaves < 2) return 1;
  *R.soft_yield = true;
  return 2;
}
int b_nop(const Op&, Regs&) { return true; }
int b_branch(const Op&, Regs& R) { *R.pc = R.target; return true; }
int b_scc0(const Op&, Regs& R) { if (!*R.scc) *R.pc = R.target; return true; }
int b_scc1(const Op&, Regs& R) { if (*R.scc) *R.pc = R.target; return true; }
int b_execz(const Op&, Regs& R) { if (!*R.exec) *R.pc = R.target; return true; }
int b_execnz(const Op&, Regs& R) { if (*R.exec) *R.pc = R.target; return true; }
int b_vccz(const Op&, Regs& R) { if (!(R.lanes == 32 ? static_cast<uint32_t>(*R.vcc) : *R.vcc)) *R.pc = R.target; return true; }
int b_vccnz(const Op&, Regs& R) { if (R.lanes == 32 ? static_cast<uint32_t>(*R.vcc) : *R.vcc) *R.pc = R.target; return true; }

// ---- Classification

enum class Cat { None, Carry, Lane, Global, Ds, Pk, Pk16, L64N, Permlane, Int, F32, F64, L64, Mad64, Cmp, Cndmask, Readfirstlane, S2, ScalarCmp, Saveexec, Branch, Sopk, Cmpk, Bitcmp, Endpgm, PkF16, PkMov, Div, FmaMix, Acc };

struct Entry {
  Cat cat = Cat::None;
  int k = 0;            // the kernel's id in its list, or the handler's choice
  int arity = 0;
  uint32_t flags = 0;   // see below
  Fn fn = nullptr;
  uint32_t sub = 0;
};
// Entry flags
constexpr uint32_t kNoClamp = 1;     // the instruction is left to the interpreter if its clamp bit is set (int add/sub saturate)
constexpr uint32_t kFmac = 2;        // the addend is the destination
constexpr uint32_t kConstMid = 4;    // fmamk: source 1 is the instruction's own constant, and omod/clamp are ignored
constexpr uint32_t kConstLast = 8;   // fmaak: source 2 is the instruction's own constant
constexpr uint32_t kFloatSrc = 32;   // an integer result from a float source: the source's modifiers apply
constexpr uint32_t kIgnoreMods = 16;
constexpr uint32_t kTrigTurns = 128;  // v_sin_f32, v_cos_f32: the kernel after the listed one on the architectures whose guides count the angle in whole turns (gfx942, gfx950)
constexpr uint32_t kIntSrc1 = 256;    // v_ldexp_f32: the second source is an integer (its bits, no float modifiers)
constexpr uint32_t kCdnaOnly = 512;   // 16-bit forms: only the CDNA parts (gfx90a, gfx942, gfx950), whose 16-bit results fill the low half of the register
constexpr uint32_t kNoOmod = 1024;    // the instruction's output modifier means something the kernel does not do
constexpr uint32_t kFloatOut = 4096;  // an integer kernel with a float result: the instruction's output modifier and clamp mean something the kernel does not do
constexpr uint32_t kRawSrc = 2048;    // the source is not a float of 32 bits (a half): float modifiers on it would flip the wrong bit
constexpr uint32_t kMinMax = 64;     // (v_min_f32, v_max_f32: opposite zeros go to the interpreter) // the interpreter ignores omod and clamp on this one

const std::unordered_map<std::string_view, Entry>& table() {
  static const std::unordered_map<std::string_view, Entry> t = [] {
    std::unordered_map<std::string_view, Entry> m;
    const auto I = [&](std::string_view n, int k, int arity, uint32_t fl = 0) { m[n] = {Cat::Int, k, arity, fl}; };
    I("v_mov_b32_e32", MOV, 1);
    I("v_not_b32_e32", NOT, 1);
    I("v_add_u32_e32", ADD, 2, kNoClamp);
    I("v_add_u32_e64", ADD, 2, kNoClamp);
    I("v_sub_u32_e32", SUB, 2, kNoClamp);
    I("v_sub_u32_e64", SUB, 2, kNoClamp);
    I("v_subrev_u32_e32", SUBREV, 2);
    I("v_mul_lo_u32", MUL_LO, 2);
    I("v_mul_hi_u32", MUL_HI_U, 2);
    I("v_mul_hi_i32", MUL_HI_I, 2);
    I("v_mul_u32_u24_e32", MUL_U24, 2);
    I("v_mul_i32_i24_e32", MUL_I24, 2);
    I("v_mad_u32_u24", MAD_U24, 3);
    I("v_mad_i32_i24", MAD_I24, 3);
    I("v_and_b32_e32", AND, 2);
    I("v_or_b32_e32", OR, 2);
    I("v_or_b32_e64", OR, 2);
    I("v_xor_b32_e32", XOR, 2);
    I("v_lshlrev_b32_e32", LSHLREV, 2);
    I("v_lshrrev_b32_e32", LSHRREV, 2);
    I("v_ashrrev_i32_e32", ASHRREV, 2);
    I("v_min_u32_e32", MIN_U, 2);
    I("v_max_u32_e32", MAX_U, 2);
    I("v_min_i32_e32", MIN_I, 2);
    I("v_max_i32_e32", MAX_I, 2);
    I("v_bfi_b32", BFI, 3);
    I("v_alignbit_b32", ALIGNBIT, 3);
    I("v_alignbyte_b32", ALIGNBYTE, 3);
    I("v_add3_u32", ADD3, 3);
    I("v_or3_b32", OR3, 3);
    I("v_xor3_b32", XOR3, 3);
    I("v_and_or_b32", AND_OR, 3);
    I("v_xad_u32", XAD, 3);
    I("v_lshl_or_b32", LSHL_OR, 3);
    I("v_lshl_add_u32", LSHL_ADD, 3);
    I("v_add_lshl_u32", ADD_LSHL, 3);
    I("v_bcnt_u32_b32", BCNT, 2);
    I("v_ffbh_u32_e32", FFBH_U, 1);
    I("v_ffbl_b32_e32", FFBL, 1);
    I("v_ffbh_i32_e32", FFBH_I, 1);
    I("v_bfe_u32", BFE_U, 3);
    I("v_med3_u32", MED3_U, 3);
    I("v_med3_i32", MED3_I, 3);
    I("v_min3_u32", MIN3_U, 3);
    I("v_max3_u32", MAX3_U, 3);
    I("v_mbcnt_lo_u32_b32", MBCNT_LO, 2);
    I("v_mbcnt_hi_u32_b32", MBCNT_HI, 2);
    I("v_cvt_i32_f32_e32", CVT_I32_F32, 1, kFloatSrc);
    I("v_cvt_u32_f32_e32", CVT_U32_F32, 1, kFloatSrc);
    I("v_cvt_f32_u32_e32", CVT_F32_U, 1, kFloatOut);
    I("v_cvt_f32_i32_e32", CVT_F32_I, 1, kFloatOut);
    I("v_cvt_f32_ubyte0_e32", CVT_F32_UBYTE0, 1, kFloatOut);
    I("v_add_u16_e32", ADD_U16, 2, kNoClamp | kCdnaOnly);
    I("v_sub_u16_e32", SUB_U16, 2, kNoClamp | kCdnaOnly);
    I("v_subrev_u16_e32", SUBREV_U16, 2, kNoClamp | kCdnaOnly);
    I("v_lshlrev_b16_e32", LSHLREV_B16, 2, kCdnaOnly);
    I("v_lshrrev_b16_e32", LSHRREV_B16, 2, kCdnaOnly);
    I("v_ashrrev_i16_e32", ASHRREV_I16, 2, kCdnaOnly);
    I("v_mul_lo_u16_e32", MUL_LO_U16, 2, kCdnaOnly);
    I("v_min_u16_e32", MIN_U16, 2, kCdnaOnly);
    I("v_max_u16_e32", MAX_U16, 2, kCdnaOnly);
    I("v_min_i16_e32", MIN_I16, 2, kCdnaOnly);
    I("v_max_i16_e32", MAX_I16, 2, kCdnaOnly);
    I("v_pack_b32_f16", PACK_B32_F16, 2, kCdnaOnly);
    I("v_bfe_i32", BFE_I, 3);
    I("v_perm_b32", PERM, 3);
    I("v_cvt_f16_f32_e32", CVT_F16_F32, 1, kFloatSrc | kNoClamp | kNoOmod | kCdnaOnly);
    // The same operations in their VOP3 form (the sources may be any of the operands, the modifiers and the constants the short form's are).
    I("v_mov_b32_e64", MOV, 1);
    I("v_not_b32_e64", NOT, 1);
    I("v_subrev_u32_e64", SUBREV, 2);
    I("v_mul_u32_u24_e64", MUL_U24, 2);
    I("v_mul_i32_i24_e64", MUL_I24, 2);
    I("v_and_b32_e64", AND, 2);
    I("v_xor_b32_e64", XOR, 2);
    I("v_lshlrev_b32_e64", LSHLREV, 2);
    I("v_lshrrev_b32_e64", LSHRREV, 2);
    I("v_ashrrev_i32_e64", ASHRREV, 2);
    I("v_min_u32_e64", MIN_U, 2);
    I("v_max_u32_e64", MAX_U, 2);
    I("v_min_i32_e64", MIN_I, 2);
    I("v_max_i32_e64", MAX_I, 2);
    I("v_cvt_i32_f32_e64", CVT_I32_F32, 1, kFloatSrc);
    I("v_cvt_u32_f32_e64", CVT_U32_F32, 1, kFloatSrc);
    I("v_cvt_f32_u32_e64", CVT_F32_U, 1, kFloatOut);
    I("v_cvt_f32_i32_e64", CVT_F32_I, 1, kFloatOut);
    I("v_cvt_f32_ubyte0_e64", CVT_F32_UBYTE0, 1, kFloatOut);
    const auto F = [&](std::string_view n, int k, int arity, uint32_t fl = 0) { m[n] = {Cat::F32, k, arity, fl}; };
    F("v_subrev_f32_e64", FSUBREV, 2);
    F("v_mul_f32_e64", FMUL, 2);
    F("v_min_f32_e64", FMIN, 2, kMinMax);
    F("v_rcp_f32_e64", FRCP, 1);
    F("v_rcp_iflag_f32_e32", FRCP, 1);
    F("v_sqrt_f32_e32", FSQRT, 1);
    F("v_rsq_f32_e32", FRSQ, 1);
    F("v_exp_f32_e32", FEXP, 1);
    F("v_log_f32_e32", FLOG, 1);
    F("v_floor_f32_e32", FFLOOR, 1);
    F("v_ceil_f32_e32", FCEIL, 1);
    F("v_trunc_f32_e32", FTRUNC, 1);
    F("v_rndne_f32_e32", FRNDNE, 1);
    F("v_fract_f32_e32", FFRACT, 1);
    F("v_sin_f32_e32", FSIN_R, 1, kTrigTurns);
    F("v_cos_f32_e32", FCOS_R, 1, kTrigTurns);
    F("v_ldexp_f32", FLDEXP, 2, kIntSrc1);
    F("v_cvt_f32_f16_e32", FCVT_F32_F16, 1, kRawSrc | kCdnaOnly);
    F("v_rcp_iflag_f32_e64", FRCP, 1);
    m["v_cndmask_b32_e64"] = {Cat::Cndmask, 0, 3};
    F("v_add_f32_e32", FADD, 2);
    F("v_add_f32_e64", FADD, 2);
    F("v_sub_f32_e32", FSUB, 2);
    F("v_sub_f32_e64", FSUB, 2);
    F("v_subrev_f32_e32", FSUBREV, 2);
    F("v_mul_f32_e32", FMUL, 2);
    F("v_fma_f32", FMA, 3);
    F("v_min_f32_e32", FMIN, 2, kMinMax);
    F("v_max_f32_e32", FMAX, 2, kMinMax);
    F("v_max_f32_e64", FMAX, 2, kMinMax);
    F("v_rcp_f32_e32", FRCP, 1);
    F("v_fmac_f32_e32", FMA, 2, kFmac);
    F("v_fmac_f32_e64", FMA, 2, kFmac);
    F("v_fmaak_f32", FMA, 3, kConstLast);
    F("v_fmamk_f32", FMA, 3, kConstMid | kIgnoreMods);
    const auto D = [&](std::string_view n, int k, int arity, uint32_t fl = 0) { m[n] = {Cat::F64, k, arity, fl}; };
    D("v_add_f64", DADD, 2);
    D("v_mul_f64", DMUL, 2);
    D("v_fma_f64", DFMA, 3);
    D("v_fmac_f64_e32", DFMA, 2, kFmac);
    D("v_rcp_f64_e32", DRCP, 1);
    D("v_sqrt_f64_e32", DSQRT, 1);
    D("v_floor_f64_e32", DFLOOR, 1);
    D("v_min_f64", DMIN, 2);
    D("v_max_f64", DMAX, 2);
    D("v_ceil_f64_e32", DCEIL, 1);
    D("v_trunc_f64_e32", DTRUNC, 1);
    D("v_rndne_f64_e32", DRNDNE, 1);
    const auto L = [&](std::string_view n, int k, int arity, uint32_t sub, uint32_t fl = 0) { m[n] = {Cat::L64, k, arity, fl, nullptr, sub}; };
    const auto LN = [&](std::string_view n, int k, int arity, uint32_t sub, uint32_t fl = 0) { m[n] = {Cat::L64N, k, arity, fl, nullptr, sub}; };
    L("v_mov_b64_e32", L_MOV, 1, 1);
    L("v_lshlrev_b64", L_LSHLREV, 2, 2);
    L("v_lshrrev_b64", L_LSHRREV, 2, 2);
    L("v_ashrrev_i64", L_ASHRREV, 2, 2);
    L("v_lshl_add_u64", L_LSHL_ADD, 3, 1 | 4);
    L("v_lshl_add_u64_g", L_LSHL_ADD_G, 3, 1 | 4);
    L("v_cvt_f64_u32_e32", L_CVT_F64_U, 1, 0);
    L("v_cvt_f64_i32_e32", L_CVT_F64_I, 1, 0);
    // (float operands and results: a source modifier, the output modifier or the clamp leaves the instruction to the interpreter)
    L("v_ldexp_f64", L_LDEXP, 2, 1, kFloatSrc | kFloatOut);
    L("v_frexp_mant_f64_e32", L_FREXP_MANT, 1, 1, kFloatSrc | kFloatOut);
    L("v_cvt_f64_f32_e32", L_CVT_F64_F32, 1, 0, kFloatSrc | kFloatOut);
    LN("v_frexp_exp_i32_f64_e32", L_FREXP_EXP, 1, 1, kFloatSrc);
    LN("v_cvt_f32_f64_e32", L_CVT_F32_F64, 1, 1, kFloatSrc | kFloatOut);
    LN("v_cvt_i32_f64_e32", L_CVT_I32_F64, 1, 1, kFloatSrc);
    LN("v_cvt_u32_f64_e32", L_CVT_U32_F64, 1, 1, kFloatSrc);
    m["v_mad_u64_u32"] = {Cat::Mad64, 0, 3};
    m["global_load_dword"] = {Cat::Global, 1, 1, 0, nullptr, 1};
    m["global_load_dwordx2"] = {Cat::Global, 1, 1, 0, nullptr, 2};
    m["global_load_dwordx3"] = {Cat::Global, 1, 1, 0, nullptr, 3};
    m["global_load_dwordx4"] = {Cat::Global, 1, 1, 0, nullptr, 4};
    m["global_load_ubyte"] = {Cat::Global, 4, 1, 0, nullptr, 1};
    m["global_load_sbyte"] = {Cat::Global, 4, 1, 1, nullptr, 1};
    m["global_load_ushort"] = {Cat::Global, 4, 1, 0, nullptr, 2};
    m["global_load_sshort"] = {Cat::Global, 4, 1, 1, nullptr, 2};
    m["global_store_byte"] = {Cat::Global, 5, 1, 0, nullptr, 1};
    m["global_store_short"] = {Cat::Global, 5, 1, 0, nullptr, 2};
    m["global_store_dword"] = {Cat::Global, 2, 1, 0, nullptr, 1};
    m["global_store_dwordx2"] = {Cat::Global, 2, 1, 0, nullptr, 2};
    m["global_store_dwordx3"] = {Cat::Global, 2, 1, 0, nullptr, 3};
    m["global_store_dwordx4"] = {Cat::Global, 2, 1, 0, nullptr, 4};
    m["ds_read_b32"] = {Cat::Ds, kDsRead, 1, 0, nullptr, 1};
    m["ds_read_b64"] = {Cat::Ds, kDsRead, 1, 0, nullptr, 2};
    m["ds_read_b96"] = {Cat::Ds, kDsRead, 1, 0, nullptr, 3};
    m["ds_read_b128"] = {Cat::Ds, kDsRead, 1, 0, nullptr, 4};
    m["ds_write_b32"] = {Cat::Ds, kDsWrite, 1, 0, nullptr, 1};
    m["ds_write_b64"] = {Cat::Ds, kDsWrite, 1, 0, nullptr, 2};
    m["ds_write_b96"] = {Cat::Ds, kDsWrite, 1, 0, nullptr, 3};
    m["ds_write_b128"] = {Cat::Ds, kDsWrite, 1, 0, nullptr, 4};
    m["ds_bpermute_b32"] = {Cat::Ds, kDsBperm, 1, 0, nullptr, 1};
    m["ds_permute_b32"] = {Cat::Ds, kDsPerm, 1, 0, nullptr, 1};
    m["ds_swizzle_b32"] = {Cat::Ds, kDsSwizzle, 1, 0, nullptr, 1};
    m["ds_add_u32"] = {Cat::Ds, kDsAdd, 1, 0, nullptr, 1};
    m["ds_read2st64_b32"] = {Cat::Ds, kDsRead2, 1, 1, nullptr, 2};
    m["ds_write2st64_b32"] = {Cat::Ds, kDsWrite2, 1, 1, nullptr, 2};
    m["ds_read2_b32"] = {Cat::Ds, kDsRead2, 1, 0, nullptr, 2};
    m["ds_write2_b32"] = {Cat::Ds, kDsWrite2, 1, 0, nullptr, 2};
    m["v_pk_add_f32"] = {Cat::Pk, FADD, 2};
    m["v_permlane16_b32"] = {Cat::Permlane, 0, 3};
    m["v_permlanex16_b32"] = {Cat::Permlane, 1, 3};
    // (the packed 16-bit integers: Entry::k is the kernel of the plain form, Entry::sub the clamped one, or 0 where a clamp changes nothing)
    m["v_pk_mul_lo_u16"] = {Cat::Pk16, PK_MULLO, 2};
    m["v_pk_add_u16"] = {Cat::Pk16, PK_ADDU, 2, 0, nullptr, PK_ADDU_C + 1};
    m["v_pk_add_i16"] = {Cat::Pk16, PK_ADDI, 2, 0, nullptr, PK_ADDI_C + 1};
    m["v_pk_sub_u16"] = {Cat::Pk16, PK_SUBU, 2, 0, nullptr, PK_SUBU_C + 1};
    m["v_pk_sub_i16"] = {Cat::Pk16, PK_SUBI, 2, 0, nullptr, PK_SUBI_C + 1};
    m["v_pk_lshlrev_b16"] = {Cat::Pk16, PK_SHL, 2};
    m["v_pk_lshrrev_b16"] = {Cat::Pk16, PK_SHR, 2};
    m["v_pk_ashrrev_i16"] = {Cat::Pk16, PK_ASHR, 2};
    m["v_pk_max_i16"] = {Cat::Pk16, PK_MAXI, 2};
    m["v_pk_min_i16"] = {Cat::Pk16, PK_MINI, 2};
    m["v_pk_max_u16"] = {Cat::Pk16, PK_MAXU, 2};
    m["v_pk_min_u16"] = {Cat::Pk16, PK_MINU, 2};
    m["v_pk_mad_u16"] = {Cat::Pk16, PK_MADU, 3, 0, nullptr, PK_MADU_C + 1};
    m["v_pk_mad_i16"] = {Cat::Pk16, PK_MADI, 3, 0, nullptr, PK_MADI_C + 1};
    m["v_pk_mul_f32"] = {Cat::Pk, FMUL, 2};
    m["v_pk_fma_f32"] = {Cat::Pk, FMA, 3};
    m["v_cndmask_b32_e32"] = {Cat::Cndmask, 0, 3};
    m["v_readlane_b32"] = {Cat::Lane, 0, 2};
    m["v_writelane_b32"] = {Cat::Lane, 1, 2};
    m["s_movk_i32"] = {Cat::Lane, 2, 0};
    m["v_readfirstlane_b32"] = {Cat::Readfirstlane, 0, 1};
    const auto S = [&](std::string_view n, Fn fn, int arity) { m[n] = {Cat::S2, 0, arity, 0, fn}; };
    S("s_mov_b32", s_mov, 1);
    S("s_mov_b64", s_mov, 1);
    S("s_add_u32", s_add_u32, 2);
    S("s_addc_u32", s_addc_u32, 2);
    S("s_sub_u32", s_sub_u32, 2);
    S("s_subb_u32", s_subb_u32, 2);
    S("s_add_i32", s_add_i32, 2);
    S("s_sub_i32", s_sub_i32, 2);
    S("s_mul_i32", s_mul_i32, 2);
    S("s_and_b32", s_and_b32, 2);
    S("s_or_b32", s_or_b32, 2);
    S("s_xor_b32", s_xor_b32, 2);
    S("s_andn2_b32", s_andn2_b32, 2);
    S("s_add_nc_u64", s_add_nc_u64, 2);
    S("s_sub_nc_u64", s_sub_nc_u64, 2);
    S("s_mul_u64", s_mul_u64, 2);
    S("s_and_b64", s_and_b64, 2);
    S("s_or_b64", s_or_b64, 2);
    S("s_xor_b64", s_xor_b64, 2);
    S("s_andn2_b64", s_andn2_b64, 2);
    S("s_lshl_b32", s_lshl_b32, 2);
    S("s_lshr_b32", s_lshr_b32, 2);
    S("s_ashr_i32", s_ashr_i32, 2);
    S("s_lshl_b64", s_lshl_b64, 2);
    S("s_lshr_b64", s_lshr_b64, 2);
    S("s_cselect_b32", s_cselect, 2);
    S("s_cselect_b64", s_cselect, 2);
    S("s_not_b32", s_not_b32, 1);
    S("s_not_b64", s_not_b64, 1);
    S("s_orn2_b32", s_orn2_b32, 2);
    S("s_orn2_b64", s_orn2_b64, 2);
    S("s_nand_b32", s_nand_b32, 2);
    S("s_nor_b32", s_nor_b32, 2);
    S("s_xnor_b32", s_xnor_b32, 2);
    S("s_nand_b64", s_nand_b64, 2);
    S("s_nor_b64", s_nor_b64, 2);
    S("s_xnor_b64", s_xnor_b64, 2);
    S("s_min_u32", s_min_u32, 2);
    S("s_max_u32", s_max_u32, 2);
    S("s_min_i32", s_min_i32, 2);
    S("s_max_i32", s_max_i32, 2);
    S("s_max_u32_ge", s_max_u32_ge, 2);
    S("s_max_i32_ge", s_max_i32_ge, 2);
    S("s_bcnt1_i32_b32", s_bcnt1_b32, 1);
    S("s_bcnt1_i32_b64", s_bcnt1_b64, 1);
    S("s_ff1_i32_b32", s_ff1_b32, 1);
    S("s_ff1_i32_b64", s_ff1_b64, 1);
    S("s_lshl1_add_u32", s_lshl1_add_u32, 2);
    S("s_lshl2_add_u32", s_lshl2_add_u32, 2);
    S("s_lshl3_add_u32", s_lshl3_add_u32, 2);
    S("s_lshl4_add_u32", s_lshl4_add_u32, 2);
    S("s_abs_i32", s_abs_i32, 1);
    S("s_bfe_u32", s_bfe_u32, 2);
    S("s_mul_hi_u32", s_mul_hi_u32, 2);
    S("s_mul_hi_i32", s_mul_hi_i32, 2);
    S("s_brev_b32", s_brev_b32, 1);
    S("s_brev_b64", s_brev_b64, 1);
    S("s_bcnt0_i32_b32", s_bcnt0_b32, 1);
    S("s_bcnt0_i32_b64", s_bcnt0_b64, 1);
    S("s_ff0_i32_b32", s_ff0_b32, 1);
    S("s_ff0_i32_b64", s_ff0_b64, 1);
    S("s_sext_i32_i8", s_sext_i8, 1);
    S("s_sext_i32_i16", s_sext_i16, 1);
    S("s_pack_ll_b32_b16", s_pack_ll, 2);
    S("s_pack_lh_b32_b16", s_pack_lh, 2);
    S("s_pack_hh_b32_b16", s_pack_hh, 2);
    S("s_pack_hl_b32_b16", s_pack_hl, 2);
    S("s_bfm_b32", s_bfm_b32, 2);
    S("s_cmov_b32", s_cmov, 1);
    S("s_cmov_b64", s_cmov, 1);
    S("s_getpc_b64", s_getpc, 0);
    m["s_addk_i32"] = {Cat::Sopk, 0, 0};
    m["s_mulk_i32"] = {Cat::Sopk, 1, 0};
    m["s_cmovk_i32"] = {Cat::Sopk, 2, 0};
    for (const char* t : {"s_cmpk_eq_i32", "s_cmpk_lg_i32", "s_cmpk_gt_i32", "s_cmpk_ge_i32", "s_cmpk_lt_i32", "s_cmpk_le_i32",
                          "s_cmpk_eq_u32", "s_cmpk_lg_u32", "s_cmpk_gt_u32", "s_cmpk_ge_u32", "s_cmpk_lt_u32", "s_cmpk_le_u32"})
      m[t] = {Cat::Cmpk, 0, 0};
    m["s_bitcmp0_b32"] = {Cat::Bitcmp, 0, 2};
    m["s_bitcmp1_b32"] = {Cat::Bitcmp, 2, 2};
    m["s_bitcmp0_b64"] = {Cat::Bitcmp, 1, 2};
    m["s_bitcmp1_b64"] = {Cat::Bitcmp, 3, 2};
    m["s_endpgm"] = {Cat::Endpgm, 0, 0};
    m["v_pk_fma_f16"] = {Cat::PkF16, 0, 3};
    m["v_pk_add_f16"] = {Cat::PkF16, 1, 2};
    m["v_pk_mul_f16"] = {Cat::PkF16, 2, 2};
    m["v_pk_min_f16"] = {Cat::PkF16, 3, 2};
    m["v_pk_max_f16"] = {Cat::PkF16, 4, 2};
    m["v_pk_mov_b32"] = {Cat::PkMov, 0, 2};
    m["v_div_scale_f32"] = {Cat::Div, 0, 3};
    m["v_div_fmas_f32"] = {Cat::Div, 1, 3};
    m["v_div_fixup_f32"] = {Cat::Div, 2, 3};
    m["v_div_scale_f64"] = {Cat::Div, 4, 3};
    m["v_div_fmas_f64"] = {Cat::Div, 5, 3};
    m["v_div_fixup_f64"] = {Cat::Div, 6, 3};
    m["v_accvgpr_read_b32"] = {Cat::Acc, 0, 1};
    m["v_accvgpr_write_b32"] = {Cat::Acc, 0, 1};
    m["v_fma_mix_f32"] = {Cat::FmaMix, 0, 3};
    m["v_fma_mixlo_f16"] = {Cat::FmaMix, 1, 3};
    m["v_fma_mixhi_f16"] = {Cat::FmaMix, 2, 3};
    const auto C = [&](std::string_view n, ScalarCmp k) { m[n] = {Cat::ScalarCmp, static_cast<int>(k), 2}; };
    C("s_cmp_eq_u32", kEqU32);
    C("s_cmp_lg_u32", kLgU32);
    C("s_cmp_lt_u32", kLtU32);
    C("s_cmp_le_u32", kLeU32);
    C("s_cmp_gt_u32", kGtU32);
    C("s_cmp_ge_u32", kGeU32);
    C("s_cmp_eq_i32", kEqI32);
    C("s_cmp_lg_i32", kLgI32);
    C("s_cmp_lt_i32", kLtI32);
    C("s_cmp_le_i32", kLeI32);
    C("s_cmp_gt_i32", kGtI32);
    C("s_cmp_ge_i32", kGeI32);
    C("s_cmp_eq_u64", kEqU64);
    C("s_cmp_lg_u64", kLgU64);
    m["s_and_saveexec_b64"] = {Cat::Saveexec, 0, 1};
    m["s_or_saveexec_b64"] = {Cat::Saveexec, 1, 1};
    m["s_andn2_saveexec_b64"] = {Cat::Saveexec, 2, 1};
    m["s_and_saveexec_b32"] = {Cat::Saveexec, 10, 1};
    m["s_or_saveexec_b32"] = {Cat::Saveexec, 11, 1};
    m["s_xor_saveexec_b32"] = {Cat::Saveexec, 12, 1};
    m["s_and_not1_saveexec_b32"] = {Cat::Saveexec, 13, 1};
    m["s_or_not1_saveexec_b32"] = {Cat::Saveexec, 14, 1};
    m["s_and_not0_saveexec_b32"] = {Cat::Saveexec, 15, 1};
    m["s_or_not0_saveexec_b32"] = {Cat::Saveexec, 16, 1};
    const auto B = [&](std::string_view n, Fn fn) { m[n] = {Cat::Branch, 0, 0, 0, fn}; };
    B("s_branch", b_branch);
    B("s_cbranch_scc0", b_scc0);
    B("s_cbranch_scc1", b_scc1);
    B("s_cbranch_execz", b_execz);
    B("s_cbranch_execnz", b_execnz);
    B("s_cbranch_vccz", b_vccz);
    B("s_cbranch_vccnz", b_vccnz);
    B("s_nop", b_nop);
    B("s_barrier", h_never);
    B("s_clause", b_nop);
    B("s_delay_alu", b_nop);
    B("s_waitcnt_depctr", b_nop);
    B("s_set_inst_prefetch_distance", b_nop);
    B("s_setprio", b_nop);
    B("s_setprio_inc_wg", b_nop);
    B("s_wakeup", b_nop);
    B("s_icache_inv", b_nop);
    B("s_waitcnt", b_waitcnt);
    B("s_wait_loadcnt", b_waitcnt);
    return m;
  }();
  return t;
}

// The Src of a vector instruction's 32-bit lane source, or false for one the tier does not model.
bool lane_src32(const Operand& o, Src& s, bool float_mods) {
  if (o.sel != 6 || o.sext || o.half || o.bits16 || o.hi) return false;
  s = Src();
  switch (o.kind) {
    case OperandKind::Vgpr:
      if (o.index >= kVgprs) return false;
      s.kind = kVgpr, s.index = o.index;
      break;
    case OperandKind::Inline:
    case OperandKind::Literal:
      s.kind = kConst, s.value = static_cast<uint32_t>(o.value);
      break;
    case OperandKind::InlineFloat: {
      s.kind = kConst;
      if (o.width >= 2) {
        const double d = o.fvalue;
        uint64_t b;
        std::memcpy(&b, &d, 8);
        s.value = static_cast<uint32_t>(b);
      } else {
        const float f = static_cast<float>(o.fvalue);
        std::memcpy(&s.value, &f, 4);
      }
      break;
    }
    case OperandKind::Sgpr:
      if (o.index + (o.width >= 2 ? 2u : 1u) > kSgprs) return false;
      s.kind = kSgpr, s.index = o.index;
      break;
    case OperandKind::Vcc: s.kind = kVcc, s.wide = o.width >= 2; break;
    case OperandKind::VccHi: s.kind = kVccHi; break;
    case OperandKind::Exec: s.kind = kExec; break;
    case OperandKind::ExecLo: s.kind = kExecLo; break;
    case OperandKind::ExecHi: s.kind = kExecHi; break;
    case OperandKind::Null: s.kind = kNull; break;
    default: return false;
  }
  if (float_mods) s.neg = o.neg, s.abs = o.abs;
  return true;
}
// A source a 16-bit float instruction reads (lane_half, lane_half_f): an inline float constant is a half, in the low 16 bits.
// Its modifiers (negation, absolute value) act on the half's sign, bit 15.
bool lane_src32_half(const Operand& o, Src& s) {
  if (o.kind != OperandKind::InlineFloat) {
    if (!lane_src32(o, s, false)) return false;
  } else {
    if (o.sel != 6 || o.sext || o.half || o.bits16 || o.hi) return false;
    s = Src();
    s.kind = kConst;
    const _Float16 h = static_cast<_Float16>(o.fvalue);
    uint16_t bits;
    std::memcpy(&bits, &h, 2);
    s.value = bits;
  }
  s.h16 = true, s.neg = o.neg, s.abs = o.abs;
  return true;
}
// The same for a 64-bit consumer (lane_src64).
bool lane_src64(const Operand& o, Src& s, bool float_mods) {
  if (o.sel != 6 || o.sext || o.half || o.bits16 || o.hi) return false;
  s = Src();
  switch (o.kind) {
    case OperandKind::Vgpr:
      if (o.index + 1 >= kVgprs) return false;
      s.kind = kVgpr, s.index = o.index, s.wide = true;
      break;
    case OperandKind::Inline:
    case OperandKind::Literal:
      s.kind = kConst;
      s.value64 = o.kind == OperandKind::Literal && o.literal_high ? static_cast<uint64_t>(o.value) << 32 : static_cast<uint64_t>(o.value);
      break;
    case OperandKind::InlineFloat: {
      s.kind = kConst;
      const double d = o.fvalue;
      std::memcpy(&s.value64, &d, 8);
      break;
    }
    case OperandKind::Sgpr:
      if (o.index + (o.width >= 2 ? 2u : 1u) > kSgprs) return false;
      s.kind = kSgpr, s.index = o.index, s.wide = o.width >= 2;
      break;
    default: return false;
  }
  if (float_mods) s.neg = o.neg, s.abs = o.abs;
  return true;
}
// A scalar instruction's source, as scalar() reads it.
bool scalar_src(const Operand& o, Src& s) {
  if (o.sel != 6 || o.half || o.bits16) return false;
  s = Src();
  switch (o.kind) {
    case OperandKind::Sgpr:
      if (o.index + (o.width >= 2 ? 2u : 1u) > kSgprs) return false;
      s.kind = kSgpr, s.index = o.index, s.wide = o.width >= 2;
      return true;
    case OperandKind::Vcc: s.kind = kVcc, s.wide = o.width >= 2; return true;
    case OperandKind::VccHi: s.kind = kVccHi; return true;
    case OperandKind::Exec: s.kind = kExec; return true;
    case OperandKind::ExecLo: s.kind = kExecLo; return true;
    case OperandKind::ExecHi: s.kind = kExecHi; return true;
    case OperandKind::Null: s.kind = kNull; return true;
    case OperandKind::Inline:
    case OperandKind::Literal:
      s.kind = kConst, s.value64 = static_cast<uint64_t>(o.value);
      return true;
    case OperandKind::InlineFloat: {
      s.kind = kConst;
      if (o.width >= 2) {
        const double d = o.fvalue;
        std::memcpy(&s.value64, &d, 8);
      } else {
        const float f = static_cast<float>(o.fvalue);
        uint32_t b;
        std::memcpy(&b, &f, 4);
        s.value64 = b;
      }
      return true;
    }
    default: return false;
  }
}
// A scalar destination, as write_scalar writes it.
bool scalar_dst(const Operand& o, Src& s) { return scalar_src(o, s) && s.kind != kConst; }
bool vgpr_dst(const Operand& o, Src& s, uint32_t regs) {
  if (o.kind != OperandKind::Vgpr || o.half || o.bits16 || o.sel != 6 || o.index + regs > kVgprs) return false;
  s = Src();
  s.kind = kVgpr, s.index = o.index;
  return true;
}

}  // namespace

namespace {

bool cdna(const Inst& in) { return in.arch == gcn::Target::Gfx90a || in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950; }

// A comparison: named by its opcode, as compare() reads it. False for a type or test the tier does not model.
void classify_cmp(Inst& in, const Tables& T) {
  if (in.src.size() < 2 || in.dst.size() > 1) return;
  const uint32_t raw = in.opcode & 0xFF;
  const bool writes_exec = (raw >= 0x30 && raw < 0x40) || (raw >= 0x50 && raw < 0x60) || (raw >= 0x70 && raw < 0x80) ||
                           (raw >= 0xB0 && raw < 0xC0) || (raw >= 0xD0 && raw < 0xE0) || raw >= 0xF0 || raw == 0x11 || raw == 0x13 ||
                           raw == 0x15;
  const uint32_t op = !writes_exec ? raw : raw < 0x20 ? raw - 1 : raw - 0x10;
  Op& f = in.fast;
  CmpFn k = nullptr;
  bool fmods = false;
  bool raw16 = false;   // (a half or a 16-bit integer source: float modifiers would act on the wrong bits)
  if (op == 0x10 || op == 0x14) {   // v_cmp_class_f32, v_cmp_class_f16: the mask is the second operand's bits
    if (op == 0x14 && !cdna(in)) return;
    k = op == 0x10 ? T.cmp_class32 : in.src[0].neg ? T.cmp_class16_neg : T.cmp_class16;
    fmods = op == 0x10;
    raw16 = op == 0x14;
  } else if (op >= 0x20 && op < 0x30) {
    if (!cdna(in)) return;
    k = T.cmp_f16[op & 0xF], raw16 = true;
  } else if (op >= 0xA0 && op < 0xB0) {
    if (!cdna(in)) return;
    k = op < 0xA8 ? T.cmp_i16[op & 7] : T.cmp_u16[op & 7], raw16 = true;
  } else if (op >= 0x40 && op < 0x50) k = T.cmp_f32[op & 0xF], fmods = true;
  else if (op >= 0xC0 && op < 0xC8) k = T.cmp_i32[op & 7];
  else if (op >= 0xC8 && op < 0xD0) k = T.cmp_u32[op & 7];
  else if ((op >= 0x60 && op < 0x70) || (op >= 0xE0 && op < 0xF0)) {
    const bool is_f = op < 0xC0;
    const Cmp64Fn k64 = is_f ? T.cmp_f64[op & 0xF] : op < 0xE8 ? T.cmp_i64[op & 7] : T.cmp_u64[op & 7];
    if (!lane_src64(in.src[0], f.src[0], is_f) || !lane_src64(in.src[1], f.src[1], is_f)) return void(f = Op());
    if (!in.dst.empty() && !scalar_dst(in.dst[0], f.dst)) return void(f = Op());
    f.kern = reinterpret_cast<AnyFn>(k64);
    f.sub = writes_exec;
    f.unit = kUnitValu;
    f.run = TIER_RUN(h_cmp64);
    return;
  } else return;
  // (a 16-bit integer compare ignores the modifiers a VOP3 form could carry; a half's act on its sign bit)
  if (raw16 && !(op == 0x14 || (op >= 0x20 && op < 0x30)) && (in.src[0].neg || in.src[0].abs || in.src[1].neg || in.src[1].abs)) return void(f = Op());
  bool ok;
  if (op == 0x14) ok = lane_src32_half(in.src[0], f.src[0]) && lane_src32(in.src[1], f.src[1], false);   // (class of a half; the mask is bits)
  else if (op >= 0x20 && op < 0x30) ok = lane_src32_half(in.src[0], f.src[0]) && lane_src32_half(in.src[1], f.src[1]);
  else ok = lane_src32(in.src[0], f.src[0], fmods) && lane_src32(in.src[1], f.src[1], op == 0x10 ? false : fmods);
  if (!ok) return void(f = Op());
  if (!in.dst.empty() && !scalar_dst(in.dst[0], f.dst)) return void(f = Op());
  f.kern = reinterpret_cast<AnyFn>(k);
  f.sub = writes_exec;
  f.unit = kUnitValu;
  f.run = TIER_RUN(h_cmp);
}

// The SDWA form of a 32-bit integer operation: sources cut to the parts they name, the result put in the part dst_sel names.
bool lane_src32_sdwa(const Operand& o, Src& s) {
  if (o.neg || o.abs) return false;
  Operand plain = o;
  plain.sel = 6, plain.sext = false;
  if (!lane_src32(plain, s, false)) return false;
  s.sel = o.sel, s.sext = o.sext;
  return true;
}
bool classify_sdwa(Inst& in) {
  if (in.arch == Target::Gfx1250 || !in.dual.empty() || in.dpp || in.dpp8 || in.bitop3 || in.scaled) return false;
  if ((in.enc != Enc::Vop1 && in.enc != Enc::Vop2) || in.dst.size() != 1 || in.short_name.empty() || in.clamp || in.omod) return false;
  const auto& tab = table();
  const auto it = tab.find(in.short_name);
  if (it == tab.end() || it->second.cat != Cat::Int) return false;
  const Entry& e = it->second;
  if ((e.flags & kFloatSrc) || in.src.size() < static_cast<size_t>(e.arity)) return false;
  Op& f = in.fast;
  if (!vgpr_dst(in.dst[0], f.dst, 1)) return false;
  for (int k = 0; k < e.arity; ++k)
    if (!lane_src32_sdwa(in.src[k], f.src[k])) return f = Op(), false;
  f.dsel = in.dst_sel, f.dunused = in.dst_unused;
  f.kern = reinterpret_cast<AnyFn>(tables().ints[e.k]);
  f.run = h_int_sdwa;
  f.unit = kUnitValu;
  return true;
}


// A DPP integer instruction (v_mov_b32_dpp, v_add_u32_dpp, v_and_b32_dpp ... in their VOP1/VOP2 forms): the first source is read through the
// control, the others as usual.
bool classify_dpp(Inst& in) {
  if (in.arch == Target::Gfx1250 || !in.dual.empty() || in.sdwa || !(in.dpp || in.dpp8) || in.bitop3 || in.scaled) return false;
  if ((in.enc != Enc::Vop1 && in.enc != Enc::Vop2) || in.dst.size() != 1 || in.short_name.empty() || in.clamp || in.omod || in.op_sel) return false;
  uint32_t probe = 0;
  if (!in.dpp8 && dpp_from(in.dpp_ctrl, 0, &probe) < 0) return false;   // (a control the interpreter does not model)
  const auto& tab = table();
  const auto it = tab.find(in.short_name);
  if (it == tab.end() || it->second.cat != Cat::Int) return false;
  const Entry& e = it->second;
  if ((e.flags & (kFloatSrc | kRawSrc)) || in.src.size() < static_cast<size_t>(e.arity)) return false;
  if ((e.flags & kCdnaOnly) && in.arch != Target::Gfx90a && in.arch != Target::Gfx942 && in.arch != Target::Gfx950) return false;
  Op& f = in.fast;
  if (!vgpr_dst(in.dst[0], f.dst, 1)) return false;
  for (int k = 0; k < e.arity; ++k)
    if (!lane_src32(in.src[k], f.src[k], false) || f.src[k].neg || f.src[k].abs) return f = Op(), false;
  if (f.src[0].kind != kVgpr) return f = Op(), false;
  f.sub = in.dpp_ctrl;
  f.off0 = static_cast<int32_t>((in.row_mask & 15) | (in.bank_mask & 15) << 4 | (in.bound_ctrl ? 1u << 8 : 0) | (in.fi ? 1u << 9 : 0) |
                                (in.dpp8 ? 1u << 10 : 0));
  f.kern = reinterpret_cast<AnyFn>(tables().ints[e.k]);
  f.run = h_int_dpp;
  f.unit = kUnitValu;
  return true;
}


// A VOPD pair: both halves have handlers of their own (classified when the halves were finalized), and X's result is one a handler writes whole.
void classify_vopd(Inst& in) {
  if (in.arch == Target::Gfx1250 || in.dual.size() != 2) return;
  const Inst& x = in.dual[0];
  const Inst& y = in.dual[1];
  for (const Inst* h : {&x, &y})
    if (!h->fast.run || h->fast.mach || h->dst.size() != 1 || h->dst[0].kind != OperandKind::Vgpr) return;
  const uint32_t xd = x.dst[0].index, xw = std::max(1u, static_cast<uint32_t>(x.dst[0].width));   // (a 64-bit result is two registers)
  if (xw > 2 || xd + xw > kVgprs) return;
  bool y_reads_x = false;
  for (const Operand& o : y.src)
    y_reads_x = y_reads_x || (o.kind == OperandKind::Vgpr && o.index < xd + xw && xd < o.index + std::max<uint32_t>(o.width, 1));
  Op& f = in.fast;
  f.mach = 12;
  f.run = h_never;
  f.off0 = static_cast<int32_t>(xd);
  f.words = static_cast<uint8_t>(xw);
  f.sub = y_reads_x ? 1 : 0;
}

}  // namespace

void tier_classify(Inst& in) {
  in.fast = Op();
  if (in.sdwa && classify_sdwa(in)) return;
  if ((in.dpp || in.dpp8) && classify_dpp(in)) return;
  if (!in.dual.empty()) {   // (VOPD: the Machine's tier_vopd runs the two halves' own handlers)
    classify_vopd(in);
    return;
  }
  // Not modelled: gfx1250 (it has its own instruction interceptions), sub-dword and cross-lane forms, 16-bit halves,
  // per-source op_sel.
  if (in.arch == Target::Gfx1250 || in.sdwa || in.dpp || in.dpp8 || (in.op_sel != 0 && in.enc != Enc::Vop3p && in.name.rfind("v_permlane", 0) != 0) || in.bitop3 || in.scaled) return;
  const Tables& T = tables();
  Op& f = in.fast;
  const bool vector_enc = in.enc == Enc::Vop1 || in.enc == Enc::Vop2 || in.enc == Enc::Vop3 || in.enc == Enc::Vopc || in.enc == Enc::Vopd;
  if (in.flags & kInstCmp) {
    if (in.enc == Enc::Vopc || in.enc == Enc::Vop3) classify_cmp(in, T);
    return;
  }
  if (in.enc == Enc::Smem) {   // (scalar memory: the Machine's scalar_load, without step()'s frame; the real-time clock reads differently each time)
    if (in.mem.smem == SmemKind::Time && in.mem.smem_realtime) return;
    f.mach = 7;
    f.run = h_never;
    return;
  }
  // The caches' write-backs and invalidates (what a fence compiles to): every access here reaches memory directly, so the instruction is a fence of the
  // host's order and a count.
  if (in.enc == Enc::Mubuf) {
    const std::string& nm = in.name;
    if (nm == "buffer_wbl2" || nm == "buffer_inv" || nm == "buffer_wbinvl1" || nm == "buffer_wbinvl1_vol" || nm == "buffer_invl2" || nm == "buffer_gl0_inv" ||
        nm == "buffer_gl1_inv") {
      f.mach = 11;
      f.run = h_never;
    }
    return;
  }
  // RDNA's separate wait counters in their SOPK form, and the ISA marker: nothing is outstanding to wait for (the scalar unit's count, as scalar_alu).
  if (in.enc == Enc::Sopk && (in.name == "s_waitcnt_vscnt" || in.name == "s_waitcnt_vmcnt" || in.name == "s_waitcnt_expcnt" || in.name == "s_waitcnt_lgkmcnt" ||
                              in.name == "s_version")) {
    f.run = b_nop;
    f.unit = kUnitSalu;
    return;
  }
  static const Entry kCarry = {Cat::Carry, 0, 0, 0, nullptr, 0};
  const Entry* ep = nullptr;
  if (vector_enc && in.name.find("_co_u32") != std::string::npos) {   // (carry_alu takes these first, under their own names)
    ep = &kCarry;
  } else {
    std::string eff = in.short_name.empty() ? in.name : in.short_name;
    if ((in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950) && (eff == "s_max_u32" || eff == "s_max_i32")) eff += "_ge";
    // (gfx942 and gfx950 take a shift count of this one over 4 as zero, CDNA3 and CDNA4 guides: a handler of its own.)
    if ((in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950) && eff == "v_lshl_add_u64") eff += "_g";
    const auto& tab = table();
    const auto it = tab.find(eff);
    if (it == tab.end()) {
      // RDNA4's waits (s_wait_alu, s_wait_kmcnt, ...): no counter is ever outstanding (s_wait_loadcnt is in the table).
      if (in.enc == Enc::Sopp && (in.flags & kInstSWait)) f.run = b_nop;
      return;
    }
    ep = &it->second;
  }
  const Entry& e = *ep;
  const auto fail = [&] { f = Op(); };
  switch (e.cat) {
    case Cat::Int:
    case Cat::F32: {
      if (!vector_enc || in.dst.size() != 1 || in.src.size() < static_cast<size_t>(e.arity)) return;
      const bool is_f = e.cat == Cat::F32;
      if (!vgpr_dst(in.dst[0], f.dst, 1)) return fail();
      if ((e.flags & kNoClamp) && in.clamp) return fail();
      if ((e.flags & kCdnaOnly) && in.arch != gcn::Target::Gfx90a && in.arch != gcn::Target::Gfx942 && in.arch != gcn::Target::Gfx950) return fail();
      if ((e.flags & (kNoOmod | kFloatOut)) && in.omod) return fail();
      if ((e.flags & kFloatOut) && in.clamp) return fail();
      // (CDNA3 and CDNA4 apply the clamp bit to the integer instructions that have one and to the conversions to float: the interpreter's.)
      if (in.clamp && (in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950) && (!is_f || (e.flags & kIgnoreMods))) return fail();
      // (gfx942 and gfx950 apply or ignore an output modifier by MODE and flush what it makes denormal, CDNA3 and CDNA4 guides, 6.5: the interpreter's.)
      if (in.omod && (in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950) && !(e.flags & kIgnoreMods)) return fail();
      for (int k = 0; k < e.arity; ++k) {
        if ((e.k == ALIGNBIT || e.k == ALIGNBYTE) && !is_f && k == 2 && in.src[2].bits16 && !in.src[2].hi && !in.src[2].sext && (in.src[2].sel == 6 || in.src[2].sel == 4)) {
          // (RDNA3 and RDNA4 name the shift count a 16-bit operand; the low half of a register, whose low bits are all the kernel reads of it)
          Operand shift = in.src[2];
          shift.bits16 = shift.half = false;
          shift.sel = 6;
          if (!lane_src32(shift, f.src[2], false)) return fail();
          continue;
        }
        if (e.flags & kRawSrc) {   // (a half: its inline float constants too)
          if (!lane_src32_half(in.src[k], f.src[k])) return fail();
        } else if (!lane_src32(in.src[k], f.src[k], (is_f && !((e.flags & kIntSrc1) && k == 1)) || (e.flags & kFloatSrc))) {
          return fail();
        }
      }
      if (is_f) {
        if (e.flags & kFmac) f.src[2] = Src(), f.src[2].kind = kVgpr, f.src[2].index = f.dst.index;
        if (e.flags & kConstLast) {
          f.src[2] = Src();
          f.src[2].kind = kConst, f.src[2].value = static_cast<uint32_t>(in.src[2].value);
        }
        if (e.flags & kConstMid) {
          f.src[1] = Src();
          f.src[1].kind = kConst, f.src[1].value = static_cast<uint32_t>(in.src[1].value);
        }
        if (!(e.flags & kIgnoreMods)) f.omod = in.omod, f.clamp = in.clamp;
        if (e.flags & kMinMax) f.sub = 1;
        switch (e.k) {
          case FADD: f.nan = kNanAdd; break;
          case FSUB: f.nan = kNanSub; break;
          case FSUBREV: f.nan = kNanSubRev; break;
          case FMUL: f.nan = kNanMul; break;
          case FMA: f.nan = kNanFma; break;
          // (one-input functions of the host's libm or one instruction: the kernel's NaN is the interpreter's, there being no second payload to choose)
          case FRCP: case FSQRT: case FRSQ: case FEXP: case FLOG: case FFLOOR: case FCEIL: case FTRUNC: case FFRACT: case FLDEXP: case FCVT_F32_F16:
            f.nan = kNanPass;
            break;
          default: break;   // v_min_f32 and v_max_f32 (MODE.IEEE decides), the rounding and the angles in turns: the interpreter's
        }
        const int kernel = (e.flags & kTrigTurns) && (in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950) ? e.k + 1 : e.k;
        f.kern = reinterpret_cast<AnyFn>(T.f32[kernel]);
        f.run = TIER_RUN(h_f32);
      } else {
        f.kern = reinterpret_cast<AnyFn>(T.ints[e.k]);
        bool rows_only = true;   // (every source a vector register as it is)
        for (const Src& src : f.src)
          if (src.kind && (src.kind != kVgpr || src.neg || src.abs)) rows_only = false;
        f.run = e.k == MOV ? TIER_RUN(h_mov) : rows_only ? TIER_RUN(h_intv) : TIER_RUN(h_int);
      }
      f.unit = kUnitValu;
      return;
    }
    case Cat::F64: {
      if (!vector_enc || in.dst.size() != 1 || in.src.size() < static_cast<size_t>(e.arity)) return;
      if (!vgpr_dst(in.dst[0], f.dst, 2)) return fail();
      if (in.omod && (in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950)) return fail();
      for (int k = 0; k < e.arity; ++k)
        if (!lane_src64(in.src[k], f.src[k], true)) return fail();
      if (e.flags & kFmac) f.src[2] = Src(), f.src[2].kind = kVgpr, f.src[2].index = f.dst.index, f.src[2].wide = true;
      f.omod = in.omod, f.clamp = in.clamp;
      if (e.k == DADD) f.nan = kNanAdd;
      else if (e.k == DMUL) f.nan = kNanMul;
      else if (e.k == DFMA && !(e.flags & kFmac)) f.nan = kNanFma;   // (v_fmac_f64 is a plain std::fma: the interpreter's)
      else if (e.k != DFMA && e.k != DRNDNE && e.k != DMIN && e.k != DMAX) f.nan = kNanPass;      // (a one-input function of the host's: no second payload to choose; nearbyint's quieting is the compiler's)
      if ((e.k == DADD || e.k == DMUL || e.k == DFMA) && !f.omod && !f.clamp) {   // (the pairs' rows, no 64-bit arrays)
        const uint32_t op = e.k == DADD ? 0 : e.k == DMUL ? 1 : 2;
        uint32_t uniform = 0;   // (the operands that are one value in every lane)
        for (uint32_t k = 0; k < (op == 2 ? 3u : 2u); ++k)
          if (f.src[k].kind != kVgpr) uniform |= 1u << k;
        f.sub = op;
        f.kern = reinterpret_cast<AnyFn>(T.f64_rows[op * 8 + uniform]);
        f.run = TIER_RUN(h_f64r);
      } else {
        if (e.k == DMIN || e.k == DMAX) f.sub = 1;   // (opposite zeros go to the interpreter)
        f.kern = reinterpret_cast<AnyFn>(T.f64[e.k]);
        f.run = TIER_RUN(h_f64);
      }
      f.unit = kUnitValu;
      return;
    }
    case Cat::L64:
    case Cat::L64N: {
      if (!vector_enc || in.dst.size() != 1 || in.src.size() < static_cast<size_t>(e.arity)) return;
      if (!vgpr_dst(in.dst[0], f.dst, e.cat == Cat::L64 ? 2 : 1)) return fail();
      if (((e.flags & kFloatOut) && (in.omod || in.clamp)) || ((e.flags & kFloatSrc) && (in.omod || in.clamp))) return fail();
      for (int k = 0; k < e.arity; ++k)
        if ((e.flags & kFloatSrc) && (in.src[k].neg || in.src[k].abs)) return fail();
      for (int k = 0; k < e.arity; ++k)
        if (!((e.sub >> k & 1) ? lane_src64(in.src[k], f.src[k], false) : lane_src32(in.src[k], f.src[k], false))) return fail();
      f.sub = e.sub;
      f.kern = reinterpret_cast<AnyFn>(T.l64[e.k]);
      f.run = e.cat == Cat::L64N ? TIER_RUN(h_l64n) : e.k == L_MOV && (e.sub & 1) ? TIER_RUN(h_mov64) : TIER_RUN(h_l64);
      f.unit = kUnitValu;
      return;
    }
    case Cat::Carry: {
      if (!vector_enc || in.dst.size() != 2 || in.src.size() < 2) return;
      const std::string& nm = in.name;
      const bool takes = nm.find("_addc_") != std::string::npos || nm.find("_subb") != std::string::npos;
      if (takes && in.src.size() < 3) return;
      if (!vgpr_dst(in.dst[0], f.dst, 1) || !scalar_dst(in.dst[1], f.dst2)) return fail();
      if (!lane_src32(in.src[0], f.src[0], false) || !lane_src32(in.src[1], f.src[1], false)) return fail();
      if (takes && !scalar_src(in.src[2], f.src[2])) return fail();
      f.sub = (nm.rfind("v_add", 0) == 0 ? 1u : 0u) | (takes ? 2u : 0u) |
              (nm.find("subbrev") != std::string::npos || nm.find("subrev") != std::string::npos ? 4u : 0u);
      f.kern = reinterpret_cast<AnyFn>(T.carry[f.sub]);
      f.run = TIER_RUN(h_carry);
      f.unit = kUnitValu;
      return;
    }
    case Cat::Lane: {
      if (e.k == 2) {   // s_movk_i32: the (signed) constant into a scalar register
        if (in.enc != Enc::Sopk || in.dst.size() != 1 || !in.src.empty()) return;
        if (!scalar_dst(in.dst[0], f.dst)) return fail();
        f.src[0].kind = kConst, f.src[0].value64 = static_cast<uint64_t>(static_cast<int64_t>(in.simm));
        f.run = s_mov;
        f.unit = kUnitSalu;
        return;
      }
      if (!vector_enc || in.dst.size() != 1 || in.src.size() < 2) return;
      if (e.k == 0) {
        if (in.src[0].kind != OperandKind::Vgpr || in.src[0].sel != 6 || in.src[0].index >= kVgprs) return;
        if (!scalar_dst(in.dst[0], f.dst) || !scalar_src(in.src[1], f.src[1])) return fail();
        f.src[0].kind = kVgpr, f.src[0].index = in.src[0].index;
        f.run = h_readlane;
      } else {
        if (!vgpr_dst(in.dst[0], f.dst, 1) || !scalar_src(in.src[0], f.src[0]) || !scalar_src(in.src[1], f.src[1])) return fail();
        f.run = h_writelane;
      }
      f.unit = kUnitValu;
      return;
    }
    case Cat::Global: {
      // (The interpreter's Machine runs these: classify only checks that its row reads are safe, and records what it does.)
      if (in.enc != Enc::Flat || in.segment != Inst::Segment::Global || in.scale_offset || in.src.empty()) return;
      const Operand& a = in.src[0];
      if (a.kind != OperandKind::Vgpr || a.sel != 6 || a.index + (in.has_saddr ? 1u : 2u) > kVgprs) return;
      const bool store = e.k == 2 || e.k == 5;
      const uint32_t regs = e.k >= 4 ? 1u : static_cast<uint32_t>(e.sub);   // (a byte or a short is one register)
      if (store ? in.src.size() < 2 : in.dst.empty()) return;
      const Operand& data = store ? in.src[1] : in.dst[0];
      if (data.kind != OperandKind::Vgpr || data.sel != 6 || data.width != regs || data.index + regs > kVgprs) return;
      if (!store && in.dst.size() != 1) return;
      f.mach = static_cast<uint8_t>(e.k);
      f.words = static_cast<uint8_t>(e.sub);
      f.sub = e.flags;
      f.run = h_never;   // (not called: step() sends `mach` instructions to the Machine)
      return;
    }
    case Cat::Ds: {
      if (in.enc != Enc::Ds || (in.flags & kInstGws) || in.gds || in.src.empty()) return;
      if (in.src[0].kind != OperandKind::Vgpr || in.src[0].index >= kVgprs || in.src[0].sel != 6) return;
      f.src[0].kind = kVgpr, f.src[0].index = in.src[0].index;
      f.sub = static_cast<uint32_t>(e.k);
      f.words = static_cast<uint8_t>(e.sub);
      f.off0 = in.offset, f.off1 = in.offset1;
      if (e.k == kDsSwizzle) {   // (the data register is the one operand; the offset is the pattern)
        if (in.dst.size() != 1 || in.dst[0].kind != OperandKind::Vgpr || in.dst[0].index >= kVgprs || in.dst[0].width != 1) return;
        f.dst.kind = kVgpr, f.dst.index = in.dst[0].index;
        f.run = TIER_RUN(h_ds);
        f.unit = kUnitLds;
        return;
      }
      if (e.k == kDsBperm || e.k == kDsPerm) {
        if (in.dst.size() != 1 || in.src.size() < 2) return;
        if (in.dst[0].kind != OperandKind::Vgpr || in.dst[0].index >= kVgprs || in.dst[0].width != 1) return;
        if (in.src[1].kind != OperandKind::Vgpr || in.src[1].index >= kVgprs) return;
        f.dst.kind = kVgpr, f.dst.index = in.dst[0].index;
        f.src[1].kind = kVgpr, f.src[1].index = in.src[1].index;
        f.run = TIER_RUN(h_ds);
        f.unit = kUnitLds;
        return;
      }
      const bool read = e.k == kDsRead || e.k == kDsRead2;
      if (read) {
        if (in.dst.size() != 1 || in.dst[0].kind != OperandKind::Vgpr || in.dst[0].index + e.sub > kVgprs) return;
        if (in.dst[0].width != e.sub) return;
        f.dst.kind = kVgpr, f.dst.index = in.dst[0].index;
      } else {
        const size_t need = e.k == kDsWrite2 ? 3 : 2;
        if (in.src.size() < need) return;
        for (size_t k = 1; k < need; ++k) {
          const uint32_t width = e.k == kDsWrite2 ? 1 : e.sub;   // (write2's two data operands are a register each; read2's destination is the pair)
          if (in.src[k].kind != OperandKind::Vgpr || in.src[k].index + width > kVgprs || in.src[k].width != width) return;
          f.src[k].kind = kVgpr, f.src[k].index = in.src[k].index;
        }
        f.store = e.k != kDsAdd;   // (ds_add_u32 is not one of the stores that wait for the group's other waves)
      }
      if (e.k == kDsRead2 || e.k == kDsWrite2) {
        // (Offsets in words: the instruction's two 8-bit counts.)
        const uint32_t unit = e.flags & 1 ? 4 * 64 : 4;   // (the st64 forms count in 64s of words)
        f.off0 = static_cast<int32_t>(static_cast<uint32_t>(in.offset) * unit);
        f.off1 = static_cast<int32_t>(static_cast<uint32_t>(in.offset1) * unit);
      }
      f.run = TIER_RUN(h_ds);
      f.unit = kUnitLds;
      return;
    }
    case Cat::Pk: {
      if (in.enc != Enc::Vop3p || in.dst.size() != 1 || in.src.size() < static_cast<size_t>(e.arity)) return;
      if (!vgpr_dst(in.dst[0], f.dst, 2)) return fail();
      uint32_t sub = 0;
      for (int k = 0; k < e.arity; ++k) {
        const Operand& o = in.src[k];
        const unsigned top_lo = in.op_sel >> k & 1, top_hi = in.op_sel_hi >> k & 1;
        Src& s = f.src[k];
        s = Src();
        if (o.sel != 6 || o.half || o.bits16) return fail();
        if (o.kind == OperandKind::Vgpr) {
          if (o.index + 2 > kVgprs) return fail();   // (either half of the pair)
          s.kind = kVgpr, s.index = o.index;
        } else if (o.kind == OperandKind::Sgpr) {
          if (o.index + (o.width >= 2 ? 2u : 1u) > kSgprs) return fail();
          s.kind = kSgpr, s.index = o.index, s.wide = o.width >= 2;
        } else if (o.kind == OperandKind::Inline || o.kind == OperandKind::InlineFloat || o.kind == OperandKind::Literal) {
          if (top_lo || top_hi) return fail();   // (the interpreter refuses the second part of a constant)
          s.kind = kConst;
          if (o.kind == OperandKind::InlineFloat) {
            const float fv = static_cast<float>(o.fvalue);
            std::memcpy(&s.value, &fv, 4);
          } else {
            s.value = static_cast<uint32_t>(o.value);
          }
        } else {
          return fail();
        }
        sub |= top_lo << k | top_hi << (3 + k) | (in.neg_lo >> k & 1) << (6 + k) | (in.neg_hi >> k & 1) << (9 + k);
      }
      f.sub = sub;
      f.clamp = in.clamp;
      f.nan = e.k == FADD ? kNanAdd : e.k == FMUL ? kNanMul : kNanPkFma;
      f.kern = reinterpret_cast<AnyFn>(T.f32[e.k]);
      f.run = h_pk;
      f.unit = kUnitValu;
      return;
    }
    case Cat::Pk16: {
      if (in.enc != Enc::Vop3p || in.dst.size() != 1 || in.src.size() < static_cast<size_t>(e.arity)) return;
      if (!vgpr_dst(in.dst[0], f.dst, 1)) return fail();
      const bool clamps = in.clamp && e.sub;
      uint32_t sub = 0;
      for (int k = 0; k < e.arity; ++k) {
        const Operand& o = in.src[k];
        Src& s = f.src[k];
        s = Src();
        if (o.sel != 6 || o.half || o.bits16) return fail();
        const unsigned top[2] = {static_cast<unsigned>(in.op_sel >> k & 1), static_cast<unsigned>(in.op_sel_hi >> k & 1)};
        if (o.kind == OperandKind::Vgpr) {
          if (o.index >= kVgprs) return fail();
          s.kind = kVgpr, s.index = o.index;
        } else if (o.kind == OperandKind::Sgpr) {
          if (o.index >= kSgprs || o.width >= 2) return fail();
          s.kind = kSgpr, s.index = o.index;
        } else if (o.kind == OperandKind::Inline || o.kind == OperandKind::InlineFloat || o.kind == OperandKind::Literal) {
          // (the interpreter's constant_half: a float constant is a half in the low 16 bits, zero above; any other is its 32 bits cut by the half)
          const auto half = [&](unsigned t) -> uint32_t {
            if (o.kind == OperandKind::InlineFloat) return t ? 0u : static_cast<uint32_t>([&] { const _Float16 h = static_cast<_Float16>(o.fvalue); uint16_t b; std::memcpy(&b, &h, 2); return b; }());
            return static_cast<uint16_t>(static_cast<uint32_t>(o.value) >> (16 * t));
          };
          s.kind = kConst, s.value = half(top[0]), s.value64 = half(top[1]);
        } else {
          return fail();
        }
        sub |= top[0] << k | top[1] << (3 + k);
      }
      f.sub = sub;
      f.kern = reinterpret_cast<AnyFn>(T.ints[clamps ? e.sub - 1 : e.k]);
      f.run = h_pk16;
      f.unit = kUnitValu;
      return;
    }
    case Cat::Permlane: {
      if (in.enc != Enc::Vop3 || in.dst.size() != 1 || in.src.size() < 3) return;
      if (!vgpr_dst(in.dst[0], f.dst, 1)) return fail();
      if (in.src[0].kind != OperandKind::Vgpr || in.src[0].sel != 6 || in.src[0].half || in.src[0].bits16 || in.src[0].index >= kVgprs) return fail();
      f.src[0].kind = kVgpr, f.src[0].index = in.src[0].index;
      if (!scalar_src(in.src[1], f.src[1]) || !scalar_src(in.src[2], f.src[2])) return fail();
      f.sub = in.op_sel & 3;
      f.off0 = e.k;
      f.run = h_permlane;
      f.unit = kUnitValu;
      return;
    }
    case Cat::Mad64: {
      if (!vector_enc || in.dst.empty() || in.dst.size() > 2 || in.src.size() < 3) return;
      if (!vgpr_dst(in.dst[0], f.dst, 2)) return fail();
      if (!lane_src32(in.src[0], f.src[0], false) || !lane_src32(in.src[1], f.src[1], false) || !lane_src64(in.src[2], f.src[2], false)) return fail();
      if (in.dst.size() == 2 && !scalar_dst(in.dst[1], f.dst2)) return fail();
      f.kern = reinterpret_cast<AnyFn>(T.mad_u64);
      f.run = TIER_RUN(h_mad_u64);
      f.unit = kUnitValu;
      return;
    }
    case Cat::Cndmask: {
      if (!vector_enc || in.dst.size() != 1 || in.src.size() < 3) return;
      if (!vgpr_dst(in.dst[0], f.dst, 1)) return fail();
      if (!lane_src32(in.src[0], f.src[0], true) || !lane_src32(in.src[1], f.src[1], true) || !scalar_src(in.src[2], f.src[2])) return fail();
      f.run = TIER_RUN(h_cndmask);
      f.unit = kUnitValu;
      return;
    }
    case Cat::Readfirstlane: {
      if (!vector_enc || in.dst.size() != 1 || in.src.size() < 1) return;
      if (in.src[0].kind != OperandKind::Vgpr || in.src[0].sel != 6 || in.src[0].index >= kVgprs) return;
      if (!scalar_dst(in.dst[0], f.dst)) return fail();
      f.src[0].kind = kVgpr, f.src[0].index = in.src[0].index;
      f.run = h_readfirstlane;
      f.unit = kUnitValu;
      return;
    }
    case Cat::S2: {
      if ((in.enc != Enc::Sop1 && in.enc != Enc::Sop2) || in.dst.size() != 1 || in.src.size() < static_cast<size_t>(e.arity)) return;
      if (!scalar_dst(in.dst[0], f.dst)) return fail();
      for (size_t k = 0; k < in.src.size() && k < 3; ++k)
        if (!scalar_src(in.src[k], f.src[k])) return fail();
      f.run = e.fn;
      // (the shapes the compilers' scalar code is made of: one-word registers and constants, no modifiers to look at)
      const auto v = s2_variants().find(e.fn);
      if (v != s2_variants().end() && f.dst.kind == kSgpr && !f.dst.wide) {
        const auto reg = [](const Src& x) { return x.kind == kSgpr && !x.wide; };
        const auto con = [](const Src& x) { return x.kind == kConst; };
        if (e.arity == 1) f.run = reg(f.src[0]) ? v->second.ss : con(f.src[0]) ? v->second.cc : e.fn;
        else if (e.arity == 2) f.run = reg(f.src[0]) && reg(f.src[1]) ? v->second.ss : reg(f.src[0]) && con(f.src[1]) ? v->second.sc
                                       : con(f.src[0]) && con(f.src[1]) ? v->second.cc : e.fn;
      }
      f.unit = kUnitSalu;
      return;
    }
    case Cat::ScalarCmp: {
      if (in.enc != Enc::Sopc || in.src.size() != 2) return;
      for (size_t k = 0; k < 2; ++k)
        if (!scalar_src(in.src[k], f.src[k])) return fail();
      f.sub = static_cast<uint32_t>(e.k);
      f.run = s_cmp;
      if (const Fn pick = s_cmp_pick(f.sub, f.src[0], f.src[1])) f.run = pick;
      f.unit = kUnitSalu;
      return;
    }
    case Cat::Acc: {
      if ((in.enc != Enc::Vop1 && in.enc != Enc::Vop3) || in.dst.size() != 1 || in.src.size() != 1) return;
      f.mach = 10;
      f.run = h_never;
      return;
    }
    case Cat::PkF16: {   // (the Machine's packed_f16_wave does the arithmetic whole-wave; it says if the instruction is not the plain case)
      if (in.enc != Enc::Vop3p || in.dst.size() != 1 || in.src.size() < static_cast<size_t>(e.arity)) return;
      f.mach = 8;
      f.sub = static_cast<uint32_t>(e.k);
      f.run = h_never;
      return;
    }
    case Cat::PkMov: {
      if (in.enc != Enc::Vop3p || in.dst.size() != 1 || in.src.size() < 2) return;
      if (!vgpr_dst(in.dst[0], f.dst, 2)) return fail();
      for (int k = 0; k < 2; ++k) {
        const Operand& o = in.src[k];
        const unsigned top = in.op_sel >> k & 1;
        Src& s = f.src[k];
        s = Src();
        if (o.sel != 6 || o.half || o.bits16) return fail();
        if (o.kind == OperandKind::Vgpr) {
          if (o.index + 2 > kVgprs) return fail();
          s.kind = kVgpr, s.index = o.index;
        } else if (o.kind == OperandKind::Sgpr) {
          if (o.index + (o.width >= 2 ? 2u : 1u) > kSgprs) return fail();
          s.kind = kSgpr, s.index = o.index, s.wide = o.width >= 2;
        } else if (o.kind == OperandKind::Inline || o.kind == OperandKind::Literal) {
          if (top) return fail();   // (the interpreter refuses the second part of a constant)
          s.kind = kConst, s.value = static_cast<uint32_t>(o.value);
        } else if (o.kind == OperandKind::InlineFloat) {
          if (top) return fail();
          const float fv = static_cast<float>(o.fvalue);
          s.kind = kConst, std::memcpy(&s.value, &fv, 4);
        } else {
          return fail();
        }
      }
      f.sub = in.op_sel & 3;
      f.run = h_pkmov;
      f.unit = kUnitValu;
      return;
    }
    case Cat::Div: {
      if (!vector_enc || in.src.size() < 3) return;
      const bool dbl = e.k >= 4, scale = (e.k & 3) == 0;
      if (!vgpr_dst(in.dst.empty() ? Operand() : in.dst[0], f.dst, dbl ? 2 : 1)) return fail();
      if (scale ? (in.dst.size() != 2 || !scalar_dst(in.dst[1], f.dst2)) : in.dst.size() != 1) return fail();
      for (int k = 0; k < 3; ++k)
        if (!(dbl ? lane_src64(in.src[k], f.src[k], true) : lane_src32(in.src[k], f.src[k], true))) return fail();
      f.sub = static_cast<uint32_t>(e.k);
      f.run = h_div;
      f.unit = kUnitValu;
      return;
    }
    case Cat::FmaMix: {
      if (in.enc != Enc::Vop3p || in.dst.size() != 1 || in.src.size() < 3) return;
      if (!vgpr_dst(in.dst[0], f.dst, 1)) return fail();
      uint32_t sub = static_cast<uint32_t>(e.k) << 12;
      for (int k = 0; k < 3; ++k) {
        const Operand& o = in.src[k];
        const bool half = in.op_sel_hi >> k & 1;
        // (a half read from an inline float constant is that constant as a half; any other source's 32 bits are shifted by op_sel)
        if (half && o.kind == OperandKind::InlineFloat) {
          if (!lane_src32_half(o, f.src[k])) return fail();
        } else if (!lane_src32(o, f.src[k], false)) {
          return fail();
        }
        sub |= (half ? 1u : 0u) << k | (half && !(o.kind == OperandKind::InlineFloat) && (in.op_sel >> k & 1) ? 1u : 0u) << (3 + k) |
               (o.neg ? 1u : 0u) << (6 + k) | (o.abs ? 1u : 0u) << (9 + k);
      }
      f.sub = sub;
      f.clamp = in.clamp;
      f.run = h_fma_mix;
      f.unit = kUnitValu;
      return;
    }
    case Cat::Endpgm: {
      if (in.enc != Enc::Sopp) return;
      f.mach = 6;   // (the Machine ends the wave)
      f.run = h_never;
      return;
    }
    case Cat::Sopk: {
      if (in.enc != Enc::Sopk || in.dst.size() != 1) return;
      if (!scalar_dst(in.dst[0], f.dst) || !scalar_src(in.dst[0], f.src[0])) return fail();
      f.sub = static_cast<uint32_t>(e.k);
      f.off0 = in.simm;
      f.run = s_sopk;
      f.unit = kUnitSalu;
      return;
    }
    case Cat::Cmpk: {
      if (in.enc != Enc::Sopk || in.opcode < 0x02 || in.opcode > 0x0d) return;
      if (in.dst.empty() && in.src.empty()) return;
      const Operand& x = in.dst.empty() ? in.src[0] : in.dst[0];   // (gfx9 keeps the compared register as the destination field, RDNA as the source)
      if (!scalar_src(x, f.src[0])) return fail();
      f.sub = (in.opcode - 0x02) % 6 + (in.opcode >= 0x08 ? 6u : 0u);
      f.off0 = in.simm;
      f.run = s_cmpk;
      f.unit = kUnitSalu;
      return;
    }
    case Cat::Bitcmp: {
      if (in.enc != Enc::Sopc || in.src.size() != 2) return;
      for (size_t k = 0; k < 2; ++k)
        if (!scalar_src(in.src[k], f.src[k])) return fail();
      f.sub = static_cast<uint32_t>(e.k);
      f.run = s_bitcmp;
      f.unit = kUnitSalu;
      return;
    }
    case Cat::Saveexec: {
      if (in.enc != Enc::Sop1 || in.dst.size() != 1 || in.src.size() != 1) return;
      if (!scalar_dst(in.dst[0], f.dst) || !scalar_src(in.src[0], f.src[0])) return fail();
      f.sub = static_cast<uint32_t>(e.k >= 10 ? e.k - 10 : e.k);
      f.run = e.k >= 10 ? s_saveexec32 : s_saveexec;
      f.unit = kUnitSalu;
      return;
    }
    case Cat::Branch: {
      if (in.enc != Enc::Sopp) return;
      f.run = e.fn;
      f.unit = (in.flags & kInstBranchCount) ? kUnitBranch : kUnitNone;
      if (e.fn == h_never) {   // s_barrier: the Machine stops the wave at it
        f.mach = 3;
        f.unit = kUnitNone;
        return;
      }
      if (e.fn == b_waitcnt) {
        const uint32_t imm = static_cast<uint32_t>(in.simm);
        if (in.name == "s_wait_loadcnt") f.sub = imm & 0x3F;
        else f.sub = in.arch == Target::Gfx1100 ? (imm >> 10) & 0x3F : (imm & 0xF) | ((imm >> 14) & 3) << 4;
      }
      return;
    }
    default: return;
  }
}

#endif  // VGPU_TIER_X86

}  // namespace vgpu::amd::gcn
