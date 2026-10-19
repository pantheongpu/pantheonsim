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
  X(MAX3_U, k_max3_u) X(XOR3, k_xor3) X(MBCNT_LO, k_mbcnt_lo) X(MBCNT_HI, k_mbcnt_hi) X(CVT_I32_F32, k_cvt_i32_f32) X(CVT_U32_F32, k_cvt_u32_f32) X(CVT_F32_U, k_cvt_f32_u) X(CVT_F32_I, k_cvt_f32_i) X(CVT_F32_UBYTE0, k_cvt_f32_ubyte0)
#define TIER_F32_OPS(X) X(FADD, k_fadd) X(FSUB, k_fsub) X(FSUBREV, k_fsubrev) X(FMUL, k_fmul) X(FMA, k_ffma) X(FMIN, k_fmin) X(FMAX, k_fmax) X(FRCP, k_frcp)
#define TIER_F64_OPS(X) X(DADD, k_dadd) X(DMUL, k_dmul) X(DFMA, k_dfma)
#define TIER_L64_OPS(X) X(MOV, l_mov) X(LSHLREV, l_lshlrev) X(LSHRREV, l_lshrrev) X(ASHRREV, l_ashrrev) X(LSHL_ADD, l_lshl_add) X(LSHL_ADD_G, l_lshl_add_g) X(CVT_F64_U, l_cvt_f64_u) X(CVT_F64_I, l_cvt_f64_i)

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
};
const Tables& tables() {
  static const Tables base = {base_impl::base_impl_ns::kIntTab, base_impl::base_impl_ns::kF32Tab, base_impl::base_impl_ns::kF64Tab,
                              base_impl::base_impl_ns::kL64Tab, base_impl::base_impl_ns::kCmpU32, base_impl::base_impl_ns::kCmpI32,
                              base_impl::base_impl_ns::kCmpF32, base_impl::base_impl_ns::kCmpU64, base_impl::base_impl_ns::kCmpI64, base_impl::base_impl_ns::kCmpF64};
  static const Tables avx2 = {avx2_impl::avx2_impl_ns::kIntTab, avx2_impl::avx2_impl_ns::kF32Tab, avx2_impl::avx2_impl_ns::kF64Tab,
                              avx2_impl::avx2_impl_ns::kL64Tab, avx2_impl::avx2_impl_ns::kCmpU32, avx2_impl::avx2_impl_ns::kCmpI32,
                              avx2_impl::avx2_impl_ns::kCmpF32, avx2_impl::avx2_impl_ns::kCmpU64, avx2_impl::avx2_impl_ns::kCmpI64, avx2_impl::avx2_impl_ns::kCmpF64};
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
inline void merge(uint32_t* d, const uint32_t* r, uint64_t ex, unsigned n) {
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

inline bool full(uint64_t ex, unsigned n) { return n == 64 ? ex == ~uint64_t{0} : (ex & 0xFFFFFFFFull) == 0xFFFFFFFFull; }

inline uint64_t rd(const Regs& R, const Src& s);
inline void fill(uint32_t* t, uint32_t v, unsigned n) {
  for (unsigned i = 0; i < n; ++i) t[i] = v;
}
inline uint32_t mods32(const Src& s, uint32_t v) {
  if (s.abs) v &= 0x7FFFFFFFu;
  if (s.neg) v ^= 0x80000000u;
  return v;
}
// A 32-bit lane source as a row: the register's own, or a copy in `t` where a modifier changes it or the value is the same in every lane.
inline const uint32_t* load32(const Src& s, Regs& R, uint32_t* t, unsigned n) {
  switch (s.kind) {
    case kVgpr: {
      const uint32_t* p = R.vgpr[s.index];
      if (!(s.neg | s.abs)) return p;
      const uint32_t keep = s.abs ? 0x7FFFFFFFu : ~0u, flip = s.neg ? 0x80000000u : 0u;
      for (unsigned i = 0; i < n; ++i) t[i] = (p[i] & keep) ^ flip;
      return t;
    }
    case kConst: fill(t, mods32(s, s.value), n); return t;
    case kSgpr: fill(t, mods32(s, R.sgpr[s.index]), n); return t;
    default: fill(t, mods32(s, static_cast<uint32_t>(rd(R, s))), n); return t;   // (VCC and EXEC, read as scalar() does and cut to 32 bits)
  }
}
// A 64-bit lane source: a register pair, a constant or scalar register(s), with the modifiers on bit 63.
inline void load64(const Src& s, Regs& R, uint64_t* t, unsigned n) {
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

inline uint64_t rd(const Regs& R, const Src& s) {
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
inline void wr(Regs& R, const Src& s, uint64_t v) {
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

// ---- Vector handlers

inline unsigned lanes_of(uint64_t ex) { return (ex >> 32) ? 64 : 32; }

// NaN anywhere among the active lanes of a binary32 / binary64 result.
inline bool nan32(const uint32_t* r, uint64_t ex, unsigned n) {
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
inline bool nan64(const uint64_t* r, uint64_t ex, unsigned n) {
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
int h_int(const Op& o, Regs& R) {
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

// A binary32 operation: the result is thrown away if any active lane is a NaN.
int h_f32(const Op& o, Regs& R) {
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
  if (nan32(r, ex, n)) return false;
  if (o.omod | o.clamp) post32(r, ex, o.omod, o.clamp);
  uint32_t* d = R.vgpr[o.dst.index];
  if (full(ex, n)) std::memcpy(d, r, n * 4);
  else merge(d, r, ex, n);
  return true;
}

// binary64: operands and result are register pairs.
int h_f64(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  if (!ex) return true;
  const unsigned n = lanes_of(ex);
  alignas(32) uint64_t a[64], b[64], c[64], r[64];
  load64(o.src[0], R, a, n);
  if (o.src[1].kind) load64(o.src[1], R, b, n);
  if (o.src[2].kind) load64(o.src[2], R, c, n);
  reinterpret_cast<K64>(o.kern)(r, a, b, c, n);
  if (nan64(r, ex, n)) return false;
  if (o.omod | o.clamp) post64(r, ex, o.omod, o.clamp);
  uint32_t* lo = R.vgpr[o.dst.index];
  uint32_t* hi = R.vgpr[o.dst.index + 1];
  if (full(ex, n)) {
    for (unsigned i = 0; i < n; ++i) lo[i] = static_cast<uint32_t>(r[i]), hi[i] = static_cast<uint32_t>(r[i] >> 32);
  } else {
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      lo[l] = static_cast<uint32_t>(r[l]), hi[l] = static_cast<uint32_t>(r[l] >> 32);
    }
  }
  return true;
}

// A 64-bit integer operation: sources by `sub` bits (bit k set: source k is a 64-bit pair, else a 32-bit count).
int h_l64(const Op& o, Regs& R) {
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
  uint32_t* lo = R.vgpr[o.dst.index];
  uint32_t* hi = R.vgpr[o.dst.index + 1];
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    lo[l] = static_cast<uint32_t>(r[l]), hi[l] = static_cast<uint32_t>(r[l] >> 32);
  }
  return true;
}

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
  if (nan32(r[0], ex, n) || nan32(r[1], ex, n)) return false;
  if (o.clamp) post32(r[0], ex, 0, true), post32(r[1], ex, 0, true);
  uint32_t* d0 = R.vgpr[o.dst.index];
  uint32_t* d1 = R.vgpr[o.dst.index + 1];
  if (full(ex, n)) {
    std::memcpy(d0, r[0], n * 4);
    std::memcpy(d1, r[1], n * 4);
  } else {
    merge(d0, r[0], ex, n);
    merge(d1, r[1], ex, n);
  }
  return true;
}

// v_mad_u64_u32: a 32x32 product added to a 64-bit value; the carry out of 64 bits is the scalar destination's bit.
int h_mad_u64(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  uint64_t carry = 0;
  if (ex) {
    const unsigned n = lanes_of(ex);
    alignas(32) uint32_t t0[64], t1[64];
    alignas(32) uint64_t c[64];
    const uint32_t* a = load32(o.src[0], R, t0, n);
    const uint32_t* b = load32(o.src[1], R, t1, n);
    load64(o.src[2], R, c, n);
    uint32_t* lo = R.vgpr[o.dst.index];
    uint32_t* hi = R.vgpr[o.dst.index + 1];
    alignas(32) uint64_t r[64];
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      const unsigned __int128 p = static_cast<unsigned __int128>(a[l]) * b[l] + c[l];
      r[l] = static_cast<uint64_t>(p);
      if (static_cast<uint64_t>(p >> 64)) carry |= uint64_t{1} << l;
    }
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      lo[l] = static_cast<uint32_t>(r[l]), hi[l] = static_cast<uint32_t>(r[l] >> 32);
    }
  }
  if (o.dst2.kind) wr(R, o.dst2, carry);
  return true;
}

// v_add_co_u32, v_addc_co_u32, v_sub_co_u32, v_subb_co_u32 and the reversed ones (carry_alu): the carry or borrow of each lane into a
// scalar mask, written whatever EXEC is. `sub`: bit 0 add, bit 1 takes a carry in (source 2, a mask), bit 2 reversed.
int h_carry(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  const bool add = o.sub & 1, takes = o.sub & 2, rev = o.sub & 4;
  const uint64_t carried_in = takes ? rd(R, o.src[2]) : 0;
  uint64_t out = 0;
  if (ex) {
    const unsigned n = lanes_of(ex);
    alignas(32) uint32_t t0[64], t1[64], r[64];
    const uint32_t* a0 = load32(o.src[0], R, t0, n);
    const uint32_t* b0 = load32(o.src[1], R, t1, n);
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      uint64_t a = a0[l], b = b0[l];
      if (rev) std::swap(a, b);
      const uint64_t c = takes ? (carried_in >> l) & 1 : 0;
      const uint64_t v = add ? a + b + c : a - b - c;
      if (add ? (v >> 32) & 1 : a < b + c) out |= uint64_t{1} << l;
      r[l] = static_cast<uint32_t>(v);
    }
    uint32_t* d = R.vgpr[o.dst.index];
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      d[l] = r[l];
    }
  }
  wr(R, o.dst2, out);
  return 1;
}

// v_cndmask_b32: lane l takes source 1 where bit l of the scalar mask is set, else source 0 (modifiers on either).
int h_cndmask(const Op& o, Regs& R) {
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
  if (full(ex, n)) std::memcpy(d, r, n * 4);
  else merge(d, r, ex, n);
  return true;
}

// v_readfirstlane_b32: the first active lane's value (lane 0 with none) into a scalar destination.
int h_readfirstlane(const Op& o, Regs& R) {
  const uint64_t ex = *R.exec;
  const unsigned lane = ex ? static_cast<unsigned>(__builtin_ctzll(ex)) : 0;
  wr(R, o.dst, R.vgpr[o.src[0].index][lane]);
  return true;
}

// A comparison: one bit a lane, into a scalar destination and, for the X forms, EXEC. `kern` is the test for the first n lanes.
int h_cmp(const Op& o, Regs& R) {
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


int h_never(const Op&, Regs&) { return 0; }

// ---- LDS
enum DsKind : uint32_t { kDsRead, kDsWrite, kDsRead2, kDsWrite2, kDsBperm };
int h_ds(const Op& o, Regs& R) {
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
    for (uint64_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
      const uint32_t from = ((addr[l] + x) >> 2) & (R.lanes - 1);
      R.vgpr[o.dst.index][l] = ex >> from & 1 ? before[from] : 0;
    }
    return 1;
  }
  const uint32_t x0 = static_cast<uint32_t>(o.off0), x1 = static_cast<uint32_t>(o.off1);
  const unsigned w = o.words;
  const uint64_t size = R.lds_size;
  const uint64_t span = o.sub == kDsRead || o.sub == kDsWrite ? 4ull * w : 4;
  // Every lane must land inside the LDS (an access past it reads zeros and writes nowhere on a card, or is an error: the
  // interpreter's business); nothing has changed yet when one does not.
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    if (static_cast<uint64_t>(static_cast<uint32_t>(addr[l] + x0)) + span > size) return 0;
    if ((o.sub == kDsRead2 || o.sub == kDsWrite2) && static_cast<uint64_t>(static_cast<uint32_t>(addr[l] + x1)) + span > size) return 0;
  }
  uint8_t* lds = R.lds;
  switch (o.sub) {
    case kDsRead:
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        const uint8_t* p = lds + static_cast<uint32_t>(addr[l] + x0);
        for (unsigned k = 0; k < w; ++k) {
          uint32_t v;
          std::memcpy(&v, p + 4 * k, 4);
          R.vgpr[o.dst.index + k][l] = v;
        }
      }
      return 1;
    case kDsWrite:
      for (uint64_t m = ex; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
        uint8_t* p = lds + static_cast<uint32_t>(addr[l] + x0);
        for (unsigned k = 0; k < w; ++k) std::memcpy(p + 4 * k, &R.vgpr[o.src[1].index + k][l], 4);
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

// A 64-bit comparison (integers, or binary64 with the source modifiers): `kern` is the test for the first n lanes.
int h_cmp64(const Op& o, Regs& R) {
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

// Two sources, one destination, SCC from the result (where the instruction sets it).
#define TIER_S2(name, ...)                                 \
  int name(const Op& o, Regs& R) {                         \
    [[maybe_unused]] const uint64_t a = rd(R, o.src[0]), b = rd(R, o.src[1]); \
    uint64_t v = 0;                                        \
    __VA_ARGS__;                                           \
    wr(R, o.dst, v);                                       \
    return true;                                           \
  }
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

// SOPC: only SCC changes. `sub` is the test and the type.
enum ScalarCmp : uint32_t {
  kEqU32, kLgU32, kLtU32, kLeU32, kGtU32, kGeU32, kEqI32, kLgI32, kLtI32, kLeI32, kGtI32, kGeI32, kEqU64, kLgU64
};
int s_cmp(const Op& o, Regs& R) {
  const uint64_t a = rd(R, o.src[0]), b = rd(R, o.src[1]);
  const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
  const int32_t sx = static_cast<int32_t>(a), sy = static_cast<int32_t>(b);
  bool r = false;
  switch (o.sub) {
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

enum class Cat { None, Carry, Lane, Global, Ds, Pk, Int, F32, F64, L64, Mad64, Cmp, Cndmask, Readfirstlane, S2, ScalarCmp, Saveexec, Branch };

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
    I("v_bfe_u32", BFE_U, 3);
    I("v_med3_u32", MED3_U, 3);
    I("v_med3_i32", MED3_I, 3);
    I("v_min3_u32", MIN3_U, 3);
    I("v_max3_u32", MAX3_U, 3);
    I("v_mbcnt_lo_u32_b32", MBCNT_LO, 2);
    I("v_mbcnt_hi_u32_b32", MBCNT_HI, 2);
    I("v_cvt_i32_f32_e32", CVT_I32_F32, 1, kFloatSrc);
    I("v_cvt_u32_f32_e32", CVT_U32_F32, 1, kFloatSrc);
    I("v_cvt_f32_u32_e32", CVT_F32_U, 1);
    I("v_cvt_f32_i32_e32", CVT_F32_I, 1);
    I("v_cvt_f32_ubyte0_e32", CVT_F32_UBYTE0, 1);
    const auto F = [&](std::string_view n, int k, int arity, uint32_t fl = 0) { m[n] = {Cat::F32, k, arity, fl}; };
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
    const auto L = [&](std::string_view n, int k, int arity, uint32_t sub) { m[n] = {Cat::L64, k, arity, 0, nullptr, sub}; };
    L("v_mov_b64_e32", L_MOV, 1, 1);
    L("v_lshlrev_b64", L_LSHLREV, 2, 2);
    L("v_lshrrev_b64", L_LSHRREV, 2, 2);
    L("v_ashrrev_i64", L_ASHRREV, 2, 2);
    L("v_lshl_add_u64", L_LSHL_ADD, 3, 1 | 4);
    L("v_lshl_add_u64_g", L_LSHL_ADD_G, 3, 1 | 4);
    L("v_cvt_f64_u32_e32", L_CVT_F64_U, 1, 0);
    L("v_cvt_f64_i32_e32", L_CVT_F64_I, 1, 0);
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
    m["ds_read_b128"] = {Cat::Ds, kDsRead, 1, 0, nullptr, 4};
    m["ds_write_b32"] = {Cat::Ds, kDsWrite, 1, 0, nullptr, 1};
    m["ds_write_b64"] = {Cat::Ds, kDsWrite, 1, 0, nullptr, 2};
    m["ds_write_b128"] = {Cat::Ds, kDsWrite, 1, 0, nullptr, 4};
    m["ds_bpermute_b32"] = {Cat::Ds, kDsBperm, 1, 0, nullptr, 1};
    m["ds_read2st64_b32"] = {Cat::Ds, kDsRead2, 1, 1, nullptr, 2};
    m["ds_write2st64_b32"] = {Cat::Ds, kDsWrite2, 1, 1, nullptr, 2};
    m["ds_read2_b32"] = {Cat::Ds, kDsRead2, 1, 0, nullptr, 2};
    m["ds_write2_b32"] = {Cat::Ds, kDsWrite2, 1, 0, nullptr, 2};
    m["v_pk_add_f32"] = {Cat::Pk, FADD, 2};
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
  if (op >= 0x40 && op < 0x50) k = T.cmp_f32[op & 0xF], fmods = true;
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
    f.run = h_cmp64;
    return;
  } else return;
  if (!lane_src32(in.src[0], f.src[0], fmods) || !lane_src32(in.src[1], f.src[1], fmods)) return void(f = Op());
  if (!in.dst.empty() && !scalar_dst(in.dst[0], f.dst)) return void(f = Op());
  f.kern = reinterpret_cast<AnyFn>(k);
  f.sub = writes_exec;
  f.unit = kUnitValu;
  f.run = h_cmp;
}

}  // namespace

void tier_classify(Inst& in) {
  in.fast = Op();
  // Not modelled: gfx1250 (it has its own instruction interceptions), VOPD, sub-dword and cross-lane forms, 16-bit halves,
  // per-source op_sel.
  if (in.arch == Target::Gfx1250 || !in.dual.empty() || in.sdwa || in.dpp || in.dpp8 || (in.op_sel != 0 && in.enc != Enc::Vop3p) || in.bitop3 || in.scaled) return;
  const Tables& T = tables();
  Op& f = in.fast;
  const bool vector_enc = in.enc == Enc::Vop1 || in.enc == Enc::Vop2 || in.enc == Enc::Vop3 || in.enc == Enc::Vopc;
  if (in.flags & kInstCmp) {
    if (in.enc == Enc::Vopc || in.enc == Enc::Vop3) classify_cmp(in, T);
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
    if (it == tab.end()) return;
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
      // (CDNA3 and CDNA4 apply the clamp bit to the integer instructions that have one and to the conversions to float: the interpreter's.)
      if (in.clamp && (in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950) && (!is_f || (e.flags & kIgnoreMods))) return fail();
      // (gfx942 and gfx950 apply or ignore an output modifier by MODE and flush what it makes denormal, CDNA3 and CDNA4 guides, 6.5: the interpreter's.)
      if (in.omod && (in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950) && !(e.flags & kIgnoreMods)) return fail();
      for (int k = 0; k < e.arity; ++k)
        if (!lane_src32(in.src[k], f.src[k], is_f || (e.flags & kFloatSrc))) return fail();
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
        f.kern = reinterpret_cast<AnyFn>(T.f32[e.k]);
        f.run = h_f32;
      } else {
        f.kern = reinterpret_cast<AnyFn>(T.ints[e.k]);
        f.run = h_int;
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
      f.kern = reinterpret_cast<AnyFn>(T.f64[e.k]);
      f.run = h_f64;
      f.unit = kUnitValu;
      return;
    }
    case Cat::L64: {
      if (!vector_enc || in.dst.size() != 1 || in.src.size() < static_cast<size_t>(e.arity)) return;
      if (!vgpr_dst(in.dst[0], f.dst, 2)) return fail();
      for (int k = 0; k < e.arity; ++k)
        if (!((e.sub >> k & 1) ? lane_src64(in.src[k], f.src[k], false) : lane_src32(in.src[k], f.src[k], false))) return fail();
      f.sub = e.sub;
      f.kern = reinterpret_cast<AnyFn>(T.l64[e.k]);
      f.run = h_l64;
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
      f.run = h_carry;
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
      if (e.k == kDsBperm) {
        if (in.dst.size() != 1 || in.src.size() < 2) return;
        if (in.dst[0].kind != OperandKind::Vgpr || in.dst[0].index >= kVgprs || in.dst[0].width != 1) return;
        if (in.src[1].kind != OperandKind::Vgpr || in.src[1].index >= kVgprs) return;
        f.dst.kind = kVgpr, f.dst.index = in.dst[0].index;
        f.src[1].kind = kVgpr, f.src[1].index = in.src[1].index;
        f.run = h_ds;
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
          if (in.src[k].kind != OperandKind::Vgpr || in.src[k].index + e.sub > kVgprs || in.src[k].width != e.sub) return;
          f.src[k].kind = kVgpr, f.src[k].index = in.src[k].index;
        }
        f.store = true;
      }
      if (e.k == kDsRead2 || e.k == kDsWrite2) {
        // (Offsets in words: the instruction's two 8-bit counts.)
        const uint32_t unit = e.flags & 1 ? 4 * 64 : 4;   // (the st64 forms count in 64s of words)
        f.off0 = static_cast<int32_t>(static_cast<uint32_t>(in.offset) * unit);
        f.off1 = static_cast<int32_t>(static_cast<uint32_t>(in.offset1) * unit);
      }
      f.run = h_ds;
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
      f.kern = reinterpret_cast<AnyFn>(T.f32[e.k]);
      f.run = h_pk;
      f.unit = kUnitValu;
      return;
    }
    case Cat::Mad64: {
      if (!vector_enc || in.dst.empty() || in.dst.size() > 2 || in.src.size() < 3) return;
      if (!vgpr_dst(in.dst[0], f.dst, 2)) return fail();
      if (!lane_src32(in.src[0], f.src[0], false) || !lane_src32(in.src[1], f.src[1], false) || !lane_src64(in.src[2], f.src[2], false)) return fail();
      if (in.dst.size() == 2 && !scalar_dst(in.dst[1], f.dst2)) return fail();
      f.run = h_mad_u64;
      f.unit = kUnitValu;
      return;
    }
    case Cat::Cndmask: {
      if (!vector_enc || in.dst.size() != 1 || in.src.size() < 3) return;
      if (!vgpr_dst(in.dst[0], f.dst, 1)) return fail();
      if (!lane_src32(in.src[0], f.src[0], true) || !lane_src32(in.src[1], f.src[1], true) || !scalar_src(in.src[2], f.src[2])) return fail();
      f.run = h_cndmask;
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
      f.unit = kUnitSalu;
      return;
    }
    case Cat::ScalarCmp: {
      if (in.enc != Enc::Sopc || in.src.size() != 2) return;
      for (size_t k = 0; k < 2; ++k)
        if (!scalar_src(in.src[k], f.src[k])) return fail();
      f.sub = static_cast<uint32_t>(e.k);
      f.run = s_cmp;
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
