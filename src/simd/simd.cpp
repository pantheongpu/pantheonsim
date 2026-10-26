// Dispatch for the SIMD kernels (include/vgpu/simd/simd.hpp). The kernels are
// in kernels.inc, compiled once for the baseline and once under a target
// pragma.
#include "vgpu/exec/tc_ampere.hpp"
#include "vgpu/simd/simd.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <type_traits>
#include <vector>

#include "vgpu/simd/fp16.hpp"

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#include <immintrin.h>
#define VGPU_SIMD_X86 1
#endif
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

// A product the card rounds before the sum must not become an FMA in a build
// that has one (the AVX2 level has FMA): no contraction anywhere in this file.
#pragma GCC optimize("fp-contract=off")

namespace vgpu::simd {

#define VGPU_LANES_ONLY 0
#define VGPU_KN   // (attributes of the lane kernels: none here; the JIT forces them inline, src/sass/jit_rt.h)
#define VGPU_KA
#define VGPU_SIMD_NS base_impl
#define VGPU_SIMD_AVX2 0
#include "kernels.inc"
#undef VGPU_SIMD_NS
#undef VGPU_SIMD_AVX2

#ifdef VGPU_SIMD_X86
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#define VGPU_SIMD_NS avx2_impl
#define VGPU_SIMD_AVX2 1
#include "kernels.inc"
#undef VGPU_SIMD_NS
#undef VGPU_SIMD_AVX2
#pragma GCC pop_options

// The AVX-512 level: the tile kernels at 512 bits (kernels512.inc), under the target pragma for the units they use; the other entries of the
// Avx512 table are the Avx2 ones.
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")
namespace avx512_impl {
#include "kernels512.inc"
}  // namespace avx512_impl
#pragma GCC pop_options
#endif

namespace {

const Kernels kBase = {Level::Base,       base_impl::f16_to_f32_k, base_impl::f32_to_f16_k, base_impl::iadd3_k,
                       base_impl::imad_k, base_impl::lop3_k,       base_impl::ffma_k,       base_impl::fadd_k,
                       base_impl::fmul_k, base_impl::ffma_m_k,   base_impl::hfma2_m_k,     base_impl::icmp_mask_k, base_impl::ibin_m_k, base_impl::imad_m_k, base_impl::dop_m_k,
                       base_impl::f64_to_f16_canon_k, base_impl::hmma_k,
                       base_impl::blend_m_k, base_impl::iadd3x_k, base_impl::imad_wide_k, base_impl::shf_k,
                       base_impl::lane_unary_k,
                       base_impl::ffma_x_k, base_impl::fadd_x_k, base_impl::fmul_x_k, base_impl::fcmp_mask_k, base_impl::mufu_k,
                       base_impl::i2f_k, base_impl::f2i_k, base_impl::fmnmx_k,
                       base_impl::prmt0_k, base_impl::dcmp_k, base_impl::hfma2_bf_m_k,
                       base_impl::tile_f32_k, base_impl::tile_i32_k, base_impl::tile_f64_k};
#ifdef VGPU_SIMD_X86
// The AVX2 table with every entry that does not use F16C. The four that do (the two conversions, the tensor kernel and
// the f16 fma) are the baseline's here; kAvx2 below has them too.
const Kernels kAvx2NoF16c = {Level::Avx2,       base_impl::f16_to_f32_k, base_impl::f32_to_f16_k, avx2_impl::iadd3_k,
                             avx2_impl::imad_k, avx2_impl::lop3_k,       avx2_impl::ffma_k,       avx2_impl::fadd_k,
                             avx2_impl::fmul_k, avx2_impl::ffma_m_k,   base_impl::hfma2_m_k,     avx2_impl::icmp_mask_k,
                             avx2_impl::ibin_m_k, avx2_impl::imad_m_k, avx2_impl::dop_m_k,
                             avx2_impl::f64_to_f16_canon_k, base_impl::hmma_k,
                             avx2_impl::blend_m_k, avx2_impl::iadd3x_k, avx2_impl::imad_wide_k, avx2_impl::shf_k,
                       avx2_impl::lane_unary_k,
                       avx2_impl::ffma_x_k, avx2_impl::fadd_x_k, avx2_impl::fmul_x_k, avx2_impl::fcmp_mask_k, avx2_impl::mufu_k,
                       avx2_impl::i2f_k, avx2_impl::f2i_k, avx2_impl::fmnmx_k,
                       avx2_impl::prmt0_k, avx2_impl::dcmp_k, avx2_impl::hfma2_bf_m_k,
                       avx2_impl::tile_f32_k, avx2_impl::tile_i32_k, avx2_impl::tile_f64_k};
const Kernels kAvx2 = {Level::Avx2,       avx2_impl::f16_to_f32_k, avx2_impl::f32_to_f16_k, avx2_impl::iadd3_k,
                       avx2_impl::imad_k, avx2_impl::lop3_k,       avx2_impl::ffma_k,       avx2_impl::fadd_k,
                       avx2_impl::fmul_k, avx2_impl::ffma_m_k,   avx2_impl::hfma2_m_k,     avx2_impl::icmp_mask_k, avx2_impl::ibin_m_k, avx2_impl::imad_m_k, avx2_impl::dop_m_k,
                       avx2_impl::f64_to_f16_canon_k, avx2_impl::hmma_k,
                       avx2_impl::blend_m_k, avx2_impl::iadd3x_k, avx2_impl::imad_wide_k, avx2_impl::shf_k,
                       avx2_impl::lane_unary_k,
                       avx2_impl::ffma_x_k, avx2_impl::fadd_x_k, avx2_impl::fmul_x_k, avx2_impl::fcmp_mask_k, avx2_impl::mufu_k,
                       avx2_impl::i2f_k, avx2_impl::f2i_k, avx2_impl::fmnmx_k,
                       avx2_impl::prmt0_k, avx2_impl::dcmp_k, avx2_impl::hfma2_bf_m_k,
                       avx2_impl::tile_f32_k, avx2_impl::tile_i32_k, avx2_impl::tile_f64_k};

// The F16C conversions against the scalar ones: every half, and a spread of
// floats. A host whose conversion differs (a stray MXCSR, an emulator) does
// not get the F16C kernels -- the same check the PTX engine makes of its F16C
// path. About 0.4 ms, so it runs on the first use of an F16C kernel and not
// when the library loads.
bool avx2_conversions_agree() {
  for (uint32_t h = 0; h < 65536; h += 8) {
    uint16_t in[8];
    float got[8];
    for (uint32_t i = 0; i < 8; ++i) in[i] = static_cast<uint16_t>(h + i);
    kAvx2.f16_to_f32(in, got, 8);
    for (uint32_t i = 0; i < 8; ++i)
      if (std::bit_cast<uint32_t>(got[i]) != f16_to_f32_bits(in[i])) return false;
  }
  uint32_t x = 0x9E3779B9u;
  for (int blk = 0; blk < 4096; ++blk) {
    float in[8];
    uint16_t got[8];
    for (int i = 0; i < 8; ++i) {
      x = x * 1664525u + 1013904223u;
      in[i] = std::bit_cast<float>(blk < 2048 ? x : (x & 0x80000000u) | (0x30000000u + (x & 0x0FFFFFFFu)));
    }
    kAvx2.f32_to_f16(in, got, 8);
    for (int i = 0; i < 8; ++i)
      if (in[i] == in[i] && got[i] != f32_to_f16_bits(in[i])) return false;
  }
  return true;
}
#endif

bool cpu_has_avx2() {
#ifdef VGPU_SIMD_X86
  static const bool has = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
  }();
  return has;
#else
  return false;
#endif
}

Level choose() {
  Level best = cpu_has_avx2() ? Level::Avx2 : Level::Base;
  if (const char* v = std::getenv("VGPU_SIMD")) {
    if (std::strcmp(v, "base") == 0) best = Level::Base;
    else if (std::strcmp(v, "avx2") == 0 && best != Level::Avx2) best = Level::Base;   // asked for more than the host has
  }
  return best;
}

// The AVX-512 units the tile kernels use (F, BW, DQ, VL and VNNI), on a host that has the AVX2 level too.
bool cpu_has_avx512() {
#ifdef VGPU_SIMD_X86
  static const bool has = [] {
    __builtin_cpu_init();
    return cpu_has_avx2() && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
  }();
  return has;
#else
  return false;
#endif
}

// Whether the Avx512 table is the one in use: asked for with VGPU_SIMD=avx512, or by default where the 512-bit tile kernels are the faster ones
// (measured per host class; left off until a host is shown to gain: Zen 4 runs 512-bit operations as two 256-bit halves). Never when the lane kernels
// are capped to base or avx2.
constexpr bool kAvx512Default = false;
bool use_avx512() {
  static const bool use = [] {
    if (level() != Level::Avx2 || !cpu_has_avx512()) return false;
    const char* v = std::getenv("VGPU_SIMD");
    if (v) return std::strcmp(v, "avx512") == 0;
    return kAvx512Default;
  }();
  return use;
}

// 0: not checked yet, 2: F16C agrees with the scalar conversions, 3: it does not (1 is unused).
std::atomic<int> g_f16c_state{0};
std::atomic<int> g_f16c_checks{0};
std::mutex g_f16c_mutex;

#ifdef VGPU_SIMD_X86
// The table of an AVX2 host once the F16C check has been made.
const Kernels& avx2_final() { return g_f16c_state.load(std::memory_order_acquire) == 2 ? kAvx2 : kAvx2NoF16c; }

// Makes the F16C check if no thread has (exactly once, whatever the number of threads asking), and returns the state.
int resolve_f16c() {
  int s = g_f16c_state.load(std::memory_order_acquire);
  if (s >= 2) return s;
  std::lock_guard<std::mutex> lock(g_f16c_mutex);
  s = g_f16c_state.load(std::memory_order_relaxed);
  if (s < 2) {
    g_f16c_checks.fetch_add(1, std::memory_order_relaxed);
    s = avx2_conversions_agree() ? 2 : 3;
    g_f16c_state.store(s, std::memory_order_release);
  }
  return s;
}

// What the engines get on an AVX2 host until the check has been made: the same table except for the four F16C entries,
// which check first (once) and then run what the table of the checked state has. The first call replaces the table
// the engines read, so later calls go straight to the kernel.
extern const Kernels kAvx2Lazy, kAvx512Lazy;
// The checked table the engines get at the level in use, and the lazy one they start on.
const Kernels& avx512_final();
const Kernels& final_table() { return use_avx512() ? avx512_final() : avx2_final(); }
const Kernels& lazy_table() { return use_avx512() ? kAvx512Lazy : kAvx2Lazy; }
void publish(const Kernels* from) {
  const Kernels* expect = from;
  detail::g_kernels.compare_exchange_strong(expect, &final_table(), std::memory_order_acq_rel);
}
void lazy_f16_to_f32(const uint16_t* in, float* out, size_t n) {
  resolve_f16c();
  publish(&lazy_table());
  final_table().f16_to_f32(in, out, n);
}
void lazy_f32_to_f16(const float* in, uint16_t* out, size_t n) {
  resolve_f16c();
  publish(&lazy_table());
  final_table().f32_to_f16(in, out, n);
}
void lazy_hfma2_m(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n,
                  unsigned halves) {
  resolve_f16c();
  publish(&lazy_table());
  final_table().hfma2_m(d, a, b, c, mask, n, halves);
}
void lazy_hmma(unsigned kind, unsigned K, bool acc_f16, const uint32_t* a, const uint32_t* b, const uint32_t* c,
               uint32_t* out) {
  resolve_f16c();
  publish(&lazy_table());
  final_table().hmma(kind, K, acc_f16, a, b, c, out);
}
const Kernels kAvx2Lazy = {Level::Avx2,       lazy_f16_to_f32,   lazy_f32_to_f16,   avx2_impl::iadd3_k,
                           avx2_impl::imad_k, avx2_impl::lop3_k, avx2_impl::ffma_k, avx2_impl::fadd_k,
                           avx2_impl::fmul_k, avx2_impl::ffma_m_k, lazy_hfma2_m,    avx2_impl::icmp_mask_k,
                           avx2_impl::ibin_m_k, avx2_impl::imad_m_k, avx2_impl::dop_m_k,
                           avx2_impl::f64_to_f16_canon_k, lazy_hmma,
                           avx2_impl::blend_m_k, avx2_impl::iadd3x_k, avx2_impl::imad_wide_k, avx2_impl::shf_k,
                       avx2_impl::lane_unary_k,
                       avx2_impl::ffma_x_k, avx2_impl::fadd_x_k, avx2_impl::fmul_x_k, avx2_impl::fcmp_mask_k, avx2_impl::mufu_k,
                       avx2_impl::i2f_k, avx2_impl::f2i_k, avx2_impl::fmnmx_k,
                       avx2_impl::prmt0_k, avx2_impl::dcmp_k, avx2_impl::hfma2_bf_m_k,
                       avx2_impl::tile_f32_k, avx2_impl::tile_i32_k, avx2_impl::tile_f64_k};

// The Avx512 tables: each Avx2 one with the three tile kernels replaced by the 512-bit forms.
Kernels make_avx512(const Kernels& base) {
  Kernels k = base;
  k.level = Level::Avx512;
  k.tile_f32 = avx512_impl::tile_f32_k;
  k.tile_i32 = avx512_impl::tile_i32_k;
  k.tile_f64 = avx512_impl::tile_f64_k;
  return k;
}
const Kernels kAvx512NoF16c = make_avx512(kAvx2NoF16c), kAvx512 = make_avx512(kAvx2), kAvx512Lazy = make_avx512(kAvx2Lazy);
const Kernels& avx512_final() { return g_f16c_state.load(std::memory_order_acquire) == 2 ? kAvx512 : kAvx512NoF16c; }
#endif

}  // namespace

const char* level_name(Level l) { return l == Level::Avx512 ? "avx512" : l == Level::Avx2 ? "avx2" : "base"; }

bool supported(Level l) { return l == Level::Base || (l == Level::Avx2 && cpu_has_avx2()) || (l == Level::Avx512 && cpu_has_avx512()); }

Level level() {
  static const Level l = choose();
  return l;
}

const Kernels& kernels_for(Level l) {
#ifdef VGPU_SIMD_X86
  if (l == Level::Avx512 && cpu_has_avx512()) {
    resolve_f16c();
    return avx512_final();
  }
  if (l == Level::Avx2 && cpu_has_avx2()) {
    resolve_f16c();
    return avx2_final();
  }
#endif
  return kBase;
}

F16cState f16c_state() {
  const int s = g_f16c_state.load(std::memory_order_acquire);
  return s == 0 ? F16cState::Unchecked : s == 2 ? F16cState::Agrees : F16cState::Disagrees;
}

namespace testing {
void reset_f16c_check() {
#ifdef VGPU_SIMD_X86
  std::lock_guard<std::mutex> lock(g_f16c_mutex);
  g_f16c_state.store(0, std::memory_order_release);
  g_f16c_checks.store(0, std::memory_order_relaxed);
  if (level() == Level::Avx2) detail::g_kernels.store(&lazy_table(), std::memory_order_release);
#endif
}
int f16c_checks_run() { return g_f16c_checks.load(std::memory_order_relaxed); }
}  // namespace testing

namespace detail {
std::atomic<const Kernels*> g_kernels __attribute__((visibility("hidden"))){&kBase};
bool g_enabled __attribute__((visibility("hidden"))) = true;
}  // namespace detail

namespace {
// Runs when the library is loaded: from here on the engines get the level this host has (an AVX2 host starts on the
// table whose F16C entries make the one-time check; see kAvx2Lazy).
const bool g_init = [] {
#ifdef VGPU_SIMD_X86
  if (level() == Level::Avx2) detail::g_kernels.store(&lazy_table(), std::memory_order_release);
#endif
  const char* v = std::getenv("VGPU_SIMD");
  detail::g_enabled = !(v && std::strcmp(v, "off") == 0);
  return true;
}();
}  // namespace

}  // namespace vgpu::simd
