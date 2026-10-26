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

// memcount's sectors_window, four lanes to a step: each lane's first and last sector (the access's last byte is at most
// 31 on, so at most the next sector) become a bit of one 64-bit word, relative to 24 sectors before the first active
// lane's, and the sectors are the word's set bits. An inactive lane in a group of four takes the first active lane's
// address (the same sector again). The first lane's address below 2^62 keeps every sum from wrapping; a lane outside
// the window, or at 2^62 or above, gives the whole warp up.
namespace avx2_impl {
__attribute__((target("popcnt"))) bool sector_window_k(const uint64_t* addrs, uint64_t m, uint64_t span, uint64_t& out) {
  const unsigned lo = static_cast<unsigned>(std::countr_zero(m));
  const unsigned hi = 64 - static_cast<unsigned>(std::countl_zero(m));
  const uint64_t a0 = addrs[lo];
  const __m256i va0 = _mm256_set1_epi64x(static_cast<long long>(a0));
  const __m256i extra = _mm256_set1_epi64x(static_cast<long long>(span - 1));
  const __m256i base = _mm256_set1_epi64x(static_cast<long long>((a0 >> 5) - 24));
  const __m256i c62 = _mm256_set1_epi64x(62);
  const __m256i one = _mm256_set1_epi64x(1);
  const __m256i lane_bit = _mm256_set_epi64x(8, 4, 2, 1);
  const __m256i zero = _mm256_setzero_si256();
  __m256i bits = zero, bad = zero, big = zero;
  for (unsigned i = lo & ~3u; i < hi; i += 4) {
    __m256i a;
    if (i + 4 <= hi) {
      a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(addrs + i));
    } else {   // (the last group reads only the lanes up to the highest active one: the array may end there)
      uint64_t tail[4] = {a0, a0, a0, a0};
      for (unsigned j = 0; i + j < hi; ++j) tail[j] = addrs[i + j];
      a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tail));
    }
    const unsigned nib = static_cast<unsigned>(m >> i) & 15u;
    if (nib != 15u) {
      const __m256i on = _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_set1_epi64x(nib), lane_bit), lane_bit);
      a = _mm256_blendv_epi8(va0, a, on);
    }
    big = _mm256_or_si256(big, a);
    const __m256i f = _mm256_srli_epi64(a, 5);
    const __m256i e = _mm256_srli_epi64(_mm256_add_epi64(a, extra), 5);
    const __m256i d = _mm256_sub_epi64(f, base);
    bad = _mm256_or_si256(bad, _mm256_or_si256(_mm256_cmpgt_epi64(d, c62), _mm256_cmpgt_epi64(zero, d)));
    const __m256i bit = _mm256_sllv_epi64(one, d);
    bits = _mm256_or_si256(bits, _mm256_or_si256(bit, _mm256_sllv_epi64(bit, _mm256_sub_epi64(e, f))));
  }
  if (!_mm256_testz_si256(bad, bad) ||
      !_mm256_testz_si256(big, _mm256_set1_epi64x(static_cast<long long>(0xC000'0000'0000'0000ull))))
    return false;
  const __m128i h = _mm_or_si128(_mm256_castsi256_si128(bits), _mm256_extracti128_si256(bits, 1));
  const uint64_t w = static_cast<uint64_t>(_mm_cvtsi128_si64(h)) | static_cast<uint64_t>(_mm_extract_epi64(h, 1));
  out = static_cast<uint64_t>(__builtin_popcountll(w));
  return true;
}

// The same four lanes at a time (a 32-lane warp: eight steps), on a host with AVX2 at the level in use (VGPU_SIMD=base
// and off keep the loop above): the fused multiply-add is correctly rounded, so a vector fma gives each lane's bits. A
// group with a NaN result -- whose operands may be NaNs, which the rule above picks from -- is run lane by lane, and
// an inactive lane is read (the arrays hold all lanes) but never written.
void fma_lanes_f64_k(uint64_t* d, const uint64_t* a, const uint64_t* b,
                                                            const uint64_t* c, uint64_t m, uint32_t width, unsigned sub) {
  const __m256i lane_bit = _mm256_set_epi64x(8, 4, 2, 1);
  // A contracted subtract negates the product (bit 0) or the addend (bit 1): the general path multiplies by -1.0, which
  // flips the sign bit of any value that is not a NaN, and a NaN lane is run lane by lane below.
  const __m256d sign = _mm256_set1_pd(-0.0);
  const __m256d nega = sub & 1 ? sign : _mm256_setzero_pd(), negc = sub & 2 ? sign : _mm256_setzero_pd();
  const double sa = sub & 1 ? -1.0 : 1.0, sc = sub & 2 ? -1.0 : 1.0;
  if (width == 32 && (m & 0xFFFFFFFFull) == 0xFFFFFFFFull) {
    // A full warp: the eight results are made first and checked for a NaN together, and stored (after every operand is
    // read, so a destination that is also a source is fine) when there is none.
    __m256d r[8];
    __m256d nan = _mm256_setzero_pd();
#pragma GCC unroll 8
    for (uint32_t i = 0; i < 8; ++i) {
      r[i] = _mm256_fmadd_pd(_mm256_xor_pd(_mm256_loadu_pd(reinterpret_cast<const double*>(a + 4 * i)), nega),
                             _mm256_loadu_pd(reinterpret_cast<const double*>(b + 4 * i)),
                             _mm256_xor_pd(_mm256_loadu_pd(reinterpret_cast<const double*>(c + 4 * i)), negc));
      nan = _mm256_or_pd(nan, _mm256_cmp_pd(r[i], r[i], _CMP_UNORD_Q));
    }
    if (!_mm256_movemask_pd(nan)) {
#pragma GCC unroll 8
      for (uint32_t i = 0; i < 8; ++i) _mm256_storeu_pd(reinterpret_cast<double*>(d + 4 * i), r[i]);
      return;
    }
  }
  for (uint32_t l = 0; l < width; l += 4) {
    const unsigned nib = static_cast<unsigned>(m >> l) & 15u;
    if (!nib) continue;
    const __m256d r = _mm256_fmadd_pd(_mm256_xor_pd(_mm256_loadu_pd(reinterpret_cast<const double*>(a + l)), nega),
                                      _mm256_loadu_pd(reinterpret_cast<const double*>(b + l)),
                                      _mm256_xor_pd(_mm256_loadu_pd(reinterpret_cast<const double*>(c + l)), negc));
    if (_mm256_movemask_pd(_mm256_cmp_pd(r, r, _CMP_UNORD_Q))) {
      for (uint32_t k = l; k < l + 4; ++k) {
        if (!(m >> k & 1)) continue;
        double x, y, z;
        std::memcpy(&x, &a[k], 8);
        std::memcpy(&y, &b[k], 8);
        std::memcpy(&z, &c[k], 8);
        const double v = __builtin_fma(sa * x, y, sc * z);
        std::memcpy(&d[k], &v, 8);
        if (v != v && (x != x || y != y || z != z)) d[k] = (y != y ? b[k] : z != z ? c[k] : a[k]) | 0x0008000000000000ull;
      }
    } else if (nib == 15u) {
      _mm256_storeu_pd(reinterpret_cast<double*>(d + l), r);
    } else {
      const __m256i on = _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_set1_epi64x(nib), lane_bit), lane_bit);
      _mm256_maskstore_pd(reinterpret_cast<double*>(d + l), on, r);
    }
  }
}
// f64 add (op 0), sub (1), mul (2) or div (3) of every lane, in a row of 32 results, four lanes at a time: the host's
// operation is correctly rounded either way, so the bits are the scalar loop's. Returns whether any lane's result is a NaN
// (an inactive lane's too: its operands are whatever the register holds).
bool f64_bin_lanes_k(unsigned op, uint64_t* r, const uint64_t* a, const uint64_t* b,
                                                        uint32_t width) {
  __m256d nan = _mm256_setzero_pd();
  for (uint32_t l = 0; l < width; l += 4) {
    const __m256d x = _mm256_loadu_pd(reinterpret_cast<const double*>(a + l));
    const __m256d y = _mm256_loadu_pd(reinterpret_cast<const double*>(b + l));
    const __m256d v = op == 0 ? _mm256_add_pd(x, y) : op == 1 ? _mm256_sub_pd(x, y) : op == 2 ? _mm256_mul_pd(x, y) : _mm256_div_pd(x, y);
    nan = _mm256_or_pd(nan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
    _mm256_storeu_pd(reinterpret_cast<double*>(r + l), v);
  }
  return _mm256_movemask_pd(nan) != 0;
}
}  // namespace avx2_impl
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
extern const Kernels kAvx2Lazy;
void publish(const Kernels* from) {
  const Kernels* expect = from;
  detail::g_kernels.compare_exchange_strong(expect, &avx2_final(), std::memory_order_acq_rel);
}
void lazy_f16_to_f32(const uint16_t* in, float* out, size_t n) {
  resolve_f16c();
  publish(&kAvx2Lazy);
  avx2_final().f16_to_f32(in, out, n);
}
void lazy_f32_to_f16(const float* in, uint16_t* out, size_t n) {
  resolve_f16c();
  publish(&kAvx2Lazy);
  avx2_final().f32_to_f16(in, out, n);
}
void lazy_hfma2_m(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n,
                  unsigned halves) {
  resolve_f16c();
  publish(&kAvx2Lazy);
  avx2_final().hfma2_m(d, a, b, c, mask, n, halves);
}
void lazy_hmma(unsigned kind, unsigned K, bool acc_f16, const uint32_t* a, const uint32_t* b, const uint32_t* c,
               uint32_t* out) {
  resolve_f16c();
  publish(&kAvx2Lazy);
  avx2_final().hmma(kind, K, acc_f16, a, b, c, out);
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
#endif

}  // namespace

const char* level_name(Level l) { return l == Level::Avx2 ? "avx2" : "base"; }

bool supported(Level l) { return l == Level::Base || (l == Level::Avx2 && cpu_has_avx2()); }

Level level() {
  static const Level l = choose();
  return l;
}

const Kernels& kernels_for(Level l) {
#ifdef VGPU_SIMD_X86
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
  if (level() == Level::Avx2) detail::g_kernels.store(&kAvx2Lazy, std::memory_order_release);
#endif
}
int f16c_checks_run() { return g_f16c_checks.load(std::memory_order_relaxed); }
}  // namespace testing

namespace detail {
std::atomic<const Kernels*> g_kernels __attribute__((visibility("hidden"))){&kBase};
bool g_enabled __attribute__((visibility("hidden"))) = true;
SectorWindowFn g_sector_window __attribute__((visibility("hidden"))) = nullptr;
}  // namespace detail

bool avx2_f64_available() {
#ifdef VGPU_SIMD_X86
  return level() == Level::Avx2;
#else
  return false;
#endif
}
#ifdef VGPU_SIMD_X86
void fma_lanes_f64_avx2(uint64_t* d, const uint64_t* a, const uint64_t* b, const uint64_t* c, uint64_t m, unsigned sub) {
  avx2_impl::fma_lanes_f64_k(d, a, b, c, m, 32, sub);
}
bool f64_bin_lanes_avx2(unsigned op, uint64_t* r, const uint64_t* a, const uint64_t* b) {
  return avx2_impl::f64_bin_lanes_k(op, r, a, b, 32);
}
#else
void fma_lanes_f64_avx2(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t, unsigned) {}
bool f64_bin_lanes_avx2(unsigned, uint64_t*, const uint64_t*, const uint64_t*) { return false; }
#endif

namespace testing {
void use_sector_window_kernel(bool on) {
#ifdef VGPU_SIMD_X86
  detail::g_sector_window = on && level() == Level::Avx2 ? avx2_impl::sector_window_k : nullptr;
#else
  (void)on;
#endif
}
}  // namespace testing

namespace {
// Runs when the library is loaded: from here on the engines get the level this host has (an AVX2 host starts on the
// table whose F16C entries make the one-time check; see kAvx2Lazy).
const bool g_init = [] {
#ifdef VGPU_SIMD_X86
  if (level() == Level::Avx2) detail::g_kernels.store(&kAvx2Lazy, std::memory_order_release);
  testing::use_sector_window_kernel(true);
#endif
  const char* v = std::getenv("VGPU_SIMD");
  detail::g_enabled = !(v && std::strcmp(v, "off") == 0);
  return true;
}();
}  // namespace

}  // namespace vgpu::simd
