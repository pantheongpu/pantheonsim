#include "vgpu/exec/tile_mma.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <type_traits>
#include <vector>

#include "vgpu/exec/numerics.hpp"
#include "vgpu/simd/fp16.hpp"
#include "vgpu/simd/simd.hpp"

// The operand code of the less common tensor forms is kept out of the hot text (the interpreters' own loops are placement-sensitive: round 2).
#define VGPU_COLD __attribute__((noinline, section(".text.unlikely")))

namespace vgpu::exec {

typedef uint32_t V4u __attribute__((vector_size(16)));
typedef int32_t V4i __attribute__((vector_size(16)));
typedef float V4f __attribute__((vector_size(16)));

namespace {

// Every fp8 code as the float the scalar decoders' double narrows to (exact: both formats' values are floats).
const std::array<float, 256>& fp8_floats(bool e5m2) {
  static const std::array<float, 256> e4 = [] {
    std::array<float, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) t[i] = static_cast<float>(fp8_value(i, false));
    return t;
  }();
  static const std::array<float, 256> e5 = [] {
    std::array<float, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) t[i] = static_cast<float>(fp8_value(i, true));
    return t;
  }();
  return e5m2 ? e5 : e4;
}

// The matrix as raw elements, row-major [mn][k]. One descriptor lookup per 16 bytes of a row (or of a column when MN-major): a swizzle
// moves 16-byte units, so the bytes within one stay together.
bool read_raw(const uint8_t* smem, size_t size, const WgmmaDesc& d, bool k_major, uint32_t eb, uint32_t rows, uint32_t K, uint64_t off_mask,
              uint8_t* raw) {
  const uint32_t run = 16 / eb;
  const bool runs = !d.lbo_abs;
  for (uint32_t mn = 0; mn < rows; ++mn) {
    if (k_major) {
      for (uint32_t k = 0; k < K;) {
        const uint32_t n = runs && k % run == 0 && k + run <= K ? run : 1;
        const uint64_t at = wgmma_smem_offset(d, true, eb, mn, k) & off_mask;
        if (at + uint64_t{n} * eb > size) return false;
        std::memcpy(raw + (size_t{mn} * K + k) * eb, smem + at, size_t{n} * eb);
        k += n;
      }
    } else {
      // MN-major: the elements of a run are consecutive rows of one column.
      if (runs && mn % run == 0 && mn + run <= rows) {
        for (uint32_t k = 0; k < K; ++k) {
          const uint64_t at = wgmma_smem_offset(d, false, eb, mn, k) & off_mask;
          if (at + uint64_t{run} * eb > size) return false;
          for (uint32_t e = 0; e < run; ++e) std::memcpy(raw + (size_t{mn + e} * K + k) * eb, smem + at + size_t{e} * eb, eb);
        }
        mn += run - 1;
      } else {
        for (uint32_t k = 0; k < K; ++k) {
          const uint64_t at = wgmma_smem_offset(d, false, eb, mn, k) & off_mask;
          if (at + eb > size) return false;
          std::memcpy(raw + (size_t{mn} * K + k) * eb, smem + at, eb);
        }
      }
    }
  }
  return true;
}

}  // namespace

float tile_decode_f32(TileElem t, uint32_t raw) {
  switch (t) {
    case TileElem::F16: return simd::f16_to_f32(static_cast<uint16_t>(raw));
    case TileElem::BF16: return std::bit_cast<float>((raw & 0xffffu) << 16);
    case TileElem::TF32: return std::bit_cast<float>(raw & 0xffffe000u);
    case TileElem::E4M3: return fp8_floats(false)[raw & 0xff];
    case TileElem::E5M2: return fp8_floats(true)[raw & 0xff];
    default: return static_cast<float>(tile_decode_i32(t, raw));
  }
}

bool wgmma_read_f32(const uint8_t* smem, size_t size, const WgmmaDesc& d, bool k_major, TileElem t, uint32_t rows, uint32_t K, bool negate,
                    bool transpose, uint64_t off_mask, float* out) {
  const uint32_t eb = tile_elem_bytes(t);
  const size_t n = size_t{rows} * K;
  thread_local std::vector<uint8_t> raw;
  thread_local std::vector<float> dec;
  raw.resize(n * eb);
  dec.resize(n);
  if (!read_raw(smem, size, d, k_major, eb, rows, K, off_mask, raw.data())) return false;
  switch (t) {
    case TileElem::F16: {
      // (n is a multiple of 8 for every wgmma shape; the kernel takes any count)
      thread_local std::vector<uint16_t> h;
      h.resize(n);
      std::memcpy(h.data(), raw.data(), n * 2);
      simd::kernels().f16_to_f32(h.data(), dec.data(), n);
      break;
    }
    case TileElem::BF16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t h;
        std::memcpy(&h, &raw[2 * i], 2);
        dec[i] = std::bit_cast<float>(uint32_t{h} << 16);
      }
      break;
    case TileElem::TF32:
      for (size_t i = 0; i < n; ++i) {
        uint32_t w;
        std::memcpy(&w, &raw[4 * i], 4);
        dec[i] = std::bit_cast<float>(w & 0xffffe000u);
      }
      break;
    case TileElem::E4M3:
    case TileElem::E5M2: {
      const auto& tab = fp8_floats(t == TileElem::E5M2);
      for (size_t i = 0; i < n; ++i) dec[i] = tab[raw[i]];
      break;
    }
    default:
      for (size_t i = 0; i < n; ++i) dec[i] = static_cast<float>(tile_decode_i32(t, raw[i]));
  }
  const uint32_t flip = negate ? 0x80000000u : 0;
  if (transpose) {
    for (uint32_t mn = 0; mn < rows; ++mn)
      for (uint32_t k = 0; k < K; ++k) out[size_t{k} * rows + mn] = std::bit_cast<float>(std::bit_cast<uint32_t>(dec[size_t{mn} * K + k]) ^ flip);
  } else {
    for (size_t i = 0; i < n; ++i) out[i] = std::bit_cast<float>(std::bit_cast<uint32_t>(dec[i]) ^ flip);
  }
  return true;
}

bool wgmma_read_i32(const uint8_t* smem, size_t size, const WgmmaDesc& d, bool k_major, TileElem t, uint32_t rows, uint32_t K, bool negate,
                    bool transpose, uint64_t off_mask, int32_t* out) {
  const uint32_t eb = tile_elem_bytes(t);
  const size_t n = size_t{rows} * K;
  thread_local std::vector<uint8_t> raw;
  raw.resize(n * eb);
  if (!read_raw(smem, size, d, k_major, eb, rows, K, off_mask, raw.data())) return false;
  const auto at = [&](size_t i) {
    const int32_t v = tile_decode_i32(t, raw[i]);
    return negate ? -v : v;
  };
  if (transpose) {
    for (uint32_t mn = 0; mn < rows; ++mn)
      for (uint32_t k = 0; k < K; ++k) out[size_t{k} * rows + mn] = at(size_t{mn} * K + k);
  } else {
    for (size_t i = 0; i < n; ++i) out[i] = at(i);
  }
  return true;
}

namespace {

// 32-bit words: register 4j + 2h + u, lane 4g + t  <->  row g + 8h, column 8j + 2t + u (the two registers of a pair interleave).
void words_to_tile(const uint32_t* regs, uint32_t N, uint32_t* tile) {
  for (uint32_t j = 0; j < N / 8; ++j)
    for (uint32_t h = 0; h < 2; ++h) {
      const uint32_t* x = regs + size_t{4 * j + 2 * h} * 32;
      const uint32_t* y = x + 32;
      for (uint32_t g = 0; g < 8; ++g) {
        V4u xv, yv;
        std::memcpy(&xv, x + 4 * g, sizeof xv);
        std::memcpy(&yv, y + 4 * g, sizeof yv);
        const V4u lo = __builtin_shuffle(xv, yv, (V4i){0, 4, 1, 5}), hi = __builtin_shuffle(xv, yv, (V4i){2, 6, 3, 7});
        std::memcpy(tile + size_t{g + 8 * h} * N + 8 * j, &lo, sizeof lo);
        std::memcpy(tile + size_t{g + 8 * h} * N + 8 * j + 4, &hi, sizeof hi);
      }
    }
}

template <bool CanonNan>
void tile_to_words(const uint32_t* tile, uint32_t N, uint32_t* regs) {
  for (uint32_t j = 0; j < N / 8; ++j)
    for (uint32_t h = 0; h < 2; ++h) {
      uint32_t* x = regs + size_t{4 * j + 2 * h} * 32;
      uint32_t* y = x + 32;
      for (uint32_t g = 0; g < 8; ++g) {
        V4u lo, hi;
        std::memcpy(&lo, tile + size_t{g + 8 * h} * N + 8 * j, sizeof lo);
        std::memcpy(&hi, tile + size_t{g + 8 * h} * N + 8 * j + 4, sizeof hi);
        V4u e = __builtin_shuffle(lo, hi, (V4i){0, 2, 4, 6}), o = __builtin_shuffle(lo, hi, (V4i){1, 3, 5, 7});
        if (CanonNan) {
          // (the float compare is false for a NaN only)
          const V4u me = reinterpret_cast<V4u>(reinterpret_cast<V4f>(e) != reinterpret_cast<V4f>(e)),
                    mo = reinterpret_cast<V4u>(reinterpret_cast<V4f>(o) != reinterpret_cast<V4f>(o));
          e = (e & ~me) | (0x7fffffffu & me);
          o = (o & ~mo) | (0x7fffffffu & mo);
        }
        std::memcpy(x + 4 * g, &e, sizeof e);
        std::memcpy(y + 4 * g, &o, sizeof o);
      }
    }
}

}  // namespace

void wgmma_acc_to_tile(const uint32_t* regs, uint32_t N, bool f16, float* tile) {
  if (!f16) {
    words_to_tile(regs, N, reinterpret_cast<uint32_t*>(tile));
    return;
  }
  // Register 2j + h: its 64 halves, in memory order, are rows g of row group h, eight columns each.
  alignas(32) float t[64];
  for (uint32_t j = 0; j < N / 8; ++j)
    for (uint32_t h = 0; h < 2; ++h) {
      uint16_t halves[64];
      std::memcpy(halves, regs + size_t{2 * j + h} * 32, sizeof halves);
      simd::kernels().f16_to_f32(halves, t, 64);
      for (uint32_t g = 0; g < 8; ++g) std::memcpy(tile + size_t{g + 8 * h} * N + 8 * j, t + 8 * g, 8 * sizeof(float));
    }
}

void wgmma_tile_to_acc(const float* tile, uint32_t N, bool f16, uint32_t* regs, bool canonical) {
  if (!f16) {
    if (canonical) tile_to_words<false>(reinterpret_cast<const uint32_t*>(tile), N, regs);
    else tile_to_words<true>(reinterpret_cast<const uint32_t*>(tile), N, regs);
    return;
  }
  alignas(32) float t[64];
  for (uint32_t j = 0; j < N / 8; ++j)
    for (uint32_t h = 0; h < 2; ++h) {
      for (uint32_t g = 0; g < 8; ++g) std::memcpy(t + 8 * g, tile + size_t{g + 8 * h} * N + 8 * j, 8 * sizeof(float));
      uint16_t halves[64];
      simd::kernels().f32_to_f16(t, halves, 64);
      for (uint32_t i = 0; i < 64; ++i)
        if (t[i] != t[i]) halves[i] = 0x7fff;   // the canonical NaN, whatever the payload
      std::memcpy(regs + size_t{2 * j + h} * 32, halves, sizeof halves);
    }
}

void wgmma_acc_to_tile_i32(const uint32_t* regs, uint32_t N, int32_t* tile) {
  words_to_tile(regs, N, reinterpret_cast<uint32_t*>(tile));
}
void wgmma_tile_to_acc_i32(const int32_t* tile, uint32_t N, uint32_t* regs) {
  tile_to_words<false>(reinterpret_cast<const uint32_t*>(tile), N, regs);
}

namespace {

// ---- mma fragments, one instantiation per element type -----------------------------------------------------------------------
//
// The layouts (tile_mma.hpp) make a dense fragment a permutation of 16-byte runs: register r of A holds, in the four lanes 4g .. 4g + 3, the 16 bytes
// of row g + 8 (r % 2), at byte 16 (r / 2) of the row; B's four lanes hold 16 bytes of column g, at byte 16 r. So the dense forms copy whole runs
// into a raw tile and decode it in one pass per type (a bulk conversion, no switch per element), and B is transposed from its columns. The sparse A
// places each lane's stored elements by their metadata nibble, with every constant of the type known at compile time.

template <FragType T> struct FragOf {
  static constexpr unsigned bits = T == FragType::F16 || T == FragType::BF16 ? 16 : T == FragType::TF32 ? 32 : T == FragType::S4 || T == FragType::U4 ? 4 : 8;
  static constexpr unsigned p = 32 / bits;   // elements a register
  static constexpr bool is_int = T >= FragType::S8;
  // The raw tile's element: whole bytes, halves or words (4-bit elements stay packed, two to a byte).
  using Raw = std::conditional_t<bits == 16, uint16_t, std::conditional_t<bits == 32, uint32_t, uint8_t>>;
  using Out = std::conditional_t<is_int, int32_t, float>;
};

// The value of one element, `raw` holding it in its low bits (what the scalar decoders take).
template <FragType T> inline typename FragOf<T>::Out decode_one(const float* fp8tab, uint32_t raw) {
  switch (T) {
    case FragType::F16: return simd::f16_to_f32(static_cast<uint16_t>(raw));
    case FragType::BF16: return std::bit_cast<float>((raw & 0xffffu) << 16);
    case FragType::TF32: return std::bit_cast<float>(raw & 0xffffe000u);
    case FragType::E4M3:
    case FragType::E5M2: return fp8tab[raw & 0xff];
    case FragType::S8: return static_cast<int32_t>(static_cast<int8_t>(raw));
    case FragType::U8: return static_cast<int32_t>(static_cast<uint8_t>(raw));
    case FragType::S4: return static_cast<int32_t>((raw & 0xfu) << 28) >> 28;
    default: return static_cast<int32_t>(raw & 0xfu);
  }
}

// n packed elements -> n values, in order.
template <FragType T> VGPU_COLD void decode_bulk(const typename FragOf<T>::Raw* raw, typename FragOf<T>::Out* out, size_t n) {
  using F = FragOf<T>;
  if constexpr (T == FragType::F16) {
    simd::kernels().f16_to_f32(raw, out, n);
  } else if constexpr (T == FragType::BF16) {
    for (size_t i = 0; i < n; ++i) out[i] = std::bit_cast<float>(uint32_t{raw[i]} << 16);
  } else if constexpr (T == FragType::TF32) {
    for (size_t i = 0; i < n; ++i) out[i] = std::bit_cast<float>(raw[i] & 0xffffe000u);
  } else if constexpr (T == FragType::E4M3 || T == FragType::E5M2) {
    const float* tab = fp8_floats(T == FragType::E5M2).data();
    for (size_t i = 0; i < n; ++i) out[i] = tab[raw[i]];
  } else if constexpr (F::bits == 8) {
    for (size_t i = 0; i < n; ++i) out[i] = decode_one<T>(nullptr, raw[i]);
  } else {
    for (size_t i = 0; i < n; i += 2) {
      out[i] = decode_one<T>(nullptr, raw[i / 2]);
      out[i + 1] = decode_one<T>(nullptr, raw[i / 2] >> 4);
    }
  }
}

// Bytes of raw tile the dense forms need at most (s4 m16n8k64: 1024 elements; f16 m16n8k16: 256 of 2 bytes; tf32 and fp8 less).
constexpr size_t kRawBytes = 4096;

template <class Out> inline Out negated(Out v) {
  if constexpr (std::is_same_v<Out, float>) return std::bit_cast<float>(std::bit_cast<uint32_t>(v) ^ 0x80000000u);
  else return -v;
}

template <FragType T> VGPU_COLD void frag_a_dense(unsigned M, unsigned K, const uint32_t* regs, typename FragOf<T>::Out* out, bool negate) {
  using F = FragOf<T>;
  const unsigned kbytes = K * F::bits / 8, nregs = M * K / (32 * F::p);
  alignas(32) typename F::Raw raw[kRawBytes / sizeof(typename F::Raw)];
  if (size_t{M} * kbytes > sizeof raw) __builtin_trap();   // (the callers' shapes are at most 16 x 64 elements)
  uint8_t* const bytes = reinterpret_cast<uint8_t*>(raw);
  for (unsigned r = 0; r < nregs; ++r)
    for (unsigned g = 0; g < 8; ++g) std::memcpy(bytes + size_t{g + 8 * (r % 2)} * kbytes + size_t{16} * (r / 2), regs + size_t{r} * 32 + 4 * g, 16);
  decode_bulk<T>(raw, out, size_t{M} * K);
  if (negate)
    for (size_t i = 0; i < size_t{M} * K; ++i) out[i] = negated(out[i]);
}

template <FragType T> VGPU_COLD void frag_b_dense(unsigned K, const uint32_t* regs, typename FragOf<T>::Out* out) {
  using F = FragOf<T>;
  const unsigned nregs = K * 8 / (32 * F::p);
  alignas(32) typename F::Raw raw[kRawBytes / sizeof(typename F::Raw)];
  alignas(32) typename F::Out col[1024];   // B^T: 8 columns of K
  if (size_t{K} * 8 * F::bits / 8 > sizeof raw || K * 8 > 1024) __builtin_trap();
  uint8_t* const bytes = reinterpret_cast<uint8_t*>(raw);
  const unsigned kbytes = K * F::bits / 8;
  for (unsigned r = 0; r < nregs; ++r)
    for (unsigned g = 0; g < 8; ++g) std::memcpy(bytes + size_t{g} * kbytes + size_t{16} * r, regs + size_t{r} * 32 + 4 * g, 16);
  decode_bulk<T>(raw, col, size_t{K} * 8);
  for (unsigned g = 0; g < 8; ++g)
    for (unsigned k = 0; k < K; ++k) out[size_t{k} * 8 + g] = col[size_t{g} * K + k];
}

template <FragType T> VGPU_COLD void frag_a_sparse(unsigned M, unsigned K, const uint32_t* regs, const uint32_t* meta, unsigned sel, typename FragOf<T>::Out* out, bool negate) {
  using F = FragOf<T>;
  constexpr unsigned bits = F::bits, p = F::p;
  constexpr bool tf32 = T == FragType::TF32, int4 = bits == 4, narrow = bits <= 8;
  constexpr unsigned chunk_elems = tf32 ? 2 : int4 ? 8 : 4, stored = chunk_elems / 2, cpr = p / stored;
  const unsigned nregs = M * K / 2 / (32 * p);
  // Which lane of a group of four holds a row's chunk's metadata (mma.sp, as an RTX 3060 places it; exec::mma_sparse_column): H lanes hold a group's
  // metadata for the 16- and 32-bit types.
  const unsigned row_bits = tf32 ? 2 * K : int4 ? K / 2 : K, holders = 2 * row_bits / 32;
  const float* tab = fp8_floats(T == FragType::E5M2).data();
  std::fill(out, out + size_t{M} * K, typename F::Out{});
  for (unsigned r = 0; r < nregs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, tt = l % 4, half = r % 2, row = g + 8 * half;
      const uint32_t v = regs[size_t{r} * 32 + l];
      for (unsigned cc = 0; cc < cpr; ++cc) {
        const unsigned chunk = tt * cpr + (r / 2) * 4 * cpr + cc;
        const unsigned src = narrow ? 4 * g + 2 * sel + half + 2 * (chunk / 8) : 4 * g + holders * sel + chunk / 4;
        const unsigned shift = narrow ? (chunk % 8) * 4 : half * 16 + (chunk % 4) * 4;
        const uint32_t nib = (src < 32 ? meta[src] : 0u) >> shift;   // (a selector the ISA does not define can name a lane past the 32)
        const unsigned idx0 = nib & 3, idx1 = (nib >> 2) & 3;
        for (unsigned k = 0; k < stored; ++k) {
          const unsigned pos = tf32 ? idx0 / 2 : int4 ? 2 * (k < 2 ? idx0 : idx1) + k % 2 : (k == 0 ? idx0 : idx1);
          const unsigned col = chunk * chunk_elems + pos;
          if (col >= K) continue;
          const unsigned e = cc * stored + k;
          const auto x = decode_one<T>(tab, bits == 32 ? v : v >> (bits * e));
          out[size_t{row} * K + col] = negate ? negated(x) : x;
        }
      }
    }
}

template <FragType T> VGPU_COLD void frag_a_t(unsigned M, unsigned K, bool sparse, const uint32_t* regs, const uint32_t* meta, unsigned sel, float* f, int32_t* iout,
                                              bool negate) {
  using F = FragOf<T>;
  typename F::Out* const out = [&] {
    if constexpr (F::is_int) return iout;
    else return f;
  }();
  if (sparse) frag_a_sparse<T>(M, K, regs, meta, sel, out, negate);
  else frag_a_dense<T>(M, K, regs, out, negate);
}

template <FragType T> VGPU_COLD void frag_b_t(unsigned K, const uint32_t* regs, float* f, int32_t* iout) {
  using F = FragOf<T>;
  if constexpr (F::is_int) frag_b_dense<T>(K, regs, iout);
  else frag_b_dense<T>(K, regs, f);
}

}  // namespace

void canon_nan_rows(uint32_t* base, uint32_t rows, uint32_t cols, size_t ld) {
  for (uint32_t r = 0; r < rows; ++r) {
    uint32_t* p = base + r * ld;
    uint32_t i = 0;
    for (; i + 4 <= cols; i += 4) {
      V4u x;
      std::memcpy(&x, p + i, sizeof x);
      const V4u nan = reinterpret_cast<V4u>(reinterpret_cast<V4f>(x) != reinterpret_cast<V4f>(x));
      x = (x & ~nan) | (0x7fffffffu & nan);
      std::memcpy(p + i, &x, sizeof x);
    }
    for (; i < cols; ++i)
      if ((p[i] & 0x7fffffffu) > 0x7f800000u) p[i] = 0x7fffffffu;
  }
}

void mma_frag_a(FragType t, unsigned M, unsigned K, bool sparse, const uint32_t* regs, const uint32_t* meta, unsigned sel, float* f, int32_t* iout,
                bool negate) {
  switch (t) {
    case FragType::F16: return frag_a_t<FragType::F16>(M, K, sparse, regs, meta, sel, f, iout, negate);
    case FragType::BF16: return frag_a_t<FragType::BF16>(M, K, sparse, regs, meta, sel, f, iout, negate);
    case FragType::TF32: return frag_a_t<FragType::TF32>(M, K, sparse, regs, meta, sel, f, iout, negate);
    case FragType::E4M3: return frag_a_t<FragType::E4M3>(M, K, sparse, regs, meta, sel, f, iout, negate);
    case FragType::E5M2: return frag_a_t<FragType::E5M2>(M, K, sparse, regs, meta, sel, f, iout, negate);
    case FragType::S8: return frag_a_t<FragType::S8>(M, K, sparse, regs, meta, sel, f, iout, negate);
    case FragType::U8: return frag_a_t<FragType::U8>(M, K, sparse, regs, meta, sel, f, iout, negate);
    case FragType::S4: return frag_a_t<FragType::S4>(M, K, sparse, regs, meta, sel, f, iout, negate);
    case FragType::U4: return frag_a_t<FragType::U4>(M, K, sparse, regs, meta, sel, f, iout, negate);
  }
}

void mma_frag_b(FragType t, unsigned K, const uint32_t* regs, float* f, int32_t* iout) {
  switch (t) {
    case FragType::F16: return frag_b_t<FragType::F16>(K, regs, f, iout);
    case FragType::BF16: return frag_b_t<FragType::BF16>(K, regs, f, iout);
    case FragType::TF32: return frag_b_t<FragType::TF32>(K, regs, f, iout);
    case FragType::E4M3: return frag_b_t<FragType::E4M3>(K, regs, f, iout);
    case FragType::E5M2: return frag_b_t<FragType::E5M2>(K, regs, f, iout);
    case FragType::S8: return frag_b_t<FragType::S8>(K, regs, f, iout);
    case FragType::U8: return frag_b_t<FragType::U8>(K, regs, f, iout);
    case FragType::S4: return frag_b_t<FragType::S4>(K, regs, f, iout);
    case FragType::U4: return frag_b_t<FragType::U4>(K, regs, f, iout);
  }
}

template <unsigned Bits> VGPU_COLD void sparse_expand_t(unsigned K, unsigned sel, const uint32_t* meta, const uint32_t* stored_vals, uint32_t* out) {
  constexpr bool tf32 = Bits == 32;
  constexpr unsigned chunk_elems = tf32 ? 2 : 4, stored = chunk_elems / 2;
  const unsigned kp = K / 2, holders = 2 * (tf32 ? 2 * K : K) / 32;
  std::fill(out, out + size_t{16} * K, 0u);
  for (unsigned r = 0; r < 16; ++r) {
    const unsigned g = r % 8, half = r / 8;
    for (unsigned chunk = 0; chunk < kp / stored; ++chunk) {
      const unsigned src = Bits <= 8 ? 4 * g + 2 * sel + half + 2 * (chunk / 8) : 4 * g + holders * sel + chunk / 4;
      const unsigned shift = Bits <= 8 ? (chunk % 8) * 4 : half * 16 + (chunk % 4) * 4;
      const uint32_t nib = (src < 32 ? meta[src] : 0u) >> shift;
      const unsigned idx0 = nib & 3, idx1 = (nib >> 2) & 3;
      for (unsigned k = 0; k < stored; ++k) {
        const unsigned col = chunk * chunk_elems + (tf32 ? idx0 / 2 : (k == 0 ? idx0 : idx1));
        if (col < K) out[size_t{r} * K + col] = stored_vals[size_t{r} * kp + chunk * stored + k];
      }
    }
  }
}

void wgmma_sparse_expand(unsigned abits, unsigned K, unsigned sel, const uint32_t* meta, const uint32_t* stored_vals, uint32_t* out) {
  switch (abits) {
    case 32: return sparse_expand_t<32>(K, sel, meta, stored_vals, out);
    case 16: return sparse_expand_t<16>(K, sel, meta, stored_vals, out);
    default: return sparse_expand_t<8>(K, sel, meta, stored_vals, out);
  }
}

void mma_acc8_to_tile_i32(const uint32_t* regs, int32_t* tile) {
  for (unsigned s = 0; s < 2; ++s)
    for (unsigned l = 0; l < 32; ++l) tile[(l / 4) * 8 + 2 * (l % 4) + s] = static_cast<int32_t>(regs[s * 32 + l]);
}
void mma_tile_to_acc8_i32(const int32_t* tile, uint32_t* regs) {
  for (unsigned s = 0; s < 2; ++s)
    for (unsigned l = 0; l < 32; ++l) regs[s * 32 + l] = static_cast<uint32_t>(tile[(l / 4) * 8 + 2 * (l % 4) + s]);
}

void mma_f64_a_tile(unsigned M, unsigned K, const uint64_t* regs, double* tile) {
  for (unsigned r = 0; r < M * K / 32; ++r)
    for (unsigned l = 0; l < 32; ++l) tile[size_t{(l / 4) + 8 * (r % 2)} * K + (l % 4) + 4 * (r / 2)] = std::bit_cast<double>(regs[size_t{r} * 32 + l]);
}
void mma_f64_b_tile(unsigned K, const uint64_t* regs, double* tile) {
  for (unsigned r = 0; r < K / 4; ++r)
    for (unsigned l = 0; l < 32; ++l) tile[size_t{(l % 4) + 4 * r} * 8 + l / 4] = std::bit_cast<double>(regs[size_t{r} * 32 + l]);
}
void mma_f64_acc_to_tile(unsigned M, const uint64_t* regs, double* tile) {
  for (unsigned s = 0; s < M * 8 / 32; ++s)
    for (unsigned l = 0; l < 32; ++l) tile[size_t{(l / 4) + 8 * (s / 2)} * 8 + 2 * (l % 4) + s % 2] = std::bit_cast<double>(regs[size_t{s} * 32 + l]);
}
void mma_f64_tile_to_acc(unsigned M, const double* tile, uint64_t* regs) {
  for (unsigned s = 0; s < M * 8 / 32; ++s)
    for (unsigned l = 0; l < 32; ++l) regs[size_t{s} * 32 + l] = std::bit_cast<uint64_t>(tile[size_t{(l / 4) + 8 * (s / 2)} * 8 + 2 * (l % 4) + s % 2]);
}

void mma_b1(unsigned M, unsigned K, bool and_op, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* d) {
  // A's row g + 8 h, in words: word w of the row (K / 32 of them) is register 2 (w / 4) + h, lane 4g + w % 4. B's column g likewise: register w / 4,
  // lane 4g + w % 4.
  const unsigned words = K / 32;
  for (unsigned s = 0; s < M * 8 / 32; ++s)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, tt = l % 4, row = g + 8 * (s / 2), col = 2 * tt + s % 2;
      int64_t acc = static_cast<int32_t>(c[s * 32 + l]);
      for (unsigned w = 0; w < words; ++w) {
        const uint32_t aw = a[size_t{2 * (w / 4) + row / 8} * 32 + 4 * (row % 8) + w % 4], bw = b[size_t{w / 4} * 32 + 4 * col + w % 4];
        acc += std::popcount(and_op ? aw & bw : aw ^ bw);
      }
      d[s * 32 + l] = static_cast<uint32_t>(static_cast<int32_t>(acc));
    }
}

}  // namespace vgpu::exec
