#include "vgpu/exec/tile_mma.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <vector>

#include "vgpu/exec/numerics.hpp"
#include "vgpu/simd/fp16.hpp"
#include "vgpu/simd/simd.hpp"

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

void wgmma_tile_to_acc(const float* tile, uint32_t N, bool f16, uint32_t* regs) {
  if (!f16) {
    tile_to_words<true>(reinterpret_cast<const uint32_t*>(tile), N, regs);
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

}  // namespace vgpu::exec
