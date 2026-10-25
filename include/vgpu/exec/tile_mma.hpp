// The operand side of the tensor-core instructions that work on whole tiles (wgmma and its SASS kin, mma.sync's wider forms): reading a
// shared-memory matrix a descriptor describes, and the elements of mma fragments, into the float or integer arrays the SIMD tile kernels
// (vgpu/simd/simd.hpp: tile_f32, tile_i32, tile_f64) take. Both engines call these with their own memory access; the arithmetic is the
// kernels', and the bits are the scalar forms' (tests/unit/test_tile_mma.cpp holds each to its scalar reference).
#pragma once

#include <cstddef>
#include <cstdint>

#include "vgpu/exec/wgmma.hpp"

namespace vgpu::exec {

// The element types of the warpgroup MMA's operands (the mma.sync fragments use the same decoding for these).
enum class TileElem : uint8_t { F16, BF16, TF32, E4M3, E5M2, S8, U8 };

inline constexpr uint32_t tile_elem_bytes(TileElem t) {
  return t == TileElem::F16 || t == TileElem::BF16 ? 2 : t == TileElem::TF32 ? 4 : 1;
}
inline constexpr bool tile_elem_is_int(TileElem t) { return t == TileElem::S8 || t == TileElem::U8; }

// The float a floating element holds (f16, bf16, tf32 -- its low 13 bits dropped -- e4m3, e5m2), as the scalar decoders give it.
// `raw` is the element's bits (`tile_elem_bytes` of them).
float tile_decode_f32(TileElem t, uint32_t raw);
// The integer an s8 or u8 element holds.
inline int32_t tile_decode_i32(TileElem t, uint32_t raw) {
  return t == TileElem::S8 ? static_cast<int32_t>(static_cast<int8_t>(raw & 0xff)) : static_cast<int32_t>(raw & 0xff);
}

// Reads the rows x K matrix that a wgmma matrix descriptor describes (K-major: row `mn` of K elements; else element (mn, k) at the
// MN-major position) out of `smem`, decoded: element (mn, k) lands in out[mn * K + k], or in out[k * rows + mn] when `transpose` (B, which
// the tile kernels want as K x N). `negate` flips every element's sign (the instruction's imm-scale of -1). `off_mask` is applied to each
// element's offset (the SASS executor keeps 24 bits). Returns false, with `out` unspecified, if any element lies outside smem[0, size):
// the caller then reads the matrix element by element, which raises the engine's own error.
bool wgmma_read_f32(const uint8_t* smem, size_t size, const WgmmaDesc& d, bool k_major, TileElem t, uint32_t rows, uint32_t K, bool negate,
                    bool transpose, uint64_t off_mask, float* out);
bool wgmma_read_i32(const uint8_t* smem, size_t size, const WgmmaDesc& d, bool k_major, TileElem t, uint32_t rows, uint32_t K, bool negate,
                    bool transpose, uint64_t off_mask, int32_t* out);

// The accumulator of one warp of a warpgroup MMA, N columns (a multiple of 8) wide, as its registers and as the 16 x N row-major tile the
// kernels work on. `regs` is N / 2 registers of 32 lane words (f16: N / 4, two halves a word), register 4j + 2h + u (f16: 2j + h) holding
// row g + 8h and column 8j + 2t + u of lane 4g + t. A NaN written back is the canonical one (f32: 0x7fffffff, f16: 0x7fff).
void wgmma_acc_to_tile(const uint32_t* regs, uint32_t N, bool f16, float* tile);
void wgmma_tile_to_acc(const float* tile, uint32_t N, bool f16, uint32_t* regs);
void wgmma_acc_to_tile_i32(const uint32_t* regs, uint32_t N, int32_t* tile);
void wgmma_tile_to_acc_i32(const int32_t* tile, uint32_t N, uint32_t* regs);

}  // namespace vgpu::exec
