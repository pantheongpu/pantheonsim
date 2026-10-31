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
// `canonical`: the tile came from simd::Kernels::tile_f32c, its NaNs already the canonical one (an f32 tile then needs no pass for them).
void wgmma_tile_to_acc(const float* tile, uint32_t N, bool f16, uint32_t* regs, bool canonical = false);
void wgmma_acc_to_tile_i32(const uint32_t* regs, uint32_t N, int32_t* tile);
void wgmma_tile_to_acc_i32(const int32_t* tile, uint32_t N, uint32_t* regs);

// Every f32 NaN of `rows` rows of `cols` words (row stride `ld`) to the canonical 0x7fffffff.
void canon_nan_rows(uint32_t* base, uint32_t rows, uint32_t cols, size_t ld);

// ---- mma.sync's wider forms, as tiles --------------------------------------------------------------------------------------------
//
// The dense and sparse m16n8kK / m8n8kK forms whose elements are f16, bf16, tf32, e4m3, e5m2, s8, u8, s4 or u4, read out of the warp's fragment
// registers (register r is the 32 lane words at regs + 32 * r) into row-major tiles the kernels take, and the f64 and .b1 forms. The layouts are the
// PTX ISA's (9.7.16.5): lane 4g + t holds, of A (M x K), row g + 8 (r % 2) and the consecutive columns t * p + 4 p (r / 2) on, p elements to a register;
// of B (K x 8), column g and the consecutive rows t * p + 4 p r on; of C and D (M x 8), slot s at row g + 8 (s / 2) and column 2 t + s % 2.
enum class FragType : uint8_t { F16, BF16, TF32, E4M3, E5M2, S8, U8, S4, U4 };
inline constexpr bool frag_is_int(FragType t) { return t >= FragType::S8; }
// The fragment type of a warpgroup MMA element type (the first seven are the same).
inline constexpr FragType frag_type_of(TileElem t) { return static_cast<FragType>(static_cast<uint8_t>(t)); }
inline constexpr unsigned frag_bits(FragType t) {
  return t == FragType::F16 || t == FragType::BF16 ? 16 : t == FragType::TF32 ? 32 : t == FragType::S4 || t == FragType::U4 ? 4 : 8;
}
// The tile of A, M x K row-major, into `f` (floating types) or `i`. `sparse`: the registers hold half of each row, placed by `meta` (the 32 lane
// words of the metadata register) and the selector `sel`, mma.sp's rule (exec::mma_sparse_column); the elements the metadata does not name are 0.
// `regs` has M * K / (sparse ? 2 : 1) / (32 * p) registers.
// `negate` flips the sign of every element the registers supply (a float's sign bit, so a zero stays a zero of the other sign; an integer is negated),
// not the zeros of a sparse A's other columns: wgmma's imm-scale-a of -1 for A in registers.
void mma_frag_a(FragType t, unsigned M, unsigned K, bool sparse, const uint32_t* regs, const uint32_t* meta, unsigned sel, float* f, int32_t* i,
                bool negate = false);
// The tile of B, K x 8 row-major.
void mma_frag_b(FragType t, unsigned K, const uint32_t* regs, float* f, int32_t* i);
// A warp's 16 rows of a sparse wgmma A that is already decoded (32-bit values, 16 rows of K / 2 stored elements, from shared memory): each row spread over its
// K columns by the metadata as above (`abits`, the element width: 8, 16 or 32), the other columns 0. The bits of the values are moved, not read.
void wgmma_sparse_expand(unsigned abits, unsigned K, unsigned sel, const uint32_t* meta, const uint32_t* stored, uint32_t* out);
// C and D of the m8n8 integer forms (two slots a lane, registers 0 and 1) and the m16n8 ones (wgmma_acc_to_tile with N = 8).
void mma_acc8_to_tile_i32(const uint32_t* regs, int32_t* tile);
void mma_tile_to_acc8_i32(const int32_t* tile, uint32_t* regs);
// The f64 forms (m8n8k4, and sm_90's m16n8k4, k8 and k16): A (M x K) register r holds, in lane 4g + t, row g + 8 (r % 2) and column t + 4 (r / 2); B (K x 8)
// register r, row t + 4 r and column g; C and D (M x 8) slot s, row g + 8 (s / 2) and column 2 t + s % 2. Each register is the 32 lanes' 64-bit values;
// `regs` is the registers one after the other.
void mma_f64_a_tile(unsigned M, unsigned K, const uint64_t* regs, double* tile);
void mma_f64_b_tile(unsigned K, const uint64_t* regs, double* tile);
void mma_f64_acc_to_tile(unsigned M, const uint64_t* regs, double* tile);
void mma_f64_tile_to_acc(unsigned M, const double* tile, uint64_t* regs);
// The .b1 forms (m8n8k128, m16n8k128 and m16n8k256): D = C + the number of positions where A's row and B's column differ (xor) or are both set (and).
// A's registers (32 bits of a row each) are M * K / 1024, B's K * 8 / 1024, C's and D's M * 8 / 32; all as 32 lane words a register.
void mma_b1(unsigned M, unsigned K, bool and_op, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* d);

}  // namespace vgpu::exec
