// A tensor copy's elements (cp.async.bulk.tensor; SASS's UTMALDG, UTMASTG,
// UTMAREDG): which elements of the tensor a copy takes, in the order they sit
// in shared memory, for both engines.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "vgpu/error.hpp"
#include "vgpu/exec/tensormap.hpp"

namespace vgpu::exec {

// The box one copy moves.
struct TmaBox {
  uint32_t gs = 0;        // bytes of global memory per element (per group of 16, packed)
  uint32_t es = 0;        // bytes of shared memory per element
  bool packed = false;    // a packed sub-byte type, copied sixteen values at a time
  uint64_t total = 0;     // elements
  std::array<uint64_t, 5> count{1, 1, 1, 1, 1};   // per dimension (tile mode)
  std::array<int64_t, 5> start{};                 // the copy's coordinates
};

struct TmaProblem {
  Err code;
  std::string what;
};

// Checks a copy of `dims` dimensions at `coords` against `map`, and sizes
// its box. `map` is adjusted for the packed types (dimension 0 then counts
// groups of 16). four_rows: .tile::gather4/.tile::scatter4, whose coords are
// the column and then four rows.
inline std::optional<TmaProblem> tma_box(TensorMap& map, uint32_t dims, const int64_t* coords, bool four_rows, bool im2col,
                                         bool to_shared, bool reduce, TmaBox* box) {
  if (map.rank != dims)
    return TmaProblem{Err::InvalidValue, "a ." + std::to_string(dims) + "d tensor copy through a rank-" +
                                             std::to_string(map.rank) + " tensor map"};
  // The packed sub-byte types (5.5.1.1) are copied sixteen values at a time:
  // `gs` bytes of global memory to an `es`-byte slot of shared memory, so
  // dimension 0 is counted in groups here. A load leaves a slot's padding as
  // it was ("un-initialized"); a store of type 15 is .b6p2x16, sixteen bytes
  // each holding a value in bits 0-5, packed into twelve.
  uint32_t gs = 0, packed_es = 0;
  const bool packed = TensorMap::packed_group(map.type, &gs, &packed_es);
  if (packed) {
    const bool padded = packed_es == 16;
    if (map.im2col) return TmaProblem{Err::UnsupportedPtx, "an im2col copy of a packed sub-byte type is not implemented"};
    if (reduce)
      return TmaProblem{Err::InvalidValue,
                        "cp.reduce.async.bulk does not support the packed sub-byte types (9.7.10.28.5.1)"};
    if (!to_shared && map.type == TmapType::U4x16Align16)
      return TmaProblem{Err::InvalidValue,
                        "a tensor store (.global.shared::cta) does not support .b4x16_p64 (9.7.10.28.5.1)"};
    if (to_shared && map.swizzle == TmapSwizzle::B128Atom64 && padded)
      return TmaProblem{Err::InvalidValue,
                        "the 128-byte swizzle in 64-byte atoms is for stores only with a padded sub-byte type"};
    if (padded ? coords[0] % 128 : coords[0] % 16)
      return TmaProblem{padded ? Err::InvalidValue : Err::UnsupportedPtx,
                        padded ? "the first coordinate of a .b4x16_p64 or .b6x16_p32 copy must be a multiple "
                                 "of 128 (9.7.10.28.5.1)"
                               : "a .b4x16 copy whose first coordinate is not a multiple of 16 is not implemented"};
    if (map.dim[0] % 16 || map.box[0] % 16)
      return TmaProblem{Err::UnsupportedPtx,
                        "a packed sub-byte tensor whose dimension 0 or box is not whole groups of 16 values"};
    map.dim[0] /= 16;
    map.box[0] /= 16;
    map.stride[0] = gs;
  }
  box->packed = packed;
  box->es = packed ? packed_es : static_cast<uint32_t>(map.stride[0]);
  box->gs = packed ? gs : box->es;
  if (map.im2col != im2col)
    return TmaProblem{Err::InvalidValue, map.im2col
                                             ? "a tile-mode tensor copy through a map made by cuTensorMapEncodeIm2col"
                                             : "an im2col tensor copy through a map made by cuTensorMapEncodeTiled"};
  // Tile mode: the box, dimension 0 fastest, each dimension's count the box
  // over its traversal stride rounded up. im2col mode (5.5.4): `pixels`
  // pixels, each `channels` channels wide.
  box->count = {1, 1, 1, 1, 1};
  box->total = 1;
  if (four_rows) {
    // .tile::gather4/.tile::scatter4 (5.5.3.4): four boxes one row high,
    // packed one after another -- as a 4-row box would be.
    if (map.rank != 2 || map.box[1] != 1)
      return TmaProblem{Err::InvalidValue,
                        ".tile::gather4/.tile::scatter4 need a 2D tensor map whose box is one row high"};
    box->count[0] = (map.box[0] + map.elem_stride[0] - 1) / map.elem_stride[0];
    box->total = 4 * box->count[0];
  } else if (map.im2col) {
    box->total = uint64_t{map.pixels} * map.channels;
  } else {
    for (uint32_t d = 0; d < map.rank; ++d) {
      box->count[d] = (map.box[d] + map.elem_stride[d] - 1) / map.elem_stride[d];
      box->total *= box->count[d];
    }
  }
  box->start = {};
  for (uint32_t d = 0; d < map.rank; ++d) box->start[d] = coords[d];
  if (packed) box->start[0] /= 16;
  return std::nullopt;
}

// Calls visit(e, gaddr, inside) for each element e of the box, in the order
// the elements sit in shared memory (element e at e * box.es before the
// swizzle): its global address, and whether it is inside the tensor (a load
// reads zero outside it, a store skips it). im2col (5.5.4): the pixels walked
// through the bounding box -- W fastest, each spatial dimension stepping by
// its traversal stride from lower to dim + upper - 1 and then starting over
// from lower with the next dimension advanced, the batch last -- beginning at
// the copy's coordinates, each read at that position plus its `offsets`.
// rows: .tile::gather4's four row coordinates.
template <class F>
void tma_walk(const TensorMap& map, const TmaBox& box, const int64_t* rows, const std::array<int64_t, 3>& offsets,
              bool four_rows, F&& visit) {
  const uint32_t nsp = map.rank >= 2 ? map.rank - 2 : 0;   // im2col's spatial dimensions
  std::array<int64_t, 5> pix = box.start;                  // im2col: the pixel being read, in [1, rank-1]
  std::array<uint64_t, 5> j{};
  for (uint64_t e = 0; e < box.total; ++e) {
    bool inside = true;
    uint64_t gaddr = map.address;
    const auto at = [&](uint32_t d, int64_t g) {
      if (g < 0 || static_cast<uint64_t>(g) >= map.dim[d]) inside = false;
      else gaddr += static_cast<uint64_t>(g) * map.stride[d];
    };
    if (four_rows) {
      at(0, box.start[0] + static_cast<int64_t>((e % box.count[0]) * map.elem_stride[0]));
      at(1, rows[e / box.count[0]]);
    } else if (map.im2col) {
      const uint64_t ch = e % map.channels;
      at(0, box.start[0] + static_cast<int64_t>(ch));
      for (uint32_t i = 0; i < nsp; ++i) at(1 + i, pix[1 + i] + offsets[i]);
      at(map.rank - 1, pix[map.rank - 1]);
    } else {
      for (uint32_t d = 0; d < map.rank; ++d) at(d, box.start[d] + static_cast<int64_t>(j[d] * map.elem_stride[d]));
    }
    visit(e, gaddr, inside);
    if (map.im2col) {
      // The next pixel, after the last channel of this one.
      if ((e + 1) % map.channels == 0) {
        uint32_t i = 0;
        for (; i < nsp; ++i) {
          pix[1 + i] += map.elem_stride[1 + i];
          if (pix[1 + i] <= static_cast<int64_t>(map.dim[1 + i]) - 1 + map.upper[i]) break;
          pix[1 + i] = map.lower[i];
        }
        if (i == nsp) ++pix[map.rank - 1];
      }
    } else {
      for (uint32_t d = 0; d < map.rank; ++d) {
        if (++j[d] < box.count[d]) break;
        j[d] = 0;
      }
    }
  }
}

}  // namespace vgpu::exec
