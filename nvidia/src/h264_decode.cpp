// The H.264 decoder: see h264_decode.hpp. This file holds the decoder object, the
// slice loop, neighbour derivation and reference picture lists; the macroblock
// syntax, the reconstruction (intra, inter, transforms) and the deblocking filter are
// the parts it includes.
#include "h264_dec_internal.hpp"

#include <cstdio>

namespace vgpu_h264 {

void Frame::alloc(int w_mbs, int h_mbs) {
  mbs_w = w_mbs;
  mbs_h = h_mbs;
  stride_y = w_mbs * 16;
  stride_c = w_mbs * 8;
  y.assign(static_cast<size_t>(stride_y) * h_mbs * 16, 0);
  u.assign(static_cast<size_t>(stride_c) * h_mbs * 8, 128);
  v.assign(static_cast<size_t>(stride_c) * h_mbs * 8, 128);
}

HeaderCtx header_ctx(const PicParams& pp) {
  HeaderCtx c;
  c.log2_max_frame_num = pp.log2_max_frame_num;
  c.poc_type = pp.poc_type;
  c.log2_max_poc_lsb = pp.log2_max_poc_lsb;
  c.delta_pic_order_always_zero = pp.delta_pic_order_always_zero;
  c.frame_mbs_only = pp.frame_mbs_only;
  c.chroma_array_type = pp.chroma_format_idc;
  c.bottom_field_pic_order_in_frame_present = pp.bottom_field_pic_order_in_frame_present;
  c.redundant_pic_cnt_present = pp.redundant_pic_cnt_present;
  c.num_ref_idx_default[0] = pp.num_ref_idx_default[0];
  c.num_ref_idx_default[1] = pp.num_ref_idx_default[1];
  c.weighted_pred = pp.weighted_pred;
  c.weighted_bipred_idc = pp.weighted_bipred_idc;
  c.cabac = pp.cabac;
  c.deblocking_control_present = pp.deblocking_control_present;
  c.pic_init_qp = pp.pic_init_qp;
  c.num_slice_groups = pp.num_slice_groups;
  c.slice_group_map_type = pp.slice_group_map_type;
  return c;
}

namespace {

// A view of a plane of the picture being decoded or of a reference picture: a frame, or one field of a frame.
struct Plane {
  uint8_t* p = nullptr;
  int stride = 0, w = 0, h = 0;
};

struct Nb {
  int addr = -1;   // macroblock address, -1 = not available
  int x = 0, y = 0;   // location inside that macroblock
};

constexpr int kQpcTable[22] = {29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39};
inline int qpc_of(int qpi) { return qpi < 30 ? qpi : kQpcTable[qpi - 30]; }

// P and B macroblock types: partition shape (0 16x16, 1 16x8, 2 8x16, 3 8x8) and prediction flags per partition.
struct BMbType {
  uint8_t shape, p0, p1;
};
constexpr BMbType kBMb[23] = {{255, 0, 0}, {0, 1, 0}, {0, 2, 0}, {0, 3, 0}, {1, 1, 1}, {2, 1, 1}, {1, 2, 2}, {2, 2, 2}, {1, 1, 2}, {2, 1, 2}, {1, 2, 1}, {2, 2, 1},
                              {1, 1, 3}, {2, 1, 3}, {1, 2, 3}, {2, 2, 3}, {1, 3, 1}, {2, 3, 1}, {1, 3, 2}, {2, 3, 2}, {1, 3, 3}, {2, 3, 3}, {3, 0, 0}};
// sub_mb_type: shape (0 8x8, 1 8x4, 2 4x8, 3 4x4), prediction flags
constexpr uint8_t kBSubShape[13] = {0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 3, 3};
constexpr uint8_t kBSubPred[13] = {0, 1, 2, 3, 1, 1, 2, 2, 3, 3, 1, 2, 3};

#include "h264_dec_core.inc"

}  // namespace

bool decode_picture(Frame* cur, const PicParams& pp, const std::vector<SliceData>& slices, std::string* error) {
  Dec d(cur, pp);
  std::string e;
  const bool ok = d.run(slices, &e);
  if (error) *error = e;
  return ok;
}

}  // namespace vgpu_h264
