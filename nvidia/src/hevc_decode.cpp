// See hevc_decode.hpp.
#include "hevc_decode.hpp"

#include "hevc_dec_internal.hpp"

namespace vgpu_hevc {

void Frame::alloc(int w, int h) {
  width = w;
  height = h;
  cwidth = w / 2;
  cheight = h / 2;
  stride_y = w;
  stride_c = w / 2;
  y.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
  u.assign(static_cast<size_t>(cwidth) * static_cast<size_t>(cheight), 0);
  v.assign(static_cast<size_t>(cwidth) * static_cast<size_t>(cheight), 0);
  col.clear();
  col_w = col_h = 0;
}

namespace {

#include "hevc_dec_cabac.inc"
#include "hevc_dec_slice.inc"
#include "hevc_dec_residual.inc"
#include "hevc_dec_intra.inc"
#include "hevc_dec_inter.inc"
#include "hevc_dec_filter.inc"

bool Dec::run(const std::vector<SliceData>& slices) {
  if (!ok_) return false;
  if (P.chroma_format_idc != 1 || P.separate_colour_plane) {
    fail("only 4:2:0 pictures are decoded");
    return false;
  }
  if (P.extended_precision || P.cabac_bypass_alignment) {
    fail("extended precision processing and CABAC bypass alignment are not decoded");
    return false;
  }
  if (P.bit_depth_luma < 8 || P.bit_depth_luma > 12 || P.bit_depth_chroma < 8 || P.bit_depth_chroma > 12) {
    fail("bit depths above 12 are not decoded");
    return false;
  }
  if (W <= 0 || H <= 0 || (W & 7) || (H & 7) || F->width != W || F->height != H) {
    fail("invalid picture size");
    return false;
  }
  if (log2_ctb < 4 || log2_ctb > 6 || log2_min_cb < 3 || log2_min_tb < 2 || log2_max_tb > 5 || log2_min_tb >= log2_min_cb || log2_max_tb > log2_ctb) {
    fail("invalid block sizes");
    return false;
  }
  // 7.4.3.2.1: the picture is a whole number of minimum-size coding blocks. A damaged SPS can break that, and then the last
  // blocks would be predicted and reconstructed past the end of the planes.
  if ((W & ((1 << log2_min_cb) - 1)) || (H & ((1 << log2_min_cb) - 1))) {
    fail("the picture size is not a multiple of the minimum coding block size");
    return false;
  }
  // scaling factors from the lists the caller resolved
  if (P.scaling_list_enabled) {
    for (int m = 0; m < 6; ++m) {
      build_scaling_factor(P.sl4[m], 16, 2, sf4_[m]);
      build_scaling_factor(P.sl8[m], 16, 3, sf8_[m]);
      build_scaling_factor(P.sl16[m], P.dc16[m], 4, sf16_[m]);
    }
    for (int m = 0; m < 2; ++m) build_scaling_factor(P.sl32[m], P.dc32[m], 5, sf32_[m]);
  }
  slices_.reserve(slices.size() + 1);
  bool first = true;
  for (const SliceData& sd : slices) {
    decode_slice_segment(sd, first);
    first = false;
    if (!ok_) break;
  }
  // whatever was decoded is finished: loop filters over the decoded part, then the motion kept for later pictures
  if (!slices_.empty()) {
    deblock();
    sao_filter();
    finish_col();
  }
  return ok_;
}

}  // namespace

bool decode_picture(Frame* cur, const PicParams& pp, const std::vector<SliceData>& slices, std::string* error) {
  std::string local;
  Dec d(cur, pp, error ? error : &local);
  return d.run(slices);
}

}  // namespace vgpu_hevc
