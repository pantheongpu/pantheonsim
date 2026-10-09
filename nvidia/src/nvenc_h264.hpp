// A conformant H.264 bitstream writer for VirtualGPU's NVENC: Annex B, CAVLC,
// every macroblock an I_PCM one, so the stream is lossless, needs no transform
// or prediction, and any H.264 decoder reads it back exactly (ffmpeg's does:
// nvidia/tests/e2e/nvenc_h264.cpp checks the samples a decoder returns). It is
// not compression -- a frame costs a little more than its raw size -- and it
// is not NVIDIA's encoder: rate control, GOP structure and every other
// encoder setting are accepted and have nothing to act on.
//
// Written from ITU-T H.264 (clauses 7.3 and 7.4 for the syntax, 8.3.5 for the
// I_PCM macroblock) and nothing else.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vgpu_nvenc {

class BitWriter {
 public:
  void put(uint32_t value, int bits) {
    for (int i = bits - 1; i >= 0; --i) bit((value >> i) & 1);
  }
  void bit(int b) {
    cur_ = static_cast<uint8_t>(cur_ << 1 | (b & 1));
    if (++n_ == 8) flush_byte();
  }
  void ue(uint32_t v) {
    const uint64_t x = static_cast<uint64_t>(v) + 1;
    int len = 0;
    while ((x >> (len + 1)) != 0) ++len;
    put(0, len);
    for (int i = len; i >= 0; --i) bit(static_cast<int>((x >> i) & 1));
  }
  void se(int32_t v) { ue(v > 0 ? static_cast<uint32_t>(2 * v - 1) : static_cast<uint32_t>(-2 * static_cast<int64_t>(v))); }
  bool aligned() const { return n_ == 0; }
  void align_zero() {
    while (n_) bit(0);
  }
  void byte(uint8_t b) { put(b, 8); }
  // rbsp_trailing_bits(): a one, then zeros to the byte boundary.
  void trailing() {
    bit(1);
    align_zero();
  }
  const std::vector<uint8_t>& bytes() const { return out_; }

 private:
  void flush_byte() {
    out_.push_back(cur_);
    cur_ = 0;
    n_ = 0;
  }
  std::vector<uint8_t> out_;
  uint8_t cur_ = 0;
  int n_ = 0;
};

// A NAL unit in Annex B form: a four-byte start code, the header, and the RBSP
// with emulation prevention bytes (clause 7.4.1: no 00 00 00/01/02/03 inside).
inline void append_nal(std::vector<uint8_t>& out, int ref_idc, int type, const std::vector<uint8_t>& rbsp) {
  out.insert(out.end(), {0, 0, 0, 1, static_cast<uint8_t>(ref_idc << 5 | type)});
  int zeros = 0;
  for (uint8_t b : rbsp) {
    if (zeros >= 2 && b <= 3) {
      out.push_back(3);
      zeros = 0;
    }
    out.push_back(b);
    zeros = b == 0 ? zeros + 1 : 0;
  }
}

struct H264Stream {
  int width = 0, height = 0;   // luma samples of the picture, as the application sees it
  int fps_num = 30, fps_den = 1;
  int mbs_x() const { return (width + 15) / 16; }
  int mbs_y() const { return (height + 15) / 16; }
  int coded_width() const { return (width + 1) & ~1; }
  int coded_height() const { return (height + 1) & ~1; }
  // Table A-1: the lowest level whose frame size and macroblock rate fit.
  int level_idc() const {
    struct L {
      int idc;
      long max_mbps, max_fs;
    };
    static const L levels[] = {{10, 1485, 99},     {11, 3000, 396},    {12, 6000, 396},     {13, 11880, 396},   {20, 11880, 396},
                               {21, 19800, 792},   {22, 20250, 1620},  {30, 40500, 1620},   {31, 108000, 3600}, {32, 216000, 5120},
                               {40, 245760, 8192}, {41, 245760, 8192}, {42, 522240, 8704},  {50, 589824, 22080}, {51, 983040, 36864},
                               {52, 2073600, 36864}, {60, 4177920, 139264}, {61, 8355840, 139264}, {62, 16711680, 139264}};
    const long fs = static_cast<long>(mbs_x()) * mbs_y();
    const long mbps = fs * fps_num / std::max(fps_den, 1);
    for (const L& l : levels)
      if (fs <= l.max_fs && mbps <= l.max_mbps) return l.idc;
    return 62;
  }
  // Baseline profile: I_PCM and CAVLC are all this stream uses.
  std::vector<uint8_t> sps() const {
    BitWriter w;
    w.put(66, 8);          // profile_idc: Baseline
    w.put(0, 8);           // constraint_set flags and reserved bits
    w.put(static_cast<uint32_t>(level_idc()), 8);
    w.ue(0);               // seq_parameter_set_id
    w.ue(0);               // log2_max_frame_num_minus4
    w.ue(2);               // pic_order_cnt_type 2: output order is decoding order
    w.ue(1);               // max_num_ref_frames
    w.bit(0);              // gaps_in_frame_num_value_allowed_flag
    w.ue(static_cast<uint32_t>(mbs_x() - 1));
    w.ue(static_cast<uint32_t>(mbs_y() - 1));
    w.bit(1);              // frame_mbs_only_flag
    w.bit(1);              // direct_8x8_inference_flag
    // Cropping is in units of two luma samples (4:2:0), so an odd width or
    // height is coded as the next even one: the picture a decoder returns is
    // coded_width() by coded_height().
    const int crop_r = mbs_x() * 16 - coded_width(), crop_b = mbs_y() * 16 - coded_height();
    if (crop_r || crop_b) {
      w.bit(1);
      w.ue(0);
      w.ue(static_cast<uint32_t>(crop_r / 2));
      w.ue(0);
      w.ue(static_cast<uint32_t>(crop_b / 2));
    } else {
      w.bit(0);
    }
    w.bit(0);              // vui_parameters_present_flag
    w.trailing();
    return w.bytes();
  }
  std::vector<uint8_t> pps() const {
    BitWriter w;
    w.ue(0);   // pic_parameter_set_id
    w.ue(0);   // seq_parameter_set_id
    w.bit(0);  // entropy_coding_mode_flag: CAVLC
    w.bit(0);  // bottom_field_pic_order_in_frame_present_flag
    w.ue(0);   // num_slice_groups_minus1
    w.ue(0);   // num_ref_idx_l0_default_active_minus1
    w.ue(0);   // num_ref_idx_l1_default_active_minus1
    w.bit(0);  // weighted_pred_flag
    w.put(0, 2);   // weighted_bipred_idc
    w.se(0);   // pic_init_qp_minus26
    w.se(0);   // pic_init_qs_minus26
    w.se(0);   // chroma_qp_index_offset
    w.bit(1);  // deblocking_filter_control_present_flag
    w.bit(0);  // constrained_intra_pred_flag
    w.bit(0);  // redundant_pic_cnt_present_flag
    w.trailing();
    return w.bytes();
  }
  // The headers a decoder needs, as the first bytes of a stream.
  std::vector<uint8_t> parameter_sets() const {
    std::vector<uint8_t> out;
    append_nal(out, 3, 7, sps());
    append_nal(out, 3, 8, pps());
    return out;
  }
  // One IDR picture from 8-bit 4:2:0 planes. `luma(x, y)` and `chroma(plane, x,
  // y)` (plane 0 = Cb, 1 = Cr) return the samples; positions past the picture
  // repeat its last row and column, which the SPS's cropping hides.
  template <class Luma, class Chroma>
  std::vector<uint8_t> idr(Luma luma, Chroma chroma, int idr_id) const {
    BitWriter w;
    w.ue(0);                              // first_mb_in_slice
    w.ue(7);                              // slice_type: I, and so is the whole picture
    w.ue(0);                              // pic_parameter_set_id
    w.put(0, 4);                          // frame_num (log2_max_frame_num is 4)
    w.ue(static_cast<uint32_t>(idr_id & 1));   // idr_pic_id
    w.bit(0);                             // no_output_of_prior_pics_flag
    w.bit(0);                             // long_term_reference_flag
    w.se(0);                              // slice_qp_delta
    w.ue(1);                              // disable_deblocking_filter_idc: off
    const int cw = (width + 1) / 2, ch = (height + 1) / 2;
    auto clampx = [&](int x, int hi) { return std::min(std::max(x, 0), hi - 1); };
    for (int my = 0; my < mbs_y(); ++my)
      for (int mx = 0; mx < mbs_x(); ++mx) {
        w.ue(25);                         // mb_type: I_PCM
        w.align_zero();                   // pcm_alignment_zero_bit
        for (int y = 0; y < 16; ++y)
          for (int x = 0; x < 16; ++x) w.byte(luma(clampx(mx * 16 + x, width), clampx(my * 16 + y, height)));
        for (int plane = 0; plane < 2; ++plane)
          for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) w.byte(chroma(plane, clampx(mx * 8 + x, cw), clampx(my * 8 + y, ch)));
      }
    w.trailing();
    std::vector<uint8_t> out;
    append_nal(out, 3, 5, w.bytes());
    return out;
  }
};

}  // namespace vgpu_nvenc
