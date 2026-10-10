// A conformant HEVC (H.265) bitstream writer for VirtualGPU's NVENC: Annex B, one
// IDR picture per frame, 16x16 coding tree blocks that are each a single PCM coding
// unit, so the stream is lossless and needs no prediction or transform, and any HEVC
// decoder returns the input exactly (nvidia/tests/e2e/nvenc_h264.cpp decodes it with
// ffmpeg and compares every sample). It is not compression -- a frame costs a little
// more than its raw size.
//
// HEVC has no CAVLC: even a picture of PCM units is arithmetic-coded, so this holds
// the CABAC encoder of ITU-T H.265 9.3.4.3 for the three syntax elements such a
// picture uses -- split_cu_flag (a context-coded bin, always 0), pcm_flag and
// end_of_slice_segment_flag (terminate bins) -- and the table rangeTabLPS (Table
// 9-46) for the one range column every coding tree unit starts from.
//
// Written from ITU-T H.265 (clauses 7.3 and 7.4 for the syntax, 9.3 for CABAC) and
// nothing else.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "nvenc_h264.hpp"   // BitWriter

namespace vgpu_nvenc {

// An HEVC NAL unit in Annex B form: a four-byte start code, the two-byte header and
// the RBSP with emulation prevention bytes.
inline void append_hevc_nal(std::vector<uint8_t>& out, int type, const std::vector<uint8_t>& rbsp) {
  out.insert(out.end(), {0, 0, 0, 1, static_cast<uint8_t>(type << 1), 1});   // layer 0, temporal id 0
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

class HevcStream {
 public:
  int width = 0, height = 0;   // luma samples of the picture, as the application sees it
  int fps_num = 30, fps_den = 1;

  // The picture is coded on a grid of 16x16 coding tree blocks, cropped to the
  // application's size (4:2:0 crops in units of two samples, so an odd size is coded
  // as the next even one).
  int ctus_x() const { return (width + 15) / 16; }
  int ctus_y() const { return (height + 15) / 16; }
  int coded_width() const { return (width + 1) & ~1; }
  int coded_height() const { return (height + 1) & ~1; }

  std::vector<uint8_t> vps() const {
    BitWriter w;
    w.put(0, 4);        // vps_video_parameter_set_id
    w.bit(1);           // vps_base_layer_internal_flag
    w.bit(1);           // vps_base_layer_available_flag
    w.put(0, 6);        // vps_max_layers_minus1
    w.put(0, 3);        // vps_max_sub_layers_minus1
    w.bit(1);           // vps_temporal_id_nesting_flag
    w.put(0xffff, 16);  // vps_reserved_0xffff_16bits
    profile_tier_level(w);
    w.bit(1);           // vps_sub_layer_ordering_info_present_flag
    w.ue(0);            // vps_max_dec_pic_buffering_minus1
    w.ue(0);            // vps_max_num_reorder_pics
    w.ue(0);            // vps_max_latency_increase_plus1
    w.put(0, 6);        // vps_max_layer_id
    w.ue(0);            // vps_num_layer_sets_minus1
    w.bit(0);           // vps_timing_info_present_flag
    w.bit(0);           // vps_extension_flag
    w.trailing();
    return w.bytes();
  }

  std::vector<uint8_t> sps() const {
    BitWriter w;
    w.put(0, 4);        // sps_video_parameter_set_id
    w.put(0, 3);        // sps_max_sub_layers_minus1
    w.bit(1);           // sps_temporal_id_nesting_flag
    profile_tier_level(w);
    w.ue(0);            // sps_seq_parameter_set_id
    w.ue(1);            // chroma_format_idc: 4:2:0
    w.ue(static_cast<uint32_t>(ctus_x() * 16));
    w.ue(static_cast<uint32_t>(ctus_y() * 16));
    const int crop_r = ctus_x() * 16 - coded_width(), crop_b = ctus_y() * 16 - coded_height();
    if (crop_r || crop_b) {
      w.bit(1);         // conformance_window_flag, offsets in units of two luma samples
      w.ue(0);
      w.ue(static_cast<uint32_t>(crop_r / 2));
      w.ue(0);
      w.ue(static_cast<uint32_t>(crop_b / 2));
    } else {
      w.bit(0);
    }
    w.ue(0);            // bit_depth_luma_minus8
    w.ue(0);            // bit_depth_chroma_minus8
    w.ue(4);            // log2_max_pic_order_cnt_lsb_minus4
    w.bit(1);           // sps_sub_layer_ordering_info_present_flag
    w.ue(0);            // sps_max_dec_pic_buffering_minus1
    w.ue(0);            // sps_max_num_reorder_pics
    w.ue(0);            // sps_max_latency_increase_plus1
    w.ue(0);            // log2_min_luma_coding_block_size_minus3: 8x8
    w.ue(1);            // log2_diff_max_min_luma_coding_block_size: 16x16 coding tree blocks
    w.ue(0);            // log2_min_luma_transform_block_size_minus2: 4x4
    w.ue(2);            // log2_diff_max_min_luma_transform_block_size: up to 16x16
    w.ue(0);            // max_transform_hierarchy_depth_inter
    w.ue(0);            // max_transform_hierarchy_depth_intra
    w.bit(0);           // scaling_list_enabled_flag
    w.bit(0);           // amp_enabled_flag
    w.bit(0);           // sample_adaptive_offset_enabled_flag
    w.bit(1);           // pcm_enabled_flag
    w.put(7, 4);        // pcm_sample_bit_depth_luma_minus1: 8 bits
    w.put(7, 4);        // pcm_sample_bit_depth_chroma_minus1
    w.ue(0);            // log2_min_pcm_luma_coding_block_size_minus3: 8x8
    w.ue(1);            // log2_diff_max_min_pcm_luma_coding_block_size: up to 16x16
    w.bit(1);           // pcm_loop_filter_disabled_flag
    w.ue(0);            // num_short_term_ref_pic_sets
    w.bit(0);           // long_term_ref_pics_present_flag
    w.bit(0);           // sps_temporal_mvp_enabled_flag
    w.bit(0);           // strong_intra_smoothing_enabled_flag
    w.bit(0);           // vui_parameters_present_flag
    w.bit(0);           // sps_extension_present_flag
    w.trailing();
    return w.bytes();
  }

  std::vector<uint8_t> pps() const {
    BitWriter w;
    w.ue(0);            // pps_pic_parameter_set_id
    w.ue(0);            // pps_seq_parameter_set_id
    w.bit(0);           // dependent_slice_segments_enabled_flag
    w.bit(0);           // output_flag_present_flag
    w.put(0, 3);        // num_extra_slice_header_bits
    w.bit(0);           // sign_data_hiding_enabled_flag
    w.bit(0);           // cabac_init_present_flag
    w.ue(0);            // num_ref_idx_l0_default_active_minus1
    w.ue(0);            // num_ref_idx_l1_default_active_minus1
    w.se(0);            // init_qp_minus26
    w.bit(0);           // constrained_intra_pred_flag
    w.bit(0);           // transform_skip_enabled_flag
    w.bit(0);           // cu_qp_delta_enabled_flag
    w.se(0);            // pps_cb_qp_offset
    w.se(0);            // pps_cr_qp_offset
    w.bit(0);           // pps_slice_chroma_qp_offsets_present_flag
    w.bit(0);           // weighted_pred_flag
    w.bit(0);           // weighted_bipred_flag
    w.bit(0);           // transquant_bypass_enabled_flag
    w.bit(0);           // tiles_enabled_flag
    w.bit(0);           // entropy_coding_sync_enabled_flag
    w.bit(0);           // pps_loop_filter_across_slices_enabled_flag
    w.bit(1);           // deblocking_filter_control_present_flag
    w.bit(0);           // deblocking_filter_override_enabled_flag
    w.bit(1);           // pps_deblocking_filter_disabled_flag
    w.bit(0);           // pps_scaling_list_data_present_flag
    w.bit(0);           // lists_modification_present_flag
    w.ue(0);            // log2_parallel_merge_level_minus2
    w.bit(0);           // slice_segment_header_extension_present_flag
    w.bit(0);           // pps_extension_present_flag
    w.trailing();
    return w.bytes();
  }

  // VPS, SPS and PPS: the headers a decoder needs, as the first bytes of a stream.
  std::vector<uint8_t> parameter_sets() const {
    std::vector<uint8_t> out;
    append_hevc_nal(out, 32, vps());
    append_hevc_nal(out, 33, sps());
    append_hevc_nal(out, 34, pps());
    return out;
  }

  // One IDR picture from 8-bit 4:2:0 planes: `luma(x, y)` and `chroma(plane, x, y)`
  // (plane 0 = Cb, 1 = Cr) return the samples; positions past the picture repeat its
  // last row and column, which the SPS's cropping hides.
  template <class Luma, class Chroma>
  std::vector<uint8_t> idr(Luma luma, Chroma chroma) const {
    // The slice segment header, then the byte_alignment() that ends it.
    BitWriter hdr;
    hdr.bit(1);         // first_slice_segment_in_pic_flag
    hdr.bit(0);         // no_output_of_prior_pics_flag (an IRAP picture)
    hdr.ue(0);          // slice_pic_parameter_set_id
    hdr.ue(2);          // slice_type: I
    hdr.se(0);          // slice_qp_delta
    hdr.bit(1);         // byte_alignment(): a one, then zeros
    hdr.align_zero();
    std::vector<uint8_t> rbsp = hdr.bytes();

    const int cw = (width + 1) / 2, ch = (height + 1) / 2;
    auto clampx = [&](int x, int hi) { return std::min(std::max(x, 0), hi - 1); };
    const int ncu = ctus_x() * ctus_y();
    Cabac cabac(rbsp);
    for (int cu = 0; cu < ncu; ++cu) {
      const int mx = cu % ctus_x(), my = cu / ctus_x();
      cabac.encode_split_cu_flag0();
      cabac.encode_terminate(1);                    // pcm_flag
      cabac.flush();                                // EncodeFlush; the last bit written was a one
      cabac.align_zero();                           // pcm_alignment_zero_bit
      for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) cabac.byte(luma(clampx(mx * 16 + x, width), clampx(my * 16 + y, height)));
      for (int plane = 0; plane < 2; ++plane)
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x) cabac.byte(chroma(plane, clampx(mx * 8 + x, cw), clampx(my * 8 + y, ch)));
      cabac.restart();                              // the engine is initialised after the samples
      cabac.encode_terminate(cu + 1 == ncu);        // end_of_slice_segment_flag
      if (cu + 1 == ncu) cabac.flush();             // its last bit is the rbsp_stop_one_bit
    }
    cabac.finish();
    std::vector<uint8_t> out;
    append_hevc_nal(out, 19, rbsp);                 // IDR_W_RADL
    return out;
  }

 private:
  int level_idc() const {
    // Table A.8: the lowest level whose luma picture size and sample rate fit.
    struct L {
      int idc;
      long max_luma_ps, max_luma_sr;
    };
    static const L levels[] = {{30, 36864, 552960},     {60, 122880, 3686400},     {63, 245760, 7372800},     {90, 552960, 16588800},
                               {93, 983040, 33177600},  {120, 2228224, 66846720},  {123, 2228224, 133693440}, {150, 8912896, 267386880},
                               {153, 8912896, 534773760}, {156, 8912896, 1069547520}, {180, 35651584, 1069547520}, {183, 35651584, 2139095040},
                               {186, 35651584, 4278190080L}};
    const long ps = static_cast<long>(ctus_x()) * 16 * ctus_y() * 16;
    const long sr = ps * fps_num / std::max(fps_den, 1);
    for (const L& l : levels)
      if (ps <= l.max_luma_ps && sr <= l.max_luma_sr) return l.idc;
    return 186;
  }

  // profile_tier_level(1, 0): Main profile, no sub-layers.
  void profile_tier_level(BitWriter& w) const {
    w.put(0, 2);        // general_profile_space
    w.bit(0);           // general_tier_flag
    w.put(1, 5);        // general_profile_idc: Main
    for (int i = 0; i < 32; ++i) w.bit(i == 1 || i == 2);   // general_profile_compatibility_flag: Main, Main 10
    w.bit(1);           // general_progressive_source_flag
    w.bit(0);           // general_interlaced_source_flag
    w.bit(0);           // general_non_packed_constraint_flag
    w.bit(1);           // general_frame_only_constraint_flag
    w.put(0, 32);       // 43 reserved bits and general_inbld_flag / reserved bit: 44 zero bits in all
    w.put(0, 12);
    w.put(static_cast<uint32_t>(level_idc()), 8);   // general_level_idc
  }

  // The CABAC encoder of H.265 9.3.4.3, with the one context split_cu_flag uses.
  class Cabac {
   public:
    explicit Cabac(std::vector<uint8_t>& out) : out_(out) { restart(); }
    void restart() {
      low_ = 0;
      range_ = 510;
      first_ = true;
      outstanding_ = 0;
    }
    // split_cu_flag with every coding unit of depth 0, so context 0 (initValue 139 at
    // SliceQpY 26: pStateIdx 0, valMps 0), and the bin is 0, the most probable symbol.
    void encode_split_cu_flag0() {
      static const uint8_t lps_q3[63] = {240, 227, 216, 205, 195, 185, 175, 166, 158, 150, 142, 135, 128, 122, 116, 110,
                                         104, 99,  94,  89,  85,  80,  76,  72,  69,  65,  62,  59,  56,  53,  50,  48,
                                         45,  43,  41,  39,  37,  35,  33,  31,  30,  28,  27,  25,  24,  23,  22,  21,
                                         20,  19,  18,  17,  16,  15,  14,  14,  13,  12,  12,  11,  11,  10,  9};
      const int q = (range_ >> 6) & 3;
      // Every coding tree unit starts from a freshly initialised engine (range 510 or,
      // after end_of_slice_segment_flag = 0, 508): column 3 of rangeTabLPS.
      const uint32_t lps = q == 3 ? lps_q3[state_] : fallback(state_, q);
      range_ -= lps;
      if (state_ < 62) ++state_;                    // transIdxMps
      renorm();
    }
    void encode_terminate(int bin) {
      range_ -= 2;
      if (bin) {
        low_ += range_;
      } else {
        renorm();
      }
    }
    // EncodeFlush (9.3.4.3.5): the last bit written is a one.
    void flush() {
      range_ = 2;
      renorm();
      put_bit((low_ >> 9) & 1);
      write_bit((low_ >> 8) & 1);
      write_bit(1);
    }
    void align_zero() {
      while (nbits_) write_bit(0);
    }
    void byte(uint8_t b) {
      if (nbits_ == 0) {
        out_.push_back(b);
      } else {
        for (int i = 7; i >= 0; --i) write_bit((b >> i) & 1);
      }
    }
    // After the last flush: the stop bit is written; pad to a byte.
    void finish() { align_zero(); }

   private:
    static uint32_t fallback(int, int) { return 6; }   // never reached: the engine is always at range 508..510 at a split flag
    void renorm() {
      while (range_ < 256) {
        if (low_ < 256) {
          put_bit(0);
        } else if (low_ >= 512) {
          low_ -= 512;
          put_bit(1);
        } else {
          low_ -= 256;
          ++outstanding_;
        }
        range_ <<= 1;
        low_ <<= 1;
      }
    }
    void put_bit(int b) {
      if (first_) {
        first_ = false;
      } else {
        write_bit(b);
      }
      while (outstanding_ > 0) {
        write_bit(1 - b);
        --outstanding_;
      }
    }
    void write_bit(int b) {
      cur_ = static_cast<uint8_t>(cur_ << 1 | (b & 1));
      if (++nbits_ == 8) {
        out_.push_back(cur_);
        cur_ = 0;
        nbits_ = 0;
      }
    }
    std::vector<uint8_t>& out_;
    uint32_t low_ = 0, range_ = 510;
    bool first_ = true;
    int outstanding_ = 0;
    int state_ = 0;       // pStateIdx of the split_cu_flag context (valMps is 0)
    uint8_t cur_ = 0;
    int nbits_ = 0;
  };
};

}  // namespace vgpu_nvenc
