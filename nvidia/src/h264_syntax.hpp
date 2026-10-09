// H.264 (ITU-T H.264 03/2009) bitstream syntax: Annex B NAL units, the RBSP bit
// reader with Exp-Golomb codes, sequence and picture parameter sets (with the VUI
// fields a decoder front end reports), and slice headers. Written from clauses
// 7.3 and 7.4 and nothing else; shared by VirtualGPU's NVDEC (nvcuvid_api.cpp),
// the software decoder behind it (h264_decode.cpp) and the tests.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vgpu_h264 {

#include "h264_tables.inc"

// ---- bit reader ---------------------------------------------------------------------------
class BitReader {
 public:
  BitReader() = default;
  BitReader(const uint8_t* d, size_t n) : d_(d), n_(n) {}
  uint32_t bit() {
    if (pos_ >= n_ * 8) {
      ++pos_;
      overrun_ = true;
      return 0;
    }
    const uint32_t b = (d_[pos_ >> 3] >> (7 - (pos_ & 7))) & 1;
    ++pos_;
    return b;
  }
  uint32_t u(int bits) {
    uint32_t v = 0;
    for (int i = 0; i < bits; ++i) v = v << 1 | bit();
    return v;
  }
  uint32_t peek(int bits) const {
    BitReader c = *this;
    return c.u(bits);
  }
  void skip(int bits) { pos_ += static_cast<size_t>(bits); }
  uint32_t ue() {
    int zeros = 0;
    while (!bit()) {
      if (++zeros > 32 || overrun_) {
        overrun_ = true;
        return 0;
      }
    }
    if (zeros == 32) return 0xFFFFFFFFu;
    return ((1u << zeros) - 1) + u(zeros);
  }
  int32_t se() {
    const uint32_t k = ue();
    return (k & 1) ? static_cast<int32_t>((k + 1) >> 1) : -static_cast<int32_t>(k >> 1);
  }
  size_t pos() const { return pos_; }
  void set_pos(size_t p) { pos_ = p; }
  size_t size_bits() const { return n_ * 8; }
  bool overrun() const { return overrun_; }
  bool byte_aligned() const { return (pos_ & 7) == 0; }
  void align() { pos_ = (pos_ + 7) & ~size_t{7}; }
  const uint8_t* data() const { return d_; }
  size_t bytes() const { return n_; }
  // more_rbsp_data(): is there anything before the rbsp_trailing_bits (the last 1 bit)?
  bool more_rbsp_data() const {
    size_t last = n_;
    while (last > 0 && d_[last - 1] == 0) --last;   // cabac_zero_words / trailing zero bytes
    if (last == 0) return false;
    const uint8_t b = d_[last - 1];
    int tz = 0;
    while (!((b >> tz) & 1)) ++tz;
    const size_t stop_bit = (last - 1) * 8 + static_cast<size_t>(7 - tz);
    return pos_ < stop_bit;
  }

 private:
  const uint8_t* d_ = nullptr;
  size_t n_ = 0;
  size_t pos_ = 0;
  bool overrun_ = false;
};

// ---- Annex B ------------------------------------------------------------------------------
struct NalRef {
  size_t start_code = 0;   // offset of the first byte of the start code (the 00 before 00 01, if any)
  size_t payload = 0;      // offset of the NAL header byte
  size_t end = 0;          // one past the last byte of the NAL unit (trailing zero bytes removed)
  int ref_idc = 0, type = 0;
};

inline std::vector<NalRef> split_annexb(const uint8_t* d, size_t n) {
  std::vector<NalRef> out;
  size_t i = 0;
  while (i + 3 <= n) {
    if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1) {
      NalRef r;
      r.start_code = (i > 0 && d[i - 1] == 0) ? i - 1 : i;
      r.payload = i + 3;
      if (!out.empty()) {
        size_t e = r.start_code;
        // the previous NAL ends at this start code; trailing_zero_8bits belong to no NAL
        while (e > out.back().payload && d[e - 1] == 0) --e;
        out.back().end = e;
      }
      if (r.payload < n) {
        r.ref_idc = d[r.payload] >> 5 & 3;
        r.type = d[r.payload] & 31;
      }
      out.push_back(r);
      i += 3;
    } else {
      ++i;
    }
  }
  if (!out.empty()) {
    size_t e = n;
    while (e > out.back().payload && d[e - 1] == 0) --e;
    out.back().end = e;
  }
  return out;
}

// The RBSP of a NAL unit: emulation prevention bytes (00 00 03) removed. `skip` bytes of header are dropped.
inline std::vector<uint8_t> unescape(const uint8_t* d, size_t n, size_t skip) {
  std::vector<uint8_t> out;
  out.reserve(n);
  int zeros = 0;
  for (size_t i = skip; i < n; ++i) {
    if (zeros >= 2 && d[i] == 3) {
      zeros = 0;
      continue;
    }
    out.push_back(d[i]);
    zeros = d[i] == 0 ? zeros + 1 : 0;
  }
  return out;
}

// ---- parameter sets -----------------------------------------------------------------------
struct Sps {
  bool valid = false;
  int profile_idc = 0, constraint_flags = 0, level_idc = 0, id = 0;
  int chroma_format_idc = 1, separate_colour_plane = 0, bit_depth_luma = 8, bit_depth_chroma = 8, qpprime_bypass = 0;
  bool scaling_matrix_present = false;
  bool list_present[12] = {};
  bool list_use_default[12] = {};
  uint8_t list4[6][16] = {};   // as coded (zig-zag order), valid where list_present
  uint8_t list8[6][64] = {};
  int log2_max_frame_num = 4, poc_type = 0, log2_max_poc_lsb = 4;
  int delta_pic_order_always_zero = 0, offset_for_non_ref_pic = 0, offset_for_top_to_bottom = 0;
  std::vector<int> offset_for_ref_frame;
  int num_ref_frames = 0, gaps_allowed = 0;
  int width_mbs = 0, height_map_units = 0;
  int frame_mbs_only = 1, mbaff = 0, direct_8x8_inference = 0;
  int crop[4] = {0, 0, 0, 0};   // left, right, top, bottom in crop units
  bool cropping = false;
  // VUI
  bool vui_present = false;
  bool aspect_present = false;
  int aspect_idc = 0, sar_w = 0, sar_h = 0;
  bool video_signal_present = false;
  int video_format = 5, video_full_range = 0;
  bool colour_description_present = false;
  int colour_primaries = 2, transfer = 2, matrix = 2;
  bool timing_present = false;
  uint32_t num_units_in_tick = 0, time_scale = 0;
  int fixed_frame_rate = 0;
  bool hrd_present = false;
  uint32_t hrd_bit_rate = 0;       // bit rate of the first CPB, in bits per second
  int cpb_removal_delay_len = 0, dpb_output_delay_len = 0;
  bool pic_struct_present = false;
  bool restriction_present = false;
  int num_reorder_frames = 0, max_dec_frame_buffering = 0;
  int frame_height_mbs() const { return height_map_units * (2 - frame_mbs_only); }
  int chroma_array_type() const { return separate_colour_plane ? 0 : chroma_format_idc; }
};

struct Pps {
  bool valid = false;
  int id = 0, sps_id = 0;
  int cabac = 0, bottom_field_pic_order_in_frame_present = 0;
  int num_slice_groups = 1, slice_group_map_type = 0;
  int slice_group_change_rate = 0;
  int num_ref_idx_default[2] = {1, 1};
  int weighted_pred = 0, weighted_bipred_idc = 0;
  int pic_init_qp = 26, pic_init_qs = 26;
  int chroma_qp_offset[2] = {0, 0};
  int deblocking_control_present = 0, constrained_intra_pred = 0, redundant_pic_cnt_present = 0;
  int transform_8x8_mode = 0;
  bool scaling_matrix_present = false;
  bool list_present[12] = {};
  bool list_use_default[12] = {};
  uint8_t list4[6][16] = {};
  uint8_t list8[6][64] = {};
  bool has_extension = false;
};

inline void parse_scaling_list(BitReader& b, uint8_t* list, int size, bool* use_default) {
  int last = 8, next = 8;
  *use_default = false;
  for (int j = 0; j < size; ++j) {
    if (next != 0) {
      const int delta = b.se();
      next = (last + delta + 256) % 256;
      *use_default = (j == 0 && next == 0);
    }
    list[j] = static_cast<uint8_t>(next == 0 ? last : next);
    last = list[j];
  }
}

inline void parse_hrd(BitReader& b, Sps* s) {
  const int cpb_cnt = static_cast<int>(b.ue()) + 1;
  const int rate_scale = static_cast<int>(b.u(4));
  b.u(4);
  for (int i = 0; i < cpb_cnt && i < 32; ++i) {
    const uint64_t rate = (static_cast<uint64_t>(b.ue()) + 1) << (6 + rate_scale);
    b.ue();
    b.u(1);
    if (i == 0 && !s->hrd_present) s->hrd_bit_rate = static_cast<uint32_t>(std::min<uint64_t>(rate, 0xFFFFFFFFu));
  }
  b.u(5);
  s->cpb_removal_delay_len = static_cast<int>(b.u(5)) + 1;
  s->dpb_output_delay_len = static_cast<int>(b.u(5)) + 1;
  b.u(5);
}

inline bool parse_sps(const uint8_t* rbsp, size_t n, Sps* s) {
  BitReader b(rbsp, n);
  *s = Sps{};
  s->profile_idc = static_cast<int>(b.u(8));
  s->constraint_flags = static_cast<int>(b.u(8));
  s->level_idc = static_cast<int>(b.u(8));
  s->id = static_cast<int>(b.ue());
  if (s->id > 31) return false;
  static const int high[] = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135};
  if (std::find(std::begin(high), std::end(high), s->profile_idc) != std::end(high)) {
    s->chroma_format_idc = static_cast<int>(b.ue());
    if (s->chroma_format_idc > 3) return false;
    if (s->chroma_format_idc == 3) s->separate_colour_plane = static_cast<int>(b.u(1));
    s->bit_depth_luma = 8 + static_cast<int>(b.ue());
    s->bit_depth_chroma = 8 + static_cast<int>(b.ue());
    if (s->bit_depth_luma > 14 || s->bit_depth_chroma > 14) return false;
    s->qpprime_bypass = static_cast<int>(b.u(1));
    s->scaling_matrix_present = b.u(1) != 0;
    if (s->scaling_matrix_present) {
      const int count = s->chroma_format_idc != 3 ? 8 : 12;
      for (int i = 0; i < count; ++i) {
        s->list_present[i] = b.u(1) != 0;
        if (!s->list_present[i]) continue;
        bool def = false;
        if (i < 6) parse_scaling_list(b, s->list4[i], 16, &def);
        else parse_scaling_list(b, s->list8[i - 6], 64, &def);
        s->list_use_default[i] = def;
      }
    }
  }
  s->log2_max_frame_num = static_cast<int>(b.ue()) + 4;
  s->poc_type = static_cast<int>(b.ue());
  if (s->log2_max_frame_num > 16 || s->poc_type > 2) return false;
  if (s->poc_type == 0) {
    s->log2_max_poc_lsb = static_cast<int>(b.ue()) + 4;
    if (s->log2_max_poc_lsb > 16) return false;
  } else if (s->poc_type == 1) {
    s->delta_pic_order_always_zero = static_cast<int>(b.u(1));
    s->offset_for_non_ref_pic = b.se();
    s->offset_for_top_to_bottom = b.se();
    const uint32_t cnt = b.ue();
    if (cnt > 255) return false;
    for (uint32_t i = 0; i < cnt; ++i) s->offset_for_ref_frame.push_back(b.se());
  }
  s->num_ref_frames = static_cast<int>(b.ue());
  s->gaps_allowed = static_cast<int>(b.u(1));
  s->width_mbs = static_cast<int>(b.ue()) + 1;
  s->height_map_units = static_cast<int>(b.ue()) + 1;
  s->frame_mbs_only = static_cast<int>(b.u(1));
  if (!s->frame_mbs_only) s->mbaff = static_cast<int>(b.u(1));
  s->direct_8x8_inference = static_cast<int>(b.u(1));
  s->cropping = b.u(1) != 0;
  if (s->cropping)
    for (int i = 0; i < 4; ++i) s->crop[i] = static_cast<int>(std::min<uint32_t>(b.ue(), 1u << 20));
  s->vui_present = b.u(1) != 0;
  if (s->vui_present) {
    s->aspect_present = b.u(1) != 0;
    if (s->aspect_present) {
      s->aspect_idc = static_cast<int>(b.u(8));
      if (s->aspect_idc == 255) {
        s->sar_w = static_cast<int>(b.u(16));
        s->sar_h = static_cast<int>(b.u(16));
      }
    }
    if (b.u(1)) b.u(1);   // overscan
    s->video_signal_present = b.u(1) != 0;
    if (s->video_signal_present) {
      s->video_format = static_cast<int>(b.u(3));
      s->video_full_range = static_cast<int>(b.u(1));
      s->colour_description_present = b.u(1) != 0;
      if (s->colour_description_present) {
        s->colour_primaries = static_cast<int>(b.u(8));
        s->transfer = static_cast<int>(b.u(8));
        s->matrix = static_cast<int>(b.u(8));
      }
    }
    if (b.u(1)) {   // chroma_loc_info_present_flag
      b.ue();
      b.ue();
    }
    s->timing_present = b.u(1) != 0;
    if (s->timing_present) {
      s->num_units_in_tick = b.u(32);
      s->time_scale = b.u(32);
      s->fixed_frame_rate = static_cast<int>(b.u(1));
    }
    const bool nal_hrd = b.u(1) != 0;
    if (nal_hrd) {
      parse_hrd(b, s);
      s->hrd_present = true;
    }
    const bool vcl_hrd = b.u(1) != 0;
    if (vcl_hrd) {
      parse_hrd(b, s);
      s->hrd_present = true;
    }
    if (nal_hrd || vcl_hrd) b.u(1);   // low_delay_hrd_flag
    s->pic_struct_present = b.u(1) != 0;
    s->restriction_present = b.u(1) != 0;
    if (s->restriction_present) {
      b.u(1);
      b.ue();
      b.ue();
      b.ue();
      b.ue();
      s->num_reorder_frames = static_cast<int>(b.ue());
      s->max_dec_frame_buffering = static_cast<int>(b.ue());
    }
  }
  if (b.overrun()) {
    // A truncated VUI is tolerated (some encoders write short ones); the rest was read.
    s->restriction_present = false;
  }
  s->valid = s->width_mbs > 0 && s->width_mbs <= 1024 && s->height_map_units > 0 && s->height_map_units <= 1024;
  return s->valid;
}

inline bool parse_pps(const uint8_t* rbsp, size_t n, const Sps* const* sps_by_id, Pps* p) {
  BitReader b(rbsp, n);
  *p = Pps{};
  p->id = static_cast<int>(b.ue());
  p->sps_id = static_cast<int>(b.ue());
  if (p->id > 255 || p->sps_id > 31) return false;
  p->cabac = static_cast<int>(b.u(1));
  p->bottom_field_pic_order_in_frame_present = static_cast<int>(b.u(1));
  p->num_slice_groups = static_cast<int>(b.ue()) + 1;
  if (p->num_slice_groups > 8) return false;
  if (p->num_slice_groups > 1) {
    p->slice_group_map_type = static_cast<int>(b.ue());
    if (p->slice_group_map_type == 0) {
      for (int i = 0; i < p->num_slice_groups; ++i) b.ue();
    } else if (p->slice_group_map_type == 2) {
      for (int i = 0; i < p->num_slice_groups - 1; ++i) {
        b.ue();
        b.ue();
      }
    } else if (p->slice_group_map_type >= 3 && p->slice_group_map_type <= 5) {
      b.u(1);
      p->slice_group_change_rate = static_cast<int>(b.ue()) + 1;
    } else if (p->slice_group_map_type == 6) {
      const uint32_t cnt = b.ue() + 1;
      int bits = 0;
      while ((1 << bits) < p->num_slice_groups) ++bits;
      for (uint32_t i = 0; i < cnt && !b.overrun(); ++i) b.u(bits);
    }
  }
  p->num_ref_idx_default[0] = static_cast<int>(b.ue()) + 1;
  p->num_ref_idx_default[1] = static_cast<int>(b.ue()) + 1;
  p->weighted_pred = static_cast<int>(b.u(1));
  p->weighted_bipred_idc = static_cast<int>(b.u(2));
  p->pic_init_qp = 26 + b.se();
  p->pic_init_qs = 26 + b.se();
  p->chroma_qp_offset[0] = b.se();
  p->chroma_qp_offset[1] = p->chroma_qp_offset[0];
  p->deblocking_control_present = static_cast<int>(b.u(1));
  p->constrained_intra_pred = static_cast<int>(b.u(1));
  p->redundant_pic_cnt_present = static_cast<int>(b.u(1));
  if (b.more_rbsp_data()) {
    p->has_extension = true;
    p->transform_8x8_mode = static_cast<int>(b.u(1));
    p->scaling_matrix_present = b.u(1) != 0;
    if (p->scaling_matrix_present) {
      const Sps* sps = p->sps_id < 32 ? sps_by_id[p->sps_id] : nullptr;
      const int chroma_format = sps ? sps->chroma_format_idc : 1;
      const int count = 6 + (chroma_format != 3 ? 2 : 6) * p->transform_8x8_mode;
      for (int i = 0; i < count; ++i) {
        p->list_present[i] = b.u(1) != 0;
        if (!p->list_present[i]) continue;
        bool def = false;
        if (i < 6) parse_scaling_list(b, p->list4[i], 16, &def);
        else parse_scaling_list(b, p->list8[i - 6], 64, &def);
        p->list_use_default[i] = def;
      }
    }
    p->chroma_qp_offset[1] = b.se();
  }
  p->valid = !b.overrun() && p->num_ref_idx_default[0] <= 32 && p->num_ref_idx_default[1] <= 32 && p->pic_init_qp >= -26 - 6 * 6 && p->pic_init_qp <= 51;
  return p->valid;
}

// The scaling matrices a picture uses, in raster order (weightScale4x4[6][16] and
// weightScale8x8[2][64] for 4:2:0): the lists of the PPS, else the SPS, else flat 16,
// with the fall-back rules of Table 7-2 for lists that are not sent.
inline void resolve_scaling(const Sps& sps, const Pps& pps, uint8_t ws4[6][16], uint8_t ws8[2][64]) {
  uint8_t l4[6][16];
  uint8_t l8[6][64];   // 8x8 lists in zig-zag order (index 0 intra Y, 1 inter Y, then Cb/Cr for 4:4:4)
  const bool flat = !sps.scaling_matrix_present && !pps.scaling_matrix_present;
  if (flat) {
    std::memset(l4, 16, sizeof l4);
    std::memset(l8, 16, sizeof l8);
  } else {
    // SPS lists first (fall-back rule set A), then the PPS lists over them (rule set B, or A if the SPS has no matrix).
    uint8_t s4[6][16], s8[6][64];
    if (!sps.scaling_matrix_present) {
      std::memset(s4, 16, sizeof s4);
      std::memset(s8, 16, sizeof s8);
    } else {
      for (int i = 0; i < 6; ++i) {
        if (sps.list_present[i] && !sps.list_use_default[i]) std::memcpy(s4[i], sps.list4[i], 16);
        else if (sps.list_present[i] || i == 0 || i == 3) std::memcpy(s4[i], i < 3 ? kDefault4x4Intra : kDefault4x4Inter, 16);
        else std::memcpy(s4[i], s4[i - 1], 16);
      }
      for (int k = 0; k < 6; ++k) {
        const int i = 6 + k;
        const bool intra = (k % 2) == 0;
        if (sps.list_present[i] && !sps.list_use_default[i]) std::memcpy(s8[k], sps.list8[k], 64);
        else if (sps.list_present[i] || k < 2) std::memcpy(s8[k], intra ? kDefault8x8Intra : kDefault8x8Inter, 64);
        else std::memcpy(s8[k], s8[k - 2], 64);
      }
    }
    if (!pps.scaling_matrix_present) {
      std::memcpy(l4, s4, sizeof l4);
      std::memcpy(l8, s8, sizeof l8);
    } else {
      const bool rule_b = sps.scaling_matrix_present;
      for (int i = 0; i < 6; ++i) {
        if (pps.list_present[i] && !pps.list_use_default[i]) std::memcpy(l4[i], pps.list4[i], 16);
        else if (pps.list_present[i]) std::memcpy(l4[i], i < 3 ? kDefault4x4Intra : kDefault4x4Inter, 16);
        else if (i == 0 || i == 3) std::memcpy(l4[i], rule_b ? s4[i] : (i == 0 ? kDefault4x4Intra : kDefault4x4Inter), 16);
        else std::memcpy(l4[i], l4[i - 1], 16);
      }
      for (int k = 0; k < 6; ++k) {
        const int i = 6 + k;
        const bool intra = (k % 2) == 0;
        if (pps.list_present[i] && !pps.list_use_default[i]) std::memcpy(l8[k], pps.list8[k], 64);
        else if (pps.list_present[i]) std::memcpy(l8[k], intra ? kDefault8x8Intra : kDefault8x8Inter, 64);
        else if (k < 2) std::memcpy(l8[k], rule_b ? s8[k] : (intra ? kDefault8x8Intra : kDefault8x8Inter), 64);
        else std::memcpy(l8[k], l8[k - 2], 64);
      }
    }
  }
  for (int i = 0; i < 6; ++i)
    for (int k = 0; k < 16; ++k) ws4[i][kZigzag4x4[k]] = l4[i][k];
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 64; ++k) ws8[i][kZigzag8x8[k]] = l8[i][k];
}

// ---- slice header -------------------------------------------------------------------------
enum SliceType { kP = 0, kB = 1, kI = 2, kSP = 3, kSI = 4 };

struct PredWeight {
  int luma_log2_denom = 0, chroma_log2_denom = 0;
  // [list][ref idx]
  int luma_weight[2][32], luma_offset[2][32];
  int chroma_weight[2][32][2], chroma_offset[2][32][2];
  bool luma_flag[2][32], chroma_flag[2][32];
};

struct Mmco {
  int op = 0, a = 0, b = 0;
};

struct RplMod {
  int idc = 0;
  uint32_t value = 0;
};

// What the slice header parse needs from the active parameter sets (or, for NVDEC's
// cuvidDecodePicture, from the picture parameters the application hands over).
struct HeaderCtx {
  int log2_max_frame_num = 4, poc_type = 0, log2_max_poc_lsb = 4, delta_pic_order_always_zero = 0;
  int frame_mbs_only = 1, chroma_array_type = 1, separate_colour_plane = 0;
  int bottom_field_pic_order_in_frame_present = 0, redundant_pic_cnt_present = 0;
  int num_ref_idx_default[2] = {1, 1};
  int weighted_pred = 0, weighted_bipred_idc = 0;
  int cabac = 0, deblocking_control_present = 0, pic_init_qp = 26;
  int num_slice_groups = 1, slice_group_map_type = 0;
  int pic_size_in_map_units = 0, slice_group_change_rate = 1;
};

struct SliceHeader {
  int nal_ref_idc = 0, nal_unit_type = 1;
  int first_mb = 0, slice_type = 0, pps_id = 0, colour_plane = 0, frame_num = 0;
  int field_pic = 0, bottom_field = 0, idr_pic_id = 0;
  int poc_lsb = 0, delta_poc_bottom = 0, delta_poc[2] = {0, 0};
  int redundant_pic_cnt = 0, direct_spatial_mv_pred = 0;
  int num_ref_idx_active[2] = {0, 0};
  std::vector<RplMod> rpl_mod[2];
  PredWeight pw;
  bool has_pred_weight = false;
  bool no_output_of_prior_pics = false, long_term_reference = false, adaptive_marking = false;
  std::vector<Mmco> mmco;
  int cabac_init_idc = 0, slice_qp_delta = 0, qp = 26;
  int disable_deblocking_filter_idc = 0, alpha_c0_offset = 0, beta_offset = 0;
  int slice_group_change_cycle = 0;
  size_t data_bit_pos = 0;   // where slice_data() starts in the RBSP
  bool idr() const { return nal_unit_type == 5; }
  bool is_b() const { return slice_type == kB; }
  bool is_intra() const { return slice_type == kI || slice_type == kSI; }
  bool is_p() const { return slice_type == kP || slice_type == kSP; }
};

inline bool parse_slice_header(BitReader& b, const HeaderCtx& c, int nal_ref_idc, int nal_unit_type, SliceHeader* h) {
  *h = SliceHeader{};
  h->nal_ref_idc = nal_ref_idc;
  h->nal_unit_type = nal_unit_type;
  h->first_mb = static_cast<int>(b.ue());
  const uint32_t st = b.ue();
  if (st > 9) return false;
  h->slice_type = static_cast<int>(st % 5);
  h->pps_id = static_cast<int>(b.ue());
  if (c.separate_colour_plane) h->colour_plane = static_cast<int>(b.u(2));
  h->frame_num = static_cast<int>(b.u(c.log2_max_frame_num));
  if (!c.frame_mbs_only) {
    h->field_pic = static_cast<int>(b.u(1));
    if (h->field_pic) h->bottom_field = static_cast<int>(b.u(1));
  }
  if (nal_unit_type == 5) h->idr_pic_id = static_cast<int>(b.ue());
  if (c.poc_type == 0) {
    h->poc_lsb = static_cast<int>(b.u(c.log2_max_poc_lsb));
    if (c.bottom_field_pic_order_in_frame_present && !h->field_pic) h->delta_poc_bottom = b.se();
  }
  if (c.poc_type == 1 && !c.delta_pic_order_always_zero) {
    h->delta_poc[0] = b.se();
    if (c.bottom_field_pic_order_in_frame_present && !h->field_pic) h->delta_poc[1] = b.se();
  }
  if (c.redundant_pic_cnt_present) h->redundant_pic_cnt = static_cast<int>(b.ue());
  if (h->slice_type == kB) h->direct_spatial_mv_pred = static_cast<int>(b.u(1));
  h->num_ref_idx_active[0] = c.num_ref_idx_default[0];
  h->num_ref_idx_active[1] = c.num_ref_idx_default[1];
  if (h->is_p() || h->is_b()) {
    if (b.u(1)) {
      h->num_ref_idx_active[0] = static_cast<int>(b.ue()) + 1;
      if (h->is_b()) h->num_ref_idx_active[1] = static_cast<int>(b.ue()) + 1;
    }
    if (h->num_ref_idx_active[0] > 32 || h->num_ref_idx_active[1] > 32) return false;
    if (!h->is_b()) h->num_ref_idx_active[1] = 0;
  } else {
    h->num_ref_idx_active[0] = h->num_ref_idx_active[1] = 0;
  }
  // ref_pic_list_modification()
  if (!h->is_intra()) {
    for (int l = 0; l < (h->is_b() ? 2 : 1); ++l) {
      if (!b.u(1)) continue;
      for (int guard = 0; guard < 66; ++guard) {
        RplMod m;
        m.idc = static_cast<int>(b.ue());
        if (m.idc == 3) break;
        if (m.idc > 5) return false;
        m.value = b.ue();
        h->rpl_mod[l].push_back(m);
      }
    }
  }
  if ((c.weighted_pred && h->is_p()) || (c.weighted_bipred_idc == 1 && h->is_b())) {
    h->has_pred_weight = true;
    PredWeight& w = h->pw;
    w.luma_log2_denom = static_cast<int>(b.ue());
    if (c.chroma_array_type != 0) w.chroma_log2_denom = static_cast<int>(b.ue());
    if (w.luma_log2_denom > 7 || w.chroma_log2_denom > 7) return false;
    for (int l = 0; l < (h->is_b() ? 2 : 1); ++l) {
      for (int i = 0; i < h->num_ref_idx_active[l]; ++i) {
        w.luma_flag[l][i] = b.u(1) != 0;
        w.luma_weight[l][i] = 1 << w.luma_log2_denom;
        w.luma_offset[l][i] = 0;
        if (w.luma_flag[l][i]) {
          w.luma_weight[l][i] = b.se();
          w.luma_offset[l][i] = b.se();
        }
        w.chroma_flag[l][i] = false;
        for (int j = 0; j < 2; ++j) {
          w.chroma_weight[l][i][j] = 1 << w.chroma_log2_denom;
          w.chroma_offset[l][i][j] = 0;
        }
        if (c.chroma_array_type != 0) {
          w.chroma_flag[l][i] = b.u(1) != 0;
          if (w.chroma_flag[l][i])
            for (int j = 0; j < 2; ++j) {
              w.chroma_weight[l][i][j] = b.se();
              w.chroma_offset[l][i][j] = b.se();
            }
        }
      }
    }
  }
  if (nal_ref_idc != 0) {
    if (nal_unit_type == 5) {
      h->no_output_of_prior_pics = b.u(1) != 0;
      h->long_term_reference = b.u(1) != 0;
    } else {
      h->adaptive_marking = b.u(1) != 0;
      if (h->adaptive_marking) {
        for (int guard = 0; guard < 70; ++guard) {
          Mmco m;
          m.op = static_cast<int>(b.ue());
          if (m.op == 0) break;
          if (m.op > 6) return false;
          if (m.op == 1 || m.op == 3) m.a = static_cast<int>(b.ue());
          if (m.op == 2) m.a = static_cast<int>(b.ue());
          if (m.op == 3 || m.op == 6) m.b = static_cast<int>(b.ue());
          if (m.op == 4) m.a = static_cast<int>(b.ue());
          h->mmco.push_back(m);
        }
      }
    }
  }
  if (c.cabac && !h->is_intra()) {
    h->cabac_init_idc = static_cast<int>(b.ue());
    if (h->cabac_init_idc > 2) return false;
  }
  h->slice_qp_delta = b.se();
  h->qp = c.pic_init_qp + h->slice_qp_delta;
  if (h->qp < 0 || h->qp > 51) return false;   // SliceQPY ranges over -QpBdOffsetY..51; 8-bit samples only here
  if (h->slice_type == kSP || h->slice_type == kSI) {
    if (h->slice_type == kSP) b.u(1);
    b.se();
  }
  if (c.deblocking_control_present) {
    h->disable_deblocking_filter_idc = static_cast<int>(b.ue());
    if (h->disable_deblocking_filter_idc > 2) return false;
    if (h->disable_deblocking_filter_idc != 1) {
      const int32_t a = b.se(), bt = b.se();
      if (a < -6 || a > 6 || bt < -6 || bt > 6) return false;
      h->alpha_c0_offset = a * 2;
      h->beta_offset = bt * 2;
    }
  }
  if (c.num_slice_groups > 1 && c.slice_group_map_type >= 3 && c.slice_group_map_type <= 5) {
    const int units = c.pic_size_in_map_units;
    const int rate = std::max(1, c.slice_group_change_rate);
    const int v = (units + rate - 1) / rate + 1;
    int bits = 0;
    while ((1 << bits) < v) ++bits;
    h->slice_group_change_cycle = static_cast<int>(b.u(bits));
  }
  h->data_bit_pos = b.pos();
  return !b.overrun();
}

}  // namespace vgpu_h264
