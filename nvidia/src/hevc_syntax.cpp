// See hevc_syntax.hpp.
#include "hevc_syntax.hpp"

#include <cstdio>

namespace vgpu_hevc {

int ceil_log2(uint32_t v) {
  int n = 0;
  while ((1ull << n) < v) ++n;
  return n;
}

// ---- scaling lists (7.3.4, 7.4.5, Tables 7-5 and 7-6) ---------------------------------------
namespace {
const uint8_t kDefaultIntra8[64] = {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20, 21, 20, 19, 21, 24, 22, 22, 24,
                                    24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31, 29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115};
const uint8_t kDefaultInter8[64] = {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24,
                                    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28, 28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91};
}  // namespace

void ScalingLists::set_default() {
  std::memset(list, 16, sizeof list);
  for (int s = 1; s < 4; ++s)
    for (int m = 0; m < 6; ++m) std::memcpy(list[s][m], m < 3 ? kDefaultIntra8 : kDefaultInter8, 64);
  for (auto& row : dc)
    for (int& v : row) v = 16;
}

bool ScalingLists::parse(BitReader& b) {
  for (int size_id = 0; size_id < 4; ++size_id) {
    for (int matrix_id = 0; matrix_id < 6; matrix_id += (size_id == 3) ? 3 : 1) {
      const int coef_num = std::min(64, 1 << (4 + (size_id << 1)));
      if (!b.bit()) {   // scaling_list_pred_mode_flag
        const uint32_t delta = b.ue();
        if (delta > static_cast<uint32_t>(size_id == 3 ? matrix_id / 3 : matrix_id)) return false;
        if (delta == 0) {
          if (size_id == 0) std::memset(list[0][matrix_id], 16, 16);
          else std::memcpy(list[size_id][matrix_id], matrix_id < 3 ? kDefaultIntra8 : kDefaultInter8, 64);
          if (size_id > 1) dc[size_id - 2][matrix_id] = 16;
        } else {
          const int ref = matrix_id - static_cast<int>(delta) * (size_id == 3 ? 3 : 1);
          std::memcpy(list[size_id][matrix_id], list[size_id][ref], static_cast<size_t>(coef_num));
          if (size_id > 1) dc[size_id - 2][matrix_id] = dc[size_id - 2][ref];
        }
      } else {
        int next = 8;
        if (size_id > 1) {
          const int d = b.se();
          if (d < -7 || d > 247) return false;
          next = d + 8;
          dc[size_id - 2][matrix_id] = next;
        }
        for (int i = 0; i < coef_num; ++i) {
          const int delta = b.se();
          if (delta < -128 || delta > 127) return false;
          next = (next + delta + 256) % 256;
          list[size_id][matrix_id][i] = static_cast<uint8_t>(next);
        }
      }
      if (b.overrun()) return false;
    }
  }
  return true;
}

// ---- profile_tier_level, short-term RPS, VUI ----------------------------------------------------
namespace {

void skip_profile_tier_level(BitReader& b, bool profile_present, int max_sub_layers_minus1, int* profile_idc, int* level_idc, int* tier, uint32_t* compat) {
  if (profile_present) {
    b.u(2);
    *tier = static_cast<int>(b.bit());
    *profile_idc = static_cast<int>(b.u(5));
    *compat = b.u(32);
    b.skip(4 + 43 + 1);
  }
  *level_idc = static_cast<int>(b.u(8));
  bool prof[8] = {}, lev[8] = {};
  for (int i = 0; i < max_sub_layers_minus1; ++i) {
    prof[i] = b.bit() != 0;
    lev[i] = b.bit() != 0;
  }
  if (max_sub_layers_minus1 > 0)
    for (int i = max_sub_layers_minus1; i < 8; ++i) b.skip(2);
  for (int i = 0; i < max_sub_layers_minus1; ++i) {
    if (prof[i]) b.skip(88);
    if (lev[i]) b.skip(8);
  }
}

// st_ref_pic_set( idx ) with `sets` holding the candidates 0..idx-1 of the SPS. `in_slice` when idx == num_short_term_ref_pic_sets.
bool parse_st_rps(BitReader& b, int idx, int num_sets, const std::vector<StRps>& sets, StRps* out, int* ref_num_delta = nullptr) {
  bool inter = false;
  if (idx != 0) inter = b.bit() != 0;
  StRps r;
  if (inter) {
    uint32_t delta_idx_minus1 = 0;
    if (idx == num_sets) delta_idx_minus1 = b.ue();
    if (delta_idx_minus1 >= static_cast<uint32_t>(idx)) return false;
    const int ref_idx = idx - static_cast<int>(delta_idx_minus1 + 1);
    const int sign = static_cast<int>(b.bit());
    const uint32_t abs_delta_minus1 = b.ue();
    if (abs_delta_minus1 > 32767) return false;
    const int delta_rps = (1 - 2 * sign) * static_cast<int>(abs_delta_minus1 + 1);
    const StRps& ref = sets[static_cast<size_t>(ref_idx)];
    const int n = ref.num_delta();
    if (ref_num_delta) *ref_num_delta = n;
    bool used[33] = {}, use_delta[33] = {};
    for (int j = 0; j <= n; ++j) {
      used[j] = b.bit() != 0;
      use_delta[j] = true;
      if (!used[j]) use_delta[j] = b.bit() != 0;
    }
    int i = 0;
    for (int j = ref.num_pos - 1; j >= 0; --j) {
      const int dpoc = ref.delta_s1[j] + delta_rps;
      if (dpoc < 0 && use_delta[ref.num_neg + j]) {
        if (i >= 16) return false;
        r.delta_s0[i] = dpoc;
        r.used_s0[i++] = used[ref.num_neg + j];
      }
    }
    if (delta_rps < 0 && use_delta[n]) {
      if (i >= 16) return false;
      r.delta_s0[i] = delta_rps;
      r.used_s0[i++] = used[n];
    }
    for (int j = 0; j < ref.num_neg; ++j) {
      const int dpoc = ref.delta_s0[j] + delta_rps;
      if (dpoc < 0 && use_delta[j]) {
        if (i >= 16) return false;
        r.delta_s0[i] = dpoc;
        r.used_s0[i++] = used[j];
      }
    }
    r.num_neg = i;
    i = 0;
    for (int j = ref.num_neg - 1; j >= 0; --j) {
      const int dpoc = ref.delta_s0[j] + delta_rps;
      if (dpoc > 0 && use_delta[j]) {
        if (i >= 16) return false;
        r.delta_s1[i] = dpoc;
        r.used_s1[i++] = used[j];
      }
    }
    if (delta_rps > 0 && use_delta[n]) {
      if (i >= 16) return false;
      r.delta_s1[i] = delta_rps;
      r.used_s1[i++] = used[n];
    }
    for (int j = 0; j < ref.num_pos; ++j) {
      const int dpoc = ref.delta_s1[j] + delta_rps;
      if (dpoc > 0 && use_delta[ref.num_neg + j]) {
        if (i >= 16) return false;
        r.delta_s1[i] = dpoc;
        r.used_s1[i++] = used[ref.num_neg + j];
      }
    }
    r.num_pos = i;
  } else {
    const uint32_t nn = b.ue(), np = b.ue();
    if (nn > 16 || np > 16) return false;
    r.num_neg = static_cast<int>(nn);
    r.num_pos = static_cast<int>(np);
    int prev = 0;
    for (int i = 0; i < r.num_neg; ++i) {
      prev -= static_cast<int>(b.ue()) + 1;
      r.delta_s0[i] = prev;
      r.used_s0[i] = b.bit() != 0;
    }
    prev = 0;
    for (int i = 0; i < r.num_pos; ++i) {
      prev += static_cast<int>(b.ue()) + 1;
      r.delta_s1[i] = prev;
      r.used_s1[i] = b.bit() != 0;
    }
  }
  if (b.overrun()) return false;
  *out = r;
  return true;
}

void parse_sub_layer_hrd(BitReader& b, int cpb_cnt, bool sub_pic, int bit_rate_scale, uint32_t* first_rate, bool* have_first) {
  for (int i = 0; i <= cpb_cnt; ++i) {
    const uint32_t br = b.ue();
    b.ue();
    if (sub_pic) {
      b.ue();
      b.ue();
    }
    b.bit();
    if (!*have_first) {
      *have_first = true;
      const uint64_t bps = (static_cast<uint64_t>(br) + 1) << (6 + bit_rate_scale);
      *first_rate = static_cast<uint32_t>(std::min<uint64_t>(bps, 0xFFFFFFFFu));
    }
  }
}

// hrd_parameters( 1, maxNumSubLayersMinus1 ) (E.2.2). Records the bit rate of the first coded CPB of the highest sub-layer.
void parse_hrd(BitReader& b, int max_sub_layers_minus1, uint32_t* bit_rate) {
  bool nal = false, vcl = false, sub_pic = false;
  int bit_rate_scale = 0;
  nal = b.bit() != 0;
  vcl = b.bit() != 0;
  if (nal || vcl) {
    sub_pic = b.bit() != 0;
    if (sub_pic) {
      b.u(8);
      b.u(5);
      b.bit();
      b.u(5);
    }
    bit_rate_scale = static_cast<int>(b.u(4));
    b.u(4);
    if (sub_pic) b.u(4);
    b.u(5);
    b.u(5);
    b.u(5);
  }
  bool have = false;
  for (int i = 0; i <= max_sub_layers_minus1; ++i) {
    const bool general = b.bit() != 0;
    bool within_cvs = true, low_delay = false;
    if (!general) within_cvs = b.bit() != 0;
    if (within_cvs) b.ue();
    else low_delay = b.bit() != 0;
    int cpb_cnt = 0;
    if (!low_delay) cpb_cnt = static_cast<int>(std::min<uint32_t>(b.ue(), 31));
    // the rate of the highest sub-layer wins: restart `have` for each, keeping the last found
    bool h = false;
    uint32_t rate = 0;
    if (nal) parse_sub_layer_hrd(b, cpb_cnt, sub_pic, bit_rate_scale, &rate, &h);
    if (vcl) parse_sub_layer_hrd(b, cpb_cnt, sub_pic, bit_rate_scale, &rate, &h);
    if (h && (i == max_sub_layers_minus1 || !have)) {
      *bit_rate = rate;
      have = true;
    }
    if (b.overrun()) return;
  }
}

void parse_vui(BitReader& b, Sps* s) {
  s->vui_present = true;
  if ((s->aspect_present = b.bit() != 0)) {
    s->aspect_idc = static_cast<int>(b.u(8));
    if (s->aspect_idc == 255) {
      s->sar_w = static_cast<int>(b.u(16));
      s->sar_h = static_cast<int>(b.u(16));
    }
  }
  if (b.bit()) b.bit();   // overscan
  if ((s->video_signal_present = b.bit() != 0)) {
    s->video_format = static_cast<int>(b.u(3));
    s->video_full_range = static_cast<int>(b.bit());
    if ((s->colour_description_present = b.bit() != 0)) {
      s->colour_primaries = static_cast<int>(b.u(8));
      s->transfer = static_cast<int>(b.u(8));
      s->matrix = static_cast<int>(b.u(8));
    }
  }
  if (b.bit()) {   // chroma_loc_info_present_flag
    b.ue();
    b.ue();
  }
  b.bit();   // neutral_chroma_indication_flag
  s->field_seq = b.bit() != 0;
  s->frame_field_info_present = b.bit() != 0;
  if ((s->default_display_window = b.bit() != 0))
    for (int i = 0; i < 4; ++i) s->def_disp_win[i] = static_cast<int>(b.ue());
  if ((s->timing_present = b.bit() != 0)) {
    s->num_units_in_tick = b.u(32);
    s->time_scale = b.u(32);
    if (b.bit()) b.ue();   // vui_poc_proportional_to_timing_flag, vui_num_ticks_poc_diff_one_minus1
    if ((s->hrd_present = b.bit() != 0)) parse_hrd(b, s->max_sub_layers - 1, &s->hrd_bit_rate);
  }
  if ((s->restriction_present = b.bit() != 0)) {
    b.bit();
    b.bit();
    b.bit();
    b.ue();
    b.ue();
    b.ue();
    b.ue();
    b.ue();
  }
}

}  // namespace

// ---- SPS (7.3.2.2) -----------------------------------------------------------------------------
bool parse_sps(const uint8_t* rbsp, size_t n, Sps* s) {
  BitReader b(rbsp, n);
  *s = Sps{};
  s->vps_id = static_cast<int>(b.u(4));
  const int max_sub_layers_minus1 = static_cast<int>(b.u(3));
  if (max_sub_layers_minus1 > 6) return false;
  s->max_sub_layers = max_sub_layers_minus1 + 1;
  b.bit();   // sps_temporal_id_nesting_flag
  skip_profile_tier_level(b, true, max_sub_layers_minus1, &s->profile_idc, &s->level_idc, &s->tier, &s->profile_compat);
  const uint32_t id = b.ue();
  if (id > 15) return false;
  s->id = static_cast<int>(id);
  const uint32_t cf = b.ue();
  if (cf > 3) return false;
  s->chroma_format_idc = static_cast<int>(cf);
  if (cf == 3) s->separate_colour_plane = static_cast<int>(b.bit());
  const uint32_t w = b.ue(), h = b.ue();
  if (w == 0 || h == 0 || w > 16888 || h > 16888) return false;
  s->width = static_cast<int>(w);
  s->height = static_cast<int>(h);
  if ((s->conformance_window = b.bit() != 0))
    for (int i = 0; i < 4; ++i) s->conf_win[i] = static_cast<int>(std::min<uint32_t>(b.ue(), 100000));
  const uint32_t bdl = b.ue(), bdc = b.ue();
  if (bdl > 8 || bdc > 8) return false;
  s->bit_depth_luma = 8 + static_cast<int>(bdl);
  s->bit_depth_chroma = 8 + static_cast<int>(bdc);
  const uint32_t lpoc = b.ue();
  if (lpoc > 12) return false;
  s->log2_max_poc_lsb = static_cast<int>(lpoc) + 4;
  s->sub_layer_ordering_info_present = b.bit() != 0;
  for (int i = s->sub_layer_ordering_info_present ? 0 : max_sub_layers_minus1; i <= max_sub_layers_minus1; ++i) {
    s->max_dec_pic_buffering[i] = static_cast<int>(std::min<uint32_t>(b.ue(), 16));
    s->max_num_reorder[i] = static_cast<int>(std::min<uint32_t>(b.ue(), 16));
    s->max_latency_increase_plus1[i] = static_cast<int>(std::min<uint32_t>(b.ue(), 0x7FFFFFFF));
  }
  if (!s->sub_layer_ordering_info_present)
    for (int i = 0; i < max_sub_layers_minus1; ++i) {
      s->max_dec_pic_buffering[i] = s->max_dec_pic_buffering[max_sub_layers_minus1];
      s->max_num_reorder[i] = s->max_num_reorder[max_sub_layers_minus1];
      s->max_latency_increase_plus1[i] = s->max_latency_increase_plus1[max_sub_layers_minus1];
    }
  const uint32_t lmin = b.ue(), ldiff = b.ue(), lmint = b.ue(), ldifft = b.ue();
  if (lmin > 3 || ldiff > 3 || lmint > 3 || ldifft > 3) return false;
  s->log2_min_cb = static_cast<int>(lmin) + 3;
  s->log2_ctb = s->log2_min_cb + static_cast<int>(ldiff);
  s->log2_min_tb = static_cast<int>(lmint) + 2;
  s->log2_max_tb = s->log2_min_tb + static_cast<int>(ldifft);
  if (s->log2_ctb > 6 || s->log2_ctb < 4 || s->log2_max_tb > 5 || s->log2_min_tb >= s->log2_min_cb || s->log2_max_tb > s->log2_ctb) return false;
  const uint32_t di = b.ue(), da = b.ue();
  if (di > 4 || da > 4) return false;
  s->max_th_depth_inter = static_cast<int>(di);
  s->max_th_depth_intra = static_cast<int>(da);
  if ((s->scaling_list_enabled = b.bit() != 0)) {
    if ((s->scaling_list_data_present = b.bit() != 0))
      if (!s->scaling.parse(b)) return false;
  }
  s->amp = b.bit() != 0;
  s->sao = b.bit() != 0;
  if ((s->pcm = b.bit() != 0)) {
    s->pcm_bit_depth_luma = static_cast<int>(b.u(4)) + 1;
    s->pcm_bit_depth_chroma = static_cast<int>(b.u(4)) + 1;
    s->log2_min_pcm_cb = static_cast<int>(b.ue()) + 3;
    s->log2_max_pcm_cb = s->log2_min_pcm_cb + static_cast<int>(b.ue());
    s->pcm_loop_filter_disabled = b.bit() != 0;
    if (s->log2_max_pcm_cb > 5 || s->pcm_bit_depth_luma > s->bit_depth_luma || s->pcm_bit_depth_chroma > s->bit_depth_chroma) return false;
  }
  const uint32_t nst = b.ue();
  if (nst > 64) return false;
  s->st_rps.resize(nst);
  for (uint32_t i = 0; i < nst; ++i) {
    std::vector<StRps> prev(s->st_rps.begin(), s->st_rps.begin() + i);
    if (!parse_st_rps(b, static_cast<int>(i), static_cast<int>(nst), prev, &s->st_rps[i])) return false;
  }
  if ((s->long_term_present = b.bit() != 0)) {
    const uint32_t nl = b.ue();
    if (nl > 32) return false;
    s->num_lt_sps = static_cast<int>(nl);
    for (int i = 0; i < s->num_lt_sps; ++i) {
      s->lt_poc_lsb_sps[i] = static_cast<int>(b.u(s->log2_max_poc_lsb));
      s->lt_used_sps[i] = b.bit() != 0;
    }
  }
  s->temporal_mvp = b.bit() != 0;
  s->strong_intra_smoothing = b.bit() != 0;
  if (b.bit()) parse_vui(b, s);
  if (b.bit()) {   // sps_extension_present_flag
    const bool range = b.bit() != 0, multilayer = b.bit() != 0, ext3d = b.bit() != 0, scc = b.bit() != 0;
    b.u(4);
    s->range_extension = range;
    if (range) {
      s->transform_skip_rotation = b.bit() != 0;
      s->transform_skip_context = b.bit() != 0;
      s->implicit_rdpcm = b.bit() != 0;
      s->explicit_rdpcm = b.bit() != 0;
      s->extended_precision = b.bit() != 0;
      s->intra_smoothing_disabled = b.bit() != 0;
      s->high_precision_offsets = b.bit() != 0;
      s->persistent_rice_adaptation = b.bit() != 0;
      s->cabac_bypass_alignment = b.bit() != 0;
    }
    if (multilayer) b.bit();   // inter_view_mv_vert_constraint_flag: the base layer is all that is decoded
    if (ext3d || scc) s->unsupported_extension = true;
  }
  if (b.overrun()) return false;
  s->valid = true;
  return true;
}

// ---- PPS (7.3.2.3) -----------------------------------------------------------------------------
bool parse_pps(const uint8_t* rbsp, size_t n, Pps* p) {
  BitReader b(rbsp, n);
  *p = Pps{};
  const uint32_t id = b.ue(), sps_id = b.ue();
  if (id > 63 || sps_id > 15) return false;
  p->id = static_cast<int>(id);
  p->sps_id = static_cast<int>(sps_id);
  p->dependent_slice_segments_enabled = b.bit() != 0;
  p->output_flag_present = b.bit() != 0;
  p->num_extra_slice_header_bits = static_cast<int>(b.u(3));
  p->sign_data_hiding = b.bit() != 0;
  p->cabac_init_present = b.bit() != 0;
  const uint32_t l0 = b.ue(), l1 = b.ue();
  if (l0 > 14 || l1 > 14) return false;
  p->num_ref_idx_default[0] = static_cast<int>(l0) + 1;
  p->num_ref_idx_default[1] = static_cast<int>(l1) + 1;
  p->init_qp = 26 + b.se();
  p->constrained_intra_pred = b.bit() != 0;
  p->transform_skip_enabled = b.bit() != 0;
  if ((p->cu_qp_delta_enabled = b.bit() != 0)) p->diff_cu_qp_delta_depth = static_cast<int>(std::min<uint32_t>(b.ue(), 3));
  p->cb_qp_offset = b.se();
  p->cr_qp_offset = b.se();
  p->slice_chroma_qp_offsets_present = b.bit() != 0;
  p->weighted_pred = b.bit() != 0;
  p->weighted_bipred = b.bit() != 0;
  p->transquant_bypass_enabled = b.bit() != 0;
  p->tiles_enabled = b.bit() != 0;
  p->entropy_coding_sync = b.bit() != 0;
  if (p->tiles_enabled) {
    const uint32_t nc = b.ue(), nr = b.ue();
    if (nc > 19 || nr > 19) return false;
    p->num_tile_cols = static_cast<int>(nc) + 1;
    p->num_tile_rows = static_cast<int>(nr) + 1;
    p->uniform_spacing = b.bit() != 0;
    if (!p->uniform_spacing) {
      for (int i = 0; i < p->num_tile_cols - 1; ++i) p->column_width_minus1[i] = static_cast<int>(std::min<uint32_t>(b.ue(), 1u << 20));
      for (int i = 0; i < p->num_tile_rows - 1; ++i) p->row_height_minus1[i] = static_cast<int>(std::min<uint32_t>(b.ue(), 1u << 20));
    }
    p->loop_filter_across_tiles = b.bit() != 0;
  }
  p->loop_filter_across_slices = b.bit() != 0;
  if ((p->deblocking_control_present = b.bit() != 0)) {
    p->deblocking_override_enabled = b.bit() != 0;
    if (!(p->deblocking_disabled = b.bit() != 0)) {
      p->beta_offset_div2 = b.se();
      p->tc_offset_div2 = b.se();
    }
  }
  if ((p->scaling_list_data_present = b.bit() != 0))
    if (!p->scaling.parse(b)) return false;
  p->lists_modification_present = b.bit() != 0;
  p->log2_parallel_merge_level = static_cast<int>(b.ue()) + 2;
  p->slice_header_extension_present = b.bit() != 0;
  if (b.bit()) {   // pps_extension_present_flag
    const bool range = b.bit() != 0, multilayer = b.bit() != 0, ext3d = b.bit() != 0, scc = b.bit() != 0;
    b.u(4);
    p->range_extension = range;
    if (range) {
      if (p->transform_skip_enabled) p->log2_max_transform_skip_size = static_cast<int>(b.ue()) + 2;
      p->cross_component_prediction = b.bit() != 0;
      if ((p->chroma_qp_offset_list_enabled = b.bit() != 0)) {
        p->diff_cu_chroma_qp_offset_depth = static_cast<int>(b.ue());
        const uint32_t len = b.ue();
        if (len > 5) return false;
        p->chroma_qp_offset_list_len = static_cast<int>(len) + 1;
        for (int i = 0; i < p->chroma_qp_offset_list_len; ++i) {
          p->cb_qp_offset_list[i] = b.se();
          p->cr_qp_offset_list[i] = b.se();
        }
      }
      p->log2_sao_offset_scale_luma = static_cast<int>(b.ue());
      p->log2_sao_offset_scale_chroma = static_cast<int>(b.ue());
    }
    if (multilayer || ext3d || scc) p->unsupported_extension = true;
  }
  if (b.overrun()) return false;
  p->valid = true;
  return true;
}

// ---- tiles (6.5.1) -----------------------------------------------------------------------------
bool TileLayout::build(const Sps& s, const Pps& p) {
  const int wc = s.width_ctbs(), hc = s.height_ctbs();
  cols = p.tiles_enabled ? p.num_tile_cols : 1;
  rows = p.tiles_enabled ? p.num_tile_rows : 1;
  if (cols > wc || rows > hc) return false;
  std::vector<int> cw(static_cast<size_t>(cols)), rh(static_cast<size_t>(rows));
  if (!p.tiles_enabled || p.uniform_spacing) {
    for (int i = 0; i < cols; ++i) cw[static_cast<size_t>(i)] = ((i + 1) * wc) / cols - (i * wc) / cols;
    for (int j = 0; j < rows; ++j) rh[static_cast<size_t>(j)] = ((j + 1) * hc) / rows - (j * hc) / rows;
  } else {
    int rem = wc;
    for (int i = 0; i < cols - 1; ++i) {
      cw[static_cast<size_t>(i)] = p.column_width_minus1[i] + 1;
      rem -= cw[static_cast<size_t>(i)];
    }
    if (rem <= 0) return false;
    cw[static_cast<size_t>(cols - 1)] = rem;
    rem = hc;
    for (int j = 0; j < rows - 1; ++j) {
      rh[static_cast<size_t>(j)] = p.row_height_minus1[j] + 1;
      rem -= rh[static_cast<size_t>(j)];
    }
    if (rem <= 0) return false;
    rh[static_cast<size_t>(rows - 1)] = rem;
  }
  col_bd.assign(static_cast<size_t>(cols) + 1, 0);
  row_bd.assign(static_cast<size_t>(rows) + 1, 0);
  for (int i = 0; i < cols; ++i) col_bd[static_cast<size_t>(i) + 1] = col_bd[static_cast<size_t>(i)] + cw[static_cast<size_t>(i)];
  for (int j = 0; j < rows; ++j) row_bd[static_cast<size_t>(j) + 1] = row_bd[static_cast<size_t>(j)] + rh[static_cast<size_t>(j)];
  const int total = wc * hc;
  rs_to_ts.assign(static_cast<size_t>(total), 0);
  ts_to_rs.assign(static_cast<size_t>(total), 0);
  tile_id.assign(static_cast<size_t>(total), 0);
  col_of_x.assign(static_cast<size_t>(wc), 0);
  row_of_y.assign(static_cast<size_t>(hc), 0);
  for (int x = 0; x < wc; ++x)
    for (int i = 0; i < cols; ++i)
      if (x >= col_bd[static_cast<size_t>(i)]) col_of_x[static_cast<size_t>(x)] = i;
  for (int y = 0; y < hc; ++y)
    for (int j = 0; j < rows; ++j)
      if (y >= row_bd[static_cast<size_t>(j)]) row_of_y[static_cast<size_t>(y)] = j;
  for (int rs = 0; rs < total; ++rs) {
    const int tbx = rs % wc, tby = rs / wc;
    const int tile_x = col_of_x[static_cast<size_t>(tbx)], tile_y = row_of_y[static_cast<size_t>(tby)];
    int v = 0;
    for (int i = 0; i < tile_x; ++i) v += rh[static_cast<size_t>(tile_y)] * cw[static_cast<size_t>(i)];
    for (int j = 0; j < tile_y; ++j) v += wc * rh[static_cast<size_t>(j)];
    v += (tby - row_bd[static_cast<size_t>(tile_y)]) * cw[static_cast<size_t>(tile_x)] + tbx - col_bd[static_cast<size_t>(tile_x)];
    rs_to_ts[static_cast<size_t>(rs)] = v;
    ts_to_rs[static_cast<size_t>(v)] = rs;
  }
  int tid = 0;
  for (int j = 0; j < rows; ++j)
    for (int i = 0; i < cols; ++i, ++tid)
      for (int y = row_bd[static_cast<size_t>(j)]; y < row_bd[static_cast<size_t>(j) + 1]; ++y)
        for (int x = col_bd[static_cast<size_t>(i)]; x < col_bd[static_cast<size_t>(i) + 1]; ++x)
          tile_id[static_cast<size_t>(rs_to_ts[static_cast<size_t>(y * wc + x)])] = tid;
  return true;
}

// ---- slice segment header (7.3.6) ----------------------------------------------------------------
bool peek_slice_pps_id(const uint8_t* rbsp, size_t n, int nal_type, int* pps_id, bool* first_slice) {
  BitReader b(rbsp, n);
  *first_slice = b.bit() != 0;
  if (is_irap(nal_type)) b.bit();
  const uint32_t id = b.ue();
  if (b.overrun() || id > 63) return false;
  *pps_id = static_cast<int>(id);
  return true;
}

namespace {

bool parse_pred_weights(BitReader& b, const SliceCtx& c, int slice_type, const int num_ref[2], PredWeights* pw) {
  *pw = PredWeights{};
  pw->luma_log2_denom = static_cast<int>(b.ue());
  if (pw->luma_log2_denom > 7) return false;
  pw->chroma_log2_denom = pw->luma_log2_denom;
  if (c.chroma_array_type != 0) {
    pw->chroma_log2_denom += b.se();
    if (pw->chroma_log2_denom < 0 || pw->chroma_log2_denom > 7) return false;
  }
  const int half_c = 1 << (c.high_precision_offsets ? c.bit_depth_chroma - 1 : 7);
  const int lists = slice_type == kSliceB ? 2 : 1;
  for (int l = 0; l < lists; ++l) {
    for (int i = 0; i < num_ref[l]; ++i) pw->luma_flag[l][i] = b.bit() != 0;
    if (c.chroma_array_type != 0)
      for (int i = 0; i < num_ref[l]; ++i) pw->chroma_flag[l][i] = b.bit() != 0;
    for (int i = 0; i < num_ref[l]; ++i) {
      pw->luma_weight[l][i] = 1 << pw->luma_log2_denom;
      pw->luma_offset[l][i] = 0;
      if (pw->luma_flag[l][i]) {
        const int dw = b.se();
        if (dw < -128 || dw > 127) return false;
        pw->luma_weight[l][i] += dw;
        pw->luma_offset[l][i] = b.se();
      }
      for (int j = 0; j < 2; ++j) {
        pw->chroma_weight[l][i][j] = 1 << pw->chroma_log2_denom;
        pw->chroma_offset[l][i][j] = 0;
      }
      if (pw->chroma_flag[l][i]) {
        for (int j = 0; j < 2; ++j) {
          const int dw = b.se();
          if (dw < -128 || dw > 127) return false;
          const int w = (1 << pw->chroma_log2_denom) + dw;
          const int doff = b.se();
          pw->chroma_weight[l][i][j] = w;
          const int v = half_c + doff - ((half_c * w) >> pw->chroma_log2_denom);
          pw->chroma_offset[l][i][j] = std::max(-half_c, std::min(half_c - 1, v));
        }
      }
    }
  }
  return !b.overrun();
}

}  // namespace

bool parse_slice_header(BitReader& b, const SliceCtx& c, int nal_type, const SliceHeader* prev, SliceHeader* out) {
  SliceHeader h;
  h.first_slice_segment_in_pic = b.bit() != 0;
  if (is_irap(nal_type)) h.no_output_of_prior_pics = b.bit() != 0;
  const uint32_t pps_id = b.ue();
  if (pps_id > 63) return false;
  h.pps_id = static_cast<int>(pps_id);
  if (!h.first_slice_segment_in_pic) {
    if (c.dependent_slice_segments_enabled) h.dependent = b.bit() != 0;
    h.segment_address = b.u(ceil_log2(static_cast<uint32_t>(c.width_ctbs * c.height_ctbs)));
    if (h.segment_address >= static_cast<uint32_t>(c.width_ctbs * c.height_ctbs)) return false;
  }
  if (h.dependent) {
    if (!prev) return false;
    const bool first = h.first_slice_segment_in_pic, no_out = h.no_output_of_prior_pics;
    const uint32_t addr = h.segment_address;
    h = *prev;
    h.first_slice_segment_in_pic = first;
    h.no_output_of_prior_pics = no_out;
    h.dependent = true;
    h.segment_address = addr;
    h.pps_id = static_cast<int>(pps_id);
    h.num_entry_points = 0;
    h.entry_offset.clear();
  } else {
    for (int i = 0; i < c.num_extra_slice_header_bits; ++i) b.bit();
    const uint32_t st = b.ue();
    if (st > 2) return false;
    h.slice_type = static_cast<int>(st);
    if (c.output_flag_present) h.pic_output_flag = b.bit() != 0;
    if (c.separate_colour_plane) h.colour_plane_id = static_cast<int>(b.u(2));
    if (!is_idr(nal_type)) {
      h.poc_lsb = static_cast<int>(b.u(c.log2_max_poc_lsb));
      h.st_rps_sps_flag = b.bit() != 0;
      if (!h.st_rps_sps_flag) {
        const size_t start = b.pos();
        if (c.sps_rps) {
          if (!parse_st_rps(b, c.num_st_rps, c.num_st_rps, *c.sps_rps, &h.st_rps, &h.st_rps_ref_num_delta)) return false;
        } else {
          b.skip(static_cast<size_t>(std::max(0, c.skip_st_rps_bits)));
        }
        h.st_rps_bits = static_cast<int>(b.pos() - start);
      } else {
        if (c.num_st_rps > 1) h.st_rps_idx = static_cast<int>(b.u(ceil_log2(static_cast<uint32_t>(c.num_st_rps))));
        if (c.sps_rps) {
          if (h.st_rps_idx >= static_cast<int>(c.sps_rps->size())) return false;
          h.st_rps = (*c.sps_rps)[static_cast<size_t>(h.st_rps_idx)];
        }
      }
      if (c.long_term_present) {
        uint32_t num_lt_sps = 0;
        if (c.num_lt_sps > 0) num_lt_sps = b.ue();
        const uint32_t num_lt_pics = b.ue();
        if (num_lt_sps > static_cast<uint32_t>(c.num_lt_sps) || num_lt_pics > 32 || num_lt_sps + num_lt_pics > 32) return false;
        h.num_lt = static_cast<int>(num_lt_sps + num_lt_pics);
        for (int i = 0; i < h.num_lt; ++i) {
          if (i < static_cast<int>(num_lt_sps)) {
            int idx = 0;
            if (c.num_lt_sps > 1) idx = static_cast<int>(b.u(ceil_log2(static_cast<uint32_t>(c.num_lt_sps))));
            if (idx >= c.num_lt_sps) return false;
            h.lt_poc_lsb[i] = c.lt_poc_lsb_sps ? c.lt_poc_lsb_sps[idx] : 0;
            h.lt_used[i] = c.lt_used_sps ? c.lt_used_sps[idx] : false;
          } else {
            h.lt_poc_lsb[i] = static_cast<int>(b.u(c.log2_max_poc_lsb));
            h.lt_used[i] = b.bit() != 0;
          }
          h.lt_msb_present[i] = b.bit() != 0;
          int cycle = 0;
          if (h.lt_msb_present[i]) cycle = static_cast<int>(std::min<uint32_t>(b.ue(), 1u << 20));
          if (i == 0 || i == static_cast<int>(num_lt_sps)) h.lt_delta_msb_cycle[i] = cycle;
          else h.lt_delta_msb_cycle[i] = cycle + h.lt_delta_msb_cycle[i - 1];
        }
      }
      if (c.temporal_mvp) h.temporal_mvp = b.bit() != 0;
    }
    if (c.sao) {
      h.sao_luma = b.bit() != 0;
      if (c.chroma_array_type != 0) h.sao_chroma = b.bit() != 0;
    }
    // NumPicTotalCurr (7-57)
    if (c.fixed_num_pic_total_curr >= 0) {
      h.num_pic_total_curr = c.fixed_num_pic_total_curr;
    } else {
      int total = 0;
      if (!is_idr(nal_type)) {
        for (int i = 0; i < h.st_rps.num_neg; ++i) total += h.st_rps.used_s0[i];
        for (int i = 0; i < h.st_rps.num_pos; ++i) total += h.st_rps.used_s1[i];
        for (int i = 0; i < h.num_lt; ++i) total += h.lt_used[i];
      }
      h.num_pic_total_curr = total;
    }
    if (h.slice_type == kSliceP || h.slice_type == kSliceB) {
      h.num_ref_idx[0] = c.num_ref_idx_default[0];
      h.num_ref_idx[1] = h.slice_type == kSliceB ? c.num_ref_idx_default[1] : 0;
      if (b.bit()) {   // num_ref_idx_active_override_flag
        const uint32_t l0 = b.ue();
        if (l0 > 14) return false;
        h.num_ref_idx[0] = static_cast<int>(l0) + 1;
        if (h.slice_type == kSliceB) {
          const uint32_t l1 = b.ue();
          if (l1 > 14) return false;
          h.num_ref_idx[1] = static_cast<int>(l1) + 1;
        }
      }
      if (c.lists_modification_present && h.num_pic_total_curr > 1) {
        const int bits = ceil_log2(static_cast<uint32_t>(h.num_pic_total_curr));
        h.list_mod[0] = b.bit() != 0;
        if (h.list_mod[0])
          for (int i = 0; i < h.num_ref_idx[0]; ++i) h.list_entry[0][i] = static_cast<int>(b.u(bits));
        if (h.slice_type == kSliceB) {
          h.list_mod[1] = b.bit() != 0;
          if (h.list_mod[1])
            for (int i = 0; i < h.num_ref_idx[1]; ++i) h.list_entry[1][i] = static_cast<int>(b.u(bits));
        }
      }
      if (h.slice_type == kSliceB) h.mvd_l1_zero = b.bit() != 0;
      if (c.cabac_init_present) h.cabac_init_flag = b.bit() != 0;
      if (h.temporal_mvp) {
        if (h.slice_type == kSliceB) h.collocated_from_l0 = b.bit() != 0;
        if ((h.collocated_from_l0 && h.num_ref_idx[0] > 1) || (!h.collocated_from_l0 && h.num_ref_idx[1] > 1)) {
          h.collocated_ref_idx = static_cast<int>(b.ue());
          if (h.collocated_ref_idx >= h.num_ref_idx[h.collocated_from_l0 ? 0 : 1]) return false;
        }
      }
      if ((c.weighted_pred && h.slice_type == kSliceP) || (c.weighted_bipred && h.slice_type == kSliceB))
        if (!parse_pred_weights(b, c, h.slice_type, h.num_ref_idx, &h.pw)) return false;
      const uint32_t fmm = b.ue();
      if (fmm > 4) return false;
      h.max_num_merge_cand = 5 - static_cast<int>(fmm);
    }
    h.slice_qp_delta = b.se();
    if (c.slice_chroma_qp_offsets_present) {
      h.cb_qp_offset = b.se();
      h.cr_qp_offset = b.se();
    }
    if (c.chroma_qp_offset_list_enabled) h.cu_chroma_qp_offset_enabled = b.bit() != 0;
    if (c.deblocking_override_enabled) h.deblocking_override = b.bit() != 0;
    h.deblocking_disabled = c.pps_deblocking_disabled;
    h.beta_offset_div2 = c.pps_beta_offset_div2;
    h.tc_offset_div2 = c.pps_tc_offset_div2;
    if (h.deblocking_override) {
      h.deblocking_disabled = b.bit() != 0;
      if (!h.deblocking_disabled) {
        h.beta_offset_div2 = b.se();
        h.tc_offset_div2 = b.se();
      }
    }
    h.loop_filter_across_slices = c.pps_loop_filter_across_slices;
    if (c.pps_loop_filter_across_slices && (h.sao_luma || h.sao_chroma || !h.deblocking_disabled)) h.loop_filter_across_slices = b.bit() != 0;
  }
  if (c.tiles_enabled || c.entropy_coding_sync) {
    const uint32_t n = b.ue();
    // at most one entry point per CTB row or tile
    if (n > static_cast<uint32_t>(c.width_ctbs * c.height_ctbs)) return false;
    h.num_entry_points = static_cast<int>(n);
    if (n > 0) {
      const uint32_t len = b.ue() + 1;
      if (len > 32) return false;
      h.entry_offset.resize(n);
      for (uint32_t i = 0; i < n; ++i) {
        h.entry_offset[i] = b.u(static_cast<int>(len)) + 1;
        if (b.overrun()) return false;
      }
    }
  }
  if (c.slice_header_extension_present) {
    const uint32_t len = b.ue();
    if (len > 256) return false;
    b.skip(static_cast<size_t>(len) * 8);
  }
  // byte_alignment(): a one bit, then zero bits up to the byte boundary
  if (!b.bit()) return false;
  while (!b.byte_aligned()) b.bit();
  if (b.overrun()) return false;
  h.data_bit_offset = b.pos();
  *out = std::move(h);
  return true;
}

}  // namespace vgpu_hevc
