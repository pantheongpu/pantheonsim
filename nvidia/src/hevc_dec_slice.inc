// Part of hevc_decode.cpp: slice segment data -- the coding tree unit, quadtree, coding unit, prediction unit and transform tree syntax
// (7.3.8), with the decoding processes they trigger.

namespace {
// Table 8-10: QpC for ChromaArrayType 1.
int qpc_table(int qpi) {
  static const int kMid[14] = {29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37};
  if (qpi < 30) return qpi;
  if (qpi >= 44) return qpi - 6;
  return kMid[qpi - 30];
}
enum PartMode { PART_2Nx2N = 0, PART_2NxN, PART_Nx2N, PART_NxN, PART_2NxnU, PART_2NxnD, PART_nLx2N, PART_nRx2N };
}  // namespace

// 6.4.1
bool Dec::avail_zs(int xc, int yc, int xn, int yn) const {
  if (xn < 0 || yn < 0 || xn >= W || yn >= H) return false;
  const int a_cur = min_tb_addr_zs_[static_cast<size_t>((yc >> log2_min_tb) * tbw_ + (xc >> log2_min_tb))];
  const int a_n = min_tb_addr_zs_[static_cast<size_t>((yn >> log2_min_tb) * tbw_ + (xn >> log2_min_tb))];
  if (a_n > a_cur) return false;
  const int cn = (yn >> log2_ctb) * wctb + (xn >> log2_ctb), cc = (yc >> log2_ctb) * wctb + (xc >> log2_ctb);
  const int sn = ctb_slice_[static_cast<size_t>(cn)];
  if (sn < 0) return false;
  if (slices_[static_cast<size_t>(sn)].addr_rs != S->addr_rs) return false;
  if (tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(cn)])] != tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(cc)])]) return false;
  return true;
}

// 6.4.2
bool Dec::avail_pb(int xcb, int ycb, int ncbs, int xpb, int ypb, int npbw, int npbh, int part_idx, int xn, int yn) const {
  const bool same_cb = xcb <= xn && ycb <= yn && xcb + ncbs > xn && ycb + ncbs > yn;
  bool a;
  if (!same_cb) {
    a = avail_zs(xpb, ypb, xn, yn);
  } else if ((npbw << 1) == ncbs && (npbh << 1) == ncbs && part_idx == 1 && ycb + npbh <= yn && xcb + npbw > xn) {
    a = false;
  } else {
    a = true;
  }
  if (a && pred_mode_[static_cast<size_t>(idx4(xn, yn))] != 1) a = false;   // not an inter-coded block
  return a;
}

Dec::Dec(Frame* cur, const PicParams& pp, std::string* err) : F(cur), P(pp), err_(err) {
  W = pp.width;
  H = pp.height;
  CW = W / 2;
  CH = H / 2;
  log2_min_cb = pp.log2_min_cb;
  log2_ctb = pp.log2_min_cb + pp.log2_diff_cb;
  ctb = 1 << log2_ctb;
  log2_min_tb = pp.log2_min_tb;
  log2_max_tb = pp.log2_min_tb + pp.log2_diff_tb;
  bd_y = pp.bit_depth_luma;
  bd_c = pp.bit_depth_chroma;
  qp_bd_offset_y = 6 * (bd_y - 8);
  qp_bd_offset_c = 6 * (bd_c - 8);
  wctb = (W + ctb - 1) >> log2_ctb;
  hctb = (H + ctb - 1) >> log2_ctb;
  pic_size_ctbs = wctb * hctb;
  w4 = (wctb << log2_ctb) >> 2;
  h4 = (hctb << log2_ctb) >> 2;
  log2_min_cu_qp_delta = log2_ctb - pp.diff_cu_qp_delta_depth;
  log2_min_cu_chroma_qp_offset = log2_ctb - pp.diff_cu_chroma_qp_offset_depth;
  sps_.width = W;
  sps_.height = H;
  sps_.log2_ctb = log2_ctb;
  pps_.tiles_enabled = pp.tiles_enabled;
  pps_.uniform_spacing = pp.uniform_spacing;
  pps_.num_tile_cols = pp.num_tile_cols;
  pps_.num_tile_rows = pp.num_tile_rows;
  for (int i = 0; i < 20; ++i) {
    pps_.column_width_minus1[i] = pp.column_width_minus1[i];
    pps_.row_height_minus1[i] = pp.row_height_minus1[i];
  }
  if (!tiles_.build(sps_, pps_)) {
    fail("invalid tile layout");
    return;
  }
  // 6.5.1
  tbw_ = wctb << (log2_ctb - log2_min_tb);
  tbh_ = hctb << (log2_ctb - log2_min_tb);
  min_tb_addr_zs_.assign(static_cast<size_t>(tbw_) * static_cast<size_t>(tbh_), 0);
  for (int y = 0; y < tbh_; ++y)
    for (int x = 0; x < tbw_; ++x) {
      const int tbx = (x << log2_min_tb) >> log2_ctb, tby = (y << log2_min_tb) >> log2_ctb;
      const int rs = wctb * tby + tbx;
      int v = tiles_.rs_to_ts[static_cast<size_t>(rs)] << ((log2_ctb - log2_min_tb) * 2);
      for (int i = 0; i < log2_ctb - log2_min_tb; ++i) {
        const int m = 1 << i;
        v += ((m & x) ? m * m : 0) + ((m & y) ? 2 * m * m : 0);
      }
      min_tb_addr_zs_[static_cast<size_t>(y * tbw_ + x)] = v;
    }
  const size_t n4 = static_cast<size_t>(w4) * static_cast<size_t>(h4);
  pred_mode_.assign(n4, 0);
  cu_flags_.assign(n4, 0);
  ct_depth_.assign(n4, 0);
  intra_mode_.assign(n4, 1);
  qp_y_.assign(n4, 0);
  tu_nz_.assign(n4, 0);
  edge_.assign(n4, 0);
  MvField none;
  std::memset(&none, 0, sizeof none);
  none.ref_idx[0] = none.ref_idx[1] = -1;
  mvf_.assign(n4, none);
  ctb_slice_.assign(static_cast<size_t>(pic_size_ctbs), -1);
  sao_.assign(static_cast<size_t>(pic_size_ctbs), SaoParams());
}

// 8.3.4: reference picture lists of a P or B slice.
bool Dec::build_ref_lists(SliceInfo& s) {
  const SliceHeader& h = s.h;
  for (int l = 0; l < 2; ++l)
    for (int i = 0; i < 16; ++i) {
      s.ref[l][i] = RefPic();
      s.ref_slot[l][i] = -1;
    }
  if (h.slice_type == kSliceI) return true;
  const int total = P.num_st_before + P.num_st_after + P.num_lt_curr;
  if (total <= 0) {
    fail("a P or B slice without reference pictures");
    return false;
  }
  for (int l = 0; l < (h.slice_type == kSliceB ? 2 : 1); ++l) {
    const int n_active = h.num_ref_idx[l];
    const int n_temp = std::max(n_active, total);
    int temp[40], temp_lt[40];
    int cnt = 0;
    while (cnt < n_temp) {
      const int first_n = l == 0 ? P.num_st_before : P.num_st_after, second_n = l == 0 ? P.num_st_after : P.num_st_before;
      const uint8_t* first = l == 0 ? P.st_before : P.st_after;
      const uint8_t* second = l == 0 ? P.st_after : P.st_before;
      for (int i = 0; i < first_n && cnt < n_temp; ++i, ++cnt) {
        temp[cnt] = first[i];
        temp_lt[cnt] = 0;
      }
      for (int i = 0; i < second_n && cnt < n_temp; ++i, ++cnt) {
        temp[cnt] = second[i];
        temp_lt[cnt] = 0;
      }
      for (int i = 0; i < P.num_lt_curr && cnt < n_temp; ++i, ++cnt) {
        temp[cnt] = P.lt_curr[i];
        temp_lt[cnt] = 1;
      }
    }
    for (int r = 0; r < n_active; ++r) {
      const int idx = h.list_mod[l] ? h.list_entry[l][r] : r;
      if (idx >= n_temp) {
        fail("list_entry out of range");
        return false;
      }
      const int slot = temp[idx] & 15;
      s.ref[l][r] = P.refs[slot];
      s.ref[l][r].long_term = temp_lt[idx] != 0 || P.refs[slot].long_term;
      s.ref_slot[l][r] = slot;
    }
  }
  // 8.3.5
  s.no_backward_pred = true;
  for (int l = 0; l < 2; ++l)
    for (int r = 0; r < h.num_ref_idx[l]; ++r)
      if (s.ref[l][r].poc > P.curr_poc) s.no_backward_pred = false;
  if (h.temporal_mvp) {
    const int cl = (h.slice_type == kSliceB && !h.collocated_from_l0) ? 1 : 0;
    s.col_pic = s.ref[cl][h.collocated_ref_idx].frame;
    s.col_slot = s.ref_slot[cl][h.collocated_ref_idx];
    s.col_poc = s.ref[cl][h.collocated_ref_idx].poc;
  }
  return true;
}

bool Dec::decode_slice_segment(const SliceData& sd, bool first) {
  if (sd.size < 3) {
    fail("slice segment NAL unit too short");
    return false;
  }
  const int nal_type = (sd.nal[0] >> 1) & 63;
  std::vector<uint8_t> rbsp = unescape(sd.nal, sd.size, 2);
  BitReader b(rbsp.data(), rbsp.size());
  SliceCtx c;
  c.log2_ctb = log2_ctb;
  c.width_ctbs = wctb;
  c.height_ctbs = hctb;
  c.log2_max_poc_lsb = P.log2_max_poc_lsb;
  c.num_st_rps = P.num_short_term_ref_pic_sets;
  c.skip_st_rps_bits = P.num_bits_st_rps;
  c.fixed_num_pic_total_curr = P.num_poc_total_curr;
  (void)first;
  c.long_term_present = P.long_term_ref_pics_present;
  c.num_lt_sps = P.num_long_term_ref_pics_sps;
  c.temporal_mvp = P.sps_temporal_mvp;
  c.sao = P.sao;
  c.chroma_array_type = 1;
  c.separate_colour_plane = false;
  c.high_precision_offsets = P.high_precision_offsets;
  c.bit_depth_chroma = bd_c;
  c.dependent_slice_segments_enabled = P.dependent_slice_segments_enabled;
  c.output_flag_present = P.output_flag_present;
  c.num_extra_slice_header_bits = P.num_extra_slice_header_bits;
  c.lists_modification_present = P.lists_modification_present;
  c.cabac_init_present = P.cabac_init_present;
  c.num_ref_idx_default[0] = P.num_ref_idx_default[0];
  c.num_ref_idx_default[1] = P.num_ref_idx_default[1];
  c.weighted_pred = P.weighted_pred;
  c.weighted_bipred = P.weighted_bipred;
  c.slice_chroma_qp_offsets_present = P.slice_chroma_qp_offsets_present;
  c.chroma_qp_offset_list_enabled = P.chroma_qp_offset_list_enabled;
  c.deblocking_override_enabled = P.deblocking_override_enabled;
  c.pps_deblocking_disabled = P.pps_deblocking_disabled;
  c.pps_beta_offset_div2 = P.pps_beta_offset_div2;
  c.pps_tc_offset_div2 = P.pps_tc_offset_div2;
  c.pps_loop_filter_across_slices = P.loop_filter_across_slices;
  c.tiles_enabled = P.tiles_enabled;
  c.entropy_coding_sync = P.entropy_coding_sync;
  c.slice_header_extension_present = P.slice_segment_header_extension_present;
  c.init_qp = P.init_qp;
  SliceHeader h;
  const SliceHeader* prev = cur_slice_idx_ >= 0 ? &slices_[static_cast<size_t>(cur_slice_idx_)].h : nullptr;
  if (!parse_slice_header(b, c, nal_type, prev, &h)) {
    fail("malformed slice segment header");
    return false;
  }
  sh_ = h;
  if (!h.dependent) {
    if (slices_.size() == slices_.capacity()) {
      fail("too many slices");
      return false;
    }
    slices_.emplace_back();
    cur_slice_idx_ = static_cast<int>(slices_.size()) - 1;
    S = &slices_.back();
    S->h = h;
    S->addr_rs = static_cast<int>(h.segment_address);
    S->slice_qp = P.init_qp + h.slice_qp_delta;
    if (!build_ref_lists(*S)) return false;
    cu_qp_offset_cb_ = cu_qp_offset_cr_ = 0;
    last_cu_qp_ = S->slice_qp;
  } else {
    if (cur_slice_idx_ < 0) {
      fail("dependent slice segment without a slice");
      return false;
    }
    S = &slices_[static_cast<size_t>(cur_slice_idx_)];
    S->h.num_entry_points = h.num_entry_points;
  }
  sd_ = rbsp.data();
  sd_size_ = rbsp.size();
  bitpos_ = h.data_bit_offset;
  ctb_addr_rs_ = static_cast<int>(h.segment_address);
  if (ctb_addr_rs_ >= pic_size_ctbs) {
    fail("slice segment address out of range");
    return false;
  }
  ctb_addr_ts_ = tiles_.rs_to_ts[static_cast<size_t>(ctb_addr_rs_)];
  decode_ctus();
  sd_ = nullptr;
  return ok_;
}

void Dec::decode_ctus() {
  const bool wpp = P.entropy_coding_sync;
  bool first_in_segment = true;
  for (;;) {
    const int rs = tiles_.ts_to_rs[static_cast<size_t>(ctb_addr_ts_)];
    ctb_addr_rs_ = rs;
    const int rx = rs % wctb, ry = rs / wctb;
    const int x_ctb = rx << log2_ctb, y_ctb = ry << log2_ctb;
    ctb_slice_[static_cast<size_t>(rs)] = static_cast<int16_t>(cur_slice_idx_);
    const bool first_in_tile = ctb_addr_ts_ == 0 || tiles_.tile_id[static_cast<size_t>(ctb_addr_ts_)] != tiles_.tile_id[static_cast<size_t>(ctb_addr_ts_ - 1)];
    const bool row_start_in_tile = rx == tiles_.col_bd[static_cast<size_t>(tiles_.col_of_x[static_cast<size_t>(rx)])];
    // 9.3.2.1: initialization at the start of the segment, of a tile and (wavefront) of a CTB row of a tile
    if (first_in_segment || first_in_tile || (wpp && row_start_in_tile)) {
      if (first_in_tile) {
        init_contexts();
        last_cu_qp_ = S->slice_qp;
      } else if (wpp && row_start_in_tile) {
        if (avail_zs(x_ctb, y_ctb, x_ctb + ctb, y_ctb - ctb) && have_wpp_) {
          std::memcpy(ctx_, wpp_ctx_, sizeof ctx_);
          std::memcpy(stat_coeff_, wpp_stat_, sizeof stat_coeff_);
        } else {
          init_contexts();
        }
        last_cu_qp_ = S->slice_qp;
      } else if (sh_.dependent && ds_valid_) {
        std::memcpy(ctx_, ds_ctx_, sizeof ctx_);
        std::memcpy(stat_coeff_, ds_stat_, sizeof stat_coeff_);
      } else {
        init_contexts();
      }
      if (first_in_segment && !sh_.dependent) last_cu_qp_ = S->slice_qp;
      cabac_init_engine();
    }
    first_in_segment = false;
    // the coding tree unit
    if (bitpos_ > sd_size_ * 8 + 64) {
      fail("slice segment data ended early");
      return;
    }
    if (sh_.sao_luma || sh_.sao_chroma) sao_syntax(rx, ry);
    else sao_[static_cast<size_t>(rs)] = SaoParams();
    coding_quadtree(x_ctb, y_ctb, log2_ctb, 0);
    if (!ok_) return;
    const int end_of_segment = decode_terminate();
    // storage for wavefront synchronisation: after the second CTB of a row in a tile
    if (wpp) {
      const int col_in_tile = rx - tiles_.col_bd[static_cast<size_t>(tiles_.col_of_x[static_cast<size_t>(rx)])];
      const int tile_w = tiles_.col_bd[static_cast<size_t>(tiles_.col_of_x[static_cast<size_t>(rx)]) + 1] - tiles_.col_bd[static_cast<size_t>(tiles_.col_of_x[static_cast<size_t>(rx)])];
      if (col_in_tile == 1 || (tile_w == 1 && col_in_tile == 0)) {
        std::memcpy(wpp_ctx_, ctx_, sizeof ctx_);
        std::memcpy(wpp_stat_, stat_coeff_, sizeof stat_coeff_);
        have_wpp_ = true;
      }
    }
    ++ctb_addr_ts_;
    if (end_of_segment) {
      if (P.dependent_slice_segments_enabled) {
        std::memcpy(ds_ctx_, ctx_, sizeof ctx_);
        std::memcpy(ds_stat_, stat_coeff_, sizeof stat_coeff_);
        ds_valid_ = true;
      }
      return;
    }
    if (ctb_addr_ts_ >= pic_size_ctbs) {
      fail("slice segment data runs past the picture");
      return;
    }
    const int nrs = tiles_.ts_to_rs[static_cast<size_t>(ctb_addr_ts_)];
    const bool new_tile = P.tiles_enabled && tiles_.tile_id[static_cast<size_t>(ctb_addr_ts_)] != tiles_.tile_id[static_cast<size_t>(ctb_addr_ts_ - 1)];
    const int nrx = nrs % wctb;
    const bool new_row = wpp && nrx == tiles_.col_bd[static_cast<size_t>(tiles_.col_of_x[static_cast<size_t>(nrx)])];
    if (new_tile || new_row) {
      if (!decode_terminate()) {   // end_of_subset_one_bit
        fail("end_of_subset_one_bit is not one");
        return;
      }
      bitpos_ = (bitpos_ + 7) & ~static_cast<size_t>(7);
    }
  }
}

void Dec::sao_syntax(int rx, int ry) {
  const int rs = ry * wctb + rx;
  SaoParams& sp = sao_[static_cast<size_t>(rs)];
  sp = SaoParams();
  bool merge_left = false, merge_up = false;
  if (rx > 0) {
    const bool left_in_slice = rs > S->addr_rs;
    const bool left_in_tile = tiles_.tile_id[static_cast<size_t>(ctb_addr_ts_)] == tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(rs - 1)])];
    if (left_in_slice && left_in_tile) merge_left = decode_bin(ctx_[C_SAO_MERGE]) != 0;
  }
  if (ry > 0 && !merge_left) {
    const bool up_in_slice = (rs - wctb) >= S->addr_rs;
    const bool up_in_tile = tiles_.tile_id[static_cast<size_t>(ctb_addr_ts_)] == tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(rs - wctb)])];
    if (up_in_slice && up_in_tile) merge_up = decode_bin(ctx_[C_SAO_MERGE]) != 0;
  }
  if (merge_left) {
    sp = sao_[static_cast<size_t>(rs - 1)];
    return;
  }
  if (merge_up) {
    sp = sao_[static_cast<size_t>(rs - wctb)];
    return;
  }
  for (int c = 0; c < 3; ++c) {
    if (!((sh_.sao_luma && c == 0) || (sh_.sao_chroma && c > 0))) continue;
    if (c == 0 || c == 1) {
      int t = 0;
      if (decode_bin(ctx_[C_SAO_TYPE])) t = decode_bypass() ? 2 : 1;
      sp.type[c] = static_cast<uint8_t>(t);
      if (c == 1) sp.type[2] = sp.type[1];
    }
    if (sp.type[c] == 0) continue;
    const int bd = c == 0 ? bd_y : bd_c;
    const int cmax = (1 << (std::min(bd, 10) - 5)) - 1;
    int abs_v[4];
    for (int i = 0; i < 4; ++i) {
      int v = 0;
      while (v < cmax && decode_bypass()) ++v;
      abs_v[i] = v;
    }
    int sign[4] = {0, 0, 0, 0};
    const int log2_scale = c == 0 ? P.log2_sao_offset_scale_luma : P.log2_sao_offset_scale_chroma;
    if (sp.type[c] == 1) {
      for (int i = 0; i < 4; ++i)
        if (abs_v[i]) sign[i] = decode_bypass();
      sp.band_pos[c] = static_cast<uint8_t>(decode_bypass_bits(5));
    } else {
      if (c == 0 || c == 1) {
        sp.eo_class[c] = static_cast<uint8_t>(decode_bypass_bits(2));
        if (c == 1) sp.eo_class[2] = sp.eo_class[1];
      }
      sign[2] = sign[3] = 1;
    }
    sp.offset[c][0] = 0;
    for (int i = 0; i < 4; ++i) sp.offset[c][i + 1] = static_cast<int16_t>((sign[i] ? -abs_v[i] : abs_v[i]) * (1 << log2_scale));
  }
}

void Dec::compute_qp_pred(int xcb, int ycb) {
  const int mask = (1 << log2_min_cu_qp_delta) - 1;
  const int xqg = xcb - (xcb & mask), yqg = ycb - (ycb & mask);
  const int prev = last_cu_qp_;
  int qa = prev, qb = prev;
  // the neighbours count only inside the current CTB (8-282, 8-283)
  const int cx = xcb >> log2_ctb, cy = ycb >> log2_ctb;
  if (xqg - 1 >= 0 && ((xqg - 1) >> log2_ctb) == cx && (yqg >> log2_ctb) == cy) qa = qp_y_[static_cast<size_t>(idx4(xqg - 1, yqg))];
  if (yqg - 1 >= 0 && ((yqg - 1) >> log2_ctb) == cy && (xqg >> log2_ctb) == cx) qb = qp_y_[static_cast<size_t>(idx4(xqg, yqg - 1))];
  qp_y_pred_ = (qa + qb + 1) >> 1;
  qp_pred_valid_ = true;
}

// QpY of the current coding unit from the prediction and CuQpDeltaVal (8-285).
void Dec::set_cu_qp() {
  if (!P.cu_qp_delta_enabled) {
    cu_qp_y_ = S->slice_qp;
    return;
  }
  cu_qp_y_ = ((qp_y_pred_ + cu_qp_delta_val_ + 52 + 2 * qp_bd_offset_y) % (52 + qp_bd_offset_y)) - qp_bd_offset_y;
}

int Dec::chroma_qp(int c_idx) const {
  const int off = c_idx == 1 ? P.cb_qp_offset + S->h.cb_qp_offset + cu_qp_offset_cb_ : P.cr_qp_offset + S->h.cr_qp_offset + cu_qp_offset_cr_;
  const int qpi = std::max(-qp_bd_offset_c, std::min(57, cu_qp_y_ + off));
  return qpc_table(qpi) + qp_bd_offset_c;
}

void Dec::fill_cu_info(int x0, int y0, int log2_cb, uint8_t pm, uint8_t flags) {
  const int n = 1 << (log2_cb - 2);
  for (int j = 0; j < n; ++j) {
    const size_t base = static_cast<size_t>(idx4(x0, y0 + 4 * j));
    std::memset(&pred_mode_[base], pm, static_cast<size_t>(n));
    std::memset(&cu_flags_[base], flags, static_cast<size_t>(n));
  }
}

void Dec::coding_quadtree(int x0, int y0, int log2_cb, int depth) {
  if (!ok_) return;
  const int size = 1 << log2_cb;
  bool split;
  if (x0 + size <= W && y0 + size <= H && log2_cb > log2_min_cb) {
    int inc = 0;
    if (avail_zs(x0, y0, x0 - 1, y0) && ct_depth_[static_cast<size_t>(idx4(x0 - 1, y0))] > depth) ++inc;
    if (avail_zs(x0, y0, x0, y0 - 1) && ct_depth_[static_cast<size_t>(idx4(x0, y0 - 1))] > depth) ++inc;
    split = decode_bin(ctx_[C_SPLIT_CU + inc]) != 0;
  } else {
    split = log2_cb > log2_min_cb;
  }
  if (P.cu_qp_delta_enabled && log2_cb >= log2_min_cu_qp_delta) {
    is_cu_qp_delta_coded_ = false;
    cu_qp_delta_val_ = 0;
    qp_pred_valid_ = false;
  }
  if (sh_.cu_chroma_qp_offset_enabled && log2_cb >= log2_min_cu_chroma_qp_offset) is_cu_chroma_qp_offset_coded_ = false;
  if (split) {
    const int x1 = x0 + (size >> 1), y1 = y0 + (size >> 1);
    coding_quadtree(x0, y0, log2_cb - 1, depth + 1);
    if (x1 < W) coding_quadtree(x1, y0, log2_cb - 1, depth + 1);
    if (y1 < H) coding_quadtree(x0, y1, log2_cb - 1, depth + 1);
    if (x1 < W && y1 < H) coding_quadtree(x1, y1, log2_cb - 1, depth + 1);
  } else {
    cu_depth_ = depth;
    coding_unit(x0, y0, log2_cb);
  }
}

void Dec::coding_unit(int x0, int y0, int log2_cb) {
  const int ncbs = 1 << log2_cb;
  cu_x_ = x0;
  cu_y_ = y0;
  cu_log2_ = log2_cb;
  cu_transquant_bypass_ = false;
  pcm_flag_ = false;
  merge_flag_cu_ = false;
  intra_split_ = false;
  part_mode_ = PART_2Nx2N;
  if (P.cu_qp_delta_enabled && !qp_pred_valid_) compute_qp_pred(x0, y0);
  if (!P.cu_qp_delta_enabled) qp_y_pred_ = S->slice_qp;
  set_cu_qp();
  if (P.transquant_bypass_enabled) cu_transquant_bypass_ = decode_bin(ctx_[C_TQ_BYPASS]) != 0;
  bool skip = false;
  if (sh_.slice_type != kSliceI) {
    int inc = 0;
    if (avail_zs(x0, y0, x0 - 1, y0) && (cu_flags_[static_cast<size_t>(idx4(x0 - 1, y0))] & kCuSkip)) ++inc;
    if (avail_zs(x0, y0, x0, y0 - 1) && (cu_flags_[static_cast<size_t>(idx4(x0, y0 - 1))] & kCuSkip)) ++inc;
    skip = decode_bin(ctx_[C_SKIP + inc]) != 0;
  }
  uint8_t flags = static_cast<uint8_t>((skip ? kCuSkip : 0) | (cu_transquant_bypass_ ? kCuBypass : 0));
  const int n4 = ncbs >> 2;
  // coding-unit wide arrays the neighbours and later stages read
  for (int j = 0; j < n4; ++j) std::memset(&ct_depth_[static_cast<size_t>(idx4(x0, y0 + 4 * j))], cu_depth_, static_cast<size_t>(n4));
  // transform / prediction edges of the coding block boundary
  for (int k = 0; k < n4; ++k) {
    edge_[static_cast<size_t>(idx4(x0, y0 + 4 * k))] |= 1;
    edge_[static_cast<size_t>(idx4(x0 + 4 * k, y0))] |= 4;
  }
  if (skip) {
    cu_intra_ = false;
    fill_cu_info(x0, y0, log2_cb, 1, flags);
    for (int j = 0; j < n4; ++j) std::memset(&tu_nz_[static_cast<size_t>(idx4(x0, y0 + 4 * j))], 0, static_cast<size_t>(n4));
    prediction_unit(x0, y0, ncbs, x0, y0, ncbs, ncbs, 0);
  } else {
    bool intra = true;
    if (sh_.slice_type != kSliceI) intra = decode_bin(ctx_[C_PRED_MODE]) != 0;
    cu_intra_ = intra;
    if (!intra || log2_cb == log2_min_cb) {
      if (intra) {
        part_mode_ = decode_bin(ctx_[C_PART_MODE]) ? PART_2Nx2N : PART_NxN;
      } else if (decode_bin(ctx_[C_PART_MODE])) {
        part_mode_ = PART_2Nx2N;
      } else if (log2_cb == log2_min_cb) {
        if (log2_cb == 3) {
          part_mode_ = decode_bin(ctx_[C_PART_MODE + 1]) ? PART_2NxN : PART_Nx2N;
        } else if (decode_bin(ctx_[C_PART_MODE + 1])) {
          part_mode_ = PART_2NxN;
        } else {
          part_mode_ = decode_bin(ctx_[C_PART_MODE + 2]) ? PART_Nx2N : PART_NxN;
        }
      } else if (!P.amp) {
        part_mode_ = decode_bin(ctx_[C_PART_MODE + 1]) ? PART_2NxN : PART_Nx2N;
      } else {
        const int horizontal = decode_bin(ctx_[C_PART_MODE + 1]);
        if (decode_bin(ctx_[C_PART_MODE + 3])) {
          part_mode_ = horizontal ? PART_2NxN : PART_Nx2N;
        } else {
          const int which = decode_bypass();
          part_mode_ = horizontal ? (which ? PART_2NxnD : PART_2NxnU) : (which ? PART_nRx2N : PART_nLx2N);
        }
      }
    }
    fill_cu_info(x0, y0, log2_cb, intra ? 2 : 1, flags);
    if (intra) {
      intra_split_ = part_mode_ == PART_NxN;
      if (part_mode_ == PART_2Nx2N && P.pcm && log2_cb >= P.log2_min_pcm && log2_cb <= P.log2_min_pcm + P.log2_diff_pcm) pcm_flag_ = decode_terminate() != 0;
      if (pcm_flag_) {
        flags |= kCuPcm;
        fill_cu_info(x0, y0, log2_cb, 2, flags);
        for (int j = 0; j < n4; ++j) {
          std::memset(&intra_mode_[static_cast<size_t>(idx4(x0, y0 + 4 * j))], 1, static_cast<size_t>(n4));
          std::memset(&tu_nz_[static_cast<size_t>(idx4(x0, y0 + 4 * j))], 0, static_cast<size_t>(n4));
        }
        pcm_sample(x0, y0, log2_cb);
      } else {
        const int np = intra_split_ ? 4 : 1;
        const int pb = intra_split_ ? ncbs / 2 : ncbs;
        int prev_flag[4], mpm_idx[4] = {0, 0, 0, 0}, rem[4] = {0, 0, 0, 0};
        for (int i = 0; i < np; ++i) prev_flag[i] = decode_bin(ctx_[C_PREV_INTRA]);
        for (int i = 0; i < np; ++i) {
          if (prev_flag[i]) {
            mpm_idx[i] = decode_bypass() ? (decode_bypass() ? 2 : 1) : 0;
          } else {
            rem[i] = static_cast<int>(decode_bypass_bits(5));
          }
        }
        for (int i = 0; i < np; ++i) {
          const int xp = x0 + (i & 1) * pb, yp = y0 + (i >> 1) * pb;
          const int mode = derive_luma_mode(xp, yp, prev_flag[i], mpm_idx[i], rem[i]);
          intra_luma_mode_[i] = mode;
          const int m4 = pb >> 2;
          for (int j = 0; j < m4; ++j) std::memset(&intra_mode_[static_cast<size_t>(idx4(xp, yp + 4 * j))], mode, static_cast<size_t>(m4));
        }
        if (!intra_split_)
          for (int i = 1; i < 4; ++i) intra_luma_mode_[i] = intra_luma_mode_[0];
        // intra_chroma_pred_mode (4:2:0: one for the coding unit)
        if (decode_bin(ctx_[C_INTRA_CHROMA])) intra_chroma_mode_ = static_cast<int>(decode_bypass_bits(2));
        else intra_chroma_mode_ = 4;
        {
          static const int kChromaModes[4] = {0, 26, 10, 1};
          const int luma = intra_luma_mode_[0];
          int m;
          if (intra_chroma_mode_ == 4) m = luma;
          else {
            m = kChromaModes[intra_chroma_mode_];
            if (m == luma) m = 34;
          }
          intra_mode_c_ = m;
        }
      }
    } else {
      const int h = ncbs / 2, q = ncbs / 4;
      switch (part_mode_) {
        case PART_2Nx2N: prediction_unit(x0, y0, ncbs, x0, y0, ncbs, ncbs, 0); break;
        case PART_2NxN:
          prediction_unit(x0, y0, ncbs, x0, y0, ncbs, h, 0);
          prediction_unit(x0, y0, ncbs, x0, y0 + h, ncbs, h, 1);
          break;
        case PART_Nx2N:
          prediction_unit(x0, y0, ncbs, x0, y0, h, ncbs, 0);
          prediction_unit(x0, y0, ncbs, x0 + h, y0, h, ncbs, 1);
          break;
        case PART_2NxnU:
          prediction_unit(x0, y0, ncbs, x0, y0, ncbs, q, 0);
          prediction_unit(x0, y0, ncbs, x0, y0 + q, ncbs, ncbs - q, 1);
          break;
        case PART_2NxnD:
          prediction_unit(x0, y0, ncbs, x0, y0, ncbs, ncbs - q, 0);
          prediction_unit(x0, y0, ncbs, x0, y0 + ncbs - q, ncbs, q, 1);
          break;
        case PART_nLx2N:
          prediction_unit(x0, y0, ncbs, x0, y0, q, ncbs, 0);
          prediction_unit(x0, y0, ncbs, x0 + q, y0, ncbs - q, ncbs, 1);
          break;
        case PART_nRx2N:
          prediction_unit(x0, y0, ncbs, x0, y0, ncbs - q, ncbs, 0);
          prediction_unit(x0, y0, ncbs, x0 + ncbs - q, y0, q, ncbs, 1);
          break;
        default:
          prediction_unit(x0, y0, ncbs, x0, y0, h, h, 0);
          prediction_unit(x0, y0, ncbs, x0 + h, y0, h, h, 1);
          prediction_unit(x0, y0, ncbs, x0, y0 + h, h, h, 2);
          prediction_unit(x0, y0, ncbs, x0 + h, y0 + h, h, h, 3);
          break;
      }
      // prediction block edges inside the coding block (8.7.2.3)
      auto vedge = [&](int xe) {
        for (int k = 0; k < n4; ++k) edge_[static_cast<size_t>(idx4(xe, y0 + 4 * k))] |= 2;
      };
      auto hedge = [&](int ye) {
        for (int k = 0; k < n4; ++k) edge_[static_cast<size_t>(idx4(x0 + 4 * k, ye))] |= 8;
      };
      if (part_mode_ == PART_Nx2N || part_mode_ == PART_NxN) vedge(x0 + ncbs / 2);
      if (part_mode_ == PART_nLx2N) vedge(x0 + ncbs / 4);
      if (part_mode_ == PART_nRx2N) vedge(x0 + 3 * ncbs / 4);
      if (part_mode_ == PART_2NxN || part_mode_ == PART_NxN) hedge(y0 + ncbs / 2);
      if (part_mode_ == PART_2NxnU) hedge(y0 + ncbs / 4);
      if (part_mode_ == PART_2NxnD) hedge(y0 + 3 * ncbs / 4);
    }
    for (int j = 0; j < n4; ++j) std::memset(&tu_nz_[static_cast<size_t>(idx4(x0, y0 + 4 * j))], 0, static_cast<size_t>(n4));
    if (!pcm_flag_) {
      bool rqt_root_cbf = true;
      if (!intra && !(part_mode_ == PART_2Nx2N && merge_flag_cu_)) rqt_root_cbf = decode_bin(ctx_[C_RQT_ROOT_CBF]) != 0;
      if (rqt_root_cbf) {
        max_trafo_depth_ = intra ? P.max_th_depth_intra + (intra_split_ ? 1 : 0) : P.max_th_depth_inter;
        transform_tree(x0, y0, x0, y0, log2_cb, 0, 0, true, true);
      } else if (intra) {
        // cannot happen: rqt_root_cbf is inferred to be 1 for intra coding units
      }
    }
  }
  // the coding unit's QpY (after any cu_qp_delta in it)
  set_cu_qp();
  for (int j = 0; j < n4; ++j) std::memset(&qp_y_[static_cast<size_t>(idx4(x0, y0 + 4 * j))], cu_qp_y_, static_cast<size_t>(n4));
  last_cu_qp_ = cu_qp_y_;
}

void Dec::pcm_sample(int x0, int y0, int log2_cb) {
  bitpos_ = (bitpos_ + 7) & ~static_cast<size_t>(7);
  const int n = 1 << log2_cb;
  const int shy = bd_y - P.pcm_bit_depth_luma, shc = bd_c - P.pcm_bit_depth_chroma;
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x) F->y[static_cast<size_t>(y0 + y) * static_cast<size_t>(F->stride_y) + static_cast<size_t>(x0 + x)] = static_cast<uint16_t>(read_bits(P.pcm_bit_depth_luma) << shy);
  const int cn = n / 2, cx = x0 / 2, cy = y0 / 2;
  for (int c = 0; c < 2; ++c) {
    std::vector<uint16_t>& pl = c == 0 ? F->u : F->v;
    for (int y = 0; y < cn; ++y)
      for (int x = 0; x < cn; ++x) pl[static_cast<size_t>(cy + y) * static_cast<size_t>(F->stride_c) + static_cast<size_t>(cx + x)] = static_cast<uint16_t>(read_bits(P.pcm_bit_depth_chroma) << shc);
  }
  cabac_init_engine();
}

int Dec::derive_luma_mode(int xpb, int ypb, int prev_flag, int mpm_idx, int rem_mode) {
  int cand[2];
  for (int k = 0; k < 2; ++k) {
    const int xn = k == 0 ? xpb - 1 : xpb, yn = k == 0 ? ypb : ypb - 1;
    int c = 1;
    if (avail_zs(xpb, ypb, xn, yn)) {
      const size_t i = static_cast<size_t>(idx4(xn, yn));
      if (pred_mode_[i] != 2 || (cu_flags_[i] & kCuPcm)) c = 1;
      else if (k == 1 && yn < ((ypb >> log2_ctb) << log2_ctb)) c = 1;
      else c = intra_mode_[i];
    }
    cand[k] = c;
  }
  int list[3];
  if (cand[0] == cand[1]) {
    if (cand[0] < 2) {
      list[0] = 0;
      list[1] = 1;
      list[2] = 26;
    } else {
      list[0] = cand[0];
      list[1] = 2 + ((cand[0] + 29) % 32);
      list[2] = 2 + ((cand[0] - 2 + 1) % 32);
    }
  } else {
    list[0] = cand[0];
    list[1] = cand[1];
    if (list[0] != 0 && list[1] != 0) list[2] = 0;
    else if (list[0] != 1 && list[1] != 1) list[2] = 1;
    else list[2] = 26;
  }
  if (prev_flag) return list[mpm_idx];
  if (list[0] > list[1]) std::swap(list[0], list[1]);
  if (list[0] > list[2]) std::swap(list[0], list[2]);
  if (list[1] > list[2]) std::swap(list[1], list[2]);
  int mode = rem_mode;
  for (int i = 0; i < 3; ++i)
    if (mode >= list[i]) ++mode;
  return mode;
}

void Dec::prediction_unit(int xcb, int ycb, int ncbs, int xpb, int ypb, int npbw, int npbh, int part_idx) {
  const bool skip = (cu_flags_[static_cast<size_t>(idx4(xcb, ycb))] & kCuSkip) != 0;
  bool merge = skip;
  if (!skip) merge = decode_bin(ctx_[C_MERGE_FLAG]) != 0;
  if (part_idx == 0) merge_flag_cu_ = merge;
  PuMotion m;
  if (merge) {
    int merge_idx = 0;
    if (sh_.max_num_merge_cand > 1) {
      if (decode_bin(ctx_[C_MERGE_IDX])) {
        merge_idx = 1;
        while (merge_idx < sh_.max_num_merge_cand - 1 && decode_bypass()) ++merge_idx;
      }
    }
    derive_merge(xcb, ycb, ncbs, xpb, ypb, npbw, npbh, part_idx, merge_idx, &m);
  } else {
    int idc = 0;   // 0 PRED_L0, 1 PRED_L1, 2 PRED_BI
    if (sh_.slice_type == kSliceB) {
      if (npbw + npbh != 12 && decode_bin(ctx_[C_INTER_PRED_IDC + cu_depth_])) idc = 2;
      else idc = decode_bin(ctx_[C_INTER_PRED_IDC + 4]);
    }
    int mvd[2][2] = {{0, 0}, {0, 0}};
    int ref_idx[2] = {-1, -1}, mvp_flag[2] = {0, 0};
    for (int l = 0; l < 2; ++l) {
      if (idc == (l == 0 ? 1 : 0)) continue;   // list not used
      ref_idx[l] = 0;
      if (sh_.num_ref_idx[l] > 1) {
        const int cmax = sh_.num_ref_idx[l] - 1;
        int v = 0;
        if (decode_bin(ctx_[C_REF_IDX])) {
          v = 1;
          if (cmax > 1) {
            if (decode_bin(ctx_[C_REF_IDX + 1])) {
              v = 2;
              while (v < cmax && decode_bypass()) ++v;
            }
          }
        }
        ref_idx[l] = v;
      }
      if (l == 1 && sh_.mvd_l1_zero && idc == 2) {
        mvd[1][0] = mvd[1][1] = 0;
      } else {
        const int g0x = decode_bin(ctx_[C_MVD_G0]), g0y = decode_bin(ctx_[C_MVD_G0]);
        int g1x = 0, g1y = 0;
        if (g0x) g1x = decode_bin(ctx_[C_MVD_G1]);
        if (g0y) g1y = decode_bin(ctx_[C_MVD_G1]);
        auto eg1 = [&]() {
          int k = 1, v = 0;
          while (decode_bypass() && k < 31) {
            v += 1 << k;
            ++k;
          }
          return v + static_cast<int>(decode_bypass_bits(k));
        };
        int vx = 0, vy = 0;
        if (g0x) {
          vx = g1x ? eg1() + 2 : 1;
          if (decode_bypass()) vx = -vx;
        }
        if (g0y) {
          vy = g1y ? eg1() + 2 : 1;
          if (decode_bypass()) vy = -vy;
        }
        mvd[l][0] = vx;
        mvd[l][1] = vy;
      }
      mvp_flag[l] = decode_bin(ctx_[C_MVP]);
    }
    for (int l = 0; l < 2; ++l) {
      if (ref_idx[l] < 0) continue;
      if (ref_idx[l] >= sh_.num_ref_idx[l]) {
        fail("ref_idx out of range");
        return;
      }
      int16_t mvp[2];
      derive_mvp(xcb, ycb, ncbs, xpb, ypb, npbw, npbh, part_idx, ref_idx[l], l, mvp_flag[l], mvp);
      for (int c = 0; c < 2; ++c) {
        const int u = (mvp[c] + mvd[l][c] + 65536) & 0xFFFF;
        m.mv[l][c] = static_cast<int16_t>(u >= 32768 ? u - 65536 : u);
      }
      m.ref_idx[l] = static_cast<int8_t>(ref_idx[l]);
    }
  }
  if (!ok_) return;
  store_motion(xpb, ypb, npbw, npbh, m);
  inter_predict(xpb, ypb, npbw, npbh, m);
}

void Dec::delta_qp_syntax() {
  if (!P.cu_qp_delta_enabled || is_cu_qp_delta_coded_) return;
  int v = 0;
  if (decode_bin(ctx_[C_QP_DELTA])) {
    v = 1;
    while (v < 5 && decode_bin(ctx_[C_QP_DELTA + 1])) ++v;
    if (v == 5) {
      int k = 0, e = 0;
      while (decode_bypass() && k < 31) {
        e += 1 << k;
        ++k;
      }
      v += e + static_cast<int>(decode_bypass_bits(k));
    }
  }
  int val = v;
  if (v && decode_bypass()) val = -v;
  is_cu_qp_delta_coded_ = true;
  cu_qp_delta_val_ = val;
  if (!qp_pred_valid_) compute_qp_pred(cu_x_, cu_y_);
  set_cu_qp();
}

void Dec::chroma_qp_offset_syntax() {
  if (!sh_.cu_chroma_qp_offset_enabled || is_cu_chroma_qp_offset_coded_) return;
  const int flag = decode_bin(ctx_[C_CQP_FLAG]);
  int idx = 0;
  if (flag && P.chroma_qp_offset_list_len > 1) {
    const int cmax = P.chroma_qp_offset_list_len - 1;
    while (idx < cmax && decode_bin(ctx_[C_CQP_IDX])) ++idx;
  }
  is_cu_chroma_qp_offset_coded_ = true;
  if (flag) {
    cu_qp_offset_cb_ = P.cb_qp_offset_list[idx];
    cu_qp_offset_cr_ = P.cr_qp_offset_list[idx];
  } else {
    cu_qp_offset_cb_ = cu_qp_offset_cr_ = 0;
  }
}

void Dec::transform_tree(int x0, int y0, int xbase, int ybase, int log2_tb, int depth, int blk_idx, bool parent_cb, bool parent_cr) {
  if (!ok_) return;
  bool split;
  if (log2_tb <= log2_max_tb && log2_tb > log2_min_tb && depth < max_trafo_depth_ && !(intra_split_ && depth == 0)) {
    split = decode_bin(ctx_[C_SPLIT_TRANSFORM + 5 - log2_tb]) != 0;
  } else {
    const bool inter_split = P.max_th_depth_inter == 0 && !cu_intra_ && part_mode_ != PART_2Nx2N && depth == 0;
    split = log2_tb > log2_max_tb || (intra_split_ && depth == 0) || inter_split;
  }
  bool cb = false, cr = false;
  if (log2_tb > 2) {
    if (depth == 0 || parent_cb) cb = decode_bin(ctx_[C_CBF_CHROMA + depth]) != 0;
    if (depth == 0 || parent_cr) cr = decode_bin(ctx_[C_CBF_CHROMA + depth]) != 0;
  } else {
    cb = parent_cb;
    cr = parent_cr;
  }
  if (split) {
    const int h = 1 << (log2_tb - 1);
    transform_tree(x0, y0, x0, y0, log2_tb - 1, depth + 1, 0, cb, cr);
    transform_tree(x0 + h, y0, x0, y0, log2_tb - 1, depth + 1, 1, cb, cr);
    transform_tree(x0, y0 + h, x0, y0, log2_tb - 1, depth + 1, 2, cb, cr);
    transform_tree(x0 + h, y0 + h, x0, y0, log2_tb - 1, depth + 1, 3, cb, cr);
  } else {
    bool cbf_luma = true;
    if (cu_intra_ || depth != 0 || cb || cr) cbf_luma = decode_bin(ctx_[C_CBF_LUMA + (depth == 0 ? 1 : 0)]) != 0;
    transform_unit(x0, y0, xbase, ybase, log2_tb, depth, blk_idx, cbf_luma, cb, cr);
  }
}

void Dec::transform_unit(int x0, int y0, int xbase, int ybase, int log2_tb, int depth, int blk_idx, bool cbf_luma, bool cb, bool cr) {
  (void)depth;
  const int n = 1 << log2_tb;
  const int n4 = n >> 2;
  // transform block edges
  for (int k = 0; k < n4; ++k) {
    edge_[static_cast<size_t>(idx4(x0, y0 + 4 * k))] |= 1;
    edge_[static_cast<size_t>(idx4(x0 + 4 * k, y0))] |= 4;
  }
  const bool cbf_chroma = cb || cr;
  if (cbf_luma || cbf_chroma) {
    delta_qp_syntax();
    if (cbf_chroma && !cu_transquant_bypass_) chroma_qp_offset_syntax();
  }
  const int pu = intra_split_ ? ((y0 - cu_y_) >= (1 << cu_log2_) / 2 ? 2 : 0) + ((x0 - cu_x_) >= (1 << cu_log2_) / 2 ? 1 : 0) : 0;
  const int luma_mode = intra_luma_mode_[pu];
  reconstruct_tb(x0, y0, log2_tb, 0, cbf_luma, luma_mode, cu_intra_);
  if (cbf_luma)
    for (int j = 0; j < n4; ++j) std::memset(&tu_nz_[static_cast<size_t>(idx4(x0, y0 + 4 * j))], 1, static_cast<size_t>(n4));
  if (log2_tb > 2) {
    reconstruct_tb(x0 / 2, y0 / 2, log2_tb - 1, 1, cb, intra_mode_c_, cu_intra_);
    reconstruct_tb(x0 / 2, y0 / 2, log2_tb - 1, 2, cr, intra_mode_c_, cu_intra_);
  } else if (blk_idx == 3) {
    reconstruct_tb(xbase / 2, ybase / 2, 2, 1, cb, intra_mode_c_, cu_intra_);
    reconstruct_tb(xbase / 2, ybase / 2, 2, 2, cr, intra_mode_c_, cu_intra_);
  }
}
