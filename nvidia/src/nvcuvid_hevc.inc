// Part of nvcuvid_api.cpp: HEVC -- cuvidDecodePicture for pictures the application describes in CUVIDHEVCPICPARAMS, decoder
// creation as the card answers it, and the parser callbacks that translate hevc_parser.cpp's events to the application's.

namespace {

template <class T>
std::vector<T> window_to_target_t(const T* src, int stride, int x0, int y0, int w, int h, uint32_t tw, uint32_t th) {
  std::vector<T> win(static_cast<size_t>(w) * static_cast<size_t>(h));
  for (int y = 0; y < h; ++y) std::memcpy(&win[static_cast<size_t>(y) * static_cast<size_t>(w)], src + static_cast<ptrdiff_t>(y0 + y) * stride + x0, static_cast<size_t>(w) * sizeof(T));
  if (static_cast<uint32_t>(w) == tw && static_cast<uint32_t>(h) == th) return win;
  ScaleMode sm;
  sm.down_num = 1;
  sm.down_den = 2;
  return resample(win, static_cast<uint32_t>(w), static_cast<uint32_t>(h), tw, th, sm);
}

// The decoded picture as the decoder's surface: NV12 for 8-bit output, P016 (samples in the high bits of 16) otherwise.
CUresult write_hevc_surface(Decoder& d, const vgpu_hevc::Frame& f, int bit_depth, Surface& s) {
  const int fw = f.width, fh = f.height;
  int l = d.info.display_area.left, t = d.info.display_area.top, r = d.info.display_area.right, b = d.info.display_area.bottom;
  if (l == 0 && t == 0 && r == 0 && b == 0) {
    r = static_cast<int>(d.width);
    b = static_cast<int>(d.height);
  }
  l = std::max(0, std::min(l, fw));
  t = std::max(0, std::min(t, fh));
  r = std::max(l + 2, std::min(r, fw));
  b = std::max(t + 2, std::min(b, fh));
  l &= ~1;
  t &= ~1;
  const int w = r - l, h = b - t;
  const uint32_t tw = d.target_w, th = d.target_h, tcw = (tw + 1) / 2, tch = (th + 1) / 2;
  const int cw = (w + 1) / 2, chh = (h + 1) / 2;
  if (!ensure_surface(d, s)) return CUDA_ERROR_OUT_OF_MEMORY;
  if (d.out_bytes == 1) {
    // 8-bit samples; a deeper picture in an NV12 decoder is rounded down to 8 bits (to within one level of the card's conversion)
    auto to8 = [&](const std::vector<uint16_t>& pl) {
      if (bit_depth == 8) return std::vector<uint8_t>(pl.begin(), pl.end());
      std::vector<uint8_t> o(pl.size());
      const int sh = bit_depth - 8, add = 1 << (sh - 1);
      for (size_t i = 0; i < pl.size(); ++i) o[i] = static_cast<uint8_t>(std::min(255, (pl[i] + add) >> sh));
      return o;
    };
    const std::vector<uint8_t> y8 = to8(f.y), u8 = to8(f.u), v8 = to8(f.v);
    const std::vector<uint8_t> yt = window_to_target_t<uint8_t>(y8.data(), f.stride_y, l, t, w, h, tw, th);
    const std::vector<uint8_t> ut = window_to_target_t<uint8_t>(u8.data(), f.stride_c, l / 2, t / 2, cw, chh, tcw, tch),
                               vt = window_to_target_t<uint8_t>(v8.data(), f.stride_c, l / 2, t / 2, cw, chh, tcw, tch);
    std::vector<uint8_t> uv(static_cast<size_t>(tcw) * 2 * tch);
    for (size_t i = 0; i < ut.size(); ++i) {
      uv[2 * i] = ut[i];
      uv[2 * i + 1] = vt[i];
    }
    cudaMemcpy2D(s.dev, d.pitch, yt.data(), tw, tw, th, cudaMemcpyHostToDevice);
    cudaMemcpy2D(static_cast<uint8_t*>(s.dev) + static_cast<size_t>(d.pitch) * th, d.pitch, uv.data(), tcw * 2, tcw * 2, tch, cudaMemcpyHostToDevice);
    return CUDA_SUCCESS;
  }
  const int up = 16 - bit_depth;
  auto to16 = [&](const std::vector<uint16_t>& pl) {
    std::vector<uint16_t> o(pl.size());
    for (size_t i = 0; i < pl.size(); ++i) o[i] = static_cast<uint16_t>(pl[i] << up);
    return o;
  };
  const std::vector<uint16_t> y16 = to16(f.y), u16 = to16(f.u), v16 = to16(f.v);
  const std::vector<uint16_t> yt = window_to_target_t<uint16_t>(y16.data(), f.stride_y, l, t, w, h, tw, th);
  const std::vector<uint16_t> ut = window_to_target_t<uint16_t>(u16.data(), f.stride_c, l / 2, t / 2, cw, chh, tcw, tch),
                              vt = window_to_target_t<uint16_t>(v16.data(), f.stride_c, l / 2, t / 2, cw, chh, tcw, tch);
  std::vector<uint16_t> uv(static_cast<size_t>(tcw) * 2 * tch);
  for (size_t i = 0; i < ut.size(); ++i) {
    uv[2 * i] = ut[i];
    uv[2 * i + 1] = vt[i];
  }
  cudaMemcpy2D(s.dev, d.pitch, yt.data(), tw * 2, tw * 2, th, cudaMemcpyHostToDevice);
  cudaMemcpy2D(static_cast<uint8_t*>(s.dev) + static_cast<size_t>(d.pitch) * th, d.pitch, uv.data(), tcw * 4, tcw * 4, tch, cudaMemcpyHostToDevice);
  return CUDA_SUCCESS;
}

// CUVIDHEVCPICPARAMS -> the decoder's parameters (the card's decoder reads the same fields).
void hevc_params_from_cuvid(const CUVIDPICPARAMS& pic, const CUVIDHEVCPICPARAMS& h, vgpu_hevc::PicParams* out) {
  using vgpu_hevc::PicParams;
  PicParams& pp = *out;
  pp.width = h.pic_width_in_luma_samples;
  pp.height = h.pic_height_in_luma_samples;
  pp.log2_min_cb = h.log2_min_luma_coding_block_size_minus3 + 3;
  pp.log2_diff_cb = h.log2_diff_max_min_luma_coding_block_size;
  pp.log2_min_tb = h.log2_min_transform_block_size_minus2 + 2;
  pp.log2_diff_tb = h.log2_diff_max_min_transform_block_size;
  pp.max_th_depth_intra = h.max_transform_hierarchy_depth_intra;
  pp.max_th_depth_inter = h.max_transform_hierarchy_depth_inter;
  pp.amp = h.amp_enabled_flag;
  pp.sao = h.sample_adaptive_offset_enabled_flag;
  pp.scaling_list_enabled = h.scaling_list_enable_flag;
  pp.strong_intra_smoothing = h.strong_intra_smoothing_enabled_flag;
  pp.sps_temporal_mvp = h.sps_temporal_mvp_enabled_flag;
  pp.pcm = h.pcm_enabled_flag;
  pp.pcm_loop_filter_disabled = h.pcm_loop_filter_disabled_flag;
  pp.pcm_bit_depth_luma = h.pcm_sample_bit_depth_luma_minus1 + 1;
  pp.pcm_bit_depth_chroma = h.pcm_sample_bit_depth_chroma_minus1 + 1;
  pp.log2_min_pcm = h.log2_min_pcm_luma_coding_block_size_minus3 + 3;
  pp.log2_diff_pcm = h.log2_diff_max_min_pcm_luma_coding_block_size;
  pp.bit_depth_luma = 8 + h.bit_depth_luma_minus8;
  pp.bit_depth_chroma = 8 + h.bit_depth_chroma_minus8;
  pp.chroma_format_idc = 1;
  pp.separate_colour_plane = h.separate_colour_plane_flag != 0;
  pp.log2_max_poc_lsb = h.log2_max_pic_order_cnt_lsb_minus4 + 4;
  pp.num_short_term_ref_pic_sets = h.num_short_term_ref_pic_sets;
  pp.long_term_ref_pics_present = h.long_term_ref_pics_present_flag;
  pp.num_long_term_ref_pics_sps = h.num_long_term_ref_pics_sps;
  pp.irap = h.IrapPicFlag;
  pp.idr = h.IdrPicFlag;
  pp.sps_range_extension = h.sps_range_extension_flag;
  pp.pps_range_extension = h.pps_range_extension_flag;
  pp.transform_skip_rotation = h.transform_skip_rotation_enabled_flag;
  pp.transform_skip_context = h.transform_skip_context_enabled_flag;
  pp.implicit_rdpcm = h.implicit_rdpcm_enabled_flag;
  pp.explicit_rdpcm = h.explicit_rdpcm_enabled_flag;
  pp.extended_precision = h.extended_precision_processing_flag;
  pp.intra_smoothing_disabled = h.intra_smoothing_disabled_flag;
  pp.high_precision_offsets = h.high_precision_offsets_enabled_flag;
  pp.persistent_rice_adaptation = h.persistent_rice_adaptation_enabled_flag;
  pp.cabac_bypass_alignment = h.cabac_bypass_alignment_enabled_flag;
  pp.log2_max_transform_skip = h.log2_max_transform_skip_block_size_minus2 + 2;
  pp.log2_sao_offset_scale_luma = h.log2_sao_offset_scale_luma;
  pp.log2_sao_offset_scale_chroma = h.log2_sao_offset_scale_chroma;
  pp.cross_component_prediction = h.cross_component_prediction_enabled_flag;
  pp.chroma_qp_offset_list_enabled = h.chroma_qp_offset_list_enabled_flag;
  pp.diff_cu_chroma_qp_offset_depth = h.diff_cu_chroma_qp_offset_depth;
  pp.chroma_qp_offset_list_len = h.chroma_qp_offset_list_enabled_flag ? h.chroma_qp_offset_list_len_minus1 + 1 : 0;
  for (int i = 0; i < 6; ++i) {
    pp.cb_qp_offset_list[i] = h.cb_qp_offset_list[i];
    pp.cr_qp_offset_list[i] = h.cr_qp_offset_list[i];
  }
  pp.dependent_slice_segments_enabled = h.dependent_slice_segments_enabled_flag;
  pp.slice_segment_header_extension_present = h.slice_segment_header_extension_present_flag;
  pp.sign_data_hiding = h.sign_data_hiding_enabled_flag;
  pp.cu_qp_delta_enabled = h.cu_qp_delta_enabled_flag;
  pp.diff_cu_qp_delta_depth = h.diff_cu_qp_delta_depth;
  pp.init_qp = 26 + h.init_qp_minus26;
  pp.cb_qp_offset = h.pps_cb_qp_offset;
  pp.cr_qp_offset = h.pps_cr_qp_offset;
  pp.constrained_intra_pred = h.constrained_intra_pred_flag;
  pp.weighted_pred = h.weighted_pred_flag;
  pp.weighted_bipred = h.weighted_bipred_flag;
  pp.transform_skip_enabled = h.transform_skip_enabled_flag;
  pp.transquant_bypass_enabled = h.transquant_bypass_enabled_flag;
  pp.entropy_coding_sync = h.entropy_coding_sync_enabled_flag;
  pp.log2_parallel_merge_level = h.log2_parallel_merge_level_minus2 + 2;
  pp.num_extra_slice_header_bits = h.num_extra_slice_header_bits;
  pp.loop_filter_across_tiles = h.loop_filter_across_tiles_enabled_flag;
  pp.loop_filter_across_slices = h.loop_filter_across_slices_enabled_flag;
  pp.output_flag_present = h.output_flag_present_flag;
  pp.num_ref_idx_default[0] = h.num_ref_idx_l0_default_active_minus1 + 1;
  pp.num_ref_idx_default[1] = h.num_ref_idx_l1_default_active_minus1 + 1;
  pp.lists_modification_present = h.lists_modification_present_flag;
  pp.cabac_init_present = h.cabac_init_present_flag;
  pp.slice_chroma_qp_offsets_present = h.pps_slice_chroma_qp_offsets_present_flag;
  pp.deblocking_override_enabled = h.deblocking_filter_override_enabled_flag;
  pp.pps_deblocking_disabled = h.pps_deblocking_filter_disabled_flag;
  pp.pps_beta_offset_div2 = h.pps_beta_offset_div2;
  pp.pps_tc_offset_div2 = h.pps_tc_offset_div2;
  pp.tiles_enabled = h.tiles_enabled_flag;
  pp.uniform_spacing = h.uniform_spacing_flag;
  pp.num_tile_cols = h.num_tile_columns_minus1 + 1;
  pp.num_tile_rows = h.num_tile_rows_minus1 + 1;
  for (int i = 0; i < 20; ++i) {
    pp.column_width_minus1[i] = h.column_width_minus1[i];
    pp.row_height_minus1[i] = h.row_height_minus1[i];
  }
  pp.num_bits_st_rps = h.NumBitsForShortTermRPSInSlice;
  pp.num_poc_total_curr = h.NumPocTotalCurr;
  pp.num_st_before = std::max(0, std::min(8, h.NumPocStCurrBefore));
  pp.num_st_after = std::max(0, std::min(8, h.NumPocStCurrAfter));
  pp.num_lt_curr = std::max(0, std::min(8, h.NumPocLtCurr));
  pp.curr_poc = h.CurrPicOrderCntVal;
  for (int i = 0; i < 8; ++i) {
    pp.st_before[i] = h.RefPicSetStCurrBefore[i];
    pp.st_after[i] = h.RefPicSetStCurrAfter[i];
    pp.lt_curr[i] = h.RefPicSetLtCurr[i];
  }
  std::memcpy(pp.sl4, h.ScalingList4x4, sizeof pp.sl4);
  std::memcpy(pp.sl8, h.ScalingList8x8, sizeof pp.sl8);
  std::memcpy(pp.sl16, h.ScalingList16x16, sizeof pp.sl16);
  std::memcpy(pp.sl32, h.ScalingList32x32, sizeof pp.sl32);
  std::memcpy(pp.dc16, h.ScalingListDCCoeff16x16, sizeof pp.dc16);
  std::memcpy(pp.dc32, h.ScalingListDCCoeff32x32, sizeof pp.dc32);
  (void)pic;
}

CUresult decode_hevc_picture(Decoder& d, CUVIDPICPARAMS* p, Surface& s) {
  const CUVIDHEVCPICPARAMS& h = p->CodecSpecific.hevc;
  vgpu_hevc::PicParams pp;
  hevc_params_from_cuvid(*p, h, &pp);
  if (pp.width <= 0 || pp.height <= 0 || pp.width > 8192 || pp.height > 8192 || (pp.width & 7) || (pp.height & 7)) return CUDA_ERROR_INVALID_IMAGE;
  std::unique_ptr<vgpu_hevc::Frame>& cur = d.hevc_frames[static_cast<size_t>(p->CurrPicIdx)];
  cur = std::make_unique<vgpu_hevc::Frame>();
  cur->alloc(pp.width, pp.height);
  cur->poc = pp.curr_poc;
  cur->uid = g_next_uid++;
  // the reference pictures: slots of the picture table
  for (int i = 0; i < 16; ++i) {
    const int idx = h.RefPicIdx[i];
    if (idx < 0 || static_cast<size_t>(idx) >= d.hevc_frames.size() || idx == p->CurrPicIdx) continue;
    vgpu_hevc::Frame* rf = d.hevc_frames[static_cast<size_t>(idx)].get();
    if (!rf || rf->y.empty()) continue;
    pp.refs[i].frame = rf;
    pp.refs[i].poc = h.PicOrderCntVal[i];
    pp.refs[i].long_term = h.IsLongTerm[i] != 0;
    rf->poc = h.PicOrderCntVal[i];
  }
  // the slice segment NAL units: each starts at its offset with a start code
  std::vector<vgpu_hevc::SliceData> slices;
  const unsigned n = p->nNumSlices ? p->nNumSlices : 1;
  const unsigned zero = 0;
  const unsigned* offs = p->pSliceDataOffsets ? p->pSliceDataOffsets : &zero;
  for (unsigned i = 0; i < n; ++i) {
    size_t a = offs[i], b = (i + 1 < n && p->pSliceDataOffsets) ? offs[i + 1] : p->nBitstreamDataLen;
    if (a >= p->nBitstreamDataLen) break;
    b = std::min<size_t>(b, p->nBitstreamDataLen);
    const uint8_t* base = p->pBitstreamData;
    size_t k = a;
    while (k + 3 <= b && !(base[k] == 0 && base[k + 1] == 0 && base[k + 2] == 1) && k - a < 4) ++k;
    if (k + 3 <= b && base[k] == 0 && base[k + 1] == 0 && base[k + 2] == 1) a = k + 3;
    while (b > a && base[b - 1] == 0) --b;
    if (b > a) slices.push_back({base + a, b - a});
  }
  std::string err;
  const bool ok = vgpu_hevc::decode_picture(cur.get(), pp, slices, &err);
  if (!ok) {
    say_once(("HEVC picture not decoded: " + err).c_str());
    const CUresult wr = write_hevc_surface(d, *cur, pp.bit_depth_luma, s);
    if (wr != CUDA_SUCCESS) return wr;
    return CUDA_ERROR_INVALID_IMAGE;
  }
  return write_hevc_surface(d, *cur, pp.bit_depth_luma, s);
}

// ---- the parser's sink: events to the application's callbacks ------------------------------------------------

class HevcCuvidSink : public vgpu_hevc::ParserSink {
 public:
  explicit HevcCuvidSink(const CUVIDPARSERPARAMS& params) : params_(params) {}

  int sequence(const vgpu_hevc::SeqInfo& i) override {
    CUVIDEOFORMAT f{};
    f.codec = cudaVideoCodec_HEVC;
    f.frame_rate.numerator = i.fps_num;
    f.frame_rate.denominator = i.fps_den;
    f.progressive_sequence = i.progressive ? 1 : 0;
    f.bit_depth_luma_minus8 = static_cast<unsigned char>(i.bit_depth_luma_minus8);
    f.bit_depth_chroma_minus8 = static_cast<unsigned char>(i.bit_depth_chroma_minus8);
    f.min_num_decode_surfaces = static_cast<unsigned char>(std::min(255, i.min_surfaces));
    f.coded_width = static_cast<unsigned>(i.coded_w);
    f.coded_height = static_cast<unsigned>(i.coded_h);
    f.display_area = {i.disp_left, i.disp_top, i.disp_right, i.disp_bottom};
    f.chroma_format = static_cast<cudaVideoChromaFormat>(i.chroma_format);
    f.bitrate = i.bitrate;
    f.display_aspect_ratio = {i.dar_x, i.dar_y};
    f.video_signal_description.video_format = static_cast<unsigned char>(i.video_format & 7);
    f.video_signal_description.video_full_range_flag = static_cast<unsigned char>(i.full_range & 1);
    f.video_signal_description.color_primaries = static_cast<unsigned char>(i.primaries);
    f.video_signal_description.transfer_characteristics = static_cast<unsigned char>(i.transfer);
    f.video_signal_description.matrix_coefficients = static_cast<unsigned char>(i.matrix);
    return params_.pfnSequenceCallback ? params_.pfnSequenceCallback(params_.pUserData, &f) : 1;
  }

  int decode(vgpu_hevc::PicDesc& d) override {
    pic_ = std::make_unique<CUVIDPICPARAMS>();
    std::memset(pic_.get(), 0, sizeof *pic_);
    CUVIDPICPARAMS& p = *pic_;
    const vgpu_hevc::PicParams& pp = d.pp;
    p.PicWidthInMbs = (pp.width + 15) / 16;
    p.FrameHeightInMbs = (pp.height + 15) / 16;
    p.CurrPicIdx = d.curr_pic_idx;
    p.nBitstreamDataLen = static_cast<unsigned>(d.data.size());
    p.pBitstreamData = d.data.data();
    p.nNumSlices = static_cast<unsigned>(d.slice_offsets.size());
    p.pSliceDataOffsets = d.slice_offsets.data();
    p.ref_pic_flag = d.ref_pic ? 1 : 0;
    p.intra_pic_flag = d.intra_pic ? 1 : 0;
    CUVIDHEVCPICPARAMS& h = p.CodecSpecific.hevc;
    h.pic_width_in_luma_samples = pp.width;
    h.pic_height_in_luma_samples = pp.height;
    h.log2_min_luma_coding_block_size_minus3 = static_cast<unsigned char>(pp.log2_min_cb - 3);
    h.log2_diff_max_min_luma_coding_block_size = static_cast<unsigned char>(pp.log2_diff_cb);
    h.log2_min_transform_block_size_minus2 = static_cast<unsigned char>(pp.log2_min_tb - 2);
    h.log2_diff_max_min_transform_block_size = static_cast<unsigned char>(pp.log2_diff_tb);
    h.pcm_enabled_flag = pp.pcm;
    h.log2_min_pcm_luma_coding_block_size_minus3 = static_cast<unsigned char>(pp.pcm ? pp.log2_min_pcm - 3 : 0);
    h.log2_diff_max_min_pcm_luma_coding_block_size = static_cast<unsigned char>(pp.pcm ? pp.log2_diff_pcm : 0);
    h.pcm_sample_bit_depth_luma_minus1 = static_cast<unsigned char>(pp.pcm ? pp.pcm_bit_depth_luma - 1 : 0);
    h.pcm_sample_bit_depth_chroma_minus1 = static_cast<unsigned char>(pp.pcm ? pp.pcm_bit_depth_chroma - 1 : 0);
    h.pcm_loop_filter_disabled_flag = pp.pcm_loop_filter_disabled;
    h.explicit_rdpcm_enabled_flag = pp.explicit_rdpcm;
    h.extended_precision_processing_flag = pp.extended_precision;
    h.intra_smoothing_disabled_flag = pp.intra_smoothing_disabled;
    h.persistent_rice_adaptation_enabled_flag = pp.persistent_rice_adaptation;
    h.cabac_bypass_alignment_enabled_flag = pp.cabac_bypass_alignment;
    h.pps_range_extension_flag = pp.pps_range_extension;
    h.cross_component_prediction_enabled_flag = pp.cross_component_prediction;
    h.chroma_qp_offset_list_enabled_flag = pp.chroma_qp_offset_list_enabled;
    h.diff_cu_chroma_qp_offset_depth = static_cast<unsigned char>(pp.diff_cu_chroma_qp_offset_depth);
    h.chroma_qp_offset_list_len_minus1 = static_cast<unsigned char>(pp.chroma_qp_offset_list_len > 0 ? pp.chroma_qp_offset_list_len - 1 : 0);
    for (int i = 0; i < 6; ++i) {
      h.cb_qp_offset_list[i] = static_cast<signed char>(pp.cb_qp_offset_list[i]);
      h.cr_qp_offset_list[i] = static_cast<signed char>(pp.cr_qp_offset_list[i]);
    }
    h.NumBitsForShortTermRPSInSlice = pp.num_bits_st_rps;
    h.NumDeltaPocsOfRefRpsIdx = d.num_delta_pocs_of_ref_rps;
    h.NumPocTotalCurr = pp.num_poc_total_curr;
    h.NumPocStCurrBefore = pp.num_st_before;
    h.NumPocStCurrAfter = pp.num_st_after;
    h.NumPocLtCurr = pp.num_lt_curr;
    h.CurrPicOrderCntVal = pp.curr_poc;
    for (int i = 0; i < 16; ++i) {
      h.RefPicIdx[i] = d.slots[i].pic_idx;
      h.PicOrderCntVal[i] = d.slots[i].pic_idx >= 0 ? d.slots[i].poc : 0;
      h.IsLongTerm[i] = d.slots[i].pic_idx >= 0 && d.slots[i].long_term;
    }
    for (int i = 0; i < 8; ++i) {
      h.RefPicSetStCurrBefore[i] = pp.st_before[i];
      h.RefPicSetStCurrAfter[i] = pp.st_after[i];
      h.RefPicSetLtCurr[i] = pp.lt_curr[i];
    }
    h.strong_intra_smoothing_enabled_flag = pp.strong_intra_smoothing;
    h.max_transform_hierarchy_depth_intra = static_cast<unsigned char>(pp.max_th_depth_intra);
    h.max_transform_hierarchy_depth_inter = static_cast<unsigned char>(pp.max_th_depth_inter);
    h.amp_enabled_flag = pp.amp;
    h.separate_colour_plane_flag = pp.separate_colour_plane;
    h.log2_max_pic_order_cnt_lsb_minus4 = static_cast<unsigned char>(pp.log2_max_poc_lsb - 4);
    h.num_short_term_ref_pic_sets = static_cast<unsigned char>(pp.num_short_term_ref_pic_sets);
    h.long_term_ref_pics_present_flag = pp.long_term_ref_pics_present;
    h.num_long_term_ref_pics_sps = static_cast<unsigned char>(pp.num_long_term_ref_pics_sps);
    h.sps_temporal_mvp_enabled_flag = pp.sps_temporal_mvp;
    h.sample_adaptive_offset_enabled_flag = pp.sao;
    h.scaling_list_enable_flag = pp.scaling_list_enabled;
    h.IrapPicFlag = pp.irap;
    h.IdrPicFlag = pp.idr;
    h.bit_depth_luma_minus8 = static_cast<unsigned char>(pp.bit_depth_luma - 8);
    h.bit_depth_chroma_minus8 = static_cast<unsigned char>(pp.bit_depth_chroma - 8);
    h.log2_max_transform_skip_block_size_minus2 = static_cast<unsigned char>(pp.log2_max_transform_skip - 2);
    h.log2_sao_offset_scale_luma = static_cast<unsigned char>(pp.log2_sao_offset_scale_luma);
    h.log2_sao_offset_scale_chroma = static_cast<unsigned char>(pp.log2_sao_offset_scale_chroma);
    h.high_precision_offsets_enabled_flag = pp.high_precision_offsets;
    h.dependent_slice_segments_enabled_flag = pp.dependent_slice_segments_enabled;
    h.slice_segment_header_extension_present_flag = pp.slice_segment_header_extension_present;
    h.sign_data_hiding_enabled_flag = pp.sign_data_hiding;
    h.cu_qp_delta_enabled_flag = pp.cu_qp_delta_enabled;
    h.diff_cu_qp_delta_depth = static_cast<unsigned char>(pp.diff_cu_qp_delta_depth);
    h.init_qp_minus26 = static_cast<signed char>(pp.init_qp - 26);
    h.pps_cb_qp_offset = static_cast<signed char>(pp.cb_qp_offset);
    h.pps_cr_qp_offset = static_cast<signed char>(pp.cr_qp_offset);
    h.constrained_intra_pred_flag = pp.constrained_intra_pred;
    h.weighted_pred_flag = pp.weighted_pred;
    h.weighted_bipred_flag = pp.weighted_bipred;
    h.transform_skip_enabled_flag = pp.transform_skip_enabled;
    h.transquant_bypass_enabled_flag = pp.transquant_bypass_enabled;
    h.entropy_coding_sync_enabled_flag = pp.entropy_coding_sync;
    h.log2_parallel_merge_level_minus2 = static_cast<unsigned char>(pp.log2_parallel_merge_level - 2);
    h.num_extra_slice_header_bits = static_cast<unsigned char>(pp.num_extra_slice_header_bits);
    h.loop_filter_across_tiles_enabled_flag = pp.loop_filter_across_tiles;
    h.loop_filter_across_slices_enabled_flag = pp.loop_filter_across_slices;
    h.output_flag_present_flag = pp.output_flag_present;
    h.num_ref_idx_l0_default_active_minus1 = static_cast<unsigned char>(pp.num_ref_idx_default[0] - 1);
    h.num_ref_idx_l1_default_active_minus1 = static_cast<unsigned char>(pp.num_ref_idx_default[1] - 1);
    h.lists_modification_present_flag = pp.lists_modification_present;
    h.cabac_init_present_flag = pp.cabac_init_present;
    h.pps_slice_chroma_qp_offsets_present_flag = pp.slice_chroma_qp_offsets_present;
    h.deblocking_filter_override_enabled_flag = pp.deblocking_override_enabled;
    h.pps_deblocking_filter_disabled_flag = pp.pps_deblocking_disabled;
    h.pps_beta_offset_div2 = static_cast<signed char>(pp.pps_beta_offset_div2);
    h.pps_tc_offset_div2 = static_cast<signed char>(pp.pps_tc_offset_div2);
    h.tiles_enabled_flag = pp.tiles_enabled;
    h.uniform_spacing_flag = pp.uniform_spacing;
    h.num_tile_columns_minus1 = static_cast<unsigned char>(pp.num_tile_cols - 1);
    h.num_tile_rows_minus1 = static_cast<unsigned char>(pp.num_tile_rows - 1);
    for (int i = 0; i < 20; ++i) {
      h.column_width_minus1[i] = static_cast<unsigned short>(pp.column_width_minus1[i]);
      h.row_height_minus1[i] = static_cast<unsigned short>(pp.row_height_minus1[i]);
    }
    h.sps_range_extension_flag = pp.sps_range_extension;
    h.transform_skip_rotation_enabled_flag = pp.transform_skip_rotation;
    h.transform_skip_context_enabled_flag = pp.transform_skip_context;
    h.implicit_rdpcm_enabled_flag = pp.implicit_rdpcm;
    std::memcpy(h.ScalingList4x4, pp.sl4, sizeof h.ScalingList4x4);
    std::memcpy(h.ScalingList8x8, pp.sl8, sizeof h.ScalingList8x8);
    std::memcpy(h.ScalingList16x16, pp.sl16, sizeof h.ScalingList16x16);
    std::memcpy(h.ScalingList32x32, pp.sl32, sizeof h.ScalingList32x32);
    std::memcpy(h.ScalingListDCCoeff16x16, pp.dc16, sizeof h.ScalingListDCCoeff16x16);
    std::memcpy(h.ScalingListDCCoeff32x32, pp.dc32, sizeof h.ScalingListDCCoeff32x32);
    return params_.pfnDecodePicture ? params_.pfnDecodePicture(params_.pUserData, &p) : 1;
  }

  int display(const vgpu_hevc::DisplayInfo& di) override {
    CUVIDPARSERDISPINFO info{};
    info.picture_index = di.pic_idx;
    info.progressive_frame = di.progressive_frame ? 1 : 0;
    info.top_field_first = di.top_field_first ? 1 : 0;
    info.repeat_first_field = di.repeat_first_field;
    info.timestamp = di.timestamp;
    return params_.pfnDisplayPicture ? params_.pfnDisplayPicture(params_.pUserData, &info) : 1;
  }

  void sei(int pic_idx, const std::vector<vgpu_hevc::SeiMessage>& msgs) override {
    if (!params_.pfnGetSEIMsg || msgs.empty()) return;
    std::vector<uint8_t> data;
    std::vector<CUSEIMESSAGE> list;
    for (const auto& m : msgs) {
      CUSEIMESSAGE c{};
      c.sei_message_type = static_cast<unsigned char>(m.type);
      c.sei_message_size = static_cast<unsigned>(m.payload.size());
      list.push_back(c);
      data.insert(data.end(), m.payload.begin(), m.payload.end());
    }
    CUVIDSEIMESSAGEINFO info{};
    info.pSEIData = data.data();
    info.pSEIMessage = list.data();
    info.sei_message_count = static_cast<unsigned>(list.size());
    info.picIdx = static_cast<unsigned>(pic_idx);
    params_.pfnGetSEIMsg(params_.pUserData, &info);
  }

 private:
  CUVIDPARSERPARAMS params_;
  std::unique_ptr<CUVIDPICPARAMS> pic_;
};

}  // namespace
