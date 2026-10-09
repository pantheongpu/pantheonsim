// Part of nvcuvid_api.cpp: H.264 -- cuvidDecodePicture for pictures the application describes in CUVIDH264PICPARAMS,
// and the parser callbacks that translate h264_parser.cpp's events to the application's.

namespace {

// Resamples a w x h plane window (x0, y0) of `src` to tw x th the way the card's scaler is measured to (resample()).
std::vector<uint8_t> window_to_target(const uint8_t* src, int stride, int x0, int y0, int w, int h, uint32_t tw, uint32_t th) {
  std::vector<uint8_t> win(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; ++y) std::memcpy(&win[static_cast<size_t>(y) * w], src + static_cast<ptrdiff_t>(y0 + y) * stride + x0, static_cast<size_t>(w));
  if (static_cast<uint32_t>(w) == tw && static_cast<uint32_t>(h) == th) return win;
  return resample(win, static_cast<uint32_t>(w), static_cast<uint32_t>(h), tw, th);
}

// The decoded frame as NV12 in the decoder's surface: the display area of the picture, scaled to the target size.
CUresult write_h264_surface(Decoder& d, const vgpu_h264::Frame& f, Surface& s) {
  const int fw = f.mbs_w * 16, fh = f.mbs_h * 16;
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
  const std::vector<uint8_t> yt = window_to_target(f.y.data(), f.stride_y, l, t, w, h, tw, th);
  const int cw = (w + 1) / 2, chh = (h + 1) / 2;
  const std::vector<uint8_t> ut = window_to_target(f.u.data(), f.stride_c, l / 2, t / 2, cw, chh, tcw, tch),
                             vt = window_to_target(f.v.data(), f.stride_c, l / 2, t / 2, cw, chh, tcw, tch);
  std::vector<uint8_t> uv(static_cast<size_t>(tcw) * 2 * tch);
  for (size_t i = 0; i < ut.size(); ++i) {
    uv[2 * i] = ut[i];
    uv[2 * i + 1] = vt[i];
  }
  if (!ensure_surface(d, s)) return CUDA_ERROR_OUT_OF_MEMORY;
  cudaMemcpy2D(s.dev, d.pitch, yt.data(), tw, tw, th, cudaMemcpyHostToDevice);
  cudaMemcpy2D(static_cast<uint8_t*>(s.dev) + static_cast<size_t>(d.pitch) * th, d.pitch, uv.data(), tcw * 2, tcw * 2, tch, cudaMemcpyHostToDevice);
  return CUDA_SUCCESS;
}

CUresult decode_h264_picture(Decoder& d, CUVIDPICPARAMS* p, Surface& s) {
  const CUVIDH264PICPARAMS& h = p->CodecSpecific.h264;
  using namespace vgpu_h264;
  PicParams pp;
  pp.log2_max_frame_num = h.log2_max_frame_num_minus4 + 4;
  pp.poc_type = h.pic_order_cnt_type;
  pp.log2_max_poc_lsb = h.log2_max_pic_order_cnt_lsb_minus4 + 4;
  pp.delta_pic_order_always_zero = h.delta_pic_order_always_zero_flag;
  pp.frame_mbs_only = h.frame_mbs_only_flag;
  pp.direct_8x8_inference = h.direct_8x8_inference_flag;
  pp.num_ref_frames = h.num_ref_frames;
  pp.chroma_format_idc = 1;
  pp.bit_depth_luma = 8 + h.bit_depth_luma_minus8;
  pp.bit_depth_chroma = 8 + h.bit_depth_chroma_minus8;
  pp.transform_bypass = h.qpprime_y_zero_transform_bypass_flag;
  pp.cabac = h.entropy_coding_mode_flag;
  pp.bottom_field_pic_order_in_frame_present = h.pic_order_present_flag;
  pp.redundant_pic_cnt_present = h.redundant_pic_cnt_present_flag;
  pp.num_ref_idx_default[0] = h.num_ref_idx_l0_active_minus1 + 1;
  pp.num_ref_idx_default[1] = h.num_ref_idx_l1_active_minus1 + 1;
  pp.weighted_pred = h.weighted_pred_flag;
  pp.weighted_bipred_idc = h.weighted_bipred_idc;
  pp.pic_init_qp = 26 + h.pic_init_qp_minus26;
  pp.deblocking_control_present = h.deblocking_filter_control_present_flag;
  pp.constrained_intra_pred = h.constrained_intra_pred_flag;
  pp.transform_8x8_mode = h.transform_8x8_mode_flag;
  pp.chroma_qp_offset[0] = h.chroma_qp_index_offset;
  pp.chroma_qp_offset[1] = h.second_chroma_qp_index_offset;
  pp.num_slice_groups = h.fmo_aso_enable ? h.num_slice_groups_minus1 + 1 : 1;
  pp.slice_group_map_type = h.slice_group_map_type;
  std::memcpy(pp.ws4, h.WeightScale4x4, sizeof pp.ws4);
  std::memcpy(pp.ws8, h.WeightScale8x8, sizeof pp.ws8);
  pp.mbs_w = p->PicWidthInMbs;
  pp.mbs_h = p->FrameHeightInMbs;
  pp.mbaff = h.MbaffFrameFlag;
  pp.field_pic = p->field_pic_flag;
  pp.bottom_field = p->bottom_field_flag;
  pp.second_field = p->second_field;
  pp.frame_num = h.frame_num;
  pp.poc[0] = h.CurrFieldOrderCnt[0];
  pp.poc[1] = h.CurrFieldOrderCnt[1];
  pp.ref_pic = p->ref_pic_flag;
  if (pp.mbs_w <= 0 || pp.mbs_h <= 0 || static_cast<uint32_t>(pp.mbs_w) * 16 > d.width + 16 * 255 || pp.mbs_w > 1024 || pp.mbs_h > 1024) return CUDA_ERROR_INVALID_IMAGE;
  // the current frame store (a second field decodes into the frame its first field made)
  std::unique_ptr<Frame>& cur = d.h264_frames[static_cast<size_t>(p->CurrPicIdx)];
  if (!cur) cur = std::make_unique<Frame>();
  if (!p->second_field || cur->uid == 0) {
    cur->uid = g_next_uid++;
    cur->alloc(pp.mbs_w, pp.mbs_h);
  }
  // the reference pictures
  for (int i = 0; i < 16; ++i) {
    const CUVIDH264DPBENTRY& e = h.dpb[i];
    if (e.PicIdx < 0 || static_cast<size_t>(e.PicIdx) >= d.h264_frames.size()) continue;
    if (e.used_for_reference == 0 && !e.not_existing) continue;
    Frame* rf = d.h264_frames[static_cast<size_t>(e.PicIdx)].get();
    if (!rf || rf->uid == 0 || rf->y.empty()) continue;
    DpbEntry de;
    de.frame = rf;
    de.frame_num = e.FrameIdx;
    de.long_term = e.is_long_term != 0;
    de.not_existing = e.not_existing != 0;
    de.used = e.used_for_reference & 3;
    de.poc[0] = e.FieldOrderCnt[0];
    de.poc[1] = e.FieldOrderCnt[1];
    rf->poc[0] = e.FieldOrderCnt[0];
    rf->poc[1] = e.FieldOrderCnt[1];
    pp.dpb.push_back(de);
  }
  // the slice NAL units: each slice starts at its offset with a start code
  std::vector<SliceData> slices;
  const unsigned n = p->nNumSlices ? p->nNumSlices : 1;
  const unsigned zero = 0;
  const unsigned* offs = p->pSliceDataOffsets ? p->pSliceDataOffsets : &zero;
  for (unsigned i = 0; i < n; ++i) {
    size_t a = offs[i], b = (i + 1 < n && p->pSliceDataOffsets) ? offs[i + 1] : p->nBitstreamDataLen;
    if (a >= p->nBitstreamDataLen) break;
    b = std::min<size_t>(b, p->nBitstreamDataLen);
    const uint8_t* base = p->pBitstreamData;
    // skip the start code (00 00 01 or 00 00 00 01) when there is one
    size_t k = a;
    while (k + 3 <= b && !(base[k] == 0 && base[k + 1] == 0 && base[k + 2] == 1) && k - a < 4) ++k;
    if (k + 3 <= b && base[k] == 0 && base[k + 1] == 0 && base[k + 2] == 1) a = k + 3;
    while (b > a && base[b - 1] == 0) --b;
    if (b > a) slices.push_back({base + a, b - a});
  }
  std::string err;
  const bool ok = decode_picture(cur.get(), pp, slices, &err);
  if (!ok) {
    say_once(("H.264 picture not decoded: " + err).c_str());
    // pictures decoded with errors keep what was decoded (the card conceals the same way)
    const CUresult wr = write_h264_surface(d, *cur, s);
    if (wr != CUDA_SUCCESS) return wr;
    return CUDA_ERROR_INVALID_IMAGE;
  }
  return write_h264_surface(d, *cur, s);
}

// ---- the parser's sink: events to the application's callbacks ------------------------------------------------

class CuvidSink : public vgpu_h264::ParserSink {
 public:
  explicit CuvidSink(const CUVIDPARSERPARAMS& params) : params_(params) {}

  int sequence(const vgpu_h264::SeqInfo& i) override {
    CUVIDEOFORMAT f{};
    f.codec = cudaVideoCodec_H264;
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

  int decode(vgpu_h264::PicDesc& d) override {
    pic_ = std::make_unique<CUVIDPICPARAMS>();
    std::memset(pic_.get(), 0, sizeof *pic_);
    CUVIDPICPARAMS& p = *pic_;
    const vgpu_h264::PicParams& pp = d.pp;
    p.PicWidthInMbs = pp.mbs_w;
    p.FrameHeightInMbs = pp.mbs_h;
    p.CurrPicIdx = d.curr_pic_idx;
    p.field_pic_flag = d.field_pic ? 1 : 0;
    p.bottom_field_flag = d.bottom_field ? 1 : 0;
    p.second_field = d.second_field ? 1 : 0;
    p.nBitstreamDataLen = static_cast<unsigned>(d.data.size());
    p.pBitstreamData = d.data.data();
    p.nNumSlices = static_cast<unsigned>(d.slice_offsets.size());
    p.pSliceDataOffsets = d.slice_offsets.data();
    p.ref_pic_flag = d.ref_pic ? 1 : 0;
    p.intra_pic_flag = d.intra_pic ? 1 : 0;
    CUVIDH264PICPARAMS& h = p.CodecSpecific.h264;
    const vgpu_h264::Sps& s = *d.sps;
    h.log2_max_frame_num_minus4 = pp.log2_max_frame_num - 4;
    h.pic_order_cnt_type = pp.poc_type;
    h.log2_max_pic_order_cnt_lsb_minus4 = pp.log2_max_poc_lsb - 4;
    h.delta_pic_order_always_zero_flag = pp.delta_pic_order_always_zero;
    h.frame_mbs_only_flag = pp.frame_mbs_only;
    h.direct_8x8_inference_flag = pp.direct_8x8_inference;
    h.num_ref_frames = pp.num_ref_frames;
    h.residual_colour_transform_flag = 0;
    h.bit_depth_luma_minus8 = static_cast<unsigned char>(pp.bit_depth_luma - 8);
    h.bit_depth_chroma_minus8 = static_cast<unsigned char>(pp.bit_depth_chroma - 8);
    h.qpprime_y_zero_transform_bypass_flag = static_cast<unsigned char>(pp.transform_bypass);
    h.entropy_coding_mode_flag = pp.cabac;
    h.pic_order_present_flag = pp.bottom_field_pic_order_in_frame_present;
    h.num_ref_idx_l0_active_minus1 = pp.num_ref_idx_default[0] - 1;
    h.num_ref_idx_l1_active_minus1 = pp.num_ref_idx_default[1] - 1;
    h.weighted_pred_flag = pp.weighted_pred;
    h.weighted_bipred_idc = pp.weighted_bipred_idc;
    h.pic_init_qp_minus26 = pp.pic_init_qp - 26;
    h.deblocking_filter_control_present_flag = pp.deblocking_control_present;
    h.redundant_pic_cnt_present_flag = pp.redundant_pic_cnt_present;
    h.transform_8x8_mode_flag = pp.transform_8x8_mode;
    h.MbaffFrameFlag = pp.mbaff;
    h.constrained_intra_pred_flag = pp.constrained_intra_pred;
    h.chroma_qp_index_offset = pp.chroma_qp_offset[0];
    h.second_chroma_qp_index_offset = pp.chroma_qp_offset[1];
    h.ref_pic_flag = d.ref_pic ? 1 : 0;
    h.frame_num = pp.frame_num;
    h.CurrFieldOrderCnt[0] = pp.poc[0];
    h.CurrFieldOrderCnt[1] = pp.poc[1];
    for (int i = 0; i < 16; ++i) {
      const vgpu_h264::DpbSlot& sl = d.slots[i];
      h.dpb[i].PicIdx = sl.pic_idx;
      h.dpb[i].FrameIdx = sl.frame_idx;
      h.dpb[i].is_long_term = sl.is_long_term;
      h.dpb[i].not_existing = sl.not_existing;
      h.dpb[i].used_for_reference = sl.used;
      h.dpb[i].FieldOrderCnt[0] = sl.poc[0];
      h.dpb[i].FieldOrderCnt[1] = sl.poc[1];
    }
    std::memcpy(h.WeightScale4x4, pp.ws4, sizeof h.WeightScale4x4);
    std::memcpy(h.WeightScale8x8, pp.ws8, sizeof h.WeightScale8x8);
    h.fmo_aso_enable = pp.num_slice_groups > 1 ? 1 : 0;
    h.num_slice_groups_minus1 = static_cast<unsigned char>(pp.num_slice_groups - 1);
    h.slice_group_map_type = static_cast<unsigned char>(pp.slice_group_map_type);
    h.pic_init_qs_minus26 = 0;
    (void)s;
    return params_.pfnDecodePicture ? params_.pfnDecodePicture(params_.pUserData, &p) : 1;
  }

  int display(const vgpu_h264::DisplayInfo& di) override {
    CUVIDPARSERDISPINFO info{};
    info.picture_index = di.pic_idx;
    info.progressive_frame = di.progressive_frame ? 1 : 0;
    info.top_field_first = di.top_field_first ? 1 : 0;
    info.repeat_first_field = di.repeat_first_field;
    info.timestamp = di.timestamp;
    return params_.pfnDisplayPicture ? params_.pfnDisplayPicture(params_.pUserData, &info) : 1;
  }

  void sei(int pic_idx, const std::vector<vgpu_h264::SeiMessage>& msgs) override {
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
