// Part of nvcuvid_api.cpp: MPEG-2 -- cuvidDecodePicture for pictures the application describes in CUVIDMPEG2PICPARAMS, decoder
// creation as the card answers it, and the parser callbacks that translate mpeg2_parser.cpp's events to the application's.

namespace {

// The decoded frame as the decoder's NV12 surface (8-bit planes, the display window scaled to the target size when it is not the
// target's own size).
CUresult write_mpeg2_surface(Decoder& d, const vgpu_mpeg2::Frame& f, Surface& s) {
  const int fw = f.width, fh = f.height;
  int l = d.info.display_area.left, t = d.info.display_area.top, r = d.info.display_area.right, b = d.info.display_area.bottom;
  if (l == 0 && t == 0 && r == 0 && b == 0) {
    r = static_cast<int>(d.width);
    b = static_cast<int>(d.height);
  }
  l = std::max(0, std::min(l, fw));
  t = std::max(0, std::min(t, fh));
  r = std::max(l + 2, std::min(r, static_cast<int>(d.width)));
  b = std::max(t + 2, std::min(b, static_cast<int>(d.height)));
  l &= ~1;
  t &= ~1;
  const int w = r - l, h = b - t;
  const uint32_t tw = d.target_w, th = d.target_h, tcw = (tw + 1) / 2, tch = (th + 1) / 2;
  const int cw = (w + 1) / 2, chh = (h + 1) / 2;
  if (!ensure_surface(d, s)) return CUDA_ERROR_OUT_OF_MEMORY;
  const std::vector<uint8_t> yt = window_to_target_t<uint8_t>(f.y.data(), f.stride_y, fw, fh, l, t, w, h, tw, th);
  const std::vector<uint8_t> ut = window_to_target_t<uint8_t>(f.u.data(), f.stride_c, fw / 2, fh / 2, l / 2, t / 2, cw, chh, tcw, tch),
                             vt = window_to_target_t<uint8_t>(f.v.data(), f.stride_c, fw / 2, fh / 2, l / 2, t / 2, cw, chh, tcw, tch);
  std::vector<uint8_t> uv(static_cast<size_t>(tcw) * 2 * tch);
  for (size_t i = 0; i < ut.size(); ++i) {
    uv[2 * i] = ut[i];
    uv[2 * i + 1] = vt[i];
  }
  cudaMemcpy2D(s.dev, d.pitch, yt.data(), tw, tw, th, cudaMemcpyHostToDevice);
  cudaMemcpy2D(static_cast<uint8_t*>(s.dev) + static_cast<size_t>(d.pitch) * th, d.pitch, uv.data(), tcw * 2, tcw * 2, tch, cudaMemcpyHostToDevice);
  return CUDA_SUCCESS;
}

CUresult decode_mpeg2_picture(Decoder& d, CUVIDPICPARAMS* p, Surface& s) {
  const CUVIDMPEG2PICPARAMS& m = p->CodecSpecific.mpeg2;
  // measured: a picture less than three macroblocks wide (below the 48 samples the decoder takes) is refused
  if (p->PicWidthInMbs < 3) return CUDA_ERROR_INVALID_VALUE;
  vgpu_mpeg2::PicParams pp;
  // the card's PicWidthInMbs is the width divided by 16 and rounded down: the macroblocks across come from the decoder's coded width
  pp.mb_width = static_cast<int>((d.width + 15) / 16);
  pp.mb_height = p->FrameHeightInMbs;
  if (pp.mb_width <= 0 || pp.mb_height <= 0 || pp.mb_width > 255 || pp.mb_height > 255) return CUDA_ERROR_INVALID_IMAGE;
  pp.field_pic = p->field_pic_flag != 0;
  pp.bottom_field = p->bottom_field_flag != 0;
  pp.second_field = p->second_field != 0;
  pp.picture_coding_type = m.picture_coding_type;
  pp.full_pel_forward = m.full_pel_forward_vector != 0;
  pp.full_pel_backward = m.full_pel_backward_vector != 0;
  for (int a = 0; a < 2; ++a)
    for (int c = 0; c < 2; ++c) pp.f_code[a][c] = m.f_code[a][c];
  pp.intra_dc_precision = m.intra_dc_precision & 3;
  pp.frame_pred_frame_dct = m.frame_pred_frame_dct != 0;
  pp.concealment_motion_vectors = m.concealment_motion_vectors != 0;
  pp.q_scale_type = m.q_scale_type != 0;
  pp.intra_vlc_format = m.intra_vlc_format != 0;
  pp.alternate_scan = m.alternate_scan != 0;
  pp.top_field_first = m.top_field_first != 0;
  std::memcpy(pp.intra_matrix, m.QuantMatrixIntra, 64);
  std::memcpy(pp.inter_matrix, m.QuantMatrixInter, 64);
  std::unique_ptr<vgpu_mpeg2::Frame>& cur = d.mpeg2_frames[static_cast<size_t>(p->CurrPicIdx)];
  if (!cur || !pp.second_field || cur->width != pp.mb_width * 16 || cur->height != pp.mb_height * 16) {
    cur = std::make_unique<vgpu_mpeg2::Frame>();
    cur->alloc(pp.mb_width * 16, pp.mb_height * 16);
  }
  auto ref = [&](int idx) -> const vgpu_mpeg2::Frame* {
    if (idx < 0 || static_cast<size_t>(idx) >= d.mpeg2_frames.size() || idx == p->CurrPicIdx) return nullptr;
    const vgpu_mpeg2::Frame* f = d.mpeg2_frames[static_cast<size_t>(idx)].get();
    return f && !f->y.empty() ? f : nullptr;
  };
  const vgpu_mpeg2::Frame* fwd = ref(m.ForwardRefIdx);
  const vgpu_mpeg2::Frame* bwd = ref(m.BackwardRefIdx);
  // the slices: each starts at its offset with a start code
  std::vector<vgpu_mpeg2::SliceData> slices;
  const unsigned n = p->nNumSlices ? p->nNumSlices : 1;
  const unsigned zero = 0;
  const unsigned* offs = p->pSliceDataOffsets ? p->pSliceDataOffsets : &zero;
  for (unsigned i = 0; i < n; ++i) {
    size_t a = offs[i], b = (i + 1 < n && p->pSliceDataOffsets) ? offs[i + 1] : p->nBitstreamDataLen;
    if (a >= p->nBitstreamDataLen) break;
    b = std::min<size_t>(b, p->nBitstreamDataLen);
    if (b > a) slices.push_back({p->pBitstreamData + a, b - a});
  }
  std::string err;
  const bool ok = vgpu_mpeg2::decode_picture(cur.get(), fwd, bwd, pp, slices, &err);
  if (!ok) {
    say_once(("MPEG-2 picture not decoded: " + err).c_str());
    const CUresult wr = write_mpeg2_surface(d, *cur, s);
    if (wr != CUDA_SUCCESS) return wr;
    return CUDA_ERROR_INVALID_IMAGE;
  }
  return write_mpeg2_surface(d, *cur, s);
}

// ---- the parser's sink: events to the application's callbacks ------------------------------------------------

class Mpeg2CuvidSink : public vgpu_mpeg2::ParserSink {
 public:
  explicit Mpeg2CuvidSink(const CUVIDPARSERPARAMS& params) : params_(params) {}

  int sequence(const vgpu_mpeg2::SeqInfo& i) override {
    CUVIDEOFORMAT f{};
    f.codec = cudaVideoCodec_MPEG2;
    f.frame_rate.numerator = i.fps_num;
    f.frame_rate.denominator = i.fps_den;
    f.progressive_sequence = i.progressive ? 1 : 0;
    f.bit_depth_luma_minus8 = 0;
    f.bit_depth_chroma_minus8 = 0;
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

  int decode(vgpu_mpeg2::PicDesc& d) override {
    pic_ = std::make_unique<CUVIDPICPARAMS>();
    std::memset(pic_.get(), 0, sizeof *pic_);
    CUVIDPICPARAMS& p = *pic_;
    const vgpu_mpeg2::PicParams& pp = d.pp;
    p.PicWidthInMbs = d.pic_width_in_mbs;
    p.FrameHeightInMbs = pp.mb_height;
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
    CUVIDMPEG2PICPARAMS& m = p.CodecSpecific.mpeg2;
    m.ForwardRefIdx = d.fwd_idx;
    m.BackwardRefIdx = d.bwd_idx;
    m.picture_coding_type = pp.picture_coding_type;
    m.full_pel_forward_vector = pp.full_pel_forward;
    m.full_pel_backward_vector = pp.full_pel_backward;
    for (int a = 0; a < 2; ++a)
      for (int c = 0; c < 2; ++c) m.f_code[a][c] = pp.f_code[a][c];
    m.intra_dc_precision = pp.intra_dc_precision;
    m.frame_pred_frame_dct = pp.frame_pred_frame_dct;
    m.concealment_motion_vectors = pp.concealment_motion_vectors;
    m.q_scale_type = pp.q_scale_type;
    m.intra_vlc_format = pp.intra_vlc_format;
    m.alternate_scan = pp.alternate_scan;
    m.top_field_first = pp.top_field_first;
    std::memcpy(m.QuantMatrixIntra, pp.intra_matrix, 64);
    std::memcpy(m.QuantMatrixInter, pp.inter_matrix, 64);
    return params_.pfnDecodePicture ? params_.pfnDecodePicture(params_.pUserData, &p) : 1;
  }

  int display(const vgpu_mpeg2::DisplayInfo& di) override {
    CUVIDPARSERDISPINFO info{};
    info.picture_index = di.pic_idx;
    info.progressive_frame = di.progressive_frame ? 1 : 0;
    info.top_field_first = di.top_field_first ? 1 : 0;
    info.repeat_first_field = di.repeat_first_field;
    info.timestamp = di.timestamp;
    return params_.pfnDisplayPicture ? params_.pfnDisplayPicture(params_.pUserData, &info) : 1;
  }

 private:
  CUVIDPARSERPARAMS params_;
  std::unique_ptr<CUVIDPICPARAMS> pic_;
};

}  // namespace
