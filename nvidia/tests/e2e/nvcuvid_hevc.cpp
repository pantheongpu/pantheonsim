// NVDEC's HEVC decoder and parser as an application drives them, against what NVIDIA's
// libnvcuvid printed on an RTX 3060 (driver 595): decoder capabilities, decoder creation
// for HEVC, and, for each stream in nvidia/tests/data/hevc, every callback of the video
// parser (the sequence format, the picture parameters handed to cuvidDecodePicture, the
// display order and timestamps) and the CRC-32 of every displayed surface. HEVC output is
// bit-exact by definition, so the CRCs of a conformant decoder are the card's.
//
// One line per fact. nvcuvid_hevc.rtx3060.txt is what the card printed;
// run_nvcuvid_hevc.sh compares this program's output with it.
//
//   nvcuvid_hevc [--dump DIR] [--only NAME] [--file PATH] [--skip NAME,NAME] DATA_DIR
#include <dlfcn.h>
#include <cuda.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "cuviddec.h"
#include "nvcuvid.h"

#undef cuvidMapVideoFrame
#undef cuvidUnmapVideoFrame

template <class E>
static void poke(E& field, uint32_t v) {
  static_assert(sizeof(E) == sizeof v, "a 32-bit enum");
  std::memcpy(&field, &v, sizeof v);
}

static std::vector<uint8_t> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static uint32_t crc32(const uint8_t* d, size_t n, uint32_t crc = 0) {
  static uint32_t table[256];
  if (!table[1]) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
  }
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ d[i]) & 255] ^ (crc >> 8);
  return ~crc;
}

static std::string dump_dir, data_dir;
static bool g_force_nv12 = false;   // 10-bit streams into a decoder that outputs NV12
static int failures = 0;

struct Run {
  CUvideodecoder dec = nullptr;
  bool crop_in_decoder = false;   // target size = the display rectangle
  CUVIDEOFORMAT fmt{};
  unsigned w = 0, h = 0;          // the surface's visible size
  int frames = 0;
  uint32_t all_crc = 0;
  std::string name;
  int create_result = 0;
  int decode_fail = 0;
  unsigned bytes = 1;             // bytes per sample of the surface (P016: 2)
};
static Run* run = nullptr;

static int CUDAAPI seq_cb(void*, CUVIDEOFORMAT* f) {
  run->fmt = *f;
  const unsigned char* sig = reinterpret_cast<const unsigned char*>(&f->video_signal_description);
  std::printf("  sequence: codec %d rate %u/%u progressive %d depth %d/%d chroma %d surfaces %d coded %ux%u display %d,%d,%d,%d bitrate %u aspect %d:%d "
              "signal %02x %02x %02x %02x seqhdr %u\n",
              f->codec, f->frame_rate.numerator, f->frame_rate.denominator, f->progressive_sequence, f->bit_depth_luma_minus8,
              f->bit_depth_chroma_minus8, f->chroma_format, f->min_num_decode_surfaces, f->coded_width, f->coded_height, f->display_area.left,
              f->display_area.top, f->display_area.right, f->display_area.bottom, f->bitrate, f->display_aspect_ratio.x, f->display_aspect_ratio.y,
              sig[0], sig[1], sig[2], sig[3], f->seqhdr_data_length);
  const unsigned dw = f->display_area.right - f->display_area.left, dh = f->display_area.bottom - f->display_area.top;
  if (run->dec) {
    cuvidDestroyDecoder(run->dec);
    run->dec = nullptr;
  }
  CUVIDDECODECREATEINFO ci{};
  ci.ulWidth = f->coded_width;
  ci.ulHeight = f->coded_height;
  ci.ulNumDecodeSurfaces = f->min_num_decode_surfaces;
  ci.CodecType = f->codec;
  ci.ChromaFormat = f->chroma_format;
  ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  ci.bitDepthMinus8 = f->bit_depth_luma_minus8;
  ci.OutputFormat = f->bit_depth_luma_minus8 && !g_force_nv12 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
  ci.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
  ci.ulMaxWidth = f->coded_width;
  ci.ulMaxHeight = f->coded_height;
  ci.ulNumOutputSurfaces = 2;
  // plain mode: the whole coded picture is the display area (a smaller one would be scaled to the target size); the frame checksums
  // are over the sequence's display rectangle of the surface
  ci.display_area = {0, 0, static_cast<short>(f->coded_width), static_cast<short>(f->coded_height)};
  if (run->crop_in_decoder) {
    ci.display_area = {static_cast<short>(f->display_area.left), static_cast<short>(f->display_area.top), static_cast<short>(f->display_area.right),
                       static_cast<short>(f->display_area.bottom)};
    ci.ulTargetWidth = dw;
    ci.ulTargetHeight = dh;
  } else {
    ci.ulTargetWidth = f->coded_width;
    ci.ulTargetHeight = f->coded_height;
  }
  run->bytes = (f->bit_depth_luma_minus8 && !g_force_nv12) ? 2 : 1;
  run->w = dw;
  run->h = dh;
  const CUresult r = cuvidCreateDecoder(&run->dec, &ci);
  std::printf("  cuvidCreateDecoder: %d\n", r);
  run->create_result = r;
  if (r) return 0;
  return f->min_num_decode_surfaces;
}

static void print_pic(const CUVIDPICPARAMS* p) {
  const CUVIDHEVCPICPARAMS& h = p->CodecSpecific.hevc;
  std::printf("  decode: mbs %dx%d index %d field %d bottom %d second %d intra %d ref %d length %u slices %u offsets", p->PicWidthInMbs,
              p->FrameHeightInMbs, p->CurrPicIdx, p->field_pic_flag, p->bottom_field_flag, p->second_field, p->intra_pic_flag, p->ref_pic_flag,
              p->nBitstreamDataLen, p->nNumSlices);
  for (unsigned i = 0; i < p->nNumSlices && i < 8; ++i) std::printf(" %u", p->pSliceDataOffsets[i]);
  std::printf(" head");
  for (unsigned i = 0; i < 6 && i < p->nBitstreamDataLen; ++i) std::printf(" %02x", p->pBitstreamData[i]);
  std::printf("\n");
  std::printf("    sps: size %dx%d cb %d+%d tb %d+%d pcm %d %d+%d depth %d/%d lf %d  rdpcm %d ext %d isd %d rice %d align %d ppsrext %d ccp %d cqo %d %d %d\n",
              h.pic_width_in_luma_samples, h.pic_height_in_luma_samples, h.log2_min_luma_coding_block_size_minus3, h.log2_diff_max_min_luma_coding_block_size,
              h.log2_min_transform_block_size_minus2, h.log2_diff_max_min_transform_block_size, h.pcm_enabled_flag, h.log2_min_pcm_luma_coding_block_size_minus3,
              h.log2_diff_max_min_pcm_luma_coding_block_size, h.pcm_sample_bit_depth_luma_minus1, h.pcm_sample_bit_depth_chroma_minus1,
              h.pcm_loop_filter_disabled_flag, h.explicit_rdpcm_enabled_flag, h.extended_precision_processing_flag, h.intra_smoothing_disabled_flag,
              h.persistent_rice_adaptation_enabled_flag, h.cabac_bypass_alignment_enabled_flag, h.pps_range_extension_flag,
              h.cross_component_prediction_enabled_flag, h.chroma_qp_offset_list_enabled_flag, h.diff_cu_chroma_qp_offset_depth, h.chroma_qp_offset_list_len_minus1);
  std::printf("    sps2: sis %d tdi %d tdp %d amp %d sep %d poclsb %d nst %d ltp %d nltsps %d tmvp %d sao %d sl %d irap %d idr %d depth %d/%d tsmax %d saoscale %d %d hp %d\n",
              h.strong_intra_smoothing_enabled_flag, h.max_transform_hierarchy_depth_intra, h.max_transform_hierarchy_depth_inter, h.amp_enabled_flag,
              h.separate_colour_plane_flag, h.log2_max_pic_order_cnt_lsb_minus4, h.num_short_term_ref_pic_sets, h.long_term_ref_pics_present_flag,
              h.num_long_term_ref_pics_sps, h.sps_temporal_mvp_enabled_flag, h.sample_adaptive_offset_enabled_flag, h.scaling_list_enable_flag,
              h.IrapPicFlag, h.IdrPicFlag, h.bit_depth_luma_minus8, h.bit_depth_chroma_minus8, h.log2_max_transform_skip_block_size_minus2,
              h.log2_sao_offset_scale_luma, h.log2_sao_offset_scale_chroma, h.high_precision_offsets_enabled_flag);
  std::printf("    pps: dep %d shext %d sdh %d cuqpd %d %d qp %d cb %d cr %d cip %d wp %d wbp %d tskip %d tqb %d wpp %d pml %d extra %d\n",
              h.dependent_slice_segments_enabled_flag, h.slice_segment_header_extension_present_flag, h.sign_data_hiding_enabled_flag, h.cu_qp_delta_enabled_flag,
              h.diff_cu_qp_delta_depth, h.init_qp_minus26, h.pps_cb_qp_offset, h.pps_cr_qp_offset, h.constrained_intra_pred_flag, h.weighted_pred_flag,
              h.weighted_bipred_flag, h.transform_skip_enabled_flag, h.transquant_bypass_enabled_flag, h.entropy_coding_sync_enabled_flag,
              h.log2_parallel_merge_level_minus2, h.num_extra_slice_header_bits);
  std::printf("    pps2: lfat %d lfas %d ofp %d l0 %d l1 %d lmp %d cip %d sco %d dfo %d dfd %d beta %d tc %d tiles %d uniform %d cols %d rows %d\n",
              h.loop_filter_across_tiles_enabled_flag, h.loop_filter_across_slices_enabled_flag, h.output_flag_present_flag,
              h.num_ref_idx_l0_default_active_minus1, h.num_ref_idx_l1_default_active_minus1, h.lists_modification_present_flag, h.cabac_init_present_flag,
              h.pps_slice_chroma_qp_offsets_present_flag, h.deblocking_filter_override_enabled_flag, h.pps_deblocking_filter_disabled_flag,
              h.pps_beta_offset_div2, h.pps_tc_offset_div2, h.tiles_enabled_flag, h.uniform_spacing_flag, h.num_tile_columns_minus1, h.num_tile_rows_minus1);
  if (h.tiles_enabled_flag) {
    std::printf("    tiles: widths");
    for (int i = 0; i < 21; ++i) std::printf(" %u", h.column_width_minus1[i]);
    std::printf(" heights");
    for (int i = 0; i < 21; ++i) std::printf(" %u", h.row_height_minus1[i]);
    std::printf("\n");
  }
  std::printf("    ext: sre %d tsr %d tsc %d irdpcm %d\n", h.sps_range_extension_flag, h.transform_skip_rotation_enabled_flag, h.transform_skip_context_enabled_flag,
              h.implicit_rdpcm_enabled_flag);
  std::printf("    rps: bits %d deltapocs %d total %d before %d after %d lt %d poc %d\n", h.NumBitsForShortTermRPSInSlice, h.NumDeltaPocsOfRefRpsIdx,
              h.NumPocTotalCurr, h.NumPocStCurrBefore, h.NumPocStCurrAfter, h.NumPocLtCurr, h.CurrPicOrderCntVal);
  std::printf("    refs:");
  int empty = 0;
  for (int i = 0; i < 16; ++i) {
    if (h.RefPicIdx[i] == -1 && h.PicOrderCntVal[i] == 0 && h.IsLongTerm[i] == 0) {
      ++empty;
      continue;
    }
    std::printf(" [%d] pic %d poc %d lt %d;", i, h.RefPicIdx[i], h.PicOrderCntVal[i], h.IsLongTerm[i]);
  }
  std::printf(" (%d empty)\n", empty);
  std::printf("    sets: before");
  for (int i = 0; i < 8; ++i) std::printf(" %u", h.RefPicSetStCurrBefore[i]);
  std::printf(" after");
  for (int i = 0; i < 8; ++i) std::printf(" %u", h.RefPicSetStCurrAfter[i]);
  std::printf(" lt");
  for (int i = 0; i < 8; ++i) std::printf(" %u", h.RefPicSetLtCurr[i]);
  std::printf(" il0");
  for (int i = 0; i < 8; ++i) std::printf(" %u", h.RefPicSetInterLayer0[i]);
  std::printf(" il1");
  for (int i = 0; i < 8; ++i) std::printf(" %u", h.RefPicSetInterLayer1[i]);
  std::printf("\n");
  unsigned nonflat = 0;
  uint32_t c = 0;
  c = crc32(&h.ScalingList4x4[0][0], 96, c);
  c = crc32(&h.ScalingList8x8[0][0], 384, c);
  c = crc32(&h.ScalingList16x16[0][0], 384, c);
  c = crc32(&h.ScalingList32x32[0][0], 128, c);
  c = crc32(h.ScalingListDCCoeff16x16, 6, c);
  c = crc32(h.ScalingListDCCoeff32x32, 2, c);
  for (int i = 0; i < 6; ++i)
    for (int k = 0; k < 16; ++k) nonflat += h.ScalingList4x4[i][k] != 16;
  std::printf("    scaling: non-flat4x4 %u crc %08x\n", nonflat, c);
}

static int CUDAAPI dec_cb(void*, CUVIDPICPARAMS* p) {
  print_pic(p);
  const CUresult r = cuvidDecodePicture(run->dec, p);
  std::printf("  cuvidDecodePicture: %d\n", r);
  if (r) ++run->decode_fail;
  return 1;
}

static int CUDAAPI disp_cb(void*, CUVIDPARSERDISPINFO* d) {
  CUVIDGETDECODESTATUS ds{};
  const CUresult sr = cuvidGetDecodeStatus(run->dec, d->picture_index, &ds);
  std::printf("  display: index %d progressive %d top-field-first %d repeat %d timestamp %lld; decode status call %d, %s\n", d->picture_index,
              d->progressive_frame, d->top_field_first, d->repeat_first_field, static_cast<long long>(d->timestamp), sr,
              ds.decodeStatus == cuvidDecodeStatus_Success || ds.decodeStatus == cuvidDecodeStatus_InProgress
                  ? "decoded"
                  : (ds.decodeStatus == cuvidDecodeStatus_Invalid ? "invalid" : "failed"));
  unsigned long long dptr = 0;
  unsigned pitch = 0;
  CUVIDPROCPARAMS pp{};
  pp.progressive_frame = d->progressive_frame;
  pp.top_field_first = d->top_field_first;
  const CUresult r = cuvidMapVideoFrame64(run->dec, d->picture_index, &dptr, &pitch, &pp);
  std::printf("  cuvidMapVideoFrame64: %d, pitch %u\n", r, pitch);
  if (r == CUDA_SUCCESS) {
    const CUVIDEOFORMAT& f = run->fmt;
    const unsigned w = run->w, h = run->h, B = run->bytes;
    const unsigned sh = run->crop_in_decoder ? h : f.coded_height;
    const unsigned x0 = run->crop_in_decoder ? 0 : f.display_area.left, y0 = run->crop_in_decoder ? 0 : f.display_area.top;
    std::vector<uint8_t> all(static_cast<size_t>(pitch) * (sh + (sh + 1) / 2));
    cuMemcpyDtoH(all.data(), static_cast<CUdeviceptr>(dptr), all.size());
    std::vector<uint8_t> out;
    for (unsigned y = 0; y < h; ++y) out.insert(out.end(), &all[static_cast<size_t>(y0 + y) * pitch + x0 * B], &all[static_cast<size_t>(y0 + y) * pitch + x0 * B] + w * B);
    for (unsigned y = 0; y < (h + 1) / 2; ++y) {
      const uint8_t* row = &all[static_cast<size_t>(sh + y0 / 2 + y) * pitch + (x0 & ~1u) * B];
      out.insert(out.end(), row, row + ((w + 1) & ~1u) * B);
    }
    const uint32_t c = crc32(out.data(), out.size());
    std::printf("  frame %d: %ux%u crc %08x luma %08x\n", run->frames, w, h, c, crc32(out.data(), static_cast<size_t>(w) * h * B));
    run->all_crc = crc32(reinterpret_cast<const uint8_t*>(&c), 4, run->all_crc);
    if (!dump_dir.empty()) {
      std::ofstream o(dump_dir + "/" + run->name + ".nv12", std::ios::binary | std::ios::app);
      o.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    }
    cuvidUnmapVideoFrame64(run->dec, dptr);
  }
  ++run->frames;
  return 1;
}

// The byte ranges of an Annex B stream's access units: an access unit ends before the next
// first slice segment of a picture, and parameter sets, SEI and delimiters before a picture belong to it.
static std::vector<std::pair<size_t, size_t>> access_units(const std::vector<uint8_t>& s, std::vector<std::vector<int>>* types = nullptr) {
  std::vector<size_t> starts;
  for (size_t i = 0; i + 3 < s.size(); ++i)
    if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1) starts.push_back(i > 0 && s[i - 1] == 0 ? i - 1 : i);
  std::vector<std::pair<size_t, size_t>> aus;
  size_t au_start = 0;
  bool have_vcl = false;
  for (size_t k = 0; k < starts.size(); ++k) {
    const size_t at = starts[k];
    size_t nal = at;
    while (s[nal] == 0) ++nal;
    ++nal;
    const int type = (s[nal] >> 1) & 63;
    if (type < 32) {
      const bool first = (s[nal + 2] & 0x80) != 0;
      if (first && have_vcl) {
        aus.push_back({au_start, at});
        au_start = at;
        have_vcl = false;
      }
      have_vcl = true;
    } else if (have_vcl && ((type >= 32 && type <= 35) || type == 39 || (type >= 41 && type <= 44) || (type >= 48 && type <= 55))) {
      aus.push_back({au_start, at});
      au_start = at;
      have_vcl = false;
    }
  }
  aus.push_back({au_start, s.size()});
  (void)types;
  return aus;
}

struct Mode {
  const char* label;
  bool per_au = true;     // one packet per access unit, with a timestamp; else the whole file in one packet
  unsigned delay = 0;
  bool crop = false;
  unsigned clock = 1000;
  unsigned parser_surfaces = 1;
  int ts_mode = 0;        // 0 regular (33 per picture), 1 irregular, 2 none, 3 the first packet only
  long long first_ts = 0;
  size_t chunk = 0;       // feed the stream in packets of this many bytes
  size_t prefix = 0;      // feed only the first bytes of the stream, in one packet, and do not end it
  bool end_of_picture = false;   // flag every packet CUVID_PKT_ENDOFPICTURE
};

static void play(const std::string& name, const std::vector<uint8_t>& data, const Mode& m) {
  std::printf("stream %s [%s]\n", name.c_str(), m.label);
  Run r;
  r.name = std::string(name) + (m.label[0] == 'p' ? "" : std::string("_") + m.label);
  r.crop_in_decoder = m.crop;
  run = &r;
  CUvideoparser parser = nullptr;
  CUVIDPARSERPARAMS pp{};
  pp.CodecType = cudaVideoCodec_HEVC;
  pp.ulMaxNumDecodeSurfaces = m.parser_surfaces;
  pp.ulClockRate = m.clock;
  pp.ulMaxDisplayDelay = m.delay;
  pp.pfnSequenceCallback = seq_cb;
  pp.pfnDecodePicture = dec_cb;
  pp.pfnDisplayPicture = disp_cb;
  const CUresult cr = cuvidCreateVideoParser(&parser, &pp);
  std::printf("  cuvidCreateVideoParser: %d\n", cr);
  if (cr) {
    ++failures;
    run = nullptr;
    return;
  }
  if (m.prefix) {
    CUVIDSOURCEDATAPACKET pk{};
    pk.payload = data.data();
    pk.payload_size = std::min(m.prefix, data.size());
    std::printf("  prefix of %zu bytes: %d\n", pk.payload_size, cuvidParseVideoData(parser, &pk));
    std::printf("  callbacks so far: %d frames displayed\n", r.frames);
    cuvidDestroyVideoParser(parser);
    if (r.dec) cuvidDestroyDecoder(r.dec);
    run = nullptr;
    return;
  }
  if (m.chunk) {
    size_t at = 0;
    int n = 0;
    while (at < data.size()) {
      const size_t len = std::min(m.chunk, data.size() - at);
      CUVIDSOURCEDATAPACKET pk{};
      pk.payload = data.data() + at;
      pk.payload_size = len;
      std::printf("  packet %d: %zu bytes (from %zu)\n", n++, len, at);
      const CUresult rc = cuvidParseVideoData(parser, &pk);
      if (rc) std::printf("  cuvidParseVideoData: %d\n", rc);
      at += len;
    }
  } else if (m.per_au) {
    const auto aus = access_units(data);
    long long ts = m.first_ts;
    long long n = 0;
    for (const auto& au : aus) {
      CUVIDSOURCEDATAPACKET pk{};
      pk.payload = data.data() + au.first;
      pk.payload_size = au.second - au.first;
      if (m.ts_mode == 0 || (m.ts_mode == 3 && n == 0) || m.ts_mode == 1) pk.flags = CUVID_PKT_TIMESTAMP;
      if (m.end_of_picture) pk.flags |= CUVID_PKT_ENDOFPICTURE;
      pk.timestamp = m.ts_mode == 1 ? m.first_ts + 100 + 7 * n * n : ts;
      ts += 33;
      ++n;
      std::printf("  packet %lld: %zu bytes, NAL types", n - 1, au.second - au.first);
      for (size_t i = au.first; i + 4 < au.second; ++i)
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) std::printf(" %d", (data[i + 3] >> 1) & 63);
      std::printf("\n");
      const CUresult rc = cuvidParseVideoData(parser, &pk);
      if (rc) std::printf("  cuvidParseVideoData: %d\n", rc);
    }
  } else {
    CUVIDSOURCEDATAPACKET pk{};
    pk.payload = data.data();
    pk.payload_size = data.size();
    pk.flags = CUVID_PKT_TIMESTAMP;
    pk.timestamp = m.first_ts;
    std::printf("  cuvidParseVideoData: %d\n", cuvidParseVideoData(parser, &pk));
  }
  std::printf("  end of stream call\n");
  CUVIDSOURCEDATAPACKET eos{};
  eos.flags = CUVID_PKT_ENDOFSTREAM;
  const CUresult er = cuvidParseVideoData(parser, &eos);
  std::printf("  end of stream: %d, %d frames displayed, %d decode failures\n", er, r.frames, r.decode_fail);
  std::printf("  summary %s: frames %d crc %08x\n", name.c_str(), r.frames, r.all_crc);
  cuvidDestroyVideoParser(parser);
  if (r.dec) cuvidDestroyDecoder(r.dec);
  run = nullptr;
}


// ---- capabilities and decoder creation: the answers of the card for HEVC -----------------------------------------------
static void caps_and_creation() {
  static const char* const codec_names[] = {"MPEG1", "MPEG2", "MPEG4", "VC1", "H264", "JPEG", "H264_SVC", "H264_MVC", "HEVC", "VP8", "VP9", "AV1"};
  for (int chroma = 0; chroma <= 3; ++chroma)
    for (int depth : {0, 2, 4, 6}) {
      CUVIDDECODECAPS c{};
      poke(c.eCodecType, cudaVideoCodec_HEVC);
      c.eChromaFormat = static_cast<cudaVideoChromaFormat>(chroma);
      c.nBitDepthMinus8 = depth;
      const CUresult r = cuvidGetDecoderCaps(&c);
      std::printf("caps %s chroma %d depth %d: %d supported %u max %ux%u mb %u min %ux%u outputs 0x%x histogram %u/%u/%u\n", codec_names[cudaVideoCodec_HEVC], chroma,
                  8 + depth, r, c.bIsSupported, c.nMaxWidth, c.nMaxHeight, c.nMaxMBCount, c.nMinWidth, c.nMinHeight, c.nOutputFormatMask, c.bIsHistogramSupported,
                  c.nCounterBitDepth, c.nMaxHistogramBins);
    }
  const auto base_info = [] {
    CUVIDDECODECREATEINFO ci{};
    ci.ulWidth = 192;
    ci.ulHeight = 144;
    ci.ulNumDecodeSurfaces = 4;
    ci.CodecType = cudaVideoCodec_HEVC;
    ci.ChromaFormat = cudaVideoChromaFormat_420;
    ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    ci.OutputFormat = cudaVideoSurfaceFormat_NV12;
    ci.ulTargetWidth = 192;
    ci.ulTargetHeight = 144;
    ci.ulNumOutputSurfaces = 2;
    ci.ulMaxWidth = 192;
    ci.ulMaxHeight = 144;
    ci.display_area = {0, 0, 192, 144};
    return ci;
  };
  const auto create = [&](const char* what, auto mutate) {
    CUVIDDECODECREATEINFO ci = base_info();
    mutate(ci);
    CUvideodecoder d = nullptr;
    const CUresult r = cuvidCreateDecoder(&d, &ci);
    std::printf("CreateDecoder(%s): %d\n", what, r);
    if (!r) cuvidDestroyDecoder(d);
  };
  create("HEVC valid", [](CUVIDDECODECREATEINFO&) {});
  for (const auto& wh : {std::pair<int, int>{16, 16}, {32, 32}, {64, 64}, {128, 128}, {144, 144}, {143, 144}, {144, 143}, {160, 144}, {144, 160}, {8192, 144}, {144, 8192}, {8192, 8192}, {8208, 144},
                         {4096, 4096}, {146, 146}, {2, 2}, {200, 144}}) {
    char name[64];
    std::snprintf(name, sizeof name, "HEVC size %dx%d", wh.first, wh.second);
    create(name, [&](CUVIDDECODECREATEINFO& c) {
      c.ulWidth = c.ulMaxWidth = c.ulTargetWidth = wh.first;
      c.ulHeight = c.ulMaxHeight = c.ulTargetHeight = wh.second;
      c.display_area = {0, 0, static_cast<short>(wh.first), static_cast<short>(wh.second)};
    });
  }
  create("HEVC 0 output surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumOutputSurfaces = 0; });
  create("HEVC 64 output surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumOutputSurfaces = 64; });
  create("HEVC target 0", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = c.ulTargetHeight = 0; });
  create("HEVC target 96x64", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = 96; c.ulTargetHeight = 64; });
  create("HEVC max size 0", [](CUVIDDECODECREATEINFO& c) { c.ulMaxWidth = c.ulMaxHeight = 0; });
  create("HEVC max size below size", [](CUVIDDECODECREATEINFO& c) { c.ulMaxWidth = c.ulMaxHeight = 64; });
  create("HEVC display area outside", [](CUVIDDECODECREATEINFO& c) { c.display_area = {0, 0, 500, 500}; });
  create("HEVC display area empty", [](CUVIDDECODECREATEINFO& c) { c.display_area = {0, 0, 0, 0}; });
  create("HEVC monochrome", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_Monochrome; });
  create("HEVC 422", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_422; });
  create("HEVC 444", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_444; });
  create("HEVC 444 YUV444 output", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_444; c.OutputFormat = cudaVideoSurfaceFormat_YUV444; });
  create("HEVC 10 bit NV12", [](CUVIDDECODECREATEINFO& c) { c.bitDepthMinus8 = 2; });
  create("HEVC 10 bit P016", [](CUVIDDECODECREATEINFO& c) { c.bitDepthMinus8 = 2; c.OutputFormat = cudaVideoSurfaceFormat_P016; });
  create("HEVC 12 bit P016", [](CUVIDDECODECREATEINFO& c) { c.bitDepthMinus8 = 4; c.OutputFormat = cudaVideoSurfaceFormat_P016; });
  create("HEVC 14 bit P016", [](CUVIDDECODECREATEINFO& c) { c.bitDepthMinus8 = 6; c.OutputFormat = cudaVideoSurfaceFormat_P016; });
  create("HEVC 8 bit P016", [](CUVIDDECODECREATEINFO& c) { c.OutputFormat = cudaVideoSurfaceFormat_P016; });
  create("HEVC 8 bit YUV444 output", [](CUVIDDECODECREATEINFO& c) { c.OutputFormat = cudaVideoSurfaceFormat_YUV444; });
  create("HEVC 0 surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 0; });
  create("HEVC 1 surface", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 1; });
  create("HEVC 32 surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 32; });
  create("HEVC 33 surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 33; });
  create("HEVC intra only", [](CUVIDDECODECREATEINFO& c) { c.ulIntraDecodeOnly = 1; });
  create("HEVC deinterlace bob", [](CUVIDDECODECREATEINFO& c) { c.DeinterlaceMode = cudaVideoDeinterlaceMode_Bob; });
  create("HEVC PreferCUDA", [](CUVIDDECODECREATEINFO& c) { c.ulCreationFlags = cudaVideoCreate_PreferCUDA; });
  create("HEVC odd target", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = 97; c.ulTargetHeight = 65; });
  {
    CUVIDPARSERPARAMS pp{};
    pp.CodecType = cudaVideoCodec_HEVC;
    pp.ulMaxNumDecodeSurfaces = 1;
    CUvideoparser p = nullptr;
    const CUresult r = cuvidCreateVideoParser(&p, &pp);
    std::printf("CreateVideoParser(HEVC): %d\n", r);
    if (!r) cuvidDestroyVideoParser(p);
  }
}

static void run_hevc_suite(const std::string& base, const std::string& only, const std::string& skip) {
  static const char* const names[] = {
      "i_only",      "p_low",      "b_flat",    "b_pyramid",  "sao_off",   "deblock_off", "deblock_offs", "amp_rect",   "ctu16",   "ctu32",      "tskip",
      "lossless",    "cu_lossless", "no_signhide", "scaling_def", "wpp",     "slices",      "weightp",      "weightb",    "no_tmvp", "main10",     "main10_i",
      "main10_wide", "cip",        "no_strong", "open_gop",   "long_gop",  "lowqp",       "highqp",       "aud_hrd",    "max_merge2", "cutree_idr", "rd_deep",
      "tu_deep",     "b_sei",      "b_aud",     "b_ps_mid",   "cra_first", "bla",         "eos_mid",      "res_change", "crop",       "crop_odd",    "size_136"};
  const Mode plain{"packets"};
  auto load = [&](const char* n) { return slurp(base + "/hevc/" + n + ".h265"); };
  const auto stream = [&](const char* n, const Mode& m) {
    const std::vector<uint8_t> d = load(n);
    if (d.empty()) {
      std::printf("FAIL: %s/hevc/%s.h265 is missing\n", base.c_str(), n);
      ++failures;
      return;
    }
    play(n, d, m);
  };
  if (!only.empty() && std::find_if(std::begin(names), std::end(names), [&](const char* n) { return only == n; }) == std::end(names)) {
    // a stream that is not in the list (for probing the card with a new file): played as it is
    stream(only.c_str(), plain);
    return;
  }
  for (const char* n : names) {
    if (!only.empty() && only != n) continue;
    if (("," + skip + ",").find(std::string(",") + n + ",") != std::string::npos) continue;
    stream(n, plain);
  }
  if (!only.empty()) return;
  // How the parser is driven: the whole file in one packet, packets of arbitrary size (the parser looks at a NAL unit once 13 bytes of a
  // slice, VPS or SPS are in), packets that end a picture, timestamps given and not given, clock rates, the parser's own surface count.
  stream("b_pyramid", Mode{"onepacket", false});
  for (size_t c : {5, 13, 14, 64, 256, 1000, 4096}) {
    static char labels[8][16];
    static int nl = 0;
    std::snprintf(labels[nl], sizeof labels[nl], "chunk%zu", c);
    Mode cm{labels[nl++]};
    cm.chunk = c;
    stream("b_pyramid", cm);
  }
  stream("b_sei", Mode{"chunk13", true, 0, false, 1000, 1, 0, 0, 13});
  {
    Mode eop{"endofpicture"};
    eop.end_of_picture = true;
    stream("b_pyramid", eop);
  }
  stream("b_pyramid", Mode{"irregular_ts", true, 0, false, 1000, 1, 1, 0});
  stream("b_pyramid", Mode{"no_ts", true, 0, false, 1000, 1, 2, 0});
  stream("b_pyramid", Mode{"first_ts", true, 0, false, 1000, 1, 3, 0});
  stream("b_pyramid", Mode{"start_5000", true, 0, false, 1000, 1, 0, 5000});
  stream("b_pyramid", Mode{"clock_90k", true, 0, false, 90000, 1, 0, 0});
  stream("b_pyramid", Mode{"clock_default", true, 0, false, 0, 1, 0, 0});
  stream("b_pyramid", Mode{"surfaces8", true, 0, false, 1000, 8});
  stream("b_pyramid", Mode{"onepacket_90k", false, 0, false, 90000});
  stream("b_pyramid", Mode{"onepacket_default", false, 0, false, 0});
  // The display delay.
  for (const char* n : {"b_pyramid", "b_flat", "open_gop"}) {
    stream(n, Mode{"delay1", true, 1});
    stream(n, Mode{"delay2", true, 2});
    stream(n, Mode{"delay4", true, 4});
  }
  for (const char* n : {"p_low", "i_only", "weightp"}) {
    stream(n, Mode{"delay1", true, 1});
    stream(n, Mode{"delay3", true, 3});
  }
  // Display area and target size: a crop that fills the target is exact.
  for (const char* n : {"crop", "crop_odd", "size_136"}) stream(n, Mode{"cropped", true, 0, true});
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::string base, only, skip, file;
  bool caps_only = false;
  Mode fm{"packets"};
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
    else if (!std::strcmp(argv[i], "--skip") && i + 1 < argc) skip = argv[++i];
    else if (!std::strcmp(argv[i], "--file") && i + 1 < argc) file = argv[++i];
    else if (!std::strcmp(argv[i], "--delay") && i + 1 < argc) fm.delay = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--onepacket")) fm.per_au = false;
    else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) fm.chunk = static_cast<size_t>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--eop")) fm.end_of_picture = true;
    else if (!std::strcmp(argv[i], "--prefix") && i + 1 < argc) fm.prefix = static_cast<size_t>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--parser-surfaces") && i + 1 < argc) fm.parser_surfaces = static_cast<unsigned>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--clock") && i + 1 < argc) fm.clock = static_cast<unsigned>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--ts-mode") && i + 1 < argc) fm.ts_mode = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--crop")) fm.crop = true;
    else if (!std::strcmp(argv[i], "--caps")) caps_only = true;
    else if (!std::strcmp(argv[i], "--nv12")) g_force_nv12 = true;
    else base = argv[i];
  }
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx;
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);
  if (caps_only) {
    caps_and_creation();
    return 0;
  }
  if (!file.empty()) {
    const std::vector<uint8_t> data = slurp(file);
    if (data.empty()) {
      std::printf("FAIL: %s is missing\n", file.c_str());
      return 1;
    }
    play(file.substr(file.rfind('/') + 1), data, fm);
    return failures ? 1 : 0;
  }
  if (base.empty()) {
    std::fprintf(stderr, "usage: nvcuvid_hevc [--dump DIR] [--only NAME] [--skip NAMES] [--file PATH] DATA_DIR\n");
    return 2;
  }
  data_dir = base;
  if (only.empty()) caps_and_creation();
  run_hevc_suite(base, only, skip);
  std::printf("%s\n", failures ? "FAIL" : "done");
  return failures ? 1 : 0;
}
