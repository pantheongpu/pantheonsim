// NVDEC's H.264 decoder and parser as an application drives them, against what
// NVIDIA's libnvcuvid printed on an RTX 3060 (driver 595): decoder capabilities for
// every codec and format, decoder creation for H.264, and, for each stream in
// nvidia/tests/data/h264, every callback of the video parser (the sequence
// format, the picture parameters handed to cuvidDecodePicture, the display
// order and timestamps) and the CRC-32 of every displayed NV12 frame. H.264 is
// bit-exact by definition, so the CRCs of a conformant decoder are the card's.
//
// One line per fact. nvcuvid_h264.rtx3060.txt is what the card printed;
// run_nvcuvid.sh compares this program's output with it.
//
//   nvcuvid_h264 [--dump DIR] [--only NAME] DATA_DIR
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

static std::string dump_dir, data_dir, update_dir;
static int tolerance = 1;
static int failures = 0;

// ---- one parsed stream ----------------------------------------------------------------
struct Run {
  CUvideodecoder dec = nullptr;
  bool crop_in_decoder = false;   // target size and display area = the display rectangle
  bool resize = false;
  unsigned display_delay = 0;
  CUVIDEOFORMAT fmt{};
  unsigned w = 0, h = 0;          // the surface's visible size
  int frames = 0;
  uint32_t all_crc = 0;
  std::string name;
  int create_result = 0;
  int decode_fail = 0;
  unsigned tw = 0, th = 0;        // the surface's size
  bool scaled = false;            // the surface is a rescaled picture: pixels are compared with the card's, not checksummed
  std::vector<std::vector<uint8_t>> pix;
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
    // A second sequence: the decoder is reconfigured only if the application asks; the
    // programs here destroy it and make a new one.
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
  ci.OutputFormat = cudaVideoSurfaceFormat_NV12;
  ci.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
  ci.ulMaxWidth = f->coded_width;
  ci.ulMaxHeight = f->coded_height;
  ci.ulNumOutputSurfaces = 2;
  ci.display_area = {static_cast<short>(f->display_area.left), static_cast<short>(f->display_area.top), static_cast<short>(f->display_area.right),
                     static_cast<short>(f->display_area.bottom)};
  if (run->crop_in_decoder) {
    ci.ulTargetWidth = dw;
    ci.ulTargetHeight = dh;
  } else if (run->resize) {
    ci.ulTargetWidth = dw / 2 * 2 + 2;
    ci.ulTargetHeight = dh / 2 * 2 + 2;
    ci.display_area = {0, 0, static_cast<short>(f->coded_width), static_cast<short>(f->coded_height)};
  } else {
    ci.ulTargetWidth = f->coded_width;
    ci.ulTargetHeight = f->coded_height;
  }
  run->tw = ci.ulTargetWidth;
  run->th = ci.ulTargetHeight;
  if (run->resize) {
    run->w = ci.ulTargetWidth;
    run->h = ci.ulTargetHeight;
  } else {
    run->w = dw;
    run->h = dh;
  }
  const CUresult r = cuvidCreateDecoder(&run->dec, &ci);
  std::printf("  cuvidCreateDecoder: %d\n", r);
  run->create_result = r;
  if (r) return 0;
  return f->min_num_decode_surfaces;
}

static void print_pic(const CUVIDPICPARAMS* p) {
  const CUVIDH264PICPARAMS& h = p->CodecSpecific.h264;
  std::printf("  decode: mbs %dx%d index %d field %d bottom %d second %d intra %d ref %d length %u slices %u offsets", p->PicWidthInMbs,
              p->FrameHeightInMbs, p->CurrPicIdx, p->field_pic_flag, p->bottom_field_flag, p->second_field, p->intra_pic_flag, p->ref_pic_flag,
              p->nBitstreamDataLen, p->nNumSlices);
  for (unsigned i = 0; i < p->nNumSlices && i < 8; ++i) std::printf(" %u", p->pSliceDataOffsets[i]);
  std::printf(" head");
  for (unsigned i = 0; i < 6 && i < p->nBitstreamDataLen; ++i) std::printf(" %02x", p->pBitstreamData[i]);
  std::printf("\n");
  std::printf("    sps: log2fn %d poc_type %d log2poc %d dpaz %d fmo %d d8x8 %d refs %d rct %d depth %d/%d bypass %d\n", h.log2_max_frame_num_minus4,
              h.pic_order_cnt_type, h.log2_max_pic_order_cnt_lsb_minus4, h.delta_pic_order_always_zero_flag, h.frame_mbs_only_flag,
              h.direct_8x8_inference_flag, h.num_ref_frames, h.residual_colour_transform_flag, h.bit_depth_luma_minus8, h.bit_depth_chroma_minus8,
              h.qpprime_y_zero_transform_bypass_flag);
  std::printf("    pps: cabac %d poc_present %d l0 %d l1 %d wp %d wbi %d qp %d dfc %d redundant %d t8x8 %d mbaff %d cip %d cqp %d cqp2 %d\n",
              h.entropy_coding_mode_flag, h.pic_order_present_flag, h.num_ref_idx_l0_active_minus1, h.num_ref_idx_l1_active_minus1,
              h.weighted_pred_flag, h.weighted_bipred_idc, h.pic_init_qp_minus26, h.deblocking_filter_control_present_flag,
              h.redundant_pic_cnt_present_flag, h.transform_8x8_mode_flag, h.MbaffFrameFlag, h.constrained_intra_pred_flag, h.chroma_qp_index_offset,
              h.second_chroma_qp_index_offset);
  std::printf("    pic: ref %d frame_num %d poc %d %d fmo %d groups %d type %d qs %d change %u\n", h.ref_pic_flag, h.frame_num, h.CurrFieldOrderCnt[0],
              h.CurrFieldOrderCnt[1], h.fmo_aso_enable, h.num_slice_groups_minus1, h.slice_group_map_type, h.pic_init_qs_minus26,
              h.slice_group_change_rate_minus1);
  std::printf("    dpb:");
  int empty = 0;
  for (int i = 0; i < 16; ++i) {
    const CUVIDH264DPBENTRY& e = h.dpb[i];
    if (e.PicIdx == -1 && e.FrameIdx == 0 && e.is_long_term == 0 && e.not_existing == 0 && e.used_for_reference == 0 && e.FieldOrderCnt[0] == 0 &&
        e.FieldOrderCnt[1] == 0) {
      ++empty;
      continue;
    }
    std::printf(" [%d] pic %d fn %d lt %d ne %d used %d poc %d %d;", i, e.PicIdx, e.FrameIdx, e.is_long_term, e.not_existing, e.used_for_reference,
                e.FieldOrderCnt[0], e.FieldOrderCnt[1]);
  }
  std::printf(" (%d empty)\n", empty);
  unsigned flat4 = 0, flat8 = 0;
  for (int i = 0; i < 6; ++i)
    for (int k = 0; k < 16; ++k) flat4 += h.WeightScale4x4[i][k] != 16;
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 64; ++k) flat8 += h.WeightScale8x8[i][k] != 16;
  std::printf("    scaling: non-flat 4x4 %u 8x8 %u", flat4, flat8);
  if (flat4 || flat8) {
    uint32_t c = crc32(&h.WeightScale4x4[0][0], 96);
    c = crc32(&h.WeightScale8x8[0][0], 128, c);
    std::printf(" crc %08x", c);
  }
  std::printf("\n");
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
    unsigned w = run->w, h = run->h;
    if (run->scaled) {
      w = run->tw;
      h = run->th;
    }
    // The surface holds the coded picture, or (decoder created with the display area as
    // its target) just the display rectangle.
    const unsigned sh = run->scaled ? h : (run->crop_in_decoder || run->resize ? h : f.coded_height);
    const unsigned x0 = run->scaled || run->crop_in_decoder || run->resize ? 0 : f.display_area.left, y0 = run->scaled || run->crop_in_decoder || run->resize ? 0 : f.display_area.top;
    std::vector<uint8_t> all(static_cast<size_t>(pitch) * (sh + (sh + 1) / 2));
    cuMemcpyDtoH(all.data(), static_cast<CUdeviceptr>(dptr), all.size());
    std::vector<uint8_t> out;
    for (unsigned y = 0; y < h; ++y) out.insert(out.end(), &all[static_cast<size_t>(y0 + y) * pitch + x0], &all[static_cast<size_t>(y0 + y) * pitch + x0] + w);
    for (unsigned y = 0; y < (h + 1) / 2; ++y) {
      const uint8_t* row = &all[static_cast<size_t>(sh + y0 / 2 + y) * pitch + (x0 & ~1u)];
      out.insert(out.end(), row, row + ((w + 1) & ~1u));
    }
    if (run->scaled) {
      std::printf("  frame %d: %ux%u scaled\n", run->frames, w, h);
      run->pix.push_back(out);
    }
    const uint32_t c = run->scaled ? 0 : crc32(out.data(), out.size());
    if (!run->scaled) std::printf("  frame %d: %ux%u crc %08x luma %08x\n", run->frames, w, h, c, crc32(out.data(), static_cast<size_t>(w) * h));
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

// The byte ranges of an Annex B stream's access units: a picture starts at a
// slice NAL whose first_mb_in_slice is 0; the parameter sets, SEI and delimiters
// before it belong to it.
static std::vector<std::pair<size_t, size_t>> access_units(const std::vector<uint8_t>& s) {
  std::vector<size_t> starts;   // offset of each start code's first zero
  for (size_t i = 0; i + 3 < s.size(); ++i)
    if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1) starts.push_back(i > 0 && s[i - 1] == 0 ? i - 1 : i);
  std::vector<std::pair<size_t, size_t>> aus;
  size_t au_start = 0;
  bool have_vcl = false;
  for (size_t k = 0; k < starts.size(); ++k) {
    const size_t at = starts[k];
    size_t nal = at;
    while (s[nal] == 0) ++nal;
    ++nal;   // after the 01
    const int type = s[nal] & 31;
    if (type == 1 || type == 5) {
      // first_mb_in_slice == 0 is a leading one bit
      const bool first = (s[nal + 1] & 0x80) != 0;
      if (first && have_vcl) {
        aus.push_back({au_start, at});
        au_start = at;
        have_vcl = false;
      }
      have_vcl = true;
    } else if (have_vcl && (type == 6 || type == 7 || type == 8 || type == 9)) {
      aus.push_back({au_start, at});
      au_start = at;
      have_vcl = false;
    }
  }
  aus.push_back({au_start, s.size()});
  return aus;
}

struct Mode {
  const char* label;
  bool per_au = true;     // one packet per access unit, with a timestamp; else the whole file in one packet
  unsigned delay = 0;
  bool crop = false, resize = false;
  unsigned clock = 1000;
  unsigned parser_surfaces = 1;
  int ts_mode = 0;        // 0 regular (33 per picture), 1 irregular, 2 none, 3 the first packet only
  long long first_ts = 0;
  size_t chunk = 0;       // feed the stream in packets of this many bytes
  bool end_of_picture = false;   // flag every packet CUVID_PKT_ENDOFPICTURE
  bool scaled = false;
};

static void play(const std::string& name, const std::vector<uint8_t>& data, const Mode& m) {
  std::printf("stream %s [%s]\n", name.c_str(), m.label);
  Run r;
  r.name = std::string(name) + (m.label[0] == 'p' ? "" : std::string("_") + m.label);
  r.crop_in_decoder = m.crop;
  r.scaled = m.scaled;
  r.resize = m.resize;
  run = &r;
  CUvideoparser parser = nullptr;
  CUVIDPARSERPARAMS pp{};
  pp.CodecType = cudaVideoCodec_H264;
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
      for (size_t i = au.first; i + 3 < au.second; ++i)
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) std::printf(" %d", data[i + 3] & 31);
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
  if (m.scaled) {
    // the card's scaler is not reproduced exactly: compare with the pixels it produced, to a level
    const std::string path = data_dir + "/nvdec/h264_" + name + "_" + m.label + ".nv12";
    if (!update_dir.empty()) {
      std::ofstream o(update_dir + "/h264_" + name + "_" + m.label + ".nv12", std::ios::binary);
      for (const auto& f : r.pix) o.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
      std::printf("  pixels %s [%s]: match the card's\n", name.c_str(), m.label);
    } else {
      const std::vector<uint8_t> want = slurp(path);
      std::vector<uint8_t> got;
      for (const auto& f : r.pix) got.insert(got.end(), f.begin(), f.end());
      int worst = 0;
      if (want.size() != got.size()) worst = 255;
      else
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(static_cast<int>(got[i]) - static_cast<int>(want[i])));
      const bool ok = worst <= tolerance;
      std::printf("  pixels %s [%s]: %s\n", name.c_str(), m.label, ok ? "match the card's" : "FAIL, differ from the card's");
      std::fprintf(stderr, "  %s [%s]: worst difference %d over %zu bytes (the card's file has %zu)\n", name.c_str(), m.label, worst, got.size(), want.size());
      if (!ok) ++failures;
    }
  }
  std::printf("  summary %s: frames %d crc %08x\n", name.c_str(), r.frames, r.scaled ? 0u : r.all_crc);
  cuvidDestroyVideoParser(parser);
  if (r.dec) cuvidDestroyDecoder(r.dec);
  run = nullptr;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::string base, only, codecs = "h264";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--update") && i + 1 < argc) update_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--tolerance") && i + 1 < argc) tolerance = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--codecs") && i + 1 < argc) codecs = argv[++i];
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
    else base = argv[i];
  }
  if (base.empty()) {
    std::fprintf(stderr, "usage: nvcuvid_h264 [--dump DIR] [--only NAME] DATA_DIR\n");
    return 2;
  }
  data_dir = base;
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx;
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);

  if (only.empty()) {
    // ---- capabilities: every codec, every format ------------------------------------------
    static const char* const codec_names[] = {"MPEG1", "MPEG2", "MPEG4", "VC1", "H264", "JPEG", "H264_SVC", "H264_MVC", "HEVC", "VP8", "VP9", "AV1"};
    for (int codec = 0; codec < 12; ++codec) {
      if (codec == cudaVideoCodec_JPEG) continue;   // nvcuvid_paths.cpp
      std::string lower;
      for (const char* p = codec_names[codec]; *p; ++p) lower += static_cast<char>(std::tolower(*p));
      if (("," + codecs + ",").find("," + lower + ",") == std::string::npos) continue;   // only the codecs the simulator decodes
      for (int chroma = 0; chroma <= 3; ++chroma)
        for (int depth : {0, 2, 4}) {
          CUVIDDECODECAPS c{};
          poke(c.eCodecType, codec);
          c.eChromaFormat = static_cast<cudaVideoChromaFormat>(chroma);
          c.nBitDepthMinus8 = depth;
          const CUresult r = cuvidGetDecoderCaps(&c);
          std::printf("caps %s chroma %d depth %d: %d supported %u max %ux%u mb %u min %ux%u outputs 0x%x histogram %u/%u/%u\n", codec_names[codec], chroma,
                      8 + depth, r, c.bIsSupported, c.nMaxWidth, c.nMaxHeight, c.nMaxMBCount, c.nMinWidth, c.nMinHeight, c.nOutputFormatMask,
                      c.bIsHistogramSupported, c.nCounterBitDepth, c.nMaxHistogramBins);
        }
    }
    // ---- decoder creation -----------------------------------------------------------------
    const auto base_info = [] {
      CUVIDDECODECREATEINFO ci{};
      ci.ulWidth = 192;
      ci.ulHeight = 128;
      ci.ulNumDecodeSurfaces = 4;
      ci.CodecType = cudaVideoCodec_H264;
      ci.ChromaFormat = cudaVideoChromaFormat_420;
      ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
      ci.OutputFormat = cudaVideoSurfaceFormat_NV12;
      ci.ulTargetWidth = 192;
      ci.ulTargetHeight = 128;
      ci.ulNumOutputSurfaces = 2;
      ci.ulMaxWidth = 192;
      ci.ulMaxHeight = 128;
      ci.display_area = {0, 0, 192, 128};
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
    create("H264 valid", [](CUVIDDECODECREATEINFO&) {});
    create("H264 16x16", [](CUVIDDECODECREATEINFO& c) { c.ulWidth = c.ulMaxWidth = c.ulTargetWidth = c.ulHeight = c.ulMaxHeight = c.ulTargetHeight = 16; });
    create("H264 32x32", [](CUVIDDECODECREATEINFO& c) { c.ulWidth = c.ulMaxWidth = c.ulTargetWidth = c.ulHeight = c.ulMaxHeight = c.ulTargetHeight = 32; });
    create("H264 4096x4096", [](CUVIDDECODECREATEINFO& c) { c.ulWidth = c.ulMaxWidth = c.ulTargetWidth = c.ulHeight = c.ulMaxHeight = c.ulTargetHeight = 4096; });
    create("H264 8192x8192", [](CUVIDDECODECREATEINFO& c) { c.ulWidth = c.ulMaxWidth = c.ulTargetWidth = c.ulHeight = c.ulMaxHeight = c.ulTargetHeight = 8192; });
    for (const auto& wh : {std::pair<int, int>{48, 16}, {47, 16}, {48, 15}, {64, 16}, {16, 64}, {4096, 16}, {4096, 4097}, {4112, 4096}, {4080, 4080}, {4096, 4096}, {2, 2}, {50, 18}}) {
      char name[64];
      std::snprintf(name, sizeof name, "H264 size %dx%d", wh.first, wh.second);
      create(name, [&](CUVIDDECODECREATEINFO& c) {
        c.ulWidth = c.ulMaxWidth = c.ulTargetWidth = wh.first;
        c.ulHeight = c.ulMaxHeight = c.ulTargetHeight = wh.second;
        c.display_area = {0, 0, static_cast<short>(wh.first), static_cast<short>(wh.second)};
      });
    }
    create("H264 0 output surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumOutputSurfaces = 0; });
    create("H264 64 output surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumOutputSurfaces = 64; });
    create("H264 target 0", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = c.ulTargetHeight = 0; });
    create("H264 target 4096x4096 from 192x128", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = c.ulTargetHeight = 4096; });
    create("H264 target 8192", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = c.ulTargetHeight = 8192; });
    create("H264 max size 0", [](CUVIDDECODECREATEINFO& c) { c.ulMaxWidth = c.ulMaxHeight = 0; });
    create("H264 max size below size", [](CUVIDDECODECREATEINFO& c) { c.ulMaxWidth = c.ulMaxHeight = 64; });
    create("H264 display area outside", [](CUVIDDECODECREATEINFO& c) { c.display_area = {0, 0, 500, 500}; });
    create("H264 display area empty", [](CUVIDDECODECREATEINFO& c) { c.display_area = {0, 0, 0, 0}; });
    create("H264 odd target", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = 97; c.ulTargetHeight = 65; });
    create("H264 monochrome", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_Monochrome; });
    create("H264 422", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_422; });
    create("H264 444", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_444; });
    create("H264 10 bit", [](CUVIDDECODECREATEINFO& c) { c.bitDepthMinus8 = 2; });
    create("H264 10 bit P016", [](CUVIDDECODECREATEINFO& c) { c.bitDepthMinus8 = 2; c.OutputFormat = cudaVideoSurfaceFormat_P016; });
    create("H264 P016 output", [](CUVIDDECODECREATEINFO& c) { c.OutputFormat = cudaVideoSurfaceFormat_P016; });
    create("H264 0 surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 0; });
    create("H264 1 surface", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 1; });
    create("H264 32 surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 32; });
    create("H264 33 surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 33; });
    create("H264 intra only", [](CUVIDDECODECREATEINFO& c) { c.ulIntraDecodeOnly = 1; });
    create("H264 target 96x64", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = 96; c.ulTargetHeight = 64; });
    create("H264 deinterlace bob", [](CUVIDDECODECREATEINFO& c) { c.DeinterlaceMode = cudaVideoDeinterlaceMode_Bob; });
    create("H264 deinterlace adaptive", [](CUVIDDECODECREATEINFO& c) { c.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive; });
    create("H264 PreferCUDA", [](CUVIDDECODECREATEINFO& c) { c.ulCreationFlags = cudaVideoCreate_PreferCUDA; });
    create("H264 SVC", [](CUVIDDECODECREATEINFO& c) { c.CodecType = cudaVideoCodec_H264_SVC; });
    create("H264 MVC", [](CUVIDDECODECREATEINFO& c) { c.CodecType = cudaVideoCodec_H264_MVC; });
    {
      CUVIDPARSERPARAMS pp{};
      for (int codec : {cudaVideoCodec_H264, cudaVideoCodec_H264_SVC, cudaVideoCodec_H264_MVC}) {
        pp.CodecType = static_cast<cudaVideoCodec>(codec);
        pp.ulMaxNumDecodeSurfaces = 1;
        CUvideoparser p = nullptr;
        const CUresult r = cuvidCreateVideoParser(&p, &pp);
        std::printf("CreateVideoParser(codec %d): %d\n", codec, r);
        if (!r) cuvidDestroyVideoParser(p);
      }
    }
  }

  // ---- the streams -----------------------------------------------------------------------
  static const char* const names[] = {"idr_mid",  "sps_novui",   "sps_notiming", "sps_norestr", "sps_level40",    "sps_mdfb1",    "sps_reorder1", "sps_p_novui", "i_cavlc",
                                      "p_cavlc",  "p_cabac",     "b_spatial",    "b_temporal",  "b_cavlc",       "high_8x8",     "high_cqm",     "high_cavlc_8x8", "weightp",
                                      "lowqp",    "lowqp_cavlc", "highqp",       "mbaff",       "mbaff_cavlc",   "multislice_b", "long_gop",     "deblock_off", "deblock_strong",
                                      "p_ref1",   "p_ref2",      "p_ref4",       "p_ref6",      "b_ref2",        "b_ref3"};
  const Mode plain{"packets"};
  auto load = [&](const char* n) { return slurp(base + "/h264/" + n + ".h264"); };
  for (const char* n : names) {
    if (!only.empty() && only != n) continue;
    const std::vector<uint8_t> data = load(n);
    if (data.empty()) {
      std::printf("FAIL: %s/h264/%s.h264 is missing\n", base.c_str(), n);
      ++failures;
      continue;
    }
    play(n, data, plain);
  }
  if (only.empty()) {
    const auto stream = [&](const char* n, const Mode& m) {
      const std::vector<uint8_t> d = load(n);
      if (d.empty()) {
        std::printf("FAIL: %s/h264/%s.h264 is missing\n", base.c_str(), n);
        ++failures;
        return;
      }
      play(n, d, m);
    };
    // How the parser is driven: the whole file in one packet, packets of arbitrary size (the parser looks at a NAL unit
    // once 256 bytes of it are in, or it is terminated), packets that end a picture, timestamps given and not given,
    // clock rates, the parser's own surface count.
    stream("b_spatial", Mode{"onepacket", false});
    for (size_t c : {64, 100, 256, 1000, 4096}) {
      static char labels[8][16];
      static int nl = 0;
      std::snprintf(labels[nl], sizeof labels[nl], "chunk%zu", c);
      Mode cm{labels[nl++]};
      cm.chunk = c;
      stream("b_spatial", cm);
    }
    {
      Mode eop{"endofpicture"};
      eop.end_of_picture = true;
      stream("b_spatial", eop);
    }
    stream("b_spatial", Mode{"irregular_ts", true, 0, false, false, 1000, 1, 1, 0});
    stream("b_spatial", Mode{"no_ts", true, 0, false, false, 1000, 1, 2, 0});
    stream("b_spatial", Mode{"first_ts", true, 0, false, false, 1000, 1, 3, 0});
    stream("b_spatial", Mode{"start_5000", true, 0, false, false, 1000, 1, 0, 5000});
    stream("b_spatial", Mode{"clock_90k", true, 0, false, false, 90000, 1, 0, 0});
    stream("b_spatial", Mode{"clock_default", true, 0, false, false, 0, 1, 0, 0});
    stream("b_spatial", Mode{"surfaces8", true, 0, false, false, 1000, 8});
    stream("b_spatial", Mode{"onepacket_90k", false, 0, false, false, 90000});
    stream("b_spatial", Mode{"onepacket_default", false, 0, false, false, 0});
    stream("b_spatial", Mode{"onepacket_ts7", false, 0, false, false, 1000, 1, 0, 7});
    stream("sps_notiming", Mode{"onepacket", false});
    stream("sps_novui", Mode{"onepacket", false});
    // The display delay. The card keeps up to three pictures back (the decode queue) when nothing else holds
    // surfaces; with B pictures its choice depends on surface pressure, and the interleaving of display and decode
    // callbacks for delays above one is only reproduced for streams without reordering (the order and timestamps of
    // the displayed pictures are the card's in every case).
    stream("b_spatial", Mode{"delay1", true, 1});
    for (const char* n : {"p_cabac", "i_cavlc", "sps_novui", "long_gop"}) {
      stream(n, Mode{"delay1", true, 1});
    }
    for (const char* n : {"p_cabac", "i_cavlc", "sps_novui"}) stream(n, Mode{"delay3", true, 3});
    for (const char* n : {"p_ref1", "p_ref2", "p_ref4", "p_ref6"}) {
      static const char* const labels[] = {"delay1", "delay2", "delay3", "delay4", "delay6"};
      static const unsigned delays[] = {1, 2, 3, 4, 6};
      for (int i = 0; i < 5; ++i) stream(n, Mode{labels[i], true, delays[i]});
    }
    stream("b_ref2", Mode{"delay1", true, 1});
    stream("b_ref3", Mode{"delay1", true, 1});
    // Display area and target size: a crop that fills the target is exact, a rescale is close to the card's.
    stream("odd_size", Mode{"cropped", true, 0, true});
    {
      Mode m{"scaled", true};
      m.scaled = true;
      stream("odd_size", m);
      Mode r{"resized", true, 0, false, true};
      r.scaled = true;
      stream("odd_size", r);
    }
  }
  std::printf("%s\n", failures ? "FAIL" : "done");
  return failures ? 1 : 0;
}
