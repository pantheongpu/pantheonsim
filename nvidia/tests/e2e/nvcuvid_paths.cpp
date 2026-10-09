// NVDEC's cuvid API as an application calls it, against what NVIDIA's
// libnvcuvid answered on an RTX 3060 (driver 595): decoder capabilities, decoder
// creation and its limits, decode / map / unmap / status, reconfiguration, context
// locks, and the video parser with Motion JPEG -- the sequence, decode and
// display callbacks with their fields, timestamps and return values -- and the
// decoded NV12 pixels of the JPEG fixtures (baseline, progressive, restart
// intervals, 4:2:0 / 4:2:2 / 4:4:4 and grey; 61x45 and 333x251), with the decoder
// cutting, padding and resampling to its target size.
//
// One line per fact. nvidia/tests/e2e/nvcuvid_paths.rtx3060.txt is what the card
// printed; run_nvcuvid.sh compares this program's output with it, and the pixels
// with the card's (nvidia/tests/data/nvdec/*.nv12, tolerance 2 levels: the
// decoder's inverse DCT is not bit-identical to NVDEC's).
//
// Not in this file: codecs other than JPEG. The card decodes H.264, HEVC and the
// rest; VirtualGPU reports them unsupported (cuvidGetDecoderCaps bIsSupported 0).
#include <dlfcn.h>
#include <cuda.h>

#include <cmath>
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

// Store a number into an enum field the API defines no name for, byte for byte: loading
// such a value back as its enum type is undefined.
template <class E>
static void poke(E& field, uint32_t v) {
  static_assert(sizeof(E) == sizeof v, "a 32-bit enum");
  std::memcpy(&field, &v, sizeof v);
}

static std::string jpeg_dir, data_dir, update_dir;
static int failures = 0;
static int tolerance = 2;

static std::vector<uint8_t> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---- the parser's callbacks, recording what they are given ------------------------
struct Run {
  CUvideodecoder dec = nullptr;
  int dec_w = 0, dec_h = 0;        // 0: the sequence's coded size
  int target_w = 0, target_h = 0;  // 0: the decoder's size
  int seq_ret = 1, dec_ret = 1;
  bool verbose = true;
  int frames = 0;
  std::vector<std::vector<uint8_t>> surfaces;   // visible region of each displayed frame
  std::vector<unsigned> pitches;
  CUVIDEOFORMAT fmt{};
};
static Run* run = nullptr;

static int CUDAAPI seq_cb(void*, CUVIDEOFORMAT* f) {
  run->fmt = *f;
  if (run->verbose)
    std::printf("  sequence: codec %d rate %u/%u progressive %d depth %d/%d chroma %d surfaces %d coded %ux%u display %d,%d,%d,%d bitrate %u aspect %d:%d "
                "signal %02x %02x %02x %02x\n",
                f->codec, f->frame_rate.numerator, f->frame_rate.denominator, f->progressive_sequence, f->bit_depth_luma_minus8,
                f->bit_depth_chroma_minus8, f->chroma_format, f->min_num_decode_surfaces, f->coded_width, f->coded_height, f->display_area.left,
                f->display_area.top, f->display_area.right, f->display_area.bottom, f->bitrate, f->display_aspect_ratio.x, f->display_aspect_ratio.y,
                reinterpret_cast<const unsigned char*>(&f->video_signal_description)[0], reinterpret_cast<const unsigned char*>(&f->video_signal_description)[1],
                reinterpret_cast<const unsigned char*>(&f->video_signal_description)[2], reinterpret_cast<const unsigned char*>(&f->video_signal_description)[3]);
  if (!run->dec) {
    CUVIDDECODECREATEINFO ci{};
    ci.ulWidth = run->dec_w ? run->dec_w : f->coded_width;
    ci.ulHeight = run->dec_h ? run->dec_h : f->coded_height;
    ci.ulNumDecodeSurfaces = run->seq_ret > 1 ? run->seq_ret : f->min_num_decode_surfaces;
    ci.CodecType = f->codec;
    ci.ChromaFormat = f->chroma_format;
    ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    ci.OutputFormat = cudaVideoSurfaceFormat_NV12;
    ci.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    ci.ulTargetWidth = run->target_w ? run->target_w : ci.ulWidth;
    ci.ulTargetHeight = run->target_h ? run->target_h : ci.ulHeight;
    ci.ulNumOutputSurfaces = 2;
    ci.ulMaxWidth = ci.ulWidth;
    ci.ulMaxHeight = ci.ulHeight;
    ci.display_area = {0, 0, static_cast<short>(ci.ulWidth), static_cast<short>(ci.ulHeight)};
    const CUresult r = cuvidCreateDecoder(&run->dec, &ci);
    if (run->verbose) std::printf("  cuvidCreateDecoder: %d\n", r);
    if (r) return 0;
  }
  return run->seq_ret;
}

static int CUDAAPI dec_cb(void*, CUVIDPICPARAMS* p) {
  if (run->verbose)
    std::printf("  decode: mbs %dx%d index %d field %d bottom %d second %d intra %d ref %d length %u slices %u\n", p->PicWidthInMbs, p->FrameHeightInMbs,
                p->CurrPicIdx, p->field_pic_flag, p->bottom_field_flag, p->second_field, p->intra_pic_flag, p->ref_pic_flag, p->nBitstreamDataLen,
                p->nNumSlices);
  const CUresult r = cuvidDecodePicture(run->dec, p);
  if (run->verbose) std::printf("  cuvidDecodePicture: %d\n", r);
  return run->dec_ret;
}

static int CUDAAPI disp_cb(void*, CUVIDPARSERDISPINFO* d) {
  CUVIDGETDECODESTATUS ds{};
  const CUresult sr = cuvidGetDecodeStatus(run->dec, d->picture_index, &ds);
  if (run->verbose)
    // The card decodes asynchronously: a picture may still be in progress (1) when its
    // display callback runs, so only "done or in progress" and "failed" are printed.
    std::printf("  display: index %d progressive %d top-field-first %d repeat %d timestamp %lld; decode status call %d, %s\n", d->picture_index,
                d->progressive_frame, d->top_field_first, d->repeat_first_field, static_cast<long long>(d->timestamp), sr,
                ds.decodeStatus == cuvidDecodeStatus_Success || ds.decodeStatus == cuvidDecodeStatus_InProgress ? "decoded"
                                                                                                                 : (ds.decodeStatus == cuvidDecodeStatus_Invalid ? "invalid" : "failed"));
  unsigned long long dptr = 0;
  unsigned pitch = 0;
  CUVIDPROCPARAMS pp{};
  pp.progressive_frame = d->progressive_frame;
  pp.top_field_first = d->top_field_first;
  const CUresult r = cuvidMapVideoFrame64(run->dec, d->picture_index, &dptr, &pitch, &pp);
  if (run->verbose) std::printf("  cuvidMapVideoFrame64: %d, pitch %u\n", r, pitch);
  if (r == CUDA_SUCCESS) {
    // The visible region: target luma, then chroma interleaved, unpitched.
    CUVIDEOFORMAT& f = run->fmt;
    unsigned w = run->target_w ? run->target_w : (run->dec_w ? run->dec_w : f.coded_width);
    unsigned h = run->target_h ? run->target_h : (run->dec_h ? run->dec_h : f.coded_height);
    std::vector<uint8_t> all(static_cast<size_t>(pitch) * (h + (h + 1) / 2));
    cuMemcpyDtoH(all.data(), static_cast<CUdeviceptr>(dptr), all.size());
    std::vector<uint8_t> out;
    for (unsigned y = 0; y < h; ++y) out.insert(out.end(), &all[static_cast<size_t>(y) * pitch], &all[static_cast<size_t>(y) * pitch] + w);
    for (unsigned y = 0; y < (h + 1) / 2; ++y)
      out.insert(out.end(), &all[static_cast<size_t>(h + y) * pitch], &all[static_cast<size_t>(h + y) * pitch] + ((w + 1) & ~1u));
    run->surfaces.push_back(std::move(out));
    run->pitches.push_back(pitch);
    cuvidUnmapVideoFrame64(run->dec, dptr);
  }
  ++run->frames;
  return 1;
}

static CUVIDPARSERPARAMS parser_params(unsigned clock = 1000, unsigned surfaces = 1) {
  CUVIDPARSERPARAMS pp{};
  pp.CodecType = cudaVideoCodec_JPEG;
  pp.ulMaxNumDecodeSurfaces = surfaces;
  pp.ulClockRate = clock;
  pp.ulMaxDisplayDelay = 0;
  pp.pfnSequenceCallback = seq_cb;
  pp.pfnDecodePicture = dec_cb;
  pp.pfnDisplayPicture = disp_cb;
  return pp;
}

// Parse `data` as Motion JPEG (one packet, timestamp 1000) with the given decoder
// and target sizes, and return the displayed surfaces.
static Run* decode(const std::vector<uint8_t>& data, int dec_w = 0, int dec_h = 0, int tw = 0, int th = 0, bool verbose = true) {
  auto* r = new Run();
  r->dec_w = dec_w;
  r->dec_h = dec_h;
  r->target_w = tw;
  r->target_h = th;
  r->verbose = verbose;
  run = r;
  CUvideoparser parser = nullptr;
  CUVIDPARSERPARAMS pp = parser_params();
  if (cuvidCreateVideoParser(&parser, &pp)) {
    std::printf("  FAIL: cuvidCreateVideoParser\n");
    ++failures;
    return r;
  }
  CUVIDSOURCEDATAPACKET pk{};
  pk.payload = data.data();
  pk.payload_size = data.size();
  pk.flags = CUVID_PKT_TIMESTAMP;
  pk.timestamp = 1000;
  const CUresult rc = cuvidParseVideoData(parser, &pk);
  if (verbose) std::printf("  cuvidParseVideoData: %d\n", rc);
  CUVIDSOURCEDATAPACKET eos{};
  eos.flags = CUVID_PKT_ENDOFSTREAM;
  const CUresult er = cuvidParseVideoData(parser, &eos);
  if (verbose) std::printf("  end of stream: %d, %d frames displayed\n", er, r->frames);
  cuvidDestroyVideoParser(parser);
  return r;
}

static void finish(Run* r) {
  if (r->dec) cuvidDestroyDecoder(r->dec);
  delete r;
  run = nullptr;
}

// Compare (or, with --update, store) the pixels of a decode.
static void pixels(const char* name, Run* r, size_t frame = 0) {
  if (frame >= r->surfaces.size()) {
    std::printf("pixels %s: FAIL, no frame\n", name);
    ++failures;
    return;
  }
  const std::vector<uint8_t>& got = r->surfaces[frame];
  const std::string path = data_dir + "/" + name + ".nv12";
  if (!update_dir.empty()) {
    std::ofstream o(update_dir + "/" + name + ".nv12", std::ios::binary);
    o.write(reinterpret_cast<const char*>(got.data()), static_cast<std::streamsize>(got.size()));
    std::printf("pixels %s: match the card's\n", name);
    std::fprintf(stderr, "  %s: stored %zu bytes\n", name, got.size());
    return;
  }
  const std::vector<uint8_t> want = slurp(path);
  if (want.size() != got.size()) {
    std::printf("pixels %s: FAIL, %zu bytes, the card's file has %zu\n", name, got.size(), want.size());
    ++failures;
    return;
  }
  int worst = 0;
  size_t over = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const int d = std::abs(static_cast<int>(got[i]) - static_cast<int>(want[i]));
    worst = std::max(worst, d);
    if (d > 1) ++over;
  }
  const bool ok = worst <= tolerance;
  std::printf("pixels %s: %s\n", name, ok ? "match the card's" : "FAIL, differ from the card's");
  std::fprintf(stderr, "  %s: worst difference %d, %zu samples off by more than one of %zu\n", name, worst, over, got.size());
  if (!ok) ++failures;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const char* env = std::getenv("VGPU_E2E_DATA");
  const std::string base = env ? env : std::string(__FILE__).substr(0, std::string(__FILE__).rfind('/')) + "/../data";
  jpeg_dir = base + "/jpeg";
  data_dir = base + "/nvdec";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--update") && i + 1 < argc) update_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--tolerance") && i + 1 < argc) tolerance = std::atoi(argv[++i]);
  }
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx;
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);

  // ---- capabilities: the JPEG engine ----------------------------------------------
  for (int chroma = 0; chroma <= 3; ++chroma)
    for (int depth : {0, 2, 4}) {
      CUVIDDECODECAPS c{};
      c.eCodecType = cudaVideoCodec_JPEG;
      c.eChromaFormat = static_cast<cudaVideoChromaFormat>(chroma);
      c.nBitDepthMinus8 = depth;
      const CUresult r = cuvidGetDecoderCaps(&c);
      std::printf("caps JPEG chroma %d depth %d: %d supported %u max %ux%u mb %u min %ux%u outputs 0x%x\n", chroma, 8 + depth, r, c.bIsSupported,
                  c.nMaxWidth, c.nMaxHeight, c.nMaxMBCount, c.nMinWidth, c.nMinHeight, c.nOutputFormatMask);
    }
  {
    CUVIDDECODECAPS c{};
    c.eCodecType = cudaVideoCodec_JPEG;
    poke(c.eChromaFormat, 9);
    cuvidGetDecoderCaps(&c);
    std::printf("caps JPEG chroma 9: supported %u\n", c.bIsSupported);
    c = {};
    poke(c.eCodecType, 99);
    c.eChromaFormat = cudaVideoChromaFormat_420;
    const CUresult cr = cuvidGetDecoderCaps(&c);
    std::printf("caps codec 99: %d supported %u\n", cr, c.bIsSupported);
  }

  // ---- decoder creation ------------------------------------------------------------
  auto base_info = [] {
    CUVIDDECODECREATEINFO ci{};
    ci.ulWidth = 192;
    ci.ulHeight = 128;
    ci.ulNumDecodeSurfaces = 2;
    ci.CodecType = cudaVideoCodec_JPEG;
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
  create("valid", [](CUVIDDECODECREATEINFO&) {});
  create("width 0", [](CUVIDDECODECREATEINFO& c) { c.ulWidth = 0; });
  create("height 0", [](CUVIDDECODECREATEINFO& c) { c.ulHeight = 0; });
  create("0 decode surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 0; });
  create("32 decode surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 32; });
  create("33 decode surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumDecodeSurfaces = 33; });
  create("0 output surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumOutputSurfaces = 0; });
  create("64 output surfaces", [](CUVIDDECODECREATEINFO& c) { c.ulNumOutputSurfaces = 64; });
  create("width 32", [](CUVIDDECODECREATEINFO& c) { c.ulWidth = c.ulMaxWidth = c.ulTargetWidth = 32; });
  create("width 40000", [](CUVIDDECODECREATEINFO& c) { c.ulWidth = c.ulMaxWidth = c.ulTargetWidth = 40000; });
  create("monochrome", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_Monochrome; });
  create("422", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_422; });
  create("444", [](CUVIDDECODECREATEINFO& c) { c.ChromaFormat = cudaVideoChromaFormat_444; });
  create("chroma 9", [](CUVIDDECODECREATEINFO& c) { poke(c.ChromaFormat, 9); });
  create("10 bit", [](CUVIDDECODECREATEINFO& c) { c.bitDepthMinus8 = 2; });
  create("P016 output", [](CUVIDDECODECREATEINFO& c) { c.OutputFormat = cudaVideoSurfaceFormat_P016; });
  create("YUV444 output", [](CUVIDDECODECREATEINFO& c) { c.OutputFormat = cudaVideoSurfaceFormat_YUV444; });
  create("YUV444 16-bit output", [](CUVIDDECODECREATEINFO& c) { c.OutputFormat = cudaVideoSurfaceFormat_YUV444_16Bit; });
  create("output format 9", [](CUVIDDECODECREATEINFO& c) { poke(c.OutputFormat, 9); });
  create("target 96x64", [](CUVIDDECODECREATEINFO& c) { c.ulTargetWidth = 96; c.ulTargetHeight = 64; });
  create("codec 77", [](CUVIDDECODECREATEINFO& c) { poke(c.CodecType, 77); });
  create("PreferCUDA", [](CUVIDDECODECREATEINFO& c) { c.ulCreationFlags = cudaVideoCreate_PreferCUDA; });
  create("max size 0", [](CUVIDDECODECREATEINFO& c) { c.ulMaxWidth = c.ulMaxHeight = 0; });
  create("max size below size", [](CUVIDDECODECREATEINFO& c) { c.ulMaxWidth = c.ulMaxHeight = 64; });
  create("display area outside", [](CUVIDDECODECREATEINFO& c) { c.display_area = {0, 0, 500, 500}; });
  std::printf("DestroyDecoder(null): %d\n", cuvidDestroyDecoder(nullptr));

  // ---- decode, status, map, unmap, reconfigure ------------------------------------------
  const std::vector<uint8_t> b420 = slurp(jpeg_dir + "/b420.jpg");
  const std::vector<uint8_t> b444 = slurp(jpeg_dir + "/b444.jpg");
  if (b420.empty() || b444.empty()) {
    std::printf("FAIL: the JPEG fixtures are missing from %s\n", jpeg_dir.c_str());
    return 1;
  }
  {
    CUVIDDECODECREATEINFO ci = base_info();
    CUvideodecoder d = nullptr;
    cuvidCreateDecoder(&d, &ci);
    const auto pic = [&](const std::vector<uint8_t>& j, int idx) {
      CUVIDPICPARAMS pp{};
      pp.pBitstreamData = j.data();
      pp.nBitstreamDataLen = static_cast<unsigned>(j.size());
      pp.CurrPicIdx = idx;
      pp.PicWidthInMbs = 4;
      pp.FrameHeightInMbs = 3;
      pp.nNumSlices = 1;
      pp.intra_pic_flag = 1;
      unsigned zero = 0;
      pp.pSliceDataOffsets = &zero;
      return cuvidDecodePicture(d, &pp);
    };
    const auto status = [&](int idx) {
      CUVIDGETDECODESTATUS s{};
      const CUresult r = cuvidGetDecodeStatus(d, idx, &s);
      std::printf("GetDecodeStatus(%d): %d, status %d\n", idx, r, static_cast<int>(s.decodeStatus));
    };
    status(0);
    status(9);
    status(-1);
    unsigned long long p0 = 0, p1 = 0;
    unsigned pitch = 0;
    CUVIDPROCPARAMS pr{};
    pr.progressive_frame = 1;
    CUresult mr = cuvidMapVideoFrame64(d, 0, &p0, &pitch, &pr);
    std::printf("MapVideoFrame (never decoded): %d, pitch %u\n", mr, pitch);
    mr = cuvidMapVideoFrame64(d, 0, &p1, &pitch, &pr);
    std::printf("MapVideoFrame (again): %d, same pointer %d\n", mr, p0 == p1);
    std::printf("MapVideoFrame (index 9): %d\n", cuvidMapVideoFrame64(d, 9, &p1, &pitch, &pr));
    std::printf("MapVideoFrame (index -1): %d\n", cuvidMapVideoFrame64(d, -1, &p1, &pitch, &pr));
    std::printf("MapVideoFrame (no parameters): %d\n", cuvidMapVideoFrame64(d, 0, &p1, &pitch, nullptr));
    {
      const CUresult u1 = cuvidUnmapVideoFrame64(d, p0), u2 = cuvidUnmapVideoFrame64(d, p0), u3 = cuvidUnmapVideoFrame64(d, 0x1234);
      std::printf("UnmapVideoFrame: %d, again: %d, unknown pointer: %d\n", u1, u2, u3);
    }
    std::printf("DecodePicture: %d\n", pic(b420, 0));
    status(0);
    status(1);
    std::printf("DecodePicture (index 1): %d\n", pic(b420, 1));
    status(1);
    std::printf("DecodePicture (index 5 of 2): %d\n", pic(b420, 5));
    std::printf("DecodePicture (index -1): %d\n", pic(b420, -1));
    std::printf("DecodePicture (4:4:4 picture, 4:2:0 decoder): %d\n", pic(b444, 0));
    std::vector<uint8_t> junk(100, 7);
    std::printf("DecodePicture (junk): %d\n", pic(junk, 0));
    status(0);
    {
      const std::vector<uint8_t> progressive = slurp(jpeg_dir + "/p420.jpg");
      std::printf("DecodePicture (progressive JPEG): %d\n", pic(progressive, 1));
      status(1);
    }
    {
      CUVIDPICPARAMS pp{};
      std::printf("DecodePicture (empty parameters): %d\n", cuvidDecodePicture(d, &pp));
      pp.pBitstreamData = b420.data();
      pp.nBitstreamDataLen = 0;
      std::printf("DecodePicture (length 0): %d\n", cuvidDecodePicture(d, &pp));
      std::printf("DecodePicture (null decoder): %d\n", cuvidDecodePicture(nullptr, &pp));
    }
    CUVIDRECONFIGUREDECODERINFO ri{};
    ri.ulWidth = 192;
    ri.ulHeight = 128;
    ri.ulTargetWidth = 192;
    ri.ulTargetHeight = 128;
    ri.ulNumDecodeSurfaces = 2;
    std::printf("ReconfigureDecoder (same): %d\n", cuvidReconfigureDecoder(d, &ri));
    ri.ulWidth = ri.ulTargetWidth = 256;
    std::printf("ReconfigureDecoder (larger than the maximum): %d\n", cuvidReconfigureDecoder(d, &ri));
    ri.ulWidth = ri.ulTargetWidth = 128;
    ri.ulHeight = ri.ulTargetHeight = 96;
    std::printf("ReconfigureDecoder (smaller): %d\n", cuvidReconfigureDecoder(d, &ri));
    ri.ulNumDecodeSurfaces = 4;
    std::printf("ReconfigureDecoder (4 surfaces): %d\n", cuvidReconfigureDecoder(d, &ri));
    CUVIDRECONFIGUREDECODERINFO zero{};
    std::printf("ReconfigureDecoder (zeros): %d\n", cuvidReconfigureDecoder(d, &zero));
    std::printf("Destroy: %d\n", cuvidDestroyDecoder(d));
  }

  // ---- context locks -------------------------------------------------------------------
  {
    CUvideoctxlock lk = nullptr;
    std::printf("CtxLockCreate: %d\n", cuvidCtxLockCreate(&lk, ctx));
    {
      const CUresult a = cuvidCtxLock(lk, 0), b = cuvidCtxUnlock(lk, 0), c = cuvidCtxUnlock(lk, 0);
      std::printf("CtxLock: %d, CtxUnlock: %d, CtxUnlock (unlocked): %d\n", a, b, c);
    }
    std::printf("CtxLockDestroy: %d\n", cuvidCtxLockDestroy(lk));
    {
      const CUresult a = cuvidCtxLock(nullptr, 0), b = cuvidCtxUnlock(nullptr, 0), c = cuvidCtxLockDestroy(nullptr);
      std::printf("CtxLock (null): %d, CtxUnlock (null): %d, CtxLockDestroy (null): %d\n", a, b, c);
    }
    CUvideoctxlock lk2 = nullptr;
    std::printf("CtxLockCreate (no context): %d\n", cuvidCtxLockCreate(&lk2, nullptr));
    if (lk2) cuvidCtxLockDestroy(lk2);
  }

  // ---- the parser ---------------------------------------------------------------------------
  {
    CUvideoparser vp = nullptr;
    CUVIDPARSERPARAMS pp = parser_params();
    std::printf("CreateVideoParser (null params): %d\n", cuvidCreateVideoParser(&vp, nullptr));
    poke(pp.CodecType, 99);
    std::printf("CreateVideoParser (codec 99): %d\n", cuvidCreateVideoParser(&vp, &pp));
    pp = parser_params();
    pp.ulMaxNumDecodeSurfaces = 0;
    std::printf("CreateVideoParser (0 surfaces): %d\n", cuvidCreateVideoParser(&vp, &pp));
    if (vp) cuvidDestroyVideoParser(vp);
    vp = nullptr;
    pp = parser_params();
    pp.pfnSequenceCallback = nullptr;
    pp.pfnDecodePicture = nullptr;
    pp.pfnDisplayPicture = nullptr;
    std::printf("CreateVideoParser (no callbacks): %d\n", cuvidCreateVideoParser(&vp, &pp));
    CUVIDSOURCEDATAPACKET pk{};
    pk.payload = b420.data();
    pk.payload_size = b420.size();
    std::printf("ParseVideoData (no callbacks): %d\n", cuvidParseVideoData(vp, &pk));
    CUVIDSOURCEDATAPACKET empty{};
    std::printf("ParseVideoData (empty packet): %d\n", cuvidParseVideoData(vp, &empty));
    static unsigned char garbage[200];
    for (int i = 0; i < 200; ++i) garbage[i] = static_cast<unsigned char>(i * 7);
    pk.payload = garbage;
    pk.payload_size = sizeof garbage;
    std::printf("ParseVideoData (garbage): %d\n", cuvidParseVideoData(vp, &pk));
    CUVIDSOURCEDATAPACKET eos{};
    eos.flags = CUVID_PKT_ENDOFSTREAM;
    std::printf("ParseVideoData (end of stream): %d\n", cuvidParseVideoData(vp, &eos));
    std::printf("DestroyVideoParser: %d\n", cuvidDestroyVideoParser(vp));
    std::printf("ParseVideoData (null parser): %d\n", cuvidParseVideoData(nullptr, &eos));
    std::printf("DestroyVideoParser (null): %d\n", cuvidDestroyVideoParser(nullptr));
  }

  // Timestamps, packet splits and callback return values, with 61x45 pictures.
  {
    const auto two_pictures = [&] {
      std::vector<uint8_t> v = b420;
      v.insert(v.end(), b444.begin(), b444.end());
      return v;
    }();
    std::printf("== two pictures in one packet, timestamp 1000\n");
    Run* r = decode(two_pictures);
    finish(r);
    std::printf("== a picture split over two packets, timestamp on the first\n");
    {
      r = new Run();
      run = r;
      CUvideoparser vp = nullptr;
      CUVIDPARSERPARAMS pp = parser_params();
      cuvidCreateVideoParser(&vp, &pp);
      CUVIDSOURCEDATAPACKET pk{};
      pk.payload = b420.data();
      pk.payload_size = b420.size() / 2;
      pk.flags = CUVID_PKT_TIMESTAMP;
      pk.timestamp = 7000;
      CUresult pr1 = cuvidParseVideoData(vp, &pk);
      std::printf("  first half: %d, %d frames\n", pr1, r->frames);
      pk.payload = b420.data() + b420.size() / 2;
      pk.payload_size = b420.size() - b420.size() / 2;
      pk.flags = 0;
      pr1 = cuvidParseVideoData(vp, &pk);
      std::printf("  second half: %d, %d frames\n", pr1, r->frames);
      cuvidDestroyVideoParser(vp);
      finish(r);
    }
    std::printf("== timestamps over several packets, then none\n");
    {
      r = new Run();
      run = r;
      CUvideoparser vp = nullptr;
      CUVIDPARSERPARAMS pp = parser_params();
      cuvidCreateVideoParser(&vp, &pp);
      CUVIDSOURCEDATAPACKET pk{};
      pk.payload = b420.data();
      pk.payload_size = b420.size();
      pk.flags = CUVID_PKT_TIMESTAMP;
      pk.timestamp = 5000;
      cuvidParseVideoData(vp, &pk);
      pk.timestamp = 9000;
      cuvidParseVideoData(vp, &pk);
      pk.flags = 0;
      cuvidParseVideoData(vp, &pk);
      cuvidDestroyVideoParser(vp);
      finish(r);
    }
    std::printf("== no timestamp flag, and the default clock\n");
    {
      r = new Run();
      run = r;
      CUvideoparser vp = nullptr;
      CUVIDPARSERPARAMS pp = parser_params(0);
      cuvidCreateVideoParser(&vp, &pp);
      CUVIDSOURCEDATAPACKET pk{};
      pk.payload = two_pictures.data();
      pk.payload_size = two_pictures.size();
      cuvidParseVideoData(vp, &pk);
      cuvidDestroyVideoParser(vp);
      finish(r);
    }
    std::printf("== sequence callback returns 2 (picture indices cycle)\n");
    {
      r = new Run();
      r->seq_ret = 2;
      run = r;
      CUvideoparser vp = nullptr;
      CUVIDPARSERPARAMS pp = parser_params();
      cuvidCreateVideoParser(&vp, &pp);
      CUVIDSOURCEDATAPACKET pk{};
      pk.payload = two_pictures.data();
      pk.payload_size = two_pictures.size();
      cuvidParseVideoData(vp, &pk);
      cuvidParseVideoData(vp, &pk);
      cuvidDestroyVideoParser(vp);
      finish(r);
    }
    std::printf("== decode callback returns 0 (no display)\n");
    {
      r = new Run();
      r->dec_ret = 0;
      run = r;
      CUvideoparser vp = nullptr;
      CUVIDPARSERPARAMS pp = parser_params();
      cuvidCreateVideoParser(&vp, &pp);
      CUVIDSOURCEDATAPACKET pk{};
      pk.payload = two_pictures.data();
      pk.payload_size = two_pictures.size();
      std::printf("  parse: %d\n", cuvidParseVideoData(vp, &pk));
      cuvidDestroyVideoParser(vp);
      finish(r);
    }
  }

  // ---- pixels ---------------------------------------------------------------------------------
  const char* fixtures[] = {"b420", "b422", "b444", "p420", "p444", "r420", "gray", "pgray", "big_q5", "big_pr"};
  for (const char* f : fixtures) {
    const std::vector<uint8_t> j = slurp(jpeg_dir + "/" + f + ".jpg");
    std::printf("== %s.jpg\n", f);
    Run* r = decode(j);
    pixels(f, r);
    finish(r);
  }
  std::printf("== the CMYK fixture (not decodable by NVDEC)\n");
  {
    const std::vector<uint8_t> j = slurp(jpeg_dir + "/cmyk2.jpg");
    Run* r = decode(j);
    std::printf("  frames displayed: %d\n", r->frames);
    finish(r);
  }
  // Decoder and target sizes against the picture.
  struct Resize {
    const char* name;
    const char* file;
    int dw, dh, tw, th;
  } resizes[] = {{"big_q5_up_400x300", "big_q5", 0, 0, 400, 300},     {"big_q5_half_168x128", "big_q5", 0, 0, 168, 128},
                 {"big_q5_cut_200x150", "big_q5", 200, 150, 0, 0},     {"b420_into_128x96", "b420", 128, 96, 0, 0},
                 {"b420_crop_62x46", "b420", 0, 0, 62, 46},            {"b420_wide_80x48", "b420", 0, 0, 80, 48},
                 {"big_q5_tall_336x300", "big_q5", 0, 0, 336, 300}};
  for (const Resize& rz : resizes) {
    const std::vector<uint8_t> j = slurp(jpeg_dir + "/" + rz.file + ".jpg");
    std::printf("== %s\n", rz.name);
    Run* r = decode(j, rz.dw, rz.dh, rz.tw, rz.th, false);
    std::printf("  frames displayed: %d, pitch %u\n", r->frames, r->pitches.empty() ? 0u : r->pitches[0]);
    pixels(rz.name, r);
    finish(r);
  }
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
