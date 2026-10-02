// nvJPEG's decode and encode paths, as torchvision, DALI and OpenCV's CUDA
// codecs call them, against what NVIDIA's nvJPEG 13.0 does on an RTX 3060:
//
//   - every fixture (nvidia/tests/data/jpeg: baseline, progressive, restart
//     markers, grey, CMYK, 4:4:4/4:2:2/4:2:0) decoded by nvjpegDecode to every
//     output format, compared to the card's output by checksum -- exactly;
//   - the batched API (nvjpegDecodeBatchedInitialize / nvjpegDecodeBatched,
//     torchvision.io.decode_jpeg's path) and the decoupled three-phase one
//     (nvjpegDecodeJpegHost / TransferToDevice / Device, nvjpegDecodeJpeg),
//     each giving the same pixels as nvjpegDecode, and a region of interest
//     the same pixels as the full decode;
//   - the stream queries, the argument checks and refusals the card makes;
//   - encoding: every subsampling, baseline and progressive, standard and
//     optimised Huffman tables, from RGB and from YUV planes, the encoder's
//     buffer bound, and quantisation tables copied from a parsed image -- each
//     bitstream decoded again and compared with its source.
//
//   nvjpeg_paths [fixture directory [--print]]
//
// --print writes the checksum table this file carries (run against NVIDIA's
// library to regenerate it). Every check passes against NVIDIA's libnvjpeg 13.0
// on the card and against VirtualGPU's.
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const std::string& what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}
static void is(long long got, long long want, const std::string& what) {
  check(got == want, what);
  if (got != want) std::printf("     got %lld, want %lld\n", got, want);
}

// Output formats CUDA 12.0's header lacks, by value.
constexpr int kNV12 = 8, kYUY2 = 9;

static std::string dir;
static std::vector<unsigned char> slurp(const std::string& name) {
  std::ifstream f(dir + "/" + name, std::ios::binary);
  return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static unsigned long long fnv(const std::vector<unsigned char>& b) {
  unsigned long long x = 1469598103934665603ull;
  for (unsigned char v : b) {
    x ^= v;
    x *= 1099511628211ull;
  }
  return x;
}

// Device planes big enough for any output of a 333x251 image.
struct Planes {
  nvjpegImage_t im{};
  explicit Planes(size_t bytes = 400 * 300 * 3) {
    for (int c = 0; c < 4; ++c) {
      cudaMalloc(&im.channel[c], bytes);
      cudaMemset(im.channel[c], 0, bytes);
      im.pitch[c] = 400 * 3;
    }
  }
  ~Planes() {
    for (int c = 0; c < 4; ++c) cudaFree(im.channel[c]);
  }
  std::vector<unsigned char> read(int c, int w, int h) const {
    std::vector<unsigned char> b(static_cast<size_t>(w) * h);
    cudaMemcpy2D(b.data(), w, im.channel[c], im.pitch[c], w, h, cudaMemcpyDeviceToHost);
    return b;
  }
};

struct Info {
  int nc = 0;
  nvjpegChromaSubsampling_t css = NVJPEG_CSS_UNKNOWN;
  int w[4] = {0}, h[4] = {0};
};

// What a format writes, as bytes per plane, concatenated: the checksum's input.
static std::vector<unsigned char> output_bytes(const Planes& p, const Info& in, int fmt) {
  std::vector<unsigned char> all;
  const auto add = [&](int c, int w, int h) {
    const std::vector<unsigned char> b = p.read(c, w, h);
    all.insert(all.end(), b.begin(), b.end());
  };
  switch (fmt) {
    case NVJPEG_OUTPUT_UNCHANGED:
    case NVJPEG_OUTPUT_YUV:
      for (int c = 0; c < in.nc; ++c) add(c, in.w[c], in.h[c]);
      break;
    case NVJPEG_OUTPUT_Y: add(0, in.w[0], in.h[0]); break;
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_BGR:
      for (int c = 0; c < 3; ++c) add(c, in.w[0], in.h[0]);
      break;
    case NVJPEG_OUTPUT_RGBI:
    case NVJPEG_OUTPUT_BGRI: add(0, in.w[0] * 3, in.h[0]); break;
    case kNV12:
      add(0, in.w[0], in.h[0]);
      add(1, in.w[1] * 2, in.h[1]);
      break;
    case kYUY2: add(0, (in.w[0] + 1) / 2 * 4, in.h[0]); break;
  }
  return all;
}

static const char* kFixtures[] = {"b444.jpg", "b422.jpg", "b420.jpg", "p420.jpg", "p444.jpg", "r420.jpg",
                                  "gray.jpg", "pgray.jpg", "cmyk2.jpg", "big_q5.jpg", "big_pr.jpg"};
static const int kFormats[] = {NVJPEG_OUTPUT_UNCHANGED, NVJPEG_OUTPUT_YUV, NVJPEG_OUTPUT_Y, NVJPEG_OUTPUT_RGB,
                               NVJPEG_OUTPUT_BGR, NVJPEG_OUTPUT_RGBI, NVJPEG_OUTPUT_BGRI, kNV12, kYUY2};

// The card's results: for each fixture and format, the status and the
// checksum of what it wrote (nvJPEG 13.0, RTX 3060).
struct Expect {
  const char* file;
  int fmt;
  int status;
  unsigned long long hash;
};
#include "nvjpeg_paths_expected.inc"

static double psnr(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
  double se = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) se += (a[i] - b[i]) * (a[i] - b[i]);
  se /= a.size();
  return se == 0 ? 99 : 10 * std::log10(255.0 * 255.0 / se);
}

int main(int argc, char** argv) {
  // The fixtures are beside this source unless a directory is given.
  dir = argc > 1 ? argv[1] : std::string(__FILE__).substr(0, std::string(__FILE__).rfind('/')) + "/../../data/jpeg";
  const bool print = argc > 2 && std::strcmp(argv[2], "--print") == 0;
  nvjpegHandle_t h;
  if (nvjpegCreateSimple(&h) != NVJPEG_STATUS_SUCCESS) {
    std::printf("FAIL: nvjpegCreateSimple\n");
    return 1;
  }
  nvjpegJpegState_t st;
  nvjpegJpegStateCreate(h, &st);

  // ---- nvjpegDecode, every fixture, every format, against the card ----------
  if (print) std::printf("static const Expect kExpect[] = {\n");
  for (const char* f : kFixtures) {
    const std::vector<unsigned char> d = slurp(f);
    if (d.empty()) {
      check(false, std::string("fixture ") + f + " is there");
      continue;
    }
    Info in;
    const int r = nvjpegGetImageInfo(h, d.data(), d.size(), &in.nc, &in.css, in.w, in.h);
    for (int fmt : kFormats) {
      Planes p;
      const int s = r ? r : nvjpegDecode(h, st, d.data(), d.size(), static_cast<nvjpegOutputFormat_t>(fmt), &p.im, 0);
      cudaDeviceSynchronize();
      const unsigned long long hash = s ? 0 : fnv(output_bytes(p, in, fmt));
      if (print) {
        std::printf("    {\"%s\", %d, %d, 0x%llxull},\n", f, fmt, s, hash);
        continue;
      }
      bool found = false;
      for (const Expect& e : kExpect)
        if (std::strcmp(e.file, f) == 0 && e.fmt == fmt) {
          found = true;
          check(s == e.status && hash == e.hash,
                std::string("nvjpegDecode ") + f + " format " + std::to_string(fmt) + " as the card");
          if (s != e.status) std::printf("     status %d, want %d\n", s, e.status);
        }
      if (!found) check(false, std::string("an expected result for ") + f + " format " + std::to_string(fmt));
    }
  }
  if (print) {
    std::printf("};\n");
    // A CMYK image to interleaved RGB, which only the decoupled API with CMYK
    // allowed makes.
    nvjpegJpegDecoder_t dec;
    nvjpegJpegState_t ds;
    nvjpegJpegStream_t js;
    nvjpegDecodeParams_t dp;
    nvjpegDecoderCreate(h, NVJPEG_BACKEND_DEFAULT, &dec);
    nvjpegDecoderStateCreate(h, dec, &ds);
    nvjpegJpegStreamCreate(h, &js);
    nvjpegDecodeParamsCreate(h, &dp);
    nvjpegDecodeParamsSetOutputFormat(dp, NVJPEG_OUTPUT_RGBI);
    nvjpegDecodeParamsSetAllowCMYK(dp, 1);
    nvjpegBufferDevice_t db;
    nvjpegBufferDeviceCreate(h, nullptr, &db);
    nvjpegStateAttachDeviceBuffer(ds, db);
    const std::vector<unsigned char> d = slurp("cmyk2.jpg");
    nvjpegJpegStreamParse(h, d.data(), d.size(), 0, 0, js);
    Planes p;
    const int r = nvjpegDecodeJpeg(h, dec, ds, js, &p.im, dp, 0);
    cudaDeviceSynchronize();
    std::printf("static const unsigned long long kCmykRgbi = 0x%llxull;   // status %d\n", fnv(p.read(0, 61 * 3, 45)), r);
    return 0;
  }

  // Baseline, progressive and restart renditions of one picture, same
  // quantisation: the same pixels.
  {
    const auto luma = [&](const char* f) {
      const std::vector<unsigned char> d = slurp(f);
      Planes p;
      nvjpegDecode(h, st, d.data(), d.size(), NVJPEG_OUTPUT_Y, &p.im, 0);
      cudaDeviceSynchronize();
      return fnv(p.read(0, 61, 45));
    };
    check(luma("b420.jpg") == luma("p420.jpg"), "progressive decodes to the baseline rendition's pixels");
    check(luma("b420.jpg") == luma("r420.jpg"), "and so does one with restart markers");
    check(luma("gray.jpg") == luma("pgray.jpg"), "and progressive grey");
  }

  // ---- image info and stream queries ---------------------------------------
  {
    Info in;
    const std::vector<unsigned char> d = slurp("b420.jpg");
    is(nvjpegGetImageInfo(h, d.data(), d.size(), &in.nc, &in.css, in.w, in.h), NVJPEG_STATUS_SUCCESS, "image info");
    check(in.nc == 3 && in.css == NVJPEG_CSS_420 && in.w[0] == 61 && in.h[0] == 45 && in.w[1] == 31 && in.h[1] == 23,
          "4:2:0, 61x45 with 31x23 chroma");
    nvjpegJpegStream_t js;
    nvjpegJpegStreamCreate(h, &js);
    const std::vector<unsigned char> p = slurp("p420.jpg");
    is(nvjpegJpegStreamParse(h, p.data(), p.size(), 0, 0, js), NVJPEG_STATUS_SUCCESS, "a progressive stream parses");
    nvjpegJpegEncoding_t enc;
    unsigned w = 0, ht = 0, n = 0, cw = 0, ch = 0;
    nvjpegChromaSubsampling_t css;
    nvjpegJpegStreamGetJpegEncoding(js, &enc);
    nvjpegJpegStreamGetFrameDimensions(js, &w, &ht);
    nvjpegJpegStreamGetComponentsNum(js, &n);
    nvjpegJpegStreamGetComponentDimensions(js, 2, &cw, &ch);
    nvjpegJpegStreamGetChromaSubsampling(js, &css);
    check(enc == NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN && w == 61 && ht == 45 && n == 3 && cw == 31 && ch == 23 &&
              css == NVJPEG_CSS_420,
          "and reports what it is");
    const std::vector<unsigned char> g = slurp("gray.jpg");
    nvjpegJpegStreamParse(h, g.data(), g.size(), 0, 0, js);
    is(nvjpegJpegStreamGetComponentDimensions(js, 1, &cw, &ch), NVJPEG_STATUS_INVALID_PARAMETER,
       "a grey image has no component 1");
    int sup = -1;
    is(nvjpegDecodeBatchedSupported(h, js, &sup), NVJPEG_STATUS_SUCCESS, "nvjpegDecodeBatchedSupported");
    is(sup, 0, "says supported (0)");
    nvjpegJpegStreamDestroy(js);
  }

  // ---- what the card refuses ------------------------------------------------
  {
    const std::vector<unsigned char> d = slurp("b420.jpg");
    Info in;
    Planes p;
    is(nvjpegGetImageInfo(h, d.data(), 100, &in.nc, &in.css, in.w, in.h), NVJPEG_STATUS_INCOMPLETE_BITSTREAM,
       "a file cut short in its headers is incomplete");
    is(nvjpegDecode(h, st, d.data(), 300, NVJPEG_OUTPUT_RGBI, &p.im, 0), NVJPEG_STATUS_INCOMPLETE_BITSTREAM,
       "and does not decode");
    is(nvjpegDecode(h, st, d.data(), d.size() - 300, NVJPEG_OUTPUT_RGBI, &p.im, 0), NVJPEG_STATUS_SUCCESS,
       "one cut short in its entropy-coded data decodes what is there");
    is(nvjpegDecode(h, st, d.data(), 0, NVJPEG_OUTPUT_RGBI, &p.im, 0), NVJPEG_STATUS_INCOMPLETE_BITSTREAM,
       "no data at all is incomplete");
    is(nvjpegDecode(h, st, nullptr, d.size(), NVJPEG_OUTPUT_RGBI, &p.im, 0), NVJPEG_STATUS_INVALID_PARAMETER,
       "a null image is refused");
    is(nvjpegDecode(h, st, d.data(), d.size(), NVJPEG_OUTPUT_RGBI, nullptr, 0), NVJPEG_STATUS_INVALID_PARAMETER,
       "so is a null destination");
    is(nvjpegDecode(h, st, d.data(), d.size(), static_cast<nvjpegOutputFormat_t>(10), &p.im, 0),
       NVJPEG_STATUS_INVALID_PARAMETER, "and a format past the last");
    unsigned char junk[64] = {0xff, 0xd8, 0xff, 0xe0, 0, 16};
    is(nvjpegGetImageInfo(h, junk, sizeof junk, &in.nc, &in.css, in.w, in.h), NVJPEG_STATUS_JPEG_NOT_SUPPORTED,
       "bytes where a marker must be are not supported");
  }

  // ---- batched ---------------------------------------------------------------
  {
    const char* files[3] = {"b420.jpg", "p444.jpg", "gray.jpg"};
    std::vector<unsigned char> d[3];
    const unsigned char* data[3];
    size_t len[3];
    unsigned long long want[3];
    Planes out[3];
    for (int i = 0; i < 3; ++i) {
      d[i] = slurp(files[i]);
      data[i] = d[i].data();
      len[i] = d[i].size();
      Planes p;
      nvjpegDecode(h, st, data[i], len[i], NVJPEG_OUTPUT_RGBI, &p.im, 0);
      cudaDeviceSynchronize();
      want[i] = fnv(p.read(0, 61 * 3, 45));
    }
    nvjpegJpegState_t bs;
    nvjpegJpegStateCreate(h, &bs);
    is(nvjpegDecodeBatchedInitialize(h, bs, 0, 1, NVJPEG_OUTPUT_RGBI), NVJPEG_STATUS_INVALID_PARAMETER,
       "a batch of none is refused");
    is(nvjpegDecodeBatchedInitialize(h, bs, 3, 0, NVJPEG_OUTPUT_RGBI), NVJPEG_STATUS_INVALID_PARAMETER,
       "so is no CPU thread");
    is(nvjpegDecodeBatchedInitialize(h, bs, 3, 1, static_cast<nvjpegOutputFormat_t>(7)), NVJPEG_STATUS_INVALID_PARAMETER,
       "and the 16-bit format");
    is(nvjpegDecodeBatchedInitialize(h, bs, 3, 1, NVJPEG_OUTPUT_RGBI), NVJPEG_STATUS_SUCCESS, "a batch of three");
    nvjpegImage_t dst[3] = {out[0].im, out[1].im, out[2].im};
    is(nvjpegDecodeBatched(h, bs, data, len, dst, 0), NVJPEG_STATUS_SUCCESS, "decodes");
    cudaDeviceSynchronize();
    bool same = true;
    for (int i = 0; i < 3; ++i) same = same && fnv(out[i].read(0, 61 * 3, 45)) == want[i];
    check(same, "each image as nvjpegDecode decodes it");
    is(nvjpegDecodeBatched(h, bs, nullptr, len, dst, 0), NVJPEG_STATUS_INVALID_PARAMETER, "null data is refused");
    std::vector<unsigned char> cut(d[0].begin(), d[0].begin() + 100);
    const unsigned char* data2[3] = {data[0], cut.data(), data[2]};
    size_t len2[3] = {len[0], cut.size(), len[2]};
    is(nvjpegDecodeBatched(h, bs, data2, len2, dst, 0), NVJPEG_STATUS_INCOMPLETE_BITSTREAM,
       "a truncated image in the batch is incomplete");
    is(nvjpegDecodeBatchedPreAllocate(h, bs, 3, 61, 45, NVJPEG_CSS_420, NVJPEG_OUTPUT_RGBI), NVJPEG_STATUS_SUCCESS,
       "pre-allocation is accepted");
    is(nvjpegDecodeBatchedParseJpegTables(h, bs, data[0], len[0]), NVJPEG_STATUS_JPEG_NOT_SUPPORTED,
       "JPEG tables are the hardware backend's");
    nvjpegDecodeParams_t dps[3];
    for (auto& dp : dps) nvjpegDecodeParamsCreate(h, &dp);
    is(nvjpegDecodeBatchedEx(h, bs, data, len, dst, dps, 0), NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED,
       "and so is nvjpegDecodeBatchedEx");
    for (auto& dp : dps) nvjpegDecodeParamsDestroy(dp);
    nvjpegJpegStateDestroy(bs);
  }

  // ---- decoupled ---------------------------------------------------------------
  {
    nvjpegJpegDecoder_t dec, hw;
    is(nvjpegDecoderCreate(h, NVJPEG_BACKEND_HARDWARE, &hw), NVJPEG_STATUS_ARCH_MISMATCH,
       "no hardware decoder on this GPU");
    is(nvjpegDecoderCreate(h, NVJPEG_BACKEND_DEFAULT, &dec), NVJPEG_STATUS_SUCCESS, "the default decoder");
    nvjpegJpegState_t ds;
    nvjpegDecoderStateCreate(h, dec, &ds);
    nvjpegJpegStream_t js;
    nvjpegJpegStreamCreate(h, &js);
    nvjpegDecodeParams_t dp;
    nvjpegDecodeParamsCreate(h, &dp);
    nvjpegDecodeParamsSetOutputFormat(dp, NVJPEG_OUTPUT_RGBI);
    Planes p;
    is(nvjpegDecodeJpegDevice(h, dec, ds, &p.im, 0), NVJPEG_STATUS_INVALID_PARAMETER,
       "the device phase before the host phase is refused");
    is(nvjpegDecodeJpegTransferToDevice(h, dec, ds, js, 0), NVJPEG_STATUS_INVALID_PARAMETER, "so is the transfer");
    nvjpegBufferPinned_t pb;
    nvjpegBufferDevice_t db;
    nvjpegBufferPinnedCreate(h, nullptr, &pb);
    nvjpegBufferDeviceCreate(h, nullptr, &db);
    nvjpegStateAttachPinnedBuffer(ds, pb);
    {
      // The host phase runs without a device buffer; the transfer does not.
      const std::vector<unsigned char> d = slurp("b420.jpg");
      nvjpegJpegStreamParse(h, d.data(), d.size(), 0, 0, js);
      is(nvjpegDecodeJpegHost(h, dec, ds, dp, js), NVJPEG_STATUS_SUCCESS, "the host phase with no device buffer");
      is(nvjpegDecodeJpegTransferToDevice(h, dec, ds, js, 0), NVJPEG_STATUS_INVALID_PARAMETER,
         "the transfer needs one attached");
      is(nvjpegDecodeJpeg(h, dec, ds, js, &p.im, dp, 0), NVJPEG_STATUS_INVALID_PARAMETER, "and so does nvjpegDecodeJpeg");
    }
    nvjpegStateAttachDeviceBuffer(ds, db);
    for (const char* f : {"b420.jpg", "p444.jpg", "gray.jpg", "big_pr.jpg"}) {
      const std::vector<unsigned char> d = slurp(f);
      Info in;
      nvjpegGetImageInfo(h, d.data(), d.size(), &in.nc, &in.css, in.w, in.h);
      Planes ref;
      nvjpegDecode(h, st, d.data(), d.size(), NVJPEG_OUTPUT_RGBI, &ref.im, 0);
      cudaDeviceSynchronize();
      const unsigned long long want = fnv(ref.read(0, in.w[0] * 3, in.h[0]));
      Planes a, b;
      nvjpegJpegStreamParse(h, d.data(), d.size(), 0, 0, js);
      int sup = -1;
      nvjpegDecoderJpegSupported(dec, js, dp, &sup);
      is(sup, 0, std::string(f) + ": the decoder takes it");
      const int r1 = nvjpegDecodeJpegHost(h, dec, ds, dp, js);
      const int r2 = nvjpegDecodeJpegTransferToDevice(h, dec, ds, js, 0);
      const int r3 = nvjpegDecodeJpegDevice(h, dec, ds, &a.im, 0);
      cudaDeviceSynchronize();
      check(!r1 && !r2 && !r3 && fnv(a.read(0, in.w[0] * 3, in.h[0])) == want,
            std::string(f) + ": three phases give nvjpegDecode's pixels");
      is(nvjpegDecodeJpeg(h, dec, ds, js, &b.im, dp, 0), NVJPEG_STATUS_SUCCESS, std::string(f) + ": nvjpegDecodeJpeg");
      cudaDeviceSynchronize();
      check(fnv(b.read(0, in.w[0] * 3, in.h[0])) == want, "with the same pixels");
    }
    // A region: the full decode's pixels; out of the image, refused.
    {
      const std::vector<unsigned char> d = slurp("big_q5.jpg");
      nvjpegJpegStreamParse(h, d.data(), d.size(), 0, 0, js);
      Planes full, roi;
      nvjpegDecode(h, st, d.data(), d.size(), NVJPEG_OUTPUT_RGBI, &full.im, 0);
      is(nvjpegDecodeParamsSetROI(dp, -1, 0, 10, 10), NVJPEG_STATUS_INVALID_PARAMETER, "a negative region is refused");
      is(nvjpegDecodeParamsSetROI(dp, 17, 9, 100, 50), NVJPEG_STATUS_SUCCESS, "a region");
      is(nvjpegDecodeJpeg(h, dec, ds, js, &roi.im, dp, 0), NVJPEG_STATUS_SUCCESS, "decodes");
      cudaDeviceSynchronize();
      const std::vector<unsigned char> f = full.read(0, 333 * 3, 251), r = roi.read(0, 300, 50);
      bool same = true;
      for (int y = 0; y < 50; ++y)
        same = same && std::memcmp(&r[static_cast<size_t>(y) * 300], &f[static_cast<size_t>(y + 9) * 333 * 3 + 17 * 3], 300) == 0;
      check(same, "to the full decode's pixels");
      nvjpegDecodeParamsSetROI(dp, 300, 9, 100, 50);
      is(nvjpegDecodeJpegHost(h, dec, ds, dp, js), NVJPEG_STATUS_BAD_JPEG, "a region past the image is refused");
      nvjpegDecodeParamsSetROI(dp, 0, 0, 0, 0);
    }
    // CMYK only with CMYK allowed. (Fresh parameters: on the card, clearing
    // a region with a zero size left the next device phase failing.)
    nvjpegDecodeParamsDestroy(dp);
    nvjpegDecodeParamsCreate(h, &dp);
    nvjpegDecodeParamsSetOutputFormat(dp, NVJPEG_OUTPUT_RGBI);
    {
      const std::vector<unsigned char> d = slurp("cmyk2.jpg");
      nvjpegJpegStreamParse(h, d.data(), d.size(), 0, 0, js);
      is(nvjpegDecodeJpegHost(h, dec, ds, dp, js), NVJPEG_STATUS_INVALID_PARAMETER, "CMYK to RGB needs CMYK allowed");
      nvjpegDecodeParamsSetAllowCMYK(dp, 1);
      Planes a;
      is(nvjpegDecodeJpeg(h, dec, ds, js, &a.im, dp, 0), NVJPEG_STATUS_SUCCESS, "and then decodes");
      cudaDeviceSynchronize();
      is(static_cast<long long>(fnv(a.read(0, 61 * 3, 45))), static_cast<long long>(kCmykRgbi),
         "to the card's RGB");
      nvjpegDecodeParamsSetAllowCMYK(dp, 0);
    }
    nvjpegDecodeParamsDestroy(dp);
    nvjpegJpegStreamDestroy(js);
    nvjpegJpegStateDestroy(ds);
    nvjpegBufferPinnedDestroy(pb);
    nvjpegBufferDeviceDestroy(db);
    nvjpegDecoderDestroy(dec);
  }

  // ---- encode --------------------------------------------------------------------
  {
    nvjpegEncoderState_t es;
    nvjpegEncoderParams_t ep;
    nvjpegEncoderStateCreate(h, &es, 0);
    nvjpegEncoderParamsCreate(h, &ep, 0);
    is(nvjpegEncoderParamsSetQuality(ep, 0, 0), NVJPEG_STATUS_INVALID_PARAMETER, "quality 0 is refused");
    is(nvjpegEncoderParamsSetQuality(ep, 101, 0), NVJPEG_STATUS_INVALID_PARAMETER, "so is 101");
    is(nvjpegEncoderParamsSetEncoding(ep, NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN, 0),
       NVJPEG_STATUS_INVALID_PARAMETER, "extended sequential encoding is refused");
    is(nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_410V, 0), NVJPEG_STATUS_JPEG_NOT_SUPPORTED,
       "4:1:0 vertical is not supported");
#if NVJPEG_VER_MAJOR >= 13   // CUDA 12.0's header has no restart interval
    is(nvjpegEncoderParamsSetRestartInterval(ep, 65536, 0), NVJPEG_STATUS_INVALID_PARAMETER,
       "a restart interval past 65535 is refused");
#endif
    // NVIDIA's buffer bound, which a caller allocates from.
    const struct {
      nvjpegChromaSubsampling_t css;
      int w, h;
      size_t want;
    } bounds[] = {{NVJPEG_CSS_444, 61, 45, 20480},    {NVJPEG_CSS_420, 61, 45, 11264},
                  {NVJPEG_CSS_411, 61, 45, 10240},    {NVJPEG_CSS_GRAY, 16, 12, 2560},
                  {NVJPEG_CSS_440, 640, 480, 1845248}, {NVJPEG_CSS_422, 1920, 1440, 8296448}};
    for (const auto& b : bounds) {
      nvjpegEncoderParamsSetSamplingFactors(ep, b.css, 0);
      size_t n = 0;
      nvjpegEncodeGetBufferSize(h, ep, b.w, b.h, &n);
      is(static_cast<long long>(n), static_cast<long long>(b.want),
         "buffer bound, css " + std::to_string(b.css) + " " + std::to_string(b.w) + "x" + std::to_string(b.h));
    }
    // A source picture.
    const int W = 61, H = 45;
    std::vector<unsigned char> rgb(W * H * 3);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        rgb[(y * W + x) * 3 + 0] = static_cast<unsigned char>(128 + 100 * std::sin(x / 6.0));
        rgb[(y * W + x) * 3 + 1] = static_cast<unsigned char>(128 + 100 * std::cos(y / 7.0));
        rgb[(y * W + x) * 3 + 2] = static_cast<unsigned char>((x * 3 + y * 2) & 255);
      }
    nvjpegImage_t src{};
    cudaMalloc(&src.channel[0], rgb.size());
    src.pitch[0] = W * 3;
    cudaMemcpy(src.channel[0], rgb.data(), rgb.size(), cudaMemcpyHostToDevice);
    const auto retrieve = [&]() {
      size_t n = 0;
      nvjpegEncodeRetrieveBitstream(h, es, nullptr, &n, 0);
      std::vector<unsigned char> out(n);
      nvjpegEncodeRetrieveBitstream(h, es, out.data(), &n, 0);
      return out;
    };
    const auto decode_rgbi = [&](const std::vector<unsigned char>& jpg, int* status) {
      Planes p;
      *status = nvjpegDecode(h, st, jpg.data(), jpg.size(), NVJPEG_OUTPUT_RGBI, &p.im, 0);
      cudaDeviceSynchronize();
      return p.read(0, W * 3, H);
    };
    // The frame header's sampling factors and the SOF marker, read back.
    const auto frame = [](const std::vector<unsigned char>& j, int* sof, int* ncomp, int* hv) {
      *sof = 0;
      for (size_t i = 2; i + 10 < j.size(); ++i)
        if (j[i] == 0xFF && (j[i + 1] == 0xC0 || j[i + 1] == 0xC2)) {
          *sof = j[i + 1];
          *ncomp = j[i + 9];
          *hv = j[i + 11];
          return;
        }
    };
    const nvjpegChromaSubsampling_t all[] = {NVJPEG_CSS_444, NVJPEG_CSS_422, NVJPEG_CSS_420, NVJPEG_CSS_440,
                                             NVJPEG_CSS_411, NVJPEG_CSS_410, NVJPEG_CSS_GRAY};
    const int luma_hv[] = {0x11, 0x21, 0x22, 0x12, 0x41, 0x42, 0x11};
    for (int enc : {0xC0, 0xC2})
      for (int opt : {0, 1})
        for (int k = 0; k < 7; ++k) {
          nvjpegEncoderParamsSetQuality(ep, 90, 0);
          nvjpegEncoderParamsSetEncoding(ep, static_cast<nvjpegJpegEncoding_t>(enc), 0);
          nvjpegEncoderParamsSetOptimizedHuffman(ep, opt, 0);
          nvjpegEncoderParamsSetSamplingFactors(ep, all[k], 0);
          const std::string tag = std::string(enc == 0xC2 ? "progressive" : "baseline") + (opt ? " optimised" : "") +
                                  " css " + std::to_string(all[k]);
          if (nvjpegEncodeImage(h, es, ep, &src, NVJPEG_INPUT_RGBI, W, H, 0) != NVJPEG_STATUS_SUCCESS) {
            check(false, tag + ": encodes");
            continue;
          }
          const std::vector<unsigned char> jpg = retrieve();
          int sof = 0, nc = 0, hv = 0, s = -1;
          frame(jpg, &sof, &nc, &hv);
          check(sof == enc && nc == (all[k] == NVJPEG_CSS_GRAY ? 1 : 3) && hv == luma_hv[k],
                tag + ": the frame says what was asked");
          const std::vector<unsigned char> back = decode_rgbi(jpg, &s);
          if (all[k] == NVJPEG_CSS_GRAY) {
            check(s == 0, tag + ": decodes");
          } else {
            const double q = psnr(rgb, back);
            check(s == 0 && q > (all[k] == NVJPEG_CSS_444 ? 30.0 : 22.0), tag + ": decodes to the source");
            if (s != 0 || q <= 22.0) std::printf("     status %d, PSNR %.1f dB\n", s, q);
          }
        }
    // Optimised tables are smaller.
    nvjpegEncoderParamsSetEncoding(ep, NVJPEG_ENCODING_BASELINE_DCT, 0);
    nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_420, 0);
    nvjpegEncoderParamsSetOptimizedHuffman(ep, 0, 0);
    nvjpegEncodeImage(h, es, ep, &src, NVJPEG_INPUT_RGBI, W, H, 0);
    const size_t standard = retrieve().size();
    nvjpegEncoderParamsSetOptimizedHuffman(ep, 1, 0);
    nvjpegEncodeImage(h, es, ep, &src, NVJPEG_INPUT_RGBI, W, H, 0);
    check(retrieve().size() < standard, "optimised Huffman tables make a smaller bitstream");
    nvjpegEncoderParamsSetOptimizedHuffman(ep, 0, 0);
    // Into device memory, as torchvision retrieves it -- straight after the
    // encode: on the card a host retrieval takes the bitstream, and a device
    // retrieval after it succeeds and writes nothing (checked below).
    nvjpegEncodeImage(h, es, ep, &src, NVJPEG_INPUT_RGBI, W, H, 0);
    {
      size_t n = 0;
      is(nvjpegEncodeRetrieveBitstreamDevice(h, es, nullptr, &n, 0), NVJPEG_STATUS_SUCCESS, "the device bitstream's size");
      unsigned char* dev;
      cudaMalloc(&dev, n);
      is(nvjpegEncodeRetrieveBitstreamDevice(h, es, dev, &n, 0), NVJPEG_STATUS_SUCCESS, "and the bitstream");
      cudaDeviceSynchronize();
      std::vector<unsigned char> host(n);
      cudaMemcpy(host.data(), dev, n, cudaMemcpyDeviceToHost);
      check(host.size() > 4 && host[0] == 0xFF && host[1] == 0xD8 && host[n - 2] == 0xFF && host[n - 1] == 0xD9,
            "a whole JPEG in device memory");
      size_t small = 10;
      is(nvjpegEncodeRetrieveBitstream(h, es, host.data(), &small, 0), NVJPEG_STATUS_INVALID_PARAMETER,
         "a buffer too small is refused");
      retrieve();
      cudaMemset(dev, 0xAB, n);
      is(nvjpegEncodeRetrieveBitstreamDevice(h, es, dev, &n, 0), NVJPEG_STATUS_SUCCESS,
         "a device retrieval after a host one succeeds");
      cudaMemcpy(host.data(), dev, n, cudaMemcpyDeviceToHost);
      check(host[0] == 0xAB && host[n - 1] == 0xAB, "and writes nothing, as on the card");
      cudaFree(dev);
    }
    // From YUV planes: 4:2:0 in, 4:2:0 out; grey in, colour out refused.
    {
      std::vector<unsigned char> y(W * H), u(31 * 23), v(31 * 23);
      for (int i = 0; i < W * H; ++i) y[i] = static_cast<unsigned char>(i % 251);
      for (size_t i = 0; i < u.size(); ++i) {
        u[i] = 100;
        v[i] = 160;
      }
      nvjpegImage_t yuv{};
      const std::vector<unsigned char>* planes[3] = {&y, &u, &v};
      const int widths[3] = {W, 31, 31}, heights[3] = {H, 23, 23};
      for (int c = 0; c < 3; ++c) {
        cudaMalloc(&yuv.channel[c], planes[c]->size());
        yuv.pitch[c] = widths[c];
        cudaMemcpy(yuv.channel[c], planes[c]->data(), planes[c]->size(), cudaMemcpyHostToDevice);
      }
      nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_420, 0);
      is(nvjpegEncodeYUV(h, es, ep, &yuv, NVJPEG_CSS_420, W, H, 0), NVJPEG_STATUS_SUCCESS, "4:2:0 planes encode");
      const std::vector<unsigned char> jpg = retrieve();
      Planes p;
      is(nvjpegDecode(h, st, jpg.data(), jpg.size(), NVJPEG_OUTPUT_YUV, &p.im, 0), NVJPEG_STATUS_SUCCESS, "and decode");
      cudaDeviceSynchronize();
      check(psnr(y, p.read(0, W, H)) > 30 && psnr(u, p.read(1, 31, 23)) > 30 && psnr(v, p.read(2, 31, 23)) > 30,
            "to the planes");
      (void)heights;
      is(nvjpegEncodeYUV(h, es, ep, &yuv, NVJPEG_CSS_GRAY, W, H, 0), NVJPEG_STATUS_INVALID_PARAMETER,
         "grey planes to a colour image are refused");
      for (int c = 0; c < 3; ++c) cudaFree(yuv.channel[c]);
    }
    // The quantisation of a parsed image, copied: its tables in the output.
    {
      const std::vector<unsigned char> d = slurp("big_q5.jpg");
      nvjpegJpegStream_t js;
      nvjpegJpegStreamCreate(h, &js);
      nvjpegJpegStreamParse(h, d.data(), d.size(), 0, 0, js);
      is(nvjpegEncoderParamsCopyQuantizationTables(ep, js, 0), NVJPEG_STATUS_SUCCESS, "quantisation tables copied");
      nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_420, 0);
      nvjpegEncodeImage(h, es, ep, &src, NVJPEG_INPUT_RGBI, W, H, 0);
      const std::vector<unsigned char> jpg = retrieve();
      // The first DQT's 64 values against the source's.
      const auto dqt = [](const std::vector<unsigned char>& j) {
        for (size_t i = 2; i + 69 < j.size(); ++i)
          if (j[i] == 0xFF && j[i + 1] == 0xDB) return std::vector<unsigned char>(j.begin() + i + 5, j.begin() + i + 69);
        return std::vector<unsigned char>();
      };
      check(!dqt(jpg).empty() && dqt(jpg) == dqt(d), "the bitstream carries the source's luma table");
      nvjpegJpegStreamDestroy(js);
    }
    cudaFree(src.channel[0]);
    nvjpegEncoderParamsDestroy(ep);
    nvjpegEncoderStateDestroy(es);
  }

  nvjpegJpegStateDestroy(st);
  nvjpegDestroy(h);
  std::printf(failures ? "FAIL: %d checks\n" : "PASS\n", failures);
  return failures != 0;
}
