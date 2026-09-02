// Differential conformance for the nvJPEG shim: header parsing, baseline
// decoding of 4:4:4, 4:2:0 and grayscale files, every output layout, and an
// encode/decode round trip.
//
// The JPEG standard does not specify the inverse DCT bit-exactly, so two
// correct decoders differ by about a count per pixel. Header facts -- component
// count, chroma subsampling, per-component dimensions -- are exact and compared
// exactly; pixel data is compared through statistics printed at a precision
// that IDCT rounding cannot move, which is what "the same image" means here.
#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifndef VGPU_DATA_DIR
#define VGPU_DATA_DIR "tests/conformance/data"
#endif

#define NJ(x) do { nvjpegStatus_t s_ = (x); if (s_ != NVJPEG_STATUS_SUCCESS) { \
  printf("%-34s status=%d\n", #x, (int)s_); return; } } while (0)

static std::vector<unsigned char> slurp(const char* name) {
  std::string path = std::string(VGPU_DATA_DIR) + "/" + name;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return {};
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> b(n > 0 ? n : 0);
  if (n > 0 && std::fread(b.data(), 1, b.size(), f) != b.size()) b.clear();
  std::fclose(f);
  return b;
}

static void stats(const char* tag, const std::vector<unsigned char>& v) {
  double sum = 0;
  for (unsigned char x : v) sum += x;
  const double mean = v.empty() ? 0 : sum / v.size();
  double var = 0;
  for (unsigned char x : v) var += (x - mean) * (x - mean);
  var = v.empty() ? 0 : var / v.size();
  // One decimal: a decoder differing by a count on some pixels cannot move
  // these, a decoder that is actually wrong moves them a lot.
  printf("%-34s n=%zu mean=%.1f stddev=%.1f\n", tag, v.size(), mean, std::sqrt(var));
}

static std::vector<unsigned char> fetch(unsigned char* d, size_t pitch, int w, int h) {
  std::vector<unsigned char> host((size_t)w * h);
  for (int y = 0; y < h; ++y)
    cudaMemcpy(host.data() + (size_t)y * w, d + (size_t)y * pitch, w, cudaMemcpyDeviceToHost);
  return host;
}

static void decode_case(nvjpegHandle_t h, nvjpegJpegState_t st, const char* file) {
  auto jpg = slurp(file);
  if (jpg.empty()) { printf("%-34s MISSING\n", file); return; }

  int ncomp = 0;
  nvjpegChromaSubsampling_t css = NVJPEG_CSS_UNKNOWN;
  int widths[NVJPEG_MAX_COMPONENT] = {0}, heights[NVJPEG_MAX_COMPONENT] = {0};
  NJ(nvjpegGetImageInfo(h, jpg.data(), jpg.size(), &ncomp, &css, widths, heights));
  printf("%-34s comps=%d css=%d dims=%dx%d chroma=%dx%d\n", file, ncomp, (int)css, widths[0],
         heights[0], widths[1], heights[1]);

  const int W = widths[0], H = heights[0];
  nvjpegImage_t img{};
  std::vector<unsigned char*> owned;
  for (int c = 0; c < 3; ++c) {
    unsigned char* p = nullptr;
    cudaMalloc(&p, (size_t)W * H * 3);
    img.channel[c] = p;
    img.pitch[c] = W * 3;
    owned.push_back(p);
  }

  NJ(nvjpegDecode(h, st, jpg.data(), jpg.size(), NVJPEG_OUTPUT_Y, &img, nullptr));
  stats((std::string(file) + " Y").c_str(), fetch(img.channel[0], img.pitch[0], W, H));

  NJ(nvjpegDecode(h, st, jpg.data(), jpg.size(), NVJPEG_OUTPUT_RGBI, &img, nullptr));
  stats((std::string(file) + " RGBI").c_str(), fetch(img.channel[0], img.pitch[0], W * 3, H));

  NJ(nvjpegDecode(h, st, jpg.data(), jpg.size(), NVJPEG_OUTPUT_RGB, &img, nullptr));
  stats((std::string(file) + " R").c_str(), fetch(img.channel[0], img.pitch[0], W, H));
  stats((std::string(file) + " G").c_str(), fetch(img.channel[1], img.pitch[1], W, H));
  stats((std::string(file) + " B").c_str(), fetch(img.channel[2], img.pitch[2], W, H));

  NJ(nvjpegDecode(h, st, jpg.data(), jpg.size(), NVJPEG_OUTPUT_BGRI, &img, nullptr));
  stats((std::string(file) + " BGRI").c_str(), fetch(img.channel[0], img.pitch[0], W * 3, H));

  for (auto* p : owned) cudaFree(p);
}

static void encode_case(nvjpegHandle_t h, nvjpegJpegState_t st) {
  const int W = 32, H = 24;
  std::vector<unsigned char> rgb((size_t)W * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      rgb[((size_t)y * W + x) * 3 + 0] = (unsigned char)((x * 7 + y * 3) % 256);
      rgb[((size_t)y * W + x) * 3 + 1] = (unsigned char)((x * 3 + y * 11) % 256);
      rgb[((size_t)y * W + x) * 3 + 2] = (unsigned char)(((x / 4 + y / 4) % 2) ? 220 : 30);
    }
  unsigned char* d = nullptr;
  cudaMalloc(&d, rgb.size());
  cudaMemcpy(d, rgb.data(), rgb.size(), cudaMemcpyHostToDevice);
  nvjpegImage_t src{};
  src.channel[0] = d;
  src.pitch[0] = W * 3;

  nvjpegEncoderState_t est;
  nvjpegEncoderParams_t ep;
  NJ(nvjpegEncoderStateCreate(h, &est, nullptr));
  NJ(nvjpegEncoderParamsCreate(h, &ep, nullptr));
  NJ(nvjpegEncoderParamsSetQuality(ep, 95, nullptr));
  NJ(nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_444, nullptr));
  NJ(nvjpegEncodeImage(h, est, ep, &src, NVJPEG_INPUT_RGBI, W, H, nullptr));
  size_t len = 0;
  NJ(nvjpegEncodeRetrieveBitstream(h, est, nullptr, &len, nullptr));
  std::vector<unsigned char> bits(len);
  NJ(nvjpegEncodeRetrieveBitstream(h, est, bits.data(), &len, nullptr));

  // The bitstream itself is not comparable -- two encoders make different legal
  // choices -- but what it decodes to is, and so is what it says about itself.
  int ncomp = 0;
  nvjpegChromaSubsampling_t css = NVJPEG_CSS_UNKNOWN;
  int widths[NVJPEG_MAX_COMPONENT] = {0}, heights[NVJPEG_MAX_COMPONENT] = {0};
  NJ(nvjpegGetImageInfo(h, bits.data(), len, &ncomp, &css, widths, heights));
  printf("%-34s comps=%d css=%d dims=%dx%d\n", "encoded q95 4:4:4", ncomp, (int)css, widths[0],
         heights[0]);

  nvjpegImage_t back{};
  unsigned char* dback = nullptr;
  cudaMalloc(&dback, (size_t)W * H * 3);
  back.channel[0] = dback;
  back.pitch[0] = W * 3;
  NJ(nvjpegDecode(h, st, bits.data(), len, NVJPEG_OUTPUT_RGBI, &back, nullptr));
  auto out = fetch(dback, back.pitch[0], W * 3, H);
  double err = 0, maxerr = 0;
  for (size_t i = 0; i < out.size() && i < rgb.size(); ++i) {
    const double e = std::fabs((double)out[i] - rgb[i]);
    err += e;
    maxerr = std::fmax(maxerr, e);
  }
  // At quality 95 a round trip is close but not lossless; the thresholds are
  // loose enough that any correct codec passes and a broken one does not.
  printf("%-34s mean error < 6: %s, max error < 40: %s\n", "encode/decode round trip",
         err / out.size() < 6.0 ? "yes" : "no", maxerr < 40.0 ? "yes" : "no");

  cudaFree(d);
  cudaFree(dback);
  nvjpegEncoderParamsDestroy(ep);
  nvjpegEncoderStateDestroy(est);
}

int main() {
  int major = 0;
  nvjpegGetProperty(MAJOR_VERSION, &major);
  printf("nvjpeg major %d\n", major);
  nvjpegHandle_t h;
  if (nvjpegCreateSimple(&h) != NVJPEG_STATUS_SUCCESS) { printf("create failed\n"); return 1; }
  nvjpegJpegState_t st;
  if (nvjpegJpegStateCreate(h, &st) != NVJPEG_STATUS_SUCCESS) { printf("state failed\n"); return 1; }

  decode_case(h, st, "fixture_444.jpg");
  decode_case(h, st, "fixture_420.jpg");
  decode_case(h, st, "fixture_gray.jpg");
  encode_case(h, st);

  nvjpegJpegStateDestroy(st);
  nvjpegDestroy(h);
  return 0;
}
