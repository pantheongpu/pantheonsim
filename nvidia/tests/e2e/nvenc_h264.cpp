// NVENC's H.264 and HEVC output, read back by an independent decoder (ffmpeg).
//
// Frames of known content are encoded through the NVENC API -- NV12, YV12, IYUV
// and the 32-bit RGB formats, at sizes that are and are not multiples of 16 and
// not even -- and the stream is decoded with ffmpeg. VirtualGPU's encoder writes
// lossless PCM coding units (macroblocks, or 16x16 coding tree blocks in HEVC),
// so the decoded samples must equal the input exactly
// (RGB input: the BT.601 limited-range conversion the card's encoder applies, to
// within one level); with --lossy, which is for NVIDIA's real library, the encoder
// is lossy and the check is a peak signal-to-noise ratio of at least 30 dB.
//
// SKIP (exit 0) when libnvidia-encode.so.1, ffmpeg or ffprobe is missing.
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <nvEncodeAPI.h>

static NV_ENCODE_API_FUNCTION_LIST f{};

namespace {

struct Planes {
  int w = 0, h = 0;                 // luma size
  std::vector<uint8_t> y, u, v;     // u, v: ((w+1)/2) x ((h+1)/2)
  int cw() const { return (w + 1) / 2; }
  int ch() const { return (h + 1) / 2; }
};

int tri(int v) {
  v &= 255;
  return v < 128 ? v : 255 - v;
}

// Smooth test content that moves from frame to frame.
Planes make_planes(int w, int h, int t) {
  Planes p;
  p.w = w;
  p.h = h;
  p.y.resize(static_cast<size_t>(w) * h);
  p.u.resize(static_cast<size_t>(p.cw()) * p.ch());
  p.v.resize(p.u.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) p.y[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(40 + tri(x + 2 * y + 11 * t) * 3 / 4);
  for (int y = 0; y < p.ch(); ++y)
    for (int x = 0; x < p.cw(); ++x) {
      p.u[static_cast<size_t>(y) * p.cw() + x] = static_cast<uint8_t>(80 + tri(3 * x + y + 7 * t) / 2);
      p.v[static_cast<size_t>(y) * p.cw() + x] = static_cast<uint8_t>(90 + tri(x + 2 * y + 5 * t) / 2);
    }
  return p;
}

// RGB content and the planes the card's encoder makes of it (BT.601 limited range;
// chroma from the 2x2 block's mean colour; positions past the edge repeat).
struct Rgb {
  int w = 0, h = 0;
  std::vector<uint8_t> r, g, b;
};
Rgb make_rgb(int w, int h, int t) {
  Rgb c;
  c.w = w;
  c.h = h;
  c.r.resize(static_cast<size_t>(w) * h);
  c.g = c.r;
  c.b = c.r;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      c.r[i] = static_cast<uint8_t>(30 + tri(x + 11 * t) * 3 / 2);
      c.g[i] = static_cast<uint8_t>(40 + tri(2 * y + 7 * t));
      c.b[i] = static_cast<uint8_t>(200 - tri(x + y + 3 * t));
    }
  return c;
}
Planes rgb_planes(const Rgb& c) {
  Planes p;
  p.w = c.w;
  p.h = c.h;
  p.y.resize(static_cast<size_t>(c.w) * c.h);
  p.u.resize(static_cast<size_t>(p.cw()) * p.ch());
  p.v.resize(p.u.size());
  auto clamp8 = [](double v) { return static_cast<uint8_t>(std::min(255L, std::max(0L, std::lround(v)))); };
  for (int y = 0; y < c.h; ++y)
    for (int x = 0; x < c.w; ++x) {
      const size_t i = static_cast<size_t>(y) * c.w + x;
      p.y[i] = clamp8(16.0 + (65.481 * c.r[i] + 128.553 * c.g[i] + 24.966 * c.b[i]) / 255.0);
    }
  for (int y = 0; y < p.ch(); ++y)
    for (int x = 0; x < p.cw(); ++x) {
      double r = 0, g = 0, b = 0;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
          const size_t i = static_cast<size_t>(std::min(2 * y + dy, c.h - 1)) * c.w + std::min(2 * x + dx, c.w - 1);
          r += c.r[i];
          g += c.g[i];
          b += c.b[i];
        }
      r /= 4;
      g /= 4;
      b /= 4;
      p.u[static_cast<size_t>(y) * p.cw() + x] = clamp8(128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0);
      p.v[static_cast<size_t>(y) * p.cw() + x] = clamp8(128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0);
    }
  return p;
}

std::string run_capture(const std::string& cmd, std::vector<uint8_t>* out) {
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return "popen failed";
  uint8_t buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, p)) > 0) out->insert(out->end(), buf, buf + n);
  const int rc = pclose(p);
  return rc == 0 ? "" : "exit status " + std::to_string(rc);
}

bool tool_exists(const char* name) {
  std::vector<uint8_t> out;
  return run_capture(std::string("command -v ") + name + " >/dev/null 2>&1", &out).empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const bool lossy = argc > 1 && !std::strcmp(argv[1], "--lossy");
  void* lib = dlopen("libnvidia-encode.so.1", RTLD_NOW);
  if (!lib) {
    std::printf("SKIP: no libnvidia-encode.so.1\n");
    return 0;
  }
  if (!tool_exists("ffmpeg") || !tool_exists("ffprobe")) {
    std::printf("SKIP: ffmpeg / ffprobe not found (the decoder that reads the stream back)\n");
    return 0;
  }
  auto create = reinterpret_cast<NVENCSTATUS (*)(NV_ENCODE_API_FUNCTION_LIST*)>(dlsym(lib, "NvEncodeAPICreateInstance"));
  f.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  if (create(&f) != NV_ENC_SUCCESS) {
    std::printf("FAIL: NvEncodeAPICreateInstance\n");
    return 1;
  }
  cudaFree(nullptr);
  CUcontext ctx = nullptr;
  cuCtxGetCurrent(&ctx);

  const std::string tmp = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/vgpu_nvenc_h264_" + std::to_string(getpid());
  const std::string file = tmp + ".es";

  const struct {
    const char* name;
    NV_ENC_BUFFER_FORMAT fmt;
    int w, h;
    bool hevc;
  } cases[] = {
      {"H.264 NV12 192x128", NV_ENC_BUFFER_FORMAT_NV12, 192, 128, false}, {"H.264 NV12 145x49", NV_ENC_BUFFER_FORMAT_NV12, 145, 49, false},
      {"H.264 NV12 257x65", NV_ENC_BUFFER_FORMAT_NV12, 257, 65, false},   {"H.264 YV12 200x100", NV_ENC_BUFFER_FORMAT_YV12, 200, 100, false},
      {"H.264 IYUV 320x180", NV_ENC_BUFFER_FORMAT_IYUV, 320, 180, false}, {"H.264 IYUV 161x51", NV_ENC_BUFFER_FORMAT_IYUV, 161, 51, false},
      {"H.264 ARGB 192x128", NV_ENC_BUFFER_FORMAT_ARGB, 192, 128, false}, {"H.264 ABGR 200x70", NV_ENC_BUFFER_FORMAT_ABGR, 200, 70, false},
      {"HEVC NV12 192x128", NV_ENC_BUFFER_FORMAT_NV12, 192, 128, true},   {"HEVC NV12 145x49", NV_ENC_BUFFER_FORMAT_NV12, 145, 49, true},
      {"HEVC NV12 257x65", NV_ENC_BUFFER_FORMAT_NV12, 257, 65, true},     {"HEVC YV12 200x100", NV_ENC_BUFFER_FORMAT_YV12, 200, 100, true},
      {"HEVC IYUV 320x180", NV_ENC_BUFFER_FORMAT_IYUV, 320, 180, true},   {"HEVC IYUV 161x51", NV_ENC_BUFFER_FORMAT_IYUV, 161, 51, true},
      {"HEVC ARGB 192x128", NV_ENC_BUFFER_FORMAT_ARGB, 192, 128, true},   {"HEVC ABGR 200x70", NV_ENC_BUFFER_FORMAT_ABGR, 200, 70, true}};
  constexpr int kFrames = 3;
  int failures = 0;

  for (const auto& c : cases) {
    const bool rgb = c.fmt == NV_ENC_BUFFER_FORMAT_ARGB || c.fmt == NV_ENC_BUFFER_FORMAT_ABGR;
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op{};
    op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    op.device = ctx;
    op.apiVersion = NVENCAPI_VERSION;
    void* enc = nullptr;
    if (f.nvEncOpenEncodeSessionEx(&op, &enc) != NV_ENC_SUCCESS) {
      std::printf("FAIL %s: could not open a session\n", c.name);
      ++failures;
      continue;
    }
    NV_ENC_PRESET_CONFIG pc{};
    pc.version = NV_ENC_PRESET_CONFIG_VER;
    pc.presetCfg.version = NV_ENC_CONFIG_VER;
    const GUID& codec = c.hevc ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;
    f.nvEncGetEncodePresetConfigEx(enc, codec, NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO_HIGH_QUALITY, &pc);
    pc.presetCfg.gopLength = 30;
    pc.presetCfg.frameIntervalP = 1;   // no B frames: output order is coding order
    NV_ENC_INITIALIZE_PARAMS ip{};
    ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
    ip.encodeGUID = codec;
    ip.presetGUID = NV_ENC_PRESET_P4_GUID;
    ip.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    ip.encodeWidth = ip.darWidth = static_cast<uint32_t>(c.w);
    ip.encodeHeight = ip.darHeight = static_cast<uint32_t>(c.h);
    ip.frameRateNum = 30;
    ip.frameRateDen = 1;
    ip.enablePTD = 1;
    ip.encodeConfig = &pc.presetCfg;
    if (f.nvEncInitializeEncoder(enc, &ip) != NV_ENC_SUCCESS) {
      std::printf("FAIL %s: initialisation refused\n", c.name);
      f.nvEncDestroyEncoder(enc);
      ++failures;
      continue;
    }
    NV_ENC_CREATE_INPUT_BUFFER ib{};
    ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
    ib.width = static_cast<uint32_t>(c.w);
    ib.height = static_cast<uint32_t>(c.h);
    ib.bufferFmt = c.fmt;
    NV_ENC_CREATE_BITSTREAM_BUFFER ob{};
    ob.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    if (f.nvEncCreateInputBuffer(enc, &ib) != NV_ENC_SUCCESS || f.nvEncCreateBitstreamBuffer(enc, &ob) != NV_ENC_SUCCESS) {
      std::printf("FAIL %s: buffer creation refused\n", c.name);
      f.nvEncDestroyEncoder(enc);
      ++failures;
      continue;
    }

    std::vector<Planes> expected;
    std::vector<uint8_t> stream;
    bool ok = true;
    for (int t = 0; t < kFrames && ok; ++t) {
      Planes in;
      Rgb col;
      if (rgb) {
        col = make_rgb(c.w, c.h, t);
        in = rgb_planes(col);
      } else {
        in = make_planes(c.w, c.h, t);
      }
      NV_ENC_LOCK_INPUT_BUFFER lk{};
      lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
      lk.inputBuffer = ib.inputBuffer;
      if (f.nvEncLockInputBuffer(enc, &lk) != NV_ENC_SUCCESS) {
        ok = false;
        break;
      }
      auto* base = static_cast<uint8_t*>(lk.bufferDataPtr);
      const int cw = in.cw(), ch = in.ch();
      if (rgb) {
        for (int y = 0; y < c.h; ++y)
          for (int x = 0; x < c.w; ++x) {
            uint8_t* px = base + static_cast<size_t>(y) * lk.pitch + static_cast<size_t>(x) * 4;
            const size_t i = static_cast<size_t>(y) * c.w + x;
            if (c.fmt == NV_ENC_BUFFER_FORMAT_ARGB) {
              px[0] = col.b[i]; px[1] = col.g[i]; px[2] = col.r[i]; px[3] = 255;
            } else {
              px[0] = col.r[i]; px[1] = col.g[i]; px[2] = col.b[i]; px[3] = 255;
            }
          }
      } else {
        for (int y = 0; y < c.h; ++y) std::memcpy(base + static_cast<size_t>(y) * lk.pitch, &in.y[static_cast<size_t>(y) * c.w], c.w);
        uint8_t* chroma = base + static_cast<size_t>(lk.pitch) * c.h;
        if (c.fmt == NV_ENC_BUFFER_FORMAT_NV12) {
          for (int y = 0; y < ch; ++y)
            for (int x = 0; x < cw; ++x) {
              chroma[static_cast<size_t>(y) * lk.pitch + 2 * x] = in.u[static_cast<size_t>(y) * cw + x];
              chroma[static_cast<size_t>(y) * lk.pitch + 2 * x + 1] = in.v[static_cast<size_t>(y) * cw + x];
            }
        } else {
          // YV12: the V plane first; IYUV: U first. Each has half the luma pitch.
          const uint32_t cp = lk.pitch / 2;
          const std::vector<uint8_t>& first = c.fmt == NV_ENC_BUFFER_FORMAT_IYUV ? in.u : in.v;
          const std::vector<uint8_t>& second = c.fmt == NV_ENC_BUFFER_FORMAT_IYUV ? in.v : in.u;
          for (int y = 0; y < ch; ++y) {
            std::memcpy(chroma + static_cast<size_t>(y) * cp, &first[static_cast<size_t>(y) * cw], cw);
            std::memcpy(chroma + static_cast<size_t>(ch) * cp + static_cast<size_t>(y) * cp, &second[static_cast<size_t>(y) * cw], cw);
          }
        }
      }
      f.nvEncUnlockInputBuffer(enc, ib.inputBuffer);
      NV_ENC_PIC_PARAMS pp{};
      pp.version = NV_ENC_PIC_PARAMS_VER;
      pp.inputBuffer = ib.inputBuffer;
      pp.outputBitstream = ob.bitstreamBuffer;
      pp.bufferFmt = c.fmt;
      pp.inputWidth = static_cast<uint32_t>(c.w);
      pp.inputHeight = static_cast<uint32_t>(c.h);
      pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
      pp.inputTimeStamp = static_cast<uint64_t>(t) * 3000;
      if (f.nvEncEncodePicture(enc, &pp) != NV_ENC_SUCCESS) {
        ok = false;
        break;
      }
      NV_ENC_LOCK_BITSTREAM lb{};
      lb.version = NV_ENC_LOCK_BITSTREAM_VER;
      lb.outputBitstream = ob.bitstreamBuffer;
      if (f.nvEncLockBitstream(enc, &lb) != NV_ENC_SUCCESS) {
        ok = false;
        break;
      }
      const auto* bytes = static_cast<const uint8_t*>(lb.bitstreamBufferPtr);
      stream.insert(stream.end(), bytes, bytes + lb.bitstreamSizeInBytes);
      f.nvEncUnlockBitstream(enc, ob.bitstreamBuffer);
      expected.push_back(std::move(in));
    }
    f.nvEncDestroyBitstreamBuffer(enc, ob.bitstreamBuffer);
    f.nvEncDestroyInputBuffer(enc, ib.inputBuffer);
    f.nvEncDestroyEncoder(enc);
    if (!ok) {
      std::printf("FAIL %s: an encode call failed\n", c.name);
      ++failures;
      continue;
    }

    FILE* o = std::fopen(file.c_str(), "wb");
    if (!o) {
      std::printf("FAIL %s: cannot write %s\n", c.name, file.c_str());
      ++failures;
      continue;
    }
    std::fwrite(stream.data(), 1, stream.size(), o);
    std::fclose(o);

    // Dimensions as the decoder reports them, then the frames.
    std::vector<uint8_t> dims, raw;
    run_capture(std::string("ffprobe -v error -f ") + (c.hevc ? "hevc" : "h264") + " -select_streams v:0 -show_entries stream=width,height -of csv=p=0 " + file + " 2>&1", &dims);
    int dw = 0, dh = 0;
    std::sscanf(std::string(dims.begin(), dims.end()).c_str(), "%d,%d", &dw, &dh);
    const std::string err = run_capture(std::string("ffmpeg -v error -f ") + (c.hevc ? "hevc" : "h264") + " -i " + file + " -f rawvideo -pix_fmt yuv420p - 2>/dev/null", &raw);
    std::remove(file.c_str());
    if (!err.empty() || dw <= 0 || dh <= 0) {
      std::printf("FAIL %s: ffmpeg could not decode the stream (%s, %d bytes, dims %dx%d)\n", c.name, err.c_str(), static_cast<int>(stream.size()), dw, dh);
      ++failures;
      continue;
    }
    const size_t frame_bytes = static_cast<size_t>(dw) * dh + 2 * static_cast<size_t>((dw + 1) / 2) * ((dh + 1) / 2);
    if (raw.size() != frame_bytes * kFrames || dw < c.w || dh < c.h || dw > c.w + 1 || dh > c.h + 1) {
      std::printf("FAIL %s: decoded %zu bytes of %dx%d, wanted %d frames covering %dx%d\n", c.name, raw.size(), dw, dh, kFrames, c.w, c.h);
      ++failures;
      continue;
    }
    // Compare the picture region of every frame.
    double worst_psnr = 1e9;
    int worst_diff = 0;
    for (int t = 0; t < kFrames; ++t) {
      const uint8_t* y = &raw[frame_bytes * t];
      const uint8_t* u = y + static_cast<size_t>(dw) * dh;
      const uint8_t* v = u + static_cast<size_t>((dw + 1) / 2) * ((dh + 1) / 2);
      const Planes& e = expected[t];
      double sq = 0;
      size_t n = 0;
      auto compare = [&](const uint8_t* got, int gpitch, const std::vector<uint8_t>& want, int w, int h) {
        for (int yy = 0; yy < h; ++yy)
          for (int xx = 0; xx < w; ++xx) {
            const int d = static_cast<int>(got[static_cast<size_t>(yy) * gpitch + xx]) - want[static_cast<size_t>(yy) * w + xx];
            worst_diff = std::max(worst_diff, std::abs(d));
            sq += static_cast<double>(d) * d;
            ++n;
          }
      };
      compare(y, dw, e.y, e.w, e.h);
      compare(u, (dw + 1) / 2, e.u, e.cw(), e.ch());
      compare(v, (dw + 1) / 2, e.v, e.cw(), e.ch());
      const double mse = sq / static_cast<double>(n);
      worst_psnr = std::min(worst_psnr, mse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse));
    }
    const int tolerance = rgb ? 1 : 0;
    const bool pass = lossy ? worst_psnr >= 30.0 : worst_diff <= tolerance;
    std::printf("%s %s: %d frames decoded as %dx%d, worst sample difference %d, PSNR %.1f dB%s\n", pass ? "PASS" : "FAIL", c.name, kFrames,
                dw, dh, worst_diff, worst_psnr, lossy ? " (lossy)" : "");
    if (!pass) ++failures;
  }
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
