// NVENC's H.264 stream decoded by NVDEC, and by ffmpeg, which must agree to the sample.
//
// Frames of known content go through the NVENC API (GOP structures, forced IDR and intra
// pictures, constant QP and bit rate control, the three profiles, odd sizes); the stream is
// decoded twice, by ffmpeg and by the cuvid API (libnvcuvid.so.1, the parser's callbacks, the
// decoder, the mapped NV12 surfaces); a conformant H.264 decoder is bit-exact, so the two
// outputs must be identical. The decoded frames must also be close to the input, the picture
// types must follow the GOP settings, and a constant-bit-rate stream must come out near its
// rate.
//
// Run against VirtualGPU's libraries it checks the encoder with the decoder of the same tree;
// run with --card it is NVIDIA's encoder and NVIDIA's decoder against ffmpeg.
//
// SKIP (exit 0) when libnvidia-encode.so.1, libnvcuvid.so.1, ffmpeg or ffprobe is missing.
#include <dlfcn.h>
#include <cuda.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <nvEncodeAPI.h>

#include "cuviddec.h"
#include "nvcuvid.h"

#undef cuvidMapVideoFrame
#undef cuvidUnmapVideoFrame

namespace {

NV_ENCODE_API_FUNCTION_LIST f{};

// ---- the cuvid API, by name -------------------------------------------------------------------------------------

struct Cuvid {
  decltype(&cuvidCreateVideoParser) create_parser;
  decltype(&cuvidParseVideoData) parse;
  decltype(&cuvidDestroyVideoParser) destroy_parser;
  decltype(&cuvidCreateDecoder) create_decoder;
  decltype(&cuvidDecodePicture) decode;
  decltype(&cuvidDestroyDecoder) destroy_decoder;
  decltype(&cuvidMapVideoFrame64) map;
  decltype(&cuvidUnmapVideoFrame64) unmap;
} cv;

// ---- content ----------------------------------------------------------------------------------------------------

int tri(int v) {
  v &= 255;
  return v < 128 ? v : 255 - v;
}
int noise(int x, int y, int t, int amp) {
  uint32_t v = static_cast<uint32_t>(x) * 73856093u ^ static_cast<uint32_t>(y) * 19349663u ^ static_cast<uint32_t>(t) * 83492791u;
  v ^= v >> 13;
  v *= 0x5bd1e995u;
  v ^= v >> 15;
  return static_cast<int>(v % static_cast<uint32_t>(2 * amp + 1)) - amp;
}

struct Planes {
  int w = 0, h = 0;
  std::vector<uint8_t> y, u, v;
  int cw() const { return (w + 1) / 2; }
  int ch() const { return (h + 1) / 2; }
};

// A moving pattern with texture; `still` freezes it after frame 3 (the encoder should skip nearly everything).
Planes make_planes(int w, int h, int t) {
  Planes p;
  p.w = w;
  p.h = h;
  p.y.resize(static_cast<size_t>(w) * h);
  p.u.resize(static_cast<size_t>(p.cw()) * p.ch());
  p.v.resize(p.u.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      p.y[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(40 + tri(x * 2 + y + 3 * t) * 3 / 4 + ((x / 16 + y / 16) & 1) * 20 + noise(x, y, t, 6));
  for (int y = 0; y < p.ch(); ++y)
    for (int x = 0; x < p.cw(); ++x) {
      p.u[static_cast<size_t>(y) * p.cw() + x] = static_cast<uint8_t>(90 + tri(3 * x + y + 2 * t) / 2 + noise(x, y, t + 100, 2));
      p.v[static_cast<size_t>(y) * p.cw() + x] = static_cast<uint8_t>(100 + tri(x + 2 * y + t) / 2 + noise(x, y, t + 200, 2));
    }
  return p;
}

// ---- running a subprocess -----------------------------------------------------------------------------------------

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

// ---- NVDEC ------------------------------------------------------------------------------------------------------

struct Decoded {
  int w = 0, h = 0;
  std::vector<std::vector<uint8_t>> frames;   // yuv420p, planar, w*h + 2*((w+1)/2)*((h+1)/2)
  std::string error;
};
Decoded* g_dec = nullptr;
CUvideodecoder g_decoder = nullptr;
CUVIDEOFORMAT g_fmt{};

int CUDAAPI seq_cb(void*, CUVIDEOFORMAT* fmt) {
  g_fmt = *fmt;
  if (g_decoder) {
    cv.destroy_decoder(g_decoder);
    g_decoder = nullptr;
  }
  CUVIDDECODECREATEINFO ci{};
  ci.ulWidth = fmt->coded_width;
  ci.ulHeight = fmt->coded_height;
  ci.ulNumDecodeSurfaces = fmt->min_num_decode_surfaces;
  ci.CodecType = fmt->codec;
  ci.ChromaFormat = fmt->chroma_format;
  ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  ci.bitDepthMinus8 = fmt->bit_depth_luma_minus8;
  ci.OutputFormat = cudaVideoSurfaceFormat_NV12;
  ci.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
  ci.ulMaxWidth = fmt->coded_width;
  ci.ulMaxHeight = fmt->coded_height;
  ci.ulNumOutputSurfaces = 2;
  // the surface is the display rectangle: the target size is its size (a different target would rescale the picture)
  ci.ulTargetWidth = static_cast<unsigned long>(fmt->display_area.right - fmt->display_area.left);
  ci.ulTargetHeight = static_cast<unsigned long>(fmt->display_area.bottom - fmt->display_area.top);
  ci.display_area = {static_cast<short>(fmt->display_area.left), static_cast<short>(fmt->display_area.top), static_cast<short>(fmt->display_area.right),
                     static_cast<short>(fmt->display_area.bottom)};
  const CUresult r = cv.create_decoder(&g_decoder, &ci);
  if (r) {
    g_dec->error = "cuvidCreateDecoder failed: " + std::to_string(r);
    return 0;
  }
  return fmt->min_num_decode_surfaces;
}
int CUDAAPI dec_cb(void*, CUVIDPICPARAMS* p) {
  const CUresult r = cv.decode(g_decoder, p);
  if (r) g_dec->error = "cuvidDecodePicture failed: " + std::to_string(r);
  return 1;
}
int CUDAAPI disp_cb(void*, CUVIDPARSERDISPINFO* d) {
  unsigned long long dptr = 0;
  unsigned pitch = 0;
  CUVIDPROCPARAMS pp{};
  pp.progressive_frame = d->progressive_frame;
  if (cv.map(g_decoder, d->picture_index, &dptr, &pitch, &pp) != CUDA_SUCCESS) {
    g_dec->error = "cuvidMapVideoFrame failed";
    return 1;
  }
  const unsigned w = static_cast<unsigned>(g_fmt.display_area.right - g_fmt.display_area.left), h = static_cast<unsigned>(g_fmt.display_area.bottom - g_fmt.display_area.top);
  const unsigned x0 = 0, y0 = 0, sh = h;
  std::vector<uint8_t> all(static_cast<size_t>(pitch) * (sh + sh / 2));
  cuMemcpyDtoH(all.data(), static_cast<CUdeviceptr>(dptr), all.size());
  cv.unmap(g_decoder, dptr);
  const unsigned cw = (w + 1) / 2, ch = (h + 1) / 2;
  std::vector<uint8_t> out(static_cast<size_t>(w) * h + 2 * static_cast<size_t>(cw) * ch);
  for (unsigned y = 0; y < h; ++y) std::memcpy(&out[static_cast<size_t>(y) * w], &all[static_cast<size_t>(y0 + y) * pitch + x0], w);
  for (unsigned y = 0; y < ch; ++y)
    for (unsigned x = 0; x < cw; ++x) {
      const uint8_t* uv = &all[static_cast<size_t>(sh + y0 / 2 + y) * pitch + (x0 & ~1u) + 2 * x];
      out[static_cast<size_t>(w) * h + static_cast<size_t>(y) * cw + x] = uv[0];
      out[static_cast<size_t>(w) * h + static_cast<size_t>(cw) * ch + static_cast<size_t>(y) * cw + x] = uv[1];
    }
  g_dec->w = static_cast<int>(w);
  g_dec->h = static_cast<int>(h);
  g_dec->frames.push_back(std::move(out));
  return 1;
}

Decoded nvdec_decode(const std::vector<uint8_t>& stream) {
  Decoded d;
  g_dec = &d;
  g_decoder = nullptr;
  CUvideoparser parser = nullptr;
  CUVIDPARSERPARAMS pp{};
  pp.CodecType = cudaVideoCodec_H264;
  pp.ulMaxNumDecodeSurfaces = 1;
  pp.ulClockRate = 1000;
  pp.ulMaxDisplayDelay = 0;
  pp.pfnSequenceCallback = seq_cb;
  pp.pfnDecodePicture = dec_cb;
  pp.pfnDisplayPicture = disp_cb;
  if (cv.create_parser(&parser, &pp) != CUDA_SUCCESS) {
    d.error = "cuvidCreateVideoParser failed";
    return d;
  }
  CUVIDSOURCEDATAPACKET pk{};
  pk.payload = stream.data();
  pk.payload_size = stream.size();
  pk.flags = CUVID_PKT_TIMESTAMP;
  cv.parse(parser, &pk);
  CUVIDSOURCEDATAPACKET eos{};
  eos.flags = CUVID_PKT_ENDOFSTREAM;
  cv.parse(parser, &eos);
  cv.destroy_parser(parser);
  if (g_decoder) cv.destroy_decoder(g_decoder);
  g_decoder = nullptr;
  g_dec = nullptr;
  return d;
}

Decoded ffmpeg_decode(const std::vector<uint8_t>& stream, const std::string& file) {
  Decoded d;
  FILE* o = std::fopen(file.c_str(), "wb");
  if (!o) {
    d.error = "cannot write " + file;
    return d;
  }
  std::fwrite(stream.data(), 1, stream.size(), o);
  std::fclose(o);
  std::vector<uint8_t> dims, raw;
  run_capture("ffprobe -v error -f h264 -select_streams v:0 -show_entries stream=width,height -of csv=p=0 " + file + " 2>&1", &dims);
  std::sscanf(std::string(dims.begin(), dims.end()).c_str(), "%d,%d", &d.w, &d.h);
  const std::string err = run_capture("ffmpeg -v error -f h264 -i " + file + " -f rawvideo -pix_fmt yuv420p - 2>/dev/null", &raw);
  std::remove(file.c_str());
  if (!err.empty() || d.w <= 0 || d.h <= 0) {
    d.error = "ffmpeg could not decode the stream (" + err + ")";
    return d;
  }
  const size_t fb = static_cast<size_t>(d.w) * d.h + 2 * static_cast<size_t>((d.w + 1) / 2) * ((d.h + 1) / 2);
  for (size_t at = 0; at + fb <= raw.size(); at += fb) d.frames.emplace_back(raw.begin() + static_cast<std::ptrdiff_t>(at), raw.begin() + static_cast<std::ptrdiff_t>(at + fb));
  return d;
}

// ---- the encoder ------------------------------------------------------------------------------------------------

struct Encoded {
  std::vector<uint8_t> stream;
  std::vector<int> types;       // NV_ENC_PIC_TYPE of every picture
  std::vector<size_t> sizes;    // bytes of every picture (parameter sets included)
  std::vector<Planes> input;
  std::string error;
};

struct Case {
  const char* name;
  int w, h, frames;
  std::function<void(NV_ENC_CONFIG&, NV_ENC_INITIALIZE_PARAMS&)> configure;
  std::function<uint32_t(int)> flags;   // NV_ENC_PIC_FLAGs of picture i
  std::function<std::string(const Encoded&, const std::vector<std::vector<uint8_t>>& decoded, int dw, int dh, bool card)> check;   // extra checks; "" = fine
};

Encoded encode(CUcontext ctx, const Case& c) {
  Encoded e;
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op{};
  op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  op.device = ctx;
  op.apiVersion = NVENCAPI_VERSION;
  void* enc = nullptr;
  if (f.nvEncOpenEncodeSessionEx(&op, &enc) != NV_ENC_SUCCESS) {
    e.error = "could not open a session";
    return e;
  }
  NV_ENC_PRESET_CONFIG pc{};
  pc.version = NV_ENC_PRESET_CONFIG_VER;
  pc.presetCfg.version = NV_ENC_CONFIG_VER;
  f.nvEncGetEncodePresetConfigEx(enc, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO_HIGH_QUALITY, &pc);
  pc.presetCfg.gopLength = 30;
  pc.presetCfg.frameIntervalP = 1;   // no B pictures: output order is coding order
  NV_ENC_INITIALIZE_PARAMS ip{};
  ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
  ip.encodeGUID = NV_ENC_CODEC_H264_GUID;
  ip.presetGUID = NV_ENC_PRESET_P4_GUID;
  ip.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
  ip.encodeWidth = ip.darWidth = static_cast<uint32_t>(c.w);
  ip.encodeHeight = ip.darHeight = static_cast<uint32_t>(c.h);
  ip.frameRateNum = 30;
  ip.frameRateDen = 1;
  ip.enablePTD = 1;
  if (c.configure) c.configure(pc.presetCfg, ip);
  ip.encodeConfig = &pc.presetCfg;
  if (f.nvEncInitializeEncoder(enc, &ip) != NV_ENC_SUCCESS) {
    e.error = "initialisation refused";
    f.nvEncDestroyEncoder(enc);
    return e;
  }
  NV_ENC_CREATE_INPUT_BUFFER ib{};
  ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
  ib.width = static_cast<uint32_t>(c.w);
  ib.height = static_cast<uint32_t>(c.h);
  ib.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
  NV_ENC_CREATE_BITSTREAM_BUFFER ob{};
  ob.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
  if (f.nvEncCreateInputBuffer(enc, &ib) != NV_ENC_SUCCESS || f.nvEncCreateBitstreamBuffer(enc, &ob) != NV_ENC_SUCCESS) {
    e.error = "buffer creation refused";
    f.nvEncDestroyEncoder(enc);
    return e;
  }
  for (int t = 0; t < c.frames && e.error.empty(); ++t) {
    Planes in = make_planes(c.w, c.h, t);
    NV_ENC_LOCK_INPUT_BUFFER lk{};
    lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
    lk.inputBuffer = ib.inputBuffer;
    if (f.nvEncLockInputBuffer(enc, &lk) != NV_ENC_SUCCESS) {
      e.error = "lock input";
      break;
    }
    auto* base = static_cast<uint8_t*>(lk.bufferDataPtr);
    for (int y = 0; y < c.h; ++y) std::memcpy(base + static_cast<size_t>(y) * lk.pitch, &in.y[static_cast<size_t>(y) * c.w], static_cast<size_t>(c.w));
    uint8_t* chroma = base + static_cast<size_t>(lk.pitch) * static_cast<size_t>(c.h);
    for (int y = 0; y < in.ch(); ++y)
      for (int x = 0; x < in.cw(); ++x) {
        chroma[static_cast<size_t>(y) * lk.pitch + 2 * x] = in.u[static_cast<size_t>(y) * in.cw() + x];
        chroma[static_cast<size_t>(y) * lk.pitch + 2 * x + 1] = in.v[static_cast<size_t>(y) * in.cw() + x];
      }
    f.nvEncUnlockInputBuffer(enc, ib.inputBuffer);
    NV_ENC_PIC_PARAMS pp{};
    pp.version = NV_ENC_PIC_PARAMS_VER;
    pp.inputBuffer = ib.inputBuffer;
    pp.outputBitstream = ob.bitstreamBuffer;
    pp.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    pp.inputWidth = static_cast<uint32_t>(c.w);
    pp.inputHeight = static_cast<uint32_t>(c.h);
    pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pp.inputTimeStamp = static_cast<uint64_t>(t) * 3000;
    pp.encodePicFlags = c.flags ? c.flags(t) : 0;
    if (f.nvEncEncodePicture(enc, &pp) != NV_ENC_SUCCESS) {
      e.error = "encode call failed";
      break;
    }
    NV_ENC_LOCK_BITSTREAM lb{};
    lb.version = NV_ENC_LOCK_BITSTREAM_VER;
    lb.outputBitstream = ob.bitstreamBuffer;
    if (f.nvEncLockBitstream(enc, &lb) != NV_ENC_SUCCESS) {
      e.error = "lock bitstream";
      break;
    }
    const auto* bytes = static_cast<const uint8_t*>(lb.bitstreamBufferPtr);
    e.stream.insert(e.stream.end(), bytes, bytes + lb.bitstreamSizeInBytes);
    e.types.push_back(static_cast<int>(lb.pictureType));
    e.sizes.push_back(lb.bitstreamSizeInBytes);
    f.nvEncUnlockBitstream(enc, ob.bitstreamBuffer);
    e.input.push_back(std::move(in));
  }
  f.nvEncDestroyBitstreamBuffer(enc, ob.bitstreamBuffer);
  f.nvEncDestroyInputBuffer(enc, ib.inputBuffer);
  f.nvEncDestroyEncoder(enc);
  return e;
}

double psnr_of(const Planes& in, const std::vector<uint8_t>& dec, int dw, int dh) {
  double sq = 0;
  size_t n = 0;
  const size_t cw = static_cast<size_t>((dw + 1) / 2), ch = static_cast<size_t>((dh + 1) / 2);
  auto add = [&](const std::vector<uint8_t>& want, int w, int h, const uint8_t* got, size_t gpitch) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double d = static_cast<double>(got[static_cast<size_t>(y) * gpitch + static_cast<size_t>(x)]) - want[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
        sq += d * d;
        ++n;
      }
  };
  add(in.y, in.w, in.h, dec.data(), static_cast<size_t>(dw));
  add(in.u, in.cw(), in.ch(), dec.data() + static_cast<size_t>(dw) * dh, cw);
  add(in.v, in.cw(), in.ch(), dec.data() + static_cast<size_t>(dw) * dh + cw * ch, cw);
  const double mse = sq / static_cast<double>(n);
  return mse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

// The profile_idc of the first SPS in an Annex B stream, or -1.
int profile_of(const std::vector<uint8_t>& s) {
  for (size_t i = 0; i + 5 < s.size(); ++i)
    if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1 && (s[i + 3] & 31) == 7) return s[i + 4];
  return -1;
}

std::string types_string(const std::vector<int>& types) {
  std::string out;
  for (int t : types) out += t == NV_ENC_PIC_TYPE_IDR ? 'I' : (t == NV_ENC_PIC_TYPE_P ? 'p' : (t == NV_ENC_PIC_TYPE_I ? 'i' : '?'));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const bool card = argc > 1 && !std::strcmp(argv[1], "--card");
  void* enclib = dlopen("libnvidia-encode.so.1", RTLD_NOW);
  void* declib = dlopen("libnvcuvid.so.1", RTLD_NOW);
  if (!enclib || !declib) {
    std::printf("SKIP: no %s\n", !enclib ? "libnvidia-encode.so.1" : "libnvcuvid.so.1");
    return 0;
  }
  if (!tool_exists("ffmpeg") || !tool_exists("ffprobe")) {
    std::printf("SKIP: ffmpeg / ffprobe not found (the reference decoder)\n");
    return 0;
  }
  auto create = reinterpret_cast<NVENCSTATUS (*)(NV_ENCODE_API_FUNCTION_LIST*)>(dlsym(enclib, "NvEncodeAPICreateInstance"));
  f.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  if (!create || create(&f) != NV_ENC_SUCCESS) {
    std::printf("FAIL: NvEncodeAPICreateInstance\n");
    return 1;
  }
#define LOAD(field, name) \
  cv.field = reinterpret_cast<decltype(cv.field)>(dlsym(declib, name)); \
  if (!cv.field) { std::printf("FAIL: %s missing from libnvcuvid\n", name); return 1; }
  LOAD(create_parser, "cuvidCreateVideoParser")
  LOAD(parse, "cuvidParseVideoData")
  LOAD(destroy_parser, "cuvidDestroyVideoParser")
  LOAD(create_decoder, "cuvidCreateDecoder")
  LOAD(decode, "cuvidDecodePicture")
  LOAD(destroy_decoder, "cuvidDestroyDecoder")
  LOAD(map, "cuvidMapVideoFrame64")
  LOAD(unmap, "cuvidUnmapVideoFrame64")
#undef LOAD
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx;
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);

  // frameIntervalP 0: an IDR picture first and then I pictures, as the card writes them (measured on an RTX 3060)
  const auto intra_only = [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
    for (size_t i = 0; i < e.types.size(); ++i)
      if (e.types[i] != (i == 0 ? NV_ENC_PIC_TYPE_IDR : NV_ENC_PIC_TYPE_I)) return "pictures are not an IDR picture and I pictures: " + types_string(e.types);
    return "";
  };
  const std::string tmp = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/vgpu_nvenc_nvdec_" + std::to_string(getpid()) + ".es";

  const std::vector<Case> cases = {
      {"IPPP, constant QP 24, 192x128", 192, 128, 8,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
         c.rcParams.constQP.qpInterP = c.rcParams.constQP.qpIntra = c.rcParams.constQP.qpInterB = 24;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         if (types_string(e.types) != "Ippppppp") return "picture types " + types_string(e.types) + ", wanted Ippppppp";
         // inter pictures cost less than the intra picture
         for (size_t i = 1; i < e.sizes.size(); ++i)
           if (e.sizes[i] >= e.sizes[0]) return "an inter picture is not smaller than the IDR picture";
         return "";
       }},
      {"gop 5 with a forced IDR and a forced intra picture", 160, 96, 12,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.gopLength = 5;
         c.encodeCodecConfig.h264Config.idrPeriod = 5;
       },
       [](int i) -> uint32_t { return i == 2 ? NV_ENC_PIC_FLAG_FORCEIDR : (i == 4 ? NV_ENC_PIC_FLAG_FORCEINTRA : 0u); },
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         const std::string t = types_string(e.types);
         // the forced IDR restarts the GOP: IDR at 0 (first), 2 (forced), 7 (five pictures later); picture 4 is an I picture
         if (t[0] != 'I' || t[2] != 'I') return "the first and the forced picture are not IDR: " + t;
         if (t[1] != 'p' || t[3] != 'p') return "pictures around the forced IDR are not P: " + t;
         return "";
       }},
      {"intra only (frameIntervalP 0)", 160, 96, 5,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) { c.frameIntervalP = 0; }, nullptr, intra_only},
      {"odd size 161x51, IPPP", 161, 51, 6, nullptr, nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int dw, int dh, bool) -> std::string {
         if (dw < 161 || dh < 51 || dw > 162 || dh > 52) return "decoded size " + std::to_string(dw) + "x" + std::to_string(dh);
         return types_string(e.types) == "Ippppp" ? "" : "picture types " + types_string(e.types);
       }},
      {"constant bit rate 400 kbit/s, 320x180, 30 pictures", 320, 180, 30,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
         c.rcParams.averageBitRate = 400000;
         c.rcParams.maxBitRate = 400000;
         c.gopLength = NVENC_INFINITE_GOPLENGTH;
         c.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         size_t total = 0;
         for (size_t s : e.sizes) total += s;
         const double kbps = static_cast<double>(total) * 8 / (static_cast<double>(e.sizes.size()) / 30.0) / 1000.0;
         std::printf("  bit rate %.0f kbit/s for a target of 400\n", kbps);
         if (kbps < 200 || kbps > 640) return "bit rate " + std::to_string(static_cast<int>(kbps)) + " kbit/s, target 400";
         return "";
       }},
      {"profile Baseline names itself in the SPS", 160, 96, 3,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) { c.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID; }, nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         return profile_of(e.stream) == 66 ? "" : "profile_idc " + std::to_string(profile_of(e.stream)) + ", wanted 66";
       }},
      {"profile Main names itself in the SPS", 160, 96, 3,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) { c.profileGUID = NV_ENC_H264_PROFILE_MAIN_GUID; }, nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         return profile_of(e.stream) == 77 ? "" : "profile_idc " + std::to_string(profile_of(e.stream)) + ", wanted 77";
       }},
      {"profile High names itself in the SPS", 160, 96, 3,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) { c.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID; }, nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         return profile_of(e.stream) == 100 ? "" : "profile_idc " + std::to_string(profile_of(e.stream)) + ", wanted 100";
       }},
  };

  int failures = 0;
  for (const Case& c : cases) {
    Encoded e = encode(ctx, c);
    if (!e.error.empty()) {
      std::printf("FAIL %s: %s\n", c.name, e.error.c_str());
      ++failures;
      continue;
    }
    const Decoded ff = ffmpeg_decode(e.stream, tmp);
    if (!ff.error.empty()) {
      std::printf("FAIL %s: %s\n", c.name, ff.error.c_str());
      ++failures;
      continue;
    }
    const Decoded nv = nvdec_decode(e.stream);
    std::string problem;
    if (!nv.error.empty()) problem = "NVDEC: " + nv.error;
    else if (nv.frames.size() != ff.frames.size() || static_cast<int>(ff.frames.size()) != c.frames)
      problem = "decoded " + std::to_string(nv.frames.size()) + " (NVDEC) and " + std::to_string(ff.frames.size()) + " (ffmpeg) frames of " + std::to_string(c.frames);
    else if (nv.w != ff.w || nv.h != ff.h)
      problem = "sizes differ: NVDEC " + std::to_string(nv.w) + "x" + std::to_string(nv.h) + ", ffmpeg " + std::to_string(ff.w) + "x" + std::to_string(ff.h);
    double worst = 99.0;
    int differing = 0;
    if (problem.empty()) {
      for (size_t i = 0; i < ff.frames.size(); ++i) {
        if (nv.frames[i] != ff.frames[i]) ++differing;
        worst = std::min(worst, psnr_of(e.input[i], ff.frames[i], ff.w, ff.h));
      }
      if (differing) problem = std::to_string(differing) + " frames differ between NVDEC and ffmpeg";
      else if (worst < 30.0 && std::strstr(c.name, "bit rate") == nullptr) problem = "worst PSNR " + std::to_string(worst) + " dB";
      else if (worst < 24.0) problem = "worst PSNR " + std::to_string(worst) + " dB";
    }
    if (problem.empty() && c.check) problem = c.check(e, ff.frames, ff.w, ff.h, card);
    size_t total = 0;
    for (size_t s : e.sizes) total += s;
    std::printf("%s %s: %zu frames, %zu bytes, types %s, worst PSNR %.1f dB, NVDEC == ffmpeg%s%s\n", problem.empty() ? "PASS" : "FAIL", c.name, ff.frames.size(), total,
                types_string(e.types).c_str(), worst, problem.empty() ? "" : " -- ", problem.c_str());
    if (!problem.empty()) ++failures;
  }
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
