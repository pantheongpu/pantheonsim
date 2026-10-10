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
// --light (run_nvenc.sh passes it for a build under a sanitizer, which runs this 10 to 20 times slower): leave out the cases of more than
// a million pixels in all, the long rate-control runs. The normal build and the card run them.
//
// SKIP (exit 0) when libnvidia-encode.so.1 or libnvcuvid.so.1 is missing. Without ffmpeg or ffprobe only the comparison with ffmpeg is skipped.
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
  std::vector<uint64_t> timestamps;   // outputTimeStamp of every picture
  std::vector<int> statuses;    // the status of every NvEncEncodePicture call
  std::vector<int> slices;      // slice NAL units of every picture
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

// The encode loop of an application that tolerates reordering: input and output buffers come from pools (the encoder may hold an input buffer until its
// picture is coded), a call answered NV_ENC_ERR_NEED_MORE_INPUT leaves its output buffer outstanding, and a call that succeeds fills the outstanding
// buffers and its own, in the order of the calls; the end of the stream flushes the rest. Packets are collected in coding order.
constexpr int kPool = 16;
int slices_in(const uint8_t* p, size_t n);

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
  pc.presetCfg.frameIntervalP = 1;   // output order is coding order, unless the case asks for B pictures
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
  NV_ENC_INPUT_PTR in_buf[kPool];
  NV_ENC_OUTPUT_PTR out_buf[kPool];
  int created_in = 0, created_out = 0;
  for (int i = 0; i < kPool; ++i) {
    NV_ENC_CREATE_INPUT_BUFFER ib{};
    ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
    ib.width = static_cast<uint32_t>(c.w);
    ib.height = static_cast<uint32_t>(c.h);
    ib.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    if (f.nvEncCreateInputBuffer(enc, &ib) != NV_ENC_SUCCESS) break;
    in_buf[created_in++] = ib.inputBuffer;
    NV_ENC_CREATE_BITSTREAM_BUFFER ob{};
    ob.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    if (f.nvEncCreateBitstreamBuffer(enc, &ob) != NV_ENC_SUCCESS) break;
    out_buf[created_out++] = ob.bitstreamBuffer;
  }
  auto cleanup = [&] {
    for (int i = 0; i < created_out; ++i) f.nvEncDestroyBitstreamBuffer(enc, out_buf[i]);
    for (int i = 0; i < created_in; ++i) f.nvEncDestroyInputBuffer(enc, in_buf[i]);
    f.nvEncDestroyEncoder(enc);
  };
  if (created_in < kPool || created_out < kPool) {
    e.error = "buffer creation refused";
    cleanup();
    return e;
  }
  std::vector<int> outstanding;   // output buffers of the calls that returned NV_ENC_ERR_NEED_MORE_INPUT
  auto collect = [&](int upto_buf) {
    for (int b : outstanding) {
      NV_ENC_LOCK_BITSTREAM lb{};
      lb.version = NV_ENC_LOCK_BITSTREAM_VER;
      lb.outputBitstream = out_buf[b];
      if (f.nvEncLockBitstream(enc, &lb) != NV_ENC_SUCCESS) {
        e.error = "lock bitstream";
        return;
      }
      const auto* bytes = static_cast<const uint8_t*>(lb.bitstreamBufferPtr);
      e.stream.insert(e.stream.end(), bytes, bytes + lb.bitstreamSizeInBytes);
      e.types.push_back(static_cast<int>(lb.pictureType));
      e.sizes.push_back(lb.bitstreamSizeInBytes);
      e.timestamps.push_back(lb.outputTimeStamp);
      e.slices.push_back(slices_in(bytes, lb.bitstreamSizeInBytes));
      f.nvEncUnlockBitstream(enc, out_buf[b]);
    }
    (void)upto_buf;
    outstanding.clear();
  };
  for (int t = 0; t < c.frames && e.error.empty(); ++t) {
    Planes in = make_planes(c.w, c.h, t);
    const int slot = t % kPool;
    NV_ENC_LOCK_INPUT_BUFFER lk{};
    lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
    lk.inputBuffer = in_buf[slot];
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
    f.nvEncUnlockInputBuffer(enc, in_buf[slot]);
    NV_ENC_PIC_PARAMS pp{};
    pp.version = NV_ENC_PIC_PARAMS_VER;
    pp.inputBuffer = in_buf[slot];
    pp.outputBitstream = out_buf[slot];
    pp.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    pp.inputWidth = static_cast<uint32_t>(c.w);
    pp.inputHeight = static_cast<uint32_t>(c.h);
    pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pp.inputTimeStamp = static_cast<uint64_t>(t) * 3000;
    pp.encodePicFlags = c.flags ? c.flags(t) : 0;
    const NVENCSTATUS st = f.nvEncEncodePicture(enc, &pp);
    outstanding.push_back(slot);
    e.statuses.push_back(static_cast<int>(st));
    if (st == NV_ENC_SUCCESS) collect(slot);
    else if (st != NV_ENC_ERR_NEED_MORE_INPUT) {
      e.error = "encode call failed";
      break;
    }
    e.input.push_back(std::move(in));
  }
  if (e.error.empty() && !outstanding.empty()) {
    NV_ENC_PIC_PARAMS eos{};
    eos.version = NV_ENC_PIC_PARAMS_VER;
    eos.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    if (f.nvEncEncodePicture(enc, &eos) != NV_ENC_SUCCESS) e.error = "end of stream refused";
    else collect(0);
  }
  cleanup();
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

// entropy_coding_mode_flag of the first PPS of an Annex B stream, or -1
int pps_entropy_flag(const std::vector<uint8_t>& s) {
  for (size_t i = 0; i + 8 < s.size(); ++i)
    if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1 && (s[i + 3] & 31) == 8) {
      std::vector<uint8_t> r;
      int zeros = 0;
      for (size_t k = i + 4; k < std::min(s.size(), i + 24); ++k) {
        if (zeros >= 2 && s[k] == 3) { zeros = 0; continue; }
        r.push_back(s[k]);
        zeros = s[k] == 0 ? zeros + 1 : 0;
      }
      size_t pos = 0;
      auto bit = [&]() { const int b = (r[pos >> 3] >> (7 - (pos & 7))) & 1; ++pos; return b; };
      auto ue = [&]() { int z = 0; while (!bit() && z < 24) ++z; uint32_t v = 0; for (int k = 0; k < z; ++k) v = v << 1 | static_cast<uint32_t>(bit()); return (1u << z) - 1 + v; };
      ue();
      ue();
      return bit();
    }
  return -1;
}

// slice NAL units (types 1 and 5) in the bytes of one packet
int slices_in(const uint8_t* p, size_t n) {
  int count = 0;
  for (size_t i = 0; i + 4 < n; ++i)
    if (p[i] == 0 && p[i + 1] == 0 && p[i + 2] == 1 && ((p[i + 3] & 31) == 1 || (p[i + 3] & 31) == 5)) ++count;
  return count;
}

// The bit rate of a coded stream at 30 pictures a second, in kbit/s
double kbps_of(const Encoded& e) {
  size_t total = 0;
  for (size_t s : e.sizes) total += s;
  return static_cast<double>(total) * 8 / (static_cast<double>(e.input.size()) / 30.0) / 1000.0;
}

std::string types_string(const std::vector<int>& types) {
  std::string out;
  for (int t : types) out += t == NV_ENC_PIC_TYPE_IDR ? 'I' : (t == NV_ENC_PIC_TYPE_P ? 'p' : (t == NV_ENC_PIC_TYPE_I ? 'i' : (t == NV_ENC_PIC_TYPE_B ? 'b' : '?')));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  bool card = false;
  bool light = false;
  const char* only = nullptr;   // --only <text>: run the cases whose names contain it
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--card")) card = true;
    else if (!std::strcmp(argv[i], "--light")) light = true;
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
  }
  void* enclib = dlopen("libnvidia-encode.so.1", RTLD_NOW);
  void* declib = dlopen("libnvcuvid.so.1", RTLD_NOW);
  if (!enclib || !declib) {
    std::printf("SKIP: no %s\n", !enclib ? "libnvidia-encode.so.1" : "libnvcuvid.so.1");
    return 0;
  }
  // ffmpeg is the reference decoder; without it only that comparison is skipped -- NVDEC still decodes every stream and is held to the same quality floor.
  const bool have_ffmpeg = tool_exists("ffmpeg") && tool_exists("ffprobe");
  if (!have_ffmpeg) std::printf("note: ffmpeg / ffprobe not found, so NVDEC's output is not compared with ffmpeg's (the pictures are still decoded and measured)\n");
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

      {"B pictures: frameIntervalP 3", 192, 128, 10,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.frameIntervalP = 3;
         c.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
         c.rcParams.constQP.qpInterP = c.rcParams.constQP.qpIntra = 26;
         c.rcParams.constQP.qpInterB = 28;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         // a P picture ends every group of three: the two pictures before it in display order wait (NEED_MORE_INPUT) and are coded after it
         const std::vector<int> want = {0, NV_ENC_ERR_NEED_MORE_INPUT, NV_ENC_ERR_NEED_MORE_INPUT, 0, NV_ENC_ERR_NEED_MORE_INPUT, NV_ENC_ERR_NEED_MORE_INPUT, 0,
                                        NV_ENC_ERR_NEED_MORE_INPUT, NV_ENC_ERR_NEED_MORE_INPUT, 0};
         if (e.statuses != want) return "call statuses are not the group-of-three pattern";
         if (types_string(e.types) != "Ipbbpbbpbb") return "picture types " + types_string(e.types) + ", wanted Ipbbpbbpbb";
         // the packets of a group are its pictures: a P picture first, with the latest time stamp of the group, then the B pictures before it
         for (size_t g = 1; g + 2 < e.timestamps.size(); g += 3) {
           if (e.timestamps[g] < e.timestamps[g + 1] || e.timestamps[g] < e.timestamps[g + 2]) return "the P picture of a group is not the latest in display order";
           if (e.timestamps[g] != static_cast<uint64_t>(g + 2) * 3000) return "time stamp of a P picture";
           std::vector<uint64_t> ts = {e.timestamps[g], e.timestamps[g + 1], e.timestamps[g + 2]};
           std::sort(ts.begin(), ts.end());
           if (ts[0] != static_cast<uint64_t>(g) * 3000 || ts[1] != static_cast<uint64_t>(g + 1) * 3000) return "a group does not hold three consecutive pictures";
         }
         return "";
       }},
      {"B pictures: frameIntervalP 4, a GOP of 8, a short group at its end and at the end of the stream", 160, 96, 11,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.frameIntervalP = 4;
         c.gopLength = 8;
         c.encodeCodecConfig.h264Config.idrPeriod = 8;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         if (types_string(e.types) != "IpbbbpbbIpb") return "picture types " + types_string(e.types) + ", wanted IpbbbpbbIpb";
         const std::vector<int> want = {0, 17, 17, 17, 0, 17, 17, 0, 0, 17, 17};
         if (e.statuses != want) return "call statuses differ from the expected pattern";
         return "";
       }},
      {"B pictures: a forced IDR picture codes the pictures that wait", 160, 96, 9,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) { c.frameIntervalP = 3; },
       [](int i) -> uint32_t { return i == 5 ? NV_ENC_PIC_FLAG_FORCEIDR : 0u; },
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         if (types_string(e.types) != "IpbbpIpbb") return "picture types " + types_string(e.types) + ", wanted IpbbpIpbb";
         return "";
       }},
      {"Baseline profile takes no B pictures", 160, 96, 6,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;
         c.frameIntervalP = 3;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         for (int st : e.statuses)
           if (st != 0) return "a call was answered with status " + std::to_string(st);
         if (types_string(e.types) != "Ippppp") return "picture types " + types_string(e.types);
         return pps_entropy_flag(e.stream) == 0 ? "" : "the PPS says CABAC";
       }},
      {"CABAC, the 8x8 transform and two slices", 192, 128, 5,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
         c.frameIntervalP = 1;
         c.encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
         c.encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;
         c.encodeCodecConfig.h264Config.sliceMode = 3;
         c.encodeCodecConfig.h264Config.sliceModeData = 2;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         if (pps_entropy_flag(e.stream) != 1) return "the PPS does not say CABAC";
         for (int s : e.slices)
           if (s != 2) return "a picture has " + std::to_string(s) + " slices, wanted 2";
         return "";
       }},
      {"CAVLC with the High profile", 192, 128, 4,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
         c.frameIntervalP = 1;
         c.encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool) -> std::string {
         return pps_entropy_flag(e.stream) == 0 ? "" : "the PPS does not say CAVLC";
       }},

      {"constant bit rate 250 kbit/s with B pictures, 320x180, 45 pictures", 320, 180, 45,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
         c.rcParams.averageBitRate = 250000;
         c.rcParams.maxBitRate = 250000;
         c.frameIntervalP = 3;
         c.gopLength = NVENC_INFINITE_GOPLENGTH;
         c.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool card) -> std::string {
         const double kbps = kbps_of(e);
         std::printf("  bit rate %.0f kbit/s for a target of 250\n", kbps);
         if (card) return "";   // the card's own accuracy is only reported
         return std::fabs(kbps - 250) <= 0.05 * 250 ? "" : "bit rate " + std::to_string(static_cast<int>(kbps)) + " kbit/s, target 250 (within 5%)";
       }},
      {"variable bit rate 1500 kbit/s, peak 3000, 640x360, 36 pictures, a GOP of 12", 640, 360, 36,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
         c.rcParams.averageBitRate = 1500000;
         c.rcParams.maxBitRate = 3000000;
         c.frameIntervalP = 1;
         c.gopLength = 12;
         c.encodeCodecConfig.h264Config.idrPeriod = 12;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool card) -> std::string {
         const double kbps = kbps_of(e);
         std::printf("  bit rate %.0f kbit/s for a target of 1500\n", kbps);
         if (card) return "";
         return std::fabs(kbps - 1500) <= 0.05 * 1500 ? "" : "bit rate " + std::to_string(static_cast<int>(kbps)) + " kbit/s, target 1500 (within 5%)";
       }},
      {"constant bit rate 80 kbit/s, 192x128, 40 pictures", 192, 128, 40,
       [](NV_ENC_CONFIG& c, NV_ENC_INITIALIZE_PARAMS&) {
         c.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
         c.rcParams.averageBitRate = 80000;
         c.rcParams.maxBitRate = 80000;
         c.frameIntervalP = 1;
         c.gopLength = NVENC_INFINITE_GOPLENGTH;
         c.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
       },
       nullptr,
       [](const Encoded& e, const std::vector<std::vector<uint8_t>>&, int, int, bool card) -> std::string {
         const double kbps = kbps_of(e);
         std::printf("  bit rate %.0f kbit/s for a target of 80\n", kbps);
         if (card) return "";
         return std::fabs(kbps - 80) <= 0.05 * 80 ? "" : "bit rate " + std::to_string(static_cast<int>(kbps)) + " kbit/s, target 80 (within 5%)";
       }},
  };

  int failures = 0;
  for (const Case& c : cases) {
    if (only && !std::strstr(c.name, only)) continue;
    if (light && static_cast<long long>(c.w) * c.h * c.frames > 1000000) {
      std::printf("left out under --light: %s\n", c.name);
      continue;
    }
    Encoded e = encode(ctx, c);
    if (!e.error.empty()) {
      std::printf("FAIL %s: %s\n", c.name, e.error.c_str());
      ++failures;
      continue;
    }
    Decoded ff;
    if (have_ffmpeg) {
      ff = ffmpeg_decode(e.stream, tmp);
      if (!ff.error.empty()) {
        std::printf("FAIL %s: %s\n", c.name, ff.error.c_str());
        ++failures;
        continue;
      }
    }
    const Decoded nv = nvdec_decode(e.stream);
    const Decoded& rf = have_ffmpeg ? ff : nv;   // the frames the quality is measured on
    std::string problem;
    if (!nv.error.empty()) problem = "NVDEC: " + nv.error;
    else if (static_cast<int>(nv.frames.size()) != c.frames || (have_ffmpeg && ff.frames.size() != nv.frames.size()))
      problem = "decoded " + std::to_string(nv.frames.size()) + " (NVDEC) and " + std::to_string(ff.frames.size()) + " (ffmpeg) frames of " + std::to_string(c.frames);
    else if (have_ffmpeg && (nv.w != ff.w || nv.h != ff.h))
      problem = "sizes differ: NVDEC " + std::to_string(nv.w) + "x" + std::to_string(nv.h) + ", ffmpeg " + std::to_string(ff.w) + "x" + std::to_string(ff.h);
    double worst = 99.0;
    int differing = 0;
    if (problem.empty()) {
      for (size_t i = 0; i < rf.frames.size(); ++i) {
        if (have_ffmpeg && nv.frames[i] != ff.frames[i]) ++differing;
        worst = std::min(worst, psnr_of(e.input[i], rf.frames[i], rf.w, rf.h));
      }
      if (differing) problem = std::to_string(differing) + " frames differ between NVDEC and ffmpeg";
      else if (worst < 30.0 && std::strstr(c.name, "bit rate") == nullptr) problem = "worst PSNR " + std::to_string(worst) + " dB";
      else if (worst < 24.0) problem = "worst PSNR " + std::to_string(worst) + " dB";
    }
    if (problem.empty() && c.check) problem = c.check(e, rf.frames, rf.w, rf.h, card);
    size_t total = 0;
    for (size_t s : e.sizes) total += s;
    std::printf("%s %s: %zu frames, %zu bytes, types %s, worst PSNR %.1f dB%s%s%s\n", problem.empty() ? "PASS" : "FAIL", c.name, rf.frames.size(), total,
                types_string(e.types).c_str(), worst, have_ffmpeg ? ", NVDEC == ffmpeg" : "", problem.empty() ? "" : " -- ", problem.c_str());
    if (!problem.empty()) ++failures;
  }
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
