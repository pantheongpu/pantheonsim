// libvgpunvcuvid -- VirtualGPU's NVDEC, presented as libnvcuvid.so.1.
//
// NVDEC is fixed-function decode silicon behind the driver; applications reach
// it through libnvcuvid's cuvid* entry points (the video parser and the
// decoder), usually via NVIDIA's own NvDecoder wrapper or FFmpeg's *_cuvid
// decoders. This library answers that API with a software decoder -- for one
// codec. Motion JPEG is a sequence of independent JPEG pictures, and a JPEG
// decoder is already here (the nvJPEG library's), so cudaVideoCodec_JPEG is
// decoded on the host and the NV12 surfaces the application maps are written
// to device memory.
//
// The subset, stated plainly:
//   decoded   Motion JPEG / JPEG pictures that the nvJPEG library decodes and the
//             card's NVDEC does: sequential 8-bit Huffman (baseline and extended),
//             any chroma subsampling or grey, restart intervals included. The
//             card refuses progressive pictures (CUDA_ERROR_INVALID_IMAGE, picture
//             status Error), so this does too. The surface is always NV12 (a 4:4:4
//             or 4:2:2 picture is averaged down to 4:2:0, a grey one has chroma
//             128), as on the RTX 3060.
//   not here  every other codec: cuvidGetDecoderCaps reports bIsSupported = 0,
//             and cuvidCreateDecoder / cuvidCreateVideoParser answer
//             CUDA_ERROR_NOT_SUPPORTED (the card accepts MPEG-1/2/4, VC-1, H.264,
//             HEVC, VP8, VP9 and AV1; a software decoder for them is not part of
//             this simulator, and an application is better off falling back to
//             its CPU decoder than receiving a wrong picture). Video sources
//             (files, URLs) need a demuxer and answer CUDA_ERROR_NOT_SUPPORTED.
//   not here  display-area and target-rectangle cropping, deinterlacing,
//             output formats other than NV12 (the card refuses them too, with
//             the statuses below).
//
// Behaviour measured on libnvcuvid.so.1 (driver 595, an RTX 3060), and kept:
//   * a surface is pitch x (target height + target height / 2) bytes, pitch the
//     target width rounded up to 512; chroma starts pitch x target-height bytes
//     in, interleaved Cb Cr; memory starts zeroed, and cuvidMapVideoFrame
//     returns the surface itself (the same pointer every time, valid before
//     any decode, never copied);
//   * a picture larger than the decoder is cut to it, a smaller one fills the
//     top-left corner;
//   * a target size within about 2% of the decoder's is a crop, otherwise the
//     picture is resampled (bilinear to enlarge; averaging to shrink, which the
//     card does exactly only for whole factors -- other shrinks differ by a few
//     levels at edges);
//   * the parser buffers bytes until a picture's end marker, delivers
//     sequence / decode / display callbacks, ignores their return values,
//     stamps the first picture of a packet with the packet's timestamp and each
//     further picture with the previous plus clock/30, and cycles the picture
//     index through the sequence callback's return value (at least the parser's
//     ulMaxNumDecodeSurfaces).
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "cuviddec.h"
#include "nvcuvid.h"
#include "nvjpeg_codec.hpp"

// The header renames the 32-bit entry points to the 64-bit ones for 64-bit
// builds; the library exports both.
#undef cuvidMapVideoFrame
#undef cuvidUnmapVideoFrame

#define VGPU_API extern "C" __attribute__((visibility("default")))

namespace {

std::mutex g_mu;

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

void say_once(const char* what) {
  static std::mutex mu;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> lock(mu);
  if (!seen.insert(what).second || quiet()) return;
  std::fprintf(stderr, "[vgpu] NVDEC: %s\n", what);
}

constexpr int kCodecs = 12;   // cudaVideoCodec_MPEG1 .. AV1

// An enum field of an application's structure as the number it holds: the card
// answers values the enums have no name for, and loading one as its enum type is
// undefined.
template <class E>
int raw(const E& e) {
  static_assert(sizeof(E) == sizeof(int), "a 32-bit enum");
  int v;
  std::memcpy(&v, &e, sizeof v);
  return v;
}

// CUresult the card answers with for a decoder it cannot size (cuda.h's name for
// it is CUDA_ERROR_NO_DEVICE).
constexpr CUresult kBadSize = CUDA_ERROR_NO_DEVICE;

struct Surface {
  void* dev = nullptr;
  cuvidDecodeStatus status = cuvidDecodeStatus_Invalid;   // Success after a decode, Error after a failed one
};

struct Decoder {
  CUVIDDECODECREATEINFO info{};
  uint32_t width = 0, height = 0;           // the decoder's coded size
  uint32_t target_w = 0, target_h = 0;      // the surface size
  uint32_t max_w = 0, max_h = 0;
  uint32_t pitch = 0;
  std::vector<Surface> surfaces;
};

std::set<Decoder*> g_decoders;

Decoder* find_decoder(CUvideodecoder d) {
  auto* p = static_cast<Decoder*>(d);
  return g_decoders.count(p) ? p : nullptr;
}

uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

size_t surface_bytes(const Decoder& d) {
  return static_cast<size_t>(d.pitch) * (d.target_h + (d.target_h + 1) / 2);
}

// The surface's device memory, zeroed on first use.
bool ensure_surface(Decoder& d, Surface& s) {
  if (s.dev) return true;
  if (cudaMalloc(&s.dev, surface_bytes(d)) != cudaSuccess) {
    s.dev = nullptr;
    return false;
  }
  cudaMemset(s.dev, 0, surface_bytes(d));
  return true;
}

void set_geometry(Decoder& d, uint32_t w, uint32_t h, uint32_t tw, uint32_t th) {
  d.width = w;
  d.height = h;
  d.target_w = tw ? tw : w;
  d.target_h = th ? th : h;
  d.pitch = align_up(d.target_w, 512);
  for (Surface& s : d.surfaces) {
    if (s.dev) cudaFree(s.dev);
    s = Surface{};
  }
}

}  // namespace

/* ---- capabilities ------------------------------------------------------- */

VGPU_API CUresult CUDAAPI cuvidGetDecoderCaps(CUVIDDECODECAPS* caps) {
  if (!caps) return CUDA_ERROR_INVALID_VALUE;
  caps->bIsSupported = 0;
  caps->nMaxWidth = caps->nMaxHeight = caps->nMaxMBCount = 0;
  caps->nMinWidth = caps->nMinHeight = 0;
  caps->nOutputFormatMask = 0;
  // The card's JPEG engine answers for any chroma format value; only 8-bit.
  if (raw(caps->eCodecType) == cudaVideoCodec_JPEG && caps->nBitDepthMinus8 == 0) {
    caps->bIsSupported = 1;
    caps->nMaxWidth = 32768;
    caps->nMaxHeight = 16384;
    caps->nMaxMBCount = 67108864;
    caps->nMinWidth = 64;
    caps->nMinHeight = 64;
    caps->nOutputFormatMask = 0x41;
  }
  return CUDA_SUCCESS;
}

/* ---- decoder ------------------------------------------------------------ */

VGPU_API CUresult CUDAAPI cuvidCreateDecoder(CUvideodecoder* out, CUVIDDECODECREATEINFO* ci) {
  if (!out || !ci) return CUDA_ERROR_INVALID_VALUE;
  const int codec = raw(ci->CodecType);
  if (codec < 0 || codec >= kCodecs) return CUDA_ERROR_INVALID_VALUE;
  // Output formats: only NV12 is created. P016 / 16-bit 4:4:4 / unknown values are
  // INVALID_VALUE on the card, 8-bit 4:4:4 is the sizing error.
  switch (raw(ci->OutputFormat)) {
    case cudaVideoSurfaceFormat_NV12: break;
    case cudaVideoSurfaceFormat_YUV444: return kBadSize;
    default: return CUDA_ERROR_INVALID_VALUE;
  }
  if (ci->ulWidth == 0 || ci->ulHeight == 0 || ci->ulWidth > 32768 || ci->ulHeight > 16384 || ci->ulNumDecodeSurfaces == 0 ||
      ci->ulNumDecodeSurfaces > 32 || ci->ulNumOutputSurfaces == 0)
    return kBadSize;
  if (codec != cudaVideoCodec_JPEG) {
    say_once("only Motion JPEG is decoded; other codecs are not supported by VirtualGPU");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  auto d = std::make_unique<Decoder>();
  d->info = *ci;
  d->max_w = static_cast<uint32_t>(ci->ulMaxWidth);
  d->max_h = static_cast<uint32_t>(ci->ulMaxHeight);
  d->surfaces.resize(ci->ulNumDecodeSurfaces);
  set_geometry(*d, static_cast<uint32_t>(ci->ulWidth), static_cast<uint32_t>(ci->ulHeight), static_cast<uint32_t>(ci->ulTargetWidth),
               static_cast<uint32_t>(ci->ulTargetHeight));
  std::lock_guard<std::mutex> lock(g_mu);
  *out = d.get();
  g_decoders.insert(d.release());
  return CUDA_SUCCESS;
}

VGPU_API CUresult CUDAAPI cuvidDestroyDecoder(CUvideodecoder dec) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  for (Surface& s : d->surfaces)
    if (s.dev) cudaFree(s.dev);
  g_decoders.erase(d);
  delete d;
  return CUDA_SUCCESS;
}

VGPU_API CUresult CUDAAPI cuvidReconfigureDecoder(CUvideodecoder dec, CUVIDRECONFIGUREDECODERINFO* info) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  if (!info) return CUDA_ERROR_INVALID_VALUE;
  // Measured: the new size may not exceed the maximum the decoder was created
  // with, and a zero size is refused.
  if (info->ulWidth == 0 || info->ulHeight == 0) return CUDA_ERROR_INVALID_VALUE;
  if ((d->max_w && info->ulWidth > d->max_w) || (d->max_h && info->ulHeight > d->max_h)) return CUDA_ERROR_INVALID_VALUE;
  if (info->ulNumDecodeSurfaces > 32) return CUDA_ERROR_INVALID_VALUE;
  if (info->ulNumDecodeSurfaces) {
    for (size_t i = info->ulNumDecodeSurfaces; i < d->surfaces.size(); ++i)
      if (d->surfaces[i].dev) cudaFree(d->surfaces[i].dev);
    d->surfaces.resize(info->ulNumDecodeSurfaces);
  }
  set_geometry(*d, info->ulWidth, info->ulHeight, info->ulTargetWidth, info->ulTargetHeight);
  return CUDA_SUCCESS;
}

namespace {

// Resampling of one 8-bit plane (src sw x sh) to tw x th. A dimension whose
// target is within the window around its source size is not resampled (the plane
// is cut or padded with zeros), as the card does; otherwise bilinear when
// enlarging and area averaging when shrinking.
bool keeps_size(uint32_t source, uint32_t target) {
  const double r = static_cast<double>(target) / source;
  return r >= 0.975 && r <= 1.018;
}

// The source samples and weights that make output sample i of a resampled axis.
std::vector<double> axis_weights(uint32_t s, uint32_t t, uint32_t i, std::vector<uint32_t>* idx) {
  std::vector<double> w;
  idx->clear();
  if (t >= s || t * 10 >= s * 9) {   // bilinear, pixel centres aligned (enlarging, and shrinking by under 10%)
    const double f = (i + 0.5) * s / t - 0.5;
    const long long f0 = static_cast<long long>(std::floor(f));
    const double a = f - static_cast<double>(f0);
    const auto clampi = [&](long long v) { return static_cast<uint32_t>(std::min<long long>(std::max<long long>(v, 0), static_cast<long long>(s) - 1)); };
    idx->push_back(clampi(f0));
    w.push_back(1 - a);
    idx->push_back(clampi(f0 + 1));
    w.push_back(a);
  } else {        // area average over the source footprint (shrinking by more)
    const double r = static_cast<double>(s) / t;
    const double lo = i * r, hi = (i + 1) * r;
    for (uint32_t k = static_cast<uint32_t>(std::floor(lo)); k < s && k < hi; ++k) {
      const double overlap = std::min(hi, k + 1.0) - std::max(lo, static_cast<double>(k));
      if (overlap > 0) {
        idx->push_back(k);
        w.push_back(overlap / r);
      }
    }
  }
  return w;
}

std::vector<uint8_t> resample(const std::vector<uint8_t>& src, uint32_t sw, uint32_t sh, uint32_t tw, uint32_t th) {
  const bool same_w = keeps_size(sw, tw), same_h = keeps_size(sh, th);
  std::vector<uint8_t> out(static_cast<size_t>(tw) * th, 0);
  if (same_w && same_h) {
    for (uint32_t y = 0; y < std::min(sh, th); ++y)
      std::memcpy(&out[static_cast<size_t>(y) * tw], &src[static_cast<size_t>(y) * sw], std::min(sw, tw));
    return out;
  }
  // Horizontal pass into doubles, then vertical.
  const uint32_t mw = same_w ? std::min(sw, tw) : tw;
  std::vector<double> tmp(static_cast<size_t>(mw) * sh);
  std::vector<uint32_t> idx;
  for (uint32_t x = 0; x < mw; ++x) {
    std::vector<double> w;
    if (!same_w) w = axis_weights(sw, tw, x, &idx);
    for (uint32_t y = 0; y < sh; ++y) {
      double v = 0;
      if (same_w) {
        v = src[static_cast<size_t>(y) * sw + x];
      } else {
        for (size_t k = 0; k < idx.size(); ++k) v += w[k] * src[static_cast<size_t>(y) * sw + idx[k]];
      }
      tmp[static_cast<size_t>(y) * mw + x] = v;
    }
  }
  const uint32_t mh = same_h ? std::min(sh, th) : th;
  for (uint32_t y = 0; y < mh; ++y) {
    std::vector<double> w;
    if (!same_h) w = axis_weights(sh, th, y, &idx);
    for (uint32_t x = 0; x < mw; ++x) {
      double v = 0;
      if (same_h) {
        v = tmp[static_cast<size_t>(y) * mw + x];
      } else {
        for (size_t k = 0; k < idx.size(); ++k) v += w[k] * tmp[static_cast<size_t>(idx[k]) * mw + x];
      }
      out[static_cast<size_t>(y) * tw + x] = static_cast<uint8_t>(std::min(255.0, std::floor(v + 0.5)));
    }
  }
  return out;
}

// Two bilinear taps for output sample i of an axis resampled from s to t samples.
std::vector<double> bilinear_taps(uint32_t s, uint32_t t, uint32_t i, std::vector<uint32_t>* idx) {
  std::vector<double> w;
  idx->clear();
  const double f = (i + 0.5) * s / t - 0.5;
  const long long f0 = static_cast<long long>(std::floor(f));
  const double a = f - static_cast<double>(f0);
  const auto clampi = [&](long long v) { return static_cast<uint32_t>(std::min<long long>(std::max<long long>(v, 0), static_cast<long long>(s) - 1)); };
  idx->push_back(clampi(f0));
  w.push_back(1 - a);
  idx->push_back(clampi(f0 + 1));
  w.push_back(a);
  return w;
}

// What NVDEC's JPEG engine decodes: a sequential Huffman picture of 8-bit samples
// (SOF0 or SOF1). Progressive pictures are refused with INVALID_IMAGE and the
// picture's status becomes Error (measured: the card fails every progressive
// fixture the nvJPEG library decodes), as is anything with another precision.
bool baseline_jpeg(const unsigned char* d, size_t n) {
  size_t i = 2;
  if (n < 4 || d[0] != 0xFF || d[1] != 0xD8) return false;
  while (i + 4 <= n) {
    if (d[i] != 0xFF) {
      ++i;
      continue;
    }
    const unsigned char m = d[i + 1];
    if (m == 0xFF) {
      ++i;
      continue;
    }
    if (m == 0x00 || m == 0x01 || (m >= 0xD0 && m <= 0xD8)) {
      i += 2;
      continue;
    }
    const size_t len = static_cast<size_t>(d[i + 2]) << 8 | d[i + 3];
    if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) return (m == 0xC0 || m == 0xC1) && i + 5 < n && d[i + 4] == 8;
    if (m == 0xDA || m == 0xD9) return false;   // image data before any frame header
    i += 2 + len;
  }
  return false;
}

}  // namespace

VGPU_API CUresult CUDAAPI cuvidDecodePicture(CUvideodecoder dec, CUVIDPICPARAMS* p) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  if (!p) return CUDA_ERROR_INVALID_VALUE;
  if (p->CurrPicIdx < 0 || static_cast<size_t>(p->CurrPicIdx) >= d->surfaces.size() || !p->pBitstreamData) return CUDA_ERROR_INVALID_VALUE;
  if (p->nBitstreamDataLen == 0) return CUDA_ERROR_INVALID_IMAGE;
  Surface& s = d->surfaces[p->CurrPicIdx];
  // Whatever goes wrong from here on, the card marks the picture as failed.
  struct Failed {
    Surface& s;
    bool ok = false;
    ~Failed() {
      if (!ok) s.status = cuvidDecodeStatus_Error;
    }
  } failed{s};
  if (!baseline_jpeg(p->pBitstreamData, p->nBitstreamDataLen)) return CUDA_ERROR_INVALID_IMAGE;

  vgpu_jpeg::Decoded jd;
  if (!vgpu_jpeg::decode_planes(p->pBitstreamData, p->nBitstreamDataLen, &jd)) return CUDA_ERROR_INVALID_IMAGE;

  // The decoder's picture, as the card's NV12 surface holds it (measured on the
  // fixtures of nvidia/tests/data/jpeg, to one level):
  //   luma    the planes on the JPEG's whole MCU grid -- the padding past the right
  //           edge is decoded data -- but rows past the picture's height are zero;
  //   chroma  at the half resolution of 4:2:0: a 4:2:0 picture's chroma is copied,
  //           any other (4:4:4, 4:2:2, 4:4:0, 4:1:1) is resampled bilinearly, pixel
  //           centres aligned, from its true chroma size to ceil(width / 2) x
  //           ceil(height / 2) -- not averaged in 2x2 blocks: for a 61x45 picture
  //           that is a scale of 61/31, and the card's samples drift against a plain
  //           2x2 average by a whole row over the picture; rows past the half height
  //           are zero. A grey picture has chroma 128 on the whole padded grid.
  const uint32_t dw = d->width, dh = d->height;
  const vgpu_jpeg::Plane& yp = jd.planes[0];
  const uint32_t fw = static_cast<uint32_t>(yp.width), fh = static_cast<uint32_t>(yp.height);
  const uint32_t pw = static_cast<uint32_t>(jd.width), ph = static_cast<uint32_t>(jd.height);   // the picture
  const uint32_t cw = fw / 2, ch = fh / 2;                                                       // padded chroma grid
  const uint32_t pcw = (pw + 1) / 2, pch = (ph + 1) / 2;                                         // chroma picture
  std::vector<uint8_t> luma = yp.data;
  for (uint32_t y = ph; y < fh; ++y) std::fill(luma.begin() + static_cast<long>(y) * fw, luma.begin() + static_cast<long>(y + 1) * fw, 0);
  std::vector<uint8_t> cb(static_cast<size_t>(cw) * ch, 0), cr(cb.size(), 0);
  if (jd.planes.size() == 1) {
    std::fill(cb.begin(), cb.end(), 128);
    std::fill(cr.begin(), cr.end(), 128);
  } else {
    for (int c = 0; c < 2; ++c) {
      const vgpu_jpeg::Plane& sp = jd.planes[1 + c];
      std::vector<uint8_t>& out = c == 0 ? cb : cr;
      const uint32_t sw0 = (pw * sp.h_samp + jd.hmax - 1) / jd.hmax, sh0 = (ph * sp.v_samp + jd.vmax - 1) / jd.vmax;   // true chroma size
      const bool same_w = sw0 == pcw, same_h = sh0 == pch;
      // Resampled vertically, the card leaves the last row of an odd count unwritten.
      const uint32_t rows = same_h ? std::min(ch, pch) : std::min(ch, pch & ~1u);
      for (uint32_t y = 0; y < rows; ++y) {
        std::vector<uint32_t> iy{y};
        std::vector<double> wy{1.0};
        if (!same_h) wy = bilinear_taps(sh0, pch, y, &iy);
        for (uint32_t x = 0; x < cw; ++x) {
          std::vector<uint32_t> ix{std::min<uint32_t>(x, sp.width - 1)};
          std::vector<double> wx{1.0};
          // Not resampled across, the padding past the edge is decoded data.
          if (!same_w) wx = bilinear_taps(sw0, pcw, x, &ix);
          double v = 0;
          for (size_t j = 0; j < iy.size(); ++j)
            for (size_t i = 0; i < ix.size(); ++i) v += wy[j] * wx[i] * sp.data[static_cast<size_t>(iy[j]) * sp.width + ix[i]];
          out[static_cast<size_t>(y) * cw + x] = static_cast<uint8_t>(std::min(255.0, std::floor(v + 0.5)));
        }
      }
    }
  }

  const uint32_t dcw = (dw + 1) / 2, dch = (dh + 1) / 2;
  std::vector<uint8_t> yd(static_cast<size_t>(dw) * dh, 0), ud(static_cast<size_t>(dcw) * dch, 0), vd(ud.size(), 0);
  for (uint32_t y = 0; y < std::min(fh, dh); ++y) std::memcpy(&yd[static_cast<size_t>(y) * dw], &luma[static_cast<size_t>(y) * fw], std::min(fw, dw));
  for (uint32_t y = 0; y < std::min(ch, dch); ++y) {
    std::memcpy(&ud[static_cast<size_t>(y) * dcw], &cb[static_cast<size_t>(y) * cw], std::min(cw, dcw));
    std::memcpy(&vd[static_cast<size_t>(y) * dcw], &cr[static_cast<size_t>(y) * cw], std::min(cw, dcw));
  }

  const uint32_t tw = d->target_w, th = d->target_h;
  const uint32_t tcw = (tw + 1) / 2, tch = (th + 1) / 2;
  const std::vector<uint8_t> yt = resample(yd, dw, dh, tw, th);
  const std::vector<uint8_t> ut = resample(ud, dcw, dch, tcw, tch), vt = resample(vd, dcw, dch, tcw, tch);
  std::vector<uint8_t> uv(static_cast<size_t>(tcw) * 2 * tch);
  for (size_t i = 0; i < ut.size(); ++i) {
    uv[2 * i] = ut[i];
    uv[2 * i + 1] = vt[i];
  }
  if (!ensure_surface(*d, s)) return CUDA_ERROR_OUT_OF_MEMORY;
  cudaMemcpy2D(s.dev, d->pitch, yt.data(), tw, tw, th, cudaMemcpyHostToDevice);
  cudaMemcpy2D(static_cast<uint8_t*>(s.dev) + static_cast<size_t>(d->pitch) * th, d->pitch, uv.data(), tcw * 2, tcw * 2, tch,
               cudaMemcpyHostToDevice);
  s.status = cuvidDecodeStatus_Success;
  failed.ok = true;
  return CUDA_SUCCESS;
}

VGPU_API CUresult CUDAAPI cuvidGetDecodeStatus(CUvideodecoder dec, int idx, CUVIDGETDECODESTATUS* status) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  if (!status || idx < 0 || static_cast<size_t>(idx) >= d->surfaces.size()) return CUDA_ERROR_INVALID_VALUE;
  std::memset(status, 0, sizeof *status);
  status->decodeStatus = d->surfaces[idx].status;
  return CUDA_SUCCESS;
}

static CUresult map_frame(CUvideodecoder dec, int idx, unsigned long long* ptr, unsigned int* pitch) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  if (!ptr || !pitch || idx < 0 || static_cast<size_t>(idx) >= d->surfaces.size()) return CUDA_ERROR_INVALID_VALUE;
  Surface& s = d->surfaces[idx];
  if (!ensure_surface(*d, s)) return CUDA_ERROR_OUT_OF_MEMORY;
  *ptr = reinterpret_cast<unsigned long long>(s.dev);
  *pitch = d->pitch;
  return CUDA_SUCCESS;
}
VGPU_API CUresult CUDAAPI cuvidMapVideoFrame64(CUvideodecoder dec, int idx, unsigned long long* ptr, unsigned int* pitch,
                                               CUVIDPROCPARAMS*) {
  return map_frame(dec, idx, ptr, pitch);
}
VGPU_API CUresult CUDAAPI cuvidMapVideoFrame(CUvideodecoder dec, int idx, unsigned int* ptr, unsigned int* pitch, CUVIDPROCPARAMS*) {
  unsigned long long p = 0;
  const CUresult r = map_frame(dec, idx, &p, pitch);
  if (ptr) *ptr = static_cast<unsigned int>(p);
  return r;
}
// A mapped frame is the surface itself, so unmapping has nothing to undo: the card
// accepts any pointer, mapped or not, twice.
VGPU_API CUresult CUDAAPI cuvidUnmapVideoFrame64(CUvideodecoder dec, unsigned long long) {
  std::lock_guard<std::mutex> lock(g_mu);
  return find_decoder(dec) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}
VGPU_API CUresult CUDAAPI cuvidUnmapVideoFrame(CUvideodecoder dec, unsigned int) {
  std::lock_guard<std::mutex> lock(g_mu);
  return find_decoder(dec) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

/* ---- context locks ------------------------------------------------------ */

namespace {
struct CtxLock {
  std::recursive_mutex mu;
  int depth = 0;   // held by the calling thread; unlocking an unlocked lock is a no-op, as on the card
};
std::set<CtxLock*> g_locks;
}  // namespace

VGPU_API CUresult CUDAAPI cuvidCtxLockCreate(CUvideoctxlock* lock, CUcontext) {
  if (!lock) return CUDA_ERROR_INVALID_VALUE;
  auto* l = new CtxLock();
  std::lock_guard<std::mutex> g(g_mu);
  g_locks.insert(l);
  *lock = reinterpret_cast<CUvideoctxlock>(l);
  return CUDA_SUCCESS;
}
VGPU_API CUresult CUDAAPI cuvidCtxLockDestroy(CUvideoctxlock lock) {
  std::lock_guard<std::mutex> g(g_mu);
  auto* l = reinterpret_cast<CtxLock*>(lock);
  if (!g_locks.count(l)) return CUDA_ERROR_INVALID_HANDLE;
  g_locks.erase(l);
  delete l;
  return CUDA_SUCCESS;
}
VGPU_API CUresult CUDAAPI cuvidCtxLock(CUvideoctxlock lock, unsigned int) {
  CtxLock* l;
  {
    std::lock_guard<std::mutex> g(g_mu);
    l = reinterpret_cast<CtxLock*>(lock);
    if (!g_locks.count(l)) return CUDA_ERROR_INVALID_HANDLE;
  }
  l->mu.lock();
  ++l->depth;
  return CUDA_SUCCESS;
}
VGPU_API CUresult CUDAAPI cuvidCtxUnlock(CUvideoctxlock lock, unsigned int) {
  CtxLock* l;
  {
    std::lock_guard<std::mutex> g(g_mu);
    l = reinterpret_cast<CtxLock*>(lock);
    if (!g_locks.count(l)) return CUDA_ERROR_INVALID_HANDLE;
  }
  if (l->depth > 0) {
    --l->depth;
    l->mu.unlock();
  }
  return CUDA_SUCCESS;
}

/* ---- the video parser: Motion JPEG ------------------------------------- */

namespace {

struct Parser {
  CUVIDPARSERPARAMS params{};
  std::vector<uint8_t> pending;   // bytes of a picture still being assembled
  bool have_format = false;
  CUVIDEOFORMAT format{};
  unsigned surfaces = 1;          // indices the picture index cycles through
  unsigned next_index = 0;
  bool armed = false;             // a packet timestamp not yet given to a picture
  int64_t armed_ts = 0;
  int64_t prev_ts = 0;
  bool have_prev = false;
};

std::set<Parser*> g_parsers;

// The length of the JPEG picture at the start of `d` (SOI to EOI), or 0 if it is
// not complete yet; -1 if the bytes are not a JPEG picture.
long jpeg_end(const std::vector<uint8_t>& d, uint32_t* w, uint32_t* h, int* chroma, int* ncomp) {
  size_t i = 0;
  if (d.size() < 2) return 0;
  if (d[0] != 0xFF || d[1] != 0xD8) return -1;
  i = 2;
  int hs[4] = {0}, vs[4] = {0}, nc = 0;
  for (;;) {
    while (i < d.size() && d[i] != 0xFF) ++i;   // entropy-coded data, or garbage between segments
    while (i < d.size() && d[i] == 0xFF) ++i;
    if (i >= d.size()) return 0;
    const uint8_t m = d[i++];
    if (m == 0xD9) return static_cast<long>(i);
    if (m == 0x00 || m == 0x01 || (m >= 0xD0 && m <= 0xD8)) continue;
    if (i + 2 > d.size()) return 0;
    const size_t len = static_cast<size_t>(d[i]) << 8 | d[i + 1];
    if (len < 2) return -1;
    if (i + len > d.size()) return 0;
    if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC && len >= 8) {
      *h = static_cast<uint32_t>(d[i + 3]) << 8 | d[i + 4];
      *w = static_cast<uint32_t>(d[i + 5]) << 8 | d[i + 6];
      nc = d[i + 7];
      *ncomp = nc;
      for (int c = 0; c < nc && c < 4 && i + 8 + 3 * c + 2 < d.size(); ++c) {
        hs[c] = d[i + 9 + 3 * c] >> 4;
        vs[c] = d[i + 9 + 3 * c] & 15;
      }
      // 4:2:2 when the luma is sampled twice as often only across.
      *chroma = nc == 1 ? cudaVideoChromaFormat_Monochrome
                        : (hs[0] == 1 && vs[0] == 1 ? cudaVideoChromaFormat_444 : (vs[0] == 1 ? cudaVideoChromaFormat_422 : cudaVideoChromaFormat_420));
    }
    i += len;
  }
}

}  // namespace

VGPU_API CUresult CUDAAPI cuvidCreateVideoParser(CUvideoparser* out, CUVIDPARSERPARAMS* params) {
  if (!out || !params) return CUDA_ERROR_INVALID_VALUE;
  const int codec = raw(params->CodecType);
  if (codec < 0 || codec >= kCodecs) return CUDA_ERROR_INVALID_SOURCE;
  if (codec != cudaVideoCodec_JPEG) {
    say_once("only Motion JPEG is parsed; other codecs are not supported by VirtualGPU");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  auto* p = new Parser();
  p->params = *params;
  p->surfaces = std::max<unsigned>(1, params->ulMaxNumDecodeSurfaces);
  std::lock_guard<std::mutex> lock(g_mu);
  g_parsers.insert(p);
  *out = p;
  return CUDA_SUCCESS;
}

VGPU_API CUresult CUDAAPI cuvidParseVideoData(CUvideoparser obj, CUVIDSOURCEDATAPACKET* packet) {
  Parser* p;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    p = static_cast<Parser*>(obj);
    if (!g_parsers.count(p)) return CUDA_ERROR_INVALID_HANDLE;
  }
  if (!packet) return CUDA_ERROR_INVALID_VALUE;
  if (packet->flags & CUVID_PKT_DISCONTINUITY) p->pending.clear();
  if (packet->payload && packet->payload_size) p->pending.insert(p->pending.end(), packet->payload, packet->payload + packet->payload_size);
  if (packet->flags & CUVID_PKT_TIMESTAMP) {
    p->armed = true;
    p->armed_ts = static_cast<int64_t>(packet->timestamp);
  }
  // A picture's duration is a thirtieth of a second in the parser's clock (10 MHz
  // unless the application sets another).
  const int64_t clock = p->params.ulClockRate ? static_cast<int64_t>(p->params.ulClockRate) : 10000000;
  const int64_t duration = clock / 30;
  for (;;) {
    // Drop bytes before the next SOI.
    size_t soi = 0;
    while (soi + 1 < p->pending.size() && !(p->pending[soi] == 0xFF && p->pending[soi + 1] == 0xD8)) ++soi;
    if (soi) p->pending.erase(p->pending.begin(), p->pending.begin() + static_cast<long>(soi));
    uint32_t w = 0, h = 0;
    int chroma = cudaVideoChromaFormat_420, ncomp = 0;
    const long n = jpeg_end(p->pending, &w, &h, &chroma, &ncomp);
    if (n == -1) {
      p->pending.erase(p->pending.begin());   // not a picture: skip a byte and look again
      continue;
    }
    if (n <= 0) break;
    std::vector<uint8_t> frame(p->pending.begin(), p->pending.begin() + n);
    p->pending.erase(p->pending.begin(), p->pending.begin() + n);
    // A picture with no frame header, or with four components (CMYK), is skipped
    // without a callback, as the card's parser does.
    if (w == 0 || h == 0 || (ncomp != 1 && ncomp != 3)) continue;
    const uint32_t cw = (w + 15) / 16 * 16, chh = (h + 15) / 16 * 16;
    if (!p->have_format || p->format.coded_width != cw || p->format.coded_height != chh ||
        p->format.chroma_format != static_cast<cudaVideoChromaFormat>(chroma) || p->format.display_area.right != static_cast<int>(w) ||
        p->format.display_area.bottom != static_cast<int>(h)) {
      CUVIDEOFORMAT f{};
      f.codec = cudaVideoCodec_JPEG;
      f.progressive_sequence = 1;
      f.min_num_decode_surfaces = 1;
      f.coded_width = cw;
      f.coded_height = chh;
      f.display_area = {0, 0, static_cast<int>(w), static_cast<int>(h)};
      f.chroma_format = static_cast<cudaVideoChromaFormat>(chroma);
      f.video_signal_description.video_full_range_flag = 1;   // JPEG is full range
      p->format = f;
      p->have_format = true;
      if (p->params.pfnSequenceCallback) {
        // The return value is the number of decode surfaces the application made; a
        // value above one widens the cycle the picture index runs through.
        const int ret = p->params.pfnSequenceCallback(p->params.pUserData, &p->format);
        p->surfaces = std::max<unsigned>(std::max<unsigned>(1, p->params.ulMaxNumDecodeSurfaces), ret > 1 ? static_cast<unsigned>(ret) : 1u);
        p->next_index %= p->surfaces;
      }
    }
    int64_t ts;
    if (p->armed) {
      ts = p->armed_ts;
      p->armed = false;
    } else {
      ts = p->have_prev ? p->prev_ts + duration : 0;
    }
    p->prev_ts = ts;
    p->have_prev = true;

    CUVIDPICPARAMS pic{};
    pic.PicWidthInMbs = static_cast<int>(cw / 16);
    pic.FrameHeightInMbs = static_cast<int>(chh / 16);
    pic.CurrPicIdx = static_cast<int>(p->next_index);
    pic.pBitstreamData = frame.data();
    pic.nBitstreamDataLen = static_cast<unsigned>(frame.size());
    pic.nNumSlices = 1;
    const unsigned zero = 0;
    pic.pSliceDataOffsets = &zero;
    pic.intra_pic_flag = 1;
    // The application's decode callback returning 0 withholds the display callback.
    const bool shown = !p->params.pfnDecodePicture || p->params.pfnDecodePicture(p->params.pUserData, &pic) != 0;
    if (shown && p->params.pfnDisplayPicture) {
      CUVIDPARSERDISPINFO disp{};
      disp.picture_index = pic.CurrPicIdx;
      disp.progressive_frame = 1;
      disp.top_field_first = 0;
      disp.repeat_first_field = 0;
      disp.timestamp = ts;
      p->params.pfnDisplayPicture(p->params.pUserData, &disp);
    }
    p->next_index = (p->next_index + 1) % p->surfaces;
  }
  return CUDA_SUCCESS;
}

VGPU_API CUresult CUDAAPI cuvidDestroyVideoParser(CUvideoparser obj) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto* p = static_cast<Parser*>(obj);
  if (!g_parsers.count(p)) return CUDA_ERROR_INVALID_HANDLE;
  g_parsers.erase(p);
  delete p;
  return CUDA_SUCCESS;
}

/* ---- video sources: files and URLs need FFmpeg-grade demuxers ------------- */

VGPU_API CUresult CUDAAPI cuvidCreateVideoSource(CUvideosource*, const char*, CUVIDSOURCEPARAMS*) {
  say_once("video sources (files, URLs) need a demuxer and are not supported by VirtualGPU");
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_API CUresult CUDAAPI cuvidCreateVideoSourceW(CUvideosource*, const wchar_t*, CUVIDSOURCEPARAMS*) {
  say_once("video sources (files, URLs) need a demuxer and are not supported by VirtualGPU");
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_API CUresult CUDAAPI cuvidDestroyVideoSource(CUvideosource) { return CUDA_ERROR_INVALID_HANDLE; }
VGPU_API CUresult CUDAAPI cuvidSetVideoSourceState(CUvideosource, cudaVideoState) { return CUDA_ERROR_INVALID_HANDLE; }
VGPU_API cudaVideoState CUDAAPI cuvidGetVideoSourceState(CUvideosource) { return cudaVideoState_Error; }
VGPU_API CUresult CUDAAPI cuvidGetSourceVideoFormat(CUvideosource, CUVIDEOFORMAT*, unsigned int) { return CUDA_ERROR_INVALID_HANDLE; }
VGPU_API CUresult CUDAAPI cuvidGetSourceAudioFormat(CUvideosource, CUAUDIOFORMAT*, unsigned int) { return CUDA_ERROR_INVALID_HANDLE; }
