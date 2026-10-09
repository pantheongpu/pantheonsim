// libvgpunvcuvid -- VirtualGPU's NVDEC, presented as libnvcuvid.so.1.
//
// NVDEC is fixed-function decode silicon behind the driver; applications reach
// it through libnvcuvid's cuvid* entry points (the video parser and the
// decoder), usually via NVIDIA's own NvDecoder wrapper or FFmpeg's *_cuvid
// decoders. This library answers that API with a software decoder -- for one
// codec. Motion JPEG is a sequence of independent JPEG pictures, and a JPEG
// decoder is already here (the nvJPEG library's), so cudaVideoCodec_JPEG is
// decoded on the host and the NV12 surfaces the application maps are written
// to device memory. Every other codec reports cuvidGetDecoderCaps
// bIsSupported = 0 (what a card without that engine says) and
// cuvidCreateDecoder refuses it: a software H.264 or HEVC decoder is not part
// of this simulator, so an application falls back to its CPU decoder rather
// than receive a wrong picture.
//
// Decoded pixels are the JPEG codec's (the 8-bit baseline, extended and
// progressive Huffman JPEG nvjpeg_api.cpp reads; see its notes on how closely
// it follows NVIDIA's inverse DCT). The picture a decode produces is the
// image in the stream's own chroma format, in an NV12 surface (Y plane, then
// interleaved Cb/Cr at 4:2:0).
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "cuviddec.h"
#include "nvcuvid.h"

// The header renames the 32-bit entry points to the 64-bit ones for 64-bit
// builds; the library exports both.
#undef cuvidMapVideoFrame
#undef cuvidUnmapVideoFrame

#define VGPU_API extern "C" __attribute__((visibility("default")))

namespace {

std::mutex g_mu;

struct Surface {
  void* dev = nullptr;
  uint32_t pitch = 0, rows = 0;   // rows of luma (the chroma plane follows)
  bool decoded = false;
  bool mapped = false;
};

struct Decoder {
  CUVIDDECODECREATEINFO info{};
  nvjpegHandle_t handle = nullptr;
  nvjpegJpegState_t state = nullptr;
  std::vector<Surface> surfaces;
  uint32_t width = 0, height = 0;   // the stream's frame size
};

std::set<Decoder*> g_decoders;

Decoder* find_decoder(CUvideodecoder d) {
  auto* p = static_cast<Decoder*>(d);
  return g_decoders.count(p) ? p : nullptr;
}

// The pitch (bytes per row) of a decoded surface.
uint32_t surface_pitch(uint32_t width) { return (width + 255u) & ~255u; }

}  // namespace

/* ---- capabilities ------------------------------------------------------- */

VGPU_API CUresult CUDAAPI cuvidGetDecoderCaps(CUVIDDECODECAPS* caps) {
  if (!caps) return CUDA_ERROR_INVALID_VALUE;
  caps->bIsSupported = 0;
  caps->nMaxWidth = caps->nMaxHeight = caps->nMaxMBCount = 0;
  caps->nMinWidth = caps->nMinHeight = 0;
  caps->nOutputFormatMask = 0;
  const int chroma = static_cast<int>(caps->eChromaFormat);
  if (caps->eCodecType == cudaVideoCodec_JPEG && chroma >= 0 && chroma <= 3 && caps->nBitDepthMinus8 == 0) {
    caps->bIsSupported = 1;
    caps->nMaxWidth = 32768;
    caps->nMaxHeight = 16384;
    caps->nMaxMBCount = 8388608;
    caps->nMinWidth = 64;
    caps->nMinHeight = 64;
    caps->nOutputFormatMask = 1u << cudaVideoSurfaceFormat_NV12;
  }
  return CUDA_SUCCESS;
}

/* ---- decoder ------------------------------------------------------------ */

VGPU_API CUresult CUDAAPI cuvidCreateDecoder(CUvideodecoder* out, CUVIDDECODECREATEINFO* ci) {
  if (!out || !ci) return CUDA_ERROR_INVALID_VALUE;
  if (ci->CodecType != cudaVideoCodec_JPEG) return CUDA_ERROR_NOT_SUPPORTED;
  if (ci->ChromaFormat > cudaVideoChromaFormat_444 || ci->bitDepthMinus8 != 0) return CUDA_ERROR_NOT_SUPPORTED;
  if (ci->ulWidth == 0 || ci->ulHeight == 0 || ci->ulNumDecodeSurfaces == 0) return CUDA_ERROR_INVALID_VALUE;
  auto* d = new Decoder();
  d->info = *ci;
  d->width = static_cast<uint32_t>(ci->ulWidth);
  d->height = static_cast<uint32_t>(ci->ulHeight);
  if (nvjpegCreateSimple(&d->handle) != NVJPEG_STATUS_SUCCESS || nvjpegJpegStateCreate(d->handle, &d->state) != NVJPEG_STATUS_SUCCESS) {
    delete d;
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  d->surfaces.resize(ci->ulNumDecodeSurfaces);
  std::lock_guard<std::mutex> lock(g_mu);
  g_decoders.insert(d);
  *out = d;
  return CUDA_SUCCESS;
}

VGPU_API CUresult CUDAAPI cuvidDestroyDecoder(CUvideodecoder dec) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  for (Surface& s : d->surfaces)
    if (s.dev) cudaFree(s.dev);
  nvjpegJpegStateDestroy(d->state);
  nvjpegDestroy(d->handle);
  g_decoders.erase(d);
  delete d;
  return CUDA_SUCCESS;
}

VGPU_API CUresult CUDAAPI cuvidDecodePicture(CUvideodecoder dec, CUVIDPICPARAMS* p) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  if (!p || !p->pBitstreamData || p->nBitstreamDataLen == 0) return CUDA_ERROR_INVALID_VALUE;
  if (p->CurrPicIdx < 0 || static_cast<size_t>(p->CurrPicIdx) >= d->surfaces.size()) return CUDA_ERROR_INVALID_VALUE;
  int ncomp = 0, w[NVJPEG_MAX_COMPONENT] = {0}, h[NVJPEG_MAX_COMPONENT] = {0};
  nvjpegChromaSubsampling_t css;
  if (nvjpegGetImageInfo(d->handle, p->pBitstreamData, p->nBitstreamDataLen, &ncomp, &css, w, h) != NVJPEG_STATUS_SUCCESS)
    return CUDA_ERROR_INVALID_IMAGE;
  Surface& s = d->surfaces[p->CurrPicIdx];
  const uint32_t fw = static_cast<uint32_t>(w[0]), fh = static_cast<uint32_t>(h[0]);
  const uint32_t pitch = surface_pitch(fw);
  if (!s.dev || s.pitch != pitch || s.rows != fh) {
    if (s.dev) cudaFree(s.dev);
    s.dev = nullptr;
    if (cudaMalloc(&s.dev, static_cast<size_t>(pitch) * fh * 3 / 2 + pitch) != cudaSuccess) return CUDA_ERROR_OUT_OF_MEMORY;
    s.pitch = pitch;
    s.rows = fh;
  }
  // The planes the JPEG decoder writes (Y, Cb, Cr at their own resolutions).
  nvjpegImage_t img{};
  void* planes[3] = {nullptr, nullptr, nullptr};
  size_t plane_pitch[3] = {0, 0, 0};
  const int planes_n = ncomp >= 3 ? 3 : 1;
  for (int c = 0; c < planes_n; ++c) {
    plane_pitch[c] = (static_cast<size_t>(w[c]) + 255u) & ~255u;
    if (cudaMalloc(&planes[c], plane_pitch[c] * static_cast<size_t>(h[c])) != cudaSuccess) {
      for (int k = 0; k < c; ++k) cudaFree(planes[k]);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    img.channel[c] = static_cast<unsigned char*>(planes[c]);
    img.pitch[c] = plane_pitch[c];
  }
  const nvjpegStatus_t st = nvjpegDecode(d->handle, d->state, p->pBitstreamData, p->nBitstreamDataLen,
                                         planes_n == 3 ? NVJPEG_OUTPUT_YUV : NVJPEG_OUTPUT_Y, &img, 0);
  CUresult result = st == NVJPEG_STATUS_SUCCESS ? CUDA_SUCCESS : CUDA_ERROR_INVALID_IMAGE;
  if (result == CUDA_SUCCESS) {
    // To NV12: luma as it is, chroma interleaved at the half resolution of 4:2:0
    // (a 4:4:4 or 4:2:2 picture is averaged down; a grey one has mid-grey chroma).
    std::vector<uint8_t> luma(static_cast<size_t>(fw) * fh), cb, cr;
    cudaMemcpy2D(luma.data(), fw, planes[0], plane_pitch[0], fw, fh, cudaMemcpyDeviceToHost);
    const uint32_t cw = (fw + 1) / 2, ch = (fh + 1) / 2;
    std::vector<uint8_t> uv(static_cast<size_t>(cw) * 2 * ch, 128);
    if (planes_n == 3) {
      const int sw = w[1], sh = h[1];
      cb.resize(static_cast<size_t>(sw) * sh);
      cr.resize(cb.size());
      cudaMemcpy2D(cb.data(), sw, planes[1], plane_pitch[1], sw, sh, cudaMemcpyDeviceToHost);
      cudaMemcpy2D(cr.data(), sw, planes[2], plane_pitch[2], sw, sh, cudaMemcpyDeviceToHost);
      const int rx = std::max(1, (w[0] + sw - 1) / sw), ry = std::max(1, (h[0] + sh - 1) / sh);   // luma samples per chroma sample
      for (uint32_t y = 0; y < ch; ++y)
        for (uint32_t x = 0; x < cw; ++x) {
          // Average the source chroma samples that fall in this 2x2 luma block.
          int acc_b = 0, acc_r = 0, n = 0;
          for (uint32_t dy = 0; dy < 2; ++dy)
            for (uint32_t dx = 0; dx < 2; ++dx) {
              const uint32_t ly = std::min(y * 2 + dy, fh - 1), lx = std::min(x * 2 + dx, fw - 1);
              const uint32_t sy = std::min<uint32_t>(ly / ry, sh - 1), sx = std::min<uint32_t>(lx / rx, sw - 1);
              acc_b += cb[static_cast<size_t>(sy) * sw + sx];
              acc_r += cr[static_cast<size_t>(sy) * sw + sx];
              ++n;
            }
          uv[(static_cast<size_t>(y) * cw + x) * 2] = static_cast<uint8_t>((acc_b + n / 2) / n);
          uv[(static_cast<size_t>(y) * cw + x) * 2 + 1] = static_cast<uint8_t>((acc_r + n / 2) / n);
        }
    }
    cudaMemcpy2D(s.dev, s.pitch, luma.data(), fw, fw, fh, cudaMemcpyHostToDevice);
    cudaMemcpy2D(static_cast<uint8_t*>(s.dev) + static_cast<size_t>(s.pitch) * fh, s.pitch, uv.data(), cw * 2, cw * 2, ch,
                 cudaMemcpyHostToDevice);
    s.decoded = true;
  }
  for (int c = 0; c < planes_n; ++c) cudaFree(planes[c]);
  return result;
}

VGPU_API CUresult CUDAAPI cuvidGetDecodeStatus(CUvideodecoder dec, int idx, CUVIDGETDECODESTATUS* status) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  if (!status || idx < 0 || static_cast<size_t>(idx) >= d->surfaces.size()) return CUDA_ERROR_INVALID_VALUE;
  status->decodeStatus = d->surfaces[idx].decoded ? cuvidDecodeStatus_Success : cuvidDecodeStatus_Invalid;
  return CUDA_SUCCESS;
}

VGPU_API CUresult CUDAAPI cuvidReconfigureDecoder(CUvideodecoder dec, CUVIDRECONFIGUREDECODERINFO* info) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  if (!info) return CUDA_ERROR_INVALID_VALUE;
  d->width = info->ulWidth;
  d->height = info->ulHeight;
  return CUDA_SUCCESS;
}

static CUresult map_frame(CUvideodecoder dec, int idx, unsigned long long* ptr, unsigned int* pitch) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  if (!ptr || !pitch || idx < 0 || static_cast<size_t>(idx) >= d->surfaces.size()) return CUDA_ERROR_INVALID_VALUE;
  Surface& s = d->surfaces[idx];
  if (!s.decoded || !s.dev) return CUDA_ERROR_INVALID_VALUE;
  s.mapped = true;
  *ptr = reinterpret_cast<unsigned long long>(s.dev);
  *pitch = s.pitch;
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
static CUresult unmap_frame(CUvideodecoder dec, unsigned long long ptr) {
  std::lock_guard<std::mutex> lock(g_mu);
  Decoder* d = find_decoder(dec);
  if (!d) return CUDA_ERROR_INVALID_HANDLE;
  for (Surface& s : d->surfaces)
    if (reinterpret_cast<unsigned long long>(s.dev) == ptr && s.mapped) {
      s.mapped = false;
      return CUDA_SUCCESS;
    }
  return CUDA_ERROR_INVALID_VALUE;
}
VGPU_API CUresult CUDAAPI cuvidUnmapVideoFrame64(CUvideodecoder dec, unsigned long long ptr) { return unmap_frame(dec, ptr); }
VGPU_API CUresult CUDAAPI cuvidUnmapVideoFrame(CUvideodecoder dec, unsigned int ptr) { return unmap_frame(dec, ptr); }

/* ---- context locks ------------------------------------------------------ */

struct VgpuCtxLock {
  std::recursive_mutex mu;
};
VGPU_API CUresult CUDAAPI cuvidCtxLockCreate(CUvideoctxlock* lock, CUcontext) {
  if (!lock) return CUDA_ERROR_INVALID_VALUE;
  *lock = reinterpret_cast<CUvideoctxlock>(new VgpuCtxLock());
  return CUDA_SUCCESS;
}
VGPU_API CUresult CUDAAPI cuvidCtxLockDestroy(CUvideoctxlock lock) {
  delete reinterpret_cast<VgpuCtxLock*>(lock);
  return CUDA_SUCCESS;
}
VGPU_API CUresult CUDAAPI cuvidCtxLock(CUvideoctxlock lock, unsigned int) {
  if (!lock) return CUDA_ERROR_INVALID_VALUE;
  reinterpret_cast<VgpuCtxLock*>(lock)->mu.lock();
  return CUDA_SUCCESS;
}
VGPU_API CUresult CUDAAPI cuvidCtxUnlock(CUvideoctxlock lock, unsigned int) {
  if (!lock) return CUDA_ERROR_INVALID_VALUE;
  reinterpret_cast<VgpuCtxLock*>(lock)->mu.unlock();
  return CUDA_SUCCESS;
}

/* ---- the video parser: Motion JPEG ------------------------------------- */

namespace {

struct Parser {
  CUVIDPARSERPARAMS params{};
  std::vector<uint8_t> pending;   // bytes of a picture still being assembled
  bool have_format = false;
  CUVIDEOFORMAT format{};
  int next_index = 0;
};

std::set<Parser*> g_parsers;

// The length of the JPEG picture at the start of `d` (SOI to EOI), or 0 if it is
// not complete yet; -1 if the bytes are not a JPEG picture.
long jpeg_end(const std::vector<uint8_t>& d, uint32_t* w, uint32_t* h, int* chroma) {
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
      for (int c = 0; c < nc && c < 4 && i + 8 + 3 * c + 2 < d.size(); ++c) {
        hs[c] = d[i + 9 + 3 * c] >> 4;
        vs[c] = d[i + 9 + 3 * c] & 15;
      }
      *chroma = nc == 1 ? cudaVideoChromaFormat_Monochrome
                        : (hs[0] == 1 && vs[0] == 1 ? cudaVideoChromaFormat_444 : (vs[0] == 1 ? cudaVideoChromaFormat_422 : cudaVideoChromaFormat_420));
    }
    i += len;
  }
}

}  // namespace

VGPU_API CUresult CUDAAPI cuvidCreateVideoParser(CUvideoparser* out, CUVIDPARSERPARAMS* params) {
  if (!out || !params) return CUDA_ERROR_INVALID_VALUE;
  if (params->CodecType != cudaVideoCodec_JPEG) return CUDA_ERROR_NOT_SUPPORTED;
  auto* p = new Parser();
  p->params = *params;
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
  if (packet->payload && packet->payload_size) p->pending.insert(p->pending.end(), packet->payload, packet->payload + packet->payload_size);
  for (;;) {
    // Drop bytes before the next SOI.
    size_t soi = 0;
    while (soi + 1 < p->pending.size() && !(p->pending[soi] == 0xFF && p->pending[soi + 1] == 0xD8)) ++soi;
    if (soi) p->pending.erase(p->pending.begin(), p->pending.begin() + soi);
    uint32_t w = 0, h = 0;
    int chroma = cudaVideoChromaFormat_420;
    const long n = jpeg_end(p->pending, &w, &h, &chroma);
    if (n <= 0) break;
    std::vector<uint8_t> frame(p->pending.begin(), p->pending.begin() + n);
    p->pending.erase(p->pending.begin(), p->pending.begin() + n);
    if (!p->have_format || p->format.coded_width != w || p->format.coded_height != h || p->format.chroma_format != static_cast<cudaVideoChromaFormat>(chroma)) {
      CUVIDEOFORMAT f{};
      f.codec = cudaVideoCodec_JPEG;
      f.progressive_sequence = 1;
      f.min_num_decode_surfaces = 1;
      f.coded_width = w;
      f.coded_height = h;
      f.display_area = {0, 0, static_cast<int>(w), static_cast<int>(h)};
      f.chroma_format = static_cast<cudaVideoChromaFormat>(chroma);
      f.display_aspect_ratio = {static_cast<int>(w), static_cast<int>(h)};
      p->format = f;
      p->have_format = true;
      if (p->params.pfnSequenceCallback && p->params.pfnSequenceCallback(p->params.pUserData, &p->format) == 0) return CUDA_ERROR_UNKNOWN;
    }
    CUVIDPICPARAMS pic{};
    pic.PicWidthInMbs = static_cast<int>((w + 15) / 16);
    pic.FrameHeightInMbs = static_cast<int>((h + 15) / 16);
    pic.CurrPicIdx = p->next_index;
    pic.pBitstreamData = frame.data();
    pic.nBitstreamDataLen = static_cast<unsigned>(frame.size());
    pic.nNumSlices = 1;
    const unsigned zero = 0;
    pic.pSliceDataOffsets = &zero;
    pic.intra_pic_flag = 1;
    if (p->params.pfnDecodePicture && p->params.pfnDecodePicture(p->params.pUserData, &pic) == 0) return CUDA_ERROR_UNKNOWN;
    CUVIDPARSERDISPINFO disp{};
    disp.picture_index = p->next_index;
    disp.progressive_frame = 1;
    disp.top_field_first = 0;
    disp.timestamp = (packet->flags & CUVID_PKT_TIMESTAMP) ? packet->timestamp : 0;
    if (p->params.pfnDisplayPicture && p->params.pfnDisplayPicture(p->params.pUserData, &disp) == 0) return CUDA_ERROR_UNKNOWN;
    p->next_index = (p->next_index + 1) % std::max<unsigned>(1, p->params.ulMaxNumDecodeSurfaces);
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

VGPU_API CUresult CUDAAPI cuvidCreateVideoSource(CUvideosource*, const char*, CUVIDSOURCEPARAMS*) { return CUDA_ERROR_NOT_SUPPORTED; }
VGPU_API CUresult CUDAAPI cuvidCreateVideoSourceW(CUvideosource*, const wchar_t*, CUVIDSOURCEPARAMS*) { return CUDA_ERROR_NOT_SUPPORTED; }
VGPU_API CUresult CUDAAPI cuvidDestroyVideoSource(CUvideosource) { return CUDA_ERROR_INVALID_HANDLE; }
VGPU_API CUresult CUDAAPI cuvidSetVideoSourceState(CUvideosource, cudaVideoState) { return CUDA_ERROR_INVALID_HANDLE; }
VGPU_API cudaVideoState CUDAAPI cuvidGetVideoSourceState(CUvideosource) { return cudaVideoState_Error; }
VGPU_API CUresult CUDAAPI cuvidGetSourceVideoFormat(CUvideosource, CUVIDEOFORMAT*, unsigned int) { return CUDA_ERROR_INVALID_HANDLE; }
VGPU_API CUresult CUDAAPI cuvidGetSourceAudioFormat(CUvideosource, CUAUDIOFORMAT*, unsigned int) { return CUDA_ERROR_INVALID_HANDLE; }
