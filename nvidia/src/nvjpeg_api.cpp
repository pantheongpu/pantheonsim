// libvgpunvjpeg -- VirtualGPU's nvJPEG, presented as libnvjpeg.so.12/13.
//
// A baseline JPEG codec, written here rather than delegated: there is no
// dependency in this repository to delegate to, and the format is frozen and
// fully specified (ITU-T T.81). Decoding is what image pipelines actually do
// on a GPU box -- DALI and torchvision's nvjpeg backend both go through this
// API -- so it is the half that matters most.
//
// Implemented: baseline sequential DCT, 8-bit, Huffman coded, one to four
// components, the common chroma subsamplings, restart markers, and encoding
// back out at a chosen quality. Progressive JPEG, 12-bit samples, arithmetic
// coding and lossless mode are rejected by name rather than approximated.
//
// The IDCT is not bit-specified by the standard, so a decode here agrees with
// NVIDIA's to within the rounding of two different but equally valid inverse
// transforms; it is not bit-identical, and the conformance test compares it
// accordingly.
#include <nvjpeg.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool quiet() { const char* q = std::getenv("VGPU_QUIET"); return q && q[0] == '1'; }

constexpr int kZigZag[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

struct HuffTable {
  // code -> value, resolved by walking lengths, which is how JPEG defines it.
  uint8_t bits[17] = {0};
  uint8_t values[256] = {0};
  int mincode[17] = {0}, maxcode[18] = {0}, valptr[17] = {0};
  bool present = false;

  void build() {
    int code = 0, k = 0;
    for (int l = 1; l <= 16; ++l) {
      valptr[l] = k;
      mincode[l] = code;
      code += bits[l];
      k += bits[l];
      maxcode[l] = code - 1;
      if (bits[l] == 0) maxcode[l] = -1;
      code <<= 1;
    }
    maxcode[17] = 0x7FFFFFFF;
    present = true;
  }
};

struct Component {
  int id = 0, h = 1, v = 1, tq = 0;
  int td = 0, ta = 0;
  int dc_pred = 0;
  int blocks_w = 0, blocks_h = 0;   // in 8x8 blocks, padded to the MCU grid
  std::vector<uint8_t> plane;       // blocks_w*8 by blocks_h*8
};

struct Frame {
  int width = 0, height = 0;
  int hmax = 1, vmax = 1;
  int mcu_w = 8, mcu_h = 8, mcus_x = 0, mcus_y = 0;
  int restart_interval = 0;
  std::vector<Component> comps;
  uint16_t quant[4][64] = {{0}};
  HuffTable dc[4], ac[4];
  bool progressive = false;
};

// Reads entropy-coded data a bit at a time, unstuffing the 0xFF 0x00 pairs the
// standard requires and stopping cleanly at the next marker.
struct BitReader {
  const uint8_t* p;
  const uint8_t* end;
  uint32_t buf = 0;
  int count = 0;
  bool hit_marker = false;

  int bit() {
    if (count == 0) {
      if (p >= end) { hit_marker = true; return 0; }
      uint8_t b = *p++;
      if (b == 0xFF) {
        if (p < end && *p == 0x00) {
          ++p;
        } else {
          hit_marker = true;
          return 0;
        }
      }
      buf = b;
      count = 8;
    }
    --count;
    return (buf >> count) & 1;
  }
  // A Huffman table in a corrupt file can decode to a symbol asking for far
  // more bits than an int holds. JPEG never needs more than 16.
  int bits(int n) {
    if (n <= 0 || n > 16) { hit_marker = true; return 0; }
    int v = 0;
    for (int i = 0; i < n; ++i) v = (v << 1) | bit();
    return v;
  }
  void align() { count = 0; }
};

int huff_decode(BitReader& br, const HuffTable& t) {
  int code = br.bit();
  int l = 1;
  while (l <= 16 && (t.maxcode[l] < 0 || code > t.maxcode[l])) {
    code = (code << 1) | br.bit();
    ++l;
  }
  if (l > 16) return 0;
  return t.values[t.valptr[l] + code - t.mincode[l]];
}

// The sign convention of JPEG's variable-length integers: values whose top bit
// is clear are negative.
int extend(int v, int n) {
  if (n <= 0 || n > 16) return 0;   // the caller has already rejected the symbol
  return v < (1 << (n - 1)) ? v - (1 << n) + 1 : v;
}

// A DCT coefficient fits comfortably in 16 bits for any real image; clamping
// keeps a corrupt file from overflowing the dequantisation multiply, which is
// signed and therefore undefined rather than merely wrong.
int clamp_coefficient(int v) {
  constexpr int kLimit = 1 << 20;
  return v < -kLimit ? -kLimit : (v > kLimit ? kLimit : v);
}

// Straightforward separable inverse DCT. Slower than the fast integer variants
// and, unlike them, exact to double precision -- which is the right trade for a
// simulator whose job is to be correct rather than quick.
void idct8x8(const int* in, uint8_t* out, int stride) {
  static double c[8][8];
  static bool init = false;
  if (!init) {
    for (int u = 0; u < 8; ++u)
      for (int x = 0; x < 8; ++x)
        c[u][x] = (u == 0 ? std::sqrt(0.125) : 0.5) * std::cos((2 * x + 1) * u * M_PI / 16.0);
    init = true;
  }
  double tmp[64];
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      double s = 0;
      for (int u = 0; u < 8; ++u) s += c[u][x] * in[y * 8 + u];
      tmp[y * 8 + x] = s;
    }
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y) {
      double s = 0;
      for (int v = 0; v < 8; ++v) s += c[v][y] * tmp[v * 8 + x];
      const double r = std::round(s) + 128.0;
      out[y * stride + x] = static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
}

void fdct8x8(const double* in, double* out) {
  static double c[8][8];
  static bool init = false;
  if (!init) {
    for (int u = 0; u < 8; ++u)
      for (int x = 0; x < 8; ++x)
        c[u][x] = (u == 0 ? std::sqrt(0.125) : 0.5) * std::cos((2 * x + 1) * u * M_PI / 16.0);
    init = true;
  }
  double tmp[64];
  for (int y = 0; y < 8; ++y)
    for (int u = 0; u < 8; ++u) {
      double s = 0;
      for (int x = 0; x < 8; ++x) s += c[u][x] * in[y * 8 + x];
      tmp[y * 8 + u] = s;
    }
  for (int u = 0; u < 8; ++u)
    for (int v = 0; v < 8; ++v) {
      double s = 0;
      for (int y = 0; y < 8; ++y) s += c[v][y] * tmp[y * 8 + u];
      out[v * 8 + u] = s;
    }
}

nvjpegStatus_t parse_and_decode(const uint8_t* data, size_t len, Frame* f, bool headers_only) {
  if (!data || len < 4) return NVJPEG_STATUS_BAD_JPEG;
  size_t i = 0;
  auto u16 = [&](size_t at) { return static_cast<int>(data[at]) << 8 | data[at + 1]; };
  if (data[0] != 0xFF || data[1] != 0xD8) return NVJPEG_STATUS_BAD_JPEG;
  i = 2;

  while (i + 1 < len) {
    if (data[i] != 0xFF) { ++i; continue; }
    uint8_t marker = data[i + 1];
    i += 2;
    if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;
    if (marker == 0xD9) break;
    if (i + 1 >= len) return NVJPEG_STATUS_BAD_JPEG;
    const int seglen = u16(i);
    if (seglen < 2 || i + seglen > len) return NVJPEG_STATUS_BAD_JPEG;
    const uint8_t* seg = data + i + 2;
    const int segn = seglen - 2;

    switch (marker) {
      case 0xC0: case 0xC1: {  // baseline / extended sequential
        if (segn < 6) return NVJPEG_STATUS_BAD_JPEG;
        if (seg[0] != 8) return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;  // 8-bit only
        f->height = seg[1] << 8 | seg[2];
        f->width = seg[3] << 8 | seg[4];
        const int nc = seg[5];
        if (nc < 1 || nc > 4 || segn < 6 + 3 * nc) return NVJPEG_STATUS_BAD_JPEG;
        f->comps.resize(nc);
        for (int c = 0; c < nc; ++c) {
          f->comps[c].id = seg[6 + c * 3];
          f->comps[c].h = seg[7 + c * 3] >> 4;
          f->comps[c].v = seg[7 + c * 3] & 0xF;
          f->comps[c].tq = seg[8 + c * 3];
          if (f->comps[c].h < 1 || f->comps[c].h > 4 || f->comps[c].v < 1 || f->comps[c].v > 4)
            return NVJPEG_STATUS_BAD_JPEG;
          // Selectors index fixed four-entry tables; the file chooses the value.
          if (f->comps[c].tq > 3) return NVJPEG_STATUS_BAD_JPEG;
          f->hmax = std::max(f->hmax, f->comps[c].h);
          f->vmax = std::max(f->vmax, f->comps[c].v);
        }
        // A header can declare 65535x65535 with four components, which is
        // 17 GB of planes. Refuse it here rather than in the allocator.
        if (f->width <= 0 || f->height <= 0 ||
            static_cast<int64_t>(f->width) * f->height > 268435456LL)
          return NVJPEG_STATUS_BAD_JPEG;
        f->mcu_w = f->hmax * 8;
        f->mcu_h = f->vmax * 8;
        f->mcus_x = (f->width + f->mcu_w - 1) / f->mcu_w;
        f->mcus_y = (f->height + f->mcu_h - 1) / f->mcu_h;
        for (auto& c : f->comps) {
          c.blocks_w = f->mcus_x * c.h;
          c.blocks_h = f->mcus_y * c.v;
          c.plane.assign(static_cast<size_t>(c.blocks_w) * 8 * c.blocks_h * 8, 0);
        }
        break;
      }
      case 0xC2:
        f->progressive = true;
        return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
      case 0xC3: case 0xC5: case 0xC6: case 0xC7:
      case 0xC9: case 0xCA: case 0xCB: case 0xCD: case 0xCE: case 0xCF:
        return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;  // lossless, arithmetic, hierarchical
      case 0xC4: {  // DHT
        int p = 0;
        while (p < segn) {
          if (p + 17 > segn) return NVJPEG_STATUS_BAD_JPEG;
          const int tc = seg[p] >> 4, th = seg[p] & 0xF;
          if (th > 3 || tc > 1) return NVJPEG_STATUS_BAD_JPEG;
          HuffTable& t = tc == 0 ? f->dc[th] : f->ac[th];
          int total = 0;
          for (int l = 1; l <= 16; ++l) { t.bits[l] = seg[p + l]; total += t.bits[l]; }
          if (p + 17 + total > segn || total > 256) return NVJPEG_STATUS_BAD_JPEG;
          std::memcpy(t.values, seg + p + 17, total);
          t.build();
          p += 17 + total;
        }
        break;
      }
      case 0xDB: {  // DQT
        int p = 0;
        while (p < segn) {
          const int pq = seg[p] >> 4, tq = seg[p] & 0xF;
          if (tq > 3) return NVJPEG_STATUS_BAD_JPEG;
          ++p;
          for (int k = 0; k < 64; ++k) {
            if (pq) {
              if (p + 1 >= segn) return NVJPEG_STATUS_BAD_JPEG;
              f->quant[tq][k] = static_cast<uint16_t>(seg[p] << 8 | seg[p + 1]);
              p += 2;
            } else {
              if (p >= segn) return NVJPEG_STATUS_BAD_JPEG;
              f->quant[tq][k] = seg[p++];
            }
          }
        }
        break;
      }
      case 0xDD:
        if (segn < 2) return NVJPEG_STATUS_BAD_JPEG;
        f->restart_interval = seg[0] << 8 | seg[1];
        break;
      case 0xDA: {  // SOS -- the entropy-coded data follows the segment
        if (f->comps.empty()) return NVJPEG_STATUS_BAD_JPEG;
        if (headers_only) return NVJPEG_STATUS_SUCCESS;
        if (segn < 1) return NVJPEG_STATUS_BAD_JPEG;
        const int ns = seg[0];
        if (segn < 1 + 2 * ns) return NVJPEG_STATUS_BAD_JPEG;
        std::vector<Component*> scan;
        for (int s = 0; s < ns; ++s) {
          const int cid = seg[1 + s * 2];
          Component* found = nullptr;
          for (auto& c : f->comps)
            if (c.id == cid) found = &c;
          if (!found) return NVJPEG_STATUS_BAD_JPEG;
          found->td = seg[2 + s * 2] >> 4;
          found->ta = seg[2 + s * 2] & 0xF;
          if (found->td > 3 || found->ta > 3) return NVJPEG_STATUS_BAD_JPEG;
          scan.push_back(found);
        }
        BitReader br{data + i + seglen, data + len};
        for (auto& c : f->comps) c.dc_pred = 0;
        int block[64];
        int mcu_count = 0;
        for (int my = 0; my < f->mcus_y; ++my)
          for (int mx = 0; mx < f->mcus_x; ++mx) {
            if (f->restart_interval && mcu_count && mcu_count % f->restart_interval == 0) {
              // Skip the RSTn marker and restart the DC predictors.
              br.align();
              const uint8_t* q = br.p;
              while (q + 1 < br.end && !(q[0] == 0xFF && q[1] >= 0xD0 && q[1] <= 0xD7)) ++q;
              if (q + 1 < br.end) br.p = q + 2;
              br.hit_marker = false;
              for (auto* c : scan) c->dc_pred = 0;
            }
            ++mcu_count;
            for (auto* c : scan) {
              for (int by = 0; by < c->v; ++by)
                for (int bx = 0; bx < c->h; ++bx) {
                  std::memset(block, 0, sizeof block);
                  const HuffTable& dct = f->dc[c->td];
                  const HuffTable& act = f->ac[c->ta];
                  if (!dct.present || !act.present) return NVJPEG_STATUS_BAD_JPEG;
                  const int t = huff_decode(br, dct);
                  if (t > 16) return NVJPEG_STATUS_BAD_JPEG;
                  const int diff = t ? extend(br.bits(t), t) : 0;
                  // Coefficients are bounded in a well-formed file; in a
                  // corrupt one the predictor would otherwise run away and
                  // overflow the multiply below.
                  c->dc_pred = clamp_coefficient(c->dc_pred + diff);
                  block[0] = c->dc_pred * f->quant[c->tq][0];
                  for (int k = 1; k < 64;) {
                    const int rs = huff_decode(br, act);
                    const int r = rs >> 4, sbits = rs & 0xF;
                    if (sbits == 0) {
                      if (r != 15) break;   // EOB
                      k += 16;
                      continue;
                    }
                    k += r;
                    if (k > 63) break;
                    block[kZigZag[k]] =
                        clamp_coefficient(extend(br.bits(sbits), sbits)) * f->quant[c->tq][k];
                    ++k;
                  }
                  const int px = (mx * c->h + bx) * 8, py = (my * c->v + by) * 8;
                  idct8x8(block, c->plane.data() + static_cast<size_t>(py) * c->blocks_w * 8 + px,
                          c->blocks_w * 8);
                }
            }
          }
        return NVJPEG_STATUS_SUCCESS;
      }
      default:
        break;   // APPn, COM and friends carry nothing this decoder needs
    }
    i += seglen;
  }
  return f->comps.empty() ? NVJPEG_STATUS_BAD_JPEG : NVJPEG_STATUS_SUCCESS;
}

nvjpegChromaSubsampling_t subsampling_of(const Frame& f) {
  if (f.comps.size() == 1) return NVJPEG_CSS_GRAY;
  if (f.comps.size() != 3) return NVJPEG_CSS_UNKNOWN;
  const int h = f.comps[0].h / f.comps[1].h, v = f.comps[0].v / f.comps[1].v;
  if (h == 1 && v == 1) return NVJPEG_CSS_444;
  if (h == 2 && v == 1) return NVJPEG_CSS_422;
  if (h == 2 && v == 2) return NVJPEG_CSS_420;
  if (h == 1 && v == 2) return NVJPEG_CSS_440;
  if (h == 4 && v == 1) return NVJPEG_CSS_411;
  if (h == 4 && v == 2) return NVJPEG_CSS_410;
  return NVJPEG_CSS_UNKNOWN;
}

// Nearest-neighbour chroma upsampling, which is what the standard's
// "replication" describes and what a baseline decoder is expected to do.
uint8_t sample(const Component& c, const Frame& f, int x, int y) {
  const int sx = std::min(x * c.h / f.hmax, c.blocks_w * 8 - 1);
  const int sy = std::min(y * c.v / f.vmax, c.blocks_h * 8 - 1);
  return c.plane[static_cast<size_t>(sy) * c.blocks_w * 8 + sx];
}

void ycbcr_to_rgb(double Y, double Cb, double Cr, uint8_t* r, uint8_t* g, uint8_t* b) {
  auto clamp = [](double v) {
    const double x = std::round(v);
    return static_cast<uint8_t>(x < 0 ? 0 : (x > 255 ? 255 : x));
  };
  *r = clamp(Y + 1.402 * (Cr - 128.0));
  *g = clamp(Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0));
  *b = clamp(Y + 1.772 * (Cb - 128.0));
}

struct Handle { int unused = 0; };
struct State { int unused = 0; };
struct EncoderState { std::string bitstream; };
struct EncoderParams {
  int quality = 70;
  nvjpegChromaSubsampling_t css = NVJPEG_CSS_444;
};

std::mutex g_mu;
std::set<const void*> g_live;
template <class T> T* track(T* p) { std::lock_guard<std::mutex> l(g_mu); g_live.insert(p); return p; }
bool known(const void* p) { std::lock_guard<std::mutex> l(g_mu); return p && g_live.count(p); }
void untrack(const void* p) { std::lock_guard<std::mutex> l(g_mu); g_live.erase(p); }

void write_plane(unsigned char* dst, size_t pitch, const std::vector<uint8_t>& src, int w, int h) {
  if (!dst || !w || !h) return;
  for (int y = 0; y < h; ++y)
    cudaMemcpy(dst + static_cast<size_t>(y) * pitch, src.data() + static_cast<size_t>(y) * w, w,
               cudaMemcpyHostToDevice);
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- handles ---- */

VGPU_EXPORT nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t* handle) {
  if (!handle) return NVJPEG_STATUS_INVALID_PARAMETER;
  *handle = reinterpret_cast<nvjpegHandle_t>(track(new Handle()));
  if (!quiet())
    std::fprintf(stderr, "[vgpu] nvJPEG ready (host baseline codec; see nvidia/docs/libraries.md)\n");
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegCreate(nvjpegBackend_t, nvjpegDevAllocator_t*,
                                        nvjpegHandle_t* handle) {
  return nvjpegCreateSimple(handle);
}
VGPU_EXPORT nvjpegStatus_t nvjpegCreateEx(nvjpegBackend_t, nvjpegDevAllocator_t*,
                                          nvjpegPinnedAllocator_t*, unsigned int,
                                          nvjpegHandle_t* handle) {
  return nvjpegCreateSimple(handle);
}
VGPU_EXPORT nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t h) {
  if (!known(h)) return NVJPEG_STATUS_INVALID_PARAMETER;
  untrack(h); delete reinterpret_cast<Handle*>(h);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStateCreate(nvjpegHandle_t h, nvjpegJpegState_t* s) {
  if (!known(h) || !s) return NVJPEG_STATUS_INVALID_PARAMETER;
  *s = reinterpret_cast<nvjpegJpegState_t>(track(new State()));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStateDestroy(nvjpegJpegState_t s) {
  if (!known(s)) return NVJPEG_STATUS_INVALID_PARAMETER;
  untrack(s); delete reinterpret_cast<State*>(s);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegGetProperty(libraryPropertyType type, int* value) {
  if (!value) return NVJPEG_STATUS_INVALID_PARAMETER;
  switch (type) {
    case MAJOR_VERSION: *value = NVJPEG_VER_MAJOR; break;
    case MINOR_VERSION: *value = NVJPEG_VER_MINOR; break;
    case PATCH_LEVEL: *value = NVJPEG_VER_PATCH; break;
    default: return NVJPEG_STATUS_INVALID_PARAMETER;
  }
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegGetCudartProperty(libraryPropertyType type, int* value) {
  return nvjpegGetProperty(type, value);
}
VGPU_EXPORT nvjpegStatus_t nvjpegSetDeviceMemoryPadding(size_t, nvjpegHandle_t) {
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegSetPinnedMemoryPadding(size_t, nvjpegHandle_t) {
  return NVJPEG_STATUS_SUCCESS;
}

/* ---- decode ---- */

VGPU_EXPORT nvjpegStatus_t nvjpegGetImageInfo(nvjpegHandle_t h, const unsigned char* data,
                                              size_t length, int* ncomp,
                                              nvjpegChromaSubsampling_t* css, int* widths,
                                              int* heights) {
  if (!known(h)) return NVJPEG_STATUS_INVALID_PARAMETER;
  Frame f;
  const nvjpegStatus_t st = parse_and_decode(data, length, &f, /*headers_only=*/true);
  if (st != NVJPEG_STATUS_SUCCESS) return st;
  if (ncomp) *ncomp = static_cast<int>(f.comps.size());
  if (css) *css = subsampling_of(f);
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; ++c) {
    const bool have = c < static_cast<int>(f.comps.size());
    if (widths)
      widths[c] = have ? (f.width * f.comps[c].h + f.hmax - 1) / f.hmax : 0;
    if (heights)
      heights[c] = have ? (f.height * f.comps[c].v + f.vmax - 1) / f.vmax : 0;
  }
  return NVJPEG_STATUS_SUCCESS;
}

VGPU_EXPORT nvjpegStatus_t nvjpegDecode(nvjpegHandle_t h, nvjpegJpegState_t s,
                                        const unsigned char* data, size_t length,
                                        nvjpegOutputFormat_t fmt, nvjpegImage_t* dst,
                                        cudaStream_t stream) {
  if (!known(h) || !known(s) || !dst) return NVJPEG_STATUS_INVALID_PARAMETER;
  Frame f;
  const nvjpegStatus_t st = parse_and_decode(data, length, &f, /*headers_only=*/false);
  if (st != NVJPEG_STATUS_SUCCESS) return st;
  cudaStreamSynchronize(stream);
  const int w = f.width, hgt = f.height;
  const bool gray = f.comps.size() == 1;

  auto luma = [&](int x, int y) { return sample(f.comps[0], f, x, y); };

  switch (fmt) {
    case NVJPEG_OUTPUT_Y: {
      std::vector<uint8_t> plane(static_cast<size_t>(w) * hgt);
      for (int y = 0; y < hgt; ++y)
        for (int x = 0; x < w; ++x) plane[static_cast<size_t>(y) * w + x] = luma(x, y);
      write_plane(dst->channel[0], dst->pitch[0], plane, w, hgt);
      return NVJPEG_STATUS_SUCCESS;
    }
    case NVJPEG_OUTPUT_RGB: case NVJPEG_OUTPUT_BGR:
    case NVJPEG_OUTPUT_RGBI: case NVJPEG_OUTPUT_BGRI: {
      const bool interleaved = fmt == NVJPEG_OUTPUT_RGBI || fmt == NVJPEG_OUTPUT_BGRI;
      const bool bgr = fmt == NVJPEG_OUTPUT_BGR || fmt == NVJPEG_OUTPUT_BGRI;
      std::vector<uint8_t> r(static_cast<size_t>(w) * hgt), g(r.size()), b(r.size());
      for (int y = 0; y < hgt; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t o = static_cast<size_t>(y) * w + x;
          if (gray) {
            r[o] = g[o] = b[o] = luma(x, y);
          } else {
            ycbcr_to_rgb(luma(x, y), sample(f.comps[1], f, x, y), sample(f.comps[2], f, x, y),
                         &r[o], &g[o], &b[o]);
          }
        }
      if (interleaved) {
        std::vector<uint8_t> il(r.size() * 3);
        for (size_t o = 0; o < r.size(); ++o) {
          il[o * 3 + 0] = bgr ? b[o] : r[o];
          il[o * 3 + 1] = g[o];
          il[o * 3 + 2] = bgr ? r[o] : b[o];
        }
        write_plane(dst->channel[0], dst->pitch[0], il, w * 3, hgt);
      } else {
        write_plane(dst->channel[0], dst->pitch[0], bgr ? b : r, w, hgt);
        write_plane(dst->channel[1], dst->pitch[1], g, w, hgt);
        write_plane(dst->channel[2], dst->pitch[2], bgr ? r : b, w, hgt);
      }
      return NVJPEG_STATUS_SUCCESS;
    }
    case NVJPEG_OUTPUT_YUV: case NVJPEG_OUTPUT_UNCHANGED: {
      for (size_t c = 0; c < f.comps.size(); ++c) {
        const Component& comp = f.comps[c];
        const int cw = (w * comp.h + f.hmax - 1) / f.hmax;
        const int ch = (hgt * comp.v + f.vmax - 1) / f.vmax;
        std::vector<uint8_t> plane(static_cast<size_t>(cw) * ch);
        for (int y = 0; y < ch; ++y)
          for (int x = 0; x < cw; ++x)
            plane[static_cast<size_t>(y) * cw + x] =
                comp.plane[static_cast<size_t>(y) * comp.blocks_w * 8 + x];
        write_plane(dst->channel[c], dst->pitch[c], plane, cw, ch);
      }
      return NVJPEG_STATUS_SUCCESS;
    }
    default:
      std::fprintf(stderr, "[vgpu] nvjpeg: output format %d is not implemented\n", (int)fmt);
      return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
  }
}

/* ---- encode ----
   Baseline JPEG with the standard Annex K Huffman tables and quality-scaled
   quantisation, which is what libjpeg and nvJPEG both emit by default. The
   bitstream will not be byte-identical to NVIDIA's -- two encoders make
   different, equally legal choices -- so the conformance test decodes what this
   produces rather than comparing bytes. */

namespace {

// Annex K's example luminance and chrominance quantisation tables, in zig-zag
// order, and the standard Huffman code tables.
const uint8_t kQLum[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
const uint8_t kQChrom[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

const uint8_t kDcLumBits[17] = {0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
const uint8_t kDcLumVals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t kDcChrBits[17] = {0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
const uint8_t kDcChrVals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t kAcLumBits[17] = {0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
const uint8_t kAcLumVals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,
    0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
    0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
    0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
const uint8_t kAcChrBits[17] = {0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
const uint8_t kAcChrVals[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33,
    0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
    0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63,
    0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

struct Encoder {
  std::string out;
  uint32_t bit_buf = 0;
  int bit_count = 0;

  void byte(uint8_t b) { out.push_back(static_cast<char>(b)); }
  void word(int v) { byte(static_cast<uint8_t>(v >> 8)); byte(static_cast<uint8_t>(v & 0xFF)); }
  void marker(uint8_t m) { byte(0xFF); byte(m); }
  void put_bits(int code, int len) {
    for (int i = len - 1; i >= 0; --i) {
      bit_buf = (bit_buf << 1) | ((code >> i) & 1);
      if (++bit_count == 8) {
        const uint8_t b = static_cast<uint8_t>(bit_buf & 0xFF);
        byte(b);
        if (b == 0xFF) byte(0x00);   // stuffing, so the byte is not read as a marker
        bit_buf = 0;
        bit_count = 0;
      }
    }
  }
  void flush_bits() {
    while (bit_count) put_bits(1, 1);   // pad with ones, as the standard says
  }
};

// Turns a (bits, values) table into code/length pairs indexed by symbol.
struct CodeTable {
  int code[256] = {0};
  int len[256] = {0};
  CodeTable(const uint8_t* bits, const uint8_t* vals, int nvals) {
    int c = 0, k = 0;
    for (int l = 1; l <= 16; ++l) {
      for (int j = 0; j < bits[l]; ++j) {
        if (k < nvals) { code[vals[k]] = c; len[vals[k]] = l; }
        ++c; ++k;
      }
      c <<= 1;
    }
  }
};

int magnitude(int v) {
  int n = 0;
  int a = v < 0 ? -v : v;
  while (a) { ++n; a >>= 1; }
  return n;
}

void scale_quant(const uint8_t* base, int quality, uint8_t* out) {
  const int q = quality < 1 ? 1 : (quality > 100 ? 100 : quality);
  const int scale = q < 50 ? 5000 / q : 200 - q * 2;
  for (int i = 0; i < 64; ++i) {
    int v = (base[i] * scale + 50) / 100;
    out[i] = static_cast<uint8_t>(v < 1 ? 1 : (v > 255 ? 255 : v));
  }
}

}  // namespace

VGPU_EXPORT nvjpegStatus_t nvjpegEncoderStateCreate(nvjpegHandle_t h, nvjpegEncoderState_t* st,
                                                    cudaStream_t) {
  if (!known(h) || !st) return NVJPEG_STATUS_INVALID_PARAMETER;
  *st = reinterpret_cast<nvjpegEncoderState_t>(track(new EncoderState()));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderStateDestroy(nvjpegEncoderState_t st) {
  if (!known(st)) return NVJPEG_STATUS_INVALID_PARAMETER;
  untrack(st); delete reinterpret_cast<EncoderState*>(st);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsCreate(nvjpegHandle_t h, nvjpegEncoderParams_t* p,
                                                     cudaStream_t) {
  if (!known(h) || !p) return NVJPEG_STATUS_INVALID_PARAMETER;
  *p = reinterpret_cast<nvjpegEncoderParams_t>(track(new EncoderParams()));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsDestroy(nvjpegEncoderParams_t p) {
  if (!known(p)) return NVJPEG_STATUS_INVALID_PARAMETER;
  untrack(p); delete reinterpret_cast<EncoderParams*>(p);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetQuality(nvjpegEncoderParams_t p, const int q,
                                                         cudaStream_t) {
  if (!known(p)) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<EncoderParams*>(p)->quality = q;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetSamplingFactors(nvjpegEncoderParams_t p,
                                                                 const nvjpegChromaSubsampling_t c,
                                                                 cudaStream_t) {
  if (!known(p)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (c != NVJPEG_CSS_444 && c != NVJPEG_CSS_420 && c != NVJPEG_CSS_422 && c != NVJPEG_CSS_GRAY)
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
  reinterpret_cast<EncoderParams*>(p)->css = c;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetOptimizedHuffman(nvjpegEncoderParams_t p, int,
                                                                  cudaStream_t) {
  // The standard tables are always legal; an encoder is free to ignore this.
  return known(p) ? NVJPEG_STATUS_SUCCESS : NVJPEG_STATUS_INVALID_PARAMETER;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetRestartInterval(nvjpegEncoderParams_t p, unsigned int,
                                                                 cudaStream_t) {
  return known(p) ? NVJPEG_STATUS_SUCCESS : NVJPEG_STATUS_INVALID_PARAMETER;
}

VGPU_EXPORT nvjpegStatus_t nvjpegEncodeImage(nvjpegHandle_t h, nvjpegEncoderState_t st,
                                             const nvjpegEncoderParams_t pp,
                                             const nvjpegImage_t* src, nvjpegInputFormat_t ifmt,
                                             int width, int height, cudaStream_t stream) {
  if (!known(h) || !known(st) || !known(pp) || !src) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (width <= 0 || height <= 0) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& state = *reinterpret_cast<EncoderState*>(st);
  const auto& params = *reinterpret_cast<const EncoderParams*>(pp);
  cudaStreamSynchronize(stream);

  // Pull the source into planar R, G, B on the host.
  const size_t n = static_cast<size_t>(width) * height;
  std::vector<uint8_t> R(n), G(n), B(n);
  auto read_plane = [&](int idx, std::vector<uint8_t>* dst, int w) {
    dst->assign(static_cast<size_t>(w) * height, 0);
    for (int y = 0; y < height; ++y)
      cudaMemcpy(dst->data() + static_cast<size_t>(y) * w,
                 src->channel[idx] + static_cast<size_t>(y) * src->pitch[idx], w,
                 cudaMemcpyDeviceToHost);
  };
  if (ifmt == NVJPEG_INPUT_RGBI || ifmt == NVJPEG_INPUT_BGRI) {
    std::vector<uint8_t> il;
    read_plane(0, &il, width * 3);
    const bool bgr = ifmt == NVJPEG_INPUT_BGRI;
    for (size_t i = 0; i < n; ++i) {
      R[i] = il[i * 3 + (bgr ? 2 : 0)];
      G[i] = il[i * 3 + 1];
      B[i] = il[i * 3 + (bgr ? 0 : 2)];
    }
  } else if (ifmt == NVJPEG_INPUT_RGB || ifmt == NVJPEG_INPUT_BGR) {
    const bool bgr = ifmt == NVJPEG_INPUT_BGR;
    read_plane(0, bgr ? &B : &R, width);
    read_plane(1, &G, width);
    read_plane(2, bgr ? &R : &B, width);
  } else {
    std::fprintf(stderr, "[vgpu] nvjpeg: input format %d is not implemented\n", (int)ifmt);
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
  }

  const bool gray = params.css == NVJPEG_CSS_GRAY;
  const int hs = (params.css == NVJPEG_CSS_420 || params.css == NVJPEG_CSS_422) ? 2 : 1;
  const int vs = params.css == NVJPEG_CSS_420 ? 2 : 1;
  const int ncomp = gray ? 1 : 3;

  std::vector<double> Y(n), Cb(n), Cr(n);
  for (size_t i = 0; i < n; ++i) {
    Y[i] = 0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i];
    Cb[i] = 128.0 - 0.168736 * R[i] - 0.331264 * G[i] + 0.5 * B[i];
    Cr[i] = 128.0 + 0.5 * R[i] - 0.418688 * G[i] - 0.081312 * B[i];
  }

  uint8_t qlum[64], qchr[64];
  scale_quant(kQLum, params.quality, qlum);
  scale_quant(kQChrom, params.quality, qchr);

  Encoder e;
  e.marker(0xD8);
  // JFIF APP0, which is what every decoder expects to see first.
  e.marker(0xE0); e.word(16);
  e.byte('J'); e.byte('F'); e.byte('I'); e.byte('F'); e.byte(0);
  e.byte(1); e.byte(1); e.byte(0);
  e.word(1); e.word(1); e.byte(0); e.byte(0);

  e.marker(0xDB); e.word(2 + (gray ? 65 : 130));
  e.byte(0); for (int i = 0; i < 64; ++i) e.byte(qlum[i]);
  if (!gray) { e.byte(1); for (int i = 0; i < 64; ++i) e.byte(qchr[i]); }

  e.marker(0xC0); e.word(8 + 3 * ncomp);
  e.byte(8); e.word(height); e.word(width); e.byte(ncomp);
  e.byte(1); e.byte(static_cast<uint8_t>((hs << 4) | vs)); e.byte(0);
  if (!gray) {
    e.byte(2); e.byte(0x11); e.byte(1);
    e.byte(3); e.byte(0x11); e.byte(1);
  }

  auto emit_dht = [&](int tc, int th, const uint8_t* bits, const uint8_t* vals, int nvals) {
    e.marker(0xC4); e.word(3 + 16 + nvals);
    e.byte(static_cast<uint8_t>((tc << 4) | th));
    for (int l = 1; l <= 16; ++l) e.byte(bits[l]);
    for (int k = 0; k < nvals; ++k) e.byte(vals[k]);
  };
  emit_dht(0, 0, kDcLumBits, kDcLumVals, 12);
  emit_dht(1, 0, kAcLumBits, kAcLumVals, 162);
  if (!gray) {
    emit_dht(0, 1, kDcChrBits, kDcChrVals, 12);
    emit_dht(1, 1, kAcChrBits, kAcChrVals, 162);
  }

  e.marker(0xDA); e.word(6 + 2 * ncomp);
  e.byte(ncomp);
  e.byte(1); e.byte(0x00);
  if (!gray) { e.byte(2); e.byte(0x11); e.byte(3); e.byte(0x11); }
  e.byte(0); e.byte(63); e.byte(0);

  const CodeTable dc_lum(kDcLumBits, kDcLumVals, 12), ac_lum(kAcLumBits, kAcLumVals, 162);
  const CodeTable dc_chr(kDcChrBits, kDcChrVals, 12), ac_chr(kAcChrBits, kAcChrVals, 162);

  const int mcu_w = 8 * hs, mcu_h = 8 * vs;
  const int mcus_x = (width + mcu_w - 1) / mcu_w, mcus_y = (height + mcu_h - 1) / mcu_h;
  int pred[3] = {0, 0, 0};

  auto encode_block = [&](const std::vector<double>& src, int px, int py, int sx, int sy,
                          const uint8_t* qt, const CodeTable& dct, const CodeTable& act,
                          int& predictor) {
    double blk[64];
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        // Sample the (possibly subsampled) plane, clamping at the image edge --
        // the standard's "replicate the last row/column" padding.
        const int ix = std::min((px + x) * sx, width - 1);
        const int iy = std::min((py + y) * sy, height - 1);
        blk[y * 8 + x] = src[static_cast<size_t>(iy) * width + ix] - 128.0;
      }
    double coef[64];
    fdct8x8(blk, coef);
    int q[64];
    for (int k = 0; k < 64; ++k) {
      const int zz = kZigZag[k];
      q[k] = static_cast<int>(std::lround(coef[zz] / qt[k]));
    }
    const int diff = q[0] - predictor;
    predictor = q[0];
    const int s = magnitude(diff);
    e.put_bits(dct.code[s], dct.len[s]);
    if (s) e.put_bits(diff > 0 ? diff : diff + (1 << s) - 1, s);
    int run = 0;
    for (int k = 1; k < 64; ++k) {
      if (q[k] == 0) { ++run; continue; }
      while (run > 15) { e.put_bits(act.code[0xF0], act.len[0xF0]); run -= 16; }
      const int sz = magnitude(q[k]);
      const int sym = (run << 4) | sz;
      e.put_bits(act.code[sym], act.len[sym]);
      e.put_bits(q[k] > 0 ? q[k] : q[k] + (1 << sz) - 1, sz);
      run = 0;
    }
    if (run) e.put_bits(act.code[0x00], act.len[0x00]);
  };

  for (int my = 0; my < mcus_y; ++my)
    for (int mx = 0; mx < mcus_x; ++mx) {
      for (int by = 0; by < vs; ++by)
        for (int bx = 0; bx < hs; ++bx)
          encode_block(Y, mx * mcu_w / 1 + bx * 8, my * mcu_h / 1 + by * 8, 1, 1, qlum, dc_lum,
                       ac_lum, pred[0]);
      if (!gray) {
        encode_block(Cb, mx * 8, my * 8, hs, vs, qchr, dc_chr, ac_chr, pred[1]);
        encode_block(Cr, mx * 8, my * 8, hs, vs, qchr, dc_chr, ac_chr, pred[2]);
      }
    }
  e.flush_bits();
  e.marker(0xD9);
  state.bitstream = std::move(e.out);
  return NVJPEG_STATUS_SUCCESS;
}

VGPU_EXPORT nvjpegStatus_t nvjpegEncodeRetrieveBitstream(nvjpegHandle_t h,
                                                         nvjpegEncoderState_t st,
                                                         unsigned char* data, size_t* length,
                                                         cudaStream_t) {
  if (!known(h) || !known(st) || !length) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& state = *reinterpret_cast<EncoderState*>(st);
  // The documented two-call protocol: a null buffer asks for the size.
  if (!data) { *length = state.bitstream.size(); return NVJPEG_STATUS_SUCCESS; }
  if (*length < state.bitstream.size()) return NVJPEG_STATUS_INVALID_PARAMETER;
  std::memcpy(data, state.bitstream.data(), state.bitstream.size());
  *length = state.bitstream.size();
  return NVJPEG_STATUS_SUCCESS;
}
