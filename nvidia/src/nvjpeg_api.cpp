// libvgpunvjpeg -- VirtualGPU's nvJPEG, presented as libnvjpeg.so.12/13.
//
// A JPEG codec written here rather than delegated: there is no dependency in
// this repository to delegate to, and the format is frozen and fully
// specified (ITU-T T.81). Decoding is what image pipelines actually do on a
// GPU box -- DALI and torchvision's nvjpeg backend both go through this API --
// so it is the half that matters most.
//
// Decoding: 8-bit DCT JPEG, baseline, extended sequential and progressive,
// Huffman coded; one, three or four components (CMYK and YCCK from Adobe's
// marker); every chroma subsampling; interleaved and single-component scans,
// restart markers, and every way the API reaches a decode: nvjpegDecode, the
// batched API (nvjpegDecodeBatchedInitialize / nvjpegDecodeBatched, which
// torchvision.io.decode_jpeg uses on CUDA), and the decoupled, three-phase one
// (nvjpegDecodeJpegHost / TransferToDevice / Device, and nvjpegDecodeJpeg) with
// its streams, decoder states, buffers and decode parameters (output format,
// region of interest, CMYK). 12-bit samples, arithmetic coding, hierarchical
// and lossless JPEG are refused by name (NVJPEG_STATUS_JPEG_NOT_SUPPORTED).
//
// What a decode produces was checked against NVIDIA's nvJPEG 13.0 on an RTX
// 3060, sample for sample: the inverse DCT is a single-precision one rounded
// as NVIDIA's rounds, chroma is upsampled by replication as NVIDIA's is, and
// the colour conversion is NVIDIA's single-precision one, so the decoded
// images are identical but for a handful of samples (see idct8x8). The behaviours around
// the edges -- truncated files, the formats each image can be decoded to, the
// order of the decoupled phases, the batched API's argument checks -- are the
// card's, each measured, as the comments on them say.
//
// Encoding: baseline and progressive, quality 1-100, every chroma subsampling
// NVIDIA's encoder takes (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, 4:1:0, grey),
// standard or optimised Huffman tables, from RGB/BGR planar or interleaved
// (nvjpegEncodeImage), YCbCr planes of any subsampling (nvjpegEncodeYUV), or
// either through nvjpegEncode, and the quantisation tables of a parsed image
// (nvjpegEncoderParamsCopyQuantizationTables). A bitstream is a correct JPEG
// of the image, not NVIDIA's bytes: two encoders make different, equally
// legal choices.
#include <nvjpeg.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

// Output formats newer than CUDA 12.0's header, by value.
constexpr int kOutUnchangedU16 = 7, kOutNV12 = 8, kOutYUY2 = 9;

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

// Reads entropy-coded data a bit at a time, unstuffing the 0xFF 0x00 pairs the
// standard requires. Past the end of the data -- a scan cut short -- it reads
// zeros, as a decoder must: NVIDIA's decodes a file truncated inside its
// entropy-coded data and reports success.
struct BitReader {
  const uint8_t* p;
  const uint8_t* end;
  uint32_t buf = 0;
  int count = 0;

  int bit() {
    if (count == 0) {
      uint8_t b = 0;
      if (p < end) {
        b = *p;
        if (b == 0xFF) {
          if (p + 1 < end && p[1] == 0x00) {
            p += 2;
          } else {
            b = 0;   // a marker: the scan's data has ended
          }
        } else {
          ++p;
        }
      }
      buf = b;
      count = 8;
    }
    --count;
    return (buf >> count) & 1;
  }
  int bits(int n) {
    if (n <= 0 || n > 16) return 0;
    int v = 0;
    for (int i = 0; i < n; ++i) v = (v << 1) | bit();
    return v;
  }
  // At a restart interval: drop the remaining bits and step over the RSTn.
  void restart() {
    count = 0;
    while (p + 1 < end && !(p[0] == 0xFF && p[1] >= 0xD0 && p[1] <= 0xD7)) ++p;
    if (p + 1 < end) p += 2;
  }
};

int huff_decode(BitReader& br, const HuffTable& t) {
  int code = br.bit();
  int l = 1;
  while (l <= 16 && (t.maxcode[l] < 0 || code > t.maxcode[l])) {
    code = (code << 1) | br.bit();
    ++l;
  }
  if (l > 16) return 0;
  const int idx = t.valptr[l] + code - t.mincode[l];
  return idx >= 0 && idx < 256 ? t.values[idx] : 0;
}

// The sign convention of JPEG's variable-length integers: values whose top bit
// is clear are negative.
int extend(int v, int n) {
  if (n <= 0 || n > 16) return 0;
  return v < (1 << (n - 1)) ? v - (1 << n) + 1 : v;
}

// A coefficient fits comfortably in 16 bits in any real image; clamping keeps
// a corrupt file from overflowing the dequantisation multiply.
int16_t clamp16(int v) { return static_cast<int16_t>(v < -32768 ? -32768 : (v > 32767 ? 32767 : v)); }

const double (*dct_basis())[8] {
  static double c[8][8];
  static std::once_flag once;
  std::call_once(once, [] {
    for (int u = 0; u < 8; ++u)
      for (int x = 0; x < 8; ++x)
        c[u][x] = (u == 0 ? std::sqrt(0.125) : 0.5) * std::cos((2 * x + 1) * u * M_PI / 16.0);
  });
  return c;
}

// Separable inverse DCT in single precision with fused multiply-adds, level
// shifted and rounded half up, as a GPU computes one. Against NVIDIA's decoder
// on an RTX 3060 this was identical on all but 10 of 5.5 million decoded
// samples across twenty test images, those within two counts: NVIDIA's
// transform rounds differently inside, where the standard leaves it free to.
void idct8x8(const int* in, uint8_t* out, int stride) {
  const double(*cd)[8] = dct_basis();
  float c[8][8];
  for (int u = 0; u < 8; ++u) for (int x = 0; x < 8; ++x) c[u][x] = static_cast<float>(cd[u][x]);
  float tmp[64];
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      float s = 0;
      for (int u = 0; u < 8; ++u) s = std::fmaf(c[u][x], static_cast<float>(in[y * 8 + u]), s);
      tmp[y * 8 + x] = s;
    }
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y) {
      float s = 0;
      for (int v = 0; v < 8; ++v) s = std::fmaf(c[v][y], tmp[v * 8 + x], s);
      const float r = std::floor(s + 128.5f);
      out[y * stride + x] = static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
}

void fdct8x8(const double* in, double* out) {
  const double(*c)[8] = dct_basis();
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

/* ======================================================================== */
/* The decoder                                                              */
/* ======================================================================== */

struct Component {
  int id = 0, h = 1, v = 1, tq = 0;
  int td = 0, ta = 0;
  int dc_pred = 0;
  int bw = 0, bh = 0;              // blocks per row and column, padded to the MCU grid
  std::vector<int16_t> coef;       // bw*bh blocks of 64, in zig-zag order, quantised
  std::vector<uint8_t> plane;      // bw*8 by bh*8 samples, after the inverse DCT
};

struct Image {
  int width = 0, height = 0, precision = 8;
  int hmax = 1, vmax = 1, mcus_x = 0, mcus_y = 0;
  int encoding = 0;                // the SOF marker: 0xC0, 0xC1, 0xC2
  int restart_interval = 0;
  int adobe = -1;                  // Adobe APP14 colour transform, -1 without the marker
  bool jfif = false;
  int orientation = 0;             // EXIF orientation, 0 without one
  std::vector<Component> comps;
  uint16_t quant[4][64] = {{0}};
  bool quant_set[4] = {false, false, false, false};
  HuffTable dc[4], ac[4];
  int scans = 0;
  bool frame() const { return !comps.empty(); }
};

// One scan's entropy-coded data: every MCU of it, by the scan's kind.
struct ScanDecoder {
  Image& im;
  std::vector<Component*> comps;
  int ss = 0, se = 63, ah = 0, al = 0;
  int eobrun = 0;

  int16_t* block(Component& c, int bx, int by) {
    return c.coef.data() + (static_cast<size_t>(by) * c.bw + bx) * 64;
  }

  void decode_block(BitReader& br, Component& c, int16_t* b) {
    const bool progressive = im.encoding == 0xC2;
    if (!progressive) {
      const int t = huff_decode(br, im.dc[c.td]);
      c.dc_pred += t ? extend(br.bits(t), t) : 0;
      c.dc_pred = std::clamp(c.dc_pred, -32768, 32767);
      b[0] = static_cast<int16_t>(c.dc_pred);
      for (int k = 1; k < 64;) {
        const int rs = huff_decode(br, im.ac[c.ta]);
        const int r = rs >> 4, s = rs & 0xF;
        if (s == 0) {
          if (r != 15) break;   // EOB
          k += 16;
          continue;
        }
        k += r;
        if (k > 63) break;
        b[k] = clamp16(extend(br.bits(s), s));
        ++k;
      }
      return;
    }
    if (ss == 0) {   // DC
      if (ah == 0) {
        const int t = huff_decode(br, im.dc[c.td]);
        c.dc_pred += t ? extend(br.bits(t), t) : 0;
        c.dc_pred = std::clamp(c.dc_pred, -32768, 32767);
        b[0] = clamp16(c.dc_pred * (1 << al));
      } else if (br.bit()) {
        b[0] = static_cast<int16_t>(b[0] | (1 << al));
      }
      return;
    }
    if (ah == 0) {   // AC, first pass (G.1.2.2)
      if (eobrun > 0) {
        --eobrun;
        return;
      }
      for (int k = ss; k <= se; ++k) {
        const int rs = huff_decode(br, im.ac[c.ta]);
        const int r = rs >> 4, s = rs & 0xF;
        if (s) {
          k += r;
          if (k > 63) break;
          b[k] = clamp16(extend(br.bits(s), s) * (1 << al));
        } else if (r < 15) {
          eobrun = (1 << r) - 1;
          if (r) eobrun += br.bits(r);
          break;
        } else {
          k += 15;   // sixteen zeros
        }
      }
      return;
    }
    // AC, refinement (G.1.2.3): a bit for each coefficient already nonzero,
    // new ones of magnitude 1 << al placed among the zeros.
    const int p1 = 1 << al, m1 = -(1 << al);
    int k = ss;
    if (eobrun == 0) {
      for (; k <= se; ++k) {
        const int rs = huff_decode(br, im.ac[c.ta]);
        int r = rs >> 4;
        int s = rs & 0xF;
        if (s) {
          s = br.bit() ? p1 : m1;
        } else if (r != 15) {
          eobrun = 1 << r;
          if (r) eobrun += br.bits(r);
          break;
        }
        do {
          int16_t& co = b[k];
          if (co != 0) {
            if (br.bit() && (co & p1) == 0) co = clamp16(co >= 0 ? co + p1 : co + m1);
          } else if (--r < 0) {
            break;
          }
          ++k;
        } while (k <= se);
        if (s && k <= 63) b[k] = static_cast<int16_t>(s);
      }
    }
    if (eobrun > 0) {
      for (; k <= se; ++k) {
        int16_t& co = b[k];
        if (co != 0 && br.bit() && (co & p1) == 0) co = clamp16(co >= 0 ? co + p1 : co + m1);
      }
      --eobrun;
    }
  }

  void run(BitReader& br) {
    for (Component* c : comps) c->dc_pred = 0;
    eobrun = 0;
    int mcu = 0;
    const auto restart_due = [&] {
      if (im.restart_interval && mcu && mcu % im.restart_interval == 0) {
        br.restart();
        for (Component* c : comps) c->dc_pred = 0;
        eobrun = 0;
      }
      ++mcu;
    };
    if (comps.size() == 1) {
      // A single-component scan covers that component's own blocks, in
      // raster order, not the MCU grid.
      Component& c = *comps[0];
      const int cw = (im.width * c.h + im.hmax - 1) / im.hmax, ch = (im.height * c.v + im.vmax - 1) / im.vmax;
      const int nbx = (cw + 7) / 8, nby = (ch + 7) / 8;
      for (int by = 0; by < nby; ++by)
        for (int bx = 0; bx < nbx; ++bx) {
          restart_due();
          decode_block(br, c, block(c, bx, by));
        }
      return;
    }
    for (int my = 0; my < im.mcus_y; ++my)
      for (int mx = 0; mx < im.mcus_x; ++mx) {
        restart_due();
        for (Component* c : comps)
          for (int by = 0; by < c->v; ++by)
            for (int bx = 0; bx < c->h; ++bx) decode_block(br, *c, block(*c, mx * c->h + bx, my * c->v + by));
      }
  }
};

void finish_image(Image& im) {
  int deq[64];
  for (Component& c : im.comps) {
    c.plane.assign(static_cast<size_t>(c.bw) * 8 * c.bh * 8, 0);
    const uint16_t* q = im.quant[c.tq];
    for (int by = 0; by < c.bh; ++by)
      for (int bx = 0; bx < c.bw; ++bx) {
        const int16_t* b = c.coef.data() + (static_cast<size_t>(by) * c.bw + bx) * 64;
        for (int k = 0; k < 64; ++k) deq[kZigZag[k]] = b[k] * q[k];
        idct8x8(deq, c.plane.data() + static_cast<size_t>(by) * 8 * c.bw * 8 + bx * 8, c.bw * 8);
      }
    std::vector<int16_t>().swap(c.coef);
  }
}

// The EXIF orientation tag (0x0112) of an APP1 "Exif" segment.
int exif_orientation(const uint8_t* s, int n) {
  if (n < 14 || std::memcmp(s, "Exif\0\0", 6) != 0) return 0;
  const uint8_t* t = s + 6;
  const int tn = n - 6;
  const bool le = t[0] == 'I';
  const auto u16 = [&](int o) { return o + 2 <= tn ? (le ? t[o] | t[o + 1] << 8 : t[o] << 8 | t[o + 1]) : 0; };
  const auto u32 = [&](int o) -> uint32_t {
    if (o + 4 > tn) return 0;
    return le ? (t[o] | t[o + 1] << 8 | t[o + 2] << 16 | static_cast<uint32_t>(t[o + 3]) << 24)
              : (static_cast<uint32_t>(t[o]) << 24 | t[o + 1] << 16 | t[o + 2] << 8 | t[o + 3]);
  };
  const uint32_t ifd = u32(4);
  if (ifd + 2 > static_cast<uint32_t>(tn)) return 0;
  const int entries = u16(static_cast<int>(ifd));
  for (int e = 0; e < entries; ++e) {
    const int at = static_cast<int>(ifd) + 2 + 12 * e;
    if (at + 12 > tn) break;
    if (u16(at) == 0x0112) return u16(at + 8);
  }
  return 0;
}

// Parses a JPEG and, when `decode` is set, decodes every scan. Results follow
// NVIDIA's 13.0 on an RTX 3060: a file cut short inside a marker segment, or
// before its first scan, is NVJPEG_STATUS_INCOMPLETE_BITSTREAM; one cut short
// inside entropy-coded data decodes what is there and succeeds; bytes where a
// marker must be are NVJPEG_STATUS_JPEG_NOT_SUPPORTED.
nvjpegStatus_t parse_jpeg(const uint8_t* data, size_t len, Image* im, bool decode) {
  if (!data) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (len < 2) return NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
  if (data[0] != 0xFF || data[1] != 0xD8) return NVJPEG_STATUS_BAD_JPEG;
  size_t i = 2;
  const auto u16 = [&](size_t at) { return static_cast<int>(data[at]) << 8 | data[at + 1]; };
  bool sof_supported = true;
  for (;;) {
    if (i >= len) break;
    if (data[i] != 0xFF) return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
    while (i < len && data[i] == 0xFF) ++i;   // fill bytes
    if (i >= len) break;
    const uint8_t marker = data[i++];
    if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;
    if (marker == 0xD9) break;
    if (i + 2 > len) return NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
    const int seglen = u16(i);
    if (seglen < 2) return NVJPEG_STATUS_BAD_JPEG;
    if (i + seglen > len) return NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
    const uint8_t* seg = data + i + 2;
    const int segn = seglen - 2;
    switch (marker) {
      case 0xC0: case 0xC1: case 0xC2: {
        if (im->frame()) return NVJPEG_STATUS_BAD_JPEG;   // a second frame
        if (segn < 6) return NVJPEG_STATUS_BAD_JPEG;
        im->encoding = marker;
        im->precision = seg[0];
        im->height = seg[1] << 8 | seg[2];
        im->width = seg[3] << 8 | seg[4];
        const int nc = seg[5];
        if (nc < 1 || nc > 4 || segn < 6 + 3 * nc) return NVJPEG_STATUS_BAD_JPEG;
        im->comps.resize(nc);
        for (int c = 0; c < nc; ++c) {
          Component& co = im->comps[c];
          co.id = seg[6 + c * 3];
          co.h = seg[7 + c * 3] >> 4;
          co.v = seg[7 + c * 3] & 0xF;
          co.tq = seg[8 + c * 3];
          if (co.h < 1 || co.h > 4 || co.v < 1 || co.v > 4 || co.tq > 3) return NVJPEG_STATUS_BAD_JPEG;
          im->hmax = std::max(im->hmax, co.h);
          im->vmax = std::max(im->vmax, co.v);
        }
        // A header can declare 65535x65535 with four components, which is
        // 17 GB of planes. Refuse it here rather than in the allocator.
        if (im->width <= 0 || im->height <= 0 || static_cast<int64_t>(im->width) * im->height > 268435456LL)
          return NVJPEG_STATUS_BAD_JPEG;
        if (im->precision != 8) sof_supported = false;   // 12-bit samples
        im->mcus_x = (im->width + im->hmax * 8 - 1) / (im->hmax * 8);
        im->mcus_y = (im->height + im->vmax * 8 - 1) / (im->vmax * 8);
        for (Component& co : im->comps) {
          co.bw = im->mcus_x * co.h;
          co.bh = im->mcus_y * co.v;
        }
        break;
      }
      case 0xC3: case 0xC5: case 0xC6: case 0xC7:
      case 0xC9: case 0xCA: case 0xCB: case 0xCD: case 0xCE: case 0xCF:
        return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;   // lossless, arithmetic, hierarchical
      case 0xC4: {   // DHT
        int p = 0;
        while (p < segn) {
          if (p + 17 > segn) return NVJPEG_STATUS_BAD_JPEG;
          const int tc = seg[p] >> 4, th = seg[p] & 0xF;
          if (th > 3 || tc > 1) return NVJPEG_STATUS_BAD_JPEG;
          HuffTable& t = tc == 0 ? im->dc[th] : im->ac[th];
          t = HuffTable{};
          int total = 0;
          for (int l = 1; l <= 16; ++l) {
            t.bits[l] = seg[p + l];
            total += t.bits[l];
          }
          if (p + 17 + total > segn || total > 256) return NVJPEG_STATUS_BAD_JPEG;
          std::memcpy(t.values, seg + p + 17, total);
          t.build();
          p += 17 + total;
        }
        break;
      }
      case 0xDB: {   // DQT
        int p = 0;
        while (p < segn) {
          const int pq = seg[p] >> 4, tq = seg[p] & 0xF;
          if (tq > 3 || pq > 1) return NVJPEG_STATUS_BAD_JPEG;
          ++p;
          if (p + 64 * (pq + 1) > segn) return NVJPEG_STATUS_BAD_JPEG;
          for (int k = 0; k < 64; ++k) {
            im->quant[tq][k] = pq ? static_cast<uint16_t>(seg[p] << 8 | seg[p + 1]) : seg[p];
            p += pq + 1;
          }
          im->quant_set[tq] = true;
        }
        break;
      }
      case 0xDD:
        if (segn < 2) return NVJPEG_STATUS_BAD_JPEG;
        im->restart_interval = seg[0] << 8 | seg[1];
        break;
      case 0xE0:
        if (segn >= 5 && std::memcmp(seg, "JFIF", 5) == 0) im->jfif = true;
        break;
      case 0xE1:
        if (!im->orientation) im->orientation = exif_orientation(seg, segn);
        break;
      case 0xEE:
        if (segn >= 12 && std::memcmp(seg, "Adobe", 5) == 0) im->adobe = seg[11];
        break;
      case 0xDA: {   // SOS: the entropy-coded data follows the segment
        if (!im->frame()) return NVJPEG_STATUS_BAD_JPEG;
        if (!sof_supported) return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
        if (!decode) return NVJPEG_STATUS_SUCCESS;
        if (segn < 1) return NVJPEG_STATUS_BAD_JPEG;
        const int ns = seg[0];
        if (ns < 1 || ns > 4 || segn < 4 + 2 * ns) return NVJPEG_STATUS_BAD_JPEG;
        if (im->comps[0].coef.empty())
          for (Component& co : im->comps) co.coef.assign(static_cast<size_t>(co.bw) * co.bh * 64, 0);
        ScanDecoder sd{*im, {}};
        for (int s = 0; s < ns; ++s) {
          Component* found = nullptr;
          for (Component& co : im->comps)
            if (co.id == seg[1 + s * 2]) found = &co;
          if (!found) return NVJPEG_STATUS_BAD_JPEG;
          found->td = seg[2 + s * 2] >> 4;
          found->ta = seg[2 + s * 2] & 0xF;
          if (found->td > 3 || found->ta > 3) return NVJPEG_STATUS_BAD_JPEG;
          sd.comps.push_back(found);
        }
        sd.ss = seg[1 + 2 * ns];
        sd.se = seg[2 + 2 * ns];
        sd.ah = seg[3 + 2 * ns] >> 4;
        sd.al = seg[3 + 2 * ns] & 0xF;
        if (im->encoding == 0xC2) {
          if (sd.se > 63 || sd.ss > sd.se || (sd.ss == 0) != (sd.se == 0) || (sd.ss > 0 && ns != 1) || sd.al > 13)
            return NVJPEG_STATUS_BAD_JPEG;
        } else {
          sd.ss = 0;
          sd.se = 63;
        }
        for (Component* c : sd.comps) {
          const bool needs_dc = im->encoding != 0xC2 || (sd.ss == 0 && sd.ah == 0);
          const bool needs_ac = im->encoding != 0xC2 || sd.ss > 0;
          if ((needs_dc && !im->dc[c->td].present) || (needs_ac && !im->ac[c->ta].present))
            return NVJPEG_STATUS_BAD_JPEG;
        }
        // Where the scan's data ends: the next marker that is not a restart.
        size_t end = i + seglen;
        while (end + 1 < len && !(data[end] == 0xFF && data[end + 1] != 0x00 && !(data[end + 1] >= 0xD0 && data[end + 1] <= 0xD7)))
          ++end;
        if (end + 1 >= len) end = len;
        BitReader br{data + i + seglen, data + end};
        sd.run(br);
        ++im->scans;
        i = end;
        continue;
      }
      default:
        break;   // other APPn, COM and the like carry nothing a decoder needs
    }
    i += seglen;
  }
  if (!im->frame()) return NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
  if (!sof_supported) return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
  if (decode) {
    if (im->scans == 0) return NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
    finish_image(*im);
  }
  return NVJPEG_STATUS_SUCCESS;
}

nvjpegChromaSubsampling_t subsampling_of(const Image& im) {
  if (im.comps.size() == 1) return NVJPEG_CSS_GRAY;
  if (im.comps.size() < 3) return NVJPEG_CSS_UNKNOWN;
  // Every chroma component the same, as in each subsampling nvJPEG names.
  for (size_t c = 2; c < im.comps.size(); ++c)
    if (im.comps[c].h != im.comps[1].h || im.comps[c].v != im.comps[1].v) return NVJPEG_CSS_UNKNOWN;
  if (im.comps[0].h % im.comps[1].h || im.comps[0].v % im.comps[1].v) return NVJPEG_CSS_UNKNOWN;
  const int h = im.comps[0].h / im.comps[1].h, v = im.comps[0].v / im.comps[1].v;
  if (h == 1 && v == 1) return NVJPEG_CSS_444;
  if (h == 2 && v == 1) return NVJPEG_CSS_422;
  if (h == 2 && v == 2) return NVJPEG_CSS_420;
  if (h == 1 && v == 2) return NVJPEG_CSS_440;
  if (h == 4 && v == 1) return NVJPEG_CSS_411;
  if (h == 4 && v == 2) return NVJPEG_CSS_410;
  return NVJPEG_CSS_UNKNOWN;
}

int comp_width(const Image& im, int c) { return (im.width * im.comps[c].h + im.hmax - 1) / im.hmax; }
int comp_height(const Image& im, int c) { return (im.height * im.comps[c].v + im.vmax - 1) / im.vmax; }

// Chroma upsampling by replication, as the standard's simplest filter and as
// NVIDIA's decoder does (identical output on the card).
uint8_t sample(const Image& im, int c, int x, int y) {
  const Component& co = im.comps[c];
  const int sx = std::min(x * co.h / im.hmax, co.bw * 8 - 1);
  const int sy = std::min(y * co.v / im.vmax, co.bh * 8 - 1);
  return co.plane[static_cast<size_t>(sy) * co.bw * 8 + sx];
}

// Rounded to nearest, ties to even, as a GPU's float-to-integer conversion
// rounds: Y 31 and Cb 253 make B 252.5, which the card wrote as 252.
uint8_t to_byte(float v) {
  const float r = std::nearbyint(v);
  return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}
// The conversion in single precision, as NVIDIA's computes it.
void ycbcr_to_rgb(float y, float cb, float cr, uint8_t* r, uint8_t* g, uint8_t* b) {
  *r = to_byte(y + 1.402f * (cr - 128.0f));
  *g = to_byte(y - 0.344136f * (cb - 128.0f) - 0.714136f * (cr - 128.0f));
  *b = to_byte(y + 1.772f * (cb - 128.0f));
}

// What a decode writes, and where.
struct OutputSpec {
  int format = NVJPEG_OUTPUT_UNCHANGED;
  bool roi = false;
  int rx = 0, ry = 0, rw = 0, rh = 0;
  bool allow_cmyk = false;
};

nvjpegStatus_t put_plane(unsigned char* dst, size_t pitch, const std::vector<uint8_t>& src, int w, int h) {
  if (!dst) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (!w || !h) return NVJPEG_STATUS_SUCCESS;
  if (pitch < static_cast<size_t>(w)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (cudaMemcpy2D(dst, pitch, src.data(), w, w, h, cudaMemcpyHostToDevice) != cudaSuccess)
    return NVJPEG_STATUS_EXECUTION_FAILED;
  return NVJPEG_STATUS_SUCCESS;
}

// Whether `format` can be produced from this image, as NVIDIA's 13.0 decides
// it (each refusal NVJPEG_STATUS_INVALID_PARAMETER on the card): NV12 only from
// 4:2:0, YUY2 only from 4:2:2, the 16-bit format never from 8-bit samples, and
// from CMYK only the components unchanged -- or, with CMYK allowed (the
// decoupled API's nvjpegDecodeParamsSetAllowCMYK), RGB, BGR and grey, but
// still not YUV.
nvjpegStatus_t check_format(const Image& im, const OutputSpec& o) {
  const nvjpegChromaSubsampling_t css = subsampling_of(im);
  const bool four = im.comps.size() == 4;
  switch (o.format) {
    case NVJPEG_OUTPUT_UNCHANGED:
      return NVJPEG_STATUS_SUCCESS;
    case NVJPEG_OUTPUT_YUV:
      return four ? NVJPEG_STATUS_INVALID_PARAMETER : NVJPEG_STATUS_SUCCESS;
    case NVJPEG_OUTPUT_Y:
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_BGR:
    case NVJPEG_OUTPUT_RGBI:
    case NVJPEG_OUTPUT_BGRI:
      if (four && !o.allow_cmyk) return NVJPEG_STATUS_INVALID_PARAMETER;
      if (im.comps.size() == 2) return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
      return NVJPEG_STATUS_SUCCESS;
    case kOutNV12:
      return css == NVJPEG_CSS_420 ? NVJPEG_STATUS_SUCCESS : NVJPEG_STATUS_INVALID_PARAMETER;
    case kOutYUY2:
      return css == NVJPEG_CSS_422 ? NVJPEG_STATUS_SUCCESS : NVJPEG_STATUS_INVALID_PARAMETER;
    default:
      return NVJPEG_STATUS_INVALID_PARAMETER;   // kOutUnchangedU16 and anything unknown
  }
}

// Writes a decoded image in `o.format`. A region of interest, which NVIDIA's
// takes for the RGB family and grey only, crops the full decode exactly (the
// card's region matched its full decode pixel for pixel).
nvjpegStatus_t write_image(const Image& im, const OutputSpec& o, nvjpegImage_t* dst) {
  if (!dst) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (const nvjpegStatus_t st = check_format(im, o); st != NVJPEG_STATUS_SUCCESS) return st;
  int x0 = 0, y0 = 0, w = im.width, h = im.height;
  if (o.roi) {
    x0 = o.rx;
    y0 = o.ry;
    w = o.rw;
    h = o.rh;
  }
  const size_t n = static_cast<size_t>(w) * h;
  const size_t nc = im.comps.size();
  // RGB at (x, y) of the image.
  const bool rgb_components = nc == 3 && (im.adobe == 0 || (!im.jfif && im.adobe < 0 && im.comps[0].id == 'R' &&
                                                            im.comps[1].id == 'G' && im.comps[2].id == 'B'));
  const auto rgb_at = [&](int x, int y, uint8_t* r, uint8_t* g, uint8_t* b) {
    if (nc == 1) {
      *r = *g = *b = sample(im, 0, x, y);
    } else if (nc == 3) {
      if (rgb_components) {
        *r = sample(im, 0, x, y);
        *g = sample(im, 1, x, y);
        *b = sample(im, 2, x, y);
      } else {
        ycbcr_to_rgb(sample(im, 0, x, y), sample(im, 1, x, y), sample(im, 2, x, y), r, g, b);
      }
    } else {
      // CMYK as Adobe stores it (inverted): R = C * K / 255, rounding a
      // remainder of exactly half down, as NVIDIA's (measured on every
      // pixel of a gradient). YCCK is converted to CMY first.
      int c = sample(im, 0, x, y), m = sample(im, 1, x, y), ye = sample(im, 2, x, y);
      const int k = sample(im, 3, x, y);
      if (im.adobe == 2) {
        uint8_t rr, gg, bb;
        ycbcr_to_rgb(static_cast<float>(c), static_cast<float>(m), static_cast<float>(ye), &rr, &gg, &bb);
        c = rr;
        m = gg;
        ye = bb;
      }
      *r = static_cast<uint8_t>((c * k + 126) / 255);
      *g = static_cast<uint8_t>((m * k + 126) / 255);
      *b = static_cast<uint8_t>((ye * k + 126) / 255);
    }
  };
  switch (o.format) {
    case NVJPEG_OUTPUT_UNCHANGED:
    case NVJPEG_OUTPUT_YUV: {
      for (size_t c = 0; c < nc; ++c) {
        const Component& co = im.comps[c];
        const int cw = comp_width(im, static_cast<int>(c)), ch = comp_height(im, static_cast<int>(c));
        std::vector<uint8_t> plane(static_cast<size_t>(cw) * ch);
        for (int y = 0; y < ch; ++y)
          std::memcpy(plane.data() + static_cast<size_t>(y) * cw, co.plane.data() + static_cast<size_t>(y) * co.bw * 8, cw);
        if (const nvjpegStatus_t st = put_plane(dst->channel[c], dst->pitch[c], plane, cw, ch); st) return st;
      }
      return NVJPEG_STATUS_SUCCESS;
    }
    case NVJPEG_OUTPUT_Y: {
      std::vector<uint8_t> plane(n);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          uint8_t& out = plane[static_cast<size_t>(y) * w + x];
          if (nc == 4 || rgb_components) {
            uint8_t r, g, b;
            rgb_at(x0 + x, y0 + y, &r, &g, &b);
            out = to_byte(0.299f * r + 0.587f * g + 0.114f * b);
          } else {
            out = sample(im, 0, x0 + x, y0 + y);
          }
        }
      return put_plane(dst->channel[0], dst->pitch[0], plane, w, h);
    }
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_BGR:
    case NVJPEG_OUTPUT_RGBI:
    case NVJPEG_OUTPUT_BGRI: {
      const bool interleaved = o.format == NVJPEG_OUTPUT_RGBI || o.format == NVJPEG_OUTPUT_BGRI;
      const bool bgr = o.format == NVJPEG_OUTPUT_BGR || o.format == NVJPEG_OUTPUT_BGRI;
      std::vector<uint8_t> r(n), g(n), b(n);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t at = static_cast<size_t>(y) * w + x;
          rgb_at(x0 + x, y0 + y, &r[at], &g[at], &b[at]);
        }
      if (interleaved) {
        std::vector<uint8_t> il(n * 3);
        for (size_t at = 0; at < n; ++at) {
          il[at * 3 + 0] = bgr ? b[at] : r[at];
          il[at * 3 + 1] = g[at];
          il[at * 3 + 2] = bgr ? r[at] : b[at];
        }
        return put_plane(dst->channel[0], dst->pitch[0], il, w * 3, h);
      }
      if (const nvjpegStatus_t st = put_plane(dst->channel[0], dst->pitch[0], bgr ? b : r, w, h); st) return st;
      if (const nvjpegStatus_t st = put_plane(dst->channel[1], dst->pitch[1], g, w, h); st) return st;
      return put_plane(dst->channel[2], dst->pitch[2], bgr ? r : b, w, h);
    }
    case kOutNV12: {   // Y, then Cb and Cr interleaved at half resolution
      const int cw = comp_width(im, 1), ch = comp_height(im, 1);
      std::vector<uint8_t> y(n), uv(static_cast<size_t>(cw) * 2 * ch);
      for (int yy = 0; yy < h; ++yy)
        std::memcpy(y.data() + static_cast<size_t>(yy) * w, im.comps[0].plane.data() + static_cast<size_t>(yy) * im.comps[0].bw * 8, w);
      for (int yy = 0; yy < ch; ++yy)
        for (int x = 0; x < cw; ++x) {
          uv[(static_cast<size_t>(yy) * cw + x) * 2] = im.comps[1].plane[static_cast<size_t>(yy) * im.comps[1].bw * 8 + x];
          uv[(static_cast<size_t>(yy) * cw + x) * 2 + 1] = im.comps[2].plane[static_cast<size_t>(yy) * im.comps[2].bw * 8 + x];
        }
      if (const nvjpegStatus_t st = put_plane(dst->channel[0], dst->pitch[0], y, w, h); st) return st;
      return put_plane(dst->channel[1], dst->pitch[1], uv, cw * 2, ch);
    }
    case kOutYUY2: {   // Y0 Cb Y1 Cr
      const int pairs = (w + 1) / 2;
      std::vector<uint8_t> out(static_cast<size_t>(pairs) * 4 * h);
      for (int y = 0; y < h; ++y)
        for (int p = 0; p < pairs; ++p) {
          uint8_t* q = out.data() + (static_cast<size_t>(y) * pairs + p) * 4;
          q[0] = sample(im, 0, std::min(2 * p, w - 1), y);
          q[1] = sample(im, 1, 2 * p, y);
          q[2] = sample(im, 0, std::min(2 * p + 1, w - 1), y);
          q[3] = sample(im, 2, 2 * p, y);
        }
      return put_plane(dst->channel[0], dst->pitch[0], out, pairs * 4, h);
    }
    default:
      return NVJPEG_STATUS_INVALID_PARAMETER;
  }
}

/* ======================================================================== */
/* Handles                                                                  */
/* ======================================================================== */

enum class Kind { Handle, State, Stream, DecodeParams, Decoder, BufferPinned, BufferDevice, EncoderState, EncoderParams };

std::mutex g_mu;
std::map<const void*, Kind> g_live;

template <class T>
T* track(T* p, Kind k) {
  std::lock_guard<std::mutex> l(g_mu);
  g_live[p] = k;
  return p;
}
bool known(const void* p, Kind k) {
  std::lock_guard<std::mutex> l(g_mu);
  const auto it = p ? g_live.find(p) : g_live.end();
  return it != g_live.end() && it->second == k;
}
template <class T>
nvjpegStatus_t destroy(void* p, Kind k) {
  if (!known(p, k)) return NVJPEG_STATUS_INVALID_PARAMETER;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.erase(p);
  }
  delete static_cast<T*>(p);
  return NVJPEG_STATUS_SUCCESS;
}

struct Handle {
  int backend = 0;
};

struct Buffer {
  size_t size = 0;
  void* ptr = nullptr;
  bool device = false;
  ~Buffer() {
    if (!ptr) return;
    if (device) cudaFree(ptr);
    else cudaFreeHost(ptr);
  }
  void resize(size_t want) {
    if (want <= size) return;
    if (ptr) {
      if (device) cudaFree(ptr);
      else cudaFreeHost(ptr);
    }
    ptr = nullptr;
    if ((device ? cudaMalloc(&ptr, want) : cudaMallocHost(&ptr, want)) == cudaSuccess) size = want;
    else size = 0;
  }
};

struct State {
  // The decoupled API's phases: what the host phase decoded, and how far a
  // decode has come (0 nothing, 1 host done, 2 transferred).
  std::unique_ptr<Image> image;
  OutputSpec out;
  int phase = 0;
  Buffer* pinned = nullptr;
  Buffer* device = nullptr;
  // nvjpegDecodeBatchedInitialize.
  int batch_size = 0;
  int batch_format = -1;
};

struct Stream {
  std::vector<uint8_t> data;
  Image header;
  bool parsed = false;
};

struct DecodeParams {
  OutputSpec out;
  int orientation = 1;   // NVJPEG_ORIENTATION_NORMAL
  int scale = 0;         // NVJPEG_SCALE_NONE
};

struct Decoder {
  int backend = 0;
};

/* ======================================================================== */
/* The encoder                                                              */
/* ======================================================================== */

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

// A Huffman table as an encoder needs it: the (bits, values) form a DHT
// segment carries, and code/length by symbol.
struct EncTable {
  uint8_t bits[17] = {0};
  std::vector<uint8_t> vals;
  int code[256] = {0};
  int len[256] = {0};

  void build() {
    std::fill(std::begin(code), std::end(code), 0);
    std::fill(std::begin(len), std::end(len), 0);
    int c = 0;
    size_t k = 0;
    for (int l = 1; l <= 16; ++l) {
      for (int j = 0; j < bits[l]; ++j, ++k) {
        if (k < vals.size()) {
          code[vals[k]] = c;
          len[vals[k]] = l;
        }
        ++c;
      }
      c <<= 1;
    }
  }
  static EncTable standard(const uint8_t* bits, const uint8_t* vals, int n) {
    EncTable t;
    std::memcpy(t.bits, bits, 17);
    t.vals.assign(vals, vals + n);
    t.build();
    return t;
  }
  // An optimal table for these symbol frequencies, by the procedure of Annex
  // K.2: code sizes from repeatedly merging the two least frequent, a
  // reserved all-ones code, lengths limited to 16 by K.3.
  static EncTable optimal(const long* freq_in) {
    long freq[257];
    std::copy(freq_in, freq_in + 256, freq);
    freq[256] = 1;   // the reserved code point, so no code is all ones
    int codesize[257] = {0}, others[257];
    std::fill(std::begin(others), std::end(others), -1);
    for (;;) {
      int c1 = -1, c2 = -1;
      long v1 = 1000000000L, v2 = 1000000000L;
      for (int i = 0; i <= 256; ++i)
        if (freq[i] && freq[i] <= v1) {
          v2 = v1;
          c2 = c1;
          v1 = freq[i];
          c1 = i;
        } else if (freq[i] && freq[i] <= v2) {
          v2 = freq[i];
          c2 = i;
        }
      if (c2 < 0) break;
      freq[c1] += freq[c2];
      freq[c2] = 0;
      ++codesize[c1];
      while (others[c1] >= 0) {
        c1 = others[c1];
        ++codesize[c1];
      }
      others[c1] = c2;
      ++codesize[c2];
      while (others[c2] >= 0) {
        c2 = others[c2];
        ++codesize[c2];
      }
    }
    int count[33] = {0};
    for (int i = 0; i <= 256; ++i)
      if (codesize[i]) ++count[std::min(codesize[i], 32)];
    for (int i = 32; i > 16; --i)
      while (count[i] > 0) {
        int j = i - 2;
        while (count[j] == 0) --j;
        count[i] -= 2;
        ++count[i - 1];
        count[j + 1] += 2;
        --count[j];
      }
    int i = 16;
    while (count[i] == 0) --i;
    --count[i];   // drop the reserved code point
    EncTable t;
    for (int l = 1; l <= 16; ++l) t.bits[l] = static_cast<uint8_t>(count[l]);
    for (int l = 1; l <= 32; ++l)
      for (int s = 0; s < 256; ++s)
        if (codesize[s] == l) t.vals.push_back(static_cast<uint8_t>(s));
    t.build();
    return t;
  }
};

struct BitWriter {
  std::string out;
  uint32_t buf = 0;
  int count = 0;
  bool counting = false;   // a first pass gathering symbol frequencies writes nothing

  void byte(uint8_t b) { out.push_back(static_cast<char>(b)); }
  void word(int v) {
    byte(static_cast<uint8_t>(v >> 8));
    byte(static_cast<uint8_t>(v & 0xFF));
  }
  void marker(uint8_t m) {
    byte(0xFF);
    byte(m);
  }
  void put(int code, int len) {
    if (counting) return;
    for (int i = len - 1; i >= 0; --i) {
      buf = (buf << 1) | ((code >> i) & 1);
      if (++count == 8) {
        const uint8_t b = static_cast<uint8_t>(buf & 0xFF);
        byte(b);
        if (b == 0xFF) byte(0x00);   // stuffing, so the byte is not read as a marker
        buf = 0;
        count = 0;
      }
    }
  }
  void flush() {
    if (counting) return;
    while (count) put(1, 1);   // pad with ones, as the standard says
  }
};

int magnitude(int v) {
  int n = 0;
  int a = v < 0 ? -v : v;
  while (a) {
    ++n;
    a >>= 1;
  }
  return n;
}

void scale_quant(const uint8_t* base, int quality, uint16_t* out) {
  const int q = std::clamp(quality, 1, 100);
  const int scale = q < 50 ? 5000 / q : 200 - q * 2;
  for (int i = 0; i < 64; ++i) {
    const int v = (base[i] * scale + 50) / 100;
    out[i] = static_cast<uint16_t>(std::clamp(v, 1, 255));
  }
}

struct EncoderParams {
  int quality = 70;
  nvjpegChromaSubsampling_t css = NVJPEG_CSS_444;
  int encoding = 0xC0;
  bool optimized = false;
  unsigned restart = 0;
  bool custom_quant = false;   // tables copied from a parsed image
  uint16_t quant[2][64] = {{0}};
};
struct EncoderState {
  std::string bitstream;
  // NVIDIA's encoder keeps the bitstream in device memory until a host
  // retrieval takes it: on the card, nvjpegEncodeRetrieveBitstreamDevice after
  // nvjpegEncodeRetrieveBitstream succeeded and wrote nothing.
  bool on_device = false;
};

// The sampling factors of luma for each subsampling the encoder takes.
void sampling_of(nvjpegChromaSubsampling_t css, int* hs, int* vs) {
  *hs = 1;
  *vs = 1;
  switch (css) {
    case NVJPEG_CSS_422: *hs = 2; break;
    case NVJPEG_CSS_420: *hs = 2; *vs = 2; break;
    case NVJPEG_CSS_440: *vs = 2; break;
    case NVJPEG_CSS_411: *hs = 4; break;
    case NVJPEG_CSS_410: *hs = 4; *vs = 2; break;
    default: break;
  }
}

// A plane to encode: samples at its own resolution.
struct Plane {
  int w = 0, h = 0;
  std::vector<float> s;
  float at(int x, int y) const {
    return s[static_cast<size_t>(std::min(y, h - 1)) * w + std::min(x, w - 1)];
  }
};

// Resamples a plane to (w, h): averaging the source samples a destination
// sample covers when shrinking, replicating when growing.
Plane resample(const Plane& p, int w, int h) {
  if (p.w == w && p.h == h) return p;
  Plane o;
  o.w = w;
  o.h = h;
  o.s.resize(static_cast<size_t>(w) * h);
  const int fx = std::max(1, (p.w + w - 1) / w), fy = std::max(1, (p.h + h - 1) / h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      if (p.w >= w && p.h >= h) {
        float sum = 0;
        int n = 0;
        for (int dy = 0; dy < fy; ++dy)
          for (int dx = 0; dx < fx; ++dx) {
            const int sx = x * p.w / w + dx, sy = y * p.h / h + dy;
            if (sx < p.w && sy < p.h) {
              sum += p.s[static_cast<size_t>(sy) * p.w + sx];
              ++n;
            }
          }
        o.s[static_cast<size_t>(y) * w + x] = n ? sum / n : 0;
      } else {
        o.s[static_cast<size_t>(y) * w + x] = p.at(x * p.w / w, y * p.h / h);
      }
    }
  return o;
}

// Encodes three (or, for grey, one) planes already at the target subsampling
// into a JFIF bitstream.
std::string encode_planes(const EncoderParams& prm, const std::vector<Plane>& planes, int width, int height) {
  const bool gray = planes.size() == 1;
  int hs = 1, vs = 1;
  if (!gray) sampling_of(prm.css, &hs, &vs);
  const int ncomp = gray ? 1 : 3;
  uint16_t qt[2][64];
  if (prm.custom_quant) {
    std::memcpy(qt, prm.quant, sizeof qt);
  } else {
    scale_quant(kQLum, prm.quality, qt[0]);
    scale_quant(kQChrom, prm.quality, qt[1]);
  }
  const int mcu_w = 8 * hs, mcu_h = 8 * vs;
  const int mcus_x = (width + mcu_w - 1) / mcu_w, mcus_y = (height + mcu_h - 1) / mcu_h;
  // Every block's quantised coefficients, in zig-zag order, by component.
  struct Blocks {
    int bw = 0, bh = 0;
    std::vector<int> q;
  };
  std::vector<Blocks> comp(ncomp);
  for (int c = 0; c < ncomp; ++c) {
    const int ch = c == 0 ? hs : 1, cv = c == 0 ? vs : 1;
    Blocks& b = comp[c];
    b.bw = mcus_x * ch;
    b.bh = mcus_y * cv;
    b.q.assign(static_cast<size_t>(b.bw) * b.bh * 64, 0);
    const Plane& p = planes[c];
    const uint16_t* t = qt[c == 0 ? 0 : 1];
    for (int by = 0; by < b.bh; ++by)
      for (int bx = 0; bx < b.bw; ++bx) {
        double blk[64], coef[64];
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x) blk[y * 8 + x] = p.at(bx * 8 + x, by * 8 + y) - 128.0;
        fdct8x8(blk, coef);
        int* out = b.q.data() + (static_cast<size_t>(by) * b.bw + bx) * 64;
        for (int k = 0; k < 64; ++k) out[k] = static_cast<int>(std::lround(coef[kZigZag[k]] / t[k]));
      }
  }

  const bool progressive = prm.encoding == 0xC2;
  // Huffman tables: the standard ones, or optimal ones from a counting pass.
  EncTable dct[2] = {EncTable::standard(kDcLumBits, kDcLumVals, 12), EncTable::standard(kDcChrBits, kDcChrVals, 12)};
  EncTable act[2] = {EncTable::standard(kAcLumBits, kAcLumVals, 162), EncTable::standard(kAcChrBits, kAcChrVals, 162)};
  long dc_freq[2][256] = {{0}}, ac_freq[2][256] = {{0}};

  // The scans: one interleaved sequential scan, or for progressive the DC of
  // every component, then each component's AC (spectral selection only).
  struct Scan {
    std::vector<int> comps;
    int ss, se;
  };
  std::vector<Scan> scans;
  if (!progressive) {
    scans.push_back({gray ? std::vector<int>{0} : std::vector<int>{0, 1, 2}, 0, 63});
  } else {
    scans.push_back({gray ? std::vector<int>{0} : std::vector<int>{0, 1, 2}, 0, 0});
    for (int c = 0; c < ncomp; ++c) scans.push_back({{c}, 1, 63});
  }

  const auto run_scan = [&](BitWriter& e, const Scan& sc, bool counting) {
    e.counting = counting;
    int pred[3] = {0, 0, 0};
    const auto dc = [&](int c, const int* q) {
      const int t = c == 0 ? 0 : 1;
      const int diff = q[0] - pred[c];
      pred[c] = q[0];
      const int s = magnitude(diff);
      if (counting) ++dc_freq[t][s];
      e.put(dct[t].code[s], dct[t].len[s]);
      if (s) e.put(diff > 0 ? diff : diff + (1 << s) - 1, s);
    };
    const auto ac = [&](int c, const int* q, int ss) {
      const int t = c == 0 ? 0 : 1;
      int run = 0;
      for (int k = ss; k < 64; ++k) {
        if (q[k] == 0) {
          ++run;
          continue;
        }
        while (run > 15) {
          if (counting) ++ac_freq[t][0xF0];
          e.put(act[t].code[0xF0], act[t].len[0xF0]);
          run -= 16;
        }
        const int sz = magnitude(q[k]);
        const int sym = (run << 4) | sz;
        if (counting) ++ac_freq[t][sym];
        e.put(act[t].code[sym], act[t].len[sym]);
        e.put(q[k] > 0 ? q[k] : q[k] + (1 << sz) - 1, sz);
        run = 0;
      }
      if (run) {
        if (counting) ++ac_freq[t][0x00];
        e.put(act[t].code[0x00], act[t].len[0x00]);
      }
    };
    const auto code_block = [&](int c, int bx, int by) {
      const int* q = comp[c].q.data() + (static_cast<size_t>(by) * comp[c].bw + bx) * 64;
      if (sc.ss == 0) dc(c, q);
      if (sc.se > 0) ac(c, q, std::max(1, sc.ss));
    };
    if (sc.comps.size() == 1 && ncomp > 1) {
      // A single-component scan: the component's blocks that hold image.
      const int c = sc.comps[0];
      const int cw = c == 0 ? width : (width + hs - 1) / hs, chh = c == 0 ? height : (height + vs - 1) / vs;
      for (int by = 0; by < (chh + 7) / 8; ++by)
        for (int bx = 0; bx < (cw + 7) / 8; ++bx) code_block(c, bx, by);
    } else {
      for (int my = 0; my < mcus_y; ++my)
        for (int mx = 0; mx < mcus_x; ++mx)
          for (int c : sc.comps) {
            const int ch = c == 0 ? hs : 1, cv = c == 0 ? vs : 1;
            for (int by = 0; by < cv; ++by)
              for (int bx = 0; bx < ch; ++bx) code_block(c, mx * ch + bx, my * cv + by);
          }
    }
    e.flush();
  };

  if (prm.optimized) {
    BitWriter count;
    for (const Scan& sc : scans) run_scan(count, sc, true);
    for (int t = 0; t < (gray ? 1 : 2); ++t) {
      if (std::any_of(std::begin(dc_freq[t]), std::end(dc_freq[t]), [](long f) { return f > 0; }))
        dct[t] = EncTable::optimal(dc_freq[t]);
      if (std::any_of(std::begin(ac_freq[t]), std::end(ac_freq[t]), [](long f) { return f > 0; }))
        act[t] = EncTable::optimal(ac_freq[t]);
    }
  }

  BitWriter e;
  e.marker(0xD8);
  // JFIF APP0, which is what every decoder expects to see first.
  e.marker(0xE0);
  e.word(16);
  for (char ch : {'J', 'F', 'I', 'F', '\0'}) e.byte(static_cast<uint8_t>(ch));
  e.byte(1);
  e.byte(1);
  e.byte(0);
  e.word(1);
  e.word(1);
  e.byte(0);
  e.byte(0);
  for (int t = 0; t < (gray ? 1 : 2); ++t) {
    e.marker(0xDB);
    e.word(2 + 65);
    e.byte(static_cast<uint8_t>(t));
    for (int i = 0; i < 64; ++i) e.byte(static_cast<uint8_t>(std::min<int>(qt[t][i], 255)));
  }
  e.marker(static_cast<uint8_t>(progressive ? 0xC2 : 0xC0));
  e.word(8 + 3 * ncomp);
  e.byte(8);
  e.word(height);
  e.word(width);
  e.byte(static_cast<uint8_t>(ncomp));
  e.byte(1);
  e.byte(static_cast<uint8_t>((hs << 4) | vs));
  e.byte(0);
  if (!gray) {
    e.byte(2);
    e.byte(0x11);
    e.byte(1);
    e.byte(3);
    e.byte(0x11);
    e.byte(1);
  }
  const auto dht = [&](int tc, int th, const EncTable& t) {
    e.marker(0xC4);
    e.word(static_cast<int>(3 + 16 + t.vals.size()));
    e.byte(static_cast<uint8_t>((tc << 4) | th));
    for (int l = 1; l <= 16; ++l) e.byte(t.bits[l]);
    for (uint8_t v : t.vals) e.byte(v);
  };
  for (int t = 0; t < (gray ? 1 : 2); ++t) {
    dht(0, t, dct[t]);
    dht(1, t, act[t]);
  }
  for (const Scan& sc : scans) {
    e.marker(0xDA);
    e.word(6 + 2 * static_cast<int>(sc.comps.size()));
    e.byte(static_cast<uint8_t>(sc.comps.size()));
    for (int c : sc.comps) {
      e.byte(static_cast<uint8_t>(c + 1));
      e.byte(c == 0 ? 0x00 : 0x11);
    }
    e.byte(static_cast<uint8_t>(sc.ss));
    e.byte(static_cast<uint8_t>(sc.se));
    e.byte(0);
    run_scan(e, sc, false);
  }
  e.marker(0xD9);
  return std::move(e.out);
}

// Planes from a source image of `fmt` (the RGB family), at full resolution,
// converted to YCbCr.
nvjpegStatus_t read_rgb(const nvjpegImage_t* src, nvjpegInputFormat_t fmt, int width, int height, std::vector<Plane>* ycc) {
  const size_t n = static_cast<size_t>(width) * height;
  std::vector<uint8_t> r(n), g(n), b(n);
  const auto read = [&](int idx, std::vector<uint8_t>* dst, int w) -> bool {
    if (!src->channel[idx] || src->pitch[idx] < static_cast<size_t>(w)) return false;
    dst->assign(static_cast<size_t>(w) * height, 0);
    return cudaMemcpy2D(dst->data(), w, src->channel[idx], src->pitch[idx], w, height, cudaMemcpyDeviceToHost) ==
           cudaSuccess;
  };
  if (fmt == NVJPEG_INPUT_RGBI || fmt == NVJPEG_INPUT_BGRI) {
    std::vector<uint8_t> il;
    if (!read(0, &il, width * 3)) return NVJPEG_STATUS_INVALID_PARAMETER;
    const bool bgr = fmt == NVJPEG_INPUT_BGRI;
    for (size_t i = 0; i < n; ++i) {
      r[i] = il[i * 3 + (bgr ? 2 : 0)];
      g[i] = il[i * 3 + 1];
      b[i] = il[i * 3 + (bgr ? 0 : 2)];
    }
  } else if (fmt == NVJPEG_INPUT_RGB || fmt == NVJPEG_INPUT_BGR) {
    const bool bgr = fmt == NVJPEG_INPUT_BGR;
    if (!read(0, bgr ? &b : &r, width) || !read(1, &g, width) || !read(2, bgr ? &r : &b, width))
      return NVJPEG_STATUS_INVALID_PARAMETER;
  } else {
    return NVJPEG_STATUS_INVALID_PARAMETER;
  }
  ycc->assign(3, Plane{});
  for (Plane& p : *ycc) {
    p.w = width;
    p.h = height;
    p.s.resize(n);
  }
  for (size_t i = 0; i < n; ++i) {
    (*ycc)[0].s[i] = 0.299f * r[i] + 0.587f * g[i] + 0.114f * b[i];
    (*ycc)[1].s[i] = 128.0f - 0.168736f * r[i] - 0.331264f * g[i] + 0.5f * b[i];
    (*ycc)[2].s[i] = 128.0f + 0.5f * r[i] - 0.418688f * g[i] - 0.081312f * b[i];
  }
  return NVJPEG_STATUS_SUCCESS;
}

// Brings full-resolution (or any-resolution) Y, Cb, Cr to the subsampling the
// parameters ask for.
std::vector<Plane> to_target(const EncoderParams& prm, std::vector<Plane> ycc, int width, int height) {
  if (prm.css == NVJPEG_CSS_GRAY) {
    ycc.resize(1);
    ycc[0] = resample(ycc[0], width, height);
    return ycc;
  }
  int hs, vs;
  sampling_of(prm.css, &hs, &vs);
  ycc[0] = resample(ycc[0], width, height);
  for (int c = 1; c < 3; ++c) ycc[c] = resample(ycc[c], (width + hs - 1) / hs, (height + vs - 1) / vs);
  return ycc;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ======================================================================== */
/* Library and handle                                                       */
/* ======================================================================== */

VGPU_EXPORT nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t* handle) {
  if (!handle) return NVJPEG_STATUS_INVALID_PARAMETER;
  *handle = reinterpret_cast<nvjpegHandle_t>(track(new Handle(), Kind::Handle));
  if (!quiet())
    std::fprintf(stderr, "[vgpu] nvJPEG ready (host codec; see nvidia/docs/libraries.md)\n");
  return NVJPEG_STATUS_SUCCESS;
}
// The hardware backends need the JPEG engine an A100 or H100 has and an RTX
// 3060 does not; there the card's library answers NVJPEG_STATUS_ARCH_MISMATCH,
// and so does this one, on every profile.
static bool hardware_backend(int b) { return b == 3 || b == 5; }   // NVJPEG_BACKEND_HARDWARE(_DEVICE)
VGPU_EXPORT nvjpegStatus_t nvjpegCreate(nvjpegBackend_t backend, nvjpegDevAllocator_t*, nvjpegHandle_t* handle) {
  if (hardware_backend(backend)) return NVJPEG_STATUS_ARCH_MISMATCH;
  return nvjpegCreateSimple(handle);
}
VGPU_EXPORT nvjpegStatus_t nvjpegCreateEx(nvjpegBackend_t backend, nvjpegDevAllocator_t*, nvjpegPinnedAllocator_t*,
                                          unsigned int, nvjpegHandle_t* handle) {
  if (hardware_backend(backend)) return NVJPEG_STATUS_ARCH_MISMATCH;
  return nvjpegCreateSimple(handle);
}
VGPU_EXPORT nvjpegStatus_t nvjpegCreateExV2(nvjpegBackend_t backend, nvjpegDevAllocatorV2_t*, nvjpegPinnedAllocatorV2_t*,
                                            unsigned int, nvjpegHandle_t* handle) {
  if (hardware_backend(backend)) return NVJPEG_STATUS_ARCH_MISMATCH;
  return nvjpegCreateSimple(handle);
}
VGPU_EXPORT nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t h) { return destroy<Handle>(h, Kind::Handle); }
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
VGPU_EXPORT nvjpegStatus_t nvjpegSetDeviceMemoryPadding(size_t, nvjpegHandle_t h) {
  return known(h, Kind::Handle) ? NVJPEG_STATUS_SUCCESS : NVJPEG_STATUS_INVALID_PARAMETER;
}
VGPU_EXPORT nvjpegStatus_t nvjpegSetPinnedMemoryPadding(size_t, nvjpegHandle_t h) {
  return known(h, Kind::Handle) ? NVJPEG_STATUS_SUCCESS : NVJPEG_STATUS_INVALID_PARAMETER;
}
VGPU_EXPORT nvjpegStatus_t nvjpegGetDeviceMemoryPadding(size_t* padding, nvjpegHandle_t h) {
  if (!known(h, Kind::Handle) || !padding) return NVJPEG_STATUS_INVALID_PARAMETER;
  *padding = 0;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegGetPinnedMemoryPadding(size_t* padding, nvjpegHandle_t h) {
  return nvjpegGetDeviceMemoryPadding(padding, h);
}
// No JPEG engine (see hardware_backend).
VGPU_EXPORT nvjpegStatus_t nvjpegGetHardwareDecoderInfo(nvjpegHandle_t h, unsigned int* engines, unsigned int* cores) {
  if (!known(h, Kind::Handle)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (engines) *engines = 0;
  if (cores) *cores = 0;
  return NVJPEG_STATUS_ARCH_MISMATCH;
}
VGPU_EXPORT nvjpegStatus_t nvjpegGetHardwareEncoderInfo(nvjpegHandle_t h, unsigned int* engines) {
  if (!known(h, Kind::Handle)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (engines) *engines = 0;
  return NVJPEG_STATUS_ARCH_MISMATCH;
}

VGPU_EXPORT nvjpegStatus_t nvjpegJpegStateCreate(nvjpegHandle_t h, nvjpegJpegState_t* s) {
  if (!known(h, Kind::Handle) || !s) return NVJPEG_STATUS_INVALID_PARAMETER;
  *s = reinterpret_cast<nvjpegJpegState_t>(track(new State(), Kind::State));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStateDestroy(nvjpegJpegState_t s) { return destroy<State>(s, Kind::State); }

/* ======================================================================== */
/* Single-image decode                                                      */
/* ======================================================================== */

VGPU_EXPORT nvjpegStatus_t nvjpegGetImageInfo(nvjpegHandle_t h, const unsigned char* data, size_t length, int* ncomp,
                                              nvjpegChromaSubsampling_t* css, int* widths, int* heights) {
  if (!known(h, Kind::Handle)) return NVJPEG_STATUS_INVALID_PARAMETER;
  Image im;
  const nvjpegStatus_t st = parse_jpeg(data, length, &im, /*decode=*/false);
  if (st != NVJPEG_STATUS_SUCCESS) return st;
  if (ncomp) *ncomp = static_cast<int>(im.comps.size());
  if (css) *css = subsampling_of(im);
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; ++c) {
    const bool have = c < static_cast<int>(im.comps.size());
    if (widths) widths[c] = have ? comp_width(im, c) : 0;
    if (heights) heights[c] = have ? comp_height(im, c) : 0;
  }
  return NVJPEG_STATUS_SUCCESS;
}

VGPU_EXPORT nvjpegStatus_t nvjpegDecode(nvjpegHandle_t h, nvjpegJpegState_t s, const unsigned char* data,
                                        size_t length, nvjpegOutputFormat_t fmt, nvjpegImage_t* dst,
                                        cudaStream_t stream) {
  if (!known(h, Kind::Handle) || !known(s, Kind::State) || !dst || !data) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (static_cast<int>(fmt) < 0 || static_cast<int>(fmt) > kOutYUY2) return NVJPEG_STATUS_INVALID_PARAMETER;
  Image im;
  nvjpegStatus_t st = parse_jpeg(data, length, &im, /*decode=*/true);
  if (st != NVJPEG_STATUS_SUCCESS) return st;
  cudaStreamSynchronize(stream);
  OutputSpec o;
  o.format = fmt;
  return write_image(im, o, dst);
}

/* ======================================================================== */
/* Batched decode                                                           */
/* ======================================================================== */

// The argument checks are the card's (nvJPEG 13.0, RTX 3060, the default
// backend): a batch of 0, no CPU threads, or a format past the 8-bit ones is
// NVJPEG_STATUS_INVALID_PARAMETER. A call to nvjpegDecodeBatched before
// initialising crashed NVIDIA's library; here it is
// NVJPEG_STATUS_INVALID_PARAMETER.
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeBatchedInitialize(nvjpegHandle_t h, nvjpegJpegState_t s, int batch_size,
                                                         int max_cpu_threads, nvjpegOutputFormat_t fmt) {
  if (!known(h, Kind::Handle) || !known(s, Kind::State)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (batch_size <= 0 || max_cpu_threads <= 0) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (static_cast<int>(fmt) < 0 || static_cast<int>(fmt) > kOutYUY2 || static_cast<int>(fmt) == kOutUnchangedU16)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& st = *reinterpret_cast<State*>(s);
  st.batch_size = batch_size;
  st.batch_format = fmt;
  return NVJPEG_STATUS_SUCCESS;
}

// Every image of the batch is decoded, into its own destination; the result
// is the first failure, or success. A truncated image in the batch gave the
// card's NVJPEG_STATUS_INCOMPLETE_BITSTREAM.
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeBatched(nvjpegHandle_t h, nvjpegJpegState_t s, const unsigned char* const* data,
                                               const size_t* lengths, nvjpegImage_t* dst, cudaStream_t stream) {
  if (!known(h, Kind::Handle) || !known(s, Kind::State) || !data || !lengths || !dst)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& st = *reinterpret_cast<State*>(s);
  if (st.batch_size <= 0) return NVJPEG_STATUS_INVALID_PARAMETER;
  cudaStreamSynchronize(stream);
  nvjpegStatus_t first = NVJPEG_STATUS_SUCCESS;
  for (int i = 0; i < st.batch_size; ++i) {
    nvjpegStatus_t r;
    if (!data[i]) {
      r = NVJPEG_STATUS_INVALID_PARAMETER;
    } else {
      Image im;
      r = parse_jpeg(data[i], lengths[i], &im, /*decode=*/true);
      if (r == NVJPEG_STATUS_SUCCESS) {
        OutputSpec o;
        o.format = st.batch_format;
        r = write_image(im, o, &dst[i]);
      }
    }
    if (r != NVJPEG_STATUS_SUCCESS && first == NVJPEG_STATUS_SUCCESS) first = r;
  }
  return first;
}

// nvjpegDecodeBatchedEx and the JPEG-table and pre-allocation calls belong to
// the hardware backend: on the card's default backend BatchedEx answers
// NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED, ParseJpegTables
// NVJPEG_STATUS_JPEG_NOT_SUPPORTED, and PreAllocate succeeds and does nothing.
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeBatchedEx(nvjpegHandle_t h, nvjpegJpegState_t s, const unsigned char* const*,
                                                 const size_t*, nvjpegImage_t*, nvjpegDecodeParams_t*, cudaStream_t) {
  if (!known(h, Kind::Handle) || !known(s, Kind::State)) return NVJPEG_STATUS_INVALID_PARAMETER;
  return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeBatchedPreAllocate(nvjpegHandle_t h, nvjpegJpegState_t s, int, int, int,
                                                          nvjpegChromaSubsampling_t, nvjpegOutputFormat_t) {
  if (!known(h, Kind::Handle) || !known(s, Kind::State)) return NVJPEG_STATUS_INVALID_PARAMETER;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeBatchedParseJpegTables(nvjpegHandle_t h, nvjpegJpegState_t s,
                                                              const unsigned char*, const size_t) {
  if (!known(h, Kind::Handle) || !known(s, Kind::State)) return NVJPEG_STATUS_INVALID_PARAMETER;
  return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
}

/* ======================================================================== */
/* JPEG streams                                                             */
/* ======================================================================== */

VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamCreate(nvjpegHandle_t h, nvjpegJpegStream_t* js) {
  if (!known(h, Kind::Handle) || !js) return NVJPEG_STATUS_INVALID_PARAMETER;
  *js = reinterpret_cast<nvjpegJpegStream_t>(track(new Stream(), Kind::Stream));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamDestroy(nvjpegJpegStream_t js) { return destroy<Stream>(js, Kind::Stream); }

static nvjpegStatus_t stream_parse(nvjpegHandle_t h, const unsigned char* data, size_t length, nvjpegJpegStream_t js) {
  if (!known(h, Kind::Handle) || !known(js, Kind::Stream) || !data) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& s = *reinterpret_cast<Stream*>(js);
  s.parsed = false;
  s.header = Image{};
  const nvjpegStatus_t st = parse_jpeg(data, length, &s.header, /*decode=*/false);
  if (st != NVJPEG_STATUS_SUCCESS) return st;
  // The data is kept, whatever save_stream says: the decode reads it later.
  s.data.assign(data, data + length);
  s.parsed = true;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamParse(nvjpegHandle_t h, const unsigned char* data, size_t length, int, int,
                                                 nvjpegJpegStream_t js) {
  return stream_parse(h, data, length, js);
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamParseHeader(nvjpegHandle_t h, const unsigned char* data, size_t length,
                                                       nvjpegJpegStream_t js) {
  return stream_parse(h, data, length, js);
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamParseTables(nvjpegHandle_t h, const unsigned char* data, size_t length,
                                                       nvjpegJpegStream_t js) {
  if (!known(h, Kind::Handle) || !known(js, Kind::Stream) || !data || !length) return NVJPEG_STATUS_INVALID_PARAMETER;
  return NVJPEG_STATUS_SUCCESS;
}
static const Stream* parsed(nvjpegJpegStream_t js) {
  return known(js, Kind::Stream) && reinterpret_cast<Stream*>(js)->parsed ? reinterpret_cast<Stream*>(js) : nullptr;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamGetJpegEncoding(nvjpegJpegStream_t js, nvjpegJpegEncoding_t* enc) {
  const Stream* s = parsed(js);
  if (!s || !enc) return NVJPEG_STATUS_INVALID_PARAMETER;
  *enc = static_cast<nvjpegJpegEncoding_t>(s->header.encoding);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamGetFrameDimensions(nvjpegJpegStream_t js, unsigned int* w, unsigned int* h) {
  const Stream* s = parsed(js);
  if (!s || !w || !h) return NVJPEG_STATUS_INVALID_PARAMETER;
  *w = static_cast<unsigned>(s->header.width);
  *h = static_cast<unsigned>(s->header.height);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamGetComponentsNum(nvjpegJpegStream_t js, unsigned int* n) {
  const Stream* s = parsed(js);
  if (!s || !n) return NVJPEG_STATUS_INVALID_PARAMETER;
  *n = static_cast<unsigned>(s->header.comps.size());
  return NVJPEG_STATUS_SUCCESS;
}
// A component the image does not have: NVJPEG_STATUS_INVALID_PARAMETER, as
// for component 1 of a grey image on the card.
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamGetComponentDimensions(nvjpegJpegStream_t js, unsigned int c,
                                                                  unsigned int* w, unsigned int* h) {
  const Stream* s = parsed(js);
  if (!s || !w || !h || c >= s->header.comps.size()) return NVJPEG_STATUS_INVALID_PARAMETER;
  *w = static_cast<unsigned>(comp_width(s->header, static_cast<int>(c)));
  *h = static_cast<unsigned>(comp_height(s->header, static_cast<int>(c)));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamGetChromaSubsampling(nvjpegJpegStream_t js, nvjpegChromaSubsampling_t* css) {
  const Stream* s = parsed(js);
  if (!s || !css) return NVJPEG_STATUS_INVALID_PARAMETER;
  *css = subsampling_of(s->header);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamGetExifOrientation(nvjpegJpegStream_t js, nvjpegExifOrientation_t* o) {
  const Stream* s = parsed(js);
  if (!s || !o) return NVJPEG_STATUS_INVALID_PARAMETER;
  *o = static_cast<nvjpegExifOrientation_t>(s->header.orientation);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegJpegStreamGetSamplePrecision(nvjpegJpegStream_t js, unsigned int* precision) {
  const Stream* s = parsed(js);
  if (!s || !precision) return NVJPEG_STATUS_INVALID_PARAMETER;
  *precision = static_cast<unsigned>(s->header.precision);
  return NVJPEG_STATUS_SUCCESS;
}

/* ======================================================================== */
/* Decode parameters, decoders, buffers                                     */
/* ======================================================================== */

VGPU_EXPORT nvjpegStatus_t nvjpegDecodeParamsCreate(nvjpegHandle_t h, nvjpegDecodeParams_t* p) {
  if (!known(h, Kind::Handle) || !p) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto* dp = new DecodeParams();
  dp->out.format = NVJPEG_OUTPUT_RGB;
  *p = reinterpret_cast<nvjpegDecodeParams_t>(track(dp, Kind::DecodeParams));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeParamsDestroy(nvjpegDecodeParams_t p) {
  return destroy<DecodeParams>(p, Kind::DecodeParams);
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetOutputFormat(nvjpegDecodeParams_t p, nvjpegOutputFormat_t fmt) {
  if (!known(p, Kind::DecodeParams)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (static_cast<int>(fmt) < 0 || static_cast<int>(fmt) > kOutYUY2) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<DecodeParams*>(p)->out.format = fmt;
  return NVJPEG_STATUS_SUCCESS;
}
// Measured: a negative offset is refused here; a region past the image is
// accepted and refused by the decode (NVJPEG_STATUS_BAD_JPEG on the card's
// host phase); a zero or negative size means the whole image.
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetROI(nvjpegDecodeParams_t p, int x, int y, int w, int h) {
  if (!known(p, Kind::DecodeParams) || x < 0 || y < 0) return NVJPEG_STATUS_INVALID_PARAMETER;
  OutputSpec& o = reinterpret_cast<DecodeParams*>(p)->out;
  o.roi = w > 0 && h > 0;
  o.rx = x;
  o.ry = y;
  o.rw = w;
  o.rh = h;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetAllowCMYK(nvjpegDecodeParams_t p, int allow) {
  if (!known(p, Kind::DecodeParams)) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<DecodeParams*>(p)->out.allow_cmyk = allow != 0;
  return NVJPEG_STATUS_SUCCESS;
}
// Scaling and orientation are the hardware backend's: the card's default
// backend accepted both here and then refused the decode (scaling with
// NVJPEG_STATUS_INVALID_PARAMETER in the host phase, an orientation other than
// normal with NVJPEG_STATUS_EXECUTION_FAILED in the device phase); this one
// refuses both in the host phase with NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED.
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetScaleFactor(nvjpegDecodeParams_t p, nvjpegScaleFactor_t f) {
  if (!known(p, Kind::DecodeParams) || static_cast<int>(f) < 0 || static_cast<int>(f) > 3)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<DecodeParams*>(p)->scale = f;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetExifOrientation(nvjpegDecodeParams_t p, nvjpegExifOrientation_t o) {
  if (!known(p, Kind::DecodeParams) || static_cast<int>(o) < 0 || static_cast<int>(o) > 8)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<DecodeParams*>(p)->orientation = o;
  return NVJPEG_STATUS_SUCCESS;
}

VGPU_EXPORT nvjpegStatus_t nvjpegDecoderCreate(nvjpegHandle_t h, nvjpegBackend_t backend, nvjpegJpegDecoder_t* d) {
  if (!known(h, Kind::Handle) || !d) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (hardware_backend(backend)) return NVJPEG_STATUS_ARCH_MISMATCH;
  if (static_cast<int>(backend) == 6) return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;   // lossless
  if (static_cast<int>(backend) < 0 || static_cast<int>(backend) > 6) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto* dec = new Decoder();
  dec->backend = backend;
  *d = reinterpret_cast<nvjpegJpegDecoder_t>(track(dec, Kind::Decoder));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecoderDestroy(nvjpegJpegDecoder_t d) { return destroy<Decoder>(d, Kind::Decoder); }
VGPU_EXPORT nvjpegStatus_t nvjpegDecoderStateCreate(nvjpegHandle_t h, nvjpegJpegDecoder_t d, nvjpegJpegState_t* s) {
  if (!known(h, Kind::Handle) || !known(d, Kind::Decoder) || !s) return NVJPEG_STATUS_INVALID_PARAMETER;
  *s = reinterpret_cast<nvjpegJpegState_t>(track(new State(), Kind::State));
  return NVJPEG_STATUS_SUCCESS;
}

// Whether a decoder takes a stream: 0 for yes (the API's convention), as the
// card answered for baseline, progressive, grey and CMYK alike.
static int supported(const Stream* s) {
  if (!s) return 1;
  const Image& im = s->header;
  return im.precision == 8 && (im.encoding == 0xC0 || im.encoding == 0xC1 || im.encoding == 0xC2) &&
                 im.comps.size() != 2
             ? 0
             : 1;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecoderJpegSupported(nvjpegJpegDecoder_t d, nvjpegJpegStream_t js,
                                                      nvjpegDecodeParams_t, int* is_supported) {
  if (!known(d, Kind::Decoder) || !parsed(js) || !is_supported) return NVJPEG_STATUS_INVALID_PARAMETER;
  *is_supported = supported(parsed(js));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeBatchedSupported(nvjpegHandle_t h, nvjpegJpegStream_t js, int* is_supported) {
  if (!known(h, Kind::Handle) || !parsed(js) || !is_supported) return NVJPEG_STATUS_INVALID_PARAMETER;
  *is_supported = supported(parsed(js));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeBatchedSupportedEx(nvjpegHandle_t h, nvjpegJpegStream_t js,
                                                          nvjpegDecodeParams_t, int* is_supported) {
  return nvjpegDecodeBatchedSupported(h, js, is_supported);
}

static nvjpegStatus_t buffer_create(nvjpegHandle_t h, void** out, bool device) {
  if (!known(h, Kind::Handle) || !out) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto* b = new Buffer();
  b->device = device;
  *out = track(b, device ? Kind::BufferDevice : Kind::BufferPinned);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferPinnedCreate(nvjpegHandle_t h, nvjpegPinnedAllocator_t*, nvjpegBufferPinned_t* b) {
  return buffer_create(h, reinterpret_cast<void**>(b), false);
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferPinnedCreateV2(nvjpegHandle_t h, nvjpegPinnedAllocatorV2_t*, nvjpegBufferPinned_t* b) {
  return buffer_create(h, reinterpret_cast<void**>(b), false);
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferDeviceCreate(nvjpegHandle_t h, nvjpegDevAllocator_t*, nvjpegBufferDevice_t* b) {
  return buffer_create(h, reinterpret_cast<void**>(b), true);
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferDeviceCreateV2(nvjpegHandle_t h, nvjpegDevAllocatorV2_t*, nvjpegBufferDevice_t* b) {
  return buffer_create(h, reinterpret_cast<void**>(b), true);
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferPinnedDestroy(nvjpegBufferPinned_t b) {
  return destroy<Buffer>(b, Kind::BufferPinned);
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferDeviceDestroy(nvjpegBufferDevice_t b) {
  return destroy<Buffer>(b, Kind::BufferDevice);
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferPinnedResize(nvjpegBufferPinned_t b, size_t size, cudaStream_t) {
  if (!known(b, Kind::BufferPinned)) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<Buffer*>(b)->resize(size);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferDeviceResize(nvjpegBufferDevice_t b, size_t size, cudaStream_t) {
  if (!known(b, Kind::BufferDevice)) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<Buffer*>(b)->resize(size);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferPinnedRetrieve(nvjpegBufferPinned_t b, size_t* size, void** ptr) {
  if (!known(b, Kind::BufferPinned)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (size) *size = reinterpret_cast<Buffer*>(b)->size;
  if (ptr) *ptr = reinterpret_cast<Buffer*>(b)->ptr;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegBufferDeviceRetrieve(nvjpegBufferDevice_t b, size_t* size, void** ptr) {
  if (!known(b, Kind::BufferDevice)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (size) *size = reinterpret_cast<Buffer*>(b)->size;
  if (ptr) *ptr = reinterpret_cast<Buffer*>(b)->ptr;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegStateAttachPinnedBuffer(nvjpegJpegState_t s, nvjpegBufferPinned_t b) {
  if (!known(s, Kind::State) || (b && !known(b, Kind::BufferPinned))) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<State*>(s)->pinned = reinterpret_cast<Buffer*>(b);
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegStateAttachDeviceBuffer(nvjpegJpegState_t s, nvjpegBufferDevice_t b) {
  if (!known(s, Kind::State) || (b && !known(b, Kind::BufferDevice))) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<State*>(s)->device = reinterpret_cast<Buffer*>(b);
  return NVJPEG_STATUS_SUCCESS;
}

/* ======================================================================== */
/* Decoupled decode                                                         */
/* ======================================================================== */

// The host phase: the whole decode, kept in the state. The card's library ran
// it without buffers attached, and so does this one; buffers attached are
// sized as NVIDIA's are used (the compressed data pinned, the planes on the
// device) so nvjpegBuffer*Retrieve reports something real.
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeJpegHost(nvjpegHandle_t h, nvjpegJpegDecoder_t d, nvjpegJpegState_t s,
                                                nvjpegDecodeParams_t p, nvjpegJpegStream_t js) {
  if (!known(h, Kind::Handle) || !known(d, Kind::Decoder) || !known(s, Kind::State) || !known(p, Kind::DecodeParams))
    return NVJPEG_STATUS_INVALID_PARAMETER;
  const Stream* stream = parsed(js);
  if (!stream) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& st = *reinterpret_cast<State*>(s);
  const auto& prm = *reinterpret_cast<DecodeParams*>(p);
  st.phase = 0;
  st.image.reset();
  if (prm.scale != 0 || (prm.orientation != 1 && prm.orientation != 0))
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
  auto im = std::make_unique<Image>();
  nvjpegStatus_t r = parse_jpeg(stream->data.data(), stream->data.size(), im.get(), /*decode=*/true);
  if (r != NVJPEG_STATUS_SUCCESS) return r;
  if (const nvjpegStatus_t fr = check_format(*im, prm.out); fr != NVJPEG_STATUS_SUCCESS) return fr;
  if (prm.out.roi) {
    // Only the RGB family and grey take a region (YUV was refused on the
    // card), and only one inside the image.
    if (prm.out.format == NVJPEG_OUTPUT_UNCHANGED || prm.out.format == NVJPEG_OUTPUT_YUV ||
        prm.out.format == kOutNV12 || prm.out.format == kOutYUY2)
      return NVJPEG_STATUS_INVALID_PARAMETER;
    if (prm.out.rx + prm.out.rw > im->width || prm.out.ry + prm.out.rh > im->height) return NVJPEG_STATUS_BAD_JPEG;
  }
  if (st.pinned) st.pinned->resize(stream->data.size());
  if (st.device) {
    size_t planes = 0;
    for (const Component& c : im->comps) planes += c.plane.size();
    st.device->resize(planes);
  }
  st.image = std::move(im);
  st.out = prm.out;
  st.phase = 1;
  return NVJPEG_STATUS_SUCCESS;
}
// Before the host phase, and without a device buffer attached to the state
// (a pinned one is not needed), the card answered
// NVJPEG_STATUS_INVALID_PARAMETER -- and so did its nvjpegDecodeJpeg.
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeJpegTransferToDevice(nvjpegHandle_t h, nvjpegJpegDecoder_t d,
                                                            nvjpegJpegState_t s, nvjpegJpegStream_t, cudaStream_t) {
  if (!known(h, Kind::Handle) || !known(d, Kind::Decoder) || !known(s, Kind::State))
    return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& st = *reinterpret_cast<State*>(s);
  if (st.phase < 1 || !st.image || !st.device || !known(st.device, Kind::BufferDevice))
    return NVJPEG_STATUS_INVALID_PARAMETER;
  st.phase = 2;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeJpegDevice(nvjpegHandle_t h, nvjpegJpegDecoder_t d, nvjpegJpegState_t s,
                                                  nvjpegImage_t* dst, cudaStream_t stream) {
  if (!known(h, Kind::Handle) || !known(d, Kind::Decoder) || !known(s, Kind::State) || !dst)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& st = *reinterpret_cast<State*>(s);
  if (st.phase < 2 || !st.image) return NVJPEG_STATUS_INVALID_PARAMETER;
  cudaStreamSynchronize(stream);
  return write_image(*st.image, st.out, dst);
}
VGPU_EXPORT nvjpegStatus_t nvjpegDecodeJpeg(nvjpegHandle_t h, nvjpegJpegDecoder_t d, nvjpegJpegState_t s,
                                            nvjpegJpegStream_t js, nvjpegImage_t* dst, nvjpegDecodeParams_t p,
                                            cudaStream_t stream) {
  nvjpegStatus_t r = nvjpegDecodeJpegHost(h, d, s, p, js);
  if (r == NVJPEG_STATUS_SUCCESS) r = nvjpegDecodeJpegTransferToDevice(h, d, s, js, stream);
  if (r == NVJPEG_STATUS_SUCCESS) r = nvjpegDecodeJpegDevice(h, d, s, dst, stream);
  return r;
}

/* ======================================================================== */
/* Encode                                                                   */
/* ======================================================================== */

VGPU_EXPORT nvjpegStatus_t nvjpegEncoderStateCreate(nvjpegHandle_t h, nvjpegEncoderState_t* st, cudaStream_t) {
  if (!known(h, Kind::Handle) || !st) return NVJPEG_STATUS_INVALID_PARAMETER;
  *st = reinterpret_cast<nvjpegEncoderState_t>(track(new EncoderState(), Kind::EncoderState));
  return NVJPEG_STATUS_SUCCESS;
}
// NVJPEG_ENC_BACKEND_HARDWARE needs a JPEG engine: NVJPEG_STATUS_ARCH_MISMATCH,
// as on an RTX 3060.
// (nvjpegEncBackend_t is CUDA 13's; CUDA 12.0's header has neither it nor
// this function.)
#if NVJPEG_VER_MAJOR >= 13
#define VGPU_ENC_BACKEND nvjpegEncBackend_t
#else
#define VGPU_ENC_BACKEND int
#endif
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderStateCreateWithBackend(nvjpegHandle_t h, nvjpegEncoderState_t* st,
                                                               VGPU_ENC_BACKEND backend, cudaStream_t stream) {
  if (static_cast<int>(backend) == 2) return NVJPEG_STATUS_ARCH_MISMATCH;
  if (static_cast<int>(backend) < 0 || static_cast<int>(backend) > 2) return NVJPEG_STATUS_INVALID_PARAMETER;
  return nvjpegEncoderStateCreate(h, st, stream);
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderStateDestroy(nvjpegEncoderState_t st) {
  return destroy<EncoderState>(st, Kind::EncoderState);
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsCreate(nvjpegHandle_t h, nvjpegEncoderParams_t* p, cudaStream_t) {
  if (!known(h, Kind::Handle) || !p) return NVJPEG_STATUS_INVALID_PARAMETER;
  *p = reinterpret_cast<nvjpegEncoderParams_t>(track(new EncoderParams(), Kind::EncoderParams));
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsDestroy(nvjpegEncoderParams_t p) {
  return destroy<EncoderParams>(p, Kind::EncoderParams);
}
// 1 to 100, as the card's library takes.
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetQuality(nvjpegEncoderParams_t p, const int q, cudaStream_t) {
  if (!known(p, Kind::EncoderParams) || q < 1 || q > 100) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& prm = *reinterpret_cast<EncoderParams*>(p);
  prm.quality = q;
  prm.custom_quant = false;
  return NVJPEG_STATUS_SUCCESS;
}
// Baseline and progressive, as NVIDIA's 13.0 encoder takes; extended
// sequential and lossless it refused (NVJPEG_STATUS_INVALID_PARAMETER).
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetEncoding(nvjpegEncoderParams_t p, nvjpegJpegEncoding_t e, cudaStream_t) {
  if (!known(p, Kind::EncoderParams)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (e != NVJPEG_ENCODING_BASELINE_DCT && e != NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<EncoderParams*>(p)->encoding = e;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetOptimizedHuffman(nvjpegEncoderParams_t p, int optimized, cudaStream_t) {
  if (!known(p, Kind::EncoderParams)) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<EncoderParams*>(p)->optimized = optimized != 0;
  return NVJPEG_STATUS_SUCCESS;
}
// Every subsampling but 4:1:0 vertical (NVJPEG_STATUS_JPEG_NOT_SUPPORTED on
// the card) and unknown (NVJPEG_STATUS_INVALID_PARAMETER).
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetSamplingFactors(nvjpegEncoderParams_t p,
                                                                 const nvjpegChromaSubsampling_t c, cudaStream_t) {
  if (!known(p, Kind::EncoderParams)) return NVJPEG_STATUS_INVALID_PARAMETER;
  if (static_cast<int>(c) == 7) return NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
  if (static_cast<int>(c) < 0 || static_cast<int>(c) > 6) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<EncoderParams*>(p)->css = c;
  return NVJPEG_STATUS_SUCCESS;
}
// Taken up to 65535 and then not used: NVIDIA's 13.0 encoder wrote the same
// bitstream, with no DRI and no restart markers, whatever the interval.
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetRestartInterval(nvjpegEncoderParams_t p, unsigned int r, cudaStream_t) {
  if (!known(p, Kind::EncoderParams) || r > 65535) return NVJPEG_STATUS_INVALID_PARAMETER;
  reinterpret_cast<EncoderParams*>(p)->restart = r;
  return NVJPEG_STATUS_SUCCESS;
}
// NVIDIA's upper bound on a bitstream, as its 13.0 computes it (measured for
// every subsampling at four sizes): bytes per pixel of 6 (4:4:4, 4:4:0), 3
// (4:2:2, 4:2:0) or 2 (4:1:1, 4:1:0, grey), over the image padded to 16 (32
// for 4:1:x), plus 2048.
VGPU_EXPORT nvjpegStatus_t nvjpegEncodeGetBufferSize(nvjpegHandle_t h, const nvjpegEncoderParams_t p, int w, int ht,
                                                     size_t* max_len) {
  if (!known(h, Kind::Handle) || !known(p, Kind::EncoderParams) || !max_len || w <= 0 || ht <= 0)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  const nvjpegChromaSubsampling_t css = reinterpret_cast<EncoderParams*>(p)->css;
  int bpp = 2, pad = 16;
  if (css == NVJPEG_CSS_444 || css == NVJPEG_CSS_440) bpp = 6;
  if (css == NVJPEG_CSS_422 || css == NVJPEG_CSS_420) bpp = 3;
  if (css == NVJPEG_CSS_411 || css == NVJPEG_CSS_410) pad = 32;
  const size_t pw = (static_cast<size_t>(w) + pad - 1) / pad * pad, ph = (static_cast<size_t>(ht) + pad - 1) / pad * pad;
  *max_len = bpp * pw * ph + 2048;
  return NVJPEG_STATUS_SUCCESS;
}

// From the RGB family (planar or interleaved). YUV and NV12 inputs belong to
// nvjpegEncodeYUV and nvjpegEncode; here the card refused them
// (NVJPEG_STATUS_INVALID_PARAMETER).
VGPU_EXPORT nvjpegStatus_t nvjpegEncodeImage(nvjpegHandle_t h, nvjpegEncoderState_t st, const nvjpegEncoderParams_t pp,
                                             const nvjpegImage_t* src, nvjpegInputFormat_t ifmt, int width, int height,
                                             cudaStream_t stream) {
  if (!known(h, Kind::Handle) || !known(st, Kind::EncoderState) || !known(pp, Kind::EncoderParams) || !src)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  if (width <= 0 || height <= 0) return NVJPEG_STATUS_INVALID_PARAMETER;
  cudaStreamSynchronize(stream);
  std::vector<Plane> ycc;
  if (const nvjpegStatus_t r = read_rgb(src, ifmt, width, height, &ycc); r != NVJPEG_STATUS_SUCCESS) return r;
  const auto& prm = *reinterpret_cast<const EncoderParams*>(pp);
  auto& state = *reinterpret_cast<EncoderState*>(st);
  state.bitstream = encode_planes(prm, to_target(prm, std::move(ycc), width, height), width, height);
  state.on_device = true;
  return NVJPEG_STATUS_SUCCESS;
}

// Y, Cb, Cr planes of the subsampling `in_css` (Cb and Cr interleaved in
// channel[1] for NV12), brought to the parameters' subsampling. Grey in with
// colour out the card refused (NVJPEG_STATUS_INVALID_PARAMETER). (Some
// shrinking combinations crashed NVIDIA's encoder with a misaligned address;
// here they are encoded.)
static nvjpegStatus_t encode_yuv(EncoderState& st, const EncoderParams& prm, const nvjpegImage_t* src,
                                 nvjpegChromaSubsampling_t in_css, bool nv12, int width, int height) {
  if (static_cast<int>(in_css) < 0 || static_cast<int>(in_css) > 6) return NVJPEG_STATUS_INVALID_PARAMETER;
  const bool gray_in = in_css == NVJPEG_CSS_GRAY;
  if (gray_in && prm.css != NVJPEG_CSS_GRAY) return NVJPEG_STATUS_INVALID_PARAMETER;
  int hs, vs;
  sampling_of(nv12 ? NVJPEG_CSS_420 : in_css, &hs, &vs);
  const int cw = (width + hs - 1) / hs, ch = (height + vs - 1) / vs;
  std::vector<Plane> ycc(gray_in ? 1 : 3);
  const auto read = [&](int idx, int w, int hh, std::vector<uint8_t>* out) -> bool {
    if (!src->channel[idx] || src->pitch[idx] < static_cast<size_t>(w)) return false;
    out->assign(static_cast<size_t>(w) * hh, 0);
    return cudaMemcpy2D(out->data(), w, src->channel[idx], src->pitch[idx], w, hh, cudaMemcpyDeviceToHost) == cudaSuccess;
  };
  std::vector<uint8_t> buf;
  if (!read(0, width, height, &buf)) return NVJPEG_STATUS_INVALID_PARAMETER;
  ycc[0] = Plane{width, height, std::vector<float>(buf.begin(), buf.end())};
  if (!gray_in) {
    if (nv12) {
      if (!read(1, cw * 2, ch, &buf)) return NVJPEG_STATUS_INVALID_PARAMETER;
      for (int c = 1; c < 3; ++c) {
        ycc[c] = Plane{cw, ch, std::vector<float>(static_cast<size_t>(cw) * ch)};
        for (size_t i = 0; i < ycc[c].s.size(); ++i) ycc[c].s[i] = buf[i * 2 + (c - 1)];
      }
    } else {
      for (int c = 1; c < 3; ++c) {
        if (!read(c, cw, ch, &buf)) return NVJPEG_STATUS_INVALID_PARAMETER;
        ycc[c] = Plane{cw, ch, std::vector<float>(buf.begin(), buf.end())};
      }
    }
  }
  st.bitstream = encode_planes(prm, to_target(prm, std::move(ycc), width, height), width, height);
  st.on_device = true;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncodeYUV(nvjpegHandle_t h, nvjpegEncoderState_t st, const nvjpegEncoderParams_t pp,
                                           const nvjpegImage_t* src, nvjpegChromaSubsampling_t css, int width,
                                           int height, cudaStream_t stream) {
  if (!known(h, Kind::Handle) || !known(st, Kind::EncoderState) || !known(pp, Kind::EncoderParams) || !src)
    return NVJPEG_STATUS_INVALID_PARAMETER;
  if (width <= 0 || height <= 0) return NVJPEG_STATUS_INVALID_PARAMETER;
  cudaStreamSynchronize(stream);
  return encode_yuv(*reinterpret_cast<EncoderState*>(st), *reinterpret_cast<const EncoderParams*>(pp), src, css,
                    false, width, height);
}
// The general form (CUDA 12.x): YUV planes of `in_css` (NVJPEG_INPUT_YUV, 1,
// which CUDA 12.0's header lacks), NV12, or the RGB
// family.
VGPU_EXPORT nvjpegStatus_t nvjpegEncode(nvjpegHandle_t h, nvjpegEncoderState_t st, const nvjpegEncoderParams_t pp,
                                        const nvjpegImage_t* src, nvjpegChromaSubsampling_t in_css,
                                        nvjpegInputFormat_t fmt, int width, int height, cudaStream_t stream) {
  if (static_cast<int>(fmt) == 1) return nvjpegEncodeYUV(h, st, pp, src, in_css, width, height, stream);
  if (static_cast<int>(fmt) == 8) {   // NVJPEG_INPUT_NV12
    if (!known(h, Kind::Handle) || !known(st, Kind::EncoderState) || !known(pp, Kind::EncoderParams) || !src ||
        width <= 0 || height <= 0)
      return NVJPEG_STATUS_INVALID_PARAMETER;
    cudaStreamSynchronize(stream);
    return encode_yuv(*reinterpret_cast<EncoderState*>(st), *reinterpret_cast<const EncoderParams*>(pp), src,
                      NVJPEG_CSS_420, true, width, height);
  }
  return nvjpegEncodeImage(h, st, pp, src, fmt, width, height, stream);
}

VGPU_EXPORT nvjpegStatus_t nvjpegEncodeRetrieveBitstream(nvjpegHandle_t h, nvjpegEncoderState_t st, unsigned char* data,
                                                         size_t* length, cudaStream_t) {
  if (!known(h, Kind::Handle) || !known(st, Kind::EncoderState) || !length) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& state = *reinterpret_cast<EncoderState*>(st);
  // The documented two-call protocol: a null buffer asks for the size.
  if (!data) {
    *length = state.bitstream.size();
    return NVJPEG_STATUS_SUCCESS;
  }
  if (*length < state.bitstream.size()) return NVJPEG_STATUS_INVALID_PARAMETER;
  std::memcpy(data, state.bitstream.data(), state.bitstream.size());
  *length = state.bitstream.size();
  state.on_device = false;
  return NVJPEG_STATUS_SUCCESS;
}
// The same, into device memory (torchvision's encode_jpeg retrieves this way).
VGPU_EXPORT nvjpegStatus_t nvjpegEncodeRetrieveBitstreamDevice(nvjpegHandle_t h, nvjpegEncoderState_t st,
                                                               unsigned char* data, size_t* length, cudaStream_t stream) {
  if (!known(h, Kind::Handle) || !known(st, Kind::EncoderState) || !length) return NVJPEG_STATUS_INVALID_PARAMETER;
  const auto& state = *reinterpret_cast<EncoderState*>(st);
  if (!data) {
    *length = state.bitstream.size();
    return NVJPEG_STATUS_SUCCESS;
  }
  if (*length < state.bitstream.size()) return NVJPEG_STATUS_INVALID_PARAMETER;
  cudaStreamSynchronize(stream);
  if (state.on_device &&
      cudaMemcpy(data, state.bitstream.data(), state.bitstream.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    return NVJPEG_STATUS_EXECUTION_FAILED;
  *length = state.bitstream.size();
  return NVJPEG_STATUS_SUCCESS;
}

// The quantisation tables of a parsed image become the encoder's (its first
// two, luma and chroma), so a re-encode keeps the original's quantisation.
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsCopyQuantizationTables(nvjpegEncoderParams_t p, nvjpegJpegStream_t js,
                                                                     cudaStream_t) {
  const Stream* s = parsed(js);
  if (!known(p, Kind::EncoderParams) || !s) return NVJPEG_STATUS_INVALID_PARAMETER;
  auto& prm = *reinterpret_cast<EncoderParams*>(p);
  const Image& im = s->header;
  if (im.comps.empty()) return NVJPEG_STATUS_INVALID_PARAMETER;
  const int luma = im.comps[0].tq, chroma = im.comps.size() > 1 ? im.comps[1].tq : luma;
  if (!im.quant_set[luma] || !im.quant_set[chroma]) return NVJPEG_STATUS_INVALID_PARAMETER;
  std::memcpy(prm.quant[0], im.quant[luma], sizeof prm.quant[0]);
  std::memcpy(prm.quant[1], im.quant[chroma], sizeof prm.quant[1]);
  prm.custom_quant = true;
  return NVJPEG_STATUS_SUCCESS;
}
// Metadata (JFIF, EXIF) is not carried over: the encoder writes its own JFIF
// header. Huffman tables are the encoder's own, standard or optimised.
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsCopyMetadata(nvjpegEncoderState_t st, nvjpegEncoderParams_t p,
                                                           nvjpegJpegStream_t js, cudaStream_t) {
  if (!known(st, Kind::EncoderState) || !known(p, Kind::EncoderParams) || !parsed(js))
    return NVJPEG_STATUS_INVALID_PARAMETER;
  return NVJPEG_STATUS_SUCCESS;
}
VGPU_EXPORT nvjpegStatus_t nvjpegEncoderParamsCopyHuffmanTables(nvjpegEncoderState_t st, nvjpegEncoderParams_t p,
                                                                nvjpegJpegStream_t js, cudaStream_t) {
  if (!known(st, Kind::EncoderState) || !known(p, Kind::EncoderParams) || !parsed(js))
    return NVJPEG_STATUS_INVALID_PARAMETER;
  return NVJPEG_STATUS_SUCCESS;
}
