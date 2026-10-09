// See nvenc_h264_enc.hpp.
#include "nvenc_h264_enc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "h264_decode.hpp"
#include "nvenc_h264.hpp"

namespace vgpu_nvenc {
namespace {

#include "h264_tables.inc"

inline int clip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }
inline uint8_t clip1(int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }
inline int median3(int a, int b, int c) { return std::max(std::min(a, b), std::min(std::max(a, b), c)); }
inline int floor_log2(uint32_t x) {
  int n = 0;
  while (x >>= 1) ++n;
  return n;
}

// Position of 4x4 block luma4x4BlkIdx inside a macroblock, in 4x4 units.
inline int blk_x(int idx) { return ((idx >> 2) & 1) * 2 + (idx & 1); }
inline int blk_y(int idx) { return ((idx >> 3) & 1) * 2 + ((idx >> 1) & 1); }

// Forward quantisation factors MF and the dequantisation scale v of 8.5.9 (flat scaling lists), by qp % 6 and
// the position class: 0 both coordinates even, 1 both odd, 2 otherwise.
constexpr int kMF[6][3] = {{13107, 5243, 8066}, {11916, 4660, 7490}, {10082, 4194, 6554}, {9362, 3647, 5825}, {8192, 3355, 5243}, {7282, 2893, 4559}};
constexpr int kScale[6][3] = {{10, 16, 13}, {11, 18, 14}, {13, 20, 16}, {14, 23, 18}, {16, 25, 20}, {18, 29, 23}};
inline int pos_class(int k) {
  const int i = k >> 2, j = k & 3;
  if ((i & 1) == 0 && (j & 1) == 0) return 0;
  if ((i & 1) == 1 && (j & 1) == 1) return 1;
  return 2;
}
constexpr int kQpcTable[22] = {29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39};
inline int qpc_of(int qp) { return qp < 30 ? qp : kQpcTable[qp - 30]; }

// ---- bit counting ---------------------------------------------------------------------------------------------

struct BitCounter {
  size_t n = 0;
  void put(uint32_t, int bits) { n += static_cast<size_t>(bits); }
  void bit(int) { ++n; }
  void ue(uint32_t v) { n += static_cast<size_t>(2 * floor_log2(v + 1) + 1); }
  void se(int32_t v) { ue(v > 0 ? static_cast<uint32_t>(2 * v - 1) : static_cast<uint32_t>(-2 * static_cast<int64_t>(v))); }
};

// ---- CAVLC residual blocks (9.2), written ---------------------------------------------------------------------

template <class W>
void write_level(W& w, int level, int sl, bool lower_by_two) {
  int code = level > 0 ? 2 * level - 2 : -2 * level - 1;
  if (lower_by_two) code -= 2;
  int prefix, suffix_size = 0, suffix = 0;
  if (sl == 0 && code < 14) {
    prefix = code;
  } else if (sl == 0 && code < 30) {
    prefix = 14;
    suffix_size = 4;
    suffix = code - 14;
  } else if (sl > 0 && code < (15 << sl)) {
    prefix = code >> sl;
    suffix_size = sl;
    suffix = code & ((1 << sl) - 1);
  } else {
    for (prefix = 15;; ++prefix) {
      const int base = (15 << sl) + (sl == 0 ? 15 : 0) + (prefix >= 16 ? (1 << (prefix - 3)) - 4096 : 0);
      suffix_size = prefix - 3;
      if (code - base >= 0 && code - base < (1 << suffix_size)) {
        suffix = code - base;
        break;
      }
    }
  }
  w.put(1, prefix + 1);
  if (suffix_size) w.put(static_cast<uint32_t>(suffix), suffix_size);
}

// `lev` holds the levels in scan order. nC: the coeff_token context (-1: chroma DC). Returns TotalCoeff.
template <class W>
int write_block(W& w, const int* lev, int max_coef, int nC) {
  int pos[16], val[16], tc = 0;
  for (int i = max_coef - 1; i >= 0; --i)
    if (lev[i]) {
      pos[tc] = i;
      val[tc] = lev[i];
      ++tc;
    }
  int t1 = 0;
  while (t1 < tc && t1 < 3 && std::abs(val[t1]) == 1) ++t1;
  if (nC >= 8) {
    if (tc == 0) w.put(3, 6);
    else w.put(static_cast<uint32_t>(((tc - 1) << 2) | t1), 6);
  } else {
    const VlcCode& c = nC == -1 ? kCoeffTokenChromaDc[t1][tc] : (nC < 2 ? kCoeffToken0 : (nC < 4 ? kCoeffToken1 : kCoeffToken2))[t1][tc];
    w.put(c.bits, c.len);
  }
  if (tc == 0) return 0;
  for (int k = 0; k < t1; ++k) w.bit(val[k] < 0);
  int sl = (tc > 10 && t1 < 3) ? 1 : 0;
  for (int k = t1; k < tc; ++k) {
    write_level(w, val[k], sl, k == t1 && t1 < 3);
    if (sl == 0) sl = 1;
    if (std::abs(val[k]) > (3 << (sl - 1)) && sl < 6) ++sl;
  }
  const int total_zeros = pos[0] + 1 - tc;
  if (tc < max_coef) {
    const VlcCode& c = max_coef == 4 ? kTotalZerosChromaDc[tc - 1][total_zeros] : kTotalZeros[tc - 1][total_zeros];
    w.put(c.bits, c.len);
  }
  int zeros_left = total_zeros;
  for (int k = 0; k < tc - 1 && zeros_left > 0; ++k) {
    const int run = pos[k] - pos[k + 1] - 1;
    const VlcCode& c = kRunBefore[std::min(zeros_left, 7) - 1][run];
    w.put(c.bits, c.len);
    zeros_left -= run;
  }
  return tc;
}

// ---- transforms ------------------------------------------------------------------------------------------------

// Forward 4x4 core transform (the matrix Cf of the quantisation derivation) of a raster residual block.
void fwd4x4(const int* x, int* out) {
  int t[16];
  for (int i = 0; i < 4; ++i) {
    const int s03 = x[4 * i] + x[4 * i + 3], d03 = x[4 * i] - x[4 * i + 3], s12 = x[4 * i + 1] + x[4 * i + 2], d12 = x[4 * i + 1] - x[4 * i + 2];
    t[4 * i] = s03 + s12;
    t[4 * i + 1] = 2 * d03 + d12;
    t[4 * i + 2] = s03 - s12;
    t[4 * i + 3] = d03 - 2 * d12;
  }
  for (int j = 0; j < 4; ++j) {
    const int s03 = t[j] + t[12 + j], d03 = t[j] - t[12 + j], s12 = t[4 + j] + t[8 + j], d12 = t[4 + j] - t[8 + j];
    out[j] = s03 + s12;
    out[4 + j] = 2 * d03 + d12;
    out[8 + j] = s03 - s12;
    out[12 + j] = d03 - 2 * d12;
  }
}

// Inverse transform of 8.5.12.2 of dequantised coefficients d (raster), added to the prediction in dst.
void idct4_add(const int* d, uint8_t* dst, int stride) {
  int t[16];
  for (int i = 0; i < 4; ++i) {
    const int* r = d + 4 * i;
    const int e0 = r[0] + r[2], e1 = r[0] - r[2], e2 = (r[1] >> 1) - r[3], e3 = r[1] + (r[3] >> 1);
    t[4 * i + 0] = e0 + e3;
    t[4 * i + 1] = e1 + e2;
    t[4 * i + 2] = e1 - e2;
    t[4 * i + 3] = e0 - e3;
  }
  for (int j = 0; j < 4; ++j) {
    const int g0 = t[j] + t[8 + j], g1 = t[j] - t[8 + j], g2 = (t[4 + j] >> 1) - t[12 + j], g3 = t[4 + j] + (t[12 + j] >> 1);
    const int h[4] = {g0 + g3, g1 + g2, g1 - g2, g0 - g3};
    for (int i = 0; i < 4; ++i) {
      uint8_t* p = dst + i * stride + j;
      *p = clip1(*p + ((h[i] + 32) >> 6));
    }
  }
}

// Quantises the coefficients of a 4x4 block into scan-order levels. `first` is 1 when the DC coefficient is
// coded elsewhere (Intra16x16, chroma): then lev[i] is scan position i + 1. Returns the number of non-zero
// levels. `dq` receives the dequantised coefficients (raster; DC left as given in dq[0]).
int quant_block(const int* w, int qp, bool intra, int first, int* lev, int* dq) {
  const int qbits = 15 + qp / 6, m = qp % 6;
  const int f = (1 << qbits) / (intra ? 3 : 6);
  int nz = 0;
  for (int i = first; i < 16; ++i) {
    const int k = kZigzag4x4[i];
    const int cls = pos_class(k);
    const int a = std::abs(w[k]);
    const int q = (a * kMF[m][cls] + f) >> qbits;
    const int level = w[k] < 0 ? -q : q;
    lev[i - first] = level;
    nz += level != 0;
    dq[k] = (level * kScale[m][cls]) << (qp / 6);
  }
  return nz;
}

// 4x4 Hadamard-transformed SAD of a raster difference block.
int satd4(const int* d) {
  int t[16], s = 0;
  for (int i = 0; i < 4; ++i) {
    const int a = d[4 * i], b = d[4 * i + 1], c = d[4 * i + 2], e = d[4 * i + 3];
    t[4 * i] = a + b + c + e;
    t[4 * i + 1] = a + b - c - e;
    t[4 * i + 2] = a - b - c + e;
    t[4 * i + 3] = a - b + c - e;
  }
  for (int j = 0; j < 4; ++j) {
    const int a = t[j], b = t[4 + j], c = t[8 + j], e = t[12 + j];
    s += std::abs(a + b + c + e) + std::abs(a + b - c - e) + std::abs(a - b - c + e) + std::abs(a - b + c - e);
  }
  return s / 2;
}

// ---- intra prediction -----------------------------------------------------------------------------------------

// Intra4x4 prediction (8.3.1.2). pt(x) is p[x, -1] for x in -1..7 (-1: the corner), pl(y) is p[-1, y].
template <class PT, class PL>
void predict4x4(int mode, bool top_av, bool left_av, PT pt, PL pl, uint8_t* dst, int stride) {
  auto put = [&](int x, int y, int v) { dst[y * stride + x] = static_cast<uint8_t>(v); };
  constexpr int N = 4;
  switch (mode) {
    case 0:
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) put(x, y, pt(x));
      break;
    case 1:
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) put(x, y, pl(y));
      break;
    case 2: {
      int s = 0, v;
      if (top_av && left_av) {
        for (int i = 0; i < N; ++i) s += pt(i) + pl(i);
        v = (s + 4) >> 3;
      } else if (left_av) {
        for (int i = 0; i < N; ++i) s += pl(i);
        v = (s + 2) >> 2;
      } else if (top_av) {
        for (int i = 0; i < N; ++i) s += pt(i);
        v = (s + 2) >> 2;
      } else {
        v = 128;
      }
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) put(x, y, v);
      break;
    }
    case 3:
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          if (x == N - 1 && y == N - 1) put(x, y, (pt(6) + 3 * pt(7) + 2) >> 2);
          else put(x, y, (pt(x + y) + 2 * pt(x + y + 1) + pt(x + y + 2) + 2) >> 2);
        }
      break;
    case 4:
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          if (x > y) put(x, y, (pt(x - y - 2) + 2 * pt(x - y - 1) + pt(x - y) + 2) >> 2);
          else if (x < y) put(x, y, (pl(y - x - 2) + 2 * pl(y - x - 1) + pl(y - x) + 2) >> 2);
          else put(x, y, (pt(0) + 2 * pt(-1) + pl(0) + 2) >> 2);
        }
      break;
    case 5:
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const int z = 2 * x - y;
          if (z >= 0 && (z & 1) == 0) put(x, y, (pt(x - (y >> 1) - 1) + pt(x - (y >> 1)) + 1) >> 1);
          else if (z >= 0) put(x, y, (pt(x - (y >> 1) - 2) + 2 * pt(x - (y >> 1) - 1) + pt(x - (y >> 1)) + 2) >> 2);
          else if (z == -1) put(x, y, (pl(0) + 2 * pl(-1) + pt(0) + 2) >> 2);
          else put(x, y, (pl(y - 2 * x - 1) + 2 * pl(y - 2 * x - 2) + pl(y - 2 * x - 3) + 2) >> 2);
        }
      break;
    case 6:
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const int z = 2 * y - x;
          if (z >= 0 && (z & 1) == 0) put(x, y, (pl(y - (x >> 1) - 1) + pl(y - (x >> 1)) + 1) >> 1);
          else if (z >= 0) put(x, y, (pl(y - (x >> 1) - 2) + 2 * pl(y - (x >> 1) - 1) + pl(y - (x >> 1)) + 2) >> 2);
          else if (z == -1) put(x, y, (pl(0) + 2 * pl(-1) + pt(0) + 2) >> 2);
          else put(x, y, (pt(x - 2 * y - 1) + 2 * pt(x - 2 * y - 2) + pt(x - 2 * y - 3) + 2) >> 2);
        }
      break;
    case 7:
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          if ((y & 1) == 0) put(x, y, (pt(x + (y >> 1)) + pt(x + (y >> 1) + 1) + 1) >> 1);
          else put(x, y, (pt(x + (y >> 1)) + 2 * pt(x + (y >> 1) + 1) + pt(x + (y >> 1) + 2) + 2) >> 2);
        }
      break;
    default:
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const int z = x + 2 * y;
          if (z > 2 * N - 3) put(x, y, pl(N - 1));
          else if (z == 2 * N - 3) put(x, y, (pl(N - 2) + 3 * pl(N - 1) + 2) >> 2);
          else if ((z & 1) == 0) put(x, y, (pl(y + (x >> 1)) + pl(y + (x >> 1) + 1) + 1) >> 1);
          else put(x, y, (pl(y + (x >> 1)) + 2 * pl(y + (x >> 1) + 1) + pl(y + (x >> 1) + 2) + 2) >> 2);
        }
      break;
  }
}

// ---- one macroblock's worth of syntax elements -------------------------------------------------------------

enum MbType { kMbI4x4 = 0, kMbI16x16 = 1, kMbInter = 2, kMbSkip = 3 };

struct MbCode {
  int type = kMbI4x4;
  int i16 = 0, chroma = 0;      // Intra16x16PredMode, intra_chroma_pred_mode
  int i4[16] = {};              // Intra4x4PredMode by luma4x4BlkIdx
  int cbp_l = 0, cbp_c = 0;
  int mvx = 0, mvy = 0;         // quarter samples
  int dc[16] = {};              // Intra16x16 DC levels, scan order
  int lv[16][16] = {};          // luma levels by blkIdx, scan order (Intra16x16: AC, scan positions 1..15 at 0..14)
  int cdc[2][4] = {};           // chroma DC levels
  int cac[2][4][15] = {};       // chroma AC levels (scan positions 1..15 at 0..14)
};

// Padding around the reference's interpolation planes, and the farthest the motion search goes.
constexpr int kPad = 48;
constexpr int kSearch = 24;

struct RefPlanes {
  int w = 0, h = 0, stride = 0;           // luma size and the padded plane stride
  std::vector<uint8_t> g, hh, vv, jj;     // integer samples, horizontal half, vertical half and centre half positions
  const uint8_t* at(const std::vector<uint8_t>& p, int x, int y) const {
    x = clip3(-kPad, w + kPad - 1, x);
    y = clip3(-kPad, h + kPad - 1, y);
    return &p[static_cast<size_t>(y + kPad) * stride + (x + kPad)];
  }
};

}  // namespace

// ---- the encoder ---------------------------------------------------------------------------------------------

struct H264Encoder::Impl {
  int w, h, fps_num, fps_den, profile;
  bool deblock;
  int mw, mh, ys, cs, cw, ch;     // macroblocks, coded luma/chroma strides, visible chroma size
  H264Stream stream;

  std::vector<uint8_t> sy, su, sv;   // the picture, padded to whole macroblocks
  std::vector<uint8_t> ry, ru, rv;   // reconstruction before the loop filter
  std::unique_ptr<vgpu_h264::Frame> ref;   // the decoded picture the next P picture predicts from
  RefPlanes rp;
  bool have_ref = false;
  int frame_num = 0, ref_frame_num = 0, pic_count = 0, uid = 0;
  bool decode_failed_reported = false;

  // per picture
  int qp = 26, qpc = 26;
  double lambda = 1, lam_sad = 1;
  PicType type = PicType::kIdr;
  std::vector<uint8_t> nz_y, nz_u, nz_v;   // TotalCoeff of every 4x4 block (chroma: of the AC blocks)
  std::vector<uint8_t> i4m;                // Intra4x4PredMode of every luma 4x4 block (2 where not Intra4x4)
  std::vector<uint8_t> mb_inter;           // 1: the macroblock is inter predicted
  std::vector<int> mvx, mvy;
  int skip_run = 0;
  EncStats st;

  Impl(int width, int height, int fn, int fd, int prof, bool dbk)
      : w(width), h(height), fps_num(fn), fps_den(fd), profile(prof), deblock(dbk) {
    mw = (w + 15) / 16;
    mh = (h + 15) / 16;
    ys = mw * 16;
    cs = mw * 8;
    cw = (w + 1) / 2;
    ch = (h + 1) / 2;
    stream.width = w;
    stream.height = h;
    stream.fps_num = fn;
    stream.fps_den = fd;
    stream.profile_idc = prof;
    ry.assign(static_cast<size_t>(ys) * mh * 16, 0);
    ru.assign(static_cast<size_t>(cs) * mh * 8, 128);
    rv = ru;
  }

  // ---- neighbour bookkeeping ----------------------------------------------------------------------------------

  int nz_stride() const { return mw * 4; }
  int nzc_stride() const { return mw * 2; }
  uint8_t& NZY(int gx, int gy) { return nz_y[static_cast<size_t>(gy) * nz_stride() + gx]; }
  uint8_t& NZC(int c, int gx, int gy) { return (c ? nz_v : nz_u)[static_cast<size_t>(gy) * nzc_stride() + gx]; }
  uint8_t& I4M(int gx, int gy) { return i4m[static_cast<size_t>(gy) * nz_stride() + gx]; }

  // nC for the coeff_token of a block at global 4x4 coordinates (gx, gy) of a plane with the given count array.
  static int nc_of(bool avail_a, int na, bool avail_b, int nb) {
    if (avail_a && avail_b) return (na + nb + 1) >> 1;
    if (avail_a) return na;
    if (avail_b) return nb;
    return 0;
  }
  int nc_luma(int gx, int gy) { return nc_of(gx > 0, gx > 0 ? NZY(gx - 1, gy) : 0, gy > 0, gy > 0 ? NZY(gx, gy - 1) : 0); }
  int nc_chroma(int c, int gx, int gy) { return nc_of(gx > 0, gx > 0 ? NZC(c, gx - 1, gy) : 0, gy > 0, gy > 0 ? NZC(c, gx, gy - 1) : 0); }

  // The predicted Intra4x4PredMode of the block at global (gx, gy).
  int pred_i4(int gx, int gy) {
    if (gx == 0 || gy == 0) return 2;
    return std::min<int>(I4M(gx - 1, gy), I4M(gx, gy - 1));
  }

  // The motion vector prediction of the 16x16 macroblock (8.4.1.3) and the P_Skip vector (8.4.1.1).
  struct Nb {
    bool avail = false;
    int ref = -1, x = 0, y = 0;
  };
  Nb nb_mv(int mx, int my) const {
    Nb n;
    if (mx < 0 || my < 0 || mx >= mw) return n;
    n.avail = true;
    const size_t a = static_cast<size_t>(my) * mw + mx;
    if (mb_inter[a]) {
      n.ref = 0;
      n.x = mvx[a];
      n.y = mvy[a];
    }
    return n;
  }
  void mv_pred(int mx, int my, int* px, int* py, bool* skip_zero) {
    const Nb a = nb_mv(mx - 1, my), b = nb_mv(mx, my - 1);
    Nb c = nb_mv(mx + 1, my - 1);
    if (!c.avail) c = nb_mv(mx - 1, my - 1);
    *skip_zero = !a.avail || !b.avail || (a.ref == 0 && a.x == 0 && a.y == 0) || (b.ref == 0 && b.x == 0 && b.y == 0);
    Nb B = b, C = c;
    if (!b.avail && !c.avail && a.avail) {
      B = a;
      C = a;
    }
    const int matches = (a.ref == 0) + (B.ref == 0) + (C.ref == 0);
    if (matches == 1) {
      const Nb& m = a.ref == 0 ? a : (B.ref == 0 ? B : C);
      *px = m.x;
      *py = m.y;
    } else {
      *px = median3(a.x, B.x, C.x);
      *py = median3(a.y, B.y, C.y);
    }
  }

  // ---- source and reconstruction access ------------------------------------------------------------------------

  const uint8_t* src_y(int mx, int my) const { return &sy[static_cast<size_t>(my) * 16 * ys + mx * 16]; }
  uint8_t* rec_y(int mx, int my) { return &ry[static_cast<size_t>(my) * 16 * ys + mx * 16]; }
  const uint8_t* src_c(int c, int mx, int my) const { return &(c ? sv : su)[static_cast<size_t>(my) * 8 * cs + mx * 8]; }
  uint8_t* rec_c(int c, int mx, int my) { return &(c ? rv : ru)[static_cast<size_t>(my) * 8 * cs + mx * 8]; }

  void load_source(const EncPicture& in) {
    sy.assign(static_cast<size_t>(ys) * mh * 16, 0);
    su.assign(static_cast<size_t>(cs) * mh * 8, 0);
    sv = su;
    for (int y = 0; y < mh * 16; ++y)
      for (int x = 0; x < ys; ++x) sy[static_cast<size_t>(y) * ys + x] = in.y[static_cast<size_t>(std::min(y, h - 1)) * w + std::min(x, w - 1)];
    for (int y = 0; y < mh * 8; ++y)
      for (int x = 0; x < cs; ++x) {
        const size_t i = static_cast<size_t>(std::min(y, ch - 1)) * cw + std::min(x, cw - 1);
        su[static_cast<size_t>(y) * cs + x] = in.u[i];
        sv[static_cast<size_t>(y) * cs + x] = in.v[i];
      }
  }

  // ---- reference interpolation planes (8.4.2.2.1) -------------------------------------------------------------

  void build_ref_planes() {
    const vgpu_h264::Frame& f = *ref;
    const int W = ys, H = mh * 16;
    rp.w = W;
    rp.h = H;
    rp.stride = W + 2 * kPad;
    const size_t n = static_cast<size_t>(rp.stride) * (H + 2 * kPad);
    rp.g.assign(n, 0);
    for (int y = -kPad; y < H + kPad; ++y)
      for (int x = -kPad; x < W + kPad; ++x)
        rp.g[static_cast<size_t>(y + kPad) * rp.stride + (x + kPad)] = f.y[static_cast<size_t>(clip3(0, H - 1, y)) * f.stride_y + clip3(0, W - 1, x)];
    const int PW = rp.stride, PH = H + 2 * kPad;
    auto G = [&](int x, int y) { return static_cast<int>(rp.g[static_cast<size_t>(clip3(0, PH - 1, y)) * PW + clip3(0, PW - 1, x)]); };
    std::vector<int16_t> b1(n);   // horizontal intermediate values
    rp.hh.assign(n, 0);
    rp.vv.assign(n, 0);
    rp.jj.assign(n, 0);
    for (int y = 0; y < PH; ++y)
      for (int x = 0; x < PW; ++x) {
        const int v = G(x - 2, y) - 5 * G(x - 1, y) + 20 * G(x, y) + 20 * G(x + 1, y) - 5 * G(x + 2, y) + G(x + 3, y);
        b1[static_cast<size_t>(y) * PW + x] = static_cast<int16_t>(v);
        rp.hh[static_cast<size_t>(y) * PW + x] = clip1((v + 16) >> 5);
        const int u = G(x, y - 2) - 5 * G(x, y - 1) + 20 * G(x, y) + 20 * G(x, y + 1) - 5 * G(x, y + 2) + G(x, y + 3);
        rp.vv[static_cast<size_t>(y) * PW + x] = clip1((u + 16) >> 5);
      }
    auto B1 = [&](int x, int y) { return static_cast<int>(b1[static_cast<size_t>(clip3(0, PH - 1, y)) * PW + x]); };
    for (int y = 0; y < PH; ++y)
      for (int x = 0; x < PW; ++x) {
        const int j1 = B1(x, y - 2) - 5 * B1(x, y - 1) + 20 * B1(x, y) + 20 * B1(x, y + 1) - 5 * B1(x, y + 2) + B1(x, y + 3);
        rp.jj[static_cast<size_t>(y) * PW + x] = clip1((j1 + 512) >> 10);
      }
  }

  // The 16x16 luma prediction at quarter-sample vector (mvx, mvy) for macroblock (mx, my).
  void pred_luma(int mx, int my, int vx, int vy, uint8_t* out) const {
    const int fx = vx & 3, fy = vy & 3;
    const int X0 = mx * 16 + (vx >> 2), Y0 = my * 16 + (vy >> 2);
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) {
        const int X = X0 + x, Y = Y0 + y;
        auto g = [&](int dx, int dy) { return static_cast<int>(*rp.at(rp.g, X + dx, Y + dy)); };
        auto b = [&](int dy) { return static_cast<int>(*rp.at(rp.hh, X, Y + dy)); };
        auto hv = [&](int dx) { return static_cast<int>(*rp.at(rp.vv, X + dx, Y)); };
        const int j = *rp.at(rp.jj, X, Y);
        int v;
        switch (fy * 4 + fx) {
          case 0: v = g(0, 0); break;
          case 1: v = (g(0, 0) + b(0) + 1) >> 1; break;
          case 2: v = b(0); break;
          case 3: v = (b(0) + g(1, 0) + 1) >> 1; break;
          case 4: v = (g(0, 0) + hv(0) + 1) >> 1; break;
          case 5: v = (b(0) + hv(0) + 1) >> 1; break;
          case 6: v = (b(0) + j + 1) >> 1; break;
          case 7: v = (b(0) + hv(1) + 1) >> 1; break;
          case 8: v = hv(0); break;
          case 9: v = (hv(0) + j + 1) >> 1; break;
          case 10: v = j; break;
          case 11: v = (j + hv(1) + 1) >> 1; break;
          case 12: v = (hv(0) + g(0, 1) + 1) >> 1; break;
          case 13: v = (hv(0) + b(1) + 1) >> 1; break;
          case 14: v = (j + b(1) + 1) >> 1; break;
          default: v = (hv(1) + b(1) + 1) >> 1; break;
        }
        out[y * 16 + x] = static_cast<uint8_t>(v);
      }
  }

  // The 8x8 chroma predictions (8.4.2.2.2) of both planes at the luma vector.
  void pred_chroma(int mx, int my, int vx, int vy, uint8_t* out_u, uint8_t* out_v) const {
    const vgpu_h264::Frame& f = *ref;
    const int fx = vx & 7, fy = vy & 7;
    const int X0 = mx * 8 + (vx >> 3), Y0 = my * 8 + (vy >> 3);
    const int W = mw * 8, H = mh * 8;
    for (int c = 0; c < 2; ++c) {
      const std::vector<uint8_t>& p = c ? f.v : f.u;
      uint8_t* out = c ? out_v : out_u;
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
          auto at = [&](int dx, int dy) { return static_cast<int>(p[static_cast<size_t>(clip3(0, H - 1, Y0 + y + dy)) * f.stride_c + clip3(0, W - 1, X0 + x + dx)]); };
          out[y * 8 + x] = static_cast<uint8_t>(((8 - fx) * (8 - fy) * at(0, 0) + fx * (8 - fy) * at(1, 0) + (8 - fx) * fy * at(0, 1) + fx * fy * at(1, 1) + 32) >> 6);
        }
    }
  }

  // ---- residual coding of blocks -----------------------------------------------------------------------------

  static int sse16(const uint8_t* a, int sa, const uint8_t* b, int sb, int n) {
    int s = 0;
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const int d = a[y * sa + x] - b[y * sb + x];
        s += d * d;
      }
    return s;
  }

  // Codes the luma of an Intra16x16 macroblock with the given prediction mode; the reconstruction goes to
  // `rec` (16x16, stride 16). Returns false if the mode needs unavailable samples.
  void predict16(int mx, int my, int mode, uint8_t* pred) {
    const uint8_t* r = rec_y(mx, my);
    const bool top = my > 0, left = mx > 0;
    int topv[17], leftv[16], corner = 128;
    for (int i = 0; i < 16; ++i) {
      topv[i] = top ? r[-ys + i] : 128;
      leftv[i] = left ? r[i * ys - 1] : 128;
    }
    if (top && left) corner = r[-ys - 1];
    switch (mode) {
      case 0:
        for (int y = 0; y < 16; ++y)
          for (int x = 0; x < 16; ++x) pred[y * 16 + x] = static_cast<uint8_t>(topv[x]);
        break;
      case 1:
        for (int y = 0; y < 16; ++y)
          for (int x = 0; x < 16; ++x) pred[y * 16 + x] = static_cast<uint8_t>(leftv[y]);
        break;
      case 2: {
        int s = 0, v;
        if (top && left) {
          for (int i = 0; i < 16; ++i) s += topv[i] + leftv[i];
          v = (s + 16) >> 5;
        } else if (left) {
          for (int i = 0; i < 16; ++i) s += leftv[i];
          v = (s + 8) >> 4;
        } else if (top) {
          for (int i = 0; i < 16; ++i) s += topv[i];
          v = (s + 8) >> 4;
        } else {
          v = 128;
        }
        std::memset(pred, v, 256);
        break;
      }
      default: {
        int H = 0, V = 0;
        for (int i = 0; i < 8; ++i) {
          H += (i + 1) * (topv[8 + i] - (6 - i >= 0 ? topv[6 - i] : corner));
          V += (i + 1) * (leftv[8 + i] - (6 - i >= 0 ? leftv[6 - i] : corner));
        }
        const int a = 16 * (leftv[15] + topv[15]);
        const int b = (5 * H + 32) >> 6, c = (5 * V + 32) >> 6;
        for (int y = 0; y < 16; ++y)
          for (int x = 0; x < 16; ++x) pred[y * 16 + x] = clip1((a + b * (x - 7) + c * (y - 7) + 16) >> 5);
      }
    }
  }

  bool i16_mode_ok(int mx, int my, int mode) const {
    switch (mode) {
      case 0: return my > 0;
      case 1: return mx > 0;
      case 2: return true;
      default: return mx > 0 && my > 0;
    }
  }

  // Quantises and reconstructs an Intra16x16 macroblock's luma from `pred`, into mc (levels) and rec (16x16, stride ys at
  // the macroblock's place in the reconstruction).
  void code_i16(int mx, int my, const uint8_t* pred, MbCode& mc) {
    const uint8_t* s = src_y(mx, my);
    int W[16][16], dcw[16];   // transform coefficients per block (raster block order), the DCs
    for (int blk = 0; blk < 16; ++blk) {
      const int bx = blk_x(blk), by = blk_y(blk);
      int r[16];
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) r[y * 4 + x] = s[(by * 4 + y) * ys + bx * 4 + x] - pred[(by * 4 + y) * 16 + bx * 4 + x];
      fwd4x4(r, W[blk]);
      dcw[by * 4 + bx] = W[blk][0];
    }
    // DC: Hadamard transform, quantisation (8.5.10 inverted)
    int t[16], had[16];
    for (int i = 0; i < 4; ++i) {
      const int a = dcw[4 * i], b = dcw[4 * i + 1], c = dcw[4 * i + 2], d = dcw[4 * i + 3];
      t[4 * i] = a + b + c + d;
      t[4 * i + 1] = a + b - c - d;
      t[4 * i + 2] = a - b - c + d;
      t[4 * i + 3] = a - b + c - d;
    }
    for (int j = 0; j < 4; ++j) {
      const int a = t[j], b = t[4 + j], c = t[8 + j], d = t[12 + j];
      had[j] = a + b + c + d;
      had[4 + j] = a + b - c - d;
      had[8 + j] = a - b - c + d;
      had[12 + j] = a - b + c - d;
    }
    const int m = qp % 6, qbits = 15 + qp / 6, f = (1 << qbits) / 3;
    int dcl[16];   // levels, matrix order
    bool any_dc = false;
    for (int k = 0; k < 16; ++k) {
      const int a = std::abs(had[k]) >> 1;
      const int q = (a * kMF[m][0] + 2 * f) >> (qbits + 1);
      dcl[k] = had[k] < 0 ? -q : q;
      any_dc |= q != 0;
    }
    for (int i = 0; i < 16; ++i) mc.dc[i] = dcl[kZigzag4x4[i]];
    // DC reconstruction (8.5.10)
    int ft[16], fdc[16];
    for (int i = 0; i < 4; ++i) {
      const int a = dcl[4 * i], b = dcl[4 * i + 1], c = dcl[4 * i + 2], d = dcl[4 * i + 3];
      ft[4 * i] = a + b + c + d;
      ft[4 * i + 1] = a + b - c - d;
      ft[4 * i + 2] = a - b - c + d;
      ft[4 * i + 3] = a - b + c - d;
    }
    for (int j = 0; j < 4; ++j) {
      const int a = ft[j], b = ft[4 + j], c = ft[8 + j], d = ft[12 + j];
      fdc[j] = a + b + c + d;
      fdc[4 + j] = a + b - c - d;
      fdc[8 + j] = a - b - c + d;
      fdc[12 + j] = a - b + c - d;
    }
    const int ls0 = 16 * kScale[m][0];
    int dcy[16];
    for (int k = 0; k < 16; ++k)
      dcy[k] = qp >= 36 ? (fdc[k] * ls0) << (qp / 6 - 6) : (fdc[k] * ls0 + (1 << (5 - qp / 6))) >> (6 - qp / 6);
    // AC
    bool any_ac = false;
    int dqs[16][16];
    for (int blk = 0; blk < 16; ++blk) {
      int dq[16] = {};
      const int nz = quant_block(W[blk], qp, true, 1, mc.lv[blk], dq);
      any_ac |= nz != 0;
      std::memcpy(dqs[blk], dq, sizeof dq);
    }
    mc.cbp_l = any_ac ? 15 : 0;
    (void)any_dc;
    uint8_t* d = rec_y(mx, my);
    for (int blk = 0; blk < 16; ++blk) {
      const int bx = blk_x(blk), by = blk_y(blk);
      int* dq = dqs[blk];
      if (!any_ac) {
        std::memset(dq, 0, sizeof(int) * 16);
        std::memset(mc.lv[blk], 0, sizeof mc.lv[blk]);
      }
      dq[0] = dcy[by * 4 + bx];
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) d[(by * 4 + y) * ys + bx * 4 + x] = pred[(by * 4 + y) * 16 + bx * 4 + x];
      idct4_add(dq, d + by * 4 * ys + bx * 4, ys);
    }
  }

  // Intra4x4 availability of the neighbours of block blk of the macroblock at (mx, my).
  struct Nb4 {
    bool top = false, left = false, corner = false, tr = false;
    int top_v[9] = {}, left_v[4] = {};   // top_v[0] = corner, top_v[1 + i] = p[i, -1]
  };
  Nb4 gather4(int mx, int my, int blk) {
    Nb4 n;
    const int bx = blk_x(blk), by = blk_y(blk);
    const int X = mx * 16 + bx * 4, Y = my * 16 + by * 4;
    n.top = by > 0 || my > 0;
    n.left = bx > 0 || mx > 0;
    n.corner = (by > 0 || my > 0) && (bx > 0 || mx > 0);
    if (by == 0) n.tr = my > 0 && (bx < 3 || mx + 1 < mw);
    else if (bx == 3) n.tr = false;
    else {
      // inside the macroblock: available when the block holding the samples came earlier in coding order
      int other = -1;
      for (int i = 0; i < 16; ++i)
        if (blk_x(i) == bx + 1 && blk_y(i) == by - 1) other = i;
      n.tr = other >= 0 && other < blk;
    }
    if (n.corner) n.top_v[0] = ry[static_cast<size_t>(Y - 1) * ys + X - 1];
    if (n.top)
      for (int i = 0; i < 4; ++i) n.top_v[1 + i] = ry[static_cast<size_t>(Y - 1) * ys + X + i];
    if (n.top && n.tr)
      for (int i = 4; i < 8; ++i) n.top_v[1 + i] = ry[static_cast<size_t>(Y - 1) * ys + X + i];
    else if (n.top)
      for (int i = 4; i < 8; ++i) n.top_v[1 + i] = n.top_v[4];
    if (n.left)
      for (int i = 0; i < 4; ++i) n.left_v[i] = ry[static_cast<size_t>(Y + i) * ys + X - 1];
    return n;
  }

  bool mode4_ok(const Nb4& n, int mode) const {
    switch (mode) {
      case 0: case 3: case 7: return n.top;
      case 1: case 8: return n.left;
      case 2: return true;
      default: return n.top && n.left && n.corner;
    }
  }

  // Intra4x4: chooses the modes, codes the 16 blocks into the reconstruction. mc.i4, mc.lv and the totals are set.
  void code_i4(int mx, int my, MbCode& mc) {
    const uint8_t* s = src_y(mx, my);
    bool any[4] = {false, false, false, false};
    for (int blk = 0; blk < 16; ++blk) {
      const int bx = blk_x(blk), by = blk_y(blk);
      const int gx = mx * 4 + bx, gy = my * 4 + by;
      const Nb4 n = gather4(mx, my, blk);
      const int predmode = (gx == 0 || gy == 0) ? 2 : std::min<int>(I4M(gx - 1, gy), I4M(gx, gy - 1));
      auto pl = [&](int y) { return y < 0 ? n.top_v[0] : n.left_v[y]; };
      auto ptc = [&](int x) { return x < 0 ? n.top_v[0] : n.top_v[1 + x]; };
      uint8_t pred[16], best_pred[16];
      int best_mode = 2;
      double best_cost = 1e30;
      for (int mode = 0; mode < 9; ++mode) {
        if (!mode4_ok(n, mode)) continue;
        predict4x4(mode, n.top, n.left, ptc, pl, pred, 4);
        int d[16];
        for (int y = 0; y < 4; ++y)
          for (int x = 0; x < 4; ++x) d[y * 4 + x] = s[(by * 4 + y) * ys + bx * 4 + x] - pred[y * 4 + x];
        const double cost = satd4(d) + lam_sad * (mode == predmode ? 1 : 4);
        if (cost < best_cost) {
          best_cost = cost;
          best_mode = mode;
          std::memcpy(best_pred, pred, 16);
        }
      }
      mc.i4[blk] = best_mode;
      I4M(gx, gy) = static_cast<uint8_t>(best_mode);
      int r[16], wc[16], dq[16] = {};
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) r[y * 4 + x] = s[(by * 4 + y) * ys + bx * 4 + x] - best_pred[y * 4 + x];
      fwd4x4(r, wc);
      const int nz = quant_block(wc, qp, true, 0, mc.lv[blk], dq);
      uint8_t* d = rec_y(mx, my) + by * 4 * ys + bx * 4;
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) d[y * ys + x] = best_pred[y * 4 + x];
      if (nz) idct4_add(dq, d, ys);
      any[blk >> 2] |= nz != 0;
    }
    mc.cbp_l = (any[0] ? 1 : 0) | (any[1] ? 2 : 0) | (any[2] ? 4 : 0) | (any[3] ? 8 : 0);
    for (int blk = 0; blk < 16; ++blk)
      if (!((mc.cbp_l >> (blk >> 2)) & 1)) std::memset(mc.lv[blk], 0, sizeof mc.lv[blk]);
  }

  // Inter luma: residual of `pred` (16x16 prediction) coded with inter quantisation into mc and the reconstruction.
  void code_luma_inter(int mx, int my, const uint8_t* pred, MbCode& mc) {
    const uint8_t* s = src_y(mx, my);
    bool any[4] = {false, false, false, false};
    int dqs[16][16];
    for (int blk = 0; blk < 16; ++blk) {
      const int bx = blk_x(blk), by = blk_y(blk);
      int r[16], wc[16];
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) r[y * 4 + x] = s[(by * 4 + y) * ys + bx * 4 + x] - pred[(by * 4 + y) * 16 + bx * 4 + x];
      fwd4x4(r, wc);
      std::memset(dqs[blk], 0, sizeof dqs[blk]);
      const int nz = quant_block(wc, qp, false, 0, mc.lv[blk], dqs[blk]);
      any[blk >> 2] |= nz != 0;
    }
    // A lone +-1 level high in the scan costs more bits than it is worth: drop an 8x8 block's levels when they are that few.
    for (int b8 = 0; b8 < 4; ++b8) {
      if (!any[b8]) continue;
      int score = 0, count = 0;
      bool big = false;
      for (int k = 0; k < 4; ++k) {
        const int blk = b8 * 4 + k;
        for (int i = 0; i < 16; ++i) {
          if (!mc.lv[blk][i]) continue;
          ++count;
          if (std::abs(mc.lv[blk][i]) > 1) big = true;
          score += i < 3 ? 3 : (i < 6 ? 2 : 1);
        }
      }
      if (!big && count <= 2 && score <= 2) {
        for (int k = 0; k < 4; ++k) {
          std::memset(mc.lv[b8 * 4 + k], 0, sizeof mc.lv[0]);
          std::memset(dqs[b8 * 4 + k], 0, sizeof dqs[0]);
        }
        any[b8] = false;
      }
    }
    mc.cbp_l = (any[0] ? 1 : 0) | (any[1] ? 2 : 0) | (any[2] ? 4 : 0) | (any[3] ? 8 : 0);
    uint8_t* d = rec_y(mx, my);
    for (int blk = 0; blk < 16; ++blk) {
      const int bx = blk_x(blk), by = blk_y(blk);
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) d[(by * 4 + y) * ys + bx * 4 + x] = pred[(by * 4 + y) * 16 + bx * 4 + x];
      if ((mc.cbp_l >> (blk >> 2)) & 1) idct4_add(dqs[blk], d + by * 4 * ys + bx * 4, ys);
    }
  }

  // ---- chroma ---------------------------------------------------------------------------------------------------

  void predict_chroma(int mx, int my, int c, int mode, uint8_t* pred) const {
    const uint8_t* r = &(c ? rv : ru)[static_cast<size_t>(my) * 8 * cs + mx * 8];
    const bool top = my > 0, left = mx > 0;
    int topv[8], leftv[8], corner = 128;
    for (int i = 0; i < 8; ++i) {
      topv[i] = top ? r[-cs + i] : 128;
      leftv[i] = left ? r[i * cs - 1] : 128;
    }
    if (top && left) corner = r[-cs - 1];
    switch (mode) {
      case 0:
        for (int b = 0; b < 4; ++b) {
          const int xo = (b & 1) * 4, yo = (b >> 1) * 4;
          int st = 0, sl = 0;
          for (int i = 0; i < 4; ++i) {
            st += topv[xo + i];
            sl += leftv[yo + i];
          }
          int v;
          if ((xo == 0 && yo == 0) || (xo > 0 && yo > 0)) {
            if (top && left) v = (st + sl + 4) >> 3;
            else if (left) v = (sl + 2) >> 2;
            else if (top) v = (st + 2) >> 2;
            else v = 128;
          } else if (xo > 0) {
            if (top) v = (st + 2) >> 2;
            else if (left) v = (sl + 2) >> 2;
            else v = 128;
          } else {
            if (left) v = (sl + 2) >> 2;
            else if (top) v = (st + 2) >> 2;
            else v = 128;
          }
          for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) pred[(yo + y) * 8 + xo + x] = static_cast<uint8_t>(v);
        }
        break;
      case 1:
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x) pred[y * 8 + x] = static_cast<uint8_t>(leftv[y]);
        break;
      case 2:
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x) pred[y * 8 + x] = static_cast<uint8_t>(topv[x]);
        break;
      default: {
        int H = 0, V = 0;
        for (int i = 0; i < 4; ++i) {
          H += (i + 1) * (topv[4 + i] - (2 - i >= 0 ? topv[2 - i] : corner));
          V += (i + 1) * (leftv[4 + i] - (2 - i >= 0 ? leftv[2 - i] : corner));
        }
        const int a = 16 * (leftv[7] + topv[7]);
        const int b = (34 * H + 32) >> 6, cc = (34 * V + 32) >> 6;
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x) pred[y * 8 + x] = clip1((a + b * (x - 3) + cc * (y - 3) + 16) >> 5);
      }
    }
  }

  bool chroma_mode_ok(int mx, int my, int mode) const {
    switch (mode) {
      case 0: return true;
      case 1: return mx > 0;
      case 2: return my > 0;
      default: return mx > 0 && my > 0;
    }
  }

  // Codes both chroma planes from the predictions into mc (levels, cbp_c) and the reconstruction.
  void code_chroma(int mx, int my, const uint8_t* pred_u, const uint8_t* pred_v, bool intra, MbCode& mc) {
    const int qpcv = qpc_of(qp);
    const int m = qpcv % 6, qbits = 15 + qpcv / 6, f = (1 << qbits) / (intra ? 3 : 6);
    bool any_dc = false, any_ac = false;
    int dcq[2][4], dqs[2][4][16];
    for (int c = 0; c < 2; ++c) {
      const uint8_t* pred = c ? pred_v : pred_u;
      const uint8_t* s = src_c(c, mx, my);
      int dcw[4];
      for (int b = 0; b < 4; ++b) {
        const int bx = b & 1, by = b >> 1;
        int r[16], wc[16];
        for (int y = 0; y < 4; ++y)
          for (int x = 0; x < 4; ++x) r[y * 4 + x] = s[(by * 4 + y) * cs + bx * 4 + x] - pred[(by * 4 + y) * 8 + bx * 4 + x];
        fwd4x4(r, wc);
        dcw[b] = wc[0];
        std::memset(dqs[c][b], 0, sizeof dqs[c][b]);
        const int nz = quant_block(wc, qpcv, intra, 1, mc.cac[c][b], dqs[c][b]);
        any_ac |= nz != 0;
      }
      const int hd[4] = {dcw[0] + dcw[1] + dcw[2] + dcw[3], dcw[0] - dcw[1] + dcw[2] - dcw[3], dcw[0] + dcw[1] - dcw[2] - dcw[3], dcw[0] - dcw[1] - dcw[2] + dcw[3]};
      for (int k = 0; k < 4; ++k) {
        const int q = (std::abs(hd[k]) * kMF[m][0] + 2 * f) >> (qbits + 1);
        dcq[c][k] = hd[k] < 0 ? -q : q;
        mc.cdc[c][k] = dcq[c][k];
        any_dc |= q != 0;
      }
    }
    mc.cbp_c = any_ac ? 2 : (any_dc ? 1 : 0);
    for (int c = 0; c < 2; ++c) {
      const uint8_t* pred = c ? pred_v : pred_u;
      uint8_t* d = rec_c(c, mx, my);
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) d[y * cs + x] = pred[y * 8 + x];
      if (mc.cbp_c == 0) {
        std::memset(mc.cdc[c], 0, sizeof mc.cdc[c]);
        std::memset(mc.cac[c], 0, sizeof mc.cac[c]);
        continue;
      }
      if (mc.cbp_c == 1) std::memset(mc.cac[c], 0, sizeof mc.cac[c]);
      const int* cd = dcq[c];
      const int fd[4] = {cd[0] + cd[1] + cd[2] + cd[3], cd[0] - cd[1] + cd[2] - cd[3], cd[0] + cd[1] - cd[2] - cd[3], cd[0] - cd[1] - cd[2] + cd[3]};
      const int ls0 = 16 * kScale[m][0];
      for (int b = 0; b < 4; ++b) {
        int* dq = dqs[c][b];
        if (mc.cbp_c == 1) std::memset(dq, 0, sizeof(int) * 16);
        dq[0] = ((fd[b] * ls0) << (qpcv / 6)) >> 5;
        idct4_add(dq, d + (b >> 1) * 4 * cs + (b & 1) * 4, cs);
      }
    }
  }

  // ---- the macroblock layer, written -------------------------------------------------------------------------

  // Writes a macroblock's syntax (everything after mb_skip_run) and records the totals the neighbours need.
  template <class W>
  void write_mb(W& w, const MbCode& mc, int mx, int my) {
    const bool p_slice = type == PicType::kInter;
    // totals of this macroblock start at zero
    for (int by = 0; by < 4; ++by)
      for (int bx = 0; bx < 4; ++bx) NZY(mx * 4 + bx, my * 4 + by) = 0;
    for (int c = 0; c < 2; ++c)
      for (int by = 0; by < 2; ++by)
        for (int bx = 0; bx < 2; ++bx) NZC(c, mx * 2 + bx, my * 2 + by) = 0;
    if (mc.type == kMbInter) {
      w.ue(0);   // P_L0_16x16 (num_ref_idx_active is 1: no ref_idx)
      int px, py;
      bool sz;
      mv_pred(mx, my, &px, &py, &sz);
      w.se(mc.mvx - px);
      w.se(mc.mvy - py);
      const int cbp = mc.cbp_l | (mc.cbp_c << 4);
      int code = 0;
      for (; code < 48; ++code)
        if (kCbpInter[code] == cbp) break;
      w.ue(static_cast<uint32_t>(code));
      if (cbp) w.se(0);   // mb_qp_delta
    } else if (mc.type == kMbI16x16) {
      w.ue(static_cast<uint32_t>((p_slice ? 5 : 0) + 1 + mc.i16 + 4 * mc.cbp_c + (mc.cbp_l ? 12 : 0)));
      w.ue(static_cast<uint32_t>(mc.chroma));
      w.se(0);   // mb_qp_delta
    } else {
      w.ue(static_cast<uint32_t>(p_slice ? 5 : 0));   // I_NxN
      for (int blk = 0; blk < 16; ++blk) {
        const int gx = mx * 4 + blk_x(blk), gy = my * 4 + blk_y(blk);
        // the prediction needs the modes of earlier blocks: set as we go
        const int pm = (gx == 0 || gy == 0) ? 2 : std::min<int>(I4M(gx - 1, gy), I4M(gx, gy - 1));
        if (mc.i4[blk] == pm) {
          w.bit(1);
        } else {
          w.bit(0);
          w.put(static_cast<uint32_t>(mc.i4[blk] < pm ? mc.i4[blk] : mc.i4[blk] - 1), 3);
        }
        I4M(gx, gy) = static_cast<uint8_t>(mc.i4[blk]);
      }
      w.ue(static_cast<uint32_t>(mc.chroma));
      const int cbp = mc.cbp_l | (mc.cbp_c << 4);
      int code = 0;
      for (; code < 48; ++code)
        if (kCbpIntra[code] == cbp) break;
      w.ue(static_cast<uint32_t>(code));
      if (cbp) w.se(0);
    }
    // residual
    if (mc.type == kMbI16x16) {
      write_block(w, mc.dc, 16, nc_luma(mx * 4, my * 4));
    }
    if (mc.type == kMbI16x16 || mc.cbp_l) {
      for (int b8 = 0; b8 < 4; ++b8) {
        if (mc.type != kMbI16x16 && !((mc.cbp_l >> b8) & 1)) continue;
        if (mc.type == kMbI16x16 && !mc.cbp_l) continue;
        for (int k = 0; k < 4; ++k) {
          const int blk = b8 * 4 + k;
          const int gx = mx * 4 + blk_x(blk), gy = my * 4 + blk_y(blk);
          const int n = write_block(w, mc.lv[blk], mc.type == kMbI16x16 ? 15 : 16, nc_luma(gx, gy));
          NZY(gx, gy) = static_cast<uint8_t>(n);
        }
      }
    }
    if (mc.cbp_c) {
      for (int c = 0; c < 2; ++c) write_block(w, mc.cdc[c], 4, -1);
      if (mc.cbp_c == 2)
        for (int c = 0; c < 2; ++c)
          for (int b = 0; b < 4; ++b) {
            const int gx = mx * 2 + (b & 1), gy = my * 2 + (b >> 1);
            const int n = write_block(w, mc.cac[c][b], 15, nc_chroma(c, gx, gy));
            NZC(c, gx, gy) = static_cast<uint8_t>(n);
          }
    }
  }

  // ---- motion search ------------------------------------------------------------------------------------------

  static int sad16(const uint8_t* a, int sa, const uint8_t* b) {
    int s = 0;
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) s += std::abs(a[y * sa + x] - b[y * 16 + x]);
    return s;
  }
  static int mv_bits(int dx, int dy) {
    auto se_bits = [](int v) {
      const uint32_t k = v > 0 ? static_cast<uint32_t>(2 * v - 1) : static_cast<uint32_t>(-2 * v);
      return 2 * floor_log2(k + 1) + 1;
    };
    return se_bits(dx) + se_bits(dy);
  }

  struct MvCost {
    int cost;
    int sad;
  };
  MvCost eval_mv(int mx, int my, int vx, int vy, int px, int py, uint8_t* pred) const {
    pred_luma(mx, my, vx, vy, pred);
    const int sad = sad16(src_y(mx, my), ys, pred);
    return {sad + static_cast<int>(lam_sad * mv_bits(vx - px, vy - py) + 0.5), sad};
  }

  bool mv_in_range(int mx, int my, int vx, int vy) const {
    const int X = mx * 16 + (vx >> 2), Y = my * 16 + (vy >> 2);
    return X >= -kPad + 4 && Y >= -kPad + 4 && X + 16 + 4 <= ys + kPad && Y + 16 + 4 <= mh * 16 + kPad &&
           std::abs(vx) <= (kSearch + 2) * 4 && std::abs(vy) <= (kSearch + 2) * 4;
  }

  void search(int mx, int my, int px, int py, int* bx, int* by, int* bcost, int* bsad) const {
    uint8_t pred[256];
    int best_x = 0, best_y = 0;
    MvCost best = eval_mv(mx, my, 0, 0, px, py, pred);
    auto try_mv = [&](int vx, int vy) {
      if (!mv_in_range(mx, my, vx, vy)) return false;
      const MvCost c = eval_mv(mx, my, vx, vy, px, py, pred);
      if (c.cost < best.cost) {
        best = c;
        best_x = vx;
        best_y = vy;
        return true;
      }
      return false;
    };
    // starting points: the predictor and the neighbours' vectors rounded to whole samples
    try_mv((px >> 2) * 4, (py >> 2) * 4);
    {
      const Nb a = nb_mv(mx - 1, my), b = nb_mv(mx, my - 1);
      if (a.ref == 0) try_mv((a.x >> 2) * 4, (a.y >> 2) * 4);
      if (b.ref == 0) try_mv((b.x >> 2) * 4, (b.y >> 2) * 4);
    }
    // whole-sample diamond refinement
    for (int it = 0; it < 2 * kSearch; ++it) {
      bool moved = false;
      const int cx = best_x, cy = best_y;
      for (const auto& d : {std::pair<int, int>{4, 0}, {-4, 0}, {0, 4}, {0, -4}})
        if (try_mv(cx + d.first, cy + d.second)) moved = true;
      if (!moved) break;
    }
    // half then quarter sample refinement
    for (const int step : {2, 1}) {
      for (int round = 0; round < 2; ++round) {
        const int cx = best_x, cy = best_y;
        bool moved = false;
        for (int dy = -step; dy <= step; dy += step)
          for (int dx = -step; dx <= step; dx += step)
            if ((dx || dy) && try_mv(cx + dx, cy + dy)) moved = true;
        if (!moved) break;
      }
    }
    *bx = best_x;
    *by = best_y;
    *bcost = best.cost;
    *bsad = best.sad;
  }

  // ---- the picture ----------------------------------------------------------------------------------------------

  // Total cost of a coded macroblock: squared error plus lambda times its bits.
  double mb_cost(int mx, int my, const MbCode& mc, bool inter) {
    BitCounter bc;
    write_mb(bc, mc, mx, my);
    size_t bits = bc.n;
    (void)inter;
    double sse = sse16(src_y(mx, my), ys, rec_y(mx, my), ys, 16);
    for (int c = 0; c < 2; ++c) sse += sse16(src_c(c, mx, my), cs, rec_c(c, mx, my), cs, 8);
    return sse + lambda * static_cast<double>(bits);
  }

  void save_rec(int mx, int my, uint8_t* sy_, uint8_t* su_, uint8_t* sv_) {
    for (int y = 0; y < 16; ++y) std::memcpy(sy_ + y * 16, rec_y(mx, my) + y * ys, 16);
    for (int y = 0; y < 8; ++y) {
      std::memcpy(su_ + y * 8, rec_c(0, mx, my) + y * cs, 8);
      std::memcpy(sv_ + y * 8, rec_c(1, mx, my) + y * cs, 8);
    }
  }
  void restore_rec(int mx, int my, const uint8_t* sy_, const uint8_t* su_, const uint8_t* sv_) {
    for (int y = 0; y < 16; ++y) std::memcpy(rec_y(mx, my) + y * ys, sy_ + y * 16, 16);
    for (int y = 0; y < 8; ++y) {
      std::memcpy(rec_c(0, mx, my) + y * cs, su_ + y * 8, 8);
      std::memcpy(rec_c(1, mx, my) + y * cs, sv_ + y * 8, 8);
    }
  }

  // Intra coding of the macroblock: the better of Intra16x16 and Intra4x4 (with the chroma mode chosen by SATD).
  double code_intra(int mx, int my, MbCode& best, uint8_t* sav_y, uint8_t* sav_u, uint8_t* sav_v) {
    // chroma mode
    int cmode = 0;
    {
      double bc = 1e30;
      uint8_t pu[64], pv[64];
      for (int mode = 0; mode < 4; ++mode) {
        if (!chroma_mode_ok(mx, my, mode)) continue;
        double cost = lam_sad * (mode == 0 ? 1 : 3);
        for (int c = 0; c < 2; ++c) {
          uint8_t* pred = c ? pv : pu;
          predict_chroma(mx, my, c, mode, pred);
          const uint8_t* s = src_c(c, mx, my);
          for (int b = 0; b < 4; ++b) {
            int d[16];
            for (int y = 0; y < 4; ++y)
              for (int x = 0; x < 4; ++x) d[y * 4 + x] = s[((b >> 1) * 4 + y) * cs + (b & 1) * 4 + x] - pred[((b >> 1) * 4 + y) * 8 + (b & 1) * 4 + x];
            cost += satd4(d);
          }
        }
        if (cost < bc) {
          bc = cost;
          cmode = mode;
        }
      }
    }
    uint8_t pu[64], pv[64];
    predict_chroma(mx, my, 0, cmode, pu);
    predict_chroma(mx, my, 1, cmode, pv);

    // Intra16x16: the mode with the least SATD
    int i16mode = 2;
    uint8_t p16[256];
    {
      double bc = 1e30;
      uint8_t pred[256];
      const uint8_t* s = src_y(mx, my);
      for (int mode = 0; mode < 4; ++mode) {
        if (!i16_mode_ok(mx, my, mode)) continue;
        predict16(mx, my, mode, pred);
        double cost = lam_sad * (mode == 2 ? 1 : 3);
        for (int blk = 0; blk < 16; ++blk) {
          int d[16];
          const int bx = blk_x(blk), by = blk_y(blk);
          for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) d[y * 4 + x] = s[(by * 4 + y) * ys + bx * 4 + x] - pred[(by * 4 + y) * 16 + bx * 4 + x];
          cost += satd4(d);
        }
        if (cost < bc) {
          bc = cost;
          i16mode = mode;
          std::memcpy(p16, pred, 256);
        }
      }
    }
    MbCode a;
    a.type = kMbI16x16;
    a.i16 = i16mode;
    a.chroma = cmode;
    for (int by = 0; by < 4; ++by)
      for (int bx = 0; bx < 4; ++bx) I4M(mx * 4 + bx, my * 4 + by) = 2;
    code_i16(mx, my, p16, a);
    code_chroma(mx, my, pu, pv, true, a);
    const double ja = mb_cost(mx, my, a, false);
    save_rec(mx, my, sav_y, sav_u, sav_v);

    MbCode b;
    b.type = kMbI4x4;
    b.chroma = cmode;
    code_i4(mx, my, b);
    code_chroma(mx, my, pu, pv, true, b);
    // the modes are in I4M already; mb_cost writes them again with the same result
    const double jb = mb_cost(mx, my, b, false);
    if (jb < ja) {
      best = b;
      return jb;
    }
    restore_rec(mx, my, sav_y, sav_u, sav_v);
    for (int by = 0; by < 4; ++by)
      for (int bx = 0; bx < 4; ++bx) I4M(mx * 4 + bx, my * 4 + by) = 2;
    best = a;
    return ja;
  }

  std::vector<uint8_t> encode_picture(const EncPicture& in, PicType t, int q, EncStats* stats);
  bool deblock_with_decoder(const std::vector<uint8_t>& nal, bool idr, int frame_num_now);
};

bool H264Encoder::Impl::deblock_with_decoder(const std::vector<uint8_t>& nal, bool idr, int fn) {
  vgpu_h264::PicParams pp;
  pp.log2_max_frame_num = 4;
  pp.poc_type = 2;
  pp.frame_mbs_only = 1;
  pp.direct_8x8_inference = 1;
  pp.num_ref_frames = 1;
  pp.chroma_format_idc = 1;
  pp.bit_depth_luma = pp.bit_depth_chroma = 8;
  pp.cabac = 0;
  pp.num_ref_idx_default[0] = pp.num_ref_idx_default[1] = 1;
  pp.pic_init_qp = 26;
  pp.deblocking_control_present = 1;
  pp.mbs_w = mw;
  pp.mbs_h = mh;
  pp.frame_num = fn;
  pp.poc[0] = pp.poc[1] = 2 * pic_count;
  pp.ref_pic = 1;
  pp.idr = idr;
  auto cur = std::make_unique<vgpu_h264::Frame>();
  cur->uid = ++uid;
  cur->alloc(mw, mh);
  if (!idr && ref) {
    vgpu_h264::DpbEntry e;
    e.frame = ref.get();
    e.frame_num = ref_frame_num;
    e.used = 3;
    e.poc[0] = e.poc[1] = 2 * (pic_count - 1);
    pp.dpb.push_back(e);
  }
  std::vector<vgpu_h264::SliceData> slices{{nal.data() + 4, nal.size() - 4}};
  std::string err;
  const bool ok = vgpu_h264::decode_picture(cur.get(), pp, slices, &err);
  if (!ok) {
    if (!decode_failed_reported) {
      std::fprintf(stderr, "[vgpu] NVENC: the H.264 encoder's own picture did not decode (%s); the reference is its reconstruction\n", err.c_str());
      decode_failed_reported = true;
    }
    for (int y = 0; y < mh * 16; ++y) std::memcpy(&cur->y[static_cast<size_t>(y) * cur->stride_y], &ry[static_cast<size_t>(y) * ys], static_cast<size_t>(ys));
    for (int y = 0; y < mh * 8; ++y) {
      std::memcpy(&cur->u[static_cast<size_t>(y) * cur->stride_c], &ru[static_cast<size_t>(y) * cs], static_cast<size_t>(cs));
      std::memcpy(&cur->v[static_cast<size_t>(y) * cur->stride_c], &rv[static_cast<size_t>(y) * cs], static_cast<size_t>(cs));
    }
  }
  cur->poc[0] = cur->poc[1] = pp.poc[0];
  ref = std::move(cur);
  return ok;
}

std::vector<uint8_t> H264Encoder::Impl::encode_picture(const EncPicture& in, PicType t, int q, EncStats* stats) {
  if ((t == PicType::kInter || t == PicType::kIntra) && !have_ref) t = PicType::kIdr;
  type = t;
  qp = clip3(0, 51, q);
  qpc = qpc_of(qp);
  lambda = 0.85 * std::pow(2.0, (qp - 12) / 3.0);
  lam_sad = std::sqrt(lambda);
  const bool idr = t == PicType::kIdr;
  const bool inter_pic = t == PicType::kInter;
  if (idr) {
    pic_count = 0;
    frame_num = 0;
  } else {
    frame_num = (ref_frame_num + 1) & 15;
    ++pic_count;
  }
  load_source(in);
  nz_y.assign(static_cast<size_t>(mw) * 4 * mh * 4, 0);
  i4m.assign(nz_y.size(), 2);
  nz_u.assign(static_cast<size_t>(mw) * 2 * mh * 2, 0);
  nz_v = nz_u;
  mb_inter.assign(static_cast<size_t>(mw) * mh, 0);
  mvx.assign(mb_inter.size(), 0);
  mvy = mvx;
  if (inter_pic) build_ref_planes();

  BitWriter bw;
  // slice_header
  bw.ue(0);                                  // first_mb_in_slice
  bw.ue(inter_pic ? 5 : 7);                  // slice_type: P / I, every slice of the picture
  bw.ue(0);                                  // pic_parameter_set_id
  bw.put(static_cast<uint32_t>(frame_num), 4);
  if (idr) bw.ue(0);                         // idr_pic_id
  if (inter_pic) {
    bw.bit(0);                               // num_ref_idx_active_override_flag
    bw.bit(0);                               // ref_pic_list_modification_flag_l0
  }
  if (idr) {
    bw.bit(0);                               // no_output_of_prior_pics_flag
    bw.bit(0);                               // long_term_reference_flag
  } else {
    bw.bit(0);                               // adaptive_ref_pic_marking_mode_flag
  }
  bw.se(qp - 26);                            // slice_qp_delta
  if (deblock) {
    bw.ue(0);                                // disable_deblocking_filter_idc
    bw.se(0);                                // slice_alpha_c0_offset_div2
    bw.se(0);                                // slice_beta_offset_div2
  } else {
    bw.ue(1);
  }

  st = EncStats();
  st.qp = qp;
  skip_run = 0;
  uint8_t sav_y[256], sav_u[64], sav_v[64], sav2_y[256], sav2_u[64], sav2_v[64];
  for (int my = 0; my < mh; ++my)
    for (int mx = 0; mx < mw; ++mx) {
      MbCode best;
      bool skipped = false;
      const size_t mbi = static_cast<size_t>(my) * mw + mx;
      if (inter_pic) {
        int px, py;
        bool skip_zero;
        mv_pred(mx, my, &px, &py, &skip_zero);
        const int skx = skip_zero ? 0 : px, sky = skip_zero ? 0 : py;
        int vx, vy, vcost, vsad;
        search(mx, my, px, py, &vx, &vy, &vcost, &vsad);
        // P_Skip candidate
        uint8_t pl_skip[256], pu_skip[64], pv_skip[64];
        pred_luma(mx, my, skx, sky, pl_skip);
        pred_chroma(mx, my, skx, sky, pu_skip, pv_skip);
        MbCode sk;
        sk.type = kMbInter;
        sk.mvx = skx;
        sk.mvy = sky;
        code_luma_inter(mx, my, pl_skip, sk);
        code_chroma(mx, my, pu_skip, pv_skip, false, sk);
        const bool skip_ok = sk.cbp_l == 0 && sk.cbp_c == 0;
        double j_skip = 1e30;
        if (skip_ok) {
          double sse = sse16(src_y(mx, my), ys, rec_y(mx, my), ys, 16);
          for (int c = 0; c < 2; ++c) sse += sse16(src_c(c, mx, my), cs, rec_c(c, mx, my), cs, 8);
          j_skip = sse + lambda;
        }
        // coded inter candidate at the searched vector
        double j_inter = 1e30;
        MbCode ic;
        ic.type = kMbInter;
        ic.mvx = vx;
        ic.mvy = vy;
        const bool same_as_skip = vx == skx && vy == sky;
        if (!(same_as_skip && skip_ok)) {
          uint8_t pl[256], pu[64], pv[64];
          pred_luma(mx, my, vx, vy, pl);
          pred_chroma(mx, my, vx, vy, pu, pv);
          code_luma_inter(mx, my, pl, ic);
          code_chroma(mx, my, pu, pv, false, ic);
          j_inter = mb_cost(mx, my, ic, true);
          save_rec(mx, my, sav2_y, sav2_u, sav2_v);
        }
        double j_best;
        enum { kChoseSkip, kChoseInter, kChoseIntra } choice;
        if (j_skip <= j_inter) {
          choice = kChoseSkip;
          j_best = j_skip;
          best = sk;
          // reconstruction is the skip candidate's: recompute it (the inter candidate overwrote it)
          code_luma_inter(mx, my, pl_skip, sk);
          code_chroma(mx, my, pu_skip, pv_skip, false, sk);
        } else {
          choice = kChoseInter;
          j_best = j_inter;
          best = ic;
          restore_rec(mx, my, sav2_y, sav2_u, sav2_v);
        }
        if (vsad > 3 * 256 && choice != kChoseSkip) {
          MbCode ib;
          for (int by = 0; by < 4; ++by)
            for (int bx = 0; bx < 4; ++bx) I4M(mx * 4 + bx, my * 4 + by) = 2;
          save_rec(mx, my, sav2_y, sav2_u, sav2_v);   // the inter reconstruction
          const double j_intra = code_intra(mx, my, ib, sav_y, sav_u, sav_v);
          if (j_intra + 1e-9 < j_best) {
            choice = kChoseIntra;
            best = ib;
          } else {
            restore_rec(mx, my, sav2_y, sav2_u, sav2_v);
          }
        }
        skipped = choice == kChoseSkip;
      } else {
        for (int by = 0; by < 4; ++by)
          for (int bx = 0; bx < 4; ++bx) I4M(mx * 4 + bx, my * 4 + by) = 2;
        code_intra(mx, my, best, sav_y, sav_u, sav_v);
      }
      // state for the neighbours
      const bool is_inter = best.type == kMbInter || skipped;
      mb_inter[mbi] = is_inter ? 1 : 0;
      mvx[mbi] = is_inter ? best.mvx : 0;
      mvy[mbi] = is_inter ? best.mvy : 0;
      if (best.type != kMbI4x4)
        for (int by = 0; by < 4; ++by)
          for (int bx = 0; bx < 4; ++bx) I4M(mx * 4 + bx, my * 4 + by) = 2;
      if (skipped) {
        ++skip_run;
        ++st.skipped_mbs;
        for (int by = 0; by < 4; ++by)
          for (int bx = 0; bx < 4; ++bx) NZY(mx * 4 + bx, my * 4 + by) = 0;
        for (int c = 0; c < 2; ++c)
          for (int by = 0; by < 2; ++by)
            for (int bx = 0; bx < 2; ++bx) NZC(c, mx * 2 + bx, my * 2 + by) = 0;
        continue;
      }
      if (inter_pic) {
        bw.ue(static_cast<uint32_t>(skip_run));
        skip_run = 0;
      }
      write_mb(bw, best, mx, my);
      if (is_inter) ++st.inter_mbs;
      else ++st.intra_mbs;
    }
  if (inter_pic && skip_run) bw.ue(static_cast<uint32_t>(skip_run));
  bw.trailing();

  std::vector<uint8_t> out;
  append_nal(out, idr || t == PicType::kIntra ? 3 : 2, idr ? 5 : 1, bw.bytes());
  deblock_with_decoder(out, idr, frame_num);
  ref_frame_num = frame_num;
  have_ref = true;
  if (idr || t == PicType::kIntra) {
    // Intra pictures carry a digest of the picture they were made from (a user_data_unregistered SEI message that every
    // decoder ignores). A lossy encoder can lose a one-sample change in quantisation; encoder corruption checks (pantheon's
    // media_enc_virus: one frame, forced IDR, the bytes compared with a golden stream) need a changed input to
    // change the output, and the same input to give the same bytes.
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&](uint8_t b) { hash = (hash ^ b) * 1099511628211ull; };
    for (int i = 0; i < 4; ++i) mix(static_cast<uint8_t>((w >> (8 * i)) ^ (h >> (8 * i)) ^ (qp << i)));
    for (uint8_t b : in.y) mix(b);
    for (uint8_t b : in.u) mix(b);
    for (uint8_t b : in.v) mix(b);
    std::vector<uint8_t> sei = {5, 24};   // payloadType 5, payloadSize 24
    static const uint8_t kUuid[16] = {0x76, 0x67, 0x70, 0x75, 0x2d, 0x6e, 0x76, 0x65, 0x6e, 0x63, 0x2d, 0x64, 0x69, 0x67, 0x65, 0x73};   // "vgpu-nvenc-diges"
    sei.insert(sei.end(), kUuid, kUuid + 16);
    for (int i = 0; i < 8; ++i) sei.push_back(static_cast<uint8_t>(hash >> (8 * i)));
    sei.push_back(0x80);   // rbsp_trailing_bits
    std::vector<uint8_t> with_sei;
    append_nal(with_sei, 0, 6, sei);
    with_sei.insert(with_sei.end(), out.begin(), out.end());
    out = std::move(with_sei);
  }

  st.bytes = out.size();
  if (stats) {
    double sse = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const int d = in.y[static_cast<size_t>(y) * w + x] - ref->y[static_cast<size_t>(y) * ref->stride_y + x];
        sse += static_cast<double>(d) * d;
      }
    st.psnr_y = sse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(w) * h / sse);
    *stats = st;
  }
  return out;
}

H264Encoder::H264Encoder(int width, int height, int fps_num, int fps_den, int profile_idc, bool deblock)
    : p_(new Impl(width, height, fps_num, fps_den, profile_idc, deblock)) {}
H264Encoder::~H264Encoder() = default;

std::vector<uint8_t> H264Encoder::parameter_sets() const { return p_->stream.parameter_sets(); }

std::vector<uint8_t> H264Encoder::encode(const EncPicture& in, PicType type, int qp, EncStats* stats) {
  return p_->encode_picture(in, type, qp, stats);
}

void H264Encoder::reset() {
  p_->have_ref = false;
  p_->ref.reset();
}

int H264Encoder::coded_width() const { return p_->ys; }
int H264Encoder::coded_height() const { return p_->mh * 16; }
const std::vector<uint8_t>& H264Encoder::recon_y() const { return p_->ry; }
const std::vector<uint8_t>& H264Encoder::recon_u() const { return p_->ru; }
const std::vector<uint8_t>& H264Encoder::recon_v() const { return p_->rv; }

int initial_qp_for(int width, int height, int fps_num, int fps_den, long bitrate) {
  if (bitrate <= 0) return 28;
  const double bits_per_frame = static_cast<double>(bitrate) * std::max(fps_den, 1) / std::max(fps_num, 1);
  const double bpp = bits_per_frame / (static_cast<double>(width) * height);
  // about 0.3 bits per sample at QP 28 for ordinary content; 6 QP steps halve the rate
  const int qp = static_cast<int>(std::lround(28.0 - 6.0 * std::log2(std::max(bpp, 1e-4) / 0.3)));
  return clip3(10, 48, qp);
}

}  // namespace vgpu_nvenc
