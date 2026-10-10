// See nvenc_hevc_enc.hpp.
#include "nvenc_hevc_enc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include "nvenc_cabac.hpp"
#include "nvenc_h264.hpp"   // BitWriter
#include "nvenc_hevc.hpp"   // append_hevc_nal

namespace vgpu_nvenc {
namespace {

#include "nvenc_hevc_tables.inc"

constexpr int kCtbLog2 = 5, kCtb = 32;   // coding tree blocks of 32x32 luma samples
constexpr int kMinCbLog2 = 3;            // coding units of 32, 16 and 8
constexpr int kMaxTrDepth = 3;           // max_transform_hierarchy_depth_intra / inter

inline int clip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }
inline uint8_t clip8(int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }
inline int ilog2(int n) {
  int l = 0;
  while ((1 << (l + 1)) <= n) ++l;
  return l;
}

// ---- transforms ---------------------------------------------------------------------------------------------------------------

// y[k] = sum_n T[k][n] x[n], the N-point forward core transform, by even/odd decomposition.
void fdct1d(const int32_t* x, int32_t* y, int N) {
  if (N == 4) {
    const int32_t s0 = x[0] + x[3], s1 = x[1] + x[2], d0 = x[0] - x[3], d1 = x[1] - x[2];
    y[0] = 64 * (s0 + s1);
    y[2] = 64 * (s0 - s1);
    y[1] = 83 * d0 + 36 * d1;
    y[3] = 36 * d0 - 83 * d1;
    return;
  }
  const DctMatrix& M = dct_matrix();
  int32_t E[16], O[16], YE[16];
  const int hn = N / 2;
  for (int n = 0; n < hn; ++n) {
    E[n] = x[n] + x[N - 1 - n];
    O[n] = x[n] - x[N - 1 - n];
  }
  fdct1d(E, YE, hn);
  for (int k = 0; k < hn; ++k) y[2 * k] = YE[k];
  const int step = 32 / N;
  for (int k = 1; k < N; k += 2) {
    int32_t s = 0;
    for (int n = 0; n < hn; ++n) s += M.t[k * step][n] * O[n];
    y[k] = s;
  }
}

// x[n] = sum_k T[k][n] y[k]
void idct1d(const int32_t* y, int32_t* x, int N) {
  if (N == 4) {
    const int32_t o0 = 83 * y[1] + 36 * y[3], o1 = 36 * y[1] - 83 * y[3], e0 = 64 * (y[0] + y[2]), e1 = 64 * (y[0] - y[2]);
    x[0] = e0 + o0;
    x[1] = e1 + o1;
    x[2] = e1 - o1;
    x[3] = e0 - o0;
    return;
  }
  const DctMatrix& M = dct_matrix();
  const int hn = N / 2;
  int32_t ye[16], E[16];
  for (int k = 0; k < hn; ++k) ye[k] = y[2 * k];
  idct1d(ye, E, hn);
  const int step = 32 / N;
  for (int n = 0; n < hn; ++n) {
    int32_t o = 0;
    for (int k = 1; k < N; k += 2) o += M.t[k * step][n] * y[k];
    x[n] = E[n] + o;
    x[N - 1 - n] = E[n] - o;
  }
}

void fdst4(const int32_t* x, int32_t* y) {
  for (int k = 0; k < 4; ++k) y[k] = kDst4[k][0] * x[0] + kDst4[k][1] * x[1] + kDst4[k][2] * x[2] + kDst4[k][3] * x[3];
}
void idst4(const int32_t* y, int32_t* x) {
  for (int n = 0; n < 4; ++n) x[n] = kDst4[0][n] * y[0] + kDst4[1][n] * y[1] + kDst4[2][n] * y[2] + kDst4[3][n] * y[3];
}

// Forward 2-D transform of an NxN residual block (raster) into coefficients (raster), scaled as the quantiser expects (8-bit video).
void forward_transform(const int32_t* res, int32_t* coef, int log2, bool dst) {
  const int N = 1 << log2;
  int32_t tmp[32 * 32], col[32], out[32];
  const int shift1 = log2 - 1, shift2 = log2 + 6;
  for (int y = 0; y < N; ++y) {   // rows
    if (dst) fdst4(res + y * N, out);
    else fdct1d(res + y * N, out, N);
    for (int k = 0; k < N; ++k) tmp[y * N + k] = (out[k] + (1 << (shift1 - 1))) >> shift1;
  }
  for (int x = 0; x < N; ++x) {   // columns
    for (int y = 0; y < N; ++y) col[y] = tmp[y * N + x];
    if (dst) fdst4(col, out);
    else fdct1d(col, out, N);
    for (int k = 0; k < N; ++k) coef[k * N + x] = (out[k] + (1 << (shift2 - 1))) >> shift2;
  }
}

// 8.6.4.2: scaled coefficients d (raster) to the residual (raster).
void inverse_transform(const int32_t* d, int32_t* res, int log2, bool dst) {
  const int N = 1 << log2;
  int32_t g[32 * 32], col[32], out[32];
  for (int x = 0; x < N; ++x) {   // columns first
    for (int y = 0; y < N; ++y) col[y] = d[y * N + x];
    if (dst) idst4(col, out);
    else idct1d(col, out, N);
    for (int y = 0; y < N; ++y) g[y * N + x] = clip3(-32768, 32767, (out[y] + 64) >> 7);
  }
  for (int y = 0; y < N; ++y) {   // then rows
    if (dst) idst4(g + y * N, out);
    else idct1d(g + y * N, out, N);
    for (int x = 0; x < N; ++x) res[y * N + x] = (out[x] + (1 << 11)) >> 12;
  }
}

// ---- intra prediction -------------------------------------------------------------------------------------------------------------

// refs: 4N + 1 samples: refs[i] = p[-1][2N-1-i] for i < 2N (left, bottom to top), refs[2N] = p[-1][-1], refs[2N+1+x] = p[x][-1]. Writes the
// NxN prediction (stride N). `filter` and `strong` come from 8.4.4.2.3.
void predict_from_refs(const int* refs, int log2, int mode, int cIdx, uint8_t* pred) {
  const int N = 1 << log2;
  auto left = [&](int y) { return refs[2 * N - 1 - y]; };   // y = -1 is the corner
  auto top = [&](int x) { return refs[2 * N + 1 + x]; };    // x = -1 is the corner
  if (mode == 0) {   // planar
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) pred[y * N + x] = static_cast<uint8_t>(((N - 1 - x) * left(y) + (x + 1) * top(N) + (N - 1 - y) * top(x) + (y + 1) * left(N) + N) >> (log2 + 1));
    return;
  }
  if (mode == 1) {   // DC
    int sum = N;
    for (int i = 0; i < N; ++i) sum += left(i) + top(i);
    const int dc = sum >> (log2 + 1);
    std::memset(pred, dc, static_cast<size_t>(N) * N);
    if (cIdx == 0 && N < 32) {
      pred[0] = static_cast<uint8_t>((left(0) + 2 * dc + top(0) + 2) >> 2);
      for (int x = 1; x < N; ++x) pred[x] = static_cast<uint8_t>((top(x) + 3 * dc + 2) >> 2);
      for (int y = 1; y < N; ++y) pred[y * N] = static_cast<uint8_t>((left(y) + 3 * dc + 2) >> 2);
    }
    return;
  }
  const int angle = kIntraAngle[mode];
  int ref[3 * 32 + 2];
  int* r = ref + 32;   // r[-32 .. 2N+...]
  if (mode >= 18) {
    for (int x = 0; x <= N; ++x) r[x] = top(x - 1);
    if (angle < 0) {
      const int last = (N * angle) >> 5;
      if (last < -1)
        for (int x = last; x <= -1; ++x) r[x] = left(-1 + ((x * kInvAngle[mode] + 128) >> 8));
    } else {
      for (int x = N + 1; x <= 2 * N; ++x) r[x] = top(x - 1);
    }
    for (int y = 0; y < N; ++y) {
      const int idx = ((y + 1) * angle) >> 5, fact = ((y + 1) * angle) & 31;
      for (int x = 0; x < N; ++x)
        pred[y * N + x] = static_cast<uint8_t>(fact ? ((32 - fact) * r[x + idx + 1] + fact * r[x + idx + 2] + 16) >> 5 : r[x + idx + 1]);
    }
    if (mode == 26 && cIdx == 0 && N < 32)
      for (int y = 0; y < N; ++y) pred[y * N] = clip8(top(0) + ((left(y) - left(-1)) >> 1));
  } else {
    for (int x = 0; x <= N; ++x) r[x] = left(x - 1);
    if (angle < 0) {
      const int last = (N * angle) >> 5;
      if (last < -1)
        for (int x = last; x <= -1; ++x) r[x] = top(-1 + ((x * kInvAngle[mode] + 128) >> 8));
    } else {
      for (int x = N + 1; x <= 2 * N; ++x) r[x] = left(x - 1);
    }
    for (int x = 0; x < N; ++x) {
      const int idx = ((x + 1) * angle) >> 5, fact = ((x + 1) * angle) & 31;
      for (int y = 0; y < N; ++y)
        pred[y * N + x] = static_cast<uint8_t>(fact ? ((32 - fact) * r[y + idx + 1] + fact * r[y + idx + 2] + 16) >> 5 : r[y + idx + 1]);
    }
    if (mode == 10 && cIdx == 0 && N < 32)
      for (int x = 0; x < N; ++x) pred[x] = clip8(left(0) + ((top(x) - top(-1)) >> 1));
  }
}

// 8.4.4.2.3: filtering of the neighbouring samples (luma only, 4:2:0). Returns the filtered array in `out` (same layout as refs).
void filter_refs(const int* refs, int log2, int mode, int* out) {
  const int N = 1 << log2;
  const int n = 4 * N + 1;
  bool filter = false;
  if (mode != 1 && N != 4) {
    const int dist = std::min(std::abs(mode - 26), std::abs(mode - 10));
    const int thres = N == 8 ? 7 : (N == 16 ? 1 : 0);
    filter = dist > thres;
  }
  if (!filter) {
    std::memcpy(out, refs, sizeof(int) * static_cast<size_t>(n));
    return;
  }
  const int corner = refs[2 * N], tl_end = refs[4 * N], bl_end = refs[0], tmid = refs[2 * N + N], lmid = refs[2 * N - N];
  if (N == 32 && std::abs(corner + tl_end - 2 * tmid) < (1 << 3) && std::abs(corner + bl_end - 2 * lmid) < (1 << 3)) {
    // bi-linear (strong) smoothing
    out[2 * N] = corner;
    out[0] = bl_end;
    out[4 * N] = tl_end;
    for (int y = 0; y < 63; ++y) out[2 * N - 1 - y] = ((63 - y) * corner + (y + 1) * bl_end + 32) >> 6;
    for (int x = 0; x < 63; ++x) out[2 * N + 1 + x] = ((63 - x) * corner + (x + 1) * tl_end + 32) >> 6;
    return;
  }
  out[0] = refs[0];
  out[n - 1] = refs[n - 1];
  for (int i = 1; i < n - 1; ++i) out[i] = (refs[i - 1] + 2 * refs[i] + refs[i + 1] + 2) >> 2;
}

// 4x4 Hadamard-transformed SAD of a raster difference block (a measure of the bits the residual will cost).
int satd4x4(const int* d, int stride) {
  int t[16], s = 0;
  for (int i = 0; i < 4; ++i) {
    const int a = d[i * stride], b = d[i * stride + 1], c = d[i * stride + 2], e = d[i * stride + 3];
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

// ---- the coded form of one coding unit ----------------------------------------------------------------------------------------------------

struct Ctxs {
  CabacCtx c[kCtxCount];
};
using CostCoder = CabacCost<kCtxCount>;
using BitCoder = CabacWriter<kCtxCount>;

// A coding unit as the syntax needs it. The transform tree is stored by node id (a 4-ary heap: the children of node n are 4n + 1 .. 4n + 4).
struct CuData {
  int x = 0, y = 0, log2 = 3;
  bool inter = false;
  bool nxn = false;
  uint8_t mode[4] = {1, 1, 1, 1};       // IntraPredModeY of each prediction block
  uint8_t mpm_flag[4] = {}, mpm_idx[4] = {}, rem_mode[4] = {};
  uint8_t chroma_idx = 4;               // intra_chroma_pred_mode
  uint8_t chroma_mode = 1;              // IntraPredModeC
  bool split[85] = {};
  uint8_t cbf_y[85] = {}, cbf_cb[85] = {}, cbf_cr[85] = {};
  std::vector<int16_t> ly, lcb, lcr;    // levels, raster; luma stride 1 << log2, chroma stride (1 << log2) / 2
  std::vector<uint8_t> ry, ru, rv;      // the reconstruction of the unit
  double cost = 0;
};

constexpr uint8_t kGroupIdx[32] = {0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9};
constexpr uint8_t kMinInGroup[10] = {0, 1, 2, 3, 4, 6, 8, 12, 16, 24};
constexpr uint8_t kCtxIdxMap4x4[16] = {0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8};

// the prediction block of an intra coding unit that holds the luma sample (x0, y0)
inline int pu_of(const CuData& cu, int x0, int y0) { return cu.nxn ? (((y0 - cu.y) >> 2) << 1) + ((x0 - cu.x) >> 2) : 0; }

// scanIdx (7.4.9.11): intra 4x4 luma and chroma blocks and 8x8 luma blocks scan along the prediction direction
inline int scan_idx_of(int log2, int cIdx, int pred_mode) {
  if (log2 == 2 || (log2 == 3 && cIdx == 0)) {
    if (pred_mode >= 6 && pred_mode <= 14) return 2;
    if (pred_mode >= 22 && pred_mode <= 30) return 1;
  }
  return 0;
}

// coeff_abs_level_remaining (9.3.3.11): a Rice prefix of up to four ones, then an Exp-Golomb suffix of order rice + 1
template <class C>
void write_remaining(C& c, int value, int rice) {
  if (value < (4 << rice)) {
    const int q = value >> rice;
    for (int i = 0; i < q; ++i) c.bypass(1);
    c.bypass(0);
    c.bypass_bits(static_cast<uint32_t>(value & ((1 << rice) - 1)), rice);
    return;
  }
  for (int i = 0; i < 4; ++i) c.bypass(1);
  int v = value - (4 << rice);
  int k = rice + 1;
  while (v >= (1 << k)) {
    c.bypass(1);
    v -= 1 << k;
    ++k;
  }
  c.bypass(0);
  c.bypass_bits(static_cast<uint32_t>(v), k);
}

// residual_coding (7.3.8.11) for one transform block of levels (raster, row stride `stride`) with at least one non-zero level.
template <class C>
void write_residual(C& c, const int16_t* lev, int stride, int log2, int cIdx, int scan_idx) {
  const ScanTables& S = scan_tables();
  const int sbl = log2 - 2;
  const int nsb = 1 << (2 * sbl);
  auto at = [&](int sb, int n, int* xc, int* yc) {
    *xc = (S.x[sbl][scan_idx][sb] << 2) + S.x[2][scan_idx][n];
    *yc = (S.y[sbl][scan_idx][sb] << 2) + S.y[2][scan_idx][n];
  };
  int last_sb = 0, last_n = 0;
  bool found = false;
  for (int sb = nsb - 1; sb >= 0 && !found; --sb)
    for (int n = 15; n >= 0; --n) {
      int xc, yc;
      at(sb, n, &xc, &yc);
      if (lev[yc * stride + xc]) {
        last_sb = sb;
        last_n = n;
        found = true;
        break;
      }
    }
  int lx, ly;
  at(last_sb, last_n, &lx, &ly);
  if (scan_idx == 2) std::swap(lx, ly);
  // last_sig_coeff_{x,y}_prefix and suffix
  {
    const int off = cIdx == 0 ? 3 * (log2 - 2) + ((log2 - 1) >> 2) : 15;
    const int shift = cIdx == 0 ? (log2 + 1) >> 2 : log2 - 2;
    const int cmax = (log2 << 1) - 1;
    const int px = kGroupIdx[lx], py = kGroupIdx[ly];
    auto prefix = [&](int base, int p) {
      for (int b = 0; b < p; ++b) c.decision(base + off + (b >> shift), 1);
      if (p < cmax) c.decision(base + off + (p >> shift), 0);
    };
    prefix(kCtxLastX, px);
    prefix(kCtxLastY, py);
    if (px > 3) c.bypass_bits(static_cast<uint32_t>(lx - kMinInGroup[px]), (px >> 1) - 1);
    if (py > 3) c.bypass_bits(static_cast<uint32_t>(ly - kMinInGroup[py]), (py >> 1) - 1);
  }
  uint8_t csbf[9][9] = {};   // coded_sub_block_flag by sub-block position (with a border of zeros on the right and below)
  int prev_c1 = 1;           // greater1Ctx state carried from the previous sub-block with coefficients
  for (int sb = last_sb; sb >= 0; --sb) {
    const int sx = S.x[sbl][scan_idx][sb], sy = S.y[sbl][scan_idx][sb];
    bool any = false;
    for (int n = 0; n < 16 && !any; ++n) {
      int xc, yc;
      at(sb, n, &xc, &yc);
      any = lev[yc * stride + xc] != 0;
    }
    bool coded = true;
    int infer_dc = 0;
    if (sb < last_sb && sb > 0) {
      const int ctx_inc = std::min(csbf[sx + 1][sy] + csbf[sx][sy + 1], 1) + (cIdx ? 2 : 0);
      c.decision(kCtxCsbf + ctx_inc, any ? 1 : 0);
      coded = any;
      infer_dc = 1;
    }
    csbf[sx][sy] = coded ? 1 : 0;
    if (!coded) continue;
    const int prev_csbf = csbf[sx + 1][sy] | (csbf[sx][sy + 1] << 1);
    int abs_v[16], neg[16], count = 0;
    if (sb == last_sb) {
      int xc, yc;
      at(sb, last_n, &xc, &yc);
      const int v = lev[yc * stride + xc];
      abs_v[count] = std::abs(v);
      neg[count] = v < 0;
      ++count;
    }
    for (int n = (sb == last_sb) ? last_n - 1 : 15; n >= 0; --n) {
      int xc, yc;
      at(sb, n, &xc, &yc);
      const int v = lev[yc * stride + xc];
      bool sig;
      if (n > 0 || !infer_dc) {
        // sig_coeff_flag, context 9.3.4.2.5
        int sig_ctx;
        const int xP = xc & 3, yP = yc & 3;
        if (log2 == 2) {
          sig_ctx = kCtxIdxMap4x4[(yc << 2) + xc];
        } else if (xc + yc == 0) {
          sig_ctx = 0;
        } else {
          if (prev_csbf == 0) sig_ctx = (xP + yP == 0) ? 2 : (xP + yP < 3) ? 1 : 0;
          else if (prev_csbf == 1) sig_ctx = (yP == 0) ? 2 : (yP == 1) ? 1 : 0;
          else if (prev_csbf == 2) sig_ctx = (xP == 0) ? 2 : (xP == 1) ? 1 : 0;
          else sig_ctx = 2;
          if (cIdx == 0) {
            if (sx > 0 || sy > 0) sig_ctx += 3;
            sig_ctx += log2 == 3 ? (scan_idx == 0 ? 9 : 15) : 21;
          } else {
            sig_ctx += log2 == 3 ? 9 : 12;
          }
        }
        sig = v != 0;
        c.decision(kCtxSig + (cIdx == 0 ? sig_ctx : 27 + sig_ctx), sig ? 1 : 0);
        if (sig) infer_dc = 0;
      } else {
        sig = true;   // the DC coefficient of a sub-block whose other coefficients are all zero
      }
      if (sig) {
        abs_v[count] = std::abs(v);
        neg[count] = v < 0;
        ++count;
      }
    }
    if (count == 0) continue;
    // greater1 flags (the first eight), one greater2 flag, signs, remaining levels
    int ctx_set = (sb > 0 && cIdx == 0) ? 2 : 0;
    if (prev_c1 == 0) ++ctx_set;
    int c1 = 1;
    int first_g1 = -1;
    const int ng1 = std::min(count, 8);
    uint8_t g1[8] = {};
    for (int j = 0; j < ng1; ++j) {
      g1[j] = abs_v[j] > 1;
      c.decision(kCtxGreater1 + (cIdx ? 16 : 0) + ctx_set * 4 + c1, g1[j]);
      if (g1[j]) {
        c1 = 0;
        if (first_g1 < 0) first_g1 = j;
      } else if (c1 > 0 && c1 < 3) {
        ++c1;
      }
    }
    prev_c1 = c1;
    uint8_t g2 = 0;
    if (first_g1 >= 0) {
      g2 = abs_v[first_g1] > 2;
      c.decision(kCtxGreater2 + (cIdx ? 4 : 0) + ctx_set, g2);
    }
    uint32_t signs = 0;
    for (int j = 0; j < count; ++j) signs = signs << 1 | static_cast<uint32_t>(neg[j]);
    c.bypass_bits(signs, count);
    int rice = 0;
    for (int j = 0; j < count; ++j) {
      const int base = 1 + (j < 8 ? g1[j] : 0) + (j == first_g1 ? g2 : 0);
      const int thres = j < 8 ? (j == first_g1 ? 3 : 2) : 1;
      if (base == thres) {
        const int rem = abs_v[j] - base;
        write_remaining(c, rem, rice);
        if (abs_v[j] > 3 * (1 << rice)) rice = std::min(rice + 1, 4);
      }
    }
  }
}

// transform_tree (7.3.8.8) with its transform units, for an intra coding unit.
template <class C>
void write_transform_tree(C& c, const CuData& cu, int node, int x0, int y0, int log2, int depth, int blk, int parent_cb, int parent_cr) {
  const int intra_split = cu.nxn ? 1 : 0;
  const int max_depth = kMaxTrDepth + intra_split;
  const bool split_coded = log2 <= 5 && log2 > 2 && depth < max_depth && !(intra_split && depth == 0);
  bool split;
  if (split_coded) {
    split = cu.split[node];
    c.decision(kCtxSplitTransform + 5 - log2, split ? 1 : 0);
  } else {
    split = log2 > 5 || (intra_split && depth == 0);
  }
  int cb = 0, cr = 0;
  if (log2 > 2) {
    if (depth == 0 || parent_cb) {
      cb = cu.cbf_cb[node];
      c.decision(kCtxCbfChroma + depth, cb);
    }
    if (depth == 0 || parent_cr) {
      cr = cu.cbf_cr[node];
      c.decision(kCtxCbfChroma + depth, cr);
    }
  } else {
    cb = parent_cb;
    cr = parent_cr;
  }
  if (split) {
    const int h = 1 << (log2 - 1);
    for (int i = 0; i < 4; ++i) write_transform_tree(c, cu, 4 * node + 1 + i, x0 + (i & 1) * h, y0 + (i >> 1) * h, log2 - 1, depth + 1, i, cb, cr);
    return;
  }
  // transform_unit
  const bool cbf_luma = cu.cbf_y[node] != 0;
  c.decision(kCtxCbfLuma + (depth == 0 ? 1 : 0), cbf_luma ? 1 : 0);
  const int size = 1 << cu.log2;
  if (cbf_luma) {
    const int scan = scan_idx_of(log2, 0, cu.mode[pu_of(cu, x0, y0)]);
    write_residual(c, cu.ly.data() + (y0 - cu.y) * size + (x0 - cu.x), size, log2, 0, scan);
  }
  const int cstride = size / 2;
  if (log2 > 2) {
    const int cx = (x0 - cu.x) / 2, cy = (y0 - cu.y) / 2;
    const int scan = scan_idx_of(log2 - 1, 1, cu.chroma_mode);
    if (cb) write_residual(c, cu.lcb.data() + cy * cstride + cx, cstride, log2 - 1, 1, scan);
    if (cr) write_residual(c, cu.lcr.data() + cy * cstride + cx, cstride, log2 - 1, 2, scan);
  } else if (blk == 3) {
    const int bx = x0 - 4 - cu.x, by = y0 - 4 - cu.y;   // the 8x8 area's origin, in luma samples
    const int scan = scan_idx_of(2, 1, cu.chroma_mode);
    if (cb) write_residual(c, cu.lcb.data() + (by / 2) * cstride + bx / 2, cstride, 2, 1, scan);
    if (cr) write_residual(c, cu.lcr.data() + (by / 2) * cstride + bx / 2, cstride, 2, 2, scan);
  }
}

// coding_unit (7.3.8.5) of an intra coding unit in an I slice: everything after split_cu_flag.
template <class C>
void write_cu_intra(C& c, const CuData& cu) {
  if (cu.log2 == kMinCbLog2) c.decision(kCtxPartMode, cu.nxn ? 0 : 1);
  const int npu = cu.nxn ? 4 : 1;
  for (int i = 0; i < npu; ++i) c.decision(kCtxPrevIntraLuma, cu.mpm_flag[i]);
  for (int i = 0; i < npu; ++i) {
    if (cu.mpm_flag[i]) {
      c.bypass(cu.mpm_idx[i] > 0);
      if (cu.mpm_idx[i] > 0) c.bypass(cu.mpm_idx[i] > 1);
    } else {
      c.bypass_bits(cu.rem_mode[i], 5);
    }
  }
  if (cu.chroma_idx == 4) {
    c.decision(kCtxChromaPred, 0);
  } else {
    c.decision(kCtxChromaPred, 1);
    c.bypass_bits(cu.chroma_idx, 2);
  }
  write_transform_tree(c, cu, 0, cu.x, cu.y, cu.log2, 0, 0, 0, 0);
}

// The cost of the split_cu_flag, the rate-distortion lambda for a QP (HM's 0.57 * 2^((QP - 12) / 3) on squared errors), built without libm
// so that every machine makes the same decisions.
double lambda_of(int qp) {
  static const double frac[3] = {1.0, 1.2599210498948732, 1.5874010519681994};
  const int x = qp - 12;
  const int q = x >= 0 ? x / 3 : -((-x + 2) / 3);
  const int r = x - 3 * q;
  return 0.57 * frac[r] * std::ldexp(1.0, q);
}

struct Decision {
  double cost = 0;
  std::vector<CuData> cus;
};

}  // namespace

struct HevcEncoder::Impl {
  int w, h, fps_num, fps_den;
  int W, H, cW, cH;      // coded luma and chroma sizes
  int ctbs_x, ctbs_y, ux, uy;
  std::vector<uint8_t> sy, su, sv;   // the picture, padded to the coded size
  std::vector<uint8_t> ry, ru, rv;   // reconstruction
  std::vector<uint8_t> ipm, dep;     // per 4x4 unit: IntraPredModeY, coding quadtree depth
  int qp = 28;
  double lambda = 1, lam_sad = 1;
  Ctxs est;                          // the contexts as the picture's decisions have advanced them
  bool have_ref = false;
  int poc = 0;
  EncStats st;

  Impl(int width, int height, int fn, int fd) : w(width), h(height), fps_num(fn), fps_den(fd) {
    W = (w + 7) & ~7;
    H = (h + 7) & ~7;
    cW = W / 2;
    cH = H / 2;
    ctbs_x = (W + kCtb - 1) / kCtb;
    ctbs_y = (H + kCtb - 1) / kCtb;
    ux = W / 4;
    uy = H / 4;
    ry.assign(static_cast<size_t>(W) * H, 0);
    ru.assign(static_cast<size_t>(cW) * cH, 128);
    rv = ru;
  }

  // ---- geometry ------------------------------------------------------------------------------------------------------------------

  int zs_of(int ux_, int uy_) const {
    int m = 0;
    for (int b = 0; b < 3; ++b) m |= (((ux_ >> b) & 1) << (2 * b)) | (((uy_ >> b) & 1) << (2 * b + 1));
    return ((uy_ >> 3) * ctbs_x + (ux_ >> 3)) * 64 + m;
  }
  // 6.4.1: is the luma sample (nx, ny) available to the block at (cx, cy)?
  bool avail(int cx, int cy, int nx, int ny) const {
    if (nx < 0 || ny < 0 || nx >= W || ny >= H) return false;
    return zs_of(nx >> 2, ny >> 2) <= zs_of(cx >> 2, cy >> 2);
  }
  uint8_t* rplane(int c) { return c == 0 ? ry.data() : (c == 1 ? ru.data() : rv.data()); }
  const uint8_t* rplane(int c) const { return c == 0 ? ry.data() : (c == 1 ? ru.data() : rv.data()); }
  const uint8_t* splane(int c) const { return c == 0 ? sy.data() : (c == 1 ? su.data() : sv.data()); }
  int stride_of(int c) const { return c == 0 ? W : cW; }

  void load_source(const EncPicture& in) {
    sy.assign(static_cast<size_t>(W) * H, 0);
    su.assign(static_cast<size_t>(cW) * cH, 0);
    sv = su;
    const int icw = (in.w + 1) / 2, ich = (in.h + 1) / 2;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) sy[static_cast<size_t>(y) * W + x] = in.y[static_cast<size_t>(std::min(y, in.h - 1)) * in.w + std::min(x, in.w - 1)];
    for (int y = 0; y < cH; ++y)
      for (int x = 0; x < cW; ++x) {
        const size_t i = static_cast<size_t>(std::min(y, ich - 1)) * icw + std::min(x, icw - 1);
        su[static_cast<size_t>(y) * cW + x] = in.u[i];
        sv[static_cast<size_t>(y) * cW + x] = in.v[i];
      }
  }

  // ---- intra prediction ---------------------------------------------------------------------------------------------------------

  // The 4N + 1 neighbouring samples of an NxN block at (x0, y0) of plane cIdx, with the substitution process of 8.4.4.2.2.
  void build_refs(int cIdx, int x0, int y0, int N, int* refs) const {
    const uint8_t* pl = rplane(cIdx);
    const int stride = stride_of(cIdx);
    const int sh = cIdx ? 1 : 0;
    const int lx = x0 << sh, ly = y0 << sh;
    const int g = 4 >> sh;   // samples that share one availability unit
    const int n = 4 * N + 1;
    bool have[129];
    int val[129];
    for (int y = 0; y < 2 * N; y += g) {
      const bool a = avail(lx, ly, lx - 1, (y0 + y) << sh);
      for (int k = 0; k < g; ++k) {
        have[2 * N - 1 - (y + k)] = a;
        val[2 * N - 1 - (y + k)] = a ? pl[static_cast<size_t>(y0 + y + k) * stride + x0 - 1] : 0;
      }
    }
    {
      const bool a = avail(lx, ly, lx - 1, ly - 1);
      have[2 * N] = a;
      val[2 * N] = a ? pl[static_cast<size_t>(y0 - 1) * stride + x0 - 1] : 0;
    }
    for (int x = 0; x < 2 * N; x += g) {
      const bool a = avail(lx, ly, (x0 + x) << sh, ly - 1);
      for (int k = 0; k < g; ++k) {
        have[2 * N + 1 + x + k] = a;
        val[2 * N + 1 + x + k] = a ? pl[static_cast<size_t>(y0 - 1) * stride + x0 + x + k] : 0;
      }
    }
    int first = -1;
    for (int i = 0; i < n; ++i)
      if (have[i]) {
        first = i;
        break;
      }
    if (first < 0) {
      for (int i = 0; i < n; ++i) refs[i] = 128;
      return;
    }
    refs[0] = have[0] ? val[0] : val[first];
    for (int i = 1; i < n; ++i) refs[i] = have[i] ? val[i] : refs[i - 1];
  }

  // The prediction of an NxN block with a mode, from the reconstruction.
  void predict_block(int cIdx, int x0, int y0, int log2, int mode, uint8_t* pred) const {
    int refs[129], f[129];
    build_refs(cIdx, x0, y0, 1 << log2, refs);
    if (cIdx == 0) {
      filter_refs(refs, log2, mode, f);
      predict_from_refs(f, log2, mode, 0, pred);
    } else {
      predict_from_refs(refs, log2, mode, cIdx, pred);
    }
  }

  // MPM derivation (8.4.2): the three candidate modes for a prediction block at (xPb, yPb).
  void mpm_list(int xPb, int yPb, int cand[3]) const {
    int a = 1, b = 1;
    if (xPb > 0) a = ipm[static_cast<size_t>(yPb >> 2) * ux + ((xPb - 1) >> 2)];
    if (yPb > 0 && (yPb - 1) >= ((yPb >> kCtbLog2) << kCtbLog2)) b = ipm[static_cast<size_t>((yPb - 1) >> 2) * ux + (xPb >> 2)];
    if (a > 34) a = 1;
    if (b > 34) b = 1;
    if (a == b) {
      if (a < 2) {
        cand[0] = 0;
        cand[1] = 1;
        cand[2] = 26;
      } else {
        cand[0] = a;
        cand[1] = 2 + ((a + 29) % 32);
        cand[2] = 2 + ((a - 2 + 1) % 32);
      }
    } else {
      cand[0] = a;
      cand[1] = b;
      cand[2] = (a != 0 && b != 0) ? 0 : ((a != 1 && b != 1) ? 1 : 26);
    }
  }
  void code_mode(CuData& cu, int pu, int mode) const {
    int cand[3];
    const int xPb = cu.x + (cu.nxn ? (pu & 1) * 4 : 0), yPb = cu.y + (cu.nxn ? (pu >> 1) * 4 : 0);
    mpm_list(xPb, yPb, cand);
    cu.mode[pu] = static_cast<uint8_t>(mode);
    cu.mpm_flag[pu] = 0;
    for (int i = 0; i < 3; ++i)
      if (cand[i] == mode) {
        cu.mpm_flag[pu] = 1;
        cu.mpm_idx[pu] = static_cast<uint8_t>(i);
      }
    if (!cu.mpm_flag[pu]) {
      std::sort(cand, cand + 3);
      int rem = mode;
      for (int i = 2; i >= 0; --i)
        if (mode > cand[i]) --rem;
      cu.rem_mode[pu] = static_cast<uint8_t>(rem);
    }
  }
  void set_ipm(int x, int y, int size, int mode) {
    for (int j = 0; j < size / 4; ++j)
      for (int i = 0; i < size / 4; ++i) ipm[static_cast<size_t>(y / 4 + j) * ux + x / 4 + i] = static_cast<uint8_t>(mode);
  }

  // ---- transform, quantisation, reconstruction of one block -------------------------------------------------------------

  // Codes the block of plane cIdx at (x0, y0) whose prediction is `pred`: levels into lev (raster, stride lev_stride), reconstruction into
  // the plane. Returns the number of non-zero levels; `sse` receives the squared error of the reconstruction.
  int code_block(int cIdx, int x0, int y0, int log2, const uint8_t* pred, int qpv, bool intra, int16_t* lev, int lev_stride, int64_t* sse) {
    const int N = 1 << log2;
    const uint8_t* s = splane(cIdx);
    uint8_t* r = rplane(cIdx);
    const int stride = stride_of(cIdx);
    int32_t res[32 * 32], coef[32 * 32], deq[32 * 32];
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) res[y * N + x] = s[static_cast<size_t>(y0 + y) * stride + x0 + x] - pred[y * N + x];
    const bool dst = cIdx == 0 && log2 == 2 && intra;
    forward_transform(res, coef, log2, dst);
    const int qbits = 21 - log2 + qpv / 6;
    const int64_t add = static_cast<int64_t>(intra ? 171 : 85) << (qbits - 9);
    const int scale = kQuantScale[qpv % 6];
    int nz = 0;
    for (int i = 0; i < N * N; ++i) {
      const int64_t a = std::abs(static_cast<int64_t>(coef[i]));
      int q = static_cast<int>((a * scale + add) >> qbits);
      q = std::min(q, 32767);
      const int v = coef[i] < 0 ? -q : q;
      lev[(i / N) * lev_stride + (i % N)] = static_cast<int16_t>(v);
      nz += v != 0;
    }
    int64_t err = 0;
    if (nz) {
      const int bd_shift = log2 + 3;
      for (int i = 0; i < N * N; ++i) {
        const int v = lev[(i / N) * lev_stride + (i % N)];
        const int64_t d = ((static_cast<int64_t>(v) * 16 * kLevelScale[qpv % 6]) << (qpv / 6)) + (int64_t{1} << (bd_shift - 1));
        deq[i] = static_cast<int32_t>(std::max<int64_t>(-32768, std::min<int64_t>(32767, d >> bd_shift)));
      }
      inverse_transform(deq, res, log2, dst);
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) r[static_cast<size_t>(y0 + y) * stride + x0 + x] = clip8(pred[y * N + x] + res[y * N + x]);
    } else {
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) r[static_cast<size_t>(y0 + y) * stride + x0 + x] = pred[y * N + x];
    }
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const int d = s[static_cast<size_t>(y0 + y) * stride + x0 + x] - r[static_cast<size_t>(y0 + y) * stride + x0 + x];
        err += d * d;
      }
    *sse += err;
    return nz;
  }

  // ---- coding one leaf of a coding unit's transform tree ---------------------------------------------------------------

  void code_leaf_luma(CuData& cu, int node, int x0, int y0, int log2, int mode, int64_t* sse) {
    uint8_t pred[32 * 32];
    predict_block(0, x0, y0, log2, mode, pred);
    const int size = 1 << cu.log2;
    cu.cbf_y[node] = code_block(0, x0, y0, log2, pred, qp, true, cu.ly.data() + (y0 - cu.y) * size + (x0 - cu.x), size, sse) > 0;
  }
  // both chroma blocks at chroma position (cx0, cy0); the flags go to `node`
  void code_leaf_chroma(CuData& cu, int node, int cx0, int cy0, int clog2, int64_t* sse) {
    uint8_t pred[16 * 16];
    const int cstride = (1 << cu.log2) / 2;
    const int qpc = chroma_qp(qp);
    for (int c = 1; c <= 2; ++c) {
      predict_block(c, cx0, cy0, clog2, cu.chroma_mode, pred);
      std::vector<int16_t>& lev = c == 1 ? cu.lcb : cu.lcr;
      const int nz = code_block(c, cx0, cy0, clog2, pred, qpc, true, lev.data() + (cy0 - cu.y / 2) * cstride + (cx0 - cu.x / 2), cstride, sse);
      (c == 1 ? cu.cbf_cb : cu.cbf_cr)[node] = nz > 0;
    }
  }

  // 8.4.3: the chroma prediction mode for intra_chroma_pred_mode idx and the luma mode
  static int chroma_mode_of(int idx, int luma_mode) {
    static const int list[4] = {0, 26, 10, 1};
    if (idx == 4) return luma_mode;
    return list[idx] == luma_mode ? 34 : list[idx];
  }

  // Chooses intra_chroma_pred_mode for the coding unit by SATD of the chroma prediction of the whole unit's chroma block.
  void choose_chroma(CuData& cu) {
    const int clog2 = cu.log2 - 1, cn = 1 << clog2;
    const int cx0 = cu.x / 2, cy0 = cu.y / 2;
    double best = 1e30;
    int best_idx = 4;
    for (int idx = 4; idx >= 0; --idx) {
      const int mode = chroma_mode_of(idx, cu.mode[0]);
      if (idx < 4) {
        bool dup = false;
        for (int j = 4; j > idx; --j)
          if (chroma_mode_of(j, cu.mode[0]) == mode) dup = true;
        if (dup) continue;
      }
      double cost = lam_sad * (idx == 4 ? 1 : 3);
      uint8_t pred[16 * 16];
      for (int c = 1; c <= 2; ++c) {
        predict_block(c, cx0, cy0, clog2, mode, pred);
        const uint8_t* s = splane(c);
        const int stride = stride_of(c);
        for (int by = 0; by < cn; by += 4)
          for (int bx = 0; bx < cn; bx += 4) {
            int d[16];
            for (int y = 0; y < 4; ++y)
              for (int x = 0; x < 4; ++x) d[y * 4 + x] = s[static_cast<size_t>(cy0 + by + y) * stride + cx0 + bx + x] - pred[(by + y) * cn + bx + x];
            cost += satd4x4(d, 4);
          }
      }
      if (cost < best) {
        best = cost;
        best_idx = idx;
      }
    }
    cu.chroma_idx = static_cast<uint8_t>(best_idx);
    cu.chroma_mode = static_cast<uint8_t>(chroma_mode_of(best_idx, cu.mode[0]));
  }

  void init_cu(CuData& cu, int x, int y, int log2) {
    cu = CuData();
    cu.x = x;
    cu.y = y;
    cu.log2 = log2;
    const size_t area = static_cast<size_t>(1) << (2 * log2);
    cu.ly.assign(area, 0);
    cu.lcb.assign(area / 4, 0);
    cu.lcr.assign(area / 4, 0);
  }

  // ---- candidate: a 2N x 2N intra coding unit with a mode, and a transform tree of one leaf (policy 0) or four (policy 1) -------------

  void eval_2nx2n(CuData& cu, int x, int y, int log2, int mode, int policy, int64_t* sse_out) {
    init_cu(cu, x, y, log2);
    const int size = 1 << log2;
    set_ipm(x, y, size, 1);   // not yet coded: the neighbours of its own prediction block are outside
    code_mode(cu, 0, mode);
    set_ipm(x, y, size, mode);
    choose_chroma(cu);
    int64_t sse = 0;
    if (policy == 0) {
      code_leaf_luma(cu, 0, x, y, log2, mode, &sse);
      code_leaf_chroma(cu, 0, x / 2, y / 2, log2 - 1, &sse);
    } else {
      cu.split[0] = true;
      const int hl = log2 - 1, hs = 1 << hl;
      for (int i = 0; i < 4; ++i) {
        const int lx = x + (i & 1) * hs, ly = y + (i >> 1) * hs;
        code_leaf_luma(cu, 1 + i, lx, ly, hl, mode, &sse);
        if (hl > 2) code_leaf_chroma(cu, 1 + i, lx / 2, ly / 2, hl - 1, &sse);
      }
      if (hl > 2) {
        cu.cbf_cb[0] = cu.cbf_cb[1] | cu.cbf_cb[2] | cu.cbf_cb[3] | cu.cbf_cb[4];
        cu.cbf_cr[0] = cu.cbf_cr[1] | cu.cbf_cr[2] | cu.cbf_cr[3] | cu.cbf_cr[4];
      } else {
        code_leaf_chroma(cu, 0, x / 2, y / 2, 2, &sse);   // the 4x4 chroma blocks of the 8x8 unit
      }
    }
    *sse_out = sse;
  }

  // candidate: an 8x8 coding unit split into four 4x4 prediction blocks (PART_NxN), each with the mode its SATD ranks best
  void eval_nxn(CuData& cu, int x, int y, int64_t* sse_out) {
    init_cu(cu, x, y, 3);
    cu.nxn = true;
    cu.split[0] = true;
    set_ipm(x, y, 8, 1);
    int64_t sse = 0;
    for (int pu = 0; pu < 4; ++pu) {
      const int px = x + (pu & 1) * 4, py = y + (pu >> 1) * 4;
      int refs[129], f[129];
      build_refs(0, px, py, 4, refs);
      int cand[3];
      mpm_list(px, py, cand);
      double best = 1e30;
      int best_mode = 1;
      for (int mode = 0; mode < 35; ++mode) {
        filter_refs(refs, 2, mode, f);
        uint8_t pred[16];
        predict_from_refs(f, 2, mode, 0, pred);
        int d[16];
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i) d[j * 4 + i] = sy[static_cast<size_t>(py + j) * W + px + i] - pred[j * 4 + i];
        const bool in_mpm = mode == cand[0] || mode == cand[1] || mode == cand[2];
        const double cost = satd4x4(d, 4) + lam_sad * (in_mpm ? 2 : 6);
        if (cost < best) {
          best = cost;
          best_mode = mode;
        }
      }
      code_mode(cu, pu, best_mode);
      set_ipm(px, py, 4, best_mode);
      code_leaf_luma(cu, 1 + pu, px, py, 2, best_mode, &sse);
    }
    choose_chroma(cu);
    code_leaf_chroma(cu, 0, x / 2, y / 2, 2, &sse);
    *sse_out = sse;
  }

  // ---- rate-distortion decisions ----------------------------------------------------------------------------------------------

  // ctxInc of split_cu_flag (9.3.4.2.2): neighbours to the left and above that are coded deeper than this node
  int split_ctx_inc(int x0, int y0, int depth) const {
    int inc = 0;
    if (x0 > 0 && dep[static_cast<size_t>(y0 >> 2) * ux + ((x0 - 1) >> 2)] > depth) ++inc;
    if (y0 > 0 && dep[static_cast<size_t>((y0 - 1) >> 2) * ux + (x0 >> 2)] > depth) ++inc;
    return inc;
  }
  bool split_flag_coded(int x, int y, int log2) const { return x + (1 << log2) <= W && y + (1 << log2) <= H && log2 > kMinCbLog2; }

  // J = D + lambda * bits of a candidate evaluated with the contexts `from`; `ctx_out` (if given) receives the contexts after it.
  double cu_cost(const CuData& cu, int64_t sse, int depth, const Ctxs& from, Ctxs* ctx_out) const {
    CostCoder cc;
    std::memcpy(cc.ctx, from.c, sizeof cc.ctx);
    if (split_flag_coded(cu.x, cu.y, cu.log2)) cc.decision(kCtxSplitCu + split_ctx_inc(cu.x, cu.y, depth), 0);
    write_cu_intra(cc, cu);
    if (ctx_out) std::memcpy(ctx_out->c, cc.ctx, sizeof cc.ctx);
    return static_cast<double>(sse) + lambda * (cc.cost / 256.0);
  }

  // The modes worth a full evaluation for an N x N coding unit: SATD ranking of the modes (all of them for 8x8, a coarse-to-fine
  // search for larger units), the best three.
  int rank_modes(int x, int y, int log2, int out[3]) {
    const int N = 1 << log2;
    int refs[129], f[129];
    build_refs(0, x, y, N, refs);
    int cand[3];
    mpm_list(x, y, cand);
    struct Sc {
      double cost;
      int mode;
    };
    Sc sc[35];
    bool done[35] = {};
    int nsc = 0;
    auto eval = [&](int mode) {
      if (done[mode]) return;
      done[mode] = true;
      filter_refs(refs, log2, mode, f);
      uint8_t pred[32 * 32];
      predict_from_refs(f, log2, mode, 0, pred);
      double cost = 0;
      for (int by = 0; by < N; by += 4)
        for (int bx = 0; bx < N; bx += 4) {
          int d[16];
          for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) d[j * 4 + i] = sy[static_cast<size_t>(y + by + j) * W + x + bx + i] - pred[(by + j) * N + bx + i];
          cost += satd4x4(d, 4);
        }
      const bool in_mpm = mode == cand[0] || mode == cand[1] || mode == cand[2];
      sc[nsc++] = {cost + lam_sad * (in_mpm ? 2 : 6), mode};
    };
    auto order = [&] { std::sort(sc, sc + nsc, [](const Sc& a, const Sc& b) { return a.cost != b.cost ? a.cost < b.cost : a.mode < b.mode; }); };
    if (N <= 8) {
      for (int m = 0; m < 35; ++m) eval(m);
    } else {
      eval(0);
      eval(1);
      for (int m = 2; m < 35; m += 2) eval(m);
      for (int m = 0; m < 3; ++m) eval(cand[m]);
      order();
      const int n0 = nsc;
      for (int i = 0; i < std::min(n0, 2); ++i) {
        const int m = sc[i].mode;
        if (m >= 2) {
          if (m - 1 >= 2) eval(m - 1);
          if (m + 1 <= 34) eval(m + 1);
        }
      }
    }
    order();
    for (int i = 0; i < 3; ++i) out[i] = sc[std::min(i, nsc - 1)].mode;
    return std::min(3, nsc);
  }

  // Copies a coding unit's reconstruction into the picture and records its modes and depth.
  void commit(const CuData& cu, int depth) {
    const int size = 1 << cu.log2;
    for (int j = 0; j < size; ++j) std::memcpy(&ry[static_cast<size_t>(cu.y + j) * W + cu.x], &cu.ry[static_cast<size_t>(j) * size], static_cast<size_t>(size));
    for (int j = 0; j < size / 2; ++j) {
      std::memcpy(&ru[static_cast<size_t>(cu.y / 2 + j) * cW + cu.x / 2], &cu.ru[static_cast<size_t>(j) * (size / 2)], static_cast<size_t>(size / 2));
      std::memcpy(&rv[static_cast<size_t>(cu.y / 2 + j) * cW + cu.x / 2], &cu.rv[static_cast<size_t>(j) * (size / 2)], static_cast<size_t>(size / 2));
    }
    if (cu.nxn)
      for (int pu = 0; pu < 4; ++pu) set_ipm(cu.x + (pu & 1) * 4, cu.y + (pu >> 1) * 4, 4, cu.mode[pu]);
    else
      set_ipm(cu.x, cu.y, size, cu.mode[0]);
    for (int j = 0; j < size / 4; ++j)
      for (int i = 0; i < size / 4; ++i) dep[static_cast<size_t>(cu.y / 4 + j) * ux + cu.x / 4 + i] = static_cast<uint8_t>(depth);
  }
  // Saves the reconstruction of a just-coded unit into it.
  void snapshot(CuData& cu) const {
    const int size = 1 << cu.log2;
    cu.ry.resize(static_cast<size_t>(size) * size);
    cu.ru.resize(static_cast<size_t>(size) * size / 4);
    cu.rv.resize(cu.ru.size());
    for (int j = 0; j < size; ++j) std::memcpy(&cu.ry[static_cast<size_t>(j) * size], &ry[static_cast<size_t>(cu.y + j) * W + cu.x], static_cast<size_t>(size));
    for (int j = 0; j < size / 2; ++j) {
      std::memcpy(&cu.ru[static_cast<size_t>(j) * (size / 2)], &ru[static_cast<size_t>(cu.y / 2 + j) * cW + cu.x / 2], static_cast<size_t>(size / 2));
      std::memcpy(&cu.rv[static_cast<size_t>(j) * (size / 2)], &rv[static_cast<size_t>(cu.y / 2 + j) * cW + cu.x / 2], static_cast<size_t>(size / 2));
    }
  }

  static bool has_residual(const CuData& cu) {
    for (int i = 0; i < 85; ++i)
      if (cu.cbf_y[i] || cu.cbf_cb[i] || cu.cbf_cr[i]) return true;
    return false;
  }

  // The best way to code the whole of one coding unit as a single unit (not split further). The winner is committed.
  void best_unsplit(int x, int y, int log2, int depth, const Ctxs& ctx, Decision* out) {
    CuData best;
    double best_cost = 1e30;
    auto consider = [&](CuData& cu, int64_t sse) {
      snapshot(cu);
      cu.cost = cu_cost(cu, sse, depth, ctx, nullptr);
      if (cu.cost < best_cost) {
        best_cost = cu.cost;
        best = cu;
      }
    };
    int modes[3];
    const int nm = rank_modes(x, y, log2, modes);
    int best_mode = modes[0];
    double best_mode_cost = 1e30;
    CuData cu;
    for (int i = 0; i < nm; ++i) {
      int64_t sse;
      eval_2nx2n(cu, x, y, log2, modes[i], 0, &sse);
      snapshot(cu);
      const double c = cu_cost(cu, sse, depth, ctx, nullptr);
      cu.cost = c;
      if (c < best_cost) {
        best_cost = c;
        best = cu;
      }
      if (c < best_mode_cost) {
        best_mode_cost = c;
        best_mode = modes[i];
      }
    }
    {
      int64_t sse;
      eval_2nx2n(cu, x, y, log2, best_mode, 1, &sse);
      consider(cu, sse);
    }
    if (log2 == kMinCbLog2) {
      int64_t sse;
      eval_nxn(cu, x, y, &sse);
      consider(cu, sse);
    }
    commit(best, depth);
    out->cost = best_cost;
    out->cus.clear();
    out->cus.push_back(std::move(best));
  }

  // The best coding of the quadtree node (x, y, log2) and everything below it, committed; `ctx` advances over it.
  void decide(int x, int y, int log2, int depth, Ctxs& ctx, Decision* out) {
    const int size = 1 << log2;
    const bool inside = x + size <= W && y + size <= H;
    const Ctxs ctx_in = ctx;
    Decision a;
    bool have_a = false;
    if (inside) {
      best_unsplit(x, y, log2, depth, ctx_in, &a);
      have_a = true;
    }
    const bool try_split = log2 > kMinCbLog2 && (!have_a || has_residual(a.cus[0]));
    if (!try_split) {
      cu_cost(a.cus[0], 0, depth, ctx_in, &ctx);   // the contexts after it
      *out = std::move(a);
      return;
    }
    Ctxs c2 = ctx_in;
    Decision b;
    if (inside) {
      CostCoder cc;
      std::memcpy(cc.ctx, c2.c, sizeof cc.ctx);
      if (split_flag_coded(x, y, log2)) cc.decision(kCtxSplitCu + split_ctx_inc(x, y, depth), 1);
      std::memcpy(c2.c, cc.ctx, sizeof cc.ctx);
      b.cost = lambda * (cc.cost / 256.0);
    }
    const int h2 = size / 2;
    for (int i = 0; i < 4; ++i) {
      const int cx = x + (i & 1) * h2, cy = y + (i >> 1) * h2;
      if (cx >= W || cy >= H) continue;
      Decision cd;
      decide(cx, cy, log2 - 1, depth + 1, c2, &cd);
      b.cost += cd.cost;
      for (CuData& u : cd.cus) b.cus.push_back(std::move(u));
    }
    if (have_a && a.cost <= b.cost) {
      commit(a.cus[0], depth);
      cu_cost(a.cus[0], 0, depth, ctx_in, &ctx);
      *out = std::move(a);
    } else {
      ctx = c2;
      *out = std::move(b);
    }
  }

  // ---- emission -----------------------------------------------------------------------------------------------------------

  void emit_quadtree(BitCoder& bc, const std::vector<CuData>& cus, size_t& idx, int x0, int y0, int log2, int depth) {
    const int size = 1 << log2;
    bool split;
    if (split_flag_coded(x0, y0, log2)) {
      split = !(cus[idx].x == x0 && cus[idx].y == y0 && cus[idx].log2 == log2);
      bc.decision(kCtxSplitCu + split_ctx_inc(x0, y0, depth), split ? 1 : 0);
    } else {
      split = log2 > kMinCbLog2;
    }
    if (split) {
      const int h2 = size / 2;
      for (int i = 0; i < 4; ++i) {
        const int cx = x0 + (i & 1) * h2, cy = y0 + (i >> 1) * h2;
        if (cx < W && cy < H) emit_quadtree(bc, cus, idx, cx, cy, log2 - 1, depth + 1);
      }
    } else {
      write_cu_intra(bc, cus[idx]);
      ++idx;
    }
  }

  // ---- parameter sets, slice header, the picture -------------------------------------------------------------------------

  int level_idc() const {
    struct L {
      int idc;
      long max_luma_ps, max_luma_sr;
    };
    static const L levels[] = {{30, 36864, 552960},     {60, 122880, 3686400},     {63, 245760, 7372800},     {90, 552960, 16588800},
                               {93, 983040, 33177600},  {120, 2228224, 66846720},  {123, 2228224, 133693440}, {150, 8912896, 267386880},
                               {153, 8912896, 534773760}, {156, 8912896, 1069547520}, {180, 35651584, 1069547520}, {183, 35651584, 2139095040},
                               {186, 35651584, 4278190080L}};
    const long ps = static_cast<long>(W) * H;
    const long sr = ps * fps_num / std::max(fps_den, 1);
    for (const L& l : levels)
      if (ps <= l.max_luma_ps && sr <= l.max_luma_sr) return l.idc;
    return 186;
  }
  void profile_tier_level(BitWriter& bw) const {
    bw.put(0, 2);   // general_profile_space
    bw.bit(0);      // general_tier_flag
    bw.put(1, 5);   // general_profile_idc: Main
    for (int i = 0; i < 32; ++i) bw.bit(i == 1 || i == 2);
    bw.bit(1);      // general_progressive_source_flag
    bw.bit(0);      // general_interlaced_source_flag
    bw.bit(0);      // general_non_packed_constraint_flag
    bw.bit(1);      // general_frame_only_constraint_flag
    bw.put(0, 32);
    bw.put(0, 12);
    bw.put(static_cast<uint32_t>(level_idc()), 8);
  }
  std::vector<uint8_t> vps() const {
    BitWriter bw;
    bw.put(0, 4);
    bw.bit(1);
    bw.bit(1);
    bw.put(0, 6);
    bw.put(0, 3);
    bw.bit(1);
    bw.put(0xffff, 16);
    profile_tier_level(bw);
    bw.bit(1);
    bw.ue(0);
    bw.ue(0);
    bw.ue(0);
    bw.put(0, 6);
    bw.ue(0);
    bw.bit(0);
    bw.bit(0);
    bw.trailing();
    return bw.bytes();
  }
  std::vector<uint8_t> sps() const {
    BitWriter bw;
    bw.put(0, 4);   // sps_video_parameter_set_id
    bw.put(0, 3);   // sps_max_sub_layers_minus1
    bw.bit(1);      // sps_temporal_id_nesting_flag
    profile_tier_level(bw);
    bw.ue(0);       // sps_seq_parameter_set_id
    bw.ue(1);       // chroma_format_idc
    bw.ue(static_cast<uint32_t>(W));
    bw.ue(static_cast<uint32_t>(H));
    const int cw = (w + 1) & ~1, chh = (h + 1) & ~1;
    if (W != cw || H != chh) {
      bw.bit(1);    // conformance_window_flag, in units of two luma samples
      bw.ue(0);
      bw.ue(static_cast<uint32_t>((W - cw) / 2));
      bw.ue(0);
      bw.ue(static_cast<uint32_t>((H - chh) / 2));
    } else {
      bw.bit(0);
    }
    bw.ue(0);       // bit_depth_luma_minus8
    bw.ue(0);       // bit_depth_chroma_minus8
    bw.ue(4);       // log2_max_pic_order_cnt_lsb_minus4: 8 bits
    bw.bit(1);      // sps_sub_layer_ordering_info_present_flag
    bw.ue(0);       // sps_max_dec_pic_buffering_minus1
    bw.ue(0);       // sps_max_num_reorder_pics
    bw.ue(0);       // sps_max_latency_increase_plus1
    bw.ue(kMinCbLog2 - 3);               // log2_min_luma_coding_block_size_minus3
    bw.ue(kCtbLog2 - kMinCbLog2);        // log2_diff_max_min_luma_coding_block_size
    bw.ue(0);                            // log2_min_luma_transform_block_size_minus2
    bw.ue(3);                            // log2_diff_max_min_luma_transform_block_size: 4 .. 32
    bw.ue(kMaxTrDepth);                  // max_transform_hierarchy_depth_inter
    bw.ue(kMaxTrDepth);                  // max_transform_hierarchy_depth_intra
    bw.bit(0);      // scaling_list_enabled_flag
    bw.bit(0);      // amp_enabled_flag
    bw.bit(0);      // sample_adaptive_offset_enabled_flag
    bw.bit(0);      // pcm_enabled_flag
    bw.ue(0);       // num_short_term_ref_pic_sets
    bw.bit(0);      // long_term_ref_pics_present_flag
    bw.bit(0);      // sps_temporal_mvp_enabled_flag
    bw.bit(1);      // strong_intra_smoothing_enabled_flag
    bw.bit(0);      // vui_parameters_present_flag
    bw.bit(0);      // sps_extension_present_flag
    bw.trailing();
    return bw.bytes();
  }
  std::vector<uint8_t> pps() const {
    BitWriter bw;
    bw.ue(0);       // pps_pic_parameter_set_id
    bw.ue(0);       // pps_seq_parameter_set_id
    bw.bit(0);      // dependent_slice_segments_enabled_flag
    bw.bit(0);      // output_flag_present_flag
    bw.put(0, 3);   // num_extra_slice_header_bits
    bw.bit(0);      // sign_data_hiding_enabled_flag
    bw.bit(0);      // cabac_init_present_flag
    bw.ue(0);       // num_ref_idx_l0_default_active_minus1
    bw.ue(0);       // num_ref_idx_l1_default_active_minus1
    bw.se(0);       // init_qp_minus26
    bw.bit(0);      // constrained_intra_pred_flag
    bw.bit(0);      // transform_skip_enabled_flag
    bw.bit(0);      // cu_qp_delta_enabled_flag
    bw.se(0);       // pps_cb_qp_offset
    bw.se(0);       // pps_cr_qp_offset
    bw.bit(0);      // pps_slice_chroma_qp_offsets_present_flag
    bw.bit(0);      // weighted_pred_flag
    bw.bit(0);      // weighted_bipred_flag
    bw.bit(0);      // transquant_bypass_enabled_flag
    bw.bit(0);      // tiles_enabled_flag
    bw.bit(0);      // entropy_coding_sync_enabled_flag
    bw.bit(0);      // pps_loop_filter_across_slices_enabled_flag
    bw.bit(1);      // deblocking_filter_control_present_flag
    bw.bit(0);      // deblocking_filter_override_enabled_flag
    bw.bit(1);      // pps_deblocking_filter_disabled_flag
    bw.bit(0);      // pps_scaling_list_data_present_flag
    bw.bit(0);      // lists_modification_present_flag
    bw.ue(0);       // log2_parallel_merge_level_minus2
    bw.bit(0);      // slice_segment_header_extension_present_flag
    bw.bit(0);      // pps_extension_present_flag
    bw.trailing();
    return bw.bytes();
  }

  std::vector<uint8_t> encode_picture(const EncPicture& in, PicType t, int q, EncStats* stats);
};

std::vector<uint8_t> HevcEncoder::Impl::encode_picture(const EncPicture& in, PicType t, int q, EncStats* stats) {
  qp = clip3(0, 51, q);
  lambda = lambda_of(qp);
  lam_sad = std::sqrt(lambda);
  if (t == PicType::kInter || t == PicType::kIntra) {
    if (!have_ref) t = PicType::kIdr;
  }
  const bool idr = t == PicType::kIdr;
  if (idr) poc = 0;
  load_source(in);
  ipm.assign(static_cast<size_t>(ux) * uy, 1);
  dep.assign(static_cast<size_t>(ux) * uy, 0);
  std::fill(ry.begin(), ry.end(), 0);
  std::fill(ru.begin(), ru.end(), 128);
  std::fill(rv.begin(), rv.end(), 128);

  // slice segment header
  BitWriter hb;
  hb.bit(1);                       // first_slice_segment_in_pic_flag
  if (idr || true) hb.bit(0);      // no_output_of_prior_pics_flag (an IRAP picture)
  hb.ue(0);                        // slice_pic_parameter_set_id
  hb.ue(2);                        // slice_type: I
  hb.se(qp - 26);                  // slice_qp_delta
  hb.bit(1);                       // byte_alignment()
  hb.align_zero();
  std::vector<uint8_t> rbsp = hb.bytes();
  (void)t;

  // initialise the contexts (initType 0 for an I slice)
  const HevcInit& init = hevc_init();
  BitCoder bc(rbsp);
  for (int i = 0; i < kCtxCount; ++i) bc.ctx[i] = cabac_init_hevc(init.v[0][i], qp);
  std::memcpy(est.c, bc.ctx, sizeof est.c);
  st = EncStats();
  st.qp = qp;
  for (int cy = 0; cy < ctbs_y; ++cy)
    for (int cx = 0; cx < ctbs_x; ++cx) {
      std::memcpy(est.c, bc.ctx, sizeof est.c);   // the real contexts, as emission left them
      Decision d;
      decide(cx * kCtb, cy * kCtb, kCtbLog2, 0, est, &d);
      size_t idx = 0;
      emit_quadtree(bc, d.cus, idx, cx * kCtb, cy * kCtb, kCtbLog2, 0);
      st.intra_mbs += static_cast<int>(d.cus.size());
      bc.terminate(cx == ctbs_x - 1 && cy == ctbs_y - 1 ? 1 : 0);   // end_of_slice_segment_flag
    }
  bc.finish();

  std::vector<uint8_t> out;
  {
    // the digest of the picture, as a prefix SEI message (see picture_digest_payload)
    std::vector<uint8_t> sei = {5, 24};   // payloadType 5 (user_data_unregistered), payloadSize 24
    const std::vector<uint8_t> payload = picture_digest_payload(in, w, h, qp);
    sei.insert(sei.end(), payload.begin(), payload.end());
    sei.push_back(0x80);   // rbsp_trailing_bits
    append_hevc_nal(out, 39, sei);
  }
  append_hevc_nal(out, 19, rbsp);
  have_ref = true;
  st.bytes = out.size();
  double sse = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int d = in.y[static_cast<size_t>(y) * w + x] - ry[static_cast<size_t>(y) * W + x];
      sse += static_cast<double>(d) * d;
    }
  st.psnr_y = sse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(w) * h / sse);
  if (stats) *stats = st;
  return out;
}

HevcEncoder::HevcEncoder(int width, int height, int fps_num, int fps_den) : p_(new Impl(width, height, fps_num, fps_den)) {}
HevcEncoder::~HevcEncoder() = default;

std::vector<uint8_t> HevcEncoder::parameter_sets() const {
  std::vector<uint8_t> out;
  append_hevc_nal(out, 32, p_->vps());
  append_hevc_nal(out, 33, p_->sps());
  append_hevc_nal(out, 34, p_->pps());
  return out;
}
std::vector<uint8_t> HevcEncoder::encode(const EncPicture& in, PicType type, int qp, EncStats* stats) { return p_->encode_picture(in, type, qp, stats); }
void HevcEncoder::reset() { p_->have_ref = false; }
int HevcEncoder::coded_width() const { return p_->W; }
int HevcEncoder::coded_height() const { return p_->H; }
const std::vector<uint8_t>& HevcEncoder::recon_y() const { return p_->ry; }
const std::vector<uint8_t>& HevcEncoder::recon_u() const { return p_->ru; }
const std::vector<uint8_t>& HevcEncoder::recon_v() const { return p_->rv; }

}  // namespace vgpu_nvenc
