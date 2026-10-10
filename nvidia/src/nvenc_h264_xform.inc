// Part of nvenc_h264_enc.cpp (included inside its anonymous namespace): the arithmetic of the H.264 encoder -- the 4x4 and 8x8 integer
// transforms with their quantisation, intra prediction of 4x4 and 8x8 blocks, the CAVLC residual writer.

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
// luma4x4BlkIdx of the 4x4 block at raster position (bx, by) of the macroblock
inline int blk_idx(int bx, int by) { return ((by >> 1) << 3) | ((bx >> 1) << 2) | ((by & 1) << 1) | (bx & 1); }

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

// ---- the 8x8 transform (8.5.13) -------------------------------------------------------------------------------------------------

// normAdjust8x8 (8.5.9): v by qp % 6 and the position class of (i, j)
constexpr int kNorm8[6][6] = {{20, 18, 32, 19, 25, 24}, {22, 19, 35, 21, 28, 26}, {26, 23, 42, 24, 33, 31}, {28, 25, 45, 26, 35, 33}, {32, 28, 51, 30, 40, 38}, {36, 32, 58, 34, 46, 43}};
inline int norm8_of(int m, int i, int j) {
  int k;
  if (i % 4 == 0 && j % 4 == 0) k = 0;
  else if (i % 2 == 1 && j % 2 == 1) k = 1;
  else if (i % 4 == 2 && j % 4 == 2) k = 2;
  else if ((i % 4 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 4 == 0)) k = 3;
  else if ((i % 4 == 0 && j % 4 == 2) || (i % 4 == 2 && j % 4 == 0)) k = 4;
  else k = 5;
  return kNorm8[m][k];
}
// The integer forward core transform of the 8x8 blocks: the matrix whose rows the inverse transform of 8.5.13 is built from (scaled by 8). Its rows have
// squared norms 512, 578, 320, 578, 512, 578, 320, 578; the quantiser factors below divide those out.
constexpr int kT8[8][8] = {{8, 8, 8, 8, 8, 8, 8, 8},          {12, 10, 6, 3, -3, -6, -10, -12}, {8, 4, -4, -8, -8, -4, 4, 8},   {10, -3, -12, -6, 6, 12, 3, -10},
                           {8, -8, -8, 8, 8, -8, -8, 8},      {6, -12, 3, 10, -10, -3, 12, -6}, {4, -8, 8, -4, -4, 8, -8, 4},   {3, -6, 10, -12, 12, -10, 6, -3}};
constexpr int kT8Norm[8] = {512, 578, 320, 578, 512, 578, 320, 578};

struct Quant8 {
  int32_t mf[6][64];   // 2^36 / (n_i n_j v): level = (|W| * mf + f) >> (22 + qp / 6)
  Quant8() {
    for (int m = 0; m < 6; ++m)
      for (int k = 0; k < 64; ++k) {
        const int i = k >> 3, j = k & 7;
        const int64_t den = static_cast<int64_t>(kT8Norm[i]) * kT8Norm[j] * norm8_of(m, i, j);
        mf[m][k] = static_cast<int32_t>(((int64_t{1} << 36) + den / 2) / den);
      }
  }
};
const Quant8& quant8() {
  static const Quant8 q;
  return q;
}

// Forward 8x8 transform of a raster residual block.
void fwd8x8(const int* x, int* out) {
  int t[64];
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 8; ++k) {
      int s = 0;
      for (int n = 0; n < 8; ++n) s += kT8[k][n] * x[8 * i + n];
      t[8 * i + k] = s;
    }
  for (int j = 0; j < 8; ++j)
    for (int k = 0; k < 8; ++k) {
      int s = 0;
      for (int n = 0; n < 8; ++n) s += kT8[k][n] * t[8 * n + j];
      out[8 * k + j] = s;
    }
}

// 8x8 inverse transform of 8.5.13 of dequantised coefficients d (raster), added to the prediction in dst.
void idct8_add(const int* d, uint8_t* dst, int stride) {
  int g[64];
  auto pass = [](const int* in, int* out) {
    const int e0 = in[0] + in[4];
    const int e1 = -in[3] + in[5] - in[7] - (in[7] >> 1);
    const int e2 = in[0] - in[4];
    const int e3 = in[1] + in[7] - in[3] - (in[3] >> 1);
    const int e4 = (in[2] >> 1) - in[6];
    const int e5 = -in[1] + in[7] + in[5] + (in[5] >> 1);
    const int e6 = in[2] + (in[6] >> 1);
    const int e7 = in[3] + in[5] + in[1] + (in[1] >> 1);
    const int f0 = e0 + e6, f1 = e1 + (e7 >> 2), f2 = e2 + e4, f3 = e3 + (e5 >> 2), f4 = e2 - e4, f5 = (e3 >> 2) - e5, f6 = e0 - e6, f7 = e7 - (e1 >> 2);
    out[0] = f0 + f7;
    out[1] = f2 + f5;
    out[2] = f4 + f3;
    out[3] = f6 + f1;
    out[4] = f6 - f1;
    out[5] = f4 - f3;
    out[6] = f2 - f5;
    out[7] = f0 - f7;
  };
  for (int i = 0; i < 8; ++i) pass(d + 8 * i, g + 8 * i);
  for (int j = 0; j < 8; ++j) {
    int col[8], out[8];
    for (int i = 0; i < 8; ++i) col[i] = g[8 * i + j];
    pass(col, out);
    for (int i = 0; i < 8; ++i) {
      uint8_t* p = dst + i * stride + j;
      *p = clip1(*p + ((out[i] + 32) >> 6));
    }
  }
}

// Quantises an 8x8 block of transform coefficients into scan-order levels (`scan` maps scan position to raster offset); `dq` receives the dequantised
// coefficients as the decoder scales them (8.5.9, flat scaling lists). Returns the number of non-zero levels.
int quant_block8(const int* w, int qp, bool intra, int* lev, int* dq, const uint8_t* scan) {
  const Quant8& q = quant8();
  const int m = qp % 6, sh = qp / 6;
  const int qbits = 22 + sh;
  const int64_t f = (int64_t{1} << qbits) / (intra ? 3 : 6);
  int nz = 0;
  for (int i = 0; i < 64; ++i) {
    const int k = scan[i];
    const int64_t a = std::abs(w[k]);
    const int level = static_cast<int>((a * q.mf[m][k] + f) >> qbits);
    const int sl = w[k] < 0 ? -level : level;
    lev[i] = sl;
    nz += sl != 0;
    const int64_t ls = 16 * norm8_of(m, k >> 3, k & 7);
    dq[k] = sl == 0 ? 0 : static_cast<int>(qp >= 36 ? (static_cast<int64_t>(sl) * ls) * (int64_t{1} << (sh - 6)) : (static_cast<int64_t>(sl) * ls + (1 << (5 - sh))) >> (6 - sh));
  }
  return nz;
}

// ---- 4x4 ----------------------------------------------------------------------------------------------------------------------------

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
int quant_block(const int* w, int qp, bool intra, int first, int* lev, int* dq, const uint8_t* scan) {
  const int qbits = 15 + qp / 6, m = qp % 6;
  const int f = (1 << qbits) / (intra ? 3 : 6);
  int nz = 0;
  for (int i = first; i < 16; ++i) {
    const int k = scan[i];
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
// SATD of a w x h block of differences (stride `stride`), in 4x4 units
int satd_block(const int* d, int stride, int w, int h) {
  int s = 0;
  for (int y = 0; y < h; y += 4)
    for (int x = 0; x < w; x += 4) {
      int b[16];
      for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) b[j * 4 + i] = d[(y + j) * stride + x + i];
      s += satd4(b);
    }
  return s;
}

// ---- intra prediction: Intra_4x4 and Intra_8x8 blocks (8.3.1.2, 8.3.2.2) -------------------------------------------------------------------

// pt(x) is p[x, -1] for x in -1..2N-1 (-1: the corner), pl(y) is p[-1, y]. The N x N prediction goes to dst.
template <int N, class PT, class PL>
void predict_nxn(int mode, bool top_av, bool left_av, PT pt, PL pl, uint8_t* dst, int stride) {
  auto put = [&](int x, int y, int v) { dst[y * stride + x] = static_cast<uint8_t>(v); };
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
        v = (s + N) >> (N == 4 ? 3 : 4);
      } else if (left_av) {
        for (int i = 0; i < N; ++i) s += pl(i);
        v = (s + N / 2) >> (N == 4 ? 2 : 3);
      } else if (top_av) {
        for (int i = 0; i < N; ++i) s += pt(i);
        v = (s + N / 2) >> (N == 4 ? 2 : 3);
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
          if (x == N - 1 && y == N - 1) put(x, y, (pt(2 * N - 2) + 3 * pt(2 * N - 1) + 2) >> 2);
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

