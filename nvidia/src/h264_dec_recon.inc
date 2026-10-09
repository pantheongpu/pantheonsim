// Part of h264_decode.cpp: sample reconstruction -- intra prediction (8.3), scaling and
// inverse transforms (8.5), I_PCM (8.3.5).

constexpr int kNorm4[6][3] = {{10, 16, 13}, {11, 18, 14}, {13, 20, 16}, {14, 23, 18}, {16, 25, 20}, {18, 29, 23}};
constexpr int kNorm8[6][6] = {{20, 18, 32, 19, 25, 24}, {22, 19, 35, 21, 28, 26}, {26, 23, 42, 24, 33, 31},
                              {28, 25, 45, 26, 35, 33}, {32, 28, 51, 30, 40, 38}, {36, 32, 58, 34, 46, 43}};

inline int norm4(int m, int i, int j) { return kNorm4[m][(i % 2 == 0 && j % 2 == 0) ? 0 : ((i % 2 == 1 && j % 2 == 1) ? 1 : 2)]; }
inline int norm8(int m, int i, int j) {
  int k;
  if (i % 4 == 0 && j % 4 == 0) k = 0;
  else if (i % 2 == 1 && j % 2 == 1) k = 1;
  else if (i % 4 == 2 && j % 4 == 2) k = 2;
  else if ((i % 4 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 4 == 0)) k = 3;
  else if ((i % 4 == 0 && j % 4 == 2) || (i % 4 == 2 && j % 4 == 0)) k = 4;
  else k = 5;
  return kNorm8[m][k];
}

// 4x4 inverse transform (8.5.12.2) of the scaled coefficients d (raster) added to dst.
void idct4_add(int32_t* d, uint8_t* dst, int stride) {
  int t[16];
  for (int i = 0; i < 4; ++i) {
    const int32_t* r = d + 4 * i;
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

void idct8_add(int32_t* d, uint8_t* dst, int stride) {
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

// Scales the 4x4 block c (8.5.12.1) with weight scale list `ws`; the DC coefficient is taken as is when dc_given.
void scale4(const int32_t* c, int qp, const uint8_t* ws, bool dc_given, int32_t* d) {
  const int m = qp % 6, sh = qp / 6;
  for (int k = 0; k < 16; ++k) {
    if (k == 0 && dc_given) {
      d[0] = c[0];
      continue;
    }
    if (!c[k]) {
      d[k] = 0;
      continue;
    }
    const int ls = ws[k] * norm4(m, k >> 2, k & 3);
    d[k] = qp >= 24 ? (c[k] * ls) << (sh - 4) : (c[k] * ls + (1 << (3 - sh))) >> (4 - sh);
  }
}

void scale8(const int32_t* c, int qp, const uint8_t* ws, int32_t* d) {
  const int m = qp % 6, sh = qp / 6;
  for (int k = 0; k < 64; ++k) {
    if (!c[k]) {
      d[k] = 0;
      continue;
    }
    const int ls = ws[k] * norm8(m, k >> 3, k & 7);
    d[k] = qp >= 36 ? (c[k] * ls) << (sh - 6) : (c[k] * ls + (1 << (5 - sh))) >> (6 - sh);
  }
}

// ---- intra prediction ------------------------------------------------------------------------------------------------
bool Dec::intra_nb_avail(int addr) const {
  if (addr < 0) return false;
  if (addr == addr_) return true;
  return mbi(addr).intra || !pp_.constrained_intra_pred;
}

namespace {
// The prediction of an NxN block (N = 4 or 8) from its neighbouring samples. pt(x) is p[x, -1] (pt(-1) the corner),
// pl(y) is p[-1, y]. `tr` says whether the samples above and to the right are available beyond N (already substituted).
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
}  // namespace

void Dec::recon_intra() {
  const int qp = mb_->qp;
  if (mb_->kind == kIPCM) return;
  if (mb_->kind == kI16x16) {
    predict_intra16(i16_mode_);
    add_residual_luma();
  } else if (mb_->kind == kI4x4) {
    for (int blk = 0; blk < 16; ++blk) {
      const int bx = blk_x(blk), by = blk_y(blk);
      IntraNb nb;
      gather_intra_nb(bx * 4, by * 4, 4, blk == 3 || blk == 11, &nb);
      if (nb.top_av && !nb.tr_av) {
        for (int i = 4; i < 8; ++i) nb.top[1 + i] = nb.top[1 + 3];
      }
      uint8_t* dst = ly_ + by * 4 * ls_ + bx * 4;
      const int mode = mb_->ipred[by * 4 + bx];
      predict_nxn<4>(mode, nb.top_av, nb.left_av, [&](int x) { return x < 0 ? nb.top[0] : nb.top[1 + x]; },
                     [&](int y) { return y < 0 ? nb.top[0] : nb.left[y]; }, dst, ls_);
      if ((cbp_ >> (blk >> 2)) & 1) {
        int32_t d[16];
        scale4(&luma_coef_[blk * 16], qp, pp_.ws4[0], false, d);
        idct4_add(d, dst, ls_);
      }
    }
  } else {   // kI8x8
    for (int b8 = 0; b8 < 4; ++b8) {
      const int bx = (b8 & 1) * 8, by = (b8 >> 1) * 8;
      IntraNb nb;
      gather_intra_nb(bx, by, 8, b8 == 3, &nb);
      if (nb.top_av && !nb.tr_av) {
        for (int i = 8; i < 16; ++i) nb.top[1 + i] = nb.top[1 + 7];
      }
      // reference sample filtering (8.3.2.2.1)
      int ft[17], fl[8];
      ft[0] = nb.top[0];
      if (nb.top_av) {
        if (nb.tl) ft[1] = (nb.top[0] + 2 * nb.top[1] + nb.top[2] + 2) >> 2;
        else ft[1] = (3 * nb.top[1] + nb.top[2] + 2) >> 2;
        for (int x = 1; x < 15; ++x) ft[1 + x] = (nb.top[x] + 2 * nb.top[1 + x] + nb.top[2 + x] + 2) >> 2;
        ft[16] = (nb.top[15] + 3 * nb.top[16] + 2) >> 2;
      }
      if (nb.tl) {
        if (!nb.top_av || !nb.left_av) {
          if (nb.top_av) ft[0] = (3 * nb.top[0] + nb.top[1] + 2) >> 2;
          else if (nb.left_av) ft[0] = (3 * nb.top[0] + nb.left[0] + 2) >> 2;
          else ft[0] = nb.top[0];
        } else {
          ft[0] = (nb.top[1] + 2 * nb.top[0] + nb.left[0] + 2) >> 2;
        }
      }
      if (nb.left_av) {
        if (nb.tl) fl[0] = (nb.top[0] + 2 * nb.left[0] + nb.left[1] + 2) >> 2;
        else fl[0] = (3 * nb.left[0] + nb.left[1] + 2) >> 2;
        for (int y = 1; y < 7; ++y) fl[y] = (nb.left[y - 1] + 2 * nb.left[y] + nb.left[y + 1] + 2) >> 2;
        fl[7] = (nb.left[6] + 3 * nb.left[7] + 2) >> 2;
      }
      uint8_t* dst = ly_ + by * ls_ + bx;
      const int mode = mb_->ipred[(by / 4) * 4 + bx / 4];
      predict_nxn<8>(mode, nb.top_av, nb.left_av, [&](int x) { return x < 0 ? ft[0] : ft[1 + x]; }, [&](int y) { return y < 0 ? ft[0] : fl[y]; }, dst, ls_);
      if ((cbp_ >> b8) & 1) {
        int32_t d[64];
        scale8(&luma_coef_[b8 * 64], qp, pp_.ws8[0], d);
        idct8_add(d, dst, ls_);
      }
    }
  }
  predict_chroma(mb_->chroma_mode);
  add_residual_chroma();
}

void Dec::gather_intra_nb(int xo, int yo, int n, bool tr_blocked, IntraNb* nb) {
  auto get = [&](int x, int y, int* out) -> bool {
    const Nb l = locate_l(xo + x, yo + y);
    if (l.addr < 0 || !intra_nb_avail(l.addr)) return false;
    *out = *luma_at(l.addr, l.x, l.y);
    return true;
  };
  nb->tl = get(-1, -1, &nb->top[0]);
  nb->top_av = true;
  for (int x = 0; x < n; ++x)
    if (!get(x, -1, &nb->top[1 + x])) nb->top_av = false;
  nb->tr_av = !tr_blocked;
  if (nb->tr_av)
    for (int x = n; x < 2 * n; ++x)
      if (!get(x, -1, &nb->top[1 + x])) nb->tr_av = false;
  nb->left_av = true;
  for (int y = 0; y < n; ++y)
    if (!get(-1, y, &nb->left[y])) nb->left_av = false;
}

void Dec::predict_intra16(int mode) {
  int top[16], left[16], tl = 128;
  bool top_av = true, left_av = true, tl_av = false;
  auto get = [&](int x, int y, int* out) -> bool {
    const Nb l = locate_l(x, y);
    if (l.addr < 0 || !intra_nb_avail(l.addr)) return false;
    *out = *luma_at(l.addr, l.x, l.y);
    return true;
  };
  for (int x = 0; x < 16; ++x)
    if (!get(x, -1, &top[x])) top_av = false;
  for (int y = 0; y < 16; ++y)
    if (!get(-1, y, &left[y])) left_av = false;
  tl_av = get(-1, -1, &tl);
  uint8_t* dst = ly_;
  if (mode == 0) {
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) dst[y * ls_ + x] = static_cast<uint8_t>(top[x]);
  } else if (mode == 1) {
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) dst[y * ls_ + x] = static_cast<uint8_t>(left[y]);
  } else if (mode == 2) {
    int s = 0, v;
    if (top_av && left_av) {
      for (int i = 0; i < 16; ++i) s += top[i] + left[i];
      v = (s + 16) >> 5;
    } else if (left_av) {
      for (int i = 0; i < 16; ++i) s += left[i];
      v = (s + 8) >> 4;
    } else if (top_av) {
      for (int i = 0; i < 16; ++i) s += top[i];
      v = (s + 8) >> 4;
    } else {
      v = 128;
    }
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) dst[y * ls_ + x] = static_cast<uint8_t>(v);
  } else {
    (void)tl_av;
    int H = 0, V = 0;
    for (int i = 0; i < 8; ++i) {
      H += (i + 1) * (top[8 + i] - (6 - i >= 0 ? top[6 - i] : tl));
      V += (i + 1) * (left[8 + i] - (6 - i >= 0 ? left[6 - i] : tl));
    }
    const int a = 16 * (left[15] + top[15]);
    const int b = (5 * H + 32) >> 6, c = (5 * V + 32) >> 6;
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) dst[y * ls_ + x] = clip1((a + b * (x - 7) + c * (y - 7) + 16) >> 5);
  }
}

void Dec::predict_chroma(int mode) {
  for (int c = 0; c < 2; ++c) {
    uint8_t* dst = c == 0 ? lu_ : lv_;
    int top[8], left[8], tl = 128;
    bool top_av = true;
    bool left_av[2] = {true, true};   // upper and lower half
    auto get = [&](int x, int y, int* out) -> bool {
      const Nb l = locate_c(x, y);
      if (l.addr < 0 || !intra_nb_avail(l.addr)) return false;
      *out = *chroma_at(c, l.addr, l.x, l.y);
      return true;
    };
    for (int x = 0; x < 8; ++x)
      if (!get(x, -1, &top[x])) top_av = false;
    for (int y = 0; y < 8; ++y)
      if (!get(-1, y, &left[y])) left_av[y >> 2] = false;
    get(-1, -1, &tl);
    if (mode == 0) {
      // DC prediction per 4x4 chroma block (8.3.4.1)
      for (int blk = 0; blk < 4; ++blk) {
        const int xo = (blk & 1) * 4, yo = (blk >> 1) * 4;
        int st = 0, sl = 0;
        for (int i = 0; i < 4; ++i) {
          st += top[xo + i];
          sl += left[yo + i];
        }
        const bool la = left_av[yo >> 2], ta = top_av;
        int v;
        if ((xo == 0 && yo == 0) || (xo > 0 && yo > 0)) {
          if (ta && la) v = (st + sl + 4) >> 3;
          else if (la) v = (sl + 2) >> 2;
          else if (ta) v = (st + 2) >> 2;
          else v = 128;
        } else if (xo > 0 && yo == 0) {
          if (ta) v = (st + 2) >> 2;
          else if (la) v = (sl + 2) >> 2;
          else v = 128;
        } else {
          if (la) v = (sl + 2) >> 2;
          else if (ta) v = (st + 2) >> 2;
          else v = 128;
        }
        for (int y = 0; y < 4; ++y)
          for (int x = 0; x < 4; ++x) dst[(yo + y) * lcs_ + xo + x] = static_cast<uint8_t>(v);
      }
    } else if (mode == 1) {
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) dst[y * lcs_ + x] = static_cast<uint8_t>(left[y]);
    } else if (mode == 2) {
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) dst[y * lcs_ + x] = static_cast<uint8_t>(top[x]);
    } else {
      int H = 0, V = 0;
      for (int i = 0; i < 4; ++i) {
        H += (i + 1) * (top[4 + i] - (2 - i >= 0 ? top[2 - i] : tl));
        V += (i + 1) * (left[4 + i] - (2 - i >= 0 ? left[2 - i] : tl));
      }
      const int a = 16 * (left[7] + top[7]);
      const int b = (34 * H + 32) >> 6, cc = (34 * V + 32) >> 6;
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) dst[y * lcs_ + x] = clip1((a + b * (x - 3) + cc * (y - 3) + 16) >> 5);
    }
  }
}

// ---- residual ----------------------------------------------------------------------------------------------------------
void Dec::add_residual_luma() {
  const int qp = mb_->qp;
  const bool intra = mb_->intra;
  if (mb_->kind == kI16x16) {
    int32_t dc[16];
    // 8.5.10: inverse transform of the DC levels, then scaling
    int32_t f[16];
    {
      int32_t tmp[16];
      for (int i = 0; i < 4; ++i) {
        const int32_t* r = i16_dc_ + 4 * i;
        // c * A, with A = [1 1 1 1; 1 1 -1 -1; 1 -1 -1 1; 1 -1 1 -1] (symmetric)
        tmp[4 * i + 0] = r[0] + r[1] + r[2] + r[3];
        tmp[4 * i + 1] = r[0] + r[1] - r[2] - r[3];
        tmp[4 * i + 2] = r[0] - r[1] - r[2] + r[3];
        tmp[4 * i + 3] = r[0] - r[1] + r[2] - r[3];
      }
      for (int j = 0; j < 4; ++j) {
        f[0 * 4 + j] = tmp[0 * 4 + j] + tmp[1 * 4 + j] + tmp[2 * 4 + j] + tmp[3 * 4 + j];
        f[1 * 4 + j] = tmp[0 * 4 + j] + tmp[1 * 4 + j] - tmp[2 * 4 + j] - tmp[3 * 4 + j];
        f[2 * 4 + j] = tmp[0 * 4 + j] - tmp[1 * 4 + j] - tmp[2 * 4 + j] + tmp[3 * 4 + j];
        f[3 * 4 + j] = tmp[0 * 4 + j] - tmp[1 * 4 + j] + tmp[2 * 4 + j] - tmp[3 * 4 + j];
      }
    }
    const int ls0 = pp_.ws4[0][0] * kNorm4[qp % 6][0];
    for (int k = 0; k < 16; ++k) {
      dc[k] = qp >= 36 ? (f[k] * ls0) << (qp / 6 - 6) : (f[k] * ls0 + (1 << (5 - qp / 6))) >> (6 - qp / 6);
    }
    for (int blk = 0; blk < 16; ++blk) {
      const int bx = blk_x(blk), by = blk_y(blk);
      int32_t c[16];
      std::memcpy(c, &luma_coef_[blk * 16], sizeof c);
      c[0] = dc[by * 4 + bx];
      bool any = false;
      for (int k = 0; k < 16; ++k) any |= c[k] != 0;
      if (!any) continue;
      int32_t d[16];
      scale4(c, qp, pp_.ws4[0], true, d);
      idct4_add(d, ly_ + by * 4 * ls_ + bx * 4, ls_);
    }
    return;
  }
  const int list = intra ? 0 : 3;
  if (mb_->t8x8) {
    for (int b8 = 0; b8 < 4; ++b8) {
      if (!((cbp_ >> b8) & 1)) continue;
      int32_t d[64];
      scale8(&luma_coef_[b8 * 64], qp, pp_.ws8[intra ? 0 : 1], d);
      idct8_add(d, ly_ + (b8 >> 1) * 8 * ls_ + (b8 & 1) * 8, ls_);
    }
  } else {
    for (int blk = 0; blk < 16; ++blk) {
      if (!((cbp_ >> (blk >> 2)) & 1)) continue;
      if (!mb_->nz[0][blk_raster(blk)]) continue;
      int32_t d[16];
      scale4(&luma_coef_[blk * 16], qp, pp_.ws4[list], false, d);
      idct4_add(d, ly_ + blk_y(blk) * 4 * ls_ + blk_x(blk) * 4, ls_);
    }
  }
}

void Dec::add_residual_chroma() {
  if (((cbp_ >> 4) & 3) == 0) return;
  const bool intra = mb_->intra;
  for (int c = 0; c < 2; ++c) {
    const int qpc = mb_->qpc[c];
    const uint8_t* ws = pp_.ws4[(intra ? 0 : 3) + 1 + c];
    // 2x2 DC transform (8.5.11)
    const int32_t* cd = chroma_dc_[c];
    int f[4] = {cd[0] + cd[1] + cd[2] + cd[3], cd[0] - cd[1] + cd[2] - cd[3], cd[0] + cd[1] - cd[2] - cd[3], cd[0] - cd[1] - cd[2] + cd[3]};
    const int ls0 = ws[0] * kNorm4[qpc % 6][0];
    int dcc[4];
    for (int k = 0; k < 4; ++k) dcc[k] = ((f[k] * ls0) << (qpc / 6)) >> 5;
    uint8_t* dst = c == 0 ? lu_ : lv_;
    for (int b = 0; b < 4; ++b) {
      int32_t cc[16];
      std::memcpy(cc, chroma_ac_[c][b], sizeof cc);
      cc[0] = dcc[b];
      bool any = false;
      for (int k = 0; k < 16; ++k) any |= cc[k] != 0;
      if (!any) continue;
      int32_t d[16];
      scale4(cc, qpc, ws, true, d);
      idct4_add(d, dst + (b >> 1) * 4 * lcs_ + (b & 1) * 4, lcs_);
    }
  }
}

void Dec::recon_inter_residual() {
  add_residual_luma();
  add_residual_chroma();
}

// I_PCM (7.3.5, 8.3.5)
void Dec::ipcm_samples() {
  if (cabac_) {
    // the arithmetic decoder stopped at the terminating bin; the samples follow at the next byte boundary
  }
  while (!br_.byte_aligned()) br_.bit();
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 16; ++x) ly_[y * ls_ + x] = static_cast<uint8_t>(br_.u(8));
  for (int c = 0; c < 2; ++c) {
    uint8_t* dst = c == 0 ? lu_ : lv_;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) dst[y * lcs_ + x] = static_cast<uint8_t>(br_.u(8));
  }
  if (cabac_) cab_.start(&br_);
  std::memset(mb_->nz, 16, sizeof mb_->nz);
  mb_->cbf_dc = 7;
  mb_->cbp = 0x2F;
  cbp_ = 0x2F;
  mb_->qp = 0;
  last_qp_delta_ = 0;
}

