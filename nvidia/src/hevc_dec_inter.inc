// Part of hevc_decode.cpp: motion vector derivation (8.5.3.2: merge mode, AMVP, temporal prediction) and inter sample prediction
// (8.5.3.3: interpolation, weighted prediction).

namespace {

// Motion equality as 8.5.3.2.3 compares candidates: the same vectors and the same reference indices.
bool same_motion(const MvField& a, const MvField& b) {
  for (int l = 0; l < 2; ++l) {
    if (a.ref_idx[l] != b.ref_idx[l]) return false;
    if (a.ref_idx[l] >= 0 && (a.mv[l][0] != b.mv[l][0] || a.mv[l][1] != b.mv[l][1])) return false;
  }
  return true;
}

int clip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }

// 8-179 .. 8-181: motion vector scaling by the ratio of picture order count distances.
void scale_mv(int16_t mv[2], int td_in, int tb_in) {
  const int td = clip3(-128, 127, td_in), tb = clip3(-128, 127, tb_in);
  if (td == 0) return;
  const int tx = (16384 + (std::abs(td) >> 1)) / td;
  const int dsf = clip3(-4096, 4095, (tb * tx + 32) >> 6);
  for (int c = 0; c < 2; ++c) {
    const int p = dsf * mv[c];
    const int s = p < 0 ? -1 : (p > 0 ? 1 : 0);
    mv[c] = static_cast<int16_t>(clip3(-32768, 32767, s * ((std::abs(p) + 127) >> 8)));
  }
}

}  // namespace

void Dec::store_motion(int xpb, int ypb, int npbw, int npbh, const PuMotion& m) {
  MvField f;
  std::memset(&f, 0, sizeof f);
  for (int l = 0; l < 2; ++l) {
    f.ref_idx[l] = m.ref_idx[l];
    if (m.ref_idx[l] >= 0) {
      f.mv[l][0] = m.mv[l][0];
      f.mv[l][1] = m.mv[l][1];
    }
  }
  for (int y = ypb; y < ypb + npbh; y += 4)
    for (int x = xpb; x < xpb + npbw; x += 4) mvf_[static_cast<size_t>(idx4(x, y))] = f;
}

// 8.5.3.2.9: the collocated motion vector of block `c` for the target reference index; false if unavailable.
bool Dec::col_mv(const ColMv& c, int ref_idx, int lx, int16_t mv[2]) {
  if (!(c.flags & 3)) return false;
  int list_col;
  if (!(c.flags & 1)) list_col = 1;
  else if (!(c.flags & 2)) list_col = 0;
  else list_col = S->no_backward_pred ? lx : (S->h.collocated_from_l0 ? 1 : 0);
  const bool col_lt = (c.flags >> (2 + list_col)) & 1;
  const RefPic& tgt = S->ref[lx][ref_idx];
  if (tgt.long_term != col_lt) return false;
  mv[0] = c.mv[list_col][0];
  mv[1] = c.mv[list_col][1];
  const int col_poc_diff = S->col_poc - c.ref_poc[list_col];
  const int curr_poc_diff = P.curr_poc - tgt.poc;
  if (tgt.long_term || col_poc_diff == curr_poc_diff || col_poc_diff == 0) return true;
  scale_mv(mv, col_poc_diff, curr_poc_diff);
  return true;
}

// 8.5.3.2.8
bool Dec::temporal_mv(int xpb, int ypb, int npbw, int npbh, int ref_idx, int lx, int16_t mv[2]) {
  if (!S->h.temporal_mvp || !S->col_pic) return false;
  const Frame* cp = S->col_pic;
  if (cp->col.empty()) return false;
  const int x_br = xpb + npbw, y_br = ypb + npbh;
  if ((ypb >> log2_ctb) == (y_br >> log2_ctb) && y_br < H && x_br < W) {
    const ColMv& c = cp->col[static_cast<size_t>((y_br >> 4) * cp->col_w + (x_br >> 4))];
    if (col_mv(c, ref_idx, lx, mv)) return true;
  }
  const int x_ctr = xpb + (npbw >> 1), y_ctr = ypb + (npbh >> 1);
  const ColMv& c = cp->col[static_cast<size_t>((y_ctr >> 4) * cp->col_w + (x_ctr >> 4))];
  return col_mv(c, ref_idx, lx, mv);
}

// 8.5.3.2.2 to 8.5.3.2.5
void Dec::derive_merge(int xcb, int ycb, int ncbs, int xpb, int ypb, int npbw, int npbh, int part_idx, int merge_idx, PuMotion* out) {
  const int orig_w = npbw, orig_h = npbh;
  const int par = P.log2_parallel_merge_level;
  int part_mode = part_mode_;
  if (par > 2 && ncbs == 8) {
    xpb = xcb;
    ypb = ycb;
    npbw = npbh = ncbs;
    part_idx = 0;
    part_mode = PART_2Nx2N;
  }
  MvField cand[6];
  int num = 0;
  const int max_cand = S->h.max_num_merge_cand;
  // spatial candidates (8.5.3.2.3)
  struct Nb {
    int x, y;
    bool avail;
    MvField f;
  } a1{xpb - 1, ypb + npbh - 1, false, {}}, b1{xpb + npbw - 1, ypb - 1, false, {}}, b0{xpb + npbw, ypb - 1, false, {}}, a0{xpb - 1, ypb + npbh, false, {}},
      b2{xpb - 1, ypb - 1, false, {}};
  auto load = [&](Nb& nb) {
    nb.avail = avail_pb(xcb, ycb, ncbs, xpb, ypb, npbw, npbh, part_idx, nb.x, nb.y);
    if (nb.avail && (xpb >> par) == (nb.x >> par) && (ypb >> par) == (nb.y >> par)) nb.avail = false;
    if (nb.avail) nb.f = mvf_[static_cast<size_t>(idx4(nb.x, nb.y))];
  };
  load(a1);
  if (part_idx == 1 && (part_mode == PART_Nx2N || part_mode == PART_nLx2N || part_mode == PART_nRx2N)) a1.avail = false;
  load(b1);
  if (part_idx == 1 && (part_mode == PART_2NxN || part_mode == PART_2NxnU || part_mode == PART_2NxnD)) b1.avail = false;
  load(b0);
  load(a0);
  load(b2);
  bool flag_a1 = a1.avail, flag_b1 = b1.avail && !(a1.avail && same_motion(a1.f, b1.f));
  bool flag_b0 = b0.avail && !(b1.avail && same_motion(b1.f, b0.f));
  bool flag_a0 = a0.avail && !(a1.avail && same_motion(a1.f, a0.f));
  bool flag_b2 = b2.avail && !(a1.avail && same_motion(a1.f, b2.f)) && !(b1.avail && same_motion(b1.f, b2.f)) && (flag_a0 + flag_a1 + flag_b0 + flag_b1 != 4);
  if (flag_a1) cand[num++] = a1.f;
  if (flag_b1) cand[num++] = b1.f;
  if (flag_b0) cand[num++] = b0.f;
  if (flag_a0) cand[num++] = a0.f;
  if (flag_b2) cand[num++] = b2.f;
  // temporal candidate
  if (num < max_cand) {
    int16_t mv0[2] = {0, 0}, mv1[2] = {0, 0};
    const bool av0 = temporal_mv(xpb, ypb, npbw, npbh, 0, 0, mv0);
    bool av1 = false;
    if (S->h.slice_type == kSliceB) av1 = temporal_mv(xpb, ypb, npbw, npbh, 0, 1, mv1);
    if (av0 || av1) {
      MvField f;
      std::memset(&f, 0, sizeof f);
      f.ref_idx[0] = av0 ? 0 : -1;
      f.ref_idx[1] = av1 ? 0 : -1;
      if (av0) {
        f.mv[0][0] = mv0[0];
        f.mv[0][1] = mv0[1];
      }
      if (av1) {
        f.mv[1][0] = mv1[0];
        f.mv[1][1] = mv1[1];
      }
      cand[num++] = f;
    }
  }
  const int num_orig = num;
  // combined bi-predictive candidates (8.5.3.2.4)
  if (S->h.slice_type == kSliceB && num_orig > 1 && num_orig < max_cand) {
    static const int kL0[12] = {0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3}, kL1[12] = {1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2};
    int comb_idx = 0;
    for (;;) {
      const MvField& l0 = cand[kL0[comb_idx]];
      const MvField& l1 = cand[kL1[comb_idx]];
      if (l0.ref_idx[0] >= 0 && l1.ref_idx[1] >= 0 &&
          (S->ref[0][l0.ref_idx[0]].poc != S->ref[1][l1.ref_idx[1]].poc || l0.mv[0][0] != l1.mv[1][0] || l0.mv[0][1] != l1.mv[1][1])) {
        MvField f;
        std::memset(&f, 0, sizeof f);
        f.ref_idx[0] = l0.ref_idx[0];
        f.ref_idx[1] = l1.ref_idx[1];
        f.mv[0][0] = l0.mv[0][0];
        f.mv[0][1] = l0.mv[0][1];
        f.mv[1][0] = l1.mv[1][0];
        f.mv[1][1] = l1.mv[1][1];
        cand[num++] = f;
      }
      ++comb_idx;
      if (comb_idx == num_orig * (num_orig - 1) || num == max_cand) break;
    }
  }
  // zero candidates (8.5.3.2.5)
  if (num < max_cand) {
    const int num_ref = S->h.slice_type == kSliceP ? S->h.num_ref_idx[0] : std::min(S->h.num_ref_idx[0], S->h.num_ref_idx[1]);
    int zero_idx = 0;
    while (num < max_cand) {
      MvField f;
      std::memset(&f, 0, sizeof f);
      const int r = zero_idx < num_ref ? zero_idx : 0;
      f.ref_idx[0] = static_cast<int8_t>(r);
      f.ref_idx[1] = S->h.slice_type == kSliceB ? static_cast<int8_t>(r) : static_cast<int8_t>(-1);
      cand[num++] = f;
      ++zero_idx;
    }
  }
  if (merge_idx >= num) merge_idx = num - 1;
  MvField sel = cand[merge_idx];
  if (sel.ref_idx[0] >= 0 && sel.ref_idx[1] >= 0 && orig_w + orig_h == 12) {
    sel.ref_idx[1] = -1;
    sel.mv[1][0] = sel.mv[1][1] = 0;
  }
  for (int l = 0; l < 2; ++l) {
    out->ref_idx[l] = sel.ref_idx[l];
    out->mv[l][0] = sel.ref_idx[l] >= 0 ? sel.mv[l][0] : 0;
    out->mv[l][1] = sel.ref_idx[l] >= 0 ? sel.mv[l][1] : 0;
  }
  // refIdx beyond the active lists cannot happen for a conforming stream; keep decoding anyway
  for (int l = 0; l < 2; ++l)
    if (out->ref_idx[l] >= S->h.num_ref_idx[l]) out->ref_idx[l] = 0;
}

// 8.5.3.2.6 / 8.5.3.2.7
void Dec::derive_mvp(int xcb, int ycb, int ncbs, int xpb, int ypb, int npbw, int npbh, int part_idx, int ref_idx, int lx, int mvp_flag, int16_t mvp[2]) {
  const int ly = 1 - lx;
  const RefPic& target = S->ref[lx][ref_idx];
  bool avail_flag_a = false, avail_flag_b = false;
  int16_t mv_a[2] = {0, 0}, mv_b[2] = {0, 0};
  // candidate A: A0 = (xPb - 1, yPb + nPbH), A1 = (xPb - 1, yPb + nPbH - 1)
  const int ax[2] = {xpb - 1, xpb - 1}, ay[2] = {ypb + npbh, ypb + npbh - 1};
  bool av_a[2];
  MvField fa[2];
  for (int k = 0; k < 2; ++k) {
    av_a[k] = avail_pb(xcb, ycb, ncbs, xpb, ypb, npbw, npbh, part_idx, ax[k], ay[k]);
    if (av_a[k]) fa[k] = mvf_[static_cast<size_t>(idx4(ax[k], ay[k]))];
  }
  const bool is_scaled = av_a[0] || av_a[1];
  for (int k = 0; k < 2 && !avail_flag_a; ++k) {
    if (!av_a[k]) continue;
    if (fa[k].ref_idx[lx] >= 0 && S->ref[lx][fa[k].ref_idx[lx]].poc == target.poc) {
      avail_flag_a = true;
      mv_a[0] = fa[k].mv[lx][0];
      mv_a[1] = fa[k].mv[lx][1];
    } else if (fa[k].ref_idx[ly] >= 0 && S->ref[ly][fa[k].ref_idx[ly]].poc == target.poc) {
      avail_flag_a = true;
      mv_a[0] = fa[k].mv[ly][0];
      mv_a[1] = fa[k].mv[ly][1];
    }
  }
  for (int k = 0; k < 2 && !avail_flag_a; ++k) {
    if (!av_a[k]) continue;
    const RefPic* rp = nullptr;
    int l_used = -1;
    if (fa[k].ref_idx[lx] >= 0 && S->ref[lx][fa[k].ref_idx[lx]].long_term == target.long_term) {
      l_used = lx;
    } else if (fa[k].ref_idx[ly] >= 0 && S->ref[ly][fa[k].ref_idx[ly]].long_term == target.long_term) {
      l_used = ly;
    }
    if (l_used < 0) continue;
    rp = &S->ref[l_used][fa[k].ref_idx[l_used]];
    avail_flag_a = true;
    mv_a[0] = fa[k].mv[l_used][0];
    mv_a[1] = fa[k].mv[l_used][1];
    if (rp->poc != target.poc && !rp->long_term && !target.long_term) scale_mv(mv_a, P.curr_poc - rp->poc, P.curr_poc - target.poc);
  }
  // candidate B: B0 = (xPb + nPbW, yPb - 1), B1 = (xPb + nPbW - 1, yPb - 1), B2 = (xPb - 1, yPb - 1)
  const int bx[3] = {xpb + npbw, xpb + npbw - 1, xpb - 1}, by[3] = {ypb - 1, ypb - 1, ypb - 1};
  bool av_b[3];
  MvField fb[3];
  for (int k = 0; k < 3; ++k) {
    av_b[k] = avail_pb(xcb, ycb, ncbs, xpb, ypb, npbw, npbh, part_idx, bx[k], by[k]);
    if (av_b[k]) fb[k] = mvf_[static_cast<size_t>(idx4(bx[k], by[k]))];
  }
  for (int k = 0; k < 3 && !avail_flag_b; ++k) {
    if (!av_b[k]) continue;
    if (fb[k].ref_idx[lx] >= 0 && S->ref[lx][fb[k].ref_idx[lx]].poc == target.poc) {
      avail_flag_b = true;
      mv_b[0] = fb[k].mv[lx][0];
      mv_b[1] = fb[k].mv[lx][1];
    } else if (fb[k].ref_idx[ly] >= 0 && S->ref[ly][fb[k].ref_idx[ly]].poc == target.poc) {
      avail_flag_b = true;
      mv_b[0] = fb[k].mv[ly][0];
      mv_b[1] = fb[k].mv[ly][1];
    }
  }
  if (!is_scaled && avail_flag_b) {
    avail_flag_a = true;
    mv_a[0] = mv_b[0];
    mv_a[1] = mv_b[1];
  }
  if (!is_scaled) {
    avail_flag_b = false;
    for (int k = 0; k < 3 && !avail_flag_b; ++k) {
      if (!av_b[k]) continue;
      int l_used = -1;
      if (fb[k].ref_idx[lx] >= 0 && S->ref[lx][fb[k].ref_idx[lx]].long_term == target.long_term) l_used = lx;
      else if (fb[k].ref_idx[ly] >= 0 && S->ref[ly][fb[k].ref_idx[ly]].long_term == target.long_term) l_used = ly;
      if (l_used < 0) continue;
      const RefPic& rp = S->ref[l_used][fb[k].ref_idx[l_used]];
      avail_flag_b = true;
      mv_b[0] = fb[k].mv[l_used][0];
      mv_b[1] = fb[k].mv[l_used][1];
      if (rp.poc != target.poc && !rp.long_term && !target.long_term) scale_mv(mv_b, P.curr_poc - rp.poc, P.curr_poc - target.poc);
    }
  }
  // candidate list
  int16_t list[3][2];
  int cnt = 0;
  if (avail_flag_a) {
    list[cnt][0] = mv_a[0];
    list[cnt++][1] = mv_a[1];
    if (avail_flag_b && (mv_a[0] != mv_b[0] || mv_a[1] != mv_b[1])) {
      list[cnt][0] = mv_b[0];
      list[cnt++][1] = mv_b[1];
    }
  } else if (avail_flag_b) {
    list[cnt][0] = mv_b[0];
    list[cnt++][1] = mv_b[1];
  }
  if (cnt < 2) {
    int16_t t[2];
    if (temporal_mv(xpb, ypb, npbw, npbh, ref_idx, lx, t)) {
      list[cnt][0] = t[0];
      list[cnt++][1] = t[1];
    }
  }
  while (cnt < 2) {
    list[cnt][0] = list[cnt][1] = 0;
    ++cnt;
  }
  mvp[0] = list[mvp_flag][0];
  mvp[1] = list[mvp_flag][1];
}

namespace {

const int8_t kLumaFilter[4][8] = {{0, 0, 0, 64, 0, 0, 0, 0}, {-1, 4, -10, 58, 17, -5, 1, 0}, {-1, 4, -11, 40, 40, -11, 4, -1}, {0, 1, -5, 17, 58, -10, 4, -1}};
const int8_t kChromaFilter[8][4] = {{0, 64, 0, 0}, {-2, 58, 10, -2}, {-4, 54, 16, -2}, {-6, 46, 28, -4}, {-4, 36, 36, -4}, {-4, 28, 46, -6}, {-2, 16, 54, -4}, {-2, 10, 58, -2}};

// Fractional sample interpolation of a w x h block into 14-bit intermediates (8.5.3.3.3); `taps` is 8 (luma) or 4 (chroma).
// (xi, yi) is the integer position of the first sample, fx / fy the fractions (quarter or eighth sample).
void interpolate(const uint16_t* plane, int stride, int pw, int ph, int xi, int yi, int fx, int fy, int w, int h, int bd, int taps, int16_t* out, int out_stride) {
  const int before = taps / 2 - 1;   // samples taken before the current one: 3 (luma) or 1 (chroma)
  const int shift1 = std::min(4, bd - 8), shift3 = std::max(2, 14 - bd);
  // gather the source window with clamped coordinates
  const int ww = w + taps - 1, wh = h + taps - 1;
  static thread_local std::vector<int32_t> win;
  win.resize(static_cast<size_t>(ww) * static_cast<size_t>(wh));
  for (int y = 0; y < wh; ++y) {
    const int sy = std::max(0, std::min(ph - 1, yi - before + y));
    const uint16_t* row = plane + static_cast<size_t>(sy) * static_cast<size_t>(stride);
    int32_t* wr = &win[static_cast<size_t>(y) * static_cast<size_t>(ww)];
    for (int x = 0; x < ww; ++x) wr[x] = row[std::max(0, std::min(pw - 1, xi - before + x))];
  }
  auto coef = [&](int frac, int i) { return taps == 8 ? static_cast<int>(kLumaFilter[frac][i]) : static_cast<int>(kChromaFilter[frac][i]); };
  if (fx == 0 && fy == 0) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) out[y * out_stride + x] = static_cast<int16_t>(win[static_cast<size_t>(y + before) * static_cast<size_t>(ww) + static_cast<size_t>(x + before)] << shift3);
  } else if (fy == 0) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int s = 0;
        for (int i = 0; i < taps; ++i) s += coef(fx, i) * win[static_cast<size_t>(y + before) * static_cast<size_t>(ww) + static_cast<size_t>(x + i)];
        out[y * out_stride + x] = static_cast<int16_t>(s >> shift1);
      }
  } else if (fx == 0) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int s = 0;
        for (int i = 0; i < taps; ++i) s += coef(fy, i) * win[static_cast<size_t>(y + i) * static_cast<size_t>(ww) + static_cast<size_t>(x + before)];
        out[y * out_stride + x] = static_cast<int16_t>(s >> shift1);
      }
  } else {
    static thread_local std::vector<int32_t> mid;
    mid.resize(static_cast<size_t>(w) * static_cast<size_t>(wh));
    for (int y = 0; y < wh; ++y)
      for (int x = 0; x < w; ++x) {
        int s = 0;
        for (int i = 0; i < taps; ++i) s += coef(fx, i) * win[static_cast<size_t>(y) * static_cast<size_t>(ww) + static_cast<size_t>(x + i)];
        mid[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = s >> shift1;
      }
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int s = 0;
        for (int i = 0; i < taps; ++i) s += coef(fy, i) * mid[static_cast<size_t>(y + i) * static_cast<size_t>(w) + static_cast<size_t>(x)];
        out[y * out_stride + x] = static_cast<int16_t>(s >> 6);
      }
  }
}

}  // namespace

void Dec::inter_predict(int xpb, int ypb, int npbw, int npbh, const PuMotion& m) {
  const SliceHeader& h = S->h;
  const bool weighted = (h.slice_type == kSliceP && P.weighted_pred) || (h.slice_type == kSliceB && P.weighted_bipred);
  static thread_local std::vector<int16_t> pred[2][3];
  for (int c = 0; c < 3; ++c) {
    const int sub = c == 0 ? 1 : 2;
    const int w = npbw / sub, ht = npbh / sub;
    const int x0 = xpb / sub, y0 = ypb / sub;
    const int bd = c == 0 ? bd_y : bd_c;
    std::vector<uint16_t>& dst = c == 0 ? F->y : (c == 1 ? F->u : F->v);
    const int stride = c == 0 ? F->stride_y : F->stride_c;
    const int pw = c == 0 ? W : CW, ph = c == 0 ? H : CH;
    for (int l = 0; l < 2; ++l) {
      if (!m.pred(l)) continue;
      pred[l][c].resize(static_cast<size_t>(w) * static_cast<size_t>(ht));
      const Frame* rf = S->ref[l][m.ref_idx[l]].frame;
      if (!rf) {
        const int16_t gray = static_cast<int16_t>((1 << (bd - 1)) << std::max(2, 14 - bd));
        std::fill(pred[l][c].begin(), pred[l][c].end(), gray);
        continue;
      }
      const std::vector<uint16_t>& src = c == 0 ? rf->y : (c == 1 ? rf->u : rf->v);
      const int sstride = c == 0 ? rf->stride_y : rf->stride_c;
      if (c == 0) {
        interpolate(src.data(), sstride, pw, ph, x0 + (m.mv[l][0] >> 2), y0 + (m.mv[l][1] >> 2), m.mv[l][0] & 3, m.mv[l][1] & 3, w, ht, bd, 8, pred[l][c].data(), w);
      } else {
        interpolate(src.data(), sstride, pw, ph, x0 + (m.mv[l][0] >> 3), y0 + (m.mv[l][1] >> 3), m.mv[l][0] & 7, m.mv[l][1] & 7, w, ht, bd, 4, pred[l][c].data(), w);
      }
    }
    const int maxv = (1 << bd) - 1;
    const int shift1 = std::max(2, 14 - bd);
    uint16_t* d = &dst[static_cast<size_t>(y0) * static_cast<size_t>(stride) + static_cast<size_t>(x0)];
    if (!weighted) {
      if (m.pred(0) && m.pred(1)) {
        const int shift2 = std::max(3, 15 - bd), off2 = 1 << (shift2 - 1);
        for (int y = 0; y < ht; ++y)
          for (int x = 0; x < w; ++x) d[y * stride + x] = static_cast<uint16_t>(clip3(0, maxv, (pred[0][c][static_cast<size_t>(y * w + x)] + pred[1][c][static_cast<size_t>(y * w + x)] + off2) >> shift2));
      } else {
        const int l = m.pred(0) ? 0 : 1;
        const int off1 = 1 << (shift1 - 1);
        for (int y = 0; y < ht; ++y)
          for (int x = 0; x < w; ++x) d[y * stride + x] = static_cast<uint16_t>(clip3(0, maxv, (pred[l][c][static_cast<size_t>(y * w + x)] + off1) >> shift1));
      }
    } else {
      const PredWeights& pw_t = h.pw;
      const int log2wd = (c == 0 ? pw_t.luma_log2_denom : pw_t.chroma_log2_denom) + shift1;
      const int off_shift = P.high_precision_offsets ? 0 : bd - 8;
      auto weight = [&](int l) { return c == 0 ? pw_t.luma_weight[l][m.ref_idx[l]] : pw_t.chroma_weight[l][m.ref_idx[l]][c - 1]; };
      auto offset = [&](int l) { return (c == 0 ? pw_t.luma_offset[l][m.ref_idx[l]] : pw_t.chroma_offset[l][m.ref_idx[l]][c - 1]) * (1 << off_shift); };
      if (m.pred(0) && m.pred(1)) {
        const int w0 = weight(0), w1 = weight(1), o0 = offset(0), o1 = offset(1);
        for (int y = 0; y < ht; ++y)
          for (int x = 0; x < w; ++x) {
            const int v = (pred[0][c][static_cast<size_t>(y * w + x)] * w0 + pred[1][c][static_cast<size_t>(y * w + x)] * w1 + ((o0 + o1 + 1) << log2wd)) >> (log2wd + 1);
            d[y * stride + x] = static_cast<uint16_t>(clip3(0, maxv, v));
          }
      } else {
        const int l = m.pred(0) ? 0 : 1;
        const int w0 = weight(l), o0 = offset(l);
        for (int y = 0; y < ht; ++y)
          for (int x = 0; x < w; ++x) {
            const int p = pred[l][c][static_cast<size_t>(y * w + x)];
            const int v = log2wd >= 1 ? ((p * w0 + (1 << (log2wd - 1))) >> log2wd) + o0 : p * w0 + o0;
            d[y * stride + x] = static_cast<uint16_t>(clip3(0, maxv, v));
          }
      }
    }
  }
}

// After the last slice: the motion field kept for later pictures' temporal prediction (one entry per 16x16 block).
void Dec::finish_col() {
  F->col_w = (W + 15) >> 4;
  F->col_h = (H + 15) >> 4;
  F->col.assign(static_cast<size_t>(F->col_w) * static_cast<size_t>(F->col_h), ColMv());
  for (int by = 0; by < F->col_h; ++by)
    for (int bx = 0; bx < F->col_w; ++bx) {
      const int x = bx << 4, y = by << 4;
      const size_t i = static_cast<size_t>(idx4(x, y));
      if (pred_mode_[i] != 1) continue;
      const int ci = (y >> log2_ctb) * wctb + (x >> log2_ctb);
      const int si = ctb_slice_[static_cast<size_t>(ci)];
      if (si < 0) continue;
      const SliceInfo& s = slices_[static_cast<size_t>(si)];
      const MvField& f = mvf_[i];
      ColMv& c = F->col[static_cast<size_t>(by * F->col_w + bx)];
      for (int l = 0; l < 2; ++l) {
        if (f.ref_idx[l] < 0 || f.ref_idx[l] >= 16) continue;
        c.flags |= static_cast<uint8_t>(1 << l);
        if (s.ref[l][f.ref_idx[l]].long_term) c.flags |= static_cast<uint8_t>(4 << l);
        c.mv[l][0] = f.mv[l][0];
        c.mv[l][1] = f.mv[l][1];
        c.ref_poc[l] = s.ref[l][f.ref_idx[l]].poc;
      }
    }
}
