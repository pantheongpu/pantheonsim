// Part of h264_decode.cpp: the deblocking filter (8.7).

namespace {

// Filters one line of samples across an edge. q0 points at sample q0, `step` is the distance between samples
// across the edge (1 for a vertical edge, the stride for a horizontal one).
void filter_line(uint8_t* q0p, int step, int bs, int alpha, int beta, int tc0, bool chroma) {
  const int p0 = q0p[-step], p1 = q0p[-2 * step], q0 = q0p[0], q1 = q0p[step];
  if (std::abs(p0 - q0) >= alpha || std::abs(p1 - p0) >= beta || std::abs(q1 - q0) >= beta) return;
  if (bs < 4) {
    int tc;
    int ap = 0, aq = 0;
    if (!chroma) {
      const int p2 = q0p[-3 * step], q2 = q0p[2 * step];
      ap = std::abs(p2 - p0);
      aq = std::abs(q2 - q0);
      tc = tc0 + (ap < beta) + (aq < beta);
    } else {
      tc = tc0 + 1;
    }
    const int delta = clip3(-tc, tc, (((q0 - p0) << 2) + (p1 - q1) + 4) >> 3);
    q0p[-step] = clip1(p0 + delta);
    q0p[0] = clip1(q0 - delta);
    if (!chroma) {
      const int p2 = q0p[-3 * step], q2 = q0p[2 * step];
      if (ap < beta) q0p[-2 * step] = static_cast<uint8_t>(p1 + clip3(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1));
      if (aq < beta) q0p[step] = static_cast<uint8_t>(q1 + clip3(-tc0, tc0, (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1));
    }
    return;
  }
  if (!chroma) {
    const int p2 = q0p[-3 * step], p3 = q0p[-4 * step], q2 = q0p[2 * step], q3 = q0p[3 * step];
    const int ap = std::abs(p2 - p0), aq = std::abs(q2 - q0);
    const bool small = std::abs(p0 - q0) < ((alpha >> 2) + 2);
    if (ap < beta && small) {
      q0p[-step] = static_cast<uint8_t>((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
      q0p[-2 * step] = static_cast<uint8_t>((p2 + p1 + p0 + q0 + 2) >> 2);
      q0p[-3 * step] = static_cast<uint8_t>((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
    } else {
      q0p[-step] = static_cast<uint8_t>((2 * p1 + p0 + q1 + 2) >> 2);
    }
    if (aq < beta && small) {
      q0p[0] = static_cast<uint8_t>((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
      q0p[step] = static_cast<uint8_t>((p0 + q0 + q1 + q2 + 2) >> 2);
      q0p[2 * step] = static_cast<uint8_t>((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
    } else {
      q0p[0] = static_cast<uint8_t>((2 * q1 + q0 + p1 + 2) >> 2);
    }
  } else {
    q0p[-step] = static_cast<uint8_t>((2 * p1 + p0 + q1 + 2) >> 2);
    q0p[0] = static_cast<uint8_t>((2 * q1 + q0 + p1 + 2) >> 2);
  }
}

}  // namespace

// Whether the 4x4 block b of a macroblock has non-zero coefficients (for transform size 8, the 8x8 block containing it).
static bool block_has_coefficients(const MbInfo& m, int b) {
  if (m.t8x8) {
    const int x0 = (b & 3) & ~1, y0 = (b >> 2) & ~1;
    return (m.nz[0][y0 * 4 + x0] | m.nz[0][y0 * 4 + x0 + 1] | m.nz[0][(y0 + 1) * 4 + x0] | m.nz[0][(y0 + 1) * 4 + x0 + 1]) != 0;
  }
  return m.nz[0][b] != 0;
}

// The motion part of the boundary strength (8.7.2.1, bS 1 or 0) between two blocks of macroblocks of the same kind (both frame or both
// field). `lim_y` is the vertical motion vector difference that counts, in the units of the macroblocks' vectors.
static int motion_strength(const MbInfo& p, int bp, const MbInfo& q, int bq, int lim_y) {
  const int b8p = ((bp >> 2) >> 1) * 2 + ((bp & 3) >> 1), b8q = ((bq >> 2) >> 1) * 2 + ((bq & 3) >> 1);
  const int rp0 = p.ref_idx[0][b8p] >= 0 ? p.ref_id[0][b8p] : -1, rp1 = p.ref_idx[1][b8p] >= 0 ? p.ref_id[1][b8p] : -1;
  const int rq0 = q.ref_idx[0][b8q] >= 0 ? q.ref_id[0][b8q] : -1, rq1 = q.ref_idx[1][b8q] >= 0 ? q.ref_id[1][b8q] : -1;
  const int np = (p.ref_idx[0][b8p] >= 0) + (p.ref_idx[1][b8p] >= 0), nq = (q.ref_idx[0][b8q] >= 0) + (q.ref_idx[1][b8q] >= 0);
  if (np != nq) return 1;
  auto far = [&](const int16_t* a, const int16_t* b) { return std::abs(a[0] - b[0]) >= 4 || std::abs(a[1] - b[1]) >= lim_y; };
  if (np == 1) {
    const int lp = p.ref_idx[0][b8p] >= 0 ? 0 : 1, lq = q.ref_idx[0][b8q] >= 0 ? 0 : 1;
    const int refp = lp == 0 ? rp0 : rp1, refq = lq == 0 ? rq0 : rq1;
    if (refp != refq) return 1;
    return far(p.mv[lp][bp], q.mv[lq][bq]) ? 1 : 0;
  }
  if (np == 0) return 0;
  if (!((rp0 == rq0 && rp1 == rq1) || (rp0 == rq1 && rp1 == rq0))) return 1;
  if (rp0 != rp1) {
    if (rp0 == rq0) return (far(p.mv[0][bp], q.mv[0][bq]) || far(p.mv[1][bp], q.mv[1][bq])) ? 1 : 0;
    return (far(p.mv[0][bp], q.mv[1][bq]) || far(p.mv[1][bp], q.mv[0][bq])) ? 1 : 0;
  }
  const bool straight = far(p.mv[0][bp], q.mv[0][bq]) || far(p.mv[1][bp], q.mv[1][bq]);
  const bool cross = far(p.mv[0][bp], q.mv[1][bq]) || far(p.mv[1][bp], q.mv[0][bq]);
  return (straight && cross) ? 1 : 0;
}

// Boundary strength between the 4x4 block bp of macroblock p and the block bq of macroblock q (8.7.2.1) for
// non-MBAFF pictures.
static int boundary_strength(const MbInfo& p, int bp, const MbInfo& q, int bq, bool mb_edge, bool vertical, bool field_pic) {
  if (p.intra || q.intra) {
    if (mb_edge && (!field_pic || vertical)) return 4;
    return 3;
  }
  if (block_has_coefficients(p, bp) || block_has_coefficients(q, bq)) return 2;
  return motion_strength(p, bp, q, bq, field_pic ? 2 : 4);
}

// The same for macroblock-adaptive frames, where field and frame macroblocks meet. `mixed` is mixedModeEdgeFlag: the two
// macroblocks are in different pairs, one a field pair and one a frame pair.
static int boundary_strength_mbaff(const MbInfo& p, int bp, const MbInfo& q, int bq, bool mb_edge, bool vertical, bool mixed) {
  if (p.intra || q.intra) {
    if (mb_edge && ((!p.field && !q.field) || vertical)) return 4;
    return 3;
  }
  if (block_has_coefficients(p, bp) || block_has_coefficients(q, bq)) return 2;
  if (mixed) return 1;
  return motion_strength(p, bp, q, bq, q.field ? 2 : 4);
}

void Dec::deblock_picture() {
#ifdef VGPU_H264_DEBUG
  if (std::getenv("VGPU_H264_NODEBLOCK")) return;
#endif
  for (int a = 0; a < nmb_; ++a) {
    if (pd_->mb[a].slice < 0) continue;
    deblock_mb(a);
  }
}

void Dec::deblock_mb(int addr) {
  const MbInfo& q = pd_->mb[addr];
  const SliceParams& sp = pd_->slices[q.slice];
  if (sp.disable_idc == 1) return;
  if (mbaff_) {
    deblock_mb_mbaff(addr);
    return;
  }
  const int mx = addr % W_, my = addr / W_;
  const bool left_edge = mx > 0 && !(sp.disable_idc == 2 && pd_->mb[addr - 1].slice != q.slice);
  const bool top_edge = my > 0 && !(sp.disable_idc == 2 && pd_->mb[addr - W_].slice != q.slice);
  const int ls = py_.stride, cs = pu_.stride;
  uint8_t* ly = py_.p + static_cast<ptrdiff_t>(my * 16) * ls + mx * 16;
  uint8_t* lu = pu_.p + static_cast<ptrdiff_t>(my * 8) * cs + mx * 8;
  uint8_t* lv = pv_.p + static_cast<ptrdiff_t>(my * 8) * cs + mx * 8;
  const bool t8 = q.t8x8;
  // bS of each edge segment: [direction][edge 0..3][segment 0..3]
  int bs[2][4][4];
  for (int dir = 0; dir < 2; ++dir)
    for (int e = 0; e < 4; ++e)
      for (int s = 0; s < 4; ++s) {
        bs[dir][e][s] = 0;
        if (e == 0) {
          if (dir == 0 && !left_edge) continue;
          if (dir == 1 && !top_edge) continue;
        } else {
          if (t8 && (e & 1)) continue;
        }
        const MbInfo* pm = &q;
        int bq, bp;
        if (dir == 0) {   // vertical edge e, segment s = row of 4x4 blocks
          bq = s * 4 + e;
          if (e == 0) {
            pm = &pd_->mb[addr - 1];
            bp = s * 4 + 3;
          } else {
            bp = bq - 1;
          }
        } else {
          bq = e * 4 + s;
          if (e == 0) {
            pm = &pd_->mb[addr - W_];
            bp = 12 + s;
          } else {
            bp = bq - 4;
          }
        }
        bs[dir][e][s] = boundary_strength(*pm, bp, q, bq, e == 0, dir == 0, field_pic_);
      }
  auto thresholds = [&](int qp_p, int qp_q, int* alpha, int* beta, int* indexa) {
    const int qpav = (qp_p + qp_q + 1) >> 1;
    const int ia = clip3(0, 51, qpav + sp.alpha_off), ib = clip3(0, 51, qpav + sp.beta_off);
    *alpha = kAlpha[ia];
    *beta = kBeta[ib];
    *indexa = ia;
  };
  // luma
  for (int dir = 0; dir < 2; ++dir)
    for (int e = 0; e < 4; ++e) {
      const MbInfo* pm = e == 0 ? &pd_->mb[addr - (dir == 0 ? 1 : W_)] : &q;
      if (e == 0 && ((dir == 0 && !left_edge) || (dir == 1 && !top_edge))) continue;
      int alpha, beta, ia;
      thresholds(pm->qp, q.qp, &alpha, &beta, &ia);
      if (alpha == 0 || beta == 0) continue;
      for (int k = 0; k < 16; ++k) {
        const int b = bs[dir][e][k >> 2];
        if (!b) continue;
        const int tc0 = b < 4 ? kTc0[b - 1][ia] : 0;
        uint8_t* qp0 = dir == 0 ? ly + k * ls + e * 4 : ly + (e * 4) * ls + k;
        filter_line(qp0, dir == 0 ? 1 : ls, b, alpha, beta, tc0, false);
      }
    }
  // chroma (4:2:0): edges 0 and 2 of the luma edges, 8 samples long
  for (int c = 0; c < 2; ++c) {
    uint8_t* base = c == 0 ? lu : lv;
    for (int dir = 0; dir < 2; ++dir)
      for (int ce = 0; ce < 2; ++ce) {
        const int e = ce * 2;
        if (e == 0 && ((dir == 0 && !left_edge) || (dir == 1 && !top_edge))) continue;
        const MbInfo* pm = e == 0 ? &pd_->mb[addr - (dir == 0 ? 1 : W_)] : &q;
        int alpha, beta, ia;
        thresholds(pm->qpc[c], q.qpc[c], &alpha, &beta, &ia);
        if (alpha == 0 || beta == 0) continue;
        for (int k = 0; k < 8; ++k) {
          const int b = bs[dir][e][k >> 1];
          if (!b) continue;
          const int tc0 = b < 4 ? kTc0[b - 1][ia] : 0;
          uint8_t* qp0 = dir == 0 ? base + k * cs + ce * 4 : base + (ce * 4) * cs + k;
          filter_line(qp0, dir == 0 ? 1 : cs, b, alpha, beta, tc0, true);
        }
      }
  }
}

// Deblocking of a macroblock of a macroblock-adaptive frame (8.7 with MbaffFrameFlag = 1). Field macroblocks filter along their
// own rows (every other row of the frame); the left edge meets a pair of the other kind row by row, and the top edge of a frame
// macroblock under a field pair is filtered twice, once per field.
void Dec::deblock_mb_mbaff(int addr) {
  const MbInfo& q = pd_->mb[addr];
  const SliceParams& sp = pd_->slices[q.slice];
  const int pair = addr >> 1, px = pair % W_, py = pair / W_, bot = addr & 1;
  const bool fld = q.field;
  const int dy = fld ? 2 : 1;
  const int ls = py_.stride, cs = pu_.stride;
  const int yI = fld ? py * 32 + bot : py * 32 + bot * 16;
  const int yC = (yI + 1) / 2;
  uint8_t* ly = py_.p + static_cast<ptrdiff_t>(yI) * ls + px * 16;
  uint8_t* cb[2] = {pu_.p + static_cast<ptrdiff_t>(yC) * cs + px * 8, pv_.p + static_cast<ptrdiff_t>(yC) * cs + px * 8};
  const ptrdiff_t lrow = static_cast<ptrdiff_t>(dy) * ls, crow = static_cast<ptrdiff_t>(dy) * cs;   // distance between the rows of this macroblock
  const bool t8 = q.t8x8;
  auto thresholds = [&](int qp_p, int qp_q, int* alpha, int* beta, int* indexa) {
    const int qpav = (qp_p + qp_q + 1) >> 1;
    const int ia = clip3(0, 51, qpav + sp.alpha_off), ib = clip3(0, 51, qpav + sp.beta_off);
    *alpha = kAlpha[ia];
    *beta = kBeta[ib];
    *indexa = ia;
  };
  auto line = [&](uint8_t* q0, ptrdiff_t step, int bs, int qp_p, int qp_q, bool chroma) {
    if (!bs) return;
    int alpha, beta, ia;
    thresholds(qp_p, qp_q, &alpha, &beta, &ia);
    if (alpha == 0 || beta == 0) return;
    filter_line(q0, static_cast<int>(step), bs, alpha, beta, bs < 4 ? kTc0[bs - 1][ia] : 0, chroma);
  };
  const int above_top = 2 * (pair - W_);   // the pair above (its top macroblock)
  const int left_top = 2 * (pair - 1);
  // which edges are filtered (8.7, steps 2c and 2d)
  bool left_edge = px > 0;
  if (left_edge && sp.disable_idc == 2 && pd_->mb[left_top].slice != q.slice) left_edge = false;
  bool top_edge = py > 0 || (!fld && bot == 1);
  if (top_edge && sp.disable_idc == 2 && !(!fld && bot == 1) && pd_->mb[above_top].slice != q.slice) top_edge = false;

  // ---- the left edge: row k of this macroblock meets the row of the left pair at the same height
  const MbInfo* left_mb[16];
  int left_row[16];
  int cleft_idx[8];   // the left macroblock (0 top, 1 bottom) of each chroma row
  bool left_field = false;
  if (left_edge) {
    left_field = pd_->mb[left_top].field;
    for (int k = 0; k < 16; ++k) {
      const int rel = fld ? bot + 2 * k : bot * 16 + k;
      const int m = left_field ? (rel & 1) : (rel >= 16 ? 1 : 0);
      left_mb[k] = &pd_->mb[left_top + m];
      left_row[k] = left_field ? rel >> 1 : rel & 15;
    }
    for (int k = 0; k < 8; ++k) {
      const int rel = fld ? bot + 2 * k : bot * 8 + k;
      cleft_idx[k] = left_field ? (rel & 1) : (rel >= 8 ? 1 : 0);
    }
  }
  const bool mixed_left = left_edge && left_field != fld;

  // ---- vertical luma edges
  int bsv[4][16] = {};
  for (int e = 0; e < 4; ++e) {
    if (e == 0 && !left_edge) continue;
    if (e > 0 && t8 && (e & 1)) continue;
    for (int k = 0; k < 16; ++k) {
      int bs, qp_p;
      if (e == 0) {
        bs = boundary_strength_mbaff(*left_mb[k], (left_row[k] >> 2) * 4 + 3, q, (k >> 2) * 4, true, true, mixed_left);
        qp_p = left_mb[k]->qp;
      } else {
        bs = boundary_strength_mbaff(q, (k >> 2) * 4 + e - 1, q, (k >> 2) * 4 + e, false, true, false);
        qp_p = q.qp;
      }
      bsv[e][k] = bs;
      line(ly + k * lrow + 4 * e, 1, bs, qp_p, q.qp, false);
    }
  }

  // ---- horizontal luma edges
  // The top edge: one pass, or two (the field passes of a frame macroblock under a field pair).
  struct TopPass {
    const MbInfo* p;
    ptrdiff_t q0_row;     // offset of the q0 row from the first row of the macroblock, in picture rows
    ptrdiff_t step;       // picture rows between successive samples across the edge
    bool mixed;
    int bs[4];
  } passes[2];
  int npass = 0;
  if (top_edge) {
    if (!fld && bot == 0 && py > 0 && pd_->mb[above_top].field) {
      for (int pass = 0; pass < 2; ++pass) passes[npass++] = {&pd_->mb[above_top + pass], pass, 2, true, {0, 0, 0, 0}};
    } else if (fld) {
      const MbInfo& ab = pd_->mb[above_top];
      passes[npass++] = {ab.field ? &pd_->mb[above_top + bot] : &pd_->mb[above_top + 1], 0, 2, !ab.field, {0, 0, 0, 0}};
    } else if (bot == 1) {
      passes[npass++] = {&pd_->mb[addr - 1], 0, 1, false, {0, 0, 0, 0}};
    } else {
      passes[npass++] = {&pd_->mb[above_top + 1], 0, 1, false, {0, 0, 0, 0}};
    }
    for (int i = 0; i < npass; ++i) {
      TopPass& tp = passes[i];
      for (int xb = 0; xb < 4; ++xb) tp.bs[xb] = boundary_strength_mbaff(*tp.p, 12 + xb, q, xb, true, false, tp.mixed);
      for (int x = 0; x < 16; ++x) line(ly + tp.q0_row * ls + x, tp.step * ls, tp.bs[x >> 2], tp.p->qp, q.qp, false);
    }
  }
  int bsh[4][4] = {};
  for (int e = 1; e < 4; ++e) {
    if (t8 && (e & 1)) continue;
    for (int xb = 0; xb < 4; ++xb) bsh[e][xb] = boundary_strength_mbaff(q, (e - 1) * 4 + xb, q, e * 4 + xb, false, false, false);
    for (int x = 0; x < 16; ++x) line(ly + 4 * e * lrow + x, lrow, bsh[e][x >> 2], q.qp, q.qp, false);
  }

  // ---- chroma (4:2:0): the luma edges 0 and 2, the strengths of the luma samples at twice the chroma coordinates in the same field
  for (int c = 0; c < 2; ++c) {
    for (int ce = 0; ce < 2; ++ce) {
      if (ce == 0 && !left_edge) continue;
      for (int kc = 0; kc < 8; ++kc) {
        const int lk = fld ? 2 * kc : 4 * (kc >> 1) + (kc & 1);
        const int qp_p = ce == 0 ? pd_->mb[left_top + cleft_idx[kc]].qpc[c] : q.qpc[c];
        line(cb[c] + kc * crow + 4 * ce, 1, bsv[2 * ce][lk], qp_p, q.qpc[c], true);
      }
    }
    for (int i = 0; i < npass; ++i) {
      const TopPass& tp = passes[i];
      for (int xc = 0; xc < 8; ++xc) line(cb[c] + tp.q0_row * cs + xc, tp.step * cs, tp.bs[xc >> 1], tp.p->qpc[c], q.qpc[c], true);
    }
    for (int xc = 0; xc < 8; ++xc) line(cb[c] + 4 * crow + xc, crow, bsh[2][xc >> 1], q.qpc[c], q.qpc[c], true);
  }
}
