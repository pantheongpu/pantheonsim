// Part of hevc_decode.cpp: the deblocking filter (8.7.2) and sample adaptive offset (8.7.3), applied to the whole picture after
// its last slice segment is decoded.

namespace {

// Table 8-12
const uint8_t kBetaTable[52] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64};
const uint8_t kTcTable[54] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24};

bool mv_far(const int16_t a[2], const int16_t b[2]) { return std::abs(a[0] - b[0]) >= 4 || std::abs(a[1] - b[1]) >= 4; }

}  // namespace

// 8.7.2.4: the boundary filtering strength between the blocks containing (xp, yp) and (xq, yq); `tu_edge` says the edge is also a
// transform block edge.
int Dec::bs_of(int xq, int yq, int xp, int yp, bool tu_edge) const {
  const size_t iq = static_cast<size_t>(idx4(xq, yq)), ip = static_cast<size_t>(idx4(xp, yp));
  if (pred_mode_[iq] == 2 || pred_mode_[ip] == 2) return 2;
  if (tu_edge && (tu_nz_[iq] || tu_nz_[ip])) return 1;
  const MvField& q = mvf_[iq];
  const MvField& p = mvf_[ip];
  const SliceInfo& sq = slices_[static_cast<size_t>(ctb_slice_[static_cast<size_t>((yq >> log2_ctb) * wctb + (xq >> log2_ctb))])];
  const SliceInfo& sp = slices_[static_cast<size_t>(ctb_slice_[static_cast<size_t>((yp >> log2_ctb) * wctb + (xp >> log2_ctb))])];
  int pr[2], qr[2];
  int np = 0, nq = 0;
  for (int l = 0; l < 2; ++l) {
    pr[l] = p.ref_idx[l] >= 0 ? sp.ref_slot[l][p.ref_idx[l]] : -1;
    qr[l] = q.ref_idx[l] >= 0 ? sq.ref_slot[l][q.ref_idx[l]] : -1;
    np += p.ref_idx[l] >= 0;
    nq += q.ref_idx[l] >= 0;
  }
  if (np != nq) return 1;
  if (np == 1) {
    const int lp = p.ref_idx[0] >= 0 ? 0 : 1, lq = q.ref_idx[0] >= 0 ? 0 : 1;
    if (pr[lp] != qr[lq]) return 1;
    return mv_far(p.mv[lp], q.mv[lq]) ? 1 : 0;
  }
  if (np == 0) return 0;
  if (!((pr[0] == qr[0] && pr[1] == qr[1]) || (pr[0] == qr[1] && pr[1] == qr[0]))) return 1;
  if (pr[0] != pr[1]) {
    if (pr[0] == qr[0]) return (mv_far(p.mv[0], q.mv[0]) || mv_far(p.mv[1], q.mv[1])) ? 1 : 0;
    return (mv_far(p.mv[0], q.mv[1]) || mv_far(p.mv[1], q.mv[0])) ? 1 : 0;
  }
  const bool c1 = mv_far(p.mv[0], q.mv[0]) || mv_far(p.mv[1], q.mv[1]);
  const bool c2 = mv_far(p.mv[0], q.mv[1]) || mv_far(p.mv[1], q.mv[0]);
  return (c1 && c2) ? 1 : 0;
}

void Dec::deblock_edges(bool vertical) {
  const int maxv_y = (1 << bd_y) - 1, maxv_c = (1 << bd_c) - 1;
  const uint8_t tu_bit = vertical ? 1 : 4, pu_bit = vertical ? 2 : 8;
  // luma: edges on the 8x8 grid, in 4-sample segments
  for (int y = 0; y < H; y += (vertical ? 4 : 8)) {
    for (int x = 0; x < W; x += (vertical ? 8 : 4)) {
      if ((vertical && x == 0) || (!vertical && y == 0)) continue;
      const uint8_t e = edge_[static_cast<size_t>(idx4(x, y))];
      if (!(e & (tu_bit | pu_bit))) continue;
      const int xp = vertical ? x - 1 : x, yp = vertical ? y : y - 1;
      const int cq = (y >> log2_ctb) * wctb + (x >> log2_ctb), cp = (yp >> log2_ctb) * wctb + (xp >> log2_ctb);
      const int sq = ctb_slice_[static_cast<size_t>(cq)], sp = ctb_slice_[static_cast<size_t>(cp)];
      if (sq < 0 || sp < 0) continue;
      const SliceInfo& SQ = slices_[static_cast<size_t>(sq)];
      if (SQ.h.deblocking_disabled) continue;
      if (slices_[static_cast<size_t>(sp)].addr_rs != SQ.addr_rs && !SQ.h.loop_filter_across_slices) continue;
      if (!P.loop_filter_across_tiles && tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(cq)])] != tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(cp)])]) continue;
      const int bs = bs_of(x, y, xp, yp, (e & tu_bit) != 0);
      if (bs == 0) continue;
      const int qp_q = qp_y_[static_cast<size_t>(idx4(x, y))], qp_p = qp_y_[static_cast<size_t>(idx4(xp, yp))];
      const bool no_p = (cu_flags_[static_cast<size_t>(idx4(xp, yp))] & kCuBypass) || ((cu_flags_[static_cast<size_t>(idx4(xp, yp))] & kCuPcm) && P.pcm_loop_filter_disabled);
      const bool no_q = (cu_flags_[static_cast<size_t>(idx4(x, y))] & kCuBypass) || ((cu_flags_[static_cast<size_t>(idx4(x, y))] & kCuPcm) && P.pcm_loop_filter_disabled);
      const int qpl = (qp_q + qp_p + 1) >> 1;
      const int beta = kBetaTable[clip3(0, 51, qpl + (SQ.h.beta_offset_div2 << 1))] * (1 << (bd_y - 8));
      const int tc = kTcTable[clip3(0, 53, qpl + 2 * (bs - 1) + (SQ.h.tc_offset_div2 << 1))] * (1 << (bd_y - 8));
      uint16_t* base = &F->y[static_cast<size_t>(y) * static_cast<size_t>(F->stride_y) + static_cast<size_t>(x)];
      // step along the edge (k) and across it (i)
      const ptrdiff_t across = vertical ? 1 : F->stride_y, along = vertical ? F->stride_y : 1;
      auto px = [&](int i, int k) -> int { return base[k * along - (i + 1) * across]; };
      auto qx = [&](int i, int k) -> int { return base[k * along + i * across]; };
      const int dp0 = std::abs(px(2, 0) - 2 * px(1, 0) + px(0, 0)), dp3 = std::abs(px(2, 3) - 2 * px(1, 3) + px(0, 3));
      const int dq0 = std::abs(qx(2, 0) - 2 * qx(1, 0) + qx(0, 0)), dq3 = std::abs(qx(2, 3) - 2 * qx(1, 3) + qx(0, 3));
      const int dpq0 = dp0 + dq0, dpq3 = dp3 + dq3, dp = dp0 + dp3, dq = dq0 + dq3, d = dpq0 + dpq3;
      if (d >= beta) continue;
      auto dsam = [&](int k, int dpq) {
        return dpq < (beta >> 2) && std::abs(px(3, k) - px(0, k)) + std::abs(qx(0, k) - qx(3, k)) < (beta >> 3) && std::abs(px(0, k) - qx(0, k)) < ((5 * tc + 1) >> 1);
      };
      const bool strong = dsam(0, 2 * dpq0) && dsam(3, 2 * dpq3);
      const bool dep = dp < ((beta + (beta >> 1)) >> 3), deq = dq < ((beta + (beta >> 1)) >> 3);
      for (int k = 0; k < 4; ++k) {
        uint16_t* b = base + k * along;
        const int p0 = b[-across], p1 = b[-2 * across], p2 = b[-3 * across], p3 = b[-4 * across];
        const int q0 = b[0], q1 = b[across], q2 = b[2 * across], q3 = b[3 * across];
        if (strong) {
          if (!no_p) {
            b[-across] = static_cast<uint16_t>(clip3(p0 - 2 * tc, p0 + 2 * tc, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3));
            b[-2 * across] = static_cast<uint16_t>(clip3(p1 - 2 * tc, p1 + 2 * tc, (p2 + p1 + p0 + q0 + 2) >> 2));
            b[-3 * across] = static_cast<uint16_t>(clip3(p2 - 2 * tc, p2 + 2 * tc, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3));
          }
          if (!no_q) {
            b[0] = static_cast<uint16_t>(clip3(q0 - 2 * tc, q0 + 2 * tc, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3));
            b[across] = static_cast<uint16_t>(clip3(q1 - 2 * tc, q1 + 2 * tc, (p0 + q0 + q1 + q2 + 2) >> 2));
            b[2 * across] = static_cast<uint16_t>(clip3(q2 - 2 * tc, q2 + 2 * tc, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3));
          }
        } else {
          int delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
          if (std::abs(delta) < tc * 10) {
            delta = clip3(-tc, tc, delta);
            if (!no_p) {
              b[-across] = static_cast<uint16_t>(clip3(0, maxv_y, p0 + delta));
              if (dep) {
                const int dlt = clip3(-(tc >> 1), tc >> 1, (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1);
                b[-2 * across] = static_cast<uint16_t>(clip3(0, maxv_y, p1 + dlt));
              }
            }
            if (!no_q) {
              b[0] = static_cast<uint16_t>(clip3(0, maxv_y, q0 - delta));
              if (deq) {
                const int dlt = clip3(-(tc >> 1), tc >> 1, (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1);
                b[across] = static_cast<uint16_t>(clip3(0, maxv_y, q1 + dlt));
              }
            }
          }
        }
      }
    }
  }
  // chroma (4:2:0): edges on the 8x8 chroma grid (16 luma samples), only where bS is 2
  for (int cy = 0; cy < CH; cy += (vertical ? 4 : 8)) {
    for (int cx = 0; cx < CW; cx += (vertical ? 8 : 4)) {
      if ((vertical && cx == 0) || (!vertical && cy == 0)) continue;
      const int x = cx * 2, y = cy * 2;
      const uint8_t e = edge_[static_cast<size_t>(idx4(x, y))];
      if (!(e & (tu_bit | pu_bit))) continue;
      const int xp = vertical ? x - 1 : x, yp = vertical ? y : y - 1;
      const int cq = (y >> log2_ctb) * wctb + (x >> log2_ctb), cp = (yp >> log2_ctb) * wctb + (xp >> log2_ctb);
      const int sq = ctb_slice_[static_cast<size_t>(cq)], sp = ctb_slice_[static_cast<size_t>(cp)];
      if (sq < 0 || sp < 0) continue;
      const SliceInfo& SQ = slices_[static_cast<size_t>(sq)];
      if (SQ.h.deblocking_disabled) continue;
      if (slices_[static_cast<size_t>(sp)].addr_rs != SQ.addr_rs && !SQ.h.loop_filter_across_slices) continue;
      if (!P.loop_filter_across_tiles && tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(cq)])] != tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(cp)])]) continue;
      if (bs_of(x, y, xp, yp, (e & tu_bit) != 0) != 2) continue;
      const int qp_q = qp_y_[static_cast<size_t>(idx4(x, y))], qp_p = qp_y_[static_cast<size_t>(idx4(xp, yp))];
      const bool no_p = (cu_flags_[static_cast<size_t>(idx4(xp, yp))] & kCuBypass) || ((cu_flags_[static_cast<size_t>(idx4(xp, yp))] & kCuPcm) && P.pcm_loop_filter_disabled);
      const bool no_q = (cu_flags_[static_cast<size_t>(idx4(x, y))] & kCuBypass) || ((cu_flags_[static_cast<size_t>(idx4(x, y))] & kCuPcm) && P.pcm_loop_filter_disabled);
      for (int c = 1; c <= 2; ++c) {
        std::vector<uint16_t>& pl = c == 1 ? F->u : F->v;
        const int qpi = ((qp_q + qp_p + 1) >> 1) + (c == 1 ? P.cb_qp_offset : P.cr_qp_offset);
        const int qpc = qpc_table(qpi);
        const int tc = kTcTable[clip3(0, 53, qpc + 2 + (SQ.h.tc_offset_div2 << 1))] * (1 << (bd_c - 8));
        uint16_t* base = &pl[static_cast<size_t>(cy) * static_cast<size_t>(F->stride_c) + static_cast<size_t>(cx)];
        const ptrdiff_t across = vertical ? 1 : F->stride_c, along = vertical ? F->stride_c : 1;
        // the 4 chroma samples of the segment belong to two luma segments with the same decisions (same coding units)
        for (int k = 0; k < 4; ++k) {
          uint16_t* b = base + k * along;
          const int p0 = b[-across], p1 = b[-2 * across], q0 = b[0], q1 = b[across];
          const int delta = clip3(-tc, tc, ((((q0 - p0) * 4) + p1 - q1 + 4) >> 3));
          if (!no_p) b[-across] = static_cast<uint16_t>(clip3(0, maxv_c, p0 + delta));
          if (!no_q) b[0] = static_cast<uint16_t>(clip3(0, maxv_c, q0 - delta));
        }
      }
    }
  }
}

void Dec::deblock() {
  deblock_edges(true);
  deblock_edges(false);
}

void Dec::sao_filter() {
  if (!P.sao) return;
  bool any = false;
  for (const SliceInfo& s : slices_) any |= s.h.sao_luma || s.h.sao_chroma;
  if (!any) return;
  for (int c = 0; c < 3; ++c) {
    std::vector<uint16_t>& dst = c == 0 ? F->y : (c == 1 ? F->u : F->v);
    const int stride = c == 0 ? F->stride_y : F->stride_c;
    const int pw = c == 0 ? W : CW, ph = c == 0 ? H : CH;
    const int sub = c == 0 ? 1 : 2;
    const int bd = c == 0 ? bd_y : bd_c;
    const int maxv = (1 << bd) - 1;
    const int ctb_w = ctb / sub, ctb_h = ctb / sub;
    std::vector<uint16_t> src(dst);
    for (int ry = 0; ry < hctb; ++ry)
      for (int rx = 0; rx < wctb; ++rx) {
        const int rs = ry * wctb + rx;
        const int si = ctb_slice_[static_cast<size_t>(rs)];
        if (si < 0) continue;
        const SliceInfo& SL = slices_[static_cast<size_t>(si)];
        if (!(c == 0 ? SL.h.sao_luma : SL.h.sao_chroma)) continue;
        const SaoParams& sp = sao_[static_cast<size_t>(rs)];
        if (sp.type[c] == 0) continue;
        const int x_ctb = rx * ctb_w, y_ctb = ry * ctb_h;
        const int x_end = std::min(pw, x_ctb + ctb_w), y_end = std::min(ph, y_ctb + ctb_h);
        // which neighbouring CTBs may be used by edge offsets
        bool nb_ok[3][3];
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            bool ok = true;
            const int nx = rx + dx, ny = ry + dy;
            if (nx < 0 || ny < 0 || nx >= wctb || ny >= hctb) ok = false;
            else if (dx != 0 || dy != 0) {
              const int nrs = ny * wctb + nx;
              const int nsi = ctb_slice_[static_cast<size_t>(nrs)];
              if (nsi < 0) ok = false;
              else {
                const SliceInfo& NS = slices_[static_cast<size_t>(nsi)];
                if (NS.addr_rs != SL.addr_rs) {
                  const bool nb_earlier = tiles_.rs_to_ts[static_cast<size_t>(nrs)] < tiles_.rs_to_ts[static_cast<size_t>(rs)];
                  if (nb_earlier && !SL.h.loop_filter_across_slices) ok = false;
                  if (!nb_earlier && !NS.h.loop_filter_across_slices) ok = false;
                }
                if (!P.loop_filter_across_tiles && tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(nrs)])] != tiles_.tile_id[static_cast<size_t>(tiles_.rs_to_ts[static_cast<size_t>(rs)])]) ok = false;
              }
            }
            nb_ok[dy + 1][dx + 1] = ok;
          }
        int band_table[32];
        std::memset(band_table, 0, sizeof band_table);
        if (sp.type[c] == 1)
          for (int k = 0; k < 4; ++k) band_table[(k + sp.band_pos[c]) & 31] = k + 1;
        static const int kHPos[4][2] = {{-1, 1}, {0, 0}, {-1, 1}, {1, -1}};
        static const int kVPos[4][2] = {{0, 0}, {-1, 1}, {-1, 1}, {-1, 1}};
        const int cls = sp.eo_class[c];
        for (int y = y_ctb; y < y_end; ++y)
          for (int x = x_ctb; x < x_end; ++x) {
            const size_t lidx = static_cast<size_t>(idx4(x * sub, y * sub));
            if ((cu_flags_[lidx] & kCuBypass) || ((cu_flags_[lidx] & kCuPcm) && P.pcm_loop_filter_disabled)) continue;
            const int v = src[static_cast<size_t>(y) * static_cast<size_t>(stride) + static_cast<size_t>(x)];
            int off;
            if (sp.type[c] == 1) {
              off = sp.offset[c][band_table[v >> (bd - 5)]];
            } else {
              int a[2];
              bool ok = true;
              for (int k = 0; k < 2 && ok; ++k) {
                const int nx = x + kHPos[cls][k], ny = y + kVPos[cls][k];
                if (nx < 0 || ny < 0 || nx >= pw || ny >= ph) {
                  ok = false;
                  break;
                }
                const int dx = nx < x_ctb ? -1 : (nx >= x_ctb + ctb_w ? 1 : 0), dy = ny < y_ctb ? -1 : (ny >= y_ctb + ctb_h ? 1 : 0);
                if (!nb_ok[dy + 1][dx + 1]) ok = false;
                a[k] = src[static_cast<size_t>(ny) * static_cast<size_t>(stride) + static_cast<size_t>(nx)];
              }
              if (!ok) continue;
              int edge_idx = 2 + (v > a[0] ? 1 : (v < a[0] ? -1 : 0)) + (v > a[1] ? 1 : (v < a[1] ? -1 : 0));
              if (edge_idx <= 2) edge_idx = edge_idx == 2 ? 0 : edge_idx + 1;
              off = sp.offset[c][edge_idx];
            }
            dst[static_cast<size_t>(y) * static_cast<size_t>(stride) + static_cast<size_t>(x)] = static_cast<uint16_t>(clip3(0, maxv, v + off));
          }
      }
  }
}
