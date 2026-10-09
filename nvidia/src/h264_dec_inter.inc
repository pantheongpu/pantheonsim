// Part of h264_decode.cpp: inter prediction -- motion data bookkeeping, motion vector prediction (8.4.1.3),
// P_Skip (8.4.1.1), direct prediction (8.4.1.2), fractional sample interpolation (8.4.2.2) and weighted
// prediction (8.4.2.3).

void Dec::set_motion(int x4, int y4, int w4, int h4, int list, int ref, int mvx, int mvy) {
  for (int y = y4; y < y4 + h4; ++y)
    for (int x = x4; x < x4 + w4; ++x) {
      mb_->mv[list][y * 4 + x][0] = static_cast<int16_t>(ref >= 0 ? mvx : 0);
      mb_->mv[list][y * 4 + x][1] = static_cast<int16_t>(ref >= 0 ? mvy : 0);
    }
  for (int y = y4 >> 1; y <= (y4 + h4 - 1) >> 1; ++y)
    for (int x = x4 >> 1; x <= (x4 + w4 - 1) >> 1; ++x) mb_->ref_idx[list][y * 2 + x] = static_cast<int8_t>(ref);
}

void Dec::set_mvd(int list, int x4, int y4, int w4, int h4, int dx, int dy) {
  const uint8_t ax = static_cast<uint8_t>(std::min(std::abs(dx), 255)), ay = static_cast<uint8_t>(std::min(std::abs(dy), 255));
  for (int y = y4; y < y4 + h4; ++y)
    for (int x = x4; x < x4 + w4; ++x) {
      mb_->mvd[list][y * 4 + x][0] = ax;
      mb_->mvd[list][y * 4 + x][1] = ay;
    }
}

int Dec::pred_flags_of(int b8) const {
  return (mb_->ref_idx[0][b8] >= 0 ? 1 : 0) | (mb_->ref_idx[1][b8] >= 0 ? 2 : 0);
}

RefPic Dec::ref_val(int list, int idx) const {
  RefPic none;
  if (idx < 0) return none;
  if (mbaff_ && mb_field_) {
    // a field of the frame RefPicList[idx / 2]: same parity for even indices, opposite for odd
    const int fi = idx >> 1;
    if (fi >= num_ref_[list]) return none;
    RefPic f = ref_[list][fi];
    if (!f.frame) return none;
    const int cur_par = addr_ & 1;
    const int par = (idx & 1) ? 1 - cur_par : cur_par;
    f.structure = 1 + par;
    f.poc = f.frame->poc[par];
    f.id = f.frame->uid * 4 + 1 + par;
    return f;
  }
  if (idx >= num_ref_[list]) return none;
  return ref_[list][idx];
}

int Dec::ref_id_of(int list, int idx) const { return ref_val(list, idx).id; }

int Dec::cur_poc_for_mb() const {
  if (mbaff_ && mb_field_) return pp_.poc[addr_ & 1];
  return cur_poc_;
}

// Motion data of the neighbouring partition covering luma location (xl, yl) of the current macroblock (8.4.1.3.2).
void Dec::get_nb_motion(int list, int xl, int yl, int* ref, int* mvx, int* mvy, bool* avail_out) {
  *ref = -1;
  *mvx = *mvy = 0;
  *avail_out = false;
  const Nb n = locate_l(xl, yl);
  if (n.addr < 0) return;
  const int bi = (n.y >> 2) * 4 + (n.x >> 2);
  const MbInfo* m;
  if (n.addr == addr_) {
    if (!((blk_done_ >> bi) & 1)) return;
    m = mb_;
  } else {
    m = &mbi(n.addr);
  }
  *avail_out = true;
  if (m->intra) return;
  const int b8 = (n.y >> 3) * 2 + (n.x >> 3);
  int r = m->ref_idx[list][b8];
  if (r < 0) return;
  int vx = m->mv[list][bi][0], vy = m->mv[list][bi][1];
  if (mbaff_) {
    if (mb_field_ && !m->field) {
      vy = vy / 2;
      r = r * 2;
    } else if (!mb_field_ && m->field) {
      vy = vy * 2;
      r = r >> 1;
    }
  }
  *ref = r;
  *mvx = vx;
  *mvy = vy;
}

void Dec::mv_pred(int list, int x4, int y4, int w4, int h4, int ref, int shape, int part_idx, int* mvx, int* mvy) {
  (void)h4;
  int ra, rb, rc, ax, ay, bx, by, cx, cy;
  bool va, vb, vc;
  get_nb_motion(list, x4 * 4 - 1, y4 * 4, &ra, &ax, &ay, &va);
  get_nb_motion(list, x4 * 4, y4 * 4 - 1, &rb, &bx, &by, &vb);
  get_nb_motion(list, (x4 + w4) * 4, y4 * 4 - 1, &rc, &cx, &cy, &vc);
  if (!vc) get_nb_motion(list, x4 * 4 - 1, y4 * 4 - 1, &rc, &cx, &cy, &vc);
  // directional prediction of 16x8 and 8x16 partitions
  if (shape == 1) {
    if (part_idx == 0 && rb == ref) { *mvx = bx; *mvy = by; return; }
    if (part_idx == 1 && ra == ref) { *mvx = ax; *mvy = ay; return; }
  } else if (shape == 2) {
    if (part_idx == 0 && ra == ref) { *mvx = ax; *mvy = ay; return; }
    if (part_idx == 1 && rc == ref) { *mvx = cx; *mvy = cy; return; }
  }
  if (!vb && !vc && va) {
    bx = cx = ax;
    by = cy = ay;
    rb = rc = ra;
  }
  const int matches = (ra == ref) + (rb == ref) + (rc == ref);
  if (matches == 1) {
    if (ra == ref) { *mvx = ax; *mvy = ay; }
    else if (rb == ref) { *mvx = bx; *mvy = by; }
    else { *mvx = cx; *mvy = cy; }
    return;
  }
  *mvx = median3(ax, bx, cx);
  *mvy = median3(ay, by, cy);
}

void Dec::derive_pskip() {
  int ra, rb, ax, ay, bx, by;
  bool va, vb;
  blk_done_ = 0;
  get_nb_motion(0, -1, 0, &ra, &ax, &ay, &va);
  get_nb_motion(0, 0, -1, &rb, &bx, &by, &vb);
  int mvx = 0, mvy = 0;
  if (va && vb && !(ra == 0 && ax == 0 && ay == 0) && !(rb == 0 && bx == 0 && by == 0)) mv_pred(0, 0, 0, 4, 4, 0, 0, 0, &mvx, &mvy);
  set_motion(0, 0, 4, 4, 0, 0, mvx, mvy);
  set_motion(0, 0, 4, 4, 1, -1, 0, 0);
  for (int b8 = 0; b8 < 4; ++b8) {
    mb_->ref_id[0][b8] = ref_id_of(0, 0);
    mb_->pred_flags[b8] = 1;
  }
}

// ---- direct prediction ---------------------------------------------------------------------------------------------
static inline int min_positive(int a, int b) { return (a >= 0 && b >= 0) ? std::min(a, b) : std::max(a, b); }

// The co-located 4x4 block for (blk8, sub4) (8.4.1.2.1): motion of the block of RefPicList1[0] at the same place.
bool Dec::colocated(int blk8, int sub4, int* ref_col, int* mv_col, int* col_ref_id, bool* col_field, int* vert_scale) {
  *ref_col = -1;
  mv_col[0] = mv_col[1] = 0;
  *col_ref_id = -1;
  *col_field = false;
  *vert_scale = 0;
  const RefPic r1v = ref_val(1, 0);
  const RefPic* r1 = &r1v;
  if (!r1->frame) return true;
  const int blk = pp_.direct_8x8_inference ? 5 * blk8 : 4 * blk8 + sub4;
  const int xcol = blk_x(blk) * 4, ycol = blk_y(blk) * 4;
  const PicData* cp = nullptr;
  int col_addr = addr_;
  int ym = ycol;
  if (!field_pic_ && !mbaff_) {
    // FRM current: the co-located picture is the frame RefPicList1[0] (coded as a frame) or one of its fields
    if (r1->frame->coded_as_frame && r1->frame->pic[0]) {
      cp = r1->frame->pic[0].get();
    } else if (r1->frame->pic[1] && r1->frame->pic[2]) {
      // complementary field pair: Table 8-6 picks the field closer in POC; Table 8-8: Fld_To_Frm
      const int top_diff = std::abs(r1->frame->poc[0] - cur_poc_), bot_diff = std::abs(r1->frame->poc[1] - cur_poc_);
      const int fld = top_diff < bot_diff ? 1 : 2;
      cp = r1->frame->pic[fld].get();
      col_addr = (addr_ / (2 * W_)) * W_ + (addr_ % W_);   // mbAddrCol4
      ym = 8 * ((addr_ / W_) % 2) + 4 * (ycol / 8);
      *vert_scale = 2;   // Fld_To_Frm
    }
  } else if (field_pic_) {
    // FLD current: colPic is the field of RefPicList1[0] (or its frame, if it was coded as a frame)
    const Frame* f = r1->frame;
    if (f->coded_as_frame && f->pic[0]) {
      cp = f->pic[0].get();
      if (!cp->mbaff) {
        col_addr = 2 * W_ * (addr_ / W_) + (addr_ % W_) + W_ * (ycol / 8);   // mbAddrCol1
        ym = (2 * ycol) % 16;
        *vert_scale = 1;   // Frm_To_Fld
      } else {
        const int x = 2 * addr_;
        if (cp->mb[x].field) {
          col_addr = 2 * addr_ + (bottom_ ? 1 : 0);   // mbAddrCol3
          ym = ycol;
        } else {
          col_addr = 2 * addr_ + (ycol / 8);   // mbAddrCol2
          ym = (2 * ycol) % 16;
          *vert_scale = 1;
        }
      }
    } else {
      const int fld = r1->structure ? r1->structure : 1;
      cp = f->pic[fld].get();
    }
  } else {
    // AFRM current (macroblock-adaptive): handled for frames coded the same way
    const Frame* f = r1->frame;
    if (f->coded_as_frame && f->pic[0]) {
      cp = f->pic[0].get();
      const MbInfo& cm = cp->mb[addr_];
      if (!mb_field_) {
        if (!cm.field) {
          col_addr = addr_;
        } else {
          const int top_diff = std::abs(f->poc[0] - cur_poc_), bot_diff = std::abs(f->poc[1] - cur_poc_);
          col_addr = 2 * (addr_ / 2) + (top_diff < bot_diff ? 0 : 1);   // mbAddrCol6
          ym = 8 * (addr_ % 2) + 4 * (ycol / 8);
          *vert_scale = 2;
        }
      } else {
        if (!cm.field) {
          col_addr = 2 * (addr_ / 2) + (ycol / 8);   // mbAddrCol7
          ym = (2 * ycol) % 16;
          *vert_scale = 1;
        } else {
          col_addr = addr_;
        }
      }
    } else if (f->pic[1] && f->pic[2]) {
      // current frame (AFRM) with a complementary field pair as the co-located picture
      if (!mb_field_) {
        const int top_diff = std::abs(f->poc[0] - cur_poc_), bot_diff = std::abs(f->poc[1] - cur_poc_);
        cp = f->pic[top_diff < bot_diff ? 1 : 2].get();
        col_addr = addr_ / 2;   // mbAddrCol5
        ym = 8 * (addr_ % 2) + 4 * (ycol / 8);
        *vert_scale = 2;
      } else {
        cp = f->pic[(addr_ & 1) ? 2 : 1].get();
        col_addr = addr_ / 2;
        ym = ycol;
      }
    }
  }
  if (!cp || col_addr < 0 || col_addr >= static_cast<int>(cp->mb.size())) return true;
  const MbInfo& cm = cp->mb[col_addr];
  *col_field = cm.field;
  if (cm.slice < 0 || cm.intra) return true;
  const int bi = (ym >> 2) * 4 + (xcol >> 2);
  const int b8 = (ym >> 3) * 2 + (xcol >> 3);
  if (cm.ref_idx[0][b8] >= 0) {
    *ref_col = cm.ref_idx[0][b8];
    mv_col[0] = cm.mv[0][bi][0];
    mv_col[1] = cm.mv[0][bi][1];
    *col_ref_id = cm.ref_id[0][b8];
  } else if (cm.ref_idx[1][b8] >= 0) {
    *ref_col = cm.ref_idx[1][b8];
    mv_col[0] = cm.mv[1][bi][0];
    mv_col[1] = cm.mv[1][bi][1];
    *col_ref_id = cm.ref_id[1][b8];
  }
  return true;
}

void Dec::derive_direct(int which) {
  uint8_t mask = 0xF;
  if (which == -2) mask = mb_->direct8;
  else if (which >= 0) mask = static_cast<uint8_t>(1 << which);
  if (sh_.direct_spatial_mv_pred) direct_spatial(mask);
  else direct_temporal(mask);
  for (int b8 = 0; b8 < 4; ++b8)
    if (mask & (1 << b8)) mb_->pred_flags[b8] = static_cast<uint8_t>(pred_flags_of(b8));
}

void Dec::direct_spatial(uint8_t mask) {
  int ref[2];
  int mvp[2][2] = {{0, 0}, {0, 0}};
  const uint16_t saved_done = blk_done_;
  blk_done_ = 0;
  for (int l = 0; l < 2; ++l) {
    int ra, rb, rc, ax, ay, bx, by, cx, cy;
    bool va, vb, vc;
    get_nb_motion(l, -1, 0, &ra, &ax, &ay, &va);
    get_nb_motion(l, 0, -1, &rb, &bx, &by, &vb);
    get_nb_motion(l, 16, -1, &rc, &cx, &cy, &vc);
    if (!vc) get_nb_motion(l, -1, -1, &rc, &cx, &cy, &vc);
    ref[l] = min_positive(ra, min_positive(rb, rc));
  }
  bool zero_pred = false;
  if (ref[0] < 0 && ref[1] < 0) {
    ref[0] = ref[1] = 0;
    zero_pred = true;
  }
  if (!zero_pred)
    for (int l = 0; l < 2; ++l)
      if (ref[l] >= 0) mv_pred(l, 0, 0, 4, 4, ref[l], 0, 0, &mvp[l][0], &mvp[l][1]);
  blk_done_ = saved_done;
  const RefPic r1v = ref_val(1, 0);
  const bool l1_short = r1v.frame && !r1v.long_term;
  for (int b8 = 0; b8 < 4; ++b8) {
    if (!(mask & (1 << b8))) continue;
    const int nsub = pp_.direct_8x8_inference ? 1 : 4;
    for (int s = 0; s < nsub; ++s) {
      bool col_zero = false;
      if (l1_short && !zero_pred && (ref[0] == 0 || ref[1] == 0)) {
        int rc, mc[2], cid, vs;
        bool cf;
        colocated(b8, s, &rc, mc, &cid, &cf, &vs);
        col_zero = rc == 0 && mc[0] >= -1 && mc[0] <= 1 && mc[1] >= -1 && mc[1] <= 1;
      }
      const int x4 = (b8 & 1) * 2 + (pp_.direct_8x8_inference ? 0 : (s & 1)), y4 = (b8 >> 1) * 2 + (pp_.direct_8x8_inference ? 0 : (s >> 1));
      const int w = pp_.direct_8x8_inference ? 2 : 1;
      for (int l = 0; l < 2; ++l) {
        int mx = 0, my = 0;
        if (!zero_pred && ref[l] >= 0 && !(ref[l] == 0 && col_zero)) {
          mx = mvp[l][0];
          my = mvp[l][1];
        }
        set_motion(x4, y4, w, w, l, ref[l], mx, my);
      }
    }
  }
}

void Dec::direct_temporal(uint8_t mask) {
  for (int b8 = 0; b8 < 4; ++b8) {
    if (!(mask & (1 << b8))) continue;
    const int nsub = pp_.direct_8x8_inference ? 1 : 4;
    for (int s = 0; s < nsub; ++s) {
      int rc, mc[2], cid, vs;
      bool cf;
      colocated(b8, s, &rc, mc, &cid, &cf, &vs);
      int ref0 = 0;
      if (rc >= 0) {
        // MapColToList0: the lowest index of RefPicList0 that references the picture the co-located block referenced
        int found = -1;
        const bool cur_field_mb = mbaff_ && mb_field_;
        const int n = cur_field_mb ? num_ref_[0] * 2 : num_ref_[0];
        for (int i = 0; i < n && found < 0; ++i) {
          const RefPic rv = ref_val(0, i);
          const RefPic* r = &rv;
          if (!r->frame) continue;
          const int rid = r->id;
          bool match;
          if (vs == 0) {
            // one to one: same kind of picture
            match = rid == cid;
            if (cur_field_mb) {
              // a field of the same frame: compare frames, then parity handled by the index parity
              const int frame_uid = cid >= 0 ? cid / 4 : -1;
              match = r->frame->uid == frame_uid && ((cid & 3) == r->structure);
            }
          } else if (vs == 1) {
            // Frm_To_Fld: the field (of the current picture's parity) of the frame referenced
            const int frame_uid = cid >= 0 ? cid / 4 : -1;
            match = r->frame->uid == frame_uid && (field_pic_ ? r->structure == 1 + (bottom_ ? 1 : 0) : (i & 1) == 0);
          } else {
            // Fld_To_Frm: the frame containing the field referenced
            const int frame_uid = cid >= 0 ? cid / 4 : -1;
            match = r->frame->uid == frame_uid;
          }
          if (match) found = i;
        }
        if (found < 0) found = 0;
        ref0 = found;
        if (vs == 1) mc[1] = mc[1] / 2;
        else if (vs == 2) mc[1] = mc[1] * 2;
      }
      const RefPic p0v = ref_val(0, ref0), p1v = ref_val(1, 0);
      const RefPic* p0 = &p0v;
      const RefPic* p1 = &p1v;
      int mv0[2] = {mc[0], mc[1]}, mv1[2] = {0, 0};
      if (p0->frame && p1->frame) {
        const int poc_cur = cur_poc_for_mb();
        const int diff10 = p1->poc - p0->poc;
        if (!p0->long_term && diff10 != 0) {
          const int tb = clip3(-128, 127, poc_cur - p0->poc);
          const int td = clip3(-128, 127, diff10);
          const int tx = (16384 + std::abs(td / 2)) / td;
          const int dsf = clip3(-1024, 1023, (tb * tx + 32) >> 6);
          mv0[0] = (dsf * mc[0] + 128) >> 8;
          mv0[1] = (dsf * mc[1] + 128) >> 8;
          mv1[0] = mv0[0] - mc[0];
          mv1[1] = mv0[1] - mc[1];
        }
      }
      const int x4 = (b8 & 1) * 2 + (pp_.direct_8x8_inference ? 0 : (s & 1)), y4 = (b8 >> 1) * 2 + (pp_.direct_8x8_inference ? 0 : (s >> 1));
      const int w = pp_.direct_8x8_inference ? 2 : 1;
      set_motion(x4, y4, w, w, 0, ref0, mv0[0], mv0[1]);
      set_motion(x4, y4, w, w, 1, 0, mv1[0], mv1[1]);
    }
  }
}

// ---- fractional sample interpolation ------------------------------------------------------------------------------
namespace {
inline int fetch(const Plane& r, int x, int y) {
  x = x < 0 ? 0 : (x >= r.w ? r.w - 1 : x);
  y = y < 0 ? 0 : (y >= r.h ? r.h - 1 : y);
  return r.p[static_cast<ptrdiff_t>(y) * r.stride + x];
}

// Luma prediction of a w x h block whose top-left integer sample position in the reference is (x0, y0) with
// quarter-sample fraction (fx, fy).
void mc_luma(const Plane& r, int x0, int y0, int fx, int fy, int w, int h, uint8_t* out, int os) {
  if (fx == 0 && fy == 0) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) out[y * os + x] = static_cast<uint8_t>(fetch(r, x0 + x, y0 + y));
    return;
  }
  const int ww = w + 5, wh = h + 5;
  int win[21 * 21];
  for (int y = 0; y < wh; ++y)
    for (int x = 0; x < ww; ++x) win[y * ww + x] = fetch(r, x0 - 2 + x, y0 - 2 + y);
  auto G = [&](int x, int y) { return win[(y + 2) * ww + x + 2]; };
  auto b1 = [&](int x, int y) { return G(x - 2, y) - 5 * G(x - 1, y) + 20 * G(x, y) + 20 * G(x + 1, y) - 5 * G(x + 2, y) + G(x + 3, y); };
  auto h1 = [&](int x, int y) { return G(x, y - 2) - 5 * G(x, y - 1) + 20 * G(x, y) + 20 * G(x, y + 1) - 5 * G(x, y + 2) + G(x, y + 3); };
  auto bb = [&](int x, int y) { return static_cast<int>(clip1((b1(x, y) + 16) >> 5)); };
  auto hh = [&](int x, int y) { return static_cast<int>(clip1((h1(x, y) + 16) >> 5)); };
  auto jj = [&](int x, int y) {
    const int j1 = b1(x, y - 2) - 5 * b1(x, y - 1) + 20 * b1(x, y) + 20 * b1(x, y + 1) - 5 * b1(x, y + 2) + b1(x, y + 3);
    return static_cast<int>(clip1((j1 + 512) >> 10));
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int v;
      switch (fy * 4 + fx) {
        case 1: v = (G(x, y) + bb(x, y) + 1) >> 1; break;                  // a
        case 2: v = bb(x, y); break;                                       // b
        case 3: v = (G(x + 1, y) + bb(x, y) + 1) >> 1; break;             // c
        case 4: v = (G(x, y) + hh(x, y) + 1) >> 1; break;                  // d
        case 5: v = (bb(x, y) + hh(x, y) + 1) >> 1; break;                 // e
        case 6: v = (bb(x, y) + jj(x, y) + 1) >> 1; break;                 // f
        case 7: v = (bb(x, y) + hh(x + 1, y) + 1) >> 1; break;             // g
        case 8: v = hh(x, y); break;                                       // h
        case 9: v = (hh(x, y) + jj(x, y) + 1) >> 1; break;                 // i
        case 10: v = jj(x, y); break;                                      // j
        case 11: v = (jj(x, y) + hh(x + 1, y) + 1) >> 1; break;            // k
        case 12: v = (G(x, y + 1) + hh(x, y) + 1) >> 1; break;             // n
        case 13: v = (hh(x, y) + bb(x, y + 1) + 1) >> 1; break;            // p
        case 14: v = (jj(x, y) + bb(x, y + 1) + 1) >> 1; break;            // q
        default: v = (hh(x + 1, y) + bb(x, y + 1) + 1) >> 1; break;        // r
      }
      out[y * os + x] = static_cast<uint8_t>(v);
    }
}

void mc_chroma(const Plane& r, int x0, int y0, int fx, int fy, int w, int h, uint8_t* out, int os) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int A = fetch(r, x0 + x, y0 + y), B = fetch(r, x0 + x + 1, y0 + y), C = fetch(r, x0 + x, y0 + y + 1), D = fetch(r, x0 + x + 1, y0 + y + 1);
      out[y * os + x] = static_cast<uint8_t>(((8 - fx) * (8 - fy) * A + fx * (8 - fy) * B + (8 - fx) * fy * C + fx * fy * D + 32) >> 6);
    }
}
}  // namespace

// The planes of reference picture `r` for the current macroblock: a frame, or a field of it.
void Dec::ref_planes(const RefPic& r, Plane* y, Plane* u, Plane* v) const {
  Frame* f = r.frame;
  const int fs = f->stride_y, cs = f->stride_c;
  const int fh = f->mbs_h * 16;
  if (r.structure == 0) {
    *y = {f->y.data(), fs, fs, fh};
    *u = {f->u.data(), cs, cs, fh / 2};
    *v = {f->v.data(), cs, cs, fh / 2};
  } else {
    const bool bot = r.structure == 2;
    *y = {f->y.data() + (bot ? fs : 0), 2 * fs, fs, fh / 2};
    *u = {f->u.data() + (bot ? cs : 0), 2 * cs, cs, fh / 4};
    *v = {f->v.data() + (bot ? cs : 0), 2 * cs, cs, fh / 4};
  }
}

void Dec::mc_block(int x4, int y4, int w4, int h4) {
  const int b8 = (y4 >> 1) * 2 + (x4 >> 1);
  const int bi = y4 * 4 + x4;
  const int w = w4 * 4, h = h4 * 4, cw = w4 * 2, ch = h4 * 2;
  uint8_t pl[2][256], pcb[2][64], pcr[2][64];
  bool used[2] = {false, false};
  int ref[2] = {mb_->ref_idx[0][b8], mb_->ref_idx[1][b8]};
  for (int l = 0; l < 2; ++l) {
    if (ref[l] < 0) continue;
    const RefPic rpv = ref_val(l, ref[l]);
    const RefPic* rp = &rpv;
    if (!rp->frame) {
      // missing reference: grey
      std::memset(pl[l], 128, sizeof pl[l]);
      std::memset(pcb[l], 128, sizeof pcb[l]);
      std::memset(pcr[l], 128, sizeof pcr[l]);
      used[l] = true;
      continue;
    }
    used[l] = true;
    Plane ry, ru, rv;
    ref_planes(*rp, &ry, &ru, &rv);
    const int mvx = mb_->mv[l][bi][0], mvy = mb_->mv[l][bi][1];
    const int xal = org_x_ + x4 * 4, yal = org_y_ + y4 * 4;
    mc_luma(ry, xal + (mvx >> 2), yal + (mvy >> 2), mvx & 3, mvy & 3, w, h, pl[l], 16);
    int mvcy = mvy;
    if (mbaff_ ? mb_field_ : field_pic_) {
      // Table 8-10: vertical chroma offset between fields of different parity
      const int cur_par = field_pic_ ? (bottom_ ? 1 : 0) : (addr_ & 1);
      const int ref_par = rp->structure == 2 ? 1 : 0;
      if (rp->structure != 0 && ref_par != cur_par) mvcy += ref_par == 0 ? 2 : -2;
      // (reference top field, current bottom field: +2... see Table 8-10)
    }
    const int xac = xal / 2, yac = yal / 2;
    mc_chroma(ru, xac + (mvx >> 3), yac + (mvcy >> 3), mvx & 7, mvcy & 7, cw, ch, pcb[l], 8);
    mc_chroma(rv, xac + (mvx >> 3), yac + (mvcy >> 3), mvx & 7, mvcy & 7, cw, ch, pcr[l], 8);
  }
  if (!used[0] && !used[1]) return;
  // weights
  enum { kDefault, kExplicit, kImplicit } mode = kDefault;
  if (sh_.is_p()) mode = pp_.weighted_pred ? kExplicit : kDefault;
  else if (pp_.weighted_bipred_idc == 1) mode = kExplicit;
  else if (pp_.weighted_bipred_idc == 2 && used[0] && used[1]) mode = kImplicit;
  int logwd[3] = {0, 0, 0}, w0[3] = {1, 1, 1}, w1[3] = {1, 1, 1}, o0[3] = {0, 0, 0}, o1[3] = {0, 0, 0};
  if (mode == kExplicit) {
    const PredWeight& pw = sh_.pw;
    const int sh_shift = (mbaff_ && mb_field_) ? 1 : 0;
    logwd[0] = pw.luma_log2_denom;
    logwd[1] = logwd[2] = pw.chroma_log2_denom;
    if (used[0]) {
      const int i = ref[0] >> sh_shift;
      w0[0] = pw.luma_weight[0][i];
      o0[0] = pw.luma_offset[0][i];
      for (int c = 0; c < 2; ++c) {
        w0[1 + c] = pw.chroma_weight[0][i][c];
        o0[1 + c] = pw.chroma_offset[0][i][c];
      }
    }
    if (used[1]) {
      const int i = ref[1] >> sh_shift;
      w1[0] = pw.luma_weight[1][i];
      o1[0] = pw.luma_offset[1][i];
      for (int c = 0; c < 2; ++c) {
        w1[1 + c] = pw.chroma_weight[1][i][c];
        o1[1 + c] = pw.chroma_offset[1][i][c];
      }
    }
  } else if (mode == kImplicit) {
    const RefPic p0v = ref_val(0, ref[0]), p1v = ref_val(1, ref[1]);
    const RefPic* p0 = &p0v;
    const RefPic* p1 = &p1v;
    int a = 32, b = 32;
    if (p0->frame && p1->frame) {
      const int poc_cur = cur_poc_for_mb();
      const int tb = clip3(-128, 127, poc_cur - p0->poc);
      const int td = clip3(-128, 127, p1->poc - p0->poc);
      if (td != 0 && !p0->long_term && !p1->long_term) {
        const int tx = (16384 + std::abs(td / 2)) / td;
        const int dsf = clip3(-1024, 1023, (tb * tx + 32) >> 6);
        if ((dsf >> 2) >= -64 && (dsf >> 2) <= 128) {
          a = 64 - (dsf >> 2);
          b = dsf >> 2;
        }
      }
    }
    logwd[0] = logwd[1] = logwd[2] = 5;
    w0[0] = w0[1] = w0[2] = a;
    w1[0] = w1[1] = w1[2] = b;
  }
  auto combine = [&](int comp, const uint8_t* s0, const uint8_t* s1, int ss, uint8_t* dst, int ds, int bw, int bh) {
    for (int y = 0; y < bh; ++y)
      for (int x = 0; x < bw; ++x) {
        int v;
        const int a = used[0] ? s0[y * ss + x] : 0, b = used[1] ? s1[y * ss + x] : 0;
        if (mode == kDefault) {
          v = (used[0] && used[1]) ? (a + b + 1) >> 1 : (used[0] ? a : b);
        } else if (used[0] && used[1]) {
          v = clip1(((a * w0[comp] + b * w1[comp] + (1 << logwd[comp])) >> (logwd[comp] + 1)) + ((o0[comp] + o1[comp] + 1) >> 1));
        } else {
          const int s = used[0] ? a : b, wt = used[0] ? w0[comp] : w1[comp], of = used[0] ? o0[comp] : o1[comp];
          if (logwd[comp] >= 1) v = clip1(((s * wt + (1 << (logwd[comp] - 1))) >> logwd[comp]) + of);
          else v = clip1(s * wt + of);
        }
        dst[y * ds + x] = static_cast<uint8_t>(v);
      }
  };
  combine(0, pl[0], pl[1], 16, ly_ + y4 * 4 * ls_ + x4 * 4, ls_, w, h);
  combine(1, pcb[0], pcb[1], 8, lu_ + y4 * 2 * lcs_ + x4 * 2, lcs_, cw, ch);
  combine(2, pcr[0], pcr[1], 8, lv_ + y4 * 2 * lcs_ + x4 * 2, lcs_, cw, ch);
}

void Dec::inter_predict_mb() {
  auto same = [&](int a, int b) {
    for (int l = 0; l < 2; ++l) {
      if (mb_->ref_idx[l][(a >> 3) * 2 + ((a & 3) >> 1)] != mb_->ref_idx[l][(b >> 3) * 2 + ((b & 3) >> 1)]) return false;
      if (mb_->ref_idx[l][(a >> 3) * 2 + ((a & 3) >> 1)] >= 0 && (mb_->mv[l][a][0] != mb_->mv[l][b][0] || mb_->mv[l][a][1] != mb_->mv[l][b][1])) return false;
    }
    return true;
  };
  bool all = true;
  for (int i = 1; i < 16 && all; ++i) all = same(0, i);
  if (all) {
    mc_block(0, 0, 4, 4);
    return;
  }
  for (int b8 = 0; b8 < 4; ++b8) {
    const int x4 = (b8 & 1) * 2, y4 = (b8 >> 1) * 2;
    const int first = y4 * 4 + x4;
    if (same(first, first + 1) && same(first, first + 4) && same(first, first + 5)) {
      mc_block(x4, y4, 2, 2);
    } else {
      for (int k = 0; k < 4; ++k) mc_block(x4 + (k & 1), y4 + (k >> 1), 1, 1);
    }
  }
}
