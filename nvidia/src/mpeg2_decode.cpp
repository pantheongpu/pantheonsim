// See mpeg2_decode.hpp. Clause numbers refer to Rec. ITU-T H.262 (02/2000).
#include "mpeg2_decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vgpu_mpeg2 {

void Frame::alloc(int w, int h) {
  width = w;
  height = h;
  stride_y = w;
  stride_c = w / 2;
  y.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
  u.assign(static_cast<size_t>(w / 2) * static_cast<size_t>(h / 2), 128);
  v.assign(static_cast<size_t>(w / 2) * static_cast<size_t>(h / 2), 128);
}

#ifdef VGPU_MPEG2_DEBUG_BLOCKS
FILE* g_block_dump = nullptr;
#endif

namespace {

struct VlcEntry {
  uint32_t code;
  int len;
  int value;
};
struct DctEntry {
  uint32_t code;
  int len;
  int run;     // -1 end of block, -2 escape
  int level;
};
#include "mpeg2_tables.inc"

// ---- lookup tables built from the code lists of Annex B ----------------------------------------------------------------
struct Lut {
  int bits = 0;
  std::vector<uint16_t> idx;   // entry index + 1, 0 = not a code
  std::vector<uint8_t> len;
};

template <class E>
Lut build_lut(const E* e, size_t n, int bits, bool (*skip)(const E&) = nullptr) {
  Lut l;
  l.bits = bits;
  l.idx.assign(static_cast<size_t>(1) << bits, 0);
  l.len.assign(static_cast<size_t>(1) << bits, 0);
  for (size_t i = 0; i < n; ++i) {
    if (skip && skip(e[i])) continue;
    const int sh = bits - e[i].len;
    const size_t lo = static_cast<size_t>(e[i].code) << sh, hi = (static_cast<size_t>(e[i].code) + 1) << sh;
    for (size_t k = lo; k < hi; ++k) {
      l.idx[k] = static_cast<uint16_t>(i + 1);
      l.len[k] = static_cast<uint8_t>(e[i].len);
    }
  }
  return l;
}

struct Tables {
  Lut inc, cbp, motion, dct0, dct1;
  Tables() {
    inc = build_lut(kMbAddrIncVlc, sizeof kMbAddrIncVlc / sizeof kMbAddrIncVlc[0], 11);
    cbp = build_lut(kCbpVlc, sizeof kCbpVlc / sizeof kCbpVlc[0], 9);
    motion = build_lut(kMotionVlc, sizeof kMotionVlc / sizeof kMotionVlc[0], 11);
    // the one-bit code of Table B.14 is for the first coefficient of a non-intra block only (7.2.2.2)
    dct0 = build_lut<DctEntry>(kDctZeroVlc, sizeof kDctZeroVlc / sizeof kDctZeroVlc[0], 16, [](const DctEntry& e) { return e.len == 1; });
    dct1 = build_lut<DctEntry>(kDctOneVlc, sizeof kDctOneVlc / sizeof kDctOneVlc[0], 16);
  }
};
const Tables& tables() {
  static const Tables t;
  return t;
}

// ---- bit reader ---------------------------------------------------------------------------------------------------------
class Bits {
 public:
  Bits(const uint8_t* d, size_t n) : d_(d), n_(n) {}
  // The next n (<= 24) bits, zeros beyond the end.
  uint32_t peek(int n) const {
    uint32_t v = 0;
    size_t byte = pos_ >> 3;
    for (int i = 0; i < 4; ++i) v = (v << 8) | (byte + static_cast<size_t>(i) < n_ ? d_[byte + static_cast<size_t>(i)] : 0u);
    return (v << (pos_ & 7)) >> (32 - n);
  }
  uint32_t get(int n) {
    const uint32_t v = n ? peek(n) : 0;
    skip(n);
    return v;
  }
  void skip(int n) { pos_ += static_cast<size_t>(n); }
  bool over() const { return pos_ > n_ * 8; }
  size_t pos() const { return pos_; }
  size_t bits_left() const { return pos_ >= n_ * 8 ? 0 : n_ * 8 - pos_; }

 private:
  const uint8_t* d_;
  size_t n_;
  size_t pos_ = 0;
};

// ---- the inverse DCT ----------------------------------------------------------------------------------------------------
#include "mpeg2_idct.inc"

// ---- constants -----------------------------------------------------------------------------------------------------------
const uint8_t kQuantScale[2][32] = {
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112}};   // Table 7-6

// Table B.12 and B.13: dct_dc_size
struct DcSize {
  uint32_t code;
  int len;
  int size;
};
const DcSize kDcLuma[] = {{0x4, 3, 0},  {0x0, 2, 1},  {0x1, 2, 2},  {0x5, 3, 3},   {0x6, 3, 4},   {0xe, 4, 5},
                          {0x1e, 5, 6}, {0x3e, 6, 7}, {0x7e, 7, 8}, {0xfe, 8, 9}, {0x1fe, 9, 10}, {0x1ff, 9, 11}};
const DcSize kDcChroma[] = {{0x0, 2, 0},  {0x1, 2, 1},  {0x2, 2, 2},  {0x6, 3, 3},   {0xe, 4, 4},   {0x1e, 5, 5},
                            {0x3e, 6, 6}, {0x7e, 7, 7}, {0xfe, 8, 8}, {0x1fe, 9, 9}, {0x3fe, 10, 10}, {0x3ff, 10, 11}};

// Tables B.2, B.3 and B.4: macroblock_type
struct MbType {
  uint32_t code;
  int len;
  bool quant, fwd, bwd, pattern, intra;
};
const MbType kMbTypeI[] = {{0x1, 1, false, false, false, false, true}, {0x1, 2, true, false, false, false, true}};
const MbType kMbTypeP[] = {{0x1, 1, false, true, false, true, false},  {0x1, 2, false, false, false, true, false}, {0x1, 3, false, true, false, false, false},
                           {0x3, 5, false, false, false, false, true}, {0x2, 5, true, true, false, true, false},   {0x1, 5, true, false, false, true, false},
                           {0x1, 6, true, false, false, false, true}};
const MbType kMbTypeB[] = {{0x2, 2, false, true, true, false, false}, {0x3, 2, false, true, true, true, false},  {0x2, 3, false, false, true, false, false},
                           {0x3, 3, false, false, true, true, false}, {0x2, 4, false, true, false, false, false}, {0x3, 4, false, true, false, true, false},
                           {0x3, 5, false, false, false, false, true}, {0x2, 5, true, true, true, true, false},   {0x3, 6, true, true, false, true, false},
                           {0x2, 6, true, false, true, true, false},  {0x1, 6, true, false, false, false, true}};

struct Mv {
  int x = 0, y = 0;
};

// A plane of a frame as a frame or as one of its fields.
struct View {
  uint8_t* p = nullptr;
  int stride = 0, w = 0, h = 0;
};

View plane_view(const Frame* f, int plane, int parity) {   // parity -1: the frame
  View v;
  if (!f) return v;
  const int cw = f->width / 2, ch = f->height / 2;
  uint8_t* base = const_cast<uint8_t*>(plane == 0 ? f->y.data() : plane == 1 ? f->u.data() : f->v.data());
  const int stride = plane == 0 ? f->stride_y : f->stride_c;
  const int w = plane == 0 ? f->width : cw, h = plane == 0 ? f->height : ch;
  if (parity < 0) {
    v = {base, stride, w, h};
  } else {
    v = {base + parity * stride, stride * 2, w, h / 2};
  }
  return v;
}

// The prediction of a bw x bh block at (x, y) displaced by the half-sample vector (mx, my), written to out (or averaged with it): 7.6.4.
void predict_block(const View& r, int x, int y, int bw, int bh, int mx, int my, uint8_t* out, int ostride, bool average) {
  const int ix = x + (mx >> 1), iy = y + (my >> 1);
  const int hx = mx & 1, hy = my & 1;
  const bool inside = ix >= 0 && iy >= 0 && ix + bw + hx <= r.w && iy + bh + hy <= r.h;
  auto at = [&](int xx, int yy) -> int {
    if (!inside) {
      xx = std::min(std::max(xx, 0), r.w - 1);
      yy = std::min(std::max(yy, 0), r.h - 1);
    }
    return r.p[static_cast<ptrdiff_t>(yy) * r.stride + xx];
  };
  for (int j = 0; j < bh; ++j) {
    uint8_t* o = out + static_cast<ptrdiff_t>(j) * ostride;
    for (int i = 0; i < bw; ++i) {
      int v;
      const int a = at(ix + i, iy + j);
      if (!hx && !hy) v = a;
      else if (hx && !hy) v = (a + at(ix + i + 1, iy + j) + 1) >> 1;
      else if (!hx && hy) v = (a + at(ix + i, iy + j + 1) + 1) >> 1;
      else v = (a + at(ix + i + 1, iy + j) + at(ix + i, iy + j + 1) + at(ix + i + 1, iy + j + 1) + 2) >> 2;
      o[i] = static_cast<uint8_t>(average ? (o[i] + v + 1) >> 1 : v);
    }
  }
}

int div2_away(int x) { return x >= 0 ? (x + 1) / 2 : -((-x + 1) / 2); }   // the specification's "//" by two

class Decoder {
 public:
  Decoder(Frame* cur, const Frame* fwd, const Frame* bwd, const PicParams& pp, std::string* err) : cur_(cur), fwd_(fwd), bwd_(bwd), pp_(pp), err_(err) {
    mbw_ = pp.mb_width;
    mbh_ = pp.field_pic ? pp.mb_height / 2 : pp.mb_height;   // macroblock rows of the picture
    parity_ = pp.field_pic ? (pp.bottom_field ? 1 : 0) : -1;
    scan_ = kScanPos[pp.alternate_scan ? 1 : 0];
    for (int c = 0; c < 3; ++c) dest_[c] = plane_view(cur, c, parity_);
  }

  bool run(const std::vector<SliceData>& slices) {
    if (!cur_ || cur_->width != mbw_ * 16 || cur_->height != pp_.mb_height * 16) return fail("the frame store does not match the picture size");
    if (pp_.picture_coding_type < 1 || pp_.picture_coding_type > 3) return fail("bad picture_coding_type");
    if (pp_.field_pic && (pp_.mb_height & 1)) return fail("odd frame height in macroblocks with field pictures");
    for (const SliceData& s : slices) decode_slice(s);
    return ok_;
  }

 private:
  bool fail(const char* what) {
    if (ok_ && err_) *err_ = what;
    ok_ = false;
    return false;
  }

  // ---- slice and macroblock layers (6.2.4, 6.2.5) ----
  void decode_slice(const SliceData& s) {
    if (s.size < 5 || s.data[0] || s.data[1] || s.data[2] != 1 || s.data[3] < 1 || s.data[3] > 0xAF) {
      fail("not a slice");
      return;
    }
    Bits b(s.data + 4, s.size - 4);
    int row = s.data[3] - 1;
    if (pp_.mb_height * 16 > 2800) row += static_cast<int>(b.get(3)) << 7;
    if (row >= mbh_) {
      fail("slice below the picture");
      return;
    }
    qcode_ = static_cast<int>(b.get(5));
    if (b.get(1)) {   // slice_extension_flag: intra_slice, slice_picture_id_enable, slice_picture_id, then the extra bytes
      b.skip(1 + 1 + 6);
      while (b.get(1)) b.skip(8);
    }
    int addr = row * mbw_ - 1;
    const int end_addr = (row + 1) * mbw_;
    reset_dc();
    pmv_ = {};
    bool first = true;
    while (!b.over()) {
      int inc = 0;
      for (;;) {
        const uint32_t pk = b.peek(11);
        if (pk == 0x8) {   // macroblock_escape
          b.skip(11);
          inc += 33;
          if (inc > 20000) break;
          continue;
        }
        const Lut& l = tables().inc;
        const uint32_t pk11 = b.peek(l.bits);
        if (!l.idx[pk11]) {
          fail("bad macroblock_address_increment");
          return;
        }
        inc += kMbAddrIncVlc[l.idx[pk11] - 1].value;
        b.skip(l.len[pk11]);
        break;
      }
      if (inc <= 0 || b.over()) {
        fail("bad macroblock_address_increment");
        return;
      }
      if (first) {
        addr += inc;
      } else {
        for (int k = 1; k < inc; ++k) {
          if (++addr >= end_addr) {
            fail("skipped macroblocks run off the row");
            return;
          }
          skipped_mb(addr);
        }
        ++addr;
      }
      if (addr >= end_addr || addr < row * mbw_) {
        fail("macroblock address outside the slice's row");
        return;
      }
      first = false;
      if (!macroblock(b, addr)) return;
      if (b.peek(23) == 0) break;   // the next start code (or the end of the data)
    }
  }

  void reset_dc() { dc_pred_[0] = dc_pred_[1] = dc_pred_[2] = 128 << pp_.intra_dc_precision; }

  struct MbMode {
    bool intra = false, fwd = false, bwd = false;
    int motion_type = 0;          // frame pictures: 1 field, 2 frame, 3 dual prime; field pictures: 1 field, 2 16x8, 3 dual prime
    int select[2][2] = {};        // motion_vertical_field_select[r][s]
    Mv vec[4][2];                 // vector'[r][s]; r 2 and 3: derived dual prime vectors
  };

  bool macroblock(Bits& b, int addr) {
    const int mbx = addr % mbw_, mby = addr / mbw_;
    // macroblock_type
    const MbType* tab = pp_.picture_coding_type == 1 ? kMbTypeI : pp_.picture_coding_type == 2 ? kMbTypeP : kMbTypeB;
    const int n = pp_.picture_coding_type == 1 ? 2 : pp_.picture_coding_type == 2 ? 7 : 11;
    const MbType* t = nullptr;
    for (int i = 0; i < n; ++i)
      if (b.peek(tab[i].len) == tab[i].code) {
        t = &tab[i];
        break;
      }
    if (!t) return fail("bad macroblock_type");
    b.skip(t->len);
    MbMode m;
    m.intra = t->intra;
    m.fwd = t->fwd;
    m.bwd = t->bwd;
    const bool frame_pic = !pp_.field_pic;
    int dct_type = 0;
    if (m.fwd || m.bwd) {
      if (frame_pic) {
        m.motion_type = pp_.frame_pred_frame_dct ? 2 : static_cast<int>(b.get(2));
      } else {
        m.motion_type = static_cast<int>(b.get(2));
      }
      if (m.motion_type == 0) return fail("reserved motion type");
    } else {
      m.motion_type = frame_pic ? 2 : 1;   // assumed frame-based / field-based
    }
    if (frame_pic && !pp_.frame_pred_frame_dct && (m.intra || t->pattern)) dct_type = static_cast<int>(b.get(1));
    if (t->quant) {
      qcode_ = static_cast<int>(b.get(5));
    }
    // motion vectors (6.2.5.2)
    const bool conceal = m.intra && pp_.concealment_motion_vectors;
    if (m.fwd || conceal) {
      if (!motion_vectors(b, 0, m)) return false;
    }
    if (m.bwd) {
      if (!motion_vectors(b, 1, m)) return false;
    }
    if (conceal) b.skip(1);   // marker_bit
    int cbp = 0;
    if (t->pattern) {
      const Lut& l = tables().cbp;
      const uint32_t pk = b.peek(l.bits);
      if (!l.idx[pk]) return fail("bad coded_block_pattern");
      cbp = kCbpVlc[l.idx[pk] - 1].value;
      b.skip(l.len[pk]);
    }
    if (m.intra) cbp = 63;
    // motion vector predictor housekeeping (7.6.3.3, 7.6.3.4, 7.6.3.5)
    if (m.intra) {
      if (!conceal) {
        pmv_ = {};
      } else {
        pmv_.v[1][0][0] = pmv_.v[0][0][0];
        pmv_.v[1][0][1] = pmv_.v[0][0][1];
      }
    } else {
      if (!m.fwd && !m.bwd) {   // P picture, no motion vector coded: zero vectors, predictors reset
        pmv_ = {};
        m.fwd = true;
        m.motion_type = frame_pic ? 2 : 1;
        m.vec[0][0] = {};
        m.select[0][0] = parity_ >= 0 ? parity_ : 0;
      }
    }
#ifdef VGPU_MPEG2_DEBUG_BLOCKS
    if (getenv("M2TRACE"))
      fprintf(stderr, "pic type %d struct %d mb (%d,%d) intra %d fwd %d bwd %d mt %d cbp %d dct %d v00 (%d,%d) v01 (%d,%d) v10 (%d,%d) v11 (%d,%d) v20 (%d,%d) v30 (%d,%d) sel %d %d %d %d\n",
              pp_.picture_coding_type, pp_.field_pic ? (pp_.bottom_field ? 2 : 1) : 3, mbx, mby, m.intra, m.fwd, m.bwd, m.motion_type, cbp, dct_type, m.vec[0][0].x,
              m.vec[0][0].y, m.vec[0][1].x, m.vec[0][1].y, m.vec[1][0].x, m.vec[1][0].y, m.vec[1][1].x, m.vec[1][1].y, m.vec[2][0].x, m.vec[2][0].y, m.vec[3][0].x,
              m.vec[3][0].y, m.select[0][0], m.select[1][0], m.select[0][1], m.select[1][1]);
#endif
    if (!m.intra) {
      last_fwd_ = m.fwd;
      last_bwd_ = m.bwd;
      last_mode_ = m;
    }
    if (m.intra) {
      // no prediction
    } else {
      form_prediction(mbx, mby, m);
    }
    // blocks
    if (m.intra || cbp) {
      int16_t blk[64];
      for (int i = 0; i < 6; ++i) {
        if (!(cbp & (32 >> i))) {
          if (!m.intra) {
            // not coded: nothing to add; a non-intra macroblock resets the DC predictors below
          }
          continue;
        }
        if (!block(b, i, m.intra, blk)) return false;
#ifdef VGPU_MPEG2_DEBUG_BLOCKS
        if (g_block_dump) {
          const int32_t hdr[4] = {mbx, mby, i, m.intra};
          fwrite(hdr, sizeof hdr, 1, g_block_dump);
          fwrite(blk, sizeof(int16_t), 64, g_block_dump);
        }
#endif
        idct(blk);
        put_block(mbx, mby, i, dct_type, blk, m.intra);
      }
    }
    if (!m.intra) reset_dc();
    return !b.over() || fail("macroblock runs past the slice data");
  }

  void skipped_mb(int addr) {
    const int mbx = addr % mbw_, mby = addr / mbw_;
    reset_dc();
    MbMode m;
    if (pp_.picture_coding_type == 2) {
      pmv_ = {};
      m.fwd = true;
      m.motion_type = pp_.field_pic ? 1 : 2;
      m.select[0][0] = parity_ >= 0 ? parity_ : 0;
    } else if (pp_.picture_coding_type == 3) {
      m = last_mode_;
      m.intra = false;
      m.fwd = last_fwd_;
      m.bwd = last_bwd_;
      m.motion_type = pp_.field_pic ? 1 : 2;
      // the vectors are the predictors (7.6.6.3, 7.6.6.4)
      for (int s = 0; s < 2; ++s) {
        m.vec[0][s] = {pmv_.v[0][s][0], pmv_.v[0][s][1]};
        m.select[0][s] = parity_ >= 0 ? parity_ : 0;
      }
    } else {
      return;   // not allowed in I pictures
    }
    form_prediction(mbx, mby, m);
  }

  struct Pmv {
    int v[2][2][2] = {};   // [r][s][t]
  };

  bool motion_vectors(Bits& b, int s, MbMode& m) {
    const bool frame_pic = !pp_.field_pic;
    int count, dmv = 0;
    bool mv_field;
    if (frame_pic) {
      if (m.motion_type == 1) count = 2, mv_field = true;
      else if (m.motion_type == 2) count = 1, mv_field = false;
      else count = 1, mv_field = true, dmv = 1;
    } else {
      if (m.motion_type == 1) count = 1, mv_field = true;
      else if (m.motion_type == 2) count = 2, mv_field = true;
      else count = 1, mv_field = true, dmv = 1;
    }
    int dm[2] = {0, 0};
    for (int r = 0; r < count; ++r) {
      if (mv_field && (count == 2 || !dmv)) m.select[r][s] = static_cast<int>(b.get(1));
      int vec[2];
      for (int t = 0; t < 2; ++t) {
        // motion_code
        const Lut& l = tables().motion;
        const uint32_t pk = b.peek(l.bits);
        if (!l.idx[pk]) return fail("bad motion_code");
        const int code = kMotionVlc[l.idx[pk] - 1].value;
        b.skip(l.len[pk]);
        const int fc = pp_.f_code[s][t];
        if (fc < 1 || fc > 9) return fail("bad f_code");
        const int r_size = fc - 1, f = 1 << r_size;
        const int high = 16 * f - 1, low = -16 * f, range = 32 * f;
        int delta;
        if (f == 1 || code == 0) {
          delta = code;
        } else {
          const int residual = static_cast<int>(b.get(r_size));
          delta = (std::abs(code) - 1) * f + residual + 1;
          if (code < 0) delta = -delta;
        }
        int prediction = pmv_.v[r][s][t];
        const bool halve = mv_field && t == 1 && frame_pic;
        if (halve) prediction >>= 1;   // DIV
        int v = prediction + delta;
        if (v < low) v += range;
        if (v > high) v -= range;
        pmv_.v[r][s][t] = halve ? v * 2 : v;
        vec[t] = v;
        if (dmv) {
          // dmvector (Table B.11)
          int d;
          if (b.get(1) == 0) d = 0;
          else d = b.get(1) ? -1 : 1;
          dm[t] = d;
        }
      }
      m.vec[r][s] = {vec[0], vec[1]};
    }
    // predictors (Tables 7-9 and 7-10)
    if (count == 1 && !dmv) {
      pmv_.v[1][s][0] = pmv_.v[0][s][0];
      pmv_.v[1][s][1] = pmv_.v[0][s][1];
    } else if (dmv) {
      pmv_.v[1][0][0] = pmv_.v[0][0][0];
      pmv_.v[1][0][1] = pmv_.v[0][0][1];
    }
    if (dmv) {
      // dual prime additional arithmetic (7.6.3.6)
      const Mv v0 = m.vec[0][0];
      if (frame_pic) {
        const int m_top = pp_.top_field_first ? 1 : 3;   // m[1][0]: from the bottom field into the top
        const int m_bot = pp_.top_field_first ? 3 : 1;   // m[0][1]
        m.vec[2][0] = {div2_away(v0.x * m_top) + dm[0], div2_away(v0.y * m_top) - 1 + dm[1]};
        m.vec[3][0] = {div2_away(v0.x * m_bot) + dm[0], div2_away(v0.y * m_bot) + 1 + dm[1]};
      } else {
        const int e = pp_.bottom_field ? 1 : -1;
        m.vec[2][0] = {div2_away(v0.x) + dm[0], div2_away(v0.y) + e + dm[1]};
      }
    }
    return true;
  }

  // ---- prediction (7.6) ----
  const Frame* ref_for(int s, int ref_parity) const {
    const Frame* f;
    if (pp_.picture_coding_type == 3) {
      f = s == 0 ? fwd_ : bwd_;
    } else if (pp_.field_pic && pp_.second_field && ref_parity != parity_) {
      f = cur_;   // the first field of this frame is the most recent field of the other parity
    } else {
      f = fwd_;
    }
    return f ? f : &gray_;
  }

  // One prediction of a block of a macroblock (the luma block of luma_h rows starting at row0 of the macroblock's rows in the destination
  // view, and the chroma blocks that go with it): from `ref` read as a frame (ref_parity -1) or as one of its fields, into the destination
  // picture or, when it is a frame picture predicted field by field, into the field dest_parity of it.
  void predict_one(const Frame* ref, int ref_parity, int dest_parity, int mbx, int mby, int row0, int luma_h, Mv v, bool average) {
    for (int c = 0; c < 3; ++c) {
      View dv = dest_[c];
      if (dest_parity >= 0) dv = plane_view(cur_, c, dest_parity);
      const int bw = c == 0 ? 16 : 8;
      const int bh = c == 0 ? luma_h : luma_h / 2;
      const int x = mbx * bw;
      const int y = c == 0 ? mby * (dest_parity >= 0 ? 8 : 16) + row0 : mby * (dest_parity >= 0 ? 4 : 8) + row0 / 2;
      if (ref == &gray_) {
        predict_gray(dv, x, y, bw, bh, average);
        continue;
      }
      const View rv = plane_view(ref, c, ref_parity);
      int mx = v.x, my = v.y;
      if (c > 0) {
        mx /= 2;   // 7.6.3.7: truncated towards zero
        my /= 2;
      }
      predict_block(rv, x, y, bw, bh, mx, my, dv.p + static_cast<ptrdiff_t>(y) * dv.stride + x, dv.stride, average);
    }
  }

  void predict_gray(const View& dv, int x, int y, int bw, int bh, bool average) {
    for (int j = 0; j < bh; ++j) {
      uint8_t* o = dv.p + static_cast<ptrdiff_t>(y + j) * dv.stride + x;
      for (int i = 0; i < bw; ++i) o[i] = static_cast<uint8_t>(average ? (o[i] + 128 + 1) >> 1 : 128);
    }
  }

  void form_prediction(int mbx, int mby, const MbMode& m) {
    const bool frame_pic = !pp_.field_pic;
    if (frame_pic) {
      if (m.motion_type == 2) {   // frame prediction
        bool first = true;
        for (int s = 0; s < 2; ++s) {
          if (!(s == 0 ? m.fwd : m.bwd)) continue;
          predict_one(ref_for(s, -1), -1, -1, mbx, mby, 0, 16, m.vec[0][s], !first);
          first = false;
        }
      } else if (m.motion_type == 1) {   // field prediction in a frame picture: one 16x8 block per field
        for (int r = 0; r < 2; ++r) {
          bool first = true;
          for (int s = 0; s < 2; ++s) {
            if (!(s == 0 ? m.fwd : m.bwd)) continue;
            const int sel = m.select[r][s];
            predict_one(ref_for(s, sel), sel, r, mbx, mby, 0, 8, m.vec[r][s], !first);
            first = false;
          }
        }
      } else {   // dual prime
        for (int p = 0; p < 2; ++p) {
          predict_one(ref_for(0, p), p, p, mbx, mby, 0, 8, m.vec[0][0], false);
          predict_one(ref_for(0, 1 - p), 1 - p, p, mbx, mby, 0, 8, m.vec[2 + p][0], true);
        }
      }
    } else {
      if (m.motion_type == 1) {   // field prediction
        bool first = true;
        for (int s = 0; s < 2; ++s) {
          if (!(s == 0 ? m.fwd : m.bwd)) continue;
          const int sel = m.select[0][s];
          predict_one(ref_for(s, sel), sel, -1, mbx, mby, 0, 16, m.vec[0][s], !first);
          first = false;
        }
      } else if (m.motion_type == 2) {   // 16x8 motion compensation
        for (int r = 0; r < 2; ++r) {
          bool first = true;
          for (int s = 0; s < 2; ++s) {
            if (!(s == 0 ? m.fwd : m.bwd)) continue;
            const int sel = m.select[r][s];
            predict_one(ref_for(s, sel), sel, -1, mbx, mby, r * 8, 8, m.vec[r][s], !first);
            first = false;
          }
        }
      } else {   // dual prime
        predict_one(ref_for(0, parity_), parity_, -1, mbx, mby, 0, 16, m.vec[0][0], false);
        predict_one(ref_for(0, 1 - parity_), 1 - parity_, -1, mbx, mby, 0, 16, m.vec[2][0], true);
      }
    }
  }

  // ---- blocks (6.2.6, 7.2, 7.4) ----
  bool block(Bits& b, int i, bool intra, int16_t* out) {
    int qf[64];
    std::memset(qf, 0, sizeof qf);
    const Tables& T = tables();
    int n = 0;
    const int cc = i < 4 ? 0 : i - 3;
    if (intra) {
      const DcSize* tab = cc == 0 ? kDcLuma : kDcChroma;
      int size = -1;
      for (int k = 0; k < 12; ++k)
        if (b.peek(tab[k].len) == tab[k].code) {
          size = tab[k].size;
          b.skip(tab[k].len);
          break;
        }
      if (size < 0) return fail("bad dct_dc_size");
      int diff = 0;
      if (size) {
        const int d = static_cast<int>(b.get(size));
        const int half = 1 << (size - 1);
        diff = d >= half ? d : d + 1 - 2 * half;
      }
      dc_pred_[cc] += diff;
      qf[0] = dc_pred_[cc];
      n = 1;
    }
    const Lut& lut = intra && pp_.intra_vlc_format ? T.dct1 : T.dct0;
    const DctEntry* ent = intra && pp_.intra_vlc_format ? kDctOneVlc : kDctZeroVlc;
    bool first = !intra;
    for (;;) {
      int run, level;
      if (first && b.peek(1) == 1) {   // Table B.14, Note 3
        b.skip(1);
        run = 0;
        level = 1;
        if (b.get(1)) level = -level;
      } else {
        const uint32_t pk = b.peek(lut.bits);
        if (!lut.idx[pk]) return fail("bad DCT coefficient code");
        const DctEntry& e = ent[lut.idx[pk] - 1];
        b.skip(lut.len[pk]);
        if (e.run == -1) break;
        if (e.run == -2) {
          run = static_cast<int>(b.get(6));
          level = static_cast<int>(b.get(12));
          if (level >= 2048) level -= 4096;
          if (level == 0 || level == -2048) return fail("forbidden escape level");
        } else {
          run = e.run;
          level = e.level;
          if (b.get(1)) level = -level;
        }
      }
      first = false;
      n += run;
      if (n > 63) return fail("DCT coefficient index past the block");
      qf[scan_[n]] = level;
      ++n;
      if (b.over()) return fail("block runs past the slice data");
      if (n > 63) {
        // the end of block code still follows
        const uint32_t pk = b.peek(lut.bits);
        if (lut.idx[pk] && ent[lut.idx[pk] - 1].run == -1) b.skip(lut.len[pk]);
        break;
      }
    }
    // inverse quantisation (7.4)
    const int qs = kQuantScale[pp_.q_scale_type ? 1 : 0][qcode_ & 31];
#ifdef VGPU_MPEG2_DEBUG_BLOCKS
    if (g_block_dump) {
      int16_t q16[64];
      for (int k = 0; k < 64; ++k) q16[k] = static_cast<int16_t>(qf[k]);
      const int32_t hdr[4] = {-1, qs, intra, pp_.intra_dc_precision};
      fwrite(hdr, sizeof hdr, 1, g_block_dump);
      fwrite(q16, sizeof(int16_t), 64, g_block_dump);
    }
#endif
    const uint8_t* W = intra ? pp_.intra_matrix : pp_.inter_matrix;
    int sum = 0;
    int f[64];
    for (int k = 0; k < 64; ++k) {
      int v = qf[k];
      if (k == 0 && intra) {
        v = v * (8 >> pp_.intra_dc_precision);
      } else if (v) {
        if (intra) v = (v * W[k] * qs * 2) / 32;
        else v = (((v * 2) + (v > 0 ? 1 : -1)) * W[k] * qs) / 32;
      }
      v = std::min(std::max(v, -2048), 2047);
      f[k] = v;
      sum += v;
    }
    if (!(sum & 1)) f[63] ^= 1;
    for (int k = 0; k < 64; ++k) out[k] = static_cast<int16_t>(f[k]);
    return true;
  }

  void put_block(int mbx, int mby, int i, int dct_type, const int16_t* blk, bool intra) {
    int c, bx, by, stride_mul = 1, row_off = 0;
    if (i < 4) {
      c = 0;
      bx = mbx * 16 + (i & 1) * 8;
      if (dct_type && !pp_.field_pic) {
        by = mby * 16;
        row_off = i >> 1;
        stride_mul = 2;
      } else {
        by = mby * 16 + (i >> 1) * 8;
      }
    } else {
      c = i - 3;
      bx = mbx * 8;
      by = mby * 8;
    }
    const View& d = dest_[c];
    uint8_t* base = d.p + static_cast<ptrdiff_t>(by + row_off) * d.stride + bx;
    const ptrdiff_t st = static_cast<ptrdiff_t>(d.stride) * stride_mul;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        int v = blk[y * 8 + x] + (intra ? 0 : base[y * st + x]);
        base[y * st + x] = static_cast<uint8_t>(std::min(std::max(v, 0), 255));
      }
  }

  Frame* cur_;
  const Frame* fwd_;
  const Frame* bwd_;
  PicParams pp_;
  std::string* err_;
  bool ok_ = true;
  int mbw_ = 0, mbh_ = 0, parity_ = -1;
  const uint8_t* scan_ = nullptr;
  View dest_[3];
  int qcode_ = 1;
  int dc_pred_[3] = {};
  Pmv pmv_;
  bool last_fwd_ = false, last_bwd_ = false;
  MbMode last_mode_;
  Frame gray_;
};

}  // namespace

bool decode_picture(Frame* cur, const Frame* fwd, const Frame* bwd, const PicParams& pp, const std::vector<SliceData>& slices, std::string* error) {
  if (error) error->clear();
  Decoder d(cur, fwd, bwd, pp, error);
  return d.run(slices);
}

}  // namespace vgpu_mpeg2
