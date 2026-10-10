// See nvenc_h264_enc.hpp.
#include "nvenc_h264_enc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "h264_decode.hpp"
#include "nvenc_cabac.hpp"
#include "nvenc_h264.hpp"

namespace vgpu_nvenc {
namespace {

#include "h264_tables.inc"
#include "nvenc_h264_xform.inc"
#include "nvenc_h264_cabac.inc"

// ---- what the neighbours of a macroblock, and the entropy coder's contexts, need to know about it -------------------------------------------------

enum MbKind : uint8_t { kKNone = 0, kKI4, kKI8, kKI16, kKInter };

struct MbInfo {
  int16_t slice = -1;          // the slice the macroblock was coded in; -1: not coded yet
  uint8_t kind = kKNone;
  bool skip = false, direct16 = false, t8 = false;
  uint8_t direct8 = 0;         // 8x8 blocks whose motion is inferred (B_Skip, B_Direct_16x16, direct sub-macroblocks)
  uint8_t cbp = 0;             // luma bits 0..3, chroma << 4
  uint8_t chroma_mode = 0;
  uint8_t cbf_dc = 0;          // coded_block_flag of the DC blocks: bit 0 Intra16x16 luma, 1 Cb, 2 Cr
  int8_t qp = 0;
  uint8_t nz[3][16];           // TotalCoeff of the luma 4x4 blocks (raster order) and of the Cb and Cr AC blocks (2x2 raster)
  int8_t ipred[16];            // Intra4x4 / Intra8x8 prediction modes by raster 4x4 block; 2 where the macroblock is not Intra_NxN
  int8_t ref[2][4];            // ref_idx by list and raster 8x8 block; -1: the list is not used
  int16_t mv[2][4][2];
  uint8_t mvd[2][4][2];        // absolute mvd, saturated
  MbInfo() { clear(); }
  void clear() {
    slice = -1;
    kind = kKNone;
    skip = direct16 = t8 = false;
    direct8 = 0;
    cbp = chroma_mode = cbf_dc = 0;
    qp = 0;
    std::memset(nz, 0, sizeof nz);
    std::memset(ipred, 2, sizeof ipred);
    std::memset(ref, -1, sizeof ref);
    std::memset(mv, 0, sizeof mv);
    std::memset(mvd, 0, sizeof mvd);
  }
  bool intra() const { return kind == kKI4 || kind == kKI8 || kind == kKI16; }
};

enum MbType { kMbI4 = 0, kMbI8, kMbI16, kMbInter, kMbSkip, kMbDirect };

// The decisions and levels of one macroblock.
struct MbCode {
  int type = kMbI4;
  bool t8 = false;
  int i16 = 0, chroma = 0;      // Intra16x16PredMode, intra_chroma_pred_mode
  int i4[16] = {};              // Intra4x4PredMode by luma4x4BlkIdx
  int i8[4] = {};               // Intra8x8PredMode by luma8x8BlkIdx
  int cbp_l = 0, cbp_c = 0;
  // inter: the partitioning (syntax), and the motion of each 8x8 block
  int shape = 0;                // 0 16x16, 1 16x8, 2 8x16, 3 8x8
  uint8_t sub_direct = 0;       // B_8x8: the sub-macroblocks predicted as direct
  int8_t ref[2][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};   // ref_idx by list and 8x8 block; -1: list not used
  int16_t mv[2][4][2] = {};
  // levels
  int dc[16] = {};              // Intra16x16 DC levels, scan order
  int lv[16][16] = {};          // luma levels by blkIdx, scan order (Intra16x16: AC, scan positions 1..15 at 0..14)
  int lv8[4][64] = {};          // 8x8 luma levels by luma8x8BlkIdx, scan order
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

// A reference picture as the motion search and the prediction see it.
struct RefPic {
  RefPlanes rp;
  std::vector<uint8_t> u, v;
  int chroma_offset = 0;   // Table 8-10: added to the vertical chroma vector when the field's parity differs from the current one
  int poc = 0;
  const std::vector<MbInfo>* motion = nullptr;   // the picture's macroblocks (direct prediction reads the co-located motion)
};

// A decoded picture kept as a reference.
struct DpbPic {
  std::shared_ptr<vgpu_h264::Frame> frame;
  int frame_num = 0, poc = 0;
  std::vector<MbInfo> motion;
};

enum SliceKind { kSliceP = 0, kSliceB = 1, kSliceI = 2 };

}  // namespace

// ---- the encoder ---------------------------------------------------------------------------------------------

struct H264Encoder::Impl {
  int w, h, fps_num, fps_den, profile;
  H264Options opt;
  bool deblock;
  int mw, mh, ys, cs, cw, ch;     // macroblocks, coded luma/chroma strides, visible chroma size
  H264Stream stream;
  bool cabac = false, t8_mode = false, b_mode = false;

  std::vector<uint8_t> sy, su, sv;   // the picture, padded to whole macroblocks
  std::vector<uint8_t> ry, ru, rv;   // reconstruction before the loop filter

  // reference pictures: the decoded pictures kept for prediction, most recently coded first
  std::vector<std::shared_ptr<DpbPic>> dpb;
  bool have_ref = false;
  int frame_num = 0, ref_frame_num = 0, uid = 0;   // frame_num of the picture being coded; of the last reference picture
  int poc_counter = 0;                             // display order of the next picture when the caller gives none
  bool decode_failed_reported = false;
  int cur_poc = 0;                                 // PicOrderCnt of the picture being coded

  // The reference pictures of the picture being coded (RefPicList0 / 1). In field coding: fields of the previous frame and of the current one. Each is a
  // picture of this encoder's size, with its interpolation planes.
  std::vector<RefPic> lists[2];
  std::vector<std::shared_ptr<DpbPic>> list_src[2];   // the pictures the lists were built from
  int num_active[2] = {0, 0};

  // Field coding (interlaced test streams): this encoder's pictures are fields of a frame `frame_h` rows high.
  bool field_coding = false;
  int frame_h = 0;
  struct FieldCtx {
    bool bottom = false, second = false;
  } fld;
  std::unique_ptr<vgpu_h264::Frame> fr_cur, fr_prev;   // the frame stores: the one being coded (both fields), the previous one
  int cur_first_parity = 0;                              // parity of the first field of fr_cur
  bool cur_idr = false;                                  // the frame being coded starts with an IDR field: nothing before it is a reference
  int field_count = 0;                                   // frames coded since the last IDR picture (for picture order counts)

  // per picture
  int qp = 26, qpc = 26;
  double lambda = 1, lam_sad = 1;
  PicType type = PicType::kIdr;
  int slice_kind = kSliceI;
  std::vector<MbInfo> mbs;
  std::vector<int> slice_start;     // first macroblock address of each slice of the picture
  int cur_slice = 0;
  int cur_mx = 0, cur_my = 0;
  const MbInfo *nA = nullptr, *nB = nullptr, *nC = nullptr, *nD = nullptr;   // neighbouring macroblocks of the current one (null: not available)
  EncStats st;

  // the entropy coder of the slice being written (the contexts of the CABAC one decide the cost of candidates)
  BitWriter* bw = nullptr;
  H264BitCoder* bc = nullptr;
  int skip_run = 0;

  Impl(int width, int height, int fn, int fd, int prof, const H264Options& o, bool fields)
      : w(width), h(fields ? height / 2 : height), fps_num(fn), fps_den(fd), profile(prof), opt(o), deblock(o.deblock), field_coding(fields), frame_h(height) {
    if (prof < 77 || fields) {
      opt.cabac = false;
      opt.max_b = 0;
    }
    if (prof < 100 || fields) opt.transform8x8 = false;
    if (fields) {
      opt.slice_mode = 0;
      opt.slice_data = 0;
      opt.num_ref = 2;
    }
    opt.num_ref = clip3(1, 4, opt.num_ref);
    opt.max_b = clip3(0, 8, opt.max_b);
    cabac = opt.cabac;
    t8_mode = opt.transform8x8;
    b_mode = opt.max_b > 0;
    mw = (w + 15) / 16;
    mh = (h + 15) / 16;
    ys = mw * 16;
    cs = mw * 8;
    cw = (w + 1) / 2;
    ch = (h + 1) / 2;
    stream.width = w;
    stream.height = frame_h;
    stream.interlaced = fields;
    stream.num_ref_frames = fields ? 2 : std::max(opt.num_ref, b_mode ? 2 : 1);
    stream.fps_num = fn;
    stream.fps_den = fd;
    stream.profile_idc = prof;
    stream.cabac = cabac;
    stream.transform_8x8 = t8_mode;
    stream.num_reorder = b_mode ? (opt.max_b >= 2 ? 2 : 1) : -1;
    ry.assign(static_cast<size_t>(ys) * mh * 16, 0);
    ru.assign(static_cast<size_t>(cs) * mh * 8, 128);
    rv = ru;
  }

  // the coefficient scan: field pictures scan the other way (8.5.6)
  const uint8_t* scan4() const { return field_coding ? kField4x4 : kZigzag4x4; }
  const uint8_t* scan8() const { return field_coding ? kField8x8 : kZigzag8x8; }

  // ---- neighbours ---------------------------------------------------------------------------------------------------------

  const MbInfo* nb_at(int mx, int my) const {
    if (mx < 0 || my < 0 || mx >= mw || my >= mh) return nullptr;
    const MbInfo& m = mbs[static_cast<size_t>(my) * mw + mx];
    return m.slice == cur_slice ? &m : nullptr;
  }
  void setup_neighbours(int mx, int my) {
    cur_mx = mx;
    cur_my = my;
    nA = nb_at(mx - 1, my);
    nB = nb_at(mx, my - 1);
    nC = nb_at(mx + 1, my - 1);
    nD = nb_at(mx - 1, my - 1);
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

  // The interpolation planes of a reference picture: rows `row0 + step * y` of the frame store f, for a frame (step 1) or one field of it (step 2, row0 = the
  // parity).
  void build_ref(RefPic& r, const vgpu_h264::Frame& f, int row0, int step, int chroma_offset) const {
    r.chroma_offset = chroma_offset;
    RefPlanes& rp = r.rp;
    const int W = ys, H = mh * 16;
    rp.w = W;
    rp.h = H;
    rp.stride = W + 2 * kPad;
    const size_t n = static_cast<size_t>(rp.stride) * (H + 2 * kPad);
    rp.g.assign(n, 0);
    for (int y = -kPad; y < H + kPad; ++y)
      for (int x = -kPad; x < W + kPad; ++x)
        rp.g[static_cast<size_t>(y + kPad) * rp.stride + (x + kPad)] = f.y[static_cast<size_t>(row0 + step * clip3(0, H - 1, y)) * f.stride_y + clip3(0, W - 1, x)];
    r.u.resize(static_cast<size_t>(cs) * mh * 8);
    r.v.resize(r.u.size());
    for (int y = 0; y < mh * 8; ++y) {
      std::memcpy(&r.u[static_cast<size_t>(y) * cs], &f.u[static_cast<size_t>(row0 + step * y) * f.stride_c], static_cast<size_t>(cs));
      std::memcpy(&r.v[static_cast<size_t>(y) * cs], &f.v[static_cast<size_t>(row0 + step * y) * f.stride_c], static_cast<size_t>(cs));
    }
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

  // The luma prediction of the bw x bh block at (px, py) of the picture, from reference r at quarter-sample vector (vx, vy), into out (stride `stride`).
  void pred_luma(const RefPic& r, int px, int py, int bw_, int bh_, int vx, int vy, uint8_t* out, int stride) const {
    const RefPlanes& rp = r.rp;
    const int fx = vx & 3, fy = vy & 3;
    const int X0 = px + (vx >> 2), Y0 = py + (vy >> 2);
    for (int y = 0; y < bh_; ++y)
      for (int x = 0; x < bw_; ++x) {
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
        out[y * stride + x] = static_cast<uint8_t>(v);
      }
  }

  // The chroma predictions (8.4.2.2.2) of both planes for the bw x bh luma block at (px, py): blocks of bw/2 x bh/2 samples.
  void pred_chroma(const RefPic& r, int px, int py, int bw_, int bh_, int vx, int vy_luma, uint8_t* out_u, uint8_t* out_v, int stride) const {
    const int vy = vy_luma + r.chroma_offset;
    const int fx = vx & 7, fy = vy & 7;
    const int X0 = px / 2 + (vx >> 3), Y0 = py / 2 + (vy >> 3);
    const int W = mw * 8, H = mh * 8;
    for (int c = 0; c < 2; ++c) {
      const std::vector<uint8_t>& p = c ? r.v : r.u;
      uint8_t* out = c ? out_v : out_u;
      for (int y = 0; y < bh_ / 2; ++y)
        for (int x = 0; x < bw_ / 2; ++x) {
          auto at = [&](int dx, int dy) { return static_cast<int>(p[static_cast<size_t>(clip3(0, H - 1, Y0 + y + dy)) * cs + clip3(0, W - 1, X0 + x + dx)]); };
          out[y * stride + x] = static_cast<uint8_t>(((8 - fx) * (8 - fy) * at(0, 0) + fx * (8 - fy) * at(1, 0) + (8 - fx) * fy * at(0, 1) + fx * fy * at(1, 1) + 32) >> 6);
        }
    }
  }

  // ---- residual coding of blocks ------------------------------------------------------------------------------------

  static int64_t sse_block(const uint8_t* a, int sa, const uint8_t* b, int sb, int bw_, int bh_) {
    int64_t s = 0;
    for (int y = 0; y < bh_; ++y)
      for (int x = 0; x < bw_; ++x) {
        const int d = a[y * sa + x] - b[y * sb + x];
        s += d * d;
      }
    return s;
  }

  // One 4x4 block: the residual of src against pred, transformed and quantised into scan-order levels `lev`; the reconstruction goes to rec. Returns the
  // number of non-zero levels.
  int code_blk4(const uint8_t* src, int ss, const uint8_t* pred, int ps, uint8_t* rec, int rs, bool intra, int qpv, int* lev) {
    int r[16], wc[16], dq[16] = {};
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x) r[y * 4 + x] = src[y * ss + x] - pred[y * ps + x];
    fwd4x4(r, wc);
    const int nz = quant_block(wc, qpv, intra, 0, lev, dq, scan4());
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x) rec[y * rs + x] = pred[y * ps + x];
    if (nz) idct4_add(dq, rec, rs);
    return nz;
  }
  int code_blk8(const uint8_t* src, int ss, const uint8_t* pred, int ps, uint8_t* rec, int rs, bool intra, int qpv, int* lev) {
    int r[64], wc[64], dq[64] = {};
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) r[y * 8 + x] = src[y * ss + x] - pred[y * ps + x];
    fwd8x8(r, wc);
    const int nz = quant_block8(wc, qpv, intra, lev, dq, scan8());
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) rec[y * rs + x] = pred[y * ps + x];
    if (nz) idct8_add(dq, rec, rs);
    return nz;
  }

  // ---- Intra16x16 -----------------------------------------------------------------------------------------------------

  void predict16(int mx, int my, int mode, uint8_t* pred) {
    const uint8_t* r = rec_y(mx, my);
    const bool top = nB != nullptr, left = nA != nullptr;
    int topv[17], leftv[16], corner = 128;
    for (int i = 0; i < 16; ++i) {
      topv[i] = top ? r[-ys + i] : 128;
      leftv[i] = left ? r[i * ys - 1] : 128;
    }
    if (top && left && nD) corner = r[-ys - 1];
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

  bool i16_mode_ok(int mode) const {
    switch (mode) {
      case 0: return nB != nullptr;
      case 1: return nA != nullptr;
      case 2: return true;
      default: return nA && nB && nD;
    }
  }

  // Quantises and reconstructs an Intra16x16 macroblock's luma from `pred`, into mc (levels) and the reconstruction.
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
    for (int k = 0; k < 16; ++k) {
      const int a = std::abs(had[k]) >> 1;
      const int q = (a * kMF[m][0] + 2 * f) >> (qbits + 1);
      dcl[k] = had[k] < 0 ? -q : q;
    }
    for (int i = 0; i < 16; ++i) mc.dc[i] = dcl[scan4()[i]];
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
      const int nz = quant_block(W[blk], qp, true, 1, mc.lv[blk], dq, scan4());
      any_ac |= nz != 0;
      std::memcpy(dqs[blk], dq, sizeof dq);
    }
    mc.cbp_l = any_ac ? 15 : 0;
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

  // ---- Intra4x4 and Intra8x8 ------------------------------------------------------------------------------------------

  // Neighbouring samples of an N x N luma block at offset (bx, by) of the macroblock: top_v[0] is the corner, top_v[1 + x] is p[x, -1] (x < 2N), left_v[y] is p[-1, y].
  struct NbN {
    bool top = false, left = false, corner = false, tr = false;
    int top_v[1 + 16] = {};
    int left_v[8] = {};
  };
  NbN gather_nxn(int mx, int my, int bx, int by, int N, bool tr_inside) const {
    NbN n;
    n.top = by > 0 || nB;
    n.left = bx > 0 || nA;
    n.corner = (bx > 0 && by > 0) || (bx > 0 && by == 0 && nB) || (bx == 0 && by > 0 && nA) || (bx == 0 && by == 0 && nD);
    if (by == 0) n.tr = bx + N < 16 ? nB != nullptr : nC != nullptr;
    else n.tr = bx + N < 16 && tr_inside;
    const int X = mx * 16 + bx, Y = my * 16 + by;
    if (n.corner) n.top_v[0] = ry[static_cast<size_t>(Y - 1) * ys + X - 1];
    if (n.top)
      for (int i = 0; i < N; ++i) n.top_v[1 + i] = ry[static_cast<size_t>(Y - 1) * ys + X + i];
    if (n.top && n.tr)
      for (int i = N; i < 2 * N; ++i) n.top_v[1 + i] = ry[static_cast<size_t>(Y - 1) * ys + X + i];
    else if (n.top)
      for (int i = N; i < 2 * N; ++i) n.top_v[1 + i] = n.top_v[N];
    if (n.left)
      for (int i = 0; i < N; ++i) n.left_v[i] = ry[static_cast<size_t>(Y + i) * ys + X - 1];
    return n;
  }
  static bool mode_nxn_ok(const NbN& n, int mode) {
    switch (mode) {
      case 0: case 3: case 7: return n.top;
      case 1: case 8: return n.left;
      case 2: return true;
      default: return n.top && n.left && n.corner;
    }
  }
  // The predicted Intra4x4PredMode / Intra8x8PredMode of the block at 4x4-unit position (bx4, by4) (8.3.1.1, 8.3.2.1).
  int pred_mode_nxn(const MbInfo& wk, int bx4, int by4) const {
    int ma, mb;
    if (bx4 > 0) ma = wk.ipred[by4 * 4 + bx4 - 1];
    else if (!nA) return 2;
    else ma = nA->ipred[by4 * 4 + 3];
    if (by4 > 0) mb = wk.ipred[(by4 - 1) * 4 + bx4];
    else if (!nB) return 2;
    else mb = nB->ipred[12 + bx4];
    return std::min(ma, mb);
  }
  // Intra_8x8 reference sample filtering (8.3.2.2.1): top_f[0] is the filtered corner, top_f[1 + x] p'[x, -1], left_f[y] p'[-1, y]
  static void filter8(const NbN& nb, int* top_f, int* left_f) {
    std::fill_n(top_f, 17, 128);
    std::fill_n(left_f, 8, 128);
    top_f[0] = nb.top_v[0];
    if (nb.top) {
      if (nb.corner) top_f[1] = (nb.top_v[0] + 2 * nb.top_v[1] + nb.top_v[2] + 2) >> 2;
      else top_f[1] = (3 * nb.top_v[1] + nb.top_v[2] + 2) >> 2;
      for (int x = 1; x < 15; ++x) top_f[1 + x] = (nb.top_v[x] + 2 * nb.top_v[1 + x] + nb.top_v[2 + x] + 2) >> 2;
      top_f[16] = (nb.top_v[15] + 3 * nb.top_v[16] + 2) >> 2;
    }
    if (nb.corner) {
      if (!nb.top || !nb.left) {
        if (nb.top) top_f[0] = (3 * nb.top_v[0] + nb.top_v[1] + 2) >> 2;
        else if (nb.left) top_f[0] = (3 * nb.top_v[0] + nb.left_v[0] + 2) >> 2;
        else top_f[0] = nb.top_v[0];
      } else {
        top_f[0] = (nb.top_v[1] + 2 * nb.top_v[0] + nb.left_v[0] + 2) >> 2;
      }
    }
    if (nb.left) {
      if (nb.corner) left_f[0] = (nb.top_v[0] + 2 * nb.left_v[0] + nb.left_v[1] + 2) >> 2;
      else left_f[0] = (3 * nb.left_v[0] + nb.left_v[1] + 2) >> 2;
      for (int y = 1; y < 7; ++y) left_f[y] = (nb.left_v[y - 1] + 2 * nb.left_v[y] + nb.left_v[y + 1] + 2) >> 2;
      left_f[7] = (nb.left_v[6] + 3 * nb.left_v[7] + 2) >> 2;
    }
  }

  // Intra4x4: chooses the modes, codes the 16 blocks into the reconstruction. mc.i4, mc.lv and cbp_l are set; wk.ipred receives the modes.
  void code_i4(int mx, int my, MbCode& mc, MbInfo& wk) {
    const uint8_t* s = src_y(mx, my);
    bool any[4] = {false, false, false, false};
    for (int blk = 0; blk < 16; ++blk) {
      const int bx4 = blk_x(blk), by4 = blk_y(blk);
      const int bx = bx4 * 4, by = by4 * 4;
      bool tr_inside = false;
      if (by4 > 0 && bx4 < 3) tr_inside = blk_idx(bx4 + 1, by4 - 1) < blk;
      const NbN n = gather_nxn(mx, my, bx, by, 4, tr_inside);
      const int pm = pred_mode_nxn(wk, bx4, by4);
      auto pl = [&](int y) { return y < 0 ? n.top_v[0] : n.left_v[y]; };
      auto ptc = [&](int x) { return x < 0 ? n.top_v[0] : n.top_v[1 + x]; };
      uint8_t pred[16], best_pred[16];
      int best_mode = 2;
      double best_cost = 1e30;
      for (int mode = 0; mode < 9; ++mode) {
        if (!mode_nxn_ok(n, mode)) continue;
        predict_nxn<4>(mode, n.top, n.left, ptc, pl, pred, 4);
        int d[16];
        for (int y = 0; y < 4; ++y)
          for (int x = 0; x < 4; ++x) d[y * 4 + x] = s[(by + y) * ys + bx + x] - pred[y * 4 + x];
        const double cost = satd4(d) + lam_sad * (mode == pm ? 1 : 4);
        if (cost < best_cost) {
          best_cost = cost;
          best_mode = mode;
          std::memcpy(best_pred, pred, 16);
        }
      }
      mc.i4[blk] = best_mode;
      wk.ipred[by4 * 4 + bx4] = static_cast<int8_t>(best_mode);
      uint8_t* d = rec_y(mx, my) + by * ys + bx;
      const int nz = code_blk4(s + by * ys + bx, ys, best_pred, 4, d, ys, true, qp, mc.lv[blk]);
      any[blk >> 2] |= nz != 0;
    }
    mc.cbp_l = (any[0] ? 1 : 0) | (any[1] ? 2 : 0) | (any[2] ? 4 : 0) | (any[3] ? 8 : 0);
    for (int blk = 0; blk < 16; ++blk)
      if (!((mc.cbp_l >> (blk >> 2)) & 1)) std::memset(mc.lv[blk], 0, sizeof mc.lv[blk]);
  }

  // Intra8x8: the same with four 8x8 blocks.
  void code_i8(int mx, int my, MbCode& mc, MbInfo& wk) {
    const uint8_t* s = src_y(mx, my);
    int cbp = 0;
    for (int b8 = 0; b8 < 4; ++b8) {
      const int bx = (b8 & 1) * 8, by = (b8 >> 1) * 8;
      const NbN n = gather_nxn(mx, my, bx, by, 8, b8 == 2);
      int top_f[17], left_f[8];
      filter8(n, top_f, left_f);
      const int pm = pred_mode_nxn(wk, bx / 4, by / 4);
      auto pl = [&](int y) { return y < 0 ? top_f[0] : left_f[y]; };
      auto ptc = [&](int x) { return x < 0 ? top_f[0] : top_f[1 + x]; };
      uint8_t pred[64], best_pred[64];
      int best_mode = 2;
      double best_cost = 1e30;
      for (int mode = 0; mode < 9; ++mode) {
        if (!mode_nxn_ok(n, mode)) continue;
        predict_nxn<8>(mode, n.top, n.left, ptc, pl, pred, 8);
        int d[64];
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x) d[y * 8 + x] = s[(by + y) * ys + bx + x] - pred[y * 8 + x];
        const double cost = satd_block(d, 8, 8, 8) + lam_sad * (mode == pm ? 1 : 4);
        if (cost < best_cost) {
          best_cost = cost;
          best_mode = mode;
          std::memcpy(best_pred, pred, 64);
        }
      }
      mc.i8[b8] = best_mode;
      for (int k = 0; k < 4; ++k) wk.ipred[(by / 4 + (k >> 1)) * 4 + bx / 4 + (k & 1)] = static_cast<int8_t>(best_mode);
      uint8_t* d = rec_y(mx, my) + by * ys + bx;
      const int nz = code_blk8(s + by * ys + bx, ys, best_pred, 8, d, ys, true, qp, mc.lv8[b8]);
      if (nz) cbp |= 1 << b8;
      else std::memset(mc.lv8[b8], 0, sizeof mc.lv8[b8]);
    }
    mc.cbp_l = cbp;
  }

  // Inter luma: residual of `pred` (16x16 prediction) coded with inter quantisation into mc and the reconstruction; with the 8x8 transform if mc.t8.
  void code_luma_inter(int mx, int my, const uint8_t* pred, MbCode& mc) {
    const uint8_t* s = src_y(mx, my);
    uint8_t* d = rec_y(mx, my);
    int cbp = 0;
    if (mc.t8) {
      for (int b8 = 0; b8 < 4; ++b8) {
        const int bx = (b8 & 1) * 8, by = (b8 >> 1) * 8;
        int nzc = code_blk8(s + by * ys + bx, ys, pred + by * 16 + bx, 16, d + by * ys + bx, ys, false, qp, mc.lv8[b8]);
        // a lone +-1 level high in the scan costs more bits than it is worth
        if (nzc == 1) {
          int pos = -1;
          for (int i = 0; i < 64; ++i)
            if (mc.lv8[b8][i]) pos = i;
          if (std::abs(mc.lv8[b8][pos]) == 1 && pos >= 10) {
            std::memset(mc.lv8[b8], 0, sizeof mc.lv8[b8]);
            for (int y = 0; y < 8; ++y)
              for (int x = 0; x < 8; ++x) d[(by + y) * ys + bx + x] = pred[(by + y) * 16 + bx + x];
            nzc = 0;
          }
        }
        if (nzc) cbp |= 1 << b8;
      }
      mc.cbp_l = cbp;
      return;
    }
    bool any[4] = {false, false, false, false};
    int dqs[16][16];
    for (int blk = 0; blk < 16; ++blk) {
      const int bx = blk_x(blk), by = blk_y(blk);
      int r[16], wc[16];
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) r[y * 4 + x] = s[(by * 4 + y) * ys + bx * 4 + x] - pred[(by * 4 + y) * 16 + bx * 4 + x];
      fwd4x4(r, wc);
      std::memset(dqs[blk], 0, sizeof dqs[blk]);
      const int nz = quant_block(wc, qp, false, 0, mc.lv[blk], dqs[blk], scan4());
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
    const bool top = nB != nullptr, left = nA != nullptr;
    int topv[8], leftv[8], corner = 128;
    for (int i = 0; i < 8; ++i) {
      topv[i] = top ? r[-cs + i] : 128;
      leftv[i] = left ? r[i * cs - 1] : 128;
    }
    if (top && left && nD) corner = r[-cs - 1];
    switch (mode) {
      case 0:
        for (int b = 0; b < 4; ++b) {
          const int xo = (b & 1) * 4, yo = (b >> 1) * 4;
          int st_ = 0, sl = 0;
          for (int i = 0; i < 4; ++i) {
            st_ += topv[xo + i];
            sl += leftv[yo + i];
          }
          int v;
          if ((xo == 0 && yo == 0) || (xo > 0 && yo > 0)) {
            if (top && left) v = (st_ + sl + 4) >> 3;
            else if (left) v = (sl + 2) >> 2;
            else if (top) v = (st_ + 2) >> 2;
            else v = 128;
          } else if (xo > 0) {
            if (top) v = (st_ + 2) >> 2;
            else if (left) v = (sl + 2) >> 2;
            else v = 128;
          } else {
            if (left) v = (sl + 2) >> 2;
            else if (top) v = (st_ + 2) >> 2;
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

  bool chroma_mode_ok(int mode) const {
    switch (mode) {
      case 0: return true;
      case 1: return nA != nullptr;
      case 2: return nB != nullptr;
      default: return nA && nB && nD;
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
        const int nz = quant_block(wc, qpcv, intra, 1, mc.cac[c][b], dqs[c][b], scan4());
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

  // ---- neighbour data for the entropy coders --------------------------------------------------------------------------

  int nc_of(bool av_a, int na, bool av_b, int nb) const {
    if (av_a && av_b) return (na + nb + 1) >> 1;
    if (av_a) return na;
    if (av_b) return nb;
    return 0;
  }
  // nC for the coeff_token of luma block (bx, by) (4x4 units)
  int nc_luma(const MbInfo& wk, int bx, int by) const {
    const bool av_a = bx > 0 || nA, av_b = by > 0 || nB;
    const int na = bx > 0 ? wk.nz[0][by * 4 + bx - 1] : (nA ? nA->nz[0][by * 4 + 3] : 0);
    const int nb = by > 0 ? wk.nz[0][(by - 1) * 4 + bx] : (nB ? nB->nz[0][12 + bx] : 0);
    return nc_of(av_a, na, av_b, nb);
  }
  int nc_chroma(const MbInfo& wk, int c, int bx, int by) const {
    const bool av_a = bx > 0 || nA, av_b = by > 0 || nB;
    const int na = bx > 0 ? wk.nz[1 + c][by * 2 + bx - 1] : (nA ? nA->nz[1 + c][by * 2 + 1] : 0);
    const int nb = by > 0 ? wk.nz[1 + c][(by - 1) * 2 + bx] : (nB ? nB->nz[1 + c][2 + bx] : 0);
    return nc_of(av_a, na, av_b, nb);
  }
  // CABAC coded_block_flag contexts (9.3.3.1.1.9)
  int cbf_inc_luma(const MbInfo& wk, bool intra, int bx, int by) const {
    const int ca = bx > 0 ? wk.nz[0][by * 4 + bx - 1] != 0 : (nA ? nA->nz[0][by * 4 + 3] != 0 : intra);
    const int cb = by > 0 ? wk.nz[0][(by - 1) * 4 + bx] != 0 : (nB ? nB->nz[0][12 + bx] != 0 : intra);
    return ca + 2 * cb;
  }
  int cbf_inc_chroma(const MbInfo& wk, bool intra, int c, int bx, int by) const {
    const int ca = bx > 0 ? wk.nz[1 + c][by * 2 + bx - 1] != 0 : (nA ? nA->nz[1 + c][by * 2 + 1] != 0 : intra);
    const int cb = by > 0 ? wk.nz[1 + c][(by - 1) * 2 + bx] != 0 : (nB ? nB->nz[1 + c][2 + bx] != 0 : intra);
    return ca + 2 * cb;
  }
  int cbf_inc_dc(bool intra, int bit) const {
    const int ca = nA ? (nA->cbf_dc >> bit) & 1 : intra;
    const int cb = nB ? (nB->cbf_dc >> bit) & 1 : intra;
    return ca + 2 * cb;
  }

  // The 8x8 block neighbouring the current macroblock's 8x8 block coordinates (cx, cy), cx and cy in -1..2.
  struct Nb8 {
    const MbInfo* m = nullptr;
    int b8 = 0;
  };
  Nb8 nb8_at(const MbInfo& wk, int cx, int cy) const {
    if (cx >= 0 && cy >= 0 && cx < 2 && cy < 2) return {&wk, cy * 2 + cx};
    if (cx < 0 && cy >= 0 && cy < 2) return {nA, cy * 2 + 1};
    if (cy < 0 && cx >= 0 && cx < 2) return {nB, 2 + cx};
    if (cx >= 2 && cy < 0) return {nC, 2};
    if (cx < 0 && cy < 0) return {nD, 3};
    return {};
  }
  struct NbMot {
    bool avail = false;
    int ref = -1, x = 0, y = 0;
  };
  NbMot nb_motion(const MbInfo& wk, unsigned done8, int list, int cx, int cy) const {
    NbMot r;
    const Nb8 n = nb8_at(wk, cx, cy);
    if (!n.m) return r;
    if (n.m == &wk && !((done8 >> n.b8) & 1)) return r;   // later in decoding order
    r.avail = true;
    const int rf = n.m->ref[list][n.b8];
    if (rf < 0 || n.m->intra()) return r;
    r.ref = rf;
    r.x = n.m->mv[list][n.b8][0];
    r.y = n.m->mv[list][n.b8][1];
    return r;
  }
  // 8.4.1.3: the motion vector prediction of the partition at (x, y), w x h samples, of the macroblock described by wk, for reference index `ref`.
  // `shape` and `part` select the directional rules of 16x8 and 8x16 partitions.
  void mv_pred(const MbInfo& wk, unsigned done8, int list, int ref, int x, int y, int w_, int shape, int part, int* px, int* py) const {
    const NbMot a = nb_motion(wk, done8, list, (x - 1) >> 3, y >> 3);
    NbMot b = nb_motion(wk, done8, list, x >> 3, (y - 1) >> 3);
    NbMot c = nb_motion(wk, done8, list, (x + w_) >> 3, (y - 1) >> 3);
    if (!c.avail) c = nb_motion(wk, done8, list, (x - 1) >> 3, (y - 1) >> 3);
    if (shape == 1) {
      if (part == 0 && b.ref == ref) { *px = b.x; *py = b.y; return; }
      if (part == 1 && a.ref == ref) { *px = a.x; *py = a.y; return; }
    } else if (shape == 2) {
      if (part == 0 && a.ref == ref) { *px = a.x; *py = a.y; return; }
      if (part == 1 && c.ref == ref) { *px = c.x; *py = c.y; return; }
    }
    NbMot B = b, C = c;
    if (!b.avail && !c.avail && a.avail) {
      B = a;
      C = a;
    }
    const int matches = (a.ref == ref) + (B.ref == ref) + (C.ref == ref);
    if (matches == 1) {
      const NbMot& m = a.ref == ref ? a : (B.ref == ref ? B : C);
      *px = m.x;
      *py = m.y;
    } else {
      *px = median3(a.x, B.x, C.x);
      *py = median3(a.y, B.y, C.y);
    }
  }
  // The P_Skip vector (8.4.1.1)
  void pskip_mv(int* sx, int* sy) const {
    MbInfo none;
    const NbMot a = nb_motion(none, 0, 0, -1, 0), b = nb_motion(none, 0, 0, 0, -1);
    if (!a.avail || !b.avail || (a.ref == 0 && a.x == 0 && a.y == 0) || (b.ref == 0 && b.x == 0 && b.y == 0)) {
      *sx = *sy = 0;
      return;
    }
    mv_pred(none, 0, 0, 0, 0, 0, 16, 0, 0, sx, sy);
  }

  // ---- the macroblock layer: partition syntax shared by both entropy coders ---------------------------------------------------

  // pred flags of an 8x8 block of the macroblock: bit 0 list 0, bit 1 list 1
  static int pf_of(const MbCode& mc, int b8) { return (mc.ref[0][b8] >= 0 ? 1 : 0) | (mc.ref[1][b8] >= 0 ? 2 : 0); }
  // the 8x8 blocks that start each partition of the shape, and the partition's size
  static int part_count(int shape) { return shape == 0 ? 1 : (shape == 3 ? 4 : 2); }
  static void part_geom(int shape, int p, int* x, int* y, int* pw, int* ph, int* first_b8) {
    switch (shape) {
      case 0: *x = 0; *y = 0; *pw = 16; *ph = 16; *first_b8 = 0; break;
      case 1: *x = 0; *y = 8 * p; *pw = 16; *ph = 8; *first_b8 = 2 * p; break;
      case 2: *x = 8 * p; *y = 0; *pw = 8; *ph = 16; *first_b8 = p; break;
      default: *x = (p & 1) * 8; *y = (p >> 1) * 8; *pw = 8; *ph = 8; *first_b8 = p; break;
    }
  }
  // The B macroblock type number (Table 7-14) of a coded inter macroblock.
  static int b_mb_type(const MbCode& mc) {
    if (mc.type == kMbDirect) return 0;
    if (mc.shape == 3) return 22;
    const int p0 = pf_of(mc, 0), p1 = pf_of(mc, mc.shape == 1 ? 2 : 1);
    if (mc.shape == 0) return p0;   // 1 L0, 2 L1, 3 Bi
    static const int pair[4][4] = {{0, 0, 0, 0}, {0, 0, 2, 4}, {0, 3, 1, 5}, {0, 6, 7, 8}};
    return 4 + 2 * pair[p0][p1] + (mc.shape == 2 ? 1 : 0);
  }
  static int p_mb_type(const MbCode& mc) { return mc.shape; }
  // sub_mb_type of 8x8 block b8 of a B_8x8 macroblock: 0 direct, 1 L0, 2 L1, 3 Bi
  static int b_sub_type(const MbCode& mc, int b8) { return (mc.sub_direct >> b8) & 1 ? 0 : pf_of(mc, b8); }
  // The I-slice mb_type number of an intra macroblock.
  static int intra_t_of(const MbCode& mc) {
    if (mc.type == kMbI16) return 1 + mc.i16 + 4 * mc.cbp_c + (mc.cbp_l ? 12 : 0);
    return 0;
  }
  // transform_size_8x8_flag is sent for an inter macroblock with luma coefficients when no partition is smaller than 8x8
  bool t8_flag_present_inter(const MbCode& mc) const {
    return t8_mode && mc.cbp_l != 0 && (mc.type != kMbDirect || true);
  }

  // Writes the CAVLC residual and updates wk's coefficient counts.
  template <class W>
  void write_residual_cavlc(W& w, const MbCode& mc, MbInfo& wk) {
    const bool i16 = mc.type == kMbI16;
    if (i16) write_block(w, mc.dc, 16, nc_luma(wk, 0, 0));
    if (mc.cbp_l) {
      for (int b8 = 0; b8 < 4; ++b8) {
        if (!((mc.cbp_l >> b8) & 1)) continue;
        for (int s = 0; s < 4; ++s) {
          const int blk = b8 * 4 + s;
          const int bx = blk_x(blk), by = blk_y(blk);
          int n;
          if (mc.t8) {
            int tmp[16];
            for (int i = 0; i < 16; ++i) tmp[i] = mc.lv8[b8][4 * i + s];
            n = write_block(w, tmp, 16, nc_luma(wk, bx, by));
          } else {
            n = write_block(w, mc.lv[blk], i16 ? 15 : 16, nc_luma(wk, bx, by));
          }
          wk.nz[0][by * 4 + bx] = static_cast<uint8_t>(n);
        }
      }
    }
    if (mc.cbp_c) {
      for (int c = 0; c < 2; ++c) write_block(w, mc.cdc[c], 4, -1);
      if (mc.cbp_c == 2)
        for (int c = 0; c < 2; ++c)
          for (int b = 0; b < 4; ++b) {
            const int n = write_block(w, mc.cac[c][b], 15, nc_chroma(wk, c, b & 1, b >> 1));
            wk.nz[1 + c][b] = static_cast<uint8_t>(n);
          }
    }
  }

  // Writes the CABAC residual and updates wk's coefficient counts and DC flags.
  template <class C>
  void write_residual_cabac(C& c, const MbCode& mc, MbInfo& wk) {
    const bool i16 = mc.type == kMbI16;
    const bool intra = mc.type == kMbI4 || mc.type == kMbI8 || mc.type == kMbI16;
    if (i16) {
      bool any = false;
      for (int i = 0; i < 16; ++i) any |= mc.dc[i] != 0;
      cabac_residual_block(c, 0, cbf_inc_dc(intra, 0), 16, mc.dc);
      if (any) wk.cbf_dc |= 1;
    }
    if (mc.cbp_l) {
      for (int b8 = 0; b8 < 4; ++b8) {
        if (!((mc.cbp_l >> b8) & 1)) continue;
        if (mc.t8) {
          int n = 0;
          for (int i = 0; i < 64; ++i) n += mc.lv8[b8][i] != 0;
          cabac_residual_block(c, 5, 0, 64, mc.lv8[b8]);
          const int x4 = (b8 & 1) * 2, y4 = (b8 >> 1) * 2;
          for (int k = 0; k < 4; ++k) wk.nz[0][(y4 + (k >> 1)) * 4 + x4 + (k & 1)] = static_cast<uint8_t>(n);
          continue;
        }
        for (int s = 0; s < 4; ++s) {
          const int blk = b8 * 4 + s;
          const int bx = blk_x(blk), by = blk_y(blk);
          const int maxc = i16 ? 15 : 16;
          int n = 0;
          for (int i = 0; i < maxc; ++i) n += mc.lv[blk][i] != 0;
          cabac_residual_block(c, i16 ? 1 : 2, cbf_inc_luma(wk, intra, bx, by), maxc, mc.lv[blk]);
          wk.nz[0][by * 4 + bx] = static_cast<uint8_t>(n);
        }
      }
    }
    if (mc.cbp_c) {
      for (int cc = 0; cc < 2; ++cc) {
        bool any = false;
        for (int i = 0; i < 4; ++i) any |= mc.cdc[cc][i] != 0;
        cabac_residual_block(c, 3, cbf_inc_dc(intra, 1 + cc), 4, mc.cdc[cc]);
        if (any) wk.cbf_dc |= static_cast<uint8_t>(2 << cc);
      }
      if (mc.cbp_c == 2)
        for (int cc = 0; cc < 2; ++cc)
          for (int b = 0; b < 4; ++b) {
            int n = 0;
            for (int i = 0; i < 15; ++i) n += mc.cac[cc][b][i] != 0;
            cabac_residual_block(c, 4, cbf_inc_chroma(wk, intra, cc, b & 1, b >> 1), 15, mc.cac[cc][b]);
            wk.nz[1 + cc][b] = static_cast<uint8_t>(n);
          }
    }
  }

  // ---- the macroblock layer, CAVLC --------------------------------------------------------------------------------------

  // Sets the parts of wk that every writer determines from the macroblock's decisions.
  void init_wk(const MbCode& mc, MbInfo& wk) const {
    const bool intra = mc.type == kMbI4 || mc.type == kMbI8 || mc.type == kMbI16;
    wk.kind = mc.type == kMbI4 ? kKI4 : mc.type == kMbI8 ? kKI8 : mc.type == kMbI16 ? kKI16 : kKInter;
    wk.skip = mc.type == kMbSkip;
    wk.direct16 = mc.type == kMbDirect || (mc.type == kMbSkip && slice_kind == kSliceB);
    wk.t8 = mc.t8;
    wk.cbp = static_cast<uint8_t>(mc.cbp_l | (mc.cbp_c << 4));
    wk.chroma_mode = static_cast<uint8_t>(intra ? mc.chroma : 0);
    wk.cbf_dc = 0;
    std::memset(wk.nz, 0, sizeof wk.nz);
    std::memset(wk.mvd, 0, sizeof wk.mvd);
    if (!intra) {
      for (int l = 0; l < 2; ++l)
        for (int b8 = 0; b8 < 4; ++b8) {
          wk.ref[l][b8] = mc.ref[l][b8];
          wk.mv[l][b8][0] = mc.ref[l][b8] >= 0 ? mc.mv[l][b8][0] : 0;
          wk.mv[l][b8][1] = mc.ref[l][b8] >= 0 ? mc.mv[l][b8][1] : 0;
        }
      wk.direct8 = mc.type == kMbDirect || mc.type == kMbSkip ? 0xF : mc.sub_direct;
    } else {
      std::memset(wk.ref, -1, sizeof wk.ref);
      std::memset(wk.mv, 0, sizeof wk.mv);
      wk.direct8 = 0;
    }
  }

  template <class W>
  static void put_te(W& w, int range, int v) {
    if (range > 1) w.ue(static_cast<uint32_t>(v));
    else if (range == 1) w.bit(v ? 0 : 1);
  }

  // the mvd of a partition against its prediction, stored in wk for the contexts of what follows
  void mvd_of(MbInfo& wk, unsigned done8, const MbCode& mc, int l, int p, int* dx, int* dy) const {
    int x, y, pw, ph, b8;
    part_geom(mc.shape, p, &x, &y, &pw, &ph, &b8);
    int px, py;
    mv_pred(wk, done8, l, mc.ref[l][b8], x, y, pw, mc.shape, p, &px, &py);
    *dx = mc.mv[l][b8][0] - px;
    *dy = mc.mv[l][b8][1] - py;
    for (int k = 0; k < 4; ++k) {
      const int bx = (k & 1) * 8, by = (k >> 1) * 8;
      if (bx >= x && bx < x + pw && by >= y && by < y + ph) {
        wk.mvd[l][k][0] = static_cast<uint8_t>(std::min(std::abs(*dx), 255));
        wk.mvd[l][k][1] = static_cast<uint8_t>(std::min(std::abs(*dy), 255));
      }
    }
  }
  static unsigned part_mask(int shape, int p) {
    int x, y, pw, ph, b8;
    part_geom(shape, p, &x, &y, &pw, &ph, &b8);
    unsigned m = 0;
    for (int k = 0; k < 4; ++k) {
      const int bx = (k & 1) * 8, by = (k >> 1) * 8;
      if (bx >= x && bx < x + pw && by >= y && by < y + ph) m |= 1u << k;
    }
    return m;
  }
  int active(int l) const { return num_active[l]; }

  // Writes everything of a coded macroblock except mb_skip_run / mb_skip_flag, and fills wk.
  template <class W>
  void write_mb_cavlc(W& w, const MbCode& mc, MbInfo& wk) {
    init_wk(mc, wk);
    const bool intra = mc.type == kMbI4 || mc.type == kMbI8 || mc.type == kMbI16;
    const int prefix = slice_kind == kSliceI ? 0 : (slice_kind == kSliceP ? 5 : 23);
    if (intra) {
      w.ue(static_cast<uint32_t>(prefix + intra_t_of(mc)));
      if (mc.type != kMbI16) {
        if (t8_mode) w.bit(mc.type == kMbI8);
        const bool i8 = mc.type == kMbI8;
        for (int i = 0; i < (i8 ? 4 : 16); ++i) {
          const int blk = i8 ? i * 4 : i;
          const int bx4 = i8 ? (i & 1) * 2 : blk_x(blk), by4 = i8 ? (i >> 1) * 2 : blk_y(blk);
          const int pm = pred_mode_nxn(wk, bx4, by4);
          const int mode = i8 ? mc.i8[i] : mc.i4[blk];
          if (mode == pm) {
            w.bit(1);
          } else {
            w.bit(0);
            w.put(static_cast<uint32_t>(mode < pm ? mode : mode - 1), 3);
          }
          if (i8)
            for (int k = 0; k < 4; ++k) wk.ipred[(by4 + (k >> 1)) * 4 + bx4 + (k & 1)] = static_cast<int8_t>(mode);
          else
            wk.ipred[by4 * 4 + bx4] = static_cast<int8_t>(mode);
        }
      }
      w.ue(static_cast<uint32_t>(mc.chroma));
      if (mc.type != kMbI16) {
        const int cbp = mc.cbp_l | (mc.cbp_c << 4);
        int code = 0;
        for (; code < 48; ++code)
          if (kCbpIntra[code] == cbp) break;
        w.ue(static_cast<uint32_t>(code));
      }
      if (mc.type == kMbI16 || mc.cbp_l || mc.cbp_c) w.se(0);   // mb_qp_delta
    } else {
      const bool is_b = slice_kind == kSliceB;
      w.ue(static_cast<uint32_t>(is_b ? b_mb_type(mc) : p_mb_type(mc)));
      if (mc.type != kMbDirect) {
        if (mc.shape == 3)
          for (int b8 = 0; b8 < 4; ++b8) w.ue(static_cast<uint32_t>(is_b ? b_sub_type(mc, b8) : 0));
        const int nparts = part_count(mc.shape);
        for (int l = 0; l < (is_b ? 2 : 1); ++l)
          for (int p = 0; p < nparts; ++p) {
            int x, y, pw, ph, b8;
            part_geom(mc.shape, p, &x, &y, &pw, &ph, &b8);
            if ((mc.sub_direct >> b8) & 1) continue;
            if (mc.ref[l][b8] < 0) continue;
            if (active(l) > 1) put_te(w, active(l) - 1, mc.ref[l][b8]);
          }
        for (int l = 0; l < (is_b ? 2 : 1); ++l) {
          unsigned done = 0;
          for (int p = 0; p < nparts; ++p) {
            int x, y, pw, ph, b8;
            part_geom(mc.shape, p, &x, &y, &pw, &ph, &b8);
            if (!((mc.sub_direct >> b8) & 1) && mc.ref[l][b8] >= 0) {
              int dx, dy;
              mvd_of(wk, done, mc, l, p, &dx, &dy);
              w.se(dx);
              w.se(dy);
            }
            done |= part_mask(mc.shape, p);
          }
        }
      }
      const int cbp = mc.cbp_l | (mc.cbp_c << 4);
      int code = 0;
      for (; code < 48; ++code)
        if (kCbpInter[code] == cbp) break;
      w.ue(static_cast<uint32_t>(code));
      if (mc.cbp_l && t8_mode) w.bit(mc.t8);
      if (cbp) w.se(0);
    }
    if (mc.type == kMbI16 || mc.cbp_l || mc.cbp_c) write_residual_cavlc(w, mc, wk);
  }

  // ---- the macroblock layer, CABAC ----------------------------------------------------------------------------------------

  int ref_ctx_inc(const MbInfo& wk, int list, int b8) const {
    int inc = 0;
    const int cx = b8 & 1, cy = b8 >> 1;
    for (int n = 0; n < 2; ++n) {
      const Nb8 nb = n == 0 ? nb8_at(wk, cx - 1, cy) : nb8_at(wk, cx, cy - 1);
      if (!nb.m || nb.m->intra() || nb.m->skip || ((nb.m->direct8 >> nb.b8) & 1)) continue;
      if (nb.m->ref[list][nb.b8] > 0) inc += n == 0 ? 1 : 2;
    }
    return inc;
  }
  int mvd_ctx_sum(const MbInfo& wk, int list, int comp, int x, int y) const {
    int sum = 0;
    for (int n = 0; n < 2; ++n) {
      const Nb8 nb = n == 0 ? nb8_at(wk, (x - 1) >> 3, y >> 3) : nb8_at(wk, x >> 3, (y - 1) >> 3);
      if (!nb.m || nb.m->intra() || nb.m->skip) continue;
      sum += nb.m->mvd[list][nb.b8][comp];
    }
    return sum;
  }
  int cbp_ctx_luma(const MbCode& mc, int b8) const {
    int inc = 0;
    for (int n = 0; n < 2; ++n) {
      const Nb8 nb = n == 0 ? nb8_at(MbInfo(), (b8 & 1) - 1, b8 >> 1) : nb8_at(MbInfo(), b8 & 1, (b8 >> 1) - 1);
      int cond;
      const int cx = n == 0 ? (b8 & 1) - 1 : (b8 & 1), cy = n == 0 ? (b8 >> 1) : (b8 >> 1) - 1;
      if (cx >= 0 && cy >= 0) cond = ((mc.cbp_l >> (cy * 2 + cx)) & 1) == 0;
      else if (!nb.m) cond = 0;
      else if (nb.m->skip) cond = 1;
      else cond = ((nb.m->cbp >> nb.b8) & 1) == 0;
      inc += cond << n;
    }
    return inc;
  }

  template <class C>
  void write_cbp_cabac(C& c, const MbCode& mc) {
    for (int b8 = 0; b8 < 4; ++b8) c.decision(73 + cbp_ctx_luma(mc, b8), (mc.cbp_l >> b8) & 1);
    int inc = 0;
    if (nA && !nA->skip && ((nA->cbp >> 4) & 3) != 0) inc += 1;
    if (nB && !nB->skip && ((nB->cbp >> 4) & 3) != 0) inc += 2;
    c.decision(77 + inc, mc.cbp_c != 0);
    if (mc.cbp_c != 0) {
      inc = 0;
      if (nA && !nA->skip && ((nA->cbp >> 4) & 3) == 2) inc += 1;
      if (nB && !nB->skip && ((nB->cbp >> 4) & 3) == 2) inc += 2;
      c.decision(77 + 4 + inc, mc.cbp_c == 2);
    }
  }

  template <class C>
  void write_skip_flag_cabac(C& c, bool skipped) {
    int inc = 0;
    if (nA && !nA->skip) ++inc;
    if (nB && !nB->skip) ++inc;
    c.decision((slice_kind == kSliceB ? 24 : 11) + inc, skipped);
  }

  template <class C>
  void write_mb_cabac(C& c, const MbCode& mc, MbInfo& wk) {
    init_wk(mc, wk);
    const bool intra = mc.type == kMbI4 || mc.type == kMbI8 || mc.type == kMbI16;
    if (intra) {
      const int t = intra_t_of(mc);
      if (slice_kind == kSliceI) {
        int inc = 0;
        if (nA && nA->kind != kKI4 && nA->kind != kKI8) ++inc;
        if (nB && nB->kind != kKI4 && nB->kind != kKI8) ++inc;
        cabac_mb_type_i(c, 3, true, inc, t);
      } else if (slice_kind == kSliceP) {
        cabac_mb_type_p(c, 0, t);
      } else {
        int inc = 0;
        if (nA && !nA->skip && !nA->direct16) ++inc;
        if (nB && !nB->skip && !nB->direct16) ++inc;
        cabac_mb_type_b(c, inc, 0, t);
      }
      if (mc.type != kMbI16) {
        if (t8_mode) c.decision(399 + (nA && nA->t8 ? 1 : 0) + (nB && nB->t8 ? 1 : 0), mc.type == kMbI8);
        const bool i8 = mc.type == kMbI8;
        for (int i = 0; i < (i8 ? 4 : 16); ++i) {
          const int blk = i8 ? i * 4 : i;
          const int bx4 = i8 ? (i & 1) * 2 : blk_x(blk), by4 = i8 ? (i >> 1) * 2 : blk_y(blk);
          const int pm = pred_mode_nxn(wk, bx4, by4);
          const int mode = i8 ? mc.i8[i] : mc.i4[blk];
          if (mode == pm) {
            c.decision(68, 1);
          } else {
            c.decision(68, 0);
            const int rem = mode < pm ? mode : mode - 1;
            c.decision(69, rem & 1);
            c.decision(69, (rem >> 1) & 1);
            c.decision(69, (rem >> 2) & 1);
          }
          if (i8)
            for (int k = 0; k < 4; ++k) wk.ipred[(by4 + (k >> 1)) * 4 + bx4 + (k & 1)] = static_cast<int8_t>(mode);
          else
            wk.ipred[by4 * 4 + bx4] = static_cast<int8_t>(mode);
        }
      }
      {
        int inc = 0;
        if (nA && nA->intra() && nA->chroma_mode != 0) ++inc;
        if (nB && nB->intra() && nB->chroma_mode != 0) ++inc;
        cabac_intra_chroma_mode(c, inc, mc.chroma);
      }
      if (mc.type != kMbI16) write_cbp_cabac(c, mc);
    } else {
      const bool is_b = slice_kind == kSliceB;
      if (is_b) {
        int inc = 0;
        if (nA && !nA->skip && !nA->direct16) ++inc;
        if (nB && !nB->skip && !nB->direct16) ++inc;
        cabac_mb_type_b(c, inc, b_mb_type(mc), -1);
      } else {
        cabac_mb_type_p(c, p_mb_type(mc), -1);
      }
      if (mc.type != kMbDirect) {
        if (mc.shape == 3)
          for (int b8 = 0; b8 < 4; ++b8) {
            if (is_b) cabac_sub_mb_type_b(c, b_sub_type(mc, b8));
            else cabac_sub_mb_type_p(c, 0);
          }
        const int nparts = part_count(mc.shape);
        for (int l = 0; l < (is_b ? 2 : 1); ++l)
          for (int p = 0; p < nparts; ++p) {
            int x, y, pw, ph, b8;
            part_geom(mc.shape, p, &x, &y, &pw, &ph, &b8);
            if ((mc.sub_direct >> b8) & 1) continue;
            if (mc.ref[l][b8] < 0) continue;
            if (active(l) > 1) cabac_ref_idx(c, ref_ctx_inc(wk, l, b8), mc.ref[l][b8]);
          }
        for (int l = 0; l < (is_b ? 2 : 1); ++l) {
          unsigned done = 0;
          for (int p = 0; p < nparts; ++p) {
            int x, y, pw, ph, b8;
            part_geom(mc.shape, p, &x, &y, &pw, &ph, &b8);
            if (!((mc.sub_direct >> b8) & 1) && mc.ref[l][b8] >= 0) {
              int dx, dy;
              // the contexts read the neighbours' mvd, so they are taken before this partition's own are stored
              const int sx = mvd_ctx_sum(wk, l, 0, x, y), sy = mvd_ctx_sum(wk, l, 1, x, y);
              mvd_of(wk, done, mc, l, p, &dx, &dy);
              cabac_mvd(c, 40, sx, dx);
              cabac_mvd(c, 47, sy, dy);
            }
            done |= part_mask(mc.shape, p);
          }
        }
      }
      write_cbp_cabac(c, mc);
      if (mc.cbp_l && t8_mode) c.decision(399 + (nA && nA->t8 ? 1 : 0) + (nB && nB->t8 ? 1 : 0), mc.t8);
    }
    if (mc.type == kMbI16 || mc.cbp_l || mc.cbp_c) {
      cabac_qp_delta(c, last_dqp_nonzero, 0);
      write_residual_cabac(c, mc, wk);
    }
  }
  bool last_dqp_nonzero = false;

  // ---- costs of candidates ------------------------------------------------------------------------------------------------------------

  struct Recon {
    uint8_t y[256], u[64], v[64];
  };
  void save_rec(int mx, int my, Recon& r) {
    for (int y = 0; y < 16; ++y) std::memcpy(r.y + y * 16, rec_y(mx, my) + y * ys, 16);
    for (int y = 0; y < 8; ++y) {
      std::memcpy(r.u + y * 8, rec_c(0, mx, my) + y * cs, 8);
      std::memcpy(r.v + y * 8, rec_c(1, mx, my) + y * cs, 8);
    }
  }
  void restore_rec(int mx, int my, const Recon& r) {
    for (int y = 0; y < 16; ++y) std::memcpy(rec_y(mx, my) + y * ys, r.y + y * 16, 16);
    for (int y = 0; y < 8; ++y) {
      std::memcpy(rec_c(0, mx, my) + y * cs, r.u + y * 8, 8);
      std::memcpy(rec_c(1, mx, my) + y * cs, r.v + y * 8, 8);
    }
  }

  // Bits a coded macroblock takes with the entropy coder's present state.
  double mb_bits(const MbCode& mc) {
    MbInfo scratch;
    scratch.slice = static_cast<int16_t>(cur_slice);
    if (cabac) {
      H264CostCoder cc(*bc);
      if (slice_kind != kSliceI) write_skip_flag_cabac(cc, false);
      write_mb_cabac(cc, mc, scratch);
      return cc.cost / 256.0;
    }
    BitCounter counter;
    write_mb_cavlc(counter, mc, scratch);
    return static_cast<double>(counter.n);
  }
  double mb_sse(int mx, int my) {
    double sse = static_cast<double>(sse_block(src_y(mx, my), ys, rec_y(mx, my), ys, 16, 16));
    for (int c = 0; c < 2; ++c) sse += static_cast<double>(sse_block(src_c(c, mx, my), cs, rec_c(c, mx, my), cs, 8, 8));
    return sse;
  }
  // Total cost of a coded macroblock: squared error plus lambda times its bits.
  double mb_cost(int mx, int my, const MbCode& mc) { return mb_sse(mx, my) + lambda * mb_bits(mc); }

  // ---- intra coding of a macroblock -------------------------------------------------------------------------------------------------

  // Intra coding of the macroblock: the best of Intra16x16, Intra4x4 and Intra8x8 (with the chroma mode chosen by SATD). The winner's reconstruction is left in
  // the picture; best_wk holds its Intra4x4/8x8 modes.
  double code_intra(int mx, int my, MbCode& best, MbInfo& best_wk) {
    // chroma mode
    int cmode = 0;
    {
      double bcost = 1e30;
      uint8_t pu[64], pv[64];
      for (int mode = 0; mode < 4; ++mode) {
        if (!chroma_mode_ok(mode)) continue;
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
        if (cost < bcost) {
          bcost = cost;
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
        if (!i16_mode_ok(mode)) continue;
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
    Recon best_rec, tmp_rec;
    double best_cost = 1e30;
    {
      MbCode a;
      a.type = kMbI16;
      a.i16 = i16mode;
      a.chroma = cmode;
      MbInfo wk;
      code_i16(mx, my, p16, a);
      code_chroma(mx, my, pu, pv, true, a);
      best_cost = mb_cost(mx, my, a);
      save_rec(mx, my, best_rec);
      best = a;
      best_wk = wk;
    }
    {
      MbCode b;
      b.type = kMbI4;
      b.chroma = cmode;
      MbInfo wk;
      code_i4(mx, my, b, wk);
      code_chroma(mx, my, pu, pv, true, b);
      const double jb = mb_cost(mx, my, b);
      if (jb < best_cost) {
        best_cost = jb;
        save_rec(mx, my, best_rec);
        best = b;
        best_wk = wk;
      }
    }
    if (t8_mode) {
      MbCode c;
      c.type = kMbI8;
      c.t8 = true;
      c.chroma = cmode;
      MbInfo wk;
      code_i8(mx, my, c, wk);
      code_chroma(mx, my, pu, pv, true, c);
      const double jc = mb_cost(mx, my, c);
      if (jc < best_cost) {
        best_cost = jc;
        save_rec(mx, my, best_rec);
        best = c;
        best_wk = wk;
      }
    }
    (void)tmp_rec;
    restore_rec(mx, my, best_rec);
    return best_cost;
  }

  // ---- inter prediction of a macroblock --------------------------------------------------------------------------------------------

  static int mv_bits(int dx, int dy) {
    auto se_bits = [](int v) {
      const uint32_t k = v > 0 ? static_cast<uint32_t>(2 * v - 1) : static_cast<uint32_t>(-2 * v);
      return 2 * floor_log2(k + 1) + 1;
    };
    return se_bits(dx) + se_bits(dy);
  }

  // The prediction of the macroblock from the motion in mc: per 8x8 block, from list 0, list 1, or their average.
  void build_pred(int mx, int my, const MbCode& mc, uint8_t* pl, uint8_t* pu, uint8_t* pv) const {
    const int px = mx * 16, py = my * 16;
    bool uniform = true;
    for (int b8 = 1; b8 < 4; ++b8)
      for (int l = 0; l < 2; ++l)
        if (mc.ref[l][b8] != mc.ref[l][0] || (mc.ref[l][0] >= 0 && (mc.mv[l][b8][0] != mc.mv[l][0][0] || mc.mv[l][b8][1] != mc.mv[l][0][1]))) uniform = false;
    const int nb = uniform ? 1 : 4;
    const int bs = uniform ? 16 : 8;
    for (int b = 0; b < nb; ++b) {
      const int bx = uniform ? 0 : (b & 1) * 8, by = uniform ? 0 : (b >> 1) * 8;
      uint8_t tl[2][256], tu[2][64], tv[2][64];
      int used = 0;
      for (int l = 0; l < 2; ++l) {
        const int ri = mc.ref[l][b];
        if (ri < 0) continue;
        const RefPic& r = lists[l][static_cast<size_t>(ri)];
        pred_luma(r, px + bx, py + by, bs, bs, mc.mv[l][b][0], mc.mv[l][b][1], tl[used], bs);
        pred_chroma(r, px + bx, py + by, bs, bs, mc.mv[l][b][0], mc.mv[l][b][1], tu[used], tv[used], bs / 2);
        ++used;
      }
      for (int y = 0; y < bs; ++y)
        for (int x = 0; x < bs; ++x) {
          const int v = used == 2 ? (tl[0][y * bs + x] + tl[1][y * bs + x] + 1) >> 1 : tl[0][y * bs + x];
          pl[(by + y) * 16 + bx + x] = static_cast<uint8_t>(v);
        }
      for (int y = 0; y < bs / 2; ++y)
        for (int x = 0; x < bs / 2; ++x) {
          const int vu = used == 2 ? (tu[0][y * (bs / 2) + x] + tu[1][y * (bs / 2) + x] + 1) >> 1 : tu[0][y * (bs / 2) + x];
          const int vv = used == 2 ? (tv[0][y * (bs / 2) + x] + tv[1][y * (bs / 2) + x] + 1) >> 1 : tv[0][y * (bs / 2) + x];
          pu[(by / 2 + y) * 8 + bx / 2 + x] = static_cast<uint8_t>(vu);
          pv[(by / 2 + y) * 8 + bx / 2 + x] = static_cast<uint8_t>(vv);
        }
    }
  }

  // ---- motion search ------------------------------------------------------------------------------------------------------------------

  struct MeResult {
    int vx = 0, vy = 0, cost = 0x7fffffff, sad = 0;
  };

  bool mv_in_range(int px, int py, int bw_, int bh_, int vx, int vy) const {
    const int X = px + (vx >> 2), Y = py + (vy >> 2);
    return X >= -kPad + 4 && Y >= -kPad + 4 && X + bw_ + 4 <= ys + kPad && Y + bh_ + 4 <= mh * 16 + kPad && std::abs(vx) <= (kSearch + 2) * 4 && std::abs(vy) <= (kSearch + 2) * 4;
  }

  // SAD of the block at (px, py), bw x bh, predicted from reference r with vector (vx, vy)
  int sad_at(const RefPic& r, int px, int py, int bw_, int bh_, int vx, int vy) const {
    const uint8_t* s = &sy[static_cast<size_t>(py) * ys + px];
    int sad = 0;
    if (((vx | vy) & 3) == 0) {
      const RefPlanes& rp = r.rp;
      const int X0 = px + (vx >> 2), Y0 = py + (vy >> 2);
      for (int y = 0; y < bh_; ++y) {
        const uint8_t* row = rp.at(rp.g, X0, Y0 + y);   // the padded plane lets a row of in-range samples be read directly
        for (int x = 0; x < bw_; ++x) sad += std::abs(s[y * ys + x] - row[x]);
      }
      return sad;
    }
    uint8_t pred[256];
    pred_luma(r, px, py, bw_, bh_, vx, vy, pred, bw_);
    for (int y = 0; y < bh_; ++y)
      for (int x = 0; x < bw_; ++x) sad += std::abs(s[y * ys + x] - pred[y * bw_ + x]);
    return sad;
  }

  // Motion search of one partition (px, py, bw x bh) in reference r: the whole-sample diamond from the given starting points, then half and quarter
  // sample refinement. `mvp` is the predicted vector the cost is measured against.
  MeResult me_search(const RefPic& r, int px, int py, int bw_, int bh_, int mvpx, int mvpy, const int (*seeds)[2], int nseeds) const {
    MeResult best;
    auto try_mv = [&](int vx, int vy) {
      if (!mv_in_range(px, py, bw_, bh_, vx, vy)) return false;
      const int sad = sad_at(r, px, py, bw_, bh_, vx, vy);
      const int cost = sad + static_cast<int>(lam_sad * mv_bits(vx - mvpx, vy - mvpy) + 0.5);
      if (cost < best.cost) {
        best.cost = cost;
        best.sad = sad;
        best.vx = vx;
        best.vy = vy;
        return true;
      }
      return false;
    };
    try_mv(0, 0);
    try_mv((mvpx >> 2) * 4, (mvpy >> 2) * 4);
    for (int i = 0; i < nseeds; ++i) try_mv((seeds[i][0] >> 2) * 4, (seeds[i][1] >> 2) * 4);
    for (int it = 0; it < 2 * kSearch; ++it) {
      bool moved = false;
      const int cx = best.vx, cy = best.vy;
      for (const auto& d : {std::pair<int, int>{4, 0}, {-4, 0}, {0, 4}, {0, -4}})
        if (try_mv(cx + d.first, cy + d.second)) moved = true;
      if (!moved) break;
    }
    for (const int step : {2, 1}) {
      for (int round = 0; round < 2; ++round) {
        const int cx = best.vx, cy = best.vy;
        bool moved = false;
        for (int dy = -step; dy <= step; dy += step)
          for (int dx = -step; dx <= step; dx += step)
            if ((dx || dy) && try_mv(cx + dx, cy + dy)) moved = true;
        if (!moved) break;
      }
    }
    return best;
  }

  // ---- residual coding and cost of an inter candidate --------------------------------------------------------------------------------

  // Codes the residual of the macroblock against the prediction of mc's motion (trying the 8x8 transform too where it is allowed), leaves the
  // reconstruction in the picture, and returns the cost J. mc.cbp_*, the levels and mc.t8 are set.
  double eval_inter(int mx, int my, MbCode& mc, Recon* rec_out) {
    uint8_t pl[256], pu[64], pv[64];
    build_pred(mx, my, mc, pl, pu, pv);
    mc.t8 = false;
    code_luma_inter(mx, my, pl, mc);
    code_chroma(mx, my, pu, pv, false, mc);
    double best = mb_cost(mx, my, mc);
    if (rec_out) save_rec(mx, my, *rec_out);
    if (t8_mode && mc.cbp_l != 0) {
      MbCode m8 = mc;
      m8.t8 = true;
      std::memset(m8.lv, 0, sizeof m8.lv);
      code_luma_inter(mx, my, pl, m8);
      code_chroma(mx, my, pu, pv, false, m8);
      if (m8.cbp_l != 0) {
        const double j8 = mb_cost(mx, my, m8);
        if (j8 < best) {
          best = j8;
          mc = m8;
          if (rec_out) save_rec(mx, my, *rec_out);
        } else {
          // put the 4x4 reconstruction back
          if (rec_out) restore_rec(mx, my, *rec_out);
        }
      } else if (rec_out) {
        restore_rec(mx, my, *rec_out);
      }
    }
    return best;
  }

  // ---- P macroblock analysis --------------------------------------------------------------------------------------------------------

  double skip_bits() {
    if (cabac) {
      H264CostCoder cc(*bc);
      write_skip_flag_cabac(cc, true);
      return cc.cost / 256.0;
    }
    return 1.0;
  }
  int ref_bits(int l, int ri) const {
    if (num_active[l] <= 1) return 0;
    return num_active[l] == 2 ? 1 : 2 * floor_log2(static_cast<uint32_t>(ri) + 1) + 1;
  }

  // The motion of the partitions of `shape` for a P macroblock, searched over the reference pictures of list 0. Returns the sum of the partitions'
  // search costs (SAD plus lambda times the bits of vector and reference index).
  int search_shape_p(int mx, int my, int shape, const int (*seed)[2], int nseed, MbCode* out) {
    MbInfo wk;
    wk.slice = static_cast<int16_t>(cur_slice);
    unsigned done = 0;
    int total = 0;
    out->type = kMbInter;
    out->shape = shape;
    for (int p = 0; p < part_count(shape); ++p) {
      int x, y, pw, ph, b8;
      part_geom(shape, p, &x, &y, &pw, &ph, &b8);
      MeResult best;
      int best_ri = 0;
      for (int ri = 0; ri < num_active[0]; ++ri) {
        int mvpx, mvpy;
        mv_pred(wk, done, 0, ri, x, y, pw, shape, p, &mvpx, &mvpy);
        int seeds[8][2];
        int ns = 0;
        for (int i = 0; i < nseed && ns < 4; ++i) {
          seeds[ns][0] = seed[i][0];
          seeds[ns][1] = seed[i][1];
          ++ns;
        }
        const NbMot na = nb_motion(wk, done, 0, (x - 1) >> 3, y >> 3), nb = nb_motion(wk, done, 0, x >> 3, (y - 1) >> 3);
        if (na.ref == ri) { seeds[ns][0] = na.x; seeds[ns][1] = na.y; ++ns; }
        if (nb.ref == ri) { seeds[ns][0] = nb.x; seeds[ns][1] = nb.y; ++ns; }
        MeResult r = me_search(lists[0][static_cast<size_t>(ri)], mx * 16 + x, my * 16 + y, pw, ph, mvpx, mvpy, seeds, ns);
        r.cost += static_cast<int>(lam_sad * ref_bits(0, ri) + 0.5);
        if (r.cost < best.cost) {
          best = r;
          best_ri = ri;
        }
      }
      total += best.cost;
      for (int k = 0; k < 4; ++k) {
        const int bx = (k & 1) * 8, by = (k >> 1) * 8;
        if (bx >= x && bx < x + pw && by >= y && by < y + ph) {
          out->ref[0][k] = static_cast<int8_t>(best_ri);
          out->mv[0][k][0] = static_cast<int16_t>(best.vx);
          out->mv[0][k][1] = static_cast<int16_t>(best.vy);
          wk.ref[0][k] = static_cast<int8_t>(best_ri);
          wk.mv[0][k][0] = static_cast<int16_t>(best.vx);
          wk.mv[0][k][1] = static_cast<int16_t>(best.vy);
        }
      }
      done |= part_mask(shape, p);
    }
    return total;
  }

  // The best coding of a macroblock of a P slice; its reconstruction is left in the picture.
  double analyze_p(int mx, int my, MbCode& best) {
    MbCode skip_mc;
    skip_mc.type = kMbSkip;
    int skx, sky;
    pskip_mv(&skx, &sky);
    for (int b8 = 0; b8 < 4; ++b8) {
      skip_mc.ref[0][b8] = 0;
      skip_mc.mv[0][b8][0] = static_cast<int16_t>(skx);
      skip_mc.mv[0][b8][1] = static_cast<int16_t>(sky);
    }
    Recon best_rec, rec;
    double best_cost = 1e30;
    // P_Skip: the 16x16 prediction at the skip vector, if no residual is left
    {
      MbCode sk = skip_mc;
      sk.type = kMbInter;   // coded as a plain inter macroblock first, to see whether any residual survives
      sk.shape = 0;
      const double j = eval_inter(mx, my, sk, &rec);
      if (sk.cbp_l == 0 && sk.cbp_c == 0) {
        const double js = mb_sse(mx, my) + lambda * skip_bits();
        best_cost = js;
        best = skip_mc;
        best_rec = rec;
      } else if (j < best_cost) {
        best_cost = j;
        best = sk;
        best_rec = rec;
      }
    }
    // 16x16 over the reference pictures
    MbCode m16;
    int seed16[1][2] = {{0, 0}};
    const int c16 = search_shape_p(mx, my, 0, seed16, 0, &m16);
    {
      const bool same_as_skip = m16.ref[0][0] == 0 && m16.mv[0][0][0] == skx && m16.mv[0][0][1] == sky;
      if (!(same_as_skip && best.type == kMbSkip)) {
        MbCode c = m16;
        const double j = eval_inter(mx, my, c, &rec);
        if (j < best_cost) {
          best_cost = j;
          best = c;
          best_rec = rec;
        }
      }
    }
    // the other partitionings, tried when the 16x16 prediction is not good already
    if (best.type != kMbSkip && c16 > 256) {
      const int seedv[1][2] = {{m16.mv[0][0][0], m16.mv[0][0][1]}};
      MbCode cands[3];
      int costs[3];
      int nb = 0;
      int bestc = 0x7fffffff, besti = -1;
      for (int shape = 1; shape <= 3; ++shape) {
        costs[nb] = search_shape_p(mx, my, shape, seedv, 1, &cands[nb]) + static_cast<int>(lam_sad * (shape == 3 ? 6 : 3));
        if (costs[nb] < bestc) {
          bestc = costs[nb];
          besti = nb;
        }
        ++nb;
      }
      if (besti >= 0 && bestc < c16) {
        MbCode c = cands[besti];
        const double j = eval_inter(mx, my, c, &rec);
        if (j < best_cost) {
          best_cost = j;
          best = c;
          best_rec = rec;
        }
      }
    }
    // intra, when no inter prediction fits
    if (best.type != kMbSkip && c16 > 3 * 256) {
      MbCode ib;
      MbInfo iw;
      save_rec(mx, my, rec);   // the last inter candidate's reconstruction is in the picture; the intra code overwrites it
      const double ji = code_intra(mx, my, ib, iw);
      if (ji + 1e-9 < best_cost) {
        best_cost = ji;
        best = ib;
        save_rec(mx, my, best_rec);
      }
    }
    restore_rec(mx, my, best_rec);
    return best_cost;
  }

  // ---- the picture ----------------------------------------------------------------------------------------------------------------

  std::vector<uint8_t> encode_picture(const EncPicture& in, PicType t, int q, int poc, EncStats* stats);
  std::vector<uint8_t> code_picture(const EncPicture& in, PicType t, int q, int poc);
  std::vector<uint8_t> encode_field_pair(const EncPicture& frame, bool top_first, PicType t, int q, EncStats* stats);
  bool deblock_with_decoder(const std::vector<std::vector<uint8_t>>& nals, bool idr, bool ref_pic, std::shared_ptr<vgpu_h264::Frame>* out);
  bool deblock_field(const std::vector<uint8_t>& nal, bool idr);

  // The first macroblock of every slice of the picture.
  void layout_slices() {
    slice_start.clear();
    const int nmb = mw * mh;
    int per = nmb;
    switch (opt.slice_mode) {
      case 0: per = opt.slice_data > 0 ? opt.slice_data : nmb; break;
      case 1: per = bytes_slice_mbs > 0 ? bytes_slice_mbs : nmb; break;
      case 2: per = opt.slice_data > 0 ? opt.slice_data * mw : nmb; break;
      default: per = opt.slice_data > 1 ? (nmb + opt.slice_data - 1) / opt.slice_data : nmb; break;
    }
    per = std::max(per, 1);
    if (opt.slice_mode == 3 && opt.slice_data > 1) {
      // slice_data slices of nearly equal size
      const int n = std::min(opt.slice_data, nmb);
      for (int i = 0; i < n; ++i) slice_start.push_back(static_cast<int>(static_cast<int64_t>(nmb) * i / n));
      return;
    }
    for (int a = 0; a < nmb; a += per) slice_start.push_back(a);
  }
  int bytes_slice_mbs = 0;   // slice mode 1: macroblocks per slice, estimated from the bytes the pictures so far took per macroblock
  double bytes_per_mb = 0;

  // Reference lists of the picture being coded
  void build_lists(int poc) {
    lists[0].clear();
    lists[1].clear();
    list_src[0].clear();
    list_src[1].clear();
    num_active[0] = num_active[1] = 0;
    if (slice_kind == kSliceI) return;
    if (!field_coding) {
      std::vector<std::shared_ptr<DpbPic>> l0, l1;
      if (slice_kind == kSliceB) {
        std::vector<std::shared_ptr<DpbPic>> before, after;
        for (auto& d : dpb) (d->poc < poc ? before : after).push_back(d);
        std::sort(before.begin(), before.end(), [](const auto& a, const auto& b) { return a->poc > b->poc; });
        std::sort(after.begin(), after.end(), [](const auto& a, const auto& b) { return a->poc < b->poc; });
        l0 = before;
        l0.insert(l0.end(), after.begin(), after.end());
        l1 = after;
        l1.insert(l1.end(), before.begin(), before.end());
        num_active[0] = 1;
        num_active[1] = 1;
      } else {
        l0 = dpb;   // most recently coded first
        num_active[0] = std::min<int>(opt.num_ref, static_cast<int>(l0.size()));
      }
      for (int l = 0; l < 2; ++l) {
        const auto& src = l == 0 ? l0 : l1;
        for (int i = 0; i < num_active[l]; ++i) {
          RefPic r;
          build_ref(r, *src[static_cast<size_t>(i)]->frame, 0, 1, 0);
          r.poc = src[static_cast<size_t>(i)]->poc;
          r.motion = &src[static_cast<size_t>(i)]->motion;
          lists[l].push_back(std::move(r));
          list_src[l].push_back(src[static_cast<size_t>(i)]);
        }
      }
    } else {
      // 8.2.4.2.5: the fields of the reference frames, the same parity first, the frame being coded counted among them
      const int cp = fld.bottom ? 1 : 0;
      auto add = [&](const vgpu_h264::Frame& f, int row0, int offset) {
        RefPic r;
        build_ref(r, f, row0, 2, offset);
        lists[0].push_back(std::move(r));
      };
      if (!fld.second) {
        if (fr_prev) {
          add(*fr_prev, cp, 0);
          add(*fr_prev, 1 - cp, 2 * (cp - (1 - cp)));
        }
      } else {
        if (fr_prev && !cur_idr) add(*fr_prev, cp, 0);
        add(*fr_cur, 1 - cp, 2 * (cp - (1 - cp)));
      }
      num_active[0] = static_cast<int>(lists[0].size());
    }
  }

  // B macroblocks (stage B): not yet
  double analyze_b(int mx, int my, MbCode& best) {
    MbInfo iw;
    return code_intra(mx, my, best, iw);
  }
  bool want_stats = false;
  std::shared_ptr<vgpu_h264::Frame> last_decoded;
};

// 0.85 * 2^((qp - 12) / 3), without libm so that every machine makes the same decisions
static double h264_lambda(int qp) {
  static const double frac[3] = {1.0, 1.2599210498948732, 1.5874010519681994};
  const int x = qp - 12;
  const int q = x >= 0 ? x / 3 : -((-x + 2) / 3);
  const int r = x - 3 * q;
  return 0.85 * frac[r] * std::ldexp(1.0, q);
}

bool H264Encoder::Impl::deblock_with_decoder(const std::vector<std::vector<uint8_t>>& nals, bool idr, bool ref_pic, std::shared_ptr<vgpu_h264::Frame>* out) {
  vgpu_h264::PicParams pp;
  pp.log2_max_frame_num = 4;
  pp.poc_type = b_mode ? 0 : 2;
  pp.log2_max_poc_lsb = 8;
  pp.frame_mbs_only = 1;
  pp.direct_8x8_inference = 1;
  pp.num_ref_frames = stream.num_ref_frames;
  pp.chroma_format_idc = 1;
  pp.bit_depth_luma = pp.bit_depth_chroma = 8;
  pp.cabac = cabac ? 1 : 0;
  pp.transform_8x8_mode = t8_mode ? 1 : 0;
  pp.num_ref_idx_default[0] = pp.num_ref_idx_default[1] = 1;
  pp.pic_init_qp = 26;
  pp.deblocking_control_present = 1;
  pp.mbs_w = mw;
  pp.mbs_h = mh;
  pp.frame_num = frame_num;
  pp.poc[0] = pp.poc[1] = cur_poc;
  pp.ref_pic = ref_pic ? 1 : 0;
  pp.idr = idr;
  auto cur = std::make_shared<vgpu_h264::Frame>();
  cur->uid = ++uid;
  cur->alloc(mw, mh);
  if (!idr) {
    for (const auto& d : dpb) {
      vgpu_h264::DpbEntry e;
      e.frame = d->frame.get();
      e.frame_num = d->frame_num;
      e.used = 3;
      e.poc[0] = e.poc[1] = d->poc;
      pp.dpb.push_back(e);
    }
  }
  std::vector<vgpu_h264::SliceData> slices;
  for (const auto& n : nals) slices.push_back({n.data(), n.size()});
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
  cur->poc[0] = cur->poc[1] = cur_poc;
  *out = std::move(cur);
  return ok;
}

std::vector<uint8_t> H264Encoder::Impl::encode_picture(const EncPicture& in, PicType t, int q, int poc, EncStats* stats) {
  if (t != PicType::kIdr && !have_ref) t = PicType::kIdr;
  if (t == PicType::kBi && dpb.size() < 2) t = PicType::kInter;
  want_stats = stats != nullptr;
  std::vector<uint8_t> out = code_picture(in, t, q, poc);
  st.bytes = out.size();
  if (stats) {
    const vgpu_h264::Frame& f = *last_decoded;
    double sse = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const int d = in.y[static_cast<size_t>(y) * w + x] - f.y[static_cast<size_t>(y) * f.stride_y + x];
        sse += static_cast<double>(d) * d;
      }
    st.psnr_y = sse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(w) * h / sse);
    *stats = st;
  }
  return out;
}

// Codes one picture: a frame, or (in field coding) the field described by `fld`.
std::vector<uint8_t> H264Encoder::Impl::code_picture(const EncPicture& in, PicType t, int q, int poc) {
  type = t;
  qp = clip3(0, 51, q);
  qpc = qpc_of(qp);
  lambda = h264_lambda(qp);
  lam_sad = std::sqrt(lambda);
  const bool idr = t == PicType::kIdr && !(field_coding && fld.second);
  slice_kind = (t == PicType::kIdr || t == PicType::kIntra) ? kSliceI : (t == PicType::kBi ? kSliceB : kSliceP);
  const bool reference = t != PicType::kBi;
  int poc_lsb = 0;
  if (!field_coding) {
    if (idr) {
      frame_num = 0;
      poc = 0;
    } else {
      frame_num = (ref_frame_num + 1) & 15;
    }
    cur_poc = 2 * poc;
    poc_lsb = cur_poc & 255;
  } else {
    if (!fld.second) {
      if (idr) {
        field_count = 0;
        frame_num = 0;
      } else {
        frame_num = (ref_frame_num + 1) & 15;
      }
    }
    poc_lsb = (4 * field_count + (fld.second ? 1 : 0)) & 0xFFFF;
  }
  load_source(in);
  const int nmb = mw * mh;
  mbs.assign(static_cast<size_t>(nmb), MbInfo());
  build_lists(cur_poc);
  layout_slices();

  std::vector<uint8_t> out;
  std::vector<std::vector<uint8_t>> nals;   // every slice's NAL unit after its start code, for the decoder
  st = EncStats();
  st.qp = qp;
  const int ref_idc = idr || t == PicType::kIntra ? 3 : (reference ? 2 : 0);
  const int cabac_set = slice_kind == kSliceI ? 0 : 1;   // cabac_init_idc 0 for P and B slices
  const bool intra_pic = idr || t == PicType::kIntra;

  if (intra_pic) {
    // Intra pictures carry a digest of the picture they were made from (a user_data_unregistered SEI message that every decoder ignores). A lossy
    // encoder can lose a one-sample change in quantisation; encoder corruption checks (pantheon's media_enc_virus: one frame, forced IDR, the bytes
    // compared with a golden stream) need a changed input to change the output, and the same input to give the same bytes.
    std::vector<uint8_t> sei = {5, 24};   // payloadType 5, payloadSize 24
    const std::vector<uint8_t> payload = picture_digest_payload(in, w, h, qp);
    sei.insert(sei.end(), payload.begin(), payload.end());
    sei.push_back(0x80);   // rbsp_trailing_bits
    append_nal(out, 0, 6, sei);
  }

  uint8_t sav_dummy = 0;
  (void)sav_dummy;
  size_t slice_bytes_total = 0;
  for (size_t si = 0; si < slice_start.size(); ++si) {
    const int first = slice_start[si], last = si + 1 < slice_start.size() ? slice_start[si + 1] : nmb;
    cur_slice = static_cast<int>(si);
    BitWriter wr;
    // slice_header
    wr.ue(static_cast<uint32_t>(field_coding ? 0 : first));   // first_mb_in_slice
    wr.ue(slice_kind == kSliceP ? 5 : (slice_kind == kSliceB ? 6 : 7));   // slice_type: every slice of the picture is of this type
    wr.ue(0);                                  // pic_parameter_set_id
    wr.put(static_cast<uint32_t>(frame_num), 4);
    if (field_coding) {
      wr.bit(1);                               // field_pic_flag
      wr.bit(fld.bottom ? 1 : 0);              // bottom_field_flag
    }
    if (idr) wr.ue(0);                         // idr_pic_id
    if (field_coding) wr.put(static_cast<uint32_t>(poc_lsb), 16);   // pic_order_cnt_lsb
    else if (b_mode) wr.put(static_cast<uint32_t>(poc_lsb), 8);
    if (slice_kind == kSliceB) wr.bit(1);      // direct_spatial_mv_pred_flag
    if (slice_kind != kSliceI) {
      const bool override_ = num_active[0] != 1 || (slice_kind == kSliceB && num_active[1] != 1);
      wr.bit(override_ ? 1 : 0);               // num_ref_idx_active_override_flag
      if (override_) {
        wr.ue(static_cast<uint32_t>(num_active[0] - 1));
        if (slice_kind == kSliceB) wr.ue(static_cast<uint32_t>(num_active[1] - 1));
      }
      wr.bit(0);                               // ref_pic_list_modification_flag_l0
      if (slice_kind == kSliceB) wr.bit(0);    // ref_pic_list_modification_flag_l1
    }
    if (ref_idc != 0) {
      if (idr) {
        wr.bit(0);                             // no_output_of_prior_pics_flag
        wr.bit(0);                             // long_term_reference_flag
      } else {
        wr.bit(0);                             // adaptive_ref_pic_marking_mode_flag
      }
    }
    if (cabac && slice_kind != kSliceI) wr.ue(0);   // cabac_init_idc
    wr.se(qp - 26);                            // slice_qp_delta
    if (deblock) {
      wr.ue(0);                                // disable_deblocking_filter_idc
      wr.se(0);                                // slice_alpha_c0_offset_div2
      wr.se(0);                                // slice_beta_offset_div2
    } else {
      wr.ue(1);
    }

    std::vector<uint8_t> rbsp;
    std::unique_ptr<H264BitCoder> coder;
    if (cabac) {
      while (!wr.aligned()) wr.bit(1);         // cabac_alignment_one_bit
      rbsp = wr.bytes();
      coder = std::make_unique<H264BitCoder>(rbsp);
      for (int i = 0; i < kCabacCtx; ++i) coder->ctx[i] = cabac_init_ctx(kCabacInit[i][cabac_set][0], kCabacInit[i][cabac_set][1], qp);
      bc = coder.get();
      bw = nullptr;
    } else {
      bw = &wr;
      bc = nullptr;
    }
    skip_run = 0;
    last_dqp_nonzero = false;
    for (int addr = first; addr < last; ++addr) {
      const int mx = addr % mw, my = addr / mw;
      setup_neighbours(mx, my);
      MbCode mc;
      if (slice_kind == kSliceI) {
        MbInfo iw;
        code_intra(mx, my, mc, iw);
      } else if (slice_kind == kSliceP) {
        analyze_p(mx, my, mc);
      } else {
        analyze_b(mx, my, mc);
      }
      const bool skipped = mc.type == kMbSkip;
      MbInfo wk;
      wk.slice = static_cast<int16_t>(cur_slice);
      wk.qp = static_cast<int8_t>(qp);
      if (cabac) {
        if (slice_kind != kSliceI) write_skip_flag_cabac(*bc, skipped);
        if (skipped) init_wk(mc, wk);
        else write_mb_cabac(*bc, mc, wk);
        bc->terminate(addr == last - 1 ? 1 : 0);   // end_of_slice_flag
      } else if (skipped) {
        ++skip_run;
        init_wk(mc, wk);
      } else {
        if (slice_kind != kSliceI) {
          wr.ue(static_cast<uint32_t>(skip_run));
          skip_run = 0;
        }
        write_mb_cavlc(wr, mc, wk);
      }
      mbs[static_cast<size_t>(addr)] = wk;
      if (skipped) ++st.skipped_mbs;
      else if (wk.intra()) ++st.intra_mbs;
      else ++st.inter_mbs;
    }
    if (cabac) {
      coder->finish();
    } else {
      if (skip_run) wr.ue(static_cast<uint32_t>(skip_run));
      wr.trailing();
      rbsp = wr.bytes();
    }
    bc = nullptr;
    bw = nullptr;
    const size_t before = out.size();
    append_nal(out, ref_idc, idr ? 5 : 1, rbsp);
    // the NAL unit as the decoder takes it: after the four-byte start code
    nals.emplace_back(out.begin() + static_cast<std::ptrdiff_t>(before) + 4, out.end());
    slice_bytes_total += rbsp.size();
  }
  if (opt.slice_mode == 1) {
    bytes_per_mb = bytes_per_mb == 0 ? static_cast<double>(slice_bytes_total) / nmb : 0.5 * bytes_per_mb + 0.5 * static_cast<double>(slice_bytes_total) / nmb;
    bytes_slice_mbs = std::max(1, static_cast<int>(opt.slice_data / std::max(bytes_per_mb * 1.1, 0.5)));
  }

  std::shared_ptr<vgpu_h264::Frame> decoded;
  if (field_coding) {
    deblock_field(nals[0], idr);
  } else if (reference || want_stats) {
    deblock_with_decoder(nals, idr, reference, &decoded);
    last_decoded = decoded;
    if (reference) {
      if (idr) dpb.clear();
      auto d = std::make_shared<DpbPic>();
      d->frame = decoded;
      d->frame_num = frame_num;
      d->poc = cur_poc;
      d->motion = mbs;
      dpb.insert(dpb.begin(), d);
      while (static_cast<int>(dpb.size()) > stream.num_ref_frames) dpb.pop_back();
      ref_frame_num = frame_num;
    }
  }
  have_ref = true;
  return out;
}

// The field pictures' loop filter and reference: like deblock_with_decoder, with the frame store shared by the two fields.
bool H264Encoder::Impl::deblock_field(const std::vector<uint8_t>& nal, bool idr) {
  vgpu_h264::PicParams pp;
  pp.log2_max_frame_num = 4;
  pp.poc_type = 0;
  pp.log2_max_poc_lsb = 16;
  pp.frame_mbs_only = 0;
  pp.direct_8x8_inference = 1;
  pp.num_ref_frames = 2;
  pp.chroma_format_idc = 1;
  pp.bit_depth_luma = pp.bit_depth_chroma = 8;
  pp.cabac = 0;
  pp.num_ref_idx_default[0] = pp.num_ref_idx_default[1] = 1;
  pp.pic_init_qp = 26;
  pp.deblocking_control_present = 1;
  pp.mbs_w = mw;
  pp.mbs_h = 2 * mh;
  pp.field_pic = 1;
  pp.bottom_field = fld.bottom;
  pp.second_field = fld.second;
  pp.frame_num = frame_num;
  const int first_poc = 4 * field_count, this_poc = first_poc + (fld.second ? 1 : 0);
  pp.poc[fld.bottom ? 1 : 0] = this_poc;
  pp.poc[fld.bottom ? 0 : 1] = fld.second ? first_poc : this_poc;
  pp.ref_pic = 1;
  pp.idr = idr;
  if (!fld.second) {
    fr_cur = std::make_unique<vgpu_h264::Frame>();
    fr_cur->uid = ++uid;
    fr_cur->alloc(mw, 2 * mh);
    cur_first_parity = fld.bottom ? 1 : 0;
    cur_idr = idr;
  }
  if (!cur_idr && fr_prev) {
    vgpu_h264::DpbEntry e;
    e.frame = fr_prev.get();
    e.frame_num = ref_frame_num;
    e.used = 3;
    e.poc[0] = fr_prev->poc[0];
    e.poc[1] = fr_prev->poc[1];
    pp.dpb.push_back(e);
  }
  if (fld.second) {
    vgpu_h264::DpbEntry e;
    e.frame = fr_cur.get();
    e.frame_num = frame_num;
    e.used = 1 << cur_first_parity;
    e.poc[cur_first_parity] = first_poc;
    e.poc[1 - cur_first_parity] = first_poc;
    pp.dpb.push_back(e);
  }
  std::vector<vgpu_h264::SliceData> slices{{nal.data(), nal.size()}};
  std::string err;
  const bool ok = vgpu_h264::decode_picture(fr_cur.get(), pp, slices, &err);
  if (!ok) {
    if (!decode_failed_reported) {
      std::fprintf(stderr, "[vgpu] NVENC: the H.264 encoder's own field did not decode (%s); the reference is its reconstruction\n", err.c_str());
      decode_failed_reported = true;
    }
    const int par = fld.bottom ? 1 : 0;
    for (int y = 0; y < mh * 16; ++y) std::memcpy(&fr_cur->y[static_cast<size_t>(2 * y + par) * fr_cur->stride_y], &ry[static_cast<size_t>(y) * ys], static_cast<size_t>(ys));
    for (int y = 0; y < mh * 8; ++y) {
      std::memcpy(&fr_cur->u[static_cast<size_t>(2 * y + par) * fr_cur->stride_c], &ru[static_cast<size_t>(y) * cs], static_cast<size_t>(cs));
      std::memcpy(&fr_cur->v[static_cast<size_t>(2 * y + par) * fr_cur->stride_c], &rv[static_cast<size_t>(y) * cs], static_cast<size_t>(cs));
    }
  }
  fr_cur->poc[fld.bottom ? 1 : 0] = this_poc;
  if (!fld.second) fr_cur->poc[fld.bottom ? 0 : 1] = this_poc;
  return ok;
}

// A frame as two field pictures: the first field (an IDR, intra or P picture), then a P field that may predict from it.
std::vector<uint8_t> H264Encoder::Impl::encode_field_pair(const EncPicture& frame, bool top_first, PicType t, int q, EncStats* stats) {
  if (!field_coding) return {};
  if ((t == PicType::kInter || t == PicType::kIntra) && !fr_prev) t = PicType::kIdr;
  const int fh = frame.h / 2, cwf = (frame.w + 1) / 2, chf = (frame.h / 2 + 1) / 2;
  EncPicture f[2];
  for (int par = 0; par < 2; ++par) {
    f[par].w = frame.w;
    f[par].h = fh;
    f[par].y.resize(static_cast<size_t>(frame.w) * fh);
    f[par].u.resize(static_cast<size_t>(cwf) * chf);
    f[par].v.resize(f[par].u.size());
    for (int y = 0; y < fh; ++y) std::memcpy(&f[par].y[static_cast<size_t>(y) * frame.w], &frame.y[static_cast<size_t>(2 * y + par) * frame.w], static_cast<size_t>(frame.w));
    // 4:2:0 chroma of an interlaced frame: its rows alternate between the fields as well
    for (int y = 0; y < chf; ++y) {
      const int src = std::min(2 * y + par, (frame.h + 1) / 2 - 1);
      std::memcpy(&f[par].u[static_cast<size_t>(y) * cwf], &frame.u[static_cast<size_t>(src) * cwf], static_cast<size_t>(cwf));
      std::memcpy(&f[par].v[static_cast<size_t>(y) * cwf], &frame.v[static_cast<size_t>(src) * cwf], static_cast<size_t>(cwf));
    }
  }
  const int p0 = top_first ? 0 : 1;
  fld.bottom = p0 == 1;
  fld.second = false;
  std::vector<uint8_t> out = code_picture(f[p0], t, q, 0);
  const size_t first_bytes = out.size();
  fld.bottom = p0 != 1;
  fld.second = true;
  const std::vector<uint8_t> second = code_picture(f[1 - p0], PicType::kInter, q, 0);
  out.insert(out.end(), second.begin(), second.end());
  ref_frame_num = frame_num;
  ++field_count;
  fr_prev = std::move(fr_cur);
  have_ref = true;
  st.bytes = out.size();
  (void)first_bytes;
  if (stats) {
    double sse = 0;
    for (int y = 0; y < frame.h; ++y)
      for (int x = 0; x < frame.w; ++x) {
        const int d = frame.y[static_cast<size_t>(y) * frame.w + x] - fr_prev->y[static_cast<size_t>(y) * fr_prev->stride_y + x];
        sse += static_cast<double>(d) * d;
      }
    st.psnr_y = sse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(frame.w) * frame.h / sse);
    *stats = st;
  }
  return out;
}


H264Encoder::H264Encoder(int width, int height, int fps_num, int fps_den, int profile_idc, const H264Options& opt)
    : p_(new Impl(width, height, fps_num, fps_den, profile_idc, opt, false)) {}
H264Encoder::H264Encoder(int width, int height, int fps_num, int fps_den, int profile_idc, bool deblock, bool field_pictures) {
  H264Options o;
  o.deblock = deblock;
  p_.reset(new Impl(width, height, fps_num, fps_den, profile_idc, o, field_pictures));
}
H264Encoder::~H264Encoder() = default;

std::vector<uint8_t> H264Encoder::parameter_sets() const { return p_->stream.parameter_sets(); }

bool H264Encoder::supports_b() const { return p_->b_mode; }

std::vector<uint8_t> H264Encoder::encode(const EncPicture& in, PicType type, int qp, EncStats* stats) {
  if (type == PicType::kIdr) p_->poc_counter = 0;
  return p_->encode_picture(in, type, qp, p_->poc_counter++, stats);
}
std::vector<uint8_t> H264Encoder::encode_at(const EncPicture& in, PicType type, int qp, int poc, EncStats* stats) {
  return p_->encode_picture(in, type, qp, poc, stats);
}

std::vector<uint8_t> H264Encoder::encode_field_pair(const EncPicture& frame, bool top_field_first, PicType type, int qp, EncStats* stats) {
  return p_->encode_field_pair(frame, top_field_first, type, qp, stats);
}

void H264Encoder::reset() {
  p_->have_ref = false;
  p_->dpb.clear();
  p_->fr_prev.reset();
  p_->fr_cur.reset();
  p_->poc_counter = 0;
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
