// The state of the HEVC decoder while it decodes one picture (hevc_decode.cpp and its hevc_dec_*.inc parts).
// Not a public header.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdio>
#include <vector>

#include "hevc_decode.hpp"
#include "hevc_syntax.hpp"

namespace vgpu_hevc {
namespace {

#include "hevc_tables.inc"

// Context variable layout: every syntax element's contexts, in one array (offsets below; sizes in comments).
enum Ctx {
  C_SAO_MERGE = 0,          // 1
  C_SAO_TYPE = 1,           // 1
  C_SPLIT_CU = 2,           // 3
  C_TQ_BYPASS = 5,          // 1
  C_SKIP = 6,               // 3
  C_PRED_MODE = 9,          // 1
  C_PART_MODE = 10,         // 4
  C_PREV_INTRA = 14,        // 1
  C_INTRA_CHROMA = 15,      // 1
  C_RQT_ROOT_CBF = 16,      // 1
  C_MERGE_FLAG = 17,        // 1
  C_MERGE_IDX = 18,         // 1
  C_INTER_PRED_IDC = 19,    // 5
  C_REF_IDX = 24,           // 2
  C_MVP = 26,               // 1
  C_SPLIT_TRANSFORM = 27,   // 3
  C_CBF_LUMA = 30,          // 2
  C_CBF_CHROMA = 32,        // 5
  C_MVD_G0 = 37,            // 1
  C_MVD_G1 = 38,            // 1
  C_QP_DELTA = 39,          // 2
  C_TSKIP = 41,             // 2: luma, chroma
  C_LAST_X = 43,            // 18
  C_LAST_Y = 61,            // 18
  C_CSBF = 79,              // 4
  C_SIG = 83,               // 44
  C_G1 = 127,               // 24
  C_G2 = 151,               // 6
  C_RDPCM_FLAG = 157,       // 2: luma, chroma
  C_RDPCM_DIR = 159,        // 2
  C_CQP_FLAG = 161,         // 1
  C_CQP_IDX = 162,          // 1
  C_RES_SCALE = 163,        // 8
  C_RES_SIGN = 171,         // 2
  C_COUNT = 173
};

// The motion of a 4x4 block.
struct MvField {
  int16_t mv[2][2];
  int8_t ref_idx[2];   // -1: the list is not used
};

// The motion data of an inter prediction unit before it is stored.
struct PuMotion {
  int16_t mv[2][2] = {};
  int8_t ref_idx[2] = {-1, -1};
  bool pred(int l) const { return ref_idx[l] >= 0; }
};

constexpr uint8_t kCuSkip = 1, kCuPcm = 2, kCuBypass = 4;

// A slice (the independent slice segment and the dependent ones that follow it): what is needed after its header is parsed.
struct SliceInfo {
  SliceHeader h;
  int addr_rs = 0;                  // SliceAddrRs
  int slice_qp = 26;
  RefPic ref[2][16];                // RefPicList0 / RefPicList1
  int ref_slot[2][16] = {};         // the slot in PicParams::refs each entry came from (picture identity)
  bool no_backward_pred = false;
  Frame* col_pic = nullptr;
  int col_slot = -1;
  int col_poc = 0;
};

struct SaoParams {
  uint8_t type[3] = {0, 0, 0};      // SaoTypeIdx
  uint8_t band_pos[3] = {0, 0, 0};
  uint8_t eo_class[3] = {0, 0, 0};
  int16_t offset[3][5] = {};        // SaoOffsetVal[cIdx][0..4]
};

class Dec {
 public:
  Dec(Frame* cur, const PicParams& pp, std::string* err);
  bool run(const std::vector<SliceData>& slices);

 private:
  // ---- configuration
  Frame* F;
  const PicParams& P;
  std::string* err_;
  bool ok_ = true;
  void fail(const char* why) {
    if (ok_ && err_) *err_ = why;
    ok_ = false;
  }

  int W = 0, H = 0, CW = 0, CH = 0;
  int log2_ctb = 4, ctb = 16, log2_min_cb = 3, log2_min_tb = 2, log2_max_tb = 5;
  int wctb = 0, hctb = 0, pic_size_ctbs = 0;
  int w4 = 0, h4 = 0;                 // picture size in 4x4 units, rounded up to whole CTBs
  int bd_y = 8, bd_c = 8;
  int qp_bd_offset_y = 0, qp_bd_offset_c = 0;
  int log2_min_cu_qp_delta = 0, log2_min_cu_chroma_qp_offset = 0;
  Pps pps_;                           // the tile layout only needs a few PPS fields
  Sps sps_;
  TileLayout tiles_;
  uint8_t sf4_[6][16], sf8_[6][64], sf16_[6][256], sf32_[2][1024];   // ScalingFactor, raster order
  std::vector<int> min_tb_addr_zs_;   // MinTbAddrZs[x][y], row-major with stride tbw_
  int tbw_ = 0, tbh_ = 0;

  // ---- per-picture arrays at 4x4 granularity (index (y>>2)*w4 + (x>>2))
  std::vector<uint8_t> pred_mode_;    // 0 not yet decoded, 1 inter, 2 intra
  std::vector<uint8_t> cu_flags_;     // kCu*
  std::vector<uint8_t> ct_depth_;
  std::vector<uint8_t> intra_mode_;
  std::vector<int8_t> qp_y_;
  std::vector<uint8_t> tu_nz_;
  std::vector<uint8_t> edge_;         // bit 0: vertical transform edge at the left, 1: vertical prediction edge, 2/3: horizontal at the top
  std::vector<MvField> mvf_;
  // ---- per-CTB
  std::vector<int16_t> ctb_slice_;    // index into slices_, -1 not decoded
  std::vector<SaoParams> sao_;
  std::vector<SliceInfo> slices_;

  // ---- slice segment state
  const uint8_t* sd_ = nullptr;       // slice segment RBSP
  size_t sd_size_ = 0;
  SliceInfo* S = nullptr;             // the current slice
  SliceHeader sh_;                    // the current slice segment header
  int cur_slice_idx_ = -1;
  int ctb_addr_rs_ = 0, ctb_addr_ts_ = 0;

  // ---- CABAC
  size_t bitpos_ = 0;
  uint32_t range_ = 510, offset_ = 0;
  uint8_t ctx_[C_COUNT];
  uint8_t wpp_ctx_[C_COUNT];
  uint8_t ds_ctx_[C_COUNT];
  uint8_t stat_coeff_[4] = {0, 0, 0, 0};
  uint8_t wpp_stat_[4] = {0, 0, 0, 0};
  uint8_t ds_stat_[4] = {0, 0, 0, 0};
  bool ds_valid_ = false;
  bool have_wpp_ = false;
  uint32_t read_bits(int n);
  void cabac_init_engine();
  int decode_bin(uint8_t& st);
  int decode_bypass();
  uint32_t decode_bypass_bits(int n);
  int decode_terminate();
  void init_contexts();

  // ---- CTU state
  int qp_y_pred_ = 0, cu_qp_y_ = 0, last_cu_qp_ = 0;
  bool qp_pred_valid_ = false;
  bool is_cu_qp_delta_coded_ = false;
  int cu_qp_delta_val_ = 0;
  bool is_cu_chroma_qp_offset_coded_ = false;
  int cu_qp_offset_cb_ = 0, cu_qp_offset_cr_ = 0;
  bool first_qg_in_slice_or_tile_ = true;
  bool cu_transquant_bypass_ = false;
  // current coding unit
  int cu_x_ = 0, cu_y_ = 0, cu_log2_ = 0;
  bool cu_intra_ = false;
  int part_mode_ = 0;
  int cu_depth_ = 0;
  bool intra_split_ = false;
  int intra_luma_mode_[4] = {1, 1, 1, 1};
  int intra_chroma_mode_ = 4;         // intra_chroma_pred_mode
  int intra_mode_c_ = 1;              // IntraPredModeC
  int max_trafo_depth_ = 0;
  bool merge_flag_cu_ = false;
  bool pcm_flag_ = false;

  // ---- availability and neighbours
  bool avail_zs(int xc, int yc, int xn, int yn) const;
  bool avail_pb(int xcb, int ycb, int ncbs, int xpb, int ypb, int npbw, int npbh, int part_idx, int xn, int yn) const;
  int idx4(int x, int y) const { return (y >> 2) * w4 + (x >> 2); }

  // ---- slice level
  bool decode_slice_segment(const SliceData& sd, bool first);
  bool build_ref_lists(SliceInfo& s);
  void decode_ctus();
  void sao_syntax(int rx, int ry);
  void coding_quadtree(int x0, int y0, int log2_cb, int depth);
  void coding_unit(int x0, int y0, int log2_cb);
  void prediction_unit(int xcb, int ycb, int ncbs, int xpb, int ypb, int npbw, int npbh, int part_idx);
  void pcm_sample(int x0, int y0, int log2_cb);
  void transform_tree(int x0, int y0, int xbase, int ybase, int log2_tb, int depth, int blk_idx, bool parent_cb, bool parent_cr);
  void transform_unit(int x0, int y0, int xbase, int ybase, int log2_tb, int depth, int blk_idx, bool cbf_luma, bool cb, bool cr);
  void delta_qp_syntax();
  void chroma_qp_offset_syntax();
  void compute_qp_pred(int xcb, int ycb);
  void set_cu_qp();
  int chroma_qp(int c_idx) const;
  void fill_cu_info(int x0, int y0, int log2_cb, uint8_t pm, uint8_t flags);

  // ---- residual
  void reconstruct_tb(int x0, int y0, int log2_tb, int c_idx, bool cbf, int pred_mode_intra, bool intra);
  bool residual_coding(int log2_tb, int c_idx, int pred_mode_intra, bool intra, int32_t* coeff, bool* ts, int* rdpcm_dir);

  // ---- intra
  void intra_predict(int x_cmp, int y_cmp, int log2_tb, int c_idx, int mode);
  int derive_luma_mode(int xpb, int ypb, int prev_flag, int mpm_idx, int rem_mode);

  // ---- inter
  void derive_merge(int xcb, int ycb, int ncbs, int xpb, int ypb, int npbw, int npbh, int part_idx, int merge_idx, PuMotion* out);
  void derive_mvp(int xcb, int ycb, int ncbs, int xpb, int ypb, int npbw, int npbh, int part_idx, int ref_idx, int lx, int mvp_flag, int16_t mvp[2]);
  bool temporal_mv(int xpb, int ypb, int npbw, int npbh, int ref_idx, int lx, int16_t mv[2]);
  bool col_mv(const ColMv& c, int ref_idx, int lx, int16_t mv[2]);
  void inter_predict(int xpb, int ypb, int npbw, int npbh, const PuMotion& m);
  void store_motion(int xpb, int ypb, int npbw, int npbh, const PuMotion& m);

  // ---- loop filters
  void deblock();
  void deblock_edges(bool vertical);
  int bs_of(int xq, int yq, int xp, int yp, bool tu_edge) const;
  void sao_filter();
  void finish_col();
};

}  // namespace
}  // namespace vgpu_hevc
