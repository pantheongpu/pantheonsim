// Part of h264_decode.cpp (included inside its anonymous namespace): the decoder object.

// Neighbouring samples gathered for Intra_NxN prediction: top[0] is the corner p[-1,-1], top[1 + x] is p[x, -1]
// (x < 2n), left[y] is p[-1, y].
struct IntraNb {
  int top[1 + 16];
  int left[8];
  bool tl = false, top_av = false, tr_av = false, left_av = false;
};

class Dec {
 public:
  Dec(Frame* cur, const PicParams& pp) : cur_(cur), pp_(pp), hc_(header_ctx(pp)) {}
  bool run(const std::vector<SliceData>& slices, std::string* err);

 private:
  // ---- picture ----
  Frame* cur_;
  const PicParams& pp_;
  HeaderCtx hc_;
  PicData* pd_ = nullptr;
  int W_ = 0, H_ = 0, nmb_ = 0;        // picture size in macroblocks (a field has half the frame's height)
  bool mbaff_ = false, field_pic_ = false, bottom_ = false;
  Plane py_, pu_, pv_;                 // the picture being decoded (frame, or one field of the frame store)
  std::string err_;
  int cur_poc_ = 0;                    // PicOrderCnt(CurrPic)
  int max_frame_num_ = 16;

  // ---- slice ----
  SliceHeader sh_;
  int slice_id_ = 0;
  std::vector<uint8_t> rbsp_;
  BitReader br_;
  Cabac cab_;
  bool cabac_ = false;
  int qp_ = 0;                         // QPY of the previous macroblock in decoding order
  std::vector<RefPic> ref_[2];         // RefPicList0 / 1 (frames or fields, as the picture's structure)
  int num_ref_[2] = {0, 0};
  bool prev_mb_skipped_ = false;
  int last_qp_delta_ = 0;

  // ---- current macroblock ----
  int addr_ = 0;
  int mbx_ = 0, mby_ = 0;              // position in macroblocks (MBAFF: of the pair, and the row of the pair)
  MbInfo* mb_ = nullptr;
  bool mb_field_ = false;              // field macroblock (MBAFF) or field picture
  uint8_t* ly_ = nullptr;              // samples of the macroblock and the strides between their rows
  uint8_t* lu_ = nullptr;
  uint8_t* lv_ = nullptr;
  int ls_ = 0, lcs_ = 0;
  int org_x_ = 0, org_y_ = 0;          // luma position of the macroblock in the (field or frame) reference geometry
  int mbaddr_a_ = -1, mbaddr_b_ = -1, nb_c_ = -1, nb_d_ = -1;   // neighbouring macroblocks (6.4.10)
  int pair_a_ = -1, pair_b_ = -1, pair_c_ = -1, pair_d_ = -1;   // MBAFF: top macroblock of the neighbouring pairs

  // syntax of the current macroblock
  int mb_type_ = 0;
  int part_shape_ = 0;                 // inter: 0 16x16, 1 16x8, 2 8x16, 3 8x8, 4 B_Direct_16x16
  uint8_t part_pred_[4] = {};          // prediction flags per partition (bit 0 list 0, bit 1 list 1)
  uint8_t sub_shape_[4] = {};
  uint8_t sub_pred_[4] = {};
  bool sub_direct_[4] = {};
  int cbp_ = 0;
  int i16_mode_ = 0;
  bool is_skip_ = false;
  int32_t luma_coef_[256];             // 16 4x4 blocks of 16 (or 4 8x8 blocks of 64), raster order inside the block
  int32_t i16_dc_[16];
  int32_t chroma_dc_[2][4];
  int32_t chroma_ac_[2][4][16];
  uint16_t blk_done_ = 0;              // 4x4 blocks of the macroblock whose motion data is final, in partition order

  // ---- slice level ----
  bool decode_slice(const SliceData& sd);
  bool build_ref_lists();
  bool decode_skipped_mb(int addr, bool last_in_run);
  bool decode_coded_mb(int addr);
  bool decode_mb_cabac(int addr, bool* skipped, bool* more);
  void do_skip_mb();
  void complete_mb();
  void finish_mb();
  void finalize_motion();

  // ---- neighbours ----
  bool avail(int addr) const { return addr >= 0 && addr < nmb_ && pd_->mb[addr].slice == slice_id_; }
  Nb locate(int xN, int yN, int maxW, int maxH) const;
  Nb locate_l(int xN, int yN) const { return locate(xN, yN, 16, 16); }
  Nb locate_c(int xN, int yN) const { return locate(xN, yN, 8, 8); }
  void setup_mb(int addr);
  const MbInfo& mbi(int addr) const { return pd_->mb[addr]; }
  uint8_t* luma_at(int addr, int x, int y) const;
  uint8_t* chroma_at(int plane, int addr, int x, int y) const;

  // ---- parsing ----
  bool parse_macroblock();
  int cabac_mb_type_i(int ctx_base, bool intra_slice);
  int cabac_mb_type();
  int cabac_sub_mb_type();
  int cabac_mb_skip();
  int cabac_ref_idx(int list, int b8);
  int cabac_mvd(int list, int comp, int blk4);
  int cabac_cbp();
  int cabac_qp_delta();
  int cabac_intra_chroma_mode();
  int cabac_cbf_ctx(int cat, int comp, int bx, int by);
  int cabac_residual_block(int cat, int ctx_inc_cbf, int max_coef, int* levels, bool field_scan);
  int cavlc_residual_block(int nC, int max_coef, int* levels);
  int nc_luma(int bx, int by);
  int nc_chroma(int comp, int bx, int by);
  bool parse_residual();
  bool parse_intra_pred_modes();
  bool parse_inter_pred();
  int te_ref_idx(int range);

  // ---- reconstruction ----
  void recon_intra();
  void add_residual_luma();
  void add_residual_chroma();
  void recon_inter_residual();
  void predict_intra16(int mode);
  void predict_chroma(int mode);
  bool intra_nb_avail(int addr) const;
  void gather_intra_nb(int xo, int yo, int n, bool tr_blocked, IntraNb* nb);
  void ipcm_samples();

  // ---- inter prediction ----
  void inter_predict_mb();
  void mc_block(int x4, int y4, int w4, int h4);
  void set_motion(int x4, int y4, int w4, int h4, int list, int ref, int mvx, int mvy);
  void set_mvd(int list, int x4, int y4, int w4, int h4, int dx, int dy);
  int pred_flags_of(int b8) const;
  void get_nb_motion(int list, int xl, int yl, int* ref, int* mvx, int* mvy, bool* avail_out);
  void mv_pred(int list, int x4, int y4, int w4, int h4, int ref, int shape, int part_idx, int* mvx, int* mvy);
  void derive_pskip();
  void derive_direct(int which);
  void direct_spatial(uint8_t mask);
  void direct_temporal(uint8_t mask);
  bool colocated(int blk8, int sub4, int* ref_col, int* mv_col, int* col_ref_id, bool* col_field, int* vert_scale);
  RefPic ref_val(int list, int idx) const;
  int ref_id_of(int list, int idx) const;
  int cur_poc_for_mb() const;
  void ref_planes(const RefPic& r, Plane* y, Plane* u, Plane* v) const;

  // ---- deblocking ----
  void deblock_picture();
  void deblock_mb(int addr);
  void deblock_mb_mbaff(int addr);

  bool fail(const char* what) {
    if (err_.empty()) err_ = what;
    return false;
  }
};

#include "h264_dec_slice.inc"
#include "h264_dec_parse.inc"
#include "h264_dec_recon.inc"
#include "h264_dec_inter.inc"
#include "h264_dec_deblock.inc"
