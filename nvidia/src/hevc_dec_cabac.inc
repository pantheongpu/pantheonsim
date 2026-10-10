// Part of hevc_decode.cpp: the CABAC arithmetic decoding engine (9.3.4.3) and the context variables (9.3.2.2).
// The engine is the one of the Recommendation, with the 9-bit offset register and one bit read per renormalization shift, so
// that the bit position after a terminating bin is exactly the one the syntax (pcm samples, byte alignment) needs.

// Table 9-52 (rangeTabLps, as in H.264's CABAC) and Table 9-53.
const uint8_t kRangeLps[64][4] = {
    {128, 176, 208, 240}, {128, 167, 197, 227}, {128, 158, 187, 216}, {123, 150, 178, 205}, {116, 142, 169, 195}, {111, 135, 160, 185}, {105, 128, 152, 175},
    {100, 122, 144, 166}, {95, 116, 137, 158},  {90, 110, 130, 150},  {85, 104, 123, 142},  {81, 99, 117, 135},   {77, 94, 111, 128},   {73, 89, 105, 122},
    {69, 85, 100, 116},   {66, 80, 95, 110},    {62, 76, 90, 104},    {59, 72, 86, 99},     {56, 69, 81, 94},     {53, 65, 77, 89},     {51, 62, 73, 85},
    {48, 59, 69, 80},     {46, 56, 66, 76},     {43, 53, 63, 72},     {41, 50, 59, 69},     {39, 48, 56, 65},     {37, 45, 54, 62},     {35, 43, 51, 59},
    {33, 41, 48, 56},     {32, 39, 46, 53},     {30, 37, 43, 50},     {29, 35, 41, 48},     {27, 33, 39, 45},     {26, 31, 37, 43},     {24, 30, 35, 41},
    {23, 28, 33, 39},     {22, 27, 32, 37},     {21, 26, 30, 35},     {20, 24, 29, 33},     {19, 23, 27, 31},     {18, 22, 26, 30},     {17, 21, 25, 28},
    {16, 20, 23, 27},     {15, 19, 22, 25},     {14, 18, 21, 24},     {14, 17, 20, 23},     {13, 16, 19, 22},     {12, 15, 18, 21},     {12, 14, 17, 20},
    {11, 14, 16, 19},     {11, 13, 15, 18},     {10, 12, 15, 17},     {10, 12, 14, 16},     {9, 11, 13, 15},      {9, 11, 12, 14},      {8, 10, 12, 14},
    {8, 9, 11, 13},       {7, 9, 11, 12},       {7, 9, 10, 12},       {7, 8, 10, 11},       {6, 8, 9, 11},        {6, 7, 9, 10},        {6, 7, 8, 9},
    {2, 2, 2, 2}};
const uint8_t kTransLps[64] = {0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9,  11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
                               24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63};

uint32_t Dec::read_bits(int n) {
  // n in 0..25; bits past the end of the data read as zero
  if (n == 0) return 0;
  const size_t byte = bitpos_ >> 3;
  const int sh = static_cast<int>(bitpos_ & 7);
  uint32_t v = 0;
  if (byte + 4 <= sd_size_) {
    v = static_cast<uint32_t>(sd_[byte]) << 24 | static_cast<uint32_t>(sd_[byte + 1]) << 16 | static_cast<uint32_t>(sd_[byte + 2]) << 8 | sd_[byte + 3];
  } else {
    for (int i = 0; i < 4; ++i) v = v << 8 | (byte + static_cast<size_t>(i) < sd_size_ ? sd_[byte + static_cast<size_t>(i)] : 0);
  }
  v <<= sh;
  bitpos_ += static_cast<size_t>(n);
  return v >> (32 - n);
}

void Dec::cabac_init_engine() {
  range_ = 510;
  offset_ = read_bits(9);
}

int Dec::decode_bin(uint8_t& st) {
  int p = st >> 1, mps = st & 1;
  const uint32_t lps = kRangeLps[p][(range_ >> 6) & 3];
  range_ -= lps;
  int bin;
  if (offset_ >= range_) {
    bin = 1 - mps;
    offset_ -= range_;
    range_ = lps;
    if (p == 0) mps = 1 - mps;
    p = kTransLps[p];
  } else {
    bin = mps;
    if (p < 62) ++p;
  }
  st = static_cast<uint8_t>(p << 1 | mps);
  if (range_ < 256) {
    const int n = __builtin_clz(range_) - 23;
    range_ <<= n;
    offset_ = (offset_ << n) | read_bits(n);
  }
  return bin;
}

int Dec::decode_bypass() {
  offset_ = (offset_ << 1) | read_bits(1);
  if (offset_ >= range_) {
    offset_ -= range_;
    return 1;
  }
  return 0;
}

uint32_t Dec::decode_bypass_bits(int n) {
  uint32_t v = 0;
  for (int i = 0; i < n; ++i) v = v << 1 | static_cast<uint32_t>(decode_bypass());
  return v;
}

int Dec::decode_terminate() {
  range_ -= 2;
  if (offset_ >= range_) return 1;
  if (range_ < 256) {
    const int n = __builtin_clz(range_) - 23;
    range_ <<= n;
    offset_ = (offset_ << n) | read_bits(n);
  }
  return 0;
}

// 9.3.2.2: the initial (pStateIdx, valMps) of every context for the slice's QP and initType.
void Dec::init_contexts() {
  static uint8_t table[3][C_COUNT];
  static bool built = false;
  if (!built) {
    for (int t = 0; t < 3; ++t) {
      uint8_t* v = table[t];
      std::memset(v, 154, C_COUNT);
      v[C_SAO_MERGE] = kInit_sao_merge[t];
      v[C_SAO_TYPE] = kInit_sao_type_idx[t];
      for (int i = 0; i < 3; ++i) v[C_SPLIT_CU + i] = kInit_split_cu_flag[3 * t + i];
      v[C_TQ_BYPASS] = kInit_cu_transquant_bypass_flag[t];
      if (t > 0) {
        for (int i = 0; i < 3; ++i) v[C_SKIP + i] = kInit_cu_skip_flag[3 * (t - 1) + i];
        v[C_PRED_MODE] = kInit_pred_mode_flag[t - 1];
        v[C_RQT_ROOT_CBF] = kInit_rqt_root_cbf[t - 1];
        v[C_MERGE_FLAG] = kInit_merge_flag[t - 1];
        v[C_MERGE_IDX] = kInit_merge_idx[t - 1];
        for (int i = 0; i < 5; ++i) v[C_INTER_PRED_IDC + i] = kInit_inter_pred_idc[5 * (t - 1) + i];
        for (int i = 0; i < 2; ++i) v[C_REF_IDX + i] = kInit_ref_idx[2 * (t - 1) + i];
        v[C_MVP] = kInit_mvp_flag[t - 1];
        v[C_MVD_G0] = kInit_abs_mvd[t == 1 ? 0 : 2];
        v[C_MVD_G1] = kInit_abs_mvd[t == 1 ? 1 : 3];
        v[C_RDPCM_FLAG] = kInit_explicit_rdpcm_flag[t - 1];
        v[C_RDPCM_FLAG + 1] = kInit_explicit_rdpcm_flag[2 + t - 1];
        v[C_RDPCM_DIR] = kInit_explicit_rdpcm_dir_flag[t - 1];
        v[C_RDPCM_DIR + 1] = kInit_explicit_rdpcm_dir_flag[2 + t - 1];
      }
      if (t == 0) v[C_PART_MODE] = kInit_part_mode[0];
      else
        for (int i = 0; i < 4; ++i) v[C_PART_MODE + i] = kInit_part_mode[1 + 4 * (t - 1) + i];
      v[C_PREV_INTRA] = kInit_prev_intra_luma_pred_flag[t];
      v[C_INTRA_CHROMA] = kInit_intra_chroma_pred_mode[t];
      for (int i = 0; i < 3; ++i) v[C_SPLIT_TRANSFORM + i] = kInit_split_transform_flag[3 * t + i];
      for (int i = 0; i < 2; ++i) v[C_CBF_LUMA + i] = kInit_cbf_luma[2 * t + i];
      for (int i = 0; i < 4; ++i) v[C_CBF_CHROMA + i] = kInit_cbf_chroma[4 * t + i];
      v[C_CBF_CHROMA + 4] = kInit_cbf_chroma[12 + t];
      for (int i = 0; i < 2; ++i) v[C_QP_DELTA + i] = kInit_cu_qp_delta_abs[2 * t + i];
      v[C_TSKIP] = kInit_transform_skip_flag[t];
      v[C_TSKIP + 1] = kInit_transform_skip_flag[3 + t];
      for (int i = 0; i < 18; ++i) {
        v[C_LAST_X + i] = kInit_last_sig_coeff_x_prefix[18 * t + i];
        v[C_LAST_Y + i] = kInit_last_sig_coeff_y_prefix[18 * t + i];
      }
      for (int i = 0; i < 4; ++i) v[C_CSBF + i] = kInit_coded_sub_block_flag[4 * t + i];
      for (int i = 0; i < 42; ++i) v[C_SIG + i] = kInit_sig_coeff_flag[42 * t + i];
      v[C_SIG + 42] = kInit_sig_coeff_flag[126 + 2 * t];
      v[C_SIG + 43] = kInit_sig_coeff_flag[126 + 2 * t + 1];
      for (int i = 0; i < 24; ++i) v[C_G1 + i] = kInit_coeff_abs_level_greater1_flag[24 * t + i];
      for (int i = 0; i < 6; ++i) v[C_G2 + i] = kInit_coeff_abs_level_greater2_flag[6 * t + i];
      v[C_CQP_FLAG] = kInit_cu_chroma_qp_offset_flag[t];
      v[C_CQP_IDX] = kInit_cu_chroma_qp_offset_idx[t];
      for (int i = 0; i < 8; ++i) v[C_RES_SCALE + i] = kInit_log2_res_scale_abs_plus1[8 * t + i];
      for (int i = 0; i < 2; ++i) v[C_RES_SIGN + i] = kInit_res_scale_sign_flag[2 * t + i];
    }
    built = true;
  }
  const int st = sh_.slice_type;
  const int init_type = st == kSliceI ? 0 : (st == kSliceP ? (sh_.cabac_init_flag ? 2 : 1) : (sh_.cabac_init_flag ? 1 : 2));
  const int qp = std::max(0, std::min(51, S->slice_qp));
  for (int i = 0; i < C_COUNT; ++i) {
    const int init = table[init_type][i];
    const int slope = init >> 4, off = init & 15;
    const int m = slope * 5 - 45, n = (off << 3) - 16;
    const int pre = std::max(1, std::min(126, ((m * qp) >> 4) + n));
    const int mps = pre <= 63 ? 0 : 1;
    const int state = mps ? pre - 64 : 63 - pre;
    ctx_[i] = static_cast<uint8_t>(state << 1 | mps);
  }
  std::memset(stat_coeff_, 0, sizeof stat_coeff_);
}
