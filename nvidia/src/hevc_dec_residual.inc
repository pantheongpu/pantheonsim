// Part of hevc_decode.cpp: residual_coding() (7.3.8.11, 9.3.4.2.3 to 9.3.4.2.7), scaling (8.6.3), the transforms (8.6.4),
// transform skip, lossless coding and residual DPCM (8.6.2, 8.6.5), and the picture construction (8.6.7).

namespace {

// ScanOrder[ log2BlockSize ][ scanIdx ][ sPos ] = (x, y) for blocks of 1x1 to 8x8 (6.5.3 to 6.5.5).
struct ScanTables {
  uint8_t xy[4][3][64][2];
  ScanTables() {
    for (int l = 0; l < 4; ++l) {
      const int n = 1 << l;
      int i = 0, x = 0, y = 0;
      bool stop = false;
      while (!stop) {   // up-right diagonal
        while (y >= 0) {
          if (x < n && y < n) {
            xy[l][0][i][0] = static_cast<uint8_t>(x);
            xy[l][0][i][1] = static_cast<uint8_t>(y);
            ++i;
          }
          --y;
          ++x;
        }
        y = x;
        x = 0;
        if (i >= n * n) stop = true;
      }
      i = 0;
      for (y = 0; y < n; ++y)
        for (x = 0; x < n; ++x, ++i) {
          xy[l][1][i][0] = static_cast<uint8_t>(x);
          xy[l][1][i][1] = static_cast<uint8_t>(y);
        }
      i = 0;
      for (x = 0; x < n; ++x)
        for (y = 0; y < n; ++y, ++i) {
          xy[l][2][i][0] = static_cast<uint8_t>(x);
          xy[l][2][i][1] = static_cast<uint8_t>(y);
        }
    }
  }
};
const ScanTables& scans() {
  static const ScanTables t;
  return t;
}

const uint8_t kCtxIdxMap4x4[16] = {0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8};
const int kLevelScale[6] = {40, 45, 51, 57, 64, 72};

// ScalingFactor of 8.6.2 / 7.4.5 for one transform size and matrix, raster order [y * n + x].
void build_scaling_factor(const uint8_t* list, int dc, int log2_size, uint8_t* out) {
  const ScanTables& st = scans();
  if (log2_size == 2) {
    for (int i = 0; i < 16; ++i) out[st.xy[2][0][i][1] * 4 + st.xy[2][0][i][0]] = list[i];
    return;
  }
  const int rep = 1 << (log2_size - 3), n = 1 << log2_size;
  for (int i = 0; i < 64; ++i) {
    const int x = st.xy[3][0][i][0], y = st.xy[3][0][i][1];
    for (int j = 0; j < rep; ++j)
      for (int k = 0; k < rep; ++k) out[(y * rep + j) * n + x * rep + k] = list[i];
  }
  if (log2_size >= 4) out[0] = static_cast<uint8_t>(dc);
}

}  // namespace

// Reads the residual_coding() syntax structure of one transform block into `coeff` (raster order, n*n, zeroed here).
// Returns false when the data is malformed.
bool Dec::residual_coding(int log2_tb, int c_idx, int pred_mode_intra, bool intra, int32_t* coeff, bool* ts_out, int* rdpcm_dir) {
  const int n = 1 << log2_tb;
  std::memset(coeff, 0, sizeof(int32_t) * static_cast<size_t>(n) * static_cast<size_t>(n));
  const ScanTables& st = scans();
  bool ts = false;
  if (P.transform_skip_enabled && !cu_transquant_bypass_ && log2_tb <= P.log2_max_transform_skip) ts = decode_bin(ctx_[C_TSKIP + (c_idx ? 1 : 0)]) != 0;
  bool explicit_rdpcm = false;
  int explicit_dir = 0;
  if (!intra && P.explicit_rdpcm && (ts || cu_transquant_bypass_)) {
    explicit_rdpcm = decode_bin(ctx_[C_RDPCM_FLAG + (c_idx ? 1 : 0)]) != 0;
    if (explicit_rdpcm) explicit_dir = decode_bin(ctx_[C_RDPCM_DIR + (c_idx ? 1 : 0)]);
  }
  *ts_out = ts;
  *rdpcm_dir = -1;
  // last significant coefficient position
  int ctx_offset, ctx_shift;
  if (c_idx == 0) {
    ctx_offset = 3 * (log2_tb - 2) + ((log2_tb - 1) >> 2);
    ctx_shift = (log2_tb + 1) >> 2;
  } else {
    ctx_offset = 15;
    ctx_shift = log2_tb - 2;
  }
  const int cmax = (log2_tb << 1) - 1;
  int px = 0, py = 0;
  while (px < cmax && decode_bin(ctx_[C_LAST_X + ctx_offset + (px >> ctx_shift)])) ++px;
  while (py < cmax && decode_bin(ctx_[C_LAST_Y + ctx_offset + (py >> ctx_shift)])) ++py;
  int last_x = px, last_y = py;
  if (px > 3) {
    const int nb = (px >> 1) - 1;
    last_x = (1 << nb) * (2 + (px & 1)) + static_cast<int>(decode_bypass_bits(nb));
  }
  if (py > 3) {
    const int nb = (py >> 1) - 1;
    last_y = (1 << nb) * (2 + (py & 1)) + static_cast<int>(decode_bypass_bits(nb));
  }
  // scanIdx (7.4.9.11)
  int scan_idx = 0;
  if (intra && (log2_tb == 2 || (log2_tb == 3 && c_idx == 0))) {
    if (pred_mode_intra >= 6 && pred_mode_intra <= 14) scan_idx = 2;
    else if (pred_mode_intra >= 22 && pred_mode_intra <= 30) scan_idx = 1;
  }
  if (scan_idx == 2) std::swap(last_x, last_y);
  if (last_x >= n || last_y >= n) {
    fail("last significant coefficient outside the block");
    return false;
  }
  const int sb_log2 = log2_tb - 2;
  const int sb_n = 1 << sb_log2;
  // position of the last coefficient in the scan
  int last_sub_block = 0, last_scan_pos = 0;
  {
    const int xs = last_x >> 2, ys = last_y >> 2;
    for (int i = 0; i < sb_n * sb_n; ++i)
      if (st.xy[sb_log2][scan_idx][i][0] == xs && st.xy[sb_log2][scan_idx][i][1] == ys) {
        last_sub_block = i;
        break;
      }
    for (int i = 0; i < 16; ++i)
      if (st.xy[2][scan_idx][i][0] == (last_x & 3) && st.xy[2][scan_idx][i][1] == (last_y & 3)) {
        last_scan_pos = i;
        break;
      }
  }
  // signHidden condition inputs
  const bool implicit_rdpcm_block = intra && P.implicit_rdpcm && ts && (pred_mode_intra == 10 || pred_mode_intra == 26);
  const bool no_sign_hiding = cu_transquant_bypass_ || implicit_rdpcm_block || explicit_rdpcm;
  const bool sig_ctx_ts = P.transform_skip_context && (ts || cu_transquant_bypass_);
  uint8_t csbf[8][8];
  std::memset(csbf, 0, sizeof csbf);
  int prev_greater1_ctx_zero = -1;   // -1: no sub-block processed yet; else whether greater1Ctx ended at 0
  int c1 = 1;
  bool first_sub_block = true;
  const int sb_type = 2 * (c_idx == 0 ? 1 : 0) + ((ts || cu_transquant_bypass_) ? 1 : 0);
  (void)prev_greater1_ctx_zero;
  for (int i = last_sub_block; i >= 0; --i) {
    const int xs = st.xy[sb_log2][scan_idx][i][0], ys = st.xy[sb_log2][scan_idx][i][1];
    bool infer_sb_dc = false;
    if (i < last_sub_block && i > 0) {
      int ctx = 0;
      if (xs < sb_n - 1) ctx += csbf[ys][xs + 1];
      if (ys < sb_n - 1) ctx += csbf[ys + 1][xs];
      csbf[ys][xs] = static_cast<uint8_t>(decode_bin(ctx_[C_CSBF + std::min(ctx, 1) + (c_idx ? 2 : 0)]));
      infer_sb_dc = true;
    } else {
      csbf[ys][xs] = 1;
    }
    uint8_t sig[16];
    std::memset(sig, 0, sizeof sig);
    int start = 15;
    if (i == last_sub_block) {
      sig[last_scan_pos] = 1;
      start = last_scan_pos - 1;
    }
    if (csbf[ys][xs]) {
      // prevCsbf for the context derivation
      int prev_csbf = 0;
      if (xs < sb_n - 1) prev_csbf |= csbf[ys][xs + 1];
      if (ys < sb_n - 1) prev_csbf |= csbf[ys + 1][xs] << 1;
      for (int nn = start; nn >= 0; --nn) {
        if (nn > 0 || !infer_sb_dc) {
          const int xp = st.xy[2][scan_idx][nn][0], yp = st.xy[2][scan_idx][nn][1];
          const int xc = (xs << 2) + xp, yc = (ys << 2) + yp;
          int sig_ctx;
          if (sig_ctx_ts) {
            sig_ctx = c_idx == 0 ? 42 : 16;
          } else if (log2_tb == 2) {
            sig_ctx = kCtxIdxMap4x4[(yc << 2) + xc];
          } else if (xc + yc == 0) {
            sig_ctx = 0;
          } else {
            switch (prev_csbf) {
              case 0: sig_ctx = (xp + yp == 0) ? 2 : (xp + yp < 3) ? 1 : 0; break;
              case 1: sig_ctx = (yp == 0) ? 2 : (yp == 1) ? 1 : 0; break;
              case 2: sig_ctx = (xp == 0) ? 2 : (xp == 1) ? 1 : 0; break;
              default: sig_ctx = 2; break;
            }
            if (c_idx == 0) {
              if (xs + ys > 0) sig_ctx += 3;
              sig_ctx += (log2_tb == 3) ? (scan_idx == 0 ? 9 : 15) : 21;
            } else {
              sig_ctx += (log2_tb == 3) ? 9 : 12;
            }
          }
          const int inc = c_idx == 0 ? sig_ctx : 27 + sig_ctx;
          sig[nn] = static_cast<uint8_t>(decode_bin(ctx_[C_SIG + inc]));
          if (sig[nn]) infer_sb_dc = false;
        } else {
          sig[0] = 1;   // inferred: the DC position of a coded sub-block with no other significant coefficient
        }
      }
    }
    // significant coefficients of the sub-block, in decoding order (high scan position first)
    int pos[16], num_sig = 0;
    for (int nn = 15; nn >= 0; --nn)
      if (sig[nn]) pos[num_sig++] = nn;
    if (num_sig == 0) continue;
    // coeff_abs_level_greater1_flag / greater2_flag
    int ctx_set = (i > 0 && c_idx == 0) ? 2 : 0;
    if (!first_sub_block && c1 == 0) ++ctx_set;
    first_sub_block = false;
    c1 = 1;
    int g1[16] = {0}, g2[16] = {0};
    int last_greater1_pos = -1;   // index into pos[]
    const int n_g1 = std::min(num_sig, 8);
    for (int k = 0; k < n_g1; ++k) {
      const int inc = ctx_set * 4 + std::min(3, c1) + (c_idx ? 16 : 0);
      g1[k] = decode_bin(ctx_[C_G1 + inc]);
      if (g1[k]) {
        c1 = 0;
        if (last_greater1_pos < 0) last_greater1_pos = k;
      } else if (c1 > 0 && c1 < 3) {
        ++c1;
      }
    }
    if (last_greater1_pos >= 0) g2[last_greater1_pos] = decode_bin(ctx_[C_G2 + ctx_set + (c_idx ? 4 : 0)]);
    const int first_sig_scan_pos = pos[num_sig - 1], last_sig_scan_pos = pos[0];
    const bool sign_hidden = !no_sign_hiding && (last_sig_scan_pos - first_sig_scan_pos > 3);
    const bool hide = P.sign_data_hiding && sign_hidden;
    const int n_signs = hide ? num_sig - 1 : num_sig;
    uint32_t signs = decode_bypass_bits(n_signs);
    signs <<= (hide ? 1 : 0);   // bit (num_sig-1-k) of `signs` is the sign of the k-th coefficient
    // coeff_abs_level_remaining
    int rice = P.persistent_rice_adaptation ? stat_coeff_[sb_type] / 4 : 0;
    bool first_remaining = true;
    int num_sig_coeff = 0;
    int sum_abs = 0;
    for (int k = 0; k < num_sig; ++k) {
      const int base = 1 + g1[k] + g2[k];
      int abs_level = base;
      const int thresh = (num_sig_coeff < 8) ? ((k == last_greater1_pos) ? 3 : 2) : 1;
      if (base == thresh) {
        // coeff_abs_level_remaining: TR prefix (up to four ones) and an EGk suffix, binarisation of 9.3.3.11
        int prefix = 0;
        while (prefix < 32 && decode_bypass()) ++prefix;
        int rem;
        if (prefix <= 3) {
          rem = (prefix << rice) + static_cast<int>(decode_bypass_bits(rice));
        } else {
          const int len = prefix - 3 + rice;
          if (len > 25) {
            fail("coefficient level too large");
            return false;
          }
          rem = (((1 << (prefix - 3)) + 3 - 1) << rice) + static_cast<int>(decode_bypass_bits(len));
        }
        if (P.persistent_rice_adaptation && first_remaining) {
          const int s = stat_coeff_[sb_type];
          if (rem >= (3 << (s / 4))) ++stat_coeff_[sb_type];
          else if (2 * rem < (1 << (s / 4)) && s > 0) --stat_coeff_[sb_type];
        }
        first_remaining = false;
        abs_level = base + rem;
        // Rice parameter update for the next coefficient
        if (abs_level > 3 * (1 << rice)) rice = P.persistent_rice_adaptation ? rice + 1 : std::min(rice + 1, 4);
      }
      ++num_sig_coeff;
      int val = abs_level;
      sum_abs += abs_level;
      const int sbit = (k == num_sig - 1 && hide) ? ((sum_abs & 1) ? 1 : 0) : static_cast<int>((signs >> (num_sig - 1 - k)) & 1);
      if (sbit) val = -val;
      if (val > 32767) val = 32767;
      if (val < -32768) val = -32768;
      const int nn = pos[k];
      const int xc = (xs << 2) + st.xy[2][scan_idx][nn][0], yc = (ys << 2) + st.xy[2][scan_idx][nn][1];
      coeff[yc * n + xc] = val;
    }
  }
  // residual DPCM direction: implicit (intra, 7.3.8.11 / 8.4.4.1) or explicit (inter)
  if (explicit_rdpcm) *rdpcm_dir = explicit_dir;
  else if (intra && P.implicit_rdpcm && (ts || cu_transquant_bypass_) && (pred_mode_intra == 10 || pred_mode_intra == 26)) *rdpcm_dir = pred_mode_intra / 26;
  return true;
}

namespace {

// 8.6.4.2: one dimension of the inverse transform; `x` has n entries at `stride` apart (frequency order), result n entries into y.
void idct_1d(const int32_t* x, int xstride, int32_t* y, int n, int last_nz) {
  const int step = 32 / n;
  for (int i = 0; i < n; ++i) {
    int64_t sum = 0;
    for (int j = 0; j <= last_nz; ++j) {
      const int32_t v = x[j * xstride];
      if (v) sum += static_cast<int64_t>(kTransMatrix[j * step][i]) * v;
    }
    y[i] = static_cast<int32_t>(sum);
  }
}

const int8_t kDst4[4][4] = {{29, 55, 74, 84}, {74, 74, 0, -74}, {84, -29, -74, 55}, {55, -84, 74, -29}};

}  // namespace

// Scaling, transform and residual modification of one transform block (8.6.2); coefficient levels in, residual samples out.
// `res` is n*n, raster order.
static void inverse_transform(int32_t* coeff, int log2_tb, bool dst, int bd, int32_t* res) {
  const int n = 1 << log2_tb;
  // find the last non-zero column / row to skip zero work
  int max_row = -1, max_col = -1;
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x)
      if (coeff[y * n + x]) {
        max_row = std::max(max_row, y);
        max_col = std::max(max_col, x);
      }
  if (max_row < 0) {
    std::memset(res, 0, sizeof(int32_t) * static_cast<size_t>(n) * static_cast<size_t>(n));
    return;
  }
  int32_t tmp[32 * 32];
  int32_t col[32];
  // stage 1: columns (vertical), first shift 7
  for (int x = 0; x < n; ++x) {
    if (x > max_col) {
      for (int y = 0; y < n; ++y) tmp[y * n + x] = 0;
      continue;
    }
    if (dst) {
      for (int i = 0; i < 4; ++i) {
        int64_t s = 0;
        for (int j = 0; j < 4; ++j) s += static_cast<int64_t>(kDst4[j][i]) * coeff[j * n + x];
        col[i] = static_cast<int32_t>(s);
      }
    } else {
      idct_1d(coeff + x, n, col, n, max_row);
    }
    for (int y = 0; y < n; ++y) {
      int32_t g = (col[y] + 64) >> 7;
      g = std::max(-32768, std::min(32767, g));
      tmp[y * n + x] = g;
    }
  }
  // stage 2: rows (horizontal)
  const int shift = 20 - bd;
  const int32_t round = shift > 0 ? 1 << (shift - 1) : 0;
  for (int y = 0; y < n; ++y) {
    int32_t row[32];
    if (dst) {
      for (int i = 0; i < 4; ++i) {
        int64_t s = 0;
        for (int j = 0; j < 4; ++j) s += static_cast<int64_t>(kDst4[j][i]) * tmp[y * n + j];
        row[i] = static_cast<int32_t>(s);
      }
    } else {
      idct_1d(tmp + y * n, 1, row, n, max_col);
    }
    for (int x = 0; x < n; ++x) res[y * n + x] = (row[x] + round) >> shift;
  }
}

void Dec::reconstruct_tb(int x0, int y0, int log2_tb, int c_idx, bool cbf, int mode, bool intra) {
  const int n = 1 << log2_tb;
  if (intra) intra_predict(x0, y0, log2_tb, c_idx, mode);
  if (!cbf) return;
  int32_t coeff[32 * 32];
  bool ts = false;
  int rdpcm_dir = -1;
  if (!residual_coding(log2_tb, c_idx, mode, intra, coeff, &ts, &rdpcm_dir)) return;
  const int bd = c_idx == 0 ? bd_y : bd_c;
  int32_t res[32 * 32];
  if (cu_transquant_bypass_) {
    const bool rotate = P.transform_skip_rotation && n == 4 && intra;
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) res[y * n + x] = rotate ? coeff[(n - 1 - y) * n + (n - 1 - x)] : coeff[y * n + x];
  } else {
    // 8.6.3 scaling
    const int qp = c_idx == 0 ? cu_qp_y_ + qp_bd_offset_y : chroma_qp(c_idx);
    const int qp_c = std::max(0, c_idx == 0 ? std::min(qp, 51 + qp_bd_offset_y) : qp);
    const int bd_shift = bd + log2_tb - 5;
    const int64_t scale = static_cast<int64_t>(kLevelScale[qp_c % 6]) << (qp_c / 6);
    const uint8_t* m = nullptr;
    if (P.scaling_list_enabled && !(ts && n > 4)) {
      const int size_id = log2_tb - 2;
      const int matrix_id = (intra ? 0 : 3) + c_idx;
      m = size_id == 0 ? sf4_[matrix_id] : size_id == 1 ? sf8_[matrix_id] : size_id == 2 ? sf16_[matrix_id] : sf32_[intra ? 0 : 1];
    }
    for (int i = 0; i < n * n; ++i) {
      if (!coeff[i]) continue;
      const int64_t f = m ? m[i] : 16;
      int64_t d = (static_cast<int64_t>(coeff[i]) * f * scale + (static_cast<int64_t>(1) << (bd_shift - 1))) >> bd_shift;
      coeff[i] = static_cast<int32_t>(std::max<int64_t>(-32768, std::min<int64_t>(32767, d)));
    }
    if (ts) {
      const bool rotate = P.transform_skip_rotation && n == 4 && intra;
      const int ts_shift = 5 + log2_tb;
      const int shift = 20 - bd;
      const int32_t round = shift > 0 ? 1 << (shift - 1) : 0;
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          const int32_t d = rotate ? coeff[(n - 1 - y) * n + (n - 1 - x)] : coeff[y * n + x];
          res[y * n + x] = ((d * (1 << ts_shift)) + round) >> shift;
        }
    } else {
      inverse_transform(coeff, log2_tb, intra && n == 4 && c_idx == 0, bd, res);
    }
  }
  if (rdpcm_dir == 0) {
    for (int y = 0; y < n; ++y)
      for (int x = 1; x < n; ++x) res[y * n + x] += res[y * n + x - 1];
  } else if (rdpcm_dir == 1) {
    for (int y = 1; y < n; ++y)
      for (int x = 0; x < n; ++x) res[y * n + x] += res[(y - 1) * n + x];
  }
  // 8.6.7 picture construction
  std::vector<uint16_t>& pl = c_idx == 0 ? F->y : (c_idx == 1 ? F->u : F->v);
  const int stride = c_idx == 0 ? F->stride_y : F->stride_c;
  const int maxv = (1 << bd) - 1;
  for (int y = 0; y < n; ++y) {
    uint16_t* row = &pl[static_cast<size_t>(y0 + y) * static_cast<size_t>(stride) + static_cast<size_t>(x0)];
    for (int x = 0; x < n; ++x) row[x] = static_cast<uint16_t>(std::max(0, std::min(maxv, static_cast<int>(row[x]) + res[y * n + x])));
  }
}
