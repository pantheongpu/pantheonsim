// Part of h264_decode.cpp: macroblock layer syntax (7.3.5) with CAVLC (9.1, 9.2) and CABAC (9.3).

// ---- CABAC syntax elements ------------------------------------------------------------------------------
int Dec::cabac_mb_skip() {
  const int base = sh_.is_b() ? 24 : 11;
  int inc = 0;
  if (mbaddr_a_ >= 0 && !mbi(mbaddr_a_).skip) ++inc;
  if (mbaddr_b_ >= 0 && !mbi(mbaddr_b_).skip) ++inc;
  return cab_.decision(base + inc);
}

// mb_field_decoding_flag (9.3.3.1.1.2): the context counts the neighbouring pairs, left and above, that are field pairs.
int Dec::cabac_field_flag() {
  const int pair = addr_ >> 1, px = pair % W_, py = pair / W_;
  int inc = 0;
  if (px > 0 && avail(2 * (pair - 1)) && mbi(2 * (pair - 1)).field) ++inc;
  if (py > 0 && avail(2 * (pair - W_)) && mbi(2 * (pair - W_)).field) ++inc;
  return cab_.decision(70 + inc);
}

// Intra macroblock types (Table 9-36) with the context offsets of I slices (3) or of the intra prefix in P (17) / B (32) slices.
int Dec::cabac_mb_type_i(int base, bool intra_slice) {
  int ctx0 = base;
  if (intra_slice) {
    int inc = 0;
    if (mbaddr_a_ >= 0 && mbi(mbaddr_a_).kind != kI4x4 && mbi(mbaddr_a_).kind != kI8x8) ++inc;
    if (mbaddr_b_ >= 0 && mbi(mbaddr_b_).kind != kI4x4 && mbi(mbaddr_b_).kind != kI8x8) ++inc;
    ctx0 = base + inc;
  }
  if (!cab_.decision(ctx0)) return 0;   // I_NxN
  if (cab_.terminate()) return 25;      // I_PCM
  // Table 9-39: the context increments of the later bins differ between I slices and the prefix in P / B slices
  int t = 1;
  t += 12 * cab_.decision(base + (intra_slice ? 3 : 1));                    // luma bit (binIdx 2)
  const int b3 = cab_.decision(base + (intra_slice ? 4 : 2));                // chroma != 0 (binIdx 3)
  if (b3) t += 4 + 4 * cab_.decision(base + (intra_slice ? 5 : 2));         // chroma == 2 (binIdx 4)
  t += 2 * cab_.decision(base + (intra_slice ? 6 : 3));                      // prediction mode, high bit
  t += 1 * cab_.decision(base + (intra_slice ? 7 : 3));                      // prediction mode, low bit
  return t;
}

int Dec::cabac_mb_type() {
  if (sh_.is_intra()) return cabac_mb_type_i(3, true);
  if (sh_.is_p()) {
    if (cab_.decision(14) == 0) {
      if (cab_.decision(15) == 0) return 3 * cab_.decision(16);   // P_L0_16x16 / P_8x8
      return 2 - cab_.decision(17);                               // P_L0_L0_16x8 / 8x16
    }
    return 5 + cabac_mb_type_i(17, false);
  }
  // B
  int inc = 0;
  if (mbaddr_a_ >= 0 && !mbi(mbaddr_a_).skip && !mbi(mbaddr_a_).direct16) ++inc;
  if (mbaddr_b_ >= 0 && !mbi(mbaddr_b_).skip && !mbi(mbaddr_b_).direct16) ++inc;
  if (!cab_.decision(27 + inc)) return 0;                              // B_Direct_16x16
  if (!cab_.decision(27 + 3)) return 1 + cab_.decision(27 + 5);       // B_L0_16x16 / B_L1_16x16
  int bits = cab_.decision(27 + 4) << 3;
  bits |= cab_.decision(27 + 5) << 2;
  bits |= cab_.decision(27 + 5) << 1;
  bits |= cab_.decision(27 + 5);
  if (bits < 8) return bits + 3;
  if (bits == 13) return 23 + cabac_mb_type_i(32, false);
  if (bits == 14) return 11;
  if (bits == 15) return 22;
  bits = bits << 1 | cab_.decision(27 + 5);
  return bits - 4;
}

int Dec::cabac_sub_mb_type() {
  if (sh_.is_p()) {
    if (cab_.decision(21)) return 0;
    if (!cab_.decision(22)) return 1;
    if (cab_.decision(23)) return 2;
    return 3;
  }
  if (!cab_.decision(36)) return 0;
  if (!cab_.decision(37)) return 1 + cab_.decision(39);
  int type = 3;
  if (cab_.decision(38)) {
    if (cab_.decision(39)) return 11 + cab_.decision(39);
    type += 4;
  }
  type += 2 * cab_.decision(39);
  type += cab_.decision(39);
  return type;
}

// ref_idx of the 8x8 block `b8` of the current macroblock, list `list`.
int Dec::cabac_ref_idx(int list, int b8) {
  const int x = (b8 & 1) * 8, y = (b8 >> 1) * 8;
  int inc = 0;
  for (int n = 0; n < 2; ++n) {
    const Nb nb = n == 0 ? locate_l(x - 1, y) : locate_l(x, y - 1);
    if (nb.addr < 0) continue;
    const MbInfo& m = nb.addr == addr_ ? *mb_ : mbi(nb.addr);
    if (m.intra || m.skip) continue;
    const int nb8 = (nb.y >> 3) * 2 + (nb.x >> 3);
    if (!(m.coded_flags[nb8] & (1 << list))) continue;
    int r = m.ref_idx[list][nb8];
    if (mbaff_ && !mb_field_ && m.field) r >>= 1;   // refIdxZeroFlagN: ref_idx > 1 when a frame macroblock sees a field one
    if (r > 0) inc += n == 0 ? 1 : 2;
  }
  int ref = 0;
  int ctx = 54 + inc;
  while (cab_.decision(ctx)) {
    ++ref;
    ctx = 54 + (ref == 1 ? 4 : 5);   // binIdx 1 uses ctxIdxInc 4, later bins 5
    if (ref >= 32) return -1;
  }
  return ref;
}

// mvd component `comp` of the 4x4 block (bx, by) of the current macroblock
int Dec::cabac_mvd(int list, int comp, int blk) {
  const int bx = blk & 3, by = blk >> 2;
  int sum = 0;
  for (int n = 0; n < 2; ++n) {
    const Nb nb = n == 0 ? locate_l(bx * 4 - 1, by * 4) : locate_l(bx * 4, by * 4 - 1);
    if (nb.addr < 0) continue;
    const MbInfo& m = nb.addr == addr_ ? *mb_ : mbi(nb.addr);
    if (m.intra || m.skip) continue;
    int v = m.mvd[list][(nb.y >> 2) * 4 + (nb.x >> 2)][comp];
    if (comp == 1 && mbaff_) {
      if (!mb_field_ && m.field) v *= 2;
      else if (mb_field_ && !m.field) v /= 2;
    }
    sum += v;
  }
  const int base = comp == 0 ? 40 : 47;
  const int inc = sum < 3 ? 0 : (sum > 32 ? 2 : 1);
  if (!cab_.decision(base + inc)) return 0;
  int mvd = 1;
  int ctx = base + 3;
  while (mvd < 9 && cab_.decision(ctx)) {
    if (mvd < 4) ++ctx;
    ++mvd;
  }
  if (mvd >= 9) {
    int k = 3;
    while (cab_.bypass()) {
      mvd += 1 << k;
      ++k;
      if (k > 24) return 0;
    }
    while (k--) mvd += cab_.bypass() << k;
  }
  return cab_.bypass() ? -mvd : mvd;
}

int Dec::cabac_cbp() {
  int cbp = 0;
  for (int b8 = 0; b8 < 4; ++b8) {
    const int x = (b8 & 1) * 8, y = (b8 >> 1) * 8;
    int inc = 0;
    for (int n = 0; n < 2; ++n) {
      const Nb nb = n == 0 ? locate_l(x - 1, y) : locate_l(x, y - 1);
      int cond;
      if (nb.addr < 0) cond = 0;
      else if (nb.addr == addr_) cond = ((cbp >> ((nb.y >> 3) * 2 + (nb.x >> 3))) & 1) == 0;
      else {
        const MbInfo& m = mbi(nb.addr);
        if (m.kind == kIPCM) cond = 0;
        else if (m.skip) cond = 1;
        else cond = ((m.cbp >> ((nb.y >> 3) * 2 + (nb.x >> 3))) & 1) == 0;
      }
      inc += cond << n;
    }
    cbp |= cab_.decision(73 + inc) << b8;
  }
  // chroma
  int chroma = 0;
  {
    int inc = 0;
    for (int n = 0; n < 2; ++n) {
      const int a = n == 0 ? mbaddr_a_ : mbaddr_b_;
      if (a < 0) continue;
      const MbInfo& m = mbi(a);
      const int c = (m.cbp >> 4) & 3;
      if (m.kind == kIPCM || (!m.skip && c != 0)) inc += 1 << n;
    }
    if (cab_.decision(77 + inc)) {
      inc = 0;
      for (int n = 0; n < 2; ++n) {
        const int a = n == 0 ? mbaddr_a_ : mbaddr_b_;
        if (a < 0) continue;
        const MbInfo& m = mbi(a);
        if (m.kind == kIPCM || (!m.skip && ((m.cbp >> 4) & 3) == 2)) inc += 1 << n;
      }
      chroma = 1 + cab_.decision(77 + 4 + inc);
    }
  }
  return cbp | chroma << 4;
}

int Dec::cabac_qp_delta() {
  int ctx = 60 + (last_qp_delta_ != 0 ? 1 : 0);
  if (!cab_.decision(ctx)) return 0;
  int v = 1;
  ctx = 60 + 2;
  while (cab_.decision(ctx)) {
    ctx = 60 + 3;
    ++v;
    if (v > 200) return 0;
  }
  return (v & 1) ? (v + 1) >> 1 : -(v >> 1);
}

int Dec::cabac_intra_chroma_mode() {
  int inc = 0;
  if (mbaddr_a_ >= 0 && mbi(mbaddr_a_).intra && mbi(mbaddr_a_).kind != kIPCM && mbi(mbaddr_a_).chroma_mode != 0) ++inc;
  if (mbaddr_b_ >= 0 && mbi(mbaddr_b_).intra && mbi(mbaddr_b_).kind != kIPCM && mbi(mbaddr_b_).chroma_mode != 0) ++inc;
  if (!cab_.decision(64 + inc)) return 0;
  if (!cab_.decision(64 + 3)) return 1;
  return 2 + cab_.decision(64 + 3);
}

// The coded_block_flag context increment for a block of category `cat` (0 luma DC of Intra16x16, 1 Intra16x16 AC,
// 2 luma 4x4, 3 chroma DC, 4 chroma AC), at 4x4 block (bx, by) of the plane (luma 4x4 units; chroma 2x2).
int Dec::cabac_cbf_ctx(int cat, int comp, int bx, int by) {
  int inc = 0;
  for (int n = 0; n < 2; ++n) {
    Nb nb;
    if (cat == 0 || cat == 3) {
      nb.addr = n == 0 ? mbaddr_a_ : mbaddr_b_;
    } else if (cat == 4) {
      nb = n == 0 ? locate_c(bx * 4 - 1, by * 4) : locate_c(bx * 4, by * 4 - 1);
    } else {
      nb = n == 0 ? locate_l(bx * 4 - 1, by * 4) : locate_l(bx * 4, by * 4 - 1);
    }
    int cond;
    if (nb.addr < 0) {
      cond = mb_->intra ? 1 : 0;
    } else {
      const MbInfo& m = nb.addr == addr_ ? *mb_ : mbi(nb.addr);
      if (m.kind == kIPCM) cond = 1;
      else if (cat == 0) cond = (m.cbf_dc & 1) ? 1 : 0;
      else if (cat == 3) cond = (m.cbf_dc >> (1 + comp)) & 1;
      else if (cat == 4) cond = m.nz[1 + comp][(nb.y >> 2) * 2 + (nb.x >> 2)] != 0;
      else cond = m.nz[0][(nb.y >> 2) * 4 + (nb.x >> 2)] != 0;
    }
    inc += cond << n;
  }
  return inc;
}

// Residual block with CABAC (7.3.5.3.3). Returns the number of non-zero coefficients (0 if coded_block_flag is 0).
// levels[] is indexed by scan position (0 .. max_coef - 1).
int Dec::cabac_residual_block(int cat, int cbf_inc, int max_coef, int* levels, bool field_scan) {
  static const int kCbfOff[6] = {0, 4, 8, 12, 16, 0};
  static const int kSigOff[6] = {0, 15, 29, 44, 47, 0};
  static const int kAbsOff[6] = {0, 10, 20, 30, 39, 0};
  const int cbf_base = cat < 5 ? 85 : 1012;
  if (cat != 5) {
    if (!cab_.decision(cbf_base + kCbfOff[cat] + cbf_inc)) return 0;
  }
  const int sig_base = cat == 5 ? (field_scan ? 436 : 402) : (field_scan ? 277 : 105) + kSigOff[cat];
  const int last_base = cat == 5 ? (field_scan ? 451 : 417) : (field_scan ? 338 : 166) + kSigOff[cat];
  const int abs_base = cat == 5 ? 426 : 227 + kAbsOff[cat];
  int sig_pos[64];
  int n = 0;
  const int last_idx = max_coef - 1;
  int i = 0;
  for (; i < last_idx; ++i) {
    int sinc, linc;
    if (cat == 3) sinc = linc = std::min(i, 2);
    else if (cat == 5) {
      sinc = field_scan ? kSig8x8Field[i] : kSig8x8Frame[i];
      linc = kLast8x8[i];
    } else sinc = linc = i;
    if (cab_.decision(sig_base + sinc)) {
      sig_pos[n++] = i;
      if (cab_.decision(last_base + linc)) break;
    }
  }
  if (i == last_idx) sig_pos[n++] = last_idx;   // the last coefficient is significant if the map did not end earlier
  int num_gt1 = 0, num_eq1 = 0;
  for (int k = n - 1; k >= 0; --k) {
    int ctx = abs_base + (num_gt1 != 0 ? 0 : std::min(4, 1 + num_eq1));
    int abs_m1 = 0;
    if (cab_.decision(ctx)) {
      ctx = abs_base + 5 + std::min(4 - (cat == 3 ? 1 : 0), num_gt1);
      abs_m1 = 1;
      while (abs_m1 < 14 && cab_.decision(ctx)) ++abs_m1;
      if (abs_m1 >= 14) {
        int kk = 0;
        while (cab_.bypass()) {
          abs_m1 += 1 << kk;
          ++kk;
          if (kk > 24) return 0;
        }
        while (kk--) abs_m1 += cab_.bypass() << kk;
      }
    }
    if (abs_m1 == 0) ++num_eq1;
    else ++num_gt1;
    const int level = abs_m1 + 1;
    levels[sig_pos[k]] = std::max(-(1 << 17), std::min(cab_.bypass() ? -level : level, 1 << 17));   // a conforming level fits 16 bits
  }
  return n;
}

// ---- CAVLC residual block (9.2) ----------------------------------------------------------------------------------
int Dec::cavlc_residual_block(int nC, int max_coef, int* levels) {
  // coeff_token
  int t1 = -1, tc = -1;
  if (nC >= 8) {
    const uint32_t v = br_.u(6);
    if (v == 3) {
      tc = 0;
      t1 = 0;
    } else {
      tc = static_cast<int>(v >> 2) + 1;
      t1 = static_cast<int>(v & 3);
      if (t1 > tc) return -1;
    }
  } else {
    const auto* table = nC == -1 ? kCoeffTokenChromaDc : (nC < 2 ? kCoeffToken0 : (nC < 4 ? kCoeffToken1 : kCoeffToken2));
    // match the code bit by bit against the table
    uint32_t code = 0;
    int len = 0;
    while (len < 16 && t1 < 0) {
      code = code << 1 | br_.bit();
      ++len;
      for (int a = 0; a < 4 && t1 < 0; ++a)
        for (int b = 0; b < 17; ++b)
          if (table[a][b].len == len && table[a][b].bits == code) {
            t1 = a;
            tc = b;
            break;
          }
    }
    if (t1 < 0) return -1;
  }
  if (tc > max_coef) return -1;
  if (tc == 0) return 0;
  int level[16];
  int suffix_len = (tc > 10 && t1 < 3) ? 1 : 0;
  for (int i = 0; i < tc; ++i) {
    if (i < t1) {
      level[i] = br_.bit() ? -1 : 1;
      continue;
    }
    int prefix = 0;
    while (!br_.bit()) {
      if (++prefix > 31 || br_.overrun()) return -1;
    }
    int code = std::min(15, prefix) << suffix_len;
    int suffix_size = (prefix == 14 && suffix_len == 0) ? 4 : (prefix >= 15 ? prefix - 3 : suffix_len);
    if (suffix_size > 0) code += static_cast<int>(br_.u(suffix_size));
    if (prefix >= 15 && suffix_len == 0) code += 15;
    if (prefix >= 16) code += (1 << (prefix - 3)) - 4096;
    if (i == t1 && t1 < 3) code += 2;
    int lv = (code & 1) ? (-code - 1) >> 1 : (code + 2) >> 1;
    level[i] = lv;
    if (suffix_len == 0) suffix_len = 1;
    if (std::abs(lv) > (3 << (suffix_len - 1)) && suffix_len < 6) ++suffix_len;
  }
  int total_zeros = 0;
  if (tc < max_coef) {
    if (max_coef == 4) {
      const auto& tab = kTotalZerosChromaDc[tc - 1];
      uint32_t code = 0;
      int len = 0;
      bool ok = false;
      while (len < 3 && !ok) {
        code = code << 1 | br_.bit();
        ++len;
        for (int z = 0; z < 4; ++z)
          if (tab[z].len == len && tab[z].bits == code) {
            total_zeros = z;
            ok = true;
            break;
          }
      }
      if (!ok) return -1;
    } else {
      const auto& tab = kTotalZeros[tc - 1];
      uint32_t code = 0;
      int len = 0;
      bool ok = false;
      while (len < 9 && !ok) {
        code = code << 1 | br_.bit();
        ++len;
        for (int z = 0; z < 16; ++z)
          if (tab[z].len == len && tab[z].bits == code) {
            total_zeros = z;
            ok = true;
            break;
          }
      }
      if (!ok) return -1;
    }
    if (total_zeros + tc > max_coef) return -1;
  }
  int run[16];
  int zeros_left = total_zeros;
  for (int i = 0; i < tc - 1; ++i) {
    if (zeros_left > 0) {
      const auto& tab = kRunBefore[std::min(zeros_left, 7) - 1];
      uint32_t code = 0;
      int len = 0;
      int r = -1;
      while (len < 11 && r < 0) {
        code = code << 1 | br_.bit();
        ++len;
        for (int z = 0; z < 15; ++z)
          if (tab[z].len == len && tab[z].bits == code) {
            r = z;
            break;
          }
      }
      if (r < 0 || r > zeros_left) return -1;
      run[i] = r;
      zeros_left -= r;
    } else {
      run[i] = 0;
    }
  }
  run[tc - 1] = zeros_left;
  int coeff_num = -1;
  for (int i = tc - 1; i >= 0; --i) {
    coeff_num += run[i] + 1;
    if (coeff_num >= max_coef) return -1;
    levels[coeff_num] = std::max(-(1 << 17), std::min(level[i], 1 << 17));   // a conforming level fits 16 bits
  }
  return tc;
}

int Dec::nc_luma(int bx, int by) {
  const Nb a = locate_l(bx * 4 - 1, by * 4), b = locate_l(bx * 4, by * 4 - 1);
  int na = 0, nb = 0;
  const bool av_a = a.addr >= 0, av_b = b.addr >= 0;
  if (av_a) {
    const MbInfo& m = a.addr == addr_ ? *mb_ : mbi(a.addr);
    na = m.nz[0][(a.y >> 2) * 4 + (a.x >> 2)];
  }
  if (av_b) {
    const MbInfo& m = b.addr == addr_ ? *mb_ : mbi(b.addr);
    nb = m.nz[0][(b.y >> 2) * 4 + (b.x >> 2)];
  }
  if (av_a && av_b) return (na + nb + 1) >> 1;
  if (av_a) return na;
  if (av_b) return nb;
  return 0;
}

int Dec::nc_chroma(int comp, int bx, int by) {
  const Nb a = locate_c(bx * 4 - 1, by * 4), b = locate_c(bx * 4, by * 4 - 1);
  int na = 0, nb = 0;
  const bool av_a = a.addr >= 0, av_b = b.addr >= 0;
  if (av_a) {
    const MbInfo& m = a.addr == addr_ ? *mb_ : mbi(a.addr);
    na = m.nz[1 + comp][(a.y >> 2) * 2 + (a.x >> 2)];
  }
  if (av_b) {
    const MbInfo& m = b.addr == addr_ ? *mb_ : mbi(b.addr);
    nb = m.nz[1 + comp][(b.y >> 2) * 2 + (b.x >> 2)];
  }
  if (av_a && av_b) return (na + nb + 1) >> 1;
  if (av_a) return na;
  if (av_b) return nb;
  return 0;
}

int Dec::te_ref_idx(int range) {
  // te(v): range is the largest value
  if (range > 1) return static_cast<int>(br_.ue());
  return br_.bit() ? 0 : 1;
}

// ---- residual (7.3.5.3) ----------------------------------------------------------------------------------
bool Dec::parse_residual() {
  std::memset(luma_coef_, 0, sizeof luma_coef_);
  std::memset(i16_dc_, 0, sizeof i16_dc_);
  std::memset(chroma_dc_, 0, sizeof chroma_dc_);
  std::memset(chroma_ac_, 0, sizeof chroma_ac_);
  const bool i16 = mb_->kind == kI16x16;
  const bool t8 = mb_->t8x8;
  const bool fscan = mb_field_;
  auto sc4 = [&](int i) { return static_cast<int>(fscan ? kField4x4[i] : kZigzag4x4[i]); };
  auto sc8 = [&](int i) { return static_cast<int>(fscan ? kField8x8[i] : kZigzag8x8[i]); };
  int levels[64];

  if (i16) {
    std::memset(levels, 0, sizeof levels);
    int n;
    if (cabac_) n = cabac_residual_block(0, cabac_cbf_ctx(0, 0, 0, 0), 16, levels, fscan);
    else n = cavlc_residual_block(nc_luma(0, 0), 16, levels);
    if (n < 0) return fail("bad Intra16x16 DC coefficients");
    if (n > 0) mb_->cbf_dc |= 1;
    for (int i = 0; i < 16; ++i) i16_dc_[sc4(i)] = levels[i];
  }
  for (int b8 = 0; b8 < 4; ++b8) {
    if (!((cbp_ >> b8) & 1)) continue;
    if (t8 && cabac_) {
      std::memset(levels, 0, sizeof levels);
      const int n = cabac_residual_block(5, 0, 64, levels, fscan);
      for (int i = 0; i < 64; ++i)
        if (levels[i]) luma_coef_[b8 * 64 + sc8(i)] = levels[i];
      const int x4 = (b8 & 1) * 2, y4 = (b8 >> 1) * 2;
      for (int k = 0; k < 4; ++k) mb_->nz[0][(y4 + (k >> 1)) * 4 + x4 + (k & 1)] = static_cast<uint8_t>(n);
      continue;
    }
    for (int s = 0; s < 4; ++s) {
      const int blk = b8 * 4 + s;
      const int bx = blk_x(blk), by = blk_y(blk);
      std::fill_n(levels, 16, 0);
      int n;
      const int maxc = i16 ? 15 : 16;
      if (cabac_) {
        n = cabac_residual_block(i16 ? 1 : 2, cabac_cbf_ctx(i16 ? 1 : 2, 0, bx, by), maxc, levels, fscan);
      } else {
        n = cavlc_residual_block(nc_luma(bx, by), maxc, levels);
      }
      if (n < 0) return fail("bad luma coefficients");
      mb_->nz[0][by * 4 + bx] = static_cast<uint8_t>(n);
      if (n == 0) continue;
      if (t8) {
        for (int i = 0; i < 16; ++i)
          if (levels[i]) luma_coef_[b8 * 64 + sc8(4 * i + s)] = levels[i];
      } else {
        for (int i = 0; i < maxc; ++i)
          if (levels[i]) luma_coef_[blk * 16 + sc4(i + (i16 ? 1 : 0))] = levels[i];
      }
    }
  }
  if (((cbp_ >> 4) & 3) != 0) {
    for (int c = 0; c < 2; ++c) {
      std::memset(levels, 0, sizeof levels);
      int n;
      if (cabac_) n = cabac_residual_block(3, cabac_cbf_ctx(3, c, 0, 0), 4, levels, fscan);
      else n = cavlc_residual_block(-1, 4, levels);
      if (n < 0) return fail("bad chroma DC coefficients");
      if (n > 0) mb_->cbf_dc |= 1 << (1 + c);
      for (int i = 0; i < 4; ++i) chroma_dc_[c][i] = levels[i];
    }
  }
  if (((cbp_ >> 4) & 3) == 2) {
    for (int c = 0; c < 2; ++c)
      for (int b = 0; b < 4; ++b) {
        const int bx = b & 1, by = b >> 1;
        std::fill_n(levels, 16, 0);
        int n;
        if (cabac_) n = cabac_residual_block(4, cabac_cbf_ctx(4, c, bx, by), 15, levels, fscan);
        else n = cavlc_residual_block(nc_chroma(c, bx, by), 15, levels);
        if (n < 0) return fail("bad chroma AC coefficients");
        mb_->nz[1 + c][by * 2 + bx] = static_cast<uint8_t>(n);
        if (n == 0) continue;
        for (int i = 0; i < 15; ++i)
          if (levels[i]) chroma_ac_[c][b][sc4(i + 1)] = levels[i];
      }
  }
  return true;
}

// ---- intra prediction modes -------------------------------------------------------------------------------------
bool Dec::parse_intra_pred_modes() {
  const bool i8 = mb_->kind == kI8x8;
  const int count = i8 ? 4 : 16;
  for (int i = 0; i < count; ++i) {
    const int blk = i8 ? i * 4 : i;
    const int bx = blk_x(blk), by = blk_y(blk);
    int prev_flag, rem = 0;
    if (cabac_) {
      prev_flag = cab_.decision(68);
      if (!prev_flag) {
        rem = cab_.decision(69);
        rem |= cab_.decision(69) << 1;
        rem |= cab_.decision(69) << 2;
      }
    } else {
      prev_flag = static_cast<int>(br_.u(1));
      if (!prev_flag) rem = static_cast<int>(br_.u(3));
    }
    // predicted mode (8.3.1.1 / 8.3.2.1)
    const Nb a = locate_l(bx * 4 - 1, by * 4), b = locate_l(bx * 4, by * 4 - 1);
    int mode_a = 2, mode_b = 2;
    bool dc_pred = false;
    for (int n = 0; n < 2; ++n) {
      const Nb& nb = n == 0 ? a : b;
      if (nb.addr < 0) {
        dc_pred = true;
        continue;
      }
      const MbInfo& m = nb.addr == addr_ ? *mb_ : mbi(nb.addr);
      if (!m.intra && pp_.constrained_intra_pred) {
        dc_pred = true;
        continue;
      }
      int mode = 2;
      if (m.kind == kI4x4 || m.kind == kI8x8) {
        if (i8 && m.kind == kI4x4) {
          // Intra8x8: the neighbouring 4x4 block named by (8-72)
          const int nb8 = (nb.y >> 3) * 2 + (nb.x >> 3);
          int k = n == 0 ? 1 : 2;
          if (n == 0 && mbaff_ && !mb_field_ && m.field && i == 2) k = 3;
          const int idx = nb8 * 4 + k;
          mode = m.ipred[blk_raster(idx)];
        } else {
          mode = m.ipred[(nb.y >> 2) * 4 + (nb.x >> 2)];
        }
      }
      (n == 0 ? mode_a : mode_b) = mode;
    }
    const int pred = dc_pred ? 2 : std::min(mode_a, mode_b);
    const int mode = prev_flag ? pred : (rem < pred ? rem : rem + 1);
    if (i8) {
      for (int k = 0; k < 4; ++k) mb_->ipred[blk_raster(blk + k)] = static_cast<int8_t>(mode);
    } else {
      mb_->ipred[by * 4 + bx] = static_cast<int8_t>(mode);
    }
  }
  return true;
}

// ---- inter prediction syntax and motion vectors ---------------------------------------------------------------------
bool Dec::parse_inter_pred() {
  const bool is_b = sh_.is_b();
  // reference indices
  const int range_unit = (mbaff_ && mb_field_) ? 2 : 1;
  int ref[2][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
  int nparts = part_shape_ == 0 ? 1 : (part_shape_ == 3 ? 4 : 2);
  // partition geometry in 8x8 blocks: which 8x8 blocks each partition covers
  auto part_b8 = [&](int p, int* first, int* count) {
    if (part_shape_ == 0) { *first = 0; *count = 4; }
    else if (part_shape_ == 1) { *first = p * 2; *count = 2; }   // 16x8: blocks 0,1 / 2,3
    else if (part_shape_ == 2) { *first = p; *count = 2; }       // 8x16: blocks 0,2 / 1,3
    else { *first = p; *count = 1; }
  };
  auto b8_of_part = [&](int p, int k) {
    int first, count;
    part_b8(p, &first, &count);
    if (part_shape_ == 2) return first + 2 * k;
    return first + k;
  };
  for (int l = 0; l < (is_b ? 2 : 1); ++l) {
    for (int p = 0; p < nparts; ++p) {
      const uint8_t pr = part_shape_ == 3 ? sub_pred_[p] : part_pred_[p];
      if (part_shape_ == 3 && sub_direct_[p]) continue;
      if (!(pr & (1 << l))) continue;
      int r = 0;
      const int active = sh_.num_ref_idx_active[l] * range_unit;
      if (mb_type_ == -1 /* P_8x8ref0 */) {
        r = 0;
      } else if (active > 1) {
        // the macroblock's own ref_idx values must be visible to the context derivation of later partitions
        const int b8_first = part_shape_ == 3 ? p : b8_of_part(p, 0);
        if (cabac_) r = cabac_ref_idx(l, b8_first);
        else r = te_ref_idx(active - 1);
        if (r < 0 || r >= active) return fail("ref_idx out of range");
      }
      ref[l][p] = r;
      // record immediately (CABAC contexts of the next partitions read it)
      int first, count;
      if (part_shape_ == 3) { first = p; count = 1; }
      else part_b8(p, &first, &count);
      for (int k = 0; k < count; ++k) {
        const int b8 = part_shape_ == 3 ? p : b8_of_part(p, k);
        mb_->ref_idx[l][b8] = static_cast<int8_t>(r);
        mb_->coded_flags[b8] |= static_cast<uint8_t>(1 << l);
      }
      (void)first;
    }
  }
  // ids of the referenced pictures
  auto set_ids = [&]() {
    for (int l = 0; l < 2; ++l)
      for (int b8 = 0; b8 < 4; ++b8) {
        const int r = mb_->ref_idx[l][b8];
        mb_->ref_id[l][b8] = r >= 0 ? ref_id_of(l, r) : -1;
      }
  };
  // Direct blocks of B_8x8 are derived first (spatial direct looks at the macroblock's neighbours only).
  if (is_b && part_shape_ == 3) {
    for (int p = 0; p < 4; ++p)
      if (sub_direct_[p]) mb_->direct8 |= static_cast<uint8_t>(1 << p);
    if (mb_->direct8) derive_direct(-2);
  }
  // motion vectors
  for (int l = 0; l < (is_b ? 2 : 1); ++l) {
    blk_done_ = 0;
    for (int p = 0; p < nparts; ++p) {
      int bx4, by4, w4, h4;   // the partition in 4x4 units
      if (part_shape_ == 3) {
        const int sx = (p & 1) * 2, sy = (p >> 1) * 2;
        if (sub_direct_[p]) {
          for (int k = 0; k < 4; ++k) blk_done_ |= static_cast<uint16_t>(1 << ((sy + (k >> 1)) * 4 + sx + (k & 1)));
          continue;
        }
        const uint8_t pr = sub_pred_[p];
        const int shape = sub_shape_[p];
        const int nsub = shape == 0 ? 1 : (shape == 3 ? 4 : 2);
        for (int s = 0; s < nsub; ++s) {
          int ox, oy;
          if (shape == 0) { ox = 0; oy = 0; w4 = 2; h4 = 2; }
          else if (shape == 1) { ox = 0; oy = s; w4 = 2; h4 = 1; }
          else if (shape == 2) { ox = s; oy = 0; w4 = 1; h4 = 2; }
          else { ox = s & 1; oy = s >> 1; w4 = 1; h4 = 1; }
          bx4 = sx + ox;
          by4 = sy + oy;
          int mvx = 0, mvy = 0;
          if (pr & (1 << l)) {
            mv_pred(l, bx4, by4, w4, h4, ref[l][p], 3, p, &mvx, &mvy);
            int dx, dy;
            if (cabac_) {
              dx = cabac_mvd(l, 0, by4 * 4 + bx4);
              dy = cabac_mvd(l, 1, by4 * 4 + bx4);
            } else {
              dx = br_.se();
              dy = br_.se();
            }
            set_mvd(l, bx4, by4, w4, h4, dx, dy);
            mvx += dx;
            mvy += dy;
          }
          set_motion(bx4, by4, w4, h4, l, (pr & (1 << l)) ? ref[l][p] : -1, mvx, mvy);
          for (int yy = 0; yy < h4; ++yy)
            for (int xx = 0; xx < w4; ++xx) blk_done_ |= static_cast<uint16_t>(1 << ((by4 + yy) * 4 + bx4 + xx));
        }
        continue;
      }
      if (part_shape_ == 0) { bx4 = 0; by4 = 0; w4 = 4; h4 = 4; }
      else if (part_shape_ == 1) { bx4 = 0; by4 = p * 2; w4 = 4; h4 = 2; }
      else { bx4 = p * 2; by4 = 0; w4 = 2; h4 = 4; }
      const uint8_t pr = part_pred_[p];
      int mvx = 0, mvy = 0;
      if (pr & (1 << l)) {
        mv_pred(l, bx4, by4, w4, h4, ref[l][p], part_shape_, p, &mvx, &mvy);
        int dx, dy;
        if (cabac_) {
          dx = cabac_mvd(l, 0, by4 * 4 + bx4);
          dy = cabac_mvd(l, 1, by4 * 4 + bx4);
        } else {
          dx = br_.se();
          dy = br_.se();
        }
        set_mvd(l, bx4, by4, w4, h4, dx, dy);
        mvx += dx;
        mvy += dy;
      }
      set_motion(bx4, by4, w4, h4, l, (pr & (1 << l)) ? ref[l][p] : -1, mvx, mvy);
      for (int yy = 0; yy < h4; ++yy)
        for (int xx = 0; xx < w4; ++xx) blk_done_ |= static_cast<uint16_t>(1 << ((by4 + yy) * 4 + bx4 + xx));
    }
  }
  set_ids();
  return !br_.overrun();
}

// ---- macroblock layer -----------------------------------------------------------------------------------------------
bool Dec::parse_macroblock() {
  is_skip_ = false;
  int t = cabac_ ? cabac_mb_type() : static_cast<int>(br_.ue());
  const bool is_b = sh_.is_b(), is_p = sh_.is_p();
  int intra_type = -1;   // I slice numbering if intra
  if (sh_.is_intra()) intra_type = t;
  else if (is_p) {
    if (t >= 5) intra_type = t - 5;
  } else if (is_b) {
    if (t >= 23) intra_type = t - 23;
  }
  if (intra_type > 25 || t < 0) return fail("bad mb_type");
  cbp_ = 0;
  if (intra_type >= 0) {
    mb_->intra = true;
    if (intra_type == 0) {
      mb_->kind = kI4x4;
      if (pp_.transform_8x8_mode) {
        const int f = cabac_ ? cab_.decision(399 + (mbaddr_a_ >= 0 && mbi(mbaddr_a_).t8x8 ? 1 : 0) + (mbaddr_b_ >= 0 && mbi(mbaddr_b_).t8x8 ? 1 : 0)) : static_cast<int>(br_.u(1));
        if (f) {
          mb_->t8x8 = true;
          mb_->kind = kI8x8;
        }
      }
    } else if (intra_type == 25) {
      mb_->kind = kIPCM;
      ipcm_samples();
      return !br_.overrun();
    } else {
      mb_->kind = kI16x16;
      const int v = intra_type - 1;
      i16_mode_ = v % 4;
      mb_->i16_mode = static_cast<uint8_t>(i16_mode_);
      const int chroma = (v / 4) % 3;
      const int luma = v >= 12 ? 15 : 0;
      cbp_ = luma | chroma << 4;
    }
    if (mb_->kind == kI4x4 || mb_->kind == kI8x8) {
      if (!parse_intra_pred_modes()) return false;
    }
    mb_->chroma_mode = static_cast<uint8_t>(cabac_ ? cabac_intra_chroma_mode() : static_cast<int>(br_.ue()));
    if (mb_->chroma_mode > 3) return fail("bad intra_chroma_pred_mode");
  } else {
    mb_->intra = false;
    mb_->kind = kInter;
    mb_type_ = t;
    bool no_sub_less_8x8 = true;
    if (is_p) {
      static const uint8_t shape[5] = {0, 1, 2, 3, 3};
      if (t > 4) return fail("bad P mb_type");
      part_shape_ = shape[t];
      part_pred_[0] = part_pred_[1] = 1;
      if (t == 4) mb_type_ = -1;   // P_8x8ref0
    } else {
      if (t == 0) {
        // B_Direct_16x16
        part_shape_ = 4;
      } else {
        if (t > 22) return fail("bad B mb_type");
        part_shape_ = kBMb[t].shape;
        part_pred_[0] = kBMb[t].p0;
        part_pred_[1] = kBMb[t].p1;
      }
    }
    if (is_b && t == 0) {
      mb_->direct16 = true;
      mb_->direct8 = 0xF;
      derive_direct(-1);
      finalize_motion();
    } else {
      if (part_shape_ == 3) {
        for (int p = 0; p < 4; ++p) {
          const int st = cabac_ ? cabac_sub_mb_type() : static_cast<int>(br_.ue());
          if (is_b) {
            if (st > 12) return fail("bad B sub_mb_type");
            sub_direct_[p] = st == 0;
            sub_shape_[p] = kBSubShape[st];
            sub_pred_[p] = kBSubPred[st];
            if (st == 0) {
              if (!pp_.direct_8x8_inference) no_sub_less_8x8 = false;
            } else if (kBSubShape[st] != 0) {
              no_sub_less_8x8 = false;
            }
          } else {
            if (st > 3) return fail("bad P sub_mb_type");
            sub_direct_[p] = false;
            sub_shape_[p] = static_cast<uint8_t>(st);
            sub_pred_[p] = 1;
            if (st != 0) no_sub_less_8x8 = false;
          }
        }
      }
      if (!parse_inter_pred()) return false;
      for (int b8 = 0; b8 < 4; ++b8) mb_->pred_flags[b8] = pred_flags_of(b8);
    }
    // coded_block_pattern
    int v;
    if (cabac_) v = cabac_cbp();
    else {
      const uint32_t code = br_.ue();
      if (code > 47) return fail("bad coded_block_pattern");
      v = kCbpInter[code];
    }
    cbp_ = v;
    if ((cbp_ & 15) && pp_.transform_8x8_mode && no_sub_less_8x8 && (!(is_b && t == 0) || pp_.direct_8x8_inference)) {
      const int f = cabac_ ? cab_.decision(399 + (mbaddr_a_ >= 0 && mbi(mbaddr_a_).t8x8 ? 1 : 0) + (mbaddr_b_ >= 0 && mbi(mbaddr_b_).t8x8 ? 1 : 0)) : static_cast<int>(br_.u(1));
      mb_->t8x8 = f != 0;
    }
    mb_->cbp = static_cast<uint8_t>(cbp_);
    if (cbp_ == 0) {
      last_qp_delta_ = 0;
      mb_->qp = static_cast<int8_t>(qp_);
      return !br_.overrun();
    }
  }
  if (mb_->intra && mb_->kind != kI16x16) {
    int v;
    if (cabac_) v = cabac_cbp();
    else {
      const uint32_t code = br_.ue();
      if (code > 47) return fail("bad coded_block_pattern");
      v = kCbpIntra[code];
    }
    cbp_ = v;
  }
  mb_->cbp = static_cast<uint8_t>(cbp_);
  if (cbp_ == 0 && mb_->kind != kI16x16) {
    last_qp_delta_ = 0;
    mb_->qp = static_cast<int8_t>(qp_);
    return !br_.overrun();
  }
  // mb_qp_delta
  const int dqp = cabac_ ? cabac_qp_delta() : br_.se();
  last_qp_delta_ = dqp;
  mb_->qp_delta_nonzero = dqp != 0;
  if (dqp < -26 || dqp > 25) return fail("mb_qp_delta out of range");
  qp_ = ((qp_ + dqp + 52) % 52);
  mb_->qp = static_cast<int8_t>(qp_);
  return parse_residual() && !br_.overrun();
}
