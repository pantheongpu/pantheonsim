// Part of nvenc_h264_enc.cpp (included inside its anonymous namespace): the CABAC syntax elements of H.264 (clause 9.3), written over a coder with
// decision(ctxIdx, bin), bypass(bin) and terminate(bin) (nvenc_cabac.hpp). The context numbers are those of Tables 9-34 to 9-39; they are the ones
// the decoder of this tree (h264_dec_parse.inc) reads.

constexpr int kCabacCtx = 1024;
using H264CostCoder = CabacCost<kCabacCtx>;
using H264BitCoder = CabacWriter<kCabacCtx>;

// Intra mb_type (Table 9-36) with the context offsets of I slices (base 3, ctx0 = base + inc) or of the intra prefix in P (17) / B (32) slices.
// `t` is the I-slice numbering: 0 I_NxN, 1..24 Intra16x16, 25 I_PCM.
template <class C>
void cabac_mb_type_i(C& c, int base, bool intra_slice, int ctx0_inc, int t) {
  c.decision(intra_slice ? base + ctx0_inc : base, t != 0);
  if (t == 0) return;
  c.terminate(t == 25);
  if (t == 25) return;
  const int v = t - 1;
  const int mode = v % 4, chroma = (v / 4) % 3, luma = v >= 12;
  c.decision(base + (intra_slice ? 3 : 1), luma);
  c.decision(base + (intra_slice ? 4 : 2), chroma != 0);
  if (chroma != 0) c.decision(base + (intra_slice ? 5 : 2), chroma == 2);
  c.decision(base + (intra_slice ? 6 : 3), mode >> 1);
  c.decision(base + (intra_slice ? 7 : 3), mode & 1);
}

// P slices: mb_type 0 P_L0_16x16, 1 P_L0_L0_16x8, 2 P_L0_L0_8x16, 3 P_8x8; `intra_t` >= 0 is an intra macroblock of that I-slice type
template <class C>
void cabac_mb_type_p(C& c, int t, int intra_t) {
  if (intra_t >= 0) {
    c.decision(14, 1);
    cabac_mb_type_i(c, 17, false, 0, intra_t);
    return;
  }
  c.decision(14, 0);
  if (t == 0 || t == 3) {
    c.decision(15, 0);
    c.decision(16, t == 3);
  } else {
    c.decision(15, 1);
    c.decision(17, t == 1);   // 2 - decision(17): 16x8 is 1
  }
}

// B slices: mb_type 0 B_Direct_16x16, 1..21 the 16x16, 16x8 and 8x16 types, 22 B_8x8; `intra_t` >= 0: an intra macroblock
template <class C>
void cabac_mb_type_b(C& c, int inc, int t, int intra_t) {
  if (intra_t >= 0) {
    c.decision(27 + inc, 1);
    c.decision(27 + 3, 1);
    c.decision(27 + 4, 1);
    c.decision(27 + 5, 1);
    c.decision(27 + 5, 0);
    c.decision(27 + 5, 1);
    cabac_mb_type_i(c, 32, false, 0, intra_t);
    return;
  }
  if (t == 0) {
    c.decision(27 + inc, 0);
    return;
  }
  c.decision(27 + inc, 1);
  if (t <= 2) {
    c.decision(27 + 3, 0);
    c.decision(27 + 5, t - 1);
    return;
  }
  c.decision(27 + 3, 1);
  if (t <= 10) {   // bits = t - 3 in 0..7
    const int bits = t - 3;
    c.decision(27 + 4, 0);
    c.decision(27 + 5, (bits >> 2) & 1);
    c.decision(27 + 5, (bits >> 1) & 1);
    c.decision(27 + 5, bits & 1);
    return;
  }
  if (t == 11 || t == 22) {   // bits 14 and 15
    c.decision(27 + 4, 1);
    c.decision(27 + 5, 1);
    c.decision(27 + 5, 1);
    c.decision(27 + 5, t == 22);
    return;
  }
  // 12..21: five bits, the first four being 8..12
  const int bits5 = t + 4;   // 16..25
  const int first4 = bits5 >> 1;
  c.decision(27 + 4, (first4 >> 3) & 1);
  c.decision(27 + 5, (first4 >> 2) & 1);
  c.decision(27 + 5, (first4 >> 1) & 1);
  c.decision(27 + 5, first4 & 1);
  c.decision(27 + 5, bits5 & 1);
}

// P sub_mb_type 0 (8x8) .. 3 (4x4)
template <class C>
void cabac_sub_mb_type_p(C& c, int st) {
  c.decision(21, st == 0);
  if (st == 0) return;
  c.decision(22, st != 1);
  if (st == 1) return;
  c.decision(23, st == 2);
}
// B sub_mb_type 0 (direct) .. 3 (Bi_8x8); sub-8x8 shapes are not used
template <class C>
void cabac_sub_mb_type_b(C& c, int st) {
  c.decision(36, st != 0);
  if (st == 0) return;
  c.decision(37, st >= 3);
  if (st < 3) {
    c.decision(39, st - 1);
    return;
  }
  // st == 3: bins 1,1, then (decision(38) == 0) -> type = 3 + 2 * d(39) + d(39) with d = 0, 0
  c.decision(38, 0);
  c.decision(39, 0);
  c.decision(39, 0);
}

template <class C>
void cabac_ref_idx(C& c, int inc, int ref) {
  int ctx = 54 + inc;
  for (int i = 0; i < ref; ++i) {
    c.decision(ctx, 1);
    ctx = 54 + (i == 0 ? 4 : 5);
  }
  c.decision(ctx, 0);
}

// mvd component: UEG3 with uCoff = 9; `base` 40 (x) or 47 (y); `sum` the neighbours' absolute mvd
template <class C>
void cabac_mvd(C& c, int base, int sum, int mvd) {
  const int inc = sum < 3 ? 0 : (sum > 32 ? 2 : 1);
  const int a = std::abs(mvd);
  if (a == 0) {
    c.decision(base + inc, 0);
    return;
  }
  c.decision(base + inc, 1);
  int ctx = base + 3;
  const int prefix_ones = std::min(a, 9);   // the unary prefix has min(a, 9) ones; a zero ends it when a < 9
  for (int i = 1; i < prefix_ones; ++i) {
    c.decision(ctx, 1);
    if (i < 4) ++ctx;
  }
  if (a < 9) {
    c.decision(ctx, 0);
  } else {
    int v = a - 9, k = 3;
    while (v >= (1 << k)) {
      c.bypass(1);
      v -= 1 << k;
      ++k;
    }
    c.bypass(0);
    while (k--) c.bypass((v >> k) & 1);
  }
  c.bypass(mvd < 0);
}

template <class C>
void cabac_qp_delta(C& c, bool prev_nonzero, int dqp) {
  const int v = dqp > 0 ? 2 * dqp - 1 : -2 * dqp;   // mapped value (Table 9-3)
  c.decision(60 + (prev_nonzero ? 1 : 0), v != 0);
  if (v == 0) return;
  int ctx = 60 + 2;
  for (int i = 1; i < v; ++i) {
    c.decision(ctx, 1);
    ctx = 60 + 3;
  }
  c.decision(ctx, 0);
}

template <class C>
void cabac_intra_chroma_mode(C& c, int inc, int mode) {
  c.decision(64 + inc, mode != 0);
  if (mode == 0) return;
  c.decision(64 + 3, mode != 1);
  if (mode == 1) return;
  c.decision(64 + 3, mode == 3);
}

// One residual block (7.3.5.3.3): cat 0 Intra16x16 DC, 1 Intra16x16 AC, 2 luma 4x4, 3 chroma DC, 4 chroma AC, 5 luma 8x8. `lev` holds max_coef levels in
// scan order. coded_block_flag is written for all but category 5 (a block with no levels writes it as 0 and nothing else).
template <class C>
void cabac_residual_block(C& c, int cat, int cbf_inc, int max_coef, const int* lev) {
  static const int kCbfOff[6] = {0, 4, 8, 12, 16, 0};
  static const int kSigOff[6] = {0, 15, 29, 44, 47, 0};
  static const int kAbsOff[6] = {0, 10, 20, 30, 39, 0};
  int last = -1;
  for (int i = 0; i < max_coef; ++i)
    if (lev[i]) last = i;
  if (cat != 5) {
    c.decision(85 + kCbfOff[cat] + cbf_inc, last >= 0);
    if (last < 0) return;
  }
  const int sig_base = cat == 5 ? 402 : 105 + kSigOff[cat];
  const int last_base = cat == 5 ? 417 : 166 + kSigOff[cat];
  const int abs_base = cat == 5 ? 426 : 227 + kAbsOff[cat];
  for (int i = 0; i < max_coef - 1; ++i) {
    int sinc, linc;
    if (cat == 3) sinc = linc = std::min(i, 2);
    else if (cat == 5) {
      sinc = kSig8x8Frame[i];
      linc = kLast8x8[i];
    } else {
      sinc = linc = i;
    }
    c.decision(sig_base + sinc, lev[i] != 0);
    if (lev[i]) {
      c.decision(last_base + linc, i == last);
      if (i == last) break;
    }
  }
  int num_gt1 = 0, num_eq1 = 0;
  for (int k = last; k >= 0; --k) {
    if (!lev[k]) continue;
    const int abs_m1 = std::abs(lev[k]) - 1;
    int ctx = abs_base + (num_gt1 != 0 ? 0 : std::min(4, 1 + num_eq1));
    c.decision(ctx, abs_m1 > 0);
    if (abs_m1 > 0) {
      ctx = abs_base + 5 + std::min(4 - (cat == 3 ? 1 : 0), num_gt1);
      const int ones = std::min(abs_m1, 14) - 1;
      for (int i = 0; i < ones; ++i) c.decision(ctx, 1);
      if (abs_m1 < 14) c.decision(ctx, 0);
      else {
        int v = abs_m1 - 14, kk = 0;
        while (v >= (1 << kk)) {
          c.bypass(1);
          v -= 1 << kk;
          ++kk;
        }
        c.bypass(0);
        while (kk--) c.bypass((v >> kk) & 1);
      }
    }
    if (abs_m1 == 0) ++num_eq1;
    else ++num_gt1;
    c.bypass(lev[k] < 0);
  }
}
