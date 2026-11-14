// The instruction modifiers the clamp rounds did not cover (test_amd_gcn_mods): SDWA's clamp, output modifier, source selects with and
// without sign, destination selects and dst_unused (gfx9 and gfx10); DPP's source negate and absolute value, bank mask and bound control
// (every target), and gfx11's VOP3 DPP forms with clamp and output modifier; VOP3P's op_sel, op_sel_hi, neg_lo, neg_hi and clamp on the
// packed float and integer instructions and the mixed-precision ones; and v_div_scale's clamp and output modifier.
//
// Every kernel has the same shape. Work-item t (global id, 1024 of them, 64 a group) reads eight dwords at in + 32 t: a, b, c, d, old.
// Each instruction below writes three dwords at out + 12 (1024 op + t): v10, v11 (zero unless the instruction writes a pair) and VCC's
// bit for the lane (zero unless the instruction writes VCC). Before each instruction v10 is set to `old`, v11 and VCC to zero, so what
// an instruction leaves of its destination shows.
//
// The kernels: sdwa_k, sdwa_n, sdwa_f (gfx9 and gfx10 only; SDWA and the DPP forms every target has), pk_k, pk_f (every target; VOP3P),
// dpp_k, dpp_n, dpp_f (gfx11 only; VOP3 DPP). _k is the compiler's mode (IEEE on, denormals kept); _n has IEEE off; _f has IEEE off and
// denormals flushed. With IEEE on or output denormals kept the output modifier does nothing (CDNA3 guide, 6.5; LLVM's SIFoldOperands).
//
// Assembled for gfx90a, gfx942, gfx1030 (wave32) and gfx1100 (wave32) by build.sh (clang's assembler with the C preprocessor: GFX90A,
// GFX942, GFX1030 or GFX1100 is defined). The gfx1030 object was run on an RX 6800.
#if defined(GFX1030) || defined(GFX1100)
#define WAVE32 1
#define VCC vcc_lo
#define ZERO_VCC s_mov_b32 vcc_lo, 0
#define VADD v_add_nc_u32_e32
#define ADD_U32_SDWA v_add_nc_u32_sdwa
#define SUB_U32_SDWA v_sub_nc_u32_sdwa
#define SUBREV_U32_SDWA v_subrev_nc_u32_sdwa
#define ADD_U32_DPP v_add_nc_u32_dpp
#else
#define VCC vcc
#define ZERO_VCC s_mov_b64 vcc, 0
#define VADD v_add_u32_e32
#define ADD_U32_SDWA v_add_u32_sdwa
#define SUB_U32_SDWA v_sub_u32_sdwa
#define SUBREV_U32_SDWA v_subrev_u32_sdwa
#define ADD_U32_DPP v_add_u32_dpp
#endif

.macro next
  v_cndmask_b32_e64 v12, 0, 1, VCC
  global_store_dwordx3 v8, v[10:12], s[8:9]
  s_add_u32 s8, s8, 0x3000
  s_addc_u32 s9, s9, 0
  v_mov_b32_e32 v10, v6
  v_mov_b32_e32 v11, 0
  ZERO_VCC
.endm

.macro prologue
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[8:9], s[0:1], 0x8          // s[8:9] = out
  s_lshl_b32 s3, s2, 6                         // the group's first work-item (64 a group)
  VADD v0, s3, v0                              // t
  v_lshlrev_b32_e32 v1, 5, v0                  // 32 t
  v_mul_u32_u24_e32 v8, 12, v0                 // 12 t
  s_waitcnt lgkmcnt(0)
  global_load_dwordx4 v[2:5], v1, s[4:5]       // a, b, c, d
  global_load_dword v6, v1, s[4:5] offset:16   // old
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v10, v6
  v_mov_b32_e32 v11, 0
  ZERO_VCC
.endm

.macro epilogue
  s_waitcnt vmcnt(0)
  s_endpgm
.endm

#ifndef GFX1100
// SDWA, and DPP without a clamp (gfx9's and gfx10's DPP word has none):
//   0 add_f32 clamp   1 add_f32 mul:2   2 add_f32 clamp mul:4   3 mul_f32 div:2   4 sub_f32 clamp src0 WORD_1 src1 WORD_0   5 max_f32 clamp
//   6 add_f32 clamp dst WORD_1 preserve   7 add_f32 -a |b| clamp   8 add_f16 clamp WORD_1 WORD_1 dst WORD_1 preserve
//   9 mul_f16 mul:2 src0 WORD_1 dst WORD_0 pad   10 add_f16 clamp mul:4   11 max_f16 clamp src1 WORD_1   12 sub_f16 clamp div:2 dst WORD_0 sext
//   13 cvt_f16_f32 clamp dst WORD_1 preserve   14 cvt_f32_f16 clamp src0 WORD_1   15 cvt_f32_f16 mul:2   16 cvt_f32_u32 clamp src0 BYTE_1
//   17 cvt_f32_i32 sext(a) WORD_1 mul:2   18 cvt_u32_f32 clamp   19 cvt_i32_f32 clamp   20 add_u32 clamp src0 WORD_1
//   21 sub_u32 clamp BYTE_0 BYTE_1   22 subrev_u32 clamp   23 add_u32 sext(a) BYTE_0 clamp   24 add_co_u32 clamp (gfx10: add_co_ci, carry in 0)
//   25 mul_u32_u24 clamp   26 mul_i32_i24 sext(a) WORD_1 clamp   27 mov_b32 src0 WORD_1 dst BYTE_1 sext   28 mov_b32 sext(a) BYTE_3 dst WORD_1 pad
//   29 and_b32 BYTE_3 WORD_0 dst BYTE_2 preserve   30 xor_b32 sext(a) BYTE_0 dst BYTE_3 sext   31 ashrrev_i32 BYTE_0 sext(b) WORD_1
//   32 min_i32 sext(a) BYTE_1 sext(b) WORD_0   33 cmp_gt_i32 sext(a) BYTE_0 b WORD_1   34 cmp_lt_f16 src0 WORD_1
//   35 add_u16 clamp WORD_1 dst WORD_1 preserve (gfx9)   36 sub_u16 clamp (gfx9)   37 mul_lo_u16 clamp (gfx9)   38 max_i16 sext(a) BYTE_0 clamp (gfx9)
//   39 div_scale_f32 a, b, a clamp   40 div_scale_f32 a, b, a mul:2   41 div_scale_f32 a, b, a   42 add_f32 clamp src0 BYTE_1 src1 WORD_1
//   43 mul_f32 clamp mul:2 dst BYTE_0 sext   44 add_f32_dpp -a |b| quad_perm:[1,0,3,2]   45 add_u32_dpp row_shr:1 bank_mask:0x5
//   46 mov_b32_dpp row_shr:1 bound_ctrl   47 max_f32   48 min_f32 (VOP3)   49 min_f16 (no modifiers: which zero and which NaN wins)
.macro sdwa_body
  v_add_f32_sdwa v10, v2, v3 clamp
  next
  v_add_f32_sdwa v10, v2, v3 mul:2
  next
  v_add_f32_sdwa v10, v2, v3 clamp mul:4
  next
  v_mul_f32_sdwa v10, v2, v3 div:2
  next
  v_sub_f32_sdwa v10, v2, v3 clamp src0_sel:WORD_1 src1_sel:WORD_0
  next
  v_max_f32_sdwa v10, v2, v3 clamp
  next
  v_add_f32_sdwa v10, v2, v3 clamp dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
  next
  v_add_f32_sdwa v10, -v2, |v3| clamp
  next
  v_add_f16_sdwa v10, v2, v3 clamp dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
  next
  v_mul_f16_sdwa v10, v2, v3 mul:2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1
  next
  v_add_f16_sdwa v10, v2, v3 clamp mul:4
  next
  v_max_f16_sdwa v10, v2, v3 clamp src1_sel:WORD_1
  next
  v_sub_f16_sdwa v10, v2, v3 clamp div:2 dst_sel:WORD_0 dst_unused:UNUSED_SEXT
  next
  v_cvt_f16_f32_sdwa v10, v2 clamp dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
  next
  v_cvt_f32_f16_sdwa v10, v2 clamp src0_sel:WORD_1
  next
  v_cvt_f32_f16_sdwa v10, v2 mul:2
  next
  v_cvt_f32_u32_sdwa v10, v2 clamp src0_sel:BYTE_1
  next
  v_cvt_f32_i32_sdwa v10, sext(v2) mul:2 src0_sel:WORD_1
  next
  v_cvt_u32_f32_sdwa v10, v2 clamp
  next
  v_cvt_i32_f32_sdwa v10, v2 clamp
  next
  ADD_U32_SDWA v10, v2, v3 clamp src0_sel:WORD_1
  next
  SUB_U32_SDWA v10, v2, v3 clamp src0_sel:BYTE_0 src1_sel:BYTE_1
  next
  SUBREV_U32_SDWA v10, v2, v3 clamp
  next
  ADD_U32_SDWA v10, sext(v2), v3 clamp src0_sel:BYTE_0
  next
#ifdef WAVE32
  v_add_co_ci_u32_sdwa v10, vcc_lo, v2, v3, vcc_lo clamp
#else
  v_add_co_u32_sdwa v10, vcc, v2, v3 clamp
#endif
  next
  v_mul_u32_u24_sdwa v10, v2, v3 clamp
  next
  v_mul_i32_i24_sdwa v10, sext(v2), v3 clamp src0_sel:WORD_1
  next
  v_mov_b32_sdwa v10, v2 dst_sel:BYTE_1 dst_unused:UNUSED_SEXT src0_sel:WORD_1
  next
  v_mov_b32_sdwa v10, sext(v2) dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_3
  next
  v_and_b32_sdwa v10, v2, v3 dst_sel:BYTE_2 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_3 src1_sel:WORD_0
  next
  v_xor_b32_sdwa v10, sext(v2), v3 dst_sel:BYTE_3 dst_unused:UNUSED_SEXT src0_sel:BYTE_0
  next
  v_ashrrev_i32_sdwa v10, v2, sext(v3) src0_sel:BYTE_0 src1_sel:WORD_1
  next
  v_min_i32_sdwa v10, sext(v2), sext(v3) src0_sel:BYTE_1 src1_sel:WORD_0
  next
  v_cmp_gt_i32_sdwa VCC, sext(v2), v3 src0_sel:BYTE_0 src1_sel:WORD_1
  next
  v_cmp_lt_f16_sdwa VCC, v2, v3 src0_sel:WORD_1
  next
#ifdef WAVE32
  s_nop 0
  next
  s_nop 0
  next
  s_nop 0
  next
  s_nop 0
  next
#else
  v_add_u16_sdwa v10, v2, v3 clamp dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
  next
  v_sub_u16_sdwa v10, v2, v3 clamp
  next
  v_mul_lo_u16_sdwa v10, v2, v3 clamp
  next
  v_max_i16_sdwa v10, sext(v2), v3 clamp src0_sel:BYTE_0
  next
#endif
  v_div_scale_f32 v10, VCC, v2, v3, v2 clamp
  next
  v_div_scale_f32 v10, VCC, v2, v3, v2 mul:2
  next
  v_div_scale_f32 v10, VCC, v2, v3, v2
  next
  v_add_f32_sdwa v10, v2, v3 clamp src0_sel:BYTE_1 src1_sel:WORD_1
  next
  v_mul_f32_sdwa v10, v2, v3 clamp mul:2 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT
  next
  v_add_f32_dpp v10, -v2, |v3| quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
  next
  ADD_U32_DPP v10, v2, v3 row_shr:1 row_mask:0xf bank_mask:0x5
  next
  v_mov_b32_dpp v10, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
  next
  v_max_f32_e32 v10, v2, v3
  next
  v_min_f32_e64 v10, v2, v3
  next
  v_min_f16_e32 v10, v2, v3
  next
.endm
#endif

// VOP3P:
//   0 pk_add_f16 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1] clamp   1 pk_add_f16 neg_lo:[0,1] neg_hi:[1,0]
//   2 pk_mul_f16 op_sel:[1,1] op_sel_hi:[0,0] clamp   3 pk_fma_f16 op_sel:[1,0,1] op_sel_hi:[0,1,0] neg_lo:[1,0,1] neg_hi:[0,1,1] clamp
//   4 pk_fma_f16 neg_lo:[0,0,1] neg_hi:[1,1,0]   5 pk_max_f16 neg_lo:[1,1] clamp   6 pk_min_f16 op_sel:[0,1] neg_hi:[1,0] clamp
//   7 pk_add_u16 op_sel:[1,0] op_sel_hi:[0,1] clamp   8 pk_add_u16 neg_lo:[1,0] neg_hi:[0,1]   9 pk_add_i16 neg_lo:[1,1] clamp
//   10 pk_sub_i16 op_sel:[0,1] op_sel_hi:[1,0] clamp   11 pk_sub_u16 op_sel:[1,1] clamp   12 pk_mul_lo_u16 op_sel:[1,1] op_sel_hi:[0,0]
//   13 pk_mad_u16 op_sel:[1,0,1] op_sel_hi:[0,0,0] clamp   14 pk_mad_i16 op_sel:[0,1,1] op_sel_hi:[1,0,1] clamp   15 pk_max_i16 op_sel:[1,1] clamp
//   16 pk_min_u16 op_sel_hi:[0,1]   17 pk_lshlrev_b16 op_sel:[1,0]   18 pk_ashrrev_i16 op_sel_hi:[0,0]   19 pk_lshrrev_b16 op_sel:[0,1] op_sel_hi:[1,0]
//   20 fma_mix_f32 op_sel:[1,0,1] op_sel_hi:[1,0,1] clamp   21 fma_mix_f32 -a |b| c op_sel_hi:[1,1,0]   22 fma_mixlo_f16 -a |b| c op_sel_hi:[1,1,0] clamp
//   23 fma_mixhi_f16 op_sel:[0,1,0] op_sel_hi:[1,1,1] clamp   24 dot2_f32_f16 neg_lo:[1,0,0] neg_hi:[0,1,0] clamp (gfx942/gfx1030/gfx1100)
//   25 pk_mul_f16 neg_lo:[1,1] neg_hi:[1,1] op_sel:[0,1] op_sel_hi:[1,0]   26 pk_max_f16 op_sel:[1,0] op_sel_hi:[0,1]   27 pk_add_f16 a, 1.0 clamp
//   28 pk_add_u16 a, -1 clamp   29 pk_fma_f32 (a,b) (c,d) (a,b) op_sel:[1,0,1] op_sel_hi:[0,1,0] neg_lo:[1,0,0] neg_hi:[0,0,1] clamp (gfx90a/gfx942)
//   30 pk_mul_f32 (a,b) (c,d) op_sel:[1,0] neg_hi:[1,1] clamp (gfx90a/gfx942)   31 pk_add_f32 (a,b) (c,d) op_sel_hi:[0,0] neg_lo:[0,1] (gfx90a/gfx942)
//   32 add_f32 mul:2   33 mul_f32 clamp div:2   34 fma_f32 clamp mul:2   35 add_f64 (a,b) (c,d) mul:4 (plain VOP3, which the tier runs)
.macro pk_body
  v_pk_add_f16 v10, v2, v3 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1] clamp
  next
  v_pk_add_f16 v10, v2, v3 neg_lo:[0,1] neg_hi:[1,0]
  next
  v_pk_mul_f16 v10, v2, v3 op_sel:[1,1] op_sel_hi:[0,0] clamp
  next
  v_pk_fma_f16 v10, v2, v3, v4 op_sel:[1,0,1] op_sel_hi:[0,1,0] neg_lo:[1,0,1] neg_hi:[0,1,1] clamp
  next
  v_pk_fma_f16 v10, v2, v3, v4 neg_lo:[0,0,1] neg_hi:[1,1,0]
  next
  v_pk_max_f16 v10, v2, v3 neg_lo:[1,1] clamp
  next
  v_pk_min_f16 v10, v2, v3 op_sel:[0,1] neg_hi:[1,0] clamp
  next
  v_pk_add_u16 v10, v2, v3 op_sel:[1,0] op_sel_hi:[0,1] clamp
  next
  v_pk_add_u16 v10, v2, v3 neg_lo:[1,0] neg_hi:[0,1]
  next
  v_pk_add_i16 v10, v2, v3 neg_lo:[1,1] clamp
  next
  v_pk_sub_i16 v10, v2, v3 op_sel:[0,1] op_sel_hi:[1,0] clamp
  next
  v_pk_sub_u16 v10, v2, v3 op_sel:[1,1] clamp
  next
  v_pk_mul_lo_u16 v10, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
  next
  v_pk_mad_u16 v10, v2, v3, v4 op_sel:[1,0,1] op_sel_hi:[0,0,0] clamp
  next
  v_pk_mad_i16 v10, v2, v3, v4 op_sel:[0,1,1] op_sel_hi:[1,0,1] clamp
  next
  v_pk_max_i16 v10, v2, v3 op_sel:[1,1] clamp
  next
  v_pk_min_u16 v10, v2, v3 op_sel_hi:[0,1]
  next
  v_pk_lshlrev_b16 v10, v2, v3 op_sel:[1,0]
  next
  v_pk_ashrrev_i16 v10, v2, v3 op_sel_hi:[0,0]
  next
  v_pk_lshrrev_b16 v10, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
  next
  v_fma_mix_f32 v10, v2, v3, v4 op_sel:[1,0,1] op_sel_hi:[1,0,1] clamp
  next
  v_fma_mix_f32 v10, -v2, |v3|, v4 op_sel_hi:[1,1,0]
  next
  v_fma_mixlo_f16 v10, -v2, |v3|, v4 op_sel_hi:[1,1,0] clamp
  next
  v_fma_mixhi_f16 v10, v2, v3, v4 op_sel:[0,1,0] op_sel_hi:[1,1,1] clamp
  next
#ifdef GFX90A
  s_nop 0
#else
  v_dot2_f32_f16 v10, v2, v3, v4 neg_lo:[1,0,0] neg_hi:[0,1,0] clamp
#endif
  next
  v_pk_mul_f16 v10, v2, v3 op_sel:[0,1] op_sel_hi:[1,0] neg_lo:[1,1] neg_hi:[1,1]
  next
  v_pk_max_f16 v10, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
  next
  v_pk_add_f16 v10, v2, 1.0 clamp
  next
  v_pk_add_u16 v10, v2, -1 clamp
  next
#ifdef WAVE32
  s_nop 0
  next
  s_nop 0
  next
  s_nop 0
  next
#else
  v_pk_fma_f32 v[10:11], v[2:3], v[4:5], v[2:3] op_sel:[1,0,1] op_sel_hi:[0,1,0] neg_lo:[1,0,0] neg_hi:[0,0,1] clamp
  next
  v_pk_mul_f32 v[10:11], v[2:3], v[4:5] op_sel:[1,0] neg_hi:[1,1] clamp
  next
  v_pk_add_f32 v[10:11], v[2:3], v[4:5] op_sel_hi:[0,0] neg_lo:[0,1]
  next
#endif
  v_add_f32_e64 v10, v2, v3 mul:2
  next
  v_mul_f32_e64 v10, v2, v3 clamp div:2
  next
  v_fma_f32 v10, v2, v3, v4 clamp mul:2
  next
  v_add_f64 v[10:11], v[2:3], v[4:5] mul:4
  next
.endm

#ifdef GFX1100
// gfx11's VOP3 DPP (the first source another lane's, quad_perm:[1,0,3,2] unless said):
//   0 add_f32 clamp   1 add_f32 mul:2 row_shl:1 bound_ctrl   2 mul_f32 -a |b| clamp mul:4   3 add_nc_u32 clamp   4 sub_nc_u32 clamp row_ror:1
//   5 add_nc_u16 clamp   6 add_nc_u16 op_sel:[1,1,1] clamp   7 fma_f32 clamp   8 max_f32 clamp dpp8:[1,0,3,2,5,4,7,6]   9 add_co_u32 clamp (VCC)
//   10 add_f16 clamp mul:2   11 mad_u32_u24 clamp   12 med3_f32 clamp mul:2   13 fma_f16 op_sel:[1,0,1,1] clamp   14 add_f32 clamp div:2 dpp8
//   15 cvt_u32_f32 clamp   16 add_nc_i32 clamp   17 mul_u32_u24 clamp   18 sad_u32 clamp   19 dot2_f32_f16 neg_lo:[1,0,0] neg_hi:[0,1,0] clamp
//   20 fma_mix_f32 op_sel_hi:[1,1,0] clamp   21 add_f32_dpp -a |b| row_shr:1 bank_mask:0x5   22 mov_b32_dpp row_shr:1 bound_ctrl
//   23 add_nc_u32_dpp dpp8:[7,6,5,4,3,2,1,0]
#define QP quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
.macro dpp_body
  v_add_f32_e64_dpp v10, v2, v3 clamp QP
  next
  v_add_f32_e64_dpp v10, v2, v3 mul:2 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
  next
  v_mul_f32_e64_dpp v10, -v2, |v3| clamp mul:4 QP
  next
  v_add_nc_u32_e64_dpp v10, v2, v3 clamp QP
  next
  v_sub_nc_u32_e64_dpp v10, v2, v3 clamp row_ror:1 row_mask:0xf bank_mask:0xf
  next
  v_add_nc_u16_e64_dpp v10, v2, v3 clamp QP
  next
  v_add_nc_u16_e64_dpp v10, v2, v3 op_sel:[1,1,1] clamp QP
  next
  v_fma_f32_e64_dpp v10, v2, v3, v4 clamp QP
  next
  v_max_f32_e64_dpp v10, v2, v3 clamp dpp8:[1,0,3,2,5,4,7,6]
  next
  v_add_co_u32_e64_dpp v10, vcc_lo, v2, v3 clamp QP
  next
  v_add_f16_e64_dpp v10, v2, v3 clamp mul:2 QP
  next
  v_mad_u32_u24_e64_dpp v10, v2, v3, v4 clamp QP
  next
  v_med3_f32_e64_dpp v10, v2, v3, v4 clamp mul:2 QP
  next
  v_fma_f16_e64_dpp v10, v2, v3, v4 op_sel:[1,0,1,1] clamp QP
  next
  v_add_f32_e64_dpp v10, v2, v3 clamp div:2 dpp8:[1,0,3,2,5,4,7,6]
  next
  v_cvt_u32_f32_e64_dpp v10, v2 clamp QP
  next
  v_add_nc_i32_e64_dpp v10, v2, v3 clamp QP
  next
  v_mul_u32_u24_e64_dpp v10, v2, v3 clamp QP
  next
  v_sad_u32_e64_dpp v10, v2, v3, v4 clamp QP
  next
  v_dot2_f32_f16_e64_dpp v10, v2, v3, v4 neg_lo:[1,0,0] neg_hi:[0,1,0] clamp QP
  next
  v_fma_mix_f32_e64_dpp v10, v2, v3, v4 op_sel_hi:[1,1,0] clamp QP
  next
  v_add_f32_dpp v10, -v2, |v3| row_shr:1 row_mask:0xf bank_mask:0x5
  next
  v_mov_b32_dpp v10, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
  next
  v_add_nc_u32_dpp v10, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
  next
.endm
#endif

#ifdef WAVE32
#define WAVE_DESC .amdhsa_wavefront_size32 1
#define WAVE_META 32
#else
#define WAVE_DESC .amdhsa_accum_offset 16
#define WAVE_META 64
#endif

// One kernel: its name, its body, IEEE mode, and the denormal mode of single precision and of half and double.
.macro kernel name, body, ieee, d32, d16
  .text
  .globl \name
  .p2align 8
  .type \name,@function
\name:
  prologue
  \body
  epilogue
  .p2align 6
  .amdhsa_kernel \name
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 32
    .amdhsa_ieee_mode \ieee
    .amdhsa_dx10_clamp 1
    .amdhsa_float_denorm_mode_32 \d32
    .amdhsa_float_denorm_mode_16_64 \d16
    WAVE_DESC
  .end_amdhsa_kernel
.endm

#ifndef GFX1100
  kernel sdwa_k, sdwa_body, 1, 3, 3
  kernel sdwa_n, sdwa_body, 0, 3, 3
  kernel sdwa_f, sdwa_body, 0, 0, 0
#else
  kernel dpp_k, dpp_body, 1, 3, 3
  kernel dpp_n, dpp_body, 0, 3, 3
  kernel dpp_f, dpp_body, 0, 0, 0
#endif
  kernel pk_k, pk_body, 1, 3, 3
  kernel pk_f, pk_body, 0, 0, 0

#define KMETA(n) \
  - { .name: n, .symbol: n.kd, .kernarg_segment_size: 16, .kernarg_segment_align: 8, .group_segment_fixed_size: 0, \
      .private_segment_fixed_size: 0, .wavefront_size: WAVE_META, .sgpr_count: 32, .vgpr_count: 16, .max_flat_workgroup_size: 64, \
      .args: [ { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }, \
               { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global } ] }

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
#ifndef GFX1100
  KMETA(sdwa_k)
  KMETA(sdwa_n)
  KMETA(sdwa_f)
#else
  KMETA(dpp_k)
  KMETA(dpp_n)
  KMETA(dpp_f)
#endif
  KMETA(pk_k)
  KMETA(pk_f)
...
  .end_amdgpu_metadata
