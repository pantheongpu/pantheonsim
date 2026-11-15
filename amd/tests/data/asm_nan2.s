// The gfx1030 NaN rule in the forms asm_nan.s left out (test_amd_gcn_nan2): SDWA, DPP and DPP8 forms, v_fmamk/v_fmaak, the legacy
// fma, packed halves, doubles (arithmetic, min/max, div_fixup, ldexp/frexp, rounding, conversions to and from them), and the dot
// products' clamp and denormals, min/max of opposite zeros and denormals, med3.
//
// Three kernels: n2_k (IEEE mode on, denormals kept, as compilers make kernels), n2_n (IEEE mode off), n2_f (IEEE mode on, denormals
// flushed). 192 work-items in groups of 64. Work-item t reads sixteen dwords at in + 64 t: half sources a, b, c (v20-v22, low halves),
// float sources a, b, c (v23-v25), double sources a, b, c (v[26:27], v[28:29], v[30:31]) and dot-product dwords a, b, c (v32-v34).
// Before each instruction v10 is set to 0x5a5a5a5a and v11 to 0xa5a5a5a5 (so a 16-bit result's high half shows); after it v[10:11] is
// stored at out + 1536 slot + 8 t.
//
// Assembled for gfx1030 (wave32) by build.sh (clang's assembler). The object was run on an RX 6800 and an RX 6700 XT (ROCm 7.2):
// asm_nan2.gfx1030.card.txt.
.macro next
  global_store_dwordx2 v8, v[10:11], s[8:9]
  s_add_u32 s8, s8, 0x600
  s_addc_u32 s9, s9, 0
  v_mov_b32_e32 v10, 0x5a5a5a5a
  v_mov_b32_e32 v11, 0xa5a5a5a5
.endm

.macro body
  v_add_f16_sdwa v10, v20, v21 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0   // 0
  next
  v_sub_f16_sdwa v10, v20, v21 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0   // 1
  next
  v_subrev_f16_sdwa v10, v20, v21 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0   // 2
  next
  v_mul_f16_sdwa v10, v20, v21 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0   // 3
  next
  v_max_f16_sdwa v10, v20, v21 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0   // 4
  next
  v_min_f16_sdwa v10, v20, v21 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0   // 5
  next
  v_add_f16_sdwa v10, -v20, |v21| dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0   // 6
  next
  v_sqrt_f16_sdwa v10, v20 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0   // 7
  next
  v_fract_f16_sdwa v10, v20 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0   // 8
  next
  v_rcp_f16_sdwa v10, -v20 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0   // 9
  next
  v_cvt_f32_f16_sdwa v10, v20 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_0   // 10
  next
  v_cvt_f16_f32_sdwa v10, v23 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD   // 11
  next
  v_add_f32_sdwa v10, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD   // 12
  next
  v_sub_f32_sdwa v10, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD   // 13
  next
  v_subrev_f32_sdwa v10, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD   // 14
  next
  v_mul_f32_sdwa v10, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD   // 15
  next
  v_mul_legacy_f32_sdwa v10, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD   // 16
  next
  v_max_f32_sdwa v10, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD   // 17
  next
  v_min_f32_sdwa v10, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD   // 18
  next
  v_add_f32_sdwa v10, -v23, |v24| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD   // 19
  next
  v_sqrt_f32_sdwa v10, v23 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD   // 20
  next
  v_fract_f32_sdwa v10, v23 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD   // 21
  next
  v_rsq_f32_sdwa v10, -v23 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD   // 22
  next
  v_add_f16_dpp v10, v20, v21 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 23
  next
  v_sub_f16_dpp v10, v20, v21 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 24
  next
  v_subrev_f16_dpp v10, v20, v21 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 25
  next
  v_mul_f16_dpp v10, v20, v21 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 26
  next
  v_max_f16_dpp v10, v20, v21 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 27
  next
  v_min_f16_dpp v10, v20, v21 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 28
  next
  v_add_f16_dpp v10, -v20, |v21| quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 29
  next
  v_sqrt_f16_dpp v10, v20 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 30
  next
  v_cvt_f32_f16_dpp v10, v20 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 31
  next
  v_add_f32_dpp v10, v23, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 32
  next
  v_sub_f32_dpp v10, v23, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 33
  next
  v_subrev_f32_dpp v10, v23, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 34
  next
  v_mul_f32_dpp v10, v23, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 35
  next
  v_max_f32_dpp v10, v23, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 36
  next
  v_min_f32_dpp v10, v23, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 37
  next
  v_add_f32_dpp v10, -v23, |v24| quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 38
  next
  v_mov_b32_e32 v10, v25
  v_fmac_f32_dpp v10, v23, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 39
  next
  v_sqrt_f32_dpp v10, v23 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 40
  next
  v_cvt_f16_f32_dpp v10, v23 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf   // 41
  next
  v_add_f16_dpp v10, v20, v21 dpp8:[0,1,2,3,4,5,6,7]   // 42
  next
  v_sub_f16_dpp v10, v20, v21 dpp8:[0,1,2,3,4,5,6,7]   // 43
  next
  v_max_f16_dpp v10, v20, v21 dpp8:[0,1,2,3,4,5,6,7]   // 44
  next
  v_add_f32_dpp v10, v23, v24 dpp8:[0,1,2,3,4,5,6,7]   // 45
  next
  v_subrev_f32_dpp v10, v23, v24 dpp8:[0,1,2,3,4,5,6,7]   // 46
  next
  v_min_f32_dpp v10, v23, v24 dpp8:[0,1,2,3,4,5,6,7]   // 47
  next
  v_mul_f32_dpp v10, v23, v24 dpp8:[0,1,2,3,4,5,6,7]   // 48
  next
  v_rcp_f32_dpp v10, v23 dpp8:[0,1,2,3,4,5,6,7]   // 49
  next
  v_fmamk_f32 v10, v23, 0x3f800000, v24   // 50
  next
  v_fmaak_f32 v10, v23, v24, 0x3f800000   // 51
  next
  v_fmamk_f32 v10, v23, 0x7fa00000, v24   // 52
  next
  v_fmaak_f32 v10, v23, v24, 0x7fa00000   // 53
  next
  v_fmamk_f32 v10, v23, 0, v24   // 54
  next
  v_fmamk_f16 v10, v20, 0x3c00, v21   // 55
  next
  v_fmaak_f16 v10, v20, v21, 0x3c00   // 56
  next
  v_fmamk_f16 v10, v20, 0x7d00, v21   // 57
  next
  v_fmaak_f16 v10, v20, v21, 0x7d00   // 58
  next
  v_fmaak_f16 v10, v20, v21, 0   // 59
  next
  v_fma_legacy_f32 v10, v23, v24, v25   // 60
  next
  v_mov_b32_e32 v10, v25
  v_fmac_legacy_f32_e32 v10, v23, v24   // 61
  next
  v_pk_add_f16 v10, v20, v21 op_sel_hi:[0,0]   // 62
  next
  v_pk_mul_f16 v10, v20, v21 op_sel_hi:[0,0]   // 63
  next
  v_pk_fma_f16 v10, v20, v21, v22 op_sel_hi:[0,0,0]   // 64
  next
  v_pk_min_f16 v10, v20, v21 op_sel_hi:[0,0]   // 65
  next
  v_pk_max_f16 v10, v20, v21 op_sel_hi:[0,0]   // 66
  next
  v_cvt_pkrtz_f16_f32 v10, v23, v24   // 67
  next
  v_add_f64 v[10:11], v[26:27], v[28:29]   // 68
  next
  v_add_f64 v[10:11], v[26:27], -v[28:29]   // 69
  next
  v_add_f64 v[10:11], -v[26:27], |v[28:29]|   // 70
  next
  v_mul_f64 v[10:11], v[26:27], v[28:29]   // 71
  next
  v_mul_f64 v[10:11], -v[26:27], -v[28:29]   // 72
  next
  v_fma_f64 v[10:11], v[26:27], v[28:29], v[30:31]   // 73
  next
  v_fma_f64 v[10:11], v[26:27], v[28:29], -v[30:31]   // 74
  next
  v_min_f64 v[10:11], v[26:27], v[28:29]   // 75
  next
  v_max_f64 v[10:11], v[26:27], v[28:29]   // 76
  next
  v_div_fixup_f64 v[10:11], v[26:27], v[28:29], v[30:31]   // 77
  next
  v_ldexp_f64 v[10:11], v[26:27], 1   // 78
  next
  v_frexp_mant_f64_e32 v[10:11], v[26:27]   // 79
  next
  v_frexp_exp_i32_f64_e32 v10, v[26:27]   // 80
  next
  v_rcp_f64_e32 v[10:11], v[26:27]   // 81
  next
  v_rsq_f64_e32 v[10:11], v[26:27]   // 82
  next
  v_sqrt_f64_e32 v[10:11], v[26:27]   // 83
  next
  v_fract_f64_e32 v[10:11], v[26:27]   // 84
  next
  v_floor_f64_e32 v[10:11], v[26:27]   // 85
  next
  v_ceil_f64_e32 v[10:11], v[26:27]   // 86
  next
  v_trunc_f64_e32 v[10:11], v[26:27]   // 87
  next
  v_rndne_f64_e32 v[10:11], v[26:27]   // 88
  next
  v_cvt_f32_f64_e32 v10, v[26:27]   // 89
  next
  v_cvt_f32_f64_e64 v10, -v[26:27]   // 90
  next
  v_cvt_f64_f32_e32 v[10:11], v23   // 91
  next
  v_cvt_f64_f32_e64 v[10:11], -|v23|   // 92
  next
  v_cvt_i32_f64_e32 v10, v[26:27]   // 93
  next
  v_cvt_u32_f64_e32 v10, v[26:27]   // 94
  next
  v_dot2_f32_f16 v10, v32, v33, v34   // 95
  next
  v_dot2_f32_f16 v10, v32, v33, v34 clamp   // 96
  next
  v_mov_b32_e32 v10, v34
  v_dot2c_f32_f16_e32 v10, v32, v33   // 97
  next
  v_dot2_i32_i16 v10, v32, v33, v34   // 98
  next
  v_dot2_i32_i16 v10, v32, v33, v34 clamp   // 99
  next
  v_dot2_u32_u16 v10, v32, v33, v34   // 100
  next
  v_dot2_u32_u16 v10, v32, v33, v34 clamp   // 101
  next
  v_dot4_i32_i8 v10, v32, v33, v34   // 102
  next
  v_dot4_i32_i8 v10, v32, v33, v34 clamp   // 103
  next
  v_dot4_u32_u8 v10, v32, v33, v34   // 104
  next
  v_dot4_u32_u8 v10, v32, v33, v34 clamp   // 105
  next
  v_dot8_i32_i4 v10, v32, v33, v34   // 106
  next
  v_dot8_i32_i4 v10, v32, v33, v34 clamp   // 107
  next
  v_dot8_u32_u4 v10, v32, v33, v34   // 108
  next
  v_dot8_u32_u4 v10, v32, v33, v34 clamp   // 109
  next
  v_mov_b32_e32 v10, v34
  v_dot4c_i32_i8_e32 v10, v32, v33   // 110
  next
  v_max_f32_e64 v10, v32, v33   // 111
  next
  v_min_f32_e64 v10, v32, v33   // 112
  next
  v_max_f16_e64 v10, v32, v33   // 113
  next
  v_min_f16_e64 v10, v32, v33   // 114
  next
  v_med3_f32 v10, v32, v33, v34   // 115
  next
  v_med3_f16 v10, v32, v33, v34   // 116
  next
.endm

.macro kernel name, ieee, denorm
  .text
  .globl \name
  .p2align 8
  .type \name,@function
\name:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[8:9], s[0:1], 0x8          // s[8:9] = out
  s_lshl_b32 s3, s2, 6                         // the group's first work-item (64 a group)
  v_add_nc_u32_e32 v0, s3, v0                  // t
  v_lshlrev_b32_e32 v1, 6, v0                  // 64 t
  v_lshlrev_b32_e32 v8, 3, v0                  // 8 t
  s_waitcnt lgkmcnt(0)
  global_load_dwordx4 v[20:23], v1, s[4:5]
  global_load_dwordx4 v[24:27], v1, s[4:5] offset:16
  global_load_dwordx4 v[28:31], v1, s[4:5] offset:32
  global_load_dwordx4 v[32:35], v1, s[4:5] offset:48
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v10, 0x5a5a5a5a
  v_mov_b32_e32 v11, 0xa5a5a5a5
  body
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel \name
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 40
    .amdhsa_next_free_sgpr 16
    .amdhsa_ieee_mode \ieee
    .amdhsa_dx10_clamp 1
    .amdhsa_float_denorm_mode_32 \denorm
    .amdhsa_float_denorm_mode_16_64 \denorm
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
.endm

  kernel n2_k, 1, 3
  kernel n2_n, 0, 3
  kernel n2_f, 1, 0

#define KMETA(n) \
  - { .name: n, .symbol: n.kd, .kernarg_segment_size: 16, .kernarg_segment_align: 8, .group_segment_fixed_size: 0, \
      .private_segment_fixed_size: 0, .wavefront_size: 32, .sgpr_count: 16, .vgpr_count: 40, .max_flat_workgroup_size: 64, \
      .args: [ { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }, \
               { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global } ] }

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  KMETA(n2_k)
  KMETA(n2_n)
  KMETA(n2_f)
...
  .end_amdgpu_metadata
