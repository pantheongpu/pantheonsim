// Vector ALU instructions the CDNA2 guide defines (and CDNA3 and CDNA4 list with the same meaning) that the executor refused on gfx90a:
// V_SIN_F16 and V_COS_F16 ("D.f16 = sin(S0.f16 * 2 * PI)", with the guide's examples), the cube-map instructions (V_CUBEID_F32, _SC, _TC, _MA:
// its pseudocode), V_CVT_RPI_I32_F32 ("D.i = (int)floor(S0.f + 0.5)"), V_CVT_OFF_F32_I4 (its table), V_FREXP_MANT_F16 and V_FREXP_EXP_I16_F16,
// V_LDEXP_F16 ("D.f16 = S0.f16 * (2 ** S1.i16)"), V_FMA_LEGACY_F16 (fused), V_MAD_LEGACY_I16 (the high half of the destination zero), V_PK_FMAC_F16 and
// V_SAT_PK_U8_I16 ("D.u32 = {16'b0, sat8(S.u[31:16]), sat8(S.u[15:0])}").
//
// valu_gaps(in, out): in is 8 dwords a lane (a half to take the sine of, then x, y and z, three halves and two packed, one exponent and
// the pair to saturate), out is 16 dwords a lane, one result each in the order of the instructions above (out[15] is not written).
//
// Assembled for gfx90a, gfx942 and gfx950 by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX950 is
// defined, and RET is how the target spells "return the old value": glc on gfx90a, sc0 after it).
#if defined(GFX90A)
#define RET glc
#else
#define RET sc0
#endif
  .text
  .globl valu_gaps
  .p2align 8
  .type valu_gaps,@function
valu_gaps:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = in, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 5, v0                  // 8 dwords of input a lane
  v_lshlrev_b32_e32 v2, 6, v0                  // 16 dwords of output a lane
  global_load_dwordx4 v[4:7], v1, s[4:5] offset:0
  global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v16, 0
  v_mov_b32_e32 v17, 0
  v_mov_b32_e32 v18, 0
  v_mov_b32_e32 v19, 0
  v_mov_b32_e32 v20, 0
  v_mov_b32_e32 v21, 0
  v_mov_b32_e32 v22, 0
  v_mov_b32_e32 v23, 0
  v_mov_b32_e32 v24, 0
  v_mov_b32_e32 v25, 0
  v_mov_b32_e32 v26, 0
  v_mov_b32_e32 v27, 0
  v_mov_b32_e32 v28, 0
  v_mov_b32_e32 v29, 0
  v_mov_b32_e32 v30, 0
  v_mov_b32_e32 v31, 0
  v_sin_f16_e32 v16, v4
  v_cos_f16_e32 v17, v4
  v_cubeid_f32 v18, v5, v6, v7
  v_cubesc_f32 v19, v5, v6, v7
  v_cubetc_f32 v20, v5, v6, v7
  v_cubema_f32 v21, v5, v6, v7
  v_cvt_rpi_i32_f32_e32 v22, v5
  v_cvt_off_f32_i4_e32 v23, v4
  v_frexp_mant_f16_e32 v24, v4
  v_frexp_exp_i16_f16_e32 v25, v4
  v_ldexp_f16_e32 v26, v8, v11
  v_fma_legacy_f16 v27, v8, v9, v10
  v_mad_legacy_i16 v28, v8, v9, v10
  v_mov_b32_e32 v29, v10
  v_pk_fmac_f16_e32 v29, v8, v9
  v_sat_pk_u8_i16_e32 v30, v11
  global_store_dword v2, v16, s[6:7] offset:0
  global_store_dword v2, v17, s[6:7] offset:4
  global_store_dword v2, v18, s[6:7] offset:8
  global_store_dword v2, v19, s[6:7] offset:12
  global_store_dword v2, v20, s[6:7] offset:16
  global_store_dword v2, v21, s[6:7] offset:20
  global_store_dword v2, v22, s[6:7] offset:24
  global_store_dword v2, v23, s[6:7] offset:28
  global_store_dword v2, v24, s[6:7] offset:32
  global_store_dword v2, v25, s[6:7] offset:36
  global_store_dword v2, v26, s[6:7] offset:40
  global_store_dword v2, v27, s[6:7] offset:44
  global_store_dword v2, v28, s[6:7] offset:48
  global_store_dword v2, v29, s[6:7] offset:52
  global_store_dword v2, v30, s[6:7] offset:56
  s_waitcnt vmcnt(0)
  s_endpgm
.Lvalu_gaps_end:
  .size valu_gaps, .Lvalu_gaps_end-valu_gaps
  .rodata
  .p2align 6
  .amdhsa_kernel valu_gaps
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 40
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 40
  .end_amdhsa_kernel
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: valu_gaps
    .symbol: valu_gaps.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata

