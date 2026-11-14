// What a 16-bit vector result leaves in the destination register's high half (test_amd_gcn_hi16): every VOP1, VOP2 and VOP3 instruction
// with a 16-bit result, a few VOP3 forms of VOP1/VOP2 ones, op_sel's destination bit, DPP, and two 32-bit results from 16-bit sources as
// a control. SDWA (dst_sel and dst_unused say it) and the packed instructions (both halves are results) are left out.
//
// One kernel, hi16: 64 work-items, one group. Work-item t reads a, b, c and old at in + 16 t; before each instruction v10 is set to old,
// and after it v10 is stored at out + 256 op + 4 t. old's high half is 0x5a00 | t, never zero, so a kept high half and a zeroed one differ.
//
// Assembled for gfx90a, gfx942, gfx1030 (wave32) and gfx1100 (wave32) by build.sh (clang's assembler with the C preprocessor: GFX90A,
// GFX942, GFX1030 or GFX1100 is defined). Each target has 75 slots; a slot a target has no instruction for stores old unchanged.
// The instruction of each slot is the gfx9 one (gfx90a, gfx942) or, after #elif, the gfx1030 one, then the gfx1100 one.
// The gfx1030 object was run on an RX 6800 and an RX 6700 XT (ROCm 7.2), which gave the same dwords: asm_hi16.gfx1030.card.txt.
#if defined(GFX1030) || defined(GFX1100)
#define VADD v_add_nc_u32_e32
#define WAVE_DESC .amdhsa_wavefront_size32 1
#define WAVE_META 32
#else
#define VADD v_add_u32_e32
#define WAVE_DESC .amdhsa_accum_offset 16
#define WAVE_META 64
#endif

.macro next
  global_store_dword v8, v10, s[8:9]
  s_add_u32 s8, s8, 0x100
  s_addc_u32 s9, s9, 0
  v_mov_b32_e32 v10, v6
.endm

  .text
  .globl hi16
  .p2align 8
  .type hi16,@function
hi16:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[8:9], s[0:1], 0x8          // s[8:9] = out
  v_lshlrev_b32_e32 v1, 4, v0                  // 16 t
  v_lshlrev_b32_e32 v8, 2, v0                  // 4 t
  s_waitcnt lgkmcnt(0)
  global_load_dwordx4 v[2:5], v1, s[4:5]       // a, b, c, old
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v6, v5
  v_mov_b32_e32 v10, v6

  v_cvt_f16_f32_e32 v10, v2   // 0
  next
  v_cvt_f16_u16_e32 v10, v2   // 1
  next
  v_cvt_f16_i16_e32 v10, v2   // 2
  next
  v_cvt_u16_f16_e32 v10, v2   // 3
  next
  v_cvt_i16_f16_e32 v10, v2   // 4
  next
  v_rcp_f16_e32 v10, v2   // 5
  next
  v_sqrt_f16_e32 v10, v2   // 6
  next
  v_rsq_f16_e32 v10, v2   // 7
  next
  v_log_f16_e32 v10, v2   // 8
  next
  v_exp_f16_e32 v10, v2   // 9
  next
  v_frexp_mant_f16_e32 v10, v2   // 10
  next
  v_frexp_exp_i16_f16_e32 v10, v2   // 11
  next
  v_floor_f16_e32 v10, v2   // 12
  next
  v_ceil_f16_e32 v10, v2   // 13
  next
  v_trunc_f16_e32 v10, v2   // 14
  next
  v_rndne_f16_e32 v10, v2   // 15
  next
  v_fract_f16_e32 v10, v2   // 16
  next
  v_sin_f16_e32 v10, v2   // 17
  next
  v_cos_f16_e32 v10, v2   // 18
  next
  v_sat_pk_u8_i16_e32 v10, v2   // 19
  next
#ifndef GFX90A
  v_cvt_norm_i16_f16_e32 v10, v2   // 20 (gfx90a: none, the executor does not run it there)
#endif
  next
#ifndef GFX90A
  v_cvt_norm_u16_f16_e32 v10, v2   // 21 (gfx90a: none, the executor does not run it there)
#endif
  next
  v_add_f16_e32 v10, v2, v3   // 22
  next
  v_sub_f16_e32 v10, v2, v3   // 23
  next
  v_subrev_f16_e32 v10, v2, v3   // 24
  next
  v_mul_f16_e32 v10, v2, v3   // 25
  next
  v_max_f16_e32 v10, v2, v3   // 26
  next
  v_min_f16_e32 v10, v2, v3   // 27
  next
  v_ldexp_f16_e32 v10, v2, v3   // 28
  next
#if defined(GFX90A) || defined(GFX942)
  v_mac_f16_e32 v10, v2, v3   // 29
#elif defined(GFX1030)
  v_fmac_f16_e32 v10, v2, v3
#else
  v_fmac_f16_e32 v10, v2, v3
#endif
  next
#if defined(GFX90A)
  // 30: none (the executor does not run v_madmk_f16 on gfx90a)
#elif defined(GFX942)
  v_madmk_f16 v10, v2, 0x3c00, v3   // 30
#elif defined(GFX1030)
  v_fmamk_f16 v10, v2, 0x3c00, v3
#else
  v_fmamk_f16 v10, v2, 0x3c00, v3
#endif
  next
#if defined(GFX90A)
  // 31: none (the executor does not run v_madak_f16 on gfx90a)
#elif defined(GFX942)
  v_madak_f16 v10, v2, v3, 0x3c00   // 31
#elif defined(GFX1030)
  v_fmaak_f16 v10, v2, v3, 0x3c00
#else
  v_fmaak_f16 v10, v2, v3, 0x3c00
#endif
  next
  v_add_f16_e64 v10, v2, v3   // 32
  next
  v_mul_f16_e64 v10, -v2, |v3|   // 33
  next
  v_cvt_f16_f32_e64 v10, v2   // 34
  next
  v_sqrt_f16_e64 v10, v2   // 35
  next
  v_max_f16_e64 v10, v2, v3   // 36
  next
#if defined(GFX90A) || defined(GFX942)
  v_add_u16_e32 v10, v2, v3   // 37
#elif defined(GFX1030)
  v_add_nc_u16 v10, v2, v3
#else
  v_add_nc_u16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_sub_u16_e32 v10, v2, v3   // 38
#elif defined(GFX1030)
  v_sub_nc_u16 v10, v2, v3
#else
  v_sub_nc_u16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_add_i16 v10, v2, v3   // 39
#elif defined(GFX1030)
  v_add_nc_i16 v10, v2, v3
#else
  v_add_nc_i16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_sub_i16 v10, v2, v3   // 40
#elif defined(GFX1030)
  v_sub_nc_i16 v10, v2, v3
#else
  v_sub_nc_i16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_mul_lo_u16_e32 v10, v2, v3   // 41
#elif defined(GFX1030)
  v_mul_lo_u16 v10, v2, v3
#else
  v_mul_lo_u16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_lshlrev_b16_e32 v10, v2, v3   // 42
#elif defined(GFX1030)
  v_lshlrev_b16 v10, v2, v3
#else
  v_lshlrev_b16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_lshrrev_b16_e32 v10, v2, v3   // 43
#elif defined(GFX1030)
  v_lshrrev_b16 v10, v2, v3
#else
  v_lshrrev_b16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_ashrrev_i16_e32 v10, v2, v3   // 44
#elif defined(GFX1030)
  v_ashrrev_i16 v10, v2, v3
#else
  v_ashrrev_i16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_max_u16_e32 v10, v2, v3   // 45
#elif defined(GFX1030)
  v_max_u16 v10, v2, v3
#else
  v_max_u16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_max_i16_e32 v10, v2, v3   // 46
#elif defined(GFX1030)
  v_max_i16 v10, v2, v3
#else
  v_max_i16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_min_u16_e32 v10, v2, v3   // 47
#elif defined(GFX1030)
  v_min_u16 v10, v2, v3
#else
  v_min_u16 v10, v2, v3
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_min_i16_e32 v10, v2, v3   // 48
#elif defined(GFX1030)
  v_min_i16 v10, v2, v3
#else
  v_min_i16 v10, v2, v3
#endif
  next
  v_mad_u16 v10, v2, v3, v4   // 49
  next
  v_mad_i16 v10, v2, v3, v4   // 50
  next
  v_fma_f16 v10, v2, v3, v4   // 51
  next
  v_div_fixup_f16 v10, v2, v3, v4   // 52
  next
  v_min3_f16 v10, v2, v3, v4   // 53
  next
  v_min3_i16 v10, v2, v3, v4   // 54
  next
  v_min3_u16 v10, v2, v3, v4   // 55
  next
  v_max3_f16 v10, v2, v3, v4   // 56
  next
  v_max3_i16 v10, v2, v3, v4   // 57
  next
  v_max3_u16 v10, v2, v3, v4   // 58
  next
  v_med3_f16 v10, v2, v3, v4   // 59
  next
  v_med3_i16 v10, v2, v3, v4   // 60
  next
  v_med3_u16 v10, v2, v3, v4   // 61
  next
#if defined(GFX90A) || defined(GFX942)
  v_fma_f16 v10, v2, v3, v4 op_sel:[1,0,0,1]   // 62
#elif defined(GFX1030)
  v_max3_f16 v10, v2, v3, v4 op_sel:[1,0,0,1]
#else
  v_max3_f16 v10, v2, v3, v4 op_sel:[1,0,0,1]
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_mad_u16 v10, v2, v3, v4 op_sel:[1,1,0,1]   // 63
#elif defined(GFX1030)
  v_add_nc_u16 v10, v2, v3 op_sel:[1,1,1]
#else
  v_add_nc_u16 v10, v2, v3 op_sel:[1,1,1]
#endif
  next
  v_fma_f16 v10, v2, v3, v4 op_sel:[0,0,0,1]   // 64
  next
  v_add_f16_dpp v10, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf   // 65
  next
  v_cvt_f16_f32_dpp v10, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf   // 66
  next
  v_cvt_f32_f16_e32 v10, v2   // 67
  next
  v_mad_u32_u16 v10, v2, v3, v4   // 68
  next
#if defined(GFX90A) || defined(GFX942)
  v_mad_f16 v10, v2, v3, v4   // 69
#elif defined(GFX1030)
  v_add_f16_dpp v10, v2, v3 dpp8:[1,0,3,2,5,4,7,6]
#else
  v_add_f16_dpp v10, v2, v3 dpp8:[1,0,3,2,5,4,7,6]
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_mad_legacy_f16 v10, v2, v3, v4   // 70
#elif defined(GFX1030)
  // 70: none
#else
  // 70: none
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_mad_legacy_u16 v10, v2, v3, v4   // 71
#elif defined(GFX1030)
  // 71: none
#else
  // 71: none
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_mad_legacy_i16 v10, v2, v3, v4   // 72
#elif defined(GFX1030)
  // 72: none
#else
  // 72: none
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_fma_legacy_f16 v10, v2, v3, v4   // 73
#elif defined(GFX1030)
  // 73: none
#else
  // 73: none
#endif
  next
#if defined(GFX90A) || defined(GFX942)
  v_div_fixup_legacy_f16 v10, v2, v3, v4   // 74
#elif defined(GFX1030)
  // 74: none
#else
  // 74: none
#endif
  next
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel hi16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 16
    .amdhsa_ieee_mode 1
    .amdhsa_dx10_clamp 1
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    WAVE_DESC
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - { .name: hi16, .symbol: hi16.kd, .kernarg_segment_size: 16, .kernarg_segment_align: 8, .group_segment_fixed_size: 0,
      .private_segment_fixed_size: 0, .wavefront_size: WAVE_META, .sgpr_count: 16, .vgpr_count: 16, .max_flat_workgroup_size: 64,
      .args: [ { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global },
               { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global } ] }
...
  .end_amdgpu_metadata
