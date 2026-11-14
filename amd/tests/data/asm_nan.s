// Which NaN a gfx1030 float instruction gives (test_amd_gcn_nan): half and single-precision arithmetic, conversions and transcendentals
// over every pair of eleven special values (NaNs of both signs, with and without a payload, signalling ones, infinities, zeros, +-1).
//
// Two kernels, nan_k (IEEE mode on, as compilers make kernels) and nan_n (IEEE mode off); 128 work-items in groups of 64. Work-item t
// reads eight dwords at in + 32 t: half sources a, b, c (v2-v4, in the low halves) and float sources a, b, c (v5-v7). Before each
// instruction v10 is zeroed; after it v10 is stored at out + 512 slot + 4 t.
//
// Assembled for gfx1030 (wave32) by build.sh (clang's assembler). The object was run on an RX 6800 and an RX 6700 XT (ROCm 7.2), which
// gave the same dwords: asm_nan.gfx1030.card.txt.
.macro next
  global_store_dword v8, v10, s[8:9]
  s_add_u32 s8, s8, 0x200
  s_addc_u32 s9, s9, 0
  v_mov_b32_e32 v10, 0
.endm

.macro body
  v_add_f16_e32 v10, v2, v3   // 0
  next
  v_sub_f16_e32 v10, v2, v3   // 1
  next
  v_subrev_f16_e32 v10, v2, v3   // 2
  next
  v_mul_f16_e32 v10, v2, v3   // 3
  next
  v_max_f16_e32 v10, v2, v3   // 4
  next
  v_min_f16_e32 v10, v2, v3   // 5
  next
  v_mov_b32_e32 v10, v4
  v_fmac_f16_e32 v10, v2, v3   // 6
  next
  v_fma_f16 v10, v2, v3, v4   // 7
  next
  v_div_fixup_f16 v10, v2, v3, v4   // 8
  next
  v_med3_f16 v10, v2, v3, v4   // 9
  next
  v_min3_f16 v10, v2, v3, v4   // 10
  next
  v_max3_f16 v10, v2, v3, v4   // 11
  next
  v_fract_f16_e32 v10, v2   // 12
  next
  v_floor_f16_e32 v10, v2   // 13
  next
  v_ceil_f16_e32 v10, v2   // 14
  next
  v_trunc_f16_e32 v10, v2   // 15
  next
  v_rndne_f16_e32 v10, v2   // 16
  next
  v_sqrt_f16_e32 v10, v2   // 17
  next
  v_rcp_f16_e32 v10, v2   // 18
  next
  v_rsq_f16_e32 v10, v2   // 19
  next
  v_log_f16_e32 v10, v2   // 20
  next
  v_exp_f16_e32 v10, v2   // 21
  next
  v_sin_f16_e32 v10, v2   // 22
  next
  v_cos_f16_e32 v10, v2   // 23
  next
  v_frexp_mant_f16_e32 v10, v2   // 24
  next
  v_cvt_f16_f32_e32 v10, v5   // 25
  next
  v_cvt_f32_f16_e32 v10, v2   // 26
  next
  v_add_f16_e64 v10, -v2, v3   // 27
  next
  v_add_f16_e64 v10, |v2|, v3   // 28
  next
  v_mul_f16_e64 v10, -v2, -v3   // 29
  next
  v_fma_f16 v10, v2, v3, -v4   // 30
  next
  v_ldexp_f16_e64 v10, v2, 1   // 31
  next
  v_add_f32_e32 v10, v5, v6   // 32
  next
  v_sub_f32_e32 v10, v5, v6   // 33
  next
  v_subrev_f32_e32 v10, v5, v6   // 34
  next
  v_mul_f32_e32 v10, v5, v6   // 35
  next
  v_mul_legacy_f32_e32 v10, v5, v6   // 36
  next
  v_max_f32_e32 v10, v5, v6   // 37
  next
  v_min_f32_e32 v10, v5, v6   // 38
  next
  v_mov_b32_e32 v10, v7
  v_fmac_f32_e32 v10, v5, v6   // 39
  next
  v_fma_f32 v10, v5, v6, v7   // 40
  next
  v_div_fixup_f32 v10, v5, v6, v7   // 41
  next
  v_med3_f32 v10, v5, v6, v7   // 42
  next
  v_min3_f32 v10, v5, v6, v7   // 43
  next
  v_max3_f32 v10, v5, v6, v7   // 44
  next
  v_fract_f32_e32 v10, v5   // 45
  next
  v_floor_f32_e32 v10, v5   // 46
  next
  v_ceil_f32_e32 v10, v5   // 47
  next
  v_trunc_f32_e32 v10, v5   // 48
  next
  v_rndne_f32_e32 v10, v5   // 49
  next
  v_sqrt_f32_e32 v10, v5   // 50
  next
  v_rcp_f32_e32 v10, v5   // 51
  next
  v_rsq_f32_e32 v10, v5   // 52
  next
  v_log_f32_e32 v10, v5   // 53
  next
  v_exp_f32_e32 v10, v5   // 54
  next
  v_sin_f32_e32 v10, v5   // 55
  next
  v_cos_f32_e32 v10, v5   // 56
  next
  v_frexp_mant_f32_e32 v10, v5   // 57
  next
  v_add_f32_e64 v10, -v5, v6   // 58
  next
  v_add_f32_e64 v10, |v5|, v6   // 59
  next
  v_mul_f32_e64 v10, -v5, -v6   // 60
  next
  v_fma_f32 v10, v5, v6, -v7   // 61
  next
  v_ldexp_f32 v10, v5, 1   // 62
  next
  v_sub_f16_e64 v10, v2, -v3   // 63
  next
  v_sub_f32_e64 v10, v5, -v6   // 64
  next
.endm

.macro kernel name, ieee
  .text
  .globl \name
  .p2align 8
  .type \name,@function
\name:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[8:9], s[0:1], 0x8          // s[8:9] = out
  s_lshl_b32 s3, s2, 6                         // the group's first work-item (64 a group)
  v_add_nc_u32_e32 v0, s3, v0                  // t
  v_lshlrev_b32_e32 v1, 5, v0                  // 32 t
  v_lshlrev_b32_e32 v8, 2, v0                  // 4 t
  s_waitcnt lgkmcnt(0)
  global_load_dwordx4 v[2:5], v1, s[4:5]
  global_load_dwordx2 v[6:7], v1, s[4:5] offset:16
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v10, 0
  body
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel \name
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 16
    .amdhsa_ieee_mode \ieee
    .amdhsa_dx10_clamp 1
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
.endm

  kernel nan_k, 1
  kernel nan_n, 0

#define KMETA(n) \
  - { .name: n, .symbol: n.kd, .kernarg_segment_size: 16, .kernarg_segment_align: 8, .group_segment_fixed_size: 0, \
      .private_segment_fixed_size: 0, .wavefront_size: 32, .sgpr_count: 16, .vgpr_count: 16, .max_flat_workgroup_size: 64, \
      .args: [ { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }, \
               { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global } ] }

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  KMETA(nan_k)
  KMETA(nan_n)
...
  .end_amdgpu_metadata
