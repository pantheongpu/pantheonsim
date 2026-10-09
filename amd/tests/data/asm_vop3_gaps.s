// VOP3 instructions a compiler emits from the range the first kernels did not reach: the 16-bit median,
// three-way and multiply-add forms, the unsigned three-way maximum, a byte-aligned extract, the sums of
// absolute differences, the byte average, and the packed and normalized conversions. rocPRIM's merge sort
// stopped on v_med3_u16. Built for gfx942 by build.sh, with clang's assembler.
//
// gaps(in, fin, out): lane l reads eight words from in[8l] (v2..v9), three floats from fin[3l] (v10..v12, and
// the same as halves in v13..v15), and writes 34 words to out[34l] (v16..v49):
//   0 med3_u16(w0,w1,w2)  1 med3_i16  2 med3_f16(h0,h1,h2)  3 min3_f16  4 max3_f16  5 mad_f16  6 mad_u16(w0,w1,w2)
//   7 mad_i16  8 max3_u32  9 alignbyte_b32  10 lerp_u8  11 sad_u8  12 sad_hi_u8  13 sad_u16  14 sad_u32  15 msad_u8
//   16 cvt_pk_u8_f32(f0, w1, w2)  17 mad_i32_i16  18 cvt_pk_i16_i32(w0,w1)  19 cvt_pknorm_i16_f32(f0,f1)
//   20 cvt_pknorm_u16_f32  21 cvt_pknorm_i16_f16(h0,h1)  22 cvt_pknorm_u16_f16  23 add_i16(w0,w1)  24 sub_i16
//   25 mul_legacy_f32(f0,f1)  26-27 qsad_pk_u16_u8(w0w1, w2, w4w5)  28-29 mqsad_pk_u16_u8
//   30-33 mqsad_u32_u8(w0w1, w2, w4..w7)
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl gaps
  .p2align 8
  .type gaps,@function
gaps:
  s_load_dwordx4 s[4:7], s[0:1], 0x0           // in, fin
  s_load_dwordx2 s[8:9], s[0:1], 0x10          // out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v50, 5, v0                 // 32 bytes of words a lane
  v_mul_u32_u24_e32 v51, 12, v0                // 12 of floats
  v_mul_u32_u24_e32 v52, 136, v0               // 136 of results
  global_load_dwordx4 v[2:5], v50, s[4:5]
  global_load_dwordx4 v[6:9], v50, s[4:5] offset:16
  global_load_dword v10, v51, s[6:7]
  global_load_dword v11, v51, s[6:7] offset:4
  global_load_dword v12, v51, s[6:7] offset:8
  s_waitcnt vmcnt(0)
  v_cvt_f16_f32_e32 v13, v10
  v_cvt_f16_f32_e32 v14, v11
  v_cvt_f16_f32_e32 v15, v12
  v_med3_u16 v16, v2, v3, v4
  v_med3_i16 v17, v2, v3, v4
  v_med3_f16 v18, v13, v14, v15
  v_min3_f16 v19, v13, v14, v15
  v_max3_f16 v20, v13, v14, v15
  v_mad_f16 v21, v13, v14, v15
  v_mad_u16 v22, v2, v3, v4
  v_mad_i16 v23, v2, v3, v4
  v_max3_u32 v24, v2, v3, v4
  v_alignbyte_b32 v25, v2, v3, v4
  v_lerp_u8 v26, v2, v3, v4
  v_sad_u8 v27, v2, v3, v4
  v_sad_hi_u8 v28, v2, v3, v4
  v_sad_u16 v29, v2, v3, v4
  v_sad_u32 v30, v2, v3, v4
  v_msad_u8 v31, v2, v3, v4
  v_cvt_pk_u8_f32 v32, v10, v3, v4
  v_mad_i32_i16 v33, v2, v3, v4
  v_cvt_pk_i16_i32 v34, v2, v3
  v_cvt_pknorm_i16_f32 v35, v10, v11
  v_cvt_pknorm_u16_f32 v36, v10, v11
  v_cvt_pknorm_i16_f16 v37, v13, v14
  v_cvt_pknorm_u16_f16 v38, v13, v14
  v_add_i16 v39, v2, v3
  v_sub_i16 v40, v2, v3
  v_mul_legacy_f32 v41, v10, v11
  v_qsad_pk_u16_u8 v[42:43], v[2:3], v4, v[6:7]
  v_mqsad_pk_u16_u8 v[44:45], v[2:3], v4, v[6:7]
  v_mqsad_u32_u8 v[46:49], v[2:3], v4, v[6:9]
  global_store_dwordx4 v52, v[16:19], s[8:9]
  global_store_dwordx4 v52, v[20:23], s[8:9] offset:16
  global_store_dwordx4 v52, v[24:27], s[8:9] offset:32
  global_store_dwordx4 v52, v[28:31], s[8:9] offset:48
  global_store_dwordx4 v52, v[32:35], s[8:9] offset:64
  global_store_dwordx4 v52, v[36:39], s[8:9] offset:80
  global_store_dwordx4 v52, v[40:43], s[8:9] offset:96
  global_store_dwordx4 v52, v[44:47], s[8:9] offset:112
  global_store_dwordx2 v52, v[48:49], s[8:9] offset:128
  s_endpgm
.Lgaps_end:
  .size gaps, .Lgaps_end-gaps

  .rodata
  .p2align 6
  .amdhsa_kernel gaps
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 56
    .amdhsa_next_free_sgpr 12
    .amdhsa_accum_offset 56
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: gaps
    .symbol: gaps.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 56
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
