// v_dot2_f32_f16: two pairs of halves multiplied and added into a float,
// with the clamp bit (which does not hold the result to [0, 1], as other
// float results' does) and the negate modifiers. Built for gfx942 by
// build.sh, with clang's assembler.
//
// dot_clamp(in, out): lane l reads a = in[3l], b = in[3l+1] (two halves
// each) and c = in[3l+2] (a float), and writes out[8l ..] = dot(a, b, c);
// the same with clamp; with neg_lo on a; with neg_hi on b; with neg_lo on c;
// with clamp and neg_lo on a; v_dot2c_f32_f16 into c; with neg_lo on b's low
// half and neg_hi on a's high one.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl dot_clamp
  .p2align 8
  .type dot_clamp,@function
dot_clamp:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // in, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_mul_u32_u24_e32 v1, 12, v0                 // three words a lane
  global_load_dwordx3 v[2:4], v1, s[4:5]       // a, b, c
  v_lshlrev_b32_e32 v1, 5, v0                  // eight words a lane
  s_waitcnt vmcnt(0)
  v_dot2_f32_f16 v8, v2, v3, v4
  v_dot2_f32_f16 v9, v2, v3, v4 clamp
  v_dot2_f32_f16 v10, v2, v3, v4 neg_lo:[1,0,0]
  v_dot2_f32_f16 v11, v2, v3, v4 neg_hi:[0,1,0]
  v_dot2_f32_f16 v12, v2, v3, v4 neg_lo:[0,0,1]
  v_dot2_f32_f16 v13, v2, v3, v4 neg_lo:[1,0,0] clamp
  v_mov_b32_e32 v14, v4
  v_dot2c_f32_f16_e32 v14, v2, v3
  v_dot2_f32_f16 v15, v2, v3, v4 neg_lo:[0,1,0] neg_hi:[1,0,0]
  global_store_dwordx4 v1, v[8:11], s[6:7]
  global_store_dwordx2 v1, v[12:13], s[6:7] offset:16
  global_store_dwordx2 v1, v[14:15], s[6:7] offset:24
  s_endpgm
.Ldot_clamp_end:
  .size dot_clamp, .Ldot_clamp_end-dot_clamp

  .rodata
  .p2align 6
  .amdhsa_kernel dot_clamp
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: dot_clamp
    .symbol: dot_clamp.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
