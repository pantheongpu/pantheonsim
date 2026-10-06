// The 24-bit multiplies of VOP2: v_mul_i32_i24, v_mul_hi_i32_i24, v_mul_u32_u24, v_mul_hi_u32_u24, each in its
// VOP2 form and (the high signed one) VOP3's. A compiler emits v_mul_hi_i32_i24 for a multiply-high of values it
// can prove fit in 24 bits (llama.cpp's mul_mat_vec_q does), and the simulator did not decode it. Built for
// gfx942 by build.sh, with clang's assembler.
//
// mulhi(in, out): lane l reads a = in[2l], b = in[2l+1] and writes 5 words to out[5l]:
//   0 mul_i32_i24(a, b)   1 mul_hi_i32_i24(a, b)   2 mul_u32_u24(a, b)   3 mul_hi_u32_u24(a, b)
//   4 the VOP3 form of mul_hi_i32_i24(a, b)
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl mulhi
  .p2align 8
  .type mulhi,@function
mulhi:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // in, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 3, v0
  global_load_dwordx2 v[2:3], v1, s[4:5]       // a, b
  s_waitcnt vmcnt(0)
  v_mul_i32_i24_e32 v4, v2, v3
  v_mul_hi_i32_i24_e32 v5, v2, v3
  v_mul_u32_u24_e32 v6, v2, v3
  v_mul_hi_u32_u24_e32 v7, v2, v3
  v_mul_hi_i32_i24_e64 v8, v2, v3
  v_mul_u32_u24_e32 v1, 20, v0                 // 5 words a lane
  global_store_dwordx4 v1, v[4:7], s[6:7]
  global_store_dword v1, v8, s[6:7] offset:16
  s_endpgm
.Lmulhi_end:
  .size mulhi, .Lmulhi_end-mulhi

  .rodata
  .p2align 6
  .amdhsa_kernel mulhi
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 9
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 12
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: mulhi
    .symbol: mulhi.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 9
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
