// v_cvt_f32_ubyte0..3: each byte of a word as a float. Built for gfx942 by
// build.sh, with clang's assembler.
//
// cvt_ubyte(in, out): lane l reads w = in[l] and writes out[8l + k] = the
// float of byte k of w, k = 0..3, then out[8l + 4 + k] = the same of in[0]
// read as a scalar, as the compiler converts a kernel argument's bytes.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl cvt_ubyte
  .p2align 8
  .type cvt_ubyte,@function
cvt_ubyte:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // in, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 2, v0
  global_load_dword v2, v1, s[4:5]
  s_load_dword s8, s[4:5], 0x0                // in[0], as a scalar
  v_lshlrev_b32_e32 v1, 5, v0                  // eight words a lane
  s_waitcnt vmcnt(0) lgkmcnt(0)
  v_cvt_f32_ubyte0_e32 v4, v2
  v_cvt_f32_ubyte1_e32 v5, v2
  v_cvt_f32_ubyte2_e32 v6, v2
  v_cvt_f32_ubyte3_e32 v7, v2
  v_cvt_f32_ubyte0_e32 v8, s8
  v_cvt_f32_ubyte1_e32 v9, s8
  v_cvt_f32_ubyte2_e32 v10, s8
  v_cvt_f32_ubyte3_e32 v11, s8
  global_store_dwordx4 v1, v[4:7], s[6:7]
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:16
  global_store_dwordx2 v1, v[10:11], s[6:7] offset:24
  s_endpgm
.Lcvt_ubyte_end:
  .size cvt_ubyte, .Lcvt_ubyte_end-cvt_ubyte

  .rodata
  .p2align 6
  .amdhsa_kernel cvt_ubyte
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 12
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 12
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: cvt_ubyte
    .symbol: cvt_ubyte.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 12
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
