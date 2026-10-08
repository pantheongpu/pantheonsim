// The reduced-precision float matrix instructions, v_mfma_f32_16x16x8_xf32 and
// v_mfma_f32_32x32x4_xf32 (gfx940 to gfx942): A and B are floats whose mantissa
// is cut to 10 bits before the products. Built for gfx942 by build.sh, with
// clang's assembler.
//
// xf32(a, b, out): each lane loads two words of A and two of B and runs both
// forms with a zero C. The 16x16x8 result is four floats a lane, 1024 bytes;
// the 32x32x4 result is sixteen a lane, 4096 bytes after it.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl xf32
  .p2align 8
  .type xf32,@function
xf32:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // a, b
  s_load_dwordx2 s[8:9], s[0:1], 0x10         // out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 3, v0
  global_load_dwordx2 v[2:3], v1, s[4:5]
  global_load_dwordx2 v[4:5], v1, s[6:7]
  s_waitcnt vmcnt(0)
  v_mfma_f32_16x16x8_xf32 v[6:9], v[2:3], v[4:5], 0
  v_mfma_f32_32x32x4_xf32 v[10:25], v[2:3], v[4:5], 0
  s_nop 7
  s_nop 7
  s_nop 7
  s_nop 7
  v_lshlrev_b32_e32 v1, 4, v0
  global_store_dwordx4 v1, v[6:9], s[8:9]
  v_lshlrev_b32_e32 v1, 6, v0
  global_store_dwordx4 v1, v[10:13], s[8:9] offset:1024
  global_store_dwordx4 v1, v[14:17], s[8:9] offset:1040
  global_store_dwordx4 v1, v[18:21], s[8:9] offset:1056
  global_store_dwordx4 v1, v[22:25], s[8:9] offset:1072
  s_endpgm
.Lxf32_end:
  .size xf32, .Lxf32_end-xf32

  .rodata
  .p2align 6
  .amdhsa_kernel xf32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 32
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: xf32
    .symbol: xf32.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 32
    .max_flat_workgroup_size: 64
    .args:
      - { .name: a, .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global }
      - { .name: b, .offset: 8, .size: 8, .value_kind: global_buffer, .address_space: global }
      - { .name: out, .offset: 16, .size: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
