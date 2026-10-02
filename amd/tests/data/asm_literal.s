// RDNA's packed 16-bit instructions with a literal, which is the whole pair:
// its low 16 bits for the low result, its top 16 for the high one. gfx9's
// packed instructions take no literal at all. hipRTC's fp16 header test
// builds v_dot2_f32_f16 with 0x42004200 (3.0 in both halves). Built for
// gfx1100 by build.sh, with clang's assembler.
//
// packed_literals(uint* out): v1 holds 1.0 low and 2.0 high;
// out[0] = v_dot2_f32_f16(v1, 0x42004200) = 1*3 + 2*3 = 9.0f;
// out[1] = v_pk_add_f16(v1, 0x44004200) = 4.0 low, 6.0 high = 0x46004400.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .text
  .globl packed_literals
  .p2align 8
  .type packed_literals,@function
packed_literals:
  s_load_b64 s[2:3], s[0:1], 0x0
  v_mov_b32_e32 v1, 0x40003c00
  v_dot2_f32_f16 v2, v1, 0x42004200, 0
  v_pk_add_f16 v3, v1, 0x44004200
  v_mov_b32_e32 v0, 0
  s_waitcnt lgkmcnt(0)
  global_store_b64 v0, v[2:3], s[2:3]
  s_endpgm
.Lpacked_literals_end:
  .size packed_literals, .Lpacked_literals_end-packed_literals

  .rodata
  .p2align 6
  .amdhsa_kernel packed_literals
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: packed_literals
    .symbol: packed_literals.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
