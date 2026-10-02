// RDNA4's scalar half instructions with inline float constants, which are
// the half's own encoding (2.0 is 0x4000), not a float's bits (whose low
// half is zero). hip-tests' __halfMath compiles to these on gfx12. Built for
// gfx1201 by build.sh, with clang's assembler.
//
// salu_half(uint* out): s_fmac_f16 3.0 + 1.0 * 2.0 = 5.0 (0x4500);
// s_add_f16 1.0 + 1.0 = 2.0 (0x4000); s_mul_f16 1.0 * 0.5 = 0.5 (0x3800);
// s_cmp_eq_f16 of that 2.0 against 2.0, as 1 or 0.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
  .text
  .globl salu_half
  .p2align 8
  .type salu_half,@function
salu_half:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_movk_i32 s4, 0x4200
  s_movk_i32 s5, 0x3c00
  s_fmac_f16 s4, s5, 2.0
  s_add_f16 s6, s5, 1.0
  s_mul_f16 s7, s5, 0.5
  s_cmp_eq_f16 s6, 2.0
  s_cselect_b32 s8, 1, 0
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, s4
  v_mov_b32_e32 v2, s6
  v_mov_b32_e32 v3, s7
  v_mov_b32_e32 v4, s8
  s_wait_kmcnt 0x0
  global_store_b128 v0, v[1:4], s[2:3]
  s_endpgm
.Lsalu_half_end:
  .size salu_half, .Lsalu_half_end-salu_half

  .rodata
  .p2align 6
  .amdhsa_kernel salu_half
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 5
    .amdhsa_next_free_sgpr 9
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: salu_half
    .symbol: salu_half.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 9
    .vgpr_count: 5
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
