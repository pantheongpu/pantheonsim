// RDNA4's wave number within its work-group, which the hardware leaves in
// TTMP8's bits 25 to 29 and LLVM reads there (llvm.amdgcn.wave.id); Triton's
// kernels split a group's work by it. Built for gfx1201 by build.sh, with
// clang's assembler.
//
// wave_id(uint* out): each work-item writes its wave's number.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
  .text
  .globl wave_id
  .p2align 8
  .type wave_id,@function
wave_id:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_bfe_u32 s4, ttmp8, 0x50019
  s_wait_kmcnt 0x0
  v_mov_b32_e32 v1, s4
  v_and_b32_e32 v0, 0x3ff, v0
  v_lshlrev_b32_e32 v2, 2, v0
  global_store_b32 v2, v1, s[2:3]
  s_endpgm
.Lwave_id_end:
  .size wave_id, .Lwave_id_end-wave_id

  .rodata
  .p2align 6
  .amdhsa_kernel wave_id
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: wave_id
    .symbol: wave_id.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 3
    .max_flat_workgroup_size: 128
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
