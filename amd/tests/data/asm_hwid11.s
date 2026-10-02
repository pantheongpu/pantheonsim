// Where a wave runs, as RDNA3 says it: HW_ID1's workgroup processor, shader
// array and engine (hardware register 23), which HIP's __smid reads. Built
// for gfx1100 by build.sh, with clang's assembler.
//
// where(uint* out): one wave a work-group, writing HW_ID1 at out[group].
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .text
  .globl where
  .p2align 8
  .type where,@function
where:
  s_load_b64 s[4:5], s[0:1], 0x0
  s_getreg_b32 s6, hwreg(23, 0, 32)
  s_lshl_b32 s8, s2, 2
  v_mov_b32_e32 v0, s8
  v_mov_b32_e32 v1, s6
  s_waitcnt lgkmcnt(0)
  s_mov_b32 exec_lo, 1                         // one lane writes
  global_store_b32 v0, v1, s[4:5]
  s_endpgm
.Lwhere_end:
  .size where, .Lwhere_end-where

  .rodata
  .p2align 6
  .amdhsa_kernel where
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 9
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: where
    .symbol: where.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 9
    .vgpr_count: 2
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
