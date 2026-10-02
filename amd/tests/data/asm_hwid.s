// Where a wave runs, as CDNA3 says it: HW_ID's compute unit, shader array and
// engine (hardware register 4) and XCC_ID's compute die (register 20), which
// HIP's __smid reads. Built for gfx942 by build.sh, with clang's assembler.
//
// where(uint* out): one wave a work-group, writing HW_ID and XCC_ID at
// out[2 * group].
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl where
  .p2align 8
  .type where,@function
where:
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_getreg_b32 s6, hwreg(4, 0, 32)
  s_getreg_b32 s7, hwreg(20, 0, 32)
  s_lshl_b32 s8, s2, 3
  v_mov_b32_e32 v2, s8
  v_mov_b32_e32 v0, s6
  v_mov_b32_e32 v1, s7
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1                            // one lane writes
  global_store_dwordx2 v2, v[0:1], s[4:5]
  s_endpgm
.Lwhere_end:
  .size where, .Lwhere_end-where

  .rodata
  .p2align 6
  .amdhsa_kernel where
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 9
    .amdhsa_accum_offset 4
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
    .wavefront_size: 64
    .sgpr_count: 9
    .vgpr_count: 3
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
