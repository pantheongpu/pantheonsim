// Where a wave runs, as RDNA4 says it: HW_ID1's workgroup processor, shader
// array and engine (hardware register 23). And register 4, which was HW_ID
// before gfx12 and is STATE_PRIV on it, holding SCC in bit 9: HIP's __smid
// still reads it on gfx12. Built for gfx1201 by build.sh, with clang's
// assembler.
//
// where(uint* out): one wave a work-group (whose number RDNA4 gives in
// TTMP9), writing HW_ID1, then STATE_PRIV with SCC set and with it clear, at
// out[3 * group].
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
  .text
  .globl where
  .p2align 8
  .type where,@function
where:
  s_load_b64 s[4:5], s[0:1], 0x0
  s_getreg_b32 s6, hwreg(23, 0, 32)
  s_cmp_eq_u32 0, 0
  s_getreg_b32 s7, hwreg(4, 0, 32)
  s_cmp_eq_u32 0, 1
  s_getreg_b32 s8, hwreg(4, 0, 32)
  s_mul_i32 s9, ttmp9, 12
  v_mov_b32_e32 v0, s9
  v_mov_b32_e32 v1, s6
  v_mov_b32_e32 v2, s7
  v_mov_b32_e32 v3, s8
  s_wait_kmcnt 0x0
  s_mov_b32 exec_lo, 1                         // one lane writes
  global_store_b96 v0, v[1:3], s[4:5]
  s_endpgm
.Lwhere_end:
  .size where, .Lwhere_end-where

  .rodata
  .p2align 6
  .amdhsa_kernel where
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 10
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
    .sgpr_count: 10
    .vgpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
