// Atomic adds from dispatches on different host threads (test_amd_gcn_atomics, issue #610). Every lane of the wave adds 1 to the same
// word, `iters` times, with no return: a wave of read-modify-writes and nothing else, so that a dispatch running alone is nearly always
// in the middle of one when another dispatch starts. Built for gfx942 by build.sh, with clang's assembler.
//
// race_global(unsigned* counter, unsigned iters): global_atomic_add_u32 with a scalar base.
// race_buffer(unsigned* counter, unsigned iters): buffer_atomic_add_u32 through a resource of the one word.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl race_global
  .p2align 8
  .type race_global,@function
race_global:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = counter
  s_load_dword s6, s[0:1], 0x8                // s6 = iters
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, 1
.Lglobal:
  global_atomic_add v1, v2, s[4:5]
  s_sub_u32 s6, s6, 1
  s_cmp_lg_u32 s6, 0
  s_cbranch_scc1 .Lglobal
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel race_global
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .globl race_buffer
  .p2align 8
  .type race_buffer,@function
race_buffer:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = counter
  s_load_dword s12, s[0:1], 0x8               // s12 = iters
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s8, s4                             // the resource: counter, no stride, 4 bytes, a 32-bit data format
  s_and_b32 s9, s5, 0xffff
  s_mov_b32 s10, 4
  s_mov_b32 s11, 0x20000
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v2, 1
.Lbuffer:
  buffer_atomic_add v2, v0, s[8:11], 0 offen
  s_sub_u32 s12, s12, 1
  s_cmp_lg_u32 s12, 0
  s_cbranch_scc1 .Lbuffer
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel race_buffer
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: race_global
    .symbol: race_global.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 8, .value_kind: by_value }
  - .name: race_buffer
    .symbol: race_buffer.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 8, .value_kind: by_value }
...
  .end_amdgpu_metadata
