// Instructions named by hand rather than left to a compiler to choose: the
// scalar add and multiply of a 16-bit constant (SOPK), at the edges where the
// add overflows. Built for gfx942 by build.sh, with clang's assembler.
//
// sopk(int* out, int x) writes, in order:
//   x + 32767, whether that overflowed (SCC), x - 32768, whether that
//   overflowed, and x * -3.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl sopk
  .p2align 8
  .type sopk,@function
sopk:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_load_dword s4, s[0:1], 0x8
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s5, s4
  s_addk_i32 s5, 0x7fff
  s_cselect_b32 s6, 1, 0
  s_mov_b32 s7, s4
  s_addk_i32 s7, 0x8000
  s_cselect_b32 s8, 1, 0
  s_mov_b32 s9, s4
  s_mulk_i32 s9, 0xfffd
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, s5
  global_store_dword v0, v1, s[2:3]
  v_mov_b32_e32 v1, s6
  global_store_dword v0, v1, s[2:3] offset:4
  v_mov_b32_e32 v1, s7
  global_store_dword v0, v1, s[2:3] offset:8
  v_mov_b32_e32 v1, s8
  global_store_dword v0, v1, s[2:3] offset:12
  v_mov_b32_e32 v1, s9
  global_store_dword v0, v1, s[2:3] offset:16
  s_endpgm
.Lsopk_end:
  .size sopk, .Lsopk_end-sopk

  .rodata
  .p2align 6
  .amdhsa_kernel sopk
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: sopk
    .symbol: sopk.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 2
    .max_flat_workgroup_size: 64
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .size: 4
        .offset: 8
        .value_kind: by_value
...
  .end_amdgpu_metadata
