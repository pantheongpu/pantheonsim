// A 16-byte store and load each as one access: group 0 stores {i, i, i, i}
// for i = 1 to 200000 with global_store_dwordx4, while the other groups,
// on other host threads, load the same sixteen bytes with global_load_dwordx4
// as often and count the loads whose four words differ (rocPRIM's look-back
// packs a tile's flag and 64-bit prefix this way). Built for gfx942 by
// build.sh, with clang's assembler.
//
// quads(uint4* data, uint* torn): torn[group] is what that group counted;
// group 0 leaves its count of stores there.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl quads
  .p2align 8
  .type quads,@function
quads:
  s_load_dwordx4 s[4:7], s[0:1], 0x0
  s_mov_b64 exec, 1                            // one lane
  v_mov_b32_e32 v4, 0
  v_mov_b32_e32 v5, 0
  s_mov_b32 s8, 0
  s_waitcnt lgkmcnt(0)
  s_cmp_eq_u32 s2, 0
  s_cbranch_scc0 .Lread
.Lwrite:
  s_add_u32 s8, s8, 1
  v_mov_b32_e32 v0, s8
  v_mov_b32_e32 v1, s8
  v_mov_b32_e32 v2, s8
  v_mov_b32_e32 v3, s8
  global_store_dwordx4 v4, v[0:3], s[4:5] sc0 sc1
  s_cmp_lt_u32 s8, 200000
  s_cbranch_scc1 .Lwrite
  v_mov_b32_e32 v5, s8
  s_branch .Ldone
.Lread:
  global_load_dwordx4 v[0:3], v4, s[4:5] sc0 sc1
  s_waitcnt vmcnt(0)
  v_xor_b32_e32 v6, v0, v1
  v_xor_b32_e32 v7, v0, v2
  v_or_b32_e32 v6, v6, v7
  v_xor_b32_e32 v7, v0, v3
  v_or_b32_e32 v6, v6, v7
  v_cmp_ne_u32_e32 vcc, 0, v6
  v_cndmask_b32_e64 v7, 0, 1, vcc
  v_add_u32_e32 v5, v5, v7
  s_add_u32 s8, s8, 1
  s_cmp_lt_u32 s8, 200000
  s_cbranch_scc1 .Lread
.Ldone:
  s_lshl_b32 s9, s2, 2
  v_mov_b32_e32 v6, s9
  global_store_dword v6, v5, s[6:7]
  s_endpgm
.Lquads_end:
  .size quads, .Lquads_end-quads

  .rodata
  .p2align 6
  .amdhsa_kernel quads
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 10
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: quads
    .symbol: quads.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 10
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
