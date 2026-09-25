// Scalar instructions named by hand: the ones rocBLAS's kernels and Tensile's
// GEMMs use that a small C kernel would not -- bit fields, 64-bit shifts,
// min and max and the SCC they set, a bit set or tested, the comparisons
// against the instruction's own constant signed and unsigned, a move only
// where the last comparison held. Built for gfx942 by build.sh, with clang's
// assembler.
//
// scalar(int* out, int x, int y) writes 26 words, in the order the test
// lists them.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl scalar
  .p2align 8
  .type scalar,@function
scalar:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_load_dwordx2 s[4:5], s[0:1], 0x8          // s4 = x, s5 = y
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1                            // one lane writes
  s_bfe_u32 s6, s4, 0x80004                   // 8 bits of x from bit 4
  s_bfe_i64 s[8:9], s[4:5], 0xc0014           // 12 bits of y:x from bit 20, signed
  s_ashr_i64 s[10:11], s[4:5], 7
  s_min_i32 s12, s4, s5
  s_cselect_b32 s13, 1, 0
  s_min_u32 s14, s4, s5
  s_cselect_b32 s15, 1, 0
  s_max_i32 s16, s4, s5
  s_cselect_b32 s17, 1, 0
  s_mov_b32 s18, s4
  s_bitset1_b32 s18, 3
  s_bitset0_b32 s18, 31
  s_bitcmp1_b32 s4, 5
  s_cselect_b32 s19, 1, 0
  s_bitcmp0_b32 s4, 5
  s_cselect_b32 s20, 1, 0
  s_cmpk_gt_u32 s4, 0x8000
  s_cselect_b32 s21, 1, 0
  s_cmpk_lt_i32 s4, 0x8000
  s_cselect_b32 s22, 1, 0
  s_cmp_le_u32 s4, s5
  s_cselect_b32 s23, 1, 0
  s_cmp_eq_i32 s4, s5
  s_cselect_b32 s24, 1, 0
  s_abs_i32 s25, s4
  s_mul_hi_i32 s26, s4, s5
  s_pack_ll_b32_b16 s27, s4, s5
  s_not_b32 s28, s4
  s_andn2_b32 s29, s4, s5
  s_mov_b32 s30, 7
  s_cmp_lg_u32 s4, s5
  s_cmov_b32 s30, s4
  s_not_b64 s[32:33], s[4:5]
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, s6
  global_store_dword v0, v1, s[2:3]
  v_mov_b32_e32 v1, s8
  global_store_dword v0, v1, s[2:3] offset:4
  v_mov_b32_e32 v1, s9
  global_store_dword v0, v1, s[2:3] offset:8
  v_mov_b32_e32 v1, s10
  global_store_dword v0, v1, s[2:3] offset:12
  v_mov_b32_e32 v1, s11
  global_store_dword v0, v1, s[2:3] offset:16
  v_mov_b32_e32 v1, s12
  global_store_dword v0, v1, s[2:3] offset:20
  v_mov_b32_e32 v1, s13
  global_store_dword v0, v1, s[2:3] offset:24
  v_mov_b32_e32 v1, s14
  global_store_dword v0, v1, s[2:3] offset:28
  v_mov_b32_e32 v1, s15
  global_store_dword v0, v1, s[2:3] offset:32
  v_mov_b32_e32 v1, s16
  global_store_dword v0, v1, s[2:3] offset:36
  v_mov_b32_e32 v1, s17
  global_store_dword v0, v1, s[2:3] offset:40
  v_mov_b32_e32 v1, s18
  global_store_dword v0, v1, s[2:3] offset:44
  v_mov_b32_e32 v1, s19
  global_store_dword v0, v1, s[2:3] offset:48
  v_mov_b32_e32 v1, s20
  global_store_dword v0, v1, s[2:3] offset:52
  v_mov_b32_e32 v1, s21
  global_store_dword v0, v1, s[2:3] offset:56
  v_mov_b32_e32 v1, s22
  global_store_dword v0, v1, s[2:3] offset:60
  v_mov_b32_e32 v1, s23
  global_store_dword v0, v1, s[2:3] offset:64
  v_mov_b32_e32 v1, s24
  global_store_dword v0, v1, s[2:3] offset:68
  v_mov_b32_e32 v1, s25
  global_store_dword v0, v1, s[2:3] offset:72
  v_mov_b32_e32 v1, s26
  global_store_dword v0, v1, s[2:3] offset:76
  v_mov_b32_e32 v1, s27
  global_store_dword v0, v1, s[2:3] offset:80
  v_mov_b32_e32 v1, s28
  global_store_dword v0, v1, s[2:3] offset:84
  v_mov_b32_e32 v1, s29
  global_store_dword v0, v1, s[2:3] offset:88
  v_mov_b32_e32 v1, s30
  global_store_dword v0, v1, s[2:3] offset:92
  v_mov_b32_e32 v1, s32
  global_store_dword v0, v1, s[2:3] offset:96
  v_mov_b32_e32 v1, s33
  global_store_dword v0, v1, s[2:3] offset:100
  s_endpgm
.Lscalar_end:
  .size scalar, .Lscalar_end-scalar

  .rodata
  .p2align 6
  .amdhsa_kernel scalar
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 40
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: scalar
    .symbol: scalar.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 40
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
      - .size: 4
        .offset: 12
        .value_kind: by_value
...
  .end_amdgpu_metadata
