// Vector instructions named by hand, as Tensile's GEMMs and rocBLAS's kernels
// use them: a comparison that narrows EXEC as well as writing its mask, a
// signed 64-bit multiply-add that says when it overflowed, packed halves and
// packed floats with op_sel choosing which part of each source feeds which
// result and neg_lo and neg_hi negating them on the way, a packed move, a
// mixed-precision multiply-add taking the top half of a register, and a
// comparison of part of a register. Built for gfx942 by build.sh, with
// clang's assembler.
//
// vector(int* out, int x, int y) writes 24 words, in the order the test
// lists them; x is the number of lanes the first comparison keeps.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl vector
  .p2align 8
  .type vector,@function
vector:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_load_dwordx2 s[4:5], s[0:1], 0x8          // s4 = x, s5 = y
  s_waitcnt lgkmcnt(0)
  // The comparisons that write EXEC: lanes below x, then of those lane 3.
  v_mbcnt_lo_u32_b32 v0, -1, 0
  v_mbcnt_hi_u32_b32 v0, -1, v0
  // Every lane multiplies its number by s40, and the carry out goes to the
  // pair s40 is in: each lane must read s40 as it was, not as an earlier
  // lane left it (rocBLAS's strided scal does exactly this).
  s_mov_b32 s40, 3
  s_mov_b32 s41, 0
  v_mad_u64_u32 v[26:27], s[40:41], s40, v0, 0
  v_readlane_b32 s42, v26, 63
  v_cmpx_gt_u32_e32 vcc, s4, v0
  s_bcnt1_i32_b64 s20, exec
  s_bcnt1_i32_b64 s21, vcc
  v_cmpx_eq_u32_e64 s[22:23], v0, 3
  s_bcnt1_i32_b64 s24, exec
  s_ff1_i32_b64 s25, exec
  s_mov_b64 exec, 1                            // from here, one lane
  // x * y added to 0x7ffffffffffffff0, which overflows.
  v_mov_b32_e32 v30, s4
  v_mov_b32_e32 v31, s5
  v_mov_b32_e32 v4, 0xfffffff0
  v_mov_b32_e32 v5, 0x7fffffff
  v_mad_i64_i32 v[2:3], s[26:27], v30, v31, v[4:5]
  // Packed halves: v6 = (1.5, -2.0), v7 = (0.25, 3.0), low half first.
  v_mov_b32_e32 v6, 0xc0003e00
  v_mov_b32_e32 v7, 0x42003400
  v_pk_add_f16 v8, v6, v7 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0]
  v_pk_max_f16 v9, v6, v7 neg_hi:[0,1]
  v_pk_fma_f16 v10, v6, v7, v6 op_sel_hi:[1,1,0]
  // Packed floats: s[8:9] = (2, 5), v[12:13] = (3, 7).
  s_mov_b32 s8, 2.0
  s_mov_b32 s9, 0x40a00000
  v_mov_b32_e32 v12, 0x40400000
  v_mov_b32_e32 v13, 0x40e00000
  v_pk_mul_f32 v[14:15], s[8:9], v[12:13] op_sel:[1,0] op_sel_hi:[0,1]
  v_pk_add_f32 v[16:17], v[12:13], s[8:9] neg_hi:[0,1]
  v_pk_mov_b32 v[18:19], v[12:13], s[8:9] op_sel:[1,1]
  // The top half of v6 as a half, times a float, plus a float.
  v_fma_mix_f32 v20, v6, v12, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0]
  // The high word of v21 against v22, and then its low word.
  v_mov_b32_e32 v21, 0x50003
  v_mov_b32_e32 v22, 5
  v_cmp_eq_u32_sdwa s[28:29], v21, v22 src0_sel:WORD_1 src1_sel:DWORD
  v_cmp_eq_u32_sdwa s[30:31], v21, v22 src0_sel:WORD_0 src1_sel:DWORD
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, s20
  global_store_dword v0, v1, s[2:3]
  v_mov_b32_e32 v1, s21
  global_store_dword v0, v1, s[2:3] offset:4
  v_mov_b32_e32 v1, s24
  global_store_dword v0, v1, s[2:3] offset:8
  v_mov_b32_e32 v1, s25
  global_store_dword v0, v1, s[2:3] offset:12
  global_store_dword v0, v2, s[2:3] offset:16
  global_store_dword v0, v3, s[2:3] offset:20
  v_mov_b32_e32 v1, s26
  global_store_dword v0, v1, s[2:3] offset:24
  global_store_dword v0, v8, s[2:3] offset:28
  global_store_dword v0, v9, s[2:3] offset:32
  global_store_dword v0, v10, s[2:3] offset:36
  global_store_dword v0, v14, s[2:3] offset:40
  global_store_dword v0, v15, s[2:3] offset:44
  global_store_dword v0, v16, s[2:3] offset:48
  global_store_dword v0, v17, s[2:3] offset:52
  global_store_dword v0, v18, s[2:3] offset:56
  global_store_dword v0, v19, s[2:3] offset:60
  global_store_dword v0, v20, s[2:3] offset:64
  v_mov_b32_e32 v1, s28
  global_store_dword v0, v1, s[2:3] offset:68
  v_mov_b32_e32 v1, s30
  global_store_dword v0, v1, s[2:3] offset:72
  v_mov_b32_e32 v1, s22
  global_store_dword v0, v1, s[2:3] offset:76
  v_mov_b32_e32 v1, s23
  global_store_dword v0, v1, s[2:3] offset:80
  v_mov_b32_e32 v1, s27
  global_store_dword v0, v1, s[2:3] offset:84
  v_mov_b32_e32 v1, s42
  global_store_dword v0, v1, s[2:3] offset:88
  v_mov_b32_e32 v1, s40
  global_store_dword v0, v1, s[2:3] offset:92
  s_endpgm
.Lvector_end:
  .size vector, .Lvector_end-vector

  .rodata
  .p2align 6
  .amdhsa_kernel vector
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 48
    .amdhsa_accum_offset 32
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: vector
    .symbol: vector.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 48
    .vgpr_count: 32
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
