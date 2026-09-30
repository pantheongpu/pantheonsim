// LDS's float read-modify-writes and compare-and-stores, and their _rtn
// forms, as hip-tests' float atomics on shared memory use them. Built for
// gfx942 by build.sh, with clang's assembler.
//
// ldsf32(in, out): lane l reads a = in[2l], b = in[2l+1] (floats' bits). For
// op k (in the order below), the lane's LDS word at k*256 + 4l is set to a,
// the op applied with b -- the compare-and-stores compare with a (so store b)
// for k 6 and 8, and with b for k 7 (so keep a) -- and the lane writes
// out[24l + 2k] = what a _rtn form handed back (0 for the others) and
// out[24l + 2k + 1] = the word after. out[24l + 18 ..] is ds_add_rtn_f64 on
// the pair (a, b) taken as one double, added to itself: handed back, after.
//
//   0 ds_add_f32
//   1 ds_add_rtn_f32
//   2 ds_min_f32
//   3 ds_min_rtn_f32
//   4 ds_max_f32
//   5 ds_max_rtn_f32
//   6 ds_cmpst_b32
//   7 ds_cmpst_f32
//   8 ds_cmpst_rtn_f32
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl ldsf32
  .p2align 8
  .type ldsf32,@function
ldsf32:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // in, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v6, 2, v0                  // its LDS word
  v_lshlrev_b32_e32 v1, 3, v0
  global_load_dwordx2 v[2:3], v1, s[4:5]       // a, b
  v_mul_u32_u24_e32 v1, 0x60, v0               // 24 words a lane
  v_mov_b32_e32 v8, 0
  s_waitcnt vmcnt(0)
  ds_write_b32 v6, v2 offset:0
  s_waitcnt lgkmcnt(0)
  ds_add_f32 v6, v3 offset:0
  ds_read_b32 v9, v6 offset:0
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:0
  ds_write_b32 v6, v2 offset:256
  s_waitcnt lgkmcnt(0)
  ds_add_rtn_f32 v8, v6, v3 offset:256
  ds_read_b32 v9, v6 offset:256
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:8
  v_mov_b32_e32 v8, 0
  ds_write_b32 v6, v2 offset:512
  s_waitcnt lgkmcnt(0)
  ds_min_f32 v6, v3 offset:512
  ds_read_b32 v9, v6 offset:512
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:16
  ds_write_b32 v6, v2 offset:768
  s_waitcnt lgkmcnt(0)
  ds_min_rtn_f32 v8, v6, v3 offset:768
  ds_read_b32 v9, v6 offset:768
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:24
  v_mov_b32_e32 v8, 0
  ds_write_b32 v6, v2 offset:1024
  s_waitcnt lgkmcnt(0)
  ds_max_f32 v6, v3 offset:1024
  ds_read_b32 v9, v6 offset:1024
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:32
  ds_write_b32 v6, v2 offset:1280
  s_waitcnt lgkmcnt(0)
  ds_max_rtn_f32 v8, v6, v3 offset:1280
  ds_read_b32 v9, v6 offset:1280
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:40
  v_mov_b32_e32 v8, 0
  ds_write_b32 v6, v2 offset:1536
  s_waitcnt lgkmcnt(0)
  ds_cmpst_b32 v6, v2, v3 offset:1536
  ds_read_b32 v9, v6 offset:1536
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:48
  ds_write_b32 v6, v2 offset:1792
  s_waitcnt lgkmcnt(0)
  ds_cmpst_f32 v6, v3, v3 offset:1792
  ds_read_b32 v9, v6 offset:1792
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:56
  ds_write_b32 v6, v2 offset:2048
  s_waitcnt lgkmcnt(0)
  ds_cmpst_rtn_f32 v8, v6, v2, v3 offset:2048
  ds_read_b32 v9, v6 offset:2048
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:64
  v_mov_b32_e32 v8, 0
  v_lshlrev_b32_e32 v7, 3, v0                  // a doubleword a lane, past the words
  ds_write_b64 v7, v[2:3] offset:4096
  s_waitcnt lgkmcnt(0)
  ds_add_rtn_f64 v[10:11], v7, v[2:3] offset:4096
  ds_read_b64 v[12:13], v7 offset:4096
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[10:13], s[6:7] offset:72
  s_endpgm
.Lldsf32_end:
  .size ldsf32, .Lldsf32_end-ldsf32

  .rodata
  .p2align 6
  .amdhsa_kernel ldsf32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 4608
    .amdhsa_next_free_vgpr 14
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: ldsf32
    .symbol: ldsf32.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 4608
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 14
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata

