// LDS's 64-bit read-modify-writes, and the _rtn forms that hand back what was
// there, with the scalar bit operations hip-tests' cooperative groups and
// atomics reach (s_brev_b64, s_bcnt0, s_ff0, s_wqm, s_cmov_b64). Built for
// gfx942 by build.sh, with clang's assembler.
//
// lds64(in, out): lane l reads a = in[2l], b = in[2l+1] (64-bit each). For
// op k (in the order below), the lane's LDS slot at k*512 + 8l is set to a,
// the op applied with b, and the lane writes out[68l + 2k] = what a _rtn
// form handed back (0 for the others) and out[68l + 2k + 1] = the slot
// after. Then out[68l + 60 ..] holds, as 32-bit words, the scalar results
// on lane 0's a: brev64 (2 words), bcnt0_b64, bcnt0_b32, ff0_b32, ff0_b64,
// wqm_b64 (2), wqm_b32, cmov_b64 taken (2), cmov_b64 not taken (2), 0.
//
//    0 ds_add_u64
//    1 ds_sub_u64
//    2 ds_rsub_u64
//    3 ds_inc_u64
//    4 ds_dec_u64
//    5 ds_min_i64
//    6 ds_max_i64
//    7 ds_min_u64
//    8 ds_max_u64
//    9 ds_and_b64
//   10 ds_or_b64
//   11 ds_xor_b64
//   12 ds_add_f64
//   13 ds_min_f64
//   14 ds_max_f64
//   15 ds_add_rtn_u64
//   16 ds_sub_rtn_u64
//   17 ds_rsub_rtn_u64
//   18 ds_inc_rtn_u64
//   19 ds_dec_rtn_u64
//   20 ds_min_rtn_i64
//   21 ds_max_rtn_i64
//   22 ds_min_rtn_u64
//   23 ds_max_rtn_u64
//   24 ds_and_rtn_b64
//   25 ds_or_rtn_b64
//   26 ds_xor_rtn_b64
//   27 ds_wrxchg_rtn_b64
//   28 ds_min_rtn_f64
//   29 ds_max_rtn_f64
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl lds64
  .p2align 8
  .type lds64,@function
lds64:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // in, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v6, 3, v0                  // its LDS slot, and its word of in
  v_lshlrev_b32_e32 v1, 4, v0
  global_load_dwordx4 v[2:5], v1, s[4:5]       // a, b
  v_mul_u32_u24_e32 v1, 0x220, v0              // 68 words of 8 bytes a lane
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  s_waitcnt vmcnt(0)
  ds_write_b64 v6, v[2:3] offset:0
  s_waitcnt lgkmcnt(0)
  ds_add_u64 v6, v[4:5] offset:0
  ds_read_b64 v[10:11], v6 offset:0
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:0
  ds_write_b64 v6, v[2:3] offset:512
  s_waitcnt lgkmcnt(0)
  ds_sub_u64 v6, v[4:5] offset:512
  ds_read_b64 v[10:11], v6 offset:512
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:16
  ds_write_b64 v6, v[2:3] offset:1024
  s_waitcnt lgkmcnt(0)
  ds_rsub_u64 v6, v[4:5] offset:1024
  ds_read_b64 v[10:11], v6 offset:1024
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:32
  ds_write_b64 v6, v[2:3] offset:1536
  s_waitcnt lgkmcnt(0)
  ds_inc_u64 v6, v[4:5] offset:1536
  ds_read_b64 v[10:11], v6 offset:1536
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:48
  ds_write_b64 v6, v[2:3] offset:2048
  s_waitcnt lgkmcnt(0)
  ds_dec_u64 v6, v[4:5] offset:2048
  ds_read_b64 v[10:11], v6 offset:2048
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:64
  ds_write_b64 v6, v[2:3] offset:2560
  s_waitcnt lgkmcnt(0)
  ds_min_i64 v6, v[4:5] offset:2560
  ds_read_b64 v[10:11], v6 offset:2560
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:80
  ds_write_b64 v6, v[2:3] offset:3072
  s_waitcnt lgkmcnt(0)
  ds_max_i64 v6, v[4:5] offset:3072
  ds_read_b64 v[10:11], v6 offset:3072
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:96
  ds_write_b64 v6, v[2:3] offset:3584
  s_waitcnt lgkmcnt(0)
  ds_min_u64 v6, v[4:5] offset:3584
  ds_read_b64 v[10:11], v6 offset:3584
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:112
  ds_write_b64 v6, v[2:3] offset:4096
  s_waitcnt lgkmcnt(0)
  ds_max_u64 v6, v[4:5] offset:4096
  ds_read_b64 v[10:11], v6 offset:4096
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:128
  ds_write_b64 v6, v[2:3] offset:4608
  s_waitcnt lgkmcnt(0)
  ds_and_b64 v6, v[4:5] offset:4608
  ds_read_b64 v[10:11], v6 offset:4608
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:144
  ds_write_b64 v6, v[2:3] offset:5120
  s_waitcnt lgkmcnt(0)
  ds_or_b64 v6, v[4:5] offset:5120
  ds_read_b64 v[10:11], v6 offset:5120
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:160
  ds_write_b64 v6, v[2:3] offset:5632
  s_waitcnt lgkmcnt(0)
  ds_xor_b64 v6, v[4:5] offset:5632
  ds_read_b64 v[10:11], v6 offset:5632
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:176
  ds_write_b64 v6, v[2:3] offset:6144
  s_waitcnt lgkmcnt(0)
  ds_add_f64 v6, v[4:5] offset:6144
  ds_read_b64 v[10:11], v6 offset:6144
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:192
  ds_write_b64 v6, v[2:3] offset:6656
  s_waitcnt lgkmcnt(0)
  ds_min_f64 v6, v[4:5] offset:6656
  ds_read_b64 v[10:11], v6 offset:6656
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:208
  ds_write_b64 v6, v[2:3] offset:7168
  s_waitcnt lgkmcnt(0)
  ds_max_f64 v6, v[4:5] offset:7168
  ds_read_b64 v[10:11], v6 offset:7168
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:224
  ds_write_b64 v6, v[2:3] offset:7680
  s_waitcnt lgkmcnt(0)
  ds_add_rtn_u64 v[8:9], v6, v[4:5] offset:7680
  ds_read_b64 v[10:11], v6 offset:7680
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:240
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:8192
  s_waitcnt lgkmcnt(0)
  ds_sub_rtn_u64 v[8:9], v6, v[4:5] offset:8192
  ds_read_b64 v[10:11], v6 offset:8192
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:256
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:8704
  s_waitcnt lgkmcnt(0)
  ds_rsub_rtn_u64 v[8:9], v6, v[4:5] offset:8704
  ds_read_b64 v[10:11], v6 offset:8704
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:272
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:9216
  s_waitcnt lgkmcnt(0)
  ds_inc_rtn_u64 v[8:9], v6, v[4:5] offset:9216
  ds_read_b64 v[10:11], v6 offset:9216
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:288
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:9728
  s_waitcnt lgkmcnt(0)
  ds_dec_rtn_u64 v[8:9], v6, v[4:5] offset:9728
  ds_read_b64 v[10:11], v6 offset:9728
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:304
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:10240
  s_waitcnt lgkmcnt(0)
  ds_min_rtn_i64 v[8:9], v6, v[4:5] offset:10240
  ds_read_b64 v[10:11], v6 offset:10240
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:320
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:10752
  s_waitcnt lgkmcnt(0)
  ds_max_rtn_i64 v[8:9], v6, v[4:5] offset:10752
  ds_read_b64 v[10:11], v6 offset:10752
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:336
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:11264
  s_waitcnt lgkmcnt(0)
  ds_min_rtn_u64 v[8:9], v6, v[4:5] offset:11264
  ds_read_b64 v[10:11], v6 offset:11264
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:352
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:11776
  s_waitcnt lgkmcnt(0)
  ds_max_rtn_u64 v[8:9], v6, v[4:5] offset:11776
  ds_read_b64 v[10:11], v6 offset:11776
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:368
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:12288
  s_waitcnt lgkmcnt(0)
  ds_and_rtn_b64 v[8:9], v6, v[4:5] offset:12288
  ds_read_b64 v[10:11], v6 offset:12288
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:384
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:12800
  s_waitcnt lgkmcnt(0)
  ds_or_rtn_b64 v[8:9], v6, v[4:5] offset:12800
  ds_read_b64 v[10:11], v6 offset:12800
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:400
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:13312
  s_waitcnt lgkmcnt(0)
  ds_xor_rtn_b64 v[8:9], v6, v[4:5] offset:13312
  ds_read_b64 v[10:11], v6 offset:13312
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:416
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:13824
  s_waitcnt lgkmcnt(0)
  ds_wrxchg_rtn_b64 v[8:9], v6, v[4:5] offset:13824
  ds_read_b64 v[10:11], v6 offset:13824
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:432
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:14336
  s_waitcnt lgkmcnt(0)
  ds_min_rtn_f64 v[8:9], v6, v[4:5] offset:14336
  ds_read_b64 v[10:11], v6 offset:14336
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:448
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  ds_write_b64 v6, v[2:3] offset:14848
  s_waitcnt lgkmcnt(0)
  ds_max_rtn_f64 v[8:9], v6, v[4:5] offset:14848
  ds_read_b64 v[10:11], v6 offset:14848
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v1, v[8:11], s[6:7] offset:464
  v_mov_b32_e32 v8, 0
  v_mov_b32_e32 v9, 0
  s_load_dwordx2 s[10:11], s[4:5], 0x0         // lane 0's a
  s_waitcnt lgkmcnt(0)
  s_brev_b64 s[12:13], s[10:11]
  s_bcnt0_i32_b64 s14, s[10:11]
  s_bcnt0_i32_b32 s15, s10
  s_ff0_i32_b32 s16, s10
  s_ff0_i32_b64 s17, s[10:11]
  s_wqm_b64 s[18:19], s[10:11]
  s_wqm_b32 s20, s10
  s_mov_b64 s[22:23], 0
  s_mov_b64 s[24:25], 0
  s_cmp_eq_u32 0, 0
  s_cmov_b64 s[22:23], s[10:11]
  s_cmp_eq_u32 0, 1
  s_cmov_b64 s[24:25], s[10:11]
  v_mov_b32_e32 v12, s12
  v_mov_b32_e32 v13, s13
  v_mov_b32_e32 v14, s14
  v_mov_b32_e32 v15, s15
  v_mov_b32_e32 v16, s16
  v_mov_b32_e32 v17, s17
  v_mov_b32_e32 v18, s18
  v_mov_b32_e32 v19, s19
  v_mov_b32_e32 v20, s20
  v_mov_b32_e32 v21, s22
  v_mov_b32_e32 v22, s23
  v_mov_b32_e32 v23, s24
  v_mov_b32_e32 v24, s25
  v_mov_b32_e32 v25, 0
  global_store_dwordx4 v1, v[12:15], s[6:7] offset:480
  global_store_dwordx4 v1, v[16:19], s[6:7] offset:496
  global_store_dwordx4 v1, v[20:23], s[6:7] offset:512
  global_store_dwordx2 v1, v[24:25], s[6:7] offset:528
  s_endpgm
.Llds64_end:
  .size lds64, .Llds64_end-lds64

  .rodata
  .p2align 6
  .amdhsa_kernel lds64
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 15360
    .amdhsa_next_free_vgpr 26
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 28
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: lds64
    .symbol: lds64.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 15360
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 26
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata

