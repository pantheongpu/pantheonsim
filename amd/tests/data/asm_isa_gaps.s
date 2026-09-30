// Instructions hip-tests' device library runs that a compiler emits only
// now and then: the integer dot products, v_fract_f32, integer adds and
// subtracts that saturate, the DPP moves across the whole wave, and the
// increment and decrement atomics in global memory and LDS. Built for gfx942
// by build.sh, with clang's assembler.
//
// gaps(in, out, counters): lane l reads a = in[4l], b = in[4l+1],
// c = in[4l+2], d = in[4l+3] and writes 20 words to out[20l]:
//   0 a + b, clamped    1 a - b, clamped    2 fract(c as a float)
//   3 dot2_i32_i16(a, b, d)   4 dot2_u32_u16(a, b, d)   5 dot4_u32_u8(a, b, d)
//   6 dot8_i32_i4(a, b, d)    7 dot8_u32_u4(a, b, d)
//   8 d + dot2c_i32_i16(a, b)  9 d + dot8c_i32_i4(a, b)
//   10-13 lane id moved by wave_shl:1, wave_shr:1, wave_rol:1, wave_ror:1
//         (0xffff where no lane is read)
//   14 counters[2l] before global_atomic_inc with limit b
//   15 counters[2l+1] before global_atomic_dec with limit b
//   16 a in LDS before ds_inc_rtn_u32 with limit b; 17 what that left,
//      handed back by ds_wrxchg_rtn_b32 as c goes in; 18 c read back
//   19 0
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl gaps
  .p2align 8
  .type gaps,@function
gaps:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // in, out
  s_load_dwordx2 s[8:9], s[0:1], 0x10         // counters
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 4, v0
  global_load_dwordx4 v[2:5], v1, s[4:5]       // a, b, c, d
  s_waitcnt vmcnt(0)
  v_add_u32_e64 v10, v2, v3 clamp
  v_sub_u32_e64 v11, v2, v3 clamp
  v_fract_f32_e32 v12, v4
  v_dot2_i32_i16 v13, v2, v3, v5
  v_dot2_u32_u16 v14, v2, v3, v5
  v_dot4_u32_u8 v15, v2, v3, v5
  v_dot8_i32_i4 v16, v2, v3, v5
  v_dot8_u32_u4 v17, v2, v3, v5
  v_mov_b32_e32 v18, v5
  v_dot2c_i32_i16_e32 v18, v2, v3
  v_mov_b32_e32 v19, v5
  v_dot8c_i32_i4_e32 v19, v2, v3
  v_mov_b32_e32 v20, 0xffff
  v_mov_b32_e32 v21, 0xffff
  v_mov_b32_e32 v22, 0xffff
  v_mov_b32_e32 v23, 0xffff
  s_nop 1
  v_mov_b32_dpp v20, v0 wave_shl:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v21, v0 wave_shr:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v22, v0 wave_rol:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v23, v0 wave_ror:1 row_mask:0xf bank_mask:0xf
  v_lshlrev_b32_e32 v6, 3, v0                   // counters, two a lane
  global_atomic_inc v24, v6, v3, s[8:9] sc0
  global_atomic_dec v25, v6, v3, s[8:9] offset:4 sc0
  v_lshlrev_b32_e32 v7, 2, v0                   // LDS, a word a lane
  ds_write_b32 v7, v2
  s_waitcnt lgkmcnt(0)
  ds_inc_rtn_u32 v26, v7, v3
  s_waitcnt lgkmcnt(0)
  ds_wrxchg_rtn_b32 v27, v7, v4
  s_waitcnt lgkmcnt(0)
  ds_read_b32 v28, v7
  v_mov_b32_e32 v29, 0
  s_waitcnt vmcnt(0) lgkmcnt(0)
  v_mul_u32_u24_e32 v1, 0x50, v0                // 20 words a lane
  global_store_dwordx4 v1, v[10:13], s[6:7]
  global_store_dwordx4 v1, v[14:17], s[6:7] offset:16
  global_store_dwordx4 v1, v[18:21], s[6:7] offset:32
  global_store_dwordx4 v1, v[22:25], s[6:7] offset:48
  global_store_dwordx4 v1, v[26:29], s[6:7] offset:64
  s_endpgm
.Lgaps_end:
  .size gaps, .Lgaps_end-gaps

  .rodata
  .p2align 6
  .amdhsa_kernel gaps
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 256
    .amdhsa_next_free_vgpr 30
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 32
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: gaps
    .symbol: gaps.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 30
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
