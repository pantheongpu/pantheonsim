// The rest of LDS's atomics and exchanges, from the CDNA ISA guides' pseudocode: DS_MSKOR_{B32,B64} (and _RTN), DS_WRXCHG2_RTN_B32 and _B64,
// DS_WRXCHG2ST64_RTN_B32 and _B64, DS_CMPST_B64, DS_CMPST_F64 and DS_CMPST_RTN_F64 (the comparison is the float one), DS_WRAP_RTN_B32,
// DS_CONDXCHG32_RTN_B64, DS_NOP, DS_APPEND and DS_CONSUME (the number of active lanes added to or taken from a counter, the old value to every
// lane), and DS_WRITE_ADDTID_B32 / DS_READ_ADDTID_B32 ("{OFFSET1, OFFSET0} + M0[15:0] + laneID * 4": M0's bits above 15 do not count).
//
// ds_gaps(init, data, out): lane L's 32 dwords of LDS (A) start as init[32 L ..]; its two dwords of B (4 L and 256 + 4 L from B) as
// init[2048 + L] and init[2112 + L]; its two quadwords of C (8 L and 512 + 8 L) as init[2176 + 2 L ..] and init[2304 + 2 L ..]. data[16 L ..]
// are its operands. out: 24 dwords a lane of what the instructions returned, then A, B and C as they end, then what ds_append and
// ds_consume gave back (48 lanes on), the counter, and the lanes' ADDTID reads.
//
// Assembled for gfx90a, gfx942 and gfx950 by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX950 is
// defined, and RET is how the target spells "return the old value": glc on gfx90a, sc0 after it).
#if defined(GFX90A)
#define RET glc
#else
#define RET sc0
#endif
  .text
  .globl ds_gaps
  .p2align 8
  .type ds_gaps,@function
ds_gaps:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = init, s[6:7] = data
  s_load_dwordx2 s[8:9], s[0:1], 0x10         // out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 7, v0                  // its 128 bytes of A (in LDS, in init and in out)
  v_lshlrev_b32_e32 v2, 2, v0                  // its words of B
  v_lshlrev_b32_e32 v3, 3, v0                  // its quadwords of C
  v_add_u32_e32 v7, 8192, v2
  v_add_u32_e32 v8, 8704, v3
  v_mov_b32_e32 v9, v1
  s_add_u32 s16, s4, 0x2000                    // init: B starts at byte 8192
  s_addc_u32 s17, s5, 0
  s_add_u32 s18, s4, 0x2200                    // and C at 8704
  s_addc_u32 s19, s5, 0
  global_load_dword v10, v1, s[4:5] offset:0
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:0
  global_load_dword v10, v1, s[4:5] offset:4
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:4
  global_load_dword v10, v1, s[4:5] offset:8
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:8
  global_load_dword v10, v1, s[4:5] offset:12
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:12
  global_load_dword v10, v1, s[4:5] offset:16
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:16
  global_load_dword v10, v1, s[4:5] offset:20
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:20
  global_load_dword v10, v1, s[4:5] offset:24
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:24
  global_load_dword v10, v1, s[4:5] offset:28
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:28
  global_load_dword v10, v1, s[4:5] offset:32
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:32
  global_load_dword v10, v1, s[4:5] offset:36
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:36
  global_load_dword v10, v1, s[4:5] offset:40
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:40
  global_load_dword v10, v1, s[4:5] offset:44
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:44
  global_load_dword v10, v1, s[4:5] offset:48
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:48
  global_load_dword v10, v1, s[4:5] offset:52
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:52
  global_load_dword v10, v1, s[4:5] offset:56
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:56
  global_load_dword v10, v1, s[4:5] offset:60
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:60
  global_load_dword v10, v1, s[4:5] offset:64
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:64
  global_load_dword v10, v1, s[4:5] offset:68
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:68
  global_load_dword v10, v1, s[4:5] offset:72
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:72
  global_load_dword v10, v1, s[4:5] offset:76
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:76
  global_load_dword v10, v1, s[4:5] offset:80
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:80
  global_load_dword v10, v1, s[4:5] offset:84
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:84
  global_load_dword v10, v1, s[4:5] offset:88
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:88
  global_load_dword v10, v1, s[4:5] offset:92
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:92
  global_load_dword v10, v1, s[4:5] offset:96
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:96
  global_load_dword v10, v1, s[4:5] offset:100
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:100
  global_load_dword v10, v1, s[4:5] offset:104
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:104
  global_load_dword v10, v1, s[4:5] offset:108
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:108
  global_load_dword v10, v1, s[4:5] offset:112
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:112
  global_load_dword v10, v1, s[4:5] offset:116
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:116
  global_load_dword v10, v1, s[4:5] offset:120
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:120
  global_load_dword v10, v1, s[4:5] offset:124
  s_waitcnt vmcnt(0)
  ds_write_b32 v1, v10 offset:124
  global_load_dword v10, v2, s[16:17] offset:0
  s_waitcnt vmcnt(0)
  ds_write_b32 v7, v10 offset:0
  global_load_dword v10, v2, s[16:17] offset:256
  s_waitcnt vmcnt(0)
  ds_write_b32 v7, v10 offset:256
  global_load_dwordx2 v[10:11], v3, s[18:19] offset:0
  s_waitcnt vmcnt(0)
  ds_write_b64 v8, v[10:11] offset:0
  global_load_dwordx2 v[10:11], v3, s[18:19] offset:512
  s_waitcnt vmcnt(0)
  ds_write_b64 v8, v[10:11] offset:512
  v_lshlrev_b32_e32 v4, 6, v0                  // the lane's 16 dwords of operands
  global_load_dwordx4 v[20:23], v4, s[6:7] offset:0
  global_load_dwordx4 v[24:27], v4, s[6:7] offset:16
  global_load_dwordx4 v[28:31], v4, s[6:7] offset:32
  global_load_dwordx4 v[32:35], v4, s[6:7] offset:48
  s_waitcnt vmcnt(0)
  s_waitcnt lgkmcnt(0)
  ds_mskor_rtn_b32 v40, v1, v20, v21 offset:0
  ds_mskor_b32 v1, v22, v23 offset:4
  ds_mskor_rtn_b64 v[42:43], v1, v[24:25], v[26:27] offset:8
  ds_mskor_b64 v1, v[28:29], v[30:31] offset:16
  ds_wrxchg2_rtn_b32 v[44:45], v1, v20, v21 offset0:6 offset1:7
  ds_wrxchg2_rtn_b64 v[48:51], v1, v[24:25], v[26:27] offset0:4 offset1:5
  ds_cmpst_b64 v1, v[28:29], v[30:31] offset:48
  ds_cmpst_f64 v1, v[32:33], v[34:35] offset:56
  ds_cmpst_rtn_f64 v[54:55], v1, v[32:33], v[34:35] offset:64
  ds_wrap_rtn_b32 v52, v1, v22, v23 offset:80
  ds_condxchg32_rtn_b64 v[56:57], v1, v[24:25] offset:88
  ds_nop
  ds_wrxchg2st64_rtn_b32 v[58:59], v7, v20, v21 offset0:0 offset1:1
  ds_wrxchg2st64_rtn_b64 v[60:63], v8, v[24:25], v[26:27] offset0:0 offset1:1
  s_waitcnt lgkmcnt(0)
  v_mul_u32_u24_e32 v5, 96, v0
  global_store_dword v5, v40, s[8:9] offset:0
  global_store_dword v5, v42, s[8:9] offset:4
  global_store_dword v5, v43, s[8:9] offset:8
  global_store_dword v5, v44, s[8:9] offset:12
  global_store_dword v5, v45, s[8:9] offset:16
  global_store_dword v5, v48, s[8:9] offset:20
  global_store_dword v5, v49, s[8:9] offset:24
  global_store_dword v5, v50, s[8:9] offset:28
  global_store_dword v5, v51, s[8:9] offset:32
  global_store_dword v5, v52, s[8:9] offset:36
  global_store_dword v5, v54, s[8:9] offset:40
  global_store_dword v5, v55, s[8:9] offset:44
  global_store_dword v5, v56, s[8:9] offset:48
  global_store_dword v5, v57, s[8:9] offset:52
  global_store_dword v5, v58, s[8:9] offset:56
  global_store_dword v5, v59, s[8:9] offset:60
  global_store_dword v5, v60, s[8:9] offset:64
  global_store_dword v5, v61, s[8:9] offset:68
  global_store_dword v5, v62, s[8:9] offset:72
  global_store_dword v5, v63, s[8:9] offset:76
  s_add_u32 s20, s8, 6144                      // A's final contents
  s_addc_u32 s21, s9, 0
  ds_read_b32 v10, v1 offset:0
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:0
  ds_read_b32 v10, v1 offset:4
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:4
  ds_read_b32 v10, v1 offset:8
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:8
  ds_read_b32 v10, v1 offset:12
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:12
  ds_read_b32 v10, v1 offset:16
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:16
  ds_read_b32 v10, v1 offset:20
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:20
  ds_read_b32 v10, v1 offset:24
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:24
  ds_read_b32 v10, v1 offset:28
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:28
  ds_read_b32 v10, v1 offset:32
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:32
  ds_read_b32 v10, v1 offset:36
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:36
  ds_read_b32 v10, v1 offset:40
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:40
  ds_read_b32 v10, v1 offset:44
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:44
  ds_read_b32 v10, v1 offset:48
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:48
  ds_read_b32 v10, v1 offset:52
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:52
  ds_read_b32 v10, v1 offset:56
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:56
  ds_read_b32 v10, v1 offset:60
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:60
  ds_read_b32 v10, v1 offset:64
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:64
  ds_read_b32 v10, v1 offset:68
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:68
  ds_read_b32 v10, v1 offset:72
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:72
  ds_read_b32 v10, v1 offset:76
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:76
  ds_read_b32 v10, v1 offset:80
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:80
  ds_read_b32 v10, v1 offset:84
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:84
  ds_read_b32 v10, v1 offset:88
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:88
  ds_read_b32 v10, v1 offset:92
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:92
  ds_read_b32 v10, v1 offset:96
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:96
  ds_read_b32 v10, v1 offset:100
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:100
  ds_read_b32 v10, v1 offset:104
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:104
  ds_read_b32 v10, v1 offset:108
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:108
  ds_read_b32 v10, v1 offset:112
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:112
  ds_read_b32 v10, v1 offset:116
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:116
  ds_read_b32 v10, v1 offset:120
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:120
  ds_read_b32 v10, v1 offset:124
  s_waitcnt lgkmcnt(0)
  global_store_dword v9, v10, s[20:21] offset:124
  s_add_u32 s22, s8, 14336                     // B's
  s_addc_u32 s23, s9, 0
  ds_read_b32 v10, v7 offset:0
  s_waitcnt lgkmcnt(0)
  global_store_dword v2, v10, s[22:23] offset:0
  ds_read_b32 v10, v7 offset:256
  s_waitcnt lgkmcnt(0)
  global_store_dword v2, v10, s[22:23] offset:256
  s_add_u32 s24, s8, 14848                     // C's
  s_addc_u32 s25, s9, 0
  ds_read_b64 v[10:11], v8 offset:0
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v3, v[10:11], s[24:25] offset:0
  ds_read_b64 v[10:11], v8 offset:512
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v3, v[10:11], s[24:25] offset:512
  s_add_u32 s26, s8, 15872                     // what ds_append and ds_consume gave back, three dwords a lane
  s_addc_u32 s27, s9, 0
  v_mov_b32_e32 v11, 10240
  v_mov_b32_e32 v10, 1000
  ds_write_b32 v11, v10
  s_waitcnt lgkmcnt(0)
  s_mov_b32 exec_lo, -1                        // 48 lanes
  s_mov_b32 exec_hi, 0xffff
  ds_append v64 offset:10240
  ds_consume v65 offset:10240
  ds_consume v66 offset:10240
  s_waitcnt lgkmcnt(0)
  v_mul_u32_u24_e32 v12, 12, v0
  global_store_dword v12, v64, s[26:27] offset:0
  global_store_dword v12, v65, s[26:27] offset:4
  global_store_dword v12, v66, s[26:27] offset:8
  s_mov_b64 exec, -1
  ds_read_b32 v10, v11
  s_waitcnt lgkmcnt(0)
  s_add_u32 s28, s8, 16896
  s_addc_u32 s29, s9, 0
  v_mov_b32_e32 v13, 0
  s_mov_b64 exec, 1
  global_store_dword v13, v10, s[28:29]             // the counter's final value
  s_mov_b64 exec, -1
  s_mov_b32 m0, 0x33100                        // M0[15:0] = 0x3100
  ds_write_addtid_b32 v22 offset:16
  s_waitcnt lgkmcnt(0)
  ds_read_addtid_b32 v67 offset:16
  s_waitcnt lgkmcnt(0)
  s_add_u32 s30, s8, 16640
  s_addc_u32 s31, s9, 0
  global_store_dword v2, v67, s[30:31]
  s_waitcnt vmcnt(0)
  s_endpgm
.Lds_gaps_end:
  .size ds_gaps, .Lds_gaps_end-ds_gaps
  .rodata
  .p2align 6
  .amdhsa_kernel ds_gaps
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 72
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 72
  .end_amdhsa_kernel
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: ds_gaps
    .symbol: ds_gaps.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 72
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata

