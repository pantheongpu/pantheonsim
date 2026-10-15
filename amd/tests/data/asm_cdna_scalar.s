// Scalar memory and control instructions the CDNA ISA guides define: the scalar stores and atomics through a buffer resource
// (S_BUFFER_STORE_DWORD*, S_BUFFER_ATOMIC_* and _X2: "addr = CalcScalarBufferAddr(...); tmp = MEM[addr].u32; ... RETURN_DATA.u32 = tmp"), the
// cache instructions that change nothing a program reads, vector GPR indexing's S_SET_GPR_IDX_IDX ("M0[7:0] = S0.u32[7:0]") and
// S_SET_GPR_IDX_MODE ("M0[15:12] = SIMM16.u16[3:0]"), S_SETVSKIP ("no VOP*/M*BUF/MIMG/DS/FLAT instructions are issued"), S_SETREG_B32, the
// conditional branches on the debug flags (not taken: nothing sets them), the performance and trace instructions, and S_SETKILL.
//
// scalar_gaps(buf, out): the atomics' returned values and the rest are stored to out by one lane (see the test for what each word is).
//
// Assembled for gfx90a, gfx942 and gfx950 by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX950 is
// defined, and RET is how the target spells "return the old value": glc on gfx90a, sc0 after it).
#if defined(GFX90A)
#define RET glc
#else
#define RET sc0
#endif
  .text
  .globl scalar_gaps
  .p2align 8
  .type scalar_gaps,@function
scalar_gaps:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = buf, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1                            // one lane stores what the scalar side finds
  v_mov_b32_e32 v0, 0
  s_mov_b32 s8, s4
  s_mov_b32 s9, s5                             // the resource: base, stride 0, 512 bytes
  s_mov_b32 s10, 512
  s_mov_b32 s11, 0x27000
  s_mov_b32 s20, 0x11111111
  s_buffer_store_dword s20, s[8:11], 0x0
  s_mov_b32 s22, 0x22222222
  s_mov_b32 s23, 0x33333333
  s_buffer_store_dwordx2 s[22:23], s[8:11], 0x8
  s_mov_b32 s24, 0x44444444
  s_mov_b32 s25, 0x55555555
  s_mov_b32 s26, 0x66666666
  s_mov_b32 s27, 0x77777777
  s_buffer_store_dwordx4 s[24:27], s[8:11], 0x10
  s_mov_b32 s60, 0x13572468
  s_buffer_atomic_swap s60, s[8:11], 0x40 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:0
  s_mov_b32 s60, 0x1000
  s_buffer_atomic_add s60, s[8:11], 0x44 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:4
  s_mov_b32 s60, 0x234
  s_buffer_atomic_sub s60, s[8:11], 0x48 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:8
  s_mov_b32 s60, 0x80000005
  s_buffer_atomic_smin s60, s[8:11], 0x4c glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:12
  s_mov_b32 s60, 0x7
  s_buffer_atomic_umin s60, s[8:11], 0x50 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:16
  s_mov_b32 s60, 0xfffffff0
  s_buffer_atomic_smax s60, s[8:11], 0x54 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:20
  s_mov_b32 s60, 0x9
  s_buffer_atomic_umax s60, s[8:11], 0x58 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:24
  s_mov_b32 s60, 0xff00ff00
  s_buffer_atomic_and s60, s[8:11], 0x5c glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:28
  s_mov_b32 s60, 0xf0f0000
  s_buffer_atomic_or s60, s[8:11], 0x60 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:32
  s_mov_b32 s60, 0xaaaa5555
  s_buffer_atomic_xor s60, s[8:11], 0x64 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:36
  s_mov_b32 s60, 0x5
  s_buffer_atomic_inc s60, s[8:11], 0x68 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:40
  s_mov_b32 s60, 0x64
  s_buffer_atomic_dec s60, s[8:11], 0x6c glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:44
  s_mov_b32 s60, 0xcafebabe
  s_mov_b32 s61, 0x24681357
  s_buffer_atomic_cmpswap s[60:61], s[8:11], 0x70 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:48
  s_mov_b32 s60, 0x55667788
  s_mov_b32 s61, 0x11223344
  s_buffer_atomic_swap_x2 s[60:61], s[8:11], 0x80 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:52
  v_mov_b32_e32 v1, s61
  global_store_dword v0, v1, s[6:7] offset:56
  s_mov_b32 s60, 0x1
  s_mov_b32 s61, 0x1
  s_buffer_atomic_add_x2 s[60:61], s[8:11], 0x88 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:60
  v_mov_b32_e32 v1, s61
  global_store_dword v0, v1, s[6:7] offset:64
  s_mov_b32 s60, 0xff00ff00
  s_mov_b32 s61, 0xff00ff00
  s_buffer_atomic_and_x2 s[60:61], s[8:11], 0x90 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:68
  v_mov_b32_e32 v1, s61
  global_store_dword v0, v1, s[6:7] offset:72
  s_mov_b32 s60, 0xdeadbeef
  s_mov_b32 s61, 0xdeadbeef
  s_mov_b32 s62, 0x5060708
  s_mov_b32 s63, 0x1020304
  s_buffer_atomic_cmpswap_x2 s[60:63], s[8:11], 0x98 glc
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s60
  global_store_dword v0, v1, s[6:7] offset:76
  v_mov_b32_e32 v1, s61
  global_store_dword v0, v1, s[6:7] offset:80
  s_dcache_discard s[4:5], 0x0
  s_dcache_discard_x2 s[4:5], 0x40
  s_dcache_inv_vol
  s_dcache_wb_vol
#if defined(GFX90A)
  s_atc_probe 0x0, s[4:5], 0x0
  s_atc_probe_buffer 0x0, s[8:11], 0x0
#endif
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v4, 111
  v_mov_b32_e32 v5, 222
  v_mov_b32_e32 v6, 333
  v_mov_b32_e32 v12, 0xaaaa
  v_mov_b32_e32 v14, 0xbbbb
  s_mov_b32 s40, 1
  s_set_gpr_idx_on s40, gpr_idx(SRC0)
  v_mov_b32_e32 v10, v4                        // reads v5
  s_mov_b32 s41, 2
  s_set_gpr_idx_idx s41
  v_mov_b32_e32 v11, v4                        // reads v6
  s_set_gpr_idx_mode gpr_idx(DST)
  v_mov_b32_e32 v12, v4                        // writes v14, from v4
  s_set_gpr_idx_off
  s_set_gpr_idx_mode gpr_idx(SRC0)             // (indexing is off: the mode alone changes nothing)
  v_mov_b32_e32 v13, v4
  global_store_dword v0, v10, s[6:7] offset:84
  global_store_dword v0, v11, s[6:7] offset:88
  global_store_dword v0, v12, s[6:7] offset:92
  global_store_dword v0, v14, s[6:7] offset:96
  global_store_dword v0, v13, s[6:7] offset:100
  v_mov_b32_e32 v20, 3
  v_mov_b32_e32 v21, 3
  v_mov_b32_e32 v22, 0xdead
  v_mov_b32_e32 v23, 0xdead
  s_setvskip 1, 0
  v_mov_b32_e32 v20, 7                         // skipped
  s_mov_b32 s50, 5                             // not
  global_store_dword v0, v22, s[6:7] offset:104    // skipped: the word keeps what it had
  ds_write_b32 v0, v22                          // skipped
  s_setvskip 0, 0
  v_mov_b32_e32 v21, 9
  v_mov_b32_e32 v24, s50
  global_store_dword v0, v20, s[6:7] offset:108
  global_store_dword v0, v21, s[6:7] offset:112
  global_store_dword v0, v24, s[6:7] offset:116
  s_getreg_b32 s53, hwreg(HW_REG_MODE, 0, 4)
  s_mov_b32 s51, 0xd
  s_setreg_b32 hwreg(HW_REG_MODE, 0, 4), s51
  s_getreg_b32 s52, hwreg(HW_REG_MODE, 0, 4)
  s_setreg_b32 hwreg(HW_REG_MODE, 0, 4), s53
  v_mov_b32_e32 v1, s52
  global_store_dword v0, v1, s[6:7] offset:120
  s_mov_b32 s54, 0
  s_cbranch_cdbgsys .Lskip0
  s_add_u32 s54, s54, 1
.Lskip0:
  s_cbranch_cdbguser .Lskip1
  s_add_u32 s54, s54, 1
.Lskip1:
  s_cbranch_cdbgsys_or_user .Lskip2
  s_add_u32 s54, s54, 1
.Lskip2:
  s_cbranch_cdbgsys_and_user .Lskip3
  s_add_u32 s54, s54, 1
.Lskip3:
  v_mov_b32_e32 v1, s54
  global_store_dword v0, v1, s[6:7] offset:124
  s_mov_b32 s55, 0
  s_incperflevel 3
  s_decperflevel 3
  s_mov_b32 m0, 5
  s_ttracedata
  s_sethalt 0
  s_setkill 0
  v_clrexcp
  s_add_u32 s55, s55, 1
  v_mov_b32_e32 v1, s55
  global_store_dword v0, v1, s[6:7] offset:128
  s_mov_b32 s56, 0x11
  v_mov_b32_e32 v1, s56
  global_store_dword v0, v1, s[6:7] offset:132
  s_waitcnt vmcnt(0)
  s_setkill 1
  s_mov_b32 s56, 0x22
  v_mov_b32_e32 v1, s56
  global_store_dword v0, v1, s[6:7] offset:136
  s_waitcnt vmcnt(0)
  s_endpgm
.Lscalar_gaps_end:
  .size scalar_gaps, .Lscalar_gaps_end-scalar_gaps
  .rodata
  .p2align 6
  .amdhsa_kernel scalar_gaps
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 256
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 32
  .end_amdhsa_kernel
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: scalar_gaps
    .symbol: scalar_gaps.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 32
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata

