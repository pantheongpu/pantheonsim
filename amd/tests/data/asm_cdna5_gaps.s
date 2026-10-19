// gfx1250's (CDNA 5) instructions the executor refused and the CDNA 5 ISA document defines: the conditional subtract and the subtract that
// clamps at zero on global memory and through a buffer ("MEM[calcAddr].u32 = tmp >= src ? tmp - src : tmp" and "if old_value < DATA.u32 then
// new_value = 0U else new_value = old_value - DATA.u32"), the global loads and stores that address by the lane's number (GLOBAL_LOAD_ADDTID_B32:
// "SGPR_ADDR + INST_OFFSET + laneID * 4"), S_MOVRELSD_2_B32 (the document's own example), S_ROUND_MODE and S_DENORM_MODE ("Set the round mode from
// an immediate: SIMM16[3:0]"; MODE's FP_ROUND is bits 3:0, FP_DENORM 7:4) and S_TTRACEDATA_IMM.
//
// Assembled for gfx1250 by build.sh, with clang's assembler.
//
// cdna5_gaps(buf, out): buf holds 192 dwords: the words cond_sub is applied to (0..31), the data (32..63), what sub_clamp and the buffer's
// cond_sub are applied to (64..95, 96..127), what the addtid load reads (128..159) and what the addtid store writes (160..191). out: the three
// old values (96 dwords), the addtid load (32), and four dwords that scalar code reads out of MODE and from s_movrelsd_2_b32.
  .text
  .globl cdna5_gaps
  .p2align 8
  .type cdna5_gaps,@function
cdna5_gaps:
  s_load_b128 s[4:7], s[0:1], 0x0              // s[4:5] = buf, s[6:7] = out
  s_wait_kmcnt 0
  v_and_b32_e32 v0, 31, v0                     // the lane
  v_lshlrev_b32_e32 v1, 2, v0                  // its dword
  global_load_b32 v3, v1, s[4:5] offset:128    // the data: buf[32 + lane]
  s_wait_loadcnt 0
  global_atomic_cond_sub_u32 v10, v1, v3, s[4:5] th:TH_ATOMIC_RETURN             // buf[lane]
  global_atomic_sub_clamp_u32 v11, v1, v3, s[4:5] offset:256 th:TH_ATOMIC_RETURN  // buf[64 + lane]
  s_mov_b32 s8, s4
  s_mov_b32 s9, s5
  s_mov_b32 s10, 0x1000
  s_mov_b32 s11, 0x31014000                    // a raw buffer with a format (what the compiler hands out)
  buffer_atomic_cond_sub_u32 v3, v1, s[8:11], null offen offset:384 th:TH_ATOMIC_RETURN   // buf[96 + lane]
  global_load_addtid_b32 v20, s[4:5] offset:512                                  // buf[128 + lane]
  v_mul_u32_u24_e32 v21, 3, v0
  v_add_nc_u32_e32 v21, 7, v21
  global_store_addtid_b32 v21, s[4:5] offset:640                                 // buf[160 + lane] = 3 lane + 7
  s_wait_loadcnt 0
  global_store_b32 v1, v10, s[6:7]
  global_store_b32 v1, v11, s[6:7] offset:128
  global_store_b32 v1, v3, s[6:7] offset:256
  global_store_b32 v1, v20, s[6:7] offset:384
  s_mov_b32 s17, 0xabcd1234
  s_mov_b32 s25, 0
  s_mov_b32 m0, 0x4014000a                    // (20 << 16) | 10, and a bit that is no part of either field
  s_movrelsd_2_b32 s5, s7                      // s25 = s17: the document's own example
  s_getreg_b32 s30, hwreg(HW_REG_WAVE_MODE, 0, 8)
  s_round_mode 0x5
  s_denorm_mode 0xa
  s_ttracedata_imm 0x12
  s_getreg_b32 s31, hwreg(HW_REG_WAVE_MODE, 0, 8)
  s_round_mode 0
  s_denorm_mode 0xf
  s_getreg_b32 s32, hwreg(HW_REG_WAVE_MODE, 0, 8)
  v_mov_b32_e32 v23, 0
  v_mov_b32_e32 v22, s25
  global_store_b32 v23, v22, s[6:7] offset:512
  v_mov_b32_e32 v22, s31
  global_store_b32 v23, v22, s[6:7] offset:516
  v_mov_b32_e32 v22, s30
  global_store_b32 v23, v22, s[6:7] offset:520
  v_mov_b32_e32 v22, s32
  global_store_b32 v23, v22, s[6:7] offset:524
  s_wait_storecnt 0
  s_endpgm
.Lcdna5_gaps_end:
  .size cdna5_gaps, .Lcdna5_gaps_end-cdna5_gaps
  .rodata
  .p2align 6
  .amdhsa_kernel cdna5_gaps
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 40
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: cdna5_gaps
    .symbol: cdna5_gaps.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 40
    .vgpr_count: 32
    .max_flat_workgroup_size: 32
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
