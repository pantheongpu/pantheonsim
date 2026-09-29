// RDNA's v_permlanex16_b32, whose op_sel is FI (read a source lane even when
// it is off) and BOUND_CTRL (zero when it is off and FI is not set), not a
// choice of halves: Triton's row maximum across a wave32 (vLLM's attention)
// sets FI, and the whole 32 bits must move. Built for gfx1100 by build.sh,
// with clang's assembler.
//
// permlane(uint* out): each lane holds 0xff800000 | its lane number, and
// reads the other row's lane of the same number (selector 0xfedcba98:76543210);
// out gets four arrays of 64 words: FI with every lane on; then, with only
// lanes 0-15 on (their sources are off), no bits, FI, and BOUND_CTRL, each
// into a register holding 7.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .text
  .globl permlane
  .p2align 8
  .type permlane,@function
permlane:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v10, 31, v0
  v_or_b32_e32 v1, 0xff800000, v10
  s_mov_b32 s4, 0x76543210
  s_mov_b32 s5, 0xfedcba98
  v_permlanex16_b32 v2, v1, s4, s5 op_sel:[1,0]
  v_mov_b32_e32 v3, 7
  v_mov_b32_e32 v4, 7
  v_mov_b32_e32 v5, 7
  s_mov_b32 exec_lo, 0xffff
  v_permlanex16_b32 v3, v1, s4, s5
  v_permlanex16_b32 v4, v1, s4, s5 op_sel:[1,0]
  v_permlanex16_b32 v5, v1, s4, s5 op_sel:[0,1]
  s_mov_b32 exec_lo, -1
  v_lshlrev_b32_e32 v6, 2, v0
  global_store_b32 v6, v2, s[2:3]
  global_store_b32 v6, v3, s[2:3] offset:256
  global_store_b32 v6, v4, s[2:3] offset:512
  global_store_b32 v6, v5, s[2:3] offset:768
  s_endpgm
.Lpermlane_end:
  .size permlane, .Lpermlane_end-permlane

  .rodata
  .p2align 6
  .amdhsa_kernel permlane
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 11
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: permlane
    .symbol: permlane.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 11
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
