// A matrix instruction of sixteen 4x4 blocks broadcasting A (CBSZ, ABID):
// each group of 2^CBSZ blocks multiplies with block ABID's A. vLLM's paged
// attention for ROCm runs v_mfma_f32_4x4x4_16b_f16 with CBSZ 4, every block
// taking block 0's query. Built for gfx942 by build.sh, with clang's
// assembler.
//
// bcast(a, a4, a2, b, out): each lane loads two words of each input (four
// halves) and runs four products, each written to out as four floats a lane,
// 256 floats apart: a broadcast with CBSZ 4 and ABID 0; with CBSZ 2 and ABID
// 1; and, without broadcast, a4 and a2 -- which the test fills with what the
// broadcasts should have taken -- so each pair must agree.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl bcast
  .p2align 8
  .type bcast,@function
bcast:
  s_load_dwordx8 s[4:11], s[0:1], 0x0         // a, a4, a2, b
  s_load_dwordx2 s[12:13], s[0:1], 0x20       // out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 3, v0
  global_load_dwordx2 v[2:3], v1, s[4:5]
  global_load_dwordx2 v[4:5], v1, s[6:7]
  global_load_dwordx2 v[6:7], v1, s[8:9]
  global_load_dwordx2 v[8:9], v1, s[10:11]
  s_waitcnt vmcnt(0)
  v_mfma_f32_4x4x4_16b_f16 v[14:17], v[2:3], v[8:9], 0 cbsz:4 abid:0
  v_mfma_f32_4x4x4_16b_f16 v[18:21], v[2:3], v[8:9], 0 cbsz:2 abid:1
  v_mfma_f32_4x4x4_16b_f16 v[22:25], v[4:5], v[8:9], 0
  v_mfma_f32_4x4x4_16b_f16 v[26:29], v[6:7], v[8:9], 0
  s_nop 7
  s_nop 7
  v_lshlrev_b32_e32 v1, 4, v0
  global_store_dwordx4 v1, v[14:17], s[12:13]
  global_store_dwordx4 v1, v[18:21], s[12:13] offset:1024
  global_store_dwordx4 v1, v[22:25], s[12:13] offset:2048
  global_store_dwordx4 v1, v[26:29], s[12:13] offset:3072
  s_endpgm
.Lbcast_end:
  .size bcast, .Lbcast_end-bcast

  .rodata
  .p2align 6
  .amdhsa_kernel bcast
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 30
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 32
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: bcast
    .symbol: bcast.kd
    .kernarg_segment_size: 40
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 30
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
