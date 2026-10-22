// Matrix multiply-accumulate instructions on binary16 inputs and a non-zero accumulator, for a test whose data makes the
// order the products are summed in show in the result (test_gcn_asm): v_mfma_f32_4x4x4_16b_f16, v_mfma_f32_16x16x16_f16
// and v_mfma_f32_32x32x8_f16. Each is mfmaN(a, b, c, out): a lane loads eight bytes of A and of B (four halves) and its
// accumulator registers from c, and stores the result registers to out. Built for gfx942 by build.sh, with clang's
// assembler.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl mfma4
  .p2align 8
  .type mfma4,@function
mfma4:
  s_load_dwordx8 s[4:11], s[0:1], 0x0         // a, b, c, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 3, v0                  // 8 bytes a lane
  global_load_dwordx2 v[2:3], v1, s[4:5]
  global_load_dwordx2 v[4:5], v1, s[6:7]
  v_lshlrev_b32_e32 v10, 4, v0                 // 16 bytes a lane
  global_load_dwordx4 v[16:19], v10, s[8:9] offset:0
  s_waitcnt vmcnt(0)
  v_mfma_f32_4x4x4_16b_f16 v[32:35], v[2:3], v[4:5], v[16:19]
  s_nop 7
  s_nop 7
  s_nop 7
  s_nop 7
  global_store_dwordx4 v10, v[32:35], s[10:11] offset:0
  s_endpgm
.Lmfma4_end:
  .size mfma4, .Lmfma4_end-mfma4

  .globl mfma16
  .p2align 8
  .type mfma16,@function
mfma16:
  s_load_dwordx8 s[4:11], s[0:1], 0x0         // a, b, c, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 3, v0                  // 8 bytes a lane
  global_load_dwordx2 v[2:3], v1, s[4:5]
  global_load_dwordx2 v[4:5], v1, s[6:7]
  v_lshlrev_b32_e32 v10, 4, v0                 // 16 bytes a lane
  global_load_dwordx4 v[16:19], v10, s[8:9] offset:0
  s_waitcnt vmcnt(0)
  v_mfma_f32_16x16x16_f16 v[32:35], v[2:3], v[4:5], v[16:19]
  s_nop 7
  s_nop 7
  s_nop 7
  s_nop 7
  global_store_dwordx4 v10, v[32:35], s[10:11] offset:0
  s_endpgm
.Lmfma16_end:
  .size mfma16, .Lmfma16_end-mfma16

  .globl mfma32
  .p2align 8
  .type mfma32,@function
mfma32:
  s_load_dwordx8 s[4:11], s[0:1], 0x0         // a, b, c, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 3, v0                  // 8 bytes a lane
  global_load_dwordx2 v[2:3], v1, s[4:5]
  global_load_dwordx2 v[4:5], v1, s[6:7]
  v_lshlrev_b32_e32 v10, 6, v0                 // 64 bytes a lane
  global_load_dwordx4 v[16:19], v10, s[8:9] offset:0
  global_load_dwordx4 v[20:23], v10, s[8:9] offset:16
  global_load_dwordx4 v[24:27], v10, s[8:9] offset:32
  global_load_dwordx4 v[28:31], v10, s[8:9] offset:48
  s_waitcnt vmcnt(0)
  v_mfma_f32_32x32x8_f16 v[32:47], v[2:3], v[4:5], v[16:31]
  s_nop 7
  s_nop 7
  s_nop 7
  s_nop 7
  global_store_dwordx4 v10, v[32:35], s[10:11] offset:0
  global_store_dwordx4 v10, v[36:39], s[10:11] offset:16
  global_store_dwordx4 v10, v[40:43], s[10:11] offset:32
  global_store_dwordx4 v10, v[44:47], s[10:11] offset:48
  s_endpgm
.Lmfma32_end:
  .size mfma32, .Lmfma32_end-mfma32

  .rodata
  .p2align 6
  .amdhsa_kernel mfma4
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 38
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 40
  .end_amdhsa_kernel

  .amdhsa_kernel mfma16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 38
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 40
  .end_amdhsa_kernel

  .amdhsa_kernel mfma32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 50
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 52
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: mfma4
    .symbol: mfma4.kd
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 38
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
  - .name: mfma16
    .symbol: mfma16.kd
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 38
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
  - .name: mfma32
    .symbol: mfma32.kd
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 50
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
