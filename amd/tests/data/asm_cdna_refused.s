// What the CDNA guides leave open and the executor therefore still refuses, so that a change which starts guessing shows up: setting the HALT
// bit (nothing here resumes a halted wave; clearing it is a no-op), s_sendmsghalt, an M0 of 64 KiB or more for gfx950's MUBUF load into LDS (CDNA4's
// text says M0[17:0] and its figure M0[15:0]) and gfx950's 6-bit transposing read (the guide gives only "Two instruction load a complete matrix").
//
// Assembled for gfx90a, gfx942 and gfx950 by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX950 is
// defined, and RET is how the target spells "return the old value": glc on gfx90a, sc0 after it).
#if defined(GFX90A)
#define RET glc
#else
#define RET sc0
#endif
  .text
  .globl halt_set
  .p2align 8
  .type halt_set,@function
halt_set:
  s_sethalt 1
  s_endpgm
.Lhalt_set_end:
  .size halt_set, .Lhalt_set_end-halt_set
  .globl halt_clear
  .p2align 8
  .type halt_clear,@function
halt_clear:
  s_sethalt 0
  s_endpgm
.Lhalt_clear_end:
  .size halt_clear, .Lhalt_clear_end-halt_clear
  .globl msg_halt
  .p2align 8
  .type msg_halt,@function
msg_halt:
  s_sendmsghalt sendmsg(MSG_INTERRUPT)
  s_endpgm
.Lmsg_halt_end:
  .size msg_halt, .Lmsg_halt_end-msg_halt
#if defined(GFX950)
  .globl m0_high_buffer_lds
  .p2align 8
  .type m0_high_buffer_lds,@function
m0_high_buffer_lds:
  s_mov_b32 s8, 0
  s_mov_b32 s9, 0
  s_mov_b32 s10, 64
  s_mov_b32 s11, 0x27000
  v_mov_b32_e32 v1, 0
  s_mov_b32 m0, 0x10000
  buffer_load_dword v1, s[8:11], 0 offen lds
  s_endpgm
.Lm0_high_buffer_lds_end:
  .size m0_high_buffer_lds, .Lm0_high_buffer_lds_end-m0_high_buffer_lds
#endif
#if defined(GFX950)
  .globl b6_transpose
  .p2align 8
  .type b6_transpose,@function
b6_transpose:
  v_mov_b32_e32 v0, 0
  ds_read_b96_tr_b6 v[2:4], v0
  s_endpgm
.Lb6_transpose_end:
  .size b6_transpose, .Lb6_transpose_end-b6_transpose
#endif
  .rodata
  .p2align 6
  .amdhsa_kernel halt_set
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel halt_clear
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel msg_halt
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
#if defined(GFX950)
  .p2align 6
  .amdhsa_kernel m0_high_buffer_lds
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
#endif
#if defined(GFX950)
  .p2align 6
  .amdhsa_kernel b6_transpose
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
#endif
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: halt_set
    .symbol: halt_set.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: halt_clear
    .symbol: halt_clear.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: msg_halt
    .symbol: msg_halt.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
#if defined(GFX950)
  - .name: m0_high_buffer_lds
    .symbol: m0_high_buffer_lds.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
#endif
#if defined(GFX950)
  - .name: b6_transpose
    .symbol: b6_transpose.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
#endif
...
  .end_amdgpu_metadata

