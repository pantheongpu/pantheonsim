// What the CDNA guides leave open and the executor therefore still refuses, so that a change which starts guessing shows up: setting the HALT
// bit (nothing here resumes a halted wave; clearing it is a no-op), s_sendmsghalt, an M0 of 64 KiB or more for gfx950's MUBUF load into LDS (CDNA4's
// text says M0[17:0] and its figure M0[15:0]), gfx950's 6-bit transposing read (the guide gives only "Two instruction load a complete matrix"), its stochastic-rounding
// conversions (the guide and AMD's software disagree on which bits of the seed are added), and the corners of its scaled narrowing conversions that
// public sources disagree on: a NaN into a 4- or 6-bit float, a scale of 255, an input modifier.
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
#if defined(GFX950)
  .globl sr_fp8_f32
  .p2align 8
  .type sr_fp8_f32,@function
sr_fp8_f32:
  v_cvt_scalef32_sr_fp8_f32 v1, v2, v3, v4
  s_endpgm
.Lsr_fp8_f32_end:
  .size sr_fp8_f32, .Lsr_fp8_f32_end-sr_fp8_f32
  .globl sr_bf8_f16
  .p2align 8
  .type sr_bf8_f16,@function
sr_bf8_f16:
  v_cvt_scalef32_sr_bf8_f16 v1, v2, v3, v4
  s_endpgm
.Lsr_bf8_f16_end:
  .size sr_bf8_f16, .Lsr_bf8_f16_end-sr_bf8_f16
  .globl sr_pk_fp4_f32
  .p2align 8
  .type sr_pk_fp4_f32,@function
sr_pk_fp4_f32:
  v_cvt_scalef32_sr_pk_fp4_f32 v1, v[2:3], v4, v5
  s_endpgm
.Lsr_pk_fp4_f32_end:
  .size sr_pk_fp4_f32, .Lsr_pk_fp4_f32_end-sr_pk_fp4_f32
  .globl sr_pk32_bf6_bf16
  .p2align 8
  .type sr_pk32_bf6_bf16,@function
sr_pk32_bf6_bf16:
  v_cvt_scalef32_sr_pk32_bf6_bf16 v[0:5], v[8:23], v24, v25
  s_endpgm
.Lsr_pk32_bf6_bf16_end:
  .size sr_pk32_bf6_bf16, .Lsr_pk32_bf6_bf16_end-sr_pk32_bf6_bf16
  .globl cvt_sr_f16_f32
  .p2align 8
  .type cvt_sr_f16_f32,@function
cvt_sr_f16_f32:
  v_cvt_sr_f16_f32 v1, v2, v3
  s_endpgm
.Lcvt_sr_f16_f32_end:
  .size cvt_sr_f16_f32, .Lcvt_sr_f16_f32_end-cvt_sr_f16_f32
  .globl cvt_sr_bf16_f32
  .p2align 8
  .type cvt_sr_bf16_f32,@function
cvt_sr_bf16_f32:
  v_cvt_sr_bf16_f32 v1, v2, v3
  s_endpgm
.Lcvt_sr_bf16_f32_end:
  .size cvt_sr_bf16_f32, .Lcvt_sr_bf16_f32_end-cvt_sr_bf16_f32
  .globl nan_into_fp4
  .p2align 8
  .type nan_into_fp4,@function
nan_into_fp4:
  v_mov_b32 v2, 0x7fc00000
  v_cvt_scalef32_pk_fp4_f32 v1, v2, v3, 1.0
  s_endpgm
.Lnan_into_fp4_end:
  .size nan_into_fp4, .Lnan_into_fp4_end-nan_into_fp4
  .globl nan_into_fp6
  .p2align 8
  .type nan_into_fp6,@function
nan_into_fp6:
  v_mov_b32 v8, 0x7e00
  v_mov_b32 v24, 1.0
  v_cvt_scalef32_pk32_fp6_f16 v[0:5], v[8:23], v24
  s_endpgm
.Lnan_into_fp6_end:
  .size nan_into_fp6, .Lnan_into_fp6_end-nan_into_fp6
  .globl scale_255
  .p2align 8
  .type scale_255,@function
scale_255:
  v_mov_b32 v3, 0x7f800000
  v_cvt_scalef32_pk_fp8_f32 v1, v2, v2, v3
  s_endpgm
.Lscale_255_end:
  .size scale_255, .Lscale_255_end-scale_255
  .globl narrow_modifier
  .p2align 8
  .type narrow_modifier,@function
narrow_modifier:
  v_cvt_scalef32_pk_fp8_f32 v1, -v2, v3, v4
  s_endpgm
.Lnarrow_modifier_end:
  .size narrow_modifier, .Lnarrow_modifier_end-narrow_modifier
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
#if defined(GFX950)
  .p2align 6
  .amdhsa_kernel sr_fp8_f32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel sr_bf8_f16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel sr_pk_fp4_f32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel sr_pk32_bf6_bf16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel cvt_sr_f16_f32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel cvt_sr_bf16_f32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel nan_into_fp4
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel nan_into_fp6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel scale_255
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 64
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel narrow_modifier
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
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
#if defined(GFX950)
  - .name: sr_fp8_f32
    .symbol: sr_fp8_f32.kd
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
  - .name: sr_bf8_f16
    .symbol: sr_bf8_f16.kd
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
  - .name: sr_pk_fp4_f32
    .symbol: sr_pk_fp4_f32.kd
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
  - .name: sr_pk32_bf6_bf16
    .symbol: sr_pk32_bf6_bf16.kd
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
  - .name: cvt_sr_f16_f32
    .symbol: cvt_sr_f16_f32.kd
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
  - .name: cvt_sr_bf16_f32
    .symbol: cvt_sr_bf16_f32.kd
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
  - .name: nan_into_fp4
    .symbol: nan_into_fp4.kd
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
  - .name: nan_into_fp6
    .symbol: nan_into_fp6.kd
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
  - .name: scale_255
    .symbol: scale_255.kd
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
  - .name: narrow_modifier
    .symbol: narrow_modifier.kd
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
#endif
...
  .end_amdgpu_metadata

