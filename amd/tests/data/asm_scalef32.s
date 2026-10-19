// gfx950's scaled conversions of two 4- or 8-bit floats to bfloat16 (CDNA4 ISA,
// V_CVT_SCALEF32_PK_BF16_FP4, _FP8 and _BF8): "scale = exponent(S1.f32)", the
// byte (FP4: OPSEL[1:0]) or 16-bit half (FP8, BF8: OPSEL[0]) of SRC0 to
// convert, low part to the low bfloat16. Triton's MX-scaled dot widens operands
// with them. Built for gfx950 by build.sh, with clang's assembler.
//
// scalef32(const uint* in, uint* out): in holds 64 words of data and 64 of
// scales (a float whose exponent field is the E8M0 scale); out eight arrays of
// 64 words: FP4 with OPSEL 0, 1, 2, 3; FP8 with 0, 1; BF8 with 0, 1.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl scalef32
  .p2align 8
  .type scalef32,@function
scalef32:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = in, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v0, 2, v0                  // its byte offset
  global_load_dword v1, v0, s[4:5] offset:0
  global_load_dword v2, v0, s[4:5] offset:256
  s_waitcnt vmcnt(0)
  v_cvt_scalef32_pk_bf16_fp4 v4, v1, v2 op_sel:[0,0,0]
  global_store_dword v0, v4, s[6:7] offset:0
  v_cvt_scalef32_pk_bf16_fp4 v4, v1, v2 op_sel:[1,0,0]
  global_store_dword v0, v4, s[6:7] offset:256
  v_cvt_scalef32_pk_bf16_fp4 v4, v1, v2 op_sel:[0,1,0]
  global_store_dword v0, v4, s[6:7] offset:512
  v_cvt_scalef32_pk_bf16_fp4 v4, v1, v2 op_sel:[1,1,0]
  global_store_dword v0, v4, s[6:7] offset:768
  v_cvt_scalef32_pk_bf16_fp8 v4, v1, v2 op_sel:[0,0,0]
  global_store_dword v0, v4, s[6:7] offset:1024
  v_cvt_scalef32_pk_bf16_fp8 v4, v1, v2 op_sel:[1,0,0]
  global_store_dword v0, v4, s[6:7] offset:1280
  v_cvt_scalef32_pk_bf16_bf8 v4, v1, v2 op_sel:[0,0,0]
  global_store_dword v0, v4, s[6:7] offset:1536
  v_cvt_scalef32_pk_bf16_bf8 v4, v1, v2 op_sel:[1,0,0]
  global_store_dword v0, v4, s[6:7] offset:1792
  s_endpgm
.Lscalef32_end:
  .size scalef32, .Lscalef32_end-scalef32

  .rodata
  .p2align 6
  .amdhsa_kernel scalef32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: scalef32
    .symbol: scalef32.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
