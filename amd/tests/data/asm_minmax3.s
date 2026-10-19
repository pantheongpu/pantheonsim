// gfx950's IEEE three-way minimum and maximum (CDNA4 ISA, V_MINIMUM3_F32 and
// V_MAXIMUM3_F32: "Select the IEEE minimum()/maximum() of three
// single-precision float inputs"; "A signaling NaN in either argument is
// propagated to the result"; D0 = v_maximum_f32(v_maximum_f32(S0, S1), S2)),
// and the packed half forms (V_PK_MINIMUM3_F16, V_PK_MAXIMUM3_F16: the same on
// each half). Inductor's Triton kernels use them for relu and maximum
// (triton_poi_fused_relu on MI350X). Built for gfx950 by build.sh, with
// clang's assembler.
//
// minmax3(const uint* in, uint* out): in holds three arrays of 64 words
// (a, b, c); out four: v_maximum3_f32, v_minimum3_f32, v_pk_maximum3_f16,
// v_pk_minimum3_f16 of (a, b, c), lane by lane.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl minmax3
  .p2align 8
  .type minmax3,@function
minmax3:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = in, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v0, 2, v0                  // its byte offset
  global_load_dword v1, v0, s[4:5] offset:0
  global_load_dword v2, v0, s[4:5] offset:256
  global_load_dword v3, v0, s[4:5] offset:512
  s_waitcnt vmcnt(0)
  v_maximum3_f32 v4, v1, v2, v3
  v_minimum3_f32 v5, v1, v2, v3
  v_pk_maximum3_f16 v6, v1, v2, v3
  v_pk_minimum3_f16 v7, v1, v2, v3
  global_store_dword v0, v4, s[6:7] offset:0
  global_store_dword v0, v5, s[6:7] offset:256
  global_store_dword v0, v6, s[6:7] offset:512
  global_store_dword v0, v7, s[6:7] offset:768
  s_endpgm
.Lminmax3_end:
  .size minmax3, .Lminmax3_end-minmax3

  .rodata
  .p2align 6
  .amdhsa_kernel minmax3
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: minmax3
    .symbol: minmax3.kd
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
