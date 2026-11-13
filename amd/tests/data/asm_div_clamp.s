// The clamp bit and the output modifier of v_div_fmas and v_div_fixup (test_amd_gcn_div_clamp, issue #613). clang folds
// fmin(fmax(a / b, 0), 1) into the clamp of the division's last instruction ("v_div_fixup_f64 v[0:1], v[4:5], v[2:3], v[0:1] clamp");
// the executor wrote the quotient unclamped.
//
// div_clamp(x32, x64, out): lane L reads the float x32[L] and the double x64[L], normal numbers. With a denominator of 1.0 and the
// numerator the quotient itself, v_div_fixup is the plain case (the quotient with the operands' sign, CDNA3 ISA), so it returns x; with
// VCC clear, v_div_fmas(x, 1.0, 0) is x * 1.0 + 0 = x. Each is then held to [0, 1] by the clamp bit. out, in dwords (lane L at +L):
//     0  v_div_fixup_f32 ... clamp           64  v_div_fmas_f32 ... clamp      128  v_div_fixup_f32 ... mul:2
//   192  v_mul_f32 1.0, x mul:2 (the output modifier of an ordinary VOP3 float result, for comparison)
//   256  v_div_fixup_f32 with neither (x)
// and in qwords from byte 1280 (lane L at +8 L): v_div_fixup_f64 ... clamp, then from byte 1792 v_div_fmas_f64 ... clamp.
//
// Assembled for gfx90a and gfx942 by build.sh (clang's assembler with the C preprocessor: GFX90A or GFX942 is defined).
  .text
  .globl div_clamp
  .p2align 8
  .type div_clamp,@function
div_clamp:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = x32
  s_load_dwordx2 s[6:7], s[0:1], 0x8          // s[6:7] = x64
  s_load_dwordx2 s[8:9], s[0:1], 0x10         // s[8:9] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 2, v0                  // 4 L
  v_lshlrev_b32_e32 v3, 3, v0                  // 8 L
  global_load_dword v2, v1, s[4:5]
  global_load_dwordx2 v[4:5], v3, s[6:7]
  s_waitcnt vmcnt(0)
  s_mov_b64 vcc, 0
  v_div_fixup_f32 v6, v2, 1.0, v2 clamp
  v_div_fmas_f32 v7, v2, 1.0, 0 clamp
  v_div_fixup_f32 v8, v2, 1.0, v2 mul:2
  v_mul_f32_e64 v9, 1.0, v2 mul:2
  v_div_fixup_f32 v14, v2, 1.0, v2
  v_div_fixup_f64 v[10:11], v[4:5], 1.0, v[4:5] clamp
  v_div_fmas_f64 v[12:13], v[4:5], 1.0, 0 clamp
  global_store_dword v1, v6, s[8:9] offset:0
  global_store_dword v1, v7, s[8:9] offset:256
  global_store_dword v1, v8, s[8:9] offset:512
  global_store_dword v1, v9, s[8:9] offset:768
  global_store_dword v1, v14, s[8:9] offset:1024
  global_store_dwordx2 v3, v[10:11], s[8:9] offset:1280
  global_store_dwordx2 v3, v[12:13], s[8:9] offset:1792
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel div_clamp
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 16
    .amdhsa_float_denorm_mode_32 3               // denormals kept, as the compiler's kernels have it (the tier runs only that mode)
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: div_clamp
    .symbol: div_clamp.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
