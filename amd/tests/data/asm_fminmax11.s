// RDNA's float atomic min and max in global memory, returning what they
// found: global_atomic_min_f32 and global_atomic_max_f32 on gfx1100. hip-tests' atomicMin/atomicMax of floats compile
// to them on gfx12. Built by build.sh, with clang's assembler.
//
// fminmax(float* out), out[0] = out[1] = 3.0 beforehand, one lane:
// min(3, 1) and max(3, 5) land in out[0] and out[1]; what each found (3.0)
// goes to out[2] and out[3].
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .text
  .globl fminmax
  .p2align 8
  .type fminmax,@function
fminmax:
  s_load_b64 s[2:3], s[0:1], 0x0
  v_and_b32_e32 v0, 0x3ff, v0
  v_cmp_eq_u32_e32 vcc_lo, 0, v0
  s_and_b32 exec_lo, exec_lo, vcc_lo
  v_mov_b32_e32 v6, 0
  v_mov_b32_e32 v2, 1.0
  v_mov_b32_e32 v3, 0x40a00000
  s_waitcnt lgkmcnt(0)
  global_atomic_min_f32 v4, v6, v2, s[2:3] glc
  global_atomic_max_f32 v5, v6, v3, s[2:3] offset:4 glc
  s_waitcnt vmcnt(0)
  global_store_b64 v6, v[4:5], s[2:3] offset:8
  s_endpgm
.Lfminmax_end:
  .size fminmax, .Lfminmax_end-fminmax

  .rodata
  .p2align 6
  .amdhsa_kernel fminmax
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 7
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: fminmax
    .symbol: fminmax.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 7
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
