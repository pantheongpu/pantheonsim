// RDNA4's shader clock, which clock64() reads as two hardware registers
// (SHADER_CYCLES_HI, then _LO, then _HI again, to see that the high word did
// not move between). Built for gfx1201 by build.sh, with clang's assembler.
//
// shader_cycles(uint* out): high, low, high, then the low word again.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
  .text
  .globl shader_cycles
  .p2align 8
  .type shader_cycles,@function
shader_cycles:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_getreg_b32 s4, hwreg(HW_REG_SHADER_CYCLES_HI)
  s_getreg_b32 s5, hwreg(HW_REG_SHADER_CYCLES_LO)
  s_getreg_b32 s6, hwreg(HW_REG_SHADER_CYCLES_HI)
  s_getreg_b32 s7, hwreg(HW_REG_SHADER_CYCLES_LO)
  v_mov_b32_e32 v1, s4
  v_mov_b32_e32 v2, s5
  v_mov_b32_e32 v3, s6
  v_mov_b32_e32 v4, s7
  v_mov_b32_e32 v0, 0
  s_wait_kmcnt 0x0
  global_store_b128 v0, v[1:4], s[2:3]
  s_endpgm
.Lshader_cycles_end:
  .size shader_cycles, .Lshader_cycles_end-shader_cycles

  .rodata
  .p2align 6
  .amdhsa_kernel shader_cycles
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 5
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: shader_cycles
    .symbol: shader_cycles.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 5
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
