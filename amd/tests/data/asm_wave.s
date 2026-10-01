// RDNA3 kernels for what a wave has whatever its work-items: scratch for
// every lane, which a function saving whole-wave registers stores to with
// every lane on, even in a work-group smaller than a wave; and the shader
// clock, which clock() reads as a hardware register (SHADER_CYCLES, 20
// bits). Built for gfx1100 by build.sh, with clang's assembler.
//
// whole_wave_scratch(uint* out), one work-item: with every lane on, each
// lane stores 0x5eed0000 | its lane number to its own scratch and reads it
// back; out[0] is lane 0's and out[1] lane 31's.
// shader_cycles(uint* out): the counter read twice, a few instructions apart.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .text
  .globl whole_wave_scratch
  .p2align 8
  .type whole_wave_scratch,@function
whole_wave_scratch:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_or_saveexec_b32 s4, -1
  v_mbcnt_lo_u32_b32 v1, -1, 0
  v_or_b32_e32 v1, 0x5eed0000, v1
  scratch_store_b32 off, v1, off offset:4
  v_mov_b32_e32 v1, 0
  scratch_load_b32 v2, off, off offset:4
  s_waitcnt vmcnt(0)
  v_readlane_b32 s5, v2, 31
  s_mov_b32 exec_lo, s4
  v_mov_b32_e32 v3, s5
  v_mov_b32_e32 v4, 0
  s_waitcnt lgkmcnt(0)
  global_store_b64 v4, v[2:3], s[2:3]
  s_endpgm
.Lwhole_wave_scratch_end:
  .size whole_wave_scratch, .Lwhole_wave_scratch_end-whole_wave_scratch

  .globl shader_cycles
  .p2align 8
  .type shader_cycles,@function
shader_cycles:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_getreg_b32 s4, hwreg(HW_REG_SHADER_CYCLES, 0, 20)
  s_nop 0
  s_nop 0
  s_getreg_b32 s5, hwreg(HW_REG_SHADER_CYCLES, 0, 20)
  v_mov_b32_e32 v1, s4
  v_mov_b32_e32 v2, s5
  v_mov_b32_e32 v0, 0
  s_waitcnt lgkmcnt(0)
  global_store_b64 v0, v[1:2], s[2:3]
  s_endpgm
.Lshader_cycles_end:
  .size shader_cycles, .Lshader_cycles_end-shader_cycles

  .rodata
  .p2align 6
  .amdhsa_kernel whole_wave_scratch
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_enable_private_segment 1
    .amdhsa_private_segment_fixed_size 16
    .amdhsa_next_free_vgpr 5
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel
  .p2align 6
  .amdhsa_kernel shader_cycles
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: whole_wave_scratch
    .symbol: whole_wave_scratch.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 16
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 5
    .max_flat_workgroup_size: 32
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
  - .name: shader_cycles
    .symbol: shader_cycles.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 3
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
