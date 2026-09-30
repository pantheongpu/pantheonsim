// The two clocks a kernel reads: s_memrealtime, the real-time clock at the
// constant 100 MHz the runtime reports as hipDeviceAttributeWallClockRate
// (what wall_clock64() reads), and s_memtime, the shader clock. Built for
// gfx942 by build.sh, with clang's assembler.
//
// realtime(long* out) writes the real-time clock, then the shader clock.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl realtime
  .p2align 8
  .type realtime,@function
realtime:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_memrealtime s[4:5]
  s_memtime s[6:7]
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1                            // one lane writes
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, s4
  global_store_dword v0, v1, s[2:3]
  v_mov_b32_e32 v1, s5
  global_store_dword v0, v1, s[2:3] offset:4
  v_mov_b32_e32 v1, s6
  global_store_dword v0, v1, s[2:3] offset:8
  v_mov_b32_e32 v1, s7
  global_store_dword v0, v1, s[2:3] offset:12
  s_endpgm
.Lrealtime_end:
  .size realtime, .Lrealtime_end-realtime

  .rodata
  .p2align 6
  .amdhsa_kernel realtime
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: realtime
    .symbol: realtime.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 2
    .max_flat_workgroup_size: 64
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
...
  .end_amdgpu_metadata
