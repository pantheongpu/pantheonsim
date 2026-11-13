// A load into accumulation registers the wave has not used yet, made for the whole wave at once (issue #621). The accumulation registers
// are a vector that grows to the highest one named; the whole-wave load took row 0 of the destination, then row 1, whose growth moved the
// vector, and wrote row 0 into the freed copy: a38 of "global_load_dwordx2 a[38:39], ..." kept what it held before. A Kokkos kernel of
// llnl/exadis loads a pointer that way, then reads through it plus an index times 88: "invalid-pointer" at a small address.
// The expected values are what the CDNA3 ISA guide gives (a global load whose data operand is accumulation registers -- the ACC bit -- writes
// the loaded dwords to them in order, as it writes VGPRs; v_accvgpr_read_b32 copies one into a VGPR), written in
// amd/tests/unit/test_gcn_issue621.cpp, not taken from the simulator.
//
// agpr_uniform(in, out): every lane loads in[50], in[51] (one address, a wave-uniform pointer) into a[38:39]; out at 16 L gets the pair.
// agpr_gather(in, out): lane L loads in[4 L], in[4 L + 1] (16-byte stride, not a run) into a[38:39]; out at 16 L gets the pair.
// agpr_half(in, out): as agpr_gather with lanes 32 to 63 off; their out is left as it was.
// agpr_gather4(in, out): lane L loads in[8 L] .. in[8 L + 3] into a[36:39]; out at 32 L gets the four.
//
// Assembled for gfx90a and gfx942 by build.sh (clang's assembler with the C preprocessor: GFX90A or GFX942 is defined).
  .text

  .globl agpr_uniform
  .p2align 8
  .type agpr_uniform,@function
agpr_uniform:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[6:7], s[0:1], 0x8          // s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 4, v0                  // 16 L
  v_mov_b32_e32 v2, 0
  global_load_dwordx2 a[38:39], v2, s[4:5] offset:200   // every lane at in + 200: in[50], in[51]
  s_waitcnt vmcnt(0)
  v_accvgpr_read_b32 v4, a38
  v_accvgpr_read_b32 v5, a39
  global_store_dwordx2 v1, v[4:5], s[6:7]
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel agpr_uniform
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel

  .globl agpr_gather
  .p2align 8
  .type agpr_gather,@function
agpr_gather:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[6:7], s[0:1], 0x8          // s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 4, v0                  // 16 L
  global_load_dwordx2 a[38:39], v1, s[4:5]     // lane L at in + 16 L: in[4 L], in[4 L + 1]
  s_waitcnt vmcnt(0)
  v_accvgpr_read_b32 v4, a38
  v_accvgpr_read_b32 v5, a39
  global_store_dwordx2 v1, v[4:5], s[6:7]
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel agpr_gather
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel

  .globl agpr_half
  .p2align 8
  .type agpr_half,@function
agpr_half:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[6:7], s[0:1], 0x8          // s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 4, v0                  // 16 L
  v_cmp_gt_u32_e32 vcc, 32, v0
  s_and_saveexec_b64 s[8:9], vcc               // lanes 0 to 31
  global_load_dwordx2 a[38:39], v1, s[4:5]     // lane L at in + 16 L
  s_waitcnt vmcnt(0)
  v_accvgpr_read_b32 v4, a38
  v_accvgpr_read_b32 v5, a39
  global_store_dwordx2 v1, v[4:5], s[6:7]
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel agpr_half
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel

  .globl agpr_gather4
  .p2align 8
  .type agpr_gather4,@function
agpr_gather4:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[6:7], s[0:1], 0x8          // s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 5, v0                  // 32 L
  global_load_dwordx4 a[36:39], v1, s[4:5]     // lane L at in + 32 L: in[8 L] .. in[8 L + 3]
  s_waitcnt vmcnt(0)
  v_accvgpr_read_b32 v4, a36
  v_accvgpr_read_b32 v5, a37
  v_accvgpr_read_b32 v6, a38
  v_accvgpr_read_b32 v7, a39
  global_store_dwordx4 v1, v[4:7], s[6:7]
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel agpr_gather4
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: agpr_uniform
    .symbol: agpr_uniform.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .agpr_count: 48
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: agpr_gather
    .symbol: agpr_gather.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .agpr_count: 48
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: agpr_half
    .symbol: agpr_half.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .agpr_count: 48
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: agpr_gather4
    .symbol: agpr_gather4.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .agpr_count: 48
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
