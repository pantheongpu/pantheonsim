// Two work-groups, one waiting for the other: group 0 polls a flag, without
// sleeping between reads, that group 1 sets -- as hipBLASLt's Stream-K GEMMs have a
// group wait for another's partial tile. The hardware keeps both resident, so
// group 0 is released; run one group at a time on one host thread, it waited
// for ever. Built for gfx942 by build.sh, with clang's assembler.
//
// wait(int* flag, int* out): group 0 writes 42 to out once the flag is set.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl wait
  .p2align 8
  .type wait,@function
wait:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = flag, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1                            // one lane
  v_mov_b32_e32 v0, 0
  s_cmp_eq_u32 s2, 0                           // the work-group's number
  s_cbranch_scc0 .Lset
.Lpoll:
  global_load_dword v1, v0, s[4:5] sc0 sc1
  s_waitcnt vmcnt(0)
  v_readfirstlane_b32 s8, v1
  s_cmp_eq_u32 s8, 0
  s_cbranch_scc0 .Lreleased
  s_branch .Lpoll
.Lreleased:
  v_mov_b32_e32 v2, 42
  global_store_dword v0, v2, s[6:7]
  s_endpgm
.Lset:
  v_mov_b32_e32 v1, 1
  global_store_dword v0, v1, s[4:5] sc0 sc1
  s_endpgm
.Lwait_end:
  .size wait, .Lwait_end-wait

  .rodata
  .p2align 6
  .amdhsa_kernel wait
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 12
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: wait
    .symbol: wait.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 3
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
