// A kernel whose stack grows at run time (.amdhsa_uses_dynamic_stack 1: recursion, indirect or virtual calls) is given the device's
// stack size of scratch per work-item by ROCm's HIP, not only its static private_segment_fixed_size (issue #585; test_amd_hip_dyn_stack
// and test_amd_gcn_dyn_stack). Built for gfx942 by build.sh, with clang's assembler.
//
// dyn_stack(unsigned* out, unsigned at): each lane writes 1000 + L to its scratch at byte `at`, reads it back and stores it to out[L].
//   Its static size is 16 bytes; the descriptor says the stack is dynamic.
// fixed_stack(unsigned* out, unsigned at): the same code, with no dynamic stack: 16 bytes are all it has.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl dyn_stack
  .p2align 8
  .type dyn_stack,@function
dyn_stack:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = out
  s_load_dword s6, s[0:1], 0x8                // s6 = at
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 2, v0                  // 4 L
  v_add_u32_e32 v2, 1000, v0
  v_mov_b32_e32 v3, s6
  scratch_store_dword v3, v2, off
  s_waitcnt vmcnt(0)
  scratch_load_dword v4, v3, off
  s_waitcnt vmcnt(0)
  global_store_dword v1, v4, s[4:5]
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel dyn_stack
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_private_segment_fixed_size 16
    .amdhsa_enable_private_segment 1
    .amdhsa_uses_dynamic_stack 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .globl fixed_stack
  .p2align 8
  .type fixed_stack,@function
fixed_stack:
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_load_dword s6, s[0:1], 0x8
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 2, v0
  v_add_u32_e32 v2, 1000, v0
  v_mov_b32_e32 v3, s6
  scratch_store_dword v3, v2, off
  s_waitcnt vmcnt(0)
  scratch_load_dword v4, v3, off
  s_waitcnt vmcnt(0)
  global_store_dword v1, v4, s[4:5]
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel fixed_stack
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_private_segment_fixed_size 16
    .amdhsa_enable_private_segment 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: dyn_stack
    .symbol: dyn_stack.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 16
    .uses_dynamic_stack: true
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 8, .value_kind: by_value }
  - .name: fixed_stack
    .symbol: fixed_stack.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 16
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 8, .value_kind: by_value }
...
  .end_amdgpu_metadata
