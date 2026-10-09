// Cross-lane reads with lanes switched off. The MI300 ISA guide's pseudocode for
// DS_BPERMUTE_B32 reads "if EXEC[src_lane] then VGPR[src_lane][DATA0]" and gives zero
// otherwise; DS_SWIZZLE_B32 says "thread_valid[j] ? thread_in[j] : 0". rocPRIM's warp
// shuffles run exactly so (its intrinsics test shuffles with half the lanes off).
// Built for gfx942 by build.sh, with clang's assembler.
//
// permute(in, out): every lane loads in[lane], every lane's two results are set to 0xAA,
// then only the lanes of EXEC = 0x0123456789ABCDEF run a bpermute from lane + 3 and a
// swizzle that swaps neighbouring lanes. out[lane] is the bpermute, out[64 + lane] the swizzle.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl permute
  .p2align 8
  .type permute,@function
permute:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // in, out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 2, v0
  global_load_dword v2, v1, s[4:5]
  v_mov_b32_e32 v4, 0xaa
  v_mov_b32_e32 v5, 0xaa
  v_add_u32_e32 v3, 3, v0
  v_lshlrev_b32_e32 v3, 2, v3
  s_waitcnt vmcnt(0)
  s_mov_b32 exec_lo, 0x89abcdef
  s_mov_b32 exec_hi, 0x01234567
  ds_bpermute_b32 v4, v3, v2
  ds_swizzle_b32 v5, v2 offset:swizzle(QUAD_PERM,1,0,3,2)
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, -1
  global_store_dword v1, v4, s[6:7]
  global_store_dword v1, v5, s[6:7] offset:256
  s_endpgm
.Lpermute_end:
  .size permute, .Lpermute_end-permute

  .rodata
  .p2align 6
  .amdhsa_kernel permute
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: permute
    .symbol: permute.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
