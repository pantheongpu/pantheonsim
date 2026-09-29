// gfx950's transposing LDS reads, 64 bits a lane of 16-, 8- and 4-bit
// elements: each lane writes a pattern of its own into LDS, then all three
// forms read it back and the test checks where each element went. Built for
// gfx950 by build.sh, with clang's assembler.
//
// tr(const long* pattern, long* out): pattern holds 64 words of 64 bits, one
// a lane; out gets three of 64, the b16, b8 and b4 reads, lane by lane.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl tr
  .p2align 8
  .type tr,@function
tr:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = pattern, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 3, v0                  // its 64 bits, in LDS and in memory
  global_load_dwordx2 v[2:3], v1, s[4:5]
  s_waitcnt vmcnt(0)
  ds_write_b64 v1, v[2:3]
  s_waitcnt lgkmcnt(0)
  ds_read_b64_tr_b16 v[4:5], v1
  ds_read_b64_tr_b8 v[6:7], v1
  ds_read_b64_tr_b4 v[8:9], v1
  s_waitcnt lgkmcnt(0)
  global_store_dwordx2 v1, v[4:5], s[6:7]
  global_store_dwordx2 v1, v[6:7], s[6:7] offset:512
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:1024
  s_endpgm
.Ltr_end:
  .size tr, .Ltr_end-tr

  .rodata
  .p2align 6
  .amdhsa_kernel tr
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 512
    .amdhsa_next_free_vgpr 10
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 12
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: tr
    .symbol: tr.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 512
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 10
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
