// Global loads straight into LDS, gfx950's forms and CDNA3's: each lane's
// element, read from global memory, lands in LDS at M0 plus the offset plus
// the lane's number times the element's size (four dwords, three, one). vLLM's
// skinny GEMMs on MI350X stage their inputs this way. Built for gfx950 by
// build.sh, with clang's assembler.
//
// lds_dma(const int* src, int* out): src holds 256 words; LDS is written three
// ways and then copied, all 768 words of it, to out.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl lds_dma
  .p2align 8
  .type lds_dma,@function
lds_dma:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = src, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 4, v0                  // its sixteen bytes of src
  v_lshlrev_b32_e32 v3, 2, v0                  // its word of LDS and of out, per 256 bytes
  s_mov_b32 m0, 0
  global_load_lds_dwordx4 v1, s[4:5]
  s_mov_b32 m0, 0x400
  global_load_lds_dwordx3 v1, s[4:5] offset:4
  s_mov_b32 m0, 0x800
  global_load_lds_dword v1, s[4:5] offset:8
  s_waitcnt vmcnt(0)
  ds_read_b32 v2, v3 offset:0
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:0
  ds_read_b32 v2, v3 offset:256
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:256
  ds_read_b32 v2, v3 offset:512
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:512
  ds_read_b32 v2, v3 offset:768
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:768
  ds_read_b32 v2, v3 offset:1024
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:1024
  ds_read_b32 v2, v3 offset:1280
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:1280
  ds_read_b32 v2, v3 offset:1536
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:1536
  ds_read_b32 v2, v3 offset:1792
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:1792
  ds_read_b32 v2, v3 offset:2048
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:2048
  ds_read_b32 v2, v3 offset:2304
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:2304
  ds_read_b32 v2, v3 offset:2560
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:2560
  ds_read_b32 v2, v3 offset:2816
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:2816
  s_endpgm
.Llds_dma_end:
  .size lds_dma, .Llds_dma_end-lds_dma

  .rodata
  .p2align 6
  .amdhsa_kernel lds_dma
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 3072
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: lds_dma
    .symbol: lds_dma.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 3072
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
