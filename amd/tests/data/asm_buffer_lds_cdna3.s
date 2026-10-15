// Buffer loads straight into LDS (the MUBUF instruction's LDS bit): each lane's
// data, read through a buffer resource, lands in LDS at M0 plus the
// instruction's offset plus the lane's number times four bytes (sixteen, for
// three or four dwords), as CDNA3 and CDNA4 ISA 9.1.9's figure gives
//   LDS_ADDR = LDSbase + M0[15:0] + inst_offset + TIDinWave * 4
//   MEM_ADDR = Base + mem_offset + inst_offset + off_vgpr + ...
// gfx942 has buffer_load_dword with it (CDNA3 ISA 9.1.9).
// The resource holds 640 bytes, so lanes whose data lies past them read zero.
// Built for gfx942 by build.sh, with clang's assembler.
//
// buf_lds942(const int* src, int* out): src holds 256 words; the LDS the
// loads fill (768 words of it) is copied to out.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl buf_lds942
  .p2align 8
  .type buf_lds942,@function
buf_lds942:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = src, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 4, v0                  // its sixteen bytes of src
  v_lshlrev_b32_e32 v3, 2, v0                  // its word of LDS and of out, per 256 bytes
  s_mov_b32 s8, s4                             // the resource: base, stride 0,
  s_mov_b32 s9, s5                             // 640 bytes, a 32-bit data format
  s_mov_b32 s10, 640
  s_mov_b32 s11, 0x27000
  s_mov_b32 s12, 16                            // a scalar offset
  s_mov_b32 m0, 0x400
  buffer_load_dword v1, s[8:11], s12 offen offset:4 lds
  s_mov_b32 m0, 0x800
  buffer_load_dword v1, s[8:11], 0 offen offset:8 lds
  s_waitcnt vmcnt(0)
  s_mov_b32 s13, 0                             // copy the LDS out, 256 bytes a pass
.Lcopy:
  v_add_u32_e32 v4, s13, v3
  ds_read_b32 v2, v4
  s_waitcnt lgkmcnt(0)
  global_store_dword v4, v2, s[6:7]
  s_add_u32 s13, s13, 256
  s_cmp_lt_u32 s13, 3072
  s_cbranch_scc1 .Lcopy
  s_waitcnt vmcnt(0)
  s_endpgm
.Lbuf_lds942_end:
  .size buf_lds942, .Lbuf_lds942_end-buf_lds942

  .rodata
  .p2align 6
  .amdhsa_kernel buf_lds942
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 3072
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: buf_lds942
    .symbol: buf_lds942.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 3072
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
