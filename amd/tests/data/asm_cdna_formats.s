// Formatted buffer accesses (CDNA2, CDNA3 and CDNA4 ISA, BUFFER_LOAD_FORMAT_*, BUFFER_STORE_FORMAT_*, TBUFFER_LOAD_FORMAT_*,
// TBUFFER_STORE_FORMAT_*): "convert the data to 32 bit integral or floating point format, then store the result into a vector register",
// the resource giving the format for MUBUF's and the instruction's own data and number formats for MTBUF's.
//
// fmt(src, out, out2): lane L (an index into a resource of 16-byte records) loads xyzw as 32-bit floats, two 16-bit unsigned integers (MTBUF),
// one 8-bit unorm channel (MTBUF, offset 12) and one float as a half (d16); out gets those 8 dwords a lane. It stores xyzw (MTBUF) into out2 at
// 16 L and xy (MUBUF, 32_32 float) at 1024 + 8 L.
//
// Assembled for gfx90a, gfx942 and gfx950 by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX950 is
// defined, and RET is how the target spells "return the old value": glc on gfx90a, sc0 after it).
#if defined(GFX90A)
#define RET glc
#else
#define RET sc0
#endif
  .text
  .globl fmt
  .p2align 8
  .type fmt,@function
fmt:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = src, s[6:7] = out
  s_load_dwordx2 s[20:21], s[0:1], 0x10       // out2
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v1, 63, v0                     // the lane is the index
  v_lshlrev_b32_e32 v2, 5, v1                  // 32 bytes of out a lane
  s_mov_b32 s8, s4
  s_and_b32 s9, s5, 0xffff
  s_or_b32 s9, s9, 0x100000                    // stride 16
  s_mov_b32 s10, 64                            // 64 records
  s_mov_b32 s11, 0x77fac                       // 32_32_32_32 float, DST_SEL XYZW
  s_mov_b32 s12, s20
  s_and_b32 s13, s21, 0xffff
  s_or_b32 s13, s13, 0x100000
  s_mov_b32 s14, 64
  s_mov_b32 s15, 0x77fac
  s_add_u32 s16, s20, 1024
  s_addc_u32 s17, s21, 0
  s_and_b32 s17, s17, 0xffff
  s_or_b32 s17, s17, 0x80000                   // stride 8
  s_mov_b32 s18, 64
  s_mov_b32 s19, 0x5ffac                       // 32_32 float
  buffer_load_format_xyzw v[4:7], v1, s[8:11], 0 idxen
  tbuffer_load_format_xy v[8:9], v1, s[8:11], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_UINT] idxen
  tbuffer_load_format_x v10, v1, s[8:11], 0 format:[BUF_DATA_FORMAT_8,BUF_NUM_FORMAT_UNORM] idxen offset:12
  buffer_load_format_d16_x v11, v1, s[8:11], 0 idxen
  s_waitcnt vmcnt(0)
  global_store_dwordx4 v2, v[4:7], s[6:7] offset:0
  global_store_dwordx2 v2, v[8:9], s[6:7] offset:16
  global_store_dword v2, v10, s[6:7] offset:24
  global_store_dword v2, v11, s[6:7] offset:28
  tbuffer_store_format_xyzw v[4:7], v1, s[12:15], 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_FLOAT] idxen
  buffer_store_format_xy v[8:9], v1, s[16:19], 0 idxen
  s_waitcnt vmcnt(0)
  s_endpgm
.Lfmt_end:
  .size fmt, .Lfmt_end-fmt
  .rodata
  .p2align 6
  .amdhsa_kernel fmt
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: fmt
    .symbol: fmt.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata

