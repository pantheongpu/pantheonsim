// Loads straight into LDS with narrow data and with M0 beyond 16 bits (CDNA2, CDNA3 and CDNA4 ISA 9.1.9 and 10.3): lane L's data is
// written to LDS at M0 + the offset + 4 L, a byte or a half widened to a dword as its instruction says ("zero extend to 32 bits", sign
// extended for the S forms), and format_x writes the channel it converts. MUBUF takes M0[15:0] as the offset ("16-bit unsigned byte
// offset from M0[15:0]"), FLAT and global M0[17:2] * 4 ("LDSoffset(M0[17:2] * 4)").
//
// buf_lds_narrow(src, out): buffer_load_{ubyte,sbyte,ushort,sshort,format_x} with the LDS bit (gfx90a, gfx942 and gfx950), then LDS
// is copied to out, 64 words a load. glob_lds_narrow(src, out): the same through global_load_lds_* (gfx942 and gfx950), whose last
// load (gfx950) puts a dword past 64 KiB.
//
// Assembled for gfx90a, gfx942 and gfx950 by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX950 is
// defined, and RET is how the target spells "return the old value": glc on gfx90a, sc0 after it).
#if defined(GFX90A)
#define RET glc
#else
#define RET sc0
#endif
  .text
#if defined(GFX950)
#define GLOBAL_LDS 70000
#else
#define GLOBAL_LDS 2048
#endif
  .globl buf_lds_narrow
  .p2align 8
  .type buf_lds_narrow,@function
buf_lds_narrow:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = src, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_lshlrev_b32_e32 v1, 2, v0                  // its off_vgpr: four bytes a lane
  v_lshlrev_b32_e32 v3, 2, v0                  // its word of LDS and of out, per 256 bytes
  s_mov_b32 s8, s4
  s_mov_b32 s9, s5                             // the resource: base, stride 0, 512 bytes, an 8-bit unorm format
  s_mov_b32 s10, 512                           // read through DST_SEL XYZW (what buffer_load_format_x takes its channel by)
  s_mov_b32 s11, 0x8fac
  s_mov_b32 m0, 0
  buffer_load_ubyte v1, s[8:11], 0 offen lds
  s_mov_b32 m0, 0x100
  buffer_load_sbyte v1, s[8:11], 0 offen lds
  s_mov_b32 m0, 0x200
  buffer_load_ushort v1, s[8:11], 0 offen lds
  s_mov_b32 m0, 0x300
  buffer_load_sshort v1, s[8:11], 0 offen lds
  s_mov_b32 s12, 3                             // (the scalar offset moves only the address in memory: the instruction's own
  s_mov_b32 m0, 0x400                          // offset moves the LDS one too)
  buffer_load_format_x v1, s[8:11], s12 offen lds
  s_mov_b32 s12, 2
  s_mov_b32 m0, 0x500
  buffer_load_ushort v1, s[8:11], s12 offen lds
#if !defined(GFX950)
  s_mov_b32 s12, 1
  s_mov_b32 m0, 0x10600                        // only M0[15:0] is the offset: the LDS offset is 0x600 (CDNA2 and CDNA3)
  buffer_load_ubyte v1, s[8:11], s12 offen lds
#endif
  s_mov_b32 m0, 0x700
  buffer_load_dword v1, s[8:11], 0 offen lds
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
  s_waitcnt vmcnt(0)
  s_endpgm
.Lbuf_lds_narrow_end:
  .size buf_lds_narrow, .Lbuf_lds_narrow_end-buf_lds_narrow
#if !defined(GFX90A)
  .globl glob_lds_narrow
  .p2align 8
  .type glob_lds_narrow,@function
glob_lds_narrow:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = src, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 2, v0                  // four bytes of src a lane
  v_lshlrev_b32_e32 v3, 2, v0
  s_mov_b32 m0, 0
  global_load_lds_ubyte v1, s[4:5]
  s_mov_b32 m0, 0x100
  global_load_lds_sbyte v1, s[4:5]
  s_mov_b32 m0, 0x200
  global_load_lds_ushort v1, s[4:5]
  s_mov_b32 m0, 0x300
  global_load_lds_sshort v1, s[4:5]
  s_mov_b32 m0, 0x403                          // M0[17:2] * 4: the low two bits are not part of the offset
  global_load_lds_dword v1, s[4:5]
  s_mov_b32 m0, 0x500
  v_add_u32_e32 v6, 2, v1                      // the address register moves the memory side alone
  global_load_lds_ushort v6, s[4:5]
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
#if defined(GFX950)
  s_mov_b32 m0, 0x10601                        // 18 bits of it: 0x10600 (LDS is 160 KiB there)
  global_load_lds_dword v1, s[4:5]
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v5, 0x10600, v3
  ds_read_b32 v2, v5
  s_waitcnt lgkmcnt(0)
  global_store_dword v3, v2, s[6:7] offset:1536
#endif
  s_waitcnt vmcnt(0)
  s_endpgm
.Lglob_lds_narrow_end:
  .size glob_lds_narrow, .Lglob_lds_narrow_end-glob_lds_narrow
#endif
  .rodata
  .p2align 6
  .amdhsa_kernel buf_lds_narrow
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 2048
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel
#if !defined(GFX90A)
  .p2align 6
  .amdhsa_kernel glob_lds_narrow
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size GLOBAL_LDS
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel
#endif
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: buf_lds_narrow
    .symbol: buf_lds_narrow.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 2048
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
#if !defined(GFX90A)
  - .name: glob_lds_narrow
    .symbol: glob_lds_narrow.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: GLOBAL_LDS
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
#endif
...
  .end_amdgpu_metadata

