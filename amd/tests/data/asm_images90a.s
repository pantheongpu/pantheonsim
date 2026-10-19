// gfx90a's (MI200) image instructions -- MIMG, which has no DIM field and only
// one sample, image_sample (AMD's MI200 ISA guide, 12.15) -- on resources the
// test writes itself in the layout of the guide's Table 41 (vgpu/amd_image.hpp):
// loads (plain, from a mip level, through 16-bit addresses, as 16-bit data),
// stores (a whole texel, with the channels DMASK leaves out written as zero),
// an atomic add, point and bilinear samples, a sample and a load of a 2D array
// (DA set), a load of a 3D image, and the resource query. Built for gfx90a by
// build.sh, with clang's assembler.
//
// images(uint* out, uint* desc): desc holds, in dwords,
//    0 T0: an 8x4 32-bit float image
//    8 T1: a 4x2 8_8_8_8 unorm image
//   16 T2: an 8x4 32-bit uint image, stored to and added to
//   24 S#: point, clamp to the last texel;  28 S#: the same, bilinear
//   32 T3: a 4x4 float image with two levels
//   40 T4: a 4x4x2 float 2D array;  48 T5: a 4x4x2 float 3D image
//   56 T6: an 8x4 8_8_8_8 uint image, all 255 to begin with
//   64 T7: an 8x4 32_32 uint image (64 bits a texel): (0xfffffff0 + l, 7 + l)
// One wave of 64 lanes, 32 of them on; lane l is texel (l % 8, l / 8) of T0.
// out gets, in words: 0 the loads; 32 point samples at texel centers; 64
// bilinear samples on the left edge of each texel; 96 T1's four channels (4 a
// lane); 224 T3's size at level l % 2 (4 a lane); 352 T3's level 1 texel
// (l % 2, l / 2 % 2); 384 T4 at (l % 4, l / 4 % 4, l / 16) by a load; 416 the same
// by a sample; 448 T5 there by a load; 480 what the atomic add found; 512 loads
// with 16-bit coordinates; 544 T1's red and green as halves; 576 T6 read
// back (4 a lane) after a store of red and alpha only; 704 what a 64-bit add of
// 0x100000020 to T7 found (2 a lane); 768 what a compare-and-swap on T2 found,
// 800 T2 after it; 832 what a 64-bit compare-and-swap on T7 found (2 a lane),
// 896 T7 after it (2 a lane).
  .amdgcn_target "amdgcn-amd-amdhsa--gfx90a"
  .text
  .globl images
  .p2align 8
  .type images,@function
images:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] out, s[6:7] desc
  s_waitcnt lgkmcnt(0)
  s_load_dwordx8 s[8:15], s[6:7], 0x0
  s_load_dwordx8 s[16:23], s[6:7], 0x20
  s_load_dwordx8 s[24:31], s[6:7], 0x40
  s_load_dwordx4 s[32:35], s[6:7], 0x60
  s_load_dwordx4 s[36:39], s[6:7], 0x70
  s_load_dwordx8 s[40:47], s[6:7], 0x80
  s_load_dwordx8 s[48:55], s[6:7], 0xa0
  s_load_dwordx8 s[56:63], s[6:7], 0xc0
  s_load_dwordx8 s[64:71], s[6:7], 0xe0
  s_load_dwordx8 s[72:79], s[6:7], 0x100
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v2, 7, v0                   // x
  v_lshrrev_b32_e32 v3, 3, v0               // y
  v_mov_b32_e32 v8, 0x3e000000              // 1/8
  v_mov_b32_e32 v9, 0x3d800000              // 1/16
  v_mov_b32_e32 v10, 0x3e800000             // 1/4
  v_cvt_f32_u32_e32 v4, v2
  v_fma_f32 v4, v4, v8, v9                  // (x + 0.5) / 8
  v_cvt_f32_u32_e32 v5, v3
  v_fma_f32 v5, v5, v10, v8                 // (y + 0.5) / 4
  v_cvt_f32_u32_e32 v1, v2
  v_mul_f32_e32 v1, v1, v8                  // x / 8
  v_lshlrev_b32_e32 v6, 2, v0               // l * 4
  v_lshlrev_b32_e32 v7, 4, v0               // l * 16

  image_load v12, v[2:3], s[8:15] dmask:0x1 unorm
  image_sample v13, v[4:5], s[8:15], s[32:35] dmask:0x1
  v_mov_b32_e32 v14, v1
  v_mov_b32_e32 v15, v5
  image_sample v14, v[14:15], s[8:15], s[36:39] dmask:0x1
  s_waitcnt vmcnt(0)
  global_store_dword v6, v12, s[4:5]
  global_store_dword v6, v13, s[4:5] offset:128
  global_store_dword v6, v14, s[4:5] offset:256

  v_and_b32_e32 v16, 3, v0
  v_bfe_u32 v17, v0, 2, 1
  image_load v[20:23], v[16:17], s[16:23] dmask:0xf unorm
  v_and_b32_e32 v18, 1, v0
  image_get_resinfo v[24:27], v18, s[40:47] dmask:0xf unorm
  s_waitcnt vmcnt(0)
  global_store_dwordx4 v7, v[20:23], s[4:5] offset:384
  global_store_dwordx4 v7, v[24:27], s[4:5] offset:896

  // T3's level 1: (l % 2, l / 2 % 2, level 1)
  v_and_b32_e32 v32, 1, v0
  v_bfe_u32 v33, v0, 1, 1
  v_mov_b32_e32 v34, 1
  image_load_mip v30, v[32:34], s[40:47] dmask:0x1 unorm
  // T4 and T5 at (l % 4, l / 4 % 4, l / 16)
  v_and_b32_e32 v36, 3, v0
  v_bfe_u32 v37, v0, 2, 2
  v_lshrrev_b32_e32 v38, 4, v0
  image_load v35, v[36:38], s[48:55] dmask:0x1 unorm da
  v_cvt_f32_u32_e32 v40, v36
  v_cvt_f32_u32_e32 v41, v37
  v_fma_f32 v40, v40, v10, v9               // x / 4 + 1/16
  v_fma_f32 v41, v41, v10, v9
  v_add_f32_e32 v40, v9, v40                // (x + 0.5) / 4
  v_add_f32_e32 v41, v9, v41
  v_cvt_f32_u32_e32 v42, v38                // the layer, as a float
  image_sample v43, v[40:42], s[48:55], s[32:35] dmask:0x1 da
  image_load v44, v[36:38], s[56:63] dmask:0x1 unorm
  s_waitcnt vmcnt(0)
  global_store_dword v6, v30, s[4:5] offset:1408
  global_store_dword v6, v35, s[4:5] offset:1536
  global_store_dword v6, v43, s[4:5] offset:1664
  global_store_dword v6, v44, s[4:5] offset:1792

  // A store, then an atomic add that hands back what it found.
  v_mul_u32_u24_e32 v45, 3, v0
  image_store v45, v[2:3], s[24:31] dmask:0x1 unorm
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v46, 5
  image_atomic_add v46, v[2:3], s[24:31] dmask:0x1 unorm glc
  // 16-bit coordinates, and 16-bit data.
  v_lshlrev_b32_e32 v47, 16, v3
  v_or_b32_e32 v47, v47, v2                 // x | y << 16
  image_load v48, v47, s[8:15] dmask:0x1 unorm a16
  image_load v49, v[16:17], s[16:23] dmask:0x3 unorm d16
  s_waitcnt vmcnt(0)
  global_store_dword v6, v46, s[4:5] offset:1920
  global_store_dword v6, v48, s[4:5] offset:2048
  global_store_dword v6, v49, s[4:5] offset:2176

  // A store of red and alpha: green and blue are written as zero.
  v_add_u32_e32 v50, 1, v0
  v_sub_u32_e32 v51, 200, v0
  image_store v[50:51], v[2:3], s[64:71] dmask:0x9 unorm
  s_waitcnt vmcnt(0)
  image_load v[52:55], v[2:3], s[64:71] dmask:0xf unorm
  s_waitcnt vmcnt(0)
  global_store_dwordx4 v7, v[52:55], s[4:5] offset:2304

  // A 64-bit add of 0x100000020, then a 64-bit compare-and-swap that finds the sum, and a 32-bit one.
  v_mov_b32_e32 v56, 0x20
  v_mov_b32_e32 v57, 1
  image_atomic_add v[56:57], v[2:3], s[72:79] dmask:0x3 unorm glc
  s_waitcnt vmcnt(0)
  v_add_co_u32_e32 v60, vcc, 0x20, v56        // the sum: the old value plus 0x100000020
  v_addc_co_u32_e32 v61, vcc, 1, v57, vcc
  v_add_u32_e32 v58, 0xbeef0000, v0
  v_mov_b32_e32 v59, 0xcafe
  image_atomic_cmpswap v[58:61], v[2:3], s[72:79] dmask:0xf unorm glc
  v_add_u32_e32 v64, 1000, v0
  v_mul_u32_u24_e32 v65, 3, v0
  v_add_u32_e32 v65, 5, v65                   // what T2 holds: 3 l + 5
  image_atomic_cmpswap v[64:65], v[2:3], s[24:31] dmask:0x3 unorm glc
  s_waitcnt vmcnt(0)
  image_load v[62:63], v[2:3], s[72:79] dmask:0x3 unorm
  image_load v66, v[2:3], s[24:31] dmask:0x1 unorm
  v_lshlrev_b32_e32 v68, 3, v0
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v68, v[56:57], s[4:5] offset:2816
  global_store_dword v6, v64, s[4:5] offset:3072
  global_store_dword v6, v66, s[4:5] offset:3200
  global_store_dwordx2 v68, v[58:59], s[4:5] offset:3328
  global_store_dwordx2 v68, v[62:63], s[4:5] offset:3584
  s_endpgm
.Limages_end:
  .size images, .Limages_end-images

// formats(uint* out, uint* desc): gfx90a's formatted buffer accesses (MUBUF
// opcodes 0-15, 38 and 39), through resources whose format is in the resource:
// desc holds, in dwords, 0 V0: 32 elements of 8_8_8_8 unorm, 4 V1: 16_16
// float, 8 V2: 8_8_8_8 uint (zero to begin with), 12 V3: 32-bit float (zero to
// begin with). Lane l, of 32, writes to out, in words: 0 V0[l] by
// buffer_load_format_xyzw (4 a lane); 128 V1[l] by _xy (2 a lane); 192 V0[l]'s
// red and green as halves by _d16_xy; 224 what a store_format_xyzw of
// (l, 2 l, 3 l, 4 l) into V2[l] reads back as (4 a lane); 352 what a
// store_format_x of l + 0.5 into V3[l] reads back as raw bits; 384 V0[l]'s red
// as a half in the high half of a register by _d16_hi_x, over 0x1234 in the
// low half; 416 V1[l] read by _d16_xy after a store_format_d16_xy of the
// pair (l + 0.5, l + 1.5) as halves into it.
  .globl formats
  .p2align 8
  .type formats,@function
formats:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] out, s[6:7] desc
  s_waitcnt lgkmcnt(0)
  s_load_dwordx4 s[8:11], s[6:7], 0x0
  s_load_dwordx4 s[12:15], s[6:7], 0x10
  s_load_dwordx4 s[16:19], s[6:7], 0x20
  s_load_dwordx4 s[20:23], s[6:7], 0x30
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32_e32 v1, 2, v0               // l * 4
  v_lshlrev_b32_e32 v20, 4, v0              // l * 16
  v_lshlrev_b32_e32 v21, 3, v0              // l * 8
  buffer_load_format_xyzw v[2:5], v0, s[8:11], 0 idxen
  buffer_load_format_xy v[6:7], v0, s[12:15], 0 idxen
  buffer_load_format_d16_xy v8, v0, s[8:11], 0 idxen
  tbuffer_load_format_xyzw v[26:29], v0, s[12:15], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UINT] idxen
  tbuffer_load_format_xyzw v[30:33], v0, s[12:15], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_UINT] idxen
  v_add_u32_e32 v10, 0, v0
  v_lshlrev_b32_e32 v11, 1, v0
  v_mul_u32_u24_e32 v12, 3, v0
  v_lshlrev_b32_e32 v13, 2, v0
  buffer_store_format_xyzw v[10:13], v0, s[16:19], 0 idxen
  v_cvt_f32_u32_e32 v18, v0
  v_add_f32_e32 v18, 0.5, v18
  buffer_store_format_x v18, v0, s[20:23], 0 idxen
  s_waitcnt vmcnt(0)
  buffer_load_format_xyzw v[14:17], v0, s[16:19], 0 idxen
  buffer_load_dword v19, v0, s[20:23], 0 idxen
  v_mov_b32_e32 v38, v0                         // the pair (l, 2 l)
  v_lshlrev_b32_e32 v39, 1, v0
  s_waitcnt vmcnt(0)
  global_store_dwordx4 v20, v[2:5], s[4:5]
  global_store_dwordx2 v21, v[6:7], s[4:5] offset:512
  global_store_dword v1, v8, s[4:5] offset:768
  global_store_dwordx4 v20, v[26:29], s[4:5] offset:1792
  global_store_dwordx4 v20, v[30:33], s[4:5] offset:2304
  global_store_dwordx4 v20, v[14:17], s[4:5] offset:896
  global_store_dword v1, v19, s[4:5] offset:1408
  // The high half alone, over a register that held 0x1234.
  v_mov_b32_e32 v22, 0x1234
  buffer_load_format_d16_hi_x v22, v0, s[8:11], 0 idxen offset:0
  // Two halves stored, and read back.
  v_cvt_f16_f32_e32 v23, v18
  v_add_f32_e32 v24, 1.0, v18
  v_cvt_f16_f32_e32 v24, v24
  v_lshlrev_b32_e32 v24, 16, v24
  v_or_b32_e32 v23, v23, v24
  buffer_store_format_d16_xy v23, v0, s[12:15], 0 idxen
  s_waitcnt vmcnt(0)
  buffer_load_format_d16_xy v25, v0, s[12:15], 0 idxen
  s_waitcnt vmcnt(0)
  global_store_dword v1, v22, s[4:5] offset:1536
  global_store_dword v1, v25, s[4:5] offset:1664
  // The pair (l, 2 l) stored as 16_16 uint into V3, which holds one 32-bit element a lane, read back raw.
  tbuffer_store_format_xy v[38:39], v0, s[20:23], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_UINT] idxen
  s_waitcnt vmcnt(0)
  buffer_load_dword v37, v0, s[20:23], 0 idxen
  s_waitcnt vmcnt(0)
  global_store_dword v1, v37, s[4:5] offset:2816
  s_endpgm
.Lformats_end:
  .size formats, .Lformats_end-formats

  .rodata
  .p2align 6
  .amdhsa_kernel images
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 72
    .amdhsa_next_free_sgpr 80
    .amdhsa_accum_offset 72
  .end_amdhsa_kernel

  .p2align 6
  .amdhsa_kernel formats
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 40
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 40
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: images
    .symbol: images.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 80
    .vgpr_count: 72
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: formats
    .symbol: formats.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
