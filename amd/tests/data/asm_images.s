// RDNA's image instructions and formatted buffer loads, on resources the
// test writes itself (vgpu/amd_image.hpp): loads, a store and an atomic,
// point and bilinear samples, a sample at an explicit level between two mip
// levels, a gather, the resource query, and loads through a buffer resource's
// format and through MTBUF's own. Built for gfx1100 by build.sh, with clang's
// assembler.
//
// images(uint* out, uint* desc): desc holds, in dwords,
//    0 T0: an 8x4 32-bit float image
//    8 T1: a 4x2 8_8_8_8 unorm image
//   16 T2: an 8x4 32-bit uint image, stored to and added to
//   24 S#: point, clamp to the last texel;  28 S#: the same, bilinear
//   32 V#: 8_8_8_8 unorm elements, 4 bytes apart
//   36 T3: a 4x4 float image with two levels;  44 S#: linear between levels
// One wave of 32 lanes; lane l is texel (l % 8, l / 8) of T0. out gets, in
// words: 0 the loads; 32 point samples at texel centers; 64 bilinear samples
// on the left edge of each texel; 96 T1's four channels (4 a lane); 224 T3's
// size at level l % 2 (4 a lane); 352 gathers at each texel's top-right
// corner (4 a lane); 480 the buffer's four channels at index l (4 a lane);
// 608 loads past the right edge; 640 loads with 16-bit coordinates; 672
// samples of T3 at LOD l / 10; 704 T1's red and green as halves; 736 what
// the atomic add found; 768 MTBUF loads of the buffer as 32-bit floats.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .text
  .globl images
  .p2align 8
  .type images,@function
images:
  s_load_b128 s[0:3], s[0:1], 0x0          // s[0:1] out, s[2:3] desc
  s_waitcnt lgkmcnt(0)
  s_load_b256 s[8:15], s[2:3], 0x0
  s_load_b256 s[16:23], s[2:3], 0x20
  s_load_b256 s[24:31], s[2:3], 0x40
  s_load_b256 s[32:39], s[2:3], 0x60
  s_load_b128 s[40:43], s[2:3], 0x80
  s_load_b256 s[44:51], s[2:3], 0x90
  s_load_b128 s[52:55], s[2:3], 0xb0
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v1, 7, v0                   // x
  v_lshrrev_b32_e32 v2, 3, v0               // y
  v_cvt_f32_u32_e32 v3, v1
  v_mov_b32_e32 v9, 0x3d800000              // 1/16
  v_fma_f32 v3, v3, 0x3e000000, v9          // (x + 0.5) / 8
  v_cvt_f32_u32_e32 v4, v2
  v_mov_b32_e32 v9, 0x3e000000              // 1/8
  v_fma_f32 v4, v4, 0x3e800000, v9          // (y + 0.5) / 4
  v_cvt_f32_u32_e32 v5, v1
  v_mul_f32_e32 v5, 0x3e000000, v5          // x / 8
  v_lshlrev_b32_e32 v6, 2, v0               // l * 4
  v_lshlrev_b32_e32 v7, 4, v0               // l * 16

  image_load v10, v[1:2], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D
  image_sample_lz v11, v[3:4], s[8:15], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D
  v_mov_b32_e32 v8, v4
  v_mov_b32_e32 v12, v5
  v_mov_b32_e32 v13, v8
  image_sample_lz v12, v[12:13], s[8:15], s[36:39] dmask:0x1 dim:SQ_RSRC_IMG_2D
  s_waitcnt vmcnt(0)
  global_store_b32 v6, v10, s[0:1]
  global_store_b32 v6, v11, s[0:1] offset:128
  global_store_b32 v6, v12, s[0:1] offset:256

  v_and_b32_e32 v16, 3, v0
  v_bfe_u32 v17, v0, 2, 1
  image_load v[20:23], v[16:17], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D
  v_and_b32_e32 v15, 1, v0
  image_get_resinfo v[24:27], v15, s[44:51] dmask:0xf dim:SQ_RSRC_IMG_2D
  s_waitcnt vmcnt(0)
  global_store_b128 v7, v[20:23], s[0:1] offset:384
  global_store_b128 v7, v[24:27], s[0:1] offset:896

  v_add_nc_u32_e32 v18, 1, v1
  v_cvt_f32_u32_e32 v18, v18
  v_mul_f32_e32 v18, 0x3e000000, v18        // (x + 1) / 8
  v_add_nc_u32_e32 v19, 1, v2
  v_cvt_f32_u32_e32 v19, v19
  v_mul_f32_e32 v19, 0x3e800000, v19        // (y + 1) / 4
  image_gather4_lz v[20:23], v[18:19], s[8:15], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D
  buffer_load_format_xyzw v[24:27], v0, s[40:43], 0 idxen
  s_waitcnt vmcnt(0)
  global_store_b128 v7, v[20:23], s[0:1] offset:1408
  global_store_b128 v7, v[24:27], s[0:1] offset:1920

  v_add_nc_u32_e32 v28, 100, v1
  v_mov_b32_e32 v29, v2
  image_load v30, v[28:29], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D
  v_lshlrev_b32_e32 v31, 16, v2
  v_or_b32_e32 v31, v31, v1                  // x | y << 16
  image_load v32, v31, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D a16
  v_cvt_f32_u32_e32 v35, v0
  v_mul_f32_e32 v35, 0x3dcccccd, v35        // l / 10
  v_mov_b32_e32 v33, 0x3f000000
  v_mov_b32_e32 v34, 0x3f000000
  image_sample_l v36, v[33:35], s[44:51], s[52:55] dmask:0x1 dim:SQ_RSRC_IMG_2D
  image_load v37, v[16:17], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D d16
  s_waitcnt vmcnt(0)
  global_store_b32 v6, v30, s[0:1] offset:2432
  global_store_b32 v6, v32, s[0:1] offset:2560
  global_store_b32 v6, v36, s[0:1] offset:2688
  global_store_b32 v6, v37, s[0:1] offset:2816

  v_mul_u32_u24_e32 v38, 3, v0
  image_store v38, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D
  s_waitcnt_vscnt null, 0x0
  v_mov_b32_e32 v39, 5
  image_atomic_add v39, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D glc
  tbuffer_load_format_x v40, v0, s[40:43], 0 format:[BUF_FMT_32_FLOAT] idxen
  s_waitcnt vmcnt(0)
  global_store_b32 v6, v39, s[0:1] offset:2944
  global_store_b32 v6, v40, s[0:1] offset:3072
  s_endpgm
.Limages_end:
  .size images, .Limages_end-images

  .rodata
  .p2align 6
  .amdhsa_kernel images
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 41
    .amdhsa_next_free_sgpr 56
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
    .wavefront_size: 32
    .sgpr_count: 56
    .vgpr_count: 41
    .max_flat_workgroup_size: 32
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
