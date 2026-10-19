// gfx940's additions: packed half and bfloat16 atomics in LDS (DS_PK_ADD_F16, DS_PK_ADD_RTN_F16, DS_PK_ADD_BF16, DS_PK_ADD_RTN_BF16) and in
// a buffer (BUFFER_ATOMIC_PK_ADD_F16; gfx950's BUFFER_ATOMIC_PK_ADD_BF16): "dst[31:16].f16 = tmp[31:16].f16 + src[31:16].f16; dst[15:0].f16 =
// ...", the old value back from the return forms; and the scratch loads straight into LDS (SCRATCH_LOAD_LDS_*: the same layout as the global ones).
//
// pk_atomics(buf, vals, out): lane L's four dwords of buf start LDS; vals[2 L] is the packed halves to add and vals[2 L + 1] the packed bfloat16s;
// buf + 1024 holds a dword a lane (and buf + 1280 another) for the buffer atomics. out: the LDS at 16 L, then 2048 + 8 L the two old values
// the _RTN forms returned, then 4096 + 4 L and + 256 the old values of the buffer atomics.
// scratch_lds(out): each lane stores a dword, a byte and a half in its private memory and loads them into LDS, as dwords.
//
// Assembled for gfx90a, gfx942 and gfx950 by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX950 is
// defined, and RET is how the target spells "return the old value": glc on gfx90a, sc0 after it).
#if defined(GFX90A)
#define RET glc
#else
#define RET sc0
#endif
  .text
#if !defined(GFX90A)
  .globl pk_atomics
  .p2align 8
  .type pk_atomics,@function
pk_atomics:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = buf, s[6:7] = vals
  s_load_dwordx2 s[12:13], s[0:1], 0x10       // out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 4, v0                  // four dwords of LDS, of buf and of out a lane
  v_lshlrev_b32_e32 v2, 3, v0                  // two dwords of vals a lane
  global_load_dwordx4 v[4:7], v1, s[4:5]
  global_load_dwordx2 v[8:9], v2, s[6:7]
  s_waitcnt vmcnt(0)
  ds_write_b128 v1, v[4:7]
  s_waitcnt lgkmcnt(0)
  ds_pk_add_f16 v1, v8 offset:0
  ds_pk_add_rtn_f16 v10, v1, v8 offset:4
  ds_pk_add_bf16 v1, v9 offset:8
  ds_pk_add_rtn_bf16 v11, v1, v9 offset:12
  s_waitcnt lgkmcnt(0)
  ds_read_b128 v[14:17], v1
  s_waitcnt lgkmcnt(0)
  s_add_u32 s14, s12, 2048                     // out: the old values at 32 B a lane, then LDS afterwards
  s_addc_u32 s15, s13, 0
  global_store_dwordx4 v1, v[14:17], s[12:13]
  v_lshlrev_b32_e32 v3, 3, v0
  global_store_dword v3, v10, s[14:15] offset:0
  global_store_dword v3, v11, s[14:15] offset:4
  s_add_u32 s16, s4, 0x400
  s_addc_u32 s17, s5, 0
  s_mov_b32 s8, s16
  s_mov_b32 s9, s17
  s_mov_b32 s10, 1024
  s_mov_b32 s11, 0x27000
  v_lshlrev_b32_e32 v18, 2, v0                 // a dword a lane in the resource
  v_mov_b32_e32 v19, v8
  buffer_atomic_pk_add_f16 v19, v18, s[8:11], 0 offen RET
#if defined(GFX950)
  s_add_u32 s18, s4, 0x500
  s_addc_u32 s19, s5, 0
  s_mov_b32 s20, s18
  s_mov_b32 s21, s19
  s_mov_b32 s22, 1024
  s_mov_b32 s23, 0x27000
  v_mov_b32_e32 v20, v9
  buffer_atomic_pk_add_bf16 v20, v18, s[8:11], 0 offen offset:256 RET
#else
  v_mov_b32_e32 v20, 0
#endif
  s_waitcnt vmcnt(0)
  s_add_u32 s24, s12, 4096
  s_addc_u32 s25, s13, 0
  global_store_dword v18, v19, s[24:25] offset:0
  global_store_dword v18, v20, s[24:25] offset:256
  s_waitcnt vmcnt(0)
  s_endpgm
.Lpk_atomics_end:
  .size pk_atomics, .Lpk_atomics_end-pk_atomics
#endif
#if !defined(GFX90A)
  .globl scratch_lds
  .p2align 8
  .type scratch_lds,@function
scratch_lds:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 2, v0
  v_mul_u32_u24_e32 v2, 7, v0
  v_add_u32_e32 v2, 1, v2                      // a dword a lane
  v_add_u32_e32 v3, 0x80, v0                   // a byte, with its top bit set
  v_add_u32_e32 v4, 0x8000, v0                 // a half, with its top bit set
  v_lshlrev_b32_e32 v4, 0, v4
  scratch_store_dword off, v2, off offset:16
  scratch_store_byte off, v3, off offset:32
  scratch_store_short off, v4, off offset:48
  s_waitcnt vmcnt(0)
  s_mov_b32 m0, 0
  scratch_load_lds_dword off, off offset:16
  s_mov_b32 m0, 1024
  scratch_load_lds_ubyte off, off offset:32
  s_mov_b32 m0, 2048
  scratch_load_lds_sbyte off, off offset:32
  s_mov_b32 m0, 3072
  scratch_load_lds_ushort off, off offset:48
  s_mov_b32 m0, 4096
  scratch_load_lds_sshort off, off offset:48
  s_waitcnt vmcnt(0)
  ds_read_b32 v5, v1 offset:16
  s_waitcnt lgkmcnt(0)
  global_store_dword v1, v5, s[4:5] offset:0
  ds_read_b32 v5, v1 offset:1056
  s_waitcnt lgkmcnt(0)
  global_store_dword v1, v5, s[4:5] offset:256
  ds_read_b32 v5, v1 offset:2080
  s_waitcnt lgkmcnt(0)
  global_store_dword v1, v5, s[4:5] offset:512
  ds_read_b32 v5, v1 offset:3120
  s_waitcnt lgkmcnt(0)
  global_store_dword v1, v5, s[4:5] offset:768
  ds_read_b32 v5, v1 offset:4144
  s_waitcnt lgkmcnt(0)
  global_store_dword v1, v5, s[4:5] offset:1024
  s_waitcnt vmcnt(0)
  s_endpgm
.Lscratch_lds_end:
  .size scratch_lds, .Lscratch_lds_end-scratch_lds
#endif
  .rodata
#if !defined(GFX90A)
  .p2align 6
  .amdhsa_kernel pk_atomics
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 32
  .end_amdhsa_kernel
#endif
#if !defined(GFX90A)
  .p2align 6
  .amdhsa_kernel scratch_lds
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 256
    .amdhsa_enable_private_segment 1
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel
#endif
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
#if !defined(GFX90A)
  - .name: pk_atomics
    .symbol: pk_atomics.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 32
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
#endif
#if !defined(GFX90A)
  - .name: scratch_lds
    .symbol: scratch_lds.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 256
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
#endif
...
  .end_amdgpu_metadata

