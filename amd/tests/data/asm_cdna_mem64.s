// Buffer atomics of 64 bits and of doubles (CDNA2, CDNA3 and CDNA4 ISA, BUFFER_ATOMIC_SWAP_X2 .. BUFFER_ATOMIC_DEC_X2, BUFFER_ATOMIC_ADD_F64,
// _MIN_F64, _MAX_F64): each applies its operation to one 64-bit word of a 256-byte buffer, and the old value comes back in the data register.
//
// mem64(buf, vals, out): vals[i] is operation i's data (vals[16] the value cmpswap compares against), out[i] the old value it
// returned; the last two return the word at the resource's last 8 bytes (in range) and one a scalar offset puts past the end (zero,
// and nothing written).
//
// Assembled for gfx90a, gfx942 and gfx950 by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX950 is
// defined, and RET is how the target spells "return the old value": glc on gfx90a, sc0 after it).
#if defined(GFX90A)
#define RET glc
#else
#define RET sc0
#endif
  .text
  .globl mem64
  .p2align 8
  .type mem64,@function
mem64:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = buf, s[6:7] = vals
  s_load_dwordx2 s[12:13], s[0:1], 0x10       // out
  s_waitcnt lgkmcnt(0)
  s_load_dwordx16 s[16:31], s[6:7], 0x0       // vals[0..7]
  s_load_dwordx16 s[32:47], s[6:7], 0x40      // vals[8..15]
  s_load_dwordx4 s[48:51], s[6:7], 0x80       // vals[16], vals[17]
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1                            // one lane does every operation
  s_mov_b32 s8, s4
  s_mov_b32 s9, s5                             // the resource: base, stride 0,
  s_mov_b32 s10, 256                           // 256 bytes, a 32-bit data format
  s_mov_b32 s11, 0x27000
  v_mov_b32 v0, 0
  v_mov_b32 v2, s16
  v_mov_b32 v3, s17
  buffer_atomic_swap_x2 v[2:3], off, s[8:11], 0 offset:0 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:0
  v_mov_b32 v2, s18
  v_mov_b32 v3, s19
  buffer_atomic_add_x2 v[2:3], off, s[8:11], 0 offset:8 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:8
  v_mov_b32 v2, s20
  v_mov_b32 v3, s21
  buffer_atomic_sub_x2 v[2:3], off, s[8:11], 0 offset:16 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:16
  v_mov_b32 v2, s22
  v_mov_b32 v3, s23
  buffer_atomic_smin_x2 v[2:3], off, s[8:11], 0 offset:24 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:24
  v_mov_b32 v2, s24
  v_mov_b32 v3, s25
  buffer_atomic_umin_x2 v[2:3], off, s[8:11], 0 offset:32 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:32
  v_mov_b32 v2, s26
  v_mov_b32 v3, s27
  buffer_atomic_smax_x2 v[2:3], off, s[8:11], 0 offset:40 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:40
  v_mov_b32 v2, s28
  v_mov_b32 v3, s29
  buffer_atomic_umax_x2 v[2:3], off, s[8:11], 0 offset:48 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:48
  v_mov_b32 v2, s30
  v_mov_b32 v3, s31
  buffer_atomic_and_x2 v[2:3], off, s[8:11], 0 offset:56 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:56
  v_mov_b32 v2, s32
  v_mov_b32 v3, s33
  buffer_atomic_or_x2 v[2:3], off, s[8:11], 0 offset:64 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:64
  v_mov_b32 v2, s34
  v_mov_b32 v3, s35
  buffer_atomic_xor_x2 v[2:3], off, s[8:11], 0 offset:72 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:72
  v_mov_b32 v2, s36
  v_mov_b32 v3, s37
  buffer_atomic_inc_x2 v[2:3], off, s[8:11], 0 offset:80 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:80
  v_mov_b32 v2, s38
  v_mov_b32 v3, s39
  buffer_atomic_dec_x2 v[2:3], off, s[8:11], 0 offset:88 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:88
  v_mov_b32 v2, s40
  v_mov_b32 v3, s41
  v_mov_b32 v4, s48
  v_mov_b32 v5, s49
  buffer_atomic_cmpswap_x2 v[2:5], off, s[8:11], 0 offset:96 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:96
  v_mov_b32 v2, s42
  v_mov_b32 v3, s43
  buffer_atomic_add_f64 v[2:3], off, s[8:11], 0 offset:104 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:104
  v_mov_b32 v2, s44
  v_mov_b32 v3, s45
  buffer_atomic_min_f64 v[2:3], off, s[8:11], 0 offset:112 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:112
  v_mov_b32 v2, s46
  v_mov_b32 v3, s47
  buffer_atomic_max_f64 v[2:3], off, s[8:11], 0 offset:120 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:120
  v_mov_b32 v2, 5
  v_mov_b32 v3, 0
  buffer_atomic_add_x2 v[2:3], off, s[8:11], 0 offset:248 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:128
  v_mov_b32 v2, 5
  v_mov_b32 v3, 0
  s_mov_b32 s20, 4
  buffer_atomic_add_x2 v[2:3], off, s[8:11], s20 offset:248 RET
  s_waitcnt vmcnt(0)
  global_store_dwordx2 v0, v[2:3], s[12:13] offset:136
  s_endpgm
.Lmem64_end:
  .size mem64, .Lmem64_end-mem64
  .rodata
  .p2align 6
  .amdhsa_kernel mem64
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel
  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: mem64
    .symbol: mem64.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata

