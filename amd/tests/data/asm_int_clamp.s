// The clamp bit of the integer instructions (test_amd_gcn_int_clamp): the VOP3b adds and subtracts with a carry out (v_add_co_u32 and its
// kin), v_mad_u64_u32 and v_mad_i64_i32, and the VOP3 forms of the other integer instructions that take a clamp. The CDNA3 and CDNA4
// guides (6.5, "For integer operations, it clamps the result to the largest and smallest representable value", and "Supports saturation"
// on each add and subtract) and an RX 6800 (gfx1030) agree: clamped, the result saturates to the range of its type; the carry out is
// the unclamped instruction's (the carry or borrow of an add or subtract, bit 64 of the 65-bit result of a 64-bit multiply-add). LLVM
// selects these for saturating arithmetic (uaddsat on gfx8 is v_add_u32_e64 ... clamp, the VOP3b form).
//
// int_clamp(in, out): work-item t (global id) reads eight dwords at in + 32 t: a, b, c64 (two dwords), c32, cin. Each instruction below
// writes three dwords at out + 12 (N op + t), N the work-item count (512): the result's low dword, its high dword (zero for a 32-bit
// result), and the carry out as 0 or 1 (zero for an instruction without one). Before each instruction the destination is zeroed, so a
// 16-bit result's other half is whatever the instruction leaves there; the test compares only the low half of those.
//
//   0 add_co clamp   1 sub_co clamp   2 subrev_co clamp   3 addc clamp   4 subb clamp   5 subbrev clamp
//   6 mad_u64_u32 clamp   7 mad_i64_i32 clamp   8 mad_u64_u32   9 mad_i64_i32   10 add_co   11 sub_co
//   12 add_u32 clamp   13 sub_u32 clamp   14 subrev_u32 clamp   15 add_i32 clamp   16 sub_i32 clamp
//   17 add_u16 clamp   18 sub_u16 clamp   19 subrev_u16 clamp (gfx10: sub_u16 b, a)   20 add_i16 clamp   21 sub_i16 clamp
//   22 pk_add_u16 clamp   23 pk_sub_u16 clamp   24 pk_add_i16 clamp   25 pk_sub_i16 clamp   26 pk_mad_u16 clamp   27 pk_mad_i16 clamp
//   28 mad_u32_u24 clamp   29 mad_i32_i24 clamp   30 mad_u16 clamp   31 mad_i16 clamp   32 mad_u32_u16 clamp   33 mad_i32_i16 clamp
//   34 mul_u32_u24 clamp   35 mul_i32_i24 clamp   36 sad_u32 clamp   37 sad_u16 clamp   38 sad_u8 clamp   39 msad_u8 clamp
//   40 sad_hi_u8 clamp   41 mul_lo_u16 clamp (gfx9; gfx10 has no clamp on it: unclamped)
//   42 mad_legacy_u16 clamp (gfx9; gfx10: mad_u16 clamp)   43 mad_legacy_i16 clamp (gfx9; gfx10: mad_i16 clamp)
//
// Assembled for gfx90a, gfx942 and gfx1030 (wave32) by build.sh (clang's assembler with the C preprocessor: GFX90A, GFX942 or GFX1030 is
// defined). The gfx1030 object was run on an RX 6800 and gave what the test expects.
#ifdef GFX1030
#define SD s20
#define CI s22
#define ZERO_SD s_mov_b32 s20, 0
#define ADD_U32 v_add_nc_u32_e64
#define SUB_U32 v_sub_nc_u32_e64
#define SUBREV_U32 v_subrev_nc_u32_e64
#define ADD_I32 v_add_nc_i32
#define SUB_I32 v_sub_nc_i32
#define ADD_U16 v_add_nc_u16
#define SUB_U16 v_sub_nc_u16
#define ADD_I16 v_add_nc_i16
#define SUB_I16 v_sub_nc_i16
#define ADDC v_add_co_ci_u32_e64
#define SUBB v_sub_co_ci_u32_e64
#define SUBBREV v_subrev_co_ci_u32_e64
#define VADD v_add_nc_u32_e32
#else
#define SD s[20:21]
#define CI s[22:23]
#define ZERO_SD s_mov_b64 s[20:21], 0
#define ADD_U32 v_add_u32_e64
#define SUB_U32 v_sub_u32_e64
#define SUBREV_U32 v_subrev_u32_e64
#define ADD_I32 v_add_i32
#define SUB_I32 v_sub_i32
#define ADD_U16 v_add_u16_e64
#define SUB_U16 v_sub_u16_e64
#define ADD_I16 v_add_i16
#define SUB_I16 v_sub_i16
#define ADDC v_addc_co_u32_e64
#define SUBB v_subb_co_u32_e64
#define SUBBREV v_subbrev_co_u32_e64
#define VADD v_add_u32_e32
#endif

.macro next
  v_cndmask_b32_e64 v12, 0, 1, SD
  global_store_dwordx3 v8, v[10:12], s[8:9]
  s_add_u32 s8, s8, 6144
  s_addc_u32 s9, s9, 0
  v_mov_b32_e32 v10, 0
  v_mov_b32_e32 v11, 0
  ZERO_SD
.endm

  .text
  .globl int_clamp
  .p2align 8
  .type int_clamp,@function
int_clamp:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[8:9], s[0:1], 0x8          // s[8:9] = out
  s_lshl_b32 s3, s2, 6                         // the group's first work-item (64 a group)
  VADD v0, s3, v0                              // t
  v_lshlrev_b32_e32 v1, 5, v0                  // 32 t
  v_mul_u32_u24_e32 v8, 12, v0                 // 12 t
  s_waitcnt lgkmcnt(0)
  global_load_dwordx4 v[2:5], v1, s[4:5]       // a, b, c64
  global_load_dwordx2 v[6:7], v1, s[4:5] offset:16   // c32, cin
  v_mov_b32_e32 v10, 0
  v_mov_b32_e32 v11, 0
  ZERO_SD
  s_waitcnt vmcnt(0)
  v_cmp_ne_u32_e64 CI, 0, v7

  v_add_co_u32_e64 v10, SD, v2, v3 clamp
  next
  v_sub_co_u32_e64 v10, SD, v2, v3 clamp
  next
  v_subrev_co_u32_e64 v10, SD, v2, v3 clamp
  next
  ADDC v10, SD, v2, v3, CI clamp
  next
  SUBB v10, SD, v2, v3, CI clamp
  next
  SUBBREV v10, SD, v2, v3, CI clamp
  next
  v_mad_u64_u32 v[10:11], SD, v2, v3, v[4:5] clamp
  next
  v_mad_i64_i32 v[10:11], SD, v2, v3, v[4:5] clamp
  next
  v_mad_u64_u32 v[10:11], SD, v2, v3, v[4:5]
  next
  v_mad_i64_i32 v[10:11], SD, v2, v3, v[4:5]
  next
  v_add_co_u32_e64 v10, SD, v2, v3
  next
  v_sub_co_u32_e64 v10, SD, v2, v3
  next
  ADD_U32 v10, v2, v3 clamp
  next
  SUB_U32 v10, v2, v3 clamp
  next
  SUBREV_U32 v10, v2, v3 clamp
  next
  ADD_I32 v10, v2, v3 clamp
  next
  SUB_I32 v10, v2, v3 clamp
  next
  ADD_U16 v10, v2, v3 clamp
  next
  SUB_U16 v10, v2, v3 clamp
  next
#ifdef GFX1030
  SUB_U16 v10, v3, v2 clamp
#else
  v_subrev_u16_e64 v10, v2, v3 clamp
#endif
  next
  ADD_I16 v10, v2, v3 clamp
  next
  SUB_I16 v10, v2, v3 clamp
  next
  v_pk_add_u16 v10, v2, v3 clamp
  next
  v_pk_sub_u16 v10, v2, v3 clamp
  next
  v_pk_add_i16 v10, v2, v3 clamp
  next
  v_pk_sub_i16 v10, v2, v3 clamp
  next
  v_pk_mad_u16 v10, v2, v3, v6 clamp
  next
  v_pk_mad_i16 v10, v2, v3, v6 clamp
  next
  v_mad_u32_u24 v10, v2, v3, v6 clamp
  next
  v_mad_i32_i24 v10, v2, v3, v6 clamp
  next
  v_mad_u16 v10, v2, v3, v6 clamp
  next
  v_mad_i16 v10, v2, v3, v6 clamp
  next
  v_mad_u32_u16 v10, v2, v3, v6 clamp
  next
  v_mad_i32_i16 v10, v2, v3, v6 clamp
  next
  v_mul_u32_u24_e64 v10, v2, v3 clamp
  next
  v_mul_i32_i24_e64 v10, v2, v3 clamp
  next
  v_sad_u32 v10, v2, v3, v6 clamp
  next
  v_sad_u16 v10, v2, v3, v6 clamp
  next
  v_sad_u8 v10, v2, v3, v6 clamp
  next
  v_msad_u8 v10, v2, v3, v6 clamp
  next
  v_sad_hi_u8 v10, v2, v3, v6 clamp
  next
#ifdef GFX1030
  v_mul_lo_u16 v10, v2, v3
  next
  v_mad_u16 v10, v2, v3, v6 clamp
  next
  v_mad_i16 v10, v2, v3, v6 clamp
  next
#else
  v_mul_lo_u16 v10, v2, v3 clamp
  next
  v_mad_legacy_u16 v10, v2, v3, v6 clamp
  next
  v_mad_legacy_i16 v10, v2, v3, v6 clamp
  next
#endif
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel int_clamp
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 32
#ifdef GFX1030
    .amdhsa_wavefront_size32 1
#else
    .amdhsa_accum_offset 16
#endif
    .amdhsa_float_denorm_mode_32 3               // denormals kept, as the compiler's kernels have it (the tier runs only that mode)
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: int_clamp
    .symbol: int_clamp.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
#ifdef GFX1030
    .wavefront_size: 32
#else
    .wavefront_size: 64
#endif
    .sgpr_count: 32
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
