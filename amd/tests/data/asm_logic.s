// The rest of the scalar bitwise family, and the 64-bit bit-field forms and
// a signed difference: or with the second source inverted, nand, nor and
// xnor in both widths, a 64-bit mask made and a 64-bit field taken, and the
// absolute difference -- each with the SCC it sets. PyTorch's own kernels,
// built by a newer compiler, reach for s_nor_b64 in their scatter and
// gather. Built for gfx942 by build.sh, with clang's assembler.
//
// logic(int* out, long a, long b) writes 24 words, in the order the test
// lists them; the 32-bit forms take the low halves.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl logic
  .p2align 8
  .type logic,@function
logic:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_load_dwordx4 s[4:7], s[0:1], 0x8          // s[4:5] = a, s[6:7] = b
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1                            // one lane writes
  s_orn2_b32 s8, s4, s6
  s_cselect_b32 s9, 1, 0
  s_nand_b32 s10, s4, s6
  s_cselect_b32 s11, 1, 0
  s_nand_b64 s[12:13], s[4:5], s[6:7]
  s_cselect_b32 s14, 1, 0
  s_nor_b32 s15, s4, s6
  s_cselect_b32 s16, 1, 0
  s_nor_b64 s[18:19], s[4:5], s[6:7]
  s_cselect_b32 s17, 1, 0
  s_xnor_b32 s20, s4, s6
  s_cselect_b32 s21, 1, 0
  s_xnor_b64 s[22:23], s[4:5], s[6:7]
  s_cselect_b32 s24, 1, 0
  s_bfm_b64 s[26:27], s6, s4                   // b's low six bits of ones, shifted by a's
  s_bfe_u64 s[28:29], s[4:5], s6               // a field of a, placed as b says
  s_cselect_b32 s25, 1, 0
  s_absdiff_i32 s30, s4, s6
  s_cselect_b32 s31, 1, 0
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, s8
  global_store_dword v0, v1, s[2:3]
  v_mov_b32_e32 v1, s9
  global_store_dword v0, v1, s[2:3] offset:4
  v_mov_b32_e32 v1, s10
  global_store_dword v0, v1, s[2:3] offset:8
  v_mov_b32_e32 v1, s11
  global_store_dword v0, v1, s[2:3] offset:12
  v_mov_b32_e32 v1, s12
  global_store_dword v0, v1, s[2:3] offset:16
  v_mov_b32_e32 v1, s13
  global_store_dword v0, v1, s[2:3] offset:20
  v_mov_b32_e32 v1, s14
  global_store_dword v0, v1, s[2:3] offset:24
  v_mov_b32_e32 v1, s15
  global_store_dword v0, v1, s[2:3] offset:28
  v_mov_b32_e32 v1, s16
  global_store_dword v0, v1, s[2:3] offset:32
  v_mov_b32_e32 v1, s18
  global_store_dword v0, v1, s[2:3] offset:36
  v_mov_b32_e32 v1, s19
  global_store_dword v0, v1, s[2:3] offset:40
  v_mov_b32_e32 v1, s17
  global_store_dword v0, v1, s[2:3] offset:44
  v_mov_b32_e32 v1, s20
  global_store_dword v0, v1, s[2:3] offset:48
  v_mov_b32_e32 v1, s21
  global_store_dword v0, v1, s[2:3] offset:52
  v_mov_b32_e32 v1, s22
  global_store_dword v0, v1, s[2:3] offset:56
  v_mov_b32_e32 v1, s23
  global_store_dword v0, v1, s[2:3] offset:60
  v_mov_b32_e32 v1, s24
  global_store_dword v0, v1, s[2:3] offset:64
  v_mov_b32_e32 v1, s26
  global_store_dword v0, v1, s[2:3] offset:68
  v_mov_b32_e32 v1, s27
  global_store_dword v0, v1, s[2:3] offset:72
  v_mov_b32_e32 v1, s28
  global_store_dword v0, v1, s[2:3] offset:76
  v_mov_b32_e32 v1, s29
  global_store_dword v0, v1, s[2:3] offset:80
  v_mov_b32_e32 v1, s25
  global_store_dword v0, v1, s[2:3] offset:84
  v_mov_b32_e32 v1, s30
  global_store_dword v0, v1, s[2:3] offset:88
  v_mov_b32_e32 v1, s31
  global_store_dword v0, v1, s[2:3] offset:92
  s_endpgm
.Llogic_end:
  .size logic, .Llogic_end-logic

  .rodata
  .p2align 6
  .amdhsa_kernel logic
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: logic
    .symbol: logic.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 2
    .max_flat_workgroup_size: 64
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .size: 8
        .offset: 8
        .value_kind: by_value
      - .size: 8
        .offset: 16
        .value_kind: by_value
...
  .end_amdgpu_metadata
