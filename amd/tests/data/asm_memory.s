// Memory reached the ways rocBLAS's kernels and Tensile's GEMMs reach it:
// through a buffer resource, whose bounds make a load past the end read zero
// and a store past it go nowhere; LDS in 64-bit pairs, 64 values apart, and
// read from far past its end (zero, as a card gives it, which Tensile uses
// to clear registers); a scalar load whose offset is a register; a lane
// sending its value to another; and a work-item's private memory through
// its flat aperture, read back as scratch -- once by offset alone, once
// from a scalar register's offset; and loads into half a register. Built for gfx942 by build.sh, with
// clang's assembler.
//
// memory(int* buf, int* out): buf holds eight words, of which the resource
// covers the first four; out gets 19 words, in the order the test lists.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl memory
  .p2align 8
  .type memory,@function
memory:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = buf, s[6:7] = out
  s_waitcnt lgkmcnt(0)
  // The resource: buf, no stride, 16 bytes, a 32-bit data format.
  s_mov_b32 s8, s4
  s_and_b32 s9, s5, 0xffff
  s_mov_b32 s10, 16
  s_mov_b32 s11, 0x20000
  // Every lane sends ten times its number to the lane after it.
  v_mbcnt_lo_u32_b32 v0, -1, 0
  v_mbcnt_hi_u32_b32 v0, -1, v0
  v_add_u32_e32 v1, 1, v0
  v_and_b32_e32 v1, 63, v1
  v_lshlrev_b32_e32 v1, 2, v1
  v_mul_u32_u24_e32 v2, 10, v0
  ds_permute_b32 v3, v1, v2
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1                            // from here, one lane
  v_mov_b32_e32 v10, 0
  global_store_dword v10, v3, s[6:7]
  // Loads: in range, past the end, and a pair half past it.
  v_mov_b32_e32 v4, 4
  buffer_load_dword v5, v4, s[8:11], 0 offen
  v_mov_b32_e32 v4, 16
  buffer_load_dword v6, v4, s[8:11], 0 offen
  v_mov_b32_e32 v4, 12
  buffer_load_dwordx2 v[8:9], v4, s[8:11], 0 offen
  s_waitcnt vmcnt(0)
  global_store_dword v10, v5, s[6:7] offset:4
  global_store_dword v10, v6, s[6:7] offset:8
  global_store_dword v10, v8, s[6:7] offset:12
  global_store_dword v10, v9, s[6:7] offset:16
  // A store in range lands; one past the end does not.
  v_mov_b32_e32 v31, 0x77
  v_mov_b32_e32 v4, 4
  buffer_store_dword v31, v4, s[8:11], 0 offen offset:4
  v_mov_b32_e32 v4, 16
  buffer_store_dword v31, v4, s[8:11], 0 offen
  // The scalar offset counts against the bounds as the rest does: 4 on 8
  // is in range, 12 on 4 is not.
  s_mov_b32 s12, 4
  v_mov_b32_e32 v4, 8
  buffer_load_dword v11, v4, s[8:11], s12 offen
  s_mov_b32 s12, 12
  v_mov_b32_e32 v4, 4
  buffer_load_dword v29, v4, s[8:11], s12 offen
  s_waitcnt vmcnt(0)
  global_store_dword v10, v11, s[6:7] offset:20
  global_store_dword v10, v29, s[6:7] offset:56
  // A scalar load whose offset is a register.
  s_mov_b32 s13, 24
  s_load_dword s14, s[4:5], s13
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v12, s14
  global_store_dword v10, v12, s[6:7] offset:24
  // Two pairs 512 bytes apart, written as 64-pair strides and read back in
  // pair units; then a read far past LDS, over a register that held 5.
  v_mov_b32_e32 v13, 0
  v_mov_b32_e32 v14, 0x11
  v_mov_b32_e32 v15, 0x22
  v_mov_b32_e32 v16, 0x33
  v_mov_b32_e32 v17, 0x44
  ds_write2st64_b64 v13, v[14:15], v[16:17] offset1:1
  ds_read2_b64 v[18:21], v13 offset1:64
  v_mov_b32_e32 v22, 0xf00000
  v_mov_b32_e32 v23, 5
  ds_read_b32 v23, v22
  s_waitcnt lgkmcnt(0)
  global_store_dword v10, v18, s[6:7] offset:28
  global_store_dword v10, v19, s[6:7] offset:32
  global_store_dword v10, v20, s[6:7] offset:36
  global_store_dword v10, v21, s[6:7] offset:40
  global_store_dword v10, v23, s[6:7] offset:44
  // Private memory: stored through the flat aperture, loaded as scratch.
  s_mov_b64 s[16:17], src_private_base
  v_mov_b32_e32 v24, 8
  v_mov_b32_e32 v25, s17
  v_mov_b32_e32 v26, 0x1234
  flat_store_dword v[24:25], v26
  s_waitcnt vmcnt(0) lgkmcnt(0)
  scratch_load_dword v27, off, off offset:8
  s_mov_b32 s18, 4
  scratch_load_dword v28, off, s18 offset:4
  s_waitcnt vmcnt(0)
  global_store_dword v10, v27, s[6:7] offset:48
  global_store_dword v10, v28, s[6:7] offset:52
  // Half-register loads, each over a register that held 0xdeadbeef: the
  // half they load, and the other cleared, as on a card with SRAM ECC.
  v_mov_b32_e32 v5, 0xdeadbeef
  v_mov_b32_e32 v6, 0xdeadbeef
  v_mov_b32_e32 v8, 0xdeadbeef
  v_mov_b32_e32 v9, 0xdeadbeef
  v_mov_b32_e32 v4, 4
  buffer_load_short_d16 v5, v4, s[8:11], 0 offen
  buffer_load_short_d16_hi v6, v4, s[8:11], 0 offen offset:8
  buffer_load_short_d16 v9, v4, s[8:11], 0 offen offset:12   // past the end
  ds_read_u16_d16_hi v8, v13
  s_waitcnt vmcnt(0) lgkmcnt(0)
  global_store_dword v10, v5, s[6:7] offset:60
  global_store_dword v10, v6, s[6:7] offset:64
  global_store_dword v10, v8, s[6:7] offset:68
  global_store_dword v10, v9, s[6:7] offset:72
  s_endpgm
.Lmemory_end:
  .size memory, .Lmemory_end-memory

// kernarg_tail(int* out): a kernel whose arguments are one pointer, 8
// bytes, that loads 32 bytes of them at once, as the compiler does when it
// widens a load of the last arguments past the segment's end (rocBLAS's
// rotmg). A runtime has to give it memory there to read; it writes 1.
  .globl kernarg_tail
  .p2align 8
  .type kernarg_tail,@function
kernarg_tail:
  s_load_dwordx8 s[4:11], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, 1
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 1
  global_store_dword v0, v1, s[4:5]
  s_endpgm
.Lkernarg_tail_end:
  .size kernarg_tail, .Lkernarg_tail_end-kernarg_tail

  .rodata
  .p2align 6
  .amdhsa_kernel kernarg_tail
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel

  .p2align 6
  .amdhsa_kernel memory
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 16
    .amdhsa_enable_private_segment 1
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 32
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: memory
    .symbol: memory.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 16
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 32
    .max_flat_workgroup_size: 64
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: kernarg_tail
    .symbol: kernarg_tail.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 2
    .max_flat_workgroup_size: 64
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
...
  .end_amdgpu_metadata
