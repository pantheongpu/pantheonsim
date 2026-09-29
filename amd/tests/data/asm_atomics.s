// The 32-bit buffer atomics beyond add and swap -- sub, the signed and
// unsigned minimum and maximum, and, or, xor, and the wrapping increment and
// decrement -- each returning what the word held before. PyTorch's
// histogram (bincount) kernels reach for buffer_atomic_or. Built for gfx942
// by build.sh, with clang's assembler.
//
// atomics(int* buf, int* out, int x): one lane applies the ten, in the
// order above, with x to buf[0] .. buf[9], through a resource covering those
// ten words, and writes each old value to out[0] .. out[9].
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl atomics
  .p2align 8
  .type atomics,@function
atomics:
  s_load_dwordx4 s[4:7], s[0:1], 0x0          // s[4:5] = buf, s[6:7] = out
  s_load_dword s12, s[0:1], 0x10              // s12 = x
  s_waitcnt lgkmcnt(0)
  // The resource: buf, no stride, 40 bytes, a 32-bit data format.
  s_mov_b32 s8, s4
  s_and_b32 s9, s5, 0xffff
  s_mov_b32 s10, 40
  s_mov_b32 s11, 0x20000
  s_mov_b64 exec, 1                            // one lane
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, s12
  buffer_atomic_sub v2, v0, s[8:11], 0 offen offset:0 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:0
  v_mov_b32_e32 v2, s12
  buffer_atomic_smin v2, v0, s[8:11], 0 offen offset:4 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:4
  v_mov_b32_e32 v2, s12
  buffer_atomic_umin v2, v0, s[8:11], 0 offen offset:8 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:8
  v_mov_b32_e32 v2, s12
  buffer_atomic_smax v2, v0, s[8:11], 0 offen offset:12 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:12
  v_mov_b32_e32 v2, s12
  buffer_atomic_umax v2, v0, s[8:11], 0 offen offset:16 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:16
  v_mov_b32_e32 v2, s12
  buffer_atomic_and v2, v0, s[8:11], 0 offen offset:20 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:20
  v_mov_b32_e32 v2, s12
  buffer_atomic_or v2, v0, s[8:11], 0 offen offset:24 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:24
  v_mov_b32_e32 v2, s12
  buffer_atomic_xor v2, v0, s[8:11], 0 offen offset:28 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:28
  v_mov_b32_e32 v2, s12
  buffer_atomic_inc v2, v0, s[8:11], 0 offen offset:32 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:32
  v_mov_b32_e32 v2, s12
  buffer_atomic_dec v2, v0, s[8:11], 0 offen offset:36 sc0
  s_waitcnt vmcnt(0)
  global_store_dword v1, v2, s[6:7] offset:36
  s_endpgm
.Latomics_end:
  .size atomics, .Latomics_end-atomics

  .rodata
  .p2align 6
  .amdhsa_kernel atomics
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: atomics
    .symbol: atomics.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 3
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
      - .size: 4
        .offset: 16
        .value_kind: by_value
...
  .end_amdgpu_metadata
