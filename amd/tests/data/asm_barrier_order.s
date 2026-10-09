// The order a work-group's waves run in after a barrier. Thread 0 writes an LDS word and
// everyone passes a barrier; every thread then reads the word, and the wave of thread 0 goes on to
// a global load and, once it has arrived, overwrites the word. On a card the other wave reads
// within a few cycles of the barrier and the first wave overwrites after the load's hundreds, so
// both read what thread 0 wrote. rocPRIM's ordered block id is read and overwritten like this
// (its LDS storage is a union with the block loader's): reading after the writer had run on to the
// overwrite gave every other wave a garbage block id and a fault in test_device_scan.
// Built for gfx942 by build.sh, with clang's assembler.
//
// order(out): 128 threads, out[tid] = the word the thread read.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl order
  .p2align 8
  .type order,@function
order:
  s_load_dwordx2 s[4:5], s[0:1], 0x0           // out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v1, 0x3ff, v0                   // the thread
  v_cmp_eq_u32_e32 vcc, 0, v1
  s_and_saveexec_b64 s[6:7], vcc
  v_mov_b32_e32 v2, 0
  v_mov_b32_e32 v3, 7
  ds_write_b32 v2, v3                           // thread 0 writes 7
  s_waitcnt lgkmcnt(0)
  s_or_b64 exec, exec, s[6:7]
  s_barrier
  v_mov_b32_e32 v2, 0
  ds_read_b32 v4, v2                            // everyone reads it
  v_cmp_gt_u32_e32 vcc, 64, v1                  // the first wave only
  s_and_saveexec_b64 s[6:7], vcc
  v_lshlrev_b32_e32 v5, 2, v1
  global_load_dword v6, v5, s[4:5]              // goes on to a global load
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v7, 99
  ds_write_b32 v2, v7                           // and overwrites the word
  s_waitcnt lgkmcnt(0)
  s_or_b64 exec, exec, s[6:7]
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32_e32 v5, 2, v1
  global_store_dword v5, v4, s[4:5]
  s_endpgm
.Lorder_end:
  .size order, .Lorder_end-order

  .rodata
  .p2align 6
  .amdhsa_kernel order
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
    .amdhsa_group_segment_fixed_size 64
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: order
    .symbol: order.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 64
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 8
    .max_flat_workgroup_size: 128
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
