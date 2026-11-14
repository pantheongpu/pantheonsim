// Six LDS instructions at one stride, for the bank-conflict counter: lane l addresses LDS byte (l * stride) % 65536, and the
// kernel issues, once each and in this order, ds_write_b32, ds_read_b32, ds_read_b64, ds_read_b128, ds_add_u32 and
// ds_read2_b32 (offset0:0 offset1:1). Built for gfx942 by build.sh, with clang's assembler.
//
// banks(stride): one wave of 64. What a card counts for each of these at a given stride is in
// amd/tests/data/lds_bank_conflicts.tsv (SQC_LDS_BANK_CONFLICT read from an RX 6800 and an RX 6700 XT); a wave64 is two wave32 halves,
// so the kernel's total is the sum over the six of twice the card's wave32 count.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl banks
  .p2align 8
  .type banks,@function
banks:
  s_load_dword s4, s[0:1], 0x0                 // the stride, in bytes
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0                     // the lane
  v_mul_u32_u24_e32 v1, s4, v0                 // lane * stride
  v_and_b32_e32 v1, 0xffff, v1                 // wrapped into the 64 KiB of LDS
  ds_write_b32 v1, v0
  s_waitcnt lgkmcnt(0)
  ds_read_b32 v2, v1
  ds_read_b64 v[4:5], v1
  ds_read_b128 v[8:11], v1
  ds_add_u32 v1, v0
  ds_read2_b32 v[12:13], v1 offset0:0 offset1:1
  s_waitcnt lgkmcnt(0)
  s_endpgm
.Lbanks_end:
  .size banks, .Lbanks_end-banks

  .rodata
  .p2align 6
  .amdhsa_kernel banks
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 16
    .amdhsa_group_segment_fixed_size 65536
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: banks
    .symbol: banks.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 4, .offset: 0, .value_kind: by_value }
...
  .end_amdgpu_metadata
