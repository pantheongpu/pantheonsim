# The handlers' time by instruction form (round sp74)

`scripts/diagpatch.py` makes a build in which `tier_step` reads the host's time-stamp counter before and after each handler and adds
the difference, the calls, the lanes that were on and the hand-backs to an entry for the instruction's form (its name and its operand
kinds: `v` a register, `vw` a register pair, `s` and `sw` a scalar register and a pair, `c` a constant, `?` VCC or EXEC, `-` negated, `|`
absolute value); `SP74_DIAG=<file>` writes the entries at exit and `scripts/diagtab.py` prints them. One host thread, knob off (the
blocks' loop is not instrumented). **Ticks a call include about 25 of the measurement's own** (the smallest handlers show 25 to 31), so
a share here is of the handlers' ticks with that in them, not of the test's time: the tables order the forms. The base was measured with
nothing else running; the final sources while builds ran beside it, which adds a few ticks to every call.

## steepest_descent, the base: 6.44e+10 ticks in the handlers, 1.19 billion instructions

| share of the handlers' ticks | ticks a call | calls (millions) | lanes on | handed back | form | handler |
|---:|---:|---:|---:|---:|---|---|
| 15.2 % | 331 | 29.6 | 61.1 | 0 | `v_max_f64 vw vw` | |
| 5.1 % | 79 | 42.0 | 64.0 | 0 | `ds_bpermute_b32 v v` | |
| 5.0 % | 236 | 13.7 | 64.0 | 0 | `v_or_b32_sdwa vsel0 v` | |
| 5.0 % | 352 | 9.1 | 64.0 | 0 | `v_max_f64 \|vw \|vw` | |
| 3.1 % | 26 | 78.0 | 62.6 | 0 | `s_waitcnt` | |
| 2.9 % | 75 | 25.2 | 64.0 | 0 | `v_cmp_lt_i64_e32 sw vw` | |
| 2.9 % | 35 | 52.4 | 63.0 | 0 | `v_and_b32_e32 c v` | |
| 2.4 % | 166 | 9.2 | 63.9 | 0 | `v_cmp_neq_f64_e32 c vw` | |
| 2.1 % | 26 | 53.4 | 6.4 | 0 | `s_or_b64 ? sw` | |
| 2.0 % | 562 | 2.3 | 63.9 | 0 | `v_frexp_exp_i32_f64_e32 vw` | |
| 1.8 % | 50 | 22.9 | 60.3 | 0 | `v_addc_co_u32_e32 v v ?w` | |
| 1.6 % | 26 | 38.8 | 62.1 | 0 | `s_and_saveexec_b64 ?w` | |
| 1.5 % | 25 | 39.1 | 64.0 | 0 | `s_lshr_b32 s c` | |
| 1.5 % | 41 | 23.0 | 64.0 | 0 | `v_cmp_gt_u32_e32 s v` | |
| 1.4 % | 25 | 36.0 | 9.4 | 0 | `s_cbranch_execz` | |
| 1.4 % | 41 | 21.8 | 53.2 | 0 | `v_cndmask_b32_e32 c v ?w` | |

## steepest_descent, the final sources: 4.81e+10 ticks in the handlers, 1.19 billion instructions

| share of the handlers' ticks | ticks a call | calls (millions) | lanes on | handed back | form | handler |
|---:|---:|---:|---:|---:|---|---|
| 6.7 % | 76 | 42.0 | 64.0 | 0 | `ds_bpermute_b32 v v` | h_ds_avx2 |
| 5.8 % | 94 | 29.6 | 61.1 | 0 | `v_max_f64 vw vw` | h_dmax_rows |
| 4.2 % | 26 | 78.0 | 62.6 | 0 | `s_waitcnt` | b_waitcnt |
| 3.7 % | 34 | 52.4 | 63.0 | 0 | `v_and_b32_e32 c v` | h_int_avx2 |
| 2.9 % | 26 | 53.4 | 6.4 | 0 | `s_or_b64 ? sw` | s_or_b64 |
| 2.5 % | 48 | 25.2 | 64.0 | 0 | `v_cmp_lt_i64_e32 sw vw` | h_cmp64_rows |
| 2.4 % | 51 | 22.9 | 60.3 | 0 | `v_addc_co_u32_e32 v v ?w` | h_carry_avx2 |
| 2.3 % | 121 | 9.1 | 64.0 | 0 | `v_max_f64 \|vw \|vw` | h_dmax_rows |
| 2.1 % | 26 | 38.8 | 62.1 | 0 | `s_and_saveexec_b64 ?w` | s_saveexec |
| 2.1 % | 25 | 39.1 | 64.0 | 0 | `s_lshr_b32 s c` | s_lshr_b32_k<1, 2> |
| 2.0 % | 41 | 23.0 | 64.0 | 0 | `v_cmp_gt_u32_e32 s v` | h_cmp_avx2 |
| 1.9 % | 43 | 21.8 | 53.2 | 0 | `v_cndmask_b32_e32 c v ?w` | h_cndmask_avx2 |
| 1.9 % | 25 | 36.0 | 9.4 | 0 | `s_cbranch_execz` | b_execz |
| 1.9 % | 65 | 13.7 | 64.0 | 0 | `v_or_b32_sdwa vsel0 v` | h_int_sdwa_rows |
| 1.7 % | 159 | 5.1 | 50.3 | 0 | `ds_read_b128 v` | h_ds_avx2 |
| 1.7 % | 35 | 23.3 | 51.1 | 0 | `v_or_b32_e32 v v` | h_intv_avx2 |

## initial_guess, the base: 1.15e+11 ticks in the handlers, 2.17 billion instructions

| share of the handlers' ticks | ticks a call | calls (millions) | lanes on | handed back | form | handler |
|---:|---:|---:|---:|---:|---|---|
| 13.6 % | 331 | 47.2 | 61.2 | 0 | `v_max_f64 vw vw` | |
| 4.5 % | 78 | 66.1 | 64.0 | 0 | `ds_bpermute_b32 v v` | |
| 4.5 % | 233 | 22.0 | 64.0 | 0 | `v_or_b32_sdwa vsel0 v` | |
| 4.3 % | 345 | 14.5 | 64.0 | 0 | `v_max_f64 \|vw \|vw` | |
| 2.8 % | 26 | 126.1 | 60.9 | 0 | `s_waitcnt` | |
| 2.7 % | 35 | 87.2 | 62.3 | 0 | `v_and_b32_e32 c v` | |
| 2.6 % | 74 | 40.0 | 63.9 | 0 | `v_cmp_lt_i64_e32 sw vw` | |
| 2.2 % | 179 | 14.5 | 64.0 | 0 | `v_cmp_neq_f64_e32 c vw` | |
| 2.1 % | 26 | 96.2 | 6.5 | 0 | `s_or_b64 ? sw` | |
| 1.7 % | 25 | 79.2 | 11.8 | 0 | `s_cbranch_execz` | |
| 1.7 % | 139 | 14.3 | 59.5 | 0 | `v_lshl_add_u64 vw c sw` | |
| 1.7 % | 539 | 3.6 | 64.0 | 0 | `v_frexp_exp_i32_f64_e32 vw` | |
| 1.5 % | 48 | 36.7 | 60.2 | 0 | `v_addc_co_u32_e32 v v ?w` | |
| 1.5 % | 26 | 65.6 | 60.1 | 0 | `s_and_saveexec_b64 ?w` | |
| 1.5 % | 25 | 66.4 | 64.0 | 0 | `s_lshr_b32 s c` | |
| 1.4 % | 41 | 37.8 | 63.7 | 0 | `v_cmp_gt_u32_e32 s v` | |

## initial_guess, the final sources: 8.87e+10 ticks in the handlers, 2.17 billion instructions

| share of the handlers' ticks | ticks a call | calls (millions) | lanes on | handed back | form | handler |
|---:|---:|---:|---:|---:|---|---|
| 5.8 % | 78 | 66.1 | 64.0 | 0 | `ds_bpermute_b32 v v` | h_ds_avx2 |
| 5.1 % | 96 | 47.2 | 61.2 | 0 | `v_max_f64 vw vw` | h_dmax_rows |
| 3.7 % | 26 | 126.1 | 60.9 | 0 | `s_waitcnt` | b_waitcnt |
| 3.4 % | 35 | 87.2 | 62.3 | 0 | `v_and_b32_e32 c v` | h_int_avx2 |
| 2.8 % | 26 | 96.2 | 6.5 | 0 | `s_or_b64 ? sw` | s_or_b64 |
| 2.3 % | 25 | 79.2 | 11.8 | 0 | `s_cbranch_execz` | b_execz |
| 2.1 % | 47 | 40.0 | 63.9 | 0 | `v_cmp_lt_i64_e32 sw vw` | h_cmp64_rows |
| 2.1 % | 50 | 36.7 | 60.2 | 0 | `v_addc_co_u32_e32 v v ?w` | h_carry_avx2 |
| 1.9 % | 26 | 65.6 | 60.1 | 0 | `s_and_saveexec_b64 ?w` | s_saveexec |
| 1.9 % | 26 | 66.4 | 64.0 | 0 | `s_lshr_b32 s c` | s_lshr_b32_k<1, 2> |
| 1.8 % | 111 | 14.5 | 64.0 | 0 | `v_max_f64 \|vw \|vw` | h_dmax_rows |
| 1.8 % | 42 | 37.8 | 63.7 | 0 | `v_cmp_gt_u32_e32 s v` | h_cmp_avx2 |
| 1.6 % | 132 | 11.0 | 34.4 | 0 | `ds_read_b128 v` | h_ds_avx2 |
| 1.6 % | 64 | 22.0 | 64.0 | 0 | `v_or_b32_sdwa vsel0 v` | h_int_sdwa_rows |
| 1.4 % | 26 | 49.8 | 63.6 | 0 | `s_cbranch_vccnz` | b_vccnz |
| 1.4 % | 40 | 31.1 | 59.7 | 0 | `v_add_u32_e32 s v` | h_int_avx2 |

## poisson, the base: 1.46e+11 ticks in the handlers, 2.86 billion instructions

| share of the handlers' ticks | ticks a call | calls (millions) | lanes on | handed back | form | handler |
|---:|---:|---:|---:|---:|---|---|
| 6.2 % | 103 | 88.1 | 11.7 | 0 | `v_mul_f64 vw vw` | |
| 4.6 % | 83 | 81.5 | 30.8 | 0 | `v_add_f64 vw vw` | |
| 4.0 % | 85 | 68.1 | 30.1 | 0 | `v_add_f64 vw -vw` | |
| 4.0 % | 87 | 66.5 | 28.9 | 0 | `v_fmac_f64_e32 sw vw vw` | |
| 3.1 % | 90 | 50.3 | 26.3 | 341682 | `v_fmac_f64_e32 vw vw vw` | |
| 2.9 % | 67 | 64.1 | 6.5 | 0 | `v_mad_u64_u32 s v c` | |
| 2.9 % | 110 | 38.9 | 14.7 | 0 | `v_lshl_add_u64 vw c vw` | |
| 2.7 % | 26 | 153.8 | 17.3 | 0 | `s_mul_i32 s s` | |
| 2.7 % | 29 | 134.2 | 7.1 | 0 | `v_mov_b32_e32 v` | |
| 2.6 % | 122 | 31.5 | 32.8 | 0 | `ds_read_b128 v` | |
| 2.6 % | 138 | 27.5 | 31.8 | 0 | `ds_write_b128 v v` | |
| 2.3 % | 25 | 132.5 | 42.8 | 0 | `s_waitcnt` | |
| 2.2 % | 96 | 33.2 | 8.3 | 0 | `v_lshl_add_u64 vw c sw` | |
| 2.0 % | 46 | 63.5 | 6.1 | 0 | `v_mad_u64_u32 s v vw` | |
| 2.0 % | 38 | 74.6 | 36.4 | 0 | `v_add_u32_e32 s v` | |
| 1.7 % | 36 | 69.6 | 62.2 | 0 | `v_cmp_gt_u32_e32 s v` | |

## poisson, the final sources: 1.32e+11 ticks in the handlers, 2.86 billion instructions

| share of the handlers' ticks | ticks a call | calls (millions) | lanes on | handed back | form | handler |
|---:|---:|---:|---:|---:|---|---|
| 4.9 % | 74 | 88.1 | 11.7 | 0 | `v_mul_f64 vw vw` | h_f64r_avx2 |
| 4.5 % | 72 | 81.5 | 30.8 | 0 | `v_add_f64 vw vw` | h_f64r_avx2 |
| 3.9 % | 75 | 68.1 | 30.1 | 0 | `v_add_f64 vw -vw` | h_f64r_avx2 |
| 3.7 % | 73 | 66.5 | 28.9 | 0 | `v_fmac_f64_e32 sw vw vw` | h_f64r_avx2 |
| 3.3 % | 67 | 64.1 | 6.5 | 0 | `v_mad_u64_u32 s v c` | h_mad_u64_avx2 |
| 3.0 % | 26 | 153.8 | 17.3 | 0 | `s_mul_i32 s s` | s_mul_i32_k<1, 1> |
| 2.9 % | 29 | 134.2 | 7.1 | 0 | `v_mov_b32_e32 v` | h_mov_avx2 |
| 2.9 % | 122 | 31.5 | 32.8 | 0 | `ds_read_b128 v` | h_ds_avx2 |
| 2.9 % | 137 | 27.5 | 31.8 | 0 | `ds_write_b128 v v` | h_ds_avx2 |
| 2.8 % | 74 | 50.3 | 26.3 | 341682 | `v_fmac_f64_e32 vw vw vw` | h_f64r_avx2 |
| 2.6 % | 26 | 132.5 | 42.8 | 0 | `s_waitcnt` | b_waitcnt |
| 2.5 % | 44 | 74.6 | 36.4 | 0 | `v_add_u32_e32 s v` | h_int_avx2 |
| 2.2 % | 47 | 63.5 | 6.1 | 0 | `v_mad_u64_u32 s v vw` | h_mad_u64_avx2 |
| 2.0 % | 67 | 38.9 | 14.7 | 0 | `v_lshl_add_u64 vw c vw` | h_l64_rows |
| 1.8 % | 35 | 69.6 | 62.2 | 0 | `v_cmp_gt_u32_e32 s v` | h_cmp_avx2 |
| 1.8 % | 27 | 88.5 | 27.8 | 0 | `s_mov_b32 c` | s_mov_k<2, 2> |

## ks_hamiltonian, the base: 1.86e+10 ticks in the handlers, 0.31 billion instructions

| share of the handlers' ticks | ticks a call | calls (millions) | lanes on | handed back | form | handler |
|---:|---:|---:|---:|---:|---|---|
| 8.9 % | 92 | 18.0 | 37.5 | 22404 | `v_fmac_f64_e32 vw vw vw` | |
| 7.5 % | 81 | 17.1 | 37.7 | 0 | `v_add_f64 vw vw` | |
| 5.8 % | 122 | 8.8 | 31.1 | 0 | `ds_read_b128 v` | |
| 5.1 % | 80 | 11.9 | 39.1 | 0 | `v_add_f64 vw -vw` | |
| 4.9 % | 163 | 5.6 | 40.8 | 0 | `v_lshl_add_u64 vw c vw` | |
| 4.8 % | 80 | 11.2 | 28.9 | 0 | `v_fmac_f64_e32 sw vw vw` | |
| 4.0 % | 143 | 5.2 | 38.2 | 0 | `ds_write_b128 v v` | |
| 2.4 % | 117 | 3.8 | 32.9 | 0 | `v_mad_u64_u32 s v c` | |
| 2.3 % | 56 | 7.7 | 49.9 | 0 | `v_mov_b64_e32 vw` | |
| 2.3 % | 87 | 4.9 | 28.2 | 0 | `v_mul_f64 vw vw` | |
| 2.2 % | 137 | 3.0 | 42.1 | 0 | `v_lshl_add_u64 vw c sw` | |
| 2.0 % | 93 | 3.9 | 33.5 | 0 | `v_mad_u64_u32 s v vw` | |
| 1.7 % | 26 | 12.3 | 39.3 | 0 | `s_waitcnt` | |
| 1.7 % | 32 | 10.0 | 32.3 | 0 | `v_mov_b32_e32 v` | |
| 1.7 % | 70 | 4.6 | 35.2 | 0 | `v_mul_f64 vw sw` | |
| 1.4 % | 104 | 2.5 | 57.8 | 0 | `v_fma_f64 -vw vw vw` | |

## ks_hamiltonian, the final sources: 1.71e+10 ticks in the handlers, 0.31 billion instructions

| share of the handlers' ticks | ticks a call | calls (millions) | lanes on | handed back | form | handler |
|---:|---:|---:|---:|---:|---|---|
| 8.7 % | 83 | 18.0 | 37.5 | 22404 | `v_fmac_f64_e32 vw vw vw` | h_f64r_avx2 |
| 7.5 % | 74 | 17.1 | 37.7 | 0 | `v_add_f64 vw vw` | h_f64r_avx2 |
| 6.3 % | 122 | 8.8 | 31.1 | 0 | `ds_read_b128 v` | h_ds_avx2 |
| 5.2 % | 75 | 11.9 | 39.1 | 0 | `v_add_f64 vw -vw` | h_f64r_avx2 |
| 4.8 % | 74 | 11.2 | 28.9 | 0 | `v_fmac_f64_e32 sw vw vw` | h_f64r_avx2 |
| 4.3 % | 142 | 5.2 | 38.2 | 0 | `ds_write_b128 v v` | h_ds_avx2 |
| 3.1 % | 94 | 5.6 | 40.8 | 0 | `v_lshl_add_u64 vw c vw` | h_l64_rows |
| 2.6 % | 117 | 3.8 | 32.9 | 0 | `v_mad_u64_u32 s v c` | h_mad_u64_avx2 |
| 2.4 % | 55 | 7.7 | 49.9 | 0 | `v_mov_b64_e32 vw` | h_mov64_avx2 |
| 2.1 % | 75 | 4.9 | 28.2 | 0 | `v_mul_f64 vw vw` | h_f64r_avx2 |
| 2.1 % | 92 | 3.9 | 33.5 | 0 | `v_mad_u64_u32 s v vw` | h_mad_u64_avx2 |
| 1.9 % | 26 | 12.3 | 39.3 | 0 | `s_waitcnt` | b_waitcnt |
| 1.9 % | 32 | 10.0 | 32.3 | 0 | `v_mov_b32_e32 v` | h_mov_avx2 |
| 1.8 % | 67 | 4.6 | 35.2 | 0 | `v_mul_f64 vw sw` | h_f64r_avx2 |
| 1.5 % | 84 | 3.0 | 42.1 | 0 | `v_lshl_add_u64 vw c sw` | h_l64_rows |
| 1.4 % | 98 | 2.5 | 57.8 | 0 | `v_fma_f64 -vw vw vw` | h_f64r_avx2 |

