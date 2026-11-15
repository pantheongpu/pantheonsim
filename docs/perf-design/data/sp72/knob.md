# The knob's A/B: blocks off against on (round sp72)

`./bench.sh knobF --build work/build --build on/build --threads 1 --runs 9 --cpus 14,15`: both sides are the same library (the `on` copy
reads a variable the bench always sets where the other reads `VGPU_AMD_BLOCKS`), so nothing but the knob differs. One host thread, 9 runs a
side, alternating; the rate is wave-instructions (or the workload's own unit) per CPU second of the device phase, the **best** run of each
side (`scripts/ab.py`). 1-minute load during the pass: 6.1 to 12.7 (median 8.1), other sessions' work plus this round's mutants and inq runs.
"counters": every counter of the counting run equal on both sides (the API rows run no kernel to count).

| workload | target | off | on | on / off | counters |
|---|---|---:|---:|---:|---|
| valu_int | mi300x | 68.51 M/s | 72.58 M/s | 1.059 | same |
| valu_int | rx6800 | 64.70 M/s | 76.71 M/s | 1.186 | same |
| valu_f32 | mi300x | 45.04 M/s | 51.94 M/s | 1.153 | same |
| valu_f32 | rx6800 | 58.54 M/s | 73.15 M/s | 1.249 | same |
| valu_f64 | mi300x | 53.38 M/s | 70.29 M/s | 1.317 | same |
| valu_f64 | rx6800 | 76.01 M/s | 90.32 M/s | 1.188 | same |
| mfma | mi300x | 7.35 M/s | 7.10 M/s | 0.966 | same |
| mfma_i8 | mi300x | 8.82 M/s | 8.96 M/s | 1.016 | same |
| mfma_fp8 | mi300x | 2.87 M/s | 2.83 M/s | 0.984 | same |
| mfma_bf16 | mi300x | 7.73 M/s | 7.90 M/s | 1.021 | same |
| mfma_f64 | mi300x | 8.68 M/s | 8.70 M/s | 1.002 | same |
| mfma_f64_4b | mi300x | 12.47 M/s | 12.59 M/s | 1.010 | same |
| smfmac | mi300x | 2.44 M/s | 2.42 M/s | 0.993 | same |
| mem | mi300x | 26.79 M/s | 23.32 M/s | 0.871 | same |
| mem | rx6800 | 53.72 M/s | 56.22 M/s | 1.047 | same |
| lds | mi300x | 24.07 M/s | 22.52 M/s | 0.935 | same |
| lds | rx6800 | 50.96 M/s | 48.31 M/s | 0.948 | same |
| narrow | mi300x | 20.21 M/s | 19.79 M/s | 0.979 | same |
| narrow | rx6800 | 26.84 M/s | 26.25 M/s | 0.978 | same |
| smem | mi300x | 90.77 M/s | 120.30 M/s | 1.325 | same |
| smem | rx6800 | 126.73 M/s | 201.19 M/s | 1.588 | same |
| buf | mi300x | 15.55 M/s | 15.17 M/s | 0.975 | same |
| buf | rx6800 | 26.65 M/s | 27.75 M/s | 1.041 | same |
| ldsx | mi300x | 19.82 M/s | 25.37 M/s | 1.280 | same |
| ldsx | rx6800 | 41.33 M/s | 43.39 M/s | 1.050 | same |
| flat | mi300x | 19.55 M/s | 19.14 M/s | 0.979 | same |
| flat | rx6800 | 33.76 M/s | 32.66 M/s | 0.967 | same |
| atomics | mi300x | 22.35 M/s | 23.87 M/s | 1.068 | same |
| atomics | rx6800 | 36.90 M/s | 38.96 M/s | 1.056 | same |
| divergent | mi300x | 59.71 M/s | 71.35 M/s | 1.195 | same |
| divergent | rx6800 | 78.36 M/s | 98.81 M/s | 1.261 | same |
| wave | mi300x | 59.44 M/s | 78.18 M/s | 1.315 | same |
| wave | rx6800 | 81.30 M/s | 116.16 M/s | 1.429 | same |
| launch | mi300x | 0.32 M/s | 0.30 M/s | 0.927 | - |
| launch | rx6800 | 0.29 M/s | 0.30 M/s | 1.018 | - |
| api | mi300x | 7.68 M/s | 8.21 M/s | 1.068 | - |
| api | rx6800 | 7.23 M/s | 7.35 M/s | 1.017 | - |
| apicheap | mi300x | 91.44 M/s | 95.76 M/s | 1.047 | - |
| apicheap | rx6800 | 85.11 M/s | 85.94 M/s | 1.010 | - |
| launch_wide | mi300x | 1788.48 /s | 1791.93 /s | 1.002 | - |
| launch_wide | rx6800 | 941.41 /s | 859.45 /s | 0.913 | - |
| launch_mid | mi300x | 13608.50 /s | 13511.20 /s | 0.993 | - |
| launch_mid | rx6800 | 7131.82 /s | 6692.74 /s | 0.938 | - |
| launch_sync | mi300x | 0.52 M/s | 0.49 M/s | 0.956 | - |
| launch_sync | rx6800 | 0.43 M/s | 0.42 M/s | 0.980 | - |
| memcpy | mi300x | 1.16 M/s | 1.17 M/s | 1.005 | - |
| memcpy | rx6800 | 1.13 M/s | 1.14 M/s | 1.008 | - |
| stream | mi300x | 0.17 M/s | 0.16 M/s | 0.992 | - |
| stream | rx6800 | 0.20 M/s | 0.18 M/s | 0.889 | - |
| graph | mi300x | 82574.11 /s | 80598.58 /s | 0.976 | - |
| graph | rx6800 | 67081.08 /s | 66360.67 /s | 0.989 | - |
| startup | mi300x | 0.00 /s | 0.00 /s | 1.052 | - |
| startup | rx6800 | 0.01 /s | 0.01 /s | 0.986 | - |
| p_dpp | mi300x | 11.29 M/s | 11.42 M/s | 1.012 | same |
| p_dpp | rx6800 | 19.10 M/s | 18.27 M/s | 0.957 | same |
| p_dpp_arch | mi300x | 8.60 M/s | 9.52 M/s | 1.106 | same |
| p_dpp_arch | rx6800 | 19.47 M/s | 20.51 M/s | 1.054 | same |
| p_dpp_scan | mi300x | 16.48 M/s | 17.22 M/s | 1.044 | same |
| p_dpp_scan | rx6800 | 26.47 M/s | 28.72 M/s | 1.085 | same |
| p_perm | mi300x | 29.51 M/s | 33.08 M/s | 1.121 | same |
| p_perm | rx6800 | 56.17 M/s | 71.21 M/s | 1.268 | same |
| p_permlane | rx6800 | 56.92 M/s | 66.04 M/s | 1.160 | same |
| p_lane | mi300x | 88.99 M/s | 104.10 M/s | 1.170 | same |
| p_lane | rx6800 | 86.34 M/s | 114.13 M/s | 1.322 | same |
| p_pk16 | mi300x | 35.21 M/s | 38.27 M/s | 1.087 | same |
| p_pk16 | rx6800 | 16.14 M/s | 16.98 M/s | 1.052 | same |
| p_pkf16 | mi300x | 7.80 M/s | 7.38 M/s | 0.946 | same |
| p_pkf16 | rx6800 | 17.71 M/s | 15.44 M/s | 0.872 | same |
| p_sdwa | mi300x | 37.45 M/s | 41.34 M/s | 1.104 | same |
| p_sdwa | rx6800 | 57.44 M/s | 69.26 M/s | 1.206 | same |
| p_vopd | mi300x | 43.63 M/s | 48.80 M/s | 1.119 | same |
| p_vopd | rx6800 | 62.29 M/s | 78.26 M/s | 1.256 | same |
| p_cmp | mi300x | 48.09 M/s | 57.76 M/s | 1.201 | same |
| p_cmp | rx6800 | 76.26 M/s | 113.55 M/s | 1.489 | same |
| p_i64 | mi300x | 43.08 M/s | 51.48 M/s | 1.195 | same |
| p_i64 | rx6800 | 63.04 M/s | 81.76 M/s | 1.297 | same |
| p_f64x | mi300x | 13.11 M/s | 13.67 M/s | 1.043 | same |
| p_f64x | rx6800 | 25.64 M/s | 23.70 M/s | 0.924 | same |
| p_smem_wide | mi300x | 81.72 M/s | 99.32 M/s | 1.215 | same |
| p_smem_wide | rx6800 | 87.01 M/s | 113.39 M/s | 1.303 | same |
| p_sbuf | mi300x | 67.67 M/s | 74.94 M/s | 1.107 | same |
| p_sbuf | rx6800 | 92.61 M/s | 101.71 M/s | 1.098 | same |

best of 9 runs a side; 82 rows, 9 runs a side each; geometric mean of on / off: 1.077; counters the same in all of the 62 rows counted
below 0.97: mfma/mi300x 0.966, mem/mi300x 0.871, lds/mi300x 0.935, lds/rx6800 0.948, flat/rx6800 0.967, launch/mi300x 0.927, launch_wide/rx6800 0.913, launch_mid/rx6800 0.938, launch_sync/mi300x 0.956, stream/rx6800 0.889, p_dpp/rx6800 0.957, p_pkf16/mi300x 0.946, p_pkf16/rx6800 0.872, p_f64x/rx6800 0.924
1.05 or more: smem/rx6800 1.588, p_cmp/rx6800 1.489, wave/rx6800 1.429, smem/mi300x 1.325, p_lane/rx6800 1.322, valu_f64/mi300x 1.317, wave/mi300x 1.315, p_smem_wide/rx6800 1.303, p_i64/rx6800 1.297, ldsx/mi300x 1.280, p_perm/rx6800 1.268, divergent/rx6800 1.261, p_vopd/rx6800 1.256, valu_f32/rx6800 1.249, p_smem_wide/mi300x 1.215, p_sdwa/rx6800 1.206, p_cmp/mi300x 1.201, p_i64/mi300x 1.195, divergent/mi300x 1.195, valu_f64/rx6800 1.188, valu_int/rx6800 1.186, p_lane/mi300x 1.170, p_permlane/rx6800 1.160, valu_f32/mi300x 1.153, p_perm/mi300x 1.121, p_vopd/mi300x 1.119, p_sbuf/mi300x 1.107, p_dpp_arch/mi300x 1.106, p_sdwa/mi300x 1.104, p_sbuf/rx6800 1.098, p_pk16/mi300x 1.087, p_dpp_scan/rx6800 1.085, atomics/mi300x 1.068, api/mi300x 1.068, valu_int/mi300x 1.059, atomics/rx6800 1.056, p_dpp_arch/rx6800 1.054, p_pk16/rx6800 1.052, startup/mi300x 1.052

An earlier pass of the same sources (`knobA`, the library before a change to a test-only function; load 8.6 to 13.8):

```

best of 9 runs a side; 82 rows, 9 runs a side each; geometric mean of on / off: 1.072; counters the same in all of the 62 rows counted
below 0.97: mem/mi300x 0.945, mem/rx6800 0.963, lds/mi300x 0.964, lds/rx6800 0.938, narrow/rx6800 0.961, flat/mi300x 0.939, api/rx6800 0.936, apicheap/mi300x 0.921, apicheap/rx6800 0.950, launch_mid/mi300x 0.965, launch_mid/rx6800 0.963, launch_sync/mi300x 0.949, stream/rx6800 0.905, p_dpp/mi300x 0.935
1.05 or more: wave/rx6800 1.414, smem/rx6800 1.366, smem/mi300x 1.343, p_lane/rx6800 1.326, p_cmp/rx6800 1.318, wave/mi300x 1.303, p_smem_wide/rx6800 1.299, divergent/rx6800 1.278, p_vopd/rx6800 1.268, valu_int/mi300x 1.261, p_i64/rx6800 1.252, p_lane/mi300x 1.206, valu_f32/rx6800 1.200, divergent/mi300x 1.195, valu_int/rx6800 1.195, p_smem_wide/mi300x 1.190, p_sdwa/rx6800 1.186, p_i64/mi300x 1.185, p_perm/rx6800 1.180, p_permlane/rx6800 1.165, p_cmp/mi300x 1.151, p_sbuf/mi300x 1.140, atomics/rx6800 1.137, valu_f32/mi300x 1.133, p_sbuf/rx6800 1.132, p_vopd/mi300x 1.127, valu_f64/rx6800 1.119, p_sdwa/mi300x 1.113, p_f64x/rx6800 1.108, p_perm/mi300x 1.107, valu_f64/mi300x 1.095, p_pk16/rx6800 1.089, p_dpp_scan/rx6800 1.077, p_pk16/mi300x 1.077, p_pkf16/rx6800 1.054, p_dpp_scan/mi300x 1.054, p_dpp_arch/mi300x 1.051, ldsx/mi300x 1.050
```

Both passes (`scripts/both.py 0.97 bench-knobA.json bench-knobF.json`):

```
pass 1: geometric mean 1.072, 14 of 82 rows below 0.97, lowest 0.905, highest 1.414
pass 2: geometric mean 1.077, 14 of 82 rows below 0.97, lowest 0.871, highest 1.588
below 0.97 in every pass: mem/mi300x (0.945, 0.871), lds/mi300x (0.964, 0.935), lds/rx6800 (0.938, 0.948), launch_mid/rx6800 (0.963, 0.938), launch_sync/mi300x (0.949, 0.956), stream/rx6800 (0.905, 0.889)
```

35 rows are at 1.05 or more in both passes (pass A, pass F): wave/rx6800 (1.41, 1.43), smem/rx6800 (1.37, 1.59), smem/mi300x (1.34, 1.33),
p_lane/rx6800 (1.33, 1.32), p_cmp/rx6800 (1.32, 1.49), wave/mi300x (1.30, 1.32), p_smem_wide/rx6800 (1.30, 1.30), divergent/rx6800 (1.28,
1.26), p_vopd/rx6800 (1.27, 1.26), p_i64/rx6800 (1.25, 1.30), valu_f32/rx6800 (1.20, 1.25), divergent/mi300x (1.20, 1.19),
p_smem_wide/mi300x (1.19, 1.22), p_sdwa/rx6800 (1.19, 1.21), valu_int/rx6800 (1.20, 1.19), p_i64/mi300x (1.19, 1.20), p_perm/rx6800 (1.18,
1.27), p_lane/mi300x (1.21, 1.17), p_permlane/rx6800 (1.17, 1.16), p_cmp/mi300x (1.15, 1.20), valu_f32/mi300x (1.13, 1.15),
valu_f64/rx6800 (1.12, 1.19), p_vopd/mi300x (1.13, 1.12), p_sbuf/mi300x (1.14, 1.11), p_perm/mi300x (1.11, 1.12), p_sdwa/mi300x (1.11,
1.10), p_sbuf/rx6800 (1.13, 1.10), valu_f64/mi300x (1.10, 1.32), p_dpp_scan/rx6800 (1.08, 1.09), p_pk16/mi300x (1.08, 1.09),
valu_int/mi300x (1.26, 1.06), atomics/rx6800 (1.14, 1.06), p_pk16/rx6800 (1.09, 1.05), p_dpp_arch/mi300x (1.05, 1.11), ldsx/mi300x (1.05,
1.28).

The rows that run no kernel (api, apicheap, stream, memcpy, launch_*, graph, startup) are the same code path on both sides: they range from
0.89 to 1.07 in these passes, which is the host's noise for one row.

Four passes over 12 workloads during the slice's development (15 runs a side; the block runner changed between them: counters in one list,
then no block starting at a wait or a store and a minimum of three instructions) gave the same picture: valu_int 1.14 to 1.20, valu_f32 1.07
to 1.22, valu_f64 1.09 to 1.26, p_i64 1.15 to 1.22, divergent 1.13 to 1.24, smem 1.27 to 1.33; mem 0.93 to 1.07, narrow 0.94 to 1.05,
lds 0.92 to 1.03, buf 0.96 to 1.06, flat 0.98 to 1.01; api (no kernel) 0.94 to 1.00.
