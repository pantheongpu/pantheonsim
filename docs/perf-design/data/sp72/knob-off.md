# The knob off: the base against this branch, and the control (round sp72)

Two questions: is a row slower than the base when the knob is off, and how much does a row move when nothing but code placement changes?
`base`: origin/sp69-hotspot-start 77a8b8c8. `work`: this branch, `VGPU_AMD_BLOCKS` unset. `noop`: the base plus one function nothing calls
in `gcn_exec.cpp`. One host thread, 9 runs a side, the best run of each side. The two A/Bs of a pass ran at the same time on two cores
(cpus 14,15 and 12,13 or 10,11), beside this round's other jobs and the other sessions' (1-minute load 6 to 13).

Summary of the two passes (pass 1 on the library before the last changes of the block runner, which the knob-off path does not run; pass F
on the final one):

```
base against work (knob off):
pass 1: geometric mean 0.996, 19 of 82 rows below 0.97, lowest 0.774, highest 1.160
pass 2: geometric mean 0.993, 26 of 82 rows below 0.97, lowest 0.791, highest 1.313
below 0.97 in every pass: valu_f32/mi300x (0.814, 0.964), valu_f64/rx6800 (0.774, 0.916), narrow/mi300x (0.902, 0.791), smem/rx6800 (0.911, 0.960), ldsx/rx6800 (0.943, 0.906), atomics/rx6800 (0.969, 0.948), api/rx6800 (0.934, 0.969), launch_wide/mi300x (0.852, 0.967), launch_mid/rx6800 (0.905, 0.810), stream/rx6800 (0.924, 0.914), p_f64x/mi300x (0.904, 0.952)
base against noop:
pass 1: geometric mean 0.993, 22 of 82 rows below 0.97, lowest 0.809, highest 1.198
pass 2: geometric mean 1.003, 17 of 82 rows below 0.97, lowest 0.857, highest 1.275
below 0.97 in every pass: valu_f32/mi300x (0.891, 0.879), buf/rx6800 (0.925, 0.953), ldsx/mi300x (0.915, 0.935), atomics/mi300x (0.941, 0.969), apicheap/mi300x (0.907, 0.943), launch_mid/mi300x (0.951, 0.949)
```

Neither can be told from the other: the geometric means are 0.993 to 0.996 for this branch and 0.993 to 1.003 for the control, 19 and 26
rows of 82 are below 0.97 for this branch and 22 and 17 for the control, and single rows range from 0.77 to 1.31 and from 0.81 to 1.28. The
3 % bar for a single row cannot be read on this host under this load.

A third measurement took the rows that were low in both passes for this branch (and valu_int), 25 runs a side, the two A/Bs again side by
side (`bench-offS`, `bench-noopS`): none of them is low a third time except valu_f32 on the MI300X (0.81, 0.96, 0.96; the control on the
same row 0.89, 0.88, 1.00), and the control moves single rows as far as the branch does. In particular valu_f64 on the RX 6800 is 1.22
for this branch and 1.23 for the control: the base's own placement of that loop is 20 % slower than either, with nothing changed that the
loop runs.

## The third measurement: 25 runs a side

base against work (knob off):

| workload | target | base | work | work / base | counters |
|---|---|---:|---:|---:|---|
| valu_f32 | mi300x | 58.38 M/s | 56.29 M/s | 0.964 | same |
| valu_f32 | rx6800 | 94.57 M/s | 96.20 M/s | 1.017 | same |
| valu_f64 | mi300x | 54.78 M/s | 50.72 M/s | 0.926 | same |
| valu_f64 | rx6800 | 68.25 M/s | 83.00 M/s | 1.216 | same |
| narrow | mi300x | 20.87 M/s | 21.05 M/s | 1.009 | same |
| narrow | rx6800 | 31.83 M/s | 29.50 M/s | 0.927 | same |
| smem | mi300x | 131.87 M/s | 119.71 M/s | 0.908 | same |
| smem | rx6800 | 110.76 M/s | 107.52 M/s | 0.971 | same |
| ldsx | mi300x | 20.43 M/s | 20.82 M/s | 1.020 | same |
| ldsx | rx6800 | 48.96 M/s | 47.62 M/s | 0.973 | same |
| p_f64x | mi300x | 15.13 M/s | 15.33 M/s | 1.013 | same |
| p_f64x | rx6800 | 23.29 M/s | 23.55 M/s | 1.011 | same |
| atomics | mi300x | 23.58 M/s | 23.23 M/s | 0.985 | same |
| atomics | rx6800 | 41.20 M/s | 39.16 M/s | 0.951 | same |
| valu_int | mi300x | 70.45 M/s | 71.98 M/s | 1.022 | same |
| valu_int | rx6800 | 99.59 M/s | 93.44 M/s | 0.938 | same |

best of 25 runs a side; 16 rows, 25 runs a side each; geometric mean of work / base: 0.988; counters the same in all of the 16 rows counted
below 0.97: valu_f32/mi300x 0.964, valu_f64/mi300x 0.926, narrow/rx6800 0.927, smem/mi300x 0.908, atomics/rx6800 0.951, valu_int/rx6800 0.938
1.05 or more: valu_f64/rx6800 1.216

base against noop:

| workload | target | base | noop | noop / base | counters |
|---|---|---:|---:|---:|---|
| valu_f32 | mi300x | 58.99 M/s | 58.94 M/s | 0.999 | same |
| valu_f32 | rx6800 | 96.51 M/s | 94.29 M/s | 0.977 | same |
| valu_f64 | mi300x | 48.99 M/s | 49.41 M/s | 1.009 | same |
| valu_f64 | rx6800 | 68.11 M/s | 83.75 M/s | 1.230 | same |
| narrow | mi300x | 21.53 M/s | 20.99 M/s | 0.975 | same |
| narrow | rx6800 | 29.22 M/s | 28.72 M/s | 0.983 | same |
| smem | mi300x | 119.73 M/s | 131.09 M/s | 1.095 | same |
| smem | rx6800 | 109.72 M/s | 109.50 M/s | 0.998 | same |
| ldsx | mi300x | 21.76 M/s | 20.24 M/s | 0.930 | same |
| ldsx | rx6800 | 47.64 M/s | 49.57 M/s | 1.041 | same |
| p_f64x | mi300x | 15.67 M/s | 15.19 M/s | 0.969 | same |
| p_f64x | rx6800 | 23.10 M/s | 23.00 M/s | 0.996 | same |
| atomics | mi300x | 24.73 M/s | 23.37 M/s | 0.945 | same |
| atomics | rx6800 | 43.28 M/s | 39.36 M/s | 0.909 | same |
| valu_int | mi300x | 67.89 M/s | 68.07 M/s | 1.003 | same |
| valu_int | rx6800 | 89.16 M/s | 90.10 M/s | 1.011 | same |

best of 25 runs a side; 16 rows, 25 runs a side each; geometric mean of noop / base: 1.002; counters the same in all of the 16 rows counted
below 0.97: ldsx/mi300x 0.930, p_f64x/mi300x 0.969, atomics/mi300x 0.945, atomics/rx6800 0.909
1.05 or more: valu_f64/rx6800 1.230, smem/mi300x 1.095


## Pass F, base against work (knob off)

| workload | target | base | work | work / base | counters |
|---|---|---:|---:|---:|---|
| valu_int | mi300x | 61.06 M/s | 63.64 M/s | 1.042 | same |
| valu_int | rx6800 | 71.31 M/s | 75.28 M/s | 1.056 | same |
| valu_f32 | mi300x | 48.94 M/s | 47.20 M/s | 0.964 | same |
| valu_f32 | rx6800 | 59.43 M/s | 62.64 M/s | 1.054 | same |
| valu_f64 | mi300x | 41.32 M/s | 42.41 M/s | 1.026 | same |
| valu_f64 | rx6800 | 81.18 M/s | 74.39 M/s | 0.916 | same |
| mfma | mi300x | 9.00 M/s | 8.00 M/s | 0.890 | same |
| mfma_i8 | mi300x | 9.43 M/s | 9.39 M/s | 0.996 | same |
| mfma_fp8 | mi300x | 2.74 M/s | 2.82 M/s | 1.030 | same |
| mfma_bf16 | mi300x | 7.63 M/s | 7.49 M/s | 0.982 | same |
| mfma_f64 | mi300x | 9.27 M/s | 9.07 M/s | 0.979 | same |
| mfma_f64_4b | mi300x | 14.26 M/s | 13.73 M/s | 0.963 | same |
| smfmac | mi300x | 2.40 M/s | 2.31 M/s | 0.959 | same |
| mem | mi300x | 21.27 M/s | 21.19 M/s | 0.996 | same |
| mem | rx6800 | 46.01 M/s | 43.40 M/s | 0.943 | same |
| lds | mi300x | 22.57 M/s | 22.72 M/s | 1.006 | same |
| lds | rx6800 | 61.50 M/s | 65.41 M/s | 1.064 | same |
| narrow | mi300x | 27.82 M/s | 22.02 M/s | 0.791 | same |
| narrow | rx6800 | 26.59 M/s | 29.14 M/s | 1.096 | same |
| smem | mi300x | 91.64 M/s | 91.34 M/s | 0.997 | same |
| smem | rx6800 | 120.78 M/s | 115.93 M/s | 0.960 | same |
| buf | mi300x | 16.87 M/s | 16.48 M/s | 0.977 | same |
| buf | rx6800 | 35.36 M/s | 29.31 M/s | 0.829 | same |
| ldsx | mi300x | 20.76 M/s | 20.68 M/s | 0.996 | same |
| ldsx | rx6800 | 45.02 M/s | 40.81 M/s | 0.906 | same |
| flat | mi300x | 18.85 M/s | 17.83 M/s | 0.946 | same |
| flat | rx6800 | 34.44 M/s | 33.55 M/s | 0.974 | same |
| atomics | mi300x | 23.50 M/s | 22.51 M/s | 0.958 | same |
| atomics | rx6800 | 39.22 M/s | 37.18 M/s | 0.948 | same |
| divergent | mi300x | 58.59 M/s | 59.10 M/s | 1.009 | same |
| divergent | rx6800 | 78.15 M/s | 86.42 M/s | 1.106 | same |
| wave | mi300x | 64.23 M/s | 62.95 M/s | 0.980 | same |
| wave | rx6800 | 81.30 M/s | 81.55 M/s | 1.003 | same |
| launch | mi300x | 0.28 M/s | 0.28 M/s | 1.015 | - |
| launch | rx6800 | 0.28 M/s | 0.30 M/s | 1.080 | - |
| api | mi300x | 7.86 M/s | 7.63 M/s | 0.971 | - |
| api | rx6800 | 8.01 M/s | 7.76 M/s | 0.969 | - |
| apicheap | mi300x | 108.36 M/s | 98.89 M/s | 0.913 | - |
| apicheap | rx6800 | 102.73 M/s | 98.78 M/s | 0.962 | - |
| launch_wide | mi300x | 1930.92 /s | 1866.23 /s | 0.967 | - |
| launch_wide | rx6800 | 858.11 /s | 884.84 /s | 1.031 | - |
| launch_mid | mi300x | 13201.96 /s | 13193.42 /s | 0.999 | - |
| launch_mid | rx6800 | 8750.31 /s | 7085.04 /s | 0.810 | - |
| launch_sync | mi300x | 0.53 M/s | 0.50 M/s | 0.945 | - |
| launch_sync | rx6800 | 0.43 M/s | 0.43 M/s | 1.011 | - |
| memcpy | mi300x | 1.15 M/s | 1.16 M/s | 1.003 | - |
| memcpy | rx6800 | 1.13 M/s | 1.14 M/s | 1.005 | - |
| stream | mi300x | 0.17 M/s | 0.17 M/s | 1.011 | - |
| stream | rx6800 | 0.20 M/s | 0.18 M/s | 0.914 | - |
| graph | mi300x | 81391.25 /s | 81109.58 /s | 0.997 | - |
| graph | rx6800 | 67111.84 /s | 88127.73 /s | 1.313 | - |
| startup | mi300x | 0.00 /s | 0.00 /s | 1.057 | - |
| startup | rx6800 | 0.01 /s | 0.01 /s | 0.997 | - |
| p_dpp | mi300x | 8.72 M/s | 8.78 M/s | 1.007 | same |
| p_dpp | rx6800 | 19.28 M/s | 18.43 M/s | 0.956 | same |
| p_dpp_arch | mi300x | 9.01 M/s | 8.77 M/s | 0.974 | same |
| p_dpp_arch | rx6800 | 18.87 M/s | 18.98 M/s | 1.006 | same |
| p_dpp_scan | mi300x | 16.50 M/s | 16.39 M/s | 0.993 | same |
| p_dpp_scan | rx6800 | 26.54 M/s | 26.55 M/s | 1.000 | same |
| p_perm | mi300x | 29.33 M/s | 29.52 M/s | 1.006 | same |
| p_perm | rx6800 | 57.76 M/s | 61.12 M/s | 1.058 | same |
| p_permlane | rx6800 | 51.62 M/s | 52.17 M/s | 1.011 | same |
| p_lane | mi300x | 88.00 M/s | 85.97 M/s | 0.977 | same |
| p_lane | rx6800 | 95.53 M/s | 91.81 M/s | 0.961 | same |
| p_pk16 | mi300x | 33.62 M/s | 33.55 M/s | 0.998 | same |
| p_pk16 | rx6800 | 16.08 M/s | 16.20 M/s | 1.008 | same |
| p_pkf16 | mi300x | 7.82 M/s | 7.54 M/s | 0.964 | same |
| p_pkf16 | rx6800 | 14.20 M/s | 15.62 M/s | 1.100 | same |
| p_sdwa | mi300x | 38.55 M/s | 37.45 M/s | 0.971 | same |
| p_sdwa | rx6800 | 55.27 M/s | 55.49 M/s | 1.004 | same |
| p_vopd | mi300x | 41.61 M/s | 42.24 M/s | 1.015 | same |
| p_vopd | rx6800 | 63.68 M/s | 62.09 M/s | 0.975 | same |
| p_cmp | mi300x | 50.35 M/s | 61.52 M/s | 1.222 | same |
| p_cmp | rx6800 | 67.97 M/s | 66.84 M/s | 0.983 | same |
| p_i64 | mi300x | 41.92 M/s | 43.55 M/s | 1.039 | same |
| p_i64 | rx6800 | 59.08 M/s | 60.84 M/s | 1.030 | same |
| p_f64x | mi300x | 13.51 M/s | 12.86 M/s | 0.952 | same |
| p_f64x | rx6800 | 23.68 M/s | 22.66 M/s | 0.957 | same |
| p_smem_wide | mi300x | 81.76 M/s | 77.91 M/s | 0.953 | same |
| p_smem_wide | rx6800 | 102.74 M/s | 102.51 M/s | 0.998 | same |
| p_sbuf | mi300x | 69.84 M/s | 75.57 M/s | 1.082 | same |
| p_sbuf | rx6800 | 87.61 M/s | 96.03 M/s | 1.096 | same |

best of 9 runs a side; 82 rows, 9 runs a side each; geometric mean of work / base: 0.993; counters the same in all of the 62 rows counted
below 0.97: valu_f32/mi300x 0.964, valu_f64/rx6800 0.916, mfma/mi300x 0.890, mfma_f64_4b/mi300x 0.963, smfmac/mi300x 0.959, mem/rx6800 0.943, narrow/mi300x 0.791, smem/rx6800 0.960, buf/rx6800 0.829, ldsx/rx6800 0.906, flat/mi300x 0.946, atomics/mi300x 0.958, atomics/rx6800 0.948, api/rx6800 0.969, apicheap/mi300x 0.913, apicheap/rx6800 0.962, launch_wide/mi300x 0.967, launch_mid/rx6800 0.810, launch_sync/mi300x 0.945, stream/rx6800 0.914, p_dpp/rx6800 0.956, p_lane/rx6800 0.961, p_pkf16/mi300x 0.964, p_f64x/mi300x 0.952, p_f64x/rx6800 0.957, p_smem_wide/mi300x 0.953
1.05 or more: graph/rx6800 1.313, p_cmp/mi300x 1.222, divergent/rx6800 1.106, p_pkf16/rx6800 1.100, p_sbuf/rx6800 1.096, narrow/rx6800 1.096, p_sbuf/mi300x 1.082, launch/rx6800 1.080, lds/rx6800 1.064, p_perm/rx6800 1.058, startup/mi300x 1.057, valu_int/rx6800 1.056, valu_f32/rx6800 1.054

## Pass F, base against noop (summary; the rows are in the same form)

```
best of 9 runs a side; 82 rows, 9 runs a side each; geometric mean of noop / base: 1.003; counters the same in all of the 62 rows counted
below 0.97: valu_f32/mi300x 0.879, mfma_fp8/mi300x 0.960, mem/rx6800 0.964, smem/mi300x 0.958, buf/rx6800 0.953, ldsx/mi300x 0.935, flat/mi300x 0.930, atomics/mi300x 0.969, atomics/rx6800 0.962, apicheap/mi300x 0.943, apicheap/rx6800 0.940, launch_wide/mi300x 0.911, launch_mid/mi300x 0.949, launch_mid/rx6800 0.951, startup/mi300x 0.910, p_smem_wide/mi300x 0.857, p_sbuf/rx6800 0.923
1.05 or more: p_cmp/rx6800 1.275, mfma_i8/mi300x 1.186, valu_f64/rx6800 1.172, narrow/mi300x 1.171, p_pkf16/rx6800 1.132, mfma_bf16/mi300x 1.119, p_sbuf/mi300x 1.082, p_f64x/mi300x 1.081, valu_f32/rx6800 1.075, ldsx/rx6800 1.065, smfmac/mi300x 1.055, mfma/mi300x 1.051
```

## Pass 1 (summaries)

```
base against work:

best of 9 runs a side; 82 rows, 9 runs a side each; geometric mean of work / base: 0.996; counters the same in all of the 62 rows counted
below 0.97: valu_f32/mi300x 0.814, valu_f32/rx6800 0.959, valu_f64/rx6800 0.774, lds/mi300x 0.892, narrow/mi300x 0.902, smem/rx6800 0.911, ldsx/rx6800 0.943, atomics/rx6800 0.969, launch/mi300x 0.944, api/mi300x 0.958, api/rx6800 0.934, launch_wide/mi300x 0.852, launch_mid/rx6800 0.905, memcpy/rx6800 0.966, stream/rx6800 0.924, p_dpp_arch/mi300x 0.888, p_vopd/rx6800 0.969, p_f64x/mi300x 0.904, p_sbuf/mi300x 0.965
1.05 or more: launch_wide/rx6800 1.160, p_pkf16/mi300x 1.153, p_dpp/rx6800 1.144, buf/rx6800 1.094, wave/rx6800 1.093, p_dpp_scan/rx6800 1.091, mfma_f64_4b/mi300x 1.083, p_perm/mi300x 1.079, flat/mi300x 1.072, smfmac/mi300x 1.068, launch_sync/mi300x 1.060, divergent/rx6800 1.059, p_pk16/mi300x 1.058, launch_mid/mi300x 1.055
base against noop:

best of 9 runs a side; 82 rows, 9 runs a side each; geometric mean of noop / base: 0.993; counters the same in all of the 62 rows counted
below 0.97: valu_int/rx6800 0.929, valu_f32/mi300x 0.891, valu_f64/rx6800 0.925, mem/mi300x 0.966, lds/rx6800 0.893, buf/mi300x 0.909, buf/rx6800 0.925, ldsx/mi300x 0.915, flat/rx6800 0.956, atomics/mi300x 0.941, divergent/mi300x 0.969, divergent/rx6800 0.865, wave/rx6800 0.959, apicheap/mi300x 0.907, launch_wide/rx6800 0.901, launch_mid/mi300x 0.951, stream/mi300x 0.858, p_dpp/mi300x 0.932, p_pk16/mi300x 0.969, p_pkf16/mi300x 0.809, p_vopd/mi300x 0.945, p_f64x/mi300x 0.916
1.05 or more: smem/mi300x 1.198, valu_int/mi300x 1.108, launch/rx6800 1.108, stream/rx6800 1.101, p_dpp_arch/mi300x 1.097, apicheap/rx6800 1.096, wave/mi300x 1.095, p_cmp/rx6800 1.069, launch_wide/mi300x 1.066, valu_f32/rx6800 1.058, startup/rx6800 1.055, p_dpp_scan/mi300x 1.051
```
