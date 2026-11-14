# Blocks, step by step: every kernel row of the AMD bench (round sp73)

`./qall.sh final 7 base noop p3 p3:VGPU_AMD_BLOCKS=1,VGPU_AMD_BLOCKS_BOUND=0 p3:VGPU_AMD_BLOCKS=1` on a quiet host: for each row the
device CPU seconds of the base (best of 7 single runs, the builds taking turns, pinned to cpus 14,15) and that time over each other build's
(above 1: faster than the base). *control*: the base plus one function nothing calls in `gcn_exec.cpp`. *knob off*: this branch,
`VGPU_AMD_BLOCKS` unset. *blocks*: `VGPU_AMD_BLOCKS=1 VGPU_AMD_BLOCKS_BOUND=0` (the Machine's instructions as members, waves that go on in
a block, the two trials). *blocks + bound*: `VGPU_AMD_BLOCKS=1` (bound integer runs too). One host thread.

| row | target | base (s) | control | knob off | blocks | blocks + bound |
|---|---|---:|---:|---:|---:|---:|
| valu_int | mi300x | 0.0859 | 0.995 | 0.997 | 1.161 | 1.326 |
| valu_int | rx6800 | 0.1369 | 0.997 | 0.998 | 1.206 | 1.341 |
| valu_f32 | mi300x | 0.0708 | 0.979 | 0.980 | 1.119 | 1.087 |
| valu_f32 | rx6800 | 0.1242 | 0.997 | 1.004 | 1.202 | 1.204 |
| valu_f64 | mi300x | 0.1182 | 0.996 | 1.000 | 1.115 | 1.101 |
| valu_f64 | rx6800 | 0.1636 | 0.994 | 0.987 | 1.085 | 1.088 |
| mem | mi300x | 0.1524 | 0.984 | 1.002 | 1.006 | 1.016 |
| mem | rx6800 | 0.2352 | 1.003 | 1.012 | 1.018 | 1.014 |
| lds | mi300x | 0.0945 | 1.001 | 1.010 | 1.011 | 1.007 |
| lds | rx6800 | 0.1500 | 0.997 | 0.995 | 1.005 | 0.990 |
| narrow | mi300x | 0.2853 | 1.003 | 1.004 | 1.006 | 1.000 |
| narrow | rx6800 | 0.6132 | 1.005 | 1.036 | 1.024 | 1.031 |
| smem | mi300x | 0.2333 | 1.023 | 0.988 | 1.324 | 1.323 |
| smem | rx6800 | 0.3923 | 1.016 | 0.990 | 1.303 | 1.306 |
| buf | mi300x | 0.2609 | 0.998 | 0.982 | 0.984 | 0.982 |
| buf | rx6800 | 0.3667 | 1.001 | 0.984 | 0.984 | 0.986 |
| ldsx | mi300x | 0.2914 | 0.996 | 1.006 | 0.999 | 1.005 |
| ldsx | rx6800 | 0.4009 | 1.001 | 1.001 | 0.996 | 0.997 |
| flat | mi300x | 0.2112 | 1.013 | 1.001 | 1.010 | 1.014 |
| flat | rx6800 | 0.3703 | 0.990 | 0.993 | 1.010 | 1.005 |
| atomics | mi300x | 0.3829 | 0.995 | 1.001 | 1.059 | 1.025 |
| atomics | rx6800 | 0.2610 | 0.988 | 0.986 | 1.032 | 1.052 |
| divergent | mi300x | 0.2116 | 1.007 | 1.016 | 1.156 | 1.158 |
| divergent | rx6800 | 0.3117 | 1.021 | 1.018 | 1.240 | 1.243 |
| wave | mi300x | 0.0814 | 0.996 | 1.002 | 1.224 | 1.233 |
| wave | rx6800 | 0.1308 | 1.004 | 0.990 | 1.281 | 1.289 |
| p_dpp | mi300x | 0.0921 | 0.998 | 1.004 | 1.044 | 1.047 |
| p_dpp | rx6800 | 0.1086 | 0.998 | 1.001 | 1.025 | 1.025 |
| p_dpp_arch | mi300x | 0.0523 | 1.013 | 1.000 | 1.090 | 1.051 |
| p_dpp_arch | rx6800 | 0.0476 | 0.991 | 0.994 | 1.059 | 1.051 |
| p_dpp_scan | mi300x | 0.0534 | 0.999 | 0.998 | 1.052 | 1.071 |
| p_dpp_scan | rx6800 | 0.0633 | 0.999 | 0.990 | 1.098 | 1.085 |
| p_perm | mi300x | 0.0470 | 1.005 | 1.016 | 1.064 | 1.079 |
| p_perm | rx6800 | 0.0614 | 1.002 | 0.998 | 1.105 | 1.115 |
| p_lane | mi300x | 0.0209 | 1.017 | 0.986 | 1.228 | 1.220 |
| p_lane | rx6800 | 0.0411 | 1.014 | 0.980 | 1.354 | 1.353 |
| p_pk16 | mi300x | 0.1469 | 1.003 | 1.022 | 1.161 | 1.160 |
| p_pk16 | rx6800 | 0.5811 | 0.990 | 1.005 | 1.043 | 1.048 |
| p_pkf16 | mi300x | 0.1993 | 1.005 | 1.008 | 1.005 | 1.005 |
| p_pkf16 | rx6800 | 0.2144 | 1.002 | 1.007 | 0.999 | 1.001 |
| p_sdwa | mi300x | 0.1005 | 0.996 | 0.994 | 1.099 | 1.199 |
| p_sdwa | rx6800 | 0.1364 | 0.998 | 0.989 | 1.180 | 1.296 |
| p_vopd | mi300x | 0.0970 | 1.001 | 1.008 | 1.135 | 1.140 |
| p_vopd | rx6800 | 0.1260 | 0.996 | 1.017 | 1.269 | 1.265 |
| p_cmp | mi300x | 0.2082 | 1.007 | 0.998 | 1.193 | 1.203 |
| p_cmp | rx6800 | 0.3142 | 0.992 | 0.965 | 1.263 | 1.282 |
| p_i64 | mi300x | 0.1404 | 0.980 | 0.974 | 1.162 | 1.242 |
| p_i64 | rx6800 | 0.1925 | 1.005 | 1.003 | 1.268 | 1.323 |
| p_f64x | mi300x | 0.4843 | 1.044 | 0.999 | 1.043 | 1.040 |
| p_f64x | rx6800 | 0.5825 | 1.017 | 0.981 | 1.050 | 1.077 |
| p_smem_wide | mi300x | 0.4732 | 0.999 | 1.005 | 1.196 | 1.200 |
| p_smem_wide | rx6800 | 0.9198 | 1.003 | 0.997 | 1.188 | 1.180 |
| p_sbuf | mi300x | 0.1014 | 1.002 | 1.030 | 1.094 | 1.098 |
| p_sbuf | rx6800 | 0.1878 | 1.013 | 1.048 | 1.101 | 1.103 |
| mfma | mi300x | 0.1978 | 1.000 | 0.990 | 0.985 | 0.983 |
| mfma_i8 | mi300x | 0.0833 | 1.005 | 1.011 | 1.018 | 0.997 |
| mfma_fp8 | mi300x | 0.2592 | 1.008 | 1.004 | 1.001 | 1.002 |
| mfma_bf16 | mi300x | 0.1886 | 0.994 | 0.996 | 0.998 | 0.995 |
| mfma_f64 | mi300x | 0.1614 | 0.999 | 1.008 | 1.005 | 1.002 |
| mfma_f64_4b | mi300x | 0.1090 | 0.999 | 1.005 | 1.003 | 0.995 |
| smfmac | mi300x | 0.2960 | 0.998 | 1.071 | 1.060 | 1.058 |
| p_permlane | rx6800 | 0.0392 | 0.998 | 1.002 | 1.179 | 1.158 |

control: geometric mean 1.001 over 62 rows; below 0.97: none
knob off: geometric mean 1.001 over 62 rows; below 0.97: p_cmp/rx6800 0.965
blocks: geometric mean 1.099 over 62 rows; below 0.97: none
blocks + bound: geometric mean 1.107 over 62 rows; below 0.97: none

The knob on against the knob off (the last column over the third): geometric mean 1.105; lowest mfma_i8 0.986, smfmac 0.988, mfma_f64_4b
0.991, mfma 0.993 (the matrix rows, which run no block to speak of); no row more than 2 % slower with blocks on than with them off.

A pass of the same command taken while the mutants' builds ran beside it is not here: its control rows went from 0.95 to 1.10.

## The final library (frame pointers in the three functions of the blocks' loop)

`./qall.sh final2 7 base noop p3 p3:VGPU_AMD_BLOCKS=1`, the table above taken again on the library of the last commit, quiet host, one
host thread:

| row | target | base (s) | control | knob off | knob on |
|---|---|---:|---:|---:|---:|
| valu_int | mi300x | 0.0891 | 0.998 | 0.996 | 1.326 |
| valu_int | rx6800 | 0.1385 | 0.997 | 0.997 | 1.345 |
| valu_f32 | mi300x | 0.0733 | 0.997 | 0.997 | 1.123 |
| valu_f32 | rx6800 | 0.1261 | 0.999 | 1.005 | 1.207 |
| valu_f64 | mi300x | 0.1207 | 0.993 | 1.010 | 1.091 |
| valu_f64 | rx6800 | 0.1677 | 1.000 | 0.993 | 1.084 |
| mem | mi300x | 0.1526 | 0.986 | 1.003 | 1.006 |
| mem | rx6800 | 0.2378 | 0.989 | 1.007 | 1.030 |
| lds | mi300x | 0.0948 | 0.996 | 1.003 | 0.990 |
| lds | rx6800 | 0.1513 | 0.997 | 0.978 | 0.989 |
| narrow | mi300x | 0.2870 | 1.003 | 0.995 | 1.010 |
| narrow | rx6800 | 0.6143 | 1.006 | 0.983 | 1.006 |
| smem | mi300x | 0.2325 | 1.016 | 0.987 | 1.319 |
| smem | rx6800 | 0.3928 | 1.018 | 0.946 | 1.314 |
| buf | mi300x | 0.2619 | 1.003 | 0.977 | 0.991 |
| buf | rx6800 | 0.3673 | 1.004 | 0.981 | 0.992 |
| ldsx | mi300x | 0.2920 | 0.994 | 1.009 | 1.007 |
| ldsx | rx6800 | 0.4034 | 1.013 | 0.988 | 1.006 |
| flat | mi300x | 0.2107 | 1.009 | 1.017 | 1.012 |
| flat | rx6800 | 0.3697 | 0.991 | 0.997 | 1.029 |
| atomics | mi300x | 0.3842 | 0.996 | 1.020 | 1.037 |
| atomics | rx6800 | 0.2605 | 0.990 | 0.959 | 1.044 |
| divergent | mi300x | 0.2105 | 1.003 | 1.006 | 1.151 |
| divergent | rx6800 | 0.3107 | 1.019 | 1.023 | 1.249 |
| wave | mi300x | 0.0809 | 1.003 | 0.989 | 1.223 |
| wave | rx6800 | 0.1297 | 1.000 | 0.981 | 1.270 |
| p_dpp | mi300x | 0.0914 | 0.994 | 0.983 | 1.037 |
| p_dpp | rx6800 | 0.1078 | 0.992 | 0.990 | 1.020 |
| p_dpp_arch | mi300x | 0.0508 | 0.994 | 0.971 | 1.084 |
| p_dpp_arch | rx6800 | 0.0471 | 0.987 | 1.005 | 1.051 |
| p_dpp_scan | mi300x | 0.0531 | 0.983 | 0.986 | 1.063 |
| p_dpp_scan | rx6800 | 0.0630 | 0.992 | 0.979 | 1.084 |
| p_perm | mi300x | 0.0467 | 0.997 | 0.990 | 1.058 |
| p_perm | rx6800 | 0.0621 | 1.002 | 1.004 | 1.100 |
| p_lane | mi300x | 0.0208 | 0.984 | 0.979 | 1.211 |
| p_lane | rx6800 | 0.0411 | 0.997 | 0.977 | 1.361 |
| p_pk16 | mi300x | 0.1459 | 1.000 | 0.999 | 1.159 |
| p_pk16 | rx6800 | 0.5808 | 0.984 | 1.009 | 1.040 |
| p_pkf16 | mi300x | 0.1997 | 1.006 | 0.997 | 1.008 |
| p_pkf16 | rx6800 | 0.2139 | 1.001 | 0.999 | 0.995 |
| p_sdwa | mi300x | 0.1003 | 0.995 | 0.995 | 1.202 |
| p_sdwa | rx6800 | 0.1368 | 0.998 | 0.991 | 1.300 |
| p_vopd | mi300x | 0.0964 | 1.001 | 0.991 | 1.141 |
| p_vopd | rx6800 | 0.1256 | 0.995 | 1.011 | 1.251 |
| p_cmp | mi300x | 0.2074 | 1.000 | 0.981 | 1.220 |
| p_cmp | rx6800 | 0.3185 | 0.988 | 1.004 | 1.290 |
| p_i64 | mi300x | 0.1412 | 0.986 | 0.964 | 1.250 |
| p_i64 | rx6800 | 0.1916 | 1.007 | 0.995 | 1.318 |
| p_f64x | mi300x | 0.4678 | 0.971 | 0.970 | 1.011 |
| p_f64x | rx6800 | 0.5838 | 1.033 | 0.991 | 1.048 |
| p_smem_wide | mi300x | 0.4726 | 1.000 | 1.001 | 1.197 |
| p_smem_wide | rx6800 | 0.9195 | 1.005 | 0.993 | 1.184 |
| p_sbuf | mi300x | 0.1014 | 1.002 | 0.994 | 1.081 |
| p_sbuf | rx6800 | 0.1872 | 1.015 | 1.002 | 1.088 |
| mfma | mi300x | 0.1969 | 0.993 | 0.985 | 0.987 |
| mfma_i8 | mi300x | 0.0824 | 0.991 | 0.996 | 0.991 |
| mfma_fp8 | mi300x | 0.2588 | 1.007 | 1.010 | 1.005 |
| mfma_bf16 | mi300x | 0.1881 | 0.995 | 0.996 | 0.998 |
| mfma_f64 | mi300x | 0.1610 | 1.000 | 0.997 | 0.983 |
| mfma_f64_4b | mi300x | 0.1084 | 0.992 | 0.996 | 0.987 |
| smfmac | mi300x | 0.2954 | 1.001 | 1.002 | 0.988 |
| p_permlane | rx6800 | 0.0390 | 1.008 | 0.991 | 1.119 |

control: geometric mean 0.999 over 62 rows; below 0.97: none
knob off: geometric mean 0.993 over 62 rows; below 0.97: smem/rx6800 0.946, atomics/rx6800 0.959, p_i64/mi300x 0.964
knob on: geometric mean 1.103 over 62 rows; below 0.97: none

The three knob-off rows below 0.97 here (smem on the RX 6800, atomics on the RX 6800, p_i64 on the MI300X) were 0.990, 0.997 and 0.974 in
the table above, whose library differs from this one by two frame pointers in functions the knob-off path does not run; the rows that were
lowest there (p_cmp on the RX 6800 0.965) are 1.00 here. That is what a rebuild does to two or three rows with link-time optimization.

## Eight simulator threads

`THREADS=8 ROWS="valu_int valu_f64 wave p_i64 smem lds mem buf narrow flat atomics divergent" MROWS=mfma ./qall.sh all8 5 p3 p3:VGPU_AMD_BLOCKS=1`:
`VGPU_THREADS=8`, the process still pinned to the two hardware threads of one core (so the threads take turns; this shows that nothing
goes wrong or slow with several threads in blocks, not a rate on eight cores). Device CPU seconds summed over threads, best of 5, the final
library, knob on against knob off:

| row | target | knob off (s) | knob on |
|---|---|---:|---:|
| valu_int | mi300x | 0.1680 | 1.387 |
| valu_int | rx6800 | 0.2680 | 1.372 |
| valu_f64 | mi300x | 0.2173 | 1.093 |
| valu_f64 | rx6800 | 0.3193 | 1.112 |
| wave | mi300x | 0.1477 | 1.191 |
| wave | rx6800 | 0.2440 | 1.289 |
| p_i64 | mi300x | 0.2601 | 1.224 |
| p_i64 | rx6800 | 0.3618 | 1.321 |
| smem | mi300x | 0.4390 | 1.310 |
| smem | rx6800 | 0.7264 | 1.379 |
| lds | mi300x | 0.1549 | 1.010 |
| lds | rx6800 | 0.2687 | 1.017 |
| mem | mi300x | 0.2830 | 1.000 |
| mem | rx6800 | 0.4433 | 1.003 |
| buf | mi300x | 0.5015 | 1.002 |
| buf | rx6800 | 0.6942 | 1.004 |
| narrow | mi300x | 0.5153 | 0.996 |
| narrow | rx6800 | 1.1435 | 0.999 |
| flat | mi300x | 0.3949 | 0.987 |
| flat | rx6800 | 0.6948 | 0.999 |
| atomics | mi300x | 1.4901 | 1.009 |
| atomics | rx6800 | 0.9297 | 1.017 |
| divergent | mi300x | 0.3898 | 1.135 |
| divergent | rx6800 | 0.5581 | 1.203 |
| mfma | mi300x | 0.4025 | 0.986 |
| p_permlane | rx6800 | 0.0689 | 1.108 |

knob on: geometric mean 1.113 over 26 rows; below 0.97: none

## A variant that was not kept: the blocks' loop in a text section of its own

To see whether those three rows could be had back, the three functions of the blocks' loop were given `gnu::section(".text.vgpu_blocks")`
(nothing else changed; the test passes, the sampler check passes 12 of 12). `./qall.sh sec 9 base noop p3fp p3 p3:VGPU_AMD_BLOCKS=1` on the
rows in doubt (*final*: the library of the last commit; *section*: the variant, knob off; *section, on*: the variant, knob on):

| row | target | base (s) | control | final | section | section, on |
|---|---|---:|---:|---:|---:|---:|
| smem | mi300x | 0.2328 | 1.011 | 0.943 | 1.010 | 1.238 |
| smem | rx6800 | 0.3948 | 1.023 | 0.930 | 1.011 | 1.217 |
| p_i64 | mi300x | 0.1419 | 0.999 | 0.966 | 0.977 | 1.265 |
| p_i64 | rx6800 | 0.1931 | 0.993 | 1.002 | 0.998 | 1.329 |
| atomics | mi300x | 0.3833 | 1.029 | 1.022 | 1.046 | 1.077 |
| atomics | rx6800 | 0.2592 | 0.985 | 0.972 | 1.031 | 1.062 |
| p_cmp | mi300x | 0.2066 | 1.006 | 1.004 | 1.013 | 1.204 |
| p_cmp | rx6800 | 0.3132 | 1.005 | 0.989 | 0.983 | 1.278 |
| buf | mi300x | 0.2620 | 1.001 | 0.969 | 0.971 | 1.000 |
| buf | rx6800 | 0.3667 | 1.003 | 0.958 | 0.970 | 1.003 |
| valu_f64 | mi300x | 0.1204 | 0.997 | 0.973 | 1.019 | 1.095 |
| valu_f64 | rx6800 | 0.1678 | 1.007 | 0.989 | 0.990 | 1.088 |
| p_dpp_arch | mi300x | 0.0519 | 1.028 | 1.008 | 0.992 | 1.084 |
| p_dpp_arch | rx6800 | 0.0472 | 0.992 | 0.990 | 1.001 | 1.049 |
| p_f64x | mi300x | 0.4819 | 1.000 | 0.998 | 1.022 | 1.039 |
| p_f64x | rx6800 | 0.5804 | 0.999 | 1.025 | 1.016 | 1.066 |
| lds | mi300x | 0.0952 | 1.005 | 0.980 | 0.993 | 1.014 |
| lds | rx6800 | 0.1518 | 1.005 | 0.971 | 0.975 | 1.013 |
| valu_int | mi300x | 0.0898 | 1.010 | 0.992 | 0.991 | 1.339 |
| valu_int | rx6800 | 0.1401 | 1.003 | 1.003 | 0.999 | 1.342 |
| smfmac | mi300x | 0.2938 | 0.996 | 0.993 | 0.983 | 0.976 |
| p_permlane | rx6800 | 0.0386 | 0.996 | 0.993 | 0.993 | 1.150 |

control: geometric mean 1.004 over 22 rows; below 0.97: none
final: geometric mean 0.985 over 22 rows; below 0.97: smem/mi300x 0.943, smem/rx6800 0.930, p_i64/mi300x 0.966, buf/mi300x 0.969, buf/rx6800 0.958
section: geometric mean 0.999 over 22 rows; below 0.97: none
section, on: geometric mean 1.127 over 22 rows; below 0.97: none

and then all 62 rows of the variant (`./qall.sh final3 7 ...`, the first table's command): control 1.002 (p_f64x on the RX 6800 0.967), knob
off **0.996** with eight rows below 0.97 (narrow on the RX 6800 0.947, mem 0.957 and 0.953, p_f64x on the RX 6800 0.956, p_dpp_arch on the
MI300X 0.963, mfma 0.963, lds on the RX 6800 0.970, buf on the MI300X 0.970), knob on 1.100 with narrow on the RX 6800 0.957 and mfma 0.961
of the base, smem's gain 1.24 and 1.21 where it was 1.31. It gave smem back and took mem and narrow: another placement, not a better one.
Not kept; the committed library is the one of the section above.
