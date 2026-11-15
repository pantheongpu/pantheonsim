# The AMD bench's kernel rows, step by step (round sp74)

`qall.sh` (round sp73's): for each row the device CPU seconds of the first build (best of n single runs, the builds taking turns, pinned
to cpus 14,15, one host thread) and that time over each other build's (above 1: faster than the first). *base*: origin/sp69-hotspot-start
a33ead36. *control*: the base plus one function nothing calls in `gcn_tier.cpp`. The host was quiet for every table here.

## The final library, all 62 rows

`./qall.sh final 7 base noop w4 base:VGPU_AMD_BLOCKS=1 w4:VGPU_AMD_BLOCKS=1`. *final*: this round, knob off. *base on*, *final on*: the
base and this round with `VGPU_AMD_BLOCKS=1`.

| row | target | base (s) | control | final | base on | final on |
|---|---|---:|---:|---:|---:|---:|
| valu_int | mi300x | 0.0866 | 1.016 | 1.015 | 1.349 | 1.350 |
| valu_int | rx6800 | 0.1398 | 1.028 | 1.027 | 1.344 | 1.372 |
| valu_f32 | mi300x | 0.0724 | 1.005 | 1.009 | 1.132 | 1.142 |
| valu_f32 | rx6800 | 0.1235 | 0.999 | 0.999 | 1.182 | 1.186 |
| valu_f64 | mi300x | 0.1199 | 0.999 | 1.027 | 1.092 | 1.120 |
| valu_f64 | rx6800 | 0.1702 | 1.005 | 1.000 | 1.100 | 1.125 |
| mem | mi300x | 0.1574 | 1.017 | 1.212 | 1.034 | 1.223 |
| mem | rx6800 | 0.2414 | 1.007 | 1.044 | 1.032 | 1.052 |
| lds | mi300x | 0.0954 | 1.010 | 1.008 | 1.016 | 1.015 |
| lds | rx6800 | 0.1512 | 0.993 | 0.991 | 1.022 | 1.021 |
| narrow | mi300x | 0.2905 | 0.999 | 1.120 | 1.029 | 1.124 |
| narrow | rx6800 | 0.6829 | 1.023 | 1.057 | 1.034 | 1.063 |
| smem | mi300x | 0.2422 | 1.043 | 1.057 | 1.317 | 1.374 |
| smem | rx6800 | 0.3952 | 1.013 | 1.011 | 1.307 | 1.322 |
| buf | mi300x | 0.2649 | 0.994 | 0.996 | 1.012 | 1.004 |
| buf | rx6800 | 0.3708 | 0.994 | 0.996 | 1.017 | 1.008 |
| ldsx | mi300x | 0.2908 | 1.004 | 0.994 | 0.996 | 0.990 |
| ldsx | rx6800 | 0.4043 | 1.005 | 0.992 | 1.020 | 1.025 |
| flat | mi300x | 0.2118 | 1.008 | 1.059 | 1.027 | 1.066 |
| flat | rx6800 | 0.3728 | 1.002 | 1.009 | 1.014 | 1.021 |
| atomics | mi300x | 0.3845 | 0.986 | 0.999 | 1.059 | 1.040 |
| atomics | rx6800 | 0.2532 | 0.965 | 0.954 | 1.052 | 1.049 |
| divergent | mi300x | 0.2030 | 0.985 | 1.045 | 1.117 | 1.224 |
| divergent | rx6800 | 0.2940 | 0.978 | 1.016 | 1.197 | 1.262 |
| wave | mi300x | 0.0817 | 1.000 | 1.004 | 1.236 | 1.238 |
| wave | rx6800 | 0.1314 | 0.980 | 1.000 | 1.313 | 1.303 |
| p_dpp | mi300x | 0.0919 | 1.103 | 0.973 | 1.044 | 1.031 |
| p_dpp | rx6800 | 0.1085 | 1.001 | 1.014 | 1.022 | 1.003 |
| p_dpp_arch | mi300x | 0.0522 | 1.002 | 0.979 | 1.240 | 1.038 |
| p_dpp_arch | rx6800 | 0.0469 | 0.996 | 0.998 | 1.028 | 1.048 |
| p_dpp_scan | mi300x | 0.0543 | 1.012 | 1.002 | 1.103 | 1.075 |
| p_dpp_scan | rx6800 | 0.0648 | 1.020 | 1.009 | 1.116 | 1.111 |
| p_perm | mi300x | 0.0466 | 0.988 | 1.001 | 1.068 | 1.068 |
| p_perm | rx6800 | 0.0613 | 0.986 | 1.003 | 1.103 | 1.113 |
| p_lane | mi300x | 0.0216 | 1.007 | 1.022 | 1.301 | 1.282 |
| p_lane | rx6800 | 0.0433 | 0.975 | 1.022 | 1.412 | 1.429 |
| p_pk16 | mi300x | 0.1462 | 1.007 | 1.009 | 1.159 | 1.157 |
| p_pk16 | rx6800 | 0.6674 | 1.001 | 1.009 | 1.041 | 1.046 |
| p_pkf16 | mi300x | 0.1982 | 1.005 | 0.996 | 0.986 | 0.980 |
| p_pkf16 | rx6800 | 0.2140 | 0.995 | 1.000 | 0.988 | 0.986 |
| p_sdwa | mi300x | 0.1019 | 1.016 | 1.834 | 1.196 | 2.543 |
| p_sdwa | rx6800 | 0.1399 | 1.015 | 1.513 | 1.308 | 2.214 |
| p_vopd | mi300x | 0.0976 | 1.018 | 1.014 | 1.160 | 1.156 |
| p_vopd | rx6800 | 0.1241 | 0.999 | 1.001 | 1.231 | 1.248 |
| p_cmp | mi300x | 0.2118 | 0.994 | 1.091 | 1.239 | 1.308 |
| p_cmp | rx6800 | 0.3175 | 0.979 | 1.043 | 1.265 | 1.351 |
| p_i64 | mi300x | 0.1484 | 1.021 | 1.091 | 1.314 | 1.441 |
| p_i64 | rx6800 | 0.1943 | 1.003 | 1.030 | 1.332 | 1.362 |
| p_f64x | mi300x | 0.4849 | 1.006 | 2.400 | 1.043 | 2.682 |
| p_f64x | rx6800 | 0.5875 | 1.015 | 2.077 | 1.075 | 2.426 |
| p_smem_wide | mi300x | 0.4857 | 1.036 | 1.033 | 1.199 | 1.227 |
| p_smem_wide | rx6800 | 0.9545 | 1.040 | 1.037 | 1.192 | 1.214 |
| p_sbuf | mi300x | 0.1017 | 0.994 | 0.996 | 1.084 | 1.089 |
| p_sbuf | rx6800 | 0.1884 | 1.007 | 1.011 | 1.091 | 1.090 |
| mfma | mi300x | 0.1990 | 0.998 | 1.000 | 1.000 | 1.002 |
| mfma_i8 | mi300x | 0.0822 | 1.003 | 0.996 | 0.994 | 0.994 |
| mfma_fp8 | mi300x | 0.2555 | 0.999 | 0.996 | 0.994 | 0.997 |
| mfma_bf16 | mi300x | 0.1895 | 0.998 | 1.005 | 1.000 | 1.005 |
| mfma_f64 | mi300x | 0.1619 | 1.008 | 1.015 | 0.985 | 1.008 |
| mfma_f64_4b | mi300x | 0.1079 | 0.965 | 0.999 | 0.987 | 0.992 |
| smfmac | mi300x | 0.2749 | 0.996 | 0.999 | 0.994 | 0.992 |
| p_permlane | rx6800 | 0.0390 | 0.986 | 1.004 | 1.163 | 1.161 |

control: geometric mean 1.004 over 62 rows; below 0.97: atomics/rx6800 0.965, mfma_f64_4b/mi300x 0.965
final: geometric mean 1.060 over 62 rows; below 0.97: atomics/rx6800 0.954
base on: geometric mean 1.115 over 62 rows; below 0.97: none
final on: geometric mean 1.184 over 62 rows; below 0.97: none

The one row of *final* below 0.97, atomics on the RX 6800 (0.954), is 0.965 in the control: it runs none of the changed handlers.

## Step 1: the row handlers (`w1`, commit a9dbf7e8) against the base

`ROWS="valu_f64 p_f64x p_cmp p_i64 valu_int p_sdwa lds" MROWS=mfma_f64 ./qall.sh s1 7 base noop w1`

| row | target | base (s) | control | step 1 |
|---|---|---:|---:|---:|
| valu_f64 | mi300x | 0.1146 | 0.991 | 0.992 |
| valu_f64 | rx6800 | 0.1652 | 1.006 | 1.004 |
| p_f64x | mi300x | 0.4663 | 0.975 | 2.320 |
| p_f64x | rx6800 | 0.5857 | 1.005 | 2.061 |
| p_cmp | mi300x | 0.2073 | 0.985 | 1.020 |
| p_cmp | rx6800 | 0.3216 | 1.003 | 1.023 |
| p_i64 | mi300x | 0.1464 | 1.006 | 1.060 |
| p_i64 | rx6800 | 0.1936 | 1.005 | 1.007 |
| valu_int | mi300x | 0.0923 | 1.025 | 1.030 |
| valu_int | rx6800 | 0.1419 | 1.020 | 1.025 |
| p_sdwa | mi300x | 0.1018 | 1.013 | 1.847 |
| p_sdwa | rx6800 | 0.1406 | 1.030 | 1.539 |
| lds | mi300x | 0.0935 | 0.993 | 0.986 |
| lds | rx6800 | 0.1516 | 1.000 | 1.000 |
| mfma_f64 | mi300x | 0.1616 | 1.006 | 1.015 |
| p_permlane | rx6800 | 0.0392 | 1.001 | 1.025 |

control: geometric mean 1.004 over 16 rows; below 0.97: none
step 1: geometric mean 1.191 over 16 rows; below 0.97: none

## Step 2, not kept: bound binary64 runs in blocks (`w2`) against step 1, knob on

`./qall.sh s2 7 w1 w1:VGPU_AMD_BLOCKS=1 w2 w2:VGPU_AMD_BLOCKS=1 w2:VGPU_AMD_BLOCKS=1,VGPU_AMD_BLOCKS_BOUND=0`. The first column is step 1
with the knob off; *step 1 on* against *step 2 on* is the comparison (valu_f64: 1.093 and 1.083 before, 1.099 and 1.072 after).

| row | target | w1 (s) | step 1 on | step 2 off | step 2 on | step 2 on, unbound |
|---|---|---:|---:|---:|---:|---:|
| valu_f64 | mi300x | 0.1168 | 1.093 | 1.010 | 1.099 | 1.075 |
| valu_f64 | rx6800 | 0.1652 | 1.083 | 0.998 | 1.072 | 1.073 |
| p_f64x | mi300x | 0.2003 | 1.087 | 1.001 | 1.129 | 1.109 |
| p_f64x | rx6800 | 0.2821 | 1.137 | 0.982 | 1.145 | 1.146 |
| p_cmp | mi300x | 0.2012 | 1.230 | 1.006 | 1.208 | 1.213 |
| p_cmp | rx6800 | 0.3115 | 1.282 | 1.002 | 1.283 | 1.270 |
| p_i64 | mi300x | 0.1386 | 1.318 | 0.997 | 1.306 | 1.194 |
| p_i64 | rx6800 | 0.1916 | 1.321 | 0.993 | 1.288 | 1.256 |
| valu_int | mi300x | 0.0899 | 1.335 | 0.996 | 1.327 | 1.184 |
| valu_int | rx6800 | 0.1393 | 1.340 | 1.004 | 1.298 | 1.208 |
| wave | mi300x | 0.0813 | 1.229 | 0.998 | 1.238 | 1.233 |
| wave | rx6800 | 0.1324 | 1.319 | 1.006 | 1.291 | 1.302 |
| mfma_f64 | mi300x | 0.1588 | 0.993 | 0.986 | 0.975 | 0.978 |
| p_permlane | rx6800 | 0.0384 | 1.147 | 1.002 | 1.119 | 1.159 |

step 1 on: geometric mean 1.203 over 14 rows; below 0.97: none
step 2 off: geometric mean 0.999 over 14 rows; below 0.97: none
step 2 on: geometric mean 1.194 over 14 rows; below 0.97: none
step 2 on, unbound: geometric mean 1.168 over 14 rows; below 0.97: none

## Step 3 as first written, and its parts (`w3`, `w3A`, `w3B`)

`w3`: step 1 plus (a) a comparison as the one test it is, the integers' on the rows of words, (b) a pair's modifiers applied in the
vectors, (c) only the groups of eight lanes that hold a lane of EXEC, in `h_f64r` and in the new handlers, (d) the lanes of a pair kept in
the order the words interleave (0, 1, 4, 5 and 2, 3, 6, 7: no move across the halves of a vector on a load or a store), in the kernel of
`v_add_f64`, `v_mul_f64`, `v_fma_f64` and in the new handlers. `w3A`: `w3` with the old kernel (no (d) there). `w3B`: `w3A` with the old
`h_f64r` (no (c) there). `./qall.sh var 9 base w1 w3 w3A w3B`:

| row | target | base (s) | step 1 | w3 | w3A | w3B |
|---|---|---:|---:|---:|---:|---:|
| valu_f64 | mi300x | 0.1180 | 0.999 | 0.935 | 0.968 | 0.993 |
| valu_f64 | rx6800 | 0.1680 | 0.996 | 0.911 | 0.967 | 0.997 |
| p_f64x | mi300x | 0.4690 | 2.305 | 2.060 | 2.155 | 2.182 |
| p_f64x | rx6800 | 0.5866 | 2.059 | 1.796 | 1.831 | 1.868 |
| p_cmp | mi300x | 0.2066 | 1.021 | 1.027 | 1.051 | 1.058 |
| p_cmp | rx6800 | 0.3186 | 1.019 | 1.017 | 1.036 | 1.049 |
| mfma_f64 | mi300x | 0.1615 | 1.015 | 1.022 | 1.014 | 1.019 |
| p_permlane | rx6800 | 0.0387 | 0.994 | 0.989 | 1.009 | 0.994 |

step 1: geometric mean 1.222 over 8 rows; below 0.97: none
w3: geometric mean 1.162 over 8 rows; below 0.97: valu_f64/mi300x 0.935, valu_f64/rx6800 0.911
w3A: geometric mean 1.194 over 8 rows; below 0.97: valu_f64/mi300x 0.968, valu_f64/rx6800 0.967
w3B: geometric mean 1.208 over 8 rows; below 0.97: none

valu_f64 loses 3 % to (c) in `h_f64r` (with every lane on it only adds work) and 4 % more to (d) in the kernel; p_f64x loses 6 to 9 % to
the new handlers' form of (b), (c) and (d) together (`w3B` against step 1), and only p_cmp gains (a). poisson with (c) in `h_f64r`: 109.6 s
of CPU against 110.9 without it (`w3A`, `w3B`, one run each). Kept from it: (a), and (c) in `h_f64r` where a lane is off.

## Step 3 as kept (`w4`, commit 05a898c7) against step 1

`ROWS="valu_f64 p_f64x p_cmp p_i64" MROWS=mfma_f64 ./qall.sh s4 9 base noop w1 w4`

| row | target | base (s) | control | step 1 | step 3 |
|---|---|---:|---:|---:|---:|
| valu_f64 | mi300x | 0.1198 | 1.004 | 1.005 | 1.028 |
| valu_f64 | rx6800 | 0.1688 | 0.992 | 0.995 | 0.990 |
| p_f64x | mi300x | 0.4680 | 0.969 | 2.292 | 2.299 |
| p_f64x | rx6800 | 0.5870 | 1.007 | 2.057 | 2.078 |
| p_cmp | mi300x | 0.2082 | 1.004 | 1.032 | 1.091 |
| p_cmp | rx6800 | 0.3209 | 1.004 | 1.028 | 1.068 |
| p_i64 | mi300x | 0.1473 | 1.007 | 1.082 | 1.084 |
| p_i64 | rx6800 | 0.1937 | 1.004 | 1.011 | 1.041 |
| mfma_f64 | mi300x | 0.1617 | 1.012 | 1.016 | 1.013 |
| p_permlane | rx6800 | 0.0388 | 0.984 | 1.010 | 0.996 |

control: geometric mean 0.999 over 10 rows; below 0.97: p_f64x/mi300x 0.969
step 1: geometric mean 1.188 over 10 rows; below 0.97: none
step 3: geometric mean 1.205 over 10 rows; below 0.97: none

## The library after the test fix (`w5`, commit fdaca835), all 62 rows again

The fix of the tier's test changed test state that is compiled into the library (`digest_instruction`), so the table was taken again:
`./qall.sh final2 7 base noop w5 w5:VGPU_AMD_BLOCKS=1`.

| row | target | base (s) | control | final | final on |
|---|---|---:|---:|---:|---:|
| valu_int | mi300x | 0.0905 | 1.017 | 1.021 | 1.367 |
| valu_int | rx6800 | 0.1411 | 1.021 | 1.024 | 1.376 |
| valu_f32 | mi300x | 0.0728 | 0.993 | 0.998 | 1.123 |
| valu_f32 | rx6800 | 0.1259 | 1.001 | 1.001 | 1.193 |
| valu_f64 | mi300x | 0.1199 | 0.999 | 1.024 | 1.107 |
| valu_f64 | rx6800 | 0.1689 | 0.999 | 0.994 | 1.117 |
| mem | mi300x | 0.1577 | 1.032 | 1.208 | 1.236 |
| mem | rx6800 | 0.2406 | 1.009 | 1.030 | 1.051 |
| lds | mi300x | 0.0941 | 0.997 | 0.991 | 1.002 |
| lds | rx6800 | 0.1514 | 1.000 | 1.000 | 1.019 |
| narrow | mi300x | 0.2945 | 1.019 | 1.133 | 1.144 |
| narrow | rx6800 | 0.6817 | 1.024 | 1.055 | 1.068 |
| smem | mi300x | 0.2425 | 1.050 | 1.055 | 1.376 |
| smem | rx6800 | 0.3941 | 1.010 | 1.010 | 1.275 |
| buf | mi300x | 0.2647 | 0.993 | 0.999 | 1.004 |
| buf | rx6800 | 0.3706 | 0.994 | 1.001 | 1.010 |
| ldsx | mi300x | 0.2938 | 1.010 | 1.007 | 0.998 |
| ldsx | rx6800 | 0.4038 | 1.003 | 0.984 | 1.014 |
| flat | mi300x | 0.2102 | 1.003 | 1.057 | 1.064 |
| flat | rx6800 | 0.3728 | 1.004 | 1.006 | 1.021 |
| atomics | mi300x | 0.3859 | 0.992 | 1.003 | 1.044 |
| atomics | rx6800 | 0.2594 | 1.007 | 0.978 | 1.043 |
| divergent | mi300x | 0.2039 | 0.990 | 1.047 | 1.220 |
| divergent | rx6800 | 0.2941 | 0.977 | 1.011 | 1.263 |
| wave | mi300x | 0.0823 | 1.010 | 1.016 | 1.252 |
| wave | rx6800 | 0.1321 | 0.989 | 1.006 | 1.313 |
| p_dpp | mi300x | 0.0825 | 1.000 | 0.876 | 0.934 |
| p_dpp | rx6800 | 0.1082 | 1.126 | 0.989 | 0.995 |
| p_dpp_arch | mi300x | 0.0516 | 1.004 | 0.958 | 1.029 |
| p_dpp_arch | rx6800 | 0.0470 | 0.988 | 0.998 | 1.046 |
| p_dpp_scan | mi300x | 0.0542 | 1.012 | 1.002 | 1.083 |
| p_dpp_scan | rx6800 | 0.0644 | 1.018 | 1.006 | 1.105 |
| p_perm | mi300x | 0.0469 | 0.998 | 1.009 | 1.077 |
| p_perm | rx6800 | 0.0619 | 1.003 | 1.011 | 1.119 |
| p_lane | mi300x | 0.0214 | 1.027 | 1.008 | 1.244 |
| p_lane | rx6800 | 0.0436 | 0.991 | 1.041 | 1.421 |
| p_pk16 | mi300x | 0.1462 | 1.007 | 1.002 | 1.167 |
| p_pk16 | rx6800 | 0.6690 | 1.006 | 1.012 | 1.048 |
| p_pkf16 | mi300x | 0.1970 | 0.999 | 0.991 | 0.970 |
| p_pkf16 | rx6800 | 0.2144 | 0.997 | 0.998 | 0.986 |
| p_sdwa | mi300x | 0.1019 | 1.010 | 1.832 | 2.518 |
| p_sdwa | rx6800 | 0.1394 | 1.013 | 1.513 | 2.210 |
| p_vopd | mi300x | 0.0961 | 0.992 | 1.007 | 1.131 |
| p_vopd | rx6800 | 0.1238 | 1.001 | 0.999 | 1.245 |
| p_cmp | mi300x | 0.2104 | 1.020 | 1.108 | 1.320 |
| p_cmp | rx6800 | 0.3174 | 0.992 | 1.045 | 1.348 |
| p_i64 | mi300x | 0.1469 | 1.017 | 1.076 | 1.428 |
| p_i64 | rx6800 | 0.1940 | 1.004 | 1.045 | 1.356 |
| p_f64x | mi300x | 0.4779 | 0.992 | 2.366 | 2.650 |
| p_f64x | rx6800 | 0.5724 | 0.982 | 2.037 | 2.346 |
| p_smem_wide | mi300x | 0.4840 | 1.030 | 1.028 | 1.216 |
| p_smem_wide | rx6800 | 0.9539 | 1.041 | 1.043 | 1.219 |
| p_sbuf | mi300x | 0.1014 | 0.994 | 0.996 | 1.070 |
| p_sbuf | rx6800 | 0.1859 | 0.997 | 1.000 | 1.084 |
| mfma | mi300x | 0.1989 | 1.001 | 1.001 | 1.001 |
| mfma_i8 | mi300x | 0.0824 | 1.007 | 1.007 | 0.992 |
| mfma_fp8 | mi300x | 0.2553 | 0.999 | 1.000 | 0.997 |
| mfma_bf16 | mi300x | 0.1893 | 0.999 | 1.004 | 1.003 |
| mfma_f64 | mi300x | 0.1618 | 1.011 | 1.019 | 1.009 |
| mfma_f64_4b | mi300x | 0.1077 | 0.976 | 0.998 | 0.984 |
| smfmac | mi300x | 0.2747 | 0.994 | 1.000 | 0.994 |
| p_permlane | rx6800 | 0.0391 | 1.007 | 1.002 | 1.165 |

control: geometric mean 1.006 over 62 rows; below 0.97: none
final: geometric mean 1.058 over 62 rows; below 0.97: p_dpp/mi300x 0.876, p_dpp_arch/mi300x 0.958
final on: geometric mean 1.179 over 62 rows; below 0.97: p_dpp/mi300x 0.934

## The DPP rows, 15 runs, twice

p_dpp on the MI300X is 0.876 above because the base ran it in 0.0825 s in that pass and in 0.0919 s in the first table: the row has a
fast and a slow way to run in one and the same binary (45 single runs of the base: 0.082 to 0.094 s; of this round's library: 0.090 to
0.097 s), and the control is 10 % faster than the base on it in three tables of four. `w6` is `w5` with `h_int_dpp`, the handler that is
80 % of these rows and that this round did not touch, aligned to 64 bytes: no better, not kept.
`ROWS="p_dpp p_dpp_arch p_dpp_scan" ./qall.sh al1 15 base noop w5 w6`, and again:

| row | target | base (s) | control | final | aligned |
|---|---|---:|---:|---:|---:|
| p_dpp | mi300x | 0.0908 | 1.100 | 0.982 | 0.983 |
| p_dpp | rx6800 | 0.1071 | 0.998 | 0.978 | 1.024 |
| p_dpp_arch | mi300x | 0.0510 | 0.996 | 0.955 | 0.972 |
| p_dpp_arch | rx6800 | 0.0469 | 0.995 | 1.004 | 1.001 |
| p_dpp_scan | mi300x | 0.0540 | 1.011 | 1.009 | 1.019 |
| p_dpp_scan | rx6800 | 0.0637 | 1.008 | 0.995 | 0.997 |

| row | target | base (s) | control | final | aligned |
|---|---|---:|---:|---:|---:|
| p_dpp | mi300x | 0.0915 | 1.090 | 0.981 | 0.981 |
| p_dpp | rx6800 | 0.1081 | 1.022 | 0.979 | 0.984 |
| p_dpp_arch | mi300x | 0.0507 | 1.005 | 0.951 | 0.956 |
| p_dpp_arch | rx6800 | 0.0473 | 1.011 | 1.008 | 1.006 |
| p_dpp_scan | mi300x | 0.0538 | 1.007 | 0.995 | 1.004 |
| p_dpp_scan | rx6800 | 0.0640 | 1.005 | 1.008 | 0.999 |
