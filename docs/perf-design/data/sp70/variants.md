# Round sp70: variants and bisects on the AMD bench

`tools/bench-amd/bench.sh --runs 9`, ratio above 1 means the second build is faster. Instruction counts identical in every row.

## commit 2 against commit 2 plus a function nothing calls in gcn_tier.cpp (the tier unit: layout noise)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| p_dpp | mi300x | 1 | 0.994 | 0.190 | same |
| p_dpp | rx6800 | 1 | 0.995 | 0.605 | same |
| p_dpp_arch | mi300x | 1 | 1.001 | 0.913 | same |
| p_dpp_arch | rx6800 | 1 | 1.010 | 0.666 | same |
| p_dpp_scan | mi300x | 1 | 1.001 | 0.931 | same |
| p_dpp_scan | rx6800 | 1 | 0.996 | 0.297 | same |
| atomics | mi300x | 1 | 1.000 | 0.546 | same |
| atomics | rx6800 | 1 | 1.002 | 0.730 | same |
| p_sbuf | mi300x | 1 | 0.999 | 1.000 | same |
| p_sbuf | rx6800 | 1 | 0.998 | 0.796 | same |
| p_f64x | mi300x | 1 | 0.996 | 0.436 | same |
| p_f64x | rx6800 | 1 | 1.000 | 1.000 | same |
| smfmac | mi300x | 1 | 0.999 | 0.845 | same |
| valu_f64 | mi300x | 1 | 1.001 | 0.730 | same |
| valu_f64 | rx6800 | 1 | 1.004 | 0.063 | same |
| p_i64 | mi300x | 1 | 0.994 | 0.436 | same |
| p_i64 | rx6800 | 1 | 1.003 | 0.863 | same |

## commit 2 against commit 3 (fmin/fmax)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| p_dpp | mi300x | 1 | 0.989 | 0.077 | same |
| p_dpp | rx6800 | 1 | 0.992 | 0.006 | same |
| p_dpp_arch | mi300x | 1 | 0.995 | 1.000 | same |
| p_dpp_arch | rx6800 | 1 | 1.003 | 0.489 | same |
| p_dpp_scan | mi300x | 1 | 1.015 | 0.605 | same |
| p_dpp_scan | rx6800 | 1 | 1.000 | 0.666 | same |
| atomics | mi300x | 1 | 1.016 | 0.297 | same |
| atomics | rx6800 | 1 | 1.007 | 0.094 | same |
| p_sbuf | mi300x | 1 | 1.015 | 0.014 | same |
| p_sbuf | rx6800 | 1 | 0.997 | 0.666 | same |
| p_f64x | mi300x | 1 | 1.081 | 0.000 | faster |
| p_f64x | rx6800 | 1 | 1.047 | 0.000 | faster |
| smfmac | mi300x | 1 | 1.005 | 0.161 | same |
| valu_f64 | mi300x | 1 | 1.003 | 0.161 | same |
| valu_f64 | rx6800 | 1 | 1.005 | 0.113 | same |
| p_i64 | mi300x | 1 | 1.003 | 1.000 | same |
| p_i64 | rx6800 | 1 | 0.994 | 0.666 | same |

## commit 2 against commit 2 plus the SDWA comparison handler alone (not kept)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| p_dpp | mi300x | 1 | 0.950 | 0.000 | slower |
| p_dpp | rx6800 | 1 | 0.966 | 0.000 | slower |
| p_dpp_arch | mi300x | 1 | 0.944 | 0.006 | slower |
| p_dpp_arch | rx6800 | 1 | 1.010 | 0.113 | same |
| p_dpp_scan | mi300x | 1 | 0.970 | 0.010 | slower |
| p_dpp_scan | rx6800 | 1 | 0.978 | 0.002 | slower |
| atomics | mi300x | 1 | 0.980 | 0.040 | same |
| atomics | rx6800 | 1 | 0.988 | 0.050 | same |
| p_sbuf | mi300x | 1 | 1.005 | 0.605 | same |
| p_sbuf | rx6800 | 1 | 1.006 | 0.050 | same |
| p_f64x | mi300x | 1 | 0.989 | 0.003 | same |
| p_f64x | rx6800 | 1 | 0.999 | 0.387 | same |
| smfmac | mi300x | 1 | 1.003 | 0.297 | same |
| valu_f64 | mi300x | 1 | 1.002 | 0.931 | same |
| valu_f64 | rx6800 | 1 | 1.012 | 0.001 | same |
| p_i64 | mi300x | 1 | 0.996 | 0.113 | same |
| p_i64 | rx6800 | 1 | 1.002 | 0.531 | same |

## base against base plus a function nothing calls in gcn_exec.cpp (the interpreter unit: layout noise)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| launch_wide | mi300x | 1 | 0.990 | 0.161 | same |
| launch_wide | rx6800 | 1 | 0.979 | 0.006 | same |
| launch_mid | mi300x | 1 | 0.992 | 0.161 | same |
| launch_mid | rx6800 | 1 | 0.984 | 0.031 | same |
| p_sbuf | mi300x | 1 | 0.973 | 0.002 | same |
| p_sbuf | rx6800 | 1 | 0.965 | 0.000 | slower |
| launch | mi300x | 1 | 1.005 | 1.000 | same |
| launch | rx6800 | 1 | 0.997 | 0.489 | same |
| atomics | mi300x | 1 | 1.005 | 0.666 | same |
| atomics | rx6800 | 1 | 0.999 | 0.730 | same |
| buf | mi300x | 1 | 1.000 | 0.931 | same |
| buf | rx6800 | 1 | 1.002 | 0.077 | same |
| p_dpp | mi300x | 1 | 1.003 | 0.931 | same |
| p_dpp | rx6800 | 1 | 1.017 | 0.050 | same |

## base against base with Pool::kMax 288 and nothing else

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| launch_wide | mi300x | 1 | 1.002 | 0.546 | same |
| launch_wide | rx6800 | 1 | 0.997 | 0.387 | same |
| launch_mid | mi300x | 1 | 0.999 | 0.863 | same |
| launch_mid | rx6800 | 1 | 0.992 | 0.190 | same |
| p_sbuf | mi300x | 1 | 0.996 | 0.730 | same |
| p_sbuf | rx6800 | 1 | 1.009 | 0.796 | same |
| launch | mi300x | 1 | 0.998 | 0.605 | same |
| launch | rx6800 | 1 | 0.993 | 0.436 | same |
| atomics | mi300x | 1 | 1.011 | 0.546 | same |
| atomics | rx6800 | 1 | 0.998 | 0.258 | same |
| buf | mi300x | 1 | 0.989 | 0.002 | same |
| buf | rx6800 | 1 | 0.995 | 0.031 | same |
| p_dpp | mi300x | 1 | 1.001 | 0.546 | same |
| p_dpp | rx6800 | 1 | 0.995 | 0.190 | same |

## base against commit 1 (one thread)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| atomics | mi300x | 1 | 0.998 | 0.436 | same |
| atomics | rx6800 | 1 | 0.966 | 0.002 | same |
| divergent | mi300x | 1 | 1.000 | 1.000 | same |
| divergent | rx6800 | 1 | 0.998 | 0.222 | same |
| mfma_bf16 | mi300x | 1 | 0.981 | 0.258 | same |
| launch_wide | mi300x | 1 | 0.968 | 0.000 | slower |
| launch_wide | rx6800 | 1 | 0.988 | 0.050 | same |
| p_lane | mi300x | 1 | 1.004 | 0.931 | same |
| p_lane | rx6800 | 1 | 1.007 | 0.161 | same |

## commit 1 against commit 2 (one thread)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| atomics | mi300x | 1 | 0.983 | 0.136 | same |
| atomics | rx6800 | 1 | 1.041 | 0.014 | faster |
| divergent | mi300x | 1 | 1.005 | 0.730 | same |
| divergent | rx6800 | 1 | 0.999 | 0.546 | same |
| mfma_bf16 | mi300x | 1 | 1.009 | 0.546 | same |
| launch_wide | mi300x | 1 | 1.000 | 0.489 | same |
| launch_wide | rx6800 | 1 | 1.019 | 0.077 | same |
| p_lane | mi300x | 1 | 0.978 | 1.000 | same |
| p_lane | rx6800 | 1 | 1.016 | 0.040 | same |

## commit 2 against commit 3 (one thread)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| atomics | mi300x | 1 | 1.018 | 0.161 | same |
| atomics | rx6800 | 1 | 1.000 | 0.931 | same |
| divergent | mi300x | 1 | 0.991 | 0.040 | same |
| divergent | rx6800 | 1 | 0.996 | 0.113 | same |
| mfma_bf16 | mi300x | 1 | 1.004 | 0.605 | same |
| launch_wide | mi300x | 1 | 0.990 | 0.340 | same |
| launch_wide | rx6800 | 1 | 0.977 | 0.024 | slower |
| p_lane | mi300x | 1 | 1.017 | 0.730 | same |
| p_lane | rx6800 | 1 | 0.995 | 0.730 | same |

## commit 3 against commit 4 (one thread)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| atomics | mi300x | 1 | 0.991 | 0.258 | same |
| atomics | rx6800 | 1 | 0.962 | 0.050 | same |
| divergent | mi300x | 1 | 0.980 | 0.004 | same |
| divergent | rx6800 | 1 | 0.984 | 0.050 | same |
| mfma_bf16 | mi300x | 1 | 0.955 | 0.222 | same |
| launch_wide | mi300x | 1 | 1.027 | 0.000 | faster |
| launch_wide | rx6800 | 1 | 0.995 | 0.077 | same |
| p_lane | mi300x | 1 | 1.041 | 0.340 | same |
| p_lane | rx6800 | 1 | 0.964 | 0.003 | slower |

## base against base plus the function nothing calls, again (one thread)

| workload | target | threads | median ratio | p | verdict |
|---|---|---|---|---|---|
| atomics | mi300x | 1 | 0.986 | 0.001 | same |
| atomics | rx6800 | 1 | 0.998 | 0.387 | same |
| divergent | mi300x | 1 | 1.002 | 0.796 | same |
| divergent | rx6800 | 1 | 0.990 | 0.136 | same |
| mfma_bf16 | mi300x | 1 | 0.971 | 0.931 | same |
| launch_wide | mi300x | 1 | 0.985 | 0.077 | same |
| launch_wide | rx6800 | 1 | 0.986 | 0.024 | same |
| p_lane | mi300x | 1 | 0.996 | 0.931 | same |
| p_lane | rx6800 | 1 | 0.994 | 1.000 | same |
