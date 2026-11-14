| program | kernel | classes | sim launches | sim s a launch | card launches | card kernel s a launch (mean) | (largest) | largest / sim | runs: verify / default |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| aop-cuda | _Z18prepare_svd_kernelILi256E9PayoffPutEviiT0_PKdPiPd | atomic-int atomic-swap | 1 | 0.315096 | 1 | 0.000470 | 0.000470 | 0.0015 | rc 0 / rc 0 |
| aop-cuda | _Z24compute_final_sum_kernelILi128EEviidPd | atomic-swap | 1 | 0.000224 | 1 | 0.000062 | 0.000062 | 0.2768 | rc 0 / rc 0 |
| aop-cuda | _Z25compute_final_beta_kernelILi128EEvPKiPd | atomic-swap | 99 | 0.000291 | 99 | 0.000066 | 0.000086 | 0.2955 | rc 0 / rc 0 |
| aop-cuda | _Z27compute_partial_beta_kernelILi128E9PayoffPutEviT0_PK | atomic-swap | 99 | 0.012409 | 99 | 0.000072 | 0.000091 | 0.0073 | rc 0 / rc 0 |
| aop-cuda | _Z27compute_partial_sums_kernelILi128EEviPKdPd | atomic-swap | 1 | 0.006799 | 1 | 0.000066 | 0.000066 | 0.0097 | rc 0 / rc 0 |
| atomicPerf-cuda | _Z26WarpRangeAtomicOnSharedMemIdEvPT_i | atomic-swap | 1 | 0.027339 | 1 | 0.000056 | 0.000056 | 0.0020 | rc 0 / rc 0 |
| atomicPerf-cuda | _Z26WarpRangeAtomicOnSharedMemIfEvPT_i | atomic-swap | 1 | 0.024035 | 1 | 0.000034 | 0.000034 | 0.0014 | rc 0 / rc 0 |
| atomicPerf-cuda | _Z27BlockRangeAtomicOnSharedMemIdEvPT_i | atomic-swap | 1 | 0.025958 | 1 | 0.000034 | 0.000034 | 0.0013 | rc 0 / rc 0 |
| atomicPerf-cuda | _Z27BlockRangeAtomicOnSharedMemIfEvPT_i | atomic-swap | 1 | 0.022714 | 1 | 0.000031 | 0.000031 | 0.0014 | rc 0 / rc 0 |
| atomicPerf-cuda | _Z28SingleRangeAtomicOnSharedMemIdEvPT_ii | atomic-swap | 1 | 0.401331 | 1 | 0.000781 | 0.000781 | 0.0019 | rc 0 / rc 0 |
| atomicPerf-cuda | _Z28SingleRangeAtomicOnSharedMemIfEvPT_ii | atomic-swap | 1 | 0.337072 | 1 | 0.000200 | 0.000200 | 0.0006 | rc 0 / rc 0 |
| bh-cuda | _Z18TreeBuildingKerneliiPViPK6float4PKfPi | atomic-int atomic-swap | 2 | 0.003521 | 2 | 0.000056 | 0.000058 | 0.0165 | rc 0 / rc 0 |
