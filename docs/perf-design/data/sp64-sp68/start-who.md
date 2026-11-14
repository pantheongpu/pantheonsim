# Which rule starts the driver: the corpus before the start-share rule (server3, stand-in driver)

`startwho.py` (this directory) on server3 against the integration build before the start-share rule (5d7535c7 plus the log line): every program of
PolyBench, the Rodinia runs of `tools/bench/real/real.py` and the reduced HeCBench list up to atomicPerf, both modes, `VGPU_THREADS=3`, profile
`nvidia/a10g`, the stand-in driver (`STUB_INIT_MS=150`), `--stop` (a run ends at the line that starts the driver). "sim" is the estimate of the
launch that started it, "spent" the account (simulated seconds of launches the card could have run). server3's simulator ran these programs about
4 times as fast as the A10G rig's (HeCBench asyncAllocation: 1.5 s here, 6.6 s there).

| program | mode | rule | kernel | sim ms | spent s | wall s (no start) |
|---|---|---|---|---:|---:|---:|
| polybench/2DCONV | exact | never started |  |  |  | 0.46 |
| polybench/2DCONV | turbo | never started |  |  |  | 0.47 |
| polybench/2MM | exact | sampled | _Z11mm2_kernel1iiiiffPfS_S_ | 9113.838 | 0.000 |  |
| polybench/2MM | turbo | sampled | _Z11mm2_kernel1iiiiffPfS_S_ | 9717.875 | 0.000 |  |
| polybench/3DCONV | exact | never started |  |  |  | 0.41 |
| polybench/3DCONV | turbo | never started |  |  |  | 0.41 |
| polybench/3MM | exact | first-launch rule | _Z11mm3_kernel3iiiiiPfS_S_ | 19.005 | 0.320 |  |
| polybench/3MM | turbo | first-launch rule | _Z11mm3_kernel3iiiiiPfS_S_ | 19.005 | 0.313 |  |
| polybench/ADI | exact | never started |  |  |  | 0.16 |
| polybench/ADI | turbo | never started |  |  |  | 0.16 |
| polybench/ATAX | exact | never started |  |  |  | 0.62 |
| polybench/ATAX | turbo | never started |  |  |  | 0.62 |
| polybench/BICG | exact | never started |  |  |  | 0.18 |
| polybench/BICG | turbo | never started |  |  |  | 0.18 |
| polybench/CORR | exact | sampled | _Z11corr_kerneliiPfS_ | 15842.662 | 0.000 |  |
| polybench/CORR | turbo | sampled | _Z11corr_kerneliiPfS_ | 16653.567 | 0.076 |  |
| polybench/COVAR | exact | sampled | _Z12covar_kerneliiPfS_ | 13456.367 | 0.005 |  |
| polybench/COVAR | turbo | sampled | _Z12covar_kerneliiPfS_ | 14519.807 | 0.016 |  |
| polybench/FDTD-2D | exact | size rule | _Z17fdtd_step1_kerneliiPfS_S_S_i | 28.234 | 0.269 |  |
| polybench/FDTD-2D | turbo | size rule | _Z17fdtd_step1_kerneliiPfS_S_S_i | 27.704 | 0.268 |  |
| polybench/GEMM | exact | never started |  |  |  | 0.48 |
| polybench/GEMM | turbo | never started |  |  |  | 0.48 |
| polybench/GEMVER | exact | never started |  |  |  | 0.39 |
| polybench/GEMVER | turbo | never started |  |  |  | 0.44 |
| polybench/GESUMMV | exact | never started |  |  |  | 0.26 |
| polybench/GESUMMV | turbo | never started |  |  |  | 0.26 |
| polybench/GRAMSCHM | exact | size rule | _Z19gramschmidt_kernel3iiPfS_S_i | 18.564 | 0.261 |  |
| polybench/GRAMSCHM | turbo | size rule | _Z19gramschmidt_kernel3iiPfS_S_i | 18.592 | 0.257 |  |
| polybench/JACOBI1D | exact | never started |  |  |  | 0.62 |
| polybench/JACOBI1D | turbo | never started |  |  |  | 0.63 |
| polybench/JACOBI2D | exact | never started |  |  |  | 0.35 |
| polybench/JACOBI2D | turbo | never started |  |  |  | 0.28 |
| polybench/LU | exact | size rule | _Z10lu_kernel2iPfi | 23.132 | 0.254 |  |
| polybench/LU | turbo | size rule | _Z10lu_kernel2iPfi | 22.981 | 0.251 |  |
| polybench/MVT | exact | never started |  |  |  | 0.75 |
| polybench/MVT | turbo | never started |  |  |  | 0.77 |
| polybench/SYR2K | exact | sampled | _Z12syr2k_kerneliiffPfS_S_ | 3170.851 | 0.000 |  |
| polybench/SYR2K | turbo | sampled | _Z12syr2k_kerneliiffPfS_S_ | 4321.695 | 0.000 |  |
| polybench/SYRK | exact | sampled | _Z11syrk_kerneliiffPfS_ | 1598.962 | 0.000 |  |
| polybench/SYRK | turbo | sampled | _Z11syrk_kerneliiffPfS_ | 1584.906 | 0.000 |  |
| rodinia/gaussian | exact | size rule | _Z4Fan2PfS_S_iii | 5.435 | 0.253 |  |
| rodinia/gaussian | turbo | size rule | _Z4Fan2PfS_S_iii | 5.320 | 0.255 |  |
| rodinia/backprop | exact | never started |  |  |  | 0.30 |
| rodinia/backprop | turbo | never started |  |  |  | 0.30 |
| rodinia/lud | exact | size rule | _Z12lud_internalPfii | 10.804 | 0.259 |  |
| rodinia/lud | turbo | size rule | _Z13lud_perimeterPfii | 1.299 | 0.254 |  |
| rodinia/needle | exact | never started |  |  |  | 0.17 |
| rodinia/needle | turbo | never started |  |  |  | 0.17 |
| rodinia/pathfinder | exact | never started |  |  |  | 0.43 |
| rodinia/pathfinder | turbo | never started |  |  |  | 0.42 |
| rodinia/srad_v2 | exact | never started |  |  |  | 0.03 |
| rodinia/srad_v2 | turbo | never started |  |  |  | 0.03 |
| rodinia/particlefilter_float | exact | size rule | _Z17find_index_kernelPdS_S_S_S_S_S_i | 8.692 | 0.322 |  |
| rodinia/particlefilter_float | turbo | size rule | _Z17likelihood_kernelPdS_S_S_S_PiS0_S_Ph | 60.148 | 0.292 |  |
| rodinia/sc_gpu | exact | never started |  |  |  | 0.42 |
| rodinia/sc_gpu | turbo | never started |  |  |  | 0.42 |
| rodinia/hotspot | exact | never started |  |  |  | 0.45 |
| rodinia/hotspot | turbo | size rule | _Z14calculate_tempiPfS_S_iiiiffffff | 0.393 | 0.250 |  |
| rodinia/bfs | exact | never started |  |  |  | 0.01 |
| rodinia/bfs | turbo | never started |  |  |  | 0.01 |
| rodinia/nn | exact | never started |  |  |  | 0.01 |
| rodinia/nn | turbo | never started |  |  |  | 0.01 |
| rodinia/lavaMD | exact | never started |  |  |  | 0.62 |
| rodinia/lavaMD | turbo | never started |  |  |  | 0.64 |
| hecbench/adam-cuda | exact | never started |  |  |  | 0.02 |
| hecbench/adam-cuda | turbo | never started |  |  |  | 0.02 |
| hecbench/aidw-cuda | exact | never started |  |  |  | cut at 130 s |
| hecbench/aidw-cuda | turbo | never started |  |  |  | cut at 130 s |
| hecbench/aligned-types-cuda | exact | size rule | _Z10testKernelIhEvPT_PKS0_i | 227.758 | 0.458 |  |
| hecbench/aligned-types-cuda | turbo | size rule | _Z10testKernelIhEvPT_PKS0_i | 230.586 | 0.474 |  |
| hecbench/aobench-cuda | exact | never started |  |  |  | 0.25 |
| hecbench/aobench-cuda | turbo | never started |  |  |  | 0.25 |
| hecbench/aop-cuda | exact | never started |  |  |  | 0.83 |
| hecbench/aop-cuda | turbo | never started |  |  |  | 0.86 |
| hecbench/asmooth-cuda | exact | never started |  |  |  | 0.12 |
| hecbench/asmooth-cuda | turbo | never started |  |  |  | 0.11 |
| hecbench/asyncAllocation-cuda | exact | never started |  |  |  | 1.53 |
| hecbench/asyncAllocation-cuda | turbo | size rule | _Z7processPKfPfii | 1.856 | 0.264 |  |
| hecbench/atan2-cuda | exact | never started |  |  |  | 0.02 |
| hecbench/atan2-cuda | turbo | never started |  |  |  | 0.02 |
| hecbench/atomicAggregate-cuda | exact | never started |  |  |  | 1.62 |
| hecbench/atomicAggregate-cuda | turbo | size rule | _Z1kPii | 310.747 | 0.311 |  |
| hecbench/atomicCost-cuda | exact | never started |  |  |  | 65.42 |
| hecbench/atomicCost-cuda | turbo | never started |  |  |  | 65.41 |
| hecbench/atomicIntrinsics-cuda | exact | never started |  |  |  | 0.22 |
| hecbench/atomicIntrinsics-cuda | turbo | never started |  |  |  | 0.21 |
| hecbench/atomicPerf-cuda | exact | never started |  |  |  | 0.51 |
| hecbench/atomicPerf-cuda | turbo | first-launch rule | _Z28SingleRangeAtomicOnGlobalMemIiEvPT_i | 15.483 | 0.263 |  |

## The same, the start-share rule (a2db3524 and later)

Default start cost unless noted. `VGPU_HOST_GPU_START_COST=0.08` stands in for the rig's slower simulator on the three short Samples (their
account never fills here at 0.25 s); with it, under the old rule, threadFenceReduction (turbo, 1.07 ms) and dct8x8 (both modes, 1.0 to 1.2 ms)
started the driver by the size rule, and MC_EstimatePiInlineP (turbo, default cost) by the first-launch rule (3.0 ms static, 1.5 s spent).

| program | mode | start-share rule |
|---|---|---|
| samples/LargeKernelParameter | both (cost 0.08) | never started |
| samples/threadFenceReduction | both (cost 0.08) | never started |
| samples/dct8x8 | exact (cost 0.08) | never started |
| samples/dct8x8 | turbo (cost 0.08) | first-launch rule, CUDAkernelQuantizationShort, 8.5 ms static (passes the static share at this cost, 4 ms; at the default 12.5 ms it does not) |
| samples/dct8x8 | turbo | never started |
| samples/MC_EstimatePiInlineP | both | never started |
| polybench/FDTD-2D | both | size rule, 28 to 38 ms |
| polybench/GRAMSCHM | both | size rule, gramschmidt_kernel3, 17.5 to 18 ms |
| polybench/LU | both | size rule, lu_kernel2, 23.5 ms |
| polybench/3MM | both | first-launch rule, mm3_kernel3, 19 ms static |
| rodinia/gaussian | both | size rule, Fan2, 5.4 to 5.7 ms |
| rodinia/lud | both | size rule, lud_internal, 11.6 to 12.9 ms (turbo started on a 1.3 ms launch before) |
| rodinia/particlefilter_float | both | size rule, 8 ms (exact), 61 ms (turbo) |
| rodinia/hotspot | turbo | never started (0.39 ms before; 1.00x on the rig in Round sp66) |
| hecbench/aligned-types-cuda | both | size rule, 229 to 236 ms |
| hecbench/asyncAllocation-cuda | turbo | never started here (1.9 ms; about 8 ms on the rig, above the 5 ms share) |
| hecbench/atomicAggregate-cuda | turbo | size rule, 310 ms |
| hecbench/atomicPerf-cuda | turbo | first-launch rule, 15.5 ms static |
