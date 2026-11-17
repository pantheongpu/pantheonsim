programs: 124 (11 cut at the time limit: hecbench/aidw-cuda, hecbench/aligned-types-cuda, hecbench/atomicCost-cuda, hecbench/atomicReduction-cuda, hecbench/babelstream-cuda, hecbench/base64e-cuda, hecbench/binomial-cuda, hecbench/fpdc-cuda, polybench/2MM, polybench/GRAMSCHM, samples/tf32TensorCoreGemm)
kernels with code logged (a kernel the mode admitted and that has a loop): 253 (248 distinct names); weighted count above the plain count: 13
cross-check with the simulator's own finder (top-level loops of kernels with no call): 124 followed loops, the census finds the same loop and turns for 124, differs for 0
kernels whose flow the census could not read (a register jump, BRX or JMP, or an unresolved branch; weighted = plain): 0

Launches the floor kept in the simulator (the first two of each kernel are said): 177, of 176 kernels

| program | kernel | grid x block | plain count (instrs) | weighted count | today's estimate ms | weighted estimate ms | floor ms | rises over the floor | loops (turns; bound) | sample cost ms (fork ms) | sample says ms, would send |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| hecbench/adam-cuda | `_Z4adamIffEvPT_S1_S1_PKT0_fffffim10adamMode_tf` | 40 x 256 | 600 | 600 | 1.9 | 1.9 | 30.0 | no | ?;none; ?;param; ?;param; ?;none (13 calls) | - | - |
| hecbench/aobench-cuda | `_Z13render_kernelPhPK7_Sphere6_Planeiii` | 256 x 256 | 472 | 472 | 9.7 | 9.7 | 30.0 | no | ?;none; ?;none; ?;none; ?;none; ?;none (2 calls) | - | - |
| hecbench/aop-cuda | `_Z18prepare_svd_kernelILi256E9PayoffPutEviiT0_PKdPiPd` | 99 x 256 | 2344 | 10384 | 18.6 | 82.2 | 30.0 | **yes** | ?;param; ?;none; ?;none; ?;none; ?;none; 16;imm; ?;none (33 calls) | 8.9 (0.7) | 111, no |
| hecbench/aop-cuda | `_Z21generate_paths_kernelILi256E9PayoffPutEviiT0_ddddPKdPd` | 128 x 256 | 160 | 160 | 1.6 | 1.6 | 30.0 | no | ?;none; ?;none (1 call) | 7.2 (0.7) | 27, no |
| hecbench/aop-cuda | `_Z22update_cashflow_kernelILi128E9PayoffPutEviT0_dPKdS3_PKiPd` | 224 x 128 | 224 | 224 | 2.0 | 2.0 | 94.5 | no | ?;reg; ?;none; ?;reg; ?;none; ?;none | 1.3 (0.4) | 18, no |
| hecbench/aop-cuda | `_Z27compute_partial_beta_kernelILi128E9PayoffPutEviT0_PKdS3_S3_PKiPd` | 128 x 128 | 344 | 344 | 1.8 | 1.8 | 30.0 | no | ?;reg; ?;none; ?;none; ?;none; ?;none; ?;none (3 calls) | 5.7 (0.7) | 5, no |
| hecbench/aop-cuda | `_Z27compute_partial_sums_kernelILi128EEviPKdPd` | 256 x 128 | 40 | 40 | 0.4 | 0.4 | 94.5 | no | ?;none; ?;none | 5.1 (1.2) | 4, no |
| hecbench/asmooth-cuda | `_Z15smoothingFilteriiiiPKfPiPf` | 1024 x 256 | 304 | 304 | 24.9 | 24.9 | 30.0 | no | ?;none; ?;none; ?;reg; ?;reg; ?;none; ?;none | - | - |
| hecbench/asmooth-cuda | `_Z9outFilteriiPKfPKiPf` | 1024 x 256 | 184 | 184 | 15.1 | 15.1 | 30.0 | no | ?;reg; ?;none; ?;none | - | - |
| hecbench/asyncAllocation-cuda | `_Z7processPKfPfii` | 396 x 256 | 240 | 375 | 7.6 | 11.9 | 30.0 | no | ?;param; 6;imm; 6;imm; ?;none (1 call) | - | - |
| hecbench/atan2-cuda | `_Z9compute_fiPKfS0_Pf` | 391 x 256 | 216 | 216 | 6.8 | 6.8 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/atan2-cuda | `_Z9compute_iiPKfS0_Pi` | 391 x 256 | 208 | 208 | 6.5 | 6.5 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/atan2-cuda | `_Z9compute_siPKfS0_Ps` | 391 x 256 | 184 | 184 | 5.8 | 5.8 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/atomicPerf-cuda | `_Z26WarpRangeAtomicOnGlobalMemIdEvPT_i` | 6048 x 256 | 32 | 32 | 15.5 | 15.5 | 30.0 | no | ?;none; ?;none | - | - |
| hecbench/atomicPerf-cuda | `_Z26WarpRangeAtomicOnSharedMemIdEvPT_i` | 6048 x 256 | 48 | 48 | 23.2 | 23.2 | 30.0 | no | ?;none; ?;none; ?;none | - | - |
| hecbench/atomicPerf-cuda | `_Z27BlockRangeAtomicOnGlobalMemIdEvPT_i` | 6048 x 256 | 32 | 32 | 15.5 | 15.5 | 30.0 | no | ?;none; ?;none | - | - |
| hecbench/atomicPerf-cuda | `_Z27BlockRangeAtomicOnSharedMemIdEvPT_i` | 6048 x 256 | 40 | 40 | 19.4 | 19.4 | 30.0 | no | ?;none; ?;none; ?;none | - | - |
| hecbench/atomicPerf-cuda | `_Z28SingleRangeAtomicOnGlobalMemIdEvPT_ii` | 6048 x 256 | 32 | 32 | 15.5 | 15.5 | 30.0 | no | ?;none; ?;none | - | - |
| hecbench/atomicPerf-cuda | `_Z28SingleRangeAtomicOnSharedMemIdEvPT_ii` | 6048 x 256 | 40 | 40 | 19.4 | 19.4 | 30.0 | no | ?;none; ?;none; ?;none | - | - |
| hecbench/atomicReduction-cuda | `_Z16atomic_reductionPiS_i` | 2048 x 256 | 40 | 40 | 6.6 | 6.6 | 377.5 | no | ?;none; ?;none | - | - |
| hecbench/attention-cuda | `_Z17attention_kernel1PKfS0_PfS1_ii` | 16 x 256 | 192 | 192 | 0.2 | 0.2 | 30.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| hecbench/attention-cuda | `_Z17attention_kernel2PKfS0_Pfi` | 16 x 256 | 152 | 152 | 0.2 | 0.2 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/attention-cuda | `_Z17attention_kernel3PKfS0_Pfii` | 1 x 256 | 200 | 200 | 0.0 | 0.0 | 30.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| hecbench/axpby-cuda | `_Z25multi_tensor_apply_kernelI18TensorListMetadataILi3EE12AxpbyFunctorIfffEJffiEEvlPViT_T0_DpT1_` | 320 x 512 | 336 | 336 | 17.2 | 17.2 | 30.0 | no | ?;none; ?;none; ?;none | - | - |
| hecbench/babelstream-cuda | `_Z10dot_kernelIfEvPKT_S2_PS0_i` | 256 x 256 | 64 | 64 | 1.3 | 1.3 | 724.8 | no | ?;none; ?;none; ?;none | - | - |
| hecbench/bgmv-cuda | `_Z18bgmv_shrink_kernelILi1024EEvPfPK6__halfS3_PKiiiif` | 16 x 1024 | 344 | 344 | 1.8 | 1.8 | 30.0 | no | ?;reg; ?;reg; ?;none | - | - |
| hecbench/bgmv-cuda | `_Z18bgmv_shrink_kernelILi128EEvPfPK6__halfS3_PKiiiif` | 16 x 128 | 312 | 312 | 0.2 | 0.2 | 30.0 | no | ?;reg; ?;reg; ?;none | - | - |
| hecbench/bgmv-cuda | `_Z18bgmv_shrink_kernelILi256EEvPfPK6__halfS3_PKiiiif` | 16 x 256 | 320 | 320 | 0.4 | 0.4 | 30.0 | no | ?;reg; ?;reg; ?;none | - | - |
| hecbench/bgmv-cuda | `_Z18bgmv_shrink_kernelILi32EEvPfPK6__halfS3_PKiiiif` | 16 x 32 | 304 | 304 | 0.0 | 0.0 | 30.0 | no | ?;reg; ?;reg; ?;none | - | - |
| hecbench/bgmv-cuda | `_Z18bgmv_shrink_kernelILi512EEvPfPK6__halfS3_PKiiiif` | 16 x 512 | 328 | 328 | 0.8 | 0.8 | 30.0 | no | ?;reg; ?;reg; ?;none | - | - |
| hecbench/bgmv-cuda | `_Z18bgmv_shrink_kernelILi64EEvPfPK6__halfS3_PKiiiif` | 16 x 64 | 304 | 304 | 0.1 | 0.1 | 30.0 | no | ?;reg; ?;reg; ?;none | - | - |
| hecbench/bh-cuda | `_Z10SortKerneliiPiPKiPViS_S_` | 24 x 256 | 208 | 208 | 0.4 | 0.4 | 30.0 | no | ?;none; ?;none (1 call) | - | - |
| hecbench/bh-cuda | `_Z12ClearKernel1iiPi` | 24 x 256 | 80 | 80 | 0.2 | 0.2 | 30.0 | no | ?;reg; ?;none; ?;none | - | - |
| hecbench/bh-cuda | `_Z12ClearKernel2iPiP6float4S_` | 24 x 256 | 96 | 96 | 0.2 | 0.2 | 30.0 | no | ?;reg; ?;none; ?;none | - | - |
| hecbench/bh-cuda | `_Z17BoundingBoxKerneliiPiS_P6float4P6float3S3_PfS_S_Pj` | 48 x 256 | 464 | 464 | 1.8 | 1.8 | 30.0 | no | ?;reg; ?;none; ?;reg; ?;reg; ?;none | - | - |
| hecbench/bh-cuda | `_Z17IntegrationKerneliffP6float4P6float2S0_` | 48 x 256 | 160 | 160 | 0.6 | 0.6 | 30.0 | no | ?;reg; ?;none; ?;none | - | - |
| hecbench/bh-cuda | `_Z18TreeBuildingKerneliiPViPK6float4PKfPi` | 48 x 256 | 168 | 168 | 0.6 | 0.6 | 30.0 | no | ?;none; ?;none; ?;none; ?;none | - | - |
| hecbench/bh-cuda | `_Z19SummarizationKerneliiPViPKiPV6float4Pi` | 24 x 256 | 832 | 832 | 1.6 | 1.6 | 30.0 | no | ?;none; ?;none; ?;none; ?;none (5 calls) | - | - |
| hecbench/bh-cuda | `_Z22ForceCalculationKerneliifffPKiS0_PK6float4P6float2PS1_PKfS0_` | 48 x 256 | 240 | 240 | 0.9 | 0.9 | 30.0 | no | ?;none (4 calls) | - | - |
| hecbench/bilateral-cuda | `_Z15bilateralFilterILi3EEvPKfPfiifff` | 128 x 256 | 600 | 600 | 6.1 | 6.1 | 30.0 | no | ?;none; ?;none (16 calls) | 7.2 (0.1) | 41, no |
| hecbench/bilateral-cuda | `_Z15bilateralFilterILi6EEvPKfPfiifff` | 128 x 256 | 416 | 5206 | 4.3 | 53.3 | 30.0 | **yes** | 7;imm; 3;imm; ?;none (10 calls) | 9.4 (0.4) | 170, no |
| hecbench/bilateral-cuda | `_Z15bilateralFilterILi9EEvPKfPfiifff` | 128 x 256 | 416 | 2693 | 4.3 | 27.6 | 30.0 | no | 10;imm; ?;none; ?;none (10 calls) | 20.6 (0.8) | 368, no |
| hecbench/blockAccess-cuda | `_Z6kernelILi256ELi4EEvPKfPhm` | 1280 x 256 | 64 | 64 | 6.6 | 6.6 | 30.0 | no | ?;none; ?;none | - | - |
| hecbench/blockAccess-cuda | `_Z9referencePKfPhm` | 1280 x 256 | 48 | 48 | 4.9 | 4.9 | 30.0 | no | ?;none; ?;none | - | - |
| hecbench/perplexity-cuda | `_Z13sigmas_kernelIifEvPKT0_PS0_ffifT_i` | 4 x 256 | 384 | 384 | 0.1 | 0.1 | 30.0 | no | ?;param; ?;reg; 0;imm; ?;reg; ?;none (1 call) | - | - |
| hecbench/radixsort2-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29DeviceRadixSortOnesweepKernelINS1_5radix10policy_hubIfNS0_8NullTypeEyE10Policy1000ELNS0_9SortOrderE0EfS6_yiiNS1_21identity_decomposer_tEEEvPT5_SC_PT3_PKSD_PT1_PKSH_PT2_PKSL_T4_iiT6_` | 9 x 384 | 2632 | 2632 | 2.8 | 2.8 | 30.0 | no | ?;reg; 1;imm; ?;none; ?;none; ?;none (7 calls) | - | - |
| hecbench/radixsort2-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort30DeviceRadixSortHistogramKernelINS1_5radix10policy_hubIfNS0_8NullTypeEyE10Policy1000ELNS0_9SortOrderE0EfyNS1_21identity_decomposer_tEEEvPT2_PKT1_SB_iiT3_` | 960 x 128 | 480 | 480 | 18.4 | 18.4 | 30.0 | no | ?;none; 1;imm; ?;reg; ?;none; ?;none; 1;imm; ?;reg; ?;none (2 calls) | - | - |
| hecbench/radixsort2-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort33DeviceRadixSortExclusiveSumKernelINS1_5radix10policy_hubIfNS0_8NullTypeEyE10Policy1000EyEEvPT0_` | 4 x 256 | 192 | 192 | 0.1 | 0.1 | 30.0 | no | ?;none (10 calls) | - | - |
| hecbench/s3d-cuda | `_Z11qssa_kernelIdEvPT_S1_S1_` | 32 x 128 | 1384 | 1384 | 1.8 | 1.8 | 30.0 | no | ?;none (10 calls) | - | - |
| hecbench/s3d-cuda | `_Z11qssa_kernelIfEvPT_S1_S1_` | 108 x 128 | 1376 | 1376 | 5.9 | 5.9 | 30.0 | no | ?;none (10 calls) | - | - |
| hecbench/s3d-cuda | `_Z11ratt_kernelIdEvPKT_PS0_S0_` | 32 x 128 | 4864 | 4864 | 6.2 | 6.2 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/s3d-cuda | `_Z11ratt_kernelIfEvPKT_PS0_S0_` | 108 x 128 | 1952 | 1952 | 8.4 | 8.4 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/s3d-cuda | `_Z11ratx_kernelIdEvPKT_S2_PS0_S3_S2_S0_` | 64 x 64 | 4472 | 4472 | 5.7 | 5.7 | 30.0 | no | ?;none (47 calls) | - | - |
| hecbench/s3d-cuda | `_Z11ratx_kernelIfEvPKT_S2_PS0_S3_S2_S0_` | 216 x 64 | 2240 | 2240 | 9.7 | 9.7 | 30.0 | no | ?;none (47 calls) | - | - |
| hecbench/s3d-cuda | `_Z12qssab_kernelIdEvPT_S1_S1_` | 64 x 64 | 976 | 976 | 1.2 | 1.2 | 30.0 | no | ?;none (19 calls) | - | - |
| hecbench/s3d-cuda | `_Z12qssab_kernelIfEvPT_S1_S1_` | 216 x 64 | 952 | 952 | 4.1 | 4.1 | 30.0 | no | ?;none (19 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt2_kernelIdEvPKT_S2_PS0_S2_S0_` | 32 x 128 | 704 | 704 | 0.9 | 0.9 | 30.0 | no | ?;none (18 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt2_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 x 128 | 584 | 584 | 2.5 | 2.5 | 30.0 | no | ?;none (18 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt3_kernelIdEvPKT_S2_PS0_S2_S0_` | 32 x 128 | 848 | 848 | 1.1 | 1.1 | 30.0 | no | ?;none (21 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt3_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 x 128 | 672 | 672 | 2.9 | 2.9 | 30.0 | no | ?;none (21 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt4_kernelIdEvPKT_S2_PS0_S2_S0_` | 32 x 128 | 816 | 816 | 1.0 | 1.0 | 30.0 | no | ?;none (21 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt4_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 x 128 | 672 | 672 | 2.9 | 2.9 | 30.0 | no | ?;none (21 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt5_kernelIdEvPKT_S2_PS0_S2_S0_` | 32 x 128 | 896 | 896 | 1.1 | 1.1 | 30.0 | no | ?;none (23 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt5_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 x 128 | 720 | 720 | 3.1 | 3.1 | 30.0 | no | ?;none (23 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt6_kernelIdEvPKT_S2_PS0_S2_S0_` | 32 x 128 | 864 | 864 | 1.1 | 1.1 | 30.0 | no | ?;none (21 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt6_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 x 128 | 704 | 704 | 3.0 | 3.0 | 30.0 | no | ?;none (21 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt7_kernelIdEvPKT_S2_PS0_S2_S0_` | 32 x 128 | 920 | 920 | 1.2 | 1.2 | 30.0 | no | ?;none (25 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt7_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 x 128 | 760 | 760 | 3.3 | 3.3 | 30.0 | no | ?;none (25 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt8_kernelIdEvPKT_S2_PS0_S2_S0_` | 32 x 128 | 928 | 928 | 1.2 | 1.2 | 30.0 | no | ?;none (25 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt8_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 x 128 | 752 | 752 | 3.2 | 3.2 | 30.0 | no | ?;none (25 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt9_kernelIdEvPKT_S2_PS0_S2_S0_` | 32 x 128 | 1120 | 1120 | 1.4 | 1.4 | 30.0 | no | ?;none (31 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratt9_kernelIfEvPKT_S2_PS0_S2_S0_` | 108 x 128 | 904 | 904 | 3.9 | 3.9 | 30.0 | no | ?;none (31 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratxb_kernelIdEvPKT_S2_PS0_S3_S2_S0_` | 64 x 64 | 3808 | 3808 | 4.9 | 4.9 | 30.0 | no | ?;none (37 calls) | - | - |
| hecbench/s3d-cuda | `_Z12ratxb_kernelIfEvPKT_S2_PS0_S3_S2_S0_` | 216 x 64 | 1976 | 1976 | 8.5 | 8.5 | 30.0 | no | ?;none (37 calls) | - | - |
| hecbench/s3d-cuda | `_Z12rdsmh_kernelIdEvPKT_PS0_S0_` | 32 x 128 | 3008 | 3008 | 3.9 | 3.9 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/s3d-cuda | `_Z12rdsmh_kernelIfEvPKT_PS0_S0_` | 108 x 128 | 1304 | 1304 | 5.6 | 5.6 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/s3d-cuda | `_Z13ratt10_kernelIdEvPKT_PS0_S0_` | 32 x 128 | 992 | 992 | 1.3 | 1.3 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/s3d-cuda | `_Z13ratt10_kernelIfEvPKT_PS0_S0_` | 108 x 128 | 416 | 416 | 1.8 | 1.8 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/s3d-cuda | `_Z7gr_baseIdEvPKT_S2_S2_PS0_S0_S0_` | 32 x 128 | 456 | 456 | 0.6 | 0.6 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/s3d-cuda | `_Z7gr_baseIfEvPKT_S2_S2_PS0_S0_S0_` | 108 x 128 | 336 | 336 | 1.5 | 1.5 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/silu-cuda | `_Z19silu_forward_kernelPKfPfm` | 4096 x 64 | 160 | 160 | 13.1 | 13.1 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/silu-cuda | `_Z20silu_backward_kernelPKfS0_Pfm` | 4096 x 64 | 176 | 176 | 14.4 | 14.4 | 30.0 | no | ?;none (2 calls) | - | - |
| hecbench/silu-cuda | `_Z20silu_forward_kernel2PKfPfm` | 1024 x 64 | 280 | 280 | 5.7 | 5.7 | 30.0 | no | ?;none; ?;none (5 calls) | - | - |
| hecbench/silu-cuda | `_Z21silu_backward_kernel2PKfS0_Pfm` | 4096 x 64 | 112 | 112 | 9.2 | 9.2 | 30.0 | no | ?;none (1 call) | - | - |
| hecbench/silu-cuda | `_Z21silu_backward_kernel3PKfS0_Pfm` | 1024 x 64 | 256 | 256 | 5.2 | 5.2 | 30.0 | no | ?;none; ?;none (5 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29DeviceRadixSortOnesweepKernelINS1_5radix10policy_hubIihyE10Policy1000ELNS0_9SortOrderE0EihyiiNS1_21identity_decomposer_tEEEvPT5_SB_PT3_PKSC_PT1_PKSG_PT2_PKSK_T4_iiT6_` | 11 x 384 | 2592 | 2592 | 3.4 | 3.4 | 30.0 | no | ?;reg; 1;imm; ?;none; ?;none; ?;none (7 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29DeviceRadixSortOnesweepKernelINS1_5radix10policy_hubIiiyE10Policy1000ELNS0_9SortOrderE0EiiyiiNS1_21identity_decomposer_tEEEvPT5_SB_PT3_PKSC_PT1_PKSG_PT2_PKSK_T4_iiT6_` | 11 x 384 | 2592 | 2592 | 3.4 | 3.4 | 30.0 | no | ?;reg; 1;imm; ?;none; ?;none; ?;none (7 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29DeviceRadixSortOnesweepKernelINS1_5radix10policy_hubIilyE10Policy1000ELNS0_9SortOrderE0EilyiiNS1_21identity_decomposer_tEEEvPT5_SB_PT3_PKSC_PT1_PKSG_PT2_PKSK_T4_iiT6_` | 22 x 384 | 1544 | 1544 | 4.1 | 4.1 | 30.0 | no | ?;reg; 1;imm; ?;none; ?;none; ?;none (7 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort29DeviceRadixSortOnesweepKernelINS1_5radix10policy_hubIisyE10Policy1000ELNS0_9SortOrderE0EisyiiNS1_21identity_decomposer_tEEEvPT5_SB_PT3_PKSC_PT1_PKSG_PT2_PKSK_T4_iiT6_` | 11 x 384 | 2656 | 2656 | 3.5 | 3.5 | 30.0 | no | ?;reg; 1;imm; ?;none; ?;none; ?;none (7 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort30DeviceRadixSortHistogramKernelINS1_5radix10policy_hubIihyE10Policy1000ELNS0_9SortOrderE0EiyNS1_21identity_decomposer_tEEEvPT2_PKT1_SA_iiT3_` | 960 x 128 | 528 | 528 | 20.3 | 20.3 | 30.0 | no | ?;none; 1;imm; ?;reg; ?;none; ?;reg; 1;imm; ?;reg; ?;none (2 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort30DeviceRadixSortHistogramKernelINS1_5radix10policy_hubIiiyE10Policy1000ELNS0_9SortOrderE0EiyNS1_21identity_decomposer_tEEEvPT2_PKT1_SA_iiT3_` | 960 x 128 | 528 | 528 | 20.3 | 20.3 | 30.0 | no | ?;none; 1;imm; ?;reg; ?;none; ?;reg; 1;imm; ?;reg; ?;none (2 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort30DeviceRadixSortHistogramKernelINS1_5radix10policy_hubIilyE10Policy1000ELNS0_9SortOrderE0EiyNS1_21identity_decomposer_tEEEvPT2_PKT1_SA_iiT3_` | 960 x 128 | 528 | 528 | 20.3 | 20.3 | 30.0 | no | ?;none; 1;imm; ?;reg; ?;none; ?;reg; 1;imm; ?;reg; ?;none (2 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort30DeviceRadixSortHistogramKernelINS1_5radix10policy_hubIisyE10Policy1000ELNS0_9SortOrderE0EiyNS1_21identity_decomposer_tEEEvPT2_PKT1_SA_iiT3_` | 960 x 128 | 528 | 528 | 20.3 | 20.3 | 30.0 | no | ?;none; 1;imm; ?;reg; ?;none; ?;reg; 1;imm; ?;reg; ?;none (2 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort33DeviceRadixSortExclusiveSumKernelINS1_5radix10policy_hubIihyE10Policy1000EyEEvPT0_` | 4 x 256 | 192 | 192 | 0.1 | 0.1 | 30.0 | no | ?;none (10 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort33DeviceRadixSortExclusiveSumKernelINS1_5radix10policy_hubIiiyE10Policy1000EyEEvPT0_` | 4 x 256 | 192 | 192 | 0.1 | 0.1 | 30.0 | no | ?;none (10 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort33DeviceRadixSortExclusiveSumKernelINS1_5radix10policy_hubIilyE10Policy1000EyEEvPT0_` | 4 x 256 | 192 | 192 | 0.1 | 0.1 | 30.0 | no | ?;none (10 calls) | - | - |
| hecbench/sortKV-cuda | `_ZN3cub17CUB_300001_SM_8606detail10radix_sort33DeviceRadixSortExclusiveSumKernelINS1_5radix10policy_hubIisyE10Policy1000EyEEvPT0_` | 4 x 256 | 192 | 192 | 0.1 | 0.1 | 30.0 | no | ?;none (10 calls) | - | - |
| polybench/3MM | `_Z11mm3_kernel1iiiiiPfS_S_` | 1024 x 256 | 232 | 232 | 19.0 | 19.0 | 30.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/3MM | `_Z11mm3_kernel2iiiiiPfS_S_` | 1024 x 256 | 232 | 232 | 19.0 | 19.0 | 30.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/ADI | `_Z11adi_kernel1iPfS_S_` | 4 x 256 | 384 | 384 | 0.1 | 0.1 | 30.0 | no | 0;imm; ?;reg; ?;none (10 calls) | - | - |
| polybench/ADI | `_Z11adi_kernel2iPfS_S_` | 4 x 256 | 144 | 144 | 0.0 | 0.0 | 30.0 | no | ?;none (1 call) | - | - |
| polybench/ADI | `_Z11adi_kernel3iPfS_S_` | 4 x 256 | 280 | 280 | 0.1 | 0.1 | 30.0 | no | 0;imm; ?;reg; ?;none (5 calls) | - | - |
| polybench/ADI | `_Z11adi_kernel4iPfS_S_i` | 4 x 256 | 168 | 168 | 0.1 | 0.1 | 30.0 | no | ?;none (2 calls) | - | - |
| polybench/ADI | `_Z11adi_kernel5iPfS_S_` | 4 x 256 | 144 | 144 | 0.0 | 0.0 | 30.0 | no | ?;none (1 call) | - | - |
| polybench/ADI | `_Z11adi_kernel6iPfS_S_i` | 4 x 256 | 152 | 152 | 0.0 | 0.0 | 30.0 | no | ?;none (1 call) | - | - |
| polybench/ATAX | `_Z12atax_kernel1iiPfS_S_` | 128 x 256 | 216 | 216 | 2.2 | 2.2 | 120.9 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/ATAX | `_Z12atax_kernel2iiPfS_S_` | 128 x 256 | 216 | 216 | 2.2 | 2.2 | 120.9 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/BICG | `_Z12bicg_kernel1iiPfS_S_` | 16 x 256 | 216 | 216 | 0.3 | 0.3 | 120.9 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/BICG | `_Z12bicg_kernel2iiPfS_S_` | 16 x 256 | 216 | 216 | 0.3 | 0.3 | 120.9 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/CORR | `_Z10std_kerneliiPfS_S_` | 8 x 256 | 232 | 232 | 0.1 | 0.1 | 30.0 | no | ?;none; ?;reg; ?;none (2 calls) | - | - |
| polybench/CORR | `_Z11mean_kerneliiPfS_` | 8 x 256 | 176 | 176 | 0.1 | 0.1 | 30.0 | no | ?;none; ?;reg; ?;none (1 call) | - | - |
| polybench/COVAR | `_Z11mean_kerneliiPfS_` | 8 x 256 | 176 | 176 | 0.1 | 0.1 | 30.0 | no | ?;none; ?;reg; ?;none (1 call) | - | - |
| polybench/GEMM | `_Z11gemm_kerneliiiffPfS_S_` | 1024 x 256 | 256 | 256 | 21.0 | 21.0 | 30.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/GEMVER | `_Z14gemver_kernel2iffPfS_S_S_` | 16 x 256 | 240 | 240 | 0.3 | 0.3 | 121.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/GEMVER | `_Z14gemver_kernel3iffPfS_S_` | 16 x 256 | 240 | 240 | 0.3 | 0.3 | 121.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/GESUMMV | `_Z14gesummv_kerneliffPfS_S_S_S_` | 16 x 256 | 128 | 128 | 0.2 | 0.2 | 241.7 | no | ?;reg; ?;reg; ?;none | - | - |
| polybench/GRAMSCHM | `_Z19gramschmidt_kernel2iiPfS_S_i` | 8 x 256 | 152 | 152 | 0.1 | 0.1 | 30.0 | no | ?;none (1 call) | - | - |
| polybench/GRAMSCHM | `_Z19gramschmidt_kernel3iiPfS_S_i` | 8 x 256 | 144 | 144 | 0.1 | 0.1 | 90.6 | no | ?;none; ?;reg; ?;none; ?;reg; ?;none | - | - |
| polybench/LU | `_Z10lu_kernel1iPfi` | 8 x 256 | 144 | 144 | 0.1 | 0.1 | 30.0 | no | ?;none (1 call) | - | - |
| polybench/MVT | `_Z11mvt_kernel1iPfS_S_` | 128 x 256 | 216 | 216 | 2.2 | 2.2 | 120.9 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| polybench/MVT | `_Z11mvt_kernel2iPfS_S_` | 128 x 256 | 216 | 216 | 2.2 | 2.2 | 120.9 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| rodinia/bfs | `_Z6KernelP4NodePiPbS2_S2_S1_i` | 8 x 512 | 64 | 64 | 0.1 | 0.1 | 30.0 | no | ?;none; ?;none | - | - |
| rodinia/hotspot | `_Z14calculate_tempiPfS_S_iiiiffffff` | 36 x 256 | 352 | 352 | 1.0 | 1.0 | 30.0 | no | ?;param; ?;none (4 calls) | - | - |
| rodinia/lavaMD | `_Z15kernel_gpu_cuda7par_str7dim_strP7box_strP11FOUR_VECTORPdS4_` | 216 x 128 | 392 | 6223 | 3.4 | 53.8 | 30.0 | **yes** | ?;reg; 0;imm; ?;none; 0;imm; 0;imm; 50;imm; ?;none (1 call) | 6.8 (0.1) | 197, no |
| rodinia/lud | `_Z13lud_perimeterPfii` | 63 x 32 | 816 | 3741 | 0.5 | 2.4 | 30.0 | no | 16;imm; 0;imm; ?;reg; ?;none; 0;imm; 0;imm; ?;none (1 call) | - | - |
| rodinia/nn | `_Z6euclidP7latLongPfiff` | 168 x 256 | 72 | 72 | 1.0 | 1.0 | 30.0 | no | ?;none (1 call) | - | - |
| rodinia/particlefilter_float | `_Z10sum_kernelPdi` | 20 x 512 | 144 | 144 | 0.5 | 0.5 | 30.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| rodinia/particlefilter_float | `_Z17likelihood_kernelPdS_S_S_S_PiS0_S_PhS_S_iiiiiiS0_S_` | 20 x 512 | 1424 | 1424 | 4.6 | 4.6 | 30.0 | no | ?;reg; ?;reg; ?;reg; ?;none; ?;reg; ?;none (20 calls) | - | - |
| rodinia/particlefilter_float | `_Z24normalize_weights_kernelPdiS_S_S_Pi` | 20 x 512 | 464 | 464 | 1.5 | 1.5 | 30.0 | no | 0;imm; 0;imm; ?;reg; ?;none (4 calls) | - | - |
| rodinia/pathfinder | `_Z14dynproc_kerneliPiS_S_iiii` | 463 x 256 | 96 | 96 | 3.6 | 3.6 | 72.7 | no | ?;param; ?;none | - | - |
| rodinia/sc_gpu | `_Z19kernel_compute_costiilP5PointiiPfS1_PiPb` | 8 x 512 | 280 | 280 | 0.4 | 0.4 | 30.0 | no | 0;imm; ?;reg; ?;reg; ?;none | - | - |
| samples/LargeKernelParameter | `_Z13kernelDefault7param_tPi` | 1 x 1 | 1608 | 12892 | 3.5 | 0.1 | 30.0 | no | 218;imm; ?;none | - | - |
| samples/MC_EstimatePiInlineP | `_Z7initRNGP17curandStateXORWOWj` | 782 x 128 | 304 | 2315 | 9.5 | 72.4 | 30.0 | **yes** | ?;none; ?;none; 5;imm; 2;imm; ?;none (2 calls) | 6.2 (0.2) | 595, no |
| samples/MC_SingleAsianOptionP | `_Z12computeValueIfEvPT_PKS0_PK11AsianOptionIS0_Ejj` | 782 x 128 | 320 | 320 | 10.0 | 10.0 | 71.7 | no | ?;none; 0;imm; ?;reg; ?;none; ?;none; ?;none; ?;none (4 calls) | 5.4 (0.4) | 12, no |
| samples/MC_SingleAsianOptionP | `_Z7initRNGP17curandStateXORWOWj` | 782 x 128 | 304 | 2315 | 9.5 | 72.4 | 71.7 | **yes** | ?;none; ?;none; 5;imm; 2;imm; ?;none (2 calls) | 5.5 (0.2) | 569, no |
| samples/StreamPriorities | `_Z13memcpy_kernelPiS_m` | 1024 x 512 | 48 | 48 | 7.9 | 7.9 | 1932.7 | no | ?;none; ?;none | - | - |
| samples/alignedTypes | `_Z10testKernelIhEvPT_S1_i` | 64 x 256 | 104 | 104 | 0.5 | 0.5 | 180.0 | no | ?;reg; ?;none; ?;none | - | - |
| samples/binaryPartitionCG | `_Z20oddEvenCountAndSumCGPiS_S_j` | 160 x 768 | 136 | 136 | 5.2 | 5.2 | 30.0 | no | ?;none (4 calls) | - | - |
| samples/clock | `__nv_static_25__66f34f0d_8_clock_cu_main__Z14timedReductionPKfPfPl` | 64 x 256 | 64 | 64 | 0.3 | 0.3 | 30.0 | no | ?;none; ?;none | - | - |
| samples/dct8x8 | `_Z27CUDAkernelQuantizationFloatPfi` | 4096 x 64 | 48 | 48 | 3.9 | 3.9 | 30.0 | no | ?;none (1 call) | - | - |
| samples/fp16ScalarProduct | `_Z26scalarProductKernel_nativePK7__half2S1_Pfm` | 128 x 128 | 104 | 104 | 0.5 | 0.5 | 30.0 | no | ?;none; ?;none | - | - |
| samples/fp16ScalarProduct | `_Z30scalarProductKernel_intrinsicsPK7__half2S1_Pfm` | 128 x 128 | 96 | 96 | 0.5 | 0.5 | 30.0 | no | ?;none; ?;none | - | - |
| samples/histogram | `_Z17histogram64KernelPjP5uint4j` | 4370 x 64 | 288 | 288 | 25.2 | 25.2 | 135.9 | no | ?;none; ?;none (1 call) | - | - |
| samples/histogram | `_Z18histogram256KernelPjS_j` | 240 x 192 | 192 | 192 | 2.8 | 2.8 | 30.0 | no | ?;none; 0;imm; ?;none (1 call) | - | - |
| samples/histogram | `_Z22mergeHistogram64KernelPjS_j` | 64 x 256 | 136 | 136 | 0.7 | 0.7 | 135.9 | no | ?;reg; ?;none | - | - |
| samples/interval | `_Z20test_interval_newtonIdEvP12interval_gpuIT_EPiS2_i` | 1024 x 64 | 1872 | 1872 | 38.3 | 38.3 | 242.1 | no | ?;none; ?;none; ?;none; ?;none (40 calls) | - | - |
| samples/jacobiCudaGraphs | `__nv_static_30__43dbd644_9_jacobi_cu_3dc991e0__Z12JacobiMethodPKfPKdfPdS3_S3_` | 66 x 256 | 624 | 3102 | 3.3 | 16.4 | 30.0 | no | ?;none; ?;reg; 15;imm; ?;none; ?;none; ?;reg; 15;imm; ?;none (22 calls) | - | - |
| samples/mergeSort | `_Z21mergeSortSharedKernelILj1EEvPjS0_S0_S0_j` | 4096 x 512 | 120 | 120 | 78.6 | 78.6 | 182.1 | no | ?;none; ?;none; ?;none; ?;none (3 calls) | - | - |
| samples/mergeSort | `_Z25generateSampleRanksKernelILj1EEvPjS0_S0_jjj` | 64 x 256 | 128 | 128 | 0.7 | 0.7 | 30.0 | no | ?;none; ?;none; ?;none | - | - |
| samples/mergeSort | `_Z26mergeRanksAndIndicesKernelPjS_jjj` | 64 x 256 | 120 | 120 | 0.6 | 0.6 | 30.0 | no | ?;none; ?;none; ?;none | - | - |
| samples/radixSortThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10radix_sort30DeviceRadixSortHistogramKernelINS1_5radix10policy_hubIjjyE10Policy1000ELNS0_9SortOrderE0EjyNS1_21identity_decomposer_tEEEvPT2_PKT1_SA_iiT3_` | 800 x 128 | 504 | 504 | 16.1 | 16.1 | 30.5 | no | ?;none; 1;imm; ?;reg; ?;none; ?;reg; 1;imm; ?;reg; ?;none (2 calls) | - | - |
| samples/radixSortThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10radix_sort33DeviceRadixSortExclusiveSumKernelINS1_5radix10policy_hubIjjyE10Policy1000EyEEvPT0_` | 4 x 256 | 216 | 216 | 0.1 | 0.1 | 30.0 | no | ?;none (10 calls) | - | - |
| samples/reduction | `_Z7reduce7IiLj256ELb1EEvPKT_PS0_j` | 64 x 256 | 104 | 104 | 0.5 | 0.5 | 30.0 | no | ?;none; ?;none; ?;none (3 calls) | - | - |
| samples/scalarProd | `_Z13scalarProdGPUPfS_S_ii` | 128 x 256 | 416 | 416 | 4.3 | 4.3 | 30.0 | no | ?;none; ?;none; ?;reg; ?;none; ?;none; ?;none; ?;none; ?;none; ?;none; ?;none; ?;none; ?;none; ?;none; ?;none (12 calls) | - | - |
| samples/segmentationTreeThrust | `_Z18getRepresentativesPKjPjj` | 4800 x 256 | 32 | 32 | 12.3 | 12.3 | 521.6 | no | ?;none; ?;none | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10merge_sort26DeviceMergeSortMergeKernelINS2_10policy_hubIN6thrust58THRUST_300001_SM_750_800_860_870_890_900_1000_1100_1200_NS12zip_iteratorIN4cuda3std3__45tupleIJNS6_10device_ptrIjEESD_EEEEEE9Policy600ESF_PNS0_8NullTypeESF_SJ_jNSA_4lessINSB_IJjjEEEEESL_SI_EEvbT2_T3_T4_PT6_PT7_T5_PSP_SP_NS1_7vsmem_tE` | 600 x 256 | 1416 | 1416 | 68.0 | 68.0 | 539.3 | no | ?;none; ?;none; ?;none | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10merge_sort30DeviceMergeSortBlockSortKernelINS2_10policy_hubIN6thrust58THRUST_300001_SM_750_800_860_870_890_900_1000_1100_1200_NS12zip_iteratorIN4cuda3std3__45tupleIJNS6_10device_ptrIjEESD_EEEEEE9Policy600ESF_PNS0_8NullTypeESF_SJ_jNSA_4lessINSB_IJjjEEEEESL_SI_EEvbT0_T1_T2_T3_T4_PT6_PT7_T5_NS1_7vsmem_tE` | 600 x 256 | 2080 | 2080 | 99.8 | 99.8 | 539.3 | no | ?;none; ?;none; ?;none; ?;none; ?;none (2 calls) | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10merge_sort30DeviceMergeSortBlockSortKernelINS2_10policy_hubIN6thrust58THRUST_300001_SM_750_800_860_870_890_900_1000_1100_1200_NS12zip_iteratorIN4cuda3std3__45tupleIJNS6_10device_ptrIjEESD_EEEEEE9Policy600ESF_PNS0_8NullTypeESF_SJ_jNSA_4lessINSB_IJjjEEEEESL_SI_EEvbT0_T1_T2_T3_T4_PT6_PT7_T5_NS1_7vsmem_tE` | 2398 x 256 | 2080 | 2080 | 399.0 | 399.0 | 592.3 | no | ?;none; ?;none; ?;none; ?;none; ?;none (2 calls) | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10merge_sort30DeviceMergeSortPartitionKernelIN6thrust58THRUST_300001_SM_750_800_860_870_890_900_1000_1100_1200_NS12zip_iteratorIN4cuda3std3__45tupleIJNS5_10device_ptrIjEESC_EEEEEjNS9_4lessINSA_IJjjEEEEESG_EEvbT_PT2_T0_SL_PSL_T1_SL_i` | 3 x 256 | 144 | 144 | 0.0 | 0.0 | 30.0 | no | ?;none; ?;none; ?;none | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10radix_sort30DeviceRadixSortHistogramKernelINS1_5radix10policy_hubIjjyE10Policy1000ELNS0_9SortOrderE0EjyNS1_21identity_decomposer_tEEEvPT2_PKT1_SA_iiT3_` | 800 x 128 | 504 | 504 | 16.1 | 16.1 | 539.7 | no | ?;none; 1;imm; ?;reg; ?;none; ?;reg; 1;imm; ?;reg; ?;none (2 calls) | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10radix_sort31DeviceRadixSortSingleTileKernelINS1_5radix10policy_hubIjjyE10Policy1000ELNS0_9SortOrderE0EjjyNS1_21identity_decomposer_tEEEvPKT1_PSA_PKT2_PSE_T3_iiT4_` | 1 x 256 | 872 | 872 | 0.3 | 0.1 | 521.7 | no | ?;param; ?;none | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail10radix_sort33DeviceRadixSortExclusiveSumKernelINS1_5radix10policy_hubIjjyE10Policy1000EyEEvPT0_` | 4 x 256 | 216 | 216 | 0.1 | 0.1 | 30.0 | no | ?;none (10 calls) | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail13unique_by_key28DeviceUniqueByKeySweepKernelINS2_10policy_hubIjiE10Policy1000EN6thrust58THRUST_300001_SM_750_800_860_870_890_900_1000_1100_1200_NS10device_ptrIjEENS8_17counting_iteratorIiNS8_11use_defaultESC_SC_EENS8_16discard_iteratorISC_EESA_PjNS0_13ScanTileStateIjLb1EEEN4cuda3std3__48equal_toIjEEjNS2_11VSMemHelperEEEvT0_T1_T2_T3_T4_T5_T6_T7_iNS1_7vsmem_tE` | 1746 x 64 | 1712 | 1712 | 59.8 | 59.8 | 521.6 | no | ?;reg; ?;reg; ?;reg; ?;reg; ?;none (30 calls) | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail4scan16DeviceScanKernelINS2_10policy_hubIjjjjN4cuda3std3__44plusIvEEE10Policy1000EN6thrust58THRUST_300001_SM_750_800_860_870_890_900_1000_1100_1200_NS10device_ptrIjEESF_NS0_13ScanTileStateIjLb1EEES9_NS0_8NullTypeEjjLb0ESI_EEvT0_T1_T2_iT3_T4_T5_` | 2558 x 128 | 1088 | 1088 | 111.3 | 111.3 | 521.6 | no | ?;none (30 calls) | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail6reduce18DeviceReduceKernelINS2_10policy_hubIN4cuda3std3__45tupleIJblEEEjN6thrust58THRUST_300001_SM_750_800_860_870_890_900_1000_1100_1200_NS8cuda_cub9__find_if7functorIS9_EEE10Policy1000ENSB_12zip_iteratorINS8_IJNSD_26transform_input_iterator_tIbNSB_10device_ptrIjEE18IsGreaterEqualThanIjEEENSB_17counting_iteratorIlNSB_11use_defaultESQ_SQ_EEEEEEEjSF_S9_NS7_10__identityEEEvT0_PT3_T1_NS0_13GridEvenShareISY_EET2_T4_` | 1024 x 256 | 1288 | 1288 | 105.5 | 105.5 | 521.7 | no | ?;none; ?;reg; ?;reg; ?;reg; ?;reg; ?;none (24 calls) | - | - |
| samples/segmentationTreeThrust | `_ZN3cub52CUB_300001_SM_750_800_860_870_890_900_1000_1100_12006detail6select23DeviceSelectSweepKernelINS2_10policy_hubIjjiLb0ELNS0_10SelectImplE0EE10Policy1000EN6thrust58THRUST_300001_SM_750_800_860_870_890_900_1000_1100_1200_NS17counting_iteratorIjNS9_11use_defaultESB_SB_EENS9_10device_ptrIjEESE_PlNS0_13ScanTileStateIiLb1EEEN4cuda3std3__410__identityENS0_8NullTypeEiNS2_19streaming_context_tIlLb1EEELS5_0EEEvT0_T1_T2_T3_T4_T5_T6_T7_iT8_NS1_7vsmem_tE` | 2276 x 128 | 2296 | 2296 | 209.0 | 209.0 | 521.6 | no | ?;reg; ?;reg; ?;reg; ?;reg; ?;reg; ?;reg; ?;reg; ?;reg; ?;none (30 calls) | - | - |
| samples/shfl_scan | `_Z14shfl_scan_testPiiS_` | 256 x 256 | 144 | 144 | 2.9 | 2.9 | 30.0 | no | ?;none (2 calls) | - | - |
| samples/shfl_scan | `_Z18shfl_intimage_rowsPK5uint4PS_` | 1080 x 120 | 432 | 432 | 18.7 | 18.7 | 74.6 | no | ?;none (8 calls) | - | - |
| samples/shfl_scan | `_Z18shfl_vertical_shflPjii` | 60 x 256 | 176 | 176 | 0.8 | 0.8 | 30.0 | no | ?;none; ?;none | - | - |
| samples/simpleCudaGraphs | `_Z6reducePfPdmm` | 512 x 512 | 128 | 128 | 10.5 | 10.5 | 120.8 | no | ?;none; ?;reg; ?;none | - | - |
| samples/simpleMultiGPU | `__nv_static_35__93317c59_17_simpleMultiGPU_cu_main__Z12reduceKernelPfS_i` | 32 x 256 | 88 | 88 | 0.2 | 0.2 | 241.7 | no | ?;reg; ?;none; ?;none | - | - |
| samples/simplePrintf | `_Z10testKerneli` | 4 x 8 | 32 | 32 | 0.0 | 0.0 | 30.0 | no | ?;none (1 call) | - | - |
| samples/sortingNetworks | `_Z17bitonicSortSharedPjS_S_S_jj` | 1024 x 512 | 120 | 120 | 19.7 | 19.7 | 30.2 | no | ?;none; ?;none; ?;none; ?;none (3 calls) | - | - |
| samples/threadFenceReduction | `_Z15reduceMultiPassILj128ELb1EEvPKfPfj` | 64 x 128 | 200 | 200 | 0.5 | 0.5 | 30.0 | no | ?;none; 0;imm; 0;imm; ?;reg; ?;none (1 call) | - | - |
| samples/threadFenceReduction | `_Z16reduceSinglePassILj128ELb1EEvPKfPfj` | 64 x 128 | 496 | 496 | 1.3 | 1.3 | 30.0 | no | ?;none; 0;imm; 0;imm; ?;reg; ?;reg; 0;imm; 0;imm; ?;reg; ?;none (10 calls) | - | - |
| samples/warpAggregatedAtomicsCG | `_Z12mapToBucketsPKiPiS1_ii` | 19531 x 512 | 104 | 104 | 325.0 | 325.0 | 504.0 | no | ?;none (3 calls) | - | - |
| samples/warpAggregatedAtomicsCG | `_Z25calculateMaxInEachBucketsPKiS0_Piii` | 19531 x 512 | 96 | 96 | 300.0 | 300.0 | 576.0 | no | ?;none (3 calls) | - | - |
