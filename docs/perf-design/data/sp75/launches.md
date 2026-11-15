# Round sp75: the round's programs, launch by launch (server3, the stand-in driver, turbo mode)

`VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1`, profile `nvidia/a10g`, `VGPU_THREADS=3`. The stand-in driver refuses every module, so a launch the
mode sends is run by the simulator and timed there. "Kept by" is the reason the mode gives for a launch under the rule before the round (the
branch's first build: the rule of 6cf503d3 with the reasons named); "new" is what this round's rule decides. `static` is the static estimate
(every instruction once a warp); `floor` what a sample asks of it. The simulator here runs these programs 2.5 to 3 times as fast as the A10G
rig's did in the summary run (4 vCPUs there), so a launch that is 0.6 s here is 1.5 s there.

## The programs of #656 and #657

| program | kernel | grid | static | live | sim s | kept by (before) | new |
|---|---|---|---:|---:|---:|---|---|
| samples/tf32TensorCoreGemm | `compute_tf32gemm_async_copy` | 160 x 256 | 116 ms | 768 MiB | 83.3 (Round sp66) | no sample possible: a grid under 256 blocks and no counted loop to follow | sampled by bounded blocks, the first block stopped at its limit: 1.37 s at least, over the 1.14 s the card path needs: **sent** |
| samples/bf16TensorCoreGemm | `compute_bf16gemm_async_copy` | 160 x 256 | 46 ms | 768 MiB | 56.6 (Round sp66) | the same | 1.42 s at least, over 1.15 s: **sent** |
| samples/cudaTensorCoreGemm | `compute_gemm` | 80 x 256 | 46 ms | 192 MiB | 5.3 to 5.5 | the same | 0.95 s at least, over 0.81 s: **sent**; the sample cost 80 ms (the fork 3 ms; the limit 40.2 ms, the timed run 35.7 ms) |
| samples/immaTensorCoreGemm (not in the summary run) | `compute_gemm_imma` | 80 x 256 | over 30 ms | | 0.94 to 0.96 | the same | its first block ends in 29 ms, under the limit (39.3 ms): 0.77 s, over 0.755 s: **sent**, by 2 %; the sample cost 67 ms |
| hecbench/aidw-cuda | `AIDW_Kernel`, launch 1 | 400 x 256 | 6.7 ms | 2.7 MiB | 77.6 | sampled (the process is 12.5 s old); the copy did not answer in 50 ms: a block takes 0.58 s | the first block stopped at 15 ms: 1.8 to 1.9 s at least: **sent**; the sample cost 30 to 34 ms (50 ms before, for no answer) |
| | `AIDW_Kernel_Tiled`, launch 2 | 400 x 256 | 84 ms | 2.7 MiB | not run to its end | sampled; the copy did not answer | with launch 1 on the card (a stand-in that computes nothing): not sampled, the driver has been started: **kept** |
| | launches 3 and 4 | | | | | sent by their history (one long simulated launch each): the 2 of 4 of the summary run | 3 by the estimate its sample left, 4 by its history (not run here) |
| samples/MC_SingleAsianOptionP | `initRNG` | 782 x 128 | 9.5 ms | 38 MiB | 1.61 to 1.64 | not sampled: static estimate under the floor (72 ms) and the process too young (5 ms) | the same: kept |
| | `generatePaths` | 782 x 128 | 11 ms | 38 MiB | 0.15 | sampled: 0.18 s, under twice the card path with the start (0.28 s): kept, rightly | the same |
| | `computeValue` | 782 x 128 | 10 ms | 38 MiB | 0.02 | sampled: 0.017 s: kept, rightly | the same |
| samples/MC_EstimatePiInlineP | `initRNG` | 782 x 128 | 9.5 ms | 4.8 MiB | 1.64 to 1.65 | not sampled: static estimate under the floor (30 ms) and the process too young (5 ms) | the same: kept |
| | `computeValue` | 782 x 128 | 3 ms | 4.8 MiB | 0.004 | sampled: 0.004 s: kept, rightly | the same |
| samples/warpAggregatedAtomicsCG | `filter_arr` | 19,532 x 512 | 250 ms | 76 MiB | 0.22 | sampled: 0.17 s, under twice 0.29 s: kept | the same |
| | `mapToBuckets` | 19,531 x 512 | 325 ms | 267 MiB | 0.26 to 0.29 | not sampled: static estimate under the floor (0.50 s: three times the faults of 267 MiB) and the process too young (0.34 s) | the same: kept |
| | `calculateMaxInEachBuckets` | 19,531 x 512 | 300 ms | 305 MiB | 0.35 to 0.38 | the same (floor 0.58 s) | the same: kept |
| rodinia/lavaMD | `kernel_gpu_cuda` | 216 x 128 | 3.4 ms | 1.6 MiB | 0.56 to 0.62 | no sample possible: a grid under 256 blocks and no counted loop to follow | a candidate of the bounded sample, not sampled: static estimate under the floor (30 ms) and the process too young (4 ms): kept |

The kernels with no followed loop: the tensor GEMMs are one loop of 3,592 to 9,088 straight-line instructions that adds the grid's size to
its counter (`UIADD3 UR36, UR36, UR4` with `UR4` loaded from `c[0x0][0xc]`) and ends on a mask of it (`ULOP3.LUT UP0, URZ, UR36, 0x1ffffc00`),
left through `@P1 CALL.REL.NOINC` to the kernel's `EXIT`. lavaMD and `initRNG` leave their loops the same way; the flow scan takes any `CALL`
for a jump it cannot see, so no loop of such a kernel is followed (and every such kernel "has a loop"). `initRNG`'s loops have immediate bounds
(`ISETP.NE R21, 0x20` inside `ISETP.GE.U32 R14, 0x5`: 32 x 5 turns, which makes its 9.5 ms 1.5 s; it takes 1.6 s).

## The programs of #659 (launches kept in programs that use the card)

Simulator seconds by reason, all launches (the stand-in driver: "sent" launches are simulated too, and no memory is ever the card's alone, so the
rule for a launch while the card holds the memory, section 9, cannot act here: on a card more of the "does not beat the card path" launches go).

| program | launches | sent | the estimate does not beat the card path | the estimate does not repay the driver start | first launches with no sample: the driver has been started | ... under the floor, young | ... no loop | ... its earlier launch was short | other |
|---|---:|---|---|---|---|---|---|---|---|
| samples/segmentationTreeThrust | 514 | 19 (4.02 s) | 438 (2.23 s) | - | 34 (0.45 s) | 5 (0.39 s) | 14 (0.20 s) | 3 (0.04 s) | 1 sampled inside its blocks with no estimate (0.12 s) |
| samples/mergeSort | 49 | 11 (2.31 s) | 32 (0.06 s) | - | 2 (0.21 s) | 3 (0.62 s) | - | 1 (0.00 s) | |
| samples/histogram | 68 | 30 (4.94 s) | 30 (0.04 s) | - | 4 (0.30 s) | 2 (0.18 s) | - | 2 (0.18 s) | |
| hecbench/radixsort2-cuda | 600 | 209 (0.44 s) | 98 (0.01 s) | 287 (0.67 s) | - | 3 (0.01 s) | - | 3 (0.01 s) | |
| rodinia/lud | 190 | 54 (0.27 s) | 109 (0.04 s) | 21 (0.21 s) | - | 1 (0.00 s) | 2 (0.04 s) | 1 (0.00 s) | 2 on a grid of one block |
| polybench/3MM | 3 | 1 (0.16 s) | - | - | - | 2 (0.32 s) | - | - | |

* segmentationTreeThrust: one kernel (a CUB sort pass) has 86 launches of about 19 ms that do not beat the card path here because every launch
  pushes its whole live memory (nothing stays on the stand-in); the rest of the 438 are launches of 0.1 to 5 ms. Of its first launches, 34
  come after the driver's start and so have no sample.
* mergeSort and histogram: what is kept is each kernel's first one or two launches (0.15 to 0.27 s each): under the floor in a young process
  before the start, no sample after it.
* radixsort2: 287 launches of 2 ms that the simulator ran before the start was bought (the account reaches 0.25 s only then).
* lud: the first 21 launches of `lud_internal` and `lud_perimeter`, before the start was bought; then launches too short for the card.
* 3MM: `mm3_kernel1` and `mm3_kernel2` (0.16 s each here, about 0.45 s on the rig): 19 ms counted, under the floor of 30 ms, 12 ms and 0.18 s
  into the process; `mm3_kernel3` goes by the first-launch rule once the account is full.

## The programs of #662

| program | launches | on the card | driver started | sampled | what the mode does |
|---|---:|---:|---|---:|---|
| hecbench/silu-cuda | 50 | 0 | no | 0 | scans 5 kernels; decides 50 launches that stay (40 of them with an estimate that is under the card path or does not repay the start) |
| hecbench/s3d-cuda | 54 | 0 | no | 0 | scans 54 kernels (27 in float on 108 or 216 blocks, 27 in double on 32 or 64; 216 to 4,864 instructions each); decides 54 launches that stay |

Neither program pays a driver start, a sample or a card launch. What is left is the mode's own work for a launch that stays; the numbers
are in README.md.
