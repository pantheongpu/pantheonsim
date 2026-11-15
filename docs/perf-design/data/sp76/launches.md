# Round sp76: the round's programs, launch by launch (server3, the stand-in driver, turbo mode)

`VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1`, profile `nvidia/a10g`, `VGPU_THREADS=3`, the build of the round. The stand-in driver refuses every
module unless it is told to be a card that keeps memory and computes nothing (`STUB_CARD_OK=1 STUB_KEEP_MEMORY=1`); a launch it refuses is
run by the simulator. server3's simulator runs these programs 2.5 to 3 times as fast as the A10G rig's.

## HeCBench aidw (#657), against a stand-in card that keeps memory and computes nothing

| launch | kernel | rule before (`VGPU_HOST_GPU_SAMPLE_STARTED=0`) | this round |
|---|---|---|---|
| 1 | `AIDW_Kernel` (400 x 256, 6.7 ms counted) | sampled (the process is 17 s old), the first block stopped at 15 ms: 1.6 s at the least against 0.45 s: sent, starts the driver | the same: 1.5 s at the least, sent; the sample 29 ms (the fork 0.5 ms) |
| 2 | `AIDW_Kernel_Tiled` (400 x 256, 84 ms counted) | `host_gpu_nosample: ... reason=the driver has been started`: simulated (the run was cut there) | `2.34 MiB only the card held pulled before a sample (0.369 ms)`; `sampled with the driver up`: the first block stopped at 15 ms (timed 15.1 ms), 2.0 s at the least against a card path of 0.25 s (an eighth of the estimate for the card's kernel; no start): **sent**; the sample 32 ms (the fork 0.5 ms) |
| 3, 4 | the two kernels again | (by their history, as in Round sp75's card run) | on the card, by the estimates their samples left |

`host_gpu_sampled: sampled=2 sent=2 timeouts=0 cost_s=0.061552 after_driver=1 copies_running_at_exit=0 copies_left=0`; `4 launches ran on the
card, 0 in the simulator`. On the rig the first block of each kernel takes about 1.5 s where it takes 0.6 s here, so both samples stop at
the limit there too.

## What a sample would say of the four programs of #656 that are under the floor in a young process

Each program once as it is and once with `VGPU_HOST_GPU_SAMPLE=always` (neither the floor nor the process's age is asked). "needs" is twice
the card path with the start, which the estimate must pass.

| program | kernel | grid | counted | floor | process age | takes here | by default | a sample says | needs | so |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| samples/MC_SingleAsianOptionP | `initRNG` | 782 x 128 | 9.5 ms | 72 ms (38 MiB live) | 5 ms | 1.35 s | not sampled: under the floor, too young | 0.79 s (one block, 3.0 ms; the sample 6.8 ms) | 0.71 s | sent here by 11 %; about 2 s on the rig: sent. The two kernels after it are then sampled with the driver up: `generatePaths` 0.117 s against a card path of 0.022 s, sent; `computeValue` 10.9 ms against 8.7 ms, kept |
| samples/MC_EstimatePiInlineP | `initRNG` | 782 x 128 | 9.5 ms | 30 ms | 4 ms | 1.46 to 1.49 s | not sampled: under the floor, too young | 0.56 s (one block, 2.2 ms; the sample 5.9 ms) | 0.64 s | kept here (the estimate is 2.6 times under the launch: one block was timed before the budget of 2 ms ended); about 1.4 s on the rig: sent |
| samples/warpAggregatedAtomicsCG | `filter_arr` | 19,532 x 512 | 250 ms | 0.14 s | 0.07 s | 0.21 s | sampled: 0.16 to 0.20 s, kept | the same | 0.57 s | kept, rightly here; about 0.5 s on the rig: kept |
| | `mapToBuckets` | 19,531 x 512 | 325 ms | 0.50 s (267 MiB) | 0.33 s | 0.26 to 0.29 s | not sampled: under the floor, too young | 0.22 s (32 blocks; the sample 5.1 ms, the fork 2.1 ms) | 0.66 s | kept; about 0.55 s on the rig: kept |
| | `calculateMaxInEachBuckets` | 19,531 x 512 | 300 ms | 0.58 s (305 MiB) | 0.66 s | 0.35 s | not sampled: under the floor, too young | 0.25 s (32 blocks; the sample 6.1 ms, the fork 2.9 ms) | 0.68 s | kept; about 0.65 s on the rig: at the bar |
| rodinia/lavaMD | `kernel_gpu_cuda` | 216 x 128 | 3.4 ms | 30 ms | 3 ms | 0.63 s | not sampled: under the floor, too young | 0.20 s (one block of the 32 it may take, 2.8 ms; the sample 6.9 ms) | 0.55 s | kept (the estimate is 3 times under the launch: the one block timed is box 0, a corner of the grid of boxes, with the fewest neighbours); about 0.5 s on the rig: at the bar |

What a safe rule would need, per program:

* `initRNG`: only the floor is in its way, and a sample would send it on the rig. What tells it from the kernels the floor is there for
  (HeCBench silu's and s3d's: 5 to 14 ms counted, under 4 ms taken) is that its loops have immediate bounds (32 turns in 5), which make its
  9.5 ms 1.5 s. That needs (1) the flow scan to follow a loop that is left through `CALL.REL.NOINC` to the kernel's `EXIT` (today any call
  makes every loop of the kernel "not followed"), (2) loops inside loops, (3) the weighted count asked of the floor for a sample by whole
  blocks (today only for a sample inside the blocks), and (4) a census of every kernel of the corpus whose count the weights raise over the
  floor, since each would be sampled in a young process (3 to 12 ms, and the page faults). Not small; not built.
* warpAggregatedAtomicsCG: not the floor. Sampled, its three kernels are kept here and on the rig they are at or under the bar: the launches
  (0.2 to 0.35 s here) do not repay a start of 0.25 s with 270 to 305 MiB to move each way. What the floor's term for the page faults costs
  there is not what keeps them.
* lavaMD: the floor of a small grid (30 ms counted) and, behind it, an estimate from one corner block. A safe rule would have to time
  blocks spread over the grid within the budget (its blocks take 3 ms: the budget of 2 ms ends after the first) and still be at the bar
  on the rig. Nothing known before the launch tells it from s3d's kernels on 108 blocks.
