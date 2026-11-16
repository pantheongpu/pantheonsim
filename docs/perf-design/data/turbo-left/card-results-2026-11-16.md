# Card results of 2026-11-16 (one A10G, nvidia/a10g profile, driver 595, CUDA 13.0 toolchain)

All numbers are from the rig run of the three scripts of this directory and `../sp76/rerun-interval-reduction.sh`, built from this tree
(the build stage, including the card CI, passed: "PASS: card CI on NVIDIA A10G"). Tags: measured unless said otherwise.

## samples/interval and samples/reduction: the single-run slowdown was noise
15 runs a form, interleaved, turbo mode, wall time:

| program | d0 (rule before sp76) | new (default) | new1 (new, d0's environment size) |
|---|---|---|---|
| interval | median 2.510 s (min 2.490, max 2.670) | 2.510 s (2.490, 2.680) | 2.500 s (2.490, 2.670) |
| reduction | median 0.870 s (0.850, 0.880) | 0.860 s (0.860, 0.870) | 0.860 s (0.850, 0.870) |

The new rule is within the spread of the old one on both programs (1.000 and 0.989 of d0). The 9 % and 7 % of the one-run pass in
`../sp76/card.md` were noise, as for FDTD-2D and segmentationTreeThrust. The script's "OUTPUT DIFFERS" lines come from a flaw in the script,
not the simulator: both programs print their own timings ("Time per equation: ...") and the script compares whole outputs; ignore them
(to fix in the script: compare with the timing lines removed).

## #660 very short launches
Card: async launch 2.78 us, launch plus sync 7.98 us, launch plus a 4-byte copy 10.47 us, a 1 MiB push 90.3 us (11.6 GB/s), a pull 87.7 us
(12.0 GB/s). `nsys` is not installed on the rig, so the card kernel time of JACOBI1D was not read (reading 3 skipped).

| program | native on the card, best of 5 | turbo today (sim time of the whole run) | launches sent |
|---|---|---|---|
| polybench/JACOBI1D | 0.454 s for 20,000 launches | 2.348 s | 0 of 20,000 (19,996 "too small") |
| rodinia/sc_gpu | 0.525 s for 1,432 | 0.932 s | 0 of 1,432 |
| samples/LargeKernelParameter | 0.386 s for 2,002 | 0.419 s | 0 of 2,002 |

What follows from it (inferred from these numbers, not run): the card's launch plus sync is 8 us against the 7 us of `Costs::launch_s`, so
that prior is about right; the 100 us `kernel_s_default` prior is what keeps these kernels out (the log prints "card path 0.110 ms" for
JACOBI1D whose launches take the simulator 0.10 to 0.17 ms, and "0.146 ms" for sc_gpu, 0.109 ms for LargeKernelParameter). JACOBI1D's
native 0.454 s includes process and context start, so its launches cost the card well under the 110 us the rule assumes: about 5 times less
work for the whole program if they were sent (2.35 s against about 0.5 s plus the moves) is the room, and LargeKernelParameter has
almost none (0.386 s native against 0.419 s). A lower prior or a measured one is the lever; changing it changes turbo decisions only
(exact mode and outputs are untouched), and has not been tried.

## #661 BMMA
Not measured. `card-bmma-verify.sh` found no `bmmaTensorCoreGemm`: no tag of NVIDIA/cuda-samples from v11.8 to v13.0 contains it or any
sample using the 1-bit `wmma` precision (checked on the rig with a grep for `precision::b1`). A test needs our own small `wmma` b1 program;
not written.
