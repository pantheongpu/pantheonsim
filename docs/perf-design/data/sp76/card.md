# Round sp76 on a card

One rented A10G (g5.xlarge, driver 595.112.07, CUDA 13.0), 2026-11-15, `card-run.sh build fork decisions changed keep pf rest`.
`turbo` / `exact` are this round's rule; `turbod0` / `exactd0` the same build with `VGPU_HOST_GPU_SAMPLE_STARTED=0`.

## The fork of a process that holds the real driver

- Card CI passes, with step 13 of `e2e_host_gpu` (a launch sampled after the driver's start).
- `fork` stage: 20 of 20 runs as expected in `exact` and 20 of 20 in `turbo`: output the same as mode off, `sampled=2 sent=2
  timeouts=0 after_driver=1 copies_left=0`. Kernel log lines with NVRM or Xid: 1 before, 1 after. No zombie, no process left.
- The fork costs 0.12 to 0.17 ms before the driver's start and 4.9 to 5.6 ms with the driver loaded.

## What changed (median of 10 runs; aidw one run a form)

| program | sim s | turbod0 s | turbo s | turbo over turbod0 | launches on the card, turbod0 -> turbo | output |
|---|---:|---:|---:|---:|---|---|
| hecbench/aidw-cuda | - | 181.57 | 44.41 | 0.245 | 3 of 4 -> 4 of 4 | not compared here |
| samples/mergeSort | 7.99 | 3.18 | 2.12 | 0.667 | 34 of 49 -> 42 of 49 | identical |
| samples/histogram | 13.24 | 3.27 | 2.30 | 0.703 | 58 of 68 -> 62 of 68 | identical |
| samples/segmentationTreeThrust | 15.73 | 6.52 | 5.25 | 0.805 | 241 of 514 -> 243 of 514 | identical |
| hecbench/atomicAggregate-cuda (3 runs) | 3.64 | 1.48 | 0.89 | 0.602 | 4 of 6 -> 5 of 6 | identical |

## What must not change

- `keep` (18 programs, 3 runs): every program within 2 % of its `d0` form in turbo and within 3 % in exact, except
  rodinia/particlefilter_float (turbo 1.07 s against 0.88 s; exact 2.33 against 2.25). It takes no sample with the driver up
  (`after_driver=0`) and its simulator runs do not print the same output twice: the race of `normalize_weights_kernel` (pf.md).
- `rest` (113 programs, ONE run a form) and the four slow ones: no program that takes a sample with the driver up is slower.
  Seven programs over 0.3 s read more than 3 % slower in that single run (rodinia/gaussian 1.113, samples/interval 1.091,
  hecbench/silu-cuda 1.090, samples/reduction 1.071, polybench/CORR 1.039, polybench/3MM 1.039, polybench/LU 1.031); none of them
  takes such a sample, so the two forms decide the same, and gaussian, silu, CORR, 3MM and LU are within 2 % in the 3-run `keep`
  table. interval and reduction were not rerun.
- `pf`: see the pf-*.md tables of the run (kept outside the repository).
