# Round sp76 on a card

A10G (g5.xlarge, driver 595.112.07, CUDA 13.0), 2026-11-15, every stage of `card-run.sh`. `d0` forms: `VGPU_HOST_GPU_SAMPLE_STARTED=0`.

- Card CI passes with step 13. `fork`: 20 of 20 runs as expected in exact and in turbo (`sampled=2 sent=2 timeouts=0 after_driver=1
  copies_left=0`), no new NVRM or Xid line, no process left. The fork costs 0.13 ms before the start, 5 ms with the driver loaded.
- `changed`, turbod0 -> turbo (median of 10; aidw one run): aidw 181.57 -> 44.41 s (4 of 4 launches on the card, 3 before);
  mergeSort 3.18 -> 2.12; histogram 3.27 -> 2.30; segmentationTreeThrust 6.52 -> 5.25; outputs identical (aidw's not compared).
  atomicAggregate 1.48 -> 0.89 (3 runs).
- `keep` (18 programs, 3 runs): all within 3 % of the d0 form except particlefilter_float (turbo 1.07 against 0.88 s), which takes
  no sample with the driver up and is the program's own race (pf.md).
- `rest` (113 programs, ONE run): no program that samples with the driver up is slower. Seven over 0.3 s read 3 to 11 % slower
  with the same decisions in both forms; five of them are within 2 % in `keep`; samples/interval (1.09) and samples/reduction
  (1.07) were not rerun.
