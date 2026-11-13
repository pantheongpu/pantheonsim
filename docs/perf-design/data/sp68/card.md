# Round sp68 on the A10G (data)

One AWS g5.xlarge (A10G, driver 595.112.07, 4 vCPU), 17:10 to about 18:50 UTC, nothing else on it. Builds `b68` (`sp68-sqrt-idiom` 18ad0ce2) and `b65`
(`sp68-with-sp65` 0a3d6c15, the scratch merge with Round sp65's streak rule), Release, `-DVGPU_LTO=OFF`, CUDA ABI headers of toolkit 13.2. Every run:
`VGPU_GPU=nvidia/a10g VGPU_THREADS=4`; sides: `sim` mode off, `x0` = `VGPU_HOST_GPU=exact VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_SQRT=0`, `x1` = the same with
`VGPU_HOST_GPU_SQRT=1`.

## Probe and e2e (`card/`)

* `hostgpu_sfu_exhaustive sqrtrn 1` (nvcc 13.0.88, sm_86): `tested=4294967296 differ=0` (sp44's result again); the harness's mutation at stride 257:
  `differ=8128` (it sees a one-ulp nudge).
* `run_host_gpu.sh` (with step 7, the sqrt kernels on the card, verify on and off): PASS. `run_host_gpu_stub.sh` (with step 28): PASS.

## A/B, 3 interleaved runs per side (`card_ab.sh`, `card/ab-*.json`)

"wall" is the whole process; "GPU Time" is what PolyBench prints for its device part (GRAMSCHM's wall is mostly its CPU reference, about 230 s).
"output hashes" hash stdout without its timing lines (knn prints a time on a line the filter keeps: its hashes differ run to run on every side, and its
output with that line dropped is the same on all three sides, `card/knn-hash.txt`). The PolyBench programs print a count of mismatches against their
CPU reference rather than their results, so their hash is a weak check; `verify` below compares the memory of every launch.

| program | set | side | wall s median (min-max) | GPU Time median | x1 vs x0 (wall) | rc | output hashes |
|---|---|---|---|---:|---:|---|---|
| CORR | ab-short | sim | 66.865 (66.011-67.174) | 26.264 |  | 0 | 96ba52add2de149b |
| CORR | ab-short | x0 | 42.229 (42.080-42.437) | 2.012 |  | 0 | 96ba52add2de149b |
| CORR | ab-short | x1 | 42.116 (42.039-52.310) | 2.011 | 1.003 | 0 | 96ba52add2de149b |
| COVAR | ab-short | sim | 67.438 (66.530-70.883) | 26.529 |  | 0 | 96ba52add2de149b |
| COVAR | ab-short | x0 | 42.383 (41.977-42.478) | 1.868 |  | 0 | 96ba52add2de149b |
| COVAR | ab-short | x1 | 42.179 (41.863-42.401) | 1.868 | 1.005 | 0 | 96ba52add2de149b |
| ADI | ab-short | sim | 0.359 (0.348-0.367) | 0.287 |  | 0 | ebfe712e2b8bd028 |
| ADI | ab-short | x0 | 0.359 (0.359-0.366) | 0.288 |  | 0 | ebfe712e2b8bd028 |
| ADI | ab-short | x1 | 0.364 (0.362-0.367) | 0.292 | 0.986 | 0 | ebfe712e2b8bd028 |
| hellinger | ab-short | sim | 43.008 (42.862-43.055) | - |  | 0 | c3ef86550c5323fa |
| hellinger | ab-short | x0 | 42.947 (42.899-43.025) | - |  | 0 | c3ef86550c5323fa |
| hellinger | ab-short | x1 | 6.758 (6.758-6.764) | - | 6.355 | 0 | c3ef86550c5323fa |
| knn | ab-short | sim | 16.963 (16.765-17.030) | - |  | 0 | 1cd0fd346df7200b b7d56911214653fb e3631d3c8be22d87 |
| knn | ab-short | x0 | 17.034 (17.015-17.138) | - |  | 0 | 468263dcb94eabb5 c496d6d6c9005074 c82c8a7678ea0a83 |
| knn | ab-short | x1 | 17.069 (16.709-17.307) | - | 0.998 | 0 | 2de2b34a54a65d11 82550c62869ee53b 9464ff3ba132ad47 |
| rod_nn | ab-short | sim | 0.021 (0.021-0.021) | - |  | 0 | 3c20c13a34d62746 |
| rod_nn | ab-short | x0 | 0.021 (0.021-0.021) | - |  | 0 | 3c20c13a34d62746 |
| rod_nn | ab-short | x1 | 0.022 (0.021-0.022) | - | 0.955 | 0 | 3c20c13a34d62746 |
| hec_nn | ab-short | sim | 0.027 (0.027-0.028) | - |  | 0 | 56d5fc657cda7d1d |
| hec_nn | ab-short | x0 | 0.027 (0.027-0.027) | - |  | 0 | 56d5fc657cda7d1d |
| hec_nn | ab-short | x1 | 0.027 (0.027-0.028) | - | 1.000 | 0 | 56d5fc657cda7d1d |
| GRAMSCHM | ab-gram | sim | 270.645 (269.948-278.350) | 45.215 |  | 0 | c9644fd81ac7b3dc |
| GRAMSCHM | ab-gram | x0 | 272.746 (263.011-274.312) | 44.589 |  | 0 | c9644fd81ac7b3dc |
| GRAMSCHM | ab-gram | x1 | 234.864 (233.071-238.609) | 3.081 | 1.161 | 0 | c9644fd81ac7b3dc |
| GRAMSCHM | ab-gram68 | x0 | 266.886 (266.886-266.886) | 45.063 |  | 0 | c9644fd81ac7b3dc |
| GRAMSCHM | ab-gram68 | x1 | 251.861 (251.861-251.861) | 44.928 | 1.060 | 0 | c9644fd81ac7b3dc |

* CORR x1: host_gpu: 1 launches ran on the card, 3 in the simulator (0 kernels not in the exact set, 0 too small, 0 unsupported launch shapes, 0 failed on the card); host_gpu: 32.0 MiB went to the card, 16.0 MiB came back 
* COVAR x1: host_gpu: 1 launches ran on the card, 2 in the simulator (0 kernels not in the exact set, 0 too small, 0 unsupported launch shapes, 0 failed on the card); host_gpu: 32.0 MiB went to the card, 16.0 MiB came back 
* ADI x1: host_gpu: 0 launches ran on the card, 2049 in the simulator (0 kernels not in the exact set, 2041 too small, 0 unsupported launch shapes, 0 failed on the card); host_gpu: 0.0 MiB went to the card, 0.0 MiB came back 
* hellinger x1: host_gpu: 1 launches ran on the card, 0 in the simulator (0 kernels not in the exact set, 0 too small, 0 unsupported launch shapes, 0 failed on the card); host_gpu: 31.5 MiB went to the card, 9.0 MiB came back 
* knn x1: host_gpu: 0 launches ran on the card, 3 in the simulator (1 kernels not in the exact set, 0 too small, 0 unsupported launch shapes, 0 failed on the card); host_gpu: 0.0 MiB went to the card, 0.0 MiB came back 
* rod_nn x1: host_gpu: 0 launches ran on the card, 1 in the simulator (0 kernels not in the exact set, 0 too small, 0 unsupported launch shapes, 0 failed on the card); host_gpu: 0.0 MiB went to the card, 0.0 MiB came back 
* hec_nn x1: host_gpu: 0 launches ran on the card, 10 in the simulator (0 kernels not in the exact set, 8 too small, 0 unsupported launch shapes, 0 failed on the card); host_gpu: 0.0 MiB went to the card, 0.0 MiB came back 
* GRAMSCHM x1: host_gpu: 6121 launches ran on the card, 23 in the simulator (0 kernels not in the exact set, 17 too small, 0 unsupported launch shapes, 0 failed on the card); host_gpu: 48.0 MiB went to the card, 16.0 MiB came back 

The short programs again, 20 interleaved runs per side, timed to the microsecond (`card_short.sh`), x0 / x1 median wall: Rodinia nn 0.0210 / 0.0211 s
(0.996), HeCBench nn 0.0271 / 0.0273 s (0.993), ADI 0.3651 / 0.3646 s (1.002); outputs equal.

## Verify (`card_verify.sh`, `card/verify.*`): `VGPU_HOST_GPU_VERIFY=1` with `VGPU_HOST_GPU_SQRT=1`, every launch run on both sides and compared

| program | build | in the set: identical / different |
|---|---|---|
| CORR | b68 | 4 / 0 |
| hellinger | b68 | 1 / 0 |
| knn | b68 | 2 / 0 |
| Rodinia nn | b68 | 1 / 0 |
| HeCBench nn | b68 | 10 / 0 |
| GRAMSCHM | b65 | 6,144 / 0 (2,048 launches of each of the three kernels; the whole run finished in the 600 s cap) |
