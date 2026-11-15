# Round sp76: wall time with the switch off (`d0`) and on (`new`)

server3, the stand-in driver, turbo mode, no log, `VGPU_THREADS=3`, the round's final build in both forms (`d0`:
`VGPU_HOST_GPU_SAMPLE_STARTED=0`). Medians, the forms interleaved, the machine's build lock kept over the runs (`rerun.py`). Against the
stand-in every launch is simulated in both forms, so a difference is what the rule's samples cost, not what the card saves. The programs
are the ones that start the driver against the stand-in (the rule cannot act in the others) and the ones the census shows the rule
touching. Another session built and ran on the machine throughout.

## Thirteen programs under 8 s, 7 runs a form

| program | d0 s | new s | new / d0 | spread of d0 |
|---|---:|---:|---:|---:|
| polybench/3MM | 1.360 | 1.348 | 0.991 | 1.353 to 1.386 |
| polybench/JACOBI2D | 0.268 | 0.269 | 1.007 | 0.257 to 0.289 |
| rodinia/gaussian | 1.764 | 1.746 | 0.990 | 1.674 to 1.807 |
| rodinia/lud | 2.290 | 2.268 | 0.990 | 2.257 to 2.306 |
| rodinia/hotspot | 0.451 | 0.450 | 0.997 | 0.431 to 0.459 |
| hecbench/asyncAllocation-cuda | 1.588 | 1.600 | 1.008 | 1.581 to 1.605 |
| hecbench/atomicAggregate-cuda | 1.659 | 1.656 | 0.998 | 1.639 to 1.715 |
| hecbench/atomicPerf-cuda | 0.529 | 0.532 | 1.004 | 0.525 to 0.535 |
| hecbench/radixsort2-cuda | 1.228 | 1.222 | 0.995 | 1.209 to 1.242 |
| hecbench/shmembench-cuda | 2.057 | 2.082 | 1.012 | 2.031 to 2.224 |
| hecbench/silu-cuda | 0.242 | 0.243 | 1.003 | 0.236 to 0.246 |
| samples/convolutionSeparable | 5.063 | 5.098 | 1.007 | 5.043 to 5.095 |
| samples/interval | 7.290 | 7.265 | 0.997 | 7.214 to 7.426 |

## The longer ones, 3 runs a form

| program | d0 s | new s | new / d0 | spread of d0 |
|---|---:|---:|---:|---:|
| polybench/FDTD-2D | 48.878 | 62.390 | 1.276 | 48.718 to 62.509 |
| polybench/LU | 12.851 | 13.069 | 1.017 | 12.841 to 12.938 |
| polybench/SYR2K | 5.502 | 5.517 | 1.003 | 5.494 to 5.509 |
| polybench/SYRK | 2.922 | 2.930 | 1.003 | 2.880 to 2.949 |
| hecbench/black-scholes-cuda | 10.565 | 10.561 | 1.000 | 10.533 to 10.598 |
| samples/mergeSort | 3.289 | 3.295 | 1.002 | 3.273 to 3.317 |
| samples/histogram | 5.416 | 5.395 | 0.996 | 5.376 to 5.440 |
| samples/segmentationTreeThrust | 20.422 | 21.939 | 1.074 | 18.902 to 21.223 |
| samples/jacobiCudaGraphs | 10.619 | 10.550 | 0.993 | 10.593 to 10.678 |
| samples/alignedTypes | 25.042 | 25.130 | 1.004 | 20.626 to 25.197 |

FDTD-2D and segmentationTreeThrust read over 3 % in that pass, each with a run of the switch-off form as slow as the slowest of the
switch-on form (see their spreads): neither takes a sample with the driver up, and their decisions are the same in both forms
(`census.md`). Run again:

| program | runs a form | d0 s | new s | new / d0 | spread of d0 |
|---|---:|---:|---:|---:|---:|
| polybench/FDTD-2D | 4 | 49.110 | 48.984 | 0.997 | 48.941 to 49.244 |
| samples/segmentationTreeThrust | 5 | 19.432 | 19.380 | 0.997 | 15.571 to 22.916 |

alignedTypes takes eleven samples with the driver up (64 ms in all, of 25 s), histogram two (12 ms of 5.4 s), mergeSort one (3 ms of
3.3 s), jacobiCudaGraphs one (1.7 ms of 10.6 s). Not timed: atomicReduction (57 s here; two samples of 2 to 7 ms).
