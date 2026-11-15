# Round sp75: wall time with the switch off (knob) and on (new)

server3, the stand-in driver, turbo mode, no log, `VGPU_THREADS=3`, the build of ae13d5e6 in both forms (`knob`: `VGPU_HOST_GPU_SAMPLE_BLOCK=0`).
Median of 9 runs a form, interleaved, each run under the machine's build lock (`rerun.py`). Against the stand-in every launch is simulated in
both forms, so a difference is what the rule's samples cost, not what the card saves: cudaTensorCoreGemm and immaTensorCoreGemm are sampled
and "sent" under `new`, and their launch is then simulated all the same (+74 ms and +67 ms: the sample). The other programs take the same
decisions in both forms. dct8x8's runs spread from 1.4 to 3.6 s in either form.

| program | knob s | new s | new / knob | spread of knob |
|---|---:|---:|---:|---:|
| polybench/3DCONV | 0.410 | 0.409 | 0.998 | 0.364 to 0.422 |
| polybench/3MM | 1.333 | 1.342 | 1.006 | 1.302 to 1.356 |
| polybench/GEMVER | 0.392 | 0.394 | 1.007 | 0.391 to 0.400 |
| polybench/GESUMMV | 0.264 | 0.264 | 1.002 | 0.259 to 0.267 |
| polybench/JACOBI1D | 0.562 | 0.559 | 0.995 | 0.552 to 0.715 |
| polybench/JACOBI2D | 0.267 | 0.265 | 0.992 | 0.259 to 0.270 |
| polybench/MVT | 0.744 | 0.742 | 0.997 | 0.737 to 0.749 |
| polybench/SYR2K | 5.523 | 5.539 | 1.003 | 5.497 to 5.620 |
| polybench/SYRK | 2.907 | 2.906 | 1.000 | 2.817 to 3.029 |
| rodinia/gaussian | 1.687 | 1.718 | 1.018 | 1.639 to 1.751 |
| rodinia/lud | 2.285 | 2.285 | 1.000 | 2.245 to 2.304 |
| rodinia/pathfinder | 0.423 | 0.429 | 1.014 | 0.413 to 0.446 |
| rodinia/srad_v2 | 0.031 | 0.031 | 1.001 | 0.029 to 0.033 |
| rodinia/particlefilter_float | 1.308 | 1.239 | 0.947 | 1.105 to 1.402 |
| rodinia/sc_gpu | 0.422 | 0.420 | 0.995 | 0.414 to 0.429 |
| rodinia/hotspot | 0.449 | 0.443 | 0.986 | 0.433 to 0.549 |
| rodinia/nn | 0.015 | 0.015 | 1.020 | 0.014 to 0.016 |
| rodinia/lavaMD | 0.619 | 0.632 | 1.021 | 0.614 to 0.642 |
| hecbench/aop-cuda | 0.817 | 0.828 | 1.014 | 0.792 to 0.849 |
| hecbench/atomicPerf-cuda | 0.523 | 0.523 | 0.999 | 0.516 to 0.528 |
| hecbench/bilateral-cuda | 0.517 | 0.521 | 1.006 | 0.515 to 0.522 |
| hecbench/s3d-cuda | 0.246 | 0.249 | 1.012 | 0.224 to 0.251 |
| hecbench/shmembench-cuda | 2.147 | 2.127 | 0.991 | 2.075 to 2.355 |
| hecbench/silu-cuda | 0.249 | 0.250 | 1.005 | 0.243 to 0.252 |
| samples/mergeSort | 3.294 | 3.292 | 1.000 | 3.285 to 3.316 |
| samples/MC_EstimatePiInlineP | 1.500 | 1.499 | 0.999 | 1.463 to 1.516 |
| samples/MC_SingleAsianOptionP | 1.663 | 1.665 | 1.002 | 1.645 to 1.682 |
| samples/dct8x8 | 1.940 | 1.526 | 0.786 | 1.403 to 3.570 |
| samples/histogram | 5.366 | 5.384 | 1.003 | 5.337 to 5.415 |
| samples/radixSortThrust | 14.183 | 14.221 | 1.003 | 14.122 to 14.388 |
| samples/cudaTensorCoreGemm | 5.648 | 5.722 | 1.013 | 5.625 to 5.661 |
| samples/immaTensorCoreGemm | 1.289 | 1.356 | 1.052 | 1.284 to 1.300 |
| samples/warpAggregatedAtomicsCG | 1.037 | 1.037 | 1.000 | 1.031 to 1.120 |
