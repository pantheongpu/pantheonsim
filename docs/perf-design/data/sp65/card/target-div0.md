Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default, VGPU_HOST_GPU_STREAK=0.

| program | sim s | exactdiv s | x sim | exactdiv card / all launches | exactdiv card s / sim s | exactdiv output |
|---|---:|---:|---:|---:|---:|---:|
| polybench/LU | 31.07 | 11.69 | 2.66x | 1390 / 4096 | - | identical |
| rodinia/gaussian | 4.10 | 0.95 | 4.33x | 495 / 1022 | - | identical |
| rodinia/lud | 3.58 | 2.81 | 1.27x | 95 / 190 | - | identical |
| rodinia/sc_gpu | 1.05 | 0.79 | 1.33x | 1063 / 1432 | - | identical |
| rodinia/hotspot | 1.18 | 1.20 | 0.99x | 0 / 1000 | - | identical |

exactdiv: geometric mean speed-up over sim 1.81x (5 programs).
