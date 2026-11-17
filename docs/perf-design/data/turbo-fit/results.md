# Card measurements, 2026-11-17 (one run each, VGPU_THREADS=4, programs from the sp22 corpus built for sm_86)

Every number is measured on the card named in its heading. "x sim" = simulator seconds / turbo seconds. Native = the same binary run natively on that card. The simulator and native columns of different rigs differ by machine. The geometric mean under each table is computed from its rows.

## A. A10G (rig 1), profile nvidia/a10g, THIS BRANCH, turbo FIT=1 (default)

| program | simulator s | turbo s | launches on card / all | turbo output vs simulator | verify words different / outside (largest abs, rel) | verify output | native s | x sim |
|---|---:|---:|---:|---|---|---|---:|---:|
| polybench/2DCONV | 0.93 | 2.01 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.85 | 0.46x |
| polybench/ATAX | 1.07 | 0.37 | 2 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.33 | 2.91x |
| polybench/FDTD-2D | 92.28 | 4.26 | 1500 / 1500 | identical | 0 / 0 (abs 0, rel 0) | identical | 4.19 | 21.69x |
| polybench/GEMM | 0.71 | 0.58 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.66 | 1.22x |
| polybench/LU | 28.21 | 1.02 | 4094 / 4096 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.94 | 27.61x |
| rodinia/backprop | 0.67 | 0.51 | 2 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.40 | 1.31x |
| rodinia/gaussian | 3.73 | 0.27 | 1022 / 1022 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.30 | 13.70x |
| rodinia/srad_v2 | 0.06 | 0.45 | 0 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.30 | 0.14x |
| rodinia/hotspot | 1.45 | 0.28 | 1000 / 1000 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 5.26x |
| rodinia/bfs | 0.01 | 0.24 | 26 / 26 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 0.06x |
| hecbench/atan2-cuda | 0.03 | 0.24 | 3 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 0.13x |
| hecbench/bilateral-cuda | 1.48 | 0.39 | 3 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.33 | 3.82x |
| hecbench/black-scholes-cuda | 28.16 | 5.93 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 5.28 | 4.75x |
| hecbench/silu-cuda | 0.65 | 0.34 | 50 / 50 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.27 | 1.93x |

Geometric mean x sim 1.79 over 14 programs; launches on the card / all launches: 7705 / 7708.

## B. A10G (rig 2), profile nvidia/a10g, origin/main turbo (before)

| program | simulator s | turbo s | launches on card / all | turbo output vs simulator | verify words different / outside (largest abs, rel) | verify output | native s | x sim |
|---|---:|---:|---:|---|---|---|---:|---:|
| polybench/2DCONV | 0.94 | 0.93 | 0 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.85 | 1.00x |
| polybench/ATAX | 1.08 | 1.07 | 0 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.33 | 1.01x |
| polybench/FDTD-2D | 92.98 | 4.82 | 1494 / 1500 | identical | 0 / 0 (abs 0, rel 0) | identical | 4.19 | 19.30x |
| polybench/GEMM | 0.74 | 0.72 | 0 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.66 | 1.03x |
| polybench/LU | 27.96 | 1.38 | 4083 / 4096 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.94 | 20.30x |
| rodinia/backprop | 0.67 | 0.83 | 1 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.40 | 0.81x |
| rodinia/gaussian | 3.73 | 0.53 | 985 / 1022 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.30 | 7.09x |
| rodinia/srad_v2 | 0.06 | 0.07 | 0 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.30 | 0.96x |
| rodinia/hotspot | 1.34 | 0.56 | 781 / 1000 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 2.39x |
| rodinia/bfs | 0.01 | 0.01 | 0 / 26 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 1.05x |
| hecbench/atan2-cuda | 0.03 | 0.03 | 0 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 1.05x |
| hecbench/bilateral-cuda | 1.48 | 1.48 | 0 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.33 | 1.00x |
| hecbench/black-scholes-cuda | 28.19 | 6.01 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 5.28 | 4.69x |
| hecbench/silu-cuda | 0.65 | 0.59 | 25 / 50 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.27 | 1.11x |

Geometric mean x sim 2.09 over 14 programs; launches on the card / all launches: 7370 / 7708.

## C. A10G (rig 1), profile nvidia/a10g, THIS BRANCH, VGPU_HOST_GPU_FIT=0 (rules kept; sim and turbo only, no verify pass)

| program | simulator s | turbo s | launches on card / all | turbo output vs simulator | verify words different / outside (largest abs, rel) | verify output | native s | x sim |
|---|---:|---:|---:|---|---|---|---:|---:|
| polybench/2DCONV | 0.93 | 0.93 | 0 / 1 | identical | - | - | 0.85 | 1.00x |
| polybench/ATAX | 1.07 | 1.07 | 0 / 2 | identical | - | - | 0.33 | 1.00x |
| polybench/FDTD-2D | 92.65 | 4.74 | 1494 / 1500 | identical | - | - | 4.19 | 19.56x |
| polybench/GEMM | 0.79 | 0.74 | 0 / 1 | identical | - | - | 0.66 | 1.07x |
| polybench/LU | 28.15 | 1.32 | 4083 / 4096 | identical | - | - | 0.94 | 21.33x |
| rodinia/backprop | 0.67 | 0.79 | 1 / 2 | identical | - | - | 0.40 | 0.85x |
| rodinia/gaussian | 3.73 | 0.53 | 985 / 1022 | identical | - | - | 0.30 | 7.03x |
| rodinia/srad_v2 | 0.07 | 0.06 | 0 / 1 | identical | - | - | 0.30 | 1.03x |
| rodinia/hotspot | 1.33 | 0.56 | 780 / 1000 | identical | - | - | 0.24 | 2.39x |
| rodinia/bfs | 0.01 | 0.01 | 0 / 26 | identical | - | - | 0.24 | 1.01x |
| hecbench/atan2-cuda | 0.03 | 0.03 | 0 / 3 | identical | - | - | 0.24 | 0.99x |
| hecbench/bilateral-cuda | 1.47 | 1.48 | 0 / 3 | identical | - | - | 0.33 | 0.99x |
| hecbench/black-scholes-cuda | 28.20 | 6.00 | 1 / 1 | identical | - | - | 5.28 | 4.70x |
| hecbench/silu-cuda | 0.65 | 0.58 | 25 / 50 | identical | - | - | 0.27 | 1.11x |

Geometric mean x sim 2.11 over 14 programs; launches on the card / all launches: 7369 / 7708.

## D. A10G (rig 1), profile nvidia/h100 (sm_90 simulated on an sm_86 card), THIS BRANCH, FIT=1

| program | simulator s | turbo s | launches on card / all | turbo output vs simulator | verify words different / outside (largest abs, rel) | verify output | native s | x sim |
|---|---:|---:|---:|---|---|---|---:|---:|
| polybench/2DCONV | 1.32 | 0.89 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.85 | 1.48x |
| polybench/ATAX | 2.38 | 0.43 | 2 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.33 | 5.58x |
| polybench/FDTD-2D | 197.99 | 4.30 | 1500 / 1500 | identical | 0 / 0 (abs 0, rel 0) | identical | 4.19 | 46.07x |
| polybench/GEMM | 1.41 | 0.53 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.66 | 2.67x |
| polybench/LU | 63.09 | 1.09 | 4094 / 4096 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.94 | 57.90x |
| rodinia/backprop | 1.03 | 0.51 | 2 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.40 | 2.00x |
| rodinia/gaussian | 8.35 | 0.27 | 1022 / 1022 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.30 | 30.86x |
| rodinia/srad_v2 | 0.08 | 0.50 | 0 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.30 | 0.15x |
| rodinia/hotspot | 1.90 | 0.27 | 1000 / 1000 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 7.13x |
| rodinia/bfs | 0.02 | 0.24 | 26 / 26 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 0.08x |
| hecbench/atan2-cuda | 0.04 | 0.25 | 3 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 0.14x |
| hecbench/bilateral-cuda | 0.75 | 0.34 | 3 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.33 | 2.18x |
| hecbench/black-scholes-cuda | 19.46 | 5.95 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 5.28 | 3.27x |
| hecbench/silu-cuda | 0.62 | 0.29 | 50 / 50 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.27 | 2.17x |

Geometric mean x sim 2.61 over 14 programs; launches on the card / all launches: 7705 / 7708.

## E. A10G (rig 2), profile nvidia/h100, origin/main turbo (before: no card path; sim and turbo only)

| program | simulator s | turbo s | launches on card / all | turbo output vs simulator | verify words different / outside (largest abs, rel) | verify output | native s | x sim |
|---|---:|---:|---:|---|---|---|---:|---:|
| polybench/2DCONV | 1.34 | 1.37 | 0 / 1 | identical | - | - | 0.85 | 0.98x |
| polybench/ATAX | 2.45 | 2.42 | 0 / 2 | identical | - | - | 0.33 | 1.01x |
| polybench/FDTD-2D | 191.53 | 198.22 | 0 / 1500 | identical | - | - | 4.19 | 0.97x |
| polybench/GEMM | 1.39 | 1.41 | 0 / 1 | identical | - | - | 0.66 | 0.99x |
| polybench/LU | 62.91 | 63.22 | 0 / 4096 | identical | - | - | 0.94 | 0.99x |
| rodinia/backprop | 1.03 | 1.02 | 0 / 2 | identical | - | - | 0.40 | 1.00x |
| rodinia/gaussian | 8.29 | 8.47 | 0 / 1022 | identical | - | - | 0.30 | 0.98x |
| rodinia/srad_v2 | 0.07 | 0.07 | 0 / 1 | identical | - | - | 0.30 | 1.04x |
| rodinia/hotspot | 1.92 | 1.93 | 0 / 1000 | identical | - | - | 0.24 | 0.99x |
| rodinia/bfs | 0.02 | 0.02 | 0 / 26 | identical | - | - | 0.24 | 1.03x |
| hecbench/atan2-cuda | 0.03 | 0.03 | 0 / 3 | identical | - | - | 0.24 | 1.01x |
| hecbench/bilateral-cuda | 0.75 | 0.74 | 0 / 3 | identical | - | - | 0.33 | 1.00x |
| hecbench/black-scholes-cuda | 19.57 | 19.63 | 0 / 1 | identical | - | - | 5.28 | 1.00x |
| hecbench/silu-cuda | 0.61 | 0.62 | 0 / 50 | identical | - | - | 0.27 | 0.99x |

Geometric mean x sim 1.00 over 14 programs; launches on the card / all launches: 0 / 7708.

## F. L4 (rig 3, sm_89, 58 SMs), profile nvidia/h100, THIS BRANCH, FIT=1

| program | simulator s | turbo s | launches on card / all | turbo output vs simulator | verify words different / outside (largest abs, rel) | verify output | native s | x sim |
|---|---:|---:|---:|---|---|---|---:|---:|
| polybench/2DCONV | 1.06 | 0.73 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.69 | 1.44x |
| polybench/ATAX | 1.89 | 0.35 | 2 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.31 | 5.44x |
| polybench/FDTD-2D | 159.06 | 3.52 | 1500 / 1500 | identical | 0 / 0 (abs 0, rel 0) | identical | 3.45 | 45.17x |
| polybench/GEMM | 1.15 | 0.46 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.52 | 2.51x |
| polybench/LU | 50.94 | 0.78 | 4094 / 4096 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.73 | 65.58x |
| rodinia/backprop | 0.81 | 0.39 | 2 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.34 | 2.06x |
| rodinia/gaussian | 7.41 | 0.25 | 1022 / 1022 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 29.68x |
| rodinia/srad_v2 | 0.06 | 0.41 | 0 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.26 | 0.14x |
| rodinia/hotspot | 1.60 | 0.30 | 1000 / 1000 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.23 | 5.41x |
| rodinia/bfs | 0.02 | 0.23 | 26 / 26 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.23 | 0.07x |
| hecbench/atan2-cuda | 0.03 | 0.23 | 3 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.23 | 0.12x |
| hecbench/bilateral-cuda | 0.61 | 0.31 | 3 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.30 | 1.94x |
| hecbench/black-scholes-cuda | 16.35 | 4.98 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 4.51 | 3.28x |
| hecbench/silu-cuda | 0.52 | 0.27 | 50 / 50 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.26 | 1.92x |

Geometric mean x sim 2.45 over 14 programs; launches on the card / all launches: 7705 / 7708.

## G. L4 (rig 3, sm_89), profile nvidia/a10g (80 SMs simulated on a 58-SM card), THIS BRANCH, FIT=1

| program | simulator s | turbo s | launches on card / all | turbo output vs simulator | verify words different / outside (largest abs, rel) | verify output | native s | x sim |
|---|---:|---:|---:|---|---|---|---:|---:|
| polybench/2DCONV | 0.70 | 0.74 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.69 | 0.95x |
| polybench/ATAX | 0.80 | 0.35 | 2 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.31 | 2.31x |
| polybench/FDTD-2D | 69.87 | 3.49 | 1500 / 1500 | identical | 0 / 0 (abs 0, rel 0) | identical | 3.45 | 20.00x |
| polybench/GEMM | 0.54 | 0.46 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.52 | 1.15x |
| polybench/LU | 20.45 | 0.79 | 4094 / 4096 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.73 | 25.96x |
| rodinia/backprop | 0.53 | 0.39 | 2 / 2 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.34 | 1.34x |
| rodinia/gaussian | 2.95 | 0.25 | 1022 / 1022 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.24 | 11.58x |
| rodinia/srad_v2 | 0.04 | 0.39 | 0 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.26 | 0.10x |
| rodinia/hotspot | 1.19 | 0.26 | 1000 / 1000 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.23 | 4.61x |
| rodinia/bfs | 0.01 | 0.23 | 26 / 26 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.23 | 0.05x |
| hecbench/atan2-cuda | 0.02 | 0.23 | 3 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.23 | 0.10x |
| hecbench/bilateral-cuda | 1.15 | 0.31 | 3 / 3 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.30 | 3.73x |
| hecbench/black-scholes-cuda | 22.67 | 5.03 | 1 / 1 | identical | 0 / 0 (abs 0, rel 0) | identical | 4.51 | 4.50x |
| hecbench/silu-cuda | 0.52 | 0.27 | 50 / 50 | identical | 0 / 0 (abs 0, rel 0) | identical | 0.26 | 1.93x |

Geometric mean x sim 1.69 over 14 programs; launches on the card / all launches: 7705 / 7708.

