Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default, VGPU_HOST_GPU_ATOMICS=1.

| program | sim s | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbo kept in sim: never/atomics/off/exact/safety/small | verify words different / outside (largest) | verify kernels outside | verify output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hecbench/aop-cuda | 1.86 | 0.83 | 2.25x | 291 / 301 | 0.03 / 0.43 | identical | 0/0/0/0/0/0 | 32785 / 0 (abs 1.62e+33, rel 3.46e-05) | 0 | identical |
| hecbench/atomicPerf-cuda | 1.12 | 0.78 | 1.44x | 12 / 18 | 0.00 / 0.53 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/bh-cuda | 0.87 | 0.70 | 1.25x | 1 / 17 | 0.00 / 0.44 | identical | 0/0/0/0/0/0 | 40147 / 12094 (abs 4.17e-07, rel 0.000318) | 2 | identical |

turbo: geometric mean speed-up over sim 1.59x (3 programs).
