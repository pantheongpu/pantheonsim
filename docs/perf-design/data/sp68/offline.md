# Round sp68: the corpus kernels the square root idiom newly admits in exact mode (no card)

Static verdicts: `hostgpu_idiom_scan` over every sm_86 kernel of the cached PolyBench, Rodinia and HeCBench binaries (539 programs, 2,613 kernels;
`scan-summary.txt`). "newly" = outside the exact set with `VGPU_HOST_GPU_DIV=1` alone, inside with `VGPU_HOST_GPU_SQRT=1` (and DIV where it says so).
Runtime check: each program run on server3 (`offline.sh`, `VGPU_HOST_GPU=exact VGPU_HOST_GPU_SQRT=1 VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_LOG=1
VGPU_HOST_GPU_MAX_MIB=0`, no driver); "plain s" is the whole process with the mode off (Release, `-DVGPU_LTO=OFF`, 4 threads, one run, wall inside the lock).

| program (args) | kernel | needs | runtime verdict | plain s | the kernel in the simulator |
|---|---|---|---|---:|---|
| PolyBench GRAMSCHM | `gramschmidt_kernel1` | SQRT | in the set (1 site) | 120.0 | 2,046 launches, 0.24 s in all (0.1 ms each); kernel3 16.6 s, kernel2 0.19 s; device part 19.6 s |
| PolyBench CORR | `std_kernel` | SQRT + DIV | in the set (1 site) | 39.9 | 1 launch; device part 10.3 s for 4 kernels |
| HeCBench hellinger (`1`) | `hellinger` | SQRT | in the set (6 sites) | 18.5 | 1 launch, most of the run |
| HeCBench knn (`1`) | `cuParallelSqrt` | SQRT | in the set (1 site) | 10.9 | 1 launch (the distance kernel beside it stays out: a bare `MUFU.RCP`) |
| Rodinia nn (`filelist -r 5 -lat 30 -lng 90`) | `euclid` | SQRT | in the set (1 site) | 0.01 | 1 launch, 1 ms |
| HeCBench nn (`... -i 10 -t`) | `nn` | SQRT | in the set (1 site) | 0.01 | 8 launches, 2 ms in all |
| HeCBench asyncAllocation (`200 5`) | `process` | SQRT | refused first for safety: module variables, device calls or device-side launches | 2.4 | - |
| HeCBench attention-paged | `norm_fill_kernel<float>` | SQRT | refused first for safety (same) | 25.8 | - |
| HeCBench ced, sobel | `sobel_kernel` (SQRT + DIV), `sobel_filter` (SQRT, 4 sites) | | not run: their input data is not in the cache | - | - |

HeCBench adam (named by Round sp43) has 4 square root sites, all recognised, and stays out for a call to a subroutine that touches memory (`@P4 CALL`
to an `expf`/`powf` path); Rodinia hotspot's `MUFU.RSQ` is the division slow path's NaN generator, not a square root.
