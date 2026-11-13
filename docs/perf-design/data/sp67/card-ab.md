# #569 on the A10G (round sp65's rig, idle: the script checked before and after each program that no run of sp65 was on it), 3 interleaved runs per side, wall s
# new = this branch, base = hostgpu-relaxed f55c8e65; verify = VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1, the real driver

| program | new plain | base verify | new verify | outputs equal |
|---|---:|---:|---:|---|
| 2DCONV | 0.941 (0.933-0.941) | 1.378 (1.364-1.382) | 1.330 (1.325-1.331) | yes |
| ATAX | 1.052 (1.050-1.058) | 1.566 (1.520-1.591) | 1.395 (1.394-1.398) | yes |
| 3DCONV | 0.720 (0.718-0.721) | 46.471 (46.330-46.571) | 8.530 (8.526-8.576) | yes |
| BICG | 0.259 (0.258-0.261) | 0.734 (0.733-0.750) | 0.612 (0.605-0.623) | yes |
| GEMVER | 0.647 (0.624-0.712) | 1.263 (1.202-1.347) | 0.982 (0.976-0.990) | yes |
| MVT | 1.354 (1.288-1.356) | 1.859 (1.719-1.914) | 1.591 (1.558-1.710) | yes |
| GESUMMV | 0.382 (0.376-0.384) | 0.819 (0.811-0.822) | 0.773 (0.770-0.794) | yes |
| JACOBI1D | 2.297 (2.240-2.315) | 4.329 (4.328-4.334) | 3.788 (3.750-3.794) | yes |
