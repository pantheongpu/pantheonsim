# #569 A/B on server3 (Zen, VGPU_THREADS=4, profile nvidia/a10g, stand-in card STUB_CARD_OK=1 STUB_KEEP_MEMORY=1), 5 interleaved runs per side, wall s
# base = hostgpu-relaxed f55c8e65, new = this branch; plain = no VGPU_HOST_GPU; verify = VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1

| program | base_plain median (min-max) | new_plain median (min-max) | base_verify median (min-max) | new_verify median (min-max) | outputs equal |
|---|---:|---:|---:|---:|---|
| 2DCONV | 0.485 (0.456-0.492) | 0.479 (0.458-0.508) | 0.699 (0.680-0.703) | 0.667 (0.656-0.669) | yes |
| 3DCONV | 0.373 (0.359-0.378) | 0.373 (0.335-0.379) | 39.295 (39.232-39.642) | 6.224 (6.181-6.234) | yes |
| ATAX | 0.523 (0.514-0.527) | 0.522 (0.522-0.530) | 0.757 (0.747-0.770) | 0.638 (0.636-0.679) | yes |
| BICG | 0.167 (0.167-0.173) | 0.168 (0.166-0.169) | 0.393 (0.390-0.405) | 0.280 (0.277-0.285) | yes |
| GESUMMV | 0.239 (0.234-0.240) | 0.239 (0.236-0.241) | 0.465 (0.458-0.475) | 0.426 (0.420-0.427) | yes |
| MVT | 0.653 (0.639-0.669) | 0.653 (0.641-0.676) | 0.876 (0.871-0.950) | 0.768 (0.746-0.818) | yes |
| GEMVER | 0.367 (0.352-0.382) | 0.392 (0.359-0.405) | 0.696 (0.676-0.724) | 0.510 (0.487-0.537) | yes |
| JACOBI1D | 0.456 (0.454-0.463) | 0.461 (0.451-0.466) | 0.682 (0.657-0.690) | 0.662 (0.653-0.665) | yes |

# the other forms, same machine and stand-in, 5 interleaved runs per side: exact and turbo without verify (no launch reached the stand-in: the outputs are the plain ones), turbo with verify

| program | base_exact median (min-max) | new_exact median (min-max) | base_turbo median (min-max) | new_turbo median (min-max) | base_tverify median (min-max) | new_tverify median (min-max) | outputs equal |
|---|---:|---:|---:|---:|---:|---:|---|
| 2DCONV | 0.487 (0.455-0.498) | 0.494 (0.487-0.497) | 0.496 (0.484-0.501) | 0.497 (0.478-0.501) | 1.097 (1.089-1.112) | 1.064 (1.059-1.077) | yes |
| 3DCONV | 0.375 (0.370-0.385) | 0.376 (0.374-0.376) | 0.377 (0.374-0.379) | 0.375 (0.366-0.381) | 39.915 (39.608-40.494) | 6.553 (6.482-6.629) | yes |
| ATAX | 0.523 (0.513-0.553) | 0.525 (0.522-0.547) | 0.521 (0.520-0.527) | 0.521 (0.511-0.524) | 0.756 (0.752-0.767) | 0.644 (0.632-0.651) | yes |
