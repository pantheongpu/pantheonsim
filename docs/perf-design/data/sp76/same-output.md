# Round sp76: the same output, the same counters

`same-output.py`: the build of PR #669 (`old`) against the round's final build (`new`), server3, `VGPU_THREADS=3`, profile `nvidia/a10g`; exact and
turbo mode against the stand-in driver (every launch simulated). stdout without the lines that carry timings; the plain simulator's
`VGPU_COUNTERS_FILE` (one line of JSON a launch, the process id taken out) and `VGPU_MEMO_FINAL_HASH` (the device's memory at exit).

| program | lines compared | off: new = old | exact: new = old = off | turbo: new = old = off | plain simulator: counters, new = old (launches) | plain simulator: memory at exit, new = old |
|---|---:|---|---|---|---|---|
| polybench/3MM | 2 | same | same | same | same (3) | same |
| polybench/GEMM | 2 | same | same | same | same (1) | same |
| polybench/SYRK | 2 | same | same | same | same (1) | same |
| rodinia/backprop | 6 | same | same | same | same (2) | same |
| rodinia/gaussian | 20 | same | same | same | same (1022) | same |
| rodinia/lud | 6 | same | same | same | same (190) | same |
| rodinia/hotspot | 8 | same | same | same | same (1000) | same |
| rodinia/nn | 10 | same | same | same | same (1) | same |
| rodinia/lavaMD | 2 | same | same | same | same (1) | same |
| hecbench/atomicPerf-cuda | 24 | same | same | same | same (18) | same |
| hecbench/radixsort2-cuda | 6 | same | same | same | differs between two runs of one build too (389 of 600 launches old against old, 391 old against new): its keys are random | same |
| hecbench/s3d-cuda | 18 | same | same | same | same (54) | same |
| hecbench/silu-cuda | 65 | same | same | same | same (50) | same |
| samples/mergeSort | 16 | same | same | same | same (49) | same |
| samples/MC_EstimatePiInlineP | 13 | same | same | same | same (2) | same |
| samples/dct8x8 | 28 | same | same | same | same (118) | same |
| samples/histogram | 43 | same | same | same | same (68) | same |
| samples/segmentationTreeThrust | 6 | same | same | same | differs between two runs of one build too (50 of 514 launches old against old, 48 old against new) | same |

The counters column is of a second pass of the script (`--counters-only`): the first hashed the file with its process ids in and called
every file different; with the id out they are the same.
