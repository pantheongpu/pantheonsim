# Static census before and after (turbo_census, A10G rig, 2026-11-17)

`turbo_census` is the tool of `tools/turbo-census` (section 17); it calls the runtime's own scan, image choice and register-call bound, so the "after" column is what the runtime does. "Before" is the tool with
`--assert-off` (no `__assertfail`, `BPT.TRAP` or register call) and `--ptx-variables-all` (any variable refuses a PTX module): the rules of PR #693.

```
turbo_census --host 86 --profile 90 [--ptx-variables-all --assert-off] bin90/*        # 78 HeCBench programs built nvcc -arch=compute_90 (PTX only), simulated as nvidia/h100
turbo_census --host 86 --host 90 --profile 86 [--ptx-variables-all --assert-off] ollama/lib/ollama/cuda_v12/libggml-cuda.so      # and cuda_v13 (Ollama 0.40.16, 2026-11-17 release tarball)
```

| input | cell (profile : host card) | kernels | before | after |
|---|---|---:|---:|---:|
| 78 HeCBench programs, sm_90 PTX only | h100 : sm_86 (retarget) | 293 | 0 (0 programs whole) | 176 (60.1 %, 33 programs whole); 77 `.global` and 40 `.const` referenced |
| Ollama cuda_v12 libggml-cuda.so (SASS) | a10g : sm_86 | 7,263 | 5,476 (75.4 %) | 7,263 (100.0 %) |
| same | a10g : sm_90 | 7,263 | 3,823 (52.6 %) | 5,609 (77.2 %); 1,654 left: `ENDCOLLECTIVE` |
| Ollama cuda_v13 libggml-cuda.so (PTX only) | a10g : sm_86 | 7,254 | 0 | 1,479 (20.4 %); 5,775 reference a variable |
| same | a10g : sm_90 | 7,254 | 0 | 228 (3.1 %); 1,251 hold sm_90 cluster forms |

Before the change, the 1,787 refused cuda_v12 kernels were `never:call-through-register` plus `never:bpt/BPT` (1,786) and one `bpt` alone. The tool's own tests (`test_turbo_census`) follow the new rules.
No libtorch was on the rig; the census of section 17 (5,488 `__assertfail` kernels of 21,614) was not repeated.
