# Turbo: assert, trap and unreferenced PTX variables (2026-11-17)

Evidence for [12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) section 20. Card: NVIDIA A10G (sm_86), driver 595.112.07, CUDA 13.0 toolkit, a 4-vCPU AWS rig; nothing here was run on Hopper or Blackwell.

| file | what |
|---|---|
| `native-fail.txt`, `sim-fail.txt`, `turbo-fail.txt` | the same programs (`src/fail.cu`, `fatal.cu`, `hang.cu`) on the real card, in the plain simulator (`VGPU_GPU=nvidia/a10g`) and with `VGPU_HOST_GPU=turbo`: every runtime call and the code it returned after an assert, `__trap()` and `__brkpt()`; a block with a waiting warp |
| `programs.tsv` | 31 HeCBench programs built as sm_90 PTX only, run native (sm_86 build), in the simulator (H100 profile), turbo with the earlier rule, turbo with this one, turbo with `VGPU_HOST_GPU_VERIFY=all`: wall seconds, launches on the card, verify counts. `-` is no value (a form that did not finish) |
| `census.md` | the static census before and after, for the HeCBench PTX builds and Ollama 0.40.16's `libggml-cuda.so` (cuda_v12, cuda_v13) |
| `mutations.txt` | each mutation of the unit tests and of the runtime, the failing checks |
| `src/` | the programs and scripts: `run_admit.sh` (the five forms of a program list), `agg.py` (the table), `run_ollama.sh` (Ollama serve under the shims, one generate), `mut_unit.py`, `mut_stub.py` |

Build and run (an AWS g5.xlarge, Ubuntu 24.04, `nvcc` 13.0): `cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVGPU_LTO=OFF`, targets `vgpucudart vgpu turbo_census test_host_gpu_logic` (and `vgpucuda vgpucublas vgpucublaslt` for Ollama);
programs `make ARCH=compute_90 EXTRA_CFLAGS="-cudart shared -w -Wno-deprecated-gpu-targets"` in each HeCBench directory (and `ARCH=sm_86` for the native column). The program arguments are in `src/programs.list` as used (reduced from HeCBench's defaults so that the simulator finishes).
Caps: 60 s per form (150 s for the first two programs). Wall seconds are one run each on a machine that did nothing else; the card was not shared. No host name, address or user name is in these files.
