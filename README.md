# pantheonsim / VirtualGPU

**Functional GPU emulation on CPUs, for development and CI.**

VirtualGPU lets software that expects a GPU run on machines that have none: it
emulates GPU devices, the driver API surface, and (growing) functional kernel
execution — *"test against a rack of GPUs from your laptop."*

```
$ ./build/vgpu info --gpu nvidia/h100        # a virtual H100, no hardware needed
$ VGPU_GPU=nvidia/b200 ./my_driver_api_app   # same binary, now sees a B200
```

**What VirtualGPU is NOT:**

- It does **not** predict performance. No timing, bandwidth, cache, occupancy,
  or tensor-core modeling — ever. Only functional behavior.
- Passing on VirtualGPU does **not** replace final validation on physical
  GPUs. The intended CI model is:

  ```
  every commit:        VirtualGPU CPU tests
  nightly/pre-release: physical GPU integration tests
  ```

## Status (early — NVIDIA MVP)

Working today, all CPU-only:

| Area | State |
| --- | --- |
| Device profiles | A10, A100, H100, H200, B200 (data-driven YAML, values are placeholders pending hardware characterization) |
| `vgpu` CLI | `list-gpus`, `info --gpu <id> [--json]`, `demo vectoradd` |
| Virtual VRAM | sparse/lazy backing — a virtual H200 claims 141 GB on a 16 GB host; OOB / use-after-free / double-free / misalignment diagnostics |
| PTX | lexer/parser for a growing subset (see ARCHITECTURE.md); precise `unsupported` errors for the rest |
| Execution | SIMT warp interpreter: 32-lane warps, divergence masks, `bar.sync` across warps, deterministic scheduling |
| Driver API | `libvgpucuda.so` + clean-room `vgpu_cuda.h`: init/discovery/context/memory/module/`cuLaunchKernel`, `cuLibrary`/`cuKernel`, `cuGetProcAddress` |
| Runtime API | `libvgpucudart` (drop-in `libcudart.so.13`): the CUDA **Runtime** API + nvcc host-registration ABI, so unmodified nvcc apps run unchanged |
| Fatbin | extracts embedded PTX from nvcc fatbins (uncompressed + zstd) |
| Proof | an nvcc-compiled CUDA program **and** the unmodified pantheon stress kernels run on the CPU; `memory_read` differential-matches a physical RTX 3060 (incl. fault injection + device printf) |

Known limitations (deliberate, documented):

- Kernels must carry **PTX** (embedded, or a fatbin containing PTX; zstd
  fatbins are decompressed). SASS-only fatbins are rejected with a precise error.
- Unmodified apps must link **shared** cudart (`nvcc -cudart shared`) so the
  loader can substitute VirtualGPU's `libcudart.so.13`. The source is untouched;
  hosting a *statically* linked cudart needs NVIDIA's undocumented driver export
  tables and is future work.
- No shared memory, warp shuffles, tensor cores, or f16/bf16 math yet. Every
  gap fails loudly (instruction, PTX line, kernel, profile), never silently.
- AMD (MI300X/MI325X/MI350X) is designed for but not started.

## Build & test

Requirements: Linux, CMake ≥ 3.20, a C++20 compiler (gcc 13+ / clang 17+).
No GPU, no CUDA toolkit, no third-party libraries.

```
./scripts/build.sh
./scripts/test.sh
```

Then:

```
./build/vgpu list-gpus
./build/vgpu info --gpu nvidia/h200
./build/vgpu demo vectoradd --gpu nvidia/h100 -n 1000000
```

### Running an unmodified CUDA application

```bash
# Build the app from unmodified source against shared cudart, then run it on a
# virtual GPU — no physical GPU involved.
nvcc -cudart shared my_app.cu -o my_app
scripts/vgpu-run.sh --gpu nvidia/h200 ./my_app
```

The pantheon stress/diagnostics kernels run this way unchanged — see
[docs/pantheon-workloads.md](docs/pantheon-workloads.md).

### Running a driver-API program against the virtual GPU

```c
#include <vgpu_cuda.h>   /* clean-room subset header, in include/ */
/* cuInit, cuDeviceGet, cuCtxCreate, cuMemAlloc, cuMemcpyHtoD,
   cuModuleLoadData(ptx_text), cuModuleGetFunction, cuLaunchKernel, ... */
```

```
cc app.c -I<repo>/include -L<repo>/build -lvgpucuda -o app
VGPU_GPU=nvidia/h200 VGPU_DEVICE_COUNT=4 ./app
```

Environment knobs:

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_GPU` | virtual GPU profile id | `nvidia/h100` |
| `VGPU_DEVICE_COUNT` | number of identical virtual devices | `1` |
| `VGPU_QUIET` | `1` silences stderr diagnostics | unset |

Errors return documented `CUresult` codes **and** print a rich diagnostic:

```
[vgpu] cuLaunchKernel: VirtualGPU error [out-of-bounds]: device memory read at
0x7fff00003140 is 0 bytes past the end of the 64-byte allocation at 0x7fff00003100
  in kernel 'vecAdd', PTX line 30
  lane 16
  instruction: ld.global.f32 %f1,[%rd8]
  GPU profile: nvidia/h100
```

These diagnostics are a core product feature: VirtualGPU aims to grow into an
ASan/TSan-analogue for GPU software (races, invalid memory, OOM injection),
which real GPUs cannot give you cheaply.

## Roadmap (abridged — see TODO.md)

1. Shared memory, warp shuffles, more PTX → broader kernel coverage
2. Static-cudart hosting (driver export tables) → no `-cudart shared` rebuild
3. `vgpu run` / `vgpu test --matrix` across profiles
4. Hardware characterization + differential fuzzing against physical GPUs
   (oracle machines) → verified profiles, conformance database, compat scores
5. Random/adversarial warp scheduling → race detection
6. AMD frontend (HIP/ROCm, CDNA3/CDNA4)

## License

Apache-2.0 (see LICENSE). The `vgpu_cuda.h` header is a clean-room subset
written from NVIDIA's public driver API documentation; it contains no NVIDIA
code. CUDA is a trademark of NVIDIA Corporation; this project is not
affiliated with or endorsed by NVIDIA or AMD.
