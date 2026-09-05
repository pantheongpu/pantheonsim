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
| Device profiles | NVIDIA A10/A100/H100/H200/B200 and AMD MI300X/MI325X/MI350X (data-driven YAML; AMD is discovery-only). Values are placeholders pending hardware characterization |
| `vgpu` CLI | `list-gpus`, `info --gpu <id> [--json]`, `demo vectoradd` |
| Virtual VRAM | sparse/lazy backing — a virtual H200 claims 141 GB on a 16 GB host; OOB / use-after-free / double-free / misalignment diagnostics |
| PTX | lexer/parser for a growing subset (see ARCHITECTURE.md); precise `unsupported` errors for the rest |
| Execution | SIMT warp interpreter: 32-lane warps, divergence masks, shared memory, `bar.sync`, warp shuffles/vote, atomics, tensor-core `wmma`, f16/f16x2, deterministic scheduling |
| Driver API | `libvgpucuda.so` + clean-room `vgpu_cuda.h`: init/discovery/context/memory/module/`cuLaunchKernel`, `cuLibrary`/`cuKernel`, `cuGetProcAddress` |
| Runtime API | `libvgpucudart` (drop-in `libcudart.so.13`): the CUDA **Runtime** API + nvcc host-registration ABI, so unmodified nvcc apps run unchanged |
| Fatbin | extracts embedded PTX from nvcc fatbins (uncompressed, zstd and LZ4 — so binaries from CUDA 12 and 13 both work) |
| Multi-GPU | a virtual rack of N devices with disjoint address windows; peer copies and per-device isolation match a real two-GPU machine |
| Vendor libraries | cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE, cuSOLVER, NCCL, NVRTC, NPP and nvJPEG under their real sonames, each differential-tested against NVIDIA's own library on a physical GPU — see [docs/libraries.md](docs/libraries.md) |
| Discovery | live telemetry + NVML; drop-in `nvidia-smi`, `rocm-smi`, `rocm_agent_enumerator`, and `lspci` output — see [docs/telemetry.md](docs/telemetry.md) |
| NVENC | `libnvidia-encode.so.1` with a deterministic content-derived encoder, so video-encode SDC tests run |
| Proof | an nvcc-compiled CUDA program **and** the unmodified pantheon stress kernels run on the CPU; `memory_read` differential-matches a physical RTX 3060 (incl. fault injection + device printf) |

Known limitations (deliberate, documented):

- Kernels must carry **PTX** (embedded, or a fatbin containing PTX; zstd
  fatbins are decompressed). SASS-only fatbins are rejected with a precise error.
- Unmodified apps must link **shared** cudart (`nvcc -cudart shared`) so the
  loader can substitute VirtualGPU's `libcudart.so.13`. The source is untouched;
  hosting a *statically* linked cudart needs NVIDIA's undocumented driver export
  tables and is future work.
- `wmma` fragment layout is VirtualGPU's own (PTX leaves it unspecified) —
  see ARCHITECTURE.md D8. bf16, `cp.async`, `mma.sync`, and textures are not
  implemented. Every gap fails loudly (instruction, PTX line, kernel,
  profile), never silently.
- OptiX (ray tracing) and NVENC (video encode) are separate NVIDIA
  subsystems, not CUDA, and are out of scope.
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

### Simulating a whole machine

```bash
build/bin/vgpu shell        # pick a GPU, CUDA/driver and OS, then get a shell
```

Inside it, `nvidia-smi`, `rocm-smi`, `rocm_agent_enumerator`, `lspci`, `dmesg`,
`uname` and `/proc/driver/nvidia/version` all reflect the machine you asked
for, and CUDA programs run on the CPU engine. See
[docs/machine-simulator.md](docs/machine-simulator.md).

### Running an unmodified CUDA application

```bash
# Build the app from unmodified source against shared cudart, then run it on a
# virtual GPU — no physical GPU involved.
nvcc -cudart shared my_app.cu -o my_app
scripts/vgpu-run.sh --gpu nvidia/h200 ./my_app
```

All 44 CUDA workloads in the pantheon stress/diagnostics suite run this way
unchanged:

```bash
scripts/run-pantheon-workloads.sh        # builds and runs the whole suite
```

See [docs/pantheon-workloads.md](docs/pantheon-workloads.md).

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
| `VGPU_TRACE` | `1` logs every runtime/driver entry point | unset |
| `VGPU_MAX_STEPS` | raises the runaway-kernel step budget; `0` removes it | built-in |
| `VGPU_THREADS` | host threads used to run blocks | auto |
| `VGPU_VRAM_MB` | virtual VRAM size, overriding the profile | profile |
| `VGPU_STRICT` | `1` turns on the checks that catch bugs hardware hides but that real compiler output trips over: integer division by zero, and storing a register nothing has written | unset |
| `VGPU_COUNTERS` | `1` prints exact per-launch performance counters | unset |

`VGPU_COUNTERS` reports what a profiler reports, except that every number is
counted rather than sampled. Instructions and thread-instructions (their ratio
is the average number of lanes doing useful work), divergent branches, memory
by space, atomics and barriers -- and two that are worth the simulator on their
own:

- **Sectors and coalescing.** Memory moves in 32-byte sectors. Every lane's
  address is in hand at the moment of the access, so the sectors a warp touches
  are counted exactly: four for a fully coalesced 32-lane load of 4-byte
  values, up to thirty-two when every lane lands in its own sector. The
  percentage is how much of the traffic was asked for rather than rounded up to.
- **Shared-memory bank conflicts.** Thirty-two banks of four bytes; lanes
  reaching different words in one bank serialize, lanes reaching the same word
  are broadcast and free. The count is the extra passes serialization forces --
  zero for a conflict-free access, thirty-one for a 32-way conflict.

A device derives both by sampling, so its answer moves between runs. These do
not move.

There is no timing model and no cache model here, so there are no cycles, no
stall reasons and no hit rates. Those are the numbers a profiler is mostly
made of, and inventing them would be worse than not having them.

`VGPU_STRICT` is off by default for a reason worth knowing. Both of those
checks find real bugs, and both fire on code that is perfectly correct: ptxas
emits arithmetic whose result is dead, ggml divides by a stride its
configuration does not use, and CUB stores an undefined register into the
unused part of a shared tile. A check that also rejects working programs
cannot be the default, so it is a mode you turn on when you are hunting.

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
