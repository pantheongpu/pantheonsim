# Running the pantheon workloads on VirtualGPU

The [pantheongpu](https://github.com/pantheongpu) GPU stress/diagnostics kernels
run **unmodified** on VirtualGPU — the same source that runs on a physical GPU.

## How it works

pantheon kernels use the CUDA **Runtime** API (`cudaMalloc`, `cudaMemcpy`,
`kernel<<<grid, block>>>()`, `cudaDeviceSynchronize`). VirtualGPU ships
`libvgpucudart` — a drop-in `libcudart.so.13` implementing that API plus the
nvcc host-registration ABI (`__cudaRegisterFatBinary`, `__cudaRegisterFunction`,
`__cudaPushCallConfiguration`, `__cudaGetKernel`, `cudaLaunchKernel`). At load
time the shim is placed ahead of NVIDIA's runtime, so the chevron launch lowers
onto our runtime, which extracts the embedded PTX from the fatbin and executes
it on the CPU SIMT engine.

### One requirement: shared cudart

The app must link the **shared** CUDA runtime so the loader can substitute our
library:

```bash
nvcc -cudart shared ...            # instead of the default static cudart
```

The source is untouched — only the runtime link mode differs. (Hosting a
*statically* linked cudart would require NVIDIA's undocumented driver export
tables; that path is future work. See ARCHITECTURE.md.)

## Recipe

```bash
# 1. Build VirtualGPU (libvgpucudart is built when the CUDA toolkit is present)
./scripts/build.sh

# 2. Compile an unmodified pantheon workload against shared cudart
nvcc -O3 -std=c++14 -Ikernels/common -x cu --gpu-architecture=sm_86 \
     -cudart shared kernels/memory_read/memory_read.cpp -o memory_read

# 3. Run it on a virtual GPU — no physical GPU involved
scripts/vgpu-run.sh --gpu nvidia/a10 --vram-mb 64 ./memory_read 0 1 50 --verify
```

Environment (also honored directly):

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_GPU` | virtual GPU profile | `nvidia/h100` |
| `VGPU_VRAM_MB` | cap advertised VRAM so `%`-of-VRAM stress tests run at laptop scale | profile VRAM |
| `VGPU_DEVICE_COUNT` | number of identical virtual devices | 1 |
| `VGPU_TRACE=1` | log every runtime/driver entry point (how to grow coverage) | off |
| `VGPU_QUIET=1` | silence diagnostics | off |

`--vram-mb` matters because several workloads size their allocation as a
percentage of device memory; on a real 24 GB card that is gigabytes of
interpreted threads. Functional behavior is identical; only the working set
shrinks.

## Status (verified against a physical RTX 3060 oracle)

Runs to completion with correct results:

| Workload | Result |
| --- | --- |
| `idle` | runs |
| `memory_read` | **PASS**; fault injection detected at index 1337 (matches 3060) |
| `memory_write` | **PASS** |
| `galpat` | **PASS** (0 gallop errors) |
| `march_test` | **PASS** (0 march errors) |
| `memory_hammer` | runs |
| `atomic_virus` | runs |
| `int_virus` | runs |
| `compute_virus` | runs (fp32 ALU hammer) |

The differential check is the point: on `memory_read --inject_error`, both the
physical 3060 and the virtual A10 report `Verification: FAIL (1 errors)` — and
VirtualGPU reproduces the device-side `printf` diagnostic byte-for-byte
(`Index: 1337 | XOR: 0x0badbeef`).

## Coverage notes

Every unsupported PTX instruction fails loudly, naming the instruction, PTX
line, kernel, and profile — never a silent wrong answer. Growing coverage is
mechanical: run under `VGPU_TRACE=1`, read the `unsupported PTX` message, add
the instruction with a unit test. Kernels using shared memory, warp shuffles,
tensor-core MMA, or SASS-only fatbins are not yet runnable and say so.
