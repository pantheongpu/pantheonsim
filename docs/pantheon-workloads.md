# Running the pantheon workloads on VirtualGPU

The [pantheongpu](https://github.com/pantheongpu) GPU stress/diagnostics suite
runs **unmodified** on VirtualGPU — the same sources that run on a physical GPU.

```bash
./scripts/build.sh
./scripts/run-pantheon-workloads.sh          # builds and runs the whole suite
```

## How it works

pantheon kernels use the CUDA **Runtime** API (`cudaMalloc`, `cudaMemcpy`,
`kernel<<<grid, block>>>()`). VirtualGPU ships `libvgpucudart` — a drop-in
`libcudart.so.13` implementing that API plus the nvcc host-registration ABI
(`__cudaRegisterFatBinary`, `__cudaRegisterFunction`,
`__cudaPushCallConfiguration`, `__cudaGetKernel`, `cudaLaunchKernel`). The
chevron launch lowers onto our runtime, which pulls the embedded PTX out of the
fatbin and executes it on the CPU SIMT engine. `libvgpucuda` provides the
driver API (`libcuda.so.1`) for workloads that link `-lcuda`.

Only two things differ from a physical-GPU run, and neither touches the source:

**1. Link the shared CUDA runtime** (`nvcc -cudart shared`) so the loader can
substitute our library. Hosting a *statically* linked cudart would require
NVIDIA's undocumented driver export tables; see ARCHITECTURE.md D6.

**2. Set a CPU-appropriate intensity** using the workloads' own CLI knobs.
These are saturation tests: a GPU retires ~10¹³ ops/s, a CPU interpreter
~2×10⁸. Lowering `--kernel_loops`/`--grid_size` and the VRAM percentage runs
the same code over a smaller working set — it does not change what executes.
The runner defaults to `--kernel_loops 2 --grid_size 4`, 2% of a 64 MB virtual
device, and a 5 s duration, which makes every workload finish in about its
requested duration.

## Environment

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_GPU` | virtual GPU profile | `nvidia/h100` |
| `VGPU_VRAM_MB` | cap advertised VRAM so %-of-VRAM tests run at laptop scale | profile VRAM |
| `VGPU_DEVICE_COUNT` | number of identical virtual devices | 1 |
| `VGPU_TRACE=1` | log every runtime/driver entry point and unmodeled attribute | off |
| `VGPU_QUIET=1` | silence diagnostics | off |

The runner additionally accepts `VGPU_WL_GPU`, `VGPU_WL_VRAM_MB`,
`VGPU_WL_DURATION`, `VGPU_WL_LOOPS`, `VGPU_WL_GRID`, `VGPU_WL_MEMPCT`,
`VGPU_WL_TIMEOUT`.

## Status

**44 of 46 workloads run to completion**, each executing for its full requested
duration. This includes the memory diagnostics (`memory_read`/`write`,
`galpat`, `march_test`, `memory_hammer`, `memory_retention`, `tlb_avalanche`),
the compute/ALU stressors (`compute_virus`, `int_virus`, `fp64_virus`,
`sfu_stress`, `atomic_virus`), the tensor-core kernels (`mma_virus`,
`tensor_virus`, `transformer_virus`, `omni_virus`), the AI-serving workloads
(`llm_prefill`, `llm_decode`, `moe_router`, `fused_attention`,
`quantized_gemm`, `rag_embedding`, `kv_cache_churn`, `speculative_decode`),
and `graph_replay` (real CUDA Graph capture/replay).

The two exceptions are **not CUDA**:

| Workload | Needs | Why it is out of scope |
| --- | --- | --- |
| `rt_virus` | OptiX (`libnvoptix.so.1`) | NVIDIA's ray-tracing library — a separate product with its own pipeline/BVH runtime |
| `media_enc_virus` | NVENC (`libnvidia-encode.so.1`) | NVIDIA's hardware video encoder — fixed-function silicon, not CUDA |

On a CPU-only machine both take their own documented "driver not installed"
path and exit cleanly. On a host that *also* has real NVIDIA driver libraries
(e.g. WSL), they load the real library and then try to reach real hardware
through the virtual device, so the runner skips them.

## Differential validation against real hardware

`memory_read` was validated against a physical RTX 3060 (the intended
characterization oracle). Build the reference with the Makefile's default
(static) cudart and the virtual run with `-cudart shared` — the source is
identical, only the runtime link differs:

| Run | Physical 3060 | Virtual A10 |
| --- | --- | --- |
| clean | `Verification: PASS (0 errors)` | `Verification: PASS (0 errors)` |
| `--inject_error` | `Verification: FAIL (1 errors)` | `Verification: FAIL (1 errors)` |

VirtualGPU also reproduces the device-side `printf` diagnostic byte-for-byte —
identical expected/actual/XOR values, from an independent CPU execution:

```
[SDC FAULT][Memory_READ] Retention/Read Error! Index: 1337 | Exp: 0xe76e5272 | Act: 0xecc3ec9d | XOR: 0x0badbeef
```

One host caveat, unrelated to VirtualGPU: on this WSL machine a `-cudart
shared` binary segfaults when run against the *real* driver (no VirtualGPU
libraries involved). Use the default static-cudart build for the physical
reference run, as above.

## Growing coverage

Every unsupported PTX instruction fails loudly, naming the instruction, PTX
line, kernel, and GPU profile — never a silent wrong answer:

```
[vgpu] cudaLaunchKernel: VirtualGPU error [unsupported-ptx]: unsupported PTX:
  neg.f32 %f34, %f51
at line 67 in kernel '_Z20voltage_droop_kerneliPfi'
  GPU profile: nvidia/a10
```

So extending coverage is mechanical: run under `VGPU_TRACE=1`, read the error,
implement the instruction, add a unit test. That loop is exactly how the
subset grew from vectorAdd to the full suite — shared memory, warp shuffles,
tensor cores, f16/f16x2, transcendentals, and bitfield ops were each added
because a specific workload demanded them.
