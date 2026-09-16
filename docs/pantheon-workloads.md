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

**44 of 46 workloads pass; 0 fail.** Each runs for its full requested duration
(5 s here; 7-8 s for the ones that also allocate a gigabyte or set up peer
access). The runner provisions 2 virtual devices so the multi-GPU workloads
take their real peer-DMA paths. Verified run:

```
44 passed, 0 failed, 2 skipped
all_reduce               PASS  8s  0.0994063 GB/s        (peer DMA, 2 virtual GPUs)
galpat                   PASS  7s  1.22755e+07 gallop-reads/s
graph_replay             PASS  5s  183403 graph-steps/s  (real capture/replay)
march_test               PASS  5s  1.5287e+07 march-ops/s
memory_read              PASS  5s  0.219581 GB/s
mma_virus                PASS  5s  0.00215057 TFLOPS     (tensor cores)
moe_router               PASS  5s  784476 ai-ops/s       (shared memory)
p2p_thrasher             PASS  7s  1.74713 GB/s          (peer DMA)
pcie_bandwidth           PASS  8s  7.23639 GB/s
transformer_virus        PASS  5s  0.00196313 TFLOPS     (wmma)
...                                (full table: run the script)
media_enc_virus          SKIP      needs NVENC (not CUDA; out of scope)
rt_virus                 SKIP      needs OptiX (not CUDA; out of scope)
```

The suite includes the memory diagnostics (`memory_read`/`write`,
`galpat`, `march_test`, `memory_hammer`, `memory_retention`, `tlb_avalanche`),
the compute/ALU stressors (`compute_virus`, `int_virus`, `fp64_virus`,
`sfu_stress`, `atomic_virus`), the tensor-core kernels (`mma_virus`,
`tensor_virus`, `transformer_virus`, `omni_virus`), the AI-serving workloads
(`llm_prefill`, `llm_decode`, `moe_router`, `fused_attention`,
`quantized_gemm`, `rag_embedding`, `kv_cache_churn`, `speculative_decode`),
`graph_replay` (real CUDA Graph capture/replay), and the multi-GPU workloads
`all_reduce` and `p2p_thrasher` (peer copies between two virtual devices).

A few workloads allocate fixed-size buffers rather than a percentage of VRAM
(`pcie_bandwidth` and `p2p_thrasher` each want two 256 MiB buffers), so the
runner gives those a larger virtual device. Configurable virtual VRAM and
device count are product features, not workarounds: `VGPU_VRAM_MB` and
`VGPU_DEVICE_COUNT` let one laptop present whatever rack the test expects.

One workload is **not CUDA** and is out of scope:

| Workload | Needs | Why it is out of scope |
| --- | --- | --- |
| `rt_virus` | OptiX (`libnvoptix.so.1`) | NVIDIA's ray-tracing library — a separate product with its own pipeline/BVH runtime |

On a CPU-only machine it takes its own documented "driver not installed" path
and exits cleanly. On a host that *also* has real NVIDIA driver libraries
(e.g. WSL), it loads the real library and then tries to reach real hardware
through the virtual device, so the runner skips it.

`media_enc_virus` runs. NVENC is a driver component rather than CUDA, but
applications load it by its bare soname, so `libvgpunvenc` stands in as
`libnvidia-encode.so.1`: the documented encode API with a deterministic,
content-derived bitstream. Identical frames encode identically and any changed
pixel changes the output, which is what the workload's golden-bitstream check
relies on. The output is not a decodable video stream. The runner skips the
workload only when the shim was not built.

## In CI

Two workflows run the workloads, at two depths:

| When | What | Where |
| --- | --- | --- |
| Every pull request | `compute_virus`, `int_virus`, `cache_latency` on one A10 | `ci.yml`, job *pantheon workloads* |
| Every merge to main, and daily at 06:23 UTC | every workload, on every NVIDIA profile (13 machines) | `workloads.yml` |

Each NVIDIA profile gets a machine, alternating CUDA 12.0 and 13.0 within each
architecture and with GPU counts above one for the multi-GPU paths:

| CUDA 12.0 | CUDA 13.0 |
| --- | --- |
| 1 x T4 (Turing 7.5) | 1 x L4 (Ada 8.9) |
| 1 x A100 (Ampere 8.0) | 2 x A100 SXM4 40GB (Ampere 8.0) |
| 2 x A10 (Ampere 8.6) | 1 x A10G (Ampere 8.6) |
| 1 x RTX 3060 (Ampere 8.6) | 8 x H100 (Hopper 9.0) |
| 1 x L40S (Ada 8.9) | 4 x H200 (Hopper 9.0) |
| 1 x H100 PCIe (Hopper 9.0) | 1 x GH200 (Grace Hopper 9.0) |
| | 2 x B200 (Blackwell 10.0, needs CUDA 12.8+) |

A *coverage* job fails the pull request that adds a profile without adding it
to the matrix. Each machine builds the workloads with pantheon's own Makefile
inside `vgpu shell`, which reads the architecture off the simulated
`nvidia-smi` exactly as it would on the card.

The run's summary page has one table, a row per workload and a column per
machine, with failures first and the reason for each. While main is failing,
an issue labelled `workloads-failing` stays open and each run comments on it;
the first run that passes everywhere closes it. It also runs daily because
pantheon lives in its own repository: a change there can break a workload
without anything here changing. *Run workflow* on the Actions page takes a
list of workloads and a pantheon ref, to check a pantheon branch before it
merges.

The same run locally, for any machine:

```bash
VGPU_WORKLOADS=all VGPU_WORKLOAD_GPU=nvidia/h100 VGPU_WORKLOAD_COUNT=8 \
VGPU_WORKLOAD_SECONDS=5 VGPU_WORKLOAD_TIMEOUT=300 \
VGPU_WORKLOAD_ARGS="--kernel_loops 2 --warmup_iters 1 --grid_size 4" \
VGPU_WORKLOAD_REPORT=/tmp/wl VGPU_PANTHEON_DIR=../pantheon \
  tests/workloads/run_pantheon_workloads.sh
python3 tests/workloads/workload_matrix.py /tmp/wl
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_WORKLOADS` | `all`, or workload names | the three quick ones |
| `VGPU_WORKLOAD_GPU` | profile; the build follows its architecture | `nvidia/a10` |
| `VGPU_WORKLOAD_COUNT` | GPUs in the machine | 1 |
| `VGPU_WORKLOAD_VRAM_MB` | memory per GPU | 1024 |
| `VGPU_WORKLOAD_SECONDS` | how long each workload runs | 2 |
| `VGPU_WORKLOAD_MEM_PCT` | percent of memory each allocates | 2 |
| `VGPU_WORKLOAD_ARGS` | extra workload flags | none |
| `VGPU_WORKLOAD_TIMEOUT` | seconds before a workload counts as hung | none |
| `VGPU_WORKLOAD_REPORT` | directory for `results.tsv`, `summary.md` and logs | a temporary one |

A result is `PASS` only when the workload exits 0, reports a measurement, and
prints no CUDA or pantheon error. Otherwise it is `FAIL`, `TIMEOUT`, or
`MISSING` when it did not build; `rt_virus` is `SKIP`, and so is
`media_enc_virus` when the NVENC shim was not built.

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
