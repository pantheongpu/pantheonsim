# Vendor libraries on VirtualGPU

VirtualGPU ships its own build of the NVIDIA libraries that CUDA programs link
against. Each is presented under the real soname, so an unmodified program
finds it by putting the shim directory first on the loader path:

```bash
nvcc -cudart shared app.cu -o app -lcublas -lcudnn -lcufft
LD_LIBRARY_PATH=build/shim ./app
```

| library | soname | what it covers |
| --- | --- | --- |
| CUDA driver | `libcuda.so.1` | contexts, modules, memory, launches |
| CUDA runtime | `libcudart.so.13` | the nvcc registration ABI, streams, events |
| NVML | `libnvidia-ml.so.1` | discovery and telemetry (`pynvml`, nvitop) |
| cuBLAS | `libcublas.so.13` | GEMM, GEMV, level‑1 |
| cuBLASLt | `libcublasLt.so.13` | descriptor matmul with ReLU/bias/GELU epilogues |
| cuDNN | `libcudnn.so.9` | convolution, activation, pooling, softmax, batchnorm |
| cuFFT | `libcufft.so.12` | C2C/R2C/C2R in 1‑D, 2‑D and 3‑D, batched |
| cuRAND | `libcurand.so.10` | host-side uniform and normal generation |
| cuSPARSE | `libcusparse.so.12` | CSR/COO SpMV and SpMM, format conversion |
| cuSOLVER | `libcusolver.so.12` | Cholesky, LU, QR, symmetric eigen, SVD |
| NCCL | `libnccl.so.2` | collectives and point-to-point across ranks |
| NVENC | `libnvidia-encode.so.1` | video encode |

## Why the math runs on the host

A vendor library is not user code. Nothing requires its internals to run on the
simulated device, so these entry points read their operands out of virtual
device memory, do the arithmetic on the host CPU, and write the result back.

That is the boundary a real system draws too: **application kernels are
simulated, vendor library calls are implemented.** It is also the difference
between usable and not. The SIMT interpreter retires ~10⁸ lane-ops/s; the host
CPU does real FLOPs about two orders of magnitude faster. A model whose matmuls
go through cuBLAS spends almost no time in the interpreter — only its own
custom kernels do.

Operands live in, and results return to, the same virtual device memory that
kernels see, so mixing library calls with simulated kernels works normally, and
allocation still fails with `cudaErrorMemoryAllocation` when the device is full.

## Verified against real hardware

`tests/conformance/run_conformance.sh` compiles each test once, runs it against
NVIDIA's library on a physical GPU and against VirtualGPU's, and diffs the
output. Anything that differs is a bug in this implementation.

| suite | result on an RTX 3060 |
| --- | --- |
| `cublas_gemm` | every GEMM path bit-identical; level‑1/2 to ~1e‑7 |
| `lt_and_rand` | cuBLASLt bit-identical; cuRAND matches distribution and reseed semantics |
| `cudnn_ops` | all 50 reported values bit-identical |
| `cufft_transforms` | all 14 bit-identical, across composite, prime, 2‑D, 3‑D and both precisions |
| `cusparse_ops` | all 15 bit-identical |
| `cusolver_factorizations` | Cholesky, LU (pivots included), QR and every solve bit-identical; one f32 eigenvalue differs by ~1e‑6 relative |
| `nccl_collectives` | all 24 bit-identical at two ranks on two physical GPUs |

Some of those numbers came out of the hardware rather than the documentation.
cuDNN rejects `CUDNN_ACTIVATION_IDENTITY` from `cudnnActivationForward`, and
cuRAND does *not* rewind its stream when the seed is set again. Both were found
by differential testing and are matched deliberately.

Reduced precision is compared with a tolerance, not bit-for-bit, wherever the
two implementations legitimately differ: these libraries compute in double and
round on the way out, so single-precision results are if anything slightly more
accurate than hardware's. A real GPU does not reproduce its own results
bit-for-bit across architectures either.

## NCCL: a file-backed transport

There is no NVLink here, so NCCL moves bytes through a directory of
memory-mapped files. Each rank publishes its contribution to its own file, the
ranks rendezvous on a shared metadata segment, and every rank computes its own
output from the peers' files.

That is what makes the ordinary deployment work: one rank per *process*, as
`torchrun` and `mpirun` launch it, where the ranks share no address space to
shortcut through. The single-process forms — `ncclCommInitAll`, and one thread
issuing every rank's call inside `ncclGroupStart`/`ncclGroupEnd` — go through
the same path.

```bash
VGPU_NCCL_DIR=/tmp/my-job     # rendezvous directory (default: $TMPDIR/vgpu-nccl-$UID)
VGPU_NCCL_TIMEOUT=300         # seconds before a stalled collective gives up
```

A collective that never matches up times out with a diagnosis rather than
hanging, because an unexplained hang is the worst way to learn that two ranks
called different collectives. The rank ceiling is 64.

`tests/e2e/run_nccl_multiproc.sh` runs 4 genuinely separate processes;
`tests/e2e/run_nccl_group.sh` runs the single-process grouped form over 4
virtual devices.

## What is not implemented

Unimplemented entry points return the library's own "not supported" status
rather than a plausible wrong answer, so a caller's fallback path still works.

- **cuBLAS**: mixed-precision `GemmEx` (f16/bf16/int8), complex types,
  triangular solves, most of level‑2/3.
- **cuDNN**: the graph/backend API of cuDNN 8+, non-NCHW layouts, non-float
  types, and every backward pass.
- **cuFFT**: multi-dimensional advanced layouts, padded embeds, callbacks,
  cuFFTXt multi-GPU.
- **cuSPARSE**: SpGEMM, SpSV/SpSM, the legacy `cusparse<t>csrmv` family
  (removed by NVIDIA in CUDA 12), blocked formats.
- **cuSOLVER**: the sparse (`cusolverSp`) and multi-GPU (`cusolverMg`) modules,
  the 64-bit generic API, the Jacobi and randomized variants.
- **NCCL**: the network plugin interface, user-defined reduction operators,
  symmetric memory windows, non-blocking communicators.
- No NPP, nvJPEG, or NVRTC. NVRTC in particular would mean shipping a CUDA C++
  compiler, which is a different project.

Add them the way the PTX subset grew: hit one, implement it, prove it against
hardware.
