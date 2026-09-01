# cuBLAS on VirtualGPU

`build/shim/libcublas.so.13` implements the cuBLAS API so programs that call
it — which is most numerical and ML code — run on VirtualGPU.

```bash
nvcc -cudart shared app.cu -o app -lcublas
LD_LIBRARY_PATH=build/shim ./app
```

## Why the math runs on the host

cuBLAS is a *library*, not user code. Nothing requires its internals to run on
the simulated device, so these entry points read their operands out of virtual
device memory, do the arithmetic on the host CPU, and write the result back.

That is deliberate, and it is the same boundary a real system draws:
**application kernels are simulated, vendor library calls are implemented.**
It also makes the difference between usable and not. The SIMT interpreter
retires ~10⁸ lane-ops/s; the host CPU does real FLOPs about two orders of
magnitude faster. A model whose matmuls go through cuBLAS therefore spends
almost no time in the interpreter — only its own custom kernels do.

Operands live in, and results return to, the same virtual device memory that
kernels see, so mixing cuBLAS calls with simulated kernels works normally, and
allocation still fails with `cudaErrorMemoryAllocation` when the device is
full.

## Verified against real cuBLAS

`tests/conformance/cublas_gemm.cu` runs the same calls against real cuBLAS on
a physical GPU and against this implementation, then diffs:

| case | result |
| --- | --- |
| `sgemm` NN | bit-identical |
| `sgemm` TT (both transposed) | bit-identical |
| `sgemm` with `beta = 0` (write-only C) | bit-identical |
| `sgemm` with padded leading dimensions | bit-identical |
| `sgemmStridedBatched` | bit-identical |
| `dgemm` | bit-identical |
| `sgemv`, `saxpy`, `sscal`, `sdot`, `snrm2` | agree to ~1e-7 relative |

Every GEMM path is bit-identical to hardware. The level-1 and level-2 routines
differ in the last digit because reductions do not associate the same way on a
GPU as on a CPU — a real GPU does not reproduce its own results bit-for-bit
across architectures either, so the harness compares those with a tolerance
(`tests/conformance/compare_numeric.py`).

## What is implemented

Handles and configuration (`cublasCreate/Destroy`, streams, pointer mode, math
mode, version/properties), `Sgemm`, `Dgemm`, `SgemmStridedBatched`, `GemmEx`
for the all-fp32 and all-fp64 forms, `Sgemv`, `Saxpy`, `Sscal`, `Sdot`,
`Snrm2`.

Not implemented — these return `CUBLAS_STATUS_NOT_SUPPORTED` rather than a
plausible wrong answer: mixed-precision `GemmEx` (f16/bf16/int8 paths),
cuBLASLt, complex types, triangular solves, and the remaining level-1/2/3
routines. Add them the same way the PTX subset grew: hit one, implement it,
prove it against hardware.

## What this does and does not unlock

A framework needs more than cuBLAS. PyTorch also links cuDNN, cuRAND,
cuSPARSE, cuSOLVER and NCCL, none of which exist here, and it ships thousands
of its own kernels that would run on the interpreter. So `import torch` does
not yet find a GPU.

What does work today is the question CI most often needs answered — *will this
fit and will it launch* — since device memory, discovery, and launch limits all
behave like the real device. See docs/telemetry.md and TODO.md.
