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

`nvidia/tests/conformance/cublas_gemm.cu` runs the same calls against real cuBLAS on
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
| `?axpy`, `?scal`, `?dot`, `?nrm2`, `i?amax`, `?tbmv` in single and double, negative increments included (`cublas_level1`) | agree to the harness's tolerance; every index identical |

Every GEMM path is bit-identical to hardware. The level-1 and level-2 routines
differ in the last digit because reductions do not associate the same way on a
GPU as on a CPU — a real GPU does not reproduce its own results bit-for-bit
across architectures either, so the harness compares those with a tolerance
(`nvidia/tests/conformance/compare_numeric.py`).

## What is implemented

Handles and configuration (`cublasCreate/Destroy`, streams, pointer mode, math
mode, version/properties), `Sgemm`, `Dgemm` and their batched and
strided-batched forms, `Sgemv`, and in both single and double
precision `axpy`, `scal`, `dot`, `nrm2`, `i?amax`, the triangular band
product `tbmv`, and `dgmm` (a matrix times a diagonal one, from either side). A negative increment walks the vector from its far end, as
BLAS defines it; `scal`, `nrm2` and `i?amax` do nothing (or return 0) for an
increment that is not positive, which is what the hardware does too.
The level 1 routines also have their 64-bit-index forms (`cublasDnrm2_64` and
so on, CUDA 12), with `i?amax_64` answering in an `int64_t`; a size or stride
beyond the range of `int` is refused.
The typed level-1 routines (`DotEx`, `DotcEx`, `Nrm2Ex`, `AsumEx`,
`I?amaxEx`, `AxpyEx`, `ScalEx`, `CopyEx`, `SwapEx` and their `_64` forms)
take half, bfloat16, single, double and complex elements, with the table of
result and execution types an RTX 3060's cuBLAS accepts. `geam` and the
least-squares `gelsBatched` are there in all four precisions.
The host and device copy helpers `cublasSetVector`, `cublasGetVector`,
`cublasSetMatrix` and `cublasGetMatrix`, and their `Async` forms, are there
too, with increments and leading dimensions on both sides.

The complex (C and Z) routines PyTorch's complex tensors reach are in
`cublas_complex.inc`, and the same templates serve the real level-2 and
level-3 routines (`ger`, `symv`, `trmv`, `trsv`, `symm`, `syrk`, `syr2k`,
`trmm`) and the rank updates `syr`, `syr2`, `her`, `her2` and `her2k` in
every type they come in. The plane rotations (`rot`, `rotg`, `rotm`, `rotmg`,
typed and Ex) match an RTX 3060 bit for bit in real arithmetic, the card's
order of fused operations repeated. `GemmEx` takes exactly the card's table
of operand and compute types — complex, and int8 into float, included — as
do `SgemmEx`, the complex Ex GEMMs (`CgemmEx`, `Cgemm3mEx`, `CherkEx`,
`CsyrkEx` and their 3m forms) and the grouped batched GEMMs; `geqrfBatched`
is LAPACK's geqrf per matrix. Each routine refuses the arguments the card
refuses, measured: an unknown fill mode, side, operation or diagonal, a
negative size, a zero increment or a short leading dimension in levels 2 and
3, nothing in level 1.

Not implemented: the legacy (pre-`_v2`) API, cuBLASXt, the band and packed
level-2 routines (`gbmv`, `sbmv`, `spmv`, `tpsv` and the rest), the batched
GEMVs, `getriBatched`/`matinvBatched`, `syrkx`/`herkx`, and most `_64` forms
of level 2 and 3. These are absent rather than stubbed, so a program that
needs one fails to load with the name it was looking for. Add them the same
way the rest grew: hit one, implement it, prove it against hardware.

## The rest of the stack

cuBLAS was the first of these; cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE,
cuSOLVER and NCCL followed, all drawing the same library-vs-kernel boundary and
all proved the same way. **nvidia/docs/libraries.md** covers the set — what each one
implements, what it deliberately does not, and how each was verified against a
physical GPU.
