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
`CsyrkEx` and their 3m forms); `geqrfBatched`
is LAPACK's geqrf per matrix. Each routine refuses the arguments the card
refuses, measured: an unknown fill mode, side, operation or diagonal, a
negative size, a zero increment or a short leading dimension in levels 2 and
3, nothing in level 1.

The band and packed level-2 routines (`gbmv`, `sbmv`/`hbmv`, `spmv`/`hpmv`,
`spr`/`hpr`, `spr2`/`hpr2`, `tbmv`, `tbsv`, `tpmv`, `tpsv`, and `tpttr`/
`trttp`) are there in every type each comes in (`cublas_packed.inc`), as are
the batched GEMVs (`gemvBatched` and `gemvStridedBatched` in S, D, C, Z and the
half and bfloat16 forms `HSH`, `HSS`, `TST`, `TSS`), `getriBatched` and
`matinvBatched`, `syrkx` and `herkx`, the complex `dgmm`, the `gemm3m` forms
and the batched `Hgemm`s (`cublas_batched.inc`). Each refuses what an RTX
3060 refuses, measured -- with one exception: the card's batched GEMVs check
only the leading dimension, and run (sometimes hang) with a negative size, an
unknown operation or a zero y increment, which this refuses instead.

Every `_64` (ILP64) form of levels 1, 2 and 3 is there. Those that only widen
`int` to `int64_t` are generated (`nvidia/tools/gen_cublas_wrappers.py`, from
the declarations in NVIDIA's header) onto the 32-bit forms; a size beyond the
range of `int` is NOT_SUPPORTED.

**The legacy API** (`cublas.h`: `cublasInit`, `cublasShutdown`,
`cublasGetError`, `cublasAlloc`, `cublasFree`, `cublasSetKernelStream` and the
handle-less BLAS routines with their `char` options and scalars by value),
which NVIDIA's `libcublas.so.13` still exports. The BLAS routines are
generated by the same tool onto the `_v2` ones, run on one library-wide
handle. As on the card: they run without `cublasInit` (and after
`cublasShutdown`), `cublasGetError` reports the latest call, success
included, and `gemv` reads any `trans` but `N` (or `C`) as `T`.

**cuBLASXt** (`cublas_xt.cpp`): its handle, device selection, block
dimension, pinning mode and CPU-ratio settings, and every routine (`gemm`,
`syrk`, `syr2k`, `syrkx`, `herk`, `her2k`, `herkx`, `symm`, `hemm`, `spmm`,
`trsm`, `trmm`) over host or device memory. GEMM is cut into blockDim tiles
handed round-robin to the selected devices, run one after another; the
other routines run whole on the first device. A handle takes one
`cublasXtDeviceSelect`, as on the card. GEMM also honours the CPU share: with a
Fortran-style routine set by `cublasXtSetCpuRoutine` and a ratio above 0 from
`cublasXtSetCpuRatio`, the last `floor(ratio * d)` rows (columns when n >= m)
of C -- d being the longer of m and n -- are computed by the caller's
routine, handed the caller's own memory, as the card does (found with a probe
routine that records its arguments).

**`cublasUint8gemmBias`** (deprecated, declared without a formula) is
`C = clamp(round((sum (op(A) - A_bias)(op(B) - B_bias) + C_bias) C_mult / 2^C_shift), 0, 255)`,
found by experiment on the card and checked on 10,505 outputs; C is not read.

**Fixed-point emulation of double precision** (`fixed_point_gemm.hpp`): under
strategy `EAGER`, a GemmEx with `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` (or any
double or double-complex GEMM when the handle's math mode has
`CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`) is computed on integer slices of the
operands as NVIDIA's Ozaki-style algorithm does: each row of A and column of B
shares one exponent, elements become 7 + 8 (s - 1)-bit integers cut into s
balanced base-256 digits (as digits of the signed integer), and the digit products with i + j <= s + 1 are
added exactly and scaled. The strategy, the mantissa control, the maximum
bit count, the offset and the bit count pointer all take effect. The slicing was
found by probing an RTX 3060 (14,000 single-element products and random
matrices for every slice count); see `e2e_blas_emulation_paths`.

The handle settings -- atomics mode and SM count target -- are kept and read
back with the card's defaults and refusals, and `cublasLoggerConfigure`, the
logger callback and `cublasXerbla` are there. What remains is listed in [libraries.md](libraries.md#what-is-not-implemented).

## The rest of the stack

cuBLAS was the first of these; cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE,
cuSOLVER and NCCL followed, all drawing the same library-vs-kernel boundary and
all proved the same way. **nvidia/docs/libraries.md** covers the set — what each one
implements, what it deliberately does not, and how each was verified against a
physical GPU.
