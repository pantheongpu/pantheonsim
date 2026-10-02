# Vendor libraries on VirtualGPU

VirtualGPU ships its own build of the NVIDIA libraries that CUDA programs link
against. Each is presented under the real soname, so an unmodified program
finds it by putting the shim directory first on the loader path:

```bash
nvcc -cudart shared app.cu -o app -lcublas -lcudnn -lcufft
LD_LIBRARY_PATH=build/shim ./app
```

Binaries built by any CUDA 12 or 13 toolkit work: the PTX nvcc embeds is LZ4
compressed by CUDA 12 and zstd compressed by CUDA 13, and both are read here.

## nvJPEG: a codec, not a wrapper

There is no JPEG library in this repository to delegate to, so nvJPEG's half of
the work is a baseline codec written against ITU-T T.81: marker parsing,
Huffman decoding, dequantisation, an inverse DCT, chroma upsampling and colour
conversion on the way in; the forward transform, quality-scaled quantisation
and the Annex K Huffman tables on the way out.

Header facts are exact and compared exactly -- component count, chroma
subsampling, per-component dimensions for 4:4:4, 4:2:0 and grayscale files. The
pixels are not bit-identical and cannot be: the standard does not specify the
inverse DCT, so two correct decoders differ by about a count per pixel. The
conformance test compares statistics at a precision that rounding cannot move,
which is the honest meaning of "the same image".

## cuDSS: a sparse direct solver of its own

As with nvJPEG there is nothing to delegate to, so `nvidia/src/cudss_solver.cpp`
is a multifrontal solver: minimum-degree ordering on the quotient graph,
supernodes from the elimination tree, and dense partial factorization of each
front -- LU with threshold partial pivoting, LDL^T (LDL^H) with 1x1 and 2x2
pivots, Cholesky -- where a column with no stable pivot in its front is delayed
to the parent's. It is deterministic and works in double precision whatever the
caller's type.

What the matrix fixes agrees with NVIDIA's library: solutions, residuals, the
inertia (exactly, for SCS's quasi-definite KKT systems), `INFO` for a matrix
passed as positive definite that is not. What the factorization chooses does
not: the permutation (NVIDIA's default is nested dissection), `LU_NNZ`,
`FLOPS`, `DIAG` (each pivot, reported by original row, depends on the order),
and `NPIVOTS`, which counts pivots replaced by the pivot epsilon -- NVIDIA's
library pivots only on the diagonal of each supernode and perturbs a zero
pivot where this one takes a 2x2 pivot or delays the column. Statuses,
defaults, sizes and phase rules were measured on an RTX 3060 where the
documentation leaves them open, and `nvidia/tests/e2e/cudss_*.cpp` pass against
both libraries. Refused with a message: the Schur complement mode, the nested
dissection tree, double-double values, and a matrix distributed across
processes.

| library | soname | what it covers |
| --- | --- | --- |
| CUDA driver | `libcuda.so.1` | contexts, modules, memory, launches |
| CUDA runtime | `libcudart.so.13` | the nvcc registration ABI, streams, events |
| NVML | `libnvidia-ml.so.1` | discovery and telemetry (`pynvml`, nvitop) |
| cuBLAS | `libcublas.so.13` | GEMM (fp32/fp64/fp16/bf16/int8 and complex), GEMV, level‑1, triangular solves, batched LU (`getrfBatched`/`getrsBatched`); for complex also rank-1/rank-k updates, Hermitian products, `trmm`/`trmv`/`trsv` |
| cuBLASLt | `libcublasLt.so.13` | descriptor matmul in fp64/fp32/fp16/bf16/fp8, strided batches, row-major layouts, ReLU/bias/GELU epilogues, FP8 tensor-wise and row-wise scales with amax |
| cuDNN | `libcudnn.so.9` | training and inference in the classic API: convolution forward, backward-data, backward-filter and backward-bias (every algorithm cuDNN lists, fused bias-activation), activation, pooling, softmax, LRN, batch normalization (with its fused add and activation, and as the cuDNN 8 normalization API), dropout, the spatial transformer, CTC loss, im2col, reductions and tensor arithmetic, each in NCHW, NHWC or any strides, in float, double, half (float or half compute) and bfloat16, INT8 convolution in NHWC; the graph API's convolution, matmul, pointwise, reduction, normalization (layer, instance, batch, RMS), pooling and concatenation graphs; RNNs |
| cuFFT | `libcufft.so.12` | C2C/R2C/C2R in 1‑D, 2‑D and 3‑D, batched, in any advanced (strided, padded) layout; the cufftXt plan and exec API, half precision included |
| cuRAND | `libcurand.so.10` | host-side uniform and normal generation; Sobol' direction vectors (Joe and Kuo's, the card's to the bit) and scramble constants |
| cuSPARSE | `libcusparse.so.12` | CSR/CSC/COO SpMV, SpMM (strided batches, fp16/bf16), SpGEMM, SDDMM, SpSV/SpSM, format conversion, CSR to CSC; legacy coo2csr, sorts and csrgeam2. SpMV, SpMM, SDDMM, SpSV/SpSM solves, sparse to dense and CSR to CSC are recorded into a captured CUDA graph and run at each launch |
| cuSOLVER | `libcusolver.so.12` | Cholesky, LU, QR (with `ungqr`/`unmqr` for complex), symmetric and Hermitian eigen, SVD, in real and complex types; the 64-bit X API, Jacobi (gesvdj, syevj, heevj) and batched forms, gesvdaStridedBatched |
| NCCL | `libnccl.so.2` | collectives and point-to-point across ranks |
| cuStateVec (cuQuantum) | `libcustatevec.so.1` | dense and diagonal gates with any controls, controlled index-bit swaps, probabilities, projection and Pauli expectation values: what QuEST's cuQuantum backend calls. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuDSS | `libcudss.so.0` | the sparse direct solver, the whole 0.8 API: LU, LDL^T, LDL^H and Cholesky in every index width, view, base and value type, several right-hand sides, the solve sub-phases, iterative refinement, batches, a factorization or solve captured into a CUDA graph -- and SCS's GPU direct backend. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| NVRTC | `libnvrtc.so.13` | compiling CUDA C++ to PTX at run time |
| NPP | `libnppc.so.13` and ten siblings | image and signal primitives |
| nvJPEG | `libnvjpeg.so.13` | baseline JPEG decode and encode |
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

`nvidia/tests/conformance/run_conformance.sh` compiles each test once, runs it against
NVIDIA's library on a physical GPU and against VirtualGPU's, and diffs the
output. Anything that differs is a bug in this implementation.

| suite | result on an RTX 3060 |
| --- | --- |
| `cublas_gemm` | every GEMM path bit-identical, mixed precision included; level‑1/2 to ~1e‑7 |
| `lt_and_rand` | cuBLASLt bit-identical; cuRAND matches distribution and reseed semantics |
| `cudnn_ops` | all 50 reported values bit-identical |
| `cudnn_backward` | all 174 lines agree to 1e‑6 relative: the three convolution passes (groups, dilation, both modes, NHWC, strided, 3‑D, double), fused bias-activation, activation, pooling, softmax and LRN backward, reductions with every index identical, op-tensor, transforms, dropout's backward pass, NHWC batch normalization with and without a fused add and activation, the cuDNN 8 normalization API, the spatial transformer's grid and sampler both ways, CTC loss from activations and from probabilities, im2col |
| `cudnn_types` | INT8 convolution identical to the integer; float-to-half and -bfloat16 bits identical; half and bfloat16 convolution, activation, pooling, softmax and batch normalization to 2e‑4 |
| `cufft_transforms` | all 14 bit-identical, across composite, prime, 2‑D, 3‑D and both precisions |
| `cusparse_ops` | all 15 bit-identical; CSR to CSC identical in every index and value, both bases, both value types, structure only and with values |
| `cublas_level1` | single and double `axpy`, `scal`, `dot`, `nrm2`, `i?amax` and `tbmv` agree, every index identical, negative increments included |
| `cusolver_factorizations` | Cholesky, LU (pivots included), QR and every solve bit-identical; one f32 eigenvalue differs by ~1e‑6 relative |
| `nccl_collectives` | all 24 bit-identical at two ranks on two physical GPUs |
| `nvrtc_jit` | identical: compile a kernel at run time, load the PTX, launch it, same numbers |
| `npp_ops` | all 48 bit-identical, across arithmetic, logic, conversion, colour, statistics, morphology and resizing |
| `nvjpeg_codec` | all 24 identical: header parsing exactly, pixels to within the IDCT's own tolerance |
| `multi_gpu` | all 13 identical to two physical GPUs |

Four of those suites, `cublas_level1`, `cusparse_ops`, `cudnn_backward` and
`cudnn_types`, also run in CI on every pull request: `e2e_library_goldens`
compiles them against the simulator alone and compares the output with what
the RTX 3060 printed (`nvidia/tests/conformance/golden/`), so a change that
makes a routine disagree with the hardware fails on a runner with no GPU.

The library majors in that table are the ones this machine has; the build
reads each soname off the installed toolkit, so on a CUDA 12 host the same
shims come out as `libcublas.so.12`, `libcufft.so.11` and so on.

Some of those numbers came out of the hardware rather than the documentation.
cuDNN rejects `CUDNN_ACTIVATION_IDENTITY` from `cudnnActivationForward`, and
cuRAND does *not* rewind its stream when the seed is set again. Both were found
by differential testing and are matched deliberately.

cuDNN's training paths were pinned down the same way, on an RTX 3060 with
cuDNN 9.27. Its gradients read the forward pass's input for ReLU, clipped ReLU
(which passes `0 < x <= coef`) and swish, and its output for sigmoid, tanh and
ELU. Max pooling's gradient goes to the first largest input of a window,
recomputed from x; `MAX_DETERMINISTIC` sends it to the first input equal to y.
An LRN window of even size reaches one channel further up than down.
`cudnnSetTensor` takes a float for a half tensor; an INT8 result is the float
`alpha * sum` rounded to nearest even and saturated. The algorithm lists have
8, 6 and 7 entries (the max counts say 10, 8 and 9) with forward `DIRECT` and
backward-filter `WINOGRAD` never runnable; dropout's reserve space is one bit
per element, least significant first, rounded up to whole words; batch
normalization fuses an add and RELU or SWISH forward, RELU alone backward;
CTC's gradient with respect to probabilities (normalization `NONE`) is
`-posterior / y` where some path passes and `y` where none does, and a label
too long for its input costs 0 with its gradient zeroed up to the batch's
longest input. All of that is matched. Where the arithmetic order is the hardware's own choice it is not:
float and double sums are accumulated exactly here and rounded once, a
`TRUE_HALF` convolution accumulates in half in its own order, and the dropout
mask comes from a different generator (kept fraction, scaling, reseeding and
the backward pass are cuDNN's). And this library runs some configurations
NVIDIA's does not -- every algorithm for every shape, dy with gaps in
backward-data and x with gaps in backward-filter, double-precision NHWC
backward-data -- rather than refusing them.

## Checked in CI without hardware

The paths PyTorch takes through these libraries are also covered by
self-checking programs in `nvidia/tests/e2e/`, which compare each result with a
host reference (a direct DFT, a dense product, a residual or reconstruction)
and so run on every pull request with no GPU. `run_lib_check.sh` builds and
runs them; each is a ctest of its own.

| test | covers | torch |
| --- | --- | --- |
| `e2e_fft_layouts` | cufftXt plans, strided and padded layouts of every rank, 2‑D/3‑D C2R, half | `torch.fft` |
| `e2e_solver_paths` | cuSOLVER X API, gesvdj/syevj and their batched forms, gesvdaStridedBatched, batched potrf/potrs; cuBLAS batched LU | `torch.linalg` |
| `e2e_sparse_paths` | coo2csr and the sorts, batched and half SpMM, SpGEMM, csrgeam2, SDDMM, SpSV/SpSM | `torch.sparse` |
| `e2e_complex_paths` | complex cuBLAS (GEMM in every batched form, GEMV, level 1, trsm, batched LU, herk, hemv) and cuSOLVER (LU, Cholesky, QR with ungqr/unmqr, heevd/heevj, gesvd/gesvdj, the X API on complex types) | complex tensors in `torch.linalg`, `@` |
| `e2e_lt_paths` | fp16/bf16 matmul with bias epilogues, strided batches, row-major layouts, FP8 scales and amax | `addmm`, `bmm`, `_scaled_mm` |
| `e2e_dnn_backward` | cuDNN's convolution passes against each other, every backward pass against finite differences, algorithm lists, status codes, dropout, an LSTM's gradients through dropout, LSTMs in half, bfloat16 and double, CTC's gradient | `conv2d`, pooling and activation backward, `nn.LSTM(dropout=)`, `ctc_loss` |
| `e2e_dnn_graph` | cuDNN graphs: conv + bias + ReLU, dgrad + ReLU backward, matmul + bias + GELU, reductions, pointwise forward and backward, layer/RMS/batch norm forward and backward, max and average pooling both ways, asymmetric padding, concatenation | `cudnn_convolution_add_relu`, cudnn-frontend |

The programs were also run against NVIDIA's own libraries on an RTX 3060, so
what they assert is what the real libraries do, not only what these do. Two
things that run turned up: cuSPARSE 13.0's batched CSR SpMM uses the first
matrix's row offsets for every member (13.2 follows the stride, as documented
and as this does), and NVIDIA's SDDMM refuses a NULL buffer even when it asked
for none. The FP8 matmuls need an sm_89 card to compare against; their output
encoding is checked against `cuda_fp8.h`'s conversion, bit for bit.

The same suite runs on a rented multi-GPU machine through
`nvidia/tools/verify-multigpu-cloud.sh`, which builds VirtualGPU there and compares
against that machine's own libraries -- so the results above are not one
laptop's. It has been run on two shapes so far, and all thirteen suites match on both:

| machine | what it adds |
| --- | --- |
| 2x H100 SXM5, CUDA 12.8 | NVLink, and everything the older toolkit does differently -- LZ4 fatbins, the `cudaGetDeviceProperties_v2` spelling, different soname majors |
| 4x H100 SXM5 | four-rank NCCL against NVIDIA's libnccl, and four ranks across four processes |
| 8x A100 80GB SXM4, sm_80 | a second architecture, eight-rank NCCL, and eight ranks across eight processes |

Reduced precision is compared with a tolerance, not bit-for-bit, wherever the
two implementations legitimately differ: these libraries compute in double and
round on the way out, so single-precision results are if anything slightly more
accurate than hardware's. A real GPU does not reproduce its own results
bit-for-bit across architectures either.

## More than one GPU

A virtual rack is `VGPU_DEVICE_COUNT` devices built from one profile. Each owns
a disjoint window of the process address space, because CUDA guarantees unified
virtual addressing: a device pointer is unique process-wide and identifies the
device that owns it. Copies, memsets, frees and address-range lookups all
resolve a device pointer against its owner rather than against whichever device
happens to be current, so a cross-device `cudaMemcpyDeviceToDevice` moves the
bytes it should. Without separate windows two devices hand out the same numeric
address for different memory and that copy silently reads the wrong buffer --
which is what it used to do.

`nvidia/tests/conformance/multi_gpu.cu` compares the semantics against a real
multi-GPU machine: enumeration, per-device allocation and kernels,
`cudaSetDevice` stickiness, allocation isolation, peer copies synchronous and
asynchronous, device-to-device through the generic entry point, and an event on
the destination device.

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

`nvidia/tests/e2e/run_nccl_multiproc.sh` runs 4 genuinely separate processes;
`nvidia/tests/e2e/run_nccl_group.sh` runs the single-process grouped form over 4
virtual devices.

## Mixed precision

`cublasGemmEx` takes fp16 or bf16 operands with a float accumulator, which is
what a tensor core does under `CUBLAS_COMPUTE_32F`, and int8 operands with an
int32 accumulator. Narrow operands are widened to float on the way in and
rounded once on the way out; the conversions come from the toolkit's own
host-callable intrinsics rather than hand-written bit twiddling, which is the
same rule this repository applies to vendor struct layouts and for the same
reason. `cublasGemmStridedBatchedEx` and `cublasHgemm` go through the same
path, and all of it is bit-identical to hardware on operands that the narrow
formats represent exactly.

## Two NPP entry points that do not match, and why they say so

Everything in the NPP subset is bit-identical to hardware except two, and both
are excluded from the conformance comparison rather than quietly claimed:

- **`nppiFilter_32f_C1R`** (general convolution). Probing NVIDIA's
  implementation with delta kernels gives a mask-to-source mapping that aliases
  positions -- kernel elements 1 and 2 read the same source pixel, as do 4, 5, 7
  and 8 -- which is neither a convolution nor a correlation and is not what the
  documentation describes. VirtualGPU implements the documented convolution.
- **`nppiResize` with `NPPI_INTER_LINEAR`**. Fitting the hardware output pixel
  by pixel shows the horizontal axis interpolating at pixel centres while the
  vertical axis samples rows exactly, with no blending at all. Nearest-neighbour
  matches and is compared; bilinear here is the standard filter.

Both are usable and both are documented as approximations. Getting an answer
that is *close* to NVIDIA's is not the same as getting NVIDIA's, and this file
is where the difference is written down.

## NVRTC: the compiler is the compiler

NVRTC turns a string of CUDA C++ into PTX at run time. That is a C++ compiler,
and there is no honest way to fake one — so this shim does not try. It writes
the program out and invokes `nvcc --ptx`, then returns the PTX and hands back
the compiler's diagnostics as the program log. The toolkit is a host-side
dependency that needs no GPU, so this works on the same CPU-only box as
everything else; if nvcc is not on PATH the compile fails with a log saying
exactly that, rather than a mystery `NVRTC_ERROR_COMPILATION`.

That closes the loop for the JIT frameworks — CuPy, Numba, Triton and PyTorch's
inductor all compile through NVRTC and then load the PTX through the driver
API, which VirtualGPU already interprets.

`nvrtcAddNameExpression` / `nvrtcGetLoweredName` work by the same route the
real implementation uses: a device variable is initialised with the
expression's address, and the mangled symbol is read back out of the generated
PTX. That is what turned up two gaps in the PTX parser — a forward-declared
`.entry` prototype, and a global initialised with another symbol's address —
both of which appear in ordinary nvcc output and are now handled.

## What is not implemented

Unimplemented entry points return the library's own "not supported" status
rather than a plausible wrong answer, so a caller's fallback path still works.

- **cuBLAS**: most of real level‑2/3; for complex types `her`/`her2`/`syr`/`syr2`,
  `her2k`/`syr2k`, the `rot` family, `geqrfBatched` and `gelsBatched`.
- **cuBLASLt**: the backward epilogues (`BGRADA`/`BGRADB`, `DRELU`, `DGELU`),
  auxiliary outputs, and the block-scaled FP8/FP4 modes.
- **cuDNN**: in the graph API, every operation but convolution, matmul,
  pointwise, reduction, normalization, pooling and concatenation (attention,
  RNG, reshape, statistics generation, ...), group normalization,
  normalization backward without the saved statistics, interpolating
  resampling (nearest, bilinear) and resampling index tensors, ragged and
  vectorized tensors; in the classic API, the vectorized layouts
  (`NCHW_VECT_C`, INT8x4/INT8x32), FP8 tensors, divisive normalization,
  fused-ops plans, tensor transform descriptors and RNN projections.
- **cuFFT**: callbacks, cuFFTXt's multi-GPU descriptors.
- **cuSPARSE**: the legacy `cusparse<t>csrmv` family (removed by NVIDIA in
  CUDA 12), the blocked (BSR) routines, complex values.
- **cuSOLVER**: the sparse (`cusolverSp`) and multi-GPU (`cusolverMg`) modules,
  the randomized variants, `sytrf` (symmetric indefinite), and `Xgeev` on a
  complex matrix.
- **NCCL**: the network plugin interface, user-defined reduction operators,
  symmetric memory windows, non-blocking communicators.
- **NVRTC**: CUBIN, LTO-IR and OptiX-IR output (SASS and vendor bitcode, neither
  of which VirtualGPU can execute — ask for PTX), precompiled headers, time
  traces.
- **NPP**: a chosen subset -- allocation, per-pixel arithmetic and logic, data
  exchange, colour conversion, thresholding, statistics, box filtering, 3x3
  morphology, mirroring, resizing, and the signal-processing equivalents. The
  rest of NPP's several thousand entry points are absent rather than
  approximated, so a program that needs more fails at link time with a name.
- **nvJPEG**: baseline sequential DCT only. Progressive JPEG, 12-bit samples,
  arithmetic coding and lossless mode are rejected by name; so are the batched
  and device-side decode APIs and the transcoding entry points.

Add them the way the PTX subset grew: hit one, implement it, prove it against
hardware.
