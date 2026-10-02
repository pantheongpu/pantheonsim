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

There is no JPEG library in this repository to delegate to, so nvJPEG is a
codec written against ITU-T T.81: baseline, extended sequential and
progressive decoding (spectral selection and successive approximation),
restart markers, single-component and interleaved scans, every chroma
subsampling, grey, and Adobe CMYK/YCCK; baseline and progressive encoding with
standard or optimised (Annex K.2) Huffman tables from RGB, BGR or YCbCr planes
of any subsampling.

Every way the API reaches a decode works: `nvjpegDecode`; the batched API
(`nvjpegDecodeBatchedInitialize`, `nvjpegDecodeBatched`), which
`torchvision.io.decode_jpeg` uses on CUDA; and the decoupled three-phase API
(`nvjpegDecodeJpegHost`, `nvjpegDecodeJpegTransferToDevice`,
`nvjpegDecodeJpegDevice`, `nvjpegDecodeJpeg`) with its JPEG streams, decoder
states, pinned and device buffers and decode parameters (output format, region
of interest, CMYK). Every output format NVIDIA's default backend writes is
written: planar, grey, planar and interleaved RGB/BGR, NV12 (from 4:2:0) and
YUY2 (from 4:2:2).

The decoded pixels are NVIDIA's. Against nvJPEG 13.0 on an RTX 3060, across
twenty test images and every output format, 10 of 5.5 million samples
differed, by one or two counts: the inverse DCT here is single precision with
fused multiply-adds, rounded half up after the level shift, as NVIDIA's
rounds; chroma is upsampled by replication; the colour conversion is NVIDIA's
single-precision one with ties to even; CMYK becomes RGB as NVIDIA's makes it
(C*K/255 with an exact half rounded down). The standard leaves the inverse
DCT's internal rounding open, and those few samples are where NVIDIA's is not
this one's. The edges are the card's too, each measured: a file cut short in
its headers is `NVJPEG_STATUS_INCOMPLETE_BITSTREAM` while one cut short in its
entropy-coded data decodes what is there; NV12 only from 4:2:0, YUY2 only from
4:2:2, CMYK to RGB only with CMYK allowed; the decoupled transfer needs a
device buffer attached; the hardware backend is `NVJPEG_STATUS_ARCH_MISMATCH`
(an RTX 3060 has no JPEG engine; NVIDIA's A100 and H100 do, and the simulator
answers the same on every profile); the batched API's argument checks;
`nvjpegEncodeGetBufferSize`'s bound. `e2e_nvjpeg_paths` checks all of it --
the decodes against the card's output by checksum -- and passes against
NVIDIA's libnvjpeg 13.0 on the card and against this one. An encoded
bitstream is a correct JPEG of its source, not NVIDIA's bytes: two encoders
make different, equally legal choices.

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

## cuTENSOR and cuTensorNet: tensor contractions, and networks of them

NVIDIA's libcutensor and libcutensornet each carry a static CUDA runtime that
cannot reach a simulated driver, so both are written here from the documented
APIs (`nvidia/include/vgpu_cutensor.h`, `vgpu_cutensornet.h`). cuTENSOR
computes on the host like cuBLAS: one loop nest per operation, in double
precision, with each operand rounded first to the precision its compute
descriptor names (half, bfloat16, TF32; single for 3XTF32, 9X16BF and 4X16F;
double for 8XINT8). cuTensorNet is built on that cuTENSOR and on the
simulator's cuSOLVER, as NVIDIA's is on theirs: a network is contracted
pairwise with `cutensorContract` into intermediates carved from the caller's
workspace, one slice at a time; QR and SVD lay the tensor out as a matrix with
`cutensorPermute` and factor it with `cusolverDn?geqrf`/`orgqr` and `gesvd`.

Statuses, attribute sizes and defaults, scalar types, FLOP and byte counts,
the padded layout of a permutation and which type and compute combinations
plan follow NVIDIA's libraries on an RTX 3060 (cuTENSOR 2.8.1, cuTensorNet
2.14, CUDA 13.0) where the documentation leaves them open:
`nvidia/tests/e2e/cutensor_paths.cpp` (440 checks) and `cutensornet_paths.cpp`
(293) pass against both, and the binaries linked against NVIDIA's pass
unchanged on the simulator. Some measured behaviour differs from the
documentation and is followed: `cutensorCreatePlan` requires a plan
preference, a contraction refuses an alignment of 0, CONJ is refused on real
data, and a repeated mode is that operand's diagonal.

What each library chooses for itself is not NVIDIA's. Kernel selection is not
modelled, so every workspace estimate is zero and a plan cache entry records
the problem only (which plans NVIDIA's cache keeps is its own rule). The
contraction path is a greedy pairwise search and slicing cuts whole contracted
modes until the intermediates fit and the minimum slice count is met, where
NVIDIA's hyper-optimizer searches further: paths, slicing, FLOP counts and
workspace sizes differ, the contracted tensor does not. A sliced extent means
what NVIDIA reports -- the extent of the mode within one slice, so a mode of
extent 8 sliced completely shows as 1 and gives 8 slices.

| library | soname | what it covers |
| --- | --- | --- |
| CUDA driver | `libcuda.so.1` | contexts, modules, memory, launches |
| CUDA runtime | `libcudart.so.13` | the nvcc registration ABI, streams, events |
| NVML | `libnvidia-ml.so.1` | discovery and telemetry (`pynvml`, nvitop) |
| cuBLAS | `libcublas.so.13` | GEMM (fp32/fp64/fp16/bf16/int8 and complex) with the Ex forms' type tables and grouped batches, levels 1, 2 and 3 in every type they come in (the plane rotations bit for bit), triangular solves, batched LU (`getrfBatched`/`getrsBatched`), QR (`geqrfBatched`) and least squares (`gelsBatched`); see [cublas.md](cublas.md) |
| cuBLASLt | `libcublasLt.so.13` | descriptor matmul in fp64/fp32/fp16/bf16/fp8, strided batches, row-major layouts, ReLU/bias/GELU epilogues, FP8 tensor-wise and row-wise scales with amax |
| cuDNN | `libcudnn.so.9` | training and inference in the classic API: convolution forward, backward-data, backward-filter and backward-bias (every algorithm cuDNN lists, fused bias-activation), activation, pooling, softmax, LRN, batch normalization (with its fused add and activation, and as the cuDNN 8 normalization API), dropout, the spatial transformer, CTC loss, im2col, reductions and tensor arithmetic, each in NCHW, NHWC or any strides, in float, double, half (float or half compute) and bfloat16, INT8 convolution in NHWC; the graph API's convolution, matmul, pointwise, reduction, normalization (layer, instance, batch, RMS), pooling and concatenation graphs; RNNs |
| cuFFT | `libcufft.so.12` | C2C/R2C/C2R in 1‑D, 2‑D and 3‑D, batched, in any advanced (strided, padded) layout; the cufftXt plan and exec API, half precision included |
| cuRAND | `libcurand.so.10` | host-side uniform and normal generation; Sobol' direction vectors (Joe and Kuo's, the card's to the bit) and scramble constants |
| cuSPARSE | `libcusparse.so.12` | CSR/CSC/COO/BSR SpMV, SpMM (strided batches, fp16/bf16), SpGEMM, SDDMM, SpSV/SpSM, format conversion, CSR to CSC, in real and complex values (A, A^T and A^H); legacy coo2csr, sorts, csrgeam2, the BSR family (bsrmv, bsrxmv, bsrmm, bsrsv2, bsrsm2, bsric02, bsrilu02, CSR to BSR and back, general blocks too), csric02 and csrilu02. SpMV, SpMM, SDDMM, SpSV/SpSM solves, sparse to dense and CSR to CSC are recorded into a captured CUDA graph and run at each launch |
| cuSOLVER | `libcusolver.so.12` | Cholesky, LU, QR (with `ungqr`/`unmqr` for complex), symmetric and Hermitian eigen, SVD, in real and complex types; symmetric indefinite (Bunch-Kaufman `sytrf`, `Xsytrs`, `sytri`), `laswp`; the 64-bit X API, `Xgeev` on real and complex matrices, Jacobi (gesvdj, syevj, heevj) and batched forms, gesvdaStridedBatched. The sparse module, cusolverSp: `csrlsvlu`/`csrlsvqr`/`csrlsvchol` (host and device), `csrlsqvqr`, `csreigvsi`, `csreigs`, the reorderings, `csrperm`, `csrzfd`, batched QR |
| cusolverMg | `libcusolverMg.so.12` | getrf/getrs, potrf/potrs/potri and syevd on a matrix spread over several devices in NVIDIA's column-block-cyclic layout |
| NCCL | `libnccl.so.2` | collectives and point-to-point across ranks |
| cuStateVec (cuQuantum) | `libcustatevec.so.1` | dense and diagonal gates with any controls, controlled index-bit swaps, probabilities, projection and Pauli expectation values: what QuEST's cuQuantum backend calls. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuDSS | `libcudss.so.0` | the sparse direct solver, the whole 0.8 API: LU, LDL^T, LDL^H and Cholesky in every index width, view, base and value type, several right-hand sides, the solve sub-phases, iterative refinement, batches, a factorization or solve captured into a CUDA graph -- and SCS's GPU direct backend. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuTENSOR | `libcutensor.so.2` | the 2.x API: contractions and trinary contractions in every type and compute combination an RTX 3060 plans (R16F, R16BF, R32F, C32F, R64F, C64F, R64F x C64F; 16F to 8XINT8), permutations with type conversion and padding, elementwise binary and trinary operations with every unary and binary operator, reductions (ADD, MUL, MAX, MIN), plan preferences, the plan cache and its file, workspace estimation, every execute call captured into a CUDA graph. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuTensorNet (cuQuantum) | `libcutensornet.so.2` | what cuQuantum Python's tensor-network contraction calls: networks built tensor by tensor (and the older descriptor and plan API), the contraction optimizer (a greedy path; slicing to a workspace limit and a minimum slice count) with its configuration and information, packed infos, workspace sizing, slice groups, conjugated inputs and hyperedges; QR, SVD (every truncation, normalization and partition) and gate splitting on cuSOLVER. Built on the simulator's cuTENSOR and cuSOLVER |
| NVRTC | `libnvrtc.so.13` | compiling CUDA C++ to PTX at run time |
| nvJitLink | `libnvJitLink.so.13` | linking PTX, or relocatable SASS, from cubins, fatbins, and host objects' and static libraries' device code into one loadable image, with the device linker's rules (below) |
| nvFatbin | `libnvfatbin.so.13` | writing fatbins at run time -- PTX, cubins, LTO-IR, a host object's relocatable PTX -- that the driver loads |
| NPP | `libnppc.so.13` and ten siblings | image and signal primitives: arithmetic, logic and shifts, colour conversion, gamma and Bayer demosaicing, statistics, histograms and integral images, box, rank and morphological filters, gradients and Canny, affine and perspective warps, rotation, remapping, resizing and mirroring -- every entry point OpenCV, DALI, FFmpeg and the CUDA Samples call but four (below) |
| nvJPEG | `libnvjpeg.so.13` | JPEG decode (baseline, progressive, CMYK; single, batched and decoupled APIs) and encode (baseline, progressive) |
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
| `npp_imgproc` | 289 results: every integer image identical (a dozen near-ties marked approximate, within a count on a pixel or two), floats to 1e‑5 -- warps, rotation, remapping, ResizeSqrPixel, mirroring, logic and shifts, alpha compositing, gamma, demosaicing, lookup, statistics, histograms, integral images, rank and morphological filters, Prewitt gradients and Canny |
| `nvjpeg_codec` | all 24 identical: header parsing exactly, pixels to within the IDCT's own tolerance |
| `multi_gpu` | all 13 identical to two physical GPUs |

Five of those suites, `cublas_level1`, `cusparse_ops`, `cudnn_backward`,
`cudnn_types` and `npp_imgproc`, also run in CI on every pull request: `e2e_library_goldens`
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
| `e2e_solver_sparse_paths` | cusolverSp: LU, QR and Cholesky solves in S/D/C/Z with every reorder, singularity, least squares, shift-inverse eigenvalues, reorderings, permutations, batched QR | `scipy`-style sparse solves |
| `e2e_solver_mg_paths` | cusolverMg on two devices: getrf/getrs, potrf/potrs/potri, syevd, IPIV's layout | multi-GPU dense solvers |
| `e2e_solver_sytrf_paths` | sytrf + Xsytrs and sytri in S/D/C/Z, both triangles, 2x2 pivots, singular D; laswp; Xgeev on complex matrices | `torch.linalg.ldl_factor`, complex `eig` |
| `e2e_sparse_paths` | coo2csr and the sorts, batched and half SpMM, SpGEMM, csrgeam2, SDDMM, SpSV/SpSM | `torch.sparse` |
| `e2e_sparse_complex_paths` | SpMV, SpMM, SDDMM, SpSV/SpSM, SpGEMM, conversions and csrgeam2 on complex values, every op; the type combinations and conjugate transposes NVIDIA's refuses | complex `torch.sparse` |
| `e2e_sparse_bsr_paths` | generic BSR (SpMV, SpMM, SDDMM) and the legacy BSR family: bsrmv/bsrxmv/bsrmm, bsrsv2/bsrsm2 with their zero pivots, bsric02/bsrilu02 (and csric02/csrilu02) with ILU's boost, CSR to BSR and back | preconditioned iterative solvers |
| `e2e_complex_paths` | complex cuBLAS (GEMM in every batched form, GEMV, level 1, trsm, batched LU, herk, hemv) and cuSOLVER (LU, Cholesky, QR with ungqr/unmqr, heevd/heevj, gesvd/gesvdj, the X API on complex types) | complex tensors in `torch.linalg`, `@` |
| `e2e_lt_paths` | fp16/bf16 matmul with bias epilogues, strided batches, row-major layouts, FP8 scales and amax | `addmm`, `bmm`, `_scaled_mm` |
| `e2e_dnn_backward` | cuDNN's convolution passes against each other, every backward pass against finite differences, algorithm lists, status codes, dropout, an LSTM's gradients through dropout, LSTMs in half, bfloat16 and double, CTC's gradient | `conv2d`, pooling and activation backward, `nn.LSTM(dropout=)`, `ctc_loss` |
| `e2e_dnn_graph` | cuDNN graphs: conv + bias + ReLU, dgrad + ReLU backward, matmul + bias + GELU, reductions, pointwise forward and backward, layer/RMS/batch norm forward and backward, max and average pooling both ways, asymmetric padding, concatenation | `cudnn_convolution_add_relu`, cudnn-frontend |

The programs were also run against NVIDIA's own libraries on an RTX 3060, so
what they assert is what the real libraries do, not only what these do. Three
things that run turned up: cuSPARSE 13.0's batched CSR SpMM uses the first
matrix's row offsets for every member (13.2 follows the stride, as documented
and as this does), NVIDIA's SDDMM refuses a NULL buffer even when it asked
for none, and it takes a conjugate transpose of a complex operand, which it
does not document, and computes neither A^H B nor anything else with it (this
refuses one with NOT_SUPPORTED). The FP8 matmuls need an sm_89 card to compare against; their output
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

Everything in `npp_api.cpp`'s part of NPP is bit-identical to hardware except
two, and both are excluded from the conformance comparison rather than quietly
claimed (the image-processing half has its own list, below):

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

## NPP: what real programs call

NPP has some ten thousand entry points; which of them matter was settled by
reading the programs that use it -- OpenCV's cudaarithm, cudaimgproc,
cudawarping and cudafilters (and its core), DALI, FFmpeg's `scale_npp`,
jetson-utils, torchvision (which calls none) and the CUDA Samples -- and
collecting every `npp*` name they call: 253 functions. 249 of them are
implemented:

| user | calls | here |
| --- | --- | --- |
| OpenCV | 194 | all: warps (affine, perspective, both directions, every depth and channel count), rotation, mirroring in place and not, the logical and shift operators with constants, magnitude, alpha compositing and premultiplication, gamma, channel swaps, masked and float mean/standard deviation, even and ranged histograms with their level and buffer helpers, rectangle standard deviation, windowed sums, box, max and min filters, dilation and erosion with masks, float thresholds, transpose |
| DALI | 12 | all: `nppiRemap` at every depth it uses, `nppiCFAToRGB` 8- and 16-bit |
| FFmpeg | 3 | all: `nppiResizeSqrPixel_8u_C1R` (nearest, linear, cubic), the YCbCr 4:2:0 plane layouts |
| CUDA Samples | 51 | all but the two in `watershedSegmentationNPP`: Canny, Prewitt gradient vectors, `nppiLUT_Linear`, `nppiCompareC`, border-replicating box filter, constant-border copy, every allocator |
| jetson-utils | 1 | `nppiCFAToRGB_8u_C1C3R` |

What is left, ranked by those users: `nppiSegmentWatershed_8u_C1IR` and
`nppiCompressMarkerLabelsUF_32u_C1IR` (with their buffer-size queries), one
CUDA Sample between them -- both absent, so the sample fails at link time with
the name. Beyond the list, nothing else of NPP's is implemented.

The conventions NPP leaves unwritten were measured on an RTX 3060 against
NPP 13.0 and are recorded at each function in `npp_core.hpp` and
`npp_imgproc.cpp`; `npp_imgproc` (above) pins them. The few places that are
not exact, and are marked so in that test rather than claimed:

- **Cubic sampling** is four-point Lagrange interpolation in single
  precision (an impulse a quarter pixel away gives -0.0547, 0.8203, 0.2734,
  -0.0391). 8-bit and float results match; a 16-bit image has about one pixel
  in a thousand a count apart, a 32-bit integer one a float ulp apart on about
  one in ten.
- **`nppiResizeSqrPixel`** cubic is a different kernel, not one of the Keys or
  Mitchell-Netravali family; Lagrange stands in for it, a few counts away.
  Super-sampling and Lanczos are not implemented (`NPP_INTERPOLATION_ERROR`).
- **`nppiAlphaComp_8u_AC4R`**: every operator's alpha and every colour is
  exact except the non-premultiplied ATOP and XOR colours, within one count.
- **`nppiFilterCannyBorder`**: on NPP 13.0 the high threshold changes nothing
  (an isolated step of magnitude 40 is an edge at thresholds 30 and 32767
  alike); VirtualGPU does the same. With the Sobel kernel, the CUDA Samples'
  parameters and moderate thresholds match pixel for pixel; a threshold down
  in the noise leaves about ten of 1,500 pixels decided differently, and the
  Scharr kernel some 25 -- its gradients are not the plain 3-10-3 ones.
- **`nppiHistogramEven`**: NVIDIA's writes one or more entries past the
  nLevels - 1 the documentation sizes the histogram for; VirtualGPU writes the
  documented ones.
- **`nppiDilate` and `nppiErode` with an off-centre anchor**: where the mask
  reaches past the ROI on the side away from the anchor, NVIDIA's reads the
  ROI's own edge pixels instead of the image beyond; VirtualGPU reads the
  image, as the documentation describes, so those edge pixels can differ.
  Centred anchors (the usual 3x3 with anchor 1,1) match.
- `nppiRemap_16s` linear, and `nppiCFAToRGB` on a tie in its green
  direction, can be a count apart on a pixel.

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

## nvJitLink and nvFatbin: PTX links to PTX, SASS to SASS

nvJitLink is the device linker as a library. NVIDIA's compiles every input to
SASS and links a cubin. VirtualGPU's links what it is given, in kind:

- **PTX**, when every input has PTX: the inputs' PTX becomes one module,
  handed out as the "cubin" -- the choice NVRTC's shim makes for
  `nvrtcGetCUBIN`, for the same reason: whatever the caller does with a cubin
  (`cuModuleLoadData`, `cuLibraryLoadData`, write it to a file for
  `cuModuleLoad`, wrap it with nvFatbin) it can do with PTX here.
  `nvJitLinkGetLinkedPtx` returns the same module, without the `-lto -ptx`
  NVIDIA's asks for.
- **SASS**, when every input has SASS for `-arch` and some input has no PTX
  (`-rdc`/`-dc` cubins, or fatbins, objects and libraries built for SASS
  only): a real linked cubin (`nvidia/src/sass_link.cpp`), which the driver
  runs as SASS. `nvJitLinkGetLinkedPtx` then returns
  `NVJITLINK_ERROR_INVALID_INPUT`, as NVIDIA's does.

The rules are a linker's, measured against NVIDIA's on an RTX 3060: a
symbol with external linkage has one definition, a strong one beating a weak
one; a second strong definition is named in the error log and dropped, the
link still succeeding; an undefined reference fails the link with
`NVJITLINK_ERROR_INTERNAL` and its name (the functions the driver supplies --
`vprintf`, `malloc`, the device runtime's -- excepted); each module's
file-scope names stay its own. A linked PTX module declares everything
before its first use, the way ptxas insists, so NVIDIA's driver JITs it as
readily as VirtualGPU runs it.

The SASS linker was written from `cuobjdump -elf` listings -- section,
symbol and relocation tables and `.nv.info` attributes -- of the relocatable
cubins CUDA 12.0's and 13.0's nvcc write for sm_75 to sm_120 and of what
NVIDIA's libnvJitLink 13.0 links them into, compared field by field at every
relocation; no NVIDIA binary was disassembled. It merges module data and
constant banks, lays out shared memory the way NVIDIA's link does (two
functions' `__shared__` variables share an offset unless some kernel reaches
both; a kernel's own come after; `extern __shared__` starts at the static end
rounded to 16; from sm_90 the driver's reserved 1 KiB is counted in the
kernel's section), applies the relocations the layout fixes (constant-bank
offsets in six instruction encodings, shared offsets), keeps those that need
a load address for the loader, and writes the attributes, call graph,
prototypes and relocation descriptors NVIDIA's driver reads. For sm_75 to
sm_90 its output matches NVIDIA's byte for byte in every code section and
relocation table, and NVIDIA's driver on the card runs it. Two things NVIDIA's
link does that this one does not: drop functions nothing calls, and, for
sm_100 and sm_120, re-finalize code from the "mercury" sections ptxas leaves
beside it (NVIDIA's linked code for those differs in scheduling, not in what
it computes); the code is taken as ptxas wrote it.

Inputs are PTX, a fatbin's PTX or relocatable SASS (the image the driver would
pick for `-arch`), the device code nvcc puts in a host object's `.nv_fatbin`
and `__nv_relfatbin` sections and in a static library's members, relocatable
cubins, and VirtualGPU's own PTX cubins. A linked cubin (`nvcc -cubin`) is
accepted and adds nothing, which is how NVIDIA's treats one; a cubin for an
architecture `-arch` cannot run is refused when it is added. SASS beside PTX
with no SASS is refused by name: NVIDIA's compiles the PTX first, and there
is no compiler here. So is LTO-IR, NVVM bitcode that only NVIDIA's compiler
reads.

nvFatbin needs no GPU at all, so it is the whole library: it writes the
container NVIDIA's writes with `-compress=false`, entry for entry, and
NVIDIA's driver loads it as VirtualGPU's loaders do. It never compresses,
and it takes a VirtualGPU cubin (PTX) as the PTX it is.

`e2e_nvjitlink_paths`, `e2e_nvjitlink_sass` and `e2e_nvfatbin_paths` check
all of this, and pass unchanged against NVIDIA's libnvJitLink and libnvfatbin
13.0 on an RTX 3060 -- and with VirtualGPU's libraries in their place on the
same card, whose driver then runs the linked PTX and SASS.

## What is not implemented

Unimplemented entry points return the library's own "not supported" status
rather than a plausible wrong answer, so a caller's fallback path still works.

- **cuBLAS**: the legacy (pre-`_v2`) API, cuBLASXt, the band and packed level-2
  routines (`gbmv`, `sbmv`, `spmv`, `tpsv` and the rest), the batched GEMVs,
  `getriBatched`/`matinvBatched`, `syrkx`/`herkx`, and most `_64` forms of
  levels 2 and 3 (absent, so a program that needs one fails to load with the
  name).
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
  CUDA 12); the tridiagonal and pentadiagonal solvers (`gtsv2`, `gpsv`), the
  pruning, coloring and `nnz`/`nnz_compress` helpers, `gebsr2gebsr` and
  `gebsr2gebsc`; sliced-ELL and blocked-ELL storage; SDDMM with a conjugate
  transpose (NVIDIA's documents none and computes something else when given
  one). cuSPARSELt is a library of its own and is not provided.
- **cuSOLVER**: the refactorization module (`cusolverRf`), cusolverSp's
  low-level preview API and its `csrlsvlu` on the device (NVIDIA ships only
  the host one), the randomized variants (`Xgesvdr`), left eigenvectors from
  `Xgeev`, and cusolverMg on a submatrix (IA, JA other than 1) or a grid with
  more than one row of devices. cusolverSp's reorderings are correct
  fill-reducing permutations but not NVIDIA's own, and when several columns
  of a Cholesky factorization are independent of one another NVIDIA's names a
  different one in `singularity`. `Xgeev` on a
  complex matrix returns its eigenvalues in NVIDIA's order up to n = 74 (both
  are LAPACK's single-shift QR there); past that NVIDIA's switches algorithm,
  as LAPACK's does, and the order can differ. NVIDIA's `sytri` (CUDA 13.0, RTX
  3060) returns success and leaves A as it was; this one computes the inverse
  the API documents.
- **cuTENSOR**: block-sparse contractions are created and checked but not
  planned (NOT_SUPPORTED; an RTX 3060 cannot plan them either, so there is no
  card to check a kernel against); just-in-time kernels (the JIT mode is
  accepted and changes nothing); the undocumented exports
  (`cutensorCreateComputeDescriptor`, extraction and insertion, ...), which
  answer NOT_SUPPORTED. A permutation whose input has a mode its output lacks
  is planned by NVIDIA's library and writes zeros on an RTX 3060; here its
  plan is refused (NOT_SUPPORTED).
- **cuTensorNet**: the state API (states, network operators, accessors,
  expectations, marginals, samplers, MPS and its projection), gradients,
  distributed execution and the undocumented exports answer NOT_SUPPORTED;
  the SVD algorithms other than `gesvd` (gesvdj, gesvdp, gesvdr) are refused,
  as are decompositions of half-precision tensors and decompositions made
  while the stream is capturing a graph (they read their results back to
  the host). Autotuning has nothing to tune and returns at once; the cache
  workspace is never used; `RUNTIME_EST` is 0, there being no timing model.
  cuQuantum Python's own calls are these (traced on an RTX 3060), but
  cuQuantum Python 26.09 does not start on the simulator yet: the runtime
  module of cuda-bindings 13 that it uses through nvmath-python, and CuPy 14,
  carry a static CUDA runtime that asks the driver for its export table.
- **NCCL**: the network plugin interface, user-defined reduction operators,
  symmetric memory windows, non-blocking communicators.
- **NVRTC**: CUBIN, LTO-IR and OptiX-IR output (SASS and vendor bitcode, neither
  of which VirtualGPU can execute — ask for PTX), precompiled headers, time
  traces.
- **nvJitLink**: LTO-IR inputs (NVVM bitcode, index files) and so link-time
  optimisation; linking SASS with PTX that has no SASS (there is no compiler
  to bring them together); in a SASS link, dropping unreachable functions,
  re-finalizing sm_100/sm_120 code, debug information (`-G`'s sections are
  not kept) and texture/surface references. Code-generation options (`-O`,
  `-maxrregcount`, `-Xptxas`, ...) are accepted and have nothing to act on.
- **nvFatbin**: compression (`-compress` is accepted, nothing is compressed)
  and `nvFatbinAddIndex`, whose index names LTO-IR libraries.
- **NPP**: the functions OpenCV, DALI, FFmpeg, jetson-utils and the CUDA
  Samples call (see "NPP: what real programs call") plus the original subset
  -- allocation, per-pixel arithmetic and logic, data exchange, colour
  conversion, thresholding, statistics, filters, morphology, resizing, and
  the signal-processing equivalents. Not implemented: watershed segmentation
  and marker-label compression (one CUDA Sample), ResizeSqrPixel's
  super-sampling and Lanczos modes, and the rest of NPP's ten thousand entry
  points, which are absent rather than approximated, so a program that needs
  more fails at link time with a name.
- **nvJPEG**: 12-bit samples, arithmetic coding, lossless and hierarchical
  JPEG (refused by name, `NVJPEG_STATUS_JPEG_NOT_SUPPORTED`); the hardware
  backend and what only it does (`nvjpegDecodeBatchedEx`, scaled decodes,
  applying an EXIF orientation, `nvjpegDecodeBatchedParseJpegTables`);
  carrying metadata or Huffman tables from a parsed image into an encode; and
  the transcoding entry points.

Add them the way the PTX subset grew: hit one, implement it, prove it against
hardware.
