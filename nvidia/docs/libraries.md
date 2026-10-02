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

## cuFile: GPUDirect Storage's compatibility mode

GPUDirect Storage moves file data straight between storage and GPU memory by
DMA, through the nvidia-fs kernel driver. A simulated GPU has no nvidia-fs, so
`libcufile.so.0` is what NVIDIA's library becomes without it: compatibility
mode, where a read is a POSIX `pread` staged through host memory into device
memory and a write the reverse. Everything a program sees -- the driver's open
count, staged and running parameters, handle and buffer registration,
`cuFileRead`/`cuFileWrite` into device, pinned, managed or pageable memory, the
batch API, the stream-ordered API, the statistics -- follows NVIDIA's libcufile
from CUDA 13.0 on an RTX 3060 without nvidia-fs, statuses included (a data-path
failure is -1 with the cuFile status in `errno`, as the card answers).
`nvidia/tests/e2e/cufile_paths.cpp` passes against both libraries, the
stream-ordered calls excepted: NVIDIA's blocks in stream memory operations
under WSL, so those are checked on the simulator only.

## nvCOMP: standard bitstreams, coded on the host

An nvCOMP chunk in a standard format is that format's bitstream -- an LZ4
block, raw Snappy, raw DEFLATE, a gzip member, a Zstandard frame -- so
`libnvcomp.so.5` reads each chunk out of device memory, codes it on the host
and writes it back, with codecs of its own written from the formats'
specifications (`nvidia/src/nvcomp_codecs.cpp`). The streams are compatible
both ways, which an RTX 3060 checked: NVIDIA's nvCOMP 5.3 decodes what these
codecs write, and they decode what it writes (`nvidia/tests/e2e/nvcomp_vectors.inc`
keeps streams of both, so CI checks one direction and the card the other).
GDeflate is NVIDIA's own layout of DEFLATE for 32-lane decoding, published as
the Internet-Draft draft-uralsky-gdeflate-00; `nvcomp_gdeflate.cpp` follows it,
and where the draft leaves the layout open the reading is the one that decodes
every stream NVIDIA's library wrote word for word -- and NVIDIA's library
decodes this encoder's streams in every block layout, including the
multi-block, stored and fixed-Huffman ones nvCOMP never writes itself.

The high-level interface -- `nvcomp::LZ4Manager` and its siblings, their
configurations, `create_manager` and `get_compression_format` -- is the C++
classes NVIDIA's headers declare, laid out member for member and vtable slot
for vtable slot (`nvidia/include/vgpu_nvcomp.hpp`), so a program compiled
against either header runs on either library. A manager cuts a buffer into
chunks and writes nvCOMP's container (`NVCOMP_NATIVE`), the bare bitstream
(`RAW`) or the bitstream after its uncompressed size (`WITH_UNCOMPRESSED_SIZE`:
4 bytes for LZ4, 8 for the others). The container's layout is not documented;
it is what NVIDIA's library writes, measured on the card -- a 64-byte header,
the format's `formatSpec.hpp` struct, each chunk's offset and size, the chunks
8-byte aligned -- and each library reads the other's
(`nvidia/tests/e2e/nvcomp_manager.cpp`). The container can also carry
checksums whose algorithm is not public (no standard CRC or hash matches
them): a policy that computes them is refused, and one that verifies them if
present decompresses and reports `nvcompErrorCannotVerifyChecksums`.

The compressed bytes differ from NVIDIA's (another encoder makes other
choices); the decompressed bytes never do. The queries -- alignments, maximum
output sizes, status strings, which options are refused -- answer what nvCOMP
5.3 answered on the card, except that Deflate's maximum output size follows
NVIDIA's to within 8 bytes and the temporary sizes are the simulator's (it
needs none). A buffer too small and a corrupt chunk are
`nvcompErrorCannotDecompress`, as documented; NVIDIA's LZ4 detects neither,
which on the card is a write past the buffer or a fault.

## NVSHMEM: one GPU per process, every heap shared

NVSHMEM runs a job of PEs, each a process with a GPU, and gives each a
symmetric heap the others read and write. In `libnvshmem_host.so.3` each PE's
heap is device memory the simulator backs with a shared file (its CUDA IPC
mechanism), and every PE maps every other PE's heap into its own device
address space: the single-node, all-peer-to-peer case of NVSHMEM, where a
kernel's store to a peer's heap is a store to shared memory. The PEs meet
through a rendezvous file named by the job's unique ID
(`nvshmemx_get_uniqueid` and `NVSHMEMX_INIT_WITH_UNIQUEID`, the bootstrap that
needs no MPI) or, for scripts, `VGPU_NVSHMEM_RANK`, `VGPU_NVSHMEM_NPES` and
`VGPU_NVSHMEM_ID`; `nvshmem_init()` with neither is a job of one PE, as with
NVIDIA's library outside a launcher.

The host API is implemented here: the symmetric heap, blocking, strided,
typed and stream-ordered puts and gets, signals, barriers, teams (strided and
2-D splits, translation, destruction), and the broadcast, fcollect and alltoall
collectives. The device API is NVIDIA's own: the `nvshmem_*` calls in a kernel
are inline functions in NVIDIA's public headers, linked with NVIDIA's
`libnvshmem_device.a` (`-rdc`). They read one struct, `nvshmemi_device_state_d`,
which the device library's init code asks this library to fill in -- heap
bases, the peers' heaps, the team table, the collectives' synchronization
arrays, laid out as the public headers declare them (`nvidia/src/nvshmem_abi.hpp`,
checked field by field against NVIDIA's headers by
`nvidia/tests/e2e/nvshmem_device.cu`). With every peer reachable by load and
store the inline code never leaves the kernel, so puts, gets, `p`/`g`, atomics,
signal operations and waits, `quiet`, `fence` and the barriers -- at thread and
block scope, in teams, and in kernels started with `nvshmemx_collective_launch`
-- run as NVIDIA compiled them. The device library contains SASS only (PTX is
shipped for sm_120 alone), which the simulator's SASS engine runs. Each PE's
heap is memory shared between the processes, and an atomic on memory the host
maps is made with the CPU's own compare-and-swap in both engines, so PEs
adding into one word at once lose no update (`e2e_ipc` races two processes'
kernels on CUDA-IPC memory; `e2e_host_atomics` races a kernel against host
atomics).

Card ground truth is thin: on an RTX 3060 under WSL NVIDIA's library
initializes a job of one PE and then has no symmetric heap (`nvshmem_malloc`
returns NULL), and two 3060s have no peer-to-peer path. What a multi-PE job
computes here follows NVSHMEM's documentation. `e2e_nvshmem_host` runs the host
API in jobs of one and three PEs; `e2e_nvshmem_device` runs the device API in a
job of three, and skips unless NVIDIA's NVSHMEM is installed (`NVSHMEM_HOME`, or
the `nvidia-nvshmem-cu13` pip package), since its headers and device library
are not the simulator's to ship.

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
| cuFile (GPUDirect Storage) | `libcufile.so.0` | compatibility mode: file I/O staged through host memory into device memory, the driver and parameter API, handle and buffer registration, batch and stream-ordered I/O, statistics; NVIDIA's statuses (CUDA 13.0) |
| nvCOMP | `libnvcomp.so.5` | the low-level batched API and the C++ manager API for LZ4, Snappy, Deflate, GDeflate, Gzip and Zstd, chunks and containers interoperable with NVIDIA's in both directions, and CRC32; Cascaded, Bitcomp and ANS refused (no public bitstream) |
| NVSHMEM | `libnvshmem_host.so.3` | the host API across a job of PEs, one simulated GPU per process, bootstrapped by unique ID; the device API of kernels built with NVIDIA's NVSHMEM headers and device library, all PEs peer to peer |
| NVRTC | `libnvrtc.so.13` | compiling CUDA C++ to PTX at run time |
| nvJitLink | `libnvJitLink.so.13` | linking PTX, fatbins, and host objects' and static libraries' device code into one loadable image, with the device linker's rules; the image is PTX (below) |
| nvFatbin | `libnvfatbin.so.13` | writing fatbins at run time -- PTX, cubins, LTO-IR, a host object's relocatable PTX -- that the driver loads |
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

## nvJitLink and nvFatbin: linking stops at PTX

nvJitLink is the device linker as a library. NVIDIA's compiles every input to
SASS and links a cubin; VirtualGPU executes PTX, so its nvJitLink links the
inputs' PTX into one module and hands that module out as the "cubin" -- the
choice NVRTC's shim makes for `nvrtcGetCUBIN`, for the same reason: whatever
the caller does with a cubin (`cuModuleLoadData`, `cuLibraryLoadData`, write
it to a file for `cuModuleLoad`, wrap it with nvFatbin) it can do with PTX
here. `nvJitLinkGetLinkedPtx` returns the same module, without the `-lto -ptx`
NVIDIA's asks for.

The linking is a linker's, measured against NVIDIA's on an RTX 3060: a
symbol with external linkage has one definition, a strong one beating a
`.weak` one; a second strong definition is named in the error log and
dropped, the link still succeeding (as NVIDIA's does); an undefined reference
fails the link with `NVJITLINK_ERROR_INTERNAL` and its name; each module's
file-scope names stay its own (two modules may both have a `twice`). The
linked module declares everything before its first use, the way ptxas
insists, so NVIDIA's driver JITs it as readily as VirtualGPU runs it.

Inputs are PTX, a fatbin's PTX (the image the driver would pick for
`-arch`), the device code nvcc puts in a host object's `.nv_fatbin` and
`__nv_relfatbin` sections and in a static library's members, and VirtualGPU's
own cubins, which are PTX. A linked cubin (`nvcc -cubin`) is accepted and adds
nothing, which is how NVIDIA's treats one. Relocatable SASS (`-rdc` or `-dc`
cubins, a fatbin with no PTX) is refused, since linking machine code means
applying its relocations and this links PTX; so is LTO-IR, NVVM bitcode that
only NVIDIA's compiler reads -- both by name in the error log, with what to
add instead. The simulator runs SASS too, but the linked image stays PTX:
the only SASS NVIDIA's linker would carry into its output is relocatable
code, which is what is refused.

nvFatbin needs no GPU at all, so it is the whole library: it writes the
container NVIDIA's writes with `-compress=false`, entry for entry, and
NVIDIA's driver loads it as VirtualGPU's loaders do. It never compresses,
and it takes a VirtualGPU cubin (PTX) as the PTX it is.

`e2e_nvjitlink_paths` and `e2e_nvfatbin_paths` check all of this, and pass
unchanged against NVIDIA's libnvJitLink and libnvfatbin 13.0 on an RTX 3060
-- and with VirtualGPU's two libraries in their place on the same card, whose
driver then runs the linked PTX.

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
- **NCCL**: the network plugin interface, user-defined reduction operators,
  symmetric memory windows, non-blocking communicators.
- **cuFile**: the nvidia-fs (DMA) path itself, RDMA and user-space file system
  handles (`CU_FILE_HANDLE_TYPE_USERSPACE_FS`, refused as
  `CU_FILE_IO_NOT_SUPPORTED`), and the POSIX bounce-buffer pool's
  configuration (accepted, nothing to configure).
- **nvCOMP**: Cascaded, Bitcomp and ANS, whose bitstreams NVIDIA does not
  publish -- every entry point answers `nvcompErrorNotSupported` -- and LZ4's
  bitshuffle option, likewise; the container's checksums (their algorithm is
  not public: computing them is refused, verifying them reports
  `nvcompErrorCannotVerifyChecksums`); the CPU and streaming gzip APIs; and
  the hardware decompression engine (the backend option is accepted;
  everything runs on the host).
- **NVSHMEM**: the MPI and OpenSHMEM bootstraps (refused by name: use the
  unique ID), PEs on more than one node and proxy or network transports, NVLink
  SHARP multicast (`nvshmemx_mc_ptr` is NULL), host-side reductions, the
  device API's own proxy and IBGDA paths (never taken: every PE is a peer).
  NVIDIA's default `NVSHMEM_MAX_TEAMS` is 256; here it is 32 unless
  set, since each team holds synchronization arrays in the symmetric heap.
- **NVRTC**: CUBIN, LTO-IR and OptiX-IR output (SASS and vendor bitcode, neither
  of which VirtualGPU can execute — ask for PTX), precompiled headers, time
  traces.
- **nvJitLink**: relocatable SASS and LTO-IR inputs (`-rdc`/`-dc` cubins, a
  fatbin with no PTX, NVVM bitcode, index files), and so link-time
  optimisation; the cubin it returns is PTX. Code-generation options (`-O`, `-maxrregcount`, `-Xptxas`, ...) are
  accepted and have nothing to act on.
- **nvFatbin**: compression (`-compress` is accepted, nothing is compressed)
  and `nvFatbinAddIndex`, whose index names LTO-IR libraries.
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
