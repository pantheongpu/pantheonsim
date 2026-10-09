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

## cuSPARSELt: the card's pruning and compressed layout

`nvidia/src/cusparselt_api.cpp` answers the cuSPARSELt 0.10 API on the host.
Everything an application can observe was measured against NVIDIA's library on
an RTX 3060, and `nvidia/tests/e2e/sparselt_paths.cpp` passes against both:

- the descriptor checks (which refusals are `INVALID_VALUE` and which
  `NOT_SUPPORTED`), attribute defaults and sizes, and the combinations sm_86
  accepts -- fp16, bf16 and tf32 with fp32 compute, int8 with int32 compute
  into int8, int32, fp16 or bf16 when both operands run along K;
- the pruning, value for value: STRIP keeps the two larger magnitudes of each
  group of four (the lower position on a tie); TILE keeps the pattern of
  largest L1 norm in each 4x4 tile, ties broken in an order measured on the
  card (fp32 uses 1:2 groups and 2x2 tiles);
- the compressed matrix: its size and buffer size (formulas fitted to every
  shape of a grid up to 320 x 320), the kept values (fp32 ones carrying the
  tf32 rounding half-unit, as the card stores them) and the 2-bit metadata in
  the card's layout;
- Matmul's rounding: operands rounded to tf32 to nearest (ties away), fp32
  accumulation, round-to-nearest-even into every output type, saturation into
  integers, ReLU's signed zero, GELU (the tanh form), the bias type (D's type,
  float for int8 inputs), alpha and beta vectors, batches and broadcasts.

Where it differs: NVIDIA's metadata layout for 8- and 16-bit values changes at
larger shapes (seen at 256 x 64) and this one keeps the smaller shapes' layout,
so compressed bytes of large matrices differ while products do not; pruning a
group or tile that holds NaN or an infinity is not the card's; 587 pairs of
TILE patterns never tie on their own on the card, so their order here is
unmeasured; `MatmulSearch` runs the product once and keeps the plan's
configuration; NVIDIA's `CompressedSize2` counts one batch until a plan has
used the descriptor, this one always counts them all; the workspace a plan
asks for is the card's for the default split-K and is never used.
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
| cuBLAS | `libcublas.so.13` | GEMM (fp32/fp64/fp16/bf16/int8 and complex) with the Ex forms' type tables and grouped batches, levels 1, 2 and 3 in every type they come in (band and packed storage included; the plane rotations bit for bit), batched GEMV in every type, triangular solves, batched LU (`getrfBatched`/`getrsBatched`/`getriBatched`/`matinvBatched`), QR (`geqrfBatched`) and least squares (`gelsBatched`), the `_64` forms, cuBLASXt over several devices and the legacy (`cublas.h`) API; see [cublas.md](cublas.md) |
| cuBLASLt | `libcublasLt.so.13` | descriptor matmul in fp64/fp32/fp16/bf16/fp8/fp4 and int8 x int8 into int32 (`CUBLAS_COMPUTE_32I`, what `torch._int_mm` calls), strided batches, row-major layouts, bias/ReLU/GELU epilogues with their auxiliary outputs and the backward ones (DRELU, DGELU, bias gradients), FP8 tensor-wise and row-wise scales with amax, and the block-scaled FP8/FP4 modes (documentation-derived) |
| cuDNN | `libcudnn.so.9` | training and inference in the classic API: convolution forward, backward-data, backward-filter and backward-bias (every algorithm cuDNN lists, fused bias-activation), activation, pooling, softmax, LRN, batch normalization (with its fused add and activation, and as the cuDNN 8 normalization API), dropout (cuDNN's own generator, mask for mask, also between an RNN's layers), the spatial transformer, CTC loss, im2col, reductions and tensor arithmetic, each in NCHW, NHWC or any strides, in float, double, half (float or half compute) and bfloat16, INT8 convolution in NHWC and, vectorized, in `NCHW_VECT_C` (INT8x4, INT8x32), divisive normalization, tensor transforms and folding, fused-ops plans (the scale-bias-activation weight gradient among them), LSTM projections and the multi-head attention API; the graph API's convolution, matmul, pointwise, reduction, normalization (layer, instance, batch, RMS, group; backward with or without the saved statistics; batch normalization across the GPUs of one process), pooling (with max pooling's index tensor), concatenation, reshape, transpose, slice, RNG, statistics-generation and softmax graphs, and scaled dot-product attention forward and backward -- the single SDPA operation and cudnn-frontend's composite graph alike, with causal, sliding-window and padding masks, bias, grouped-query heads, dropout, paged K/V caches and ragged (packed) sequences -- over ragged and INT8x4/INT8x32-vectorized tensors; RNNs |
| cuFFT | `libcufft.so.12` | C2C/R2C/C2R in 1‑D, 2‑D and 3‑D, batched, in any advanced (strided, padded) layout; the cufftXt plan and exec API, half precision included; multi-GPU plans (`cufftXtSetGPUs`, `cufftXtMalloc`/`cufftXtMemcpy` descriptors, `cufftXtExecDescriptor*`, `cufftXtQueryPlan`) with each GPU's part on its own simulated device, in NVIDIA's natural, shuffled and 1‑D string orders; LTO callbacks (`cufftXtSetJITCallback`) given as PTX |
| cuRAND | `libcurand.so.10` | host-side uniform and normal generation; Sobol' direction vectors (Joe and Kuo's, the card's to the bit) and scramble constants |
| cuSPARSE | `libcusparse.so.12` | every entry point NVIDIA's 13.0 exports. CSR/CSC/COO/BSR SpMV, SpMM (strided batches, fp16/bf16/int8), SpGEMM (and SpGEMMreuse), SDDMM, SpSV/SpSM (with updateMatrix), format conversion, CSR to CSC, in real and complex values (A, A^T and A^H); Blocked-ELL SpMM and sliced-ELL SpMV; sparse vectors (SpVV, Axpby, Gather, Scatter, Rot); the tridiagonal and pentadiagonal solvers (gtsv2, gtsv2_nopivot, gtsv2StridedBatch, gtsvInterleavedBatch, gpsvInterleavedBatch); legacy coo2csr, the CSR/CSC/COO sorts, csrgeam2, gemvi, the BSR family (bsrmv, bsrxmv, bsrmm, bsrsv2, bsrsm2, bsric02, bsrilu02, CSR to BSR and back, gebsr2gebsr, gebsr2gebsc), csric02 and csrilu02, pruning, csrcolor, nnz and compression, unsorted CSR. SpMV, SpMM, SDDMM, SpSV/SpSM solves, sparse to dense and CSR to CSC are recorded into a captured CUDA graph and run at each launch |
| cuSOLVER | `libcusolver.so.12` | Cholesky, LU, QR (with `ungqr`/`unmqr` for complex), symmetric and Hermitian eigen, SVD, in real and complex types; the reductions and their back-transforms (`sytrd`/`hetrd`, `orgtr`/`ungtr`, `ormtr`/`unmtr`, `gebrd`, `orgbr`/`ungbr`), `potri`, `lauum`, selected and generalized eigen (`syevdx`/`heevdx`, `sygvd`/`hegvd`, `sygvdx`/`hegvdx`, `sygvj`/`hegvj`); symmetric indefinite (Bunch-Kaufman `sytrf`, `Xsytrs`, `sytri`), `laswp`; the iterative refinement solvers (`<t1><t2>gesv`/`gels`, `IRSXgesv`/`IRSXgels`); the 64-bit X API with `Xgetrf`/`Xgetrs`, `Xtrtri`, `Xsyevdx`, `Xgesvd`, `Xgesvdp`, `Xgesvdr` and `Xlarft`, `Xgeev` (right eigenvectors) on real and complex matrices, Jacobi (gesvdj, syevj, heevj) and batched forms, gesvdaStridedBatched. The sparse module, cusolverSp: `csrlsvlu`/`csrlsvqr`/`csrlsvchol` (host and device), `csrlsqvqr`, `csreigvsi`, `csreigs`, the reorderings (`symrcm`, `symamd` and `symmdq` give NVIDIA's own permutations), `csrperm`, `csrzfd`, batched QR, and the low-level preview API (LU on the host, QR and Cholesky on the host and the device, step by step). The refactorization module, cusolverRf, single and batched |
| cusolverMg | `libcusolverMg.so.12` | getrf/getrs, potrf/potrs/potri and syevd on a matrix, or getrf/getrs and potrf/potrs/potri on a submatrix (IA, JA), spread over several devices in NVIDIA's column-block-cyclic layout |
| NCCL | `libnccl.so.2` | collectives (all-to-all, gather and scatter included, and the `nccl*Config` forms of each) and point-to-point across ranks; ncclCommSplit, ncclCommShrink, ncclCommGetUniqueId + ncclCommGrow, ncclCommRevoke, ncclCommSuspend/Resume/MemStats, ncclCommInitRankScalable, non-blocking communicators, pre-multiplied sums with host or device scalars, the `ncclParam*` registry; the device API's host side answers as the RTX 3060 pair does (unsupported) |
| cuStateVec (cuQuantum) | `libcustatevec.so.1` | dense and diagonal gates with any controls, controlled index-bit swaps, probabilities, projection and Pauli expectation values: what QuEST's cuQuantum backend calls. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuDSS | `libcudss.so.0` | the sparse direct solver, the whole 0.8 API: LU, LDL^T, LDL^H and Cholesky in every index width, view, base and value type, several right-hand sides, the solve sub-phases, iterative refinement, batches, a factorization or solve captured into a CUDA graph -- and SCS's GPU direct backend. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuFile (GPUDirect Storage) | `libcufile.so.0` | compatibility mode: file I/O staged through host memory into device memory, the driver and parameter API, handle and buffer registration, batch and stream-ordered I/O, statistics; NVIDIA's statuses (CUDA 13.0) |
| nvCOMP | `libnvcomp.so.5` | the low-level batched API and the C++ manager API for LZ4, Snappy, Deflate, GDeflate, Gzip and Zstd, chunks and containers interoperable with NVIDIA's in both directions, and CRC32; Cascaded, Bitcomp and ANS refused (no public bitstream) |
| NVSHMEM | `libnvshmem_host.so.3` | the host API across a job of PEs, one simulated GPU per process, bootstrapped by unique ID; the device API of kernels built with NVIDIA's NVSHMEM headers and device library, all PEs peer to peer |
| cuSPARSELt | `libcusparseLt.so.0` | 2:4 structured sparse matrix products, the whole 0.10 API: dense and structured descriptors with batches, fp16, bf16, tf32 and int8 (into int8, int32, fp16, bf16) in either operand, transposes and both orders, STRIP and TILE pruning and the prune check value for value with the card, compression with the card's sizes and layout, bias, ReLU, GELU and alpha/beta vectors, the search, graph capture. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
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
| `nccl_comm_ops` (e2e) | all 74 checks of NCCL 2.29's surface pass against NCCL 2.29.7, and all CHECKS_PLACEHOLDER against NCCL 2.31.2, at two ranks on two physical GPUs: split, shrink, grow, revoke, suspend and resume, memory statistics, non-blocking, pre-multiplied sums, all-to-all, gather, scatter, scalable init, windows, the `nccl*Config` forms, the `ncclParam*` registry, the device API's host side, and their error codes |
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
`TRUE_HALF` convolution accumulates in half in its own order. Dropout is not
an exception any more: its masks are cuDNN's, see below. And this library runs some configurations
NVIDIA's does not -- every algorithm for every shape, dy with gaps in
backward-data and x with gaps in backward-filter, double-precision NHWC
backward-data -- rather than refusing them.

Dropout is cuDNN's own, mask for mask. Its states buffer, which the
program can read, holds one cuRAND XORWOW state (48 bytes) per thread of the
dropout kernel -- 768 a streaming multiprocessor, 21504 on the RTX 3060,
which is what `cudnnDropoutGetStatesSize` reports; the card's rule for other
GPUs is not known, and 768 per SM is assumed for them -- thread t seeded as
`curand_init(seed, t, 0)` seeds it (Marsaglia's recurrence, whose subsequence
jump is a GF(2) matrix power computed here). `cudnnDropoutForward` gives
element i, in the tensor's logical order, the next output of thread i % T and
keeps it when `curand_uniform` of it exceeds p, writing `x * (1 / (1 - p))`
in float (in double for double data); the reserve space holds the mask one bit
an element. The states buffer after a set and after each pass is word for
word the card's, so `cudnnRestoreDropoutDescriptor` resumes a stream and a
second pass continues it. Edges matched: a states buffer smaller than
`cudnnDropoutGetStatesSize` says is `BAD_PARAM` (a larger one is fine), a null
one is accepted and leaves the descriptor's buffer as it was (a forward pass
through a descriptor that never had one is `BAD_PARAM`), a probability
outside [0, 1] is accepted, bfloat16 is `NOT_SUPPORTED`. An RNN's dropout
between layers is the same kernel: after each layer but the last, each
direction of the next layer gets a mask of its own over the whole of the
lower layer's output, `[T][B][hidden * dirs]`, forward direction first, layer
after layer. The classic multi-head attention API's two dropouts are the same
kernel too: the attention dropout is one application over the probabilities,
`[batch][beam][head][query step][key step]`, the post dropout one over the
output vectors, `[batch][beam][query step][output]`, applied after the output
projection and before the residual is added, both over the dimensions of the
data the call was given, padded steps included.
`e2e_dnn_dropout`, `e2e_dnn_rnn_dropout` and `e2e_dnn_attn_dropout` check all of
it against a host model on the card's library and on this one. Not measured,
and so not claimed: an RNN batch of sequences shorter than the longest
(padded I/O), and the states size of GPUs other than the RTX 3060.

Two more things the card does that were first taken for refusals. The classic
API's fused `CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD` (the weight gradient of
a convolution whose input is `activation(x * eqScale + eqBias)`, the affine
result rounded to the data type, as `CONV_BNSTATS` does) runs on an RTX 3060;
it had been written down as unsupported because the configuration first
tried, half data with a float `dw`, is. The plans cuDNN makes: x, dy and dw
all half or all float (a mix, bfloat16 and double are `NOT_SUPPORTED`), float
compute type, RELU, IDENTITY or no activation, scale and bias each optional
and half or float, spatial batch-norm mode, groups, strides, dilation and
padding as the convolution has them, x and dw in either layout and dy NHWC
unless dw is NCHW too; with nothing to fuse at all the three layouts have to
agree. The sum is exact, rounded once to dw's type. `e2e_dnn_classic_paths`
checks it against the card's library and this one. The other fused ops are
unchanged: `CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION` and the two after it
are marked "reserved for future use" in cuDNN's own header and are
`NOT_SUPPORTED` on the card as here. And the graph API's multi-GPU batch
normalization (`CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS`, and the
backward one) runs on the card's two GPUs, which have no peer access, with the
peer tensors in pinned host memory: every execution computes with the batches
of all the GPUs together -- mean, inverse variance, output, running statistics
(variance unbiased over the combined count) and, backward, `dx` -- while the
gradients of the scale and bias come back as the combined sums divided by the
number of GPUs (measured with two; with more it is assumed). The words inside
the peer tensors are cuDNN's own protocol (a pair of a value and a flag for
each statistic, as far as the tensors show), which this library does not use:
its executions meet in memory, so they must be threads of one process, and a
lone execution of a two-GPU graph fails after `VGPU_CUDNN_PEER_TIMEOUT_MS`
(60 s) with `CUDNN_STATUS_EXECUTION_FAILED` where NVIDIA's kernel would wait.
`e2e_dnn_multigpu_norm` (two GPUs) checks half and float, forward and
backward, on both libraries.

The graph API's attention was pinned down against cuDNN 9.27 on an RTX 3060
through cudnn-frontend 1.30, the frontend PyTorch and Transformer Engine
build their attention with. The frontend emits it in two forms, and both run
here as the hardware runs them: a single `SDPA_FWD` operation (its
"unified" node, which carries a softmax operation and a subgraph of score
modifiers -- bias, causal and sliding-window masks -- run on the scores), and
a composite graph of matmuls, pointwise operations, a softmax, diagonal-band
masks, an RNG and reshapes, the backward pass always the composite one. For
that to run as a graph the general operations grew what it relies on: matmuls
whose batch dimensions group (K and V heads shared by several query heads),
per-batch M, N and K overrides with a padding value (padded sequences),
grouped reductions, view-only reshapes (a transpose through permuted
strides), scalars of lower rank broadcast from the right, and tensors in
workspace memory. `e2e_dnn_attention` runs nineteen configurations forward
and backward on the RTX 3060 against a double-precision reference -- half,
bfloat16 and float; top-left and bottom-right causal masks; a sliding window;
grouped-query heads; bias and its gradient; padding; paged K and V caches; packed (ragged)
sequences; an interleaved layout; dropout -- and NVIDIA's library agrees in
every one it has an engine for (it has none for float's backward pass), as
VirtualGPU does in all of them. Dropout keeps each probability with chance
1 - p from a Philox4x32-10 stream keyed by the graph's seed and offset, as
cuDNN documents its RNG operation, and the backward pass regenerates the
forward pass's mask from the same pair; the mask's layout is cuDNN's
kernel's own (on an RTX 3060 it repeats with the row's position in an MMA
tile and depends on the sequence length's tiling) and is not reproduced, so
the kept fraction, scaling and reproducibility match while the individual
elements kept differ. Also measured and matched: max pooling's index tensor
is INT8, the maximum's row-major position within its window with padded taps
counted, and the backward pass may read it in place of x; nearest and
bilinear resampling refuse a window other than 2 when the descriptor is
finalized (cuDNN documents this for bilinear; the hardware also does it for
nearest). Interpolation has one configuration with an engine, which cuDNN
documents for its runtime-fusion engines: bilinear upsampling by 2 (NHWC, float,
window 2, strides 1/2, pre-padding 1/2, post-padding 1, so that the output
is twice the input). It runs on the RTX 3060: output i samples the input at
`s = i * stride - pre + window / 2 - 1/2` per dimension (that is i / 2),
clamped to the input, between the pixels either side of s by their distance
(zero and edge-value padding give the same output); half and bfloat16 data get an engine
whose plan cannot be built (cudnn-frontend's build_plans fails). Everything
else has no engine -- nearest in every configuration (cuDNN documents that),
NCHW, other scales and paddings, backward upsampling (cuDNN documents that
too) -- and is refused here the same way. An earlier version of this
paragraph said the card had no interpolation engine at all; it had been probed
with parameters outside the documented ones. An INT8x4 vectorized convolution (a channel dimension holding vectors
of 4) saturates as the classic API's INT8 convolution does.

The rest of the classic API was measured on the same card and matched.
`NCHW_VECT_C` tensors report strides in vectors (2x8x3x5 INT8x4 is 240
bytes); INT8x4 and UINT8x4 convolve to INT8x4 or FLOAT (NCHW on both
implicit GEMMs, NHWC on the precomputed one), INT8x32 to INT8x32 only, with
no dilation and no backward pass; `cudnnReorderFilterAndBias` permutes an
INT8x32 filter in 32-byte chunks of eight output channels (and its bias in
blocks of 32), which a `CUDNN_NO_REORDER` convolution reads back. INT8
pooling and activation round to nearest even after alpha and beta; FP8,
BOOLEAN and INT64 classic descriptors are BAD_PARAM. Divisive normalization
is `x / (K + alpha / n^d * sum (x_j - m)^2)^beta` over the window, and its
backward pass gathers over each element's own window (which differs from
the exact gradient only for even windows). A tensor transform pads, then
folds channels as `padBefore + offset * C + c`; folding and padding take no
beta. `cudnnGetFoldedConvBackwardDataDescriptors` pads K to a multiple of 8
and C to `floor(roundup(C * sh * sw, 8) / (sh * sw))`. LSTM projections
keep the weight space's matrices before all its biases (as every RNN now
does), and cell clipping limits the state where it is read, with float
bounds, reporting cy unclipped. Fused-ops plans run BNSTATS convolution
(half NHWC, the affine result rounded to half) and both batch-norm
finalizations. Multi-head attention keeps W_Q, W_K, W_V, W_O and then the
biases in the weight buffer, input element slowest, and writes steps past a
sequence as the output bias plus the residual.

## Checked in CI without hardware

The paths PyTorch takes through these libraries are also covered by
self-checking programs in `nvidia/tests/e2e/`, which compare each result with a
host reference (a direct DFT, a dense product, a residual or reconstruction)
and so run on every pull request with no GPU. `run_lib_check.sh` builds and
runs them; each is a ctest of its own.

| test | covers | torch |
| --- | --- | --- |
| `e2e_fft_layouts` | cufftXt plans, strided and padded layouts of every rank, 2‑D/3‑D C2R, half | `torch.fft` |
| `e2e_fft_multigpu` | cuFFT on two devices: what multi-GPU planning refuses, batched, 2‑D/3‑D (x split, then y) and 1‑D (strings, input-shuffled) descriptors, R2C/C2R, device-to-device copies; passes on an RTX 3060 pair too | multi-GPU FFTs |
| `e2e_fft_callbacks` | LTO load and store callbacks in C2C (strided, batched, callerInfo), R2C, Z2Z and C2R; what cufftXtSetJITCallback and planning refuse; libcufft.so's NOT_IMPLEMENTED legacy callbacks | cuFFT callbacks (CUDA 12.6+) |
| `e2e_solver_paths` | cuSOLVER X API, gesvdj/syevj and their batched forms, gesvdaStridedBatched, batched potrf/potrs; cuBLAS batched LU | `torch.linalg` |
| `e2e_solver_sparse_paths` | cusolverSp: LU, QR and Cholesky solves in S/D/C/Z with every reorder, singularity, least squares, shift-inverse eigenvalues, reorderings (NVIDIA's permutations for symrcm, symamd, symmdq), permutations, batched QR | `scipy`-style sparse solves |
| `e2e_solver_mg_paths` | cusolverMg on two devices: getrf/getrs, potrf/potrs/potri, syevd, IPIV's layout, submatrices, the grids NVIDIA's refuses | multi-GPU dense solvers |
| `e2e_solver_dense_paths` | the reductions and back-transforms, potri/lauum, syevdx and the generalized eigensolvers, the iterative refinement solvers, Xgetrf/Xgetrs, Xtrtri, Xsyevdx, Xgesvd, Xgesvdp, Xgesvdr, Xlarft, Xgeev's left-eigenvector refusal, the handle modes and Jacobi getters | `torch.linalg.eigh` on generalized problems, `cholesky_inverse`, mixed-precision solves |
| `e2e_solver_rf_paths` | cusolverRf: setup, analyze, refactor, solve, the documented defaults, zero pivots and the boost, the unit-diagonal formats and split factors, batched (on the simulator) | sparse refactorization loops (circuit simulation) |
| `e2e_solver_sparse_ll_paths` | cusolverSp's low-level preview API: threshold LU, QR with a shift and least squares, Cholesky in its elimination tree's postorder, host and device, the call-order refusals | sparse direct solvers built on the preview API |
| `e2e_solver_sytrf_paths` | sytrf + Xsytrs and sytri in S/D/C/Z, both triangles, 2x2 pivots, singular D; laswp; Xgeev on complex matrices | `torch.linalg.ldl_factor`, complex `eig` |
| `e2e_sparse_paths` | coo2csr and the sorts, batched and half SpMM, SpGEMM, csrgeam2, SDDMM, SpSV/SpSM | `torch.sparse` |
| `e2e_sparse_complex_paths` | SpMV, SpMM, SDDMM, SpSV/SpSM, SpGEMM, conversions and csrgeam2 on complex values, every op; the type combinations and conjugate transposes NVIDIA's refuses | complex `torch.sparse` |
| `e2e_sparse_bsr_paths` | generic BSR (SpMV, SpMM, SDDMM) and the legacy BSR family: bsrmv/bsrxmv/bsrmm, bsrsv2/bsrsm2 with their zero pivots, bsric02/bsrilu02 (and csric02/csrilu02) with ILU's boost, CSR to BSR and back | preconditioned iterative solvers |
| `e2e_sparse_tridiag_paths` | gtsv2 (pivoting), gtsv2_nopivot and gtsv2StridedBatch (PCR, and CR past 2048 and 512 unknowns: which unknowns a zero pivot spoils), the interleaved Thomas, LU and QR and the pentadiagonal QR with what each leaves in its inputs, in S, D, C and Z | ADI and spline solvers, PyTorch's `torch.linalg` tridiagonal paths |
| `e2e_sparse_vector_paths` | sparse vectors (SpVV in every compute type, Axpby, Gather, Scatter, Rot), gemvi, Blocked-ELL SpMM and DenseToSparse, sliced-ELL SpMV, and what NVIDIA's refuses for each | sparse optimizers, block-sparse attention |
| `e2e_sparse_helper_paths` | pruning (by threshold and percentage), nnz and compression, unsorted CSR, gebsr2gebsr/gebsr2gebsc, csrcolor, SpGEMMreuse, SpGEMM's product count and memory estimate, SpMMOp's refusal, SpSV/SpSM updateMatrix, the logger, the CSC sort | model pruning, multigrid setup |
| `e2e_complex_paths` | complex cuBLAS (GEMM in every batched form, GEMV, level 1, trsm, batched LU, herk, hemv) and cuSOLVER (LU, Cholesky, QR with ungqr/unmqr, heevd/heevj, gesvd/gesvdj, the X API on complex types) | complex tensors in `torch.linalg`, `@` |
| `e2e_lt_paths` | fp16/bf16 matmul with bias epilogues, strided batches, row-major layouts, FP8 scales and amax | `addmm`, `bmm`, `_scaled_mm` |
| `e2e_lt_epilogue_paths` | RELU_AUX/GELU_AUX's mask and input, DRELU/DGELU and their bias gradients, BGRADA/BGRADB, in fp16/bf16/fp32/fp64, and what the card refuses | a training step's backward pass (cuBLASLt-fused linear layers) |
| `e2e_lt_blockscaled_paths` | MXFP8 and NVFP4 block scales in the tiled layout, the 128-element and 128x128 FP32 forms, D's block quantization and its output scales (simulator only: documentation-derived) | `_scaled_mm` with block scales |
| `e2e_blas_packed_paths`, `e2e_blas_batched_paths`, `e2e_blas_64_paths`, `e2e_blas_legacy_paths`, `e2e_blas_xt_paths` | cuBLAS's band and packed level 2, batched GEMV, getri/matinv, syrkx/herkx, the `_64` forms, the handle settings, the legacy API and cuBLASXt on two devices | SciPy-style BLAS callers, multi-GPU GEMM |
| `e2e_dnn_backward` | cuDNN's convolution passes against each other, every backward pass against finite differences, algorithm lists, status codes, dropout, an LSTM's gradients through dropout, LSTMs in half, bfloat16 and double, CTC's gradient | `conv2d`, pooling and activation backward, `nn.LSTM(dropout=)`, `ctc_loss` |
| `e2e_dnn_classic_paths` | cuDNN's classic API beyond the training paths: INT8x4/UINT8x4/INT8x32 convolution and fused bias-ReLU against an integer reference (also through `cudnnReorderFilterAndBias`), transforms to and from `NCHW_VECT_C`, INT8 pooling and activation, divisive normalization against its formula and finite differences, padding/folding/unfolding transforms and the folded backward-data pipeline, LSTM projections and clipping against a host LSTM and finite differences, the RNN getters, fused-ops plans, multi-head attention forward and both gradients | `nn.LSTM(proj_size=)`, `nn.MultiheadAttention`-style models, INT8 inference engines |
| `e2e_dnn_dropout` | cuDNN's classic dropout bit for bit: the states buffer cudnnSetDropoutDescriptor fills (cuRAND XORWOW, one state per kernel thread), the output, the reserve space and the states after a pass in float, half and double, a second pass and a restored descriptor continuing the streams, the backward pass, and the statuses at the edges | `nn.Dropout` through cuDNN in other frameworks, RNN dropout |
| `e2e_dnn_rnn_dropout` | the dropout between an RNN's layers, mask for mask, in two and three layers, unidirectional and bidirectional, against the same host model | `nn.LSTM(dropout=)`, `nn.GRU(dropout=)` |
| `e2e_dnn_multigpu_norm` | multi-GPU batch normalization, forward and backward, half and float, two GPUs and two threads, peer tensors in pinned host memory | apex-style synchronized batch norm through cudnn-frontend |
| `e2e_dnn_graph` | cuDNN graphs: conv + bias + ReLU, dgrad + ReLU backward, matmul + bias + GELU, reductions, pointwise forward and backward, layer/RMS/batch/group norm forward and backward, backward without saved statistics, max and average pooling both ways, max pooling's index tensor, asymmetric padding, concatenation, statistics generation, RNG, reshape, transpose, slice, an INT8x4 vectorized convolution | `cudnn_convolution_add_relu`, cudnn-frontend |
| `e2e_dnn_attention` | cuDNN scaled dot-product attention built by cudnn-frontend 1.30 (fetched): the unified and composite forms forward and backward, causal (both alignments) and sliding-window masks, bias, grouped-query heads, padding, paged K/V caches, ragged sequences, dropout, half/bfloat16/float | `scaled_dot_product_attention` with the cuDNN backend, Transformer Engine |

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

Beyond the collectives:

- **`ncclCommSplit`** is a collective on the parent: every rank publishes its
  color and key, and the members of each color agree on a rendezvous name for
  the child derived from the parent's. Ranks are ordered by key, ties by old
  rank; `NCCL_SPLIT_NOCOLOR` gets a NULL communicator; a NULL config inherits
  the parent's. **`ncclCommShrink`** is called only by the surviving ranks,
  who already agree on who survives, so it needs no exchange.
  **`ncclCommInitRankScalable`** takes the same ids on every rank and joins one
  rendezvous named from all of them.
- **Non-blocking communicators** (`ncclConfig_t.blocking = 0`, or
  `NCCL_COMM_BLOCKING=0`) run their work on a background thread: init,
  collectives, `ncclGroupEnd` and `ncclCommFinalize` return `ncclInProgress`,
  and `ncclCommGetAsyncError` reports `ncclInProgress` until the work is done.
  A split of a non-blocking parent returns `ncclSuccess` and fills in the new
  communicator when the parent settles. As on NCCL, a call on a communicator
  whose previous operation has not finished is `ncclInvalidArgument`.
- **`ncclCommAbort`** also marks the rendezvous, so a peer waiting on the
  aborted rank gives up with `ncclRemoteError` instead of waiting out
  `VGPU_NCCL_TIMEOUT`.
- **Pre-multiplied sums** (`ncclRedOpCreatePreMulSum`): each rank's input is
  scaled by its own scalar, read at creation for `ncclScalarHostImmediate` and
  when the collective runs for `ncclScalarDevice`. fp32 and fp64 accumulate as
  a chain of fused multiply-adds in rank order, which is what NCCL computes on
  every fp64 element measured. NCCL's fp32 order follows the ring's chunks,
  which start the chain at different ranks, so an element can differ by an ulp
  (one of eight in the measured case). fp16 rounds each product first, as NCCL
  does.

- **`ncclCommGetUniqueId` and `ncclCommGrow`.** An id's first 16 bytes name the
  rendezvous the old and the new ranks meet at, derived from the parent's
  rendezvous and the number of grows it has seen; the rest is a nonce, so every
  call returns a different id, as NCCL's does. Because every existing rank
  derives the same name, the non-root form (`uniqueId = NULL`) needs no id
  passed to it. Existing ranks keep their numbers, a new rank brings its own,
  and a rank number or id used twice is `ncclInvalidArgument` rather than a
  hang. Where NCCL fails a new rank that arrives before the existing ranks have
  started (`ncclInternalError` after a fraction of a second), this waits; that
  is the one deliberate difference.
- **`ncclCommRevoke`** is local, and unblocks whatever the revoked
  communicator is waiting for: a rank alone in an all-reduce is released, and
  the call that was waiting returns `ncclSuccess`, as the stream completes on
  NCCL. After it every collective and send/receive is `ncclInvalidUsage`, a
  second revoke and `ncclCommFinalize` are `ncclInvalidArgument`, and split,
  shrink, suspend and destroy still work. A non-blocking communicator answers
  `ncclInProgress` and settles once its queued work has unwound.
- **`ncclCommSuspend` and `ncclCommResume`** are collective barriers (one rank
  waits exactly as long as another is late), valid in a group and on a
  non-blocking communicator. Only bit 0 (`NCCL_SUSPEND_MEM`) suspends; suspending
  twice or resuming what is running is `ncclInvalidUsage`. On the card, work
  issued to a suspended communicator faults on the buffers it released; here it
  is refused with `ncclInvalidUsage`. **`ncclCommMemStats`**: NCCL reports
  GPU memory it holds for the communicator (12 MiB it can release, 4 MiB it
  cannot, for a two-rank one), and suspending frees exactly the releasable part.
  This transport allocates no device memory, so the three sizes are zero and
  only the "suspended" statistic is live.
- **`nccl*Config` collectives** check their `ncclCollConfig_t` as NCCL 2.31.2
  does and then run the plain collective: size at least 64, magic, a
  `forceAlgSelection` of 0 or 1, a `CTAPolicy` unset or 0..3, in that order and
  before the communicator is looked at; a refused config does not spoil the
  group it was issued in. `algSelection` is a comma-separated list of algorithm
  names (`ring`, `tree`, `collnetdirect`, `collnetchain`, `nvls`, `nvlstree`,
  `pat`; any case; `^` in front means every other one). With `forceAlgSelection`
  left at 1, an unknown name or a selection that leaves nothing available is
  `ncclInvalidArgument`: AllReduce has ring and tree, Broadcast, Reduce,
  AllGather and ReduceScatter ring, and AlltoAll, Gather and Scatter none --
  the same sets the PCIe card has; NVLS, CollNet and PAT are not available.
  With 0 it falls back to automatic selection. CTA counts, cluster size,
  profiler tags and extension lists are accepted whatever they hold, as NCCL
  accepts them.
- **`ncclParam*`** reproduces NCCL 2.31.2's parameter registry as measured:
  six public parameters (`NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, `NCCL_DEBUG_FILE`,
  `NCCL_DEBUG_TIMESTAMP_FORMAT`, `NCCL_DEBUG_TIMESTAMP_LEVELS`,
  `NCCL_SET_THREAD_NAME`) and three private ones listed with
  `NCCL_PARAM_DUMP_ALL=true`. Types, defaults, documentation text and the way
  each environment variable is parsed are the card's, so are the getters (a
  getter takes the parameter's own type and is `ncclInvalidArgument` for any
  other), and `ncclParamDumpAll` writes the same text to stdout. The rest of
  NCCL's environment variables are not in the registry on the card either.
- **The device API's host side** answers as the RTX 3060 pair does:
  `ncclCommQueryProperties` reports `deviceApiSupport = false` and
  `hostRmaSupport = false` (no multimem, no GIN, one LSA team), and
  `ncclDevCommCreate` is `ncclInvalidUsage`. The team queries
  (`ncclTeamWorld`, `ncclTeamLsa`, `ncclTeamRail`, `ncclTeamRankToWorld`) and
  the requirement helpers (`ncclLsaBarrierCreateRequirement`,
  `ncclLLA2ACreateRequirement`, `ncclLLA2ACalcSlots`,
  `ncclGinBarrierCreateRequirement`) are plain host arithmetic, reproduced from
  the card's outputs.

Error codes and edge cases where the documentation is silent were measured on
NCCL 2.29.7 and 2.31.2 with two RTX 3060s, and
`nvidia/tests/e2e/nccl_comm_ops.cu` and `nccl_multiproc.cu` pass against both
libraries (sections for a newer API than the library in use say "skipped"). Four
things are deliberately not there:

- **Symmetric memory windows.** `ncclCommWindowRegister` returns `ncclSuccess`
  and a NULL window, which is NCCL's own answer on a machine without the peer
  mappings windows are built on (the RTX 3060 pair gives exactly that), and a
  collective on the buffer works as it always does. A real window promises
  device-side loads and stores into the peers' memory -- the device API's LSA
  pointers -- and another process's simulated device is reachable only
  through a file.
- **The device API.** Kernels that use a device communicator (`ncclDevComm`)
  reach peers through LSA pointers, multimem addresses or GIN, none of which
  can exist between simulated devices that share no address space; and the
  `libnccl_device` bitcode that NCCL ships for linking such kernels has nothing
  to link against. `ncclDevCommCreate` refuses and `ncclCommQueryProperties`
  says why, which is the RTX 3060 pair's answer too, so a program that checks
  support first takes its fallback.
- **One-sided operations.** `ncclPutSignal`, `ncclSignal` and `ncclWaitSignal`
  are `ncclInvalidArgument` ("host RMA is not supported in this
  communicator"), every time, which is what NCCL 2.31.2 does on the card
  with or without `numRmaCtx` and `numRmaSig` configured. NCCL moves a put
  through a GIN transport or (2.32) a socket proxy that needs GDRCopy; this
  transport would need a progress thread in every process holding a window,
  and there is no card here whose signal, context and ordering behaviour that
  could be measured against.
- **The network plugin interface.** A net plugin is a library NCCL loads to
  drive a NIC (`NCCL_NET_PLUGIN`); there is no network transport here for one
  to replace, so those variables are not read.

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
- **`nppiResizeSqrPixel`** cubic uses four Lagrange weights rounded the way
  NPP's are, and the order of fused multiply-adds NPP's float output shows:
  bit-identical to NPP 13.0 on the resize cases `npp_imgproc` pins. Lanczos
  (windowed sinc, three lobes, not widened when shrinking) matches the card
  at the factors tried to a float rounding and differs by up to 2e-3
  relative between them, where NPP's sinc is tabulated or approximated;
  super-sampling (area average, both factors below one, else
  `NPP_RESIZE_FACTOR_ERROR`) is compared as approximate. The warps'
  super-sampling is still `NPP_INTERPOLATION_ERROR`.
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

- **cuBLAS**: `cublasUint8gemmBias` (deprecated, and undocumented in
  cuBLAS 13: the card runs it, but there is no definition to implement), and
  the exported names the header does not declare (`cublas?bdmm`,
  `cublasGet/SetBackdoor`, `cublasGet/SetEnvironmentMode`); a program that
  needs one fails to load with the name. cuBLASXt runs GEMM's tiles across
  the selected devices but every other routine whole on the first, and never
  hands work to the CPU (`cublasXtSetCpuRatio` is kept, not used). The
  emulation controls (`cublasSetEmulationStrategy` and the fixed-point
  mantissa settings) are kept and read back; nothing is emulated, every GEMM
  being exact to its precision already.
- **cuBLASLt**: the auxiliary buffer's own scale and amax for FP8 epilogues
  (`EPILOGUE_AUX_SCALE_POINTER`, `EPILOGUE_AUX_AMAX_POINTER`), per-batch block
  scales, and the experimental scaling modes (`VEC32_MN_K4_UE8M0`,
  `VEC128_MN_K4_UE8M0`, per-batch tensor-wide scales). The block-scaled FP8
  and FP4 modes are implemented from NVIDIA's documentation only --
  documentation-derived, not card-verified -- since no card here (an RTX
  3060, sm_86) runs them; the backward epilogues match the card, except that
  GELU and its derivative are exact where the card's fp32 tanh is
  approximate (within about 5e-5).
- **cuDNN**: in the graph API, interpolating resampling beyond bilinear
  upsampling by 2 (the one configuration cuDNN has an engine for; nearest
  has none, which cuDNN documents), FP8
  and block-scaled (MXFP8) attention and the block-scale (de)quantize
  operations, attention's block masks and cumulative sequence lengths,
  sinks in the backward attention operation, reordered (INT8x32-interleaved)
  filters, multi-GPU normalization across processes (its executions meet
  in memory, so they must be threads of one process; with more than two GPUs
  the gradients' division by the number of GPUs is assumed), the MoE
  backward (cuDNN's engine for it wants Hopper or Blackwell and, documented, cuBLASLt 13.5, newer than this stack's),
  band-matrix and standalone RoPE operations (no engine on the RTX 3060, the
  only GPU measured; whether Hopper and Blackwell have one was not checked:
  the AWS H100 launch for it was denied), and the dropout mask layout of the fused attention
  kernels (the mask is drawn from the documented Philox generator but not
  placed as the kernels place it; the classic API's dropout, RNN and
  multi-head attention included, is cuDNN's own bit for bit); in the
  classic API, the fused ops cuDNN's header marks "reserved for future use"
  (`CONV_SCALE_BIAS_ADD_ACTIVATION` and the two undocumented ones), which
  the card refuses too, multi-head attention's one-to-one query mapping
  with beams (refused by the hardware as well), and the dropout masks of
  an RNN batch of sequences shorter than the longest (padded I/O), whose
  layout was not measured.
- **cuFFT**: legacy callbacks (`cufftXtSetCallback` with a device function
  pointer), which NVIDIA ships only in its static library: its `libcufft.so`
  answers every legacy callback call with `CUFFT_NOT_IMPLEMENTED`, and so does
  this one, which stands in for `libcufft.so` (a program linked against
  `libcufft_static` carries NVIDIA's own cuFFT). LTO callbacks whose image is
  LTO-IR -- NVVM bitcode, which only NVIDIA's compiler reads -- fail the plan
  as a callback that does not link fails it on the card (`NVJITLINK_FAILURE`,
  CUDA 13.2's answer; 13.0's is `INTERNAL_ERROR`); PTX images, as text or in
  a fatbin, work. LTO callbacks on multi-GPU plans fail as NVIDIA's do. The
  multi-GPU layouts were measured on two GPUs; with more, they follow the
  documentation (batches and planes dealt out in order, 1‑D strings over the
  GPUs in order), and the 1‑D factor choice past 2^27 points keeps the last
  measured one.
- **cuSPARSE**: the legacy `cusparse<t>csrmv` family (removed by NVIDIA in
  CUDA 12); SDDMM with a conjugate transpose (NVIDIA's documents none and
  computes something else when given one); `cusparseSpMMOp`, whose operators
  are LTO-IR (NVVM bitcode) that VirtualGPU cannot compile -- `_createPlan`
  answers as NVIDIA's does when nvJitLink refuses them (INTERNAL_ERROR).
  `csrcolor` gives a proper coloring, but not NVIDIA's colors (its algorithm is
  undocumented and randomized). Where NVIDIA's 13.0 does something no caller
  can mean, this does what the documentation says instead: `csr2csr_compress`
  keeps |a| > tol as `nnz_compress` counts (NVIDIA's drops negative real
  entries and leaves their slots unwritten), a negative pruning threshold keeps
  every entry (NVIDIA's returns column indices past n), `gpsvInterleavedBatch`
  with an algo other than 0 is NOT_SUPPORTED (NVIDIA's does nothing and
  reports success). The solvers agree with NVIDIA's to rounding, not bit for
  bit: they compute in double.
- **cuSPARSELt**: FP8 and FP4 inputs (sm_89 and later on NVIDIA's library;
  their scale modes are accepted and ignored), fp16 compute (no sm_86 kernel on NVIDIA's library
  either), and GELU outside int8 output (refused there too); see the section
  above for where the compressed layout and the search differ.
- **cuSOLVER**: left eigenvectors from `Xgeev` (NVIDIA's CUDA 13.0 and 13.2
  libraries answer jobvl = VECTOR with INTERNAL_ERROR and document right
  eigenvectors only; this does the same), `csrmetisnd`'s METIS permutation
  (NVIDIA's runs METIS 5.1.0's `METIS_NodeND` with its default options on
  A + A^T without the diagonal -- a reference METIS 5.1.0 build gave its
  permutation on 99 of 100 matrices -- and VirtualGPU carries no METIS, so it
  returns `symmdq`'s minimum-degree permutation instead), cusolverSp's
  `csrlsvlu` on the device (NVIDIA ships only the host one), and cusolverMg
  grids with more than one row of devices (NVIDIA's refuses them too, at
  `cusolverMgCreateDeviceGrid`). Measured differences: when several columns of
  a Cholesky factorization are independent of one another NVIDIA's names a
  different one in `singularity`. `Xgeev` on a complex matrix returns its
  eigenvalues in NVIDIA's order up to n = 74 (both are LAPACK's single-shift
  QR there); past that NVIDIA's switches algorithm, as LAPACK's does, and the
  order can differ. NVIDIA's `sytri` (CUDA 13.0, RTX 3060) returns success and
  leaves A as it was; this one computes the inverse the API documents.
  NVIDIA's batched cusolverRf crashed on every input tried (a cudaFree of an
  invalid pointer in `cusolverRfBatchAnalyze`/`BatchRefactor`, CUDA 13.0 and
  13.2), so the batched forms follow the documentation unmeasured. The
  low-level preview QR factors in place on NVIDIA's, so a second `csrqrFactor`
  without a new setup refactors its own output; here each Factor starts from
  the setup's matrix. NVIDIA's cusolverMg getrf on a submatrix that starts
  below its diagonal block (IA > JA) returns nothing recognisable; this
  returns the submatrix's LU. The iterative refinement solvers can take one
  refinement step more or fewer than NVIDIA's (its GMRES variants and some
  n = 200 systems), and workspace sizes (`_bufferSize`) are this library's
  own; Jacobi sweep counts are this implementation's, and singular vectors
  for repeated singular values can differ by a rotation, as LAPACK's
  documentation allows.
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
- **NCCL**: symmetric memory windows (registration returns a NULL window, as
  NCCL does without peer mappings) and the network plugin interface (there is
  no network to plug into); see "NCCL: a file-backed transport". Every entry
  point NCCL 2.31.2 exports is here and answers as the card does, except
  `ncclSetEncryption` (2.32, TLS for the bootstrap sockets, which this transport
  does not have) and the two GIN requirement helpers that no Linux header
  declares. Refused by name, with the card's code: `ncclDevCommCreate` (device
  API), `ncclPutSignal`, `ncclSignal` and `ncclWaitSignal` (host RMA); a window
  query on any window is `ncclInvalidArgument`, there being none. Not
  reproduced: the sizes `ncclCommMemStats` reports (zero, there is no device
  memory behind a communicator), NCCL's failure of a grow whose new rank
  arrives first, and work issued to a suspended communicator (a fault on the
  card, `ncclInvalidUsage` here). Shrink refuses an excluded rank outside the
  communicator, which NCCL 2.29.7 accepts and miscounts.
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
  and marker-label compression (one CUDA Sample), and the rest of NPP's ten thousand entry
  points, which are absent rather than approximated, so a program that needs
  more fails at link time with a name.
- **Device runtime** (cudadevrt, dynamic parallelism), on both engines: all of
  `cuda_device_runtime_api.h` that CUDA 12 and 13 still offer to a kernel --
  device-side launches (`<<<>>>`, `cudaGetParameterBuffer` /
  `cudaLaunchDevice`, tail and fire-and-forget streams, named streams and
  events), the pending-launch limit, `cudaMemcpyAsync`, `cudaMemcpy2DAsync`,
  `cudaMemcpy3DAsync` and the memset family, `cudaMalloc` and `cudaFree`,
  `cudaFuncGetAttributes`, `cudaDeviceGetAttribute`, `cudaDeviceGetLimit`,
  the cache configuration, the occupancy queries (and
  `cudaOccupancyMaxPotentialBlockSize` from them), `cudaGetErrorString`,
  `cudaGetErrorName`, `cudaRuntimeGetVersion`, the last error and
  `cudaGetDevice`/`cudaGetDeviceCount` -- under CDP2's names, and CDP1's
  (`-DCUDA_FORCE_CDP1_IF_SUPPORTED`, which has `cudaDeviceSynchronize`, for
  parts before Hopper). Each call returns what an RTX 3060 returned
  (`nvidia/docs/sass.md`, "Device runtime"). What remains: a parameter buffer
  launched a second time (the card runs it again; here it is
  `cudaErrorInvalidValue`); `cudaMemcpy3DAsync` between `cudaArray`s
  (`cudaErrorInvalidValue`; the card's answer is unmeasured) and copies of
  shared or local memory; `cudaGraphKernelNode*` updates from a kernel and
  cooperative groups' multi-grid and `cudaCG*` calls into the library
  (refused by name, as is any other `__cuda_syscall_*` the library makes);
  and kernels running concurrently with their parent: children run after it,
  so a kernel that waits (spinning on a flag) for a child it launched, or a
  child that waits for its parent, hangs here where it runs on the card. A
  device-side `cudaMemsetAsync` fills with its value; the card's wrote zeros
  whatever it was.
- **nvJPEG**: 12-bit samples, arithmetic coding, lossless and hierarchical
  JPEG (refused by name, `NVJPEG_STATUS_JPEG_NOT_SUPPORTED`); the hardware
  backend and what only it does (`nvjpegDecodeBatchedEx`, scaled decodes,
  applying an EXIF orientation, `nvjpegDecodeBatchedParseJpegTables`);
  carrying metadata or Huffman tables from a parsed image into an encode; and
  the transcoding entry points.

Add them the way the PTX subset grew: hit one, implement it, prove it against
hardware.
