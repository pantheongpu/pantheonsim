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

| library | soname | what it covers |
| --- | --- | --- |
| CUDA driver | `libcuda.so.1` | contexts, modules, memory, launches |
| CUDA runtime | `libcudart.so.13` | the nvcc registration ABI, streams, events |
| NVML | `libnvidia-ml.so.1` | discovery and telemetry (`pynvml`, nvitop) |
| cuBLAS | `libcublas.so.13` | GEMM (fp32/fp64/fp16/bf16/int8), GEMV, level‑1 |
| cuBLASLt | `libcublasLt.so.13` | descriptor matmul with ReLU/bias/GELU epilogues |
| cuDNN | `libcudnn.so.9` | convolution, activation, pooling, softmax, batchnorm |
| cuFFT | `libcufft.so.12` | C2C/R2C/C2R in 1‑D, 2‑D and 3‑D, batched |
| cuRAND | `libcurand.so.10` | host-side uniform and normal generation |
| cuSPARSE | `libcusparse.so.12` | CSR/COO SpMV and SpMM, format conversion |
| cuSOLVER | `libcusolver.so.12` | Cholesky, LU, QR, symmetric eigen, SVD |
| NCCL | `libnccl.so.2` | collectives and point-to-point across ranks |
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

`tests/conformance/run_conformance.sh` compiles each test once, runs it against
NVIDIA's library on a physical GPU and against VirtualGPU's, and diffs the
output. Anything that differs is a bug in this implementation.

| suite | result on an RTX 3060 |
| --- | --- |
| `cublas_gemm` | every GEMM path bit-identical, mixed precision included; level‑1/2 to ~1e‑7 |
| `lt_and_rand` | cuBLASLt bit-identical; cuRAND matches distribution and reseed semantics |
| `cudnn_ops` | all 50 reported values bit-identical |
| `cufft_transforms` | all 14 bit-identical, across composite, prime, 2‑D, 3‑D and both precisions |
| `cusparse_ops` | all 15 bit-identical |
| `cusolver_factorizations` | Cholesky, LU (pivots included), QR and every solve bit-identical; one f32 eigenvalue differs by ~1e‑6 relative |
| `nccl_collectives` | all 24 bit-identical at two ranks on two physical GPUs |
| `nvrtc_jit` | identical: compile a kernel at run time, load the PTX, launch it, same numbers |
| `npp_ops` | all 48 bit-identical, across arithmetic, logic, conversion, colour, statistics, morphology and resizing |
| `nvjpeg_codec` | all 24 identical: header parsing exactly, pixels to within the IDCT's own tolerance |
| `multi_gpu` | all 13 identical to two physical GPUs |

The library majors in that table are the ones this machine has; the build
reads each soname off the installed toolkit, so on a CUDA 12 host the same
shims come out as `libcublas.so.12`, `libcufft.so.11` and so on.

Some of those numbers came out of the hardware rather than the documentation.
cuDNN rejects `CUDNN_ACTIVATION_IDENTITY` from `cudnnActivationForward`, and
cuRAND does *not* rewind its stream when the seed is set again. Both were found
by differential testing and are matched deliberately.

The same suite runs on a rented multi-GPU machine through
`tools/verify-multigpu-cloud.sh`, which builds VirtualGPU there and compares
against that machine's own libraries -- so the results above are not one
laptop's. It has been run on two shapes so far, and all thirteen suites match on both:

| machine | what it adds |
| --- | --- |
| 2x H100 SXM5, CUDA 12.8 | NVLink, and everything the older toolkit does differently -- LZ4 fatbins, the `cudaGetDeviceProperties_v2` spelling, different soname majors |
| 8x A100 80GB SXM4, sm_80 | a second architecture, eight-rank NCCL against NVIDIA's libnccl, and eight ranks across eight processes |

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

`tests/conformance/multi_gpu.cu` compares the semantics against a real
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

`tests/e2e/run_nccl_multiproc.sh` runs 4 genuinely separate processes;
`tests/e2e/run_nccl_group.sh` runs the single-process grouped form over 4
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

- **cuBLAS**: complex types, triangular solves, most of level‑2/3, and the
  pointer-array batched forms (`cublasSgemmBatched` and friends).
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
