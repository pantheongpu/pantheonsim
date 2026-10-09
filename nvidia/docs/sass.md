# SASS: NVIDIA machine code

VirtualGPU runs a kernel's SASS -- the machine code the GPU itself executes --
whenever the binary carries SASS the simulated GPU can run, as the real driver
does. PTX is the fallback: it is JIT-compiled (here, interpreted) when there is
no suitable SASS, and it stays the path for programs that ship only PTX.

This replaces an earlier policy of executing PTX only. Running SASS is what
puts the simulator at the level the hardware works at: 255 physical registers
per thread and their allocation, predicate and uniform registers, convergence
barriers, the constant banks the driver fills, and the exact instruction
sequence ptxas chose -- including whatever ptxas got wrong. It is also what
lets SASS-only programs run at all.

## Sources

SASS is not documented by NVIDIA. Everything here comes from public material
and measurement, never from disassembling NVIDIA's driver or runtime:

- **Encodings**: Mesa's NAK compiler (`src/nouveau/compiler/nak`, MIT), which
  emits SASS for Volta and later and is itself checked against nvdisasm; and,
  for instructions NAK does not emit, NVIDIA's public disassembler.
- **Decode oracle**: `nvdisasm` / `cuobjdump -sass`, shipped with every CUDA
  toolkit. The decoder is checked instruction by instruction against it over a
  corpus (below), the way the AMD decoder is checked against llvm-objdump.
- **Execution oracles**: the simulator's own PTX interpreter, already checked
  against hardware, running the PTX of the same kernel -- a binary carrying
  both must produce the same results either way; and a physical RTX 3060
  (sm_86) for instruction semantics, bit for bit.

## Architecture

```
fatbin -> cubin (ELF) loader -> decoder (per-arch tables) -> SASS warp executor
            .nv.info: params, regs,          |                    |
            shared, EXIT offsets        disassembler         memory, textures,
            relocations, constants     (checked vs nvdisasm)  runtime: shared with
                                                               the PTX interpreter
```

- **Selection** (`driver_api.cpp`, `runtime_api.cpp`): a fatbin's ELF image for
  the device's architecture wins over its PTX, by the driver's rules: `sm_XY`
  SASS runs on the same major at minor Y or newer; `sm_XYa` only on XY (an
  `sm_100a` cubin is no candidate on a B300, 10.3, or a Vera Rubin, 10.7, and
  the fatbin's plain PTX, when it has some, runs there instead). Whether a
  cubin is `a` is not in `e_flags`: it is the record with attribute 9 in
  `.nv.compat` (1 for `a`, 0 for plain and for `f`, measured on cubins CUDA
  12.0's and 13.0's nvcc write).
- **Loader**: kernel sections `.text.<name>`, per-kernel attributes from
  `.nv.info.<name>` (parameter layout, register count, shared memory, barrier
  count), constant sections, globals and relocations.
- **Kernel handle**: a SASS kernel is an ordinary `ptx::EntryFn` (name,
  parameter sizes, attributes) with the decoded code attached, so everything
  above the launch -- argument marshalling, occupancy, attributes -- is shared.
- **Executor**: a warp is 32 lanes; state is the architectural state: R0-R254
  (RZ), P0-P6 (PT), UR0-UR62 (URZ), UP0-UP6, the convergence barriers of
  BSSY/BSYNC, special registers, and the constant banks (c[0x0] holds the
  launch: grid and block sizes, the stack, and the parameters). Scoreboards and
  stall counts in the control bits are timing, which VirtualGPU does not model;
  every instruction completes before the next.

## Selection and fallback

A binary's fatbin is searched for an ELF image the GPU can run (above). The
SASS executor then checks every instruction of it: an image with one the
executor does not run is passed over for the fatbin's PTX when there is PTX,
so a program keeps working while the SASS path grows; with no PTX the image
runs and the kernel reports the instruction it stopped at. A binary with SASS
only for another architecture and no PTX gets `cudaErrorNoKernelImageForDevice`,
as on the hardware.

| Variable | Effect |
| --- | --- |
| `VGPU_SASS=0` | PTX whenever the binary has it |
| `VGPU_SASS=1` | the SASS even where the executor lacks something (to find what) |
| `VGPU_SASS_LOG=1` | say which code each module runs, and why a fallback happened |
| `VGPU_SASS_REFUSE=<op>` | treat an op as unsupported (tests the fallback) |
| `VGPU_SASS_TRACE=<warp>` | every instruction that warp of block (0,0,0) runs, with its results |
| `VGPU_SASS_TRACE_KERNEL=<text>` | only in kernels whose name holds the text |
| `VGPU_KERNEL_DIGEST=<file>` | per launch, hashes (and NaN/Inf counts) of the memory its arguments reach: run once with `VGPU_SASS=0` and once without, and `diff` names the first kernel that differs |
| `VGPU_KERNEL_DIGEST_DUMP=<dir>` | with the digest, each allocation's bytes too, as `<dir>/<launch>.<address>`: `cmp -l` of two runs' files for the differing allocation names the bytes (which a hash cannot) |

## Coverage

Every generation from Turing to Blackwell runs, sm_75 through sm_120a:

| Target | Profiles | What its SASS adds, and runs here |
| --- | --- | --- |
| sm_75 | T4 | the base: integer/float ALU, memory, control, textures and surfaces, HMMA/IMMA |
| sm_80, sm_86, sm_89 | A100, A10, RTX 3060/3080 Ti, L4, L40S | LDGSTS, LDSM, REDUX, bf16/tf32/fp8/sparse MMA, sm_80's mbarriers (ATOMS.ARRIVE) |
| sm_90 | H100, H200, GH200 | clusters (UCGABAR, distributed shared memory, st.async, red.async), mbarriers (SYNCS), TMA (UTMA*, UBLK*: tile, im2col, multicast, reductions), warpgroup MMA (HGMMA/IGMMA/QGMMA/BGMMA), stmatrix, setmaxnreg, collectives |
| sm_100, sm_103 | B200, B300 | the uniform float datapath, tcgen05 (LDTM/STTM, UTC*MMA of every kind, UTCCP, UTCSHIFT, the Tensor Memory allocator), TMA gather4/scatter4 and CTA pairs, cluster launch control |
| sm_120 | RTX 5090 | sm_120's integer and float forms, block-scaled MMA |

Instructions the executor does not run, and so leave a kernel to its PTX:
`LDGMC` (multimem: it needs a multicast object, which `cuMulticastCreate` and the
rest do not make here, and the PTX engine has no multicast memory either), TMA's
`im2col::w` modes (nor does the PTX engine: the halo walk is in the ISA's figures
only), texture fetches with a LOD bias or clamp (`TEX.LB`, `.LC` -- ptxas emits
none from PTX, whose `tex` has no such operand) and per-texel gather offsets
(`TLD4.PTP`). A `WARPSYNC.COLLECTIVE` reached from different code paths of
one warp is refused when it happens: what the hardware does with the lanes then is not in
any document, and no PTX this was tried with produced one.

What does run, besides the plain forms: texture fetches with an offset (`.AOFFI`, the
packed register after the LOD), a depth reference (`.DC`), a residency predicate and
half-precision results (`.F16.RN`, before sm_90); `SUST.P`; and a cooperative launch
(`cudaLaunchAttributeCooperative`) of clusters, every block resident, each cluster with
its own barrier and distributed shared memory; and, on sm_120f, the packed integer
`VIADD`/`VIMNMX` forms (a lane at a time: 8- and 16-bit lanes, `.SAT`, negated A,
`.RELU`; derived from ptxas output for the PTX forms, compared with a host loop,
not checked against a card). `executes()` lists the opcodes the
executor runs; a decoded opcode outside it (`HMNMX2`, `F2IP`, `JMP`, `LDGMC` --
the first three no decoder produces) takes the kernel to its PTX, and the "not
implemented yet" faults at the foot of the instruction groups are unreachable for
the ones it lists.

Dynamic parallelism runs on SASS as on PTX. A program using it links CUDA's
device runtime library into its cubin; the loader resolves the relocations
that brings (`R_CUDA_G64`, function descriptors: a kernel's descriptor is its
code address), and the device runtime's public entry points
(`cuda_device_runtime_api.h`, listed under "Device runtime" below) run as
builtins in place of the library's code, whose own calls into the driver
(`__cuda_syscall_*`) are left to fail, by name, if anything reaches them. Both
engines decode their calls their own way -- a PTX call slot, a register pair --
and hand them to the half they share, `include/vgpu/exec/devrt.hpp`.

### Device runtime

A kernel can call what `cuda_device_runtime_api.h` declares, under the names
CDP2 compiles to (`__cudaCDP2<Name>`, CUDA 12 and 13) or, built with
`-DCUDA_FORCE_CDP1_IF_SUPPORTED`, CDP1's (`cuda<Name>`); both engines answer
both. What each call does and returns, errors included, was measured on an
RTX 3060 with CUDA 13.0's and 12.0's toolchains (the library's own symbols
are public; no NVIDIA binary was disassembled). The measurements:

| Call | On the card |
| --- | --- |
| `<<<>>>`, `cudaLaunchDeviceV2` | `cudaGetParameterBufferV2` always hands out a buffer, whatever the configuration; the launch validates: a zero dimension, a block of more than 1024 threads or a block `z` of 65, a grid `y` of 65536, shared memory past the opt-in maximum (49153 bytes until the host raised the kernel's `cudaFuncAttributeMaxDynamicSharedMemorySize`, 101376 after) and a block past `__launch_bounds__` all return `cudaErrorInvalidConfiguration` (9) and set the thread's last error. A buffer launched twice launches twice. |
| pending launches | the grids launched and not yet complete (queued, running, or waiting for children) are bounded by `cudaLimitDevRuntimePendingLaunchCount`, never below 32 (limits 0, 1, 10, 31 and 32 all stopped at 32); the launch past it returns `cudaErrorLaunchPendingCountExceeded` (69). A chain of grids each launching the next therefore stops at the limit plus one level (2049 grids by default); there is no nesting limit of 24 (that was the synchronization depth). Launching without ever waiting is fine as long as earlier grids finish. |
| streams | 0 (the block's implicit stream), 1, 2 (per thread), 3 (`cudaStreamTailLaunch`), 4 (`cudaStreamFireAndForget`) and created streams are valid; `5` to `8`, `0x10`, `0x100`, the graph streams and `~0` are `cudaErrorInvalidValue`. The card keeps no record that a stream was destroyed: launching into one, and destroying it again, succeed. `cudaStreamCreateWithFlags` takes 0 and `cudaStreamNonBlocking` only (2, 7 are 1); `cudaStreamDestroy` of 0, per-thread or garbage is 1. |
| events | `cudaEventCreateWithFlags` takes 2, 3, 10 and 11 (no timing is required; the other values of 0 to 15 are 1); recording into a stream that is none is 1; destroying twice is 0. A garbage event handle crashed the kernel, so none is tested. `cudaEventRecordWithFlags` is declared but not in the library: nvlink fails. |
| completion order | a grid is complete when everything it launched is. A tail launch runs after the launching grid and everything else it launched: fire-and-forget grids (before or after the tail launch), grids in named streams, their children, and the tail launches of its children; tail launches run in order. Launches into the block's default stream and a named stream keep their order; an event orders a stream after another. |
| `cudaMemcpyAsync`, `cudaMemcpy2DAsync`, `cudaMemcpy3DAsync` | device to device only: kinds 3 and `cudaMemcpyDefault` work, 0, 1, 2 and anything else are `cudaErrorInvalidMemcpyDirection` (21); a stream that is none is `cudaErrorUnknown` (999); null pointers with data to move are 1; 2D pitches smaller than the width are `cudaErrorInvalidPitchValue` (12). A copy runs in its stream's order, after the grids launched before it and before those after. Host pointers (pinned, managed) with `cudaMemcpyDefault` work too. |
| `cudaMemsetAsync`, 2D and 3D | return success, in stream order, and **write zero whatever the value is** (0x5a, 0xff, 1 and -1 all gave 0; the same on CUDA 12.0's library and driver 596.36). VirtualGPU writes the value: the documented behaviour, and what a program that passes 0 sees on both. A pitch smaller than the width is 1. |
| `cudaMalloc`, `cudaFree` | the device heap, as `malloc` and `free` (`cudaLimitMallocHeapSize`: 7 MiB of an 8 MiB heap, not 8, could be had). 0 bytes is 1 and a request the heap cannot hold is `cudaErrorMemoryAllocation` (2); either leaves `*p` alone. Memory a kernel allocated is there for the next kernel. `cudaFree(nullptr)` is 0, a `malloc`'d block may be freed by it, freeing a local or global variable is a quiet 0, a pointer to nowhere crashes the kernel, and freeing a block twice does too. |
| `cudaFuncGetAttributes` | fills the first seven fields (the three sizes, `maxThreadsPerBlock`, `numRegs`, `ptxVersion`, `binaryVersion`), the same as the host's call for the kernel, and leaves the rest of the struct as it was. A function that is not a kernel crashes. |
| `cudaDeviceGetAttribute` | attributes 1 to 148 give what the host's call gives for them, except 131 (1); 0 gives success and 0; 149 and up are 1. A device that does not exist is 101, and the value is left alone. |
| `cudaDeviceGetLimit` | what the host last set (`cudaDeviceSetLimit`), all seven limits. |
| `cudaDeviceGetCacheConfig` | the host's current configuration. `cudaDeviceGetSharedMemConfig` the bank size (4 bytes). |
| `cudaRuntimeGetVersion` | 6000, from CUDA 12.0's library and 13.0's alike. |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (and `WithFlags`, whose flags are ignored) | the same count as the host's; a block size of 0 or less writes 0 and returns 1. `cudaOccupancyMaxPotentialBlockSize`, which the headers build from these, works in a kernel. |
| `cudaGetErrorString`, `cudaGetErrorName` | the host's text, for every code (0 to 1099 compared), `unrecognized error code` past them, as a pointer into device memory. |
| last error | each thread's own: a failed call sets it, a successful one leaves it, `cudaGetLastError` clears it, and a kernel that ends with one set does not fail its launch. |
| `cudaDeviceSynchronize` | not in CDP2's header: CUDA 12 and 13 reject a kernel that calls it. Built for CDP1 it works on a part before Hopper (the module fails to load on sm_90 and later) and waits for the grids launched by the threads of the calling block, not another block's. |
| not callable | nvcc 13.0 rejects a kernel that calls `cudaStreamCreate`, `cudaStreamCreateWithPriority`, `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaEventCreate`, `cudaEventSynchronize`, `cudaEventElapsedTime`, the synchronous `cudaMemcpy` and `cudaMemset`, `cudaMallocAsync`, `cudaDeviceSetLimit`, `cudaFuncSetAttribute`, `cudaDeviceReset` and `cudaThreadSynchronize`: the device runtime's list is what the header declares. |
| `cudaGetParameterBuffer`, `cudaLaunchDevice` | the older pair works from C++: any buffer size, a null buffer for a kernel with no parameters, the same refusals as `cudaLaunchDeviceV2`. |

How VirtualGPU runs it, which differs from the card only where the card's
choice is not one a program may rely on:

- A child grid runs after its parent grid has finished, in launch order, parent
  block by parent block; its own children run (and its tail launches) before
  the next. All non-tail launches of a grid run before its tail launches. The
  count of pending launches is the card's, kept across the whole tree of
  grids; when it is full, the grids queued so far (the tail launches wait) run
  to completion at once, as the card's would finish while the parent went on,
  so a parent launching thousands of children runs. `cudaDeviceSynchronize`
  (CDP1) runs the calling block's queued grids.
- A parameter buffer is read when the launch is issued (the grid then
  runs from a copy), and a buffer can be launched once, where the card's
  launched it again.
- A copy or fill is queued with the grids, in order, and runs in its place.
  Device-side `cudaMemcpy3DAsync` takes pitched pointers only (a `cudaArray`
  is 1).
- Attributes, limits, the cache configuration and the error strings are
  asked of the library that launched the kernel (the runtime shim, or the
  driver shim for a static cudart), so a kernel sees what its host sees; a
  program that calls the device runtime from a bare engine launch gets
  `cudaErrorNotSupported` for those.

Kernel parameters past 4 KiB (CUDA 12.1 and later, up to 32764 bytes) come
with `KPARAM_INFO_V2` records and sit further into bank 0, past 0x8000, which
the 16-bit bank offsets hold as negative numbers: a bank address wraps in the
bank's 64 KiB.

The tensor map (`cuTensorMapEncodeTiled`) keeps its tile-mode fields where
NVIDIA's descriptor has them -- found with ptxas, one `tensormap.replace` field
at a time -- because SASS rewrites a map in place with plain stores.

## Member masks of the `*.sync` warp instructions

ptxas compiles `__ballot_sync`, `__shfl_sync`, `__match_any_sync`,
`__reduce_add_sync` and `__syncwarp` with a constant mask to the bare
instruction, which ignores the mask. A mask it cannot see through (a
lane-dependent value, a kernel argument) goes through code that checks it at
run time: `R2UR`/`REDUX.OR` and a `BRA.DIV` or `BRA.CONV` choose between the
bare instruction and, when the lanes name different masks, one `WARPSYNC` and
instruction per distinct mask, each over the lanes that named it. A thread its
own mask leaves out traps there ("an illegal instruction was encountered",
715; `__ballot_sync((1u << lane) - 1, p)`, an exclusive prefix, is the usual
way in, and is the HeCBench `bscan` benchmark). Both engines do what an RTX
3060 does, as `e2e_sass_archs`' `sync_masks` pins; the PTX engine reads the
operand to tell the constant from the register (and ptxas's constant
propagation, for a register set once from an immediate). A shuffle from a lane
that is not running it reads 0, not the lane's register. The SASS executor's
`BRA.DIV`/`BRA.CONV` take the whole group when the guard holds for some lanes
of it and not others, and `WARPSYNC` lets out one mask's lanes at a time.

## Tests

- `test_sass_decode`: every instruction in `nvidia/tests/data/sass/<arch>.txt`
  (sm_75 to sm_120a; encoding and nvdisasm's text, gathered by
  `nvidia/tools/sass-corpora.sh` from this repository's CUDA sources,
  PyTorch's CUDA code and the probes in `nvidia/tests/data/sass/probes`, which
  cover each instruction family's forms) decodes and prints back exactly as
  nvdisasm printed it.
- `e2e_sass_path`: which code a binary runs -- SASS by default, PTX as the
  fallback, the overrides -- and that both give the right answer.
- `e2e_sass_archs`: twenty programs built for each generation's SASS, sm_75 to
  sm_120, plus Hopper's (wgmma, TMA, tensor maps, stmatrix) for sm_90a and
  Blackwell's tensor core for sm_100a, each run by default -- checked to be
  running its SASS -- and on its PTX; the two must agree.
  `nvidia/tests/e2e/sass_archs.cu` keeps the forms that once ran wrong. The
  dynamic-parallelism programs among them (`dynamic_parallelism`,
  `cdp_device_api`, `cdp1_device_sync` -- built for CDP1, sm_75 to sm_89 --
  and `rdc_device_api`) are built `-rdc=true` with cudadevrt.
  `cdp_device_api` is the device runtime's check: every call above, against
  what the card returned. `e2e_device_runtime` runs it and `cdp1_device_sync`
  alone on four generations.
- Every other CUDA end-to-end test runs on SASS wherever its binary carries
  it, which is the default now.
