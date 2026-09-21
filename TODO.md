# TODO / status

Updated: 2026-09-01 (rev 4). See ARCHITECTURE.md for the design behind these.

## Implemented (tested)

- M0 build system: CMake + zero-dep C++20; `scripts/build.sh`, `scripts/test.sh`
- M1 profiles: a10/a100/h100/h200/b200 (placeholders, `verified: false`);
  `vgpu list-gpus`, `vgpu info --gpu <id> [--json]`
- M2 runtime core: devices, primary-context model, sparse virtual VRAM,
  H2D/D2H/D2D, OOB/UAF/double-free/interior-free/misalignment diagnostics
- M4 PTX parser: ld/st(param/global), mov, cvta, add/sub/mul/min/max/div/rem,
  and/or/xor/shl/shr, mad.lo, fma, mul.wide, setp, selp, predication (@/@!),
  bra, bar.sync 0, ret/exit; sregs tid/ntid/ctaid/nctaid/laneid;
  0f/0d float literals; precise unsupported-PTX errors
- M5 SIMT interpreter: 32-lane warps, mask divergence + reconvergence stack,
  functional cross-warp bar.sync, deterministic round-robin scheduler,
  step-budget infinite-loop guard, per-lane fault context
- M6 vectorAdd end-to-end: `vgpu demo vectoradd` (exact verification)
- M3 driver API: libvgpucuda.so — cuInit, version, device get/count/name/
  totalmem/attributes/CC, ctx create/destroy/set/get/sync, primary ctx
  retain/release, mem alloc/free/HtoD/DtoH/DtoD/getinfo, moduleLoadData(Ex)/
  unload/getFunction, cuLaunchKernel(kernelParams), error name/string;
  VGPU_GPU / VGPU_DEVICE_COUNT / VGPU_QUIET
- M7 (real): unmodified nvcc-compiled CUDA apps run on VirtualGPU via
  libvgpucudart (CUDA Runtime API + nvcc host-registration ABI +
  cuLibrary/cuKernel + fatbin PTX extraction incl. zstd). Verified with the
  external C11 driver-API harness AND an nvcc-compiled vectorAdd e2e test.
- Thrust and CUB run unmodified, and are under test: thrust::sort/reduce/
  inclusive_scan, and CUB's device-level DeviceReduce, DeviceScan (decoupled
  look-back, so it depends on ordering across blocks) and DeviceRadixSort. A
  radix sort is a tuned multi-kernel pipeline with its own temporary storage
  and warp primitives throughout, so it exercises far more than a hand-written
  kernel does. Also verified by probe, not yet pinned by a test: streams and
  events with cross-stream waits, managed and pinned memory, pitched 2D
  allocation with cudaMemcpy2D, the async memory pool (cudaMallocAsync), the
  >48 KiB dynamic shared-memory opt-in, and occupancy queries.
- Pantheon workloads: the pantheongpu stress/diagnostics kernels run
  unmodified (idle, memory_read/write, galpat, march_test, memory_hammer,
  atomic/int/compute virus). memory_read differential-matches a physical RTX
  3060 including fault-injection + device printf. See docs/pantheon-workloads.md.
- PTX additions: cvt (int<->float, rounding modes), neg, abs, prmt.b32, not,
  shf funnel shifts, atomics (add/min/max/and/or/xor/exch/cas), vector
  ld/st.v2/v4, predicate logic (and/or/xor/not.pred), .local memory frames,
  module .global variables, aggregate by-value params, device printf (vprintf),
  transcendentals (ex2/lg2/sin/cos/sqrt/rsqrt/rcp/tanh), bfe/bfi/brev/popc/clz,
  mad.wide, mov pack/unpack, NaN-aware setp forms, inline-asm register locals.
- PTX found by probing what real toolchains emit rather than by reading the
  spec, which is how several of these stayed missing: `lop3` (ptxas fuses
  bitwise chains into it, so optimized PTX is dense with them), `red` (an
  atomic whose result is discarded -- what an unused atomicAdd() compiles to,
  which in a reduction is every call), `slct`, `testp`, `sad`/`vabsdiff`,
  `match.any/all.sync`, `mul24`, `szext`, `fns`, `bfind[.shiftamt]`,
  `elect.sync`, `isspacep`. Cache-management hints (`prefetch`,
  `createpolicy`, `applypriority`, `discard`) and the scheduling directives
  `griddepcontrol` and `setmaxnreg` are accepted and do nothing, each for a
  stated reason rather than a shrug.
- Half precision beyond f16x2: f16, bf16, f16x2 and bf16x2 arithmetic
  (add/sub/mul/fma/neg/min/max), the same four types on every transcendental,
  and atom/red.add on all of them. bf16 is a different decode, not a scaled
  f16 -- it carries f32's exponent range with a 7-bit mantissa.
- `min.NaN`/`max.NaN`, which propagate a NaN instead of returning the other
  operand. The plain forms follow fmin/fmax; the two disagree on exactly the
  inputs a kernel clamping to keep NaNs visible cares about.
- Separately compiled builds (`-rdc=true`). Such a build leaves the primary
  fatbin empty -- 16 bytes, just a header -- and hangs the real device code off
  the wrapper's fourth field as a NULL-terminated list of *relocatable*
  fatbins. Device linking would consume them, but with a PTX-only `-code` there
  is nothing for nvlink to link, so the pieces arrive still separate and the
  runtime puts them together: PTX from each is concatenated, cross-piece
  references resolve by name, and only the first piece's `.version`/`.target`
  header survives (it is the translation unit; the rest are libraries built for
  whatever the toolkit's default architecture was). An unresolved call is
  reported when it is *reached* rather than at load, because the device-runtime
  library declares functions the driver supplies and defines them nowhere.
  Verified on a genuine two-unit build: device functions and a `__constant__`
  defined in one translation unit, used from a kernel in another.
- `__constant__` and `__device__` variables reached from the host:
  `cudaMemcpyToSymbol`/`FromSymbol` (and the Async forms),
  `cudaGetSymbolAddress`/`Size`, fed by `__cudaRegisterVar`. On the kernel side
  `ld.const` is a global read of a range nothing writes -- the read-only-ness
  is a promise the program makes, not one this engine enforces.
- Non-inlined device functions (`.func`): a real call with its own register
  file, `.local` frame and path stack, so divergence inside a callee and
  recursion both work. Parameters and the return value travel as call slots
  rather than a parameter buffer, because each lane passes its own arguments.
  The callee runs to completion inside the caller's instruction, which is what
  makes recursion fall out of the host stack -- and means a warp does not yield
  mid-call, so a barrier inside a device function is refused by name rather
  than silently skipping the rest of the body. Structs and arrays pass and
  return by value: a call slot is a per-lane byte buffer, so `st.param
  [param0+8]` lands where it should. Indirect calls work too: device functions
  have addresses in a window of their own, an array global can be initialised
  with a list of symbols (`= {f, g, h}` -- a function-pointer table), and a
  call through a register resolves the address back to the function. All
  participating lanes must agree on the target; a divergent function pointer
  is refused rather than picking one body and running it for everyone.
- Builtins a kernel can call: `vprintf`, `__assertfail` (a failed `assert()`
  reports its message and source location, and `cudaErrorAssert`), and the
  device heap -- `malloc`/`free` from inside a kernel, backed by the same
  allocator `cudaMalloc` uses so a device allocation gets the same
  out-of-bounds and use-after-free checking. Capped at CUDA's default 8 MiB;
  memory allocated there is reachable from the host here and is not on a
  device, which is a permissive difference and recorded as one.
- The SIMD video instructions (`vadd4`, `vsub4`, `vabsdiff4`, `vmin4`,
  `vmax4`, `vavrg4` and the 2-way forms), `set` (setp's sibling that writes a
  value, where an integer destination gets all-ones for true and a float one
  gets 1.0), `atom.inc`/`.dec` (which wrap against the operand rather than
  counting), and `abs` on the half types.
- FP8: `cvt` between e4m3x2/e5m2x2 and f32/f16x2/bf16x2, with `.satfinite`.
  The two formats are not one shape with a different bias -- e4m3 spends its
  top exponent on ordinary numbers and has no infinity, so 448 is its largest
  finite value and 1000 saturates to it, while e5m2 is IEEE-shaped and
  represents 1000 as 1024.
- `mbarrier` (init/inval/arrive/arrive_drop/test_wait/try_wait[.parity]/
  pending_count) and `cp.async.mbarrier.arrive`: the split barrier that
  cuda::barrier and cuda::pipeline are built on. Nothing blocks -- a wait is a
  predicate and the kernel spins, and an incomplete wait yields its scheduler
  turn so the warps it is waiting for can run.
- Shared memory: static + dynamic (extern) .shared, per-block zeroed frames,
  ld/st/atom.shared, correct space-relative addressing (cvta to/from generic).
- Warp shuffles (shfl.sync up/down/bfly/idx, + predicate output) and
  vote/ballot.
- Tensor cores: wmma.load.{a,b,c}, wmma.mma m16n16k16 with f16 or bf16 inputs
  and an f32 accumulator (row/col layouts), wmma.store.d -- so `nvcuda::wmma`'s
  load_matrix_sync/mma_sync/store_matrix_sync all work and a 16x16x16 GEMM
  through the public API matches a host reference for both B layouts and both
  element types -- and tf32's m16n16k8, whose A is 16x8 and B is 8x16, so its
  load computes the address from the layout directly rather than relying on the
  cancellation the square shapes get for free. The rectangular m8n32k16 and
  m32n8k16 variants are still refused by name;
  ldmatrix.m8n8.x{1,2,4}[.trans], mma.sync.m16n8k{8,16,32} over f16/bf16/tf32/
  s8, and movmatrix.m8n8.trans (the register-only transpose).
- Asynchronous copy: cp.async.{ca,cg} with commit_group / wait_group / wait_all
  and the src-size zero-fill form. The copy is deferred until the wait rather
  than performed on the spot, so a kernel that reads its destination early sees
  what the hardware would, not what a synchronous copy would have hidden.
- Warp membership: %lanemask_{eq,lt,le,gt,ge}, %warpid, activemask, bar.red,
  redux.sync. An undeclared %name is now reported as a special register this
  engine does not have rather than treated as a register nothing has written.
- The rest of the special-register set a kernel is likely to read: %smid and
  %nsmid (blocks are placed round robin over the profile's SM count -- a real
  placement, and what a persistent kernel needs to partition work), %gridid,
  %dynamic_smem_size, %total_smem_size, and the clock family
  (%clock, %clock_hi, %clock64, %globaltimer{,_lo,_hi}).

  **The clock family is a counter, not a time.** There is no timing model here,
  and inventing a number that looked like nanoseconds would be the same mistake
  as reporting a cache hit rate. What exists is a deterministic count of
  instructions issued by the block, which is monotonic -- and that is the only
  property most kernels use these for: spin-with-a-deadline and exponential
  backoff need the value to *advance*, not to be accurate. So a kernel that
  waits on %clock64 terminates, and a kernel that measures with it gets a
  reproducible number that is not a duration. Refusing the register instead
  failed the whole kernel over something it read only to decide when to stop
  waiting.

  Still refused, each by name: the thread-block cluster registers (%clusterid,
  %cluster_ctaid, %cluster_ctarank, %is_explicit_cluster), which need a cluster
  concept the scheduler does not have; %pm0-%pm7, which are hardware
  performance-monitor counters with no defined value unless a profiler set
  them; and %current_graph_exec.
- f16 (software IEEE binary16) and packed f16x2 arithmetic.
- CUDA Graphs: real stream capture -> record -> replay.
- Multi-GPU: peer access queries and cudaMemcpyPeer(Async) across virtual
  devices (all_reduce and p2p_thrasher take their real peer-DMA paths).
- Configurable virtual VRAM (VGPU_VRAM_MB); VGPU_TRACE coverage-growth logging.

## Hardware characterization

`nvidia/tools/characterize.cu` + `nvidia/tools/characterize-telemetry.sh` read a physical
device and emit a profile with `verified: true`.
`nvidia/tools/characterize-cloud.sh <instance-type> <region>` does the whole thing on
a rented GPU -- launch, characterize, capture conformance references, and
terminate (termination is registered before launch and then confirmed, because
an instance left running bills by the hour).
`nvidia/tools/compare-profile.py` diffs a measured profile against the one in the
tree, so corrections are visible rather than silently applied.

Verified against real hardware, eleven devices across seven architectures --
including the first AMD part:

| profile | device | how |
| --- | --- | --- |
| `nvidia/rtx3060` | RTX 3060 (sm_86) | local |
| `nvidia/a10` | A10 (sm_86) | Lambda `gpu_1x_a10` |
| `nvidia/a100-sxm4-40gb` | A100 SXM4 40GB (sm_80) | Lambda `gpu_1x_a100_sxm4` |
| `nvidia/h100` | H100 SXM5 80GB (sm_90) | Lambda `gpu_1x_h100_sxm5` |
| `nvidia/gh200-480gb` | GH200 480GB (sm_90, Grace) | Lambda `gpu_1x_gh200` |
| `nvidia/h100-pcie` | H100 80GB PCIe (sm_90) | Lambda `gpu_1x_h100_pcie` |
| `nvidia/t4` | Tesla T4 (sm_75, Turing) | EC2 `g4dn.xlarge` |
| `nvidia/a10g` | A10G (sm_86) | EC2 `g5.xlarge` |
| `nvidia/l4` | L4 (sm_89, Ada Lovelace, AD104) | EC2 `g6.xlarge` |
| `nvidia/l40s` | L40S (sm_89, Ada Lovelace, AD102) | EC2 `g6e.2xlarge` |
| `amd/mi325x` | MI325X (gfx942, CDNA3) | DigitalOcean `gpu-mi325x1-256gb` |

All ten NVIDIA parts match the physical device on **512 conformance values each** -- the
same binary run on hardware and on VirtualGPU, diffed.

**The AMD one is discovery, not execution.** `amd/mi325x` describes a real
MI325X -- 304 CUs, 64-lane wavefronts, 64 KiB LDS, gfx942 -- and nothing can
run on it yet, because the interpreter's warp is 32 lanes wide. The profile is
deliberately ahead of the engine rather than rounded to fit it.

It also cost three droplets to get, and two of those were avoidable. The first
two attempts compiled a HIP program on the rented machine and failed
identically: DigitalOcean's AMD image ships `hipcc` but not the HIP development
headers, so `hip/hip_runtime.h` exists nowhere under `/opt/rocm`. The third run
dumped `rocminfo`, which was installed all along and reports everything the
schema needs, and every subsequent iteration was free.
`amd/tools/rocminfo-to-profile.py` parses it at home for that reason.

MI300X was the intended target and is not launchable: a create was attempted in
all sixteen available DigitalOcean regions and every one answered "Size is not
available in this region". MI325X is the same gfx942 CDNA3 target, so only
`vram_bytes` differs between them.

Ada was the last architecture gap in the NVIDIA range. It stayed open for a
while because capacity and quota were both against it: us-east-1 had no L4
capacity when it was first tried, and the G-instance vCPU quota is still zero
in every region except us-east-1. It launched on the third availability zone
the script tried, which is the reason that loop exists.

**H200 is a special case worth naming.** It is the same GH100 die as the H100
SXM5 at the same compute capability, so every field the execution model
consults -- the whole limits block, the features, the SM count -- is inherited
from the *verified* `nvidia/h100`, not from a datasheet. Diff the two profiles
and the functional part is identical; only `vram_bytes`, `mem_clock_max_mhz`
and `pci_device_id` differ, and of those only the first is functional.

It stays `verified: false` anyway, and `vram_bytes` is the reason. That is the
one field that differs between two *physically identical* cards, so the single
functional value not inherited is also the single least inferable one.
Inheriting the rest correctly does not make it measured. Lambda offers no H200,
so closing this means EC2 `p5en.48xlarge` -- eight of them, which is why it is
still open.

**`vram_bytes` is a property of a configuration, not of a model.** Two A10s
characterized months apart differ by 1.5 GiB, which is the ECC reservation:
one had ECC on and the other off. Two H100 SXM5s differ by 10 MiB of
driver-reserved memory. Both readings in each pair are correct, so
`nvidia/tools/compare-profile.py` reports this field separately rather than as a
correction to apply. Matching a specific device exactly is what `VGPU_VRAM_MB`
is for.

The A10G is worth a profile of its own rather than aliasing the A10: same
architecture and compute capability, but 80 SMs against 72 and a 300 W cap
against 150. SM count times blocks-per-SM is what decides how a library splits
work, so treating them as one part would report the wrong occupancy ceiling
for every kernel.

What characterization corrected in the documentation-derived placeholders:
- A10 `vram_bytes` 25769803776 -> 23696375808 (datasheet "24 GB"; the device
  reports 22.07 GiB) and `temperature_max_c` 85 -> 98.
- H100 `vram_bytes` 85899345920 -> 85028896768.
Clocks, power caps, SM counts, shared-memory limits and PCI ids were already
right. Capacity values being wrong is exactly what this process is for.

Still documentation-derived: H200, B200, and all three AMD profiles. B200 had
no Lambda capacity; AMD parts are not offered there. The A100 80GB was read
from a physical device on an 8x instance.

## Register and occupancy modeling

PTX declares *virtual* registers, so counting declarations says nothing about
what a thread occupies. `src/ptx/regalloc.cpp` solves liveness over the
control-flow graph to a fixed point and takes the peak, counts a 64-bit value
as a register pair as the hardware does, keeps predicates in their own file,
and rounds to the allocation granularity.

That count is functional, not decorative:
- a block needing more registers than the device allows fails with
  "too many resources requested for launch", as on hardware;
- `__launch_bounds__` (`.maxntid` / `.reqntid`) is parsed and enforced;
- `cudaFuncGetAttributes` reports real `numRegs` and `localSizeBytes`, and
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` does the standard occupancy
  calculation instead of returning a placeholder.

Checked against a physical RTX 3060: a simple kernel reports **8 registers and
6 blocks/SM on both**. Measured against `ptxas -v` across the pantheon kernels,
this analysis lands a little *under* the real allocation -- 8 vs 14, 16 vs 20,
24 vs 26, 16 vs 24 -- because ptxas keeps values live longer than the data flow
requires in order to hide latency. So treat it as a lower bound on what a
thread needs: the launch refusal only fires when a kernel is genuinely
impossible, and the occupancy figure is optimistic by the same margin.

It used to err the other way, and far harder. Approximating a live range as
first-definition-to-last-use and then extending everything that touched a loop
across the whole loop body made every value in a grid-stride kernel look
simultaneously live: 328 registers for a kernel ptxas compiles into 14, and 10
of the 46 pantheon workloads refused to launch. The lesson is that a
conservative estimate is only safe while it stays under the hardware limit --
past that it stops being caution and starts being a false negative.

## Ecosystem tools that work today

- **pynvml** and anything built on it (nvitop, gpustat, monitoring agents,
  framework memory queries) — verified against a 4-GPU virtual rack.
- **lspci**, via generated PCI configuration space.
- **nvidia-smi / rocm-smi / rocm_agent_enumerator** — supplied by VirtualGPU
  (see docs/telemetry.md for why the stock nvidia-smi binary cannot be used).

**Vendor libraries.** cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE,
cuSOLVER, NCCL, NVRTC, NPP and nvJPEG are implemented under their real sonames, each verified
against NVIDIA's own library on a physical GPU: cuDNN, cuFFT and cuSPARSE are
bit-identical on every value the conformance suite reports, cuSOLVER on
everything but one f32 eigenvalue, and NCCL on all 24 values at two ranks
across two physical GPUs. The math runs on the host rather than through the
interpreter, because a vendor library is not user code — see nvidia/docs/libraries.md
for the boundary, the per-library scope, and what each one deliberately refuses.

NVRTC works by invoking the toolkit's own nvcc, which runs on the host and
needs no GPU, so kernels compiled through NVRTC reach the interpreter through
the driver API like any other PTX. The JIT frameworks are a separate question,
because none of them uses NVRTC:

- **Numba works.** It talks to `libcuda` directly and never loads a cudart. It
  compiles Python to PTX itself and assembles it through `cuLinkCreate` /
  `cuLinkAddData` / `cuLinkComplete`, which VirtualGPU implements by merging
  the PTX inputs and returning the merged text as the completed image -- the
  module loader consumes PTX, so there is no cubin to emit. Verified on shared
  memory with block reductions, atomics, 2D grids, math intrinsics, a tiled
  matmul, `shfl_up_sync` scans and streams.
- **Triton works, with a one-line hook.** It compiles all the way to PTX and
  then shells out to `ptxas` for a cubin, which is the one artifact in its
  pipeline VirtualGPU cannot load. `nvidia/tools/vgpu_triton.py` ends the pipeline at
  PTX using `knobs.runtime.add_stages_inspection_hook`, Triton's own extension
  point. Verified on a fused softmax, a `tl.dot` matmul (real `mma.sync` and
  `ldmatrix`) and an atomic reduction.
- **CuPy does not work.** It statically links the CUDA runtime rather than
  loading `libcudart.so`, so `LD_LIBRARY_PATH` never reaches it; its embedded
  runtime asks the driver for an export table and fails at
  `getDeviceCount()` with `cudaErrorSoftwareValidityNotEstablished` long
  before any kernel is compiled. Unblocking it needs the `cuGetExportTable`
  work below, not anything in the interpreter.

Multi-GPU is verified against real hardware in four places: the local two-GPU
box, and rented 2x H100 SXM5, 4x H100 SXM5 and 8x A100 80GB instances. All
thirteen conformance suites match on every one of them, with NCCL compared
against NVIDIA's libnccl at two, four and eight ranks respectively. The 2x
instance runs CUDA 12.8, so it also covers the older toolkit's LZ4 fatbins,
its cudaGetDeviceProperties_v2 spelling and its soname majors; the 8x is
sm_80, a second architecture.

Not yet: `nvidia-smi topo -m`, DCGM. PyTorch also ships thousands of its own kernels,
which would run on the interpreter, so `import torch` finding a usable GPU is
still a separate question from library coverage.

## A race the simulator found in llama.cpp

`FLASH_ATTN_EXT` runs about 2950 cases against the CPU backend. Fourteen fail,
all at `hsk=192, hsv=128` with a batch above one -- the asymmetric head-size
shape MLA models use -- by margins around 0.1 against a 5e-4 tolerance.

The cause is upstream, in ggml's `flash_attn_ext_f16` mma kernel:

- `flash_attn_ext_f16_process_tile` writes `tile_Q` (shared) at the top, and
  its first `__syncthreads()` comes *after* those writes.
- The matching sync at the *end* of the function is conditional:
  `if (np > 1) __syncthreads();`.
- The caller loop calls `process_tile` repeatedly with no barrier between
  calls.

So when `np == 1` and one block processes more than one tile, the next tile's
writes to `tile_Q` are not separated from the previous tile's reads by any
barrier. That is a write-after-read race on shared memory.

A block processes more than one tile only when stream-k splits work unevenly,
which is why the failures are so narrow. The evidence:

| configuration | blocks | tiles | result |
| --- | --- | --- | --- |
| stream-k disabled | -- | -- | 36/36 pass |
| grid 64, 128, 256 (even multiples of 32 tiles) | 1 tile per block | 32 | pass |
| grid 144 (not a multiple) | blocks span tile boundaries | 32 | 14 fail |
| grid 144, `__syncthreads()` made unconditional | -- | -- | 36/36 pass |

A barrier cannot change arithmetic, only ordering, so a result that changes
when one is added had an observable ordering. Real hardware tolerates it
because warps in a block advance together and the window is small; the
round-robin scheduler here does not, which is the whole point of running on
this rather than on a device.

Ruled out along the way, each with a test rather than an argument: the
attention math itself, `fastdiv`/`fastmodulo`, 64-bit division, widening
multiplies, integer conversions, unsigned compares, `vote.ballot`, the mma
fragment layout (checked element for element against a physical A10), buffer
sizing, and the write/read pairing between the two kernels -- a block wrote
`-0.387939 / 3.341256 / 5.117198` and the fixup read back exactly those.

Not fixed here, because it is not this project's bug to fix. What *is* this
project's to do is diagnose it rather than quietly return different numbers.

That detector now exists: `VGPU_RACE=1` finds this bug in a single run, naming
the kernel, the PTX line and the two warps involved. It is quiet on SOFT_MAX,
RMS_NORM, CUMSUM and all three pantheon workloads.

It also reported one candidate in `MUL_MAT`. **Checked, and it is neither a
second bug nor a false positive**: it is a real race whose outcome cannot
differ, and the detector was right to see it and wrong to stop on it.

The kernel is `mul_mat_q<GGML_TYPE_Q4_0, 16, need_check=true>`, and the write is
in `ggml_cuda_mmq_load_tiles_q4_0`. With `need_check` on, the tile loader clamps
out-of-range rows:

    if (fallback) { i = min(i, i_max); }
    const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;
    x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;

Rows past `i_max` all collapse onto `i_max`, so several warps recompute the same
source pointer, read the same block, and store the same bytes to the same shared
word. Two warps, no barrier between them, one address -- and the value is the
same whichever wins.

So the detector now compares the bytes. A store that leaves shared memory
exactly as it found it cannot be observed by anyone -- no reader and no other
writer can tell whether it happened before or after -- and does not turn a
conflicting access into a race. The write is still *recorded*, so a later store
of a different value is caught against it; only the report is suppressed.
`VGPU_RACE=2` reports these too, for the strict definition.

Three results, each with its control:

| Run | Before | After |
| --- | --- | --- |
| `MUL_MAT`, `VGPU_RACE=1` | aborts at case 49 | 1253/1253, silent |
| `MUL_MAT`, `VGPU_RACE=2` | aborts at case 49 | aborts at case 49, same word |
| `FLASH_ATTN_EXT`, `VGPU_RACE=1` | reports the real race | reports the real race |

The last row is the one that matters: the llama.cpp flash-attention race is a
write-read on values that genuinely differ, and it still fires. The change
narrows what counts as observable, not what the detector looks at.

## Known out of scope (not CUDA)

- `rt_virus` needs **OptiX** (NVIDIA's ray-tracing library, loaded from
  libnvoptix.so.1), a separate NVIDIA subsystem, not CUDA; emulating it is a
  distinct project. It fails with the vendor library's own error rather than a
  VirtualGPU error. (`media_enc_virus` runs: `libvgpunvenc` implements NVENC.)

## Partially implemented

- Divergence: min-PC reconvergence -- paths at the same pc merge and the lowest
  pc runs next, so bar.sync after a divergent region works. Not full IPDOM:
  irreducible control flow is not handled.
- cuCtxSetCurrent(NULL) pops rather than clearing a per-thread binding; the
  current-context stack is process-global, not thread-local.
- M7 proper: needs the CUDA *runtime* API shim + fatbin PTX extraction to run
  an unmodified nvcc-built binary (embedded-PTX driver-API apps work today).
- Reliability: ECC counts by location, retired pages, remapped rows and PCIe
  error counters are injected with `vgpu fault` and read by nvidia-smi, NVML
  and rocm-smi (docs/telemetry.md). `vgpu fault arm` delivers bit flips and
  ECC errors to a running kernel's loads, and a session's dmesg carries Xid
  and AER lines. `vgpu fault arm --hang` stalls a launch, and `vgpu fault
  throttle` makes clock-event reasons active with readings to match. ECC
  errors and Xids are NVML events (nvmlEventSetWait). `vgpu fault arm --on
  store|shared` puts faults on kernel stores and shared-memory loads. A
  kernel's own faults log Xid 31 (MMU fault) and Xid 13 (SM exception), and
  `vgpu fault stuck` fixes a bit at an address for every read. Atomics take
  what loads and stores take; `--on alu` puts silent flips in floating-point
  and matrix results. AMD: amd-smi (list, metric -e/-k, ras --cper)
  and amdgpu's dmesg lines. Not yet: faults on copies (a copy engine's own
  errors), PCIe AER and amdgpu RAS counts in sysfs, CPER record files,
  clock-change and power events.

## Not implemented (fails loudly, never silently)

This list was stale for a while, which is its own kind of wrong: it still named
textures, grid sync and host-pinned memory long after all three worked. A
roadmap that overstates what is missing misleads as much as one that overstates
what is done.

- PTX: `wgmma`, TMA (`cp.async.bulk`) and the cluster *memory* model,
  inline-asm-only instructions. (`mbarrier` is done -- init, inval, arrive,
  arrive_drop, test_wait, try_wait and pending_count, including the .parity
  form. Its transaction-counting modifiers, `expect_tx` and `complete_tx`, are
  refused by name: they exist to pair a barrier with a TMA copy, and with no
  `cp.async.bulk` to complete those bytes such a barrier would hang.)
  (Textures, surfaces and grid sync are done. The thread-block cluster
  scheduling level is done: `%clusterid`, `%nclusterid`, `%cluster_ctaid`,
  `%cluster_nctaid`, `%cluster_ctarank`, `%cluster_nctarank` and
  `%is_explicit_cluster` all report, driven by a cluster shape that comes from
  `.reqnctapercluster` or from `cudaLaunchAttributeClusterDimension`. What is
  still missing is the part that makes a cluster more than a numbering:
  distributed shared memory -- `.shared::cluster`, `mapa`, cluster barriers --
  which is a memory-model change, and stays refused rather than approximated.)
- Runtime: async copies, the virtual memory management API
  (cuMemAddressReserve…). (Managed memory and host-pinned memory are done.)
- Dynamic parallelism (a kernel launching a kernel). Taking a kernel's address
  in device code now says so by name instead of reporting an unknown symbol,
  which sent you looking for a typo in a name that was right there. Running it
  would need a child grid scheduled from inside the parent's instruction
  stream, which nothing here can do.
- Frontends: cubin/SASS loading, and AMD execution -- HIP runtime and the CDNA
  ISA. AMD *discovery* exists: `amd/tools/rocminfo-to-profile.py` reads a real
  MI325X and `amd/profiles/mi325x.yaml` is verified against one. The warp width
  is no longer the blocker: the interpreter is warp-width parametric, masks are
  64-bit, and a 64-lane profile launches and executes. What is missing now is
  the front-end -- HIP compiles to a GCN code object, not to PTX, so there is
  nothing yet to feed a 64-lane wavefront.
- Tooling: trace record/replay, conformance DB + compat scores. (`vgpu run`,
  `vgpu test --matrix`, shared-memory race detection, the random and
  adversarial schedulers, and fault injection are done.)

## Registers and counters: what is modelled, and what cannot be

Two questions that look alike and are not.

**Special registers are per *architecture*, not per GPU model.** `%tid`,
`%laneid`, `%smid` and `%clock64` are PTX ISA instructions, identical on a T4
and a B200. What varies is which exist -- the ten thread-block cluster
registers require sm_90 -- and what they return, which is already profile
driven (`%nsmid` is 132 on an H100 and 58 on an L4). The parser knows 46 names,
the cluster registers included. What they return is still approximate in three
places: `%smid` is round-robin rather than real placement, `%clock`, `%clock64`
and `%globaltimer` count instructions rather than time, and only `%envreg1-2`
are set. `%pm0`-`%pm7` stay refused on purpose: they are undefined unless a
profiler configured them, so a silent zero would be a confidently wrong answer.

**Performance counters here are not hardware counters, and so are the same on
every profile by construction.** `global_sectors` is computed from the
addresses every lane issued, not read from a monitor wired into one chip's
memory subsystem. That is why it is exact and reproducible where a device's is
sampled and moves between runs -- and why it does not vary by GPU. Real
hardware counter sets do differ per chip, which is a fact about physical
monitors rather than about programs.

So "support all counters for every GPU" is not achievable and not desirable:
the ones that differ per device are overwhelmingly timing-derived -- cycles,
stall reasons, hit rates, DRAM throughput -- and producing them would mean
inventing a timing model. `vgpu counters` prints both lists, what is reported
and what is not with the reason for each, because a gap that is written down is
a decision and a gap you find by its absence is a defect.

What was added rather than argued about: a per-opcode histogram
(`inst_by_opcode`) alongside the nine classes, since knowing a kernel is
memory-heavy is less useful than knowing it is memory-heavy because of
`ld.global.nc`. And the characterization scripts now capture `ncu
--query-metrics` from each physical device, so the boundary becomes per-device
data -- "an L4 exposes N metrics, this produces M" is checkable -- rather than
a claim to be taken on trust.

Still countable and still missing: register-spill traffic split out from
ordinary local traffic, and predicated-off lanes as a first-class number
(derivable today from `instructions * 32 - thread_instructions`).

## Performance

Two changes moved the needle most recently, both found by profiling rather than
by guessing:

- **The grid runs on every core.** Blocks are independent by definition, so
  each host thread takes a slice of them. `VGPU_THREADS` sets the count and
  defaults to the machine's; 1 restores the old strictly serial block order,
  which is what a kernel with a data race needs to stay reproducible. Device
  atomics take a stripe lock when the launch is threaded, because a fixed lane
  order is only atomic within one thread -- without it a 256-block atomicAdd
  test lost a third of its increments.
- **Chunk lookup is an array index, not a tree walk.** Sparse VRAM chunks were
  a std::map per allocation, so every scalar load and store walked a red-black
  tree. They are now a flat array of atomic pointers: O(1), no allocation on
  the read path, and lock-free for the threaded case.

Measured on an 8-core box: vectorAdd 2M elements 169 ms -> 50 ms, and the
pantheon memory_write workload at --duration 1 went from 49 s to 7.5 s.

Measure with `tools/bench.sh` (vectorAdd) and a register-heavy kernel.
Profile with gprof; guessing has been wrong every time so far.

Done (1.5x on memory-bound, 2.0x on ALU-bound):
- Register names are interned to dense ids at parse time; the interpreter
  indexes a flat register file instead of hashing a name per operand.
- Per-instruction scratch is no longer zero-initialized. `Lanes r{}` was a
  256-byte memset on *every* instruction -- about 6 GB of pointless memset in
  the ALU benchmark, and the single largest win found.
- Hot ALU paths choose the operation and width once per warp instead of
  per lane, and iterate only active lanes (`for_active`).
- Link-time optimization for Release builds.

Measured now: vectorAdd 2M elements ~145 ms; ~1.05G lane-ops of dense FMA
~0.85 s. Profiling says the remaining time is in the per-lane loops
themselves, which is where it should be for an interpreter.

Narrower lane storage was implemented and measured: registers are now split
into a 32-bit and a 64-bit file by declared width, with native 32-bit paths
for the hot integer, float and fma cases. Measured A/B against the previous
single 64-bit file:

| benchmark | wide | narrow |
| --- | --- | --- |
| vectorAdd 2M (memory-bound) | ~147 ms | ~148 ms |
| dense f32 FMA | ~0.86 s | ~0.83 s |
| 120 live registers | ~0.90 s | ~0.88 s |

So about 3% on register-heavy kernels and nothing on memory-bound ones --
far less than hoped. The reason is that the register file was already small
enough to sit in L1 (30 registers x 256 B is under 8 KB), so halving it does
not remove a bottleneck that was not there. It is kept because it is correct,
it halves per-warp register memory (which will matter as resident warp counts
grow), and a 32-bit register file with 64-bit values in pairs is what the
hardware actually does.

Next, in order of expected payoff:
1. **PTX -> internal IR -> LLVM JIT** (ARCHITECTURE.md D1). Profiling now puts
   the time in per-lane interpretation itself, which is exactly what a JIT
   removes. This is the real answer; further interpreter micro-optimization
   has hit diminishing returns.
2. Block-level parallelism across host threads, behind the scheduler
   abstraction so determinism is preserved.

A GPU still retires ~10^13 ops/s, so saturation-style stress tests are run at
reduced intensity via their own CLI knobs; see
scripts/run-pantheon-workloads.sh.

## Next milestones (order)

1. **Interpreter speed**: intern register names to dense indices at parse
   time (see Performance above) — the single biggest win available without
   the JIT.
2. **Scheduler: random and adversarial modes** -- done. `VGPU_SCHEDULER` picks
   between `deterministic`, `random` and `adversarial`, and
   `VGPU_SCHEDULER_SEED` makes the last two replayable: the same seed replays
   the same execution exactly, which is the property that makes a race
   fixable rather than merely observed.

   The half that mattered was not which warp runs next but *for how long*.
   Under the deterministic scheduler a warp runs from one barrier to the next
   without interruption, so two warps in the same epoch never interleave at
   all. The shared-memory detector still finds their conflict, because it
   reasons about epochs rather than orderings -- but a race through *global*
   memory produces no detector report, and with no interleaving it produces no
   wrong answer either. It simply does not appear. The new modes preempt
   mid-warp, and a test pins the difference: a non-atomic increment from two
   warps loses updates under the adversarial order and does not under the
   deterministic one.

   Still to do: an adversarial mode guided by what the warps are about to
   touch. The scheduler is not told about memory, so today it maximises
   switching and lets that do the work rather than aiming at a specific pair of
   conflicting accesses. Aiming would mean feeding the race detector's shadow
   state back into scheduling.
3. **Characterization harness v0**: run the same micro-tests on a physical
   GPU (bench/ rents them) and on virtual profiles, diff, and start flipping
   `verified` bits in the profiles.
4. **Static cudart hosting**: satisfy NVIDIA's undocumented driver export
   tables (cuGetExportTable dark API) so binaries built with the *default*
   (static) cudart also run without a `-cudart shared` rebuild. **Investigated
   and stopped, with a reason** -- see nvidia/docs/dark-api.md for the full bootstrap
   map. The static runtime asks for seven tables (three of them mandatory:
   without them the process aborts before `main`), queries the device through
   the ordinary documented API, and then fails its own validity self-test with
   `cudaErrorSoftwareValidityNotEstablished`. Three hypotheses were tested and
   eliminated: a missing table, unfilled out-parameters, and an incomplete
   device model. The decisive observation is that the runtime never performs a
   *functional* test -- no allocation, no launch, no result compared -- so the
   validity decision comes from the table interactions alone. Getting past it
   means producing exact values for slots with no specification, obtainable
   only from NVIDIA's internal headers or by disassembling their runtime.
   Neither is available to a clean-room project, so this stays where it is.
   Now has two more consumers. Nsight Systems collects through its own bundled
   CUPTI, loaded by absolute path from its install directory, and that copy
   reaches the driver the same way -- so `nsys` produces a report with OS
   runtime traces and no CUDA data. nvprof works, because its path goes through
   the public CUPTI this does implement. See nvidia/docs/cupti.md. CuPy is the other:
   it links the runtime statically and dies in the same place, at
   `getDeviceCount()`, before it compiles anything.
5. **More PTX as workloads demand it**: bf16, cp.async, mma.sync, the
   lane-mask family and the extended-precision carry family (`add.cc`/`addc`,
   `sub.cc`/`subc`, `mad.lo.cc`/`madc.hi`) are done -- driven by llama.cpp's
   flash attention, CUB's radix sort and Numba's 64-bit index arithmetic, which
   is the way to pick the next one too.

   **Grid sync is done**, and it turned out not to be a PTX gap at all.
   `cg::this_grid().sync()` compiles to no special instruction: it is an atomic
   increment of a counter in device memory and a spin on that counter, using
   ops the interpreter already had. What it needs is a *scheduler* that holds
   every block resident and interleaves them -- running blocks one at a time,
   which the programming model permits and this did, deadlocks the first block
   to arrive. Blocks are now suspendable, a cooperative launch round-robins
   them with a bounded turn so a spinning warp yields, and the barrier's
   workspace address is served through `%envreg1`/`%envreg2` the way the driver
   supplies it. `cudaLaunchCooperativeKernel` and `cuLaunchCooperativeKernel`
   refuse a grid too large to be resident, because such a kernel does not run
   slowly, it hangs. See docs/cooperative.md.

   **Textures and surfaces are done for the point-sampled case**, which is what
   the overwhelming majority of CUDA code uses: `tex.1d`/`tex.2d`/`tex.3d`,
   `suld`, `sust`, plus `cudaCreateTextureObject`, `cudaCreateSurfaceObject`,
   `cudaMallocArray` and the array copies. Backing memory can be linear
   (`tex1Dfetch`, which is how ML code uses textures -- as a cached load),
   pitched 2D, or a `cudaArray`. All four addressing modes, the integer and
   float channel kinds, `cudaReadModeNormalizedFloat`, and the
   absent-channel rule (0 for x/y/z, 1 for w) are implemented.

   Deliberately refused rather than approximated: `cudaFilterModeLinear`.
   Interpolation between texels is a documented weighted average, but hardware
   computes the weights in a fixed-point format with 8 fractional bits, so a
   float implementation would differ from the device in the low bits -- which
   is precisely what the differential testing here exists to catch. Also
   refused: mipmaps, layered and cubemap textures, sRGB, anisotropy, and the
   `.clamp`/`.zero` surface out-of-range policies. See nvidia/docs/textures.md.

   `wgmma` is what is left, and it is not workload-driven yet: nothing in
   llama.cpp or the pantheon suite uses it, and it needs TMA and mbarrier
   alongside it to be worth having.
