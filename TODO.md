# TODO / status

Updated: 2026-10-07 (rev 5). See ARCHITECTURE.md for the design behind these.

Rev 5 is an audit, not an addition: every claim below was checked against the tree at
`origin/main` 2ce1f45 (code, the CMake test list, the docs, the git log). The checks were
static -- a test named here was found in the tree and registered, not re-run for this
revision. Where an older bullet contradicted a newer one, the older one was corrected.
The status snapshot, the inventories and the gap register at the end are new in rev 5.

## Status at a glance (rev 5)

| Area | State | Where to look |
| --- | --- | --- |
| NVIDIA execution | PTX interpreter and a SASS executor (the default; sm_75 to sm_120a); both run unmodified nvcc programs through `libvgpucudart` / `libvgpucuda` | nvidia/docs/sass.md, "Implemented" |
| AMD execution | GCN/CDNA and RDNA executors (64- and 32-lane); unmodified hipcc programs from ROCm 6.4 to 7.2 on MI250X, MI300X, MI325X, MI350X, RX 6900 XT, RX 7900 XTX, RX 9070 XT | amd/README.md |
| Device profiles | 16 NVIDIA (12 read from hardware) and 7 AMD (1 read from hardware) | "Profile inventory" |
| NVIDIA libraries | cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE, cuSOLVER, NCCL, NVRTC, NPP, nvJPEG, cuDSS, cuSPARSELt, cuTENSOR, cuTensorNet, cuStateVec, cuFile, nvCOMP, NVSHMEM, nvJitLink, nvFatbin, NVENC | nvidia/docs/libraries.md |
| AMD libraries | rocBLAS, hipBLASLt, hipSPARSELt, MIOpen, rocPRIM/hipCUB, rocRAND, rocFFT, rocSPARSE, rocSOLVER, Composable Kernel, RCCL; ROCm SMI, AMD SMI, rocprofv3/rocprofiler-sdk shims | amd/README.md |
| Frameworks | PyTorch (CUDA on 12 whole-model checks; ROCm on six AMD GPUs), vLLM (ROCm), llama.cpp and Ollama (every NVIDIA and AMD profile), Triton, Numba, `torch.compile` | nvidia/docs/pytorch.md, "CI inventory" |
| Health, diagnostics, RAS | NVML (incl. the reads DCGM makes), nvidia-smi, rocm-smi, amd-smi, RVS's sensors, `vgpu fault`, BAR0/MMIO register model, memory-pattern tests | docs/telemetry.md, docs/registers.md |
| Not possible by design | hardware performance counters that need a timing model (`%pm0`-`%pm7`, stall reasons, hit rates), timing itself | "Registers and counters" |
| Blocked | static-cudart binaries (undocumented driver export tables), Nsight Systems collection, OptiX | "Next milestones" 4, "Known out of scope" |
| Open work | the "Gap register" at the end of this file | |

## Implemented (tested)

- M0 build system: CMake + zero-dep C++20; `scripts/build.sh`, `scripts/test.sh`
- M1 profiles: 16 NVIDIA and 7 AMD (see "Profile inventory" below; 12 NVIDIA and
  1 AMD are read from hardware); `vgpu list-gpus`, `vgpu info --gpu <id> [--json]`
  (`tests/unit/test_profile.cpp`)
- M2 runtime core: devices, primary-context model, sparse virtual VRAM,
  H2D/D2H/D2D, OOB/UAF/double-free/interior-free/misalignment diagnostics
- M4 PTX parser: ld/st(param/global), mov, cvta, add/sub/mul/min/max/div/rem,
  and/or/xor/shl/shr, mad.lo, fma, mul.wide, setp, selp, predication (@/@!),
  bra, bar.sync (all sixteen barriers since), ret/exit; sregs tid/ntid/ctaid/nctaid/laneid;
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
  look-back, so it depends on ordering across blocks) and DeviceRadixSort
  (`e2e_libraries`). A radix sort is a tuned multi-kernel pipeline with its own
  temporary storage and warp primitives throughout, so it exercises far more
  than a hand-written kernel does. Streams and events with cross-stream waits
  (`e2e_capture_streams`, `e2e_per_thread_stream`), managed and pinned memory
  (`e2e_runtime_conformance`, `e2e_managed_module`), pitched 2D allocation with
  cudaMemcpy2D (`e2e_graph_shapes`, `e2e_runtime_conformance`), the >48 KiB
  dynamic shared-memory opt-in (`shared_max`) and occupancy queries
  (`e2e_occupancy_rules`) were verified by probe first and are pinned by those
  tests now.
- Stream-ordered memory pools (cudaMallocAsync, cudaMemPoolCreate,
  cudaMallocFromPoolAsync, cudaFreeAsync, the pool attributes, TrimTo,
  cudaDeviceSetMemPool): a pool holds what is freed to it up to its release
  threshold and hands it out again, so an allocator asks the device once and
  reuses after that -- what PyTorch's async allocator and RMM are built on.
  The default threshold is 0, which the API documents as giving memory straight
  back, so caching starts when a program sets one. Used and reserved totals and
  their high-water marks are what a pool reports, and a mark resets by writing
  zero to it. e2e_mempool checks reuse (the same pointer comes back), the
  statistics, trimming, an explicit pool, and the refusals.
- Graphs built node by node, not only captured: cudaGraphCreate,
  cudaGraphAddKernelNode, AddMemcpyNode(1D), AddMemsetNode, AddEmptyNode,
  AddChildGraphNode, Add/RemoveDependencies, DestroyNode, the queries
  (GetNodes, GetRootNodes, GetEdges, NodeGetType, NodeGetDependencies,
  NodeGetDependentNodes), the per-node parameters, Clone, NodeFindInClone,
  cudaGraphExecKernelNodeSetParams and a real cudaGraphExecUpdate. A graph is a
  DAG now rather than a list: nodes carry their dependencies, a launch runs them
  in an order that respects them, and a dependency that would close a cycle is
  refused. Capture builds the same structure, each operation depending on the
  one before it, so a captured graph can be read back with the same calls.
  cudaGraphDebugDotPrint draws the real nodes and edges, where it used to print
  an empty graph. e2e_graph_build builds a diamond, launches it, changes a
  node's parameters on the instantiated graph without rebuilding, updates it
  from the graph, clones it and runs it as a child graph. A 2D fill or a pitched
  copy in a node used to be refused; both are nodes like any other now (see the
  graph-shapes bullet below, `e2e_graph_shapes`), and only a 2D fill with pitch 0
  is refused.
- The graph node types that are not device work: a host function
  (cudaGraphAddHostNode) runs on the CPU when the graph reaches it, and event
  record and wait nodes let a graph be timed and joined to work outside it. A
  node can be switched off in an instantiated graph (cudaGraphNodeSetEnabled) --
  kernel, memset and memcpy nodes, as documented -- so a program skips a step
  this launch instead of rebuilding the graph around it. The whole
  cudaGraphExec*NodeSetParams family is there now, so any node whose parameters
  can change without rebuilding can have them changed. cudaGraphExecUpdate
  compares which nodes each node depends on, not only how many: a graph rewired
  between two nodes used to pass as the same shape and have its parameters taken.
  e2e_graph_nodes covers all of it.
- Nsight Compute's command line over the simulator's counters: `vgpu ncu`, which
  is what `ncu` runs inside `vgpu shell`. NVIDIA's ncu cannot attach to a
  simulated driver -- it reaches the driver through undocumented interfaces and
  reports it "failed to connect" -- so Pantheon's --profile failed at its counter
  pass. This answers the commands Pantheon drives (--query-metrics, --csv,
  --metrics, --set, launch and kernel filters, --log-file) with the counters the
  engine records for every launch, written per launch to VGPU_COUNTERS_FILE. It
  lists and reports only metrics counted exactly under Nsight Compute's own
  definitions -- global and shared load/store instruction counts, global sectors
  and requests, bytes used per sector, shared bank conflicts, twelve in all --
  and names the rest as not reported rather than inventing them. Pantheon's
  --profile now completes: twelve metrics validated, the other seventy-four
  recorded as skipped in its manifest, and its timeline pass still uses NVIDIA's
  nsys, which produces a genuine report with OS-runtime rows and no CUDA ones.
  Along the way, vector loads and stores were counted as one element in the byte
  totals, a quarter of an ld.v4's traffic; they count every element now.
  e2e_ncu checks values against kernels whose metrics follow from their
  addresses.
- Streams are distinct. Every cudaStreamCreate used to return the same handle,
  0x1 -- cudaStreamLegacy -- so two streams compared equal, a capture begun on
  one captured the work sent to every other, and a kernel launched on a second
  stream during a capture never ran: it went into the first stream's graph. A
  stream is a record now and its address is its handle, and it answers for
  itself: cudaStreamGetFlags, GetPriority (clamped to the device's range, as
  documented), GetDevice, GetId (unique, including each thread's per-thread
  stream), and Get/Set/CopyAttributes. The default streams cannot be destroyed.
  e2e_stream_identity.
- cudaDeviceSetLimit and cudaDeviceGetLimit, per device, with the defaults and
  the rounding and clamping CUDA documents. The heap limit is real: malloc() in a
  kernel draws from a heap of cudaLimitMallocHeapSize bytes, per device, so a
  program that raises it gets the room and one past it gets null -- the engine
  had said this was not wired. The heap and printf limits refuse to change once
  a kernel that calls malloc/free or printf has launched, decided from the
  kernel's code. The stack size sets each thread's alloca stack (`e2e_stack_limit`);
  the printf-buffer size is recorded and reported but bounds nothing here.
  e2e_device_limits allocates up to a raised limit and one
  past it. The heap is one per device whichever engine a kernel runs on (the
  device's memory manager keeps it): a block one kernel allocates a later one
  may free, SASS or PTX, free() refunds the budget, and cudaDeviceReset empties
  it -- blocks leaked before a reset used to keep counting against the limit.
  e2e_device_heap, both engines and across them (an RTX 3060 passes it).
- A capture records everything a stream is given. Several entry points ran their
  work the moment they were called on a capturing stream, so the graph came back
  without it and every replay silently left it out: 2D copies and fills, copies
  to and from a symbol, peer copies, host functions (which ran once, at capture),
  graph launches, and stream-ordered allocation (a captured malloc/free pair
  freed the memory before the graph ever ran). Each is now what CUDA documents
  it becomes -- a copy or fill node with its shape, a host node, a child-graph
  node, an allocation or free node the graph owns -- and what CUDA does not allow
  under capture (a stream callback, synchronizing or querying the capturing
  stream) invalidates the capture with the error it documents. A captured
  cooperative or clustered launch keeps that on replay; it ran as a plain launch
  before, and grid.sync() trapped.
  Capture works across streams: a stream that waits on an event recorded in a
  capture joins it with its own position, so a fork and a join come back as a
  graph with two branches, and cudaStreamEndCapture refuses a fork left
  unjoined. An event recorded in a capture stands for work not yet run and
  cannot be queried (cudaErrorCapturedEvent); cudaEventRecordWithFlags with
  cudaEventRecordExternal is a record node that times the graph. The boundary
  rules are CUDA's, each with its documented error: a wait on an event from
  outside (isolation), one that would merge two captures, forking the legacy
  stream, capturing on it, ending a capture from a stream that only joined.
  Graph fill nodes with 2- and 4-byte elements wrote the value's low byte into
  every byte -- a 4-byte fill of 2 gave 0x02020202 -- and now write the value
  into every element, which a test of mine had asserted the wrong way round.
  Copy and fill nodes carry their full shape, so 2D fills and pitched 3D copies
  are nodes like any other, and read back exactly as they were given.
  Not yet: capture modes. Global and thread-local capture prohibit "potentially
  unsafe" calls (cudaMalloc, a synchronous copy) anywhere in the process; every
  capture here behaves as relaxed.
  Two checks guard these classes: lint_capture_coverage fails when an exported
  function that takes a stream neither consults capture nor says why it need
  not, and lint_shim_symbols fails when a shim library has a reference nothing
  can resolve -- which one of these changes had at first, declared in an
  anonymous namespace and defined outside it, and which only the Python-driven
  tests noticed. e2e_graph_shapes and
  e2e_capture_streams cover the behaviour.
- Splicing into a stream capture, which is what a library does when the stream
  it was handed turns out to be capturing: cudaStreamGetCaptureInfo reports the
  capture's own graph, the nodes the next operation will depend on, and an id
  unique to the capture sequence, and cudaStreamUpdateCaptureDependencies sets or
  extends that dependency set. A library adds its own nodes to the graph and says
  the capture continues from them, so one stream can capture a fork and a join.
  The query used to report an active capture with no graph and no dependencies,
  and under CUDA 13 -- which gave both calls an edge-data argument -- the plain
  cudaStreamGetCaptureInfo was not exported at all. The graph handed out during
  the capture is the one cudaStreamEndCapture returns, and the capture owns it:
  cudaGraphDestroy refuses it until then. cudaLaunchKernelEx honours
  cudaLaunchAttributeCooperative, which it used to drop, so a kernel calling
  grid.sync() through the extended launch form gets its grid barrier instead of
  trapping. e2e_capture_splice covers both.
- Memory a graph owns: cudaGraphAddMemAllocNode allocates when the graph reaches
  it and cudaGraphAddMemFreeNode frees it, so scratch space belongs to the graph
  rather than to the program for the graph's whole life. The address is fixed
  across instantiations and launches, as CUDA documents, because the address
  space is reserved when the node is built and only the physical memory behind it
  comes and goes -- which is what the mapping API (cuMemMap and friends) is for.
  cudaDeviceGetGraphMemAttribute reports what graphs hold: `used` between an
  allocation node running and the free that ends it, `reserved` for what the
  device has actually handed over, which outlives a free because the memory is
  kept for the next launch. cudaDeviceGraphMemTrim gives that back and the
  address survives it. The documented restrictions are enforced rather than
  ignored: such a graph cannot be cloned, cannot be a child of another graph,
  cannot have nodes or edges removed, has one instantiation at a time, and its
  allocation is freed once and in one graph. An allocation the graph does not free
  outlives the launch and the program frees it; with
  cudaGraphInstantiateFlagAutoFreeOnLaunch each launch frees what the last one
  left. e2e_graph_memory checks the address really does not move, including across
  a trim and across a free from outside the graph.
- Device memory shared between processes (cudaIpcGetMemHandle,
  cudaIpcOpenMemHandle, cudaIpcCloseMemHandle and the event handles): an
  exported allocation moves into a file of its own in the machine directory,
  mapped MAP_SHARED and still at the same device address, and the importing
  process maps the same file at an address of its own. Both then read and write
  the same bytes, kernels included -- which is what vLLM shares a KV cache
  through and what NCCL uses for peer buffers on one host. A process cannot
  import its own export, as CUDA documents. An event another process recorded is
  complete by the time its handle can be read, since every operation here
  finishes before its call returns, so a cross-process wait on it is satisfied
  rather than skipped. e2e_ipc runs two processes against one simulated machine:
  the importer sees what the exporter's kernel wrote, writes back, and the
  exporter sees that. Exporting to another *machine* is still refused; there is
  no fabric here.
- Managed-memory hints: cudaMemAdvise records what it is told -- read-mostly, a
  preferred location, which devices access a range -- and
  cudaMemRangeGetAttribute(s) reports it, answering with a value only where
  every byte of the range agrees, as CUDA does. Nothing migrates, because there
  is one physical copy of managed memory here, so the hints change no result;
  they are state a program sets and reads. Both calls now refuse memory that is
  not managed and a device that does not exist, where they used to return
  success for any pointer at all.
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
- PTX found by a differential probe -- 95 instruction forms run on an RTX 3060
  and on the simulator, the results compared -- which turned up both gaps
  and silent errors. Now implemented: `prmt`'s six modes; `setp`/`set` with
  `.and/.or/.xor` and setp's `p|q`; `mad24` (and `mul24.hi`, which took bits
  47:24 instead of 47:16); `add/sub.sat.s32`; `.sat` on f32 add/sub/mul/fma;
  `min/max.xorsign.abs`; `cvt` with `.relu`, `.rz`/`.satfinite` on the
  packed forms, and to tf32; `.f16x2` register declarations; `pmevent`.
  Fixed, having been accepted and ignored: fma/mad's `.rz/.rm/.rp`, `.sat`
  and `.ftz`; every half-precision modifier (`.sat`, `.ftz`, `.NaN`,
  `.xorsign.abs`, `.relu`, the rounding modes); the packed cvt's rounding
  mode. Also as the card does: f32 NaN results are 0x7fffffff; min/max take
  the other operand for a signalling NaN too and order -0 below +0; `.sat`
  sends -0 to +0; bf16 fma rounds once (a double-rounding tie went to even);
  `isspacep.global` is true for any address outside shared and local memory.
  e2e_ptx_forms hashes 127 variants against the card, and two held-out input
  sets match too. Left as they are, and why: `ex2.approx.f16x2`,
  `rcp.approx.ftz.f64` (the card returns only a 32-bit-accurate high word)
  and `sin.approx` of tiny inputs follow the documented error bounds, not the
  card's bits; ptxas miscompiles `szext` with a register width on sm_86 (the
  card extends from 0 bits), so the ISA's meaning is kept.
- A second sweep, 378 variants of the everyday instructions (integer and bit
  operations, f32/f64 with every rounding mode and .ftz, comparisons,
  conversions between every integer and float width, half precision),
  e2e_ptx_sweep. Fixed from it: sqrt/rcp.{rn,rz,rm,rp} ignored the rounding
  mode (and .ftz); f32 add/sub/mul/div/min/max ignored .ftz; testp classified
  f32 values as doubles (an f32 subnormal came out normal) and the card calls
  zero normal; neg/abs give the canonical NaN (f32 and halves) and leave an
  f64 NaN untouched; rem by zero is all ones, not the dividend; fns finds
  nothing from a base past 31 and returns 0 for an offset of INT_MIN; a NaN
  converted to an integer is 0 from f32/f16 into 32 bits or fewer and the
  destination's sign bit otherwise; cvt.f32.f16 of a NaN is 0x7fffffff and
  cvt.ftz.f64.f32 widens the canonical NaN; setp/set on f16x2 with p|q and
  with an integer destination.
- A third sweep, 153 variants of warp instructions (shfl in every mode and
  clamp, vote, match, redux, the lane masks, bar.red), atomics and reductions
  on global and shared memory (the value returned and the memory left),
  sub-word and vector memory, and 16/64-bit integer arithmetic:
  e2e_ptx_warp_mem. Fixed from it: atom.cas compared with c and stored b, so
  a matching CAS stored nothing (also its own PR against main); shfl.up
  bounded by c's clamp field as the ISA's maxLane, not minLane; match.all
  writes the member mask or 0; atom.add.f32 flushes subnormals and writes
  the canonical NaN; a 16-bit shift's amount is a whole .u32; bfe/bfi.64 take
  the whole position and length; bfe.s64 of all 64 bits kept (1 << 64 was
  undefined); cnot.
- A fourth sweep, 90 variants of the memory forms compilers emit (ld/st with
  every cache operator, .volatile and the ordering qualifiers and scopes,
  atomics with their semantics and scopes, fences and membar, generic
  addressing and cvta, isspacep, mbarrier, cp.async with its zero fill,
  ldmatrix, movmatrix): e2e_ptx_memory_forms. All matched but one:
  mbarrier.pending_count.b64 takes an arrival's state token (the ISA's form),
  which carries the count pending before that arrive; it was parsed as
  taking the barrier's address.
- Multiply-add contraction as the code generator performs it: a mul and the
  add or sub consuming its product, neither with a rounding modifier, run as
  one fma (src/ptx/contract.cpp). The PTX ISA allows it, ptxas does it, and
  an RTX 3060's results show when: f32 and f64, .ftz forms too, either
  operand order and both directions of sub; a product used only by adds and
  subs fuses into each of them, one used any other way into none; the first
  operand's product when both are; never across a branch. Found by
  PolyBench's ADI, whose `x - y*a` came out one ulp off in a third of its
  output and now matches the card bit for bit. VGPU_PTX_CONTRACT=0 turns it
  off.
- A fifth sweep, half precision: every f16, f16x2, bf16 and bf16x2 form of
  add/sub/mul/fma/min/max/neg/abs with every modifier, setp and set with
  every comparison (and f32's, with and without .ftz), and cvt between the
  half types and every integer width with each rounding mode and .sat --
  600 candidates, of which sm_86's ptxas takes 324, plus 56 f32 comparisons:
  e2e_ptx_half_forms, 380 variants, also matching on two held-out input
  sets. Found: .ftz on setp and set was dropped for every type, and on half
  neg/abs, so subnormal operands were compared and negated as themselves;
  half min/max left the sign of min(+0, -0) to std::fmin (the card orders
  -0 below +0, as for f32, which .ftz makes of every negative subnormal);
  cvt.f32.bf16 gave NaNs the canonical NaN where the card shifts the bits
  (a signalling NaN stays signalling); and cvt.f64.f16 dropped a NaN's sign
  and payload, which the card keeps, made quiet.
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
  A call pushes a frame onto the warp -- the caller's paths, registers and
  slots -- and the callee's last `ret` pops it, so a warp can stop in the
  middle of a device function and let the others run: `__syncthreads()`,
  `__syncthreads_count()` and named barriers inside functions nvcc did not
  inline work, which every `-G` build needs (e2e_device_function_barriers,
  built at -O3 with `__noinline__` and with -G; an RTX 3060 passes both).
  This used to run the callee to completion inside the call instruction and
  refuse a barrier there. A barrier inside a call that only some of the warp's
  lanes made is still refused: the others wait in the caller, so it could
  never complete. Structs and arrays pass and
  return by value: a call slot is a per-lane byte buffer, so `st.param
  [param0+8]` lands where it should. Indirect calls work too: device functions
  have addresses in a window of their own, an array global can be initialised
  with a list of symbols (`= {f, g, h}` -- a function-pointer table), and a
  call through a register resolves the address back to the function. Lanes
  may call different targets -- a virtual call over objects of different
  types: the lanes that share the first lane's target make the call, the
  rest split off at the call and take the next target, and the paths merge
  after it, each lane with its own return value
  (e2e_divergent_indirect_calls, virtual methods and a per-lane
  function-pointer table at -O3 and -G; an RTX 3060 passes both).
- Builtins a kernel can call: `vprintf`, `__assertfail` (a failed `assert()`
  reports its message and source location, and `cudaErrorAssert`), and the
  device heap -- `malloc`/`free` from inside a kernel, backed by the same
  allocator `cudaMalloc` uses so a device allocation gets the same
  out-of-bounds and use-after-free checking. The heap is per device and sized by
  `cudaLimitMallocHeapSize` (default 8 MiB; see the cudaDeviceSetLimit bullet,
  `e2e_device_limits`, `e2e_device_heap`). The host may not free a device-heap
  block: `cudaFree`, `cuMemFree` and `cuMemGetAddressRange` refuse it.
- The video instructions, all of them: the scalar `vadd`, `vsub`,
  `vabsdiff`, `vmin`, `vmax`, `vshl`, `vshr`, `vmad` and `vset` with operand
  selectors, `.sat`, the secondary `.add`/`.min`/`.max` and the destination
  merge, and the SIMD `vadd2/4`, `vsub2/4`, `vavrg2/4`, `vabsdiff2/4`,
  `vmin2/4`, `vmax2/4` and `vset2/4` with lane selectors, masks and `.add`.
  On sm_70+ ptxas emulates them, and the card departs from the ISA's
  pseudocode in places (vmad takes whole 32-bit operands as signed; the
  shifts wrap their intermediate at 34 bits), so the simulator does what an
  RTX 3060 does, over 652 variants (e2e_video_forms). `set` (setp's sibling that writes a
  value, where an integer destination gets all-ones for true and a float one
  gets 1.0), `atom.inc`/`.dec` (which wrap against the operand rather than
  counting), and `abs` on the half types.
- Narrow formats: `cvt` between e4m3x2/e5m2x2, the OCP MX e2m3x2/e3m2x2/e2m1x2,
  ue8m0x2 and s2f6x2 and f32/f16x2/bf16x2 (PTX ISA 9.7.10.24), with
  `.satfinite` (NaN to +MAX_NORM for the types without one), `.relu`,
  ue8m0's `.rz`/`.rp`, and the `.scaled::n2::ue8m0` factors of bf16x2 and
  s2f6x2. Refused by name: `.rs` for the x4 types (figures 41-42 do not
  say how a and b share their random bits), sm_107f's `.rz`, `.pzo`,
  `.scaled::n1` and ue5m3x2, and s2f6x2 from bf16x2 (its pseudocode and text
  disagree).
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
  cancellation the square shapes get for free. Since 2026-09-25 every other
  WMMA combination too -- f16 accumulators, the rectangular m8n32k16 and
  m32n8k16 shapes, s8/u8 (with .satfinite), f64 m8n8k4 with its rounding
  modes, s4/u4 m8n8k32 and b1 m8n8k128 (.xor.popc and .and.popc), and the
  optional stride. The fragment layout is unspecified by the ISA, so these
  use one of their own (the logical matrix, spread in order over lanes and
  registers); the combinations above keep theirs. wmma_types.cu checks all 21
  shape/type/layout combinations through mma.h, exactly;
  ldmatrix.m8n8.x{1,2,4}[.trans], the 8-bit .m8n16 and .m16n16.trans
  (fp6/fp4 expanded into the low bits of each byte, .s4 to .s8), and
  movmatrix.m8n8.trans (the
  register-only transpose). mma.sync in every form the ISA lists: Volta's m8n8k4 f16 (four
  products, any layouts, f16/f32 accumulators), f64 m8n8k4 and m16n8k4/8/16
  in each rounding mode, tf32, f16/bf16, s8/u8 and s4/u4 (mixed signedness,
  .satfinite), .b1 .and/.xor.popc, e4m3/e5m2, and sm_120's
  `.kind::f8f6f4` (fp6/fp4 in 8-bit containers) and block-scaled
  `.kind::mxf8f6f4`/`mxf4`/`mxf4nvf4` with their scale-data selectors
  (9.7.16.3, figures 46-48); and mma.sp in every type, shape and selector
  (f16/bf16, tf32 1:2, s8/u8, s4/u4 4:8 in pairs, fp8, and the sm_120
  kinds at m16n8k64/k128). A simulated RTX 5090 (nvidia/rtx5090) runs them.
  e2e_mma_forms runs the 123 forms an RTX 3060 has -- each fed fragments,
  its D hashed -- and compares with the hashes the GPU gave. Three things
  came out of the hardware rather than the ISA: tf32 inputs lose their low
  13 mantissa bits; .satfinite clamps after every 128 bits of K (twice in
  m16n8k32 int8 and m16n8k64 int4), not once at the end; and the 8- and
  4-bit sparse forms carry metadata a row to a lane (lane 4g + 2 * selector
  for row g, the next for g + 8, the pair after for chunks 8-15) where
  f16/bf16/tf32 carry four chunks of both rows. fp8 and the sm_90 f64 shapes,
  which the 3060 cannot run, are checked against a host GEMM. The
  accumulation order of a float sum is unspecified and not modelled: with
  inputs whose products round, m16n8k8 tf32 differs from the GPU in low bits.
  CUTLASS's 19 SM80 sparse GEMM tests pass.
  stmatrix.m8n8.x{1,2,4}[.trans] (and .m16n8.trans.b8) is the store counterpart of ldmatrix: the warp
  writes the 8x8 matrices its registers hold back to shared memory, which is how
  a kernel gets an mma result out of registers for the next stage. e2e_stmatrix
  checks where every element lands and that a fragment stored by one instruction
  and loaded by the other comes back unchanged.
- `atom.{exch,cas}.b128` (sm_90): 16 aligned bytes, the operands `.b128`
  register pairs, under the atomics' striped lock. `st.bulk` (sm_100): zeroes
  shared memory, a multiple of 8 bytes up to 16 MiB. `istypep`: false for every
  handle, as an RTX 3060 answers for texture and surface objects; there are no
  `.texref` variables since CUDA 12.
- The per-thread stack (PTX ISA 9.7.19): `alloca`, `stacksave` and
  `stackrestore`, which nvcc emits for `alloca()` in device code. Each thread
  has a stack of `LaunchConfig::stack_bytes` (cudaLimitStackSize, 1 KiB by
  default) that alloca carves down from. A device function's allocas are freed
  when it returns. Overflow, touching memory below the stack pointer (never
  allocated, or freed by stackrestore) and restoring to anything but a saved
  pointer are reported. e2e_alloca_stack matches an RTX 3060's results.
- Asynchronous copy: cp.async.{ca,cg} with commit_group / wait_group / wait_all
  and the src-size zero-fill form; an empty group counts toward wait_group N. The copy is deferred until the wait rather
  than performed on the spot, so a kernel that reads its destination early sees
  what the hardware would, not what a synchronous copy would have hidden.
- Warp membership: %lanemask_{eq,lt,le,gt,ge}, %warpid, activemask, bar.red,
  redux.sync. An undeclared %name is now reported as a special register this
  engine does not have rather than treated as a register nothing has written.
- `bar.red` on all sixteen barriers, with or without a thread count, with the
  barrier and count in registers, and guarded (the warp must agree, as for
  `bar.sync`). With a count, only the warps that arrive share the answer, so
  two groups can reduce on barriers of their own at once. An RTX 3060 runs
  the same test (e2e_bar_red_named). Writing it, the simulator caught a race
  in the test itself: one barrier reused by two groups of warps that were not
  ordered against each other, which the card had passed by timing luck.
- The driver's reserved shared memory (sm_80 and later) and the
  `%reserved_smem_offset_{begin,end,cap,0,1}` registers that locate it.
  cooperative_groups keeps the barriers and exchange slots of tiles of more
  than one warp (`tiled_partition<64>` and up) there. The layout is the one an
  RTX 3060 reports for kernels from none to 48 KiB of shared memory: the
  region starts at `%total_smem_size` (the kernel's own shared memory in
  128-byte allocation units, as cuda_occupancy.h gives them; the register now
  reports it rounded, as the card does), `end` is 288 bytes on and `cap` 1 KiB
  on. Only a module that reads the registers gets the region backed, so every
  other kernel's shared overruns are still caught. Before compute capability
  8.0 the registers are refused. Along the way: a warp that spins on a flag
  another warp of its block sets (which is how those tiles synchronise) spun
  forever under the deterministic scheduler, whose turns ended only at
  barriers; a turn longer than 65,536 instructions now ends at the next
  backward branch. e2e_cg_multi_warp_tiles passes on the card and on the
  simulator under all three schedulers.
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

  (The cluster registers, %aggr_smem_size and %current_graph_exec have
  since been implemented; what stays refused is %pm0-%pm7 -- see
  "Registers and counters" below for the whole list.)
- Device-side graph launch (CUDA 12): a kernel's cudaGraphLaunch of a graph
  instantiated with cudaGraphInstantiateFlagDeviceLaunch, into
  cudaStreamGraphFireAndForget, cudaStreamGraphTailLaunch or
  cudaStreamGraphFireAndForgetAsSibling, with the programming guide's
  execution environments: fire-and-forget and sibling graphs run when the
  launching kernel is done, tail launches once its graph and their children
  are (a tail graph's own tails before those queued ahead of it), all before
  the host launch completes. A device launch runs the graph as last uploaded.
  %current_graph_exec reads the device graph's handle on both engines (SASS:
  the bank-0 word ptxas loads, 0x120/0x130/0x190/0x2e8 by generation). Every
  refusal is the RTX 3060's: only the three graph streams; a null, not
  uploaded, running or already queued graph; 120 fire-and-forget launches per
  execution, 255 queued tails; one tail self-launch at a time -- all
  cudaErrorInvalidValue -- a handle not instantiated for device launch faults,
  and a kernel that calls cudaGraphLaunch launched outside a graph is refused
  with cudaErrorNotSupported. Instantiation for device launch refuses what the
  card refuses (empty graphs, empty/host/event nodes, dynamic parallelism,
  pageable or managed copies, AutoFreeOnLaunch). e2e_device_graph_launch,
  both engines; it passes on the card.
- nvJitLink links relocatable SASS (-rdc/-dc cubins, objects, libraries) into
  a linked cubin that runs as SASS (nvidia/src/sass_link.cpp; see
  nvidia/docs/libraries.md). Written from cuobjdump -elf listings of NVIDIA's
  inputs and outputs; matches NVIDIA's link byte for byte in code and
  relocations for sm_75-sm_90, and NVIDIA's driver runs its output.
  e2e_nvjitlink_sass, sm_75 to sm_120, passes on the card with either
  nvJitLink.
- nvJPEG: progressive decoding, multi-scan and restart handling, CMYK/YCCK,
  NV12/YUY2 output; the batched API torchvision.io.decode_jpeg uses and the
  decoupled three-phase API with streams, decoder states, buffers and decode
  parameters (region of interest, CMYK); progressive encoding, optimised
  Huffman tables, every subsampling, YUV and NV12 input, device retrieval,
  copied quantisation tables. Decoded pixels match NVIDIA's nvJPEG 13.0 on an
  RTX 3060 on all but 10 of 5.5 million samples (within two counts), and every
  refusal is the card's. e2e_nvjpeg_paths passes against both libraries.
- NPP: every entry point OpenCV (cudaarithm, cudaimgproc, cudawarping,
  cudafilters), DALI, FFmpeg's scale_npp, jetson-utils and the CUDA Samples
  call, but watershed segmentation and marker-label compression (one CUDA
  Sample): warps, rotation, remapping, ResizeSqrPixel, mirroring, logic and
  shifts with constants, alpha compositing, gamma, Bayer demosaicing, LUT,
  masked/float statistics, histograms, integral images, rank and morphological
  filters, Prewitt gradients, Canny (nvidia/src/npp_imgproc.cpp,
  npp_core.hpp). Sampling, rounding and fused-arithmetic conventions measured
  on an RTX 3060 against NPP 13.0; npp_imgproc's 289 results match it (golden
  in e2e_library_goldens), the near-ties listed in libraries.md marked.
  Remaining: watershed, marker labels, ResizeSqrPixel super-sampling and
  Lanczos, Scharr Canny's exact edges, the resize cubic kernel.
- Multi-GPU: peer access queries and cudaMemcpyPeer(Async) across virtual
  devices (`e2e_runtime_conformance`, `e2e_graph_shapes`, `test_runtime`;
  all_reduce and p2p_thrasher take their real peer-DMA paths in
  `pantheon_workloads`). f16 and CUDA Graphs have their own, longer bullets
  above and in the half-precision sweep.
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

Verified against real hardware: thirteen devices (twelve NVIDIA across six
architectures, plus the first AMD part, CDNA3). Eleven are in the table; the
A100 80GB (`nvidia/a100`) was read from a physical device on an 8x instance,
as the note after the table says:

| profile | device | how |
| --- | --- | --- |
| `nvidia/rtx3060` | RTX 3060 (sm_86) | local |
| `nvidia/rtx3080ti` | RTX 3080 Ti (sm_86, GA102) | server1; its registers are mapped too |
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

The eleven NVIDIA parts in the table match the physical device on **512
conformance values each** -- the same binary run on hardware and on VirtualGPU,
diffed (`nvidia/tests/conformance/run_conformance.sh`,
`nvidia/tools/verify-profile.sh`).

**The AMD one was discovery first, and is execution now.** `amd/mi325x`
describes a real MI325X -- 304 CUs, 64-lane wavefronts, 64 KiB LDS, gfx942 --
and was the first AMD profile read from hardware. When it was written nothing
could run on it, because the interpreter's warp was 32 lanes wide; the AMD
engine (`amd/src/gcn_exec.cpp`, 64-lane wavefronts) runs gfx942 kernels now,
and vLLM generates on a simulated MI300X and MI325X (`amd_vllm`). See
`amd/README.md`.

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

Still not read from hardware: H200 (inherited from H100), B200 (its framebuffer
was measured, the rest is public documentation), B300, RTX 5090, and six AMD
profiles (MI250X, MI300X, MI350X, RX 6900 XT, RX 7900 XTX, RX 9070 XT). B200 had
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
  calculation instead of returning a placeholder -- with NVIDIA's allocation
  rules as CUDA's occupancy calculator (cuda_occupancy.h) states them:
  registers per warp in units of 256 from four sub-partitions, the per-block
  file checked with a block's warps rounded up to those, and shared memory in
  128-byte units (256 before compute capability 8.0) with the driver's
  reserved kilobyte added to every block. An RTX 3060's API agrees with the
  calculator in all 22 cases of e2e_occupancy_rules, which checks the
  simulator against it on five GPUs. It is not affected by how many named
  barriers a kernel uses, on the card either.

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
cuSOLVER, NCCL, NVRTC, NPP and nvJPEG are implemented under their real sonames, as are cuDSS,
cuSPARSELt, cuTENSOR, cuTensorNet, cuStateVec, cuFile, nvCOMP, NVSHMEM, nvJitLink and nvFatbin (the whole
list, with what each refuses, is nvidia/docs/libraries.md). The first group was verified
against NVIDIA's own library on a physical GPU: cuFFT and cuSPARSE are
bit-identical on every value the conformance suite reports, cuDNN on its
forward suite and to 1e-6 relative on its training suite, cuSOLVER on
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
- **Triton works unmodified under `vgpu run` and `vgpu shell`.** It compiles to
  PTX and shells out to `ptxas` for a cubin; `build/bin/vgpu-ptxas` stands in
  for `ptxas` through `TRITON_PTXAS_PATH`, and `torch.compile` works too
  (nvidia/docs/jit.md). `nvidia/tools/vgpu_triton.py` (Triton's own
  `knobs.runtime.add_stages_inspection_hook` extension point, ending the
  pipeline at PTX) is the older route and the fallback outside those commands.
  Verified on a fused softmax, a `tl.dot` matmul (real `mma.sync` and
  `ldmatrix`) and an atomic reduction.
- **CuPy does not work.** It statically links the CUDA runtime rather than
  loading `libcudart.so`, so `LD_LIBRARY_PATH` never reaches it; its embedded
  runtime asks the driver for an export table and fails at
  `getDeviceCount()` with `cudaErrorSoftwareValidityNotEstablished` long
  before any kernel is compiled. Unblocking it needs the `cuGetExportTable`
  work below, not anything in the interpreter.

Multi-GPU is verified against real hardware in four places: the local two-GPU
box, and rented 2x H100 SXM5, 4x H100 SXM5 and 8x A100 80GB instances. All
the conformance suites that existed then (thirteen; there are seventeen in
`nvidia/tests/conformance/` now) match on every one of them, with NCCL compared
against NVIDIA's libnccl at two, four and eight ranks respectively. The 2x
instance runs CUDA 12.8, so it also covers the older toolkit's LZ4 fatbins,
its cudaGetDeviceProperties_v2 spelling and its soname majors; the 8x is
sm_80, a second architecture.

`nvidia-smi topo -m` is implemented (`src/cli/smi.cpp`; `run_smi_queries.sh`,
`run_smi_cli.sh`; only `-p2p` and `-i` are refused). DCGM's NVML half is
answered (docs/telemetry.md; `nvml_health`, `nvml_dcgm`); DCGM itself
has not been run against the simulator. PyTorch's official CUDA build runs
unmodified from the T4 to the B200 and the RTX 5090 (nvidia/docs/pytorch.md;
12 whole-model checks, `e2e_pytorch_models_<gpu>`); its CUDA 13 wheels carry
PTX for compute_120 only, so they run only on `nvidia/rtx5090`.

## A race the simulator found in llama.cpp

Status (rev 5): a record of one investigation, with no reproducer in this tree --
it ran against an external llama.cpp checkout's `test-backend-ops`, not
against anything under `ci/external/`, and the llama.cpp commit is not
recorded. Whether the upstream bug has been fixed since is unknown. What *is*
in the tree, and tested, is the detector: `VGPU_RACE=1` and `=2`
(`src/exec/interpreter.cpp:535-548`; README.md), with
`a_shared_race_between_warps_is_reported`, `two_warps_writing_the_same_value_is_not_reported`,
`the_same_value_write_is_still_a_race_under_strict_mode`,
`a_differing_write_after_a_redundant_one_is_still_reported` and
`one_warp_reusing_its_own_shared_words_is_not_a_race` in tests/unit/test_exec3.cpp.

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
  distinct project. It takes its documented "driver not installed" path (and is
  skipped on hosts with the real driver libraries): `OUT_OF_SCOPE="rt_virus"` in
  `tests/workloads/run_pantheon_workloads.sh`, docs/pantheon-workloads.md.
  (`media_enc_virus` runs per the prose in that document: `libvgpunvenc`
  implements NVENC; its status block still lists it as skipped, which is a
  contradiction inside that document, not here.)
- Nsight Systems and Nsight Compute as the vendor ships them. `vgpu ncu` is this
  project's own `ncu` (src/cli/ncu.cpp, `e2e_ncu`); `nsys` runs but collects no
  CUDA data (it uses its own bundled CUPTI, see nvidia/docs/cupti.md).

## Partially implemented

Rev 5 moved the entries that are complete features with tests out of this
section (they follow, under their own heading). What is left here is partial in
the sense that something specific is still missing, and each entry says what.

- Divergence: min-PC reconvergence (`src/exec/interpreter.cpp`) -- paths at the
  same pc merge and the lowest pc runs next, so bar.sync after a divergent
  region works (`tests/unit/test_barriers.cpp`). Not full IPDOM: irreducible
  control flow is not handled, and nothing pins that with a test. (The header
  comment at the top of interpreter.cpp still says barriers in divergent code
  are rejected and IPDOM is planned; it predates the min-PC code and is stale.)
- cuCtxSetCurrent(NULL) pops the top of the context stack rather than clearing
  the thread's binding (`nvidia/src/driver_api.cpp:938`; untested -- the
  neighbouring `cuCtxPopCurrent` is covered by `driver_gaps.cpp`). The stack
  itself is thread-local (`driver_api.cpp:175`); an earlier version of this
  entry said it was process-global, which was wrong.
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
  and amdgpu's dmesg lines. The session's sysfs has live AER stats and amdgpu
  RAS counts, `vgpu fault lose` drops a GPU off the bus (Xid 79), and `vgpu
  fault link` degrades its PCIe link. NVML's clock and P-state events are
  documented as Kepler-only and its power-source event as a laptop's, so none
  is offered. `--on copy` puts faults on copies out of device memory, and
  `--rate` arms any fault at a seeded rate per access. A lost AMD GPU is
  amdgpu's permanent PCI failure: the recovery in the kernel log, gone from
  rocm-smi and amd-smi, registers reading all ones. `amd-smi ras --cper
  --folder` writes amdgpu's CPER records for ECC errors, decoded by AMD's own
  ras-decode to the AFIDs it prints. Not yet: the CPER ring in debugfs, fatal
  and bad-page-threshold records, and a lost AMD GPU's sysfs entries going
  away.
  Tests: `tests/unit/test_ras.cpp`, `test_telemetry`, `fault_cli`, `e2e_fault_datapath`,
  `nvml_health`, `nvml_dcgm`, `amd_cper`, `amd_lost_gpu`, `amd_session`.
  What is still not modelled, in one list (docs/telemetry.md has the line for each):
  NVIDIA -- NVLink error injection and counters (the counters read zero and the far
  end of a link is not modelled), ECC mode changes and persistent accounting
  (NOT_SUPPORTED), the row-remapper bank histogram, retired-page addresses (placeholders
  derived from the device UUID, not measured), infoROM; AMD -- the CPER debugfs ring,
  fatal and bad-page-threshold records, a lost GPU's sysfs/hwmon entries going away
  (they stay), violation status, PCIe throughput, bad-page threshold,
  `rsmi_dev_gpu_metrics_info_get`, fabric/SoC/video clocks, every setter and reset
  (amd/src/amd_smi.cpp, rocm_smi.cpp); amd-smi's `--csv`, `ras --follow` and `ras --afid`
  (src/cli/amdsmi.cpp). AMD ECC, memory-sensor and link values follow AMD's
  documentation and have not been compared with a real MI-series card.
  NVML and nvidia-smi -q answer the health surface (docs/telemetry.md, "Health
  and diagnostic queries"): ECC by location, retired pages and remapped rows
  per card family, clock-event reasons and violation times, PCIe replays,
  NVLink state and error counters on the SXM profiles, accounting, and
  `nvidia-smi -r` and `nvlink -s/-e`; DCGM's other NVML reads too (energy
  counter, brand and board id, topology and affinity, supported and application
  clocks). Not yet: NVLink error injection, the
  row-remapper bank histogram (needs a bank count per card), real retired
  page addresses, `-q -x` for the new sections, a persistent accounting mode
  and ECC mode changes.
- Registers (docs/registers.md): PCI configuration space from a register
  database (registers/pci-config.yaml) -- header, power management, MSI, PCI
  Express with the live link, AER fed by the fault model, BAR sizing -- read,
  written and dumped with `vgpu regs`, every access logged, and rendered by the
  real lspci. AMD MMIO behind BAR5 (`--space mmio`): engine status and the SMU
  mailbox, from the amdgpu headers, on Aldebaran's IP bases until checked on an
  MI300, and NBIO's strap, VRAM size and partition modes, which the session's
  sysfs partition files are written from. NVIDIA MMIO behind BAR0, from NVIDIA's own published
  register headers (MIT, nvidia/registers/LICENSES/): identity with the
  architecture in the header's own fields on every profile and the measured
  card's value to the bit, whether the card is a virtual function, which
  engines a bound driver leaves running, the VBIOS scratch words, the BAR1 and
  BAR2 block registers, and a timer that advances with the engine's clock;
  plus the registers each architecture's own headers define (`arch` in the
  database; Turing through Blackwell, anything a header does not publish stays
  unmapped): interrupt enables with Turing's SET/CLEAR, memory and L2 ECC
  error counters fed by `vgpu fault` (what a memory diagnostic reads), MMU
  fault buffers, framebuffer flush addresses, the 64-word scratch arrays,
  thermal scratch and Blackwell's confidential-computing scratch.
  AMD UMC ECC (channels 0-3 of instance 0), its MCA status and the THM
  temperature are mapped on assumed Aldebaran bases.
  The SMU mailbox answers the header's failed and unknown-command codes, the
  NBIO link controller's speed straps are mapped, and GB100's topology table
  version. Not yet: the UMC's other channels and instances (beyond BAR5, via
  the indirect pair), a per-channel uncorrectable count (the UMC header has
  none), NBIO PCIe link status in BAR5 (its segment is not in the Aldebaran
  base table), SRBM, XGMI/PCS, and the NVIDIA blocks whose headers publish no
  usable values (the topology rows, PRI errors, temperature). A C API (vgpu_regs.h,
  libvgpuregs) gives bring-up software the same access. tools/regprobe
  captures real cards (read-only) to check and map the model against. AMD's
  gpu_metrics table (v1.5), amdgpu's busy and memory files and its hwmon (read
  by `sensors`) are published live in the session's sysfs. An RTX 3080 Ti's
  whole configuration space, read as root, is replayed for Ampere GeForce
  profiles, its registers found through its capability chain; an
  nvidia/rtx3080ti profile of that card is verified on 512 values, and its
  measured BAR0 (identification 0xb72000a1, unmapped 0xbadf5040) is its MMIO
  space. Next: map more of BAR0 as headers publish it, and capture data-center and AMD cards. The
  session's /sys/bus/pci devices carry config, resource, IDs and link files
  from the registers, and NVML and nvidia-smi read the link through them, each
  access logged under the tool's name. Every GPU model's registers and
  power-on values are kept in <vendor>/registers/gpus/ (`vgpu regs export`), and every
  simulated GPU of the model starts from them.
  Tests: `tests/unit/test_regs.cpp`, `regs_cli`, `c_harness_regs`; the register database is
  `registers/pci-config.yaml`, `nvidia/registers/mmio.yaml`, `amd/registers/mmio.yaml`, and
  the BAR0 map is gated per architecture (#293). Also not modelled: partition switching
  (docs/registers.md:200), AMD UMC bases beyond the assumed Aldebaran ones, MCA
  address/syndrome/IPID, and registers an architecture's headers do not give values for
  (Ampere/Ada memory-controller ECC counters, Blackwell's dev_ltc_zb, dev_fuse_zb and dev_tmr).

## Implemented: Hopper, Blackwell, barriers, dynamic parallelism and CUTLASS runs

Moved here from "Partially implemented" in rev 5: each entry below is a
complete feature with tests. The names it refuses are listed with it, and again
in "Not implemented".

- Hopper's warpgroup MMA (sm_90a): `wgmma.fence`, `.commit_group`,
  `.wait_group` and `.mma_async` in every dense form the ISA defines -- f16
  (f16 or f32 accumulator), bf16, tf32, e4m3/e5m2 in any pairing, s8/u8 in any
  signedness with and without `.satfinite` -- for every N from 8 to 256, A from
  registers or shared memory, B from shared memory, through the 64-bit matrix
  descriptor with all four swizzle modes (none, 32B, 64B, 128B) and both
  majornesses, the negate immediates and scale-d. Each warp of the warpgroup
  computes its own sixteen rows, which is exact because the rows divide that
  way (figures 151-158); the product completes when issued, one of the orders
  the asynchronous model allows, so a kernel that reads its accumulator
  before `wgmma.wait_group` is not caught. Checked two ways: unit tests lay
  out shared memory from the ISA's worked examples (figures 169-173), and an
  e2e test (nvidia/tests/e2e/wgmma_cute.cu) lets CuTe -- NVIDIA's own layout
  code, from a pinned CUTLASS release -- build the tiles, descriptors and
  fragments for seventeen configurations and compares every element exactly.
  Since 2026-09-27 the sparse forms too (`wgmma.mma_async.sp`, f16/bf16,
  tf32, fp8 and s8/u8): A holds M x K/2 -- in registers as mma.sp's fragment,
  in shared memory as the packed matrix in the ordinary canonical layout --
  and each warp expands its 16 rows with its own metadata, by mma.sp's rule
  for the same type (the ISA's wgmma metadata figures are mma.sp's). And
  `.b1` (m64nNk256 `.and.popc`), eight bits to a byte in shared memory.
  CUTLASS's 20 SM90 sparse GEMM tests pass (warp-specialized, cooperative,
  ping-pong, 2x1 clusters, fp8 fast accumulation); `.b1`, which CUTLASS has
  no test for, against a host reference. Refused by name: a descriptor with
  a nonzero base offset (the ISA does not say how it moves the pattern), and
  `wgmma` under any target but `.target sm_90a`. Loading now
  follows the target suffixes: a fatbin's `sm_90a` PTX is preferred over its
  `sm_90` one on a 9.0 device (nvcc -arch=sm_90a embeds both; only the first
  has the arch-specific instructions), `sm_XYa` code loads only on exactly
  X.Y, and `sm_XYf` within the family. `brkpt` parses and faults only if
  reached (CuTe places one on an unreachable path).

- TMA and clusters (sm_90): `cp.async.bulk` between global and shared
  memory, `cp.async.bulk.tensor` in one to five dimensions (tile mode) with
  the 32B, 64B and 128B swizzles, traversal strides and zero fill past the
  tensor's edge, both directions; bulk groups; the mbarrier transaction
  counts (`expect_tx`, `complete_tx`, `arrive.expect_tx`); and
  `barrier.cluster`, with a cluster's blocks resident together and
  interleaved, each with its own warp scheduler. Tensor maps come from
  `cuTensorMapEncodeTiled`, which checks every requirement cuda.h documents,
  and reach runtime programs through `cudaGetDriverEntryPoint`, which answers
  from this simulator's libcuda even when NVIDIA's is installed. A load's data
  lands in shared memory when its barrier is next looked at -- a moment the
  asynchronous model allows, and one at which a kernel that reads the tile
  without waiting sees the old contents; a store is written when issued, so
  a kernel that overwrites its source before `wait_group.read` is not caught.
  Checked three ways: unit tests (tests/unit/test_hopper.cpp) build the
  expected swizzled tile from the ISA's own swizzle table; CUTLASS's Hopper
  TMA and bulk-copy unit tests run unmodified and pass, all but the 1D tests,
  whose testbed writes 256 elements into a 128-element buffer, which this
  reports as the out-of-bounds access it is (run_cutlass_hopper.sh); and a
  CUTLASS-style GEMM -- 2-CTA clusters, a 3-stage TMA pipeline, wgmma -- is
  compared exactly (tma_gemm_cute.cu). Getting there fixed four things any
  kernel could hit: labels are scoped to their `{ }` block (inline asm
  repeats them); a lane can reach `bar.sync` from code nvcc placed after the
  kernel's `ret`; dynamic shared memory starts at its declared alignment; and
  `cvta.param` no longer adds the parameter window twice. A step-budget error
  now names the instruction the warp was spinning in.

- Distributed shared memory (sm_90): the blocks of a cluster reach each
  other's shared memory. `mapa` (generic and `.shared::cluster`, 32- and
  64-bit), `getctarank`, `isspacep.shared::cluster`, `cvta` to and from
  `.shared::cluster`, and `ld`, `st`, `atom` and `red` on `.shared::cluster`
  addresses; `mbarrier.arrive`, `arrive_drop`, `expect_tx` and `complete_tx`
  on another block's barrier (the ISA allows only those remotely, and a
  remote arrive must discard its state -- both refused by name otherwise);
  `st.async` and `red.async`, whose bytes complete on the destination's
  barrier; multicast TMA (`.multicast::cluster` with a ctaMask, tensor and
  plain), which lands the same bytes at the same offset in every block the
  mask names, each completing its own barrier; and
  `cp.async.bulk.shared::cluster.shared::cta`, one block's shared memory into
  another's. A shared::cluster address carries its block's rank above the
  offset, with zero meaning the issuing block, so every ordinary shared
  address is already a valid shared::cluster address for its own block, as
  the ISA requires. What hardware leaves undefined is reported instead:
  reaching a block that has exited ("its shared memory went with it" -- the
  reason a kernel ends with `cluster.sync()`), a rank past the cluster, and
  an mbarrier or st.async whose barrier is not in the block it writes. The
  race detector orders warps by their own block's barriers, so remote
  accesses are not checked for races. Checked three ways: unit tests
  (tests/unit/test_dsmem.cpp, each confirmed to fail when the feature it
  covers is broken); CUTLASS's own `tma_mcast_load` test, unmodified
  (run_cutlass_hopper.sh), which fails on the previous build and passes on
  this one; and a CUDA C++ program using cooperative_groups'
  `map_shared_rank`, `__cluster_query_shared_rank` and `cluster.sync()`,
  with clusters from `__cluster_dims__` and from `cudaLaunchKernelEx`
  (nvidia/tests/e2e/dsmem_cluster.cu): a ring exchange and a histogram whose
  bins are spread across the cluster, both exact.

- TMA reductions (sm_90): `cp.reduce.async.bulk` in all three forms -- a
  tensor box into global memory (what CuTe's `SM90_TMA_REDUCE_ADD` emits), a
  plain range into global memory, and a plain range into another block's
  shared memory completing on its mbarrier -- with every operation and type
  the ISA's tables allow (9.7.10.28.4.2, 9.7.10.28.5.4) and the others
  refused by name. Each element is an atomic read-modify-write, as the ISA
  makes it; floating-point add rounds to nearest even and keeps subnormals
  (`.noftz`, required for halves and the default for floats); min and max
  of a NaN give the other operand. The tensor form takes its element type
  from the map and, like a tensor store, leaves elements past the tensor's
  edge alone. A reduction is made when issued -- one of the moments the
  asynchronous proxy allows -- so, as for bulk stores, a kernel that
  overwrites its source before `wait_group.read` is not caught. Checked by
  unit tests over the values that separate a right reduction from a
  plausible one (a wrapping add, a signed min against INT_MIN, half ties to
  even and subnormals, inc's wrap, a 64-bit xor; each confirmed to fail with
  the arithmetic broken), and by a CuTe program (tma_reduce_cute.cu) in
  which several blocks reduce into the same tiles, f32 through a 128B
  swizzle and f16 unswizzled, exact with one host thread and with eight
  (`e2e_tma_reduce_cute`; it failed before this work).

- Tensor maps changed on the device (sm_90a): `tensormap.replace` on a map
  in global or shared memory, every field the ISA names (address, rank, box
  and global extents, strides, traversal strides, element type, interleave,
  swizzle, fill), and `tensormap.cp_fenceproxy`, which publishes a map edited
  in shared memory. This is how CUTLASS's grouped and pointer-array GEMMs
  point one descriptor at each group's tensors. Two things the ISA leaves to
  the reader, settled from evidence: the element type uses the ISA's own
  numbering (Table 36), which is not the driver's; and `global_stride` is in
  bytes from PTX ISA 8.5 and in 16-byte units before it -- the ISA does not
  say, but CuTe passes bytes when compiled by CUDA 12.5 or later and the
  stride shifted right by 4 before that, so the module's `.version` decides.
  A value no map from `cuTensorMapEncodeTiled` could hold (a box past 256, a
  traversal stride of 0) is refused rather than carried into a copy, and
  the Blackwell-only 96B swizzle and 32B-with-8B-flip atomicity are refused
  by name. (Blackwell's packed 4/6-bit types are done: nvidia/docs/blackwell.md.) Checked by unit tests (each field read
  back by the host's decoder; the ISA's 9 is f64 where the driver's 9 is
  bf16; a retargeted map loaded through under PTX 8.3 and 8.5; each confirmed
  to fail with the rule it covers broken), and by a CuTe program
  (tensormap_replace_cute.cu) that retargets a descriptor through CuTe's own
  helpers in shared memory and in global memory and loads through it,
  exact (`e2e_tensormap_replace_cute`; it failed before this work).

- TMA's im2col mode (sm_90): `cuTensorMapEncodeIm2col` in the driver and
  through `cudaGetDriverEntryPoint`, with the checks cuda.h documents (the
  corners' ranges per rank, a non-empty box, channels up to 256, pixels up
  to 1024), `cp.async.bulk.tensor` `.im2col` loads with their offsets
  (multicast too), and `.im2col_no_offs` stores and reductions. The ISA
  shows the walk in figures rather than words, so it is taken from the code
  that relies on it -- CuTe's im2col traits and CUTLASS's convolution
  corners: the box runs from the lower corner to dim + upper - 1 along each
  of W, H and D, stepped by the traversal stride, W fastest, then the batch;
  a load starts at the instruction's pixel and reads each pixel at its base
  plus the im2col offsets, zero outside the tensor. Checked by unit tests
  (the corners at each rank's width, the encoder's refusals, a walk that
  wraps and runs off the batch, the store side, a map used in the wrong
  mode), by tma_im2col.cu, which compares every tile of four convolutions
  (strides, dilation, padding, 1D and 2D) with the im2col matrix worked out
  from the definition of a convolution, and by CUTLASS's own SM90 conv2d
  fprop test, whose eight tile and cluster shapes pass against its host
  reference and fail with the offsets broken.

- Blackwell Ultra (sm_103a, a simulated B300 as nvidia/b300): the fp4 MMAs
  at K = 96 with three or six scale factors a row, shared-memory descriptors
  with an absolute leading-dimension address, and `tcgen05.ld.red`. CUTLASS's
  SM103 fp4 GEMMs run in e2e_cutlass_sm103. See nvidia/docs/blackwell.md.
- Blackwell's tensor core (sm_100a/sm_100f, PTX ISA 9.7.18): Tensor Memory
  (128 lanes x 512 columns per CTA) allocated with `tcgen05.alloc`/`dealloc`
  -- for a CTA pair with `.cta_group::2` -- and checked for leaks at exit; an
  allocation larger than what is free waits for another warp's dealloc, as
  the ISA's blocking alloc does (and is reported as a deadlock when no other
  warp of the CTA is left to free anything);
  `tcgen05.ld`/`st` in all five shapes with pack/unpack, each warp kept to
  its quarter of the lanes; `tcgen05.mma` for `.kind::f16`, `tf32`,
  `f8f6f4` (the 8-, 6- and 4-bit types, K- or MN-major) and `i8`, and
  block-scaled (`mxf8f6f4`, `mxf4`, `mxf4nvf4`, UE8M0 and UE4M3 factors
  from Tensor Memory), dense or with sparse A (`tcgen05.mma.sp`, the
  metadata as figures 287-292 lay it out), one CTA (M = 64/128) or a pair
  (M = 128/256), A from shared or Tensor Memory, with the data-path layouts
  A-D and F of figures 211-222; `tcgen05.cp` (every shape, `.warpx4` and
  `.warpx2`, fp4/fp6 decompression), `tcgen05.shift`; `tcgen05.commit`
  (multicast too), the fences and waits.
  Around it: cluster launch control (`try_cancel` takes over clusters that
  have not started, so CUTLASS's persistent loop really loops), `.b128`
  registers, TMA's `.cta_group::2`, `.tile::gather4`/`scatter4` and packed
  fp4/fp6 tensors, and the 128-byte swizzle in 32- and 64-byte atoms for
  TMA, `tensormap.replace` and the tcgen05 descriptor. A CTA's shared
  addresses carry its cluster rank in bits 24 and up, as CUTLASS's 2-SM
  kernels assume. Checked by unit tests (the ld/st figures as tables, every
  kind and layout against a host GEMM, the metadata placement, the pair
  layouts, the launch-control takeover, TMA through the peer bit) and by
  CUTLASS's own SM100 GEMM tests, unmodified, against its host reference:
  dense f16 (1-SM, 2-SM, stream-K), f8 and s8 (24 cases); sparse f16, tf32,
  f8 and s8 (88), and fp4/fp6 (16); block-scaled, every combination of
  mxf4/mxf6/mxf8/nvf4 in TN and NT (102 cases in 20 files); block-scaled
  sparse mxf8 and mxf6 (18). Weight-stationary MMAs (`.ws`: CUTLASS never
  issues them, and the ISA's zero-column-mask examples contradict each
  other) and `.ashift` are refused by name.
  See nvidia/docs/blackwell.md.

- The CTA's sixteen barriers (PTX ISA 9.7.15.1): `bar.sync` and
  `barrier.sync` on any of barriers 0-15, with or without a thread count,
  `bar.arrive`, register operands, and a guarded barrier the whole warp
  agrees on. A barrier with a count completes when that many threads have
  arrived, a warp counting as all its threads (the ISA "marks warps'
  arrival"); one without is the whole CTA. Warp-specialized kernels hand
  work between producer and consumer warps this way, so none of CUTLASS's
  Hopper GEMMs or convolutions loaded before -- found by running its conv
  test. A block whose warps can only wait on barriers that can no longer
  complete is reported as a deadlock, naming the barrier and how far it
  got; before, the block quietly ended with the warp still waiting.

- Dynamic parallelism (CDP2, 2026-09-25): kernels launch kernels through the
  device runtime's entry points as the CUDA programming guide documents them
  for code generators (__cudaCDP2GetParameterBufferV2 and
  __cudaCDP2LaunchDeviceV2). Kernels have addresses in their own window and
  the runtime gives every launch a table of them; the parameter buffer is
  device memory laid out as the child's parameters. A child runs after its
  parent grid and before the launch returns, in launch order (parent block by
  parent block, so it is the same with any number of host threads), each
  complete -- its own children included -- before the next: a schedule CUDA
  allows for every device-side stream, since it promises no concurrency
  between parent and child. CUDA's limit applies: 2048 pending launches (the
  24 this first wrote down was the synchronization depth, not a nesting
  limit; see the entry below). Checked by test_dynpar and
  dynamic_parallelism.cu (built with -rdc: fan-out, nesting, order, the tail
  and fire-and-forget streams, a struct parameter).

- The device runtime, completed (2026-10-05), on both engines: everything
  cuda_device_runtime_api.h gives a kernel -- cudaMemcpyAsync/2D/3D and the
  memset family, cudaMalloc/cudaFree (the device heap), cudaFuncGetAttributes,
  cudaDeviceGetAttribute/GetLimit, the cache configuration, the occupancy
  queries, cudaGetErrorString/Name, cudaRuntimeGetVersion, the older
  cudaGetParameterBuffer/cudaLaunchDevice pair, stream and event validation --
  under CDP2's names and CDP1's (cudaDeviceSynchronize for parts before
  Hopper). Each call's result, errors included, was measured on an RTX 3060
  (nvidia/docs/sass.md, "Device runtime"), which also corrected what the
  first version assumed: there is no nesting limit of 24 (a chain stops at the
  pending-launch limit, 2048 by default, with cudaErrorLaunchPendingCountExceeded;
  a limit below 32 is 32), a tail launch waits for every other grid the grid
  launched, and the card's device-side cudaMemsetAsync writes zero whatever
  the value is. The pending count is the card's, and when it is full the
  queued grids run at once. Both engines share include/vgpu/exec/devrt.hpp;
  the library's own answers (attributes, limits, error strings) come from the
  CUDA shim that launched the kernel. Checked by cdp_device_api.cu (against
  the card) and cdp1_device_sync.cu in e2e_sass_archs and e2e_device_runtime.

- CUTLASS's SM90 GEMM unit tests, run unmodified (2026-09-25), found: the
  register estimate ignored launch bounds (.maxntid/.minnctapersm/.maxnreg
  now cap it, as ptxas does, spilling the rest); an mbarrier instruction whose
  lanes name different blocks' barriers was refused (lanes are now handled
  barrier by barrier); cudaFuncAttributeNonPortableClusterSizeAllowed was
  ignored (clusters of up to 16 now launch once it is set); and dp2a was
  missing. The cluster warp-specialized cooperative test passes all 22 cases
  and the pointer-array test its 2.
  The ping-pong kernel's wrong 16-row A slices at 2x4x1 (and the group
  GEMM's, at 2x2x1) were one bug: each warp of a warpgroup read its operands
  of a `wgmma` from shared memory when it got to the instruction, so a warp
  that finished early could release the stage and let the cluster peer that
  multicasts A refill it before the last warp had read. The warpgroup now
  reads them once, at the first warp's issue, and `wgmma.wait_group` holds a
  warp until all four have issued what it waits for (letting the first warp
  through alone hung the ping-pong kernel's SIMT-epilogue variant at 2x2x1:
  the stage it released was refilled past a lagging warp's parity wait).
  The group GEMM also found
  `cvt.sat` and `.ftz` ignored and tiny fp16 results flushed to zero (its
  silu epilogue's expf leans on `cvt.sat.f32.f32`); conversions now match an
  RTX 3060 bit for bit, NaN encodings included.
- CUTLASS's SM80 sparse GEMM tests (all 19 pass) found three more: an empty
  `cp.async` group did not count toward `wait_group N`, so a wait left an
  older real group pending; `bar.red` voted for a partial warp when some of
  its lanes were on another path, and let a warp looping back vote into the
  round the others were still collecting; and a `.reg` declared inside
  `{ }` did not hide the outer register of the same name, which
  `__syncthreads_and`'s inline asm relies on. The last two hung CUTLASS's
  split-K semaphore wait whenever the blocks ran on more than one host
  thread. `ex2.approx` differs from the hardware's by an ulp in about 30% of
  inputs (it is approximate, and matching it bit for bit would take the SFU's
  internals), which moves `expf` by an ulp too.

## Not implemented (fails loudly, never silently)

This list was stale for a while, which is its own kind of wrong: it still named
textures, grid sync and host-pinned memory long after all three worked, and
(until rev 5) cubin/SASS loading, async copies and AMD's rocBLAS GEMMs. A
roadmap that overstates what is missing misleads as much as one that overstates
what is done. Rev 5 re-tested each item below by finding the refusing message in
the source; the file and line are given so the next audit can repeat it. "Gap
register" at the end of this file adds the vendor-library, SASS, AMD and
health-surface refusals, which are far more numerous than the ones named here.

- **PTX, refused by name** (`src/ptx/parser.cpp`, `src/exec/interpreter.cpp`,
  `include/vgpu/exec/tensormap.hpp`):
  - TMA's `.im2col::w` modes (Blackwell; parser.cpp:4770, and on SASS
    src/sass/exec_ops.inc:3800); see nvidia/docs/blackwell.md.
  - TMA attribute overrides and reports (parser.cpp:4770, :4651).
  - Interleaved tensor-map layouts and the 128B swizzle's 8-byte-flip variant
    `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B` (tensormap.hpp:275-276, :362-363).
  - tcgen05's `.ashift` (parser.cpp:2980), `.ws` with A in Tensor Memory below
    M = 128, and `.ws.sp` (interpreter.cpp:11305-11312) -- the ISA draws neither
    layout.
  - sm_107 additions (`kind::ti16`, `decompress::lut`).
  - Inline-asm-only instructions: anything the parser has no name for is refused
    by that name, not run as something else.
  - Not on this list any more: the TMA out-of-bounds NaN fill (`oobFill`) is
    *accepted* -- stored in the descriptor, carried by `tensormap.replace`
    (interpreter.cpp:9277) and honoured by loads that go out of bounds
    (interpreter.cpp:9534, tests/unit/test_hopper.cpp:1236). What is not
    established is the exact value the hardware writes, which the ISA does not
    state; treat it as unverified, not as unsupported.
  - Done, so not refused: `wgmma`, TMA, mbarrier transaction counts,
    `barrier.cluster`, distributed shared memory, tcgen05 `.ws`, textures,
    surfaces, grid sync (see the "Implemented: Hopper, Blackwell..." section).
- **Runtime**: exporting a virtual-memory-management handle to another process
  (device memory here is this process's own sparse backing; cuMemAddressReserve,
  cuMemCreate, cuMemMap, cuMemSetAccess and the rest are done, with the faults
  named for unmapped, no-access and read-only mappings). `cudaLaunchCooperativeKernelMultiDevice`
  and grids too large to be resident are refused (docs/cooperative.md). The
  async copy that this item used to name is done: `cp.async`, `cp.async.bulk`,
  `cp.reduce.async.bulk`, stream-ordered allocation, and AMD's queued
  `hipMemcpyAsync` (amd/src/hip_api.cpp:1705).
- **Frontends**: nothing missing on NVIDIA -- PTX and SASS both run (SASS is the
  default engine on every generation from sm_75 to sm_120a, src/sass/,
  nvidia/docs/sass.md; `VGPU_SASS=0` selects PTX). Static-cudart binaries are the
  separate, documented blocker under "Next milestones" item 4. On AMD, execution is
  done (amd/README.md); what it refuses is in the gap register.
- **Tooling**: trace record/replay and a conformance database with compatibility
  scores. (`vgpu run`, `vgpu test --matrix`, shared-memory race detection, the
  random and adversarial schedulers, and fault injection are done.) Counters
  that need a timing model are refused by design, below.
- **Profilers**: Nsight Compute and Nsight Systems as the vendor ships them (see
  "Known out of scope"); CUPTI's Callback API delivers the runtime, resource and synchronize domains only (no driver, module, graph or NVTX callbacks), the Event
  and Profiling metrics APIs are absent, and nothing derived from time is
  reported (nvidia/docs/cupti.md:70-139).

## Registers and counters: what is modelled, and what cannot be

Two questions that look alike and are not.

**Special registers are per *architecture*, not per GPU model.** `%tid`,
`%laneid`, `%smid` and `%clock64` are PTX ISA instructions, identical on a T4
and a B200. What varies is which exist -- the thread-block cluster registers
require sm_90 -- and what they return, which is already profile driven
(`%nsmid` is 132 on an H100 and 58 on an L4). Every special register in the
PTX ISA (chapter "Special Registers", ISA 9.0) is either implemented, on both
engines, or refused by name:

| Register | Status |
| --- | --- |
| `%tid`, `%ntid`, `%ctaid`, `%nctaid` | implemented |
| `%laneid`, `%warpid`, `%nwarpid` | implemented (`%warpid` is the warp's index in its block, which is what this engine's scheduling makes it) |
| `%lanemask_eq/_le/_lt/_ge/_gt` | implemented |
| `%smid`, `%nsmid` | implemented; `%smid` is a round-robin placement, distinct among resident blocks |
| `%gridid` | implemented: a serial number per launch |
| `%clusterid`, `%nclusterid`, `%cluster_ctaid`, `%cluster_nctaid`, `%cluster_ctarank`, `%cluster_nctarank`, `%is_explicit_cluster` | implemented (sm_90+), with clusters from `__cluster_dims__` and `cudaLaunchKernelEx`; a launch without a cluster behaves as 1x1x1 |
| `%clock`, `%clock_hi`, `%clock64`, `%globaltimer`, `%globaltimer_lo`, `%globaltimer_hi` | implemented as a deterministic, monotonic count of the block's instructions -- not a time (no timing model) |
| `%envreg0`-`%envreg31` | implemented; the driver sets only `%envreg1`/`%envreg2` (a cooperative launch's grid-barrier workspace), the rest read 0 |
| `%dynamic_smem_size`, `%total_smem_size`, `%aggr_smem_size` | implemented; `%aggr_smem_size` refused below sm_90 or PTX ISA 8.1, as ptxas refuses it |
| `%reserved_smem_offset_begin/_end/_cap/_0/_1` | implemented (sm_80+), as an RTX 3060 reports them |
| `%current_graph_exec` | implemented: the device graph the kernel runs in, 0 outside one (device-side graph launch; on SASS the bank-0 word ptxas loads it from) |
| `%pm0`-`%pm7`, `%pm0_64`-`%pm7_64` | **refused by name**: hardware performance-monitor counters, undefined unless a profiler configured them, and with no timing model there is nothing to count, so a silent zero would be a confidently wrong answer |

A `%`-name that is none of these is refused as an unknown special register,
never read as an unwritten ordinary register. What they return is still
approximate in three places: `%smid` is round-robin rather than real
placement, the clock family counts instructions rather than time, and only
`%envreg1-2` are set.

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
(derivable today from `instructions * 32 - thread_instructions`). Spill traffic
is not counted on either engine: on the PTX path no spill exists (ptxas allocates
registers and this executes PTX), and on the SASS path spills are real local
loads and stores that *could* be split out and are not. The note in
`vgpu counters` (src/cli/main.cpp:103-114) still gives only the PTX reason and
should say both. Done, not missing: the per-opcode histogram `inst_by_opcode`
(tests/unit/test_exec3.cpp:2657) and the `ncu --query-metrics` capture
(nvidia/tools/characterize-cloud.sh).

## Performance

**2026-09-25: fast paths, measured on the pantheon workloads that hit their
watchdog at full size.** perf (extracted from Ubuntu's linux-tools package;
no install needed) put int_virus's time in int_bin, dispatch, read_operand
and write_reg: every 32-bit operand widened into a 64-lane array of 64-bit
values, and the operation decoded again for every lane. What changed:

- The instructions kernels live in take a fast path when they need nothing
  special: 32- and 64-bit integer ops, mad.lo, mov, setp, integer cvt,
  mul.wide, f32 arithmetic, f32/f64/f16x2 fma, and wmma.mma. It reads the
  narrow register file in place, decides the operation once per
  instruction, and is called straight from the warp loop, skipping step()
  and dispatch(). `VGPU_FASTPATH=0` turns every one of them off, and
  tests/unit/test_fastpath.cpp runs each form both ways over the values
  that break arithmetic and requires the same bits.
- fma uses the host's FMA unit in a lane loop (the baseline x86-64 build
  made std::fma a libm call per lane); f16 decodes from a table and rounds
  on the bits; wmma's 16x16x16 multiply-accumulate is AVX2 where available,
  with contraction off so no build fuses its products into an FMA.
- Loads and stores no longer allocate on every instruction, and a
  multi-threaded launch hands out blocks as threads free up.

Single-thread simulated instructions per second, A/B against the previous
main on the same machine (warp instructions; `VGPU_COUNTERS` over CPU time):

| workload | before | after |
| --- | --- | --- |
| int_virus | 7.8 M | 38-60 M |
| compute_virus | 11.5 M | 35 M |
| pulse_virus | 17.7 M | 57 M |
| memory_retention_bake | 11.7 M | 35 M |
| fp64_virus | 1.8 M | 5-6 M |
| omni_virus | 4.0 M | 10.5 M |
| mma_virus | 0.3 M | 1.1 M |

At full size (the default grid, 30 s, --mem 99, a 4 GB A10 profile) all
seven used to exceed pantheon's 360 s watchdog. On main after the merge,
on a machine other jobs were also loading (load average 20-30), five
finish:

| workload | full size |
| --- | --- |
| int_virus | 72 s |
| memory_bank_thrash | 116 s |
| pulse_virus | 161 s |
| omni_virus | 175 s |
| memory_retention_bake | 195 s |
| fp64_virus | watchdog (360 s) |
| mma_virus | watchdog (360 s) when this table was taken; 186 s since 2026-09-27, see below |

fp64_virus is held back by the subnormal assists below; mma_virus by the
matrix work itself, which is still about 1 M warp instructions a second.

Since then (2026-09-27), profiled with perf again:

- mma_virus's time was the WMMA fast path's bookkeeping, not the multiply.
  It scanned all 32 fragment operands for a 64-bit register on every
  execution (now decided once by the parser), converted f16 through a
  512 KB double table (now F16C, eight halves at a time, used only after it
  matches the table for all 65536 inputs), moved fragments one element at a
  time (now AVX2 8x8 transposes) and zero-filled tiles every element of
  which it then wrote. One host thread, grid 4: 6.4 -> 19.7 GFLOPS. At full
  size on 20 cores it finishes in 186 s, against 361 s before on the same
  machine -- inside pantheon's 360 s watchdog now.
- Register files are no longer zero-filled at every block start (nvcc
  declared ~1500 registers per warp in mma_virus). A register's first write
  zero-fills it only when the write leaves lanes out, so those lanes still
  read zero; a unit test fails without that.
- The f64 fma fast path reads its register operands in place instead of
  through read_operand: fp64_virus, one thread, grid 4, 1.67 -> 2.14
  GFLOPS. At full size that is 99 -> 86 s per launch, and with five warmup
  launches before the timed one it still exceeds the watchdog: at 10000
  loops its FMA chains decay into subnormals, where the host's assists
  dominate. Doing the FMAs four lanes at a time with AVX was measured and
  was no faster (the assist is not cheaper for a vector), so it was not
  kept.

One limit found and left alone: on Intel cores a subnormal operand or result
costs every FP operation a microcode assist, and fp64_virus's FMA chains
decay into that range -- it runs about 3.7x faster with flush-to-zero set.
Flushing changes the answers, so it is not used; running the FMA unit with
FTZ/DAZ and recomputing the affected lanes exactly in integer arithmetic was
built, verified, and measured slower than the assists (it lost the
vectorization and added a branch per lane), so it was removed.

Earlier: two changes moved the needle, both found by profiling rather than
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
Profile with perf; guessing has been wrong every time so far.

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
1. **PTX -> internal IR -> LLVM JIT** (ARCHITECTURE.md D1). Not started: nothing
   in the tree uses LLVM (CMakeLists.txt has none apart from nvJitLink's own
   linker). The profile this item was written from has moved on -- the
   2026-09-25 and 09-27 rewrites found the remaining overhead in operand
   decoding and bookkeeping rather than in per-lane interpretation -- so
   re-profile before committing to a JIT.
2. Block-level parallelism across host threads, behind the scheduler
   abstraction so determinism is preserved. **Done**: the grid runs on every core
   (`VGPU_THREADS`, default `host_cpus()`, interpreter.cpp:11997).

A GPU still retires ~10^13 ops/s, so saturation-style stress tests are run at
reduced intensity via their own CLI knobs; see
scripts/run-pantheon-workloads.sh.

## Next milestones (order)

0. **AMD** -- the list this item held is done; rev 5 checked each entry
   (amd/README.md has the evidence). rocBLAS runs 162,807 of its own quick
   float and double tests, and its half, bfloat16, int8 and FP8 GEMMs run;
   hipBLASLt, MIOpen, rocPRIM/hipCUB (`tests/hipcc/prim.cpp`), rocRAND, rocFFT,
   rocSPARSE, rocSOLVER (`amd/tests/libraries`, `amd_libraries`), hipSPARSELt,
   Composable Kernel (`amd_pytorch_ck`) and RCCL, single- and multi-process with
   IPC memory handles (`amd_pytorch_distributed`, `amd_pytorch_multi_gpu`), run
   under PyTorch for ROCm on six GPUs (`amd_pytorch_*`) and vLLM (`amd_vllm`);
   pinned and managed memory and asynchronous streams are done; the HSA/KFD
   layer (rocminfo, `rocm_agent_enumerator`, `/sys/class/kfd`) is done
   (`amd_hsa`, `amd_hip_on_hsa`, `test_amd_kfd`); textures and images are done
   on RDNA (MI300 has none, correctly); gfx950 (MI350X) is done
   (`amd_pytorch_mi350x`); OpenMP offload and OpenCL run on all six
   (`amd_openmp`, `amd_opencl`). **Still open on AMD**: gfx9 image
   instructions (MI250X), a double `trsv` path in rocBLAS
   (status unconfirmed), the instruction forms the executor refuses by name
   (gap register), and anything in the gap register's AMD part.
1. **Interpreter speed**: intern register names to dense indices at parse
   time. **Done** (see Performance above, "Register names are interned to dense
   ids").
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
   `verified` bits in the profiles. **Done as v0, now ongoing per profile**: the
   harness exists (`nvidia/tools/characterize*.sh`, `characterize.cu`,
   `compare-profile.py`, `verify-profile.sh`, `amd/tools/characterize-do.sh`) and
   12 NVIDIA profiles and `amd/mi325x` are verified. Open: `h200`, `b200`,
   `b300`, `rtx5090` and six AMD profiles (Hardware characterization).
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

   **Textures and surfaces are done**: point and linear filtering, 1D/2D/3D,
   layered and cubemap (and layered cubemap) textures, layered surfaces,
   mipmaps with an explicit level of detail (`tex.level`, and plain fetches
   of mipmapped textures), and gather (`tld4`), over linear memory, pitched
   2D, arrays and mipmapped arrays, with every addressing mode and read mode.
   Linear filtering used to be refused because the guide gives the formula
   but not the arithmetic; the arithmetic was measured on an RTX 3060 until
   every sample matched bit for bit (weights in 1/256ths split z, x, y with
   measured rounding sides, one exact sum rounded ties-away, 1D as 2D at
   y = 0, the LOD's truncations), and the e2e tests hash tens of thousands
   of results against the hardware's. Refused by name: `tex.grad` (its LOD
   comes from undocumented approximate units), linear filtering of signed
   8-bit normalized texels, `tld4` on layered/cubemap textures, anisotropy
   and resource views. The `.clamp`/`.zero` surface policies are done, as an
   RTX 3060 applies them. See nvidia/docs/textures.md. Border
   colours are done: converted to the texture's format by rules measured over
   280,000 colours (e2e_border_colour, 705 cases). Measuring them turned up
   four older mistakes, now fixed: an absent w read 1 (the card reads 0), a
   mipmapped texture's unnormalized coordinates were taken as unnormalized
   (the card normalizes them), half NaNs were quieted, and the float filter
   was an exact sum where the card truncates each value below its
   footprint's largest. Mipmapped layered and cubemap textures are done
   too (e2e_texture_mip_layers, 240 cases). So is sRGB, through the texture unit's
   measured decode table and its block-exponent blend (e2e_texture_srgb,
   228 cases).

   `wgmma`, TMA and distributed shared memory are done now (see "Hopper's
   warpgroup MMA", "TMA and clusters" and "Distributed shared memory"). What
   was left of Hopper -- `cp.reduce.async.bulk`, `tensormap.replace` and
   TMA's im2col mode -- is done too (see "TMA reductions", "Tensor maps
   changed on the device" and "TMA's im2col mode"). Blackwell's gather and
   scatter modes are done; `.im2col::w` is refused (nvidia/docs/blackwell.md).

## Gap work in flight (2026-10-07)

Pull requests open when rev 5 was written, each closing a gap from the register below. They are
listed so the register is not read as if nothing were being done; a merged PR should move its entry
into "Implemented" and out of here. All were written by agents from public documentation and
public headers, tested by their authors under ASan + UBSan, and are awaiting CI and review.

| PR | Gap it closes |
| --- | --- |
| pantheonsim #301 | ROCm SMI: `rsmi_dev_gpu_metrics_info_get`, `rsmi_dev_metrics_header_info_get`, 52 `rsmi_*` functions the public header declares but the library did not export |
| pantheonsim #302 | AMD SMI getters that were `NOT_SUPPORTED` stubs: VRAM usage, busy percent, perf and overdrive levels, bad pages, compute-process info |
| pantheonsim #303 | HSA 1.1: `hsa_agent_iterate_caches`, `hsa_cache_get_info`, `hsa_isa_iterate_wavefronts`, `hsa_wavefront_get_info`, `hsa_isa_compatible`, `hsa_signal_group_*` |
| pantheonsim #304 | NVRTC: the real `libnvrtc-builtins` beside a pip-wheel `libnvrtc` (every PyTorch jiterator kernel failed without nvcc) |
| pantheonsim #305 | cuBLASLt int8 x int8 -> int32 (`CUBLAS_COMPUTE_32I`), what `torch._int_mm` calls |
| pantheonsim #306 | SASS: `LEA` with a negated addend carries out of a zero (a bf16 im2col read 2^32 elements out of bounds on a T4) |
| pantheonsim #307 | PTX `brx.idx` with `.branchtargets`, and `.calltargets` on indirect calls (a dense `switch` refused the whole kernel) |
| pantheonsim #308 | Driver `cuLaunchKernelEx` reads its attribute list (the stride is 72 bytes in every CUDA 12 header; clustered launches from Triton and CUTLASS were refused) |
| pantheonsim #309 | Runtime exports the toolkit has: `cudaMemset3D(Async)`, `cudaChooseDevice`, `cudaInitDevice`, 2D array copies, descriptor getters, `cudaFuncGetName`/`GetParamInfo`, ... |
| pantheonsim #310 | Driver stream-ordered memory pools (`cuMemPool*`, `cuMemAllocAsync` through the current pool) |
| pantheonsim #311 | Driver `cuPointerGetAttributes`, `cuPointerSetAttribute` and the missing `cuPointerGetAttribute` answers |
| pantheonsim #290 | The memory-pattern e2e test and the `cuda_memtest` and `gpu-burn` external suites |
| pantheonworkloads #10 | `arch-*` and `lib-*` workload runs on `sim:nvidia`, and `unsupported_ok` for ops real GPUs lack |

## Profile inventory (rev 6)

45 profiles (rev 5 counted 23 at 2ce1f45; rev 6 adds eleven NVIDIA profiles and rev 7 ten more, built from public documents only, all `verified: false`, see nvidia/docs/profiles.md). `verified: true` means the
profile's values were read from a physical device (see "Hardware characterization").

| id | vendor | arch | cc / gfx | verified |
| --- | --- | --- | --- | --- |
| `nvidia/t4` | NVIDIA | Turing | 7.5 | yes |
| `nvidia/a100` (A100 SXM4 80GB) | NVIDIA | Ampere | 8.0 | yes |
| `nvidia/a100-sxm4-40gb` | NVIDIA | Ampere | 8.0 | yes |
| `nvidia/a10` | NVIDIA | Ampere | 8.6 | yes |
| `nvidia/a10g` | NVIDIA | Ampere | 8.6 | yes |
| `nvidia/rtx3060` | NVIDIA | Ampere | 8.6 | yes |
| `nvidia/rtx3080ti` | NVIDIA | Ampere | 8.6 | yes |
| `nvidia/l4` | NVIDIA | Ada | 8.9 | yes |
| `nvidia/l40s` | NVIDIA | Ada | 8.9 | yes |
| `nvidia/h100` | NVIDIA | Hopper | 9.0 | yes |
| `nvidia/h100-pcie` | NVIDIA | Hopper | 9.0 | yes |
| `nvidia/gh200-480gb` | NVIDIA | Hopper | 9.0 | yes |
| `nvidia/h200` | NVIDIA | Hopper | 9.0 | **no** (inherits H100; `vram_bytes` not measured) |
| `nvidia/b200` | NVIDIA | Blackwell | 10.0 | **no** (framebuffer measured, the rest public documentation) |
| `nvidia/b300` | NVIDIA | Blackwell Ultra | 10.3 | **no** |
| `nvidia/rtx5090` | NVIDIA | Blackwell | 12.0 | **no** |
| `nvidia/gb200` | NVIDIA | Blackwell | 10.0 | **no** (B200's limits; memory from NVIDIA's per-superchip figure) |
| `nvidia/vr200` | NVIDIA | Rubin | 10.7 | **no** (placeholder PCI id 0x7F10) |
| `nvidia/thor` | NVIDIA | Blackwell | 11.0 | **no** (placeholder PCI id 0x7F11; SM count derived) |
| `nvidia/rtx-pro-6000` | NVIDIA | Blackwell | 12.0 | **no** |
| `nvidia/gb10` | NVIDIA | Blackwell | 12.1 | **no** (SM count derived from the 6,144 CUDA cores) |
| `nvidia/rtx4090` | NVIDIA | Ada | 8.9 | **no** |
| `nvidia/l40` | NVIDIA | Ada | 8.9 | **no** |
| `nvidia/rtx3090` | NVIDIA | Ampere | 8.6 | **no** |
| `nvidia/a40` | NVIDIA | Ampere | 8.6 | **no** |
| `nvidia/a30` | NVIDIA | Ampere | 8.0 | **no** (SM count derived from OEM Tensor Core counts) |
| `nvidia/rtx2080ti` | NVIDIA | Turing | 7.5 | **no** |
| `nvidia/rtx-pro-6000-server` | NVIDIA | Blackwell | 12.0 | **no** (the g7e part; the rental was not obtained, see nvidia/docs/profiles.md) |
| `nvidia/rtx-pro-6000-max-q` | NVIDIA | Blackwell | 12.0 | **no** |
| `nvidia/rtx6000-ada` | NVIDIA | Ada | 8.9 | **no** |
| `nvidia/rtx-a6000` | NVIDIA | Ampere | 8.6 | **no** |
| `nvidia/rtx-a5000` | NVIDIA | Ampere | 8.6 | **no** |
| `nvidia/a100-80gb-pcie` | NVIDIA | Ampere | 8.0 | **no** (limits inherited from the read A100s) |
| `nvidia/h100-nvl` | NVIDIA | Hopper | 9.0 | **no** (SM count derived from the published FP64 rate) |
| `nvidia/h200-nvl` | NVIDIA | Hopper | 9.0 | **no** (SM count derived the same way) |
| `nvidia/rtx4080` | NVIDIA | Ada | 8.9 | **no** |
| `nvidia/rtx3070` | NVIDIA | Ampere | 8.6 | **no** |
| `amd/mi325x` | AMD | CDNA3 | gfx942 | yes (rocminfo on a physical card) |
| `amd/mi300x` | AMD | CDNA3 | gfx942 | **no** |
| `amd/mi250x` | AMD | CDNA2 | gfx90a | **no** (one die: 110 CUs, 64 GB) |
| `amd/mi350x` | AMD | CDNA4 | gfx950 | **no** |
| `amd/mi455x` | AMD | CDNA5 | gfx1250 | **no** (AMD's announcements and LLVM; no card has been seen; clocks, power and ids are placeholders) |
| `amd/rx6900xt` | AMD | RDNA2 | gfx1030 | **no** |
| `amd/rx7900xtx` | AMD | RDNA3 | gfx1100 | **no** |
| `amd/rx9070xt` | AMD | RDNA4 | gfx1201 | **no** |

Counting caveat: the audit counted `verified: true` per profile file; a file can carry
the key per field, so "yes" here means the profile as a whole is recorded as verified,
and a future audit should check the fields too.

## CI inventory (rev 5)

Workflows under `.github/workflows/` at 2ce1f45.

| Workflow | Triggers | Jobs |
| --- | --- | --- |
| `ci.yml` | push to main, PR, dispatch (`ml_tests`, `cutlass_sass`, `nvidia_only`) | toolchain-image lookup; build + unit tests; pantheon workloads (60 min); SASS on every architecture (CUDA 13 container); Ollama v0.21.0 on every NVIDIA GPU; Ollama on seven AMD GPUs (mi300x, mi325x, mi350x, mi250x, rx6900xt, rx7900xtx, rx9070xt); AMD code objects and CDNA decoding; PyTorch/ROCm/the rest on the pantheonsim.com runner; **ASan + UBSan**; **ThreadSanitizer** |
| `action.yml` | push, PR, dispatch | the repo's own GitHub Action under test: CUDA, HIP, CUDA 12.6 from NVIDIA's repository, Ubuntu 22.04 (nvidia/t4 and amd/mi300x), a root container with no sudo, CMake projects (cmake-cuda, cmake-hip) |
| `ci-image.yml` / `ci-image-ref.yml` | push, weekly, dispatch / workflow_call | toolchain images cuda12.0 and cuda13.0 |
| `cuda-toolkits.yml` | push, PR, daily, dispatch | build + tests on CUDA 12.8 (ubuntu-24.04) and CUDA 12.4 (ubuntu-22.04; skips `amd_tools`, `regs_cli`, `amd_hipcc`, which need GLIBCXX_3.4.30) |
| `external-smoke.yml` | PR touching nvidia/, src/, include/, ci/external/ | fast subset of the outside suites on nvidia/rtx3060 (cuda-samples, hecbench, rodinia, polybench), with SASS-vs-PTX digests |
| `external-suites.yml` | nightly, dispatch | cuda-samples, hecbench, rodinia, polybench on nvidia/rtx3060 (SASS and/or PTX); hip-tests (ROCm 7.1, sharded, per-GPU baselines); BabelStream (HIP) with a `vgpu fault stuck` cell check |
| `lazy-modules-nightly.yml` | daily, PR | the e2e suite with every module parsed lazily |
| `rocblas-nightly.yml` | daily, PR | AMD's own `rocblas-test` in shards on mi300x, mi350x, mi250x |
| `rocm-libraries.yml` | push, PR, weekly | PyTorch, rocBLAS and hipSPARSELt on simulated AMD GPUs |
| `sass-ptx.yml` | daily, PR | the suite on PTX (`VGPU_SASS=0`); SASS and PTX leave the same memory (digest compare) |
| `vllm-nightly.yml` | daily, PR | vLLM on mi300x, mi325x, mi250x, mi350x, rx7900xtx, rx9070xt |
| `workloads.yml` | push, daily, PR, dispatch | pantheon workloads: one row per NVIDIA profile (1 to 8 GPUs, CUDA 12.0 or 13.0); `coverage` fails when a profile has no row |

What no workflow covers: an AMD row in `workloads` or its `coverage` check, and the
real-GPU card check designed in docs/ci-card-check.md (design, not enabled). The
external suites are pinned in `ci/external/suites.sh` (cuda-samples v13.0, hecbench,
rodinia, polybench, hip-tests rocm-7.1.0, BabelStream 5.0). The hip-tests runs carry
per-GPU baselines and disabled-case lists (`ci/external/hip-tests/baseline-*.tsv`,
`disabled-gfx*.txt`: 562, 561, 562 and 589 disabled cases for gfx942, gfx1030,
gfx1100 and gfx1201); those disabled cases are an unreconciled backlog. The
SASS-vs-PTX digest differences not yet reconciled are in `ci/external/digest-known.txt`.

Sanitizer rules the code has to meet (learnt the hard way in rev 5): ASan + UBSan run with
`detect_leaks=1` and `halt_on_error`, so free what you allocate (a user object's handle, a
`cudaHostAlloc` buffer in a test), form wrapping arithmetic unsigned (`v_mul_i32_i24` was a
signed overflow), and a test that loads two simulator shims in one process is an ODR
violation to ASan, because each shim embeds the same core: give it
`ASAN_OPTIONS=...:detect_odr_violation=0` in its CMake `ENVIRONMENT` (see
`test_runtime_user_objects`, `VGPU_SAN_ODR_OFF`, `nvidia/tests/e2e/run_lib_check.sh`).
CUDA 12.4 and 12.8 headers are what CI compiles against, so newer APIs need
`CUDART_VERSION` / `NVML_API_VERSION` guards.

## Test inventory (rev 5)

57 `vgpu_unit_test` and about 190 `add_test` entries in `CMakeLists.txt`, grouped. Some names
are generated per GPU (`e2e_pytorch_models_<gpu>`, `amd_pytorch_<gpu>`, `e2e_nccl_comm_ops_N`)
and the per-library path tests (cuDSS, cuTENSOR, cuTensorNet, cuSPARSELt, cuStateVec, cuFile,
nvCOMP, nvFatbin, nvJitLink, NVRTC, cuFFT, cuSOLVER, cuSPARSE, cuBLAS, cuBLASLt) are
registered through loops, so they are not all individually greppable.

- **Core and PTX unit**: test_yamlish, test_profile, test_driver_version, test_memory, test_ptx,
  test_ptx_link, test_exec, test_exec2, test_exec3, test_fastpath, test_hopper, test_blackwell,
  test_dsmem, test_barriers, test_dynpar, test_regalloc, test_robustness, test_runtime,
  test_e2e_parity, test_sass_decode.
- **Telemetry, RAS, registers**: test_telemetry, test_ras, test_regs, smi_format, smi_queries,
  smi_cli, regs_cli, fault_cli, shell_cli, c_harness_vector_add, c_harness_regs, nvml_api,
  nvml_health, nvml_dcgm.
- **NVIDIA library unit**: test_cudss_solver, test_nvcomp_codecs, test_nccl_symbols,
  test_runtime_user_objects, test_cublaslt_algo.
- **AMD unit**: test_amd_codeobject, test_amd_chip, test_amd_kfd, test_amd_bundle, test_amd_hostcall,
  test_amd_rocprofiler, test_amd_hip_at_exit, test_amd_rocm_smi (+radeon), test_amd_amd_smi
  (+radeon, +`rvs_*`), test_amd_hip_abi, test_amd_hip_grid_*, test_amd_hipcc_*, and
  test_amd_gcn_{exec,ops,math,memory,bytes,int64,atomics,mixed,half,calls,builtins,packed,spill,
  crosslane,doubles,narrow,lds,idioms,asm,abandon,counters,globals,grid}.
- **AMD end to end**: amd_hip, amd_hip_abi, amd_hipcc, amd_hipcc_disasm, amd_gcn_disasm, amd_hsa,
  amd_hip_on_hsa, amd_rocblas, amd_rocblas_disasm, amd_rocm_versions, amd_libraries, amd_hipsparselt,
  amd_debugger, amd_tools, amd_amdsmi_python, amd_rocprofv3, amd_rocprofiler_abi,
  amd_memtest_patterns, amd_workgroup, amd_lost_gpu, amd_nbio, amd_cper, amd_session, amd_vllm,
  amd_ollama, amd_pytorch (per GPU, plus `_compile`, `_models`, `_ck`, `_distributed`, `_multi_gpu`).
- **CUDA runtime end to end**: e2e_vector_add, e2e_symbols, e2e_driver_abi, e2e_runtime_conformance,
  e2e_vgpu_run, e2e_private_loads, isolation_fallback, quiet_semantics, e2e_deferred_errors,
  e2e_per_thread_stream, e2e_stream_identity, e2e_device_limits, e2e_device_heap,
  e2e_occupancy_rules, e2e_reserved_smem_attr, e2e_vmm, e2e_mempool, e2e_ipc, e2e_managed_module,
  e2e_graph_{build,nodes,shapes,conditional,memory,cublas,cusparse}, e2e_capture_{streams,splice},
  e2e_cooperative_grid, e2e_dynamic_parallelism, e2e_device_graph_launch, e2e_rdc_link,
  lint_capture_coverage, lint_shim_symbols.
- **PTX forms and numerics**: e2e_ptx_forms (127 variants), e2e_ptx_sweep (378), e2e_ptx_warp_mem
  (153), e2e_ptx_memory_forms (90), e2e_ptx_half_forms (380), e2e_video_forms (652),
  e2e_mma_forms (123), e2e_mul_add_contraction, e2e_div_approx, e2e_printf_formats,
  e2e_device_{functions,function_barriers,intrinsics,last_error}, e2e_divergent_indirect_calls,
  e2e_atomic_cas, e2e_host_atomics, e2e_warp_spin_lock, e2e_ordered_vectors, e2e_bar_red_named,
  e2e_cg_multi_warp_tiles, e2e_alloca_stack, e2e_stack_limit, e2e_modern_dtypes.
- **Textures and surfaces**: e2e_textures, e2e_texture_{filtering,layers,mipmaps,gather,mip_layers,
  srgb,int_coords}, e2e_border_colour, e2e_surface_oob.
- **Tensor cores, Hopper, Blackwell, CUTLASS**: e2e_mma_layout, e2e_wmma_{gemm,types} (21 shape/type/layout
  combinations), e2e_stmatrix, e2e_wgmma_cute, e2e_tma_{gemm_cute,reduce_cute,im2col},
  e2e_tensormap_replace_cute, e2e_dsmem_cluster, e2e_block_semaphore,
  e2e_cutlass_{hopper,gemm,fmha,sm100,sm100_conv,sm103,sm120}.
- **SASS**: e2e_sass_path, e2e_sass_archs, e2e_nvjitlink_sass.
- **Libraries**: e2e_libraries, e2e_library_goldens, e2e_cublas_tight, e2e_dnn_{paths,classic_paths},
  e2e_mixed_apis, e2e_nccl_{multiproc,group,multiproc_2,comm_ops}, e2e_nvshmem_device, e2e_nvenc.
- **Profiling and tooling**: e2e_ncu, e2e_nvprof, e2e_cupti_activity, e2e_shell_nvcc_native,
  e2e_jit_frameworks, test_matrix_exit.
- **PyTorch, Ollama, workloads**: e2e_pytorch_models_<gpu>, e2e_pytorch_ops_rtx5090, e2e_ollama,
  pantheon_workloads (skips when no pantheongpu/pantheon checkout is present).
- **Faults and registers**: e2e_fault_datapath, nvml_{api,health,dcgm}.

## Gap register (rev 5)

Everything the tree refuses, stubs or does not model that rev 5 could find, grouped by area, each
with the file where the refusal lives so the next audit can re-test it. The rule is the file's
own: a gap that is written down is a decision, and a gap found by its absence is a defect. Counts
are from a case-insensitive grep over `src/`, `nvidia/src/`, `amd/src/`, `include/` and the docs:
roughly 674 `NOT_SUPPORTED`, 755 `unsupported`, 505 `refus*`, 92 `not implemented`, 64 `not
supported`, 31 `not modelled`, 30 `not yet`, 10 `unimplemented`; no `FIXME`, and no real `TODO`
marker (every `todo` hit is a variable name or a pointer to this file). Most sites are the loud
refusals the project wants; the heaviest files are `src/ptx/parser.cpp` (about 400),
`nvidia/src/generated/cublas_64.cpp` (166), `src/exec/interpreter.cpp` (124),
`nvidia/src/runtime_api.cpp` (117) and `cudnn_api.cpp` (106). The register below groups them; it is
not a promise to remove them -- several are right to stay refused (undocumented hardware
behaviour, timing).

### NVIDIA vendor libraries (nvidia/docs/libraries.md "What is not implemented", from line 746)

- **cuBLAS**: `cublasUint8gemmBias`; undeclared exports (`cublas?bdmm`, `Get/SetBackdoor`,
  `Get/SetEnvironmentMode`); cuBLASXt tiles GEMM only, with no CPU offload; emulation controls are
  inert. Generated stubs answer `..._NOT_SUPPORTED` and print "is not implemented by VirtualGPU"
  (nvidia/src/generated/cublas_stubs.cpp:29).
- **cuBLASLt**: FP8 aux scale/amax, per-batch block scales, UE8M0 modes; the block-scaled modes are
  derived from documentation, not checked against a card.
- **cuDNN graph API**: interpolating resample, FP8/MXFP8 attention, block masks, sinks in backward
  attention, INT8x32 reordered filters, multi-GPU norm, MoE/RoPE/band ops, dropout-mask layout; PyTorch's
  cuDNN graphs with ops beyond convolution, matmul, pointwise, reduction, normalization and pooling are
  refused at finalize (nvidia/docs/pytorch.md:76-82). **Classic API**: Volta/Turing fused ops, undocumented
  ops, RNN/attention dropout masks.
- **cuFFT**: legacy callbacks (`CUFFT_NOT_IMPLEMENTED`), LTO-IR callbacks; multi-GPU layouts measured on
  two GPUs only.
- **cuSPARSE**: the `csrmv` family, SDDMM conjugate transpose, SpMMOp (LTO-IR), `csrcolor` colours
  differ from NVIDIA's, `gpsvInterleavedBatch` with algo != 0; solvers compute in double.
  **cuSPARSELt**: FP8/FP4, fp16 compute, GELU outside int8.
- **cuSOLVER**: `Xgeev` left eigenvectors, `csrmetisnd` (no METIS), `csrlsvlu` on device, Mg multi-row
  grids; a list of measured differences from NVIDIA's output in libraries.md.
- **cuTENSOR**: block-sparse (not planned), JIT mode is a no-op. **cuTensorNet**: state API, gradients,
  distributed execution, non-gesvd SVD, half-precision decompositions, capture; cuQuantum Python 26.09
  does not start (static cudart).
- **NCCL**: symmetric-memory windows, the network plugin; stubs that answer `ncclInvalidUsage`:
  `ncclCommRevoke/Grow/GetUniqueId/Suspend/Resume/MemStats`, `ncclPutSignal/Signal/WaitSignal`,
  `ncclDevCommCreate/Destroy`, `ncclGetLsaMultimemDevicePointer`, `ncclGetPeerDevicePointer`; not
  exported: the `nccl*Config` forms, `ncclParam*`, GIN.
- **cuFile**: nvidia-fs DMA, RDMA, userspace-FS handles. **nvCOMP**: Cascaded, Bitcomp and ANS (all
  NotSupported), LZ4 bitshuffle, checksums, CPU/streaming gzip, the hardware decompression engine.
  **NVSHMEM**: MPI/OpenSHMEM bootstrap, multi-node, proxy transports, multimem, host reductions;
  `NVSHMEM_MAX_TEAMS=32`.
- **NVRTC**: CUBIN, LTO-IR and OptiX-IR output, precompiled headers, time traces. **nvJitLink**: LTO-IR,
  SASS+PTX mixes, dead-function removal, re-finalizing sm_100/120, `-G` debug sections, texture refs.
  **nvFatbin**: compression and `nvFatbinAddIndex`.
- **NPP**: watershed, marker-label compression, ResizeSqrPixel super-sampling and Lanczos
  (`NPP_INTERPOLATION_ERROR`). **nvJPEG**: 12-bit, arithmetic, lossless and hierarchical JPEG, the
  hardware backend, EXIF orientation, transcoding. **NVENC**: unimplemented function-table slots
  return `NV_ENC_ERR_UNIMPLEMENTED` (nvidia/src/nvenc_api.cpp:348-355). No NVTX or nvcuvid/NVDEC.
- **Device runtime**: `cudaMemcpyAsync`, `cudaMemsetAsync` and `cudaMalloc` from a kernel,
  `cudaFuncGetAttributes`, `cudaDeviceGetAttribute`, occupancy queries and
  `cudaGetParameterBuffer`/`cudaLaunchDevice` (libraries.md:920-928).

### CUDA runtime and driver (nvidia/src/runtime_api.cpp, driver_api.cpp)

- Memory-pool handle types other than none (runtime_api.cpp:2706, :6959); `cudaHostRegisterReadOnly` (:2855);
  texture/surface channel kinds (:3295-3366); resource views (:3742); `maxAnisotropy` above 1 (:3754).
- Graphs: edge data other than the default (:5126, :7291, :7308, :7315, :7515); clone/parent restrictions
  (:5345-5363, :5770); a CUDA array in a memcpy node (:5931); conditional-graph restrictions (:6199, :6476);
  child-graph ownership (:7476, :7505); external semaphores not modelled (:7438). Capture modes other than
  Relaxed (`cudaThreadExchangeStreamCaptureMode` is a stub, :3053).
- Driver: exec affinity (driver_api.cpp:880), `requestedHandleTypes` (:1330), further stubs (:1485-1500).
  `cudaDeviceGetAttribute` (runtime_api.cpp:1660) and `cuDeviceGetAttribute` (driver_api.cpp:587) answer 0
  for an attribute the profile does not model, with a note on stderr.
- `cuCtxSetCurrent(NULL)` pops instead of clearing (see "Partially implemented").

### PTX and SASS execution

- PTX parser: `.ashift`; sm_107 and sm_107f forms (src/ptx/parser.cpp:1790, :1851, :1860, :1884, :2952,
  :2984-2988); cvt `.rs` for x4; `.hi` for anything but `mul` (:4138); `wmma` kinds other than
  load/mma/store.d (:2218); multi-sample textures (:4472); `suld`/`sust` `.p` (:4534); `tex.level` LOD (:4480);
  `tex.grad`, `tld4` on layered/cubemap and with a level (:4478-4498); signed 8-bit normalized linear
  filtering (src/exec/interpreter.cpp:1326, :1445); anisotropy and resource views (driver_api.cpp:3252, :3261).
- Device printf: `%ls` and `%n` (include/vgpu/exec/device_printf.hpp:115, :122). TMA: sub-byte im2col and
  `.b4x16` alignment (include/vgpu/exec/tma.hpp:50, :64); tensor-map restrictions (tensormap.hpp:210, :262, :351).
- SASS executor (nvidia/docs/sass.md "Coverage"): `LDGMC` (multimem); TMA `im2col::w` (src/sass/exec_ops.inc:3800);
  texture forms with an LOD clamp, offsets, depth compare or LOD bias (exec_ops.inc:2381-2386);
  `WARPSYNC.COLLECTIVE` from divergent paths (:2886); some `SYNCS.ARRIVE` modes (:3631, :3661); tcgen05 forms
  (:4164); a cooperative launch with clusters (src/sass/exec.cpp:818); and five generic "SASS: <op> is not
  implemented yet" sites (exec_ops.inc:857, :1708, :2141, :2238, :3185).
- Blackwell (nvidia/docs/blackwell.md): the `tcgen05.alloc` blocking wait (:36) and the refused forms listed at
  :81, :103, :118, :127, :157, :170, :214, :223, :232-233, :276-292.
- Static cudart, `cuGetExportTable`: blocked (see "Next milestones" 4; nvidia/docs/dark-api.md).
- CUPTI (nvidia/docs/cupti.md): per-API activity controls, the timestamp callback and device-side timestamps
  (:70); no Callback API deliveries, no Event or Profiling metrics, nothing derived from time (:121-139).
- Performance counters and `%pm0`-`%pm7`: refused by design (no timing model).

### AMD (amd/README.md, amd/src)

- Instruction forms refused by name (amd/README.md:212-218, :234, :256-262): `xf32` MFMA forms, broadcast
  modifiers, a wave with lanes off, cross-wave DPP forms, filling a sub-dword destination with sign or
  preserving it, the output multiplier, and a packed op needing a constant's second half; BLGP lane patterns
  (:513, :515); decoded-but-not-implemented sites in `amd/src/gcn_exec.cpp` (:1360, :1391, :1410, :1467, :1817,
  :2164, :3276, :3366, :3741, :3975, :4336, :4644, :4653, :4870, :5217, :5623, :5713); GWS semaphores (:5447),
  texel offsets (:4225), cube arrays (:4161), a flat access reaching LDS or private memory (:3896); GDS
  (amd/include/vgpu/amd_exec.hpp:109); MFMA input rounding not modelled (amd/src/gcn_decode.cpp:982); the 6-bit
  form 0xe1 (:760); undecoded RDNA opcodes, operands and ray-tracing image instructions
  (amd/src/rdna_decode.cpp:148, :279, :335, :563).
- Runtime: hostcall services other than printf, i.e. device malloc and the address sanitizer
  (amd/src/hostcall.cpp:121); linking several code objects into one (amd/src/hip_api.cpp:6982); stream-capture
  refusals (amd/src/hip_graph.inc:942, :2638); sRGB over linear memory (amd/src/hip_images.inc:742); compressed
  offload bundles need zlib or zstd and refuse other methods (amd/src/bundle.cpp:43-146); graphics interop
  (`hsa_amd_image_create`, `hsa_amd_interop_map_buffer`, `hsa_amd_interop_unmap_buffer`,
  `hsa_executable_agent_global_variable_define`, amd/src/hsa_api.cpp:1984-1991).
- Profiling: no PC sampling, thread trace or HSA trace, and no records of the runtime's internal kernels
  (amd/README.md:576).
- Workloads and frameworks: `fused_attention` does not build for AMD (needs a 64-bit mask on wave64); `rt_virus`
  and `media_enc_virus` skip; `mma_virus` needs rocWMMA headers (amd/README.md:609-615); vLLM runs eager only
  (no graphs, no `torch.compile`), one short sequence, 3 GB device (:513, :515); the ROCm Validation Suite was
  never built or run, only read (:643).
- Open on the AMD list: rocBLAS double `trsv` (see "Next milestones" 0). OpenMP offload (`amd_openmp`), OpenCL (`amd_opencl`) and `offload-arch` are done; gfx90a image instructions (MIMG on gfx9) are not, though MI250X reports image support.

### Health, RAS and registers (docs/telemetry.md, docs/registers.md, docs/machine-simulator.md)

- NVML/NVIDIA: the far end of an NVLink is not modelled and its index is NOT_SUPPORTED (telemetry.md:145);
  NVLink error counters read zero and `nvlink -s` speeds exist for NVLink 3 and 4 only (:176); ECC mode cannot be
  changed (:181); some paths cannot arm ECC errors (:236); `--on alu` is bit flips with no counting (:210);
  rocm-smi's RAS table columns are not checked against a card (src/cli/smi.cpp:1716); resetting ECC errors
  (:2232); retired-page addresses are placeholders (telemetry.md:45).
- AMD: a lost GPU's `/sys/class/drm` and hwmon entries stay (telemetry.md:349); AMD ECC, memory-sensor and link
  values follow AMD's documentation and are not checked on a real MI-series card (:425); the CPER debugfs ring and
  fatal/bad-page-threshold records (:449); amd-smi `--csv`, `ras --follow`, `ras --afid`
  (src/cli/amdsmi.cpp:465, :529, :532, :540); violation status, PCIe throughput, bad-page threshold and every
  setter and reset are `NOT_SUPPORTED` (amd/src/amd_smi.cpp:466); fabric/SoC/video/display clocks
  (amd_smi.cpp:928, rocm_smi.cpp:758-762); `cu_occupancy` (amd_smi.cpp:1079); VBIOS, serial, firmware and
  energy are N/A in a session and `/dev/kfd` and `/dev/dri` do not exist outside isolated mode
  (docs/machine-simulator.md:137-148).
- DCGM: its NVML reads are answered; DCGM's own binaries, CUDA and profiling-counter needs have not been run.
- Registers (docs/registers.md): partition switching (:200); AMD UMC bases are assumed, other channels and
  instances and MCA address/syndrome/IPID are unmapped (:228-235); the NVIDIA topology rows, PRI errors, temperature
  sensors, Ampere/Ada memory-controller ECC counters and Blackwell's `dev_ltc_zb`, `dev_fuse_zb` and `dev_tmr`
  have no published values (:345, :367-401); per-partition count distribution (:377); header-defined firmware
  actions (amd/src/regs.cpp:139).

### Found by the rev 5 gap analysis and not implemented

NVIDIA (found by comparing the names in `cuda_runtime_api.h`, `cuda.h` and `nvml.h`, 12.0 to 12.9,
with what `nvidia/src` defines):

- **Driver graph API and stream capture** (about 100 entry points): absent, and `cuStreamGetCaptureInfo`
  is a stub. The driver's streams are synchronous and keep none of the runtime's roughly 3000 lines of
  graph machinery. The largest remaining gap for driver-API users.
- **Stream memory operations**: `cuStreamWaitValue32/64`, `cuStreamWriteValue64`, `cuStreamBatchMemOp`. A
  wait on a false condition cannot be modelled on a synchronous stream.
- **Other driver additions**: `cuLibraryLoadFromFile`, `GetGlobal`, `GetManaged`, `GetKernelCount`,
  `EnumerateKernels`; `cuModuleGetFunctionCount`, `EnumerateFunctions`; `cuFuncGetName`, `GetModule`,
  `GetParamInfo`; `cuKernelGetLibrary`, `GetParamInfo`; `cuStreamGetCtx`, `GetId`; `cuEventRecordWithFlags`;
  `cuCtxRecordEvent`, `WaitEvent`; `cuMemRangeGetAttribute(s)`; `cuMemcpy3DPeer`; `cuDeviceGetP2PAttribute`,
  `GetLuid`; `cuUserObject*`; green contexts; the texture-reference mipmap getters.
- **Runtime leftovers**: `cudaGetFuncBySymbol`, `cudaGetKernel`; graph kernel-node attribute calls, the
  generic `cudaGraph*NodeSetParams`, the 12.3 `_v2` edge-data graph calls; the external memory and semaphore
  interop family; `cudaMemcpyBatchAsync`; `cudaOccupancyAvailableDynamicSMemPerBlock` (the semantics for
  infeasible requests are unclear); `cudaDeviceGetTexture1DLinearMaxWidth`; `cudaSetValidDevices`.
- **Cluster occupancy**: `cudaOccupancyMaxActiveClusters` and `cudaOccupancyMaxPotentialClusterSize` need the
  SM-to-GPC grouping, which no profile records and the public documentation does not give; they refuse by
  name once the runtime-exports PR (#309) merges.
- **NVML setters**: persistence mode, compute mode, power limit, clock locks, fan control, MIG/GPU-instance
  and vGPU management, GPM and units are absent, and `nvidia-smi` has no `-pm`, `-pl` or `-c`. Deliberately
  not done: an in-process-only setter would make `nvidia-smi -pm 1` look successful while the next process
  shows no change, so persistence across processes needs a design decision first.
- **PTX**: `multimem` (needs the multicast fabric, which the driver refuses), `sured`, `txq`, `ldu`.

AMD:

- **HSA**: `hsa_amd_signal_wait_any` (what its timeout returns is not settled by the public header),
  `hsa_amd_queue_intercept_*` (what rocprofiler uses; needs queue interception),
  `hsa_amd_register_deallocation_callback` and its deregister, `hsa_amd_memory_migrate`, `hsa_amd_spm_*`,
  `hsa_amd_ipc_signal_*`, the deprecated `hsa_code_object_*`, `hsa_executable_create` and the finalizer calls,
  and `hsa_ven_amd_aqlprofile_*`.
- **AMD SMI data with no profile datum**: board, VBIOS and firmware info, cache info, violation status and PCIe
  throughput stay `NOT_SUPPORTED`; filling them in would mean guessing field values. ROCm SMI's XCD counter
  stays refused because the metrics table fills all 8 gfxclks even on Radeon, so a count would be wrong.
- **ISA**: MFMA with BLGP lane patterns outside the f8f6f4 forms, MFMA with lanes off in EXEC, a few
  high-half-constant forms, and `s_sendmsg_rtn` messages beyond the ones answered. The public sources
  consulted did not settle the BLGP semantics, so they stay refused by name. (The GEMM forms that `TODO.md`
  used to list as missing -- half, bf16, int8, fp8 MFMA and SMFMAC for gfx90a, gfx942 and gfx950, the f8f6f4
  and scaled forms, DPP wave and row broadcast forms -- already exist.)
- **HIP API surface**: no gap found against the 5.7 header (only macros and templates were missing).

What rev 5 could not check: nothing here ran on a real GPU or on ROCm 7.x. The static audit reads sources
and tests; the gap branches were tested by their authors under ASan + UBSan with CUDA 12.8 headers (the
runtime one also syntax-checked against 12.4), and none ran the full ctest suite, a TSan build, or any e2e
program that needs nvcc.

### PyTorch sweep: known failures

`nvidia/tests/pytorch/sweep/known_failures.txt` lists the sweep's checks that do not match the CPU on the
simulator (nvidia/rtx5090). The CI-wired `e2e_pytorch_sweep` prints each as `XFAIL` with its numbers and fails
only on a new failure or on a listed one that starts passing (`XPASS`: delete its line). Each is work to do:

- **Graphs (4 checks)**: capture and replay of cuDNN + cuBLAS, a whole training step, Adam with
  `capturable=True`, `make_graphed_callables` all fail with "operation failed due to a previous error during
  capture". Which call errors under stream capture is not identified (cuDNN computes on the host, which a
  capture cannot record).
- **torch.compile `reduce-overhead`**: 1.2 scaled difference from the CPU (allowed 0.001); it replays a captured
  CUDA graph, probably the same gap.
- **torch.compile gather / scatter_add / index_select**: an Inductor kernel fails to load, `cuModuleLoadData`
  answers `unsupported-ptx` (the log shows only that line, not the unsupported feature) and the driver call
  returns "operation not supported". Find the PTX feature first.
- **Numeric**: the tiny causal transformer after three AdamW steps (0.0027 against 0.002). The kernel or
  reduction order that drifts is not isolated; no tolerance was loosened. (SGD with OneCycleLR and gradient
  clipping, and `clip_grad_norm_` (foreach) / `clip_grad_value_`, drifted when the sweep was written and match
  the CPU on the current main; they are no longer listed.)

### Tooling, CI and process

- Trace record/replay and a conformance database with compatibility scores: not started.
- A memory-aware adversarial scheduler (feeding the race detector's shadow state back into scheduling): not
  started ("Next milestones" 2).
- LLVM JIT (ARCHITECTURE.md D1): not started.
- Real-GPU card check (docs/ci-card-check.md): designed on a 3080 Ti, not enabled.
- AMD rows in `workloads.yml` and its `coverage` check: absent.
- The hip-tests disabled-case lists and the SASS-vs-PTX digest differences (see "CI inventory"): unreconciled.
- Pantheon workloads: 44 of 46 pass and `media_enc_virus` and `rt_virus` are skipped, per
  docs/pantheon-workloads.md; that document's status block and prose disagree about `media_enc_virus`.
- GitHub Action constraints (docs/github-action.md:82-124): it must use the shared cudart, needs
  `library-path: false` when linking vendor libraries, Ubuntu 22.04's CUDA 11.5 is too old (so 12.6 is
  installed), and its timings mean nothing.
- External suites: `gpu-burn` and `cuda-memtest` exist as PR #290 (not yet merged when rev 5 was written); until it
  merges the NVIDIA memory tests are the pantheon `memory_*`, `galpat`, `march_test` and `memory_hammer`
  workloads, and the AMD ones are `amd_memtest_patterns`.

### Documents that disagreed with the code at rev 5

Fixed in the same change as this file unless marked: README.md's "SASS-only fatbins are rejected" (:67) and
its roadmap (:278-285); docs/telemetry.md:454-456 and `src/cli/serve.cpp:8` ("AMD execution is not
implemented"); README.md:60 on Nsight; the header comment of `src/exec/interpreter.cpp` (barriers in
divergent code, IPDOM); `vgpu counters`' spill note (`src/cli/main.cpp:103-114`, left as a comment in this
file's "Registers and counters" because it is output text, not documentation).
