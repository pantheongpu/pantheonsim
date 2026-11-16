# Gap register: what is refused, stubbed or not modelled

The detail behind [TODO.md](../TODO.md): the out-of-scope list, the partially implemented features, the loud refusals, the milestone
notes and the gap register of revision 5 (text of 2026-10-22, unedited; each block is preceded by an HTML comment naming its line range in
the old TODO.md). Entries are older than some of the code, so search the tree for the name before relying on one: the driver graph API
(`nvidia/src/driver_graph.cpp`), stream memory operations and the `cuLibrary*` / `cuFunc*` additions listed as missing under "Found by the
rev 5 gap analysis" exist now. The finished work that these entries sit beside is in [history/TODO-done.md](history/TODO-done.md).

<!-- old TODO.md lines 898-1021 -->
## Known out of scope (not CUDA)

- `rt_virus` needs **OptiX** (NVIDIA's ray-tracing library, loaded from
  libnvoptix.so.1), a separate NVIDIA subsystem, not CUDA; emulating it is a
  distinct project. It takes its documented "driver not installed" path (and is
  skipped on hosts with the real driver libraries): `OUT_OF_SCOPE="rt_virus"` in
  `tests/workloads/run_pantheon_workloads.sh`, docs/pantheon-workloads.md.
  (`media_enc_virus` runs there: `libvgpunvenc` implements NVENC, now writing
  real (lossless PCM) H.264 and HEVC streams that are identical for identical
  frames, as the workload's golden-bitstream check needs. It is skipped only when
  the NVENC shim was not built, and always by the older
  `scripts/run-pantheon-workloads.sh`.)
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
  control flow is not handled, and nothing pins that with a test.
- (cuCtxSetCurrent(NULL) pops the top of the context stack; the card does the same, measured and checked by
  `driver_context_gaps`, so this is not a bug.)
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
  space. Next: map more of BAR0 as headers publish it, and capture data-center and AMD Instinct cards (the RX 6800 and RX 6700 XT are captured in amd/registers/measurements). The
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

<!-- old TODO.md lines 1297-1351 -->
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
  - TMA's `.im2col::w` modes (Blackwell; `cp.async.bulk` in parser.cpp, and on SASS
    src/sass/exec_ops.inc, "im2col::w tensor copies are not implemented"); see nvidia/docs/blackwell.md.
  - TMA attribute overrides and reports (the `cp.async.bulk` and `cp.reduce.async.bulk` modifier checks in parser.cpp).
  - Interleaved tensor-map layouts and the 128B swizzle's 8-byte-flip variant
    `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B` (tensormap.hpp, the `swizzle == 5` checks at :276 and :363).
  - tcgen05's `.ashift` (parser.cpp, `tcgen05.mma.ashift`), `.ws` with A in Tensor Memory below
    M = 128, and `.ws.sp` (interpreter.cpp, the two `refuse(".ws ...` calls in the tcgen05.mma checks) -- the ISA draws neither
    layout.
  - sm_107's `tcgen05.mma` `.decompress` modifiers and `tcgen05.commit`'s `.multicast::cluster::32b`/`.sync_restrict` (`kind::ti16` itself is implemented, see below).
  - Inline-asm-only instructions: anything the parser has no name for is refused
    by that name, not run as something else.
  - Not on this list any more: the TMA out-of-bounds NaN fill (`oobFill`) is
    *accepted* -- stored in the descriptor, carried by `tensormap.replace`
    (interpreter.cpp, `oob_nan`) and honoured by loads that go out of bounds
    (interpreter.cpp, `map.oob_nan` in the tile load; tests/unit/test_hopper.cpp). What is not
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
  `hipMemcpyAsync` (amd/src/hip_api.cpp, `hipMemcpyAsync`).
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
  "Known out of scope"); CUPTI's Callback API covers the runtime, driver, resource, synchronize and NVTX domains (driver calls outside the documented list, and driver-API pool and graph calls, raise none), the Event
  and Profiling metrics APIs are absent, and nothing derived from time is
  reported (nvidia/docs/cupti.md, "What is not implemented, and why").

<!-- old TODO.md lines 1781-1907 -->
## Next milestones (order)

0. **AMD** -- the list this item held is done; rev 5 checked each entry
   (amd/README.md has the evidence). rocBLAS runs 162,807 of its own quick
   float and double tests, and its half, bfloat16, int8 and FP8 GEMMs run;
   hipBLASLt, MIOpen, rocPRIM/hipCUB (`amd/tests/hipcc/prim.cpp`), rocRAND, rocFFT,
   rocSPARSE, rocSOLVER (`amd/tests/libraries`, `amd_libraries`), hipSPARSELt,
   Composable Kernel (`amd_pytorch_ck`) and RCCL, single- and multi-process with
   IPC memory handles (`amd_pytorch_distributed`, `amd_pytorch_multi_gpu`), run
   under PyTorch for ROCm on six GPUs (`amd_pytorch_*`) and vLLM (`amd_vllm`);
   pinned and managed memory and asynchronous streams are done; the HSA/KFD
   layer (rocminfo, `rocm_agent_enumerator`, `/sys/class/kfd`) is done
   (`amd_hsa`, `amd_hip_on_hsa`, `test_amd_kfd`); textures and images are done
   on RDNA (MI300 has none, correctly); gfx950 (MI350X) is done
   (`amd_pytorch_mi350x`); OpenMP offload and OpenCL run on all six
   (`amd_openmp`, `amd_opencl`). **Still open on AMD**: the instruction forms the executor refuses by name
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
   14 NVIDIA profiles and three AMD ones (`mi325x`, `rx6800`, `rx6700xt`) are verified.
   Open: `h200`, `b200`, `b300`, `rtx5090`, the rest of the "Profile inventory"'s
   **no** rows and seven AMD profiles (Hardware characterization).
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
   CUPTI, loaded by absolute path from its install directory. Presenting the
   shim's libcupti at that path (a private mount) loads, and exposes an
   undocumented CUPTI function the injection library requires, then a host-side
   failure to recognize the GPU (findings in nvidia/docs/cupti.md) -- so `nsys`
   produces a report with OS runtime traces and no CUDA data. nvprof works, because its path goes through
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
   of results against the hardware's. `tex.grad` is done for 1D and 2D
   textures (layered or not) of power-of-two size, bit for bit (round 4; see "Textures" in nvidia/docs/textures.md).
   (BC1 to BC7, 10:10:10:2, resource views, any anisotropy, linear
   filtering of signed 8-bit normalized texels and `tld4` on layered and
   cubemap textures are done: nvidia/docs/textures.md). The `.clamp`/`.zero` surface policies are done, as an
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

<!-- old TODO.md lines 2090-2411 -->
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
The `file:line` and `docs:line` references in this register are from rev 5 (2ce1f45) and have drifted since; search for the quoted name or section instead of jumping to the line.

### NVIDIA vendor libraries (nvidia/docs/libraries.md "What is not implemented", from line 746)

- **cuBLAS**: undeclared exports (`cublas?bdmm`, `Get/SetBackdoor`, `Get/SetEnvironmentMode`);
  cuBLASXt tiles GEMM only (its CPU share works for GEMM); FP64 fixed-point emulation works under
  EAGER (FIXED control bit-exact to the card; DYNAMIC chooses its own bit count), BF16x9 does not exist.
  Generated stubs answer `..._NOT_SUPPORTED` and print "is not implemented by VirtualGPU"
  (nvidia/src/generated/cublas_stubs.cpp:29).
- **cuBLASLt**: FP8 aux scale/amax, per-batch block scales, UE8M0 modes; the block-scaled modes are
  derived from documentation, not checked against a card.
- **cuDNN graph API**: interpolating resample beyond bilinear upsampling by 2 (the one configuration with an
  engine), FP8/MXFP8 attention, block masks, INT8x32 reordered filters, multi-GPU norm gradient scaling with more than two GPUs (assumed),
  MoE backward / band ops / standalone RoPE (no engine on the RTX 3060; Hopper/Blackwell unchecked). Done in
  round 4: multi-GPU norm across processes (shared pinned file on the card, IPC memory on the sim; `e2e_dnn_multigpu_norm`), the unified SDPA node's dropout mask, element for element (Philox4x32-7, `e2e_dnn_sdpa_mask`). Done in round 3: multi-GPU norm (threads of one process),
  bilinear 2x, classic/RNN/attention dropout bit for bit (cuRAND XORWOW states), SCALE_BIAS_ACTIVATION_WGRAD.
  **Classic API**: the fused ops marked "reserved for future use", (RNN dropout with padded I/O: done in round 4, `e2e_dnn_rnn_dropout`.)
- **cuFFT**: legacy callbacks (`CUFFT_NOT_IMPLEMENTED`), LTO-IR callbacks; multi-GPU layouts measured on
  two GPUs only.
- **cuSPARSE**: SDDMM conjugate transpose (NVIDIA's computes garbage for complex), SpMMOp (LTO-IR),
  `csrcolor` colours differ from NVIDIA's (documented properties checked on the card),
  `gpsvInterleavedBatch` with algo != 0 (documented as unsupported); solvers compute in double. The
  `csrmv` family is not in NVIDIA's CUDA 12/13 libraries, so there is nothing to match.
  **cuSPARSELt**: FP8/FP4, fp16 compute, GELU outside int8.
- **cuSOLVER**: `Xgeev` left eigenvectors, `csrlsvlu` on device and Mg multi-row grids (NVIDIA's own
  library has none of the three); `csrmetisnd` runs a vendored METIS 5.1.0 (77/78 test permutations
  equal the card's); a list of measured differences from NVIDIA's output in libraries.md.
- **cuTENSOR**: block-sparse (not planned), JIT mode is a no-op (the card's kernel cache fills after
  a batched contraction; ours stays empty). **cuTensorNet**: state API, gradients,
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
- **NVRTC**: without the toolkit's libnvrtc, LTO-IR, OptiX-IR, Tile IR, precompiled headers and the
  time traces (refused by name; with it every output is NVIDIA's own). **nvJitLink**: without the
  toolkit's libnvJitLink, LTO-IR, index files and SASS+PTX mixes (refused by name); re-finalizing
  sm_100/120 code from the mercury sections (the format is undocumented; the frame table of a dropped
  function is cleared, not removed, there), legacy texture/surface references, and NVIDIA's shared-memory
  layout across a kernel and device functions in two modules (valid, not the same offsets).
  **nvFatbin**: `nvFatbinAddIndex` (NVIDIA's takes an index file no tool can write), and compressed
  bytes identical to NVIDIA's (the same flags and sizes within a few percent).
- **NPP**: ResizeSqrPixel super-sampling and Lanczos (`NPP_INTERPOLATION_ERROR`); watershed labels
  where neighbouring values are equal and 4-way watershed beyond 4x4 are inexact. **nvJPEG**: 12-bit,
  arithmetic, lossless and hierarchical JPEG (the card refuses or gets them wrong), the hardware
  backend, EXIF orientation. **NVENC**: AV1, 10-bit and 4:4:4
  input, P/B pictures and rate control (H.264 and HEVC are lossless PCM IDR pictures). **NVDEC**: every codec but Motion
  JPEG; progressive JPEG (refused as the card refuses it). **nvCOMP**: Cascaded, Bitcomp and ANS
  (unpublished bitstreams), the container checksums, `LZ4CPUManager`, LZ4 bitshuffle through the
  managers. **cuFile**: nvidia-fs DMA. No NVTX.
- **Device runtime**: `cudaMemcpyAsync`, `cudaMemsetAsync` and `cudaMalloc` from a kernel,
  `cudaFuncGetAttributes`, `cudaDeviceGetAttribute`, occupancy queries and
  `cudaGetParameterBuffer`/`cudaLaunchDevice` (libraries.md:920-928).

### CUDA runtime and driver (nvidia/src/runtime_api.cpp, driver_api.cpp)

Done in round 3 (branch r3-runtime-driver), each against the RTX 3060: capture modes (Global, ThreadLocal, Relaxed and
`cudaThreadExchangeStreamCaptureMode`: `runtime_capture_modes`); memory-pool handle types, read-only host registration,
exported VMM memory as a file descriptor (`runtime_memory_gaps`); graph edge data, clone/parent restrictions, a CUDA
array in a memcpy node, child-graph ownership and external semaphore nodes (`runtime_graph_gaps`); driver contexts, flags,
ids, exec affinity, limits, multicast stubs (`driver_context_gaps`);
normalized, block-compressed (BC1 to BC5) and 10:10:10:2 texture formats, resource views and any `maxAnisotropy`, in the
runtime and the driver (`runtime_texture_gaps`, `driver_texture_gaps`). The conditional-graph restrictions were already
the card's, an attribute id the device-attribute table does not know is an error (`CUDA_ERROR_INVALID_VALUE`), not a 0,
and `cuCtxSetCurrent(NULL)` removes the current context from the stack as a pop does -- that is what the card does
(checked in `driver_context_gaps`), so the old register entries for the last two were not bugs.

Still open:

- (Done in round 4: BC6H and BC7 textures. The card decodes both exactly as the Khronos Data Format Specification's
  BPTC chapter says, apart from BC6H's signed zero; `bc-modes/*` in `runtime_texture_gaps` and `driver_texture_gaps`
  hash 512 blocks of every mode, point sampled and filtered.)
- External memory and semaphore import (needs Vulkan, Direct3D or NvSciBuf; the card's runtime crashes on invalid handles).
- Cluster occupancy on a part whose GPC count NVIDIA does not publish (H100 PCIe, B200, B300, GB200, GB10, Thor, Rubin): refused by name; see nvidia/docs/clusters.md.

### PTX and SASS execution

Done in round 3 (branch r3-ptx-sass; each with a test, and the texture, surface and printf ones
bit for bit against an RTX 3060): the offset, depth-reference, destination-predicate and half-precision
operands of `tex` and `tld4`, `tld4` on layered, cube and cube-array textures, linear filtering of
signed 8-bit normalized texels (an exact fit), `sust.p`, device printf `%ls` and `%n`, a cooperative
launch of clusters on SASS, and -- from PTX ISA 9.4, derived from the documentation and not checked
against a card (no ptxas or GPU for sm_107) -- `spcompress`/`spdecompress`, `tcgen05.ld{.red}.spcompress`,
`tcgen05.mma.kind::ti16`, `cvt` `.rz` / `.pzo` / `.scaled::n1::ue8m0`, the packed `.u8x4`/`.s8x4`/`.u16x2`/`.s16x2`
integer instructions (the SASS side of them on sm_120f compared with a host loop), the four-wide narrow-float
`add`/`sub`/`mul`/`fma` of sm_100a and sm_103a (PTX engine only: ptxas refuses them), K = 64 for the 8-bit tcgen05 kinds, the 128-lane scale-factor A layout, and UE4M3
with `.block32`. Not items at all: `wmma` has only `load`, `mma` and `store.d` in the ISA, and `.hi` exists
only for `mul`, `mad`, `mul24` and `mad24`; the five "not implemented yet" faults in `src/sass/exec_ops.inc`
are unreachable for the opcodes `executes()` lists; `tcgen05.alloc`'s blocking wait was done long ago.

Still open, each with the reason:

- `tex.grad` on cube textures (the model is in `nvidia/docs/textures.md` and matches the card on 31,083 of
  31,083 fetches with the card's own `MUFU.RCP`, but the simulator's reciprocal is the correctly rounded one and the
  card's differs by an ulp on 13% of inputs: a table with a quadratic interpolation, not reproduced when this was
  written; the table is modelled since 2026-11-15, `numerics/math-functions.md`, and `tex.grad` has not been re-measured with it) and with
  `maxAnisotropy` above 1; `tex.2dms`/`tex.a2dms`: CUDA cannot make a multi-sample texture; BC6H and BC7 texture
  objects (the arrays work; see the runtime and driver section).
- `cvt.rs` to the x4 types: figures 41 and 42 do not say how a and b split their shared random bits;
  `.ue5m3x2` and UE5M3 scale factors: the ISA gives no exponent bias; `tcgen05.mma.ashift` and
  `decompress::lut::b`: only figures; `.kind::mxf4`'s K = 128 and sparsity version 1; K = 128 sparse for the
  8-bit kinds; the 128-lane A layout over a CTA pair at M = 128.
- TMA: `im2col::w` / `im2col_no_offs::w` (figures 16-20 only), im2col of packed sub-byte types, interleaved
  layouts and `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B` (`tensormap.hpp`), a `.b4x16` copy that does not start
  on a group of 16 (the ISA is silent).
- SASS executor: `LDGMC` / `multimem` (needs `cuMulticast*`, which are the runtime's and refuse), texture
  fetches with a LOD bias or clamp and per-texel gather offsets (no PTX produces them), `WARPSYNC.COLLECTIVE`
  from different paths (no document, no sample), `SYNCS.ARRIVE` modes 6 and 7 and `UTCCP` shapes 1 and 7
  (no PTX produces them).
- Static cudart, `cuGetExportTable`: blocked (see "Next milestones" 4; nvidia/docs/dark-api.md).
- CUPTI (nvidia/docs/cupti.md, "What is not implemented, and why"): Unified Memory counters, the `OVERHEAD`
  kinds beyond function loading and buffer request, driver-API pool and graph callbacks, no Event or Profiling
  metrics, nothing derived from time.
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
- Runtime: the address sanitizer's hostcall report is checked against the device library's argument
  order only, no instrumented code ran it (amd/src/hostcall.cpp); linking several code objects into one (amd/src/hip_api.cpp:6982); stream-capture
  refusals (amd/src/hip_graph.inc:942, :2638); sRGB over linear memory (amd/src/hip_images.inc:742); compressed
  offload bundles need zlib or zstd and refuse other methods (amd/src/bundle.cpp:43-146); graphics interop
  (`hsa_amd_image_create`, `hsa_amd_interop_map_buffer`, `hsa_amd_interop_unmap_buffer`,
  `hsa_executable_agent_global_variable_define`, amd/src/hsa_api.cpp:1984-1991).
- Profiling: PC sampling is host-trap only and by instruction count (no clock); no stochastic PC sampling, thread trace or HSA trace, and no records of the runtime's internal kernels
  (amd/README.md:576).
- Workloads and frameworks: `fused_attention` does not build for AMD (needs a 64-bit mask on wave64); `rt_virus`
  and `media_enc_virus` skip; `mma_virus` needs rocWMMA headers (amd/README.md:609-615); vLLM with graphs,
  `torch.compile` and Triton works (amd_vllm_graphs, amd_pytorch_graphs, amd_pytorch_compile); the ROCm Validation Suite runs
  (amd/README.md, "ROCm Validation Suite"), except `perf`, `iet` and `tst`, whose pass marks are rates against time, power and heat the simulator does not model, and `mem`, which takes six minutes a GPU.
- Open on the AMD list: OpenMP offload (`amd_openmp`), OpenCL (`amd_opencl`) and `offload-arch` are done. gfx90a image instructions are done, except the `_pck` forms and LWE (amd/README.md, "Textures"). rocBLAS double `trsv`/`trsm` was checked and nothing was found wrong.

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
- **Cluster occupancy** is answered where the profile has GPC and TPC counts from NVIDIA (derived spread of the SMs over the GPCs, said once on stderr); parts without them are refused by name. Nothing is checked against a card with clusters. See nvidia/docs/clusters.md.
- **NVML setters**: persistence mode, compute mode, power limit, clock locks, fan control, MIG/GPU-instance
  and vGPU management, GPM and units are absent, and `nvidia-smi` has no `-pm`, `-pl` or `-c`. Deliberately
  not done: an in-process-only setter would make `nvidia-smi -pm 1` look successful while the next process
  shows no change, so persistence across processes needs a design decision first.
- **PTX**: `multimem` (needs the multicast fabric, which the driver refuses), `sured`, and the `txq` queries other than width, height and depth (a kernel using one for a texture object does not launch on an RTX 3060).

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
only on a new failure or on a listed one that starts passing (`XPASS`: delete its line). One is left, and it is
not work for the simulator:

- **Numeric**: the tiny causal transformer after three AdamW steps (0.0024 to 0.0027 against 0.002) is not a
  simulator bug and stays listed. The key bias of each attention layer has a gradient that is zero in exact
  arithmetic (softmax is unchanged when the same number is added to every key) and ~1e-9 of rounding noise in
  float; AdamW divides it by its own square root, so every device turns its own noise into a +-lr step. A real
  RTX 3060 running the same PyTorch differs from the CPU by about 0.0020 the same way (the same two
  `in_proj_bias` tensors), and with those gradients zeroed the CPU, the simulator and the 3060 agree
  (`tiny causal transformer ... key biases left out of the update`, a full-tier check, passes). No tolerance was
  loosened. The entry can only go if the check changes what it tests.
  2026-11-15: the entry went another way. With the model of the card's approximate-math unit (`numerics/math-functions.md`)
  the simulator's noise changed and the check is within 0.002 on the CI runner's PyTorch (the sweep printed XPASS, which
  fails until the line is removed; the new difference was not printed). The line is now a comment in `known_failures.txt`.
  It is still noise: another PyTorch, or `VGPU_MUFU_EXACT=0`, can bring it back.

### CUDA Samples known differences

The nightly's `cuda-samples` job (ci/external/suites.sh, ci/external/cuda-samples.txt) compares each of
NVIDIA's CUDA Samples v13.0 with what an RTX 3060 printed. `ci/external/cuda-samples-known.txt` lists the
programs that still differ, each with a class and a reason. It is a ratchet: a listed program that still
differs is reported, one that starts matching fails the job until it is taken off, and any other difference
fails as before. What was found on 2026-10-15 (run 38576253348 listed 18 differences; the others of
the 133 matched):

Fixed in the simulator, each with a test that fails without the fix:

- `cudaCompressibleMemory`: `CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED` answered 0, where an RTX 3060,
  L4, L40S and RTX PRO 6000 answer 1 and hand out memory whose properties say it is compressed, so the sample
  waived itself (exit 2). `e2e_vmm`.
- `MersenneTwisterGP11213`: `curandCreateGeneratorHost` copied its numbers to the caller's host array as if
  it were device memory, so the reference was zeros (`L1 norm: INF`). `e2e_rand_paths`.
- `convolutionFFT2D`: cudaMalloc pointers were 256-byte aligned and a texture over one needs 512
  (`cudaCreateTextureObject` refused the second test's buffer). `e2e_malloc_alignment`.
- `quasirandomGenerator` and its NVRTC twin: the SASS decoder read the byte select of `R2P` (`R2P PR, R5.B1`,
  `.B2`, `.B3`, which ptxas makes of an unrolled shift-and-test loop) from the wrong bits from Ampere on, so
  every group of tests after the first repeated the first group's. `e2e_constant_table_xor`.
- `conjugateGradientCudaGraphs`: a cuBLAS call captured into a graph replayed in the handle's pointer mode at the
  replay, not the one it was made in; the sample sets host mode for one `cublasSaxpy(&alpha)` and device mode
  after it, so the host alpha was fetched as a device pointer, read as zero, and the loop diverged (residual up
  by about 1025 an iteration, 10000 iterations, a timeout). `e2e_graph_cg_capture` compares the captured
  conjugate-gradient loop with the call-by-call one, scalar by scalar.

Fixed in the suite: `cuSolverDn_LinearSolver`, `cuSolverSp_LowlevelCholesky` and `cuSolverSp_LowlevelQR`
had a `|` in their markers, which split the list's line; and the nine cuSPARSE and cuSOLVER C++ samples did
not link where `libnvJitLink.so.13` is not installed (see `build()` in suites.sh).

Still different, each listed in `cuda-samples-known.txt`:

- `UnifiedMemoryPerf` (simulator gap): every launch runs to completion inside the launch call (ARCHITECTURE: the
  engine is synchronous), so a kernel that waits for the host to do something after launching it never ends,
  and the step budget stops it (`a warp ran 1073741824 instructions`, then `cudaErrorLaunchTimeout`).
  Reproducer:
  `__global__ void spin(volatile unsigned* f) { while (*f < 1); }  int main() { unsigned* f; cudaMallocHost(&f, 4); *f = 0; cudaStream_t s; cudaStreamCreate(&s); spin<<<1,1,0,s>>>(f); *f = 1; cudaStreamSynchronize(s); }`
  finishes on a card and hangs here. The fix is kernels that run beside the host thread (a worker per stream, with
  events and `cudaStreamWaitEvent` ordering them): `nvidia/src/runtime_api.cpp` (the launch entry points) and
  `src/runtime/runtime.cpp`. Large, and every eager-error and capture path assumes the synchronous engine.
- `simpleCUFFT_callback` (cannot match): the sample links `libcufft_static.a`, which carries NVIDIA's own CUDA
  runtime; that asks the driver for its internal export table (`cuGetExportTable`), which a simulated driver
  does not have, and `cufftMakePlan1d` returns `CUFFT_INTERNAL_ERROR`. The simulator's cuFFT shim has legacy
  callbacks of its own (`e2e_fft_legacy_callbacks`), for programs that link the shared library.
- `quasirandomGenerator` and `quasirandomGenerator_nvrtc`: with the decoder fixed the simulated GPU's numbers
  agree with the CPU's to an L1 norm of 7.3e-12, where the card printed 9.439909E-08. The sample's CPU reference
  is built by the sample's own host code from a table it generates, and the card's norm is that of a table built by
  another compiler. The first norm printed is a property of the host build as much as of the GPU. Not proven here:
  nothing but the card could say which build the 9.4e-8 came from.
- `graphMemoryFootprint` and `simpleAttributes` (too slow): the first launches a 16.7M-thread kernel sixteen times,
  each thread spinning on `clock()` for 8900 clocks (the simulator's `clock()` is its instruction count); it did not
  finish in 3000 s. The second launches 65535 x 1024 threads that each loop about 700 times on a shared counter.
  Neither is run (class `skipped`), where simpleAttributes used to burn the nightly's 600 s timeout every night.

Also found, not in the list: on the hosted toolchain image `convolutionFFT2D` matches the card but its SASS and PTX
engines leave different bytes after `spPostprocess2D_kernel` (launch 5; the digest in `external-smoke.yml`),
which has not been looked into.

### Tooling, CI and process

- Trace record/replay and a conformance database with compatibility scores: not started.
- A memory-aware adversarial scheduler (feeding the race detector's shadow state back into scheduling): not
  started ("Next milestones" 2).
- LLVM JIT (ARCHITECTURE.md D1): not started.
- Real-GPU card check (docs/ci-card-check.md): designed on a 3080 Ti, not enabled.
- AMD rows in `workloads.yml` and its `coverage` check: absent.
- The hip-tests disabled-case lists and the SASS-vs-PTX digest differences (see "CI inventory"): unreconciled.
- Pantheon workloads: 44 of 46 pass with `scripts/run-pantheon-workloads.sh`, which skips `media_enc_virus` and
  `rt_virus`; the CI runner (`tests/workloads/run_pantheon_workloads.sh`) runs `media_enc_virus` too and skips
  only `rt_virus` (docs/pantheon-workloads.md).
- GitHub Action constraints (docs/github-action.md:82-124): it must use the shared cudart, needs
  `library-path: false` when linking vendor libraries, Ubuntu 22.04's CUDA 11.5 is too old (so 12.6 is
  installed), and its timings mean nothing.
- External suites: `gpu-burn` and `cuda-memtest` merged as PR #290 (`ci/external/gpu-burn.txt`,
  `ci/external/cuda-memtest.txt`, `external-suites.yml`); the pantheon `memory_*`, `galpat`, `march_test` and
  `memory_hammer` workloads and, on AMD, `amd_memtest_patterns` cover the same ground.

