# Status and limitations

What PantheonSim does today, and what it deliberately does not. The [README](../README.md) has the short version.

The engine is vendor-neutral; NVIDIA and CUDA support lives in [`nvidia/`](../nvidia/README.md) and
AMD and ROCm in [`amd/`](../amd/README.md). [ARCHITECTURE.md](../ARCHITECTURE.md) has the map.

Working today, all CPU-only:

| Area | State |
| --- | --- |
| Device profiles | Twelve NVIDIA profiles verified against physical cards across six architectures — Turing, Ampere sm_80/sm_86, Ada, Hopper, Grace-Hopper — each matching its device on 512 conformance values. `h200` and `b200` are not yet characterized. Fifteen more NVIDIA profiles (H200, B200, B300, GB200, Rubin, Jetson Thor, RTX 5090, RTX PRO 6000, GB10 in the DGX Spark, RTX 4090, L40, RTX 3090, A40, A30, RTX 2080 Ti) are built from public documents only and marked `verified: false`: nothing in them was measured, and Rubin and Thor carry placeholder PCI ids because NVIDIA has published none. [nvidia/docs/profiles.md](../nvidia/docs/profiles.md) lists them. AMD: MI325X is read from a physical card; MI300X and MI350X are placeholders. Each profile's header says where its values came from |
| `vgpu` CLI | `list-gpus`, `info --gpu <id> [--json]`, `demo vectoradd` |
| Virtual VRAM | sparse/lazy backing — a virtual H200 claims 141 GB on a 16 GB host, past `VGPU_MEMORY_RAM_MB` it spills to disk, and a host-memory guard turns "the host is out of RAM" into a device out-of-memory error instead of an OOM kill; OOB / use-after-free / double-free / misalignment diagnostics |
| PTX | lexer/parser for a growing subset (see [ARCHITECTURE.md](../ARCHITECTURE.md)); precise `unsupported` errors for the rest |
| Execution | SIMT warp interpreter: 32-lane warps, divergence masks, shared memory, `bar.sync`, warp shuffles/vote, atomics, tensor-core `wmma`, f16/f16x2, deterministic scheduling |
| Driver API | `libvgpucuda.so` + clean-room `vgpu_cuda.h`: init/discovery/context/memory/module/`cuLaunchKernel`, `cuLibrary`/`cuKernel`, `cuGetProcAddress` |
| Runtime API | `libvgpucudart` (drop-in `libcudart.so.13`): the CUDA **Runtime** API + nvcc host-registration ABI, so unmodified nvcc apps run unchanged |
| Fatbin | extracts embedded PTX from nvcc fatbins (uncompressed, zstd and LZ4 — so binaries from CUDA 12 and 13 both work) |
| Virtual memory | the `cuMem*` mapping API: address space reserved without memory, physical handles with no address, mappings that are unusable until access is granted — so a buffer grows in place, which is what PyTorch's expandable segments and NCCL's windows need. Reserved-but-unmapped, no-access and read-only faults each say which they were |
| Multi-GPU | a virtual rack of N devices with disjoint address windows; peer copies and per-device isolation match a real two-GPU machine; `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` read as the driver reads them, and every device attribute and `cudaDeviceProp` member (L2, bus width, texture limits, PCI identity, UUID matching NVML's) answers from one table checked against an RTX 3060 — see [nvidia/docs/devices.md](../nvidia/docs/devices.md) |
| Vendor libraries | cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE, cuSPARSELt, cuSOLVER, cuTENSOR, cuTensorNet, NCCL, NVRTC, NPP and nvJPEG under their real sonames, each differential-tested against NVIDIA's own library on a physical GPU; also cuStateVec, cuDSS, nvJitLink, nvFatbin, cuFile (GPUDirect Storage's compatibility mode), nvCOMP (its standard formats and GDeflate, interoperable with NVIDIA's) and NVSHMEM (one simulated GPU per PE process, NVIDIA's device API on the simulator's host library) — see [nvidia/docs/libraries.md](../nvidia/docs/libraries.md) |
| PyTorch | PyTorch's official CUDA 13 build runs unmodified on the simulated NVIDIA GPUs, T4 to B200 and RTX 5090: GPT, ResNet, LSTM, U-Net, ViT, MoE and DLRM training all match the CPU — see [nvidia/docs/pytorch.md](../nvidia/docs/pytorch.md). PyTorch for ROCm runs on the simulated AMD GPUs — see [amd/README.md](../amd/README.md) |
| Python JIT | Numba runs unmodified (its PTX is assembled through the driver's JIT link API); Triton runs unmodified under `vgpu run` and `vgpu shell` (`vgpu-ptxas` stands in for `ptxas`) — see [nvidia/docs/jit.md](../nvidia/docs/jit.md) |
| Discovery | live telemetry + NVML; drop-in `nvidia-smi`, `rocm-smi`, `amd-smi`, `rocm_agent_enumerator`, and `lspci` output — see [docs/telemetry.md](telemetry.md) |
| Registers | PCI configuration space from a register database: header, PM, MSI, PCI Express and AER, live link and error state, BAR sizing; `vgpu regs list/read/write/dump/log` with an access log; AMD MMIO (engine status, the SMU mailbox) from the amdgpu headers; a C API, `vgpu_regs.h` — see [docs/registers.md](registers.md) |
| AMD | unmodified hipcc-built programs, from ROCm 6.4, 7.0, 7.1 or 7.2, run on a simulated MI300X: the HIP runtime ABI, CDNA3 (gfx942) code checked instruction by instruction against ROCm's llvm-objdump and executed on every host core, device-side `printf`, and every pantheon workload with `--verify`. AMD's `rocprofv3` runs unmodified, with the counters the interpreter counts exactly. So does rocBLAS: AMD's own quick tests pass (162,807 float and double ones, and the half, bfloat16, int8 and FP8 GEMMs run) — see [amd/README.md](../amd/README.md) |
| Profiling | CUPTI under its real soname: the Activity API (kernels, copies, fills, waits, streams, devices, runtime API calls, memory and pool records, graph traces, events and overhead, correlated) and the Callback API (every runtime function, resource including modules and graphs, and synchronize domains); `torch.profiler` collects a GPU trace, checked against the trace NVIDIA's CUPTI prints on an RTX 3060; unmodified nvprof traces a program on the T4 profile. `vgpu ncu` is this project's own `ncu`; Nsight Systems and Nsight Compute as NVIDIA ships them are not supported (`nsys` collects no CUDA data). See [nvidia/docs/cupti.md](../nvidia/docs/cupti.md) |
| NVENC / NVDEC | `libnvidia-encode.so.1` answering the API as an RTX 3060 does, with compressing H.264 (CABAC or CAVLC, 8x8 transform, P and B pictures, slices) and HEVC (intra, P and B pictures) encoders and rate control, a lossless PCM stream for the lossless tuning, all of which ffmpeg decodes (and, being deterministic, video-encode SDC tests run), and `libnvcuvid.so.1` decoding Motion JPEG, H.264 (Baseline, Main and High, progressive and interlaced), HEVC (Main, Main 10 and the 4:2:0 range extension tools) and MPEG-2 (Main profile, frame and field pictures) into the card's surfaces, bit-exact with the card except MPEG-2's inverse DCT, which is within a level — see [nvidia/docs/libraries.md](../nvidia/docs/libraries.md) |
| Proof | an nvcc-compiled CUDA program **and** the unmodified pantheon stress kernels run on the CPU; `memory_read` differential-matches a physical RTX 3060 (incl. fault injection + device printf) |

Known limitations (deliberate, documented):

- Kernels run from **SASS** when the binary carries it (the default engine on
  every generation from sm_75 to sm_120a, see [nvidia/docs/sass.md](../nvidia/docs/sass.md))
  and from **PTX** otherwise (embedded, or in a fatbin; zstd fatbins are
  decompressed). `VGPU_SASS=0` forces the PTX path.
- Unmodified apps must link **shared** cudart (`nvcc -cudart shared`) so the
  loader can substitute VirtualGPU's `libcudart.so.13`. The source is untouched;
  hosting a *statically* linked cudart needs NVIDIA's undocumented driver export
  tables and is future work. This is what stops CuPy, which links cudart
  statically — see [nvidia/docs/jit.md](../nvidia/docs/jit.md).
- `wmma` fragment layout is VirtualGPU's own (PTX leaves it unspecified) —
  see [ARCHITECTURE.md](../ARCHITECTURE.md) D8. bf16, `cp.async`, `mma.sync`, `ldmatrix`, the
  extended-precision carry family, textures, surfaces, grid sync, named
  barriers, Hopper's warpgroup MMA (`wgmma`), TMA (multicast, reductions and
  im2col included), thread-block cluster barriers and distributed shared
  memory are implemented -- checked against CuTe and CUTLASS's own Hopper
  unit tests, its convolutions among them. Every gap fails loudly
  (instruction, PTX line, kernel, profile), never silently.
- NVENC (video encode) is served by `libvgpunvenc`, presented as
  `libnvidia-encode.so.1`: the API as an RTX 3060 answers it, with compressing
  H.264 (CABAC or CAVLC, 8x8 transform, P and B pictures, slices) and HEVC (intra,
  P and B pictures) encoders and rate control, and the lossless tuning written as
  lossless PCM pictures; any decoder reads them (ffmpeg's is the test) and encoder
  stress and corruption checks run. NVDEC
  (`libvgpunvcuvid`, presented as `libnvcuvid.so.1`) decodes Motion JPEG and
  H.264 (Baseline, Main and High profile, progressive frames, bit-exact with the
  card's NVDEC) and reports every other codec
  unsupported. OptiX (ray tracing) is a separate
  NVIDIA subsystem, not CUDA, and is out of scope.
- AMD runs gfx942 (MI300X, MI325X), gfx950 (MI350X), gfx90a (MI250X),
  gfx1030 (Radeon RX 6900 XT and RX 6800, RDNA2), gfx1031 (Radeon RX 6700 XT), gfx1100 (Radeon RX 7900 XTX, RDNA3) and
  gfx1201 (Radeon RX 9070 XT, RDNA4) code, through its own HIP and HSA runtimes; rocBLAS, hipBLASLt, MIOpen and
  PyTorch run on it — see [amd/README.md](../amd/README.md).

## Roadmap (abridged)

The full list is in [TODO.md](../TODO.md).

Done: shared memory, warp shuffles and the PTX and SASS coverage listed in
TODO.md; `vgpu run` and `vgpu test --matrix`; random and adversarial warp
scheduling with race detection; the AMD frontend (HIP/ROCm, CDNA2 to CDNA4 and
RDNA2 to RDNA4); hardware characterization for 12 NVIDIA profiles and one AMD
profile. Still open:

1. Static-cudart hosting (driver export tables) → no `-cudart shared` rebuild
   (investigated and blocked, see [TODO.md](../TODO.md))
2. Differential fuzzing against physical GPUs (oracle machines) → the rest of
   the verified profiles, a conformance database, compat scores
3. The gap register at the end of TODO.md
