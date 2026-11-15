# TODO: what is still open

Short list of open work, grouped by area. Each line points to where the detail lives. Revision 6 (2026-11-09): the old 2,438-line file
was split; finished work is in [docs/history/TODO-done.md](docs/history/TODO-done.md), the refusals and gaps with their evidence are in
[docs/gap-register.md](docs/gap-register.md) (its entries are from 2026-10-22 and some are out of date: search the tree for the name before
working on one). What runs and what does not, for users: [docs/status.md](docs/status.md). Design: [ARCHITECTURE.md](ARCHITECTURE.md).

## Status at a glance

| Area | State | Where to look |
| --- | --- | --- |
| NVIDIA execution | PTX interpreter and a SASS executor (the default; sm_75 to sm_120a); both run unmodified nvcc programs through `libvgpucudart` / `libvgpucuda` | nvidia/docs/sass.md, "Coverage" |
| AMD execution | GCN/CDNA and RDNA executors (64- and 32-lane); unmodified hipcc programs from ROCm 6.4 to 7.2 on all 10 AMD profiles (MI250X, MI300X, MI325X, MI350X, MI455X, RX 6900 XT, RX 6800, RX 6700 XT, RX 7900 XTX, RX 9070 XT), 8 gfx targets ([docs/status.md](docs/status.md)) | amd/README.md |
| Device profiles | 38 NVIDIA (14 read from hardware) and 10 AMD (3 read from hardware) | [docs/history/TODO-done.md](docs/history/TODO-done.md), "Profile inventory" |
| NVIDIA libraries | cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE, cuSOLVER, NCCL, NVRTC, NPP, nvJPEG, cuDSS (0.8 and 1.0), cuSPARSELt, cuTENSOR, cuTensorNet, cuStateVec, cuFile, nvCOMP, NVSHMEM, nvJitLink, nvFatbin, NVENC | nvidia/docs/libraries.md |
| AMD libraries | rocBLAS, hipBLASLt, hipSPARSELt, MIOpen, rocPRIM/hipCUB, rocRAND, rocFFT, rocSPARSE, rocSOLVER, Composable Kernel, RCCL; ROCm SMI, AMD SMI, rocprofv3/rocprofiler-sdk shims | amd/README.md |
| Frameworks | PyTorch (CUDA on 12 whole-model checks; ROCm on six AMD GPUs), vLLM (ROCm), llama.cpp and Ollama (every NVIDIA and AMD profile), Triton, Numba, `torch.compile` | nvidia/docs/pytorch.md, [docs/history/TODO-done.md](docs/history/TODO-done.md), "CI inventory" |
| Health, diagnostics, RAS | NVML (incl. the reads DCGM makes), nvidia-smi, rocm-smi, amd-smi, RVS's sensors, `vgpu fault`, BAR0/MMIO register model, memory-pattern tests | docs/telemetry.md, docs/registers.md |
| Not possible by design | hardware performance counters that need a timing model (`%pm0`-`%pm7`, stall reasons, hit rates), timing itself | [docs/history/TODO-done.md](docs/history/TODO-done.md), "Registers and counters" |
| Blocked | static-cudart binaries (undocumented driver export tables), Nsight Systems collection, OptiX | [docs/gap-register.md](docs/gap-register.md), "Next milestones" 4, "Known out of scope" |
| Open work | the lists below and [docs/gap-register.md](docs/gap-register.md) | |

(Counts are from 2026-10-22; `vgpu list-gpus` and the inventories in [docs/history/TODO-done.md](docs/history/TODO-done.md) are the
current sources.)

## Open items

### Execution model

- Kernels run to completion inside their launch call, so a kernel that waits for the host after launching it never ends
  ([#523](https://github.com/pantheongpu/pantheonsim/issues/523); `UnifiedMemoryPerf` in [gap-register.md](docs/gap-register.md), "CUDA Samples
  known differences"). Fix is a worker per stream with events ordering them; large.
- `shfl.sync` returns 0 for a lane of the mask that is still in a long loop and reports nothing
  ([#488](https://github.com/pantheongpu/pantheonsim/issues/488)).
- Divergence is min-PC reconvergence, not full IPDOM; irreducible control flow is not handled and no test pins that
  ([gap-register.md](docs/gap-register.md), "Partially implemented").
- Memory-aware adversarial scheduler (feed the race detector's shadow state into scheduling): not started
  ([gap-register.md](docs/gap-register.md), "Next milestones" 2).
- Trace record/replay and a conformance database with compatibility scores: not started.
- LLVM JIT (ARCHITECTURE.md, D1): not started. (Tier 2 is a different thing: a C++ JIT, opt-in, see Performance below.)

### Blocked or out of scope

- Static-cudart binaries (undocumented `cuGetExportTable`): blocked, also blocks CuPy, `simpleCUFFT_callback` and cuQuantum Python
  ([nvidia/docs/dark-api.md](nvidia/docs/dark-api.md)).
- Nsight Systems and Nsight Compute as NVIDIA ships them; OptiX (`rt_virus`); counters that need a timing model
  (`%pm0`-`%pm7`, stall reasons, hit rates). See [docs/status.md](docs/status.md), [nvidia/docs/cupti.md](nvidia/docs/cupti.md).

### NVIDIA

- PTX and SASS forms refused by name: TMA `im2col::w`, interleaved tensor maps, `tcgen05` `.ashift` and some `.ws` shapes, `multimem`,
  `tex.grad` on cube textures, `cvt.rs` to x4 types; each with its reason in [gap-register.md](docs/gap-register.md), "PTX and SASS execution".
- Vendor libraries: the per-library list of unsupported modes (cuBLASLt FP8 aux scales, cuDNN graph corners, cuFFT callbacks, NCCL symmetric
  memory, nvCOMP Cascaded/Bitcomp/ANS, NVENC beyond H.264/HEVC, ...) is in [gap-register.md](docs/gap-register.md), "NVIDIA vendor libraries".
- Runtime and driver: external memory and semaphore import, cluster occupancy on parts without published GPC counts
  ([gap-register.md](docs/gap-register.md), "CUDA runtime and driver").
- NVML setters (persistence, power limit, clocks, MIG) and NVLink error injection: absent by design until persistence across processes has a
  design ([gap-register.md](docs/gap-register.md), "Found by the rev 5 gap analysis", "Health, RAS and registers").
- Profiles not verified on a card: `h200`, `b200`, `b300`, `rtx5090` and the other unverified rows of the profile inventory
  ([docs/status.md](docs/status.md), [nvidia/docs/profiles.md](nvidia/docs/profiles.md)).

### AMD

- Instruction forms refused by name (MFMA BLGP patterns and `xf32`, a wave with lanes off in some forms, cross-wave DPP, GWS, GDS, cube arrays,
  RDNA ray tracing) and HSA/HIP corners: [gap-register.md](docs/gap-register.md), "AMD" and "Found by the rev 5 gap analysis".
- AMD SMI and ROCm SMI values with no profile datum (violation status, PCIe throughput, board and firmware info, setters) stay `NOT_SUPPORTED`.
- Profilers: PC sampling is host-trap only; no stochastic sampling, thread trace or HSA trace ([amd/README.md](amd/README.md)).
- `fused_attention` does not build for AMD; `mma_virus` needs rocWMMA headers; `rt_virus` and `media_enc_virus` skip
  ([docs/pantheon-workloads.md](docs/pantheon-workloads.md)).
- Seven AMD profiles are not verified on a card; AMD ECC, memory-sensor and link values follow AMD's documentation, unchecked on an MI-series card.

### Health, RAS and registers

- Not modelled: NVLink far end and error counters, ECC mode changes, retired-page addresses (placeholders), the CPER debugfs ring, a lost
  AMD GPU's sysfs entries going away, partition switching, UMC channels beyond the assumed Aldebaran bases
  ([docs/telemetry.md](docs/telemetry.md), [docs/registers.md](docs/registers.md), [gap-register.md](docs/gap-register.md), "Health, RAS and registers").
- Capture data-centre and Instinct cards' registers with `tools/regprobe` to map more of BAR0 ([docs/registers.md](docs/registers.md)).

### CI and process

- Real-GPU card check (design on a 3080 Ti, not enabled): [docs/ci-card-check.md](docs/ci-card-check.md).
- AMD rows in `workloads.yml` and its `coverage` check; reconcile the hip-tests disabled-case lists and the SASS-vs-PTX digest differences
  ([docs/ci.md](docs/ci.md)).
- CUDA Samples that still differ from the card (`UnifiedMemoryPerf`, `simpleCUFFT_callback`, `quasirandomGenerator`, `graphMemoryFootprint`,
  `simpleAttributes`) are listed in `ci/external/cuda-samples-known.txt`, with the reasons in [gap-register.md](docs/gap-register.md).
- PyTorch sweep: one known numeric failure (tiny causal transformer after three AdamW steps), not a simulator bug
  (`nvidia/tests/pytorch/sweep/known_failures.txt`).
- GitHub Action constraints (shared cudart, Ubuntu 22.04 toolkit too old, timings meaningless): [docs/github-action.md](docs/github-action.md).

### Performance

What is measured, what the knobs do, and what was tried without gain: [docs/performance.md](docs/performance.md). Open or opt-in:

- Tier 2 (`VGPU_JIT=1`), launch-level result caching (`VGPU_MEMO=1`) and host-GPU mode (`VGPU_HOST_GPU`) are opt-in; making any a default
  needs its own measurement ([docs/perf-design/](docs/perf-design/README.md)).
- Release-wide PGO was measured to give no speed-up on held-out programs and stays an opt-in tool ([docs/pgo.md](docs/pgo.md)).
- Leads measured below the bar and not built (store-forward stalls, register-file clear, memory counting in the PTX and AMD engines, ...):
  [docs/performance.md](docs/performance.md), "What did not pay off".
- Tensor-core sums of the sm_86 model cost 3.5 to 4.5 times the time of those instructions (`VGPU_TC_EXACT`). (The model is no
  longer sm_86's alone: every profile whose `numerics.tensor_sum` is `ampere` pays this, and the `hopper` ones run in scalar code.)
