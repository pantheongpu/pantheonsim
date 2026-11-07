# Status and limitations

This is what runs today and what does not. [TODO.md](../TODO.md) has the long list of gaps. It is older than some of the code, so check an entry against the tree before relying on it.

## What does not work

- **No timing.** It models what a GPU computes, not how long it takes. `%clock`, `%clock64` and `%globaltimer` count the block's instructions. Counters that need timing (stall reasons, cache hit rates) are not provided. It is an interpreter on CPU cores and is much slower than a GPU.
- **Launches are synchronous.** A kernel runs to completion inside the launch call. `cudaLaunchHostFunc` runs its function inline, so event and stream queries report complete at once, and a host function that waits on the caller hangs ([#487](https://github.com/pantheongpu/pantheonsim/issues/487)).
- **`shfl.sync` bug.** It returns 0 for a lane of the mask that is still in a long loop, and reports nothing ([#488](https://github.com/pantheongpu/pantheonsim/issues/488)). Other unsupported instructions and API calls fail with an error that names them.
- **Shared cudart only.** Build CUDA programs with `nvcc -cudart shared`. Hosting a static cudart needs NVIDIA's undocumented driver export tables ([TODO.md](../TODO.md), "Next milestones"). CuPy links it statically and does not run ([nvidia/docs/jit.md](../nvidia/docs/jit.md)).
- **Not supported:** OptiX; NVIDIA's Nsight Systems and Nsight Compute (`vgpu ncu` is our own `ncu`, and `nsys` collects no CUDA data); NVDEC for MPEG-1, MPEG-4, VC-1, VP8, VP9 and AV1.
- **Smaller gaps.** The `wmma` fragment layout is our own, because PTX leaves it open ([ARCHITECTURE.md](../ARCHITECTURE.md), D8). Branch reconvergence is min-PC, so irreducible control flow is not handled.

## GPU profiles

There are 48 (`vgpu list-gpus`): 38 NVIDIA and 10 AMD. Fourteen NVIDIA profiles and three AMD profiles were read from a physical card (`verified: true`). The other 31 come from public documents and nothing in them was measured. Rubin (`vr200`) and Thor carry placeholder PCI ids because NVIDIA has published none. Each profile's header says where its values came from.

- Verified NVIDIA: T4, A10, A10G, A100 (`a100` and `a100-sxm4-40gb`), H100, H100 PCIe, L4, L40S, RTX 3060, RTX 3080 Ti, GH200, RTX PRO 6000 Server Edition, RTX PRO 4500.
- Verified AMD: MI325X, RX 6800, RX 6700 XT.
- Not verified: the rest, including H200, B200, B300, GB200, RTX 4090, RTX 5090, MI250X, MI300X, MI350X, MI455X, RX 6900 XT, RX 7900 XTX and RX 9070 XT.

Details: [nvidia/docs/profiles.md](../nvidia/docs/profiles.md), [amd/README.md](../amd/README.md).

## NVIDIA and CUDA

Programs built by `nvcc` from CUDA 12 or 13 run. Kernels execute from SASS when the binary has it (sm_75 to sm_120a) and from PTX otherwise; `VGPU_SASS=0` forces PTX ([sass.md](../nvidia/docs/sass.md)). The PTX subset is large, with tensor-core, TMA and Hopper `wgmma` instructions. It is checked against CuTe and CUTLASS unit tests.

- Driver and runtime APIs, under the real sonames: contexts, memory, modules, streams, events, graphs, memory pools, `cuMem*`, multi-GPU. Every function in `cuda.h` and `cuda_runtime_api.h` is exported; unimplemented ones return "not supported". See [driver-api.md](driver-api.md) and [devices.md](../nvidia/docs/devices.md).
- Memory is sparse, so a simulated H200 reports 141 GB on a smaller host. Out-of-bounds, use-after-free, double-free and misaligned accesses are reported ([config.md](config.md)).
- Libraries: cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE, cuSPARSELt, cuSOLVER, cuTENSOR, cuTensorNet, cuStateVec, cuDSS, NCCL, NVRTC, NPP, nvJPEG, nvJitLink, nvFatbin, cuFile, nvCOMP, NVSHMEM, NVENC and NVDEC. Where a physical GPU was available they were compared with NVIDIA's own; see [libraries.md](../nvidia/docs/libraries.md).
- PyTorch's CUDA 13 build runs. Its 12 whole-model checks were run on seven profiles and compared with the CPU. Numba and Triton run ([pytorch.md](../nvidia/docs/pytorch.md), [jit.md](../nvidia/docs/jit.md)).
- NVML, `nvidia-smi`, `lspci`, a PCI register model, and CUPTI's Activity and Callback APIs ([telemetry.md](telemetry.md), [registers.md](registers.md), [cupti.md](../nvidia/docs/cupti.md)).

## AMD and ROCm

`hipcc` programs from ROCm 6.4, 7.0, 7.1 and 7.2 run on the simulator's own HIP and HSA runtimes. Supported targets are gfx90a, gfx942, gfx950, gfx1250, gfx1030, gfx1031, gfx1100 and gfx1201. rocBLAS, hipBLASLt, MIOpen, RCCL, PyTorch for ROCm, `rocprofv3`, `rocm-smi` and `amd-smi` run. rocBLAS passes 162,807 float and double tests of its own quick suite. See [amd/README.md](../amd/README.md) and [rocprofiler.md](../amd/docs/rocprofiler.md).

## Open work

The main items are static cudart, asynchronous streams and host functions (#487), the `shfl.sync` wait (#488), and reading more profiles from physical cards. The full list is in [TODO.md](../TODO.md).
