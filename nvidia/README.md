# NVIDIA

Everything in PantheonSim that is specific to NVIDIA GPUs and CUDA. The engine
it runs on -- the PTX interpreter, device memory, the scheduler, the `vgpu`
command line -- is vendor-neutral and lives at the top of the repository.

| Folder | What |
| --- | --- |
| `src/` | The CUDA driver and runtime, NVML, CUPTI, NVENC and the vendor libraries (cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE, cuSOLVER, NCCL, NVRTC, NPP, nvJPEG), each built under its real soname |
| `include/` | `vgpu_cuda.h`, the driver ABI a C program links against |
| `profiles/` | Device profiles, one per GPU. Verified profiles were measured on a physical card; each file's header says where its values came from |
| `third_party/` | Public cuDNN and NCCL headers the libraries are built against |
| `tools/` | `nvidia-smi` and `nvcc` for simulated machines, and the scripts that characterize and verify a profile on real hardware |
| `tests/` | End-to-end tests that compile and run CUDA programs, conformance tests against NVIDIA's own libraries, and the C driver harness |
| `docs/` | cuBLAS, CUPTI, the driver export tables, Python JIT frameworks, the libraries, textures |

A profile id is `nvidia/<name>`, and its file is `profiles/<name>.yaml` here.
