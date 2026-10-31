# Compilers for your own programs

PantheonSim runs programs that are already built, and does not compile them.
To build your own CUDA or HIP program you need the vendor's compiler on the same
computer. `make` does not install either: they are large and come under their own
licences.

| You want to build | You need | Size, roughly | Where |
| --- | --- | --- | --- |
| CUDA programs (`.cu`) | the CUDA toolkit, for `nvcc` | 3 to 7 GB | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), or `sudo apt-get install nvidia-cuda-toolkit` on Ubuntu (older CUDA) |
| HIP programs for AMD GPUs | ROCm, for `hipcc` | 5 to 20 GB | [rocm.docs.amd.com](https://rocm.docs.amd.com), the AMD repository for your distribution |
| Python, PyTorch, prebuilt binaries | nothing | | `pip install torch`, then `make run APP=python ARGS="train.py" PRELOAD=1` |

You do not need a GPU or a GPU driver for either toolkit. Install only the
toolkit, not the driver packages (on NVIDIA's repository that is the `cuda-toolkit-*`
package, not `cuda` or `cuda-drivers`; on AMD's, the user-space ROCm packages without
the `amdgpu-dkms` kernel driver).

## Using them

```bash
nvcc -cudart shared my_app.cu -o my_app     # -cudart shared lets the simulator stand in for the runtime
hipcc --offload-arch=gfx942 my_app.hip -o my_app   # gfx942 is the MI300X; build for the GPU you simulate
```

Inside `./run` they are the same programs. `build/vgpu info --gpu amd/mi300x` shows
the `gfx` name of a simulated AMD GPU. `make doctor` reports which compiler it finds.

Without a compiler, `./run` still works: `nvidia-smi`, Python, PyTorch and any program
built elsewhere run; `nvcc` and `hipcc` are simply not there to call. Programs built with CUDA 12 or 13
run on a simulator built for the same CUDA major (the default build is 13; the
[container images](docker.md) come in both), and ROCm 6.4 and 7.x programs work
([status.md](status.md)).
