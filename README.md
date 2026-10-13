# PantheonSim

**Run CUDA and HIP programs, unmodified, on a CPU. No GPU needed.**

PantheonSim simulates NVIDIA and AMD GPUs (a T4, an H100, a B200, an MI300X) on
an ordinary CPU. Programs built with the real `nvcc` or `hipcc` run against it
unchanged, and `nvidia-smi`, `rocm-smi` and the vendor libraries report the GPUs
you asked for. Learn GPU programming without a GPU, and test GPU code in CI on
standard runners.

[![The PantheonSim playground: nvcc and nvidia-smi on eight simulated H100s](docs/img/playground.png)](https://pantheonsim.com/play/)

PantheonSim checks what your code does, not how fast it runs: there is no
performance model, and a run on physical GPUs before a release is still the
final word. The intended CI model is simulated GPUs on every commit, physical
ones nightly or before a release. The simulator engine and its command line are
called VirtualGPU (`vgpu`).

## Quick start

Pick one.

- **In your browser.** [pantheonsim.com/play](https://pantheonsim.com/play/) gives
  you a Linux machine with simulated GPUs, an editor, a terminal, and `nvcc` or
  `hipcc`. No sign-up.
- **In GitHub Actions.** One step, on the standard runners:

  ```yaml
  - uses: pantheongpu/setup-pantheonsim@v0
    with:
      gpu: nvidia/h100        # or amd/mi300x
  - run: cmake -S . -B build && cmake --build build && ctest --test-dir build
  ```

  More at [pantheonsim.com/ci](https://pantheonsim.com/ci/) and in
  [docs/github-action.md](docs/github-action.md).
- **On your own machine.** Follow [Build & test](#build--test) and
  [Run your first program](#run-your-first-program) below, or
  [pull the container image](docs/docker.md).

## Build & test

You need Linux, CMake 3.20 or newer, and a C++20 compiler (gcc 13+ or
clang 17+). No GPU, no CUDA toolkit and no third-party libraries.

```bash
./scripts/build.sh              # builds build/vgpu
./scripts/test.sh               # optional: runs the test suite
build/vgpu list-gpus            # check it works: lists the GPUs it can simulate
build/vgpu demo vectoradd       # runs a built-in kernel end to end
```

## Run your first program

Build your program with the real compiler, then run it with `vgpu run`:

```bash
nvcc -cudart shared my_app.cu -o my_app        # NVIDIA: link the shared CUDA runtime
build/vgpu run --gpu nvidia/h100 ./my_app

hipcc --offload-arch=gfx942 my_app.hip -o my_app   # AMD: build for the GPU you will simulate
build/vgpu run --gpu amd/mi300x ./my_app
```

`vgpu run` puts the simulator's libraries in front of the real ones and runs the
program in place, so its output, exit code and signals are the program's own.
The `-cudart shared` flag matters: it lets the loader swap in the simulator's
CUDA runtime, and a program that links the runtime statically cannot be
simulated (see [Status and limitations](docs/status.md)). Python programs work
too, PyTorch included: `build/vgpu run --preload python train.py` (`--preload` is
what lets the simulator reach libraries that an interpreter loads later).

Choose the GPU with `--gpu <vendor>/<model>`, such as `nvidia/t4`, `nvidia/h100`
or `amd/mi300x`. `vgpu list-gpus` lists them all and `vgpu info --gpu nvidia/h100`
shows one. Add `--count 8` for eight of them.

To explore a whole simulated machine instead, `build/vgpu shell` asks which GPU,
CUDA or ROCm version and OS you want, then gives you a shell where `nvidia-smi`,
`rocm-smi`, `lspci` and `dmesg` all describe it. See
[docs/machine-simulator.md](docs/machine-simulator.md).

## Options

The ones you will use most (the full list is in [docs/cli.md](docs/cli.md)):

| Option | What it does |
| --- | --- |
| `--gpu <vendor/model>` | Which GPU to simulate. Default `nvidia/h100`. |
| `--count <n>` | How many of them. |
| `--vram-mb <n>` | Memory per GPU, in MiB, instead of the model's. |
| `--threads <n>` | Host threads that run blocks. Default: one per core. |
| `--race` | Report shared-memory data races between warps. |
| `--strict` | Refuse undefined behaviour, such as dividing by zero. |
| `--counters` | Print exact per-kernel instruction and memory counters. |
| `--quiet` | Silence the simulator's messages on stderr. |

Each of these sets an environment variable that you can also set yourself, which
is how you configure a program that does not go through `vgpu run`:
`VGPU_GPU`, `VGPU_DEVICE_COUNT`, `VGPU_VRAM_MB`, `VGPU_THREADS`, `VGPU_RACE`,
`VGPU_STRICT`, `VGPU_COUNTERS` and `VGPU_QUIET`. The variables for memory spill,
the SASS or PTX engine and tracing are in
[docs/cli.md](docs/cli.md#environment-variables).

## Commands

| Command | What it does |
| --- | --- |
| `vgpu list-gpus`, `vgpu info` | List the GPU models and show one. |
| `vgpu run` | Run a program on a simulated GPU. |
| `vgpu shell` | An interactive simulated machine. |
| `vgpu test --matrix` | Run a program on every GPU model and compare the results. |
| `vgpu ncu`, `vgpu counters` | Exact performance counters, in Nsight Compute's command-line form. |
| `vgpu debug` | An AMD kernel debugger: breakpoints, stepping, registers. |
| `vgpu fault` | Inject ECC and PCIe errors, hangs or a lost GPU, to test health tools. |
| `vgpu regs` | Read, write and dump a device's registers. |
| `vgpu smi`, `vgpu serve` | Live telemetry for monitoring tools. |

Most take `--help`; `vgpu shell` and `vgpu serve` do not. Every option is in
[docs/cli.md](docs/cli.md).

## Finding bugs

Errors return the documented CUDA codes and say exactly what went wrong:

```
[vgpu] cuLaunchKernel: VirtualGPU error [out-of-bounds]: device memory read at
0x200000003140 is 0 bytes past the end of the 64-byte allocation at 0x200000003100
  in kernel 'vecAdd', PTX line 30
  lane 16
  instruction: ld.global.f32 %f1,[%rd8]
  GPU profile: nvidia/h100
```

`--race` finds unordered shared-memory access that hardware usually hides, and
`--strict` catches undefined behaviour real compiler output trips over. Both
are off by default because each also flags code that is correct, and each costs
time. See [docs/diagnostics.md](docs/diagnostics.md).

## What works, and what does not

- **NVIDIA:** twelve GPU models verified against physical cards, from Turing to
  Hopper, plus fifteen built from public documents (including the B200). CUDA
  runtime and driver APIs, the main vendor libraries (cuBLAS, cuDNN, cuFFT,
  NCCL and more), PyTorch, Numba and Triton run unmodified.
- **AMD:** unmodified `hipcc` programs on MI300X and the other CDNA and RDNA
  models, rocBLAS and PyTorch for ROCm.
- **Not simulated:** timing. There is no cycle, stall or cache model, so no
  performance numbers. Statically linked CUDA runtimes (CuPy) and OptiX are not
  supported.
- Anything unsupported fails loudly, naming the instruction, kernel and GPU,
  never silently.

The full status table and the known limitations are in
[docs/status.md](docs/status.md).

## More documentation

- [docs/cli.md](docs/cli.md): every command, option and environment variable.
- [docs/github-action.md](docs/github-action.md) and [docs/docker.md](docs/docker.md): CI and containers.
- [docs/machine-simulator.md](docs/machine-simulator.md): `vgpu shell`.
- [docs/diagnostics.md](docs/diagnostics.md): counters, races, strict mode.
- [docs/driver-api.md](docs/driver-api.md): a program written against the CUDA driver API.
- [docs/telemetry.md](docs/telemetry.md) and [docs/registers.md](docs/registers.md): `nvidia-smi`, `rocm-smi`, PCI and registers.
- [docs/pantheon-workloads.md](docs/pantheon-workloads.md): the 44 stress and diagnostics workloads.
- [nvidia/README.md](nvidia/README.md) and [amd/README.md](amd/README.md): the NVIDIA and AMD sides.
- [ARCHITECTURE.md](ARCHITECTURE.md): how it is built. [TODO.md](TODO.md): what is open.

## Help, contributing and security

Questions and ideas go in [Discussions](https://github.com/pantheongpu/pantheonsim/discussions),
bugs in [issues](https://github.com/pantheongpu/pantheonsim/issues). How changes
are decided is in [GOVERNANCE.md](GOVERNANCE.md), and security problems should be
reported privately as [SECURITY.md](SECURITY.md) describes.

## License

Apache-2.0 (see LICENSE). The `vgpu_cuda.h` header is a clean-room subset
written from NVIDIA's public driver API documentation; it contains no NVIDIA
code. The AMD register offsets, fields and SMU message numbers in
`amd/registers/mmio.yaml` come from the Linux kernel's amdgpu headers, under
AMD's MIT license (`amd/registers/LICENSES/amdgpu-headers.txt`), and the NVIDIA
register offsets and fields in `nvidia/registers/mmio.yaml` from the published
register headers of NVIDIA's open GPU kernel modules, under their MIT license
(`nvidia/registers/LICENSES/nvidia-open-gpu-kernel-modules.txt`). Both are
register maps only: no vendor code is in this project. CUDA is a
trademark of NVIDIA Corporation; this project is not affiliated with or
endorsed by NVIDIA or AMD.
