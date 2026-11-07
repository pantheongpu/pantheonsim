# PantheonSim

PantheonSim runs CUDA and HIP programs on a CPU, with no GPU. It supplies fake driver, runtime and library files and a profile for each GPU model (T4, H100, MI300X and 45 more), so a program built with the real `nvcc` or `hipcc` runs against it. It checks what a program computes. It does not model how fast a GPU would run it.

## Limits

- There is no timing. Timings mean nothing, and `%clock` and `%globaltimer` count instructions. It is also much slower than a GPU.
- CUDA programs must be built with `nvcc -cudart shared`. Static cudart (CuPy, for example) does not work.
- Kernels run to completion inside the launch call, and `cudaLaunchHostFunc` runs its function inline ([#487](https://github.com/pantheongpu/pantheonsim/issues/487)).
- Known bug: `shfl.sync` returns 0 for a lane that is still in a long loop ([#488](https://github.com/pantheongpu/pantheonsim/issues/488)).
- OptiX and NVIDIA's Nsight tools are not supported.
- 31 of the 48 GPU profiles come from public documents, not from a card.

See [status and limits](docs/status.md).

## Quickstart

With Docker (no GPU or driver needed):

```bash
docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus
docker run --rm -e VGPU_GPU=nvidia/h100 -e VGPU_DEVICE_COUNT=2 ghcr.io/pantheongpu/pantheonsim:latest nvidia-smi
```

The image has no `nvcc` or `hipcc`; see [docs/docker.md](docs/docker.md) to build a program in a container.

From source, on Linux (WSL works) with CMake and a C++20 compiler:

```bash
make     # builds (offers to install missing tools first)
./run    # starts a simulated GPU machine; type exit to leave
```

`./run` reads `pantheonsim.yaml` and writes one on the first run. `make doctor` lists missing tools. `make demo` runs a vectorAdd kernel on a simulated H100 and ends with PASS, `make run APP=./my_app` runs one program, and `make test` runs the tests. `make help` lists every target. To compile your own programs you need `nvcc` or `hipcc` installed ([docs/toolchains.md](docs/toolchains.md)).

## Settings

`build/vgpu list-gpus` prints the GPU ids. Put settings in a file and run `./run my.yaml`; `pantheonsim.example.yaml` is a sample. For example:

```yaml
gpu: nvidia/h100
count: 8
strict: true   # refuse undefined behaviour, such as dividing by zero
```

All settings:

| Key | Values | Default | What it does |
| --- | --- | --- | --- |
| `gpu` | an id from `list-gpus`, such as `nvidia/h100` | `nvidia/h100` | The GPU model to present |
| `count` | 1 to 16 | 1 | How many identical devices |
| `vram_mb` | MiB, at least 1 | the profile's own | Memory per device |
| `cuda_version` | `major.minor`, such as `13.0` | this build's | The CUDA version `nvidia-smi` and NVML report |
| `driver_version` | such as `580.65.06` | this build's | The NVIDIA driver version reported |
| `rocm_version` | such as `6.4.0` | `6.2.0` in the machine shell | The ROCm version `amd-smi version` reports |
| `os` | `ubuntu`, `rocky` or `debian`, with a version (`ubuntu:24.04`) | `ubuntu:22.04` | The operating system the machine presents |
| `ram_mb` | MiB, at least 1 | unlimited | Host RAM device memory may use before it spills to `memory_dir` |
| `memory_dir` | a directory | never spill | Where device memory past `ram_mb` goes (runs at disk speed) |
| `host_reserve_mb` | MiB, 0 or more | 6% of the limit, 1 to 4 GiB | Memory the guard leaves to the rest of your machine |
| `host_guard` | `true` or `false` | `true` | `false` turns the host-memory guard off |
| `threads` | 1 to 65536 | one per core (half under `make run` and `./run`) | Host threads that run blocks |
| `max_steps` | 0 or more | built in | Per-thread instruction budget against a kernel that never ends; `0` removes it |
| `sass` | `auto`, `off` or `only` | `auto` | The engine for NVIDIA kernels: SASS when the binary has it, or force PTX, or insist on SASS |
| `simd` | `auto`, `base`, `avx2` or `off` | `auto` | Caps the vector instruction set; `off` is scalar code |
| `scheduler` | `deterministic`, `random` or `adversarial` | `deterministic` | Warp scheduling order, to find order-dependent bugs (set `sass: off` with the last two) |
| `scheduler_seed` | 0 or more | 0 | Seed for `random` and `adversarial`; the same seed replays the same run |
| `strict` | `true` or `false` | `false` | Refuse undefined behaviour, such as integer division by zero |
| `race` | `off`, `on` or `all` | `off` | Report unordered shared-memory access between warps |
| `counters` | `true` or `false` | `false` | Print exact per-launch performance counters |
| `quiet` | `true` or `false` | `false` | Silence every simulator message on stderr |
| `trace` | `true` or `false` | `false` | Log every runtime and driver entry point |

Each setting is described in [docs/config.md](docs/config.md). It also runs [in a browser](https://pantheonsim.com/play/), [in CI](docs/github-action.md) and [in Docker](docs/docker.md).

## More

[Getting started](docs/getting-started.md), [the machine](docs/machine-simulator.md), [commands](docs/cli.md), [status and limits](docs/status.md), [architecture](ARCHITECTURE.md), [contributing](CONTRIBUTING.md), [governance](GOVERNANCE.md), [security](SECURITY.md), [code of conduct](CODE_OF_CONDUCT.md). Questions go in [Discussions](https://github.com/pantheongpu/pantheonsim/discussions), bugs in [issues](https://github.com/pantheongpu/pantheonsim/issues).

Apache-2.0. CUDA is a trademark of NVIDIA. This project is not affiliated with NVIDIA or AMD.
