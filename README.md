# PantheonSim

**Run CUDA and HIP programs, unmodified, on a CPU. No GPU needed.** It pretends to be an NVIDIA or AMD GPU (a T4, an H100, an MI300X), so programs built with the real `nvcc` or `hipcc` run on it unchanged. It checks what your code does, not how fast it runs: there is no timing model, so a run on a physical GPU before a release still counts.

[![CI](https://github.com/pantheongpu/pantheonsim/actions/workflows/ci.yml/badge.svg)](https://github.com/pantheongpu/pantheonsim/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/pantheongpu/pantheonsim)](https://github.com/pantheongpu/pantheonsim/releases)
[![License](https://img.shields.io/github/license/pantheongpu/pantheonsim)](LICENSE)

[![The PantheonSim playground: nvcc and nvidia-smi on eight simulated H100s](docs/img/playground.png)](https://pantheonsim.com/play/)

## Try it in 60 seconds

With Docker (no GPU, no driver, no `--gpus` flag):

```bash
docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus
docker run --rm -e VGPU_GPU=nvidia/h100 -e VGPU_DEVICE_COUNT=2 ghcr.io/pantheongpu/pantheonsim:latest nvidia-smi
```

The second command prints `nvidia-smi` output for two simulated H100s. The image has no compiler (`nvcc` and `hipcc` are not in it); see [docs/docker.md](docs/docker.md) to build a program inside a container, or use the source build below.

**What it is:** a CPU-side simulator of NVIDIA and AMD GPUs, so programs compiled with the real `nvcc` or `hipcc` run on a machine without a GPU. **What it isn't:** a performance model (timings mean nothing), and not every CUDA or HIP feature is covered; [Status and limits](docs/status.md) lists what is.

## Run it from source

```bash
make     # installs cmake and a C++ compiler if missing (asks first), then builds
./run    # starts a simulated GPU machine and takes you inside; type exit to leave
```

`./run` reads `pantheonsim.yaml` and writes one on the first run. To start from a ready-made file instead, use the sample: `./run pantheonsim.example.yaml` runs it as it is (one simulated H100), and `cp pantheonsim.example.yaml pantheonsim.yaml` makes it your default to edit. It lists every setting, and "Choose your machine" below explains them.

You need Linux (WSL works), a C++20 compiler and CMake, and no GPU. Run both commands from the repository root: `make: *** No targets specified` or `./run: No such file or directory` means you are in another directory or on an old branch, so `git checkout main && git pull` first. `make doctor` checks the tools and says what to install, `make deps` installs them, and `make build` builds without installing anything. The build takes a few minutes and uses half your CPUs (`make JOBS=8` to choose). The simulator also runs on half your CPUs; set `threads: N` in the settings file, or `make run THREADS=N`, to change that. `./run` before `make` builds for you.

Other targets:

```bash
make demo                           # run a built-in vectorAdd kernel on a simulated H100; ends with PASS
make run APP=./my_app ARGS="..."    # run one program on the simulated GPU, without the machine shell
make test                           # build and run the test suite
make help                           # every target and setting
```

Inside, `nvidia-smi` shows the GPUs, and you can compile, download programs with `curl` or `git clone`, and run them; for Python packages use a virtual environment (`python3 -m venv ~/v && ~/v/bin/pip install ...`); system packages (apt) are installed on your own machine, outside.

## Choose your machine

Which GPUs: `make list-gpus` (or `build/vgpu list-gpus`) prints every id you can use, 48 NVIDIA and AMD cards from a T4 to a B300 and from an RX 6700 XT to an MI455X. Use the id as printed: `gpu: nvidia/t4` in the settings file, or `make run GPU=nvidia/t4`. `build/vgpu info --gpu nvidia/t4` shows one card. Some profiles were measured on a physical card and some come from public documents only; [Status and limits](docs/status.md) says which.

Put the settings in a file, for example `my.yaml`, then `./run my.yaml`. [pantheonsim.example.yaml](pantheonsim.example.yaml) lists every setting and runs as it is (`./run pantheonsim.example.yaml`); copy it to `pantheonsim.yaml` to make it the default, which is also what `./run` writes for you on a first run:

```yaml
gpu: nvidia/h100     # which GPU (build/vgpu list-gpus shows them all)
count: 8             # how many
strict: true         # refuse undefined behaviour, such as dividing by zero
```

Every setting the file accepts (the full text for each is in [Settings](docs/config.md), and `build/vgpu config keys` prints it from the program):

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

In a browser: [pantheonsim.com/play](https://pantheonsim.com/play/). In CI: `pantheongpu/setup-pantheonsim@v0` ([docs/github-action.md](docs/github-action.md)), or `scripts/ci-setup.sh` on GitLab CI, CircleCI, Forgejo, Jenkins and Buildkite ([docs/ci-other-platforms.md](docs/ci-other-platforms.md)). In a container: `docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus` ([docs/docker.md](docs/docker.md)).

## Docs

[Getting started](docs/getting-started.md) (step by step, your own programs, first-run fixes) | [Settings](docs/config.md) | [The machine](docs/machine-simulator.md) | [Compilers](docs/toolchains.md) | [Commands](docs/cli.md) | [Status and limits](docs/status.md) | [Architecture](ARCHITECTURE.md) | [Governance](GOVERNANCE.md) | [Security](SECURITY.md)

Questions go in [Discussions](https://github.com/pantheongpu/pantheonsim/discussions), bugs in [issues](https://github.com/pantheongpu/pantheonsim/issues). Apache-2.0 (see LICENSE); CUDA is a trademark of NVIDIA, and this project is not affiliated with NVIDIA or AMD.
