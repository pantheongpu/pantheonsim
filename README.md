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

Inside, `nvidia-smi` shows the GPUs, and you can compile, download programs with `curl` or `git clone`, and run them; for Python packages use a virtual environment (`python3 -m venv ~/v && ~/v/bin/pip install ...`); system packages (apt) are installed on your own machine, outside.

## Choose your machine

Put the settings in a file (`./run` writes a commented one), for example `my.yaml`, then `./run my.yaml`:

```yaml
gpu: nvidia/h100     # which GPU (build/vgpu list-gpus shows them all)
count: 8             # how many
strict: true         # refuse undefined behaviour, such as dividing by zero
```

In a browser: [pantheonsim.com/play](https://pantheonsim.com/play/). In CI: `pantheongpu/setup-pantheonsim@v0` ([docs/github-action.md](docs/github-action.md)), or `scripts/ci-setup.sh` on GitLab CI, CircleCI, Forgejo, Jenkins and Buildkite ([docs/ci-other-platforms.md](docs/ci-other-platforms.md)). In a container: `docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus` ([docs/docker.md](docs/docker.md)).

## Docs

[Getting started](docs/getting-started.md) (step by step, your own programs, first-run fixes) | [Settings](docs/config.md) | [The machine](docs/machine-simulator.md) | [Compilers](docs/toolchains.md) | [Commands](docs/cli.md) | [Status and limits](docs/status.md) | [Architecture](ARCHITECTURE.md) | [Governance](GOVERNANCE.md) | [Security](SECURITY.md)

Questions go in [Discussions](https://github.com/pantheongpu/pantheonsim/discussions), bugs in [issues](https://github.com/pantheongpu/pantheonsim/issues). Apache-2.0 (see LICENSE); CUDA is a trademark of NVIDIA, and this project is not affiliated with NVIDIA or AMD.
