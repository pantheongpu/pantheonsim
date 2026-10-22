# Getting started

The [README](../README.md) has the two commands (`make`, then `./run`). This page
is everything around them: the step-by-step build, running your own programs
without the machine shell, choosing the simulated machine, and the first-run
problems people hit.

## Try it step by step

`make` on its own does the next three things in order. Do them one at a time when
you want to see each result.

You need a Linux machine with a C++ compiler and CMake, and no GPU. The first
command checks for them:

```bash
make doctor
```

`make doctor` installs nothing. It lists what is missing and the command that
installs it. `make deps` installs it for you on Debian and Ubuntu: it lists the
packages and asks first (`make deps YES=1` answers yes; with no terminal it
installs nothing and prints the command instead). When `doctor` says "Ready",
build the simulator (a few minutes), then run a built-in test kernel on a
simulated H100:

```bash
make build
```

```bash
make demo
```

The demo ends with a line starting `PASS`. That is your first simulation.
`make help` lists everything else you can do, and `./run` takes you inside a
simulated machine (see below).

### Step inside the machine

`./run` starts a simulated machine from the settings file and gives you a shell
in it. `nvidia-smi` and `rocm-smi` answer for the simulated GPUs, and you can
compile, download and run programs as on your own computer.
[machine-simulator.md](machine-simulator.md) says what the machine is and what
it shares with your computer; [toolchains.md](toolchains.md) is about `nvcc` and
`hipcc`. `./run my.yaml -- nvidia-smi -L` runs one command in it and exits.

### What works inside, and what does not

The machine is your own computer with the GPU surfaces replaced, not a separate
system. What follows was read from `src/cli/shell.cpp` and checked on an Ubuntu 24.04
host (Python 3.12) with `unshare -r -m -u`, the call `vgpu shell` makes. It was not run
through `vgpu shell` itself, and no end-to-end test exercises pip.

- **Files are your real files.** Your home and current directory are the host's, so
  anything you create or download persists after `exit`, owned by you. Only
  `/proc/driver`, `/sys/class/drm`, `/etc/os-release` and the PCI and hwmon trees differ.
- **Network is the host's.** `curl`, `wget` and `git clone` work as outside.
- **No `sudo`, no `apt` inside.** You are `root` only in name: `apt-get install` fails
  ("are you root?") and `/usr` is not writable. Install system packages on your own
  machine, outside the session (`make deps` does this for the build tools).
- **Python packages need a virtual environment.** Verified to work with the
  user-namespace call above (venv created, `pip install` of a small package, import):

  ```bash
  python3 -m venv ~/v
  ~/v/bin/pip install torch      # or any package
  ~/v/bin/python my_script.py
  ```

  A plain `pip install x` and `pip install --user x` are refused on Ubuntu 24.04 and
  similar systems with "externally-managed-environment" (PEP 668). Not tested: installing
  `torch` itself in the machine through `./run`, and `conda`.
- **Compilers are the host's** `nvcc` and `hipcc`; see [toolchains.md](toolchains.md).
  Build CUDA programs with `-cudart shared`.
- **Where user namespaces are restricted** (`./run` says so), the machine is not
  isolated and you keep your own user, so `sudo` works as it does outside your session.

To run a Python script in the venv without a terminal: `./run my.yaml -- ~/v/bin/python my_script.py`.

### Run your own program

Build your program with the real compiler. The `-cudart shared` flag lets the
simulator stand in for the CUDA runtime.

```bash
nvcc -cudart shared my_app.cu -o my_app
```

```bash
make run APP=./my_app GPU=nvidia/h100
```

For an AMD GPU, build for the GPU you will simulate (`gfx942` is the MI300X),
then run it the same way:

```bash
hipcc --offload-arch=gfx942 my_app.hip -o my_app
```

```bash
make run APP=./my_app GPU=amd/mi300x
```

Python works too, PyTorch included: `make run APP=python ARGS="train.py" PRELOAD=1`.
`make list-gpus` shows every GPU you can pick.

## Choose your simulated machine

Instead of typing the GPU and the options every time, write them once in a file
called `pantheonsim.yaml`:

```bash
make init
```

That writes a commented starter file in the current directory. Edit it, for example:

```yaml
gpu: nvidia/h100     # which GPU
count: 8             # how many of them
strict: true         # refuse undefined behaviour, such as dividing by zero
race: on             # report shared-memory data races
```

```bash
make config
```

`make config` prints every setting in effect and where each one came from, and
`make run` uses them. A setting given on the `make` line (`make run APP=./my_app
COUNT=2`) wins over the file. Every key, the order of precedence and the
Makefile variables are in [docs/config.md](config.md).

## Other ways to use it

- **In your browser:** [pantheonsim.com/play](https://pantheonsim.com/play/) is
  a Linux machine with simulated GPUs, an editor, a terminal and `nvcc`. No
  sign-up.
- **In GitHub Actions:** one step with `pantheongpu/setup-pantheonsim@v0` on
  the standard runners. See [pantheonsim.com/ci](https://pantheonsim.com/ci/)
  and [docs/github-action.md](github-action.md).
- **In a container:** `docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus`
  needs no build. See [docs/docker.md](docker.md).

## When something goes wrong

- **`make doctor` says cmake or the compiler is too old.** You need CMake 3.20
  or newer and g++ 12 or newer (or clang++ 17). `doctor` prints the install
  command for your system. On Ubuntu 22.04 that means `sudo apt-get install g++-12`; then
  run `export CXX=g++-12`, `make clean` and `make build`.
- **Your CUDA program fails with a message about the static runtime.** It was
  built without `-cudart shared`. Rebuild with it. Programs that link the CUDA
  runtime statically, such as CuPy, cannot be simulated yet
  ([docs/status.md](status.md)).
- **An AMD program says it carries code for the wrong GPU** (`hipErrorNoBinaryForGpu`).
  Build with the `gfx` name of the simulated GPU. `build/vgpu info --gpu amd/mi300x`
  shows it on the `gfx target` line.
- **A run stops with `cudaErrorMemoryAllocation` or `hipErrorOutOfMemory`.**
  The simulator will not let your computer run out of memory, so it refuses an
  allocation instead of letting the system kill programs. Ask for less memory
  (`vram_mb`), let the rest spill to disk (`ram_mb` and `memory_dir`), or lower
  the memory it keeps free (`host_reserve_mb`, which sets
  `VGPU_MEMORY_HOST_RESERVE_MB`). See
  [the memory settings](config.md#a-small-machine-that-spills-to-disk).
- **`nvidia-smi` or `amd-smi` shows a CUDA, driver or ROCm version you do not want.** The
  versions are settings: `cuda_version`, `driver_version` and `rocm_version`.
  They change what the tools report, not your compiler.
- **A result looks wrong and you want to know why.** Run with `STRICT=1`,
  `RACE=1` or `COUNTERS=1` ([docs/diagnostics.md](diagnostics.md)).

## Where to read more

- [docs/config.md](config.md): the settings file, every key, the Makefile variables.
- [docs/cli.md](cli.md): every `vgpu` command, option and environment variable.
- [docs/diagnostics.md](diagnostics.md): counters, race and strict checks, error messages.
- [docs/status.md](status.md): what works, what does not, and the supported GPUs.
- [docs/machine-simulator.md](machine-simulator.md): `make shell`, a whole simulated machine with `nvidia-smi` and `rocm-smi`.
- [docs/driver-api.md](driver-api.md), [docs/telemetry.md](telemetry.md), [docs/registers.md](registers.md), [docs/pantheon-workloads.md](pantheon-workloads.md): the driver API, monitoring tools, device registers, stress workloads.
- [docs/ci-merge-conflicts.md](ci-merge-conflicts.md): why CI cancels the runs of a pull request that has a merge conflict, and what to do.
- [nvidia/README.md](../nvidia/README.md) and [amd/README.md](../amd/README.md): the NVIDIA and AMD sides.
- [ARCHITECTURE.md](../ARCHITECTURE.md) is how it is built and [TODO.md](../TODO.md) is what is open.

The engine and its command line are called VirtualGPU (`vgpu`). `make` prints
each `vgpu` command before it runs, so you can copy it.

## Help, contributing and security

Questions and ideas go in [Discussions](https://github.com/pantheongpu/pantheonsim/discussions),
bugs in [issues](https://github.com/pantheongpu/pantheonsim/issues). How changes
are decided is in [GOVERNANCE.md](../GOVERNANCE.md), and security problems should be
reported privately as [SECURITY.md](../SECURITY.md) describes.

## License

Apache-2.0 (see ../LICENSE). The `vgpu_cuda.h` header is a clean-room subset
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
