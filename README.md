# PantheonSim

**Run CUDA and HIP programs, unmodified, on a CPU. No GPU needed.**

PantheonSim pretends to be an NVIDIA or AMD GPU (a T4, an H100, a B200, an
MI300X) on an ordinary computer. Programs built with the real `nvcc` or `hipcc`
run on it unchanged, so you can learn GPU programming, or test GPU code in CI,
without owning a GPU.

[![The PantheonSim playground: nvcc and nvidia-smi on eight simulated H100s](docs/img/playground.png)](https://pantheonsim.com/play/)

One honest limit: it checks what your code does, not how fast it runs. There is
no timing model, so a run on a physical GPU before a release still counts.

## Try it

You need a Linux machine with a C++ compiler and CMake, and no GPU. The first
command checks for them:

```bash
make doctor
```

`make doctor` installs nothing. It lists what is missing and the command that
installs it. When it says "Ready", build the simulator (a few minutes), then run
a built-in test kernel on a simulated H100:

```bash
make build
```

```bash
make demo
```

The demo ends with a line starting `PASS`. That is your first simulation.
Typing `make` on its own lists everything else you can do.

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
Makefile variables are in [docs/config.md](docs/config.md).

## Other ways to use it

- **In your browser:** [pantheonsim.com/play](https://pantheonsim.com/play/) is
  a Linux machine with simulated GPUs, an editor, a terminal and `nvcc`. No
  sign-up.
- **In GitHub Actions:** one step with `pantheongpu/setup-pantheonsim@v0` on
  the standard runners. See [pantheonsim.com/ci](https://pantheonsim.com/ci/)
  and [docs/github-action.md](docs/github-action.md).
- **In a container:** `docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus`
  needs no build. See [docs/docker.md](docs/docker.md).

## When something goes wrong

- **`make doctor` says cmake or the compiler is too old.** You need CMake 3.20
  or newer and g++ 13 or newer (or clang++ 17). `doctor` prints the install
  command for your system. On Ubuntu 22.04 that means g++ 13 from a PPA; then
  run `export CXX=g++-13`, `make clean` and `make build`.
- **Your CUDA program fails with a message about the static runtime.** It was
  built without `-cudart shared`. Rebuild with it. Programs that link the CUDA
  runtime statically, such as CuPy, cannot be simulated yet
  ([docs/status.md](docs/status.md)).
- **An AMD program says it carries code for the wrong GPU** (`hipErrorNoBinaryForGpu`).
  Build with the `gfx` name of the simulated GPU. `build/vgpu info --gpu amd/mi300x`
  shows it on the `gfx target` line.
- **A run stops with `cudaErrorMemoryAllocation` or `hipErrorOutOfMemory`.**
  The simulator will not let your computer run out of memory, so it refuses an
  allocation instead of letting the system kill programs. Ask for less memory
  (`vram_mb`), let the rest spill to disk (`ram_mb` and `memory_dir`), or lower
  the memory it keeps free (`host_reserve_mb`, which is the variable
  `VGPU_MEMORY_HOST_RESERVE_MB`). See
  [the memory settings](docs/config.md#a-small-machine-that-spills-to-disk).
- **`nvidia-smi` shows a CUDA, driver or ROCm version you do not want.** The
  versions are settings: `cuda_version`, `driver_version` and `rocm_version`.
  They change what the tools report, not your compiler.
- **A result looks wrong and you want to know why.** Run with `STRICT=1`,
  `RACE=1` or `COUNTERS=1` ([docs/diagnostics.md](docs/diagnostics.md)).

## Blocks that wait on one another

A real GPU keeps every block of a grid resident at once when the grid fits it (the
multiprocessors times the blocks each holds, from the kernel's registers, shared
memory and threads), and starts the rest as those finish. A block may therefore
spin on a word another block writes: a grid barrier made of atomics, a chain of
look-back flags, a producer in one block and its consumers in others, a lock that
blocks take while they all start together (RXMesh's `PatchLock` test).

The simulator used to run each host thread's blocks one after another, so the first
block of such a kernel spun for ever on a block that had not started, or the answer
changed from run to run. Now a kernel with a loop that polls global memory (the
machine code is searched for a loop with an atomic that returns its old value, or a
volatile or relaxed load, whose address does not change, and no store) runs the
blocks the device holds at once together: each host thread keeps a window of
resident blocks and gives every one a short turn in order, and the next block of the
grid starts when one finishes, so a grid larger than the device holds still runs its
first blocks together and the rest later, as a card does. A grid of fewer than 16
blocks runs on one thread with all its blocks resident from the start, in a fixed
order, so a program like the patch locks gives the same answer on every run. A
larger grid uses every host thread it has two blocks for, each starting with one block as
before and holding one more at a time only after no block on any thread has stored
or finished for three milliseconds, so a scan with a look-back loop keeps its speed.
Blocks that stand still sleep rather than spin.

Kernels with no such loop run as before. The variable `VGPU_LOCKSTEP` set to `0` turns
this off (the old behaviour, for debugging), and set to `all` runs every launch this way
(to check that nothing depends on the order blocks start in); `VGPU_LOCKSTEP_LOG` set to `1`
says when a launch used it. It applies to SASS, PTX and AMD kernels; clusters, cooperative
launches (which always did) and launches under a non-default `VGPU_SCHEDULER` are
left alone. The code is `exec/lockstep.hpp`; details in
[docs/cooperative.md](docs/cooperative.md#lockstep).

## Where to read more

- [docs/config.md](docs/config.md): the settings file, every key, the Makefile variables.
- [docs/cli.md](docs/cli.md): every `vgpu` command, option and environment variable.
- [docs/diagnostics.md](docs/diagnostics.md): counters, race and strict checks, error messages.
- [docs/status.md](docs/status.md): what works, what does not, and the supported GPUs.
- [docs/machine-simulator.md](docs/machine-simulator.md): `make shell`, a whole simulated machine with `nvidia-smi` and `rocm-smi`.
- [docs/driver-api.md](docs/driver-api.md), [docs/telemetry.md](docs/telemetry.md), [docs/registers.md](docs/registers.md), [docs/pantheon-workloads.md](docs/pantheon-workloads.md): the driver API, monitoring tools, device registers, stress workloads.
- [docs/ci-merge-conflicts.md](docs/ci-merge-conflicts.md): why CI cancels the runs of a pull request that has a merge conflict, and what to do.
- [nvidia/README.md](nvidia/README.md) and [amd/README.md](amd/README.md): the NVIDIA and AMD sides.
- [ARCHITECTURE.md](ARCHITECTURE.md) is how it is built and [TODO.md](TODO.md) is what is open.

The engine and its command line are called VirtualGPU (`vgpu`). `make` prints
each `vgpu` command before it runs, so you can copy it.

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
