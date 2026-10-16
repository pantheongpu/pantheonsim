# The settings file

Every setting of the simulator is a command-line flag or a `VGPU_*` environment
variable, which is a lot to remember. A settings file keeps the ones you care
about in one place, so that `vgpu run ./my_app` simulates the machine you
described there. The [Makefile](../Makefile) at the root of the repository uses
the same file: `make init`, edit, `make run APP=./my_app`.

```bash
vgpu config init        # writes ./pantheonsim.yaml, commented, ready to edit
vgpu config show        # prints every setting in effect and where it came from
vgpu run ./my_app       # runs on what the file says
```

The file changes no behaviour of the simulator itself. `vgpu` reads it and, for
each key, sets the matching `VGPU_*` variable unless that variable is already
set. The libraries your program loads only ever see the variables, so
everything in [cli.md](cli.md) about those variables still holds.

## Where the file is looked for

The first of these that exists is used. Only one file is read; they are not
merged.

1. `--config <path>` before the command: `vgpu --config rack.yaml run ./my_app`
2. the `VGPU_CONFIG` environment variable
3. `./pantheonsim.yaml`, in the current directory
4. `$XDG_CONFIG_HOME/pantheonsim/config.yaml`, which is
   `~/.config/pantheonsim/config.yaml` when `XDG_CONFIG_HOME` is not set

A file named by 1 or 2 that does not exist is an error. For 3 and 4, no file
means no settings and everything is as it was without this feature. A file that
exists but cannot be read or is not valid stops the command with the reason, as
below; it is never skipped silently. `vgpu --help`, `vgpu --version` and
`vgpu config` do not read the file for their own work, so a typo in it never
locks you out of the commands that explain it.

## Precedence

```
command-line flag  >  environment variable  >  settings file  >  default
```

A flag always won over the environment, and the file sits below both. So the
file is the machine you usually want, an exported variable changes it for a
shell session, and a flag changes it for one command.

```yaml
# pantheonsim.yaml
gpu: nvidia/t4
count: 2
```

```bash
vgpu run ./my_app                              # 2 x T4: the file
VGPU_DEVICE_COUNT=4 vgpu run ./my_app          # 4 x T4: the environment beats the file
VGPU_DEVICE_COUNT=4 vgpu run --count 8 ./my_app   # 8 x T4: a flag beats both
```

`vgpu config show` prints the result and the reason for each value.

## The format

A flat list of `key: value` lines. It is the small part of YAML that fits that
shape; nothing else is accepted, so a mistake is an error and not a guess.

- `# comment` runs to the end of the line, after a value or on its own line
  (a `#` inside quotes is part of the value).
- Blank lines are fine.
- A value may be wrapped in `"double"` or `'single'` quotes. Quotes are only
  needed for a value that contains ` #`.
- No indentation, no nesting, no lists, and each key at most once.
- A key without a value is an error. To use the default, delete the line.
- Booleans are `true` or `false` (`yes`, `no`, `on` and `off` work too).

What goes wrong is said with the file, the line and what is allowed:

```
$ vgpu info
vgpu: pantheonsim.yaml:3: unknown key 'tread'; did you mean 'threads'?
$ vgpu info
vgpu: pantheonsim.yaml:2: count: expected a whole number from 1 to 16, got 'lots'
$ vgpu info
vgpu: pantheonsim.yaml:1: gpu: 'nvidia/h1000' is not a GPU profile; did you mean 'nvidia/h100'? (`vgpu list-gpus` prints them)
```

## Keys

`vgpu config keys` prints this list from the program itself. In the table, "sets"
is the environment variable a key stands for, so [cli.md](cli.md) has the full
story of each one.

**Device**

| Key | Accepts | Sets | Default | What it does |
| --- | --- | --- | --- | --- |
| `gpu` | a profile id from `vgpu list-gpus`, such as `nvidia/h100` or `amd/mi300x` | `VGPU_GPU` | `nvidia/h100` | The GPU model to present. Checked against the built-in profiles. |
| `count` | whole number, 1 to 16 | `VGPU_DEVICE_COUNT` | 1 | How many identical devices. |
| `vram_mb` | whole number of MiB, at least 1 | `VGPU_VRAM_MB` | the profile's own | Memory per device, instead of the profile's. |
| `cuda_version` | `major.minor`, such as `13.0` | `VGPU_CUDA_VERSION` | this build's | The CUDA version `nvidia-smi` and NVML report. |
| `driver_version` | dotted version, such as `580.65.06` | `VGPU_DRIVER_VERSION` | this build's | The NVIDIA driver version `nvidia-smi` and NVML report. |
| `rocm_version` | dotted version, such as `6.4.0` | `VGPU_ROCM_VERSION` | this build's | The ROCm version `rocm-smi` and `amd-smi` report. |
| `os` | `ubuntu`, `rocky` or `debian`, with an optional version (`ubuntu:24.04`) | `VGPU_SHELL_OS` | `ubuntu:22.04` | The operating system `vgpu shell` presents. Read by `vgpu shell` only. |

**Memory** (see [Memory](#a-small-machine-that-spills-to-disk) below)

| Key | Accepts | Sets | Default | What it does |
| --- | --- | --- | --- | --- |
| `ram_mb` | whole number of MiB, at least 1 | `VGPU_MEMORY_RAM_MB` | unlimited | Host RAM that device memory may use before the rest goes to `memory_dir`. |
| `memory_dir` | a directory | `VGPU_MEMORY_DIR` | never spill | Where device memory past `ram_mb` goes. A card larger than the machine runs, at the speed of the disk. |
| `host_reserve_mb` | whole number of MiB, 0 or more | `VGPU_MEMORY_HOST_RESERVE_MB` | 6% of the memory limit, 1 to 4 GiB | Memory the guard leaves to the rest of the machine. |
| `host_guard` | `true` or `false` | `VGPU_MEMORY_HOST_GUARD` | `true` | `false` turns the host-memory guard off. |

**Execution**

| Key | Accepts | Sets | Default | What it does |
| --- | --- | --- | --- | --- |
| `threads` | whole number, 1 to 65536 | `VGPU_THREADS` | one per core | Host threads that run blocks. |
| `max_steps` | whole number, 0 or more | `VGPU_MAX_STEPS` | built in | The per-thread instruction budget against a kernel that never ends. `0` removes it. |
| `sass` | `auto`, `off` or `only` | `VGPU_SASS` | `auto` | The engine for NVIDIA kernels. `auto` runs SASS when the binary carries it and PTX otherwise, `off` forces PTX, `only` insists on SASS and does not fall back. |
| `simd` | `auto`, `base`, `avx2` or `off` | `VGPU_SIMD` | `auto` | Caps the vector instruction set the engines use; `off` uses scalar code. For ruling them out when a result looks wrong. |
| `scheduler` | `deterministic`, `random` or `adversarial` | `VGPU_SCHEDULER` | `deterministic` | The warp scheduling order. `random` and `adversarial` find bugs that depend on which warp runs first. PTX engine only, so set `sass: off` with them. |
| `scheduler_seed` | whole number, 0 or more | `VGPU_SCHEDULER_SEED` | 0 | Seed for `random` and `adversarial`; the same seed replays the same run. |

**Checks and output** (see [diagnostics](diagnostics.md))

| Key | Accepts | Sets | Default | What it does |
| --- | --- | --- | --- | --- |
| `strict` | `true` or `false` | `VGPU_STRICT` | `false` | Refuse undefined behaviour: integer division by zero, storing a register nothing has written. |
| `race` | `off`, `on` or `all` | `VGPU_RACE` | `off` | Report unordered shared-memory access between warps. `all` also reports stores that change nothing. |
| `counters` | `true` or `false` | `VGPU_COUNTERS` | `false` | Print exact per-launch performance counters. |
| `quiet` | `true` or `false` | `VGPU_QUIET` | `false` | Silence every simulator message on stderr, errors included. |
| `trace` | `true` or `false` | `VGPU_TRACE` | `false` | Log every runtime and driver entry point. |

Switching a key off (`false`, `off`, `auto`) means "as if the key were not
there": the variable is left unset, not set to `0`. So `strict: false` does not
switch off a `VGPU_STRICT=1` you exported, because the environment is above the
file. The one key that sets a variable to switch something off is
`host_guard: false`, which sets `VGPU_MEMORY_HOST_GUARD=0`.

Variables that are not in this list (`VGPU_SASS_LOG`, `VGPU_PTX_CONTRACT`, the
trace selectors and so on) are for diagnosing the simulator itself. Set them in
the environment as before.

## Commands

```
vgpu config show [flags]      every setting, its value, where it came from
vgpu config init [PATH] [--force]   write a commented starter file
vgpu config keys              the table above, from the program
```

`config show` takes the flag form of every key (`--gpu`, `--count`, `--vram-mb`,
`--ram-mb`, `--memory-dir`, `--scheduler-seed`, and so on; `--strict`, `--race`,
`--counters`, `--quiet` and `--trace` take no value) so you can ask what a command
line would resolve to:

```
$ vgpu config show --count 8
settings file: pantheonsim.yaml
precedence:    flag > environment > file > default

key              value                               from
gpu              nvidia/t4                           file pantheonsim.yaml:1
count            8                                   flag --count (pantheonsim.yaml:2 says 2)
vram_mb          the profile's own                   default
cuda_version     this build's                        default
...
```

The `from` column is one of `default`, `file <path>:<line>`, `environment
<VARIABLE>` or `flag --<name>`. When a flag or a variable hides a value in the
file, it says which line and what it said.

`config init` refuses to overwrite a file that exists; pass `--force` to
replace it. With no path it writes `./pantheonsim.yaml`; give it
`~/.config/pantheonsim/config.yaml` for a per-user default (the directory is
created).

## Examples

**An NVIDIA machine.** Eight H100s, and the checks that catch bugs hardware
hides:

```yaml
gpu: nvidia/h100
count: 8
strict: true
race: on
```

**An AMD machine.** A single MI300X, reporting the ROCm release you test against
(supported releases: 6.4 and 7.x):

```yaml
gpu: amd/mi300x
count: 1
rocm_version: 7.2.0
```

**A small-memory machine that spills to disk.** <a id="a-small-machine-that-spills-to-disk"></a>
A laptop with 16 GB of RAM can still run a program that wants an 80 GB card.
Device memory beyond `ram_mb` goes to files under `memory_dir`, at the speed of
the disk, and the host guard keeps the simulator from taking the whole machine:

```yaml
gpu: nvidia/h100
ram_mb: 8192                     # use at most 8 GiB of RAM for device memory
memory_dir: /var/tmp/pantheonsim # the rest goes here; needs disk space for it
host_reserve_mb: 4096            # always leave 4 GiB to the rest of the machine
```

Without `memory_dir`, an allocation that would not fit fails with
`cudaErrorMemoryAllocation` or `hipErrorOutOfMemory` instead of the operating
system's out-of-memory killer firing.

**Finding an ordering bug.** Run the same program under random warp order, and
replay a failing run with its seed:

```yaml
sass: off
scheduler: random
scheduler_seed: 17
```

## What `cuda_version`, `driver_version` and `rocm_version` do

They change what the machine reports: what `nvidia-smi`, NVML (`nvmlSystemGetCudaDriverVersion`
and friends), `rocm-smi`, `amd-smi` and a `vgpu shell`'s `nvcc --version` print,
so you can check that a program or an installer takes the version it finds. They
do not change the compiler. Your own `nvcc` or `hipcc` still builds the program,
at whatever release is installed on your machine, and the simulator runs the
result. `cuda_version` must be `major.minor`, the form the driver accepts.

## `vgpu shell`

`vgpu shell` takes its defaults from the same file: `gpu`, `count`, `vram_mb`,
`cuda_version`, `rocm_version`, `driver_version` and `os` become the shell's
`--gpu`, `--count`, `--vram-mb`, `--cuda`, `--rocm`, `--driver` and `--os`, and a
flag you type still wins. The shell's other options (`--kernel`, `--hostname`,
`--load`) have no key.

Run with `-y` (`vgpu shell -y`, which is what `make shell` does) to take the file
as it is. Without `-y` the shell asks its questions as before, with the file's
values as the answers it offers. The one exception is the operating system: the
menu always starts at Ubuntu 22.04, so use `-y` to get `os` from the file.

`./run` at the root of the repository is `build/vgpu --config FILE shell -y` with
the file checked first, and starts the machine from `./pantheonsim.yaml` when no
file is named (writing the commented starter if there is none).

## The Makefile

`make help` lists them. They are thin wrappers that print the command they run.
`make` with no target is `make deps build`, and then `./run` starts the machine
(`./run my.yaml` for another settings file):

| Target | Runs |
| --- | --- |
| `make` | `make deps`, then `make build` (with the compiler `deps` found, if the default is too old), then prints what to type next |
| `make deps` | `scripts/deps.sh`: installs cmake and a C++20 compiler when missing (Debian and Ubuntu; it asks first, `YES=1` answers yes, no terminal means it only prints the command; `PPA=1` with `YES=1` adds the g++ 13 PPA on Ubuntu 22.04). Never installs CUDA or ROCm |
| `make doctor` | `scripts/doctor.sh`: checks cmake, a C++20 compiler and memory, and says what to install |
| `make build` | `scripts/build.sh` at low priority (`NICE=` to change) |
| `make demo` | `build/vgpu demo vectoradd`, building first if needed |
| `make list-gpus` | `build/vgpu list-gpus` |
| `make init` | `build/vgpu config init` (`FORCE=1` to overwrite) |
| `make config` | `build/vgpu config show` |
| `make shell` | `build/vgpu shell -y` (`ASK=1` to be asked) |
| `make run APP=./my_app ARGS="..."` | `build/vgpu run ./my_app ...`, building first if needed |
| `make test` | `scripts/test.sh` |
| `make clean` | `rm -rf build` |

Variables on the make command line become the matching flags and beat the file:
`GPU=`, `COUNT=`, `VRAM_MB=`, `THREADS=`, `MAX_STEPS=`, `STRICT=1`, `RACE=1`,
`COUNTERS=1`, `QUIET=1`, `TRACE=1`, `CONFIG=` (the file to use), and `PRELOAD=1` for
`run`, which is what a Python program needs. For `shell`: `CUDA_VERSION=`,
`ROCM_VERSION=`, `DRIVER_VERSION=` and `SHELL_OS=`. For `demo`: `N=`. Only the
make command line counts, so a `COUNT` or `CUDA_VERSION` that happens to be in your
shell's environment does nothing.
