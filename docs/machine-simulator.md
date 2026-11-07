# `vgpu shell` — the machine simulator

Answer a few questions, get a shell where the machine has the GPUs you asked
for. The usual tools work; so does any CUDA program you compile and run.

## Start it with `./run`

After `make`, `./run` starts the machine from your settings file, with no
questions:

```bash
./run                              # the machine in ./pantheonsim.yaml (a starter is written if missing)
./run my.yaml                      # the machine in my.yaml
./run my.yaml -- nvidia-smi -L     # one command in the machine; exits with its status
```

It is `build/vgpu --config FILE shell -y` with three conveniences: the file is
checked first and a mistake stops with the line and the reason; a missing build is
built with `make build` (never installing packages); and where the system forbids
user namespaces (Ubuntu 24.04 can; see below) it adds `--no-isolate` and says so,
instead of failing. The rest of this page is `vgpu shell` itself.

```bash
./scripts/build.sh
build/bin/vgpu shell
```

```
  VirtualGPU machine simulator
  Configure the machine to simulate; press Enter to accept a default.

  Available GPUs:        (the real list has every profile; this is an excerpt)
    nvidia/a10     NVIDIA A10                   24 GiB
    nvidia/a100    NVIDIA A100-SXM4-80GB        80 GiB
    nvidia/h100    NVIDIA H100 80GB HBM3        80 GiB
    nvidia/h200    NVIDIA H200                  141 GiB
    nvidia/b200    NVIDIA B200                  192 GiB
    amd/mi300x     AMD Instinct MI300X          192 GiB
    amd/mi325x     AMD Instinct MI325X          256 GiB
    amd/mi350x     AMD Instinct MI350X          288 GiB
    amd/mi455x     AMD Instinct MI455X          432 GiB

  GPU model [nvidia/h100]: nvidia/h200
  How many GPUs [1]: 8
  VRAM per GPU (MiB) [144384]:
  CUDA version [13.0]: 12.6
  Driver version [580.00.00]: 560.35.03
  OS (number) [1]: 2
  Kernel version [6.8.0-45-generic]:
  Hostname [vgpu-sim]: dgx-h200
  Simulated GPU load 0..1 (0 = idle) [0]: 0.8
```

Then:

```
(vgpu 8xNVIDIA H200) ~$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03            Driver Version: 560.35.03     CUDA Version: 12.6    |
...
|   0  NVIDIA H200                  On  | 00000000:01:00.0 Off |                  N/A |
| 25%   43C    P0        531W / 700W |        0MiB / 144384MiB |    80%      Default |

(vgpu 8xNVIDIA H200) ~$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  560.35.03  Release Build  (VirtualGPU)

(vgpu 8xNVIDIA H200) ~$ uname -a
Linux dgx-h200 6.8.0-45-generic #1 SMP x86_64 x86_64 x86_64 GNU/Linux

(vgpu 8xNVIDIA H200) ~$ nvcc -cudart shared vector_add.cu -o app && ./app
PASS
```

Scripted use skips the questions:

```bash
build/bin/vgpu shell -y --gpu amd/mi300x --count 8 --rocm 6.2.0 \
  --driver 6.3.6 --os ubuntu:24.04 --hostname mi300-node --load 0.65
```

| Flag | Meaning |
| --- | --- |
| `--gpu`, `--count 1..16`, `--vram-mb` | the rack |
| `--cuda`, `--rocm`, `--driver` | reported toolkit and driver versions; the CUDA and driver defaults are the ones this build's runtime carries (what `build/bin/nvidia-smi --version` prints outside a session), the ROCm default is 6.2.0 |
| `--os ubuntu:22.04\|ubuntu:24.04\|rocky\|debian`, `--kernel`, `--hostname` | machine identity |
| `--load 0..1` | background load, so the telemetry columns move |
| `--no-isolate` | skip the namespaces (session tools only) |
| `-y` / `--no-prompt` | non-interactive |
| `-c <command>` | run one command and exit with its status (128+N if a signal killed it) |

Every value is checked before the session starts: a flag with no value, a
number that is not a whole number in range, or a version that is not dotted
digits is an error (exit 2) naming the flag, not a quiet default. `-c` runs
under `$SHELL`; bash gets the session rcfile, and other shells (dash, zsh) run
without it, with the session's `PATH` and `LD_LIBRARY_PATH` already set.

## What works inside the session

| Command | Behavior |
| --- | --- |
| `nvidia-smi` | live table, `-q [-d SECTIONS]`, `-x`, `--query-gpu=... --format=csv`, `-L`, `-i`, `-l`/`-lms`, `topo -m`, `--version`, `-h` |
| `rocm-smi` | ROCm concise-info table; `-a`, `-i`, `--showproductname`, `--showmeminfo`, `-t`, `-P`, `-u`, `-c`, `-s`, `-g`, `-f`, `-v`, `--showbus`, `--showhw`, `--showdriverversion`, `--showpids`, `--showtopo`, `--showmemuse`, `--showperflevel`, `--showvoltage`, `--showenergycounter`, `--showcomputepartition`, `--showmemorypartition`, `--showpagesinfo`, `--showrasinfo`, `--showxgmierr`, `--showfwinfo`, `--showmaxpower`, `--showserial`, `--showuniqueid`, `-d N`, `--json`, `--csv` |
| `amd-smi` | `list`, `static` (every section), `metric` (usage, power, clock, temperature, PCIe, ECC, fan, memory, ...), `process`, `topology`, `monitor`, `partition`, `xgmi`, `bad-pages`, `firmware`, `ras --cper`, `version`; `--json`, `-g N` |
| `rocm_agent_enumerator` | `gfx000` plus each AMD device's own target (gfx942, gfx950, gfx90a, gfx1030, gfx1031, gfx1100, gfx1201, gfx1250) |
| ROCm's `rocm_agent_enumerator` | reads the session's KFD topology (`/sys/class/kfd/kfd/topology`) in an isolated AMD session, unmodified; the session also has `/dev/kfd` and `/dev/dri` |
| ROCm's `rocm-smi` | runs unmodified in an isolated AMD session, on VirtualGPU's `librocm_smi64` (the session sets `ROCM_SMI_LIB_PATH` and shows `/sys/module/amdgpu` loaded), and reports what VirtualGPU's `rocm-smi` does |
| `rocminfo` | ROCm's own where the host has it (it reads the simulated agents through the session's HSA runtime), otherwise VirtualGPU's, which prints the same |
| `lspci` | the real lspci over the simulated devices' config space (isolated, over the session's `/sys/bus/pci`, so `-k` and `-vv` show the bound driver), naming cards the host's PCI ID database predates; without pciutils, VirtualGPU's own lspci, which prints the same for `-n`/`-nn`, `-D`, `-d`, `-s`, `-k`, `-v`, `-m`/`-mm`, `-vmm`, `-x`..`-xxxx` and `-t` |
| `dmesg` | synthetic kernel ring buffer: boot, PCI enumeration, driver bring-up |
| `uname` | the configured kernel and hostname, for any of `-asnrvmpio`, combined (`-sr`) or long (`--kernel-release`) |
| `hostname` | the configured hostname |
| `nvcc --version` | the configured CUDA version (a real nvcc still compiles) |
| `cat /proc/driver/nvidia/version` | the simulated NVRM banner |
| `ls /proc/driver/nvidia/gpus/` | one entry per device, with an `information` file |
| `ls /sys/class/drm/` | `card0..cardN`, each with `device/vendor` and `device/device` |
| any CUDA program | runs on the CPU engine via the shims |

## How it works

Two layers:

1. **Always.** A session `bin/` directory goes first on `PATH` and supplies the
   tools; the shim directory goes first on `LD_LIBRARY_PATH` and supplies
   `libcuda.so.1`, `libcudart.so.13`, `libnvidia-ml.so.1` and
   `libnvidia-encode.so.1`; the session process holds the virtual devices open,
   which is what publishes telemetry.

   Both search paths are re-asserted *after* your `~/.bashrc` is sourced —
   otherwise a line like `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:...`
   silently shadows the shim and programs reach NVIDIA's real cudart instead.

2. **When user namespaces are available** (the default; `--no-isolate` to skip).
   The tool re-executes itself under `unshare -r -m -u`, bind-mounts the
   synthesized `/proc/driver`, `/sys/class/drm` and `/etc/os-release` over the
   real ones, and sets `--hostname` in the session's own UTS namespace. That is
   what makes unmodified programs — not just our wrappers — see the simulated
   machine: `gethostname()` and `/proc/sys/kernel/hostname` report the
   configured name. procfs will not accept new entries, so the whole
   `/proc/driver` directory is overlaid rather than a single file.

   No privileges are needed: the mounts and the hostname live in unprivileged
   namespaces and vanish with the session; the host's hostname is untouched. If
   namespaces are unavailable the session still runs, with the tools working
   (the session's `hostname` and `uname` still print the configured name) and
   the `/proc` paths left alone — and says so.

## Network, files and programs in the session

The session is a view of your own computer with the GPU surfaces replaced, so
downloading and running programs works as it does outside. Read from
`src/cli/shell.cpp` (`unshare -r -m -u`, no network, PID or filesystem namespace)
and checked on a real host:

- **Network:** the host's. `curl`, `git clone`, `pip install` and `wget` reach the
  internet exactly as they do from your own shell; no flag is needed.
- **Files:** the host's. Your home, your current directory and every installed
  program are there; only `/proc/driver`, `/sys/class/drm`, `/etc/os-release` and
  the PCI and hwmon trees are replaced.
- **Users:** you are `root` inside (`-r` maps your user to uid 0), and files owned
  by your user show as owned by root, system files as `nobody`. That is a user
  namespace's picture, not real privilege: you cannot write to `/usr`, and
  `sudo` and `apt install` do not work in the session. Install packages from
  outside it (`make deps` and your package manager), or use user-level installs
  in it: `python3 -m venv ~/v && ~/v/bin/pip install torch`,
  conda, or a downloaded tarball in your home. On Ubuntu 24.04 and other systems
  that mark Python as externally managed (PEP 668) a plain `pip install` and
  `pip install --user` are refused; use a virtual environment.
- **Compilers:** the host's `nvcc` and `hipcc`, if installed. Build CUDA programs
  with `-cudart shared`. See [toolchains.md](toolchains.md).
- **With `--rootfs`** none of the "files" above hold: see [A private root](#a-private-root---rootfs).
- **With `--no-isolate`** (what `./run` uses on a system that restricts user
  namespaces) everything above holds, except that `/proc` and `/etc` show your
  real system and you keep your own user, so `sudo` works as it does outside.

## A private root: `--rootfs`

By default the session is a view of your own computer (above). `--rootfs` gives
it a machine of its own instead: a new root on a tmpfs, built inside the mount
namespace `vgpu shell` already creates and entered with `pivot_root`, so the
host's files are not there unless you add them.

```bash
vgpu shell -y --rootfs                                  # nothing of yours is visible
vgpu shell -y --rootfs -v ~/project:/work -v ~/data:/data:ro --home ~/.vgpu-home
vgpu shell -y --rootfs -v "$PWD":/work -c 'cd /work && make test'
```

What the session has:

| Path | What it is |
| --- | --- |
| `/usr`, `/bin`, `/sbin`, `/lib*` | The host's, **read-only**, so compilers, Python and the dynamic loader work. On a host where `/bin`, `/lib` and `/sbin` are links into `/usr` they are links here. |
| `/etc` (a few entries) | The host's, read-only: `ld.so.cache`, `ld.so.conf*`, `resolv.conf`, `hosts`, `host.conf`, `nsswitch.conf`, `gai.conf`, `passwd`, `group`, `ssl`, `ca-certificates*`, `pki`, `crypto-policies`, `alternatives`, `bash.bashrc`, `profile*`, `shells`, `localtime`, `terminfo`, `inputrc`, `mime.types`, `protocols`, `services`, `environment`. `/etc/os-release` is the simulated machine's. Nothing else of the host's `/etc` is there (no `shadow`, `sudoers`, `ssh`, other programs' settings); the rest of `/etc` is the session's own, empty and writable. |
| `/root` (`$HOME`), `/home`, `/opt`, `/tmp`, `/var/tmp`, `/work`, `/run`, `/mnt`, `/srv` | Empty and the session's own, in memory, discarded on exit. The one thing in them that is not yours is the simulator's own install and its session directory, which sit at their host paths (see below), so `/tmp` holds a `vgpu-session-N` directory and `/home` or `/opt` hold the path down to the build if it lives there. |
| `/proc`, `/sys` | As in a default session: the host's, with the synthesized `/proc/driver`, `/sys/class/drm`, hwmon and PCI trees (and on an AMD machine the KFD topology and `amdgpu` module) over them. `/proc` shows the host's processes; there is no PID namespace. |
| `/dev` | `null`, `zero`, `full`, `random`, `urandom`, `tty`, a fresh `devpts` (`pts`, `ptmx`) and `shm`. On an AMD machine also the simulated `/dev/kfd` and `/dev/dri/*`, which open as `/dev/null` as they do in a default session. |
| the simulator | The directory holding `vgpu`, `shim/` and `bin/` is mounted **read-only at its host path**, so `PATH`, `LD_LIBRARY_PATH` and the session's scripts are unchanged. The session directory (tools, telemetry, generated files) is at its usual `/tmp/vgpu-session-N`. |
| `/usr/local/cuda`, `/opt/rocm` | Where a real machine has them. If the host has a CUDA toolkit at `/usr/local/cuda` it is there (through `/usr`, or, when the link leads outside `/usr`, mounted read-only where it points); if the host has `/opt/rocm` it is mounted read-only there. If it has none, nothing is created and the banner says so. No file is invented. |

The network is the host's, as in a default session. You are `root` in a user
namespace (files you own show as root), `HOME=/root`, and the working directory
is `/root`; the host's home directory, `TMPDIR` and cwd are not there.

**Adding host directories: `-v HOST:DEST[:ro]`**, repeatable, as in `docker run -v`:

- Read-write unless `:ro`; `:ro` makes the mount and everything mounted below it read-only (writes fail with `EROFS`).
- `HOST` is a file or a directory and must exist (symlinks are followed); `DEST` must be absolute, not `/`, and not under `/proc` or `/sys`. A bad `-v` exits `2` before a session starts. Paths cannot contain `:`.
- `DEST` and its parents are created inside the session. Under the read-only system trees they cannot be created, so a new directory there is an error (exit `1`); put your mounts under `/work`, `/data`, `/mnt` and the like.
- Mounts are applied in the order given, after the system, so a later `-v` goes over an earlier one, and a `-v` over an existing system path (`-v ./hosts:/etc/hosts:ro`, `-v ./mylibs:/usr/local/lib/x`) replaces what is there; that is allowed and means what it says. `-v` onto `/` is refused.
- `-v` and `--home` need `--rootfs`: without it the session already sees the host's files, so they are an error (exit `2`) that says to add it.
- `--home DIR` is `-v DIR:/root` read-write, so shell history, `~/.cache` and a virtual environment survive between runs. Without it `/root` is a tmpfs.

**When user namespaces are not available** (Ubuntu 24.04 restricts them with AppArmor:
`kernel.apparmor_restrict_unprivileged_userns=1`; some containers refuse them),
a default session falls back to the host's files with a message, but `--rootfs`
**does not**: you asked for isolation, so it exits `1` with the reason, before
anything runs. Use `./run` (which falls back) or drop `--rootfs`. `VGPU_SHELL_NO_NAMESPACES=1`
reproduces this on a host that has namespaces, for tests.

Limits: the session's tmpfs counts against the host's memory (there is no size cap);
`apt`, `sudo` and anything that writes to `/usr` do not work (a read-only system by design;
install into `/root` with `--home`, or into a mounted directory); the host's PIDs are visible in `/proc`.

## Processes share the machine

Telemetry is a directory of per-process segments merged by readers, so the
session and the workloads inside it appear as several processes on one machine:
memory adds up per device, and each contributing process is listed. A workload
exiting does not erase the machine, and a segment left by a killed process is
cleaned up rather than reported as phantom hardware.

## Honest limits

- The synthetic power/temperature/clock/voltage columns are a model driven by
  real utilization, not predictions. See [telemetry.md](telemetry.md).
- What a simulated AMD card has no value for -- VBIOS, serial numbers,
  firmware versions, energy counters -- `rocm-smi` and `amd-smi` report as
  N/A rather than invent. `/dev/kfd` and `/dev/dri` exist in an isolated AMD
  session but open as `/dev/null`, and a kernel-driver request (an ioctl) is
  refused: programs reach the GPUs through the session's HIP and HSA runtimes,
  not the kernel.
- `dmesg` output is generated for the session, not a real kernel log; there is
  no kernel driver here to log anything.
- The session simulates a machine's *GPU-facing* surfaces. It is not a
  container or a VM: the filesystem, packages and CPU are the host's.
