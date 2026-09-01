# `vgpu shell` — the machine simulator

Answer a few questions, get a shell where the machine has the GPUs you asked
for. The usual tools work; so does any CUDA program you compile and run.

```bash
./scripts/build.sh
build/bin/vgpu shell
```

```
  VirtualGPU machine simulator
  Configure the machine to simulate; press Enter to accept a default.

  Available GPUs:
    nvidia/a10     NVIDIA A10                   24 GiB
    nvidia/a100    NVIDIA A100-SXM4-80GB        80 GiB
    nvidia/h100    NVIDIA H100 80GB HBM3        80 GiB
    nvidia/h200    NVIDIA H200                  141 GiB
    nvidia/b200    NVIDIA B200                  192 GiB
    amd/mi300x     AMD Instinct MI300X          192 GiB   (discovery only: AMD execution unimplemented)
    amd/mi325x     AMD Instinct MI325X          256 GiB   (discovery only: AMD execution unimplemented)
    amd/mi350x     AMD Instinct MI350X          288 GiB   (discovery only: AMD execution unimplemented)

  GPU model [nvidia/h100]: nvidia/h200
  How many GPUs [1]: 8
  VRAM per GPU (MiB) [144384]:
  CUDA version [12.4]: 12.6
  Driver version [550.54.15]: 560.35.03
  OS (number) [1]: 2
  Kernel version [6.8.0-45-generic]:
  Hostname [vgpu-sim]: dgx-h200
  Simulated GPU load 0..1 (0 = idle) [0]: 0.8
```

Then:

```
(vgpu 8xNVIDIA H200) ~$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| VGPU-SMI 0.1.0                  Driver Version: 560.35.03     CUDA Version: 12.6    |
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
| `--gpu`, `--count`, `--vram-mb` | the rack |
| `--cuda`, `--rocm`, `--driver` | reported toolkit and driver versions |
| `--os ubuntu:22.04\|ubuntu:24.04\|rocky\|debian`, `--kernel`, `--hostname` | machine identity |
| `--load 0..1` | background load, so the telemetry columns move |
| `--no-isolate` | skip the mount namespace (session tools only) |
| `-y` / `--no-prompt` | non-interactive |

## What works inside the session

| Command | Behavior |
| --- | --- |
| `nvidia-smi` | live table, CSV via `--query-gpu=...`, `--version` |
| `rocm-smi` | ROCm concise-info table |
| `rocm_agent_enumerator` | `gfx000` plus one target per virtual AMD device |
| `lspci` | the real lspci, fed synthesized PCI config space |
| `dmesg` | synthetic kernel ring buffer: boot, PCI enumeration, driver bring-up |
| `uname -r` / `-a` | the configured kernel and hostname |
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
   The tool re-executes itself under `unshare -r -m` and bind-mounts the
   synthesized `/proc/driver`, `/sys/class/drm` and `/etc/os-release` over the
   real ones. That is what makes unmodified programs — not just our wrappers —
   see the simulated machine. procfs will not accept new entries, so the whole
   `/proc/driver` directory is overlaid rather than a single file.

   No privileges are needed: the mounts live in an unprivileged user namespace
   and vanish with the session. If namespaces are unavailable the session still
   runs, with the tools working and the `/proc` paths left alone — and says so.

## Processes share the machine

Telemetry is a directory of per-process segments merged by readers, so the
session and the workloads inside it appear as several processes on one machine:
memory adds up per device, and each contributing process is listed. A workload
exiting does not erase the machine, and a segment left by a killed process is
cleaned up rather than reported as phantom hardware.

## Honest limits

- The synthetic power/temperature/clock/voltage columns are a model driven by
  real utilization, not predictions. See [telemetry.md](telemetry.md).
- AMD devices are **discovery only**: `rocm-smi`, `rocm_agent_enumerator` and
  `lspci` report them, but launching a kernel fails with a clear
  "warp size 64 is unsupported" error.
- `dmesg` output is generated for the session, not a real kernel log; there is
  no kernel driver here to log anything.
- The session simulates a machine's *GPU-facing* surfaces. It is not a
  container or a VM: the filesystem, packages and CPU are the host's.
