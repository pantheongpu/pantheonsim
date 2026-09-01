# Device discovery and telemetry

VirtualGPU presents its virtual devices to the tools people actually use to
look at GPUs: `nvidia-smi`, `rocm-smi`, `rocm_agent_enumerator`, `lspci`, and
anything speaking NVML.

```bash
./scripts/build.sh
build/bin/vgpu serve --gpu nvidia/h100 --count 4 --load 0.9 --alloc-mb 8192 &

build/bin/nvidia-smi              # nvidia-smi-style table
build/bin/rocm-smi                # ROCm concise-info table
build/bin/rocm_agent_enumerator   # gfx targets, one per line
build/bin/vgpu smi --lspci        # PCI listing
```

`vgpu serve` presents a rack without running kernels. Any workload running
against VirtualGPU publishes the same telemetry, so you can watch a real run:

```bash
LD_LIBRARY_PATH=build/shim ./my_cuda_app &
build/bin/nvidia-smi
```

## What is real and what is modelled

This distinction matters, because VirtualGPU does not model performance.

| Reading | Source |
| --- | --- |
| `memory.used` / `memory.total` | **REAL** — exact allocator state |
| `utilization.gpu` | **REAL** — fraction of wall time inside kernel launches |
| kernels launched, bytes moved | **REAL** — exact counters |
| power, temperature, voltage, clocks, fan, perf state | **SYNTHETIC** |

The synthetic columns are a deterministic first-order model driven by the real
utilization above: power tracks load quickly, temperature lags behind it with
thermal mass, both decay when idle, and clocks/voltage step with a boost curve.
They make monitoring tools and dashboards behave correctly. They are **not**
predictions of any physical device's power or thermal behavior. `vgpu smi
--explain` prints this at the terminal.

An idle virtual device reports idle values (ambient temperature, a small idle
draw, clocked down, P8) — not zeros, because a device that has done no work is
idle, not powered off.

## NVML

`build/shim/libnvidia-ml.so.1` implements the documented NVML API. Tools that
use it — pynvml, DCGM exporters, framework memory queries, monitoring agents —
work against VirtualGPU by putting the shim directory on `LD_LIBRARY_PATH`:

```bash
LD_LIBRARY_PATH=build/shim python -c "import pynvml; pynvml.nvmlInit(); ..."
```

Queries VirtualGPU cannot answer return `NVML_ERROR_NOT_SUPPORTED`, which tools
render as `N/A` — the honest result rather than an invented number. ECC state
is one such query: virtual memory has no ECC to report.

**The stock `nvidia-smi` binary will not run against this NVML.** It gates
startup on `nvmlInternalGetExportTable`, an undocumented internal vtable (the
NVML analogue of `cuGetExportTable`), and fabricating one would mean
nvidia-smi calling through entries whose contracts are unpublished. VirtualGPU
declines it and ships `build/bin/nvidia-smi` instead, which renders the same
telemetry. On a CPU-only machine there is no stock `nvidia-smi` anyway — nor
`rocm-smi` nor `rocm_agent_enumerator` — so supplying these commands is the
fix, not a workaround.

## lspci

`vgpu smi --lspci` prints the listing directly. `vgpu smi --lspci-dump`
synthesizes a PCI configuration space in `lspci -x` format, which the **real**
`lspci` renders:

```bash
build/bin/vgpu smi --lspci-dump > /tmp/vgpu-pci.dump
lspci -F /tmp/vgpu-pci.dump
```

```
01:00.0 3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)
02:00.0 3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)
```

The names come from the host's own `pci.ids`, resolved from the vendor/device
ids in the generated config space — so the device ids in the profiles are
correct, not decorative. Each virtual device gets its own bus slot
(`0000:01:00.0`, `0000:02:00.0`, …), class `0x030200` (3D controller).

## AMD

AMD profiles (`amd/mi300x`, `amd/mi325x`, `amd/mi350x`) are **discovery only**.
`rocm-smi`, `rocm_agent_enumerator`, and `lspci` report them correctly, and
`rocm_agent_enumerator` prints the right ISA targets (`gfx942` for CDNA3,
`gfx950` for CDNA4). Launching a kernel on one fails with a clear
"warp size 64 is unsupported" error — AMD *execution* is not implemented, and
VirtualGPU says so rather than producing wrong answers. See TODO.md.

## Where telemetry lives

A small file-backed shared segment, so observers in other processes can read
it: `$XDG_RUNTIME_DIR/vgpu-telemetry`, or `/tmp/vgpu-telemetry-<uid>`.
Override with `VGPU_TELEMETRY_PATH`; disable with `VGPU_TELEMETRY=0`. Readers
use a seqlock, so a snapshot is never half-updated, and a segment left behind
by a dead process reports no devices rather than phantom hardware.
