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
| memory temperature | **SYNTHETIC**, and only on profiles whose real cards report a sensor |
| ECC mode, retired pages or remapped rows | **PROFILE** — which the card has; every count is zero, because nothing faults in simulated memory |
| PCIe link generation and width | **PROFILE** — the link real cards of the model most often run at |
| PCIe replay and error counters | zero — a simulated link has no transport errors |
| clock-event reasons | GPU idle only — nothing slows a simulated clock |

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
render as `N/A` — the honest result rather than an invented number.

## Reliability and link in profiles

Four optional `telemetry` keys describe what a card reports about memory
reliability and its PCIe link:

| Key | Meaning | Source |
| --- | --- | --- |
| `ecc` | the card has ECC and ships with it on | datasheet |
| `memory` | `hbm` or `gddr`: HBM cards remap failing rows, GDDR cards with ECC retire pages | datasheet |
| `memory_temperature` | the driver reports a memory sensor | real runs; most HBM cards report none |
| `pcie_link` | the link real cards most often run at, such as `"Gen4 x8"` | the benchmark database of real runs |

The link is a fact about how cards are hosted, not the slot they fit: a T4 is
an x16 card that clouds attach at x8, and a GH200's Hopper die reaches its
Grace CPU over NVLink-C2C with an x1 PCIe link. From these, nvidia-smi and NVML
answer ECC mode, the ECC counters, retired pages or remapped rows, the link and
the memory sensor the way a real card of that model does. A card without ECC
still answers the SRAM breakdown (`ecc.errors.uncorrected.*.sram.*`) with 0, as
a real RTX 3060 does.

**The stock `nvidia-smi` binary will not run against this NVML.** It gates
startup on `nvmlInternalGetExportTable`, an undocumented internal vtable (the
NVML analogue of `cuGetExportTable`), and fabricating one would mean
nvidia-smi calling through entries whose contracts are unpublished. VirtualGPU
declines it and ships `build/bin/nvidia-smi` instead, which renders the same
telemetry. On a CPU-only machine there is no stock `nvidia-smi` anyway — nor
`rocm-smi` nor `rocm_agent_enumerator` — so supplying these commands is the
fix, not a workaround.

## Injecting faults

A health tool is tested against the errors it exists to catch, and on real
hardware an uncorrectable ECC error cannot be provoked on demand. Here it is one
command, and every surface a tool reads reports it:

```bash
vgpu fault inject --gpu 0 --ecc uncorrected                 # one, in device memory
vgpu fault inject --ecc corrected --location l2_cache --count 5
vgpu fault inject --pcie replay --count 3
vgpu fault show
vgpu fault reset --volatile        # a driver reload
nvidia-smi -i 0 -p 1               # zero the aggregate ECC counts
```

The counts live in files, not in any process, because a real card's counts do
not reset when the program that caused them exits. Volatile counts sit with the
machine's telemetry, so a new `vgpu shell` session starts at zero; aggregate
counts sit in `VGPU_STATE_DIR` (default `~/.local/state/vgpu`), keyed by the
device UUID. An uncorrectable error in device memory also retires a page on a
GDDR card or remaps a row on an HBM card, pending until the next driver load. A
card without ECC refuses ECC injection: it has nowhere to count one.

nvidia-smi's ECC, retired-page and remapped-row fields, its table, NVML's ECC
and PCIe counters, and rocm-smi's RAS blocks all read these counts. Nothing
in the simulator produces them on its own yet: they come only from injection.

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
`gfx950` for CDNA4). `rocm-smi` takes its documented `-a`, `--showid`,
`--showproductname`, `--showmeminfo vram`, `--showtemp`, `--showpower`,
`--showuse`, `--showvbios`, `--showclocks`, `--showfan`, `--showmaxpower`,
`--showserial`, `--showuniqueid`, `--showmemvendor`, `--showrasinfo`, `-d N`,
`--json` and `--csv`, the same inside `vgpu shell` and out. The key names of
the newer sections and the AMD profiles' ECC, memory sensor and link values
follow AMD's documentation and have not yet been checked against a real
MI-series card.
Outside a session both tools describe `VGPU_GPU`; with no AMD GPU configured,
`rocm_agent_enumerator` lists only `gfx000` and says on stderr how to pick one. Launching a kernel on one fails with a clear
"warp size 64 is unsupported" error — AMD *execution* is not implemented, and
VirtualGPU says so rather than producing wrong answers. See TODO.md.

## Where telemetry lives

A small file-backed shared segment, so observers in other processes can read
it: `$XDG_RUNTIME_DIR/vgpu-telemetry`, or `/tmp/vgpu-telemetry-<uid>`.
Override with `VGPU_TELEMETRY_PATH`; disable with `VGPU_TELEMETRY=0`. Readers
use a seqlock, so a snapshot is never half-updated, and a segment left behind
by a dead process reports no devices rather than phantom hardware.
