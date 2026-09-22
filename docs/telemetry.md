# Device discovery and telemetry

VirtualGPU presents its virtual devices to the tools people actually use to
look at GPUs: `nvidia-smi`, `rocm-smi`, `amd-smi`, `rocm_agent_enumerator`,
`lspci`, and anything speaking NVML.

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
and PCIe counters, rocm-smi's RAS blocks and amd-smi's ECC metrics all read
these counts. Nothing in the simulator produces them on its own: they come only
from injection. On an AMD card the session's dmesg carries amdgpu's lines
instead of Xids: `amdgpu 0000:01:00.0: amdgpu: 2 uncorrectable hardware errors
detected in umc block`.

`vgpu fault arm` goes one step further and puts faults in a running program's
path. The next device-memory loads of its kernels take them:

```bash
vgpu fault arm --bitflip --count 3         # three loads return a flipped bit
vgpu fault arm --ecc corrected --count 2   # counted; the data is intact
vgpu fault arm --ecc uncorrected           # counted, logged, and the kernel fails
```

A bit flip is silent -- nothing counts it, as nothing counts a fault ECC does
not cover -- which is what a memory test or a silent-data-corruption check
exists to catch. An uncorrectable error fails the kernel with
`cudaErrorECCUncorrectable` (214), which poisons the context as on a real card.

`--on` moves a fault to other accesses. `--on store` flips a bit of what a
kernel's store writes, so memory holds the wrong value and every later read
finds it, a copy back to the host included; ECC errors cannot be armed there,
since ECC is checked when memory is read. `--on shared` puts any of the three
on shared-memory loads: shared memory and L1 are one SRAM in the SM, so its
ECC errors count as the L1 cache's, and an uncorrectable one fails the kernel
without retiring a page or row of device memory.

```bash
vgpu fault arm --bitflip --on store --count 2
vgpu fault arm --ecc uncorrected --on shared
```

`--on alu` flips a bit of a floating-point, math or matrix (tensor core)
result: silent data corruption, which nothing counts because no ECC covers an
ALU, and which only a check of the results can find. Integer results are left
alone, since they are mostly addresses and loop counters, where a flip is a
crash or a hang rather than silent corruption. The bit is in the upper half of
the result, so the error is not a last-place difference inside a tolerance.
Whether a single flip is caught depends on the check, as on hardware: a later
pass that recomputes the value, or a clamp that resets it, masks it, and a
check of only the last pass misses a flip in an earlier one.

```bash
vgpu fault arm --bitflip --on alu --count 5
```

`--on copy` puts a fault on copies out of device memory -- to the host, to
another buffer or to another GPU. A corrected error is counted; an
uncorrectable one is counted, logged, and fails the copy with
`cudaErrorECCUncorrectable`, which poisons the context as a kernel's would; a
bit flip corrupts what that one copy delivers and leaves memory as it was.

Instead of a count, a fault can be armed at a rate: every access at the target
takes one with a given probability, until the rate is set back to 0 or the
driver reloads. Errors then grow with memory traffic, as a failing part's do,
which is what an error-rate threshold or an SBE-rate policy is tested against.
The generator is seeded, so a run takes the same number of faults every time.

```bash
vgpu fault arm --ecc corrected --rate 1e-6 --seed 7
vgpu fault arm --bitflip --on copy --rate 1e-4
vgpu fault arm --ecc corrected --rate 0      # stop
```

An armed fault strikes whichever access comes next. A stuck cell stays at one
address: a bit that reads as 0 or 1 whatever is written to it, as a failed cell
does, on every read -- a kernel's loads and atomics, and copies back to the
host. It is what an address-aware memory test looks for; Pantheon's galpat,
unmodified, reports the cell's element and the bit that differs.

```bash
vgpu fault stuck --offset 0x100 --bit 2 --value 1   # up to 16 cells per GPU
vgpu fault stuck --clear
```

The offset counts bytes from the start of the device's memory, where its first
allocation starts; allocations follow in order and are never moved or reused.
A driver reload (`vgpu fault reset --volatile`) does not mend a stuck cell.

Inside `vgpu shell`, these errors also reach `dmesg` the way the driver and the
kernel log them: an uncorrectable ECC error as Xid 48, the page or row it takes
out of service as Xid 63, and PCIe errors as AER lines. A kernel's own faults
are logged too, as the driver logs them: a read or write of an address no
allocation covers as Xid 31, an MMU fault naming the page and the access, and a
misaligned access, or a shared- or local-memory access out of range, as Xid 13,
an exception the SM raises. Faults only this simulator finds (a data race, an
uninitialized register) have no Xid.

The session's sysfs keeps the counts too, and they read live: in each PCI
device's directory (`/sys/bus/pci/devices/0000:01:00.0`, which
`/sys/class/drm/cardN/device` also leads to), `aer_dev_correctable`, `aer_dev_nonfatal` and
`aer_dev_fatal` in the kernel's AER stats form, and on an AMD card amdgpu's
`ras/umc_err_count`, `ras/gfx_err_count` and the other blocks' (`ue: N`,
`ce: N`). An injected PCIe counter lands on the AER bit it corresponds to: a
replay is a replay-timer Timeout, a replay rollover is Rollover, an LCRC or bad
TLP is BadTLP, an unspecified correctable error is RxErr, and unspecified
non-fatal and fatal errors are a completion timeout and a data link protocol
error, the default severities of each.

A GPU can also fall off the bus:

```bash
vgpu fault lose --gpu 1        # Xid 79
vgpu fault lose --gpu 1 --clear
```

NVML still counts it but answers `NVML_ERROR_GPU_IS_LOST` for its handle and
every query about it; nvidia-smi reports the other GPUs, prints "Unable to
determine the device handle for GPU...: GPU is lost" for it and exits 15; and a
program's next launch on it fails with `cudaErrorLaunchFailure` (719), what
programs report when it happens. A driver reload does not bring it back;
`--clear` does, as a reset would.

An AMD GPU is lost the way amdgpu loses one. The kernel log has the PCI core's
recovery and the driver's answers to it (drivers/pci/pcie/err.c and
amdgpu_pci_error_detected): the fatal AER error, a frozen channel, a slot reset
that fails, then the permanent failure, which amdgpu answers by letting the
device go:

```
pcieport 0000:00:01.0: AER: Uncorrected (Fatal) error received: 0000:01:00.0
amdgpu 0000:01:00.0: amdgpu: PCI error: detected callback!!
amdgpu 0000:01:00.0: amdgpu: pci_channel_io_frozen: state(2)!!
pcieport 0000:00:01.0: AER: subordinate device reset failed
amdgpu 0000:01:00.0: amdgpu: PCI error: detected callback!!
amdgpu 0000:01:00.0: amdgpu: pci_channel_io_perm_failure: state(3)!!
pcieport 0000:00:01.0: AER: device recovery failed
```

After that the driver has no device to report: rocm-smi, amd-smi and
rocm_agent_enumerator list the GPUs that are left, and the ones after it move
up. `vgpu` keeps every GPU's index, and the lost one's registers (`vgpu regs`,
configuration space and MMIO alike) read as all ones, as a device that no
longer answers on PCIe reads, and writes to them go nowhere. That is true of a
lost NVIDIA GPU's registers too. Not yet: the session's `/sys/class/drm` and
hwmon entries for the GPU stay in place.

A link can train below what card and slot support, as a bad riser, a dirty
contact or a marginal slot leaves it:

```bash
vgpu fault link --gpu 0 --width 4          # x4 of x8
vgpu fault link --gpu 1 --gen 1 --width 2
vgpu fault link --clear
```

The link's current generation and width read degraded -- nvidia-smi's
`pcie.link.gen.current` and `pcie.link.width.current`, `-q`'s GPU Link Info,
NVML's `nvmlDeviceGetCurrPcieLink*` -- beside its unchanged maximum. A driver
reload does not retrain it; `--clear` does.

Two more faults a health tool has to handle:

```bash
vgpu fault arm --hang --seconds 30      # the next launch stalls, then times out
vgpu fault arm --hang                   # ... or stalls until the process is stopped
vgpu fault throttle --reason sw_thermal_slowdown --seconds 60
vgpu fault throttle --clear
```

A hung launch shows the device fully busy, as a hung card does, and then fails
with `cudaErrorLaunchTimeout` and Xid 8 -- or never returns, which is what a
watchdog exists to catch. A throttle makes clock-event reasons active, and every
reading agrees with them: nvidia-smi's and NVML's reasons and their time
counters, a thermal slowdown's temperature at the slowdown threshold, a power
cap's draw at the limit, and a slowdown's SM clock pulled down.

Every ECC error and Xid is also an NVML event, so a health daemon blocked in
`nvmlEventSetWait` wakes up the way it does on a real machine, whichever
process injected or hit the fault. A card offers the events it could raise:
Xid on all of them, single- and double-bit ECC only where the card has ECC
(`nvmlDeviceGetSupportedEventTypes`), and registering for any other type is
`NVML_ERROR_NOT_SUPPORTED`. A waiter sees what happens after it registers, and
the last 32 events per GPU are kept for waiters that fall behind; a driver
reload (`vgpu fault reset --volatile`) clears them with the other volatile
state. One `vgpu fault inject --count N` raises one ECC event, not N.

## lspci

`vgpu smi --lspci` prints the listing directly. `vgpu smi --lspci-dump`
writes each device's configuration space, from the register model (see
[registers.md](registers.md)), in `lspci -xxxx` format, which the **real**
`lspci` renders -- with `-vvv`, capabilities, link state and AER status too:

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

`amd-smi` answers the commands health tools run: `list`, `metric -e` (ECC
totals) and `-k` (ECC counts per RAS block, device memory as UMC and the
on-chip memories as GFX), and `ras --cper`, which lists recent ECC errors as
CPER records in amd-smi's text table -- a table even under `--json`, as the
real tool prints it. Other metrics, `--csv` and CPER record files (`--folder`)
are refused as not modelled. JSON output is amd-smi's: a list with one object
per GPU, keyed by `gpu`, indented by four.

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
