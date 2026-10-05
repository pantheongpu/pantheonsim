# NVML

`libnvidia-ml.so.1` (`build/shim/`) exports every function CUDA's `nvml.h`
declares -- 408 in CUDA 13.2's, 307 in the 12.0 one Ubuntu ships -- and a test
(`nvml_coverage`) extracts them from the header the shim was built against and
checks each is exported, so a tool that looks entry points up by name (pynvml
raises `FunctionNotFound`; nvitop does not catch it) never finds one missing.

An entry point answers in this order of preference:

1. **A real answer**, from what the simulator has: the profile, the telemetry
   sample, the reliability state (`vgpu fault`), the host (CPU and NUMA
   affinity, as `vgpu smi topo` reads them).
2. **The status the real driver gives on that kind of GPU**, usually
   `NVML_ERROR_NOT_SUPPORTED`, which every tool renders as N/A. Never a plausible
   wrong number.
3. **Setters** (`Set*`, `Clear*`, `Reset*`, `Create*`, `Destroy*`, `Modify*`,
   `Remove*`, `Discover*`): described below.

A versioned structure with a wrong version word is
`NVML_ERROR_ARGUMENT_VERSION_MISMATCH` (the MIG getters that the card answers
with `INVALID_ARGUMENT` instead keep that answer), and both spellings NVIDIA
exports (`nvmlFoo`, `nvmlFoo_v2`) are exported.

## What the card says

The return codes (and, where they are fixed facts of the card, the values) were
read from an RTX 3060 through NVIDIA's own NVML 13.0 -- GPU index 1 of this
project's machine, the WSL driver 596.36 -- by a probe that calls every getter
(`nvidia/tools/nvml_card_probe_gen.py` generates it from an `nvml.h`). Nothing that
writes was called. The result is checked in as
`nvidia/tests/data/nvml_rtx3060_matrix.tsv`: 236 getters, their return codes on
the card and on the simulator, the fixed-fact values, and a reason for each of
the 13 places the simulator differs. `nvml_matrix` runs the same probe against the
shim and compares; with `VGPU_NVML_CARD=<libnvidia-ml.so.1>` the same script checks
the card column against NVIDIA's own library on a machine that has one.

The differences are: WSL answers (no `/dev/nvidia`, no CPU or NUMA affinity,
the WDDM driver model) where a Linux driver answers the other way; and
BAR1 usage, PCIe throughput, energy and the VBIOS version, which the simulator
has no data for. The matrix is of a GeForce card: datacenter profiles (T4, A10,
A100, H100, ...) have no card behind them and follow `nvml.h`'s documentation
and the datasheets, as each source comment says.

Functions first declared in CUDA 13.2 (and 13.0, 12.9, ... for the ones the card
predates) cannot be checked against NVIDIA's library 13.0; their statuses come
from `nvml.h`.

## Profile facts

Each profile has an `nvml:` section (`NvmlClass` in `include/vgpu/profile.hpp`):
memory bus width, brand, fans, NVLink ports and generation, C2C links, MIG, the
media engines, and for the RTX 3060 the clock and performance-state tables, the
overclocking offset ranges, the fan range and the inforom versions. Each line of
a yaml says where its value came from: a datasheet (the bus widths agree with the
datasheets' memory bandwidths at the profile's memory clock) or the card. A
profile that records no table answers the entry point that needs it
`NOT_SUPPORTED`; the clock lists, the fans and the inforom exist for the 3060
only.

## What is real

| Group | Entry points | Answer |
| --- | --- | --- |
| Identity | brand, bus type, board id, module id, PDI, memory bus width, PCI info with class, handle by versioned UUID | profile, `describe_device` |
| Clocks | clock by id, supported memory and graphics clocks, P-state ranges, current clock frequencies, performance modes, overclocking offsets, clock-event reasons, violation times | profile tables (RTX 3060); the reasons and violation times are those injected with `vgpu fault throttle` |
| Fans, thermals, power | fan count, speed, target, range, policy, RPM, cooler, versioned temperature, margin, thermal settings, dynamic P-states, power source and state, PowerMizer | profile; a running fan's RPM is N/A (no tachometer curve) |
| PCIe | speed, link max speed, GPU max generation, replay counter | the registers and the reliability state |
| Affinity | CPU and memory affinity, NUMA node, topology nearest GPUs and GPU set, common ancestor | the host; every pair of GPUs shares a host bridge (`PHB`) |
| ECC and memory | default mode, detailed errors, SRAM status, retired pages, pending retirement, remapped rows, repair status, unrepairable flag | `vgpu fault` state; page addresses and times are named when NVML first sees a page (see below) |
| Processes | the three running-process lists (v1 layout), process details, accounting (mode, pids, stats, buffer size) | the telemetry's process list |
| Engines | encoder, decoder, JPEG, OFA utilization, encoder capacity/stats/sessions, FBC | zero (nothing encodes), for GPUs that have the engines |
| Samples | `nvmlDeviceGetSamples` | one sample, the current reading |
| NVLink | state, version, capability, remote device type, fabric info, P2P status | ports from the profile, none connected |
| System events | create, register, wait, free | nothing binds or unbinds a simulated GPU: waits time out |
| MIG | MIG mode, the instance getters with MIG off | see below |

Accounting: while it is on, each NVML call about accounting folds the
telemetry's process list into a record file (a process seen for the first
time starts a record; one that has gone is finished). A process that starts and
ends between two observations is never seen, and utilization is not attributed
to processes, so it reads `NVML_VALUE_NOT_AVAILABLE`.

Retired pages: the reliability state counts them by cause; their addresses and
times are made up once, when NVML first sees the count grow (an address from the
device's UUID, 4 KiB aligned inside its memory; the time of the sighting), and kept.

## Not supported, by design

| Group | Why |
| --- | --- |
| vGPU (`nvmlVgpu*`, `nvmlGetVgpu*`, the scheduler, placements, heterogeneous mode) | no vGPU manager on a bare-metal machine; instance and type ids are invalid |
| S-class units (`nvmlUnit*`, `nvmlSystemGetHicVersion`) | none attached: the count is 0 |
| GPM (`nvmlGpm*`) | the simulator is functional and has no counters to report; `nvmlGpmSampleAlloc/Free` and the metric checks work |
| Confidential computing, PRM, power smoothing, workload power profiles | no such hardware or firmware |
| MIG partitioning | MIG mode reads back (disabled) on MIG-capable profiles and enabling it is `NOT_SUPPORTED`; no instance can be created, so instance handles are invalid |
| NVLink traffic, errors, remote end, bandwidth mode | the ports exist, none is cabled; no fabric manager |
| Application clocks, auto boost, API restriction | deprecated and absent on the card |
| ECC, retired pages, GSP, hostname, DRAM encryption, driver model | absent on the card, or Windows only |
| BAR1 usage, PCIe throughput, energy | need a window or an integral the telemetry does not keep |
| Clock tables, fans and inforom for profiles other than the RTX 3060 | no card was read; see Profile facts |

## Setters

The real driver demands root for every setter. Here a setter takes effect when
the process is root, or when `VGPU_NVML_ROOT=1` says it is (a simulated machine is
a user's, and so are the tests); `VGPU_NVML_ROOT=0` refuses even root. Without
permission the call is `NVML_ERROR_NO_PERMISSION`. The checks run in this order:
the handle, the arguments (`INVALID_ARGUMENT`), whether the GPU has the feature
(`NOT_SUPPORTED`), permission, then the change. nvml.h documents the codes, not
which wins; no setter was called on the card.

What a setter changes is written to a small file per device, keyed by the
device's UUID, so a later process (nvidia-smi, a monitoring agent) sees it, and the
getters read it back:

- Settings the real driver forgets at a driver reload live in the session's
  runtime directory (`telemetry::default_path`, so a new session starts clean):
  persistence mode, compute mode, power limit, clock locks (which hold
  `nvmlDeviceGetClockInfo` inside the range) and offsets, fan policy and speed,
  PowerMizer, accounting mode and records, drain state and hot removal.
- Settings the driver keeps (the pending ECC mode, retired-page records) live in
  the state directory (`VGPU_STATE_DIR`, else `~/.local/state/vgpu`).
- The ECC counters are the reliability state's: `nvmlDeviceClearEccErrorCounts`
  zeroes the volatile or the aggregate counts. A changed ECC mode shows as
  pending; a reboot or GPU reset is not simulated.
- `nvmlDeviceRemoveGpu` takes a GPU out of every query (it is not counted or
  found) until `nvmlDeviceDiscoverGpus` brings it back; a GPU with processes on
  it is `NVML_ERROR_IN_USE`.

## Header tiers

CI builds the shim against four `nvml.h` versions (CUDA 12.0, 12.8, 13.0, 13.2),
and the later ones declare functions and types the earlier ones lack. Each such
function is inside a tier macro (`VGPU_NVML_12_4`, `12_8`, `12_9`, `13_0`,
`13_2` in `nvml_common.inc`), defined when a macro that header introduced exists.
The six headers from 12.0 to 13.2 were each checked to declare nothing the shim
leaves out. The files: `nvml_device.inc` (identity, clocks, fans, power, PCIe,
affinity), `nvml_ras.inc` (ECC, retirement, accounting, engines, samples),
`nvml_link.inc` (NVLink, topology, events), `nvml_mig.inc`, `nvml_vgpu.inc`
(vGPU, units, GPM, confidential computing) and `nvml_set.inc` (setters).

## Tests

`nvml_coverage`, `nvml_matrix`, and one test per category, each run per profile
by `run_nvml_category.sh`: `nvml_device`, `nvml_ras`, `nvml_link`, `nvml_setters`,
`nvml_mig`, `nvml_vgpu` (and the older `nvml_api`).
