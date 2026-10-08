# Registers

A VirtualGPU device has register spaces the way a card does, each backed by a
register database: every register is declared once, with its offset, width,
access and what backs its value, and every tool reads the same state through
it. Two kinds of space: **PCI configuration space**, the same layout on every
vendor's card, and each vendor's **MMIO registers** -- an AMD GPU's behind BAR5,
from the Linux amdgpu headers, and an NVIDIA GPU's behind BAR0, from NVIDIA's
own published register headers.

```bash
vgpu regs list                          # the database: offset, width, access, backing
vgpu regs read link_status              # read and decode a register
vgpu regs write command 0x0002          # write it, with its semantics
vgpu regs dump --extended               # all 4096 bytes
vgpu regs log                           # who read and wrote what
```

`--gpu N` picks the device (0 by default); `read` and `write` take a register's
name or its offset (`0x08a`), and `--size 1|2|4` for a partial access.

## Where it lives

What every PCI device shares is in one place, and what only one vendor's cards
have is in that vendor's directory:

| | Shared | AMD | NVIDIA |
| --- | --- | --- | --- |
| Register databases | `registers/pci-config.yaml` | `amd/registers/mmio.yaml` | `nvidia/registers/mmio.yaml` |
| Each GPU model's registers | | `amd/registers/gpus/` | `nvidia/registers/gpus/` |
| Captured from real cards | | | `nvidia/registers/measurements/` |
| Code | `src/core/regs.cpp`: the databases' semantics, configuration space and its capability chains, BAR sizing, AER status, the shared state and log, the model files | `amd/src/regs.cpp`: BARs, identity, engine status, the SMU mailbox, NBIO, amdgpu's sysfs files; `metrics.cpp`, `cper.cpp` | `nvidia/src/regs.cpp`: BARs, identity, the replayed configuration space, BAR0's undeclared answer, the GeForce idle link |

The engine reaches a vendor's part through one table of hooks
(`include/vgpu/regs_vendor.hpp`): its MMIO space, BAR layout, captured
configuration space, current link generation, the backings only it has, the
writes its device logic answers, and its driver files in sysfs.

## The database

`registers/pci-config.yaml` declares each register:

```yaml
link_status:
  offset: "0x08a"
  width: 16
  access: ro
  backing: link.status
  fields: ["3:0 current_link_speed", "9:4 negotiated_link_width", "12 slot_clock"]
  surfaces: [lspci, nvml.pcie.current, smi.pcie.link.current]
  status: done
```

| Key | Meaning |
| --- | --- |
| `offset`, `width` | where it is, in bytes and bits (8, 16, 24 or 32) |
| `access` | `ro`; `rw` (writes kept, only `write_mask` bits change); `rw1c` (writing 1 clears a status bit until its cause recurs); `bar` (a base address register) |
| `reset` | the value after reset, when nothing backs the register |
| `backing` | where the value comes from: `profile.*` (identity and class), `bar.N`, `link.*` (the PCIe link, degraded or not), `aer.*` and `devsta` (the reliability counts) |
| `fields`, `surfaces` | bit fields, and the tools and files that read the register |
| `status` | `done` behaves as on hardware; `model` is a plausible value, not measured on a card |

The database is embedded at build time. A value is never a guess: a register
either has a reset value or a backing, and a backing the engine does not know
is an error, not a zero.

## Each GPU's registers

The database says what a register is; `<vendor>/registers/gpus/<model>.yaml`
says what it holds on one GPU model. There is a file for every profile, listing
every register of each space the model has, at the offset the model has it
(a card whose capability chain is laid out differently has its registers where
its chain puts them), with the value it reads as at power-on:

```yaml
profile: amd/mi300x
pci_device_id: 0x74a1
layout: generic

config:
  device_id:                   [0x002, 16, ro, 0x74a1]
  link_status:                 [0x08a, 16, ro, 0x1105, link.status]
```

A register with a fifth element follows the device's state -- the link, the
BARs, error status, the engines, the SMU mailbox -- and its value is the one it
has at power-on. Every other value is fixed for the model. The files are
embedded at build time, and every simulated GPU of the model, in every process,
starts from them: two runs of the same GPU read the same registers.

The files are generated, not written by hand:

```bash
vgpu regs export amd/mi300x                 # one model's file, to stdout
vgpu regs export --out .                    # every profile's, at the repository root
```

Change a register in the database or a profile, then regenerate. `regs_cli`
fails when a committed file differs from what `export` writes, and the unit
tests fail when a profile has no file, two files claim one PCI device ID, or a
file disagrees with the database (a register missing, unknown, at another
offset, with another width or access, or fixed where the database has it
live). A stale file is refused with the command that regenerates it, never
half applied.

To try other values without a rebuild, point `VGPU_REGISTERS_DIR` at a copy of
the directory and edit it. The simulator reads that directory in place of the
embedded files:

```bash
cp -r amd/registers/gpus /tmp/gpus
$EDITOR /tmp/gpus/mi300x.yaml
VGPU_REGISTERS_DIR=/tmp/gpus VGPU_GPU=amd/mi300x vgpu regs read subsystem_id
```

## Configuration space

The type-0 header; capabilities for power management (0x60), MSI (0x68) and
PCI Express (0x78); and in the extended space, Advanced Error Reporting at
0x100. What is live:

- **Identity and class**: vendor and device IDs from the profile; the class
  code is a 3D controller for NVIDIA data-center cards, a VGA controller for a
  GeForce, and a processing accelerator for AMD Instinct cards.
- **Link**: Link Capabilities carries the profile's link, Link Status the link
  as trained now, so `vgpu fault link` shows as a downgraded link.
- **Errors**: AER's correctable and uncorrectable status and the PCIe Device
  Status error bits are set by the PCIe errors `vgpu fault inject --pcie`
  counts, and writing 1 clears them until another arrives. The injected counter
  lands on the AER bit it corresponds to, as in the session's AER stats files
  (see [telemetry.md](telemetry.md)).
- **BARs**: a sizing probe (write all ones, read back) returns each BAR's size.
  NVIDIA: BAR0 16 MiB of registers; BAR1 the framebuffer aperture, 64-bit and
  prefetchable (256 MiB on a GeForce, the framebuffer rounded up to a power of
  two on data-center cards); BAR3 32 MiB; an I/O BAR on a GeForce. AMD: BAR0 the
  framebuffer aperture, BAR2 the doorbells, BAR5 the registers. Sizes and
  addresses are a model.
- **Read-write registers** (command, MSI, device and link control, the AER
  masks and severities) keep what is written.

What is written is kept in a state file beside the reliability state, shared by
every process on the machine: one process's write is the next one's read. A
database that changes starts the state again.

## AMD MMIO

`--space mmio` reaches an AMD Instinct GPU's registers behind BAR5, as the
driver and umr do: 32-bit accesses at byte addresses in the BAR.

```bash
vgpu regs list --space mmio
vgpu regs read --space mmio grbm_status
```

| Register | Offset | What it does |
| --- | --- | --- |
| `grbm_status`, `grbm_status2` | 0x08010, 0x08008 | graphics engine status: GUI_ACTIVE and the busy units while the GPU works; FIFOs available and DB and CB clean when idle |
| `cp_stat`, `rlc_stat` | 0x08680, 0x3b010 | the command processor's and RLC's busy bits |
| `smu_message`, `smu_argument`, `smu_response` | 0x58a08, 0x58a48, 0x58a68 | the SMU mailbox (MP1 C2PMSG_66, _82, _90) |
| `nbio_strap0` | 0x034d8 | the strap amdgpu reads the revision from: the device ID, the revision, the function enabled |
| `nbio_config_memsize` | 0x0378c | the VRAM in MiB, which amdgpu sizes memory by; it follows the device, `VGPU_VRAM_MB` included |
| `nbio_ep_pcie_lc_speed_cntl` | 0x03564 | the link controller's speed straps (Gen2 to Gen5 enabled), up to the profile's highest PCIe generation; model |
| `nbio_partition_compute_status` | 0x03a0c | the compute partition mode: SPX |
| `nbio_partition_mem_status`, `nbio_partition_mem_cap` | 0x03a10, 0x03a08 | the memory partition mode, NPS1, and the modes the GPU supports: NPS1 and NPS4 on MI300X and MI325X, NPS1 and NPS2 on MI350X |

The mailbox works as the driver drives it: clear the response, write the
argument, write the message, and the SMU answers -- `1` in the response
register, and its reply in the argument register. It answers TestMessage (the
argument plus one), GetSmuVersion, GetDriverIfVersion and GetMetricsVersion,
and answers the result codes of `smu_v13_0_6_ppsmc.h`: a message the header
defines but the model has no handler for fails (`0xff`), and a number the header
does not define is an unknown command (`0xfe`). The model never answers `0xfc`
(busy) or `0xfd` (prerequisite rejected): it answers at once, and the header
publishes no condition for either. The codes are the header's; which message
gets which is a model:

```bash
vgpu regs write --space mmio smu_response 0
vgpu regs write --space mmio smu_argument 0
vgpu regs write --space mmio smu_message 0x2        # GetSmuVersion
vgpu regs read --space mmio smu_argument            # 0x00556f00: 85.111.0
```

The SMU's metrics reach tools as a table, not registers: `gpu_metrics` in the
device's sysfs directory, the binary struct amd-smi and rocm-smi read. Inside
`vgpu shell` an AMD GPU publishes it in the driver's `gpu_metrics_v1_5` layout
(360 bytes, format 1, content 5), live: hotspot and memory temperature, socket
power, GFX and memory activity, the link's width and speed, the PCIe replay,
rollover, recovery and NAK counts `vgpu fault inject --pcie` adds, and the
graphics and memory clocks. What the simulator has nothing for -- energy,
XGMI, video engines -- reads as all ones, as the driver leaves fields a GPU does
not report. Later kernels publish later versions of the table.

The partition modes reach tools as amdgpu's sysfs files, which the session
writes from the NBIO registers the way the driver reads them:
`current_compute_partition` (`SPX`), `current_memory_partition` (`NPS1`) and
`available_memory_partition` (`NPS1, NPS4` on MI300X). CDNA3's pair of
supported modes is the one amdgpu itself assumes for GC 9.4.3 and 9.4.4 when
it cannot read the register (gmc_v9_0.c); MI350X's is AMD's documented pair.
Switching partitions is not modelled, so `available_compute_partition` is not
published: which compute modes are offered depends on the memory mode and the
XCC count in ways not checked yet.

The register offsets, bit fields and message numbers come from the amdgpu
headers (`gc_9_4_3_*.h`, `mp_13_0_6_offset.h`, `nbio_7_9_0_*.h`, `smu_v13_0_6_ppsmc.h`, and for
UMC and THM `umc_6_7_0_*.h` and `thm_13_0_2_*.h`; their
MIT notices are in `amd/registers/LICENSES/amdgpu-headers.txt`), and each entry's
`source` names the symbol. They are offsets from an IP block's base, and MI300
learns its bases at boot from its IP discovery table; the database uses
Aldebaran's (MI200's), the same GFX9 family. That, the engine status values and
the firmware version are a model until checked on a card.

### UMC ECC and thermal

Behind the same BAR5 are the memory controller's ECC registers and the
thermal sensor, from `umc_6_7_0_*.h` (Aldebaran's UMC) and `thm_13_0_2_*.h`,
each entry naming its header symbol in `source`.

| Registers | Offset | Behaviour |
| --- | --- | --- |
| `umc0_ch{0-3}_ecc_ctrl`, `_ecc_err_cnt_sel` | (0x14000 + regUMCCH*n*_0_EccCtrl / EccErrCntSel) * 4 | rw, only the header's fields stick; power-on value is a model (write and read ECC on, count enabled) |
| `umc0_ch{0-3}_ecc_err_cnt` | (0x14000 + regUMCCH*n*_0_EccErrCnt) * 4 | ro, 16 bits: the injected correctable device-memory errors since load (`vgpu fault`), shared over the four channels in turn |
| `umc0_mca_status_lo`, `_hi` | (0x14000 + regMCA_UMC_UMC0_MCUMC_STATUST0) * 4 | ro, the 64-bit MCA status as two dwords: Val and En with UECC and UC after an injected uncorrected error, else Val and En with CECC after a corrected one, else 0 |
| `thm_tcon_cur_tmp` | (0x16600 + regTHM_TCON_CUR_TMP) * 4 | ro, `CUR_TEMP` holds the GPU temperature in 0.125 C steps |

Status of these entries: the UMC instance 0 segment 0 base (0x14000) and THM
segment 0 base (0x16600) are Aldebaran's, from `aldebaran_ip_offset.h`, and are
assumed for MI300, not checked on a card; every entry says "not measured".
Channels 4-7 (segment 1, 0x54000) and UMC instances 1-3 (0x94000 and up) lie
beyond the 512 KiB BAR5 model and are reached on a card through the indirect
index and data registers, which are not mapped, so they are not in the map. MI300 itself uses `umc_12_0_0` (its own MCA-based layout); that
header carries a differently worded notice and is not used here. The mapping
of the card's error count to channels, the power-on control values, the status
bit pattern and the temperature step are models, not measured; the per-channel
uncorrectable count and MCA address, syndrome and IPID registers are not mapped.
`umc_6_7_0` publishes no per-channel uncorrectable count (`EccErrCnt` counts
correctable errors; uncorrectable ones show in the MCA status's UECC bit, which
is per instance), so none is mapped. The PCIe link's trained speed and width
(`regBIF_CFG_DEV0_EPF0_0_LINK_STATUS`) are in segment index 8, which Aldebaran's
NBIO base table does not have; they stay in configuration space, and only the
link controller's speed straps (segment 2) are mapped in BAR5.

## From C

Bring-up software reaches the same registers through `vgpu_regs.h` and
`libvgpuregs.so` (in the build directory), with the same semantics and the same
access log as `vgpu regs`:

```c
#include "vgpu_regs.h"

vgpu_regs* mmio;
uint32_t off, resp;
vgpu_regs_open(0, "mmio", &mmio);                     /* or "config" */
vgpu_regs_find("mmio", "smu_message", &off, NULL);
vgpu_regs_write(mmio, off, 4, 0x2);                   /* GetSmuVersion */
vgpu_regs_find("mmio", "smu_response", &off, NULL);
vgpu_regs_read(mmio, off, 4, &resp);                  /* 1: answered */
vgpu_regs_close(mmio);
```

Each call returns `VGPU_REGS_OK` or a negative code -- `EINVAL` for a bad
access, `ENODEV` for no such GPU, `ENOTSUP` for a space the GPU does not have,
`ENOENT` for an unknown name -- and `vgpu_regs_last_error()` says what went
wrong. A read reflects the device at the moment of the read, so polling engine
status sees it change. Build against it with `-Iinclude -Lbuild -lvgpuregs`;
`tests/c_harness/regs_capi.c` is a complete example.

## Measured on real hardware

`tools/regprobe/regprobe.py` runs on a machine with a real GPU and captures what
the model is checked against. Every step is read-only.

```bash
python3 tools/regprobe/regprobe.py capture out/                 # config space, sysfs, lspci, nvidia-smi -q
sudo python3 tools/regprobe/regprobe.py mmio out/ --bdf 0000:0a:00.0 --range 0x0-0x1000
sudo python3 tools/regprobe/regprobe.py correlate out/ --bdf 0000:0a:00.0 \
    --range 0x0-0x1000 --seconds 120 --load "./a-gpu-burn"       # what follows temperature, clocks, power
python3 tools/regprobe/regprobe.py compare out/ --profile nvidia/rtx3060 --vgpu build/vgpu
```

`capture` reads all 4096 bytes of configuration space as root, and the 64-byte
header the kernel gives anyone otherwise. `mmio` reads a BAR's registers, each
value written to disk as it is read, so a read that hangs the card keeps
everything before it. `correlate` samples registers idle and then under load
beside nvidia-smi's readings, and ranks the offsets whose values follow them --
candidates for what each register is. A sweep of an undocumented region can
hang a GPU or its host: sweep a card no one else is using, starting small.

What a real card read is kept in `<vendor>/registers/measurements/`, and each register
it changed carries a `measured:` note that `vgpu regs read` prints.

A card whose whole configuration space was captured is replayed for the
profiles of its family. So far: an RTX 3080 Ti (GA102), read as root, replayed
for Ampere GeForce profiles. The database's registers are found through that
card's capability chain, as a driver finds them -- its AER is at 0x420, not the
generic layout's 0x100, so `aer_correctable_status` is at 0x430 -- and keep
their live behaviour; every byte the database does not declare is the captured
card's: its thirteen capabilities, from Virtual Channel and L1 substates to
Resizable BAR and lane margining. `lspci -vvv` of the simulated RTX 3060 matches
the real card's but for the IDs and the BAR addresses, and the
`nvidia/rtx3080ti` profile -- characterized from the same card, verified on
512 conformance values -- matches it byte for byte but for the BAR addresses its
host's firmware chose. Other cards use the generic layout.

An NVIDIA GPU whose id NVIDIA has not published (`nvidia/vr200`, `nvidia/thor`)
carries a placeholder id from 0xFE00 up, above every id in NVIDIA's open kernel
modules name table; `vgpu regs export` says so in the file's header. See
`nvidia/docs/profiles.md`.

`--space mmio` on an NVIDIA GPU is its BAR0 (`nvidia/registers/mmio.yaml`).
Every offset and bit field there comes from the published register headers of
NVIDIA's open GPU kernel modules, which carry an MIT license
(`nvidia/registers/LICENSES/nvidia-open-gpu-kernel-modules.txt` names the files,
and each entry's `source` names the symbol). The values are this project's:

- `pmc_boot_0` at 0x0 is the card's identity, in the header's own fields. The
  architecture is the published id for the profile's architecture -- Turing
  0x16 through Blackwell 0x1a -- so software that reads a card's family off
  BAR0 gets the right answer on every profile. The die (`implementation`) is
  filled in only for the card that was read, which answered 2, GA102; every
  other model reports 0 rather than a guess. On the measured card the whole
  register reads `0xb72000a1`, which is what that card gave, and its low byte is
  the revision configuration space reports.
- `pmc_boot_1` at 0x4 says whether software is on a card or on a virtual
  function of one. It reads 0, a real card, which is what this simulator
  presents -- and what the measured card answered.
- `pmc_enable` at 0x200 is which engines a bound driver leaves running: the host
  and graphics engines, two copy engines, the power and security
  microcontrollers, the performance monitor, and the display engine only where
  there are display outputs. A model, marked as one. The video bits are
  declared because software decodes them and read 0: which decoders and
  encoders a card has differs by model and the profiles do not record it.
- The VBIOS scratch words at 0x1400 and the BAR1 and BAR2 block registers keep
  what is written, which is their whole contract.
- `ptimer_vf_timer_0` at 0x9800 advances with the clock the engine keeps time
  by, so software polling it sees time pass. It measures simulation, not a
  device, and the model's own file records where it starts.

Registers the map does not declare answer `0xbadf5040`, which is what the
measured card answered from 0xc on.

### Registers an architecture's headers define

Past that common core, the map holds registers that only some architectures'
headers publish. Each carries `arch:` in `nvidia/registers/mmio.yaml` -- one or
more of `turing`, `ampere`, `ada`, `hopper`, `blackwell`, or the Blackwell dies
`gb100` (B200, B300; compute capability 10.x) and `gb20x` (the GeForce RTX 50
series; 12.x) -- and a GPU has the register only if its architecture is named.
On any other GPU the offset is unmapped and reads `0xbadf5040`: no published
header says what is there, so any value would be invented. The same offset can
mean different registers on different architectures (the memory ECC counters
move between Turing and Hopper), and a name with `_gh100` or `_gb20x` is the
other architecture's copy. `vgpu regs list` shows the `arch` of each register;
`vgpu regs read` on a GPU that lacks one says so; the `vgpu regs dump` and
export files list only what the GPU has.

The headers are a sample of each chip's registers, not a complete map, and the
tree publishes different blocks for different chips. What is mapped, by block:

| Block | Registers | Architectures | Behaviour |
|---|---|---|---|
| PMC interrupts | `pmc_intr_1`, `pmc_intr_en_1`, `pmc_intr_en_set_{0,1}`, `pmc_intr_en_clear_{0,1}` | Turing | The header makes `NV_PMC_INTR_EN` read-only and changes it through SET and CLEAR; that is modelled (SET/CLEAR read zero, a direct write to the enable is ignored on Turing and kept on the others). No interrupt is ever raised. Model. |
| PMC device enable | `pmc_device_enable_0` | Ampere | Header reset (all disabled), kept as written. Model: it is not tied to `pmc_enable`, because the device-to-bit assignment comes from the topology table. |
| PMC confidential computing | `pmc_zb_scratch_reset_2_{0..15}` | GB100 | 16 scratch words; word 4 holds the CC mode bits (named from the header's addendum). All zero: the profiles record no CC mode. |
| PBUS scratch | `pbus_sw_scratch_{4..63}` | Ampere, Ada, Hopper, GB100 | The headers give 64 words on these (Turing's gives no size, so it keeps four). Scratch. |
| PFB flush address | `pfb_niso_flush_sysmem_addr{,_hi}`, `pfb_fbhub_pcie_flush_sysmem_addr_{lo,hi}` | Ampere; Hopper | Written by the driver, kept as written (the high word keeps its defined 24 bits on Ampere). |
| PFB MMU | `pfb_pri_mmu_page_fault_ctrl`, `pfb_pri_mmu_fault_buffer_{get,put}_{0,1}` | Turing | No fault is raised, so PUT stays 0; GET keeps its pointer; the control register starts at the header's default (send none). Model. |
| MMU ECC | `pfb_pri_mmu_{l2tlb,hubtlb,fillunit}_ecc_uncorrected_err_count` | Turing | The injector has no location for MMU SRAMs: read zero, keep what is written. Model. |
| Memory ECC | `pfb_fbpa_0_ecc_ded_count_{0,1}`; `..._{0..3}_gh100` | Turing; Hopper | See below. |
| L2 ECC | `pltcg_ltc0_lts0_l2_cache_ecc_uncorrected_err_count` | Turing | See below. |
| Topology table | `ptop_device_info_cfg` | GB100 | Only the format version, 2 at reset, which is all GB100's `dev_top.h` publishes as a value; the other fields read zero. The table's rows (`NV_PTOP_DEVICE_INFO2(i)`, from 0x022800) are not mapped. Model. |
| Thermal scratch | `therm_i2cs_scratch`, `therm_i2cs_scratch_gb20x` | Hopper, GB100; GB20x | Scratch. The GB20x header puts it elsewhere. |

**ECC counters**, the registers a memory diagnostic reads to learn whether its
test pattern tripped ECC: the double-bit (uncorrectable) count of the first
memory partition (`NV_PFB_FBPA_0_ECC_DED_COUNT`) and of the first L2 slice. They
report what `vgpu fault` injected as an *uncorrectable* error in device memory
or L2 since the driver loaded -- the same counts NVML reports; a corrected error
does not count here. The injector keeps one device-wide count rather than one per
partition, so it is shown on counter 0 of the array and the others read zero:
the sum is the device's count, the distribution is not modelled. As on
hardware, a write sets the counter (write 0 to clear); it counts up from
what was written, and a driver reload restarts the count (a counter that was
written before the reload is only right again until its count climbs back past
the count it was written at). A model, marked as one.

```
vgpu fault inject --ecc uncorrected --count 3              # device memory
vgpu regs read --space mmio pfb_fbpa_0_ecc_ded_count_0   # 3  (Turing)
vgpu regs write --space mmio pfb_fbpa_0_ecc_ded_count_0 0
```

What the headers do *not* publish, and so what stays unmapped everywhere: the
device topology table's rows (PTOP's `DEVICE_INFO2` has a layout, in GA100's and
GB100's headers, but the values are per chip; only GB100's version is mapped),
the PRI ring station and PRI error registers (none of the headers fetched for
Turing, Ampere, Hopper or Blackwell defines one), the temperature
sensors (the thermal header gives a scratch word only), the Ampere and Ada
memory-controller ECC counters (no header), and Blackwell's zero-based
`dev_ltc_zb`, `dev_fuse_zb` and `dev_tmr` registers, whose offsets are inside
a unit whose BAR0 base the headers do not give. A caveat on the core above:
`pbus_bar1_block`, `pbus_bar2_block` and the interrupt pair at 0x100 and 0x140
are declared for every architecture, but the headers define them for Turing
(and, for BAR1, Ampere) only; they stay mapped everywhere until the other
architectures are checked, and are not gated.

The map grew from the headers rather than from more measurement for a reason
worth recording: a read-only sweep of the rest of that card's BAR0 halted its
microcontroller (Xid 62) and the GPU needed a reset. Reading a card register by
register is not a safe way to map it while a driver is bound. `vgpu regs list` shows which capability each register belongs
to; `vgpu regs read` takes a name and finds it on the chosen GPU, and the C
API's `vgpu_regs_locate` does the same.

## Who reads the registers

NVML and nvidia-smi report the PCIe link by reading it from the registers --
Link Capabilities for the maximum, Link Status for the link as trained -- as a
driver does, so the link every tool reports is the one the registers hold.

Every access is logged -- offset, size, value, and the process that made it,
by the name it was started under -- and `vgpu regs log` shows the newest 256.
The drop-in tools run under their own names, so the log tells nvidia-smi from
rocm-smi from a Python program using NVML:

```
19:50:46.596  nvidia-smi       pid 914796  read   0x084 link_capabilities        4 bytes = 0x00030905
19:50:46.596  nvidia-smi       pid 914796  read   0x08a link_status              2 bytes = 0x1043
19:50:46.613  python3          pid 914797  read   0x084 link_capabilities        4 bytes = 0x00030905
```

A whole-space read, as `lspci` dumps it, is one entry.

## sysfs

Inside `vgpu shell`, each GPU is a PCI device where the kernel keeps one:
`/sys/devices/pci0000:00/0000:00:0N.0/0000:0N:00.0`, reached from
`/sys/bus/pci/devices` (the session's devices in place of the host's) and from
`/sys/class/drm/cardN/device`. Its files are written from the registers and
rewritten whenever the device changes -- a register write, an injected error, a
degraded link:

| File | Content |
| --- | --- |
| `config` | the 4096-byte configuration space |
| `resource` | start, end and flags of each BAR, then the ROM and SR-IOV slots, as the kernel prints them |
| `vendor`, `device`, `class`, `revision`, `subsystem_vendor`, `subsystem_device` | the IDs, in hex |
| `current_link_speed`, `current_link_width`, `max_link_speed`, `max_link_width` | the link, as `8.0 GT/s PCIe` and `8` |
| `aer_dev_correctable`, `aer_dev_nonfatal`, `aer_dev_fatal` | the AER stats (see [telemetry.md](telemetry.md)) |
| `ras/*_err_count` | amdgpu's per-block counts, on AMD cards |
| `gpu_metrics` | the SMU's metrics table, on AMD cards (see AMD MMIO above) |
| `gpu_busy_percent`, `mem_busy_percent`, `mem_info_vram_total`, `mem_info_vram_used`, `mem_info_vis_vram_*` | amdgpu's activity and memory counts, on AMD cards |
| `hwmon/hwmonN/` | amdgpu's hwmon, on AMD cards: `temp2` (junction) and `temp3` (memory) with their `crit`, `crit_hyst` and `emergency` limits, `power1_average` and `power1_cap*`, `freq1` (sclk) and `freq2` (mclk), in hwmon's units -- the set an MI300 exposes, which has no edge sensor or voltage |

`/sys/class/hwmon` in the session lists the GPUs' hwmon devices, so
`sensors` reads them as it reads a real amdgpu:

```
amdgpu-pci-0100
Adapter: PCI adapter
junction:     +38.0°C  (crit = +100.0°C, hyst = +95.0°C)
                       (emerg = +110.0°C)
mem:          +38.0°C  (crit = +100.0°C, hyst = +95.0°C)
                       (emerg = +110.0°C)
PPT:         368.93 W  (cap = 750.00 W)
```

Utilization, temperature and power change on their own, so the session
rewrites these files every second, and at once when a fault or a register write
changes the device. The limits' hysteresis and emergency margins are a model.

## lspci

`vgpu smi --lspci-dump` writes each device's configuration space from the
register model, all 4096 bytes, in `lspci -xxxx` form. The real `lspci` renders
it:

```bash
vgpu fault link --gen 3 --width 8
vgpu smi --lspci-dump > /tmp/pci.dump
lspci -F /tmp/pci.dump -vvv
```

```
		LnkSta:	Speed 8GT/s (downgraded), Width x8 (downgraded)
```

A file given to `lspci -F` has no sysfs behind it, so `lspci` cannot tell that
the upper half of a 64-bit BAR belongs to the BAR before it, and lists it as a
region of its own. A dump of a real card with BARs above 4 GiB renders the same
way.
