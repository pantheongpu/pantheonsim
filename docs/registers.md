# Registers

A VirtualGPU device has register spaces the way a card does, each backed by a
register database: every register is declared once, with its offset, width,
access and what backs its value, and every tool reads the same state through
it. Two spaces so far: **PCI configuration space**, the same layout on every
vendor's card, and an **AMD GPU's MMIO registers** behind BAR5, from the Linux
amdgpu headers. NVIDIA's MMIO waits on a decision about where its register
definitions may come from.

```bash
vgpu regs list                          # the database: offset, width, access, backing
vgpu regs read link_status              # read and decode a register
vgpu regs write command 0x0002          # write it, with its semantics
vgpu regs dump --extended               # all 4096 bytes
vgpu regs log                           # who read and wrote what
```

`--gpu N` picks the device (0 by default); `read` and `write` take a register's
name or its offset (`0x08a`), and `--size 1|2|4` for a partial access.

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

The mailbox works as the driver drives it: clear the response, write the
argument, write the message, and the SMU answers -- `1` in the response
register, and its reply in the argument register. It answers TestMessage (the
argument plus one), GetSmuVersion, GetDriverIfVersion and GetMetricsVersion,
and refuses any other message with `0xfe`, unknown command:

```bash
vgpu regs write --space mmio smu_response 0
vgpu regs write --space mmio smu_argument 0
vgpu regs write --space mmio smu_message 0x2        # GetSmuVersion
vgpu regs read --space mmio smu_argument            # 0x00556f00: 85.111.0
```

The register offsets, bit fields and message numbers come from the amdgpu
headers (`gc_9_4_3_*.h`, `mp_13_0_6_offset.h`, `smu_v13_0_6_ppsmc.h`; their
MIT notice is in `registers/LICENSES/amdgpu-headers.txt`), and each entry's
`source` names the symbol. They are offsets from an IP block's base, and MI300
learns its bases at boot from its IP discovery table; the database uses
Aldebaran's (MI200's), the same GFX9 family. That, the engine status values and
the firmware version are a model until checked on a card.

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
