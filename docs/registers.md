# Registers

A VirtualGPU device has register spaces the way a card does, each backed by a
register database: every register is declared once, with its offset, width,
access and what backs its value, and every tool reads the same state through
it. The first space is **PCI configuration space**, the same layout on every
vendor's card. MMIO behind BAR0 comes next, AMD first, since its register
definitions are public.

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

## The access log

Every access is logged -- offset, size, value, and the process that made it
-- and `vgpu regs log` shows the newest 256. A whole-space read, as `lspci`
dumps it, is one entry.

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
