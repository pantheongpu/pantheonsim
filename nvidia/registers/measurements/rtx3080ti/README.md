# NVIDIA GeForce RTX 3080 Ti (GA102), measured

Captured with `tools/regprobe/regprobe.py` on 2026-09-22 on server1, Linux
6.8.0, NVIDIA driver 595.84, card idle: the configuration space as root, all
4096 bytes, and the first registers behind BAR0.

| File | What it is |
| --- | --- |
| `config.bin` | configuration space, 0x000-0xfff, as the bound driver left it |
| `lspci-vvv.txt`, `lspci-xxxx.txt` | `lspci -vvv -nn` and `lspci -xxxx` of it |
| `resource`, `*_link_*`, `aer_dev_nonfatal` | the device's sysfs files |
| `bar0-head.tsv` | BAR0 0x0-0xc: `0xb72000a1` (the chip's identification, revision a1), then zeros, then `0xbadf5040` from 0xc -- the value a protected or absent register reads as |

The simulator replays `config.bin` for Ampere GeForce profiles (the RTX 3060):
every register the database declares is found through this card's capability
chain and keeps its live behaviour -- link status, error status, BAR sizing --
and every other byte is this card's. Against it, the model differs only in the
IDs and the BAR addresses (tests/unit/test_regs.cpp).

What it showed that the generic layout did not have:

- A longer standard chain: power management, MSI (enabled by the driver), PCI
  Express as a legacy endpoint with 256-byte payloads, ASPM L0s and L1, clock
  power management and LTR, then a vendor-specific capability at 0xb4.
- Eleven extended capabilities: Virtual Channel (0x100), Power Budgeting
  (0x128), LTR (0x250), L1 PM Substates (0x258), AER (0x420, no ECRC),
  vendor-specific (0x600), Secondary PCI Express (0x900), Resizable BAR (0xbb0:
  BAR1 256 MiB of 64 MiB to 16 GiB), 16 GT/s physical layer (0xc1c), lane
  margining (0xd00) and data link features (0xe00).
- The command register as a bound driver leaves it (0x0407), a multifunction
  header (0x80: function 1 is the HDMI audio controller), and the link at
  2.5 GT/s while idle of 16 GT/s.

The IRQ line and BAR addresses are this host's; the subsystem IDs this board's.
