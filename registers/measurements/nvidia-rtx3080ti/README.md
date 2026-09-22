# NVIDIA GeForce RTX 3080 Ti (GA102), measured

Captured with `tools/regprobe/regprobe.py capture` on 2026-09-22, Linux 6.8.0-139-generic, NVIDIA
driver 595.84, card idle, run as an ordinary user: so configuration space is the
64-byte header the kernel gives anyone, and the capabilities, the extended space
and BAR0 are not here -- those need root.

| File | What it is |
| --- | --- |
| `config-header.bin` | configuration space, bytes 0x00-0x3f |
| `lspci-vvv.txt` | `lspci -vvv -nn`, capabilities denied without root |
| `resource`, `*_link_*`, `aer_dev_nonfatal` | the device's sysfs files |

What it changed in the model (registers/pci-config.yaml, `measured:`):

- `command` reads 0x0407 on a bound card: memory and bus mastering on, INTx off
  for MSI, I/O on for the I/O BAR.
- `header_type` is 0x80: a GeForce is multifunction (function 1 is its HDMI
  audio controller).
- The link drops to 2.5 GT/s (Gen1) while idle, of 16 GT/s (Gen4) x16.
- The BAR layout matched the model as it was: BAR0 16 MiB 32-bit, BAR1 256 MiB
  and BAR3 32 MiB 64-bit prefetchable, BAR5 128 bytes of I/O.
- The kernel's AER stats list more counters than the model printed; the list is
  now the one this kernel prints.

The addresses, IRQ and subsystem IDs are this card's and this host's, not the
model's.
