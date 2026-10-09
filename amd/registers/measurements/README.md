# AMD Radeon RX 6800 and RX 6700 XT, measured

Read from two cards in one machine on 2026-10-09: Ubuntu 24.04, Linux 7.0 with amdgpu, ROCm 7.2.0,
both on PCIe 4.0 x16, idle. The machine's integrated GPU (gfx1036) is not profiled.

| Directory | Card | Chip |
| --- | --- | --- |
| `rx6800/` | Radeon RX 6800, 16 GB, VBIOS 113-D4120900-101, PCI 1002:73bf rev c3, subsystem 1002:0e3a | Navi 21, gfx1030, 60 CUs |
| `rx6700xt/` | Radeon RX 6700 XT, 12 GB, VBIOS 113-D5121100-101, PCI 1002:73df rev c1, subsystem 1002:0e36 | Navi 22, gfx1031, 40 CUs |

| File | What it is |
| --- | --- |
| `rocminfo.txt` | `rocminfo` with the other GPUs removed (the host CPU's agent stays) |
| `characterize-hip.txt` | `amd/tools/characterize-hip.cpp` on the card: HIP's device properties |
| `amd-smi-static.json` | `amd-smi static --json` for the card |
| `kfd-topology.txt` | the card's `/sys/class/kfd/kfd/topology/nodes/<n>`: properties, memory banks, caches, links |
| `lspci-vvv.txt`, `lspci-xxxx.txt` | `lspci -vvv` and `lspci -xxxx` of it (all 4096 bytes of configuration space) |

The board's serial number, unique id and the UUID ROCm derives from it are replaced by `(removed)`,
`0` and `GPU-XX`.

What the profiles took from these (`amd/profiles/rx6800.yaml`, `rx6700xt.yaml`): the compute units, HIP's
multiprocessor count and limits, memory size, clocks, power caps and thermal limits, PCI ids, the L2, and
for `amd/include/vgpu/amd_chip.hpp` the shader-engine and cache layout.

Not taken: the revision (c3 and c1) and the subsystem id (0e3a and 0e36). The register database derives
both from the device id (`amd/src/regs.cpp`, `src/core/telemetry.cpp`), so a simulated card reads the
generic Radeon revision c0 and a subsystem id equal to its device id. Making them the card's own needs a
field in the shared telemetry structure.
