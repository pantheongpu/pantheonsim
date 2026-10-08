# NVIDIA device profiles

`VGPU_GPU=nvidia/<name>` (or `vgpu list`) picks the GPU being simulated. A
profile (`nvidia/profiles/<name>.yaml`) holds what a program can ask a GPU: its
compute capability, SM count, per-SM limits and memory, and the values
`nvidia-smi` shows. The compute capability decides which modules load: PTX for
a target newer than the device is refused, SASS runs on the same major version
and the same or a later minor, and the `a` and `f` targets are narrower still.

**Nothing in the second table has been measured on a card.** Each is
`verified: false`, built only from public documents, and its header comment says
where every number came from, which ones are derived, and which were copied from
the nearest profile because NVIDIA has not published them. Read the header
before trusting a number.

## Read from a physical device (`verified: true`)

| Profile | Architecture | Compute capability | SMs |
| --- | --- | --- | --- |
| `nvidia/t4` | Turing | 7.5 | 40 |
| `nvidia/a100`, `nvidia/a100-sxm4-40gb` | Ampere | 8.0 | 108 |
| `nvidia/a10`, `nvidia/a10g`, `nvidia/rtx3060`, `nvidia/rtx3080ti` | Ampere | 8.6 | 72, 80, 28, 80 |
| `nvidia/l4`, `nvidia/l40s` | Ada Lovelace | 8.9 | 58, 142 |
| `nvidia/h100`, `nvidia/h100-pcie`, `nvidia/gh200-480gb` | Hopper | 9.0 | 132, 114, 132 |

## Public documentation only (`verified: false`)

| Profile | Architecture | Compute capability | SMs | Memory | What is not NVIDIA's own figure |
| --- | --- | --- | --- | --- | --- |
| `nvidia/h200` | Hopper | 9.0 | 132 | 141 GiB HBM3e | inherits the H100 |
| `nvidia/b200` | Blackwell | 10.0 | 148 | 179 GiB (measured total) HBM3e | SM count |
| `nvidia/b300` | Blackwell Ultra | 10.3 | 148 | 268 GiB HBM3e | SM count |
| `nvidia/gb200` | Blackwell | 10.0 | 148 | 186 GiB HBM3e | SM count and clocks are the B200's; the memory is half of NVIDIA's per-superchip figure |
| `nvidia/rtx5090` | Blackwell | 12.0 | 170 | 32 GiB GDDR7 | |
| `nvidia/rtx-pro-6000` | Blackwell | 12.0 | 188 | 96 GiB GDDR7 | none: NVIDIA's whitepaper gives the SM count |
| `nvidia/vr200` (Rubin) | Rubin | 10.7 | 224 | 288 GiB HBM4 | **placeholder PCI id 0xFE10**; shared memory per SM from the CUDA programming guide's 10.7 column; threads, blocks, power, clocks copied from the B300/B200 |
| `nvidia/thor` (Jetson AGX Thor) | Blackwell | 11.0 | 20 | 128 GiB LPDDR5X, shared with the CPU | **placeholder PCI id 0xFE11**; SM count derived from NVIDIA's 2560 CUDA cores; threads and blocks copied from sm_100 |
| `nvidia/gb10` (DGX Spark) | Blackwell | 12.1 | 48 | 128 GiB LPDDR5x, shared with the CPU | SM count derived from NVIDIA's 6,144 CUDA cores |
| `nvidia/rtx4090` | Ada Lovelace | 8.9 | 128 | 24 GiB GDDR6X | none |
| `nvidia/l40` | Ada Lovelace | 8.9 | 142 | 48 GiB GDDR6 | SM clock is the L40S's |
| `nvidia/rtx3090` | Ampere | 8.6 | 82 | 24 GiB GDDR6X | none |
| `nvidia/a40` | Ampere | 8.6 | 84 | 48 GiB GDDR6 | none |
| `nvidia/a30` | Ampere | 8.0 | 56 | 24 GiB HBM2 | SM count derived from two OEM listings' 224 Tensor Cores; clocks are the A100's |
| `nvidia/rtx2080ti` | Turing | 7.5 | 68 | 11 GiB GDDR6 | none |

The memory column is NVIDIA's "GB", which for these boards is binary (an A100
40 GB reports 40960 MiB). A real card shows a little less in `nvidia-smi`
because the driver keeps some, so each figure is an upper bound until measured.

## Placeholder PCI ids

NVIDIA has published no PCI device id for Rubin or for Thor's GPU: NVIDIA's open
GPU kernel modules name table (`g_nv_name_released.h`) has no entry for either.
Every profile needs a unique id (its register file is found by it), so these two
carry **placeholders, not NVIDIA's ids**: `nvidia/vr200` is 0xFE10 and
`nvidia/thor` is 0xFE11. Both are above every id in that table (its highest is
0xC838), so neither collides with a real NVIDIA part. The profile header, the
`telemetry.pci_device_id` comment and the generated register file's header all
say so (the generator marks any NVIDIA id from 0xFE00 up). Replace them, in the
profile, and regenerate with `vgpu regs export`, when NVIDIA publishes the ids.
The AMD MI300X and MI350X are placeholder profiles too.

## Not yet profiles

- **H20**: NVIDIA publishes no SM count for it.
