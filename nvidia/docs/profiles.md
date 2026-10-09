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
| `nvidia/rtx-pro-6000-server` | Blackwell | 12.0 | 188 |

## Public documentation only (`verified: false`)

| Profile | Architecture | Compute capability | SMs | Memory | What is not NVIDIA's own figure |
| --- | --- | --- | --- | --- | --- |
| `nvidia/h200` | Hopper | 9.0 | 132 | 141 GiB HBM3e | inherits the H100 |
| `nvidia/b200` | Blackwell | 10.0 | 148 | 179 GiB (measured total) HBM3e | SM count |
| `nvidia/b300` | Blackwell Ultra | 10.3 | 148 | 268 GiB HBM3e | SM count |
| `nvidia/gb200` | Blackwell | 10.0 | 148 | 186 GiB HBM3e | SM count and clocks are the B200's; the memory is half of NVIDIA's per-superchip figure |
| `nvidia/rtx5090` | Blackwell | 12.0 | 170 | 32 GiB GDDR7 | |
| `nvidia/rtx-pro-6000` | Blackwell | 12.0 | 188 | 96 GiB GDDR7 | none: NVIDIA's whitepaper gives the SM count |
| `nvidia/vr200` (Rubin) | Rubin | 10.7 | 224 | 288 GiB HBM4 | **placeholder PCI id 0x7F10**; shared memory per SM from the CUDA programming guide's 10.7 column; threads, blocks, power, clocks copied from the B300/B200; Ollama 0.21.0's ggml-cuda has no build for exactly 10.7, so Ollama itself rejects the GPU |
| `nvidia/thor` (Jetson AGX Thor) | Blackwell | 11.0 | 20 | 128 GiB LPDDR5X, shared with the CPU | **placeholder PCI id 0x7F11**; SM count derived from NVIDIA's 2560 CUDA cores; threads and blocks copied from sm_100 |
| `nvidia/gb10` (DGX Spark) | Blackwell | 12.1 | 48 | 128 GiB LPDDR5x, shared with the CPU | SM count derived from NVIDIA's 6,144 CUDA cores |
| `nvidia/rtx4090` | Ada Lovelace | 8.9 | 128 | 24 GiB GDDR6X | none |
| `nvidia/l40` | Ada Lovelace | 8.9 | 142 | 48 GiB GDDR6 | SM clock is the L40S's |
| `nvidia/rtx3090` | Ampere | 8.6 | 82 | 24 GiB GDDR6X | none |
| `nvidia/a40` | Ampere | 8.6 | 84 | 48 GiB GDDR6 | none |
| `nvidia/a30` | Ampere | 8.0 | 56 | 24 GiB HBM2 | SM count derived from two OEM listings' 224 Tensor Cores; clocks are the A100's |
| `nvidia/rtx2080ti` | Turing | 7.5 | 68 | 11 GiB GDDR6 | none |
| `nvidia/rtx-pro-6000-max-q` | Blackwell | 12.0 | 188 | 96 GiB GDDR7 | SM clock derived from the data sheet's 110 TFLOPS |
| `nvidia/rtx6000-ada` | Ada Lovelace | 8.9 | 142 | 48 GiB GDDR6 | persisting L2 copied from the L40S |
| `nvidia/rtx-a6000` | Ampere | 8.6 | 84 | 48 GiB GDDR6 | none |
| `nvidia/rtx-a5000` | Ampere | 8.6 | 64 | 24 GiB GDDR6 | SM clock derived from the data sheet's 27.8 TFLOPS; L2 is GA102's |
| `nvidia/a100-80gb-pcie` | Ampere | 8.0 | 108 | 80 GiB HBM2e | limits inherited from the A100 SXM4 profiles that were read from cards |
| `nvidia/h100-nvl` | Hopper | 9.0 | 132 | 94 GiB HBM3 | **SM count derived** from the published 30 TFLOPS FP64; clock copied from the H100 PCIe; memory clock derived |
| `nvidia/h200-nvl` | Hopper | 9.0 | 132 | 141 GiB HBM3e | SM count and clock derived the same way |
| `nvidia/rtx4080` | Ada Lovelace | 8.9 | 76 | 16 GiB GDDR6X | none |
| `nvidia/rtx3070` | Ampere | 8.6 | 46 | 8 GiB GDDR6 | none |

The memory column is NVIDIA's "GB", which for these boards is binary (an A100
40 GB reports 40960 MiB). A real card shows a little less in `nvidia-smi`
because the driver keeps some, so each figure is an upper bound until measured.

## GPC layout

A profile may carry a `layout:` section (`gpcs`, `tpcs`, `sms_per_tpc`): the GPC and TPC counts NVIDIA's
whitepapers publish for the part, which cluster occupancy needs. The spread of the SMs over the GPCs is always derived,
and a part with no published GPC count has none. See [clusters.md](clusters.md) for the table and the rules.

## Placeholder PCI ids

NVIDIA has published no PCI device id for Rubin or for Thor's GPU. Rechecked on
2026-10-09 against the current main branch of NVIDIA's open GPU kernel modules
(commit ca2d03e, 2026-10-07): the name table (`g_nv_name_released.h`) has no entry
for either, and the same table does carry B200 (0x2901), GB200 (0x2941), GB10
(0x2E12), B300 (0x3182), GB300 (0x31C2, 0x31C3) and every RTX PRO Blackwell
(the ids of the existing profiles are all in it). The driver source names the Rubin dies
(GR100, GR102) and publishes two **ranges** of self-hosted Rubin ids in
`detect-self-hosted.h` (0x3040-0x307f and 0x30c0-0x30ff), which say where Rubin ids
will fall and not which one a product has; Thor, an integrated Tegra part, is in
neither the table nor the ranges.
Every profile needs a unique id (its register file is found by it), so these two
carry **placeholders, not NVIDIA's ids**: `nvidia/vr200` is 0x7F10 and
`nvidia/thor` is 0x7F11. Neither appears in any entry of that table or in either
range, so neither collides with a real NVIDIA part, and both are below 0x8000: CUDA reports the id as
`device << 16 | vendor` in a signed int, which a real id (always below 0x8000)
never overflows. The profile header, the
`telemetry.pci_device_id` comment and the generated register file's header all
say so (the generator marks these two ids, listed in `src/core/regs.cpp`). Replace them, in the
profile, and regenerate with `vgpu regs export`, when NVIDIA publishes the ids.
The AMD MI300X and MI350X placeholders are placeholders too.

## Double-precision rate

`cudaDevAttrSingleToDoublePrecisionPerfRatio` (`device_attributes.cpp`) follows
the "Throughput of Native Arithmetic Instructions" table in NVIDIA's CUDA C++
Best Practices Guide: fp32 over fp64 results per clock per SM is 32 for 7.5, 2
for 8.0, 64 for 8.6 and 8.9, 2 for 9.0 and 10.0, and **64 for 10.3 and 12.x** (the
table gives 10.3 and 12.x one fp64 cell, 2 against 128 for fp32; the code said 2 for
10.3, the B200's rate, until 2026-10-09). The table has no 10.7 or 11.0 column.
Rubin (10.7) is **4**, derived from the 130 TFLOPS fp32 and 33 TFLOPS fp64 vector rates
in NVIDIA's "Inside the NVIDIA Rubin Platform" blog (Table 3), not from a card or from
CUDA's table. Thor (11.0) is **unpublished**: NVIDIA's Jetson Thor data sheet lists AI
throughput only, and the value 64 is the 12.x stand-in, labelled as one in the code.

## Rented-card characterization, 2026-10-09: not done

Closing the unverified list needs a card of each model. On 2026-10-09 a g7e.2xlarge
(RTX PRO 6000 Blackwell Server Edition) was asked for in us-east-1 and had no capacity; it was read later that night in us-east-2 (see the table below). H200 and B200 are only sold as p5en.48xlarge and p6-b200.48xlarge (192 vCPUs, 8 GPUs); the
account's "Running On-Demand P instances" and "All P Spot Instance Requests" quotas are 64 vCPUs,
so neither can launch before a quota increase. `nvidia/tools/characterize.cu` now also writes the
`cuda:` section (bus width, clock, copy engines, persisting L2) from the card, so the next run
replaces the datasheet values in that section too. L40 is not rentable: g6e is the L40S.

## Not yet profiles

- **H20**: NVIDIA publishes no SM count for it.
- **GH200 144GB** (0x2348): the GPU's memory bandwidth and power are in no NVIDIA page read for this.
- **A100 40GB PCIe** (0x20F1): NVIDIA's A100 page lists no column for it.
- **Jetson Orin and T4G** (compute capability 8.7): integrated Tegra parts, which the device model
  (a PCI card with its own memory) does not describe, and no SM 8.7 SASS has been exercised.

## What the AWS measurements of 2026-10-09 could verify

Cards rented for the narrow-precision work ([lowprec.md](lowprec.md)) read more
than the library probes needed. Nothing here flips a `verified` flag: a profile is
`verified: true` only when the whole characterization (`nvidia/tools/characterize.cu`)
was read from the card, and these runs read only what is listed.

| Profile | Card | What was read | Result |
| --- | --- | --- | --- |
| `nvidia/l4` | AWS g6.xlarge, driver 595.91.07, CUDA 13.2 | `device_attributes --dump` (`nvidia/tests/data/cuda_attributes_l4.card.txt`), `nvidia-smi -q`, PCI ids, clocks, power limit, memory total | Matches the profile except: `persistingL2CacheMaxSize` is 34603008 (the profile had the A100's three quarters of L2, 37748736: **corrected**); `cudaDevAttrMemoryPoolSupportedHandleTypes` 9; the PCI bus, UUID and subsystem id (where the card sits, not the model's); and the capabilities the simulator does not implement (43 attributes, host memory pools, DMA-BUF, RDMA, fabric handles ...). The nvidia-smi values in the profile (23034 MiB, 72 W, 2040 and 6251 MHz, 0x27B8) are the card's. |
| `nvidia/l4` | the same | cuBLASLt, cuSPARSELt and the FP8 conversions, 6900 lines | reproduced line for line ([lowprec.md](lowprec.md)) |
| `nvidia/rtx3060` | the development machine | the same probes | reproduced line for line |
| `nvidia/l40s` | AWS g6e.2xlarge, driver 595.91.07, CUDA 13.2, cuBLAS 13.3, cuSPARSELt 0.10.0.12 (2026-10-09, `r4-aws-verify`) | the whole characterization (`profile.yaml`, 142 SMs, 384-bit, 350 W, 2520/9001 MHz, 0x26B9), `ptx_semantics` and `control_flow` (512 values), `device_attributes --dump` (`nvidia/tests/data/cuda_attributes_l40s.card.txt`), `nvidia-smi -q` (`nvidia/tests/data/smi/nvidia-smi-q.l40s.txt`), `lt`, `sparselt`, `cvt` ([lowprec.md](lowprec.md)) | `nvidia/l40s` was already `verified: true`; the profile reproduces the conformance values and all three narrow-precision transcripts, which are **byte-identical to the L4's** (the sm_89 FP8 kernel tables of cuBLASLt and cuSPARSELt do not depend on the part). One value was a guess and is **corrected**: `persisting_l2_bytes` is 69206016 (eleven sixteenths of the 96 MB L2, the ratio the L4 showed), not 75497472 (three quarters). `nvidia/l40` and `nvidia/rtx6000-ada` (the same AD102 die and L2) follow it. `nvidia/rtx4090` (derived as three quarters of its 72 MB L2, 56623104) and `nvidia/rtx5090` and the RTX PRO 6000 profiles (three quarters) are NOT changed: the ratio is two data points (L4, L40S), not a rule. The same attribute dump differs from the simulator on 58 lines, all of the kinds listed for the L4 (capabilities the simulator does not implement, bus ids). |
| `nvidia/rtx4090` | none | none | Same compute capability as the L4, so the FP8 kernel table the L4 answered is applied to it; not a measurement of it. |
| `nvidia/l40` | none | none | The L40S's die; not a measurement of it. |
| `nvidia/h100`, `nvidia/h100-pcie`, `nvidia/gh200-480gb`, `nvidia/h200` | none | none | AWS had no `p5.4xlarge` capacity (three attempts, six zones). The Hopper FP8 rules are documentation-derived. |
| `nvidia/rtx-pro-6000-server` | AWS g7e.2xlarge (us-east-2), driver 595.91.07, CUDA 13.2, cuBLAS 13.3 (2026-10-09, `r4-aws-verify`) | the whole characterization (188 SMs, 512-bit, 600 W, 2430/12481 MHz, 0x2BB5), `ptx_semantics` and `control_flow` (512 values), `device_attributes --dump`, `nvidia-smi -q` (`nvidia/tests/data/smi/nvidia-smi-q.rtx-pro-6000-server.txt`), `lt`, `sparselt`, `cvt`, `ptx120` and the natively run programs ([lowprec.md](lowprec.md), `nvidia/tests/data/card/rtx-pro-6000-server/`) | **`verified: true`**: the simulator reproduces the 512 conformance values. **Corrected**: boost clock 2430 MHz (was the Workstation Edition's 2617), memory clock 12481, persisting L2 83886080 (five sixths of L2, was 3/4), CUDA memory total 101975851008 B (97252 MiB; nvidia-smi 97887 MiB). The narrow-precision probes are NOT reproduced yet (`known-gaps.txt`). `nvidia/rtx-pro-6000` (the Workstation Edition, same GB202) still carries the derived 3/4 persisting L2 and 2617 MHz. |
| `nvidia/rtx-pro-6000` | none | none | The Workstation Edition; not a measurement of it (see the Server Edition's row for the GB202's numbers). |
| `nvidia/b200`, `nvidia/b300` and the other Blackwell profiles | none | none | An eight-GPU instance is the only way to rent them and the round's rules exclude it. |
