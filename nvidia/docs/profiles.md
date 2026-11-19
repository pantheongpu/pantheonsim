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
| `nvidia/rtx-pro-4500` | Blackwell | 12.0 | 82 |

## Public documentation only (`verified: false`)

| Profile | Architecture | Compute capability | SMs | Memory | What is not NVIDIA's own figure |
| --- | --- | --- | --- | --- | --- |
| `nvidia/h200` | Hopper | 9.0 | 132 | 141 GiB HBM3e | inherits the H100 |
| `nvidia/b200` | Blackwell | 10.0 | 148 | 179 GiB (measured total) HBM3e | SM count |
| `nvidia/b300` | Blackwell Ultra | 10.3 | 148 | 268 GiB HBM3e | SM count |
| `nvidia/gb200` | Blackwell | 10.0 | 148 | 186 GiB HBM3e | SM count and clocks are the B200's; the memory is half of NVIDIA's per-superchip figure |
| `nvidia/rtx5090` | Blackwell | 12.0 | 170 | 31.8 GiB GDDR7 (measured framebuffer, an upper bound) | |
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

## Numerics and provenance (`numerics:`, `provenance:`, `evidence:`)

Three optional sections say what a device's arithmetic units do where that is not plain IEEE arithmetic, and what each value
in the profile rests on. They are parsed by the same loader as the rest (`src/core/profile.cpp`, the restricted YAML of
`include/vgpu/yamlish.hpp`: one level of nesting, inline lists), are embedded in the binaries with the profile, and
`vgpu info --gpu <id>` prints them (`--json` too). AMD and Intel profiles use `provenance:` and `evidence:` with the same rules;
`numerics:` is NVIDIA's only for now.

```yaml
# nvidia/profiles/t4.yaml
numerics:
  tensor_sum: ampere               # sequential | ampere | hopper
  tensor_sum_chunk: 8              # products to a chunk, f16 and bf16
  tensor_sum_chunk_tf32: 4         # ... tf32
  tensor_sum_grid: 25              # a chunk's terms are cut at 2^(top - 25)
provenance:
  measured: [numerics.tensor_sum, numerics.tensor_sum_chunk, numerics.tensor_sum_grid]
  inferred: [numerics.tensor_sum_chunk_tf32]
evidence:
  numerics: "card: Tesla T4 (this model), AWS; driver: 595.112.07; toolkit: CUDA 13.0; date: 2026-11-15. simd_lane_forms built -DNO80 prints ..."
  numerics.tensor_sum_chunk_tf32: "not measurable on this card: a Turing tensor core has no tf32 form. ..."
```

```yaml
# nvidia/profiles/b200.yaml
numerics:
  tensor_sum: hopper               # sequential | ampere | hopper
  tensor_sum_chunk: 16             # products to a chunk, f16 and bf16
  tensor_sum_chunk_tf32: 8         # ... tf32
  tensor_sum_grid: 26              # a chunk's terms are cut at 2^(top - 26)
provenance:
  inferred: [numerics.tensor_sum, numerics.tensor_sum_chunk, numerics.tensor_sum_chunk_tf32, numerics.tensor_sum_grid]
evidence:
  numerics: "inferred from data recorded by others on a B200 (compute capability 10.0), NOT run by us on a card. ..."
```
(Until 2026-11-16 this profile said `tensor_sum: sequential` with `tensor_sum_published: hopper`; so did the compute capability 12.0 profiles
(`rtx5090` and the `rtx-pro-*` ones) until 2026-11-17, when an RTX PRO 6000 Blackwell Server Edition was run. No profile has a
`tensor_sum_published` now; the key stays for a model that is published and not yet measured or recorded.)

### `numerics:`

The sum the device's tensor core makes of f16, bf16 and tf32 products (`wmma`, `mma.sync`, SASS `HMMA`). **The engines read it
from the loaded profile**; no engine compares a compute capability for it (`tc86::sum_model(profile)` in
`include/vgpu/exec/tc_ampere.hpp` is the one place). The rule itself, the cards and what is still unmeasured are in
[docs/matrix-numerics.md](../../docs/matrix-numerics.md).

| Key | Value | Meaning |
| --- | --- | --- |
| `tensor_sum` | `sequential`, `ampere` or `hopper` (required when the section is there) | The model in force by default (`vgpu::TensorSum`). `sequential` is `acc += a * b` in f32: what no card is known to do, and what a device gets whose tensor core has not been measured. |
| `tensor_sum_chunk` | integer | Products to a chunk, f16 and bf16: 8 (`ampere`), 16 (`hopper`). |
| `tensor_sum_chunk_tf32` | integer | The same for tf32: 4, 8. |
| `tensor_sum_grid` | integer | A chunk's terms are cut at `2^(top - grid)`: 25, 26. |
| `tensor_sum_published` | `ampere` or `hopper` (optional, only beside `tensor_sum: sequential`) | The model the literature gives the device, which no card of ours has checked and no recorded data covers. Never in force by default: `VGPU_TC_EXACT=published` turns it on (no profile uses it now: `b200` and `gb200` have the rule in force). |

| `tensor_sum_fp8` | `sequential`, `ada` or `via_f16` (optional) | The sum of the e4m3 and e5m2 `mma.sync` forms (`vgpu::Fp8Sum`, `include/vgpu/exec/tc_fp8.hpp`). Absent is `sequential`. `ada`: chunks of 16, terms cut at `2^(top - 14)`, an f32 result cut to 13 fraction bits. `via_f16`: ptxas turns the form into f16 `HMMA`s, so the `tensor_sum` rule sums them and C is added afterwards (needs a `tensor_sum` that is not `sequential`). |
| `tensor_sum_fp8_chunk`, `_grid`, `_out_bits` | integers (with `ada` only) | 16, 14, 13: written out as the f16 numbers are. Absent with `sequential` and `via_f16`. |
| `tensor_sum_wgmma_fp8` | `sequential` or `hopper` (optional) | The sum of the dense e4m3 and e5m2 forms of `wgmma` (PTX) and `QGMMA` (SASS) (`vgpu::WgmmaFp8Sum`, `include/vgpu/exec/tc_fp8.hpp`). Absent is `sequential`. `hopper`: one chunk of 32 products and the accumulator, terms cut at `2^(top - 14)`, an f32 result cut to 13 fraction bits. |
| `tensor_sum_wgmma_fp8_chunk`, `_grid`, `_out_bits` | integers (with `hopper` only) | 32, 14, 13: written out as the others are. Absent with `sequential`. |

Rules the loader enforces (a profile that breaks one does not load, and the message names the file and the key):

- `tensor_sum_fp8: ada` needs its three numbers and they must be 16, 14 and 13; `sequential` and `via_f16` must have none; `via_f16` needs a
  `tensor_sum` other than `sequential`; the numbers without the name are refused (`test_profile`).
- `tensor_sum_wgmma_fp8: hopper` needs its three numbers and they must be 32, 14 and 13; `sequential` must have none; the numbers without the name are refused (`test_profile`).

- The three numbers are required with `ampere` and `hopper` and must be that model's (8 / 4 / 25, 16 / 8 / 26); with `sequential`
  they must be absent. The engines implement these two rules (scalar code for both, vector kernels for the first), not a sum of any
  chunk and grid, so other numbers would be printed and hashed and would not be what runs. They are written out so that a profile
  says what its card does, not only a name. A new rule is new engine code first.
- A key the table does not have is refused, and so is the section on a profile that is not NVIDIA's.
- **Every key in `numerics:` must have a provenance class** (below). A numerics value decides result bits, so none goes without one.
- A profile without the section has the sequential sum and no published model. Every NVIDIA profile in the tree has the section
  (`test_profile`, `every_builtin_profile_tags_its_numerics`).

The numerics are part of the result cache's key (`memo::profile_hash`, `src/runtime/memo.cpp`): two profiles that differ only in
a numerics value never share a cached result. The classes and the evidence text are not: they change no result.

`VGPU_TC_EXACT` works on top of the profile, as before: `0` is the sequential sum whatever the profile says (the way back to the earlier results on `b200` and `gb200`), `force` gives a
profile whose `tensor_sum` is `sequential` the `ampere` rule, `published` gives it its `tensor_sum_published`.

### `provenance:`

Exactly four classes, each a list of keys. A key is named as it is written in the file: `section.key`
(`numerics.tensor_sum`, `limits.multiprocessors`) or a bare top-level key (`vram_bytes`).

| Class | Means | Its evidence says |
| --- | --- | --- |
| `measured` | Read from, or reproduced bit for bit against, a physical card **of this profile's model**, by this project. | The card, where it was, driver, toolkit, date, and what was compared. |
| `documented` | The vendor's specification, data sheet or documentation. | The document. |
| `inferred` | Not known of this device itself: carried over from a measured device of the same compute capability or die, or taken from a published model that no card of ours has checked. | The device or the publication it is carried over from. |
| `unknown` | None of these. The value is what the simulator does, not a claim about the device. | Optional. |

Rules the loader enforces:

- A class that is not one of the four is refused.
- **A key carries one class**: a key in two lists, or twice in one, is refused (and a class written twice is a duplicate key).
- A listed key must be one the profile has, so a mistyped name cannot tag nothing.
- `measured`, `documented` and `inferred` are claims and need an evidence entry (the key's own or its section's).
- A key in `placeholders:` can only be `unknown`.

**A key that is not listed has no per-key class.** That is the default, and it is deliberate: nothing is turned into `measured`
or `documented` by a rule. What is said of an unlisted key is what was said before these sections existed:

- if it is in `placeholders:` it is a stand-in, and `DeviceProfile::provenance_of` reports it as `unknown`;
- otherwise the profile-wide `verified:` flag stands (true: the characterization, `nvidia/tools/characterize.cu`, was read from a
  card of this model; false: it was not), together with the comments in the file, which say key by key where a value came from.
  `verified: true` does not make every key measured (the H100's L2 size and thermal limit are data-sheet values in a verified
  profile) and `verified: false` does not make every key documented (many are copied from the nearest profile), so the flag is not
  translated into a class. `vgpu info` prints only the keys that have a class.

Keys get a class when somebody checks what they rest on. Today that is the tensor-core sum on all 38 NVIDIA profiles and the
values of the two Radeon cards that were read (`amd/profiles/rx6800.yaml`, `rx6700xt.yaml`: what HIP, rocminfo and amd-smi
reported on the card, kept in `amd/registers/measurements/`, is `measured`; the per-multiprocessor block limit, which no tool
reports, is `inferred` from the RX 6900 XT profile). No AMD matrix arithmetic is in any profile, and none is measured.

### `evidence:`

One text per tagged key, or per section for all of its tagged keys that have no entry of their own (`numerics: "..."`). An
entry that belongs to no tagged key is refused. For a measured value the text starts with the same greppable fields in every
profile: `card: <model> (this model), <where>; driver: <version>; toolkit: <version>; date: <date>.`, then what was compared and
where the files are (`grep -n "card:" nvidia/profiles/*.yaml amd/profiles/*.yaml` lists every measurement a profile rests on).
The text is one quoted line and may not contain a double quote.

### The tensor-core sum of each profile

| Class of `numerics.tensor_sum` | Profiles | Why |
| --- | --- | --- |
| `measured` (8) | `rtx3060`, `a10g`, `t4`, `l4`, `l40s`, `a100-sxm4-40gb` (`ampere`); `h100-pcie`, `rtx-pro-6000-server` (`hopper`) | A card of exactly this model was run (2026-11-15 and, for the RTX 3060, before). The T4's `tensor_sum_chunk_tf32` alone is `inferred`: a Turing card has no tf32 form to measure. |
| `inferred` (26) | 7.5: `rtx2080ti`. 8.0: `a100` (the SXM4 80 GB), `a100-80gb-pcie`, `a30`. 8.6: `a10`, `a40`, `rtx-a5000`, `rtx-a6000`, `rtx3070`, `rtx3080ti`, `rtx3090`. 8.9: `l40`, `rtx4080`, `rtx4090`, `rtx6000-ada`. 9.0: `h100` (the SXM5 80 GB, **the default profile**), `h100-nvl`, `h200`, `h200-nvl`, `gh200-480gb`. 10.0 (`hopper`, since 2026-11-16): `b200`, `gb200`. 12.0 (`hopper`, since 2026-11-17): `rtx5090`, `rtx-pro-4500`, `rtx-pro-6000`, `rtx-pro-6000-max-q` | Another model of the same compute capability was run, not this one; the sum is the measured card's. For `b200` and `gb200` no card of ours was run at all: the sum is inferred from 20,000 vectors recorded by others on a B200 (docs/numerics/tcpapers.md). |
| `unknown` (4) | 10.3 `b300`, 10.7 `vr200`, 11.0 `thor`, 12.1 `gb10` (nothing published) | No card of the compute capability was run. They keep the sequential sum. |

Moving the sum from a table keyed on the compute capability into the profiles changed no result: `test_exec3`
(`tc_model_devices`) holds each of the 38 profiles to the model that table gave it, in every `VGPU_TC_EXACT` mode, except `b200` and `gb200` (the deliberate departure of 2026-11-16) and the five 12.0 profiles (2026-11-17): Hopper instead of sequential. A card run
that turns an `inferred` or `unknown` into `measured` changes the profile, its row in `test_profile` and, if the model changes,
its pin in `test_exec3` together.

## Peer access and managed memory (`cuda: peer_access`, `concurrent_managed_access`)

Two facts about a board that the other profile fields do not say, both `true` unless a profile says `false`:

* `peer_access`: two devices of this profile can reach each other's memory (NVLink or a PCIe peer path).
  `cudaDeviceCanAccessPeer`, `cudaDeviceGetP2PAttribute`, `cuDeviceGetP2PAttribute` and NVML's
  `nvmlDeviceGetP2PStatus` answer from it, `cudaDeviceEnablePeerAccess` and `cuCtxEnablePeerAccess` are
  `cudaErrorPeerAccessUnsupported` without it, and NVSHMEM starts no job whose PEs sit on such devices
  ("Peer GPU 1 is not accessible"). A copy between the two devices still works, through the host.
* `concurrent_managed_access`: `cudaDevAttrConcurrentManagedAccess`. Without it `cudaMemPrefetchAsync` is
  `cudaErrorInvalidDevice` and so is the advice that names a device; the other advice is accepted and forgotten.

`nvidia/rtx3060` says `false` for both: measured on the pair of RTX 3060s this was developed on (driver 13.0, WSL2;
`nvidia/tests/data/p2p_managed.rtx3060.expected`, from `run_p2p_managed.sh --card`). The `false` of
`concurrent_managed_access` is WSL's: a native Linux GeForce card answers 1 (the RTX 3080 Ti of
`managed_paths.cpp` does, and its profile keeps the default). The `false` of `peer_access` is the pair's, and it is
not known whether a GeForce pair under native Linux answers the same; no other GeForce profile says `false`
because none was measured (the RTX 3080 Ti is a single card on its machine).

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
was read from the card, and these runs read only what is listed. "512 values" below is 256 from `ptx_semantics` plus 256 from `control_flow`; the definition and the list of verified profiles are in [docs/status.md](../../docs/status.md#gpu-profiles).

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
