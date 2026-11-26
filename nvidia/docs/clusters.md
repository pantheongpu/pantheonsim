# Cluster occupancy

`cudaOccupancyMaxActiveClusters`, `cudaOccupancyMaxPotentialClusterSize` and the driver's
`cuOccupancyMaxActiveClusters` / `cuOccupancyMaxPotentialClusterSize` (CUTLASS sizes its persistent
grid with the first). `nvidia/tests/e2e/cluster_occupancy.cu` calls all four for a plain kernel.

## What was measured

On an RTX 3060 (sm_86, no clusters), against NVIDIA's runtime and driver:

| call | answer |
| --- | --- |
| either query, the configuration names no cluster dimension | success, 0 |
| either query, the configuration names a cluster dimension (even 1x1x1, even 0) | `cudaErrorInvalidClusterSize` (`CUDA_ERROR_INVALID_CLUSTER_SIZE`), the result untouched |
| a null result pointer | `cudaErrorInvalidValue` |
| a null kernel | `cudaErrorInvalidDeviceFunction` (runtime), `CUDA_ERROR_INVALID_VALUE` (driver) |
| a block that does not fit, with no cluster dimension | success, 0 |

A null configuration, or a function pointer that is not a kernel, crashes the card's runtime; this one
answers `cudaErrorInvalidValue` / `cudaErrorInvalidDeviceFunction`. The shim's rtx3060 profile prints the
card's answers for `cluster_occupancy.cu` line for line (`nvidia/tests/data/cluster_occupancy.rtx3060.expected`,
`run_cluster_occupancy.sh --card --update` writes it).

## What rests on documentation

There was no part with clusters to measure. For compute capability 9.0 and later the answers follow what
NVIDIA documents, and **none of them is checked against a card**:

* a cluster is co-scheduled on one GPC (CUDA programming guide, "Thread Block Clusters");
* up to 8 blocks per cluster is portable; a kernel that allows non-portable sizes
  (`cudaFuncAttributeNonPortableClusterSizeAllowed`) may have up to 16 (the H100's table of limits; the
  launch check in `validate_launch` applies the same 16 from compute capability 9.0 on, so a size the occupancy
  query accepts is one a launch accepts);
* `cudaOccupancyMaxPotentialClusterSize` ignores the cluster dimension of the configuration, reports a
  required size (`__cluster_dims__`) if the kernel has one, and is 8 by default and 16 for a kernel that allows it;
* `cudaOccupancyMaxActiveClusters` needs the cluster size in the configuration or in the kernel (not
  both, and equal if both), else `cudaErrorInvalidValue` (the header says "the function will return an error"
  without saying which; a guess); a size over the limit is `cudaErrorInvalidClusterSize`;
* the blocks per SM are the ones `cudaOccupancyMaxActiveBlocksPerMultiprocessor` gives for the configuration,
  and a GPC holds `blocks per SM x its SMs` of them. **Assumed, not documented:** the blocks of one cluster may
  share an SM. For a kernel that fits one block per SM, which is what clusters are used for, this does not matter.

`vgpu/exec/cluster.hpp` has the rules; `tests/unit/test_profile.cpp` checks them on the H100 layout.

## The GPC layout

How many clusters fit depends on how many SMs each GPC has. NVIDIA's architecture whitepapers publish, for
many parts, the number of GPCs, TPCs and SMs ("H100 SXM5: 8 GPCs, 66 TPCs, 2 SMs/TPC, 132 SMs"); they do not
say how many TPCs each GPC of a part with some disabled has. So a profile's `layout:` section
(`gpcs`, `tpcs`, `sms_per_tpc`) holds the counts it has a source for, the source is in a comment beside it,
and the **spread is derived**: the enabled TPCs are divided over the GPCs as evenly as they go, the first GPCs
getting the extra (the H100 SXM5 is then two GPCs of 18 SMs and six of 16). The parser refuses a layout that
does not add up to the profile's SM count. The first time an answer rests on a spread the shim says so on stderr
(`counts_derived: true` marks counts that are themselves another part's: a GH200's GPU or an H200 has the
H100 SXM5's SM count, and NVIDIA gives no GPC count of its own for it).

Where NVIDIA has published no GPC count the call is **refused by name** (`cudaErrorNotSupported` /
`CUDA_ERROR_NOT_SUPPORTED`, with a message once) rather than answered from a layout made up for it, for
what needs the layout (a cluster larger than one block, or the largest size when non-portable sizes are allowed).
That is now GB10, Thor and Rubin (`vr200`: NVIDIA's Rubin architecture post gives 224 SMs and says the SMs are
organised in GPCs, and states no count; press readings of its block diagram say 8 GPCs of unequal size, which is
not NVIDIA's figure).

Every key of a profile's `layout:` carries a class (`provenance:`, `evidence:`), and none is `measured`: no card of ours has
run a cluster occupancy query on a part with clusters.

* `documented`: NVIDIA's own figure for the part (the whitepaper or architecture post names the part's GPCs, TPCs and SMs).
* `inferred`: carried over from the same die or from another part with the same SM count (`counts_derived: true`).
  The spread of the enabled TPCs over the GPCs is derived on every profile and is never published.

Two cases answer from what is published without a made-up number:

* **B200, GB200, B300** (inferred): NVIDIA's Blackwell Ultra architecture post says the full two-die GPU has 160 SMs
  in 8 GPCs and that the available SM count varies by SKU; the CUDA Blackwell tuning guide gives 16 as the
  non-portable cluster size of a B200 and says to size a persistent grid with `cudaOccupancyMaxActiveClusters`,
  without a GPC figure. The profiles' 148 SMs are 74 of the 80 TPCs, so the 8 GPCs are carried over: two GPCs of
  20 SMs and six of 18 (the spread is derived). A cluster of 8 then gives 16 clusters of one block per SM, of 2 gives 74.
  A real B200 may have its disabled TPCs elsewhere; one run of `cluster_occupancy.cu` on a B200 would settle it
  (`run_cluster_occupancy.sh --card --update` with `CLUSTER_OCC_SLUG=b200 CLUSTER_OCC_ARCH=sm_100a`).
* **H100 PCIe** (documented): the Hopper whitepaper gives `7 or 8 GPCs` with 57 TPCs. The profile keeps both
  (`layout.gpcs: 7`, `layout.gpcs_alt: 8`) and a query is answered where both counts give the same number (for example
  clusters of 2: 57 pairs of TPCs either way, so 114 with two blocks per SM) and refused by name where they differ
  (clusters of 4: 28 against 25).

| profiles with a layout | GPCs / TPCs / SMs | source |
| --- | --- | --- |
| H100 SXM5 (`h100`) | 8 / 66 / 132 | Hopper Architecture In-Depth |
| `gh200-480gb`, `h200`, `h100-nvl`, `h200-nvl` | the same | derived: the same SM count |
| `rtx5090` | 11 / 85 / 170 | RTX Blackwell whitepaper |
| `h100-pcie` | 7 or 8 / 57 / 114 | Hopper Architecture In-Depth (both counts kept) |
| `b200`, `gb200`, `b300` | 8 / 74 / 148 | inferred: full Blackwell GPU, 8 GPCs and 160 SMs (Blackwell Ultra post) |
| `rtx-pro-4500` | 7 / 41 / 82 | inferred: GB203, 7 GPCs and 84 SMs (RTX Blackwell whitepaper) |
| `rtx-pro-6000`, `-max-q` | 12 / 94 / 188 | RTX Blackwell whitepaper |
| `rtx-pro-6000-server` | the same | derived: the same SM count |
| Ampere: `a100*` 7/54/108, `a40`, `rtx-a6000` 7/42/84, `rtx3090` 7/41/82, `rtx3080ti` 7/40/80, `rtx3070` 6/23/46 | | A100 and GA102 whitepapers, Ada whitepaper Table 3 |
| Turing: `t4` 5/20/40, `rtx2080ti` 6/34/68 | | Ada whitepaper Tables 1 and 5 |
| Ada: `rtx4090` 11/64/128, `rtx4080` 7/38/76, `l4` 5/29/58, `l40`, `l40s`, `rtx6000-ada` 12/71/142 | | Ada and RTX Blackwell whitepapers |

The Ampere, Turing and Ada layouts decide nothing (those parts have no clusters); they are recorded because
the figures are published. No device attribute depends on the layout: `cudaDevAttrClusterLaunch` is the only
cluster attribute CUDA has, and it is set from the compute capability. The RTX 3060 and the other profiles
whose GPC count the whitepapers do not give (A10, A10G, A30, RTX A5000 and the like) have no layout, and neither have `gb10`, `thor` and `vr200`.
