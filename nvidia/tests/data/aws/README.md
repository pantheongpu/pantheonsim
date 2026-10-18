# CUPTI traces from rented GPUs, against the simulator

Every CUPTI e2e case (`nvidia/tests/e2e/cupti_*.cu`) was built on a rented machine and run
against NVIDIA's own libcupti, 2026-10-18. `nvidia/tools/trace-compare.sh` then ran the same
binaries against the simulator (`VGPU_GPU=nvidia/<profile>`; simulator built from main at
42919cd4 in the CI images for CUDA 13.0 and 12.8) and diffed.

Collectors, one per provider: `trace-aws.sh <type>`, `trace-lambda.py <type>`,
`trace-vast.py "<model>"`, all running `trace-collect-remote.sh` on the machine. Each launches one
instance and terminates it, also on error. `TRACE_CUDA=13-0` pins the toolkit, `TRACE_AMI` picks
an older AWS image, `TRACE_ONLY="case ..."` runs just those cases.

Per card directory: `<case>.txt` card trace, `<case>.sim` simulator trace, `<case>.diff`,
`info.txt`. The tables are `summary-cuda13.0.md`, `summary-cuda12.8.md` and `summary-aws-cuda13.2.md`.

## Cards

| directory | card | toolkit, driver | simulator profile | compared |
|---|---|---|---|---|
| rtx4090 | RTX 4090 (sm_89, **a 48 GB card**, 49140 MiB; the profile says 24 GB) | 13.0, 580.178 (Vast) | rtx4090 | yes |
| rtx5090 | RTX 5090 (sm_120) | 13.0, 595.58 (Vast) | rtx5090 | yes |
| g4dn.xlarge | T4 (sm_75) | 13.2, 595.91 (AWS) | t4 | yes |
| g5.xlarge | A10G (sm_86) | 13.2, 595.91 | a10g | yes |
| g6.xlarge | L4 (sm_89) | 13.2, 595.91 | l4 | yes |
| g6e.2xlarge | L40S (sm_89) | 13.2, 595.91 | l40s | yes |
| g7.2xlarge | RTX PRO 4500 Blackwell (sm_120) | 13.2, 595.91 | rtx-pro-4500 (new) | yes |
| gpu_1x_a10 | A10 (sm_86) | 12.8, 570.148 (Lambda) | a10 | yes |
| gpu_1x_a100_sxm4 | A100 40 GB SXM4 (sm_80) | 12.8, 570.148 | a100-sxm4-40gb | yes |
| gpu_1x_h100_pcie | H100 PCIe (sm_90) | 12.8, 570.148 | h100-pcie | yes |
| g5g.xlarge | T4G (sm_75, arm64 host) | 13.2, 595.91 | none | card only: arm64 binaries do not run on the x86 simulator |
| g4dn.xlarge-cuda13.0 | T4 built with CUDA 13.0 on the AWS machine | 13.0 | | against the 3060's files |

Not collected: V100 (`p3dn.24xlarge` needs an older image and its vCPUs were taken by another
job), RTX PRO 6000 (g7e), A100 (p4d, p4de), H100 SXM (p5), H200 (p5en), B200 and B300 (p6): no
capacity in us-east-1 for 90 minutes. The `trace_driver` re-run (below) could not be repeated on
g6e.2xlarge and g7.2xlarge, whose vCPU quota was in use by another job.

`nvidia/rtx-pro-4500` is a new profile read from the g7.2xlarge with `characterize-aws.sh`
(82 SMs, 32 GiB, PCI 0x2C3A, 165 W). ECC, memory type, PCIe link, L2 size and the GPC layout were
not read and are left out.

## Result

**On CUDA 13.0 with a native Linux driver (RTX 4090, RTX 5090) the simulator matches the card on
12 of the 15 cases that run, and the remaining 3 are explained.** The 16th case, `compat12`, is skipped
by its own program on this toolkit.

Identical: buffers, extcorr, filter, filter_driver, graph, memory, nsys, nvtx, overhead, trace,
trace_driver, um.

Three cases differ, by the same lines on both cards, and each is a **native Linux vs WSL** difference
(the expected files come from an RTX 3060 under WSL):

- `params`, `driver_params`: `cuMemPoolImportFromShareableHandle`, `cudaImportExternalMemory` and
  `cudaImportExternalSemaphore` return 999 natively, 801 under WSL.
  `cudaDeviceFlushGPUDirectRDMAWrites` returns 0 natively, 801 under WSL.
- `resource`: WSL creates one extra stream record, which shifts later stream ids by one.

The simulator answers like WSL unless told otherwise. `VGPU_PLATFORM=linux` (nvidia/src/platform.hpp,
nvidia/docs/cupti.md) switches these three behaviors to the native ones, and then the simulator matches
the RTX 4090 and RTX 5090 on **all 15** cases (`summary-cuda13.0-platform-linux.md`). CI runs the three
cases that differ against `cupti_<case>.linux.expected` as `e2e_cupti_<case>_linux`.

**On CUDA 13.2 (the AWS machines)** four more cases differ (`filter`, `filter_driver`, `memory`,
`um`), and they match again under 13.0 on the same hardware, so they are CUDA 13.2's libcupti, not
the cards and not the simulator: a stream shown on synchronous allocations, an extra callback
name, `CUPTI_SUCCESS` on disabling the counter. Following 13.2 needs a simulator build against the
13.2 headers and a CI leg for it; the CI images have 12.0, 12.8 and 13.0.

**On CUDA 12.8 (Lambda, driver 570)**: `params` and `driver_params` crash on the card (exit 139), as
those cases are written for the CUDA 13 headers, so they have no trace. The rest matches except two
single lines: `nvtx` (the simulator prints one more `nvtxRangePop` callback) and `graph` (one more
resource callback for a cloned node). `compat12`, which targets the 12.x entry points, runs and is
identical. `resource` shows the same platform difference as above.

The result does not depend on the GPU: the same cases differ, by the same number of lines, on
every card that ran the same toolkit (T4, A10G, L4, L40S, RTX PRO 4500 on 13.2; A10, A100, H100 on
12.8; RTX 4090, 5090 on 13.0).

## A mistake that was found and fixed

The first AWS runs showed `trace_driver` differing by 354 lines on every card. That was the
collector, not the simulator: it built every case with `-cudart shared`, but the CI runner builds
`trace_driver` as a pure driver-API program (`-x c++ -cudart none`), and loading the runtime changes
what initializes the driver when. With the build corrected, `trace_driver` is identical on all eight
cards re-run. The AWS tables carry the corrected rows for g4dn, g5 and g6; g6e.2xlarge and
g7.2xlarge keep the old row, marked.

Timing values are not part of these traces; the simulator has no timing model.
