# CUPTI traces from rented AWS GPUs, against the simulator

Collected 2026-10-18 in us-east-1 with `nvidia/tools/trace-aws.sh <instance-type>`
(one instance at a time, each terminated by the script). Every CUPTI e2e case
(`nvidia/tests/e2e/cupti_*.cu`) was built on the rig with its CUDA 13.2 toolkit
and run against NVIDIA's own libcupti (driver 595.91.07). `nvidia/tools/trace-compare.sh`
then ran the same binaries against the simulator (`VGPU_GPU=nvidia/<profile>`, CI
image CUDA 13.0, build of main at 42919cd4) and diffed.

Per instance directory: `<case>.txt` is the card trace, `<case>.sim` the simulator
trace, `<case>.diff` the difference, and `info.txt` the device. `summary.md` is the table.

| instance | card | profile used | compared |
|---|---|---|---|
| g4dn.xlarge | Tesla T4 (sm_75) | t4 | yes |
| g5.xlarge | A10G (sm_86) | a10g | yes |
| g6.xlarge | L4 (sm_89) | l4 | yes |
| g6e.2xlarge | L40S (sm_89) | l40s | yes |
| g5g.xlarge | T4G (sm_75, arm64 host) | none | card traces only: arm64 binaries do not run on the x86 simulator |
| g7.2xlarge | RTX PRO 4500 Blackwell (sm_120) | rtx-pro-4500 (added with this work) | yes |

Not collected: p3 (V100) is not offered to this account in us-east-1. g7e (RTX PRO 6000),
p4d, p4de, p5.4xlarge, p5.48xlarge, p5en, p6-b200 and p6-b300 had no capacity in any zone
at the time of the run.

## Result

`nvidia/rtx-pro-4500` is a new profile read from this g7.2xlarge with `characterize-aws.sh`
(82 SMs, 32 GiB, PCI 0x2C3A, 165 W). It is `verified: true` for what the script reads; ECC, memory
type, PCIe link, L2 size and the GPC layout were not read and are left out. Its traces behave like the
other cards'.

Per compared card there are 16 cases. 7 are identical, 1 (`compat12`) is skipped by the
program itself on a CUDA 13.2 toolkit, and 8 differ. **The same 8 differ, by the same number
of lines, on all five compared cards**, so none of the differences depends on the GPU model
(T4, A10G, L4, L40S, RTX PRO 4500).

Identical: buffers, extcorr, graph, nsys, nvtx, overhead, trace.

Differing (card vs simulator), on every card:

- `driver_params`, `params`: `cuMemPoolImportFromShareableHandle`, `cudaImportExternalMemory`
  and `cudaImportExternalSemaphore` return 999 on the card, 801 on the simulator.
  `cudaDeviceFlushGPUDirectRDMAWrites` succeeds on the card, 801 on the simulator.
- `memory`: synchronous allocations report `stream=set` on the card, `stream=none` on the simulator.
- `um`: disabling the counter returns `CUPTI_SUCCESS` on the card, `CUPTI_ERROR_NOT_READY` on the simulator.
- `filter`, `filter_driver`: the simulator lists one or two extra callback names.
- `resource`: the card has one fewer stream record than the simulator, which shifts later stream ids by one.
- `trace_driver`: the simulator emits a `cuInit` callback in the setup phase that the card does not.

## Why they differ: the toolkit, mostly

The existing expected files (`nvidia/tests/data/cupti_*.expected`) were printed by an RTX 3060
under WSL with CUDA 13.0, and the simulator agrees with all of them. To find out whether the
differences above come from the toolkit or from the platform, the same cases were built with
CUDA 13.0 on a native T4 (`TRACE_CUDA=13-0 nvidia/tools/trace-aws.sh g4dn.xlarge`; traces in
`g4dn.xlarge-cuda13.0/`) and compared with the 3060's files. Result:

- **Toolkit (13.2 vs 13.0), not platform:** `filter`, `filter_driver`, `memory`, `um` and
  most of `trace_driver` match the 3060 exactly under 13.0. The simulator is right for 13.0;
  CUDA 13.2's libcupti behaves differently (stream shown on synchronous allocations, an
  extra callback name, `CUPTI_SUCCESS` on disabling the counter, `cuInit` ordering).
  Following 13.2 needs a simulator build against the 13.2 headers and a CI leg to match;
  the CI image has no 13.2 toolkit.
- **Device, not behavior:** `trace` and `trace_driver` still differ by one line each: the
  device record names the T4 instead of the 3060.
- **Platform (native Linux vs WSL), still there under 13.0:**
  - `params`, `driver_params`: the three import calls (`cudaMemPoolImportFromShareableHandle`,
    `cudaImportExternalMemory`, `cudaImportExternalSemaphore`) return 999 natively, 801 under
    WSL; `cudaDeviceFlushGPUDirectRDMAWrites` returns 0 natively, 801 under WSL.
  - `resource`: WSL creates one extra stream record, which shifts later stream ids by one.

The simulator follows the WSL answers for the platform items. Making it follow native Linux
needs a platform setting (and a second set of expected files); that was not done here.

Timing values are not part of these traces; the simulator has no timing model.
