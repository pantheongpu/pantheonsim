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
| g7.2xlarge | RTX PRO 4500 Blackwell (sm_120) | none exists | card traces only |

Not collected: p3 (V100) is not offered to this account in us-east-1. g7e (RTX PRO 6000),
p4d, p4de, p5.4xlarge, p5.48xlarge, p5en, p6-b200 and p6-b300 had no capacity in any zone
at the time of the run.

## Result

Per compared card there are 16 cases. 7 are identical, 1 (`compat12`) is skipped by the
program itself on a CUDA 13.2 toolkit, and 8 differ. **The same 8 differ, by the same number
of lines, on all four cards**, so none of the differences depends on the GPU model
(T4, A10G, L4, L40S).

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

The existing expected files (`nvidia/tests/data/cupti_*.expected`, printed by an RTX 3060
under WSL with CUDA 13.0) agree with the simulator on each of these points. So they are
differences between that oracle and a native-Linux datacenter driver 595 with CUDA 13.2,
not regressions. Which factor causes each one (WSL vs native Linux, or driver and toolkit
version) was not separated; that needs a native-Linux GeForce card, or a WSL run with the
13.2 toolkit.

Timing values are not part of these traces; the simulator has no timing model.
