# Study 14: can an NPU speed up the simulator? (stub)

Status: stub, 2026-11-17. **Nothing here is measured or built.** It records the question, the reasoning so far, and the smallest experiment that would answer it.

**Question.** Many machines now ship an NPU (AMD XDNA / Ryzen AI, Intel AI Boost, Qualcomm Hexagon, Apple Neural Engine). Can any part of the simulator's work run on one, exact or approximate, and be faster than the CPU path?

## 1. Where the simulator's time goes (from studies 1, 6 and 13)

Interpreting SIMT instructions: dispatch, register rows, integer and floating-point lane arithmetic, memory accesses, counters. None of that maps onto an NPU, which is a fixed-function low-precision matrix engine programmed with a graph or dataflow compiler, not a general core. **Verdict for the bulk of the engine: no.**

## 2. The one candidate: tile matrix ops

The tensor-core class (`mma`, `imma`, `wgmma`, `tile_mma`, `imma_frag`, MFMA/WMMA in the AMD engine) is a small matrix multiply per warp. Two cases:

| operand / accumulate | NPU can do it | bit-exact against the simulator |
| --- | --- | --- |
| INT8 x INT8 into INT32 | yes (all NPUs) | **yes in principle**: integer accumulation has no rounding |
| BF16 / FP16 into FP32 | yes | **no**: accumulation order and rounding differ; not bit-exact against the simulator |
| FP32 / FP64 | no | not applicable |

Library calls (cuBLAS, rocBLAS GEMMs) are the only place a transfer to an NPU could pay for itself: a per-warp tile is far smaller than the cost of moving it. Study 4 already records that library kernels are native; an NPU would be one more engine behind that recognition, not a change to the ISA-level interpreter.

## 3. Constraints

- **Zero dependencies.** Each vendor has its own stack (XDNA: XRT and the amdxdna driver; Intel: OpenVINO; Qualcomm: QNN). Any use would be an opt-in plug-in loaded at run time, never a build dependency.
- **Availability.** Most CI runners and our test machines have no NPU; the experiment needs one machine that does (server3, a Ryzen 7 PRO 8840U, is the candidate; **whether its driver stack is usable is not known**).
- **Latency.** Host-to-NPU transfer and graph launch cost likely exceed the compute of small tiles.
- **Competition.** Host-GPU mode (studies 7 to 12) already sends matrix work to a real card with far better software support.

## 4. The experiment that would settle it

1. Check that an NPU and its driver stack work on one machine (no simulator involved).
2. Run an INT8 GEMM (e.g. 1024 x 1024 x 1024) through the vendor stack and through the simulator's CPU path; compare the INT32 outputs **bit for bit** and the wall times.
3. Repeat for sizes 64, 128, 256 to find the break-even size, if any.

**Exit criteria.** Worth pursuing only if the INT8 results are bit-identical and the NPU beats the CPU path at a size that real library GEMMs reach. Otherwise record the numbers here and close the study as "do not".

## 5. Not in scope

BF16/FP16 approximate offload (a mode that is not bit-exact; decide only after step 2), any use for non-matrix work, and a build-time dependency on any vendor SDK.

## Status log

- 2026-11-17: stub written; priority low until the host-GPU modes are further along.
