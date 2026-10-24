
## 8. Phase 0 result (2026-10-23): `vgpu check` on an A10G

Branch `vgpu-check`: `vgpu check` (`src/cli/check.cpp`, `check_compare.hpp`), `test_check_compare` (9 tests) and `check_exit` (36 cases, CPU-only machine), built with g++ 13 / CUDA 13.2 on a g5.4xlarge (not the CI image). On the card: PolyBench/GPU (20) and Rodinia (11, the round-12 argument lists), sim profile `nvidia/a10g`, card run twice each. **30 of 31 agree** (6 byte-identical, 24 identical after dropping timing lines); the one difference is `srad_v2` (4,096 numbers, up to 30% apart: the out-of-bounds read the card hides, 07 section 3.2). No program needed noise masking. Raw: `data/check/`.
Found while testing: PolyBench's `Non-Matching CPU-GPU Outputs ...: 0` line contains "CPU", so a timing filter drops it and the self-check goes unread. The filter now keeps lines that state an outcome and masks only numbers that carry a time unit; unit-tested with 0 against 5.
Not done: HeCBench and CUDA Samples were not run through `vgpu check`; the build used the host toolchain, not the CI image; the CUPTI shim did not build on the rig (no `cupti.h`), which `check` does not need.
