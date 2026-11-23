# FTTN on the simulator, every NVIDIA profile with a tensor core, both engines

Issue #664, first item. FTTN (Li, Li, Fang, Swirydowicz, Laguna, Gopalakrishnan, CCGrid 2024, arXiv 2403.00232, github.com/LLNL/FTTN, MIT)
was fetched, patched and built by `tools/fttn/fetch.sh` and `build.sh` (tile counts settable, no test changed) and run on the simulator at
the current `main` with `tools/fttn/run-sim.sh`: each profile with the binaries of the architecture it was built for (and the profile's own
architecture where there is one; the device chooses its sum, not the cubin), on both NVIDIA engines (the PTX interpreter, `VGPU_SASS=0`, and
the SASS executor, `VGPU_SASS=1`). Every run's line is in [fttn/simulator-main.txt](fttn/simulator-main.txt); `tools/fttn/sim_table.py` made
the table below from it. FTTN's summary row is *subnormals / extra bits / rounding / block width*, per format; the last column of
the motivating example is D of `D = C - A x B` at K = 2^13 (the real-number answer is 191.984375). The same programs ran on cards of ours
(RTX 3060, Tesla T4, A10G, L4, L40S, A100, H100 PCIe, RTX PRO 6000 Blackwell Server Edition): their rows are in
[fttn/cards-fttn.txt](fttn/cards-fttn.txt), and the last column says whether the simulator's rows are those.

## The paper's Table IV (NVIDIA and AMD rows), as the paper prints it

| Format | A100 | H100 | V100 | MI250X | MI100 |
|---|---|---|---|---|---|
| FP16 | yes / 1 / truncate / 8 | yes / >= 2 / truncate / >= 16 | yes / 0 / truncate / 4 | no / 3 / RTN-TE / 1 | yes / 3 / RTN-TE / 4 |
| BF16 | yes / 1 / truncate / 8 | yes / >= 2 / truncate / >= 16 | - | no / 3 / RTN-TE / 1 | - |
| TF32 | yes / 1 / RTN-TE / 4 | yes / >= 2 / truncate / 4 | - | - | - |
| FP32 | - | - | - | yes / 3 / RTN-TE / 1 | - |
| FP64 | yes / 3 / RTN-TE / 1 | yes / 3 / RTN-TE / 1 | - | yes / 3 / RTN-TE / 1 | - |

Where the paper's table and the programs it ships disagree, the cards we ran side with the program, and say so in
[../matrix-numerics.md](../matrix-numerics.md): the A100 TF32 "RTN-TE" is "Trucate" in the program's output on every Ampere-rule card; the
H100's "2 or more extra bits" prints as 1 (FTTN's one-bit test passes, and its summary does not report the second); FTTN prints the BF16
block width in hexadecimal after a hex dump, so "10" is 16.

## The simulator, NVIDIA (current main)

| Profile | Cubin | FP16 | BF16 | TF32 | FP64 | Motivating D | Engines | Same as the card's row |
|---|---|---|---|---|---|---|---|---|
| rtx2080ti | sm_75 | yes / 1 / truncate / 8 | - | - | - | 0 | same | yes (t4, 2 rows) |
| rtx3060 | sm_75 | yes / 1 / truncate / 8 | - | - | - | 0 | same | yes (t4, 2 rows) |
| t4 | sm_75 | yes / 1 / truncate / 8 | - | - | - | 0 | same | yes (t4, 2 rows) |
| a100 | sm_80 | yes / 1 / truncate / 8 | yes / 1 / truncate / 8 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 0 | same | yes (a100, 5 rows) |
| rtx3060 | sm_80 | yes / 1 / truncate / 8 | yes / 1 / truncate / 8 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 0 | same | yes (a100, 5 rows) |
| a10g | sm_86 | yes / 1 / truncate / 8 | yes / 1 / truncate / 8 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 0 | same | yes (rtx3060, 5 rows) |
| rtx3060 | sm_86 | yes / 1 / truncate / 8 | yes / 1 / truncate / 8 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 0 | same | yes (rtx3060, 5 rows) |
| l4 | sm_89 | yes / 1 / truncate / 8 | yes / 1 / truncate / 8 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 0 | same | yes (l4, 5 rows) |
| l40s | sm_89 | yes / 1 / truncate / 8 | yes / 1 / truncate / 8 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 0 | same | yes (l4, 5 rows) |
| gh200-480gb | sm_90 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | yes (h100, 5 rows) |
| h100 | sm_90 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | yes (h100, 5 rows) |
| h100-pcie | sm_90 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | yes (h100, 5 rows) |
| h200 | sm_90 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | yes (h100, 5 rows) |
| b200 | sm_100 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | no card of ours of that capability |
| b300 | sm_100 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | 0 | same | no card, no published model (the sequential sum) |
| gb200 | sm_100 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | no card of ours of that capability |
| vr200 | sm_100 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | 0 | same | no card, no published model (the sequential sum) |
| thor | sm_110 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | 0 | same | no card, no published model (the sequential sum) |
| gb10 | sm_120 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | yes / 3 / RTN / 1 | 0 | same | no card, no published model (the sequential sum) |
| rtx-pro-6000 | sm_120 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | yes (rtx-pro-6000-server, 5 rows) |
| rtx-pro-6000-server | sm_120 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | yes (rtx-pro-6000-server, 5 rows) |
| rtx5090 | sm_120 | yes / 1 / truncate / 16 | yes / 1 / truncate / 16 | yes / 1 / truncate / 4 | yes / 3 / RTN / 1 | 191.875 | same | yes (rtx-pro-6000-server, 5 rows) |

Reading it:

- **Every profile that has a card of ours of its compute capability prints that card's rows**: sm_75 (T4), sm_80 (A100), sm_86 (RTX 3060,
  A10G), sm_89 (L4, L40S), sm_90 (H100 PCIe), sm_120 (RTX PRO 6000 Blackwell Server Edition), on all five programs, on both engines, and the
  two engines print the same everywhere. The rows are those of the paper's A100 for sm_75 through sm_89 (FP16/BF16 1 / truncate / 8, TF32 4)
  and of the paper's H100 for sm_90 (16; the paper's "2 or more bits" is the card's and the simulator's second bit); the motivating example
  is 0 on the first group and 191.875 on the second, as on the cards and as the paper has it for A100 and H100.
- **b200 and gb200** print the Hopper rows too, but no card of ours has run them: that is the rule the profile names (`inferred`, from the
  B200 vectors others recorded, docs/numerics/tcpapers.md), not a measurement.
- **b300, vr200, thor, gb10** print the rows of the plain sequential f32 sum (3 extra bits, round to nearest, width 1, motivating example
  0): no card, no paper and no recorded data covers compute capabilities 10.3, 10.7, 11.0 and 12.1. That is the simulator's placeholder,
  not a claim about those parts.
- **What changed since the 2026-11-15 table** (`fttn/simulator-after.txt`): b200 and the sm_120 profiles printed the sequential row then; they
  took the Hopper rule on 2026-11-16 and 2026-11-17. h100-pcie, h200, gh200-480gb, gb200, rtx-pro-6000-server, thor, gb10, b300 and vr200
  were not in that run. The sm_75 cubin of the rtx3060 profile and the sm_80 one print the same rows as its own sm_86 build.
- **AMD** was not rerun: nothing on the AMD matrix path changed after the 2026-11-15 run (`git log` of `amd/src/mfma_math.hpp` and
  `amd/profiles/`), and its rows are in `fttn/simulator-after.txt` and `simulator-after-optin.txt`. The default MI250X rule (sum in double,
  in order) prints yes / 3 / RTN / 16 against the paper's no / 3 / RTN-TE / 1; the opt-in `VGPU_MFMA_CDNA2=fttn` prints the paper's row.
  No AMD matrix hardware of ours has run FTTN, so nothing AMD is `measured`.

## What was not covered

- V100 (the paper's block of 4, no extra bit): no profile and CUDA 13 cannot build sm_70.
- MI100 (the paper's block of 4, 3 bits, RTN-TE): no gfx908 profile or target.
- The FP8 forms and `wgmma`: FTTN has no FP8 program. The FP8 `mma.sync` and dense FP8 `wgmma` forms are checked against recorded vectors in
  [fp8-vectors.md](fp8-vectors.md).
