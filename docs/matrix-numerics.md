# Matrix instruction numerics: what is measured, what is published, what is not known

A tensor core or matrix core does not add its products the way `acc += a * b` does, and no two generations do it the same way.
This page says, for every profile, which sum the simulator makes for the floating-point matrix instructions and what that rests
on. Three words are used strictly:

- **measured**: a card of ours ran a program and the simulator reproduces its bits. The card, driver, toolkit and date are given.
- **published**: a paper gives a bit-level model or a feature; no card of ours has checked it.
- **unmeasured**: neither. The simulator then keeps a plain sum that no card is known to make.

Nothing here is a timing model. Integer forms (IMMA, the int8 MFMA forms) are exact wrapping sums everywhere and are not discussed.

## The NVIDIA rule

Every f16, bf16 and tf32 tensor-core form measured so far is one rule with two numbers (`include/vgpu/exec/tc_ampere.hpp`):

- K is cut into chunks of `L` products, in order. A chunk is its exact products and the accumulator, summed in one step: every
  term is aligned to the largest of their frame positions `top` and cut toward zero at `2^(top - G)`, the cut terms are added
  exactly, and the sum is cut toward zero to an f32 (24 significant bits), or rounded to nearest even to f16 when the accumulator
  is f16. The result is the next chunk's accumulator.
- A product's frame position is `exp(a) + exp(b) + 1`, exponents clamped at the format's smallest normal one. The accumulator's
  is its leading bit + 1; a subnormal accumulator counts at the smallest normal exponent of its format. Zero terms take no part,
  except that a chunk starting from a zero accumulator has a floor on `top` (`-106 - G` for an f32 result, `6 - G` for an f16 one).
- An f32 result below 2^-126 is cut onto the subnormal grid. A zero result is +0 whatever the signs. Infinities and NaNs are IEEE
  on exact products, the NaN canonical (0x7fffffff).

| Model (`tc86::SumModel`) | `L` f16, bf16 | `L` tf32 | `G` | In the published models' terms |
|---|---|---|---|---|
| `Ampere` | 8 | 4 | 25 | block 8 / 4, 24 fraction bits, floors -132 / -20 |
| `Hopper` | 16 | 8 (an m16n8k4 is one chunk; a wmma 16x16x8 fragment compiles to two m16n8k4, so chunks of 4) | 26 | block 16 / 8, 25 fraction bits, floors -133 / -21 |
| `Sequential` | 1 | 1 | - | `acc += a * b` in f32, round to nearest even. Not a card's sum. |

`tc86::device_sum_model(compute capability)` is the one table from a device to its model; the engines ask it and compare no sm
number. The device chooses, not the architecture a cubin was built for (an H100 gives its own bits for sm_75, sm_80, sm_86, sm_89
and sm_90 builds of the same program, and an RTX 3060 its own for sm_75 and sm_80 builds).

## NVIDIA profiles

| Compute capability | Profiles | f16 / bf16 / tf32 sum | Status | Evidence |
|---|---|---|---|---|
| 7.5 (Turing) | t4, rtx2080ti | `Ampere` (f16 only: a Turing card has no bf16 or tf32 form) | measured | Tesla T4 |
| 8.0 (GA100) | a100, a100-80gb-pcie, a100-sxm4-40gb, a30 | `Ampere` | measured | A100-SXM4-40GB |
| 8.6 (GA10x) | rtx3060, rtx3070, rtx3080ti, rtx3090, a10, a10g, a40, rtx-a5000, rtx-a6000 | `Ampere` | measured | RTX 3060, A10G |
| 8.9 (Ada) | l4, l40, l40s, rtx4080, rtx4090, rtx6000-ada | `Ampere` | measured | L4, L40S |
| 9.0 (Hopper) | h100, h100-pcie, h100-nvl, h200, h200-nvl, gh200-480gb | `Hopper` | measured | H100 PCIe |
| 10.0, 12.0 (Blackwell) | b200, gb200; rtx5090, rtx-pro-4500, rtx-pro-6000, -server, -max-q | `Sequential` by default; `Hopper` with `VGPU_TC_EXACT=published` | published, not measured by us | MMA-Sim and Khattak-Mikaitis give B200 and RTX PRO 6000 Blackwell the Hopper numbers |
| 10.3, 10.7, 11.0, 12.1, 8.7 | b300, vr200, thor, gb10 | `Sequential` | unmeasured | no source covers them |
| 7.0 (Volta) | no profile | - | published | block 4, no extra bit (`G` 24); a V100 profile would be new, and CUDA 13 cannot build sm_70 |

Everywhere: the f64 form (DMMA) is a chain of IEEE double fused multiply-adds in the order of K, which is what FTTN's fp64
program prints on every card we ran it on (RTX 3060, A10G, L4, L40S, A100, H100: three extra bits, round to nearest even, sticky
bit, width 1) and what the simulator has always done. `mma.m8n8k4`, the sparse forms, the FP8 forms (QMMA), `wgmma` and
`tcgen05.mma` are **unmeasured** and keep the sequential sum (or their own existing rule); the published models cover some of
them and none is on by default.

### Card runs behind the table

All built with `-cudart shared`; the two probes are `nvidia/tests/e2e/simd_lane_forms.cu` (60 lines) and `float_half_forms.cu`
(169 lines), whose RTX 3060 lines are `nvidia/tests/data/*.rtx3060.txt`.

| Card | Where | Driver | Toolkit | Date | What it printed |
|---|---|---|---|---|---|
| RTX 3060 (sm_86) | local | - | CUDA 13.0 | before 2026-10 and 2026-11-15 | 106,496 element-by-element results that fixed the rule; both probe files; FTTN; the two corner inputs S1 and S2 |
| Tesla T4 (sm_75) | AWS | 595.112.07 | CUDA 13.0 | 2026-11-15 | simd_lane_forms built `-DNO80`: the RTX 3060's 56 remaining lines, the four wmma f16 lines among them (`simd_lane_forms.t4.txt`); FTTN fp16. float_half_forms has not run on it |
| A10G (sm_86) | AWS | 595.112.07 | CUDA 13.0 | 2026-11-15 | both probes: the RTX 3060's lines, 60 of 60 and 169 of 169; FTTN |
| L4, L40S (sm_89) | AWS | 595.112.07 | CUDA 13.0 | 2026-11-15 | both probes, sm_86 and sm_89 builds: the RTX 3060's lines; FTTN |
| A100-SXM4-40GB (sm_80) | Lambda | 570.148.08 | nvcc 12.8.93 | 2026-11-15 | both probes: the RTX 3060's lines; FTTN; a dump of 8,388,608 wmma results, every one the `Ampere` rule's |
| H100 PCIe (sm_90) | Lambda | 570.148.08 | nvcc 12.8.93 | 2026-11-15 | both probes, its own lines (`simd_lane_forms.h100.txt`: 7 tensor lines differ from the RTX 3060's; `float_half_forms.h100.txt`: the 14 `mma_` lines of random operands differ, the 30 `mma_edge` cases and all else are equal), the same from sm_80, sm_86 and sm_89 builds; FTTN; a dump of 8,388,608 wmma results, every one the `Hopper` rule's |

The simulator's own code was run over both dumps (`tools/fttn/tc_dump_check.cpp`, server3, 2026-11-15): H100 dump with the
`Hopper` rule 8,388,608 of 8,388,608; A100 dump with the `Ampere` rule 8,388,608 of 8,388,608; the rules crossed 8,366,093 of
8,388,608 each way, so the data tells them apart. What the dumps can and cannot say: their f16 and bf16 operands are mostly
special values (about 500 results a form and style have only finite operands), so the H100's `G` rests mainly on the tf32 form
(20,773 results separate 26 from 25) and its block of 16 on 100 to 450 results a form; the probe lines (`mma.sync` m16n8k16 and
m16n8k8 in f16, bf16 and tf32 on random registers) and FTTN's block test agree.

File hashes of every card output are in `docs/numerics/fttn/cards-probe-sha256.txt`; FTTN's rows per card and cubin are in
`docs/numerics/fttn/cards-fttn.txt`.

### The corners in the subnormal range

Four details of the rule were not reached by the probes that first measured it and were wrong in the simulator until now (the
published models have them; found by comparing the two, pantheonsim issue #664):

| | Corner | Before | Now | Card |
|---|---|---|---|---|
| R1 | an f32 result below 2^-126 | rounded to nearest onto the subnormal grid | cut toward zero onto it | RTX 3060, input S2 (needs R1 and R2): 0x00000041 |
| R2 | a chunk that starts from a zero accumulator, all products tiny | aligned at the largest product | `top` has a floor | RTX 3060, input S1: 0x0000 |
| R3 | a subnormal accumulator | positioned by its own leading bit | counts at 2^-126 (f16: 2^-14) | published models only |
| R4 | a negative sum that rounds to zero | -0 | +0 | published models only |

They change results only for operands near the bottom of a format. The vector kernels keep the tiles where none can arise; a
tile with an f16 accumulator, or with bf16 / tf32 operands small enough for a product below 2^-125, goes to the scalar code.
`tools/fttn/tc_corners.cu` prints all six inputs on a card.

### Switches

`VGPU_TC_EXACT`: unset, each device sums by its model. `0`: the sequential f32 sum on every device, as the earliest releases
(bit for bit; pinned by `simd_lane_forms.*.tcseq.txt`). `force`: a device without a measured model takes `Ampere`.
`published`: a device without a measured model takes the published one where there is one (Blackwell: `Hopper`).

Results that changed with this work, all of matrix instructions: the t4, rtx2080ti, a100*, a30, l4, l40, l40s, rtx4080,
rtx4090 and rtx6000-ada profiles (sequential to `Ampere`); the h100*, h200* and gh200 profiles (sequential to `Hopper`); on the
SASS engine an sm_86 cubin on a profile that is not 8.6 (it took the RTX 3060's sum whatever the profile: now the device's, as
the PTX engine always did) and an sm_75 / sm_80 / sm_89 cubin on an 8.6 profile; and the four corners on every `Ampere` profile.
`VGPU_TC_EXACT=0` gives the sequential sums back; the corners have no switch (the earlier answers were wrong on the card).

## AMD profiles

**No AMD matrix-core hardware was available to us. Nothing on the AMD side is measured**; it is checked against papers only.

| Target | Profiles | f16 / bf16 MFMA, WMMA | f32 MFMA | f64 MFMA | Status |
|---|---|---|---|---|---|
| gfx90a (CDNA2) | mi250x | default: the K products and C summed in double, in order, rounded once. Opt-in `VGPU_MFMA_CDNA2=fttn`: see below | double sum of the instruction's K, rounded once | fused multiply-add chain | unmeasured; opt-in rule from the FTTN paper alone |
| gfx942 (CDNA3) | mi300x, mi325x | double sum, rounded once | same | chain | unmeasured |
| gfx950, gfx1250 | mi350x, mi455x | double sum, rounded once | same | chain | unmeasured; no published model |
| gfx1100, gfx1201 (RDNA3, RDNA4) | rx7900xtx, rx9070xt | WMMA: double sum, rounded once | - | - | unmeasured |

The published bit-level models (MMA-Sim) give CDNA1, CDNA2 and CDNA3 three different sums, none of them the double sum, and the
f32 forms a chain of f32 fused multiply-adds. They are not implemented: no card or published hardware vector checks them, and a
paper-only model does not become a default here.

### The opt-in MI250X rule and what the two papers say

`VGPU_MFMA_CDNA2=fttn` makes gfx90a's f16 and bf16 forms with an f32 accumulator sum as the FTTN paper's MI250X row reads
(subnormal inputs and outputs not handled, three extra bits, round to nearest even, block width 1): a subnormal element of A or
B and a subnormal C are read as zero, each product is added to the running f32 sum with one rounding, a subnormal sum becomes
zero (`amd/src/mfma_math.hpp`; test `test_amd_cdna2_matrix`, through rocWMMA code objects). The paper ran the two 16x16x16
forms; the other f16 / bf16 shapes get the rule by extension. Not fixed by the paper and chosen: when a subnormal running sum
is flushed, and NaN payloads.

It is off by default because the summary row does not fix a rule. MMA-Sim's CDNA2 model (products added in pairs, the pairs in
fours, then to the accumulator, every sum rounded and flushed) gives the same FTTN row: FTTN's block test puts two half-ulps
in neighbouring K and concludes "width 1" from either. The two differ on the paper's own motivating example:

| Sum of `D = C - A x B`, K = 2^13 (the paper's section V) | D |
|---|---|
| one f32 rounding a product (the opt-in rule) | 0 |
| pairs, then fours, each rounded (MMA-Sim's CDNA2 model; computed by hand, not by MMA-Sim) | 255.875 |
| exact blocks of 4, one rounding (MMA-Sim's CDNA1 model) | 255.875 |
| double sum of the instruction's 16, one rounding (the simulator's default) | 192 |
| the FTTN paper: MI250X 0, MI100 255.875 | |

So the paper's MI250X value agrees with the opt-in rule and not with MMA-Sim's MI250X model, and its MI100 value agrees with
MMA-Sim's MI100 model. Only an MI250X can say which is right.

FTTN's programs also say less about the f32 and f64 forms than its table suggests: rocWMMA spreads a 16x16x16 f32 or f64
multiply over four instructions of K = 4 with neighbouring K in different instructions, so its "width 1" for FP32 and FP64 on
AMD is about the chain of instructions, not the sum inside one. The simulator prints the paper's FP32 and FP64 rows already.

Adding an MI100 (gfx908, CDNA1) profile would take a profile file, the gfx908 target in the AMD decoder and executor (not
there today), and a sum for its f16 and bf16 forms.

## Recreating the FTTN paper on the simulator

The paper: X. Li, A. Li, B. Fang, K. Swirydowicz, I. Laguna, G. Gopalakrishnan, "FTTN: Feature-Targeted Testing for Numerical
Properties of NVIDIA & AMD Matrix Accelerators", CCGrid 2024, arXiv 2403.00232; code github.com/LLNL/FTTN (MIT). FTTN's sources
are not in this repository: `tools/fttn/fetch.sh` fetches them at a fixed commit and applies `tools/fttn/fttn.patch` (tile
counts settable on the command line; an sm_75 build without the tf32 and f64 kernels; no test is changed). `motivating.cpp` is
ours: FTTN ships the GEMM of its motivating example but none of its programs calls it.

Its results table against the simulator, per FTTN's own summary row (subnormals / extra bits / rounding / block width):

| Format | Paper | Simulator before | Simulator now | Our card |
|---|---|---|---|---|
| FP16 A100 | yes / 1 / truncate / 8 | a100: yes / 3 / RTN / 1 (different) | same as paper | A100: same as paper |
| FP16 H100 | yes / >= 2 / truncate / >= 16 | h100: yes / 3 / RTN / 1 (different) | yes / 1 / truncate / 16 | H100: yes / 1 / truncate / 16 |
| FP16 V100 | yes / 0 / truncate / 4 | no profile | no profile | not run |
| FP16 MI250X | no / 3 / RTN-TE / 1 | mi250x: yes / 3 / RTN / 16 (different) | default unchanged; opt-in: same as paper | none |
| FP16 MI100 | yes / 3 / RTN-TE / 4 | no profile | no profile | none |
| BF16 A100 | yes / 1 / truncate / 8 | different | same as paper | same as paper |
| BF16 H100 | yes / >= 2 / truncate / >= 16 | different | yes / 1 / truncate / 16 | the same as the simulator now |
| BF16 MI250X | no / 3 / RTN-TE / 1 | yes / 3 / RTN / 16 (different) | default unchanged; opt-in: same as paper | none |
| TF32 A100 | yes / 1 / **RTN-TE** / 4 | different | yes / 1 / truncate / 4 | A100: yes / 1 / truncate / 4 |
| TF32 H100 | yes / >= 2 / truncate / 4 | different | yes / 1 / truncate / 4 | the same as the simulator now |
| FP32 MI250X | yes / 3 / RTN-TE / 1 | same | same | none |
| FP64 A100, H100 | yes / 3 / RTN-TE / 1 | same | same | same |
| FP64 MI250X | yes / 3 / RTN-TE / 1 | same | same | none |

Where the table and FTTN's own programs disagree, the cards side with neither blindly:

- **TF32 on A100, "RTN-TE"**: FTTN's tf32 program prints "Trucate" on every Ampere-rule card (RTX 3060, A10G, A100, L4, L40S)
  and the bits are truncation. The table's cell is not what the program prints.
- **H100, "1 extra bit"**: on our H100 FTTN prints 1 where the paper says two or more. The card has two (the cut is one bit
  lower than Ampere's); FTTN's one-bit test passes and its summary does not report the second.
- **BF16 width "10"**: FTTN prints the width after a hexadecimal dump has left the stream in hex. "10" is 16.

The motivating example (`D = C - A x B`, the real-number answer is 191.984375):

| | Paper | Simulator before | Simulator now | Our card |
|---|---|---|---|---|
| t4, rtx3060, a10g, a100, l4, l40s | A100, V100: 0 | 0 | 0 | 0 on all six |
| h100 | 191.875 | 0 | 191.875 | 191.875 |
| b200, rtx-pro-6000, rtx5090 | - | 0 | 0 (191.875 with `VGPU_TC_EXACT=published`) | none |
| mi250x | 0 | 192 | 192 (0 with `VGPU_MFMA_CDNA2=fttn`) | none |
| mi300x, mi325x, rx7900xtx, rx9070xt | - | 192 | 192 | none |

Every run's row is in `docs/numerics/fttn/simulator-before.txt`, `simulator-after.txt` and `simulator-after-optin.txt`
(both engines for NVIDIA). What did not run: mi350x and mi455x (rocWMMA 6.4.4 has no gfx950 or gfx1250, and a gfx942 binary has
no kernel image for them, which is right); bf16 and fp16 for gfx1201 (rocWMMA 6.4.4 does not compile them); FP32 and FP64 on
gfx1100 / gfx1201 (RDNA has no f32 or f64 matrix instruction: rocWMMA emits none and FTTN prints a row of failures). No program
failed on the simulator for a missing instruction.

Open, found on the way: an sm_75 build of `float_half_forms` folds `__hmul2` and `__half2float` into one `HMUL2.F32`, and the
SASS executor's answer for that line (`half_form_30`) differs from the PTX engine's and from the RTX 3060's line; no Turing card
has run the program, so it is left out of the sm_75 SASS comparison and not changed.

## What a card is still wanted for

`tools/fttn/card-run.sh <out-dir>` runs everything on a rented card (FTTN, both probes against the pinned file of the card's
architecture, the million-result dump, the corners) and makes no cloud call. Then `tools/fttn/fit_tc_model.py <dump>` fits the
rule family, and `tc_dump_check <dump> ampere|hopper` holds the simulator's code to the card. A rule goes into
`device_sum_model` only when that says 100 %.

| Card | Why |
|---|---|
| B200 or RTX PRO 6000 Blackwell (sm_100, sm_120) | no run of ours; the published model is opt-in until then |
| Tesla T4 | `float_half_forms -DNO80` (the open `HMUL2.F32` line; the m16n8k8 mma lines) |
| any Ampere, Ada or Hopper card | the corners R1, R3, R4 (S1 and S2 are on the RTX 3060 only; an H100's S1 and S2 are the published model's word) |
| V100 (sm_70) | block 4 and no extra bit are published; needs a CUDA 12 toolkit and a new profile |
| MI250X, MI300X, MI100 | everything AMD |

## Sources

- FTTN: arXiv 2403.00232, github.com/LLNL/FTTN (MIT).
- MMA-Sim: Xie, Xu, Wang, Yang, Yang, "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units", arXiv 2511.10909 (MIT).
- Khattak, Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004.
