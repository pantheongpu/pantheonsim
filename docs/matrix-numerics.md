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

Which model a device takes is written in its profile, `numerics.tensor_sum` in `nvidia/profiles/<name>.yaml`, with the numbers of
the model beside it and the class of evidence it rests on (`provenance:`, `evidence:`): **the profile fields are the source of
truth**, and the table below is a summary of them ([nvidia/docs/profiles.md](../nvidia/docs/profiles.md), "Numerics and
provenance", has the schema). `tc86::device_sum_model(profile)` reads the field; the engines ask it and compare no sm number and
no compute capability. (Until the profiles carried it, `tc86::device_sum_model(compute capability)` was the one table from a
device to its model; `test_exec3` keeps that table and holds every profile to it, so the move changed no result.) The device
chooses, not the architecture a cubin was built for (an H100 gives its own bits for sm_75, sm_80, sm_86, sm_89
and sm_90 builds of the same program, and an RTX 3060 its own for sm_75 and sm_80 builds).

## The FP8 forms (`mma.sync` with e4m3 or e5m2 operands)

Until 2026-11-18 these took the sequential f32 sum on every device but sm_120. Recorded vectors (Khattak and Mikaitis, arXiv 2512.07004: 5,000
a file for an RTX 1000 Ada, an L40S, an H100, an H200 and a B200) show three different things, and all three are reproduced bit for bit by both
engines now ([numerics/fp8-vectors.md](numerics/fp8-vectors.md): the files, the fit, the SASS). **No card of ours ran an FP8 form: every rule
here is `inferred`.** Profile key `numerics.tensor_sum_fp8`; `VGPU_TC_EXACT=0` gives the sequential sum back.

| `tensor_sum_fp8` | Devices | What it is | Evidence |
|---|---|---|---|
| `ada` | sm_89: l4, l40, l40s, rtx4080, rtx4090, rtx6000-ada | chunks of 16 products, terms cut toward zero 13 bits below the largest (`grid 14` in the f16 rule's terms), an f32 result cut to 13 fraction bits (an f16 one rounded to nearest even); numbers in the profile | 30,000 of 30,000 recorded Ada and L40S vectors; MMA-Sim's Ada rule |
| `via_f16` | sm_90 and sm_100: the h100, h200 and gh200 profiles, b200, gb200 | ptxas converts to f16 and issues two f16 `HMMA.16816` that start from zero, K split by bit 1 of the index; C is added afterwards by `FADD` (`HADD2` for f16). The device's own f16 rule (`tensor_sum`, Hopper numbers) sums each `HMMA` | 40,000 of 40,000 recorded H100 (f16 accumulator), H200 (f16) and B200 (f32 and f16) vectors; the SASS of both cubins |
| (absent: sequential) | sm_120: its own narrow-float sum, measured, see below; b300, vr200, thor, gb10: unknown | | |

The H100 and H200 f32-accumulator files were recorded through `wgmma` and are not `mma.sync` results; a rule fits all 20,000 of them (one chunk of 32, cut
13 bits down, result cut to 13 bits) and is the `wgmma` FP8 rule since 2026-11-19 (profile key `numerics.tensor_sum_wgmma_fp8: hopper`, `inferred`; dense forms; numerics/fp8-vectors.md). The SASS engine runs a cubin as built, so it
had the right chain on sm_90 and sm_100 already; the PTX interpreter had the sequential sum and disagreed with it on a third of the vectors.

## NVIDIA profiles

| Compute capability | Profiles | f16 / bf16 / tf32 sum | Status | Evidence |
|---|---|---|---|---|
| 7.5 (Turing) | t4, rtx2080ti | `Ampere` (f16 only: a Turing card has no bf16 or tf32 form) | measured | Tesla T4 |
| 8.0 (GA100) | a100, a100-80gb-pcie, a100-sxm4-40gb, a30 | `Ampere` | measured | A100-SXM4-40GB |
| 8.6 (GA10x) | rtx3060, rtx3070, rtx3080ti, rtx3090, a10, a10g, a40, rtx-a5000, rtx-a6000 | `Ampere` | measured | RTX 3060, A10G |
| 8.9 (Ada) | l4, l40, l40s, rtx4080, rtx4090, rtx6000-ada | `Ampere` | measured | L4, L40S |
| 9.0 (Hopper) | h100, h100-pcie, h100-nvl, h200, h200-nvl, gh200-480gb | `Hopper` | measured | H100 PCIe |
| 10.0 (Blackwell) | b200, gb200 | `Hopper` by default (since 2026-11-16; before: `Sequential`) | inferred: data recorded by others on a B200, not run by us | Khattak-Mikaitis's 20,000 B200 vectors: 0 mismatches with `Hopper`, 8,020 with `Sequential`; MMA-Sim gives Blackwell the same numbers |
| 12.0 (Blackwell) | rtx5090, rtx-pro-4500, rtx-pro-6000, rtx-pro-6000-server, rtx-pro-6000-max-q | `Hopper` by default (since 2026-11-17; before: `Sequential`, with `Hopper` opt-in) | measured on rtx-pro-6000-server (RTX PRO 6000 Blackwell Server Edition); inferred on the other four | both probes print the H100's lines, FTTN width 16 and 191.875, the six corners and 8,388,608 of 8,388,608 dump results are the `Hopper` rule's; MMA-Sim gives RTX PRO 6000 Blackwell the same numbers |
| 10.3, 10.7, 11.0, 12.1, 8.7 | b300, vr200, thor, gb10 | `Sequential` | unmeasured | no source covers them |
| 7.0 (Volta) | no profile | - | published | block 4, no extra bit (`G` 24); a V100 profile would be new, and CUDA 13 cannot build sm_70 |

The Status column is per compute capability: a card of that capability was run. Per profile the profile says more, in
`provenance:`. `measured` there means a card of **that model** was run; a profile of another model of the same compute capability
is `inferred`; a profile whose capability no card of ours has run is `unknown` and keeps the sequential sum, except b200 and gb200, whose capability (10.0) a B200 *somebody else* recorded 20,000 vectors on: their Hopper rule is `inferred` from that data (`vgpu info --gpu
<id>` prints the class and the evidence):

| Class of `numerics.tensor_sum` in the profile | Profiles |
|---|---|
| `measured` | rtx3060, a10g, t4 (its tf32 chunk is `inferred`: no tf32 form on Turing), l4, l40s, a100-sxm4-40gb, h100-pcie, **rtx-pro-6000-server** (`hopper`, 2026-11-17) |
| `inferred` | rtx2080ti; a100, a100-80gb-pcie, a30; a10, a40, rtx-a5000, rtx-a6000, rtx3070, rtx3080ti, rtx3090; l40, rtx4080, rtx4090, rtx6000-ada; h100 (the default profile), h100-nvl, h200, h200-nvl, gh200-480gb; **b200, gb200** (`hopper`, from the B200 vectors recorded by Khattak and Mikaitis, 2026-11-16: not a card run of ours, and not run on a GB200); **rtx5090, rtx-pro-4500, rtx-pro-6000, rtx-pro-6000-max-q** (`hopper`, 2026-11-17: from the RTX PRO 6000 Blackwell Server Edition that was run; the same compute capability, other models) |
| `unknown` | b300, vr200, thor, gb10 (nothing published, no card, no recorded data) |

Everywhere: the f64 form (DMMA) is a chain of IEEE double fused multiply-adds in the order of K, which is what FTTN's fp64
program prints on every card we ran it on (RTX 3060, A10G, L4, L40S, A100, H100: three extra bits, round to nearest even, sticky
bit, width 1) and what the simulator has always done. `mma.m8n8k4`, the sparse forms, `wgmma` and
`tcgen05.mma` are **unmeasured** and keep the sequential sum (or their own existing rule); the published models cover some of
them and none is on by default. The FP8 `mma.sync` forms have rules since 2026-11-18, all `inferred` ("The FP8 forms" above).

### Card runs behind the table

All built with `-cudart shared`; the two probes are `nvidia/tests/e2e/simd_lane_forms.cu` (60 lines) and `float_half_forms.cu`
(169 lines), whose RTX 3060 lines are `nvidia/tests/data/*.rtx3060.txt`.

| Card | Where | Driver | Toolkit | Date | What it printed |
|---|---|---|---|---|---|
| RTX 3060 (sm_86) | local | - | CUDA 13.0 | before 2026-10 and 2026-11-15 | 106,496 element-by-element results that fixed the rule; both probe files; FTTN; the two corner inputs S1 and S2 |
| Tesla T4 (sm_75) | AWS | 595.112.07 | CUDA 13.0 | 2026-11-15 | simd_lane_forms built `-DNO80`: the RTX 3060's 56 remaining lines, the four wmma f16 lines among them (`simd_lane_forms.t4.txt`); FTTN fp16 |
| Tesla T4 (sm_75) | AWS g4dn.xlarge | 595.112.07 | nvcc 13.0.88 and 13.2.51 | 2026-11-17 | float_half_forms built `-DNO80`: all 156 lines of that build are the RTX 3060's, `half_form_30` (the HMUL2.F32 line) and the f16 `mma_` lines among them; simd_lane_forms 56 of 56 again; FTTN fp16 and the motivating example (0); the corners S1 `0x0000` and R4 `0x0000` (f16 only); all ten approximate-math operations over 2^32 inputs and the 65,536-pattern half forms, the same as the RTX 3060; `mathfn` lines |
| RTX PRO 6000 Blackwell Server Edition (sm_120) | AWS g7e.2xlarge (us-east-2) | 595.112.07 | nvcc 13.0.88 and 13.2.51 | 2026-11-17 | both probes print the **H100's** lines, 60 of 60 and 169 of 169 (`simd_lane_forms.h100.txt`, `float_half_forms.h100.txt`), the same from sm_86, sm_90, sm_100 and sm_120 builds; FTTN fp16 width 16, bf16 width 10 as FTTN prints it (16), tf32 width 4, fp64 as every card, motivating example 191.875, the same from every cubin architecture; the six corners; a dump of 8,388,608 wmma results, every one the `Hopper` rule's (`tc_dump_check`; `fit_tc_model.py` agrees: chunk 16, grid 26, cut trunc, final rz; the `Ampere` rule 8,366,093); the ten approximate-math operations over 2^32 inputs, the same as the RTX 3060; `mathfn` lines |
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
`docs/numerics/fttn/cards-fttn.txt`. The mutants the unit tests were held to (21, all killed once two tests were added) are in
`docs/numerics/fttn/mutants.txt`.

### The corners in the subnormal range

Four details of the rule were not reached by the probes that first measured it and were wrong in the simulator until now (the
published models have them; found by comparing the two, pantheonsim issue #664):

| | Corner | Before | Now | Card |
|---|---|---|---|---|
| R1 | an f32 result below 2^-126 | rounded to nearest onto the subnormal grid | cut toward zero onto it | RTX 3060, input S2 (needs R1 and R2): 0x00000041; RTX PRO 6000 Blackwell: R1 `0x006b1f35` (cut) |
| R2 | a chunk that starts from a zero accumulator, all products tiny | aligned at the largest product | `top` has a floor | RTX 3060, input S1: 0x0000; Tesla T4 S1 `0x0000`; RTX PRO 6000 Blackwell: S1 `0x0001` and S2 `0x00000042` (the `Hopper` floors), R2 `0x00000000` |
| R3 | a subnormal accumulator | positioned by its own leading bit | counts at 2^-126 (f16: 2^-14) | RTX PRO 6000 Blackwell (the `Hopper` rule): `0x00411fcc`, where the `Ampere` rule gives `0x00411fcd` |
| R4 | a negative sum that rounds to zero | -0 | +0 | Tesla T4 and RTX PRO 6000 Blackwell: `0x0000` (+0) |

They change results only for operands near the bottom of a format. The vector kernels keep the tiles where none can arise; a
tile with an f16 accumulator, or with bf16 / tf32 operands small enough for a product below 2^-125, goes to the scalar code.
`tools/fttn/tc_corners.cu` prints all six inputs on a card.

### Switches

`VGPU_TC_EXACT`: unset, each device sums by its model. `0`: the sequential f32 sum on every device, as the earliest releases
(bit for bit; pinned by `simd_lane_forms.*.tcseq.txt`). `force`: a device without a measured model takes `Ampere`.
`published`: a device without a model in force takes the published one where there is one (none is left: b200, gb200 and the sm_120 Blackwell profiles have `Hopper` in force, so it changes nothing).
**The way back for b200 and gb200** is `VGPU_TC_EXACT=0`, the existing switch: the earlier sequential sum, bit for bit, on them and on every other device (there is no per-device opt-out and none was added).
In the profile's terms: `0` ignores `numerics.tensor_sum`; `force` and `published` act only on a profile whose `tensor_sum` is
`sequential`, giving it `ampere` or its `numerics.tensor_sum_published`.

Results that changed with this work, all of matrix instructions: the t4, rtx2080ti, a100*, a30, l4, l40, l40s, rtx4080,
rtx4090 and rtx6000-ada profiles (sequential to `Ampere`); the h100*, h200* and gh200 profiles (sequential to `Hopper`); on the
SASS engine an sm_86 cubin on a profile that is not 8.6 (it took the RTX 3060's sum whatever the profile: now the device's, as
the PTX engine always did) and an sm_75 / sm_80 / sm_89 cubin on an 8.6 profile; and the four corners on every `Ampere` profile.
`VGPU_TC_EXACT=0` gives the sequential sums back; the corners have no switch (the earlier answers were wrong on the card).

**2026-11-19: the dense FP8 `wgmma` forms changed** on the sm_90 profiles (h100, h100-pcie, h100-nvl, h200, h200-nvl, gh200-480gb) in both engines (PTX `wgmma`, SASS `QGMMA`):
sequential f32 sum to one chunk of 32 products, terms cut 13 bits below the largest, result cut to 13 bits. e4m3 and e5m2, f32 and f16 accumulators. Evidence: 20,000 vectors recorded
on an H100 and an H200, `inferred` (numerics/fp8-vectors.md). `VGPU_TC_EXACT=0` gives the old results back. The sparse forms, `wgmma` f16, bf16 and tf32, and `tcgen05.mma` are unchanged.

**2026-11-18: the FP8 `mma.sync` forms changed** on the sm_89 profiles (l4, l40, l40s, rtx4080, rtx4090, rtx6000-ada: sequential to `ada`) and, on the PTX engine, on the
sm_90 and sm_100 profiles (h100*, h200*, gh200-480gb, b200, gb200: sequential to `via_f16`; the SASS engine's results are unchanged, they were this already). e4m3 and
e5m2, f32 and f16 accumulators, K = 16 and 32. Evidence: recorded vectors, `inferred` (numerics/fp8-vectors.md). `VGPU_TC_EXACT=0` gives the old results back. `wgmma`,
`tcgen05.mma`, the sparse forms and sm_120 are unchanged.

**2026-11-16: Blackwell b200 and gb200 changed too.** Their tensor-core sums (f16, bf16, tf32: wmma, `mma.sync`, SASS `HMMA`) went from the sequential
f32 sum to the Hopper rule by default, so the outputs of a program that uses those instructions on the b200 or gb200 profile change
(not `tcgen05.mma`, `wgmma`, the sparse, FP8 and f64 forms: unchanged). Evidence: the 20,000 vectors Khattak and Mikaitis recorded on
a B200 are reproduced bit for bit by the Hopper rule and 8,020 of them are not by the sequential sum; class `inferred` (others' data,
not a run of ours). `VGPU_TC_EXACT=0` gives the old results back. b300, vr200, thor, gb10 and (until 2026-11-17, below) the sm_120 profiles were unchanged:
no recorded vectors cover them.

**2026-11-17: the compute capability 12.0 Blackwell profiles changed too** (rtx5090, rtx-pro-4500, rtx-pro-6000, rtx-pro-6000-server,
rtx-pro-6000-max-q). Their tensor-core sums (f16, bf16, tf32: wmma, `mma.sync`, SASS `HMMA`) went from the sequential f32 sum to the Hopper
rule by default, so the outputs of a program that uses those instructions on these profiles change (not `wgmma`, the sparse, FP8 and f64
forms: unchanged; sm_120 has no `tcgen05.mma`). Evidence: an RTX PRO 6000 Blackwell Server Edition of ours (AWS g7e.2xlarge, driver 595.112.07, nvcc
13.0.88 and 13.2.51) prints the H100's lines in both probes, the Hopper rule's FTTN rows, the Hopper rule's answer to all six corners and 8,388,608
of 8,388,608 wmma results; the sequential sum is not the card's (6 of the 60 and 13 of the 169 probe lines differ from it, FTTN's motivating example
gives 0 where the card gives 191.875). `rtx-pro-6000-server` is `measured`, the other four are `inferred` from it. `VGPU_TC_EXACT=0` gives the
old results back. b300, vr200, thor and gb10 are unchanged: nothing covers them.

## AMD profiles

**No AMD matrix-core hardware was available to us. Nothing on the AMD side is measured**; it is checked against papers only.
The AMD sums below are in the code (`amd/src/mfma_math.hpp`), not in the profiles: no AMD profile has a `numerics:` section, so
none carries a class for its matrix arithmetic, and none could honestly be anything but `unknown`. (The `provenance:` sections of
`amd/profiles/rx6800.yaml` and `rx6700xt.yaml` tag device limits read from those two cards, which have no matrix instructions.)

| Target | Profiles | f16 / bf16 MFMA, WMMA | f32 MFMA | f64 MFMA | Status |
|---|---|---|---|---|---|
| gfx90a (CDNA2) | mi250x | default: the K products and C summed in double, in order, rounded once. Opt-in `VGPU_MFMA_CDNA2=fttn`, or `VGPU_MFMA_MODEL=khattak`: see below | double sum of the instruction's K, rounded once | fused multiply-add chain | unmeasured; the two opt-in rules are from papers alone, and disagree |
| gfx942 (CDNA3) | mi300x, mi325x | double sum, rounded once. Opt-in `VGPU_MFMA_MODEL=khattak` (f16, bf16, xf32, fp8) | same | chain | unmeasured; the opt-in model is from one paper |
| gfx950, gfx1250 | mi350x, mi455x | double sum, rounded once | same | chain | unmeasured; no published model |
| gfx1100, gfx1201 (RDNA3, RDNA4) | rx7900xtx, rx9070xt | WMMA: double sum, rounded once | - | - | unmeasured |

The published bit-level models (MMA-Sim; Khattak, Mikaitis and Graziani) give CDNA1, CDNA2 and CDNA3 three different sums, none of
them the double sum, and the f32 forms a chain of f32 fused multiply-adds. MMA-Sim's are not implemented (they are transcribed in
`tools/mmasim/`, to compare against). Khattak, Mikaitis and Graziani's are, opt-in, below. No card or recorded hardware vector checks
either, and a paper-only model does not become a default here.

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

### The opt-in models of Khattak, Mikaitis and Graziani (2026-11-19)

Khattak, Mikaitis and Graziani, "Accurate Models of AMD Matrix Cores", arXiv 2609.14845, give MATLAB models of MI100 (CDNA1), MI210
and MI250 (CDNA2) and MI300A and MI300X (CDNA3), refined until they matched hardware bit for bit on random inputs (their claim; we have
not read the full text, and the repository's `model_validation/` holds no AMD vectors). The models are in
`north-numerical-computing/MATLAB-tensor-core` (BSD-2-Clause; read at commit 6fd05208). `VGPU_MFMA_MODEL=khattak` runs them
(`amd/src/mfma_models.hpp`, written from the MATLAB files, not copied; class **published**: no card of ours has run one):

| Model | Form | One block of `fma` products and the running sum |
|---|---|---|
| CDNA1 (MI100) | f16 (`fma` 4), bf16 (2) | exact sum, rounded once to nearest even, subnormals kept. No gfx908 target here: the function is tested, not wired |
| CDNA2 (gfx90a) | f16, bf16 (`fma` 4) | subnormal inputs read as zero, each product rounded to f32, the 4 added as a tree of f32 additions, then the running sum, every sum below 2^-126 flushed |
| CDNA3 (gfx942) | f16, bf16 (8), xf32 (4), fp8 (16) | products aligned to the largest exponent with one extra bit, bits shifted out cut toward zero, summed as integers, then added to the running sum (eight more bits kept when the sum is the bigger; what is shifted out cut toward minus infinity), one rounding to nearest even. fp8: the even-numbered and the odd-numbered products are summed separately, then together |

The blocks of an instruction's K chain through the running sum. On gfx90a the model takes the place of `VGPU_MFMA_CDNA2=fttn`. It
covers the tile path and the reference path; not the sparse, scaled, f32, f64 and integer forms, nor gfx950 (no model). Off by default.

**How it was checked, and what that is not.** MATLAB and Octave were not at hand, so none of it is checked against the paper's own code.
(1) `tools/mfma-models/model_ref.py` is a second writing of the MATLAB files in exact rational arithmetic; the C++ agrees with it on
all 600 pinned vectors (`amd/tests/data/mfma_models/`, CTest `test_amd_mfma_models`) and on 30,000 more (3,000 for each of the ten groups, seed 7,
`tools/mfma-models/mfma_models_check.cpp`): both are one reader's reading. (2) Three vectors worked out by hand are in the test. (3)
rocWMMA's 16x16x16 f16 and bf16 instructions on gfx90a and gfx942 give the model's answers, on both paths. The xf32 and fp8 forms are wired
the same way but only checked as functions, not through an instruction (no code object for them is built here).

**What the models say, against the other sources:**

| `D = C - A x B`, K = 2^13 (the paper's section V) | D |
|---|---|
| CDNA1 model (MI100) | 255.875: the value the FTTN paper reports for an MI100 |
| CDNA2 model (MI250) | 255.875, as MMA-Sim's CDNA2 model; **the FTTN paper reports 0 for an MI250X** |
| CDNA3 model (MI300) | 191.8125 (the real-number answer is 191.984375; the paper gives 191.875 for an H100; no MI300 value) |

So the CDNA1 model reproduces the paper's MI100 number, and both bit-level models of the CDNA2 card (this one and MMA-Sim's) disagree with
the one MI250X result on record. Two papers about the same hardware say different things, and the models' authors claim a bit-for-bit
match on random inputs while FTTN's GEMM goes through rocWMMA (which may not issue the one instruction the models describe). We cannot
tell which; only an MI250X can. That is why neither is a default.

Against MMA-Sim's transcription (40,000 vectors a form, two seeds: [mmasim-crosscheck.md](numerics/mmasim-crosscheck.md)), the models agree on
the results that are finite: CDNA1 f16 0 mismatches, CDNA2 f16 and bf16 `_1k` 0, CDNA3 f16 3 to 10 of 40,000 (one unit in the last place), bf16 2 to 5, xf32 1 to
11; CDNA2 bf16 without `_1k` about 600 (MMA-Sim sums in pairs there, the model in fours, see below). The other mismatches are at the
overflow edge, with inputs near 2^64: about 2,100 to 3,400 of 40,000 for CDNA2 bf16 and CDNA3 bf16 and xf32, and 5 to 10 for CDNA1 bf16 and CDNA2 f16
(a subnormal times an infinity, below).

**What the MATLAB files do that looks unintended** (followed as written; each is a place a card could show the model wrong):
- CDNA2 bf16 without `_1k`: the comment in `MI210MC.m` says `fma = 2` and MMA-Sim has pairs, the code sets 4 for every input format.
- CDNA3 `rd_norm_aware = 1` in `MI300AMC.m` never reaches the arithmetic: `Generic_BFMA_TC.m` reads it only if the field `rd_borrow_carry` exists, and nothing sets that.
- A product past 2^128: both signs give NaN, one sign gives infinity of the sign of `max(prd)`, which is the wrong one when a negative overflow meets a positive product (about half of the overflow mismatches against MMA-Sim).
- Infinities and NaN are settled over the whole row before subnormal inputs are flushed, so a subnormal times infinity is infinity (MMA-Sim flushes first: NaN).
- The sum of two overflowing partial sums is not an overflow check (MMA-Sim's f32 additions overflow and meet the opposite infinity).
- A result of zero is +0 whatever the signs; `GEMM.m` does not list xf32 with tf32 (its CPFloat rounding of inputs would not apply to it).

## Recreating the FTTN paper on the simulator

The table below is the 2026-11-15 one. The run against the current profiles, every NVIDIA profile with a tensor core on both engines, is
[numerics/fttn-simulators.md](numerics/fttn-simulators.md); which numerics of every profile are measured, published, inferred or unknown is
[numerics/per-profile.md](numerics/per-profile.md).

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
| b200 | - | 0 | 191.875 (from 2026-11-16; 0 with `VGPU_TC_EXACT=0`) | none |
| rtx-pro-6000-server | - | 0 | 191.875 (from 2026-11-17; 0 with `VGPU_TC_EXACT=0`) | 191.875 (RTX PRO 6000 Blackwell Server Edition) |
| rtx-pro-6000, rtx5090, rtx-pro-4500, rtx-pro-6000-max-q | - | 0 | 191.875 (from 2026-11-17; 0 with `VGPU_TC_EXACT=0`) | none of these models (the server edition's card says the same of the compute capability) |
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

## Three more papers, run on the RTX 3060 and the simulator

Fasi et al. (PeerJ CS 2021), Khattak and Mikaitis (arXiv 2512.07004, HPEC 2025) and FPRev (Xie et al., USENIX ATC 2025, arXiv
2411.00442) were run the way FTTN was; everything is in [numerics/tcpapers.md](numerics/tcpapers.md) and the programs in
`tools/tcpapers/`. In short: on the RTX 3060 every program prints the same on the card and the simulator (both engines); the
5,000-vector validation sets Khattak and Mikaitis recorded on A100, A2, Ada, L40S, H100 and H200 cards are reproduced bit for bit by
the profile of that card, and a **B200's** by the Hopper rule (0 of 20,000 vectors, against 8,020 mismatches from the sequential sum
the b200 profile had until 2026-11-16, and has again with `VGPU_TC_EXACT=0`), so the b200 and gb200 profiles now use that rule by default
(the sm_120 profiles were on the sequential sum until 2026-11-17, because their data has no RTX PRO 6000; a card of ours has since been run, see the card runs above). FPRev's summation trees of the tensor-core loops are the 9-way (8+1) and 17-way (16+1) fused sums the paper reports
for an A100 and an H100. The one measured disagreement is the simulator's cuBLAS, which adds every product type in one sequential chain
where the card's does not. CI: `e2e_tcpapers` (`nvidia/tests/e2e/run_tcpapers.sh`, pinned files in `nvidia/tests/data/tcpapers/`).

## What a card is still wanted for

`tools/fttn/card-run.sh <out-dir>` runs everything on a rented card (FTTN, both probes against the pinned file of the card's
architecture, the million-result dump, the corners) and makes no cloud call. Then `tools/fttn/fit_tc_model.py <dump>` fits the
rule family, and `tc_dump_check <dump> ampere|hopper` holds the simulator's code to the card. A rule goes into
`device_sum_model` only when that says 100 %: that is, into the profile of the card's model as `numerics.tensor_sum` with the
class `measured` and the run as its evidence, and into the other profiles of its compute capability as `inferred`.

| Card | Why |
|---|---|
| B200 (sm_100) | no run of ours; the Hopper rule is in force on b200 and gb200 as `inferred` from others' recorded vectors, until a card of ours says `measured` (tcgen05 and the GB200 itself are unmeasured) |
| any Ampere, Ada or Hopper card | the corners R1, R3, R4 (S1 and S2 are on the RTX 3060 only; an H100's S1 and S2 are the published model's word) |
| any Ada, Hopper or datacenter Blackwell card | an FP8 `mma.sync` run: `khattak_fp8_vectors` on the card makes the `ada` and `via_f16` rules `measured` (and says whether a driver's own `ptxas` lowers it as ours does) |
| H100 | FP8 `wgmma` with a nonzero accumulator or an f16 accumulator (the rule is implemented from zero-accumulator vectors), and f16, bf16, tf32 `wgmma`: no recorded data |
| V100 (sm_70) | block 4 and no extra bit are published; needs a CUDA 12 toolkit and a new profile |
| MI250X, MI300X, MI100 | everything AMD |

## The other place bits can differ: math functions

The matrix instructions are not the only arithmetic a card does its own way. The math functions (`sinf`, `expf`, `powf`, ...)
are library code over the `MUFU` instructions (NVIDIA) and `v_exp/v_log/v_sin/v_cos/v_rcp/v_rsq` (AMD), which a card reads
from tables. The simulator computes the NVIDIA ones as an RTX 3060 does, bit for bit over every operand (measured; a Tesla
T4, an L4 and an A10G print the same math-library results, and an A10G and an H100 PCIe the same instruction results over
every operand, and an RTX PRO 6000 Blackwell Server Edition the same on all ten as well; `VGPU_MUFU_EXACT=0` gives the
function at host precision, as before), and the AMD ones at the host's precision. What was compared, against which
published results and which cards, what the simulator's own two engines disagreed on, the probe and the model:
[numerics/math-functions.md](numerics/math-functions.md).

## Sources

- FTTN: arXiv 2403.00232, github.com/LLNL/FTTN (MIT).
- MMA-Sim: Xie, Xu, Wang, Yang, Yang, "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units", arXiv 2511.10909 (MIT).
- Khattak, Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004.
- Fasi, Higham, Mikaitis, Pranesh, "Numerical behavior of NVIDIA tensor cores", PeerJ Computer Science 7:e330, 2021;
  github.com/north-numerical-computing/tensor-cores-numerical-behavior (GPL-2.0; run, not copied).
- Khattak, Mikaitis, HPEC 2025 test methodology (github.com/faiziktk/IEEE_HPEC2025_block_FMA_tests, no licence stated; run, not copied) and
  github.com/north-numerical-computing/MATLAB-tensor-core (BSD-2: models and recorded test vectors).
- Xie, Gao, Wang, Xue, "Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations" (FPRev), USENIX ATC 2025,
  arXiv 2411.00442, github.com/peichenxie/FPRev (MIT).
