# Cross-check against MMA-Sim

MMA-Sim (Xie, Xu, Wang, Yang, Yang, "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units", arXiv 2511.10909, MICRO 2026;
github.com/microsoft/MMA-Sim, MIT, Copyright Microsoft Corporation) is a set of Python models of the matrix instructions of NVIDIA
Volta to RTX Blackwell and AMD CDNA1 to CDNA3, which its authors say give bits identical to the cards' (their `tests/equivalence/` run each
model against a kernel of the instruction on a GPU). This page maps its models to the simulator's rules, says what was run, and lists every
disagreement. The vocabulary of `docs/matrix-numerics.md` is kept: *measured* (a card of ours), *published*, *unmeasured*.

**Pinned source:** commit `b1c5b5849a32a32c229a3e4a072b58df2dd50ce6` (2026-11-04, "Fix dropped infinite and NaN accumulators in gtr_fdpa on
CDNA3 BF8 (#25)"), package `mmasim` 2.1.1. Read, not executed: it was cloned into a directory outside the repository and only read.

## What was and was not run

| Run | Status |
|---|---|
| Real MMA-Sim (Python, torch >= 2.9) over the same vectors, compared by `mmasim_xcheck compare` | **NOT RUN.** server2 (the one host this was allowed on) has no torch, and installing it there was out of scope; the WSL machine is barred from tests. `tools/mmasim/mmasim_ref.py` is written for it and is **untested**. Everything below that says "MMA-Sim" is therefore **a C++ transcription of MMA-Sim's arithmetic, made by reading its source** (`arithmetic/fdpa.py`, `ftz_mul_add.py`, `fma.py`, `nv_ptx/sim.py`, `amd/sim.py`), and is only as faithful as that reading. |
| Our rule against the transcription, 54 NVIDIA forms and 11 AMD forms, 220,000 vectors each, two seeds | Run on server2 on 2026-11-18 (load below 4, 13 GB free, `nice -n 19`, one core, one `g++ -O2` of one file, temp dir deleted afterwards). Logs: `docs/numerics/mmasim/run-seed1.txt`, `run-seed2.txt`. |
| The harness's power: our rule made wrong on purpose (`--mut-grid`, `--mut-chunk`) and the transcription made wrong in each of the four corners (`--mut-theirs 1..4`) | Run; every change is seen (`docs/numerics/mmasim/power-test-theirs-mutations.txt`). |

The transcription is not independent of this repository's reading: the same person read the same paper. What the run does show is that
`tc86::mma_wide` and MMA-Sim's `t_fdpa` as written give the same bits, so no hidden difference sits in either code. A real run on a machine
with torch is the missing step: `mmasim_xcheck export <dir>`, `python3 tools/mmasim/mmasim_ref.py <dir>`, `mmasim_xcheck compare <dir>`.

## Terms

| MMA-Sim (`fdpa.py`) | Khattak and Mikaitis | This repository |
|---|---|---|
| `F`: a term is cut toward zero at `2^(e_max - F)`, where `e` of a product is `e_a + e_b` with the product's significand in `[1, 4)`, and of the accumulator its leading bit | "extra alignment bits": the bits kept beyond the 24 of an f32 significand | grid `G`: a term is cut at `2^(top - G)`, where `top` of a product is `exp(a) + exp(b) + 1` and of the accumulator its leading bit + 1 |
| `F = 23 + extra` | `extra = F - 23` | `G = F + 1 = 24 + extra` |
| Volta 23 | 0 | G 24 (no model: a V100 profile would be new) |
| Turing, Ampere, Ada 24 | 1 | G 25 (`Ampere`) |
| Hopper, Blackwell, RTX Blackwell 25 | 2 | G 26 (`Hopper`) |
| `L_max`: the chunk (`min(K, L_max)` products and the accumulator in one step) | block / FMA width | `chunk`, `chunk_tf32` |
| `e_zero` -132 (f32), -20 (f16) Ampere; one lower on Hopper; Volta -131 / -19 | | `zero_acc_floor(G) = -106 - G` (f32), `6 - G` (f16) on `top`, i.e. `e = top - 1`: -132 / -20 at G 25 |
| `rho` `RZ-FP32` / `RNE-FP16` | truncate / RN | `to_f32_rz`, `to_f16_rn` |
| `+ 0.0` after the cut | | "a zero result is +0" (R4) |
| `frexp_and_normalize(c)` with `e_subnormal` -126 / -14 | | accumulator's frame position clamped at -125 / -13 (R3) |
| `ldexp_and_normalize` with `e_subnormal = -126` for RZ | | `finish_f32_rz` cuts onto the subnormal grid (R1) |
| `e[s == 0] = e_zero` for every zero term | | the floor only when the accumulator is zero (R2); with a nonzero accumulator the floor is below the accumulator's own frame (-126 > -132; f16 -14 > -20), so the two forms coincide |

R1 to R4 are the four subnormal corners of `docs/matrix-numerics.md`. The terms agree one for one; no reconciliation beyond `G = F + 1` was needed.

## Mapping, per architecture

`T_FDPA` is MMA-Sim's truncated fused dot-product-add: the NVIDIA rule. Alignment: the largest frame position, every term cut toward zero
at `F` bits below it; summed exactly; result cut (f32) or rounded to nearest even (f16 accumulator); subnormals as above. L is in products
per chunk (`L_max` = bytes per chunk / bytes of an operand).

| Architecture (MMA-Sim name) | Instruction forms in MMA-Sim | F | L f16 / bf16 / tf32 | e_zero f32 / f16 | Our rule | Result of the run |
|---|---|---|---|---|---|---|
| Volta | `mma.m8n8k4` f16 (f32 or f16 accumulator) | 23 | 4 / - / - | -131 / -19 | none shipped: V100 has no profile. The family at chunk 4, G 24 reproduces it | 0 of 440,000 mismatches |
| Turing | `m16n8k8` f16; `m8n8k4` falls back to Volta's numbers | 24 (m8n8k4: 23) | 8 / - / - (m8n8k4: 4) | -132 / -20 | `Ampere` (measured, Tesla T4) for `m16n8k8`; `m8n8k4` none | 0 of 880,000 |
| Ampere | `m16n8k8`, `m16n8k16` f16, bf16, tf32 (`k4`, `k8`) | 24 | 8 / 8 / 4 | -132 / -20 | `Ampere` (measured: RTX 3060, A10G, A100-SXM4-40GB) | 0 of 1,760,000 |
| Ada Lovelace | as Ampere, plus FP8 `m16n8k16/k32` | 24 (FP8: 13, result `RZ-E8M13`) | 8 / 8 / 4 | -132 / -20 | `Ampere` (measured: L4, L40S); FP8 unmeasured, sequential | 0 of 1,760,000 (f16, bf16, tf32); FP8 not compared |
| Hopper | `mma.sync` as Ampere; `wgmma` f16, bf16, tf32, FP8 | 25 (wgmma FP8: 13) | 16 / 16 / 8 (wgmma: 16 / 16 / 8) | -133 / -21 | `Hopper` for `mma.sync` and wmma (measured: H100 PCIe) | 0 of 1,760,000 (`mma`) and 880,000 (`wgmma`) |
| Blackwell (B200) | `mma.sync` as Ampere; `tcgen05.mma` f16, bf16, tf32, FP8, block-scaled | 25 | 16 / 16 / 8 | -133 / -21 | `Hopper` by default since 2026-11-16 (inferred from Khattak and Mikaitis's B200 vectors) | 0 of 1,760,000 (`mma`) and 880,000 (`tcgen05`) |
| RTX Blackwell (sm_120) | `mma.sync` as Ampere; sm_120 `kind::f8f6f4` and block-scaled FP6/FP4 | 25 (`mxf4` k64: G 16 groups, F 35) | 16 / 16 / 8 | -133 / -21 | `Hopper` by default since 2026-11-17 (measured: RTX PRO 6000 Blackwell Server Edition) | 0 of 1,760,000 |

Our numbers are the table of `docs/matrix-numerics.md`: `Ampere` chunk 8 (tf32 4), G 25; `Hopper` chunk 16 (tf32 8), G 26. F, L, e_zero
and rho of the left half are the tables at the top of `nv_ptx/sim.py` (`_mma_F_bits`, `_mma_L_max_bytes`, `_mma_e_zero_for_fp16/fp32`).

AMD. MMA-Sim has four different sums; ours has the double sum (default) and one opt-in rule. `E_FDPA` is exact blocks, `FTZ_MUL_ADD`
pairs, `TR_FDPA` truncate-and-floor in two stages, `FMA` a chain.

| Architecture | Forms | MMA-Sim model | Subnormals | Rounding | Our rule |
|---|---|---|---|---|---|
| CDNA1 (MI100) | f16 (blocks of 4), bf16 (blocks of 2) | `E_FDPA`: a block's products and the accumulator summed exactly, one rounding to f32 | handled | nearest even | no gfx908 target. The default double sum (K products added in order in double, rounded once) |
| CDNA1, CDNA2, CDNA3 | f32 (non-xf32) and f64 | `FMA`: a chain of fused multiply-adds | handled | nearest even per step | f32: the default double sum rounded once (a different sum); f64: a chain of fma, the same |
| CDNA2 (MI250X) | f16, bf16 `_1k`: P = 4; bf16 without `_1k`: P = 2 | `FTZ_MUL_ADD`: inputs and products flushed to zero (product keeps its sign), summed in pairs (P = 2) or pairs of pairs (P = 4) in f32 with every sum flushed, then added to the accumulator, flushed | flushed everywhere | nearest even per add | default: the double sum. Opt-in `VGPU_MFMA_CDNA2=fttn`: every product added to the f32 sum with one rounding in order of K, inputs, running sum and C flushed |
| CDNA3 (MI300X) | f16, bf16 (L 8), xf32 (L 4); FP8 `GTR_FDPA` (L 16, odd and even halves) | `TR_FDPA`: products cut at F 24 below the largest, the sum and the accumulator floored to F2 31, normalised, rounded to f32 | handled (f16: -14, else -126) | nearest even | the default double sum |

## Results

Vectors: 8 classes per form in equal parts (random bit patterns incl. inf and NaN; finite random; exponents in a window of 9 around the bias;
tiny products; products around the smallest f32 or f16 subnormal; the second half of K cancelling the first; special values among window
values; mixed), plus three targeted classes (tiny products all negative; a subnormal accumulator; two to four tiny products and a zero
accumulator, the shape of the corner probes S1 and S2). PRNG splitmix64, seeds 1 and 2; accumulators zero, minus zero, in the window, tiny,
subnormal, random.

**NVIDIA.** 11,880,000 comparisons per seed, 54 forms, **0 mismatches**, both seeds. Per form 220,000 vectors, of which, on a typical form,
about 45,000 lie in R1's neighbourhood, 30,000 in R2's, 27,000 in R3's and 3,600 in R4's (coverage lines in the logs; the f16-input forms have
no R1, and R2 only with a zero accumulator). So the four corners are exercised, not skirted: with the transcription made wrong in one corner
the run shows it (power test: no floor, 50 mismatches in an Ampere bf16 form; rounding a subnormal result, 13,874; a negative zero kept,
600 in an f16-accumulator form; a subnormal accumulator at its leading bit, 7,715), and our rule with `G` one off or the chunk one off
differs in 14 to 41 percent of vectors (a Hopper chunk 17 is clamped to K = 16, so no difference there is expected).

| Corner | Who handles it | Result |
|---|---|---|
| R1 subnormal f32 result cut | both | agree |
| R2 floor under a zero accumulator | both | agree |
| R3 subnormal accumulator at -126 / -14 | both | agree |
| R4 negative sum rounding to zero is +0 | both | agree |

**AMD, ours against the transcription** (seed 1; seed 2 within 0.2 percent). Every form disagrees; none of this is a corner (the corner
columns are NVIDIA-only):

| Architecture and form | Column | Vectors | Mismatches |
|---|---|---|---|
| CDNA1 f16 16x16x16 (E_FDPA, in-window vectors only) | default double sum | 165,041 | 45,731 |
| CDNA1 bf16 16x16x8 (E_FDPA, in-window only) | default | 80,792 | 12,098 |
| CDNA1 / CDNA2 / CDNA3 f32 (FMA chain) | default | 220,000 each | 43,350 / 43,349 / 43,638 |
| CDNA2 f16 16x16x16 (FTZ_MUL_ADD P 4) | default / opt-in | 220,000 | 111,497 / 67,268 |
| CDNA2 bf16_1k 16x16x16 (P 4) | default / opt-in | 220,000 | 105,433 / 78,402 |
| CDNA2 bf16 16x16x8 (P 2) | default / opt-in | 220,000 | 90,263 / 60,523 |
| CDNA3 f16 16x16x16 (TR_FDPA L 8) | default | 220,000 | 64,990 |
| CDNA3 bf16 16x16x16 | default | 220,000 | 55,467 |
| CDNA3 xf32 16x16x8 (L 4) | default | 220,000 | 45,413 |

"In-window": the exact-sum transcription works in 128-bit fixed point and skips a vector whose terms span more than 120 bits (25 percent of
f16 vectors, 63 percent of bf16 ones); it is not MMA-Sim's own `exact_sum`.

## The models of Khattak, Mikaitis and Graziani against the same transcription (2026-11-19)

A second published model per AMD card, opt-in `VGPU_MFMA_MODEL=khattak` ([../matrix-numerics.md](../matrix-numerics.md)), run as one more column of
`mmasim_xcheck` (`g++ -std=c++20 -O2 -I include -I amd/src tools/mmasim/mmasim_xcheck.cpp`, then `run --n 40000 --seed S --filter CDNA --show 3`;
on the pantheonsim.com host, 2 cores, no network). 40,000 vectors a form; the E_FDPA transcription handles only part of the vectors (the "exact
window"). "Finite" is the mismatches in which neither result is an infinity or a NaN. Full output: [mmasim/khattak-run-seed1.txt](mmasim/khattak-run-seed1.txt),
[seed 2](mmasim/khattak-run-seed2.txt). Neither side is a card: this is two published models against each other, as read by us.

| Form | MMA-Sim model | Mismatches, seed 1 (finite) | Seed 2 (finite) |
|---|---|---|---|
| CDNA1 f16 | `E_FDPA` | 0 of 29,959 (0) | 0 of 30,100 (0) |
| CDNA1 bf16 | `E_FDPA` | 5 of 14,771 (0) | 5 of 14,737 (0) |
| CDNA2 f16 | `FTZ_MUL_ADD` P = 4 | 10 (0) | 7 (0) |
| CDNA2 bf16 `_1k` | `FTZ_MUL_ADD` P = 4 | 3,425 (0) | 3,321 (0) |
| CDNA2 bf16 | `FTZ_MUL_ADD` P = 2 | 2,785 (612) | 2,739 (603) |
| CDNA3 f16 | `TR_FDPA` | 10 (10) | 3 (3) |
| CDNA3 bf16 | `TR_FDPA` | 2,878 (2) | 2,737 (5) |
| CDNA3 xf32 | `TR_FDPA` | 2,202 (1) | 2,126 (11) |

What the mismatches are:
- **Finite, CDNA2 bf16 without `_1k` (about 600):** MMA-Sim adds the products of this form in pairs (P = 2); the MATLAB code sums them in fours (its comment says 2, its code does not).
- **Finite, CDNA3 (1 to 11 in 40,000):** one unit in the last place, the two models cut or keep the bits of the smaller operand differently in rare cases (an example: `c = 0.5`, results `3f00001b` and `3f00001a`). Not analysed further.
- **Overflow edge, bf16 and xf32 (2,100 to 3,400):** the sign of an infinity when a product passes 2^128 (the MATLAB code takes `max(prd) * Inf`; with the sign taken from the overflowing product instead, the CDNA2 bf16 `_1k` count falls from 3,425 to 1,854, CDNA3 bf16 from 2,878 to 966 and xf32 from 2,202 to 496, on seed 1; that variant is not in the code), and sums that overflow in MMA-Sim's f32 additions but not in the per-product check of the MATLAB code (NaN against an infinity).
- **A subnormal times an infinity, f16 (CDNA2, 7 to 10) and CDNA1 bf16 (5):** the MATLAB code settles infinities before flushing (infinity), MMA-Sim flushes first (NaN).

Where the two agree, they are two readings of one set of hardware behaviour, not an independent confirmation: both papers' authors say they matched cards.
On the finite results of f16 on CDNA1 and CDNA2 they agree on every vector.

## Disagreements and which side is wrong

1. **AMD default (the double sum) against every MMA-Sim AMD model.** Hypothesis: ours is wrong. The double sum is a placeholder that this
   repository's own page calls unmeasured; MMA-Sim's models are said to be checked against cards, including CDNA1 to CDNA3. The first
   mismatching vectors are one-ulp differences (a chain that rounds per step against one rounding), but also the large ones below.
   *Fix* (not made): implement `TR_FDPA` for gfx942 and `E_FDPA` / `FTZ_MUL_ADD` for gfx908 / gfx90a, and the f32 forms as a chain of fma; they need
   card evidence before they become defaults, like the rest. *A card settles it*: MMA-Sim's own `mmasim_kernels` HIP kernels run an
   instruction on a card; feeding them the vectors `mmasim_xcheck export` writes decides all three architectures at once (an MI300X for CDNA3,
   an MI250X for CDNA2, an MI100 for CDNA1).
2. **The gfx90a opt-in rule against `FTZ_MUL_ADD`** (67,268 of 220,000 on f16). Both rules say "width 1" to FTTN's block test, so FTTN's
   row cannot choose (stated in `docs/matrix-numerics.md`). Hypothesis: one of them is wrong and the card decides: the FTTN motivating example
   (0 for the opt-in rule and the paper's MI250X, 255.875 by hand for MMA-Sim's model) is the shortest probe; MMA-Sim's `E_FDPA` also gives 255.875
   and is the paper's MI100 value. *Fix if MMA-Sim is right*: add `FTZ_MUL_ADD` as the gfx90a rule (pairs flushed, then fours, then C) in place of the
   sequential one.
3. **Overflow.** MMA-Sim's CDNA2 `FTZ_MUL_ADD` form adds in f32, so a bf16 product above the f32 range, or an f32 sum above it, becomes infinity
   in the chain, and `inf - inf` gives NaN where the double sum stays finite or gives the other infinity (vectors in the log: `ours=ff800000
   mmasim=ffc00000`; `ours=7f800000 mmasim=ffc00000`). Hypothesis: the cards do overflow in f32 (a double-wide accumulator would be a design the
   papers do not report), so the double sum is wrong here too; the card settles it with two large bf16 products.
4. **CDNA3 xf32 sign of an underflowed zero** (`ours=80000000 mmasim=00000000`): our double sum rounds a product of about 1e-46 to -0 in f32;
   `TR_FDPA` floors it into +0. Same fix as 1.
5. **The CDNA1 exact sum's zero sign** is the transcription's guess (an exact zero from cancellation is +0, all terms minus zero is -0);
   MMA-Sim's `exact_sum` / `normalize_parts_to_fp32` may differ. Untested; `mmasim_ref.py` run would show.
6. **NVIDIA instructions MMA-Sim models and the simulator does not.** (a) `m8n8k4` on Volta and Turing: no model in the simulator
   (sequential); the family at chunk 4, G 24 matches MMA-Sim on 880,000 vectors, but a V100 and a T4 `m8n8k4` run is what would make it a rule.
   (b) `wgmma` and `tcgen05.mma` f16, bf16, tf32: the simulator keeps the sequential sum ("unmeasured" on the main page) while MMA-Sim gives them the
   Hopper numbers (F 25, L = 32 bytes); the rule on its own agrees with MMA-Sim on these forms (rows `Hopper wgmma` and `Blackwell tcgen05` above), so
   enabling it there needs only a card: an H100 `wgmma` with `mmasim_kernels` vectors, and a B200 for `tcgen05`. (c) Hopper's `wgmma` FP8 (F 13) is
   implemented since 2026-11-19 (it reproduces all 20,000 recorded H100 and H200 vectors; [fp8-vectors.md](fp8-vectors.md)); Ada's FP8 (F 13, result
   `RZ-E8M13`) is `ada` there. Not implemented: Blackwell's FP8 and FP6/FP4 and the block-scaled forms (`ST_FDPA`, F 25, L 32; `GST_FDPA` for k64,
   G 16, F 35): no rule here; unmeasured.
7. **No NVIDIA disagreement** was found. Where our `Hopper` rule is `inferred` (b200, gb200, rtx5090 and the other sm_120 profiles) MMA-Sim
   is the third source that gives the same numbers; it is not a card of ours.

## What is unverified

- The real MMA-Sim has not been run on anything (see the top). The transcription could differ from it where torch behaves in a way read wrongly:
  `torch.frexp` of an infinity or NaN (assumed exponent 0), float32 products of significands (taken as exact), `torch.round` as half-even.
- The AMD columns say "the transcription differs", not which side matches a card; no AMD card was used.
- The vectors are mine, not MMA-Sim's own test vectors (theirs are uniform random 32-bit patterns); the exponent windows and the targeted
  classes are chosen for the corners.
- A result "agrees" only for the forms in the table; sparse, FP8, FP6/FP4, block-scaled, `m8n8k4` on a real Turing, f64 and integer forms were not compared.

## Files

`tools/mmasim/mmasim_xcheck.cpp` (build line and modes in its header), `tools/mmasim/mmasim_ref.py` (the real MMA-Sim side, untested),
`docs/numerics/mmasim/` (logs). Nothing here changes a result of the simulator.
