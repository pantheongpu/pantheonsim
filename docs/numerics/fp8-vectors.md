# The e4m3 and e5m2 `mma.sync` forms against recorded vectors

Issue #664. Khattak and Mikaitis ("Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004; data in
github.com/north-numerical-computing/MATLAB-tensor-core, `model_validation/`, BSD-2-Clause) recorded 5,000 FP8 dot products a file on an
RTX 1000 Ada and an L40S (e4m3 and e5m2, f32 accumulator; f16 accumulator on the Ada), and on an H100, an H200 and a B200 (f32 and f16
accumulators). Each is a 32-term vector: `mma.sync m16n8k32`, the vector in row 0 of A and column 0 of B. The FP8 files were not run before
([tcpapers.md](tcpapers.md) listed them under "not done"). Nothing here was recorded by us: **every FP8 rule below is `inferred`**, never
`measured`, and no card of ours ran an FP8 form of `mma.sync`.

`tools/tcpapers/khattak_fp8_vectors.cu` is our program (it runs a vector as an `mma.sync` on whatever device the CUDA runtime gives it and
compares bits), `khattak_convert.py` writes the files, `fit_fp8.py` fits the chunk, cut and rounding of the common rule to a file, and
`nvidia/tests/e2e/run_tcpapers.sh` holds 24 vectors a file to the simulator in CI (`VGPU_KHATTAK_VECTORS=<dir>` runs all 5,000).

## Before and after, on the simulator

Mismatches of the 500 first vectors of each file, `main` at 82b0edb0 (before), and of all 5,000 (now), both engines. "PTX" is the PTX
interpreter, "SASS" the SASS executor running the cubin `nvcc` made.

| File (vectors recorded on) | Profile | Before, SASS | Before, PTX | Now, SASS and PTX |
|---|---|---|---|---|
| Ada e4m3, e5m2, f32 accumulator | l40s, rtx6000-ada | 500 | 500 | 0 of 5,000 |
| Ada e4m3 / e5m2, f16 accumulator | l40s, rtx6000-ada | 118 / 138 | 118 / 138 | 0 of 5,000 |
| L40S e4m3, e5m2, f32 accumulator | l40s | 500 | 500 | 0 of 5,000 |
| H100 e4m3 / e5m2, f16 accumulator | h100 | 0 | 179 / 149 | 0 of 5,000 |
| H200 e4m3 / e5m2, f16 accumulator | h100 profile, h200 | 0 | 185 / 176 | 0 of 5,000 |
| B200 e4m3 / e5m2, f32 accumulator | b200 | 0 | 181 / 184 | 0 of 5,000 |
| B200 e4m3 / e5m2, f16 accumulator | b200 | 0 | 183 / 164 | 0 of 5,000 |

(The h200 profile's rows are those of h100; `run_tcpapers.sh` runs the H200 files on h200.) 70,000 vectors, both engines.

The SASS engine was already right on sm_90 and sm_100 and the PTX interpreter was not: they disagreed on a third of the vectors. The cause is
in the cubin. `cuobjdump -sass` of the same `mma.sync` built with nvcc 13.0.88 (the CI image's):

```
sm_89   QMMA ...                                   a native FP8 tensor-core instruction
sm_90   F2FP.F16.E4M3.UNPACK_B  x4   HMMA.16816.F32 R12, R4, R12, RZ    (k = 0 1 4 5 8 9 ..., C = 0)
        F2FP.F16.E4M3.UNPACK_B  x2   HMMA.16816.F32 R4, R4, R16, R12    (k = 2 3 6 7 10 11 ..., C = the first)
                                     FADD R11, R4, R11                  (C added on its own)
sm_100  the same, and HMMA.16816.F16 ... HADD2 for an f16 accumulator
```

So on Hopper and datacenter Blackwell an FP8 `mma.sync` is a conversion to f16, two f16 `HMMA`s starting from zero, K split by bit 1 of the index
(an operand register holds four bytes; the first `HMMA` takes each thread's first two, the second its last two), and then C added by a
separate f32 `FADD` (`HADD2` for an f16 accumulator). The device's f16 rule (chunks of 16, Hopper numbers) sums each `HMMA`; that is
`numerics.tensor_sum_fp8: via_f16`. The PTX interpreter now does the same; the SASS engine always did. Caveats: this is what *our* `ptxas` (nvcc
13.0.88) emits; a card's driver compiles PTX itself, and we have not seen its code. The recorded vectors agree with the lowering above on 40,000 of
40,000 f16- and f32-accumulator vectors of B200, H100 and H200 (the f16-accumulator ones for H100 and H200), which is the evidence for it.

Ada has a native FP8 instruction and its own sum: chunks of 16 products in order, each term cut toward zero 13 bits below the largest term's
exponent (a product's exponent is the sum of its factors' exponents), the exact sum cut toward zero to a 14-bit significand (f32), or rounded to
nearest even to binary16; the f16 accumulator goes in rounded to binary16, as the paper's own program converts it. This is MMA-Sim's Ada rule
(Xie et al., arXiv 2511.10909: F = 13, block 16, result RZ to e8m13) written out again, and it reproduces the vectors:
`fit_fp8.py` over the first 300 vectors of each Ada and L40S file gives `L=16 F=13 rz13` 300 of 300 (the next best rules: 240 and 239),
over 150 vectors of each f16 file (c rounded to binary16 first) 150 of 150, and the simulator's `ada` sum reproduces all 30,000 Ada and L40S vectors on both engines (table above).
MMA-Sim also gives a zero term the exponent -132 (-20 for an f16 accumulator); with FP8 operands every product's exponent is at least -12 and
an accumulator's at least -126 (f16: -14), so that floor is below every other term and cannot change a result: it is left out.

## `wgmma` with FP8 on Hopper (2026-11-19)

The H100 and H200 f32-accumulator files were recorded with a zero accumulator (the paper's own validation script zeroes C for them: `wgmma`, not
`mma.sync`): neither the Ada nor the `via_f16` rule fits them (the simulator's `mma.sync` gives 500 and 307 mismatches of 500, with and without the
file's C). A rule of the same family does: **one chunk of 32 products, terms cut 13 bits below the largest, result cut to 13 fraction bits**
(`fit_fp8.py ... --zero-c --L 32 --n 5000`: **5,000 of 5,000** vectors of each of H100 e4m3, H100 e5m2, H200 e4m3 and H200 e5m2; the next best grid, 14, gives
4,133, 4,548, 4,133 and 4,548). It is MMA-Sim's Hopper `wgmma` FP8 rule. Since 2026-11-19 the simulator's `wgmma` (PTX engine) and `QGMMA` (SASS engine) use it for the
dense e4m3 and e5m2 forms with an f32 or f16 accumulator, on the sm_90 profiles (`numerics.tensor_sum_wgmma_fp8: hopper`, class **inferred**: the vectors are
others' and no card of ours ran a `wgmma`). It is `fp8_dot` with `kHopperWgmmaFp8Rule` (chunk 32, grid 14, 13 result bits), the accumulator joining the chunk as it
does on Ada. `tests/unit/test_wgmma_fp8.cpp` holds 48 of the recorded vectors (those the sequential sum gets wrong) to the rule and to a real `wgmma.mma_async`, both PTX routes. `nvidia/tests/e2e/run_wgmma_fp8_vectors.sh` runs 24 vectors of each file through a real `wgmma.mma_async` m64n8k32 on both engines (ptxas's `QGMMA` and the PTX engine); with `VGPU_KHATTAK_VECTORS` it runs all 5,000 of each: **0 mismatches of 20,000 on both engines**, 2026-11-19 (the sequential sum, `VGPU_TC_EXACT=0`, misses 9 to 13 of each 24).

What is not known: the rule is checked with a zero accumulator only (a non-zero accumulator joining the chunk is MMA-Sim's model, not something these files show);
an f16 accumulator has no recorded data for `wgmma`; the sparse FP8 forms keep the sequential sum; `wgmma` f16, bf16, tf32 and `tcgen05.mma` have no recorded data at all.

## What the data also says, not implemented

- **sm_120.** The B200 vectors do not describe an RTX PRO 6000 Blackwell: the sm_120 profiles give 164 to 183 mismatches of 500 on them. The card
  of ours measured the narrow-float sum there (exact sum cut toward zero: 18 of 27 forms bit for bit, nine not).
- **Sparse forms, the FP6 and FP4 forms, block-scaled forms on sm_90 and sm_100:** no recorded data.

## Evidence files

[fp8/before-main-82b0edb0-500-vectors.txt](fp8/before-main-82b0edb0-500-vectors.txt) (the simulator before: the first 500 vectors of each file, both engines, with and
without the file's C), [fp8/run_tcpapers-full-sets.txt](fp8/run_tcpapers-full-sets.txt) (`run_tcpapers.sh` with the full vector sets, after: every line), [fp8/fit_fp8-output.txt](fp8/fit_fp8-output.txt)
(the fits), [fp8/mutants.txt](fp8/mutants.txt).
