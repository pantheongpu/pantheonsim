# Round 10b, ffma2: FFMA2, FMUL2 and FADD2 on the SASS engine (sm_100)

Branch `speed10-ffma2` (from `speed-round10`, 52fd1903). Measured on the round 10b rig (c7i.4xlarge, CPUs 0-3 and 8-11 for this workstream, timing
pinned to CPU 1, container image `pantheonsim-ci:cuda13.0`, nvcc/ptxas/nvdisasm 13.0). The other workstream built and ran on the other eight CPUs, so
`uptime` is noted with each timing.

**NOT VERIFIED ON HARDWARE.** No sm_100 card was available (Lambda had no B200, RunPod is not allowed, AWS p6 had no capacity). Everything below is
checked three ways that are not a card: nvdisasm's text for encodings nvcc/ptxas produced, the PTX engine's `f32x2` on the simulator (differential),
and the host's float arithmetic with the rounding mode set. What a B200 does for a NaN's payload, for `.FMZ`, or for a form ptxas never emitted here
is unknown, and the decoder leaves those to the PTX engine (below).

## What was wrong, and the result

`77_blackwell_fmha` built for `sm_100a` stopped the SASS check at `SASS: unknown opcode 0xc49 (sm_100)` (FFMA2 with a uniform-register operand), so
every FMHA kernel ran on the PTX interpreter. FFMA2, FMUL2 and FADD2 are now decoded, printed as nvdisasm prints them, and executed; FMHA runs on
SASS, passes its own `--verify`, and takes a third of the time (table below). `FMNMX3` was already decoded and executed (the measure report listed it
by mistake); it is not touched.

## Encodings: how they were derived

Never from memory: every field below is read off nvdisasm.

1. nvcc/ptxas for `sm_100a` on probes of PTX `fma/add/mul.{rn,rz,rm,rp}{,.ftz}.f32x2` with 64-bit register operands (`data/round10/ffma2/probes/`):
   plain, rounding and flush; halves swapped, either half broadcast, `neg.f32` and `abs.f32` on one or both halves before the operation, kernel
   parameters (which ptxas puts in uniform registers), immediates, a value used twice, predicated. `nvdisasm -c -hex` gives the text and both 64-bit
   halves of each instruction.
2. For every form found, **every single bit of the 128** was flipped and the word given back to nvdisasm (raw `-b SM100a`); the text that changed names
   the field (`probes/bitprobe.py`; output in `bitflips_reg.txt`, `bitflips_forms.txt`).
3. The cubin of `77_blackwell_fmha` built for `sm_100a` (the recipe of `round10-real-workloads.md`; `nvcc -arch=compute_100a -code=sm_100a,compute_100a
   --use_fast_math` of CUTLASS v4.8.0's example): every FFMA2/FMUL2/FADD2 in it, by shape, with its count (`fmha-shapes.txt`).

### Field map (sm_100, sm_100a, sm_103)

| bits | field |
| --- | --- |
| 0-8 | opcode: `0x49` FFMA2, `0x4a` FMUL2, `0x4b` FADD2 (the same low nine bits as sm_120's `UVIMNMX` 0x4a, which has no packed fp32 instruction: the decoder takes these three only for sm_100 to sm_119) |
| 9-11 | the ALU operand form, as seen from ptxas: 1 registers; 2 an immediate third operand (FFMA2) or second (FADD2); 4 an immediate second operand (FFMA2, FMUL2); 6 a uniform register second operand (FFMA2, FMUL2); 7 a uniform register third operand (FFMA2) or second (FADD2). The decoder reads every form 1-7 by the table the scalar ops use (3 and 5, a constant bank, are refused) |
| 12-15 | guard predicate |
| 16-23, 24-31 | destination pair Rd, first operand Ra |
| 32-39 | second operand Rb (a register or a uniform register); 32-63 the immediate (one float) |
| 62, 63 | `\|.\|` and `-` of the operand in the field at 32 (also a uniform register, and FADD2's second operand) |
| 64-71 | the operand in the field at 64 (FFMA2's third operand; the second when the third is an immediate or uniform) |
| 72, 73 | `-` and `\|.\|` of Ra |
| 74, 75 | `\|.\|` and `-` of the operand in the field at 64 |
| 76 | `.FMZ` (FFMA2, FMUL2): not emitted by ptxas for any PTX tried; with `.FTZ` nvdisasm prints `.INVALID3` |
| 78, 79 | rounding: 0 RN, 1 `.RM`, 2 `.RP`, 3 `.RZ`, as FFMA's |
| 80 | `.FTZ` |
| 81 / 82 / 83 | Ra: `.F32x2.LO_HI` (the halves swapped) / `.F32` (one 32-bit register for both lanes) / `.NP` |
| 87 / 88 | Rb of FFMA2 and FMUL2: `.LO_HI` / `.F32` (no `.NP`) |
| 84 / 85 / 86 | the third operand of FFMA2, **and the second operand of FADD2**: `.LO_HI` / `.F32` / `.NP` |
| 91 | set when a uniform register is an operand (nvdisasm refuses the word when it is flipped on a register form) |
| 122, 123, 124 | `.reuse` of a, b, c (a scheduling hint; FADD2's second operand is c) |

Neither swizzle bit set is `.F32x2.HI_LO` (the pair as it is); both set is `.INVALID3`. An immediate prints as the float alone (`2`, `0.5`),
`FFMA2 ... 2.NP` with `.NP`; ptxas sets the `.F32` bit for it. A uniform operand with `|.|` prints as `|UR8.F32x2.HI_LO|` (the bars close after the
swizzle; a vector register's close before: `|R2|.F32x2.HI_LO`). The constant-bank forms (9-11 = 3 and 5) are **not** ptxas output on sm_100 (it loads
parameters into uniform registers with `LDCU`, as the corpus shows for the scalar FFMA, FADD and FMUL), and a hand-made word was refused by nvdisasm;
the decoder refuses them too.

### What the modifiers mean

Settled by the PTX that ptxas lowered (`probes/probe5.cu`, `probe6.cu`: one kernel per variant, so the instruction can be matched to the PTX; listing in
`probe5-sass.txt`, `probe6-sass.txt`):

* The swizzle comes first: `.LO_HI` reads the pair swapped, `.F32` reads one 32-bit register for both lanes (an immediate and a `UR.F32` likewise).
* Then `|.|` and `-` act on the two lanes that come out, `.NP` making the second lane's `-` the opposite of the first's, `|.|` unchanged:
  PTX `neg.f32` of the low half is `-R.NP`, of the high half `R.NP` (no `-`), of both `-R`; `-|R|.NP` is `-|lo|, +|hi|` and `|R|.NP` is
  `+|lo|, -|hi|`. The same four with a swap before or after the negation gave `.LO_HI` forms with `.NP` that agree with this order (the six
  cases of `probe6-sass.txt`). Per-half `|.|` alone (one lane) is not expressible: ptxas emits scalar `FADD R, |R|, -RZ` for it.
* `.FTZ` flushes subnormal inputs and the result; the rounding is for each lane's operation.

## The decoder and executor

* `src/sass/decode.cpp`, `dec_f32x2`: the three instructions, with each operand's swizzle and `.NP` kept in `Instr::f[3]` (4 bits per source: mode,
  `.NP`) and the text in the operand's suffix; `f[0]` the rounding mode, `f[1]` `.FTZ`; `f[4]` set for the forms nothing here knows the meaning of
  (`.FMZ`, both swizzle bits, an immediate without the `.F32` bit). `to_text` matches nvdisasm for all of the corpus lines below.
* `src/sass/exec_ops.inc`, `exec_float`: the executor reads each source as a pair or a broadcast word, applies the swizzle, then `|.|`, `-` and `.NP`, and
  calls `vgpu::exec::f32x2_pair` (new, `include/vgpu/exec/f32x2.hpp`) under the host rounding mode; `executes()` runs all of them except the forms with
  `f[4]`, which stay with the PTX engine (`unsupported()` names them).
* **The two engines share the lane arithmetic.** The PTX interpreter's `OpF32x2` code was moved into that header unchanged and both call it. One
  behaviour changed, deliberately: a NaN result is now the canonical `0x7fffffff`, as the scalar f32 operations of both engines give. Before, the PTX
  engine returned whatever NaN the host's `fmaf`/`+`/`*` made, which differs with the operand order the compiler happens to choose: the first
  differential run (SASS against PTX over NaN operands) differed in 113,269 of the 2048 x 4 x 194 = 1,589,248 results, all of them NaN payloads or signs (SASS code
  and PTX code are the same helper compiled into two places). **A parent decision:** this changes PTX `f32x2` NaN results (only NaN results). No
  card measurement exists for FFMA2; the scalar `FFMA`/`FADD`/`FMUL` of the cards measured (an RTX 3060) give `0x7fffffff`.
* Counters: the PTX engine counts `OpF32x2` as `InstClass::Fp32`, one thread instruction per lane per instruction; FFMA2/FMUL2/FADD2 are `Fp32` too, so
  a kernel's fp32 instruction class counts one per packed instruction on either engine (SASS has fewer instructions overall than the PTX that does the
  swaps and signs with `mov.b64`, `neg.f32`, `abs.f32`).
* **The tier does not take them, by measurement** (the task said to add a tier handler if the tier handles FFMA/FMUL/FADD): `VGPU_TIER=1
  VGPU_TIER_STATS=1` on FMHA case 1 shows 308,325,403 warp instructions of which the tier ran 279,413,737 (90.6 %), the interpreter 21,958,002 straight-line
  ones; **FFMA2 16,384 + FMUL2 16,384 + FADD2 17,152 = 49,920 warp instructions, 0.016 % of the run**. A handler would change nothing measurable;
  the tier declines the form (`Form::None`) and the interpreter runs it, with identical bits (the end-to-end test runs the tier on). The tier's
  interpreter share is `VIADD` (21.8 M, 7 %): the next thing to look at for FMHA, not this workstream's.

## Tests

| test | what it holds | seen to fail on |
| --- | --- | --- |
| `test_sass_decode` (existing, now over the new corpus lines) | 685 lines per corpus (sm_100, sm_100a, sm_103a): what nvcc emitted for `nvidia/tests/e2e/packed_f32x2.cu`, the nine shapes of FMHA's cubin, and 577 variants of the modifier bits that nvdisasm named (`INVALID3`, `.FMZ`, `.NP` with `.F32`, `\|UR\|` forms ...); each decodes, prints as nvdisasm printed it, and is run by the executor or refused on purpose | breaks 5, 6, 9 |
| `tests/unit/test_f32x2_sass.cpp` (new, in `ctest`) | the corpus lines decode and `runs_instr`; a cubin of them has nothing `unsupported()` refuses (before: `SASS: unknown opcode 0xc49 (sm_100)`); forms of unknown meaning are decoded, printed and refused by name, also an immediate with its `.F32` bit cleared; sm_120 keeps `UVIMNMX` at the same low opcode; the shared arithmetic against IEEE 754 by hand (ties, four rounding modes, overflow, flush, zeros, the canonical NaN) | breaks 5, 6, 7, 9 |
| `nvidia/tests/e2e/packed_f32x2.cu`, `e2e_packed_f32x2` (new; also in `e2e_sass_archs`) | nvcc/ptxas for sm_100 and sm_100a: 194 operand-form slots (every rounding mode with and without `.ftz`; each of 15 modifiers on every operand of FMA/ADD/MUL; combinations; parameter pairs, a parameter's word for both lanes, two kinds of immediate in every position; a value used twice) x 2048 threads x 4 launches over zeros of both signs, subnormals, infinities, NaNs with payloads, the largest and smallest finite values; plus 24 known answers worked out by hand. **The SASS engine, the PTX engine (`VGPU_SASS=0`) and the tier (`VGPU_TIER=1`) must print identical results, and the SASS run must not fall back to PTX**, and the results must equal the host's arithmetic (NaNs as NaNs) | breaks 1 to 8 (1-4 and 8: the SASS results differ from PTX's and from the host's; 5 and 6: the SASS run falls back to PTX; 7: SASS and PTX differ in NaN payloads) |
| `test_decode_same_bits` | the digests of the decoder's output over the corpora and over flipped and random words were regenerated: see "same bits" below | n/a |

"Seen to fail" is `data/round10/ffma2/deliberate-breaks.txt`: one build with each break behind an environment variable (not in the committed code),
every test run under each. Break 0 is the control (all pass); 1 `.NP` ignored, 2 `.LO_HI` ignored, 3 no rounding mode, 4 no `.ftz`, 5 FADD2's second
operand's swizzle bits read from FFMA2's place, 6 FADD2 not dispatched, 7 NaN not canonicalised, 8 `-` before `|.|`, 9 sm_120 decoded like sm_100.
Each is caught by at least one test (results in the file); breaks 1 to 4 and 8 only by the end-to-end test, which needs nvcc.

### Same bits

With the new corpora in place, the decoder's full dump (`VGPU_DECODE_DUMP`, `test_decode_same_bits`) of the unmodified build (`build-base`) and of this
one over 1,111,608 (word, architecture) pairs -- the corpora, 120,000 bit-flipped words, random words, every architecture -- differs in exactly **9,576** pairs, and every one of them is a word with low opcode bits 0x49 (4,862 pairs), 0x4a (2,386) or 0x4b (2,328) and a non-zero
operand form, on sm_100 or sm_103 (the two architectures of the list that now decode them); none outside (a script over the two dumps, `dumpdiff.sh` in the
notes, classified every differing line; the first try found the text of `|UR56|.F32` of the half-precision ops changed by my own printing change, which
became the per-operand `bars_after` flag, and the op numbers shifted by three, which is why the new ops come last in `ops.inc`).
The digests in `test_decode_same_bits.cpp` were regenerated for this change (the corpora grew by 685 lines each in three files, and the decoder
now decodes the three opcodes on sm_100 and sm_103): **the parent should expect a conflict in that table with any other branch that changed the
corpora or the decoder.** The new ops are appended to `ops.inc` so that no existing op's number moves.

## FMHA, SASS against the PTX fallback

`77_blackwell_fmha`, the four cases of `run_cutlass_fmha.sh` (`--verify`, `--iterations=0`), built for `sm_100a` as `-code=sm_100a,compute_100a`
(SASS and PTX in one binary), simulated B200, one host thread pinned to CPU 1 of the rig, `build` (this branch) and `build-base` (`speed-round10`)
alternated three times per case (wall seconds, whole process, `uptime` load 1.0 except where noted):

| case | base: runs the PTX engine (s) | this branch: SASS (s) | ratio | result |
| --- | ---: | ---: | ---: | --- |
| 1: b1 h4 q512 k512 d128, no mask | 97.5 (97.4-97.5) | 33.3 (33.2-33.7; the first run at load 23) | 2.93 | `[OK]` both |
| 2: same, causal | 83.8 (83.8-83.9) | 28.5 (28.5-30.9) | 2.94 | `[OK]` both |
| 3: b2 h4 h_k2 q512 k512 d64 | 86.3 (86.3-86.4) | 30.6 (30.6-30.7) | 2.82 | `[OK]` both |
| 4: varlen, causal+residual, d64 | 137.8 (137.8-137.8) | 41.5 (41.5-41.6) | 3.32 | `[OK]` both |

Raw output: `data/round10/ffma2/fmha-pairs.txt`. The base log says `running PTX instead of SASS: ... SASS: unknown opcode 0xc49 (sm_100)` in every run; this
branch's says `running SASS: an sm_100 cubin`. Nothing is left unsupported in FMHA's cubin: all four cases run all their kernels on SASS (no other opcode
stood behind FFMA2).

With the tier (`VGPU_TIER=1`, case 1, one run, not repeated): 26.9 s; 90.6 % of the warp instructions in the tier (above).

## What is not known, not measured, left

* **Everything about the hardware** (see the top). Specifically: the NaN a card returns from FFMA2 (the canonical `0x7fffffff` here is the scalar
  instructions' and the PTX engine's scalar rule); `.FMZ` (refused); an immediate or a swizzle outside what ptxas emitted (refused); whether the lower
  lane's rounding or flush has any quirk the PTX engine's `f32x2` shares (the two engines agree with IEEE 754 and with the host in every case run, so
  if the PTX `f32x2` is wrong, then so is the host's `fmaf` under each rounding mode, which no part of this suggests; no reason was found to doubt the lower lane).
* sm_103 is decoded (its corpus has the lines) but only sm_100 and sm_100a were executed; the decoder takes the three opcodes for every architecture from sm_100 up to (not including) sm_120, which includes any sm_10x it is given and was not tried.
* The tier does not run these (0.016 % of FMHA's warp instructions); no tier timing beyond the single run above. FMHA's tier fallbacks are
  dominated by `VIADD` (7 %), which is not this change.
* Not measured: PyTorch or any other workload on Blackwell SASS that uses these instructions (none was available); the counters of a FMHA run
  against the PTX engine's beyond the instruction class.
* The `data/` files: `bitflips_*.txt` (nvdisasm on every flipped bit), `probes/` (the nvcc sources), `fmha-shapes.txt`, `fmha-f32x2-lines.txt` (the nine
  FMHA shapes in corpus format), `variants.py` and `fmha_patterns.py` (how the corpus lines were made), `deliberate-breaks.txt`, `fmha-pairs.txt`,
  `fmha-tier-stats.txt`.

## How the corpus lines were made (reproducible)

```
nvidia/tools/sass-corpus.py --per-shape 1 <arch> nvidia/tests/data/sass/<arch>.txt <arch cubin of probes/packed_f32x2.cu> fmha-f32x2-lines.txt
docs/perf-design/data/round10/ffma2/variants.py <arch> <corpus> variants.txt      # nvdisasm names each flipped-bit variant
nvidia/tools/sass-corpus.py --per-shape 1 <arch> ... variants.txt
```
for `sm_100`, `sm_100a` and `sm_103a` (nvcc 13.0), and only the 685 lines of the packed instructions were appended to each checked-in corpus (the
tool renumbers every address, which would have turned a 685-line change into a 9,000-line one). The probe is `nvidia/tests/data/sass/probes/packed_f32x2.cu`
(it includes the end-to-end program), so `nvidia/tools/sass-corpora.sh` keeps producing them.
