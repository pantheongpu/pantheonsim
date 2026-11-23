# AMD: the same instruction on different architectures

Found 2026-11-23 by running one-instruction kernels (`amd/tests/data/xarch.c`, built for gfx90a, gfx942, gfx950, gfx1030, gfx1100,
gfx1201 and gfx1250 by `amd/tests/data/build.sh`; `amd/tests/unit/test_amd_cross_arch.cpp`) on the same 4,761 edge-case operand
pairs (zeros of both signs, denormals, infinities, NaNs with payloads, the ends of the half and float ranges) and diffing the
targets. `VGPU_XARCH_DUMP=<dir> build/test_amd_cross_arch` writes every target's results for a diff of your own. The census covers about 120 instruction
forms: f32, f16 and f64 arithmetic, minimum and maximum, the transcendentals, conversions, the integer byte and dot forms, 8-bit floats,
the cube-map instructions, the wave-wide forms (ballot, lane reads and permutes, DPP, `mbcnt`) and one f16 matrix instruction.
Where two targets differ, the question is always which is the hardware's; the answer is one of three kinds:

* **(a) a simulator bug**: a target is wrong against a card, a guide or its sibling targets. Fixed, with the card or guide named.
* **(b) a real difference, modelled**: the test gives each architecture's value.
* **(c) no evidence**: left as it was, tagged `unknown` or `inferred` in the issue, with the card that would settle it.

## Found and fixed

| Issue | Finding | Evidence | Class |
|---|---|---|---|
| #721 | `v_sin_f32`/`v_cos_f32`: gfx90a, gfx1100, gfx1201 took `sin(2 pi x)` in a double; gfx1030 only had the whole-turn sine; the card keeps the fraction of a turn for any argument and has exact zeros at half turns (sine) and odd quarter turns (cosine) | RX 6800: `tools/numerics/amd/gfx1030.vectors.txt` and `gfx1030.trig64ths.txt`; CDNA3/4/5 guides | (a); measured gfx1030, inferred others |
| #722 | `v_rcp/rsq/sqrt/exp/log_f32` kept denormal operands and results; the card reads and writes zero | RX 6800 and RX 6700 XT, all 2^32 inputs | (a); measured gfx1030/1031, inferred others |
| #723 | `v_cube*_f32` refused on every RDNA target; `v_sin_f16`, `v_cos_f16` refused on gfx1201 | RDNA2, RDNA3 and RDNA4 ISAs list them | (a); documented |
| #724 | gfx1201 ran RDNA4's `_num` minimum, maximum and median with the legacy rules (`max(+0,-0)` was -0, a signalling NaN beat a number); gfx1100's signed zero was the host's | RDNA4 ISA ("IEEE maximumNumber()"); RX 6800 for gfx1030, inferred for gfx1100 | (a); documented gfx1201, inferred gfx1100 |
| #725 | RX 6900 XT profile disagreed with the measured RX 6800 and RX 6700 XT (same gfx1030) on four limits and the grid limit | HIP device properties of both cards | (a); inferred |
| #726 | FP8 FNUZ (gfx942) against OCP (gfx950, gfx1201, gfx1250); `v_dot2` denormal flush (gfx942, gfx950); wave-width forms; which DPP and permute forms exist | CDNA3, CDNA4, RDNA4 guides | (b); asserted per architecture |

## Not settled: needs a card

| Issue | What | Card that settles it |
|---|---|---|
| #727 | The NaN a result takes (payload, sign, which of two NaN operands) is the card's on gfx1030 only; the others return what x86 returns | MI210/MI250X, MI300X, MI350X, RX 7900 XTX, RX 9070 XT |
| #728 | `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64` are 24-bit on gfx1030/1031 and exact in the simulator | the RX 6800 on server2 for gfx1030/1031 (a model still has to be found); the others need their cards |
| #729 | The low bits of the f32 transcendentals | the hash probe on each architecture's card |
| #730 | Matrix-core accumulation, every AMD profile; no MI100 (gfx908) profile | MI210/MI250X, MI300X, MI350X, RX 7900 XTX, RX 9070 XT, MI100 |

Per-profile limits that no card read (`max_grid_dim`, `registers_per_block`, `shared_mem_per_sm` of the gfx11, gfx12 and CDNA profiles)
are in #725.

## What was not run

The census compares single instructions. The rocBLAS, hipBLASLt, PyTorch, Ollama and vLLM jobs were not re-run per architecture for
this round: their outputs are the product of the instructions above and of library algorithm choices that the simulator
leaves to the library, which sees the profile's `gcn_arch`. The next step is to diff the digests of a few deterministic library calls
per architecture once CI has run the PR.
