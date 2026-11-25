# Math functions and scalar floating point: the simulator against published results

The tensor and matrix cores are one place where a simulated card's bits can differ from a real one's
([matrix-numerics.md](../matrix-numerics.md)). The math functions are the other, and the larger one: `sinf`, `expf`, `powf`
and the rest are library code (NVIDIA's libdevice, AMD's OCML) built on a few hardware instructions that approximate a
function (`MUFU.RCP/RSQ/EX2/LG2/SIN/COS/TANH` on NVIDIA; `v_rcp/v_rsq/v_exp/v_log/v_sin/v_cos` on AMD). Until 2026-11-15 the
simulator computed those instructions at the host's precision; a card reads them from tables that are within an ulp or two
of that, so a program that calls a math function printed other last bits on the simulator than on the card. The NVIDIA
instructions are now computed as an RTX 3060 computes them, bit for bit over every input ([The card's unit](#the-cards-unit-mufu-as-an-rtx-3060-computes-it));
the AMD ones are still the function at host precision: measured on two RDNA2 cards in 2026-11-16, not modelled ([the AMD section](#the-amd-instructions-on-rdna2-cards-measured-not-modelled-2026-11-16)).

This page records the second group of published work of issue #664, in the order it was done: what was compared, how, what
the simulator's side of the comparison found and what the published sources said a card would give (written before any card
was run, and kept as written, with the outcome noted where a card has since decided a question), then
[the card runs](#the-card-runs-2026-11-15), the probe of the instructions themselves, the model fitted to it and its
evidence. Three words are used strictly, as in matrix-numerics.md: **measured** (a card of ours ran it), **published** (a
paper or post gives it; no card of ours has checked it), **unmeasured** (neither). A fourth is used for the model on
architectures no card of ours ran: **inferred** (measured on Turing, Ampere and Ada, and given to the others unchanged).

What changed in the simulator: `MUFU` (every one of its ten sub-operations) in the SASS executor, and the PTX
interpreter's `.approx` and `div.full` forms, give the card's bits. `VGPU_MUFU_EXACT=0` gives the earlier results (the
function at host precision, rounded once). The AMD math instructions are untouched.

## The test set (`tools/numerics/`)

Our own code; nothing is copied from the sources. The Varity generator (MIT) is fetched when a run needs it.

| File | What it is |
|---|---|
| `mathfn.cu` | 44 functions in single and double precision (the trigonometric, hyperbolic, exponential and logarithmic functions and their inverses, `pow`, `sqrt`, `rsqrt`, `cbrt`, `erf`, `erfc`, `lgamma`, `tgamma`, `sinpi`, `cospi`, the rounding functions, `fmod`, `remainder`, `hypot`, `atan2`, `ldexp`, `frexp`, `1/x`, `x/y`, `1/sqrt(x)`), each over four classes of input. One source for nvcc and hipcc. |
| `published.tsv` | The published inputs `mathfn` evaluates as its `worst` class: 252 rows, with the error and the library each was published for. |
| `fp_examples.cu` | The worked examples of Whitehead and Fit-Florea, with the answers the paper documents. |
| `varity_cases.cu` | The two case studies of the Varity GPU paper, with the paper's literals and with the same values read from memory. |
| `varity/fetch.sh`, `gen.py`, `build.sh`, `run.sh` | 150 double and 150 float random programs from LLNL's Varity (commit f5e04921, seed 20261115), each built `-O0`, `-O3`, `-O0` fast-math and `-O3` fast-math and run over three input sets. |
| `build.sh` | Builds the three programs for an architecture: the default build, `mathfn_fast` (`-use_fast_math` / `-ffast-math`), `fp_examples_nofma`, `varity_cases` at `-O0` and `-O3`. |
| `run-sim.sh` | Runs everything on the simulator, one directory a configuration (profile, cubin architecture, engine). |
| `card-run.sh` | Builds and runs everything on a real card with the machine's own compiler. No cloud call. |
| `mufu_kernel.cu`, `mufu_probe.cu` | The probe of the approximate instructions themselves: one `MUFU` sub-operation over all 2^32 inputs on a card, as hashes, as a dump of a range, or checked against the simulator's model ([The probe](#the-probe-one-instruction-every-input)). |
| `mufu-card-run.sh`, `mufu/rtx3060.hashes.txt` | The probe's hash pass for another card: ten operations, all inputs, about a second each, compared with the RTX 3060's hashes without shipping dumps. |
| `mufu_forms.cu` | Every PTX approximate form (`rcp/rsqrt/sqrt/sin/cos/ex2/lg2/tanh.approx`, with and without `.ftz`, the double forms, `div.approx` and `div.full`) on a card, on the operands where ptxas puts more than one instruction around the form. |
| `amd_trans_probe.cpp`, `amd-trans-card-run.sh`, `amd/gfx1030.hashes.txt`, `amd/gfx1030.vectors.txt` | The same probe for the AMD instructions (`v_rcp/rsq/sqrt/exp/log/sin/cos_f32` over all 2^32 inputs as hashes, a check against a double reference, a dump, and the f64 forms on a sample), the card runner, and the RX 6800's hashes (the RX 6700 XT's are identical) and 77 selected operands ([the AMD section](#the-amd-instructions-on-rdna2-cards-measured-not-modelled-2026-11-16)). |
| `compare.py` | Compares two runs: per function, precision and class, how many results differ and by how many ulp; with `--truth`, each run's error against the true value beside the published figure. |

`mathfn`'s classes:

- `worst`: the published inputs of `published.tsv` for the function (every input, whichever library it was published for).
- `edges`: 284 values a function (zeros, subnormals, the overflow and underflow thresholds of the exponentials, huge
  trigonometric arguments, integers and half-integers around 2^23 and 2^52, the gamma function's poles, infinities, quiet
  and signaling NaNs with payloads); for a two-argument function all 3,844 pairs of a list of 62.
- `rand_bits`: 131,072 uniformly random bit patterns (every exponent; NaNs too).
- `rand_mid`: 131,072 values of random sign and mantissa with the exponent uniform in [-12, 7].

It prints one line a function, precision and class: the count, an FNV-1a hash of every result's bits, and the first inputs
and results (all of them for `worst`), so two runs are compared line by line; `--dump` also writes every result, which is
what `compare.py` counts ulp over. The pattern is that of `nvidia/tests/e2e/simd_lane_forms.cu`.

```
tools/numerics/build.sh nvidia <bin> 75 86            # needs nvcc; amd: needs hipcc
tools/numerics/run-sim.sh <bin> <build> <out> nvidia/t4:75:sass nvidia/t4:75:ptx amd/mi300x:gfx942
tools/numerics/compare.py <out>/nvidia.t4.75.sass <out>/nvidia.t4.75.ptx
tools/numerics/card-run.sh <card-out>                 # on the card
tools/numerics/compare.py <out>/nvidia.rtx3060.86.sass <card-out>/sm_86 --truth
```

## Sources, and how far each was checked

| Source | Checked | Used for |
|---|---|---|
| Gladman, Innocente, Mather, Ozaki, Zimmermann, "Accuracy of Mathematical Functions in Single, Double, Double Extended, and Quadruple Precision" | Read: the 2.42 edition ([PDF](https://homepages.loria.fr/PZimmermann/papers/glibc242.pdf): CUDA 12.8 on a Grace-Hopper system, ROCm 6.2.4 on an MI50) and the paper's repository ([gitlab.inria.fr/zimmerma/math_accuracy](https://gitlab.inria.fr/zimmerma/math_accuracy) at c2b897fb, 2026-09-06), whose data files `latex/results_cuda` and `latex/results_rocm` hold the newer runs (CUDA 13.3.1 on a T4, ROCm 7.2.4 on an MI300X; single precision exhaustive). The 2.43 PDF (February 2026) was fetched and has the same CUDA and ROCm versions as 2.42. | The `worst` class: 246 rows, the largest known error of each function and the input that gives it. |
| Zahid, Laguna, Le, "Testing GPU Numerics: Finding Numerical Differences Between NVIDIA and AMD GPUs", [arXiv 2410.09172](https://arxiv.org/abs/2410.09172) | Read (the PDF). Its generator, [LLNL/Varity](https://github.com/LLNL/Varity) (MIT), was cloned and read; the published repository writes CUDA programs that pass host arrays to the kernel and has no HIP output, so the paper's own GPU version of it is not what is public. | `varity_cases.cu`; the generator run. |
| N. Juffa, NVIDIA developer forum | Read: the thread the issue names ([MUFU.RCP and MUFU.RSQ replacements](https://forums.developer.nvidia.com/t/weekend-project-exploring-the-feasibility-of-replacing-mufu-rcp-and-mufu-rsq/346566): its ulp figures are for his replacement code, not for `MUFU`), and, through a fetch-and-summarize tool that quotes the page (the forum rate-limited direct reads), the [expf](https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528), [exp question](https://forums.developer.nvidia.com/t/question-about-cuda-exp-function/237910), [tanf](https://forums.developer.nvidia.com/t/accuracy-optimized-implementation-of-tanf-without-performance-impact/219720) and [erff](https://forums.developer.nvidia.com/t/accuracy-optimized-implementation-of-erff-without-performance-impact/218114) threads. The quoted figures below are from those summaries and were not re-read on the page itself. | The library's largest errors over time; one concrete input with its result on two toolkits. |
| Laguna, Gopalakrishnan, "Finding Inputs that Trigger Floating-Point Exceptions in GPUs via Bayesian Optimization" (Xscope, SC22) | **The paper was not read.** [LLNL/Xscope](https://github.com/LLNL/Xscope) (MIT) was cloned: it is the search tool and the list of the 59 CUDA functions it searches; it holds no list of found inputs. | Nothing is taken from it. The `edges` class is our own list of the kinds of input the tool looks for (results that are NaN, infinite or subnormal). The tool itself needs a card and was not run. |
| Whitehead, Fit-Florea, "Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs" | Read: [the web edition](https://docs.nvidia.com/cuda/floating-point/) and the original PDF, which alone prints the dot-product vectors and results. | `fp_examples.cu`. |

## What the simulator's side found

Run on one build machine (the simulator only), 2026-11-15: NVIDIA programs built by nvcc 13.0.88, AMD programs by hipcc of
ROCm 6.4.4. NVIDIA: the profiles t4, rtx3060, a10g, l4, h100 and rtx-pro-6000, each with an sm_75 build and with a build
for its own architecture (sm_75, 86, 86, 89, 90, 120), on both engines. AMD: mi300x and mi325x (gfx942), rx6800 and rx6900xt
(gfx1030).

### Results do not depend on the profile

- NVIDIA: every line of every program is the same on all six profiles for one build and one engine, and the SASS lines of
  the sm_75, sm_86, sm_89 and sm_90 builds are the same lines. No profile carries a math table today, so this is what is
  expected; the test holds it. (An sm_75 cubin on a profile of another generation prints the PTX interpreter's lines whichever
  engine is asked for: only t4 runs the sm_75 SASS, so for that build t4's SASS lines differ from the other profiles' on the
  three NaN lines below and on nothing else.)
- NVIDIA sm_120: the sm_120 build prints the other builds' lines in the default build. Under `-use_fast_math` its `log`, `log2` and
  `log10` (double) of the three negative subnormal edges are NaN where the other builds give -infinity, on both engines: the
  compiler's code for sm_120 differs, not the profile.
- AMD: mi300x and mi325x print the same lines, and rx6800 and rx6900xt. gfx942 against gfx1030 (two different code
  objects) differ in the default build on NaNs only: the sign of the NaN `acos`, `acosh`, `cosh`, `exp2`, `log2`, `log1p`
  and `erfc` return, and `ceil`, `floor`, `trunc` and `frexp` of a signaling NaN (returned as it came for gfx942, quieted
  for gfx1030). Every result that is a number is the same on both. Under `-ffast-math` they differ on 87 of 329 lines, in
  numbers too (`sin` and `cos` of huge arguments, `sinh`, `asinh`, `acosh`, `atanh`, `log1p`): the two compilers' code
  differs and so does the simulator's `v_sin_f32`/`v_cos_f32` for the two targets. Which of it a card shares is unmeasured.

### The two NVIDIA engines disagreed, and what was done

The SASS executor runs what ptxas made; the PTX interpreter runs the PTX and is the path for a program without suitable
SASS. A kernel carrying both must give the same bits either way. Before this work it did not, on these lines of the
131,072-result classes (sm_86 build, rtx3060 profile; the same on every profile):

| Function | Results that differed | Cause |
|---|---|---|
| `rsqrt` (double) | 13 % (17,171 of 131,072), by 1 ulp | `rsqrt.approx.f64` was the correctly rounded root's reciprocal; ptxas makes it `MUFU.RSQ64H` and one correction step, which is not always the nearest double |
| `hypot` (double) | 10 % (12,131 and 14,387), by 1 or 2 ulp | `rsqrt.approx.ftz.f64` was computed from all 64 bits of the operand; it is a function of the upper 32 (`MUFU.RSQ64H`), as the PTX ISA documents, and libdevice refines that value |
| `asinh` (double) | 0.7 % (959), by 1 or 2 ulp | the same; `rcp.approx.ftz.f64` (`MUFU.RCP64H`) is the other form of the kind |
| `atan`, `cosh`, `tanh`, `erfc` (double) | 1 to 13 results each | the same |
| `exp2f` | 21 results, all subnormal, by 1 ulp | `ex2.approx.f32` without `.ftz`: ptxas squares `MUFU.EX2(x / 2)` below -126, so a subnormal is rounded twice |
| `cbrtf` | 7 results, subnormal arguments, by 1 ulp | `lg2.approx.f32` without `.ftz`: ptxas makes it `MUFU.LG2(x * 2^24) - 24` for a subnormal |
| `sinf`, `cosf`, `tanf` with `-use_fast_math` | `sinf` and `tanf` 88 to 96 % of random arguments, `cosf` 44 to 51 %, by any amount | `sin.approx` and `cos.approx` take radians, `MUFU.SIN/COS` turns, and ptxas multiplies by 1/2pi rounded toward zero first; the interpreter took the sine of the argument itself |
| `erf` (double), NaN arguments | 80 results, the NaN's payload | not found: see below |

The interpreter now forms each of these the way ptxas does (`src/exec/interpreter.cpp`, `OpMath`; `.ftz` on a double form
is kept by the parser to tell the two apart). `MUFU` itself is untouched: the SASS executor's results are bit for bit what
they were, and the interpreter's change only where they were not the SASS executor's. After it:

- 326 of 329 lines are the same from both engines, in the default build and in the `-use_fast_math` build, for every
  architecture from sm_75 to sm_120: no result that is a number differs, over 131,072 arguments a class and function.
- The three lines left are NaN payloads: `erf(double)` of a NaN (80 results: SASS returns 0x7fffffffe0000000, the canonical
  f32 NaN widened, PTX the argument quieted) and `hypot(double)` of a signaling NaN and a zero (9 pairs: SASS quiets it,
  PTX returns it as it came). They are listed in `nvidia/tests/data/numerics_engines.txt` and were not traced to their
  instruction; which one a card gives is unmeasured.
- Of the 3,600 results of the Varity programs on the t4 profile (the one profile that runs the sm_75 SASS), 5 differ
  between the engines: three are the sign of a NaN a program computes from constants (SASS `nan`, PTX `-nan`), one is
  a single-precision result one ulp apart at `-O0` (a multiply and an add ptxas fused and the interpreter's contraction
  rule, `src/ptx/contract.cpp`, did not), and one a `-use_fast_math` program whose branch on subnormal constants goes the
  other way. Not fixed.

Counts before and after: [mathfn/engines.txt](mathfn/engines.txt).

### A bug in the AMD engine, found by the `-O0` build

`varity_cases` built `-O0` for gfx1030 stopped on the rx6800 profile with a read at an address that is no allocation's.
Unoptimized code keeps its locals on the stack and reaches a slot two ways: it stores through the private segment buffer
(`buffer_store_dword v, off, s[0:3], s33`, the stack pointer scaled by the wave's size) and loads through a flat pointer made
from the same register (`src_private_base + (s33 >> 5) + offset` in a wave32 kernel). The buffer's address was unswizzled
with an index stride of 64 lanes for every kernel; for a wave32 kernel that put a slot at half the offset the pointer names.
The stride is now the wave's size (`amd/src/gcn_exec.cpp`, `scratch_buffer_access`, and the resource the kernel is handed).
A wave64 kernel is unaffected, and so is a wave32 kernel that only uses the buffer (optimized code).
`test_amd_hipcc_scratch_flat` runs a kernel that does both, built `-O0` for gfx1030.

What it changed, measured on this test set: of the 900 results of the Varity programs built `-O0` for gfx1030, 441 did
not run before (the launch stopped) and 10 came out as another value; all 900 run now. The 3,600 results of the gfx942
programs, the `-O3` results of gfx1030 and every `mathfn` line of both targets are bit for bit what they were.

### Fast math on AMD compiles some kernels to nothing

Ten of the 300 Varity programs built `-O3 -ffast-math` by hipcc have a kernel of size 0: the program divides zero by zero,
`-ffast-math` promises there is no NaN, and clang deletes the function body (no `s_endpgm`). The simulator refuses such a
launch ("code ends inside an instruction"); a card would run off the end of its code. `varity/build.sh` detects them and
leaves them out of a run (`empty-kernel.txt`), so the card script never launches one.

## What the published sources say a card gives

`compare.py --truth` gives each result's error against the true value (mpmath, 400 bits). If the simulator returned the
card's bits at a published worst case, its error there would be the published one.

### Double precision: the simulator has the published error at every worst case

All 31 double-precision inputs of CUDA 13.3.1's column (`acos` 1.53 ulp ... `tgamma` 10.1 ulp; `lgamma`'s 5.1e15 too) give,
on the simulator, an error that rounds to the published figure. The double-precision functions take only `MUFU.RCP64H` and
`MUFU.RSQ64H` from the hardware, as a starting value they refine, and no difference from a card shows at these inputs.
That is evidence from 31 inputs, not a proof for the functions: the exhaustive runs of the paper cannot be repeated for
doubles, and the card run's 131,072-result classes are the first wider check.

### Single precision: 13 of 30 agree; for 17 the published error is one or two ulp further out

For 17 functions the simulator's result at CUDA's worst case is closer to the true value than the published error allows:
the card, per the paper, returns a neighbour of the simulator's result. `compare.py` names the neighbour whose error is
the published figure:

| Function | Input | Simulator (error) | Published error (CUDA 13.3.1, T4) | The result that implies |
|---|---|---|---|---|
| `acos` | 0x1.1f9022p-1 | 0x3f7973b6 (0.665 ulp) | 1.34 ulp | 0x3f7973b4 (-2) |
| `asinh` | 0x1.fb1de4p-2 | 0x3ef431cc (0.261 ulp) | 1.74 ulp | 0x3ef431ce (+2) |
| `atan` | 0x1.037a78p+0 | 0x3f4aca14 (0.210 ulp) | 1.21 ulp | 0x3f4aca15 (+1) |
| `atanh` | 0x1.fc607ep-8 | 0x3bfe318c (1.150 ulp) | 3.16 ulp | 0x3bfe318a (-2) |
| `cbrt` | 0x1.f2c564p-7 | 0x3e7dc6a2 (0.010 ulp) | 0.991 ulp | 0x3e7dc6a3 (+1) |
| `cosh` | 0x1.b9b52cp+2 | 0x43f87c9a (1.332 ulp) | 2.34 ulp | 0x43f87c99 (-1) |
| `erf` | 0x1.02039ap+0 | 0x3f588f97 (0.011 ulp) | 1.02 ulp | 0x3f588f98 (+1) |
| `erfc` | 0x1.5bdddcp+1 | 0x38fe7377 (2.487 ulp) | 4.49 ulp | 0x38fe7375 (-2) |
| `exp` | -0x1.bcc73p+3 | 0x3576db34 (0.931 ulp) | 1.94 ulp | 0x3576db35 (+1) |
| `exp10` | 0x1.34200cp+5 | 0x7f76a1a6 (1.069 ulp) | 2.07 ulp | 0x7f76a1a7 (+1) |
| `exp2` | -0x1.f8a014p+6 | 0x0072daf5 (0.383 ulp) | 2.39 ulp | 0x0072daf3 (-2) |
| `expm1` | 0x1.634e6ep-2 | 0x3ed45ef1 (0.355 ulp) | 1.36 ulp | 0x3ed45ef2 (+1) |
| `rsqrt` | 0x1.040732p-126 | 0x5efe0269 (0.486 ulp) | 1.52 ulp | 0x5efe0267 (-2) |
| `sinh` | 0x1.6dfdap+0 | 0x3ffc05bc (0.939 ulp) | 2.94 ulp | 0x3ffc05be (+2) |
| `tan` | 0x1.f93ac6p+124 | 0xc0783a89 (2.096 ulp) | 3.1 ulp | 0xc0783a88 (+1) |
| `tanh` | 0x1.405748p-1 | 0x3f0e1874 (0.185 ulp) | 1.82 ulp | 0x3f0e1876 (+2) |
| `tgamma` | -0x1.f77d32p+0 | 0x4178865f (2.337 ulp) | 4.34 ulp | 0x4178865d (-2) |

**Outcome (measured, RTX 3060, CUDA 13.0, 2026-11-15):** the card returns the implied result for 15 of the 17. For `erf`
(0x3f588f97) and `expm1` (0x3ed45ef1) it returns what the simulator returned, not the neighbour: the paper's library is
13.3.1 and ours 13.0, the first caution below. With the card's unit modelled the simulator returns the card's result for
all 17.

The 13 that agree are `acosh`, `asin`, `atan2`, `cos`, `cospi`, `hypot`, `log`, `log10`, `log1p`, `log2`, `pow`, `sin` and
`sinpi`; `lgamma`'s 1.35e7 ulp at -0x1.3a7fcap+1 is reproduced as well. For `exp`, `exp2`, `cbrt` and `tan` the code
was traced in the kernel's SASS to `MUFU.EX2`, `MUFU.LG2` or `MUFU.RCP`, the approximate instructions the simulator computes at
host precision; that the other 13 of the 17 go through such an instruction, and that the 13 that agree do not, is the likely
explanation and was not checked function by function. The card run decides it.

Two cautions. The paper's library is CUDA 13.3.1 and ours 13.0: a function whose code changed between them would show here
as a difference that is not the hardware's (the paper's own older runs differ from its newer ones for `acos`, `asin`,
`atan`, `cbrt`, `erf` and `tgamma`). And the implied result is an inference from a three-digit figure, not a printed
result. The card run settles both.

### One printed result: `expf(-4.75f)`

Juffa's 2023 forum reply prints `expf(-4.75f)` as 0x3c0dbfd7 with CUDA 11.8 and 0x3c0dbfd6 with CUDA 9.x. The simulator
gives 0x3c0dbfd7 (nvcc 13.0), the correctly rounded value. He gives the largest error of `expf` as 2.31917 ulp in CUDA 8
(a Quadro K2200, sm_50) and 1.93099 ulp in the 2023 library, which is the paper's 1.94 ulp at -0x1.bcc73p+3; of `tanf` as
3.096 ulp in CUDA 11 (the paper: 3.10); of `erff` as 1.14 ulp in CUDA 11 (the paper's older run: 1.14). He attributes
`expf`'s and `erff`'s errors to `MUFU.EX2`.

### AMD

Against ROCm 7.2.4's column (an MI300X) the simulator's mi300x profile, running ROCm 6.4.4's library, has the published
error at 15 of 31 single-precision worst cases and 29 of 32 double-precision ones. Where it does not, the cause may be the
library version as well as the instructions the simulator computes exactly (`tgamma(-0x1.c4acaep+3)`: 43.7 ulp here, 8.31
published, which is the library, not an instruction). No AMD card has run any of this.

### The Varity paper's two cases

| | Paper: nvcc on an NVIDIA card | Paper: hipcc on an AMD card | Simulator, NVIDIA (both engines, `-O0` and `-O3`) | Simulator, AMD |
|---|---|---|---|---|
| `fmod(1.5917195493481116e+289, 1.5793E-307)` | 1.4424471839615771e-307 | 7.1923082856620736e-309 | 7.1923082856620736e-309 | 7.1923082856620736e-309 |
| Figure 4's program | 8.6551990944767196e-306 | 9.3404611450291972e-306 | 9.3404611450291972e-306 | 9.3404611450291972e-306 |
| `ceil(1.5955E-125)` | 0 | 1 | 1 | 1 |
| Figure 5's program | Inf | 1.34887e-306 | 1.34887e-306 | 1.34887e-306 |

The exact remainder is 7.1923082856620736e-309 (a subnormal), and the ceiling of a positive number is 1: the simulator
gives the correct values where the paper reports an NVIDIA card giving others. Either the simulator's double-precision
`fmod` and `ceil` differ from a card's on these inputs, which would be a bug of the first order in code that is not `MUFU`,
or the paper's nvcc (its version is not ours) compiled them differently. This is the first thing the card run answers
(`varity_cases.O0.txt`, eight lines).

**Outcome (measured):** an RTX 3060, a Tesla T4, an L4 and an A10G print the simulator's eight lines, at `-O0` and `-O3`
(nvcc 13.0.88): 7.1923082856620736e-309, 9.3404611450291972e-306, 1 and 1.34887e-306. The values the paper reports for an
NVIDIA card are not what these four cards give with this toolkit; the simulator's `fmod` and `ceil` are the cards'.

### The Varity generator run

300 programs, four builds, three input sets: 3,600 results a configuration. On NVIDIA all 3,600 are the same on every
profile and, but for the five results above, on both engines. On AMD the two architectures differ on 34 to 65 results a build, and the
ten empty kernels above are not run. The paper's differences between vendors (Table IX: 13,877 of its FP32 `-O3`
fast-math tests) are between a card and a card; simulator against card is what `card-run.sh` adds.

**Outcome (measured, RTX 3060):** of the 3,600 results of the sm_86 programs, the SASS executor gave the card's on 3,598
while `MUFU` was at host precision (one `-O0` and one `-O3` fast-math result differed) and gives the card's on all 3,600
with the card's unit.

### Whitehead and Fit-Florea

| Example | Documented | Simulator |
|---|---|---|
| (A + B) + C and A + (B + C), section 2.2 | 0x41300002, 0x41300001 | the same, NVIDIA and AMD |
| `a * a + b`, a = 0x3F800001, b = 0xBF800002 (Figure 1), fused | 2^-46 (0x28800000) | the same |
| the same as written, nvcc's default | 2^-46: the GPU fuses | 0x28800000 on both engines |
| the same with `-fmad=false` (section 4.3) | 0 | 0 |
| the same as written, hipcc `-O3` | not documented | 0 for gfx942 and gfx1030: hipcc's code does not fuse it |
| dot product, serial / FMA / parallel (Figure 5) | 0x3D653510, 0x3D653501, 0x3D653500 | 0x3D6533F0, 0x3D6533F6, 0x3D6533E0 |
| `cos(5992555.0)` | two CPU libraries' results | 0x3E9649454BADE22A (3.320904615e-07, correctly rounded); `cosf`: 0x34B24A2A |

The dot-product line is not a difference of the simulator's: the paper prints its vectors to seven digits, and those
digits give other sums (their exact dot product is 0.05595783, the paper's 0.05595875), so its three results cannot be
reproduced from the text. The simulator's three are what IEEE arithmetic gives for the printed vectors (checked in exact
rational arithmetic), and the paper's point, that the three orders give three answers, holds.

### `-use_fast_math`: the instructions themselves

With `-use_fast_math` nvcc compiles `sinf`, `cosf`, `tanhf`, `sqrtf`, `logf` and `1/x` to the `MUFU` instruction with
little or nothing around it. This is where the card and the simulator are known to differ most, from our own earlier
measurement on an A10G (rounds sp44 and sp59, which are not kept in this repository):
`MUFU.RCP` is 1 ulp off on about 13 % of mantissas, the card's `MUFU.SIN/COS` are fixed-point with about 3.6e-7 absolute
error, and `tanh` differs on 5.6 % of float patterns. `mathfn_fast` puts every function of the set on that footing in one
run.

## The card run

(Written before the runs; the runs are the next section.)

`tools/numerics/card-run.sh <out-dir>` builds everything with the machine's compiler and runs it on the card (about a
minute of GPU time; the compiles take longer). It makes no cloud call; with `VARITY=0` no network call at all. Cards:

| Card | Why |
|---|---|
| RTX 3060 (local, sm_86) | the reference card of the other probes |
| Tesla T4 (sm_75) | the paper's card for CUDA 13.3.1: its figures should be reproduced exactly there |
| A10G (sm_86), L4 (sm_89), H100 (sm_90) | whether `MUFU` differs between generations: the simulator prints the same lines for all four architectures' builds, so any line on which two cards differ is the hardware's |
| a Blackwell card, if one is had | the same |
| RX 6800, RX 6700 XT (the local AMD box) | everything AMD; the NaN rules of gfx1030 above |

The card's own architecture is the directory to compare (`sm_86/` on an RTX 3060): the card runs the SASS nvcc made, which
is what the SASS executor runs. A card later than Turing also gets the sm_75 build, whose SASS it cannot run: its driver
compiles the PTX, and those lines say whether the driver's compiler gives what nvcc's does.
`compare.py <simulator-out>/nvidia.<profile>.<sm>.sass <card-out>/sm_<sm> --truth` prints, per function and class, how
many results differ and by how many ulp, and for the worst class whether the card has the published error. What it will
decide: the 17 implied results above; the paper's `fmod` and `ceil`; whether double precision agrees beyond 31 inputs;
how much of single precision differs in ordinary use (the `rand_mid` class) and by how much; whether two cards of
different generations agree with each other.

## The card runs (2026-11-15)

`card-run.sh` ran on four cards, each with nvcc 13.0.88 programs for its own architecture and for sm_75:

| Card | Architecture | Driver | Where |
|---|---|---|---|
| GeForce RTX 3060 | sm_86 (Ampere) | 610.88 | the local card |
| Tesla T4 | sm_75 (Turing) | 595.112.07 | a rented card |
| L4 | sm_89 (Ada) | 595.112.07 | a rented card |
| A10G | sm_86 (Ampere) | 595.112.07 | a rented card |

One Blackwell card ran part of this (2026-11-17, an RTX PRO 6000 Blackwell Server Edition, sm_120: the section "Blackwell measured" at the end of this page); everything below about the other Blackwell parts (sm_100, sm_103) is *inferred* or *unmeasured*. An H100 PCIe (sm_90, a rented
card, driver 570.148.08, nvcc 12.8.93, 2026-11-15) ran the probe's hash pass (the same as the RTX 3060 on all ten operations, see
"Which devices get the model") and `mathfn` built with its own toolkit: 302 of the 329 lines are the four cards', in both builds
and in the default and the `-use_fast_math` program; the 27 others are acosh, asinh, atan2 and cbrt in single precision and asin,
asinh, atan2 and atanh in double. With the instructions equal, those 27 are the library code of CUDA 12.8 against CUDA 13.0's, not
the card (the pinned lines are CUDA 13.0's for that reason); no card ran both toolkits to show it directly.

**The four cards print the same lines.** All 329 lines of `mathfn` and all 329 of `mathfn_fast` (131,072 results a random
class) are identical on the four cards, and on each card the lines of its own architecture's build are the lines of the
sm_75 build. `fp_examples`, `fp_examples_nofma` and `varity_cases` (`-O0` and `-O3`) are identical too, and are the
simulator's lines. So, as far as these programs see it, Turing, Ampere and Ada have one math unit: its behaviour belongs
in the executor, not in a device profile.

**The simulator against the card** (the sm_86 build on the rtx3060 profile, SASS; the same counts for the sm_75 build on the
t4 profile):

| | Lines that are the card's, of 329 | The lines that are not |
|---|---|---|
| `mathfn`, before (MUFU at host precision; `VGPU_MUFU_EXACT=0` today) | 263 | 66: f32 `acos asin asinh atan atanh cbrt cosh erf erfc exp exp10 exp2 lgamma rsqrt sinh tan tanh tgamma`; f64 `cbrt` (worst), `hypot` and `rsqrt` (rand_mid) |
| `mathfn_fast`, before | 225 | 104: those, and f32 `sin cos log log2 log10 pow hypot sqrt rcp div rsqrt_div` |
| `mathfn`, with the card's unit, SASS executor | **329** | none |
| `mathfn_fast`, with the card's unit, SASS executor | **329** | none |
| `mathfn` and `mathfn_fast`, with the card's unit, PTX interpreter | 326 each | `erf f64 edges`, `erf f64 rand_bits`, `hypot f64 edges`: the NaN payloads of `nvidia/tests/data/numerics_engines.txt`, which are not `MUFU`'s (the SASS executor's NaN is the card's, the interpreter's is not) |

The counts are the same at 4,096 results a class (the size the test pins) and at 131,072 (the size the four cards ran); at
131,072 the simulator's two files, the first results of every line included, are byte for byte the RTX 3060's. Of the 3,600
Varity results 3,598 were the card's before and all 3,600 are now. With `VGPU_MUFU_EXACT=0` the simulator prints, line for
line, what it printed before (the pinned files as they were: 329 of 329 and 329 of 329).

The double-precision lines that differed (`cbrt`, `hypot`, `rsqrt`) go through `MUFU.RCP64H` and `MUFU.RSQ64H`, which the
card does not compute as the upper word of the correctly rounded value either (below).

## The probe: one instruction, every input

`tools/numerics/mufu_probe.cu` runs one `MUFU` sub-operation over all 2^32 operands on a card.

- **The instruction is alone.** The kernels (`mufu_kernel.cu`) hold the PTX form `rcp.approx.ftz.f32`, which ptxas turns
  into a lone `MUFU.RCP Rd, Rs`. The probe loads the compiled cubin, finds every `MUFU` in its code (128-bit instructions
  whose opcode field, bits 0-8, is 0x108) and rewrites the sub-operation field (bits 74-77) to the one asked for: `COS`,
  `SIN`, `EX2`, `LG2`, `RCP`, `RSQ`, `RCP64H`, `RSQ64H`, `SQRT`, `TANH`. PTX cannot produce most of them alone
  (`sin.approx` is a multiply by 1/2pi and `MUFU.SIN`; the 64H forms are half of a double). `--cubin-out` writes the
  rewritten code, and `nvdisasm` of it shows `MUFU.<OP> Rd, Rs` and no other `MUFU` (`mufu-card-run.sh` checks that on
  every run and keeps the listing in `sass.txt`). `--abs` and `--neg` set the operand modifiers (`MUFU.RCP Rd, -|Rs|`).
- **The operand** is the 32-bit pattern itself: a float's bits, or for the 64H forms the upper word of a double.
- **`--hash`**: the results in blocks of 4,096 consecutive operands; a block's hash is FNV-1a 64 over its results (one
  32-bit result a step), computed on the card; a chunk of 2^26 operands is hashed over its blocks' hashes. 64 chunks and
  the whole, an operation: `tools/numerics/mufu/rtx3060.hashes.txt` (650 lines, 20 KB) is the RTX 3060's. About a second of
  GPU time an operation.
- **`--dump`**: the raw results of a range, for fitting.
- **`--check`**: every result against the simulator's model, `include/vgpu/exec/mufu_model.hpp`, which the probe includes
  as it is (the header is plain C++17 for that reason). It prints how many of the range differ and the first of them.
  About a minute an operation on one host core.

`tools/numerics/mufu-card-run.sh <out-dir>` builds both for the card in the machine, runs the hash pass of all ten
operations, compares with the RTX 3060's file and says, per operation, "the same as the RTX 3060 on all 2^32 inputs" or
which chunks differ; `MUFU_CHECK=1` adds the check against the model and `MUFU_DUMP="RCP:0x3f800000:0x800000"` a dump of a
range of a card that differs. This is the run to make on an H100 and on a Blackwell card.

## The card's unit: `MUFU` as an RTX 3060 computes it

**Result.** For each of the ten sub-operations the model gives the RTX 3060's result on 4,294,967,296 of 4,294,967,296
operands (`mufu_probe --check`, the final header, 2026-11-15):

| Operation | Operands on which the model is the card | How the card differs from the function rounded to nearest (measured over the binades [0.5, 4) of the argument unless said) |
|---|---|---|
| `MUFU.RCP` | 4,294,967,296 of 4,294,967,296 | 13.2 % of mantissas by one ulp (4.5 % below, 8.7 % above); error within 0.975 ulp |
| `MUFU.RSQ` | all | 21 to 22 % by one ulp, 8 mantissas of an even binade by two; error within 1.52 ulp |
| `MUFU.SQRT` | all | 17 % by one ulp; error within 0.95 ulp |
| `MUFU.EX2` | all | 31 to 39 % by one ulp; error -0.75 to +1.40 ulp (arguments in [1, 2)) |
| `MUFU.LG2` | all | absolute error within 2.2e-7; near 1 that is far more than an ulp of the result (log2(1 + 2^-23) is 2.79e-7 on the card, 1.72e-7 in truth) |
| `MUFU.SIN`, `MUFU.COS` | all, each | absolute error within 3.4e-7 (a quarter turn of arguments at full resolution); the argument (in turns) is cut to 25 fraction bits |
| `MUFU.TANH` | all | absolute error within 8e-6 (about 130 ulp near 1); the result is the argument below 2^-8 |
| `MUFU.RCP64H`, `MUFU.RSQ64H` | all, each | the single-precision unit on the upper 20 mantissa bits, cut to 20 bits: the result is not the upper word of the correctly rounded value on 6.9 % of upper words (`RCP64H`, one above) and 5.7 % (`RSQ64H`, one above, rarely one below) |

With each operand modifier (`|Rs|`, `-Rs`, `-|Rs|`) the model was checked on 16,777,216 positive and 16,777,216 negative
operands (from 0x3f000000 and from 0xbf000000) for `RCP`, `SIN`, `RCP64H`, `RSQ64H` and `TANH` and differs on none: the modifier is applied to the operand's sign bit before the unit,
for the 64H forms too (the simulator did not apply it to those before).

**Structure.** Every operation is one datapath: the leading bits of the argument select an interval, and the result is

    Y = c0 + c1 * x + c2 * sq19(x)

where `x` is the 17 argument bits below the interval index, `c0`, `c1`, `c2` are the interval's table row, and `Y` is a
fixed-point number with 40 fraction bits that is *cut* (never rounded) to the result's format. `sq19(x)` is the square of
`x` as a carry-save array forms it when its low columns are left out: with the partial products of the folded array (bit
`i` alone at weight 2^(2i); bits `i < j` together at weight 2^(i+j+1)), the sum of those of weight 2^19 or more, shifted
right by 19. It is not `x * x >> 19`: the products below 2^19 would carry into the kept columns and do not.

The structure, a table of second-degree polynomials indexed by the leading bits of the argument with a truncated squarer
and a fused accumulation, is the one published by Piñeiro, Oberman, Muller and Bruguera, "High-Speed Function
Approximation Using a Minimax Quadratic Interpolator", IEEE Transactions on Computers 54(3), 2005. The paper was recalled
as the hypothesis for the fit and was not re-read for this work; no coefficient, constant, bit width or line of code is
taken from it or from any other source. Every number in the model (the interval boundaries, the 17 bits, the column 19,
the 40 fraction bits, each table row, each constant, each special case) was determined from the RTX 3060's results as
described next.

**How it was fitted** (reproducible from dumps of a few binades an operation; the scripts are not in the tree):

1. *Intervals.* The error against the true function, averaged over 1,024 consecutive mantissas, jumps where the table
   row changes: at multiples of 2^16 mantissa units for `RCP` (128 intervals of the mantissa), of 2^17 for `RSQ`, `SQRT`,
   `EX2` and `LG2` (64), with a second half of the table for the odd exponents of `RSQ` and `SQRT`.
2. *The squarer.* In one interval, "is there a real (c0, c1, c2) with result = floor(c0 + c1 x + c2 T(x)) for every x" is a
   linear program. It is infeasible for `T(x) = x^2` (by 0.09 ulp), for the square of `x` without its low 2 to 10 bits,
   and for the folded array cut at every column but one: at column 19 (of a 17-bit `x`) it is feasible, in every interval
   of every operation.
3. *The row.* With that `T`, least squares gives c1 and c2 to within a fraction of a unit; they are integers at 2^-40, and
   for the pair of integers nearest the estimate the set of c0 that reproduce *every* result of the interval is an
   interval of integers, computed exactly (the largest lower bound and the smallest upper bound over the 2^17 results). It
   is 2 to 128 units wide and never empty.
4. *The constant.* The sets of c0 of all intervals of an operation share their low bits: c0 is a table entry with 26
   significant bits plus one constant an operation, the same for every interval (`RCP` 0x1f9c, `RSQ` 0x7ffe, `SQRT`
   0x7fff, `EX2` 0x5f88 in units of 2^-17 of the last place; `LG2` 0xd14 in units of 2^-40). Where several constants
   reproduce everything (four for `RCP`, `EX2` and `LG2`, two for `RSQ`) the smallest is used. `TANH`'s c0 are multiples
   of 2^-24 with no constant; `SIN`'s rows are multiples of 2^-37 throughout.
5. *Everything else* (exponent handling, flushing, the special operands, the conversion of a fixed-point result to a float)
   was read off `--check` runs over all 2^32 operands: each rule below was added because a run printed the operands on which
   the model without it was not the card.

**The operations.** `m` is the 23-bit mantissa, `e` the exponent. A subnormal operand is read as a zero of its sign and no
result is subnormal (except in `TANH`); every NaN result is 0x7fffffff.

| Operation | Index and x | Result | Rules found by the full runs |
|---|---|---|---|
| `RCP` | `m[22:16]` (128 rows); x = `m[15:0]` doubled | Y >> 17 is the 24-bit significand of 2/m; exponent -e - 1 | below the normal range: zero; a zero operand: infinity |
| `RSQ` | the exponent's parity, then `m[22:17]` (2 x 64 rows); x = `m[16:0]` | significand of 2/sqrt(m) or 2/sqrt(2m) | 1/sqrt of an even power of two is exact (the polynomial alone gives one ulp less); a negative operand: NaN; -0: -infinity |
| `SQRT` | as `RSQ` | significand of sqrt(m) or sqrt(2m) | a negative operand: NaN; -0: -0 |
| `EX2` | the argument's magnitude as a fixed-point number with 23 fraction bits (lower bits dropped): fraction bits [22:17] (64 rows); x = fraction bits [16:0] | significand of 2^fraction; the integer part is the exponent | a negative argument is complemented, not negated: fraction 2^23 - 1 - f and integer -i - 1, unless f is zero (then -i); 128 and above: infinity; below the normal range: zero |
| `LG2` | `m[22:17]` (64 rows); x = `m[16:0]` | e * 2^36 + (Y >> 4), a fixed-point number with 36 fraction bits, cut to 24 significant bits | a negative value is negated as a one's complement (one unit of 2^-36 short); log2 of exactly 1 is 0 (the polynomial gives 1.07e-7); zero: -infinity; negative: NaN |
| `SIN`, `COS` | the argument's magnitude in turns, cut to 25 fraction bits; `COS` adds a quarter turn; the top two bits are the quadrant; the other 23 bits `z`, complemented (`2^23 - 1 - z`) in the odd quadrants: `z[22:17]` (64 rows), x = `z[16:0]` | Y (a multiple of 2^-37) cut to 24 significant bits; the sign from the quadrant and, for `SIN`, the argument | 2^24 turns and above: phase 0; sin(-0) is -0 and the sine of a half turn is -0; infinity: NaN |
| `TANH` | by exponent, for 2^-8 <= abs(x) < 8: 1, 1, 1, 2, 2, 4, 4, 4, 8, 16, 8 rows a binade (51 rows); x = the next 17 mantissa bits; the mantissa bits below are not read | Y cut to 24 significant bits | below 2^-8 the result is the argument itself (a subnormal too); 8 and above: 1 |
| `RCP64H`, `RSQ64H` | the upper word's 20 mantissa bits as the upper 20 of `m` | the single-precision significand cut to 20 bits, with the double's exponent | the same special operands as `RCP` and `RSQ`; the lower word of the double is not read |

The tables are `include/vgpu/exec/mufu_tables.inc`: 627 rows (128 `RCP`, 128 `RSQ`, 128 `SQRT`, 64 `EX2`, 64 `LG2`, 64
`SIN`/`COS`, 51 `TANH`), 22 KB of text, generated from the fits. The model is `include/vgpu/exec/mufu_model.hpp`, about 200
lines with no dependency.

**What a machine without a card holds it to.** `tests/unit/test_mufu_model.cpp` computes the probe's hashes from the model
and compares them with the card's file: one chunk of 2^26 operands an operation in every run, all 64 chunks of all ten
operations, which is every operand there is, with `VGPU_EXHAUSTIVE=1`. It also holds the model to the card's results on
310 listed operands an operation (`nvidia/tests/data/mufu/rtx3060.vectors.txt`).

## The AMD instructions on RDNA2 cards: measured, not modelled (2026-11-16)

The same probe as for `MUFU`, for the AMD transcendental instructions: `tools/numerics/amd_trans_probe.cpp` (plain hipcc kernels,
the instruction forced with inline asm: `v_rcp_f32`, `v_rsq_f32`, `v_sqrt_f32`, `v_exp_f32`, `v_log_f32`, `v_sin_f32`,
`v_cos_f32`) over all 2^32 inputs, hashed in the `mufu_probe` format (4096-result blocks, FNV-1a 64, chunks of 2^26).
`tools/numerics/amd-trans-card-run.sh` runs it on another card and compares with `tools/numerics/amd/gfx1030.hashes.txt`.
Cards: an RX 6800 (gfx1030) and an RX 6700 XT (gfx1031), ROCm 7.2.0 hipcc, one device named at a time.

**Measured: the two cards agree.** All seven operations have identical hashes on the RX 6800 and the RX 6700 XT, over all
2^32 inputs, in both kernel modes. The kernel's f32 denormal mode changes one result: `v_sin_f32` of a denormal (kept
denormal: the sine of a turn fraction, `0x00000001` gives `0x00000006`; flushed: zero). The other six are the same in both
modes: they flush a denormal operand and a denormal result whatever the mode says. 81 selected operands with the results are
in `tools/numerics/amd/gfx1030.vectors.txt`.

**Measured: none is the correctly rounded result.** Against the function evaluated in double precision, over every
input with an ordinary result (default mode; `sin` and `cos` over |x| <= 256 turns, which the ISA guide gives as their range):

| Operation | results not the nearest float | largest error (ulp of the true value) | note |
|---|---|---|---|
| `v_rcp_f32` | 10.7 % (452,384,016 of 4,227,173,206) | 0.88 | never 1 ulp or more |
| `v_rsq_f32` | 11.0 % | 0.86 | |
| `v_sqrt_f32` | 15.4 % | 0.92 | (a denormal operand gives zero) |
| `v_exp_f32` | 1.4 % | 0.78 | |
| `v_log_f32` | 49.3 % | 1.00 | |
| `v_sin_f32`, `v_cos_f32` | 95 % and 10 % | the error is absolute near the zeros (relative error up to 9.8 ulp); not characterized further | |

The documented bound (1 ulp, RDNA2 ISA guide) is not exceeded by the five that have a relative bound. The simulator computes
them at the host's precision, rounded once (`1.0f / x`, `std::sqrt`, `std::exp2`, `std::log2`; sine and cosine through
a double), so it differs from the card on those percentages of inputs, and also on operands that are special on the card:
a denormal operand of `v_rcp_f32`, `v_rsq_f32`, `v_sqrt_f32`, `v_log_f32` is a zero on the card (`v_rcp`: infinity; `v_log`:
-infinity) and a finite number on the host; a denormal result (`v_rcp_f32(2^127)`, `v_exp_f32(-127)`) is zero on the card;
the NaN of a negative operand to `v_rsq_f32`/`v_sqrt_f32`/`v_log_f32` is `0xffc00000` on the card (read from the
vectors file; the host's `-nan` is the same bits on x86).

**Run on 2026-11-23 (issues #721, #722, `test_amd_gcn_cross_arch`):** the simulator's gfx1030 differed from the card on 82 of the
7 x 77 selected operands: every denormal operand and result of the first five, and the cosine (and the sine of an operand that
is not a whole number of turns) of the largest floats. Now the interpreter and the tier share `amd/src/transcendental.hpp`:
the five flush a denormal operand and a denormal result to a zero of its sign (measured on gfx1030 and gfx1031, **inferred** for
every other target: LLVM's AMDGPU lowering scales a denormal operand of f32 sqrt, rsq, log and exp on every target because the
instruction does not take one, and the machine-readable ISAs of CDNA3, RDNA2, RDNA3, RDNA4 and CDNA5 describe all seven in the same
words); the sine and cosine count the angle in turns and keep its fraction for any size of argument, on every target (before,
only gfx942, gfx950 and gfx1250 did: gfx90a, gfx1100 and gfx1201 took `sin(2 pi x)` in a double, and gfx1030 had the whole-turn sine
only), with the card's exact zeros. The card's v_sin_f32 and v_cos_f32 of the multiples of 1/64 turn from -4 to 4 and of the
floats next to the multiples from 1/64 to 1 (`tools/numerics/amd_trig_probe.cpp`, `tools/numerics/amd/gfx1030.trig64ths.txt`, the RX 6800
on server2, ROCm 7.2.0): the sine of a multiple of 1/2 and the cosine of an odd multiple of 1/4 are +0, where 2 pi in a double
leaves 1.2e-16 or 6.1e-17; everywhere else the card is within 4e-7 of the double's value on these 642 operands. What is still the host's:
the low bits of the ordinary results of `v_exp_f32`, `v_log_f32`, `v_sin_f32`, `v_cos_f32`, on every target, and of all seven on the targets
but gfx1030 and gfx1031 (issues #729, #741).

**Run on 2026-11-25 (issues #729, #741, `test_amd_rdna2_trans`): `v_rcp_f32`, `v_rsq_f32` and `v_sqrt_f32` are the RX 6800's, bit for bit
over all 2^32 operands, on gfx1030 and gfx1031 (measured).** `amd/src/rdna2_trans.cpp`; the other targets keep the host's result (no card).
What the earlier attempts could not reach (a closed form) was not needed: the model is a table in another form, small enough to carry.
- The result's mantissa depends on the operand's mantissa only. **Measured**, on dumps of seven binades each of `rcp`, `rsq` and `sqrt`:
  the 23 result mantissa bits are the same function of the 23 operand bits in every binade (for `rsq` and `sqrt` one function for an even
  exponent, one for an odd), the exponent is shifted as the function's exponent is, and a negative operand of `rcp` is the positive one's
  result with the sign flipped (all 2^23 operands of one binade). So five dumps of 2^23 results (`amd_trans_probe <op> dump`: `rcp`
  [1, 2), `rsq` [1, 2) and [2, 4), `sqrt` [1, 2) and [2, 4)) are the card's whole behaviour on normal operands.
- Each is a least-squares cubic per 2^18 mantissas (32 segments; the squarer's operand with its low bit cleared for `rcp` and `rsq`, two bits for
  `sqrt`, found by trying 0 to 8 bits), evaluated in 128-bit integers, plus a list of the mantissas where the card is one unit in the last place
  away from the cubic's floor (65,945 of the 2^23 for `rcp`, 0.79 %; 66 to 74 thousand for each of the others; the direction is fixed by the
  fraction of the cubic: up from one half). Nothing is fitted to hide a larger error: the cubic's own misses are 0.8 % at every segment width from
  2^18 down to 2^10, the noise of the card's truncated terms (about 1/64 ulp), which is why no cubic, quadratic or table row is exact and why the earlier
  attempts stopped at 99.2 %. The exceptions are the card's integer truncations written down, not a model of them.
- `tools/numerics/amd_trans_table_fit.py` writes `amd/src/rdna2_trans_data.inc` (700 KB, the gaps of the lists as base64) from the five dumps;
  the same arithmetic is `rdna2_trans.cpp`. `test_amd_rdna2_trans` runs all 2^32 operands of each function through it and compares the 64 chunk
  hashes of each with `tools/numerics/amd/gfx1030.hashes.txt` (the card's, written 2026-11-16, before the model existed): all 192 chunks equal.
  Special operands (zero, denormal, infinity, NaN, a negative operand of `rsq` and `sqrt`) are the rule above and are in the same hashes.
- Not the card's but the same on both cards: an RX 6700 XT's hashes are the RX 6800's on all seven functions (2026-11-16), so gfx1031 gets
  the same table; it was not dumped separately. gfx1032 and the other RDNA2 parts the loader maps to gfx1030 are **inferred**.
- Not done, and why: `v_exp_f32`, `v_log_f32`, `v_sin_f32`, `v_cos_f32`. The same method needs the card's behaviour as a function of one
  mantissa, and these are not: `v_exp_f32` of x uses a fraction of x finer than 2^-24 (adjacent operands of [0.25, 0.5) give different results
  in 22 percent of pairs, even and odd alike), so its table is a function of the operand's binade too (about 130 of them), and `v_log_f32` and the
  trigonometric forms add the exponent or the turn count to a fixed-point result. A table per binade is some two megabytes of exception lists for `exp`
  alone; none was built.

**Measured: the double forms are about 24 bits.** (Still not modelled. 2026-11-25: the instruction's result depends on the top 30 or so bits of the operand's mantissa, not 23
(over 2^32 adjacent operands the result changes 434 times), the results of the operands that are floats are not the single-precision instruction's
(`v_rcp_f32` is the double result rounded to nearest on 88 % of the 2^23 mantissas of [1, 2)), the result is a multiple of 2^-35 and its error is
at most 0.47 of a single-precision ulp, and it is not a cubic of 2^18 mantissas (the cubic's residual is 0.3 ulp with a parity term of 31 units of 2^-35): a different pipeline, with
2^30 inputs a binade, so the exception-list method above would need about 8 million entries a binade. Its special operands are the host's rule: the card's
results on 327 selected operands and 1,024 random ones are `tools/numerics/amd/gfx1030.vectors64.txt` and `gfx1030.sample64.txt`, tested by
`test_amd_gcn_cross_arch` with the size of the difference pinned (2^-22).) `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64` on 2^27 pseudo-random operands (and
operands near 1): the result has 22 to 26 significant bits (the low 29 to 32 bits of the significand are zero; `v_sqrt_f64(1.0)`
is 1 + 2^-24), the largest distance from the correctly rounded double is 2^28 ulp (2.6e8 for `rcp`), and the results are
never the correct double except near powers of two. The compiler refines them with fused multiply-adds. Both cards agree
on every sampled operand. Not characterized exhaustively (2^64 operands).

**Not done: an exact model (round 8, 2026-11-16, second attempt; still no model, no switch).** The AMD instructions are
unchanged. What the second attempt measured (RX 6800, dumps of the binade [1, 2) of each operation, 2^23 results, fits by
`tools/numerics/amd_trans_fit.py`; the segment tests are on 2^18 consecutive mantissas):

- **Measured: a cubic per segment of 2^18 mantissas (32 a binade) is the form.** The result as an integer in its own ulp
  is the floor of a real cubic of the segment offset to within a slack of -0.015 to -0.03 ulp (about one to two
  1/64-ulp steps), for `rcp`, `rsq`, `sqrt` and `exp`; a quadratic of the same segment misses by 0.13 to 10 ulp. Rows of
  2^16 to 2^11 were tested by the first attempt (and a quadratic per row of 2^10 fits exactly, which is a table of 8,192 rows
  a binade, not a datapath).
- **Measured: the squarer operand has its low bit cleared for `rcp` and `rsq`.** A cubic with `fq*fq`, `fq = f & ~1`, in
  place of `f*f` takes the slack of `rcp` segment 3 from -0.0572 to -0.0165 ulp and `rsq` segments 3/17/25 from
  -0.030/-0.023/-0.019 to -0.016/-0.015/-0.015; clearing 2 or more bits is worse. The violations of the plain cubic are by
  input parity (even inputs one side, odd inputs the other). `sqrt` is flat across 0 to 2 bits (-0.033 to -0.020) and `exp`
  prefers no truncation (-0.015). This is the same shape the public AMD emulator describes for RDNA3 and RDNA4 (a cubic over
  64 segments of 18-bit offsets, "the squared input has its low bit cleared"; source: the `rocjitsu` pull request #12510
  in ROCm/rocm-systems, read for the shape only, no coefficient taken; **documented by a third party**, not measured on RDNA2).
- **Measured: the linear term is an integer multiple with a 14-bit fraction.** With the linear term as `rne(B * f / 2^14)`
  (B an integer) and the other terms free reals, the best slack has a sharp peak at one B per segment (`rcp` segment 3:
  B = -1,753,039, slack -1.11 steps of 1/64 ulp, one step either side -1.99/-2.1, falling 1.3 steps per unit of B;
  segment 17: B = -894,410). That fixes six guard bits (final result = sum >> 6) and a 14-bit linear fraction.
- **Not explained: about one more 1/64-ulp step of noise.** At the best integer linear term the slack is still -1.1 steps
  (-0.96 on segment 17), so some further truncation (the quadratic/cubic terms in coarser units, or two separately floored
  products) is not reproduced. The best model reached (real-valued cubic coefficients from the linear program, floored)
  reproduces **99.2 % of the `rcp` inputs of segments 3 and 17** (0.82 % and 0.79 % of inputs differ by one ulp), against
  100 % required. An outside project that reproduces the RX 6700 XT with an RDNA3-shaped model needs a correction list of
  2,761 inputs a binade for `rcp` (a third-party claim from its pull request, `boykopovar/AnyPS5` #9305; not verified here), which
  is the same conclusion: a cubic plus a sparse residual, not a closed structure.
- **Not done:** `log` (the result's ulp changes inside the binade, no fixed-point form was set up), `sin`, `cos`, the other
  segments and binades, the integer widths of the quadratic and cubic coefficients, and every test of a candidate against the
  hashes. A residual list large enough to be exact (about 66,000 inputs of the 2^23 of a binade for the best model here) is the card's
  result table in another form and was not shipped; **no inexact model is on by default and there is no opt-out because
  there is nothing to opt out of.**

What the next attempt needs: the integer pipeline of the quadratic and cubic terms (their units and truncation), found
per segment by scanning the cubic coefficients as integers in the way the linear term was scanned (the peak in B shows
the method works); the dumps are 32 MB a binade (`amd_trans_probe <OP> dump <start> <count>`), the hashes above are the
test.

## The PTX forms

The PTX interpreter computes each approximate form as ptxas (CUDA 13.0; the same code for sm_75, sm_86 and sm_90) expands
it around the one instruction, read off the SASS of `tools/numerics/mufu_forms.cu`:

| PTX form | What ptxas makes |
|---|---|
| `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `sqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`, `lg2.approx.ftz.f32`, `tanh.approx.f32` | the instruction alone |
| `sin.approx.f32`, `cos.approx.f32` (with or without `.ftz`) | `FMUL.RZ x, 0.15915494`, then `MUFU.SIN` / `MUFU.COS` |
| `rcp.approx.f32` | s = 2^24 where abs(x) is below the smallest normal number, 0.25 where it is above 2^126, else 1: `s * MUFU.RCP(x * s)` |
| `rsqrt.approx.f32` | abs(x) below the smallest normal number: `MUFU.RSQ(x * 2^24) * 4096` |
| `sqrt.approx.f32` | the same range: `MUFU.SQRT(x * 2^24) * 2^-12` |
| `ex2.approx.f32` | x below -126: `MUFU.EX2(x * 0.5)` squared |
| `lg2.approx.f32` | abs(x) below the smallest normal number: `MUFU.LG2(x * 2^24) + -24` |
| `rcp.approx.ftz.f64`, `rsqrt.approx.ftz.f64` | `MUFU.RCP64H` / `MUFU.RSQ64H`: upper word in, upper word out |
| `rsqrt.approx.f64` | `MUFU.RSQ64H` and one correction step; a subnormal scaled by 2^54 |
| `div.approx.ftz.f32` | `a * MUFU.RCP(b)`, a `.ftz` multiply |
| `div.approx.f32` | abs(b) below the smallest normal number: both operands times 2^24; then `a * MUFU.RCP(b)` |
| `div.full.ftz.f32` | abs(b) above 2^126: both operands times 0.25 (`.ftz`); then `a * MUFU.RCP(b)` |
| `div.full.f32` | both scalings, then `a * MUFU.RCP(b)` |

`mufu_forms` ran on the RTX 3060 over 52 operands a single-precision form (subnormals, the smallest normal numbers, the
values around 2^126 and -126, zeros, infinities, NaNs, a few ordinary values), 32 a double form and 144 operand pairs a
division: 1,452 results in `nvidia/tests/data/mufu/rtx3060.forms.txt`. The interpreter gives the card's result on every one
of them (`test_mufu_model`). `div.full.f32` used to be the exact quotient in the interpreter and `a * rcp(b)` in the SASS
executor; both are now what the card computes. With `VGPU_MUFU_EXACT=0` it is the exact quotient again.

The half-precision forms are the next section.

## The half-precision forms (2026-11-16)

Until now these were the function at host precision (the PTX interpreter) or, in the SASS executor, not a half-precision form at
all: `MUFU.<op>.F16` read its source register as a float. Both are the card's now, on the RTX 3060.

**Which forms exist** (measured: nvcc 13.0 and ptxas 13.0, sm_75 to sm_120, compile only except sm_86). `ex2.approx.f16`,
`ex2.approx.f16x2`, `tanh.approx.f16` and `tanh.approx.f16x2` compile from sm_75 on to `MUFU.EX2.F16` / `MUFU.TANH.F16` (the
packed forms: two of them, the second reading `R.H1`, and a `PRMT` joining the halves; nothing else around the instruction).
`ex2.approx.ftz.bf16`, `.bf16x2`, `tanh.approx.bf16` and `.bf16x2` need sm_90 or later (ptxas: "requires .target sm_90 or higher")
and compile to `MUFU.EX2.BF16` / `MUFU.TANH.BF16`. `rcp`, `rsqrt`, `sqrt`, `sin`, `cos` and `lg2` have no half-precision
`.approx` form in PTX (ptxas rejects `rcp.approx.f16` and the others: "Unexpected instruction types"), and `ex2.approx.ftz.f16`
is illegal. The library's half functions (`hexp2`, `hrcp`, `hsqrt`, `h2rcp`, ...) do not use the F16 instruction: nvcc widens the
half with `HADD2.F32`, runs the single-precision `MUFU` and packs with `F2FP.PACK_AB`, which the simulator already did right.
The F16 instruction does exist for every one of the eight sub-operations (the probe rewrites the `MUFU` of its kernel as it does
for the 32-bit forms: `--f16`, and `--h1` for the register's high half), so the SASS executor has all eight.

**What the card does** (measured, RTX 3060, sm_86): all 65,536 half patterns of each of `COS`, `SIN`, `EX2`, `LG2`, `RCP`, `RSQ`,
`SQRT` and `TANH` with the F16 bit, and (`mufu_probe --f16 --check`, over the 2^32 registers, so every value of the unused
half too) the high-half select, `|x|` and `-x`:

> `MUFU.<op>.F16` widens the half exactly to a float, runs the single-precision unit ([above](#the-cards-unit-mufu-as-an-rtx-3060-computes-it))
> and cuts the float to a half **toward zero**. The result is in the low 16 bits, the upper 16 are zero. A float of 2^16 or
> more is an infinity (so 65504 is the largest finite result, and the floats from 65504 up to 2^16 cut to it); the unit's NaN
> is 0x7fff; a result below 2^-24 is a zero of its sign; half-precision subnormal results are kept (cut, not flushed). The
> operand's half is bits 0-15, or bits 16-31 with `.H1` (instruction bits 60-61); `|x|` and `-x` act on that half's sign bit.

That is `model::eval_f16` (`include/vgpu/exec/mufu_model.hpp`), a function of the 32-bit model and no table. No fit was needed:
the 8 x 65,536 results of the card equal it on every pattern. Nothing is stored beyond twelve hashes
(`tools/numerics/mufu/rtx3060.f16.hashes.txt`). The PTX forms were also run on the card as PTX (`tools/numerics/mufu_f16_forms.cu`):
`ex2.approx.f16` and `tanh.approx.f16` give the hashes of `MUFU.EX2.F16` and `MUFU.TANH.F16`; the `.f16x2` forms, on the
65,536 registers (x, (x * 40503 + 12345) mod 65536), equal two independent halves.

**Before and after, of the 65,536 patterns of each form, against the card** (the "before" is `VGPU_MUFU_EXACT=0`, the earlier
code verbatim, run by `test_mufu_model`):

| Form | Engine | Differed from the card before | After |
|---|---|---|---|
| `ex2.approx.f16` | PTX interpreter (host precision, rounded to nearest) | 19,417 | 0 |
| `tanh.approx.f16` | PTX interpreter | 14,736 | 0 |
| `ex2.approx.f16x2`, `tanh.approx.f16x2` | PTX interpreter | the same, a half at a time (the lanes are independent) | 0 |
| `MUFU.EX2.F16`, `.COS.F16`, `.LG2.F16`, `.RCP.F16`, `.RSQ.F16` | SASS executor (the register read as a float) | 65,536 each | 0 |
| `MUFU.SQRT.F16` | SASS executor | 65,535 | 0 |
| `MUFU.SIN.F16` | SASS executor | 57,344 | 0 |
| `MUFU.TANH.F16` | SASS executor | 51,183 | 0 |

The PTX interpreter's errors were the rounding: it rounded the exact result to nearest, the card cuts a float of the unit's
result toward zero (the one-ulp-low half results). Its NaN and infinity cases already agreed. The SASS executor's were a bug:
the F16 bit was ignored, and the register of a half pattern was read as a float (a tiny subnormal) and the function of that
written as a float.

**The encoding changes from sm_90** (nvcc 13.0 output, `cuobjdump -sass`, the high instruction word of `MUFU`, opcode 0x108; the
same kernels on every architecture). sm_75 to sm_89: `.F16` is instruction bit 73 (`EX2.F16` 0x...280000000a00). sm_90, sm_100 and
sm_120: `.F16` moved to bit 72 (`EX2.F16` 0x...0900) and bit 73 is `.BF16` (`EX2.BF16` 0x...0a00, `TANH.BF16` 0x...2600); the op field
(bits 74-77) and the `.H1` part (bits 60-61) are unchanged. The decoder reads the bits by the architecture of the cubin. Before
2026-11-16 it read bit 73 as `.F16` everywhere, so on sm_90 and later a real `.F16` (bit 72) ran as a plain 32-bit `MUFU` on a half
register and a real `.BF16` (bit 73) ran as `.F16`, both wrong. Evidence: the encoding is nvcc output (sm_90, sm_100, sm_120 all
print the same words); only the H100 ran.

**The BF16 forms** (measured on an H100 PCIe, 2026-11-16, driver 570.148.08; `mufu_probe --bf16`, which sets the bit-73 form on the
sm_90 cubin, over all 2^32 register values of each of the eight sub-operations, low half and `.H1`, 0 differences):

> `MUFU.<op>.BF16` is the single-precision unit run on the bfloat16 widened by a shift (`bits << 16`), the upper 16 bits of the
> float result (so the float cut toward zero to a bfloat16). No special case: NaN, infinity and subnormal inputs agree with it
> too. The result is in the low 16 bits, the upper 16 are zero.

That is `mufu::eval_bf16` (`include/vgpu/exec/mufu_model.hpp`). The PTX forms `ex2.approx.ftz.bf16`, `.bf16x2`, `tanh.approx.bf16`
and `.bf16x2` compile on sm_90 to `MUFU.EX2.BF16` / `MUFU.TANH.BF16` and give the SASS hashes (`tools/numerics/mufu_bf16_forms.cu`).
The F16 forms ptxas emits for `ex2.approx.f16`, `.f16x2`, `tanh.approx.f16` and `.f16x2` on sm_90 (bit 72) give the same four hashes
as the RTX 3060. The 12 hashes are `tools/numerics/mufu/h100.bf16.hashes.txt` (FNV-1a 64 over the 65,536 results of a step, the
probe's `--hash16`) and `h100.f16-ptx.hashes.txt`; the SASS executor and the PTX interpreter now give them, and `test_mufu_model`
checks it. **Blackwell: inferred** (the encoding is the same words as sm_90; the execution was not run on a Blackwell card).
`VGPU_MUFU_EXACT=0` gives the earlier results (the PTX bf16 forms at host precision, rounded to nearest; counters do not change).

**Architectures**: the half-precision forms were measured on the RTX 3060 (sm_86) only. Turing (sm_75: the forms compile there),
Ada, Hopper and Blackwell cards were not asked; they get the model as **inferred** from the 32-bit unit, which the cards of
those architectures were shown to share ([below](#which-devices-get-the-model-and-on-what-evidence)), and the F16 cut is
**unknown** on them. `tools/numerics/mufu-card-run.sh` now also hashes the half-precision forms (`hashes16.txt`), so a card of
another architecture decides it in seconds. `VGPU_MUFU_EXACT=0` gives the earlier results for these forms too.

## Which devices get the model, and on what evidence

| Devices | Evidence | Word |
|---|---|---|
| RTX 3060 (sm_86) | the probe: every operand of every operation; `mathfn`, `mathfn_fast`, the PTX forms, Varity | measured |
| A10G (sm_86), H100 PCIe (sm_90) | `mufu-card-run.sh`: all ten operations over all 2^32 inputs, 64 hashes an operation, every one the RTX 3060's (2026-11-15; the A10G with nvcc 13.0, the H100 with nvcc 12.8) | measured |
| Tesla T4 (sm_75), L4 (sm_89), A10G (sm_86) | `mathfn` and `mathfn_fast`: 329 + 329 lines of 131,072 results a class identical to the RTX 3060's, with the card's own build and the sm_75 build. The probe did not run on them | measured through the math library; the instruction-level run (`mufu-card-run.sh`) is still to make |
| every other Turing, Ampere and Ada profile | the same architectures as the four | inferred from the four |
| the other Hopper profiles (h100, h100-nvl, h200, h200-nvl, gh200) | the same architecture as the H100 PCIe | inferred from the H100 PCIe |
| Blackwell sm_120 (RTX PRO 6000 Blackwell Server Edition) | `mathfn` (329 lines, both builds) and all ten operations over 2^32 inputs, 2026-11-17 | **measured** (see "Blackwell measured") |
| Blackwell (sm_100, sm_103 profiles) | none: no card of these ran `mathfn` or the probe | **inferred (measured on Turing, Ampere, Ada, Hopper and sm_120)** |

The model is not in the profiles: every device gets it. If a Blackwell card prints other hashes, the model
becomes a profile property, as the tensor-core sum is (`numerics.tensor_sum`, matrix-numerics.md).

## The switch

`VGPU_MUFU_EXACT=0` turns the model off in both engines: `MUFU`, the `.approx` forms and `div.full` are then what they
were in every earlier release (the function in double precision, rounded once; `div.full` the exact quotient), bit for
bit: with it the simulator prints the 329 + 329 lines pinned before this change. It is read at every launch, like
`VGPU_TC_EXACT`, and is part of the launch memo's key (as every `VGPU_*` variable that can change a result is). Cost of the
model: `MUFU.RCP`, `RSQ` and `SQRT` run as scalar integer code a lane where they were an AVX2 kernel a warp; the others
were scalar libm calls before and are a table lookup and three multiplies now. Counters do not change.

Measured (2026-11-15, one 16-core x86 host without a card, SASS executor, rtx3060 profile, the sm_86 programs of this directory
with `--n 4194304`, whole-program wall time, three runs each, alternating): `mathfn` 44.3, 44.4 and 48.3 s with the model against
43.2, 43.3 and 43.5 s with `VGPU_MUFU_EXACT=0`; `mathfn_fast` 43.7, 43.8 and 43.8 s against 42.9, 42.9 and 43.0 s. That is 2 to 3%
on programs that do little but call math functions; the time includes the programs' own host work, so the kernels' share is
larger than the figure by an amount not measured. No other program was timed.

## What is not done

- Blackwell: an sm_120 card ran `mathfn` and the probe (below); the sm_100 and sm_103 profiles get the model as *inferred*; a Blackwell card's
  `mufu-card-run.sh` output decides it.
- The probe ran on the RTX 3060, and its hash pass on an A10G and an H100 PCIe. The T4 and the L4 are measured through the
  math library (329 + 329 lines), not instruction by instruction.
- `mathfn` under a toolkit other than CUDA 13.0: an H100's CUDA 12.8 build differs on 27 lines, and no simulator run of that
  build was compared with it.
- The half-precision forms: the F16 forms are the card's on the RTX 3060 (the previous section), measured on that card alone;
  Turing, Ada and Blackwell were not measured for these forms. The bf16 forms (sm_90 and later) are measured on an H100 PCIe
  (the previous sections); Blackwell has the same encoding and is inferred.
- The AMD math instructions (`v_rcp_f32`, `v_rsq_f32`, `v_exp_f32`, `v_log_f32`, `v_sin_f32`, `v_cos_f32`) were unchanged by
  that round, and no AMD card had run any of it (`v_rcp_f32`, `v_rsq_f32` and `v_sqrt_f32` of gfx1030 are the card's since 2026-11-25, above).
- The NaN an `erf(double)` of a NaN returns, and `hypot(double)` of a signaling NaN and a zero, still differ between the
  engines (three lines of 329). The SASS executor's is the card's; the PTX interpreter's instruction was not traced.
- The PTX interpreter mirrors ptxas for the forms listed under "The PTX forms", from the SASS of nvcc 13.0 for sm_75, sm_86
  and sm_90 (the float instructions of the three are the same; sm_100 and sm_120 code was not compared for these forms).
  `rcp.approx.f64` without `.ftz` and `sqrt.approx.f64` are not accepted by ptxas 13.0 and stay as they were.
- How the card's own driver compiles a PTX module at run time (`cuModuleLoadData` of PTX text) was not compared with what
  nvcc's ptxas makes; the expansions are nvcc's.
- Speed: the cost of the model in a whole program was not measured.
- Xscope was not run and its paper not read. The Bessel functions of the accuracy paper are not in `mathfn`.
- The AMD lines have no test in CI (it has no hipcc); they were run by hand. The NVIDIA lines do (`e2e_numerics_mathfn`).

## Tests

- `test_mufu_model` (`tests/unit/test_mufu_model.cpp`; no card, no compiler): the model on the card's 3,100 vectors and on
  the card's hashes (a chunk of 2^26 operands an operation; every operand with `VGPU_EXHAUSTIVE=1`); `sq19` against its
  definition for every `x`; the SASS executor on the vectors, running the probe's own kernel (the cubin nvcc made, with
  `MUFU` rewritten as the probe rewrites it) through the fast path, the general path and with the operand modifiers; the
  probe's hash kernel on the SASS executor; the PTX interpreter on the vectors and on the card's 1,452 results of the PTX
  forms; `VGPU_MUFU_EXACT=0` in both engines.
- `e2e_numerics_mathfn` (`nvidia/tests/e2e/run_numerics_mathfn.sh`): the cards' lines pinned
  (`nvidia/tests/data/numerics_mathfn.txt`, `numerics_mathfn_fast.txt`: an RTX 3060's since 2026-11-15, the simulator's own
  bits before), from four SASS runs (two profiles of each of two architectures); PTX against SASS line by line with the
  listed exceptions; `fp_examples`' documented answers; `compare.py` on a doctored copy; with `VGPU_MUFU_EXACT=0`, 66 and
  104 lines are not the card's, as before the model.
- `test_mufu_model`, the half-precision forms: the model against the card's hashes over all 65,536 patterns of each of the eight
  `MUFU.<op>.F16` and of the four PTX forms; the SASS executor (probe cubin with the F16 bit set) over all 65,536 registers of
  each operation, through the fast and the general path, and with `.H1`, `|x|` and `-x`; the PTX interpreter on all 65,536
  halves of `ex2.approx.f16`, `tanh.approx.f16` and on the 65,536 registers of the `.f16x2` forms; `VGPU_MUFU_EXACT=0`, which
  gives the earlier results (19,417 and 14,736 patterns off the card for the two PTX forms). Mutants: [mathfn/mutants.txt](mathfn/mutants.txt)
  (six edits to the half-precision code, each failing at least one case).
- `test_mufu_model`, the bf16 forms and the sm_90 encoding: `eval_bf16` against the H100's 12 hashes (65,536 inputs each), the
  decoder on sm_86, sm_90 and sm_120 words (bit 72 is `.F16` and bit 73 is `.BF16` from sm_90; bit 73 is `.F16` before), the SASS
  executor on an sm_90 cubin with the BF16 form and the F16 form (bit 72), and the PTX interpreter on the four PTX bf16 forms.
- `test_ptx_approx_forms`: each expansion, on inputs where it differs from the function at host precision; the
  single-precision values are the card's now.
- `test_amd_hipcc_scratch_flat`: the wave32 stack slot.
- Mutants: [mathfn/mutants.txt](mathfn/mutants.txt). The before-and-after counts of the engines:
  [mathfn/engines.txt](mathfn/engines.txt). The simulator against the published figures, every row (before the model):
  [mathfn/published-vs-simulator.txt](mathfn/published-vs-simulator.txt).

### Pinned files the model changed

Every one was re-recorded from a card run, not from the simulator, and says so in its header:

| File | What changed | Re-recorded from |
|---|---|---|
| `nvidia/tests/data/numerics_mathfn.txt` | 66 of 329 lines (the functions listed under "The card runs") | an RTX 3060's `mathfn --n 4096` (sm_86 build) |
| `nvidia/tests/data/numerics_mathfn_fast.txt` | 104 of 329 lines | an RTX 3060's `mathfn_fast --n 4096` |
| `tests/unit/test_ptx_approx_forms.cpp` | 10 expected values of `ex2.approx.f32` (2), `lg2.approx.f32` (1), `sin.approx` (5) and `cos.approx.ftz.f32` (2): nine in their last bits, the cosine of 3e18 altogether | the same instructions and operands on an RTX 3060 (`mufu_forms.cu`'s kernels) |
| `tests/unit/test_exec3.cpp` | 1 expected value: `div.full.f32` of 0x3cae08fc by 0x41e66e7d is 0x3a415888 (`a * MUFU.RCP(b)`), was the exact quotient 0x3a415887 | the same instruction on an RTX 3060 |

Run and unchanged (they pass as they are, with the model on): every other unit test of the PTX interpreter, the SASS executor,
the tier, the JIT, the SIMD kernels and the memo (129 test programs), and the e2e tests whose programs
use these instructions: `e2e_simd_lane_forms`, `e2e_float_half_forms`, `e2e_round9_probe` (their card lines), `e2e_ptx_sweep`,
`e2e_div_approx`, `e2e_sass_tier`, `e2e_sass_fastpath` and `e2e_sass_archs` (every e2e program built for
ten architectures, each run from its SASS and from its PTX, which must print the same: 300 program and architecture pairs
agree; its one failure on the build machine, `runtime_conformance` for sm_75, is two driver-entry checks of a partial build
without the driver shim and fails the same way with `VGPU_MUFU_EXACT=0`). `numerics_cases.txt` did not change (the cards
print it); `numerics_engines.txt` changed in its comment only. The rest of the e2e suite (it needs nvcc on the build machine)
was not run for this change: CI is the check, and `VGPU_MUFU_EXACT=0` shows whether a difference is this change's.

## Blackwell measured (RTX PRO 6000 Blackwell Server Edition, 2026-11-17)

AWS g7e.2xlarge (us-east-2), driver 595.112.07, nvcc 13.0.88 and 13.2.51 (both present on the image). Issue #664.

- `tools/numerics/mufu-card-run.sh`: all ten approximate-math operations (COS, SIN, EX2, LG2, RCP, RSQ, RCP64H, RSQ64H, SQRT, TANH) give the
  RTX 3060's hashes on all 2^32 inputs. With the A10G and the H100 PCIe that is Ampere, Hopper and now Blackwell (sm_120) on which the model is exact. The
  PTX-level half forms (`ex2.approx.f16`, `.f16x2`, `tanh.approx.f16`, `.f16x2`) are the RTX 3060's too. The script's eight raw `MUFU.<op>.F16`
  rows say "DIFFER" on this card, and that is the probe, not the card: it rewrites the opcode bits of an sm_86 encoding, and on sm_120 the same bits
  decode as `MUFU.<op>.BF16` (disassembled: `MUFU.EX2.BF16`), so what was hashed is the bf16 form. The probe needs an sm_120 encoding for the F16 rows before
  they say anything about this card. (nvdisasm was not on the image; it was installed with `apt-get install cuda-nvdisasm-13-0`.)
- `mathfn`, CUDA 13.0, `--n 4096`, own-architecture (sm_120) build and sm_75 build: the default build is 329 of 329 lines of the pinned file
  (`nvidia/tests/data/numerics_mathfn.txt`) in both; `-use_fast_math` is 329 of 329 from the sm_75 build (the driver compiles the PTX) and **326 of 329 from the
  sm_120 build**: `log`, `log2` and `log10` in double, class `edges`. The inputs are the negative subnormals (-5e-324, -1e-323): nvcc's sm_120
  code returns NaN (log of a negative number), the sm_75 build's JIT result is -inf. Same three lines with CUDA 13.2 at `--n 131072`; a Tesla T4 (sm_75 build)
  and this card's sm_75 build print the same 329 lines in both builds. This is the compiler's code for the double `log` under fast-math, not an
  instruction of the card; the simulator's SASS executor runs the sm_120 cubin and prints the card's 329 of 329 for both programs on `rtx-pro-6000-server`
  (the PTX interpreter 326: the three NaN-payload lines of `numerics_engines.txt`). No test pins an sm_120 build's lines.
- `fp_examples`, `fp_examples_nofma` and `varity_cases` (-O0, -O3) print the same lines on this card and on the T4. The generated Varity programs
  differ between runs (the generator is not seeded in `card-run.sh`), so their 900 lines a build are not compared across cards.
