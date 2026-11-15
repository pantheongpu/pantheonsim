# Math functions and scalar floating point: the simulator against published results

The tensor and matrix cores are one place where a simulated card's bits can differ from a real one's
([matrix-numerics.md](../matrix-numerics.md)). The math functions are the other, and the larger one: `sinf`, `expf`, `powf`
and the rest are library code (NVIDIA's libdevice, AMD's OCML) built on a few hardware instructions that approximate a
function (`MUFU.RCP/RSQ/EX2/LG2/SIN/COS/TANH` on NVIDIA; `v_rcp/v_rsq/v_exp/v_log/v_sin/v_cos` on AMD). The simulator computes
those instructions at the host's precision; a card reads them from tables that are within an ulp or two of that, so a
program that calls a math function can print other last bits on the simulator than on the card.

This page records the second group of published work of issue #664: what was compared, how, what the simulator's side of
the comparison found, and what a card run will add. **No card was run for this page.** Three words are used strictly, as in
matrix-numerics.md: **measured** (a card of ours ran it), **published** (a paper or post gives it; no card of ours has
checked it), **unmeasured** (neither). Everything about a card below is *published* or *unmeasured* unless it cites an
earlier measurement of ours.

Nothing here changes what the simulator's `MUFU` or the AMD math instructions return. That needs the card data this page
prepares for (a table per instruction and architecture) and is a later step.

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

### The Varity generator run

300 programs, four builds, three input sets: 3,600 results a configuration. On NVIDIA all 3,600 are the same on every
profile and, but for the five results above, on both engines. On AMD the two architectures differ on 34 to 65 results a build, and the
ten empty kernels above are not run. The paper's differences between vendors (Table IX: 13,877 of its FP32 `-O3`
fast-math tests) are between a card and a card; simulator against card is what `card-run.sh` adds.

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
measurement on an A10G ([docs/perf-design/12-host-gpu-turbo.md](../perf-design/12-host-gpu-turbo.md), section 5):
`MUFU.RCP` is 1 ulp off on about 13 % of mantissas, the card's `MUFU.SIN/COS` are fixed-point with about 3.6e-7 absolute
error, and `tanh` differs on 5.6 % of float patterns. `mathfn_fast` puts every function of the set on that footing in one
run.

## The card run

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

## What is not done

- No card run. No change to `MUFU` or to the AMD math instructions.
- The NaN an `erf(double)` of a NaN returns differs between the engines and was not traced.
- The PTX interpreter mirrors ptxas for the forms listed above, from the SASS of nvcc 13.0 for sm_75 to sm_120. Forms that
  do not occur in the test set (`rcp.approx.f64` without `.ftz`, `sqrt.approx.f64`, the half-precision forms) are as they
  were: the function at host precision.
- Xscope was not run and its paper not read. The Bessel functions of the accuracy paper are not in `mathfn`.
- The AMD lines have no test in CI (it has no hipcc); they were run by hand. The NVIDIA lines do (`e2e_numerics_mathfn`).

## Tests

- `e2e_numerics_mathfn` (`nvidia/tests/e2e/run_numerics_mathfn.sh`): the simulator's lines pinned
  (`nvidia/tests/data/numerics_*.txt`: its own bits, not a card's), PTX against SASS line by line with the listed
  exceptions, four profiles and two cubin architectures against the same pins, `fp_examples`' documented answers,
  `compare.py` on a doctored copy.
- `test_ptx_approx_forms`: each expansion, on inputs where it differs from the function at host precision.
- `test_amd_hipcc_scratch_flat`: the wave32 stack slot.
- Mutants: [mathfn/mutants.txt](mathfn/mutants.txt). The before-and-after counts of the engines:
  [mathfn/engines.txt](mathfn/engines.txt). The simulator against the published figures, every row:
  [mathfn/published-vs-simulator.txt](mathfn/published-vs-simulator.txt).
