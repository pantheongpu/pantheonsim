# Narrow-precision paths: FP8, FP4, block scales

FP8 and FP4 matmuls, their scale modes and the conversions around them are the
part of the NVIDIA stack the RTX 3060 on the development machine cannot run:
FP8 needs sm_89, the block-scaled modes sm_90 and sm_100. This page says which
parts were checked against NVIDIA's own libraries on a GPU that has them, how,
and which parts are still written from NVIDIA's documentation alone.

## How a path is checked

Each probe is one program that prints one line per case. The same source runs
against NVIDIA's library on a real GPU (`--card`) and against the shims on a
simulated one; the card's transcript is committed, and the simulator must
print it back, line for line. The cases use small exact values (halves,
integers) so a product and a sum are exact and the order of accumulation does
not matter; each line carries the status of every call and an FNV-1a hash of
what was written (D, the auxiliary output, the amax values, the block scales).

| Probe | Program | Library | Cases |
| --- | --- | --- | --- |
| `lt` | `nvidia/tests/e2e/lowprec_lt.cu` | cuBLASLt | 1278 descriptors |
| `sparselt` | `nvidia/tests/e2e/lowprec_sparselt.cpp` | cuSPARSELt | 1096 problems |
| `cvt` | `nvidia/tests/e2e/lowprec_cvt.cu` | PTX `cvt` to and from e4m3x2, e5m2x2 | 9 forms x 512 values |
| `ptx120` | `mma_blockscale.cu`, `narrow_cvt.cu`, `ldmatrix_forms.cu` | sm_120a's `mma.sync` block-scaled forms, fp4/fp6 conversions, ldmatrix expansions | (no card transcript yet, below) |

```
nvidia/tests/e2e/run_lowprec.sh lt                       # the simulator, every profile that has a transcript
nvidia/tests/e2e/run_lowprec.sh lt l4                    # one profile
nvidia/tests/e2e/run_lowprec.sh lt --card l4             # NVIDIA's library on this GPU, against the transcript
nvidia/tests/e2e/run_lowprec.sh lt --card l4 --update    # rewrite the transcript from this GPU
```

The transcripts are `nvidia/tests/data/lowprec/<probe>.<profile>.txt` (the
profile is `nvidia/<name>`); a line starting with `#` names the machine and is
not compared. `PROBE_DUMP=<dir>` makes a probe write every case's inputs and
outputs there, to find which element differs. `LOWPREC_LIB_DIR` points `--card`
at another set of NVIDIA libraries (a pip wheel's, for instance).

## What has a card behind it

| Profile | Read from | What |
| --- | --- | --- |
| `nvidia/l4` (sm_89) | an AWS g6.xlarge (three sessions), driver 595.91, cuBLAS 13.3 (CUDA 13.2), cuSPARSELt 0.10.0.12 | `lt`, `sparselt`, `cvt`; `cuda_attributes_l4.card.txt`; `nvidia-smi -q` (serial number, PDI and UUID redacted) |
| `nvidia/rtx3060` (sm_86) | the development machine, the same libraries | `lt`, `sparselt`; cuDNN 9.27 for `dnn_int8x32` |

Nothing was measured on Hopper or Blackwell: AWS had no `p5.4xlarge` or
`g7e.2xlarge` capacity in us-east-1 at nine attempts over four hours (every zone
for `p5.4xlarge`; for a while no `g6.xlarge` or `g6e.xlarge` either), the Spot
quota for the G and P families is 0, and no other region has any quota for them.
So the H100, RTX PRO 6000 and B200 columns are documentation-derived (below), and
the `ptx120` probe has its programs and runner but no transcript. Nothing was
launched that was not terminated: three `g6.xlarge` sessions in all (about two
hours of GPU time, under $2).

The cuBLAS version matters. The transcripts are cuBLAS 13.3's (CUDA 13.2). On the
L4 the CUDA 13.0 library differs from it on 9 lines of the first 1247 (only the
per-batch scale mode, which 13.0's header does not have), 12.9's on 93 and 12.8's on
172, so a program built for an older cuBLAS can see older answers. cuBLAS 13.8
(the pip wheel, `nvidia-cublas==13.8.1.7`, kept as
`nvidia/tests/data/lowprec/other/lt.l4.cublas13.8.txt`) differs on 52 lines: it
knows the packed `VEC128/VEC32_MN_K4_UE8M0` modes (it answers `NOT_SUPPORTED` on the
L4, in the same family as `VEC32_UE8M0`, where 13.3 answers `INVALID_VALUE` for an
unknown mode) and has fp32-output kernels for `GELU_BIAS`, `RELU_AUX_BIAS` and
`GELU_AUX_BIAS` that 13.3 refuses. The shim follows 13.3.

A transcript lists the cases its card ran. `run_lowprec.sh` compares those and not
the cases a probe has grown since, so a probe can gain a group before every card has
run it (the L4 and RTX 3060 transcripts are the same 1278 now).

## What the cards say

**cuBLASLt FP8, below sm_89.** Every FP8 or FP4 descriptor is refused: the
heuristic returns no algorithm (`NOT_SUPPORTED`), or `INVALID_VALUE` where the
descriptor is wrong in itself. The same rules, in this order, hold on the L4:

1. *The descriptor's own consistency* (the same on both GPUs): E5M2 x E5M2 has
   no kernel anywhere; a scale type other than fp32 under an fp32 compute type;
   `AMAX_D` set under a 16- or 32-bit D; the auxiliary scale or amax pointer
   (before Hopper); an explicit auxiliary type under ReLU, or one that is not D's
   (bf16 under an FP8 D); C and D in different orders; scale modes of different
   families for A and B (scalar, FP32 vectors, `VEC16`, `VEC32` and the
   per-batch scalar are families; `VEC16` goes with an FP4 A, `VEC32` with an
   FP8 A); a block mode without its scale pointers; an FP4 D without
   `VEC16`; an E4M3 D under a block or vector scale with no D_OUT mode; an
   auxiliary output under a block scale.
2. *Whether a kernel exists* (sm_89 only; nothing below): A and B E4M3 or E5M2
   and both contiguous along K (op(A) = T and op(B) = N for column-major
   storage, or the row-major equivalents); C and D bf16, fp16 or fp32 alike, or
   C bf16/fp16 under an E4M3 D (E5M2 D only with one E5M2 operand); fp32
   compute (also its `FAST_16F` and `FAST_TF32` forms, not 16F or 64F); every
   leading dimension a multiple of 16 bytes; scalar scales only; the bias in
   BF16 under an FP8 or fp32 D, else in D's type; no `GELU_BIAS`,
   `RELU_AUX_BIAS` or `GELU_AUX_BIAS` into fp32; the backward epilogues
   (`DRELU`, `DGELU`) into 16-bit and fp32 D, their `_BGRAD` forms only into 16-bit,
   none into FP8; no bias gradient from A or B; into E5M2 only the epilogues
   without an auxiliary output or GELU.

**cuBLASLt arithmetic.** The product and the epilogue are fp32 (a bias that is
not exact in fp32 moves an FP8 rounding tie); FP8 D rounds to nearest even and
saturates; `AMAX_D` is the largest magnitude before D's scale, over every batch;
`RELU_AUX_BIAS` and `GELU_AUX_BIAS` add the bias to the product already rounded
to D's type (fp16 under an FP8 D); the bias gradient sums D as stored, in fp32;
under an FP8 D the default auxiliary output is bf16; `beta != 0` with no C is
`INVALID_VALUE` before any algorithm is looked for. The L4's results are
reproduced bit for bit; GELU alone is checked against the function (the card's
fp32 tanh is approximate), not hashed.

**cuSPARSELt.** FP8 (E4M3, E5M2) works on sm_89 as int8 does: fp32 compute into
fp16, bf16 or fp32 with both operands contiguous along K, the compressed layout
and sizes of int8, GELU only into bf16; an RTX 3060 accepts the matmul
descriptor and refuses at the algorithm selection. Scale pointers are ignored,
the block scale modes and FP8 outputs are refused (on both GPUs). E2M1 passes
the matmul descriptor for K a multiple of 128 and is refused at the algorithm
selection below Blackwell. fp16 compute has no kernel on either. The plan's
workspace is the size of the compression buffer, and a plan sizes fp16
metadata from the operand's logical shape (a bare descriptor from the larger of
the two orientations).

**FP8 conversions.** `cvt` to e4m3x2 and e5m2x2 with `.satfinite` saturates;
`.relu` sends -0 and every negative to +0; a NaN becomes 0x7F in E5M2 as in E4M3.

**cuDNN.** An INT8x32 convolution in the graph API with the filter reordered by
`cudnnReorderFilterAndBias` (`CUDNN_TENSOR_REORDERING_INT8x32`) gives the host
reference on an RTX 3060 and on the shim (`dnn_int8x32.cu`).

## Still written from the documentation

Marked "documentation-derived" in the code, and not checked against a card:

- cuBLASLt from sm_90: which FP8 layouts, types and epilogues each GPU takes
  (the shim applies the L4's descriptor checks and no kernel table); outer-vector,
  128-element and 128x128 scales (Hopper and later); the auxiliary output's own
  scale and amax for an FP8 auxiliary type; MXFP8 (`VEC32_UE8M0`), NVFP4
  (`VEC16_UE4M3`), D's block quantization and its output scales (Blackwell);
  the tiled scale layout; one scale tensor per batch back to back;
  `PER_BATCH_SCALAR_32F`; cuBLAS 13.8's packed `VEC128/VEC32_MN_K4_UE8M0`.
- cuSPARSELt on Hopper and Blackwell, and sparse FP4.
- cuDNN FP8 and MXFP8 attention (not implemented), and the F16x16 and FP8-128x4
  tensor reorderings.
- The block-scaled `tcgen05` forms of `docs/blackwell.md` (sm_100 only; AWS offers
  it only as an eight-GPU p6-b200.48xlarge, which the round's rules exclude).
- sm_120's `mma.sync` block-scaled forms, fp4/fp6 conversions and ldmatrix
  expansions: the program set `ptx120` is ready, its transcript needs a g7e.

## Running it on a GPU that has them

On an H100 (`p5.4xlarge`) or RTX PRO 6000 (`g7e.2xlarge`), with the repository's
`nvidia/tests`, `tests/shim_guard.sh` and `nvidia/include/vgpu_cusparselt.h` copied
over and cuSPARSELt's `libcusparseLt.so.0` (the pip wheel `nvidia-cusparselt-cu13`):

```
run_lowprec.sh lt --card h100 --update          # cuBLASLt from the toolkit, the transcript for nvidia/h100
LOWPREC_LIB_DIR=<wheel lib dir> run_lowprec.sh sparselt --card h100 --update
run_lowprec.sh cvt --card h100 --update
run_lowprec.sh ptx120 --card rtx-pro-6000 --update
```

then `run_lowprec.sh <probe>` on the simulator shows what the shim does not yet
reproduce there: the L4's rules are what it applies.
