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
| `ptx120` | `mma_blockscale.cu`, `narrow_cvt.cu`, `ldmatrix_forms.cu` | sm_120a's `mma.sync` block-scaled forms, fp4/fp6 conversions, ldmatrix expansions | 29 + 12 + 12 hashes |

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
| `nvidia/l40s` (sm_89) | an AWS g6e.2xlarge (2026-10-09), driver 595.91.07, cuBLAS 13.3 (wheel `nvidia-cublas==13.3.*`), cuSPARSELt 0.10.0.12 | `lt`, `sparselt`, `cvt`: line for line the L4's (the simulator's `l40s` profile reproduces them) |
| `nvidia/rtx-pro-6000-server` (sm_120) | an AWS g7e.2xlarge (2026-10-09), driver 595.91.07, cuBLAS 13.3 (wheel), cuSPARSELt 0.10.0.12 | `lt`, `sparselt`, `cvt`, `ptx120`; `cuda_attributes_rtx-pro-6000-server.card.txt`; `nvidia-smi -q`; the programs of `nvidia/tests/data/card/rtx-pro-6000-server/`. The simulator reproduces `lt` and `cvt` line for line, `sparselt` but for one case of 1097 and `ptx120` for 18 of the 29 `mma_blockscale` forms (`known-gaps.txt`; "What the RTX PRO 6000 says" below) |
| `nvidia/rtx3060` (sm_86) | the development machine, the same libraries | `lt`, `sparselt`; cuDNN 9.27 for `dnn_int8x32` |

Nothing was measured on Hopper (no `p5.4xlarge` capacity came in any of three regions over five hours) or on a data-center Blackwell; an L40S (g6e.2xlarge) answered exactly as the L4 did, and an RTX PRO 6000 Blackwell Server Edition (g7e.2xlarge, us-east-2, after a quota increase there) gave the sm_120 transcripts below. Earlier: AWS had no `p5.4xlarge` or
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

## What the RTX PRO 6000 says (sm_120)

Read on 2026-10-09 from an AWS g7e.2xlarge. The shims first followed the L4's rules for every GPU of compute
capability 9 and later, plus documentation for the block-scaled modes, and the card disagreed with that in
three places (539 of 1279 `lt` cases, 99 of 1097 `sparselt` cases, 23 of 29 `mma_blockscale` hashes). Round 5
closed most of it from the transcripts and from the probes' raw dumps (`PROBE_DUMP`, kept in the session
archive, not committed: 13 MB); what is left is listed at the end of each bullet. `known-gaps.txt` lists the
probes that still differ; `run_lowprec.sh <probe> rtx-pro-6000-server` prints the diff (keep the simulator's
side with `LOWPREC_KEEP=<dir>`).

* **cuBLASLt (`lt`): reproduced, 1279 of 1279 lines.** The card's kernel table is the L4's, plus the
  differences below (`nvidia/src/cublaslt_api.cpp`, `narrow_kernel_check`; only compute capability 12.0 follows
  it, 8.9 and the documentation-derived Hopper and data-center Blackwell paths are unchanged). Of the 1278 cases
  the card answers the L4's way in all but 116, and in those it is more permissive:
  - *Layout.* The L4 wants A and B both contiguous along K. With a column-major D the card wants only B
    (`NN` and `TN`, and any storage orders that make B K-contiguous: `fp8order`), and with a row-major D only
    A (D^T = op(B)^T op(A)^T swaps the roles). The backward epilogues (DRELU, DGELU and their BGRAD forms)
    want both. The block-scaled kernels want both (TN only).
  - *Alignment.* A and B need a 16-byte leading dimension; C and D need a 16-byte contiguous extent (the
    rows of a column-major D: `fp8dims 8x8x16` with an E4M3 D and `17x16x16` are refused) but accept any
    leading dimension (`fp8ld pad_d1`).
  - *Block-scaled matmuls* have kernels: MXFP8 (UE8M0 per 32) and NVFP4 (UE4M3 per 16), A and B of one kind, C
    and D of one type (bf16, f16, f32) or an E4M3 or FP4 D with a bf16 or fp16 C, the default, ReLU, bias
    and GELU epilogues (not the auxiliary ones), batches, `alpha`/`beta`; an E5M2 D, an FP8 D under FP4
    operands, E5M2 x E5M2 and a block-quantized D with fewer than 32 rows are refused.
  - *Refused as on the L4:* the Hopper 128-element and 128x128 FP32 scale modes, outer-vector and per-batch
    scales, an auxiliary scale or amax, an FP8 auxiliary type (so that part of the older "documentation-derived"
    text no longer applies to 12.0), and `VEC128/VEC32_MN_K4_UE8M0` with INVALID_VALUE (cuBLAS 13.3 does not
    know them; they are taken from 10.0 up, not on 12.x).
  - *D's output scale tensor* is written only where D has block scales: the tile padding keeps whatever the
    buffer held (the probe's 0xEE), and the quantization multiplies each value by the scale's **fp32
    reciprocal**, which is why -300 over an E4M3 scale of 120 reaches the E2M1 rounding as -2.5000002 and goes to
    -3 (a double reciprocal gave a tie, and -2).
  - *Natively run programs.* The card's own checks in `lt_paths`/`lt_blockscaled_paths` (the program run
    on the card, `card/` directory) agree: VEC128_32F/BLK128x128_32F A/B scales are refused on sm_120, and the MX/NV
    D-scale checks of that test (written from the documentation) fail on the card. Run on the simulator with
    the `rtx-pro-6000-server` profile, `lt_paths` and `lt_blockscaled_paths` now fail the same checks, one and
    four, as the card does.
* **cuSPARSELt (`sparselt`): reproduced but for one of 1097 cases** (`nvidia/src/cusparselt_api.cpp`, "sm_120"
  comments; sm_120 only, the others keep the L4's rules).
  - *Sizes.* FP8 and FP4 compress to values + metadata where the metadata is 2048 bytes for every 128 x 128
    tile of the logical operand (128 x 256 for FP4), whole tiles (the L4's tiles are 64 x 128: `dims/e4m3_*`);
    the compress buffer is a constant 49216 bytes (FP4: 98368), whatever the shape or batch count, and the
    plan asks for no workspace. The card's metadata *layout* inside the region was not decoded: the library
    keeps its codes in the L4's layout when that fits and row-major otherwise, so a compressed buffer is
    readable by this library only.
  - *Behaviour.* ReLU gives +0 for a negative value (the L4 and the 3060: a signed zero), GELU takes E4M3/E5M2
    into fp16, bf16 and fp32, `cusparseLtMatmulSearch` on an FP8 or FP4 plan succeeds and leaves D alone.
  - *FP4 (E2M1) is sparse in pairs:* of every 8 values along K, two pairs of neighbours are kept (4:8), the two of
    larger |a| + |b|, the earlier on a tie (STRIP; all 1024 groups of `dims/e2m1_64x64x128` agree); the
    metadata is one code per group of eight; TILE pruning of FP4 was not measured and is refused.
  - *Block scales.* FP4 matmuls **always read A's and B's scale pointers as tiled UE4M3 tensors, one scale per 32
    values of K, whatever the scale modes say** (SCALAR with the probe's 64-byte buffer of floats reads zeros
    beyond it and gives the card's `dims/e2m1_*` output, bit for bit; VEC64_UE8M0 reads the same bytes as
    E4M3, so a 0x7F byte makes NaNs: the probe's `e2m1_vec64_bf16` D). FP8 (E4M3 only; E5M2 is refused at the
    algorithm selection, and so are fp16 and int8) takes UE8M0 per 64 values of K when either block mode is set
    (VEC32_UE4M3 and VEC64_UE8M0 gave the same D for the same bytes) and ignores the pointers otherwise, as the
    L4 does. The scale layout is the tiled one of cuBLASLt (128 rows by 4 blocks), and the sum is taken block
    by block: the block's products summed, times the scales, **rounded toward zero to fp32**, added to the
    accumulator, which rounds toward zero as well; a zero is +0.
  - *What is left: `scales/e4m3_vec32_128x128x256`.* Its scale bytes (the probe's UE4M3 table read as UE8M0)
    make every term denormal-sized, and 14 of its 16384 cells end one denormal bit, or a sign of zero, away from
    the card's (the best rounding model tried leaves 11). The case is the only line left in `known-gaps.txt`.
* **`mma.sync` block-scaled forms (`ptx120`): 18 of 29 `mma_blockscale` hashes reproduced** (was 6). The rule
  both engines now follow on compute capability 12 for E4M3, E5M2, E3M2, E2M3, E2M1 operands (`exec::mma_narrow_sum`,
  `exec_mma` in `src/exec/interpreter.cpp`, QMMA in `src/sass/exec_ops.inc`): the products and their block scales
  are summed **exactly with C and rounded once, toward zero, to fp32** (RZ, not the nearest-even float sums the
  engines did), a zero result is +0, NaN/Inf as IEEE; and an **E2M1 operand in a byte is read as an E2M3 of the byte's low
  six bits** (the same number when bits 0-1 and 6-7 are zero, as the ISA's layout leaves them; random bytes in the
  container are what the probe has). This took `narrow_cvt` and `ldmatrix_forms` (24 of 24, unchanged) to
  `mma_blockscale`'s `f8f6f4_e4m3_e2m1`, `e3m2_*`, `e2m3_*`, `e2m1_*`, `mx8_e2m1_e3m2` (3), `mxf4` (2), `nvf4`
  (4) forms. The derivation is in `nvidia/tests/e2e/mma_accumulate_probe.cu`'s header, and a Python re-implementation
  of the inputs and rules agrees with the card on exactly these 18 and with the old rule on the others.
  - *What is left: nine dense forms and the two sparse ones* (`f8f6f4_e4m3_e4m3`, `e5m2_e4m3`, `e5m2_e2m1`, the three
    `mx8_e4m3_e4m3` and three `mx8_e5m2_e2m3`, `sp_mx8`). Every one has an E4M3 or E5M2 operand against another
    that spans many binades (products from 2^-18 to 2^17 or 2^-32 to 2^30), NaNs and infinities among the
    operands, or both; the 18 that match have none of the combination. Not an accumulator of limited width
    (every fixed-point alignment from 8 to 47 bits, grouped by 4, 8, 16, 32 consecutive terms, with C in or
    out of the alignment, was tried against all 29 hashes), not a product cap, not a different NaN or infinity
    result for a cell (all 8 infinity cells of `e5m2_e2m1` were enumerated), not the NaN pattern, and not the
    sticky tail of a sparser sum. The hashes cannot say which cells differ. `nvidia/tests/e2e/mma_blockscale_dump.cu`
    prints every lane's four results of the 29 forms raw and `mma_accumulate_probe.cu` runs controlled
    experiments (gap, position, C, special values, signs of zero) for E4M3 x E4M3, E5M2 x E4M3, E3M2 x E4M3,
    E5M2 x E2M1 and the MX form; `nvidia/tools/card-session-remote.sh` runs both on a g7e (about a minute), and the
    output is the next step. No AWS instance was launched for this round: two other GPU instances
    (`pw-stage2a-a`, `pw-stage2a-b`) were already alive in us-east-1, which is the ledger's two-instance cap.
* **Attributes.** Against `device_attributes --dump`: the boost clock (2430 MHz, was 2617), the memory clock,
  the persisting L2 (83886080), the memory total and `MaxAccessPolicyWindowSize` (134217728 on sm_120, 134213632
  below; fixed in `nvidia/src/device_attributes.cpp`) and `UnifiedFunctionPointers` (1; fixed) were the
  simulator's. The other differences are the capabilities the simulator does not implement (the same
  categories as the L4's), and the PCI bus id.
* **Natively run programs (`nvidia/tests/data/card/rtx-pro-6000-server/native/`).** `mma_blockscale`,
  `dsmem_cluster`, `cooperative_cluster`, `stmatrix`, `vector_atomics`, `mma_fragment_layout`, `wmma_gemm`,
  `uldc_narrow`, `lt_epilogue_paths` pass on the card. `mma_forms` fails one form of 123
  (`s8_m16n8k32_satfinite`: the card gives 78eaef8c79913f66, the RTX 3060 4168a5c5ce5afb48 that the test
  expects), `wmma_types` does not compile for sm_120a (the CUDA 13.2 headers: unsupported operation), and the two
  FP8 attention tests fail on the card's cuDNN 9.27: the output quantized to E4M3 is 4.76 FP8 steps from the
  documented formula in the backend test and 29 in the frontend's (S's amax and O's amax are as documented), so
  the documented formula for `scale_O` quantization is not what cuDNN computes on sm_120.
  - *FP8 attention: the committed log has only the verdicts.* The worst element of the backend test is 4.7619
    steps, which as the test measures it (tolerance 0.13 |want| + 0.02) is a pair such as want 0.25 against 0
    or 0.5, or 5.5 against 2 or 9: a factor of two or the value flushed, not a rounding. Both amax values agree
    with the formulas (one is a maximum, so a permuted or mis-scaled O could still produce it). That points at
    where O8 is written or how scale_O reaches it, and the bytes are needed to say which.
    `nvidia/tests/e2e/dnn_fp8_attention_probe.cu` is the program for the next g7e: it runs the same graph in
    thirteen configurations (each factor on its own, scales that are not powers of two, and "uniform" ones, Q = 0
    and V constant, whose answer is E4M3(descale_V V scale_O) in closed form) and prints the raw O8 bytes, both
    amax values and, for each of nine variants of the formulas (no E4M3 rounding of P, scale_S not used, no
    descale_S, no scale_O, scale_O twice, round toward zero, fp16 O, a row-max rescale of P) and two layouts
    (O read as [S][H][D] or [H][D][S]), how many of the 256 bytes it reproduces. On a Hopper or Blackwell profile of the
    simulator the documented row matches in full, which checks the probe.

## Still written from the documentation

Marked "documentation-derived" in the code, and not checked against a card:

- cuBLASLt from sm_90, except an RTX PRO 6000 (12.0, measured above): which FP8 layouts, types and epilogues each GPU takes
  (the shim applies the L4's descriptor checks and no kernel table); outer-vector,
  128-element and 128x128 scales (Hopper and later); the auxiliary output's own
  scale and amax for an FP8 auxiliary type; MXFP8 (`VEC32_UE8M0`), NVFP4
  (`VEC16_UE4M3`), D's block quantization and its output scales (Blackwell);
  the tiled scale layout; one scale tensor per batch back to back;
  `PER_BATCH_SCALAR_32F`; cuBLAS 13.8's packed `VEC128/VEC32_MN_K4_UE8M0`.
- cuSPARSELt on Hopper and Blackwell other than 12.0 (sparse FP4 and block scales are implemented for 12.0 only).
- cuDNN FP8 and MXFP8 attention (not implemented), and the F16x16 and FP8-128x4
  tensor reorderings.
- The block-scaled `tcgen05` forms of `docs/blackwell.md` (sm_100 only; AWS offers
  it only as an eight-GPU p6-b200.48xlarge, which the round's rules exclude).
- sm_120's `mma.sync` block-scaled forms: measured (above); the simulator does not reproduce 11 of the 29 hashes.

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
