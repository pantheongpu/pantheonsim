# Three more papers on matrix-unit numerics, run on a card and on the simulator

Pantheonsim issue #664 asked for the FTTN paper to be recreated and for the other papers on how tensor cores add to be run against
the simulator ([../matrix-numerics.md](../matrix-numerics.md) has FTTN). This page is the rest that needed no cloud card: the local
RTX 3060 (sm_86, driver 610.88, nvcc 13.0.88) and the simulator (profile `rtx3060`, both engines, and the profiles of other
architectures). Words as there: **measured** (a card of ours printed it), **published** (a paper or a published data set says it,
no card of ours has), **inferred**, **unknown**.

| Paper | Code and licence | What we ran | Our own code |
|---|---|---|---|
| Fasi, Higham, Mikaitis, Pranesh, "Numerical behavior of NVIDIA tensor cores", PeerJ CS 7:e330, 2021 | github.com/north-numerical-computing/tensor-cores-numerical-behavior, **GPL-2.0** | its programs, unmodified except for one bug fix (below), on the card and on the simulator; not copied | `tools/tcpapers/tc_features.cu` implements the features the paper lists |
| Khattak, Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004, and the HPEC 2025 test paper | github.com/north-numerical-computing/MATLAB-tensor-core (**BSD-2**, models and recorded test vectors); github.com/faiziktk/IEEE_HPEC2025_block_FMA_tests (**no licence stated**) | the HPEC feature program (run only, not copied) on both; the recorded vectors of 8 cards through the simulator and the card | `tools/tcpapers/khattak_vectors.cu`, `khattak_convert.py`, `khattak-fetch.sh`; 24 vectors a file of three cards are in the repository with the BSD notice |
| Xie, Gao, Wang, Xue, "Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations" (FPRev), USENIX ATC 2025, arXiv 2411.00442 | github.com/peichenxie/FPRev, **MIT** (also at microsoft/RepDL) | the method, on our own program | `tools/tcpapers/fprev_probe.cu` |

Run everything with `nvidia/tests/e2e/run_tcpapers.sh` (CTest `e2e_tcpapers`; `VGPU_KHATTAK_VECTORS=<dir>` adds the full sets).
Pinned files are in `nvidia/tests/data/tcpapers/` (under 100 KB; `NOTICE` says where each comes from).

## Headline

- On the RTX 3060 the simulator and the card print **the same** for every program of the three papers, on both engines: Fasi's four
  programs (all checks pass on both), the HPEC feature program (all values equal), our features and trees. No difference was found.
- The 5,000-vector validation sets that Khattak and Mikaitis recorded on **other cards** are reproduced bit for bit by the simulator
  with the profile of that card: A100, A2, Ada, L40S, H100, H200, and **B200 with the Hopper rule**: 20,000 vectors a card (f16
  with an f32 accumulator, f16 with an f16 accumulator, bf16, tf32), 0 mismatches, both engines. The sequential sum the b200
  profile used to have mismatches 8,020 of those 20,000 vectors. The rule for a B200 is therefore not "published only": a data set
  recorded on a B200 agrees with it (class **inferred**: data recorded by others, we have not run a B200).
  **Update 2026-11-16: the b200 and gb200 profiles now use the Hopper rule by default** (`numerics.tensor_sum: hopper`, class
  `inferred`); this changes the matrix results of those two profiles, and `VGPU_TC_EXACT=0` gives the earlier sequential results
  back. sm_103 (b300), sm_120 (rtx5090, rtx-pro-*), thor, gb10 and vr200 did not change: no recorded vectors cover them. Where
  this document says "b200 default" or "sequential" for a b200 below, it describes the run as recorded on 2026-11-15, which is
  what `VGPU_TC_EXACT=0` prints now; "with `VGPU_TC_EXACT=published`" for a b200 is the default now.
- The RTX 3060 itself reproduces the recorded A100, A2, Ada and L40S vectors (4 x 20,000 = 80,000 vectors, 0 mismatches): the
  Ampere rule is one rule across GA100, GA10x and AD10x for these shapes, now on data from five more card models than ours.
- FPRev's trees: the simulator's tensor-core loops have the structure the paper reports for A100 and H100 (a 9-way and a 17-way
  fused sum per step), equal to the card's on the RTX 3060.
- Mismatches that are real, all measured below: the simulator's **cuBLAS** matrix products add in one sequential chain for every
  data type, where the card's cuBLAS does not (table in the FPRev section).

## Fasi et al. (2021)

Their suite has one program a Volta, Turing and Ampere precision configuration. CUDA 13 cannot build sm_70, so `test-V100` was not
run; the other four were built for each architecture the simulator has a profile for.

- **The programs on the card and the simulator** (RTX 3060 against profile `rtx3060`, SASS and PTX): `T4-A100-binary16` 17 of 17
  checks pass, `A100-bf16` 13 of 13, `A100-tf32` 13 of 13, `A100-binary64` 11 of 11, printed text identical to the card's. Per
  profile and architecture in [tcpapers/external-programs.txt](tcpapers/external-programs.txt): t4, a100, l4 and rtx3060 print all
  passes (the paper: a T4 and an A100 differ only in binary64, which the 3060 does too).
- **Hopper and the sequential sum fail three checks** (binary16: "products are accumulated in binary32", "sum starts from largest
  element" and D5 "monotonicity of the dot product"; bf16 and tf32: the same three): the checks encode what a Volta, Turing or
  Ampere core does, and the h100 profile and the b200 profile (Hopper rule by default since 2026-11-16, `VGPU_TC_EXACT=published` before) print the same failures; the b200 profile with `VGPU_TC_EXACT=0` (the earlier default, sequential)
  fails more (bf16: 7 of 13). No H100 of ours has run the suite; on the simulator this is the Hopper rule working as set, not a finding about a card.
- **A bug in the suite.** `tc_test_numerics-A100-bf16.cu` and `-A100-tf32.cu` write `h_b[j-1]` outside the `if (j>0)` they
  meant (the missing braces): at `j == 0` they write one element before a heap array. On a card that is silent; on the simulator the
  heap check aborts the program at exit ("free(): invalid next size"). Found with AddressSanitizer; the simulator is not at fault.
  The runs above are of copies with the braces added; on the RTX 3060 the original and the copy print the same text. (Not reported
  upstream.)
- **Our own implementation** (`tc_features.cu`, one line a measurement, bits in hex; the RTX 3060's 34 lines are pinned): subnormal
  inputs and outputs, exact products, f32 accumulation, extra alignment bits (`align_max_j`: the largest j for which 1 + 2^-j - 1 is
  not lost; 24 is one extra bit, 25 two), whether the alignment truncates, the number of products added in one step (`fma_size`),
  the final rounding in both signs, a sum of sixteen equal products. A few lines, RTX 3060 = a100 = l4 = t4 (the Ampere rule,
  **measured** on the 3060, as Fasi found for T4 and A100) against h100 = b200 (default since 2026-11-16) = rtx-pro-6000 with `VGPU_TC_EXACT=published`:

| Feature (f16 inputs, f32 accumulator) | RTX 3060 and simulator ampere profiles | h100, b200 (default since 2026-11-16), rtx-pro-6000 with `VGPU_TC_EXACT=published` | b200 with `VGPU_TC_EXACT=0` (the earlier default, sequential) |
|---|---|---|---|
| subnormal inputs read (A1) | yes | yes | yes |
| `align_max_j` (extra alignment bits) | 24 (one bit) | 25 (two bits) | 28 (the scan's limit: no cut) |
| alignment truncates | yes (`2^-24`) | keeps the bit (`1.5 * 2^-24`) | keeps it |
| `fma_size` | 8 (bf16 8, tf32 4) | 16 (bf16 16, tf32 4 through wmma: two m16n8k4) | 2 (sequential) |
| final rounding f32 | toward zero | toward zero | to nearest even |
| final rounding f16 accumulator | to nearest even | to nearest even | to nearest even |

  Published numbers (Khattak and Mikaitis's models, `models/*.m`): A100 and Ada 8 products and 1 extra alignment bit, tf32 4;
  H100 and B200 16 products and 2 extra bits, tf32 8; all truncating. The simulator's H100 tf32 4 is the wmma shape (a 16x16x8
  fragment is two m16n8k4); the mma.m16n8k8 path is 8 ([../matrix-numerics.md](../matrix-numerics.md)). The pinned `tc_features.h100.txt`
  is the simulator's output, **not** an H100's.

## Khattak and Mikaitis (2025)

- **Feature tests (HPEC).** The program's values: RTX 3060 and the simulator's `rtx3060`: 1 extra alignment bit passes, 2 fails, FMA
  size 8 (tf32 4), truncation, delayed normalisation, binary16 output rounds to nearest even: identical on card and simulator, as in
  the paper's A100 row (extra alignment bits 1, FMA 8, truncation). The h100 profile prints 2 bits and FMA "0" (the program tops out
  below 16), carry 4: the Hopper rule's own numbers; not compared with a card. See `external-programs.txt`.
- **Recorded vectors, the real comparison.** `model_validation/` of their repository holds 5,000 dot products a card and format with
  the result of the card: V100, A100, A2, Ada, L40S, H100, H200, B200 (and the FP8 sets, run in [fp8-vectors.md](fp8-vectors.md)). `khattak_vectors.cu` runs a
  vector as a wmma with the vector in row 0 of A and column 0 of B and compares bits. Mismatches out of 5,000, per file
  (f16 with f32 accumulator / f16 with f16 accumulator / bf16 / tf32):

| Recorded card | RTX 3060 (card) | Simulator, that card's profile | Simulator, b200 with `VGPU_TC_EXACT=0` (the earlier default) |
|---|---|---|---|
| A100, A2, Ada, L40S | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 (a100; rtx3060 for the A2, an sm_86 part; rtx6000-ada and l40s) | - |
| H100, H200 | 1,757 / 1,311 / 1,080 / 498 (H100); 1,755 / 1,362 / 1,091 / 483 (H200) | 0 / 0 / 0 / 0 (h100, h200) | - |
| B200 | 1,812 / 1,316 / 1,073 / 481 | 0 / 0 / 0 / 0 (b200 by default since 2026-11-16, `VGPU_TC_EXACT=published` before; rtx-pro-6000 with `published`, the same vectors) | 3,168 / 2 / 2,584 / 2,266 (b200 with `VGPU_TC_EXACT=0`) |
| V100 | 1,200 (f32 accumulator), 0 (f16) | no profile | - |

  The vectors were converted to the program's own format (`tools/tcpapers/khattak-fetch.sh`, pinned commit 79143bd). A sample of
  them (24 a file, chosen so that an Ampere sum fails on 16 of the H100 and B200 ones) is in the repository and runs in CI; the
  two engines agree on every file. The rtx-pro-6000 line is the same B200 vectors: the paper says an RTX PRO 6000 Blackwell has the
  B200's model but its data set has no RTX PRO 6000 vectors, so **sm_120 stays inferred**.

## FPRev

The method needs only a way to run a sum with chosen inputs. Terms i and j are set to +M and -M and every other term to a small
"one"; the count of ones left in the result gives the size of the smallest sub-sum holding both, and the matrix of those sizes is the tree
(`fprev_probe.cu` says how). Our program, not FPRev's (MIT; not copied), runs on a card and the simulator unchanged. Printed
trees (`a-b` is the consecutive terms a to b, one node with several children is a fused sum, the accumulator is not a leaf):

| Kernel | n | RTX 3060 (card) = simulator rtx3060, a100 | simulator h100 (and b200 by default since 2026-11-16) |
|---|---|---|---|
| f32 loop `acc += x` | 32 | `(((0-1) 2) 3) ...` the chain | same |
| f32 `fma` loop | 32 | the chain | same |
| f32 warp `__shfl_down` tree | 32 | `((((0 16) (8 24)) ...` | same |
| f32 shared-memory halving tree | 64 | `(((((0 32) (16 48)) ...` | same |
| wmma f16 and bf16, f32 accumulator | 64 | `((((((((0-7) 8-15) 16-23) ...) 56-63)`: 8 products a step, each step a 9-way node | `((((0-15) 16-31) 32-47) 48-63)`: 16 a step, a 17-way node |
| wmma tf32 | 32 | 4 a step | 4 a step (wmma = two m16n8k4) |

The paper (arXiv 2411.00442 v3, section 6.2) reports a 5-way tree on a V100, 9-way on an A100 and 17-way on an H100 for half-precision
matrix multiplication: **(4+1), (8+1) and (16+1)-term fused sums**. The card's and the simulator's A100-rule trees are the 9-way
ones; the h100 profile's the 17-way one. The V100 is not run (no profile). An H100 of ours has not run FPRev, so the 17-way tree is
the paper's and the simulator's, not ours measured.

Two things the method taught us, on the way:

- **A probe artifact, fixed in our program.** With ones of 1 and M of 2^30, a fused sum that keeps 26 bits kept a running count of
  32 or more and drew a wrong H100 tree (`(32-47)` as a separate node). M has to be large enough that the count is lost beside it;
  the program uses 65504 * 65504 and ones of 2^-4 (the comment says why).
- **cuBLAS differs, measured.** A batched 16x16 product with K = n through `cublasSgemmStridedBatched` and
  `cublasGemmStridedBatchedEx` (f16 and bf16 inputs with f32 compute, tf32 compute), the card's real cuBLAS (CUDA 13.0) against the
  simulator's emulation:

| cuBLAS call, n = 64 | Card (RTX 3060, CUDA 13.0 cuBLAS) | Simulator (rtx3060) |
|---|---|---|
| sgemm | two chains of 32 added together (a split along K in two) | one chain of 64 |
| GemmEx f16 or bf16 input, f32 compute | the tensor-core chain: 8 a step, as the wmma kernel | one chain of 64, one term a step |
| GemmEx f32 data, tf32 compute | two chains of 32, as sgemm (not the tensor core, at this shape) | one chain of 64 |

  At other K (measured, card only): K = 32: sgemm and the tf32 compute are two chains of 16, f16 the chain of 8 (so the split in two
  halves is the card's at K = 32 and 64); K = 96 and K = 128: sgemm is not a tree (the probe's counts are inconsistent, as a split
  with an order that depends on the run or an atomic join would give); f16 at K = 96 and 128 is a chain of 8-product steps for the
  first 64 and later steps each computed alone and added one by one. This is the library choosing a kernel by shape, which the
  simulator's emulation does not do, so it is **not changed**: no single measured rule covers it. The simulator's f16 and bf16 GemmEx
  gives the numbers of a sequential f32 sum, which no card's tensor core does.

## Not done

- **FP8 (E4M3, E5M2) vectors.** Done since 2026-11-18 for `mma.sync` on both engines, 70,000 vectors: [fp8-vectors.md](fp8-vectors.md). Not done: `wgmma`
  with FP8 (a rule fits the H100 and H200 files, the simulator's `wgmma` keeps the sequential sum) and the sparse forms.
- **Making Blackwell's published rule the default** for the profiles with no recorded data. Done on 2026-11-16 for b200 and gb200 only
  (their B200 vectors agree with it, 0 of 20,000, and the sequential sum does not, 8,020). sm_103 (b300), sm_120 (RTX PRO 6000, 5090),
  thor, gb10 and vr200 stay sequential by default: no recorded data covers them. A one-line change per profile
  (`numerics.tensor_sum`, with the chunk and grid keys) when data does. `tcgen05.mma` is not covered by either set of vectors.
- **V100.** Block of 4 products, no extra bit (published; 1,200 of 5,000 recorded f32-accumulator vectors differ from the Ampere
  rule on the card). No profile; the loader accepts two rules only (`include/vgpu/profile.hpp`), CUDA 13 cannot build sm_70.
- **cuBLAS shape-dependent order.** Above.
- An H100 or B200 card of ours has not run any of these programs: every Hopper and Blackwell line is the simulator's, checked against
  the published vectors and trees.

## Mutation evidence

`mutants.txt` here: one constant of the two rules in `include/vgpu/profile.hpp` changed at a time, the simulator rebuilt on server3,
`run_tcpapers.sh` run: each must fail.
