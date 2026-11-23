# m8n8k4, the sparse forms, wgmma and tcgen05.mma on f16, bf16 and tf32 (issue #664)

Follow-up of [tcpapers.md](tcpapers.md), [fp8-vectors.md](fp8-vectors.md) and [../matrix-numerics.md](../matrix-numerics.md), 2026-11-23. Words as there: **measured** (a card of ours printed it),
**inferred** (recorded vectors of others, or a published rule, or a measured device of the same compute capability), **unknown**. Nothing here is a timing model.
Profile keys (schema in [../../nvidia/docs/profiles.md](../../nvidia/docs/profiles.md)): `numerics.tensor_sum_m8n8k4`, `tensor_sum_sparse`, `tensor_sum_wgmma`, `tensor_sum_tcgen05`;
the arithmetic is [tc_forms.hpp](../../include/vgpu/exec/tc_forms.hpp); `VGPU_TC_EXACT=0` gives the earlier sequential results back for all four.

| Form | Profiles | Sum | Class | Evidence |
|---|---|---|---|---|
| `mma.m8n8k4` f16, f16 or f32 accumulator, PTX engine | t4 (7.5) | `volta`: one step of 4 products and C, terms cut at 2^(top - 24) | **measured** | Tesla T4 (g4dn.xlarge, driver 595.112.07, nvcc 13.0.88): the 5,000 + 5,000 vectors recorded on a V100, 0 mismatches each; 800 inputs of ours each, the T4's results reproduced |
| | rtx2080ti | the same | inferred | same compute capability |
| `mma.m8n8k4`, f32 accumulator | the 19 profiles of 8.0, 8.6, 8.9 | `chain_c_last`: `D = (((p0 + p1) + p2) + p3) + C`, each add rounded to nearest even in f32 | **measured** on a10g, inferred on the other 18 | A10G (g5.xlarge, same driver, sm_86 build): the V100 vectors 5,000 of 5,000, our 800 inputs 800 of 800 |
| `mma.m8n8k4`, f16 accumulator, 8.x | - | not modelled: sequential f32 sum | unknown | the A10G's results are close to c + p0 + p2 + p1 + p3 in f32 rounded to f16 (12 of 6,000 differ) and not equal to it; nothing fits all |
| `mma.sp`, `HMMA.SP` f16, bf16, tf32 | the 19 profiles of 8.0, 8.6, 8.9 | `ampere` over the stored products: chunks of 8 (tf32 4), terms cut at 2^(top - 25) | **measured** on a10g, inferred on the other 18 | A10G: eight modes, 800 inputs each, every finite result (2:4 pairs or 1:2 positions chosen by uniform metadata) |
| `wgmma` f16, bf16, tf32, dense | the six sm_90 profiles | `hopper` (chunks of 16, tf32 8, grid 26), C in the chunk | inferred | MMA-Sim (arXiv 2511.10909): same numbers as the device's `mma.sync`, which is measured on an H100 PCIe; 96 recorded H100 `mma.sync` vectors through a real `wgmma` in `test_tc_forms`; no `wgmma` of any card |
| `tcgen05.mma` `.kind::f16`, `.kind::tf32`, dense | b200, gb200 | `hopper` | inferred | MMA-Sim says the same; no recorded `tcgen05` data; `test_tcgen05_tiles` holds both routes to each other and to the sequential sum's difference |

## How the T4 and A10G results were taken

Our program [`tools/tcpapers/m8n8k4_vectors.cu`](../../tools/tcpapers/m8n8k4_vectors.cu) (a vector in row 0 of A and column 0 of B of one `mma.sync.aligned.m8n8k4`, as
`khattak_vectors` does for `wmma`) and [`sparse_vectors.cu`](../../tools/tcpapers/sparse_vectors.cu) (the same for `mma.sp`: the stored values of A's row 0 and a full column of B; one metadata
word for every lane, so no lane mapping of the metadata enters: every group of four columns stores the pair of positions the pattern names). Inputs
from [`m8n8k4_gen.py`](../../tools/tcpapers/m8n8k4_gen.py) and [`sparse_gen.py`](../../tools/tcpapers/sparse_gen.py) (splitmix64; random finite bit patterns, an exponent window, tiny and subnormal
operands, cancelling halves, one large term with small ones, and an accumulator that is zero, minus zero, in the window, tiny or subnormal). The cards' results are
in `nvidia/tests/data/tcpapers/` (`m8n8k4.t4.fp32.txt`, `.fp16.txt`, `m8n8k4.a10g.fp32.txt`, `m8n8k4.a10g.fp16.txt` (not modelled), `sparse/a10g.<mode>.txt`: 300 vectors a mode of the 800 run) and
`nvidia/tests/e2e/run_tcpapers.sh` holds the simulator to them on both engines (`VGPU_TC_EXACT=0` must not reproduce them); `V100.fp16.fp32.txt` and `.fp16.txt` are 24 of the V100 vectors
(Khattak and Mikaitis, BSD-2, NOTICE). `test_tc_forms` does the same through hand-written PTX.

What the cards showed:

- **A Turing m8n8k4 is a Volta one.** `HMMA.884` on the T4 reproduces all 10,000 vectors a V100 recorded: 4 products and C in one step, one bit less than an Ampere `mma.sync` keeps
  (Valpey et al., arXiv 2502.15999, and MMA-Sim say the same of Turing; this is the first run of ours). The earlier note that "Turing in the paper and here are not the same claim"
  (architecture-papers.md) is settled: the same chip, the same rule, for `m8n8k4` and `m16n8k8` apart (grid 24 and chunks of 4, against grid 25 and chunks of 8).
- **On sm_80 and later `m8n8k4` is not a tensor-core instruction at all.** ptxas emulates it (the A10G's result is bit-exact `(((p0+p1)+p2)+p3)+C` in f32, subnormals and all, 5,800 of 5,800),
  so an Ampere or Ada card gives neither a Volta step's bits nor the sequential `C + p0 + ...` the simulator gave before: 2,153 of the V100's 5,000 f32 vectors differ on the A10G,
  and the earlier simulator result differs from the card's. This is the cubin's code (nvcc 13.0.88's ptxas), not the hardware's: the SASS engine runs such a cubin as built and agreed
  before; the PTX engine now does the same sum. The f16-accumulator form is emulated differently and is not modelled.
- **The sparse forms are the dense rule over the stored products.** The Ampere rule (grid 25) with a chunk of 8 stored products (16 dense columns of a 2:4 row; tf32: 4 stored, 8 dense) gives every
  finite result of every mode, f16, bf16 and tf32, f32 and f16 accumulators. A place that A does not store takes no part: an infinity or a NaN there in B does not reach the sum.
  The simulator's earlier sparse sum was the sequential one over the expanded matrix (and made a NaN of such a place).
- Not decided by any run: Hopper and Blackwell `mma.sp` (the rule at 16 stored products is the guess; the profiles keep the sequential sum), `mma.sp` int and FP8 forms
  (integers are exact; FP8 sparse keeps the sequential sum), `tcgen05.mma.sp`, the FP4, FP6 and block-scaled kinds of `tcgen05.mma`, `wgmma` with an f16 or bf16 accumulator beyond
  the dense f16 form, non-zero-accumulator `wgmma` on a card.

## What needs a card we have not run

| Item | Card | What to run | Rough cost |
|---|---|---|---|
| `wgmma` f16, bf16, tf32 (zero and non-zero C, f32 and f16 accumulators) | H100 (p5.4xlarge or a Lambda H100) | [`tools/tcpapers/khattak_wgmma_vectors.cu`](../../tools/tcpapers/khattak_wgmma_vectors.cu) on the H100 files: 5,000 vectors a file, and `m8n8k4_vectors` for whether Hopper accepts the form | 0.5 h of a single H100: about 3 USD on Lambda, 7 USD on AWS |
| `tcgen05.mma` `.kind::f16`, `.kind::tf32`, FP8 and block-scaled kinds | B200 (p6-b200.48xlarge, 8 GPUs) | the same program built for sm_100a with a `tcgen05` kernel; not written (needs a Tensor Memory harness) | ~1 h of an 8-GPU node, about 100 USD on demand: not started |
| `mma.sp` on Hopper and Blackwell | H100 | `sparse_vectors` built for sm_90 | with the first row |
| `mma.m8n8k4` f16 accumulator on sm_80 and later | any Ampere card (A10G again) | more vectors, ptxas's emulation read with `cuobjdump -sass` (the toolkit's `cuobjdump` was not on the DLAMI's path) | 1 h of a g5.xlarge, 1 USD |

## AWS ledger (2026-11-23)

| Instance | Type | Launched (UTC) | Terminated | Cost |
|---|---|---|---|---|
| i-0b1b0d0d635615f92 | g4dn.xlarge (T4), us-east-1, on demand | 20:59 | 21:10 (TerminateInstances; no volume left: `DescribeVolumes` on the project tag is empty) | about 0.1 USD |
| i-01954203c23d44787 | g5.xlarge (A10G), us-east-1, on demand | 20:59 | 21:10 (same) | about 0.2 USD |

100 GB gp3 root volumes with DeleteOnTermination, no other storage, a 200-minute dead-man shutdown in user data, tags `Project=pantheonsim-matrix`.
No V100, H100, H200 or B200 was launched.
