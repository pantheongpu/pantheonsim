# More papers on how architectures differ: what was checked (issue #708)

Follow-up to [tcpapers.md](tcpapers.md). Everything here was run on the local RTX 3060 (sm_86, driver 610.88, nvcc 13.0.88, GPU 0 alone) on
2026-11-19 and on the simulator (both engines). Words as there: **measured** (a card of ours printed it), **published** (a paper or data
set says it), **inferred**, **unknown**.

## Valpey et al., arXiv 2502.15999 (Volta, Turing, Ampere; Titan V, RTX 2080 Ti, RTX A6000, CUDA 11.8)

What the paper says of an Ampere tensor core, against our rule ([tc_ampere.hpp](../../include/vgpu/exec/tc_ampere.hpp)):

| Claim of the paper | Our rule | Result |
|---|---|---|
| Products are exact in f32 (table 1) | exact products | agrees (`tc_features product_exact`, [tcpapers.md](tcpapers.md)) |
| All terms, C included, are aligned to the largest and added in one group; order does not matter (property 3) | a chunk is the 8 products and C, summed in one step | agrees: eight products of very different size give the same bits forwards and in reverse (`valpey_probe`, `eight_forward`, `eight_reverse`, both `3f801c00`) |
| Ampere keeps one bit more than the 24-bit mantissa below the largest term; the rest is discarded | terms truncated toward zero at `2^(top-25)` | agrees (`align_max_j` 24 in `tc_features`) |
| The accumulation truncates; it is **not** round-to-zero, the two differ in an effective subtraction with a large exponent gap (section 4.2, table 3: a = 2, c = -2^-40 gives 2, not 2 - 2^-23) | truncation of the aligned terms, then of the sum | agrees: `p2_c-2^-40` is `40000000` on the card and on both engines; the ones around it (below) say where the cut is |
| Intermediate sums are not normalised (property 4) | exact sum of the aligned terms | agrees: `1 + 2^-24 + 2^-24` is `3f800001` (1 + 2^-23), in either order |
| Final f16 result is rounded to nearest by the tensor core | f16 accumulator rounds to nearest even | agrees (`tc_features` `round_pos`, `round_neg` for f16h) |
| 9-term (Ampere) accumulator needs 4 carry-out bits (solver timed out; not proved) | the aligned terms are added exactly: no carry can be lost | not testable as stated; `carry_sum` in `tc_features` (sixteen equal products) agrees |
| Volta and Turing: 4 products and C (5 terms), **all** shifted-out bits discarded; identical on both | the `Ampere` rule on Turing too: 8 products, one kept bit (measured on a Tesla T4 with `mma.sync m16n8k8`, [matrix-numerics.md](../matrix-numerics.md)) | **not comparable here**, see below |

The truncation cases (`valpey_probe`, f16 inputs, f32 accumulator; one product `p` and an accumulator `c`; bits of the result; the card, and the
simulator's `sass` and `ptx` engines, which agree: `nvidia/tests/data/tcpapers/valpey.rtx3060.txt`, in `e2e_tcpapers`):

| p | c | result | reading |
|---|---|---|---|
| 2 | -2^-40, -2^-26, -2^-25, -2^-24, -1.5 * 2^-24 | `40000000` (2) | c lies below the cut at `2^-23`: dropped, even where round-to-zero would give `3fffffff` |
| 2 | -2^-23 | `3fffffff` | on the cut: kept |
| 2, -2 | +2^-25, +2^-24 | `40000000`, `c0000000` | dropped |
| 1 | -2^-26, -2^-25 | `3f800000` (1) | below the cut at `2^-24`: dropped |
| 1 | -2^-24, -3 * 2^-25 | `3f7fffff` | on and above the cut (-3 * 2^-25 is cut to -2^-24) |
| 1.5 | -2^-25, -2^-24 | `3fc00000`, `3fbfffff` | the cut follows the largest term's exponent |

**What could not be settled.** The paper's Volta and Turing model describes `HMMA.884` (`mma.m8n8k4`, 4 products), which a Volta runs. Our Turing
measurement is of `mma.m16n8k8` (`HMMA.1688`, 8 products). The two forms can differ on one chip, so "Turing" in the paper and "Turing" here
are not the same claim. The simulator's `m8n8k4` form keeps the sequential f32 sum (a known gap, #664), and the paper's Algorithm 1 (5 terms, shifted bits
discarded) plus the 5,000 recorded V100 vectors of [MATLAB-tensor-core](https://github.com/north-numerical-computing/MATLAB-tensor-core)
(1,200 mismatches of 5,000 for the f32 accumulator on the RTX 3060) are what a rule for it would be written and checked against. No V100 profile exists
(CUDA 13 cannot build sm_70). Not done.

The paper's table 5 (an error-correction case) is not reproduced: its inputs print garbled in the HTML version we read.

## Monotonicity of the block adder (Mikaitis, arXiv 2304.01407), added 2026-11-27

The paper's result: an adder that aligns every term to the largest exponent and drops the bits that fall off is not monotonic. Raise one input past a
power of two, the others are aligned one place higher and lose bits, and the result can fall. The rule of [tc_ampere.hpp](../../include/vgpu/exec/tc_ampere.hpp)
is such an adder, so the effect must exist for each (block width, grid) the profiles use. [`test_tc_monotonicity`](../../tests/unit/test_tc_monotonicity.cpp)
(unit test, no card) holds it to that, for Volta 4/24, Ampere 8/25 and tf32 4/25, Hopper 16/26 and tf32 8/26:

- **a decrease exists**: an accumulator of 2 - 2^-23 against 2, with the other terms of the chunk just below the coarser cut: the larger accumulator gives the smaller
  result (a scan of the size of the small terms finds it for every rule; it would fail for a rounding adder or a wider grid);
- **monotone at a fixed alignment**: with the largest term held, raising any other product never lowers the result (20,000 random chunks a rule).

Class: **published** (the paper's property) checked on the **measured** and **inferred** rules; it says nothing new about any card. Whether a card shows the decrease is
the same question as whether its rule is right, which the card runs already answer for the measured devices.

## The MATLAB models that were not compared before

`north-numerical-computing/MATLAB-tensor-core` holds, at our pinned commit 79143bd (and at 6fd05208, 2026-11-18: only the README changed, and
`models/` and `model_validation/` are byte-identical), models we had not read:

| Model | What it is | Against our rules |
|---|---|---|
| `RTXPRO6000TC.m` | calls `B200TC.m` | `B200TC`: 16 products and C, 2 extra alignment bits, truncating, f32 result `rz`; f16 result `rne`. That is our `hopper` rule, which is the default for sm_120 (measured on an RTX PRO 6000 Blackwell Server Edition, 2026-11-17). Agrees for f16, bf16, tf32. Its FP8 branch (32 products, interleaved, `neab` 2) is **not** what we measured for sm_120 FP8 ("exact sum cut toward zero", 18 of 27 forms): the model is a hypothesis for the other nine forms, **not tested** (the card's raw results are not in the repository) |
| `A30TC.m` | calls `A100TC.m` | `A100TC`: 8 products, 1 extra bit, truncating (tf32: 4), `rne` for an f16 result: our `Ampere` rule; a30 has it |
| `B200TCRN.m` | `B200TC` with the final f32 rounding `rne` (not `rz`), f16 minimum exponent -15 (not -21) | no recorded vectors in the repository use it; the recorded B200 vectors all agree with `rz` (0 mismatches of 20,000). Nothing to change. It may describe a different instruction (`tcgen05.mma`), which has no recorded data: a hypothesis only |
| `H100TC.m` FP8 branch | 32 products, `neab` -10, `rne` | looks like the `wgmma` FP8 rule found in [fp8-vectors.md](fp8-vectors.md) (one chunk of 32, 13 bits below the largest): same family; not checked number by number |

The model comments also say `A100TC` takes 2 products for bf16 "without `_1k`"; our rule takes 8 for bf16 on the forms we probed (`mma.m16n8k8`, wmma)
and reproduces 5,000 recorded A100 bf16 vectors, so that note concerns a form we do not have. Not investigated further.

## `atomicAdd` with subnormal operands (forum reports, not a paper)

Probe: [atomic_subnormal.cu](../../tools/tcpapers/atomic_subnormal.cu), 8 float and 6 double cases, global and shared memory, result used (`ATOM`) and
unused (`RED`), built with and without `-ftz=true`. RTX 3060, 2026-11-19; the output is identical with and without `-ftz`:

| | global memory | shared memory |
|---|---|---|
| float | flushes subnormal inputs and results to a signed zero (`ATOMG.E.ADD.F32.FTZ.RN`, `RED.E.ADD.F32.FTZ.RN`): `0x1 + 0x0` is 0, `0x7fffff + 0x7fffff` is 0, `0x00c00000 - 0x00800000` is 0 | **keeps them**: `0x1 + 0x0` is `0x1`, `0x7fffff + 0x7fffff` is `0xfffffe`, `0x00c00000 - 0x00800000` is `0x400000` (ptxas makes `LDS`, `FADD`, `ATOMS.CAST.SPIN`, a compare-and-swap loop; the `FADD` has no `.FTZ`, with or without `-ftz=true`) |
| double | keeps them (`0x1 + 0x1` is `0x2`) | keeps them |

So the forum report holds on this card: global float atomics flush whatever `-ftz` says, shared ones do not. The simulator had flushed in both. Fixed in
the PTX interpreter (`atom` and `red` with `.shared` no longer flush; `src/exec/interpreter.cpp`). The SASS engine already ran the card's real
compare-and-swap loop for shared memory, and `ATOMG`'s `.FTZ` for global, so it matched. Not covered: a generic-address float atomic that points into
shared memory (the card's `ATOM.E.ADD.F32.FTZ` on a shared window was not probed; the interpreter keeps flushing there), and `atomicAdd` on f16 and
bf16 pairs. The test is `e2e_atomic_subnormal` (the card's 56 lines, both engines, both `-ftz` settings).

## Feature checklists: Hopper (Luo et al.) and Blackwell (Jarmusch et al.), added 2026-11-23

Items 4 and 5 of #708. A program-visible feature is one whose result, error or limit a program can read. No timing is listed (the simulator has none). "Test" names what holds
the simulator to something; **card** says whether any of ours has run the feature (none of Hopper or datacenter Blackwell has, except the numbers marked measured below).

### Hopper (arXiv 2501.12084; Luo et al. also give memory and speed results, not listed)

| Feature | What a program can see | Test in the repository | Held to a card? | Open |
|---|---|---|---|---|
| Distributed shared memory (clusters, `mapa`, `ld.shared::cluster`, cluster barriers) | values through a peer block's shared memory, cluster size and occupancy limits, the errors for a bad rank | `test_dsmem`, `e2e_dsmem_cluster`, `e2e_sass_archs` (`cooperative_cluster`), `e2e_cluster_occupancy` (occupancy: an RTX 3060's `cluster_occupancy.rtx3060.expected` for the rule, not Hopper's limits) | no | the maximum portable and non-portable cluster size, `cudaOccupancyMaxActiveClusters` on an H100: a probe is `cluster_occupancy.cu`; needs an H100 |
| Tensor Memory Accelerator (`cp.async.bulk.tensor`, tensor maps, im2col, reduce, multicast) | tile contents for every layout, swizzle and out-of-bounds fill, the error for a bad map | `test_hopper`, `e2e_tma_gemm_cute`, `e2e_tma_im2col`, `e2e_tma_reduce_cute`, CUTLASS Hopper | no | whether a card faults on the same bad maps; the fill values for out-of-bounds NaN |
| DPX instructions (`__vimax3_s32_relu`, `__viaddmax_s16x2`, `__vibmin_u32`, ...) | integer results | `e2e_dpx_intrinsics`: NVIDIA's own host implementations (`crt/device_functions.hpp`) as the oracle, SASS (`VIMNMX`, `VIADDMNMX`, `VIMNMX3`) and PTX, sm_80 to sm_120 | the host implementation is NVIDIA's; no H100 output file | an H100 run of the same program would make the oracle a card |
| `wgmma` numerics | f16, bf16, tf32 (this change) and FP8 (#711) results | `test_wgmma_fp8`, `test_tc_forms`, `e2e_wgmma_vectors`, `e2e_wgmma_fp8_vectors` | no: **inferred** from MMA-Sim and from vectors recorded by others with `mma.sync` | [`khattak_wgmma_vectors.cu`](../../tools/tcpapers/khattak_wgmma_vectors.cu) on an H100, zero and non-zero C, f32 and f16 accumulators |
| `setmaxnreg`, `elect.sync`, `fence.proxy.async`, `mbarrier` transaction counts | register-count errors, ordering | `e2e_sass_archs` (`setmaxnreg`), `test_hopper` | no | |

### Blackwell (arXiv 2507.10789 on an RTX 5080, 2512.02189 on a B200)

| Feature | What a program can see | Test in the repository | Held to a card? | Open |
|---|---|---|---|---|
| `tcgen05.mma` (f16, tf32, f8f6f4, mxf8f6f4, mxf4, nvf4, i8; `.ws`; `.cta_group::2`) | results, the descriptors the ISA allows, the errors for the rest | `test_blackwell`, `test_tcgen05_fast`, `test_tcgen05_tiles`, `e2e_sass_archs` (`tcgen05_gemm`), CUTLASS sm_100 | no. Numerics: `.kind::f16` and `.kind::tf32` dense are the Hopper rule, **inferred** (this change); the other kinds keep the sequential sum | recorded `tcgen05` vectors, or a B200 run (B200TCRN.m of Khattak and Mikaitis is a round-to-nearest variant that could be this instruction) |
| Tensor Memory (`tcgen05.alloc`, `.ld`, `.st`, `.cp`, `.shift`) | the 512 columns, allocation failures, the lane windows of a warp | `test_blackwell`, `e2e_sass_archs` (`tmem_alloc_pair`) | no | |
| FP4, FP6, block-scaled MMA (`.kind::mxf4`, `mxf8f6f4`, sm_120 `mma.sync` with e2m1, e3m2, e2m3) | results | `test_sass_mma_tiles`, `e2e_tcpapers` (sm_120: 18 of 27 forms bit for bit on an RTX PRO 6000 Blackwell Server Edition) | sm_120 yes, partly (**measured**); sm_100 no | the nine sm_120 forms that differ; MMA-Sim's `ST_FDPA` and `GST_FDPA` for sm_100 are not implemented (no recorded vectors) |
| Sparse GEMM (`tcgen05.mma.sp`, `mma.sp` on sm_120) | results, metadata layouts | `test_tcgen05_tiles` (layouts), `test_mma_tiles` | layouts of `mma.sp` on sm_86 only; numerics of the sparse forms on an A10G (**measured**, this change) | Hopper and Blackwell `mma.sp` and `tcgen05.mma.sp` numerics |
| Decompression engine (`tcgen05.cp` with `.b4x16_p64`, `.b6x16_p32`) | the expanded bits | `test_blackwell` | no | |

Not listed because it is timing or energy: every latency, throughput, power and clock result of both papers.
