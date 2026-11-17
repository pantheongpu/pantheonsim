# Turbo leftovers: #660, #661 and the rest of #656 (2026-11-16)

Written on a machine with no card and nothing was run to make it: every number below is copied from a file in this repository (named), and the
scripts in this directory are **not run**. They are for the A10G rig. Claims are tagged documented (a file says it), measured (a card or
server3 run in a named file), inferred, or unknown.

## #660 very short launches (JACOBI1D, sc_gpu, LargeKernelParameter): needs a card

What decides them today (documented, `include/vgpu/host_gpu.hpp`: `Costs`, `should_offload`, `start_share_saved`): a launch goes only if the
simulator's estimate is at least `min_sim_s` (200 us) and more than `margin` (2) times the card path; the card path is
`launch_s` (7 us) + the kernel's card time + the moves, and the card time is the prior `kernel_s_default` (100 us) until the card has run that
kernel once. A kernel is never sent when it takes the simulator under 200 us, so the card never runs it and the prior is never replaced.

| program | launches | what the simulator takes | source |
|---|---:|---|---|
| polybench/JACOBI1D | 20,000, none sent | 2.26 s in all, 113 us a launch | measured, `data/sp69/card.md` section 2 |
| rodinia/sc_gpu | 1,432, none sent | 0.6 ms a launch; sent 1029 of 1432 under the account alone (1.24x) and 0 under the 0.5 ms share | measured, `data/sp69/card.md` section 1 |
| samples/LargeKernelParameter | 2,002, none sent | 0.2 to 0.3 ms a launch against a card path of 0.107 ms; when sent (round sp66) 0.22 s was lost on 134 to 163 launches | measured, header comment of `start_share_launches` |

The decision for these is a pure function of constants, so a log on server3 against the stand-in driver would print "the estimate does not
beat the card path" and teach nothing: it was not run. Nothing in the simulator is wrong. The lever is the two priors, and only a card says what
they should be: the card's real cost of one tiny launch with a sync (unknown; the 7 us is an estimate of the launch call alone), whether the
arrays stay on the card between launches (the streak rule of sp65 is meant for that; documented, `12-host-gpu-turbo.md` section 9), and
the kernel time of a stencil of 32 x 128 threads (inferred: single-digit microseconds; unknown). `card-short-launch.sh` measures all of it and
says in its header what each outcome means. If the sync latency is over about 50 us the issue closes as "no gain" (the issue allows it).

## #661 refused kernels: BMMA is the one part a card decides; the rest is feature work

| item | what is needed | tag |
|---|---|---|
| `BMMA` (1-bit tensor MMA) | it was in the turbo `never` class; it is in the `tensor` class since 2026-11-16 (`12-host-gpu-turbo.md` section 2). The samples have no BMMA program, so `nvidia/tests/e2e/bmma_gemm.cu` is our own (wmma `precision::b1`, 8x8x128, XOR+popc and AND+popc, 3 kernels, 6 shapes, 36 cases, host popcount self-check). Measured 2026-11-16: an RTX 3060 (CUDA 13.0, sm_86 SASS has `BMMA.88128.XOR.POPC` and `.AND.POPC`) matches the host reference in 36 of 36 cases, and the simulator prints the same bytes as the card (checksum `0xae570291798d39ce`) from sm_86 SASS, the PTX engine and a compute_80 build; pinned as `e2e_bmma_gemm`. Decided by the card: on an A10G (g5.xlarge, driver 595.112.07, CUDA 13.2 nvcc, 2026-11-16) turbo `VGPU_HOST_GPU_VERIFY=all` `ATOL=0 RTOL=0` gave other_identical=36, other_different=0; host_gpu_turbo launches_card=36, words_compared=775680, words_different=0, words_outside=0; the output is byte-identical to the simulator's; exact mode verify=all also other_identical=36 (a probe: BMMA is not in the exact set). Files in `card-2026-11-16-bmma/`. Admitted to the tensor class, with a unit test; the exact set is unchanged. Admitted | simulator equals card: measured on an RTX 3060 and an A10G; other cards with BMMA: inferred (integer arithmetic) |
| `QMMA`, sm_90/sm_100 tensor instructions | not on the A10G (sm_86): no card here can run them. H100/B200 needed | unknown |
| texture and surface instructions (dct8x8, simpleCubemapTexture, simpleLayeredTexture, simpleSurfaceWrite) | the card would hold texture objects the simulator keeps in its own tables: mirroring them is a design, not a fix. About 1 s of the issue's 4 s | not started |
| argument into mapped host memory (zerocopy, simpleIPC, simpleZeroCopy: 3.4 s) | the card must see the host's mapping at the same address, the simulator's pointer being its own | not started |
| cooperative launch (simpleAWBarrier) | needs `cuLaunchCooperativeKernel` with the simulator's grid-wide barrier semantics | not started |

None of these is a pure simulator-side bug, so nothing was changed.

## #656 the rest: a loop-weighted static count for `initRNG` (design; not built)

The design in the issue and in `12-host-gpu-turbo.md` sections 13 and 14 is not clear enough to build without a census, so only the numbers
are set down (measured on server3 against the stand-in driver, `data/sp76/launches.md` and `data/sp76/start-model.md`):

| program | kernel | counted | taken | floor | age | a sample says | needs | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MC_SingleAsianOptionP | `initRNG` (782 x 128) | 9.5 ms | 1.35 s | 72 ms | 5 ms | 0.79 s | 0.71 s | sent here by 11 %, sent on the rig (inferred: 2.5 times slower) |
| MC_EstimatePiInlineP | `initRNG` | 9.5 ms | 1.46 s | 30 ms | 4 ms | 0.56 s | 0.64 s | kept here (one block timed, 2.6 times under), sent on the rig (inferred) |
| warpAggregatedAtomicsCG, lavaMD | | | | | | | | kept, and not the floor's doing (moves of 270 to 305 MiB; one corner block) |

So the rule would help two programs, about 3 s each on the rig (inferred from the 3.4 s and 3.1 s of the 2026-11-13 summary run). What it
needs, per `launches.md`: (1) the flow scan following a loop left through `CALL.REL.NOINC` to the kernel's `EXIT` (today any call makes every
loop of the kernel "not followed"); (2) loops inside loops; (3) the weighted count asked of the floor for a sample by whole blocks (today only
inside the blocks); (4) a census, simulator-only and doable on server3, of every kernel of the corpus whose weighted count rises over the floor,
since each would then be sampled about 5 ms into its process at a cost of 3 to 12 ms and its page faults. Outputs would be bit-identical (the
rule only moves a launch between two paths whose bits are equal in exact mode and defined in turbo); decisions would change only for kernels
the census names. Not built because step 1 changes the flow scan that every sampled kernel depends on, and the census (4) comes first: if it
names more than a handful of kernels the rule is a net loss of sample time. First step when someone builds it: write the census as a mode of
`data/sp75/census.py` that prints each kernel's weighted count beside its plain count, with no change to any decision.

## Files

| file | what |
|---|---|
| `card-short-launch.sh` | #660: native and mode times of the three programs, an empty-kernel micro benchmark (launch, launch + sync, copies), the kernel time from nsys. Not run |
| `card-bmma-verify.sh` | #661: `bmma_gemm` (our own program) under exact and turbo `VGPU_HOST_GPU_VERIFY=all`. Run on an A10G 2026-11-16 (results in `card-2026-11-16-bmma/`) |
| `../sp76/rerun-interval-reduction.sh` | the card rerun of samples/interval and samples/reduction (15 runs, three forms, one with the old environment size). Not run |
