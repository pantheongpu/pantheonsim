# Bugs in other people's programs that the simulator found

Status: written 2026-11-18 from a search of this tree and, for the upstream column, a quick `gh search issues` (one query per bug,
GitHub only, no web search, so "none known" means "not found on GitHub in that query", not "does not exist"). Nothing below has
been posted. Each draft needs its reproducer re-run on current versions before anyone sends it.

| # | Program | Bug | Version or commit | Evidence in this tree | Upstream report |
| --- | --- | --- | --- | --- | --- |
| 1 | llama.cpp / ggml `flash_attn_ext_f16` (mma) | write-after-read race on shared `tile_Q` | **not recorded** (an external `test-backend-ops` checkout of about 2026-09-04) | [history/TODO-done.md](history/TODO-done.md) "A race the simulator found in llama.cpp"; the detector `VGPU_RACE`, tests in `tests/unit/test_exec3.cpp` (`a_shared_race_between_warps_is_reported` and four more). **No reproducer in the tree.** | none known (searched `flash_attn_ext_f16 tile_Q`, ggml-org/llama.cpp: nothing); whether it is fixed since is unknown |
| 2 | Rodinia `srad_v2` (CUDA) | out-of-bounds global reads before the boundary fix-up | gpu-rodinia `9c10d3e` is the corpus pin (`ci/external/rodinia.txt`); the sim run was on the 3060 | [perf-design/09-host-gpu-spike.md](perf-design/09-host-gpu-spike.md) ("the simulator stops at the out-of-bounds read the card hides"), [12-host-gpu-turbo.md](perf-design/12-host-gpu-turbo.md) l.787, 792 (the card faulted on it, measured) | **exists**: JuliaParallel/rodinia#12 "[CUDA][srad_v2] Out-of-bounds neighbor reads occur before boundary correction" (open, 2026-09-04, filed by someone else). Not Rodinia's own repository; no report found there |
| 3 | Rodinia `particlefilter_float` | run-to-run nondeterminism; docs call it "the program's own race" | `9c10d3e` | `perf-design/data/round12/e2_notes.json` ("native nondeterministic (time-seeded / uninitialised)"), `perf-design/data/sp38/memo-corpus-correctness-*.md`, `12-host-gpu-turbo.md` l.685, 719 | none known. **The cause is not established**: the notes say both "time-seeded" and "race", and the first is not a bug. |
| 4 | HeCBench `btree` | counters differ between two runs of the same build, "a race in the program" | HeCBench checkout at `~/.cache/HeCBench`; commit not recorded | `perf-design/rounds/02-sass-tier-rounds-8-9.md` l.590-592 (one line, no diagnosis) | none known. **No analysis of the race exists**; it is a one-line observation. (Rodinia also has a `b+tree`; the doc means HeCBench's.) |
| 5 | NVIDIA `ptxas` (CUDA 13.0 on sm_86) | miscompiles `szext` with a register width: the card extends from 0 bits | ptxas of CUDA 13.0 (the 3060 corpus) | [history/TODO-done.md](history/TODO-done.md) l.283 (one sentence). `e2e_ptx_forms` hashes the variants against the card; the unit test `mul24_szext_and_fns` in `tests/unit/test_exec3.cpp` checks the ISA meaning only. **The miscompiling PTX is not saved** | none known (NVIDIA has no public tracker for ptxas; the developer forum is the channel; not searched) |

Gaps to close before a report can be sent: the reproducer and exact version for 1, 3, 4 and 5; the diagnosis for 3 and 4. For 3 and 4
nothing should be sent until someone shows the race (for example with `VGPU_RACE=1` on the simulator, or compute-sanitizer
`racecheck` on a card); a report that says "it varies" is not useful to maintainers.

## Drafts (not posted)

### 1. ggml: missing barrier in `flash_attn_ext_f16` when stream-k gives a block more than one tile

> **flash_attn_ext_f16 (mma): `tile_Q` write-after-read race when a block processes more than one tile and `np == 1`**
>
> Version: <llama.cpp commit, to fill in> built for sm_86, CUDA 13.0. Found by running `test-backend-ops -o FLASH_ATTN_EXT` under a
> simulator that schedules warps round-robin; the same shape is a latent race on hardware.
>
> `flash_attn_ext_f16_process_tile` writes `tile_Q` (shared memory) at its start, and its first `__syncthreads()` comes after those
> writes. The matching barrier at the end of the function is conditional (`if (np > 1) __syncthreads();`), and the caller loops over
> tiles without a barrier between calls. With `np == 1` and a block that handles more than one tile (stream-k splits work unevenly),
> the next tile's writes to `tile_Q` are not ordered after the previous tile's reads.
>
> Observed: `FLASH_ATTN_EXT` with `hsk=192, hsv=128`, batch > 1, fails 14 of about 2950 cases (error around 0.1 against 5e-4) with a
> grid of 144; grids of 64, 128, 256 (one tile per block) pass; stream-k disabled passes 36/36; making the `__syncthreads()`
> unconditional passes 36/36.
>
> Suggested fix: make the trailing `__syncthreads()` unconditional (or put one at the loop in the caller).
> To check on hardware: `compute-sanitizer --tool racecheck` on the failing case.

### 2. Rodinia `srad_v2`: do not report a new issue; add a comment to JuliaParallel/rodinia#12

> We ran `srad_v2` on a GPU simulator that checks every global access against its allocation, and it stops at the same reads this
> issue describes (`srad_cuda_1` and `srad_cuda_2` load `J_cuda[index_n]`, `J_cuda[index_s]`, `C_cuda[index_s]` before the boundary
> blocks overwrite the shared value). In the simulator's host-GPU mode the launch also faulted on the card (card model not recorded: to fill in).
> Moving the loads behind the `by == 0` / `by == gridDim.y - 1` tests fixes both. Confirming the report.

(Whether the original repository, `cuda/srad_v2` in Virginia's Rodinia, has an issue tracker with this is not checked.)

### 3. Rodinia `particlefilter_float`: not ready

> Draft held. What is known: two runs on the same card give different estimates, and on the simulator without its cache too. What
> must be found first: whether the difference comes from `time()` seeding (by design) or from an unordered write between blocks or
> an uninitialised read. If it is a race, the report needs the kernel, the two buffers and a `racecheck` log; with the seed fixed
> (`srand` of a constant) the question is whether runs still differ.

### 4. HeCBench `btree`: not ready

> Draft held. What is known: `VGPU_COUNTERS` differ between two runs of the same simulator build, so two blocks interact through
> memory in an order-dependent way. What must be found first: which kernel and which words (`VGPU_RACE=1` names the shared-memory
> case; a global-memory case needs `racecheck` on a card).

### 5. NVIDIA developer forum: `szext` with a register width on sm_86

> **ptxas on sm_86 treats the width operand of `szext` as 0 when it is a register**
>
> CUDA 13.0 (ptxas <version>), sm_86. PTX:
> `szext.clamp.s32 %r6, %r4, %r5;` with `%r4 = 0xFF`, `%r5 = 8` (a register, not an immediate). The PTX ISA says this sign-extends
> from bit 7 and gives -1. The generated SASS for the register form gives <result, to fill in> as if the width were 0. The
> immediate-width form is correct. Minimal program and `cuobjdump -sass` output attached: <to be produced: the PTX that
> miscompiled was not kept>.

## Where this should live

The five rows are claims in `history/TODO-done.md` and the perf notes. Once a reproducer exists for a row, put it under
`nvidia/tests/` or `ci/external/` and replace the "evidence" cell with its path.
