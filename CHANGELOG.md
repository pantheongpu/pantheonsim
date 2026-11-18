# Changelog

PantheonSim is versioned by the `project()` line in `CMakeLists.txt`; `vgpu --version` prints it.
Each release is tagged `vX.Y.Z` and described on the
[releases page](https://github.com/pantheongpu/pantheonsim/releases). It simulates GPUs functionally, so a
release says what runs and what it produces, never how fast a real GPU would run it.

## Unreleased

- **CI: the runs of merged, closed and superseded pull requests are cancelled.** After a PR merged, its sanitizer jobs kept shared runners for hours while the open PRs queued behind them. `ci/cancel-stale-runs.sh`, a second step of the cancel-conflicting-prs workflow (after every push to main and every 20 minutes), cancels every queued or running pull_request run whose head commit is not the head of an open PR. Push, schedule and dispatch runs are not touched.
- **Removed: host-GPU offload (`VGPU_HOST_GPU=exact` and `turbo`).** The public simulator no longer sends any launch to a real GPU: it is the plain simulator
  only, and every run gives the same bits on any machine. Gone with it: the setting `host_gpu` and `host_gpu_log` (`vgpu config` and the settings file),
  `VGPU_HOST_GPU` and the whole `VGPU_HOST_GPU_*` family (`_LOG`, `_VERIFY`, `_TIMEOUT`, `_WIDER`, `_DIV`, `_SQRT`, `_START_*`, `_SAMPLE*`, `_FIRST_BIG`, `_SHORT`,
  `_STREAK`, `_STATIC`, `_WRITE_THROUGH`, `_MAX_MIB`, `_TRACE`, `_LIBCUDA`, `_NO_FIRST_LAUNCH`, and the ones only turbo read: `_FIT`, `_PRESTART`, `_TURBO_OFF`,
  `_ATOMICS`, `_ATOL`, `_RTOL`, `_TOL_TYPE`, `_HISTORY`, `_COOP`, `_HOSTMAP`, `_DEVICES`, `_PTX_VARIABLES`), `VGPU_HOSTGPU_CAPTURE*`, the card gate in the memory manager,
  the exit-summary lines of the mode, the `tools/turbo-census` tool, the benchmark harness, probes and card-CI scripts for it, their tests, and the design notes and
  measurement data for it under `docs/perf-design`. A program that sets `VGPU_HOST_GPU` now runs in the simulator as if it had not. Released versions keep the code under
  their Apache-2.0 licence; nothing else in the simulator's results changed.
- **CI: a memory share left by a killed runner no longer blocks the host.** A runner that is restarted or killed mid-job never runs its "Give back the memory reservation" step, and the share held the slot for up to 4 hours while every other heavy job on the host waited for it and failed (the `ollama on amd/mi300x` failure on #701). A waiting `ci/jobs.sh --check` now asks GitHub whether the run behind each share has ended and drops the shares of runs that have; any doubt keeps the share. The check steps pass `GH_TOKEN`.
- **The compute capability 12.0 Blackwell profiles sum by the Hopper rule, measured on an RTX PRO 6000 Blackwell Server Edition** (issue #664). `rtx-pro-6000-server`
  (`measured`) and `rtx5090`, `rtx-pro-4500`, `rtx-pro-6000`, `rtx-pro-6000-max-q` (`inferred` from it) took the sequential f32 sum before: wmma, `mma.sync` and SASS
  `HMMA` in f16, bf16 and tf32 now give the card's bits (both probes print the H100's lines, 60 of 60 and 169 of 169; FTTN's motivating example 191.875 instead of 0;
  the six subnormal corners; 8,388,608 of 8,388,608 wmma results). The card ran on AWS (g7e.2xlarge, driver 595.112.07, nvcc 13.0.88 and 13.2.51, 2026-11-17).
  `VGPU_TC_EXACT=0` gives the earlier results back; the `tensor_sum_published` key is no longer used by any profile. A Tesla T4 also ran `float_half_forms`:
  every line of its sm_75 build is the RTX 3060's, so the SASS executor's answer for `half_form_30` (HMUL2.F32 on sm_75) is the one that is wrong; it is still left
  out of the sm_75 SASS comparison. docs/matrix-numerics.md.
- **OpenCL on the MI455X is documented as an upstream ROCm limit, and `amd_opencl` expects it** (issue #651). ROCm's `libamdocl64`, 7.1.0 and 7.2.4 alike, logs `Unsupported HSA device gfx1250` and drops the device although the simulator offers both agents; the other nine GPUs list both devices and pass. `amd/mi455x` is back in the test's cases and passes only when the runtime says exactly that (a missing device with any other reason still fails); if a ROCm release lists the device its checks run. The CI job unpacks ROCm 7.2.4's OpenCL runtime now (`fetch-rocm-opencl.sh DIR [VERSION]`).
- **AMD counters: cache lines of global and flat accesses** (issue #650, the second half). `VGPU_DISPATCH_STATS` also prints `vmem_lines_rd` and `vmem_lines_wr`, the distinct vector L1 lines the active lanes touch (64 bytes on CDNA, 128 on RDNA and gfx1250; the table is in `docs/cli.md`).
  It is modelled, not verified against a card, and no rocprofiler counter is mapped to it. The test is `test_amd_gcn_cache_lines`.
- **CI: the playground host can take burst jobs when two of its three slots are free** (`tools/burst-runners/host-burst.sh`, docs/ci.md).
- **A SASS `RET` with a condition predicate returns only the lanes where both predicates hold** (issue #691). On sm_90 ptxas ends a `__noinline__`
  function that returns early when two cells are equal with a `RET` whose second predicate sits in the same bits as `BRA` and `EXIT` have it; the decoder
  dropped it, so lanes that were not equal returned without running the rest, and about a third of an atomic counter's increments were lost. The test
  is `e2e_ret_condition_predicate`.
- **PTX with an `.extern` variable is judged as NVIDIA's driver judges it** (issue #681, follow-up to the entry below). The same RTX 3080 Ti was
  asked about the cases the first change did not cover, and answered differently from the simulator in four of them. Now, as on the card:
  an `.extern` variable with an initialiser (`.extern .global .u32 e = 5;`) fails the load with `CUDA_ERROR_INVALID_PTX`
  ("No initial value is allowed for external symbol"); so does an unsized `.extern` array (`x[]`) that the module does not define
  ("Unresolved extern variable ... in whole program compilation"; linking another module that defines it is fine); so does an `.extern`
  whose only definition is a plain one without `.visible`, `.weak` or `.common` (".extern variable cannot be resolved by a .static");
  `cuLinkComplete` on an input with an undefined `.extern` variable fails with `CUDA_ERROR_INVALID_SOURCE` and "Undefined reference to 'x'" in the
  error log (it linked before, and gave the variable zeroed storage), while the same input linked with one that defines it works; and the info log
  of `cuModuleLoadDataEx` carries the `ptxas warning : Unresolved extern variable 'x' in whole program compilation, ignoring extern qualifier`
  that the card prints for the zeroed-storage case. Not changed: a plain definition **before** an `.extern` declaration of the same name (the order
  not measured on the card). Tests: `c_harness_extern_global` (its second module had a plain definition, which the card refuses; it is `.visible` now)
  and `e2e_driver_extern_global` (same fix, and the unsized array it loaded is gone, since the card refuses it).
- **LDS: the 12-byte case of a 96-bit access is measured** (follow-up to the bank-conflict rule). `ds_read_b96` and `ds_write_b96` at a base that
  is a multiple of 12 but not of 16 count no conflicts on an RX 6800, like 4 and 8, and the alignment is a property of the whole access (a stride
  of 12 gives no conflicts although lane 0 is aligned), so the model's 16-byte rule stands. 32 rows are in `amd/tests/data/lds_bank_alignment.tsv`,
  checked by `test_amd_gcn_lds_banks`. Also measured: on the RX 6800 only the SQ counters work under `rocprofv2` (no TA, TCP or L2 counters), and
  `rocprofv3 --pmc` aborts there even with the iGPU hidden, so the bytes counters can be checked against the card only through the instruction counts
  they are built from (copy, triad, float4 copy, half-lane and reuse kernels: FLAT, VALU and SMEM per wave match the simulator exactly), not
  against a memory counter.
- **A program that `dlopen`s libcuda after libcudart has started finds the driver initialised** (issue #685). On a card the runtime brings the driver up, so a
  program that links only libcudart and loads `libcuda.so.1` itself (cedana-samples' `jit_modules` and `vmm_pools`) can call `cuCtxGetDevice` with no
  `cuInit` of its own. The driver shim kept its own flag, and the runtime's lookup of `cuInit` cannot see a library opened `RTLD_LOCAL` or opened late:
  the call answered `CUDA_ERROR_NOT_INITIALIZED`. The shim now initialises itself on its first call when the runtime has started, and the thread gets the
  runtime's primary context as it does on a card.
- **A copy, memset or device-to-device copy over several back-to-back `cuMemMap` chunks of one reserved range works** (issue #686). Each mapping is an
  allocation of its own and a bulk access was checked against the first one, so a copy of two chunks or more failed with "runs past the end of the
  2097152-byte mapping" (PyTorch's expandable segments, vLLM's sleep mode). A read, write or fill is now made a mapping at a time where the next mapping
  starts where the last ended; a range that runs into a gap is still an overrun.
- **Behaviour change: a read of `%globaltimer` costs 64 ns of the simulated clock, so a busy-wait on it no longer runs out of steps** (issue #687). `%clock`,
  `%clock64` and `%globaltimer` count one per warp instruction, so a kernel that waited two seconds on `%globaltimer` (cedana-samples' `stream_race`
  does, to keep a stream busy) ran about two billion instructions, reached the default step budget (1,073,741,824) and stopped as a possible infinite
  loop, after some twenty seconds of host time; on an RTX 3080 Ti it just runs 2 s. A read of the timer is slow on a card, and now charges 64 on top of the
  instruction (once per warp read, not per lane; `%globaltimer`, `%globaltimer_lo` and SR_GLOBALTIMERLO, in the PTX and SASS engines, the SASS tier too):
  a wait of N ns is about N / 68 passes of a four-instruction loop, so two seconds is some 125 million instructions and the wait has to last about
  seventeen seconds to reach the budget. The 64 is a modelled latency, not a measured one. What changes for a program: `%globaltimer` values grow
  faster, by 64 per read, and `%clock64` is unchanged in the SASS engine (in the PTX engine both read one counter, as before, so `%clock64` moves by the
  same 64 when the timer is read). A wait on `%clock64` still costs one instruction per cycle. The step-budget error now says a long wait on a clock reaches
  it and that `--max-steps 0` lifts it, and `docs/status.md` describes both.
- **CI: the CUDA toolkits workflow retries its apt steps.** NVIDIA's mirror is sometimes mid-sync (`File has unexpected size ... Mirror sync in progress?`),
  which failed `build + tests, CUDA 12.4` on a pull request that changed nothing there; the install now tries each network step three times.
- **An undefined `.extern .global` (or `.const`) in a PTX module is a zero-initialised variable, as on NVIDIA's driver** (issue #681). A module that
  declares `.extern .global .align 4 .u32 x;` and defines it nowhere used to fail at the first launch that read it
  (`CUDA_ERROR_NOT_FOUND: unknown symbol`), and `cuModuleGetGlobal` could not find it. On a real RTX 3080 Ti the driver drops the qualifier in a
  whole-program module (`ptxas warning : Unresolved extern variable`), gives the variable its declared size, zero, and lets `cuModuleGetGlobal`
  return it and the host write it. The simulator does the same. PoCL's CUDA backend declares `_global_offset_x` this way and launches every kernel
  with it, so OpenCL through PoCL can now launch kernels on the simulated NVIDIA GPUs. A name the module defines itself keeps its definition.
- **CI: the longest tests are split into shards, and the slowest tests are reported** (CI work). `test_amd_gcn_gather_same_bits` (2,461 s under ASan),
  `test_amd_gcn_memory_same_bits` (1,866 s), `test_amd_matrix_fast` (1,553 s) and `e2e_sass_tier` (1,830 s under ASan, 1,580 s under TSan) now run as
  eight shards each, side by side (`..._shard_KofN`; each shard keeps every item's seed, and a `_shards_partition` test proves the shards are the whole
  test). The ml job no longer re-runs the heaviest simulator-only tests that the unit job runs. Every job that runs ctest puts its 20 slowest tests
  in the job summary (`ci/slowest_tests.sh`), and a test over 30 minutes fails the job. No check was removed.
- **Behaviour change: MUFU .F16 and .BF16 on sm_90 and later (Hopper, Blackwell) are the card's.** From sm_90 the instruction's `.F16`
  bit moved from instruction bit 73 to bit 72 and bit 73 became `.BF16` (nvcc output for sm_90, sm_100 and sm_120; the same words).
  The decoder read bit 73 as `.F16` for every architecture, so on sm_90 and later a real `MUFU.<op>.F16` ran as a plain 32-bit
  `MUFU` on a half register and a real `MUFU.<op>.BF16` ran as `.F16`. It reads them by architecture now, and `MUFU.<op>.BF16` and
  the PTX `ex2.approx.ftz.bf16`, `.bf16x2`, `tanh.approx.bf16` and `.bf16x2` are the single-precision unit on the bfloat16 widened
  by a shift, the upper 16 bits of the result: equal to an H100 PCIe on all 2^32 register values of all eight sub-operations.
  **Programs for sm_90 and later that use f16 or bf16 `ex2`/`tanh` or compiler-emitted `MUFU .F16` / `.BF16` change their outputs;
  they now equal the card's** (the PTX bf16 forms were the function at host precision, rounded to nearest). Counters do not
  change. **To get the earlier results back:** `VGPU_MUFU_EXACT=0`. Evidence: Hopper measured (H100 PCIe); Blackwell inferred (the
  encoding is from nvcc output, the execution was not run on a Blackwell card). Tests: `test_mufu_model`, `test_decode_same_bits`
  (its flipped and random digests moved for the MUFU words with bit 72 set on sm_90 and later, see its header); details in
  [docs/numerics/math-functions.md](docs/numerics/math-functions.md).
- **Behaviour change: the b200 and gb200 profiles sum tensor-core products by the Hopper rule by default, not sequentially.** On
  `nvidia/b200` and `nvidia/gb200` (compute capability 10.0), `wmma`, `mma.sync` and SASS `HMMA` in f16, bf16 and tf32 now add the
  products in chunks of 16 (tf32: 8) with the terms cut at 2^(top - 26), the rule of an H100, instead of one at a time in f32.
  **The outputs of programs that use those instructions on these two profiles change** (and so do their cached results: the numerics
  are part of the result-cache key). Counters do not. Why: the 20,000 test vectors Khattak and Mikaitis recorded on a B200
  (arXiv 2512.07004; f16 with f32 and f16 accumulators, bf16, tf32) are reproduced bit for bit by this rule and 8,020 of them are
  not by the sequential sum. The class is `inferred` (`vgpu info --gpu nvidia/b200`): that is data recorded by others, **we have not
  run a B200**, and no GB200 was recorded. `tcgen05.mma`, `wgmma`, the sparse, FP8 and f64 forms are not covered and did not change.
  **To get the earlier results back:** `VGPU_TC_EXACT=0` (the sequential sum on every device, as before; no new variable).
  `VGPU_TC_EXACT=published` no longer does anything on b200 and gb200 (it is already in force). Not changed: sm_103 (b300), sm_120
  (rtx5090, rtx-pro-*), thor, gb10 and vr200 keep the sequential sum by default (no recorded vectors cover them; the sm_120
  profiles keep the published Hopper model opt-in). Tests: `test_exec3` (`blackwell_b200_gb200_default_to_the_hopper_rule_and_the_rest_do_not`,
  pins of `tc_model_devices`), `test_profile` (provenance rows), `info_cli`, `simd_lane_forms`, `tcpapers`; details in
  [docs/matrix-numerics.md](docs/matrix-numerics.md) and [docs/numerics/tcpapers.md](docs/numerics/tcpapers.md).
- **The setup action installs NVTX's headers** (issue #677). A project that includes `nvtx3/nvToolsExt.h` (Arbor does, in `cuda_api.hpp`) failed to compile on a runner with a versioned CUDA toolkit, because the action installed the other libraries' headers but not NVTX's. It installs `cuda-nvtx-<version>` now. NVTX calls are no-ops with no profiler attached, so annotated programs run unchanged on the simulated GPU.
- **Double arithmetic in a kernel no longer inherits the host program's flush-to-zero** (issue #672). A host built with `nvc -fast`
  (or `-ffast-math`) turns FTZ and DAZ on in its own MXCSR, and the engines did a kernel's arithmetic with the host's
  instructions under that setting, so a subnormal double operand read as zero and a subnormal result was written as zero. NVHPC
  26.9's `fmod(double)` reads the exponent of a subnormal remainder through a multiply by 2^54: on the simulated H100 `fmod(1.5, 1)`
  was 1.4e-17 (the McStas PSI_DMC runs reached 39 % of their reference). A launch now clears FTZ and DAZ (and ARM's FZ) for the
  kernel and puts the host's back afterwards (`include/vgpu/exec/host_fp.hpp`, test `test_launch_fp_flush`); a kernel that asks
  for `.FTZ` still flushes, by the engine's own rule.
- **The half-precision approximate math forms give a card's bits** (issue #664, follow-up). `ex2.approx.f16`, `.f16x2` and
  `tanh.approx.f16`, `.f16x2` (PTX interpreter) and `MUFU.<op>.F16` (SASS executor, all eight sub-operations) are what an RTX 3060
  computes, bit for bit over all 65,536 half patterns of each: the single-precision unit on the exactly widened half, cut toward
  zero to a half (`eval_f16` in `include/vgpu/exec/mufu_model.hpp`). Before: `ex2.approx.f16` differed on 19,417 of 65,536
  patterns and `tanh.approx.f16` on 14,736 (host precision, rounded to nearest); the SASS executor ignored the F16 bit and read
  the register as a float (65,536 of 65,536 patterns off for `MUFU.EX2.F16`, `COS`, `LG2`, `RCP`, `RSQ`). Also decoded now: the
  `.H1` half-select of `MUFU.<op>.F16`. **Measured on sm_86 only**: Turing, Ada, Hopper and Blackwell cards were not asked
  (inferred from the 32-bit unit); the bf16 forms (sm_90 and later) are unchanged and unmeasured. `VGPU_MUFU_EXACT=0` restores
  the earlier results for these forms too. `tools/numerics/mufu-card-run.sh` hashes them (`hashes16.txt`) for another card; see
  [docs/numerics/math-functions.md](docs/numerics/math-functions.md).
- **The approximate math instructions give a real card's bits** (issue #664). `MUFU` (SASS) and the PTX `.approx` forms
  were the function at the host's precision, rounded once; a card computes them in a small fixed-point unit, and its
  results differ in the last bit on 13 to 39 % of arguments (`RCP`, `RSQ`, `SQRT`, `EX2`), by an absolute error of a few
  1e-7 (`LG2`, `SIN`, `COS`) or by up to about 130 ulp (`TANH`). The unit of an RTX 3060 was measured over all 2^32
  operands of each of the ten sub-operations (`tools/numerics/mufu_probe.cu`) and a model fitted that reproduces every one
  of them: 4,294,967,296 of 4,294,967,296 for `COS`, `SIN`, `EX2`, `LG2`, `RCP`, `RSQ`, `RCP64H`, `RSQ64H`, `SQRT` and `TANH`
  (`include/vgpu/exec/mufu_model.hpp`; [docs/numerics/math-functions.md](docs/numerics/math-functions.md) has the
  measurements, the model and its evidence). Both engines use it; the PTX interpreter forms each `.approx` instruction and
  `div.full.f32` as ptxas expands it around the one instruction.
  - **This changes the results of existing programs**, which is the point: any kernel that calls `expf`, `powf`, `tanhf`,
    `rsqrtf`, `cbrt` and the other library functions built on these instructions, or is built with `-use_fast_math`, now
    prints the card's last bits. On the math-function test set (`tools/numerics/mathfn`, 329 lines a build) the SASS executor
    printed the card's line on 263 lines of the default build and 225 of the `-use_fast_math` build; it prints it on all
    329 and all 329 (an RTX 3060, a Tesla T4, an L4 and an A10G print identical lines). Programs that only add, multiply,
    divide (`/` without fast math) and take `sqrtf` are unchanged: those are IEEE operations on the card and here.
    Counters do not change.
  - **`VGPU_MUFU_EXACT=0` gives the earlier results**, bit for bit (the function at host precision; `div.full` the exact
    quotient). Read at every launch.
  - Measured: RTX 3060 (every operand), an A10G and an H100 PCIe (every operand, as hashes), and through the math library a
    Tesla T4 and an L4. **Blackwell cards were not measured**: their profiles get the same model, inferred (measured on
    Turing, Ampere, Ada, Hopper).
    `tools/numerics/mufu-card-run.sh` is the run to make on one: all ten operations over all inputs as hashes, compared with
    the RTX 3060's (`tools/numerics/mufu/rtx3060.hashes.txt`), in seconds.
  - Also measured and now followed: the `|x|` and `-x` operand modifiers of `MUFU.RCP64H` and `MUFU.RSQ64H` apply to the
    upper word's sign (the SASS executor ignored them on those two forms).
  - Pinned test files that changed, each re-recorded from a card and saying so in its header:
    `nvidia/tests/data/numerics_mathfn.txt` and `numerics_mathfn_fast.txt` (now an RTX 3060's lines; 66 and 104 lines
    changed), ten expected values of `tests/unit/test_ptx_approx_forms.cpp` and one of `tests/unit/test_exec3.cpp`
    (`div.full.f32`, which was the exact quotient in the PTX interpreter and is `a * MUFU.RCP(b)` on the card). The card
    lines of `e2e_simd_lane_forms`, `e2e_float_half_forms` and `e2e_round9_probe` did not change.
  - Not modelled: the half-precision forms (`MUFU` with the F16 bit, `.approx.f16`), and AMD's math instructions.
- Math functions (issue #664, second group): a test set that compares a run of the math library with another run line by
  line, simulator against simulator or against a card (`tools/numerics/`: 44 functions in both precisions over published
  worst cases, edges and random arguments; the worked examples of NVIDIA's floating-point paper; the Varity GPU paper's
  cases and generator), a card script (`tools/numerics/card-run.sh`) and what the published results say a card gives where
  the simulator gives something else ([docs/numerics/math-functions.md](docs/numerics/math-functions.md)). No card was run
  for this step and it left `MUFU` unchanged (the entry above is the card runs and the change). Two things it found change results:
  - **A program run from its PTX now gives the bits it gives from its SASS** in the math library. The PTX interpreter
    computed `sin.approx`, `cos.approx`, `ex2.approx` and `lg2.approx` without `.ftz`, `rcp.approx.ftz.f64`,
    `rsqrt.approx.ftz.f64` and `rsqrt.approx.f64` as the function at host precision; ptxas makes other code of each, and
    the same kernel differed between the engines: `rsqrt(double)` on 13 % of arguments and `hypot(double)` on 10 % (by 1 or
    2 ulp), `asinh(double)` on 0.7 %, `exp2f` on subnormal results, `cbrtf` on subnormal arguments, and `sinf`, `cosf` and
    `tanf` under `-use_fast_math` on most arguments. The interpreter now forms them as ptxas does. A program that runs from
    SASS (the default wherever the binary carries SASS the device can run) is unchanged, bit for bit; a program that runs
    from PTX and calls these functions changes in its last bits, or, for the fast-math sine and cosine of a large argument,
    altogether, to what its SASS gives. Left as they were: the NaN `erf(double)` returns for a NaN and `hypot(double)` for a
    signaling one (payloads only).
  - **AMD, wave32 (RDNA) kernels built `-O0`**: a stack slot stored through the private segment buffer and loaded through a
    flat pointer was not the same slot (the buffer's index stride was 64 lanes for every kernel; it is the wave's size), so
    unoptimized code that passes the address of a local to a function read another slot or stopped on an invalid address:
    on the rx6800 profile 441 of 900 results of the Varity programs did not run and 10 were wrong. Optimized code and
    wave64 kernels are unchanged.

- Device profiles say what their numerics are and what each value rests on (issue #664, first step of architecture-aware
  simulation; [nvidia/docs/profiles.md](nvidia/docs/profiles.md), "Numerics and provenance"). **No result changes.**
  - New optional profile sections: `numerics:` (the tensor-core sum: `tensor_sum`, its chunk sizes and grid, and
    `tensor_sum_published` where only a published model exists), `provenance:` (four classes: `measured`, `documented`, `inferred`,
    `unknown`; one class a key) and `evidence:` (card, driver, toolkit and date, or the document). A profile with a numerics key that
    has no class, an unknown class, a key in two classes, a claim without evidence or numbers the engines do not implement does not load.
  - The tensor-core sum of every NVIDIA profile is now read from its profile instead of a table keyed on the compute capability.
    Each of the 38 profiles has the model it had (held by `test_exec3`, in every `VGPU_TC_EXACT` mode). Seven are `measured` (the
    card of that model was run: rtx3060, a10g, t4, l4, l40s, a100-sxm4-40gb, h100-pcie), twenty `inferred` from a measured card of
    the same compute capability (the default `nvidia/h100` among them), and the eleven Blackwell and later profiles are `unknown`
    and keep the sequential sum; seven of those name the published model that `VGPU_TC_EXACT=published` turns on.
  - `amd/rx6800` and `amd/rx6700xt` tag the limits that were read from those cards. No AMD behaviour moved, and no AMD matrix
    arithmetic is tagged: none has been measured.
  - `vgpu info` prints the numerics, the classes and the evidence; `--json` gains `numerics`, `provenance` and `evidence` objects.
  - The numerics are part of the result cache's key, so entries cached by an earlier build are not reused (a one-time miss).
- NVIDIA runtime: `cudaStreamWaitEvent` on an event that is still pending keeps the stream waiting after `cudaEventDestroy` of that event, as on a card. A wait (and a queued record) used to look the event up by its handle when the stream's worker reached it, found it destroyed and took it for complete, so the waiting stream ran early. This is what made Ta-Cuda's batch tests (tacuda#5) fail now and then under the default asynchronous launches and pass with `VGPU_ASYNC_LAUNCH=0`. The queued item now holds the event's state itself.
- AMD engine, the tier's handlers for double-precision and 64-bit integer instructions ([round sp74](docs/perf-design/rounds/11-rounds-sp36-sp41.md),
  [design](docs/perf-design/13-amd-compiled-tier.md), section 13). On hosts with AVX2, with `VGPU_AMD_BLOCKS` off or on:
  - `v_min_f64` and `v_max_f64`, `v_rcp_f64`, `v_sqrt_f64` and the rounding instructions, the 64-bit comparisons (`v_cmp_*_f64`, `_i64`,
    `_u64`), `v_lshlrev_b64`, `v_lshrrev_b64`, `v_lshl_add_u64`, the conversions between doubles and 32-bit values, `v_frexp_*_f64` and
    `v_ldexp_f64` work on the register pairs' rows instead of gathering them into 64-bit lanes; the SDWA forms of the integer instructions
    work on whole rows. Results are the same bits, a NaN's, a signed zero's and a denormal's included: a lane the row form does not do
    sends the instruction to the handler it had.
  - `v_add_f64`, `v_mul_f64` and `v_fma_f64` in a wave with lanes off run only the groups of lanes that are on.
  - The AMD benchmark's double-precision probe (p_f64x) runs 2.4 times (MI300X) and 2.1 times (RX 6800) as fast, p_sdwa 1.8 and 1.5,
    its 62 kernel rows 1.06 in the geometric mean (1.18 with `VGPU_AMD_BLOCKS=1`); no row is more than 3 % slower beyond a control with
    no change. inq's unit tests on a simulated MI300X: eigensolvers::steepest_descent uses 18 % less CPU, ground_state::initial_guess
    12 %, systems::electrons 7 %, solvers::poisson 4 to 5 %; none is slower.
- Tensor cores: the sum now follows the device, and more devices have a measured one ([docs/matrix-numerics.md](docs/matrix-numerics.md),
  issue #664). **Results of f16, bf16 and tf32 `wmma`, `mma.sync` and `HMMA` change on these profiles**; nothing else changes.
  - Compute capability 7.5 (t4, rtx2080ti), 8.0 (a100*, a30) and 8.9 (l4, l40, l40s, rtx4080, rtx4090, rtx6000-ada) take the RTX 3060's
    rule (chunks of 8 products, tf32 4, truncating). Measured 2026-11-15: a Tesla T4, an A100-SXM4-40GB, an A10G, an L4 and an L40S print
    the RTX 3060's probe lines; the A100 also 8,388,608 of 8,388,608 dumped results.
  - Compute capability 9.0 (h100*, h200*, gh200-480gb) takes an H100's: the same rule in chunks of 16 (tf32 8), cut one bit lower.
    Measured on an H100 PCIe: 8,388,608 of 8,388,608 dumped results and the card's own probe lines on both engines. The FTTN paper's
    motivating example now gives 191.875 on the h100 profile, as on the card and in the paper (it gave 0).
  - The SASS engine chose the sum by the cubin's architecture and the PTX engine by the profile, so the two disagreed for an sm_86
    binary on the default (h100) profile, and an sm_80 binary on an rtx3060 profile did not sum as the card does. Both ask the device.
  - Four corners of the rule in the subnormal range are fixed (two confirmed on an RTX 3060, two from the published models): a
    subnormal f32 result is cut, not rounded; a chunk starting from a zero accumulator has an alignment floor; a subnormal accumulator
    counts at the smallest normal exponent; a negative sum that rounds to zero is +0.
  - `VGPU_TC_EXACT=0` still gives the sequential sums everywhere. New values: `force` (the RTX 3060's rule on devices without a measured
    one) and `published` (Blackwell profiles take the published model, the H100's numbers: no card of ours has run it, so it is off by
    default). Blackwell and the other unmeasured capabilities keep the sequential sum.
  - Cost: the H100 rule and every tile with an f16 accumulator run in scalar code; their speed was not measured.
- AMD: `VGPU_MFMA_CDNA2=fttn` (off by default) makes the mi250x profile's f16 and bf16 matrix instructions sum as the FTTN paper's
  MI250X row reads (subnormals as zero, one f32 rounding after every product). Paper only: no AMD matrix-core card has been measured,
  and the default sums of every AMD profile are unchanged.
- `tools/fttn/`: fetches, builds and runs the FTTN test suite (arXiv 2403.00232) on the simulator and on a card, with the paper's
  motivating example, a card script, a fitter for tensor-core dumps and a checker that holds the simulator's code to a card's dump.
## [0.2.1] - 2026-11-15

Fixes and additions since 0.2.0. The two bugs that release shipped are fixed: `shfl.sync` returning 0 for a lane still in a long loop
([#488](https://github.com/pantheongpu/pantheonsim/issues/488)) and a kernel that could not read another simulated GPU's memory after
`cudaDeviceEnablePeerAccess` ([#598](https://github.com/pantheongpu/pantheonsim/issues/598)). The release carries the simulator prebuilt for the
GitHub Action. The list below is every entry that was under Unreleased.

- AMD counters: the `VGPU_DISPATCH_STATS` line now ends with the bytes the active lanes ask for (`vmem_bytes_rd`, `vmem_bytes_wr`, `lds_bytes_rd`, `lds_bytes_wr`),
  the vector instruction mix (`mix=`), the matrix work by type (`mops=`) and the kernel's name (`kernel=`). The mix and the matrix work were already counted
  and are now printed; the bytes are new. A reader that splits the line on `=` and converts every value to a number must skip the text fields
  (`tools/bench-amd/bench.py` does). Requested in [discussion 631](https://github.com/pantheongpu/pantheonsim/discussions/631), issues #648 and #649.
- AMD counters: LDS bank conflicts. `VGPU_DISPATCH_STATS` also prints `lds_bank_conflicts`, the passes through LDS's 32 banks beyond the first.
  The rule was measured on an RX 6800 (`SQC_LDS_BANK_CONFLICT`, 906 combinations of instruction, stride and active lanes,
  all reproduced; data in `amd/tests/data/lds_bank_conflicts.tsv`, probe in `amd/tools/lds-bank-probe`) and is applied unchecked to the other targets.
  It is counted only when `VGPU_DISPATCH_STATS` is set. Sectors and cache lines are not counted yet (issue #650).
- AMD engine, blocks (`VGPU_AMD_BLOCKS=1`, still off by default), second round ([round sp73](docs/perf-design/rounds/11-rounds-sp36-sp41.md),
  [design](docs/perf-design/13-amd-compiled-tier.md)):
  - **Fixed**: the first form of blocks made the interpreter slower with the knob off. `v_fma_f64` loops (the bench's valu_f64) ran 13 %
    slower on the simulated MI300X and 23 % on the RX 6800, other rows 2 to 5 %. With the knob off the interpreter's loop is again the
    code it was: 0.993 to 1.001 of the rate before blocks over the bench's 62 kernel rows in two builds (a control with no change:
    0.999 to 1.001). Single rows still move 3 to 5 % from one link of the library to the next (smem on the RX 6800 is 5 % low in this one).
  - The three functions of the blocks' loop keep frame pointers: a profiler that walks the stack in its signal handler crashed one run in
    four of an inq test with the knob on, and no longer does (12 of 12 runs).
  - Memory instructions, scalar loads and waits are inside blocks, and a wave whose turn ends inside a block goes on in it. The rows
    that lost with the knob on in the first form no longer do (`lds` 0.92 to 0.96 before, 0.99 to 1.01 now; `mem` 1.00 to 1.02): where
    waves give up their turn every few instructions blocks turn themselves off.
  - Runs of 32-bit integer instructions on vector registers and constants call their lane kernels directly (`VGPU_AMD_BLOCKS_BOUND=0`
    turns that off): valu_int 1.33 times the rate without blocks (1.16 to 1.21 without the binding), p_sdwa 1.20 to 1.30.
  - With the knob on, one host thread: geometric mean of the bench's 62 kernel rows 1.11, arithmetic rows 1.09 to 1.35, no row more than
    2 % slower than with it off; inq's unit tests use 1 to 4 % less CPU (round sp72's 16 % for one of them was a loaded host's number). Results, counters and errors are those of one instruction at a time.
- AMD engine, the first piece of a compiled tier for AMD kernels ([design](docs/perf-design/13-amd-compiled-tier.md),
  [round sp72](docs/perf-design/rounds/11-rounds-sp36-sp41.md)): with `VGPU_AMD_BLOCKS=1` (off by default) a straight-line run of
  instructions the pre-decoded tier has handlers for goes as one *block*: the handlers are called back to back, what was asked before each
  instruction is asked once, and the run's counters are added once. Results, counters, errors, the order waves and work-groups take turns in
  and lockstep are those of one instruction at a time (`test_amd_gcn_blocks_same_bits`, `bench_amd_counters_blocks`). On one
  host thread the AMD benchmark's arithmetic rows run 6 to 49 % faster with it (valu_int 1.06 to 1.26, valu_f32 1.13 to 1.25, valu_f64 1.10
  to 1.32, p_i64 1.19 to 1.30, wave 1.30 to 1.43; geometric mean of all 82 rows 1.07) and inq's unit tests use 0 to 16 % less CPU; the
  memory rows do not gain and the LDS row is 4 to 6 % slower, which is why it is opt-in.
  `VGPU_AMD_BLOCKS_STATS=1` lists the blocks that ran most. Measured first: the dispatch loop is 18 to 35 % of an arithmetic-bound kernel on
  this engine and operand decode 17 to 32 %, and the same arithmetic written as one host loop over a wave's lanes costs 13 to 22 times less
  than the tier, which is what the design's later layers are for.
- AMD run-time compilation (hiprtc) is cached ([round sp71](docs/perf-design/rounds/11-rounds-sp36-sp41.md)). A program that compiles kernels at
  run time (VkFFT's plans, CuPy's and PyTorch's kernels) calls ROCm's own `libhiprtc`, about 150 ms a kernel on any GPU; `build/shim` now has a
  `libhiprtc` (`.so.6` and `.so.7`) that loads ROCm's, forwards every entry point, and keeps what a compile returned -- the code, the log, the
  lowered names -- in memory and in a file under `~/.cache/pantheonsim/hiprtc-v1`, keyed by everything the compile reads (source, name, headers,
  options, name expressions, the device's architecture, the compiler's environment, and the toolchain's libraries and files). The same compile, in
  the process or a later one, gets the bytes ROCm's library returned. Whatever the key cannot name completely is compiled by ROCm every time, as
  before: includes that were not given (except the guarded `<hip/hip_runtime.h>`), `__DATE__`/`__FILE__`, options such as `-g` and `-include`,
  failures, the bitcode and the link API. Six of inq's unit tests on the simulated MI300X: 217 s with the cache off, 172 s cold, 64 s warm (410, 345
  and 237 CPU-s), the same outputs. On by default; `VGPU_HIPRTC_CACHE=0` or `mem` (settings key `hiprtc_cache`), `VGPU_HIPRTC_CACHE_DIR`,
  `VGPU_HIPRTC_CACHE_MAX_MB` (256), `VGPU_HIPRTC_CACHE_STATS=1`, `VGPU_HIPRTC_REAL` ([cli.md](docs/cli.md)). Run-time compilation still needs
  ROCm's compiler installed, as it did.

- Host-GPU mode (experimental, opt-in), both modes: the share of the driver start that a launch must save by itself (the entry below) kept the
  start from programs of many short launches that repay it many times over: Rodinia hotspot, 1,000 launches of about 1 ms, was 2.27x faster in
  turbo mode before the share and 1.00x with it. A kernel the simulator has already run 200 times is now taken to be launched as many times
  again, and its launch must repay the start within the launches of that kernel run so far (at most 500) instead of within 50, when those
  are more (`VGPU_HOST_GPU_START_REPEATS`, `0` for the rule before). The four short CUDA Samples the share protects still do not start the
  driver: two launch a kernel about 100 times, one has two launches in all, and the launches of the fourth save under 0.5 ms each. On an
  A10G hotspot's turbo run is 1.85x faster than the simulator again (1.05 s to 0.57 s) and the four run at the simulator's time.
- AMD engine, speed on a program of doubles, reductions and managed memory (inq's unit tests on the simulated MI300X,
  [round sp70](docs/perf-design/rounds/11-rounds-sp36-sp41.md)); results, counters and the order work-groups run in are unchanged:
  - **A default changes**: without `VGPU_THREADS` a dispatch is spread over one host thread for each physical core the process may use
    (`include/vgpu/host_cores.hpp`), not one for each hardware thread. On an 8-core, 16-thread host the second thread of a core bought no wall
    time: the default now uses 17 to 52 % less CPU time for the same wall time (ground_state::initial_guess 280 to 134 CPU-s, solvers::poisson
    327 to 189, wall time within 3 %), and four tests with large grids run at once finish 3 to 4 % sooner. `VGPU_THREADS` still takes any number. The NVIDIA engines are unchanged.
  - The per-thread pool of vector register files holds 288 files instead of 32: the work-groups of a kernel with barriers (a reduction) are
    held several at a time and end together, and each burst took fresh 64 KB files zeroed whole (initial_guess 44.3 s to 38.3 s, 374 to 280
    CPU-s; systems::electrons 162 to 122 CPU-s).
  - `v_min_f64` and `v_max_f64` are four lanes at a time where the host has AVX2, with the C library's fmin/fmax bits
    (eigensolvers::steepest_descent 45 to 41 CPU-s).
  - A wave's three- and four-word loads and stores of managed or host memory (`hipMallocManaged`, `hipHostMalloc`) go to memory as one batch
    on one host thread, as its one- and two-word ones did (hamiltonian::ks_hamiltonian with `VGPU_THREADS=1`: 30.7 s to 27.4 s).
- NVIDIA driver: `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleLoadFatBinary` and `cuModuleLoad` of a fatbin with nothing the
  GPU runs (SASS only for other architectures, PTX only for newer ones) answer `CUDA_ERROR_NO_BINARY_FOR_GPU` (209), as an RTX 3060
  does for sm_90-, sm_89- and sm_75-only fatbins and a compute_90 PTX-only one; the simulator answered `CUDA_ERROR_NOT_SUPPORTED` with
  "fatbin contains no PTX image" (or `CUDA_ERROR_INVALID_PTX`). NVHPC's first load of its SASS-only fatbin (before it links the code
  with `cuLink*`) now fails as on a real GPU, silently.
- cuBLAS Ex routines, the type checks in the card's order: in `cublasGemmEx`, its batched and strided forms, A and B of two types
  are `NOT_SUPPORTED` before a bad argument, and an unknown data type (outside 0..15) is `INVALID_VALUE` after it, as is an unknown
  compute type in the batched forms; `cublasCsyrkEx`, `cublasCherkEx` and their 3m forms check their arguments first, take n = 0 whatever
  the types, and answer an unknown type `INVALID_VALUE`. A CUDA 10 compute type (`CUDA_R_32F`, `CUDA_C_32F`, `CUDA_R_64F`, `CUDA_C_64F`,
  `CUDA_R_16F`, `CUDA_R_32I`) is taken as the card takes it. `cublasGemmBatchedEx` with a negative batch count is
  `CUBLAS_STATUS_EXECUTION_FAILED` once its other checks pass, success with an empty C, and `INVALID_VALUE` for a null C array;
  `cublasGemmStridedBatchedEx` with no batch, or a negative count, succeeds once its checks pass. The simulator answered
  `NOT_SUPPORTED` for every type problem before the arguments, and `INVALID_VALUE` for any negative count.
- cuSPARSE: `cusparseSpMV` and `cusparseSpMV_bufferSize` take only the fifteen (A, x, y, compute) type combinations an RTX 3060's
  cuSPARSE 13.0 takes, measured over all 10,000 combinations of the ten types for CSR, CSC and COO and each operation; the others are
  `CUSPARSE_STATUS_NOT_SUPPORTED` with the card's "type combination ... is not supported" line. The simulator computed 1257 more (a
  single A with half or double vectors of mixed kinds, int8 into double, ...).
- NVIDIA runtime: `cudaGetDeviceFlags` answers the current context's flags, or the device's primary context's with
  `cudaDeviceMapHost` when no context is current, as an RTX 3060 (CUDA 13.0) does (after `cudaSetDeviceFlags`, `cudaInitDevice` with
  `cudaInitDeviceFlagsAreValid`, or `cuDevicePrimaryCtxSetFlags`); it answered 0, and success for a null pointer. `cudaSetDeviceFlags`
  now sets the primary context's flags (on an active context too) and makes the context active and current, and refuses a value above
  0xff or two scheduling modes with `cudaErrorInvalidValue`; it took any value and did nothing.
- NVIDIA runtime, stream capture: anything a program does on the legacy stream while a blocking stream takes part in a capture is
  now `cudaErrorStreamCaptureImplicit` (906), is not done, and invalidates every capture a blocking stream takes part in, in every
  capture mode, as an RTX 3060 (CUDA 13.0) answers: a launch, a copy or fill (the synchronous `cudaMemcpy` and `cudaMemset` too), an
  event record, `cudaStreamQuery` and `cudaStreamSynchronize`, a host function, a stream-ordered allocation or free, a prefetch, a graph
  launch or upload, and the stream's flags, priority, id, device and attributes; `cudaStreamIsCapturing(0)` and
  `cudaStreamGetCaptureInfo(0)` answer it without invalidating. A copy or fill of nothing, an unknown copy kind and a null result
  pointer are answered first, as on the card. The simulator did all of them (only `cudaStreamWaitEvent` was refused) and built a graph
  that lacked the work. The simulator's own libraries' runtime calls are not counted. `cudaStreamIsCapturing` with a null status is
  `cudaErrorInvalidValue` (it answered success). Driver API calls on the legacy stream are not covered yet.
- NVIDIA driver: a call made after `exit()` has torn the driver down, from the destructor of a library that loaded libcuda itself, now
  answers `CUDA_ERROR_DEINITIALIZED` (4), as NVIDIA's driver does (#646). NVHPC's OpenACC runtime (26.9) dlopens libcuda, retains the primary
  context once and releases it from its destructor, after the driver's own teardown; the simulator answered that release from its destroyed
  state as one with nothing retained (`CUDA_ERROR_INVALID_CONTEXT`, "released more times than it was retained"), and every `nvc -acc`
  program, McCode/McStas `--openacc` instruments among them, ended with "Accelerator Fatal Error" and exit status 1 after a correct run. An
  RTX 3060 answers the same release 4, which NVHPC ignores. A release with nothing retained while the program runs is still 201, as on the
  card.
- `vgpu run --gpu amd/...` needs only the HIP shim ([#647](https://github.com/pantheongpu/pantheonsim/issues/647)): it no longer refuses
  to start when `build/shim/libcuda.so.1` is missing, and says so when `libamdhip64.so.6` is, or when the program needs an AMD library the
  build does not have (ROCm SMI, AMD SMI, rocprofiler-sdk), with the target that builds it. `cmake --build build --target vgpuhip` now
  also builds `vgpu` and writes the HIP links in `build/shim` on every build, not only when the library is relinked, so a HIP-only build
  runs. NVIDIA runs still require `libcuda.so.1`, with the same message.

- Host-GPU mode (experimental, opt-in), both modes: while the real driver is not started, a launch that would start it only because the program
  has already simulated as long as the start costs (the start-cost account) must now also save a share of the start by itself: 5 ms, or 12.5 ms
  for a kernel's first launch judged by its static estimate (`VGPU_HOST_GPU_START_LAUNCHES`, `0` for the rule before). Four short CUDA Samples
  (LargeKernelParameter, threadFenceReduction, MC_EstimatePiInlineP, dct8x8) paid a 0.15 to 0.25 s start for launches of a few milliseconds and
  ran 7 to 47 % slower than in the simulator; launches long enough to repay the start, and every launch once the driver is up, are decided as
  before. With `VGPU_HOST_GPU_LOG` the launch that starts the driver says which rule sent it.

- Tier 2 (`VGPU_JIT=1`, still off by default), the compile ([round sp64](docs/perf-design/rounds/11-rounds-sp36-sp41.md)): a section
  whose compile is running is now on the plain run loop instead of the counting loop, which cost the program 2 to 8 % for as long as the
  compile took (a whole run when the compile did not finish before the end, as at sixteen threads). With GCC the runtime header that every
  generated source includes is compiled once into the cache's runtime directory (about 53 MB) after the first compile of a process that
  found none, and every later compile reads it: 0.33 to 0.37 s less a compile on the measuring host, half of a small section's, the same
  object bytes; `VGPU_JIT_PCH=0` turns it off. `VGPU_JIT_STATS` prints a `jit-compile:` line for each compile (why, size, when it was asked
  for, started and done, how it ended) and a `jit-pch:` line. At sixteen threads a compile longer than the program still never finishes.
  Results and counters are unchanged.

- Host-GPU mode (experimental, opt-in), both `exact` and `turbo`: while the card holds memory the simulator does not (the launch before ran
  there), a launch of a kernel with an estimate is weighed with what each side really moves: keeping it in the simulator costs its time plus
  pulling back every chunk only the card holds, sending it costs the launch, the kernel's time on the card and the push of what only the
  simulator holds. Before, a short launch between two card launches stayed in the simulator for its own time alone, pulled the whole live memory
  back, and the next card launch pushed it all again. Measured on one A10G (Round sp65): PolyBench LU 11.6 s to 1.5 s in turbo mode and with
  `VGPU_HOST_GPU_DIV=1`, Rodinia gaussian 0.99 s to 0.53 s, GRAMSCHM's traffic from 98 GB to 40 GB each way; outputs identical.
  `VGPU_HOST_GPU_STREAK=0` keeps the old rule; the summary's `host_gpu_streak:` line counts the launches sent this way
  (design notes, since removed, section 9). A launch with a grid or block dimension of 0 now stays in the simulator, which
  reports its own configuration error, instead of failing on the card (it ended the mode with an internal error when the card held memory).
- Host-GPU mode (experimental, opt-in), exact mode: `VGPU_HOST_GPU_SQRT=1` also admits kernels whose only instructions outside the exact set are
  the IEEE square root sequence nvcc builds for `sqrtf` and `__fsqrt_rn` (`MUFU.RSQ`, a range check, a refinement, a slow-path subroutine), alone or
  beside the division sequence of `VGPU_HOST_GPU_DIV=1`. The sequence is recognised by what it computes, every path of it against the code an A10G
  ran over all 2^32 inputs with no difference; a bare `rsqrtf` stays out. Off by default: unlike the division's, the refinement is not exact for every
  reciprocal square root one ulp off, so it is exact only on a card whose `MUFU.RSQ` passes `tools/probes/hostgpu_mufu_card_experiment.sh` (so far the
  A10G). On an A10G (Round sp68) every launch of the admitted kernels was identical under `VGPU_HOST_GPU_VERIFY`; HeCBench hellinger ran 6.4x faster;
  PolyBench GRAMSCHM's device part went from 44.6 s to 3.1 s once combined with Round sp65's rule (on its own the one-thread kernel stays in the
  simulator for its size). `tools/probes/hostgpu_idiom_scan` lists, without running a program, which of its kernels are in the exact set and by which
  idiom ([docs/cli.md](docs/cli.md), [rounds](docs/perf-design/rounds/11-rounds-sp36-sp41.md)).

- Host-GPU mode (experimental, opt-in), `VGPU_HOST_GPU_VERIFY`: after each compare the chunks whose bytes on the card equal the simulator's
  stay on the card, and only the others (and what the host wrote since) are pushed before the next launch; before, all live memory was pushed
  before every launch. The simulator still runs each launch with no card copy valid, and the program's output is the simulator's as before.
  On a stand-in card (server3, Round sp67) PolyBench 3DCONV under verify goes from 39.3 s to 6.2 s (0.37 s plain), the other seven programs
  measured from 1.0x to 1.4x faster; on an A10G 3DCONV 46.5 s to 8.5 s and the others 1.04x to 1.29x faster, every launch identical; the exit summary's "MiB went to the card" is smaller accordingly
  (design notes, since removed, section 8, pantheonsim #569).

- Host-GPU mode (experimental, opt-in), both `exact` and `turbo`: what a kernel's module holds besides code no longer keeps the kernel in the
  simulator by itself. A kernel that uses the module's `__device__` or `__constant__` variables can run on the card: the card is given a copy of
  the module whose references to the variables are the simulator's addresses (so `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`,
  `cudaGetSymbolAddress` and the simulator's own kernels see the same copy), and the constant banks with `__constant__` variables are filled from
  the simulator's before each launch. A kernel is judged by the external functions it calls itself, not by its module's: one that calls `malloc`,
  `free` or `assert`, or launches kernels, stays in the simulator; a module with `__managed__` variables still keeps all its kernels there. A live
  texture object no longer keeps every launch of the device off the card (texture and surface instructions are refused by the kernel's code, as
  they always were), nor does a launch from a CUDA graph (a kernel that sets a conditional handle is kept by what it calls). Exact mode's results are still the simulator's bits. Turbo mode also admits kernels that call `printf` with an immediate absolute call (the card's lines are
  printed when the launch is waited for, after the program's earlier output; `VGPU_HOST_GPU_TURBO_OFF=printf` keeps them out; nvcc 13.0 calls
  `vprintf` through a register, which is not followed yet, so on that toolkit such kernels still stay) and, by default, float
  atomics with a returned value, compare-and-swap and exchange (`VGPU_HOST_GPU_ATOMICS=0` keeps them out; an opt-in before).
  `VGPU_HOST_GPU_WIDER=0` restores the earlier rules for A/B runs, and with `VGPU_HOST_GPU_LOG` the exit summary lists per kernel the launches kept
  in the simulator and why, with their simulator seconds (`host_gpu_kept:`) (design notes, since removed, section 10;
  Round sp66; pantheonsim #587). Measured on one A10G: the 45 programs whose decisions change go from 1.08x to 1.12x
  (exact) and from 1.20x to 1.60x (turbo) against the simulator, outputs identical; four short programs lose 0.13 to 0.24 s to the driver start.
- Under `vgpu run --preload`, where the driver is loaded privately (#597), CUPTI and NVSHMEM now find it: they looked driver
  functions up in the global scope only, which a private library is not in. CUPTI's callbacks named a placeholder for the current
  context instead of the one the program's driver calls find current, and `nvshmemx_cumodule_init` and `nvshmemx_culibrary_init`
  failed, so a module loaded with the driver API never got its `nvshmemi_device_state_d`. They now ask the runtime's lookup, which
  tries the private driver first (`nvidia/src/driver_symbol.hpp`). So does the runtime's own check for driver-API managed memory
  before deferring an asynchronous copy, which still looked in the global scope only. `e2e_preload_torch_like` runs a third program
  (`preload_torch_peers.cpp`), and `lint_driver_lookups` keeps any shim other than the driver from looking a driver function up in
  the global scope alone.
- `cuProfilerStart`, `cuProfilerStop`, `cudaProfilerStart` and `cudaProfilerStop` answer as an RTX 3060 does (CUDA 13.0); they
  used to succeed always. The driver's return `CUDA_ERROR_NOT_INITIALIZED` before `cuInit`, `CUDA_ERROR_INVALID_CONTEXT` with no
  current context, `CUDA_ERROR_CONTEXT_IS_DESTROYED` when the current context is dead (destroyed, or a primary context released
  to nothing or reset), and a dead context's fault. The runtime's are runtime calls like any other: they start the runtime and make
  the primary context current on the calling thread, and answer a dead context's fault or `cudaErrorNoDevice`. And after
  `cuDevicePrimaryCtxReset`, the runtime's next call makes the primary context active again, as on the card
  (`cuDevicePrimaryCtxGetState` kept reporting it inactive). New test `test_profiler_control`.
- Runtime: every call that fails makes its error the thread's last error, as on an RTX 3060. Only the calls that went through the
  runtime's guard did: `cudaEventElapsedTime` on an event never recorded, `cudaStreamWaitEvent` with an unknown flag,
  `cudaStreamBeginCapture`/`cudaStreamEndCapture` refusals, `cudaMemcpy2D` with a pitch too small, `cudaMemcpy3D` and `cudaMemset3D`
  with bad arguments, a captured event's `cudaErrorCapturedEvent` and others left `cudaGetLastError` at success. A query's not-ready
  still sets nothing, and a call the simulator makes inside another does not count (found by differential testing against an RTX 3060;
  `e2e_card_edges_last_error`).
- Runtime: a copy kind that is none of the five is `cudaErrorInvalidMemcpyDirection` (21) from `cudaMemcpy`, `cudaMemcpyAsync`, the 2D
  copies and `cudaMemcpy3D`, as on an RTX 3060, not `cudaErrorInvalidValue` (`e2e_card_edges_copy_kind`).
- Runtime: 2D pitches are checked as an RTX 3060 checks them. `cudaMemset2D` of more than one row with a pitch narrower than the row is
  `cudaErrorInvalidValue` and writes nothing; it used to fill the overlapping rows and succeed. `cudaMemcpy2D` of a single row accepts a
  pitch of 0 (it was `cudaErrorInvalidPitchValue`), and a set pitch narrower than the row is `cudaErrorInvalidValue` there
  (`e2e_card_edges_pitch`).
- Runtime: `cudaEventCreateWithFlags`, `cudaHostAlloc` and `cudaMallocManaged` refuse flags the card refuses (an unknown bit;
  Interprocess without DisableTiming; managed memory with 0 or both attach flags) with `cudaErrorInvalidValue` and create nothing.
  They took any value (`e2e_card_edges_alloc_flags`).
- Runtime: once a stream capture is invalidated, launches, `cudaMemcpyAsync`, `cudaMemsetAsync`, `cudaEventRecord` and
  `cudaStreamWaitEvent` on its streams fail with `cudaErrorStreamCaptureInvalidated` until `cudaStreamEndCapture`, as on an RTX 3060;
  they were recorded and succeeded. `cudaEventSynchronize` and `cudaEventQuery` of a captured event also invalidate a Relaxed-mode
  capture, and answer `cudaErrorInvalidValue` once it is invalidated; `cudaStreamEndCapture` from a stream that joined the capture
  invalidates it. `e2e_capture_streams` no longer continues a capture after those two calls (`e2e_card_edges_capture`).
- Runtime: `cudaPointerGetAttributes` of an address in the device range that no live allocation covers (a freed pointer, one past
  the end) answers success and `cudaMemoryTypeUnregistered` with device -2, as an RTX 3060 does for any address CUDA does not know. It
  was `cudaErrorInvalidValue`, which also became the last error (`e2e_runtime_conformance`).
- cuBLAS: `cublasSetPointerMode` refuses a mode other than HOST and DEVICE, and `cublasSetMathMode` a value above 31, keeping the old
  setting; both were stored, and an unknown pointer mode made later calls read host scalars as device pointers.
  `cublasGetStatusName`/`cublasGetStatusString` give the card's names and sentences (ARCH_MISMATCH, MAPPING_ERROR, EXECUTION_FAILED
  and LICENSE_ERROR were all "CUBLAS_STATUS_INTERNAL_ERROR", and the string was the name). An empty `cublasSgemm` may pass a null C,
  and `cublasSetMatrix`/`cublasGetMatrix` with a short leading dimension no longer leave a runtime last error
  (`e2e_card_edges_cublas`).
- cuSOLVER: `getrs` refuses an unknown operation, negative sizes and short leading dimensions; it solved with them, and `n = -1`
  aborted the program with `std::length_error`. `potrf`, `potrs`, `syevd` and `gesvd` refuse unknown fill modes and jobs, negative
  sizes and short leading dimensions with LAPACK's info (minus the argument's position), `getrf` with `lda < m` succeeds with info -4,
  and `gesvd` with `m < n` is `CUSOLVER_STATUS_INVALID_VALUE` with info 0 (it was NOT_SUPPORTED). `gesvd` with `jobu = 'O'` or
  `jobvt = 'O'` works, writing the vectors over A; it was refused. `cusolverDnSetAdvOptions` refuses unknown functions and algorithms
  (`e2e_card_edges_cusolver`).
- cuSPARSE: `cusparseCreateCsr/Csc/Coo` refuse an unknown index base and more entries than the matrix has places,
  `cusparseCreateDnMat` an unknown order and a leading dimension below a column (COL) or row (ROW), and `cusparseSpMV` an unknown
  operation, a null alpha or beta, a non-CSR algorithm for a CSR matrix, and a compute type the card does not offer for the data
  (`CUSPARSE_STATUS_NOT_SUPPORTED`, also from `cusparseSpMV_bufferSize`). The `cusparseSetMat*` setters refuse values their enums do
  not have, destroying a null handle or descriptor succeeds, and `cusparseGetErrorName`/`String` give the card's texts
  (`e2e_card_edges_cusparse`).
- cuBLAS: `?gemv` takes what an RTX 3060's cuBLAS 13.0 takes. A negative `m` or `n` is an empty product and a zero
  increment is a stride of zero (`incx = 0` reads `x[0]` for every element; with `incy = 0`, `y[0]` keeps row 0's result):
  the simulator refused them with `CUBLAS_STATUS_INVALID_VALUE`, leaving `y` as it was where the card computes it. An
  operation other than N is a transpose (`CUBLAS_OP_CONJG` is `conj(A)` for `cgemv`/`zgemv` and `A` for the real types; it
  was taken as `A^H`), and `cgemv`/`zgemv` refuse an `lda` below `m`, which they read past. The other level-2 routines
  and the GEMMs (`?gemm`, `cublasHgemm`, `?gemm3m`) print reference BLAS's line for the first bad argument on stdout, as
  the card does (" ** On entry to SGER   parameter number 5 had an illegal value"); `?symv`/`?hemv` refuse silently, and
  `?gemv` prints nothing (found by differential testing; `e2e_card_edges_blas2`).
- cuBLAS: the level-3 routines and the extensions print the card's line for their first bad argument on stdout too:
  `?symm`, `?hemm`, `?syrk`, `?herk`, `?syr2k`, `?her2k`, `?syrkx`, `?herkx`, `?trsm`, `?trmm`, `?geam` (and `_64`),
  `?dgmm`, `?gemmBatched`, `?gemmStridedBatched`, `?trsmBatched`, `cublasHgemm(Strided)Batched`,
  `cublasCgemm3m(Strided)Batched`, `cublasGemmEx`, `cublasGemmBatchedEx`, `cublasGemmStridedBatchedEx` (and `_64`),
  `cublasSgemmEx`, `cublasCgemmEx`, `cublasCgemm3mEx` and the Ex herk/syrk forms, each with the field and position an
  RTX 3060's cuBLAS 13.0 prints (" ** On entry to SSYMM  parameter number 7 had an illegal value", "SBATCHED TRSM ",
  "GEMM_EX " with GemmEx's own positions). `cublasCgemm3mBatched` printed `CgemmBatched`'s field. `cublasGemmEx` and its
  batched forms refuse an algorithm outside `CUBLAS_GEMM_DEFAULT`..`ALGO23` and `DEFAULT_TENSOR_OP`..`ALGO15_TENSOR_OP`,
  as the card does; they ran it (found by differential testing; `e2e_card_edges_blas3`).
- Runtime: `cudaMemcpy3D` and `cudaMemcpy3DAsync` between pitched pointers check pitches and `ysize` as an RTX 3060 does. One
  row of one slice accepts a pitch of 0 (it was `cudaErrorInvalidPitchValue`) and refuses a set pitch narrower than the
  row with `cudaErrorInvalidValue`, the 3D form of the one-row `cudaMemcpy2D` fix; more than one slice needs a `ysize` of
  at least the rows (0 included: the simulator took 0 for the rows and copied); a set `ysize` that does not hold the rows
  from `pos.y` is `cudaErrorInvalidValue` (it copied overlapping slices). An extent with a side of 0 copies nothing; the
  simulator copied one row or one slice for a height or depth of 0 (`e2e_card_edges_pitch`).
- cuSOLVER: `potrf` (`cusolverDn?potrf`, `cusolverDnXpotrf`) of a matrix that is not positive definite leaves A as an
  RTX 3060 does: the columns before the failing one factored, NaN on the failing diagonal, and then, from the lower
  triangle, the rest factored with that column taken as 0; from the upper, NaN in every row from the failing one. The
  simulator left A untouched (measured up to n = 32; the card factors larger matrices in blocks, which changes the part
  after the failing column). `getrf` without pivoting multiplies the column below a zero pivot by 0 and goes on, as
  the card does; it skipped the column, leaving it as it was (`e2e_card_edges_cusolver`, `test_host_dense`).
- cuSPARSE: `cusparseSpMV_bufferSize` asks for the workspace the card asks for (11 bytes and up for a CSR matrix with
  entries) and `cusparseSpMV` refuses a null buffer then; it asked for 0 and took null. A null handle, matrix or vector
  is `CUSPARSE_STATUS_INVALID_VALUE` from both (it was NOT_INITIALIZED, or success for the size).
  `cusparseCsr2cscEx2`/`_bufferSize` take `CUSPARSE_CSR2CSC_ALG1` only and refuse a null handle, the size is never 0,
  and a null buffer is refused (`e2e_card_edges_cusparse`).
- cuSPARSE: `cusparseSpMV_bufferSize` asks for the workspace an RTX 3060's cuSPARSE 13.0 asks for with every format,
  algorithm, operation and index type for single, double and complex values (and half and bfloat16 with `CSR_ALG2`
  and `COO_ALG2`); every one was given the single-precision CSR size, where the card asks 515 bytes and up for
  `CSR_ALG2` (CSC's for A^T), 524 and up for COO and nothing for COO's A^T, and steps at other entry counts for double
  (2048) and complex values. With COO and A^T, `cusparseSpMV` takes a null buffer. `cusparseSpMV` and its
  `_bufferSize` print the card's line on stderr for a refused argument (" ** On entry to cusparseSpMV() parameter
  number 10 (externalBuffer) had an illegal value: NULL pointer", dimension mismatches, a conjugate transpose of
  real data, an unsupported type combination, an algorithm of another format), unless `VGPU_QUIET=1`; the size
  refuses a null alpha or beta, an unknown operation, x or y of the wrong size and an algorithm of another format
  (CSR's for COO, COO's for CSC), which it took, and an unknown compute type is `CUSPARSE_STATUS_INVALID_VALUE` from
  both, as on the card (it was NOT_SUPPORTED) (found by differential testing; `e2e_card_edges_spmv`).
- Driver: `cuCtxSetCurrent` takes a destroyed context, as an RTX 3060 does (it was `CUDA_ERROR_INVALID_CONTEXT`), and every
  call that needs a context answers `CUDA_ERROR_CONTEXT_IS_DESTROYED` while the current one is dead: destroyed, a primary
  context released to nothing, or reset. `cudaDeviceReset` now resets the driver's primary context as
  `cuDevicePrimaryCtxReset` does, so the driver's calls are `CUDA_ERROR_CONTEXT_IS_DESTROYED` until the runtime's next
  call that uses the device; they succeeded. That call, after `cuCtxSetCurrent(NULL)` or destroying the program's own
  context, makes the primary context current again, as on the card (the thread was left with none). The runtime's
  queries (`cudaGetDeviceCount`, `cudaGetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`,
  `cudaPointerGetAttributes`) no longer make the primary context current or active, as on the card
  (`e2e_card_edges_context`).
- Driver and runtime: with a destroyed context current, `cuCtxGetDevice` names its device as an RTX 3060 (CUDA 13.0) does
  (it was `CUDA_ERROR_INVALID_CONTEXT`), and the runtime's calls that use the device (`cudaFree(0)`, `cudaMalloc`,
  `cudaDeviceSynchronize`, `cudaStreamCreate`, ...) answer `cudaErrorDeviceUninitialized` and leave it current; they made
  the primary context current and succeeded. For a dead context (destroyed, or a primary one not active) `cuCtxGetId` is
  `CUDA_ERROR_CONTEXT_IS_DESTROYED`, and `cuCtxGetApiVersion` is that for the current context and
  `CUDA_ERROR_INVALID_CONTEXT` for a named one; they succeeded, or answered INVALID_CONTEXT. `cuCtxGetFlags` answers 0 for
  a primary context that is not active (it was `CUDA_ERROR_CONTEXT_IS_DESTROYED`). `cudaInitDevice` makes the primary
  context active, current on no thread, and applies `deviceFlags` to it with `cudaInitDeviceFlagsAreValid`, refusing a
  bad value (then leaving the context current, as the card does); it did neither, and refused any other bit of `flags`,
  which the card ignores (found by differential testing; `e2e_card_edges_context`).
- Runtime: stream capture as an RTX 3060 (CUDA 13.0) does it, in every mode. A wait that would merge two captures
  (`cudaErrorStreamCaptureMerge`) invalidates both, not only the waiting stream's; the legacy stream's
  `cudaStreamWaitEvent` is `cudaErrorStreamCaptureImplicit` and invalidates the capture while a blocking stream is
  capturing (whatever the event) or when the event is a capture's; once an event's capture is invalidated, a wait on
  it from a stream outside it is `cudaErrorInvalidValue` (the stream was forked into the dead capture) and from one of
  its streams `cudaErrorStreamCaptureInvalidated`. `cudaStreamAddCallback` on a capturing stream is refused and the
  capture goes on without it; it was invalidated (found by differential testing; `e2e_card_edges_capture`,
  `e2e_capture_streams`).
- AMD: the SDWA, DPP and VOP3P modifiers and the denormal and IEEE modes around them, audited against the guides and an RX 6800
  (follow-up to pantheonsim #613). The output modifier (`mul:2`, `mul:4`, `div:2`) did nothing on gfx942 and gfx950 only with IEEE mode
  or output denormals on; it now does nothing then on gfx90a, gfx1030 and gfx1100 too (CDNA2 to CDNA4 guides, 6.5, the RDNA3 guide,
  7.2.3.1, and the card, which
  a code object's defaults always hit), and applied, a result it makes denormal is flushed keeping its sign while a zero becomes +0;
  the tier leaves an output modifier to the interpreter there. `v_div_scale_f32` and `v_div_scale_f64` wrote their result unclamped
  and unscaled: the card clamps it (a NaN to 0) and scales it like any VOP3 float result (every target, gfx1250's too). neg_lo and neg_hi on the packed integer
  instructions (`v_pk_add_u16` and the rest) were ignored by the interpreter and the tier; they flip the half's top bit, as the card
  does. gfx90a and gfx1030 take the guides' rule for `v_min`/`v_max` (f16, f32, f64, packed): -0 is below +0, and with IEEE mode a
  signalling NaN gives a NaN, without it the other operand (the CDNA2 guide's pseudocode, and the card); they took the host's fmin and
  fmax; there a denormal operand is a zero where MODE flushes input
  denormals (the result is an operand, which the host's DAZ left as it was). A half's denormals are flushed where `float_denorm_mode_16_64`
  says (the card does; they were always kept). gfx1030's SDWA comparisons (`v_cmp_*_sdwa`) were refused as not implemented. gfx11's
  VOP3 DPP with op_sel read the first source from its own lane, not the shuffled one. On gfx1030 `v_dot2_f32_f16` flushes a denormal
  addend whatever MODE says and keeps denormal halves, as the card does. New test `test_amd_gcn_mods`
  (`amd/tests/data/asm_mods.s`: 50 SDWA and DPP, 36 VOP3P and VOP3 and 24 gfx11 VOP3 DPP instructions over 1024 edge inputs, in three modes,
  gfx90a, gfx942, gfx1030 and gfx1100, tier off and on; the gfx1030 kernels run on the card gave exactly what the test expects).
  test_amd_gcn_same_bits: 96 lines re-recorded, the ones these change (its header lists them).
- AMD: a 16-bit vector result on gfx1030 now keeps the destination register's high half, as an RX 6800 and an RX 6700 XT do for every
  16-bit VOP1, VOP2 and VOP3 result (e32, e64, DPP, DPP8; follow-up to pantheonsim #613); the executor zeroed it as gfx9 does.
  `v_sat_pk_u8_i16` still zeroes it, as the cards and the guide's `{16'b0, ...}` say. gfx90a's `v_mad_f16`, `v_mad_u16`, `v_mad_i16`,
  `v_fma_f16` and `v_div_fixup_f16` keep it too (the CDNA2 guide: "hi 16 bits are preserved"), as gfx942 and gfx950 already did;
  their `_legacy` forms still zero it. gfx1030 ran none of `v_frexp_mant_f16`, `v_frexp_exp_i16_f16`, `v_sin_f16`, `v_cos_f16`,
  `v_ldexp_f16`, `v_sat_pk_u8_i16`, `v_cvt_norm_i16_f16` and `v_cvt_norm_u16_f16` ("decoded but not implemented"); they run now, and
  give the cards' bits (the sine and cosine of an infinity are 0xfe00 there). New test `test_amd_gcn_hi16` (`amd/tests/data/asm_hi16.s`,
  75 instruction slots, gfx90a, gfx942 and gfx1030, tier off and on; the gfx1030 one against the cards' dwords).
- AMD: gfx1030's NaNs are now the card's (an RX 6800 and an RX 6700 XT; follow-up to pantheonsim #613). A NaN result is the first NaN
  source in operand order after its abs/neg (so `v_sub` and `v_subrev` flip the sign of the NaN they pass on), quieted with IEEE mode
  on and passed through as it is (a signalling NaN too) with IEEE mode off; with no NaN source (inf - inf, 0 * inf, sqrt(-1),
  fract(inf)) it is the negative default NaN, 0xfe00 or 0xffc00000; conversions carry the payload over; `v_fma`'s 0 * inf beats a NaN
  addend; `v_min`/`v_max` with IEEE mode let a signalling NaN win; `v_min3`, `v_max3` and `v_med3` follow the guides' pseudocode over
  those (with -0 below +0). The executor took NaNs from the host's arithmetic, in whichever order kept the old digests. `v_div_fixup`
  takes S2's NaN, then S1's, and makes an infinity of a NaN quotient, as the card does. `v_sin_f32` of a whole number of turns is +0.
  `v_div_fixup_f16` gave the quotient's magnitude for 0/y on every target; it gives the signed zero of the guides' pseudocode, as the
  card does. The tier hands gfx1030's NaN results, its `v_div_fixup_f32` and its `v_sin_f32` to the interpreter. New test
  `test_amd_gcn_nan` (`amd/tests/data/asm_nan.s`: 65 f16 and f32 instructions over pairs of 11 special values, IEEE mode on and off,
  every dword against the cards'). test_amd_gcn_same_bits: 31 lines re-recorded (its header lists them).
- AMD: gfx1030's NaN rule now covers the forms the first measurement left out (an RX 6800 and an RX 6700 XT, which agree; follow-up to
  pantheonsim #613): the SDWA, DPP and DPP8 forms, `v_fmamk`/`v_fmaak` (f16, f32), the legacy `v_fma_legacy_f32`/`v_fmac_legacy_f32`,
  the packed `v_pk_add/mul/fma/min/max_f16` (per half), the doubles (`v_add/mul/fma/min/max/div_fixup/ldexp/frexp_mant_f64`, the
  rounding and reciprocal instructions, `v_cvt_f32_f64`, `v_cvt_f64_f32`) and `v_dot2_f32_f16`/`v_dot2c_f32_f16` (the first NaN of
  a.lo, b.lo, a.hi, b.hi, c). A narrowing conversion quiets a signalling NaN with IEEE mode off too. `v_med3`/`v_min3`/`v_max3` with
  a denormal source where MODE flushes input denormals take it as the zero of its sign (the executor kept the denormal). gfx1030's
  `v_dot2_f32_f16` adds the two products before the addend, as its `v_dot2c_f32_f16` already did (1 * d + 1 * -d + c is c on the
  card; adding c first lost it). gfx1030 refused `v_mul_legacy_f32_sdwa` (its executor name came out as "v_mul_legac_e32"),
  `v_fma_legacy_f32` and `v_fmac_legacy_f32`; they run now (a zero factor makes the product +0, the cards' 0 * NaN + -0 is +0).
  `v_cvt_f32_f64_e64` ignored its source's abs and neg on every target; it applies them (the card does; the guides' VOP3 NEG and ABS
  fields act on every float source). The tier hands gfx1030's double NaNs back to the interpreter and leaves its float 64-bit
  conversions, `v_div_fixup_f64` and its packed f16 arithmetic to it. Not modelled: gfx1030's `v_sqrt_f64` of a positive number is
  an approximation on the card (sqrt(1) is 0x3ff0000004000000). New test `test_amd_gcn_nan2` (`amd/tests/data/asm_nan2.s`: 117
  slots over pairs of 12 special values, IEEE mode on, off, and on with denormals flushed; every result, the high half of the 16-bit
  ones included, against the cards'). test_amd_gcn_same_bits: 19 lines re-recorded (its header lists them).
- AMD: gfx1100 refused eleven instructions as "decoded but not implemented": `v_frexp_mant_f16`, `v_frexp_exp_i16_f16`, `v_sin_f16`,
  `v_cos_f16`, `v_ldexp_f16`, `v_sat_pk_u8_i16`, `v_cvt_norm_i16_f16`, `v_cvt_norm_u16_f16`, and the VOP3 `v_ashrrev_i16`, `v_max_i16`
  and `v_min_i16`. They run now, through the code gfx1030 uses (AMD's machine-readable RDNA3 ISA describes each in the same words as its
  RDNA2 one; no RDNA3 card was run). `test_amd_gcn_hi16` runs them on gfx1100 against the gfx1030 cards' low halves and checks that
  every gfx1100 16-bit result keeps the other half of its register. test_amd_gcn_same_bits: 7 gfx1100 lines re-recorded.
  `amd/tests/hipcc/rdna3.cpp` expected `v_add_f32_e64 ... div:2` to halve in a default-mode (IEEE on) hipcc kernel; it now expects the
  modifier ignored (`rdna3.gfx1100` and `rdna3.w64.gfx1100` rebuilt: host code only, the device code is the same).
- AMD: the clamp bit of the integer instructions saturates their result on every target, as the CDNA3 and CDNA4 guides say (6.5, and
  "Supports saturation" on each add and subtract) and as an RX 6800 does. The VOP3b adds and subtracts with a carry out (`v_add_co_u32`,
  `v_sub_co_u32`, `v_subrev_co_u32`, `v_addc_co_u32`, `v_subb_co_u32`, `v_subbrev_co_u32`, RDNA's `_co_ci_` forms) and `v_mad_u64_u32` /
  `v_mad_i64_i32` ignored it (gfx9's decoder did not read the bit, and neither the interpreter nor the tier applied it), so
  `v_add_co_u32_e64 ... clamp`, LLVM's unsigned saturating add on gfx8, wrapped; clamped, a sum past the type's range is its maximum, a
  difference below zero is zero, and the carry out stays the unclamped one's. `v_mad_u32_u24`, `v_mad_i32_i24`, `v_mul_u32_u24`,
  `v_mul_i32_i24`, `v_sad_u8`, `v_sad_hi_u8`, `v_sad_u16`, `v_msad_u8`, `v_mul_lo_u16` and `v_mad_legacy_u16` ignored it too, and
  `v_subrev_u32`, `v_subrev_u16`, `v_mad_u16`, `v_mad_i16`, `v_mad_u32_u16` and `v_mad_i32_i16` applied it on gfx942 and gfx950 only.
  `v_mad_i64_i32`'s scalar bit is bit 64 of the 65-bit result (the guides' pseudocode, and the card's): the sign of the exact result, not
  whether it overflowed int64, which it was. New test `test_amd_gcn_int_clamp` (gfx90a, gfx942 and gfx1030, tier off and on; its gfx1030
  kernel run on the card gave what the test expects) (follow-up to pantheonsim #613).
- SASS: a lane group the simulator lets another group of its warp past in the middle of a loop (the turn a group gives after many
  backward branches while other lanes wait, which is the simulator's own fairness and not a YIELD) gets its uniform registers and
  predicates back as it left them when it runs again. ptxas keeps a counted loop's index on one side of a divergent branch in a uniform
  register that the other side reuses, because the hardware runs one side to its BSYNC or EXIT unless it meets a YIELD; the other side's
  write landed in the loop. pravirkr/loki's sm_90 EP scorer (`kernel_ep_shift_add_score_c2c`) walked UR4 through a row of shared memory
  while the lanes of invalid leaves loaded a pointer into UR4, and after the 64th trip stopped with "a shared address (0x8ded482c) naming
  block 141 of a cluster of 1"; the PTX path and the card were right. `SR_CgaCtaId` was not involved: it reads 0 in a launch without
  clusters. New unit test `test_sass_uniform_switch`, SASS and PTX (pantheonsim #642).
- Runtime: a copy, fill or kernel launch issued on a stream behind a host function that has not finished returns at once and runs
  after it, in stream order, as on an RTX 3060. It used to run inside the call after waiting for the host function, so a host function
  that waits for something the program does after the call (Celeritas' `AsyncSnapshotTest.device_pending`) hung the program.
  `cudaMemcpyAsync`, `cudaMemsetAsync`, the 2D and 3D copies and fills between pointers, the symbol copies, `cudaMemcpyPeerAsync`,
  `cudaFreeAsync` and kernel launches on any stream the runtime made are queued on the stream's lane when the memory they touch is
  device or page-locked memory; they see what the host function wrote, and events, stream queries and every synchronizing call wait for
  them. Copies of pageable memory, managed memory and arrays still wait in the call. `cudaFree`, `cudaFreeHost`, `cudaHostUnregister`
  and `cudaDeviceReset` wait for queued work first. The 2D, 3D and symbol copies on a stream also follow host functions on that stream
  now (on a non-blocking stream they ran before them), and `cudaMemcpyPeer` and the synchronous symbol copies follow host functions on
  the legacy default stream. New e2e test `e2e_async_host_copies`; [design](docs/perf-design/streams.md) (pantheonsim #634).
- `nvidia-smi nvlink` takes the options of the real subcommand, not only `-s` and `-e`: `-c`/`--capabilities`, `-cBridge`/`--checkBridge`,
  `-gt d|r`, `-i` and `-l` after `nvlink`, several displays at once, and `-h` or no option for nvidia-smi's own usage text. A GPU without NVLink
  prints its name and exit 0 for every display, as the real tool does; on the SXM profiles `-c` prints each link's capabilities as
  `nvmlDeviceGetNvLinkCapability` reports them, `-gt` zero traffic, and `-cBridge` no link lines (the links run through the baseboard, not a
  bridge). The remote-end and setting displays (`-p`, `-R`, `-ec`, `-gLowPwrInfo`, `-gBwMode`, `-gLWidth`, `-info`) are refused on a GPU with
  links, the changes (`-re`, `-sBwMode`, ...) everywhere, and an unknown option prints `Option "X" is not recognized.` and exits 2 as nvidia-smi
  does. jpmorganchase/QOKit runs `nvidia-smi nvlink -cBridge` with `check=True` when its tests are collected (pantheonsim #640).
- AMD: a load into accumulation registers that the wave had not used yet, made for the whole wave at once (`global_load_dwordx2 a[38:39], v2,
  s[6:7] offset:200`, every lane at one address, as a Kokkos kernel loads a pointer out of a structure), fills every register it names. The
  register file grows to the highest accumulation register named, and growing it for the second register moved the first one's row, which
  then went to the freed copy: a38 kept what it held before, and the next load through the pointer stopped with "invalid-pointer ...
  address is not inside any device allocation" at a small address (llnl/exadis on the simulated MI300X, pantheonsim #621). The file now grows
  to the last register before any row is taken (the whole-wave loads of two, three and four dwords and the whole-wave store of three or
  four). It surfaced with host-mapped memory taking the whole-wave path (#507): exadis's data is managed memory, which went lane by lane before.
- cuRAND: `curandGeneratePoisson`, `curandCreatePoissonDistribution`, `curandDestroyDistribution`,
  `curandGeneratePoissonMethod`, `curandGenerateSeeds` and `curandGetProperty` are exported. A program that called
  one stopped with "undefined symbol" (RubixML/numpower's `NumPower::poisson`; pantheonsim #629). The methods are
  the ones an RTX 3060's cuRAND uses for each generator and lambda: Marsaglia-Tsang-Wang's square histogram, built
  with the card's aliases, so `curand_discrete` in a kernel and a Sobol' generator's Poisson integers are the card's;
  the inversion and the rounded normal above the card's cut-offs ([libraries](nvidia/docs/libraries.md); `e2e_rand_poisson`).
- AMD: loading a library's code objects takes 40 % less time. The 62 Tensile code objects of a small rocBLAS program (`amd/tests/rocblas/blas`,
  sgemm, dgemm and the rest, ROCm 7.1 rocBLAS on a simulated MI300X) loaded in 0.44 s instead of 0.75 s and the program ran in 1.61 s
  instead of 1.96 s (median of 8 alternating runs on an ubuntu-24.04 runner; best 1.53 s against 1.84 s). The time was page faults
  and copies, not arithmetic: a kept bundle entry is inflated straight into its place instead of through a buffer that is then
  copied, the zstd buffer is kept from one load to the next, the code object is read where it lies instead of from a copy, `.text`
  and a linked image are written once instead of zeroed first, and the buffers of tens of megabytes are asked for in 2 MB pages
  (`madvise(MADV_HUGEPAGE)`); the process took 190,000 page faults instead of 350,000. The bytes read are the same:
  `test_amd_codeobject_same_bits` gives the recorded digests, and the 2,838 code objects ROCm 7.1 ships (rocBLAS's and hipBLASLt's
  libraries) give the same digests before and after. `VGPU_LOAD_PLAIN=1` reads them as before (`test_amd_codeobject_same_bits_plain`);
  `test_amd_bundle` adds bundles larger than the buffer, read both ways. AMD's `rocblas-test` gains nothing from it: its start-up
  (57 s a process, reading `rocblas_gtest.data`, against 0.1 to 0.4 s of loading) is AMD's client, not the simulator (pantheonsim #565).
- NVRTC without the toolkit's libnvrtc (the shim's nvcc path, or `VGPU_NVRTC=nvcc`) now searches includes as NVRTC does: no host
  directory, the toolkit headers in `__CUDACC_RTC__` mode, the headers given to `nvrtcCreateProgram` and the `-I`/`--include-path`
  directories. A program that brings its own `<cstdint>` (ThrustRTC 0.3.20) or defines `int64_t` itself (PyTorch's jiterator) compiles,
  where nvcc's host headers made it fail with "invalid redeclaration of type name int64_t"; `#include <stdint.h>` and the other host headers
  fail to open, as with NVRTC. Where the toolkit's headers share `/usr/include` with glibc's (Debian's and Ubuntu's
  `nvidia-cuda-toolkit`), the compile sees links to the CUDA headers alone. What NVRTC predefines (`size_t`, `ptrdiff_t`, `clock_t`, `NULL`, `std::move`, `std::forward`,
  `std::initializer_list`, `printf`, `assert`) is there without an include, measured on CUDA 13.0's NVRTC; the PTX matches NVRTC's for the
  programs compared. ThrustRTC's fill and reduce run under `vgpu run` on `nvidia/a100` (ThrustRTC 0.3.20's own occupancy calculator does not
  know compute capability 9.x, so its kernels fail on an H100, real or simulated). The nvcc path also no longer leaves an empty
  `$TMPDIR/vgpu-nvrtc-<pid>` directory behind per process. New test `e2e_nvrtc_paths_nvcc` (pantheonsim #619).
- AMD: `hipModuleGetFunction` no longer prints "the code object has no kernel named ..." when another loaded module has the
  kernel. rocBLAS and hipBLASLt (Tensile) find a GEMM kernel by asking each loaded code object in turn and clear the
  `hipErrorNotFound` of the misses, so every new GEMM kernel printed the message up to once per earlier module, which read as a
  failed GEMM in runs whose GEMMs were right (double and complex-double GEMMs through rocBLAS 7.2 on `amd/mi300x` checked against a
  host reference). The error code and `hipGetLastError` are unchanged, and a kernel that no loaded module has is still reported (#636).
- Driver: `cuGetProcAddress` / `cuGetProcAddress_v2`, and with them `cudaGetDriverEntryPoint` and `...ByVersion`, resolve a
  function's base name by the CUDA version asked for, as an RTX 3060 does: `cuGraphExecUpdate` at 12000 or later is
  `cuGraphExecUpdate_v2` (it was not found, so cuda-core 1.3.0 refused the driver and every numba-cuda launch failed),
  `cuCtxCreate` is `_v3` at 12000 and `_v4` at 13000, `cuStreamGetCaptureInfo` is `_v3` at 13000 (it was `_v2`), and so on for every
  name. The table comes from NVIDIA's typedef headers (`PFN_<name>_v<version>`; `scripts/gen_driver_proc_versions.py`,
  `nvidia/src/driver_proc_versions.inc`), with the per-thread-stream forms for `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM`. As on
  the card, a name asked for at a version older than the function is `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`, and a name that
  is not found returns `CUDA_SUCCESS` with a null function and the status (it returned `CUDA_ERROR_NOT_FOUND`). The card maps
  `cuGraphInstantiate` to `cuGraphInstantiate_v2` at every version from 11.0, not to `cuGraphInstantiateWithFlags`, and so does the
  shim. New unit test `test_driver_proc_address` compares all 3,976 answers the card's driver gave (497 names at 11040, 12000,
  12040 and 13000, both stream modes; `tools/probes/driver_proc_address_probe.cpp`) with the shim's; `lint_driver_proc_versions`
  checks the table against the toolkit at hand (pantheonsim #626).
- Driver: `cuProfilerInitialize` is exported (deprecated, still in the CUDA 13 driver; cudarc 0.19 binds it when it loads libcuda and
  panicked without it). It answers `CUDA_ERROR_NOT_INITIALIZED` before `cuInit` and `CUDA_ERROR_NOT_SUPPORTED` after, whatever the
  arguments and with or without a context, as an RTX 3060 does. Of the other names the CUDA 13.0 driver stub exports, the shim
  now also exports the per-thread forms of the CUDA 11 stream memory-operation and capture calls (`cuStreamWaitValue32_ptsz`,
  `cuStreamBatchMemOp_ptsz`, `cuStreamBeginCapture_ptsz`, `cuStreamGetCaptureInfo_ptsz` and the rest, which the card hands a CUDA
  11 caller that asks for the per-thread default stream) and the unsuffixed `cuMemGetAddressRange`; only the debugger's `cudbg*`
  entry points remain unexported (pantheonsim #627).
- Host-GPU mode (experimental, opt-in), both `exact` and `turbo`: a launch on fewer than 256 blocks, which the sample of whole blocks could not
  judge, is now sampled *inside* its blocks when its kernel has a counted loop: the forked copy runs each block until its warps stand in the
  loop, reads how far each lane's induction register is from the bound its compare reads, and times one turn of the loop on one warp; the
  launch's simulator time is the turns left at that cost. Measured on one A10G (Round sp61): PolyBench COVAR 64.7 s to 41.8 s and CORR 64.8 s to
  42.0 s in both modes (their long kernel, 24.5 s in the simulator, takes 1.67 s on the card path), outputs identical. Grids of 256 blocks or more
  are sampled as before; `VGPU_HOST_GPU_SAMPLE=whole` keeps the rule of Round sp60b (design notes, since removed, section 8).
- Host-GPU mode (experimental, opt-in), both `exact` and `turbo`: a launch of a kernel with a loop that the mode knows nothing of yet is now *sampled*
  before the simulator is left to run it. The process is forked, the copy runs a few of the launch's blocks (up to 32, a few milliseconds), reports
  their times and ends; if they say that the launch repays the card path and the start of the real driver, the card runs the launch, otherwise
  the simulator does. Before, such a launch went to the card only on a static count of the kernel's instructions, which is low by the trip
  count of its loops, so programs that launch each kernel once kept their one long launch in the simulator. `VGPU_HOST_GPU_SAMPLE=0` switches
  the sampling off; `=log` reports what it would do with no card. Measured on one A10G (Round sp60b): PolyBench SYRK 5.16 s to 1.30 s, SYR2K
  9.81 s to 2.37 s, 2MM 172 s to 122 s in both modes, HeCBench black-scholes 25.1 s to 6.1 s in turbo; geometric mean over 61 programs from 1.12x
  to 1.18x (`exact`) and from 1.14x to 1.27x (`turbo`) against the simulator, outputs identical. Kernels launched on a few large blocks (PolyBench
  COVAR, CORR) are not reached (design notes, since removed, section 8; pantheonsim #587).
- Host-GPU mode (experimental, opt-in): `VGPU_HOST_GPU=turbo` also runs on the machine's real GPU the kernels whose result is the
  card's own and not the simulator's bits: the approximate functions (`MUFU`), packed half arithmetic, float reductions, calls inside the
  kernel, tensor-core instructions and integer atomics that return a value; float atomics with a returned value, compare-and-swap and
  exchange only with `VGPU_HOST_GPU_ATOMICS=1`; textures and unnamed instructions never. `VGPU_HOST_GPU_TURBO_OFF` switches classes
  off. With `VGPU_HOST_GPU_VERIFY` the card's memory is compared with the simulator's by tolerance (`VGPU_HOST_GPU_ATOL`,
  `VGPU_HOST_GPU_RTOL`, `VGPU_HOST_GPU_TOL_TYPE`) instead of byte for byte, and the exit summary has lines for scripts (`host_gpu_turbo:`, `host_gpu_refusals:`,
  `host_gpu_outside:`). A launch that does not come back from the card ends the process after `VGPU_HOST_GPU_TIMEOUT` seconds (600 in
  turbo mode). `exact` mode is unchanged, except for two bugs: a kernel refused for its module is no longer reported as "not in the
  exact set" from its second launch on, and a float reduction (`RED.ADD.F32`, which adds in arrival order) is now refused on sm_75 to
  sm_89 as it was meant to be (the check did not recognise the decoder's spelling of the type). Measured on one A10G (Round sp59): geometric mean 1.23x over 59 programs
  of PolyBench/GPU, Rodinia and HeCBench against 1.12x for `exact`, outputs identical. `VGPU_HOST_GPU_VERIFY` now gives the card copy up before the
  simulator runs the launch, in both modes: the simulator's atomics could lose updates under verify, and the card ran on its own earlier results.
  Design and limits (notes since removed); the benchmark harness is
  `tools/bench/hostgpu/run.py`, the card probes `tools/probes/hostgpu_turbo_card_probes.sh`. `docs/cli.md` now lists every
  `VGPU_HOST_GPU_*` variable (pantheonsim #587).
- Tier 2 (`VGPU_JIT=1`, still off by default), the cold run ([round sp63](docs/perf-design/rounds/11-rounds-sp36-sp41.md)): a section the
  share rule keeps waits for its cache lookup on the plain run loop too, so short programs no longer pay for the counting loop before
  the lookup (ADI, asmooth, sortKV, atan2, adam lost 1 to 9 % with `VGPU_JIT=1`; now within 0 to 2 %). The regions chosen for a section
  are recorded before its compile, so a compile that the end of the program cancels (silently now) is started at the next run's lookup
  instead of at the threshold (`VGPU_JIT_HISTORY=0` turns that off): asyncAllocation, shorter than its compile, is warm from its third run
  instead of never at one thread (at sixteen threads such compiles still never finish). The cache directory is named by the library's GNU build id, other builds' directories untouched for 14 days are
  removed, and the cache is capped (`VGPU_JIT_CACHE_MAX_MB`, default 256, least recently loaded objects first). The notice about a missing
  compiler or an unusable cache is given at most once a day per cache directory, and `VGPU_JIT=auto` turns the JIT on without those notices.
  Results and counters are unchanged.
- Kernels no longer die with SIGFPE when the host program enabled floating-point traps (`feenableexcept`, gfortran
  `-ffpe-trap=invalid,zero,overflow`, which SPECFEM3D's configure puts in `FLAGS_CHECK`). The engines did the device's arithmetic
  under the host's traps, on the launching thread and on the worker threads that copy its environment, so the first inf or NaN a
  kernel made killed the process; an RTX 3060 runs the same program with the traps on. A launch (PTX, SASS and AMD) now runs with
  the traps masked, and the host gets its whole floating-point environment back afterwards, traps, rounding mode and sticky flags,
  none raised by the kernel (`include/vgpu/exec/host_fp.hpp`). New unit tests `test_launch_fp_traps` and `test_amd_gcn_fp_traps`
  (pantheonsim #605).
- Runtime: driver functions taken from `cudaGetDriverEntryPoint` / `cudaGetDriverEntryPointByVersion` by a program that does not
  link libcuda (gputools does this) see what the runtime has done, as on an RTX 3060: the driver is initialised (they returned
  `CUDA_ERROR_NOT_INITIALIZED`), a thread the runtime has used has its primary context current (they returned
  `CUDA_ERROR_INVALID_CONTEXT` from `cuModuleLoadData`), and a thread it has not used has none. Before any runtime call,
  `cudaGetDriverEntryPoint` initialises the driver and makes no context current, as on the card. New e2e test
  `e2e_driver_entry_late` (pantheonsim #604).
- PTX: `ld.param::entry`, `ld.param::func` and `st.param::func`, the explicit `.param` state spaces of PTX ISA 8.3, are accepted as
  `ld.param` / `st.param`; jax 0.12's kernels failed to load ("unrecognized ld/st modifier '.param::entry'"). `st.param::entry`
  stays refused, as ptxas refuses it. Unit test `ld_param_entry_and_func_state_spaces` in `test_exec3` (pantheonsim #589).
- Runtime: `cudaDestroyTextureObject` and `cudaDestroySurfaceObject` of a handle that is no object (0, one never made, one already
  destroyed) return `cudaSuccess` and leave the last error alone, as an RTX 3060 does; they returned `cudaErrorInvalidValue`, which
  MaCh3's teardown left in the last error for CUDAProb3's next check to exit on. Unit test in `test_runtime_exports`
  (pantheonsim #618).
- cuBLAS: a dot product (`?dot`, `?dotu`, `?dotc`, `DotEx`, `DotcEx`, their `_64` forms) in the default host pointer mode
  with its result pointer in device memory stores the result there, as an RTX 3060 does; it died with SIGSEGV inside the
  library, which stopped every device XC integration of GauXC (its `gdot` makes this call). The other reductions (`?nrm2`,
  `?asum`, `i?amax`, `i?amin` and their Ex and `_64` forms), and a dot product with n <= 0, crash on the card with a device
  result in host mode; they now return `CUBLAS_STATUS_INVALID_VALUE`, said once on stderr, instead of crashing. Measured per
  routine on the card ([cuBLAS](nvidia/docs/cublas.md)). New unit test `test_cublas_host_mode_device_result` (pantheonsim #638).
- NPP, CUDA 12 build: `nppGetStreamContext`, `nppGetStream` and `nppSetStream` (NPP 12's own stream, which CUDA 13's NPP removed)
  are exported from the shim, filling the context as NPP 12.3 does on an RTX 3060. New e2e test `e2e_npp_stream_context` (skips
  on CUDA 13) (pantheonsim #617).
- NPP: `nppiFilterBoxBorder_8u_C1R`, `nppiGradientVectorPrewittBorder_8u16s_C1R` and `nppiFilterCannyBorder_8u_C1R` (and
  their `_Ctx` forms) read the right pixels with a nonzero `oSrcOffset`. They took `pSrc` for the image's origin, but it points at
  the pixel `oSrcOffset` names, as an RTX 3060 (NPP 13.0) shows; so a ROI inside the image read pixels `oSrcOffset` too far on,
  past the image's end. A zero offset is unchanged. The `npp_imgproc` golden (`e2e_library_goldens`) has the box filter and the
  gradients on ROIs at an offset reaching the source's right and bottom edge, measured on the card. Canny is not pinned there:
  at a nonzero offset the card's result depends on what its scratch buffer held before the call; the simulator gives the
  full image's result for the ROI, which is what the card gives at offset zero.
- NPP: `nppiFilterGaussBorder_8u_C1R_Ctx` and its plain CUDA 12 form, with every mask NPP gives it (3x3 to 15x15), bit for bit
  as NPP 13.0 and NPP 12.3 on an RTX 3060 on random, constant and impulse images. Measured: a separable single-precision filter,
  vertical then horizontal, each an fma chain over the taps in the order 1, 0, 2, ..., n-1, truncated to 8 bits (so a constant
  image can come out a count low, 3 to 2 with 3x3, which is what an earlier attempt could not explain); the taps are the card's
  (the 32f form's impulse response is exactly their products), with sigma 0.6 + 0.4 * (n/2) rather than the header's formula.
  Replicate is the only border, the source image's edge replicates (pSrc is the pixel at `oSrcOffset`), and the argument checks
  and their order are the card's: null pointer, size, offset out of range, border mode, mask size, then an empty ROI succeeds and
  a destination step below the ROI width is `NPP_STEP_ERROR`. The card's output is pinned in the `npp_imgproc` golden
  (`e2e_library_goldens`) (pantheonsim #617).
- HIP: `hipGraphInstantiateWithParams` takes `hipGraphInstantiateFlagUpload` and `hipGraphInstantiateFlagDeviceLaunch`, as ROCm's
  HIP does there (all four flags); it refused anything but AutoFreeOnLaunch and UseNodePriority with `hipErrorInvalidValue`, so
  Colvars, which instantiates its graphs with Upload and an `uploadStream`, stopped on its first frame. With Upload the graph is
  uploaded on `uploadStream` as `hipGraphUpload` does, which queues no work here (the executable graph is already the one a launch
  runs); `result_out` and `errNode_out` are set as ROCm sets them. DeviceLaunch only shows in `hipGraphExecGetFlags`, as in ROCm.
  `hipGraphInstantiateWithFlags` still refuses both, as ROCm does. New e2e test `amd_graph_instantiate_params` (pantheonsim #628).
- HIP: a failed `hipModuleLoad` is the thread's last error, as in ROCm's HIP, where every API returns through `HIP_RETURN`: a missing
  file (`hipErrorFileNotFound`) and an empty one (`hipErrorInvalidImage`) were returned but not recorded, so `hipGetLastError` then said
  `hipSuccess`. OpenMM's HIP platform reads `hipGetLastError` after a miss in its kernel cache and stopped with "Detected HIP error: no
  error" at the first kernel of every `TestHip*` program. `hipModuleGetFunction` and `hipModuleGetGlobal` now check their arguments in
  ROCm's order and with its codes: a null module is `hipErrorInvalidResourceHandle` (`hipModuleGetFunction` said `hipErrorInvalidValue`)
  and an empty kernel name is `hipErrorInvalidValue`. New e2e test `amd_module_last_error`, which also holds the other module and link
  entry points to recording their errors (pantheonsim #635).

- Peer access: after `cudaDeviceEnablePeerAccess` (or `cuCtxEnablePeerAccess`), a kernel on one simulated device reads, writes and
  atomically updates the peer device's allocations in place, as on a card (unified addressing). It stopped with `[invalid-pointer]
  ... (belongs to a different device)` and `cudaErrorIllegalAddress`, so a program that asks `cudaDeviceCanAccessPeer` and takes the
  direct route when the answer is yes failed (accl-kaust/GPU-IBS on two simulated H100s, pantheonsim #598). Without peer access, or
  after `cudaDeviceDisablePeerAccess`, the access still faults, as it does on two RTX 3060s (measured). Only an access that missed every
  allocation of the device looks at the peer; launch caching (`VGPU_MEMO=1`) and host-GPU mode stand aside while a peer link exists.
  New test `e2e_peer_kernel` (SASS and PTX paths).
- `vgpu run --preload`: PyTorch starts again, and its profiler lists what it did before. Both broke when the driver became a private load
  (pantheonsim #597, below). The runtime looked the driver up in the global scope only, so it never initialised it nor handed it its
  primary context, and PyTorch 2.10's `torch.cuda` init, which asks `cuDevicePrimaryCtxGetState` with no `cuInit` of its own, failed with
  "CUDA driver error: initialization error" on every check. The runtime now keeps the privately loaded driver's handle and asks it
  first, and `cudaGetDriverEntryPoint` no longer reloads that driver with `RTLD_GLOBAL` (#604's late-load path), which put it in the
  global scope after all. And each shim carries its own copy of the profiling engine: the unversioned driver in the global scope was
  what made every shim use one, so without it cuBLAS's internal copies were reported as the program's `cudaMemcpy`. The engine's
  functions now sit in a version node every shim shares (`VGPU_PROFILING_1`). New test `e2e_preload_torch_like`: programs that use
  the shim as PyTorch does, with no PyTorch.
- `vgpu run --preload`: `import ThrustRTC` (PyPI 0.3.20) no longer crashes with a segmentation fault (pantheonsim #597). ThrustRTC
  keeps the driver's and NVRTC's functions in exported pointers named like them (`cuInit`, `nvrtcVersion`, ...), and with the shim's
  `libcuda.so.1` preloaded its references to its own `cuInit` bound to the shim's function, whose code its `dlsym` result then
  overwrote (NVIDIA's own `libcuda.so.1` preloaded does the same). The driver's libraries and NVRTC are now loaded privately
  (`RTLD_LOCAL`) by the preloaded runtime as it starts (`VGPU_PRELOAD_PRIVATE`), so every later request for those sonames still gets
  the simulator's, and the global scope no longer holds their functions. New test `e2e_preload_fnptr`. ThrustRTC's kernels still do
  not compile through the simulator's NVRTC (its own `cstdint` meets the host's `/usr/include`).
- `e2e_sass_jit` (steps shard): a fault run whose compile did not happen (it outlasted its time limit on a loaded runner, or the
  compiler was killed) no longer fails the comparison with the tier. The JIT's notice on stderr ("pantheonsim: JIT: ... stay on the
  tier") was the only difference: the fault itself was reported the same, byte for byte, in 72 runs of the tier, the JIT and the JIT
  without exceptions on an idle and on a loaded machine (pantheonsim #611). The fault runs give a compile 840 s and take a run that
  could not compile once more; a failure now names the notice or shows the diff, and a compiler killed on its first compile checks the
  retry.
- `test_lockstep_look`: `a_block_that_stands_still_is_still_found_out_and_slept_on` no longer depends on how many turns a thread
  takes in 100 ms of wall time, which failed under load and under ThreadSanitizer (pantheonsim #612). `lockstep_worker` takes the
  nap it sleeps between passes as a callable (the engines' sleeps, unchanged); the test counts the naps: the first after the third
  look, then 5, 10, 20 ... 500 microseconds, and none for a block that computes.
- AMD: `v_div_fixup_f32`/`_f64` and `v_div_fmas_f32`/`_f64` with the VOP3 `clamp` bit or an output modifier (`mul:2`, `mul:4`,
  `div:2`) wrote the result unchanged, in the interpreter and in the tier. clang folds `fmin(fmax(a / b, 0), 1)` into the clamp of the
  division's last instruction (`v_div_fixup_f64 ... clamp`) at -O2 and up, so such code got the unclamped quotient (llnl/exadis on gfx942).
  They now clamp and scale as every other VOP3 float result does; `v_div_fixup_f16` already did. New test `test_amd_gcn_div_clamp`
  (pantheonsim #613).
- AMD: an atomic add (or any read-modify-write) of a dispatch that started while another dispatch ran alone could be lost. The lone
  dispatch decided at the start of an atomic instruction that it needed no lock, and the other's atomic landed between its read and its
  write. A dispatch now marks such an instruction before it looks at the count of running dispatches, and a dispatch that starts waits
  for that instruction to end; results of programs that run one dispatch at a time are unchanged. New test
  `a_dispatch_that_starts_while_another_runs_alone_does_not_lose_atomic_adds` in `test_amd_gcn_atomics` (pantheonsim #610).
- AMD: a kernel built with `.amdhsa_uses_dynamic_stack 1` (recursion, indirect or virtual calls) gets
  max(`hipLimitStackSize`, its static `private_segment_fixed_size`) bytes of scratch per work-item, as ROCm's HIP gives it
  (CLR `VirtualGPU::submitKernelInternal`), instead of only its static size; an HSA dispatch packet's `private_segment_size` is
  honoured where it is larger, `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK` reports the descriptor's bit, and
  `hipDeviceSetLimit(hipLimitStackSize)` rounds up to 16 bytes and refuses more than a wave's scratch offset addresses (131004 bytes on
  gfx9), as CLR does. New tests `test_amd_gcn_dyn_stack` and `test_amd_hip_dyn_stack` (pantheonsim #585).
- Tests: `e2e_group_ballot_barrier`, a cell-list neighbour build in the shape of fastMD's `nl_build_kernel` (groups of four lanes
  at different copies of a group-masked `__ballot_sync`, then a full-mask `__shfl_down_sync` and `__syncthreads`), on a simulated
  H100 as PTX and as SASS, every count and listed neighbour checked against the host; it passes on an RTX 3060. fastMD's
  `VerletList.GrowsOnOverflowAndStaysComplete` (and the rest of `test_verlet`) and `Simulation.LayerStressSumRules` pass on main
  on both paths (pantheonsim #574).
- cuBLAS: the band and packed matrix-vector routines (`tbmv`, `tpmv`, `gbmv`, `sbmv`/`hbmv`, `spmv`/`hpmv`, every type) read the
  band where it is stored and loop over it, instead of building the full n x n matrix on every call. SCS's diagonal scaling
  (`cublasDtbmv` with k = 0 on a vector of 1500) mapped 34 MiB, took 8,000 page faults and summed 2.25 million products on each
  call; with transparent huge pages at `madvise`, as GitHub's runners ship them, that was most of the run. SCS's `max_ent` solve
  on a 7.0 kernel with `madvise`: 574 s before, 5.5 s after, the same iterations; the answers are the full loops' bits (an
  infinity or NaN in x still runs the full loops, which make NaN of the zeros it meets). `tbsv`/`tpsv` still use the full matrix.
  Making a device also fixes glibc's heap thresholds where its dynamic rule ends up (32 MiB mmap, 64 MiB trim), so a
  host-computed library call that loads and frees a few MiB of operands (cuSPARSE SpMV) no longer gives the heap back to the
  kernel and faults it in again each time (`VGPU_HOST_HEAP=0`, or a threshold set in the environment, keeps glibc's own).
  The setup action has a `huge-pages` input (default `true`): on a GitHub-hosted runner whose transparent huge pages are
  `madvise` it sets `always` for the job when sudo works, says what it did, and never fails the step
  (`scripts/thp-always.sh`). New tests `test_cublas_band_heap`, `test_host_heap` and `lint_thp_always` (pantheonsim #551).
- Setup action: a release carries the simulator prebuilt for linux-x86_64, one tarball and `.sha256` per runner and toolkit the
  action sets up (ubuntu-24.04 with CUDA 12.0, 12.6, 13.0 or none, ubuntu-22.04 with CUDA 12.6 or none), built and attached by
  `.github/workflows/release-binaries.yml` when a `v*` tag is pushed. The action, pinned at that release, downloads the tarball,
  checks its sha256 and unpacks it instead of building the simulator (8 to 24 minutes on a cold cache), and falls back to the cache
  and the source build when there is no tarball for it or the download or checksum fails. New input `prebuilt` (`false` always
  builds), outputs `prebuilt` and `prebuilt-asset`; `docs/github-action.md` gives the measured setup times (pantheonsim #599).
- SASS engine on hosts with AVX-512 (F/BW/DQ/VL and VNNI): binary64 add, multiply and fused multiply-add (`DADD`, `DMUL`, `DFMA`) run
  sixteen lanes to a step instead of four. Programs that compute in `double` run 3 to 39 % faster on a Zen 4 host (eight of nine
  measured gain 5 % or more: lavaMD 1.18x, RSBench 1.39x, LULESH 1.07x); single-precision programs and hosts without AVX-512 are
  unchanged, the results are the same bits, and `VGPU_SIMD=avx2` keeps the 256-bit kernel. New unit test `test_lanes512` (skipped on
  a host without AVX-512).
- Tier 2 (`VGPU_JIT=1`, still off by default): a code section the cache has nothing for no longer runs in the counting loop all the
  way to its compile threshold. It waits on the plain run loop, where only the warp instructions of each call are added up, and is
  counted region by region again for the last `VGPU_JIT_PROFILE` (4,000,000) instructions before the threshold. Programs that never
  reach the threshold paid 2 to 16 % for the counting with `VGPU_JIT=1`; results and counters are unchanged
  ([round sp57](docs/perf-design/rounds/11-rounds-sp36-sp41.md), which also says why the JIT stays opt-in).
- CI and setup action: when the apt mirror host of a GitHub-hosted runner image (azure.archive.ubuntu.com) does not resolve, the
  toolchain install points apt at the first of archive.ubuntu.com and four mirrors outside ubuntu.com whose name does resolve,
  instead of failing three times with exit 100 (`scripts/apt-mirror-fallback.sh`, run before every hosted-runner `apt-get update`;
  it changes nothing when the mirror resolves, and says so when no fallback resolves either).
- AMD: a scalar load whose destination is VCC (`s_load_dwordx2 vcc, s[94:95], 0x0`, which LLVM writes when it runs short of scalar
  registers) fills VCC. It wrote the loaded pair to s0 and s1, which on gfx90a hold the private segment buffer, so the kernel's next
  stack access stopped with "buffer_load_dword through a buffer that adds the lane's id" (Kokkos kernels such as OPALX's
  MultipoleT on the simulated MI250X, pantheonsim #579). Every SMEM destination that is not an SGPR (vcc_lo, vcc_hi, the pair,
  the trap temporaries) is written where it lives.
- AMD: a scratch access sums its scalar offset and its instruction offset in 32 bits, as the ISA gives the address
  (`flat_scratch + swizzle(sgpr_offset + inst.offset, threadID)`): `scratch_store_dwordx4 off, v[0:3], s0 offset:32` with s0 = -32
  is the work-item's byte 0, where it reached for 2^32 and stopped with "past the 1136 bytes a work-item reserved" (pantheonsim #578).
- AMD: a packed float instruction (`v_pk_add_f32`, `v_pk_mul_f32`, `v_pk_fma_f32`) that reads the constant 0 reads zero in both halves;
  `v_pk_add_f32 v[2:3], v[2:3], 0 neg_lo:[1,1] neg_hi:[1,1]`, a float negate, stopped with "takes the second part of a constant" and left
  mwa_hyperbeam's single-precision `fee_kernel` output at zero (pantheonsim #580). A nonzero constant's second half stays refused.

- CI: a heavy job (SASS, ollama, ollama-amd, ASan, TSan) takes a reservation before it samples the memory, not after. `ci/jobs.sh --check`
  now takes a share (the profile's minimum) in a ledger under `/tmp/pantheonsim-ci-reserve`, which the pool's containers bind-mount, waits
  while the shares leave no room (capacity 5 GiB: one heavy job per host, `CI_RESERVE_CAPACITY_GIB` raises it), lets the machine settle
  5 s and only then samples. A new last step, `ci/jobs.sh --release` (`if: always()`), gives the share back; one a killed runner never
  released expires after 4 hours. Before, two jobs on the two runners of one host both passed on the same free memory and the host's
  out-of-memory killer took them down (pantheonsim #575, #576). The ollama downloads retry once when the archive arrives partial, and
  the clean-up step no longer fails on a step-summary file the runner has already removed. The test is `lint_ci_jobs` (the ledger) and
  `lint_ci_reservation` (every job releases and mounts it).
- Setup action: the build cache is saved by an `actions/cache/save` step right after the build (on a miss only) instead of by
  `actions/cache`'s post step, which runs only when the whole job succeeds; a caller whose tests failed no longer throws the build away
  and rebuilds it for about 15 minutes next run (pantheonsim #582).
- CUDA runtime and driver: a kernel's `maxThreadsPerBlock` (`cudaFuncGetAttributes`, `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`)
  counts whole warps per sub-partition, as the launch check does. A kernel of 104 registers a thread reported 608 threads and a
  launch of 608 was refused ("cannot place a single block ... limited by registers"); it reports 512, as an RTX 5080 does for
  the same kernel.
- PTX: `ld.weak` and `st.weak` are accepted (the explicit spelling of the default memory semantics, PTX ISA 6.0 and later). They
  stopped GiMMiK 4.0's matrix-multiply kernels, and so PyFR's CUDA backend, at module load (pantheonsim #554).
- NVRTC (the built-in compiler, used when no toolkit is visible): `--ftz=`, `--prec-div=` and `--prec-sqrt=` are accepted beside
  `--fmad=`, as the real NVRTC takes them; PyFR passes `--ftz=true` to every compile (pantheonsim #555).
- CUDA driver: `cuModuleLoadDataEx` fills `CU_JIT_ERROR_LOG_BUFFER` (and the size slot) when a load fails, and malformed PTX is
  `CUDA_ERROR_INVALID_PTX` rather than `CUDA_ERROR_NOT_SUPPORTED`; a `.version` past 9.3 is `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, and
  text that does not begin with `.version` is `CUDA_ERROR_INVALID_IMAGE`. `cuLinkAddData` checks its PTX as it is added and fails
  the same three ways, with the log in the link state's error buffer, instead of `cuLinkComplete` succeeding on garbage. All
  measured on an RTX 3060 (gocudrv's `TestRealModuleJITLog` and `TestRealLinkerBadPTX`, pantheonsim #558).
- PTX: a `.reg` range may declare up to 1,048,576 registers (it was 65,536). ptxas allocates physical registers and counts only what
  a kernel uses, so nvcc's unrolled BN254 field arithmetic declares 125,511 and loads on a GPU; the per-block limit on registers a
  kernel uses is unchanged (pantheonsim #559).
- PTX: lanes of one group that sit at two copies of a `vote.sync` naming the whole group (nvcc tail-duplicates a ballot into both arms
  of an `if`) now vote together, as they do on a card, where each lane gets its group's ballot. The simulator ran the first copy
  for its lanes alone, and the `bar.sync` after it stopped with "cannot be reached by every lane of the warp", as in fastMD's
  neighbour-list build (pantheonsim #574). Measured on an RTX 3060 (sm_86, CUDA 13.0).

- SASS: a loop that spins with `__nanosleep` as its back-off (lanes of one warp racing for a slot with `atomicCAS`, the losers
  sleeping until the winner publishes) runs to the end. ptxas puts no YIELD in such a loop, so the lane that won stood at the
  BSYNC closing it and the spinners ran until the exec limit; a NANOSLEEP now lets the other lanes and warps go first, as a YIELD
  does, and the spin breaker applies (eliphatfs/cuacd's mesh_audit edge table, pantheonsim #553).
- `VGPU_HOST_STATS=1` prints one line of host-kernel counters when a simulated program exits (CPU user and system time, page faults,
  context switches, threads, huge-page use, heap-chunk `madvise` and `mmap` counts, worker-pool threads and calls, kernel, CPU count and
  THP mode); `VGPU_HOST_STATS=30` also prints one every 30 seconds, flushed, and `VGPU_HOST_STATS_FILE` redirects them. For finding why a
  run is slower on one kernel than another (pantheonsim #551).
- CI: the workloads summary and the "Pantheon workloads failing on main" issue name the machines whose job wrote no results (a hosted
  runner lost mid-build leaves no report, and the count alone did not say which) (pantheonsim #545).
- PTX and SASS `min`/`max`: the SASS tier could not decode `HMNMX2`, the instruction behind `min`/`max` on `f16`, `f16x2`, `bf16` and
  `bf16x2` (`__hmin2`, `__hmax2`, `min.NaN.f16`, ...), so a kernel using one stopped with an illegal-memory-access error; it now runs
  them, with `.ftz`, `.NaN` and `.xorsign.abs` as the PTX engine and an RTX 3060 give. `FMNMX.XORSIGN` (`min.xorsign.abs.f32`) ignored
  its sign rule on SASS and returned an unsigned magnitude. The PTX engine accepts the three-input `min`/`max{.ftz}{.NaN}{.abs}.f32` of
  sm_100 that ptxas emits (it was an unsupported-PTX error). `max.s32.relu`, `__viaddmax_s32_relu` and the other `.relu` forms of
  pantheonsim #547 already ran since #451; new e2e test `e2e_minmax_forms` sweeps every sibling form (the relu and DPX ones are in
  `e2e_dpx_intrinsics`).
- SASS engine, packed half arithmetic: with `HMNMX2` decoded, modules such as one holding `min.f16` no longer fall back to the PTX engine
  wholesale, which exposed `HADD2`, `HMUL2` and `HFMA2` forms the SASS engine got wrong against an RTX 3060 (51 of 380 forms of
  `e2e_ptx_half_forms`, and 4 of `e2e_ptx_forms`): `.FTZ` on an f16 result (subnormal inputs and result to a signed zero), `.SAT` (a -0 result is +0), `.RELU`
  (a negative or -0 result is +0), `F2FP` with a directed rounding mode (`cvt.rz`, `.rm`, `.rp` to f16 and bf16) and its one-conversion
  `MERGE_C` form (`cvt.rn.relu.bf16.f32`, which was read as an f16 pair) and `.SATFINITE` (`cvt.rn.satfinite.f16x2.f32`, whose
  flag was read as part of the type). They now give the card's bits, on the interpreter and
  the tier (the tier leaves those forms to the interpreter). `e2e_ptx_half_forms` runs on the SASS interpreter, the tier and the PTX engine
  and fails if the SASS did not run.
- cuFFT: an in-place real transform (`cufftExecR2C`, `D2Z`, `C2R`, `Z2D` on one buffer, no advanced layout) reads and writes the
  padded real array cuFFT documents (each row of the fastest axis is `2 * (n / 2 + 1)` reals, a batch is the complex distance
  apart). It read the array as dense, so 2D and 3D results were wrong and so was a 1D batch (pantheonsim #548).
- cuBLAS and cuBLASLt: `CUBLAS_TF32_TENSOR_OP_MATH`, `CUBLAS_COMPUTE_32F_FAST_TF32` and the FP16/BF16 fast compute types round the
  FP32 inputs of the host-computed GEMMs (A and B, not C) as a tensor core does: TF32 to nearest with ties away from zero
  (measured on an RTX 3060), FP16 and BF16 to nearest even, for `Sgemm`, `SgemmEx`, `Cgemm`, the batched forms, `GemmEx` and
  `cublasLtMatmul`, from the size the card starts using tensor cores; `NVIDIA_TF32_OVERRIDE=0` is honoured. Before, the result was the plain
  FP32 one in every mode (pantheonsim #549; the rules and what is modelled are in `nvidia/docs/cublas.md`).
- PTX engine, `wmma`: a fragment holds its matrix elements where a card puts them, for every shape and type (f16, bf16, tf32, s8/u8,
  s4/u4, b1, f64 operands; f16, f32, s32 and f64 accumulators). It used a layout of its own, so code that reads or writes
  fragment elements one by one (`acc.x[i] *= (a.x[i] > 0)`, how tiny-cuda-nn masks a ReLU's gradient, and the epilogues of CUTLASS and
  hand-written kernels) met the wrong element: tiny-cuda-nn's `FullyFusedMLP` double-backward test gave mean relative errors of 0.6
  to 0.8 for three of its five rows on the default SASS engine too, because a module holding one SASS instruction the engine did not decode (the packed half
  maximum, `HMNMX2`, until pantheonsim #547) ran on the PTX engine wholesale. The three code paths that held old layouts (f16 and bf16 `m16n16k16` with an f32 accumulator, tf32, and the generic
  one) are one now, and the tile route takes tf32 and the f16/bf16 `m16n16k16` forms with f16 accumulators too. The layout is the
  Turing-and-later one, measured on an RTX 3060 (a Volta card, which was not measured, may differ)
  (pantheonsim #550, new test `e2e_wmma_fragment_layout`).
- PTX engine: a `wmma.load` whose registers nothing reads is dead code and does nothing, as on a card where ptxas removes it. tiny-cuda-nn's
  fused backward pass loads a fragment of its last layer from a null pointer and uses none of it, which stopped the program with an
  invalid-pointer error at address 0 on the simulator only (pantheonsim #550, in `test_exec3`).
- CUDA runtime and driver: a zero-byte `cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaHostAlloc` or `cudaMallocAsync` succeeds
  and gives a NULL pointer, as on a GPU, where it gave a distinct non-NULL one (Gpufit tests its unused buffers against NULL); the
  driver's `cuMemHostAlloc`/`cuMemAllocHost` of 0 bytes do the same, and `cuMemAlloc(0)` is `CUDA_ERROR_INVALID_VALUE` with the
  result cleared (pantheonsim #515).
- CUDA runtime and driver: `cudaMemGetInfo` and `cuMemGetInfo` accept a NULL for the free or the total pointer, or both, and fill
  the other, as on a GPU. They used to answer invalid value, which stopped cuda-api-wrappers' device_management example
  (pantheonsim #519).
- CUDA driver: calls that work with no current context on a GPU now do. A stream-ordered call that names a stream of its own
  (`cuMemAllocAsync`, `cuMemAllocFromPoolAsync`, `cuMemFreeAsync`, the `Async` copies and fills including `cuMemcpy3DAsync`,
  `cuStreamSynchronize`, `cuStreamQuery`) takes its context from the stream; `cuMemFree` and `cuMemFreeHost` use the context that
  owns the pointer; `cuMemAddressReserve` and `cuMemAddressFree` need none. The legacy, per-thread and NULL streams and the
  synchronous copies still answer `CUDA_ERROR_INVALID_CONTEXT`. This is how cuda-api-wrappers drives the API (retain the primary
  context, push it for one call, pop it), and five of its examples stopped on it (pantheonsim #518).
- CUDA driver: `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_CONTEXT)` (and `cuPointerGetAttributes`) returns the context a
  `cuMemAlloc`, `cuMemAllocPitch`, `cuMemAllocManaged` or `cuMemHostAlloc` allocation was made in, not the one current on the
  calling thread (pantheonsim #521).
- CUDA driver: `cuModuleGetFunction` returns the same `CUfunction` for the same module and name every time, and it is the handle
  `cuModuleEnumerateFunctions` lists, so programs that compare handles (cuda-api-wrappers' module_management example) find them
  equal (pantheonsim #520).
- CUDA runtime: `cudaLaunchHostFunc` (and `cudaLaunchHostFunc_v2`) enqueues the function and returns; it runs on a worker thread for
  the stream, after the stream's earlier host work and before its later work. An event recorded behind it, and `cudaStreamQuery`, report
  `cudaErrorNotReady` until it has finished, as does `cudaEventElapsedTime`; `cudaStreamSynchronize`, `cudaEventSynchronize` and
  `cudaDeviceSynchronize` wait for it; `cudaStreamWaitEvent` on a pending event holds the other stream back; the legacy default stream
  orders against blocking streams. A host function that waits for something the caller does after enqueueing it no longer hangs the
  call (pantheonsim #487). Copies and fills still run to completion inside their calls; kernel launches are the next entry;
  the driver API's `cuLaunchHostFunc` is the entry after that. Programs that never queue a host function see no change. `cudaErrorNotReady` from a query is no longer left as the last error (`cudaGetLastError`), as on a card.
- CUDA driver: `cuLaunchHostFunc` (and `cuLaunchHostFunc_v2`) enqueues the function and returns, as the runtime's does and as a GPU does
  (pantheonsim #487, the driver half). It runs on the stream's worker thread, in order with the stream's other host functions, and what
  the stream is given afterwards waits for it: `cuLaunchKernel` and its variants, every `cuMemcpy*` and `cuMemset*` form (the
  synchronous ones on the legacy stream wait for blocking streams' host functions only), stream memory operations and
  `cuStreamAddCallback`. `cuStreamQuery` is `CUDA_ERROR_NOT_READY` while one is pending, `cuStreamSynchronize` and `cuCtxSynchronize` wait
  for it, and a driver event recorded behind it is `CUDA_ERROR_NOT_READY` from `cuEventQuery` and `cuEventElapsedTime` until it has finished
  (`cuEventSynchronize` waits, `cuEventElapsedTime` measures to the moment it finished), `cuStreamWaitEvent` holds the other stream back.
  A host function that waits for something the caller does after queueing it no longer hangs the call. Programs that never queue a host
  function see no change. Not covered: stream-ordered allocation and free, prefetch and attach, external semaphores and `cuMemMapArrayAsync`
  do not wait for earlier host functions, and `cuStreamAddCallback` still runs its callback in the call (after the earlier host
  functions) instead of later.
- CUDA runtime: a kernel launched with `cudaLaunchKernel` (`<<<>>>`) on a stream made with `cudaStreamNonBlocking` returns before the
  kernel has run, so `cudaEventQuery` and `cudaStreamQuery` on work behind it report `cudaErrorNotReady`, and so does
  `cudaEventElapsedTime`, until it has finished (pantheonsim #523; the issue's repro now prints 600, 600, then 0 after the
  synchronize, as an RTX 3060 does). The kernel runs on the stream's worker thread under the API lock, so every call that takes the
  lock (copies, frees, allocations, the driver API) waits for it as it waited inside the launch, and a copy or launch on the same
  stream, `cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaDeviceSynchronize` and `cudaStreamWaitEvent` order after it. Results,
  counters, `VGPU_KERNEL_DIGEST`, error text and sticky faults are the same as before; a launch the device refuses for its
  configuration is still refused in the launch. A kernel that ends in a failure that is not a sticky fault (ran past
  `VGPU_MAX_STEPS`) reports it at the launching thread's next call instead of at the launch. It is not used on the default streams,
  with a profiler or `VGPU_TRACE`, `VGPU_RACE`, `VGPU_MAX_STEPS` or `VGPU_STRICT`, for cooperative, cluster and graph launches, while
  a managed allocation exists, or while a CUDA library shim (cuBLAS, cuDNN, cuFFT, ...) is loaded, whose math runs on the calling
  thread with no lock; those launches run inside the call as before. `VGPU_ASYNC_LAUNCH=all` adds streams made with `cudaStreamCreate`;
  `VGPU_ASYNC_LAUNCH=0` turns it off. Design and limits: docs/perf-design/streams.md.
- PTX: `mov.u32%r1, %laneid;` (no space between the type and the operand, which nvcc emits for inline asm written `"mov.u32%0, ..."`) is
  accepted, as ptxas does; a `%` only ever opens a register name, so it now ends the opcode word (pantheonsim #522).
- PTX: `elect.sync _|%p, mask;` and `elect.sync %r|_, mask;` accept the sink `_` for either result, as ptxas does (cubecl's
  `plane_elect`) (pantheonsim #503).
- PTX: `setp` and `set` on `.bf16` (and `.bf16x2`) compare the values, not the raw bits: the sign counts, `-0 == +0`, and NaN is
  unordered, so `setp.nan.bf16`, `setp.neu.bf16` and the ordered forms give the answers they give on `.f16` (pantheonsim #502).
- PTX: `shfl.sync`, `vote.sync`, `match.sync`, `redux.sync` and `bar.warp.sync` wait for every lane that any running lane's member mask
  names, not just the first lane's. With two half-warp tiles, the tile that did not contain the first lane went on while a member of
  its own tile was still in a long loop and read 0 for it (HierarchicalKV's `tlp_v2_upsert*`). Only the lanes whose members are away
  wait; the rest of the path runs on (pantheonsim #488).
- libcudart exports the nine `cudaGL*` interop functions the CUDA 13.0 runtime does (`cudaGLRegisterBufferObject`, `cudaGLMapBufferObject`,
  `cudaGLSetGLDevice`, `cudaGLGetDevices`, ...). A program that only names one of them, such as COLMAP's SiftGPU, failed to start when linked with
  `-z now` ("undefined symbol: cudaGLRegisterBufferObject"). With no OpenGL context on a simulated GPU they answer what an RTX 3060 answers with none:
  `cudaErrorInvalidGraphicsContext` for register, unregister and `cudaGLGetDevices`, `cudaErrorInvalidValue` for map, unmap and the map flags.
- `VGPU_MEMO=1` (opt-in, off by default and then unchanged): launch-level result caching. A launch whose kernel, grid, arguments,
  environment and device-memory history have been seen before is not run; its stored write set, counters and kernel digest are applied
  (`VGPU_MEMO_DIR` keeps entries between runs, so a second run of a deterministic program hits from its first launch;
  `VGPU_MEMO_VERIFY` runs every hit for real as well and compares). The device's memory is named by a hash chain over what the
  program did to it, not by its content, and anything the cache cannot describe (virtual memory management, IPC, pinned or managed
  memory a kernel can reach, device malloc and printf, cooperative and cluster launches, textures, atomics on several threads, a
  profiler, the race detector, an armed fault, ...) restarts the chain, so the cost on a miss is a budgeted before-and-after hash of
  the allocations the launch can write and the answer is never a stale one. Knobs: [docs/cli.md](docs/cli.md); rules, tests, mutants
  and measurements: [docs/perf-design/rounds/10-rounds-sp19-sp27.md](docs/perf-design/rounds/10-rounds-sp19-sp27.md#round-sp35-launch-level-result-caching-built-tested-measured-branch-sp35-memo-base-originmain-9f80d40e).
- The container image builds again. `scripts/package-build.sh` left `shim/libcudss.so.1` pointing into `cudss1/`, a directory the image does not
  have, so the build stopped at "links that lead nowhere" from the cuDSS 1.0 change onward. A lint test (`lint_package_build`) now builds a small
  tree with the same layout.
- SASS: `ULDC` with a uniform base register (`ULDC.64 UR6, c[0x3][UR8+-0x72e0]`) takes its offset as a signed 16-bit number and wraps in the
  64 KiB bank, as `LDC` does. ptxas uses it to reach constant data past byte 32768, and the simulator read zero there on the Ampere and Ada
  profiles (an inverse NTT in FIDESlib came out all zeros) (pantheonsim #499).
- `vgpu shell --rootfs`: opt-in private root. The session gets its own root on a tmpfs (entered with `pivot_root` inside the mount
  namespace the shell already makes): the host's `/usr`, `/bin`, `/sbin`, `/lib*` and the `/etc` entries a loader, resolver, TLS
  and shell need come through read-only; `/root`, `/home`, `/opt`, `/tmp`, `/var/tmp` and `/work` start empty and are discarded on
  exit; `/dev` is minimal; `/proc`, `/sys` and the simulated GPU surfaces are as before; `/usr/local/cuda` and `/opt/rocm` appear
  only where the host has a toolkit. `-v HOST:DEST[:ro]` (repeatable) adds host paths the way `docker run -v` does and `--home DIR`
  keeps `/root`. Where user namespaces are refused it exits `1` rather than fall back to the host's files. Default behaviour is
  unchanged. See [docs/machine-simulator.md](docs/machine-simulator.md#a-private-root---rootfs).
- PTX: `abs.s16` takes the sign from bit 15 of the register. It used to read the whole 32-bit register as s32, so a negative
  value written with zero upper bits (`mov.b16 a, 0xff81`, `cvt.u16.u32`) came back unchanged instead of as `0x007f`.
- PTX accepts unary `~` on an integer constant (`and.b64 %rd3, %rd3, ~0x2;`), alone, nested, with `-` or in
  parentheses, at the width of the instruction; the lexer used to stop with `unexpected character '~'`
  (pantheonsim #483).
- PTX: a negated predicate source (`and.pred %p2, !%p0, %p1;`, also `or`, `xor`, `not.pred` and the predicate of `selp`) is accepted. drjit-core
  emits it for boolean operations, and the module used to fail to load with `expected operand, got '!'`.
- PTX: `ldu.global` (the uniform load, scalar and `.v2`/`.v4`) runs as the `ld.global` it stands for.
- PTX: the array-element address `var[imm]` (`st.local.b32 arr[0], %f0;`, also `[arr[1]]`) is accepted for `.local`, `.shared`,
  `.global` and `.const` arrays. The index counts elements of the declared type; one outside the array is refused.
- PTX: the array-element address with a register index, `st.local.b32 arr[%r15], %f16;` and `ld.local.b32 %f100, arr[%r99];` (a 32- or
  64-bit register; `.local`, `.shared`, `.global` and `.const` arrays), is accepted: the address is the variable's plus the register times
  the element size, per lane, unchecked as on the card. Dr.Jit emits it for every opaque-index array access, and the module used to fail to
  load with `bad integer literal '%r0'` (pantheonsim #530).
- The runtime exports the OpenGL interop functions of `cuda_gl_interop.h` (`cudaGraphicsGLRegisterBuffer`, `cudaGraphicsGLRegisterImage`,
  `cudaGLGetDevices`, `cudaGLSetGLDevice` and the legacy `cudaGL*BufferObject` family), so a program linked with `-z now` that merely
  references one (COLMAP's SiftGPU) starts instead of dying with `undefined symbol: cudaGLRegisterBufferObject`. They answer as an RTX 3060
  with no GL context does: `cudaErrorInvalidGraphicsContext`, `cudaErrorInvalidValue` for a buffer that was never registered
  (pantheonsim #528). `tests/lint/check_header_exports.py` now checks `cuda_gl_interop.h` too.
- PTX: a `.func` can take its parameters and return its value in `.reg`s (`.func (.reg .b32 r) f(.reg .u64 data)`), and a call
  (direct, or through a `.callprototype` with `.reg`s) passes registers or numbers for them.
- libcuda exports the OpenGL, EGL and VDPAU interop functions (`cuGraphicsGLRegisterBuffer`,
  `cuGraphicsGLRegisterImage`, `cuGLGetDevices`, `cuGLCtxCreate`, the legacy `cuGL*BufferObject` family,
  `cuGraphicsEGLRegisterImage`, `cuEGLStream*`, `cuVDPAU*`, `cuGraphicsVDPAURegister*`) and `cuGetProcAddress`
  finds them. They answer `CUDA_ERROR_NOT_SUPPORTED`, since a simulated GPU has no display stack; before, a loader
  such as Dr.Jit that looks the names up when it loads libcuda turned its CUDA backend off.
- `cuLaunchKernel`, `cuLaunchKernelEx` and graph kernel nodes take the `extra` argument form
  (`CU_LAUNCH_PARAM_BUFFER_POINTER`, `CU_LAUNCH_PARAM_BUFFER_SIZE`, `CU_LAUNCH_PARAM_END`) that cudarc and Dr.Jit
  launch with. The buffer holds the kernel's parameters at the offsets `cuFuncGetParamInfo` reports. A list with no
  buffer or size, a size short of the parameters, a repeated or unknown key, no `CU_LAUNCH_PARAM_END`, or both
  `kernelParams` and `extra` is `CUDA_ERROR_INVALID_VALUE`; it used to be `CUDA_ERROR_NOT_SUPPORTED` for every list.
- cuDSS 1.0 runs. The simulator's cuDSS is now presented as `libcudss.so.1` (1.0: its version macros,
  `cudaDataType_t` signatures, `cudssSetJitCacheDir` and the pinned-memory handler functions, the four new
  config parameters, Bunch-Kaufman accepted) with `libcudss.so.0` (0.8) kept beside it, unchanged, so programs
  built against either major run. A program built against NVIDIA's 1.0, such as SCS's `run_tests_cudss`, no
  longer stops with "needs libcudss.so.1, but this build of the simulator has no libcudss.so" (#426).
- A kernel run from PTX no longer fails to launch when its register estimate is up to twice the block's register
  file. The estimate is the peak of live virtual registers and ptxas needs fewer, so ggml's `rms_norm_f32`
  (88 estimated, 64 allowed at 1,024 threads) stopped Ollama running Gemma 3 on the RTX PRO 6000 profile, the one
  profile whose Ollama build has no cubin. The excess is charged as spill; a launch several times over still fails.
- The GitHub Action has a `source-dir` output, the simulator's source tree. A project that has the pantheon
  workloads can run `tests/workloads/run_pantheon_workloads.sh` from its own CI with one `uses:` and no second
  checkout (`docs/pantheon-workloads.md`).
- The pantheon workload runner reports a workload that printed `Skipping: <reason>` as skipped, with the reason,
  not as passed.

## [0.2.0] - 2026-10-24

### Getting started
- `make` installs what is missing (it asks first) and builds; `./run` starts a simulated GPU machine from a
  settings file. The README is those two commands; the rest is in `docs/getting-started.md`, `docs/config.md`
  and `docs/cli.md`. The simulator uses half your CPUs unless you say otherwise.
- `scripts/ci-setup.sh` does for GitLab CI, CircleCI, Forgejo, Jenkins and Buildkite what the GitHub Action does
  (`docs/ci-other-platforms.md`).
- Every document was checked against the code, and the profile counts, command lists, ROCm versions and
  limitations were corrected. 48 profiles (38 NVIDIA, 10 AMD); 14 NVIDIA and 3 AMD are read from physical cards.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates (bug, something unsupported, a wrong GPU profile) and a
  pull request template say how to send a change and what it needs.

### Faster
- The SASS engine's pre-decoded tier is on by default; the AMD engine has a tier of its own, and matrix and
  packed-math fast paths. Atomics, kernel launch and the first launch of a fatbin cost less. A second tier that
  compiles hot straight-line code (`VGPU_JIT=1`) is available and off by default. Every fast path produces
  the same bits as the path it replaces, and the old path stays available through the documented `VGPU_*`
  switches so the two can be compared (`docs/cli.md`).
- Blocks that wait on one another (grid-wide spins, decoupled look-back) now run together, taking turns when
  they cannot all be resident. This applies to SASS, PTX and AMD code.

### NVIDIA
- New profile: RTX PRO 4500 (read from a card).
- CUPTI traces were compared with the real thing on 19 rented GPUs; `torch.profiler` traces are checked against a
  card in CI. `VGPU_PLATFORM=linux` switches CUPTI to the answers a Linux driver gives (the default is `wsl`,
  `nvidia/docs/cupti.md`).
- SASS: a cloned device function runs its own copy; a yielding warp rotates through its lane groups.
- LuisaCompute's CUDA backend runs (scalar `suld`/`sust`, `txq`, the device runtime library in `cuLinkAddData`);
  cuTensorNet and cuda-samples fixes.

### AMD
- `rocprofv3` traces of small HIP programs match what it recorded on a real RX 6800 and RX 6700 XT.
- gfx942 and gfx950 were swept against AMD's instruction guides; gfx90a images, kernel-to-host calls (hostcall),
  PC sampling, the ROCm Validation Suite and vLLM graphs run (`amd/README.md`).

### Known limits
- Functional only: there is no timing model, so a run on a physical GPU before shipping still counts.
- Programs must link the shared CUDA runtime (`nvcc -cudart shared`); a statically linked runtime needs NVIDIA's
  undocumented driver export tables (this is why CuPy does not work).
- Nsight Systems and Nsight Compute as NVIDIA ships them, and OptiX, are not supported. See `docs/status.md`.

## [0.1.0] - 2026-10-15

First tagged release: simulated NVIDIA and AMD GPUs, so CUDA and HIP programs run unmodified on a machine with
no GPU (development, teaching and CI). GitHub Actions through `pantheongpu/setup-pantheonsim`, a container on
`ghcr.io/pantheongpu/pantheonsim`, or a build from source.
