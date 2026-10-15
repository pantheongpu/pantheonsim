# Simulator speed: design studies (round 7, r7-design)

> **Status (round 8):** route (a) of study 1 is built, opt-in: `VGPU_TIER=1`, `src/sass/tier*`; measurements and the tests that hold it to
> the interpreter are in [docs/performance.md](../performance.md#the-pre-decoded-tier-round-8-vgpu_tier1). Routes (b) and (c) are not built.

Feasibility studies for the largest possible speed-ups, with a throwaway prototype that measures potential. **No
file under `src/` was changed.** The prototype is on branch `r7-design-proto` (head `040114fc`,
`tools/r7proto/`), which is not for merging; the raw measurements are in `data/`.

| study | verdict | one line of evidence |
| --- | --- | --- |
| [1 Whole-kernel translation](01-whole-kernel-translation.md) | (a) closure compilation: **do**. (b) generated C++ via host compiler: **do later, opt-in**. (c) x86 template JIT: **do not** | same bits and same counters on 10 kernels; ALU-bound loops 1.6-5x (closure, baseline ISA), 6-10x (closure, AVX2/FMA), 19-35x (generated, AVX2) over today; memory-bound kernels stop at ~3x until the memory path is fixed |
| [2 Tiering and caching](02-tiering-and-caching.md) | tier 1 always on; tier 2 (compiler) opt-in with an on-disk cache keyed on relocated instruction words | break-even ~5,000 warp instructions (tier 1) and 4-74 M (tier 2); launch constants as parameters cost nothing, so the cache is argument-independent |
| [3 Memoising deterministic kernels](03-result-memoisation.md) | **not by default; later, opt-in** | 6 of 8 pantheon workloads repeat their own post-state on 99.6-100% of launches, 2 never do; but stress runs are the experiment, and the input set is the hard part |
| [4 Recognising library kernels](04-library-kernel-recognition.md) | **do not** | cuBLAS etc. are already native (cublas_api.cpp:3-14); recognising user-code GEMMs needs bit-exact accumulation order proofs per template instantiation |
| [5 Sharding and sampling](05-sharding-and-sampling.md) | **do not** (both) | threads already scale 7.7x on 16; sampling changes results by definition and breaks self-verifying workloads |

## What the prototype is

`engine.cpp` (closure-compiled executor for ~14 SASS ops), `codegen.cpp` (C++ source for straight-line ALU runs,
compiled with the host `g++` and `dlopen`ed), `r7proto.cpp` (harness: four executors, alternating, 9 repetitions,
bit and counter comparison against the real engine), `coverage.cpp` (static coverage over the pantheon workload
cubins), `kernels/` (vecadd, saxpy, ffma_loop, int_loop and the `ops_*` stress kernels). Reproduce:

```
git -C <clone> worktree add -b r7-design-proto-repro <dir> origin/r7-design-proto
# build libvgpu.a from origin/main 443e79c0 (cmake --build build --target vgpu), then
S=<src of that tree> O=<scratch dir> <dir>/tools/r7proto/build.sh        # builds r7proto and r7proto_avx2
nvcc -cubin -arch=sm_86 -O3 tools/r7proto/kernels/k.cu -o k.cubin        # and ops.cu -> ops.cubin
./r7proto_avx2 k.cubin ffma_loop grid=16 block=256 iters=1000 reps=9 variants=base,proto,genr,genf opt=2 isa=avx2
./r7proto_avx2 ops.cubin ops_float grid=64 block=48 reps=2 variants=base,proto,genr,genf   # same-bits stress
tools/r7proto/sweep/sweep.sh        # the table in study 1
```

The pantheon cubins come from the public pantheon sources built with nvcc 13.0 for sm_86 (`cuobjdump -xelf all`).

## Findings outside the studies that other round-7 owners may want

Each was observed with the profiler or by reading code on `443e79c0`; none was changed here.

1. **SASS FFMA calls libm per lane** on the baseline build (`std::fma` in `fast_alu`, exec_ops.inc:1252; `nm -u` shows
   `fmaf`): with `-mfma` the closure tier's FFMA kernel went from 1.6x to 8.4x over today (r7-simd, r7-sass).
2. **`std::popcount` is a libgcc call** (`__popcountdi2`, ~2% of int_virus) on the baseline build (r7-build).
3. **`count_access` allocates and sorts two `std::vector`s per memory instruction** (exec_ops.inc:1786); the exact
   counters can be computed without them (prototype `count_global`, identical counters on 4 memory kernels) (r7-sass, r7-memory).
4. **`MemoryManager::load_run`/`store_run` decline whenever an access-fault hook is attached** (memory.cpp:1075), and the
   runtime attaches one to every device (runtime.cpp:1019-1024), so the TMA run path and any coalesced path always
   take the scalar route in production. With the hook ignored, contiguous warp accesses ran 12-21x faster in the prototype (r7-memory).
5. `fast_alu` re-derives operand kinds on every execution (`plain()` checks, staging buffers) although the decoded
   `Instr` never changes; the int_virus profile puts 78% in `fast_alu` and its staging lambdas, and the closure tier
   (which decides once) runs the same loop at 8 ns instead of 41 ns per warp instruction on the same ISA.

## Honesty notes

* Load on the host was 10-12 during the sweep (20 vCPUs, other agents building); speed-ups are ratios of adjacent
  alternating runs, medians of 9, ranges in `data/prototype-sweep.txt`. Do not read the absolute M/s as stable.
* Everything is SASS (the default engine) on sm_86. The PTX engine, aarch64, AVX-512, divergent/barrier/atomic
  kernels, shared and local memory, and the interpreter-escape path of a production tier were not prototyped.
* Coverage numbers are static counts over 144 kernels of the pantheon workloads, not dynamic weights.
* The ARCHITECTURE.md D1 text (lines 91-95) still names LLVM as the upgrade path and contradicts line 65; I did not
  edit it (outside my file ownership). It should be amended when a tier is chosen.
