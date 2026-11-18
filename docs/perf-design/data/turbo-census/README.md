# Turbo refusal census (2026-11-17)

Question: for every built corpus binary, each host card and each simulated profile, which image would turbo mode give the card for each kernel, and when the kernel stays
in the simulator, why. Reading: [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) sections 15 and 16. Nothing here changes turbo's behaviour.

## The tool

* `tools/turbo-census/turbo_census` (`census.hpp`, `turbo_census.cpp`; CMake target `turbo_census`, CPU only, no driver, no device): reads the `.nv_fatbin` of an ELF file, one module per
  fatbin container, and prints one line per kernel with one cell per (profile, host card). Defaults: hosts sm_75 (T4), sm_86 (A10G, RTX 3060, RTX 3080 Ti), sm_89 (L4), sm_90 (H100); profiles
  sm_86 (`nvidia/a10g`) and sm_90 (`nvidia/h100`).
* `tools/turbo-census/aggregate.py`: ranks the reasons, joins simulator seconds, writes the files below.
* `tests/unit/test_turbo_census.cpp` (CTest `test_turbo_census`, no card): holds the tool to turbo's own classifier on hand-made cubins (sm_86 and sm_90) and PTX texts.

**What is the runtime's own code (linked, not copied):** `pick_card_image`, `pick_retarget_image`, `retarget_ptx_text`, `describe_candidates`, `scan_ptx_for_card`, `turbo_class`, `call_needs_scan`,
`call_stays_inside`, `call_to_entry`, `turbo_refusal`, `extern_kind`, `extern_admitted`, `variable_layout_mismatch`, `card_image_at_sim_addresses`, `sass::analyze`, `sass::reachable`, the fatbin reader
(`card_source_from_fatbin`) and the simulator's own image choice (`pick_cubin`).
**What is not shared, and was written again:** the walk over a kernel's code and its callees and the order in which the refusals are tried (`scan_kernel`, `classify_module`). In the runtime they are
inside `Impl::module_card` and `Impl::kernel_eligible` (`src/runtime/host_gpu.cpp`), which need a live `Impl` (driver state, memory manager, launch arguments) and cannot be called alone; the tool follows
them statement by statement and the unit test holds the two to each other (three or more mutations: `mut_*.txt`). If that code changes, `census.hpp` must change with it. A cross-check against the
runtime's own census of the same programs (`../sp66/census.md`, a run with a stand-in driver) is in the report.

**What the census cannot see** (decided per launch, not per kernel): arguments that point into host-mapped memory, VMM/IPC/managed allocations in use, cooperative launches, the fit check (block size,
registers, shared memory, grid, window against the card's memory), the card's refusal of PTX the text scan admits, a kernel whose card launch faults (Rodinia srad_v2), and all the performance rules (size,
sample, first launch, start cost, bans, streak). A PTX module reports only its first blocker (`scan_ptx_for_card` stops at one). The kernel list is every entry point of a binary's fatbin, reached or not.

## Files

* `cells.tsv`: per cell, kernels admitted, how many by the `unlisted` class, which image kind, programs fully admitted.
* `reasons.tsv`: per cell, every refusal reason: kernels (as any reason / as the runtime's first reason / as the only reason), programs, simulator seconds of the timed programs that have it, top 5 programs.
* `programs.tsv`: per program, simulator seconds (median over the tables of this directory's parent, `-` when none) and for the four cells a10g:sm_86, h100:sm_86, a10g:sm_90, h100:sm_90 `all` or `<n> ok;<reason>=<kernels>`.
* `libs.tsv`: the same reasons for libtorch_cuda.so (PyTorch 2.10+cu130) and the Ollama cuda_v13 libggml-cuda.so, where no timing exists.
* `mut_*.txt`: the mutations of the tool, each shown failing in `test_turbo_census`.

## Regenerate

```
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVGPU_LTO=OFF -S . -B build && ninja -C build turbo_census test_turbo_census
build/turbo_census --out census.tsv <binaries or .so files>
python3 tools/turbo-census/aggregate.py census.tsv --out docs/perf-design/data/turbo-census --markdown summary.txt --program-cells p86:h86,p90:h86,p86:h90,p90:h90
```
Build on a machine with no other load (server3 rules of the round brief); the 654 corpus binaries (648 MB) take 18 s, libtorch_cuda.so plus libggml-cuda.so a few minutes and 2.7 GB of memory.
