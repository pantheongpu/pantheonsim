# Round sp66: what keeps turbo mode's launches in the simulator, and what was lifted (data)

The design is section 10 of [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) ("What a module holds besides code"), and section 2 for the
atomics default. The round's text is in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md).

## Before the card (server3: GCC 13.3, `Release`, `-Werror`, no LTO, `VGPU_THREADS=3`, profile `nvidia/a10g`, the stand-in driver)

Every build and run under the machine's build lock, `nice 19`, `-j3`, holds of under 5 minutes. The corpus is the sm_86 binaries of PolyBench/GPU,
Rodinia, the reduced HeCBench list and the CUDA Samples (13.0) that run in the simulator (`tools/bench/real/real.py`'s manifest).

| file | what |
|---|---|
| `census.py` | runs each program once in turbo mode with `VGPU_HOST_GPU_LOG=1` against the stand-in driver (`nvidia/tests/e2e/host_gpu_stub_driver.c`, which refuses every module, so every launch runs in the simulator and is timed there) and keeps the mode's `host_gpu_kept` lines: per kernel and reason, launches and simulator seconds |
| `table.py` | the census tables: per reason, and per program, weighted by simulator seconds; two censuses side by side |
| `census.md`, `census.json` | `table.py` over the census of the rules before the round (`old`, the build of commit aa0b45d6, which only names the reasons) and of this round's (`new`) |
| `mutants.py`, `mutants.txt` | mutants of the card image, the code relocation writer and the new helpers of `include/vgpu/host_gpu.hpp`, each a copy of one header outside the tree; `tests/unit/test_host_gpu_logic.cpp` must fail against each |
| `atomics.py` | the card's kernel time a launch against the simulator's for every kernel with a float atomic with a value, a CAS or an EXCH, from one `VGPU_HOST_GPU_VERIFY=all` run (card stage) |
| `card-run.sh` | the card part, by stage: `build` (the card CI, with the round's new e2e steps), `changed` (the programs whose decisions change, 3 runs of sim, exact, exactw0, turbo, turbow0, and verify), `atomics`, `rest` (the rest of the corpus, one run of sim, exact, turbo) |

## On the card (one A10G, driver 595.112.07, nvcc 13.0)

| file | what |
|---|---|
| `card-ci.txt` | the card CI's verdict lines: the first pass (both e2e tests failed on the printf step: `LEPC`, then nvcc 13.0's call to `vprintf` through a register) and the second, on the branch head's build (unit test and both e2e tests pass) |
| `changed.md` | the 45 programs whose decisions change: sim, exact, exactn, turbo, turbon (3 runs; programs over 20 s one), and a verify run. Its `exactn` and `turbon` columns (and `repeat.md`'s) are `VGPU_HOST_GPU_WIDER=0`: the harness's forms of that name when the round ran, renamed `exactw0` and `turbow0` when the round was merged with sp65, whose `exactn` and `turbon` are `VGPU_HOST_GPU_STREAK=0` |
| `rest.md` | the rest of PolyBench, Rodinia and HeCBench: sim, exact, turbo, one run |
| `repeat.md` | ten runs of each program that read over 3 % slower than the old rules (second pass, the head's build) |
| `exact-verify.txt` | `VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1` of every program whose exact decisions the round changed |
| `atomics.md` | `atomics.py`'s table: the card's kernel time against the simulator's for each CAS, float-with-value and EXCH kernel |
