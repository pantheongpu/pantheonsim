# Round sp61: progress inside a block (data)

The design is section 8 of [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) ("What it did not reach, and progress inside a block").
The round's text is appended to [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md) after the card part.

## Before the card (server3: GCC 13.3, `Release`, `-Werror`, no LTO, `VGPU_THREADS=4`, profile `nvidia/a10g`, no driver)

Every build and run under the machine's build lock, `nice 19`, `-j3`. The corpus is the sm_86 binaries of PolyBench/GPU, Rodinia and the
reduced HeCBench list (`tools/bench/real/pack_corpora.sh`), run by `../sp60b/samplelog.py`.

| file | what |
|---|---|
| `table.py` | per sampled launch (the `host_gpu_sample:` lines samplelog.py keeps): blocks, inside or whole blocks, live memory, static estimate, age, whether the default rule would sample it (`affordable`), cost, warps followed, turns left read, steps a turn, estimate, what the simulator took, ratio, whether it would be sent |
| `offline-turbo.md` | `table.py` over `VGPU_HOST_GPU=turbo VGPU_HOST_GPU_SAMPLE=log` (every candidate sampled, nothing sent), the three suites without GRAMSCHM, 2MM, LU, FDTD-2D and the HeCBench programs over 120 s; binomial was cut at 150 s |
| `offline-exact.md` | the same in exact mode, PolyBench and Rodinia |
| `stub.txt` | COVAR, CORR, SYRK with the stand-in driver: off, `stub` and `tstub` (sent and refused: the simulator runs it, output identical), `stubnop` (sent to a card that keeps memory and computes nothing: the output changes, so the launch went) |
| `mutants.py`, `mutants-pass1.txt` | mutants of `include/vgpu/exec/block_progress.hpp` and of the loop finder and rule in `include/vgpu/host_gpu.hpp`, each a copy of one header outside the tree; the first pass and what was done about its survivors (in the round's text) |
| `card-run.sh` | the card part, by stage: `build`, `ci`, `target` (COVAR and CORR, 3 runs of sim, exact, exactw, turbo, turbow), `ab` (the programs with a candidate inside its blocks, 3 runs, then the rest of PolyBench and Rodinia, 1 run), `log`, `repeat` |

What the offline pass says:

* Launches that are candidates inside their blocks (grid under 256 blocks, a followed loop): COVAR, CORR, ATAX (2), BICG (2), GEMVER (2),
  GESUMMV, MVT (2), Rodinia sc_gpu, HeCBench aop and bh (4). Of these only COVAR's `covar_kernel` and CORR's `corr_kernel` pass the floor
  (`affordable=1`), so only they are sampled by default; no other program's decisions change. Estimate over the simulator's time: COVAR
  1.02 to 1.15, CORR 1.30 to 1.62; the other candidates 0.67 to 1.78 on launches of 0.03 to 0.22 s, plus four HeCBench launches of about 1 ms
  estimated 8 to 14 times high (none near the bar). No candidate other than COVAR and CORR would be sent.
* Launches sampled by whole blocks are sp60b's path: the ones that would be sent are SYRK, SYR2K, binomial, black-scholes and shmembench,
  as in sp60b.
* In exact mode `corr_kernel` is admitted (exact set) on this build: CORR's long kernel is sampled and would be sent in both modes.

## On the card (AWS g5.xlarge, us-east-1b, instance i-0d6ab20cf7cabbd11: A10G, driver 595.112.07, 4 vCPU, nvcc 13.0; 2026-11-13)

Simulator: `RelWithDebInfo`, no LTO (`tools/card-ci/run-card-ci.sh`), profile `nvidia/a10g`; the same corpus binaries.

| file | what |
|---|---|
| `card-ci.txt` | the card CI's lines of verdict, first build: everything passed but `e2e_host_gpu_stub` (step 22: the test kernel's unrolled loop, see the round's D) |
| `card-stub.txt` | the `ci` stage after the test kernel's fix: `e2e_host_gpu_stub`, `test_host_gpu_logic`, `hostgpu_bench_selftest` pass |
| `target.md` | COVAR and CORR, 3 runs of sim, exact, exactw, turbo, turbow (the harness's table) |
| `ab-inside.md` | the nine other programs with a candidate inside its blocks, and COVAR and CORR again (1 run), 3 runs a form |
| `ab-rest.md` | the rest of PolyBench and Rodinia, 1 run a form |
| `repeat.md` | ten runs a form of the five programs `ab-rest` read more than 3 % apart |
| `ab.md`, `ab.json` | `summary.py` over the four: per program the medians, the ratio to the form that samples whole blocks only, what was sampled and sent, the outputs; compact records |
| `sample-log.txt` | one `VGPU_HOST_GPU_LOG=1` run of COVAR and CORR in each mode: the sample's line and what the card took |
