# Round sp76: a kernel's first launch once the driver is up (data)

The design is section 13 of [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md); the round's text is *Round sp76* in
[../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md). Tracking: pantheonsim #657, #659 (and Rodinia
particlefilter_float, #662 and the rest of #656, measured and not changed).

Everything here was built and run on server3 (Ryzen 7 PRO 8840U, GCC 13.3, `Release`, `-Werror`, no LTO, `VGPU_THREADS=3` unless said,
profile `nvidia/a10g`), against the stand-in driver (`nvidia/tests/e2e/host_gpu_stub_driver.c`). **server3 has no card: nothing in this
directory is a card result but `card.md`, which is the A10G run of `card-run.sh` made afterwards (2026-11-15).** Builds and runs held the machine's build lock, `nice -n 19`, `-j3`; another session built and ran under the
same lock all the while, so wall times here are to within a few percent at best and a pass made an hour later can read 20 % apart (the
tables say when). The corpus is the sm_86 binaries of `tools/bench/real/pack_corpora.sh` (nvcc 13.0). For the stand-in test the compiler
part of the dev machine's CUDA 13.0 toolkit was copied into the round's scratch directory, named with `VGPU_NVCC`, and deleted with it.

| file | what |
|---|---|
| `launches.md` | HeCBench aidw launch by launch with the switch off and on (a stand-in card that keeps memory and computes nothing); what a sample would say of the four programs of #656 that are under the floor, and what a safe rule for each would need |
| `census.md` | every program once with `VGPU_HOST_GPU_SAMPLE_STARTED=0` (`d0`) and once with the rule (`new`), one build (`../sp75/census.py --form d0=BUILD,VGPU_HOST_GPU_SAMPLE_STARTED=0 --form new=BUILD`, with the build lock kept over the runs of 150 s instead of taken per run: another session held it in turns): the launches the rule touches, and the programs whose decisions differ (`../sp75/table.py --old d0 --new new`) |
| `rerun.md` | wall times of the programs that start the driver, the switch off against on, interleaved (`../sp75/rerun.py` with the same forms and the lock kept likewise) |
| `pf.py`, `pf_race.py`, `fixtime.c`, `pf.md` | Rodinia particlefilter_float with its clock seed fixed: `VGPU_HOST_GPU_SAMPLE_BLOCK=0` against the rule of Round sp75 per seed, and the program run by run at one and at four host threads |
| `start_model.py`, `start-model.md` | #662: the launches of 114 programs replayed under the start rule as it is and five others |
| `mutants.py`, `e2e-mutants.sh`, `mutants.txt` | mutants of the rule in the header against `tests/unit/test_host_gpu_logic.cpp`, and of the mode's part against `nvidia/tests/e2e/run_host_gpu_stub.sh` |
| `same-output.py`, `same-output.md` | stdout of programs with the mode off, in exact mode and in turbo mode (stand-in driver), and the plain simulator's counters and memory at exit, the build of PR #669 against this round's |
| `card-run.sh` | the card run: stages `build`, `fork`, `decisions`, `changed`, `keep`, `pf`, `rest`. Run on an A10G on 2026-11-15: `card.md` |
| `card.md` | what the card run showed (A10G): the fork, the programs changed, the keep list, the rest of the corpus |
| `rerun-interval-reduction.sh` | the card rerun of samples/interval and samples/reduction, which read 9 % and 7 % slower in the one-run pass: not yet run |

## Tests on server3

```
./build/test_host_gpu_logic                                   # 119/119 tests passed (115 before the round)
VGPU_NVCC=$PWD/cuda/bin/nvcc VGPU_BUILD_DIR=$PWD/build VGPU_HOST_GPU_TEST_PROFILE=nvidia/a10g bash src/nvidia/tests/e2e/run_host_gpu_stub.sh   # PASS, steps 1 to 44 (also with the default profile, nvidia/rtx3060)
python3 src/tools/bench/hostgpu/run.py --dry-run              # PASS: 0 checks failed (the forms exactd0 and turbod0)
python3 docs/perf-design/data/sp76/mutants.py --src src --work mut          # 21 of 21 killed (mutants.txt)
bash docs/perf-design/data/sp76/e2e-mutants.sh src mut-e2e cuda/bin/nvcc    # 5 of 5 killed (mutants.txt)
python3 docs/perf-design/data/sp76/same-output.py --corp corp --stub libstubcuda.so --old build-old --new build --programs ...   # 0 different outputs; counters and memory at exit the same (same-output.md)
```

`nvidia/tests/e2e/run_host_gpu.sh` (the card test) has a new step 13; it was not run on server3 (it needs a card) and passed on the A10G (`card.md`).

## The card run (`card-run.sh`; one A10G, 2026-11-15: results in `card.md`)

From the root of the tree on the rig, corpora in `~/corp`, build as in `../summary-2026-11-13/README.md` for the timed stages
(`SP76_BUILD`). What each stage should show:

```
docs/perf-design/data/sp76/card-run.sh all ~/sp76-out        # or stage by stage: build fork decisions changed keep pf rest
```

| stage | what runs | what to read | expected |
|---|---|---|---|
| `build` | card CI | `card-ci.txt` | unit tests pass; `e2e_host_gpu` passes with its step 13 (**the first fork of a process that holds the real driver**: `a sample with the driver up: host_gpu_sampled: sampled=2 sent=2 timeouts=0 ... after_driver=1 ... copies_left=0`); `e2e_host_gpu_stub` passes, steps 1 to 44. If step 13 fails with a wrong output, a hang or a driver error, stop: the rule's default must be off (`sample_started` in `src/runtime/host_gpu.cpp`) until it is understood; a copy that did not answer (`timeouts=1`) three times running is a busy machine, not that |
| `fork` | the two-kernel program of the stand-in test, 20 runs in exact mode and 20 in turbo, every launch sampled | `fork.txt` | `exact: 20 of 20 runs as expected, 0 with a copy that did not answer`, the same for turbo (a few lost copies on a busy rig are not a failure; a wrong output or a non-zero exit is); the kernel log's count of `NVRM` and `Xid` lines the same before and after; no zombie; `fork_s` of the second sample (the driver loaded) against the first: here 0.5 ms each with the stand-in, on the rig the first was 0.5 to 3.4 ms in Round sp60b and the second is not known (under 10 ms is no concern: it is inside the wait) |
| `decisions` | one logged turbo run of aidw, segmentationTreeThrust, histogram, mergeSort, silu | `decisions.txt`, `log-*.txt` | aidw: `AIDW_Kernel` sampled and sent, then `... MiB only the card held pulled before a sample`, `AIDW_Kernel_Tiled sampled with the driver up: ... (the last block stopped at the limit: at least) ... without a start: sent`, `4 launches ran on the card, 0 in the simulator`, `after_driver=1`. The three Samples: `sampled with the driver up` lines for the kernels first launched after the start, no `reason=the driver has been started`, `after_driver` over 0. silu: as in Round sp75's run (the start bought at the 31st launch, 20 on the card): nothing of it was changed |
| `changed` | aidw, turbo against `turbod0`, one run; segmentationTreeThrust, histogram, mergeSort, `sim,turbo,turbod0`, 10 runs | `aidw.txt`, `partly.md` | aidw `turbod0` 193 s with 3 of 4 on the card (Round sp75's card run), turbo about 45 s with 4 of 4 (what that run left after its 149 s launch; the two forms print the same). The Samples: turbo at or under `turbod0` (their kept first launches were 0.15 to 0.27 s each here, about 2.5 times that on the rig: histogram and mergeSort 0.3 to 0.6 s of 2.8 s each if all are sent, segmentationTreeThrust up to 1 s of 5.6 s); more launches on the card than Round sp75's 282 / 514, 34 / 49 and 58 / 68 |
| `keep` | the keep list, hotspot and silu, `sim,exact,exactd0,turbo,turbod0`, 3 runs | `keep.md` | no program more than 3 % slower than its `d0` form; outputs as they were (`identical`, particlefilter apart); the same launches on the card or more |
| `pf` | particlefilter_float, turbo against `turbob0` (the switch of Round sp75): the clock fixed at one host thread, six seeds; then run by run at four host threads and at one, the clock fixed; then at four with the clock free | `pf-fixed.md`, `pf-race4.md`, `pf-race1.md`, `pf-race4-free.md` | one host thread: turbo over turbob0 within 0.97 to 1.03 at every seed, one output a seed in every form, the mode's decisions the same. Four host threads, the clock fixed: the first launch of `find_index_kernel` in two groups (34 and 178 ms in Round sp75's run), several answers, and inside a group the forms' wall times equal; how many runs of a form fall in each group may differ (12 and 19 of 30 in Round sp75's run): if `turbo1` (the rule on, the environment of `turbob0`'s size) falls as `turbob0` does, the environment tilts the race and not the rule. One host thread: one group, one answer. The clock free: as with it fixed |
| `rest` | the rest of the corpus, `sim,turbo,turbod0`, 1 run | `rest.md`, `leftover.txt` | no program more than 3 % slower than `turbod0` (one run: a program over it is run again, 10 runs); `zombies: 0`, `processes of the corpus left: 0` |

What only the card can show: the fork against the real driver (correctness and cost), the time the launches now sent save (the stand-in
simulates them), and the pull before a sample at the card's real copy rates.
