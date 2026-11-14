# Round sp72: a compiled tier for AMD kernels, phase 1 (data)

The text is *Round sp72* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md); the design is
[../../13-amd-compiled-tier.md](../../13-amd-compiled-tier.md). server3 (Ryzen 7 PRO 8840U, 8 cores and 16 hardware threads, GCC 13.3, `Release`
with the default link-time optimization unless a line says otherwise), every build under the machine's build lock, `nice -n 19`, `-j3`. No
card, no `perf`. The host was shared with other sessions throughout (1-minute load 4 to 16): timed runs were pinned to one physical core
(`taskset -c 14,15`), sides alternated, and a table gives the *best* run of each side (another session's work makes a run slower, never
faster); the medians are in the same files.

Trees on the host (each a copy of the source with its own `build/`): `base` (origin/sp69-hotspot-start 77a8b8c8), `work` (this branch),
`noop` (the base plus one function nothing calls in `gcn_exec.cpp`: the control for code placement), `on` (the `work` library with the knob
on: the name of the variable it reads replaced, in the binary, by one the bench always sets, so that the two sides of the knob's A/B are byte
for byte the same code), `mut` and `ct` (this branch without LTO: the mutants, and ctest).

| file | what |
|---|---|
| `ceiling.md` | stage 1: nanoseconds per wave-instruction, the classes of host work for six bench rows and four inq tests, the native loops |
| `blocks-stats.txt` | `VGPU_AMD_BLOCKS_STATS` of the bench's ALU rows and of four inq tests: the share of instructions in blocks, run lengths, the hottest blocks' listings |
| `knob.md` | the knob's A/B (off against on, the same code): every bench row on both targets, one host thread, and the rows at all threads |
| `knob-off.md` | the base against this branch with the knob off, and the base against the no-op control |
| `inq.md` | the eight inq tests with the knob off and on: wall and CPU seconds, the output's checksum |
| `mutants.txt` | `scripts/mutants.py`: one edit of the block runner at a time, `test_amd_gcn_blocks_same_bits` rebuilt and expected to fail |
| `ctest.txt` | the AMD ctest set of this branch, with the knob off and with `VGPU_AMD_BLOCKS=1` in the environment |
| `scripts/classify.py` | samples of the instruction pointer into classes of host work (by function and by host instruction) |
| `scripts/sprof2sig.py` | round sp69's sampler output as the lines `classify.py` reads |
| `scripts/native_ceiling.cpp` | the bench's three ALU kernels as host loops over a wave's 64 lanes |
| `scripts/ab.py`, `scripts/both.py` | an A/B table from `tools/bench-amd/bench.py --json` files; the rows that are low in every one of several passes |
| `scripts/inqtab.py` | the inq table from `inqab.sh`'s lines |
| `scripts/mutants.py` | the mutants |
| `scripts/segv.c` | an `LD_PRELOAD` that prints where a `SIGSEGV` happened and, under it, the context a `SIGPROF` handler interrupted: how the crash of the stack-walking sampler with blocks on was traced to the unwinder (the round's caveats) |
| `scripts/*.sh` | the runners (paths are server3's): `build.sh`, `buildmut.sh`, `buildct.sh`, `bench.sh`, `run1.sh` (one bench workload), `prof.sh` and `inqprof.sh` (samples, classified), `inq.sh` (one inq test), `inqab.sh` (the tests off and on, alternating), `mkon.sh` (the `on` library), `stats.sh` (`blocks-stats.txt`) |

Commands (from the directory that holds the trees):

```
./build.sh /tmp/sp72-amd-tier/work work.log vgpu vgpuhip test_amd_gcn_blocks_same_bits
./mkon.sh
./bench.sh knobA --build work/build --build on/build --threads 1 --runs 9 --cpus 14,15     # -> out/bench-knobA.json
python3 -I ab.py off on out/bench-knobA.json out/bench-knobB.json
./bench.sh off1 --build base/build --build work/build --threads 1 --runs 9 --cpus 14,15
./bench.sh noop1 --build base/build --build noop/build --threads 1 --runs 9 --cpus 12,13
HZ=10000 ./prof.sh base/build base valu_int 30                                             # -> out/prof-base-valu_int.txt
./inqprof.sh base/build base steepest_descent ks_hamiltonian initial_guess poisson
./inqab.sh t1 2 1 steepest_descent ks_hamiltonian initial_guess poisson                    # -> out/inq-t1.txt
./run1.sh work/build valu_int VGPU_AMD_BLOCKS=1 VGPU_AMD_BLOCKS_STATS=3
python3 -I mutants.py --src /tmp/sp72-amd-tier/mut --build /tmp/sp72-amd-tier/mut/build
```

inq is the build of round sp70 (`/tmp/sp69-inq` on the host: ROCm 6.4.4 and inq's dependencies extracted from their debs, `all_unit_tests` for
gfx942), used read-only; `inq.sh` runs a test in a directory of this round.
