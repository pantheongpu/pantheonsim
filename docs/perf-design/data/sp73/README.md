# Round sp73: blocks for AMD kernels, phase 2 (data)

The text is *Round sp73* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md); the design is
[../../13-amd-compiled-tier.md](../../13-amd-compiled-tier.md), section 12. server3 (Ryzen 7 PRO 8840U, 8 cores and 16 hardware threads, GCC
13.3, `Release` with the default link-time optimization unless a line says otherwise), every build under the machine's build lock,
`nice -n 19`, `-j3`. The host was quiet: tables were taken with nothing else of this round running.

Trees on the host: `base` (origin/sp69-hotspot-start 3e4c7f3f), `noop` (the control: the base plus one function nothing calls in
`gcn_exec.cpp`), `work` (round sp72 as committed, after the merge of the base), `p2` and `p3` (this round's steps; `p3` is the final
sources), `basewi` (the base with `Wave` 16 bytes and `Inst` 8 bytes larger), `mut` and `ct` (the final sources without LTO: the mutants,
ctest).

| file | what |
|---|---|
| `steps.md` | every kernel row of the bench: control, knob off, blocks, blocks with bound runs, against the base |
| `knob-off.md` | how the knob-off costs were found: the builds on the way, row by row, the stack-shift test, the API rows |
| `inq.md` | inq's tests with the knob off and on: one host thread (three passes) and the default (two) |
| `blocks-stats.txt` | `VGPU_AMD_BLOCKS_STATS` of bench rows and four inq tests on the final build |
| `mutants.txt` | `scripts/mutants.py`: one edit of the block runner (or of the binding in the tier) at a time, the test rebuilt and expected to fail |
| `ctest.txt` | the AMD ctest set, knob off and with `VGPU_AMD_BLOCKS=1` in every test's environment |
| `scripts/quick.sh`, `qall.sh`, `qtab.py` | one bench row on several builds in turn (best device CPU seconds of n runs, pinned); all kernel rows; the table |
| `scripts/mutants.py` | the mutants of the runner's second form |
| `scripts/mkon2.sh` | a copy of a build with the knob on (for `bench.py`, which takes builds, not environments) |
| `scripts/stackshift.sh`, `tlspad.c` | a row's rate with every thread's stack moved by 16 to 3,856 bytes (an `LD_PRELOAD` with that much thread-local storage) |

The runners of round sp72 (`../sp72/scripts/`: `build.sh`, `bench.sh`, `run1.sh`, `inq.sh`, `inqab.sh`, `stats.sh`, `ab.py`, `inqtab.py`) were
used as they are, with `BUILD=p3/build` for the last three.

```
./quick.sh 7 valu_f64 gfx1030 amd/rx6800 base work p3 noop          # one row, four builds
./qall.sh final 7 base noop p3 p3:VGPU_AMD_BLOCKS=1,VGPU_AMD_BLOCKS_BOUND=0 p3:VGPU_AMD_BLOCKS=1
python3 -I qtab.py out/q-final.txt control=noop "knob off=p3" blocks=p3:VGPU_AMD_BLOCKS=1,VGPU_AMD_BLOCKS_BOUND=0 "blocks + bound=p3:VGPU_AMD_BLOCKS=1"
./stackshift.sh valu_f64 gfx1030 amd/rx6800 base work noop
BUILD=p3/build CPUS=14,15 ./inqab.sh t1 3 1 steepest_descent ks_hamiltonian initial_guess poisson
python3 -I mutants.py --src /tmp/sp72-amd-tier/mut --build /tmp/sp72-amd-tier/mut/build
```
