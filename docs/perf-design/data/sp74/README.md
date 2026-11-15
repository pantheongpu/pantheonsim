# Round sp74: the tier's row handlers for the 64-bit forms (data)

The text is *Round sp74* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md); the design is
[../../13-amd-compiled-tier.md](../../13-amd-compiled-tier.md), section 13. server3 (Ryzen 7 PRO 8840U, 8 cores and 16 hardware threads, GCC
13.3, `Release` with the default link-time optimization unless a line says otherwise), every build under the machine's build lock,
`nice -n 19`, `-j3`. The host was quiet for every timing table: one job of this round at a time.

Trees on the host: `base` (origin/sp69-hotspot-start a33ead36), `noop` (the control: the base plus one function nothing calls in
`gcn_tier.cpp`), `w1` (step 1, commit a9dbf7e8), `w2` (step 2, not kept), `w3`, `w3A`, `w3B` (step 3 as first written and its parts),
`w4` (the handlers' final sources, commit 05a898c7), `w5` (the same after the fix of the tier's test, commit fdaca835), `w6` (`w5` with
`h_int_dpp` aligned to 64 bytes, not kept), `dev`, `mut` and `ct` (without LTO: development, the mutants, ctest), `diag` and `diag2`
(the base and the round's sources with `scripts/diagpatch.py` applied).

| file | what |
|---|---|
| `forms.md` | the handlers' time by instruction form, four inq tests: the base and the final sources |
| `bench.md` | the bench's kernel rows: the final library (all 62), each step, the variants of step 3 |
| `inq.md` | inq's tests: the base against the final library at one host thread and at the default, knob off and on, the control, the steps |
| `profile.md` | samples of four inq tests by class of host work, the base, the final library and the final library with blocks |
| `mutants.txt` | `scripts/mutants.py`: one edit of a row handler at a time, the tier's test rebuilt and expected to fail |
| `ctest.txt` | the AMD ctest set at the merged head and on the final sources, knob off and with `VGPU_AMD_BLOCKS=1` |
| `bound-f64-not-kept.patch` | step 2 (bound binary64 runs in blocks, with its test case), against commit a9dbf7e8 |
| `scripts/diagpatch.py`, `diagtab.py` | the diagnostic build (time-stamp ticks around each handler, by form; `SP74_DIAG=<file>`) and its table |
| `scripts/mk.sh`, `inq2.sh`, `inq2tab.py` | an LTO tree of the base plus changed files; two builds of inq's tests in turn; their table |
| `scripts/mutants.py` | the mutants |

The runners of rounds sp72 and sp73 (`../sp72/scripts/`: `build.sh`, `buildct.sh`, `bench.sh`, `run1.sh`, `inq.sh`, `inqprof.sh`,
`classify.py`, `sprof2sig.py`; `../sp73/scripts/`: `quick.sh`, `qall.sh`, `qtab.py`, `mkon2.sh`) were used as they are.

```
python3 -I diagpatch.py diag && ./build.sh /tmp/sp72-amd-tier/diag diag.log vgpu vgpuhip
taskset -c 10,11 ./inq.sh dg diag/build steepest_descent VGPU_THREADS=1 SP74_DIAG=/tmp/sp72-amd-tier/out/diag-steepest_descent.raw
python3 -I diagtab.py diag/build/libamdhip64.so.6.4 out/diag-steepest_descent.raw 45
./mk.sh w4                                                                    # head.tar plus push/, vgpu and vgpuhip
./qall.sh final 7 base noop w4 base:VGPU_AMD_BLOCKS=1 w4:VGPU_AMD_BLOCKS=1
CPUS=14,15 ./inq2.sh f1off 2 1 base w4 -- steepest_descent ks_hamiltonian initial_guess poisson
./inq2.sh fdoff 2 default base w4 -- laplacian gradient steepest_descent ks_hamiltonian initial_guess transform electrons poisson
./inqprof.sh w4/build w4 steepest_descent ks_hamiltonian initial_guess poisson
python3 -I mutants.py --src /tmp/sp72-amd-tier/mut --build /tmp/sp72-amd-tier/mut/build
```
