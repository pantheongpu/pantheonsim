Data of round sp63 (the section "Round sp63" of `../../rounds/11-rounds-sp36-sp41.md`): the JIT's cold run, two builds side by side.

* `t1.md`, `t1.json`: one thread, 51 programs, sides Ba / Na (base and changed build, nothing set), Bbc / Nbc (`VGPU_JIT=1` cold), Nbw (warm), Nbs (the second run on an emptied cache). The changed build of this pass also started the compiler with `vfork` (not kept).
* `t1-recheck.md`, `t1-recheck.json`: the programs that start a compiler, and JACOBI1D, again on the final build, five runs.
* `td.md`, `td.json`: the default thread count, 18 programs, three runs, final build.
* `setup-probe.txt`: what `VGPU_JIT=1` costs short programs, and the same process with no unit (`scripts/setup_probe.py`). `spawn-fork-vfork.txt`: the compiler's spawn, `fork` against `vfork`.
* `mutants.log`: `scripts/mutants.sh`. `tests.log`: the ctest legs of the JIT and the run-loop tests.

The `.md` tables give each side's speed-up (its build's default median over the side's; "default B/N" is the base default over the changed default). The `.json` files hold per program the same-bits verdict, every run's wall by side, the warm fill's `[objects, bytes, compiles, compile ms, kept, declined]`, and the objects in the cache after each cold or second run. Scripts (written for server3; `real.py` of `tools/bench/real` and round sp57's `sp57.py` helpers are in `sp63.py`): `sp63.py` the driver, `table.py`, `setup_probe.py`, `mutants.sh`.
