Data of round sp64 (the section "Round sp64" of `../../rounds/11-rounds-sp36-sp41.md`): the JIT's compile and the run around it, at one and at sixteen threads.
Host server3 (Ryzen 7 PRO 8840U, 8 cores / 16 threads, g++ 13.3), every build and timed run under `flock /tmp/pantheonsim-builder.lock` and `nice -n 19`.
Builds: B = the base, `hostgpu-relaxed` f55c8e65; N = the first commit of the round (65b096b2: the plain loop while a compile runs, the precompiled header,
the compile log); P = N with the pieces and the jobs (not kept). Release, `-g1`, `VGPU_LTO=OFF`, `VGPU_JIT_MUTANTS=ON`.

* `timeline.md`: where a compile's time goes and when it starts, per program, at one and at sixteen threads (base build, `strace` of the compiler's processes).
* `probes.md`: the measurements that chose what to build: what a running compile costs the run (a compiler that never finishes against one that fails at once),
  the code `-O1` and `-fno-unroll-loops` make, the compiler's time per flag set and per part of the runtime header, a precompiled header, fifteen
  simulator threads against sixteen while a compile runs.
* `td.md`, `td.json`: sixteen threads, round sp63's 18 programs, sides of both builds B and N (`scripts/sp64.py`, 5 runs, 3 over one second);
  `td-recheck.md`: four of them again, 9 runs; `gemver-recheck.md`: GEMVER and JACOBI1D, 9 runs, at one and sixteen threads, B, N and P.
  `t1.md`, `t1.json`: one thread (`taskset -c 12-15`, two cores), 15 programs. `tdq.md`: the second run at sixteen threads on a cache that already
  has the precompiled header.
* `sequence.md`: runs one after the other on one cache (first, second, ... run), B, N and P, at sixteen and at one thread.
* `pieces-not-kept.patch`: the pieces and jobs of section 5 (P against N, the source only; its unit tests are described in the round), not kept.
* `mutants.log`, `tests.log`: `scripts/mutants_all.sh` on the kept build N, and the unit tests and run-loop ctest legs that ran.

The `.md` tables give each side's speed-up (its build's default median over the side's); "N/B cold" and the like are the changed build's speed-up over the base
build's, each against its own default. The `.json` files hold per program the same-bits verdict and every run's wall by side (`scripts/table64.py --compact`).
Scripts (written for server3; `real.py` of `tools/bench/real` must lie beside them): `sp64.py` the driver (round sp63's with the sides bp and bq),
`ab.py` the small A/B driver of the probes, `table64.py`, `timeline.py`, `ccost.sh`, `ccost2.sh`, `hdrcost.sh`, `pchprobe.sh`, `mutants.sh`,
`mutants_all.sh`, `bld.sh` (a build in lock holds of at most 280 s), `sleepcxx` and `failcxx` (the fake compilers of the probes).
