# METIS 5.1.0 (graph partitioning and fill-reducing orderings)

Source: the METIS 5.1.0 distribution by George Karypis (Regents of the
University of Minnesota), <https://github.com/KarypisLab/METIS> /
<https://karypis.github.io/glaros/files/sw/metis/metis-5.1.0.tar.gz>, licensed
under the Apache License 2.0 (`LICENSE.txt`). It is here for one caller:
`cusolverSpXcsrmetisndHost`, which NVIDIA documents as "a wrapper of
`METIS_NodeND`" linking "64-bit metis-5.1.0", so the permutation it returns
is METIS's. nvidia/src/cusolver_metis.cpp is the only code that calls it.

Only what `METIS_NodeND` reaches is kept: `libmetis/` without the k-way,
recursive-bisection, mesh and checking entry points, and five GKlib files
(`error`, `mcore`, `memory`, `random`, `timers`) with GKlib's headers. The
sources are compiled into libcusolver with hidden visibility, so none of
their symbols is exported.

Changes from the distribution, all marked in the files:

* `include/metis.h`: `IDXTYPEWIDTH` is 64 (NVIDIA's is the 64-bit build, and
  `csrmetisnd`'s `options` array is `int64_t`).
* `GKlib/random.c`: `rand()`/`srand()` are replaced by `random_r()`/
  `initstate_r()` on a thread-local state, which draw glibc's `rand()`
  sequence exactly. The original reseeds the host process's `rand()` on every
  call and shares one state between threads. The two draws of `gk_randint64`,
  unsequenced in C, are taken high word first.
