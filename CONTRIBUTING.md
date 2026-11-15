# Contributing to PantheonSim

Thank you for helping. This page says how to get a change in. How decisions are
made is in [GOVERNANCE.md](GOVERNANCE.md), the rules for taking part are in the
[Code of Conduct](CODE_OF_CONDUCT.md), and security problems go to
[SECURITY.md](SECURITY.md), not to a public issue.

## How PantheonSim is developed

PantheonSim has one maintainer, Saqib Khan. Most of its code, tests and
documentation were written with AI coding assistants (Anthropic's Claude),
directed and reviewed by the maintainer. That is why the project grew to its
present size in a few months and why pull requests here carry no second
reviewer. We do not ask you to trust the code because of who or what wrote it.
Here is the evidence that exists: about 900 ctest tests are registered, and CI
runs them on every pull request (the ROCm, PyTorch and rocBLAS ones on a runner
that has ROCm installed), under AddressSanitizer, UBSan and ThreadSanitizer too;
on that runner, a test that skips without a stated reason fails its job. CUDA
12.4, 12.8 and 12.9 run in a toolkit matrix beside the main CUDA 13 jobs, as do
nightly runs of rocBLAS's quick suite, PyTorch model checks, vLLM, and public
suites (cuda-samples, Rodinia, PolyBench, HeCBench) on simulated GPUs. Expected
outputs and the 17 "verified" GPU profiles were recorded by hand on physical
cards (14 NVIDIA profiles, among them the T4, RTX 3060, A10G, A100, H100 and
GH200, and 3 AMD ones, the MI325X, RX 6800 and RX 6700 XT; the full list is in
[docs/status.md](docs/status.md#gpu-profiles)); that comparison is not yet
repeated automatically. What does not work is listed in
[known limits](docs/status.md) with its issue. If you find a wrong result,
please open an issue: a wrong answer that reports no error is the bug we most
want to know about.

## Start with a question or an issue

- Questions and ideas: [Discussions](https://github.com/pantheongpu/pantheonsim/discussions).
- Bugs and gaps: [issues](https://github.com/pantheongpu/pantheonsim/issues). The
  templates ask for the exact error message and the profile, because the
  simulator names what it does not support and that message is usually the whole
  report.
- Anything large (a new vendor, a new library, a change to how existing programs
  behave) starts as an issue or a Discussion, so the design is agreed before the
  code is written.

## The rules every change is reviewed against

These are the project's fixed points (see [GOVERNANCE.md](GOVERNANCE.md)):

- PantheonSim models what a GPU does, never how fast it does it. There is no
  timing model and none will be added.
- Programs run unmodified. A test that passes on a real GPU should pass on the
  simulator without being edited.
- Anything unsupported fails with an error that names what is missing
  (instruction, API, kernel, profile). It never silently does the wrong thing.
  Known exceptions are listed in [docs/status.md](docs/status.md); none is open
  (the v0.2.0 release had two, fixed in v0.2.1:
  [#488](https://github.com/pantheongpu/pantheonsim/issues/488) and
  [#598](https://github.com/pantheongpu/pantheonsim/issues/598)).
- Runs are reproducible: with `VGPU_THREADS=1`, the same program and inputs give
  bit-identical results.
- Vendor interfaces are written from public documentation, and vendor code comes
  in only as published headers under open licenses.

## Build and test

You need Linux (WSL works), CMake 3.20 or newer and a C++20 compiler (g++ 12 or
newer, or clang++ 17). No GPU is needed.

```bash
make doctor   # checks the tools and says what to install
make          # installs what is missing (it asks first) and builds
make test     # builds if needed and runs the whole suite (ctest)
```

To run part of the suite: `ctest --test-dir build -R <regex> --output-on-failure`.
A full build takes a few minutes and the full suite takes a long time (about an
hour on a hosted runner), so run the tests that cover what you changed and let
CI run the rest. The checks in `tests/lint/` need no build.

Tests that compile CUDA or HIP programs need `nvcc` or `hipcc`; they skip when
the toolchain is missing, so a green local run without one says nothing about
them. Say in the pull request which you ran.

## What a change needs

- **A test that CI runs.** Every fix and feature comes with a test registered in
  `CMakeLists.txt`. Prefer a plain test that needs no framework and no GPU: a
  unit test under `tests/unit` or `nvidia/tests/unit` / `amd/tests/unit`, or an
  end-to-end script under `nvidia/tests/e2e` / `amd/tests/e2e`. Show that the
  test fails without your change.
- **Evidence for a behaviour claim.** The simulator is checked against real
  hardware wherever it can be: libraries are compared with NVIDIA's own on a
  physical GPU, profiles are read from cards, and traces are compared with
  recordings. If you say "this is what a GPU does", point to the document, the
  header, or the card output the claim comes from.
- **A loud failure for what you leave out.** If a change handles only part of
  an instruction or an API, make the rest fail with a message that names it,
  and add it to the gap list in [docs/gap-register.md](docs/gap-register.md) (open work is summarised in [TODO.md](TODO.md)).
- **Documentation that matches.** Update the page that describes the behaviour
  (`docs/`, `nvidia/docs/`, `amd/docs/`, or the README of that vendor), and the
  counts in [docs/status.md](docs/status.md) if you add or change a profile or a
  library. `tests/lint/` checks several of these (the settings table, the
  example settings file, the commands in the README).
- **A changelog line** for anything a user would notice, in
  [CHANGELOG.md](CHANGELOG.md) under the newest unreleased version.

### Changes that make it faster

A speed change must not change a single result bit or counter. Every fast path
keeps the old path behind a documented `VGPU_*` switch, and the golden tests
(`test_amd_gcn_same_bits`, `e2e_sass_tier` and the others named in
[docs/performance.md](docs/performance.md)) compare the two. Measure on an idle
machine with the alternating A/B mode in `tools/bench/bench.sh` and give the
median and the best of several runs, not one number. Timings on a shared hosted
runner move by tens of percent; do not argue from them.

### Adding or changing a GPU profile

A profile is data in `nvidia/profiles/` or `amd/profiles/`. Read the values from
a physical card with the characterization tools in `nvidia/tools` and
`amd/tools` and mark the profile `verified: true` only if you did; otherwise it
stays `verified: false` and its header comment says where every number came
from. Then: list it in `nvidia/docs/profiles.md` (or the AMD README), update
the counts in `docs/status.md`, and, only if it has a compute capability no profile
has yet, add a row for it to the machines in `.github/workflows/workloads.yml` (a
check fails when a compute capability has none). A new card of an existing capability
needs no row: the tests that run on every profile cover its data.

An NVIDIA profile also says which sum its tensor core makes and what that rests on (`numerics:`, `provenance:`, `evidence:`;
[nvidia/docs/profiles.md](nvidia/docs/profiles.md), "Numerics and provenance"). Tag a value `measured` only if a card of that
exact model was run and name the card, driver, toolkit and date in the evidence; a model of the same compute capability as a
measured card is `inferred`; with no card of the capability the sum stays `sequential` and `unknown`. A profile whose numerics
have no class does not load. Add the profile's row to `every_builtin_profile_tags_its_numerics` (`tests/unit/test_profile.cpp`)
and its pin to `kTensorSumPins` (`tests/unit/test_exec3.cpp`). Other keys may be tagged the same way when you have checked what
they rest on; leave a key untagged rather than guess.

## Third-party code

Do not add a file whose license forbids redistribution. NVIDIA's cuDNN headers
are downloaded at build time, not committed; `tests/lint/check_no_proprietary_nvidia.sh`
fails the build on proprietary NVIDIA terms. Published headers under an open
license can be added with their notice kept, and they go in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Pull requests

- One theme per pull request, kept small enough to review. Related fixes belong
  together; unrelated ones do not.
- Say what changes and why, and how you checked it. The template asks for this.
- CI runs on every pull request, and a pull request is merged once it passes.
  The heavy jobs are slow and some run on the maintainer's own runners (the
  PyTorch and ROCm tests need a machine with ROCm installed), so expect a long
  wait. A first pull request from a new contributor may need a maintainer to
  approve its CI run.
- A pull request that conflicts with `main` is not tested; rebase and push again
  ([docs/ci-merge-conflicts.md](docs/ci-merge-conflicts.md)).
- Contributions are accepted under the project's Apache-2.0 license, with no
  separate contributor agreement.
