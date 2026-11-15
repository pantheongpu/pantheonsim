# Governance

PantheonSim is a small project with one maintainer. This file says who decides
what, how changes get in, and how that will change as more people contribute.

## Roles

- **Maintainer:** Saqib Khan ([@saqibkh](https://github.com/saqibkh)). Reviews and
  merges pull requests, decides on design questions, and owns releases and the
  GitHub Action.
- **Contributors:** anyone who opens an issue, answers a question in
  [Discussions](https://github.com/pantheongpu/pantheonsim/discussions), or sends a
  pull request.

## How changes are made

- Every change goes through a pull request. `main` is protected, and that
  applies to the maintainer too. CI runs on every pull request, and a pull
  request is merged only once its required checks pass (see below).
- The required checks are these jobs of the `CI` workflow
  (`.github/workflows/ci.yml`):
  - `build + unit tests`
  - `pantheon workloads`
  - `AMD code objects and CDNA decoding`
  - `image / find the toolchain images` (the job the two above wait for)

  The first three run on GitHub's hosted runners. `main`'s branch protection
  requires these checks, and the CI badge in the README shows the `CI`
  workflow on `main`. A change to this list changes both this file and the
  branch protection.
- The heavy jobs run on the project's self-hosted runners, which share their
  memory: `SASS on every architecture`, `ollama on every GPU`, `ollama on
  <AMD GPU>` (one per AMD GPU), `PyTorch, ROCm and the rest of the suite on the
  pantheonsim.com runner`, `ASan + UBSan` and `ThreadSanitizer`. They run on
  every pull request too and their results are shown on it, but they are not
  required: they can fail because a runner ran out of memory or went away,
  not because of the change. The maintainer reads every failure before
  merging. A failure that comes from the change blocks the merge like a
  required check would, and a failure that points to a real bug gets an
  issue. A job is never deleted or skipped to make CI green.
- Bigger changes (a new vendor, a new library, anything that changes how the
  simulator behaves for existing programs) start as an issue or a Discussion, so
  the design can be agreed before the code is written.
- A few rules shape every decision, and pull requests are reviewed against them:
  - PantheonSim models what a GPU does, never how fast it does it. There is no
    performance model and there won't be one.
  - Programs run unmodified. A test that passes on a real GPU should pass on the
    simulator without being edited.
  - Anything the simulator does not support fails with an error that names what
    is missing. It never silently does the wrong thing. There is no known open
    violation. The v0.2.0 release had two, fixed in v0.2.1: `shfl.sync` returned 0
    without an error for a lane still in a long loop
    ([#488](https://github.com/pantheongpu/pantheonsim/issues/488)), and
    `cudaDeviceCanAccessPeer` reported peer access that a kernel then could not use
    ([#598](https://github.com/pantheongpu/pantheonsim/issues/598)). A violation
    is listed in [docs/status.md](docs/status.md) until it is fixed in a release.
  - Runs are reproducible: with `VGPU_THREADS=1`, the same program and inputs
    give bit-identical results (see ARCHITECTURE.md).
  - Vendor interfaces are written from public documentation. Vendor code comes
    in only as published headers under open licenses, with their notices kept.

## Becoming a maintainer

Someone who has made sustained, good contributions (code, reviews, or help for
users) can be invited to become a maintainer. New maintainers are added to this
file by pull request. Once there is more than one maintainer, design decisions
that the maintainers don't agree on are settled by discussion in the open, and
the lead maintainer makes the final call if that fails.

## License

PantheonSim is licensed under Apache-2.0. Contributions are accepted under the
same license, with no separate contributor agreement.

## Changing this document

Changes to governance go through a pull request to this file, like any other
change.
