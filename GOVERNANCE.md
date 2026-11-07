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
  request is merged only once it passes.
- Bigger changes (a new vendor, a new library, anything that changes how the
  simulator behaves for existing programs) start as an issue or a Discussion, so
  the design can be agreed before the code is written.
- A few rules shape every decision, and pull requests are reviewed against them:
  - PantheonSim models what a GPU does, never how fast it does it. There is no
    performance model and there won't be one.
  - Programs run unmodified. A test that passes on a real GPU should pass on the
    simulator without being edited.
  - Anything the simulator does not support fails with an error that names what
    is missing. It never silently does the wrong thing.
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
