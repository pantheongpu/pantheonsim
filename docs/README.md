# Documentation index

One line per document. Start with the repository [README](../README.md); the design is in [ARCHITECTURE](../ARCHITECTURE.md), release notes on the
[releases page](https://github.com/pantheongpu/pantheonsim/releases), open work in [TODO](../TODO.md). Vendor-specific documents live in `nvidia/docs/` and `amd/` (see the
[ARCHITECTURE](../ARCHITECTURE.md) layout).

## Users

* [getting-started.md](getting-started.md): install, first run, the settings file.
* [config.md](config.md): every setting of `pantheonsim.yaml` and its `VGPU_*` variable.
* [cli.md](cli.md): every command and every `VGPU_*` environment variable.
* [status.md](status.md): what runs today, what does not, the GPU profiles.
* [toolchains.md](toolchains.md): compilers for your own programs (nvcc, hipcc).
* [driver-api.md](driver-api.md): running a driver-API program.
* [docker.md](docker.md): the container image.
* [github-action.md](github-action.md): CUDA and HIP tests in GitHub Actions.
* [ci-other-platforms.md](ci-other-platforms.md): other CI systems.
* [machine-simulator.md](machine-simulator.md): `vgpu shell`, the simulated machine.
* [diagnostics.md](diagnostics.md): counters, race detector, strict mode, errors.
* [telemetry.md](telemetry.md): device discovery, NVML, nvidia-smi, rocm-smi, fault injection.
* [registers.md](registers.md): the PCI and MMIO register model.
* [cooperative.md](cooperative.md): cooperative launch and the grid barrier.
* [matrix-numerics.md](matrix-numerics.md): which sum each profile's tensor core or matrix core makes, and whether that is measured on a card, published, or unmeasured.
* [numerics/tcpapers.md](numerics/tcpapers.md): three more papers on matrix-unit numerics (Fasi et al., Khattak and Mikaitis, FPRev) run on an RTX 3060 and the simulator: what matched, the recorded A100 / H100 / B200 test vectors, the one cuBLAS mismatch.
* [numerics/per-profile.md](numerics/per-profile.md): for every profile, which matrix numerics are measured on a card, inferred or unknown (generated from `vgpu info`).
* [numerics/fttn-simulators.md](numerics/fttn-simulators.md): FTTN on the simulator, every NVIDIA profile with a tensor core, both engines, against the paper's Table IV and the cards of ours.
* [numerics/fp8-vectors.md](numerics/fp8-vectors.md): the e4m3 and e5m2 `mma.sync` forms against 70,000 recorded vectors (RTX 1000 Ada, L40S, H100, H200, B200): the `ada` and `via_f16` sums, and why the PTX interpreter had disagreed with the SASS executor.
* [numerics/math-functions.md](numerics/math-functions.md): the math functions against the published accuracy results (CUDA and ROCm) and against four cards, the test set of `tools/numerics/`, and the approximate-math unit (`MUFU`) of an RTX 3060: the probe over every input, the model that reproduces it bit for bit, `VGPU_MUFU_EXACT`.
* [exact-threads.md](exact-threads.md): `VGPU_EXACT_THREADS`, blocks on several host threads with the bits of running them in order.
* [pantheon-workloads.md](pantheon-workloads.md): running the pantheon workloads.

## Contributors

* [../CONTRIBUTING.md](../CONTRIBUTING.md): how to change the project, tests, new profiles.
* [ci.md](ci.md): where CI jobs run, how long they take, how they fail.
* [ci-runners.md](ci-runners.md): which runner a job uses.
* [ci-merge-conflicts.md](ci-merge-conflicts.md): CI and merge conflicts.
* [ci-card-check.md](ci-card-check.md): the e2e programs on a real GPU (design, not enabled).
* [design-async-launch.md](design-async-launch.md): plan for kernels that run beside the host thread (not implemented).
* [found-bugs.md](found-bugs.md): bugs in other people's programs the simulator found, with evidence and unsent report drafts.
* [../TODO.md](../TODO.md): open work by area.
* [gap-register.md](gap-register.md): what is refused, stubbed or not modelled, with evidence.
* [history/TODO-done.md](history/TODO-done.md): finished work and the rev 5 audit notes.

## Performance

* [performance.md](performance.md): how fast it is, which switches matter, what did not pay off.
* [perf-design/README.md](perf-design/README.md): design studies (tiering, JIT, memoisation) and round notes.
* [perf-design/rounds/README.md](perf-design/rounds/README.md): the full round-by-round measurement log, indexed.
* [pgo.md](pgo.md): profile-guided release builds (opt-in).

## Internals

* [../ARCHITECTURE.md](../ARCHITECTURE.md): layers, engines, decisions.
* [../nvidia/docs/](../nvidia/docs): SASS, PTX, libraries, textures, CUPTI, profiles, dark API (one file per topic).
* [../amd/README.md](../amd/README.md): the AMD engines, libraries and gaps.
