# Simulator performance

Work in progress (round 7). This page will hold the measured profiles of both
engines; for now it documents the harness.

## The harness

```
tools/bench/bench.sh --build build                       # time every workload, SASS and PTX
tools/bench/bench.sh --ab buildA buildB --runs 5         # alternating A B A B ... on one program
tools/bench/bench.sh --check-counters --build build      # counts, not times, against tools/bench/expected
tools/bench/bench.sh mix --build build                   # instruction-class mix per workload
tools/bench/bench.sh --update-expected --build build     # rewrite tools/bench/expected/cuda-<release>.json
```

See `tools/bench/bench.sh --help` for the options. Python 3 standard library only
(run with `python3 -I`), and an nvcc whose major matches the shim in the build
directory: the workloads are `tools/bench/kernels/bench.cu`, compiled once for
sm_86 (SASS and PTX in one binary) and cached under `~/.cache/vgpu-bench`.
