# Real-program A/B (tools/bench/real)

`tools/bench` and `tools/bench-amd` time small synthetic kernels. This
directory times whole prebuilt CUDA programs on the simulator, with two builds
side by side, so a speed-up can be checked against real code and the next
target found.

Corpora (all built for sm_86 with `-cudart shared`, simulator side only; no
GPU card is used):

| corpus | programs | source |
|---|---|---|
| polybench | 20 PolyBench/GPU kernels, MINI dataset | `~/.cache/polybenchGpu/bin` |
| rodinia | 12 Rodinia apps, sizes scaled up from the card/sim runner so they run for seconds | `~/.cache/gpu-rodinia/bin` |
| hecbench | the 34 reduced HeCBench benchmarks (`reduced.txt`) | `~/.cache/HeCBench` |
| samples | the CUDA Samples v13.0 that pass on the simulator (`sim_main.txt`) | `~/.cache/cuda-samples` |

## Use

```
tools/bench/real/pack_corpora.sh /tmp/corp          # on the dev machine: copies files only
tar -C /tmp/corp -cf - . | ssh server3 'mkdir -p /tmp/corp && tar -C /tmp/corp -xf -'
# on the timing box (both builds need <build>/shim):
python3 tools/bench/real/real.py --before /tmp/base0/build --after /tmp/sp5-fin3/build \
    --corp /tmp/corp --cpu 0 --runs 5 --threads 1 --timeout 150 --json t1.json
python3 tools/bench/real/real.py ... --threads default      # second table
python3 tools/bench/real/profile.py --build /tmp/sp5-fin3/build --corp /tmp/corp \
    --programs polybench/LU,hecbench/bh-cuda --cpu 14     # where the time goes
```

* Each run is `taskset -c <cpu> <program>` with `VGPU_QUIET=1`,
  `VGPU_GPU=nvidia/rtx3060`, `LD_LIBRARY_PATH=<build>/shim` and (unless
  `--threads default`) `VGPU_THREADS=1`. BEFORE and AFTER alternate, one run at
  a time; the table shows median wall seconds and median user CPU seconds.
* The harness waits while the 1-minute load average is above `--max-load`
  (default 8). Timing under load is not usable: say so in a report if the box
  never calms down.
* Programs slower than `--long-secs` (30) in their first pair get only
  `--long-runs` (3) runs. `--skip` drops known timeouts; a run longer than
  `--timeout` is killed and shown as TIMEOUT.
* "outputs" is `same` when the normalised stdout/stderr (lines carrying
  timings, rates or the platform banner are dropped) and, for Rodinia, the
  files the program wrote hash equal in every run of both builds. "checks"
  lists the PASS/FAIL/Non-Matching/RMSE lines the program printed. A DIFFER
  row is a correctness question before it is a speed one.
* Wall time includes the host-side part of the program (CPU reference
  checks in PolyBench and many HeCBench programs): `profile.py` reports the
  share of cycles spent in the program's own binary, which no simulator
  change can remove.
* Very short programs (< 0.3 s) time process start-up, not simulation.
