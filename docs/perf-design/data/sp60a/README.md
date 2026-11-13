# Round sp60a: does launch-level result caching slow the default simulator? (data)

The text is *Round sp60a* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md). Scripts: [`scripts/`](scripts/README.md).

Box: server3 (Ryzen 7 8840U, 8 cores / 16 threads, 30 GiB, Ubuntu 24.04, g++ 13.3, no NVIDIA driver, shared with another session through the
machine-wide lock). Build: `origin/main` 1baa3ad3, `Release` with LTO, CUDA 13.0 headers. Profile `nvidia/a10g`. Programs: the sm_86 binaries of
PolyBench/GPU (20), Rodinia (12) and the reduced HeCBench list (34), with the arguments of `tools/bench/real/real.py`. Left out everywhere: the nine that
took 100 s or more in round sp59 (2MM, FDTD-2D, GRAMSCHM, aidw, aligned-types, atomicReduction, babelstream, base64e, binomial) and `fpdc-cuda` (its
second run in one directory does not end, in the default form too).

Forms: `a` nothing set; `b` `VGPU_MEMO=0`; `c` `VGPU_HOST_GPU=exact`; `m` `VGPU_MEMO=1`. Every run has `VGPU_QUIET=1`. A ratio `a/x` above 1 means the
default is slower than form `x`.

| file | what |
|---|---|
| `default.md`, `default.json` | default threads, not pinned, one program at a time: a, b, c, m alternating; 5 runs (3 when the first run is over 4 s, 1 over 30 s). 56 programs |
| `t1.md`, `t1.json` | `VGPU_THREADS=1`, pinned: a, b, c. 55 programs (`atomicCost` left out: one of its three runs was killed). 44 rows are of the first pass, which ran three programs side by side on three cores; the 11 rows that pass disturbed (runs killed or failing in all three forms at once, spreads over 10 %) are of a second pass, one program at a time |
| `memo-stats-default.md` | what the cache did in form `m` of the default-threads pass (`VGPU_MEMO_STATS_FILE`, last run of each program) |
| `identity-t1.md` | `VGPU_THREADS=1`, one run each of a, a again, b, c, m with `VGPU_MEMO_FINAL_HASH`, `VGPU_COUNTERS_FILE` (pid removed) and `VGPU_KERNEL_DIGEST`: are the output and the three files of every form those of the default? (With `VGPU_COUNTERS_FILE` set, host-GPU mode turns itself off, so form `c` is here the variable alone.) |
| `4cpu.md` | six programs on four logical CPUs of two cores (`taskset -c 0-3`, the shape of the A10G rig of round sp59), default threads, a and c, 7 runs each |
| `warm-t1.md` | `VGPU_THREADS=1`: off against `VGPU_MEMO=1` with a `VGPU_MEMO_DIR`: one cold run, then the median of two warm runs; the hits are the last warm run's |
| `sp59-baseline.md` | from round sp59's `bench.json` (branch `hostgpu-relaxed`): the forms that sent nothing to the card against the plain simulator, the plain simulator of one pass against another, the headline numbers with CORR and SYRK set to 1.00 |

The per-run records (`runs-*.jsonl`, 1.1 MB) are not kept: every table here is `tables.py`, `identity.py` or the inline summaries over them.
