Scripts of round sp60a (run on the timing box, never on the development machine).

* `ab.py`: runs each program of the corpus (`tools/bench/real/real.py`'s list and arguments) under several environments of one build, alternating, and
  appends a record per run to a JSON-lines file. Forms: `a` default, `A` the same again, `b` `VGPU_MEMO=0`, `c` `VGPU_HOST_GPU=exact`, `L` the same with
  `VGPU_HOST_GPU_LOG=1`, `m` `VGPU_MEMO=1`. It takes the machine-wide builder lock itself and gives it up between runs after `--hold` seconds.
  `--identity` adds `VGPU_MEMO_FINAL_HASH`, `VGPU_COUNTERS_FILE` and `VGPU_KERNEL_DIGEST` to every run and records their hashes.
* `tables.py`: medians, spreads, the ratios `a/<form>` and their geometric means from such a file.
* `sp59_baseline.py`: reads round sp59's `bench.json` (branch `hostgpu-relaxed`): the forms that sent nothing to the card against the plain simulator,
  the plain simulator of one pass against another, and the headline numbers with the two affected programs set to 1.00.

```
python3 ab.py --build B --corp C --src S --tmp T --forms a,b,c,m --threads default --timeout 150 --skip <long programs> --out runs-default.jsonl
python3 ab.py ... --forms a,b,c --threads 1 --workers 3 --cpus "4;8;12" --out runs-t1.jsonl        # (three side by side: see the round text, not repeated)
python3 ab.py ... --forms a,b,c --threads 1 --cpus 4 --programs <eleven> --out runs-t1b.jsonl
python3 ab.py ... --forms a,A,b,c,m --threads 1 --cpus 4 --identity --runs 1 --out runs-identity-t1.jsonl
python3 ab.py ... --forms L --threads default --runs 1 --out runs-hglog.jsonl
python3 ab.py ... --forms a,c --threads default --cpus 0-3 --runs 7 --mid-runs 7 --programs SYRK,SYR2K,GEMVER,MVT,particlefilter_float,LU --out runs-4cpu.jsonl
python3 tables.py --runs runs-default.jsonl --json default.json > default.md
python3 sp59_baseline.py <sp59>/bench.json > sp59-baseline.md
```
