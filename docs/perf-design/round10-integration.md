# Round 10: the three workstreams together

`speed-round10` = origin/main 89d6bcd5 + step 0 (measurements) + `speed10-load` (first-launch cost of a fatbin) + `speed10-small` (PTX atomics with
one lookup). Built Release on the dedicated rig (c7i.4xlarge, nothing else running) and compared with the build of main from before the round
(`~/sim/build`, source 89d6bcd5). Raw files: `data/round10/integration/`.

**Tests** (targeted, 22 of 22 passed on the merged build): test_decode_same_bits, test_unsupported_memo, test_pick_cubin, test_atom_rmw,
test_sass_decode, test_sass_code_at, test_tier, test_fastpath, e2e_sass_path, e2e_sass_fastpath, e2e_sass_tier, e2e_sass_archs, bench_counters,
bench_counters_tier, e2e_cupti_trace, e2e_cupti_trace_driver, e2e_cupti_memory, and the e2e_ptx_* forms. NOT run here: the whole ctest, the
sanitizer legs (CI), and the other PyTorch groups beyond the quick tier.

**PyTorch quick sweep** (4 cores, tier off, same checks, same ok counts in every group; `compile` fails identically in both, no Inductor on the rig):
104.1 s before, 59.0 s after (1.76x).

**First launch of a new fatbin in one process** (`data/round10/mini2.py`, rtx5090 profile), seconds: add 0.146 -> 0.026, mul 0.091 -> 0.015,
relu 0.124 -> 0.021, sum 0.219 -> 0.041, softmax 0.163 -> 0.061, layer_norm 0.058 -> 0.026.

**Harness A/B** (7 runs, pinned to CPU 4; counters identical): PTX `atomics` 2.04 -> 3.47 Minst/s (1.70x). SASS `atomics` 5.99 -> 6.18, `launch` 384 -> 397 k/s,
`warp` 19.29 -> 19.71. Within noise: `int_alu`, `mem` (-1.4% SASS), `divergent`, `api`, `startup`. Slightly slower: SASS `shared` 13.35 -> 13.02 (-2.5%),
PTX `shared` 9.33 -> 8.98 (-3.8%). Both workstreams saw build-to-build code-layout effects of this size on `shared`; not chased.
