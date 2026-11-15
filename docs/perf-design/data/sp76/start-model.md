# Round sp76, #662: the rules that could buy the driver start, replayed over the launches of the corpus

`start_model.py collect` ran 114 programs of the corpora on server3 (the build of PR #669, `VGPU_THREADS=3`, profile `nvidia/a10g`) in turbo
mode with a start that is never bought, no sampling and no first-launch rule, and kept every launch's kernel, grid and simulator time, the
allocations and the host's writes. `start_model.py model` replays each record under the history estimate, the size rule and one of the
start rules below, and prices the launches: kept launches at the simulator's time, the start at 0.25 s, a card launch at its path (a card
kernel of 25 us, 1.1 ms more for a kernel's first, a push of the live memory after any launch the simulator ran or any host write). The
model has no sample and no first-launch rule; a program whose start a sampled launch buys is the same under every rule. Only programs for
which some rule buys the start are listed. The seconds are of the launches and the start, not of the program: a share of the program's time.

| rule | the start is bought by a launch the size rule would send with the driver up, when |
|---|---|
| `cur` | the account of simulator time spent on admitted launches has reached the start, and the launch saves its share (section 11 of the design): the rule as it is |
| `regret` | ... what the launches the size rule would have sent with the driver up would have saved has reached the start (the launches a kernel's history needs first are not counted), and the share |
| `acct1.5`, `acct2` | ... the account has reached 1.5 or 2 times the start, and the share |
| `kcount` | ... the account, the share, and as many launches again of this kernel as it has had, each saving what this one saves, repay the start |
| `pcount` | ... the account, the share, and as many admitted launches again as the program has had, each saving what this one saves, repay the start |

At the simulator's times of server3 (x 1) and scaled to the A10G rig's (x 2.75): on the rig the card run of Round sp75 bought silu's start
at the 31st launch (the model: the 26th), hotspot's at about the 300th (297th), FDTD-2D's at the 7th (7th).

At server3's own times (x 1; the table is not kept) the sums over the programs for which some rule buys the start are: cur 26.98 s, regret 31.07 s, acct1.5 28.56 s, acct2 29.87 s, kcount 28.69 s, pcount 27.21 s.

Simulator times x 2.75. Seconds of the launches and the start under each rule (the launch that buys the start / launches sent); `sim`: all launches simulated.

| program | launches | sim s | cur | regret | acct1.5 | acct2 | kcount | pcount |
|---|---:|---:|---|---|---|---|---|---|
| hecbench/aop-cuda | 301 | 2.483 | 0.889 (9 / 291) | 1.329 (72 / 228) | 0.889 (9 / 291) | 0.889 (9 / 291) | 1.290 (66 / 234) | 0.992 (24 / 276) |
| hecbench/asyncAllocation-cuda | 800 | 4.481 | 1.106 (49 / 752) | 1.205 (72 / 729) | 1.217 (74 / 727) | 1.327 (95 / 706) | 1.145 (58 / 743) | 1.145 (58 / 743) |
| hecbench/atomicAggregate-cuda | 6 | 5.935 | 2.329 (2 / 4) | 2.329 (2 / 4) | 2.329 (2 / 4) | 2.329 (2 / 4) | 2.329 (2 / 4) | 2.329 (2 / 4) |
| hecbench/atomicReduction-cuda | 496 | 158.722 | 3.000 (2 / 488) | 2.933 (3 / 488) | 3.000 (2 / 488) | 2.933 (3 / 488) | 3.000 (2 / 488) | 3.000 (2 / 488) |
| hecbench/radixsort2-cuda | 600 | 3.447 | 0.588 (55 / 455) | 0.687 (73 / 440) | 0.687 (73 / 440) | 0.817 (97 / 420) | 1.079 (145 / 380) | 0.588 (55 / 455) |
| hecbench/silu-cuda | 50 | 0.516 | 0.522 (26 / 25) | 0.516 (- / 0) | 0.637 (37 / 14) | 0.516 (- / 0) | 0.516 (- / 0) | 0.522 (26 / 25) |
| polybench/FDTD-2D | 1500 | 135.806 | 0.807 (7 / 1494) | 1.144 (11 / 1490) | 0.807 (7 / 1494) | 0.807 (7 / 1494) | 1.220 (12 / 1489) | 0.807 (7 / 1494) |
| polybench/JACOBI2D | 40 | 0.815 | 0.522 (16 / 25) | 0.656 (22 / 19) | 0.630 (21 / 20) | 0.769 (27 / 14) | 0.856 (31 / 10) | 0.542 (17 / 24) |
| polybench/LU | 4094 | 45.213 | 0.658 (8 / 4087) | 0.941 (14 / 4081) | 0.848 (12 / 4083) | 0.941 (14 / 4081) | 0.753 (10 / 4085) | 0.658 (8 / 4087) |
| rodinia/gaussian | 1022 | 5.635 | 0.556 (26 / 997) | 0.638 (32 / 991) | 0.662 (34 / 989) | 0.787 (44 / 979) | 0.583 (28 / 995) | 0.556 (26 / 997) |
| rodinia/hotspot | 1000 | 1.001 | 0.575 (297 / 704) | 0.575 (297 / 704) | 0.647 (373 / 628) | 0.767 (493 / 508) | 0.575 (297 / 704) | 0.575 (297 / 704) |
| rodinia/lud | 190 | 1.459 | 0.547 (18 / 173) | 0.649 (24 / 167) | 0.649 (24 / 167) | 0.789 (33 / 158) | 0.598 (21 / 170) | 0.547 (18 / 173) |
| rodinia/particlefilter_float | 36 | 4.585 | 1.042 (9 / 28) | 1.558 (13 / 24) | 1.042 (9 / 28) | 1.042 (9 / 28) | 1.042 (9 / 28) | 1.042 (9 / 28) |
| samples/StreamPriorities | 4 | 1.459 | 1.176 (2 / 2) | 1.072 (3 / 2) | 1.072 (3 / 2) | 1.072 (3 / 2) | 1.176 (2 / 2) | 1.176 (2 / 2) |
| samples/alignedTypes | 384 | 109.646 | 7.362 (2 / 360) | 7.339 (3 / 360) | 7.362 (2 / 360) | 7.362 (2 / 360) | 7.362 (2 / 360) | 7.362 (2 / 360) |
| samples/convolutionSeparable | 34 | 12.247 | 1.750 (3 / 30) | 1.749 (4 / 30) | 1.750 (3 / 30) | 1.750 (3 / 30) | 1.750 (3 / 30) | 1.750 (3 / 30) |
| samples/histogram | 68 | 13.453 | 1.843 (3 / 58) | 1.794 (5 / 59) | 1.794 (5 / 59) | 1.794 (5 / 59) | 1.843 (3 / 58) | 1.843 (3 / 58) |
| samples/interval | 100 | 16.315 | 0.581 (3 / 98) | 0.905 (5 / 96) | 0.743 (4 / 97) | 0.905 (5 / 96) | 0.581 (3 / 98) | 0.581 (3 / 98) |
| samples/jacobiCudaGraphs | 2955 | 23.986 | 0.601 (35 / 2920) | 0.682 (45 / 2910) | 0.722 (50 / 2905) | 0.848 (65 / 2890) | 0.666 (43 / 2912) | 0.666 (43 / 2912) |
| samples/mergeSort | 49 | 8.764 | 3.212 (9 / 37) | 3.212 (9 / 37) | 3.212 (9 / 37) | 3.212 (9 / 37) | 3.212 (9 / 37) | 3.212 (9 / 37) |
| samples/radixSortThrust | 600 | 38.784 | 0.723 (6 / 593) | 0.975 (11 / 590) | 0.805 (9 / 592) | 0.975 (11 / 590) | 0.805 (9 / 592) | 0.723 (6 / 593) |
| samples/reduction | 201 | 9.525 | 1.088 (6 / 98) | 1.263 (10 / 96) | 1.263 (10 / 96) | 1.351 (12 / 95) | 1.088 (6 / 98) | 1.088 (6 / 98) |
| samples/segmentationTreeThrust | 514 | 18.522 | 5.847 (19 / 259) | 6.194 (41 / 243) | 5.847 (19 / 259) | 5.847 (19 / 259) | 6.194 (41 / 243) | 5.847 (19 / 259) |
| samples/simpleCudaGraphs | 24 | 1.689 | 0.591 (5 / 20) | 0.858 (9 / 16) | 0.724 (7 / 18) | 0.858 (9 / 16) | 0.591 (5 / 20) | 0.591 (5 / 20) |
| samples/simpleHyperQ | 65 | 31.457 | 2.229 (3 / 60) | 2.229 (4 / 60) | 2.229 (3 / 60) | 2.229 (3 / 60) | 2.229 (3 / 60) | 2.229 (3 / 60) |
| samples/sortingNetworks | 290 | 14.096 | 1.792 (3 / 282) | 2.670 (5 / 280) | 1.792 (3 / 282) | 2.179 (4 / 281) | 1.792 (3 / 282) | 1.792 (3 / 282) |
| samples/streamOrderedAllocation | 21 | 0.274 | 0.512 (21 / 1) | 0.274 (- / 0) | 0.274 (- / 0) | 0.274 (- / 0) | 0.274 (- / 0) | 0.274 (- / 0) |
| samples/transpose | 808 | 9.164 | 0.735 (40 / 755) | 0.806 (51 / 744) | 0.857 (59 / 736) | 0.984 (78 / 717) | 0.793 (49 / 746) | 0.793 (49 / 746) |

Sums: cur 43.18 s, regret 47.18 s, acct1.5 44.49 s, acct2 46.35 s, kcount 45.34 s, pcount 43.23 s
More than 3 % over `cur` under `regret`: hecbench/aop-cuda +50 %, hecbench/asyncAllocation-cuda +9 %, hecbench/radixsort2-cuda +17 %, polybench/FDTD-2D +42 %, polybench/JACOBI2D +26 %, polybench/LU +43 %, rodinia/gaussian +15 %, rodinia/lud +19 % and 9 more
More than 3 % over `cur` under `acct1.5`: hecbench/asyncAllocation-cuda +10 %, hecbench/radixsort2-cuda +17 %, hecbench/silu-cuda +22 %, polybench/JACOBI2D +21 %, polybench/LU +29 %, rodinia/gaussian +19 %, rodinia/hotspot +13 %, rodinia/lud +19 % and 6 more
More than 3 % over `cur` under `acct2`: hecbench/asyncAllocation-cuda +20 %, hecbench/radixsort2-cuda +39 %, polybench/JACOBI2D +48 %, polybench/LU +43 %, rodinia/gaussian +41 %, rodinia/hotspot +34 %, rodinia/lud +44 %, samples/interval +56 % and 6 more
More than 3 % over `cur` under `kcount`: hecbench/aop-cuda +45 %, hecbench/asyncAllocation-cuda +4 %, hecbench/radixsort2-cuda +84 %, polybench/FDTD-2D +51 %, polybench/JACOBI2D +64 %, polybench/LU +14 %, rodinia/gaussian +5 %, rodinia/lud +9 % and 4 more
More than 3 % over `cur` under `pcount`: hecbench/aop-cuda +12 %, hecbench/asyncAllocation-cuda +4 %, polybench/JACOBI2D +4 %, samples/jacobiCudaGraphs +11 %, samples/transpose +8 %
