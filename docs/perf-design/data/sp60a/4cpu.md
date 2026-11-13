| program | runs | a s (spread) | c s (spread) | a/c | outputs |
|---|---:|---:|---:|---:|---|
| polybench/GEMVER | 7 | 0.471 (18%) | 0.471 (25%) | 1.000 | same |
| polybench/LU | 7 | 15.486 (1%) | 15.463 (2%) | 1.002 | same |
| polybench/MVT | 7 | 0.826 (4%) | 0.818 (5%) | 1.009 | same |
| polybench/SYR2K | 7 | 5.878 (1%) | 5.885 (2%) | 0.999 | same |
| polybench/SYRK | 7 | 3.099 (3%) | 3.098 (2%) | 1.000 | same |
| rodinia/particlefilter_float | 7 | 1.692 (14%) | 1.720 (11%) | 0.984 | DIFFER |

a/c: geometric mean 0.9990 over 6 programs of 0.3 s or more (smallest 0.984, largest 1.009; above 1.03: 0, below 0.97: 0)
