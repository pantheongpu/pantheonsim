#!/usr/bin/env python3
"""Segment fits of the AMD transcendental results (docs/numerics/math-functions.md, "AMD transcendental").

Needs numpy and scipy and a dump of the card's results, written by
  amd_trans_probe <OP> dump 3f800000 8388608 > d.bin        (a binade of inputs: the mantissas of [1, 2))
Usage: amd_trans_fit.py d.bin <segment 0..31> [<squarer-truncation bits j, default 0 and 1>]

Segment = 2^18 consecutive mantissas (the top 5 mantissa bits select it). The result, as an integer in units of its own
ulp, is fitted with a linear program: the largest s such that  y + s <= P(f) <= y + 1 - s  for every input f of the
segment, P a real polynomial. s >= 0 means the results of the segment are the floor of that polynomial. P = a0 + a1 f +
a2 fq^2 + a3 f^3 (fq = f with its j low bits cleared) is the form measured on the RX 6800 (a cubic; the squarer operand
with one low bit cleared for rcp and rsq). Prints s per truncation. Our own code."""
import sys
import numpy as np
from scipy.optimize import linprog


def slack(ys, cols):
    n = len(ys)
    v = np.stack(cols, axis=1)
    a = np.vstack([np.hstack([-v, np.ones((n, 1))]), np.hstack([v, np.ones((n, 1))])])
    b = np.concatenate([-ys, ys + 1])
    c = np.zeros(v.shape[1] + 1)
    c[-1] = -1
    r = linprog(c, A_ub=a, b_ub=b, bounds=[(None, None)] * (v.shape[1] + 1), method="highs")
    return r.x[-1]


def main():
    r = np.fromfile(sys.argv[1], dtype="<u4")[: 1 << 23].astype(np.int64)
    seg = int(sys.argv[2])
    js = [int(a) for a in sys.argv[3:]] or [0, 1]
    n = 1 << 18
    y = r - (r.min() // (1 << 23)) * (1 << 23)          # the result's mantissa integer (one binade)
    ys = y[seg * n:(seg + 1) * n].astype(float)
    ys -= ys[0]
    f = np.arange(n)
    for j in js:
        fq = ((f >> j) << j).astype(float)
        cub = slack(ys, [np.ones(n), f / n, (fq / n) ** 2, (f / n) ** 3])
        print(f"segment {seg} squarer truncated by {j} bits: cubic slack {cub:.4f} ulp")
    print(f"segment {seg} plain quadratic: slack {slack(ys, [np.ones(n), f / n, (f / n) ** 2]):.4f} ulp")


if __name__ == "__main__":
    main()
