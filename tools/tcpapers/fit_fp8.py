#!/usr/bin/env python3
"""Fits the one-parameter-family sum of a tensor core to recorded FP8 dot products: how many of the vectors of a file does each
(chunk length L, grid F, output rounding) reproduce bit for bit?  Our own script; the family is the one the paper of Xie et al.
(arXiv 2511.10909) and Khattak and Mikaitis (arXiv 2512.07004) describe: K is cut into chunks of L products in order; a chunk is its
exact products and the accumulator, each term cut toward zero at 2^(e_max - F) where e_max is the largest of the terms' exponents
(a product's exponent is the sum of its factors'), summed exactly, and rounded once.

  fit_fp8.py <file from khattak_convert.py> <e4m3|e5m2> [--zero-c] [--order consecutive|halves] [--n 600] [--L 16,32]

Run it with python3 -I on a file that is data, not code. Needs no packages.
"""
import sys
from fractions import Fraction

EMIN = {"e4m3": -6, "e5m2": -14}

def f32(bits):
    import struct
    return struct.unpack("<f", struct.pack("<I", bits))[0]

def exp_of(x, emin):                         # floor(log2|x|), at least emin; zero has none
    if x == 0:
        return None
    m, e = abs(Fraction(x)), 0
    n, d = m.numerator, m.denominator
    e = n.bit_length() - d.bit_length()
    if Fraction(2) ** e > m:
        e -= 1
    return max(e, emin)

def trunc(q):                                # toward zero, Fraction -> int
    return int(q) if q >= 0 else -int(-q)

def round_out(v, rho):                       # v: exact Fraction; returns the float32 (or float16 value) as a Fraction
    if v == 0:
        return Fraction(0)
    sign = -1 if v < 0 else 1
    m = abs(v)
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if Fraction(2) ** e > m:
        e -= 1
    if rho == "rne16":
        e = max(e, -14); g = Fraction(2) ** (e - 10)
        q = m / g; r = int(q); rem = q - r
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and r & 1):
            r += 1
        return sign * r * g
    bits = 13 if rho == "rz13" else 23
    e = max(e, -126); g = Fraction(2) ** (e - bits)
    return sign * trunc(m / g) * g

def chunk(terms, c, F, e_zero, rho, emin):
    # terms: list of (exact value, exponent or None); c: (value, exponent or None)
    items = [(v, (e if e is not None else e_zero)) for v, e in terms] + [(c[0], c[1] if c[1] is not None else e_zero)]
    e_max = max(e for _, e in items)
    s = sum(trunc(v * Fraction(2) ** (F - e_max)) for v, _ in items)
    return round_out(Fraction(s) * Fraction(2) ** (e_max - F), rho)

def dot(a, b, c, L, F, rho, fmt, order):
    emin = EMIN[fmt]
    idx = list(range(32))
    if order == "halves":                    # chunk = the same positions of both 16-wide halves of K
        idx = [i for j in range(0, 16, L // 2) for i in list(range(j, j + L // 2)) + list(range(16 + j, 16 + j + L // 2))] if L < 32 else idx
    acc, ez = Fraction(c), (-21 if rho == "rne16" else -133)
    for s in range(0, 32, L):
        terms = []
        for i in idx[s:s + L]:
            ea, eb = exp_of(a[i], emin), exp_of(b[i], emin)
            terms.append((Fraction(a[i]) * Fraction(b[i]), None if ea is None or eb is None else ea + eb))
        acc = chunk(terms, (acc, exp_of(acc, -14 if rho == "rne16" else -126)), F, ez, rho, emin)
    return acc

def main():
    path, fmt = sys.argv[1], sys.argv[2]
    zero_c = "--zero-c" in sys.argv
    order = sys.argv[sys.argv.index("--order") + 1] if "--order" in sys.argv else "consecutive"
    f16out = ".fp16." in path
    limit = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 600
    rows = []
    for line in open(path):
        if len(rows) >= limit:
            break
        t = [int(x, 16) for x in line.split()]
        if len(t) != 66:
            continue
        v = [f32(x) for x in t]
        if any(x != x or x in (float("inf"), float("-inf")) for x in v):
            continue
        c = 0.0 if zero_c else v[64]
        if f16out:      # the paper's own program converts c to binary16 before the instruction
            import struct
            c = struct.unpack("<e", struct.pack("<e", c))[0]
        rows.append((v[:32], v[32:64], c, v[65]))
    print(f"{path}: {len(rows)} finite vectors")
    res = []
    Ls = [int(x) for x in sys.argv[sys.argv.index("--L") + 1].split(",")] if "--L" in sys.argv else [8, 16, 32]
    for L in Ls:
        for F in range(10, 29):
            for rho in (("rne16",) if f16out else ("rz23", "rz13")):
                ok = sum(1 for a, b, c, d in rows if dot(a, b, c, L, F, rho, fmt, order) == Fraction(d))
                res.append((ok, L, F, rho))
    res.sort(reverse=True)
    for ok, L, F, rho in res[:5]:
        print(f"  L={L:2d} F={F:2d} {rho}: {ok}/{len(rows)}")

main()
