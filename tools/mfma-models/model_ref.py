#!/usr/bin/env python3
"""A second writing of the AMD matrix-core models of Khattak, Mikaitis and Graziani (arXiv 2609.14845; github.com/north-numerical-computing/
MATLAB-tensor-core, BSD-2-Clause, read at commit 6fd05208), in exact rational arithmetic, and the vector generator that holds
amd/src/mfma_models.hpp to it. It is the author's reading of the MATLAB files, not their output: MATLAB and Octave are not at hand,
so no number below was produced by the paper's own code.

  model_ref.py gen <file> [--n N] [--seed S]     write test vectors with this file's answers
Standard library only.
"""
import argparse
import math
import random
import struct
from fractions import Fraction as F

INF = float("inf")


def f32_round(x):
    """Fraction -> nearest binary32 (ties to even, subnormals kept), as a float; inf past the range."""
    if x == 0:
        return 0.0
    neg = x < 0
    x = abs(x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if x < F(2) ** e:
        e -= 1
    ulp = max(e - 23, -149)
    q = x / F(2) ** ulp
    n = q.numerator // q.denominator
    rem = q - n
    if rem > F(1, 2) or (rem == F(1, 2) and n % 2):
        n += 1
    r = F(n) * F(2) ** ulp
    v = INF if r >= F(2) ** 128 else float(r)
    return -v if neg else v


def f32_add(x, y):
    s = F(x) + F(y)
    if s == 0:
        return -0.0 if (math.copysign(1, x) < 0 and math.copysign(1, y) < 0) else 0.0
    return f32_round(s)


# ---- formats ---------------------------------------------------------------------------------------------------------------
def decode(fmt, bits):
    if fmt == "h":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    if fmt == "b":
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    if fmt == "F":
        return struct.unpack("<f", struct.pack("<I", ((bits >> 18) & 1) << 31 | ((bits >> 10) & 255) << 23 | (bits & 1023) << 13))[0]
    s = -1.0 if bits & 0x80 else 1.0
    if fmt == "e":   # e4m3 FNUZ: bias 8, no infinity, 0x80 is the NaN
        ex, m = (bits >> 3) & 15, bits & 7
        return s * ((m / 8) * 2.0 ** -7 if ex == 0 else (1 + m / 8) * 2.0 ** (ex - 8))
    ex, m = (bits >> 2) & 31, bits & 3   # g: e5m2 FNUZ, bias 16
    return s * ((m / 4) * 2.0 ** -15 if ex == 0 else (1 + m / 4) * 2.0 ** (ex - 16))


WIDTH = {"h": 4, "b": 4, "F": 5, "e": 2, "g": 2}
EMIN = {"h": -14, "b": -126, "F": -126, "e": -7, "g": -15}


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32val(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def exp2floor(x):
    """floor(log2 |x|) of a nonzero float."""
    return math.frexp(abs(x))[1] - 1


# ---- the models ------------------------------------------------------------------------------------------------------------
def block_size(model, ta):
    if model == "cdna1":
        return 2 if ta == "b" else 4
    if model == "cdna2":
        return 4
    return 4 if ta == "F" else 16 if ta in "eg" else 8


def cdna1_block(a, b, c):
    if abs(c) == INF:
        return c
    prods = [F(x) * F(y) for x, y in zip(a, b) if x != 0 and y != 0]
    if not prods and c == 0:
        return 0.0
    return f32_round(sum(prods, F(0)) + F(c))


def limit_check(model, a, b):
    """The products may not reach 2^128: the model's prd_limit. Returns None or the answer."""
    ps = [x * y for x, y in zip(a, b) if not (model == "cdna3" and (x == 0 or y == 0))]
    hi = any(p >= 2.0 ** 128 for p in ps)
    lo = any(p <= -(2.0 ** 128) for p in ps)
    if not (hi or lo):
        return None
    if hi and lo:
        return float("nan")
    m = max(ps)
    return float("nan") if m == 0 else (INF if m > 0 else -INF)


def cdna2_block(ta, tb, a, b, c):
    r = limit_check("cdna2", a, b)
    if r is not None:
        return r
    if abs(c) == INF:
        return c
    min_a, min_b = 2.0 ** EMIN[ta], 2.0 ** EMIN[tb]
    prod = []
    for x, y in zip(a, b):
        if abs(x) < min_a or abs(y) < min_b:
            x = y = 0.0
        p = f32_round(F(x) * F(y)) if x * y != 0 else 0.0
        prod.append(0.0 if abs(p) < 2.0 ** -126 else p)

    def flush(v):
        return math.copysign(0.0, v) if abs(v) < 2.0 ** -126 else v

    def tree(p):
        if len(p) == 1:
            return p[0]
        h = len(p) // 2
        return flush(f32_add(tree(p[:h]), tree(p[h:])))

    return flush(f32_add(flush(c), tree(prod)))


def cdna3_block(ta, tb, a, b, c):
    r = limit_check("cdna3", a, b)
    if r is not None:
        return r
    if abs(c) == INF:
        return c
    fp8 = ta in "eg"
    items = []   # (significand, exponent, negative)
    for x, y in zip(a, b):
        if x == 0 or y == 0:
            if fp8:
                items.append((F(0), -1024, False))
            continue
        ex = max(exp2floor(x), EMIN[ta])
        ey = max(exp2floor(y), EMIN[tb])
        sig = F(x) / F(2) ** ex * (F(y) / F(2) ** ey)
        items.append((sig, ex + ey, sig < 0))
    if not fp8 and not items:
        return 0.0 if c == 0 else c
    neab = 1

    def gsum(group):   # the products aligned to the largest exponent, each cut toward zero; (exponent, sum)
        if not group:
            return (0, 0)
        mx = max(e for _, e, _ in group)
        total = 0
        for sig, e, neg in group:
            v = int(abs(sig) * 2 ** 23) << neab
            sh = mx - e
            v = v >> sh if sh <= 31 else 0
            total += -v if neg else v
        return (mx, total)

    def two(p1, p2, eab1, eab2, check):   # (exp, signed sum) -> (exp, signed sum); neab may grow
        nonlocal neab
        (e1, s1), (e2, s2) = p1, p2
        n1, n2 = s1 < 0, s2 < 0
        m1, m2 = abs(s1), abs(s2)
        mx = max(e1, e2)
        sh = abs(e1 - e2)
        if sh > 512:
            sh = 0
        if e1 < mx:
            neab += eab1
            m1 <<= eab1
            m2 <<= eab1
            m1 = (m1 >> sh) + (1 if (m1 & ((1 << sh) - 1)) and n1 else 0)
        elif e2 < mx:
            neab += eab2
            m1 <<= eab2
            m2 <<= eab2
            adj = bool(m2 & ((1 << sh) - 1)) and n2
            if check and e2 < mx - 25:
                adj = False
            m2 = (m2 >> sh) + (1 if adj else 0)
        return (mx, (-m1 if n1 else m1) + (-m2 if n2 else m2))

    if fp8:
        pe = gsum(items[0::2])
        po = gsum(items[1::2])
        prod = two(pe, po, 0, 0, False)
    else:
        prod = gsum(items)
    if c == 0:
        cp = (-126, 0)
    else:
        ce = max(exp2floor(c), -126)
        cm = int(abs(F(c)) / F(2) ** ce * 2 ** (23 + neab))
        cp = (ce, -cm if c < 0 else cm)
    e, total = two(prod, cp, 8, 0, fp8)
    return f32_round(F(total) * F(2) ** (e - 23 - neab))


def model_dot(model, ta, tb, a, b, c):
    ps = [x * y for x, y in zip(a, b)]
    allv = ps + [c]
    if any(v != v for v in allv) or (INF in allv and -INF in allv):
        return float("nan")
    if INF in allv:
        return INF
    if -INF in allv:
        return -INF
    f = block_size(model, ta)
    d = c
    for i in range(0, len(a), f):
        x, y = list(a[i:i + f]), list(b[i:i + f])
        x += [0.0] * (f - len(x))
        y += [0.0] * (f - len(y))
        if model == "cdna1":
            d = cdna1_block(x, y, d)
        elif model == "cdna2":
            d = cdna2_block(ta, tb, x, y, d)
        else:
            d = cdna3_block(ta, tb, x, y, d)
        if d != d:
            return d
    return d


# ---- vectors ---------------------------------------------------------------------------------------------------------------
GROUPS = [   # model, A format, B format, K
    ("cdna1", "h", "h", 16), ("cdna1", "b", "b", 8),
    ("cdna2", "h", "h", 16), ("cdna2", "b", "b", 16),
    ("cdna3", "h", "h", 16), ("cdna3", "b", "b", 16), ("cdna3", "F", "F", 8),
    ("cdna3", "e", "e", 32), ("cdna3", "g", "g", 32), ("cdna3", "e", "g", 32),
]
FIELDS = {"h": (5, 10), "b": (8, 7), "F": (8, 10), "e": (4, 3), "g": (5, 2)}


def bias_of(fmt):
    eb = FIELDS[fmt][0]
    return (1 << (eb - 1)) - 1 + (1 if fmt in "eg" else 0)


def rand_code(r, fmt, cls):
    """A random code of the format; the class says where its exponent lies."""
    eb, mb = FIELDS[fmt]
    top = (1 << eb) - 1
    if fmt in "hbF":
        top -= 1   # no infinity or NaN code here: the special class puts those in
    bias = bias_of(fmt)
    if cls == 2:
        ex = 0 if r.random() < 0.6 else r.randint(0, 2)   # subnormal and tiny inputs
    elif cls == 1:
        ex = bias + r.randint(-1, 1)
    elif cls == 7:
        ex = r.randint(max(0, top - 4), top)
    elif cls == 10:   # tiny products: binary32 subnormal results for the wide-exponent formats
        ex = bias - 62 + r.randint(-6, 6) if eb == 8 else bias + r.randint(-3, 3)
    elif cls == 11:
        ex = r.randint(0, top)
    else:
        ex = bias + r.randint(-min(8, bias), min(8, top - bias))
    ex = max(0, min(top, ex))
    bits = (r.getrandbits(1) << (eb + mb)) | (ex << mb) | r.getrandbits(mb)
    if fmt in "eg" and bits == 0x80:
        bits = 0
    return bits


def special_code(r, fmt):
    eb, mb = FIELDS[fmt]
    code = (((1 << eb) - 1) << mb) | (r.choice([0, 0, 1]) if mb else 0)   # an infinity, or a NaN
    return (r.getrandbits(1) << (eb + mb)) | code


def f32_with_exp(r, ex):
    """A random binary32 of unbiased exponent ex (subnormal below -126), random sign."""
    v = f32_round(F(math.ldexp(1.0 + r.getrandbits(23) / 2.0 ** 23, ex)))
    return -v if r.getrandbits(1) else v


def rand_f32(r, cls, a, b):
    ps = [abs(x * y) for x, y in zip(a, b) if x * y != 0 and abs(x * y) != INF and x * y == x * y]
    ref = exp2floor(max(ps)) if ps else 0
    if cls == 3 and ps:
        ex = sum((F(x) * F(y) for x, y in zip(a, b)), F(0))
        v = f32_round(-ex) if abs(ex) < F(2) ** 127 else 1.0
        if r.random() < 0.7 and v == v and abs(v) != INF:
            v = f32val((f32bits(v) + r.choice([-1, 0, 1])) & 0xFFFFFFFF)
        return v
    if cls == 5:
        return f32val(r.getrandbits(23) | r.getrandbits(1) << 31) if r.random() < 0.6 else f32_with_exp(r, r.randint(-132, -120))
    if cls == 4:
        return f32_with_exp(r, max(-149, min(127, ref + r.randint(10, 30))))
    if cls == 6 and r.random() < 0.5:
        return 0.0 if r.random() < 0.7 else -0.0
    if cls == 9:
        return r.choice([1.0, -1.0, 2.0 ** 24, 1.5, 3.0]) * 2.0 ** r.randint(-3, 3)
    if r.random() < 0.12:
        return 0.0
    return f32_with_exp(r, max(-149, min(120, ref + r.randint(-26, 4))))


def make(r, model, ta, tb, K, cls):
    ac, bc = [], []
    for _ in range(K):
        if cls == 6 and r.random() < 0.45:
            ac.append(0)
            bc.append(rand_code(r, tb, cls))
        elif cls == 6 and r.random() < 0.3:
            ac.append(rand_code(r, ta, cls))
            bc.append(0)
        else:
            ac.append(rand_code(r, ta, cls))
            bc.append(rand_code(r, tb, cls))
    a = [decode(ta, x) for x in ac]
    b = [decode(tb, x) for x in bc]
    c = rand_f32(r, cls, a, b)
    if cls == 8 and ta in "hbF":
        for _ in range(r.randint(1, 2)):
            i = r.randrange(K)
            (ac if r.random() < 0.5 else bc)[i] = special_code(r, ta)
        if r.random() < 0.3:
            c = r.choice([INF, -INF, float("nan")])
        a = [decode(ta, x) for x in ac]
        b = [decode(tb, x) for x in bc]
    return ac, bc, c, a, b


def gen(path, n, seed):
    r = random.Random(seed)
    with open(path, "w") as out:
        out.write("# model A-format B-format K | A codes | B codes | C bits | D bits (the answer of tools/mfma-models/model_ref.py); regenerate with\n")
        out.write("#   tools/mfma-models/model_ref.py gen <file> --n %d --seed %d\n" % (n, seed))
        for model, ta, tb, K in GROUPS:
            for i in range(n):
                ac, bc, c, a, b = make(r, model, ta, tb, K, i % 12)
                d = model_dot(model, ta, tb, a, b, c)
                dbits = 0x7FC00000 if d != d else f32bits(d)
                out.write("%s %s %s %d | %s | %s | %08x | %08x\n" % (
                    model, ta, tb, K,
                    " ".join("%0*x" % (WIDTH[ta], x) for x in ac),
                    " ".join("%0*x" % (WIDTH[tb], x) for x in bc),
                    f32bits(c) if c == c else 0x7FC00000, dbits))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen"])
    ap.add_argument("file")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    o = ap.parse_args()
    gen(o.file, o.n, o.seed)
