#!/usr/bin/env python3
"""Fit a tensor-core sum rule to a card's dump of wmma results.

The dump is what tools/fttn/tc_dump.cu writes on a card (nvidia/tests/e2e/simd_lane_forms.cu's tensor-core part with its inputs and
outputs written out): dump/in.{ha,hb,hc,fc,bfa,bfb}.<style>.bin and dump/out.<form>.<style>.bin, uint32 words, style random | edges.

The family of rules tried is the one measured on an RTX 3060 (include/vgpu/exec/tc_ampere.hpp) with its numbers left open:

  * K is cut into chunks of `chunk` products, in order;
  * a chunk is its exact products and the accumulator, each aligned to the largest of their frame positions `top` and cut at
    2^(top - grid) -- toward zero (`trunc`), toward minus infinity (`floor`) or not at all (`exact`);
  * a product's frame position is exp(a) + exp(b) + 1 (exponents clamped at the format's smallest normal one); the accumulator's is
    its leading bit + 1; zero terms take no part;
  * the aligned terms are added exactly and the sum is cut to 24 significant bits toward zero (`rz`) or rounded to nearest even
    (`rn`); an f16 accumulator's result is rounded to nearest even to f16.

  fit_tc_model.py <dump-dir>                     every form and style against every rule of the family: the match counts
  fit_tc_model.py <dump-dir> --rule chunk=16,grid=26,cut=trunc,final=rz [--show 5]   one rule, and its first mismatches

A rule belongs in the simulator only if it reproduces every result of every form.
"""
import itertools
import sys

import numpy as np


def load(path):
    return np.fromfile(path, dtype=np.uint32)


def f16_to_f64(h):
    return h.astype(np.uint16).view(np.float16).astype(np.float64)


def decode(vals, emin, sigbits):
    """finite doubles holding f16 / bf16 / tf32 values -> (signed integer significand, exponent of its lsb, frame exponent)"""
    m, e = np.frexp(vals)                      # vals = m * 2^e, 0.5 <= |m| < 1
    ex = np.maximum(e - 1, emin)               # exponent, clamped: a subnormal counts at emin
    lsb = ex - (sigbits - 1)
    sig = np.rint(np.ldexp(vals, -lsb)).astype(np.int64)
    return sig, lsb.astype(np.int64), ex.astype(np.int64)


def bitlen(x):
    """bit length of non-negative int64 values"""
    out = np.zeros(x.shape, dtype=np.int64)
    nz = x > 0
    out[nz] = np.floor(np.log2(x[nz].astype(np.float64))).astype(np.int64) + 1
    # log2 of a value just below a power of two can round up: correct it
    over = nz & ((np.int64(1) << np.minimum(out - 1, 62)) > x)
    out[over] -= 1
    under = nz & (out < 63) & ((np.int64(1) << np.minimum(out, 62)) <= x)
    out[under] += 1
    return out


def shift(v, s, cut):
    """v * 2^s for signed v; a right shift cuts toward zero (trunc) or toward minus infinity (floor)"""
    left = s >= 0
    sl = np.where(left, s, 0)
    sr = np.where(left, 0, np.minimum(-s, 63))
    if cut == 'floor':
        return np.where(left, v << sl, v >> sr)
    mag = np.abs(v)
    return np.sign(v) * np.where(left, mag << sl, mag >> sr)


def model(A, B, C, emin, sigbits, chunk, grid, cut, final, acc16):
    """A, B: (n, K) doubles; C: (n,) doubles; infinities and NaNs allowed. Returns (result doubles, valid mask).

    A chunk with an infinity or a NaN among its operands and the accumulator is decided as IEEE arithmetic on exact products
    decides it (a NaN operand, 0 * inf and inf - inf: a NaN; else the infinity there is), as tc_ampere.hpp's chunk_special."""
    n, K = A.shape
    finA, finB = np.isfinite(A), np.isfinite(B)
    A0, B0 = np.where(finA, A, 0.0), np.where(finB, B, 0.0)
    sa, la, ea = decode(A0, emin, sigbits)
    sb, lb, eb = decode(B0, emin, sigbits)
    psig = sa * sb
    plsb = la + lb
    ptop = np.where(psig != 0, ea + eb + 1, -10**6)
    with np.errstate(invalid='ignore', over='ignore'):
        P = A * B                                # 0 * inf is a NaN here, as wanted; a finite product is exact in double
    acc = C.copy()
    ok = np.ones(n, dtype=bool)
    for k0 in range(0, K, chunk):
        sl = slice(k0, min(K, k0 + chunk))
        special = ~np.isfinite(acc) | ~finA[:, sl].all(axis=1) | ~finB[:, sl].all(axis=1)
        nan = np.isnan(acc) | np.isnan(P[:, sl]).any(axis=1)
        pinf = (np.isposinf(acc)) | np.isposinf(P[:, sl]).any(axis=1)
        ninf = (np.isneginf(acc)) | np.isneginf(P[:, sl]).any(axis=1)
        sres = np.where(nan | (pinf & ninf), np.nan, np.where(pinf, np.inf, np.where(ninf, -np.inf, 0.0)))
        accf = np.where(np.isfinite(acc), acc, 0.0)
        am, ae = np.frexp(accf)
        asig = np.rint(np.ldexp(accf, 24 - ae)).astype(np.int64)     # 24 significant bits (a float); f16 values fit too
        alsb = (ae - 24).astype(np.int64)
        atop = np.where(asig != 0, ae.astype(np.int64), -10**6)
        top = np.maximum(ptop[:, sl].max(axis=1), atop)
        allzero = top == -10**6
        if cut == 'exact':
            g = np.minimum(np.where(psig[:, sl] != 0, plsb[:, sl], 10**6).min(axis=1), np.where(asig != 0, alsb, 10**6))
            g = np.where(allzero, 0, g)
            wide = (top - g > 60) & ~allzero & ~special
            ok &= ~wide
            g = np.where(top - g > 60, top - 60, g)
        else:
            g = np.where(allzero, 0, top - grid)
        c = 'trunc' if cut == 'exact' else cut
        S = shift(asig, alsb - g, c)
        for t in range(sl.start, sl.stop):
            S = S + shift(psig[:, t], plsb[:, t] - g, c)
        if acc16:
            v = np.ldexp(S.astype(np.float64), g)                   # |S| < 2^53 here
            with np.errstate(over='ignore'):
                fin = v.astype(np.float16).astype(np.float64)
        else:
            drop = np.maximum(bitlen(np.abs(S)) - 24, 0)
            mag = np.abs(S)
            kept = (mag >> drop) << drop
            if final == 'rn':
                rem = mag - kept
                half = np.where(drop > 0, np.int64(1) << np.maximum(drop - 1, 0), 0)
                up = (drop > 0) & ((rem > half) | ((rem == half) & (((mag >> drop) & 1) == 1)))
                kept = kept + np.where(up, np.int64(1) << drop, 0)
            v = np.sign(S) * np.ldexp(kept.astype(np.float64), g)
            with np.errstate(over='ignore'):
                fin = v.astype(np.float32).astype(np.float64)       # exact unless it overflows (an infinity) or is subnormal
        fin = np.where(allzero, 0.0, fin)
        acc = np.where(special, sres, fin)
    return acc, ok


FORMS = ('wmma_f16_f32acc', 'wmma_f16_f16acc', 'wmma_bf16_f32acc', 'wmma_tf32_f32acc')


def operands(d, form, style, t0=0, t1=None):
    """tiles t0 to t1 -> A (n, K), B (n, K), C (n,), result bits (n,), emin, sigbits, acc16, the dump's tile count"""
    def halves(name):
        return load(f'{d}/in.{name}.{style}.bin').view(np.uint16)
    out = load(f'{d}/out.{form}.{style}.bin')
    fc = load(f'{d}/in.fc.{style}.bin')
    if form == 'wmma_tf32_f32acc':
        tiles = out.size // 256
        K = 8
        f = (fc & np.uint32(0xFFFFE000)).view(np.float32).astype(np.float64)   # the tensor core reads the top 19 bits
        a = f[:tiles * 128].reshape(tiles, 16, K)
        b = f[1024:1024 + tiles * 128].reshape(tiles, 16, K)                   # [tile][column j][k]
        c = fc[2048:2048 + tiles * 256].view(np.float32).astype(np.float64).reshape(tiles, 16, 16)
        emin, sig, acc16 = -126, 11, False
        res = out.reshape(tiles, 16, 16)
    else:
        acc16 = form == 'wmma_f16_f16acc'
        tiles = (out.size * (2 if acc16 else 1)) // 256
        K = 16
        if form.startswith('wmma_bf16'):
            cvt = lambda h: (h.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            a, b = cvt(halves('bfa')), cvt(halves('bfb'))
            emin, sig = -126, 8
        else:
            a, b = f16_to_f64(halves('ha')), f16_to_f64(halves('hb'))
            emin, sig = -14, 11
        a = a[:tiles * 256].reshape(tiles, 16, K)
        b = b[:tiles * 256].reshape(tiles, 16, K)
        if acc16:
            c = f16_to_f64(halves('hc'))[:tiles * 256].reshape(tiles, 16, 16)
            res = out.view(np.uint16)[:tiles * 256].reshape(tiles, 16, 16)
        else:
            c = fc[:tiles * 256].view(np.float32).astype(np.float64).reshape(tiles, 16, 16)
            res = out.reshape(tiles, 16, 16)
    a, b, c, res = a[t0:t1], b[t0:t1], c[t0:t1], res[t0:t1]
    # every output (t, i, j): row i of A, column j of B
    A = np.repeat(a[:, :, None, :], 16, axis=2).reshape(-1, K)
    B = np.repeat(b[:, None, :, :], 16, axis=1).reshape(-1, K)
    return A, B, c.reshape(-1), res.reshape(-1), emin, sig, acc16, tiles


def compare_tiles(d, form, style, rule, t0, t1):
    with np.errstate(invalid='ignore'):
        A, B, C, res, emin, sig, acc16, tiles = operands(d, form, style, t0, t1)
    fin = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1) & np.isfinite(C)
    chunk = rule['chunk'] if form != 'wmma_tf32_f32acc' else rule.get('chunk_tf32', min(rule['chunk'], 4))
    got, ok = model(A, B, C, emin, sig, chunk, rule['grid'], rule['cut'], rule['final'], acc16)
    rb = res.astype(np.uint32)
    with np.errstate(over='ignore', invalid='ignore'):
        if acc16:
            gb = got.astype(np.float16).view(np.uint16).astype(np.uint32)
            gb = np.where(np.isnan(got), 0x7FFF, gb)                 # the cards write the canonical NaN
            rnan = ((rb & 0x7C00) == 0x7C00) & ((rb & 0x3FF) != 0)
        else:
            gb = got.astype(np.float32).view(np.uint32)
            gb = np.where(np.isnan(got), 0x7FFFFFFF, gb)
            rnan = ((rb & 0x7F800000) == 0x7F800000) & ((rb & 0x7FFFFF) != 0)
    zero = lambda x: (x & (0x7FFF if acc16 else 0x7FFFFFFF)) == 0
    same = (gb == rb) | (zero(gb) & (rb == 0))                       # a zero sum is +0 on the cards measured so far
    same |= np.isnan(got) & rnan if rule.get('nan') == 'any' else False
    return int(ok.sum()), int((same & ok).sum()), int((~fin & ok).sum()), np.nonzero(ok & ~same)[0], (A, B, C, rb, gb), tiles


def compare(d, form, style, rule, batch=256):
    """the whole dump, `batch` tiles at a time (a million results a form would otherwise take gigabytes)"""
    n = ok = special = 0
    t0, tiles, first = 0, None, None
    while tiles is None or t0 < tiles:
        a, b, c, bad, data, tiles = compare_tiles(d, form, style, rule, t0, t0 + batch)
        n, ok, special = n + a, ok + b, special + c
        if first is None or (len(first[0]) == 0 and len(bad)):
            first = (bad, data)
        t0 += batch
    return n, ok, special, first[0], first[1]


def parse_rule(text):
    r = {'chunk': 8, 'grid': 25, 'cut': 'trunc', 'final': 'rz'}
    for kv in text.split(','):
        k, v = kv.split('=')
        r[k] = int(v) if v.lstrip('-').isdigit() else v
    return r


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    styles = ('random', 'edges')
    if '--rule' in sys.argv:
        rule = parse_rule(sys.argv[sys.argv.index('--rule') + 1])
        show = int(sys.argv[sys.argv.index('--show') + 1]) if '--show' in sys.argv else 0
        total = good = 0
        for form in FORMS:
            for style in styles:
                n, ok, special, bad, (A, B, C, rb, gb) = compare(d, form, style, rule)
                total += n
                good += ok
                print(f'{form:18s} {style:6s} results {n:8d}  same {ok:8d}  differ {n - ok:7d}  (with an infinity or a NaN among the operands: {special})')
                for i in bad[:show]:
                    print('   out', i, 'card %08x rule %08x' % (rb[i], gb[i]), 'C', float(C[i]).hex())
                    print('      A', [float(x).hex() for x in A[i]])
                    print('      B', [float(x).hex() for x in B[i]])
        print(f'rule {rule}: {good} of {total} results reproduced ({100.0 * good / max(total, 1):.4f} %)')
        return 0 if good == total else 1
    rows = []
    for chunk, grid, cut, final in itertools.product((1, 2, 4, 8, 16), (24, 25, 26, 27, 28, 0), ('trunc', 'floor'), ('rz', 'rn')):
        if grid == 0:
            if cut == 'floor':
                continue
            rule = {'chunk': chunk, 'grid': 0, 'cut': 'exact', 'final': final}
        else:
            rule = {'chunk': chunk, 'grid': grid, 'cut': cut, 'final': final}
        counts = []
        for form in FORMS:
            n = ok = 0
            for style in styles:
                a, b, _, _, _ = compare(d, form, style, rule)
                n += a
                ok += b
            counts.append((ok, n))
        rows.append((sum(c[0] for c in counts), rule, counts))
    rows.sort(key=lambda r: -r[0])
    print('best rules (same / results per form: ' + ', '.join(FORMS) + ')')
    for score, rule, counts in rows[:12]:
        print(f"chunk={rule['chunk']:2d} (tf32 {min(rule['chunk'], 4)}) grid={rule['grid'] or 'exact':>5} cut={rule['cut']:5s} final={rule['final']}: " +
              '  '.join(f'{ok}/{n}' for ok, n in counts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
