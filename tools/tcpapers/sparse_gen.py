#!/usr/bin/env python3
"""Writes input vectors for sparse_vectors.cu: s_0..s_{K/2-1} b_0..b_{K-1} pattern c d, one line a vector, 8 hex digits a token (float32 bits; the
operands hold values of the format; d is a placeholder 00000000, the result is what a card or the simulator computes and `sparse_vectors --dump` prints).

  sparse_gen.py <mode> <count> <seed>      mode: f16k16 f16k32 bf16k16 bf16k32 tf32k8 tf32k16 f16k32h f16k16h

The same eight classes as m8n8k4_gen.py (random bit patterns; an exponent window; tiny; cancelling; one large term and small ones; subnormal; ...), the
accumulator zero, minus zero, in the window, tiny, subnormal or random. Our own generator (splitmix64).
"""
import struct, sys
M = (1 << 64) - 1
class Rng:
    def __init__(s, seed): s.x = seed & M
    def next(s):
        s.x = (s.x + 0x9E3779B97F4A7C15) & M
        z = s.x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        return z ^ (z >> 31)
    def below(s, n): return s.next() % n

def f2b(f): return struct.unpack('<I', struct.pack('<f', f))[0]
def b2f(b): return struct.unpack('<f', struct.pack('<I', b))[0]
def h2f(h): return struct.unpack('<e', struct.pack('<H', h))[0]

def value(r, fmt, cls):
    s = r.below(2)
    if fmt == 'f16':
        e = {0: r.below(31), 2: r.below(6), 5: 0, 4: 15 + r.below(10)}.get(cls, 15 - 4 + r.below(9))
        return h2f((s << 15) | (e << 10) | r.below(1024))
    mant_bits = 7 if fmt == 'bf16' else 10
    e = {0: 1 + r.below(254), 2: 1 + r.below(30), 5: 0, 4: 127 + r.below(20)}.get(cls, 127 - 4 + r.below(9))
    return b2f((s << 31) | (e << 23) | (r.below(1 << mant_bits) << (23 - mant_bits)))

def small(r, fmt):
    return value(r, fmt, 2)

def gen(mode, count, seed):
    fmt = 'f16' if mode.startswith('f16') else 'bf16' if mode.startswith('bf16') else 'tf32'
    K = int(mode.split('k')[1].rstrip('h')); S = K // 2
    r = Rng(seed)
    for i in range(count):
        cls = i % 8
        s = [value(r, fmt, cls) for _ in range(S)]
        b = [value(r, fmt, cls) for _ in range(K)]
        if cls == 4:
            for k in range(1, S): s[k] = small(r, fmt)
        pat = r.below(2 if fmt == 'tf32' else 6)
        cc = r.below(6)
        if cc == 0: c = 0.0
        elif cc == 1: c = -0.0
        elif cc == 2: c = b2f((r.below(2) << 31) | ((127 - 6 + r.below(9)) << 23) | r.below(1 << 23))
        elif cc == 3: c = b2f((r.below(2) << 31) | (r.below(30) << 23) | r.below(1 << 23))
        elif cc == 4: c = b2f((r.below(2) << 31) | r.below(1 << 23))
        else: c = b2f((r.below(2) << 31) | ((1 + r.below(200)) << 23) | r.below(1 << 23))
        if mode.endswith('h'):
            c = h2f((r.below(2) << 15) | (r.below(31) << 10) | r.below(1024)) if cc >= 2 else c
        toks = [f2b(x) for x in s] + [f2b(x) for x in b] + [pat, f2b(c), 0]
        print(" ".join("%08x" % t for t in toks))

if __name__ == '__main__':
    gen(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
