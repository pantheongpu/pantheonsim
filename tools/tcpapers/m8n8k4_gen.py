#!/usr/bin/env python3
"""Writes input vectors for m8n8k4_vectors.cu: a_0..a_3 b_0..b_3 c d, one line a vector, 8 hex digits a token (float32 bits; a and b hold binary16
values; d is a placeholder 00000000, the result is what a card or the simulator computes and `m8n8k4_vectors --dump` prints).

  m8n8k4_gen.py <count> <seed> [fp32|fp16]

Eight classes in equal parts: random finite binary16 bit patterns; exponents in a window of 9; tiny operands (products around the smallest f32 and
binary16 subnormals); the second half of the products cancelling the first; one large term and small ones (what an alignment cut shows); all operands
subnormal; products that sum to a tie; mixed. The accumulator is zero, minus zero, in the window, tiny, subnormal or random. Our own generator
(splitmix64); no input of the paper's is used.
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

def h2f(h):   # binary16 bits -> float
    return struct.unpack('<e', struct.pack('<H', h))[0]
def f2b(f): return struct.unpack('<I', struct.pack('<f', f))[0]
def half(sign, e, m): return (sign << 15) | (e << 10) | m      # biased exponent e (0 subnormal)

def rand_half(r, cls):
    s = r.below(2); m = r.below(1024)
    if cls == 0:                      # any finite
        e = r.below(31)
    elif cls in (1, 3, 6):            # window around 1
        e = 15 - 4 + r.below(9)
    elif cls == 2:                    # tiny
        e = r.below(6)
    elif cls == 4:                    # one large, the rest small is decided by the caller; here the large one
        e = 15 + r.below(10)
    elif cls == 5:                    # subnormal
        e = 0
    else:
        e = r.below(31)
    return half(s, e, m)

def gen(count, seed, out):
    r = Rng(seed)
    for i in range(count):
        cls = i % 8
        a = [rand_half(r, cls) for _ in range(4)]
        b = [rand_half(r, cls) for _ in range(4)]
        if cls == 3:                   # the second half cancels the first
            a[2], b[2] = a[0] ^ 0x8000, b[0]
            a[3], b[3] = a[1] ^ 0x8000, b[1]
        if cls == 4:                   # small terms after a large one
            for k in range(1, 4):
                a[k] = half(r.below(2), r.below(8), r.below(1024))
        cc = r.below(6)
        if cc == 0: c = 0.0
        elif cc == 1: c = -0.0
        elif cc == 2: c = struct.unpack('<f', struct.pack('<I', (r.below(2) << 31) | ((127 - 6 + r.below(9)) << 23) | r.below(1 << 23)))[0]
        elif cc == 3: c = struct.unpack('<f', struct.pack('<I', (r.below(2) << 31) | ((r.below(30)) << 23) | r.below(1 << 23)))[0]
        elif cc == 4: c = struct.unpack('<f', struct.pack('<I', (r.below(2) << 31) | r.below(1 << 23)))[0]   # subnormal f32
        else: c = struct.unpack('<f', struct.pack('<I', (r.below(2) << 31) | ((1 + r.below(200)) << 23) | r.below(1 << 23)))[0]
        if out == 'fp16':              # an f16 accumulator
            c = h2f(half(r.below(2), r.below(31) if cc >= 3 else (0 if cc == 4 else 15 - 4 + r.below(9)), r.below(1024))) if cc >= 2 else c
        toks = [f2b(h2f(x)) for x in a] + [f2b(h2f(x)) for x in b] + [f2b(c), 0]
        print(" ".join("%08x" % t for t in toks))

if __name__ == '__main__':
    gen(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'fp32')
