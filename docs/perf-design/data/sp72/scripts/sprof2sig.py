#!/usr/bin/env python3
"""Samples of the round sp69 sampler (sprof.c: <prefix>.<pid>.samples and .maps) as the lines classify.py reads ("<count> <object>+<offset>").

  python3 -I sprof2sig.py <prefix>.<pid> > samples.txt

Only the sampled instruction of each sample is kept (the call chain is not used). An object's offset is the address less the
start of the object's first mapping, which is the address objdump and nm print for a shared object.
"""
import collections
import struct
import sys

prefix = sys.argv[1]
SLOT = 26
maps, base = [], {}
for line in open(prefix + ".maps"):
    p = line.split()
    if len(p) < 6:
        continue
    lo, hi = (int(x, 16) for x in p[0].split("-"))
    base.setdefault(p[5], lo)
    if "x" in p[1]:
        maps.append((lo, hi, p[5]))
data = open(prefix + ".samples", "rb").read()
out = collections.Counter()
for i in range(len(data) // (SLOT * 8)):
    d, _tid, pc = struct.unpack_from("<3Q", data, i * SLOT * 8)
    if d == 0:
        continue
    for lo, hi, path in maps:
        if lo <= pc < hi:
            out[(path, pc - base[path])] += 1
            break
    else:
        out[("[unknown]", pc)] += 1
for (path, off), n in out.items():
    print("%d %s+%#x" % (n, path, off))
