#!/usr/bin/env python3
"""Writes the work-items' input dwords for amd_nan_card_run: the scheme of amd/tests/unit/test_gcn_nan.cpp (asm_nan.s) or test_gcn_nan2.cpp (asm_nan2.s).
usage: amd_nan_inputs.py nan|nan2 out.bin"""
import struct
import sys

if sys.argv[1] == "nan":
    H = [0x7e00, 0xfe00, 0x7d55, 0x7c01, 0xfc01, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x3c00, 0xbc00]
    F = [0x7fc00000, 0xffc00000, 0x7fd55555, 0x7f800001, 0xff800001, 0x7f800000, 0xff800000, 0, 0x80000000, 0x3f800000, 0xbf800000]
    out = b""
    for t in range(128):
        i, j, k = t % 11, (t // 11) % 11, (t * 7 + 3) % 11
        out += struct.pack("<8I", H[i], H[j], H[k], F[i], F[j], F[k], 0, 0)
else:
    N = 12
    H = [0x7e00, 0xfe00, 0x7f55, 0x7d55, 0x7c01, 0xfc01, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x3c00, 0xbc00]
    F = [0x7fc00000, 0xffc00000, 0x7fd55555, 0x7f955555, 0x7f800001, 0xff800001, 0x7f800000, 0xff800000, 0, 0x80000000, 0x3f800000, 0xbf800000]
    D = [0x7ff8000000000000, 0xfff8000000000000, 0x7ffd555555555555, 0x7ff5555555555555, 0x7ff0000000000001, 0xfff0000000000001,
         0x7ff0000000000000, 0xfff0000000000000, 0, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000]
    DOT = [0, 0x80000000, 1, 0x807fffff, 0x00010001, 0x83ff03ff, 0x3c003c00, 0xbc00bc00, 0x7bff7bff, 0x7c00fc00, 0x7e00fe00, 0x7fffffff,
           0xffffffff, 0x7f7f7f7f, 0x80808080, 0x7f7fffff]
    out = b""
    for t in range(192):
        i, j, k = t % N, (t // N) % N, (t * 7 + 3) % N
        p, q, r = t % 16, (t * 3 + t // 16) % 16, (t * 5 + 7) % 16
        out += struct.pack("<16I", H[i], H[j], H[k], F[i], F[j], F[k], D[i] & 0xffffffff, D[i] >> 32, D[j] & 0xffffffff, D[j] >> 32,
                           D[k] & 0xffffffff, D[k] >> 32, DOT[p], DOT[q], DOT[r], 0)
open(sys.argv[2], "wb").write(out)
