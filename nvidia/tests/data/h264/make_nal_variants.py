#!/usr/bin/env python3
"""Writes the streams of extra NAL units that nvcuvid_h264 plays: p_sei and b_sei (a user_data_unregistered SEI message before every
slice), p_aud (an access unit delimiter before every slice), and p_ps_mid (the sequence and picture parameter sets repeated, with the
level changed, before the ninth slice). Run from this directory; needs p_cabac.h264 and b_spatial.h264."""
import re


def nals(d):
    idx = [m.start() for m in re.finditer(b'\x00\x00\x01', d)]
    out = []
    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else len(d)
        n = d[s + 3:e]
        while n and n[-1] == 0:
            n = n[:-1]
        out.append(n)
    return out


def unescape(n):
    out = bytearray()
    z = 0
    for b in n:
        if z >= 2 and b == 3:
            z = 0
            continue
        out.append(b)
        z = z + 1 if b == 0 else 0
    return bytes(out)


def escape(r):
    out = bytearray()
    z = 0
    for b in r:
        if z >= 2 and b <= 3:
            out.append(3)
            z = 0
        out.append(b)
        z = z + 1 if b == 0 else 0
    return bytes(out)


def build(ns, before_slice, at=None):
    res = b''
    k = 0
    for n in ns:
        if n[0] & 31 in (1, 5):
            if at is None or k == at:
                for x in before_slice:
                    res += b'\x00\x00\x00\x01' + x
            k += 1
        res += b'\x00\x00\x00\x01' + n
    return res


sei = bytes([6, 5, 17]) + bytes(range(16)) + b'x' + b'\x80'
for src, dst in (('p_cabac', 'p_sei'), ('b_spatial', 'b_sei')):
    open(dst + '.h264', 'wb').write(build(nals(open(src + '.h264', 'rb').read()), [sei]))
ns = nals(open('p_cabac.h264', 'rb').read())
open('p_aud.h264', 'wb').write(build(ns, [bytes([9, 0xF0])]))
sps = [n for n in ns if n[0] & 31 == 7][0]
pps = [n for n in ns if n[0] & 31 == 8][0]
r = bytearray(unescape(sps))
r[3] += 1   # level_idc + 1
open('p_ps_mid.h264', 'wb').write(build(ns, [bytes([sps[0]]) + escape(bytes(r[1:])), pps], at=8))
