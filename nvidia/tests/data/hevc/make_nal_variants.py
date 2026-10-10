#!/usr/bin/env python3
"""Writes the HEVC fixtures that differ from a plain x265 stream only in their NAL units, with a few lines of Python from the
streams make_fixtures.sh writes: extra SEI messages, access unit delimiters and repeated parameter sets before pictures, streams
that start at a CRA picture or contain a BLA picture or an end of sequence NAL unit, and two streams joined with different
picture sizes. The card's parser must give its own callbacks for each (nvcuvid_hevc.rtx3060.txt); they are committed, not made by
the tests.
"""
import re
import sys


def nals(data):
    """[(type, bytes with the start code)] of an Annex B stream (the start code is 00 00 00 01 for every unit)."""
    starts = [m.start() for m in re.finditer(b'\x00\x00\x01', data)]
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(data)
        while e > s and data[e - 1] == 0 and i + 1 < len(starts):
            e -= 1
        body = data[s + 3:e]
        out.append(((body[0] >> 1) & 63, b'\x00\x00\x00\x01' + body))
    return out


def join(units):
    return b''.join(u for _, u in units)


def sei_unit(payload_type, payload, nal_type=39):
    rb = bytes([payload_type, len(payload)]) + payload + b'\x80'
    esc = bytearray()
    zeros = 0
    for b in rb:
        if zeros >= 2 and b <= 3:
            esc.append(3)
            zeros = 0
        esc.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return (nal_type, b'\x00\x00\x00\x01' + bytes([nal_type << 1, 1]) + bytes(esc))


def read(name):
    return open(name + '.h265', 'rb').read()


def write(name, units):
    open(name + '.h265', 'wb').write(join(units))


def vcl(t):
    return t < 32


def first_slice(u):
    return (u[6] & 0x80) != 0   # first_slice_segment_in_pic_flag: first bit after the 4-byte start code and the 2-byte header


def insert_before_pictures(units, make, every=1):
    out = []
    n = 0
    for t, u in units:
        if vcl(t) and first_slice(u):
            if n % every == 0:
                out.extend(make(n))
            n += 1
        out.append((t, u))
    return out


# an SEI message (user data unregistered, 20 bytes) before every picture
b = nals(read('b_pyramid'))
write('b_sei', insert_before_pictures([x for x in b], lambda n: [sei_unit(5, bytes(range(16, 36)))]))
# an access unit delimiter before every picture (pic_type 2 = I, P and B slices)
write('b_aud', insert_before_pictures(b, lambda n: [(35, b'\x00\x00\x00\x01\x46\x01\x50')]))
# the parameter sets repeated before every other picture, unchanged
ps = [x for x in b if x[0] in (32, 33, 34)]
write('b_ps_mid', insert_before_pictures(b, lambda n: ps if n % 3 == 2 else []))

# a stream that starts at a CRA picture with leading pictures: the RASL pictures are not decodable and not output
g = nals(read('open_gop'))
idx = [i for i, (t, u) in enumerate(g) if t == 21]
print('CRA pictures at NAL', idx, file=sys.stderr)
start = idx[1]
head = [x for x in g if x[0] in (32, 33, 34)]
write('cra_first', head + g[start:])
# a CRA picture turned into a BLA_W_LP picture (type 16): the same, mid-stream
bl = []
for i, (t, u) in enumerate(g):
    if i == idx[1]:
        u = u[:4] + bytes([16 << 1]) + u[5:]
        t = 16
    bl.append((t, u))
write('bla', bl)
# an end of sequence NAL unit before a CRA picture: it is decoded as a random access point (no RASL output)
eo = []
for i, (t, u) in enumerate(g):
    if i == idx[1]:
        eo.append((36, b'\x00\x00\x00\x01\x48\x01'))
    eo.append((t, u))
write('eos_mid', eo)
# two sequences of different sizes joined: the second starts with its own parameter sets and an IDR picture
write('res_change', nals(read('b_flat')) + nals(read('p_low')) + nals(read('crop')))
