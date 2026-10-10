#!/usr/bin/env python3
"""Measures the columns in which NPP's 4-way watershed leaves a pixel that flows east unwritten (the table
kUnwritten4 in nvidia/src/npp_core.hpp), on a card.

usage: npp_watershed_bands.py <npp_wsb probe binary> [max width]

The probe is tools/probes/npp_watershed_probe.cu built with -lnppif -lnppisu -lnppc -lcudart (run it with the
CUDA 13 libraries). An image whose rows are east-flowing ramps (255, 254, ...) has no pixel flowing any other way,
so a column x whose pixels come back unchanged is one NPP left unwritten. Ramps are 250 pixels long at most (8-bit
values), so wider images use a sawtooth of period 250 and a second one of period 241 for the two columns each
leaves out; the result is the set of t = width - 1 - x for each width, which repeats every 112 widths from a width
of 12 on (and below that is the same set cut at t <= width - 1).
"""
import struct, subprocess, sys, os, tempfile
import numpy as np


def run(probe, images):
    d = tempfile.mkdtemp()
    fi, fo = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
    with open(fi, "wb") as f:
        f.write(struct.pack("<I", len(images)))
        for im in images:
            h, w = im.shape
            f.write(struct.pack("<II", w, h))
            f.write(np.ascontiguousarray(im, dtype=np.uint8).tobytes())
    subprocess.check_call([probe, "1", fi, fo])  # 1: 4-way (nppiNormL1)
    data = open(fo, "rb").read()
    pos, out = 0, []
    for im in images:
        n = im.size
        out.append(np.frombuffer(data, np.uint8, n, pos).reshape(im.shape))
        pos += 5 * n  # the segmented image, then 4 bytes a label
    return out


def measure(probe, widths, period):
    ims = [np.tile(np.array([255 - (x % period) for x in range(w)], np.uint8), (5, 1)) for w in widths]
    flags = {}
    for w, im, seg in zip(widths, ims, run(probe, ims)):
        flags[w] = {x: bool((seg[:, x] == im[:, x]).all()) for x in range(w - 1) if x % period not in (0, period - 1)}
    return flags


def main():
    probe = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 1099
    widths = list(range(2, top + 1))
    a, b = measure(probe, widths, 250), measure(probe, widths, 241)
    masks = [0] * 112
    for w in widths:
        t_set = set()
        for x in range(w - 1):
            f = a[w].get(x, b[w].get(x))
            if f:
                t_set.add(w - 1 - x)
        if w >= 300:
            masks[w % 112] = sum(1 << t for t in t_set)
    print("kUnwritten4[112] = {" + ", ".join(map(str, masks)) + "};")


if __name__ == "__main__":
    main()
