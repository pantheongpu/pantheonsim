#!/usr/bin/env python3
"""Round 6: prints what NPP's watershed labels do on every 2-valued image of a small size, for equal neighbours.

usage: npp_watershed_ties.py <npp_wsb probe binary> <norm 0|1> <width> <height>

The probe is tools/probes/npp_watershed_probe.cu (see npp_watershed_bands.py). Run it with the CUDA 13 libraries
on the card. Each line is the image (row by row), then the label NPP writes for every pixel, then the segmented
image. 2^(w*h) images: keep w*h <= 14.
"""
import os, struct, subprocess, sys, tempfile
import numpy as np

def run(probe, images, four):
    d = tempfile.mkdtemp()
    fi, fo = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
    with open(fi, "wb") as f:
        f.write(struct.pack("<I", len(images)))
        for im in images:
            h, w = im.shape
            f.write(struct.pack("<II", w, h))
            f.write(np.ascontiguousarray(im, dtype=np.uint8).tobytes())
    subprocess.check_call([probe, "1" if four else "0", fi, fo])
    data = open(fo, "rb").read()
    pos, out = 0, []
    for im in images:
        n = im.size
        seg = np.frombuffer(data, np.uint8, n, pos).reshape(im.shape); pos += n
        lab = np.frombuffer(data, np.uint32, n, pos).reshape(im.shape); pos += 4 * n
        out.append((seg, lab))
    return out

def main():
    probe, four, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    n = w * h
    images = [np.array([(m >> i) & 1 for i in range(n)], np.uint8).reshape(h, w) for m in range(1 << n)]
    for im, (seg, lab) in zip(images, run(probe, images, bool(four))):
        print("".join(map(str, im.flat)), "->", " ".join(map(str, lab.flat.tolist())), "| seg", "".join(map(str, seg.flat.tolist())))

if __name__ == "__main__":
    main()
