#!/usr/bin/env python3
"""Writes the fuzz targets' seed corpora from inputs the repository already has.

usage: make_seeds.py ROOT OUTDIR      (OUTDIR/ptx, OUTDIR/sass, OUTDIR/driver)

  ptx     PTX sources: tests/kernels/*.ptx, nvidia/tests/data/**/*.ptx and the PTX raw strings of the unit tests
  sass    instruction words sampled from nvidia/tests/data/sass/sm_*.txt (the nvdisasm corpus), and the cubins there
  driver  the PTX sources and cubins again, as module images, and a few memory-call streams
"""
import os
import pathlib
import re
import struct
import sys

ARCHS = [75, 80, 86, 87, 89, 90, 100, 103, 120, 121]
MAX_PTX = 80          # per corpus: small on purpose, the fuzzer grows it
MAX_PTX_BYTES = 6000


def ptx_sources(root):
    found = []
    for pat in ("tests/kernels/**/*.ptx", "nvidia/tests/data/**/*.ptx"):
        for p in sorted(root.glob(pat)):
            found.append(p.read_bytes())
    raw = re.compile(r'R"([A-Za-z_]*)\((.*?)\)\1"', re.S)
    cand = []
    for d in ("tests/unit", "nvidia/tests/unit"):
        for p in sorted((root / d).glob("*.cpp")):
            for m in raw.finditer(p.read_text(errors="replace")):
                body = m.group(2)
                if ".entry" in body and ".version" in body and len(body) <= MAX_PTX_BYTES:
                    cand.append(body.encode())
    seen, uniq = set(), []
    for b in found + cand:
        if b not in seen and len(b) <= MAX_PTX_BYTES:
            seen.add(b)
            uniq.append(b)
    step = max(1, len(uniq) // MAX_PTX)
    return uniq[::step][:MAX_PTX]


def sass_words(root):
    out = {}
    for p in sorted((root / "nvidia/tests/data/sass").glob("sm_*.txt")):
        sm = int(re.match(r"sm_(\d+)", p.stem).group(1))
        lines = [l for l in p.read_text().splitlines() if l and l[0] != "#"]
        step = max(1, len(lines) // 96)
        words = b""
        for l in lines[::step][:96]:
            h = l.split("\t")[0]
            hi, lo = int(h[:16], 16), int(h[16:32], 16)
            words += struct.pack("<QQ", lo, hi)
        out.setdefault(sm, b"")
        out[sm] += words
    return out


def main():
    root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for d in ("ptx", "sass", "driver"):
        (out / d).mkdir(parents=True, exist_ok=True)
    srcs = ptx_sources(root)
    for i, b in enumerate(srcs):
        (out / "ptx" / f"ptx{i:03}").write_bytes(b)
        (out / "driver" / f"mod{i:03}").write_bytes(b"\x00" + b)
    for sm, words in sass_words(root).items():
        if sm in ARCHS:
            (out / "sass" / f"words_sm{sm}").write_bytes(bytes([ARCHS.index(sm) << 1]) + words)
    for i, p in enumerate(sorted(root.glob("nvidia/tests/data/**/*.cubin"))):
        img = p.read_bytes()
        (out / "sass" / f"cubin{i}").write_bytes(bytes([(ARCHS.index(86) << 1) | 1]) + img)
        (out / "driver" / f"cubin{i}").write_bytes(b"\x00" + img)
    # Memory-call streams: allocate, copy in, launch, copy out, free.
    (out / "driver" / "mem0").write_bytes(
        b"\x01" + bytes(range(64)) + bytes([0, 0, 1, 0, 0, 0, 9, 1, 2, 3, 4, 5, 6, 7, 8, 3, 0, 16, 0, 0, 4, 0, 16, 0, 0, 2, 0]))
    (out / "driver" / "mem1").write_bytes(b"\x01" + os.urandom(0) + bytes((i * 37 + 11) & 255 for i in range(200)))
    print(f"seeds: {len(srcs)} ptx, {len(list((out / 'sass').iterdir()))} sass, {len(list((out / 'driver').iterdir()))} driver")


main()
