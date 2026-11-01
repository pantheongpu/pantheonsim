#!/usr/bin/env python3
"""The simulator's hot translation units are plain machine code on 64-byte function boundaries.

The SASS engine (src/sass/exec.cpp), the PTX interpreter (src/exec/interpreter.cpp), the memory manager (src/core/memory.cpp), the SIMD
kernels (src/simd/simd.cpp, lanes.cpp) and the AMD pre-decoded tier (amd/src/gcn_tier.cpp) are built outside link-time optimization with -falign-functions=64 -falign-loops=32 (CMakeLists.txt,
VGPU_HOT_UNIT_FLAGS). Inside the link-time-optimized library every unrelated edit re-ordered the code and re-rolled the inlining of the engines'
hot paths, and with it the speed of rows that run unchanged code at equal instruction counts, by up to 14 % on a Zen 5 host (the tensor kernels)
and 3 to 5 % on the memory, launch and atomics rows (docs/performance.md, Round sp14). Outside LTO a unit's code depends on its own source only,
and on a fixed alignment its layout does not move when something linked ahead of it does.

This reads the unit's object file (ELF section headers; the standard library only) and fails when
  * it carries LTO bytecode (.gnu.lto_*): the unit went through the link-time optimizer after all, or
  * an executable section of it other than the cold/startup ones is aligned to less than 64 bytes: the flag did not reach it, or
  * its .text is aligned to less than 4096 bytes: the first line of the source (an `asm` that raises the section's alignment to a page, so that
    where the unit starts modulo a page does not depend on the units linked ahead of it) is gone.
usage: check_hot_unit_layout.py OBJECT_DIR   (CMakeFiles/vgpu.dir of the build: the units' objects lie under it at their source paths)
"""
import os
import struct
import sys

UNITS = ("src/sass/exec.cpp.o", "src/exec/interpreter.cpp.o", "src/core/memory.cpp.o", "src/simd/simd.cpp.o", "src/simd/lanes.cpp.o",
         "amd/src/gcn_tier.cpp.o")
SHF_EXECINSTR = 0x4
SKIP = (".text.unlikely", ".text.startup", ".text.exit")   # GCC aligns the cold halves of functions to 1 and the static initializers to 16
ALIGN = 64
PAGE = 4096


def sections(path):
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"\x7fELF" or d[4] != 2 or d[5] != 1:
        raise SystemExit("%s: not a 64-bit little-endian ELF object" % path)
    shoff, = struct.unpack_from("<Q", d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    raw = []
    for i in range(shnum):
        name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)
        raw.append((name, flags, off, size, align))
    stroff = raw[shstrndx][2]
    out = []
    for name, flags, off, size, align in raw:
        end = d.index(b"\0", stroff + name)
        out.append((d[stroff + name:end].decode(), flags, size, align))
    return out


def check(path):
    problems, text = [], 0
    for name, flags, size, align in sections(path):
        if name.startswith(".gnu.lto_"):
            problems.append("%s: has LTO bytecode (%s): the unit must be built with -fno-lto" % (path, name))
            break
    for name, flags, size, align in sections(path):
        if not flags & SHF_EXECINSTR or size == 0 or name.startswith(SKIP):
            continue
        text += size
        if name == ".text" and align < PAGE:
            problems.append("%s: section .text is aligned to %d bytes, not %d: the page-alignment line at the top of the source is missing" % (path, align, PAGE))
        if align < ALIGN:
            problems.append("%s: section %s is aligned to %d bytes, not %d: -falign-functions=%d did not reach it" % (path, name, align, ALIGN, ALIGN))
    if text < 256:
        problems.append("%s: only %d bytes of code: this is not the unit's machine code" % (path, text))
    return problems


def main(argv):
    if len(argv) != 1:
        print(__doc__)
        return 2
    objs = [os.path.join(argv[0], u) for u in UNITS]
    missing = [o for o in objs if not os.path.isfile(o)]
    if missing:
        print("missing objects: " + ", ".join(missing))
        return 1
    bad = []
    for o in objs:
        bad += check(o)
    for b in bad[:20]:
        print(b)
    if len(bad) > 20:
        print("... and %d more" % (len(bad) - 20))
    if not bad:
        print("hot units: %d objects, plain machine code on %d-byte function boundaries" % (len(objs), ALIGN))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
