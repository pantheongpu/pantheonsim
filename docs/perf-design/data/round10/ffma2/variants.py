#!/usr/bin/env python3
"""Variants of the packed fp32 instructions in a corpus, for nvdisasm to name.

  variants.py <arch> <corpus.txt> <out.txt>

Takes a few FFMA2/FMUL2/FADD2 lines of a corpus (nvidia/tests/data/sass/sm_100a.txt): for each mnemonic one of each operand
form (registers, a uniform register, an immediate, one register for both lanes), and for each sets one or two of the bits
the field map of docs/perf-design/round10-ffma2.md says are modifiers (|.| and -, .FTZ, .FMZ, the rounding mode, the swizzles,
.NP) that ptxas has not been seen to set together; has nvdisasm disassemble them (as a raw instruction stream, like sass-corpus.py does) and writes the
encodings with nvdisasm's text in the corpus format, to be fed to sass-corpus.py as an input, which keeps one of each shape.
"""
import importlib.util
import os
import sys
import tempfile

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
spec = importlib.util.spec_from_file_location('sass_corpus', os.path.join(root, 'nvidia/tools/sass-corpus.py'))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

BITS = [62, 63, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]
PAIRS = [(76, 80), (78, 79), (82, 83), (85, 86), (81, 82), (87, 88), (84, 85), (72, 83), (73, 83), (62, 63), (74, 75),
         (81, 83), (84, 86), (76, 78), (80, 78)]

arch, corpus, outfile = sys.argv[1:4]
seen, encs, bases = set(), [], set()


def form(text):
    """The operand forms of a line, without registers, modifiers and rounding: what one base per kind is chosen by."""
    t = text.split(None, 1)[1] if text.startswith('@') else text
    mnemonic, _, rest = t.partition(' ')
    ops = [o.strip() for o in rest.split(',')][1:]
    kinds = []
    for o in ops:
        if 'UR' in o:
            kinds.append('UR.F32' if '.F32 ' in o + ' ' and 'F32x2' not in o else 'UR')
        elif o.startswith('RZ') or 'R' in o.replace('RZ', ''):
            kinds.append('R.F32' if '.F32 ' in o + ' ' and 'F32x2' not in o else 'R')
        else:
            kinds.append('imm')
    return mnemonic.split('.')[0], tuple(kinds)


for line in open(corpus):
    enc, addr, text = line.rstrip('\n').split('\t', 2)
    op = text.split()[1] if text.startswith('@') else text.split()[0]
    if op.split('.')[0] not in ('FFMA2', 'FMUL2', 'FADD2'):
        continue
    if form(text) in bases or '.NP' in text or 'INVALID' in text or 'FMZ' in text:
        continue
    bases.add(form(text))
    hi, lo = int(enc[:16], 16), int(enc[16:], 16)
    for flips in [(b,) for b in BITS] + PAIRS:
        h, l = hi, lo
        for b in flips:
            if b >= 64:
                h ^= 1 << (b - 64)
            else:
                l ^= 1 << b
        if (h, l) not in seen:
            seen.add((h, l))
            encs.append(f'{h:016x}{l:016x}')
texts = sc.raw_disassemble(arch, encs, tempfile.mkdtemp())
kept = {}
for enc, text in zip(encs, texts):
    if text and sc.shape(text) not in kept:   # one of each shape is all the corpus keeps
        kept[sc.shape(text)] = (enc, text)
with open(outfile, 'w') as f:
    for enc, text in kept.values():
        f.write(f'{enc}\t0\t{text}\n')
print(len(encs), 'variants,', len(kept), 'shapes')
