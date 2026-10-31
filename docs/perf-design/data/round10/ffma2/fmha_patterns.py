#!/usr/bin/env python3
"""The FFMA2/FMUL2/FADD2 instructions of a cubin, by shape.

  fmha_patterns.py <cubin> <out.txt>

Disassembles the cubin with nvdisasm (through nvidia/tools/sass-corpus.py's reader), keeps the three packed fp32
instructions, and writes one line per shape (the text with register numbers and immediates abstracted) in the corpus
format `<128-bit hex>\t<address>\t<nvdisasm text>`, the first instruction of that shape in the cubin, to be fed to
sass-corpus.py as an input. The count of every shape goes to stdout. Used on 77_blackwell_fmha's sm_100a cubin.
"""
import importlib.util
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
spec = importlib.util.spec_from_file_location('sass_corpus', os.path.join(root, 'nvidia/tools/sass-corpus.py'))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

WANT = {'FFMA2', 'FMUL2', 'FADD2'}
counts, first = {}, {}
for enc, addr, text in sc.disassemble(sys.argv[1]):
    if sc.opcode(text).split('.')[0] not in WANT:
        continue
    s = sc.shape(text)
    counts[s] = counts.get(s, 0) + 1
    first.setdefault(s, (enc, addr, text))
with open(sys.argv[2], 'w') as f:
    for s, (enc, addr, text) in sorted(first.items()):
        f.write(f'{enc}\t{addr:x}\t{text}\n')
for s, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f'{n:6d}  {s}')
print(f'{sum(counts.values())} instructions, {len(counts)} shapes')
