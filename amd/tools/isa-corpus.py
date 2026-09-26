#!/usr/bin/env python3
"""Writes amd/tests/data/isa_corpus.txt: instructions from real gfx942 code,
each with the encoding and the text llvm-objdump gives it, one of every
distinct shape -- mnemonic, operand kinds, modifiers -- up to a few of each
mnemonic. test_amd_gcn decodes every one and compares, so the decoder is
checked against what ROCm's libraries actually contain, in CI, without ROCm.

  amd/tools/isa-corpus.py <llvm-objdump> <out> <code object>...

Only encodings and their disassembly are kept, never whole kernels.
"""
import re
import subprocess
import sys

objdump, out, objects = sys.argv[1], sys.argv[2], sys.argv[3:]
PER_MNEMONIC = 6
line_re = re.compile(r'^\t(\S+)(.*?)\s*// [0-9A-F]+: ((?:[0-9A-F]{8} ?)+)$')
seen, per = {}, {}


def shape(mnem, ops):
    # Registers, numbers and labels made anonymous: what is left is the shape.
    s = re.sub(r'\b[vsa]\[\d+:\d+\]', 'R', ops)
    s = re.sub(r'\b[vsa]\d+\b', 'r', s)
    s = re.sub(r'0x[0-9a-fA-F]+|-?\b\d+(\.\d+)?\b', 'n', s)
    s = re.sub(r'label_\w+|\w+_\w*\d\w*', 'L', s)
    return mnem + s


for obj in objects:
    proc = subprocess.Popen([objdump, '-d', '--mcpu=gfx942', obj], stdout=subprocess.PIPE, text=True,
                            errors='replace')
    for line in proc.stdout:
        m = line_re.match(line.rstrip('\n'))
        if not m:
            continue
        mnem, ops, words = m.group(1), m.group(2), m.group(3).split()
        # A branch prints its target as a label, which only the whole object has.
        if 'branch' in mnem or mnem == 's_call_b64' or 'label_' in ops:
            continue
        key = shape(mnem, ops)
        if key in seen or per.get(mnem, 0) >= PER_MNEMONIC:
            continue
        text = re.sub(r'\s+', ' ', (mnem + ops).strip())
        seen[key] = (' '.join(words), text)
        per[mnem] = per.get(mnem, 0) + 1
    proc.wait()

with open(out, 'w') as f:
    f.write('# gfx942 instructions from ROCm\'s libraries: the encoding (little-endian 32-bit words)\n'
            '# and llvm-objdump\'s text for it. Written by amd/tools/isa-corpus.py.\n')
    for key in sorted(seen, key=lambda k: seen[k][1]):
        words, text = seen[key]
        f.write(f'{words}\t{text}\n')
print(f'{len(seen)} instructions, {len(per)} mnemonics')
