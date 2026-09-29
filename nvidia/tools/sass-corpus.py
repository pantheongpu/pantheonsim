#!/usr/bin/env python3
"""Gathers a SASS corpus: encodings and NVIDIA's own disassembly of them.

  sass-corpus.py [--per-shape N] <arch> <out.txt> <input>...

Each input is a cubin (ELF), anything carrying a fatbin -- an executable, a
shared library such as PyTorch's libtorch_cuda.so -- or another corpus file
(.txt), whose encodings are taken again: the way to cut a small corpus from a
large one. The ELF images for <arch>
(sm_86, sm_90a, ...) are extracted with cuobjdump, disassembled with nvdisasm,
and every instruction is written as

  <128-bit encoding, high word first, hex> TAB <address, hex> TAB <nvdisasm text>

The text is nvdisasm's for the instruction on its own: the gathered
instructions are written out as one raw instruction stream and disassembled
again with `nvdisasm --binary`, so nothing in it depends on the function it
came from (in a cubin nvdisasm names branch targets with labels local to the
function). The address is the instruction's place in that stream, because
branch targets print as absolute addresses. The decoder test reads these back
and checks that each encoding decodes and prints exactly as nvdisasm printed
it.

A real binary holds the same instruction shapes over and over, so the corpus
keeps a bounded number of encodings for each shape (the text with register
numbers and immediates abstracted), which keeps every modifier combination the
inputs contain while dropping repeats. Existing lines in <out.txt> are kept,
so runs over different inputs accumulate.

Needs cuobjdump and nvdisasm from a CUDA toolkit on PATH (or CUDA_HOME).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

PER_SHAPE = 4

INSN = re.compile(r'^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\*\s*0x([0-9a-f]{16})\s*\*/\s*$')
HIGH = re.compile(r'^\s*/\*\s*0x([0-9a-f]{16})\s*\*/\s*$')


def tool(name):
    home = os.environ.get('CUDA_HOME')
    if home and os.path.exists(os.path.join(home, 'bin', name)):
        return os.path.join(home, 'bin', name)
    found = shutil.which(name)
    if not found:
        sys.exit(f'{name} not found (set CUDA_HOME)')
    return found


def shape(text):
    s = re.sub(r'\b(U?R)\d+\b', r'\1#', text)
    s = re.sub(r'\b(U?P)\d\b', r'\1#', s)
    s = re.sub(r'\bB\d+\b', 'B#', s)
    s = re.sub(r'-?0x[0-9a-f]+', '#', s)
    s = re.sub(r'(?<![\w.])-?\d+(\.\d+)?(e[+-]?\d+)?\b', '#', s)
    return re.sub(r'\s+', ' ', s).strip()


def opcode(text):
    return re.sub(r'^@!?U?P[T0-6]\s+', '', text).split()[0]


def disassemble(elf):
    """Yields (hex128, address, text) for every instruction in a cubin."""
    out = subprocess.run([tool('nvdisasm'), '-hex', '-c', elf], capture_output=True, text=True)
    if out.returncode != 0:
        print(f'warning: nvdisasm failed on {elf}: {out.stderr.strip()[:200]}', file=sys.stderr)
        return
    pending = None
    for line in out.stdout.splitlines():
        m = INSN.match(line)
        if m:
            pending = (int(m.group(1), 16), ' '.join(m.group(2).split()), m.group(3))
            continue
        h = HIGH.match(line)
        if h and pending:
            addr, text, lo = pending
            yield h.group(1) + lo, addr, text
            pending = None


def raw_disassemble(arch, encodings, tmp):
    """nvdisasm's text for each encoding, disassembled as a raw stream."""
    raw = os.path.join(tmp, 'corpus.bin')
    with open(raw, 'wb') as f:
        for enc in encodings:
            hi, lo = int(enc[:16], 16), int(enc[16:], 16)
            f.write(lo.to_bytes(8, 'little') + hi.to_bytes(8, 'little'))
    sm = 'SM' + arch[3:]
    out = subprocess.run([tool('nvdisasm'), '-b', sm, '-hex', raw], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f'nvdisasm -b {sm} failed: {out.stderr.strip()[:300]}')
    texts = {}
    for line in out.stdout.splitlines():
        m = INSN.match(line)
        if m:
            texts[int(m.group(1), 16)] = ' '.join(m.group(2).split())
    return [texts.get(i * 16) for i in range(len(encodings))]


def elfs_for(path, arch, tmp):
    """The cubins for `arch` inside `path` (itself, if it is one)."""
    with open(path, 'rb') as f:
        if f.read(4) == b'\x7fELF' and path.endswith('.cubin'):
            return [path]
    listing = subprocess.run([tool('cuobjdump'), '-lelf', path], capture_output=True, text=True).stdout
    names = [l.split(':', 1)[1].strip() for l in listing.splitlines() if l.startswith('ELF file')]
    want = [n for n in names if re.search(rf'\.{re.escape(arch)}\.cubin$', n)]
    got = []
    for n in want:
        subprocess.run([tool('cuobjdump'), '-xelf', n, path], cwd=tmp, capture_output=True)
        p = os.path.join(tmp, n)
        if os.path.exists(p):
            got.append(p)
    return got


def main():
    global PER_SHAPE
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '--per-shape':
        PER_SHAPE = int(args[1])
        args = args[2:]
    if len(args) < 3:
        sys.exit(__doc__)
    arch, out_path, inputs = args[0], args[1], args[2:]
    lines, seen, per_shape = [], set(), {}
    if os.path.exists(out_path):
        for line in open(out_path):
            enc, _, text = line.rstrip('\n').split('\t', 2)
            lines.append(line.rstrip('\n'))
            seen.add(enc)
            per_shape[shape(text)] = per_shape.get(shape(text), 0) + 1
    added = 0
    for path in inputs:
        if path.endswith('.txt'):
            for line in open(path):
                enc, _, text = line.rstrip('\n').split('\t', 2)
                k = shape(text)
                if enc in seen or per_shape.get(k, 0) >= PER_SHAPE:
                    continue
                seen.add(enc)
                per_shape[k] = per_shape.get(k, 0) + 1
                lines.append(f'{enc}\t0\t{text}')
                added += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            for elf in elfs_for(path, arch, tmp):
                for enc, addr, text in disassemble(elf):
                    if enc in seen:
                        continue
                    k = shape(text)
                    if per_shape.get(k, 0) >= PER_SHAPE:
                        continue
                    seen.add(enc)
                    per_shape[k] = per_shape.get(k, 0) + 1
                    lines.append(f'{enc}\t0\t{text}')
                    added += 1
    lines.sort(key=lambda l: (opcode(l.split('\t')[2]), l))
    # Every line's text is nvdisasm's for the instruction alone, at its place
    # in one raw stream.
    encodings = [l.split('\t')[0] for l in lines]
    with tempfile.TemporaryDirectory() as tmp:
        texts = raw_disassemble(arch, encodings, tmp)
    lines = [f'{enc}\t{i * 16:x}\t{t}' for i, (enc, t) in enumerate(zip(encodings, texts)) if t]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{out_path}: {added} added, {len(lines)} in all, {len(per_shape)} shapes')


if __name__ == '__main__':
    main()
