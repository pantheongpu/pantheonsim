#!/usr/bin/env python3
"""Writes amd/tests/data/isa_corpus_packed.txt: gfx942 instructions of the packed 16-bit float, dot product,
half conversion, half comparison, sine/cosine and packed 16-bit integer families, in every combination of the
modifiers they take (op_sel, op_sel_hi, neg_lo, neg_hi, clamp, abs, neg, omod) and of source kinds (vector
register, scalar register, inline constant), assembled by ROCm's llvm-mc. test_amd_gcn_packed_same_bits runs
each on random state and compares a digest of the result with amd/tests/data/packed_same_bits_digests.txt, which
was recorded from the build before these instructions were made faster.

  amd/tools/packed-corpus.py <llvm-mc> <out>
"""
import itertools
import random
import re
import subprocess
import sys

llvm_mc, out = sys.argv[1], sys.argv[2]
rng = random.Random(942)
lines = []

def add(text):
    lines.append(text)

vsrc = ['v1', 'v2', 'v3']
srcs3 = [['v1', 'v2', 'v3'], ['v1', 's5', 'v3'], ['s6', 'v2', 'v3'], ['v1', 'v2', 's7'], ['v1', '1.0', 'v3'], ['-2.0', 'v2', 'v3'],
         ['v1', 'v2', '0.5'], ['v4', 'v4', 'v4'], ['v1', 'v1', 'v1'], ['0', 'v2', 'v3'], ['v1', 'v2', '-4.0'], ['v1', '3', 'v3']]
srcs2 = [['v1', 'v2'], ['v1', 's5'], ['s6', 'v2'], ['v1', '1.0'], ['-2.0', 'v2'], ['v4', 'v4'], ['0.5', 'v3'], ['v1', '7']]

def mods(n, with_clamp=True, sample=None):
    """modifier strings for an instruction with n sources"""
    bits = list(itertools.product([0, 1], repeat=n))
    combos = []
    for osel in bits:
        for ohi in bits:
            for nlo in bits:
                for nhi in bits:
                    combos.append((osel, ohi, nlo, nhi))
    rng.shuffle(combos)
    if sample:
        combos = combos[:sample]
    res = []
    for osel, ohi, nlo, nhi in combos:
        parts = []
        if any(osel): parts.append('op_sel:[' + ','.join(map(str, osel)) + ']')
        if ohi != tuple([1] * n): parts.append('op_sel_hi:[' + ','.join(map(str, ohi)) + ']')   # (all ones is the default)
        if any(nlo): parts.append('neg_lo:[' + ','.join(map(str, nlo)) + ']')
        if any(nhi): parts.append('neg_hi:[' + ','.join(map(str, nhi)) + ']')
        if with_clamp and rng.random() < 0.25: parts.append('clamp')
        res.append(' '.join(parts))
    return res

# packed 16-bit float arithmetic
for op, n in (('v_pk_fma_f16', 3), ('v_pk_add_f16', 2), ('v_pk_mul_f16', 2)):
    ms = mods(n, True, 60)
    for i, m in enumerate(ms):
        srcs = (srcs3 if n == 3 else srcs2)[i % (len(srcs3) if n == 3 else len(srcs2))]
        add(f'{op} v0, {", ".join(srcs)} {m}'.strip())
    for srcs in (srcs3 if n == 3 else srcs2):
        add(f'{op} v0, {", ".join(srcs)}')
        add(f'{op} v0, {", ".join(srcs)} clamp')
# dot products
for m in mods(3, True, 120):
    srcs = rng.choice(srcs3)
    add(f'v_dot2_f32_f16 v0, {", ".join(srcs)} {m}'.strip())
for srcs in srcs3:
    add(f'v_dot2_f32_f16 v0, {", ".join(srcs)}')
    add(f'v_dot2_f32_f16 v0, {", ".join(srcs)} clamp')
for s0 in ['v1', 's5', '1.0', '0', '-2.0', '7']:
    add(f'v_dot2c_f32_f16 v0, {s0}, v2')
    add(f'v_dot2c_f32_f16_e32 v1, {s0}, v1')
# conversions
for s in ['v1', 's5', '1.0', '-2.0', '0', '0x3f800000', '0x7fc00001', '0x477ff000']:
    add(f'v_cvt_f16_f32 v0, {s}')
    add(f'v_cvt_f32_f16 v0, {s}')
for s in ['v1', 's5', '-v1', '|v1|', '-|v1|', '1.0']:
    for tail in ['', ' clamp', ' mul:2', ' mul:4', ' div:2', ' clamp mul:2']:
        add(f'v_cvt_f16_f32_e64 v0, {s}{tail}')
        add(f'v_cvt_f32_f16_e64 v0, {s}{tail}')
# comparisons of halves
for cond in ['f', 'lt', 'eq', 'le', 'gt', 'lg', 'ge', 'o', 'u', 'nge', 'nlg', 'ngt', 'nle', 'neq', 'nlt', 'tru']:
    add(f'v_cmp_{cond}_f16_e32 vcc, v1, v2')
    add(f'v_cmp_{cond}_f16_e32 vcc, 1.0, v2')
    add(f'v_cmp_{cond}_f16_e32 vcc, s5, v2')
    add(f'v_cmp_{cond}_f16_e64 s[0:1], -|v1|, v2')
    add(f'v_cmp_{cond}_f16_e64 s[2:3], |v1|, -v2')
    add(f'v_cmp_{cond}_f16_e64 s[2:3], v1, s5')
    add(f'v_cmpx_{cond}_f16_e32 vcc, v1, v2')
    add(f'v_cmpx_{cond}_f16_e64 s[4:5], -v1, |v2|')
for s in ['v1', 's5']:
    add(f'v_cmp_class_f16_e32 vcc, {s}, v2')
    add(f'v_cmp_class_f16_e64 s[0:1], -{s}, v2')
# sine and cosine
for op in ('v_sin_f32', 'v_cos_f32'):
    for s in ['v1', 's5', '1.0', '0.5', '-0.25', '0', '0x3e800000']:
        add(f'{op} v0, {s}')
    for s in ['v1', '-v1', '|v1|', '-|v1|', 's5']:
        for tail in ['', ' clamp', ' mul:2', ' mul:4', ' div:2', ' clamp mul:2']:
            add(f'{op}_e64 v0, {s}{tail}')
# packed 16-bit integers
for op, n in (('v_pk_add_u16', 2), ('v_pk_add_i16', 2), ('v_pk_sub_u16', 2), ('v_pk_sub_i16', 2), ('v_pk_mul_lo_u16', 2),
              ('v_pk_lshlrev_b16', 2), ('v_pk_lshrrev_b16', 2), ('v_pk_ashrrev_i16', 2), ('v_pk_max_i16', 2),
              ('v_pk_min_i16', 2), ('v_pk_max_u16', 2), ('v_pk_min_u16', 2), ('v_pk_mad_u16', 3), ('v_pk_mad_i16', 3)):
    for i, m in enumerate(mods(n, True, 24)):
        srcs = (srcs3 if n == 3 else srcs2)[i % (len(srcs3) if n == 3 else len(srcs2))]
        # neg_lo / neg_hi are floating-point modifiers: the integer forms take only op_sel and clamp
        m = re.sub(r'neg_(lo|hi):\[[01,]*\]\s*', '', m).strip()
        add(f'{op} v0, {", ".join(srcs)} {m}'.strip())
    for srcs in (srcs3 if n == 3 else srcs2):
        add(f'{op} v0, {", ".join(srcs)}')
        add(f'{op} v0, {", ".join(srcs)} clamp')

# assemble
seen = {}
result = []
for text in lines:
    proc = subprocess.run([llvm_mc, '--arch=amdgcn', '--mcpu=gfx942', '--show-encoding'], input=text + '\n', capture_output=True, text=True)
    m = re.search(r'encoding: \[([^\]]*)\]', proc.stdout)
    if not m:
        continue
    data = bytes(int(b, 16) for b in m.group(1).replace(' ', '').split(','))
    words = ' '.join('%08X' % int.from_bytes(data[i:i + 4], 'little') for i in range(0, len(data), 4))
    if words in seen:
        continue
    seen[words] = 1
    result.append((words, re.sub(r'\s+', ' ', text)))
# VOP3P by hand, for the modifier combinations the assembler refuses to spell (op_sel on a dot product, say): the opcode, the
# three sources and every modifier bit, as the encoding lays them out (the dword's neg_hi at 10:8, op_sel at 13:11,
# op_sel_hi[2] at 14, clamp at 15; the second dword's sources at 8:0, 17:9 and 26:18, op_sel_hi[1:0] at 28:27, neg_lo at 31:29).
SRC = {'v1': 0x101, 'v2': 0x102, 'v3': 0x103, 'v4': 0x104, 's5': 5, 's6': 6, '1.0': 0xF2, '-2.0': 0xF5, '0.5': 0xF0, '0': 0x80, '-4.0': 0xF7, '3': 0x83}
for name, opc in (('v_pk_fma_f16', 0x0E), ('v_pk_add_f16', 0x0F), ('v_pk_mul_f16', 0x10), ('v_dot2_f32_f16', 0x23)):
    for _ in range(260):
        sel = [rng.choice(list(SRC)) for _ in range(3)]
        osel = [rng.randint(0, 1) for _ in range(3)]
        ohi = [rng.randint(0, 1) for _ in range(3)]
        nlo = [rng.randint(0, 1) for _ in range(3)]
        nhi = [rng.randint(0, 1) for _ in range(3)]
        clamp = rng.random() < 0.3
        w0 = 0xD3800000 | opc << 16 | clamp << 15 | ohi[2] << 14 | (osel[0] | osel[1] << 1 | osel[2] << 2) << 11 | \
             (nhi[0] | nhi[1] << 1 | nhi[2] << 2) << 8
        w1 = SRC[sel[0]] | SRC[sel[1]] << 9 | SRC[sel[2]] << 18 | (ohi[0] | ohi[1] << 1) << 27 | (nlo[0] | nlo[1] << 1 | nlo[2] << 2) << 29
        words = '%08X %08X' % (w0, w1)
        n_src = 3 if name in ('v_pk_fma_f16', 'v_dot2_f32_f16') else 2
        text = f'{name} v0, {", ".join(sel[:n_src])} op_sel:{osel} op_sel_hi:{ohi} neg_lo:{nlo} neg_hi:{nhi}' + (' clamp' if clamp else '')
        if words not in seen:
            seen[words] = 1
            result.append((words, text))

with open(out, 'w') as f:
    f.write('# gfx942 packed 16-bit float, dot product, half conversion, comparison, sine/cosine and packed integer instructions in\n'
            '# every modifier combination: the encoding (little-endian 32-bit words) and the text. Written by amd/tools/packed-corpus.py.\n')
    for words, text in sorted(result, key=lambda r: r[1]):
        f.write(f'{words}\t{text}\n')
print(len(result), 'instructions of', len(lines), 'written to', out)
