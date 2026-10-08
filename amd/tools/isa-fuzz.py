#!/usr/bin/env python3
"""Writes a decoder corpus for a target LLVM knows but no ROCm library has code for yet (gfx1250):
for each instruction of AMD's machine-readable ISA specification, a few encodings of it -- the
opcode in place, the other fields zero, random or the largest -- and the text llvm-mc gives them.
An encoding LLVM will not disassemble is left out, so the corpus holds what LLVM and the
specification agree on. The same format as amd/tools/isa-corpus.py's, which test_amd_gcn reads.

  amd/tools/isa-fuzz.py [--per=3] [--seed=1] <llvm-mc> <mcpu> <amdgpu_isa_*.xml> <out>

Only the plain forms of the encodings (no literal, no DPP), and no branches, whose text names a
label in llvm-objdump's output and a number in llvm-mc's.
"""
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

args = sys.argv[1:]
per, seed = 3, 1
while args and args[0].startswith('--'):
    k, v = args.pop(0)[2:].split('=')
    if k == 'per':
        per = int(v)
    elif k == 'seed':
        seed = int(v)
mc, mcpu, xml_path, out_path = args
isa = ET.parse(xml_path).getroot().find('ISA')
random.seed(seed)

encodings = {}
for e in isa.iter('Encoding'):
    name = e.findtext('EncodingName')
    bits = int(e.findtext('BitCount'))
    fields = {}
    for f in e.iter('Field'):
        ranges = [(int(r.findtext('BitOffset')), int(r.findtext('BitCount'))) for r in f.iter('Range')]
        fields[f.findtext('FieldName')] = ranges
    ids = [int(i.text, int(i.get('Radix', '2'))) for i in e.iter('EncodingIdentifier')]
    mask = int(e.findtext('EncodingIdentifierMask'), 2)
    encodings[name] = (bits, fields, ids, mask)


def field_value(word, ranges):
    v, shift = 0, 0
    for off, n in ranges:
        v |= ((word >> off) & ((1 << n) - 1)) << shift
        shift += n
    return v


def put(word, ranges, value):
    shift = 0
    for off, n in ranges:
        word &= ~(((1 << n) - 1) << off)
        word |= ((value >> shift) & ((1 << n) - 1)) << off
        shift += n
    return word


def disassemble(word, nbytes):
    data = ','.join(f'0x{(word >> (8 * i)) & 0xFF:02x}' for i in range(nbytes))
    p = subprocess.run([mc, '-disassemble', '-triple=amdgcn', f'-mcpu={mcpu}'], input=data, capture_output=True,
                       text=True)
    if p.returncode != 0 or 'warning' in p.stderr or 'error' in p.stderr or 'Invalid' in p.stdout:
        return None
    lines = [l.strip() for l in p.stdout.splitlines() if l.strip() and not l.strip().startswith('.')]
    return lines[0] if len(lines) == 1 else None


# Instructions whose text LLVM builds in ways not worth copying here, or whose operands the decoder takes in a
# form of its own: the message and dependency-counter immediates (s_sendmsg, s_wait_alu), the swizzle pattern,
# the prefetch and address-translation probes with their own operand layout, and the conversions LLVM
# prints without a suffix.
SKIP = re.compile(r'^(s_sendmsg|s_wait_alu|ds_swizzle|s_prefetch|s_buffer_prefetch|s_atc_probe|v_cvt_pk_f16_(fp8|bf8)'
                  r'|v_pipeflush|v_nop|v_add_(max|min)_|s_version|s_getreg|s_setreg|s_incperflevel|s_decperflevel|s_set_vgpr_msb|s_denorm_mode|s_barrier_wait|s_barrier_leave|s_nop|s_sleep|s_sethalt|s_setprio|s_endpgm|s_monitor_sleep|s_wait_|s_delay_alu)')
# Source codes for the special registers (src_vccz, src_execz, the flat-scratch and POPS ones) the decoder does not
# take yet; a source field that lands on one is moved to an ordinary SGPR.
SRC_FIELDS = ('SRC0', 'SRC1', 'SRC2', 'SSRC0', 'SSRC1', 'SDATA_SRC')


def tame_sources(word, fields, name):
    for fn in SRC_FIELDS:
        if fn not in fields:
            continue
        v = field_value(word, fields[fn])
        # (240 to 248 are the inline floats, which LLVM prints as a half's bits in a 16-bit integer operation.)
        if 209 <= v <= 239 or 249 <= v <= 254 or (240 <= v <= 248 and re.search(r'_[ui]16', name)):
            word = put(word, fields[fn], v & 0x3F)
    return word


rows = []
for inst in isa.find('Instructions'):
    if SKIP.match(inst.findtext('InstructionName').lower()):
        continue
    if inst.findtext('InstructionFlags/IsBranch') == 'TRUE' or inst.findtext('InstructionFlags/IsConditionalBranch') == 'TRUE':
        continue
    for ie in inst.find('InstructionEncodings'):
        name = ie.findtext('EncodingName')
        if ie.findtext('EncodingCondition') != 'default' or name not in encodings or 'LITERAL' in name or 'DPP' in name:
            continue
        bits, fields, ids, mask = encodings[name]
        if 'OP' not in fields:
            continue
        op = int(ie.find('Opcode').text, int(ie.find('Opcode').get('Radix', '10')))
        base = None
        for i in ids:
            if field_value(i, fields['OP']) == op:
                base = i & mask
                break
        if base is None:
            continue
        # Only the fields the instruction uses as operands (and its cache-policy, offset and scale bits) are
        # varied: a bit the specification gives no meaning to makes LLVM print something no program means.
        used = {o.findtext('FieldName') for o in ie.iter('Operand')}
        used |= {'TH', 'SCOPE', 'NV', 'OFFSET', 'IOFFSET', 'SCALE_OFFSET', 'SOFFSET', 'IDXEN', 'OFFEN', 'CLAMP',
                 'OMOD', 'SAVE_EXP', 'DMASK'} & set(fields)
        # A register pair, quad or wider starts at a multiple of its width in a program; LLVM rounds an odd
        # start down when it prints one, which is not a difference worth keeping in the corpus.
        align = {o.findtext('FieldName'): max(1, int(o.findtext('OperandSize') or 32) // 32) for o in ie.iter('Operand')}
        free = 0
        for fn in used:
            for off, n in fields.get(fn, []):
                free |= ((1 << n) - 1) << off
        free &= ~mask
        for fn, a in align.items():
            if a > 1 and fn in fields and fn not in ('VDATA', 'VDST', 'VSRC', 'DATA0', 'DATA1', 'DATA'):
                # SGPR fields: clear the bits below the alignment (only meaningful for the first range).
                off, n = fields[fn][0]
                free &= ~(((1 << (a.bit_length() - 1)) - 1) << off)
        for k in range(per):
            w = base
            if k == 1:
                w |= random.getrandbits(bits) & free
            elif k == 2:
                w |= random.getrandbits(bits) & free & random.getrandbits(bits) & random.getrandbits(bits)
            w = put(w, fields['OP'], op)
            w = tame_sources(w, fields, inst.findtext('InstructionName').lower())
            text = disassemble(w, bits // 8)
            if text:
                words = ' '.join(f'{(w >> (32 * i)) & 0xFFFFFFFF:08X}' for i in range(bits // 32))
                rows.append((words, text))
seen = set()
with open(out_path, 'w') as f:
    f.write(f'# {mcpu} instructions: the encoding (little-endian 32-bit words) and llvm-mc\'s text for it.\n'
            f'# Written by amd/tools/isa-fuzz.py from AMD\'s ISA specification; no hardware or library code.\n')
    for words, text in rows:
        if (words, text) in seen:
            continue
        seen.add((words, text))
        f.write(f'{words}\t{text}\n')
print(len(seen), 'lines written to', out_path)
