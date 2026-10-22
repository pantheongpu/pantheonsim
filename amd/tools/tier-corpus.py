#!/usr/bin/env python3
"""Writes amd/tests/data/isa_corpus_tier_<mcpu>.txt: hand-written instructions in the forms the AMD tier's handlers take (every operand kind,
the VOP3 modifiers, SDWA selectors, literals), each with the encoding llvm-mc gives it. Only test_amd_gcn_tier_same_bits reads them, running
each both ways (tier on, tier off) and comparing everything the instruction changed, so there are no recorded results to keep in step: the
files can be regenerated whenever a handler is added.

  amd/tools/tier-corpus.py <llvm-mc> <mcpu> <out>

A line llvm-mc refuses for the target is left out (not every instruction exists on every part).
"""
import itertools
import re
import subprocess
import sys

llvm_mc, mcpu, out = sys.argv[1], sys.argv[2], sys.argv[3]

lines = []


def add(s):
    lines.append(s)


# ---- vector, one source
F1 = ['v_rcp_f32', 'v_rcp_iflag_f32', 'v_sqrt_f32', 'v_rsq_f32', 'v_exp_f32', 'v_log_f32', 'v_floor_f32', 'v_ceil_f32', 'v_trunc_f32',
      'v_rndne_f32', 'v_fract_f32', 'v_sin_f32', 'v_cos_f32', 'v_cvt_f32_u32', 'v_cvt_f32_i32', 'v_cvt_f32_ubyte0', 'v_cvt_i32_f32',
      'v_cvt_u32_f32', 'v_cvt_f16_f32', 'v_mov_b32', 'v_not_b32', 'v_cvt_f64_f32']
H1 = ['v_cvt_f32_f16']
for op in F1 + H1:
    for ops in ['v1, v2', 'v200, v255', 'v1, s3', 'v1, vcc_lo', 'v1, 0.5', 'v1, -2.0', 'v1, 0x41f00000', 'v1, 7', 'v1, -1']:
        add('%s_e32 %s' % (op, ops))
    for ops in ['v1, v2', 'v1, -v2', 'v1, |v2|', 'v1, -|v2|', 'v1, v2 clamp', 'v1, v2 mul:2', 'v1, v2 div:2', 'v1, v2 mul:4',
                'v1, -v2 clamp mul:2', 'v1, s2', 'v1, 0.5', 'v1, 0x41f00000']:
        add('%s_e64 %s' % (op, ops))

# ---- vector, two sources
F2 = ['v_add_f32', 'v_sub_f32', 'v_subrev_f32', 'v_mul_f32', 'v_min_f32', 'v_max_f32', 'v_fmac_f32']
I2 = ['v_add_u32', 'v_sub_u32', 'v_subrev_u32', 'v_and_b32', 'v_or_b32', 'v_xor_b32', 'v_lshlrev_b32', 'v_lshrrev_b32', 'v_ashrrev_i32',
      'v_min_u32', 'v_max_u32', 'v_min_i32', 'v_max_i32', 'v_mul_u32_u24', 'v_mul_i32_i24', 'v_add_u16', 'v_sub_u16', 'v_subrev_u16',
      'v_lshlrev_b16', 'v_lshrrev_b16', 'v_ashrrev_i16', 'v_mul_lo_u16', 'v_min_u16', 'v_max_u16', 'v_min_i16', 'v_max_i16']
for op in F2 + I2:
    for ops in ['v1, v2, v3', 'v1, v2, v1', 'v1, s2, v3', 'v1, 1.0, v3', 'v1, 0x40490fdb, v3', 'v1, 5, v3', 'v1, vcc_lo, v3', 'v1, -1, v3']:
        add('%s_e32 %s' % (op, ops))
    for ops in ['v1, v2, v3', 'v1, s2, s3', 'v1, 0.5, s3', 'v1, 0x3f9e0652, v3', 'v1, 7, v3', 'v1, v2, 12']:
        add('%s_e64 %s' % (op, ops))
for op in F2:
    for ops in ['v1, -v2, v3', 'v1, v2, -v3', 'v1, |v2|, |v3|', 'v1, -|v2|, v3', 'v1, v2, v3 clamp', 'v1, v2, v3 mul:2', 'v1, v2, v3 div:2',
                'v1, v2, v3 mul:4 clamp']:
        add('%s_e64 %s' % (op, ops))
add('v_fmaak_f32 v1, v2, v3, 0x40490fdb')
add('v_fmamk_f32 v1, v2, 0x40490fdb, v3')
add('v_fmaak_f32 v1, 2.0, v3, 0x3f800000')
add('v_fmamk_f32 v1, s2, 0x3f800000, v3')

# ---- v_cndmask, ldexp, the three-source forms
for ops in ['v1, v2, v3, vcc', 'v1, s2, v3, s[4:5]', 'v1, 1.0, v3, s[6:7]', 'v1, -v2, v3, s[4:5]', 'v1, v2, |v3|, vcc', 'v1, v2, v3, exec',
            'v1, 0x3f9e0652, v3, vcc', 'v1, -|v2|, -v3, s[10:11]']:
    add('v_cndmask_b32_e64 ' + ops)
add('v_cndmask_b32_e32 v1, v2, v3, vcc')
for ops in ['v1, v2, v3', 'v1, -v2, v3', 'v1, |v2|, s3', 'v1, 2.0, 7', 'v1, v2, -3', 'v1, v2, v3 clamp', 'v1, v2, v3 mul:2']:
    add('v_ldexp_f32 ' + ops)
THREE = ['v_fma_f32', 'v_bfe_i32', 'v_perm_b32', 'v_bfe_u32', 'v_bfi_b32', 'v_alignbit_b32', 'v_add3_u32', 'v_mad_u32_u24', 'v_med3_u32']
for op in THREE:
    for ops in ['v1, v2, v3, v4', 'v1, s2, v3, v4', 'v1, 1.0, v3, v4', 'v1, v2, v3, 0x3f800000', 'v1, 7, v3, 5', 'v1, v2, s3, s4']:
        add('%s %s' % (op, ops))
for ops in ['v1, -v2, v3, v4', 'v1, v2, -v3, -v4', 'v1, |v2|, v3, v4', 'v1, v2, v3, -|v4|', 'v1, v2, v3, v4 clamp', 'v1, v2, v3, v4 mul:2',
            'v1, v2, v3, v4 div:2']:
    add('v_fma_f32 ' + ops)
for ops in ['v1, v2, v3', 'v1, 0x3c00, v3', 'v1, s2, v3']:
    add('v_pack_b32_f16 ' + ops)

# ---- SDWA
SEL = ['BYTE_0', 'BYTE_1', 'BYTE_2', 'BYTE_3', 'WORD_0', 'WORD_1', 'DWORD']
UNUSED = ['UNUSED_PAD', 'UNUSED_SEXT', 'UNUSED_PRESERVE']
SD2 = ['v_add_u32', 'v_sub_u32', 'v_subrev_u32', 'v_and_b32', 'v_or_b32', 'v_xor_b32', 'v_lshlrev_b32', 'v_lshrrev_b32', 'v_ashrrev_i32',
       'v_min_u32', 'v_max_u32', 'v_min_i32', 'v_max_i32', 'v_mul_u32_u24', 'v_mul_i32_i24']
SD1 = ['v_mov_b32', 'v_not_b32', 'v_cvt_f32_ubyte0', 'v_cvt_f32_u32', 'v_cvt_f32_i32']
combos = [(d, u, a, b) for d in SEL for u in UNUSED for a, b in [('DWORD', 'DWORD'), ('BYTE_1', 'WORD_1'), ('WORD_1', 'BYTE_3'), ('BYTE_2', 'BYTE_0')]]
combos = combos[::3] + [('BYTE_1', 'UNUSED_PRESERVE', 'BYTE_3', 'WORD_0'), ('WORD_1', 'UNUSED_SEXT', 'WORD_0', 'WORD_1')]
for op in SD2:
    for d, u, a, b in combos:
        add('%s_sdwa v1, v2, v3 dst_sel:%s dst_unused:%s src0_sel:%s src1_sel:%s' % (op, d, u, a, b))
    add('%s_sdwa v1, sext(v2), sext(v3) dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:WORD_1' % op)
    add('%s_sdwa v1, sext(v2), v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:DWORD' % op)
    add('%s_sdwa v1, s2, v3 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:DWORD src1_sel:WORD_0' % op)
    add('%s_sdwa v1, v2, v3 clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD' % op)
for op in SD1:
    for d, u, a, b in combos:
        add('%s_sdwa v1, v2 dst_sel:%s dst_unused:%s src0_sel:%s' % (op, d, u, a))
    add('%s_sdwa v1, sext(v2) dst_sel:BYTE_1 dst_unused:UNUSED_SEXT src0_sel:BYTE_1' % op)
for op in ['v_add_f32', 'v_mul_f32', 'v_cvt_f32_f16', 'v_add_u16', 'v_lshlrev_b16']:
    add('%s_sdwa v1, v2, v3 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD' % op if op != 'v_cvt_f32_f16' else
        '%s_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1' % op)

# ---- comparisons
CMPF = ['f', 'lt', 'eq', 'le', 'gt', 'lg', 'ge', 'o', 'u', 'nge', 'nlg', 'ngt', 'nle', 'neq', 'nlt', 'tru']
CMPI = ['f', 'lt', 'eq', 'le', 'gt', 'ne', 'ge', 't']
for t in ['f16', 'f32', 'f64']:
    for c in CMPF:
        if t == 'f64' and c in ('tru',):
            continue
        for pre in ['v_cmp', 'v_cmpx']:
            add('%s_%s_%s_e32 vcc, v1, v2' % (pre, c, t))
            add('%s_%s_%s_e32 vcc, 0.5, v2' % (pre, c, t))
            add('%s_%s_%s_e64 s[2:3], v1, v2' % (pre, c, t))
            add('%s_%s_%s_e64 s[2:3], -v1, |v2|' % (pre, c, t))
            add('%s_%s_%s_e64 vcc, s1, 2.0' % (pre, c, t))
for t in ['i16', 'u16', 'i32', 'u32']:
    for c in CMPI:
        for pre in ['v_cmp', 'v_cmpx']:
            add('%s_%s_%s_e32 vcc, v1, v2' % (pre, c, t))
            add('%s_%s_%s_e32 vcc, 7, v2' % (pre, c, t))
            add('%s_%s_%s_e64 s[2:3], v1, v2' % (pre, c, t))
            add('%s_%s_%s_e64 s[2:3], s1, 100' % (pre, c, t))
for t in ['f16', 'f32', 'f64']:
    for pre in ['v_cmp', 'v_cmpx']:
        add('%s_class_%s_e32 vcc, v1, v2' % (pre, t))
        add('%s_class_%s_e64 s[2:3], v1, v2' % (pre, t))
        add('%s_class_%s_e64 s[2:3], -v1, 0x3ff' % (pre, t))
        add('%s_class_%s_e64 s[2:3], |v1|, 0x60' % (pre, t))
        add('%s_class_%s_e32 vcc, 2.0, v2' % (pre, t))
for t in ['f16', 'f32']:
    for c in ['lt', 'ge', 'u', 'nge']:
        add('v_cmp_%s_%s_sdwa vcc, v1, v2 src0_sel:WORD_1 src1_sel:DWORD' % (c, t))

# ---- scalar
S1 = ['s_brev_b32', 's_brev_b64', 's_bcnt0_i32_b32', 's_bcnt0_i32_b64', 's_ff0_i32_b32', 's_ff0_i32_b64', 's_sext_i32_i8', 's_sext_i32_i16',
      's_cmov_b32', 's_cmov_b64', 's_mov_b32', 's_not_b32', 's_abs_i32', 's_ff1_i32_b32']
for op in S1:
    wide = op.endswith('64')
    d = 's[4:5]' if wide else 's4'
    for src in (['s[6:7]', '0x1234567', '-1', 'vcc', 'exec'] if wide else ['s6', '0x1234567', '-1', '63', 'vcc_lo', 'exec_lo']):
        add('%s %s, %s' % (op, d, src))
S2 = ['s_mul_hi_u32', 's_mul_hi_i32', 's_pack_ll_b32_b16', 's_pack_lh_b32_b16', 's_pack_hh_b32_b16', 's_pack_hl_b32_b16', 's_bfm_b32',
      's_add_u32', 's_addc_u32', 's_sub_u32', 's_lshl_b32', 's_min_u32', 's_max_i32']
for op in S2:
    for ops in ['s4, s6, s7', 's4, s6, 0x1234567', 's4, 0x7fffffff, s7', 's4, -1, 65', 's4, vcc_lo, s7', 's4, s4, s4']:
        add('%s %s' % (op, ops))
add('s_getpc_b64 s[4:5]')
add('s_getpc_b64 s[10:11]')
for op in ['s_addk_i32', 's_mulk_i32', 's_cmovk_i32']:
    for k in ['0', '1', '-1', '0x7fff', '-32768', '12345']:
        add('%s s4, %s' % (op, k))
for c in ['eq', 'lg', 'gt', 'ge', 'lt', 'le']:
    for t in ['i32', 'u32']:
        for k in ['0', '1', '-1', '0x7fff', '0x8000', '65535', '-32768']:
            add('s_cmpk_%s_%s s4, %s' % (c, t, k))
for op in ['s_bitcmp0_b32', 's_bitcmp1_b32']:
    for ops in ['s4, s5', 's4, 31', 's4, 33', '0x80000000, 31', 's4, 0x1f']:
        add('%s %s' % (op, ops))
for op in ['s_bitcmp0_b64', 's_bitcmp1_b64']:
    for ops in ['s[4:5], s6', 's[4:5], 63', 's[4:5], 65', 's[4:5], 0x20']:
        add('%s %s' % (op, ops))
add('s_endpgm')

# ---- assemble: drop what the target refuses
def assemble(ls):
    src = '\n'.join(ls) + '\n'
    p = subprocess.run([llvm_mc, '-triple=amdgcn', '-mcpu=' + mcpu, '-show-encoding'], input=src, capture_output=True, text=True)
    bad = set(int(m.group(1)) for m in re.finditer(r'<stdin>:(\d+):\d+: error', p.stderr))
    return p, bad


p, bad = assemble(lines)
lines = [l for i, l in enumerate(lines, 1) if i not in bad]
p, bad = assemble(lines)
if bad:
    sys.exit('still refused: %s' % [lines[i - 1] for i in sorted(bad)][:5])
rows = []
for m in re.finditer(r'^\t(.*?)\s*; encoding: \[(.*?)\]$', p.stdout, re.M):
    text, enc = m.group(1), m.group(2)
    data = [int(b, 16) for b in enc.split(',')]
    data += [0] * (-len(data) % 4)
    words = ['%08X' % (data[i] | data[i + 1] << 8 | data[i + 2] << 16 | data[i + 3] << 24) for i in range(0, len(data), 4)]
    rows.append('%s\t%s' % (' '.join(words), re.sub(r'\s+', ' ', text)))
assert len(rows) == len(lines), (len(rows), len(lines))
with open(out, 'w') as f:
    f.write('\n'.join(rows) + '\n')
print('%s: %d instructions' % (out, len(rows)))
