#!/usr/bin/env python3
"""Writes an assembly file with every matrix instruction (MFMA, SMFMAC, WMMA) a target has, in a few operand
shapes each, for amd/tools/isa-corpus.py to turn into amd/tests/data/isa_corpus_matrix_<target>.txt:

  amd/tools/matrix-asm.py gfx942 > m.s
  llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj m.s -o m.o
  amd/tools/isa-corpus.py --mcpu=gfx942 --all llvm-objdump isa_corpus_matrix_gfx942.txt m.o

Mnemonics the assembler refuses are left out (stderr says which when run with llvm-mc -show-encoding).
"""
import re
import sys

mcpu = sys.argv[1]
BITS = {'f32': 32, 'xf32': 32, 'f16': 16, 'bf16': 16, 'i8': 8, 'fp8': 8, 'bf8': 8, 'f64': 64}


def regs(n, base, kind='v'):
    return f'{kind}[{base}:{base + n - 1}]' if n > 1 else f'{kind}{base}'


def mfma_names():
    cdna3 = """f32_32x32x1_2b_f32 f32_16x16x1_4b_f32 f32_4x4x1_16b_f32 f32_32x32x2_f32 f32_16x16x4_f32
      f32_32x32x4_2b_f16 f32_16x16x4_4b_f16 f32_4x4x4_16b_f16 f32_32x32x8_f16 f32_16x16x16_f16
      i32_32x32x4_2b_i8 i32_16x16x4_4b_i8 i32_4x4x4_16b_i8 i32_32x32x16_i8 i32_16x16x32_i8
      f32_32x32x4_2b_bf16 f32_16x16x4_4b_bf16 f32_4x4x4_16b_bf16 f32_32x32x8_bf16 f32_16x16x16_bf16
      f32_16x16x8_xf32 f32_32x32x4_xf32 f64_16x16x4_f64 f64_4x4x4_4b_f64
      f32_16x16x32_bf8_bf8 f32_16x16x32_bf8_fp8 f32_16x16x32_fp8_bf8 f32_16x16x32_fp8_fp8
      f32_32x32x16_bf8_bf8 f32_32x32x16_bf8_fp8 f32_32x32x16_fp8_bf8 f32_32x32x16_fp8_fp8"""
    gfx950 = """f32_16x16x32_f16 f32_32x32x16_f16 f32_16x16x32_bf16 f32_32x32x16_bf16 i32_16x16x64_i8 i32_32x32x32_i8"""
    names = cdna3.split()
    if mcpu == 'gfx950':
        names = [n for n in names if 'xf32' not in n] + gfx950.split()
    return names


def mfma_lines():
    out = []
    for n in mfma_names():
        m = re.match(r'(f32|i32|f64)_(\d+)x(\d+)x(\d+)(?:_(\d+)b)?_(\w+)$', n)
        if not m:
            continue
        o, M, N, K, B, t = m.groups()
        M, N, K, B = int(M), int(N), int(K), int(B or 1)
        if t.count('_'):
            t = t.split('_')[0]
        bits = BITS[t.split('_')[0]]
        lanes_per_block = 64 // B
        groups = lanes_per_block // M
        per_lane = K // groups if K >= groups else 1
        areg = max(1, per_lane * bits // 32)
        outs = M * N * B // 64 * (2 if o == 'f64' else 1)
        mn = f'v_mfma_{n}'
        out.append(f'{mn} {regs(outs, 0)}, {regs(areg, 64)}, {regs(areg, 80)}, 0')
        out.append(f'{mn} {regs(outs, 32)}, {regs(areg, 64)}, {regs(areg, 80)}, {regs(outs, 32)}')
        out.append(f'{mn} {regs(outs, 0, "a")}, {regs(areg, 64)}, {regs(areg, 80)}, {regs(outs, 0, "a")}')
        for const in (['1.0', '-0.5', '1'] if o != 'i32' else ['1', '-2']):
            out.append(f'{mn} {regs(outs, 0, "a")}, {regs(areg, 64)}, {regs(areg, 80)}, {const}')
        if B > 1:
            out.append(f'{mn} {regs(outs, 0)}, {regs(areg, 64)}, {regs(areg, 80)}, {regs(outs, 0)} cbsz:1 abid:1')
    return out


def smfmac_lines():
    out = []
    names = """f32_16x16x32_f16 f32_32x32x16_f16 f32_16x16x32_bf16 f32_32x32x16_bf16 i32_16x16x64_i8 i32_32x32x32_i8
      f32_16x16x64_bf8_bf8 f32_16x16x64_bf8_fp8 f32_16x16x64_fp8_bf8 f32_16x16x64_fp8_fp8
      f32_32x32x32_bf8_bf8 f32_32x32x32_bf8_fp8 f32_32x32x32_fp8_bf8 f32_32x32x32_fp8_fp8""".split()
    if mcpu == 'gfx950':
        names = """f32_16x16x32_f16 f32_32x32x16_f16 f32_16x16x32_bf16 f32_32x32x16_bf16 i32_16x16x64_i8 i32_32x32x32_i8
          f32_16x16x64_f16 f32_32x32x32_f16 f32_16x16x64_bf16 f32_32x32x32_bf16 i32_16x16x128_i8 i32_32x32x64_i8
          f32_16x16x128_bf8_bf8 f32_16x16x128_bf8_fp8 f32_16x16x128_fp8_bf8 f32_16x16x128_fp8_fp8
          f32_32x32x64_bf8_bf8 f32_32x32x64_bf8_fp8 f32_32x32x64_fp8_bf8 f32_32x32x64_fp8_fp8""".split()
    for n in names:
        m = re.match(r'(f32|i32)_(\d+)x(\d+)x(\d+)_(\w+?)(?:_(\w+))?$', n)
        o, M, N, K, t, _ = m.groups()
        M, N, K = int(M), int(N), int(K)
        bits = BITS[t]
        per_lane = K // (64 // M)
        outs = M * N // 64
        areg = max(1, per_lane // 2 * bits // 32)
        breg = max(1, per_lane * bits // 32)
        mn = f'v_smfmac_{n}'
        out.append(f'{mn} {regs(outs, 20)}, {regs(areg, 2)}, {regs(breg, 40)}, v1')
        out.append(f'{mn} {regs(outs, 0, "a")}, {regs(areg, 8)}, {regs(breg, 56, "a")}, v60 cbsz:1 abid:1')
        out.append(f'{mn} {regs(outs, 4)}, {regs(areg, 12, "a")}, {regs(breg, 24)}, v3 abid:1')
    return out


def wmma_lines():
    out = []
    g11 = mcpu.startswith('gfx11')
    # (name, A regs, B regs, D regs)
    if g11:
        forms = [('f32_16x16x16_f16', 8, 8, 8), ('f32_16x16x16_bf16', 8, 8, 8), ('f16_16x16x16_f16', 8, 8, 8),
                 ('bf16_16x16x16_bf16', 8, 8, 8), ('i32_16x16x16_iu8', 4, 4, 8), ('i32_16x16x16_iu4', 2, 2, 8)]
    else:
        forms = [('f32_16x16x16_f16', 4, 4, 8), ('f32_16x16x16_bf16', 4, 4, 8), ('f16_16x16x16_f16', 4, 4, 4),
                 ('bf16_16x16x16_bf16', 4, 4, 4), ('i32_16x16x16_iu8', 2, 2, 8), ('i32_16x16x16_iu4', 1, 1, 8),
                 ('i32_16x16x32_iu4', 2, 2, 8), ('f32_16x16x16_fp8_fp8', 2, 2, 8), ('f32_16x16x16_fp8_bf8', 2, 2, 8),
                 ('f32_16x16x16_bf8_fp8', 2, 2, 8), ('f32_16x16x16_bf8_bf8', 2, 2, 8)]
    for n, a, b, d in forms:
        mn = f'v_wmma_{n}'
        out.append(f'{mn} {regs(d, 0)}, {regs(a, 16)}, {regs(b, 32)}, 0')
        out.append(f'{mn} {regs(d, 8)}, {regs(a, 16)}, {regs(b, 32)}, {regs(d, 8)}')
        if 'iu' in n:
            out.append(f'{mn} {regs(d, 0)}, {regs(a, 16)}, {regs(b, 32)}, 7')
            out.append(f'{mn} {regs(d, 0)}, {regs(a, 16)}, {regs(b, 32)}, {regs(d, 0)} neg_lo:[1,1,0]')
            out.append(f'{mn} {regs(d, 0)}, {regs(a, 16)}, {regs(b, 32)}, {regs(d, 0)} neg_lo:[0,1,0]')
            out.append(f'{mn} {regs(d, 0)}, {regs(a, 16)}, {regs(b, 32)}, {regs(d, 0)} neg_lo:[1,0,0] clamp')
        else:
            for const in ('1.0', '-0.5', '1'):
                out.append(f'{mn} {regs(d, 0)}, {regs(a, 16)}, {regs(b, 32)}, {const}')
            if g11 and n.startswith(('f16_', 'bf16_')):
                out.append(f'{mn} {regs(d, 0)}, {regs(a, 16)}, {regs(b, 32)}, {regs(d, 0)} op_sel:[0,0,1]')
    return out


lines = mfma_lines() + smfmac_lines() if mcpu in ('gfx942', 'gfx950') else wmma_lines()
print('.text')
for l in lines:
    print('\t' + l)
