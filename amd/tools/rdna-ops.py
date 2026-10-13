#!/usr/bin/env python3
"""Writes amd/src/rdna_ops_<arch>.inc: every instruction of an RDNA
architecture, from AMD's machine-readable ISA specification (MIT-licensed,
https://gpuopen.com/machine-readable-isa/), as a table the RDNA decoder
reads -- its encoding, opcode and name, and its explicit operands in the
order the assembler writes them, each with the field that holds it, what
kind of operand it is, and how many bits wide.

  amd/tools/rdna-ops.py <amdgpu_isa_rdna3.xml> <out.inc>
  amd/tools/rdna-ops.py --images <amdgpu_isa_rdna3.xml> <rdna_images_rdna3.inc>

Only the encodings the decoder handles are kept (not graphics' export,
interpolation or LDS-direct ones). The image instructions (MIMG; RDNA4's
VIMAGE and VSAMPLE) go to a file of their own with --images, as names
only: their operands' widths come from each instruction's DMASK, DIM and
modifiers, which the decoder works out itself. The generated files are
checked in, so building needs no XML.
"""
import sys
import xml.etree.ElementTree as ET

images = len(sys.argv) > 1 and sys.argv[1] == '--images'
if images:
    sys.argv.pop(1)
xml_path, out_path = sys.argv[1], sys.argv[2]
isa = ET.parse(xml_path).getroot().find('ISA')
arch = isa.find('Architecture').findtext('ArchitectureName')

# The XML's encodings, by the decoder's names for them.
ENCODINGS = {
    'ENC_SOP1': 'Sop1', 'ENC_SOP2': 'Sop2', 'ENC_SOPK': 'Sopk', 'ENC_SOPC': 'Sopc', 'ENC_SOPP': 'Sopp',
    'ENC_SMEM': 'Smem', 'ENC_VOP1': 'Vop1', 'ENC_VOP2': 'Vop2', 'ENC_VOPC': 'Vopc', 'ENC_VOP3': 'Vop3',
    'VOP3_SDST_ENC': 'Vop3', 'ENC_VOP3P': 'Vop3p', 'ENC_DS': 'Ds', 'ENC_FLAT': 'Flat',
    'ENC_FLAT_GLOBAL': 'Flat', 'ENC_FLAT_SCRATCH': 'Flat', 'ENC_MUBUF': 'Mubuf', 'ENC_MTBUF': 'Mtbuf',
    'VOPDXY': 'Vopd', 'VOPDXY_X': 'Vopd', 'VOPDXY_Y': 'Vopd', 'VOPD3XY': 'Vopd', 'ENC_VOP3PX2': 'Vop3p', 'ENC_VOP3PX3': 'Vop3p',   # CDNA 5 lists the X and Y halves apart
    # RDNA4's names for the same encodings.
    'ENC_VOP3SD': 'Vop3', 'ENC_VFLAT': 'Flat', 'ENC_VGLOBAL': 'Flat', 'ENC_VSCRATCH': 'Flat',
    'ENC_VBUFFER': 'Mubuf', 'ENC_VOPD': 'Vopd', 'ENC_VDS': 'Ds', 'ENC_FLAT_GLBL': 'Flat',
}
# Operand kinds, by the XML's operand types.
KINDS = {
    'OPR_VGPR': 'Vgpr', 'OPR_SRC': 'Src', 'OPR_SRC_VGPR': 'Vgpr', 'OPR_SRC_VGPR_OR_INLINE': 'Src',
    'OPR_SSRC': 'Ssrc', 'OPR_SSRC_LANESEL': 'Ssrc', 'OPR_SREG': 'Sreg', 'OPR_SDST': 'Sdst',
    'OPR_SDST_NULL': 'Sdst', 'OPR_SREG_M0_INL': 'Ssrc', 'OPR_SMEM_OFFSET': 'Ssrc', 'OPR_SIMM16': 'Simm16',
    'OPR_SIMM32': 'Simm32', 'OPR_LABEL': 'Label', 'OPR_HWREG': 'Hwreg', 'OPR_SENDMSG': 'Sendmsg',
    'OPR_WAITCNT': 'Waitcnt', 'OPR_WAITCNT_DEPCTR': 'Depctr', 'OPR_DELAY': 'Delay', 'OPR_CLAUSE': 'Simm16',
    'OPR_VERSION': 'Simm16', 'OPR_SIMM8': 'Simm16', 'OPR_SIMM24': 'Simm16', 'OPR_VCC': 'Vcc', 'OPR_EXEC': 'Exec',
    # RDNA4's.
    'OPR_SREG_M0': 'Ssrc', 'OPR_SREG_LITERAL': 'Ssrc', 'OPR_SMEM_OFFSET_NOK': 'Ssrc', 'OPR_SIMM5': 'Simm16',
    'OPR_SENDMSG_RTN': 'Sendmsg', 'OPR_SSRC_BARRIER_ID': 'Ssrc', 'OPR_WAIT_MEM_DS': 'Waitcnt', 'OPR_SLEEP': 'Simm16',
    'OPR_WAIT_ALU': 'Depctr', 'OPR_WAIT_EVENT': 'Simm16',
    # CDNA 5's. Most are implicit operands (the memory a load reads, the PC a call writes) and never reach the
    # table; the explicit ones are an s_set_vgpr_msb immediate, a scalar register of a tensor instruction, a
    # 64-bit literal, and the plain and no-inline-constant vector sources.
    'OPR_SET_VGPR_MSB': 'Simm16', 'OPR_SGPR': 'Sreg', 'OPR_SIMM64': 'Simm64', 'OPR_SRC_NOINLINE': 'Src',
    'OPR_SRC_SIMPLE': 'Src', 'OPR_DSMEM': 'Ssrc', 'OPR_GPUMEM': 'Ssrc', 'OPR_SDST_EXEC': 'Sdst',
    'OPR_SSRC_SPECIAL_SCC': 'Ssrc', 'OPR_PC': 'Ssrc', 'OPR_SDST_M0': 'Sdst',
    # RDNA2's.
    'OPR_SREG_NONULL': 'Sreg', 'OPR_SRC_NOLDS': 'Src', 'OPR_SSRC_NOLDS': 'Ssrc', 'OPR_VGPR_OR_LDS': 'Src',
    'OPR_ATTR': 'Simm16', 'OPR_PARAM': 'Simm16',
}
# The image encodings, for --images: RDNA2 and RDNA3's MIMG, and RDNA4's
# VIMAGE and VSAMPLE, whose opcodes the segment keeps apart.
IMAGE_ENCODINGS = {'ENC_MIMG': 0, 'ENC_VIMAGE': 0, 'ENC_VSAMPLE': 1}
if images:
    ENCODINGS = {e: 'Mimg' for e in IMAGE_ENCODINGS}
elif 'CDNA 5' in arch:
    ENCODINGS['ENC_VIMAGE'] = 'Mimg'   # CDNA 5 has the tensor data mover's two, which are not image accesses
# Fields the decoder reads itself rather than as operands.
SKIP_FIELDS = {'LITERAL'}

rows = {}
aliases = {}
for inst in isa.find('Instructions'):
    name = inst.findtext('InstructionName').lower()
    alias = inst.find('AliasedInstructionNames')
    if alias is not None:
        aliases[name] = [a.text.lower() for a in alias]
    for ie in inst.find('InstructionEncodings'):
        enc_xml = ie.findtext('EncodingName')
        cond = ie.findtext('EncodingCondition')
        # The plain form: condition "default", or "no literal" spelled out.
        # An instruction that always carries a literal (v_fmaak_f32) is only
        # listed under its encoding's _INST_LITERAL form, which stands in.
        priority = 0
        if (enc_xml.endswith('_INST_LITERAL') or enc_xml.endswith(('_INST_LITERAL_X', '_INST_LITERAL_Y'))) and cond == 'default':
            enc_xml, priority = {'VOP2_INST_LITERAL': 'ENC_VOP2', 'VOPDXY_INST_LITERAL': 'VOPDXY',
                                 'VOPDXY_INST_LITERAL_X': 'VOPDXY_X', 'VOPDXY_INST_LITERAL_Y': 'VOPDXY_Y',
                                 'SOPK_INST_LITERAL': 'ENC_SOPK', 'SOP2_INST_LITERAL': 'ENC_SOP2'}.get(enc_xml, enc_xml), 1
        if enc_xml not in ENCODINGS or not (cond == 'default' or cond.startswith(('Nothas', '!has_'))):
            continue
        enc = ENCODINGS[enc_xml]
        opcode = int(ie.find('Opcode').text, int(ie.find('Opcode').get('Radix', '10')))
        segment = {'ENC_VIMAGE': 3, 'ENC_VOP3PX2': 1, 'ENC_VOP3PX3': 2, 'VOPD3XY': 1, 'ENC_FLAT_GLOBAL': 1, 'ENC_FLAT_GLBL': 1, 'ENC_VGLOBAL': 1, 'ENC_FLAT_SCRATCH': 2, 'ENC_VSCRATCH': 2}.get(enc_xml, 0)
        if images:
            segment = IMAGE_ENCODINGS[enc_xml]
        ops = []
        operands = ie.find('Operands') if not images else None
        for o in sorted(operands if operands is not None else [], key=lambda o: int(o.get('Order'))):
            if o.get('IsImplicit') == 'true':
                continue
            field, kind = o.findtext('FieldName'), o.findtext('OperandType')
            if field in SKIP_FIELDS and kind not in ('OPR_SIMM32', 'OPR_SIMM16'):
                continue
            if kind not in KINDS:
                raise SystemExit(f'{name}: operand type {kind} not handled')
            ops.append((field or '', KINDS[kind], int(o.findtext('OperandSize') or 0), o.get('Output') == 'true'))
        key = (enc, segment, opcode, enc_xml == 'VOP3_SDST_ENC' or enc_xml == 'ENC_VOP3SD')
        # An instruction may be listed twice under one encoding (aliases); the
        # first spelling is the assembler's, and a plain form beats a
        # literal-only one.
        if key not in rows or rows[key][2] > priority:
            rows[key] = (name, ops, priority)

# Instructions ROCm's LLVM and AMD's ISA document have but the XML leaves out. V_FMAMK_F64 and V_FMAAK_F64 (VOP2 35 and 36)
# are in the document's instruction list, which says they "imply the use of a 64-bit literal"; hip-tests' double-precision
# math functions use them. Each takes the 64-bit constant as K (SIMM64 here), as the 32-bit forms take a 32-bit one.
if 'CDNA 5' in arch and not images:
    f64 = [('VDST', 'Vgpr', 64, True)]
    rows.setdefault(('Vop2', 0, 35, False), ('v_fmamk_f64', f64 + [('SRC0', 'Src', 64, False), ('LITERAL', 'Simm64', 64, False), ('VSRC1', 'Vgpr', 64, False)], 0))
    rows.setdefault(('Vop2', 0, 36, False), ('v_fmaak_f64', f64 + [('SRC0', 'Src', 64, False), ('VSRC1', 'Vgpr', 64, False), ('LITERAL', 'Simm64', 64, False)], 0))

# The VOPD dot products (opcodes 12 and 13, in both the 64-bit and the 96-bit forms, as X or as Y) are in ROCm's LLVM (llvm-mc
# -mcpu=gfx1250 assembles them) but not in the XML. They are the VOP2 dot-product-accumulates, v_dot2acc_f32_f16 and
# v_dot2acc_f32_bf16, issued as half of a pair: the destination is also the addend.
if 'CDNA 5' in arch and not images:
    for seg in (0, 1):
        for opcode, name in ((12, 'v_dual_dot2acc_f32_f16'), (13, 'v_dual_dot2acc_f32_bf16')):
            rows.setdefault(('Vopd', seg, opcode, False), (name, [('VDSTX', 'Vgpr', 32, True), ('SRCX0', 'Src', 32, False), ('VSRCX1', 'Vgpr', 32, False)], 0))

with open(out_path, 'w') as f:
    what = 'every image instruction' if images else 'every instruction the RDNA decoder handles'
    f.write(f'// {arch}: {what}, from AMD\'s machine-readable\n'
            '// ISA specification (Copyright (c) Advanced Micro Devices, Inc., MIT license).\n'
            '// Written by amd/tools/rdna-ops.py; do not edit.\n'
            '//\n'
            + ('// {encoding, segment (RDNA4: 0 VIMAGE, 1 VSAMPLE), opcode, SDST form, name, {}}\n' if images else
            '// {encoding, segment (FLAT: 0 flat, 1 global, 2 scratch), opcode, SDST form,\n'
            '//  name, {{field, kind, bits, output}...}}\n'))
    for (enc, segment, opcode, sdst), (name, ops, _) in sorted(rows.items()):
        o = ', '.join(f'{{"{fl}", K::{k}, {b}, {"true" if out else "false"}}}' for fl, k, b, out in ops)
        f.write(f'{{Enc::{enc}, {segment}, {opcode}, {"true" if sdst else "false"}, "{name}", {{{o}}}}},\n')
if images:
    print(f'{len(rows)} image instructions written to {out_path}')
    sys.exit(0)
# The older names the specification records for renamed instructions
# (global_load_b32 was global_load_dword), which the executor knows them by.
alias_path = out_path.replace('.inc', '_aliases.inc')
with open(alias_path, 'w') as f:
    f.write(f'// {arch}: each renamed instruction\'s earlier names, from the same specification.\n'
            '// Written by amd/tools/rdna-ops.py; do not edit.\n')
    for name, olds in sorted(aliases.items()):
        f.write('{"' + name + '", {' + ', '.join(f'"{o}"' for o in olds) + '}},\n')
print(f'{len(rows)} instructions written to {out_path}, {len(aliases)} renamings to {alias_path}')
