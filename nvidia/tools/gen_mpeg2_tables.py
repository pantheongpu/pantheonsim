#!/usr/bin/env python3
"""Extracts the variable length code tables of ITU-T Rec. H.262 | ISO/IEC 13818-2 (2000 edition) Annex B, the scan orders of
Figures 7-2 and 7-3 and the default intra quantisation matrix from the PDF and writes nvidia/src/mpeg2_tables.inc.

usage: gen_mpeg2_tables.py T-REC-H.262-200002-S.pdf > nvidia/src/mpeg2_tables.inc

Every table is checked for being a prefix code (no code is the start of another) before it is written."""
import sys
import fitz
fitz.TOOLS.mupdf_display_errors(False)

PAGES = {'B1': 134, 'B9': 140, 'B10': 141, 'B14': [143, 144, 145, 146], 'B15': [147, 148, 149, 150], 'scan': 78, 'defmat': 64}


def rows_of(doc, pn):
    p = doc[pn - 1]
    rows = {}
    for w in p.get_text('words'):
        y = round(w[1] / 3)
        rows.setdefault(y, []).append((w[0], w[4]))
    return [sorted(rows[y]) for y in sorted(rows)]


def isbin(t):
    return t != '' and all(c in '01' for c in t)


def check_prefix(name, codes):
    for a, _ in codes:
        for b, _ in codes:
            if a is not b and a != b and b.startswith(a):
                raise SystemExit('%s: %s is a prefix of %s' % (name, a, b))
    seen = set()
    for c, _ in codes:
        if c in seen:
            raise SystemExit('%s: duplicate code %s' % (name, c))
        seen.add(c)


def kraft(codes):
    return sum(2.0 ** -len(c) for c, _ in codes)


def parse_b1(doc):
    out = []
    for r in rows_of(doc, PAGES['B1']):
        left = [t for x, t in r if x < 235 and isbin(t)]
        lval = [t for x, t in r if 235 <= x < 250 and t.isdigit()]
        right = [t for x, t in r if 300 <= x < 420 and isbin(t)]
        rval = [t for x, t in r if x >= 424]
        if left and lval and len(left) >= 1 and left[0] and r[0][0] < 100:
            out.append((''.join(left), int(lval[0])))
        if right and rval:
            v = rval[0]
            out.append((''.join(right), 34 if v.startswith('macroblock_escape') else int(v)))
    return out


def parse_b9(doc):
    out = []
    for r in rows_of(doc, PAGES['B9']):
        if not r or r[0][0] > 130 or not isbin(r[0][1]):
            continue
        left = [t for x, t in r if x < 250 and isbin(t)]
        lval = [t for x, t in r if 250 <= x <= 262 and t.isdigit()]
        right = [t for x, t in r if 300 <= x < 420 and isbin(t)]
        rval = [t for x, t in r if x >= 424 and t.isdigit()]
        if left and lval:
            out.append((''.join(left), int(lval[0])))
        if right and rval:
            out.append((''.join(right), int(rval[0])))
    return out


def parse_b10(doc):
    out = []
    for r in rows_of(doc, PAGES['B10']):
        code = [t for x, t in r if 155 <= x < 330 and isbin(t)]
        val = [t for x, t in r if x >= 355]
        if code and val and code[0] and r[0][0] < 170:
            v = val[0].replace('–', '-').replace('−', '-')
            try:
                out.append((''.join(code), int(v)))
            except ValueError:
                pass
    return out


def parse_dct(doc, pages):
    out = []   # (code, run, level); run -1: end of block, -2: escape
    first_only = None
    for pn in pages:
        for r in rows_of(doc, pn):
            toks = [(x, t) for x, t in r]
            if any(t == 'B.16' for _, t in toks):
                break
            if not toks or toks[0][0] > 130 or not (isbin(toks[0][1]) or (toks[0][1].endswith('s') and isbin(toks[0][1][:-1]))):
                continue
            if any(t == 'NOTE' for _, t in toks):
                continue
            code = ''
            sign = False
            for x, t in toks:
                if x >= 300:
                    break
                if isbin(t):
                    code += t
                elif t == 's':
                    sign = True
                elif t.endswith('s') and isbin(t[:-1]):
                    code += t[:-1]
                    sign = True
            words = [t for x, t in toks]
            if 'Escape' in words:
                out.append((code, -2, 0, False))
                continue
            if 'End' in words:
                out.append((code, -1, 0, False))
                continue
            nums = [t for x, t in toks if x >= 300 and t.isdigit()]
            if len(nums) < 2 or not sign:
                raise SystemExit('unparsed DCT row on page %d: %r' % (pn, toks))
            out.append((code, int(nums[0]), int(nums[1]), 'Note' in ''.join(words) and False))
    return out


def parse_scan(doc):
    rs = rows_of(doc, PAGES['scan'])
    tabs = []
    cur = []
    for r in rs:
        dig = [(x, t) for x, t in r if t.isdigit()]
        nums = [t for x, t in dig]
        if len(nums) == 9 and 205 < dig[0][0] < 220 and int(nums[0]) == len(cur):
            cur.append([int(n) for n in nums[1:]])
        if len(cur) == 8:
            tabs.append(cur)
            cur = []
    return tabs   # [alt][v][u] = n


def parse_defmat(doc):
    rows = []
    for r in rows_of(doc, PAGES['defmat']):
        dig = [(x, t) for x, t in r if t.isdigit()]
        nums = [t for x, t in dig]
        if len(nums) == 9 and 205 < dig[0][0] < 220 and int(nums[0]) == len(rows):
            rows.append([int(n) for n in nums[1:]])
        if len(rows) == 8:
            break
    return rows


def emit_vlc(name, codes, note=''):
    print('// %s%s' % (name, note))
    print('const VlcEntry %s[] = {' % name)
    for c, v in codes:
        print('    {0x%x, %d, %d},   // %s' % (int(c, 2), len(c), v, c))
    print('};')


def main():
    doc = fitz.open(sys.argv[1])
    print('// Generated by nvidia/tools/gen_mpeg2_tables.py from ITU-T Rec. H.262 (02/2000) Annex B, Figures 7-2 and 7-3 and 6.3.11. Do not edit.')
    b1 = parse_b1(doc)
    b9 = parse_b9(doc)
    b10 = parse_b10(doc)
    assert len(b1) == 34 and len(b9) == 64 and len(b10) == 33, (len(b1), len(b9), len(b10))
    for n, t in (('B.1', b1), ('B.9', b9), ('B.10', b10)):
        check_prefix(n, t)
    assert sorted(v for _, v in b1) == list(range(1, 35))
    assert sorted(v for _, v in b9) == list(range(64))
    assert sorted(v for _, v in b10) == list(range(-16, 17))
    emit_vlc('kMbAddrIncVlc', b1, ' (Table B.1; value 34 is macroblock_escape)')
    emit_vlc('kCbpVlc', b9, ' (Table B.9; coded_block_pattern_420)')
    emit_vlc('kMotionVlc', b10, ' (Table B.10; motion_code)')
    for tname, key in (('kDctZeroVlc', 'B14'), ('kDctOneVlc', 'B15')):
        d = parse_dct(doc, PAGES[key])
        print('// %s: Table %s; run -1 end of block, -2 escape; the code lengths include no sign bit' % (tname, key.replace('B', 'B.')))
        print('const DctEntry %s[] = {' % tname)
        for c, run, lev, _ in d:
            print('    {0x%x, %d, %d, %d},   // %s' % (int(c, 2), len(c), run, lev, c))
        print('};')
        nonspecial = [(c, r) for c, r, l, _ in d]
        # B.14 has two codes starting with 1: '1' (first coefficient of a non-intra block) and '11' (the rest); they are not prefix-free
        # on their own, so the check leaves the first-coefficient code out
        chk = [(c, r) for c, r in nonspecial if not (key == 'B14' and c == '1')]
        check_prefix(key, chk)
        print('// %s: %d entries, Kraft sum %.6f' % (tname, len(d), kraft(chk)), file=sys.stderr)
    scans = parse_scan(doc)
    assert len(scans) == 2
    for a, tab in enumerate(scans):
        flat = [x for row in tab for x in row]
        assert sorted(flat) == list(range(64))
    print('// kScanPos[alternate_scan][n] = v * 8 + u of the n-th coefficient in scan order (Figures 7-2 and 7-3)')
    print('const uint8_t kScanPos[2][64] = {')
    for tab in scans:
        pos = [0] * 64
        for v in range(8):
            for u in range(8):
                pos[tab[v][u]] = v * 8 + u
        print('    {' + ', '.join(str(p) for p in pos) + '},')
    print('};')
    dm = parse_defmat(doc)
    assert len(dm) == 8
    print('// kDefaultIntraMatrix[v * 8 + u] (6.3.11)')
    print('const uint8_t kDefaultIntraMatrix[64] = {' + ', '.join(str(x) for row in dm for x in row) + '};')
    print('// %d B.1, %d B.9, %d B.10 codes' % (len(b1), len(b9), len(b10)), file=sys.stderr)


main()
