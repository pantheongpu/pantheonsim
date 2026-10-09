#!/usr/bin/env python3
"""Writes nvidia/src/h264_tables.inc: the constant tables of ITU-T H.264 that the
decoder and encoder in this repository need, read out of the Recommendation itself
(the 03/2009 edition, which has the High profiles) rather than typed in:

  Table 7-3, 7-4   default scaling lists
  Table 8-13, 8-14 zig-zag and field scans (4x4 and 8x8)
  Table 8-16, 8-17 deblocking thresholds alpha', beta' and tC0'
  Table 9-5        coeff_token (nC ranges 0-1, 2-3, 4-7, 8+, and the chroma DC code)
  Table 9-7..9-10  total_zeros (4x4 and chroma DC 2x2) and run_before
  Tables 9-12..9-33  CABAC context initialisation (m, n) for I slices and cabac_init_idc 0..2
  Table 9-43       ctxIdxInc of significant_coeff_flag / last_significant_coeff_flag in 8x8 blocks
  Table 9-44       rangeTabLPS

Table 9-45 (the state transition table) is not text in the PDF; transIdxLPS is
written in the decoder source from the Recommendation's table and checked by the
arithmetic decoder reading real streams.

usage: gen_h264_tables.py <T-REC-H.264-200903-S.pdf> <out.inc>
The PDF is the ITU-T's free download (itu.int/rec/T-REC-H.264); it is not kept in the repository.
Needs PyMuPDF (pip install pymupdf).
"""
import re
import sys

import fitz


def rows_of(page, ybin=3):
    """Words of a page grouped into rows: [(y, [(x, text), ...]), ...] in reading order."""
    rows = {}
    for w in page.get_text('words'):
        rows.setdefault(round(w[1] / ybin), []).append((w[0], w[4]))
    return [(y * ybin, sorted(r)) for y, r in sorted(rows.items())]


def find_pages(doc, title):
    """Pages (0-based) whose body starts a table titled `title` (not the table of contents)."""
    out = []
    for i in range(len(doc)):
        t = doc[i].get_text()
        if re.search(r'^Table %s( \(continued\)| \(concluded\))? – ' % re.escape(title), t, re.M) and i > 60:
            out.append(i)
    return out


def num(t):
    return int(t.replace('−', '-'))


# ---------------------------------------------------------------- scaling lists, scans
def default_scaling(doc):
    p = find_pages(doc, '7-3')[0]
    rows = rows_of(doc[p])
    intra4 = inter4 = None
    for y, r in rows:
        txt = [t for x, t in r]
        if txt[:1] == ['Default_4x4_Intra[']:
            intra4 = [num(t) for x, t in r if re.fullmatch(r"\d+", t)]
        if txt[:1] == ['Default_4x4_Inter[']:
            inter4 = [num(t) for x, t in r if re.fullmatch(r'\d+', t)]
    # 8x8: pages with 7-4 and its continuations
    intra8, inter8 = [], []
    for p in range(94, 97):
        for y, r in rows_of(doc[p]):
            txt = [t for x, t in r]
            if txt[:1] == ['Default_8x8_Intra[']:
                intra8 += [num(t) for x, t in r if re.fullmatch(r'\d+', t)]
            if txt[:1] == ['Default_8x8_Inter[']:
                inter8 += [num(t) for x, t in r if re.fullmatch(r'\d+', t)]
    return intra4, inter4, intra8, inter8


def scans(doc):
    """zigzag / field scans as raster offsets (row * size + column); cij = row i, column j."""
    def cij(t, n):
        m = re.fullmatch(r'c(\d)(\d)', t)
        return int(m.group(1)) * n + int(m.group(2))
    zz4, fld4, zz8, fld8 = [], [], [], []
    for y, r in rows_of(doc[202]):
        txt = [t for x, t in r]
        if txt[:1] == ['zig-zag']:
            zz4 = [cij(t, 4) for t in txt[1:]]
        if txt[:1] == ['field']:
            fld4 = [cij(t, 4) for t in txt[1:]]
    for pn in (203,):
        for y, r in rows_of(doc[pn]):
            txt = [t for x, t in r]
            if txt[:1] == ['zig-zag']:
                zz8 += [cij(t, 8) for t in txt[1:]]
            if txt[:1] == ['field']:
                fld8 += [cij(t, 8) for t in txt[1:]]
    return zz4, fld4, zz8, fld8


def deblock(doc):
    alpha, beta = [], []
    for pn in (230,):
        for y, r in rows_of(doc[pn]):
            txt = [t for x, t in r]
            if txt[:1] == ['α′']:
                alpha += [num(t) for t in txt[1:]]
            if txt[:1] == ['β′']:
                beta += [num(t) for t in txt[1:]]
    tc0 = {1: [], 2: [], 3: []}
    for pn in (231,):
        for y, r in rows_of(doc[pn]):
            txt = [t for x, t in r]
            if txt[:3] and txt[0] == 'bS' and txt[1] == '=':
                tc0[int(txt[2])] += [num(t) for t in txt[3:]]
    return alpha, beta, tc0


# ---------------------------------------------------------------- CAVLC
def split_codes(words, bounds):
    """words: [(x, text)] of one table row; bounds: ascending x where each column starts.
    Returns the concatenated bit string of each column ('' where '-' or empty)."""
    cols = [''] * len(bounds)
    for x, t in words:
        k = max(i for i, b in enumerate(bounds) if x >= b - 4)
        cols[k] += t
    return cols


def coeff_token(doc):
    # columns: nC 0..1, 2..3, 4..7, 8+, -1 (chroma DC 4:2:0); the -2 column (4:2:2) is dropped
    bounds = [135, 225, 305, 370, 412, 462]
    table = {}   # (t1, tc) -> [code0..code4]
    for pn in (240, 241, 242):
        for y, r in rows_of(doc[pn], 2):
            ws = [(x, t) for x, t in r]
            if not ws or not re.fullmatch(r'[0-3]', ws[0][1]) or ws[0][0] > 70:
                continue
            if len(ws) < 3 or not re.fullmatch(r'\d+', ws[1][1]) or not (90 < ws[1][0] < 110):
                continue
            t1, tc = int(ws[0][1]), int(ws[1][1])
            cols = split_codes(ws[2:], bounds)
            table[(t1, tc)] = [c.replace('-', '') for c in cols[:5]]
    return table


def total_zeros(doc):
    """Returns tz4[tzVlcIndex 1..15][total_zeros] = code, tzc[1..3][total_zeros] (chroma DC 2x2)."""
    tz4 = {}
    # Table 9-7: tzVlcIndex 1..7 on page 245 (index 244), 9-8: 8..15 on page 246
    def grab(pn, headers_y_text):
        out = {}
        rows = rows_of(doc[pn], 2)
        # header row: the one that is all small integers starting with the first column index
        for i, (y, r) in enumerate(rows):
            nums = [t for x, t in r]
            if all(re.fullmatch(r'\d+', t) for t in nums) and len(nums) in (7, 8, 3) and int(nums[0]) in (1, 8) and y > 200 * 0:
                hdr = [(x, int(t)) for x, t in r]
                if hdr[0][1] != headers_y_text:
                    continue
                bounds = [x for x, _ in hdr]
                for y2, r2 in rows[i + 1:]:
                    ws = r2
                    if not ws or not re.fullmatch(r'\d+', ws[0][1]) or ws[0][0] > bounds[0] - 5:
                        break
                    tz = int(ws[0][1])
                    cols = split_codes(ws[1:], bounds)
                    for (bx, idx), c in zip(hdr, cols):
                        c = c.replace('-', '')
                        if c:
                            out.setdefault(idx, {})[tz] = c
                return out
        return out
    tz4.update(grab(244, 1))
    tz4.update(grab(245, 8))
    # chroma DC 2x2: page 246, header row 1 2 3 (three entries)
    tzc = {}
    rows = rows_of(doc[245], 2)
    for i, (y, r) in enumerate(rows):
        nums = [t for x, t in r]
        if len(nums) == 3 and nums == ['1', '2', '3']:
            bounds = [x for x, _ in r]
            for y2, r2 in rows[i + 1:]:
                if not r2 or not re.fullmatch(r'\d', r2[0][1]) or r2[0][0] > bounds[0] - 5:
                    break
                tz = int(r2[0][1])
                cols = split_codes(r2[1:], bounds)
                for idx, c in zip((1, 2, 3), cols):
                    c = c.replace('-', '')
                    if c:
                        tzc.setdefault(idx, {})[tz] = c
            break
    return tz4, tzc


def run_before(doc):
    rb = {}
    rows = rows_of(doc[246], 2)
    for i, (y, r) in enumerate(rows):
        nums = [t for x, t in r]
        if nums == ['1', '2', '3', '4', '5', '6', '>6']:
            bounds = [x for x, _ in r]
            for y2, r2 in rows[i + 1:]:
                if not r2 or not re.fullmatch(r'\d+', r2[0][1]) or r2[0][0] > bounds[0] - 5:
                    break
                run = int(r2[0][1])
                cols = split_codes(r2[1:], bounds)
                for k, c in enumerate(cols):
                    c = c.replace('-', '')
                    if c:
                        rb.setdefault(k + 1, {})[run] = c
            break
    return rb


# ---------------------------------------------------------------- CABAC
def cabac_init(doc):
    """Narrow tables (9-12 .. 9-17): a ctxIdx header row, then 'm' and 'n' rows, with a label before
    each (I slices / cabac_init_idc 0..2) or none (the same values for every slice type)."""
    init = {}
    for pn in range(251, 254):
        rows = rows_of(doc[pn], 3)
        for i, (y, r) in enumerate(rows):
            toks = [t for x, t in r]
            if len(toks) >= 6 and all(re.fullmatch(r'\d+', t) for t in toks) and all(int(toks[k + 1]) == int(toks[k]) + 1 for k in range(len(toks) - 1)) \
                    or (toks[-3:] == ['399', '400', '401']):
                if not (i + 1 < len(rows) and 'm' in [t for x, t in rows[i + 1][1]]):
                    continue
                ctxs = [int(t) for t in toks]
                j = i + 1
                mrow = None
                while j < len(rows):
                    tt = [t for x, t in rows[j][1]]
                    if 'm' in tt or 'n' in tt:
                        k = tt.index('m') if 'm' in tt else tt.index('n')
                        label = tt[:k]
                        if label and label[0] == 'I':
                            sets = [0]
                        elif len(label) == 1 and re.fullmatch(r'[012]', label[0]):
                            sets = [int(label[0]) + 1]
                        else:
                            sets = [0, 1, 2, 3]
                        vals = tt[k + 1:]
                        if tt[k] == 'm':
                            mrow = (sets, vals)
                        else:
                            sets, mv = mrow
                            assert len(mv) == len(vals) == len(ctxs), (ctxs, mv, vals)
                            for a, b, c in zip(ctxs, mv, vals):
                                if b == 'na':
                                    continue
                                ent = init.setdefault(a, [None] * 4)
                                for s in sets:
                                    ent[s] = (num(b), num(c))
                        j += 1
                        continue
                    break
    return init


def cabac_init_wide(doc, init):
    """The wide tables (9-18 ...): rows 'ctxIdx mI nI m0 n0 m1 n1 m2 n2' twice per line."""
    for pn in range(254, 273):
        for y, r in rows_of(doc[pn], 3):
            toks = [t for x, t in r]
            if not all(re.fullmatch(r'[−-]?\d+|na', t) for t in toks):
                continue
            if len(toks) == 9 or len(toks) == 18 or len(toks) == 7 + 0 and False:
                groups = [toks[i:i + 9] for i in range(0, len(toks), 9)]
            elif len(toks) in (8, 17):
                continue
            else:
                continue
            for g in groups:
                ctx = int(g[0])
                if pn == 261 and ctx == 641:
                    ctx = 541   # the 03/2009 edition prints 641 in this row of Table 9-26 (between 540 and 542)
                vals = g[1:]
                pairs = [(num(vals[2 * k]), num(vals[2 * k + 1])) for k in range(4)]
                init[ctx] = pairs
    return init


def range_tab_lps(doc):
    tab = [[0] * 4 for _ in range(64)]
    rows = rows_of(doc[299], 3)
    for y, r in rows:
        toks = [(x, t) for x, t in r]
        if len(toks) >= 10 and all(re.fullmatch(r'\d+', t) for x, t in toks):
            v = [int(t) for x, t in toks]
            if len(v) == 10 and v[0] < 32 and v[5] >= 32:
                tab[v[0]] = v[1:5]
                tab[v[5]] = v[6:10]
    return tab


def sig8x8(doc):
    """Table 9-43: (frame sig, field sig, last) ctxIdxInc for levelListIdx 0..62; a row holds two groups of
    four numbers (levelListIdx, frame, field, last)."""
    frame, field, last = {}, {}, {}
    for pn in (294, 295):
        for y, r in rows_of(doc[pn], 3):
            toks = [t for x, t in r]
            if len(toks) in (8, 4) and all(re.fullmatch(r'\d+', t) for t in toks) and r[0][0] < 130:
                v = [int(t) for t in toks]
                for g in range(0, len(v), 4):
                    frame[v[g]], field[v[g]], last[v[g]] = v[g + 1], v[g + 2], v[g + 3]
    return frame, field, last


def coded_block_pattern(doc):
    """Table 9-4 (a): codeNum -> coded_block_pattern for Intra_4x4/Intra_8x8 and for Inter, ChromaArrayType 1 or 2."""
    intra, inter = {}, {}
    for pn in (235, 236, 237):
        rows = rows_of(doc[pn], 3)
        started = False
        for y, r in rows:
            toks = [t for x, t in r]
            if toks[:3] == ['(b)', 'ChromaArrayType', 'is']:
                break
            if len(toks) == 3 and all(re.fullmatch(r'\d+', t) for t in toks) and r[0][0] > 190 and r[0][0] < 210 and r[1][0] > 290 and r[2][0] > 380:
                k = int(toks[0])
                intra[k], inter[k] = int(toks[1]), int(toks[2])
    return intra, inter


def main(pdf, out):
    doc = fitz.open(pdf)
    o = []
    w = o.append
    w('// Generated by nvidia/tools/gen_h264_tables.py from ITU-T H.264 (03/2009). Do not edit.\nstruct VlcCode {\n  uint8_t len;\n  uint16_t bits;\n};\n')
    i4, j4, i8, j8 = default_scaling(doc)
    assert len(i4) == 16 and len(j4) == 16 and len(i8) == 64 and len(j8) == 64, (len(i4), len(j4), len(i8), len(j8))
    zz4, f4, zz8, f8 = scans(doc)
    assert len(zz4) == 16 and len(f4) == 16 and len(zz8) == 64 and len(f8) == 64, (len(zz4), len(f4), len(zz8), len(f8))
    assert sorted(zz4) == list(range(16)) and sorted(f4) == list(range(16)) and sorted(zz8) == list(range(64)) and sorted(f8) == list(range(64))

    def arr(name, ty, vals, per=16):
        s = 'static const %s %s[%d] = {' % (ty, name, len(vals))
        for k, v in enumerate(vals):
            if k % per == 0:
                s += '\n    '
            s += '%d, ' % v
        return s.rstrip(', ') + '\n};\n'
    w(arr('kDefault4x4Intra', 'uint8_t', i4))
    w(arr('kDefault4x4Inter', 'uint8_t', j4))
    w(arr('kDefault8x8Intra', 'uint8_t', i8))
    w(arr('kDefault8x8Inter', 'uint8_t', j8))
    w('// scan[idx] = raster offset (row * size + column) of the idx-th coefficient\n')
    w(arr('kZigzag4x4', 'uint8_t', zz4))
    w(arr('kField4x4', 'uint8_t', f4))
    w(arr('kZigzag8x8', 'uint8_t', zz8))
    w(arr('kField8x8', 'uint8_t', f8))
    alpha, beta, tc0 = deblock(doc)
    assert len(alpha) == 52 and len(beta) == 52 and all(len(v) == 52 for v in tc0.values()), (len(alpha), len(beta), [len(v) for v in tc0.values()])
    w(arr('kAlpha', 'uint8_t', alpha, 26))
    w(arr('kBeta', 'uint8_t', beta, 26))
    w('static const uint8_t kTc0[3][52] = {\n')
    for bs in (1, 2, 3):
        w('    {' + ', '.join(str(v) for v in tc0[bs]) + '},\n')
    w('};\n')

    ct = coeff_token(doc)
    # 4 luma columns x (t1 0..3, tc 0..16), chroma DC column
    exp = {(t1, tc) for tc in range(17) for t1 in range(min(tc, 3) + 1)}
    assert set(ct) == exp, sorted(exp - set(ct))[:10]
    names = ['kCoeffToken0', 'kCoeffToken1', 'kCoeffToken2', 'kCoeffToken3', 'kCoeffTokenChromaDc']
    w('// coeff_token: {bit length, code} indexed [TrailingOnes][TotalCoeff]; length 0 = not a code\n')
    for c in range(5):
        w('static const VlcCode %s[4][17] = {\n' % names[c])
        for t1 in range(4):
            ent = []
            for tc in range(17):
                code = ct.get((t1, tc), [''] * 5)[c]
                ent.append('{%d, 0x%x}' % (len(code), int(code, 2) if code else 0))
            w('    {' + ', '.join(ent) + '},\n')
        w('};\n')
    tz4, tzc = total_zeros(doc)
    assert sorted(tz4) == list(range(1, 16)), sorted(tz4)
    for idx in range(1, 16):
        assert sorted(tz4[idx]) == list(range(0, 17 - idx)), (idx, sorted(tz4[idx]))
    assert sorted(tzc) == [1, 2, 3]
    w('// total_zeros for 4x4 blocks: [tzVlcIndex - 1][total_zeros]\n')
    w('static const VlcCode kTotalZeros[15][16] = {\n')
    for idx in range(1, 16):
        ent = []
        for tz in range(16):
            code = tz4[idx].get(tz, '')
            ent.append('{%d, 0x%x}' % (len(code), int(code, 2) if code else 0))
        w('    {' + ', '.join(ent) + '},\n')
    w('};\n')
    w('// total_zeros for 2x2 chroma DC: [tzVlcIndex - 1][total_zeros]\n')
    w('static const VlcCode kTotalZerosChromaDc[3][4] = {\n')
    for idx in (1, 2, 3):
        ent = []
        for tz in range(4):
            code = tzc[idx].get(tz, '')
            ent.append('{%d, 0x%x}' % (len(code), int(code, 2) if code else 0))
        w('    {' + ', '.join(ent) + '},\n')
    w('};\n')
    rb = run_before(doc)
    assert sorted(rb) == list(range(1, 8)), sorted(rb)
    w('// run_before: [min(zerosLeft, 7) - 1][run_before]\n')
    w('static const VlcCode kRunBefore[7][15] = {\n')
    for z in range(1, 8):
        ent = []
        for run in range(15):
            code = rb[z].get(run, '')
            ent.append('{%d, 0x%x}' % (len(code), int(code, 2) if code else 0))
        w('    {' + ', '.join(ent) + '},\n')
    w('};\n')
    # prefix-freeness checks
    def check_prefix_free(codes, what):
        cs = [c for c in codes if c]
        for a in cs:
            for b in cs:
                if a is not b and a != b and b.startswith(a):
                    raise SystemExit('%s: %s is a prefix of %s' % (what, a, b))
        if len(set(cs)) != len(cs):
            raise SystemExit('%s: duplicate code' % what)
    for c in range(5):
        check_prefix_free([ct[k][c] for k in ct], 'coeff_token col %d' % c)
    for idx in range(1, 16):
        check_prefix_free(list(tz4[idx].values()), 'total_zeros %d' % idx)
    for idx in (1, 2, 3):
        check_prefix_free(list(tzc[idx].values()), 'total_zeros chroma %d' % idx)
    for z in range(1, 8):
        check_prefix_free(list(rb[z].values()), 'run_before %d' % z)

    init = cabac_init(doc)
    init = cabac_init_wide(doc, init)
    missing = [c for c in range(1024) if c != 276 and c not in init]
    assert not missing, missing[:40]
    w('// CABAC initialisation: [ctxIdx][0 = I slices, 1..3 = cabac_init_idc 0..2][m, n]; ctx without a value for I slices hold 0,0\n')
    w('static const int8_t kCabacInit[1024][4][2] = {\n')
    for c in range(1024):
        pairs = init.get(c, [(0, 0)] * 4)
        pairs = [p if p is not None else (0, 0) for p in pairs]
        w('    {' + ', '.join('{%d, %d}' % p for p in pairs) + '},  // %d\n' % c)
    w('};\n')
    # which contexts have I-slice values (for a self check only)
    tab = range_tab_lps(doc)
    assert tab[0] == [128, 176, 208, 240] and tab[63] == [2, 2, 2, 2] and all(all(v for v in row) for row in tab)
    w('static const uint8_t kRangeTabLps[64][4] = {\n')
    for row in tab:
        w('    {' + ', '.join(str(v) for v in row) + '},\n')
    w('};\n')
    f, fl, la = sig8x8(doc)
    assert sorted(f) == list(range(63)) and sorted(fl) == list(range(63)) and sorted(la) == list(range(63)), (len(f), len(fl), len(la))
    w(arr('kSig8x8Frame', 'uint8_t', [f[i] for i in range(63)]))
    w(arr('kSig8x8Field', 'uint8_t', [fl[i] for i in range(63)]))
    w(arr('kLast8x8', 'uint8_t', [la[i] for i in range(63)]))
    ci, ce = coded_block_pattern(doc)
    assert sorted(ci) == list(range(48)) and sorted(ce) == list(range(48)), (len(ci), len(ce))
    assert sorted(ci.values()) == list(range(48)) and sorted(ce.values()) == list(range(48))
    w('// coded_block_pattern of me(v) (ChromaArrayType 1 or 2), indexed by codeNum: Intra_4x4/Intra_8x8, then Inter\n')
    w(arr('kCbpIntra', 'uint8_t', [ci[i] for i in range(48)]))
    w(arr('kCbpInter', 'uint8_t', [ce[i] for i in range(48)]))
    open(out, 'w').write(''.join(o))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
