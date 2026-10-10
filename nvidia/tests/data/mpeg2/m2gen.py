#!/usr/bin/env python3
"""A writer of MPEG-2 video elementary streams with chosen or random syntax, for the fixtures FFmpeg's encoder cannot make: field
pictures, 16x8 and dual prime motion compensation, concealment motion vectors, pulldown flags, every combination of the picture coding
extension, and blocks with chosen coefficients (which probe the inverse DCT of whatever decodes them).

Nothing here decodes: the pictures are made of random syntax elements that obey the bitstream's rules (motion vectors that stay inside the
reference pictures, DC values inside the legal range, ...). Decoders are compared on what they make of them. The variable length codes
come from nvidia/src/mpeg2_tables.inc, the tables the decoder reads, extracted from Rec. H.262.

usage as a module: import m2gen; w = m2gen.Stream(...); w.sequence(...); w.picture(...); open(f, 'wb').write(w.bytes())
"""
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, '..', '..', '..', 'src', 'mpeg2_tables.inc')


def load_tables(path=TABLES):
    text = open(path).read()
    out = {}
    for name in ('kMbAddrIncVlc', 'kCbpVlc', 'kMotionVlc', 'kDctZeroVlc', 'kDctOneVlc'):
        m = re.search(r'const \w+ %s\[\] = \{(.*?)\n\};' % name, text, re.S)
        rows = re.findall(r'\{0x([0-9a-f]+), (\d+), (-?\d+)(?:, (-?\d+))?\}', m.group(1))
        out[name] = [tuple(int(x, 16) if i == 0 else int(x) for i, x in enumerate(r) if x != '') for r in rows]
    m = re.search(r'kScanPos\[2\]\[64\] = \{(.*?)\n\};', text, re.S)
    out['scan'] = [[int(x) for x in row.split(',')] for row in re.findall(r'\{([^{}]*)\}', m.group(1))]
    return out


T = load_tables()
ZIGZAG = T['scan'][0]
ALTSCAN = T['scan'][1]

# macroblock_type codes: (code, len) by (quant, fwd, bwd, pattern, intra)
MBTYPE = {
    1: {(0, 0, 0, 0, 1): (1, 1), (1, 0, 0, 0, 1): (1, 2)},
    2: {(0, 1, 0, 1, 0): (1, 1), (0, 0, 0, 1, 0): (1, 2), (0, 1, 0, 0, 0): (1, 3), (0, 0, 0, 0, 1): (3, 5), (1, 1, 0, 1, 0): (2, 5),
        (1, 0, 0, 1, 0): (1, 5), (1, 0, 0, 0, 1): (1, 6)},
    3: {(0, 1, 1, 0, 0): (2, 2), (0, 1, 1, 1, 0): (3, 2), (0, 0, 1, 0, 0): (2, 3), (0, 0, 1, 1, 0): (3, 3), (0, 1, 0, 0, 0): (2, 4),
        (0, 1, 0, 1, 0): (3, 4), (0, 0, 0, 0, 1): (3, 5), (1, 1, 1, 1, 0): (2, 5), (1, 1, 0, 1, 0): (3, 6), (1, 0, 1, 1, 0): (2, 6),
        (1, 0, 0, 0, 1): (1, 6)},
}
DC_LUMA = [(4, 3), (0, 2), (1, 2), (5, 3), (6, 3), (0xe, 4), (0x1e, 5), (0x3e, 6), (0x7e, 7), (0xfe, 8), (0x1fe, 9), (0x1ff, 9)]
DC_CHROMA = [(0, 2), (1, 2), (2, 2), (6, 3), (0xe, 4), (0x1e, 5), (0x3e, 6), (0x7e, 7), (0xfe, 8), (0x1fe, 9), (0x3fe, 10), (0x3ff, 10)]
QSCALE = [[2 * i for i in range(32)],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112]]
DEFAULT_INTRA = [8, 16, 19, 22, 26, 27, 29, 34, 16, 16, 22, 24, 27, 29, 34, 37, 19, 22, 26, 27, 29, 34, 34, 38, 22, 22, 26, 27, 29, 34, 37, 40,
                 22, 26, 27, 29, 32, 35, 40, 48, 26, 27, 29, 32, 35, 40, 48, 58, 26, 27, 29, 34, 38, 46, 56, 69, 27, 29, 35, 38, 46, 56, 69, 83]


class Bits:
    def __init__(self):
        self.buf = bytearray()
        self.cur = 0
        self.n = 0

    def put(self, value, nbits):
        assert 0 <= value < (1 << nbits) or nbits == 0, (value, nbits)
        for i in range(nbits - 1, -1, -1):
            self.cur = (self.cur << 1) | ((value >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.buf.append(self.cur)
                self.cur = 0
                self.n = 0

    def align(self):
        while self.n:
            self.put(0, 1)

    def start_code(self, code):
        self.align()
        self.buf += bytes([0, 0, 1, code])

    def bytes(self):
        self.align()
        return bytes(self.buf)


def vlc_table(name):
    return {e[2]: (e[0], e[1]) for e in T[name]}


MB_INC = vlc_table('kMbAddrIncVlc')
CBP = vlc_table('kCbpVlc')
MOTION = vlc_table('kMotionVlc')


def dct_table(name):
    d = {}
    esc = eob = first = None
    for code, ln, run, level in T[name]:
        if run == -1:
            eob = (code, ln)
        elif run == -2:
            esc = (code, ln)
        elif ln == 1 and name == 'kDctZeroVlc':
            first = (code, ln)
        else:
            d[(run, level)] = (code, ln)
    return d, esc, eob, first


DCT0 = dct_table('kDctZeroVlc')
DCT1 = dct_table('kDctOneVlc')


class Stream:
    def __init__(self, width, height, progressive_sequence=True, frame_rate_code=5, aspect=1, seed=1, intra_matrix=None, inter_matrix=None,
                 profile_level=0x48, low_delay=0, bit_rate=0x3ffff, display=None):
        self.w = width
        self.h = height
        self.prog = progressive_sequence
        self.out = bytearray()
        self.rnd = random.Random(seed)
        self.mbw = (width + 15) // 16
        self.mbh = (height + 15) // 16 if progressive_sequence else 2 * ((height + 31) // 32)
        self.frame_rate_code = frame_rate_code
        self.aspect = aspect
        self.intra_matrix = intra_matrix
        self.inter_matrix = inter_matrix
        self.matrices = [list(intra_matrix) if intra_matrix else list(DEFAULT_INTRA), list(inter_matrix) if inter_matrix else [16] * 64]
        self.profile_level = profile_level
        self.low_delay = low_delay
        self.bit_rate = bit_rate
        self.display = display

    # ---- headers ----
    def sequence_header(self, repeat_matrices=True):
        b = Bits()
        b.start_code(0xB3)
        b.put(self.w & 0xfff, 12)
        b.put(self.h & 0xfff, 12)
        b.put(self.aspect, 4)
        b.put(self.frame_rate_code, 4)
        b.put(self.bit_rate & 0x3ffff, 18)
        b.put(1, 1)
        b.put(112, 10)
        b.put(0, 1)
        for m, loaded in ((self.intra_matrix, self.intra_matrix is not None), (self.inter_matrix, self.inter_matrix is not None)):
            b.put(1 if loaded else 0, 1)
            if loaded:
                zz = ZIGZAG
                for n in range(64):
                    b.put(m[zz[n]], 8)
        # sequence_extension
        b.start_code(0xB5)
        b.put(1, 4)
        b.put(self.profile_level, 8)
        b.put(1 if self.prog else 0, 1)
        b.put(1, 2)
        b.put(self.w >> 12, 2)
        b.put(self.h >> 12, 2)
        b.put(self.bit_rate >> 18, 12)
        b.put(1, 1)
        b.put(0, 8)
        b.put(self.low_delay, 1)
        b.put(0, 2)
        b.put(0, 5)
        if self.display:
            b.start_code(0xB5)
            b.put(2, 4)
            b.put(self.display.get('video_format', 5), 3)
            cd = self.display.get('colour', None)
            b.put(1 if cd else 0, 1)
            if cd:
                for v in cd:
                    b.put(v, 8)
            b.put(self.display.get('w', self.w), 14)
            b.put(1, 1)
            b.put(self.display.get('h', self.h), 14)
        self.out += b.bytes()

    def gop(self, closed=True, broken=False, time_code=0):
        b = Bits()
        b.start_code(0xB8)
        b.put(time_code, 25)
        b.put(1 if closed else 0, 1)
        b.put(1 if broken else 0, 1)
        self.out += b.bytes()

    def sequence_end(self):
        self.out += bytes([0, 0, 1, 0xB7])

    def bytes(self):
        return bytes(self.out)

    # ---- pictures ----
    def picture(self, ptype, temporal_reference, structure=3, tff=True, prog_frame=None, repeat_first_field=False, f_code=None, intra_dc_precision=0,
                frame_pred_frame_dct=None, concealment=False, q_scale_type=0, intra_vlc_format=0, alternate_scan=0, mb_source=None, quant_ext=None,
                refs=None, density=0.3, vbv_delay=0xffff, chroma_420_type=None, slice_qcode=0, **extra):
        """Writes one picture (a frame, or one field when structure is 1 or 2). mb_source(pic) -> list per row of macroblock descriptors is
        optional: when None the macroblocks are random with the given density of coded blocks."""
        if frame_pred_frame_dct is None:
            frame_pred_frame_dct = 1 if self.prog else 0
        if prog_frame is None:
            prog_frame = 1 if self.prog else 0
        if f_code is None:
            f_code = [[3, 3], [3, 3]] if ptype == 3 else ([[3, 3], [15, 15]] if (ptype == 2 or concealment) else [[15, 15], [15, 15]])
        if chroma_420_type is None:
            chroma_420_type = prog_frame
        pic = dict(ptype=ptype, structure=structure, tff=tff, f_code=f_code, dc_prec=intra_dc_precision, fpfd=frame_pred_frame_dct,
                   conceal=concealment, qst=q_scale_type, ivlc=intra_vlc_format, alt=alternate_scan, density=density, slice_qcode=slice_qcode)
        pic.update(extra)
        b = Bits()
        b.start_code(0x00)
        b.put(temporal_reference & 0x3ff, 10)
        b.put(ptype, 3)
        b.put(vbv_delay, 16)
        if ptype in (2, 3):
            b.put(0, 1)
            b.put(7, 3)
        if ptype == 3:
            b.put(0, 1)
            b.put(7, 3)
        b.put(0, 1)   # extra_bit_picture
        b.start_code(0xB5)
        b.put(8, 4)
        for s in range(2):
            for t in range(2):
                b.put(f_code[s][t], 4)
        b.put(intra_dc_precision, 2)
        b.put(structure, 2)
        b.put(1 if tff else 0, 1)
        b.put(frame_pred_frame_dct, 1)
        b.put(1 if concealment else 0, 1)
        b.put(q_scale_type, 1)
        b.put(intra_vlc_format, 1)
        b.put(alternate_scan, 1)
        b.put(1 if repeat_first_field else 0, 1)
        b.put(chroma_420_type, 1)
        b.put(prog_frame, 1)
        b.put(0, 1)
        if quant_ext:
            b.start_code(0xB5)
            b.put(3, 4)
            li, lni = quant_ext.get('intra'), quant_ext.get('inter')
            b.put(1 if li else 0, 1)
            if li:
                for n in range(64):
                    b.put(li[ZIGZAG[n]], 8)
                self.matrices[0] = list(li)
            b.put(1 if lni else 0, 1)
            if lni:
                for n in range(64):
                    b.put(lni[ZIGZAG[n]], 8)
                self.matrices[1] = list(lni)
            b.put(0, 1)
            b.put(0, 1)
        pic['bits'] = b
        rows = self.mbh // 2 if structure != 3 else self.mbh
        for row in range(rows):
            self.slice(pic, row, mb_source)
        self.out += b.bytes()

    # ---- slices and macroblocks ----
    def slice(self, pic, row, mb_source):
        b = pic['bits']
        b.start_code(1 + row)
        qcode = pic['slice_qcode'] or (self.rnd.randint(1, 31) if pic['qst'] else self.rnd.randint(1, 20))
        st = dict(qcode=qcode, dc=[128 << pic['dc_prec']] * 3, pmv=[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], prev=None, row=row)
        b.put(qcode, 5)
        b.put(0, 1)
        skipped = 0
        first = True
        explicit = mb_source(pic, row, self) if mb_source else None
        for col in range(self.mbw):
            mb = explicit[col] if explicit is not None else self.next_mb(pic, st, col, row)
            if mb is None:     # skipped macroblock
                skipped += 1
                if pic['ptype'] == 2:
                    st['pmv'] = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
                st['dc'] = [128 << pic['dc_prec']] * 3
                continue
            if first:
                inc = col + 1
                first = False
            else:
                inc = skipped + 1
            skipped = 0
            self.put_increment(b, inc)
            self.put_macroblock(pic, st, mb)
        # a slice cannot end with a skipped macroblock: next_mb never skips the last one

    def put_increment(self, b, inc):
        while inc > 33:
            b.put(0x8, 11)
            inc -= 33
        code, ln = MB_INC[inc]
        b.put(code, ln)



# ======================================================================================================================
# macroblock writing and random macroblock generation
# ======================================================================================================================
def div2_away(x):
    return (x + 1) // 2 if x >= 0 else -((-x + 1) // 2)


def trunc_div(x, d):
    q = abs(x) // d
    return q if x >= 0 else -q


def fits(rw, rh, x, y, bw, bh, vx, vy):
    """The block at (x, y) displaced by the half-sample vector lies inside a reference of rw x rh samples."""
    ix, iy = x + (vx >> 1), y + (vy >> 1)
    hx, hy = vx & 1, vy & 1
    return ix >= 0 and iy >= 0 and ix + bw + hx <= rw and iy + bh + hy <= rh


def put_motion_component(b, delta, f):
    """motion_code and motion_residual for a delta already in [-16 f, 16 f - 1]."""
    if delta == 0:
        code, res = 0, 0
    elif f == 1:
        code, res = delta, 0
    else:
        a = abs(delta)
        code = (a - 1) // f + 1
        res = (a - 1) % f
        if delta < 0:
            code = -code
    c, ln = MOTION[code]
    b.put(c, ln)
    if f != 1 and code != 0:
        b.put(res, f.bit_length() - 1)


def encode_coefs(b, tbl, coefs, start, first_nonintra):
    """coefs: list of (n, level) with n the scan index, ascending. Writes run/level codes and the end of block."""
    d, esc, eob, first = tbl
    prev = start - 1
    for idx, (n, level) in enumerate(coefs):
        run = n - prev - 1
        prev = n
        a = abs(level)
        sign = 1 if level < 0 else 0
        if first_nonintra and idx == 0 and run == 0 and a == 1:
            b.put(first[0], first[1])
            b.put(sign, 1)
        elif (run, a) in d and not (first_nonintra and idx == 0 and (run, a) == (0, 1)):
            code, ln = d[(run, a)]
            b.put(code, ln)
            b.put(sign, 1)
        else:
            b.put(esc[0], esc[1])
            b.put(run, 6)
            b.put(level & 0xfff, 12)
    b.put(eob[0], eob[1])


def reset_pmv():
    return [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]


def put_macroblock(self, pic, st, mb):
    b = pic['bits']
    ptype = pic['ptype']
    t = mb['type']
    key = (t['quant'], t['fwd'], t['bwd'], t['pattern'], t['intra'])
    code, ln = MBTYPE[ptype][key]
    b.put(code, ln)
    frame_pic = pic['structure'] == 3
    intra = t['intra']
    conceal = intra and pic['conceal']
    if t['fwd'] or t['bwd']:
        if frame_pic:
            if not pic['fpfd']:
                b.put(mb['motion_type'], 2)
        else:
            b.put(mb['motion_type'], 2)
    if frame_pic and not pic['fpfd'] and (intra or t['pattern']):
        b.put(mb.get('dct_type', 0), 1)
    if t['quant']:
        st['qcode'] = mb['qcode']
        b.put(mb['qcode'], 5)
    motion_type = mb.get('motion_type', 2 if frame_pic else 1)
    if not (t['fwd'] or t['bwd']):
        motion_type = 2 if frame_pic else 1
    if t['fwd'] or conceal:
        self.put_vectors(pic, st, mb, 0, motion_type)
    if t['bwd']:
        self.put_vectors(pic, st, mb, 1, motion_type)
    if conceal:
        b.put(1, 1)
    # predictor housekeeping (7.6.3.3, 7.6.3.4)
    if intra:
        if conceal:
            st['pmv'][1][0] = list(st['pmv'][0][0])
        else:
            st['pmv'] = reset_pmv()
    elif not (t['fwd'] or t['bwd']):
        st['pmv'] = reset_pmv()
    cbp = mb.get('cbp', 63 if intra else 0)
    if t['pattern']:
        c, ln = CBP[cbp]
        b.put(c, ln)
    if intra:
        cbp = 63
    tbl = DCT1 if (intra and pic['ivlc']) else DCT0
    for i in range(6):
        if not (cbp & (32 >> i)):
            continue
        blk = mb['blocks'][i]
        if intra:
            cc = 0 if i < 4 else i - 3
            target, coefs = blk
            diff = target - st['dc'][cc]
            st['dc'][cc] = target
            size = abs(diff).bit_length()
            c, ln = (DC_LUMA if cc == 0 else DC_CHROMA)[size]
            b.put(c, ln)
            if size:
                b.put(diff if diff > 0 else diff + (1 << size) - 1, size)
            encode_coefs(b, tbl, coefs, 1, False)
        else:
            encode_coefs(b, tbl, blk, 0, True)
    if not intra:
        st['dc'] = [128 << pic['dc_prec']] * 3
    st['prev'] = mb


def put_vectors(self, pic, st, mb, s, motion_type):
    b = pic['bits']
    frame_pic = pic['structure'] == 3
    if frame_pic:
        count, field_fmt, dmv = {1: (2, True, 0), 2: (1, False, 0), 3: (1, True, 1)}[motion_type]
    else:
        count, field_fmt, dmv = {1: (1, True, 0), 2: (2, True, 0), 3: (1, True, 1)}[motion_type]
    pmv = st['pmv']
    for r in range(count):
        if field_fmt and (count == 2 or not dmv):
            b.put(mb['sel'][(r, s)], 1)
        vec = mb['mv'][(r, s)]
        for t in range(2):
            fc = pic['f_code'][s][t]
            f = 1 << (fc - 1)
            high, low, rng = 16 * f - 1, -16 * f, 32 * f
            pred = pmv[r][s][t]
            halve = field_fmt and t == 1 and frame_pic
            if halve:
                pred >>= 1
            delta = vec[t] - pred
            if delta < low:
                delta += rng
            if delta > high:
                delta -= rng
            assert low <= delta <= high, (vec, pred, delta, fc)
            put_motion_component(b, delta, f)
            pmv[r][s][t] = vec[t] * 2 if halve else vec[t]
            if dmv:
                d = mb['dmv'][t]
                if d == 0:
                    b.put(0, 1)
                elif d == 1:
                    b.put(2, 2)
                else:
                    b.put(3, 2)
    if count == 1 and not dmv:
        pmv[1][s] = list(pmv[0][s])
    elif dmv:
        pmv[1][0] = list(pmv[0][0])


# ---- random macroblocks -------------------------------------------------------------------------------------------------
def random_levels(rnd, count, big=0.05, maxlevel=6):
    lv = []
    for _ in range(count):
        if rnd.random() < big:
            v = rnd.randint(1, 600)
        else:
            v = rnd.randint(1, maxlevel)
        lv.append(-v if rnd.random() < 0.5 else v)
    return lv


def budgeted(self, pic, st, mb, intra, coefs, budget):
    """Drops coefficients until the block cannot saturate the inverse DCT (the sum of the absolute dequantised values stays within
    `budget`); decoders differ on out-of-range streams, and the fixtures are not about those."""
    qcode = mb.get('qcode', st['qcode']) if mb is not None else st['qcode']
    qs = QSCALE[pic['qst']][qcode]
    scan = ALTSCAN if pic['alt'] else ZIGZAG
    W = self.matrices[0 if intra else 1]
    out = []
    total = 0
    for n, lv in coefs:
        v = abs(lv) * W[scan[n]] * qs / 16.0
        if total + v > budget:
            continue
        total += v
        out.append((n, lv))
    return out


def random_intra_block(self, pic, st, cc, level_cap, mb=None):
    rnd = self.rnd
    prec = pic['dc_prec']
    center = 128 << prec
    target = max(0, min((1 << (8 + prec)) - 1, center + rnd.randint(-(30 << prec), 30 << prec)))
    nz = min(63, int(rnd.expovariate(1 / 6)))
    pos = sorted(rnd.sample(range(1, 64), nz))
    lv = random_levels(rnd, nz, big=0.05 if pic.get('unsafe') else 0.0, maxlevel=level_cap)
    coefs = list(zip(pos, lv))
    if not pic.get('unsafe'):
        coefs = self.budgeted(pic, st, mb, True, coefs, 300)
    return (target, coefs)


def random_inter_block(self, pic, st, level_cap, mb=None):
    rnd = self.rnd
    nz = 1 + min(62, int(rnd.expovariate(1 / 4)))
    pos = sorted(rnd.sample(range(0, 64), nz))
    lv = random_levels(rnd, nz, big=0.05 if pic.get('unsafe') else 0.0, maxlevel=level_cap)
    coefs = list(zip(pos, lv))
    if not pic.get('unsafe'):
        coefs = self.budgeted(pic, st, mb, False, coefs, 300)
        if not coefs:
            coefs = [(rnd.randint(0, 63), 1)]
    return coefs


def pred_blocks(self, pic, mbx, mby, mb):
    """The prediction blocks a macroblock descriptor asks for, as (kind, x, y, w, h, vector) in luma samples of the reference view; used
    to keep every prediction inside its reference picture. kind: 'frame' or 'field'."""
    t = mb['type']
    frame_pic = pic['structure'] == 3
    out = []
    mt = mb.get('motion_type', 2 if frame_pic else 1)
    for s in range(2):
        if not (t['fwd'] if s == 0 else t['bwd']):
            continue
        if frame_pic:
            if mt == 2:
                out.append(('frame', mbx * 16, mby * 16, 16, 16, mb['mv'][(0, s)]))
            elif mt == 1:
                for r in range(2):
                    out.append(('field', mbx * 16, mby * 8, 16, 8, mb['mv'][(r, s)]))
            else:
                out.append(('field', mbx * 16, mby * 8, 16, 8, mb['mv'][(0, 0)]))
                for k in (2, 3):
                    out.append(('field', mbx * 16, mby * 8, 16, 8, mb['mv'][(k, 0)]))
        else:
            if mt == 1:
                out.append(('field', mbx * 16, mby * 16, 16, 16, mb['mv'][(0, s)]))
            elif mt == 2:
                for r in range(2):
                    out.append(('field', mbx * 16, mby * 16 + 8 * r, 16, 8, mb['mv'][(r, s)]))
            else:
                out.append(('field', mbx * 16, mby * 16, 16, 16, mb['mv'][(0, 0)]))
                out.append(('field', mbx * 16, mby * 16, 16, 16, mb['mv'][(2, 0)]))
    return out


def blocks_fit(self, pic, mbx, mby, mb):
    W, Hf = self.mbw * 16, self.mbh * 16
    for kind, x, y, bw, bh, v in pred_blocks(self, pic, mbx, mby, mb):
        rh = Hf if kind == 'frame' else Hf // 2
        if not fits(W, rh, x, y, bw, bh, v[0], v[1]):
            return False
        cx, cy = x // 2, y // 2
        cvx, cvy = trunc_div(v[0], 2), trunc_div(v[1], 2)
        if not fits(W // 2, rh // 2, cx, cy, bw // 2, bh // 2, cvx, cvy):
            return False
    return True


def random_vec(self, pic, s, t_range):
    rnd = self.rnd
    fx = pic['f_code'][s][0]
    fy = pic['f_code'][s][1]
    lim_x = min(16 << (fx - 1), t_range) - 1
    lim_y = min(16 << (fy - 1), t_range) - 1
    return (rnd.randint(-lim_x, lim_x), rnd.randint(-lim_y, lim_y))


def random_vectors(self, pic, mb, mt, fwd, bwd, concealment=False):
    rnd = self.rnd
    frame_pic = pic['structure'] == 3
    ptype = pic['ptype']
    small = 24 if mt != 3 else 6
    for s in range(2):
        if not (fwd if s == 0 else bwd):
            continue
        count = (2 if mt == 1 else 1) if frame_pic else (2 if mt == 2 else 1)
        for r in range(count):
            v = self.random_vec(pic, s, small)
            if frame_pic and mt == 1:
                # field vectors in frame pictures: the vertical component has half the range (7.6.3.2)
                lim = (16 << (pic['f_code'][s][1] - 1)) // 2 - 1
                v = (v[0], max(-lim, min(lim, v[1])))
            mb['mv'][(r, s)] = v
            mb['sel'][(r, s)] = rnd.randint(0, 1)
            if pic.get('second_after_intra') and ptype == 2:
                mb['sel'][(r, s)] = 1 - (1 if pic['structure'] == 2 else 0)
    if mt == 3 and not concealment:
        mb['dmv'] = (rnd.randint(-1, 1), rnd.randint(-1, 1))
        v0 = mb['mv'][(0, 0)]
        if frame_pic:
            tff = pic['tff']
            m_top, m_bot = (1, 3) if tff else (3, 1)
            mb['mv'][(2, 0)] = (div2_away(v0[0] * m_top) + mb['dmv'][0], div2_away(v0[1] * m_top) - 1 + mb['dmv'][1])
            mb['mv'][(3, 0)] = (div2_away(v0[0] * m_bot) + mb['dmv'][0], div2_away(v0[1] * m_bot) + 1 + mb['dmv'][1])
        else:
            e = 1 if pic['structure'] == 2 else -1
            mb['mv'][(2, 0)] = (div2_away(v0[0]) + mb['dmv'][0], div2_away(v0[1]) + e + mb['dmv'][1])


def random_mb(self, pic, st, col, row):
    """One random non-skipped macroblock descriptor for the picture at (col, row)."""
    rnd = self.rnd
    ptype = pic['ptype']
    frame_pic = pic['structure'] == 3
    dens = pic['density']
    level_cap = rnd.choice([1, 2, 3, 6, 12])
    for attempt in range(60):
        mb = {'type': {}, 'mv': {}, 'sel': {}}
        if ptype == 1:
            intra, fwd, bwd = True, False, False
        elif ptype == 2:
            intra = rnd.random() < 0.12
            fwd = not intra and rnd.random() < 0.85
            bwd = False
        else:
            intra = rnd.random() < 0.1
            fwd = bwd = False
            if not intra:
                k = rnd.random()
                fwd = k < 0.7
                bwd = k > 0.3
        quant = rnd.random() < 0.15
        if intra:
            pattern = False
        elif fwd or bwd:
            pattern = rnd.random() < dens + 0.3
        else:
            pattern = True      # a P macroblock without motion vectors carries coefficients
        if quant and not intra:
            pattern = True      # macroblock_quant comes with a pattern for non-intra macroblocks
        key = (int(quant), int(fwd), int(bwd), int(pattern), int(intra))
        if key not in MBTYPE[ptype]:
            continue
        mb['type'] = dict(quant=int(quant), fwd=int(fwd), bwd=int(bwd), pattern=int(pattern), intra=int(intra))
        if quant:
            mb['qcode'] = rnd.randint(1, 31) if pic['qst'] else rnd.randint(1, 24)
        if pic.get('restrict_intra_ref') and ptype == 2 and not intra and not fwd:
            continue
        # prediction mode
        if fwd or bwd:
            if frame_pic:
                if pic['fpfd']:
                    mt = 2
                else:
                    mt = rnd.choice([1, 2, 2] + ([3] if ptype == 2 and not bwd else []))
            else:
                mt = rnd.choice([1, 1, 2] + ([3] if ptype == 2 and not bwd else []))
            if pic.get('no_dual') and mt == 3:
                mt = 1
            mb['motion_type'] = mt
            self.random_vectors(pic, mb, mt, fwd, bwd)
        elif intra and pic['conceal']:
            mt = 2 if frame_pic else 1
            mb['motion_type'] = mt
            self.random_vectors(pic, mb, mt, True, False, concealment=True)
        elif ptype == 2:
            mb['motion_type'] = 2 if frame_pic else 1
        if frame_pic and not pic['fpfd'] and (intra or pattern):
            mb['dct_type'] = rnd.randint(0, 1)
        if intra:
            mb['blocks'] = [self.random_intra_block(pic, st, 0 if i < 4 else i - 3, level_cap, mb) for i in range(6)]
        elif pattern:
            cbp = 0
            for i in range(6):
                if rnd.random() < 0.5:
                    cbp |= 32 >> i
            if cbp == 0:
                cbp = 1 << rnd.randint(0, 5)
            mb['cbp'] = cbp
            mb['blocks'] = [self.random_inter_block(pic, st, level_cap, mb) if cbp & (32 >> i) else None for i in range(6)]
        if (fwd or bwd) and not self.blocks_fit(pic, col, row, mb):
            continue
        return mb
    # fall back to an intra macroblock
    mb = {'type': dict(quant=0, fwd=0, bwd=0, pattern=0, intra=1), 'mv': {}, 'sel': {}}
    mb['blocks'] = [self.random_intra_block(pic, st, 0 if i < 4 else i - 3, 2, mb) for i in range(6)]
    if pic['conceal']:
        mb['motion_type'] = 2 if frame_pic else 1
        mb['mv'] = {(0, 0): (0, 0)}
        mb['sel'] = {(0, 0): 0}
    return mb


def skip_fits(self, pic, st, col, row):
    """Whether a skipped macroblock at (col, row) of a B picture predicts from inside its references."""
    prev = st['prev']
    if prev is None or prev['type']['intra']:
        return False
    frame_pic = pic['structure'] == 3
    mb = {'type': dict(fwd=prev['type']['fwd'], bwd=prev['type']['bwd']), 'motion_type': 2 if frame_pic else 1, 'mv': {}}
    for s in range(2):
        mb['mv'][(0, s)] = (st['pmv'][0][s][0], st['pmv'][0][s][1])
    return self.blocks_fit(pic, col, row, mb)


def next_mb(self, pic, st, col, row):
    rnd = self.rnd
    ptype = pic['ptype']
    can_skip = ptype in (2, 3) and 0 < col < self.mbw - 1 and not pic.get('no_skip') and rnd.random() < 0.15
    if can_skip and ptype == 3 and not self.skip_fits(pic, st, col, row):
        can_skip = False
    if can_skip:
        return None
    return self.random_mb(pic, st, col, row)


for _f in (put_macroblock, put_vectors, budgeted, random_intra_block, random_inter_block, blocks_fit, random_vec, random_vectors, random_mb, skip_fits, next_mb):
    setattr(Stream, _f.__name__, _f)
