#!/usr/bin/env python3
"""Compares two runs of the numerics programs (tools/numerics): the simulator's two engines, two profiles, or the simulator
and a card.

    compare.py <a> <b> [--all] [--truth] [--published tools/numerics/published.tsv]

<a> and <b> are line files (mathfn's, fp_examples', varity_cases', a Varity run's) or directories holding files of the same
names (two output directories of card-run.sh or run-sim.sh; pairs are matched by file name).

For mathfn lines it prints, per function, precision and class whose hashes differ: how many results differ and by how many
ulp (units in the last place between the two results: the distance between their bit patterns on the number line), with NaN
payloads, the sign of zero and NaN-against-number counted apart. If a dump directory stands beside the line file
(<name>.txt -> <name>.dump, written by mathfn --dump) every result is counted; without one, the results the line prints
(all of the worst class, the first four of the others). --all also lists the lines that agree.

--truth (needs mpmath) adds, for the worst class, each run's error against the true value beside the published figure of
published.tsv, and the result that figure implies ("implies: a" if run a has the published error, else the neighbour of a's
result that has it): a run that reproduces the published library on the published hardware shows the published error.
Other files are compared line by line by their first word.

Exit status: 0 if nothing differs, 1 otherwise.
"""
import array
import os
import re
import struct
import sys

LINE = re.compile(r'^(\w+) (f32|f64) (\w+) n=(\d+) hash=([0-9a-f]{16}) \|(.*)$')


def parse(path):
    """-> (mathfn lines {(fn, prec, class): (n, hash, [(x, y, r, aux)])}, other lines {key: text})"""
    math, other = {}, {}
    with open(path, errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            m = LINE.match(line)
            if not m:
                key = line.split(' ', 1)[0]
                while key in other:
                    key += "'"
                other[key] = line
                continue
            pairs = []
            for item in m.group(6).split():
                lhs, rhs = item.split('=')
                x, _, y = lhs.partition(',')
                r, _, aux = rhs.partition('/')
                pairs.append((int(x, 16), int(y, 16) if y else None, int(r, 16), int(aux) if aux else None))
            math[(m.group(1), m.group(2), m.group(3))] = (int(m.group(4)), m.group(5), pairs)
    return math, other


def order(bits, width):
    """A bit pattern's place on the number line (-0 and +0 share one)."""
    sign = bits >> (width - 1)
    mag = bits & ((1 << (width - 1)) - 1)
    return -mag if sign else mag


def unorder(value, width):
    """The bit pattern at a place on the number line (order's inverse; zero is +0)."""
    return value if value >= 0 else (1 << (width - 1)) | -value


def is_nan(bits, width):
    exp_bits, mant_bits = (8, 23) if width == 32 else (11, 52)
    return (bits >> mant_bits) & ((1 << exp_bits) - 1) == (1 << exp_bits) - 1 and bits & ((1 << mant_bits) - 1) != 0


class Tally:
    def __init__(self):
        self.n = self.differ = self.nan_payload = self.zero_sign = self.nan_number = self.max_ulp = self.sum_ulp = 0
        self.one = self.two = self.more = 0
        self.example = None

    def add(self, x, y, a, b, width):
        self.n += 1
        if a == b:
            return
        self.differ += 1
        na, nb = is_nan(a, width), is_nan(b, width)
        if na and nb:
            self.nan_payload += 1
            kind = 'NaN payload'
        elif na or nb:
            self.nan_number += 1
            kind = 'NaN against a number'
        else:
            d = abs(order(a, width) - order(b, width))
            if d == 0:
                self.zero_sign += 1
                kind = 'sign of zero'
            else:
                self.sum_ulp += d
                self.one += d == 1
                self.two += d == 2
                self.more += d > 2
                kind = None
                if d > self.max_ulp:
                    self.max_ulp = d
                    self.example = (x, y, a, b)
        if self.example is None and kind:
            self.example = (x, y, a, b)

    def text(self, width):
        h = '%%0%dx' % (width // 4)
        s = '%d of %d differ' % (self.differ, self.n)
        if self.one or self.two or self.more:
            s += ': 1 ulp %d, 2 ulp %d, more %d (max %d ulp)' % (self.one, self.two, self.more, self.max_ulp)
        for count, name in ((self.nan_payload, 'NaN payload'), (self.zero_sign, 'sign of zero'), (self.nan_number, 'NaN against a number')):
            if count:
                s += '; %s %d' % (name, count)
        if self.example:
            x, y, a, b = self.example
            s += '; e.g. x=' + h % x + ((',' + h % y) if y is not None else '') + ': ' + h % a + ' / ' + h % b
        return s


def read_words(path, width):
    arr = array.array('I' if width == 32 else 'Q')
    with open(path, 'rb') as f:
        arr.frombytes(f.read())
    if sys.byteorder != 'little':
        arr.byteswap()
    return arr


def dump_of(path):
    d = re.sub(r'\.txt$', '', path) + '.dump'
    return d if os.path.isdir(d) else None


def dump_tally(da, db, key, arity2):
    fn, prec, cls = key
    width = 32 if prec == 'f32' else 64
    name = '%s.%s.%s' % (fn, prec, cls)
    ra, rb = os.path.join(da, name + '.r'), os.path.join(db, name + '.r')
    if not (os.path.exists(ra) and os.path.exists(rb)):
        return None
    a, b = read_words(ra, width), read_words(rb, width)
    if len(a) != len(b):
        return None
    if cls == 'worst':
        xs = read_words(os.path.join(da, name + '.x'), width)
        ys = read_words(os.path.join(da, name + '.y'), width) if arity2 else None
    else:
        base = os.path.join(da, 'in.%s.%s' % (prec, cls))
        xs = read_words(base + ('.x2' if arity2 else '.x1'), width)
        ys = read_words(base + '.y2', width) if arity2 else None
    t = Tally()
    for i in range(len(a)):
        if a[i] == b[i]:
            t.n += 1
        else:
            t.add(xs[i], ys[i] if ys is not None else None, a[i], b[i], width)
    return t


def to_float(bits, width):
    return struct.unpack('<f' if width == 32 else '<d', struct.pack('<I' if width == 32 else '<Q', bits))[0]


def truth_error(fn, x, y, r, width):
    """The result's distance from the true value, in ulp of the true value; None where there is no finite true value."""
    import mpmath
    mpmath.mp.prec = 400
    mp = mpmath.mpf
    fx = to_float(x, width)
    fr = to_float(r, width)
    if fx != fx or fr != fr or fx in (float('inf'), float('-inf')) or fr in (float('inf'), float('-inf')):
        return None
    X = mp(fx)
    Y = mp(to_float(y, width)) if y is not None else None
    table = {
        'sin': mpmath.sin, 'cos': mpmath.cos, 'tan': mpmath.tan, 'asin': mpmath.asin, 'acos': mpmath.acos, 'atan': mpmath.atan,
        'sinh': mpmath.sinh, 'cosh': mpmath.cosh, 'tanh': mpmath.tanh, 'asinh': mpmath.asinh, 'acosh': mpmath.acosh,
        'atanh': mpmath.atanh, 'exp': mpmath.exp, 'exp2': lambda v: mp(2) ** v, 'exp10': lambda v: mp(10) ** v,
        'expm1': mpmath.expm1, 'log': mpmath.log, 'log2': lambda v: mpmath.log(v, 2), 'log10': mpmath.log10,
        'log1p': mpmath.log1p, 'sqrt': mpmath.sqrt, 'rsqrt': lambda v: 1 / mpmath.sqrt(v),
        'cbrt': lambda v: mpmath.sign(v) * mpmath.cbrt(abs(v)), 'erf': mpmath.erf,
        'erfc': mpmath.erfc, 'lgamma': lambda v: mpmath.log(abs(mpmath.gamma(v))), 'tgamma': mpmath.gamma,
        'sinpi': mpmath.sinpi, 'cospi': mpmath.cospi, 'rcp': lambda v: 1 / v, 'rsqrt_div': lambda v: 1 / mpmath.sqrt(v),
    }
    table2 = {'pow': lambda a, b: a ** b, 'atan2': mpmath.atan2, 'hypot': mpmath.hypot, 'fmod': mpmath.fmod, 'div': lambda a, b: a / b}
    try:
        if fn in table and Y is None:
            true = table[fn](X)
        elif fn in table2 and Y is not None:
            true = table2[fn](X, Y)
        else:
            return None
        if not mpmath.isfinite(true) or true == 0 or mpmath.im(true) != 0:
            return None
        true = mpmath.re(true)
    except Exception:
        return None
    mant, emin = (24, -126) if width == 32 else (53, -1022)
    e = max(int(mpmath.floor(mpmath.log(abs(true), 2))), emin)
    return float(abs(mp(fr) - true) / mp(2) ** (e - mant + 1))


def load_published(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            c = line.rstrip('\n').split('\t')
            rows.append({'fn': c[0], 'prec': c[1], 'x': c[2], 'y': c[3], 'ulp': c[4], 'lib': c[5], 'src': c[6],
                         'bits': c[7] if len(c) > 7 else None})
    return rows


def bits_of(text, prec):
    v = float.fromhex(text)
    return struct.unpack('<I', struct.pack('<f', v))[0] if prec == 'f32' else struct.unpack('<Q', struct.pack('<d', v))[0]


def compare_files(pa, pb, opts):
    ma, oa = parse(pa)
    mb, ob = parse(pb)
    differ = 0
    da, db = dump_of(pa), dump_of(pb)
    same = 0
    for key in ma:
        if key not in mb:
            print('  only in %s: %s %s %s' % (pa, *key))
            differ += 1
            continue
        (n, ha, pairs_a), (nb, hb, pairs_b) = ma[key], mb[key]
        if ha == hb and n == nb:
            same += 1
            if opts['all']:
                print('  %-10s %s %-9s same (%d results)' % (key[0], key[1], key[2], n))
            continue
        differ += 1
        if n != nb:
            print('  %-10s %s %-9s the runs hold %d and %d results: run both with the same --n' % (*key, n, nb))
            continue
        width = 32 if key[1] == 'f32' else 64
        t = None
        if da and db:
            t = dump_tally(da, db, key, bool(pairs_a and pairs_a[0][1] is not None))
        scope = 'every result'
        if t is None:
            t = Tally()
            for (x, y, a, aux_a), (_, _, b, aux_b) in zip(pairs_a, pairs_b):
                t.add(x, y, a, b, width)
            scope = 'the %d results the line prints' % t.n if key[2] != 'worst' else 'every result'
            if t.differ == 0:
                print('  %-10s %s %-9s hashes differ; the %d printed results agree (run mathfn --dump for the count)' % (*key, t.n))
                continue
        print('  %-10s %s %-9s %s [%s]' % (key[0], key[1], key[2], t.text(width), scope))
    for key in mb:
        if key not in ma:
            print('  only in %s: %s %s %s' % (pb, *key))
            differ += 1
    for key in oa:
        if key not in ob:
            print('  only in %s: %s' % (pa, oa[key]))
            differ += 1
        elif oa[key] != ob[key]:
            print('  < %s\n  > %s' % (oa[key], ob[key]))
            differ += 1
        else:
            same += 1
    for key in ob:
        if key not in oa:
            print('  only in %s: %s' % (pb, ob[key]))
            differ += 1
    print('  %d lines agree, %d differ' % (same, differ))
    if opts['published'] and (ma or mb):
        published(ma, mb, load_published(opts['published']), opts['truth'])
    return differ


def published(ma, mb, rows, truth):
    """The worst class against the published figures: result bits where the source prints them, the error in ulp with --truth."""
    printed = False
    for row in rows:
        key = (row['fn'], row['prec'], 'worst')
        if key not in ma and key not in mb:
            continue
        width = 32 if row['prec'] == 'f32' else 64
        h = '%%0%dx' % (width // 4)
        x = bits_of(row['x'], row['prec'])
        y = bits_of(row['y'], row['prec']) if row['y'] != '-' else None
        got = []
        for m in (ma, mb):
            r = None
            for (px, py, pr, _) in m.get(key, (0, '', []))[2]:
                if px == x and py == y:
                    r = pr
            got.append(r)
        if got[0] is None and got[1] is None:
            continue
        cells = []
        for r in got:
            if r is None:
                cells.append('-')
                continue
            cell = h % r
            if truth:
                err = truth_error(row['fn'], x, y, r, width)
                cell += ' (%s ulp)' % ('%.3f' % err if err is not None else '-')
            cells.append(cell)
        if not printed:
            print('  published inputs: function precision x[,y] | a | b | published (library, source)')
            printed = True
        pub = (row['bits'] + ' ' if row['bits'] else '') + (row['ulp'] + ' ulp ' if row['ulp'] != '-' else '')
        if truth and row['ulp'] != '-' and got[0] is not None:
            pub += implied(row, x, y, got[0], width)
        mark = ''
        if row['bits']:
            mark = ' ' + '/'.join('=' if r is not None and h % r == row['bits'] else 'differs' for r in got)
        print('    %-9s %s %s%s | %s | %s | %s(%s, %s)%s' % (row['fn'], row['prec'], row['x'], (',' + row['y']) if y is not None else '',
                                                           cells[0], cells[1], pub, row['lib'], row['src'], mark))


def implied(row, x, y, r, width):
    """The result the published error implies, searched within four ulp of run a's: the one whose error is the published figure."""
    try:
        want = float(row['ulp'])
    except ValueError:
        return ''
    if want > 100:
        return ''
    import math
    digit = 10.0 ** (math.floor(math.log10(want)) - 2)   # the figures are printed to three digits, some rounded up
    hits = []
    for k in range(-4, 5):
        cand = unorder(order(r, width) + k, width)
        err = truth_error(row['fn'], x, y, cand, width)
        if err is not None and want - 1.5 * digit <= err <= want + 0.6 * digit:
            hits.append((k, cand))
    if not hits:
        return '[implies: nothing within 4 ulp of a] '
    if any(k == 0 for k, _ in hits):
        return '[implies: a] '
    h = '%%0%dx' % (width // 4)
    return '[implies: %s] ' % ' or '.join(h % cand + ' (a %+d ulp)' % k for k, cand in hits)


def main(argv):
    opts = {'all': False, 'truth': False, 'published': None}
    paths = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--all':
            opts['all'] = True
        elif a == '--truth':
            opts['truth'] = True
        elif a == '--published' and i + 1 < len(argv):
            i += 1
            opts['published'] = argv[i]
        elif a.startswith('-'):
            print(__doc__)
            return 2
        else:
            paths.append(a)
        i += 1
    if len(paths) != 2:
        print(__doc__)
        return 2
    if opts['truth'] and not opts['published']:
        opts['published'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'published.tsv')
    a, b = paths
    differ = 0
    if os.path.isdir(a) and os.path.isdir(b):
        names = sorted(n for n in os.listdir(a) if n.endswith('.txt') and os.path.isfile(os.path.join(b, n)))
        missing = sorted(set(n for n in os.listdir(a) + os.listdir(b) if n.endswith('.txt')) - set(names))
        for n in names:
            print('%s:' % n)
            differ += compare_files(os.path.join(a, n), os.path.join(b, n), opts)
        for n in missing:
            print('%s: in one directory only' % n)
    else:
        differ = compare_files(a, b, opts)
    return 1 if differ else 0


if __name__ == '__main__':
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # compare.py ... | head
    except (ImportError, AttributeError, ValueError):
        pass
    sys.exit(main(sys.argv[1:]))
