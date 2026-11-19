"""The drift detection of the application smoke checks: framework-free, no torch, no GPU, no simulator.

app_smoke.py is only as good as its comparison with the card's recorded tensors, so this checks the
comparison itself and the committed reference file:
  * every workload app_smoke.py runs has an entry in the file, the file has no other entry, and each entry's
    shape and value count agree (the file is read by app_smoke_ref.read_ref, which refuses a malformed one);
  * a result identical to the reference passes, for every entry;
  * a bit-for-bit entry fails on one changed bit (the last bit of one value), on a NaN, on a changed shape;
  * a tolerance entry passes with an error at half the tolerance and fails at twice it, fails on a NaN where
    the card has a number, and fails when values are missing;
  * a file written by write_ref reads back to the same bits.
"""
import math
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import app_smoke_ref as ref  # noqa: E402

REF = os.path.join(os.environ.get('VGPU_E2E_DATA', os.path.join(HERE, '..', 'data')), 'pytorch_app_smoke.rtx3060.ref')
failures = []


def expect(cond, what):
    if not cond:
        failures.append(what)


def flip_last_bit(v):
    (b,) = struct.unpack('<I', struct.pack('<f', v))
    return struct.unpack('<f', struct.pack('<I', b ^ 1))[0]


def main():
    header, entries = ref.read_ref(REF)
    expect(header and 'RTX 3060' in header[0] and 'torch' in header[0], 'the header names the card and torch version')
    expect(list(entries) == list(ref.TOLERANCES), 'entries %s differ from the workloads %s' % (list(entries), list(ref.TOLERANCES)))
    # app_smoke.py's workload list is checked against the tolerance table in its own module (an assert at import);
    # here the source is read so the same holds without torch.
    src = open(os.path.join(HERE, 'app_smoke.py')).read()
    for name in ref.TOLERANCES:
        expect(('def %s(d)' % name) in src, 'app_smoke.py has no workload %s' % name)
    for name, e in entries.items():
        vals, shape = e['values'], e['shape']
        tol = ref.TOLERANCES[name][0]
        expect(e['tol'] == tol, '%s: the file says tol %r, the table %r' % (name, e['tol'], tol))
        expect(all(math.isfinite(v) for v in vals), '%s: the reference holds a NaN or an infinity' % name)
        ok, msg = ref.compare(e, shape, list(vals))
        expect(ok, '%s: an identical result fails: %s' % (name, msg))
        ok, _ = ref.compare(e, list(reversed(shape)) if len(set(shape)) > 1 else shape + [1], list(vals))
        expect(not ok, '%s: a changed shape passes' % name)
        ok, _ = ref.compare(e, shape, list(vals[:-1]))
        expect(not ok, '%s: a missing value passes' % name)
        bad_nan = list(vals)
        bad_nan[0] = float('nan')
        ok, _ = ref.compare(e, shape, bad_nan)
        expect(not ok, '%s: a NaN passes' % name)
        if tol == 0.0:
            one = list(vals)
            one[-1] = flip_last_bit(one[-1])
            ok, _ = ref.compare(e, shape, one)
            expect(not ok, '%s: bit for bit, yet one changed bit passes' % name)
        else:
            scale = max(abs(v) for v in vals)
            for factor, want in ((0.5, True), (2.0, False)):
                moved = list(vals)
                i = max(range(len(vals)), key=lambda k: abs(vals[k]))
                moved[i] = vals[i] + factor * tol * scale
                ok, msg = ref.compare(e, shape, moved)
                expect(ok == want, '%s: an error of %gx the tolerance gives %s (%s), wanted %s' % (name, factor, ok, msg, want))
    # Round trip through write_ref.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'x.ref')
        ref.write_ref(p, ['a header'], {n: (e['shape'], e['values']) for n, e in entries.items()})
        h2, e2 = ref.read_ref(p)
        expect(h2 == ['a header'] and all(e2[n]['hex'] == entries[n]['hex'] for n in entries), 'write_ref and read_ref do not round trip')
        with open(p, 'a') as f:
            f.write('{"name": "gemm_f32", "tol": 0, "shape": [1], "hex": "00000000"}\n')
        try:
            ref.read_ref(p)
            expect(False, 'a duplicate entry is accepted')
        except ValueError:
            pass
    if failures:
        for f in failures:
            print('FAIL ' + f)
        return 1
    print('ok   app smoke reference: %d entries, drift detected as documented' % len(entries))
    return 0


if __name__ == '__main__':
    sys.exit(main())
