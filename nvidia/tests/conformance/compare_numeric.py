#!/usr/bin/env python3
"""Compare two conformance outputs, allowing float rounding differences.

Reductions (dot products, norms, GEMV accumulation) do not associate the same
way on a GPU as on a CPU, so demanding identical text would fail for results
that are correct. Integers and text must still match exactly; only floating
point values are compared with a relative tolerance.

    compare_numeric.py <a> <b> [rel-tolerance]
"""
import re
import sys

FLOAT = re.compile(r'-?\d+\.\d+(?:[eE][-+]?\d+)?')


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    a_lines = open(a_path).read().splitlines()
    b_lines = open(b_path).read().splitlines()
    # Ignore loader chatter that is not part of the result.
    strip = lambda ls: [l for l in ls if 'no version information' not in l]
    a_lines, b_lines = strip(a_lines), strip(b_lines)
    if len(a_lines) != len(b_lines):
        print(f'line count differs: {len(a_lines)} vs {len(b_lines)}', file=sys.stderr)
        return 1
    for i, (la, lb) in enumerate(zip(a_lines, b_lines), 1):
        # The lines must be identical once their float values are removed.
        if FLOAT.sub('<f>', la) != FLOAT.sub('<f>', lb):
            print(f'line {i} structure differs:\n  {la}\n  {lb}', file=sys.stderr)
            return 1
        for x, y in zip(FLOAT.findall(la), FLOAT.findall(lb)):
            fx, fy = float(x), float(y)
            scale = max(abs(fx), abs(fy), 1e-30)
            if abs(fx - fy) / scale > tol:
                print(f'line {i}: {fx} vs {fy} (relative {abs(fx-fy)/scale:.2e} > {tol})',
                      file=sys.stderr)
                return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
