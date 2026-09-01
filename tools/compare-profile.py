#!/usr/bin/env python3
"""Compare a hardware-characterized profile against the one in the repo.

Prints only the fields that differ, so a documentation-derived placeholder can
be corrected from what the device actually reports.

    tools/compare-profile.py <measured.yaml> <profiles/nvidia/x.yaml>
"""
import re
import sys


def load(path):
    """Parse the restricted-YAML profile subset into a flat key -> value map."""
    out, section = {}, None
    for line in open(path):
        line = line.split('#')[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(':')
        val = val.strip()
        if indent == 0:
            section = key if not val else None
            if val:
                out[key] = val
        elif section:
            out[f'{section}.{key}'] = val
    return out


def main():
    measured, existing = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(measured) | set(existing))
    diffs = []
    for k in keys:
        m, e = measured.get(k), existing.get(k)
        if m is None or e is None or m == e:
            continue
        # id/model naturally differ when the repo profile is a different SKU.
        diffs.append((k, e, m))
    name = sys.argv[2].split('/')[-1]
    if not diffs:
        print(f'{name}: matches hardware on every shared field')
        return 0
    print(f'{name}: {len(diffs)} field(s) differ (repo -> hardware)')
    for k, e, m in diffs:
        print(f'  {k:<44} {e:>18}  ->  {m}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
