#!/usr/bin/env python3
"""Compare a hardware-characterized profile against the one in the repo.

Prints only the fields that differ, so a documentation-derived placeholder can
be corrected from what the device actually reports.

    nvidia/tools/compare-profile.py <measured.yaml> <nvidia/profiles/x.yaml>
"""
import re
import sys


# Fields whose value is a property of the *configuration* of a device rather
# than of its model, so a difference between two physically identical cards is
# expected rather than a correction to make.
#
# vram_bytes is the one that bites: an A10 with ECC enabled reports about
# 1.5 GiB less than the same A10 with it off, and two H100s on different driver
# versions differed by 10 MiB of reserved memory. Both readings are right.
CONFIG_DEPENDENT = {"vram_bytes"}


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
    config = [d for d in diffs if d[0] in CONFIG_DEPENDENT]
    real = [d for d in diffs if d[0] not in CONFIG_DEPENDENT]
    if not real:
        print(f'{name}: matches hardware on every shared field')
    else:
        print(f'{name}: {len(real)} field(s) differ (repo -> hardware)')
        for k, e, m in real:
            print(f'  {k:<44} {e:>18}  ->  {m}')
    for k, e, m in config:
        # Not a correction to make: see CONFIG_DEPENDENT above.
        print(f'  [configuration] {k:<30} {e:>18}  vs  {m}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
