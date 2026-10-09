#!/usr/bin/env python3
"""Makes cupti_params.inc, which cupti_params.cu includes: a switch from a runtime
callback id to a line naming the fields of its parameter structure and their
values. A development tool -- run it, review the diff, commit the output; the
build does not run it.

It lists only the functions whose parameter structure is the same in the CUDA
12.0, 12.8 and 13.x toolkits (the callback id of the newest version and every
field, spelled the same), so that the one file compiles with all of them, and
only structures whose fields are plain values: a pointer is printed as set or
null, a number as itself, and anything else (a struct passed by value) by name.

usage: gen_cupti_params.py <out.inc> <meta.h:cbid.h> <meta.h:cbid.h> ...
(the first pair is the toolkit the output is made from)
"""
import re
import sys


def load(meta_path, cbid_path):
    meta = open(meta_path, encoding="utf-8", errors="replace").read()
    cb = open(cbid_path, encoding="utf-8", errors="replace").read()
    structs = {}
    for m in re.finditer(r"typedef struct (\w+)_params_st \{(.*?)\} \w+_params;", meta, re.S):
        fields = [re.sub(r"\s+", " ", d.strip()) for d in m.group(2).split(";") if d.strip()]
        structs[m.group(1)] = fields
    latest = {}
    for m in re.finditer(r"CUPTI_RUNTIME_TRACE_CBID_(\w+?)_v(\d+)\s*=\s*(\d+)", cb):
        fn, ver, ident = m.group(1), int(m.group(2)), int(m.group(3))
        if "_ptsz" in fn or "_ptds" in fn:
            continue
        if fn not in latest or ident > latest[fn][1]:
            latest[fn] = (ver, ident)
    return {fn: (f"{fn}_v{v[0]}", structs.get(f"{fn}_v{v[0]}")) for fn, v in latest.items()}


def main():
    out = sys.argv[1]
    sets = [load(*a.split(":")) for a in sys.argv[2:]]
    first = sets[0]
    rows = []
    for fn in sorted(first):
        name, fields = first[fn]
        if fields is None or not fields:
            continue
        if any(fn not in s or s[fn] != first[fn] for s in sets[1:]):
            continue
        names = []
        ok = True
        for decl in fields:
            if "(" in decl or "[" in decl or ":" in decl:
                ok = False
                break
            m = re.search(r"([A-Za-z_]\w*)\s*$", decl)
            if not m:
                ok = False
                break
            names.append(m.group(1))
        if not ok:
            continue
        body = " + ".join(f'field("{n}", p->{n})' for n in names)
        rows.append(
            f"    case CUPTI_RUNTIME_TRACE_CBID_{name}: {{ const auto* p = static_cast<const {name}_params*>(d); "
            f"return {body}; }}"
        )
    with open(out, "w") as f:
        f.write("// Made by gen_cupti_params.py; see there.\n")
        f.write("\n".join(rows) + "\n")
    print(len(rows), "functions")


main()
