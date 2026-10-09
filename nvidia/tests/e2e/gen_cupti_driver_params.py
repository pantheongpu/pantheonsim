#!/usr/bin/env python3
"""Makes cupti_driver_params.inc, which cupti_driver_params.cu includes: a switch
from a driver callback id to a line naming the fields of its parameter structure
and their values. A development tool -- run it, review the diff, commit the
output; the build does not run it. (gen_cupti_params.py is the runtime's.)

It lists only the functions whose parameter structure is the same in every
toolkit given (the callback id's name and every field, spelled the same), so
that the one file compiles with all of them, and only structures whose fields
are plain values: a pointer is printed as set or null, a number as itself, and
anything else (a struct passed by value) by its size.

usage: gen_cupti_driver_params.py <out.inc> <generated_cuda_meta.h:cupti_driver_cbid.h> ...
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
    out = {}
    for m in re.finditer(r"CUPTI_DRIVER_TRACE_CBID_(cu\w+?)\s*=\s*(\d+)", cb):
        fn = m.group(1)
        if fn in structs:
            out[fn] = structs[fn]
    return out


def main():
    out = sys.argv[1]
    sets = [load(*a.split(":")) for a in sys.argv[2:]]
    first = sets[0]
    rows = []
    for fn in sorted(first):
        fields = first[fn]
        if not fields:
            continue
        if any(fn not in s or s[fn] != fields for s in sets[1:]):
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
            f"    case CUPTI_DRIVER_TRACE_CBID_{fn}: {{ const auto* p = static_cast<const {fn}_params*>(d); "
            f"return {body}; }}"
        )
    with open(out, "w") as f:
        f.write("// Made by gen_cupti_driver_params.py; see there.\n")
        f.write("\n".join(rows) + "\n")
    print(len(rows), "functions")


main()
