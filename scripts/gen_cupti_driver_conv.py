#!/usr/bin/env python3
"""Makes the table that turns a driver call's arguments into the parameter
structure CUPTI's callback API hands a subscriber (cupti_driver_conv.inc).

The structures are the toolkit's own, in generated_cuda_meta.h, so this reads
them rather than keeping a copy: for every callback id of cupti_driver_cbid.h
that has a `<function>_params` structure whose fields are plain values, one row
that copies the call's i-th argument into the i-th field by the field's size. A
structure with an array, a bit-field or a function-pointer declarator for a field
has no row (the call is then not delivered to callbacks; the DRIVER activity
records still have it). A function the header gives no structure takes no
arguments (cuCtxSynchronize, cuProfilerStart) and is added by hand, because a
missing structure is also what the header does for the calls it no longer
describes.

Each row says how many arguments the call has and the size of each, and the
call is delivered only when the driver shim passed as many arguments of the
sizes the toolkit's fields have: a mismatch means the shim declares the function
differently from NVIDIA's, and reading its arguments as the toolkit's fields
would hand a profiler bytes the program never passed.

The rows made by hand in cupti_api.cpp, which are checked against an RTX 3060,
win over these.

usage: gen_cupti_driver_conv.py <generated_cuda_meta.h> <cupti_driver_cbid.h> <out.inc>
"""
import re
import sys


def main(meta_path, cbid_path, out_path):
    rows = []
    try:
        meta = open(meta_path, encoding="utf-8", errors="replace").read()
        cbids = open(cbid_path, encoding="utf-8", errors="replace").read()
    except OSError:
        open(out_path, "w").write("")
        return

    structs = {}
    for m in re.finditer(r"typedef struct (\w+)_params_st \{(.*?)\} \w+_params;", meta, re.S):
        name, body = m.group(1), m.group(2)
        fields = []
        ok = True
        for decl in body.split(";"):
            decl = decl.strip()
            if not decl:
                continue
            if "(" in decl or "[" in decl or ":" in decl:
                ok = False
                break
            f = re.search(r"([A-Za-z_]\w*)\s*$", decl)
            if not f:
                ok = False
                break
            fields.append(f.group(1))
        structs[name] = fields if ok else None

    for m in re.finditer(r"CUPTI_DRIVER_TRACE_CBID_(cu\w+?)\s*=", cbids):
        fn = m.group(1)
        fields = structs.get(fn)
        if not fields:
            continue
        copies = " ".join(
            f"std::memcpy(&p->{f}, a[{i}], sizeof p->{f});" for i, f in enumerate(fields)
        )
        checks = " && ".join(
            f"z[{i}] == sizeof(std::declval<{fn}_params&>().{f})" for i, f in enumerate(fields)
        )
        rows.append(
            f'{{"{fn}", Conv{{CUPTI_DRIVER_TRACE_CBID_{fn}, '
            f"[](void* st, const void* const* a) {{ auto* p = new (st) {fn}_params(); "
            f"static_assert(sizeof({fn}_params) <= 512, \"parameter structure too large\"); "
            f"{copies} }}, {len(fields)}, "
            f"[](const uint16_t* z) {{ return {checks}; }}}}}},"
        )

    with open(out_path, "w") as out:
        out.write("\n".join(rows) + ("\n" if rows else ""))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
