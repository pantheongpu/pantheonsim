#!/usr/bin/env python3
"""Makes the table that turns a runtime call's arguments into the parameter
structure CUPTI's callback API hands a subscriber (cupti_runtime_conv.inc).

The structures are the toolkit's own, in generated_cuda_runtime_api_meta.h, so
this reads them rather than keeping a copy: for every callback id of
cupti_runtime_cbid.h that has a `<function>_v<N>_params` structure whose fields
are plain values, one row that copies the call's i-th argument into the i-th
field by the field's size. A function with no structure takes no arguments.
A structure with an array or a function pointer for a field has no row, and the
call is then not delivered to callbacks (the activity records still have it).

usage: gen_cupti_runtime_conv.py <generated_cuda_runtime_api_meta.h> <cupti_runtime_cbid.h> <out.inc>
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

    for m in re.finditer(r"CUPTI_RUNTIME_TRACE_CBID_(\w+?)_v(\d+)\s*=", cbids):
        fn, ver = m.group(1), m.group(2)
        key = f"{fn}_v{ver}"
        if key in structs:
            fields = structs[key]
            if fields is None:
                continue
            copies = " ".join(
                f"std::memcpy(&p->{f}, a[{i}], sizeof p->{f});" for i, f in enumerate(fields)
            )
            rows.append(
                f'{{"{fn}", Conv{{CUPTI_RUNTIME_TRACE_CBID_{key}, '
                f"[](void* st, const void* const* a) {{ auto* p = new (st) {key}_params(); "
                f"static_assert(sizeof({key}_params) <= 512, \"parameter structure too large\"); "
                f"{copies if copies else '(void)a;'} }}, {len(fields)}}}}},"
            )
        else:
            # No structure: the function takes no arguments.
            rows.append(f'{{"{fn}", Conv{{CUPTI_RUNTIME_TRACE_CBID_{key}, nullptr, 0}}}},')

    with open(out_path, "w") as out:
        out.write("\n".join(rows) + ("\n" if rows else ""))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
