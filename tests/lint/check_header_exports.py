#!/usr/bin/env python3
"""Every function the CUDA toolkit's headers declare is a symbol the shims export.

A program binds the functions it calls by name when it is loaded (PyTorch is linked
with -z now, so a missing name fails `import torch`) or the first time it calls
one, and a shim that lacks a function the toolkit declares fails that program with
an unresolved symbol: nothing says which behaviour was wanted, and no status code
says "not implemented". A function a shim cannot honour is still exported and
answers the library's own "not supported" status, so the name resolves and the
caller gets an answer it can act on (see the "unimplemented entry point" rule in
nvidia/docs/libraries.md).

This preprocesses the toolkit's cuda_runtime_api.h and cuda.h (twice: as a program
sees them by default, and with CUDA_API_PER_THREAD_DEFAULT_STREAM, which renames
the copy and launch functions to _ptsz / _ptds), takes the functions they declare
under the names the preprocessor leaves them with (cuMemAlloc_v2, cudaMemcpy_ptds)
and checks each against the dynamic symbols of the built runtime and driver shims.

usage: check_header_exports.py [--list] [<cuda include dir> [<shim dir>]]
  --list   print every missing name and exit 0 (how the gaps were found)
SKIPs where there is no toolkit header or no shim at hand.
"""
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Declared by the headers and deliberately not exported, each with why. Empty on purpose: an
# unsupported function is exported and refuses (a status code), it is not left unresolved.
EXEMPT = {}

# Names the headers declare that no CUDA library exports: private macros' targets and the like.
NOT_FUNCTIONS = set()


def find_include(arg):
    candidates = [arg] if arg else []
    candidates += [os.environ.get("CUDA_ABI_INCLUDE", ""), "/usr/local/cuda/include", "/usr/include"]
    for c in candidates:
        if c and (pathlib.Path(c) / "cuda.h").exists() and (pathlib.Path(c) / "cuda_runtime_api.h").exists():
            return pathlib.Path(c)
    return None


def preprocess(include, header, defines, cplusplus):
    cmd = ["g++" if cplusplus else "gcc", "-E", "-P", "-x", "c++" if cplusplus else "c", f"-I{include}",
           *defines, str(include / header)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"cannot preprocess {header}: {out.stderr.strip().splitlines()[-1:]}")
    return out.stdout


def declared(text, prefix):
    """The names of the functions declared at file scope: `extern ... name (...) ;`."""
    names = set()
    text = re.sub(r"__attribute__\s*\(\(.*?\)\)", " ", text, flags=re.S)
    text = re.sub(r"__declspec\s*\([^)]*\)", " ", text)
    for stmt in re.split(r"[;{}]", text):
        s = " ".join(stmt.split())
        if "(" not in s or s.startswith('extern "C"'):
            continue
        # The runtime's declarations start with `extern`; the driver's are `CUresult name (...)`.
        if not (s.startswith("extern ") or s.startswith("CUresult ")):
            continue
        head = s.split("(")[0]
        m = re.search(r"\b(" + prefix + r"[A-Za-z0-9_]*)\s*$", head)
        if m:
            names.add(m.group(1))
    return names


def exported(shim, pattern):
    syms = set()
    for lib in sorted({p.resolve() for p in shim.glob(pattern)}):
        out = subprocess.run(["nm", "-D", "--defined-only", str(lib)], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[1] in "TWtwiI":
                syms.add(parts[2].split("@")[0])
    return syms


def main():
    args = [a for a in sys.argv[1:] if a != "--list"]
    list_only = "--list" in sys.argv[1:]
    include = find_include(args[0] if args else "")
    if include is None:
        print("SKIP: no CUDA headers to compare the shims' exports with")
        return 0
    shim = pathlib.Path(args[1]) if len(args) > 1 else pathlib.Path(os.environ.get("VGPU_BUILD_DIR", ROOT / "build")) / "shim"
    runtime_syms = exported(shim, "libcudart.so.*")
    driver_syms = exported(shim, "libcuda.so.*")
    if not runtime_syms or not driver_syms:
        print("SKIP: the runtime and driver shims are not built")
        return 0

    missing = {}
    for header, prefix, syms, cplusplus in (("cuda_runtime_api.h", "cuda", runtime_syms, True),
                                            ("cuda.h", "cu", driver_syms, False)):
        names = set()
        for defines in ([], ["-DCUDA_API_PER_THREAD_DEFAULT_STREAM"]):
            names |= declared(preprocess(include, header, defines, cplusplus), prefix)
        names -= NOT_FUNCTIONS
        # cuda.h's own helpers and the CUDA-defined handle-conversion macros are not entry points.
        miss = sorted(n for n in names if n not in syms and n not in EXEMPT)
        missing[header] = (len(names), miss)

    bad = 0
    for header, (total, miss) in missing.items():
        print(f"{header}: {total} functions declared, {len(miss)} not exported")
        for n in miss:
            print(f"    {n}")
        bad += len(miss)
    if list_only:
        return 0
    if bad:
        print("FAIL: a function the toolkit declares is not exported (export it, refusing with the library's own\n"
              "      'not supported' status if the shim cannot do what it asks)")
        return 1
    print("PASS: every function cuda_runtime_api.h and cuda.h declare is exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
