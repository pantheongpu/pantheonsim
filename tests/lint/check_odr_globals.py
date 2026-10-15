#!/usr/bin/env python3
"""No data symbol is defined, with external linkage, in code that libcuda and libcudart both contain.

libcuda and libcudart each link libvgpu.a, and a few source files (fatbin.cpp,
fatbin_sass.cpp) are compiled into both. A test that loads both libraries into
one process under AddressSanitizer then sees every strong data definition twice,
and ASan's ODR check aborts it ("odr-violation"). That failed CI three times, in
a place nobody looked until the sanitizer job ran, hours after the push:
regs::kSysfsFiles, the hwmon file table, and simd's g_kernels / g_enabled.

The fix each time was to give the object internal linkage (`static`, an
anonymous namespace) or hidden visibility. This finds the next one on the
first, plain build:

  1. the data symbols both libraries export with default visibility
     (`nm -D --defined-only`, types B D R G S);
  2. for each, the definitions in the object files that went into the two
     libraries (CMakeFiles/vgpucuda.dir, CMakeFiles/vgpucudart.dir, libvgpu.a).
     A weak definition (`inline` variable, template static member, function
     static of an inline function) is a COMDAT the linker merges and ASan does
     not check. A strong one (type B D R C) is the violation.

A symbol that is deliberately shared goes in tests/lint/odr_globals.allow, one
regular expression per line with the reason after a `#`; a symbol traced to
no object that mentions our own namespace (vgpu) also fails, because it
cannot be classified.

usage: check_odr_globals.py [--build DIR] [--allow FILE]
       check_odr_globals.py --lib A.so --lib B.so [--obj X.o|X.a ...]   (any libraries; the self-test)
Prints SKIP when the libraries are not built.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = set("BDRGS")          # strong data in the dynamic symbol table
STRONG_OBJ = set("BDRCGS")   # a strong definition in an object file
SKIP_PREFIX = ("__", "_edata", "_end", "_fini", "_init", "_ITM_", "_Jv_")


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return r.stdout


def dyn_data(lib):
    """{symbol: type} for the strong data symbols a shared library defines."""
    out = {}
    for line in run(["nm", "-D", "--defined-only", lib]).splitlines():
        p = line.split()
        if len(p) != 3 or p[1] not in DATA:
            continue
        name = p[2].split("@")[0]
        if name.startswith(SKIP_PREFIX):
            continue
        out[name] = p[1]
    return out


def object_defs(paths):
    """{symbol: set of types} over every object file and archive member in `paths`."""
    defs = {}
    for path in paths:
        for line in run(["nm", "--defined-only", path]).splitlines():
            p = line.split()
            if len(p) != 3 or not re.fullmatch(r"[A-Za-z]", p[1]):
                continue
            defs.setdefault(p[2], set()).add(p[1])
    return defs


def allow_list(path):
    rules = []
    if path and os.path.exists(path):
        for n, line in enumerate(open(path), 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            if "#" not in line:
                sys.exit("%s:%d: an allow-list entry needs its reason after a #" % (path, n))
            rules.append(re.compile(text))
    return rules


def find_libs(build):
    def real(pattern):
        hits = sorted(glob.glob(os.path.join(build, "shim", pattern)))
        return os.path.realpath(hits[0]) if hits else None
    return real("libcuda.so.1"), real("libcudart.so.[0-9]*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.environ.get("VGPU_BUILD_DIR", os.path.join(ROOT, "build")))
    ap.add_argument("--allow", default=os.path.join(ROOT, "tests", "lint", "odr_globals.allow"))
    ap.add_argument("--lib", action="append", default=[])
    ap.add_argument("--obj", action="append", default=[])
    a = ap.parse_args()

    if a.lib:
        libs, objs = a.lib, a.obj
    else:
        cuda, cudart = find_libs(a.build)
        if not cuda or not cudart:
            print("SKIP: libcuda and libcudart are not both built in %s" % a.build)
            return 0
        libs = [cuda, cudart]
        objs = glob.glob(os.path.join(a.build, "CMakeFiles", "vgpucuda.dir", "**", "*.o"), recursive=True)
        objs += glob.glob(os.path.join(a.build, "CMakeFiles", "vgpucudart.dir", "**", "*.o"), recursive=True)
        archive = os.path.join(a.build, "libvgpu.a")
        if os.path.exists(archive):
            objs.append(archive)
    if len(libs) < 2:
        sys.exit("need two libraries")

    shared = None
    for lib in libs:
        syms = set(dyn_data(lib))
        shared = syms if shared is None else shared & syms
    defs = object_defs(objs)
    allow = allow_list(a.allow)

    bad, unknown, allowed, comdat = [], [], [], 0
    for s in sorted(shared):
        types = defs.get(s)
        if types is None:
            # Nothing in the objects: only our own namespace is worth failing for.
            if "vgpu" in s and not any(r.search(s) for r in allow):
                unknown.append(s)
            continue
        if not (types & STRONG_OBJ):
            comdat += 1                    # inline / template / COMDAT: merged by the linker
            continue
        if any(r.search(s) for r in allow):
            allowed.append(s)
            continue
        bad.append(s)

    for s in bad + unknown:
        why = "defined with external linkage in code both libraries contain" if s in bad else \
              "exported by both libraries and found in no object file, so it cannot be classified"
        print("FAIL %s\n       %s" % (run(["c++filt", s]).strip() or s, why))
    if bad or unknown:
        print("\nAn external-linkage data definition in code libcuda and libcudart both link is an ODR violation\n"
              "under AddressSanitizer in any test that loads both. Give it internal linkage (static / anonymous\n"
              "namespace), make it `inline`, or hide it (__attribute__((visibility(\"hidden\")))). A deliberately\n"
              "shared one goes in tests/lint/odr_globals.allow with the reason.")
        return 1
    print("odr globals: %d data symbols exported by both libraries (%d inline/COMDAT, %d allowed): none with a "
          "strong definition in shared code: PASS" % (len(shared), comdat, len(allowed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
