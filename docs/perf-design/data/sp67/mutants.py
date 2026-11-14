#!/usr/bin/env python3
"""Round sp67 (#569): mutants of the verify compare and of keep_verified_chunks (include/vgpu/host_gpu_track.hpp). Each mutant is a copy of the header with
one change, put first on the include path of a build of tests/unit/test_host_gpu_track.cpp against the tree's libvgpu.a (the library's own uses of the
header are not rebuilt: the test's instantiations are its own); the test must fail.

  mutants.py --src <tree> --build <build dir> --work <scratch dir> [--lock /tmp/pantheonsim-builder.lock] [--only N]
"""
import argparse
import os
import shutil
import subprocess

H = "include/vgpu/host_gpu_track.hpp"
MUTANTS = [
    ("compare: a chunk that differs is listed as agreeing", "        equal[i].push_back(off / kChunkSize);\n        same(i, base + off, n);",
     "        equal[i].push_back(off / kChunkSize);\n        same(i, base + off, n);\n      } else if (true) {\n        equal[i].push_back(off / kChunkSize);\n        differ(i, base + off, card[i].data() + off, scratch.data(), n);"),
    ("compare: the short last chunk is compared over a whole chunk", "const uint64_t n = std::min<uint64_t>(kChunkSize, size - off);\n      mem.read(", "const uint64_t n = kChunkSize;\n      mem.read("),
    ("compare: the lists are not cleared between compares", "    equal[i].clear();\n", ""),
    ("keep: every listed chunk becomes Both, whatever its state", "if (c < a.r.chunks() && a.r.at(c) == Where::Host) {", "if (c < a.r.chunks()) {"),
    ("keep: an agreeing chunk becomes Card, not Both", "a.r.set(c, Where::Both);", "a.r.set(c, Where::Card);"),
    ("keep: every chunk of the allocation becomes Both", "    for (uint64_t c : equal[i])\n      if (c < a.r.chunks()", "    for (uint64_t c = 0; c < a.r.chunks(); ++c)\n      if (c < a.r.chunks()"),
    ("keep: a range of another size is not skipped", "if (it == res.end() || it->second.size != ranges[i].second || equal[i].empty()) continue;",
     "if (it == res.end() || equal[i].empty()) continue;"),
    ("keep: the lookups are not voided", "  if (!newly.empty()) mem.card_void_lookups(newly);", "  (void)mem;"),
    ("keep: the new count is not published", "    kept += here;\n    update(a);", "    kept += here;"),
    ("keep: an allocation that had a copy is voided, one that had none is not", "    if (!was) newly.push_back(a.base);", "    if (was) newly.push_back(a.base);"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    ap.add_argument("--only", type=int)
    a = ap.parse_args()
    orig = open(os.path.join(a.src, H)).read()
    killed = 0
    for n, (name, old, new) in enumerate(MUTANTS, 1):
        if a.only and n != a.only:
            continue
        assert orig.count(old) == 1, f"mutant {n} does not apply: {name}"
        d = os.path.join(a.work, f"m{n}")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(os.path.join(d, "include/vgpu"))
        open(os.path.join(d, H), "w").write(orig.replace(old, new))
        exe = os.path.join(d, "test")
        cc = ["flock", a.lock, "nice", "-n", "19", "c++", "-DVGPU_TOOLKIT_CUDART_VERSION=13000", f"-I{d}/include", f"-I{a.src}/tests/framework",
              f"-I{a.src}/include", f"-I{a.src}/nvidia/include", f"-I{a.src}/amd/include", f"-I{a.build}/generated", "-O1", "-std=gnu++20",
              os.path.join(a.src, "tests/unit/test_host_gpu_track.cpp"), "-o", exe, os.path.join(a.build, "libvgpu.a"), "-ldl", "-lpthread"]
        c = subprocess.run(cc, capture_output=True, text=True)
        if c.returncode:
            print(f"M{n} {name}: does not compile\n{c.stderr[-800:]}")
            continue
        r = subprocess.run(["flock", a.lock, "nice", "-n", "19", "timeout", "300", exe], capture_output=True, text=True)
        fails = [l.split("]", 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("[ FAIL")]
        ok = r.returncode != 0
        killed += ok
        print(f"M{n} {name}: {'KILLED' if ok else 'SURVIVED'} (exit {r.returncode}; failing: {', '.join(sorted(set(fails))) or '-'})", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    print(f"{killed} of {len(MUTANTS) if not a.only else 1} killed")


if __name__ == "__main__":
    main()
