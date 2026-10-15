#!/usr/bin/env python3
"""Every exported driver function takes arguments of the sizes the toolkit declares.

The driver shim declares its entry points with its own spellings of the types
(a `void*` for a state the toolkit calls CUlinkState, an `int*` for an enum
pointer), which is harmless while each argument is the size the toolkit's
declaration gives it. CUPTI's callback API reads the arguments by the sizes of
the toolkit's parameter-structure fields (nvidia/src/cupti_api.cpp), and a call
whose arguments differ in size from NVIDIA's is not delivered to a subscriber
rather than read from the wrong bytes. This finds those calls before a
profiler does: a shim function declared with a 4-byte argument where the
toolkit has 8 is a bug in the shim's ABI as well.

It takes the parameter list of each `VGPU_EXPORT CUresult name(...)` in
nvidia/src/driver_api.cpp and compiles, against the toolkit's cuda.h, a
static_assert per function that the two lists have the same number of
arguments of the same sizes. SKIPs where no cuda.h is at hand.

usage: check_driver_arg_sizes.py [<cuda include dir>]
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "nvidia" / "src" / "driver_api.cpp"


def find_header(arg):
    candidates = [arg] if arg else []
    candidates += [os.environ.get("CUDA_ABI_INCLUDE", ""), "/usr/local/cuda/include", "/usr/include"]
    for c in candidates:
        if c and (pathlib.Path(c) / "cuda.h").exists():
            return pathlib.Path(c)
    return None


def exports(text):
    """Yields (name, parameter text) for each exported CUresult function."""
    for m in re.finditer(r"^VGPU_EXPORT CUresult (\w+)\(", text, re.M):
        i, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        params = " ".join(text[m.end():i - 1].split())
        yield m.group(1), ("" if params == "void" else params)


def main():
    include = find_header(sys.argv[1] if len(sys.argv) > 1 else "")
    if include is None:
        print("SKIP: no cuda.h to compare the driver shim's declarations with")
        return 0
    header = (include / "cuda.h").read_text(errors="replace")
    declared = set(re.findall(r"CUDAAPI\s+(cu\w+)\s*\(", header))
    # cuda.h declares cuMemAlloc and renames it with `#define cuMemAlloc
    # cuMemAlloc_v2`, so a versioned export is compared with the declaration of
    # its base name, with the rename in force.
    renamed = {v: b for b, v in re.findall(r"#\s*define\s+(cu\w+)\s+(cu\w+_v\d+)\b", header)}
    wanted = []   # (name, params, expression naming the toolkit's function, needs the rename off)
    skipped = []
    sweeps = [SOURCE.with_name(n) for n in ("driver_sweep.inc", "driver_132.inc")]   # the functions the header sweeps added
    for name, params in exports(SOURCE.read_text() + "".join(f.read_text() for f in sweeps if f.exists())):
        if name in renamed and renamed[name] in declared:
            wanted.append((name, params, renamed[name], False))
        elif name in declared:
            wanted.append((name, params, name, True))
        else:
            skipped.append((name, params))
    wanted.sort(key=lambda w: w[3])   # the renamed ones first: undoing a rename is for the rest
    prelude = """
#include <array>
#include <cstddef>
#include <cuda.h>
template <class F> struct sizes;
template <class... A> struct sizes<CUresult (*)(A...)> {
  static constexpr std::array<std::size_t, 17> v{sizeof(A)...};
  static constexpr std::size_t n = sizeof...(A);
};
template <class A, class B> constexpr bool same_sizes() {
  if (sizes<A>::n != sizes<B>::n) return false;
  for (std::size_t i = 0; i < sizes<A>::n; ++i)
    if (sizes<A>::v[i] != sizes<B>::v[i]) return false;
  return true;
}
"""
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "check.cpp"
        # Some names are declared only in blocks the header compiles out, so a
        # name the compiler does not know is dropped and the rest checked again.
        for _ in range(8):
            lines = []
            for name, params, base, undef in wanted:
                if undef:
                    lines.append(f"#undef {base}")
                lines.append(f"using S_{name} = CUresult (*)({params});")
                lines.append(f'static_assert(same_sizes<S_{name}, decltype(&{base})>(), "{name}");')
            src.write_text(prelude + "\n".join(lines) + "\n")
            r = subprocess.run(["g++", "-std=c++20", "-fsyntax-only", "-w", f"-I{include}", str(src)],
                               capture_output=True, text=True)
            unknown = set(re.findall(r"[‘'](cu\w+)[’'] was not declared", r.stderr))
            unknown -= {n for n, *_ in wanted if f"static assertion failed: {n}" in r.stderr}
            if not unknown:
                break
            gone = [w for w in wanted if (w[0] if not w[3] else w[2]) in unknown]
            skipped += [(w[0], w[1]) for w in gone]
            wanted = [w for w in wanted if w not in gone]
    bad = sorted(set(re.findall(r"static assertion failed: (\w+)", r.stderr)))
    other = [l for l in r.stderr.splitlines() if "error:" in l and "static assertion failed" not in l]
    if bad or other:
        for n in bad:
            print(f"FAIL: {n} takes arguments of other sizes than the toolkit declares")
        for l in other[:10]:
            print(l)
        return 1
    if os.environ.get("SHOW_SKIPPED"): print(" ".join(n for n, _ in skipped))
    print(f"driver argument sizes: {len(wanted)} functions match the toolkit's declarations "
          f"({len(skipped)} not declared there by that name): PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
