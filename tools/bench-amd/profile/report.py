#!/usr/bin/env python3
"""Turn libsigprof's samples into function and source-line tables (addr2line; stdlib only).

  python3 -I report.py samples.txt [--top 25] [--lines] [--inclusive] [--inline-callee]

Self time (the default): each sample is charged to the function the sampled instruction is in -- the outermost
non-inlined one, so an inlined callee is charged to the function that contains it; --inline-callee charges the
innermost inlined function instead (which helper of vector_alu the time is in). --lines charges the source line
(needs a -g build, which changes no code). --inclusive charges every function on the sampled call chain once
(needs SIGPROF_STACK=1 at record time; a chain is at most 10 frames deep, so the deepest callers may be missing).
"""
import argparse
import collections
import re
import subprocess
import sys

NOISE = [("vgpu::amd::(anonymous namespace)::", ""), ("vgpu::amd::gcn::", "gcn::"), ("vgpu::amd::", ""),
         ("(anonymous namespace)::", ""), ("std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >", "std::string"),
         ("std::basic_string_view<char, std::char_traits<char> >", "std::string_view")]


def strip_args(name):
    """Drop a trailing parameter list: the last balanced (...) group, if the name ends with it (or with const)."""
    name = re.sub(r"\s*\[clone [^\]]*\]", "", name.strip())
    name = re.sub(r"\s+const$", "", name)
    if not name.endswith(")"):
        return name
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        depth += name[i] == ")"
        depth -= name[i] == "("
        if depth == 0:
            return name[:i].rstrip()
    return name


def tidy(name):
    name = strip_args(name)
    name = re.sub(r"^auto ", "", name)
    for a, b in NOISE:
        name = name.replace(a, b)
    return name[:100]


def resolve(by_obj, exe_path):
    """{(obj, off): [(function, 'file:line'), ...innermost first]}"""
    out = {}
    for obj, offs in by_obj.items():
        exe = exe_path if obj == "/proc/self/exe" else obj
        offs = sorted(offs)
        res = subprocess.run(["addr2line", "-f", "-i", "-C", "-a", "-p", "-e", exe] + [hex(o) for o in offs],
                             capture_output=True, text=True)
        cur = None
        for line in res.stdout.splitlines():
            m = re.match(r"^0x([0-9a-f]+): (.*)$", line)
            if m:
                cur = out.setdefault((obj, int(m.group(1), 16)), [])
                body = m.group(2)
            else:
                body = line.strip()
                body = body[len("(inlined by) "):] if body.startswith("(inlined by) ") else body
            mm = re.match(r"(.*?) at (.*)$", body)
            if cur is not None:
                cur.append((mm.group(1), mm.group(2)) if mm else (body, "?"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--exe", default="/proc/self/exe", help="the program, for samples in it")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--lines", action="store_true")
    ap.add_argument("--inclusive", action="store_true")
    ap.add_argument("--inline-callee", action="store_true")
    ap.add_argument("--callers-of", help="instead of the table: who calls (the first frame above) a sampled leaf function "
                                         "whose name contains this (needs SIGPROF_STACK=1)")
    ap.add_argument("--within", help="only samples whose call chain passes through a function with this substring "
                                     "(needs SIGPROF_STACK=1): --within Machine::step leaves out the host program's own work")
    a = ap.parse_args()
    samples, total = [], 0
    by_obj = collections.defaultdict(set)
    with open(a.samples) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            n = int(parts[0])
            frames = []
            for i, p in enumerate(parts[1:]):
                obj, off = p.rsplit("+", 1)
                off = int(off, 16) - (1 if i else 0)   # a return address: the call, not the next instruction
                frames.append((obj, off))
                by_obj[obj].add(off)
            samples.append((n, frames))
            total += n
    syms = resolve(by_obj, a.exe)
    tab = collections.Counter()

    def label(obj, off, line_view=False):
        fr = syms.get((obj, off)) or [("??", "?")]
        if line_view:
            return fr[0][1].split(" (discriminator")[0].replace("/tmp/speedup-amdbase/src/", "")
        name = tidy(fr[0][0] if a.inline_callee else fr[-1][0])
        if "/shim/" not in obj and "libvgpu" not in obj and obj != "/proc/self/exe":
            name = "[%s] %s" % (obj.rsplit("/", 1)[-1], name)
        return name

    if a.callers_of:
        callers = collections.Counter()
        for n, frames in samples:
            if len(frames) > 1 and a.callers_of in label(*frames[0]):
                callers[label(*frames[1])] += n
        print("callers of %s" % a.callers_of)
        for k, n in callers.most_common(a.top):
            print("%7d  %s" % (n, k))
        return
    kept = 0
    for n, frames in samples:
        if not frames:
            continue
        if a.within and not any(a.within in label(o, f) for o, f in frames):
            continue
        kept += n
        if a.inclusive:
            seen = set()
            for obj, off in frames:
                seen.add(label(obj, off))
            for s in seen:
                tab[s] += n
        else:
            tab[label(frames[0][0], frames[0][1], a.lines)] += n
    total = kept if a.within else total
    print("total samples %d%s" % (total, " (within %s)" % a.within if a.within else ""))
    for k, n in tab.most_common(a.top):
        print("%6.2f%%  %7d  %s" % (100.0 * n / total, n, k))


if __name__ == "__main__":
    main()
