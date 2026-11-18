#!/usr/bin/env python3
"""The profile counts in docs/status.md match the profile files.

Counts nvidia/profiles/*.yaml and amd/profiles/*.yaml, and those with `verified: true`, and compares them with the sentence
"There are N (...): A NVIDIA and B AMD. <word> NVIDIA profiles and <word> AMD profiles were read from a physical card" and with
"The other M come from public documents" in docs/status.md. Also checks the digit forms "14 NVIDIA and 3 AMD" / "17 verified" in
README.md, CONTRIBUTING.md and ARCHITECTURE.md where they appear. Standard library only, no build.
Usage: check_profile_counts.py [repo-root]
"""
import glob, os, re, sys

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()

def num(t):
    t = t.lower()
    return int(t) if t.isdigit() else WORDS.index(t)

def count(vendor):
    files = sorted(glob.glob(os.path.join(root, vendor, "profiles", "*.yaml")))
    ver = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            if any(re.match(r"^verified:\s*true\s*(#.*)?$", l) for l in fh):
                ver += 1
    return len(files), ver

def read(p):
    with open(os.path.join(root, p), encoding="utf-8") as f:
        return f.read()

nv, nv_ver = count("nvidia")
am, am_ver = count("amd")
total = nv + am
problems = []

s = read("docs/status.md")
m = re.search(r"There are (\d+) \(`vgpu list-gpus`\): (\d+) NVIDIA and (\d+) AMD\. (\w+) NVIDIA profiles and (\w+) AMD profiles were read "
              r"from a physical card[^.]*\.\s*The other (\d+) come from", s)
if not m:
    problems.append("docs/status.md: the GPU profiles sentence was not found in the expected form")
else:
    got = (int(m[1]), int(m[2]), int(m[3]), num(m[4]), num(m[5]), int(m[6]))
    want = (total, nv, am, nv_ver, am_ver, total - nv_ver - am_ver)
    names = ("total", "NVIDIA", "AMD", "verified NVIDIA", "verified AMD", "unverified")
    for n, g, w in zip(names, got, want):
        if g != w:
            problems.append("docs/status.md says %s %d, the tree has %d" % (n, g, w))

# Other documents restate the numbers: each must agree with the tree.
def other(path, pats):
    t = read(path)
    for pat, fn in pats:
        for mm in re.finditer(pat, t):
            msg = fn(mm)
            if msg:
                problems.append("%s: %s" % (path, msg))

pair = r"(\d+) NVIDIA (?:profiles )?and (\d+|three) AMD"
def chk_pair(mm):
    a, b = int(mm[1]), num(mm[2])
    if (a, b) != (nv_ver, am_ver):
        return "says %d NVIDIA and %d AMD verified, the tree has %d and %d" % (a, b, nv_ver, am_ver)
for p in ("ARCHITECTURE.md", "CONTRIBUTING.md", "README.md"):
    other(p, [(pair, chk_pair)])
other("CONTRIBUTING.md", [(r"the (\d+) \"verified\"", lambda mm: None if int(mm[1]) == nv_ver + am_ver else
                           "says %s verified profiles, the tree has %d" % (mm[1], nv_ver + am_ver))])
other("README.md", [(r"(\d+) of the (\d+) GPU profiles come from public", lambda mm: None if (int(mm[1]), int(mm[2])) ==
                     (total - nv_ver - am_ver, total) else "says %s of %s, the tree has %d of %d" % (mm[1], mm[2], total - nv_ver - am_ver, total))])
other("README.md", [(r"The (\d+) that were read", lambda mm: None if int(mm[1]) == nv_ver + am_ver else "says %s read from a card, the tree has %d" % (mm[1], nv_ver + am_ver))])

for p in problems:
    print("FAIL " + p)
if problems:
    sys.exit(1)
print("ok: %d profiles (%d NVIDIA, %d AMD), %d verified (%d NVIDIA, %d AMD)" % (total, nv, am, nv_ver + am_ver, nv_ver, am_ver))
