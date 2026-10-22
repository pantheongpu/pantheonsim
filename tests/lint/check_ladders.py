#!/usr/bin/env python3
"""No long `op == "name"_op` if-ladder is left in amd/src/gcn_exec.cpp.

The interpreter tells instructions apart by a switch on the name's hash (amd/tools/ladder-to-switch.py turns an if-ladder into
one). An instruction that reaches an if-ladder of 80 rungs costs a hash comparison per rung: that was 7-9 % of an RX 6800
scalar-load kernel's time when scalar_alu was one (a merge added a rung whose body has a `break`, and the converter, which
leaves a chain alone that has one a switch would capture, skipped the whole chain without a word).

This reads every if-chain of the file with the converter's own parser and fails for one whose first rungs are plain
`op == "..."_op` tests naming MAX_LABELS or more instructions (the converter's own limit is 4; a few short chains are fine).
"""
import importlib.util
import os
import re
import sys

MAX_LABELS = 12

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = os.path.join(root, "amd", "src", "gcn_exec.cpp")
spec = importlib.util.spec_from_file_location("ladder_to_switch", os.path.join(root, "amd", "tools", "ladder-to-switch.py"))
lts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lts)

text = open(src).read()
bad = []
for m in re.finditer(r"\bif\s*\(", text):
    if text[:m.start()].rstrip().endswith("else"):
        continue
    ls = text.rfind("\n", 0, m.start()) + 1
    if text[ls:m.start()].lstrip().startswith("//"):
        continue
    try:
        parsed = lts.parse_chain(lts.Src(text), m.start())
    except (RuntimeError, ValueError, IndexError):
        continue
    if not parsed:
        continue
    rungs = parsed[0]
    k = 0
    while k < len(rungs) and rungs[k]["pure"]:
        k += 1
    labels = sum(len(lts.NAME.findall(r["cond"])) for r in rungs[:k])
    if labels >= MAX_LABELS:
        bad.append((text.count("\n", 0, m.start()) + 1, k, labels))
for line, k, labels in bad:
    print("amd/src/gcn_exec.cpp:%d: an if-ladder of %d rungs naming %d instructions; run amd/tools/ladder-to-switch.py on the file. "
          "If it converts nothing, a rung's body has a `break` outside a loop of its own: write that rung without one." % (line, k, labels))
if bad:
    sys.exit(1)
print("ok: no if-ladder of %d or more instruction names" % MAX_LABELS)
