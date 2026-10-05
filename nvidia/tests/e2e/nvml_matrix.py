"""The simulator's NVML getters return what an RTX 3060's NVML returns.

usage: nvml_matrix.py <probe> <library> <device index> <matrix.tsv> [--card]

<probe> is nvidia/tools/nvml_card_probe_gen.py's program, built from the toolchain's
nvml.h; it prints "<function> <return code> <leading bytes of the first output>" for
every getter. The matrix file (nvidia/tests/data/nvml_rtx3060_matrix.tsv) holds what
NVIDIA's NVML answered for each on the card and what the simulator answers, with the
reason wherever the two differ.

Without --card the probe runs against the shim and each answer must be the matrix's
simulator column, and the fixed-fact values (memory bus width, brand, fans, ...) must
begin with the card's bytes. With --card it runs against NVIDIA's own library and each
answer must be the card column, which is how the matrix is regenerated and checked on a
machine with the card. A function the toolchain's header does not declare is not probed
and is skipped.
"""
import subprocess, sys

probe, lib, index, matrix = sys.argv[1:5]
card_mode = "--card" in sys.argv
want = {}
for line in open(matrix):
    if line.startswith("#") or not line.strip():
        continue
    name, card, sim, value, note = (line.rstrip("\n").split("\t") + [""] * 5)[:5]
    want[name] = (card, sim, value, note)

# A card or a shim that crashes in one getter loses the rest of the output; say which.
run = subprocess.run([probe, lib, index], capture_output=True, text=True, timeout=240)
got = {}
for line in run.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 2:
        got[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
fails = 0
compared = 0
for name, (rc, hexval) in sorted(got.items()):
    if name not in want:
        continue
    card, sim, value, note = want[name]
    expected = card if card_mode else sim
    compared += 1
    ok = rc == expected
    if ok and value and rc == "0" and not hexval.startswith(value):
        ok = False
        rc = f"0 with {hexval[:2 * len(value)]}"
        expected = f"0 with {value}"
    if not ok:
        fails += 1
        print(f"FAIL {name}: {rc}, expected {expected}")
# Every difference from the card is explained.
for name, (card, sim, value, note) in want.items():
    if card != sim and not note:
        fails += 1
        print(f"FAIL {name}: the simulator answers {sim} where the card answers {card}, and the matrix says nothing about why")
print(f"{compared} getters compared ({'card' if card_mode else 'simulator'} column), {fails} failed;"
      f" {sum(1 for v in want.values() if v[0] != v[1])} documented differences from the card")
if compared < 100 or run.returncode != 0:
    print("FAIL the probe produced too little output", run.returncode, run.stderr[-300:])
    fails += 1
sys.exit(1 if fails else 0)
