#!/usr/bin/env python3
"""Writes the one-line-a-vector files tools/tcpapers/khattak_vectors.cu reads, from a checkout of
github.com/north-numerical-computing/MATLAB-tensor-core (model_validation/). Our own script.

  khattak_convert.py <MATLAB-tensor-core checkout> <out dir> [first N vectors, default all]

Files written: <out>/<card>.<fmt>.<out>.txt  (card A100 A2 Ada L40S H100 H200 B200 V100; fmt fp16 bf16 tf32 e4m3 e5m2; out fp32 fp16).
The e4m3 and e5m2 files are for tools/tcpapers/khattak_fp8_vectors.cu (mma.sync m16n8k32, K = 32). A vector's width K is the card's
dot-product length in the paper's own validation script (V100 4, A100 / A2 / Ada / L40S 8 for f16 and bf16, H100 / H200 / B200 16,
tf32 4, FP8 32).
"""
import os, struct, sys

K16 = {"V100": 4, "A100": 8, "A2": 8, "Ada": 8, "L40S": 8, "H100": 16, "H200": 16, "B200": 16}

def bits32(s):          # a 32-character binary string to 8 hex digits
    return "%08x" % int(s, 2)

def main():
    root, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    os.makedirs(out, exist_ok=True)
    for card in sorted(K16):
        for fmt in ("fp16", "bf16", "tf32", "e4m3", "e5m2"):
            d = os.path.join(root, "model_validation", card, fmt.upper() if fmt[0] == "e" else fmt)
            if not os.path.isdir(d):
                continue
            k = 32 if fmt[0] == "e" else 4 if fmt == "tf32" else K16[card]
            tag = fmt.upper() if fmt[0] == "e" else fmt
            a = open(os.path.join(d, f"a_{card}_{tag}.txt")).read().split("\n")
            b = open(os.path.join(d, f"b_{card}_{tag}.txt")).read().split("\n")
            c = open(os.path.join(d, f"c_{card}_fp32.txt")).read().split()
            for o in ("fp32", "fp16"):
                dp = os.path.join(d, f"d_{card}_{o}.txt")
                if not os.path.exists(dp):
                    continue
                dd = open(dp).read().split()
                lines = []
                for i in range(len(dd)):
                    ai, bi = a[i].split(), b[i].split()
                    assert len(ai) == k and len(bi) == k, (card, fmt, i, len(ai), k)
                    lines.append(" ".join(ai + bi + [bits32(c[i]), bits32(dd[i])]))
                if n:
                    lines = lines[:n]
                with open(os.path.join(out, f"{card}.{fmt}.{o}.txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")

main()
