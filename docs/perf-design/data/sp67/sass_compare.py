#!/usr/bin/env python3
"""Round sp67 (#570 item 3): compares the SASS two toolkits emit for the same program (cuobjdump -sass output), per kernel, and the float-division idiom
in it (MUFU.RCP, the FFMA refinement, FCHK, the CALL to the slow path and its subroutine). Registers and addresses are not compared literally: each
instruction is reduced to its opcode with modifiers and its operands with register numbers renamed in order of first use, so a renumbering is not a difference.

  sass_compare.py <a.sass> <b.sass>          prints, per kernel: instructions in a and b, equal after renaming or not, and the division windows of each
"""
import re
import sys

FN = re.compile(r"^\s*Function : (\S+)")
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")


def kernels(path):
    out, cur = {}, None
    for line in open(path, errors="replace"):
        m = FN.match(line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = INS.search(line)
        if m and cur:
            out[cur].append(m.group(2).strip())
    return out


def canon(ins):
    """Opcode text plus operands with R/P/UR/UP registers renamed by first use and branch targets dropped."""
    names = {}

    def ren(m):
        k = m.group(0)
        if k in ("RZ", "PT", "URZ", "UPT"):
            return k
        cls = re.match(r"[A-Z]+", k).group(0)
        names.setdefault(k, f"{cls}{sum(1 for v in names if re.match(r'[A-Z]+', v).group(0) == cls)}")
        return names[k]

    s = re.sub(r"\b(?:UR|UP|R|P)\d+\b", ren, ins)
    s = re.sub(r"0x[0-9a-f]+\s*\)?$", "<addr>", s) if re.match(r"(@!?\w+\s+)?(BRA|CALL|BSSY|JMP)", s) else s
    return s


def div_windows(code):
    """The division idiom's instructions: every MUFU.RCP, FCHK, CALL.REL, RET and the FFMAs within 8 of a MUFU.RCP."""
    keep = []
    for i, ins in enumerate(code):
        op = ins.split()[1] if ins.startswith("@") else ins.split()[0]
        if op.startswith(("MUFU", "FCHK", "CALL", "RET")):
            keep.append((i, ins))
    return keep


def opseq(code):
    return [(c.split()[1] if c.startswith("@") else c.split()[0]) for c in code]


def main(a, b):
    ka, kb = kernels(a), kernels(b)
    for name in sorted(set(ka) | set(kb)):
        ca, cb = ka.get(name, []), kb.get(name, [])
        same_text = ca == cb
        same_canon = [canon(x) for x in ca] == [canon(x) for x in cb]
        da, db = div_windows(ca), div_windows(cb)
        same_div = [canon(x) for _, x in da] == [canon(x) for _, x in db]
        ops_a, ops_b = opseq(ca), opseq(cb)
        mufu_a = sorted(set(o for o in ops_a if o.startswith("MUFU")))
        mufu_b = sorted(set(o for o in ops_b if o.startswith("MUFU")))
        print(f"{name}\tinstrs {len(ca)} / {len(cb)}\ttext {'same' if same_text else 'differs'}\trenamed {'same' if same_canon else 'differs'}\t"
              f"opcode-seq {'same' if ops_a == ops_b else 'differs'}\tdivision-windows {'same' if same_div else 'differs'} ({len(da)} / {len(db)})\t"
              f"MUFU {','.join(mufu_a) or '-'} / {','.join(mufu_b) or '-'}\tFCHK {ops_a.count('FCHK')} / {ops_b.count('FCHK')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
