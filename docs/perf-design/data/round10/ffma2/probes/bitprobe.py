import subprocess, sys, re, os, tempfile
def dis(encs, sm="SM100a"):
    d = tempfile.mkdtemp()
    raw = os.path.join(d, "x.bin")
    with open(raw, "wb") as f:
        for hi, lo in encs:
            f.write(lo.to_bytes(8, "little") + hi.to_bytes(8, "little"))
    out = subprocess.run(["nvdisasm", "-b", sm, "-hex", raw], capture_output=True, text=True)
    texts = {}
    for line in out.stdout.splitlines():
        m = re.match(r'^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\*\s*0x', line)
        if m: texts[int(m.group(1), 16)] = " ".join(m.group(2).split())
    return [texts.get(i * 16, "<none: %s>" % out.stderr.strip()[:60]) for i in range(len(encs))]
def flips(hi, lo, sm="SM100a"):
    encs = [(hi, lo)]
    for b in range(128):
        if b < 64: encs.append((hi, lo ^ (1 << b)))
        else: encs.append((hi ^ (1 << (b - 64)), lo))
    t = [dis([e], sm)[0] for e in encs]
    print("BASE", hex(hi), hex(lo), "|", t[0])
    for b in range(128):
        if t[b + 1] != t[0]: print("  bit %3d: %s" % (b, t[b + 1]))
if __name__ == "__main__":
    for a in sys.argv[1:]:
        hi, lo = a.split(":")
        flips(int(hi, 16), int(lo, 16))
