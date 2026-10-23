#!/usr/bin/env python3
"""Classify the difference between two runs (tags) of every case produced by corpus.py.
usage: compare.py <rundir> <tagA> <tagB> [--json out.json] [--md]
classes: identical (bytes, rc, files) | timing-only (identical after the timing filter) | numeric-close (same text skeleton, every number within
1e-4 relative or 1e-9 absolute) | differs | A-fails / B-fails (one side rc!=0 or timed out) | both-fail-same | both-fail-differ"""
import sys, os, json, re, glob, difflib

TIME = re.compile(r"(?i)(\btime\b|\btimes\b|elapsed|\bsec\b|\bsecs\b|seconds|\bmsec|\bms\b|\busec|\bus\b|\bmicro|gflop|tflop|gops|\btops\b|gb/s|gib/s|mb/s|bandwidth|throughput|latency|runtime|speedup|"
                  r"\bfps\b|per second|/s\b|/sec\b|\bclock|\bcycles?\b|\bperformance\b|\bperf\b|\bkernel execution\b|\bduration\b|\btook\b|\bwall|\bcpu\b|\brss\b|\bmemory used|\bFOM\b|\bgpu:|\bGPU MEM|\bFLOPS?\b|\bGB\b.*\bs\b|\bdate\b|\btimestamp)")
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|inf")


def load(d, tag):
    out = {}
    for p in glob.glob(os.path.join(d, tag, "*.json")):
        r = json.load(open(p))
        out[r["corpus"] + "/" + r["case"]] = r
    return out


def text(r):
    return r["stdout"] + (("\n--stderr--\n" + r["stderr"]) if r["stderr"].strip() else "")


BARE = re.compile(r"^\s*[-+]?[\d.]+(e[-+]?\d+)?\s*$", re.I)
EXTRA = re.compile(r"(?i)(\d\s?s\b|\bTOOK\b|\bTIME TO\b|\bBACK:|\bFREE TIME|memcopy:|\bGPU Execution|\bcontext=0x|shm\d+|\b0x[0-9a-f]{6,}\b)")


IDENT = re.compile(r"(?i)(A10G|H100|RTX ?5090|GeForce|\bL4\b|L40S|A100|Tesla T4|\bT4\b|Ampere|Hopper|Blackwell|Turing|Ada Lovelace|compute capability|Multi-?Processors|Compute \d\.\d|"
                    r"Driver version|\bsm_\d\d|Device Name|device \d with name|Using CUDA Device|CUDA device|\bGPU Device|running on GPU|Estimating Pi on GPU|Pricing option on GPU|\bGPU \d?:?\s*NVIDIA|SMs?\b|multiprocessor)")
IDENT_ON = os.environ.get("IDENT") == "1"  # IDENT=1: also drop lines that only state which device it is (name, capability, SM count, driver version)


def filt(t):
    """Drops timing lines: those with a time/throughput word or a number followed by 's', and a bare-number line that follows one
    (PolyBench prints 'GPU Time in seconds:' then the number on the next line). Pointers and pids in names are dropped too."""
    out, prev_timing = [], False
    for l in t.splitlines():
        timing = bool(TIME.search(l) or EXTRA.search(l) or (IDENT_ON and IDENT.search(l)))
        if timing or not l.strip() or (prev_timing and BARE.match(l)):
            prev_timing = timing or (prev_timing and bool(BARE.match(l)))
            continue
        prev_timing = False
        out.append(l.rstrip())
    return "\n".join(out)


def close(a, b):
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        if x == y:
            continue
        if NUM.sub("#", x) != NUM.sub("#", y):
            return False
        for u, v in zip(NUM.findall(x), NUM.findall(y)):
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                if u != v: return False
                continue
            if abs(fu - fv) > max(1e-9, 1e-4 * max(abs(fu), abs(fv))) and not (fu != fu and fv != fv):
                return False
    return True


def ok(r):
    return r["rc"] == 0 and not r["timeout"]


def classify(a, b):
    if a is None or b is None:
        return "missing", ""
    if not ok(a) or not ok(b):
        if ok(a): return "B-fails", ("timeout" if b["timeout"] else "rc=%s" % b["rc"])
        if ok(b): return "A-fails", ("timeout" if a["timeout"] else "rc=%s" % a["rc"])
        same = a["rc"] == b["rc"] and a["timeout"] == b["timeout"] and filt(text(a)) == filt(text(b))
        return ("both-fail-same" if same else "both-fail-differ"), "rc=%s/%s" % (a["rc"], b["rc"])
    ta, tb = text(a), text(b)
    fa, fb = filt(ta), filt(tb)
    files_same = a["files"] == b["files"]
    if ta == tb and files_same: return "identical", ""
    if fa == fb and files_same: return "timing-only", ""
    if fa == fb and not files_same:
        return "differs", "output files: " + ",".join(sorted(k for k in set(a["files"]) | set(b["files"]) if a["files"].get(k) != b["files"].get(k)))
    if close(fa, fb) and files_same: return "numeric-close", ""
    d = [l for l in difflib.unified_diff(fa.splitlines(), fb.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    extra = "" if files_same else " [files differ: " + ",".join(sorted(k for k in set(a["files"]) | set(b["files"]) if a["files"].get(k) != b["files"].get(k))) + "]"
    return "differs", (" | ".join(x[:110] for x in d[:2]) + extra)


if __name__ == "__main__":
    d, ta, tb = sys.argv[1:4]
    A, B = load(d, ta), load(d, tb)
    res = {}
    for k in sorted(set(A) | set(B)):
        res[k] = classify(A.get(k), B.get(k))
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    from collections import Counter
    c = Counter(v[0] for v in res.values())
    print(ta, "vs", tb, dict(c), "of", len(res))
    for k, (cl, why) in res.items():
        if cl not in ("identical", "timing-only"):
            print("  %-14s %-44s %s" % (cl, k, why))
