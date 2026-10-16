#!/usr/bin/env python3
"""Summarise a tools/verify/run_corpora.sh results directory.

    python3 -I tools/verify/summarize.py <results dir> [--pairs B:A,Bt:A,Bt:B,Bp:Ap,A2:A] [--diff-lines N]

For every program and every pair X:Y of configurations it compares X against Y on five things:
exit status, stdout, stderr (the simulator's diagnostics, with the counter blocks and tier
statistics removed), the files the program wrote (md5), and the counters (VGPU_COUNTERS blocks
in stderr and the VGPU_COUNTERS_FILE records). Writes report.md and pairs.tsv into the results
directory and prints the report.

Lines that carry a timing or rate (see TIMING_WORDS) have their numbers replaced by N before they
are compared: the report says how many programs were equal only after that. Programs in a
"timed" corpus (time-based workloads: the number of launches depends on the speed of the host)
are compared by their verification lines and by the SET of distinct counter records, not by the
sequence.

A pair's verdict:
  identical          same exit status, output, files and counters (both sides ran to the end, rc 0)
  identical-fail     the same, but both sides exit non-zero: a failure that is not a difference
  nondeterministic   differs, but A and A2 (the same build run twice) differ in the same component
  different          differs and A does not differ from itself there
  timeout-vs-done    one side hit the timeout and the other did not: not comparable
  both-timeout       both sides hit the timeout: not comparable
  n/a                a side was not run
"""
import argparse
import difflib
import json
import os
import re
import statistics
import sys

TIMING_WORDS = re.compile(
    r"(\d\s*s\b|\btook\b|execution|\bback\b|time|elapsed|second|\bsecs?\b|\bms\b|msec|usec|\bus\b|µs|gflop|tflop|gb/s|mb/s|gbps|throughput|bandwidth|"
    r"speedup|latency|\brate\b|/s\b|per second|cycles|clock|duration|runtime|perf|fps|gigaflops|"
    r"kernel execution|total |average|avg|elapsed)", re.I)
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
VERIFY = re.compile(r"(verif|\bpass(ed)?\b|\bfail|error|mismatch|golden|checksum|valid|correct|\bok\b|invalid)", re.I)
COUNTER_HEAD = "[vgpu][counters]"
TIER_RE = re.compile(r"^(\[vgpu\]\s*)?tier[ :(]")
PID = re.compile(r"\bpid[= ]\d+", re.I)


def read(path, binary=False):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return data if binary else data.decode("utf-8", "replace")


def status_of(rc):
    if rc is None:
        return "missing"
    if rc in (124, 137, 143):
        return "timeout"
    return "ok" if rc == 0 else "fail"


PURE_NUM = re.compile(r"^\s*[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?\s*$")


def mask_timing(lines):
    out = []
    prev_timing = False
    for ln in lines:
        timing = bool(TIMING_WORDS.search(ln))
        # a bare number on the line after a timing label ("GPU Time in seconds:" / "0.091")
        if timing or (prev_timing and PURE_NUM.match(ln)):
            ln = re.sub(r"\s+", " ", NUM.sub("N", ln)).strip()
        out.append(ln)
        prev_timing = timing
    return out


def split_stderr(text):
    """-> (other lines, counter blocks as tuples of lines, tier-statistics lines)"""
    other, blocks, tier = [], [], []
    cur = None
    for ln in text.splitlines():
        if ln.startswith(COUNTER_HEAD):
            cur = [ln]
            blocks.append(cur)
        elif cur is not None and ln.startswith("    "):
            cur.append(ln)
        else:
            cur = None
            if TIER_RE.match(ln):
                tier.append(ln)
            else:
                other.append(ln)
    return other, [tuple(b) for b in blocks], tier


class Run:
    def __init__(self, d):
        self.dir = d
        rc = read(os.path.join(d, "rc"))
        self.rc = int(rc.strip()) if rc and rc.strip().lstrip("-").isdigit() else None
        self.status = status_of(self.rc)
        self.meta = {}
        for ln in (read(os.path.join(d, "meta")) or "").splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                self.meta[k] = v
        self.secs = float(self.meta.get("secs", "nan"))
        self.stdout = (read(os.path.join(d, "stdout")) or "")
        err = read(os.path.join(d, "stderr")) or ""
        self.stderr_other, self.blocks, self.tier = split_stderr(err)
        self.files = (read(os.path.join(d, "files.md5")) or "")
        recs = []
        for ln in (read(os.path.join(d, "counters.jsonl")) or "").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            r.pop("pid", None)
            recs.append(r)
        self.records = recs


def timed_map_blocks(run):
    """kernel+shape -> set of the counter lines that follow it"""
    m = {}
    for blk in run.blocks:
        m.setdefault(blk[0], set()).add(blk[1:])
    return m


def timed_map_records(run):
    m = {}
    for r in run.records:
        k = (r.get("kernel"), tuple(r.get("grid", ())), tuple(r.get("block", ())))
        v = json.dumps({a: b for a, b in r.items() if a not in ("launch", "kernel", "grid", "block")}, sort_keys=True)
        m.setdefault(k, set()).add(v)
    return m


def shared_mismatch(a, b):
    """keys both sides launched whose value sets have nothing in common (a kernel whose counts depend on the
    data, such as a clock reading, gives several values inside ONE run: that is not a difference)"""
    return sorted(str(k) for k in a if k in b and not (a[k] & b[k]))


def timed_counters_equal(x, y):
    """Time-based programs launch a host-speed-dependent number of kernels, some with host-speed-dependent
    shapes: compare the counter values of the launches (kernel, grid, block) both sides have."""
    bad = shared_mismatch(timed_map_blocks(x), timed_map_blocks(y)) or shared_mismatch(timed_map_records(x), timed_map_records(y))
    return (not bad, not bad)


def component_diffs(x, y, timed, card=False):
    """-> dict component -> (raw_equal, norm_equal)"""
    res = {}
    res["rc"] = (x.rc == y.rc, x.rc == y.rc)

    def cmp_lines(a, b):
        raw = a == b
        if timed:
            a2 = [l for l in a if VERIFY.search(l)]
            b2 = [l for l in b if VERIFY.search(l)]
            return raw, a2 == b2
        return raw, mask_timing(a) == mask_timing(b)

    res["stdout"] = cmp_lines(x.stdout.splitlines(), y.stdout.splitlines())
    res["stderr"] = cmp_lines(x.stderr_other, y.stderr_other)
    # files: a name+md5 listing; time-based programs write timestamps and rates
    res["files"] = (x.files == y.files, x.files == y.files)
    if timed:
        cb = timed_counters_equal(x, y)
        cr = cb
    else:
        cb = (x.blocks == y.blocks,) * 2
        cr = (x.records == y.records,) * 2
    res["counters"] = (cb[0] and cr[0], cb[1] and cr[1])
    if card:   # the real card has no simulator diagnostics and no counters: compare what a user sees
        for k in ("stderr", "counters"):
            res[k] = (True, True)
    return res


def first_diff(a, b, n):
    d = list(difflib.unified_diff(a, b, "X", "Y", n=0, lineterm=""))
    return [l[:200] for l in d[2:2 + n]]


def diff_text(x, y, comp, n, timed):
    if comp == "rc":
        return [f"rc {x.rc} vs {y.rc}"]
    if comp == "stdout":
        a, b = x.stdout.splitlines(), y.stdout.splitlines()
    elif comp == "stderr":
        a, b = x.stderr_other, y.stderr_other
    elif comp == "files":
        a, b = x.files.splitlines(), y.files.splitlines()
    else:
        a = [" | ".join(blk) for blk in x.blocks]
        b = [" | ".join(blk) for blk in y.blocks]
        if not a and not b:
            a = [json.dumps(r, sort_keys=True) for r in x.records]
            b = [json.dumps(r, sort_keys=True) for r in y.records]
        if timed:
            ma, mb = timed_map_blocks(x), timed_map_blocks(y)
            bad = shared_mismatch(ma, mb)
            if not bad:
                ma, mb = timed_map_records(x), timed_map_records(y)
                bad = shared_mismatch(ma, mb)
            a = [f"{k}: {sorted(ma[k])}" for k in ma if str(k) in bad]
            b = [f"{k}: {sorted(mb[k])}" for k in mb if str(k) in bad]
    if comp in ("stdout", "stderr") and not timed:
        a, b = mask_timing(a), mask_timing(b)
    if comp in ("stdout", "stderr") and timed:
        a, b = [l for l in a if VERIFY.search(l)], [l for l in b if VERIFY.search(l)]
    return first_diff(a, b, n) or ["(no line diff: lengths %d vs %d)" % (len(a), len(b))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--pairs", default="B:A,Bt:A,Bt:B,Bp:Ap,A2:A")
    ap.add_argument("--diff-lines", type=int, default=6)
    args = ap.parse_args()
    root = os.path.abspath(args.results)
    modes = {}
    try:
        for ln in open(os.path.join(root, "programs.tsv")):
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 4:
                modes[(f[0], f[1])] = f[2]
    except OSError:
        pass
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p]
    corpora = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != "pantheon-bin")
    rows = []        # (corpus, prog, pair, verdict, comps, detail lines)
    runs = {}        # (corpus, prog) -> {cfg: Run}
    for c in corpora:
        for p in sorted(os.listdir(os.path.join(root, c))):
            pd = os.path.join(root, c, p)
            if not os.path.isdir(pd):
                continue
            runs[(c, p)] = {cfg: Run(os.path.join(pd, cfg)) for cfg in sorted(os.listdir(pd)) if os.path.isdir(os.path.join(pd, cfg))}
    stats = {}  # (corpus, pair) -> verdict -> count
    selfdiff = {}
    for (c, p), cfgs in sorted(runs.items()):
        mode = next((r.meta["mode"] for r in cfgs.values() if "mode" in r.meta), modes.get((c, p)))
        timed = mode == "timed" or (mode is None and c == "pantheon")
        # which components differ between A and A2: those are not stable even on one build
        unstable = set()
        if "A" in cfgs and "A2" in cfgs and cfgs["A"].status != "missing" and cfgs["A2"].status != "missing" \
                and cfgs["A"].status != "timeout" and cfgs["A2"].status != "timeout":
            cd = component_diffs(cfgs["A"], cfgs["A2"], timed)
            unstable = {k for k, (raw, norm) in cd.items() if not norm}
        selfdiff[(c, p)] = unstable
        for (xn, yn) in pairs:
            x, y = cfgs.get(xn), cfgs.get(yn)
            key = (c, f"{xn}:{yn}")
            v = stats.setdefault(key, {})
            if not x or not y or x.rc is None or y.rc is None:
                verdict, comps, det = "n/a", [], []
            elif x.status == "timeout" and y.status == "timeout":
                verdict, comps, det = "both-timeout", [], []
            elif x.status == "timeout" or y.status == "timeout":
                verdict, comps, det = "timeout-vs-done", ["rc"], [f"{xn} {x.status} ({x.secs:.0f}s), {yn} {y.status} ({y.secs:.0f}s)"]
            else:
                cd = component_diffs(x, y, timed, "C" in (xn, yn))
                bad = [k for k, (raw, norm) in cd.items() if not norm]
                masked = [k for k, (raw, norm) in cd.items() if norm and not raw]
                if not bad:
                    ok = x.status == "ok" and y.status == "ok"
                    verdict = "identical" if ok else "identical-fail"
                    if masked:
                        verdict += "*"      # equal only after masking timing lines / numbers
                    comps, det = masked, []
                else:
                    if all(k in unstable for k in bad):
                        verdict = "nondeterministic"
                    else:
                        verdict = "different"
                    comps = bad
                    det = []
                    for k in bad:
                        det.append(f"[{k}]")
                        det.extend("    " + l for l in diff_text(x, y, k, args.diff_lines, timed))
            v[verdict] = v.get(verdict, 0) + 1
            rows.append((c, p, f"{xn}:{yn}", verdict, comps, det))

    out = []
    w = out.append
    w("# Corpus verification report")
    w("")
    w(f"Results directory: `{root}`")
    n_prog = len(runs)
    w(f"Programs: {n_prog}; pairs compared: {', '.join(f'{a}:{b}' for a, b in pairs)}")
    loads = []
    for cfgs in runs.values():
        for r in cfgs.values():
            try:
                loads.append(float(r.meta.get("load_at_start", "0").split()[0]))
            except (ValueError, IndexError):
                pass
    if loads:
        w(f"Load average at the start of each run: median {statistics.median(loads):.1f}, max {max(loads):.1f}")
    w("")
    w("`*` after identical means equal only after replacing the numbers on timing/rate lines by N.")
    w("")
    verdict_order = ["identical", "identical*", "identical-fail", "identical-fail*", "nondeterministic", "different",
                     "timeout-vs-done", "both-timeout", "n/a"]
    for c in corpora:
        w(f"## {c}")
        w("")
        w("| pair | " + " | ".join(verdict_order) + " |")
        w("| --- |" + " ---: |" * len(verdict_order))
        for (xn, yn) in pairs:
            v = stats.get((c, f"{xn}:{yn}"), {})
            w(f"| {xn} vs {yn} | " + " | ".join(str(v.get(k, 0)) for k in verdict_order) + " |")
        w("")
        # status table of the baseline and the build under test
        sa = {}
        for (cc, p), cfgs in runs.items():
            if cc != c:
                continue
            for cfg, r in cfgs.items():
                sa.setdefault(cfg, {}).setdefault(r.status, 0)
                sa[cfg][r.status] += 1
        w("Exit status per configuration: " + "; ".join(
            f"{cfg}: " + ", ".join(f"{k} {n}" for k, n in sorted(st.items())) for cfg, st in sorted(sa.items())))
        w("")
        # coverage: how much was actually compared
        launches = {}
        tier_run = tier_all = 0
        for (cc, p), cfgs in runs.items():
            if cc != c:
                continue
            for cfg, r in cfgs.items():
                launches[cfg] = launches.get(cfg, 0) + len(r.records)
            bt = cfgs.get("Bt")
            if bt:
                for ln in bt.tier:
                    m = re.match(r"tier: (\d+) of (\d+) warp instructions", ln)
                    if m:
                        tier_run += int(m.group(1))
                        tier_all += int(m.group(2))
        w("Kernel launches recorded (VGPU_COUNTERS_FILE): " + ", ".join(f"{k} {v}" for k, v in sorted(launches.items())))
        if tier_all:
            w("")
            w(f"Bt: the tier ran {tier_run} of {tier_all} warp instructions ({100.0 * tier_run / tier_all:.1f}%) in the programs that printed tier statistics.")
        w("")
        # wall time (informational)
        tot = {}
        for (cc, p), cfgs in runs.items():
            if cc != c:
                continue
            done = [r for r in cfgs.values() if r.status == "ok"]
            if len(done) == len(cfgs):
                for cfg, r in cfgs.items():
                    tot[cfg] = tot.get(cfg, 0.0) + r.secs
        if tot:
            w("Wall seconds summed over programs that finished (rc 0) in every configuration (informational, host was shared): "
              + ", ".join(f"{k} {v:.0f}" for k, v in sorted(tot.items())))
            w("")
    # unstable programs
    w("## Programs that differ from themselves (A vs A2)")
    w("")
    any_u = False
    for (c, p), u in sorted(selfdiff.items()):
        if u:
            any_u = True
            w(f"- {c}/{p}: {', '.join(sorted(u))}")
    if not any_u:
        w("none")
    w("")
    # failures on both sides
    w("## Failed or timed out (exit status per configuration)")
    w("")
    w("| corpus/program | " + " | ".join(sorted({cfg for cf in runs.values() for cfg in cf})) + " |")
    cfg_names = sorted({cfg for cf in runs.values() for cfg in cf})
    w("| --- |" + " --- |" * len(cfg_names))
    for (c, p), cfgs in sorted(runs.items()):
        if all(r.status == "ok" for r in cfgs.values()):
            continue
        w(f"| {c}/{p} | " + " | ".join(
            (f"{cfgs[n].rc}" if n in cfgs else "-") for n in cfg_names) + " |")
    w("")
    w("## Every difference")
    w("")
    nd = 0
    for (c, p, pair, verdict, comps, det) in rows:
        if verdict in ("different", "nondeterministic", "timeout-vs-done"):
            nd += 1
            w(f"### {c}/{p}  {pair}  {verdict}  ({', '.join(comps)})")
            w("```")
            w("\n".join(det))
            w("```")
    if not nd:
        w("none")
    text = "\n".join(out) + "\n"
    with open(os.path.join(root, "report.md"), "w") as f:
        f.write(text)
    with open(os.path.join(root, "pairs.tsv"), "w") as f:
        f.write("corpus\tprogram\tpair\tverdict\tcomponents\n")
        for (c, p, pair, verdict, comps, det) in rows:
            f.write(f"{c}\t{p}\t{pair}\t{verdict}\t{','.join(comps)}\n")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
