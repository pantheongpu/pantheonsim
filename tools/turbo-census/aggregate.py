#!/usr/bin/env python3
"""Aggregate the output of turbo_census into the ranked refusal tables.

usage: aggregate.py CENSUS.tsv [--timing-dir docs/perf-design/data] [--out DIR] [--markdown FILE]

Input: the tab-separated lines of turbo_census (H header, M module, K kernel). A program is a binary; its name is corpus/name
(hecbench/X, polybench/X, rodinia/X, samples/X, or the file name for any other path).
Weights: simulator seconds per program from the markdown tables under --timing-dir (columns "off s", "sim s", "simulator s",
"all sim s" of rows named polybench/ rodinia/ hecbench/ samples/); the median over the tables that have the program. A program with
no timing counts in kernels and programs but contributes no seconds (the tables say how many programs have one).
"""
import argparse, collections, glob, os, re, statistics, sys

TIMING_COLS = {"off s", "sim s", "simulator s", "all sim s"}


def program_name(path):
    p = path.split("/")
    if "HeCBench" in p:
        return "hecbench/" + p[p.index("src") + 1] if "src" in p else "hecbench/" + p[-1]
    if "polybenchGpu" in p:
        return "polybench/" + p[-1]
    if "gpu-rodinia" in p:
        return "rodinia/" + p[-1]
    if "Samples" in p:
        return "samples/" + p[-1]
    return p[-1]


def load_timing(root):
    t = collections.defaultdict(list)
    for f in glob.glob(os.path.join(root, "**/*.md"), recursive=True):
        hdr = None
        for line in open(f, errors="replace"):
            if not line.startswith("|"):
                hdr = None
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if hdr is None:
                hdr = [x.lower() for x in c]
                continue
            if set(line.strip()) <= set("|-: "):
                continue
            m = re.match(r"^(polybench|rodinia|hecbench|samples)/(\S+)$", c[0])
            if not m:
                continue
            for i, h in enumerate(hdr):
                if h in TIMING_COLS and i < len(c):
                    try:
                        t[c[0]].append(float(c[i].replace("s", "").strip()))
                    except ValueError:
                        pass
    return {k: statistics.median(v) for k, v in t.items() if v}


def reason_key(b):
    """A blocker as the ranked table names it: never:tma/UTMALDG -> never:tma; ptx:retarget-form(x) -> ptx:retarget-form."""
    b = b.split("/")[0]
    return re.sub(r"\(.*\)", "", b)


def parse(path):
    cells, modules, kernels = [], {}, []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if f[0] == "H":
            cells = f[2:]
        elif f[0] == "M":
            modules[(f[1], f[2])] = f[3:]
        elif f[0] == "K":
            kernels.append(f)
    return cells, modules, kernels


def cell_of(x):
    img, v = x.split(":", 1)
    if v.startswith("ok"):
        return img, True, v.endswith("unlisted"), []
    return img, False, False, v.split(",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("--timing-dir", default="docs/perf-design/data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--program-cells", default=None, help="comma list of cells for programs.tsv (default: all)")
    a = ap.parse_args()
    cells, modules, kernels = parse(a.census)
    timing = load_timing(a.timing_dir)
    progs = collections.defaultdict(lambda: collections.defaultdict(list))   # program -> cell -> [(image, ok, unlisted, blockers, kernel, module)]
    for k in kernels:
        prog = program_name(k[1])
        for ci, name in enumerate(cells):
            progs[prog][name].append(cell_of(k[4 + ci]) + (k[3], k[2]))
    allprogs = sorted(progs)
    timed = [p for p in allprogs if p in timing]
    out = []
    w = out.append
    sumw = sum(timing[p] for p in timed)
    w("programs %d, kernels %d, programs with a simulator time %d (%.0f s in all)" % (len(allprogs), len(kernels), len(timed), sumw))
    corp = collections.Counter(p.split("/")[0] for p in allprogs)
    w("per corpus (programs): " + ", ".join("%s %d" % kv for kv in sorted(corp.items())))
    tables = {}
    for name in cells:
        total = len(kernels)
        ok = sum(1 for p in allprogs for r in progs[p][name] if r[1])
        unl = sum(1 for p in allprogs for r in progs[p][name] if r[1] and r[2])
        img = collections.Counter(r[0] for p in allprogs for r in progs[p][name] if r[1])
        prog_all_ok = sum(1 for p in allprogs if all(r[1] for r in progs[p][name]))
        timed_all_ok = sum(timing[p] for p in timed if all(r[1] for r in progs[p][name]))
        # reasons: a kernel counts once for each distinct reason it has; `first` is the runtime's own (the first tried)
        rs = collections.defaultdict(lambda: {"first": 0, "any": 0, "sole": 0, "progs": set(), "sole_progs": set(), "sec": 0.0, "detail": collections.Counter()})
        for p in allprogs:
            prog_reasons = set()
            prog_refused = []
            for r in progs[p][name]:
                if r[1]:
                    continue
                keys = []
                for b in r[3]:
                    key = reason_key(b)
                    if key.startswith("ptx:") and r[0] in ("retarget", "ptx-jit"):
                        key += " [" + r[0] + "]"
                    if key not in keys:
                        keys.append(key)
                prog_refused.append(set(keys))
                for i, key in enumerate(keys):
                    d = rs[key]
                    d["any"] += 1
                    d["first"] += i == 0
                    d["sole"] += len(keys) == 1
                    d["progs"].add(p)
                    prog_reasons.add(key)
                for b in r[3]:   # the part after "/": the mnemonic or directive the scan met
                    if "/" in b:
                        rs[reason_key(b) + (" [" + r[0] + "]" if reason_key(b).startswith("ptx:") and r[0] in ("retarget", "ptx-jit") else "")]["detail"][b.split("/", 1)[1]] += 1
            only = set.union(*prog_refused) if prog_refused else set()
            if len(only) == 1:
                rs[next(iter(only))]["sole_progs"].add(p)
        for key, d in rs.items():
            d["sec"] = sum(timing[p] for p in d["progs"] if p in timing)
            d["sole_sec"] = sum(timing[p] for p in d["sole_progs"] if p in timing)
        tables[name] = dict(total=total, ok=ok, unl=unl, img=img, prog_all_ok=prog_all_ok, timed_all_ok=timed_all_ok, rs=rs)
    # the plain-text cell table
    w("")
    w("cell | kernels admitted | of which only because class unlisted | exact / lower-minor / ptx-jit / retarget | programs fully admitted | timed programs fully admitted (s of %.0f)" % sumw)
    for name in cells:
        t = tables[name]
        w("%s | %d of %d (%.1f%%) | %d | %d / %d / %d / %d | %d of %d | %.0f" % (name, t["ok"], t["total"], 100.0 * t["ok"] / t["total"], t["unl"], t["img"]["exact"], t["img"]["lower-minor"], t["img"]["ptx-jit"], t["img"]["retarget"], t["prog_all_ok"], len(allprogs), t["timed_all_ok"]))
    ranked = {}
    for name in cells:
        t = tables[name]
        rows = sorted(t["rs"].items(), key=lambda kv: (-kv[1]["sec"], -kv[1]["any"]))
        ranked[name] = rows
        w("")
        w("== %s: refusal reasons (kernels: any / first / sole; programs; share of kernels; timed s at stake: programs with the reason / programs whose only reason it is)" % name)
        for key, d in rows:
            top = sorted(d["progs"], key=lambda p: -sum(1 for r in progs[p][name] if not r[1] and any(reason_key(b) == key.split(" [")[0] for b in r[3])))[:5]
            w("%-40s k %4d / %4d / %4d  p %3d  %.2f%%  %.1f s / %.1f s   top: %s%s" % (key, d["any"], d["first"], d["sole"], len(d["progs"]), 100.0 * d["any"] / t["total"], d["sec"], d["sole_sec"], ", ".join(top), ("   detail: " + ", ".join("%s %d" % kv for kv in d["detail"].most_common(4))) if d["detail"] else ""))
    text = "\n".join(out)
    if a.markdown:
        open(a.markdown, "w").write(text + "\n")
    else:
        print(text)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "cells.tsv"), "w") as f:
            f.write("cell\tkernels\tadmitted\tunlisted_only\texact\tlower_minor\tptx_jit\tretarget\tprograms\tprograms_fully_admitted\ttimed_programs\ttimed_seconds_fully_admitted\n")
            for name in cells:
                t = tables[name]
                f.write("\t".join(map(str, [name, t["total"], t["ok"], t["unl"], t["img"]["exact"], t["img"]["lower-minor"], t["img"]["ptx-jit"], t["img"]["retarget"], len(allprogs), t["prog_all_ok"], len(timed), "%.1f" % t["timed_all_ok"]])) + "\n")
        with open(os.path.join(a.out, "reasons.tsv"), "w") as f:
            f.write("cell\treason\tkernels_any\tkernels_first\tkernels_sole\tprograms\tprograms_sole\ttimed_seconds_with_reason\ttimed_seconds_sole\ttop_programs\tdetail\n")
            for name in cells:
                for key, d in ranked[name]:
                    top = sorted(d["progs"], key=lambda p: -sum(1 for r in progs[p][name] if not r[1] and any(reason_key(b) == key.split(" [")[0] for b in r[3])))[:5]
                    f.write("\t".join(map(str, [name, key, d["any"], d["first"], d["sole"], len(d["progs"]), len(d["sole_progs"]), "%.1f" % d["sec"], "%.1f" % d["sole_sec"], ";".join(top), ";".join("%s=%d" % kv for kv in d["detail"].most_common(5))])) + "\n")
        with open(os.path.join(a.out, "programs.tsv"), "w") as f:
            pcells = a.program_cells.split(",") if a.program_cells else cells
            f.write("program\tsim_seconds\tkernels\t" + "\t".join(pcells) + "\n")
            for p in allprogs:
                row = [p, "%.2f" % timing[p] if p in timing else "-", str(len(progs[p][cells[0]]))]
                for name in pcells:
                    rs_ = progs[p][name]
                    n_ok = sum(1 for r in rs_ if r[1])
                    if n_ok == len(rs_):
                        row.append("all")
                    else:
                        c = collections.Counter(reason_key(r[3][0]) for r in rs_ if not r[1])
                        row.append("%d ok;" % n_ok + ";".join("%s=%d" % kv for kv in sorted(c.items())))
                f.write("\t".join(row) + "\n")


if __name__ == "__main__":
    main()
