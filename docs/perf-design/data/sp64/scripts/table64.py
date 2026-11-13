#!/usr/bin/env python3
"""table.py <sp64.py result>.json [--compact out.json]: round sp64's tables (round sp63's, with the base build's warm and second-run sides, the sides
bp and bq of sp64.py, and the changed build against the base build side by side: "N/B cold" is (Na/Nbc) / (Ba/Bbc), likewise warm and second run).
Round sp63's description follows: round sp63's tables.  Speed-ups are the default's median wall over the side's, each build against
its own default (Bbc against Ba, N* against Na); "default B/N" is Ba's median over Na's (the default path of the change, an A/A check).  Above 1 is faster.
--compact writes the file kept in the repository: per program the same-bits verdict, every run's wall by side, and the warm side's fill
[objects, bytes, compiles, compile ms, kept, declined]."""
import json, math, statistics, sys
NAME = {"Ba": "default B/N", "Bbc": "base JIT cold", "Bbw": "base JIT warm", "Bbs": "base JIT second run", "Nbc": "JIT cold", "Nbp": "JIT cold, header there",
        "Nbw": "JIT warm", "Nbs": "JIT second run", "Nbq": "JIT second run, header there"}
VS = {"N/B cold": ("Nbc", "Bbc"), "N/B warm": ("Nbw", "Bbw"), "N/B second run": ("Nbs", "Bbs")}
def gm(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
def key(x): return (x["out"], x["cnt"], x["fin"], x["rc"])
d = json.load(open(sys.argv[1])); skipped = {k: v.get("skipped") for k, v in d.items() if "sides" not in v}; d = {k: v for k, v in d.items() if "sides" in v}
sides = [s for s in NAME if any(s in v["sides"] for v in d.values())]
sides += [k for k, (n, b) in VS.items() if n in sides and b in sides]
for k in VS: NAME[k] = k
rows, comp = [], {}
def ratio(med, s):
    if s in VS:
        n, b = VS[s]; return (med["Na"] / med[n]) / (med["Ba"] / med[b])
    return med["Ba"] / med["Na"] if s == "Ba" else med[s[0] + "a"] / med[s]
for pid, v in d.items():
    S = v["sides"]; med = {s: statistics.median(x["wall"] for x in r) for s, r in S.items()}
    allr = [x for r in S.values() for x in r] + [x for r in v.get("pre", {}).values() for x in r] + list(v.get("fill", {}).values())
    ak = set(key(x) for s in ("Ba", "Na") if s in S for x in S[s]); det = len(ak) == 1
    diff = sorted(set(s for s, r in S.items() for x in r if key(x) not in ak) | set("pre-" + s for s, r in v.get("pre", {}).items() for x in r if key(x) not in ak) |
                  set("fill" for f in v.get("fill", {}).values() if key(f) not in ak))
    fw = v.get("fill", {}).get("Nbw", {}); st = fw.get("stats", {})
    rows.append(dict(pid=pid, med=med, det=det, diff=diff, n=v["runs"], objs=fw.get("objs", 0), cms=st.get("compile_ms", 0.0), objs_after=v.get("objs", {})))
    comp[pid] = dict(same="default differs from itself" if not det else ("DIFFERENT " + ",".join(diff) if diff else "same"),
                     walls={s: [round(x["wall"], 3) for x in r] for s, r in S.items()},
                     fill=[fw.get("objs"), fw.get("bytes"), st.get("compiles"), st.get("compile_ms"), st.get("accepted"), st.get("declined")] if fw else None,
                     objects_after={s: o for s, o in v.get("objs", {}).items() if any(o)})
if "--compact" in sys.argv: json.dump(comp, open(sys.argv[sys.argv.index("--compact") + 1], "w"), separators=(",", ":"))
print("| program | runs | default s | " + " | ".join(NAME[s] for s in sides) + " | objects (warm fill) | compile s | same bits |")
print("|---|---:|---:|" + "---:|" * len(sides) + "---:|---:|---|")
for r in sorted(rows, key=lambda r: r["pid"]):
    sb = ("same" if not r["diff"] else "DIFFERENT: " + ",".join(r["diff"])) if r["det"] else "default differs from itself"
    print("| %s | %d | %.3f | %s | %d | %.1f | %s |" % (r["pid"], r["n"], r["med"]["Na"], " | ".join("%.3f" % ratio(r["med"], s) for s in sides), r["objs"], r["cms"] / 1000, sb))
print()
print("| programs | n | " + " | ".join(NAME[s] for s in sides) + " |"); print("|---|---:|" + "---:|" * len(sides))
groups = [(c, [r for r in rows if r["pid"].startswith(c + "/")]) for c in ("polybench", "rodinia", "hecbench")] + [("all", rows),
          ("with objects", [r for r in rows if r["objs"]]), ("without", [r for r in rows if not r["objs"]]), ("0.2 s and longer", [r for r in rows if r["med"]["Na"] >= 0.2])]
for c, rs in groups: print("| %s | %d | %s |" % (c, len(rs), " | ".join("%.3f" % gm([ratio(r["med"], s) for r in rs]) for s in sides)))
print()
for s in sides:
    slow = sorted((ratio(r["med"], s), r["pid"]) for r in rows if ratio(r["med"], s) < 1 / 1.02)
    print("%s: %d of %d more than 2 %% slower%s" % (NAME[s], len(slow), len(rows), ": " + ", ".join("%s %.3f" % (p, x) for x, p in slow) if slow else ""))
s3 = sorted((round(ratio(r["med"], s), 3), r["pid"], s) for r in rows for s in VS if s in sides and r["med"]["Na"] >= 0.2 and ratio(r["med"], s) < 1 / 1.03)
print("the keep bar: programs of 0.2 s or more more than 3 % slower on the changed build than on the base build (cold, warm, second run, each against its own default):", s3)
print("same bits: %d programs, %d equal in every run of every side, default differs from itself: %s, DIFFERENT: %s" % (
    len(rows), sum(1 for r in rows if r["det"] and not r["diff"]), [r["pid"] for r in rows if not r["det"]], [(r["pid"], r["diff"]) for r in rows if r["det"] and r["diff"]]))
print("left out:", skipped)
print("objects in the cache after each timed run (cold and second-run sides):", {r["pid"]: {k: o for k, o in r["objs_after"].items() if any(o) and k != "Nbw"} for r in rows if any(any(o) for k, o in r["objs_after"].items() if k != "Nbw")})
