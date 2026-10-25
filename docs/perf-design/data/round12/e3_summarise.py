#!/usr/bin/env python3
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
nat = collections.defaultdict(dict)   # (build, side) -> workload -> rec
sim = collections.defaultdict(dict)   # (build, profile) -> workload -> rec
for r in rows:
    if r["side"] == "sim": sim[(r["build"], r["profile"])][r["workload"]] = r
    else: nat[(r["build"], r["side"])][r["workload"]] = r
def p(d): return sum(1 for r in d.values() if r["passed"]), len(d)
print("| build (SASS+PTX of) | native A10G, embedded SASS | native A10G, CUDA_FORCE_PTX_JIT=1 | first native failure |")
print("|---|---|---|---|")
for b in sorted({k[0] for k in nat}, key=lambda s: int(s[3:])):
    a, c = nat[(b, "native-sass")], nat[(b, "native-ptx")]
    f = next((r["tail"] for r in list(a.values()) + list(c.values()) if not r["passed"] and r["tail"]), [""])
    print("| %s | %d/%d PASS | %d/%d PASS | %s |" % (b, *p(a), *p(c), (f[0] if f else "")[:90]))
print()
print("| program build | simulated profile | sim engine | sim PASS | native A10G (JIT of PTX / SASS) can run it? | workloads where sim fails |")
print("|---|---|---|---|---|---|")
for (b, prof), d in sim.items():
    eng = collections.Counter(e for r in d.values() for e in r["engine_log"])
    eng = {e: n for e, n in eng.items()}
    n_s, n_p = p(nat[(b, "native-sass")]), p(nat[(b, "native-ptx")])
    fails = [w + (":" + r["tail"][0][:60] if r["tail"] else "") for w, r in d.items() if not r["passed"]]
    print("| %s | %s | %s | %d/%d | SASS %d/%d, PTX-JIT %d/%d | %s |" % (b, prof, ", ".join("%s x%d" % kv for kv in sorted(eng.items())) or "-", *p(d), *n_s, *n_p, "; ".join(fails)[:200] or "-"))
