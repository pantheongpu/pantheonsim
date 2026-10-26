#!/usr/bin/env python3
"""E4(a): aggregate the ltrace -c output of corpus.py census runs (native runs, calls at the program's own PLT: programs' direct
calls into libcudart / libcuda / the CUDA libraries). usage: census.py <rundir>/census > census.md"""
import sys, json, glob, re, collections
calls, progs, per = collections.Counter(), collections.defaultdict(set), {}
n = 0
for p in sorted(glob.glob(sys.argv[1] + "/*.json")):
    r = json.load(open(p)); n += 1
    got = False
    for l in r.get("ltrace", "").splitlines():
        m = re.match(r"\s*[\d.]+\s+[\d.]+\s+\d+\s+(\d+)\s+(\S+)\s*$", l)
        if m:
            c, f = int(m[1]), m[2]
            calls[f] += c; progs[f].add(r["corpus"] + "/" + r["case"]); got = True
    per[r["corpus"] + "/" + r["case"]] = (got, r["rc"], r["timeout"])
def fam(f):
    if f.startswith("__cuda"): return "runtime (compiler-generated registration/launch glue)"
    if f.startswith("cuda") and f[4:5].isupper() and not f.startswith(("cuda_", "cudaq")): return "CUDA runtime API"
    for pre, name in (("cublasLt", "cuBLASLt"), ("cublas", "cuBLAS"), ("cufft", "cuFFT"), ("curand", "cuRAND"), ("cusparse", "cuSPARSE"), ("cusolver", "cuSOLVER"),
                      ("cudnn", "cuDNN"), ("nvml", "NVML"), ("nvrtc", "NVRTC"), ("nccl", "NCCL"), ("npp", "NPP"), ("nvjpeg", "nvJPEG")):
        if f.startswith(pre): return name
    if f.startswith("cu") and f[2:3].isupper(): return "CUDA driver API"
    return "other"
bf = collections.defaultdict(list)
for f in calls: bf[fam(f)].append(f)
print("Programs traced: %d; programs with at least one traced call: %d; no ltrace output (timeout, static CUDA runtime, or failed): %d\n" %
      (n, sum(1 for v in per.values() if v[0]), sum(1 for v in per.values() if not v[0])))
print("| API family | distinct functions | total calls | programs using it |")
print("|---|---:|---:|---:|")
for k, fs in sorted(bf.items(), key=lambda kv: -sum(calls[f] for f in kv[1])):
    print("| %s | %d | %d | %d |" % (k, len(fs), sum(calls[f] for f in fs), len(set().union(*[progs[f] for f in fs]))))
for k in ("CUDA runtime API", "CUDA driver API", "runtime (compiler-generated registration/launch glue)"):
    print("\n### %s: every function seen (calls over all programs / programs using it)\n" % k)
    print(", ".join("`%s` %d/%d" % (f, calls[f], len(progs[f])) for f in sorted(bf[k], key=lambda f: -calls[f])))
for k in sorted(bf):
    if k in ("CUDA runtime API", "CUDA driver API", "runtime (compiler-generated registration/launch glue)", "other"): continue
    print("\n### %s\n" % k)
    print(", ".join("`%s` %d/%d" % (f, calls[f], len(progs[f])) for f in sorted(bf[k], key=lambda f: -calls[f])))
print("\n### other symbols matched by the pattern (libc / thrust etc.)\n")
print(", ".join("`%s` %d/%d" % (f, calls[f], len(progs[f])) for f in sorted(bf["other"], key=lambda f: -calls[f])[:40]))
