import sys; sys.path.insert(0,'/tmp/claude-1002/an')
from lib import *
base='/tmp/claude-1002/data/pa'
def counters_total(p):
    n=0
    for l in open(p,errors='replace'):
        m=re.search(r'inst_executed=(\d+)',l)
        if m: n+=int(m.group(1))
    return n
ws=['int_virus','fp64_virus','compute_virus','mma_virus','omni_virus','tensor_virus','sfu_stress','scheduler_virus','transformer_virus','atomic_virus']
print('| workload | Minst/s off | Minst/s on | x | warp instr / 20 s (off, on) | in tier % | top fallbacks (share of all warp instr) |')
print('|---|---:|---:|---:|---|---:|---|')
agg=collections.defaultdict(lambda: collections.Counter())
for w in ws:
    r={0:[],1:[]}; ins={0:[],1:[]}; cov=[]; fb=collections.Counter(); ft=0
    for t in (0,1):
        for f in sorted(glob.glob(f'{base}/{w}_r*_t{t}.time')):
            b=f[:-5]; tm=readtime(f)
            if not tm: continue
            cpu=tm[1]
            n=int(open(b+'.insts').read().strip() or 0) if t==0 else None
            if t==1:
                s=parse_stats(b+'.stats'); n=s['tot'] if s else 0
                if s and s['tot']:
                    cov.append(100*s['intier']/s['tot'])
                    if not fb:
                        for c in (s['straight'],s['ends']):
                            fb.update(c)
                        ft=s['tot']
                        agg[w].update(s['straight']); agg[w].update(s['ends'])
            if n: r[t].append(n/cpu/1e6); ins[t].append(n)
    if not r[0]: continue
    top=', '.join(f'{k} {100*v/ft:.1f}%' for k,v in fb.most_common(3)) if ft else ''
    print(f"| {w} | {med(r[0]):.2f} ({min(r[0]):.2f}-{max(r[0]):.2f}) | {med(r[1]):.2f} ({min(r[1]):.2f}-{max(r[1]):.2f}) | {med(r[1])/med(r[0]):.2f} | {med(ins[0])/1e6:.1f} M, {med(ins[1])/1e6:.1f} M | {med(cov):.1f} | {top} |")
