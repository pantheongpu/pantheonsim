import sys,re,glob,os,collections
B='/tmp/claude-1002/data/soak'
pat=re.compile(r'^\s*[0-9.]+\s*$|updates per|done in|\d+ (October|September) 20|min=|avg=|time|elapsed|GB/s|GiB/s|MB/s|throughput|flops|bandwidth|\bms\b|\bus\b|\bsec|seconds|speed|perf|MElem|Mkeys|rate|duration|cycles|latency|clock',re.I)
def norm(p):
    try: s=open(p,errors='replace').read().split('\n')
    except Exception: return None
    return [l for l in s if not pat.search(l) and not l.startswith('[vgpu] running')]
def rd(p):
    try: return open(p).read().split()
    except Exception: return None
suites=sys.argv[1:] or ['polybench','rodinia','hecbench','cuda-samples']
for s in suites:
    d0=f'{B}/{s}_t0'; d1=f'{B}/{s}_t1'
    if not os.path.isdir(d0+'/logs'): continue
    names=sorted(os.path.basename(f)[:-4] for f in glob.glob(d0+'/logs/*.log'))
    same=0; diffs=[]; stat=0; rssr=[]; walls=[0,0]
    # statuses from results.tsv
    def tsv(d):
        r={}
        for l in open(d+'/results.tsv'):
            p=l.rstrip('\n').split('\t')
            r[p[0]]=(p[1],p[2],p[3] if len(p)>3 else '')
        return r
    r0=tsv(d0); r1=tsv(d1)
    for n in names:
        a=norm(f'{d0}/logs/{n}.log'); b=norm(f'{d1}/logs/{n}.log')
        if a is None or b is None: continue
        s0=r0.get(n); s1=r1.get(n)
        if a==b and s0 and s1 and s0[0]==s1[0] and (s0[2]==s1[2]): same+=1
        else: diffs.append((n,s0,s1,[l for l in a if l not in b][:2],[l for l in b if l not in a][:2]))
        x=rd(f'{d0}/rss/{n}'); y=rd(f'{d1}/rss/{n}')
        if x and y and len(x)==4 and len(y)==4: rssr.append((float(x[3])/1024,float(y[3])/1024,n))
    print(f'## {s}: {len(names)} programs, identical output and verdict: {same}, differ: {len(diffs)}')
    for n,s0,s1,la,lb in diffs: print('  DIFF',n,s0,s1,'|',[l[:80] for l in la],'|',[l[:80] for l in lb])
    if rssr:
        import statistics as st
        print('  peak RSS MB off/on: median %.0f / %.0f ; largest on-off: %s'%(st.median(x[0] for x in rssr),st.median(x[1] for x in rssr),max(rssr,key=lambda x:x[1]-x[0])))
        print('  total wall (sum of per-program) off %.0f s, on %.0f s'%(sum(float(rd(f'{d0}/rss/{n}')[0]) for _,_,n in rssr),sum(float(rd(f'{d1}/rss/{n}')[0]) for _,_,n in rssr)))
