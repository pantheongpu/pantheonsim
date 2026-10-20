import re, glob, os, statistics as st, collections
def readtime(p):
    try:
        lines=open(p).read().strip().split('\n')
        l=lines[-1].split()
        return float(l[0]), float(l[1])+float(l[2]), int(l[3])/1024.0, any('signal' in x or 'Command exited' in x for x in lines)
    except Exception: return None
def parse_stats(p):
    tot=0; intier=0; native=collections.Counter(); straight=collections.Counter(); ends=collections.Counter(); nproc=0
    if not os.path.exists(p): return None
    for line in open(p):
        m=re.match(r'tier: (\d+) of (\d+) warp',line)
        if m: intier+=int(m.group(1)); tot+=int(m.group(2)); nproc+=1; continue
        for key,c in (('tier native: ',native),('tier interpreter (straight): ',straight),('tier interpreter (ends a run): ',ends)):
            if line.startswith(key):
                for kv in line[len(key):].split():
                    k,_,v=kv.rpartition('=')
                    if k: c[k]+=int(v)
    return dict(tot=tot,intier=intier,native=native,straight=straight,ends=ends,nproc=nproc)
def norm(path):
    s=open(path,errors='replace').read()
    s=re.sub(r' +\(\d+\.\d+ s\)','',s)
    return s
def med(x): return st.median(x) if x else float('nan')
