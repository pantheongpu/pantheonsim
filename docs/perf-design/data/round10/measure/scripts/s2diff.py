import sys,re,glob,os,collections
B='/tmp/claude-1002/data/s2'
def norm(p):
    s=open(p,errors='replace').read()
    s=re.sub(r'\(\d+ ms( total)?\)','(N ms)',s)
    s=re.sub(r' +\(\d+\.\d+ s\)','',s)
    s=re.sub(r'terminate called without an active exception\n?','',s)
    return s
for label in sorted(os.listdir(B)):
    d=f'{B}/{label}'; res=collections.Counter(); bad=[]; lines=0
    for f in sorted(glob.glob(d+'/*.t0.out')):
        b=f[:-7]; n=os.path.basename(b)
        if not (os.path.exists(b+'.t1.rc') and os.path.exists(b+'.t0.rc')): continue
        a=norm(f); c=norm(b+'.t1.out'); lines+=a.count('\n')
        ra=open(b+'.t0.rc').read().strip(); rc=open(b+'.t1.rc').read().strip()
        ra=ra if ra!='134' else '0'; rc=rc if rc!='134' else '0'
        if a==c and ra==rc: res['identical']+=1
        else:
            res['DIFF']+=1
            la=a.split('\n'); lc=c.split('\n')
            bad.append((n,ra,rc,[x[:100] for x in la if x not in lc][:2],[x[:100] for x in lc if x not in la][:2]))
    print(label,dict(res),'total lines compared',lines)
    for x in bad: print('  DIFF',x)
