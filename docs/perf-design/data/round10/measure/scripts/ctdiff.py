import sys,re,glob,os,collections
B='/tmp/claude-1002/data/ct'
def norm(p):
    s=open(p,errors='replace').read()
    s=re.sub(r'\(\d+ ms( total)?\)','(N ms)',s)
    s=re.sub(r'\d+(\.\d+)? (ms|TFLOPS/s|GFLOPS)','N',s)
    return s
res=collections.Counter(); bad=[]
for f in sorted(glob.glob(B+'/*_t0.out')):
    b=f[:-7]
    if 'perf_' in b or 'probe' in b: continue
    if not os.path.exists(b+'_t0.rc') or not os.path.exists(b+'_t1.rc'): continue
    a=norm(f); c=norm(b+'_t1.out')
    ra=open(b+'_t0.rc').read().strip(); rc=open(b+'_t1.rc').read().strip()
    if a==c and ra==rc: res['identical']+=1
    else: res['DIFF']+=1; bad.append((os.path.basename(b),ra,rc))
print(dict(res)); print(bad)
