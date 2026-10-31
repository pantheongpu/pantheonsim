import sys,re,collections
def sym(l):
    m=re.match(r'\s*[0-9a-f]+ (.*) \((.*)\)$',l)
    if not m: return None
    return re.sub(r'\(.*','',m.group(1))[:70]
blocks=[]; cur=[]
for l in sys.stdin:
    if l.strip()=='':
        if cur: blocks.append(cur); cur=[]
    else: cur.append(l)
if cur: blocks.append(cur)
N=len(blocks); callers=collections.Counter(); inpick=0; dec_only=0
for b in blocks:
    ch=[sym(l) for l in b[1:]]; ch=[c for c in ch if c]
    for i,c in enumerate(ch):
        if c.startswith('__snprintf_chk') or c.startswith('snprintf') or c.startswith('vsnprintf') or c.startswith('__vsnprintf'):
            j=i
            while j<len(ch) and (ch[j].startswith('__vsn') or ch[j].startswith('snprintf') or ch[j].startswith('__snprintf') or ch[j].startswith('vsnprintf') or ch[j].startswith('_IO') or ch[j].startswith('__printf') or ch[j].startswith('__GI') or ch[j].startswith('[unknown]') or ch[j].startswith('__mem') or ch[j].startswith('__strchrnul')): j+=1
            callers[ch[j] if j<len(ch) else '?']+=1; break
print('snprintf callers (share of samples):')
for k,n in callers.most_common(8): print(f'{100*n/N:6.2f}% {k}')
print('total with snprintf', 100*sum(callers.values())/N)
