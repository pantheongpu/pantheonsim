import sys,re,collections
def sym(l):
    m=re.match(r'\s*[0-9a-f]+ (.*) \((.*)\)$',l)
    if not m: return None
    return re.sub(r'\(.*','',m.group(1))[:80]+'|'+m.group(2).split('/')[-1]
blocks=[]; cur=[]
for l in sys.stdin:
    if l.strip()=='':
        if cur: blocks.append(cur); cur=[]
    else: cur.append(l)
if cur: blocks.append(cur)
keys=('vgpu::sass::launch','Device::run_kernel','lockstep_worker','extract_elf','extract_images','ZSTD_','libzstd','vgpu::sass::decode','vgpu::sass::unsupported','vgpu::sass::parse','pick_cubin','blas','gemm','Gemm','host_','vgpu::','PyImport_Import','import_find_and_load','_PyEval_EvalFrameDefault','Py_')
inn=collections.Counter(); dsos=collections.Counter(); n=0; depth=collections.Counter()
for b in blocks:
    ch=[sym(l) for l in b[1:]]; ch=[c for c in ch if c]
    if not ch: depth['empty chain']+=1; n+=1; continue
    if any(any(k in s for k in keys) for s in ch): continue
    n+=1; inn[ch[0]]+=1; dsos[ch[0].split('|')[-1]]+=1; depth[len(ch)<4]+=1
print('other samples',n, dict(depth))
for k,v in dsos.most_common(6): print('dso',v,k)
for k,v in inn.most_common(12): print(v,k)
