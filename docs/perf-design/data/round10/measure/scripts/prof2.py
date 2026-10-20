import sys,re,collections
# classify samples by chain (perf script innermost-first)
def sym(l):
    m=re.match(r'\s*[0-9a-f]+ (.*) \((.*)\)$',l)
    if not m: return None
    return re.sub(r'\(.*','',m.group(1))[:100]+'|'+m.group(2).split('/')[-1]
blocks=[]; cur=[]
for l in sys.stdin:
    if l.strip()=='':
        if cur: blocks.append(cur); cur=[]
    else: cur.append(l)
if cur: blocks.append(cur)
N=len(blocks)
alloc=collections.Counter(); cats=collections.Counter(); sub=collections.defaultdict(collections.Counter)
inner=collections.Counter()
def has(chain,*keys): return any(any(k in s for k in keys) for s in chain)
for b in blocks:
    ch=[sym(l) for l in b[1:]]; ch=[c for c in ch if c]
    comm=b[0].split()[0]
    if has(ch,'vgpu::sass::launch','Device::run_kernel','lockstep_worker'): c='SASS execution (launch/run_kernel)'
    elif has(ch,'extract_elf','extract_images','ZSTD_','libzstd'): c='fatbin decompress (zstd)'
    elif has(ch,'vgpu::sass::decode','vgpu::sass::unsupported','vgpu::sass::parse','float_text','signed_hex'): c='SASS decode/parse (incl. its malloc, snprintf)'
    elif has(ch,'pick_cubin'): c='pick_cubin, other'
    elif has(ch,'blas','gemm','Gemm','host_'): c='host-side library (cuBLAS host gemm etc)'
    elif has(ch,'vgpu::'): c='other simulator'
    elif has(ch,'PyImport_Import','import_find_and_load'): c='python import'
    elif has(ch,'_PyEval_EvalFrameDefault','Py_'): c='python/torch (host op code)'
    else: c='other (kernel/loader/unknown)'
    cats[c]+=1
    if not ch:
        cats['stack lost (python, libtorch, loader)']+=1; continue
    if re.search(r'^(malloc|cfree|free|operator new|operator delete|_int_malloc|_int_free|malloc_consolidate|__libc_calloc|__memset|__memcpy|asm_exc_page_fault|\[unknown\]\|libc)',ch[0]) or re.search(r'^(operator (new|delete)|malloc|cfree)',ch[1] if len(ch)>1 else ''): alloc[c]+=1
    if c=='pick_cubin, other':
        inner[tuple(s.split('|')[0][:60] for s in ch[:3])]+=1
print('samples',N)
for c,n in cats.most_common(): print(f'{100*n/N:6.2f}%  {n:5d}  {c}')
print('allocator-ish (innermost frame malloc/free/new/delete/page-fault/libc) within category:')
for c,n in alloc.most_common(): print(f'{100*n/N:6.2f}%  {c}')
for k,n in inner.most_common(0): print(n,k)
