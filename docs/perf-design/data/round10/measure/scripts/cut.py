import sys; sys.path.insert(0,'/tmp/claude-1002/an')
from lib import *
base='/tmp/claude-1002/data/ct'
cases=['tma_load','sm80_sparse','sm90_group_gemm','sm90_pingpong','sm90_sparse_f16','sm100_f16','sm100_f8','sm100_blockscaled','sm120_f4_f6','sm120_bs_mxf4','fmha1','fmha2','fmha3','fmha4']
names={'sm80_sparse':'gemm_sm80_sparse_f16'}
print('| case | sim GPU | wall off (s) | wall on (s) | on/off | RSS off / on MB | warp instr | in tier % | result |')
print('|---|---|---:|---:|---:|---:|---:|---:|---|')
for c in cases:
    n=names.get(c,c)
    d={0:[],1:[]}
    for t in (0,1):
        for f in sorted(glob.glob(f'{base}/{n}_r*_t{t}.time')):
            r=readtime(f)
            if r: d[t].append(r)
    if not d[0]: continue
    s=None
    for f in sorted(glob.glob(f'{base}/{n}_r*_t1.stats')):
        s=parse_stats(f); break
    w0=[x[0] for x in d[0]]; w1=[x[0] for x in d[1]]
    o0=open(sorted(glob.glob(f'{base}/{n}_r*_t0.out'))[0],errors='replace').read()
    res=re.findall(r'\[  PASSED  \] (\d+) tests?',o0) or (['%d [OK]'%o0.count(' [OK]')] if ' [OK]' in o0 else ['?'])
    gpu=''
    print(f"| {c} | | {med(w0):.1f} ({min(w0):.1f}-{max(w0):.1f}) | {med(w1):.1f} ({min(w1):.1f}-{max(w1):.1f}) | {med(w1)/med(w0):.3f} | {med([x[2] for x in d[0]]):.0f} / {med([x[2] for x in d[1]]):.0f} | {s['tot'] if s else 0:,} | {100*s['intier']/s['tot'] if s and s['tot'] else 0:.1f} | {res[0]} passed; n={len(w0)} |")
