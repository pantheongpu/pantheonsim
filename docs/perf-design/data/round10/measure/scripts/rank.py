import sys; sys.path.insert(0,'/tmp/claude-1002/an')
from lib import *
B='/tmp/claude-1002/data'
srcs=collections.OrderedDict()   # label -> parsed stats
def add(label,path):
    s=parse_stats(path)
    if s and s['tot']: srcs[label]=s
for g in sorted({os.path.basename(f).split('_r')[0] for f in glob.glob(f'{B}/sw/full/*_t1.stats')}):
    for f in sorted(glob.glob(f'{B}/sw/full/{g}_r*_t1.stats')):
        s=parse_stats(f)
        if s and s['tot']: srcs['torch:'+g]=s; break
for f in sorted(glob.glob(f'{B}/ct/*_r1_t1.stats')):
    add('cutlass:'+os.path.basename(f).split('_r1')[0],f)
for f in sorted(glob.glob(f'{B}/pa/*_r1_t1.stats')):
    add('pantheon:'+os.path.basename(f).split('_r1')[0],f)
for suite in ('polybench','rodinia','hecbench','cuda-samples'):
    for f in sorted(glob.glob(f'{B}/soak/{suite}_t1/stats/*.stats')):
        add(suite+':'+os.path.basename(f)[:-6],f)
tot=sum(s['tot'] for s in srcs.values()); intier=sum(s['intier'] for s in srcs.values())
print('sources',len(srcs),'warp instr',tot,'in tier %.1f%%'%(100*intier/tot))
fb=collections.Counter(); where=collections.defaultdict(collections.Counter); kind={}
for lab,s in srcs.items():
    for k,c in ((  'straight',s['straight']),('ends a run',s['ends'])):
        for m,v in c.items():
            fb[m]+=v; where[m][lab]+=v; kind[m]=k
print('fallback (interpreter) instructions listed:',sum(fb.values()),'= %.2f%% of all warp instr; all-interpreter = %.2f%%'%(100*sum(fb.values())/tot,100*(tot-intier)/tot))
print('| rank | mnemonic | kind | warp instr | % of all | % of interpreter share | main sources |')
print('|---:|---|---|---:|---:|---:|---|')
for i,(m,v) in enumerate(fb.most_common(int(sys.argv[1]) if len(sys.argv)>1 else 40),1):
    w=', '.join(f'{k} {100*x/v:.0f}%' for k,x in where[m].most_common(3))
    print(f'| {i} | {m} | {kind[m]} | {v:,} | {100*v/tot:.2f} | {100*v/(tot-intier):.1f} | {w} |')
# by family: strip modifiers
fam=collections.Counter()
for m,v in fb.items(): fam[m.split('.')[0]]+=v
print('\nby base opcode:')
for m,v in fam.most_common(25): print(f'{m} {v:,} {100*v/tot:.2f}%')
# macro average: each source counts equally
macro=collections.Counter(); nsrc=len(srcs)
cnt=collections.Counter()
for lab,s in srcs.items():
    for c in (s['straight'],s['ends']):
        for m,v in c.items():
            macro[m]+=v/s['tot']; 
            if v/s['tot']>0.001: cnt[m]+=1
print('\nmacro-averaged (each of %d sources counts equally):'%nsrc)
print('| rank | mnemonic | mean % of a source\'s warp instr | sources where it is >0.1% |')
print('|---:|---|---:|---:|')
for i,(m,v) in enumerate(macro.most_common(25),1): print(f'| {i} | {m} | {100*v/nsrc:.3f} | {cnt[m]} |')
# how complete are the lists
mx=max(len(s['straight']) for s in srcs.values()); mx2=max(len(s['ends']) for s in srcs.values())
print('max list lengths',mx,mx2)
cov=collections.Counter()
for lab,s in srcs.items():
    listed=sum(s['straight'].values())+sum(s['ends'].values()); real=s['tot']-s['intier']
    cov['listed']+=listed; cov['real']+=real
print('listed %d of %d interpreter instrs (%.1f%%)'%(cov['listed'],cov['real'],100*cov['listed']/max(1,cov['real'])))
