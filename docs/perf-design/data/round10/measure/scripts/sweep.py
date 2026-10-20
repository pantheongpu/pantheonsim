import sys; sys.path.insert(0,'/tmp/claude-1002/an')
from lib import *
base='/tmp/claude-1002/data/sw'
for tier in ('quick','full'):
    print('##',tier)
    print('| group | wall off (s) | wall on (s) | on/off | cpu off | cpu on | RSS off MB | RSS on MB | warp instr | in tier % | runs |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    groups=sorted({os.path.basename(f).split('_r')[0] for f in glob.glob(f'{base}/{tier}/*_t0.time')})
    TW0=TW1=0; allin=allt=0
    for g in groups:
        d={0:[],1:[]}
        for t in (0,1):
            for f in sorted(glob.glob(f'{base}/{tier}/{g}_r*_t{t}.time')):
                r=readtime(f)
                if r: d[t].append(r)
        w0=[x[0] for x in d[0]]; w1=[x[0] for x in d[1]]
        s=None
        for f in sorted(glob.glob(f'{base}/{tier}/{g}_r*_t1.stats')):
            s=parse_stats(f)
            if s and s['tot']: break
        TW0+=med(w0); TW1+=med(w1)
        if s: allin+=s['intier']; allt+=s['tot']
        print(f"| {g} | {med(w0):.2f} ({min(w0):.2f}-{max(w0):.2f}) | {med(w1):.2f} ({min(w1):.2f}-{max(w1):.2f}) | {med(w1)/med(w0):.3f} | {med([x[1] for x in d[0]]):.2f} | {med([x[1] for x in d[1]]):.2f} | {med([x[2] for x in d[0]]):.0f} | {med([x[2] for x in d[1]]):.0f} | {s['tot'] if s else 0:,} | {100*s['intier']/s['tot'] if s and s['tot'] else 0:.1f} | {len(w0)} |")
    print(f"| sum of medians | {TW0:.1f} | {TW1:.1f} | {TW1/TW0:.3f} | | | | | {allt:,} | {100*allin/allt:.1f} | |")
    # output diffs
    nd=collections.Counter(); real=[]
    for f in sorted(glob.glob(f'{base}/{tier}/*_t0.out')):
        b=f[:-7]; a=norm(f); c=norm(b+'_t1.out')
        if a!=c:
            la=a.split('\n'); lc=c.split('\n')
            dl=[x for x in la if x not in lc]+[x for x in lc if x not in la]
            if all(x.startswith('terminate called') for x in dl): nd['terminate-at-exit only']+=1
            else: nd['OTHER']+=1; real.append((os.path.basename(b),dl[:3]))
        else: nd['identical']+=1
    print(nd); print(real)
    ab=collections.Counter()
    for f in glob.glob(f'{base}/{tier}/*.time'):
        r=readtime(f)
        if r and r[3]: ab['t1' if f.endswith('_t1.time') else 't0']+=1
    print('aborts (signal 6)',dict(ab), 'of', len(glob.glob(f'{base}/{tier}/*_t0.time')),'per mode')
