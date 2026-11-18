import os,re,sys,glob,json
R=sys.argv[1]
progs=sorted({d.rsplit('.',1)[0] for d in os.listdir(R)})
def st(d):
    try:
        s=open(f'{R}/{d}/status').read()
    except: return None,None
    rc=int(re.search(r'rc=(\d+)',s).group(1)); w=float(re.search(r'wall=([\d.]+)',s).group(1)); return rc,w
def err(d):
    try: return open(f'{R}/{d}/err.txt',errors='replace').read()
    except: return ''
def out(d):
    try: return open(f'{R}/{d}/out.txt',errors='replace').read()
    except: return ''
tim=re.compile(r'(tim|Tim|PERF|GF/s|time|Time|\bus\b|\(us\)|\bms\b|\(ms\)|elapsed|Elapsed|GB/s|GFLOP|Gflop|GFlop|throughput|Throughput|speed|[0-9]+\.[0-9]+ ?s\b|rate|bandwidth|Bandwidth)')
def keep(o): return [l for l in o.splitlines() if not tim.search(l)]
def num(e,k):
    m=re.search(k,e); return int(m.group(1)) if m else None
rows=[]
for p in progs:
    row={'prog':p}
    for f in ['native','sim','before','after','verify']:
        rc,w=st(f'{p}.{f}') if os.path.isdir(f'{R}/{p}.{f}') else (None,None)
        row[f+'_rc']=rc; row[f+'_wall']=w
    for f in ['before','after','verify']:
        e=err(f'{p}.{f}')
        row[f+'_card']=num(e,r'launches_card=(\d+)'); row[f+'_sim']=num(e,r'launches_sim=(\d+)')
    e=err(f'{p}.verify')
    row['v_words']=num(e,r'words_compared=(\d+)'); row['v_diff']=num(e,r'words_different=(\d+)'); row['v_out']=num(e,r'words_outside=(\d+)')
    a=keep(out(f'{p}.after')); s=keep(out(f'{p}.sim')); v=keep(out(f'{p}.verify'))
    row['after_eq_sim']=(a==s); row['verify_eq_sim']=(v==s)
    rows.append(row)
json.dump(rows,open(sys.argv[2],'w'),indent=1)
hdr='prog native sim before after(s) speedup after_card/sim verify(card,sim,words,diff,out) out_eq'
print(hdr)
for r in rows:
    sp=(r['sim_wall']/r['after_wall']) if r['sim_wall'] and r['after_wall'] else 0
    print(f"{r['prog']:26s} {r['native_wall'] or 0:6.2f} {r['sim_wall'] or 0:7.2f} {r['before_wall'] or 0:7.2f} {r['after_wall'] or 0:7.2f} {sp:5.2f}x  {r['after_card']}/{r['after_sim']}  rc={r['after_rc']}  v={r['verify_card']},{r['verify_sim']},{r['v_words']},{r['v_diff']},{r['v_out']} rc={r['verify_rc']}  eq={r['after_eq_sim']},{r['verify_eq_sim']}")

with open(sys.argv[3],'w') as f:
    f.write("program\tnative_s\tsim_s\tbefore_s\tafter_s\tafter_launches_card\tafter_launches_sim\tbefore_launches_card\tverify_launches_card\tverify_words\tverify_words_different\tverify_words_outside\toutput_same_as_sim\tnotes\n")
    for r in rows:
        notes=[]
        for k in ['native','sim','before','after','verify']:
            if r[k+'_rc'] not in (0,None): notes.append(k+':rc'+str(r[k+'_rc']))
        f.write("\t".join(str(x) if x is not None else "-" for x in [r['prog'].replace('-cuda',''), r['native_wall'], r['sim_wall'], r['before_wall'], r['after_wall'], r['after_card'], r['after_sim'], r['before_card'], r['verify_card'], r['v_words'], r['v_diff'], r['v_out'], (r['after_eq_sim'] if r['sim_rc']==0 and r['after_rc']==0 else '-'), ";".join(notes) or '-'])+"\n")
