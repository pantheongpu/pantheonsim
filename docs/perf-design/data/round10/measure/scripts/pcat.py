import sys,re,subprocess,collections,glob,os
perf=glob.glob('/usr/lib/linux-tools/*/perf')[0]
def cat(sym,dso):
    if dso.startswith('libcudart') or dso.startswith('libcuda'):
        if re.search(r'mma|gmma|tcgen05|wgmma|tc_rows|mx_float|smem_load|sparse_column|narrow_sum|f16_to_f32|wgmma_smem',sym): return 'tensor-core math (mma/gmma/tcgen05)'
        if re.search(r'Interpreter',sym): return 'PTX interpreter'
        if re.search(r'tier::|h_[a-z0-9_]+\(void\*|simd::|tier_run',sym): return 'tier handlers / SIMD kernels'
        if re.search(r'exec_tma|MemoryManager|mem_read|mem_write|exec_mem|BigReaderLock|memcount|run_allocation|store_run|resolve',sym): return 'memory system / TMA'
        if re.search(r'step_warp|Runner::|lockstep|execute|runs_instr',sym): return 'SASS scheduler / interpreter (non-tier)'
        if re.search(r'decode|parse|unsupported|Hashtable|Operand|snprintf|basic_string|float_text|fixed_table',sym): return 'SASS decode / load'
        return 'other simulator'
    if dso.startswith('libc.so') or dso.startswith('libstdc++') or dso.startswith('libm'): return 'libc / libm'
    if dso.startswith('[kernel'): return 'kernel'
    return 'test program (host reference, fills, checks)'
for f in sys.argv[1:]:
    out=subprocess.run(['sudo',perf,'report','-i',f,'--no-children','--sort','dso,sym','-g','none','--stdio','--percent-limit','0.02'],capture_output=True,text=True).stdout
    c=collections.Counter(); tot=0
    for l in out.split('\n'):
        m=re.match(r'\s*([0-9.]+)%\s+(\S+)\s+\[.\]\s+(.*)$',l)
        if not m: continue
        p=float(m.group(1)); c[cat(m.group(3),m.group(2))]+=p; tot+=p
    print(os.path.basename(f)[5:-5], ' | '.join(f'{k}: {v*100/tot:.1f}%' for k,v in c.most_common(6)))
