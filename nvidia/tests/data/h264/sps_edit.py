#!/usr/bin/env python3
"""Rewrites the SPS of an Annex B H.264 stream: sps_edit.py in out key=value ...
keys: vui=none|full|notiming|norestr  level=N  refs=N  reorder=N  mdfb=N  timing=TICKS/SCALE"""
import sys

class BR:
    def __init__(s, b): s.b=b; s.p=0
    def u(s,n):
        v=0
        for _ in range(n):
            v=(v<<1)|((s.b[s.p>>3]>>(7-(s.p&7)))&1); s.p+=1
        return v
    def ue(s):
        z=0
        while s.u(1)==0: z+=1
        return (1<<z)-1+s.u(z)
    def se(s):
        k=s.ue(); return (k+1)//2 if k&1 else -(k//2)
class BW:
    def __init__(s): s.bits=[]
    def u(s,n,v):
        for i in range(n-1,-1,-1): s.bits.append((v>>i)&1)
    def ue(s,v):
        x=v+1; n=x.bit_length()-1
        s.u(n,0); s.u(n+1,x)
    def se(s,v): s.ue(2*v-1 if v>0 else -2*v)
    def bytes(s):
        b=s.bits+[1]
        while len(b)%8: b.append(0)
        out=bytearray()
        for i in range(0,len(b),8):
            v=0
            for k in range(8): v=(v<<1)|b[i+k]
            out.append(v)
        return bytes(out)

def unescape(d):
    out=bytearray(); z=0
    for x in d:
        if z>=2 and x==3: z=0; continue
        out.append(x); z=z+1 if x==0 else 0
    return bytes(out)
def escape(d):
    out=bytearray(); z=0
    for x in d:
        if z>=2 and x<=3: out.append(3); z=0
        out.append(x); z=z+1 if x==0 else 0
    return bytes(out)

def edit_sps(rbsp, opts):
    r=BR(rbsp); w=BW()
    prof=r.u(8); w.u(8,prof); cons=r.u(8); w.u(8,cons); lvl=r.u(8)
    if 'level' in opts: lvl=int(opts['level'])
    w.u(8,lvl)
    w.ue(r.ue())
    assert prof in (66,77,88,100)
    if prof==100:
        cf=r.ue(); w.ue(cf); assert cf==1
        w.ue(r.ue()); w.ue(r.ue()); w.u(1,r.u(1))
        sm=r.u(1); w.u(1,sm); assert sm==0
    w.ue(r.ue())              # log2_max_frame_num
    pt=r.ue(); w.ue(pt)
    if pt==0: w.ue(r.ue())
    elif pt==1: raise SystemExit('poc type 1 unsupported')
    refs=r.ue()
    if 'refs' in opts: refs=int(opts['refs'])
    w.ue(refs)
    w.u(1,r.u(1))
    w.ue(r.ue()); w.ue(r.ue())
    fmo=r.u(1); w.u(1,fmo)
    if not fmo: w.u(1,r.u(1))
    w.u(1,r.u(1))
    crop=r.u(1); w.u(1,crop)
    if crop:
        for _ in range(4): w.ue(r.ue())
    vui=r.u(1)
    mode=opts.get('vui','full')
    if not vui or mode=='none':
        w.u(1,0); return w.bytes()
    # parse the VUI that x264 writes
    asp=r.u(1); aspv=None
    if asp:
        idc=r.u(8); aspv=(idc,None)
        if idc==255: aspv=(idc,(r.u(16),r.u(16)))
    ovs=r.u(1)
    if ovs: r.u(1)
    vs=r.u(1); vsv=None
    if vs:
        vf=r.u(3); fr=r.u(1); cd=r.u(1); cdv=None
        if cd: cdv=(r.u(8),r.u(8),r.u(8))
        vsv=(vf,fr,cdv)
    cl=r.u(1)
    if cl: r.ue(); r.ue()
    ti=r.u(1); tiv=None
    if ti: tiv=(r.u(32),r.u(32),r.u(1))
    nh=r.u(1); vh=r.u(1); assert not nh and not vh
    ps=r.u(1)
    br=r.u(1); brv=None
    if br: brv=(r.u(1),r.ue(),r.ue(),r.ue(),r.ue(),r.ue(),r.ue())
    # emit
    w.u(1,1)
    w.u(1,1 if aspv else 0)
    if aspv:
        w.u(8,aspv[0])
        if aspv[1]: w.u(16,aspv[1][0]); w.u(16,aspv[1][1])
    w.u(1,0)
    w.u(1,1 if vsv else 0)
    if vsv:
        w.u(3,vsv[0]); w.u(1,vsv[1]); w.u(1,1 if vsv[2] else 0)
        if vsv[2]:
            for x in vsv[2]: w.u(8,x)
    w.u(1,0)
    if mode=='notiming' or ('timing' in opts and opts['timing']=='none'): ti=0
    elif 'timing' in opts:
        t,s=opts['timing'].split('/'); tiv=(int(t),int(s),1); ti=1
    w.u(1,1 if (ti and tiv) else 0)
    if ti and tiv: w.u(32,tiv[0]); w.u(32,tiv[1]); w.u(1,tiv[2])
    w.u(1,0); w.u(1,0)
    w.u(1,ps)
    if mode=='norestr': br=0
    if 'reorder' in opts or 'mdfb' in opts:
        br=1
        if brv is None: brv=(1,2,1,16,16,0,0)
        brv=list(brv)
        if 'reorder' in opts: brv[5]=int(opts['reorder'])
        if 'mdfb' in opts: brv[6]=int(opts['mdfb'])
    w.u(1,1 if (br and brv) else 0)
    if br and brv:
        w.u(1,brv[0])
        for x in brv[1:]: w.ue(x)
    return w.bytes()

def main():
    src,dst=sys.argv[1],sys.argv[2]
    opts=dict(a.split('=') for a in sys.argv[3:])
    d=open(src,'rb').read()
    # split into NALs
    out=bytearray(); i=0; starts=[]
    n=len(d)
    while i+3<=n:
        if d[i]==0 and d[i+1]==0 and d[i+2]==1:
            starts.append(i+3); i+=3
        else: i+=1
    for k,s in enumerate(starts):
        e=(starts[k+1]-3) if k+1<len(starts) else n
        while e>s and d[e-1]==0: e-=1
        nal=d[s:e]
        t=nal[0]&31
        if t==7:
            rb=unescape(nal[1:])
            # strip trailing: find rbsp trailing is handled by re-encode (reader ignores)
            new=edit_sps(rb,opts)
            nal=bytes([nal[0]])+escape(new)
        out+=b'\x00\x00\x00\x01'+nal
    open(dst,'wb').write(out)
main()
