#!/usr/bin/env python3
"""How well the cubic-plus-exceptions model of amd_trans_table_fit.py does on one binade of v_exp_f32 operands (docs/numerics/math-functions.md, "AMD").
usage: amd_exp_binade_fit.py            (edit the three run() lines: a dump of the card's results, written by
  amd_trans_probe EXP dump <first operand, hex> 8388608 > ex.bin, and the bits of the 2^x binade's base: 0x3f800000 for results in [1, 2), 0x40000000 for [2, 4))
Prints the number of the 2^23 operands where the card is not the floor of the cubic of its 2^18-operand segment. Needs numpy. Our own code."""
import sys,numpy as np
N=1<<18
def run(fn,base):
    bits=np.fromfile(fn,dtype='<u4')[:1<<23].astype(np.int64); y=bits-base
    tot=0;used=0
    for j in (1,0):
        tot=0
        for s in range(32):
            f=np.arange(N); ys=y[s*N:(s+1)*N].astype(float); ref=ys[0]
            t=f/N; fq=((f>>j)<<j)/N
            A=np.stack([np.ones(N),t,fq**2,t**3],1)
            c,*_=np.linalg.lstsq(A,ys-ref+0.5,rcond=None)
            tot+=(np.floor(A@c)!=ys-ref).sum()
        print(fn,'j',j,'exceptions',tot,tot/(1<<23),flush=True)
run('ex_3f000000.bin',0x3f800000)   # x in [0.5,1): 2^x in [1.41,2)
run('ex_3e800000.bin',0x3f800000)
run('ex_3f800000.bin',0x40000000)   # x in [1,2): 2^x in [2,4) exp 128
