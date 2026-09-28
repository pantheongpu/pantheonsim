# Generates ptx_sweep.cu (python3 ptx_sweep_gen.py, in this directory): the
# everyday PTX instructions -- integer and bit operations, f32 and f64 with
# every rounding mode and .ftz, comparisons, conversions between every
# integer and float width, half precision -- each over the same 256 operand
# triples. The .approx forms are left out: they follow documented error
# bounds, not the card's bits.
import os, sys
V=[]
def add(t,a): V.append((t,a))
F='{ .reg .f32 a,b,c; mov.b32 a, %1; mov.b32 b, %2; mov.b32 c, %3; OP; mov.b32 %0, a; }'
FP='{ .reg .f32 a,b,c; .reg .pred p; mov.b32 a, %1; mov.b32 b, %2; mov.b32 c, %3; OP; selp.u32 %0, 1, 0, p; }'
D='{ .reg .f64 a,b,c; mov.b64 a, {%1, %2}; mov.b64 b, {%2, %3}; mov.b64 c, {%3, %1}; OP; mov.b64 {%0, _}, a; }'
DH='{ .reg .f64 a,b,c; mov.b64 a, {%1, %2}; mov.b64 b, {%2, %3}; mov.b64 c, {%3, %1}; OP; mov.b64 {_, %0}, a; }'
L='{ .reg .b64 a,b,c; mov.b64 a, {%1, %2}; mov.b64 b, {%2, %3}; mov.b64 c, {%3, %1}; OP; mov.b64 {%0, _}, a; }'
LH='{ .reg .b64 a,b,c; mov.b64 a, {%1, %2}; mov.b64 b, {%2, %3}; mov.b64 c, {%3, %1}; OP; mov.b64 {_, %0}, a; }'
# integer
for t in ('u32','s32'):
    for op in ('div','rem'):
        add(op+'.'+t, '%s.%s %%0, %%1, %%2;'%(op,t))
    for op in ('min','max'): add(op+'.'+t, '%s.%s %%0, %%1, %%2;'%(op,t))
    add('mul.hi.'+t, 'mul.hi.%s %%0, %%1, %%2;'%t)
    add('mad.hi.'+t, 'mad.hi.%s %%0, %%1, %%2, %%3;'%t)
    add('mad.lo.'+t, 'mad.lo.%s %%0, %%1, %%2, %%3;'%t)
    add('shr.'+t, 'shr.%s %%0, %%1, %%2;'%t)
    add('bfe.'+t, 'bfe.%s %%0, %%1, %%2, %%3;'%t)
    add('bfind.'+t, 'bfind.%s %%0, %%1;'%t)
    add('bfind.shiftamt.'+t, 'bfind.shiftamt.%s %%0, %%1;'%t)
    add('dp2a.lo.'+t+'.u32', 'dp2a.lo.%s.u32 %%0, %%1, %%2, %%3;'%t)
    add('dp2a.hi.'+t+'.s32', 'dp2a.hi.%s.s32 %%0, %%1, %%2, %%3;'%t)
    add('dp4a.'+t+'.s32', 'dp4a.%s.s32 %%0, %%1, %%2, %%3;'%t)
    add('dp4a.'+t+'.u32', 'dp4a.%s.u32 %%0, %%1, %%2, %%3;'%t)
    add('sad.'+t, 'sad.%s %%0, %%1, %%2, %%3;'%t)
add('shl.b32', 'shl.b32 %0, %1, %2;')
add('abs.s32', 'abs.s32 %0, %1;'); add('neg.s32', 'neg.s32 %0, %1;')
add('bfi.b32', 'bfi.b32 %0, %1, %2, %3, %1;')
add('brev.b32', 'brev.b32 %0, %1;'); add('popc.b32', 'popc.b32 %0, %1;'); add('clz.b32', 'clz.b32 %0, %1;')
add('fns.b32', 'fns.b32 %0, %1, %2, %3;')
add('bmsk.clamp', 'bmsk.clamp.b32 %0, %1, %2;'); add('bmsk.wrap', 'bmsk.wrap.b32 %0, %1, %2;')
for m in ('l.wrap','l.clamp','r.wrap','r.clamp'): add('shf.'+m, 'shf.%s.b32 %%0, %%1, %%2, %%3;'%m)
add('prmt', 'prmt.b32 %0, %1, %2, %3;')
add('lop3 0x96', 'lop3.b32 %0, %1, %2, %3, 0x96;'); add('lop3 0xe8', 'lop3.b32 %0, %1, %2, %3, 0xe8;')
add('add.cc/addc', '{ add.cc.u32 %0, %1, %2; addc.cc.u32 %0, %0, %3; addc.u32 %0, %0, 0; }')
add('sub.cc/subc', '{ sub.cc.u32 %0, %1, %2; subc.cc.u32 %0, %0, %3; subc.u32 %0, %0, 0; }')
add('mad.lo.cc/madc.hi', '{ .reg .u32 t; mad.lo.cc.u32 t, %1, %2, %3; madc.hi.u32 %0, %1, %2, t; }')
for t in ('s16','u16'):
    add('mul.wide.'+t, '{ .reg .%s x,y; cvt.%s.u32 x, %%1; cvt.%s.u32 y, %%2; mul.wide.%s %%0, x, y; }'%(t,t,t,t))
for op in ('div.s64','rem.s64','div.u64','rem.u64','mul.hi.s64','mul.hi.u64','min.s64','max.u64','shr.s64','shl.b64','popc.b64','clz.b64','brev.b64','bfind.s64'):
    if op.startswith(('popc','clz','bfind')):
        add(op, '{ .reg .b64 a; mov.b64 a, {%1, %2}; '+op+' %0, a; }')
    elif op.startswith(('shr','shl')):
        add(op+' lo', L.replace('OP', op+' a, a, %3'))
    elif op.startswith('brev'):
        add(op+' hi', LH.replace('OP', op+' a, a'))
    else:
        add(op+' lo', L.replace('OP', op+' a, a, b')); add(op+' hi', LH.replace('OP', op+' a, a, b'))
add('mul.wide.s32 hi', '{ .reg .s64 w; mul.wide.s32 w, %1, %2; mov.b64 {_, %0}, w; }')
add('mad.wide.u32 hi', '{ .reg .u64 w, c; mov.b64 c, {%3, %1}; mad.wide.u32 w, %1, %2, c; mov.b64 {_, %0}, w; }')
# f32 IEEE
for rnd in ('rn','rz','rm','rp'):
    for op in ('add','sub','mul'): add('%s.%s.f32'%(op,rnd), F.replace('OP','%s.%s.f32 a, a, b'%(op,rnd)))
    add('div.%s.f32'%rnd, F.replace('OP','div.%s.f32 a, a, b'%rnd))
    add('sqrt.%s.f32'%rnd, F.replace('OP','sqrt.%s.f32 a, a'%rnd))
    add('rcp.%s.f32'%rnd, F.replace('OP','rcp.%s.f32 a, a'%rnd))
    add('div.%s.ftz.f32'%rnd, F.replace('OP','div.%s.ftz.f32 a, a, b'%rnd))
    add('add.%s.ftz.f32'%rnd, F.replace('OP','add.%s.ftz.f32 a, a, b'%rnd))
    add('mul.%s.ftz.f32'%rnd, F.replace('OP','mul.%s.ftz.f32 a, a, b'%rnd))
    for op in ('add','mul','div','fma'):
        body = '%s.%s.f64 a, a, b%s'%(op, rnd, ', c' if op=='fma' else '')
        add('%s.%s.f64 lo'%(op,rnd), D.replace('OP',body)); add('%s.%s.f64 hi'%(op,rnd), DH.replace('OP',body))
    add('sqrt.%s.f64 hi'%rnd, DH.replace('OP','sqrt.%s.f64 a, a'%rnd))
    add('rcp.%s.f64 hi'%rnd, DH.replace('OP','rcp.%s.f64 a, a'%rnd))
add('min.f32', F.replace('OP','min.f32 a, a, b')); add('max.ftz.f32', F.replace('OP','max.ftz.f32 a, a, b'))
add('min.f64 hi', DH.replace('OP','min.f64 a, a, b')); add('max.f64 lo', D.replace('OP','max.f64 a, a, b'))
add('abs.f32', F.replace('OP','abs.f32 a, a')); add('neg.ftz.f32', F.replace('OP','neg.ftz.f32 a, a')); add('abs.ftz.f32', F.replace('OP','abs.ftz.f32 a, a'))
add('copysign.f32', F.replace('OP','copysign.f32 a, a, b'))
for tp in ('finite','infinite','number','notanumber','normal','subnormal'):
    add('testp.'+tp, FP.replace('OP','testp.%s.f32 p, a'%tp))
for cmp in ('eq','ne','lt','le','gt','ge','lo','ls','hi','hs','equ','neu','ltu','leu','gtu','geu','num','nan'):
    ty = 'u32' if cmp in ('lo','ls','hi','hs') else 'f32'
    add('setp.%s.%s'%(cmp,ty), '{ .reg .pred p; setp.%s.%s p, %%1, %%2; selp.u32 %%0, 1, 0, p; }'%(cmp,ty))
add('set.lt.f32.f32', '{ .reg .f32 a,b,d; mov.b32 a, %1; mov.b32 b, %2; set.lt.f32.f32 d, a, b; mov.b32 %0, d; }')
add('set.gtu.u32.f32', 'set.gtu.u32.f32 %0, %1, %2;')
add('selp.b32', '{ .reg .pred p; setp.ne.u32 p, %3, 0; selp.b32 %0, %1, %2, p; }')
add('slct.s32.f32', 'slct.s32.f32 %0, %1, %2, %3;'); add('slct.u32.s32', 'slct.u32.s32 %0, %1, %2, %3;')
# cvt
for dst in ('s32','u32','s16','u16','s8','u8','s64','u64'):
    for rnd in ('rni','rzi','rmi','rpi'):
        for sat in ('','.sat'):
            if dst in ('s64','u64'):
                add('cvt.%s%s.%s.f32 lo'%(rnd,sat,dst), '{ .reg .f32 a; .reg .b64 d; mov.b32 a, %%1; cvt.%s%s.%s.f32 d, a; mov.b64 {%%0, _}, d; }'%(rnd,sat,dst))
                add('cvt.%s%s.%s.f32 hi'%(rnd,sat,dst), '{ .reg .f32 a; .reg .b64 d; mov.b32 a, %%1; cvt.%s%s.%s.f32 d, a; mov.b64 {_, %%0}, d; }'%(rnd,sat,dst))
            elif dst in ('s32','u32'):
                add('cvt.%s%s.%s.f32'%(rnd,sat,dst), '{ .reg .f32 a; mov.b32 a, %%1; cvt.%s%s.%s.f32 %%0, a; }'%(rnd,sat,dst))
            else:
                add('cvt.%s%s.%s.f32'%(rnd,sat,dst), '{ .reg .f32 a; .reg .%s d; mov.b32 a, %%1; cvt.%s%s.%s.f32 d, a; cvt.u32.%s %%0, d; }'%(dst.replace('s8','s16').replace('u8','u16') if dst in('s8','u8') else dst, rnd,sat,dst, 'u16' if dst in ('s8','u8','s16','u16') else dst))
    if dst in ('s32','u32'): add('cvt.rzi.%s.f64'%dst, '{ .reg .f64 a; .reg .b64 d; mov.b64 a, {%%1, %%2}; cvt.rzi.%s.f64 %%0, a; }'%dst)
for src in ('s32','u32'):
    for rnd in ('rn','rz','rm','rp'):
        add('cvt.%s.f32.%s'%(rnd,src), '{ .reg .f32 d; cvt.%s.f32.%s d, %%1; mov.b32 %%0, d; }'%(rnd,src))
for src in ('s64','u64'):
    for rnd in ('rn','rz','rm','rp'):
        add('cvt.%s.f32.%s'%(rnd,src), '{ .reg .f32 d; .reg .b64 a; mov.b64 a, {%%1, %%2}; cvt.%s.f32.%s d, a; mov.b32 %%0, d; }'%(rnd,src))
        add('cvt.%s.f64.%s hi'%(rnd,src), '{ .reg .f64 d; .reg .b64 a; mov.b64 a, {%%1, %%2}; cvt.%s.f64.%s d, a; mov.b64 {_, %%0}, d; }'%(rnd,src))
for rnd in ('rn','rz','rm','rp'):
    add('cvt.%s.f32.f64'%rnd, '{ .reg .f64 a; .reg .f32 d; mov.b64 a, {%%1, %%2}; cvt.%s.f32.f64 d, a; mov.b32 %%0, d; }'%rnd)
    add('cvt.%s.ftz.f32.f64'%rnd, '{ .reg .f64 a; .reg .f32 d; mov.b64 a, {%%1, %%2}; cvt.%s.ftz.f32.f64 d, a; mov.b32 %%0, d; }'%rnd)
    add('cvt.%s.f16.f32'%rnd, '{ .reg .f32 a; .reg .b16 h; mov.b32 a, %%1; cvt.%s.f16.f32 h, a; mov.b32 %%0, {h, h}; }'%rnd)
    add('cvt.%s.f16.f64'%rnd, '{ .reg .f64 a; .reg .b16 h; mov.b64 a, {%%1, %%2}; cvt.%s.f16.f64 h, a; mov.b32 %%0, {h, h}; }'%rnd)
    add('cvt.%si.f32.f32'%rnd, F.replace('OP','cvt.%si.f32.f32 a, a'%rnd))
    add('cvt.%s.f16.s32'%rnd, '{ .reg .b16 h; cvt.%s.f16.s32 h, %%1; mov.b32 %%0, {h, h}; }'%rnd)
add('cvt.f32.f16', '{ .reg .b16 h; .reg .f32 d; mov.b32 {h, _}, %1; cvt.f32.f16 d, h; mov.b32 %0, d; }')
add('cvt.f64.f32 hi', '{ .reg .f32 a; .reg .f64 d; mov.b32 a, %1; cvt.f64.f32 d, a; mov.b64 {_, %0}, d; }')
add('cvt.ftz.f64.f32 hi', '{ .reg .f32 a; .reg .f64 d; mov.b32 a, %1; cvt.ftz.f64.f32 d, a; mov.b64 {_, %0}, d; }')
add('cvt.rzi.s32.f16', '{ .reg .b16 h; mov.b32 {h, _}, %1; cvt.rzi.s32.f16 %0, h; }')
add('cvt.rn.bf16.f32', '{ .reg .f32 a; .reg .b16 h; mov.b32 a, %1; cvt.rn.bf16.f32 h, a; mov.b32 %0, {h, h}; }')
add('cvt.rn.bf16x2.f32', '{ .reg .f32 a,b; mov.b32 a, %1; mov.b32 b, %2; cvt.rn.bf16x2.f32 %0, a, b; }')
add('cvt.sat.f32.f32', F.replace('OP','cvt.sat.f32.f32 a, a'))
add('cvt.ftz.f32.f32', F.replace('OP','cvt.ftz.f32.f32 a, a'))
for tn in (('s8','s32'),('u8','u32'),('s16','u32'),('u16','s32'),('s8','u32'),('u8','s8')):
    add('cvt.sat.%s.%s'%tn, '{ .reg .%s d; cvt.sat.%s.%s d, %%1; cvt.u32.%s %%0, d; }'%('s16' if tn[0]=='s8' else 'u16' if tn[0]=='u8' else tn[0], tn[0], tn[1], 'u16') if tn[1] in ('s32','u32') else '{ .reg .s8 x; .reg .u16 d; cvt.s8.u32 x, %1; cvt.sat.u8.s8 d, x; cvt.u32.u16 %0, d; }')
add('cvt.pack.sat.s16.s32', 'cvt.pack.sat.s16.s32 %0, %1, %2;')
add('cvt.pack.sat.u8.s32.b32', 'cvt.pack.sat.u8.s32.b32 %0, %1, %2, %3;')
# half
for op in ('add','sub','mul'):
    for t in ('f16x2',): add('%s.rn.%s'%(op,t), '%s.rn.%s %%0, %%1, %%2;'%(op,t))
add('fma.rn.f16x2', 'fma.rn.f16x2 %0, %1, %2, %3;'); add('fma.rn.bf16x2', 'fma.rn.bf16x2 %0, %1, %2, %3;')
for op in ('min','max'):
    for t in ('f16x2','bf16x2'): add('%s.%s'%(op,t), '%s.%s %%0, %%1, %%2;'%(op,t))
add('neg.f16x2', 'neg.f16x2 %0, %1;'); add('abs.f16x2', 'abs.f16x2 %0, %1;'); add('neg.bf16x2', 'neg.bf16x2 %0, %1;')
add('setp.lt.f16x2', '{ .reg .pred p, q; setp.lt.f16x2 p|q, %1, %2; selp.u32 %0, 1, 2, p; selp.u32 %0, %0, 4, q; }')
add('set.lt.u32.f16x2', 'set.lt.u32.f16x2 %0, %1, %2;')

for dst in ('s64','u64'):
    add('cvt.rzi.%s.f64 lo'%dst, '{ .reg .f64 a; .reg .b64 d; mov.b64 a, {%%1, %%2}; cvt.rzi.%s.f64 d, a; mov.b64 {%%0, _}, d; }'%dst)
    add('cvt.rzi.%s.f64 hi'%dst, '{ .reg .f64 a; .reg .b64 d; mov.b64 a, {%%1, %%2}; cvt.rzi.%s.f64 d, a; mov.b64 {_, %%0}, d; }'%dst)
for dst in ('s16','u16'):
    add('cvt.rzi.%s.f64'%dst, '{ .reg .f64 a; .reg .%s d; mov.b64 a, {%%1, %%2}; cvt.rzi.%s.f64 d, a; cvt.u32.u16 %%0, d; }'%(dst,dst))
add('cvt.rzi.u32.f16', '{ .reg .b16 h; mov.b32 {h, _}, %1; cvt.rzi.u32.f16 %0, h; }')
add('cvt.ftz.f64.f32 lo', '{ .reg .f32 a; .reg .f64 d; mov.b32 a, %1; cvt.ftz.f64.f32 d, a; mov.b64 {%0, _}, d; }')
add('cvt.f64.f32 lo', '{ .reg .f32 a; .reg .f64 d; mov.b32 a, %1; cvt.f64.f32 d, a; mov.b64 {%0, _}, d; }')
add('neg.f32', F.replace('OP','neg.f32 a, a'))
add('abs.f64 hi', DH.replace('OP','abs.f64 a, a')); add('neg.f64 hi', DH.replace('OP','neg.f64 a, a'))
add('abs.f64 lo', D.replace('OP','abs.f64 a, a'))
add('neg.f16', '{ .reg .b16 h; mov.b32 {h, _}, %1; neg.f16 h, h; mov.b32 %0, {h, h}; }')
add('abs.bf16', '{ .reg .b16 h; mov.b32 {h, _}, %1; abs.bf16 h, h; mov.b32 %0, {h, h}; }')
add('testp.normal.f64 zeros', '{ .reg .f64 a; .reg .pred p; mov.b64 a, {0, %2}; and.b64 a, a, 0x8000000000000000; testp.normal.f64 p, a; selp.u32 %0, 1, 0, p; }')
for tp in ('normal','subnormal','notanumber','finite'):
    add('testp.%s.f64'%tp, '{ .reg .f64 a; .reg .pred p; mov.b64 a, {%%1, %%2}; testp.%s.f64 p, a; selp.u32 %%0, 1, 0, p; }'%tp)
add('sqrt.approx.f32 NaN only', F.replace('OP','sqrt.rn.f32 a, a; testp.notanumber.f32 %%p, a').replace('OP','') if False else F.replace('OP','sqrt.rn.f32 a, a'))
add('rcp.rn.ftz.f32', F.replace('OP','rcp.rn.ftz.f32 a, a')); add('sqrt.rn.ftz.f32', F.replace('OP','sqrt.rn.ftz.f32 a, a'))
add('min.ftz.f32', F.replace('OP','min.ftz.f32 a, a, b')); add('sub.rn.ftz.f32', F.replace('OP','sub.rn.ftz.f32 a, a, b'))
add('rem.u64 by-zero-ish lo', L.replace('OP','rem.u64 a, a, %3') if False else L.replace('OP','rem.u64 a, a, c'))
add('div.s64 by-c lo', L.replace('OP','div.s64 a, a, c'))

# ex2/lg2 approx excluded (approximate); sqrt.approx excluded.
out = ['''// The everyday PTX instructions -- integer, bit, float with every rounding
// mode, conversions, comparisons, half precision -- each variant over the
// same 256 operand triples, its results hashed
// and compared with the card's (sm_86), recorded in ptx_sweep_expected.inc
// (`ptx_sweep --print` prints them; `--dump N` prints variant N's values).
// Generated by ptx_sweep_gen.py. Prints PASS on the last line, and runs the
// same on a GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "ptx_sweep_expected.inc"
};
constexpr int kN = 256;
__global__ void k(int v, const unsigned* in, unsigned* out) {
  const int i = threadIdx.x;
  const unsigned a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
  unsigned d = 0;
  switch (v) {''']
for n, (tag, asm) in enumerate(V):
    out.append('    case %d: asm volatile("%s" : "=r"(d) : "r"(a), "r"(b), "r"(c)); break;' % (n, asm.replace('"', '\\"')))
out.append('''  }
  out[i] = d;
}
static const char* kTags[] = {''')
for tag, asm in V:
    out.append('  "%s",' % tag)
out.append('''};
int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  const int dump = argc > 2 && std::strcmp(argv[1], "--dump") == 0 ? atoi(argv[2]) : -1;
  const int nv = sizeof kTags / sizeof kTags[0];
  // Random bits, and values that exercise the modifiers: f32 and packed-half
  // edges (+-0, +-1, NaN, subnormals, near-overflow, near-1, rounding ties).
  static const unsigned edge[] = {
      0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7fc00000u, 0xff800001u, 0x00000001u, 0x007fffffu,
      0x7f7fffffu, 0x3f7fffffu, 0x3f800001u, 0x3f801000u, 0x3dcccccdu, 0x40000000u, 0xc0400000u, 0x7f800000u,
      0x3c00bc00u, 0x7e003c00u, 0x00010002u, 0x7bff7c00u, 0x3bff3c01u, 0x80000000u, 0x3f80bf80u, 0x7fc0ff80u,
      0x12345678u, 0x9abcdef0u, 0x007fffffu, 0x7fffffffu, 0x00800000u, 0xffffffffu, 0x0f1e2d3cu, 0x00000003u};
  unsigned h[3 * kN];
  unsigned long long s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < 3 * kN; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    h[i] = static_cast<unsigned>(s);
    if ((s >> 40) % 3 == 0) h[i] = edge[(s >> 48) % 32];
  }
  unsigned *din, *dout;
  cudaMalloc(&din, sizeof h);
  cudaMalloc(&dout, kN * 4);
  cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);
  int bad = 0;
  for (int v = 0; v < nv; ++v) {
    k<<<1, kN>>>(v, din, dout);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("variant %d (%s) failed to run\\n", v, kTags[v]); return 1; }
    unsigned o[kN];
    cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    unsigned long long hash = 1469598103934665603ull;
    for (unsigned x : o) { hash ^= x; hash *= 1099511628211ull; }
    if (v == dump) {
      for (int i = 0; i < kN; ++i) printf("%3d a=%08x b=%08x c=%08x d=%08x\\n", i, h[3 * i], h[3 * i + 1], h[3 * i + 2], o[i]);
      return 0;
    }
    if (print) { printf("    {\\"%s\\", 0x%016llxull},\\n", kTags[v], hash); continue; }
    if (dump >= 0) continue;
    if (v >= (int)(sizeof kExpected / sizeof kExpected[0]) || hash != kExpected[v].hash) {
      if (bad++ < 40) printf("%s: 0x%016llx, the card gave 0x%016llx\\n", kTags[v], hash,
                             v < (int)(sizeof kExpected / sizeof kExpected[0]) ? kExpected[v].hash : 0ull);
    }
  }
  if (print) return 0;
  const int ne = sizeof kExpected / sizeof kExpected[0];
  printf("%d PTX sweep variants, %d differ\\n%s\\n", nv, bad, bad || ne != nv ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}''')
open(os.path.join(os.path.dirname(__file__) or '.', 'ptx_sweep.cu'), 'w').write('\n'.join(out) + '\n')
print(len(V), 'variants')
