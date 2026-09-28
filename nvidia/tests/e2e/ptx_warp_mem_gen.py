# Generates ptx_warp_mem.cu (python3 ptx_warp_mem_gen.py, in this directory):
# warp instructions (shfl in every mode and clamp, vote, match, redux, the
# lane masks, bar.red), atomics and reductions on global and shared memory in
# every operation and type (the value returned and the memory left), sub-word
# and vector loads and stores, and 16- and 64-bit integer arithmetic -- each
# over the same 256 threads' operands, eight warps.
import os
V=[]
def add(t,a): V.append((t,a))
# %0 = d (u32 out), %1,%2,%3 = a,b,c (u32), %4 = per-lane global scratch (u64, holds c on entry),
# %5 = the block's shared scratch base (u32 shared address, 256 words, word i holds c of lane i).
M = 0xffffffff
for mode in ('up','down','bfly','idx'):
    for clamp in ('0x1f','0x1c1f','0x0','0x101f'):
        add('shfl.%s c=%s'%(mode,clamp), '{ .reg .u32 l; and.b32 l, %%2, 31; shfl.sync.%s.b32 %%0, %%1, l, %s, 0xffffffff; }'%(mode,clamp))
    add('shfl.%s p'%mode, '{ .reg .pred p; .reg .u32 l; and.b32 l, %%2, 31; shfl.sync.%s.b32 %%0|p, %%1, l, 0x1f, 0xffffffff; @!p add.u32 %%0, %%0, 1; }'%mode)
    add('shfl.%s imm'%mode, 'shfl.sync.%s.b32 %%0, %%1, 5, 0x1f, 0xffffffff;'%mode)
for op in ('all','any','uni'):
    add('vote.%s'%op, '{ .reg .pred p, q; and.b32 %%0, %%1, 1; setp.ne.u32 p, %%0, 0; vote.sync.%s.pred q, p, 0xffffffff; selp.u32 %%0, 1, 0, q; }'%op)
    add('vote.%s mostly'%op, '{ .reg .pred p, q; setp.ne.u32 p, %%1, 0; vote.sync.%s.pred q, p, 0xffffffff; selp.u32 %%0, 1, 0, q; }'%op)
add('vote.ballot', '{ .reg .pred p; and.b32 %0, %1, 1; setp.ne.u32 p, %0, 0; vote.sync.ballot.b32 %0, p, 0xffffffff; }')
add('vote.ballot !p', '{ .reg .pred p; and.b32 %0, %1, 3; setp.ne.u32 p, %0, 0; vote.sync.ballot.b32 %0, !p, 0xffffffff; }')
add('match.any', '{ .reg .u32 x; and.b32 x, %1, 3; match.any.sync.b32 %0, x, 0xffffffff; }')
add('match.all', '{ .reg .u32 x; .reg .pred p; and.b32 x, %1, 1; match.all.sync.b32 %0|p, x, 0xffffffff; @p add.u32 %0, %0, 7; }')
add('match.any.b64', '{ .reg .b64 x; mov.b64 x, {%1, %2}; and.b64 x, x, 0x0000000100000001; match.any.sync.b64 %0, x, 0xffffffff; }')
for op,t in (('add','u32'),('add','s32'),('min','u32'),('min','s32'),('max','u32'),('max','s32'),('and','b32'),('or','b32'),('xor','b32')):
    add('redux.%s.%s'%(op,t), 'redux.sync.%s.%s %%0, %%1, 0xffffffff;'%(op,t))
for r in ('laneid','lanemask_eq','lanemask_lt','lanemask_le','lanemask_gt','lanemask_ge'):
    add('%'+r, 'mov.u32 %%0, %%%%%s;'%r if False else 'mov.u32 %0, %' + r + ';')
add('activemask', 'activemask.b32 %0;')
add('bar.red.popc', '{ .reg .pred p; and.b32 %0, %1, 1; setp.ne.u32 p, %0, 0; bar.red.popc.u32 %0, 0, p; }')
add('bar.red.and', '{ .reg .pred p, q; setp.ne.u32 p, %1, 0; bar.red.and.pred q, 0, p; selp.u32 %0, 1, 0, q; }')
add('bar.red.or', '{ .reg .pred p, q; and.b32 %0, %1, 0x100; setp.ne.u32 p, %0, 0; bar.red.or.pred q, 0, p; selp.u32 %0, 1, 0, q; }')
# atomics on the lane's own global word (returns old; memory checked too)
for op,t in (('add','u32'),('add','s32'),('min','u32'),('min','s32'),('max','u32'),('max','s32'),('inc','u32'),('dec','u32'),('and','b32'),('or','b32'),('xor','b32'),('exch','b32')):
    add('atom.global.%s.%s'%(op,t), 'atom.global.%s.%s %%0, [%%4], %%1;'%(op,t))
    add('red.global.%s.%s'%(op,t) if op!='exch' else 'red skip', ('red.global.%s.%s [%%4], %%1; mov.u32 %%0, 0;'%(op,t)) if op!='exch' else 'mov.u32 %0, 0;')
add('atom.global.cas.b32', 'atom.global.cas.b32 %0, [%4], %2, %1;')
add('atom.global.cas.b32 hit', '{ .reg .u32 o; ld.global.u32 o, [%4]; atom.global.cas.b32 %0, [%4], o, %1; }')
add('atom.global.add.f32', '{ .reg .f32 x, o; mov.b32 x, %1; atom.global.add.f32 o, [%4], x; mov.b32 %0, o; }')
add('atom.global.add.noftz.f16x2', 'atom.global.add.noftz.f16x2 %0, [%4], %1;')
add('atom.global.add.noftz.f16', '{ .reg .b16 x, o; mov.b32 {x, _}, %1; atom.global.add.noftz.f16 o, [%4], x; mov.b32 %0, {o, o}; }')
add('atom.global.add.u64 lo', '{ .reg .u64 x, o; mov.b64 x, {%1, %2}; atom.global.add.u64 o, [%4], x; mov.b64 {%0, _}, o; }')
add('atom.global.exch.b64 hi', '{ .reg .b64 x, o; mov.b64 x, {%1, %2}; atom.global.exch.b64 o, [%4], x; mov.b64 {_, %0}, o; }')
add('atom.global.max.s64 lo', '{ .reg .s64 x, o; mov.b64 x, {%1, %2}; atom.global.max.s64 o, [%4], x; mov.b64 {%0, _}, o; }')
add('atom.global.add.f64 hi', '{ .reg .f64 x, o; mov.b64 x, {%1, %2}; atom.global.add.f64 o, [%4], x; mov.b64 {_, %0}, o; }')
# shared-memory contention: every lane hits one word; the final value is order-independent
for op,t in (('add','u32'),('min','u32'),('max','s32'),('and','b32'),('or','b32'),('xor','b32')):
    add('atom.shared.%s.%s contended'%(op,t), '{ .reg .u32 o; atom.shared.%s.%s o, [%%5], %%1; bar.sync 0; ld.shared.u32 %%0, [%%5]; }'%(op,t))
add('atom.shared.inc.u32 contended', '{ .reg .u32 o; atom.shared.inc.u32 o, [%5], 1000; bar.sync 0; ld.shared.u32 %0, [%5]; }')
add('atom.shared.add.f32 lane', '{ .reg .f32 x, o; .reg .u32 ad; mov.b32 x, %1; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; atom.shared.add.f32 o, [ad], x; mov.b32 %0, o; }')
# sub-word and vector memory
add('ld.global.s8', '{ .reg .s16 x; st.global.u32 [%4], %1; ld.global.s8 x, [%4+1]; cvt.s32.s16 %0, x; }')
add('ld.global.u8', '{ .reg .u16 x; st.global.u32 [%4], %1; ld.global.u8 x, [%4+3]; cvt.u32.u16 %0, x; }')
add('ld.global.s16', '{ .reg .s16 x; st.global.u32 [%4], %1; ld.global.s16 x, [%4+2]; cvt.s32.s16 %0, x; }')
add('ld.global.s8 into s32', '{ st.global.u32 [%4], %1; ld.global.s8 %0, [%4+2]; }')
add('ld.global.u16 into u32', '{ st.global.u32 [%4], %1; ld.global.u16 %0, [%4+2]; }')
add('st.global.b8', '{ st.global.b8 [%4+2], %1; ld.global.u32 %0, [%4]; }')
add('st.global.b16', '{ st.global.b16 [%4+2], %1; ld.global.u32 %0, [%4]; }')
add('st.global.v2.b16', '{ .reg .b16 x, y; mov.b32 {x, y}, %1; st.global.v2.b16 [%4], {y, x}; ld.global.u32 %0, [%4]; }')
add('mov.b32 unpack b8', '{ .reg .b8 w, x, y, z; mov.b32 {w, x, y, z}, %1; mov.b32 %0, {z, w, y, x}; }') if False else None
# 16-bit and 64-bit integer arithmetic
for op in ('add','sub','min','max'):
    for t in ('u16','s16'):
        add('%s.%s'%(op,t), '{ .reg .%s x, y; cvt.%s.u32 x, %%1; cvt.%s.u32 y, %%2; %s.%s x, x, y; cvt.u32.%s %%0, x; }'%(t,t,t,op,t,'u16'))
for t in ('u16','s16'):
    add('mul.lo.%s'%t, '{ .reg .%s x, y; cvt.%s.u32 x, %%1; cvt.%s.u32 y, %%2; mul.lo.%s x, x, y; cvt.u32.u16 %%0, x; }'%(t,t,t,t))
    add('mul.hi.%s'%t, '{ .reg .%s x, y; cvt.%s.u32 x, %%1; cvt.%s.u32 y, %%2; mul.hi.%s x, x, y; cvt.u32.u16 %%0, x; }'%(t,t,t,t))
    add('div.%s'%t, '{ .reg .%s x, y; cvt.%s.u32 x, %%1; cvt.%s.u32 y, %%2; div.%s x, x, y; cvt.u32.u16 %%0, x; }'%(t,t,t,t))
    add('rem.%s'%t, '{ .reg .%s x, y; cvt.%s.u32 x, %%1; cvt.%s.u32 y, %%2; rem.%s x, x, y; cvt.u32.u16 %%0, x; }'%(t,t,t,t))
    add('shr.%s'%t, '{ .reg .%s x; cvt.%s.u32 x, %%1; shr.%s x, x, %%2; cvt.u32.u16 %%0, x; }'%(t,t,t))
    add('mad.lo.%s'%t, '{ .reg .%s x, y, z; cvt.%s.u32 x, %%1; cvt.%s.u32 y, %%2; cvt.%s.u32 z, %%3; mad.lo.%s x, x, y, z; cvt.u32.u16 %%0, x; }'%(t,t,t,t,t))
add('shl.b16', '{ .reg .b16 x; cvt.u16.u32 x, %1; shl.b16 x, x, %2; cvt.u32.u16 %0, x; }')
add('abs.s16', '{ .reg .s16 x; cvt.s16.u32 x, %1; abs.s16 x, x; cvt.u32.u16 %0, x; }')
add('neg.s16', '{ .reg .s16 x; cvt.s16.u32 x, %1; neg.s16 x, x; cvt.u32.u16 %0, x; }')
add('not.b16', '{ .reg .b16 x; cvt.u16.u32 x, %1; not.b16 x, x; cvt.u32.u16 %0, x; }')
add('cnot.b32', 'cnot.b32 %0, %1;')
add('cnot.b16', '{ .reg .b16 x; cvt.u16.u32 x, %1; and.b16 x, x, 3; cnot.b16 x, x; cvt.u32.u16 %0, x; }')
add('and.b16', '{ .reg .b16 x, y; cvt.u16.u32 x, %1; cvt.u16.u32 y, %2; and.b16 x, x, y; cvt.u32.u16 %0, x; }')
add('cvt.s8.s32 sat', '{ .reg .s16 x; cvt.sat.s8.s32 x, %1; cvt.u32.u16 %0, x; }') if False else None
for t in ('s64','u64'):
    add('add.%s hi'%t, '{ .reg .%s x, y; mov.b64 x, {%%1, %%2}; mov.b64 y, {%%3, %%1}; add.%s x, x, y; mov.b64 {_, %%0}, x; }'%(t,t))
    add('mul.lo.%s hi'%t, '{ .reg .%s x, y; mov.b64 x, {%%1, %%2}; mov.b64 y, {%%3, %%1}; mul.lo.%s x, x, y; mov.b64 {_, %%0}, x; }'%(t,t))
    add('mad.lo.%s hi'%t, '{ .reg .%s x, y; mov.b64 x, {%%1, %%2}; mov.b64 y, {%%3, %%1}; mad.lo.%s x, x, y, y; mov.b64 {_, %%0}, x; }'%(t,t))
    add('mad.hi.%s hi'%t, '{ .reg .%s x, y; mov.b64 x, {%%1, %%2}; mov.b64 y, {%%3, %%1}; mad.hi.%s x, x, y, y; mov.b64 {_, %%0}, x; }'%(t,t))
    add('shr.%s lo'%t, '{ .reg .%s x; mov.b64 x, {%%1, %%2}; shr.%s x, x, %%3; mov.b64 {%%0, _}, x; }'%(t,t))
    add('bfe.%s lo'%t, '{ .reg .%s x; mov.b64 x, {%%1, %%2}; bfe.%s x, x, %%3, %%1; mov.b64 {%%0, _}, x; }'%(t,t))
    add('min.%s hi'%t, '{ .reg .%s x, y; mov.b64 x, {%%1, %%2}; mov.b64 y, {%%3, %%1}; min.%s x, x, y; mov.b64 {_, %%0}, x; }'%(t,t))
add('bfi.b64 hi', '{ .reg .b64 x, y; mov.b64 x, {%1, %2}; mov.b64 y, {%3, %1}; bfi.b64 x, x, y, %3, %2; mov.b64 {_, %0}, x; }')
add('add.cc.u64/addc hi', '{ .reg .u64 x, y; mov.b64 x, {%1, %2}; mov.b64 y, {%3, %1}; add.cc.u64 x, x, y; addc.u32 %0, %3, 0; }')
add('mul.wide.u32 lo', '{ .reg .u64 w; mul.wide.u32 w, %1, %2; mov.b64 {%0, _}, w; }')
add('abs.s64 hi', '{ .reg .s64 x; mov.b64 x, {%1, %2}; abs.s64 x, x; mov.b64 {_, %0}, x; }')
add('neg.s64 hi', '{ .reg .s64 x; mov.b64 x, {%1, %2}; neg.s64 x, x; mov.b64 {_, %0}, x; }')
add('copysign.f64 hi', '{ .reg .f64 x, y; mov.b64 x, {%1, %2}; mov.b64 y, {%3, %1}; copysign.f64 x, x, y; mov.b64 {_, %0}, x; }')
add('selp.f64 hi', '{ .reg .f64 x, y; .reg .pred p; mov.b64 x, {%1, %2}; mov.b64 y, {%3, %1}; setp.ne.u32 p, %3, 0; selp.f64 x, x, y, p; mov.b64 {_, %0}, x; }')
add('slct.f32.f32', '{ .reg .f32 x, y, z; mov.b32 x, %1; mov.b32 y, %2; mov.b32 z, %3; slct.f32.f32 x, x, y, z; mov.b32 %0, x; }')
add('mov.b32 unpack/pack b16', '{ .reg .b16 x, y; mov.b32 {x, y}, %1; mov.b32 %0, {y, x}; }')
add('mov.b64 unpack/pack b32', '{ .reg .b64 w; .reg .b32 x, y; mov.b64 w, {%1, %2}; mov.b64 {x, y}, w; add.u32 %0, x, y; }')
V=[v for v in V if v and v[0]!='red skip']

out = ["""// Warp, atomic, memory and 16/64-bit integer PTX forms, each variant run by
// 256 threads (eight warps) over the same operand triples, with a word of
// global and of shared memory per thread; each variant's results and the
// memory it left are hashed and compared with an RTX 3060's (sm_86), recorded
// in ptx_warp_mem_expected.inc (`ptx_warp_mem --print` prints them; `--dump N`
// prints variant N's values). Generated by ptx_warp_mem_gen.py. Prints PASS on
// the last line, and runs the same on a GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "ptx_warp_mem_expected.inc"
};
constexpr int kN = 256;
__global__ void k(int v, const unsigned* in, unsigned* out, unsigned long long* g) {
  __shared__ unsigned sh[kN];
  const int i = threadIdx.x;
  const unsigned a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
  g[i] = c | ((unsigned long long)b << 32);
  sh[i] = c;
  __syncthreads();
  const unsigned sa = (unsigned)__cvta_generic_to_shared(sh);
  unsigned d = 0;
  switch (v) {"""]
for n, (tag, asm) in enumerate(V):
    out.append('    case %d: asm volatile("%s" : "=r"(d) : "r"(a), "r"(b), "r"(c), "l"(&g[i]), "r"(sa), "r"((unsigned)i) : "memory"); break;' % (n, asm.replace('"', '\\"')))
out.append("""  }
  out[2 * i] = d;
  out[2 * i + 1] = (unsigned)g[i] ^ (unsigned)(g[i] >> 32);
}
static const char* kTags[] = {""")
for tag, asm in V:
    out.append('  "%s",' % tag)
out.append("""};
int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  const int dump = argc > 2 && std::strcmp(argv[1], "--dump") == 0 ? atoi(argv[2]) : -1;
  const int nv = sizeof kTags / sizeof kTags[0];
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
  unsigned long long* dg;
  cudaMalloc(&din, sizeof h);
  cudaMalloc(&dout, 2 * kN * 4);
  cudaMalloc(&dg, kN * 8);
  cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);
  int bad = 0;
  for (int v = 0; v < nv; ++v) {
    k<<<1, kN>>>(v, din, dout, dg);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("variant %d (%s) failed to run\\n", v, kTags[v]); return 1; }
    unsigned o[2 * kN];
    cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    unsigned long long hash = 1469598103934665603ull;
    for (unsigned x : o) { hash ^= x; hash *= 1099511628211ull; }
    if (v == dump) {
      for (int i = 0; i < kN; ++i)
        printf("%3d a=%08x b=%08x c=%08x d=%08x m=%08x\\n", i, h[3 * i], h[3 * i + 1], h[3 * i + 2], o[2 * i], o[2 * i + 1]);
      return 0;
    }
    if (print) { printf("    {\\"%s\\", 0x%016llxull},\\n", kTags[v], hash); continue; }
    if (dump >= 0) continue;
    const int ne = sizeof kExpected / sizeof kExpected[0];
    if (v >= ne || hash != kExpected[v].hash)
      if (bad++ < 40) printf("%s: 0x%016llx, the card gave 0x%016llx\\n", kTags[v], hash, v < ne ? kExpected[v].hash : 0ull);
  }
  if (print) return 0;
  const int ne = sizeof kExpected / sizeof kExpected[0];
  printf("%d warp, atomic and memory forms, %d differ\\n%s\\n", nv, bad, bad || ne != nv ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}""")
open(os.path.join(os.path.dirname(__file__) or '.', 'ptx_warp_mem.cu'), 'w').write('\n'.join(out) + '\n')
print(len(V), 'variants')
