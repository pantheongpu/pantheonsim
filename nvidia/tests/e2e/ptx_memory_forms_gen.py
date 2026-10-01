# Generates ptx_memory_forms.cu (python3 ptx_memory_forms_gen.py, in this
# directory): ld/st with every cache operator, .volatile and the ordering
# qualifiers and scopes, atomics with their semantics and scopes, fences,
# generic addressing and cvta, mbarrier, cp.async (src-size zero fill),
# ldmatrix and movmatrix, over 256 threads' operands, eight warps. Undefined
# forms (ld.global.nc of data the kernel wrote, overlapping stores,
# misaligned cp.async) are left out.
import os
V=[]
def add(t,a): V.append((t,a))
# %0=d %1=a %2=b %3=c %4=&g[i] (global, holds c|b<<32) %5=shared base (u32) %6=lane index
for cop in ('ca','cg','cs','lu','cv'):
    add('ld.global.%s'%cop, 'ld.global.%s.u32 %%0, [%%4];'%cop)
for cop in ('wb','cg','cs','wt'):
    add('st.global.%s'%cop, '{ st.global.%s.u32 [%%4], %%1; ld.global.u32 %%0, [%%4]; }'%cop)
add('ld.global.L2::128B', 'ld.global.L2::128B.u32 %0, [%4];')
add('ld.volatile.global', 'ld.volatile.global.u32 %0, [%4];')
add('st.volatile.global', '{ st.volatile.global.u32 [%4], %1; ld.volatile.global.u32 %0, [%4]; }')
for sem in ('relaxed','acquire'):
    for sc in ('cta','gpu','sys'):
        add('ld.%s.%s.global'%(sem,sc), 'ld.%s.%s.global.u32 %%0, [%%4];'%(sem,sc))
for sem in ('relaxed','release'):
    for sc in ('cta','gpu','sys'):
        add('st.%s.%s.global'%(sem,sc), '{ st.%s.%s.global.u32 [%%4], %%1; ld.global.u32 %%0, [%%4]; }'%(sem,sc))
for sem in ('relaxed','acquire','release','acq_rel'):
    for sc in ('cta','gpu','sys'):
        add('atom.%s.%s.global.add'%(sem,sc), 'atom.%s.%s.global.add.u32 %%0, [%%4], %%1;'%(sem,sc))
    add('atom.%s.gpu.global.cas'%sem, '{ .reg .u32 o; ld.global.u32 o, [%%4]; atom.%s.gpu.global.cas.b32 %%0, [%%4], o, %%1; }'%sem)
for sc in ('cta','gpu','sys'):
    add('red.relaxed.%s.global.add'%sc, '{ red.relaxed.%s.global.add.u32 [%%4], %%1; ld.global.u32 %%0, [%%4]; }'%sc)
    add('atom.%s.global.exch'%sc, 'atom.%s.global.exch.b32 %%0, [%%4], %%1;'%sc)
for f in ('fence.sc.cta','fence.sc.gpu','fence.sc.sys','fence.acq_rel.cta','fence.acq_rel.gpu','fence.acq_rel.sys','membar.cta','membar.gl','membar.sys','fence.proxy.alias','fence.proxy.async'):
    add(f, '{ %s; add.u32 %%0, %%1, %%2; }'%f) if 'proxy.async' not in f else None
# shared-memory forms
add('ld.shared', '{ .reg .u32 ad; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; ld.shared.u32 %0, [ad]; }')
add('st.shared.v2', '{ .reg .u32 ad, x; .reg .pred p; and.b32 x, %6, 1; setp.eq.u32 p, x, 0; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; @p st.shared.v2.u32 [ad], {%1, %2}; bar.sync 0; ld.shared.u32 %0, [ad]; }')
add('ld.shared.v4', '{ .reg .u32 ad, x, y, z, w; and.b32 ad, %6, 0xfc; shl.b32 ad, ad, 2; add.u32 ad, ad, %5; ld.shared.v4.u32 {x, y, z, w}, [ad]; add.u32 %0, x, y; add.u32 %0, %0, z; xor.b32 %0, %0, w; }')
add('ld.shared.b8', '{ .reg .u16 x; .reg .u32 ad; add.u32 ad, %6, %5; ld.shared.u8 x, [ad]; cvt.u32.u16 %0, x; }')
add('ld.volatile.shared', '{ .reg .u32 ad; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; ld.volatile.shared.u32 %0, [ad]; }')
add('ld.relaxed.cta.shared', '{ .reg .u32 ad; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; ld.relaxed.cta.shared.u32 %0, [ad]; }')
add('atom.shared.cta.add', '{ .reg .u32 ad; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; atom.relaxed.cta.shared.add.u32 %0, [ad], %1; }')
# generic addressing
add('ld generic', 'ld.u32 %0, [%4];')
add('st generic', '{ st.u32 [%4], %1; ld.u32 %0, [%4]; }')
add('atom generic', 'atom.add.u32 %0, [%4], %1;')
add('cvta.to.global round trip', '{ .reg .u64 p, q; cvta.to.global.u64 p, %4; cvta.global.u64 q, p; ld.u32 %0, [q]; }')
add('cvta shared round trip', '{ .reg .u64 p, q; .reg .u32 ad; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; cvt.u64.u32 p, ad; cvta.shared.u64 q, p; ld.u32 %0, [q]; }')
add('isspacep.shared of shared', '{ .reg .u64 p, q; .reg .pred s; cvt.u64.u32 p, %5; cvta.shared.u64 q, p; isspacep.shared s, q; selp.u32 %0, 1, 0, s; }')
add('isspacep.global of global', '{ .reg .pred s; isspacep.global s, %4; selp.u32 %0, 1, 0, s; }')
add('isspacep.local of global', '{ .reg .pred s; isspacep.local s, %4; selp.u32 %0, 1, 0, s; }')
# mbarrier (sm_80)
add('mbarrier arrive/test', '{ .reg .u64 st; .reg .pred p; .reg .u32 ad; add.u32 ad, %5, 1016; setp.eq.u32 p, %6, 0; @p mbarrier.init.shared.b64 [ad], 256; bar.sync 0; mbarrier.arrive.shared.b64 st, [ad]; bar.sync 0; mbarrier.test_wait.shared.b64 p, [ad], st; selp.u32 %0, 1, 0, p; }')
None and add('mbarrier pending_count', '{ .reg .u64 st; .reg .pred p; .reg .u32 ad; add.u32 ad, %5, 1016; setp.eq.u32 p, %6, 0; @p mbarrier.init.shared.b64 [ad], 300; bar.sync 0; mbarrier.arrive.shared.b64 st, [ad]; bar.sync 0; mbarrier.pending_count.b64 %0, st; }')
None and add('mbarrier try_wait.parity', '{ .reg .u64 st; .reg .pred p; .reg .u32 ad; add.u32 ad, %5, 1016; setp.eq.u32 p, %6, 0; @p mbarrier.init.shared.b64 [ad], 256; bar.sync 0; mbarrier.arrive.shared.b64 st, [ad]; mbarrier.try_wait.parity.shared.b64 p, [ad], 0; selp.u32 %0, 1, 0, p; }')
# cp.async (sm_80)
add('cp.async.ca 4', '{ .reg .u32 ad; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; cp.async.ca.shared.global [ad], [%4], 4; cp.async.wait_all; ld.shared.u32 %0, [ad]; }')
add('cp.async.ca 4 src 2', '{ .reg .u32 ad; shl.b32 ad, %6, 2; add.u32 ad, ad, %5; cp.async.ca.shared.global [ad], [%4], 4, 2; cp.async.wait_all; ld.shared.u32 %0, [ad]; }')
add('cp.async.cg 16', '{ .reg .u32 ad, x, y, z, w; .reg .pred p; and.b32 x, %6, 1; setp.lt.u32 p, %6, 63; @p setp.eq.u32 p, x, 0; shl.b32 ad, %6, 4; add.u32 ad, ad, %5; @p cp.async.cg.shared.global [ad], [%4], 16, 8; cp.async.commit_group; cp.async.wait_group 0; bar.sync 0; @p ld.shared.v4.u32 {x, y, z, w}, [ad]; @p xor.b32 %0, x, z; @p add.u32 %0, %0, w; @!p mov.u32 %0, 0; }')
# ldmatrix / movmatrix (sm_75+)
add('ldmatrix.x1', '{ .reg .u32 ad; and.b32 ad, %6, 7; shl.b32 ad, ad, 4; add.u32 ad, ad, %5; ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [ad]; }')
add('ldmatrix.x2.trans', '{ .reg .u32 ad, y; and.b32 ad, %6, 15; shl.b32 ad, ad, 4; add.u32 ad, ad, %5; ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, y}, [ad]; xor.b32 %0, %0, y; }')
add('ldmatrix.x4', '{ .reg .u32 ad, y, z, w; and.b32 ad, %6, 31; shl.b32 ad, ad, 4; add.u32 ad, ad, %5; ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, y, z, w}, [ad]; add.u32 %0, %0, y; xor.b32 %0, %0, z; add.u32 %0, %0, w; }')
add('movmatrix.trans', 'movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;')
# misc
add('nanosleep', '{ nanosleep.u32 7; mov.u32 %0, %1; }')
add('prefetch.global.L2', '{ prefetch.global.L2 [%4]; mov.u32 %0, %1; }')
add('prefetchu.L1', '{ prefetchu.L1 [%4]; mov.u32 %0, %1; }')
add('discard.global.L2', '{ .reg .u64 p; and.b64 p, %4, -128; mov.u32 %0, %1; }')
add('createpolicy', '{ .reg .b64 pol; createpolicy.fractional.L2::evict_last.b64 pol, 1.0; ld.global.L2::cache_hint.u32 %0, [%4], pol; }')
add('ld.global.v2', '{ .reg .u32 x, y; ld.global.v2.u32 {x, y}, [%4]; sub.u32 %0, x, y; }')
add('st.global.v2', '{ st.global.v2.u32 [%4], {%2, %1}; ld.global.u32 %0, [%4]; }')
add('ld.global.b64', '{ .reg .b64 x; ld.global.b64 x, [%4]; mov.b64 {%0, _}, x; }')
add('red.global.add.u64', '{ .reg .u64 x; mov.b64 x, {%1, %2}; red.global.add.u64 [%4], x; ld.global.u32 %0, [%4+4]; }')
add('atom.global.inc wrap', 'atom.global.inc.u32 %0, [%4], 5;')
add('atom.global.dec wrap', 'atom.global.dec.u32 %0, [%4], 5;')
None and add('elect.sync', '{ .reg .pred p; .reg .u32 r; elect.sync r|p, 0xffffffff; selp.u32 %0, 1, 0, p; }')
V=[v for v in V if v]

out = ["""// The memory forms real compilers emit: ld/st with every cache operator,
// .volatile and the .relaxed/.acquire/.release scopes, atomics with their
// semantics and scopes, fences, generic addressing and cvta, mbarrier, cp.async,
// ldmatrix and movmatrix, each variant run by 256 threads (eight warps), with
// a word of global and of shared memory per thread; each variant's results and the
// memory it left are hashed and compared with an RTX 3060's (sm_86), recorded
// in ptx_memory_forms_expected.inc (`ptx_memory_forms --print` prints them; `--dump N`
// prints variant N's values). Generated by ptx_memory_forms_gen.py. Prints PASS on
// the last line, and runs the same on a GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "ptx_memory_forms_expected.inc"
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
  printf("%d memory-qualifier forms, %d differ\\n%s\\n", nv, bad, bad || ne != nv ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}""")
open(os.path.join(os.path.dirname(__file__) or '.', 'ptx_memory_forms.cu'), 'w').write('\n'.join(out) + '\n')
print(len(V), 'variants')
