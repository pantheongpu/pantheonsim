# Generates ptx_forms.cu (python3 ptx_forms_gen.py, in this directory): PTX
# forms a differential probe against an RTX 3060 found missing or wrong --
# prmt's modes, setp/set with a boolean operation, mul24/mad24, the .sat,
# rounding and .xorsign.abs modifiers, the half-precision modifiers, and the
# cvt forms with .relu, .rz and tf32 -- each over the same 256 operand triples.
import os
V = []   # (tag, asm using %0=d %1=a %2=b %3=c, oldest CUDA whose ptxas takes it)
# The needs value is major*10+minor. Variants a toolkit's ptxas does not know
# are compiled out there and reported as not built, not as passing: CI's
# hosted jobs use Ubuntu's CUDA 12.0, and cvt's .satfinite on f16x2/bf16x2
# came with PTX ISA 8.1 (CUDA 12.1).
def add(tag, asm, needs=0): V.append((tag, asm, needs))
for m in ('f4e', 'b4e', 'rc8', 'ecl', 'ecr', 'rc16'):
    add('prmt.' + m, 'prmt.b32.' + m + ' %0, %1, %2, %3;')
for bop in ('and', 'or', 'xor'):
    for cmp, ty in (('lt', 'u32'), ('ge', 's32'), ('ne', 'f32'), ('gtu', 'f32')):
        add('setp.%s.%s.%s p' % (cmp, bop, ty), '{ .reg .pred p, q; setp.ne.u32 q, %%3, 0; setp.%s.%s.%s p, %%1, %%2, q; selp.u32 %%0, 1, 0, p; }' % (cmp, bop, ty))
        add('setp.%s.%s.%s p|q !c' % (cmp, bop, ty), '{ .reg .pred p, q, r; setp.ne.u32 r, %%3, 0; setp.%s.%s.%s p|q, %%1, %%2, !r; selp.u32 %%0, 1, 2, p; selp.u32 %%0, %%0, 4, q; }' % (cmp, bop, ty))
    add('set.lt.%s.u32.s32' % bop, '{ .reg .pred r; setp.ne.u32 r, %%3, 0; set.lt.%s.u32.s32 %%0, %%1, %%2, r; }' % bop)
    add('set.le.%s.f32.f32' % bop, '{ .reg .pred r; setp.ne.u32 r, %%3, 0; set.le.%s.f32.f32 %%0, %%1, %%2, !r; }' % bop)
add('setp.lt.u32 p|q', '{ .reg .pred p, q; setp.lt.u32 p|q, %1, %2; selp.u32 %0, 1, 2, p; selp.u32 %0, %0, 4, q; }')
for f in ('mul24.lo.u32', 'mul24.lo.s32', 'mul24.hi.u32', 'mul24.hi.s32'):
    add(f, f + ' %0, %1, %2;')
for f in ('mad24.lo.u32', 'mad24.lo.s32', 'mad24.hi.u32', 'mad24.hi.s32', 'mad24.hi.sat.s32'):
    add(f, f + ' %0, %1, %2, %3;')
add('add.sat.s32', 'add.sat.s32 %0, %1, %2;')
add('sub.sat.s32', 'sub.sat.s32 %0, %1, %2;')
add('mad.hi.sat.s32', 'mad.hi.sat.s32 %0, %1, %2, %3;')
F = '{ .reg .f32 a,b,c; mov.b32 a, %1; mov.b32 b, %2; mov.b32 c, %3; OP; mov.b32 %0, a; }'
for rnd in ('rn', 'rz', 'rm', 'rp'):
    add('fma.%s.f32' % rnd, F.replace('OP', 'fma.%s.f32 a, a, b, c' % rnd))
    add('fma.%s.sat.f32' % rnd, F.replace('OP', 'fma.%s.sat.f32 a, a, b, c' % rnd))
    add('fma.%s.ftz.f32' % rnd, F.replace('OP', 'fma.%s.ftz.f32 a, a, b, c' % rnd))
    add('mad.%s.f32' % rnd, F.replace('OP', 'mad.%s.f32 a, a, b, c' % rnd))
    add('add.%s.sat.f32' % rnd, F.replace('OP', 'add.%s.sat.f32 a, a, b' % rnd))
    add('mul.%s.sat.f32' % rnd, F.replace('OP', 'mul.%s.sat.f32 a, a, b' % rnd))
add('sub.sat.f32', F.replace('OP', 'sub.sat.f32 a, a, b'))
D = '{ .reg .f64 a,b,c; mov.b64 a, {%1, %2}; mov.b64 b, {%2, %3}; mov.b64 c, {%3, %1}; OP; mov.b64 {%0, _}, a; }'
D2 = '{ .reg .f64 a,b,c; mov.b64 a, {%1, %2}; mov.b64 b, {%2, %3}; mov.b64 c, {%3, %1}; OP; mov.b64 {_, %0}, a; }'
for rnd in ('rz', 'rm', 'rp'):
    add('fma.%s.f64 lo' % rnd, D.replace('OP', 'fma.%s.f64 a, a, b, c' % rnd))
    add('fma.%s.f64 hi' % rnd, D2.replace('OP', 'fma.%s.f64 a, a, b, c' % rnd))
for op in ('min', 'max'):
    add(op + '.xorsign.abs.f32', F.replace('OP', op + '.xorsign.abs.f32 a, a, b'))
    add(op + '.NaN.xorsign.abs.f32', F.replace('OP', op + '.NaN.xorsign.abs.f32 a, a, b'))
    for t in ('f16x2', 'bf16x2'):
        add(op + '.xorsign.abs.' + t, op + '.xorsign.abs.' + t + ' %0, %1, %2;')
        add(op + '.NaN.' + t, op + '.NaN.' + t + ' %0, %1, %2;')
        add(op + '.NaN.xorsign.abs.' + t, op + '.NaN.xorsign.abs.' + t + ' %0, %1, %2;')
for op in ('add', 'sub', 'mul'):
    for mod in ('sat', 'ftz', 'ftz.sat'):
        add('%s.%s.f16x2' % (op, mod), '%s.%s.f16x2 %%0, %%1, %%2;' % (op, mod))
for mod in ('sat', 'ftz', 'relu', 'ftz.sat', 'ftz.relu'):
    add('fma.rn.%s.f16x2' % mod, 'fma.rn.%s.f16x2 %%0, %%1, %%2, %%3;' % mod)
add('fma.rn.relu.bf16x2', 'fma.rn.relu.bf16x2 %0, %1, %2, %3;')
P = '{ .reg .f32 a,b; mov.b32 a, %1; mov.b32 b, %2; OP; }'
for t in ('f16x2', 'bf16x2'):
    for mod in ('rn', 'rz', 'rn.relu', 'rz.relu', 'rn.satfinite', 'rz.relu.satfinite'):
        add('cvt.%s.%s.f32' % (mod, t), P.replace('OP', 'cvt.%s.%s.f32 %%0, a, b' % (mod, t)),
            121 if 'satfinite' in mod else 0)
for t in ('f16', 'bf16'):
    add('cvt.rn.relu.%s.f32' % t, '{ .reg .f32 a; .reg .b16 h; mov.b32 a, %1; cvt.rn.relu.' + t + '.f32 h, a; mov.b32 %0, {h, h}; }')
add('cvt.rna.tf32.f32', '{ .reg .f32 a; mov.b32 a, %1; cvt.rna.tf32.f32 %0, a; }')
add('mov.b32 f16x2', '{ .reg .f16x2 h; mov.b32 h, %1; mov.b32 %0, h; }')

out = ['''// PTX forms a differential probe against an RTX 3060 found missing or
// wrong, each variant over the same 256 operand triples, its results hashed
// and compared with the card's (sm_86), recorded in ptx_forms_expected.inc
// (`ptx_forms --print` prints them; `--dump N` prints variant N's values).
// Generated by ptx_forms_gen.py. Prints PASS on the last line, and runs the
// same on a GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "ptx_forms_expected.inc"
};
constexpr int kN = 256;
#define CUDA_VER (__CUDACC_VER_MAJOR__ * 10 + __CUDACC_VER_MINOR__)
__global__ void k(int v, const unsigned* in, unsigned* out) {
  const int i = threadIdx.x;
  const unsigned a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
  unsigned d = 0;
  switch (v) {''']
for n, (tag, asm, needs) in enumerate(V):
    line = '    case %d: asm volatile("%s" : "=r"(d) : "r"(a), "r"(b), "r"(c)); break;' % (n, asm.replace('"', '\\"'))
    if needs:
        line = '#if CUDA_VER >= %d\n%s\n#endif' % (needs, line)
    out.append(line)
out.append('''  }
  out[i] = d;
}
static const char* kTags[] = {''')
for tag, asm, needs in V:
    out.append('  "%s",' % tag)
out.append('''};
static const int kNeeds[] = {''' + ', '.join(str(n) for _, _, n in V) + '''};
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
  int bad = 0, unbuilt = 0;
  for (int v = 0; v < nv; ++v) {
    if (kNeeds[v] > CUDA_VER) {
      if (print) { printf("    {\\"%s\\", 0x0ull},  // not built by this toolkit\\n", kTags[v]); continue; }
      printf("%s: not built (needs CUDA %d.%d)\\n", kTags[v], kNeeds[v] / 10, kNeeds[v] % 10);
      ++unbuilt;
      continue;
    }
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
  printf("%d PTX forms, %d differ, %d not built by this toolkit\\n%s\\n", nv, bad, unbuilt,
         bad || ne != nv ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}''')
open(os.path.join(os.path.dirname(__file__) or '.', 'ptx_forms.cu'), 'w').write('\n'.join(out) + '\n')
print(len(V), 'variants')
