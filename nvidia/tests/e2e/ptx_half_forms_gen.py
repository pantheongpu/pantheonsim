# Generates ptx_half_forms.cu (python3 ptx_half_forms_gen.py, in this
# directory): the half-precision PTX forms an RTX 3060 (sm_86) takes -- f16,
# f16x2, bf16 and bf16x2 add/sub/mul/fma/min/max/neg/abs with every modifier
# (.rn, .ftz, .sat, .relu, .oob, .NaN, .xorsign.abs), setp and set with every
# comparison (and f32's, with and without .ftz), and cvt between the half
# types and every integer width with each rounding mode and .sat -- each over the same 256 operand triples, as the
# earlier sweeps (ptx_sweep_gen.py). Forms sm_86's ptxas refuses are left out
# (REFUSED below, found by assembling each variant alone); the .approx forms
# follow documented error bounds, not the card's bits, and are not here.
import os
V = []
REFUSED = {  # sm_86's ptxas (CUDA 13.0): sm_90-only features and modifiers the form does not take
    'fma.rn.oob.f16', 'fma.rn.oob.relu.f16', 'fma.rn.oob.f16x2', 'fma.rn.oob.relu.f16x2',
    'add.bf16', 'add.rn.bf16', 'add.ftz.bf16', 'add.sat.bf16',
    'add.ftz.sat.bf16', 'add.rn.ftz.bf16', 'add.rn.sat.bf16', 'add.rn.ftz.sat.bf16',
    'sub.bf16', 'sub.rn.bf16', 'sub.ftz.bf16', 'sub.sat.bf16',
    'sub.ftz.sat.bf16', 'sub.rn.ftz.bf16', 'sub.rn.sat.bf16', 'sub.rn.ftz.sat.bf16',
    'mul.bf16', 'mul.rn.bf16', 'mul.ftz.bf16', 'mul.sat.bf16',
    'mul.ftz.sat.bf16', 'mul.rn.ftz.bf16', 'mul.rn.sat.bf16', 'mul.rn.ftz.sat.bf16',
    'fma.rn.ftz.bf16', 'fma.rn.sat.bf16', 'fma.rn.ftz.sat.bf16', 'fma.rn.ftz.relu.bf16',
    'fma.rn.oob.bf16', 'fma.rn.oob.relu.bf16', 'min.ftz.bf16', 'min.ftz.NaN.bf16',
    'min.ftz.xorsign.abs.bf16', 'min.ftz.NaN.xorsign.abs.bf16', 'max.ftz.bf16', 'max.ftz.NaN.bf16',
    'max.ftz.xorsign.abs.bf16', 'max.ftz.NaN.xorsign.abs.bf16', 'neg.ftz.bf16', 'abs.ftz.bf16',
    'setp.eq.bf16', 'setp.eq.ftz.bf16', 'setp.ne.bf16', 'setp.ne.ftz.bf16',
    'setp.lt.bf16', 'setp.lt.ftz.bf16', 'setp.le.bf16', 'setp.le.ftz.bf16',
    'setp.gt.bf16', 'setp.gt.ftz.bf16', 'setp.ge.bf16', 'setp.ge.ftz.bf16',
    'setp.equ.bf16', 'setp.equ.ftz.bf16', 'setp.neu.bf16', 'setp.neu.ftz.bf16',
    'setp.ltu.bf16', 'setp.ltu.ftz.bf16', 'setp.leu.bf16', 'setp.leu.ftz.bf16',
    'setp.gtu.bf16', 'setp.gtu.ftz.bf16', 'setp.geu.bf16', 'setp.geu.ftz.bf16',
    'setp.num.bf16', 'setp.num.ftz.bf16', 'setp.nan.bf16', 'setp.nan.ftz.bf16',
    'add.bf16x2', 'add.rn.bf16x2', 'add.ftz.bf16x2', 'add.sat.bf16x2',
    'add.ftz.sat.bf16x2', 'add.rn.ftz.bf16x2', 'add.rn.sat.bf16x2', 'add.rn.ftz.sat.bf16x2',
    'sub.bf16x2', 'sub.rn.bf16x2', 'sub.ftz.bf16x2', 'sub.sat.bf16x2',
    'sub.ftz.sat.bf16x2', 'sub.rn.ftz.bf16x2', 'sub.rn.sat.bf16x2', 'sub.rn.ftz.sat.bf16x2',
    'mul.bf16x2', 'mul.rn.bf16x2', 'mul.ftz.bf16x2', 'mul.sat.bf16x2',
    'mul.ftz.sat.bf16x2', 'mul.rn.ftz.bf16x2', 'mul.rn.sat.bf16x2', 'mul.rn.ftz.sat.bf16x2',
    'fma.rn.ftz.bf16x2', 'fma.rn.sat.bf16x2', 'fma.rn.ftz.sat.bf16x2', 'fma.rn.ftz.relu.bf16x2',
    'fma.rn.oob.bf16x2', 'fma.rn.oob.relu.bf16x2', 'min.ftz.bf16x2', 'min.ftz.NaN.bf16x2',
    'min.ftz.xorsign.abs.bf16x2', 'min.ftz.NaN.xorsign.abs.bf16x2', 'max.ftz.bf16x2', 'max.ftz.NaN.bf16x2',
    'max.ftz.xorsign.abs.bf16x2', 'max.ftz.NaN.xorsign.abs.bf16x2', 'neg.ftz.bf16x2', 'abs.ftz.bf16x2',
    'setp.eq.bf16x2', 'set.eq.u32.bf16x2', 'setp.eq.ftz.bf16x2', 'set.eq.ftz.u32.bf16x2',
    'setp.ne.bf16x2', 'set.ne.u32.bf16x2', 'setp.ne.ftz.bf16x2', 'set.ne.ftz.u32.bf16x2',
    'setp.lt.bf16x2', 'set.lt.u32.bf16x2', 'setp.lt.ftz.bf16x2', 'set.lt.ftz.u32.bf16x2',
    'setp.le.bf16x2', 'set.le.u32.bf16x2', 'setp.le.ftz.bf16x2', 'set.le.ftz.u32.bf16x2',
    'setp.gt.bf16x2', 'set.gt.u32.bf16x2', 'setp.gt.ftz.bf16x2', 'set.gt.ftz.u32.bf16x2',
    'setp.ge.bf16x2', 'set.ge.u32.bf16x2', 'setp.ge.ftz.bf16x2', 'set.ge.ftz.u32.bf16x2',
    'setp.equ.bf16x2', 'set.equ.u32.bf16x2', 'setp.equ.ftz.bf16x2', 'set.equ.ftz.u32.bf16x2',
    'setp.neu.bf16x2', 'set.neu.u32.bf16x2', 'setp.neu.ftz.bf16x2', 'set.neu.ftz.u32.bf16x2',
    'setp.ltu.bf16x2', 'set.ltu.u32.bf16x2', 'setp.ltu.ftz.bf16x2', 'set.ltu.ftz.u32.bf16x2',
    'setp.leu.bf16x2', 'set.leu.u32.bf16x2', 'setp.leu.ftz.bf16x2', 'set.leu.ftz.u32.bf16x2',
    'setp.gtu.bf16x2', 'set.gtu.u32.bf16x2', 'setp.gtu.ftz.bf16x2', 'set.gtu.ftz.u32.bf16x2',
    'setp.geu.bf16x2', 'set.geu.u32.bf16x2', 'setp.geu.ftz.bf16x2', 'set.geu.ftz.u32.bf16x2',
    'setp.num.bf16x2', 'set.num.u32.bf16x2', 'setp.num.ftz.bf16x2', 'set.num.ftz.u32.bf16x2',
    'setp.nan.bf16x2', 'set.nan.u32.bf16x2', 'setp.nan.ftz.bf16x2', 'set.nan.ftz.u32.bf16x2',
    'cvt.rm.relu.f16.f32', 'cvt.rp.relu.f16.f32', 'cvt.rni.s8.bf16', 'cvt.rni.sat.s8.bf16',
    'cvt.rzi.s8.bf16', 'cvt.rzi.sat.s8.bf16', 'cvt.rmi.s8.bf16', 'cvt.rmi.sat.s8.bf16',
    'cvt.rpi.s8.bf16', 'cvt.rpi.sat.s8.bf16', 'cvt.rn.bf16.s8', 'cvt.rz.bf16.s8',
    'cvt.rm.bf16.s8', 'cvt.rp.bf16.s8', 'cvt.rni.u8.bf16', 'cvt.rni.sat.u8.bf16',
    'cvt.rzi.u8.bf16', 'cvt.rzi.sat.u8.bf16', 'cvt.rmi.u8.bf16', 'cvt.rmi.sat.u8.bf16',
    'cvt.rpi.u8.bf16', 'cvt.rpi.sat.u8.bf16', 'cvt.rn.bf16.u8', 'cvt.rz.bf16.u8',
    'cvt.rm.bf16.u8', 'cvt.rp.bf16.u8', 'cvt.rni.s16.bf16', 'cvt.rni.sat.s16.bf16',
    'cvt.rzi.s16.bf16', 'cvt.rzi.sat.s16.bf16', 'cvt.rmi.s16.bf16', 'cvt.rmi.sat.s16.bf16',
    'cvt.rpi.s16.bf16', 'cvt.rpi.sat.s16.bf16', 'cvt.rn.bf16.s16', 'cvt.rz.bf16.s16',
    'cvt.rm.bf16.s16', 'cvt.rp.bf16.s16', 'cvt.rni.u16.bf16', 'cvt.rni.sat.u16.bf16',
    'cvt.rzi.u16.bf16', 'cvt.rzi.sat.u16.bf16', 'cvt.rmi.u16.bf16', 'cvt.rmi.sat.u16.bf16',
    'cvt.rpi.u16.bf16', 'cvt.rpi.sat.u16.bf16', 'cvt.rn.bf16.u16', 'cvt.rz.bf16.u16',
    'cvt.rm.bf16.u16', 'cvt.rp.bf16.u16', 'cvt.rni.s32.bf16', 'cvt.rni.sat.s32.bf16',
    'cvt.rzi.s32.bf16', 'cvt.rzi.sat.s32.bf16', 'cvt.rmi.s32.bf16', 'cvt.rmi.sat.s32.bf16',
    'cvt.rpi.s32.bf16', 'cvt.rpi.sat.s32.bf16', 'cvt.rn.bf16.s32', 'cvt.rz.bf16.s32',
    'cvt.rm.bf16.s32', 'cvt.rp.bf16.s32', 'cvt.rni.u32.bf16', 'cvt.rni.sat.u32.bf16',
    'cvt.rzi.u32.bf16', 'cvt.rzi.sat.u32.bf16', 'cvt.rmi.u32.bf16', 'cvt.rmi.sat.u32.bf16',
    'cvt.rpi.u32.bf16', 'cvt.rpi.sat.u32.bf16', 'cvt.rn.bf16.u32', 'cvt.rz.bf16.u32',
    'cvt.rm.bf16.u32', 'cvt.rp.bf16.u32', 'cvt.rni.s64.bf16', 'cvt.rni.sat.s64.bf16',
    'cvt.rzi.s64.bf16', 'cvt.rzi.sat.s64.bf16', 'cvt.rmi.s64.bf16', 'cvt.rmi.sat.s64.bf16',
    'cvt.rpi.s64.bf16', 'cvt.rpi.sat.s64.bf16', 'cvt.rn.bf16.s64', 'cvt.rz.bf16.s64',
    'cvt.rm.bf16.s64', 'cvt.rp.bf16.s64', 'cvt.rni.u64.bf16', 'cvt.rni.sat.u64.bf16',
    'cvt.rzi.u64.bf16', 'cvt.rzi.sat.u64.bf16', 'cvt.rmi.u64.bf16', 'cvt.rmi.sat.u64.bf16',
    'cvt.rpi.u64.bf16', 'cvt.rpi.sat.u64.bf16', 'cvt.rn.bf16.u64', 'cvt.rz.bf16.u64',
    'cvt.rm.bf16.u64', 'cvt.rp.bf16.u64', 'cvt.rn.sat.bf16.f32', 'cvt.rz.sat.bf16.f32',
    'cvt.rm.sat.bf16.f32', 'cvt.rm.relu.bf16.f32', 'cvt.rp.sat.bf16.f32', 'cvt.rp.relu.bf16.f32',
    'cvt.ftz.f32.bf16', 'cvt.f64.bf16 hi', 'cvt.rn.bf16.f16', 'cvt.rn.f16.bf16',
}
def add(t, a):
    if t not in REFUSED: V.append((t, a))
# Scalars use each operand's low half; the result is written to both halves.
S = '{ .reg .b16 x, y, z, r; mov.b32 {x, _}, %1; mov.b32 {y, _}, %2; mov.b32 {z, _}, %3; OP; mov.b32 %0, {r, r}; }'
SP = '{ .reg .b16 x, y; .reg .pred p, q; mov.b32 {x, _}, %1; mov.b32 {y, _}, %2; OP; selp.u32 %0, 1, 0, p; }'
PP = '{ .reg .pred p, q; OP; selp.u32 %0, 1, 2, p; selp.u32 %0, %0, 4, q; }'
TYPES = ('f16', 'f16x2', 'bf16', 'bf16x2')
def form(t, op):
    return S.replace('OP', op.replace('D', 'r').replace('A', 'x').replace('B', 'y').replace('C', 'z')) if 'x2' not in t else \
        op.replace('D', '%0').replace('A', '%1').replace('B', '%2').replace('C', '%3') + ';'
for t in TYPES:
    for op in ('add', 'sub', 'mul'):
        for m in ('', '.rn', '.ftz', '.sat', '.ftz.sat', '.rn.ftz', '.rn.sat', '.rn.ftz.sat'):
            add('%s%s.%s' % (op, m, t), form(t, '%s%s.%s D, A, B' % (op, m, t)))
    for m in ('', '.ftz', '.sat', '.ftz.sat', '.relu', '.ftz.relu', '.oob', '.oob.relu'):
        add('fma.rn%s.%s' % (m, t), form(t, 'fma.rn%s.%s D, A, B, C' % (m, t)))
    for op in ('min', 'max'):
        for m in ('', '.ftz', '.NaN', '.ftz.NaN', '.xorsign.abs', '.ftz.xorsign.abs', '.NaN.xorsign.abs',
                  '.ftz.NaN.xorsign.abs'):
            add('%s%s.%s' % (op, m, t), form(t, '%s%s.%s D, A, B' % (op, m, t)))
    for op in ('neg', 'abs'):
        for m in ('', '.ftz'):
            add('%s%s.%s' % (op, m, t), form(t, '%s%s.%s D, A' % (op, m, t)))
    for cmp in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'equ', 'neu', 'ltu', 'leu', 'gtu', 'geu', 'num', 'nan'):
        for m in ('', '.ftz'):
            if 'x2' in t:
                add('setp.%s%s.%s' % (cmp, m, t), PP.replace('OP', 'setp.%s%s.%s p|q, %%1, %%2' % (cmp, m, t)))
                add('set.%s%s.u32.%s' % (cmp, m, t), 'set.%s%s.u32.%s %%0, %%1, %%2;' % (cmp, m, t))
            else:
                add('setp.%s%s.%s' % (cmp, m, t), SP.replace('OP', 'setp.%s%s.%s p, x, y' % (cmp, m, t)))
INTS = ('s8', 'u8', 's16', 'u16', 's32', 'u32', 's64', 'u64')
for t in ('f16', 'bf16'):
    for it in INTS:
        wide = it.endswith('64')
        for r in ('rni', 'rzi', 'rmi', 'rpi'):
            for sat in ('', '.sat'):
                if wide:
                    op = '{ .reg .b16 x; .reg .%s d; mov.b32 {x, _}, %%1; cvt.%s%s.%s.%s d, x; mov.b64 {%%0, _}, d; }' % (it, r, sat, it, t)
                elif it in ('s8', 'u8', 's16', 'u16'):
                    op = '{ .reg .b16 x; .reg .%s d; mov.b32 {x, _}, %%1; cvt.%s%s.%s.%s d, x; cvt.u32.%s %%0, d; }' % ('b16' if it.endswith('16') else it, r, sat, it, t, 'u16' if it.endswith('16') else it)
                else:
                    op = '{ .reg .b16 x; mov.b32 {x, _}, %%1; cvt.%s%s.%s.%s %%0, x; }' % (r, sat, it, t)
                add('cvt.%s%s.%s.%s' % (r, sat, it, t), op)
        for r in ('rn', 'rz', 'rm', 'rp'):
            if wide:
                op = '{ .reg .b16 h; .reg .%s s; mov.b64 s, {%%1, %%2}; cvt.%s.%s.%s h, s; mov.b32 %%0, {h, h}; }' % (it, r, t, it)
            elif it in ('s8', 'u8', 's16', 'u16'):
                op = '{ .reg .b16 h; .reg .%s s; cvt.%s.u32 s, %%1; cvt.%s.%s.%s h, s; mov.b32 %%0, {h, h}; }' % (it if it.endswith('8') else it, it, r, t, it)
            else:
                op = '{ .reg .b16 h; cvt.%s.%s.%s h, %%1; mov.b32 %%0, {h, h}; }' % (r, t, it)
            add('cvt.%s.%s.%s' % (r, t, it), op)
    for r in ('rn', 'rz', 'rm', 'rp'):
        for sat in ('', '.sat', '.relu'):
            add('cvt.%s%s.%s.f32' % (r, sat, t), '{ .reg .f32 a; .reg .b16 h; mov.b32 a, %%1; cvt.%s%s.%s.f32 h, a; mov.b32 %%0, {h, h}; }' % (r, sat, t))
    add('cvt.f32.%s' % t, '{ .reg .b16 h; .reg .f32 d; mov.b32 {h, _}, %%1; cvt.f32.%s d, h; mov.b32 %%0, d; }' % t)
    add('cvt.ftz.f32.%s' % t, '{ .reg .b16 h; .reg .f32 d; mov.b32 {h, _}, %%1; cvt.ftz.f32.%s d, h; mov.b32 %%0, d; }' % t)
    add('cvt.f64.%s hi' % t, '{ .reg .b16 h; .reg .f64 d; mov.b32 {h, _}, %%1; cvt.f64.%s d, h; mov.b64 {_, %%0}, d; }' % t)
# f32 comparisons with and without .ftz, which the parser used to drop for
# every type (subnormal operands compare as their signed zero).
F32P = '{ .reg .f32 x, y; .reg .pred p; mov.b32 x, %1; mov.b32 y, %2; OP; selp.u32 %0, 1, 0, p; }'
for cmp in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'equ', 'neu', 'ltu', 'leu', 'gtu', 'geu', 'num', 'nan'):
    for m in ('', '.ftz'):
        add('setp.%s%s.f32' % (cmp, m), F32P.replace('OP', 'setp.%s%s.f32 p, x, y' % (cmp, m)))
        add('set.%s%s.u32.f32' % (cmp, m), 'set.%s%s.u32.f32 %%0, %%1, %%2;' % (cmp, m))
add('cvt.rn.bf16.f16', '{ .reg .b16 h, g; mov.b32 {h, _}, %1; cvt.rn.bf16.f16 g, h; mov.b32 %0, {g, g}; }')
add('cvt.rn.f16.bf16', '{ .reg .b16 h, g; mov.b32 {h, _}, %1; cvt.rn.f16.bf16 g, h; mov.b32 %0, {g, g}; }')

out = ['''// Half-precision PTX -- f16, f16x2, bf16 and bf16x2 arithmetic with every
// modifier, comparisons, and conversions to and from every integer width --
// each variant over the same 256 operand triples, its results hashed
// and compared with the card's (sm_86), recorded in ptx_half_forms_expected.inc
// (`ptx_sweep --print` prints them; `--dump N` prints variant N's values).
// Generated by ptx_half_forms_gen.py. Prints PASS on the last line, and runs the
// same on a GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "ptx_half_forms_expected.inc"
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
  printf("%d half-precision PTX forms, %d differ\\n%s\\n", nv, bad, bad || ne != nv ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}''')
open(os.path.join(os.path.dirname(__file__) or '.', 'ptx_half_forms.cu'), 'w').write('\n'.join(out) + '\n')
print(len(V), 'variants')
