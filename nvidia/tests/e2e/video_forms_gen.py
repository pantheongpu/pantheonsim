# Generates video_forms.cu (python3 video_forms_gen.py, in this directory): every video instruction form in a representative
# spread of types, selectors, saturation, secondary operations and merges.
import itertools
V = []  # (tag, asm text using %0=d %1=a %2=b %3=c)
T = ['u32', 's32']
types3 = [('u32','u32','u32'), ('s32','s32','s32'), ('s32','u32','s32'), ('u32','s32','u32')]
# scalar arithmetic
for op in ['vadd', 'vsub', 'vabsdiff', 'vmin', 'vmax']:
    for (d,a,b) in types3:
        for sel in [('', ''), ('.b1', '.h1'), ('.h0', '.b3')]:
            V.append((f'{op}.{d}.{a}.{b} a{sel[0] or "-"} b{sel[1] or "-"}', f'{op}.{d}.{a}.{b} %0, %1{sel[0]}, %2{sel[1]};'))
        V.append((f'{op}.{d}.{a}.{b}.sat', f'{op}.{d}.{a}.{b}.sat %0, %1, %2;'))
        for sec in ['add', 'min', 'max']:
            V.append((f'{op}.{d}.{a}.{b}.{sec}', f'{op}.{d}.{a}.{b}.{sec} %0, %1.b2, %2, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.sat.add', f'{op}.{d}.{a}.{b}.sat.add %0, %1, %2.h0, %3;'))
        for ds in ['b0', 'b3', 'h1']:
            V.append((f'{op}.{d}.{a}.{b} d.{ds}', f'{op}.{d}.{a}.{b} %0.{ds}, %1, %2, %3;'))
            V.append((f'{op}.{d}.{a}.{b}.sat d.{ds}', f'{op}.{d}.{a}.{b}.sat %0.{ds}, %1.h1, %2.b0, %3;'))
# the four-operand form with neither a secondary operation nor a merge, which
# ptxas takes
for op in ['vadd', 'vabsdiff', 'vmin']:
    for (d,a,b) in [('u32','u32','u32'), ('s32','s32','s32')]:
        V.append((f'{op}.{d}.{a}.{b} c-only', f'{op}.{d}.{a}.{b} %0, %1, %2, %3;'))
# shifts
for op in ['vshl', 'vshr']:
    for (d, a) in [('u32','u32'), ('s32','s32'), ('s32','u32'), ('u32','s32')]:
        for mode in ['clamp', 'wrap']:
            V.append((f'{op}.{d}.{a}.u32.{mode}', f'{op}.{d}.{a}.u32.{mode} %0, %1, %2;'))
            V.append((f'{op}.{d}.{a}.u32.{mode} sel', f'{op}.{d}.{a}.u32.{mode} %0, %1.h1, %2.b0;'))
        V.append((f'{op}.{d}.{a}.u32.sat.clamp', f'{op}.{d}.{a}.u32.sat.clamp %0, %1, %2.b1;'))
        V.append((f'{op}.{d}.{a}.u32.clamp.add', f'{op}.{d}.{a}.u32.clamp.add %0, %1, %2.b2, %3;'))
        V.append((f'{op}.{d}.{a}.u32.wrap d.h0', f'{op}.{d}.{a}.u32.wrap %0.h0, %1, %2.b3, %3;'))
# vmad
for (d,a,b) in [('u32','u32','u32'), ('s32','s32','s32'), ('s32','u32','s32'), ('s32','s32','u32')]:
    V.append((f'vmad.{d}.{a}.{b}', f'vmad.{d}.{a}.{b} %0, %1, %2, %3;'))
    V.append((f'vmad.{d}.{a}.{b} sel', f'vmad.{d}.{a}.{b} %0, %1.h1, %2.b2, %3;'))
    V.append((f'vmad.{d}.{a}.{b}.sat', f'vmad.{d}.{a}.{b}.sat %0, %1, %2, %3;'))
    V.append((f'vmad.{d}.{a}.{b}.shr7', f'vmad.{d}.{a}.{b}.shr7 %0, %1.h0, %2.h1, %3;'))
    V.append((f'vmad.{d}.{a}.{b}.sat.shr15', f'vmad.{d}.{a}.{b}.sat.shr15 %0, %1, %2, %3;'))
    V.append((f'vmad.{d}.{a}.{b}.po', f'vmad.{d}.{a}.{b}.po %0, %1.b1, %2.b3, %3;'))
    V.append((f'vmad.{d}.{a}.{b}.po.sat.shr7', f'vmad.{d}.{a}.{b}.po.sat.shr7 %0, %1, %2, %3;'))
    if d == 's32':
        V.append((f'vmad.{d}.{a}.{b} -a', f'vmad.{d}.{a}.{b} %0, -%1, %2, %3;'))
        V.append((f'vmad.{d}.{a}.{b} -c', f'vmad.{d}.{a}.{b} %0, %1, %2, -%3;'))
        V.append((f'vmad.{d}.{a}.{b}.sat -b', f'vmad.{d}.{a}.{b}.sat %0, %1.h0, -%2, %3;'))
# vset
for (a,b) in [('u32','u32'), ('s32','s32'), ('s32','u32')]:
    for cmp in ['eq', 'ne', 'lt', 'le', 'gt', 'ge']:
        V.append((f'vset.{a}.{b}.{cmp}', f'vset.{a}.{b}.{cmp} %0, %1.b1, %2.h0;'))
    V.append((f'vset.{a}.{b}.lt.add', f'vset.{a}.{b}.lt.add %0, %1, %2, %3;'))
    V.append((f'vset.{a}.{b}.ge.max', f'vset.{a}.{b}.ge.max %0, %1, %2, %3;'))
    V.append((f'vset.{a}.{b}.ne d.b2', f'vset.{a}.{b}.ne %0.b2, %1, %2, %3;'))
# SIMD 2
for op in ['vadd2', 'vsub2', 'vavrg2', 'vabsdiff2', 'vmin2', 'vmax2']:
    for (d,a,b) in [('u32','u32','u32'), ('s32','s32','s32'), ('s32','u32','s32')]:
        V.append((f'{op}.{d}.{a}.{b}', f'{op}.{d}.{a}.{b} %0, %1, %2, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.sat', f'{op}.{d}.{a}.{b}.sat %0, %1, %2, %3;'))
        V.append((f'{op}.{d}.{a}.{b} mask/sel', f'{op}.{d}.{a}.{b} %0.h1, %1.h21, %2.h03, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.sat mask', f'{op}.{d}.{a}.{b}.sat %0.h0, %1.h00, %2.h33, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.add', f'{op}.{d}.{a}.{b}.add %0, %1, %2, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.add mask', f'{op}.{d}.{a}.{b}.add %0.h1, %1.h12, %2, %3;'))
# SIMD 4
for op in ['vadd4', 'vsub4', 'vavrg4', 'vabsdiff4', 'vmin4', 'vmax4']:
    for (d,a,b) in [('u32','u32','u32'), ('s32','s32','s32'), ('u32','s32','u32')]:
        V.append((f'{op}.{d}.{a}.{b}', f'{op}.{d}.{a}.{b} %0, %1, %2, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.sat', f'{op}.{d}.{a}.{b}.sat %0, %1, %2, %3;'))
        V.append((f'{op}.{d}.{a}.{b} mask/sel', f'{op}.{d}.{a}.{b} %0.b20, %1.b7250, %2.b0123, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.sat mask', f'{op}.{d}.{a}.{b}.sat %0.b31, %1.b4444, %2.b6061, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.add', f'{op}.{d}.{a}.{b}.add %0, %1, %2, %3;'))
        V.append((f'{op}.{d}.{a}.{b}.add mask', f'{op}.{d}.{a}.{b}.add %0.b310, %1.b3210, %2.b1357, %3;'))
# vset2 / vset4
for (a,b) in [('u32','u32'), ('s32','s32'), ('u32','s32')]:
    for cmp in ['eq', 'lt', 'ge']:
        V.append((f'vset2.{a}.{b}.{cmp}', f'vset2.{a}.{b}.{cmp} %0, %1, %2, %3;'))
        V.append((f'vset4.{a}.{b}.{cmp}', f'vset4.{a}.{b}.{cmp} %0, %1, %2, %3;'))
    V.append((f'vset2.{a}.{b}.ne mask', f'vset2.{a}.{b}.ne %0.h0, %1.h30, %2.h12, %3;'))
    V.append((f'vset4.{a}.{b}.gt mask', f'vset4.{a}.{b}.gt %0.b21, %1.b7654, %2, %3;'))
    V.append((f'vset2.{a}.{b}.le.add', f'vset2.{a}.{b}.le.add %0, %1, %2, %3;'))
    V.append((f'vset4.{a}.{b}.lt.add', f'vset4.{a}.{b}.lt.add %0.b320, %1, %2.b5432, %3;'))

out = []
out.append('''// Every video instruction (PTX ISA 9.7.20) in a spread of types, operand
// selectors, saturation, secondary operations, merges and masks, run over the
// same 256 operand triples, each variant's results hashed and compared with an
// RTX 3060's (sm_86), recorded in video_forms_expected.inc (`video_forms
// --print` prints them). Generated by video_forms_gen.py.
// Prints PASS on the last line, and runs the same on a GPU.
#include <cstdio>
#include <cstdlib>\n#include <cstring>
struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "video_forms_expected.inc"
};
constexpr int kN = 256;
__global__ void k(int v, const unsigned* in, unsigned* out) {
  const int i = threadIdx.x;
  const unsigned a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
  unsigned d = 0;
  switch (v) {''')
for n, (tag, asm) in enumerate(V):
    out.append(f'    case {n}: asm volatile("{asm}" : "=r"(d) : "r"(a), "r"(b), "r"(c)); break;')
out.append('''  }
  out[i] = d;
}
static const char* kTags[] = {''')
for tag, asm in V:
    out.append(f'  "{tag}",')
out.append('''};
int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  const int dump = argc > 2 && std::strcmp(argv[1], "--dump") == 0 ? atoi(argv[2]) : -1;
  const int nv = sizeof kTags / sizeof kTags[0];
  unsigned h[3 * kN];
  unsigned long long s = 0x9E3779B97F4A7C15ull;
  const unsigned edge[] = {0u, 1u, 0x7Fu, 0x80u, 0xFFu, 0x7FFFu, 0x8000u, 0xFFFFu, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu,
                           0x01FF807Fu, 0x80017FFEu, 0x00000020u, 0x0000001Fu, 0x00000021u};
  for (int i = 0; i < 3 * kN; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    h[i] = (i < 3 * 16 * 3) ? edge[(i / 3 + i % 3 * 5) % 16] : static_cast<unsigned>(s);
  }
  for (int i = 0; i < 16; ++i) { h[3 * i] = edge[i]; h[3 * i + 1] = edge[(i * 7 + 3) % 16]; h[3 * i + 2] = edge[(i * 5 + 1) % 16]; }
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
    if (v >= (int)(sizeof kExpected / sizeof kExpected[0]) || hash != kExpected[v].hash) {
      if (bad++ < 30) printf("%s: 0x%016llx, the card gave 0x%016llx\\n", kTags[v], hash,
                             v < (int)(sizeof kExpected / sizeof kExpected[0]) ? kExpected[v].hash : 0ull);
    }
  }
  if (print) return 0;
  printf("%d video forms, %d differ\\n", nv, bad);
  puts(bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}''')
open(__import__('os').path.join(__import__('os').path.dirname(__file__) or '.', 'video_forms.cu'),'w').write('\n'.join(out)+'\n')
print(len(V), 'variants')
