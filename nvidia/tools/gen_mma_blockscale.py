#!/usr/bin/env python3
# Writes nvidia/tests/e2e/mma_blockscale.cu (python3 nvidia/tools/gen_mma_blockscale.py <out>): sm_120's block-scaled and
# fp6/fp4 mma.sync forms, one kernel each, their results hashed for the
# SASS run to match the PTX one.
import sys
forms = []  # (name, ptx instruction, shape k, a regs, b regs, sparse, scale kind)
for ta in ["e4m3", "e5m2", "e3m2", "e2m3", "e2m1"]:
    for tb in ["e4m3", "e2m1"]:
        forms.append((f"f8f6f4_{ta}_{tb}", f"mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.{ta}.{tb}.f32", 4, 2, None))
for ta, tb in [("e4m3", "e4m3"), ("e2m1", "e3m2"), ("e5m2", "e2m3")]:
    for sel in [(0, 0, 0, 0), (1, 1, 2, 3), (3, 0, 1, 2)]:
        forms.append((f"mx8_{ta}_{tb}_s{''.join(map(str, sel))}",
                      f"mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.{ta}.{tb}.f32.ue8m0", 4, 2, ("scale", sel)))
for sel in [(0, 0, 0, 0), (2, 1, 2, 3)]:
    forms.append((f"mxf4_s{''.join(map(str, sel))}",
                  "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 4, 2, ("scale", sel)))
for vec, st, sels in [("2X", "ue8m0", [(0, 0, 0, 0), (2, 1, 2, 1)]), ("4X", "ue4m3", [(0, 0, 0, 0), (0, 1, 0, 3)])]:
    for sel in sels:
        forms.append((f"nvf4_{vec}_{st}_s{''.join(map(str, sel))}",
                      f"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::{vec}.f32.e2m1.e2m1.f32.{st}", 4, 2, ("scale", sel)))
sparse = []
for sel in [(0, 0, 0, 0), (1, 1, 3, 2)]:
    sparse.append((f"sp_mx8_s{''.join(map(str, sel))}",
                   "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0", sel))

out = []
w = out.append
w("// sm_120's block-scaled and fp6/fp4 mma.sync forms (OMMA, QMMA and their")
w("// .SF and .SP forms in SASS), one kernel each over the same inputs, their")
w("// results hashed: nvidia/tests/e2e/run_sass_archs.sh runs it on SASS and")
w("// on its PTX, which must agree. Generated; the scale selectors vary.")
w("#include <cstdio>")
w("#include <cuda_runtime.h>")
w("")
w("__device__ unsigned mix(unsigned h, unsigned v) { return (h ^ v) * 16777619u; }")
w("")
def kernel(name, instr, sel, sp):
    w(f"__global__ void k_{name}(const unsigned* in, unsigned* out) {{")
    w("  const unsigned l = threadIdx.x;")
    w("  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];")
    w("  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];")
    w("  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;")
    w("#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)")
    if sel is None:
        w(f'  asm volatile("{instr} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%10,%11,%12,%13}};"')
        w('               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),')
        w('                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));')
    elif not sp:
        s = sel
        w(f'  asm volatile("{instr} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%10,%11,%12,%13}}, %14, {{{s[0]}, {s[1]}}}, %15, {{{s[2]}, {s[3]}}};"')
        w('               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),')
        w('                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));')
    else:
        s = sel
        w("  unsigned b2 = in[l + 288], b3 = in[l + 320];")
        w(f'  asm volatile("{instr} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9,%10,%11}}, {{%12,%13,%14,%15}}, %16, 0x0, %17, {{{s[0]}, {s[1]}}}, %18, {{{s[2]}, {s[3]}}};"')
        w('               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3),')
        w('                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(meta), "r"(sa), "r"(sb));')
    w("#endif")
    w("  unsigned h = 2166136261u;")
    w("  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));")
    w("  out[l] = h;")
    w("}")
    w("")
for n, instr, *_ , scale in [(f[0], f[1], f[2], f[3], f[4]) for f in forms]:
    kernel(n, instr, None if scale is None else scale[1], False)
for n, instr, sel in sparse:
    kernel(n, instr, sel, True)
names = [f[0] for f in forms] + [s[0] for s in sparse]
w("int main() {")
w("  unsigned h_in[352];")
w("  for (unsigned i = 0; i < 352; ++i) h_in[i] = (i * 2654435761u) ^ (i >> 3) * 40503u;")
w("  // Scale factors: ue8m0 near 127 (2^-4..2^3), ue4m3 small normals; the")
w("  // sparse metadata a valid ordered pattern (index pairs 0,1 / 1,2 / 0,3 / 2,3).")
w("  for (unsigned i = 192; i < 256; ++i) {")
w("    unsigned v = 0;")
w("    for (int b = 0; b < 4; ++b) v |= ((123u + (i * 7 + b * 3) % 8) & 0xFF) << (8 * b);")
w("    h_in[i] = v;")
w("  }")
w("  for (unsigned i = 256; i < 288; ++i) h_in[i] = 0x4E4E4E4Eu ^ (i & 1 ? 0x0A0A0A0Au : 0u);")
w("  unsigned *d_in, *d_out, h_out[32];")
w("  cudaMalloc(&d_in, sizeof h_in);")
w("  cudaMalloc(&d_out, sizeof h_out);")
w("  cudaMemcpy(d_in, h_in, sizeof h_in, cudaMemcpyHostToDevice);")
w("  unsigned long long all = 0;")
for n in names:
    w(f"  k_{n}<<<1, 32>>>(d_in, d_out);")
    w("  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);")
    w("  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;")
    w(f'    std::printf("{n:<24} %016llx\\n", f); all ^= f; }}')
w('  const cudaError_t e = cudaDeviceSynchronize();')
w('  if (e != cudaSuccess) { std::printf("FAIL: %s\\n", cudaGetErrorString(e)); return 1; }')
w('  std::printf("all %016llx\\nPASS\\n", all);')
w("  return 0;")
w("}")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
