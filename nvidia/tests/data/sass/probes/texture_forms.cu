// Texture fetches with the operands beyond coordinates: offsets (AOFFI, in a
// register), a depth reference (DC), tld4 on every geometry, half-precision
// results (.F16.RN, before sm_90), and the destination predicate.
#include <cuda_runtime.h>
#define T(name, ...) extern "C" __global__ void name(cudaTextureObject_t t, float* o, int* io) { float r0, r1, r2, r3; __VA_ARGS__; o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3; }
#define OUT4 : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
T(lvl_off_dc_2d, asm volatile("tex.level.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], %7, {%8,%9}, %10;" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "f"(o[12]), "r"(io[0]), "r"(io[1]), "f"(o[13])))
T(lvl_off_2d, asm volatile("tex.level.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], %7, {%8,%9};" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "f"(o[12]), "r"(io[0]), "r"(io[1])))
T(lvl_dc_2d, asm volatile("tex.level.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], %7, %8;" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "f"(o[12]), "f"(o[13])))
T(off_3d, asm volatile("tex.3d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}], {%8,%9,%10,%10};" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "f"(o[12]), "r"(io[0]), "r"(io[1]), "r"(io[2])))
T(off_a2d, asm volatile("tex.a2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}], {%8,%9};" OUT4 : "l"(t), "r"(io[5]), "f"(o[11]), "f"(o[12]), "r"(io[0]), "r"(io[1])))
T(off_1d, asm volatile("tex.1d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5}], {%6};" OUT4 : "l"(t), "f"(o[10]), "r"(io[0])))
T(off_imm_2d, asm volatile("tex.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], {-3,5};" OUT4 : "l"(t), "f"(o[10]), "f"(o[11])))
T(gat_off, asm volatile("tld4.g.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], {%7,%8};" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "r"(io[0]), "r"(io[1])))
T(gat_off_dc, asm volatile("tld4.g.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], {%7,%8}, %9;" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "r"(io[0]), "r"(io[1]), "f"(o[13])))
T(gat_dc, asm volatile("tld4.g.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], %7;" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "f"(o[13])))
T(gat_a2d, asm volatile("tld4.r.a2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}];" OUT4 : "l"(t), "r"(io[5]), "f"(o[11]), "f"(o[12])))
T(gat_cube, asm volatile("tld4.r.cube.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}];" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "f"(o[12])))
T(gat_acube, asm volatile("tld4.r.acube.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%8}];" OUT4 : "l"(t), "r"(io[5]), "f"(o[10]), "f"(o[11]), "f"(o[12])))
T(base_off, asm volatile("tex.base.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}], {%7,%8};" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "r"(io[0]), "r"(io[1])))
T(cube_dc, asm volatile("tex.cube.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6,%7,%7}], %8;" OUT4 : "l"(t), "f"(o[10]), "f"(o[11]), "f"(o[12]), "f"(o[13])))
T(pred_2d, { unsigned p; asm volatile("{ .reg .pred q; tex.2d.v4.f32.f32 {%0,%1,%2,%3}|q, [%5, {%6,%7}]; selp.u32 %4, 1, 0, q; }" : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3), "=r"(p) : "l"(t), "f"(o[10]), "f"(o[11])); io[9] = p; })
extern "C" __global__ void half_4(cudaTextureObject_t t, unsigned short* o) {
  unsigned short a, b, c, d;
  asm volatile("tex.2d.v4.f16.f32 {%0,%1,%2,%3}, [%4, {%5,%6}];" : "=h"(a), "=h"(b), "=h"(c), "=h"(d) : "l"(t), "f"(1.5f), "f"(threadIdx.x * 0.25f));
  o[0] = a; o[1] = b; o[2] = c; o[3] = d;
}
extern "C" __global__ void half_2(cudaTextureObject_t t, unsigned* o) {
  unsigned a, b;
  asm volatile("tex.1d.v2.f16x2.f32 {%0,%1}, [%2, {%3}];" : "=r"(a), "=r"(b) : "l"(t), "f"(threadIdx.x * 0.25f));
  o[0] = a; o[1] = b;
}
