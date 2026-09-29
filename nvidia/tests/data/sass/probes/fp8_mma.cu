// Decoder probe: m16n8k32 FP8 MMA (QMMA) and FP8 conversions (F2FP). sm_89+.
#define M(i, t) asm volatile("mma.sync.aligned.m16n8k32.row.col.f32." t ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" \
  : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])); o[i] = d[0]+d[1]+d[2]+d[3];
#define H(i, t) asm volatile("mma.sync.aligned.m16n8k32.row.col.f16." t ".f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};" \
  : "=r"(h[0]), "=r"(h[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(hc[0]), "r"(hc[1])); o[i] = h[0]+h[1];
__global__ void k(const unsigned* g, float* o) {
  unsigned a[4], b[2], h[2], hc[2] = {g[9], g[10]}; float c[4], d[4];
  for (int i = 0; i < 4; ++i) { a[i] = g[i]; c[i] = __uint_as_float(g[i+4]); }
  b[0] = g[7]; b[1] = g[8];
  M(0, "e4m3.e4m3") M(1, "e4m3.e5m2") M(2, "e5m2.e4m3") M(3, "e5m2.e5m2")
  H(4, "e4m3.e4m3") H(5, "e5m2.e5m2")
  unsigned short p; float x = o[0], y = o[1];
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(p) : "f"(x), "f"(y)); o[8] = p;
  asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(p) : "f"(x), "f"(y)); o[9] = p;
  unsigned r; asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(p)); o[10] = r;
  asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(r) : "h"(p)); o[11] = r;
}
