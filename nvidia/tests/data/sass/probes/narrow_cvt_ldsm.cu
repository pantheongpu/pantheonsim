// sm_120a: the packed narrow conversions (F2FP to and from E2M1, E3M2, E2M3,
// E4M3, E5M2 and UE8M0) and ldmatrix's 8-bit and unpacking forms (LDSM
// .U4x16P64TO8, .U6x16P32TO8, .8 with .M816 and .MT1616), which CUTLASS's
// f8f6f4 and block-scaled GEMMs use. nvcc -arch=sm_120a -cubin.
__global__ void narrow(const float* x, unsigned* o) {
  const float a = x[threadIdx.x], b = x[threadIdx.x + 32];
  unsigned r[16];
  asm volatile("{ .reg .b8 c0, c1; .reg .b16 h1, h2, h3, h4, h5;\n"
               "cvt.rn.satfinite.e2m1x2.f32 c0, %16, %17;\n"
               "cvt.rn.relu.satfinite.e2m1x2.f32 c1, %16, %17;\n"
               "cvt.rn.satfinite.e3m2x2.f32 h1, %16, %17;\n"
               "cvt.rn.satfinite.e2m3x2.f32 h2, %16, %17;\n"
               "cvt.rn.satfinite.e4m3x2.f32 h3, %16, %17;\n"
               "cvt.rn.satfinite.e5m2x2.f32 h4, %16, %17;\n"
               "cvt.rz.satfinite.ue8m0x2.f32 h5, %16, %17;\n"
               "cvt.rn.f16x2.e2m1x2 %0, c0;\n"
               "cvt.rn.f16x2.e3m2x2 %1, h1;\n"
               "cvt.rn.f16x2.e2m3x2 %2, h2;\n"
               "cvt.rn.f16x2.e4m3x2 %3, h3;\n"
               "cvt.rn.f16x2.e5m2x2 %4, h4;\n"
               "cvt.rn.bf16x2.ue8m0x2 %5, h5;\n"
               "mov.b32 %6, {c0, c1, c0, c1};\n"
               "mov.b32 %7, {h1, h2};\n"
               "mov.b32 %8, {h3, h4};\n"
               "cvt.u32.u16 %9, h5; }"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "f"(a), "f"(b));
  unsigned s = 0;
  for (int i = 0; i < 10; ++i) s += r[i];
  o[threadIdx.x] = s;
}

__global__ void ldsm(unsigned* o) {
  __shared__ __align__(16) unsigned char sm[1024];
  for (int i = threadIdx.x; i < 1024; i += 32) sm[i] = static_cast<unsigned char>(i);
  __syncwarp();
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(sm)) + threadIdx.x * 16;
  unsigned r[4], s = 0;
  asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a)); s += r[0] + r[1] + r[2] + r[3];
  asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a)); s += r[0] + r[1] + r[2] + r[3];
  asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a)); s += r[0] + r[1];
  asm volatile("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {%0}, [%1];" : "=r"(r[0]) : "r"(a)); s += r[0];
  asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a)); s += r[0] + r[1];
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a)); s += r[0] + r[1] + r[2] + r[3];
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a)); s += r[0] + r[1] + r[2] + r[3];
  asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a)); s += r[0] + r[1];
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a)); s += r[0] + r[1];
  o[threadIdx.x] = s;
}
