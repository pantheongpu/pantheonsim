// Probe: packed fp32 instructions (PTX f32x2, SASS FFMA2/FADD2/FMUL2). Each thread applies each form to loaded pairs and stores the result.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define F2(name, ...) __device__ __forceinline__ uint64_t name(uint64_t a, uint64_t b, uint64_t c) { uint64_t d; asm volatile(__VA_ARGS__ : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
F2(fma_rn,  "fma.rn.f32x2 %0, %1, %2, %3;")
F2(fma_rz,  "fma.rz.f32x2 %0, %1, %2, %3;")
F2(fma_rm,  "fma.rm.f32x2 %0, %1, %2, %3;")
F2(fma_rp,  "fma.rp.f32x2 %0, %1, %2, %3;")
F2(fma_ftz, "fma.rn.ftz.f32x2 %0, %1, %2, %3;")
F2(add_rn,  "add.rn.f32x2 %0, %1, %2; // %3")
F2(add_rz,  "add.rz.f32x2 %0, %1, %2; // %3")
F2(add_ftz, "add.rn.ftz.f32x2 %0, %1, %2; // %3")
F2(mul_rn,  "mul.rn.f32x2 %0, %1, %2; // %3")
F2(mul_rp,  "mul.rp.f32x2 %0, %1, %2; // %3")
F2(mul_ftz, "mul.rn.ftz.f32x2 %0, %1, %2; // %3")
extern "C" __global__ void probe(const uint64_t* in, uint64_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
  uint64_t* o = out + 11 * i;
  o[0] = fma_rn(a, b, c);  o[1] = fma_rz(a, b, c);  o[2] = fma_rm(a, b, c);  o[3] = fma_rp(a, b, c); o[4] = fma_ftz(a, b, c);
  o[5] = add_rn(a, b, c);  o[6] = add_rz(a, b, c);  o[7] = add_ftz(a, b, c);
  o[8] = mul_rn(a, b, c);  o[9] = mul_rp(a, b, c);  o[10] = mul_ftz(a, b, c);
}
