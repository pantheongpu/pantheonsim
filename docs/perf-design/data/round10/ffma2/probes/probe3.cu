#include <cstdint>
#include <cuda_runtime.h>
#define OP3(name, text) __device__ __forceinline__ uint64_t name(uint64_t a, uint64_t b, uint64_t c) { uint64_t d; asm(text : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
OP3(fma2, "fma.rn.f32x2 %0, %1, %2, %3;")
OP3(fma2f, "fma.rn.ftz.f32x2 %0, %1, %2, %3;")
__device__ __forceinline__ uint64_t mul2(uint64_t a, uint64_t b) { uint64_t d; asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
__device__ __forceinline__ uint64_t add2(uint64_t a, uint64_t b) { uint64_t d; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
extern "C" __global__ void probe3(const uint64_t* in, uint64_t* out, uint64_t k0, uint64_t k1, uint32_t kk, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  uint64_t a = in[3*i], b = in[3*i+1], c = in[3*i+2]; uint64_t* o = out + 24*i;
  o[0] = fma2(a, k0, c);            // const bank / uniform as b
  o[1] = fma2(a, b, k1);            // as c
  o[2] = fma2(k0, b, c);            // as a
  o[3] = fma2(a, 0x3f8000003f800000ull, c);   // immediates
  o[4] = fma2(a, b, 0x4000000040000000ull);
  o[5] = fma2(0x40a0000040a00000ull, b, c);
  o[6] = mul2(a, k0);  o[7] = mul2(a, 0x3f0000003f000000ull); o[8] = mul2(k1, b);
  o[9] = add2(a, k0);  o[10] = add2(a, 0x3f8000003f800000ull); o[11] = add2(k1, b);
  o[12] = add2(a, 0);  o[13] = mul2(a, 0x40000000bf800000ull);
  uint64_t u = ((uint64_t)kk << 32) | kk;
  o[14] = fma2(a, u, c); o[15] = mul2(u, b); o[16] = add2(a, u);
  o[17] = fma2(a, b, u);
  o[18] = fma2(a, b, a);  o[19] = fma2(a, a, a); o[20] = mul2(a, a); o[21] = add2(a, a);
  if (blockIdx.x == 3) o[22] = fma2f(a, b, c);
  o[23] = fma2(a, (uint64_t)blockIdx.x * 0x100000001ull, c);
}
