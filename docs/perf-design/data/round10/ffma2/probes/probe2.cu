#include <cstdint>
#include <cuda_runtime.h>
#define PK(name, body) __device__ __forceinline__ uint64_t name(uint64_t a) { uint64_t d; asm volatile("{ .reg .b32 l,h; mov.b64 {l,h}, %1; " body " }" : "=l"(d) : "l"(a)); return d; }
PK(bcast_lo, "mov.b64 %0, {l,l};")
PK(bcast_hi, "mov.b64 %0, {h,h};")
PK(swap,     "mov.b64 %0, {h,l};")
PK(neg_both, "neg.f32 l,l; neg.f32 h,h; mov.b64 %0, {l,h};")
PK(neg_lo,   "neg.f32 l,l; mov.b64 %0, {l,h};")
PK(neg_hi,   "neg.f32 h,h; mov.b64 %0, {l,h};")
PK(abs_both, "abs.f32 l,l; abs.f32 h,h; mov.b64 %0, {l,h};")
PK(nabs_both,"abs.f32 l,l; abs.f32 h,h; neg.f32 l,l; neg.f32 h,h; mov.b64 %0, {l,h};")
__device__ __forceinline__ uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) { uint64_t d; asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
__device__ __forceinline__ uint64_t mul2(uint64_t a, uint64_t b) { uint64_t d; asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
__device__ __forceinline__ uint64_t add2(uint64_t a, uint64_t b) { uint64_t d; asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
extern "C" __global__ void probe2(const uint64_t* in, uint64_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  uint64_t a = in[3*i], b = in[3*i+1], c = in[3*i+2]; uint64_t* o = out + 20*i;
  o[0]=fma2(bcast_lo(a),b,c); o[1]=fma2(bcast_hi(a),b,c); o[2]=fma2(swap(a),b,c);
  o[3]=fma2(a,bcast_lo(b),c); o[4]=fma2(a,swap(b),c); o[5]=fma2(a,b,bcast_hi(c)); o[6]=fma2(a,b,swap(c));
  o[7]=fma2(neg_both(a),b,c); o[8]=fma2(a,neg_both(b),c); o[9]=fma2(a,b,neg_both(c));
  o[10]=fma2(neg_lo(a),b,c); o[11]=fma2(a,neg_hi(b),c); o[12]=fma2(abs_both(a),b,c); o[13]=fma2(a,nabs_both(b),c);
  o[14]=mul2(neg_both(a),b); o[15]=add2(a,neg_both(b)); o[16]=add2(abs_both(a),swap(b)); o[17]=mul2(bcast_lo(a),swap(b));
  o[18]=add2(bcast_hi(a),bcast_lo(b)); o[19]=mul2(a,nabs_both(b));
}
