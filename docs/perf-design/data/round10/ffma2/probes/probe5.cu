#include <cstdint>
#include <cuda_runtime.h>
// Per-half sign modifiers on each operand: which of them ptxas folds into NP forms.
#define HM(name, body) __device__ __forceinline__ uint64_t name(uint64_t a) { uint64_t d; asm volatile("{ .reg .b32 l,h; mov.b64 {l,h}, %1; " body " mov.b64 %0, {l,h}; }" : "=l"(d) : "l"(a)); return d; }
HM(nl, "neg.f32 l,l;")
HM(nh, "neg.f32 h,h;")
HM(al, "abs.f32 l,l;")
HM(ah, "abs.f32 h,h;")
HM(nal, "abs.f32 l,l; neg.f32 l,l;")
HM(nah, "abs.f32 h,h; neg.f32 h,h;")
HM(nl_ah, "neg.f32 l,l; abs.f32 h,h;")
HM(al_nh, "abs.f32 l,l; neg.f32 h,h;")
HM(nl_nah, "neg.f32 l,l; abs.f32 h,h; neg.f32 h,h;")
HM(nal_nh, "abs.f32 l,l; neg.f32 l,l; neg.f32 h,h;")
HM(al_nah, "abs.f32 l,l; abs.f32 h,h; neg.f32 h,h;")
HM(nal_ah, "abs.f32 l,l; neg.f32 l,l; abs.f32 h,h;")
HM(nl_nh, "neg.f32 l,l; neg.f32 h,h;")
HM(a_both, "abs.f32 l,l; abs.f32 h,h;")
HM(na_both, "abs.f32 l,l; abs.f32 h,h; neg.f32 l,l; neg.f32 h,h;")
__device__ __forceinline__ uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) { uint64_t d; asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
__device__ __forceinline__ uint64_t add2(uint64_t a, uint64_t b) { uint64_t d; asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
__device__ __forceinline__ uint64_t mul2(uint64_t a, uint64_t b) { uint64_t d; asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
extern "C" __global__ void k_c_nl(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nl(in[3*i+2])); }
extern "C" __global__ void k_b_nl(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nl(in[3*i+1])); }
extern "C" __global__ void k_a_nl(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nl(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nh(in[3*i+2])); }
extern "C" __global__ void k_b_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nh(in[3*i+1])); }
extern "C" __global__ void k_a_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nh(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_al(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], al(in[3*i+2])); }
extern "C" __global__ void k_b_al(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], al(in[3*i+1])); }
extern "C" __global__ void k_a_al(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(al(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], ah(in[3*i+2])); }
extern "C" __global__ void k_b_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], ah(in[3*i+1])); }
extern "C" __global__ void k_a_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(ah(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_nal(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nal(in[3*i+2])); }
extern "C" __global__ void k_b_nal(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nal(in[3*i+1])); }
extern "C" __global__ void k_a_nal(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nal(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nah(in[3*i+2])); }
extern "C" __global__ void k_b_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nah(in[3*i+1])); }
extern "C" __global__ void k_a_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nah(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_nl_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nl_ah(in[3*i+2])); }
extern "C" __global__ void k_b_nl_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nl_ah(in[3*i+1])); }
extern "C" __global__ void k_a_nl_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nl_ah(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_al_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], al_nh(in[3*i+2])); }
extern "C" __global__ void k_b_al_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], al_nh(in[3*i+1])); }
extern "C" __global__ void k_a_al_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(al_nh(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_nl_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nl_nah(in[3*i+2])); }
extern "C" __global__ void k_b_nl_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nl_nah(in[3*i+1])); }
extern "C" __global__ void k_a_nl_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nl_nah(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_nal_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nal_nh(in[3*i+2])); }
extern "C" __global__ void k_b_nal_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nal_nh(in[3*i+1])); }
extern "C" __global__ void k_a_nal_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nal_nh(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_al_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], al_nah(in[3*i+2])); }
extern "C" __global__ void k_b_al_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], al_nah(in[3*i+1])); }
extern "C" __global__ void k_a_al_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(al_nah(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_nal_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nal_ah(in[3*i+2])); }
extern "C" __global__ void k_b_nal_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nal_ah(in[3*i+1])); }
extern "C" __global__ void k_a_nal_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nal_ah(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_nl_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], nl_nh(in[3*i+2])); }
extern "C" __global__ void k_b_nl_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], nl_nh(in[3*i+1])); }
extern "C" __global__ void k_a_nl_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(nl_nh(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_a_both(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], a_both(in[3*i+2])); }
extern "C" __global__ void k_b_a_both(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], a_both(in[3*i+1])); }
extern "C" __global__ void k_a_a_both(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(a_both(in[3*i]), in[3*i+1]); }
extern "C" __global__ void k_c_na_both(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = fma2(in[3*i], in[3*i+1], na_both(in[3*i+2])); }
extern "C" __global__ void k_b_na_both(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = add2(in[3*i], na_both(in[3*i+1])); }
extern "C" __global__ void k_a_na_both(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; out[i] = mul2(na_both(in[3*i]), in[3*i+1]); }
