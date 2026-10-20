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

#define PK(name, body) __device__ __forceinline__ uint64_t name(uint64_t a) { uint64_t d; asm volatile("{ .reg .b32 l,h; mov.b64 {l,h}, %1; " body " }" : "=l"(d) : "l"(a)); return d; }
PK(swap, "mov.b64 %0, {h,l};")
PK(bl, "mov.b64 %0, {l,l};")
PK(bh, "mov.b64 %0, {h,h};")
extern "C" __global__ void k_c_swap_nl(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], swap(nl(c))); }
extern "C" __global__ void k_c_swap_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], swap(nh(c))); }
extern "C" __global__ void k_c_nl_swap(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], nl(swap(c))); }
extern "C" __global__ void k_c_nh_swap(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], nh(swap(c))); }
extern "C" __global__ void k_c_swap_nal_ah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], swap(nal_ah(c))); }
extern "C" __global__ void k_c_swap_al_nah(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], swap(al_nah(c))); }
extern "C" __global__ void k_c_bl_neg(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], nl_nh(bl(c))); }
extern "C" __global__ void k_c_bh_neg(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], nl_nh(bh(c))); }
extern "C" __global__ void k_c_bl_nh(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], nh(bl(c))); }
extern "C" __global__ void k_c_bl_a(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], a_both(bl(c))); }
extern "C" __global__ void k_c_bh_na(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], na_both(bh(c))); }
extern "C" __global__ void k_c_swap_neg(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], nl_nh(swap(c))); }
extern "C" __global__ void k_c_swap_a(const uint64_t* in, uint64_t* out) { int i = threadIdx.x; uint64_t c = in[3*i+2]; out[i] = fma2(in[3*i], in[3*i+1], a_both(swap(c))); }
