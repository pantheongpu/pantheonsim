// sm_120's block-scaled and fp6/fp4 mma.sync forms (OMMA, QMMA and their
// .SF and .SP forms in SASS), one kernel each over the same inputs, their
// results hashed: nvidia/tests/e2e/run_sass_archs.sh runs it on SASS and
// on its PTX, which must agree. Generated; the scale selectors vary.
#include <cstdio>
#include <cuda_runtime.h>

__device__ unsigned mix(unsigned h, unsigned v) { return (h ^ v) * 16777619u; }

__global__ void k_f8f6f4_e4m3_e4m3(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e4m3_e2m1(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e5m2_e4m3(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e5m2_e2m1(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e3m2_e4m3(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e3m2_e2m1(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e2m3_e4m3(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e2m3_e2m1(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e2m1_e4m3(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_f8f6f4_e2m1_e2m1(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e4m3_e4m3_s0000(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e4m3_e4m3_s1123(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {1, 1}, %15, {2, 3};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e4m3_e4m3_s3012(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {3, 0}, %15, {1, 2};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e2m1_e3m2_s0000(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e3m2.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e2m1_e3m2_s1123(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e3m2.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {1, 1}, %15, {2, 3};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e2m1_e3m2_s3012(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e3m2.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {3, 0}, %15, {1, 2};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e5m2_e2m3_s0000(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e5m2.e2m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e5m2_e2m3_s1123(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e5m2.e2m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {1, 1}, %15, {2, 3};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mx8_e5m2_e2m3_s3012(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e5m2.e2m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {3, 0}, %15, {1, 2};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mxf4_s0000(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_mxf4_s2123(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {2, 1}, %15, {2, 3};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_nvf4_2X_ue8m0_s0000(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_nvf4_2X_ue8m0_s2121(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {2, 1}, %15, {2, 1};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_nvf4_4X_ue4m3_s0000(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_nvf4_4X_ue4m3_s0103(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 1}, %15, {0, 3};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_sp_mx8_s0000(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  unsigned b2 = in[l + 288], b3 = in[l + 320];
  asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, %17, {0, 0}, %18, {0, 0};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(meta), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

__global__ void k_sp_mx8_s1132(const unsigned* in, unsigned* out) {
  const unsigned l = threadIdx.x;
  unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96];
  unsigned b0 = in[l + 128], b1 = in[l + 160], sa = in[l + 192], sb = in[l + 224], meta = in[l + 256];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, c0 = 0.5f * l, c1 = 1.0f, c2 = -2.0f, c3 = 0.25f;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  unsigned b2 = in[l + 288], b3 = in[l + 320];
  asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, %17, {1, 1}, %18, {3, 2};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(meta), "r"(sa), "r"(sb));
#endif
  unsigned h = 2166136261u;
  h = mix(h, __float_as_uint(d0)); h = mix(h, __float_as_uint(d1)); h = mix(h, __float_as_uint(d2)); h = mix(h, __float_as_uint(d3));
  out[l] = h;
}

int main() {
  unsigned h_in[352];
  for (unsigned i = 0; i < 352; ++i) h_in[i] = (i * 2654435761u) ^ (i >> 3) * 40503u;
  // Scale factors: ue8m0 near 127 (2^-4..2^3), ue4m3 small normals; the
  // sparse metadata a valid ordered pattern (index pairs 0,1 / 1,2 / 0,3 / 2,3).
  for (unsigned i = 192; i < 256; ++i) {
    unsigned v = 0;
    for (int b = 0; b < 4; ++b) v |= ((123u + (i * 7 + b * 3) % 8) & 0xFF) << (8 * b);
    h_in[i] = v;
  }
  for (unsigned i = 256; i < 288; ++i) h_in[i] = 0x4E4E4E4Eu ^ (i & 1 ? 0x0A0A0A0Au : 0u);
  unsigned *d_in, *d_out, h_out[32];
  cudaMalloc(&d_in, sizeof h_in);
  cudaMalloc(&d_out, sizeof h_out);
  cudaMemcpy(d_in, h_in, sizeof h_in, cudaMemcpyHostToDevice);
  unsigned long long all = 0;
  k_f8f6f4_e4m3_e4m3<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e4m3_e4m3         %016llx\n", f); all ^= f; }
  k_f8f6f4_e4m3_e2m1<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e4m3_e2m1         %016llx\n", f); all ^= f; }
  k_f8f6f4_e5m2_e4m3<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e5m2_e4m3         %016llx\n", f); all ^= f; }
  k_f8f6f4_e5m2_e2m1<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e5m2_e2m1         %016llx\n", f); all ^= f; }
  k_f8f6f4_e3m2_e4m3<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e3m2_e4m3         %016llx\n", f); all ^= f; }
  k_f8f6f4_e3m2_e2m1<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e3m2_e2m1         %016llx\n", f); all ^= f; }
  k_f8f6f4_e2m3_e4m3<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e2m3_e4m3         %016llx\n", f); all ^= f; }
  k_f8f6f4_e2m3_e2m1<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e2m3_e2m1         %016llx\n", f); all ^= f; }
  k_f8f6f4_e2m1_e4m3<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e2m1_e4m3         %016llx\n", f); all ^= f; }
  k_f8f6f4_e2m1_e2m1<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("f8f6f4_e2m1_e2m1         %016llx\n", f); all ^= f; }
  k_mx8_e4m3_e4m3_s0000<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e4m3_e4m3_s0000      %016llx\n", f); all ^= f; }
  k_mx8_e4m3_e4m3_s1123<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e4m3_e4m3_s1123      %016llx\n", f); all ^= f; }
  k_mx8_e4m3_e4m3_s3012<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e4m3_e4m3_s3012      %016llx\n", f); all ^= f; }
  k_mx8_e2m1_e3m2_s0000<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e2m1_e3m2_s0000      %016llx\n", f); all ^= f; }
  k_mx8_e2m1_e3m2_s1123<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e2m1_e3m2_s1123      %016llx\n", f); all ^= f; }
  k_mx8_e2m1_e3m2_s3012<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e2m1_e3m2_s3012      %016llx\n", f); all ^= f; }
  k_mx8_e5m2_e2m3_s0000<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e5m2_e2m3_s0000      %016llx\n", f); all ^= f; }
  k_mx8_e5m2_e2m3_s1123<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e5m2_e2m3_s1123      %016llx\n", f); all ^= f; }
  k_mx8_e5m2_e2m3_s3012<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mx8_e5m2_e2m3_s3012      %016llx\n", f); all ^= f; }
  k_mxf4_s0000<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mxf4_s0000               %016llx\n", f); all ^= f; }
  k_mxf4_s2123<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("mxf4_s2123               %016llx\n", f); all ^= f; }
  k_nvf4_2X_ue8m0_s0000<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("nvf4_2X_ue8m0_s0000      %016llx\n", f); all ^= f; }
  k_nvf4_2X_ue8m0_s2121<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("nvf4_2X_ue8m0_s2121      %016llx\n", f); all ^= f; }
  k_nvf4_4X_ue4m3_s0000<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("nvf4_4X_ue4m3_s0000      %016llx\n", f); all ^= f; }
  k_nvf4_4X_ue4m3_s0103<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("nvf4_4X_ue4m3_s0103      %016llx\n", f); all ^= f; }
  k_sp_mx8_s0000<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("sp_mx8_s0000             %016llx\n", f); all ^= f; }
  k_sp_mx8_s1132<<<1, 32>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  { unsigned long long f = 1469598103934665603ULL; for (unsigned v : h_out) f = (f ^ v) * 1099511628211ULL;
    std::printf("sp_mx8_s1132             %016llx\n", f); all ^= f; }
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  std::printf("all %016llx\nPASS\n", all);
  return 0;
}
