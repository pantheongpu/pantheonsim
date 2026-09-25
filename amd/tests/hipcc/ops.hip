// What the pantheon workloads' kernels compile to, beyond what the other
// fixtures cover: the instructions ROCm 7's compiler emits for the idioms the
// workloads use -- a wave adding one value for all its active lanes, 64-bit
// scalar arithmetic, a 64-bit compare-and-swap, halves built from floats, the
// comparisons a NaN answers "yes" to, a memory fence, a condition every lane
// agrees on, and a loop each lane leaves at its own time. The last kernel
// writes a report to the host with printf; it is here to be decoded, and the
// tests do not run it.
//
// Built by build.sh with hipcc, which needs ROCm; the code object and the
// assembler's listing of it are checked in.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef unsigned long long u64;

// the wave adds one value for all its active lanes
__global__ void count_active(unsigned* counter, const int* in) {
  if (in[threadIdx.x] > 0) atomicAdd(counter, 1u);
}
// 64-bit scalar arithmetic, and a vector shift
__global__ void wide(u64 a, u64 b, u64* out, const u64* v) {
  const int t = threadIdx.x;
  out[t] = (a - b) + (v[t] >> (t & 63)) + (a >> 8);
}
// bytes to floats, a reciprocal square root, a not, 24-bit multiplies
__global__ void misc(const unsigned* a, const float* f, float* out, unsigned* o2) {
  const int t = threadIdx.x;
  out[t] = static_cast<float>(a[t] & 0xff) + __frsqrt_rn(f[t]) + f[t] * f[(t + 1) & 63] + 0.001f;
  o2[t] = ~a[t] + __umul24(a[t], a[(t + 1) & 63]) + ((a[t] + a[(t + 2) & 63]) << 3) + ((a[t] & 0xf0f0f0f0u) | 32);
}
// a 64-bit compare-and-swap, and three words at a time
__global__ void cas64(u64* p, const float3* in, float3* out) {
  const int t = threadIdx.x;
  const u64 old = p[t];
  atomicCAS(&p[t], old, old * 3 + 1);
  float3 v = in[t];
  out[t] = make_float3(v.x * 2.0f, v.y * 2.0f, v.z * 2.0f);
}
// a float halved into both halves of a pair
__global__ void split(const float* a, __half2* out) {
  const int t = threadIdx.x;
  const __half h = __float2half(a[t] * 0.5f);
  out[t] = __halves2half2(h, h);
}
// two halves packed
__global__ void pack(const float* a, const float* b, __half2* out) {
  const int t = threadIdx.x;
  out[t] = __halves2half2(__float2half(a[t]), __float2half(b[t]));
}
// comparisons a NaN answers yes to
__global__ void nan_cmp(const float* a, const float* b, int* out) {
  const int t = threadIdx.x;
  out[t] = !(a[t] > b[t]) + 2 * !(a[t] < b[t]) + 4 * (a[t] != b[t]) + 8 * (a[t] == b[t]);
}
// a fence between two stores
__global__ void fenced(int* p) {
  p[threadIdx.x] = threadIdx.x;
  __threadfence();
  p[threadIdx.x + 64] = threadIdx.x;
}
// scalar comparisons and a branch on VCC
__global__ void branches(const int* a, int* out, int k) {
  const int t = threadIdx.x;
  int r = 0;
  if (k > 7) r += 1;
  if (k >= 3) r += 2;
  if (k != 5) r += 4;
  if (k == 500) r += 8;
  if (static_cast<unsigned>(k) > 9u) r += 16;
  out[t] = r + (a[t] < 4 ? 32 : 0);
}
// the rest of the idioms the workloads use
__global__ void more(const unsigned* a, const unsigned* b, const float* f, const float* g, unsigned* o, float* of,
                     unsigned k, u64 x, u64 y) {
  const int t = threadIdx.x;
  o[t] = ~a[t];
  o[t + 64] = __umul24(a[t], b[t]) + b[(t + 1) & 63];
  o[t + 128] = 1u << (a[t] & 31);
  o[t + 192] = (a[t] & k) | b[t];
  o[t + 256] = static_cast<unsigned>((static_cast<u64>(k) << 32 | k) - a[t]);
  of[t] = f[t] * g[t] + 0.123f;
  of[t + 64] = fabsf(f[t]) * g[t];
  if (x == y) of[t + 128] = 1.0f;
  if ((k << 2) >= static_cast<unsigned>(t)) of[t + 192] = 2.0f;
}
// a condition every lane agrees on, worked out by the vector unit
__global__ void uniform_float(float fa, float fb, int* out) {
  int r = 0;
  if (fa > fb) r = 1;
  else if (fa < fb * 0.5f) r = 2;
  out[threadIdx.x] = r;
}
// halves built from floats and packed
__global__ void halves(const float* a, const float* b, __half2* out) {
  const int t = threadIdx.x;
  out[t] = __floats2half2_rn(a[t] * 0.5f, b[t] * 0.25f);
  __half2 r;
  r.x = __float2half(a[t]);
  r.y = __float2half(b[t] + 1.0f);
  out[t + 64] = r;
}
// a report the device writes to the host, which is decoded and not run here
__global__ void report(const int* a) {
  if (a[threadIdx.x] < 0) printf("negative at %d: %d\n", threadIdx.x, a[threadIdx.x]);
}
// signed scalar comparisons, a condition every lane agrees on, and work in
// both arms of a divergent branch
__global__ void shapes(const int* a, int* out, int k, int m, float fa, float fb, unsigned long long base) {
  const int t = threadIdx.x;
  int r = 0;
  if (k >= m) r += 1;
  if (!(fa >= fb)) r += 2;
  if (a[t] & 1) r += a[t] * 3;
  else r -= a[t] / 7;
  out[t] = r;
  reinterpret_cast<unsigned long long*>(out)[64 + t] = base - static_cast<unsigned long long>(a[t]);
}
// a loop each lane leaves at its own time
__global__ void walk(const int* next, int* out) {
  int at = threadIdx.x, steps = 0;
  while (at >= 0 && steps < 64) {
    at = next[at];
    ++steps;
    if (at == 13) break;
  }
  out[threadIdx.x] = steps;
}
