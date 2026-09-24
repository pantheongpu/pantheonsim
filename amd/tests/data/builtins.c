// What a kernel gets for the library functions it calls: rounding, the sine
// and the cosine, a power of two applied to a float, the smallest and largest
// of three, bytes moved about, and the counter a wave reads to time itself.
// Built for gfx942 by build.sh.
typedef unsigned u32;
typedef unsigned long u64;

__attribute__((amdgpu_kernel)) void rounding(const float* a, float* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  float x = a[t];
  out[t] = __builtin_floorf(x) + __builtin_ceilf(x) * 2.0f + __builtin_rintf(x) * 4.0f +
           __builtin_truncf(x) * 8.0f;
}
__attribute__((amdgpu_kernel)) void turns(const float* a, float* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_sinf(a[t]) + __builtin_cosf(a[t]);
}
__attribute__((amdgpu_kernel)) void scaled(const float* a, const int* e, float* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_ldexpf(a[t], e[t]);
}
__attribute__((amdgpu_kernel)) void three(const int* a, const int* b, const int* c, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  int x = a[t], y = b[t], z = c[t];
  int lo = x < y ? x : y;
  lo = lo < z ? lo : z;
  int hi = x > y ? x : y;
  hi = hi > z ? hi : z;
  out[t] = lo + hi * 2;
}
__attribute__((amdgpu_kernel)) void three_f(const float* a, const float* b, const float* c, float* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_fminf(__builtin_fminf(a[t], b[t]), c[t]) +
           __builtin_fmaxf(__builtin_fmaxf(a[t], b[t]), c[t]);
}
__attribute__((amdgpu_kernel)) void swapped(const u32* a, u32* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  u32 x = a[t];
  out[t] = ((x & 0xff) << 24) | ((x >> 8) & 0xff) << 16 | ((x >> 16) & 0xff) << 8 | (x >> 24);
}
// four words at a time, which is one load and one store
typedef unsigned v4 __attribute__((ext_vector_type(4)));
__attribute__((amdgpu_kernel)) void four_at_a_time(const v4* a, v4* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = a[t] * 3u + 7u;
}
__attribute__((amdgpu_kernel)) void timed(u64* out) {
  u64 first = __builtin_amdgcn_s_memtime();
  u64 second = __builtin_amdgcn_s_memrealtime();
  out[0] = first;
  out[1] = second;
}
