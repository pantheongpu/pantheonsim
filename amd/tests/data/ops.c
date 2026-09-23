// Kernels that exercise the instructions a real kernel uses -- integer and
// float math, division, conversions, loops, two-dimensional grids, atomics
// and 64-bit values -- so the decoder and the executor are checked against
// the compiler's own output and against what the C means. Built for gfx942 by
// build.sh; no ROCm needed.
typedef unsigned int uint;
__attribute__((amdgpu_kernel)) void int_math(const int* a, const int* b, int* out, int n) {
  int i = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
  if (i < n) {
    int x = a[i], y = b[i];
    out[i] = (x * y) + (x / 3) + (y % 5) - (x & y) + (x | y) ^ (x >> 2);
  }
}
__attribute__((amdgpu_kernel)) void float_math(const float* a, const float* b, float* out, int n) {
  int i = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
  if (i < n) {
    float x = a[i], y = b[i];
    out[i] = x * y + x / y + (x < y ? x : y) + __builtin_fmaf(x, y, 1.0f);
  }
}
__attribute__((amdgpu_kernel)) void loop_sum(const float* in, float* out, int n) {
  int i = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
  float s = 0;
  for (int k = 0; k < n; ++k) s += in[k] * (float)(k + i);
  out[i] = s;
}
__attribute__((amdgpu_kernel)) void grid2d(const float* in, float* out, int w) {
  uint x = __builtin_amdgcn_workgroup_id_x() * 8 + __builtin_amdgcn_workitem_id_x();
  uint y = __builtin_amdgcn_workgroup_id_y() * 8 + __builtin_amdgcn_workitem_id_y();
  out[y * w + x] = in[x * w + y] + (float)(x + y);
}
__attribute__((amdgpu_kernel)) void atomics(int* counter, const int* in, int n) {
  int i = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
  if (i < n && in[i] > 0) __atomic_fetch_add(counter, in[i], __ATOMIC_RELAXED);
}
__attribute__((amdgpu_kernel)) void long_math(const long* a, long* out, int n) {
  int i = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
  if (i < n) out[i] = a[i] * 3 + (a[i] >> 7);
}
