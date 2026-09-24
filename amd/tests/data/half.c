// Half precision one value at a time, which is what a kernel gets for
// _Float16 where it does not pack two of them into a register. Built for
// gfx942 by build.sh.
typedef _Float16 h;

// the arithmetic, including the fused multiply-add the compiler finds
__attribute__((amdgpu_kernel)) void h_math(const h* a, const h* b, h* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  h x = a[t], y = b[t];
  out[t] = x * y + x - y;
}
// a plain multiply, with nothing for it to be folded into
__attribute__((amdgpu_kernel)) void h_scale(const h* a, const h* b, h* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = a[t] * b[t];
}
// the smaller and the larger of two, and a constant of its own
__attribute__((amdgpu_kernel)) void h_choose(const h* a, const h* b, h* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  h x = a[t], y = b[t];
  h m = x < y ? x : y, M = x > y ? x : y;
  out[t] = (x + y) * (m - M) + (h)0.5f;
}
// between a half and a float, in both directions
__attribute__((amdgpu_kernel)) void h_convert(const h* a, const float* g, float* f, h* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  f[t] = (float)a[t] * 2.0f;
  out[t] = (h)(g[t] + 1.0f);
}
// the comparisons, including the ones a NaN answers differently
__attribute__((amdgpu_kernel)) void h_cmp(const h* a, const h* b, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (a[t] < b[t]) + 2 * (a[t] >= b[t]) + 4 * (a[t] != b[t]) + 8 * (a[t] == b[t]) +
           16 * (a[t] > b[t] ? 1 : 0);
}
