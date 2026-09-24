// The rest of what a kernel does with doubles: comparing them, rounding
// them, the smallest and largest of two, and the conversions to and from
// them. The arithmetic itself is in math.c; this is everything around it.
// Built for gfx942 by build.sh.
typedef unsigned u32;
__attribute__((amdgpu_kernel)) void cmp(const double* a, const double* b, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (a[t] < b[t]) + 2 * (a[t] >= b[t]) + 4 * (a[t] == b[t]) + 8 * (a[t] != b[t]);
}
__attribute__((amdgpu_kernel)) void pick(const double* a, const double* b, double* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  double x = a[t], y = b[t];
  out[t] = __builtin_fmin(x, y) + __builtin_fmax(x, y) + __builtin_fabs(x) + __builtin_floor(y) +
           __builtin_trunc(x) + __builtin_ceil(y) + __builtin_rint(x);
}
__attribute__((amdgpu_kernel)) void conv(const double* a, const float* f, const int* i, const u32* u,
                                         double* out, float* of, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (double)f[t] + (double)i[t] + (double)u[t];
  of[t] = (float)a[t];
}
__attribute__((amdgpu_kernel)) void roots(const double* a, double* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_sqrt(a[t] < 0 ? -a[t] : a[t]);
}
