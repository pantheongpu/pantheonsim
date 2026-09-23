// The rest of what a kernel does with numbers: doubles, packed half
// precision, the absolute value and min/max modifiers, the transcendentals,
// bit counting, and the atomics beyond an add. Built for gfx942 by build.sh.
typedef _Float16 half;
typedef half half2 __attribute__((ext_vector_type(2)));
// doubles
__attribute__((amdgpu_kernel)) void dbl_math(const double* a, const double* b, double* out, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  if (i < n) { double x = a[i], y = b[i]; out[i] = x * y + x / y + __builtin_fma(x, y, 1.0); }
}
// absolute value, min and max: the VOP3 modifiers and the clamp
__attribute__((amdgpu_kernel)) void mods(const float* a, const float* b, float* out, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  if (i < n) {
    float x = a[i], y = b[i];
    out[i] = __builtin_fabsf(x) + __builtin_fminf(x, y) + __builtin_fmaxf(-x, y);
  }
}
// half precision, packed two at a time
__attribute__((amdgpu_kernel)) void half_math(const half2* a, const half2* b, half2* out, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  if (i < n) out[i] = a[i] * b[i] + a[i];
}
// transcendentals
__attribute__((amdgpu_kernel)) void transcendental(const float* a, float* out, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  if (i < n) out[i] = __builtin_sqrtf(a[i]) + __builtin_exp2f(a[i]) + __builtin_log2f(a[i]);
}
// integer min/max, popcount, clz
__attribute__((amdgpu_kernel)) void bits(const unsigned* a, unsigned* out, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  if (i < n) {
    unsigned x = a[i];
    out[i] = __builtin_popcount(x) + __builtin_clz(x | 1) + (x > 100u ? x : 100u);
  }
}
// enough live values to spill to scratch
__attribute__((amdgpu_kernel)) void spill(const float* in, float* out, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  float v[40];
  for (int k = 0; k < 40; ++k) v[k] = in[(i + k) % n] * (float)k;
  float s = 0;
  for (int k = 0; k < 40; ++k) s += v[k] * v[(k * 7 + 3) % 40];
  out[i] = s;
}
// more atomics
__attribute__((amdgpu_kernel)) void atomics2(int* p, const int* in, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  if (i < n) {
    __atomic_fetch_or(p, in[i], __ATOMIC_RELAXED);
    __atomic_fetch_and(p + 1, in[i], __ATOMIC_RELAXED);
    int expected = 0;
    __atomic_compare_exchange_n(p + 2, &expected, in[i], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
}
