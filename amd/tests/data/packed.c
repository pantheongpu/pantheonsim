// Two floats in a register pair, both computed at once, which is what the
// compiler does with a vector of two or four. Built for gfx942 by build.sh.
typedef float v2 __attribute__((ext_vector_type(2)));

__attribute__((amdgpu_kernel)) void pk_mul(const v2* a, const v2* b, v2* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = a[t] * b[t];
}
__attribute__((amdgpu_kernel)) void pk_add(const v2* a, const v2* b, v2* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = a[t] + b[t];
}
// one multiply-add over both, with each source also feeding the other half
__attribute__((amdgpu_kernel)) void pk_mix(const v2* a, const v2* b, const v2* c, v2* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = a[t] * b[t] + c[t];
}
// the same with constants, which one value serves both halves of
__attribute__((amdgpu_kernel)) void pk_scale(const v2* a, v2* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = a[t] * 2.0f + 1.0f;
}
