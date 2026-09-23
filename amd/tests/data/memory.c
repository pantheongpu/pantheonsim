// Where a kernel's values live besides registers: a private array too big to
// keep in them (scratch), LDS with an atomic, a value read from another lane,
// and a pointer that may be either LDS or device memory (flat). Built for
// gfx942 by build.sh.
__attribute__((address_space(3))) extern float lds[256];
// a private array indexed at run time: it cannot live in registers
__attribute__((amdgpu_kernel)) void scratch(const int* idx, float* out, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  float v[32];
  for (int k = 0; k < 32; ++k) v[k] = (float)(k * i);
  float s = 0;
  for (int k = 0; k < 8; ++k) s += v[idx[(i + k) % n] & 31];
  out[i] = s;
}
// atomics on LDS
__attribute__((amdgpu_kernel)) void lds_atomic(const float* in, float* out, int n) {
  unsigned t = __builtin_amdgcn_workitem_id_x();
  __attribute__((address_space(3))) float* p = &lds[t & 7];
  *p = 0;
  __builtin_amdgcn_s_barrier();
  __atomic_fetch_add((__attribute__((address_space(3))) int*)p, (int)in[t], __ATOMIC_RELAXED);
  __builtin_amdgcn_s_barrier();
  out[t] = lds[t & 7];
}
// a shuffle across the wave
__attribute__((amdgpu_kernel)) void shuffle(const int* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  int v = in[t];
  out[t] = __builtin_amdgcn_ds_bpermute((t ^ 1) << 2, v) + __builtin_amdgcn_readfirstlane(v);
}
// a generic pointer: it may be global or LDS, so the compiler uses flat access
__attribute__((amdgpu_kernel)) void generic_ptr(float* g, int use_lds, float* out) {
  unsigned t = __builtin_amdgcn_workitem_id_x();
  float* p = use_lds ? (float*)&lds[t & 15] : &g[t];
  *p = (float)t;
  out[t] = *p * 2.0f;
}
