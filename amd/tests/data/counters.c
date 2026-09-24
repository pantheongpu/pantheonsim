// What a profiler counts, in a kernel with no branches: every wave issues each
// of its instructions exactly once, so each count is the listing's times the
// number of waves. Built for gfx942 by build.sh.
typedef _Float16 counted_half4 __attribute__((ext_vector_type(4)));
typedef float counted_float4 __attribute__((ext_vector_type(4)));
__attribute__((address_space(3))) extern float counted_lds[128];
__attribute__((amdgpu_kernel)) void counted_mix(const float* in, float* out, int* hits, float* g, int use_lds) {
  unsigned t = __builtin_amdgcn_workitem_id_x();
  unsigned i = __builtin_amdgcn_workgroup_id_x() * 128 + t;
  float v = in[i];                                    // a global load
  counted_lds[t] = v * 2.0f;                          // LDS
  __builtin_amdgcn_s_barrier();
  float w = counted_lds[(t + 1) & 127];
  __atomic_fetch_add(&hits[i], 1, __ATOMIC_RELAXED);  // a global atomic, an address per lane
  // a generic pointer, which the launch points at LDS or at device memory
  float* p = use_lds ? (float*)&counted_lds[t] : &g[i];
  *p = w;
  counted_half4 a = {(_Float16)v, 1, 2, 3}, b = {1, (_Float16)w, 1, 1};
  counted_float4 c = {w, 0, 0, 0};
  counted_float4 d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  out[i] = d[0] + d[1] + *p;
}
// A group that is not a multiple of 64 ends in a wave with fewer lanes.
__attribute__((amdgpu_kernel)) void counted_copy(const float* in, float* out) {
  unsigned i = __builtin_amdgcn_workgroup_id_x() * 128 + __builtin_amdgcn_workitem_id_x();
  out[i] = in[i];
}
