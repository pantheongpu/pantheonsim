// The kernels the code-object tests read, built for gfx942 by build.sh.
// Written against clang's AMDGPU builtins rather than HIP, so it builds with
// no ROCm installed.

// out[i] = a[i] + b[i], one work-item per element.
__attribute__((amdgpu_kernel)) void vector_add(const float* a, const float* b, float* out, int n) {
  int i = __builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x();
  if (i < n) out[i] = a[i] + b[i];
}

// A reduction through LDS, so a kernel here reserves group segment, uses a
// barrier, and takes a scalar by value.
__attribute__((address_space(3))) extern float partial[256];   // LDS, shared by the work-group

__attribute__((amdgpu_kernel)) void reduce_sum(const float* in, float* out, int n, float scale) {
  unsigned t = __builtin_amdgcn_workitem_id_x();
  unsigned i = __builtin_amdgcn_workgroup_id_x() * 256 + t;
  partial[t] = i < (unsigned)n ? in[i] * scale : 0.0f;
  __builtin_amdgcn_s_barrier();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (t < stride) partial[t] += partial[t + stride];
    __builtin_amdgcn_s_barrier();
  }
  if (t == 0) out[__builtin_amdgcn_workgroup_id_x()] = partial[0];
}
