// A kernel that reaches variables of its own -- what a __device__ global
// compiles to: the module's memory, and code that adds a constant to the
// program counter to find it. Built for gfx942 by build.sh.
__attribute__((visibility("protected"))) int counters[64];
__attribute__((visibility("protected"))) float scale = 3.5f;
__attribute__((amdgpu_kernel)) void use_global(const int* in, int* out, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  if (i < n) { counters[i & 63] = in[i]; out[i] = (int)((float)counters[i & 63] * scale); }
}
