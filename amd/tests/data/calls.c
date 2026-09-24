__attribute__((noinline)) int helper(int x, int y) { return x * y + (x ^ y); }
__attribute__((amdgpu_kernel)) void caller(const int* a, const int* b, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = helper(a[t], b[t]) + helper(b[t], 3);
}
