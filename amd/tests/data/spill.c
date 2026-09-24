// More values than a kernel has registers for. The machine has a second bank
// of them, the accumulation registers, and the compiler moves what will not
// fit there and back rather than to memory. Built for gfx942 by build.sh.

// Forty of them, indexed at run time, in a function the compiler kept whole.
__attribute__((noinline)) int deep(const int* p, int n) {
  int local[40];
  for (int i = 0; i < 40; ++i) local[i] = p[i % n] + i;
  int s = 0;
  for (int i = 0; i < 40; ++i) s += local[(i * 7) % 40];
  return s;
}
__attribute__((amdgpu_kernel)) void user(const int* a, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = deep(a, n) + t;
}
