// 64-bit integers, which a register pair holds and a carry threads together,
// and the conversions between a double and an integer. Built for gfx942 by
// build.sh.
typedef unsigned long u64;
typedef long i64;

// the arithmetic: a multiply, a shift, a division and a remainder
__attribute__((amdgpu_kernel)) void i64_math(const i64* a, const i64* b, i64* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  i64 x = a[t], y = b[t] | 1;
  out[t] = x * y + (x >> 3) - (x / y) + (x % y);
}
// the carry a 64-bit add and subtract thread through a register pair
__attribute__((amdgpu_kernel)) void u64_addsub(const u64* a, const u64* b, u64* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (a[t] + b[t]) ^ (a[t] - b[t]);
}
// counting the bits of one
__attribute__((amdgpu_kernel)) void u64_bits(const u64* a, u64* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  u64 x = a[t] | 1;
  out[t] = (u64)__builtin_popcountl(x) + (u64)__builtin_clzl(x) + (u64)__builtin_ctzl(x);
}
// comparing them, which is one instruction over a pair
__attribute__((amdgpu_kernel)) void u64_cmp(const u64* a, const u64* b, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (a[t] == b[t]) + 2 * (a[t] < b[t]);
}
// a double turned into an integer and back, and a float from a double
__attribute__((amdgpu_kernel)) void dbl_convert(const double* a, int* oi, float* of, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  oi[t] = (int)a[t];
  of[t] = (float)a[t] + (float)(unsigned)oi[t];
}
