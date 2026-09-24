// What a kernel gets for the idioms it writes out by hand: a clamp, a rotate,
// a bitfield insert, a float taken apart into mantissa and exponent, the dot
// products, and two floats packed into halves. Built for gfx942 by build.sh.
typedef unsigned u32;
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef __fp16 hp2 __attribute__((ext_vector_type(2)));

// an int held between two bounds, which is the median of three
__attribute__((amdgpu_kernel)) void clampi(const int* a, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  int x = a[t];
  out[t] = x < -100 ? -100 : (x > 100 ? 100 : x);
}
// a rotate, written as two shifts
__attribute__((amdgpu_kernel)) void rotl(const u32* a, const u32* s, u32* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  u32 x = a[t], k = s[t] & 31;
  out[t] = (x << k) | (x >> ((32 - k) & 31));
}
// 32 bits out of two registers side by side, the first above the second: a
// rotate is the case where both are the same, and this is where they are not
__attribute__((amdgpu_kernel)) void funnel(const u32* hi, const u32* lo, const u32* s, u32* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (u32)((((unsigned long)hi[t] << 32) | lo[t]) >> (s[t] & 31));
}
// some bits from one value and the rest from another
__attribute__((amdgpu_kernel)) void insert(const u32* a, const u32* b, u32* out, u32 m, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (a[t] & m) | (b[t] & ~m);
}
// a float as a mantissa and a power of two, and its fractional part
__attribute__((amdgpu_kernel)) void parts(const float* a, float* fr, int* ex, float* mant, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  int e;
  mant[t] = __builtin_frexpf(a[t], &e);
  ex[t] = e;
  fr[t] = a[t] - __builtin_floorf(a[t]);
}
// four signed bytes times four, and two halves times two, each added to a start
__attribute__((amdgpu_kernel)) void dots(const u32* a, const u32* b, int* o4, float* o2, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  o4[t] = __builtin_amdgcn_sdot4((int)a[t], (int)b[t], 7, 0);
  h2 x = __builtin_bit_cast(h2, a[t]), y = __builtin_bit_cast(h2, b[t]);
  o2[t] = __builtin_amdgcn_fdot2(x, y, 1.0f, 0);
}
// two floats narrowed to halves, rounded toward zero, in one register
__attribute__((amdgpu_kernel)) void pkrtz(const float* a, const float* b, hp2* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_amdgcn_cvt_pkrtz(a[t], b[t]);
}
