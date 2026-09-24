// Values packed into parts of a register, and a private array of them. The
// compiler reaches a named byte or half of a register for all of this, and
// writes one of them where it can, rather than moving whole registers about.
// Built for gfx942 by build.sh.
typedef unsigned u32;
typedef unsigned char u8;
typedef unsigned short u16;
typedef _Float16 h;

// two floats narrowed to halves and put in one register
__attribute__((amdgpu_kernel)) void pack_halves(const float* a, const float* b, u32* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  h lo = (h)a[t], hi = (h)b[t];
  u32 l = 0, g = 0;
  __builtin_memcpy(&l, &lo, 2);
  __builtin_memcpy(&g, &hi, 2);
  out[t] = (l & 0xffff) | (g << 16);
}
// a half of the result written into the high half of a register, with one
// of the sources the same for every work-item
__attribute__((amdgpu_kernel)) void pair_of_shorts(const u16* a, u32 s, u32* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  u16 v = (u16)(a[t] + s);
  out[t] = (u32)v | ((u32)(u16)((u32)v * (u32)(u16)s) << 16);
}
// a shift whose amount is the same for every work-item, over a half
__attribute__((amdgpu_kernel)) void shifted(const u16* a, u32 s, u32* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (u32)(u16)(s - a[t]) | ((u32)(u16)((u16)s << (a[t] & 7)) << 16);
}
// a private array of bytes, which is too big for registers and goes to the
// work-item's own memory a byte at a time
__attribute__((amdgpu_kernel)) void spill_bytes(const int* idx, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  signed char v[64];
  for (int i = 0; i < 64; ++i) v[i] = (signed char)(idx[i % n] + i);
  int s = 0;
  for (int i = 0; i < 24; ++i) s += v[idx[(t + i) % n] & 63];
  out[t] = s;
}
