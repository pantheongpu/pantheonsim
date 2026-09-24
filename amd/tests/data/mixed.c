// Arithmetic that mixes widths, which the compiler does with an instruction
// that reads part of a register: a byte or a word of a source, taken with or
// without its sign, in a 32-bit operation. Built for gfx942 by build.sh.
typedef unsigned char u8;

// a word of each source, sign-extended
__attribute__((amdgpu_kernel)) void word_sext(const short* a, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  short s = (short)(a[t] - 1000);
  out[t] = (int)s + (int)(short)(a[t] << 1);
}
// a byte of each source, without a sign
__attribute__((amdgpu_kernel)) void byte_zext(const u8* a, const u8* b, unsigned* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (unsigned)(u8)(a[t] + b[t]) + (unsigned)(u8)(a[t] * 3);
}
// a word from one source and a byte from the other, both with their sign
__attribute__((amdgpu_kernel)) void mixed_widths(const signed char* a, const short* b, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (int)(signed char)(a[t] + 1) * (int)(short)(b[t] - 2);
}
