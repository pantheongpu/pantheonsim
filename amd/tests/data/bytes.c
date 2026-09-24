// What a kernel does with values narrower than a register: bytes and shorts,
// signed and unsigned, loaded, computed on and stored back. Built for gfx942
// by build.sh.
typedef unsigned char u8;
typedef unsigned short u16;

// unsigned bytes: loaded zero-extended, stored back as bytes
__attribute__((amdgpu_kernel)) void bytes_u(const u8* in, u8* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (u8)(in[t] * 3 + 7);
}
// signed bytes: loaded sign-extended, and widened to an int
__attribute__((amdgpu_kernel)) void bytes_i(const signed char* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = in[t] - 5;
}
// unsigned shorts, kept as shorts throughout
__attribute__((amdgpu_kernel)) void shorts_u(const u16* a, const u16* b, u16* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (u16)(a[t] + b[t] * 2);
}
// signed shorts, widened to an int: the sign has to survive the narrowing
__attribute__((amdgpu_kernel)) void shorts_i(const short* a, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (short)(a[t] - 1000);
}
// a short widened where it is read, which is one instruction, and a byte the
// same way
__attribute__((amdgpu_kernel)) void widen_signed(const short* a, const signed char* b, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (int)a[t] + (int)b[t];
}
// a short that is widened where it is read, which is what the mask costs
__attribute__((amdgpu_kernel)) void widen(const u16* a, unsigned* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = (unsigned)(u16)(a[t] + 7u);
}
