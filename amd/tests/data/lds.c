// The rest of what a kernel does with LDS: values wider and narrower than a
// word, a float atomic, and the swizzle, where lanes trade values through the
// LDS unit without touching LDS itself. Built for gfx942 by build.sh.
typedef unsigned char u8;
typedef unsigned short u16;
// Named for itself: the fixtures are compiled together by the disassembly
// test, and v4 is already a vector of unsigned ints in builtins.c.
typedef float lds_quad __attribute__((ext_vector_type(4)));

__attribute__((address_space(3))) extern double lds_doubles[64];
__attribute__((address_space(3))) extern lds_quad lds_quads[64];
__attribute__((address_space(3))) extern u8 lds_bytes[64];
__attribute__((address_space(3))) extern u16 lds_shorts[64];
__attribute__((address_space(3))) extern signed char lds_signed[64];
__attribute__((address_space(3))) extern float lds_floats[8];

// two words at a time
__attribute__((amdgpu_kernel)) void doubles_in_lds(const double* in, double* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  lds_doubles[t] = in[t] * 2.0;
  __builtin_amdgcn_s_barrier();
  out[t] = lds_doubles[(t + 1) & 63] + lds_doubles[(t + 5) & 63];
}
// four words at a time
__attribute__((amdgpu_kernel)) void quads_in_lds(const lds_quad* in, lds_quad* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  lds_quads[t] = in[t];
  __builtin_amdgcn_s_barrier();
  out[t] = lds_quads[(t + 1) & 63];
}
// a byte, a half, and a byte whose sign matters
__attribute__((amdgpu_kernel)) void narrow_in_lds(const u8* a, const u16* b, const signed char* c, int* out,
                                                   int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  lds_bytes[t] = a[t];
  lds_shorts[t] = b[t];
  lds_signed[t] = c[t];
  __builtin_amdgcn_s_barrier();
  out[t] = lds_bytes[(t + 1) & 63] + lds_shorts[(t + 2) & 63] + lds_signed[(t + 3) & 63];
}
// eight slots, each the float sum of the eight lanes that share it
__attribute__((amdgpu_kernel)) void float_sums_in_lds(const float* in, float* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  lds_floats[t & 7] = 0.0f;
  __builtin_amdgcn_s_barrier();
  __atomic_fetch_add(&lds_floats[t & 7], in[t], __ATOMIC_RELAXED);
  __builtin_amdgcn_s_barrier();
  out[t] = lds_floats[t & 7];
}
// Each lane adds the lane whose number differs from its own in one bit, for
// each of the five bits: afterwards every lane holds the sum of its group of
// 32. That is what the swizzle's masks have to mean for it to come out.
__attribute__((amdgpu_kernel)) void butterfly(const int* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  int v = in[t];
  v += __builtin_amdgcn_ds_swizzle(v, (1 << 10) | 0x1f);
  v += __builtin_amdgcn_ds_swizzle(v, (2 << 10) | 0x1f);
  v += __builtin_amdgcn_ds_swizzle(v, (4 << 10) | 0x1f);
  v += __builtin_amdgcn_ds_swizzle(v, (8 << 10) | 0x1f);
  v += __builtin_amdgcn_ds_swizzle(v, (16 << 10) | 0x1f);
  out[t] = v;
}
// the other shapes a pattern takes: a group reversed, one lane of a group of
// eight handed to all of them, and the first of each four
__attribute__((amdgpu_kernel)) void swizzle_shapes(const int* in, int* reversed, int* broadcast, int* quad,
                                                    int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  reversed[t] = __builtin_amdgcn_ds_swizzle(in[t], (31 << 10) | 0x1f);
  broadcast[t] = __builtin_amdgcn_ds_swizzle(in[t], (5 << 5) | 0x18);
  quad[t] = __builtin_amdgcn_ds_swizzle(in[t], 0x8000);
}
// a lane reading another lane twice over, the second time from the register
// the first one wrote
__attribute__((amdgpu_kernel)) void permute_twice(const int* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  int v = __builtin_amdgcn_ds_bpermute(((t + 1) & 63) << 2, in[t]);
  v = __builtin_amdgcn_ds_bpermute(((t + 2) & 63) << 2, v);
  out[t] = v;
}
