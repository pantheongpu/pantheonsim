// A lane reading another lane's register: the cross-lane form, which a wave
// adds itself up with. Built for gfx942 by build.sh.
//
// The first kernel is the sequence AMD's own reduction is written as: shift
// the row by one, two, four and eight, adding as it goes, then carry the last
// lane of each row into the rows above it. What comes out of it is the
// running total up to each lane, and the last lane holds the whole wave's.

__attribute__((amdgpu_kernel)) void wave_sum(const int* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  int v = in[t];
  v += __builtin_amdgcn_update_dpp(0, v, 0x111, 0xf, 0xf, 1);   // row_shr:1
  v += __builtin_amdgcn_update_dpp(0, v, 0x112, 0xf, 0xf, 1);   // row_shr:2
  v += __builtin_amdgcn_update_dpp(0, v, 0x114, 0xf, 0xe, 1);   // row_shr:4, banks 1 to 3
  v += __builtin_amdgcn_update_dpp(0, v, 0x118, 0xf, 0xc, 1);   // row_shr:8, banks 2 and 3
  v += __builtin_amdgcn_update_dpp(0, v, 0x142, 0xa, 0xf, 1);   // row_bcast:15, rows 1 and 3
  v += __builtin_amdgcn_update_dpp(0, v, 0x143, 0xc, 0xf, 1);   // row_bcast:31, rows 2 and 3
  out[t] = v;
}
// four lanes all taking the first of their four
__attribute__((amdgpu_kernel)) void quad_first(const int* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_amdgcn_update_dpp(0, in[t], 0x00, 0xf, 0xf, 1);
}
// the row read backwards
__attribute__((amdgpu_kernel)) void mirrored(const int* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_amdgcn_update_dpp(0, in[t], 0x140, 0xf, 0xf, 1);
}
// a lane with no lane to read, where the instruction says to leave it alone:
// it keeps what it had, which is what the first argument stands for
__attribute__((amdgpu_kernel)) void kept(const int* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_amdgcn_update_dpp(-1, in[t], 0x111, 0xf, 0xf, 0);
}
// across the whole wave rather than within a row: the one form of this that
// nothing here pins down, so it is decoded and refused rather than guessed
__attribute__((amdgpu_kernel)) void across(const int* in, int* out, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __builtin_amdgcn_update_dpp(0, in[t], 0x130, 0xf, 0xf, 1);
}
