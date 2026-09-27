// How a kernel learns the shape of the grid it is part of. Built for gfx942
// by build.sh.
__attribute__((amdgpu_kernel)) void from_implicit(unsigned* out) {
  unsigned t = __builtin_amdgcn_workitem_id_x();
  __attribute__((address_space(4))) const unsigned* imp =
      (__attribute__((address_space(4))) const unsigned*)__builtin_amdgcn_implicitarg_ptr();
  __attribute__((address_space(4))) const unsigned short* imp16 =
      (__attribute__((address_space(4))) const unsigned short*)imp;
  out[t] = imp[0] * 1000 + imp16[6];   // block_count_x, then group_size_x
}
__attribute__((amdgpu_kernel)) void from_packet(unsigned* out) {
  unsigned t = __builtin_amdgcn_workitem_id_x();
  __attribute__((address_space(4))) const unsigned short* p =
      (__attribute__((address_space(4))) const unsigned short*)__builtin_amdgcn_dispatch_ptr();
  __attribute__((address_space(4))) const unsigned* p32 =
      (__attribute__((address_space(4))) const unsigned*)p;
  out[t] = p[2] * 1000 + p32[3];   // workgroup_size_x, then grid_size_x
}
// Where each work-item is in the grid, and the remainder the runtime says the
// grid leaves: out[i] = 1 + its number in its group + 1000 * remainder_x. A
// work-item the grid does not have writes nothing.
__attribute__((amdgpu_kernel)) void where(unsigned* out) {
  unsigned g = __builtin_amdgcn_workgroup_id_x(), t = __builtin_amdgcn_workitem_id_x();
  __attribute__((address_space(4))) const unsigned short* imp16 =
      (__attribute__((address_space(4))) const unsigned short*)__builtin_amdgcn_implicitarg_ptr();
  out[g * 64 + t] = 1 + t + 1000 * imp16[9];   // hidden_remainder_x, at byte 18
}
