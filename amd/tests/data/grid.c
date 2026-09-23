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
