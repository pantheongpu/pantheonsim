// The atomics beyond an add, and the ones that give back what was there.
typedef unsigned long u64;
__attribute__((address_space(3))) extern int shared_slots[64];

// each lane's read-modify-write, with the value that was there returned
__attribute__((amdgpu_kernel)) void returning(int* p, int* out, const int* in, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  out[t] = __atomic_fetch_add(&p[0], in[t], __ATOMIC_RELAXED);
  out[t] += __atomic_fetch_sub(&p[1], in[t], __ATOMIC_RELAXED);
  out[t] += __atomic_fetch_xor(&p[2], in[t], __ATOMIC_RELAXED);
  out[t] += __atomic_exchange_n(&p[3], in[t], __ATOMIC_RELAXED);
}
// a float, and a 64-bit value
__attribute__((amdgpu_kernel)) void wide(float* f, u64* q, const int* in, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  __atomic_fetch_add(f, (float)in[t], __ATOMIC_RELAXED);
  __atomic_fetch_add(q, (u64)in[t], __ATOMIC_RELAXED);
}
// the same in LDS, where a work-group shares them
__attribute__((amdgpu_kernel)) void shared(int* out, const int* in, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  shared_slots[t & 7] = 0;
  shared_slots[(t & 7) + 8] = -1000000;
  __builtin_amdgcn_s_barrier();
  __atomic_fetch_xor((__attribute__((address_space(3))) int*)&shared_slots[t & 7], in[t], __ATOMIC_RELAXED);
  __atomic_fetch_max((__attribute__((address_space(3))) int*)&shared_slots[(t & 7) + 8], in[t], __ATOMIC_RELAXED);
  __builtin_amdgcn_s_barrier();
  out[t] = shared_slots[t & 7] + shared_slots[(t & 7) + 8];
}
