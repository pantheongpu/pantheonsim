// Work-groups that wait on one another, for the lockstep launch's host-thread rule (test_amd_lockstep_threads): a grid of fewer than 16
// work-groups takes the host threads when its kernel waits on loads only, and keeps ONE thread when it takes a lock with an atomic.
// One wave in a group: lane 0 does the work.
typedef unsigned u32;

// The last group stores the flag; the others spin on it with loads, then record what they saw.   (loads only)
__attribute__((amdgpu_kernel)) void flag_wait(u32* flag, u32* out, u32 groups) {
  if (__builtin_amdgcn_workitem_id_x() != 0) return;
  const u32 b = __builtin_amdgcn_workgroup_id_x();
  if (b == groups - 1) {
    __atomic_store_n(flag, 77u, __ATOMIC_RELAXED);
    return;
  }
  u32 v;
  while ((v = __atomic_load_n(flag, __ATOMIC_RELAXED)) == 0) {
  }
  out[b] = v;
}

// Every group takes the lock with an exchange, counts itself, and lets go.   (an atomic)
__attribute__((amdgpu_kernel)) void lock_chain(u32* lock, u32* count, u32 groups) {
  if (__builtin_amdgcn_workitem_id_x() != 0) return;
  while (__atomic_exchange_n(lock, 1u, __ATOMIC_RELAXED)) {
  }
  const u32 c = __atomic_load_n(count, __ATOMIC_RELAXED);
  __atomic_store_n(count, c + 1, __ATOMIC_RELAXED);
  __atomic_store_n(lock, 0u, __ATOMIC_RELEASE);
}

// The last group stores the flag; the others wait for it with loads, then take the lock (word 2).   (both)
__attribute__((amdgpu_kernel)) void both(u32* flag, u32* count, u32 groups) {
  if (__builtin_amdgcn_workitem_id_x() != 0) return;
  const u32 b = __builtin_amdgcn_workgroup_id_x();
  if (b == groups - 1) {
    __atomic_store_n(flag, 1u, __ATOMIC_RELAXED);
  } else {
    while (__atomic_load_n(flag, __ATOMIC_RELAXED) == 0) {
    }
  }
  while (__atomic_exchange_n(flag + 2, 1u, __ATOMIC_RELAXED)) {
  }
  const u32 c = __atomic_load_n(count, __ATOMIC_RELAXED);
  __atomic_store_n(count, c + 1, __ATOMIC_RELAXED);
  __atomic_store_n(flag + 2, 0u, __ATOMIC_RELEASE);
}
