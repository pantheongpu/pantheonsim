// An atomic read-modify-write on memory the host maps.
//
// Device memory is the simulator's own: its atomics are serialized by the
// engines' locks, which is enough within the process. Memory the host maps --
// managed and pinned buffers, and buffers shared with another process through
// CUDA IPC -- can also be updated by something those locks do not see: host
// code using the CPU's atomics, or a kernel in another process (NVSHMEM's PEs
// add into one another's heaps). There an atomic has to be the CPU's own, a
// compare-and-swap loop on the bytes themselves, for no update to be lost.
#ifndef VGPU_EXEC_HOST_ATOMIC_HPP
#define VGPU_EXEC_HOST_ATOMIC_HPP

#include <cstdint>

namespace vgpu::exec {

// The engines' device-memory atomics (SASS and PTX) take the CPU's compare-and-swap on each word, with no lock, when the process
// runs its grids on at least this many host threads, and the striped locks below that. Measured (Ryzen 9 9950X, a warp's 32 atomics on 14
// cache lines): a lock pair per line costs about 1.8 ns, a locked instruction per word about 4 ns, so on one thread the locks are 1.25x
// faster; two threads are no contention yet (locks still faster); from four the stripe lines bounce between the threads and the
// swaps scale (0.77x -> 1.8x at 4 threads, 2.6x at 8) where the locks do not. The choice is the process's, made from the configured
// thread count (VGPU_THREADS, else the CPUs), not the launch's: two launches that overlap must not meet on one word under different
// roads.
inline constexpr unsigned kLockFreeAtomicsFromThreads = 4;

// Replaces the `size`-byte value at p (2, 4 or 8 bytes, naturally aligned)
// with compute(old), atomically; returns old.
template <class F>
uint64_t host_atomic_rmw(uint8_t* p, uint32_t size, F&& compute) {
  switch (size) {
    case 2: {
      auto* q = reinterpret_cast<uint16_t*>(p);
      uint16_t o = __atomic_load_n(q, __ATOMIC_SEQ_CST);
      for (;;) {
        const uint16_t nv = static_cast<uint16_t>(compute(uint64_t{o}));
        if (__atomic_compare_exchange_n(q, &o, nv, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) return o;
      }
    }
    case 4: {
      auto* q = reinterpret_cast<uint32_t*>(p);
      uint32_t o = __atomic_load_n(q, __ATOMIC_SEQ_CST);
      for (;;) {
        const uint32_t nv = static_cast<uint32_t>(compute(uint64_t{o}));
        if (__atomic_compare_exchange_n(q, &o, nv, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) return o;
      }
    }
    default: {
      auto* q = reinterpret_cast<uint64_t*>(p);
      uint64_t o = __atomic_load_n(q, __ATOMIC_SEQ_CST);
      for (;;) {
        const uint64_t nv = compute(o);
        if (__atomic_compare_exchange_n(q, &o, nv, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) return o;
      }
    }
  }
}

// A 2-byte atomic on the swap road: a compare-and-swap on the aligned 32-bit word that holds it, so it is atomic against the 4-byte
// atomics on that word (a plain 16-bit store beside a 4-byte swap would lose one of them). `compute(old16)` gives the new 16 bits.
// False, having done nothing, where Mem::atomic_rmw declines (the caller then takes its lock).
template <class Mem, class F>
bool atomic_rmw16(Mem& mem, uint64_t addr, F&& compute, uint64_t* old_out) {
  if (addr & 1) return false;
  const unsigned sh = static_cast<unsigned>(addr & 2) * 8;
  uint64_t found = 0;
  if (!mem.atomic_rmw(addr & ~uint64_t{3}, 4, [&](uint64_t w) -> uint64_t {
        const uint64_t nv = compute((w >> sh) & 0xffff);
        return (w & ~(uint64_t{0xffff} << sh)) | ((nv & 0xffff) << sh);
      }, &found, true))
    return false;
  *old_out = (found >> sh) & 0xffff;
  return true;
}

}  // namespace vgpu::exec

#endif  // VGPU_EXEC_HOST_ATOMIC_HPP
