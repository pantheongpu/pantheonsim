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

}  // namespace vgpu::exec

#endif  // VGPU_EXEC_HOST_ATOMIC_HPP
