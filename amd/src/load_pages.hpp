// The host memory a code object is read into: a library's code objects inflate to tens of megabytes each, and every
// 4 KB page of a fresh buffer is a fault of its own when it is first written. Asked for in 2 MB pages (transparent huge
// pages, where the kernel gives them), the same bytes cost a 512th of the faults. With the other changes of
// pantheonsim #565, the 62 code objects of a small rocBLAS program (amd/tests/rocblas/blas) loaded in 0.44 s instead of
// 0.75 s, and the process took 190,000 page faults instead of 350,000. It is advice only: what the buffer holds is the
// same either way, and a kernel without transparent huge pages ignores it.
//
// VGPU_LOAD_PLAIN=1 reads code objects as before #565 -- no advice, a bundle inflated through its buffer and copied
// out, a linked object's image zeroed before its sections are copied in -- which the tests compare against.
#pragma once

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace vgpu::amd {

inline bool plain_load() {
  const char* v = std::getenv("VGPU_LOAD_PLAIN");
  return v && v[0] == '1';
}

// Advises the whole 2 MB pages inside [p, p + n) to be huge pages, before anything is written to them.
inline void prefer_huge_pages(const void* p, size_t n) {
#ifdef MADV_HUGEPAGE
  constexpr uintptr_t kHuge = uintptr_t{2} << 20;
  if (n < 2 * kHuge || plain_load()) return;
  const uintptr_t from = (reinterpret_cast<uintptr_t>(p) + kHuge - 1) & ~(kHuge - 1);
  const uintptr_t to = (reinterpret_cast<uintptr_t>(p) + n) & ~(kHuge - 1);
  if (to > from) ::madvise(reinterpret_cast<void*>(from), to - from, MADV_HUGEPAGE);   // (refused: nothing is lost)
#else
  (void)p, (void)n;
#endif
}

}  // namespace vgpu::amd
