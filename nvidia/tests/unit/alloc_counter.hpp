// Counts the heap a test's process allocates through operator new, so a test can say how much a call held at its peak.
//
// Include it from ONE translation unit of a test program: it replaces the global operator new and delete, which the
// program's shared libraries (libvgpucublas, libvgpucublaslt) resolve to as well, so their std::vector temporaries
// are counted too. Sizes are what malloc really handed out (malloc_usable_size), the same at delete, so the live
// total returns to where it was when everything is freed.
//
//   alloc_counter::reset_peak();          // peak := live now
//   ... the call ...
//   size_t extra = alloc_counter::peak() - base;    // base = alloc_counter::live() before the call
#pragma once
#include <malloc.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace alloc_counter {
inline std::atomic<size_t> g_live{0}, g_peak{0};
inline size_t live() { return g_live.load(); }
inline size_t peak() { return g_peak.load(); }
inline void reset_peak() { g_peak.store(g_live.load()); }
inline void* note(void* p) {
  if (!p) return p;
  const size_t now = g_live.fetch_add(malloc_usable_size(p)) + malloc_usable_size(p);
  size_t pk = g_peak.load();
  while (now > pk && !g_peak.compare_exchange_weak(pk, now)) {}
  return p;
}
// The operator new below is malloc, so the free is the matching one; GCC 13 cannot see through the replacement.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
inline void forget(void* p) {
  if (!p) return;
  g_live.fetch_sub(malloc_usable_size(p));
  std::free(p);
}
#pragma GCC diagnostic pop
inline void* aligned(size_t n, size_t al) {
  void* p = nullptr;
  if (al < sizeof(void*)) al = sizeof(void*);
  if (posix_memalign(&p, al, n ? n : 1) != 0) return nullptr;
  return p;
}
}  // namespace alloc_counter

void* operator new(size_t n) {
  void* p = alloc_counter::note(std::malloc(n ? n : 1));
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](size_t n) { return operator new(n); }
void* operator new(size_t n, const std::nothrow_t&) noexcept { return alloc_counter::note(std::malloc(n ? n : 1)); }
void* operator new[](size_t n, const std::nothrow_t&) noexcept { return alloc_counter::note(std::malloc(n ? n : 1)); }
void* operator new(size_t n, std::align_val_t al) {
  void* p = alloc_counter::note(alloc_counter::aligned(n, static_cast<size_t>(al)));
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](size_t n, std::align_val_t al) { return operator new(n, al); }
void operator delete(void* p) noexcept { alloc_counter::forget(p); }
void operator delete[](void* p) noexcept { alloc_counter::forget(p); }
void operator delete(void* p, size_t) noexcept { alloc_counter::forget(p); }
void operator delete[](void* p, size_t) noexcept { alloc_counter::forget(p); }
void operator delete(void* p, std::align_val_t) noexcept { alloc_counter::forget(p); }
void operator delete[](void* p, std::align_val_t) noexcept { alloc_counter::forget(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { alloc_counter::forget(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { alloc_counter::forget(p); }
