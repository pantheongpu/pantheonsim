#include "vgpu/memory.hpp"

#include <atomic>
#include <cstring>

#include "vgpu/error.hpp"

namespace vgpu {
namespace {

// A kernel's scalar access, as an atomic on the backing bytes. The chunk is an
// unsigned-char array, which C++20 lets hold objects of other trivial types.
template <typename T>
uint64_t relaxed_load(const uint8_t* p) {
  return std::atomic_ref<T>(*reinterpret_cast<T*>(const_cast<uint8_t*>(p)))
      .load(std::memory_order_relaxed);
}
template <typename T>
void relaxed_store(uint8_t* p, uint64_t v) {
  std::atomic_ref<T>(*reinterpret_cast<T*>(p)).store(static_cast<T>(v), std::memory_order_relaxed);
}

// A scalar in place, as an atomic of its own width -- or byte by byte when the
// chunk's storage is not aligned for that width, so no path copies the bytes a
// concurrent store is writing.
uint64_t load_at(const uint8_t* p, uint32_t size) {
  if (reinterpret_cast<uintptr_t>(p) % size == 0) {
    switch (size) {
      case 1: return relaxed_load<uint8_t>(p);
      case 2: return relaxed_load<uint16_t>(p);
      case 4: return relaxed_load<uint32_t>(p);
      default: return relaxed_load<uint64_t>(p);
    }
  }
  uint64_t v = 0;
  for (uint32_t i = 0; i < size; ++i) v |= relaxed_load<uint8_t>(p + i) << (8 * i);
  return v;
}
void store_at(uint8_t* p, uint32_t size, uint64_t v) {
  if (reinterpret_cast<uintptr_t>(p) % size == 0) {
    switch (size) {
      case 1: relaxed_store<uint8_t>(p, v); return;
      case 2: relaxed_store<uint16_t>(p, v); return;
      case 4: relaxed_store<uint32_t>(p, v); return;
      default: relaxed_store<uint64_t>(p, v); return;
    }
  }
  for (uint32_t i = 0; i < size; ++i) relaxed_store<uint8_t>(p + i, (v >> (8 * i)) & 0xff);
}

// Formats an address as 0x-prefixed hex for diagnostics.
struct Hex {
  uint64_t v;
};
std::ostream& operator<<(std::ostream& os, Hex h) {
  auto f = os.flags();
  os << "0x" << std::hex << h.v;
  os.flags(f);
  return os;
}

}  // namespace

uint64_t MemoryManager::alloc(uint64_t size) {
  if (size == 0) throw Error::make(Err::InvalidValue, "cudaMalloc-style allocation of 0 bytes is invalid");
  if (used_ + size > capacity_ || used_ + size < used_)
    throw Error::make(Err::OutOfMemory, "device out of memory: requested ", size, " bytes, ", used_,
                      " of ", capacity_, " bytes already in use");
  uint64_t base = next_va_;
  uint64_t padded = (size + kAllocAlign - 1) / kAllocAlign * kAllocAlign;
  next_va_ += padded;
  high_water_va_ = next_va_;
  used_ += size;
  Allocation a;
  a.size = size;
  a.chunk_count = static_cast<size_t>((size + kChunkSize - 1) / kChunkSize);
  a.chunks = std::make_unique<std::atomic<uint8_t*>[]>(a.chunk_count);
  live_.emplace(base, std::move(a));
  notify_usage();
  return base;
}

void MemoryManager::free(uint64_t ptr) {
  auto it = live_.find(ptr);
  if (it != live_.end()) {
    used_ -= it->second.size;
    // Quarantine the freed range so use-after-free stays diagnosable, evicting
    // the oldest entry once the bound is reached.
    freed_[ptr] = FreedRecord{it->second.size, freed_seq_};
    freed_order_[freed_seq_++] = ptr;
    while (freed_order_.size() > kQuarantineEntries) {
      auto oldest = freed_order_.begin();
      auto stale = freed_.find(oldest->second);
      // Only drop it if this record is the one that entry refers to.
      if (stale != freed_.end() && stale->second.seq == oldest->first) freed_.erase(stale);
      freed_order_.erase(oldest);
    }
    live_.erase(it);
    notify_usage();
    return;
  }
  if (freed_.count(ptr))
    throw Error::make(Err::DoubleFree, "double free of device pointer ", Hex{ptr}, " (allocation of ",
                      freed_.at(ptr).size, " bytes was already freed)");
  // Interior pointer into a live allocation?
  auto up = live_.upper_bound(ptr);
  if (up != live_.begin()) {
    auto prev = std::prev(up);
    if (ptr < prev->first + prev->second.size)
      throw Error::make(Err::InvalidFree, "free of interior device pointer ", Hex{ptr},
                        ": allocation starts at ", Hex{prev->first}, " (", prev->second.size,
                        " bytes); pass the base pointer");
  }
  if (ptr >= va_base_ && ptr < high_water_va_)
    throw Error::make(Err::InvalidPointer, "free of device pointer ", Hex{ptr},
                      " that is not a live allocation; it is inside the retired address range, so "
                      "it was most likely freed earlier (beyond the ", kQuarantineEntries,
                      "-entry double-free quarantine)");
  throw Error::make(Err::InvalidPointer, "free of unknown device pointer ", Hex{ptr},
                    " (never returned by an allocation)");
}

const MemoryManager::Allocation& MemoryManager::resolve(uint64_t addr, uint64_t len, const char* op,
                                                        uint64_t* base_out) const {
  auto up = live_.upper_bound(addr);
  if (up != live_.begin()) {
    auto prev = std::prev(up);
    uint64_t base = prev->first;
    const Allocation& a = prev->second;
    if (addr < base + a.size) {
      // Computed as a remaining-length rather than an end-address: `addr + len`
      // wraps for a length near UINT64_MAX, and a wrapped sum compares *below*
      // the end, so the check passed exactly the accesses it exists to stop.
      const uint64_t remaining = base + a.size - addr;
      if (len > remaining)
        throw Error::make(Err::OutOfBounds, op, " of ", len, " bytes at ", Hex{addr},
                          " runs past the end of the ", a.size, "-byte allocation at ", Hex{base},
                          " (last valid byte: ", Hex{base + a.size - 1}, ")");
      if (base_out) *base_out = base;
      return a;
    }
    // The VA range up to the alignment-padded end belongs to this allocation:
    // an access there is an overrun, which deserves a better diagnosis than
    // "unknown pointer".
    uint64_t padded = (a.size + kAllocAlign - 1) / kAllocAlign * kAllocAlign;
    if (addr < base + padded)
      throw Error::make(Err::OutOfBounds, op, " at ", Hex{addr}, " is ", addr - (base + a.size),
                        " bytes past the end of the ", a.size, "-byte allocation at ", Hex{base});
  }
  // Freed allocation?
  auto fup = freed_.upper_bound(addr);
  if (fup != freed_.begin()) {
    auto prev = std::prev(fup);
    if (addr < prev->first + prev->second.size)
      throw Error::make(Err::UseAfterFree, op, " at ", Hex{addr}, " touches freed allocation ",
                        Hex{prev->first}, " (", prev->second.size, " bytes); device memory was freed");
  }
  if (addr >= va_base_ && addr < high_water_va_)
    throw Error::make(Err::UseAfterFree, op, " at ", Hex{addr},
                      " is inside the retired address range: the allocation it belonged to was "
                      "freed (older than the ", kQuarantineEntries,
                      "-entry quarantine, so its size is no longer recorded)");
  throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                    ": address is not inside any device allocation",
                    addr < kDeviceVaBase   ? " (looks like a host pointer, not a device pointer)"
                    : !owns(addr)      ? " (belongs to a different device)"
                                       : "");
}

MemoryManager::Allocation& MemoryManager::resolve_mut(uint64_t addr, uint64_t len, const char* op,
                                                      uint64_t* base_out) {
  return const_cast<Allocation&>(resolve(addr, len, op, base_out));
}

uint8_t* MemoryManager::materialize(Allocation& a, uint64_t chunk_idx) {
  // First touch materializes the chunk. Two threads can race here; the loser
  // gives its copy back and uses the winner's, so the pointer a reader sees is
  // always the one that stays. The backing hands chunks out zeroed -- untouched
  // device memory reads as zero -- from the heap or, past its RAM limit, from a
  // file on disk.
  uint8_t* fresh = backing::acquire();
  uint8_t* expected = nullptr;
  if (a.chunks[chunk_idx].compare_exchange_strong(expected, fresh, std::memory_order_acq_rel))
    return fresh;
  backing::release(fresh);
  return expected;
}

void MemoryManager::fill(uint64_t dst, const uint8_t* pattern, uint32_t pattern_len, uint64_t len) {
  if (len == 0) return;
  uint64_t base = 0;
  Allocation& a = resolve_mut(dst, len, "device memory fill", &base);
  bool all_zero = true;
  for (uint32_t i = 0; i < pattern_len; ++i)
    if (pattern[i] != 0) all_zero = false;
  uint64_t off = dst - base;
  uint64_t done = 0;  // bytes filled so far; fixes the pattern's phase
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
    // Zeroing a chunk that was never touched is a no-op that would otherwise
    // cost 64 KiB of host RAM to express.
    if (chunk || !all_zero) {
      if (!chunk) chunk = materialize(a, chunk_idx);
      if (pattern_len == 1) {
        std::memset(chunk + chunk_off, pattern[0], static_cast<size_t>(n));
      } else {
        uint8_t* p = chunk + chunk_off;
        for (uint64_t i = 0; i < n; ++i) p[i] = pattern[(done + i) % pattern_len];
      }
    }
    off += n;
    done += n;
    len -= n;
  }
}

uint64_t MemoryManager::resident_bytes() const {
  uint64_t chunks = 0;
  for (const auto& [base, a] : live_) {
    (void)base;
    if (!a.chunks) continue;
    for (size_t i = 0; i < a.chunk_count; ++i)
      if (a.chunks[i].load(std::memory_order_relaxed)) ++chunks;
  }
  return chunks * kChunkSize;
}

void MemoryManager::write(uint64_t dst, const void* src, uint64_t len) {
  if (len == 0) return;
  if (host_maps_) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    if (const HostMap* m = find_host_map_locked(dst, len)) {
      std::memcpy(m->host + (dst - m->base), src, len);
      return;
    }
  }
  // A null host buffer would be dereferenced by the memcpy below and take the
  // process down with a signal, losing the diagnosis. Saying which argument was
  // null is the whole point of running on a simulator.
  if (!src)
    throw Error::make(Err::InvalidPointer,
                      "device memory write from a NULL host pointer (", len, " bytes to ",
                      Hex{dst}, ")");
  uint64_t base = 0;
  Allocation& a = resolve_mut(dst, len, "device memory write", &base);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint64_t off = dst - base;
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
    if (!chunk) chunk = materialize(a, chunk_idx);
    std::memcpy(chunk + chunk_off, s, n);
    s += n;
    off += n;
    len -= n;
  }
}

const MemoryManager::HostMap* MemoryManager::find_host_map_locked(uint64_t addr,
                                                                   uint64_t len) const {
  if (!host_maps_) return nullptr;
  for (const HostMap& m : host_maps_->maps)
    if (addr >= m.base && addr + len <= m.base + m.len) return &m;
  return nullptr;
}

void MemoryManager::map_host(uint64_t addr, void* host, uint64_t len) {
  if (!host_maps_) host_maps_ = std::make_unique<HostMaps>();
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  host_maps_->maps.push_back(HostMap{addr, len, static_cast<uint8_t*>(host)});
}

void MemoryManager::unmap_host(uint64_t addr) {
  if (!host_maps_) return;
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  auto& v = host_maps_->maps;
  for (size_t i = 0; i < v.size(); ++i)
    if (v[i].base == addr) {
      v.erase(v.begin() + static_cast<long>(i));
      break;
    }
}

bool MemoryManager::is_host_mapped(uint64_t addr) const {
  if (!host_maps_) return false;
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  return find_host_map_locked(addr, 1) != nullptr;
}

void MemoryManager::read(uint64_t src, void* dst, uint64_t len) const {
  if (len == 0) return;
  // Managed memory is the caller's own buffer; there is no chunk table to walk.
  if (host_maps_) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    if (const HostMap* m = find_host_map_locked(src, len)) {
      std::memcpy(dst, m->host + (src - m->base), len);
      return;
    }
  }
  if (!dst)
    throw Error::make(Err::InvalidPointer,
                      "device memory read into a NULL host pointer (", len, " bytes from ",
                      Hex{src}, ")");
  uint64_t base = 0;
  const Allocation& a = resolve(src, len, "device memory read", &base);
  uint8_t* d = static_cast<uint8_t*>(dst);
  uint64_t off = src - base;
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    const uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
    if (!chunk)
      std::memset(d, 0, n);  // untouched device memory reads as zero (documented)
    else
      std::memcpy(d, chunk + chunk_off, n);
    d += n;
    off += n;
    len -= n;
  }
}

bool MemoryManager::find_allocation(uint64_t addr, uint64_t* base, uint64_t* size) const {
  auto up = live_.upper_bound(addr);
  if (up == live_.begin()) return false;
  auto prev = std::prev(up);
  if (addr >= prev->first + prev->second.size) return false;
  if (base) *base = prev->first;
  if (size) *size = prev->second.size;
  return true;
}

uint64_t MemoryManager::load_scalar(uint64_t addr, uint32_t size) const {
  if (size != 1 && size != 2 && size != 4 && size != 8)
    throw Error::make(Err::Internal, "load_scalar: bad size ", size);
  if (addr % size != 0)
    throw Error::make(Err::MisalignedAccess, "load of ", size, " bytes at ", Hex{addr},
                      " is not naturally aligned (real GPUs fault on this)");
  const uint8_t* p = nullptr;
  switch (scalar_location(addr, size, /*create=*/false, &p)) {
    case ScalarAt::Chunk:
      return load_at(p, size);
    case ScalarAt::Untouched:
      // Zero, answered here. Falling back to read() instead was a race: a
      // first store on another thread could materialize the chunk between
      // this check and read()'s copy, which then memcpy'd bytes an atomic
      // store was writing. ThreadSanitizer caught it on main.
      return 0;
    case ScalarAt::HostMap:
      break;
  }
  uint64_t v = 0;
  read(addr, &v, size);  // little-endian host assumption, documented in ARCHITECTURE.md
  return v;
}

void MemoryManager::store_scalar(uint64_t addr, uint32_t size, uint64_t value) {
  if (size != 1 && size != 2 && size != 4 && size != 8)
    throw Error::make(Err::Internal, "store_scalar: bad size ", size);
  if (addr % size != 0)
    throw Error::make(Err::MisalignedAccess, "store of ", size, " bytes at ", Hex{addr},
                      " is not naturally aligned (real GPUs fault on this)");
  const uint8_t* p = nullptr;
  if (scalar_location(addr, size, /*create=*/true, &p) == ScalarAt::Chunk) {
    store_at(const_cast<uint8_t*>(p), size, value);
    return;
  }
  write(addr, &value, size);  // a managed host mapping, behind its own lock
}

// Where a kernel's scalar load or store lives: bytes in a chunk (materialized
// first when this is a store), memory nothing has written (only ever answered
// for a load), or managed memory -- the caller's own buffer, which takes the
// general path behind its lock.
//
// Why this exists. Blocks run on several host threads, and a kernel is allowed
// to share memory between blocks -- CUB's decoupled look-back scan, under
// Thrust's sort and CUB's radix sort, publishes each block's prefix for later
// blocks to spin on. These accesses used to be a plain memcpy, which made that
// a C++ data race: undefined behaviour, reported by ThreadSanitizer the first
// time CI ran it. It worked in release builds because an aligned 8-byte copy
// is one instruction on x86, which nothing promises -- a torn read of a tile
// status is a wrong prefix or a spin that never ends. A GPU's accesses are
// atomic at this granularity; so are these now, relaxed, which on x86 is the
// same single instruction and costs nothing. Ordering is the fences' job (see
// the interpreter), not the load's.
//
// A naturally aligned scalar cannot straddle a chunk -- kChunkSize is a
// multiple of 8 -- so it is always one location in one chunk.
MemoryManager::ScalarAt MemoryManager::scalar_location(uint64_t addr, uint32_t size, bool create,
                                                       const uint8_t** where) const {
  if (host_maps_) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    if (find_host_map_locked(addr, size)) return ScalarAt::HostMap;
  }
  uint64_t base = 0;
  auto& a = const_cast<Allocation&>(resolve(addr, size, create ? "device memory write"
                                                                 : "device memory read", &base));
  const uint64_t off = addr - base;
  const uint64_t chunk_idx = off / kChunkSize;
  uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
  if (!chunk) {
    if (!create) return ScalarAt::Untouched;
    chunk = const_cast<MemoryManager*>(this)->materialize(a, chunk_idx);
  }
  *where = chunk + off % kChunkSize;
  return ScalarAt::Chunk;
}

}  // namespace vgpu
