#include "vgpu/memory.hpp"

#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

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

// A `size`-byte scalar whose every byte is `b`, and whether `v` is one.
uint64_t repeated(uint8_t b, uint32_t size) {
  const uint64_t all = 0x0101010101010101ull * b;
  return size == 8 ? all : all & ((uint64_t{1} << (8 * size)) - 1);
}
bool all_bytes(const uint8_t* p, uint64_t n, uint8_t b) {
  for (uint64_t i = 0; i < n; ++i)
    if (p[i] != b) return false;
  return true;
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
  ExclusiveGuard table_guard(table_lock_.get());
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
  ExclusiveGuard table_guard(table_lock_.get());
  // An allocation that was exported for another process lives in a file now,
  // not in chunks. Freeing it takes the mapping and the file with it, which is
  // what the exporting process owns.
  if (const auto sh = shared_.find(ptr); sh != shared_.end() && sh->second.owner) {
    used_ -= sh->second.size;
    freed_[ptr] = FreedRecord{sh->second.size, freed_seq_};
    freed_order_[freed_seq_++] = ptr;
    drop_shared(sh);
    notify_usage();
    return;
  }
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

const MemoryManager::Allocation* MemoryManager::resolve_mapped(uint64_t addr, uint64_t len,
                                                               const char* op, uint64_t* base_out,
                                                               bool writing) const {
  if (maps_.empty() && reserved_.empty()) return nullptr;
  auto mu = maps_.upper_bound(addr);
  if (mu != maps_.begin()) {
    auto prev = std::prev(mu);
    const uint64_t va = prev->first;
    const Mapping& m = prev->second;
    if (addr - va < m.size) {
      const uint64_t remaining = va + m.size - addr;
      if (len > remaining)
        throw Error::make(Err::OutOfBounds, op, " of ", len, " bytes at ", Hex{addr},
                          " runs past the end of the ", m.size, "-byte mapping at ", Hex{va},
                          " (mapped address space ends at ", Hex{va + m.size - 1}, ")");
      // Mapped is not yet usable: CUDA grants access separately, and a device
      // faults on memory it was never given access to.
      if (!m.readable && !m.writable)
        throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                          " is mapped but no device has been given access to it "
                          "(cuMemSetAccess)");
      if (writing && !m.writable)
        throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                          " is through a read-only mapping (cuMemSetAccess granted read access "
                          "only)");
      const auto h = handles_.find(m.handle);
      if (h == handles_.end())
        throw Error::make(Err::Internal, op, " at ", Hex{addr}, ": mapping has no handle");
      // The handle's own offset 0 sits this far below the mapping's address, so
      // the caller's `addr - base` lands at the right offset in its chunks.
      if (base_out) *base_out = va - m.offset;
      return &h->second.mem;
    }
  }
  auto ru = reserved_.upper_bound(addr);
  if (ru != reserved_.begin()) {
    auto prev = std::prev(ru);
    if (addr - prev->first < prev->second.size)
      throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                        " is inside reserved address space with nothing mapped there (cuMemMap)");
  }
  return nullptr;
}

const MemoryManager::Allocation& MemoryManager::resolve(uint64_t addr, uint64_t len, const char* op,
                                                        uint64_t* base_out, bool writing) const {
  SharedGuard table_guard(table_lock_.get());
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
  if (const Allocation* mapped = resolve_mapped(addr, len, op, base_out, writing)) return *mapped;
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
  // A managed or pinned buffer that has since been freed lives at a host
  // address, outside every device window; without this a kernel touching it
  // was told it had a stray host pointer rather than a freed one.
  if (host_maps_ && !is_device_va(addr)) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    for (const HostMap& r : host_maps_->retired)
      if (addr >= r.base && addr - r.base < r.len)
        throw Error::make(Err::UseAfterFree, op, " at ", Hex{addr}, " touches host buffer ",
                          Hex{r.base}, " (", r.len, " bytes), which has been freed or unregistered");
  }
  throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                    ": address is not inside any device allocation",
                    !is_device_va(addr)  ? " (looks like a host pointer, not a device pointer)"
                    : !owns(addr)      ? " (belongs to a different device)"
                                       : "");
}

MemoryManager::Allocation& MemoryManager::resolve_mut(uint64_t addr, uint64_t len, const char* op,
                                                      uint64_t* base_out) {
  return const_cast<Allocation&>(resolve(addr, len, op, base_out, /*writing=*/true));
}

uint8_t* MemoryManager::materialize(Allocation& a, uint64_t chunk_idx) {
  // A chunk becomes real on the first write that needs its bytes: from nothing,
  // which reads as zero, or from a uniform chunk, which reads as its byte. Two
  // threads can race here; the loser gives its copy back and uses the winner's,
  // so the pointer a reader sees is always the one that stays -- and a reader
  // that saw the tag read the same byte the real chunk starts with. The backing
  // hands chunks out zeroed, from the heap or, past its RAM limit, from disk.
  uint8_t* expected = a.chunks[chunk_idx].load(std::memory_order_acquire);
  while (!expected || is_uniform(expected)) {
    uint8_t* fresh = backing::acquire();
    if (expected && uniform_byte(expected)) std::memset(fresh, uniform_byte(expected), kChunkSize);
    if (a.chunks[chunk_idx].compare_exchange_strong(expected, fresh, std::memory_order_acq_rel))
      return fresh;
    backing::release(fresh);
  }
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
    if (pattern_len == 1 && chunk_off == 0 && n == kChunkSize) {
      // A whole chunk of one byte is that byte: no RAM, no disk. Whatever it
      // held goes back to the backing, and zero is simply untouched memory.
      // Host calls do not run beside kernels -- launches are synchronous -- so
      // nothing is reading the chunk being replaced.
      uint8_t* old = a.chunks[chunk_idx].exchange(pattern[0] ? uniform_chunk(pattern[0]) : nullptr,
                                                  std::memory_order_acq_rel);
      if (old && !is_uniform(old)) backing::release(old);
    } else if (pattern_len == 1 && is_uniform(chunk) && uniform_byte(chunk) == pattern[0]) {
      // Already that byte throughout.
    } else if (chunk || !all_zero) {
      // Zeroing a chunk that was never touched is a no-op that would otherwise
      // cost 64 KiB of host RAM to express.
      if (!chunk || is_uniform(chunk)) chunk = materialize(a, chunk_idx);
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
  SharedGuard table_guard(table_lock_.get());
  uint64_t chunks = 0;
  for (const auto& [base, a] : live_) {
    (void)base;
    if (!a.chunks) continue;
    for (size_t i = 0; i < a.chunk_count; ++i) {
      const uint8_t* c = a.chunks[i].load(std::memory_order_relaxed);
      if (c && !is_uniform(c)) ++chunks;
    }
  }
  return chunks * kChunkSize;
}

void MemoryManager::write(uint64_t dst, const void* src, uint64_t len) {
  if (len == 0) return;
  if (host_maps_ && host_maps_->may_contain(dst)) {
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
    // Writing a uniform chunk's own byte changes nothing, and must not turn
    // 64 KiB of nothing into a chunk: a stress test rewrites its fill pattern.
    if (!(is_uniform(chunk) && all_bytes(s, n, uniform_byte(chunk)))) {
      if (!chunk || is_uniform(chunk)) chunk = materialize(a, chunk_idx);
      std::memcpy(chunk + chunk_off, s, n);
    }
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
  // The allocator hands freed host addresses out again, so an old record of a
  // free at this address now describes someone else's live buffer.
  std::erase_if(host_maps_->retired, [&](const HostMap& r) {
    return r.base < addr + len && addr < r.base + r.len;
  });
  if (addr < host_maps_->lo.load(std::memory_order_relaxed))
    host_maps_->lo.store(addr, std::memory_order_relaxed);
  if (addr + len > host_maps_->hi.load(std::memory_order_relaxed))
    host_maps_->hi.store(addr + len, std::memory_order_relaxed);
}

void MemoryManager::unmap_host(uint64_t addr) {
  if (!host_maps_) return;
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  auto& v = host_maps_->maps;
  for (size_t i = 0; i < v.size(); ++i)
    if (v[i].base == addr) {
      auto& retired = host_maps_->retired;
      if (retired.size() >= kRetiredHostMaps) retired.erase(retired.begin());
      retired.push_back(HostMap{v[i].base, v[i].len, nullptr});
      v.erase(v.begin() + static_cast<long>(i));
      break;
    }
}

void MemoryManager::free_all() {
  ExclusiveGuard table_guard(table_lock_.get());
  // A reset takes mapped memory, reservations and handles with it, as it takes
  // allocations: nothing survives it on a real device either.
  maps_.clear();
  reserved_.clear();
  for (auto& [id, h] : handles_) used_ -= h.size;
  handles_.clear();

  // Exported allocations first: they are not in live_, and a reset takes them.
  while (true) {
    auto it = std::find_if(shared_.begin(), shared_.end(),
                           [](const auto& kv) { return kv.second.owner; });
    if (it == shared_.end()) break;
    used_ -= it->second.size;
    drop_shared(it);
  }
  std::vector<uint64_t> bases;
  bases.reserve(live_.size());
  for (const auto& [base, a] : live_) bases.push_back(base);
  for (uint64_t base : bases) free(base);
}

// ---- virtual memory management ------------------------------------------------
//
// Validation is CUDA's: sizes and addresses are multiples of the granularity,
// a mapping goes inside a reservation, and nothing overlaps.

namespace {
uint64_t round_up(uint64_t v, uint64_t to) { return (v + to - 1) / to * to; }
}  // namespace

uint64_t MemoryManager::reserve(uint64_t size, uint64_t alignment) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (size == 0 || size % kVmmGranularity)
    throw Error::make(Err::InvalidValue, "reserving ", size,
                      " bytes of address space: the size must be a non-zero multiple of the ",
                      kVmmGranularity, "-byte granularity");
  if (alignment == 0) alignment = kVmmGranularity;
  if (alignment % kVmmGranularity || (alignment & (alignment - 1)))
    throw Error::make(Err::InvalidValue, "address space alignment ", alignment,
                      " is not a power of two multiple of the ", kVmmGranularity,
                      "-byte granularity");
  const uint64_t base = round_up(next_va_, alignment);
  // Address space is this device's 1 TiB window; a reservation past it would
  // hand out addresses another device owns.
  if (base + size > va_base_ + kDeviceVaStride || base + size < base)
    throw Error::make(Err::OutOfMemory, "reserving ", size,
                      " bytes of address space would run past this device's window");
  next_va_ = base + size;
  high_water_va_ = next_va_;
  reserved_.emplace(base, Reservation{size});
  return base;
}

void MemoryManager::address_free(uint64_t va, uint64_t size) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = reserved_.find(va);
  if (it == reserved_.end() || it->second.size != size)
    throw Error::make(Err::InvalidValue, "freeing address space at ", Hex{va}, " of ", size,
                      " bytes: no reservation of exactly that address and size");
  auto m = maps_.lower_bound(va);
  if (m != maps_.end() && m->first < va + size)
    throw Error::make(Err::InvalidValue, "freeing address space at ", Hex{va},
                      " while ", Hex{m->first}, " is still mapped (cuMemUnmap first)");
  reserved_.erase(it);
}

uint64_t MemoryManager::create_handle(uint64_t size) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (size == 0 || size % kVmmGranularity)
    throw Error::make(Err::InvalidValue, "creating ", size,
                      " bytes of device memory: the size must be a non-zero multiple of the ",
                      kVmmGranularity, "-byte granularity");
  if (used_ + size > capacity_ || used_ + size < used_)
    throw Error::make(Err::OutOfMemory, "device out of memory: requested ", size, " bytes, ", used_,
                      " of ", capacity_, " bytes already in use");
  used_ += size;
  Handle h;
  h.size = size;
  h.mem.size = size;
  h.mem.chunk_count = static_cast<size_t>((size + kChunkSize - 1) / kChunkSize);
  h.mem.chunks = std::make_unique<std::atomic<uint8_t*>[]>(h.mem.chunk_count);
  const uint64_t id = next_handle_++;
  handles_.emplace(id, std::move(h));
  notify_usage();
  return id;
}

uint64_t MemoryManager::handle_size(uint64_t handle) const {
  SharedGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "no such memory handle: ", handle);
  return it->second.size;
}

void MemoryManager::retain_handle(uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "no such memory handle: ", handle);
  ++it->second.refs;
}

uint64_t MemoryManager::retain_handle_at(uint64_t va) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto mu = maps_.upper_bound(va);
  if (mu == maps_.begin()) return 0;
  auto prev = std::prev(mu);
  if (va - prev->first >= prev->second.size) return 0;
  const uint64_t handle = prev->second.handle;
  retain_handle(handle);
  return handle;
}

void MemoryManager::release_handle(uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "releasing memory handle ", handle,
                      ", which does not exist (already released?)");
  if (it->second.refs == 0)
    throw Error::make(Err::InvalidValue, "releasing memory handle ", handle,
                      ", which holds no references");
  --it->second.refs;
  collect_handle(handle);
}

void MemoryManager::collect_handle(uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end() || it->second.refs || it->second.mapped) return;
  used_ -= it->second.size;
  handles_.erase(it);
  notify_usage();
}

void MemoryManager::map(uint64_t va, uint64_t size, uint64_t offset, uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (size == 0 || size % kVmmGranularity || va % kVmmGranularity)
    throw Error::make(Err::InvalidValue, "mapping ", size, " bytes at ", Hex{va},
                      ": the address and size must be multiples of the ", kVmmGranularity,
                      "-byte granularity");
  // cuMemMap documents that the offset into the handle must be zero.
  if (offset != 0)
    throw Error::make(Err::InvalidValue, "mapping at an offset of ", offset,
                      " bytes into a handle: cuMemMap takes an offset of 0");
  auto h = handles_.find(handle);
  if (h == handles_.end())
    throw Error::make(Err::InvalidValue, "mapping memory handle ", handle, ", which does not exist");
  if (size > h->second.size - offset)
    throw Error::make(Err::InvalidValue, "mapping ", size, " bytes of memory handle ", handle,
                      ", which holds ", h->second.size, " bytes");
  // Inside one reservation, and it must be.
  auto ru = reserved_.upper_bound(va);
  bool inside = false;
  if (ru != reserved_.begin()) {
    auto prev = std::prev(ru);
    inside = va - prev->first < prev->second.size && va + size <= prev->first + prev->second.size;
  }
  if (!inside)
    throw Error::make(Err::InvalidValue, "mapping ", size, " bytes at ", Hex{va},
                      ": the range is not inside one reservation (cuMemAddressReserve)");
  // Nothing else may be mapped there.
  auto after = maps_.lower_bound(va);
  if (after != maps_.end() && after->first < va + size)
    throw Error::make(Err::InvalidValue, "mapping ", size, " bytes at ", Hex{va},
                      ": ", Hex{after->first}, " is already mapped");
  if (after != maps_.begin()) {
    auto before = std::prev(after);
    if (va - before->first < before->second.size)
      throw Error::make(Err::InvalidValue, "mapping ", size, " bytes at ", Hex{va},
                        ": it overlaps the mapping at ", Hex{before->first});
  }
  maps_.emplace(va, Mapping{size, offset, handle, false, false});
  ++h->second.mapped;
}

void MemoryManager::unmap(uint64_t va, uint64_t size) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = maps_.find(va);
  if (it == maps_.end() || it->second.size != size)
    throw Error::make(Err::InvalidValue, "unmapping ", size, " bytes at ", Hex{va},
                      ": no mapping of exactly that address and size");
  const uint64_t handle = it->second.handle;
  maps_.erase(it);
  auto h = handles_.find(handle);
  if (h != handles_.end() && h->second.mapped) --h->second.mapped;
  collect_handle(handle);
}

void MemoryManager::set_access(uint64_t va, uint64_t size, bool readable, bool writable) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (size == 0 || size % kVmmGranularity || va % kVmmGranularity)
    throw Error::make(Err::InvalidValue, "granting access to ", size, " bytes at ", Hex{va},
                      ": the address and size must be multiples of the ", kVmmGranularity,
                      "-byte granularity");
  // Every byte of the range has to be mapped: granting access to address space
  // with nothing behind it would leave a pointer that faults on first use.
  uint64_t at = va;
  while (at < va + size) {
    auto it = maps_.find(at);
    if (it == maps_.end() || it->second.size > va + size - at)
      throw Error::make(Err::InvalidValue, "granting access to ", size, " bytes at ", Hex{va},
                        ": ", Hex{at}, " is not the start of a mapping inside that range");
    it->second.readable = readable;
    it->second.writable = writable;
    at += it->second.size;
  }
}

bool MemoryManager::access_at(uint64_t va, bool* readable, bool* writable) const {
  SharedGuard table_guard(table_lock_.get());
  auto mu = maps_.upper_bound(va);
  if (mu == maps_.begin()) return false;
  auto prev = std::prev(mu);
  if (va - prev->first >= prev->second.size) return false;
  if (readable) *readable = prev->second.readable;
  if (writable) *writable = prev->second.writable;
  return true;
}

// ---- memory another process can map -------------------------------------------

uint64_t MemoryManager::share(uint64_t ptr, const std::string& path) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (const auto it = shared_.find(ptr); it != shared_.end()) return it->second.size;
  const auto live = live_.find(ptr);
  if (live == live_.end())
    throw Error::make(Err::InvalidPointer, "sharing device pointer ", Hex{ptr},
                      " that is not the base of a live allocation");
  const uint64_t size = live->second.size;
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0)
    throw Error::make(Err::Internal, "sharing device memory: could not create ", path, ": ",
                      std::strerror(errno));
  // Sparse: the file is as large as the allocation, and costs only the pages
  // that are written -- the same property the sparse chunks have.
  void* host = nullptr;
  if (::ftruncate(fd, static_cast<off_t>(size)) == 0) {
    host = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (host == MAP_FAILED) host = nullptr;
  }
  const int err = errno;
  ::close(fd);
  if (!host) {
    ::unlink(path.c_str());
    throw Error::make(Err::Internal, "sharing device memory: could not map ", path, ": ",
                      std::strerror(err));
  }
  // Move what is there now, then let the allocation's chunks go: from here on
  // the file is the memory, at the same device address.
  read_chunks(live->second, 0, static_cast<uint8_t*>(host), size);
  live_.erase(live);
  map_host(ptr, host, size);
  shared_[ptr] = SharedRegion{ptr, size, host, path, /*owner=*/true};
  return size;
}

bool MemoryManager::is_shared(uint64_t ptr) const {
  SharedGuard table_guard(table_lock_.get());
  return shared_.count(ptr) != 0;
}

uint64_t MemoryManager::adopt(const std::string& path, uint64_t size) {
  ExclusiveGuard table_guard(table_lock_.get());
  const int fd = ::open(path.c_str(), O_RDWR);
  if (fd < 0)
    throw Error::make(Err::InvalidValue, "opening shared device memory ", path, ": ",
                      std::strerror(errno));
  void* host = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  const int err = errno;
  ::close(fd);
  if (host == MAP_FAILED)
    throw Error::make(Err::Internal, "mapping shared device memory ", path, ": ",
                      std::strerror(err));
  // An address of this device's own, never reused, so the imported buffer can
  // never be confused with a local allocation -- present or freed.
  const uint64_t va = next_va_;
  next_va_ += (size + kAllocAlign - 1) / kAllocAlign * kAllocAlign;
  high_water_va_ = next_va_;
  map_host(va, host, size);
  shared_[va] = SharedRegion{va, size, host, path, /*owner=*/false};
  return va;
}

void MemoryManager::abandon(uint64_t va) {
  ExclusiveGuard table_guard(table_lock_.get());
  const auto it = shared_.find(va);
  if (it == shared_.end() || it->second.owner)
    throw Error::make(Err::InvalidPointer, "closing shared device memory at ", Hex{va},
                      ": this process did not map it from another");
  drop_shared(it);
}

void MemoryManager::drop_shared(std::map<uint64_t, SharedRegion>::iterator it) {
  unmap_host(it->second.va);
  ::munmap(it->second.host, it->second.size);
  if (it->second.owner) ::unlink(it->second.path.c_str());
  shared_.erase(it);
}

bool MemoryManager::is_host_mapped(uint64_t addr) const {
  if (!host_maps_) return false;
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  return find_host_map_locked(addr, 1) != nullptr;
}

void MemoryManager::read(uint64_t src, void* dst, uint64_t len) const {
  if (len == 0) return;
  // Managed memory is the caller's own buffer; there is no chunk table to walk.
  if (host_maps_ && host_maps_->may_contain(src)) {
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
  read_chunks(a, src - base, d, len);
  if (any_stuck()) access_fault_->on_read(src - va_base_, d, len);
  // Only copies read device memory in bulk -- kernels read it a scalar at a
  // time -- so this is where a fault armed on copies is taken.
  if (access_fault_ && access_fault_->copy_pending &&
      __atomic_load_n(access_fault_->copy_pending, __ATOMIC_RELAXED))
    access_fault_->on_copy(src, d, len);
}

void MemoryManager::read_chunks(const Allocation& a, uint64_t off, uint8_t* d, uint64_t len) const {
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    const uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
    if (!chunk)
      std::memset(d, 0, n);  // untouched device memory reads as zero (documented)
    else if (is_uniform(chunk))
      std::memset(d, uniform_byte(chunk), n);
    else
      std::memcpy(d, chunk + chunk_off, n);
    d += n;
    off += n;
    len -= n;
  }
}

bool MemoryManager::find_allocation(uint64_t addr, uint64_t* base, uint64_t* size) const {
  SharedGuard table_guard(table_lock_.get());
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
  uint64_t v = 0;
  switch (scalar_location(addr, size, /*store=*/nullptr, &p)) {
    case ScalarAt::Chunk:
      v = load_at(p, size);
      break;
    case ScalarAt::Uniform:
      v = repeated(uniform_byte(p), size);
      break;
    case ScalarAt::Untouched:
      // Zero, answered here. Falling back to read() instead was a race: a
      // first store on another thread could materialize the chunk between
      // this check and read()'s copy, which then memcpy'd bytes an atomic
      // store was writing. ThreadSanitizer caught it on main.
      v = 0;
      break;
    case ScalarAt::Unchanged:  // only answered for a store
    case ScalarAt::HostMap:
      // Host memory, mapped: no device ECC covers it, so no fault is taken.
      read(addr, &v, size);  // little-endian host assumption, documented in ARCHITECTURE.md
      return v;
  }
  // What the cells hold, stuck bits included (little-endian, as above), and
  // then whatever fault is armed for the load.
  if (any_stuck()) access_fault_->on_read(addr - va_base_, reinterpret_cast<uint8_t*>(&v), size);
  if (access_fault_ && access_fault_->load_pending &&
      __atomic_load_n(access_fault_->load_pending, __ATOMIC_RELAXED))
    return access_fault_->on_load(addr, size, v);
  return v;
}

void MemoryManager::store_scalar(uint64_t addr, uint32_t size, uint64_t value) {
  if (size != 1 && size != 2 && size != 4 && size != 8)
    throw Error::make(Err::Internal, "store_scalar: bad size ", size);
  if (addr % size != 0)
    throw Error::make(Err::MisalignedAccess, "store of ", size, " bytes at ", Hex{addr},
                      " is not naturally aligned (real GPUs fault on this)");
  // A fault armed on stores changes what reaches device memory, so every later
  // read finds it. Mapped host memory has no device ECC and takes none.
  if (access_fault_ && access_fault_->store_pending &&
      __atomic_load_n(access_fault_->store_pending, __ATOMIC_RELAXED) && !is_host_mapped(addr))
    value = access_fault_->on_store(addr, size, value);
  const uint8_t* p = nullptr;
  switch (scalar_location(addr, size, &value, &p)) {
    case ScalarAt::Chunk:
      store_at(const_cast<uint8_t*>(p), size, value);
      return;
    case ScalarAt::Unchanged:
      return;
    default:
      break;
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
MemoryManager::ScalarAt MemoryManager::scalar_location(uint64_t addr, uint32_t size,
                                                       const uint64_t* store,
                                                       const uint8_t** where) const {
  const bool create = store != nullptr;
  if (host_maps_ && host_maps_->may_contain(addr)) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    if (find_host_map_locked(addr, size)) return ScalarAt::HostMap;
  }
  uint64_t base = 0;
  auto& a = const_cast<Allocation&>(resolve(addr, size, create ? "device memory write"
                                                                 : "device memory read", &base));
  const uint64_t off = addr - base;
  const uint64_t chunk_idx = off / kChunkSize;
  uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
  if (is_uniform(chunk)) {
    if (!create) { *where = chunk; return ScalarAt::Uniform; }
    if (*store == repeated(uniform_byte(chunk), size)) return ScalarAt::Unchanged;
  }
  if (!chunk || is_uniform(chunk)) {
    if (!chunk && !create) return ScalarAt::Untouched;
    chunk = const_cast<MemoryManager*>(this)->materialize(a, chunk_idx);
  }
  *where = chunk + off % kChunkSize;
  return ScalarAt::Chunk;
}

}  // namespace vgpu
