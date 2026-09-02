#include "vgpu/memory.hpp"

#include <cstring>

#include "vgpu/error.hpp"

namespace vgpu {
namespace {

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
      // Compare against the bytes remaining rather than addr + len: a length
      // near UINT64_MAX makes that sum wrap, and the check then passes for an
      // access that runs off the end. `addr` is inside the allocation here, so
      // `remaining` cannot underflow.
      const uint64_t remaining = a.size - (addr - base);
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
  // frees its copy and uses the winner's, so the pointer a reader sees is
  // always the one that stays. make_unique value-initializes, so the chunk
  // arrives zeroed -- untouched device memory reads as zero.
  auto fresh = std::make_unique<uint8_t[]>(kChunkSize);
  uint8_t* expected = nullptr;
  if (a.chunks[chunk_idx].compare_exchange_strong(expected, fresh.get(),
                                                  std::memory_order_acq_rel))
    return fresh.release();
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

void MemoryManager::write(uint64_t dst, const void* src, uint64_t len) {
  if (len == 0) return;
  // The host pointer comes straight from the application. CUDA reports a null
  // one as an invalid argument; dereferencing it instead turns a recoverable
  // application bug into a dead process.
  if (!src)
    throw Error::make(Err::InvalidValue, "device memory write of ", len,
                      " bytes from a NULL host pointer");
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

void MemoryManager::read(uint64_t src, void* dst, uint64_t len) const {
  if (len == 0) return;
  if (!dst)
    throw Error::make(Err::InvalidValue, "device memory read of ", len,
                      " bytes into a NULL host pointer");
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
  write(addr, &value, size);
}

}  // namespace vgpu
