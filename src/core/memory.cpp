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
  live_.emplace(base, Allocation{size, {}});
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
      if (addr + len > base + a.size)
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

void MemoryManager::write(uint64_t dst, const void* src, uint64_t len) {
  if (len == 0) return;
  uint64_t base = 0;
  Allocation& a = resolve_mut(dst, len, "device memory write", &base);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint64_t off = dst - base;
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    auto& chunk = a.chunks[chunk_idx];
    if (!chunk) {
      chunk = std::make_unique<uint8_t[]>(kChunkSize);
      std::memset(chunk.get(), 0, kChunkSize);
    }
    std::memcpy(chunk.get() + chunk_off, s, n);
    s += n;
    off += n;
    len -= n;
  }
}

void MemoryManager::read(uint64_t src, void* dst, uint64_t len) const {
  if (len == 0) return;
  uint64_t base = 0;
  const Allocation& a = resolve(src, len, "device memory read", &base);
  uint8_t* d = static_cast<uint8_t*>(dst);
  uint64_t off = src - base;
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    auto it = a.chunks.find(chunk_idx);
    if (it == a.chunks.end())
      std::memset(d, 0, n);  // untouched device memory reads as zero (documented)
    else
      std::memcpy(d, it->second.get() + chunk_off, n);
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
