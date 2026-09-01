// Virtual device memory.
//
// Design:
//  - Device pointers live in a fake VA range starting at kDeviceVaBase so they
//    can never be confused with host pointers.
//  - Backing is sparse: 64 KiB chunks materialize on first write. A virtual
//    H200 can advertise 141 GB of VRAM on a 16 GB laptop; only touched pages
//    cost host RAM. Untouched device memory reads as zero (documented
//    VirtualGPU behavior; real GPUs leave it undefined).
//  - Virtual addresses are handed out monotonically and never reused, so
//    use-after-free is detectable for the lifetime of the process.
//  - Every access is bounds-checked and produces a rich diagnostic on failure
//    (these diagnostics are a product feature for CI, not just debug aids).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace vgpu {

inline constexpr uint64_t kDeviceVaBase = 0x7fff'0000'0000ull;
inline constexpr uint64_t kAllocAlign = 256;  // matches CUDA's documented minimum alignment
inline constexpr uint64_t kChunkSize = 64 * 1024;

class MemoryManager {
 public:
  explicit MemoryManager(uint64_t capacity_bytes) : capacity_(capacity_bytes) {}

  // Allocates `size` bytes of virtual device memory. size == 0 is invalid.
  uint64_t alloc(uint64_t size);

  // Frees an allocation. `ptr` must be the exact base returned by alloc().
  void free(uint64_t ptr);

  // Bulk copies (the H2D/D2H/D2D building blocks).
  void write(uint64_t dst, const void* src, uint64_t len);
  void read(uint64_t src, void* dst, uint64_t len) const;

  // Scalar access for the interpreter. size must be 1/2/4/8 and the address
  // naturally aligned. Values are zero-extended into the returned u64.
  uint64_t load_scalar(uint64_t addr, uint32_t size) const;
  void store_scalar(uint64_t addr, uint32_t size, uint64_t value);

  uint64_t used() const { return used_; }
  uint64_t capacity() const { return capacity_; }
  size_t live_allocations() const { return live_.size(); }

 private:
  struct Allocation {
    uint64_t size = 0;
    // chunk index -> chunk bytes; absent chunks read as zero.
    std::map<uint64_t, std::unique_ptr<uint8_t[]>> chunks;
  };
  struct FreedRecord {
    uint64_t size = 0;
  };

  // Maps addr to (allocation base, allocation); throws with diagnostics.
  const Allocation& resolve(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out) const;
  Allocation& resolve_mut(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out);

  uint64_t capacity_;
  uint64_t used_ = 0;
  uint64_t next_va_ = kDeviceVaBase;
  std::map<uint64_t, Allocation> live_;        // base -> allocation
  std::map<uint64_t, FreedRecord> freed_;      // base -> record (for UAF/double-free reporting)
};

}  // namespace vgpu
